import asyncio
from concurrent.futures import ThreadPoolExecutor

import fakeredis
import httpx
import pytest
import redis.exceptions as rex
from fastapi.testclient import TestClient

from app import main
from app.budget import SpendTracker
from app.cache import ResponseCache, SingleFlight, cache_key, is_cacheable
from app.config import settings
from app.providers.openai import OpenAIProvider
from app.ratelimit import TokenBucketLimiter
from app.reliability import ReliableRouter, RetryPolicy

URL = "/v1/chat/completions"
H = {"Authorization": "Bearer gw-a"}
BODY = {"model": "gpt-4o-mini", "temperature": 0,
        "messages": [{"role": "user", "content": "What is your refund policy?"}]}


def completion(text="30 days."):
    return {"model": "gpt-4o-mini-2024-07-18",
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1000}}


# ---------- key + policy unit tests ----------

def test_key_ignores_field_order_and_irrelevant_fields():
    a = {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0}
    b = {"temperature": 0, "user": "alice", "stream": False, "metadata": {"a": 1},
         "messages": [{"content": "x", "role": "user"}], "model": "m"}
    assert cache_key("t", a) == cache_key("t", b)


@pytest.mark.parametrize("change", [
    {"model": "other"}, {"temperature": 0.5}, {"max_tokens": 50}, {"top_p": 0.9},
    {"messages": [{"role": "user", "content": "different"}]},
    {"response_format": {"type": "json_object"}},
])
def test_any_output_affecting_field_changes_the_key(change):
    assert cache_key("t", {**BODY, **change}) != cache_key("t", BODY)


def test_key_is_scoped_per_api_key_name():
    assert cache_key("team-a", BODY) != cache_key("team-b", BODY)


@pytest.mark.parametrize("payload,headers,expected", [
    ({"temperature": 0}, {}, True),
    ({"temperature": 0.0}, {}, True),
    ({}, {}, False),                                   # default temperature = non-deterministic
    ({"temperature": 0.7}, {}, False),
    ({}, {"x-cache-mode": "force"}, True),
    ({"temperature": 0}, {"x-cache-mode": "bypass"}, False),
    ({"temperature": 0, "stream": True}, {}, False),   # streams are never cached
    ({"stream": True}, {"x-cache-mode": "force"}, False),
])
def test_is_cacheable(payload, headers, expected):
    assert is_cacheable(payload, headers) is expected


async def test_response_cache_roundtrip_ttl_size_and_flush():
    r = fakeredis.FakeAsyncRedis()
    c = ResponseCache(r, ttl_s=60, max_bytes=2000)
    assert await c.set("cache:a:1", "openai", completion()) is True
    assert (await c.get("cache:a:1"))["provider"] == "openai"
    assert 0 < await r.ttl("cache:a:1") <= 60
    assert await c.set("cache:a:2", "openai", {"error": "x"}) is False          # not a completion
    assert await ResponseCache(r, 60, max_bytes=10).set("cache:a:3", "p", completion()) is False
    await c.set("cache:b:1", "openai", completion())
    assert await c.flush("a") == 1 and await c.get("cache:b:1") is not None
    assert await c.flush() == 1


# ---------- SingleFlight ----------

async def test_singleflight_runs_once_for_concurrent_callers():
    sf, calls = SingleFlight(), {"n": 0}

    async def work():
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return "result"

    out = await asyncio.gather(*[sf.do("k", work) for _ in range(5)])
    assert calls["n"] == 1
    assert [r for r, _ in out] == ["result"] * 5
    assert sum(leader for _, leader in out) == 1
    await sf.do("k", work)               # finished entries are removed: this runs again
    assert calls["n"] == 2


async def test_singleflight_error_reaches_all_waiters_and_is_not_sticky():
    sf, calls = SingleFlight(), {"n": 0}

    async def boom():
        calls["n"] += 1
        await asyncio.sleep(0.02)
        raise ValueError("upstream down")

    res = await asyncio.gather(*[sf.do("k", boom) for _ in range(3)], return_exceptions=True)
    assert all(isinstance(r, ValueError) for r in res) and calls["n"] == 1
    with pytest.raises(ValueError):
        await sf.do("k", boom)
    assert calls["n"] == 2


async def test_cancelled_waiter_does_not_cancel_the_shared_work():
    sf, calls = SingleFlight(), {"n": 0}

    async def work():
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return "ok"

    t1 = asyncio.create_task(sf.do("k", work))
    t2 = asyncio.create_task(sf.do("k", work))
    await asyncio.sleep(0.01)
    t2.cancel()
    assert (await t1)[0] == "ok" and calls["n"] == 1


# ---------- API behaviour ----------

class FakeStore:
    def __init__(self): self.records = []
    async def record(self, r): self.records.append(r)
    async def close(self): pass
    async def summary(self, d): return []


async def no_sleep(_): pass


@pytest.fixture
def make_client(monkeypatch):
    made = []

    def make(providers, key="team-a:gw-a:1000,team-b:gw-b:1000", **cache_kw):
        monkeypatch.setattr(settings, "gateway_keys", key)
        monkeypatch.setattr(settings, "admin_key", "secret")
        c = TestClient(main.app)
        c.__enter__()
        made.append(c)
        fake = fakeredis.FakeAsyncRedis()
        c.app.state.limiter = TokenBucketLimiter(fake)
        c.app.state.spend = SpendTracker(fake)
        c.app.state.cache = ResponseCache(fake, **cache_kw)
        c.app.state.singleflight = SingleFlight()
        c.app.state.store = FakeStore()
        c.app.state.router = ReliableRouter(providers, RetryPolicy(max_attempts=1), sleep=no_sleep)
        return c

    yield make
    for c in made:
        c.__exit__(None, None, None)


def upstream(name, calls, fail=None, delay=0.0):
    async def handler(request: httpx.Request):
        calls[name] = calls.get(name, 0) + 1
        if delay:
            await asyncio.sleep(delay)
        if fail and fail.get("on"):
            return httpx.Response(503, text="down")
        return httpx.Response(200, json=completion(f"from {name}"))
    return OpenAIProvider(httpx.AsyncClient(transport=httpx.MockTransport(handler)), name=name)


def test_miss_then_hit_calls_upstream_once(make_client):
    calls = {}
    c = make_client([upstream("openai", calls)])
    r1 = c.post(URL, json=BODY, headers=H)
    r2 = c.post(URL, json=BODY, headers=H)
    assert (r1.status_code, r2.status_code) == (200, 200)
    assert r1.headers["x-gateway-cache"] == "miss" and r2.headers["x-gateway-cache"] == "hit"
    assert r2.headers["x-gateway-provider"] == "openai"
    assert r1.json() == r2.json() and calls["openai"] == 1


def test_usage_records_show_cost_then_savings(make_client):
    c = make_client([upstream("openai", {})])
    c.post(URL, json=BODY, headers=H)
    c.post(URL, json=BODY, headers=H)
    first, second = c.app.state.store.records
    assert first.cache_hit is False and first.cost_usd == pytest.approx(0.00075)
    assert second.cache_hit is True and second.cost_usd == 0.0
    assert second.saved_usd == pytest.approx(first.cost_usd)
    assert (second.prompt_tokens, second.completion_tokens) == (0, 0)


def test_hits_do_not_add_to_spend(make_client):
    import asyncio as aio
    c = make_client([upstream("openai", {})])
    for _ in range(4):
        c.post(URL, json=BODY, headers=H)
    spent = aio.new_event_loop().run_until_complete(c.app.state.spend.get("team-a"))
    assert spent == pytest.approx(0.00075)          # only the single real call


def test_non_deterministic_requests_are_not_cached(make_client):
    calls = {}
    c = make_client([upstream("openai", calls)])
    body = {k: v for k, v in BODY.items() if k != "temperature"}
    r1 = c.post(URL, json=body, headers=H)
    c.post(URL, json=body, headers=H)
    assert calls["openai"] == 2 and "x-gateway-cache" not in r1.headers


def test_force_and_bypass_headers(make_client):
    calls = {}
    c = make_client([upstream("openai", calls)])
    body = {k: v for k, v in BODY.items() if k != "temperature"}
    c.post(URL, json=body, headers={**H, "X-Cache-Mode": "force"})
    r = c.post(URL, json=body, headers={**H, "X-Cache-Mode": "force"})
    assert r.headers["x-gateway-cache"] == "hit" and calls["openai"] == 1
    c.post(URL, json=BODY, headers={**H, "X-Cache-Mode": "bypass"})   # skips read and write
    c.post(URL, json=BODY, headers={**H, "X-Cache-Mode": "bypass"})
    assert calls["openai"] == 3


def test_cache_is_not_shared_between_api_keys(make_client):
    calls = {}
    c = make_client([upstream("openai", calls)])
    c.post(URL, json=BODY, headers=H)
    r = c.post(URL, json=BODY, headers={"Authorization": "Bearer gw-b"})
    assert r.headers["x-gateway-cache"] == "miss" and calls["openai"] == 2


def test_fallback_answers_are_never_cached(make_client):
    calls, openai_down = {}, {"on": True}
    c = make_client([upstream("openai", calls, fail=openai_down), upstream("ollama", calls)])
    r1 = c.post(URL, json=BODY, headers=H)
    assert r1.headers["x-gateway-provider"] == "ollama"
    assert asyncio.new_event_loop().run_until_complete(
        c.app.state.cache.get(cache_key("team-a", BODY))) is None
    openai_down["on"] = False                                     # primary recovers
    r2 = c.post(URL, json=BODY, headers=H)
    assert r2.headers["x-gateway-provider"] == "openai" and r2.headers["x-gateway-cache"] == "miss"
    assert c.post(URL, json=BODY, headers=H).headers["x-gateway-cache"] == "hit"


def test_hits_skip_the_token_limit(make_client):
    c = make_client([upstream("openai", {})], key="team-a:gw-a:1000:1000:0")
    body = {**BODY, "max_tokens": 900}                           # estimate ~904 of a 1000 limit
    codes = [c.post(URL, json=body, headers=H).status_code for _ in range(6)]
    assert codes == [200] * 6                                    # would 429 if hits charged tokens


def test_errors_are_not_cached(make_client):
    calls, down = {}, {"on": True}
    c = make_client([upstream("openai", calls, fail=down)])
    assert c.post(URL, json=BODY, headers=H).status_code == 502
    down["on"] = False
    r = c.post(URL, json=BODY, headers=H)
    assert r.status_code == 200 and r.headers["x-gateway-cache"] == "miss"


def test_streaming_requests_bypass_the_cache(make_client):
    def sse(request):
        return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n')
    p = OpenAIProvider(httpx.AsyncClient(transport=httpx.MockTransport(sse)), name="openai")
    c = make_client([p])
    assert c.post(URL, json={**BODY, "stream": True}, headers=H).status_code == 200
    assert asyncio.new_event_loop().run_until_complete(
        c.app.state.cache.get(cache_key("team-a", BODY))) is None


def test_concurrent_identical_requests_make_one_upstream_call(make_client):
    calls = {}
    c = make_client([upstream("openai", calls, delay=0.3)])
    with ThreadPoolExecutor(8) as pool:
        rs = list(pool.map(lambda _: c.post(URL, json=BODY, headers=H), range(8)))
    assert all(r.status_code == 200 for r in rs)
    assert calls["openai"] == 1                                  # 8 requests, 1 paid call
    kinds = sorted(r.headers["x-gateway-cache"] for r in rs)
    assert kinds.count("miss") == 1 and kinds.count("coalesced") + kinds.count("hit") == 7
    recs = c.app.state.store.records
    assert sum(1 for r in recs if r.cost_usd) == 1               # only the leader cost money


def test_admin_flush(make_client):
    calls = {}
    c = make_client([upstream("openai", calls)])
    c.post(URL, json=BODY, headers=H)
    assert c.post("/admin/cache/flush").status_code == 403
    r = c.post("/admin/cache/flush?key_name=team-a", headers={"X-Admin-Key": "secret"})
    assert r.json() == {"deleted": 1}
    assert c.post(URL, json=BODY, headers=H).headers["x-gateway-cache"] == "miss"
    assert calls["openai"] == 2


def test_redis_failure_fails_open(make_client):
    class Broken:
        async def get(self, *a, **k): raise rex.ConnectionError("down")
        async def set(self, *a, **k): raise rex.ConnectionError("down")
    calls = {}
    c = make_client([upstream("openai", calls)])
    c.app.state.cache = Broken()
    assert c.post(URL, json=BODY, headers=H).status_code == 200
    assert calls["openai"] == 1


def test_cache_disabled_when_ttl_zero(monkeypatch):
    monkeypatch.setattr(settings, "cache_ttl_s", 0)
    monkeypatch.setattr(settings, "gateway_keys", "team-a:gw-a:1000")
    with TestClient(main.app) as c:
        assert c.app.state.cache is None
