import fakeredis
import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from app import main
from app.budget import SpendTracker
from app.config import settings
from app.providers.openai import OpenAIProvider
from app.ratelimit import TokenBucketLimiter
from app.reliability import ReliableRouter, RetryPolicy
from app.tokens import estimate_tokens

H = {"Authorization": "Bearer gw-a"}
URL = "/v1/chat/completions"


def body(max_tokens=None):
    b = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    if max_tokens:
        b["max_tokens"] = max_tokens
    return b


# ---------- unit tests ----------

def test_estimate_tokens_uses_prompt_and_max_tokens():
    p = {"messages": [{"role": "user", "content": "x" * 400}], "max_tokens": 100}
    assert estimate_tokens(p) == 100 + 100 + 4  # 400/4 + overhead + max_tokens


def test_estimate_tokens_defaults_completion_guess():
    assert estimate_tokens({"messages": []}) == 256


def test_estimate_tokens_handles_content_parts():
    p = {"messages": [{"role": "user", "content": [{"type": "text", "text": "x" * 40}]}],
         "max_tokens": 1}
    assert estimate_tokens(p) == 10 + 4 + 1


@pytest_asyncio.fixture
async def limiter():
    c = fakeredis.FakeAsyncRedis()
    yield TokenBucketLimiter(c)
    await c.aclose()


async def remaining(limiter, bucket, cap, per_min):
    return (await limiter.check(bucket, cap, per_min, cost=0)).remaining


async def test_adjust_refund_charge_and_clamps(limiter):
    assert (await limiter.check("b", 100, 6, cost=100)).allowed
    assert await remaining(limiter, "b", 100, 6) < 1
    await limiter.adjust("b", 100, 6, 40)                      # refund
    assert await remaining(limiter, "b", 100, 6) == pytest.approx(40, abs=1)
    await limiter.adjust("b", 100, 6, 1000)                    # capped at capacity
    assert await remaining(limiter, "b", 100, 6) == pytest.approx(100, abs=1)
    await limiter.adjust("b", 100, 6, -1000)                   # floored at 0
    assert await remaining(limiter, "b", 100, 6) == pytest.approx(0, abs=1)


async def test_spend_tracker_accumulates_in_micro_dollars():
    t = SpendTracker(fakeredis.FakeAsyncRedis())
    await t.add("team", 0.000006)
    await t.add("team", 0.000006)
    await t.add("team", 0)          # ignored
    assert await t.get("team") == pytest.approx(0.000012)
    assert await t.get("other") == 0.0


# ---------- API tests ----------

UP = {"usage": {"prompt_tokens": 11, "completion_tokens": 7}}


def upstream(request):
    return httpx.Response(200, json={
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [{"message": {"content": "hello"}}], "usage": UP["usage"]})


def failing(request):
    return httpx.Response(503, text="down")


def prov(name, handler):
    return OpenAIProvider(httpx.AsyncClient(transport=httpx.MockTransport(handler)), name=name)


async def no_sleep(_):
    pass


class NullStore:
    async def record(self, r): pass
    async def close(self): pass
    async def summary(self, d): return []


@pytest.fixture
def make_client(monkeypatch):
    clients = []

    def make(providers, key="team-a:gw-a:1000:1000:0.001"):
        UP["usage"] = {"prompt_tokens": 11, "completion_tokens": 7}
        monkeypatch.setattr(settings, "gateway_keys", key)
        c = TestClient(main.app)
        c.__enter__()
        clients.append(c)
        fake = fakeredis.FakeAsyncRedis()
        c.app.state.limiter = TokenBucketLimiter(fake)
        c.app.state.spend = SpendTracker(fake)
        c.app.state.store = NullStore()
        c.app.state.router = ReliableRouter(providers, RetryPolicy(max_attempts=1), sleep=no_sleep)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def tpm_left(c):
    import asyncio
    return asyncio.new_event_loop().run_until_complete(
        c.app.state.limiter.check("team-a:tpm", 1000, 1000, cost=0)).remaining


def test_request_larger_than_tpm_gets_413(make_client):
    c = make_client([prov("openai", upstream)])
    r = c.post(URL, json=body(max_tokens=5000), headers=H)
    assert r.status_code == 413 and r.json()["limit_per_minute"] == 1000


def test_tpm_blocks_once_real_usage_fills_the_bucket(make_client):
    c = make_client([prov("openai", upstream)])
    UP["usage"] = {"prompt_tokens": 10, "completion_tokens": 440}   # 450 real tokens
    codes = [c.post(URL, json=body(max_tokens=450), headers=H).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = c.post(URL, json=body(max_tokens=450), headers=H)
    assert r.json()["error"] == "token rate limit exceeded" and "retry-after" in r.headers


def test_overestimate_is_refunded_after_response(make_client):
    c = make_client([prov("openai", upstream)])
    assert c.post(URL, json=body(max_tokens=900), headers=H).status_code == 200  # est ~904
    assert tpm_left(c) == pytest.approx(1000 - 18, abs=5)                        # real = 18


def test_failed_request_refunds_estimate(make_client):
    c = make_client([prov("openai", failing)])
    assert c.post(URL, json=body(max_tokens=900), headers=H).status_code == 502
    assert tpm_left(c) == pytest.approx(1000, abs=5)


def test_budget_blocks_with_402_then_stays_blocked(make_client):
    c = make_client([prov("openai", upstream)], key="team-a:gw-a:1000:0:0.001")  # no TPM limit
    UP["usage"] = {"prompt_tokens": 1000, "completion_tokens": 1000}  # ~$0.00075 each
    assert c.post(URL, json=body(), headers=H).status_code == 200   # spent 0.00075 < 0.001
    assert c.post(URL, json=body(), headers=H).status_code == 200   # spent 0.0015 >= 0.001
    r = c.post(URL, json=body(), headers=H)
    assert r.status_code == 402
    assert r.json()["error"] == "monthly budget exceeded"
    assert r.json()["budget_usd"] == 0.001


def test_fallback_to_free_provider_does_not_burn_budget(make_client):
    c = make_client([prov("openai", failing), prov("ollama", upstream)],
                    key="team-a:gw-a:1000:0:0.001")  # no TPM limit
    UP["usage"] = {"prompt_tokens": 100000, "completion_tokens": 100000}
    for _ in range(3):
        assert c.post(URL, json=body(), headers=H).status_code == 200
    import asyncio
    spent = asyncio.new_event_loop().run_until_complete(c.app.state.spend.get("team-a"))
    assert spent == 0.0


def test_no_tpm_or_budget_configured_means_unlimited(make_client):
    c = make_client([prov("openai", upstream)], key="team-a:gw-a:1000")
    assert c.post(URL, json=body(max_tokens=10**6), headers=H).status_code == 200


def test_redis_failures_fail_open_for_all_limits(make_client):
    import redis.exceptions as rex

    class Broken:
        async def check(self, *a, **k): raise rex.ConnectionError("down")
        async def adjust(self, *a, **k): raise rex.ConnectionError("down")
        async def get(self, *a, **k): raise rex.ConnectionError("down")
        async def add(self, *a, **k): raise rex.ConnectionError("down")

    c = make_client([prov("openai", upstream)])
    c.app.state.limiter = Broken()
    c.app.state.spend = Broken()
    assert c.post(URL, json=body(max_tokens=100), headers=H).status_code == 200
