import json

import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.budget import SpendTracker
from app.config import settings
from app.providers.openai import OpenAIProvider
from app.ratelimit import TokenBucketLimiter
from app.reliability import ReliableRouter, RetryPolicy

H = {"Authorization": "Bearer gw-a"}
URL = "/v1/chat/completions"
BODY = {"model": "gpt-4o-mini", "stream": True,
        "messages": [{"role": "user", "content": "hi"}]}
SEEN = {}  # what the upstream received


def sse(*events: dict | str) -> bytes:
    out = b""
    for e in events:
        out += (e if isinstance(e, str) else "data: " + json.dumps(e)).encode() + b"\n\n"
    return out


def delta(text):
    return {"model": "gpt-4o-mini-2024-07-18", "choices": [{"delta": {"content": text}}]}


USAGE = {"model": "gpt-4o-mini-2024-07-18", "choices": [],
         "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


def good_stream(request: httpx.Request) -> httpx.Response:
    SEEN["payload"] = json.loads(request.content)
    return httpx.Response(200, content=sse(delta("Hel"), delta("lo"), USAGE, "data: [DONE]"))


def stream_without_usage(request):
    return httpx.Response(200, content=sse(delta("x" * 40), "data: [DONE]"))


class DyingStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield sse(delta("partial"))
        raise httpx.ReadError("connection reset")


def dies_midway(request):
    return httpx.Response(200, stream=DyingStream())


def down(request):
    return httpx.Response(503, text="down")


def prov(name, handler):
    return OpenAIProvider(httpx.AsyncClient(transport=httpx.MockTransport(handler)), name=name)


class NoStreamProvider:           # like the Anthropic provider: chat() only
    name = "nostream"
    async def chat(self, payload): return {}


class FakeStore:
    def __init__(self): self.records = []
    async def record(self, r): self.records.append(r)
    async def close(self): pass
    async def summary(self, d): return []


async def no_sleep(_): pass


@pytest.fixture
def make_client(monkeypatch):
    made = []

    def make(providers, key="team-a:gw-a:1000:0:0"):
        SEEN.clear()
        monkeypatch.setattr(settings, "gateway_keys", key)
        c = TestClient(main.app)
        c.__enter__()
        made.append(c)
        fake = fakeredis.FakeAsyncRedis()
        c.app.state.limiter = TokenBucketLimiter(fake)
        c.app.state.spend = SpendTracker(fake)
        c.app.state.store = FakeStore()
        c.app.state.router = ReliableRouter(providers, RetryPolicy(max_attempts=1), sleep=no_sleep)
        return c

    yield make
    for c in made:
        c.__exit__(None, None, None)


def test_stream_is_passed_through_and_usage_recorded(make_client):
    c = make_client([prov("openai", good_stream)])
    r = c.post(URL, json=BODY, headers=H)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-gateway-provider"] == "openai"
    assert "data: [DONE]" in r.text and '"Hel"' in r.text
    # we asked upstream to include usage in the final chunk
    assert SEEN["payload"]["stream_options"]["include_usage"] is True
    (rec,) = c.app.state.store.records
    assert (rec.status, rec.provider, rec.fallback) == (200, "openai", False)
    assert (rec.prompt_tokens, rec.completion_tokens) == (11, 7)
    assert rec.cost_usd == pytest.approx((11 * 0.15 + 7 * 0.60) / 1_000_000)


def test_falls_over_before_first_byte(make_client):
    c = make_client([prov("openai", down), prov("ollama", good_stream)])
    r = c.post(URL, json=BODY, headers=H)
    assert r.status_code == 200 and r.headers["x-gateway-provider"] == "ollama"
    (rec,) = c.app.state.store.records
    assert rec.fallback is True and rec.cost_usd == 0.0


def test_midstream_failure_ends_with_error_event_and_is_recorded(make_client):
    c = make_client([prov("openai", dies_midway), prov("ollama", good_stream)])
    r = c.post(URL, json=BODY, headers=H)
    assert "partial" in r.text and "upstream stream failed" in r.text
    assert "ollama" not in r.headers["x-gateway-provider"]   # no switching after the first byte
    (rec,) = c.app.state.store.records
    assert rec.status == 502 and rec.completion_tokens >= 1  # estimated from "partial"
    assert c.app.state.router.breakers["openai"]._failures == 1


def test_missing_usage_chunk_falls_back_to_estimate(make_client):
    c = make_client([prov("openai", stream_without_usage)])
    assert c.post(URL, json=BODY, headers=H).status_code == 200
    (rec,) = c.app.state.store.records
    assert rec.completion_tokens == 10            # 40 chars // 4
    assert rec.prompt_tokens >= 1 and rec.status == 200


def test_all_providers_failing_gives_502_json(make_client):
    c = make_client([prov("openai", down)])
    r = c.post(URL, json=BODY, headers=H)
    assert r.status_code == 502 and "all providers failed" in r.json()["detail"]
    (rec,) = c.app.state.store.records
    assert (rec.provider, rec.status) == ("none", 502)


def test_providers_without_streaming_are_skipped(make_client):
    c = make_client([NoStreamProvider(), prov("ollama", good_stream)])
    assert c.post(URL, json=BODY, headers=H).headers["x-gateway-provider"] == "ollama"
    c2 = make_client([NoStreamProvider()])
    r = c2.post(URL, json=BODY, headers=H)
    assert r.status_code == 502 and "not supported" in r.json()["detail"]


def test_token_limit_is_reconciled_after_stream(make_client):
    import asyncio
    c = make_client([prov("openai", good_stream)], key="team-a:gw-a:1000:1000:0")
    assert c.post(URL, json={**BODY, "max_tokens": 900}, headers=H).status_code == 200  # est ~904
    left = asyncio.new_event_loop().run_until_complete(
        c.app.state.limiter.check("team-a:tpm", 1000, 1000, cost=0)).remaining
    assert left == pytest.approx(1000 - 18, abs=5)          # real usage was 11 + 7


def test_upstream_client_error_passes_through(make_client):
    def bad(request): return httpx.Response(400, json={"error": {"message": "bad model"}})
    c = make_client([prov("openai", bad)])
    r = c.post(URL, json=BODY, headers=H)
    assert r.status_code == 400 and r.json()["detail"]["error"]["message"] == "bad model"
