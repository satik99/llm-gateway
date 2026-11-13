import asyncio

import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient

from app import main, metrics
from app.budget import SpendTracker
from app.cache import ResponseCache, SingleFlight
from app.config import settings
from app.providers.openai import OpenAIProvider
from app.ratelimit import TokenBucketLimiter
from app.reliability import CircuitBreaker, ReliableRouter, RetryPolicy

URL = "/v1/chat/completions"
H = {"Authorization": "Bearer gw-a"}
BODY = {"model": "gpt-4o-mini", "temperature": 0,
        "messages": [{"role": "user", "content": "What is your refund policy?"}]}


def val(name, **labels):
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


async def no_sleep(_): pass


def provider(name, status=200):
    async def handler(request):
        if status != 200:
            return httpx.Response(status, text="down")
        return httpx.Response(200, json={
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1000}})
    return OpenAIProvider(httpx.AsyncClient(transport=httpx.MockTransport(handler)), name=name)


@pytest.fixture
def client(monkeypatch):
    made = []

    def make(providers, keys="team-a:gw-a:1000"):
        monkeypatch.setattr(settings, "gateway_keys", keys)
        c = TestClient(main.app)
        c.__enter__()
        made.append(c)
        fake = fakeredis.FakeAsyncRedis()
        c.app.state.limiter = TokenBucketLimiter(fake)
        c.app.state.spend = SpendTracker(fake)
        c.app.state.cache = ResponseCache(fake)
        c.app.state.semantic = None
        c.app.state.singleflight = SingleFlight()
        c.app.state.store = None
        c.app.state.router = ReliableRouter(
            providers, RetryPolicy(max_attempts=1), sleep=no_sleep,
            breaker_factory=lambda: CircuitBreaker(2, 60))
        metrics.bind_breakers(c.app.state.router.breakers)
        return c

    yield make
    for c in made:
        c.__exit__(None, None, None)


def test_miss_then_hit_are_counted_with_tokens_cost_and_savings(client):
    c = client([provider("openai")])
    b = {k: val(k, key="team-a", provider="openai", status="200", cache="miss") for k in ["gateway_requests_total"]}
    h0 = val("gateway_requests_total", key="team-a", provider="openai", status="200", cache="hit")
    t0 = val("gateway_tokens_total", key="team-a", provider="openai", type="prompt")
    s0 = val("gateway_saved_usd_total", key="team-a")
    c.post(URL, json=BODY, headers=H)
    c.post(URL, json=BODY, headers=H)
    assert val("gateway_requests_total", key="team-a", provider="openai", status="200", cache="miss") == b["gateway_requests_total"] + 1
    assert val("gateway_requests_total", key="team-a", provider="openai", status="200", cache="hit") == h0 + 1
    assert val("gateway_tokens_total", key="team-a", provider="openai", type="prompt") == t0 + 1000
    assert val("gateway_saved_usd_total", key="team-a") == pytest.approx(s0 + 0.00075)
    assert val("gateway_request_duration_seconds_count", provider="openai", cache="hit") >= 1


def test_unauthenticated_and_rate_limited_requests_are_labelled(client):
    c = client([provider("openai")], keys="team-a:gw-a:1")
    u0 = val("gateway_requests_total", key="unauthenticated", provider="none", status="401", cache="none")
    l0 = val("gateway_limited_total", key="team-a", reason="rpm")
    c.post(URL, json=BODY, headers={"Authorization": "Bearer nope"})
    c.post(URL, json={**BODY, "temperature": 1}, headers=H)
    r = c.post(URL, json={**BODY, "temperature": 1}, headers=H)
    assert r.status_code == 429
    assert val("gateway_requests_total", key="unauthenticated", provider="none", status="401", cache="none") == u0 + 1
    assert val("gateway_limited_total", key="team-a", reason="rpm") == l0 + 1
    assert val("gateway_requests_total", key="team-a", provider="none", status="429", cache="none") >= 1


def test_fallback_and_circuit_state(client):
    c = client([provider("openai", status=503), provider("ollama")])
    f0 = val("gateway_fallbacks_total", provider="ollama")
    for i in range(2):
        r = c.post(URL, json={**BODY, "temperature": 1, "max_tokens": i + 1}, headers=H)
        assert r.headers["x-gateway-provider"] == "ollama"
    assert val("gateway_fallbacks_total", provider="ollama") == f0 + 2
    assert val("gateway_circuit_state", provider="openai") == 2      # breaker opened after 2 failures
    assert val("gateway_circuit_state", provider="ollama") == 0


def test_all_providers_failed_is_a_502(client):
    c = client([provider("openai", status=503)])
    e0 = val("gateway_requests_total", key="team-a", provider="none", status="502", cache="none")
    assert c.post(URL, json={**BODY, "temperature": 1}, headers=H).status_code == 502
    assert val("gateway_requests_total", key="team-a", provider="none", status="502", cache="none") == e0 + 1


def test_metrics_endpoint_exposes_text_and_can_be_protected(client, monkeypatch):
    c = client([provider("openai")])
    c.post(URL, json=BODY, headers=H)
    r = c.get("/metrics")
    assert r.status_code == 200 and "gateway_requests_total" in r.text
    assert "gateway_circuit_state" in r.text and "gateway_inflight_requests" in r.text
    monkeypatch.setattr(settings, "metrics_token", "s3cret")
    assert c.get("/metrics").status_code == 403
    assert c.get("/metrics", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_inflight_returns_to_zero(client):
    c = client([provider("openai")])
    c.post(URL, json=BODY, headers=H)
    assert val("gateway_inflight_requests") == 0
