import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import settings
from app.providers.openai import OpenAIProvider
from app.budget import SpendTracker
from app.ratelimit import TokenBucketLimiter
from app.reliability import ReliableRouter, RetryPolicy

BODY = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
H = {"Authorization": "Bearer gw-a"}


class FakeStore:
    def __init__(self):
        self.records = []

    async def record(self, r):
        self.records.append(r)

    async def summary(self, days):
        return [{"key_name": "team-a", "requests": len(self.records)}]

    async def close(self):
        pass


def ok(request):
    return httpx.Response(200, json={
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7}})


def down(request):
    return httpx.Response(503, text="down")


def provider(name, handler):
    return OpenAIProvider(httpx.AsyncClient(transport=httpx.MockTransport(handler)), name=name)


async def no_sleep(_):
    pass


def make_client(monkeypatch, providers):
    monkeypatch.setattr(settings, "gateway_keys", "team-a:gw-a:100")
    monkeypatch.setattr(settings, "admin_key", "secret")
    c = TestClient(main.app)
    c.__enter__()
    fake = fakeredis.FakeAsyncRedis()
    c.app.state.limiter = TokenBucketLimiter(fake)
    c.app.state.spend = SpendTracker(fake)
    c.app.state.router = ReliableRouter(providers, RetryPolicy(max_attempts=1), sleep=no_sleep)
    c.app.state.store = FakeStore()
    return c


@pytest.fixture
def client(monkeypatch):
    c = make_client(monkeypatch, [provider("openai", ok)])
    yield c
    c.__exit__(None, None, None)


@pytest.fixture
def fallback_client(monkeypatch):
    c = make_client(monkeypatch, [provider("openai", down), provider("ollama", ok)])
    yield c
    c.__exit__(None, None, None)


def test_success_is_recorded_with_tokens_and_cost(client):
    assert client.post("/v1/chat/completions", json=BODY, headers=H).status_code == 200
    (r,) = client.app.state.store.records
    assert (r.key_name, r.provider, r.fallback, r.status) == ("team-a", "openai", False, 200)
    assert (r.prompt_tokens, r.completion_tokens) == (11, 7)
    assert r.served_model == "gpt-4o-mini-2024-07-18"
    assert r.cost_usd == pytest.approx((11 * 0.15 + 7 * 0.60) / 1_000_000)


def test_fallback_is_flagged_and_free(fallback_client):
    r = fallback_client.post("/v1/chat/completions", json=BODY, headers=H)
    assert r.headers["x-gateway-provider"] == "ollama"
    (rec,) = fallback_client.app.state.store.records
    assert rec.fallback is True and rec.cost_usd == 0.0


def test_total_failure_is_recorded_as_502(monkeypatch):
    c = make_client(monkeypatch, [provider("openai", down)])
    try:
        assert c.post("/v1/chat/completions", json=BODY, headers=H).status_code == 502
        (rec,) = c.app.state.store.records
    finally:
        c.__exit__(None, None, None)
    assert (rec.provider, rec.status) == ("none", 502)


def test_admin_requires_key(client):
    assert client.get("/admin/usage").status_code == 403
    assert client.get("/admin/usage", headers={"X-Admin-Key": "wrong"}).status_code == 403
    r = client.get("/admin/usage?days=7", headers={"X-Admin-Key": "secret"})
    assert r.status_code == 200 and r.json()["days"] == 7


def test_store_down_does_not_break_requests(client):
    client.app.state.store = None
    assert client.post("/v1/chat/completions", json=BODY, headers=H).status_code == 200
    assert client.get("/admin/usage", headers={"X-Admin-Key": "secret"}).status_code == 503
