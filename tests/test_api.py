import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import settings
from app.providers.openai import OpenAIProvider
from app.ratelimit import TokenBucketLimiter
from app.reliability import ReliableRouter


def fake_upstream(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "gateway_keys", "team-a:gw-a:2")
    with TestClient(main.app) as c:
        # swap real dependencies for fakes after startup
        c.app.state.limiter = TokenBucketLimiter(fakeredis.FakeAsyncRedis())
        c.app.state.router = ReliableRouter([OpenAIProvider(
            httpx.AsyncClient(transport=httpx.MockTransport(fake_upstream)))])
        yield c


BODY = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}


def test_rejects_bad_key(client):
    r = client.post("/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_proxies_and_rate_limits(client):
    h = {"Authorization": "Bearer gw-a"}
    assert client.post("/v1/chat/completions", json=BODY, headers=h).json()["choices"][0]["message"]["content"] == "hello"
    assert client.post("/v1/chat/completions", json=BODY, headers=h).status_code == 200
    r = client.post("/v1/chat/completions", json=BODY, headers=h)
    assert r.status_code == 429 and "retry-after" in r.headers


def test_redis_down_fails_open(client):
    import redis.exceptions

    class BrokenLimiter:
        async def check(self, *a, **k):
            raise redis.exceptions.ConnectionError("redis down")

    client.app.state.limiter = BrokenLimiter()
    r = client.post("/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer gw-a"})
    assert r.status_code == 200
