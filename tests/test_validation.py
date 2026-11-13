import fakeredis
import pytest
from fastapi.testclient import TestClient

from app import main
from app.budget import SpendTracker
from app.config import settings
from app.ratelimit import TokenBucketLimiter

H = {"Authorization": "Bearer gw-a", "Content-Type": "application/json"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "gateway_keys", "team-a:gw-a:100:1000")
    with TestClient(main.app) as c:
        fake = fakeredis.FakeAsyncRedis()
        c.app.state.limiter = TokenBucketLimiter(fake)
        c.app.state.spend = SpendTracker(fake)
        yield c


@pytest.mark.parametrize("raw", [
    b'{"model":"x","messages":[]}]',          # trailing garbage (the real-world typo)
    b"not json at all",
    b"",
])
def test_malformed_json_is_400_not_500(client, raw):
    assert client.post("/v1/chat/completions", content=raw, headers=H).status_code == 400


@pytest.mark.parametrize("payload", [
    [],                                        # JSON but not an object
    {"model": "x"},                            # no messages
    {"model": "x", "messages": "hi"},          # wrong type
    {"model": "x", "messages": ["hi"]},        # messages not objects
])
def test_bad_shapes_are_400(client, payload):
    assert client.post("/v1/chat/completions", json=payload, headers=H).status_code == 400
