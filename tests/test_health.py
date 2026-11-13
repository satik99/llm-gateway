import fakeredis
import pytest
from fastapi.testclient import TestClient

from app import main
from app.cache import ResponseCache
from app.config import settings


@pytest.mark.real_redis_client
def test_healthz_reports_redis_down_instead_of_hiding_it(monkeypatch):
    monkeypatch.setattr(settings, "gateway_keys", "t:k:10")
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1/0")   # nothing listens there
    with TestClient(main.app) as c:
        body = c.get("/healthz").json()
    assert body["ok"] is True and body["redis"] is False and body["cache"] is False


def test_healthz_reports_all_good(monkeypatch):
    monkeypatch.setattr(settings, "gateway_keys", "t:k:10")
    with TestClient(main.app) as c:
        fake = fakeredis.FakeAsyncRedis()
        c.app.state.redis = fake
        c.app.state.cache = ResponseCache(fake)
        body = c.get("/healthz").json()
    assert body["redis"] is True and body["cache"] is True
