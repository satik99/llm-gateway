import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def _isolate_from_dotenv(monkeypatch):
    """Tests must not depend on the developer's .env: no real embedder, and no real
    semantic cache (each test installs its own fakes after startup)."""
    monkeypatch.setattr(settings, "embed_provider", "")


@pytest.fixture(autouse=True)
def _no_real_redis(monkeypatch, request):
    """App startup must never open a socket to a real Redis during tests (it leaked
    connections across event loops and could touch the developer's real cache)."""
    if request.node.get_closest_marker("real_redis_client"):
        return
    import fakeredis
    import redis.asyncio as aioredis

    monkeypatch.setattr(aioredis, "from_url", lambda *a, **k: fakeredis.FakeAsyncRedis())


def pytest_configure(config):
    config.addinivalue_line("markers", "real_redis_client: test builds the real redis client (to a dead port)")


@pytest.fixture(autouse=True)
def _no_real_database(monkeypatch, request):
    """Never connect app startup to the developer's real Postgres. Tests that need a database
    set their own DSN (TEST_DATABASE_URL, always a *_test database)."""
    monkeypatch.setattr(settings, "database_url", "")
