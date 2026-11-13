"""Key management against a real Postgres (skipped without TEST_DATABASE_URL, a *_test database)."""
import os

import fakeredis
import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from app import main
from app.budget import SpendTracker
from app.cache import ResponseCache, SingleFlight
from app.config import settings
from app.keystore import KeyStore, NameTaken, NotFound, hash_key
from app.providers.openai import OpenAIProvider
from app.ratelimit import TokenBucketLimiter
from app.reliability import ReliableRouter, RetryPolicy

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set")
URL = "/v1/chat/completions"
ADMIN = {"X-Admin-Key": "secret"}
BODY = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}


def _guard():
    dbname = DSN.rsplit("/", 1)[-1].split("?")[0]
    if not dbname.endswith("_test"):
        pytest.fail(f"TEST_DATABASE_URL must end in _test (got {dbname!r}); these tests wipe api_keys")


@pytest_asyncio.fixture
async def ks():
    _guard()
    s = await KeyStore.connect(DSN, cache_ttl_s=60)
    await s._pool.execute("TRUNCATE api_keys")
    yield s
    await s.close()


# ---------- KeyStore ----------

async def test_create_lookup_and_only_the_hash_is_stored(ks):
    info, token = await ks.create("team-x", 60, 1000, 5.0)
    assert token.startswith("gw-") and info["key_prefix"] == token[:7]
    assert "key_hash" not in info
    key = await ks.lookup(token)
    assert (key.name, key.rpm, key.tpm, key.monthly_budget_usd) == ("team-x", 60, 1000, 5.0)
    row = await ks._pool.fetchrow("SELECT * FROM api_keys WHERE name='team-x'")
    assert row["key_hash"] == hash_key(token) and token not in "".join(str(v) for v in row.values())
    assert await ks.lookup("gw-wrong") is None


async def test_duplicate_name_and_validation(ks):
    await ks.create("a", 1, 0, 0)
    with pytest.raises(NameTaken):
        await ks.create("a", 1, 0, 0)


async def test_revoke_rotate_and_update_take_effect_immediately_in_process(ks):
    _, t1 = await ks.create("a", 10, 0, 0)
    assert await ks.lookup(t1) is not None                     # now cached
    _, t2 = await ks.rotate("a")
    assert await ks.lookup(t1) is None and (await ks.lookup(t2)).name == "a"
    await ks.update("a", {"rpm": 99, "tpm": 500})
    assert (await ks.lookup(t2)).rpm == 99
    out = await ks.revoke("a")
    assert out["active"] is False and out["revoked_at"]
    assert await ks.lookup(t2) is None
    with pytest.raises(NotFound):
        await ks.rotate("a")                                   # revoked keys can't rotate
    await ks.update("a", {"active": True})
    assert (await ks.lookup(t2)).name == "a"                   # reactivation restores the secret


async def test_cache_serves_stale_when_db_is_down_but_unknown_keys_get_unavailable(ks):
    _, t = await ks.create("a", 10, 0, 0)
    await ks.lookup(t)
    await ks._pool.close()
    assert (await ks.lookup(t)).name == "a"
    from app.keystore import KeyStoreUnavailable
    with pytest.raises(KeyStoreUnavailable):
        await ks.lookup("gw-never-seen")


# ---------- through the HTTP API ----------

def upstream():
    async def handler(request):
        return httpx.Response(200, json={
            "model": "gpt-4o-mini", "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    return OpenAIProvider(httpx.AsyncClient(transport=httpx.MockTransport(handler)), name="openai")


async def no_sleep(_): pass


@pytest.fixture
def client(monkeypatch):
    _guard()
    monkeypatch.setattr(settings, "database_url", DSN)
    monkeypatch.setattr(settings, "gateway_keys", "static-team:gw-static:100")
    monkeypatch.setattr(settings, "admin_key", "secret")
    monkeypatch.setattr(settings, "key_cache_ttl_s", 0.0)       # no caching: see changes at once
    with TestClient(main.app) as c:
        fake = fakeredis.FakeAsyncRedis()
        c.app.state.limiter = TokenBucketLimiter(fake)
        c.app.state.spend = SpendTracker(fake)
        c.app.state.cache = None
        c.app.state.semantic = None
        c.app.state.store = None
        c.app.state.router = ReliableRouter([upstream()], RetryPolicy(max_attempts=1), sleep=no_sleep)
        yield c


def test_full_lifecycle_over_http(client):
    c = client
    assert c.post("/admin/keys", json={"name": "a1", "rpm": 5}).status_code == 403         # needs admin
    r = c.post("/admin/keys", json={"name": "lc-team", "rpm": 2, "tpm": 0}, headers=ADMIN)
    assert r.status_code == 201
    token = r.json()["key"]
    H = {"Authorization": f"Bearer {token}"}

    assert c.post(URL, json=BODY, headers=H).status_code == 200            # new key works at once
    assert c.post(URL, json=BODY, headers=H).status_code == 200
    assert c.post(URL, json=BODY, headers=H).status_code == 429            # its own rpm=2 applies
    assert c.patch("/admin/keys/lc-team", json={"rpm": 100}, headers=ADMIN).json()["rpm"] == 100

    listed = c.get("/admin/keys", headers=ADMIN).json()["keys"]
    mine = next(k for k in listed if k["name"] == "lc-team")
    assert "key" not in mine and "key_hash" not in mine and mine["key_prefix"] == token[:7]

    new = c.post("/admin/keys/lc-team/rotate", headers=ADMIN).json()["key"]
    assert c.post(URL, json=BODY, headers=H).status_code == 401            # old secret dead
    assert c.post(URL, json=BODY, headers={"Authorization": f"Bearer {new}"}).status_code in (200, 429)

    assert c.delete("/admin/keys/lc-team", headers=ADMIN).json()["active"] is False
    assert c.post(URL, json=BODY, headers={"Authorization": f"Bearer {new}"}).status_code == 401
    assert c.delete("/admin/keys/nope", headers=ADMIN).status_code == 404


def test_validation_conflicts_and_static_keys_still_work(client):
    c = client
    assert c.post(URL, json=BODY, headers={"Authorization": "Bearer gw-static"}).status_code == 200
    assert c.post("/admin/keys", json={"name": "BAD NAME", "rpm": 5}, headers=ADMIN).status_code == 422
    assert c.post("/admin/keys", json={"name": "ok-1", "rpm": 0}, headers=ADMIN).status_code == 422
    assert c.post("/admin/keys", json={"name": "static-team", "rpm": 5}, headers=ADMIN).status_code == 409
    assert c.post("/admin/keys", json={"name": "dup-1", "rpm": 5}, headers=ADMIN).status_code in (201, 409)
    assert c.post("/admin/keys", json={"name": "dup-1", "rpm": 5}, headers=ADMIN).status_code == 409
    assert c.post(URL, json=BODY, headers={"Authorization": "Bearer gw-nope"}).status_code == 401
    c.delete("/admin/keys/dup-1", headers=ADMIN)


def test_key_admin_requires_database(monkeypatch):
    monkeypatch.setattr(settings, "gateway_keys", "t:k:5")
    monkeypatch.setattr(settings, "admin_key", "secret")
    with TestClient(main.app) as c:
        assert c.get("/admin/keys", headers=ADMIN).status_code == 503
