"""Integration test against a real Postgres.
Run with: TEST_DATABASE_URL=postgresql://gateway:gateway@localhost:5433/gateway_test pytest
Skipped when the variable is unset."""
import os

import pytest
import pytest_asyncio

from app.store import UsageRecord, UsageStore

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set")


@pytest_asyncio.fixture
async def store():
    # These tests TRUNCATE the usage table. Refuse to touch anything but a *_test database.
    dbname = DSN.rsplit("/", 1)[-1].split("?")[0]
    if not dbname.endswith("_test"):
        pytest.fail(f"TEST_DATABASE_URL must point at a database ending in _test (got {dbname!r}); "
                    "this test deletes all rows in `usage`.")
    s = await UsageStore.connect(DSN)
    await s._pool.execute("TRUNCATE usage")
    yield s
    await s.close()


def rec(key, provider="openai", cost=0.001, fallback=False, status=200, lat=100):
    return UsageRecord(key, "gpt-4o-mini", "gpt-4o-mini-2024-07-18", provider,
                       fallback, 10, 5, cost, lat, status)


async def test_summary_groups_by_key_and_provider(store):
    await store.record(rec("team-a", cost=0.002, lat=100))
    await store.record(rec("team-a", cost=0.004, lat=300))
    await store.record(rec("team-a", provider="ollama", cost=0.0, fallback=True))
    rows = await store.summary(days=1)
    a_openai = next(r for r in rows if r["key_name"] == "team-a" and r["provider"] == "openai")
    assert a_openai["requests"] == 2
    assert a_openai["cost_usd"] == pytest.approx(0.006)
    assert a_openai["avg_latency_ms"] == 200
    a_ollama = next(r for r in rows if r["key_name"] == "team-a" and r["provider"] == "ollama")
    assert a_ollama["fallbacks"] == 1
    assert rows[0]["key_name"] == "team-a"  # most expensive first


async def test_null_cost_is_stored_and_summed_as_zero(store):
    await store.record(rec("team-a", cost=None))
    rows = await store.summary(days=1)
    assert rows[0]["cost_usd"] == 0.0


async def test_cache_hits_and_savings_are_summarised(store):
    await store.record(rec("team-a", cost=0.001))                     # a real call
    for _ in range(3):                                                # three cache hits
        r = rec("team-a", cost=0.0)
        r.cache_hit, r.saved_usd = True, 0.001
        await store.record(r)
    row = (await store.summary(days=1))[0]
    assert row["requests"] == 4 and row["cache_hits"] == 3
    assert row["cost_usd"] == pytest.approx(0.001)
    assert row["saved_usd"] == pytest.approx(0.003)


async def test_schema_upgrade_is_idempotent_on_an_old_table(store):
    # simulate a pre-Day-8 database: drop the new columns, then connect again
    await store._pool.execute("ALTER TABLE usage DROP COLUMN cache_hit, DROP COLUMN saved_usd")
    s2 = await UsageStore.connect(DSN)           # must re-add them without error
    await s2.record(rec("team-a"))
    await UsageStore.connect(DSN)                # and be safe to run again
    assert (await s2.summary(days=1))[0]["cache_hits"] == 0
    await s2.close()


async def test_daily_groups_by_day_and_key(store):
    await store.record(rec("team-a", cost=0.002))
    await store.record(rec("team-a", cost=0.004))
    await store.record(rec("team-b", cost=0.001, status=502))
    rows = await store.daily(days=1)
    a = next(r for r in rows if r["key_name"] == "team-a")
    b = next(r for r in rows if r["key_name"] == "team-b")
    assert a["requests"] == 2 and a["cost_usd"] == pytest.approx(0.006) and a["tokens"] == 30
    assert b["errors"] == 1 and len({r["day"] for r in rows}) == 1
