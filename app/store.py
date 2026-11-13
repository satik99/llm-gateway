"""Usage log in Postgres: one row per request."""
import logging
from dataclasses import dataclass

import asyncpg

log = logging.getLogger("gateway.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id                BIGSERIAL PRIMARY KEY,
    ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
    key_name          TEXT NOT NULL,
    requested_model   TEXT,
    served_model      TEXT,
    provider          TEXT NOT NULL,       -- 'none' when every provider failed
    fallback          BOOLEAN NOT NULL,    -- served by something other than the primary
    prompt_tokens     INT NOT NULL DEFAULT 0,
    completion_tokens INT NOT NULL DEFAULT 0,
    cost_usd          NUMERIC(12, 6),      -- NULL = unknown price
    latency_ms        INT NOT NULL,
    status            INT NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_key_ts ON usage (key_name, ts);
-- Day 8 (idempotent, safe on an existing table):
ALTER TABLE usage ADD COLUMN IF NOT EXISTS cache_hit BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE usage ADD COLUMN IF NOT EXISTS saved_usd NUMERIC(12, 6) NOT NULL DEFAULT 0;
"""

SUMMARY_SQL = """
SELECT key_name,
       provider,
       count(*)                                   AS requests,
       count(*) FILTER (WHERE fallback)           AS fallbacks,
       count(*) FILTER (WHERE status >= 400 AND status <> 499) AS errors,
       count(*) FILTER (WHERE status = 499)       AS client_aborts,
       count(*) FILTER (WHERE cache_hit)          AS cache_hits,
       coalesce(sum(saved_usd), 0)                AS saved_usd,
       coalesce(sum(prompt_tokens), 0)            AS prompt_tokens,
       coalesce(sum(completion_tokens), 0)        AS completion_tokens,
       coalesce(sum(cost_usd), 0)                 AS cost_usd,
       round(avg(latency_ms))                     AS avg_latency_ms
FROM usage
WHERE ts > now() - make_interval(days => $1)
GROUP BY key_name, provider
ORDER BY cost_usd DESC, requests DESC
"""

DAILY_SQL = """
SELECT (ts AT TIME ZONE 'UTC')::date                AS day,
       key_name,
       count(*)                                      AS requests,
       count(*) FILTER (WHERE cache_hit)             AS cache_hits,
       count(*) FILTER (WHERE status >= 400 AND status <> 499) AS errors,
       coalesce(sum(prompt_tokens + completion_tokens), 0)     AS tokens,
       coalesce(sum(cost_usd), 0)                    AS cost_usd,
       coalesce(sum(saved_usd), 0)                   AS saved_usd
FROM usage
WHERE ts > now() - make_interval(days => $1)
GROUP BY 1, 2
ORDER BY 1 DESC, cost_usd DESC, key_name
"""


@dataclass
class UsageRecord:
    key_name: str
    requested_model: str | None
    served_model: str | None
    provider: str
    fallback: bool
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    latency_ms: int
    status: int
    cache_hit: bool = False      # served from cache or coalesced onto another request
    saved_usd: float = 0.0       # what this request would have cost without the cache


class UsageStore:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str) -> "UsageStore":
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5, timeout=5)
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA)
        return cls(pool)

    async def record(self, r: UsageRecord) -> None:
        # Logging must never break a request: swallow and log DB errors.
        try:
            await self._pool.execute(
                """INSERT INTO usage (key_name, requested_model, served_model, provider,
                       fallback, prompt_tokens, completion_tokens, cost_usd, latency_ms, status,
                       cache_hit, saved_usd)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)""",
                r.key_name, r.requested_model, r.served_model, r.provider, r.fallback,
                r.prompt_tokens, r.completion_tokens, r.cost_usd, r.latency_ms, r.status,
                r.cache_hit, r.saved_usd,
            )
        except Exception as e:  # noqa: BLE001
            log.error("failed to record usage: %r", e)

    async def summary(self, days: int) -> list[dict]:
        rows = await self._pool.fetch(SUMMARY_SQL, days)
        return [{**dict(r), "cost_usd": float(r["cost_usd"]), "saved_usd": float(r["saved_usd"])}
                for r in rows]

    async def daily(self, days: int) -> list[dict]:
        """Per key, per UTC day: requests, cache hits, errors, tokens, cost, savings."""
        rows = await self._pool.fetch(DAILY_SQL, days)
        return [{**dict(r), "day": r["day"].isoformat(), "cost_usd": float(r["cost_usd"]),
                 "saved_usd": float(r["saved_usd"])} for r in rows]

    async def close(self) -> None:
        await self._pool.close()
