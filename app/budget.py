"""Monthly spend counter per key, in Redis (fast path). Postgres `usage` stays the
source of truth for reporting. Stored as integer micro-dollars so INCRBY stays exact.

Soft limit: concurrent in-flight requests can overshoot the budget slightly,
because spend is only known after a response comes back."""
from datetime import datetime, timezone

import redis.asyncio as redis

MICRO = 1_000_000
TTL_SECONDS = 40 * 24 * 3600


class SpendTracker:
    def __init__(self, client: redis.Redis):
        self._r = client

    @staticmethod
    def _key(name: str, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        return f"spend:{name}:{now:%Y-%m}"

    async def get(self, name: str) -> float:
        v = await self._r.get(self._key(name))
        return int(v) / MICRO if v else 0.0

    async def add(self, name: str, usd: float) -> None:
        micros = round(usd * MICRO)
        if micros <= 0:
            return
        k = self._key(name)
        pipe = self._r.pipeline()
        pipe.incrby(k, micros)
        pipe.expire(k, TTL_SECONDS)
        await pipe.execute()
