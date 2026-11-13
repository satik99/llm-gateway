"""Exact-match response cache (Redis) + single-flight de-duplication.

Policy (deliberately conservative: a wrong cache hit is worse than a miss):
  * only non-streaming requests;
  * only when the output is meant to be deterministic: temperature == 0,
    or the caller opts in with header `X-Cache-Mode: force`; `X-Cache-Mode: bypass` opts out;
  * the key is a hash of EVERYTHING that can change the output (model, messages, temperature,
    max_tokens, tools, ...). We drop only fields known not to matter (IGNORED_FIELDS), so an
    unknown new parameter makes keys differ (a miss) rather than risking a wrong hit;
  * keys are scoped per API key name, so one team never sees another team's answers;
  * answers served by a FALLBACK provider are never stored, otherwise a lower-quality answer
    from the outage would outlive the outage.
"""
import asyncio
import hashlib
import json

import redis.asyncio as redis

IGNORED_FIELDS = {"stream", "stream_options", "user", "metadata", "store"}


def cache_key(scope: str, payload: dict) -> str:
    material = {k: v for k, v in payload.items() if k not in IGNORED_FIELDS}
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"cache:{scope}:{hashlib.sha256(blob.encode()).hexdigest()}"


def is_cacheable(payload: dict, headers) -> bool:
    mode = (headers.get("x-cache-mode") or "").strip().lower()
    if mode in ("bypass", "no-store") or payload.get("stream"):
        return False
    if mode == "force":
        return True
    return payload.get("temperature") == 0


class ResponseCache:
    def __init__(self, client: redis.Redis, ttl_s: int = 3600, max_bytes: int = 262_144):
        self._r = client
        self._ttl = ttl_s
        self._max_bytes = max_bytes

    async def get(self, key: str) -> dict | None:
        raw = await self._r.get(key)
        return json.loads(raw) if raw else None

    async def set(self, key: str, provider: str, body: dict) -> bool:
        if not body.get("choices"):          # never cache anything that isn't a real completion
            return False
        blob = json.dumps({"provider": provider, "body": body})
        if len(blob) > self._max_bytes:
            return False
        await self._r.set(key, blob, ex=self._ttl)
        return True

    async def flush(self, scope: str | None = None) -> int:
        pattern = f"cache:{scope}:*" if scope else "cache:*"
        deleted = 0
        async for k in self._r.scan_iter(match=pattern, count=500):
            deleted += await self._r.delete(k)
        return deleted


class SingleFlight:
    """If N identical requests arrive while the first is still in flight, run it once and
    share the result. Stops a burst of identical cache misses from becoming N paid calls.

    The shared work runs in its own task: a waiter that disconnects (is cancelled) does not
    cancel the work for the others. In-process only; with several workers each one
    de-duplicates separately."""

    def __init__(self):
        self._inflight: dict[str, asyncio.Task] = {}

    async def do(self, key: str, fn):
        """Returns (result, was_leader). Exceptions from fn propagate to every waiter."""
        task = self._inflight.get(key)
        leader = task is None
        if leader:
            task = asyncio.ensure_future(fn())
            self._inflight[key] = task

            def _done(t: asyncio.Task):
                self._inflight.pop(key, None)
                if not t.cancelled():
                    t.exception()            # mark retrieved: avoids "never retrieved" warnings
            task.add_done_callback(_done)
        return await asyncio.shield(task), leader
