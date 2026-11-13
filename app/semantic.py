"""Semantic cache: serve a stored answer when a NEW question is close enough in meaning to an
OLD one ("what's your refund policy?" ~ "how do I get my money back?").

Safety rules (a wrong semantic hit is a wrong answer, so these are strict):
  * a hit is only possible inside the same BUCKET = same api key + same everything-except-the-
    last-user-message (model, temperature, max_tokens, system prompt, earlier turns). So a
    different system prompt or conversation history can never match;
  * similarity must reach SEMANTIC_THRESHOLD (default 0.92); tune it with tools/eval_semantic.py;
  * only the last user message is embedded; non-text content opts out;
  * fallback-provider answers are never stored (same rule as the exact cache).

Storage is Redis: per bucket one hash of float32 vectors, one hash of responses, one sorted set
of ages for eviction. Lookup is a brute-force cosine over at most `max_entries` vectors (numpy).
"""
import base64
import hashlib
import json
import re
import time

import numpy as np
import redis.asyncio as redis

from .cache import IGNORED_FIELDS


def query_text(payload: dict) -> str | None:
    """The text to embed: the last message if it is a plain-text user message."""
    msgs = payload.get("messages") or []
    if not msgs or msgs[-1].get("role") != "user":
        return None
    c = msgs[-1].get("content")
    if isinstance(c, str):
        return c.strip() or None
    if isinstance(c, list) and all(isinstance(p, dict) and p.get("type") == "text" for p in c):
        return " ".join(p.get("text", "") for p in c).strip() or None
    return None                      # images, tool results, ...: not eligible


def bucket_key(scope: str, payload: dict) -> str:
    material = {k: v for k, v in payload.items() if k not in IGNORED_FIELDS | {"messages"}}
    material["_prefix"] = (payload.get("messages") or [])[:-1]
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{scope}:{hashlib.sha256(blob.encode()).hexdigest()[:32]}"


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9$%]+", s.lower())


def wording_conflict(a: str, b: str, max_overlap: float) -> bool:
    """Minimal-pair guard. Embeddings rate "enable X" / "disable X" or "10 km to miles" /
    "10 miles to km" as near-identical, yet the answers differ. A real paraphrase uses
    different words; a near-miss reuses almost all of them and changes one that matters.
    So: if two questions share >= max_overlap of their words (Jaccard) but are not the same
    word sequence, do NOT treat them as the same question. Costs a few safe paraphrases."""
    wa, wb = _words(a), _words(b)
    if wa == wb:
        return False
    sa, sb = set(wa), set(wb)
    return bool(sa | sb) and len(sa & sb) / len(sa | sb) >= max_overlap


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _pack(v: np.ndarray) -> str:
    return base64.b64encode(v.astype(np.float32).tobytes()).decode()


def _unpack(s) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)


class SemanticCache:
    def __init__(self, client: redis.Redis, threshold: float = 0.92, ttl_s: int = 3600,
                 max_entries: int = 500, max_bytes: int = 262_144,
                 max_wording_overlap: float = 0.4):
        self._r, self.threshold, self._ttl = client, threshold, ttl_s
        self._max_entries, self._max_bytes = max_entries, max_bytes
        self._max_overlap = max_wording_overlap      # > 1 disables the guard

    async def lookup(self, bucket: str, vec, query: str = "") -> tuple[dict, float] | None:
        """Best stored answer in this bucket with similarity >= threshold that also passes the
        wording guard, else None."""
        raw = await self._r.hgetall(f"semv:{bucket}")
        if not raw:
            return None
        ids = list(raw)
        mat = np.stack([_unpack(raw[i]) for i in ids])
        sims = mat @ _unit(vec)
        for i in np.argsort(-sims)[:5]:
            if float(sims[i]) < self.threshold:
                return None
            blob = await self._r.hget(f"semb:{bucket}", ids[i])
            if not blob:
                continue
            entry = json.loads(blob)
            if query and wording_conflict(query, entry.get("query", ""), self._max_overlap):
                continue
            return entry, float(sims[i])
        return None

    async def store(self, bucket: str, vec, query: str, provider: str, body: dict) -> bool:
        if not body.get("choices"):
            return False
        blob = json.dumps({"provider": provider, "body": body, "query": query})
        if len(blob) > self._max_bytes:
            return False
        eid = hashlib.sha1(query.encode()).hexdigest()
        v, b, t = f"semv:{bucket}", f"semb:{bucket}", f"semt:{bucket}"
        pipe = self._r.pipeline()
        pipe.hset(v, eid, _pack(_unit(vec)))
        pipe.hset(b, eid, blob)
        pipe.zadd(t, {eid: time.time()})
        for k in (v, b, t):
            pipe.expire(k, self._ttl)
        await pipe.execute()
        over = await self._r.zcard(t) - self._max_entries
        if over > 0:                                   # evict the oldest entries
            old = [m for m, _ in await self._r.zpopmin(t, over)]
            await self._r.hdel(v, *old)
            await self._r.hdel(b, *old)
        return True

    async def flush(self, scope: str | None = None) -> int:
        deleted = 0
        for prefix in ("semv", "semb", "semt"):
            async for k in self._r.scan_iter(match=f"{prefix}:{scope or ''}*", count=500):
                deleted += await self._r.delete(k)
        return deleted
