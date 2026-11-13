"""API keys in Postgres, managed through /admin/keys.

* Only a SHA-256 hash of each key is stored (keys are 192-bit random tokens, so a fast hash is
  safe). The plaintext is returned exactly once, at creation or rotation.
* Lookups are cached in-process for KEY_CACHE_TTL_S seconds so authentication does not hit
  Postgres on every request. Admin changes clear the cache of the process that served them;
  other worker processes pick changes up within the TTL (worst-case revoke delay = the TTL).
* If Postgres is unreachable, an already-cached key keeps working (stale) and unknown keys get
  503 instead of 401, so clients know to retry rather than think their key is wrong.
"""
import asyncio
import hashlib
import logging
import re
import secrets
import time

import asyncpg

from .auth import ApiKey

log = logging.getLogger("gateway.keystore")

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id                 BIGSERIAL PRIMARY KEY,
    name               TEXT NOT NULL UNIQUE,
    key_hash           TEXT NOT NULL UNIQUE,
    key_prefix         TEXT NOT NULL,
    rpm                INT  NOT NULL CHECK (rpm > 0),
    tpm                INT  NOT NULL DEFAULT 0 CHECK (tpm >= 0),
    monthly_budget_usd NUMERIC(12, 4) NOT NULL DEFAULT 0 CHECK (monthly_budget_usd >= 0),
    active             BOOLEAN NOT NULL DEFAULT true,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    rotated_at         TIMESTAMPTZ,
    revoked_at         TIMESTAMPTZ
);
"""

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
PUBLIC_COLS = "name, key_prefix, rpm, tpm, monthly_budget_usd, active, created_at, rotated_at, revoked_at"


class KeyStoreUnavailable(Exception):
    """Postgres could not be reached and there is no cached answer."""


class NameTaken(Exception):
    pass


class NotFound(Exception):
    pass


def hash_key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_token() -> str:
    return "gw-" + secrets.token_urlsafe(24)


def _api_key(row) -> ApiKey:
    return ApiKey(name=row["name"], key=row["key_prefix"], rpm=row["rpm"], tpm=row["tpm"],
                  monthly_budget_usd=float(row["monthly_budget_usd"]))


def _public(row) -> dict:
    d = dict(row)
    d["monthly_budget_usd"] = float(d["monthly_budget_usd"])
    for k in ("created_at", "rotated_at", "revoked_at"):
        d[k] = d[k].isoformat() if d[k] else None
    return d


class KeyStore:
    def __init__(self, pool: asyncpg.Pool, cache_ttl_s: float = 10.0):
        self._pool = pool
        self._ttl = cache_ttl_s
        self._cache: dict[str, tuple[float, ApiKey | None]] = {}

    @classmethod
    async def connect(cls, dsn: str, cache_ttl_s: float = 10.0) -> "KeyStore":
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
        await pool.execute(SCHEMA)
        return cls(pool, cache_ttl_s)

    async def close(self) -> None:
        await self._pool.close()

    def invalidate(self) -> None:
        self._cache.clear()

    # ---- authentication path ----
    async def lookup(self, token: str) -> ApiKey | None:
        h, now = hash_key(token), time.monotonic()
        hit = self._cache.get(h)
        if hit and hit[0] > now:
            return hit[1]
        try:
            row = await self._pool.fetchrow(
                "SELECT name, key_prefix, rpm, tpm, monthly_budget_usd FROM api_keys "
                "WHERE key_hash = $1 AND active", h)
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, asyncio.TimeoutError) as e:
            log.error("key lookup failed: %r", e)
            if hit:                                  # stale beats down
                return hit[1]
            raise KeyStoreUnavailable() from e
        key = _api_key(row) if row else None
        if len(self._cache) > 10_000:                # bound memory against junk-token floods
            self._cache.clear()
        self._cache[h] = (now + self._ttl, key)
        return key

    # ---- admin operations ----
    async def create(self, name: str, rpm: int, tpm: int, budget: float) -> tuple[dict, str]:
        token = new_token()
        try:
            row = await self._pool.fetchrow(
                f"INSERT INTO api_keys (name, key_hash, key_prefix, rpm, tpm, monthly_budget_usd) "
                f"VALUES ($1,$2,$3,$4,$5,$6) RETURNING {PUBLIC_COLS}",
                name, hash_key(token), token[:7], rpm, tpm, budget)
        except asyncpg.UniqueViolationError as e:
            raise NameTaken(name) from e
        self.invalidate()
        return _public(row), token

    async def list(self) -> list[dict]:
        rows = await self._pool.fetch(f"SELECT {PUBLIC_COLS} FROM api_keys ORDER BY name")
        return [_public(r) for r in rows]

    async def get(self, name: str) -> dict:
        row = await self._pool.fetchrow(f"SELECT {PUBLIC_COLS} FROM api_keys WHERE name = $1", name)
        if row is None:
            raise NotFound(name)
        return _public(row)

    async def update(self, name: str, changes: dict) -> dict:
        allowed = {"rpm", "tpm", "monthly_budget_usd", "active"}
        sets, args = [], []
        for k, v in changes.items():
            if k in allowed and v is not None:
                args.append(v)
                sets.append(f"{k} = ${len(args)}")
        if not sets:
            return await self.get(name)
        if changes.get("active") is False:
            sets.append("revoked_at = now()")
        elif changes.get("active") is True:
            sets.append("revoked_at = NULL")
        args.append(name)
        row = await self._pool.fetchrow(
            f"UPDATE api_keys SET {', '.join(sets)} WHERE name = ${len(args)} RETURNING {PUBLIC_COLS}",
            *args)
        if row is None:
            raise NotFound(name)
        self.invalidate()
        return _public(row)

    async def revoke(self, name: str) -> dict:
        return await self.update(name, {"active": False})

    async def rotate(self, name: str) -> tuple[dict, str]:
        token = new_token()
        row = await self._pool.fetchrow(
            f"UPDATE api_keys SET key_hash = $1, key_prefix = $2, rotated_at = now() "
            f"WHERE name = $3 AND active RETURNING {PUBLIC_COLS}", hash_key(token), token[:7], name)
        if row is None:
            raise NotFound(name)                      # unknown, or revoked (re-activate first)
        self.invalidate()
        return _public(row), token
