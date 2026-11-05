"""Token-bucket rate limiter backed by Redis.

The whole check-and-consume runs inside one Lua script, so it is atomic:
two gateway instances can't both read "1 token left" and both spend it.
Time comes from Redis (TIME), not the app server, so clock skew between
gateway instances doesn't matter.
"""
import redis.asyncio as redis

LUA_TOKEN_BUCKET = """
local key          = KEYS[1]
local capacity     = tonumber(ARGV[1])
local refill_per_s = tonumber(ARGV[2])
local cost         = tonumber(ARGV[3])

local t = redis.call('TIME')
local now_ms = t[1] * 1000 + math.floor(t[2] / 1000)

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then
  tokens = capacity
  ts = now_ms
end

-- refill based on elapsed time, capped at capacity
local elapsed = math.max(0, now_ms - ts)
tokens = math.min(capacity, tokens + elapsed * refill_per_s / 1000)

local allowed = 0
local retry_after_ms = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_after_ms = math.ceil((cost - tokens) / refill_per_s * 1000)
end

redis.call('HSET', key, 'tokens', tostring(tokens), 'ts', now_ms)
-- expire idle buckets once they'd be full again
redis.call('PEXPIRE', key, math.ceil(capacity / refill_per_s * 1000) + 1000)

return {allowed, tostring(tokens), retry_after_ms}
"""


class RateLimitResult:
    __slots__ = ("allowed", "remaining", "retry_after_ms")

    def __init__(self, allowed: bool, remaining: float, retry_after_ms: int):
        self.allowed = allowed
        self.remaining = remaining
        self.retry_after_ms = retry_after_ms


class TokenBucketLimiter:
    def __init__(self, client: redis.Redis):
        self._script = client.register_script(LUA_TOKEN_BUCKET)

    async def check(
        self, bucket: str, capacity: int, per_minute: int, cost: int = 1
    ) -> RateLimitResult:
        """Consume `cost` tokens from `bucket`.

        capacity   - burst size
        per_minute - sustained refill rate
        cost       - 1 for request limits; use estimated tokens for TPM limits
        """
        allowed, remaining, retry_ms = await self._script(
            keys=[f"rl:{bucket}"], args=[capacity, per_minute / 60.0, cost]
        )
        return RateLimitResult(bool(allowed), float(remaining), int(retry_ms))
