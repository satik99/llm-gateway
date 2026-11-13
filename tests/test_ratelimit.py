import asyncio

import fakeredis
import pytest
import pytest_asyncio

from app.ratelimit import TokenBucketLimiter


@pytest_asyncio.fixture
async def limiter():
    client = fakeredis.FakeAsyncRedis()
    yield TokenBucketLimiter(client)
    await client.aclose()


async def test_allows_up_to_capacity_then_blocks(limiter):
    results = [await limiter.check("k", capacity=3, per_minute=60) for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert results[-1].retry_after_ms > 0


async def test_refills_over_time(limiter):
    for _ in range(2):
        await limiter.check("k", capacity=2, per_minute=600)  # 10 tokens/sec
    assert not (await limiter.check("k", capacity=2, per_minute=600)).allowed
    await asyncio.sleep(0.25)  # ~2.5 tokens refilled, capped at 2
    assert (await limiter.check("k", capacity=2, per_minute=600)).allowed


async def test_buckets_are_independent(limiter):
    await limiter.check("a", capacity=1, per_minute=1)
    assert not (await limiter.check("a", capacity=1, per_minute=1)).allowed
    assert (await limiter.check("b", capacity=1, per_minute=1)).allowed


async def test_concurrent_requests_never_overspend(limiter):
    # 50 simultaneous requests against a bucket of 10: exactly 10 must pass.
    rs = await asyncio.gather(
        *[limiter.check("k", capacity=10, per_minute=1) for _ in range(50)]
    )
    assert sum(r.allowed for r in rs) == 10
