import pytest

from app.providers.base import ProviderError
from app.reliability import (AllProvidersFailed, CircuitBreaker, ReliableRouter,
                             RetryPolicy)


class FakeProvider:
    """Fails the first `fail_times` calls (or forever if None), then succeeds."""
    def __init__(self, name, fail_times=0):
        self.name, self.fail_times, self.calls = name, fail_times, 0

    async def chat(self, payload):
        self.calls += 1
        if self.fail_times is None or self.calls <= self.fail_times:
            raise ProviderError(self.name, 503, "boom")
        return {"from": self.name}


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


async def no_sleep(_): pass


def router(providers, clock=None, attempts=3, threshold=5):
    clock = clock or Clock()
    return ReliableRouter(
        providers, RetryPolicy(max_attempts=attempts),
        breaker_factory=lambda: CircuitBreaker(threshold, 30, clock), sleep=no_sleep), clock


async def test_retries_then_succeeds():
    p = FakeProvider("a", fail_times=2)
    r, _ = router([p])
    assert await r.chat({}) == ("a", {"from": "a"})
    assert p.calls == 3


async def test_falls_back_when_primary_exhausted():
    a, b = FakeProvider("a", None), FakeProvider("b")
    r, _ = router([a, b], threshold=100)
    assert (await r.chat({}))[0] == "b"
    assert a.calls == 3 and b.calls == 1


async def test_all_fail_raises():
    r, _ = router([FakeProvider("a", None), FakeProvider("b", None)], threshold=100)
    with pytest.raises(AllProvidersFailed):
        await r.chat({})


async def test_breaker_opens_and_skips_provider():
    a, b = FakeProvider("a", None), FakeProvider("b")
    r, _ = router([a, b], threshold=3)
    await r.chat({})                    # a fails 3x, trips; falls to b
    assert r.breakers["a"].state == "open"
    calls_before = a.calls
    await r.chat({})                    # a must be skipped entirely
    assert a.calls == calls_before


async def test_half_open_probe_recovers():
    a, b = FakeProvider("a", fail_times=3), FakeProvider("b")
    r, clock = router([a, b], threshold=3)
    await r.chat({})                    # a trips (3 failures), b serves
    clock.t += 31                       # past reset timeout -> half_open
    assert (await r.chat({}))[0] == "a" # probe succeeds (a recovered)
    assert r.breakers["a"].state == "closed"


async def test_half_open_probe_failure_reopens():
    a, b = FakeProvider("a", None), FakeProvider("b")
    r, clock = router([a, b], threshold=3)
    await r.chat({})
    clock.t += 31
    await r.chat({})                    # probe fails -> open again
    assert r.breakers["a"].state == "open"
    assert a.calls == 4                 # 3 originally + exactly 1 probe


async def test_client_errors_are_not_retried():
    from fastapi import HTTPException

    class BadRequest:
        name, calls = "a", 0
        async def chat(self, payload):
            self.calls += 1
            raise HTTPException(400, "bad")

    p = BadRequest()
    r, _ = router([p, FakeProvider("b")])
    with pytest.raises(HTTPException):
        await r.chat({})
    assert p.calls == 1
