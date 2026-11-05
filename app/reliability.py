"""Day 3: retries with backoff + jitter, circuit breaker, provider fallback.

Everything here takes injectable `clock` / `sleep` so tests run instantly.
State is per-process (one breaker per provider per gateway instance).
Moving breaker state to Redis is a possible later upgrade.
"""
import asyncio
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from .providers.base import Provider, ProviderError


class CircuitBreaker:
    """closed -> (N consecutive failures) -> open -> (timeout) -> half_open
    half_open lets ONE probe through: success -> closed, failure -> open."""

    def __init__(self, failure_threshold: int = 5, reset_timeout_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._clock = clock
        self.state = "closed"
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    def allow(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if self._clock() - self._opened_at >= self.reset_timeout_s:
                self.state = "half_open"
                self._probe_in_flight = False
            else:
                return False
        # half_open: only one probe at a time
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def record_success(self) -> None:
        self.state = "closed"
        self._failures = 0
        self._probe_in_flight = False

    def record_failure(self) -> None:
        self._probe_in_flight = False
        if self.state == "half_open":
            self._trip()
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self.state = "open"
        self._opened_at = self._clock()


class AllProvidersFailed(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class RetryPolicy:
    max_attempts: int = 3          # total tries per provider
    base_delay_s: float = 0.2
    max_delay_s: float = 2.0


def backoff_delay(attempt: int, policy: RetryPolicy) -> float:
    """Full jitter: uniform(0, min(cap, base * 2^attempt)).
    Jitter stops many clients from retrying in lockstep (retry storms)."""
    return random.uniform(0, min(policy.max_delay_s, policy.base_delay_s * 2**attempt))


class ReliableRouter:
    def __init__(
        self,
        providers: list[Provider],
        policy: RetryPolicy | None = None,
        breaker_factory: Callable[[], CircuitBreaker] = CircuitBreaker,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.providers = providers
        self.policy = policy or RetryPolicy()
        self.breakers = {p.name: breaker_factory() for p in providers}
        self._sleep = sleep

    async def chat(self, payload: dict) -> tuple[str, dict]:
        """Returns (provider_name, response). Tries providers in order."""
        errors: list[str] = []
        for provider in self.providers:
            breaker = self.breakers[provider.name]
            if not breaker.allow():
                errors.append(f"{provider.name}: circuit open")
                continue
            try:
                return provider.name, await self._with_retries(provider, breaker, payload)
            except ProviderError as e:
                errors.append(str(e))
        raise AllProvidersFailed(errors)

    async def _with_retries(self, provider: Provider, breaker: CircuitBreaker,
                            payload: dict) -> dict:
        last: ProviderError | None = None
        for attempt in range(self.policy.max_attempts):
            try:
                result = await provider.chat(payload)
                breaker.record_success()
                return result
            except ProviderError as e:   # only retryable errors are ProviderError;
                last = e                 # client 4xx raise HTTPException and pass through
                breaker.record_failure()
                if breaker.state == "open":
                    break                # tripped mid-retry: stop hammering it
                if attempt < self.policy.max_attempts - 1:
                    await self._sleep(backoff_delay(attempt, self.policy))
        assert last is not None
        raise last
