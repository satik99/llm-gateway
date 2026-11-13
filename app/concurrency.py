"""Per-key cap on requests being served at once (protects the upstream, and other teams, from
one client that opens hundreds of slow connections). Counted per process."""
from collections import defaultdict


class Slot:
    def __init__(self, limiter: "ConcurrencyLimiter | None", key: str):
        self._limiter, self._key, self._held = limiter, key, limiter is not None
        self.handed_off = False      # a streaming response now owns the release

    def hand_off(self) -> None:
        self.handed_off = True

    def release(self) -> None:   # idempotent: safe to call from several cleanup paths
        if self._held:
            self._held = False
            self._limiter._n[self._key] -= 1


class ConcurrencyLimiter:
    def __init__(self, cap: int):
        self.cap = cap               # <= 0 disables the limit
        self._n: dict[str, int] = defaultdict(int)

    def acquire(self, key: str) -> Slot | None:
        """A Slot, or None when `key` is already at the cap."""
        if self.cap <= 0:
            return Slot(None, key)
        if self._n[key] >= self.cap:
            return None
        self._n[key] += 1
        return Slot(self, key)
