"""Token-bucket rate limiter.

CoinGecko's free tier allows roughly 10-30 calls/minute and answers 429 rather
than queueing, so the client has to pace itself. A token bucket rather than a
fixed inter-call sleep lets a burst of cache misses go out back-to-back up to
`capacity` and then settle to the sustained rate, which is what the API actually
tolerates, while still bounding calls over any long window.

The clock and sleep function are injected so tests can drive the bucket with
virtual time instead of wall-clock sleeps.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class TokenBucket:
    def __init__(
        self,
        *,
        rate_per_minute: float,
        capacity: float | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        resolved_capacity = float(rate_per_minute) if capacity is None else float(capacity)
        if resolved_capacity <= 0:
            raise ValueError("capacity must be positive")

        self._refill_per_second = float(rate_per_minute) / 60.0
        self._capacity = resolved_capacity
        self._tokens = resolved_capacity
        self._monotonic = monotonic
        self._sleep = sleep
        self._updated_at = monotonic()
        self._lock = threading.Lock()

    @property
    def capacity(self) -> float:
        return self._capacity

    @property
    def tokens(self) -> float:
        """Current token count, refilled to now. Primarily for tests and diagnostics."""
        with self._lock:
            self._refill()
            return self._tokens

    def _refill(self) -> None:
        now = self._monotonic()
        elapsed = now - self._updated_at
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_second)
        self._updated_at = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        if tokens > self._capacity:
            raise ValueError(f"cannot acquire {tokens} tokens from a bucket of {self._capacity}")
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until `tokens` are available. Returns the seconds spent waiting."""
        if tokens > self._capacity:
            raise ValueError(f"cannot acquire {tokens} tokens from a bucket of {self._capacity}")
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                wait_for = (tokens - self._tokens) / self._refill_per_second
            # Sleep outside the lock. Another thread may take the tokens we were
            # waiting on, so the loop re-checks rather than assuming success.
            self._sleep(wait_for)
            waited += wait_for
