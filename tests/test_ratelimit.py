from __future__ import annotations

import threading
import time

import pytest

from screener.data.ratelimit import TokenBucket


class VirtualClock:
    """Deterministic monotonic clock. `sleep` advances time instead of waiting."""

    def __init__(self) -> None:
        self.now = 0.0
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.now += max(seconds, 0.0)


def test_burst_up_to_capacity_then_paces() -> None:
    clock = VirtualClock()
    bucket = TokenBucket(
        rate_per_minute=60, capacity=5, monotonic=clock.monotonic, sleep=clock.sleep
    )

    for _ in range(5):
        assert bucket.acquire() == 0.0
    assert clock.now == 0.0

    # Bucket is empty; at 60/min the sixth call waits exactly one second.
    waited = bucket.acquire()
    assert waited == pytest.approx(1.0)
    assert clock.now == pytest.approx(1.0)


def test_refills_at_the_configured_rate() -> None:
    clock = VirtualClock()
    bucket = TokenBucket(
        rate_per_minute=120, capacity=10, monotonic=clock.monotonic, sleep=clock.sleep
    )
    for _ in range(10):
        bucket.acquire()
    assert bucket.tokens == pytest.approx(0.0)

    clock.now += 2.5  # 120/min == 2/s, so 2.5s buys 5 tokens
    assert bucket.tokens == pytest.approx(5.0)

    clock.now += 1000.0  # never exceeds capacity
    assert bucket.tokens == pytest.approx(10.0)


def test_try_acquire_does_not_block() -> None:
    clock = VirtualClock()
    bucket = TokenBucket(
        rate_per_minute=60, capacity=2, monotonic=clock.monotonic, sleep=clock.sleep
    )
    assert bucket.try_acquire()
    assert bucket.try_acquire()
    assert not bucket.try_acquire()
    assert clock.now == 0.0


def test_rejects_impossible_request() -> None:
    bucket = TokenBucket(rate_per_minute=60, capacity=3)
    with pytest.raises(ValueError):
        bucket.acquire(4)


def test_budget_is_honoured_under_concurrency() -> None:
    """Real threads, real clock: N calls must take at least the implied time.

    The bucket starts full, so the first `capacity` calls are free and only the
    remainder is paced. Anything faster than that floor means the limiter is
    losing tokens to a race.
    """
    rate_per_minute = 600.0  # 10/s, keeps the test under a second
    capacity = 5
    calls_per_thread = 5
    threads = 6
    total_calls = calls_per_thread * threads

    bucket = TokenBucket(rate_per_minute=rate_per_minute, capacity=capacity)
    barrier = threading.Barrier(threads)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            for _ in range(calls_per_thread):
                bucket.acquire()
        except BaseException as exc:
            errors.append(exc)

    started = time.monotonic()
    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(timeout=30)
    elapsed = time.monotonic() - started

    assert not errors
    assert all(not thread.is_alive() for thread in workers)

    paced_calls = total_calls - capacity
    floor = paced_calls / (rate_per_minute / 60.0)
    # 5% tolerance for the clock read at the start of the first acquire.
    assert elapsed >= floor * 0.95, f"{total_calls} calls took {elapsed:.3f}s, floor {floor:.3f}s"
    assert bucket.tokens <= capacity
