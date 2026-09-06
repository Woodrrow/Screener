from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from screener.data.cache import DiskCache


class FrozenClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


def test_key_is_order_independent_but_value_sensitive() -> None:
    a = DiskCache.make_key(
        "/coins/markets", {"page": 1, "vs_currency": "usd"}, date_bucket="2025-03-04"
    )
    b = DiskCache.make_key(
        "/coins/markets", {"vs_currency": "usd", "page": 1}, date_bucket="2025-03-04"
    )
    assert a == b

    other_day = DiskCache.make_key(
        "/coins/markets", {"page": 1, "vs_currency": "usd"}, date_bucket="2025-03-05"
    )
    other_page = DiskCache.make_key(
        "/coins/markets", {"page": 2, "vs_currency": "usd"}, date_bucket="2025-03-04"
    )
    assert len({a, other_day, other_page}) == 3


def test_hit_and_miss_are_counted(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    key = DiskCache.make_key("/x", {}, date_bucket="2025-03-04")

    assert cache.get(key) is None
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0

    cache.set(key, {"value": 42})
    assert cache.get(key) == {"value": 42}
    assert cache.stats.hits == 1
    assert cache.stats.writes == 1


def test_entries_expire(tmp_path: Path) -> None:
    clock = FrozenClock(datetime(2025, 3, 4, 12, 0, tzinfo=UTC))
    cache = DiskCache(tmp_path, now=clock)
    key = DiskCache.make_key("/x", {}, date_bucket="2025-03-04")
    cache.set(key, "payload", ttl_seconds=60)

    clock.now += timedelta(seconds=59)
    assert cache.get(key) == "payload"

    clock.now += timedelta(seconds=2)
    assert cache.get(key) is None
    assert cache.stats.expired == 1


def test_no_ttl_never_expires(tmp_path: Path) -> None:
    clock = FrozenClock(datetime(2025, 3, 4, tzinfo=UTC))
    cache = DiskCache(tmp_path, now=clock)
    key = DiskCache.make_key("/x", {}, date_bucket="2025-03-04")
    cache.set(key, "payload", ttl_seconds=None)

    clock.now += timedelta(days=3650)
    assert cache.get(key) == "payload"


def test_corrupt_entry_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    key = DiskCache.make_key("/x", {}, date_bucket="2025-03-04")
    cache.set(key, "payload")
    path = tmp_path / key[:2] / f"{key}.json"
    path.write_text("{not json")

    assert cache.get(key) is None


def test_writes_leave_no_temp_files(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    for index in range(5):
        cache.set(DiskCache.make_key(f"/x{index}", {}, date_bucket="2025-03-04"), index)
    assert not list(tmp_path.rglob("*.tmp"))
    assert cache.clear() == 5
