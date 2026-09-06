from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from screener.data.history import HistoryStore
from screener.paths import Layout


class RecordingSource:
    """A DataSource that serves a synthetic series and records what was asked for."""

    def __init__(self, start: date, days: int, base_price: float = 100.0) -> None:
        self.start = start
        self.days = days
        self.base_price = base_price
        self.requests: list[tuple[str, int]] = []

    @property
    def name(self) -> str:
        return "recording"

    def fetch_universe(self, *, top_n: int, vs_currency: str = "usd") -> pd.DataFrame:
        raise NotImplementedError

    def fetch_price_history(
        self, coin_id: str, *, days: int, vs_currency: str = "usd"
    ) -> pd.DataFrame:
        self.requests.append((coin_id, days))
        # Trailing `days` calendar days ending today inclusive, like the API.
        end = self.start + timedelta(days=self.days)
        first = end - timedelta(days=days)
        rows = []
        cursor = first
        while cursor <= end:
            offset = (cursor - self.start).days
            rows.append(
                {
                    "date": cursor,
                    "price": self.base_price + offset,
                    "market_cap": (self.base_price + offset) * 1_000,
                    "volume": 50.0 + offset,
                }
            )
            cursor += timedelta(days=1)
        return pd.DataFrame(rows)


def test_first_update_fetches_the_full_window(layout: Layout) -> None:
    start = date(2025, 1, 1)
    source = RecordingSource(start, days=30)
    store = HistoryStore(layout.history)
    today = start + timedelta(days=30)

    update = store.update(source, "bitcoin", days=30, today=today)

    assert source.requests == [("bitcoin", 30)]
    assert update.rows_before == 0
    # The in-progress day is dropped, so the newest stored bar is yesterday.
    assert store.last_date("bitcoin") == today - timedelta(days=1)


def test_second_update_only_fetches_the_gap(layout: Layout) -> None:
    start = date(2025, 1, 1)
    store = HistoryStore(layout.history)
    first_today = start + timedelta(days=30)
    store.update(RecordingSource(start, days=30), "bitcoin", days=365, today=first_today)
    rows_after_first = len(store.read("bitcoin"))

    later = first_today + timedelta(days=7)
    source = RecordingSource(start, days=37)
    update = store.update(source, "bitcoin", days=365, today=later)

    # The newest stored bar is `first_today - 1`, so the gap to `later` is 8
    # days; +2 for the partial day and one bar of overlap. Not 365.
    assert source.requests == [("bitcoin", 10)]
    assert update.rows_before == rows_after_first
    assert update.rows_added == 7
    assert store.last_date("bitcoin") == later - timedelta(days=1)


def test_update_is_skipped_when_already_current(layout: Layout) -> None:
    start = date(2025, 1, 1)
    store = HistoryStore(layout.history)
    today = start + timedelta(days=30)
    store.update(RecordingSource(start, days=30), "bitcoin", days=90, today=today)

    source = RecordingSource(start, days=30)
    update = store.update(source, "bitcoin", days=90, today=today - timedelta(days=1))

    assert update.skipped
    assert source.requests == []


def test_partial_current_day_is_never_stored(layout: Layout) -> None:
    start = date(2025, 1, 1)
    store = HistoryStore(layout.history)
    today = start + timedelta(days=10)
    store.update(RecordingSource(start, days=10), "bitcoin", days=30, today=today)

    assert today not in store.read("bitcoin")["date"].tolist()


def test_merge_prefers_the_newer_observation(layout: Layout) -> None:
    store = HistoryStore(layout.history)
    stored = pd.DataFrame(
        [
            {"date": date(2025, 1, 1), "price": 10.0, "market_cap": 1.0, "volume": 1.0},
            {"date": date(2025, 1, 2), "price": 11.0, "market_cap": 1.0, "volume": 1.0},
        ]
    )
    store.write("bitcoin", stored)

    revised = pd.DataFrame(
        [
            {"date": date(2025, 1, 2), "price": 11.5, "market_cap": 2.0, "volume": 2.0},
            {"date": date(2025, 1, 3), "price": 12.0, "market_cap": 3.0, "volume": 3.0},
        ]
    )
    merged = store.merge("bitcoin", revised)

    assert merged["date"].tolist() == [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]
    assert merged["price"].tolist() == [10.0, 11.5, 12.0]


def test_history_only_grows(layout: Layout) -> None:
    """A later narrow-window run must not destroy an earlier wide one."""
    start = date(2025, 1, 1)
    store = HistoryStore(layout.history)
    wide_today = start + timedelta(days=200)
    store.update(RecordingSource(start, days=200), "bitcoin", days=200, today=wide_today)
    wide_rows = len(store.read("bitcoin"))

    later = wide_today + timedelta(days=3)
    store.update(RecordingSource(start, days=203), "bitcoin", days=7, today=later)

    assert len(store.read("bitcoin")) >= wide_rows


def test_price_panel_is_sorted_and_column_ordered(layout: Layout) -> None:
    store = HistoryStore(layout.history)
    for coin_id, price in (("solana", 20.0), ("bitcoin", 10.0)):
        store.write(
            coin_id,
            pd.DataFrame(
                [
                    {
                        "date": date(2025, 1, 2),
                        "price": price + 1,
                        "market_cap": 1.0,
                        "volume": 1.0,
                    },
                    {"date": date(2025, 1, 1), "price": price, "market_cap": 1.0, "volume": 1.0},
                ]
            ),
        )

    panel = store.price_panel(["solana", "bitcoin", "solana"])

    assert list(panel.columns) == ["bitcoin", "solana"]
    assert panel.index.tolist() == [date(2025, 1, 1), date(2025, 1, 2)]
    assert panel["bitcoin"].tolist() == [10.0, 11.0]


def test_coin_ids_cannot_escape_the_store_directory(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    assert store.path_for("../../etc/passwd").parent == tmp_path / "history"
