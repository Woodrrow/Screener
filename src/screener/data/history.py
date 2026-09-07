"""Per-coin daily price history.

Kept separate from snapshots because it serves a different purpose: snapshots are
the point-in-time universe, history is the price series used for realised
volatility, backtest marks and benchmark curves. Appending incrementally matters
on a 10-30 call/minute budget - refetching a full 365-day window for 250 coins
every run would take the better part of an hour.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from screener.data.schema import HISTORY_COLUMNS, validate_history
from screener.data.source import DataSource


@dataclass(frozen=True)
class HistoryUpdate:
    coin_id: str
    rows_before: int
    rows_after: int
    fetched_days: int
    skipped: bool

    @property
    def rows_added(self) -> int:
        return self.rows_after - self.rows_before


class HistoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, coin_id: str) -> Path:
        # Coin ids are lowercase slugs from the API, but a hostile or odd id
        # must never escape the store directory.
        safe = coin_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self.root / f"{safe}.parquet"

    def exists(self, coin_id: str) -> bool:
        return self.path_for(coin_id).exists()

    def available_coins(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.parquet"))

    def read(self, coin_id: str) -> pd.DataFrame:
        path = self.path_for(coin_id)
        if not path.exists():
            return pd.DataFrame(columns=list(HISTORY_COLUMNS))
        frame = pd.read_parquet(path)
        return _normalise(frame)

    def last_date(self, coin_id: str) -> date | None:
        frame = self.read(coin_id)
        if frame.empty:
            return None
        value = frame["date"].iloc[-1]
        return value if isinstance(value, date) else pd.Timestamp(value).date()

    def write(self, coin_id: str, frame: pd.DataFrame) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(coin_id)
        out = _normalise(frame)
        validate_history(out)
        out.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        return path

    def merge(self, coin_id: str, incoming: pd.DataFrame) -> pd.DataFrame:
        """Union stored and incoming rows, preferring incoming on a date clash.

        A later fetch of the same day is the more settled figure (CoinGecko
        revises the current day as it accrues), so incoming wins.
        """
        existing = self.read(coin_id)
        combined = pd.concat([existing, _normalise(incoming)], ignore_index=True)
        combined = combined.drop_duplicates(subset="date", keep="last")
        return _normalise(combined)

    def update(
        self,
        source: DataSource,
        coin_id: str,
        *,
        days: int,
        today: date,
        force_full: bool = False,
    ) -> HistoryUpdate:
        existing = self.read(coin_id)
        rows_before = len(existing)
        last = self.last_date(coin_id)

        if force_full or last is None:
            fetch_days = days
        else:
            gap = (today - last).days
            if gap <= 0:
                return HistoryUpdate(coin_id, rows_before, rows_before, 0, skipped=True)
            # +2 covers the partial current day plus one day of overlap, so a
            # revised final bar is corrected rather than frozen at its first value.
            fetch_days = min(days, gap + 2)

        incoming = source.fetch_price_history(coin_id, days=fetch_days)
        incoming = _drop_partial_day(incoming, today)
        # The store only ever grows. Trimming to the requested window would
        # mean a later `--days 30` run silently destroyed a year of history.
        merged = self.merge(coin_id, incoming)
        self.write(coin_id, merged)
        return HistoryUpdate(coin_id, rows_before, len(merged), fetch_days, skipped=False)

    def panel(self, coin_ids: list[str], column: str = "price") -> pd.DataFrame:
        """Wide date x coin_id frame for one stored column.

        Sorted on both axes explicitly, so a panel never depends on filesystem
        listing order or on the order coin ids were passed in.
        """
        if column not in HISTORY_COLUMNS or column == "date":
            raise ValueError(f"{column!r} is not a stored history column")
        columns: dict[str, pd.Series[float]] = {}
        for coin_id in sorted(set(coin_ids)):
            frame = self.read(coin_id)
            if frame.empty:
                continue
            series = frame.set_index("date")[column]
            columns[coin_id] = series[~series.index.duplicated(keep="last")]
        if not columns:
            return pd.DataFrame()
        wide = pd.DataFrame(columns)
        wide.index.name = "date"
        return wide.sort_index().loc[:, sorted(wide.columns)]

    def price_panel(self, coin_ids: list[str]) -> pd.DataFrame:
        return self.panel(coin_ids, "price")


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in HISTORY_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out = out.loc[:, list(HISTORY_COLUMNS)]
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.date
    for column in ("price", "market_cap", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out.drop_duplicates(subset="date", keep="last")
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


def _drop_partial_day(frame: pd.DataFrame, today: date) -> pd.DataFrame:
    """Discard the in-progress day.

    Storing a partial bar would let a mid-day close leak into a factor as if it
    were a settled daily close, and the next run would silently change history.
    """
    if frame.empty:
        return frame
    return frame.loc[frame["date"] < today].reset_index(drop=True)
