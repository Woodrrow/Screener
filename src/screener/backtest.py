"""Reconstructing a point-in-time universe from the history store.

Read the caveat before the code. A backtest built this way is contaminated by
survivorship: the coins with stored history are the coins that were in a recent
snapshot, so the sample already knows which projects were still around. That
inflates returns in a way no amount of careful cost modelling repairs, which is
why `screener backtest` requires an explicit acknowledgement flag and stamps a
warning into everything it produces.

What is reconstructed honestly:

- prices, market caps and volumes are the stored daily observations at `as_of`;
- returns are computed from those prices, not read from a snapshot column that
  was measured at a different time;
- the all-time high is a running maximum over history up to `as_of`, never the
  snapshot's `ath` field, which is today's high and would let a 2024 rebalance
  measure drawdown against a peak that had not happened yet.

What is not available, and is therefore switched off rather than faked: supply.
CoinGecko's free tier serves no point-in-time circulating or max supply, and
today's figures would understate past dilution - flattering exactly the coins
that were emitting hardest. Both fields are left null, so the dilution factor
scores every coin as the worst case and contributes nothing to the ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from screener.data.history import HistoryStore

RETURN_WINDOWS: dict[str, int] = {
    "ret_24h": 1,
    "ret_7d": 7,
    "ret_14d": 14,
    "ret_30d": 30,
    "ret_200d": 200,
    "ret_1y": 365,
}

SURVIVORSHIP_WARNING = (
    "This backtest is built from the coins that are in the universe today, so it "
    "cannot see anything that was delisted, abandoned or collapsed before the "
    "snapshot was taken. Treat the result as a check that the machinery works, "
    "not as evidence of edge."
)


@dataclass(frozen=True)
class UniverseBuilder:
    """Builds a snapshot-shaped frame for any date the history store covers."""

    prices: pd.DataFrame
    market_caps: pd.DataFrame
    volumes: pd.DataFrame
    metadata: pd.DataFrame

    @classmethod
    def from_history(
        cls, history: HistoryStore, coin_ids: list[str], metadata: pd.DataFrame
    ) -> UniverseBuilder:
        return cls(
            prices=history.panel(coin_ids, "price"),
            market_caps=history.panel(coin_ids, "market_cap"),
            volumes=history.panel(coin_ids, "volume"),
            metadata=metadata,
        )

    @property
    def dates(self) -> list[date]:
        return [] if self.prices.empty else sorted(self.prices.index)

    def _as_of_row(self, frame: pd.DataFrame, as_of: date) -> pd.Series[float]:
        if frame.empty:
            return pd.Series(dtype="float64")
        usable = [day for day in frame.index if day <= as_of]
        if not usable:
            return pd.Series(dtype="float64")
        # Most recent observation at or before as_of, per coin. ffill across the
        # truncated window rather than reading one row, so a coin that missed a
        # day is still valued from its own last print.
        return frame.loc[usable].ffill().iloc[-1]

    def frame_for(self, as_of: date) -> pd.DataFrame:
        if self.prices.empty:
            return pd.DataFrame()
        history = self.prices.loc[[day for day in self.prices.index if day <= as_of]]
        if history.empty:
            return pd.DataFrame()

        price_now = self._as_of_row(self.prices, as_of)
        traded_today = self.prices.loc[as_of] if as_of in self.prices.index else None

        records = []
        for coin_id in sorted(self.prices.columns):
            current = float(price_now.get(coin_id, np.nan))
            if not np.isfinite(current) or current <= 0:
                continue
            # A coin with no print on the rebalance date is not tradeable that
            # day; leaving it out of the universe is the same rule the live
            # screener applies to a coin that has vanished from the snapshot.
            if traded_today is not None:
                today_price = traded_today.get(coin_id, np.nan)
                if not (pd.notna(today_price) and float(today_price) > 0):
                    continue

            series = history[coin_id].dropna()
            if series.empty:
                continue

            record: dict[str, object] = {
                "coin_id": coin_id,
                "current_price": current,
                "market_cap": float(self._as_of_row(self.market_caps, as_of).get(coin_id, np.nan)),
                "total_volume": float(self._as_of_row(self.volumes, as_of).get(coin_id, np.nan)),
                # Running max to date. Never the snapshot's ath field.
                "ath": float(series.max()),
                "atl": float(series.min()),
                # No point-in-time supply data exists on the free tier.
                "circulating_supply": np.nan,
                "max_supply": np.nan,
                "total_supply": np.nan,
                "fully_diluted_valuation": np.nan,
                "ret_1h": np.nan,
            }
            for column, window in RETURN_WINDOWS.items():
                record[column] = _trailing_return(series, as_of, window)
            records.append(record)

        if not records:
            return pd.DataFrame()

        frame = pd.DataFrame(records)
        frame = frame.merge(self.metadata, on="coin_id", how="left")
        if "symbol" not in frame.columns:
            frame["symbol"] = frame["coin_id"].str.upper()
        if "name" not in frame.columns:
            frame["name"] = frame["coin_id"]
        frame["symbol"] = frame["symbol"].fillna(frame["coin_id"].str.upper())
        frame["name"] = frame["name"].fillna(frame["coin_id"])
        frame["market_cap_rank"] = (
            frame["market_cap"].rank(ascending=False, method="min").astype("Int64")
        )
        return frame.sort_values("coin_id", kind="mergesort").reset_index(drop=True)


def _trailing_return(series: pd.Series[float], as_of: date, window: int) -> float:
    """Return over `window` calendar days, or NaN when the history is too short.

    NaN rather than a shorter-window substitute: a 30-day factor computed over
    nine days is a different factor wearing the same name, and the WORST missing
    policy will handle the gap.
    """
    target = as_of - timedelta(days=window)
    earlier = series.loc[[day for day in series.index if day <= target]]
    if earlier.empty:
        return float("nan")
    start = float(earlier.iloc[-1])
    end = float(series.iloc[-1])
    if start <= 0:
        return float("nan")
    return end / start - 1.0


def metadata_from_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ("coin_id", "symbol", "name") if column in snapshot.columns]
    return snapshot.loc[:, columns].drop_duplicates(subset="coin_id").reset_index(drop=True)
