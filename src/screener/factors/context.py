"""What a factor is allowed to see.

The context is the no-lookahead boundary for factor computation: it carries an
`as_of` date and a price panel that is asserted to contain nothing after it. A
factor cannot reach past `as_of` because there is nothing past `as_of` to reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


class LookaheadError(AssertionError):
    """Raised when data timestamped after `as_of` reaches a factor."""


@dataclass(frozen=True)
class FactorContext:
    universe: pd.DataFrame
    as_of: date
    price_panel: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        if "coin_id" not in self.universe.columns:
            raise ValueError("universe must carry a coin_id column")
        if self.price_panel.empty:
            return
        latest = max(self.price_panel.index)
        latest_date = latest if isinstance(latest, date) else pd.Timestamp(latest).date()
        if latest_date > self.as_of:
            raise LookaheadError(
                f"price panel ends {latest_date}, which is after as_of {self.as_of}"
            )

    @property
    def coin_ids(self) -> pd.Index:
        return pd.Index(self.universe["coin_id"].astype(str), name="coin_id")

    def column(self, name: str) -> pd.Series[float]:
        """A numeric universe column, indexed by coin_id."""
        if name not in self.universe.columns:
            return pd.Series(float("nan"), index=self.coin_ids, dtype="float64")
        values = pd.to_numeric(self.universe[name], errors="coerce")
        return pd.Series(values.to_numpy(dtype="float64"), index=self.coin_ids, dtype="float64")

    def empty_series(self) -> pd.Series[float]:
        return pd.Series(float("nan"), index=self.coin_ids, dtype="float64")

    def prices_for(self, coin_id: str) -> pd.Series[float]:
        if self.price_panel.empty or coin_id not in self.price_panel.columns:
            return pd.Series(dtype="float64")
        series = self.price_panel[coin_id].dropna()
        return pd.Series(series.to_numpy(dtype="float64"), index=series.index, dtype="float64")
