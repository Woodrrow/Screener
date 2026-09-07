"""Distance from the all-time high.

Two factors from one quantity, because two strategies want opposite signs of it:

- `drawdown_depth` is how far below the ATH a coin sits. A value or
  mean-reversion preset wants this high.
- `trend_health` is proximity to the ATH. A trend preset wants that high.

The ATH is read from the universe frame, which for a live screen is the ATH as
of the snapshot date. The backtest reconstructs the universe with a running-max
ATH computed only from history up to the rebalance date - using the snapshot's
present-day ATH inside a backtest would be scoring 2024 against a high that had
not happened yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.factors.context import FactorContext
from screener.factors.registry import Direction, NaPolicy, register


def _price_over_ath(context: FactorContext) -> pd.Series[float]:
    price = context.column("current_price")
    ath = context.column("ath")
    ratio = np.where((ath > 0) & (price > 0), price / ath, np.nan)
    # A new ATH prints above 1.0 briefly; clip so "at the high" is the ceiling
    # rather than letting a fresh high outscore itself.
    return pd.Series(np.clip(ratio, 0.0, 1.0), index=price.index, dtype="float64")


@register(
    "drawdown_depth",
    direction=Direction.HIGHER_IS_BETTER,
    na_policy=NaPolicy.WORST,
    description="Fractional distance below the all-time high; deeper scores higher.",
)
def drawdown_depth(context: FactorContext) -> pd.Series[float]:
    return 1.0 - _price_over_ath(context)


@register(
    "trend_health",
    direction=Direction.HIGHER_IS_BETTER,
    na_policy=NaPolicy.WORST,
    description="Proximity to the all-time high; nearer scores higher.",
)
def trend_health(context: FactorContext) -> pd.Series[float]:
    return _price_over_ath(context)
