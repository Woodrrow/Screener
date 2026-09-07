"""Turnover.

24h volume over market cap. On a $1,000 book no position is large enough to move
a market, so this is not a capacity constraint - it is a proxy for whether a
price is being discovered at all. A coin that trades 0.2% of its cap a day has a
price that is mostly stale, and every factor computed from that price inherits
the staleness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.factors.context import FactorContext
from screener.factors.registry import Direction, NaPolicy, register


@register(
    "liquidity",
    direction=Direction.HIGHER_IS_BETTER,
    na_policy=NaPolicy.WORST,
    description="24h volume divided by market cap.",
)
def liquidity(context: FactorContext) -> pd.Series[float]:
    volume = context.column("total_volume")
    market_cap = context.column("market_cap")
    return pd.Series(
        np.where(market_cap > 0, volume / market_cap, np.nan),
        index=volume.index,
        dtype="float64",
    )
