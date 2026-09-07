"""Trailing return factors.

Three separate horizons rather than one blended momentum score, because the
weights that suit them differ: 7d is largely noise and mean-reverting, 30d is
the classic cross-sectional momentum window, and 200d is closer to a trend
filter. A preset that wants all three can have all three; one that wants only
the slow leg is not forced to take the fast one with it.
"""

from __future__ import annotations

import pandas as pd

from screener.factors.context import FactorContext
from screener.factors.registry import Direction, NaPolicy, register


@register(
    "momentum_7d",
    direction=Direction.HIGHER_IS_BETTER,
    na_policy=NaPolicy.WORST,
    description="Trailing 7-day return.",
)
def momentum_7d(context: FactorContext) -> pd.Series[float]:
    return context.column("ret_7d")


@register(
    "momentum_30d",
    direction=Direction.HIGHER_IS_BETTER,
    na_policy=NaPolicy.WORST,
    description="Trailing 30-day return.",
)
def momentum_30d(context: FactorContext) -> pd.Series[float]:
    return context.column("ret_30d")


@register(
    "momentum_200d",
    direction=Direction.HIGHER_IS_BETTER,
    na_policy=NaPolicy.WORST,
    description="Trailing 200-day return.",
)
def momentum_200d(context: FactorContext) -> pd.Series[float]:
    return context.column("ret_200d")
