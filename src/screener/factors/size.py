"""Size tilt.

Smaller market caps score higher. Ranked on market cap with the direction
inverted rather than on 1/market_cap: the reciprocal of a cross-section spanning
six orders of magnitude is numerically nasty, and the percentile rank of the
reciprocal is the reciprocal-free rank anyway.
"""

from __future__ import annotations

import pandas as pd

from screener.factors.context import FactorContext
from screener.factors.registry import Direction, NaPolicy, register


@register(
    "size_tilt",
    direction=Direction.LOWER_IS_BETTER,
    na_policy=NaPolicy.WORST,
    description="Inverse market cap: smaller scores higher.",
)
def size_tilt(context: FactorContext) -> pd.Series[float]:
    return context.column("market_cap")
