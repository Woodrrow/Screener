"""Emission and dilution.

Circulating over max supply: how much of the eventual float is already trading.
A coin at 0.35 has two thirds of its supply still to arrive, and every unlock is
a seller the price has to absorb.

A null max supply means uncapped emission. Dropping those rows would quietly
excuse the worst offenders, and scoring them neutral would rank them above a
coin with a known 60% float. They are scored as the worst case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.factors.context import FactorContext
from screener.factors.registry import Direction, NaPolicy, register


@register(
    "dilution",
    direction=Direction.HIGHER_IS_BETTER,
    na_policy=NaPolicy.WORST,
    description="Circulating supply over max supply; uncapped emission scores worst.",
)
def dilution(context: FactorContext) -> pd.Series[float]:
    circulating = context.column("circulating_supply")
    max_supply = context.column("max_supply")
    ratio = np.where(max_supply > 0, circulating / max_supply, np.nan)
    # Reported circulating occasionally exceeds reported max after a supply
    # change. Clip rather than let a data error score above a fully-floated coin.
    return pd.Series(np.clip(ratio, 0.0, 1.0), index=circulating.index, dtype="float64")
