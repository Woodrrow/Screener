"""Realised volatility, as a penalty.

Trailing 60-day annualised standard deviation of daily log returns, computed
from the history store rather than the snapshot, so it is available in a
backtest on the same terms as live.

Direction is LOWER_IS_BETTER, which is how a penalty term is expressed here: all
configured weights stay positive and sum to 1, and the sign lives with the
factor that owns it rather than in a weight the config has to get right.

A coin with too little history scores worst, not neutral. Thin history means a
new listing, and a new listing is the most volatile thing in the universe -
scoring it neutral would hand it a free pass on the one factor meant to catch it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.factors.context import FactorContext
from screener.factors.registry import Direction, NaPolicy, register

WINDOW_DAYS = 60
MIN_OBSERVATIONS = 30
TRADING_DAYS_PER_YEAR = 365  # crypto trades every day


@register(
    "volatility",
    direction=Direction.LOWER_IS_BETTER,
    na_policy=NaPolicy.WORST,
    description="Trailing 60-day annualised realised volatility; lower scores higher.",
    needs_history=True,
)
def volatility(context: FactorContext) -> pd.Series[float]:
    result = context.empty_series()
    if context.price_panel.empty:
        return result

    for coin_id in result.index:
        prices = context.prices_for(str(coin_id))
        if len(prices) < MIN_OBSERVATIONS + 1:
            continue
        window = prices.iloc[-(WINDOW_DAYS + 1) :]
        returns = np.diff(np.log(window.to_numpy(dtype="float64")))
        returns = returns[np.isfinite(returns)]
        if len(returns) < MIN_OBSERVATIONS:
            continue
        result.loc[coin_id] = float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    return result
