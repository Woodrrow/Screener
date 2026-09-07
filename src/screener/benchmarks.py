"""Benchmarks.

Three, not one. Beating BTC is the headline everyone quotes, but it conflates
two different claims: that crypto went up, and that the factors picked the right
crypto. The equal-weight basket of the top 8 by market cap, rebalanced on the
same schedule and charged the same costs, separates them - it is what you would
have got for no research at all. If the strategy does not beat that, the factors
added nothing.

Every benchmark pays entry costs. A buy-and-hold comparison at zero cost is a
free option the strategy does not get.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd

from screener.config import PortfolioConfig
from screener.portfolio.costs import CostModel
from screener.portfolio.engine import prices_on, run_simulation
from screener.portfolio.ledger import Ledger

# Given a signal date, return the investable universe as it stood then.
UniverseProvider = Callable[[date], pd.DataFrame | None]


def market_cap_ranking(universe: pd.DataFrame) -> pd.DataFrame:
    """Rank the investable universe by market cap, largest first.

    Deliberately the same filtered universe the strategy sees: an "equal weight
    top 8" that included USDT and stETH would be a different asset class, not a
    fair control.
    """
    if universe.empty:
        return pd.DataFrame(columns=pd.Index(["coin_id", "rank", "composite_score"]))
    ordered = universe.sort_values(
        ["market_cap", "coin_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    return pd.DataFrame(
        {
            "coin_id": ordered["coin_id"].astype(str),
            "rank": range(1, len(ordered) + 1),
            "composite_score": [float(len(ordered) - index) for index in range(len(ordered))],
        }
    )


def buy_and_hold(
    price_panel: pd.DataFrame,
    coin_id: str,
    *,
    capital: float,
    costs: CostModel,
    start: date,
    end: date,
) -> pd.Series[float]:
    """Buy once on the first day with a price, then mark daily to `end`."""
    ledger = Ledger(cash=capital, costs=costs)
    equity: dict[date, float] = {}
    bought = False

    day = start
    while day <= end:
        prices = prices_on(price_panel, day)
        ledger.observe_prices(prices)
        price = prices.get(coin_id)
        if not bought and price is not None:
            ledger.buy(
                coin_id,
                usd=capital,
                intended_price=price,
                trade_date=day,
                prices=prices,
                reason="buy_and_hold_entry",
            )
            bought = True
        equity[day] = ledger.equity(prices)
        day += timedelta(days=1)

    curve = pd.Series(equity, dtype="float64").sort_index()
    curve.index.name = "date"
    return curve


def equal_weight_basket(
    price_panel: pd.DataFrame,
    *,
    universe_for: UniverseProvider,
    config: PortfolioConfig,
    capital: float,
    start: date,
    end: date,
    positions: int = 8,
) -> pd.Series[float]:
    """Top-N by market cap, equal weight, same cadence and same costs.

    Runs through the same engine as the strategy so the comparison is like for
    like: identical fill assumptions, identical hysteresis-free rebalancing,
    identical cost model.
    """
    benchmark_config = config.model_copy(
        update={
            "initial_capital": capital,
            "max_positions": positions,
            "entry_rank_threshold": positions,
            "exit_rank_threshold": positions,
            "sizing": "equal_weight",
            "max_weight_per_position": max(config.max_weight_per_position, 1.0 / positions),
        }
    )

    def ranking_for(signal_date: date) -> pd.DataFrame | None:
        universe = universe_for(signal_date)
        if universe is None or universe.empty:
            return None
        return market_cap_ranking(universe)

    result = run_simulation(
        price_panel=price_panel,
        ranking_for=ranking_for,
        config=benchmark_config,
        start=start,
        end=end,
        initial_capital=capital,
    )
    return result.equity_curve
