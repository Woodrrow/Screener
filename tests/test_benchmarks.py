from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from screener.benchmarks import buy_and_hold, equal_weight_basket, market_cap_ranking
from screener.config import PortfolioConfig
from screener.portfolio.costs import CostModel
from screener.portfolio.schedule import Cadence

START = date(2025, 1, 6)  # Monday
COSTS = CostModel(fee_bps=10.0, slippage_bps=30.0)


def panel(series: dict[str, list[float]]) -> pd.DataFrame:
    length = len(next(iter(series.values())))
    index = [START + timedelta(days=offset) for offset in range(length)]
    return pd.DataFrame(series, index=pd.Index(index, name="date"))


def test_buy_and_hold_pays_entry_costs() -> None:
    """A zero-cost benchmark is a free option the strategy does not get."""
    prices = panel({"bitcoin": [100.0] * 5})
    curve = buy_and_hold(
        prices, "bitcoin", capital=1000.0, costs=COSTS, start=START, end=prices.index[-1]
    )
    assert curve.iloc[-1] == pytest.approx(1000.0 / (1.003 * 1.001))
    assert curve.iloc[-1] < 1000.0


def test_buy_and_hold_tracks_the_price() -> None:
    prices = panel({"bitcoin": [100.0, 100.0, 200.0]})
    curve = buy_and_hold(
        prices, "bitcoin", capital=1000.0, costs=COSTS, start=START, end=prices.index[-1]
    )
    assert curve.iloc[-1] == pytest.approx(curve.iloc[0] * 2, rel=1e-9)


def test_buy_and_hold_of_an_absent_coin_stays_in_cash() -> None:
    prices = panel({"bitcoin": [100.0] * 4})
    curve = buy_and_hold(
        prices, "ethereum", capital=1000.0, costs=COSTS, start=START, end=prices.index[-1]
    )
    assert (curve == 1000.0).all()


def test_market_cap_ranking_is_largest_first_with_a_stable_tiebreak() -> None:
    universe = pd.DataFrame(
        [
            {"coin_id": "zebra", "market_cap": 100.0},
            {"coin_id": "alpha", "market_cap": 100.0},
            {"coin_id": "whale", "market_cap": 900.0},
        ]
    )
    ranking = market_cap_ranking(universe)
    assert ranking["coin_id"].tolist() == ["whale", "alpha", "zebra"]
    assert ranking["rank"].tolist() == [1, 2, 3]


def test_market_cap_ranking_of_an_empty_universe_is_empty() -> None:
    assert market_cap_ranking(pd.DataFrame()).empty


def test_equal_weight_basket_holds_the_top_n_and_pays_costs() -> None:
    coins = [f"coin{index}" for index in range(10)]
    prices = panel({coin: [100.0] * 15 for coin in coins})
    universe = pd.DataFrame(
        [{"coin_id": coin, "market_cap": 1_000.0 - index * 10} for index, coin in enumerate(coins)]
    )
    config = PortfolioConfig(
        initial_capital=1000.0,
        max_positions=8,
        max_weight_per_position=0.2,
        cash_buffer=0.0,
        min_position_usd=10.0,
    )

    curve = equal_weight_basket(
        prices,
        universe_for=lambda _: universe,
        config=config,
        capital=1000.0,
        start=START,
        end=prices.index[-1],
        positions=8,
    )

    # Flat prices, so all that happens is the entry cost.
    assert curve.iloc[-1] == pytest.approx(1000.0 / (1.003 * 1.001))
    assert len(curve) == 15


def test_equal_weight_basket_uses_the_same_cadence_as_the_strategy() -> None:
    coins = [f"coin{index}" for index in range(9)]
    prices = panel({coin: [100.0] * 30 for coin in coins})
    calls: list[date] = []

    def universe_for(signal_date: date) -> pd.DataFrame:
        calls.append(signal_date)
        return pd.DataFrame(
            [{"coin_id": coin, "market_cap": 1_000.0 - index} for index, coin in enumerate(coins)]
        )

    config = PortfolioConfig(
        initial_capital=1000.0,
        max_positions=8,
        max_weight_per_position=0.2,
        cash_buffer=0.0,
        min_position_usd=10.0,
        rebalance=Cadence.WEEKLY,
    )
    equal_weight_basket(
        prices,
        universe_for=universe_for,
        config=config,
        capital=1000.0,
        start=START,
        end=prices.index[-1],
        positions=8,
    )
    # Five Mondays in a 30-day window starting on one, and every signal date is
    # strictly before its fill.
    assert len(calls) == 5
    assert all(day.weekday() == 6 for day in calls)


def test_equal_weight_basket_skips_dates_with_no_universe() -> None:
    prices = panel({"a": [100.0] * 10})
    config = PortfolioConfig(cash_buffer=0.0, max_positions=8, max_weight_per_position=0.2)
    curve = equal_weight_basket(
        prices,
        universe_for=lambda _: None,
        config=config,
        capital=1000.0,
        start=START,
        end=prices.index[-1],
    )
    assert (curve == 1000.0).all()
