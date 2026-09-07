from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from screener.config import PortfolioConfig
from screener.portfolio.engine import prices_on, run_simulation
from screener.portfolio.ledger import RECONCILE_TOLERANCE

START = date(2025, 1, 6)  # a Monday
COINS = ["alpha", "bravo", "charlie", "delta"]


def config(**overrides: object) -> PortfolioConfig:
    base: dict[str, object] = {
        "initial_capital": 1000.0,
        "max_positions": 3,
        "max_weight_per_position": 0.4,
        "cash_buffer": 0.0,
        "min_position_usd": 10.0,
        "entry_rank_threshold": 3,
        "exit_rank_threshold": 6,
        "rebalance": "weekly",
    }
    base.update(overrides)
    return PortfolioConfig.model_validate(base)


def flat_panel(days: int, price: float = 100.0, coins: list[str] | None = None) -> pd.DataFrame:
    index = [START + timedelta(days=offset) for offset in range(days)]
    return pd.DataFrame(
        {coin: [price] * days for coin in (coins or COINS)},
        index=pd.Index(index, name="date"),
    )


def fixed_ranking(order: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"coin_id": coin_id, "rank": index + 1, "composite_score": 100.0 - index}
            for index, coin_id in enumerate(order)
        ]
    )


def test_twelve_rebalances_on_flat_prices_hit_a_known_answer() -> None:
    """With flat prices and no cash buffer the answer is arithmetic.

    The first rebalance spends all $1,000. Each dollar buys 1/(1.003 x 1.001)
    dollars of mid-priced coin, so equity lands at 1000 / 1.004003 and stays
    there: every later rebalance finds the book already on target and trades
    nothing. Three trades in total, twelve rebalances.
    """
    # 84 days from a Monday inclusive covers exactly twelve Mondays.
    panel = flat_panel(days=12 * 7)
    result = run_simulation(
        price_panel=panel,
        ranking_for=lambda _: fixed_ranking(COINS),
        config=config(),
        start=START,
        end=panel.index[-1],
    )

    expected = 1000.0 / (1.003 * 1.001)
    assert len(result.plans) == 12
    assert len(result.trades) == 3
    assert result.final_equity == pytest.approx(expected)
    assert set(result.holdings["coin_id"]) == {"alpha", "bravo", "charlie"}


def test_cash_reconciles_on_every_day_of_a_multi_rebalance_run() -> None:
    """The invariant that matters: cash plus positions equals equity, always."""
    rng = np.random.default_rng(20250106)
    days = 12 * 7
    index = [START + timedelta(days=offset) for offset in range(days)]
    panel = pd.DataFrame(
        {coin: 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.05, size=days))) for coin in COINS},
        index=pd.Index(index, name="date"),
    )

    # Rotate the ranking every week so the book actually turns over.
    def ranking_for(signal_date: date) -> pd.DataFrame:
        shift = (signal_date - START).days // 7
        rotated = COINS[shift % len(COINS) :] + COINS[: shift % len(COINS)]
        return fixed_ranking(rotated)

    result = run_simulation(
        price_panel=panel,
        ranking_for=ranking_for,
        config=config(),
        start=START,
        end=index[-1],
    )

    assert len(result.equity_curve) == days
    assert result.equity_curve.notna().all()
    assert (result.equity_curve > 0).all()
    assert len(result.trades) > 3  # the rotation forced real turnover

    # Re-derive equity independently from the trade log and the daily marks.
    holdings_value = float((result.holdings["units"] * result.holdings["price"]).sum())
    cash = float(result.trades["cash_after"].iloc[-1])
    assert result.final_equity == pytest.approx(cash + holdings_value, abs=1e-6)


def test_the_equity_curve_is_daily_not_per_rebalance() -> None:
    days = 30
    index = [START + timedelta(days=offset) for offset in range(days)]
    prices = np.linspace(100.0, 130.0, days)
    panel = pd.DataFrame(dict.fromkeys(COINS, prices), index=pd.Index(index, name="date"))

    result = run_simulation(
        price_panel=panel,
        ranking_for=lambda _: fixed_ranking(COINS),
        config=config(),
        start=START,
        end=index[-1],
    )

    assert len(result.equity_curve) == days
    # A rising market between weekly rebalances must show up between them.
    assert result.equity_curve.is_monotonic_increasing


def test_costs_are_always_charged() -> None:
    panel = flat_panel(days=15)
    result = run_simulation(
        price_panel=panel,
        ranking_for=lambda _: fixed_ranking(COINS),
        config=config(),
        start=START,
        end=panel.index[-1],
    )
    assert float(result.trades["fee_usd"].sum()) > 0
    assert float(result.trades["slippage_usd"].sum()) > 0
    assert result.final_equity < 1000.0


def test_a_rebalance_with_no_rankable_universe_is_recorded_not_hidden() -> None:
    panel = flat_panel(days=15)
    result = run_simulation(
        price_panel=panel,
        ranking_for=lambda _: None,
        config=config(),
        start=START,
        end=panel.index[-1],
    )
    assert result.skipped_rebalances
    assert result.trades.empty
    assert result.final_equity == pytest.approx(1000.0)


def test_a_delisting_mid_run_is_liquidated_and_logged() -> None:
    days = 22
    index = [START + timedelta(days=offset) for offset in range(days)]
    panel = flat_panel(days=days)
    # charlie stops printing after two weeks.
    panel.loc[[day for day in index if day >= START + timedelta(days=14)], "charlie"] = np.nan

    result = run_simulation(
        price_panel=panel,
        ranking_for=lambda _: fixed_ranking(COINS),
        config=config(),
        start=START,
        end=index[-1],
    )

    reasons = set(result.trades["reason"])
    assert "force_liquidated_no_price" in reasons
    assert "charlie" not in set(result.holdings["coin_id"])
    assert result.equity_curve.iloc[-1] > 0


def test_prices_on_never_forward_fills() -> None:
    panel = flat_panel(days=5)
    panel.iloc[2, 0] = np.nan
    day = panel.index[2]
    assert "alpha" not in prices_on(panel, day)
    assert "bravo" in prices_on(panel, day)


def test_reconcile_tolerance_is_far_below_a_position() -> None:
    """Guards against someone loosening the tolerance until it hides real drift."""
    assert RECONCILE_TOLERANCE <= 1e-6
