from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from screener.config import PortfolioConfig
from screener.errors import LookaheadError
from screener.portfolio.costs import CostModel
from screener.portfolio.ledger import Ledger, Position
from screener.portfolio.simulator import Reason, Simulator, assert_execution_after_signal

SIGNAL = date(2025, 1, 5)
FILL = date(2025, 1, 6)
COSTS = CostModel(fee_bps=10.0, slippage_bps=30.0)


def config(**overrides: object) -> PortfolioConfig:
    base: dict[str, object] = {
        "initial_capital": 1000.0,
        "max_positions": 3,
        "max_weight_per_position": 0.4,
        "cash_buffer": 0.0,
        "min_position_usd": 10.0,
        "entry_rank_threshold": 3,
        "exit_rank_threshold": 6,
    }
    base.update(overrides)
    return PortfolioConfig.model_validate(base)


def ranking(order: list[str], *, vols: dict[str, float] | None = None) -> pd.DataFrame:
    rows = [
        {
            "coin_id": coin_id,
            "rank": index + 1,
            "composite_score": 100.0 - index,
            "raw_volatility": (vols or {}).get(coin_id, 0.5),
        }
        for index, coin_id in enumerate(order)
    ]
    return pd.DataFrame(rows)


def prices_for(order: list[str], price: float = 100.0) -> dict[str, float]:
    return dict.fromkeys(order, price)


def test_execution_must_follow_the_signal() -> None:
    assert_execution_after_signal(SIGNAL, FILL)
    with pytest.raises(LookaheadError, match="strictly later"):
        assert_execution_after_signal(SIGNAL, SIGNAL)
    with pytest.raises(LookaheadError):
        assert_execution_after_signal(FILL, SIGNAL)


def test_plan_refuses_a_same_day_fill() -> None:
    simulator = Simulator(config(), COSTS)
    ledger = Ledger(cash=1000.0, costs=COSTS)
    order = ["a", "b", "c"]
    with pytest.raises(LookaheadError):
        simulator.plan(
            ledger,
            ranking=ranking(order),
            prices=prices_for(order),
            signal_date=SIGNAL,
            execution_date=SIGNAL,
        )


def test_first_rebalance_buys_the_top_ranks() -> None:
    simulator = Simulator(config(), COSTS)
    ledger = Ledger(cash=1000.0, costs=COSTS)
    order = ["a", "b", "c", "d", "e"]

    plan = simulator.plan(
        ledger,
        ranking=ranking(order),
        prices=prices_for(order),
        signal_date=SIGNAL,
        execution_date=FILL,
    )
    simulator.execute(ledger, plan, prices_for(order))

    assert sorted(ledger.positions) == ["a", "b", "c"]
    assert {action.reason for action in plan.buys} == {Reason.ENTRY}
    assert ledger.reconcile(prices_for(order)) == pytest.approx(1000.0 / (1.003 * 1.001))


def test_hysteresis_prevents_churn_on_an_oscillating_rank() -> None:
    """A coin bouncing between rank 3 and rank 5 must not be traded every week.

    Buying at 3 and selling at 5 would pay a round trip each oscillation; with
    the exit band at 6 the position simply rides through.
    """
    order_a = ["a", "b", "c", "d", "e", "f"]
    order_b = ["a", "b", "d", "e", "c", "f"]  # c drops from rank 3 to rank 5

    def run(entry: int, exit_: int) -> int:
        simulator = Simulator(config(entry_rank_threshold=entry, exit_rank_threshold=exit_), COSTS)
        ledger = Ledger(cash=1000.0, costs=COSTS)
        day = FILL
        for step in range(12):
            order = order_a if step % 2 == 0 else order_b
            plan = simulator.plan(
                ledger,
                ranking=ranking(order),
                prices=prices_for(order),
                signal_date=day - timedelta(days=1),
                execution_date=day,
            )
            simulator.execute(ledger, plan, prices_for(order))
            day += timedelta(days=7)
        return len(ledger.trades)

    with_hysteresis = run(3, 6)
    without_hysteresis = run(3, 3)

    assert with_hysteresis < without_hysteresis
    # Three entries and nothing else: the book never turns over.
    assert with_hysteresis == 3


def test_a_coin_that_falls_past_the_exit_band_is_sold() -> None:
    simulator = Simulator(config(), COSTS)
    ledger = Ledger(cash=1000.0, costs=COSTS)
    order = ["a", "b", "c", "d", "e", "f", "g"]
    prices = prices_for(order)

    plan = simulator.plan(
        ledger, ranking=ranking(order), prices=prices, signal_date=SIGNAL, execution_date=FILL
    )
    simulator.execute(ledger, plan, prices)

    demoted = ["a", "b", "d", "e", "f", "g", "c"]  # c falls to rank 7, past exit 6
    plan = simulator.plan(
        ledger,
        ranking=ranking(demoted),
        prices=prices,
        signal_date=FILL,
        execution_date=FILL + timedelta(days=7),
    )
    simulator.execute(ledger, plan, prices)

    assert "c" not in ledger.positions
    exits = [action for action in plan.actions if action.coin_id == "c"]
    assert exits[0].reason == Reason.EXIT_RANK


def test_a_coin_that_leaves_the_universe_is_sold_with_that_reason() -> None:
    simulator = Simulator(config(), COSTS)
    ledger = Ledger(cash=1000.0, costs=COSTS)
    order = ["a", "b", "c"]
    prices = prices_for(order)
    plan = simulator.plan(
        ledger, ranking=ranking(order), prices=prices, signal_date=SIGNAL, execution_date=FILL
    )
    simulator.execute(ledger, plan, prices)

    # c is filtered out of the investable universe but still has a price.
    survivors = ["a", "b", "d"]
    plan = simulator.plan(
        ledger,
        ranking=ranking(survivors),
        prices=prices_for(["a", "b", "c", "d"]),
        signal_date=FILL,
        execution_date=FILL + timedelta(days=7),
    )
    assert [action.reason for action in plan.actions if action.coin_id == "c"] == [
        Reason.EXIT_UNIVERSE
    ]


def test_a_delisted_coin_is_force_liquidated_at_its_last_known_price() -> None:
    """No price today means no trade today - except we must still get out."""
    simulator = Simulator(config(), COSTS)
    ledger = Ledger(cash=1000.0, costs=COSTS)
    order = ["a", "b", "c"]
    prices = prices_for(order)
    plan = simulator.plan(
        ledger, ranking=ranking(order), prices=prices, signal_date=SIGNAL, execution_date=FILL
    )
    simulator.execute(ledger, plan, prices)
    units_before = ledger.positions["c"].units

    vanished = {"a": 100.0, "b": 100.0}  # c has no print at all
    plan = simulator.plan(
        ledger,
        ranking=ranking(["a", "b", "d"]),
        prices=vanished,
        signal_date=FILL,
        execution_date=FILL + timedelta(days=7),
    )
    forced = [action for action in plan.actions if action.coin_id == "c"]
    assert forced[0].action == "liquidate"
    assert forced[0].reason == Reason.FORCED
    assert forced[0].price == 100.0

    simulator.execute(ledger, plan, vanished)
    assert "c" not in ledger.positions
    log = ledger.trade_log()
    liquidation = log.loc[log["reason"] == Reason.FORCED].iloc[0]
    assert liquidation["units"] == pytest.approx(units_before)


def test_a_coin_with_no_price_ever_is_written_off_not_dropped() -> None:
    simulator = Simulator(config(), COSTS)
    ledger = Ledger(cash=1000.0, costs=COSTS)
    ledger.positions["orphan"] = Position(
        coin_id="orphan", units=5.0, cost_basis_usd=100.0, opened_on=SIGNAL
    )

    plan = simulator.plan(
        ledger,
        ranking=ranking(["a", "b", "c"]),
        prices=prices_for(["a", "b", "c"]),
        signal_date=SIGNAL,
        execution_date=FILL,
    )
    assert [action.action for action in plan.actions if action.coin_id == "orphan"] == ["write_off"]

    simulator.execute(ledger, plan, prices_for(["a", "b", "c"]))
    assert "orphan" not in ledger.positions
    assert (ledger.trade_log()["reason"] == Reason.WRITE_OFF).any()


def test_capacity_displaces_the_worst_ranked_holding() -> None:
    simulator = Simulator(config(max_positions=2, max_weight_per_position=0.6), COSTS)
    ledger = Ledger(cash=1000.0, costs=COSTS)
    order = ["a", "b", "c"]
    prices = prices_for(order)
    plan = simulator.plan(
        ledger, ranking=ranking(order), prices=prices, signal_date=SIGNAL, execution_date=FILL
    )
    simulator.execute(ledger, plan, prices)
    assert sorted(ledger.positions) == ["a", "b"]


def test_a_dry_plan_and_a_real_plan_agree() -> None:
    simulator = Simulator(config(), COSTS)
    ledger = Ledger(cash=1000.0, costs=COSTS)
    order = ["a", "b", "c", "d"]
    prices = prices_for(order)
    kwargs = {
        "ranking": ranking(order),
        "prices": prices,
        "signal_date": SIGNAL,
        "execution_date": FILL,
    }
    planned = simulator.plan(ledger, **kwargs)  # type: ignore[arg-type]
    repeated = simulator.plan(ledger, **kwargs)  # type: ignore[arg-type]
    assert planned.actions == repeated.actions


def test_buys_are_scaled_rather_than_starved_by_ordering() -> None:
    """If cash runs short, every buy is cut proportionally, not the last one."""
    simulator = Simulator(config(cash_buffer=0.0), COSTS)
    ledger = Ledger(cash=1000.0, costs=COSTS)
    order = ["a", "b", "c"]
    prices = prices_for(order)
    plan = simulator.plan(
        ledger, ranking=ranking(order), prices=prices, signal_date=SIGNAL, execution_date=FILL
    )
    simulator.execute(ledger, plan, prices)

    values = [position.units * 100.0 for position in ledger.positions.values()]
    assert max(values) - min(values) < 1e-6
