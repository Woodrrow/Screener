from __future__ import annotations

from datetime import date

import pytest

from screener.portfolio.costs import CostModel
from screener.portfolio.ledger import Ledger, ReconciliationError, ShortSaleError

DAY = date(2025, 1, 6)
COSTS = CostModel(fee_bps=10.0, slippage_bps=30.0)


def _ledger(cash: float = 1000.0) -> Ledger:
    return Ledger(cash=cash, costs=COSTS)


def test_buy_leaves_cash_and_positions_reconciled() -> None:
    ledger = _ledger()
    prices = {"bitcoin": 100.0}
    ledger.buy("bitcoin", usd=500.0, intended_price=100.0, trade_date=DAY, prices=prices)

    assert ledger.cash == pytest.approx(500.0)
    equity = ledger.reconcile(prices)
    assert equity == pytest.approx(ledger.cash + ledger.positions_value(prices))
    # The only value lost is the fee plus the slippage.
    assert equity == pytest.approx(
        1000.0 - ledger.trades[0].fee_usd - ledger.trades[0].slippage_usd
    )


def test_a_full_round_trip_reconciles_to_the_cost() -> None:
    ledger = _ledger()
    prices = {"bitcoin": 100.0}
    ledger.buy("bitcoin", usd=1000.0, intended_price=100.0, trade_date=DAY, prices=prices)
    ledger.liquidate("bitcoin", intended_price=100.0, trade_date=DAY, prices=prices)

    assert ledger.positions == {}
    total_costs = sum(record.fee_usd + record.slippage_usd for record in ledger.trades)
    assert ledger.cash == pytest.approx(1000.0 - total_costs)
    assert ledger.reconcile(prices) == pytest.approx(ledger.cash)


def test_cash_never_goes_negative_even_when_asked_to_overspend() -> None:
    ledger = _ledger(100.0)
    ledger.buy(
        "bitcoin", usd=10_000.0, intended_price=100.0, trade_date=DAY, prices={"bitcoin": 100.0}
    )
    assert ledger.cash >= 0.0
    assert ledger.cash == pytest.approx(0.0, abs=1e-9)


def test_selling_more_than_held_is_refused() -> None:
    ledger = _ledger()
    prices = {"bitcoin": 100.0}
    ledger.buy("bitcoin", usd=500.0, intended_price=100.0, trade_date=DAY, prices=prices)
    units = ledger.positions["bitcoin"].units
    with pytest.raises(ShortSaleError):
        ledger.sell("bitcoin", units=units * 2, intended_price=100.0, trade_date=DAY, prices=prices)


def test_partial_sale_scales_the_cost_basis() -> None:
    ledger = _ledger()
    prices = {"bitcoin": 100.0}
    ledger.buy("bitcoin", usd=1000.0, intended_price=100.0, trade_date=DAY, prices=prices)
    basis_before = ledger.positions["bitcoin"].cost_basis_usd
    units = ledger.positions["bitcoin"].units

    ledger.sell("bitcoin", units=units / 4, intended_price=100.0, trade_date=DAY, prices=prices)

    assert ledger.positions["bitcoin"].units == pytest.approx(units * 0.75)
    assert ledger.positions["bitcoin"].cost_basis_usd == pytest.approx(basis_before * 0.75)


def test_negative_cash_is_caught_by_reconcile() -> None:
    ledger = _ledger()
    ledger.cash = -1.0
    with pytest.raises(ReconciliationError, match="cash is negative"):
        ledger.reconcile({})


def test_price_falls_back_to_the_last_one_seen() -> None:
    ledger = _ledger()
    ledger.observe_prices({"bitcoin": 100.0})
    ledger.buy(
        "bitcoin", usd=500.0, intended_price=100.0, trade_date=DAY, prices={"bitcoin": 100.0}
    )

    # The coin stops printing; it is still valued at its last known price.
    assert ledger.price_for("bitcoin", {}) == 100.0
    assert ledger.positions_value({}) > 0


def test_write_off_removes_a_position_and_logs_it() -> None:
    ledger = _ledger()
    ledger.buy("ghost", usd=500.0, intended_price=100.0, trade_date=DAY, prices={"ghost": 100.0})
    record = ledger.write_off("ghost", trade_date=DAY, prices={}, reason="vanished")

    assert record is not None
    assert record.action == "write_off"
    assert "ghost" not in ledger.positions
    assert ledger.trade_log()["action"].tolist() == ["buy", "write_off"]


def test_trade_log_carries_the_full_audit_trail() -> None:
    ledger = _ledger()
    prices = {"bitcoin": 100.0}
    ledger.buy(
        "bitcoin",
        usd=500.0,
        intended_price=100.0,
        trade_date=DAY,
        prices=prices,
        rank=3,
        reason="entry_top_rank",
    )
    row = ledger.trade_log().iloc[0]

    assert row["intended_price"] == 100.0
    assert row["fill_price"] == pytest.approx(100.30)
    assert row["fee_usd"] > 0
    assert row["slippage_usd"] > 0
    assert row["rank"] == 3
    assert row["reason"] == "entry_top_rank"
    assert row["cash_after"] == pytest.approx(500.0)


def test_holdings_view_is_sorted_and_priced() -> None:
    ledger = _ledger()
    prices = {"bitcoin": 100.0, "aave": 50.0}
    for coin_id, price in sorted(prices.items(), reverse=True):
        ledger.buy(coin_id, usd=200.0, intended_price=price, trade_date=DAY, prices=prices)

    holdings = ledger.holdings(prices)
    assert holdings["coin_id"].tolist() == ["aave", "bitcoin"]
    assert holdings["value_usd"].sum() == pytest.approx(ledger.positions_value(prices))
