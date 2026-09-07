from __future__ import annotations

import pytest

from screener.portfolio.costs import CostModel, Side, ZeroCostError


def test_defaults_are_ten_and_thirty_bps() -> None:
    costs = CostModel()
    assert costs.fee_rate == pytest.approx(0.0010)
    assert costs.slippage_rate == pytest.approx(0.0030)


def test_a_costless_simulation_is_refused() -> None:
    with pytest.raises(ZeroCostError):
        CostModel(fee_bps=0.0, slippage_bps=0.0)
    with pytest.raises(ZeroCostError):
        CostModel(fee_bps=10.0, slippage_bps=0.0)
    with pytest.raises(ZeroCostError):
        CostModel(fee_bps=0.0, slippage_bps=30.0)


def test_negative_costs_are_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        CostModel(fee_bps=-1.0, slippage_bps=30.0)


def test_slippage_always_moves_against_the_trader() -> None:
    costs = CostModel(fee_bps=10.0, slippage_bps=30.0)
    assert costs.fill_price(Side.BUY, 100.0) == pytest.approx(100.30)
    assert costs.fill_price(Side.SELL, 100.0) == pytest.approx(99.70)


def test_buy_arithmetic_against_hand_computed_values() -> None:
    """10 units at an intended $100, 10bps fee, 30bps slippage.

    fill      = 100 * 1.0030            = 100.30
    gross     = 10 * 100.30             = 1003.00
    fee       = 1003.00 * 0.0010        =    1.003
    slippage  = 10 * (100.30 - 100.00)  =    3.00
    cash out  = 1003.00 + 1.003         = 1004.003
    """
    costs = CostModel(fee_bps=10.0, slippage_bps=30.0)
    fill = costs.execute(Side.BUY, 10.0, 100.0)

    assert fill.fill_price == pytest.approx(100.30)
    assert fill.gross_usd == pytest.approx(1003.00)
    assert fill.fee_usd == pytest.approx(1.003)
    assert fill.slippage_usd == pytest.approx(3.00)
    assert fill.cash_delta == pytest.approx(-1004.003)
    assert fill.total_cost_usd == pytest.approx(4.003)


def test_sell_arithmetic_against_hand_computed_values() -> None:
    """10 units at an intended $100.

    fill     = 100 * 0.9970            =  99.70
    gross    = 997.00
    fee      = 997.00 * 0.0010         =   0.997
    slippage = 10 * (100.00 - 99.70)   =   3.00
    cash in  = 997.00 - 0.997          = 996.003
    """
    costs = CostModel(fee_bps=10.0, slippage_bps=30.0)
    fill = costs.execute(Side.SELL, 10.0, 100.0)

    assert fill.fill_price == pytest.approx(99.70)
    assert fill.fee_usd == pytest.approx(0.997)
    assert fill.slippage_usd == pytest.approx(3.00)
    assert fill.cash_delta == pytest.approx(996.003)


def test_a_round_trip_at_an_unchanged_price_loses_exactly_the_costs() -> None:
    costs = CostModel(fee_bps=10.0, slippage_bps=30.0)
    buy = costs.execute(Side.BUY, 10.0, 100.0)
    sell = costs.execute(Side.SELL, 10.0, 100.0)

    net = buy.cash_delta + sell.cash_delta
    assert net == pytest.approx(-(buy.total_cost_usd + sell.total_cost_usd))
    # ~80bps round trip on $1,000 of notional.
    assert net == pytest.approx(-8.0, abs=0.05)


def test_units_affordable_never_overspends() -> None:
    costs = CostModel(fee_bps=10.0, slippage_bps=30.0)
    units = costs.units_affordable(500.0, 100.0)
    fill = costs.execute(Side.BUY, units, 100.0)
    assert -fill.cash_delta == pytest.approx(500.0)


def test_zero_and_negative_sizes_are_rejected() -> None:
    costs = CostModel()
    with pytest.raises(ValueError, match="units must be positive"):
        costs.execute(Side.BUY, 0.0, 100.0)
    with pytest.raises(ValueError, match="intended price must be positive"):
        costs.execute(Side.BUY, 1.0, 0.0)
    assert costs.units_affordable(-5.0, 100.0) == 0.0
