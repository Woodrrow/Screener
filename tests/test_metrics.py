from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from screener import metrics

START = date(2025, 1, 1)


def curve(values: list[float]) -> pd.Series[float]:
    index = [START + timedelta(days=offset) for offset in range(len(values))]
    return pd.Series(values, index=pd.Index(index, name="date"), dtype="float64")


def test_total_return_is_end_over_start() -> None:
    assert metrics.total_return(curve([100.0, 110.0, 120.0])) == pytest.approx(0.2)


def test_annualised_return_compounds_over_the_actual_window() -> None:
    """A 10% gain over 365 days annualises to 10%; over 73 days it does not."""
    year = curve([100.0] * 365 + [110.0])
    assert metrics.annualised_return(year) == pytest.approx(0.10, rel=1e-3)

    short = curve([100.0] * 73 + [110.0])
    assert metrics.annualised_return(short) > 0.5


def test_volatility_of_a_flat_curve_is_zero() -> None:
    assert metrics.volatility(curve([100.0] * 10)) == pytest.approx(0.0)


def test_sharpe_of_a_flat_curve_is_undefined_not_infinite() -> None:
    """A zero denominator is missing information, not a perfect score."""
    assert np.isnan(metrics.sharpe(curve([100.0] * 10)))


def test_sharpe_matches_a_hand_computation() -> None:
    values = [100.0, 101.0, 100.0, 102.0, 101.0, 103.0]
    series = curve(values)
    returns = pd.Series(values).pct_change().dropna()
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(365)
    assert metrics.sharpe(series) == pytest.approx(float(expected))


def test_sortino_ignores_upside_deviation() -> None:
    """An all-up curve has no downside, so Sortino is undefined rather than huge."""
    rising = curve([100.0, 101.0, 102.0, 103.0])
    assert np.isnan(metrics.sortino(rising))

    mixed = curve([100.0, 99.0, 101.0, 100.0, 103.0])
    assert metrics.sortino(mixed) > metrics.sharpe(mixed)


def test_max_drawdown_measures_peak_to_trough() -> None:
    values = curve([100.0, 120.0, 60.0, 90.0])
    assert metrics.max_drawdown(values) == pytest.approx(-0.5)
    assert metrics.drawdown_series(values).iloc[-1] == pytest.approx(-0.25)


def test_calmar_is_annualised_return_over_drawdown_depth() -> None:
    values = curve([100.0] * 100 + [80.0, 120.0])
    expected = metrics.annualised_return(values) / abs(metrics.max_drawdown(values))
    assert metrics.calmar(values) == pytest.approx(expected)


def test_calmar_of_a_curve_that_never_fell_is_undefined() -> None:
    assert np.isnan(metrics.calmar(curve([100.0, 110.0, 120.0])))


def test_rolling_sharpe_needs_a_full_window() -> None:
    assert metrics.rolling_sharpe(curve([100.0] * 10), window=30).empty
    rng = np.random.default_rng(3)
    noisy = curve(list(100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, size=100)))))
    rolled = metrics.rolling_sharpe(noisy, window=30)
    assert not rolled.empty
    assert rolled.notna().all()


def test_correlation_of_a_series_with_itself_is_one() -> None:
    rng = np.random.default_rng(5)
    series = curve(list(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=50)))))
    assert metrics.correlation(series, series) == pytest.approx(1.0)


def test_correlation_needs_overlapping_dates() -> None:
    left = curve([100.0, 101.0, 102.0])
    right = pd.Series(
        [100.0, 101.0],
        index=pd.Index([START + timedelta(days=90), START + timedelta(days=91)], name="date"),
        dtype="float64",
    )
    assert np.isnan(metrics.correlation(left, right))


# ------------------------------------------------------------------ trade log


def trade(
    day: date,
    action: str,
    coin_id: str,
    units: float,
    price: float,
    *,
    rank: int | None = None,
) -> dict[str, object]:
    gross = units * price
    return {
        "trade_date": day.isoformat(),
        "action": action,
        "coin_id": coin_id,
        "units": units,
        "intended_price": price,
        "fill_price": price,
        "gross_usd": gross,
        "fee_usd": gross * 0.001,
        "slippage_usd": gross * 0.003,
        "cash_after": 0.0,
        "equity_after": 0.0,
        "rank": rank,
        "reason": action,
    }


def test_cost_drag_totals_and_scales_to_capital() -> None:
    trades = pd.DataFrame([trade(START, "buy", "a", 10.0, 100.0)])
    drag = metrics.cost_drag(trades, 1000.0)
    assert drag["fees_usd"] == pytest.approx(1.0)
    assert drag["slippage_usd"] == pytest.approx(3.0)
    assert drag["pct_of_capital"] == pytest.approx(0.004)


def test_cost_drag_of_an_empty_log_is_zero() -> None:
    assert metrics.cost_drag(pd.DataFrame(), 1000.0)["total_usd"] == 0.0


def test_turnover_is_traded_notional_over_equity() -> None:
    trades = pd.DataFrame(
        [
            trade(START, "buy", "a", 1.0, 500.0),
            trade(START, "buy", "b", 1.0, 500.0),
        ]
    )
    equity = curve([1000.0, 1000.0])
    equity.index = pd.Index([START, START + timedelta(days=1)], name="date")
    assert metrics.turnover(trades, equity) == pytest.approx(1.0)


def test_round_trips_are_matched_fifo() -> None:
    trades = pd.DataFrame(
        [
            trade(START, "buy", "a", 1.0, 100.0, rank=1),
            trade(START + timedelta(days=10), "buy", "a", 1.0, 200.0, rank=4),
            trade(START + timedelta(days=20), "sell", "a", 1.0, 300.0),
        ]
    )
    # The first lot is the one closed, so the holding period is 20 days and the
    # entry rank recorded is 1, not 4.
    assert metrics.average_holding_days(trades) == pytest.approx(20.0)
    assert metrics.win_rate(trades) == pytest.approx(1.0)


def test_a_losing_round_trip_counts_against_the_win_rate() -> None:
    trades = pd.DataFrame(
        [
            trade(START, "buy", "a", 1.0, 100.0, rank=1),
            trade(START + timedelta(days=5), "sell", "a", 1.0, 50.0),
            trade(START, "buy", "b", 1.0, 100.0, rank=2),
            trade(START + timedelta(days=5), "sell", "b", 1.0, 150.0),
        ]
    )
    assert metrics.win_rate(trades) == pytest.approx(0.5)


def test_a_write_off_is_not_counted_as_a_round_trip() -> None:
    """It has no proceeds and no meaningful holding period; a fake -100% would skew both."""
    trades = pd.DataFrame(
        [
            trade(START, "buy", "a", 1.0, 100.0, rank=1),
            trade(START + timedelta(days=5), "write_off", "a", 1.0, 0.0),
        ]
    )
    assert np.isnan(metrics.win_rate(trades))


def test_open_positions_are_not_counted() -> None:
    trades = pd.DataFrame([trade(START, "buy", "a", 1.0, 100.0, rank=1)])
    assert np.isnan(metrics.average_holding_days(trades))


def test_decile_table_groups_by_entry_rank() -> None:
    trades = pd.DataFrame(
        [
            trade(START, "buy", "a", 1.0, 100.0, rank=1),
            trade(START + timedelta(days=5), "sell", "a", 1.0, 150.0),
            trade(START, "buy", "b", 1.0, 100.0, rank=10),
            trade(START + timedelta(days=5), "sell", "b", 1.0, 60.0),
        ]
    )
    table = metrics.return_by_entry_rank_decile(trades)
    assert not table.empty
    assert int(table["trades"].sum()) == 2


def test_summary_flags_a_thin_sample() -> None:
    thin = metrics.summarise(curve([100.0] * 10), "thin")
    assert thin.is_statistically_thin
    fat = metrics.summarise(curve([100.0] * 200), "fat")
    assert not fat.is_statistically_thin
