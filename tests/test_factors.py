from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from screener import factors
from screener.factors.context import FactorContext, LookaheadError
from screener.factors.registry import Direction, NaPolicy
from screener.factors.volatility import MIN_OBSERVATIONS, TRADING_DAYS_PER_YEAR, WINDOW_DAYS

AS_OF = date(2025, 3, 4)


def _context(frame: pd.DataFrame, panel: pd.DataFrame | None = None) -> FactorContext:
    return FactorContext(
        universe=frame.reset_index(drop=True),
        as_of=AS_OF,
        price_panel=pd.DataFrame() if panel is None else panel,
    )


def _value(series: pd.Series[float], coin_id: str) -> float:
    return float(series.loc[coin_id])


def test_every_registered_factor_returns_one_value_per_coin(
    universe_frame: pd.DataFrame,
) -> None:
    context = _context(universe_frame)
    for spec in factors.all_specs():
        values = spec.fn(context)
        assert list(values.index) == list(context.coin_ids), spec.name
        assert values.dtype == np.float64, spec.name


def test_momentum_reads_the_stored_fractions(universe_frame: pd.DataFrame) -> None:
    context = _context(universe_frame)
    assert _value(factors.get("momentum_7d").fn(context), "bitcoin") == pytest.approx(0.03)
    assert _value(factors.get("momentum_30d").fn(context), "ethereum") == pytest.approx(-0.04)
    assert _value(factors.get("momentum_200d").fn(context), "solana") == pytest.approx(1.20)


def test_liquidity_is_volume_over_market_cap(universe_frame: pd.DataFrame) -> None:
    context = _context(universe_frame)
    values = factors.get("liquidity").fn(context)
    assert _value(values, "bitcoin") == pytest.approx(3.0e10 / 1.2e12)


def test_dilution_scores_uncapped_emission_as_missing(universe_frame: pd.DataFrame) -> None:
    """Null max supply must reach the ranker as NaN so the WORST policy applies."""
    context = _context(universe_frame)
    values = factors.get("dilution").fn(context)

    assert _value(values, "chainlink") == pytest.approx(0.62)
    assert np.isnan(values.loc["inflato"])

    spec = factors.get("dilution")
    assert spec.na_policy is NaPolicy.WORST
    ranked = factors.percentile_rank(values, direction=spec.direction, na_policy=spec.na_policy)
    assert ranked.loc["inflato"] == 0.0


def test_dilution_clips_reported_overshoot() -> None:
    frame = pd.DataFrame([{"coin_id": "odd", "circulating_supply": 120.0, "max_supply": 100.0}])
    values = factors.get("dilution").fn(_context(frame))
    assert _value(values, "odd") == 1.0


def test_drawdown_and_trend_health_are_complements(universe_frame: pd.DataFrame) -> None:
    context = _context(universe_frame)
    depth = factors.get("drawdown_depth").fn(context)
    health = factors.get("trend_health").fn(context)

    assert _value(depth, "bitcoin") == pytest.approx(1.0 - 60000.0 / 73000.0)
    assert (depth + health).dropna().round(12).eq(1.0).all()


def test_trend_health_clips_a_fresh_high() -> None:
    frame = pd.DataFrame([{"coin_id": "breakout", "current_price": 110.0, "ath": 100.0}])
    context = _context(frame)
    assert _value(factors.get("trend_health").fn(context), "breakout") == 1.0
    assert _value(factors.get("drawdown_depth").fn(context), "breakout") == 0.0


def test_size_tilt_prefers_the_small(universe_frame: pd.DataFrame) -> None:
    spec = factors.get("size_tilt")
    assert spec.direction is Direction.LOWER_IS_BETTER

    context = _context(universe_frame)
    ranked = factors.percentile_rank(
        spec.fn(context), direction=spec.direction, na_policy=spec.na_policy
    )
    assert ranked.loc["tiny-coin"] > ranked.loc["bitcoin"]


def _flat_panel(coin_ids: list[str], days: int, drift: float = 0.0) -> pd.DataFrame:
    index = [AS_OF - timedelta(days=offset) for offset in range(days - 1, -1, -1)]
    data = {coin: [100.0 * (1.0 + drift) ** step for step in range(days)] for coin in coin_ids}
    return pd.DataFrame(data, index=pd.Index(index, name="date"))


def test_volatility_of_a_constant_series_is_zero() -> None:
    panel = _flat_panel(["steady"], days=WINDOW_DAYS + 5)
    frame = pd.DataFrame([{"coin_id": "steady"}])
    values = factors.get("volatility").fn(_context(frame, panel))
    assert _value(values, "steady") == pytest.approx(0.0)


def test_volatility_matches_a_hand_computed_figure() -> None:
    rng = np.random.default_rng(1234)
    steps = rng.normal(0.0, 0.02, size=WINDOW_DAYS)
    prices = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], steps])))
    index = [AS_OF - timedelta(days=offset) for offset in range(len(prices) - 1, -1, -1)]
    panel = pd.DataFrame({"noisy": prices}, index=pd.Index(index, name="date"))

    expected = float(np.std(steps, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    values = factors.get("volatility").fn(_context(pd.DataFrame([{"coin_id": "noisy"}]), panel))
    assert _value(values, "noisy") == pytest.approx(expected)


def test_volatility_uses_only_the_trailing_window() -> None:
    """A calm recent window must not be polluted by an ancient shock."""
    calm = _flat_panel(["coin"], days=WINDOW_DAYS + 1)
    long_history = _flat_panel(["coin"], days=WINDOW_DAYS + 200)
    long_history.iloc[0, 0] = 1.0  # a violent move 200+ days ago

    calm_vol = factors.get("volatility").fn(_context(pd.DataFrame([{"coin_id": "coin"}]), calm))
    long_vol = factors.get("volatility").fn(
        _context(pd.DataFrame([{"coin_id": "coin"}]), long_history)
    )
    assert _value(calm_vol, "coin") == pytest.approx(_value(long_vol, "coin"))


def test_thin_history_scores_worst_not_neutral() -> None:
    panel = _flat_panel(["newcomer"], days=MIN_OBSERVATIONS - 5)
    spec = factors.get("volatility")
    values = spec.fn(_context(pd.DataFrame([{"coin_id": "newcomer"}]), panel))

    assert np.isnan(values.loc["newcomer"])
    assert spec.na_policy is NaPolicy.WORST


def test_context_refuses_a_panel_that_runs_past_as_of() -> None:
    panel = pd.DataFrame(
        {"coin": [1.0, 2.0]},
        index=pd.Index([AS_OF, AS_OF + timedelta(days=1)], name="date"),
    )
    with pytest.raises(LookaheadError, match="after as_of"):
        _context(pd.DataFrame([{"coin_id": "coin"}]), panel)


def test_context_allows_a_panel_ending_exactly_at_as_of() -> None:
    panel = pd.DataFrame({"coin": [1.0]}, index=pd.Index([AS_OF], name="date"))
    assert _context(pd.DataFrame([{"coin_id": "coin"}]), panel).as_of == AS_OF


def test_registering_a_duplicate_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):
        factors.register("momentum_7d", direction=Direction.HIGHER_IS_BETTER, description="dupe")(
            lambda context: context.empty_series()
        )


def test_unknown_factor_lookup_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="momentum_30d"):
        factors.get("no_such_factor")
