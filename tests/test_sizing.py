from __future__ import annotations

import pytest

from screener.config import PortfolioConfig, Sizing
from screener.portfolio.sizing import cap_weights, dollar_targets, raw_weights


def _config(**overrides: object) -> PortfolioConfig:
    base: dict[str, object] = {
        "max_positions": 4,
        "max_weight_per_position": 0.4,
        "cash_buffer": 0.0,
        "min_position_usd": 50.0,
    }
    base.update(overrides)
    return PortfolioConfig.model_validate(base)


def test_equal_weight_splits_evenly() -> None:
    weights = raw_weights(["b", "a", "c"], mode=Sizing.EQUAL_WEIGHT, scores={}, volatilities={})
    assert weights == {
        "a": pytest.approx(1 / 3),
        "b": pytest.approx(1 / 3),
        "c": pytest.approx(1 / 3),
    }


def test_score_weighted_is_proportional_to_the_composite() -> None:
    weights = raw_weights(
        ["a", "b"], mode=Sizing.SCORE_WEIGHTED, scores={"a": 75.0, "b": 25.0}, volatilities={}
    )
    assert weights["a"] == pytest.approx(0.75)
    assert weights["b"] == pytest.approx(0.25)


def test_score_weighted_falls_back_to_equal_when_all_scores_are_zero() -> None:
    weights = raw_weights(
        ["a", "b"], mode=Sizing.SCORE_WEIGHTED, scores={"a": 0.0, "b": 0.0}, volatilities={}
    )
    assert weights["a"] == pytest.approx(0.5)


def test_inverse_vol_favours_the_calmer_coin() -> None:
    weights = raw_weights(
        ["calm", "wild"],
        mode=Sizing.INVERSE_VOL,
        scores={},
        volatilities={"calm": 0.2, "wild": 0.8},
    )
    assert weights["calm"] == pytest.approx(0.8)
    assert weights["wild"] == pytest.approx(0.2)


def test_inverse_vol_treats_missing_volatility_as_the_worst_in_the_set() -> None:
    """Missing data is sized as the most volatile name, never the least.

    The guarantee is a ceiling, not a strict penalty: with a single observation
    the missing coin ties with it, because there is no worse figure to borrow.
    The volatility factor has already scored the same coin bottom of its
    percentile, so this is the second line of defence rather than the first.
    """
    weights = raw_weights(
        ["calm", "wild", "unknown"],
        mode=Sizing.INVERSE_VOL,
        scores={},
        volatilities={"calm": 0.2, "wild": 0.8},
    )
    assert weights["unknown"] == pytest.approx(weights["wild"])
    assert weights["unknown"] < weights["calm"]


def test_inverse_vol_with_no_volatility_data_at_all_is_equal_weight() -> None:
    weights = raw_weights(["a", "b"], mode=Sizing.INVERSE_VOL, scores={}, volatilities={})
    assert weights == {"a": pytest.approx(0.5), "b": pytest.approx(0.5)}


def test_cap_redistributes_the_excess() -> None:
    capped = cap_weights({"a": 0.7, "b": 0.2, "c": 0.1}, cap=0.4)
    assert capped["a"] == pytest.approx(0.4)
    assert sum(capped.values()) == pytest.approx(1.0)
    # The 0.3 of excess goes 2:1 to b and c, matching their relative sizes.
    assert capped["b"] == pytest.approx(0.4)
    assert capped["c"] == pytest.approx(0.2)


def test_cap_leaves_the_remainder_in_cash_when_everything_is_capped() -> None:
    capped = cap_weights({"a": 0.5, "b": 0.5}, cap=0.3)
    assert capped == {"a": pytest.approx(0.3), "b": pytest.approx(0.3)}
    assert sum(capped.values()) == pytest.approx(0.6)


def test_dollar_targets_respect_the_cash_buffer() -> None:
    config = _config(cash_buffer=0.02)
    targets = dollar_targets(
        ["a", "b", "c"], equity=1000.0, config=config, scores={}, volatilities={}
    )
    assert sum(targets.values()) == pytest.approx(980.0)


def test_positions_under_the_minimum_are_dropped_and_the_rest_resized() -> None:
    """Dropping the runt lifts everyone else, which is the point of the floor."""
    config = _config(max_positions=8, max_weight_per_position=0.5, min_position_usd=50.0)
    targets = dollar_targets(
        ["a", "b", "c", "d", "e"],
        equity=200.0,
        config=config,
        scores={},
        volatilities={},
    )
    # 200/5 = 40 each, under the floor. 200/4 = 50 exactly, which clears it.
    assert len(targets) == 4
    assert all(value >= 50.0 for value in targets.values())


def test_everything_under_the_minimum_yields_an_empty_book() -> None:
    config = _config(min_position_usd=500.0)
    assert dollar_targets(["a"], equity=100.0, config=config, scores={}, volatilities={}) == {}


def test_no_selection_means_no_targets() -> None:
    assert dollar_targets([], equity=1000.0, config=_config(), scores={}, volatilities={}) == {}


def test_sizing_is_order_independent() -> None:
    config = _config()
    forward = dollar_targets(
        ["a", "b", "c"], equity=1000.0, config=config, scores={}, volatilities={}
    )
    reverse = dollar_targets(
        ["c", "b", "a"], equity=1000.0, config=config, scores={}, volatilities={}
    )
    assert forward == reverse
