from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from screener.factors.percentile import composite_score, percentile_rank
from screener.factors.registry import Direction, NaPolicy


def _series(values: list[float], names: list[str] | None = None) -> pd.Series[float]:
    index = names or [f"c{i}" for i in range(len(values))]
    return pd.Series(values, index=index, dtype="float64")


def test_higher_is_better_maps_best_to_one() -> None:
    ranked = percentile_rank(_series([1.0, 2.0, 3.0, 4.0]), direction=Direction.HIGHER_IS_BETTER)
    assert ranked.tolist() == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]


def test_lower_is_better_inverts_the_scale() -> None:
    ranked = percentile_rank(_series([1.0, 2.0, 3.0, 4.0]), direction=Direction.LOWER_IS_BETTER)
    assert ranked.tolist() == [1.0, pytest.approx(2 / 3), pytest.approx(1 / 3), 0.0]


def test_ties_take_the_average_rank() -> None:
    """Identical values must score identically, not be split by row order."""
    ranked = percentile_rank(_series([5.0, 5.0, 5.0, 9.0]), direction=Direction.HIGHER_IS_BETTER)
    assert ranked.iloc[0] == ranked.iloc[1] == ranked.iloc[2]
    assert ranked.iloc[3] == 1.0
    # Three-way tie at ranks 1-3 averages to 2, so (2-1)/(4-1).
    assert ranked.iloc[0] == pytest.approx(1 / 3)


def test_all_equal_values_all_score_the_same() -> None:
    ranked = percentile_rank(_series([2.0] * 5), direction=Direction.HIGHER_IS_BETTER)
    assert ranked.nunique() == 1


def test_missing_values_do_not_shift_real_ranks() -> None:
    with_gaps = _series([1.0, np.nan, 3.0, np.nan, 4.0])
    without = _series([1.0, 3.0, 4.0], names=["c0", "c2", "c4"])

    ranked = percentile_rank(with_gaps, direction=Direction.HIGHER_IS_BETTER)
    baseline = percentile_rank(without, direction=Direction.HIGHER_IS_BETTER)

    assert ranked.loc[["c0", "c2", "c4"]].tolist() == baseline.tolist()


def test_na_policies() -> None:
    values = _series([1.0, np.nan, 3.0])
    worst = percentile_rank(values, direction=Direction.HIGHER_IS_BETTER, na_policy=NaPolicy.WORST)
    neutral = percentile_rank(
        values, direction=Direction.HIGHER_IS_BETTER, na_policy=NaPolicy.NEUTRAL
    )
    best = percentile_rank(values, direction=Direction.HIGHER_IS_BETTER, na_policy=NaPolicy.BEST)

    assert worst.iloc[1] == 0.0
    assert neutral.iloc[1] == 0.5
    assert best.iloc[1] == 1.0


def test_all_missing_falls_back_to_the_policy() -> None:
    ranked = percentile_rank(
        _series([np.nan, np.nan]), direction=Direction.HIGHER_IS_BETTER, na_policy=NaPolicy.WORST
    )
    assert ranked.tolist() == [0.0, 0.0]


def test_single_observation_scores_neutral() -> None:
    """One coin has no cross-section, so it cannot be best or worst."""
    ranked = percentile_rank(_series([7.0]), direction=Direction.HIGHER_IS_BETTER)
    assert ranked.tolist() == [0.5]


def test_non_numeric_input_is_coerced_not_crashed() -> None:
    values = pd.Series(["1.0", "oops", "3.0"], index=["a", "b", "c"])
    ranked = percentile_rank(cast("pd.Series[float]", values), direction=Direction.HIGHER_IS_BETTER)
    assert ranked.loc["b"] == 0.0
    assert ranked.loc["c"] == 1.0


def test_composite_is_bounded_and_weight_ordered() -> None:
    percentiles = pd.DataFrame({"a": [0.0, 0.5, 1.0], "b": [1.0, 0.5, 0.0]}, index=["x", "y", "z"])
    scores = composite_score(percentiles, {"a": 0.25, "b": 0.75})

    assert scores.tolist() == [75.0, 50.0, 25.0]
    assert scores.between(0.0, 100.0).all()


def test_composite_rejects_a_factor_with_no_percentile_column() -> None:
    percentiles = pd.DataFrame({"a": [0.5]}, index=["x"])
    with pytest.raises(KeyError, match="no percentile column"):
        composite_score(percentiles, {"a": 0.5, "missing": 0.5})
