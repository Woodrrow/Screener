"""Cross-sectional percentile ranking.

Percentiles rather than z-scores: crypto cross-sections have tails fat enough
that a single 400% mover would dominate a z-blend, and the composite would be
"whatever pumped hardest this week" wearing a factor model's clothes. A
percentile caps any one coin's contribution to a factor at 1.0.
"""

from __future__ import annotations

import pandas as pd

from screener.factors.registry import Direction, NaPolicy

_NA_VALUE: dict[NaPolicy, float] = {
    NaPolicy.WORST: 0.0,
    NaPolicy.NEUTRAL: 0.5,
    NaPolicy.BEST: 1.0,
}


def percentile_rank(
    values: pd.Series[float],
    *,
    direction: Direction,
    na_policy: NaPolicy = NaPolicy.WORST,
) -> pd.Series[float]:
    """Rank to [0, 1] where 1 is best under `direction`.

    Ties take the average rank, so N identical values all score the same rather
    than being ordered by row position. Missing values are ranked separately
    rather than dropped: they are assigned the policy's score after the real
    values are ranked, so a gap never shifts anybody else's percentile.
    """
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    present = numeric.dropna()
    result = pd.Series(_NA_VALUE[na_policy], index=numeric.index, dtype="float64")

    count = len(present)
    if count == 0:
        return result
    if count == 1:
        result.loc[present.index] = 0.5
        return result

    ascending = direction is Direction.HIGHER_IS_BETTER
    ranks = present.rank(method="average", ascending=ascending)
    result.loc[present.index] = (ranks - 1.0) / (count - 1.0)
    return result


def composite_score(percentiles: pd.DataFrame, weights: dict[str, float]) -> pd.Series[float]:
    """Weighted sum of percentiles, rescaled to 0-100.

    Weights are all positive and sum to 1 (directions carry the sign), so the
    result is bounded by construction and comparable across presets.
    """
    missing = sorted(set(weights) - set(percentiles.columns))
    if missing:
        raise KeyError(f"no percentile column for factors: {missing}")
    ordered = sorted(weights)
    weighted = sum(percentiles[name] * weights[name] for name in ordered)
    return pd.Series(weighted, index=percentiles.index, dtype="float64") * 100.0
