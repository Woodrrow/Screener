"""Filters, factors, composite score, ranking.

The pipeline is deterministic end to end: exclusions run in a fixed precedence,
factors are evaluated in sorted name order, ties in the percentile rank take the
average rank, and the final sort breaks ties on coin_id. The same snapshot and
the same weights produce the same file, byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from screener import factors
from screener import universe as universe_filters
from screener.config import UniverseConfig
from screener.data.history import HistoryStore
from screener.factors.context import FactorContext
from screener.factors.percentile import composite_score, percentile_rank

IDENTITY_COLUMNS = (
    "coin_id",
    "symbol",
    "name",
    "market_cap",
    "market_cap_rank",
    "current_price",
    "total_volume",
)


@dataclass(frozen=True)
class ScreenResult:
    as_of: date
    preset: str
    weights: dict[str, float]
    ranking: pd.DataFrame
    excluded: pd.DataFrame

    @property
    def coin_ids(self) -> list[str]:
        return [str(value) for value in self.ranking["coin_id"]]

    def top(self, count: int) -> pd.DataFrame:
        return self.ranking.head(count)

    def rank_of(self, coin_id: str) -> int | None:
        match = self.ranking.loc[self.ranking["coin_id"] == coin_id, "rank"]
        return None if match.empty else int(match.iloc[0])


def load_price_panel(history: HistoryStore, coin_ids: list[str], as_of: date) -> pd.DataFrame:
    """Prices up to and including `as_of`, and never past it.

    The truncation happens here rather than in the factor so that every factor
    inherits it, and `FactorContext` asserts it independently.
    """
    panel = history.price_panel(coin_ids)
    if panel.empty:
        return panel
    return panel.loc[[index for index in panel.index if index <= as_of]]


def run_screen(
    snapshot: pd.DataFrame,
    *,
    as_of: date,
    weights: dict[str, float],
    preset: str,
    universe_config: UniverseConfig,
    price_panel: pd.DataFrame | None = None,
) -> ScreenResult:
    classified = universe_filters.classify(
        snapshot,
        min_market_cap_usd=universe_config.min_market_cap_usd,
        min_volume_24h_usd=universe_config.min_volume_24h_usd,
        min_turnover=universe_config.min_turnover,
        extra_stablecoin_symbols=frozenset(universe_config.extra_stablecoin_symbols),
        extra_excluded_coin_ids=frozenset(universe_config.extra_excluded_coin_ids),
    )
    investable = classified.loc[~classified["excluded"]].reset_index(drop=True)

    context = FactorContext(
        universe=investable,
        as_of=as_of,
        price_panel=pd.DataFrame() if price_panel is None else price_panel,
    )

    raw: dict[str, pd.Series[float]] = {}
    percentiles: dict[str, pd.Series[float]] = {}
    for name in sorted(weights):
        spec = factors.get(name)
        values = spec.fn(context)
        raw[name] = values
        percentiles[name] = percentile_rank(
            values, direction=spec.direction, na_policy=spec.na_policy
        )

    percentile_frame = pd.DataFrame(percentiles, index=context.coin_ids)
    scores = composite_score(percentile_frame, weights)

    ranking = investable.loc[
        :, [column for column in IDENTITY_COLUMNS if column in investable.columns]
    ].copy()
    ranking["composite_score"] = scores.to_numpy()
    for name in sorted(weights):
        ranking[f"pct_{name}"] = percentile_frame[name].to_numpy()
    for name in sorted(weights):
        ranking[f"raw_{name}"] = raw[name].to_numpy()

    # Descending score, coin_id ascending as the tie-break. mergesort is stable,
    # so equal scores keep the coin_id order rather than an arbitrary one.
    ranking = ranking.sort_values(
        ["composite_score", "coin_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))

    excluded = (
        classified.loc[classified["excluded"], ["coin_id", "symbol", "name", "exclusion_reason"]]
        .sort_values("coin_id", kind="mergesort")
        .reset_index(drop=True)
    )
    return ScreenResult(
        as_of=as_of,
        preset=preset,
        weights=dict(sorted(weights.items())),
        ranking=ranking,
        excluded=excluded,
    )


def write_ranking_csv(result: ScreenResult, path: Path) -> Path:
    """Write the ranking with fixed float formatting so runs are diffable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    result.ranking.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
    return path
