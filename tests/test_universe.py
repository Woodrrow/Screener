from __future__ import annotations

import pandas as pd

from screener.universe import ExclusionReason, apply_filters, classify, exclusion_summary

MIN_MARKET_CAP = 50_000_000.0
MIN_VOLUME = 5_000_000.0
MIN_TURNOVER = 0.01


def _classify(
    frame: pd.DataFrame,
    *,
    extra_stablecoin_symbols: frozenset[str] | None = None,
    extra_excluded_coin_ids: frozenset[str] | None = None,
) -> pd.DataFrame:
    return classify(
        frame,
        min_market_cap_usd=MIN_MARKET_CAP,
        min_volume_24h_usd=MIN_VOLUME,
        min_turnover=MIN_TURNOVER,
        extra_stablecoin_symbols=extra_stablecoin_symbols,
        extra_excluded_coin_ids=extra_excluded_coin_ids,
    )


def _apply(
    frame: pd.DataFrame,
    *,
    extra_stablecoin_symbols: frozenset[str] | None = None,
    extra_excluded_coin_ids: frozenset[str] | None = None,
) -> pd.DataFrame:
    return apply_filters(
        frame,
        min_market_cap_usd=MIN_MARKET_CAP,
        min_volume_24h_usd=MIN_VOLUME,
        min_turnover=MIN_TURNOVER,
        extra_stablecoin_symbols=extra_stablecoin_symbols,
        extra_excluded_coin_ids=extra_excluded_coin_ids,
    )


def _reason(frame: pd.DataFrame, coin_id: str) -> str:
    return str(frame.loc[frame["coin_id"] == coin_id, "exclusion_reason"].iloc[0])


def test_listed_and_heuristic_stablecoins_are_excluded(universe_frame: pd.DataFrame) -> None:
    classified = _classify(universe_frame)
    assert _reason(classified, "tether") == ExclusionReason.STABLECOIN_LISTED
    assert _reason(classified, "usd-coin") == ExclusionReason.STABLECOIN_LISTED
    # Not on any list; only the price-pin plus flat-30d heuristic catches it.
    assert _reason(classified, "newbuck") == ExclusionReason.STABLECOIN_HEURISTIC


def test_dollar_priced_altcoin_survives_the_heuristic(universe_frame: pd.DataFrame) -> None:
    """A $1.01 coin that moved 25% in a month is not a peg."""
    classified = _classify(universe_frame)
    assert _reason(classified, "dollar-lookalike") == ExclusionReason.NONE


def test_derivatives_are_excluded_by_name_and_by_id(universe_frame: pd.DataFrame) -> None:
    classified = _classify(universe_frame)
    assert _reason(classified, "staked-ether") == ExclusionReason.DERIVATIVE_LISTED
    assert _reason(classified, "wrapped-bitcoin") == ExclusionReason.DERIVATIVE_LISTED
    # "Rocket Pool ETH" says nothing about staking; only the id list catches it.
    assert _reason(classified, "rocket-pool-eth") == ExclusionReason.DERIVATIVE_LISTED


def test_liquidity_floors_apply_in_precedence_order(universe_frame: pd.DataFrame) -> None:
    classified = _classify(universe_frame)
    assert _reason(classified, "tiny-coin") == ExclusionReason.BELOW_MARKET_CAP_FLOOR
    assert _reason(classified, "illiquid-coin") == ExclusionReason.BELOW_VOLUME_FLOOR
    assert _reason(classified, "sleepy-coin") == ExclusionReason.BELOW_TURNOVER_FLOOR


def test_missing_price_is_excluded(universe_frame: pd.DataFrame) -> None:
    classified = _classify(universe_frame)
    assert _reason(classified, "ghost-token") == ExclusionReason.MISSING_PRICE


def test_survivors_are_the_expected_set(universe_frame: pd.DataFrame) -> None:
    kept = _apply(universe_frame)
    assert kept["coin_id"].tolist() == [
        "bitcoin",
        "cardano",
        "chainlink",
        "dogecoin",
        "dollar-lookalike",
        "ethereum",
        "inflato",
        "solana",
    ]
    assert kept.index.tolist() == list(range(len(kept)))


def test_extra_config_lists_are_honoured(universe_frame: pd.DataFrame) -> None:
    kept = _apply(
        universe_frame,
        extra_stablecoin_symbols=frozenset({"DLK"}),
        extra_excluded_coin_ids=frozenset({"inflato"}),
    )
    assert "dollar-lookalike" not in kept["coin_id"].tolist()
    assert "inflato" not in kept["coin_id"].tolist()


def test_classification_is_deterministic(universe_frame: pd.DataFrame) -> None:
    shuffled = universe_frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    first = _apply(universe_frame)
    second = _apply(shuffled)
    pd.testing.assert_frame_equal(first, second)


def test_summary_counts_every_exclusion(universe_frame: pd.DataFrame) -> None:
    classified = _classify(universe_frame)
    summary = exclusion_summary(classified)
    assert int(summary["coins"].sum()) == int(classified["excluded"].sum())
