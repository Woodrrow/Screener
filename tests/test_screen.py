from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from screener.config import UniverseConfig
from screener.data.history import HistoryStore
from screener.factors.context import LookaheadError
from screener.paths import Layout
from screener.screen import load_price_panel, run_screen, write_ranking_csv

AS_OF = date(2025, 3, 4)
UNIVERSE = UniverseConfig(
    top_n=250,
    min_market_cap_usd=50_000_000.0,
    min_volume_24h_usd=5_000_000.0,
    min_turnover=0.01,
)
BALANCED = {
    "momentum_30d": 0.20,
    "momentum_200d": 0.15,
    "liquidity": 0.15,
    "dilution": 0.15,
    "trend_health": 0.10,
    "volatility": 0.15,
    "size_tilt": 0.10,
}


def _screen(frame: pd.DataFrame, panel: pd.DataFrame | None = None):  # type: ignore[no-untyped-def]
    return run_screen(
        frame,
        as_of=AS_OF,
        weights=BALANCED,
        preset="balanced",
        universe_config=UNIVERSE,
        price_panel=panel,
    )


def test_ranking_covers_exactly_the_investable_set(universe_frame: pd.DataFrame) -> None:
    result = _screen(universe_frame)

    assert sorted(result.coin_ids) == [
        "bitcoin",
        "cardano",
        "chainlink",
        "dogecoin",
        "dollar-lookalike",
        "ethereum",
        "inflato",
        "solana",
    ]
    assert result.ranking["rank"].tolist() == list(range(1, 9))
    assert len(result.excluded) == 10


def test_every_factor_percentile_is_written_alongside_the_composite(
    universe_frame: pd.DataFrame,
) -> None:
    """A ranking nobody can audit is a ranking nobody should trade."""
    result = _screen(universe_frame)
    for name in BALANCED:
        assert f"pct_{name}" in result.ranking.columns
        assert f"raw_{name}" in result.ranking.columns

    row = result.ranking.iloc[0]
    recomputed = sum(row[f"pct_{name}"] * weight for name, weight in BALANCED.items()) * 100.0
    assert row["composite_score"] == pytest.approx(recomputed)


def test_composite_is_bounded_zero_to_one_hundred(universe_frame: pd.DataFrame) -> None:
    result = _screen(universe_frame)
    assert result.ranking["composite_score"].between(0.0, 100.0).all()


def test_ranking_is_deterministic_under_row_shuffling(universe_frame: pd.DataFrame) -> None:
    shuffled = universe_frame.sample(frac=1.0, random_state=99).reset_index(drop=True)
    pd.testing.assert_frame_equal(_screen(universe_frame).ranking, _screen(shuffled).ranking)


def test_identical_scores_break_ties_on_coin_id() -> None:
    frame = pd.DataFrame(
        [
            {
                "coin_id": coin_id,
                "symbol": coin_id.upper(),
                "name": coin_id,
                "current_price": 10.0,
                "market_cap": 1e9,
                "market_cap_rank": 1,
                "total_volume": 1e8,
                "circulating_supply": 5e7,
                "max_supply": 1e8,
                "ath": 20.0,
                "ret_30d": 0.1,
                "ret_200d": 0.2,
            }
            for coin_id in ("zebra", "alpha", "mango")
        ]
    )
    result = _screen(frame)
    assert result.coin_ids == ["alpha", "mango", "zebra"]


def test_csv_output_is_byte_identical_across_runs(
    universe_frame: pd.DataFrame, tmp_path: Path
) -> None:
    first = write_ranking_csv(_screen(universe_frame), tmp_path / "a.csv")
    second = write_ranking_csv(_screen(universe_frame), tmp_path / "b.csv")
    assert first.read_bytes() == second.read_bytes()


def test_history_backed_factors_change_the_ranking(
    universe_frame: pd.DataFrame, layout: Layout
) -> None:
    """With no history every coin ties on volatility; with history they separate."""
    without = _screen(universe_frame)

    store = HistoryStore(layout.history)
    coin_ids = without.coin_ids
    for offset, coin_id in enumerate(coin_ids):
        # A different amplitude per coin, so realised vol is ordered by offset.
        rows = [
            {
                "date": AS_OF - timedelta(days=day),
                "price": 100.0 + (offset + 1) * (day % 2),
                "market_cap": 1e9,
                "volume": 1e7,
            }
            for day in range(90, -1, -1)
        ]
        store.write(coin_id, pd.DataFrame(rows))

    panel = load_price_panel(store, coin_ids, AS_OF)
    with_history = _screen(universe_frame, panel)

    assert with_history.ranking["pct_volatility"].nunique() > 1
    assert without.ranking["pct_volatility"].nunique() == 1


def test_price_panel_is_truncated_at_as_of(layout: Layout) -> None:
    store = HistoryStore(layout.history)
    rows = [
        {
            "date": AS_OF + timedelta(days=offset),
            "price": 100.0 + offset,
            "market_cap": 1e9,
            "volume": 1e7,
        }
        for offset in (-2, -1, 0, 1, 2)
    ]
    store.write("bitcoin", pd.DataFrame(rows))

    panel = load_price_panel(store, ["bitcoin"], AS_OF)
    assert max(panel.index) == AS_OF


def test_a_panel_running_past_as_of_is_rejected(
    universe_frame: pd.DataFrame, layout: Layout
) -> None:
    panel = pd.DataFrame(
        {"bitcoin": [1.0, 2.0]},
        index=pd.Index([AS_OF, AS_OF + timedelta(days=1)], name="date"),
    )
    with pytest.raises(LookaheadError):
        _screen(universe_frame, panel)


def test_excluded_frame_explains_every_drop(universe_frame: pd.DataFrame) -> None:
    result = _screen(universe_frame)
    assert set(result.excluded.columns) == {"coin_id", "symbol", "name", "exclusion_reason"}
    assert result.excluded["exclusion_reason"].ne("").all()
    assert result.excluded["coin_id"].is_monotonic_increasing
