from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from screener.backtest import UniverseBuilder, metadata_from_snapshot
from screener.data.history import HistoryStore
from screener.paths import Layout

START = date(2024, 1, 1)


def _write_history(store: HistoryStore, coin_id: str, prices: list[float]) -> None:
    rows = [
        {
            "date": START + timedelta(days=offset),
            "price": price,
            "market_cap": price * 1_000_000,
            "volume": price * 50_000,
        }
        for offset, price in enumerate(prices)
    ]
    store.write(coin_id, pd.DataFrame(rows))


def _builder(layout: Layout, series: dict[str, list[float]]) -> UniverseBuilder:
    store = HistoryStore(layout.history)
    for coin_id, prices in series.items():
        _write_history(store, coin_id, prices)
    metadata = pd.DataFrame(
        [{"coin_id": coin_id, "symbol": coin_id.upper(), "name": coin_id} for coin_id in series]
    )
    return UniverseBuilder.from_history(store, sorted(series), metadata)


def test_ath_is_a_running_max_not_the_present_day_high(layout: Layout) -> None:
    """The whole point: a 2024 rebalance cannot know about a 2025 peak."""
    prices = [100.0, 150.0, 120.0, 400.0, 380.0]
    builder = _builder(layout, {"coin": prices})

    early = builder.frame_for(START + timedelta(days=2))
    late = builder.frame_for(START + timedelta(days=4))

    assert float(early["ath"].iloc[0]) == 150.0  # not 400
    assert float(late["ath"].iloc[0]) == 400.0


def test_supply_is_left_null_rather_than_backfilled_from_today(layout: Layout) -> None:
    builder = _builder(layout, {"coin": [100.0] * 5})
    frame = builder.frame_for(START + timedelta(days=4))

    assert frame["circulating_supply"].isna().all()
    assert frame["max_supply"].isna().all()


def test_returns_are_computed_from_stored_prices(layout: Layout) -> None:
    prices = [100.0 + offset for offset in range(40)]
    builder = _builder(layout, {"coin": prices})
    as_of = START + timedelta(days=39)
    frame = builder.frame_for(as_of)

    row = frame.iloc[0]
    assert float(row["current_price"]) == 139.0
    assert float(row["ret_7d"]) == (139.0 / 132.0) - 1.0
    assert float(row["ret_30d"]) == (139.0 / 109.0) - 1.0


def test_a_window_longer_than_the_history_returns_nan(layout: Layout) -> None:
    builder = _builder(layout, {"coin": [100.0] * 10})
    frame = builder.frame_for(START + timedelta(days=9))
    assert np.isnan(float(frame["ret_200d"].iloc[0]))
    assert np.isnan(float(frame["ret_1y"].iloc[0]))


def test_a_coin_with_no_print_on_the_day_is_not_investable(layout: Layout) -> None:
    store = HistoryStore(layout.history)
    _write_history(store, "alive", [100.0] * 6)
    _write_history(store, "halted", [100.0] * 3)
    metadata = pd.DataFrame(
        [{"coin_id": coin, "symbol": coin.upper(), "name": coin} for coin in ("alive", "halted")]
    )
    builder = UniverseBuilder.from_history(store, ["alive", "halted"], metadata)

    frame = builder.frame_for(START + timedelta(days=5))
    assert frame["coin_id"].tolist() == ["alive"]


def test_market_cap_rank_is_derived_from_stored_caps(layout: Layout) -> None:
    builder = _builder(layout, {"big": [200.0] * 5, "small": [10.0] * 5})
    frame = builder.frame_for(START + timedelta(days=4))
    ranks = dict(zip(frame["coin_id"], frame["market_cap_rank"], strict=True))
    assert ranks["big"] == 1
    assert ranks["small"] == 2


def test_frames_are_sorted_and_metadata_joined(layout: Layout) -> None:
    builder = _builder(layout, {"zeta": [10.0] * 5, "alpha": [20.0] * 5})
    frame = builder.frame_for(START + timedelta(days=4))
    assert frame["coin_id"].tolist() == ["alpha", "zeta"]
    assert frame["symbol"].tolist() == ["ALPHA", "ZETA"]


def test_a_date_before_any_history_yields_nothing(layout: Layout) -> None:
    builder = _builder(layout, {"coin": [100.0] * 5})
    assert builder.frame_for(START - timedelta(days=1)).empty


def test_metadata_helper_dedupes_on_coin_id() -> None:
    snapshot = pd.DataFrame(
        [
            {"coin_id": "a", "symbol": "A", "name": "Alpha", "market_cap": 1.0},
            {"coin_id": "a", "symbol": "A", "name": "Alpha", "market_cap": 2.0},
        ]
    )
    metadata = metadata_from_snapshot(snapshot)
    assert metadata["coin_id"].tolist() == ["a"]
    assert list(metadata.columns) == ["coin_id", "symbol", "name"]
