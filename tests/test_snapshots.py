from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.data.schema import SNAPSHOT_COLUMNS, SNAPSHOT_SCHEMA_VERSION, SchemaError
from screener.data.snapshots import SnapshotStore, WriteStatus


def test_write_stamps_schema_version_and_date(
    snapshot_store: SnapshotStore, universe_frame: pd.DataFrame, sample_day: date
) -> None:
    result = snapshot_store.write(universe_frame, sample_day, source="fixture")
    assert result.status is WriteStatus.WRITTEN
    assert result.path.name == "2025-03-04.parquet"

    stored = snapshot_store.read(sample_day)
    assert list(stored.columns) == list(SNAPSHOT_COLUMNS)
    assert set(stored["schema_version"]) == {SNAPSHOT_SCHEMA_VERSION}
    assert set(stored["snapshot_date"]) == {"2025-03-04"}
    assert set(stored["source"]) == {"fixture"}


def test_rerun_is_idempotent_and_does_not_overwrite(
    snapshot_store: SnapshotStore, universe_frame: pd.DataFrame, sample_day: date
) -> None:
    snapshot_store.write(universe_frame, sample_day, source="fixture")
    original = snapshot_store.read(sample_day)

    mutated = universe_frame.copy()
    mutated.loc[mutated["coin_id"] == "bitcoin", "current_price"] = 1.0

    result = snapshot_store.write(mutated, sample_day, source="fixture")
    assert result.status is WriteStatus.SKIPPED_EXISTS
    pd.testing.assert_frame_equal(snapshot_store.read(sample_day), original)


def test_force_overwrites(
    snapshot_store: SnapshotStore, universe_frame: pd.DataFrame, sample_day: date
) -> None:
    snapshot_store.write(universe_frame, sample_day, source="fixture")
    mutated = universe_frame.copy()
    mutated.loc[mutated["coin_id"] == "bitcoin", "current_price"] = 1.0

    result = snapshot_store.write(mutated, sample_day, source="fixture", force=True)
    assert result.status is WriteStatus.OVERWRITTEN

    stored = snapshot_store.read(sample_day)
    assert float(stored.loc[stored["coin_id"] == "bitcoin", "current_price"].iloc[0]) == 1.0


def test_round_trip_is_deterministic_regardless_of_input_order(
    snapshot_store: SnapshotStore, universe_frame: pd.DataFrame, sample_day: date
) -> None:
    snapshot_store.write(universe_frame, sample_day, source="fixture")
    first = snapshot_store.read(sample_day)

    shuffled = universe_frame.sample(frac=1.0, random_state=11).reset_index(drop=True)
    snapshot_store.write(shuffled, date(2025, 3, 5), source="fixture")
    second = snapshot_store.read(date(2025, 3, 5))

    pd.testing.assert_frame_equal(
        first.drop(columns=["snapshot_date"]), second.drop(columns=["snapshot_date"])
    )


def test_returns_are_stored_as_fractions(written_snapshot: SnapshotStore, sample_day: date) -> None:
    stored = written_snapshot.read(sample_day)
    btc = stored.loc[stored["coin_id"] == "bitcoin"].iloc[0]
    assert float(btc["ret_7d"]) == pytest.approx(0.03)
    assert float(btc["ret_30d"]) == pytest.approx(0.09)


def test_date_navigation(snapshot_store: SnapshotStore, universe_frame: pd.DataFrame) -> None:
    for day in (date(2025, 3, 1), date(2025, 3, 4), date(2025, 3, 11)):
        snapshot_store.write(universe_frame, day, source="fixture")

    assert snapshot_store.available_dates() == [
        date(2025, 3, 1),
        date(2025, 3, 4),
        date(2025, 3, 11),
    ]
    assert snapshot_store.latest_date() == date(2025, 3, 11)

    found = snapshot_store.read_on_or_before(date(2025, 3, 9))
    assert found is not None
    assert found[0] == date(2025, 3, 4)
    assert snapshot_store.read_on_or_before(date(2025, 2, 1)) is None


def test_duplicate_coin_ids_are_rejected(
    snapshot_store: SnapshotStore, universe_frame: pd.DataFrame, sample_day: date
) -> None:
    doubled = pd.concat([universe_frame, universe_frame.head(1)], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate coin_ids"):
        snapshot_store.write(doubled, sample_day, source="fixture")


def test_missing_snapshot_raises(snapshot_store: SnapshotStore) -> None:
    with pytest.raises(FileNotFoundError):
        snapshot_store.read(date(1999, 1, 1))
