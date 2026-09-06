"""Shared fixtures.

The whole suite runs offline. `_no_network` fails any test that tries to open a
socket, so a future change that reaches for the live API is a test failure rather
than a slow, flaky, rate-limited test run.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from screener.data.coingecko import _markets_to_frame
from screener.data.snapshots import SnapshotStore
from screener.paths import Layout

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("network access is not allowed in the test suite")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def markets_rows() -> list[dict[str, Any]]:
    payload = json.loads((FIXTURES / "markets_sample.json").read_text())
    assert isinstance(payload, list)
    return payload


@pytest.fixture
def universe_frame(markets_rows: list[dict[str, Any]]) -> pd.DataFrame:
    return _markets_to_frame(markets_rows, source="fixture")


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    return Layout(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports").ensure()


@pytest.fixture
def snapshot_store(layout: Layout) -> SnapshotStore:
    return SnapshotStore(layout.snapshots)


@pytest.fixture
def sample_day() -> date:
    return date(2025, 3, 4)


@pytest.fixture
def written_snapshot(
    snapshot_store: SnapshotStore, universe_frame: pd.DataFrame, sample_day: date
) -> Iterator[SnapshotStore]:
    snapshot_store.write(universe_frame, sample_day, source="fixture")
    yield snapshot_store
