"""Immutable dated snapshots of the universe.

This is the point-in-time guarantee. The screener reads snapshots and never the
live API, so a rank computed for 2025-03-04 uses only what was knowable on
2025-03-04. Snapshots are write-once: re-running the same day is a no-op unless
`--force` is passed, and the no-op is reported rather than silent.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from screener.data.schema import (
    SNAPSHOT_COLUMNS,
    SNAPSHOT_NUMERIC_COLUMNS,
    SNAPSHOT_SCHEMA_VERSION,
    validate_snapshot,
)


class WriteStatus(enum.StrEnum):
    WRITTEN = "written"
    SKIPPED_EXISTS = "skipped_exists"
    OVERWRITTEN = "overwritten"


@dataclass(frozen=True)
class SnapshotWriteResult:
    status: WriteStatus
    path: Path
    rows: int


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, day: date) -> Path:
        return self.root / f"{day.isoformat()}.parquet"

    def exists(self, day: date) -> bool:
        return self.path_for(day).exists()

    def available_dates(self) -> list[date]:
        days: list[date] = []
        for path in self.root.glob("*.parquet"):
            try:
                days.append(date.fromisoformat(path.stem))
            except ValueError:
                continue  # not a snapshot file; ignore rather than fail the run
        return sorted(days)

    def latest_date(self) -> date | None:
        days = self.available_dates()
        return days[-1] if days else None

    def normalise(self, frame: pd.DataFrame, day: date, *, source: str) -> pd.DataFrame:
        out = frame.copy()
        out["schema_version"] = SNAPSHOT_SCHEMA_VERSION
        out["snapshot_date"] = day.isoformat()
        if "source" not in out.columns:
            out["source"] = source
        for column in SNAPSHOT_COLUMNS:
            if column not in out.columns:
                out[column] = pd.NA
        for column in SNAPSHOT_NUMERIC_COLUMNS:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        out["market_cap_rank"] = pd.to_numeric(out["market_cap_rank"], errors="coerce").astype(
            "Int64"
        )
        out["schema_version"] = out["schema_version"].astype("int32")
        out = out.loc[:, list(SNAPSHOT_COLUMNS)]
        # Sort by coin_id, not by rank: rank has ties and NaNs, coin_id is unique.
        out = out.sort_values("coin_id", kind="mergesort").reset_index(drop=True)
        validate_snapshot(out)
        return out

    def write(
        self, frame: pd.DataFrame, day: date, *, source: str, force: bool = False
    ) -> SnapshotWriteResult:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(day)
        existed = path.exists()
        if existed and not force:
            return SnapshotWriteResult(WriteStatus.SKIPPED_EXISTS, path, len(self.read(day)))

        out = self.normalise(frame, day, source=source)
        out.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        status = WriteStatus.OVERWRITTEN if existed else WriteStatus.WRITTEN
        return SnapshotWriteResult(status, path, len(out))

    def read(self, day: date) -> pd.DataFrame:
        path = self.path_for(day)
        if not path.exists():
            raise FileNotFoundError(f"no snapshot for {day.isoformat()} at {path}")
        frame = pd.read_parquet(path)
        validate_snapshot(frame)
        return frame.sort_values("coin_id", kind="mergesort").reset_index(drop=True)

    def read_latest(self) -> tuple[date, pd.DataFrame]:
        day = self.latest_date()
        if day is None:
            raise FileNotFoundError(f"no snapshots in {self.root}")
        return day, self.read(day)

    def read_on_or_before(self, day: date) -> tuple[date, pd.DataFrame] | None:
        """The most recent snapshot at or before `day`.

        The simulator uses this to resolve 'the signal that was knowable at t'
        without assuming a snapshot exists for every calendar day.
        """
        candidates = [candidate for candidate in self.available_dates() if candidate <= day]
        if not candidates:
            return None
        chosen = candidates[-1]
        return chosen, self.read(chosen)
