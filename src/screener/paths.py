"""Filesystem layout.

Everything is resolved from a single data root so tests can point the whole
stack at a tmp_path without monkeypatching module globals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path("data")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_CONFIG_DIR = Path("config")


@dataclass(frozen=True)
class Layout:
    data_dir: Path
    reports_dir: Path = DEFAULT_REPORTS_DIR

    @property
    def snapshots(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def history(self) -> Path:
        return self.data_dir / "history"

    @property
    def portfolio(self) -> Path:
        return self.data_dir / "portfolio"

    @property
    def cache(self) -> Path:
        return self.data_dir / "cache"

    def ensure(self) -> Layout:
        for path in (self.snapshots, self.history, self.portfolio, self.cache, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self


def default_layout(data_dir: Path | None = None, reports_dir: Path | None = None) -> Layout:
    return Layout(
        data_dir=data_dir if data_dir is not None else DEFAULT_DATA_DIR,
        reports_dir=reports_dir if reports_dir is not None else DEFAULT_REPORTS_DIR,
    )
