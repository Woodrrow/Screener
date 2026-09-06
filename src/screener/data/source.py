"""The provider-agnostic data interface.

The screener depends on this protocol, never on a concrete client, so a second
provider can be dropped in by writing an adapter that returns the canonical
frames in `screener.data.schema`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataSource(Protocol):
    @property
    def name(self) -> str:
        """Short provider identifier, written into every snapshot row."""

    def fetch_universe(self, *, top_n: int, vs_currency: str = "usd") -> pd.DataFrame:
        """Return the top `top_n` coins by market cap in SNAPSHOT_COLUMNS shape.

        The frame carries no `schema_version` or `snapshot_date`; the snapshot
        store stamps those, so a source cannot lie about when it was taken.
        """

    def fetch_price_history(
        self, coin_id: str, *, days: int, vs_currency: str = "usd"
    ) -> pd.DataFrame:
        """Return daily price/market cap/volume in HISTORY_COLUMNS shape.

        The most recent row may be a partial day; the history store drops it
        rather than storing a stub that a later run would have to correct.
        """
