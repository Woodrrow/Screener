"""Canonical column schemas.

Every DataSource normalises to these shapes, so adding a second provider means
writing an adapter and nothing else. The schema version is written into each
snapshot so a later reader can tell whether it is looking at a file it
understands rather than guessing from the columns present.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

SNAPSHOT_SCHEMA_VERSION: Final = 1
HISTORY_SCHEMA_VERSION: Final = 1

# Return columns are stored as FRACTIONS (0.05 == +5%). CoinGecko serves these
# as percentages; the adapter divides by 100 exactly once, at the boundary.
RETURN_COLUMNS: Final[tuple[str, ...]] = (
    "ret_1h",
    "ret_24h",
    "ret_7d",
    "ret_14d",
    "ret_30d",
    "ret_200d",
    "ret_1y",
)

SNAPSHOT_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "snapshot_date",
    "source",
    "coin_id",
    "symbol",
    "name",
    "current_price",
    "market_cap",
    "market_cap_rank",
    "fully_diluted_valuation",
    "total_volume",
    "circulating_supply",
    "total_supply",
    "max_supply",
    "ath",
    "ath_date",
    "atl",
    "atl_date",
    *RETURN_COLUMNS,
    "last_updated",
)

SNAPSHOT_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "current_price",
    "market_cap",
    "fully_diluted_valuation",
    "total_volume",
    "circulating_supply",
    "total_supply",
    "max_supply",
    "ath",
    "atl",
    *RETURN_COLUMNS,
)

HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "price",
    "market_cap",
    "volume",
)


class SchemaError(ValueError):
    """Raised when a stored frame does not match the schema it claims."""


def validate_snapshot(frame: pd.DataFrame) -> None:
    missing = [column for column in SNAPSHOT_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError(f"snapshot is missing columns: {missing}")
    if frame.empty:
        return
    versions = set(frame["schema_version"].unique().tolist())
    if versions != {SNAPSHOT_SCHEMA_VERSION}:
        raise SchemaError(
            f"snapshot schema_version {sorted(versions)} != expected {SNAPSHOT_SCHEMA_VERSION}"
        )
    duplicates = frame["coin_id"].duplicated()
    if bool(duplicates.any()):
        offenders = sorted(frame.loc[duplicates, "coin_id"].unique().tolist())
        raise SchemaError(f"snapshot contains duplicate coin_ids: {offenders}")


def validate_history(frame: pd.DataFrame) -> None:
    missing = [column for column in HISTORY_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError(f"history is missing columns: {missing}")
    if frame.empty:
        return
    if bool(frame["date"].duplicated().any()):
        raise SchemaError("history contains duplicate dates")
    if not frame["date"].is_monotonic_increasing:
        raise SchemaError("history is not sorted by date")
