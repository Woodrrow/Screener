"""Universe exclusions.

Snapshots store the raw pull. Filtering happens here, at read time, from config -
so revising the stablecoin list or a liquidity floor re-screens history without
refetching anything, and a snapshot never bakes in the opinions of the version of
the code that took it.

Every exclusion carries a machine-readable reason so a ranking can be audited
back to "why isn't X in here".
"""

from __future__ import annotations

import enum
import re
from typing import Final, cast

import pandas as pd


class ExclusionReason(enum.StrEnum):
    NONE = ""
    MISSING_PRICE = "missing_price"
    MISSING_MARKET_CAP = "missing_market_cap"
    STABLECOIN_LISTED = "stablecoin_listed"
    STABLECOIN_HEURISTIC = "stablecoin_heuristic"
    PEGGED_COMMODITY = "pegged_commodity"
    DERIVATIVE_LISTED = "derivative_listed"
    DERIVATIVE_NAME = "derivative_name"
    BELOW_MARKET_CAP_FLOOR = "below_market_cap_floor"
    BELOW_VOLUME_FLOOR = "below_volume_floor"
    BELOW_TURNOVER_FLOOR = "below_turnover_floor"


# Dollar-pegged tokens. A stablecoin has no price factor to speak of: its
# momentum is noise around a peg and its drawdown is a depeg, so including one
# would hand the momentum factors a coin that structurally cannot trend.
STABLECOIN_SYMBOLS: Final[frozenset[str]] = frozenset(
    {
        "USDT",
        "USDC",
        "DAI",
        "USDE",
        "SUSDE",
        "FDUSD",
        "USDS",
        "SUSDS",
        "PYUSD",
        "TUSD",
        "USDD",
        "FRAX",
        "LUSD",
        "GUSD",
        "USDP",
        "USD1",
        "RLUSD",
        "BUSD",
        "USDB",
        "CRVUSD",
        "GHO",
        "SUSD",
        "DOLA",
        "MIM",
        "USDX",
        "USDY",
        "USDL",
        "USDF",
        "USDG",
        "EURC",
        "EURS",
        "EURT",
        "USTC",
        "SDAI",
        "BUIDL",
        "USDO",
        "USDA",
        "DEUSD",
        "USR",
        "FXUSD",
    }
)

# Pegged to a commodity rather than a currency. Same problem as a stablecoin:
# the token tracks an external asset, not a crypto risk premium.
PEGGED_COMMODITY_SYMBOLS: Final[frozenset[str]] = frozenset({"PAXG", "XAUT", "XAUM", "KAU"})

# Wrapped, staked, bridged and restaked tokens are claims on an underlying that
# is already in the universe. Holding both double-counts the same exposure, and
# at eight positions on $1,000 that is a third of the book in one asset.
DERIVATIVE_NAME_PATTERN: Final = re.compile(
    r"\b(?:"
    r"wrapped|staked|restaked|bridged|liquid\s+stak\w*|peg(?:ged)?|"
    r"binance[- ]peg|wormhole|synthetic"
    r")\b",
    re.IGNORECASE,
)

# Derivatives whose display name does not say so.
DERIVATIVE_COIN_IDS: Final[frozenset[str]] = frozenset(
    {
        "rocket-pool-eth",
        "solv-btc",
        "mantle-staked-ether",
        "jito-staked-sol",
        "marinade-staked-sol",
        "binance-staked-sol",
        "bybit-staked-sol",
        "kelp-dao-restaked-eth",
        "renzo-restaked-eth",
        "stakestone-ether",
        "bitcoin-avalanche-bridged-btc-b",
        "msol",
        "bnsol",
        "clbtc",
        "lbtc",
        "ether-fi-staked-eth",
        "swell-ethereum",
        "sky-dollar",
    }
)

DERIVATIVE_SYMBOLS: Final[frozenset[str]] = frozenset(
    {
        "WBTC",
        "WETH",
        "WBETH",
        "WEETH",
        "WSTETH",
        "STETH",
        "RETH",
        "CBETH",
        "METH",
        "EZETH",
        "RSETH",
        "OSETH",
        "SFRXETH",
        "FRXETH",
        "SOLVBTC",
        "LBTC",
        "CLBTC",
        "BTCB",
        "MSOL",
        "JITOSOL",
        "BNSOL",
        "JUPSOL",
        "BSOL",
        "SWETH",
        "ANKRETH",
        "STSOL",
        "WBNB",
        "WAVAX",
        "WMATIC",
        "WPOL",
        "WHYPE",
        "WSOL",
        "STBTC",
        "PUMPBTC",
        "TBTC",
        "CBBTC",
        "EBTC",
        "UNIBTC",
    }
)

STABLE_PRICE_TOLERANCE: Final = 0.03  # within 3% of $1.00
STABLE_MAX_ABS_30D_RETURN: Final = 0.02  # and under 2% absolute 30-day move


def classify(
    frame: pd.DataFrame,
    *,
    min_market_cap_usd: float,
    min_volume_24h_usd: float,
    min_turnover: float,
    extra_stablecoin_symbols: frozenset[str] | None = None,
    extra_excluded_coin_ids: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Return `frame` with `excluded` and `exclusion_reason` columns added.

    Reasons are applied in a fixed precedence so the label for a coin that trips
    several rules is deterministic rather than dict-order dependent.
    """
    out = frame.copy()
    symbols = out["symbol"].astype("string").str.upper().fillna("")
    names = out["name"].astype("string").fillna("")
    coin_ids = out["coin_id"].astype("string").fillna("")
    price = pd.to_numeric(out["current_price"], errors="coerce")
    market_cap = pd.to_numeric(out["market_cap"], errors="coerce")
    volume = pd.to_numeric(out["total_volume"], errors="coerce")
    ret_30d = pd.to_numeric(out["ret_30d"], errors="coerce")

    stable_symbols = STABLECOIN_SYMBOLS | (extra_stablecoin_symbols or frozenset())
    excluded_ids = DERIVATIVE_COIN_IDS | (extra_excluded_coin_ids or frozenset())

    turnover = volume / market_cap

    # The heuristic needs both legs. A coin pinned near $1.00 with a real 30-day
    # move is a coincidence (a $1 altcoin), not a peg; requiring the flat return
    # keeps those in the universe.
    near_peg = (price - 1.0).abs() <= STABLE_PRICE_TOLERANCE
    flat_30d = ret_30d.abs() < STABLE_MAX_ABS_30D_RETURN
    stable_heuristic = (near_peg & flat_30d).fillna(False)

    rules: list[tuple[ExclusionReason, pd.Series[bool]]] = [
        (ExclusionReason.MISSING_PRICE, price.isna() | (price <= 0)),
        (ExclusionReason.MISSING_MARKET_CAP, market_cap.isna() | (market_cap <= 0)),
        (ExclusionReason.STABLECOIN_LISTED, symbols.isin(stable_symbols)),
        (ExclusionReason.PEGGED_COMMODITY, symbols.isin(PEGGED_COMMODITY_SYMBOLS)),
        (ExclusionReason.STABLECOIN_HEURISTIC, stable_heuristic),
        (
            ExclusionReason.DERIVATIVE_LISTED,
            coin_ids.isin(excluded_ids) | symbols.isin(DERIVATIVE_SYMBOLS),
        ),
        (
            ExclusionReason.DERIVATIVE_NAME,
            names.str.contains(DERIVATIVE_NAME_PATTERN, regex=True, na=False),
        ),
        (ExclusionReason.BELOW_MARKET_CAP_FLOOR, market_cap < min_market_cap_usd),
        (ExclusionReason.BELOW_VOLUME_FLOOR, volume.isna() | (volume < min_volume_24h_usd)),
        (ExclusionReason.BELOW_TURNOVER_FLOOR, turnover.isna() | (turnover < min_turnover)),
    ]

    reason = pd.Series(str(ExclusionReason.NONE), index=out.index, dtype="string")
    for label, mask in rules:
        unset = reason == str(ExclusionReason.NONE)
        reason = reason.mask(unset & mask.fillna(False).astype(bool), str(label))

    out["turnover"] = turnover
    out["exclusion_reason"] = reason
    out["excluded"] = reason != str(ExclusionReason.NONE)
    return out


def apply_filters(
    frame: pd.DataFrame,
    *,
    min_market_cap_usd: float,
    min_volume_24h_usd: float,
    min_turnover: float,
    extra_stablecoin_symbols: frozenset[str] | None = None,
    extra_excluded_coin_ids: frozenset[str] | None = None,
) -> pd.DataFrame:
    """`classify` then drop, preserving the deterministic coin_id ordering."""
    classified = classify(
        frame,
        min_market_cap_usd=min_market_cap_usd,
        min_volume_24h_usd=min_volume_24h_usd,
        min_turnover=min_turnover,
        extra_stablecoin_symbols=extra_stablecoin_symbols,
        extra_excluded_coin_ids=extra_excluded_coin_ids,
    )
    kept = classified.loc[~classified["excluded"]]
    return kept.sort_values("coin_id", kind="mergesort").reset_index(drop=True)


def exclusion_summary(classified: pd.DataFrame) -> pd.DataFrame:
    counts = (
        classified.loc[classified["excluded"], "exclusion_reason"]
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="coins")
    )
    ordered = counts.sort_values(["coins", "reason"], ascending=[False, True])
    return cast(pd.DataFrame, ordered.reset_index(drop=True))
