"""Configuration models.

`extra="forbid"` everywhere: a typo in a YAML key is a silent behaviour change
otherwise, and a screener that quietly ignores half its config is worse than one
that refuses to start.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from screener import factors
from screener.portfolio.schedule import Cadence


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceConfig(StrictModel):
    provider: Literal["coingecko"] = "coingecko"
    vs_currency: str = "usd"
    rate_per_minute: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=5, ge=1)


class UniverseConfig(StrictModel):
    top_n: int = Field(default=250, gt=0)
    min_market_cap_usd: float = Field(default=50_000_000.0, ge=0)
    min_volume_24h_usd: float = Field(default=5_000_000.0, ge=0)
    # Volume/market cap. A coin that turns over less than 1% of its cap a day
    # cannot absorb even a $125 position without the flat slippage assumption
    # becoming a fiction.
    min_turnover: float = Field(default=0.01, ge=0)
    extra_stablecoin_symbols: tuple[str, ...] = ()
    extra_excluded_coin_ids: tuple[str, ...] = ()


WEIGHT_SUM_TOLERANCE = 1e-9

DEFAULT_PRESETS: dict[str, dict[str, float]] = {
    "balanced": {
        "momentum_30d": 0.20,
        "momentum_200d": 0.15,
        "liquidity": 0.15,
        "dilution": 0.15,
        "trend_health": 0.10,
        "volatility": 0.15,
        "size_tilt": 0.10,
    }
}


class ScreenConfig(StrictModel):
    source: SourceConfig = SourceConfig()
    universe: UniverseConfig = UniverseConfig()
    presets: dict[str, dict[str, float]] = Field(default_factory=lambda: dict(DEFAULT_PRESETS))
    default_preset: str = "balanced"

    @field_validator("presets")
    @classmethod
    def _validate_presets(cls, presets: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        if not presets:
            raise ValueError("at least one preset must be defined")
        known = set(factors.available())
        for name, weights in presets.items():
            if not weights:
                raise ValueError(f"preset {name!r} has no weights")
            unknown = sorted(set(weights) - known)
            if unknown:
                raise ValueError(
                    f"preset {name!r} references unknown factors {unknown}; "
                    f"known factors are {sorted(known)}"
                )
            negative = sorted(key for key, value in weights.items() if value < 0)
            if negative:
                # Penalty factors carry their sign in the factor's direction, so
                # a negative weight here is always a mistake and never a shortcut.
                raise ValueError(
                    f"preset {name!r} has negative weights {negative}; express a penalty "
                    "with a LOWER_IS_BETTER factor, not a negative weight"
                )
            total = sum(weights.values())
            if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
                raise ValueError(f"preset {name!r} weights sum to {total!r}, expected 1.0")
        return presets

    @model_validator(mode="after")
    def _validate_default_preset(self) -> ScreenConfig:
        if self.default_preset not in self.presets:
            raise ValueError(
                f"default_preset {self.default_preset!r} is not one of {sorted(self.presets)}"
            )
        return self

    def weights_for(self, preset: str | None) -> tuple[str, dict[str, float]]:
        name = preset or self.default_preset
        if name not in self.presets:
            raise KeyError(f"unknown preset {name!r}; defined presets: {sorted(self.presets)}")
        return name, dict(self.presets[name])


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    loaded = yaml.safe_load(path.read_text())
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
    return loaded


def load_screen_config(path: Path) -> ScreenConfig:
    return ScreenConfig.model_validate(load_yaml(path))


class Sizing(StrEnum):
    EQUAL_WEIGHT = "equal_weight"
    SCORE_WEIGHTED = "score_weighted"
    INVERSE_VOL = "inverse_vol"


class PortfolioConfig(StrictModel):
    initial_capital: float = Field(default=1000.0, gt=0)
    max_positions: int = Field(default=8, gt=0)
    # Below this a position's round-trip costs are a meaningful fraction of any
    # edge it could carry, so the slot is better left in cash.
    min_position_usd: float = Field(default=50.0, ge=0)
    sizing: Sizing = Sizing.EQUAL_WEIGHT
    rebalance: Cadence = Cadence.WEEKLY
    max_weight_per_position: float = Field(default=0.20, gt=0, le=1.0)
    cash_buffer: float = Field(default=0.02, ge=0, lt=1.0)
    entry_rank_threshold: int = Field(default=8, gt=0)
    exit_rank_threshold: int = Field(default=15, gt=0)
    # A no-trade band on existing positions, as a fraction of the target value.
    # Defaults to 0.0: full rebalance to target every period, which is the
    # expensive, pessimistic choice. Raise it to see how much of the drag is
    # churn rather than selection.
    rebalance_band: float = Field(default=0.0, ge=0, lt=1.0)
    fee_bps: float = Field(default=10.0, gt=0)
    slippage_bps: float = Field(default=30.0, gt=0)
    no_shorts: bool = True
    no_leverage: bool = True
    preset: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> PortfolioConfig:
        if self.exit_rank_threshold < self.entry_rank_threshold:
            raise ValueError(
                f"exit_rank_threshold ({self.exit_rank_threshold}) must be at least "
                f"entry_rank_threshold ({self.entry_rank_threshold}); an exit tighter than "
                "the entry inverts the hysteresis and churns the book"
            )
        if not self.no_shorts or not self.no_leverage:
            # There is no short or margin machinery in the ledger, so allowing
            # these would silently produce a long-only result under a name that
            # claims otherwise.
            raise ValueError("no_shorts and no_leverage cannot be disabled")
        capacity = self.max_weight_per_position * self.max_positions
        if capacity < 1.0 - self.cash_buffer:
            raise ValueError(
                f"max_weight_per_position * max_positions = {capacity:.2f} cannot reach "
                f"the {1.0 - self.cash_buffer:.2f} target invested fraction"
            )
        return self


def load_portfolio_config(path: Path) -> PortfolioConfig:
    return PortfolioConfig.model_validate(load_yaml(path))
