"""Configuration models.

`extra="forbid"` everywhere: a typo in a YAML key is a silent behaviour change
otherwise, and a screener that quietly ignores half its config is worse than one
that refuses to start.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


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


class ScreenConfig(StrictModel):
    source: SourceConfig = SourceConfig()
    universe: UniverseConfig = UniverseConfig()


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
