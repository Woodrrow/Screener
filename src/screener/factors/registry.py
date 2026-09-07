"""Factor registry.

Factors are looked up by name from config, so adding one means writing a
function and decorating it - the screener never changes. Each factor declares
its direction and its missing-data policy rather than baking a sign into a
weight, which keeps every configured weight positive and summing to 1.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import pandas as pd

from screener.factors.context import FactorContext

FactorFn = Callable[[FactorContext], "pd.Series[float]"]


class Direction(enum.StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class NaPolicy(enum.StrEnum):
    """What a missing raw value scores after ranking.

    WORST is the default everywhere. Scoring a gap as neutral quietly rewards
    coins for having no data, which in this universe correlates with being new,
    thin, or badly indexed - exactly the coins that should not float to the top.
    """

    WORST = "worst"
    NEUTRAL = "neutral"
    BEST = "best"


@dataclass(frozen=True)
class FactorSpec:
    name: str
    fn: FactorFn
    direction: Direction
    na_policy: NaPolicy
    description: str
    needs_history: bool = False


_REGISTRY: Final[dict[str, FactorSpec]] = {}


def register(
    name: str,
    *,
    direction: Direction,
    description: str,
    na_policy: NaPolicy = NaPolicy.WORST,
    needs_history: bool = False,
) -> Callable[[FactorFn], FactorFn]:
    def decorate(fn: FactorFn) -> FactorFn:
        if name in _REGISTRY:
            raise ValueError(f"factor {name!r} is already registered")
        _REGISTRY[name] = FactorSpec(
            name=name,
            fn=fn,
            direction=direction,
            na_policy=na_policy,
            description=description,
            needs_history=needs_history,
        )
        return fn

    return decorate


def get(name: str) -> FactorSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown factor {name!r}; known factors: {available()}") from None


def available() -> list[str]:
    return sorted(_REGISTRY)


def all_specs() -> list[FactorSpec]:
    return [_REGISTRY[name] for name in available()]
