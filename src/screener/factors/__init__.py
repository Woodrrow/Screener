"""Factor library.

Importing this package registers every shipped factor. New families are added by
writing a module here and importing it below; nothing in the screener changes.
"""

from __future__ import annotations

from screener.factors import (  # noqa: F401 - imported for registration side effects
    drawdown,
    liquidity,
    momentum,
    size,
    supply,
    volatility,
)
from screener.factors.context import FactorContext, LookaheadError
from screener.factors.percentile import composite_score, percentile_rank
from screener.factors.registry import (
    Direction,
    FactorSpec,
    NaPolicy,
    all_specs,
    available,
    get,
    register,
)

__all__ = [
    "Direction",
    "FactorContext",
    "FactorSpec",
    "LookaheadError",
    "NaPolicy",
    "all_specs",
    "available",
    "composite_score",
    "get",
    "percentile_rank",
    "register",
]
