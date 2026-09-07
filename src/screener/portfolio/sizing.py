"""Turning a selected set of coins into dollar targets.

Kept separate from the rebalance loop because it is the part with the fiddly
edge cases - a weight cap that has to be redistributed, a position that lands
under the minimum and drags the others up when it is dropped - and those are
easier to test in isolation than through a full simulation.
"""

from __future__ import annotations

from collections.abc import Mapping

from screener.config import PortfolioConfig, Sizing

WEIGHT_TOLERANCE = 1e-12
# Guards the reciprocal in inverse-vol sizing. A coin with a realised vol of
# zero over 60 days is a stale price, not a riskless asset.
MIN_VOLATILITY = 1e-4


def raw_weights(
    selected: list[str],
    *,
    mode: Sizing,
    scores: Mapping[str, float],
    volatilities: Mapping[str, float],
) -> dict[str, float]:
    if not selected:
        return {}
    coins = sorted(selected)
    equal = {coin: 1.0 / len(coins) for coin in coins}

    if mode is Sizing.EQUAL_WEIGHT:
        return equal

    if mode is Sizing.SCORE_WEIGHTED:
        weights = {coin: max(float(scores.get(coin, 0.0)), 0.0) for coin in coins}
        total = sum(weights.values())
        return equal if total <= 0 else {coin: value / total for coin, value in weights.items()}

    # Inverse vol. A coin with no usable volatility estimate is treated as the
    # most volatile name in the set, so missing data shrinks the position rather
    # than quietly earning it the largest one.
    observed = [
        float(volatilities[coin])
        for coin in coins
        if coin in volatilities and float(volatilities[coin]) == float(volatilities[coin])
    ]
    fallback = max(observed) if observed else None
    if fallback is None:
        return equal

    weights = {}
    for coin in coins:
        value = volatilities.get(coin)
        vol = float(value) if value is not None and float(value) == float(value) else fallback
        weights[coin] = 1.0 / max(vol, MIN_VOLATILITY)
    total = sum(weights.values())
    return equal if total <= 0 else {coin: value / total for coin, value in weights.items()}


def cap_weights(weights: Mapping[str, float], cap: float) -> dict[str, float]:
    """Clamp each weight to `cap`, spreading the excess over the rest.

    If every name ends up at the cap the weights sum to less than one; the
    remainder stays in cash rather than being forced into an oversized position.
    """
    current = {coin: float(value) for coin, value in sorted(weights.items())}
    for _ in range(len(current) + 1):
        over = {coin: value for coin, value in current.items() if value > cap + WEIGHT_TOLERANCE}
        if not over:
            break
        excess = sum(value - cap for value in over.values())
        under = {coin: value for coin, value in current.items() if coin not in over}
        for coin in over:
            current[coin] = cap
        if not under:
            break
        pool = sum(under.values())
        for coin, value in under.items():
            share = (value / pool) if pool > 0 else (1.0 / len(under))
            current[coin] = value + excess * share
    return current


def dollar_targets(
    selected: list[str],
    *,
    equity: float,
    config: PortfolioConfig,
    scores: Mapping[str, float],
    volatilities: Mapping[str, float],
) -> dict[str, float]:
    """Dollar target per coin, after the cap, the cash buffer and the minimum.

    A name whose target lands under `min_position_usd` is dropped and the set is
    re-sized, because the alternative - holding a $30 position - pays two lots of
    costs for an exposure too small to matter.
    """
    investable = max(equity, 0.0) * (1.0 - config.cash_buffer)
    remaining = sorted(selected)

    while remaining:
        weights = cap_weights(
            raw_weights(remaining, mode=config.sizing, scores=scores, volatilities=volatilities),
            config.max_weight_per_position,
        )
        targets = {coin: weight * investable for coin, weight in weights.items()}
        below = sorted(
            (value, coin) for coin, value in targets.items() if value < config.min_position_usd
        )
        if not below:
            return targets
        # Drop the smallest offender only: removing it lifts the others, which
        # often brings the rest back above the floor in one pass.
        remaining.remove(below[0][1])

    return {}
