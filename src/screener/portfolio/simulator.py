"""The rebalance engine.

One engine, two callers: `backtest` replays it over reconstructed history and
`papertrade` runs a single step against today's snapshot. Both go through
`plan` then `execute`, so a dry run is the plan without the execute and is
guaranteed to describe what a real run would do.

The no-lookahead rule is enforced here, not documented here: a plan whose signal
is not strictly older than its execution date raises.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

from screener.config import PortfolioConfig
from screener.errors import LookaheadError
from screener.portfolio.costs import CostModel
from screener.portfolio.ledger import Ledger, TradeRecord
from screener.portfolio.sizing import dollar_targets


class Reason:
    ENTRY = "entry_top_rank"
    ADD = "rebalance_add"
    TRIM = "rebalance_trim"
    HOLD = "hold_within_hysteresis"
    EXIT_RANK = "exit_rank_exceeded"
    EXIT_UNIVERSE = "left_investable_universe"
    EXIT_CAPACITY = "displaced_by_higher_rank"
    FORCED = "force_liquidated_no_price"
    WRITE_OFF = "written_off_no_price_ever"


@dataclass(frozen=True)
class PlannedAction:
    action: str
    coin_id: str
    rank: int | None
    current_usd: float
    target_usd: float
    delta_usd: float
    price: float
    reason: str

    def as_row(self) -> dict[str, object]:
        return {
            "action": self.action,
            "coin_id": self.coin_id,
            "rank": self.rank,
            "current_usd": self.current_usd,
            "target_usd": self.target_usd,
            "delta_usd": self.delta_usd,
            "price": self.price,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RebalancePlan:
    signal_date: date
    execution_date: date
    equity_before: float
    targets: dict[str, float]
    actions: tuple[PlannedAction, ...]

    @property
    def sells(self) -> list[PlannedAction]:
        # A write-off has a delta of -0.0, which is not less than zero, so it has
        # to be named explicitly or the position would silently survive the run.
        return [
            action
            for action in self.actions
            if action.delta_usd < 0 or action.action == "write_off"
        ]

    @property
    def buys(self) -> list[PlannedAction]:
        return [action for action in self.actions if action.delta_usd > 0]

    def to_frame(self) -> pd.DataFrame:
        columns = [
            "action",
            "coin_id",
            "rank",
            "current_usd",
            "target_usd",
            "delta_usd",
            "price",
            "reason",
        ]
        if not self.actions:
            return pd.DataFrame(columns=pd.Index(columns))
        return pd.DataFrame([action.as_row() for action in self.actions], columns=pd.Index(columns))


def _numeric_map(coin_ids: list[str], series: pd.Series[float]) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return {coin_id: float(value) for coin_id, value in zip(coin_ids, values, strict=True)}


def assert_execution_after_signal(signal_date: date, execution_date: date) -> None:
    if execution_date <= signal_date:
        raise LookaheadError(
            f"signal dated {signal_date} cannot be executed at {execution_date}; "
            "execution must be strictly later than the data the signal was built from"
        )


class Simulator:
    def __init__(self, config: PortfolioConfig, costs: CostModel | None = None) -> None:
        self.config = config
        self.costs = costs or CostModel(fee_bps=config.fee_bps, slippage_bps=config.slippage_bps)

    # ------------------------------------------------------------------ plan

    def plan(
        self,
        ledger: Ledger,
        *,
        ranking: pd.DataFrame,
        prices: Mapping[str, float],
        signal_date: date,
        execution_date: date,
    ) -> RebalancePlan:
        assert_execution_after_signal(signal_date, execution_date)
        config = self.config
        # Record today's prices before planning, so the last-known-price fallback
        # used for a force liquidation is populated no matter which caller we
        # are running under.
        ledger.observe_prices(prices)

        coin_ids = [str(value) for value in ranking["coin_id"].astype(str)]
        ranks = {
            coin_id: int(value)
            for coin_id, value in _numeric_map(coin_ids, ranking["rank"]).items()
        }
        scores = _numeric_map(coin_ids, ranking["composite_score"])
        volatilities: dict[str, float] = {}
        if "raw_volatility" in ranking.columns:
            volatilities = _numeric_map(coin_ids, ranking["raw_volatility"])

        equity = ledger.reconcile(prices)
        held = sorted(ledger.positions)
        actions: list[PlannedAction] = []

        # 1. Coins we hold that have no tradeable price today. Absence of a
        #    price is a delisting; absence from the ranking is only a filter.
        forced: list[str] = []
        for coin_id in held:
            if prices.get(coin_id, 0.0) > 0:
                continue
            fallback = ledger.last_prices.get(coin_id, 0.0)
            position = ledger.positions[coin_id]
            forced.append(coin_id)
            actions.append(
                PlannedAction(
                    action="liquidate" if fallback > 0 else "write_off",
                    coin_id=coin_id,
                    rank=ranks.get(coin_id),
                    current_usd=position.units * fallback,
                    target_usd=0.0,
                    delta_usd=-(position.units * fallback),
                    price=fallback,
                    reason=Reason.FORCED if fallback > 0 else Reason.WRITE_OFF,
                )
            )

        tradeable_held = [coin_id for coin_id in held if coin_id not in forced]

        # 2. Hysteresis. A holding survives while it stays inside the exit band,
        #    which is wider than the entry band on purpose.
        keepers = [
            coin_id
            for coin_id in tradeable_held
            if coin_id in ranks and ranks[coin_id] <= config.exit_rank_threshold
        ]
        keepers.sort(key=lambda coin_id: (ranks[coin_id], coin_id))
        # If more survive than we have slots, the worst-ranked lose theirs.
        displaced = keepers[config.max_positions :]
        keepers = keepers[: config.max_positions]

        candidates = [
            coin_id
            for coin_id, rank in sorted(ranks.items(), key=lambda item: (item[1], item[0]))
            if rank <= config.entry_rank_threshold
            and coin_id not in keepers
            and prices.get(coin_id, 0.0) > 0
        ]
        selected = keepers + candidates[: max(0, config.max_positions - len(keepers))]

        targets = dollar_targets(
            selected,
            equity=equity,
            config=config,
            scores=scores,
            volatilities=volatilities,
        )

        # 3. Exits: held, tradeable, and not in the target set.
        for coin_id in tradeable_held:
            if coin_id in targets:
                continue
            price = float(prices[coin_id])
            current = ledger.positions[coin_id].units * price
            if coin_id not in ranks:
                reason = Reason.EXIT_UNIVERSE
            elif coin_id in displaced:
                reason = Reason.EXIT_CAPACITY
            else:
                reason = Reason.EXIT_RANK
            actions.append(
                PlannedAction(
                    action="sell",
                    coin_id=coin_id,
                    rank=ranks.get(coin_id),
                    current_usd=current,
                    target_usd=0.0,
                    delta_usd=-current,
                    price=price,
                    reason=reason,
                )
            )

        # 4. Adjustments and entries.
        for coin_id in sorted(targets):
            price = float(prices[coin_id])
            existing = ledger.positions.get(coin_id)
            current = existing.units * price if existing else 0.0
            target = targets[coin_id]
            delta = target - current

            if existing is None:
                actions.append(
                    PlannedAction(
                        action="buy",
                        coin_id=coin_id,
                        rank=ranks.get(coin_id),
                        current_usd=0.0,
                        target_usd=target,
                        delta_usd=delta,
                        price=price,
                        reason=Reason.ENTRY,
                    )
                )
                continue

            if abs(delta) <= config.rebalance_band * target:
                actions.append(
                    PlannedAction(
                        action="hold",
                        coin_id=coin_id,
                        rank=ranks.get(coin_id),
                        current_usd=current,
                        target_usd=target,
                        delta_usd=0.0,
                        price=price,
                        reason=Reason.HOLD,
                    )
                )
                continue

            actions.append(
                PlannedAction(
                    action="buy" if delta > 0 else "sell",
                    coin_id=coin_id,
                    rank=ranks.get(coin_id),
                    current_usd=current,
                    target_usd=target,
                    delta_usd=delta,
                    price=price,
                    reason=Reason.ADD if delta > 0 else Reason.TRIM,
                )
            )

        actions.sort(key=lambda action: (action.delta_usd >= 0, action.coin_id))
        return RebalancePlan(
            signal_date=signal_date,
            execution_date=execution_date,
            equity_before=equity,
            targets=targets,
            actions=tuple(actions),
        )

    # --------------------------------------------------------------- execute

    def execute(
        self, ledger: Ledger, plan: RebalancePlan, prices: Mapping[str, float]
    ) -> list[TradeRecord]:
        """Sells first, then buys, so the buys are funded by real cash."""
        executed: list[TradeRecord] = []

        for action in sorted(plan.sells, key=lambda item: item.coin_id):
            if action.action == "write_off":
                record = ledger.write_off(
                    action.coin_id,
                    trade_date=plan.execution_date,
                    prices=prices,
                    reason=action.reason,
                )
            else:
                units = min(
                    abs(action.delta_usd) / action.price, ledger.positions[action.coin_id].units
                )
                record = ledger.sell(
                    action.coin_id,
                    units=units,
                    intended_price=action.price,
                    trade_date=plan.execution_date,
                    prices=prices,
                    rank=action.rank,
                    reason=action.reason,
                )
            if record is not None:
                executed.append(record)

        buys = sorted(plan.buys, key=lambda item: (item.rank if item.rank else 10**9, item.coin_id))
        wanted = sum(action.delta_usd for action in buys)
        # Scale every buy by the same factor rather than filling the first few
        # and starving the last: ordering should not decide who gets funded.
        scale = 1.0 if wanted <= ledger.cash else (ledger.cash / wanted if wanted > 0 else 0.0)

        for action in buys:
            record = ledger.buy(
                action.coin_id,
                usd=action.delta_usd * scale,
                intended_price=action.price,
                trade_date=plan.execution_date,
                prices=prices,
                rank=action.rank,
                reason=action.reason,
            )
            if record is not None:
                executed.append(record)

        ledger.reconcile(prices)
        return executed
