"""Daily mark-to-market loop shared by the backtest and the paper trader.

Positions are marked every day, not only on rebalance dates, so the equity curve
and every statistic derived from it (drawdown, rolling Sharpe) reflect what the
book was actually worth rather than a sampled-weekly approximation that hides
the worst days.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from screener.config import PortfolioConfig
from screener.portfolio.costs import CostModel
from screener.portfolio.ledger import Ledger, TradeRecord
from screener.portfolio.schedule import Cadence, rebalance_dates
from screener.portfolio.simulator import RebalancePlan, Simulator

# Given a signal date, return the ranking that was knowable on that date, or
# None when there is not enough history to rank anything yet.
RankingProvider = Callable[[date], pd.DataFrame | None]


@dataclass
class SimulationResult:
    equity_curve: pd.Series[float]
    trades: pd.DataFrame
    holdings: pd.DataFrame
    plans: list[RebalancePlan] = field(default_factory=list)
    skipped_rebalances: list[date] = field(default_factory=list)

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else 0.0


def prices_on(panel: pd.DataFrame, day: date) -> dict[str, float]:
    """Prices observed on `day`, excluding coins with no print.

    No forward fill: a coin with no price today is a coin that cannot be traded
    today, and pretending otherwise is how a backtest ends up transacting in
    something that had already stopped trading.
    """
    if panel.empty or day not in panel.index:
        return {}
    values = panel.loc[[day]].astype("float64").to_numpy()[0]
    return {
        str(coin_id): float(value)
        for coin_id, value in zip(panel.columns, values, strict=True)
        if np.isfinite(value) and value > 0
    }


def run_simulation(
    *,
    price_panel: pd.DataFrame,
    ranking_for: RankingProvider,
    config: PortfolioConfig,
    start: date,
    end: date,
    costs: CostModel | None = None,
    initial_capital: float | None = None,
) -> SimulationResult:
    simulator = Simulator(config, costs)
    ledger = Ledger(
        cash=initial_capital if initial_capital is not None else config.initial_capital,
        costs=simulator.costs,
    )
    scheduled = set(rebalance_dates(start, end, Cadence(config.rebalance)))

    equity: dict[date, float] = {}
    plans: list[RebalancePlan] = []
    skipped: list[date] = []

    day = start
    while day <= end:
        prices = prices_on(price_panel, day)
        ledger.observe_prices(prices)

        if day in scheduled:
            # The signal may only use data strictly older than the fill.
            signal_date = day - timedelta(days=1)
            ranking = ranking_for(signal_date)
            if ranking is None or ranking.empty:
                skipped.append(day)
            else:
                plan = simulator.plan(
                    ledger,
                    ranking=ranking,
                    prices=prices,
                    signal_date=signal_date,
                    execution_date=day,
                )
                simulator.execute(ledger, plan, prices)
                plans.append(plan)

        equity[day] = ledger.reconcile(prices)
        day += timedelta(days=1)

    curve = pd.Series(equity, dtype="float64").sort_index()
    curve.index.name = "date"
    return SimulationResult(
        equity_curve=curve,
        trades=ledger.trade_log(),
        holdings=ledger.holdings(prices_on(price_panel, end)),
        plans=plans,
        skipped_rebalances=skipped,
    )


def step_once(
    ledger: Ledger,
    *,
    config: PortfolioConfig,
    ranking: pd.DataFrame,
    prices: Mapping[str, float],
    signal_date: date,
    execution_date: date,
    costs: CostModel | None = None,
    dry_run: bool = False,
) -> tuple[RebalancePlan, list[TradeRecord]]:
    """One rebalance, for the paper trader.

    A dry run builds the identical plan and stops before executing it, so what
    it prints is what a live run would do - not a separate code path that could
    drift from the real one.
    """
    simulator = Simulator(config, costs)
    plan = simulator.plan(
        ledger,
        ranking=ranking,
        prices=prices,
        signal_date=signal_date,
        execution_date=execution_date,
    )
    if dry_run:
        return plan, []
    return plan, simulator.execute(ledger, plan, prices)
