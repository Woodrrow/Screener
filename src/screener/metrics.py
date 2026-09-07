"""Performance statistics.

All of these are computed from the daily equity curve rather than from
rebalance-to-rebalance returns, so a drawdown that opened and closed inside a
week is still counted. Annualisation uses 365 days: crypto does not close.

Where a statistic is undefined - no downside days for Sortino, a flat curve for
Sharpe - the result is NaN rather than a large number. A Sharpe of "infinity"
on nine observations is not a good result, it is a missing denominator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365
# Below this many observations the statistics are too noisy to be worth
# reporting as anything other than context.
MIN_OBSERVATIONS = 30


def daily_returns(equity: pd.Series[float]) -> pd.Series[float]:
    series = pd.to_numeric(equity, errors="coerce").astype("float64").sort_index()
    return series.pct_change().dropna()


def total_return(equity: pd.Series[float]) -> float:
    if len(equity) < 2:
        return float("nan")
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    return end / start - 1.0 if start > 0 else float("nan")


def annualised_return(equity: pd.Series[float]) -> float:
    if len(equity) < 2:
        return float("nan")
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    if start <= 0 or end <= 0:
        return float("nan")
    return float((end / start) ** (DAYS_PER_YEAR / days) - 1.0)


def volatility(equity: pd.Series[float]) -> float:
    returns = daily_returns(equity)
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(DAYS_PER_YEAR))


def sharpe(equity: pd.Series[float], risk_free_rate: float = 0.0) -> float:
    """Excess return over volatility. Risk-free defaults to zero.

    Zero rather than a T-bill yield: this is a long-only crypto book held in
    stablecoin-free cash, and picking a rate would imply a precision the rest of
    the model does not have.
    """
    returns = daily_returns(equity)
    if len(returns) < 2:
        return float("nan")
    excess = returns - (risk_free_rate / DAYS_PER_YEAR)
    deviation = float(excess.std(ddof=1))
    if deviation == 0:
        return float("nan")
    return float(excess.mean() / deviation * np.sqrt(DAYS_PER_YEAR))


def sortino(equity: pd.Series[float], risk_free_rate: float = 0.0) -> float:
    returns = daily_returns(equity)
    if len(returns) < 2:
        return float("nan")
    excess = returns - (risk_free_rate / DAYS_PER_YEAR)
    downside = excess[excess < 0]
    if downside.empty:
        return float("nan")
    deviation = float(np.sqrt((downside**2).mean()))
    if deviation == 0:
        return float("nan")
    return float(excess.mean() / deviation * np.sqrt(DAYS_PER_YEAR))


def drawdown_series(equity: pd.Series[float]) -> pd.Series[float]:
    series = pd.to_numeric(equity, errors="coerce").astype("float64").sort_index()
    peak = series.cummax()
    return series / peak - 1.0


def max_drawdown(equity: pd.Series[float]) -> float:
    if equity.empty:
        return float("nan")
    return float(drawdown_series(equity).min())


def calmar(equity: pd.Series[float]) -> float:
    depth = max_drawdown(equity)
    if not np.isfinite(depth) or depth == 0:
        return float("nan")
    return annualised_return(equity) / abs(depth)


def rolling_sharpe(equity: pd.Series[float], window: int = 30) -> pd.Series[float]:
    returns = daily_returns(equity)
    if len(returns) < window:
        return pd.Series(dtype="float64")
    mean = returns.rolling(window).mean()
    deviation = returns.rolling(window).std(ddof=1)
    rolled = (mean / deviation.replace(0.0, np.nan)) * np.sqrt(DAYS_PER_YEAR)
    return pd.Series(rolled.dropna(), dtype="float64")


def correlation(equity: pd.Series[float], benchmark: pd.Series[float]) -> float:
    left, right = daily_returns(equity), daily_returns(benchmark)
    joined = pd.concat([left, right], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return float("nan")
    matrix = np.corrcoef(joined.iloc[:, 0], joined.iloc[:, 1])
    return float(matrix[0, 1])


# ------------------------------------------------------------------ trade log


def cost_drag(trades: pd.DataFrame, starting_capital: float) -> dict[str, float]:
    if trades.empty:
        return {"fees_usd": 0.0, "slippage_usd": 0.0, "total_usd": 0.0, "pct_of_capital": 0.0}
    fees = float(trades["fee_usd"].sum())
    slippage = float(trades["slippage_usd"].sum())
    total = fees + slippage
    return {
        "fees_usd": fees,
        "slippage_usd": slippage,
        "total_usd": total,
        "pct_of_capital": total / starting_capital if starting_capital > 0 else float("nan"),
    }


def turnover(trades: pd.DataFrame, equity: pd.Series[float]) -> float:
    """Average one-way traded notional per rebalance, as a fraction of equity.

    One-way rather than the two-way convention, because a book that sells one
    position and buys another has turned over one position, not two.
    """
    if trades.empty or equity.empty:
        return 0.0
    traded = trades.groupby("trade_date")["gross_usd"].sum()
    if traded.empty:
        return 0.0
    equity_by_day = {str(day): float(value) for day, value in equity.items()}
    ratios = [
        float(value) / equity_by_day[str(day)]
        for day, value in traded.items()
        if str(day) in equity_by_day and equity_by_day[str(day)] > 0
    ]
    return float(np.mean(ratios)) if ratios else 0.0


@dataclass(frozen=True)
class _TradeRow:
    coin_id: str
    trade_date: date
    action: str
    units: float
    gross_usd: float
    fee_usd: float
    rank: float


def _trade_rows(trades: pd.DataFrame) -> list[_TradeRow]:
    """Pull the log into plain typed records before matching.

    Working from lists rather than the DataFrame keeps the FIFO matcher free of
    per-cell type coercion, which is where this kind of code usually goes wrong.
    """
    dates = [pd.Timestamp(value).date() for value in trades["trade_date"]]
    coin_ids = [str(value) for value in trades["coin_id"]]
    actions = [str(value) for value in trades["action"]]
    units = pd.to_numeric(trades["units"], errors="coerce").to_numpy(dtype="float64")
    gross = pd.to_numeric(trades["gross_usd"], errors="coerce").to_numpy(dtype="float64")
    fees = pd.to_numeric(trades["fee_usd"], errors="coerce").to_numpy(dtype="float64")
    rank_column = (
        trades["rank"] if "rank" in trades.columns else pd.Series(np.nan, index=trades.index)
    )
    ranks = pd.to_numeric(rank_column, errors="coerce").to_numpy(dtype="float64")

    rows = [
        _TradeRow(
            coin_id=coin_ids[index],
            trade_date=dates[index],
            action=actions[index],
            units=float(units[index]),
            gross_usd=float(gross[index]),
            fee_usd=float(fees[index]),
            rank=float(ranks[index]),
        )
        for index in range(len(trades))
    ]
    return sorted(rows, key=lambda row: (row.coin_id, row.trade_date))


@dataclass
class _Lot:
    units: float
    cost: float
    opened: date
    rank: float


ROUND_TRIP_COLUMNS = ["coin_id", "held_days", "pnl_usd", "entry_rank"]


def _round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """Match sales against the buys that opened them, FIFO, per coin."""
    if trades.empty:
        return pd.DataFrame(columns=pd.Index(ROUND_TRIP_COLUMNS))

    closed: list[dict[str, Any]] = []
    lots: dict[str, list[_Lot]] = {}

    for row in _trade_rows(trades):
        book = lots.setdefault(row.coin_id, [])
        if row.action == "buy":
            book.append(
                _Lot(
                    units=row.units,
                    cost=row.gross_usd + row.fee_usd,
                    opened=row.trade_date,
                    rank=row.rank,
                )
            )
            continue
        if row.action != "sell" or row.units <= 0:
            # A write-off has no proceeds and no meaningful holding period; it is
            # already in the trade log, and counting it as a round trip would put
            # a fake -100% into the win rate.
            continue

        proceeds_per_unit = (row.gross_usd - row.fee_usd) / row.units
        remaining = row.units
        while remaining > 1e-12 and book:
            lot = book[0]
            matched = min(remaining, lot.units)
            cost_per_unit = lot.cost / lot.units if lot.units > 0 else 0.0
            closed.append(
                {
                    "coin_id": row.coin_id,
                    "held_days": (row.trade_date - lot.opened).days,
                    "pnl_usd": matched * (proceeds_per_unit - cost_per_unit),
                    "entry_rank": lot.rank,
                }
            )
            lot.units -= matched
            lot.cost -= matched * cost_per_unit
            remaining -= matched
            if lot.units <= 1e-12:
                book.pop(0)

    return pd.DataFrame(closed, columns=pd.Index(ROUND_TRIP_COLUMNS))


def win_rate(trades: pd.DataFrame) -> float:
    closed = _round_trips(trades)
    if closed.empty:
        return float("nan")
    return float((closed["pnl_usd"] > 0).mean())


def average_holding_days(trades: pd.DataFrame) -> float:
    closed = _round_trips(trades)
    if closed.empty:
        return float("nan")
    return float(closed["held_days"].mean())


def return_by_entry_rank_decile(trades: pd.DataFrame, deciles: int = 10) -> pd.DataFrame:
    """Realised P&L grouped by the rank that triggered the entry."""
    closed = _round_trips(trades).dropna(subset=["entry_rank"])
    if closed.empty:
        return pd.DataFrame(columns=pd.Index(["decile", "trades", "mean_pnl_usd", "total_pnl_usd"]))
    ranks = pd.to_numeric(closed["entry_rank"], errors="coerce")
    closed = closed.assign(decile=((ranks - 1) // max(1, int(np.ceil(ranks.max() / deciles)))) + 1)
    return (
        closed.groupby("decile", sort=True)
        .agg(
            trades=("pnl_usd", "size"),
            mean_pnl_usd=("pnl_usd", "mean"),
            total_pnl_usd=("pnl_usd", "sum"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------- the bundle


@dataclass(frozen=True)
class PerformanceSummary:
    label: str
    observations: int
    total_return: float
    annualised_return: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def is_statistically_thin(self) -> bool:
        return self.observations < MIN_OBSERVATIONS

    def as_row(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "observations": self.observations,
            "total_return": self.total_return,
            "annualised_return": self.annualised_return,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            **self.extras,
        }


def summarise(equity: pd.Series[float], label: str) -> PerformanceSummary:
    return PerformanceSummary(
        label=label,
        observations=len(equity),
        total_return=total_return(equity),
        annualised_return=annualised_return(equity),
        volatility=volatility(equity),
        sharpe=sharpe(equity),
        sortino=sortino(equity),
        max_drawdown=max_drawdown(equity),
        calmar=calmar(equity),
    )
