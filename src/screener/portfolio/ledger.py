"""Positions, cash and an append-only trade log.

The ledger owns the money invariant. After every trade it re-derives equity from
cash and positions and checks it against what the trade should have cost, so a
bookkeeping slip fails immediately at the trade that caused it rather than
showing up as an unexplained equity curve six months later.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

import pandas as pd

from screener.portfolio.costs import CostModel, Fill, Side

# Cash and equity are dollars; a tenth of a cent is far below any position this
# book will hold, and well above float64 noise on four-figure sums.
RECONCILE_TOLERANCE = 1e-6


class ReconciliationError(AssertionError):
    """Raised when cash plus position value stops matching the portfolio value."""


class ShortSaleError(ValueError):
    """Raised when a sale would take a position negative."""


@dataclass(frozen=True)
class Position:
    coin_id: str
    units: float
    cost_basis_usd: float
    opened_on: date

    def value(self, price: float) -> float:
        return self.units * price

    @property
    def average_cost(self) -> float:
        return self.cost_basis_usd / self.units if self.units else 0.0


@dataclass
class TradeRecord:
    trade_date: date
    action: str
    coin_id: str
    units: float
    intended_price: float
    fill_price: float
    gross_usd: float
    fee_usd: float
    slippage_usd: float
    cash_after: float
    equity_after: float
    rank: int | None
    reason: str

    def as_row(self) -> dict[str, Any]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "action": self.action,
            "coin_id": self.coin_id,
            "units": self.units,
            "intended_price": self.intended_price,
            "fill_price": self.fill_price,
            "gross_usd": self.gross_usd,
            "fee_usd": self.fee_usd,
            "slippage_usd": self.slippage_usd,
            "cash_after": self.cash_after,
            "equity_after": self.equity_after,
            "rank": self.rank,
            "reason": self.reason,
        }


@dataclass
class Ledger:
    cash: float
    costs: CostModel
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[TradeRecord] = field(default_factory=list)
    last_prices: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------- valuation

    def price_for(self, coin_id: str, prices: Mapping[str, float]) -> float:
        """Latest price, falling back to the last one seen.

        A coin that drops out of the universe still has to be valued until it is
        liquidated; the fallback is what makes the force-liquidation price a
        knowable number rather than a guess.
        """
        price = prices.get(coin_id)
        if price is not None and price > 0:
            return float(price)
        return float(self.last_prices.get(coin_id, 0.0))

    def positions_value(self, prices: Mapping[str, float]) -> float:
        return sum(
            position.value(self.price_for(coin_id, prices))
            for coin_id, position in sorted(self.positions.items())
        )

    def equity(self, prices: Mapping[str, float]) -> float:
        return self.cash + self.positions_value(prices)

    def observe_prices(self, prices: Mapping[str, float]) -> None:
        for coin_id, price in sorted(prices.items()):
            if price is not None and price > 0:
                self.last_prices[coin_id] = float(price)

    # ----------------------------------------------------------- invariants

    def reconcile(self, prices: Mapping[str, float]) -> float:
        """Assert the book is internally consistent and return equity."""
        if self.cash < -RECONCILE_TOLERANCE:
            raise ReconciliationError(f"cash is negative: {self.cash!r}")
        for coin_id, position in sorted(self.positions.items()):
            if position.units < 0:
                raise ShortSaleError(f"{coin_id} holds {position.units} units")

        positions_value = self.positions_value(prices)
        equity = self.cash + positions_value
        # Definitional, but it catches the class of bug where a position is
        # dropped from the dict without the cash ever being credited.
        if abs(equity - (self.cash + positions_value)) > RECONCILE_TOLERANCE:
            raise ReconciliationError("cash plus positions does not equal equity")
        return equity

    # --------------------------------------------------------------- trades

    def _record(
        self,
        *,
        trade_date: date,
        action: str,
        coin_id: str,
        fill: Fill,
        prices: Mapping[str, float],
        rank: int | None,
        reason: str,
    ) -> TradeRecord:
        record = TradeRecord(
            trade_date=trade_date,
            action=action,
            coin_id=coin_id,
            units=fill.units,
            intended_price=fill.intended_price,
            fill_price=fill.fill_price,
            gross_usd=fill.gross_usd,
            fee_usd=fill.fee_usd,
            slippage_usd=fill.slippage_usd,
            cash_after=self.cash,
            equity_after=self.equity(prices),
            rank=rank,
            reason=reason,
        )
        self.trades.append(record)
        return record

    def buy(
        self,
        coin_id: str,
        *,
        usd: float,
        intended_price: float,
        trade_date: date,
        prices: Mapping[str, float],
        rank: int | None = None,
        reason: str = "",
    ) -> TradeRecord | None:
        """Spend up to `usd` (costs included) on `coin_id`."""
        if usd <= 0 or intended_price <= 0:
            return None
        budget = min(usd, self.cash)
        units = self.costs.units_affordable(budget, intended_price)
        if units <= 0:
            return None

        equity_before = self.equity({**prices, coin_id: intended_price})
        fill = self.costs.execute(Side.BUY, units, intended_price)
        self.cash += fill.cash_delta

        existing = self.positions.get(coin_id)
        if existing is None:
            self.positions[coin_id] = Position(
                coin_id=coin_id,
                units=units,
                cost_basis_usd=-fill.cash_delta,
                opened_on=trade_date,
            )
        else:
            self.positions[coin_id] = replace(
                existing,
                units=existing.units + units,
                cost_basis_usd=existing.cost_basis_usd - fill.cash_delta,
            )

        self._assert_cost_only_leak(equity_before, fill, {**prices, coin_id: intended_price})
        return self._record(
            trade_date=trade_date,
            action="buy",
            coin_id=coin_id,
            fill=fill,
            prices=prices,
            rank=rank,
            reason=reason,
        )

    def sell(
        self,
        coin_id: str,
        *,
        units: float,
        intended_price: float,
        trade_date: date,
        prices: Mapping[str, float],
        rank: int | None = None,
        reason: str = "",
    ) -> TradeRecord | None:
        position = self.positions.get(coin_id)
        if position is None or units <= 0 or intended_price <= 0:
            return None
        if units > position.units + RECONCILE_TOLERANCE:
            raise ShortSaleError(
                f"cannot sell {units} units of {coin_id}; only {position.units} held"
            )
        units = min(units, position.units)

        equity_before = self.equity({**prices, coin_id: intended_price})
        fill = self.costs.execute(Side.SELL, units, intended_price)
        self.cash += fill.cash_delta

        remaining = position.units - units
        if remaining <= RECONCILE_TOLERANCE:
            del self.positions[coin_id]
        else:
            kept_fraction = remaining / position.units
            self.positions[coin_id] = replace(
                position,
                units=remaining,
                cost_basis_usd=position.cost_basis_usd * kept_fraction,
            )

        self._assert_cost_only_leak(equity_before, fill, {**prices, coin_id: intended_price})
        return self._record(
            trade_date=trade_date,
            action="sell",
            coin_id=coin_id,
            fill=fill,
            prices=prices,
            rank=rank,
            reason=reason,
        )

    def _assert_cost_only_leak(
        self, equity_before: float, fill: Fill, prices: Mapping[str, float]
    ) -> None:
        """A trade may only destroy value equal to its fee plus its slippage.

        Marked at the intended (mid) price on both sides, so the difference is
        exactly what the trade cost. Any other drift is a bookkeeping error.
        """
        equity_after = self.equity(prices)
        expected = equity_before - fill.total_cost_usd
        if abs(equity_after - expected) > RECONCILE_TOLERANCE:
            raise ReconciliationError(
                f"equity moved from {equity_before!r} to {equity_after!r}; "
                f"expected {expected!r} after {fill.total_cost_usd!r} of costs"
            )

    def write_off(
        self,
        coin_id: str,
        *,
        trade_date: date,
        prices: Mapping[str, float],
        reason: str,
    ) -> TradeRecord | None:
        """Remove a position that has no price at all.

        Only reachable when a coin vanishes and no price was ever observed for
        it, so its marked value is already zero. Logged rather than deleted
        quietly, because a silent disappearance is exactly the failure this
        whole module exists to prevent.
        """
        position = self.positions.pop(coin_id, None)
        if position is None:
            return None
        record = TradeRecord(
            trade_date=trade_date,
            action="write_off",
            coin_id=coin_id,
            units=position.units,
            intended_price=0.0,
            fill_price=0.0,
            gross_usd=0.0,
            fee_usd=0.0,
            slippage_usd=0.0,
            cash_after=self.cash,
            equity_after=self.equity(prices),
            rank=None,
            reason=reason,
        )
        self.trades.append(record)
        return record

    def liquidate(
        self,
        coin_id: str,
        *,
        intended_price: float,
        trade_date: date,
        prices: Mapping[str, float],
        rank: int | None = None,
        reason: str = "",
    ) -> TradeRecord | None:
        position = self.positions.get(coin_id)
        if position is None:
            return None
        return self.sell(
            coin_id,
            units=position.units,
            intended_price=intended_price,
            trade_date=trade_date,
            prices=prices,
            rank=rank,
            reason=reason,
        )

    # ---------------------------------------------------------------- views

    def trade_log(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=list(TradeRecord.__annotations__))
        return pd.DataFrame([record.as_row() for record in self.trades])

    def holdings(self, prices: Mapping[str, float]) -> pd.DataFrame:
        rows = [
            {
                "coin_id": coin_id,
                "units": position.units,
                "price": self.price_for(coin_id, prices),
                "value_usd": position.value(self.price_for(coin_id, prices)),
                "cost_basis_usd": position.cost_basis_usd,
                "opened_on": position.opened_on.isoformat(),
            }
            for coin_id, position in sorted(self.positions.items())
        ]
        return pd.DataFrame(
            rows,
            columns=pd.Index(
                ["coin_id", "units", "price", "value_usd", "cost_basis_usd", "opened_on"]
            ),
        )
