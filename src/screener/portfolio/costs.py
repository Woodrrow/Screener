"""Fees and slippage.

Costs are not optional. At $1,000 with eight positions, a full turnover is eight
round trips of roughly 80bps each; a simulation that omits them is not a
pessimistic estimate of the strategy, it is a different strategy. The model
therefore refuses to be constructed with both legs at zero.

Slippage is a flat fraction of the intended price rather than a modelled order
book. That is a real limitation and it flatters small caps most, since a flat
30bps is generous for a $50m coin and punitive for BTC. It is stated in the
README rather than hidden behind a more complicated-looking model that would be
no better calibrated.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

BPS = 1e-4


class Side(enum.StrEnum):
    BUY = "buy"
    SELL = "sell"


class ZeroCostError(ValueError):
    """Raised when a cost model would let a simulation run free."""


@dataclass(frozen=True)
class Fill:
    side: Side
    units: float
    intended_price: float
    fill_price: float
    gross_usd: float
    fee_usd: float
    slippage_usd: float
    cash_delta: float

    @property
    def total_cost_usd(self) -> float:
        return self.fee_usd + self.slippage_usd


@dataclass(frozen=True)
class CostModel:
    fee_bps: float = 10.0
    slippage_bps: float = 30.0

    def __post_init__(self) -> None:
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs cannot be negative")
        if self.fee_bps <= 0 or self.slippage_bps <= 0:
            raise ZeroCostError(
                "both fee_bps and slippage_bps must be positive; a zero-cost simulation "
                "is not a result this tool will produce"
            )

    @property
    def fee_rate(self) -> float:
        return self.fee_bps * BPS

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps * BPS

    def fill_price(self, side: Side, intended_price: float) -> float:
        """Always against the trader: buys fill high, sells fill low."""
        if intended_price <= 0:
            raise ValueError(f"intended price must be positive, got {intended_price}")
        direction = 1.0 if side is Side.BUY else -1.0
        return intended_price * (1.0 + direction * self.slippage_rate)

    def execute(self, side: Side, units: float, intended_price: float) -> Fill:
        """Price a trade of `units` at `intended_price`.

        `cash_delta` is signed from the portfolio's point of view: negative on a
        buy, positive on a sell, with the fee subtracted in both directions.
        """
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        price = self.fill_price(side, intended_price)
        gross = units * price
        fee = gross * self.fee_rate
        slippage = units * abs(price - intended_price)
        cash_delta = -(gross + fee) if side is Side.BUY else (gross - fee)
        return Fill(
            side=side,
            units=units,
            intended_price=intended_price,
            fill_price=price,
            gross_usd=gross,
            fee_usd=fee,
            slippage_usd=slippage,
            cash_delta=cash_delta,
        )

    def units_affordable(self, cash: float, intended_price: float) -> float:
        """Largest position buyable with `cash`, costs included.

        Solving cash >= units * fill * (1 + fee_rate) rather than sizing on the
        intended price and discovering the shortfall afterwards - that is how a
        simulator ends up with negative cash and a reconciliation failure.
        """
        if cash <= 0:
            return 0.0
        price = self.fill_price(Side.BUY, intended_price)
        return cash / (price * (1.0 + self.fee_rate))
