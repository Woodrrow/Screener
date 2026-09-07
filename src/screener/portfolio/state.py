"""Persisted paper-trading state.

The forward record is the point of the whole tool, so it is written as JSON with
sorted keys and a schema version: readable in a diff, reviewable in a pull
request, and safe to append to week after week.

A config fingerprint travels with the state. Changing the rules mid-track-record
is legitimate, but doing it silently would make the resulting equity curve
meaningless, so a mismatch is surfaced rather than swallowed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from screener.config import PortfolioConfig, UniverseConfig
from screener.portfolio.costs import CostModel
from screener.portfolio.ledger import Ledger, Position, TradeRecord

STATE_SCHEMA_VERSION = 1


def config_fingerprint(
    config: PortfolioConfig,
    *,
    universe: UniverseConfig | None = None,
    weights: Mapping[str, float] | None = None,
) -> str:
    """Hash of everything that defines the strategy, not just the sizing rules.

    The universe floors and the factor weights decide what gets bought every bit
    as much as max_positions does - a market cap floor change can swap the entire
    book. Hashing only PortfolioConfig would let a screen.yaml edit split a track
    record in half without anybody being told.
    """
    payload = json.dumps(
        {
            "portfolio": config.model_dump(mode="json"),
            "universe": None if universe is None else universe.model_dump(mode="json"),
            "weights": None if weights is None else dict(sorted(weights.items())),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class PortfolioState:
    initial_capital: float
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    last_prices: dict[str, float] = field(default_factory=dict)
    last_rebalance: date | None = None
    last_signal_date: date | None = None
    equity_curve: dict[date, float] = field(default_factory=dict)
    config_fingerprint: str = ""
    schema_version: int = STATE_SCHEMA_VERSION

    # ------------------------------------------------------------ conversion

    @classmethod
    def fresh(cls, config: PortfolioConfig, fingerprint: str = "") -> PortfolioState:
        return cls(
            initial_capital=config.initial_capital,
            cash=config.initial_capital,
            config_fingerprint=fingerprint or config_fingerprint(config),
        )

    def to_ledger(self, costs: CostModel) -> Ledger:
        return Ledger(
            cash=self.cash,
            costs=costs,
            positions=dict(self.positions),
            last_prices=dict(self.last_prices),
        )

    def absorb(self, ledger: Ledger) -> None:
        self.cash = ledger.cash
        self.positions = dict(ledger.positions)
        self.last_prices = dict(ledger.last_prices)

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_fingerprint": self.config_fingerprint,
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "last_rebalance": None
            if self.last_rebalance is None
            else self.last_rebalance.isoformat(),
            "last_signal_date": (
                None if self.last_signal_date is None else self.last_signal_date.isoformat()
            ),
            "positions": [
                {
                    "coin_id": position.coin_id,
                    "units": position.units,
                    "cost_basis_usd": position.cost_basis_usd,
                    "opened_on": position.opened_on.isoformat(),
                }
                for _, position in sorted(self.positions.items())
            ],
            "last_prices": {key: self.last_prices[key] for key in sorted(self.last_prices)},
            "equity_curve": {
                day.isoformat(): self.equity_curve[day] for day in sorted(self.equity_curve)
            },
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> PortfolioState:
        version = int(payload.get("schema_version", 0))
        if version != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"state schema_version {version} != expected {STATE_SCHEMA_VERSION}; "
                "refusing to guess at an older layout"
            )
        positions = {
            str(entry["coin_id"]): Position(
                coin_id=str(entry["coin_id"]),
                units=float(entry["units"]),
                cost_basis_usd=float(entry["cost_basis_usd"]),
                opened_on=date.fromisoformat(str(entry["opened_on"])),
            )
            for entry in payload.get("positions", [])
        }
        return cls(
            initial_capital=float(payload["initial_capital"]),
            cash=float(payload["cash"]),
            positions=positions,
            last_prices={
                str(key): float(value)
                for key, value in dict(payload.get("last_prices", {})).items()
            },
            last_rebalance=(
                date.fromisoformat(payload["last_rebalance"])
                if payload.get("last_rebalance")
                else None
            ),
            last_signal_date=(
                date.fromisoformat(payload["last_signal_date"])
                if payload.get("last_signal_date")
                else None
            ),
            equity_curve={
                date.fromisoformat(key): float(value)
                for key, value in dict(payload.get("equity_curve", {})).items()
            },
            config_fingerprint=str(payload.get("config_fingerprint", "")),
            schema_version=version,
        )


class StateStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def trades_path(self) -> Path:
        return self.root / "trades.csv"

    def exists(self) -> bool:
        return self.state_path.exists()

    def load(self) -> PortfolioState | None:
        if not self.state_path.exists():
            return None
        return PortfolioState.from_json(json.loads(self.state_path.read_text()))

    def save(self, state: PortfolioState) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state.as_json(), indent=2, sort_keys=True) + "\n"
        self.state_path.write_text(payload)
        return self.state_path

    def append_trades(self, records: list[TradeRecord]) -> Path | None:
        """Append-only. The log is the audit trail; it is never rewritten."""
        if not records:
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([record.as_row() for record in records])
        header = not self.trades_path.exists()
        with self.trades_path.open("a", newline="") as handle:
            frame.to_csv(handle, index=False, header=header, lineterminator="\n")
        return self.trades_path

    def read_trades(self) -> pd.DataFrame:
        if not self.trades_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.trades_path)
