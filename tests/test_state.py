from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from screener.config import PortfolioConfig
from screener.portfolio.costs import CostModel
from screener.portfolio.ledger import Ledger, Position
from screener.portfolio.state import (
    STATE_SCHEMA_VERSION,
    PortfolioState,
    StateStore,
    config_fingerprint,
)

DAY = date(2025, 3, 3)


def _state() -> PortfolioState:
    return PortfolioState(
        initial_capital=1000.0,
        cash=123.45,
        positions={
            "bitcoin": Position("bitcoin", 0.01, 600.0, DAY),
            "solana": Position("solana", 2.0, 280.0, DAY),
        },
        last_prices={"bitcoin": 62000.0, "solana": 145.0},
        last_rebalance=DAY,
        last_signal_date=date(2025, 2, 24),
        equity_curve={DAY: 1010.0, date(2025, 3, 2): 1005.0},
        config_fingerprint="abc123",
    )


def test_round_trips_through_json(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    store.save(_state())
    restored = store.load()

    assert restored is not None
    assert restored.cash == pytest.approx(123.45)
    assert sorted(restored.positions) == ["bitcoin", "solana"]
    assert restored.positions["solana"].units == 2.0
    assert restored.last_rebalance == DAY
    assert restored.equity_curve[date(2025, 3, 2)] == 1005.0


def test_written_json_is_stable_and_sorted(tmp_path: Path) -> None:
    """The state file lands in a git diff every week; it has to be readable."""
    store = StateStore(tmp_path)
    first = store.save(_state()).read_text()
    second = store.save(_state()).read_text()

    assert first == second
    assert first.endswith("\n")
    positions_order = [line for line in first.splitlines() if '"coin_id"' in line]
    assert "bitcoin" in positions_order[0]


def test_an_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    store.save(_state())
    store.state_path.write_text(
        store.state_path.read_text().replace(
            f'"schema_version": {STATE_SCHEMA_VERSION}', '"schema_version": 99'
        )
    )
    with pytest.raises(ValueError, match="schema_version"):
        store.load()


def test_missing_state_loads_as_none(tmp_path: Path) -> None:
    assert StateStore(tmp_path).load() is None


def test_fresh_state_starts_with_the_configured_capital() -> None:
    config = PortfolioConfig()
    state = PortfolioState.fresh(config)
    assert state.cash == config.initial_capital
    assert state.config_fingerprint == config_fingerprint(config)
    assert state.positions == {}


def test_fingerprint_changes_with_the_rules() -> None:
    base = PortfolioConfig()
    changed = PortfolioConfig(exit_rank_threshold=20)
    assert config_fingerprint(base) != config_fingerprint(changed)
    assert config_fingerprint(base) == config_fingerprint(PortfolioConfig())


def test_ledger_round_trip_preserves_the_book() -> None:
    costs = CostModel()
    state = _state()
    ledger = state.to_ledger(costs)
    assert ledger.cash == pytest.approx(123.45)
    assert ledger.positions_value({"bitcoin": 62000.0, "solana": 145.0}) == pytest.approx(910.0)

    ledger.cash = 50.0
    state.absorb(ledger)
    assert state.cash == 50.0


def test_trade_log_is_append_only(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    costs = CostModel()
    ledger = Ledger(cash=1000.0, costs=costs)
    ledger.buy(
        "bitcoin", usd=500.0, intended_price=100.0, trade_date=DAY, prices={"bitcoin": 100.0}
    )
    store.append_trades(ledger.trades)
    first_len = len(store.trades_path.read_text().splitlines())

    ledger.trades.clear()
    ledger.sell(
        "bitcoin", units=1.0, intended_price=101.0, trade_date=DAY, prices={"bitcoin": 101.0}
    )
    store.append_trades(ledger.trades)

    lines = store.trades_path.read_text().splitlines()
    assert len(lines) == first_len + 1  # one header, two data rows
    assert lines[0].startswith("trade_date,")
    assert len(store.read_trades()) == 2


def test_appending_nothing_writes_nothing(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    assert store.append_trades([]) is None
    assert not store.trades_path.exists()
    assert store.read_trades().empty
