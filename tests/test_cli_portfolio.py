from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from screener import cli
from screener.data.history import HistoryStore
from screener.data.snapshots import SnapshotStore

runner = CliRunner()

SIGNAL_DAY = date(2025, 3, 3)  # Monday
FILL_DAY = date(2025, 3, 10)  # the following Monday

SCREEN_YAML = """
source:
  provider: coingecko
universe:
  top_n: 50
  min_market_cap_usd: 50000000
  min_volume_24h_usd: 5000000
  min_turnover: 0.01
default_preset: balanced
presets:
  balanced:
    momentum_30d: 0.5
    liquidity: 0.3
    volatility: 0.2
"""

PORTFOLIO_YAML = """
initial_capital: 1000.0
max_positions: 4
min_position_usd: 50.0
sizing: equal_weight
rebalance: weekly
max_weight_per_position: 0.3
cash_buffer: 0.02
entry_rank_threshold: 4
exit_rank_threshold: 6
fee_bps: 10
slippage_bps: 30
"""


@pytest.fixture
def portfolio_env(tmp_path: Path, universe_frame: pd.DataFrame) -> dict[str, Path]:
    screen_path = tmp_path / "screen.yaml"
    screen_path.write_text(SCREEN_YAML)
    portfolio_path = tmp_path / "portfolio.yaml"
    portfolio_path.write_text(PORTFOLIO_YAML)

    data_dir = tmp_path / "data"
    snapshots = SnapshotStore(data_dir / "snapshots")
    snapshots.write(universe_frame, SIGNAL_DAY, source="fixture")

    # The fill snapshot is the same universe with every price 10% higher, so the
    # book has something to mark against.
    later = universe_frame.copy()
    later["current_price"] = pd.to_numeric(later["current_price"], errors="coerce") * 1.1
    snapshots.write(later, FILL_DAY, source="fixture")

    rng = np.random.default_rng(11)
    history = HistoryStore(data_dir / "history")
    for coin_id in sorted(universe_frame["coin_id"].astype(str)):
        steps = rng.normal(0.0, 0.03, size=120)
        prices = 100.0 * np.exp(np.cumsum(steps))
        index = [FILL_DAY - timedelta(days=offset) for offset in range(len(prices) - 1, -1, -1)]
        history.write(
            coin_id,
            pd.DataFrame({"date": index, "price": prices, "market_cap": 1e9, "volume": 1e7}),
        )

    return {"screen": screen_path, "portfolio": portfolio_path, "data": data_dir, "root": tmp_path}


def _papertrade(env: dict[str, Path], *extra: str) -> Any:
    return runner.invoke(
        cli.app,
        [
            "papertrade",
            "-c",
            str(env["screen"]),
            "-p",
            str(env["portfolio"]),
            "--data-dir",
            str(env["data"]),
            *extra,
        ],
    )


def test_papertrade_executes_and_persists(portfolio_env: dict[str, Path]) -> None:
    result = _papertrade(portfolio_env)
    assert result.exit_code == 0, result.output

    state_path = portfolio_env["data"] / "portfolio" / "state.json"
    trades_path = portfolio_env["data"] / "portfolio" / "trades.csv"
    assert state_path.exists()
    assert trades_path.exists()

    state = json.loads(state_path.read_text())
    assert state["last_signal_date"] == SIGNAL_DAY.isoformat()
    assert state["last_rebalance"] == FILL_DAY.isoformat()
    assert len(state["positions"]) == 4
    # Cash plus positions must equal the recorded equity.
    equity = state["equity_curve"][FILL_DAY.isoformat()]
    holdings = sum(
        position["units"] * state["last_prices"][position["coin_id"]]
        for position in state["positions"]
    )
    assert equity == pytest.approx(state["cash"] + holdings)


def test_papertrade_fills_at_the_later_snapshot_price(
    portfolio_env: dict[str, Path], universe_frame: pd.DataFrame
) -> None:
    """The signal is the older snapshot; the fill must use the newer prices."""
    _papertrade(portfolio_env)
    state = json.loads((portfolio_env["data"] / "portfolio" / "state.json").read_text())
    trades = pd.read_csv(portfolio_env["data"] / "portfolio" / "trades.csv")

    prices_at_signal = {
        str(coin_id): float(price)
        for coin_id, price in zip(
            universe_frame["coin_id"].astype(str),
            pd.to_numeric(universe_frame["current_price"], errors="coerce").to_numpy(
                dtype="float64"
            ),
            strict=True,
        )
    }
    for coin_id, intended in zip(
        trades["coin_id"].astype(str), trades["intended_price"].astype("float64"), strict=True
    ):
        assert intended == pytest.approx(prices_at_signal[str(coin_id)] * 1.1)
    assert state["last_signal_date"] < state["last_rebalance"]


def test_dry_run_changes_nothing(portfolio_env: dict[str, Path]) -> None:
    result = _papertrade(portfolio_env, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert not (portfolio_env["data"] / "portfolio" / "state.json").exists()
    assert not (portfolio_env["data"] / "portfolio" / "trades.csv").exists()


def test_dry_run_and_a_real_run_plan_the_same_trades(portfolio_env: dict[str, Path]) -> None:
    dry = _papertrade(portfolio_env, "--dry-run")
    live = _papertrade(portfolio_env)
    dry_rows = [line for line in dry.output.splitlines() if "entry_top_rank" in line]
    live_rows = [line for line in live.output.splitlines() if "entry_top_rank" in line]
    assert dry_rows == live_rows


def test_a_second_run_on_the_same_day_is_not_due(portfolio_env: dict[str, Path]) -> None:
    _papertrade(portfolio_env)
    result = _papertrade(portfolio_env)
    assert result.exit_code == 0
    assert "No rebalance due" in result.output


def test_force_overrides_the_cadence(portfolio_env: dict[str, Path]) -> None:
    _papertrade(portfolio_env)
    result = _papertrade(portfolio_env, "--force")
    assert result.exit_code == 0, result.output
    assert "Executed" in result.output


def test_one_snapshot_is_never_enough(tmp_path: Path, universe_frame: pd.DataFrame) -> None:
    screen_path = tmp_path / "screen.yaml"
    screen_path.write_text(SCREEN_YAML)
    portfolio_path = tmp_path / "portfolio.yaml"
    portfolio_path.write_text(PORTFOLIO_YAML)
    data_dir = tmp_path / "data"
    SnapshotStore(data_dir / "snapshots").write(universe_frame, SIGNAL_DAY, source="fixture")

    result = runner.invoke(
        cli.app,
        [
            "papertrade",
            "-c",
            str(screen_path),
            "-p",
            str(portfolio_path),
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 0
    assert "at least two snapshots" in result.output


def test_backtest_refuses_without_the_survivorship_acknowledgement(
    portfolio_env: dict[str, Path],
) -> None:
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "-c",
            str(portfolio_env["screen"]),
            "-p",
            str(portfolio_env["portfolio"]),
            "--data-dir",
            str(portfolio_env["data"]),
            "--reports-dir",
            str(portfolio_env["root"] / "reports"),
        ],
    )
    assert result.exit_code == 2
    assert "This backtest is built from the coins" in result.output
    assert "Refusing to run" in result.output


def test_backtest_runs_and_writes_its_outputs(portfolio_env: dict[str, Path]) -> None:
    reports = portfolio_env["root"] / "reports"
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "-c",
            str(portfolio_env["screen"]),
            "-p",
            str(portfolio_env["portfolio"]),
            "--data-dir",
            str(portfolio_env["data"]),
            "--reports-dir",
            str(reports),
            "--from",
            (FILL_DAY - timedelta(days=60)).isoformat(),
            "--to",
            FILL_DAY.isoformat(),
            "--acknowledge-survivorship",
        ],
    )
    assert result.exit_code == 0, result.output
    written = sorted(path.name for path in reports.glob("backtest-*.csv"))
    assert len(written) == 2

    curve = pd.read_csv(next(reports.glob("backtest-equity-*.csv")))
    assert len(curve) == 61
    assert (curve["equity_usd"] > 0).all()
    # The warning is printed on every run, not only when it is refused.
    assert "This backtest is built from the coins" in result.output
