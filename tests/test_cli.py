from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from screener import cli
from screener.data.snapshots import SnapshotStore

runner = CliRunner()


class StubSource:
    def __init__(self, universe: pd.DataFrame) -> None:
        self.universe = universe
        self.universe_calls = 0
        self.history_calls: list[str] = []

    @property
    def name(self) -> str:
        return "stub"

    def fetch_universe(self, *, top_n: int, vs_currency: str = "usd") -> pd.DataFrame:
        self.universe_calls += 1
        return self.universe.head(top_n)

    def fetch_price_history(
        self, coin_id: str, *, days: int, vs_currency: str = "usd"
    ) -> pd.DataFrame:
        self.history_calls.append(coin_id)
        today = datetime.now(UTC).date()
        rows = [
            {
                "date": today - timedelta(days=offset),
                "price": 100.0 + offset,
                "market_cap": 1e9,
                "volume": 1e7,
            }
            for offset in range(days, 0, -1)
        ]
        return pd.DataFrame(rows)


@pytest.fixture
def cli_env(
    tmp_path: Path, universe_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, StubSource]:
    config_path = tmp_path / "screen.yaml"
    config_path.write_text(
        "source:\n"
        "  provider: coingecko\n"
        "universe:\n"
        "  top_n: 50\n"
        "  min_market_cap_usd: 50000000\n"
        "  min_volume_24h_usd: 5000000\n"
        "  min_turnover: 0.01\n"
        "default_preset: balanced\n"
        "presets:\n"
        "  balanced:\n"
        "    momentum_30d: 0.5\n"
        "    liquidity: 0.3\n"
        "    volatility: 0.2\n"
        "  value:\n"
        "    drawdown_depth: 0.6\n"
        "    dilution: 0.4\n"
    )
    data_dir = tmp_path / "data"
    source = StubSource(universe_frame)
    monkeypatch.setattr(cli, "_build_source", lambda config, layout: source)
    return config_path, data_dir, source


def _run(args: list[str]) -> Any:
    return runner.invoke(cli.app, args)


def test_snapshot_writes_and_reports_exclusions(
    cli_env: tuple[Path, Path, StubSource],
) -> None:
    config_path, data_dir, source = cli_env
    result = _run(["snapshot", "-c", str(config_path), "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    assert source.universe_calls == 1
    assert "8" in result.output  # eight investable coins survive the fixture filters
    assert "stablecoin_listed" in result.output

    today = datetime.now(UTC).date()
    stored = SnapshotStore(data_dir / "snapshots").read(today)
    assert len(stored) == 18
    # The provider stamps its own name at fetch time; the store never rewrites it.
    assert set(stored["source"]) == {"fixture"}


def test_snapshot_rerun_does_not_refetch_or_overwrite(
    cli_env: tuple[Path, Path, StubSource],
) -> None:
    config_path, data_dir, source = cli_env
    _run(["snapshot", "-c", str(config_path), "--data-dir", str(data_dir)])
    result = _run(["snapshot", "-c", str(config_path), "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    assert "already exists" in result.output
    assert source.universe_calls == 1  # the provider was never called a second time


def test_snapshot_force_refetches(cli_env: tuple[Path, Path, StubSource]) -> None:
    config_path, data_dir, source = cli_env
    _run(["snapshot", "-c", str(config_path), "--data-dir", str(data_dir)])
    result = _run(["snapshot", "-c", str(config_path), "--data-dir", str(data_dir), "--force"])

    assert result.exit_code == 0, result.output
    assert source.universe_calls == 2


def test_history_requires_a_snapshot_first(cli_env: tuple[Path, Path, StubSource]) -> None:
    config_path, data_dir, _ = cli_env
    result = _run(["history", "-c", str(config_path), "--data-dir", str(data_dir)])

    assert result.exit_code == 1
    assert "No snapshots yet" in result.output


def test_history_covers_the_investable_set_plus_benchmarks(
    cli_env: tuple[Path, Path, StubSource],
) -> None:
    config_path, data_dir, source = cli_env
    _run(["snapshot", "-c", str(config_path), "--data-dir", str(data_dir)])
    result = _run(["history", "-c", str(config_path), "--data-dir", str(data_dir), "--days", "20"])

    assert result.exit_code == 0, result.output
    # The eight survivors, and bitcoin/ethereum are already among them.
    assert set(source.history_calls) == {
        "bitcoin",
        "cardano",
        "chainlink",
        "dogecoin",
        "dollar-lookalike",
        "ethereum",
        "inflato",
        "solana",
    }
    assert (data_dir / "history" / "bitcoin.parquet").exists()


def test_history_accepts_explicit_coins(cli_env: tuple[Path, Path, StubSource]) -> None:
    config_path, data_dir, source = cli_env
    result = _run(
        [
            "history",
            "-c",
            str(config_path),
            "--data-dir",
            str(data_dir),
            "--days",
            "10",
            "--coin",
            "solana",
        ]
    )

    assert result.exit_code == 0, result.output
    # Benchmarks are always maintained, even when coins are named explicitly.
    assert set(source.history_calls) == {"bitcoin", "ethereum", "solana"}


def test_config_typos_fail_loudly(tmp_path: Path) -> None:
    config_path = tmp_path / "screen.yaml"
    config_path.write_text("universe:\n  top_end: 50\n")
    result = _run(["snapshot", "-c", str(config_path), "--data-dir", str(tmp_path / "data")])

    assert result.exit_code != 0
    assert "top_end" in str(result.exception) or "top_end" in result.output


def test_screen_ranks_and_writes_a_csv(
    cli_env: tuple[Path, Path, StubSource], tmp_path: Path
) -> None:
    config_path, data_dir, _ = cli_env
    _run(["snapshot", "-c", str(config_path), "--data-dir", str(data_dir)])
    output = tmp_path / "ranking.csv"
    result = _run(
        [
            "screen",
            "-c",
            str(config_path),
            "--data-dir",
            str(data_dir),
            "--preset",
            "value",
            "--output",
            str(output),
        ]
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    header = output.read_text().splitlines()[0].split(",")
    assert header[:2] == ["rank", "coin_id"]
    assert "pct_drawdown_depth" in header
    assert "raw_dilution" in header
    assert len(output.read_text().splitlines()) == 9  # header plus eight investable coins


def test_screen_warns_when_history_is_missing(
    cli_env: tuple[Path, Path, StubSource],
) -> None:
    """The balanced preset uses volatility, which needs the history store."""
    config_path, data_dir, _ = cli_env
    _run(["snapshot", "-c", str(config_path), "--data-dir", str(data_dir)])
    result = _run(["screen", "-c", str(config_path), "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    assert "No price history found" in result.output


def test_screen_requires_a_snapshot(cli_env: tuple[Path, Path, StubSource]) -> None:
    config_path, data_dir, _ = cli_env
    result = _run(["screen", "-c", str(config_path), "--data-dir", str(data_dir)])
    assert result.exit_code == 1
    assert "No snapshots yet" in result.output


def test_screen_rejects_an_unknown_snapshot_date(
    cli_env: tuple[Path, Path, StubSource],
) -> None:
    config_path, data_dir, _ = cli_env
    _run(["snapshot", "-c", str(config_path), "--data-dir", str(data_dir)])
    result = _run(
        ["screen", "-c", str(config_path), "--data-dir", str(data_dir), "--as-of", "1999-01-01"]
    )
    assert result.exit_code == 1
    assert "No snapshot for 1999-01-01" in result.output


def test_factors_command_lists_the_registry() -> None:
    result = _run(["factors"])
    assert result.exit_code == 0
    assert "momentum_30d" in result.output
    assert "lower_is_better" in result.output
