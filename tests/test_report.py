from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from screener.report import ReportInputs, build_markdown, write_report

START = date(2025, 1, 1)


def _curve(length: int, drift: float = 0.001, seed: int = 1) -> pd.Series[float]:
    rng = np.random.default_rng(seed)
    values = 1000.0 * np.exp(np.cumsum(rng.normal(drift, 0.02, size=length)))
    index = [START + timedelta(days=offset) for offset in range(length)]
    return pd.Series(values, index=pd.Index(index, name="date"), dtype="float64")


def _trades() -> pd.DataFrame:
    rows = []
    for index, coin in enumerate(["alpha", "bravo", "charlie"]):
        rows.append(
            {
                "trade_date": (START + timedelta(days=1)).isoformat(),
                "action": "buy",
                "coin_id": coin,
                "units": 1.0,
                "intended_price": 100.0,
                "fill_price": 100.3,
                "gross_usd": 100.3,
                "fee_usd": 0.1,
                "slippage_usd": 0.3,
                "cash_after": 0.0,
                "equity_after": 1000.0,
                "rank": index + 1,
                "reason": "entry_top_rank",
            }
        )
        rows.append(
            {
                "trade_date": (START + timedelta(days=30)).isoformat(),
                "action": "sell",
                "coin_id": coin,
                "units": 1.0,
                "intended_price": 120.0 - index * 30,
                "fill_price": 119.6 - index * 30,
                "gross_usd": 119.6 - index * 30,
                "fee_usd": 0.1,
                "slippage_usd": 0.3,
                "cash_after": 100.0,
                "equity_after": 1010.0,
                "rank": index + 1,
                "reason": "exit_rank_exceeded",
            }
        )
    return pd.DataFrame(rows)


def _inputs(length: int = 120, **overrides: object) -> ReportInputs:
    base: dict[str, object] = {
        "label": "Test run",
        "equity": _curve(length),
        "trades": _trades(),
        "benchmarks": {
            "Buy and hold BTC": _curve(length, seed=2),
            "Buy and hold ETH": _curve(length, seed=3),
            "Equal-weight top 8": _curve(length, seed=4),
        },
        "starting_capital": 1000.0,
        "generated_on": date(2025, 6, 1),
    }
    base.update(overrides)
    return ReportInputs(**base)  # type: ignore[arg-type]


def test_report_writes_markdown_and_four_charts(tmp_path: Path) -> None:
    path = write_report(_inputs(), tmp_path, stem="run")

    assert path.name == "run.md"
    charts = sorted(item.name for item in tmp_path.glob("*.png"))
    assert charts == [
        "run-deciles.png",
        "run-drawdown.png",
        "run-equity.png",
        "run-sharpe.png",
    ]
    assert all((tmp_path / name).stat().st_size > 0 for name in charts)


def test_markdown_reports_all_three_benchmarks(tmp_path: Path) -> None:
    text = build_markdown(_inputs(), {}, tmp_path)
    assert "Buy and hold BTC" in text
    assert "Buy and hold ETH" in text
    assert "Equal-weight top 8" in text
    assert "BTC correlation" in text


def test_markdown_carries_the_full_statistic_set(tmp_path: Path) -> None:
    text = build_markdown(_inputs(), {}, tmp_path)
    for heading in ("Annualised", "Sharpe", "Sortino", "Max DD", "Calmar", "Vol"):
        assert heading in text
    for measure in ("Turnover", "Win rate", "Average holding period", "Total cost drag"):
        assert measure in text


def test_a_thin_sample_is_called_out(tmp_path: Path) -> None:
    """Nine weeks of data must not be presented as a track record."""
    text = build_markdown(_inputs(length=20), {}, tmp_path)
    assert "daily observations" in text
    assert "edge and noise" in text


def test_a_long_sample_is_not_flagged(tmp_path: Path) -> None:
    text = build_markdown(_inputs(length=400), {}, tmp_path)
    assert "edge and noise" not in text


def test_the_survivorship_warning_is_reproduced_when_given(tmp_path: Path) -> None:
    text = build_markdown(_inputs(survivorship_warning="only survivors here"), {}, tmp_path)
    assert "Survivorship warning" in text
    assert "only survivors here" in text


def test_notes_are_rendered(tmp_path: Path) -> None:
    text = build_markdown(_inputs(notes=["dilution is inert in a backtest"]), {}, tmp_path)
    assert "- dilution is inert in a backtest" in text


def test_charts_are_linked_relative_to_the_report(tmp_path: Path) -> None:
    inputs = _inputs()
    charts = {"Equity": tmp_path / "run-equity.png"}
    text = build_markdown(inputs, charts, tmp_path)
    assert "![Equity](run-equity.png)" in text


def test_an_empty_trade_log_does_not_break_the_report(tmp_path: Path) -> None:
    path = write_report(_inputs(trades=pd.DataFrame()), tmp_path, stem="empty")
    text = path.read_text()
    assert "No closed round trips yet" in text
    assert (tmp_path / "empty-deciles.png").exists()


def test_a_short_curve_renders_the_sharpe_chart_as_a_message(tmp_path: Path) -> None:
    write_report(_inputs(length=10), tmp_path, stem="short")
    assert (tmp_path / "short-sharpe.png").stat().st_size > 0


def test_markdown_is_deterministic(tmp_path: Path) -> None:
    first = build_markdown(_inputs(), {}, tmp_path)
    second = build_markdown(_inputs(), {}, tmp_path)
    assert first == second
