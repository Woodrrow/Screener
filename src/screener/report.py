"""Markdown report and charts.

The report leads with the caveats rather than burying them, and every table that
could be read as a track record carries its observation count next to it. A
Sharpe computed over eleven weeks is printed with the number eleven beside it so
nobody has to go looking for why it seems too good.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in CI, and no interactive backend to hang on

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from screener import metrics

FIGURE_SIZE = (10, 5)
FIGURE_DPI = 120


@dataclass(frozen=True)
class ReportInputs:
    label: str
    equity: pd.Series[float]
    trades: pd.DataFrame
    benchmarks: dict[str, pd.Series[float]]
    starting_capital: float
    generated_on: date
    survivorship_warning: str | None = None
    notes: list[str] | None = None


def _format_pct(value: float) -> str:
    """Signed, for quantities that can go either way (returns, drawdown)."""
    return "n/a" if pd.isna(value) else f"{value:+.2%}"


def _format_magnitude(value: float) -> str:
    """Unsigned, for quantities that cannot be negative (vol, turnover, win rate)."""
    return "n/a" if pd.isna(value) else f"{value:.2%}"


def _format_ratio(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2f}"


def _summary_table(inputs: ReportInputs) -> str:
    rows = [metrics.summarise(inputs.equity, inputs.label)]
    rows.extend(metrics.summarise(curve, name) for name, curve in sorted(inputs.benchmarks.items()))

    header = (
        "| Series | Days | Total | Annualised | Vol | Sharpe | Sortino | Max DD | Calmar |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    lines = [
        f"| {row.label} | {row.observations} | {_format_pct(row.total_return)} | "
        f"{_format_pct(row.annualised_return)} | {_format_magnitude(row.volatility)} | "
        f"{_format_ratio(row.sharpe)} | {_format_ratio(row.sortino)} | "
        f"{_format_pct(row.max_drawdown)} | {_format_ratio(row.calmar)} |"
        for row in rows
    ]
    return header + "\n".join(lines) + "\n"


def _trading_table(inputs: ReportInputs) -> str:
    costs = metrics.cost_drag(inputs.trades, inputs.starting_capital)
    rows = [
        ("Trades", f"{len(inputs.trades)}"),
        (
            "Turnover per rebalance",
            _format_magnitude(metrics.turnover(inputs.trades, inputs.equity)),
        ),
        ("Win rate (closed round trips)", _format_magnitude(metrics.win_rate(inputs.trades))),
        ("Average holding period", f"{metrics.average_holding_days(inputs.trades):.1f} days"),
        ("Fees", f"${costs['fees_usd']:,.2f}"),
        ("Slippage", f"${costs['slippage_usd']:,.2f}"),
        (
            "Total cost drag",
            f"${costs['total_usd']:,.2f} ({costs['pct_of_capital']:.2%} of starting capital)",
        ),
    ]
    btc = inputs.benchmarks.get("Buy and hold BTC")
    if btc is not None:
        rows.append(
            ("BTC correlation (daily)", _format_ratio(metrics.correlation(inputs.equity, btc)))
        )

    body = "\n".join(f"| {name} | {value} |" for name, value in rows)
    return "| Measure | Value |\n| --- | ---: |\n" + body + "\n"


def _decile_table(trades: pd.DataFrame) -> str:
    frame = metrics.return_by_entry_rank_decile(trades)
    if frame.empty:
        return "_No closed round trips yet._\n"
    lines = [
        f"| {decile} | {count} | ${mean:,.2f} | ${total:,.2f} |"
        for decile, count, mean, total in zip(
            [int(value) for value in frame["decile"].astype("int64")],
            [int(value) for value in frame["trades"].astype("int64")],
            [float(value) for value in frame["mean_pnl_usd"].astype("float64")],
            [float(value) for value in frame["total_pnl_usd"].astype("float64")],
            strict=True,
        )
    ]
    return (
        "| Entry rank decile | Round trips | Mean P&L | Total P&L |\n"
        "| ---: | ---: | ---: | ---: |\n" + "\n".join(lines) + "\n"
    )


# ---------------------------------------------------------------------- charts


def _finish(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


def plot_equity(inputs: ReportInputs, path: Path) -> Path:
    fig, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.plot(inputs.equity.index, inputs.equity.to_numpy(), label=inputs.label, linewidth=2)
    for name, curve in sorted(inputs.benchmarks.items()):
        axis.plot(curve.index, curve.to_numpy(), label=name, linewidth=1, alpha=0.8)
    axis.axhline(inputs.starting_capital, color="grey", linewidth=0.8, linestyle=":")
    axis.set_title("Equity curve vs benchmarks")
    axis.set_ylabel("Portfolio value (USD)")
    axis.legend(loc="upper left", fontsize="small")
    axis.grid(alpha=0.3)
    return _finish(fig, path)


def plot_drawdown(inputs: ReportInputs, path: Path) -> Path:
    underwater = metrics.drawdown_series(inputs.equity)
    fig, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.fill_between(underwater.index, underwater.to_numpy(), 0.0, alpha=0.4)
    axis.plot(underwater.index, underwater.to_numpy(), linewidth=1)
    axis.set_title("Drawdown (underwater)")
    axis.set_ylabel("Drawdown")
    axis.grid(alpha=0.3)
    return _finish(fig, path)


def plot_rolling_sharpe(inputs: ReportInputs, path: Path, window: int = 30) -> Path:
    rolled = metrics.rolling_sharpe(inputs.equity, window=window)
    fig, axis = plt.subplots(figsize=FIGURE_SIZE)
    if rolled.empty:
        axis.text(
            0.5,
            0.5,
            f"Not enough history for a {window}-day rolling Sharpe",
            ha="center",
            va="center",
        )
        axis.set_axis_off()
    else:
        axis.plot(rolled.index, rolled.to_numpy(), linewidth=1.2)
        axis.axhline(0.0, color="grey", linewidth=0.8)
        axis.set_ylabel("Sharpe")
        axis.grid(alpha=0.3)
    axis.set_title(f"Rolling {window}-day Sharpe")
    return _finish(fig, path)


def plot_rank_deciles(inputs: ReportInputs, path: Path) -> Path:
    frame = metrics.return_by_entry_rank_decile(inputs.trades)
    fig, axis = plt.subplots(figsize=FIGURE_SIZE)
    if frame.empty:
        axis.text(0.5, 0.5, "No closed round trips yet", ha="center", va="center")
        axis.set_axis_off()
    else:
        axis.bar(frame["decile"].to_numpy(), frame["total_pnl_usd"].to_numpy())
        axis.axhline(0.0, color="grey", linewidth=0.8)
        axis.set_xlabel("Entry rank decile (1 = best ranked)")
        axis.set_ylabel("Realised P&L (USD)")
        axis.grid(alpha=0.3, axis="y")
    axis.set_title("Realised return by entry-rank decile")
    return _finish(fig, path)


# --------------------------------------------------------------------- report


def build_markdown(inputs: ReportInputs, chart_paths: dict[str, Path], out_dir: Path) -> str:
    summary = metrics.summarise(inputs.equity, inputs.label)
    parts: list[str] = [
        f"# {inputs.label}",
        "",
        f"Generated {inputs.generated_on.isoformat()}. "
        f"Starting capital ${inputs.starting_capital:,.2f}.",
        "",
    ]

    if inputs.survivorship_warning:
        parts += ["> **Survivorship warning.** " + inputs.survivorship_warning, ""]

    if summary.is_statistically_thin:
        parts += [
            f"> **{summary.observations} daily observations.** Under "
            f"{metrics.MIN_OBSERVATIONS} days there is no meaningful distinction between "
            "edge and noise here; read the numbers below as a description of what happened, "
            "not as evidence of anything.",
            "",
        ]

    parts += [
        "## Performance",
        "",
        _summary_table(inputs),
        "",
        "## Trading",
        "",
        _trading_table(inputs),
        "",
    ]
    parts += ["## Return by entry rank", "", _decile_table(inputs.trades), ""]

    if chart_paths:
        parts += ["## Charts", ""]
        for title, path in sorted(chart_paths.items()):
            relative = path.relative_to(out_dir) if path.is_relative_to(out_dir) else path
            parts += [f"### {title}", "", f"![{title}]({relative.as_posix()})", ""]

    if inputs.notes:
        parts += ["## Notes", "", *[f"- {note}" for note in inputs.notes], ""]

    return "\n".join(parts)


def write_report(inputs: ReportInputs, out_dir: Path, stem: str = "report") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "Equity curve vs benchmarks": plot_equity(inputs, out_dir / f"{stem}-equity.png"),
        "Drawdown": plot_drawdown(inputs, out_dir / f"{stem}-drawdown.png"),
        "Rolling 30-day Sharpe": plot_rolling_sharpe(inputs, out_dir / f"{stem}-sharpe.png"),
        "Return by entry-rank decile": plot_rank_deciles(inputs, out_dir / f"{stem}-deciles.png"),
    }
    path = out_dir / f"{stem}.md"
    path.write_text(build_markdown(inputs, charts, out_dir))
    return path
