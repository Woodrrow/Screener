"""A single self-contained HTML dashboard.

The CSVs are complete but unreadable, and the markdown report is a wall of
tables. This renders one file you can open from disk - no server, no CDN, no
network - so the weekly output is something you actually look at.

Design notes worth keeping:

- Factor percentiles are a continuous magnitude, so they get a sequential
  one-hue ramp rather than eight categorical colours. Magnitude is encoded as
  distance from the surface in both themes: pale-on-light and bright-on-dark,
  each stepped for its own surface rather than flipped automatically.
- Cell text colour is pinned per ramp step so every number clears 4.5:1 against
  the swatch it sits on. The table is also the accessible twin of the heatmap:
  every value is a readable number, not colour alone.
- Charts are inline SVG/CSS generated here. No chart library, because this file
  has to open from a filesystem years from now.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from screener import metrics

# Sequential blue ramp (steps 100-700) with a foreground pinned to each step.
# Ordered low -> high for the light surface; the dark surface uses the reverse,
# so in both themes a bigger number sits further from the background.
_RAMP: list[tuple[str, str]] = [
    ("#cde2fb", "#0b0b0b"),
    ("#9ec5f4", "#0b0b0b"),
    ("#6da7ec", "#0b0b0b"),
    ("#3987e5", "#0b0b0b"),
    ("#256abf", "#ffffff"),
    ("#184f95", "#ffffff"),
    ("#0d366b", "#ffffff"),
]
BINS = len(_RAMP)


@dataclass(frozen=True)
class DashboardData:
    as_of: date
    generated_on: date
    preset: str
    weights: dict[str, float]
    universe_size: int
    ranking: pd.DataFrame
    exclusions: pd.DataFrame
    starting_capital: float
    equity: pd.Series[float] | None = None
    benchmarks: dict[str, pd.Series[float]] = field(default_factory=dict)
    holdings: pd.DataFrame | None = None
    trades: pd.DataFrame | None = None
    last_signal_date: date | None = None
    last_rebalance: date | None = None

    @property
    def has_record(self) -> bool:
        return self.equity is not None and len(self.equity) > 0


def _floats(frame: pd.DataFrame, column: str) -> list[float]:
    """A column as plain Python floats.

    Going through numpy rather than itertuples keeps every value a real float
    instead of the union pandas advertises for a heterogeneous row.
    """
    if column not in frame.columns:
        return [float("nan")] * len(frame)
    return [
        float(value)
        for value in pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
    ]


def _exclusion_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    counts = _floats(frame, "coins")
    return [
        {"reason": str(frame["reason"].iloc[index]), "coins": int(counts[index])}
        for index in range(len(frame))
    ]


def _factor_names(weights: dict[str, float]) -> list[str]:
    return sorted(weights)


def _series_payload(series: pd.Series[float]) -> list[dict[str, Any]]:
    return [
        {"d": day.isoformat() if isinstance(day, date) else str(day), "v": float(value)}
        for day, value in sorted(series.items())
    ]


def build_payload(data: DashboardData) -> dict[str, Any]:
    """Everything the page needs, as plain JSON-able data, deterministically ordered."""
    factors = _factor_names(data.weights)
    ranking = data.ranking

    rows: list[dict[str, Any]] = []
    for index in range(len(ranking)):
        row = ranking.iloc[index]
        entry: dict[str, Any] = {
            "rank": int(row["rank"]),
            "id": str(row["coin_id"]),
            "sym": str(row.get("symbol", "")),
            "name": str(row.get("name", "")),
            "score": round(float(row["composite_score"]), 2),
            "price": float(row["current_price"]) if pd.notna(row.get("current_price")) else None,
            "mcap": float(row["market_cap"]) if pd.notna(row.get("market_cap")) else None,
            "pct": {},
        }
        for name in factors:
            column = f"pct_{name}"
            value = row.get(column)
            entry["pct"][name] = None if value is None or pd.isna(value) else round(float(value), 4)
        rows.append(entry)

    payload: dict[str, Any] = {
        "asOf": data.as_of.isoformat(),
        "generatedOn": data.generated_on.isoformat(),
        "preset": data.preset,
        "weights": {name: data.weights[name] for name in factors},
        "factors": factors,
        "universeSize": data.universe_size,
        "startingCapital": data.starting_capital,
        "rows": rows,
        "exclusions": _exclusion_rows(data.exclusions),
        "hasRecord": data.has_record,
        "lastSignalDate": data.last_signal_date.isoformat() if data.last_signal_date else None,
        "lastRebalance": data.last_rebalance.isoformat() if data.last_rebalance else None,
        "equity": _series_payload(data.equity) if data.equity is not None else [],
        "benchmarks": {
            name: _series_payload(curve) for name, curve in sorted(data.benchmarks.items())
        },
        "holdings": [],
        "trades": [],
        "stats": {},
    }

    if data.holdings is not None and not data.holdings.empty:
        frame = data.holdings
        units = _floats(frame, "units")
        price = _floats(frame, "price")
        value = _floats(frame, "value_usd")
        cost = _floats(frame, "cost_basis_usd")
        payload["holdings"] = [
            {
                "id": str(frame["coin_id"].iloc[index]),
                "units": units[index],
                "price": price[index],
                "value": value[index],
                "cost": cost[index],
                "pnl": value[index] - cost[index],
                "opened": str(frame["opened_on"].iloc[index]),
            }
            for index in range(len(frame))
        ]

    if data.trades is not None and not data.trades.empty:
        recent = data.trades.tail(40).reset_index(drop=True)
        ranks = _floats(recent, "rank")
        fill = _floats(recent, "fill_price")
        gross = _floats(recent, "gross_usd")
        fees = _floats(recent, "fee_usd")
        slip = _floats(recent, "slippage_usd")
        payload["trades"] = [
            {
                "date": str(recent["trade_date"].iloc[index]),
                "action": str(recent["action"].iloc[index]),
                "id": str(recent["coin_id"].iloc[index]),
                "rank": None if pd.isna(ranks[index]) else int(ranks[index]),
                "price": fill[index],
                "gross": gross[index],
                "cost": fees[index] + slip[index],
                "reason": str(recent["reason"].iloc[index]),
            }
            for index in range(len(recent))
        ]

    if data.has_record and data.equity is not None:
        summary = metrics.summarise(data.equity, "Strategy")
        drag = metrics.cost_drag(
            data.trades if data.trades is not None else pd.DataFrame(), data.starting_capital
        )
        payload["stats"] = {
            "equity": float(data.equity.iloc[-1]),
            "totalReturn": summary.total_return,
            "maxDrawdown": summary.max_drawdown,
            "sharpe": summary.sharpe,
            "observations": summary.observations,
            "thin": summary.is_statistically_thin,
            "costUsd": drag["total_usd"],
            "costPct": drag["pct_of_capital"],
        }
    elif data.trades is not None and not data.trades.empty:
        drag = metrics.cost_drag(data.trades, data.starting_capital)
        payload["stats"] = {"costUsd": drag["total_usd"], "costPct": drag["pct_of_capital"]}

    return payload


def _ramp_css() -> tuple[str, str]:
    """Heat-cell classes for each surface.

    The dark surface reverses the ramp so a high value is still the step
    furthest from the background rather than the one closest to it.
    """
    light = "\n".join(
        f"      .h{index} {{ background: {bg}; color: {fg}; }}"
        for index, (bg, fg) in enumerate(_RAMP)
    )
    dark = "\n".join(
        f"        .h{index} {{ background: {bg}; color: {fg}; }}"
        for index, (bg, fg) in enumerate(reversed(_RAMP))
    )
    return light, dark


_TEMPLATE_PATH = Path(__file__).parent / "templates" / "dashboard.html"


def _template() -> str:
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def render_dashboard(data: DashboardData) -> str:
    """The whole page as one string. No external requests, ever."""
    payload = json.dumps(build_payload(data), sort_keys=True, separators=(",", ":"))
    # Escape every "<" as its JSON unicode form. "</script" ends the element
    # outright, and a bare "<script" pushes the tokenizer into an escaped state
    # where the real closing tag stops working - so neither can be allowed
    # through. "<" never appears in JSON syntax itself, only inside strings,
    # so this round-trips exactly.
    payload = payload.replace("<", "\\u003c")
    light, dark = _ramp_css()
    title = f"Screener - {data.as_of.isoformat()}"

    html = _template()
    for token, value in (
        ("__TITLE__", title),
        ("__AS_OF__", data.as_of.isoformat()),
        ("__PRESET__", data.preset),
        ("__GENERATED__", data.generated_on.isoformat()),
        ("__RAMP_LIGHT__", light),
        ("__RAMP_DARK__", dark),
        ("__BINS__", str(BINS)),
        ("__PAYLOAD__", payload),
    ):
        html = html.replace(token, value)
    return html


def write_dashboard(data: DashboardData, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(data), encoding="utf-8")
    return path
