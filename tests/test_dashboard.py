from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from screener.dashboard import BINS, DashboardData, build_payload, render_dashboard, write_dashboard

AS_OF = date(2026, 9, 7)
WEIGHTS = {"momentum_30d": 0.5, "liquidity": 0.3, "volatility": 0.2}


def _ranking() -> pd.DataFrame:
    rows = []
    for index, coin in enumerate(["bitcoin", "ethereum", "solana"]):
        row = {
            "rank": index + 1,
            "coin_id": coin,
            "symbol": coin[:3].upper(),
            "name": coin.title(),
            "market_cap": 1e12 / (index + 1),
            "current_price": 100.0 * (index + 1),
            "composite_score": 80.0 - index * 10,
        }
        for offset, name in enumerate(sorted(WEIGHTS)):
            row[f"pct_{name}"] = round(0.1 * (index + offset), 4)
            row[f"raw_{name}"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def _exclusions() -> pd.DataFrame:
    return pd.DataFrame([{"reason": "stablecoin_listed", "coins": 12}])


def _data(**overrides: object) -> DashboardData:
    base: dict[str, object] = {
        "as_of": AS_OF,
        "generated_on": AS_OF,
        "preset": "balanced",
        "weights": WEIGHTS,
        "universe_size": 250,
        "ranking": _ranking(),
        "exclusions": _exclusions(),
        "starting_capital": 1000.0,
    }
    base.update(overrides)
    return DashboardData(**base)  # type: ignore[arg-type]


def _with_record() -> DashboardData:
    days = [AS_OF - timedelta(days=offset) for offset in range(9, -1, -1)]
    equity = pd.Series([1000.0 + i * 3 for i in range(10)], index=pd.Index(days, name="date"))
    benchmark = pd.Series([1000.0 - i for i in range(10)], index=pd.Index(days, name="date"))
    holdings = pd.DataFrame(
        [
            {
                "coin_id": "bitcoin",
                "units": 0.01,
                "price": 60000.0,
                "value_usd": 600.0,
                "cost_basis_usd": 590.0,
                "opened_on": AS_OF.isoformat(),
            }
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "trade_date": AS_OF.isoformat(),
                "action": "buy",
                "coin_id": "bitcoin",
                "units": 0.01,
                "intended_price": 60000.0,
                "fill_price": 60180.0,
                "gross_usd": 601.8,
                "fee_usd": 0.6,
                "slippage_usd": 1.8,
                "cash_after": 398.0,
                "equity_after": 999.0,
                "rank": 1,
                "reason": "entry_top_rank",
            }
        ]
    )
    return _data(
        equity=equity,
        benchmarks={"Buy and hold BTC": benchmark},
        holdings=holdings,
        trades=trades,
        last_rebalance=AS_OF,
    )


def _payload_from_html(html: str) -> dict[str, Any]:
    match = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    assert match is not None
    parsed: dict[str, Any] = json.loads(match.group(1))
    return parsed


def test_page_is_entirely_self_contained() -> None:
    """It has to open from a filesystem years from now, offline."""
    html = render_dashboard(_with_record())
    assert "http://" not in html
    assert "https://" not in html
    assert not re.search(r"<script[^>]+\bsrc=", html)
    assert not re.search(r"<link[^>]+\bhref=", html)
    assert "@import" not in html


def test_render_is_deterministic() -> None:
    assert render_dashboard(_data()) == render_dashboard(_data())


def test_payload_carries_every_factor_percentile() -> None:
    payload = build_payload(_data())
    assert payload["factors"] == sorted(WEIGHTS)
    first = payload["rows"][0]
    assert set(first["pct"]) == set(WEIGHTS)
    assert first["id"] == "bitcoin"


def test_closing_script_tag_in_data_cannot_break_the_page() -> None:
    ranking = _ranking()
    ranking.loc[0, "name"] = "</script><script>alert(1)</script>"
    html = render_dashboard(_data(ranking=ranking))

    # Exactly two script elements: the JSON island and the page's own logic.
    # Neither "</script" nor a bare "<script" may reach the parser from data.
    assert len(re.findall(r"<script", html)) == 2
    payload = _payload_from_html(html)
    # The value round-trips intact - it is escaped, not stripped.
    assert payload["rows"][0]["name"] == "</script><script>alert(1)</script>"


def test_empty_state_when_the_record_has_not_started() -> None:
    payload = build_payload(_data())
    assert payload["hasRecord"] is False
    assert payload["equity"] == []
    assert payload["holdings"] == []
    assert build_payload(_data())["trades"] == []


def test_record_payload_includes_curves_holdings_and_trades() -> None:
    payload = build_payload(_with_record())
    assert payload["hasRecord"] is True
    assert len(payload["equity"]) == 10
    assert "Buy and hold BTC" in payload["benchmarks"]
    assert payload["holdings"][0]["pnl"] == pytest.approx(10.0)
    assert payload["trades"][0]["cost"] == pytest.approx(2.4)
    stats = payload["stats"]
    assert stats["observations"] == 10
    assert stats["thin"] is True


def test_thin_sample_is_flagged_in_the_payload() -> None:
    """Ten days must not be presented as a track record."""
    assert build_payload(_with_record())["stats"]["thin"] is True


def test_heat_ramp_defines_both_surfaces_and_reverses() -> None:
    html = render_dashboard(_data())
    light = re.search(r"\.h0 \{ background: (#\w+); color: (#\w+); \}", html)
    assert light is not None
    # The dark block must map h0 to the opposite end of the same ramp, so a big
    # number is the step furthest from the surface in either theme.
    dark_block = html.split(':root[data-theme="dark"] {')[-1]
    dark_h0 = re.search(r"\.h0 \{ background: (#\w+);", dark_block)
    assert dark_h0 is not None
    assert dark_h0.group(1) != light.group(1)
    assert html.count(".h0 {") == 3  # base, media query, and explicit toggle


def test_every_ramp_step_is_defined() -> None:
    html = render_dashboard(_data())
    for index in range(BINS):
        assert f".h{index} {{" in html


def test_write_dashboard_creates_the_file(tmp_path: Path) -> None:
    path = write_dashboard(_data(), tmp_path / "nested" / "dashboard.html")
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("<meta charset")


def test_missing_percentiles_survive_as_null() -> None:
    ranking = _ranking()
    ranking.loc[1, "pct_liquidity"] = None
    payload = build_payload(_data(ranking=ranking))
    assert payload["rows"][1]["pct"]["liquidity"] is None
