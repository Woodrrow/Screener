from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from screener.data.coingecko import (
    API_KEY_HEADER,
    CoinGeckoClient,
    CoinGeckoError,
    market_chart_to_frame,
)
from screener.data.schema import HISTORY_COLUMNS


class FakeResponse:
    def __init__(
        self, status_code: int, payload: Any = None, headers: dict[str, str] | None = None
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = "" if payload is None else str(payload)

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


class FakeSession:
    """Stands in for requests.Session. Queue responses; every call is recorded."""

    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"unexpected extra request to {url}")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _client(tmp_path: Path, session: FakeSession, **kwargs: Any) -> CoinGeckoClient:
    slept: list[float] = []
    client = CoinGeckoClient(
        cache_dir=tmp_path / "cache",
        rate_per_minute=100_000,  # the limiter is tested separately
        session=session,  # type: ignore[arg-type]
        sleep=slept.append,
        today=lambda: date(2025, 3, 4),
        **kwargs,
    )
    client.slept = slept  # type: ignore[attr-defined]
    return client


def test_markets_are_normalised_to_the_canonical_schema(
    tmp_path: Path, markets_rows: list[dict[str, Any]]
) -> None:
    session = FakeSession([FakeResponse(200, markets_rows)])
    client = _client(tmp_path, session)

    frame = client.fetch_universe(top_n=250)

    assert frame["coin_id"].is_monotonic_increasing
    btc = frame.loc[frame["coin_id"] == "bitcoin"].iloc[0]
    assert btc["symbol"] == "BTC"
    # Percentages in, fractions out - exactly once, at the boundary.
    assert float(btc["ret_30d"]) == pytest.approx(0.09)
    assert float(btc["ret_1y"]) == pytest.approx(0.80)
    assert pd.isna(frame.loc[frame["coin_id"] == "inflato", "max_supply"].iloc[0])


def test_pagination_requests_only_what_is_needed(
    tmp_path: Path, markets_rows: list[dict[str, Any]]
) -> None:
    page_one = markets_rows * 14  # 252 rows, more than one page's worth
    session = FakeSession([FakeResponse(200, page_one[:250]), FakeResponse(200, page_one[:10])])
    client = _client(tmp_path, session)

    client.fetch_universe(top_n=260)

    assert len(session.calls) == 2
    assert session.calls[0]["params"]["per_page"] == 250
    assert session.calls[0]["params"]["page"] == 1
    assert session.calls[1]["params"]["per_page"] == 10
    assert session.calls[1]["params"]["page"] == 2


def test_second_identical_call_is_served_from_cache(
    tmp_path: Path, markets_rows: list[dict[str, Any]]
) -> None:
    session = FakeSession([FakeResponse(200, markets_rows)])
    client = _client(tmp_path, session)

    first = client.fetch_universe(top_n=250)
    second = client.fetch_universe(top_n=250)

    assert len(session.calls) == 1
    pd.testing.assert_frame_equal(first, second)
    assert client.cache.stats.hits == 1


def test_retries_on_429_and_honours_retry_after(
    tmp_path: Path, markets_rows: list[dict[str, Any]]
) -> None:
    session = FakeSession(
        [
            FakeResponse(429, headers={"Retry-After": "7"}),
            FakeResponse(503),
            FakeResponse(200, markets_rows),
        ]
    )
    client = _client(tmp_path, session)

    frame = client.fetch_universe(top_n=250)

    assert len(frame) == len(markets_rows)
    slept: list[float] = client.slept  # type: ignore[attr-defined]
    assert slept[0] == pytest.approx(7.0)
    assert slept[1] > 0.0  # exponential backoff, jittered


def test_transport_errors_are_retried(tmp_path: Path, markets_rows: list[dict[str, Any]]) -> None:
    session = FakeSession([requests.ConnectionError("reset"), FakeResponse(200, markets_rows)])
    client = _client(tmp_path, session)
    assert len(client.fetch_universe(top_n=250)) == len(markets_rows)


def test_gives_up_after_max_retries(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(429) for _ in range(3)])
    client = _client(tmp_path, session, max_retries=3)
    with pytest.raises(CoinGeckoError, match="failed after 3 attempts"):
        client.fetch_universe(top_n=10)


def test_client_errors_are_not_retried(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(401, payload="unauthorised")])
    client = _client(tmp_path, session)
    with pytest.raises(CoinGeckoError, match="HTTP 401"):
        client.fetch_universe(top_n=10)
    assert len(session.calls) == 1


def test_api_key_is_sent_as_the_demo_header(
    tmp_path: Path, markets_rows: list[dict[str, Any]]
) -> None:
    session = FakeSession([FakeResponse(200, markets_rows)])
    client = _client(tmp_path, session, api_key="secret-key")
    client.fetch_universe(top_n=10)
    assert session.calls[0]["headers"][API_KEY_HEADER] == "secret-key"


def test_market_chart_resamples_hourly_points_to_daily_closes() -> None:
    """Under 90 days CoinGecko serves hourly points; we take the last per UTC day."""
    day_one = 1_740_960_000_000  # 2025-03-03T00:00:00Z
    hour = 3_600_000
    payload = {
        "prices": [
            [day_one, 100.0],
            [day_one + 12 * hour, 110.0],
            [day_one + 23 * hour, 105.0],
            [day_one + 24 * hour, 106.0],
            [day_one + 30 * hour, 120.0],
        ],
        "market_caps": [
            [day_one, 1_000.0],
            [day_one + 12 * hour, 1_100.0],
            [day_one + 23 * hour, 1_050.0],
            [day_one + 24 * hour, 1_060.0],
            [day_one + 30 * hour, 1_200.0],
        ],
        "total_volumes": [
            [day_one, 10.0],
            [day_one + 12 * hour, 11.0],
            [day_one + 23 * hour, 12.0],
            [day_one + 24 * hour, 13.0],
            [day_one + 30 * hour, 14.0],
        ],
    }

    frame = market_chart_to_frame(payload)

    assert list(frame.columns) == list(HISTORY_COLUMNS)
    assert frame["date"].tolist() == [date(2025, 3, 3), date(2025, 3, 4)]
    assert frame["price"].tolist() == [105.0, 120.0]
    assert frame["volume"].tolist() == [12.0, 14.0]


def test_market_chart_handles_an_empty_body() -> None:
    frame = market_chart_to_frame({"prices": [], "market_caps": [], "total_volumes": []})
    assert frame.empty
    assert list(frame.columns) == list(HISTORY_COLUMNS)
