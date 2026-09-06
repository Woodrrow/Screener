"""CoinGecko adapter.

Free tier is roughly 10-30 calls/minute, so every request goes through a token
bucket and a disk cache before it reaches the network. Responses are cached
under a key that includes the UTC date with no expiry, which is deliberate: a
day's pull is immutable once taken, so re-running `snapshot` on the same day is
free and returns byte-identical data.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
import requests

from screener.data.cache import DiskCache
from screener.data.ratelimit import TokenBucket
from screener.data.schema import HISTORY_COLUMNS, SNAPSHOT_COLUMNS

API_BASE: Final = "https://api.coingecko.com/api/v3"
API_KEY_ENV: Final = "COINGECKO_API_KEY"
API_KEY_HEADER: Final = "x-cg-demo-api-key"
MAX_PER_PAGE: Final = 250
PRICE_CHANGE_PERIODS: Final = "1h,24h,7d,14d,30d,200d,1y"
RETRY_STATUS: Final = frozenset({408, 429, 500, 502, 503, 504})

# CoinGecko field -> canonical column. Returns arrive as percentages.
_RETURN_FIELDS: Final[dict[str, str]] = {
    "price_change_percentage_1h_in_currency": "ret_1h",
    "price_change_percentage_24h_in_currency": "ret_24h",
    "price_change_percentage_7d_in_currency": "ret_7d",
    "price_change_percentage_14d_in_currency": "ret_14d",
    "price_change_percentage_30d_in_currency": "ret_30d",
    "price_change_percentage_200d_in_currency": "ret_200d",
    "price_change_percentage_1y_in_currency": "ret_1y",
}

_PASSTHROUGH_FIELDS: Final[dict[str, str]] = {
    "id": "coin_id",
    "symbol": "symbol",
    "name": "name",
    "current_price": "current_price",
    "market_cap": "market_cap",
    "market_cap_rank": "market_cap_rank",
    "fully_diluted_valuation": "fully_diluted_valuation",
    "total_volume": "total_volume",
    "circulating_supply": "circulating_supply",
    "total_supply": "total_supply",
    "max_supply": "max_supply",
    "ath": "ath",
    "ath_date": "ath_date",
    "atl": "atl",
    "atl_date": "atl_date",
    "last_updated": "last_updated",
}


class CoinGeckoError(RuntimeError):
    """A request failed after exhausting retries, or returned an unusable body."""


class CoinGeckoClient:
    """A `DataSource` backed by the CoinGecko public API."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        rate_per_minute: float = 10.0,
        api_key: str | None = None,
        session: requests.Session | None = None,
        max_retries: int = 5,
        backoff_base_seconds: float = 1.0,
        backoff_cap_seconds: float = 60.0,
        timeout_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        today: Callable[[], date] | None = None,
    ) -> None:
        self._cache = DiskCache(cache_dir)
        self._bucket = TokenBucket(rate_per_minute=rate_per_minute, sleep=sleep)
        self._api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        self._session = session or requests.Session()
        self._max_retries = max_retries
        self._backoff_base = backoff_base_seconds
        self._backoff_cap = backoff_cap_seconds
        self._timeout = timeout_seconds
        self._sleep = sleep
        # Jitter only perturbs retry timing, never output, so a fixed seed keeps
        # test runs reproducible without weakening the backoff.
        self._rng = rng or random.Random(0)
        self._today = today or (lambda: datetime.now(UTC).date())

    @property
    def name(self) -> str:
        return "coingecko"

    @property
    def cache(self) -> DiskCache:
        return self._cache

    # ------------------------------------------------------------------ HTTP

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json", "user-agent": "crypto-factor-screener/0.1"}
        if self._api_key:
            headers[API_KEY_HEADER] = self._api_key
        return headers

    def _retry_delay(self, attempt: int, response: requests.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    capped: float = min(float(retry_after), self._backoff_cap)
                    return capped
                except ValueError:
                    pass
        delay: float = min(self._backoff_base * float(2**attempt), self._backoff_cap)
        jitter: float = 0.5 + self._rng.random() * 0.5
        return delay * jitter

    def _get(self, endpoint: str, params: Mapping[str, Any]) -> Any:
        key = self._cache.make_key(endpoint, params, date_bucket=self._today().isoformat())
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        url = f"{API_BASE}{endpoint}"
        last_error: str = "no attempt was made"
        for attempt in range(self._max_retries):
            self._bucket.acquire()
            try:
                response = self._session.get(
                    url, params=dict(params), headers=self._headers(), timeout=self._timeout
                )
            except requests.RequestException as exc:
                last_error = f"transport error: {exc}"
                self._sleep(self._retry_delay(attempt, None))
                continue

            if response.status_code in RETRY_STATUS:
                last_error = f"HTTP {response.status_code}"
                self._sleep(self._retry_delay(attempt, response))
                continue
            if response.status_code >= 400:
                raise CoinGeckoError(
                    f"{url} returned HTTP {response.status_code}: {response.text[:200]}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise CoinGeckoError(f"{url} returned a non-JSON body: {exc}") from exc

            # No expiry: the key already carries the date, so an entry can only
            # ever be served for the day it was fetched.
            self._cache.set(key, payload, ttl_seconds=None)
            return payload

        raise CoinGeckoError(
            f"{url} failed after {self._max_retries} attempts (last: {last_error})"
        )

    # ------------------------------------------------------------- DataSource

    def fetch_universe(self, *, top_n: int, vs_currency: str = "usd") -> pd.DataFrame:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        rows: list[dict[str, Any]] = []
        pages = (top_n + MAX_PER_PAGE - 1) // MAX_PER_PAGE
        for page in range(1, pages + 1):
            per_page = min(MAX_PER_PAGE, top_n - len(rows))
            payload = self._get(
                "/coins/markets",
                {
                    "vs_currency": vs_currency,
                    "order": "market_cap_desc",
                    "per_page": per_page,
                    "page": page,
                    "sparkline": "false",
                    "price_change_percentage": PRICE_CHANGE_PERIODS,
                    "locale": "en",
                },
            )
            if not isinstance(payload, list):
                raise CoinGeckoError(
                    f"/coins/markets page {page} returned {type(payload).__name__}"
                )
            if not payload:
                break
            rows.extend(payload)
            if len(rows) >= top_n:
                break
        return _markets_to_frame(rows[:top_n], source=self.name)

    def fetch_price_history(
        self, coin_id: str, *, days: int, vs_currency: str = "usd"
    ) -> pd.DataFrame:
        if days <= 0:
            raise ValueError("days must be positive")
        payload = self._get(
            f"/coins/{coin_id}/market_chart",
            {"vs_currency": vs_currency, "days": days},
        )
        if not isinstance(payload, dict):
            raise CoinGeckoError(f"market_chart for {coin_id} returned {type(payload).__name__}")
        return market_chart_to_frame(payload)


def _markets_to_frame(rows: Sequence[Mapping[str, Any]], *, source: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(SNAPSHOT_COLUMNS))

    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {
            canonical: row.get(raw) for raw, canonical in _PASSTHROUGH_FIELDS.items()
        }
        for raw, canonical in _RETURN_FIELDS.items():
            value = row.get(raw)
            record[canonical] = None if value is None else float(value) / 100.0
        record["source"] = source
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    frame["symbol"] = frame["symbol"].astype("string").str.upper()
    frame["name"] = frame["name"].astype("string")
    frame["coin_id"] = frame["coin_id"].astype("string")
    frame["source"] = frame["source"].astype("string")
    for column in ("ath_date", "atl_date", "last_updated"):
        frame[column] = frame[column].astype("string")
    # Deterministic ordering independent of the API's page boundaries.
    return frame.sort_values("coin_id", kind="mergesort").reset_index(drop=True)


def market_chart_to_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Collapse CoinGecko's three parallel [ms, value] series into daily rows.

    Under 90 days the API serves hourly points and the `interval=daily` parameter
    is paid-only, so we resample ourselves: the last observation inside each UTC
    day. Taking the last (not the mean) keeps a daily close consistent with the
    over-90-day series, which is already a daily snapshot.
    """
    series = {
        "price": _ms_series(payload.get("prices", [])),
        "market_cap": _ms_series(payload.get("market_caps", [])),
        "volume": _ms_series(payload.get("total_volumes", [])),
    }
    if all(part.empty for part in series.values()):
        return pd.DataFrame(columns=list(HISTORY_COLUMNS))

    frame = pd.concat(series, axis=1)
    frame.index = pd.to_datetime(frame.index, utc=True)
    daily = frame.groupby(frame.index.date).last()
    daily.index.name = "date"
    result = daily.reset_index()
    result["date"] = pd.to_datetime(result["date"]).dt.date
    for column in ("price", "market_cap", "volume"):
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.loc[:, list(HISTORY_COLUMNS)]
    return result.sort_values("date", kind="mergesort").reset_index(drop=True)


def _ms_series(points: Any) -> pd.Series[Any]:
    if not isinstance(points, list) or not points:
        return pd.Series(dtype="float64")
    stamps = [int(point[0]) for point in points]
    values = [float(point[1]) if point[1] is not None else float("nan") for point in points]
    index = pd.to_datetime(pd.Series(stamps, dtype="int64"), unit="ms", utc=True)
    return pd.Series(values, index=index, dtype="float64")
