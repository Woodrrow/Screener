"""On-disk response cache keyed by endpoint + params + date.

The date is part of the key rather than only the TTL because the screener's unit
of work is a day: a `/coins/markets` pull on 2025-03-04 is a different object
from the same pull on 2025-03-05 even though the URL is identical. Keying on the
date makes yesterday's response un-reachable by accident, which is one fewer way
to leak future data into a point-in-time pull.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    expired: int = 0
    writes: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses


class DiskCache:
    def __init__(self, root: Path, *, now: Callable[[], datetime] = _utcnow) -> None:
        self.root = root
        self._now = now
        self.stats = CacheStats()

    @staticmethod
    def make_key(
        endpoint: str,
        params: Mapping[str, Any] | None = None,
        *,
        date_bucket: str,
    ) -> str:
        # sort_keys makes the digest independent of dict insertion order, which
        # is part of the determinism guarantee.
        payload = json.dumps(
            {"endpoint": endpoint, "params": dict(params or {}), "date": date_bucket},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            self.stats.misses += 1
            return None
        try:
            entry = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            # A torn or corrupt entry is a miss, never an error: the caller can
            # always refetch, and failing the run over a bad cache file would be
            # a worse outcome than one extra API call.
            self.stats.misses += 1
            return None

        expires_at = entry.get("expires_at")
        if expires_at is not None and self._now() >= datetime.fromisoformat(expires_at):
            self.stats.expired += 1
            self.stats.misses += 1
            return None

        self.stats.hits += 1
        return entry["payload"]

    def set(self, key: str, payload: Any, *, ttl_seconds: float | None = None) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        now = self._now()
        entry = {
            "key": key,
            "stored_at": now.isoformat(),
            "expires_at": (
                None if ttl_seconds is None else (now + timedelta(seconds=ttl_seconds)).isoformat()
            ),
            "payload": payload,
        }
        # Atomic replace: a crash mid-write leaves the old entry, not a half file.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(entry, handle, separators=(",", ":"), default=str)
            Path(tmp_name).replace(path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        self.stats.writes += 1

    def clear(self) -> int:
        removed = 0
        for path in sorted(self.root.rglob("*.json")):
            path.unlink()
            removed += 1
        return removed
