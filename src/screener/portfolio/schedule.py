"""Rebalance cadence.

Two views of the same setting. A backtest needs the list of dates up front; a
papertrade run that fires from a cron needs to answer "is one due today" without
assuming the scheduler fired exactly on time. The elapsed-days form is the
tolerant one: a workflow that runs a day late still rebalances rather than
skipping a week.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from datetime import date, timedelta


class Cadence(enum.StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


MIN_DAYS_BETWEEN: dict[Cadence, int] = {
    Cadence.DAILY: 1,
    Cadence.WEEKLY: 7,
    # 28 rather than 30, so a month-end run is never pushed into the next month.
    Cadence.MONTHLY: 28,
}


def rebalance_dates(start: date, end: date, cadence: Cadence) -> list[date]:
    """Every rebalance date in [start, end] inclusive.

    Weekly anchors on Monday and monthly on the 1st, so the schedule depends on
    the calendar rather than on when the backtest happened to begin - two
    backtests over overlapping windows rebalance on the same days.
    """
    if end < start:
        return []
    anchors: dict[Cadence, Callable[[date], bool]] = {
        Cadence.DAILY: lambda _: True,
        Cadence.WEEKLY: lambda day: day.weekday() == 0,
        Cadence.MONTHLY: lambda day: day.day == 1,
    }
    is_anchor = anchors[cadence]

    days: list[date] = []
    cursor = start
    while cursor <= end:
        if is_anchor(cursor):
            days.append(cursor)
        cursor += timedelta(days=1)
    # A window that contains no anchor still deserves one rebalance, otherwise a
    # short backtest silently does nothing at all.
    if not days:
        days.append(start)
    return days


def is_due(today: date, last_rebalance: date | None, cadence: Cadence) -> bool:
    if last_rebalance is None:
        return True
    return (today - last_rebalance).days >= MIN_DAYS_BETWEEN[cadence]
