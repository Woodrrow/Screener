from __future__ import annotations

from datetime import date

from screener.portfolio.schedule import Cadence, is_due, rebalance_dates


def test_weekly_anchors_on_monday() -> None:
    days = rebalance_dates(date(2025, 1, 1), date(2025, 1, 31), Cadence.WEEKLY)
    assert days == [
        date(2025, 1, 6),
        date(2025, 1, 13),
        date(2025, 1, 20),
        date(2025, 1, 27),
    ]
    assert all(day.weekday() == 0 for day in days)


def test_monthly_anchors_on_the_first() -> None:
    days = rebalance_dates(date(2025, 1, 15), date(2025, 4, 30), Cadence.MONTHLY)
    assert days == [date(2025, 2, 1), date(2025, 3, 1), date(2025, 4, 1)]


def test_daily_is_every_day() -> None:
    days = rebalance_dates(date(2025, 1, 1), date(2025, 1, 5), Cadence.DAILY)
    assert len(days) == 5


def test_anchoring_is_calendar_based_not_window_based() -> None:
    """Two overlapping backtests must rebalance on the same days."""
    wide = rebalance_dates(date(2025, 1, 1), date(2025, 3, 1), Cadence.WEEKLY)
    narrow = rebalance_dates(date(2025, 2, 3), date(2025, 3, 1), Cadence.WEEKLY)
    assert set(narrow) <= set(wide)


def test_a_window_with_no_anchor_still_gets_one_rebalance() -> None:
    days = rebalance_dates(date(2025, 1, 7), date(2025, 1, 9), Cadence.WEEKLY)
    assert days == [date(2025, 1, 7)]


def test_reversed_window_is_empty() -> None:
    assert rebalance_dates(date(2025, 2, 1), date(2025, 1, 1), Cadence.WEEKLY) == []


def test_is_due_is_tolerant_of_a_late_scheduler() -> None:
    last = date(2025, 1, 6)
    assert is_due(date(2025, 1, 13), last, Cadence.WEEKLY)
    assert is_due(date(2025, 1, 15), last, Cadence.WEEKLY)  # two days late, still due
    assert not is_due(date(2025, 1, 12), last, Cadence.WEEKLY)


def test_first_ever_run_is_always_due() -> None:
    assert is_due(date(2025, 1, 1), None, Cadence.WEEKLY)


def test_monthly_uses_twenty_eight_days() -> None:
    last = date(2025, 1, 31)
    assert not is_due(date(2025, 2, 20), last, Cadence.MONTHLY)
    assert is_due(date(2025, 2, 28), last, Cadence.MONTHLY)
