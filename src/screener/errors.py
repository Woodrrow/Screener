"""Errors shared across the pipeline."""

from __future__ import annotations


class LookaheadError(AssertionError):
    """Raised when data timestamped at or after execution reaches a signal.

    An AssertionError rather than a ValueError because this is an invariant of
    the whole system, not a bad argument: if it fires, every number downstream
    of it is worthless.
    """
