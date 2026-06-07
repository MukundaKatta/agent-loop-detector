"""Detect agent tool-call loops — same tool+args repeated N times in a sliding window.

Zero dependencies — uses json, collections, hashlib from the standard library.
"""

from __future__ import annotations

import collections
import hashlib
import json
from dataclasses import dataclass
from typing import Any


class LoopDetected(Exception):
    """Raised when a tool-call loop is detected."""

    def __init__(self, tool_name: str, count: int, window: int) -> None:
        self.tool_name = tool_name
        self.count = count
        self.window = window
        super().__init__(
            f"loop detected: tool '{tool_name}' called {count} times in last {window} calls"
        )


@dataclass
class CallRecord:
    """A single recorded tool call."""

    tool_name: str
    input_hash: str
    input_repr: str

    def matches(self, other: CallRecord) -> bool:
        return self.tool_name == other.tool_name and self.input_hash == other.input_hash


def _hash_input(tool_name: str, input_data: dict[str, Any]) -> str:
    """Compute a stable hash of (tool_name, input_data)."""
    try:
        serialized = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        serialized = repr(input_data)
    payload = f"{tool_name}:{serialized}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class LoopDetector:
    """Detect repeated tool calls within a sliding window.

    Example::

        detector = LoopDetector(max_repeats=3, window=10)

        for turn in agent_loop:
            tool_name, input_data = get_tool_call(turn)
            detector.check_and_record(tool_name, input_data)  # raises LoopDetected
            result = call_tool(tool_name, input_data)
    """

    def __init__(
        self,
        *,
        max_repeats: int = 3,
        window: int = 10,
        raise_on_loop: bool = True,
    ) -> None:
        """
        Args:
            max_repeats: how many times the same (tool_name, input) must appear
                in the last ``window`` calls before a loop is declared.
            window: sliding window size (number of most recent calls to inspect).
            raise_on_loop: if True (default), raise LoopDetected; if False, return True.
        """
        if max_repeats < 1:
            raise ValueError("max_repeats must be >= 1")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.max_repeats = max_repeats
        self.window = window
        self.raise_on_loop = raise_on_loop
        self._history: collections.deque[CallRecord] = collections.deque(maxlen=window)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record(self, tool_name: str, input_data: dict[str, Any] | None = None) -> None:
        """Record a tool call without checking for loops."""
        if input_data is None:
            input_data = {}
        h = _hash_input(tool_name, input_data)
        try:
            repr_str = json.dumps(input_data, sort_keys=True)
        except (TypeError, ValueError):
            repr_str = repr(input_data)
        self._history.append(CallRecord(tool_name=tool_name, input_hash=h, input_repr=repr_str))

    def check(self, tool_name: str, input_data: dict[str, Any] | None = None) -> bool:
        """Check if calling this tool now would constitute a loop.

        Does NOT record the call. Returns True if loop detected; raises if
        raise_on_loop is True.
        """
        if input_data is None:
            input_data = {}
        h = _hash_input(tool_name, input_data)
        count = sum(
            1 for rec in self._history
            if rec.tool_name == tool_name and rec.input_hash == h
        )
        if count >= self.max_repeats:
            if self.raise_on_loop:
                raise LoopDetected(tool_name, count, self.window)
            return True
        return False

    def check_and_record(
        self, tool_name: str, input_data: dict[str, Any] | None = None
    ) -> bool:
        """Check for a loop, then record the call.

        Behaviour depends on ``raise_on_loop``:

        * If ``raise_on_loop`` is True (default) and a loop is detected, this
          raises :class:`LoopDetected` *before* the call is recorded, so the
          offending call never enters the window.
        * If ``raise_on_loop`` is False, the call is always recorded and the
          return value indicates whether the call constituted a loop.

        Returns:
            ``False`` when no loop was detected, ``True`` when a loop was
            detected and ``raise_on_loop`` is False.
        """
        result = self.check(tool_name, input_data)
        self.record(tool_name, input_data)
        return result

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def call_count(self, tool_name: str | None = None) -> int:
        """Total calls recorded in the current window (optionally filtered by name)."""
        if tool_name is None:
            return len(self._history)
        return sum(1 for r in self._history if r.tool_name == tool_name)

    def repeat_count(self, tool_name: str, input_data: dict[str, Any] | None = None) -> int:
        """How many times this exact (tool_name, input) appears in the window."""
        if input_data is None:
            input_data = {}
        h = _hash_input(tool_name, input_data)
        return sum(1 for r in self._history if r.tool_name == tool_name and r.input_hash == h)

    def history(self) -> list[CallRecord]:
        """Ordered list of recorded calls (oldest → newest)."""
        return list(self._history)

    def is_looping(self, tool_name: str, input_data: dict[str, Any] | None = None) -> bool:
        """Return True if this call would be considered a loop (no raise)."""
        if input_data is None:
            input_data = {}
        h = _hash_input(tool_name, input_data)
        count = sum(1 for r in self._history if r.tool_name == tool_name and r.input_hash == h)
        return count >= self.max_repeats

    def reset(self) -> None:
        """Clear all recorded history."""
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return (
            f"LoopDetector(max_repeats={self.max_repeats}, "
            f"window={self.window}, calls={len(self._history)})"
        )


def make_loop_detector(
    max_repeats: int = 3,
    window: int = 10,
    *,
    raise_on_loop: bool = True,
) -> LoopDetector:
    """Convenience constructor."""
    return LoopDetector(max_repeats=max_repeats, window=window, raise_on_loop=raise_on_loop)
