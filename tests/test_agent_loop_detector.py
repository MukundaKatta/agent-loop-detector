"""Tests for agent-loop-detector.

These tests use the Python standard-library ``unittest`` framework only, so
they run without any third-party dependencies::

    python3 -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path

# Support running ``python3 -m unittest discover -s tests`` directly from a
# checkout (src/ layout) without installing the package first.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_loop_detector import (  # noqa: E402  (import after sys.path setup)
    CallRecord,
    LoopDetected,
    LoopDetector,
    make_loop_detector,
)


class BasicRecordAndCheckTests(unittest.TestCase):
    def test_no_loop_different_tools(self):
        d = LoopDetector(max_repeats=3, window=10)
        d.record("search", {"q": "a"})
        d.record("search", {"q": "b"})
        d.record("search", {"q": "c"})
        self.assertFalse(d.check("search", {"q": "d"}))

    def test_no_loop_same_tool_different_input(self):
        d = LoopDetector(max_repeats=3, window=10)
        for i in range(5):
            d.record("search", {"q": str(i)})
        self.assertFalse(d.check("search", {"q": "new"}))

    def test_loop_raises_at_threshold(self):
        d = LoopDetector(max_repeats=3, window=10)
        d.record("search", {"q": "same"})
        d.record("search", {"q": "same"})
        d.record("search", {"q": "same"})
        with self.assertRaises(LoopDetected):
            d.check("search", {"q": "same"})

    def test_loop_detected_below_threshold_ok(self):
        d = LoopDetector(max_repeats=3, window=10)
        d.record("search", {"q": "x"})
        d.record("search", {"q": "x"})
        self.assertFalse(d.check("search", {"q": "x"}))  # only 2, need 3

    def test_raise_on_loop_false_returns_true(self):
        d = LoopDetector(max_repeats=2, window=10, raise_on_loop=False)
        d.record("tool", {"a": 1})
        d.record("tool", {"a": 1})
        self.assertIs(d.check("tool", {"a": 1}), True)  # no raise

    def test_raise_on_loop_false_no_exception(self):
        d = LoopDetector(max_repeats=2, window=10, raise_on_loop=False)
        d.record("tool", {})
        d.record("tool", {})
        result = d.check("tool", {})  # should not raise
        self.assertIs(result, True)


class CheckAndRecordTests(unittest.TestCase):
    def test_check_and_record_records_call(self):
        d = LoopDetector(max_repeats=3, window=10)
        d.check_and_record("tool", {"x": 1})
        self.assertEqual(d.call_count(), 1)

    def test_check_and_record_raises_before_recording(self):
        d = LoopDetector(max_repeats=2, window=10)
        d.record("tool", {})
        d.record("tool", {})
        with self.assertRaises(LoopDetected):
            d.check_and_record("tool", {})
        # Should not have recorded the third call
        self.assertEqual(d.repeat_count("tool", {}), 2)

    def test_check_and_record_no_loop_returns_false(self):
        d = LoopDetector(max_repeats=3, window=10)
        result = d.check_and_record("tool", {"q": "x"})
        self.assertIs(result, False)

    def test_check_and_record_no_raise_records_loop_call(self):
        # When raise_on_loop is False, the looping call is still recorded.
        d = LoopDetector(max_repeats=2, window=10, raise_on_loop=False)
        d.record("tool", {"a": 1})
        d.record("tool", {"a": 1})
        result = d.check_and_record("tool", {"a": 1})
        self.assertIs(result, True)
        self.assertEqual(d.repeat_count("tool", {"a": 1}), 3)

    def test_check_and_record_none_input(self):
        d = LoopDetector(max_repeats=3, window=10)
        d.check_and_record("tool")  # input_data defaults to None -> {}
        self.assertEqual(d.repeat_count("tool", {}), 1)


class LoopDetectedExceptionTests(unittest.TestCase):
    def test_loop_detected_attributes(self):
        d = LoopDetector(max_repeats=2, window=5)
        d.record("web_search", {"q": "test"})
        d.record("web_search", {"q": "test"})
        with self.assertRaises(LoopDetected) as ctx:
            d.check("web_search", {"q": "test"})
        exc = ctx.exception
        self.assertEqual(exc.tool_name, "web_search")
        self.assertEqual(exc.count, 2)
        self.assertEqual(exc.window, 5)

    def test_loop_detected_str(self):
        exc = LoopDetected("my_tool", 3, 10)
        self.assertIn("my_tool", str(exc))
        self.assertIn("3", str(exc))

    def test_loop_detected_is_exception(self):
        self.assertTrue(issubclass(LoopDetected, Exception))


class SlidingWindowTests(unittest.TestCase):
    def test_window_evicts_old_calls(self):
        d = LoopDetector(max_repeats=2, window=3)
        d.record("tool", {"x": 1})
        d.record("tool", {"x": 1})
        # window=3, now add 3 more to push the first two out
        d.record("other", {})
        d.record("other", {})
        d.record("other", {})
        # The two {"x": 1} calls should now be evicted
        self.assertFalse(d.check("tool", {"x": 1}))

    def test_window_size_respected(self):
        d = LoopDetector(max_repeats=3, window=5)
        for _ in range(10):
            d.record("fill", {"a": "b"})
        self.assertEqual(len(d), 5)

    def test_window_only_counts_recent(self):
        d = LoopDetector(max_repeats=2, window=4)
        d.record("t", {"x": 1})
        d.record("t", {"x": 1})
        # Need 4 more records to push both t's out of window=4
        d.record("a", {})
        d.record("b", {})
        d.record("c", {})
        d.record("d", {})
        # Now t+{x:1} count in window is 0
        self.assertFalse(d.check("t", {"x": 1}))


class InputHashingTests(unittest.TestCase):
    def test_input_key_order_insensitive(self):
        d = LoopDetector(max_repeats=2, window=10)
        d.record("tool", {"a": 1, "b": 2})
        d.record("tool", {"b": 2, "a": 1})
        # Both should map to the same hash
        self.assertEqual(d.repeat_count("tool", {"a": 1, "b": 2}), 2)

    def test_different_values_different_hash(self):
        d = LoopDetector(max_repeats=2, window=10)
        d.record("tool", {"q": "hello"})
        d.record("tool", {"q": "world"})
        self.assertEqual(d.repeat_count("tool", {"q": "hello"}), 1)

    def test_none_input_treated_as_empty_dict(self):
        d = LoopDetector(max_repeats=2, window=10)
        d.record("tool", None)
        d.record("tool", {})
        # Both should match
        self.assertEqual(d.repeat_count("tool", {}), 2)

    def test_non_serializable_input_falls_back_to_repr(self):
        # A value json cannot serialize must not crash record/check; it falls
        # back to repr() and the same object should still match itself.
        d = LoopDetector(max_repeats=2, window=10)
        sentinel = object()
        d.record("tool", {"obj": sentinel})
        d.record("tool", {"obj": sentinel})
        self.assertEqual(d.repeat_count("tool", {"obj": sentinel}), 2)


class InspectionTests(unittest.TestCase):
    def test_call_count_all(self):
        d = LoopDetector(max_repeats=3, window=10)
        d.record("a", {})
        d.record("b", {})
        d.record("c", {})
        self.assertEqual(d.call_count(), 3)

    def test_call_count_filtered(self):
        d = LoopDetector(max_repeats=3, window=10)
        d.record("search", {"q": "x"})
        d.record("search", {"q": "y"})
        d.record("read", {"path": "/f"})
        self.assertEqual(d.call_count("search"), 2)
        self.assertEqual(d.call_count("read"), 1)

    def test_repeat_count(self):
        d = LoopDetector(max_repeats=5, window=10)
        d.record("tool", {"a": 1})
        d.record("tool", {"a": 1})
        d.record("tool", {"a": 2})
        self.assertEqual(d.repeat_count("tool", {"a": 1}), 2)
        self.assertEqual(d.repeat_count("tool", {"a": 2}), 1)
        self.assertEqual(d.repeat_count("tool", {"a": 9}), 0)

    def test_is_looping_false(self):
        d = LoopDetector(max_repeats=3, window=10)
        d.record("t", {})
        d.record("t", {})
        self.assertFalse(d.is_looping("t", {}))

    def test_is_looping_true(self):
        d = LoopDetector(max_repeats=3, window=10)
        d.record("t", {})
        d.record("t", {})
        d.record("t", {})
        self.assertTrue(d.is_looping("t", {}))

    def test_history_order(self):
        d = LoopDetector(max_repeats=5, window=10)
        d.record("a", {})
        d.record("b", {})
        d.record("c", {})
        h = d.history()
        self.assertEqual([r.tool_name for r in h], ["a", "b", "c"])

    def test_history_returns_copy(self):
        d = LoopDetector(max_repeats=5, window=10)
        d.record("a", {})
        h = d.history()
        h.clear()
        self.assertEqual(len(d), 1)


class ResetTests(unittest.TestCase):
    def test_reset_clears_history(self):
        d = LoopDetector(max_repeats=2, window=10)
        d.record("tool", {})
        d.record("tool", {})
        d.reset()
        self.assertEqual(d.call_count(), 0)
        self.assertFalse(d.check("tool", {}))

    def test_reset_allows_reuse(self):
        d = LoopDetector(max_repeats=2, window=10)
        d.record("t", {})
        d.record("t", {})
        self.assertTrue(d.is_looping("t", {}))
        d.reset()
        d.record("t", {})
        self.assertFalse(d.is_looping("t", {}))


class ValidationTests(unittest.TestCase):
    def test_max_repeats_zero_raises(self):
        with self.assertRaises(ValueError):
            LoopDetector(max_repeats=0, window=10)

    def test_window_zero_raises(self):
        with self.assertRaises(ValueError):
            LoopDetector(max_repeats=3, window=0)

    def test_negative_values_raise(self):
        with self.assertRaises(ValueError):
            LoopDetector(max_repeats=-1, window=10)
        with self.assertRaises(ValueError):
            LoopDetector(max_repeats=3, window=-5)


class FactoryTests(unittest.TestCase):
    def test_make_loop_detector(self):
        d = make_loop_detector(max_repeats=2, window=5)
        self.assertEqual(d.max_repeats, 2)
        self.assertEqual(d.window, 5)

    def test_make_loop_detector_defaults(self):
        d = make_loop_detector()
        self.assertEqual(d.max_repeats, 3)
        self.assertEqual(d.window, 10)

    def test_make_loop_detector_raise_on_loop_flag(self):
        d = make_loop_detector(max_repeats=1, window=3, raise_on_loop=False)
        self.assertFalse(d.raise_on_loop)


class ReprAndLenTests(unittest.TestCase):
    def test_repr(self):
        d = LoopDetector(max_repeats=3, window=10)
        self.assertIn("LoopDetector", repr(d))

    def test_repr_includes_counts(self):
        d = LoopDetector(max_repeats=4, window=7)
        d.record("x", {})
        text = repr(d)
        self.assertIn("max_repeats=4", text)
        self.assertIn("window=7", text)
        self.assertIn("calls=1", text)

    def test_len(self):
        d = LoopDetector(max_repeats=3, window=10)
        d.record("x", {})
        d.record("y", {})
        self.assertEqual(len(d), 2)


class CallRecordTests(unittest.TestCase):
    def test_call_record_matches(self):
        r1 = CallRecord(tool_name="a", input_hash="abc", input_repr="{}")
        r2 = CallRecord(tool_name="a", input_hash="abc", input_repr="{}")
        self.assertTrue(r1.matches(r2))

    def test_call_record_no_match_different_hash(self):
        r1 = CallRecord(tool_name="a", input_hash="abc", input_repr="{}")
        r2 = CallRecord(tool_name="a", input_hash="xyz", input_repr="{}")
        self.assertFalse(r1.matches(r2))

    def test_call_record_no_match_different_name(self):
        r1 = CallRecord(tool_name="a", input_hash="abc", input_repr="{}")
        r2 = CallRecord(tool_name="b", input_hash="abc", input_repr="{}")
        self.assertFalse(r1.matches(r2))


if __name__ == "__main__":
    unittest.main()
