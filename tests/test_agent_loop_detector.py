"""Tests for agent-loop-detector."""

import pytest

from agent_loop_detector import CallRecord, LoopDetected, LoopDetector, make_loop_detector


# ---------------------------------------------------------------------------
# Basic record + check
# ---------------------------------------------------------------------------

def test_no_loop_different_tools():
    d = LoopDetector(max_repeats=3, window=10)
    d.record("search", {"q": "a"})
    d.record("search", {"q": "b"})
    d.record("search", {"q": "c"})
    assert not d.check("search", {"q": "d"})

def test_no_loop_same_tool_different_input():
    d = LoopDetector(max_repeats=3, window=10)
    for i in range(5):
        d.record("search", {"q": str(i)})
    assert not d.check("search", {"q": "new"})

def test_loop_raises_at_threshold():
    d = LoopDetector(max_repeats=3, window=10)
    d.record("search", {"q": "same"})
    d.record("search", {"q": "same"})
    d.record("search", {"q": "same"})
    with pytest.raises(LoopDetected):
        d.check("search", {"q": "same"})

def test_loop_detected_below_threshold_ok():
    d = LoopDetector(max_repeats=3, window=10)
    d.record("search", {"q": "x"})
    d.record("search", {"q": "x"})
    assert not d.check("search", {"q": "x"})  # only 2, need 3

def test_raise_on_loop_false_returns_true():
    d = LoopDetector(max_repeats=2, window=10, raise_on_loop=False)
    d.record("tool", {"a": 1})
    d.record("tool", {"a": 1})
    assert d.check("tool", {"a": 1}) is True  # no raise

def test_raise_on_loop_false_no_exception():
    d = LoopDetector(max_repeats=2, window=10, raise_on_loop=False)
    d.record("tool", {})
    d.record("tool", {})
    result = d.check("tool", {})  # should not raise
    assert result is True

# ---------------------------------------------------------------------------
# check_and_record
# ---------------------------------------------------------------------------

def test_check_and_record_records_call():
    d = LoopDetector(max_repeats=3, window=10)
    d.check_and_record("tool", {"x": 1})
    assert d.call_count() == 1

def test_check_and_record_raises_before_recording():
    d = LoopDetector(max_repeats=2, window=10)
    d.record("tool", {})
    d.record("tool", {})
    with pytest.raises(LoopDetected):
        d.check_and_record("tool", {})
    # Should not have recorded the third call
    assert d.repeat_count("tool", {}) == 2

def test_check_and_record_no_loop_returns_false():
    d = LoopDetector(max_repeats=3, window=10)
    result = d.check_and_record("tool", {"q": "x"})
    assert result is False

# ---------------------------------------------------------------------------
# LoopDetected exception
# ---------------------------------------------------------------------------

def test_loop_detected_attributes():
    d = LoopDetector(max_repeats=2, window=5)
    d.record("web_search", {"q": "test"})
    d.record("web_search", {"q": "test"})
    try:
        d.check("web_search", {"q": "test"})
        assert False, "Should have raised"
    except LoopDetected as e:
        assert e.tool_name == "web_search"
        assert e.count == 2
        assert e.window == 5

def test_loop_detected_str():
    exc = LoopDetected("my_tool", 3, 10)
    assert "my_tool" in str(exc)
    assert "3" in str(exc)

# ---------------------------------------------------------------------------
# Sliding window behavior
# ---------------------------------------------------------------------------

def test_window_evicts_old_calls():
    d = LoopDetector(max_repeats=2, window=3)
    d.record("tool", {"x": 1})
    d.record("tool", {"x": 1})
    # window=3, now add 3 more to push the first two out
    d.record("other", {})
    d.record("other", {})
    d.record("other", {})
    # The two {"x": 1} calls should now be evicted
    assert not d.check("tool", {"x": 1})

def test_window_size_respected():
    d = LoopDetector(max_repeats=3, window=5)
    for _ in range(10):
        d.record("fill", {"a": "b"})
    assert len(d) == 5

def test_window_only_counts_recent():
    d = LoopDetector(max_repeats=2, window=4)
    d.record("t", {"x": 1})
    d.record("t", {"x": 1})
    # Need 4 more records to push both t's out of window=4
    d.record("a", {})
    d.record("b", {})
    d.record("c", {})
    d.record("d", {})
    # Now t+{x:1} count in window is 0
    assert not d.check("t", {"x": 1})

# ---------------------------------------------------------------------------
# Input hashing — order shouldn't matter
# ---------------------------------------------------------------------------

def test_input_key_order_insensitive():
    d = LoopDetector(max_repeats=2, window=10)
    d.record("tool", {"a": 1, "b": 2})
    d.record("tool", {"b": 2, "a": 1})
    # Both should map to the same hash
    assert d.repeat_count("tool", {"a": 1, "b": 2}) == 2

def test_different_values_different_hash():
    d = LoopDetector(max_repeats=2, window=10)
    d.record("tool", {"q": "hello"})
    d.record("tool", {"q": "world"})
    assert d.repeat_count("tool", {"q": "hello"}) == 1

def test_none_input_treated_as_empty_dict():
    d = LoopDetector(max_repeats=2, window=10)
    d.record("tool", None)
    d.record("tool", {})
    # Both should match
    assert d.repeat_count("tool", {}) == 2

# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

def test_call_count_all():
    d = LoopDetector(max_repeats=3, window=10)
    d.record("a", {})
    d.record("b", {})
    d.record("c", {})
    assert d.call_count() == 3

def test_call_count_filtered():
    d = LoopDetector(max_repeats=3, window=10)
    d.record("search", {"q": "x"})
    d.record("search", {"q": "y"})
    d.record("read", {"path": "/f"})
    assert d.call_count("search") == 2
    assert d.call_count("read") == 1

def test_repeat_count():
    d = LoopDetector(max_repeats=5, window=10)
    d.record("tool", {"a": 1})
    d.record("tool", {"a": 1})
    d.record("tool", {"a": 2})
    assert d.repeat_count("tool", {"a": 1}) == 2
    assert d.repeat_count("tool", {"a": 2}) == 1
    assert d.repeat_count("tool", {"a": 9}) == 0

def test_is_looping_false():
    d = LoopDetector(max_repeats=3, window=10)
    d.record("t", {})
    d.record("t", {})
    assert not d.is_looping("t", {})

def test_is_looping_true():
    d = LoopDetector(max_repeats=3, window=10)
    d.record("t", {})
    d.record("t", {})
    d.record("t", {})
    assert d.is_looping("t", {})

def test_history_order():
    d = LoopDetector(max_repeats=5, window=10)
    d.record("a", {})
    d.record("b", {})
    d.record("c", {})
    h = d.history()
    assert [r.tool_name for r in h] == ["a", "b", "c"]

def test_history_returns_copy():
    d = LoopDetector(max_repeats=5, window=10)
    d.record("a", {})
    h = d.history()
    h.clear()
    assert len(d) == 1

# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_clears_history():
    d = LoopDetector(max_repeats=2, window=10)
    d.record("tool", {})
    d.record("tool", {})
    d.reset()
    assert d.call_count() == 0
    assert not d.check("tool", {})

def test_reset_allows_reuse():
    d = LoopDetector(max_repeats=2, window=10)
    d.record("t", {})
    d.record("t", {})
    assert d.is_looping("t", {})
    d.reset()
    d.record("t", {})
    assert not d.is_looping("t", {})

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_max_repeats_zero_raises():
    with pytest.raises(ValueError):
        LoopDetector(max_repeats=0, window=10)

def test_window_zero_raises():
    with pytest.raises(ValueError):
        LoopDetector(max_repeats=3, window=0)

# ---------------------------------------------------------------------------
# make_loop_detector
# ---------------------------------------------------------------------------

def test_make_loop_detector():
    d = make_loop_detector(max_repeats=2, window=5)
    assert d.max_repeats == 2
    assert d.window == 5

def test_make_loop_detector_defaults():
    d = make_loop_detector()
    assert d.max_repeats == 3
    assert d.window == 10

# ---------------------------------------------------------------------------
# repr and len
# ---------------------------------------------------------------------------

def test_repr():
    d = LoopDetector(max_repeats=3, window=10)
    assert "LoopDetector" in repr(d)

def test_len():
    d = LoopDetector(max_repeats=3, window=10)
    d.record("x", {})
    d.record("y", {})
    assert len(d) == 2

# ---------------------------------------------------------------------------
# CallRecord
# ---------------------------------------------------------------------------

def test_call_record_matches():
    r1 = CallRecord(tool_name="a", input_hash="abc", input_repr="{}")
    r2 = CallRecord(tool_name="a", input_hash="abc", input_repr="{}")
    assert r1.matches(r2)

def test_call_record_no_match_different_hash():
    r1 = CallRecord(tool_name="a", input_hash="abc", input_repr="{}")
    r2 = CallRecord(tool_name="a", input_hash="xyz", input_repr="{}")
    assert not r1.matches(r2)

def test_call_record_no_match_different_name():
    r1 = CallRecord(tool_name="a", input_hash="abc", input_repr="{}")
    r2 = CallRecord(tool_name="b", input_hash="abc", input_repr="{}")
    assert not r1.matches(r2)
