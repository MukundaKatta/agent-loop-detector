# agent-loop-detector

Detect agent tool-call loops — the same tool called with the same arguments
repeated `N` times inside a sliding window of recent calls. Zero dependencies,
standard library only.

LLM agents that drive tools in a loop sometimes get stuck repeating the exact
same action (re-running the same search, re-reading the same file, retrying the
same failing command). `agent-loop-detector` gives you a tiny, dependency-free
guard you can drop into the agent loop to catch that and break out before you
burn tokens, money, or time.

## Install

```bash
pip install agent-loop-detector
```

Requires Python 3.10+.

## Quick start

```python
from agent_loop_detector import LoopDetector, LoopDetected

# Declare a loop when the same (tool, args) appears 3 times in the last 10 calls.
detector = LoopDetector(max_repeats=3, window=10)

for _ in range(5):
    try:
        # check_and_record raises LoopDetected when the threshold is reached.
        detector.check_and_record("search", {"q": "same query"})
        print("calling tool: search")
    except LoopDetected as exc:
        print(f"stopping: {exc}")
        break
```

Output:

```text
calling tool: search
calling tool: search
calling tool: search
stopping: loop detected: tool 'search' called 3 times in last 10 calls
```

### In a real agent loop

```python
from agent_loop_detector import LoopDetector, LoopDetected

detector = LoopDetector(max_repeats=3, window=10)

while not done:
    tool_name, tool_args = agent.next_tool_call()
    try:
        detector.check_and_record(tool_name, tool_args)
    except LoopDetected as exc:
        agent.add_observation(f"You appear to be looping: {exc}. Try a different approach.")
        continue
    result = run_tool(tool_name, tool_args)
    agent.add_observation(result)
```

### Non-raising mode

If you would rather branch on a boolean than catch an exception, construct the
detector with `raise_on_loop=False`:

```python
detector = LoopDetector(max_repeats=3, window=10, raise_on_loop=False)

if detector.check_and_record(tool_name, tool_args):
    # loop detected — handle it (the call is still recorded in this mode)
    handle_loop()
else:
    run_tool(tool_name, tool_args)
```

## How it works

- Each call is identified by `(tool_name, input_data)`. The input dict is
  serialized with `json.dumps(..., sort_keys=True)` so key order does not
  matter; inputs that cannot be JSON-serialized fall back to `repr()`.
- A SHA-256 hash of that identifier is stored in a fixed-size
  `collections.deque` (the sliding `window`). Old calls are evicted
  automatically once `window` is exceeded.
- A loop is declared when the same identifier appears at least `max_repeats`
  times within the window.

## API

```python
from agent_loop_detector import LoopDetector, LoopDetected, CallRecord, make_loop_detector

d = LoopDetector(max_repeats=3, window=10, raise_on_loop=True)
# or: d = make_loop_detector(max_repeats=3, window=10, raise_on_loop=True)
```

| Method | Description |
| --- | --- |
| `d.record(tool, args=None)` | Record a call without checking for a loop. |
| `d.check(tool, args=None)` | Check whether this call is a loop. Raises `LoopDetected` (or returns `True` when `raise_on_loop=False`); does **not** record. |
| `d.check_and_record(tool, args=None)` | `check` then `record`. In raising mode it raises *before* recording the offending call. |
| `d.is_looping(tool, args=None)` | `True`/`False`, never raises and never records. |
| `d.repeat_count(tool, args=None)` | How many times this exact `(tool, args)` is in the window. |
| `d.call_count(tool=None)` | Total calls in the window, optionally filtered by tool name. |
| `d.history()` | Ordered `list[CallRecord]` (oldest → newest); a copy. |
| `d.reset()` | Clear the window. |
| `len(d)` | Number of calls currently in the window. |

`args` defaults to `None`, which is treated as an empty dict `{}`.

`LoopDetected` carries `.tool_name`, `.count`, and `.window` attributes so you
can inspect or log the details.

## Zero dependencies

Standard library only: `json`, `collections`, `hashlib`, `dataclasses`,
`typing`. Nothing else.

## Development

Run the test suite (standard-library `unittest`, no third-party deps):

```bash
python3 -m unittest discover -s tests
```

## License

MIT — see [LICENSE](LICENSE).
