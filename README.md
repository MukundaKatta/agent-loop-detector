# agent-loop-detector

Detect agent tool-call loops — same tool+args repeated N times in a sliding window. Zero dependencies.

## Install

```bash
pip install agent-loop-detector
```

## Usage

```python
from agent_loop_detector import LoopDetector

detector = LoopDetector(max_repeats=3, window=10)

for turn in agent_loop:
    tool_name, input_data = get_tool_call(turn)
    detector.check_and_record(tool_name, input_data)  # raises LoopDetected
    result = call_tool(tool_name, input_data)
```

## API

```python
d = LoopDetector(max_repeats=3, window=10, raise_on_loop=True)

d.record("search", {"q": "same"})        # record without checking
d.check("search", {"q": "same"})         # check without recording
d.check_and_record("search", {"q": "x"}) # check then record

d.is_looping("search", {"q": "same"})    # True/False, no raise
d.repeat_count("search", {"q": "same"})  # int
d.call_count()                           # total in window
d.history()                              # list[CallRecord]
d.reset()                                # clear window
```

## Zero dependencies

Standard library only: `json`, `collections`, `hashlib`. Nothing else.
