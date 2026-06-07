"""Test package for agent-loop-detector.

Adds the ``src/`` layout directory to ``sys.path`` so the test suite can be run
straight from a checkout without installing the package first::

    python3 -m unittest discover -s tests
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
