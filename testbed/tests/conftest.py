"""Makes the testbed importable regardless of where pytest is invoked."""

import pathlib
import sys

TESTBED_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(TESTBED_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTBED_ROOT))
