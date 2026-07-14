"""Launch the vendored `wasi_test_runner` package without installing it.

The upstream runner now obtains its wait timeout from the runtime adapter.
`wamr-zig.py` implements that hook using `WAMR_TESTSUITE_TIMEOUT`, so this
launcher only needs to make the vendored package importable and delegate to
its entry point.
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_UPSTREAM_RUNNER_DIR = _REPO_ROOT / "tests" / "wasi-testsuite" / "test-runner"

# Make the vendored `wasi_test_runner` package importable without
# requiring the user to set PYTHONPATH manually.
sys.path.insert(0, str(_UPSTREAM_RUNNER_DIR))

from wasi_test_runner import __main__ as _upstream_main  # noqa: E402


if __name__ == "__main__":
    sys.exit(_upstream_main.main())
