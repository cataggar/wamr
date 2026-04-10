"""wamrc-bin — WebAssembly Micro Runtime AOT compiler."""

import os
import subprocess
import sys
from pathlib import Path

_EXT = ".exe" if sys.platform == "win32" else ""


def wamrc() -> None:
    """Run the wamrc AOT compiler binary."""
    binary = Path(__file__).parent / f"wamrc{_EXT}"
    if not binary.exists():
        print(f"wamrc binary not found at {binary}", file=sys.stderr)
        sys.exit(1)
    args = [str(binary), *sys.argv[1:]]
    if sys.platform != "win32":
        os.execv(args[0], args)
    else:
        raise SystemExit(subprocess.call(args))
