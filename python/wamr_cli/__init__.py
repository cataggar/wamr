"""wamr-bin — WebAssembly Micro Runtime CLI tools."""

import os
import subprocess
import sys
from pathlib import Path

_TOOLS = [
    "wamr",
    "wamrc",
]

_EXT = ".exe" if sys.platform == "win32" else ""


def _binary_path(tool_name: str) -> Path:
    """Return the path to a wamr tool binary."""
    return Path(__file__).parent / f"{tool_name}{_EXT}"


def _run(tool_name: str) -> None:
    """Run a wamr tool binary, replacing the current process on Unix."""
    binary = _binary_path(tool_name)
    if not binary.exists():
        print(f"{tool_name} binary not found at {binary}", file=sys.stderr)
        sys.exit(1)
    args = [str(binary), *sys.argv[1:]]
    if sys.platform != "win32":
        os.execv(args[0], args)
    else:
        raise SystemExit(subprocess.call(args))


def wamr() -> None:
    _run("wamr")


def wamrc() -> None:
    _run("wamrc")
