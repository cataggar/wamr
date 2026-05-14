"""wasi-testsuite adapter for the in-tree Zig-based wamr CLI.

Mirrors `wasm-micro-runtime.py` from upstream wasi-testsuite but reads the
runtime path from the `WAMR` environment variable (set by
`zig build wasi-testsuite`) instead of `IWASM`. Lives outside the vendored
submodule (`tests/wasi-testsuite/`) so this repo can edit it without a
fork. Keep the contract aligned with upstream — any drift will silently
break the runner.

The wamr CLI uses a Wasmtime-shaped subcommand layout
(`wamr run <file.wasm>`, `wamr version`), so the adapter inserts `run` /
`version` after the binary path.
"""

import subprocess
import os
import shlex
from pathlib import Path
from typing import Dict, List, Tuple


WAMR = shlex.split(os.getenv("WAMR", "wamr"))


def get_name() -> str:
    return "wamr-zig"


def get_version() -> str:
    result = subprocess.run(
        WAMR + ["version"],
        encoding="UTF-8",
        capture_output=True,
        check=True,
    )
    output = result.stdout.splitlines()[0].split(" ")
    return output[1]


def get_wasi_versions() -> List[str]:
    # Declares both Preview 1 (covered by `zig build wasi-testsuite`) and
    # Preview 3 (covered by `zig build wasi-p3-testsuite`, issue #489) so
    # the upstream `UnsupportedWasiTestExcludeFilter` doesn't auto-skip
    # the wasm32-wasip3 fixtures. Per-test gating for incomplete P3
    # adapter coverage lives in `tests/wasi-p3-testsuite-skip.json`.
    return ["wasm32-wasip1", "wasm32-wasip3"]


def get_wasi_worlds() -> List[str]:
    return ["wasi:cli/command"]


def compute_argv(
    test_path: str,
    args_env_dirs: Tuple[List[str], Dict[str, str], List[Tuple[Path, str]]],
    proposals: List[str],
    wasi_world: str,
    wasi_version: str,
) -> List[str]:
    argv: List[str] = []
    argv += WAMR
    argv += ["run"]
    args, env, dirs = args_env_dirs

    for k, v in env.items():
        argv += ["--env", f"{k}={v}"]

    for host, guest in dirs:
        argv += ["--map-dir", f"{host}::{guest}"]

    argv += [test_path]
    argv += args
    return argv
