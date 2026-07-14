"""wasi-testsuite adapter for upstream Wasmtime — used by the parity
gate (`zig build wasi-p3-testsuite-wasmtime`, CI job
`wasi-p3-testsuite-wasmtime`).

Drives the same `wasm32-wasip3` fixtures as `wamr-zig.py` through a
`wasmtime` CLI binary so a wamr regression that Wasmtime *also*
exhibits gets classified as a fixture bug rather than a wamr bug.
See `scripts/diff-testsuite-reports.py` for the classifier.

This file lives outside the vendored wasi-testsuite submodule
(`tests/wasi-testsuite/adapters/wasmtime.py`) so the in-tree repo can
edit it without forking the upstream — the upstream adapter is
treated as a reference and tracked there. Keep this in sync with the
upstream contract (`tests/wasi-testsuite/test-runner/wasi_test_runner/
runtime_adapter.py`); any drift will silently break the runner.

The runtime binary is resolved from the `WASMTIME` env var (set by
`build.zig`) and defaults to `wasmtime` on `PATH`. Preopen
directories are snapshotted into per-test tempdirs to match the
isolation behaviour of `wamr-zig.py` (#564) so a filesystem fixture
that mutates a preopen doesn't pollute state for subsequent tests in
the same run.

Tracks wamr issue #583 C1 (Wasmtime parity matrix).
"""

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# `shlex.split` so `WASMTIME="wasmtime --some-flag"` works — the user
# (or `build.zig`) can supply additional flags ahead of our injected
# ones. The upstream wasi-testsuite adapter uses the same pattern.
WASMTIME = shlex.split(os.getenv("WASMTIME", "wasmtime"))


def get_name() -> str:
    return "wasmtime"


def get_version() -> str:
    # `wasmtime --version` prints `wasmtime <version> (<commit> <date>)`.
    # The version-query path must not pick up any optional flags the
    # caller bundled into `WASMTIME` (e.g. `-W…`) — drop them by
    # slicing to the binary alone.
    result = subprocess.run(
        WASMTIME[0:1] + ["--version"],
        encoding="UTF-8",
        capture_output=True,
        check=True,
    )
    output = result.stdout.splitlines()[0].split(" ")
    return output[1]


def get_wasi_versions() -> List[str]:
    # Mirror `wamr-zig.py`: declare both Preview 1 (the legacy
    # `zig build wasi-testsuite` corpus, exercised through Wasmtime
    # only on demand) and Preview 3 (the parity gate's main target).
    return ["wasm32-wasip1", "wasm32-wasip3"]


def get_wasi_worlds() -> List[str]:
    # Same world declarations as `wamr-zig.py`: cli/command for the
    # default fixtures and http/service for `http-service`, which
    # exports `wasi:http/incoming-handler@0.3.0.handle`.
    return ["wasi:cli/command", "wasi:http/service"]


def _isolate_root(root: Optional[Path]) -> Optional[Path]:
    """Snapshot the preopened root into a fresh tempdir so a
    filesystem test that mutates the mapped directory doesn't pollute
    state for subsequent tests in the run. Matches the behaviour of
    `wamr-zig.py._isolate_root` so a side-by-side parity diff
    isn't perturbed by host-FS state drift. The tempdirs are leaked
    intentionally — they're tiny and the per-suite TMPDIR is cleared
    between CI invocations.
    """
    if root is None:
        return None
    root_path = Path(root)
    if not root_path.is_dir():
        return root_path
    snapshot = Path(tempfile.mkdtemp(prefix="wasmtime-fs-"))
    shutil.copytree(root_path, snapshot, dirs_exist_ok=True, symlinks=True)
    return snapshot


def compute_argv(
    test_path: str,
    args_env_root: Tuple[List[str], Dict[str, str], Optional[Path]],
    proposals: List[str],
    wasi_world: str,
    wasi_version: str,
) -> List[str]:
    argv: List[str] = []
    argv += WASMTIME
    args, env, root = args_env_root

    for k, v in env.items():
        argv += ["--env", f"{k}={v}"]

    isolated_root = _isolate_root(root)
    if isolated_root:
        argv += ["--dir", f"{isolated_root}::/"]

    argv += [test_path]
    argv += args
    _add_wasi_version_options(argv, wasi_version, proposals, wasi_world)
    return argv


def _add_wasi_version_options(
    argv: List[str],
    wasi_version: str,
    proposals: List[str],
    wasi_world: str,
) -> None:
    """Insert the wasmtime-side WASI feature flags before the wasm
    module path so the user's `WASMTIME=` overrides take precedence.

    Mirrors `tests/wasi-testsuite/adapters/wasmtime.py` — Preview 3
    requires `-Wcomponent-model-async -Sp3`, plus `,http` and
    `,inherit-network` when the fixture declares the corresponding
    proposals. `wasi:http/service` fixtures additionally need the
    `serve` subcommand and an ephemeral bind so the testsuite
    runner's URL-scrape can find the listener.
    """
    splice_pos = len(WASMTIME)
    while splice_pos > 1 and argv[splice_pos - 1].startswith("-"):
        splice_pos -= 1

    if wasi_world == "wasi:http/service":
        argv[splice_pos:splice_pos] = ["serve", "-Scli", "--addr=127.0.0.1:0"]
        splice_pos += 1

    if wasi_version == "wasm32-wasip3":
        flags_from_proposals = ""
        if "http" in proposals:
            flags_from_proposals += ",http"
        if "sockets" in proposals:
            flags_from_proposals += ",inherit-network"
        argv[splice_pos:splice_pos] = [
            "-Wcomponent-model-async",
            f"-Sp3{flags_from_proposals}",
        ]
