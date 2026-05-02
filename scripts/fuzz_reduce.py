#!/usr/bin/env python3
"""Reduce a fuzz crasher by repeatedly running the matching ``fuzz-<target>``
harness against shrunken candidates.

The script treats a candidate as "still reproducing" if running the harness
for a short duration over a single-element corpus either exits non-zero or
leaves a named crasher (e.g. ``diff-mismatch-*.wasm``) in the crashes
directory.

Two reduction strategies are attempted in order:

1. ``wasm-tools shrink`` if the binary is on ``PATH``. The predicate is a
   tiny shell wrapper that re-invokes this script in ``--predicate-mode``.
2. A built-in byte-level shrinker that deletes contiguous ranges of bytes
   and accepts the shorter candidate when it still reproduces.

The smallest reproducer is written next to the original input as
``<original>.reduced.wasm`` unless ``--out`` is given.

Usage::

    scripts/fuzz_reduce.py <target> <crasher.wasm> [options]

See ``tests/fuzz/README.md`` for the regression vs. private-security policy
that decides whether a reduced reproducer should land as a regression seed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HARNESS_DIR = REPO_ROOT / "zig-out" / "bin"
NAMED_CRASHER_GLOBS = ("diff-mismatch-*.wasm",)


def harness_path(target: str, harness_dir: Path) -> Path:
    candidate = harness_dir / f"fuzz-{target}"
    if not candidate.exists():
        sys.exit(
            f"fuzz-{target} not found at {candidate}. Run "
            "'zig build fuzz -Doptimize=ReleaseSafe' first."
        )
    return candidate


def reproduces(
    harness: Path,
    candidate_bytes: bytes,
    duration: int,
    extra_args: list[str],
) -> bool:
    """Return True if running the harness against ``candidate_bytes`` either
    aborts non-zero or leaves a named crasher artifact behind.
    """
    with tempfile.TemporaryDirectory(prefix="fuzz-reduce-") as tmp:
        tmp_path = Path(tmp)
        corpus = tmp_path / "corpus"
        crashes = tmp_path / "crashes"
        corpus.mkdir()
        crashes.mkdir()
        (corpus / "candidate.wasm").write_bytes(candidate_bytes)

        argv: list[str] = [
            str(harness),
            "--corpus",
            str(corpus),
            "--crashes",
            str(crashes),
            "--duration",
            str(duration),
            *extra_args,
        ]
        proc = subprocess.run(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode != 0:
            return True
        for pattern in NAMED_CRASHER_GLOBS:
            if any(crashes.glob(pattern)):
                return True
        return False


def byte_level_shrink(
    harness: Path,
    initial: bytes,
    duration: int,
    extra_args: list[str],
    max_passes: int,
) -> bytes:
    """Greedy delta-debug-style byte shrinker.

    Repeatedly halves the deletion window and tries to delete every aligned
    chunk. Accepts the shorter candidate when the harness still reproduces.
    """
    best = initial
    pass_idx = 0
    while pass_idx < max_passes:
        improved = False
        n = len(best)
        chunk = max(1, n // 2)
        while chunk > 0:
            offset = 0
            while offset < len(best):
                end = min(offset + chunk, len(best))
                candidate = best[:offset] + best[end:]
                if candidate and reproduces(harness, candidate, duration, extra_args):
                    print(
                        f"  shrink chunk={chunk} offset={offset}: "
                        f"{len(best)} -> {len(candidate)} bytes",
                        file=sys.stderr,
                    )
                    best = candidate
                    improved = True
                else:
                    offset += chunk
            chunk //= 2
        if not improved:
            break
        pass_idx += 1
    return best


def try_wasm_tools_shrink(
    target: str,
    harness: Path,
    input_path: Path,
    out_path: Path,
    duration: int,
    extra_args: list[str],
) -> bool:
    """Use ``wasm-tools shrink`` when available.

    Returns True if it ran. The predicate script re-invokes this module in
    predicate mode so a single Python file is enough.
    """
    if shutil.which("wasm-tools") is None:
        return False
    with tempfile.TemporaryDirectory(prefix="fuzz-reduce-wt-") as tmp:
        tmp_path = Path(tmp)
        predicate = tmp_path / "predicate.sh"
        predicate.write_text(
            "#!/usr/bin/env bash\n"
            "exec '{python}' '{script}' --predicate-mode "
            "--harness '{harness}' --duration {duration} "
            "{extra} -- \"$1\"\n".format(
                python=sys.executable,
                script=str(Path(__file__).resolve()),
                harness=str(harness),
                duration=duration,
                extra=" ".join(f"--extra {a}" for a in extra_args),
            )
        )
        predicate.chmod(0o755)
        try:
            subprocess.run(
                [
                    "wasm-tools",
                    "shrink",
                    str(predicate),
                    str(input_path),
                    "--output",
                    str(out_path),
                ],
                check=False,
            )
            return out_path.exists()
        except Exception as e:  # noqa: BLE001
            print(f"wasm-tools shrink failed: {e}", file=sys.stderr)
            return False


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", help="fuzz target name (loader, component-loader, interp, aot, diff, canon)")
    parser.add_argument("input", nargs="?", type=Path, help="path to crasher .wasm")
    parser.add_argument("--out", type=Path, default=None, help="output path for reduced reproducer")
    parser.add_argument("--duration", type=int, default=2, help="seconds per predicate run")
    parser.add_argument(
        "--harness-dir",
        type=Path,
        default=DEFAULT_HARNESS_DIR,
        help="directory containing fuzz-<target> binaries",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="extra argv pass-through to the harness (e.g. --extra --fuel --extra 100000)",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=4,
        help="maximum number of byte-level shrink passes",
    )
    parser.add_argument(
        "--no-wasm-tools",
        action="store_true",
        help="skip wasm-tools shrink even if installed",
    )
    parser.add_argument(
        "--predicate-mode",
        action="store_true",
        help="internal: run as a wasm-tools shrink predicate",
    )
    parser.add_argument(
        "--harness",
        type=Path,
        help="internal: path to fuzz-<target> binary (used by --predicate-mode)",
    )
    parser.add_argument(
        "predicate_input",
        nargs="?",
        type=Path,
        help="internal: candidate wasm passed by wasm-tools shrink",
    )
    args = parser.parse_args(argv)

    if args.predicate_mode:
        if not args.harness or not args.predicate_input:
            sys.exit("--predicate-mode requires --harness and a candidate path")
        bytes_ = args.predicate_input.read_bytes()
        ok = reproduces(args.harness, bytes_, args.duration, args.extra)
        return 0 if ok else 1

    if not args.target or not args.input:
        parser.error("target and input are required outside --predicate-mode")
    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")

    harness = harness_path(args.target, args.harness_dir)
    initial = args.input.read_bytes()
    print(f"verifying initial reproduction ({len(initial)} bytes)...", file=sys.stderr)
    if not reproduces(harness, initial, args.duration, args.extra):
        sys.exit("initial input does not reproduce; check target/duration/extra args")

    out = args.out or args.input.with_suffix(".reduced.wasm")
    best = initial

    if not args.no_wasm_tools and shutil.which("wasm-tools") is not None:
        print("attempting wasm-tools shrink...", file=sys.stderr)
        wt_out = out.with_suffix(".wt.wasm")
        if try_wasm_tools_shrink(args.target, harness, args.input, wt_out, args.duration, args.extra):
            wt_bytes = wt_out.read_bytes()
            if reproduces(harness, wt_bytes, args.duration, args.extra) and len(wt_bytes) < len(best):
                print(
                    f"  wasm-tools: {len(initial)} -> {len(wt_bytes)} bytes",
                    file=sys.stderr,
                )
                best = wt_bytes
            wt_out.unlink(missing_ok=True)

    print("running byte-level shrinker...", file=sys.stderr)
    best = byte_level_shrink(harness, best, args.duration, args.extra, args.max_passes)

    out.write_bytes(best)
    print(
        f"done: {len(initial)} -> {len(best)} bytes ({len(best) * 100 // max(1, len(initial))}% of original)",
        file=sys.stderr,
    )
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
