#!/usr/bin/env python3
"""One-shot Zig mutation-testing helper.

The tool mutates one source location at a time, runs a test command, records
whether the command failed (KILLED) or passed (SURVIVED), and restores the
original target file after every mutant and on process exit.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import html
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
from typing import Iterable

DEFAULT_TEST_CMD = (
    'unset ZIG_LOCAL_CACHE_DIR; '
    'export ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache"; '
    'zig build test --summary all 2>&1 | tail -5'
)

CONTROL_RE = re.compile(r"^(?P<indent>\s*)(?P<kw>continue|break|return);\s*(?://.*)?$")
COMPARISON_RE = re.compile(r"(?<![<>=!])(?:<=|>=|<|>)(?![<>=])")
INT_LITERAL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<value>[01])(?![A-Za-z0-9_])(?=\s*(?:[;,)}]|[+\-*/%<>=!&|?:]))"
)
OFF_BY_ONE_RE = re.compile(r"(?P<op>[+\-])\s+1\b")


@dataclasses.dataclass(frozen=True)
class Mutant:
    line_no: int
    original_line: str
    mutated_line: str
    mutator: str
    detail: str


@dataclasses.dataclass(frozen=True)
class Result:
    mutant: Mutant
    status: str
    exit_code: int
    seconds: float
    output_tail: str


def split_code_comment(line: str) -> tuple[str, str]:
    """Split at a Zig line comment outside ordinary string literals."""
    in_string = False
    escaped = False
    i = 0
    while i < len(line) - 1:
        ch = line[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "/" and line[i + 1] == "/":
            return line[:i], line[i:]
        i += 1
    return line, ""


def replace_span(text: str, start: int, end: int, replacement: str) -> str:
    return text[:start] + replacement + text[end:]


def in_ranges(line_no: int, ranges: list[tuple[int, int]]) -> bool:
    return not ranges or any(start <= line_no <= end for start, end in ranges)


def line_mutants(line: str, line_no: int) -> Iterable[Mutant]:
    code, comment = split_code_comment(line)
    newline = "\n" if line.endswith("\n") else ""
    line_body = line[:-1] if newline else line
    stripped = line_body.strip()

    if "clearAll(" in code and stripped.endswith(";"):
        indent = line_body[: len(line_body) - len(line_body.lstrip())]
        mutated = f"{indent}// mutation_test: no-op {line_body.lstrip()}{newline}"
        yield Mutant(line_no, line.rstrip("\n"), mutated, "clearAll -> no-op", "comment out clearAll line")

    for match in COMPARISON_RE.finditer(code):
        op = match.group(0)
        replacement = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}[op]
        mutated = replace_span(code, match.start(), match.end(), replacement) + comment
        yield Mutant(
            line_no,
            line.rstrip("\n"),
            mutated,
            f"comparison {op} -> {replacement} at column {match.start() + 1}",
            "swap relational operator",
        )

    control = CONTROL_RE.match(line_body)
    if control:
        indent = control.group("indent")
        kw = control.group("kw")
        mutated = f"{indent}// mutation_test: deleted {kw};{newline}"
        yield Mutant(line_no, line.rstrip("\n"), mutated, f"delete {kw};", "delete standalone control statement")

    for match in INT_LITERAL_RE.finditer(code):
        value = match.group("value")
        replacement = "1" if value == "0" else "0"
        mutated = replace_span(code, match.start("value"), match.end("value"), replacement) + comment
        yield Mutant(
            line_no,
            line.rstrip("\n"),
            mutated,
            f"integer {value} -> {replacement} at column {match.start('value') + 1}",
            "flip zero/one integer literal",
        )

    for match in OFF_BY_ONE_RE.finditer(code):
        op = match.group("op")
        replacement = "- 1" if op == "+" else "+ 1"
        mutated = replace_span(code, match.start(), match.end(), replacement) + comment
        yield Mutant(
            line_no,
            line.rstrip("\n"),
            mutated,
            f"off-by-one {op} 1 -> {replacement} at column {match.start() + 1}",
            "flip +/- 1",
        )


def collect_mutants(lines: list[str], ranges: list[tuple[int, int]]) -> list[Mutant]:
    mutants: list[Mutant] = []
    for idx, line in enumerate(lines, start=1):
        if not in_ranges(idx, ranges):
            continue
        mutants.extend(line_mutants(line, idx))
    return mutants


def write_target_atomically(target: Path, text: str) -> None:
    tmp = target.with_name(f".{target.name}.mutation-test.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


def tail(text: str, lines: int = 5) -> str:
    return "\n".join(text.splitlines()[-lines:])


def run_command(command: str, cwd: Path, timeout_seconds: float | None) -> tuple[int, str]:
    process = subprocess.Popen(
        ["bash", "-o", "pipefail", "-c", command],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
        return process.returncode, stdout
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate()
        stdout = (stdout or "") + f"\nmutation_test.py: timed out after {timeout_seconds} seconds"
        return 124, stdout


def markdown_escape(value: str) -> str:
    value = value.replace("\n", "\\n")
    value = value.replace("|", "\\|")
    return html.escape(value)


def render_report(
    target: Path,
    command: str,
    seed: int | None,
    limit: int | None,
    ranges: list[tuple[int, int]],
    total_available: int,
    results: list[Result],
) -> str:
    survived = sum(1 for result in results if result.status == "SURVIVED")
    killed = sum(1 for result in results if result.status == "KILLED")
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    range_text = ", ".join(f"{start}:{end}" for start, end in ranges) if ranges else "all lines"
    sorted_results = sorted(results, key=lambda r: (0 if r.status == "SURVIVED" else 1, r.mutant.line_no, r.mutant.mutator))

    try:
        target_text = target.relative_to(Path.cwd()).as_posix()
    except ValueError:
        target_text = target.as_posix()

    out = [
        f"# Mutation test report: `{target_text}`",
        "",
        f"Generated: {now}",
        "",
        "## Summary",
        "",
        f"- Result: {survived} survived / {killed} killed / {len(results)} executed",
        f"- Mutants available before sampling: {total_available}",
        f"- Seed: {seed if seed is not None else 'none'}",
        f"- Limit: {limit if limit is not None else 'none'}",
        f"- Line ranges: {range_text}",
        f"- Test command: `{command}`",
        "",
        "## Mutants",
        "",
        "SURVIVED rows are sorted first.",
        "",
        "| Result | Line | Mutator | Original line | Exit | Seconds | Output tail |",
        "| --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for result in sorted_results:
        out.append(
            "| {status} | {line} | {mutator} | `{original}` | {exit_code} | {seconds:.1f} | `{output}` |".format(
                status=result.status,
                line=result.mutant.line_no,
                mutator=markdown_escape(result.mutant.mutator),
                original=markdown_escape(result.mutant.original_line.strip()),
                exit_code=result.exit_code,
                seconds=result.seconds,
                output=markdown_escape(result.output_tail),
            )
        )
    out.append("")
    return "\n".join(out)


def parse_ranges(values: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for value in values:
        if ":" not in value:
            raise argparse.ArgumentTypeError(f"expected START:END, got {value!r}")
        start_s, end_s = value.split(":", 1)
        start = int(start_s)
        end = int(end_s)
        if start < 1 or end < start:
            raise argparse.ArgumentTypeError(f"invalid range {value!r}")
        ranges.append((start, end))
    return ranges


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one-shot mutation tests for a Zig source file.")
    parser.add_argument("target", type=Path, help="Zig source file to mutate")
    parser.add_argument("--seed", type=int, help="Shuffle mutants with this deterministic seed")
    parser.add_argument("--limit", type=int, help="Maximum mutants to execute after optional shuffling")
    parser.add_argument("--test-cmd", default=DEFAULT_TEST_CMD, help="Shell command whose non-zero exit kills a mutant")
    parser.add_argument("--out", type=Path, help="Markdown report path")
    parser.add_argument("--timeout", type=float, help="Seconds before a mutant test command is killed")
    parser.add_argument(
        "--line-range",
        action="append",
        default=[],
        metavar="START:END",
        help="Only generate mutants in this inclusive line range; may be repeated",
    )
    parser.add_argument("--list", action="store_true", help="List selected mutants without running tests")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    target = args.target.resolve()
    if not target.exists():
        print(f"target does not exist: {target}", file=sys.stderr)
        return 2
    if target.suffix != ".zig":
        print(f"target is not a .zig file: {target}", file=sys.stderr)
        return 2

    ranges = parse_ranges(args.line_range)
    original = target.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    mutants = collect_mutants(lines, ranges)
    total_available = len(mutants)
    if args.seed is not None:
        rng = random.Random(args.seed)
        rng.shuffle(mutants)
    if args.limit is not None:
        mutants = mutants[: args.limit]

    if args.list:
        for idx, mutant in enumerate(mutants, start=1):
            print(f"{idx}: line {mutant.line_no}: {mutant.mutator}: {mutant.original_line.strip()}")
        print(f"selected {len(mutants)} of {total_available} mutants")
        return 0

    interrupted = False

    def restore_and_signal(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        write_target_atomically(target, original)
        raise KeyboardInterrupt(f"received signal {signum}")

    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, restore_and_signal)

    results: list[Result] = []
    try:
        for idx, mutant in enumerate(mutants, start=1):
            mutated_lines = list(lines)
            mutated_lines[mutant.line_no - 1] = mutant.mutated_line
            write_target_atomically(target, "".join(mutated_lines))
            start = time.monotonic()
            exit_code, output = run_command(args.test_cmd, Path.cwd(), args.timeout)
            seconds = time.monotonic() - start
            status = "SURVIVED" if exit_code == 0 else "KILLED"
            results.append(Result(mutant, status, exit_code, seconds, tail(output)))
            print(
                f"[{idx}/{len(mutants)}] {status} line {mutant.line_no}: {mutant.mutator} ({seconds:.1f}s)",
                flush=True,
            )
            write_target_atomically(target, original)
    finally:
        write_target_atomically(target, original)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    report = render_report(target, args.test_cmd, args.seed, args.limit, ranges, total_available, results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)

    survived = sum(1 for result in results if result.status == "SURVIVED")
    print(f"summary: {survived} survived / {len(results) - survived} killed / {len(results)} executed")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
