"""Shared optimize-mode helpers for benchmark scripts."""

from __future__ import annotations

import re
from pathlib import Path

OPTIMIZE_MODES = ("ReleaseFast", "ReleaseSafe")
OPTIMIZE_CHOICES = OPTIMIZE_MODES + ("both",)


def zig_build_command(worktree: Path, optimize: str) -> list[str]:
    manifest = (worktree / "build.zig.zon").read_text()
    match = re.search(r'\.minimum_zig_version\s*=\s*"0\.(16|17)\.[^"]*"', manifest)
    if match is None:
        raise ValueError("benchmark ref must declare Zig 0.16 or 0.17")
    if match.group(1) == "16":
        return ["zig016", "build", f"-Doptimize={optimize}"]
    modes = {
        "Debug": "debug",
        "ReleaseFast": "fast",
        "ReleaseSafe": "safe",
        "ReleaseSmall": "small",
    }
    return ["zig", "build", f"-Doptimize={modes[optimize]}"]


def parse_optimize_modes(value: str) -> list[str]:
    if value == "both":
        return list(OPTIMIZE_MODES)
    if value not in OPTIMIZE_MODES:
        raise ValueError(f"unsupported optimize mode: {value}")
    return [value]


def optimize_slug(value: str) -> str:
    return value.removeprefix("Release").lower()


def fmt_ratio(numerator: float | int | None, denominator: float | int | None) -> str:
    if numerator is None or denominator in (None, 0):
        return "—"
    return f"×{float(numerator) / float(denominator):.2f}"
