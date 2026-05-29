"""Shared optimize-mode helpers for benchmark scripts."""

from __future__ import annotations

OPTIMIZE_MODES = ("ReleaseFast", "ReleaseSafe")
OPTIMIZE_CHOICES = OPTIMIZE_MODES + ("both",)


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
