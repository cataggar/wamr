# Regression seeds

Each subdirectory `<target>/` holds minimized fuzz inputs that previously
exposed a real bug, were reduced with `scripts/fuzz_reduce.py`, fixed in
the relevant code path, and locked in here as a permanent regression seed
for `fuzz-<target>`.

## Rules

- Seeds must be deterministic: same bytes, same outcome on every run.
- Each file must be ≤ 4 KB. Larger reproducers belong in workflow
  artifacts, not in the repo.
- Add a one-line filename hint that matches the original issue or PR
  number where the fix landed (e.g., `259-canon-utf16-overrun.wasm`).
- Only add a seed for a **fixed** bug. Inputs that still expose an
  unfixed panic, UB, abort, or potential security bug must be reported
  privately per `SECURITY.md` and `SECURITY_PROCESS.md`, not committed.
- Never include reproducers containing PII, proprietary content, or
  other sensitive bytes the original reporter did not consent to share.

The fuzz workflow seeds these alongside `tests/malformed/fuzz` and
`tests/spec-json` so every PR run replays the regression coverage.
