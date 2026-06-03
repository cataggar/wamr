---
name: aot-pass-bisect
description: Localize which IR optimization pass introduces an AOT codegen miscompile in a specific function. Use after the aot-diff-debug skill has narrowed the bug to a known function index but the responsible IR pass is unknown. Combines WAMR_AOT_SKIP_PASS env knobs (#762) with --cache / --cache-dir codegen reuse (#763) so each bisect cycle re-codegens only the suspect function and finishes in seconds instead of minutes.
---
# AOT IR-Pass Bisection

Find the IR optimization pass responsible for an AOT miscompile in a
specific function. This skill applies the **bisect knobs** added for
issues #743 / #761 — once `aot-diff-debug` has identified a function
that codegen mishandles, this skill narrows the responsibility down
to a specific IR pass without spending 8-40 min per recompile cycle.

## When to Use

Use this skill when ALL of the following hold:

- An AOT-compiled program produces wrong output (caught by
  `aot-diff-debug` or by failing test output).
- The bug **goes away with `-O0`** (single-module) or
  `WAMR_AOT_PASSES_LIMIT=0` (any path) — proving the bug is in the
  optimization pipeline, not the frontend / SSA / codegen.
- You know (from `aot-diff-debug` Phase 3 or runtime `local_func[N]`
  trap output) WHICH function carries the bug, but not WHICH pass
  introduced it.
- You're working with a **large component** (the keyvault repro for
  #743 has ~12 k functions, recompile = 8-40 min) where re-running
  the full pipeline per bisect cycle is the dominant cost.

Do **not** use this skill when:

- The bug reproduces with `-O0` — that means it's in the frontend /
  SSA promotion / regalloc / codegen, not the passes. Go to
  `aot-diff-debug` instead.
- You don't yet know which function is at fault — find that first.

## Prerequisite: localize the function

If you only know "AOT output is wrong somewhere" and not which
function trapped or computed a wrong value, run the `aot-diff-debug`
skill first to narrow to a specific `local_func[N]`. Pass indices
without a target function are still useful but produce many false
positives across the module's 12 k functions.

When wamr's AOT runtime traps it prints a `local_func[N]` decoration
on the stack trace. Save that N — it's the bisect target.

## Workflow

### Step 1: Pin baseline behaviour

Before bisecting, prove the bug is in the optimization pipeline and
not somewhere else. Two cheap probes:

```sh
# Probe 1: does -O0 fix it?  If yes → bug is in optimization passes.
wamrc compile-component -O0 input.wasm -o out.cwasm.json
wamr <out> | check-output  # expect: bug gone

# Probe 2: does limiting to 0 passes fix it?  Same answer.
WAMR_AOT_PASSES_LIMIT=0 wamrc compile-component input.wasm -o out.cwasm.json
wamr <out> | check-output  # expect: bug gone
```

If both probes still reproduce the bug, the responsible code is NOT
in the optimization pipeline — go back to `aot-diff-debug`.

### Step 2: Set up the codegen cache

Bisecting without `--cache-dir` recompiles every function (12 k+ for
keyvault) on every cycle, which is the 8-40 min cost the issue
flagged. With `--cache-dir`, only the function whose pipeline you
change re-codegens — every other function reuses its cached bytes.

```sh
# Single-module compile path.
mkdir -p /tmp/bisect-cache
wamrc compile --cache /tmp/bisect-cache/foo.cache \
  input.wasm -o out.cwasm
# Output line: "Codegen cache: 0 reused, K recompiled (of K functions)"

# Component compile path (recommended for the #743 keyvault shape).
mkdir -p /tmp/bisect-cache
wamrc compile-component --cache-dir /tmp/bisect-cache \
  input.wasm -o out.cwasm.json
# Per-core cache files written as /tmp/bisect-cache/core0.cache etc.
```

The first invocation is full-cost (populates the cache). Every
invocation after that with the same `--cache-dir` runs in
seconds-per-bisect-cycle territory IF you only narrow the pipeline
for one function via `WAMR_AOT_SKIP_PASS=...:fn=<N>`.

Cache mismatches (different wamrc build, different `--target`,
changed module structure) are detected via header checks and degrade
to a clean "full recompile" with a warning. They're never errors —
safe to leave `--cache-dir` set across unrelated runs.

### Step 3: Identify the pipeline length

```sh
# Look up how many passes the current target has.
grep -nE "pub fn defaultPassesForTarget" src/compiler/ir/passes.zig
# Then count entries in `default_passes` / `x86_64_default_passes` (or
# read off the warning text wamrc emits when WAMR_AOT_PASSES_LIMIT is
# overshot).
```

For x86_64 the pipeline is roughly 20-30 passes. Each pass has a
stable index in the slice returned by `defaultPassesForTarget`.

### Step 4: Binary-search the responsible pass

Use `WAMR_AOT_SKIP_PASS=<idx>:fn=<N>` to drop pass `<idx>` ONLY for
the suspect function. Every other function keeps the full pipeline,
so the rest of the module remains correct. Combined with
`--cache-dir`, each cycle only re-codegens function N (≈ 1 second).

Two-question binary search per round:

```sh
# Helper macro: each cycle is one wamrc invocation + one wamr run.
bisect_cycle() {
  local pass_idx="$1"
  WAMR_AOT_SKIP_PASS="${pass_idx}:fn=11040" \
    wamrc compile-component --cache-dir /tmp/bisect-cache \
    input.wasm -o out.cwasm.json 2>&1 | grep "bisect\|reused"
  wamr out.cwasm.json | check-output
}

# Round 1 — narrow to first or second half of the pipeline.
bisect_cycle 0-14    # if bug GONE → responsible pass is in 0..14
bisect_cycle 15-29   # else                                in 15..29

# Round 2 — narrow further on whichever half the bug came from.
bisect_cycle 0-7     # subdivide
# ...repeat ~log₂(N_passes) ≈ 5 cycles total to localize one pass.
```

A "bug GONE" cycle confirms the responsible pass is in the skipped
range. A "bug STILL REPRODUCES" cycle confirms it's elsewhere.

Single-pass spec form (the final localization step):

```sh
WAMR_AOT_SKIP_PASS="15:fn=11040" wamrc compile-component --cache-dir ...
```

If the bug disappears with exactly one pass skipped, that pass is
the culprit. Confirm by reading the pass name from the warning
wamrc emits at startup:

```
warning: [#761 bisect] SKIP pass 15 for 1 func(s)
```

Map pass index → name by counting entries in
`src/compiler/ir/passes.zig:defaultPassesForTarget` or by enabling
`--dump-ir-after=<pass_name>` (which prints the canonical name in
each dump filename).

### Step 5: Confirm with a minimal reproducer

Once you've named the pass:

1. Dump pre-pass and post-pass IR for the suspect function:
   ```sh
   wamrc compile --dump-ir-after=initial \
     --dump-ir-after=<bad_pass_name> \
     --dump-ir-functions='func11040*' \
     --dump-ir-out=/tmp/ir-dumps \
     input.wasm -o /dev/null
   ```
2. Diff the two snapshots — the rewrite the pass performed on
   function 11040 IS the bug (or at least sets up the bug).
3. Hand the diff + the bisect spec
   (`WAMR_AOT_SKIP_PASS=15:fn=11040` makes the bug go away) to a
   maintainer or use it to build a unit test for the
   passes_<pass_name>_test.zig sibling.

## Multi-spec and pass-range syntax

```sh
# Single pass index.
WAMR_AOT_SKIP_PASS=15

# Pass index + function filter.
WAMR_AOT_SKIP_PASS=15:fn=11040

# Multiple functions for the same pass.
WAMR_AOT_SKIP_PASS=15:fn=11040,11041,11050-11055

# Inclusive pass-index range (skip 15 through 19 inclusive).
WAMR_AOT_SKIP_PASS=15-19:fn=11040

# Several independent skips (use the plural form, ';'-separated).
WAMR_AOT_SKIP_PASSES="15;17:fn=11040;12-15"

# Cap the per-function pipeline at the first N passes.
WAMR_AOT_PASSES_LIMIT=30:fn=11040
```

Atomic parsing: a typo anywhere in a `WAMR_AOT_SKIP_PASSES` spec
discards the entire spec with a one-line warning and falls back to
the full pipeline. No silent partial application.

## What the cache reuses (and what it does not)

The `--cache-dir` cache stores **per-function codegen output**
(native bytes + inter-function call patches). On a recompile cycle:

- Functions with **unchanged IR** hit the cache and skip codegen
  entirely — these are >99% of the module during a single-function
  bisect.
- Functions with **changed IR** (your bisect target) re-run the
  filtered pipeline and re-codegen.
- The IR optimization passes themselves are **not cached** — they
  still run for every function every cycle. This is fast for most
  passes; the dominant cost on large modules is regalloc + codegen,
  which IS cached.

Time savings estimate: bisect cycle drops from 8-40 min to ~1-20
min on the #743 keyvault shape (codegen ≈ 50% of total compile
time). Full pass-bisection across 25 cycles: ~3-17 hours → ~30 min
- 8 hours.

## Anti-Patterns

| Anti-Pattern | Why It Wastes Time | Do Instead |
|---|---|---|
| Bisecting without `--cache-dir` | Every cycle re-codegens all 12 k functions | Set up the cache first (Step 2) |
| Bisecting without `:fn=<N>` | Module-global skips can mask the bug behind unrelated false negatives or trip other functions | Always scope the skip to the suspect function |
| Skimming the cwasm bytes for clues | Codegen output is opaque post-regalloc | Dump IR before/after the suspect pass via `--dump-ir-after` |
| Re-running a failing cycle "to make sure" | Bisect knobs are deterministic | Trust the result; spend the cycle on the next subdivision |
| Stashing `WAMR_AOT_SKIP_PASS` in your shell rc | The next unrelated `wamrc` invocation will silently apply it | Use a one-shot `WAMR_AOT_SKIP_PASS=... wamrc ...` form per cycle |
| Verifying pass-by-pass without first confirming `-O0` reproduces a fix | Bug might be in frontend or codegen; bisect won't converge | Always run Step 1 first |

## References

- `docs/wamrc-bisect.md` — full reference for the `WAMR_AOT_SKIP_PASS{,ES}`
  / `WAMR_AOT_PASSES_LIMIT` knobs (syntax, semantics, verifier
  suppression, inlining-round caveat).
- `src/compiler/ir/passes.zig` — `defaultPassesForTarget` is the
  source of truth for pass indices; the per-function fixpoint in
  `runPassesWithOptions` is where the bisect filter is consumed.
- `src/compiler/aot_bisect.zig` — env-var parser + process-global
  `PassBisectSpec` (PR #762).
- `src/compiler/codegen_cache.zig` — canonical IR hasher + cache
  file format (PR #763).
- Issue #743 — the keyvault miscompile this workflow was built to
  unblock.
- Issue #761 — the meta-issue covering the bisect-knob + cache
  feature.

## Companion skill

If you don't yet know which function is at fault, run the
**`aot-diff-debug`** skill first. That skill compares interpreter vs
AOT output to identify the divergence and narrow to a specific
function; this skill then takes over to localize the responsible
pass.
