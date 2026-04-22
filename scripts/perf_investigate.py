#!/usr/bin/env python3
"""Select a performance optimization idea and generate a GitHub issue body.

Usage: python3 scripts/perf_investigate.py

Environment variables (set by the workflow):
  GH_TOKEN      – GitHub token for querying existing issues.
  IDEA_INDEX    – Force a specific idea (0-based index) or "auto".

Reads bench-baseline.txt and spec-baseline.txt from the current directory.
Writes issue-body.md and sets GitHub Actions step outputs.

Called by .github/workflows/perf-investigate.yml.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# ── Curated optimization ideas ──────────────────────────────────────────
# Each idea targets a specific area of the compiler.  The list is pruned
# against the current codebase: ideas already implemented (constant
# folding, CSE, XOR zeroing, CMOV select, fused CMP+JCC, address-mode
# folding, TEST-for-eqz) are excluded.
#
# Inspiration: wasmtime / Cranelift, LLVM, V8 TurboFan, Intel
# Optimization Reference Manual, wasm-micro-runtime upstream.

IDEAS = [
    {
        "id": "dce-reenable",
        "title": "re-enable dead code elimination safely",
        "area": "IR passes",
        "description": (
            "The DCE pass exists in `passes.zig` but is disabled because "
            "it does not track implicit uses of call arguments.  Fix the "
            "use-def analysis to count call-arg operands so DCE can be "
            "safely enabled in the default pass pipeline.\n\n"
            "This should reduce code size and improve I-cache utilisation "
            "by removing instructions whose results are never consumed."
        ),
        "files": ["src/compiler/ir/passes.zig"],
        "reference": (
            "Cranelift DCE: "
            "https://github.com/bytecodealliance/wasmtime/blob/main/"
            "cranelift/codegen/src/dce.rs"
        ),
        "bench_hint": (
            "The existing atomic bench should show smaller code sizes.  "
            "Consider adding a multi-expression benchmark body with "
            "several dead intermediate values to make the effect visible."
        ),
    },
    {
        "id": "strength-reduction-shift",
        "title": "strength reduction — multiply by power-of-2 to shift",
        "area": "IR passes",
        "description": (
            "Add an IR peephole pass that replaces `mul(x, C)` with "
            "`shl(x, log2(C))` when C is a compile-time constant power "
            "of two.  `imul` has 3-cycle latency on modern x86-64; `shl` "
            "is 1 cycle.\n\n"
            "Walk each block's instructions, match `.mul` with one "
            "operand defined by `iconst_32` / `iconst_64`, check "
            "`std.math.log2`, and rewrite in-place."
        ),
        "files": ["src/compiler/ir/passes.zig", "src/compiler/ir/ir.zig"],
        "reference": (
            "Cranelift algebraic opts: "
            "https://github.com/bytecodealliance/wasmtime/blob/main/"
            "cranelift/codegen/src/opts/algebraic.isle"
        ),
        "bench_hint": (
            "Add a benchmark body in `bench_codegen.zig` that builds an "
            "IR function with `mul(vreg, 8)` to measure the improvement."
        ),
    },
    {
        "id": "shift-immediate",
        "title": "shift-immediate specialisation for constant shift counts",
        "area": "x86-64 codegen",
        "description": (
            "x86-64 shift instructions (`shl`, `shr`, `sar`) accept an "
            "immediate byte operand (imm8), but the backend may load the "
            "shift count into CL unconditionally.  When the shift count "
            "is a compile-time constant, emit `shl reg, imm8` directly "
            "to save a register and a MOV.\n\n"
            "Check `compile.zig` for the `.shl`, `.shr_s`, `.shr_u` "
            "cases and add a constant-operand fast path."
        ),
        "files": ["src/compiler/codegen/x86_64/compile.zig"],
        "reference": (
            "Intel SDM Vol. 2, SHL/SHR instruction encoding: immediate "
            "operand form uses opcode /4 ib."
        ),
        "bench_hint": (
            "Add a benchmark body with `shl(vreg, iconst 3)` to measure "
            "code-size and cycle reduction."
        ),
    },
    {
        "id": "bounds-check-hoist",
        "title": "hoist redundant memory bounds checks",
        "area": "x86-64 codegen",
        "description": (
            "Every wasm load/store emits a bounds check against the "
            "linear memory size.  When multiple loads/stores access the "
            "same base with increasing offsets, only the largest offset "
            "needs a check.  Group adjacent memory accesses by base "
            "register and emit a single check for the maximum offset.\n\n"
            "Wasmtime's Cranelift and V8 TurboFan both perform "
            "bounds-check elimination for memory-intensive wasm code."
        ),
        "files": [
            "src/compiler/codegen/x86_64/compile.zig",
            "src/compiler/ir/passes.zig",
        ],
        "reference": (
            "Cranelift heap access: "
            "https://github.com/bytecodealliance/wasmtime/blob/main/"
            "cranelift/wasm/src/heap.rs"
        ),
        "bench_hint": (
            "Add a benchmark body with three consecutive loads from "
            "base+0, base+4, base+8 to measure the check reduction."
        ),
    },
    {
        "id": "bench-expansion",
        "title": "expand benchmark coverage beyond atomics",
        "area": "benchmarks",
        "description": (
            "The current `bench_codegen.zig` only measures atomic "
            "operations.  Many codegen optimisations (register "
            "allocation, instruction selection, constant folding) are "
            "invisible to these benchmarks.\n\n"
            "Add benchmark bodies for:\n"
            "- Binary arithmetic (`add`, `sub`, `mul`, `div_u`)\n"
            "- Comparisons + branches (`br_if` with a condition chain)\n"
            "- Memory load/store sequences\n"
            "- Function calls (`call` + `ret` round-trip)\n"
            "- Mixed integer/float pipelines\n\n"
            "Keep the existing RDTSC-based harness; just add new "
            "`bodyXxx` functions and entries in the benchmarks array."
        ),
        "files": ["src/compiler/bench_codegen.zig"],
        "reference": (
            "Sightglass benchmark suite: "
            "https://github.com/bytecodealliance/sightglass"
        ),
        "bench_hint": (
            "This IS the benchmark improvement — verify that the new "
            "bodies compile and produce plausible cycle counts."
        ),
    },
    {
        "id": "small-memcpy-inline",
        "title": "inline small fixed-size memory.copy and memory.fill",
        "area": "x86-64 codegen",
        "description": (
            "When `memory.copy` or `memory.fill` is called with a "
            "compile-time-known small size (≤ 64 bytes), emit inline "
            "MOV/MOVS sequences instead of calling the runtime helper.  "
            "This avoids function-call overhead and lets the CPU "
            "pipeline the stores.\n\n"
            "LLVM and Cranelift both inline small memcpy/memset.  The "
            "threshold can be tuned; 64 bytes (8 qwords) is typical."
        ),
        "files": ["src/compiler/codegen/x86_64/compile.zig"],
        "reference": (
            "LLVM SelectionDAG memcpy lowering: uses REP MOVSB for "
            "large copies, unrolled MOV for small."
        ),
        "bench_hint": (
            "Add a benchmark with `memory.copy` of 8, 16, 32, 64 bytes "
            "to measure call-vs-inline trade-off."
        ),
    },
    {
        "id": "branch-threading",
        "title": "branch threading for chained conditional jumps",
        "area": "IR passes",
        "description": (
            "If block A branches to block B, and block B's only "
            "instruction is another branch on the same condition, "
            "rewrite A to jump directly to B's target.  This eliminates "
            "one branch and one basic-block transition.\n\n"
            "This pattern is especially common after other optimisation "
            "passes simplify blocks down to just a terminator."
        ),
        "files": ["src/compiler/ir/passes.zig"],
        "reference": (
            "LLVM JumpThreading: "
            "https://llvm.org/docs/Passes.html#jump-threading"
        ),
        "bench_hint": (
            "Add a benchmark body with a chain of `br_if` instructions "
            "on the same condition to measure the effect."
        ),
    },
    {
        "id": "regalloc-hints",
        "title": "register hints for calling-convention constraints",
        "area": "register allocation",
        "description": (
            "x86-64 division requires the dividend in RAX and produces "
            "the quotient in RAX.  The register allocator can avoid "
            "unnecessary MOVs by hinting that div operands/results "
            "should prefer RAX/RDX.\n\n"
            "Similarly, function return values should prefer RAX, and "
            "call arguments should prefer the SysV ABI registers "
            "(RDI, RSI, RDX, RCX, R8, R9).  Adding hints to the "
            "regalloc reduces move-insertion overhead."
        ),
        "files": [
            "src/compiler/ir/regalloc.zig",
            "src/compiler/codegen/x86_64/compile.zig",
        ],
        "reference": (
            "Cranelift regalloc2 hints: "
            "https://github.com/bytecodealliance/regalloc2"
        ),
        "bench_hint": (
            "The existing `atomic_cmpxchg` benchmark exercises the RAX "
            "constraint path — check if cycles/op drops."
        ),
    },
    {
        "id": "loop-invariant-motion",
        "title": "loop-invariant code motion (LICM)",
        "area": "IR passes",
        "description": (
            "Move computations that produce the same result on every "
            "loop iteration into the loop's preheader block.  This "
            "requires loop detection (identify back-edges via DFS on "
            "the CFG) and a dominator-tree query to verify safety.\n\n"
            "LICM is one of the highest-impact optimisations for "
            "numerical wasm workloads (PolyBench, CoreMark)."
        ),
        "files": [
            "src/compiler/ir/passes.zig",
            "src/compiler/ir/analysis.zig",
        ],
        "reference": (
            "Cranelift LICM: "
            "https://github.com/bytecodealliance/wasmtime/blob/main/"
            "cranelift/codegen/src/licm.rs"
        ),
        "bench_hint": (
            "This optimisation is hard to benchmark with the current "
            "micro-bench.  Consider using the spec-test pass rate as "
            "the primary metric, or adding a loop-heavy benchmark body."
        ),
    },
    {
        "id": "lea-compound-addr",
        "title": "peephole — LEA for compound address computation",
        "area": "x86-64 codegen",
        "description": (
            "x86-64 LEA can compute `base + index*scale + displacement` "
            "in a single µop.  When the IR has a chain of "
            "`add(add(base, shl(index, 2)), const_offset)`, the backend "
            "can fuse this into one LEA instead of separate ADD and SHL "
            "instructions.\n\n"
            "This pattern appears frequently in array indexing and "
            "struct field access in wasm linear memory."
        ),
        "files": ["src/compiler/codegen/x86_64/compile.zig"],
        "reference": (
            "Intel Optimization Manual §3.5.1.3: LEA instruction for "
            "address generation."
        ),
        "bench_hint": (
            "Add a benchmark body with "
            "`add(add(base, shl(idx, 2)), 16)` to measure LEA fusion."
        ),
    },
]


def get_tried_ideas() -> set[str]:
    """Query perf-investigation issues to find previously tried idea IDs."""
    try:
        result = subprocess.run(
            [
                "gh", "issue", "list",
                "--label", "perf-investigation",
                "--state", "all",
                "--json", "title",
                "--limit", "200",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        issues = json.loads(result.stdout)
        # Extract idea IDs from titles like "perf: <idea title>"
        titles = {issue["title"] for issue in issues}
        return titles
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()


def select_idea(
    tried_titles: set[str],
    force_index: int | None = None,
) -> dict:
    """Pick the next untried idea, or cycle back to the first."""
    if force_index is not None and 0 <= force_index < len(IDEAS):
        return IDEAS[force_index]

    for idea in IDEAS:
        full_title = f"perf: {idea['title']}"
        if full_title not in tried_titles:
            return idea

    # All ideas tried — start over from the beginning.
    return IDEAS[0]


def read_baseline(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except FileNotFoundError:
        return "(not produced)"


def main() -> None:
    # Parse optional idea index from environment.
    force_index = None
    idx_env = os.environ.get("IDEA_INDEX", "auto")
    if idx_env != "auto":
        try:
            force_index = int(idx_env)
        except ValueError:
            pass

    tried = get_tried_ideas()
    idea = select_idea(tried, force_index)

    bench = read_baseline("bench-baseline.txt")
    spec = read_baseline("spec-baseline.txt")

    title = f"perf: {idea['title']}"

    files_md = ", ".join(f"`{f}`" for f in idea["files"])

    body = f"""\
## Performance Investigation: {idea['title']}

**Area:** {idea['area']}
**Suggested files:** {files_md}

### Current baseline

<details><summary>Codegen bench (cycles/op, code bytes)</summary>

```
{bench[:4000]}
```

</details>

<details><summary>Spec-tests AOT pass rate</summary>

```
{spec}
```

</details>

### What to do

{idea['description']}

### Benchmark tip

{idea['bench_hint']}

### Reference

{idea['reference']}

### Acceptance criteria

1. **`zig build test`** — all existing tests pass
2. **`zig build bench`** — improvement or no regression in cycles/op and code size
3. **`zig build spec-tests-aot`** — pass rate unchanged or improved
4. Add a unit test or benchmark body for the new optimisation if applicable

### How to verify

```bash
zig build -Doptimize=ReleaseFast
zig build bench            # compare cycles/op and code size
zig build spec-tests-aot   # verify pass rate
zig build test             # ensure no regressions
```
"""

    Path("issue-body.md").write_text(body)

    # Set GitHub Actions step outputs.
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"title={title}\n")
            f.write(f"idea_id={idea['id']}\n")

    print(f"Selected: {idea['id']} — {title}")


if __name__ == "__main__":
    main()
