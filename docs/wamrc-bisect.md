# wamrc AOT codegen bisection knobs

Three env vars let you narrow a suspected IR-optimisation miscompile down
to a single (pass, function) pair without recompiling the rest of the
module with a partial pipeline. Introduced for #743 keyvault bisection
(issue #761).

The knobs apply to every `wamrc` invocation — single-module `compile` and
component `compile-component` — and are honoured by every
`runPassesWithOptions` caller.

## Syntax

```text
WAMR_AOT_SKIP_PASS=<spec>           # at most one skip spec
WAMR_AOT_SKIP_PASSES=<spec>[;<spec>...]  # multiple, ';'-separated
WAMR_AOT_PASSES_LIMIT=<limit_spec>  # cap pipeline length
```

```text
spec        := <pass_idx_or_range> [ ':fn=' <fn_list> ]
limit_spec  := <usize>             [ ':fn=' <fn_list> ]
pass_idx_or_range := <usize> [ '-' <usize> ]   # inclusive range
fn_list     := <fn_item> { ',' <fn_item> }
fn_item     := <usize> [ '-' <usize> ]         # inclusive range
```

Pass indices refer to positions in the per-function pipeline returned by
`passes.defaultPassesForTarget(target_arch)` — i.e. the pipeline that
`runPassesWithOptions` iterates per function. The infrastructure passes
(`inlineSmallFunctions`, `promoteLocalsToSSA`, `lowerPhisToLocals`,
`scrubUnreachableBlocks`) live outside that slice and are never affected
by these knobs.

Function indices are 0-based module-level function indices (same numbering
as `--dump-ir-after`'s `func_index` and the AOT runtime's
`local_func[N]` traps).

## Examples

```sh
# Skip pass 15 in every function (module-global; useful first cut).
WAMR_AOT_SKIP_PASS=15 wamrc compile-component …

# Skip pass 15 only in function 11040. Every other function runs the
# full pipeline, so the rest of the module remains correct.
WAMR_AOT_SKIP_PASS=15:fn=11040 wamrc compile-component …

# Skip a pass range across two specific functions (e.g. paired bisects
# on caller + callee).
WAMR_AOT_SKIP_PASS=15-19:fn=11040,11041 wamrc compile-component …

# Two independent skips — pass 15 everywhere AND pass 17 only in fn 11040.
WAMR_AOT_SKIP_PASSES="15;17:fn=11040" wamrc compile-component …

# Cap the per-function pipeline at 30 passes for function 11040 only.
WAMR_AOT_PASSES_LIMIT=30:fn=11040 wamrc compile-component …
```

## Diagnostics

On startup wamrc logs one line per active knob:

```text
warning: [#761 bisect] SKIP pass 15 for 1 func(s)
warning: [#761 bisect] LIMIT pipeline to first 30 passes for 1 func(s)
```

Parse errors degrade gracefully — the offending knob is ignored with a
single warning, and the rest of the pipeline runs unmodified:

```text
warning: [#761 bisect] WAMR_AOT_SKIP_PASS=abc: InvalidNumber — ignoring
```

## Verifier suppression

Per-function: any function whose pipeline is narrowed by the knobs runs
with the IR verifier off, because partial pipelines often leave IR in a
state that later cleanups would normalise and trip benign structural
checks. Unaffected functions still verify normally — most of the
module retains soundness coverage during a bisect.

## Interaction with module-level inlining

`runPassesWithOptions` normally runs **two** outer rounds of
`inlineSmallFunctions` interleaved with per-function passes. When any
bisect knob is active the second round is **suppressed** — otherwise
the second `inlineSmallFunctions` would observe IR that has already
been partially-bisected for the target function and could leak those
effects into callers/callees that were supposed to be unaffected.

Practical consequence: the first round of `inlineSmallFunctions` still
runs (over vanilla post-frontend IR) before any per-function filtering,
so unaffected functions retain the *first*-round inlining behaviour
they would have had without the bisect; affected functions lose any
*second*-round inlining. For a single-pass-skip bisect that's what
users want — every function still gets first-round inlining identical
to a non-bisected baseline.

## Mapping pass indices to pass names

Pass indices are positional in `passes.defaultPassesForTarget(target)`.
To look up a name from an index, grep
`src/compiler/ir/passes.zig` for `default_passes` (target-independent)
or `x86_64_default_passes` (x86_64 overlay) and count from zero. The
`--dump-ir-after=<pass_name>` flag emits the canonical name in each
dump filename, which is the easiest way to confirm the mapping before
running a bisect cycle.

## Limitations / out of scope (Phase 1)

* The knobs do not change pass ordering or repeat counts — only
  inclusion. To experiment with reordering, edit
  `defaultPassesForTarget` and rebuild.
* Selective-recompile (cwasm cache by function) is Phase 2 of #761 — a
  bisect cycle today still recompiles every function in the module.
  With the per-function skip, the cycle's runtime change is restricted
  to one function but every function is still re-codegened.
* The infrastructure passes (`inlineSmallFunctions`,
  `promoteLocalsToSSA`, `lowerPhisToLocals`,
  `scrubUnreachableBlocks`) are not addressable by index — they run
  unconditionally on every function (subject to the
  outer-inlining-round suppression above). `WAMR_AOT_PASSES_LIMIT=0`
  removes every indexed pass but does NOT remove these.
* `fn=<lo>-<hi>` ranges are expanded into an explicit `[]u32` and
  scanned linearly per (function, pass) check, so very wide ranges
  (e.g. `fn=0-10000` on a 12 k-function module) are O(N) per check.
  The parser rejects ranges wider than 1 000 000 entries as a typo
  guard. For "all functions" omit `fn=` entirely.

## Related

* `#743` — the active bug this was built to unblock.
* `#757` — `wamrc verify` differential testing (complementary: oracle).
* `#755` — per-function codegen trace knobs (`trace_select`,
  `trace_write_def_typed`); the optimiser-side counterpart of those
  knobs is what this file documents.
