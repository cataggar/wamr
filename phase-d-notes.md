# Phase D diagnosis — composed components on AOT (#662)

## Status

**Partial progress, blocked on AOT canon-lower trampoline plumbing.**
Loader/CLI fix landed in this branch; the wasi-p2 skip entries for the
two composed-component fixtures (`zig-calculator-cmd.composed.wasm`,
`mixed-zig-rust-calc.composed.wasm`) and the `build.zig` `wireComponentRun`
skip flags remain in place because the AOT runtime cannot yet bridge the
cross-sub-component function imports those fixtures introduce.

## Reproduction (after this branch's loader fix)

```
zig build install component-examples
./zig-out/bin/wamr run zig-out/component-examples/zig-calculator-cmd.composed.wasm
```

```
wamr: auto-precompiling 6 cores in memory (no on-disk bundle; pass --precompiled-dir or run `wamrc compile-component` to persist)
warning: [aot reject] core module 0 import 'docs:adder/add@0.1.0.add' (function) cannot be resolved by AOT yet (#644)
Error: failed to instantiate component (the `wamr` CLI runs all components through the AOT runtime; see issue #644)
```

Wasmtime parity (control):

```
wasmtime run -S cli-exit-with-code zig-out/component-examples/zig-calculator-cmd.composed.wasm
40 + 2 = 42
100 + 200 = 300  (exit 0)
```

## Root cause

`wabt component compose -d` produces a top-level component shaped like:

```
component
├── component  (the original `wabt component new` calculator-cmd, with 5 cores)
├── component  (the original `wabt component new` adder,          with 1 core)
├── instance  (instantiate calculator-cmd sub-component)
├── instance  (instantiate adder sub-component)
└── alias / export  (re-expose wasi:cli/run from the inner instance)
```

The actual core modules — including the calculator-cmd's main wasip1
core whose `_start` does the printing — live **inside the nested
sub-components**, not at the top level. So:

1. `wamr.component_loader.load(...)` returns a top-level `Component`
   with `core_modules.len == 0` and `core_instances.len == 0`. Each
   element of `components[]` recursively holds the real cores.
2. Sub-component instantiation in `src/component/instance.zig`
   (the `sub_instances` loop at the end of `instantiateWithOptions`)
   recurses correctly on the interpreter, because each sub-component
   has its own non-empty `core_instances` that drive the
   `module_arena`-backed interpreter path.
3. The wamr `run` CLI is AOT-only (#644), so it needs every core in
   every nested sub-component to be precompiled and reachable through
   `Options.precompiled_cores`.

## What this branch does fix

Two narrow changes, both required by any future complete fix:

1. **`src/main.zig` auto-precompile walker.** Recursively walks
   `parsed.components` to discover every core module across the
   sub-component tree, compiles each, and emits a `PrecompiledCore`
   entry per core.

2. **`src/component/core_backend.zig` `PrecompiledCore` scoping.**
   Adds an optional `core_wasm: ?[]const u8` raw-bytes identity to
   `PrecompiledCore`. `Options.findPrecompiled` now takes the
   `component.core_modules[idx].data` slice plus the local
   `module_idx`. Slice identity is stable across re-parses of the same
   `wasm_data` (both `main.zig` and `wasi_cli_adapter.zig` re-parse
   the same buffer), so a single `precompiled_cores` slice can be
   shared by the root component and every sub-component without their
   local `module_idx` values colliding.

3. **`src/component/instance.zig` sub-instance options propagation.**
   The `sub_instances` loop now uses
   `instantiateWithOptions(subcomp, allocator, inst.options)` instead
   of the bare `instantiate(...)`, so nested sub-components inherit
   the parent's `precompiled_cores` + `aot_only` flag.

With (1)–(3) in place, the loader/CLI side of Phase D is done: every
core in a composed-component tree gets discovered, compiled, and the
matching cwasm artifact reaches the right sub-component instantiation
point.

## What still blocks running the fixtures

The calculator-cmd's main wasip1 core, after composition, imports a
**non-WASI** function:

```
import function docs:adder/add@0.1.0.add : (i32, i32) -> (i32);
```

The composer wires this import to a `canon.lower` of the adder
sub-component's `add` export. On the **interpreter** side this is
handled by installing a `componentTrampoline` + per-slot
`ComponentTrampolineCtx` on the importing core's `host_func_entries`
slot, which bridges the core call back into the host `HostFunc` that
lifts arguments, calls the adder core, and lowers the result.

On the **AOT** side, that trampoline plumbing is explicitly deferred
per `src/component/instance.zig:1659-1665`:

> Phase 1: AOT cores skip cross-instance import wiring /
> canon-lower trampoline plumbing (handled by the interp path below
> for interp cores). Feasibility check above guarantees we only
> commit to AOT for cores that don't need any of that. Richer
> wiring lands in a follow-up.

The pre-flight feasibility check (`firstUnsupportedAotImport`,
`src/component/instance.zig:2546`) only accepts function imports
whose `module_name` is a known WASI or spectest host; everything
else — including all composed component-level WIT imports — is
rejected and surfaces `error.AotImportUnresolvable` under
`aot_only = true`.

## Estimated work for the full fix

Adding canon-lower trampoline support on AOT is the missing piece.
Sketch:

- Extend `src/runtime/aot/host_trampolines.zig`: the existing
  `TrampolinePool` already produces `componentTrampoline`-style stubs
  for canon-lowered AOT functions inside a single core. The composed
  case needs the same stub family to be wired as *imports* on the
  importing AOT core's import-function table — i.e., for each
  unresolved function import that matches a `canon.lower`-backed
  sibling sub-component export, install a pool slot pointing at the
  sibling's lifted `HostFunc` and patch the AOT instance's import
  slot to call into it.
- Relax `firstUnsupportedAotImport` to recognize those resolvable
  function imports (need to consult the parent component's
  canon/instance index spaces, which means the feasibility check has
  to take the `Component*` and the `with`-args of the importing
  `core_instances.instantiate`, not just the parsed `AotModule`).
- Add cross-instance memory/table sharing for AOT cores that the
  current code path already supports through
  `imported_table_overrides` / `imported_memory_overrides` —
  composed-component cores share memory across cores in the same
  sub-component (e.g. the calculator-cmd has 5 cores that need to
  reach the main memory), so this must work end-to-end on AOT.

Realistic LoC: ~400–800 in `src/runtime/aot/`, `src/component/
instance.zig`, and `src/runtime/aot/host_trampolines.zig`, plus
fixtures and tests. Substantially more than the ~200-line cap in
the Phase D prompt.

## Recommended split

- **This branch (`wamr-662-d`)**: land the loader/CLI walker +
  `PrecompiledCore` scoping. They are safe no-ops for every
  non-composed fixture and the only reasonable foundation for the
  AOT trampoline work. Keep the wasi-p2 skip entries + the
  `build.zig` gates intact (do *not* drop them yet).
- **Follow-up (issue #662 phase D2)**: AOT canon-lower trampolines
  for cross-sub-component function imports. Drops the skip entries +
  flips the gates.
