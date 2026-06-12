# zig-exit

A WASI command component that exits with a nonzero code (issue #436
fixture).

The Zig source defines its own `_start` and calls
`wasi_snapshot_preview1.proc_exit(7)` after writing a short marker line.
The wasm-tools wasi-preview1 component adapter rewrites that
`proc_exit` into `wasi:cli/exit.exit-with-code(7)`, which wamr's
`WasiCliAdapter` captures as the run's numeric exit code.

## Build pipeline

```sh
# 1. Build the core wasm with a custom _start.
zig build-exe -target wasm32-wasi -O ReleaseSmall \
    -fno-entry --export=_start \
    -femit-bin=zig-exit.core.wasm src/main.zig

# 2. Wrap it as a component. `wabt component new` auto-attaches the
#    bundled wasi-preview1 → preview2 adapter when the core imports
#    `wasi_snapshot_preview1.*`; a `<name>.core.wasm` input yields
#    `<name>.wasm`.
wabt component new zig-exit.core.wasm
# -> zig-exit.wasm
```

The repo's root `build.zig` automates both steps:

```sh
zig build component-examples           # build + validate all examples
zig build component-examples-run       # run this one through ./zig-out/bin/wamr
```

## Run

```console
$ ./zig-out/bin/wamr zig-out/component-examples/zig-exit.wasm
exiting with code 7
$ echo $?
7
```

## What this exercises

- The component-model exit-code propagation path:
  guest `proc_exit(rc)` → wasm-tools adapter's exported `proc_exit` →
  `wasi:cli/exit.exit-with-code(rc)` → wamr's `cliExitWithCode`
  (stashes `rc` on the `WasiCliAdapter` and traps) → `RunOutcome.exit_code`
  → `main.zig:runComponent` → host `std.process.exit(rc as u8)`.
  Issue #448 gated wamr's WASI auto-resolution so the adapter route
  actually runs (it used to be clobbered by the bare `wasiProcExit`).
- Pairs with `zig-hello` (normal return → exit code 0) for end-to-end
  coverage of both component exit shapes.
