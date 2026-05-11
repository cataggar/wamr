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
    src/main.zig

# 2. Wrap it as a component using the wasi-preview1 command adapter.
wabt component new main.wasm \
    --adapt wasi_snapshot_preview1=wasi_snapshot_preview1.command.wasm \
    -o zig-exit.component.wasm
```

The repo's root `build.zig` automates both steps:

```sh
zig build component-examples           # build + validate all examples
zig build component-examples-run       # run this one through ./zig-out/bin/wamr
```

## Run

```console
$ ./zig-out/bin/wamr zig-out/component-examples/zig-exit.component.wasm
exiting with code 7
$ echo $?
7
```

## What this exercises

- The component-model exit-code propagation path added in #436:
  `proc_exit(rc)` → wamr's `wasiProcExit` (the import is bound via the
  wasm-tools wasi-preview1 adapter, but wamr's wasi auto-resolution
  currently dispatches it directly) → `ModuleInstance.exit_code_sink`
  → `WasiCliAdapter.exit_code` → `RunOutcome.exit_code` →
  `main.zig:runComponent` → host `std.process.exit(rc as u8)`.
- Pairs with `zig-hello` (normal return → exit code 0) for end-to-end
  coverage of both component exit shapes.
