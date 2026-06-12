# zig-hello

A minimal WASI command component written in Zig.

The Zig source defines its own `_start` and writes a greeting via
`wasi_snapshot_preview1.fd_write`. The wasi-preview1 component adapter
wraps that core `_start` into a `wasi:cli/run` instance export with no
extra component-level imports beyond the standard CLI surface.

## Build pipeline

```sh
# 1. Build the core wasm with a custom _start.
zig build-exe -target wasm32-wasi -O ReleaseSmall \
    -fno-entry --export=_start \
    -femit-bin=zig-hello.core.wasm src/main.zig

# 2. Wrap it as a component. `wabt component new` auto-attaches the
#    bundled wasi-preview1 → preview2 adapter when the core imports
#    `wasi_snapshot_preview1.*`; a `<name>.core.wasm` input yields
#    `<name>.wasm`.
wabt component new zig-hello.core.wasm
# -> zig-hello.wasm
```

The repo's root `build.zig` automates both steps:

```sh
zig build examples           # build + validate all examples
zig build examples-run       # run this one through ./zig-out/bin/wamr
```

## Run

```console
$ ./zig-out/bin/wamr zig-out/examples/zig-hello.wasm
hello from zig component
```

This example runs end-to-end through wamr today (exit code 0).

## What this exercises

- Zig 0.16 producing a `wasm32-wasi` core module with no use of
  `std.start.startWasi` (so no `proc_exit`).
- `wabt component new --adapt` wrapping a preview1 core into a
  Component-Model component.
- wamr's `wasi:cli/run` instance-export lifting and `wasi:cli/stdout`
  + `wasi:io/streams` host adapters.

## Why a custom `_start` instead of `pub fn main`

Zig 0.16's WASI command prologue calls `std.os.wasi.proc_exit(rc)` when
`main` returns. Through the preview1 adapter, that becomes a component
trap unless the runtime translates it back into
`wasi:cli/exit.exit-with-code(rc)`. wamr's CLI exit-code path for
preview1 `proc_exit` is incomplete in the current preview, so we just
return from `_start` without calling `proc_exit` to keep the example's
outcome unambiguous on this runtime. Other runtimes (e.g., Wasmtime)
accept either shape.
