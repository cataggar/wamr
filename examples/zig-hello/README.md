# zig-hello

A minimal `wasi:cli` command component written in Zig. It exports
`wasi:cli/run@0.2.6`, prints a greeting to stdout, and returns a process
exit code. The canonical-ABI plumbing lives in the shared `wasi_cli`
helper (`src/guest/wasi_cli.zig`).

## Build

```sh
zig build examples       # build + validate all examples
```

Produces `zig-out/examples/zig-hello.wasm`.

## Run

```sh
# wamr
wamr run zig-out/examples/zig-hello.wasm

# wasmtime
wasmtime run -S cli-exit-with-code zig-out/examples/zig-hello.wasm
```

```console
hello from zig component
```

Both runtimes print the greeting and exit with the code returned by
`run()` (`0`).

