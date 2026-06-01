# Fuzz seed corpus

This directory contains hand-written WAT seeds plus committed `.wasm` encodings for the `fuzz-diff` interpreter-vs-AOT harness. The binaries are committed so `zig build fuzz` does not need an external WAT assembler.

`zig build fuzz` installs the `.wasm` seeds to `zig-out/fuzz-seeds/`. To replay them locally:

```sh
./zig-out/bin/fuzz-diff --corpus-dir=zig-out/fuzz-seeds --duration-ms=10000
```

Each module is importless, declares inline memory, and exports one nullary scalar function whose name matches the file basename. The return value depends on a load after a CFG or instruction barrier so unsafe load forwarding should diverge between interpreter and AOT execution.
