# CoreMark Benchmark

[CoreMark](https://www.eembc.org/coremark) is a simple processor benchmark. This directory
builds and runs CoreMark using Zig's cross-compiler for both native and wasm32-wasi targets.

## Prerequisites

1. Build `wamr` and `wamrc` from the repo root:
   ```
   zig build
   ```
2. Clone the CoreMark source:
   ```
   cd tests/benchmarks/coremark
   git clone https://github.com/eembc/coremark.git
   ```

## Building

```
zig build                          # compile native + wasm
zig build aot                      # compile .wasm → .aot via wamrc
```

## Running

```
zig build run-native               # run native benchmark
zig build run-aot                  # run AOT benchmark via wamr
zig build run-interp               # run interpreter benchmark via wamr
zig build bench                    # run all three
```

## Options

```
zig build -Diterations=1000        # reduce iterations for quick testing
zig build -Dwamrc=/path/to/wamrc   # custom wamrc path
zig build -Dwamr=/path/to/wamr     # custom wamr path
```

## Notes

- The AOT and interpreter modes require `wamrc`/`wamr` to support all wasm opcodes
  used by CoreMark. As the Zig-based toolchain matures, these modes will become functional.
- Native mode works immediately and can be used as the baseline comparison.

