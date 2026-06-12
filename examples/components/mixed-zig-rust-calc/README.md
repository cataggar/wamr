# mixed-zig-rust-calc

A polyglot WebAssembly component composition: a **Zig** library
component (the [`zig-adder`](../zig-adder)) is linked into a **Rust**
command component (`command/`) via `wabt component compose`. The pair
produces a runnable WASI command whose `add` lift is implemented in Zig
and whose CLI / IO logic is implemented in Rust.

This mirrors the polyglot story the bytecodealliance
[component-docs adder/calculator/command tutorial][bca] demonstrates,
filling in the missing Zig column.

[bca]: https://github.com/bytecodealliance/component-docs/tree/main/component-model/examples/tutorial

## Layout

```
mixed-zig-rust-calc/
├── README.md               (this file)
└── command/                Rust half (wit-bindgen + cargo, wasm32-wasip1)
    ├── Cargo.toml
    ├── src/lib.rs          uses `wit_bindgen::generate!` to import docs:adder
    └── wit/
        ├── world.wit       `world app { import docs:adder/add@0.1.0; }`
        └── deps/adder/world.wit
```

The Zig half is reused from [`../zig-adder`](../zig-adder); no Zig sources
live under this directory.

## Build pipeline

```sh
# 1. Build the Zig adder library component (see ../zig-adder/README.md).
#    `wabt component new` embeds the WIT (--wit) and wraps in one step
#    (wabt ≥ v3.0.0-dev.13); a <name>.core.wasm input yields <name>.wasm.
zig build-exe -target wasm32-freestanding -O ReleaseSmall \
    -fno-entry --export="docs:adder/add@0.1.0#add" \
    -femit-bin=zig-adder.core.wasm ../zig-adder/src/main.zig
wabt component new --world adder --wit ../zig-adder/wit zig-adder.core.wasm

# 2. Build the Rust command (wasi-preview1 binary; adapter auto-attached).
(cd command && cargo build --release --target wasm32-wasip1)
cp command/target/wasm32-wasip1/release/mixed_zig_rust_command.wasm \
    rust-command.core.wasm
wabt component new --world app --wit command/wit rust-command.core.wasm

# 3. Compose — wires the Rust command's `docs:adder/add@0.1.0` import
#    against the Zig adder's matching export.
wabt component compose -d zig-adder.wasm rust-command.wasm \
    -o mixed-zig-rust-calc.composed.wasm
```

Or, more concisely, via the repo's root `build.zig`:

```sh
zig build component-examples
```

That step produces `zig-out/component-examples/mixed-zig-rust-calc.composed.wasm`.

## Prerequisites

- Zig 0.16.x (already required for the rest of the repo).
- `wabt` (cataggar/wabt v3.0.0-dev.13+) on `PATH`. The wasi-preview1
  → component adapter is bundled inside `wabt` and auto-attached
  by `wabt component new` — no external adapter download required.
- A Rust toolchain with the `wasm32-wasip1` target installed:
  `rustup target add wasm32-wasip1`.

## Runtime status

The composed component validates with `wabt validate` and runs in both
[Wasmtime][wasmtime] and wamr. `zig build component-examples-run` runs
this example end-to-end through `./zig-out/bin/wamr` and asserts the
expected two-line output.

[wasmtime]: https://wasmtime.dev/

## Why this direction (Zig adder + Rust command, not the reverse)

Zig is being demonstrated as a **producer** of reusable component
libraries that drop into an existing polyglot ecosystem. The Rust half
already has a well-documented build path (cargo + `wit-bindgen`); the
Zig half being the implementation of `add` is the more interesting
artifact in a wamr-flavoured tutorial. (The reverse direction —
Rust adder + Zig command — is left as an exercise; it would compose
identically using the same `wabt` recipe.)
