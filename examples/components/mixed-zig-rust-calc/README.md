# mixed-zig-rust-calc

A polyglot WebAssembly component composition: a **Zig** library
component (the [`zig-adder`](../zig-adder)) is linked into a **Rust**
command component (`command/`) via `wasm-tools compose`. The pair
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
zig build-exe -target wasm32-freestanding -O ReleaseSmall \
    -fno-entry --export="docs:adder/add@0.1.0#add" \
    ../zig-adder/src/main.zig
wasm-tools component embed --world adder ../zig-adder/wit main.wasm \
    -o adder.embed.wasm
wasm-tools component new adder.embed.wasm -o zig-adder.component.wasm

# 2. Build the Rust command (wasi-preview1 binary).
(cd command && cargo build --release --target wasm32-wasip1)
wasm-tools component embed --world app command/wit \
    command/target/wasm32-wasip1/release/mixed_zig_rust_command.wasm \
    -o command.embed.wasm
wasm-tools component new command.embed.wasm \
    --adapt wasi_snapshot_preview1.command.wasm \
    -o rust-command.component.wasm

# 3. Compose — wires the Rust command's `docs:adder/add@0.1.0` import
#    against the Zig adder's matching export.
wasm-tools compose -d zig-adder.component.wasm rust-command.component.wasm \
    -o mixed-zig-rust-calc.composed.wasm
```

Or, more concisely, via the repo's root `build.zig`:

```sh
zig build component-examples
```

That step produces `zig-out/component-examples/mixed-zig-rust-calc.composed.wasm`.

## Prerequisites

- Zig 0.16.x (already required for the rest of the repo).
- `wasm-tools` 1.220+ on `PATH`.
- A Rust toolchain with the `wasm32-wasip1` target installed:
  `rustup target add wasm32-wasip1`.
- The `wasi_snapshot_preview1.command.wasm` adapter (downloaded by the
  build step from a pinned Wasmtime release; see the top-level
  [`examples/components/README.md`](../README.md)).

## Runtime status

The composed component validates with `wasm-tools validate` and runs in
[Wasmtime][wasmtime]. On wamr's current preview-state runtime it loads
but does not execute, for the same reason described in
[`../zig-calculator-cmd/README.md`](../zig-calculator-cmd/README.md):
`wasm-tools compose` emits the top-level `wasi:cli/run@0.2.x` export as
an *aliased* instance, while wamr's `registerInstanceExport` currently
only walks `.local` instance refs.

[wasmtime]: https://wasmtime.dev/

## Why this direction (Zig adder + Rust command, not the reverse)

Zig is being demonstrated as a **producer** of reusable component
libraries that drop into an existing polyglot ecosystem. The Rust half
already has a well-documented build path (cargo + `wit-bindgen`); the
Zig half being the implementation of `add` is the more interesting
artifact in a wamr-flavoured tutorial. (The reverse direction —
Rust adder + Zig command — is left as an exercise; it would compose
identically using the same `wasm-tools` recipe.)
