# zig-calculator-cmd

A Zig WASI command component that imports the
[`docs:adder@0.1.0`](../zig-adder) library component and prints two
sample sums to stdout. This mirrors the `command` half of the
bytecodealliance [component-docs adder/calculator/command tutorial][bca]
(simplified to skip the `calculator` middle layer) but written entirely
in Zig.

[bca]: https://github.com/bytecodealliance/component-docs/tree/main/component-model/examples/tutorial

## What it imports / exports

```wit
package docs:zigcalc@0.1.0;

world app {
    import docs:adder/add@0.1.0;
}
```

The Zig source declares the import directly with `extern "pkg:ns/iface@ver"`:

```zig
extern "docs:adder/add@0.1.0" fn add(x: u32, y: u32) u32;
```

After `wabt component new --world app …` the import becomes a
component-level `docs:adder/add@0.1.0::add` import, which any
implementation of the adder world (including
[`../zig-adder`](../zig-adder)) can satisfy. The wasi-preview1 adapter
contributes the standard `wasi:cli/*`, `wasi:io/*`, `wasi:clocks/*`,
`wasi:filesystem/*`, and `wasi:random/*` imports.

## Build pipeline

```sh
# 1. core wasm with custom _start (no proc_exit).
zig build-exe -target wasm32-wasi -O ReleaseSmall \
    -fno-entry --export=_start -femit-bin=zig-calculator-cmd.core.wasm src/main.zig

# 2. embed the `app` world (--wit) and encode the component in one step
#    (wabt ≥ v3.0.0-dev.13); the bundled wasi-preview1 → preview2 adapter
#    auto-attaches for the unresolved `wasi_snapshot_preview1.*` imports.
#    A <name>.core.wasm input yields <name>.wasm.
wabt component new --world app --wit wit zig-calculator-cmd.core.wasm

# 3. compose with an adder implementation (e.g. ../zig-adder).
wabt component compose -d zig-adder.wasm \
    zig-calculator-cmd.wasm \
    -o final.component.wasm
```

Or via the repo's root `build.zig`:

```sh
zig build component-examples
```

That step produces `zig-out/component-examples/zig-calculator-cmd.component.wasm`
(unresolved `docs:adder/add@0.1.0` import) and a composed
`zig-out/component-examples/zig-calculator-cmd.composed.wasm` with the
Zig adder linked in.

## Runtime status

The composed component validates with `wabt validate` and runs in both
[Wasmtime][wasmtime] and wamr. `zig build component-examples-run` runs
this example end-to-end through `./zig-out/bin/wamr` and asserts the
expected two-line output.

[wasmtime]: https://wasmtime.dev/

## Notes

- We deliberately use a hand-rolled `_start` (instead of `pub fn main`)
  for the same reason as [`../zig-hello`](../zig-hello) — to avoid a
  preview1 `proc_exit` call that would surface as a trap until wamr's
  exit-code path is complete.
- Argument parsing is omitted; the example focuses on the import linkage.
  A future variant could parse `argv` via `std.os.wasi.args_sizes_get` /
  `args_get` to match the BCA tutorial's `1 2 add → "1 + 2 = 3"` shape.
- `wabt component compose` and `wac` produce equivalent component
  shapes for this example; wamr's component loader accepts both.
