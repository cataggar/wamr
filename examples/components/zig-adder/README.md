# zig-adder

A pure-Zig export-only component implementing the bytecodealliance
[component-docs adder tutorial][bca-tutorial] world `docs:adder@0.1.0`.

[bca-tutorial]: https://github.com/bytecodealliance/component-docs/tree/main/component-model/examples/tutorial

```wit
package docs:adder@0.1.0;

interface add {
    add: func(x: u32, y: u32) -> u32;
}

world adder {
    export add;
}
```

The Zig source is a single function whose name uses the canonical-ABI
mangled form `"docs:adder/add@0.1.0#add"`, which `wasm-tools component
embed --world adder` recognises as the lift target for the `add` function
in the `docs:adder/add@0.1.0` interface. The component has no imports.

## Build pipeline

```sh
zig build-exe -target wasm32-freestanding -O ReleaseSmall \
    -fno-entry --export="docs:adder/add@0.1.0#add" \
    src/main.zig

wasm-tools component embed --world adder wit main.wasm -o main.embed.wasm
wasm-tools component new main.embed.wasm -o zig-adder.component.wasm
```

Or via the repo's root `build.zig`:

```sh
zig build component-examples
```

## Run

This is a library component. It has no `wasi:cli/run` export, so it
cannot be invoked directly by the wamr CLI:

```console
$ ./zig-out/bin/wamr zig-out/component-examples/zig-adder.component.wasm
Error: component does not expose a top-level `run` export. ...
```

It is meant to be **composed** with a command component that imports
`docs:adder/add@0.1.0` — see [`../zig-calculator-cmd`](../zig-calculator-cmd)
and [`../mixed-zig-rust-calc`](../mixed-zig-rust-calc).

## Notes

- The function uses Zig's wrapping addition (`+%`) so out-of-range hosts
  inputs don't produce signed-overflow traps. The WIT signature is
  `u32 → u32` and unsigned wrap is the correct semantic match.
- Built `wasm32-freestanding` (not `wasm32-wasi`) and stripped to ~400
  bytes after `component new`.
