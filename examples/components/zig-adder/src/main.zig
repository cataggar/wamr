//! Pure-Zig export-only component implementing the bytecodealliance
//! `docs:adder@0.1.0` tutorial world. Exports `add: func(x: u32, y: u32) -> u32`.
//!
//! Build pipeline:
//!   1. `zig build-exe -target wasm32-freestanding -fno-entry \
//!         --export="docs:adder/add@0.1.0#add" -O ReleaseSmall src/main.zig`
//!   2. `wasm-tools component embed --world adder wit main.wasm \
//!         -o main.embed.wasm`
//!   3. `wasm-tools component new main.embed.wasm \
//!         -o zig-adder.component.wasm`
//!
//! No WASI imports — the canonical-ABI core export name
//! `"docs:adder/add@0.1.0#add"` is the only public surface. The pkg/iface
//! prefix matches the WIT in `wit/world.wit` and is what
//! `wasm-tools component embed` recognises as the lift target for the
//! `add` function in interface `docs:adder/add@0.1.0`.
//!
//! Saturating add (`+%`) avoids a signed-overflow trap on host-driven
//! fuzz-style inputs; semantically the WIT signature is u32 → u32.
//!
//! See ../README.md for the broader example layout.

export fn @"docs:adder/add@0.1.0#add"(x: u32, y: u32) u32 {
    return x +% y;
}
