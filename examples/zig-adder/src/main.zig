//! Pure-Zig export-only component implementing the bytecodealliance
//! `docs:adder@0.1.0` tutorial world. Exports `add: func(x: u32, y: u32) -> u32`.
//!
//! No WASI imports — the canonical-ABI core export name
//! `"docs:adder/add@0.1.0#add"` is the only public surface. The pkg/iface
//! prefix matches the WIT in `wit/world.wit` and is the lift target for
//! the `add` function in interface `docs:adder/add@0.1.0`.
//!
//! Saturating add (`+%`) avoids a signed-overflow trap on host-driven
//! fuzz-style inputs; semantically the WIT signature is u32 → u32.
//!
//! See ../README.md for the broader example layout.

export fn @"docs:adder/add@0.1.0#add"(x: u32, y: u32) u32 {
    return x +% y;
}
