//! Zig WASI command component that imports the `docs:adder` library
//! component and prints two sample sums to stdout. The shape mirrors the
//! `command` half of the bytecodealliance `component-docs` adder /
//! calculator / command tutorial (simplified to skip the calculator
//! intermediary), and is composed with `../zig-adder` (or the equivalent
//! Rust adder, in `../mixed-zig-rust-calc`) via `wasm-tools compose`.
//!
//! Build pipeline (driven by the repo's root `build.zig`):
//!   1. zig build-exe -target wasm32-wasi -O ReleaseSmall \
//!         -fno-entry --export=_start src/main.zig
//!   2. wasm-tools component embed --world app wit main.wasm \
//!         -o main.embed.wasm
//!   3. wasm-tools component new main.embed.wasm \
//!         --adapt wasi_snapshot_preview1.command.wasm \
//!         -o zig-calculator-cmd.component.wasm
//!
//! The `extern "docs:adder/add@0.1.0" fn add(...)` declaration becomes a
//! component-level import of `docs:adder/add@0.1.0::add` after the embed
//! step, which `wasm-tools compose` later wires to the Zig adder's
//! matching export. We deliberately keep a hand-rolled `_start` (instead
//! of `pub fn main`) for the same reason as the `zig-hello` example — it
//! avoids `proc_exit` so the run unwinds cleanly through the adapter.
//!
//! Argument parsing is omitted for clarity; demonstrating the import
//! linkage is the focus. A future variant could parse `argv` via
//! `std.os.wasi.args_sizes_get` / `args_get` to mirror the BCA tutorial
//! `wasmtime run final.wasm 1 2 add` pattern.

const std = @import("std");

extern "docs:adder/add@0.1.0" fn add(x: u32, y: u32) u32;

fn writeAll(bytes: []const u8) void {
    var nwritten: usize = 0;
    _ = std.os.wasi.fd_write(
        1,
        &.{.{ .base = bytes.ptr, .len = bytes.len }},
        1,
        &nwritten,
    );
}

fn writeLine(buf: []u8, comptime fmt: []const u8, args: anytype) void {
    const out = std.fmt.bufPrint(buf, fmt, args) catch return;
    writeAll(out);
}

export fn _start() void {
    var buf: [128]u8 = undefined;

    const a = add(40, 2);
    writeLine(&buf, "40 + 2 = {d}\n", .{a});

    const b = add(100, 200);
    writeLine(&buf, "100 + 200 = {d}\n", .{b});
}
