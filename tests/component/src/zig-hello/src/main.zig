//! Minimal WASI command component, written in Zig.
//!
//! This is the smallest end-to-end Zig component example shipped with
//! wamr: a hand-rolled `_start` writes a greeting to fd 1 (stdout) via
//! `wasi_snapshot_preview1.fd_write` and returns. Building with
//! `-fno-entry --export=_start` overrides Zig's default `std.start`
//! prologue, so the component never calls `proc_exit`. Many component
//! runtimes (including wamr today) surface a preview1 `proc_exit` call
//! as a trap rather than translating it through the wasi-preview1
//! adapter back into `wasi:cli/exit.exit-with-code`, so a normal return
//! is the most portable shape for a "hello world" example.
//!
//! See ../README.md for the build pipeline and runtime notes.

const std = @import("std");

export fn _start() void {
    const msg = "hello from zig component\n";
    var nwritten: usize = 0;
    _ = std.os.wasi.fd_write(
        1,
        &.{.{ .base = msg.ptr, .len = msg.len }},
        1,
        &nwritten,
    );
}
