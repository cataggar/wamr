//! WASI command component that exercises the exit-code path (issue #436).
//!
//! Writes a marker line to fd 1 via `wasi_snapshot_preview1.fd_write`, then
//! calls `wasi_snapshot_preview1.proc_exit(7)`. The numeric code reaches
//! the host via `ModuleInstance.exit_code_sink` →
//! `WasiCliAdapter.exit_code` → `RunOutcome.exit_code` →
//! `main.zig:runComponent`, which exits the host process with 7.
//!
//! Pair with `zig-hello` which exercises the normal-return path.

const std = @import("std");

export fn _start() void {
    const msg = "exiting with code 7\n";
    var nwritten: usize = 0;
    _ = std.os.wasi.fd_write(
        1,
        &.{.{ .base = msg.ptr, .len = msg.len }},
        1,
        &nwritten,
    );
    std.os.wasi.proc_exit(7);
}
