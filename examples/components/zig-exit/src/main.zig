//! WASI command component that exercises the exit-code path (issue #436).
//!
//! Writes a marker line to fd 1 via `wasi_snapshot_preview1.fd_write`, then
//! calls `wasi_snapshot_preview1.proc_exit(7)`. Both imports are bound by
//! the wabt-bundled wasi-preview1 → preview2 adapter (auto-attached by
//! `wabt component new`); the adapter's `proc_exit` body rewrites the
//! call into `wasi:cli/exit.exit-with-code(7)`, which lands on
//! `WasiCliAdapter.cliExitWithCode` → `adapter.exit_code` →
//! `RunOutcome.exit_code` → `main.zig:runComponent`, which exits the
//! host process with 7.
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
