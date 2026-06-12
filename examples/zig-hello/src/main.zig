//! Minimal native-`wasi:cli` command component, written in Zig.
//!
//! The smallest end-to-end Zig component example shipped with wamr: it
//! writes a greeting to stdout and returns an explicit exit code. Unlike
//! a preview1 `_start` command (which reaches stdout through the
//! wabt-bundled wasi-preview1 → preview2 adapter), this guest speaks
//! `wasi:cli` directly — it exports `wasi:cli/run@0.2.6` and writes
//! through `wasi:cli/stdout` + `wasi:io/streams`, then reports its exit
//! code via `wasi:cli/exit.exit-with-code`.
//!
//! The canonical-ABI plumbing (host imports, the `wasi:cli/run` export
//! wiring, the exit-code path) lives in the shared `wasi_cli` helper
//! module (`@import("wasi_cli")`; source at `src/guest/wasi_cli.zig`).
//! This file is just the entry point.
//!
//! See ../README.md for the build pipeline and runtime notes.

const cli = @import("wasi_cli");

comptime {
    cli.exportRun(run);
}

fn run() u8 {
    cli.print("hello from zig component\n");
    return 0; // process exit code (0 = success)
}
