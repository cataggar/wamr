//! CoreMark benchmark runner for the Zig AOT backend.
//!
//! Loads a CoreMark `.wasm` (compiled against `wasi_snapshot_preview1`),
//! AOT-compiles it through the same pipeline used by `spec-test-runner`
//! and the differential tests, and invokes `_start`.
//!
//! Provides a `coremark-aot` build step so CoreMark gates the AArch64
//! (and x86-64) Zig AOT backend directly — no dependency on the C
//! `iwasm`/`wamrc` binaries.
//!
//! Usage:
//!     zig build coremark-aot
//!     # or, after `zig build`:
//!     ./zig-out/bin/coremark-aot-runner path/to/coremark_wasi_nofp.wasm
//!
//! The harness routes WASI calls (`fd_write`, `clock_time_get`,
//! `proc_exit`, etc.) through `src/runtime/aot/host_bridge.zig`. A
//! successful run therefore exercises the full compile → map-executable
//! → invoke loop for every CoreMark kernel (list processing, matrix,
//! state-machine, CRC) over ~tens of thousands of iterations.

const std = @import("std");
const builtin = @import("builtin");
const aot_harness = @import("aot_harness.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(init.arena.allocator());

    if (args.len < 2) {
        std.debug.print(
            \\usage: coremark-aot-runner <path-to-coremark.wasm>
            \\
            \\Runs a CoreMark-style WASI wasm module through the Zig AOT
            \\backend. Import resolution is handled automatically for
            \\`wasi_snapshot_preview1`.
            \\
        , .{});
        std.process.exit(2);
    }
    const wasm_path = args[1];

    if (!aot_harness.can_exec_aot) {
        std.debug.print(
            "coremark-aot-runner: AOT execution not supported on this target ({s}); skipping.\n",
            .{@tagName(builtin.cpu.arch)},
        );
        return;
    }

    const wasm_bytes = std.Io.Dir.cwd().readFileAlloc(io, wasm_path, allocator, @enumFromInt(128 * 1024 * 1024)) catch |err| {
        std.debug.print("coremark-aot-runner: failed to read {s}: {s}\n", .{ wasm_path, @errorName(err) });
        std.process.exit(2);
    };
    defer allocator.free(wasm_bytes);

    std.debug.print("============> run {s} (zig AOT)\n", .{wasm_path});

    // `invoke_start` is best-effort: it only triggers if the wasm has a
    // start *section*. WASI binaries (like CoreMark) instead expose
    // `_start` as an **export** that the runtime is expected to call.
    // Instantiate without auto-invoking the start section, then find
    // and call the `_start` export explicitly.
    const h = aot_harness.Harness.initWithOptions(
        allocator,
        wasm_bytes,
        null,
        .{ .invoke_start = true },
    ) catch |err| {
        std.debug.print("coremark-aot-runner: harness init failed: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
    defer h.deinit();

    if (h.findFuncExport("_start")) |start_idx| {
        const aot_runtime = @import("wamr").aot_runtime;
        var results: [1]aot_runtime.ScalarResult = undefined;
        const ft = h.getFuncType(start_idx) orelse {
            std.debug.print("coremark-aot-runner: _start has no type info\n", .{});
            std.process.exit(1);
        };
        _ = aot_runtime.callFuncScalar(
            h.inst,
            start_idx,
            ft.params,
            &.{},
            &.{},
            &results,
        ) catch |err| {
            // `proc_exit` in CoreMark terminates via a trap in the
            // interpreter but is a no-op in the AOT bridge; any other
            // error is a real failure.
            std.debug.print("coremark-aot-runner: _start returned error: {s}\n", .{@errorName(err)});
            std.process.exit(1);
        };
    } else {
        std.debug.print("coremark-aot-runner: no `_start` export found\n", .{});
        std.process.exit(1);
    }

    std.debug.print("============> {s} completed\n", .{wasm_path});
}
