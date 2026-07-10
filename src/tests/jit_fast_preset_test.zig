//! #860 — fast/baseline compile preset for the in-process JIT path.
//!
//! Two things are verified here:
//!
//!  1. Correctness: the `.fast` preset (`passes.jit_fast_passes` /
//!     `x86_64_jit_fast_passes`) still produces a correctly-executing
//!     AOT module — skipping the heavier global-analysis passes must
//!     never change observable behavior, only compile latency /
//!     steady-state codegen quality.
//!  2. The measured effect: compiling a real, non-trivial module
//!     (CoreMark) with `.fast` is faster than the `.full` default and
//!     the resulting `.cwasm` still loads successfully. This is a
//!     coarse regression guard (`fast` must not regress to `>= full`
//!     time) rather than a strict perf gate — see the PR description
//!     for the actual measured numbers on this repo's benchmark
//!     hardware, gathered via `WAMR_AOT_PASS_TIMING=1 wamrc compile`
//!     (aggregated per-pass elapsed time) plus direct `.full` vs
//!     `.fast` timing of `compileCoreWasm` itself.

const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");

const aot_loader_mod = wamr.aot_loader;
const aot_runtime_mod = wamr.aot_runtime;
const component_aot_compile = wamr.component_aot_compile;

const can_exec_aot = switch (@import("builtin").cpu.arch) {
    .x86_64, .aarch64 => true,
    else => false,
};

/// Portable monotonic-clock read in nanoseconds, mirroring
/// `passes.passTimingNowNs` (this repo's `std` version has no
/// `std.time.Timer`).
fn nowNs() u64 {
    return switch (comptime builtin.os.tag) {
        .linux => blk: {
            const linux = std.os.linux;
            var ts: linux.timespec = undefined;
            const rc = linux.clock_gettime(.MONOTONIC, &ts);
            if (rc != 0) break :blk 0;
            break :blk @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
        .macos, .ios, .tvos, .watchos, .visionos => blk: {
            var ts: std.c.timespec = undefined;
            if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) break :blk 0;
            break :blk @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
        .windows => blk: {
            const ntdll = std.os.windows.ntdll;
            var counter: std.os.windows.LARGE_INTEGER = undefined;
            var freq: std.os.windows.LARGE_INTEGER = undefined;
            _ = ntdll.RtlQueryPerformanceCounter(&counter);
            _ = ntdll.RtlQueryPerformanceFrequency(&freq);
            const ticks: u128 = @intCast(counter);
            const hz: u128 = @intCast(freq);
            break :blk @as(u64, @truncate(ticks * std.time.ns_per_s / hz));
        },
        else => 0,
    };
}

// Same "add42" fixture used by `jit_thread_safety_test.zig`: one
// exported function `add42(x: i32) -> i32` computing `x + 42`.
const add42_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x05, 0x03, 0x01, 0x00, 0x01,
    0x07, 0x12, 0x02,
    0x06, 'm', 'e', 'm', 'o', 'r', 'y', 0x02, 0x00,
    0x05, 'a', 'd', 'd', '4', '2', 0x00, 0x00,
    0x0a, 0x09, 0x01, 0x07, 0x00,
    0x20, 0x00,
    0x41, 0x2a,
    0x6a,
    0x0b,
};

// CoreMark's WASI core module: 73 functions, loops, and branches —
// unlike `add42_wasm` this actually exercises the passes the `.fast`
// preset skips (loop-invariant hoisting, GVN, dominator-based
// redundant-load forwarding), so it's the fixture used for the
// compile-time comparison below.
const coremark_wasm = @embedFile("coremark_wasm");

test "#860: .fast preset produces a correctly-executing module" {
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;
    const cwasm = try component_aot_compile.compileCoreWasm(gpa, &add42_wasm, .{
        .pass_preset = .fast,
    });
    defer gpa.free(cwasm);

    var module = try aot_loader_mod.load(cwasm, gpa);
    defer aot_loader_mod.unload(&module, gpa);

    const inst = try aot_runtime_mod.instantiate(&module, gpa);
    defer aot_runtime_mod.destroy(inst);
    try aot_runtime_mod.mapCodeExecutable(inst);

    const func_idx = aot_runtime_mod.findExportFunc(inst, "add42") orelse return error.ExportNotFound;
    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;
    const results = try aot_runtime_mod.callFuncScalar(
        inst,
        func_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 100 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 142), results[0].i32);
}

test "#860: .fast preset compiles CoreMark faster than .full and still loads" {
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    const full_start = nowNs();
    const full_cwasm = try component_aot_compile.compileCoreWasm(gpa, coremark_wasm, .{
        .pass_preset = .full,
    });
    defer gpa.free(full_cwasm);
    const full_ns = nowNs() - full_start;

    const fast_start = nowNs();
    const fast_cwasm = try component_aot_compile.compileCoreWasm(gpa, coremark_wasm, .{
        .pass_preset = .fast,
    });
    defer gpa.free(fast_cwasm);
    const fast_ns = nowNs() - fast_start;

    std.debug.print(
        "[#860] CoreMark compileCoreWasm: full={d}us ({d} bytes) fast={d}us ({d} bytes)\n",
        .{ full_ns / 1000, full_cwasm.len, fast_ns / 1000, fast_cwasm.len },
    );

    // Both presets must produce a module that still loads (correctness);
    // the JIT default (`.fast`) must not be slower than `.full` — this
    // is a loose regression guard, not a strict perf gate, since CI
    // hardware timing has noise, but `.fast` doing strictly less work
    // than `.full` should never lose on any host.
    var full_module = try aot_loader_mod.load(full_cwasm, gpa);
    defer aot_loader_mod.unload(&full_module, gpa);
    var fast_module = try aot_loader_mod.load(fast_cwasm, gpa);
    defer aot_loader_mod.unload(&fast_module, gpa);

    try std.testing.expect(fast_ns <= full_ns);
}
