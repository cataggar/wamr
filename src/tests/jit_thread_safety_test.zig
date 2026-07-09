//! #859 — thread-safety stress test for the in-process JIT compile
//! entry points.
//!
//! Audit summary (full writeup in the PR description / commit
//! message): `src/compiler/**` has exactly one genuinely shared,
//! non-`threadlocal` mutable global — `aot_bisect.global` (the #761
//! codegen-bisection spec) — plus two more in
//! `src/component/core_backend.zig`: `aot_debug_enabled` and
//! `trap_cross_memory_enabled`. All three are diagnostic/debug-only
//! toggles that both shipped CLIs (`wamr`, `wamrc`) set exactly once,
//! at startup, from an env var, before any compile or run activity
//! begins — never touched again during the process lifetime. Every
//! other piece of per-compile state (IR module, pass pipeline,
//! codegen buffers, `codegen_cache.Cache`) is either a function-local
//! value or explicitly threaded through call parameters — nothing is
//! shared across concurrent compiles by default. `threadlocal var`
//! usage elsewhere (`verifier.zig`, `analysis.zig`, `passes.zig`) is
//! inherently per-thread-safe by construction and not a concern.
//!
//! Conclusion: **yes, two threads can call `compileCoreWasm` /
//! `precompileComponentInMemory` concurrently on distinct modules
//! without corrupting each other's output** — *provided* the three
//! diagnostic globals above are left untouched during the concurrent
//! window, which matches how both CLIs already use them (configure
//! once at startup, never reconfigure mid-flight). This test proves
//! the positive case: N threads each independently compile, load,
//! instantiate, execute, and destroy a small AOT module at the same
//! time, and every thread observes the correct, uncorrupted result
//! for its own distinct input.

const std = @import("std");
const wamr = @import("wamr");

const aot_loader_mod = wamr.aot_loader;
const aot_runtime_mod = wamr.aot_runtime;
const component_aot_compile = wamr.component_aot_compile;

const can_exec_aot = switch (@import("builtin").cpu.arch) {
    .x86_64, .aarch64 => true,
    else => false,
};

// The "add42" core wasm fixture already used elsewhere in this test
// suite (e.g. `component_aot_canonlift_test.zig`): one exported
// function `add42(x: i32) -> i32` computing `x + 42`. Small, no
// imports, deterministic — ideal for a concurrent-compile stress test
// since each thread can pick a distinct input and independently
// verify its own output.
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

const ThreadCount = 8;

const ThreadResult = struct {
    ok: bool = false,
    got: i32 = 0,
    want: i32 = 0,
    err_name: []const u8 = "",
};

fn compileLoadRunDestroy(idx: usize, result: *ThreadResult, gpa: std.mem.Allocator) void {
    // Distinct input per thread, so a cross-thread mixup (reading
    // another thread's compiled code, IR, or result) shows up as a
    // wrong `got` value rather than being masked by every thread
    // computing the same answer.
    const input: i32 = @intCast(idx * 1000);
    result.want = input + 42;

    const cwasm = component_aot_compile.compileCoreWasm(gpa, &add42_wasm, .{}) catch |err| {
        result.err_name = @errorName(err);
        return;
    };
    defer gpa.free(cwasm);

    var module = aot_loader_mod.load(cwasm, gpa) catch |err| {
        result.err_name = @errorName(err);
        return;
    };
    defer aot_loader_mod.unload(&module, gpa);

    const inst = aot_runtime_mod.instantiate(&module, gpa) catch |err| {
        result.err_name = @errorName(err);
        return;
    };
    defer aot_runtime_mod.destroy(inst);

    aot_runtime_mod.mapCodeExecutable(inst) catch |err| {
        result.err_name = @errorName(err);
        return;
    };

    const func_idx = aot_runtime_mod.findExportFunc(inst, "add42") orelse {
        result.err_name = "add42 export not found";
        return;
    };

    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;
    const results = aot_runtime_mod.callFuncScalar(
        inst,
        func_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = input }},
        &results_buf,
    ) catch |err| {
        result.err_name = @errorName(err);
        return;
    };

    result.got = switch (results[0]) {
        .i32 => |v| v,
        else => {
            result.err_name = "unexpected result type";
            return;
        },
    };
    result.ok = true;
}

test "#859: N threads JIT-compiling+running distinct calls concurrently all get correct, uncorrupted results" {
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;
    var results: [ThreadCount]ThreadResult = [_]ThreadResult{.{}} ** ThreadCount;
    var threads: [ThreadCount]std.Thread = undefined;

    for (0..ThreadCount) |i| {
        threads[i] = try std.Thread.spawn(.{}, compileLoadRunDestroy, .{ i, &results[i], gpa });
    }
    for (threads) |t| t.join();

    for (results, 0..) |r, i| {
        if (r.err_name.len > 0) {
            std.debug.print("thread {d} failed: {s}\n", .{ i, r.err_name });
        }
        try std.testing.expect(r.ok);
        try std.testing.expectEqual(r.want, r.got);
    }
}
