//! #892 follow-up coverage for the lazy-JIT spike (#862): lazy-eligible
//! leaf functions now defer their per-function IR pass pipeline as well
//! as x86_64 codegen.
//!
//! The tests below cover three things:
//!   1. `runPassesWithOptions(... .{ .lazy_skip = ... })` skips
//!      `promoteLocalsToSSA` / `lowerPhisToLocals` / preset fixpoint /
//!      `scrubUnreachableBlocks` for lazy functions, while
//!      `runFunctionPassesWithOptions` can replay that exact pipeline
//!      later on one retained function.
//!   2. A lazily-deferred function that contains non-trivial control
//!      flow and locals still executes correctly on the first call.
//!   3. The 200-leaf synthetic benchmark now compares eager compile,
//!      historical lazy codegen-only compile, and the new full-lazy
//!      "skip passes + codegen" mode.
const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");

const config = wamr.config;
const component_aot_compile = wamr.component_aot_compile;
const aot_loader_mod = wamr.aot_loader;
const aot_runtime_mod = wamr.aot_runtime;
const core_loader_mod = wamr.loader;
const frontend_mod = wamr.frontend;
const passes = wamr.passes;
const lazy_jit = wamr.lazy_jit;

const can_exec_aot = switch (builtin.cpu.arch) {
    .x86_64 => true, // #862 spike: x86_64 only, see docs/design/lazy-jit-spike.md
    else => false,
};

/// Portable monotonic-clock read in nanoseconds (mirrors
/// `jit_fast_preset_test.zig`'s `nowNs` -- this repo's `std` has no
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
        else => 0,
    };
}

// Generated via:
//   wasm-tools parse lazy_fixture.wat -o lazy_fixture.wasm
// (module
//   (func $add1 (export "add1") (param i32) (result i32)
//     local.get 0 i32.const 1 i32.add)
//   (func $phi_local (export "phi_local") (param i32) (param i32) (result i32)
//     (local i32)
//     local.get 1
//     if
//       local.get 0 i32.const 1 i32.add local.set 2
//     else
//       local.get 0 i32.const 2 i32.add local.set 2
//     end
//     local.get 2)
//   (func $mul2 (export "mul2") (param i32) (result i32)
//     local.get 0 i32.const 2 i32.mul)
//   (func $never_called (export "never_called") (param i32) (result i32)
//     local.get 0 i32.const 1 i32.sub))
const lazy_fixture_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x0c, 0x02, 0x60,
    0x01, 0x7f, 0x01, 0x7f, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f, 0x03, 0x05,
    0x04, 0x00, 0x01, 0x00, 0x00, 0x07, 0x2a, 0x04, 0x04, 0x61, 0x64, 0x64,
    0x31, 0x00, 0x00, 0x09, 0x70, 0x68, 0x69, 0x5f, 0x6c, 0x6f, 0x63, 0x61,
    0x6c, 0x00, 0x01, 0x04, 0x6d, 0x75, 0x6c, 0x32, 0x00, 0x02, 0x0c, 0x6e,
    0x65, 0x76, 0x65, 0x72, 0x5f, 0x63, 0x61, 0x6c, 0x6c, 0x65, 0x64, 0x00,
    0x03, 0x0a, 0x34, 0x04, 0x07, 0x00, 0x20, 0x00, 0x41, 0x01, 0x6a, 0x0b,
    0x1a, 0x01, 0x01, 0x7f, 0x20, 0x01, 0x04, 0x40, 0x20, 0x00, 0x41, 0x01,
    0x6a, 0x21, 0x02, 0x05, 0x20, 0x00, 0x41, 0x02, 0x6a, 0x21, 0x02, 0x0b,
    0x20, 0x02, 0x0b, 0x07, 0x00, 0x20, 0x00, 0x41, 0x02, 0x6c, 0x0b, 0x07,
    0x00, 0x20, 0x00, 0x41, 0x01, 0x6b, 0x0b, 0x00, 0x2d, 0x04, 0x6e, 0x61,
    0x6d, 0x65, 0x01, 0x26, 0x04, 0x00, 0x04, 0x61, 0x64, 0x64, 0x31, 0x01,
    0x09, 0x70, 0x68, 0x69, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x02, 0x04,
    0x6d, 0x75, 0x6c, 0x32, 0x03, 0x0c, 0x6e, 0x65, 0x76, 0x65, 0x72, 0x5f,
    0x63, 0x61, 0x6c, 0x6c, 0x65, 0x64,
};

fn deinitLazyOut(allocator: std.mem.Allocator, out: *component_aot_compile.LazyJitOut) void {
    out.ir_module.deinit();
    allocator.free(out.lazy_local_indices);
    out.* = .{};
}

fn measureCompileMinNs(
    allocator: std.mem.Allocator,
    wasm_bytes: []const u8,
    opts: component_aot_compile.PrecompileOptions,
) !struct { ns: u64, deferred: usize } {
    var best_ns = std.math.maxInt(u64);
    var deferred: usize = 0;

    var iter: usize = 0;
    while (iter < 3) : (iter += 1) {
        if (opts.lazy_jit) {
            var lazy_out: component_aot_compile.LazyJitOut = .{};
            const start_ns = nowNs();
            const cwasm = try component_aot_compile.compileCoreWasmCached(
                allocator,
                wasm_bytes,
                opts,
                .{ .lazy_jit_out = &lazy_out },
            );
            const elapsed_ns = nowNs() - start_ns;
            if (elapsed_ns < best_ns) best_ns = elapsed_ns;
            deferred = lazy_out.lazy_local_indices.len;
            allocator.free(cwasm);
            deinitLazyOut(allocator, &lazy_out);
        } else {
            const start_ns = nowNs();
            const cwasm = try component_aot_compile.compileCoreWasmCached(allocator, wasm_bytes, opts, .{});
            const elapsed_ns = nowNs() - start_ns;
            if (elapsed_ns < best_ns) best_ns = elapsed_ns;
            allocator.free(cwasm);
        }
    }

    return .{ .ns = best_ns, .deferred = deferred };
}

const PassProbe = struct {
    eager_counts: [4]usize = [_]usize{0} ** 4,

    fn callback(ctx: *anyopaque, info: passes.DumpInfo) !void {
        if (std.mem.eql(u8, info.pass_name, "inlineSmallFunctions")) return;
        const self: *PassProbe = @ptrCast(@alignCast(ctx));
        if (info.func_index < self.eager_counts.len) {
            self.eager_counts[info.func_index] += 1;
        }
    }
};

test "#892 lazy-JIT: eager pipeline skips lazy functions and helper replays it later" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;

    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    const module = try core_loader_mod.load(&lazy_fixture_wasm, arena.allocator());
    var ir_module = try frontend_mod.lowerModule(&module, gpa);
    defer ir_module.deinit();

    const lazy_skip = try lazy_jit.findLazyEligibleLeaves(&module, &ir_module, gpa);
    defer gpa.free(lazy_skip);
    try std.testing.expectEqual(@as(usize, 4), lazy_skip.len);
    for (lazy_skip) |skip| try std.testing.expect(skip);

    var eager_probe = PassProbe{};
    _ = try passes.runPassesWithOptions(
        &ir_module,
        passes.passesForPreset(.x86_64, .full),
        gpa,
        .{
            .lazy_skip = lazy_skip,
            .dump_hook = .{ .ctx = &eager_probe, .callback = PassProbe.callback },
        },
    );
    for (eager_probe.eager_counts) |count| {
        try std.testing.expectEqual(@as(usize, 0), count);
    }

    var replay_probe = PassProbe{};
    _ = try passes.runFunctionPassesWithOptions(
        &ir_module.functions.items[1],
        1,
        ir_module.import_count,
        passes.passesForPreset(.x86_64, .full),
        gpa,
        .{
            .lazy_skip = lazy_skip,
            .dump_hook = .{ .ctx = &replay_probe, .callback = PassProbe.callback },
        },
    );
    try std.testing.expect(replay_probe.eager_counts[1] > 0);
    try std.testing.expectEqual(@as(usize, 0), replay_probe.eager_counts[0]);
    try std.testing.expectEqual(@as(usize, 0), replay_probe.eager_counts[2]);
    try std.testing.expectEqual(@as(usize, 0), replay_probe.eager_counts[3]);
}

test "#892 lazy-JIT: deferred functions compile correctly on first call after skipped eager passes" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    var lazy_out: component_aot_compile.LazyJitOut = .{};
    const cwasm = try component_aot_compile.compileCoreWasmCached(
        gpa,
        &lazy_fixture_wasm,
        .{ .lazy_jit = true },
        .{ .lazy_jit_out = &lazy_out },
    );
    defer gpa.free(cwasm);
    var lazy_out_owned_by_driver = false;
    defer if (!lazy_out_owned_by_driver) deinitLazyOut(gpa, &lazy_out);

    // All 4 functions are leaf + uncalled + the module has no element
    // segments, so all 4 should be lazy-eligible.
    try std.testing.expectEqual(@as(usize, 4), lazy_out.lazy_local_indices.len);
    try std.testing.expect(!lazy_out.eager_passes_ran_for_lazy);

    var module = try aot_loader_mod.load(cwasm, gpa);
    defer aot_loader_mod.unload(&module, gpa);

    // None of the 4 functions should have real code in the emitted
    // `.cwasm` — each deferred function's text-section slice is empty,
    // proving codegen was genuinely skipped, not just reordered.
    try std.testing.expect(module.text_section == null or module.text_section.?.len == 0);

    const inst = try aot_runtime_mod.instantiate(&module, gpa);
    defer aot_runtime_mod.destroy(inst);
    try aot_runtime_mod.mapCodeExecutable(inst);

    const driver = try component_aot_compile.setupLazyJit(inst, lazy_out, gpa);
    lazy_out_owned_by_driver = true;
    defer driver.deinit();

    // Every function starts pending.
    try std.testing.expect(inst.lazy_jit.pending[0]);
    try std.testing.expect(inst.lazy_jit.pending[1]);
    try std.testing.expect(inst.lazy_jit.pending[2]);
    try std.testing.expect(inst.lazy_jit.pending[3]);

    const add1_idx = aot_runtime_mod.findExportFunc(inst, "add1") orelse return error.ExportNotFound;
    const phi_local_idx = aot_runtime_mod.findExportFunc(inst, "phi_local") orelse return error.ExportNotFound;
    const mul2_idx = aot_runtime_mod.findExportFunc(inst, "mul2") orelse return error.ExportNotFound;

    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;

    // First call to add1: compiles on demand, returns the correct result.
    const add1_results = try aot_runtime_mod.callFuncScalar(
        inst,
        add1_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 41 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 42), add1_results[0].i32);

    // First call to phi_local: compiles on demand after the deferred
    // per-function IR pipeline runs, then executes the branchy/local-heavy
    // function correctly.
    const phi_local_true = try aot_runtime_mod.callFuncScalar(
        inst,
        phi_local_idx,
        &.{ .i32, .i32 },
        &.{.i32},
        &.{ .{ .i32 = 40 }, .{ .i32 = 1 } },
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 41), phi_local_true[0].i32);

    // First call to mul2: compiles on demand, returns the correct result.
    const mul2_results = try aot_runtime_mod.callFuncScalar(
        inst,
        mul2_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 21 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 42), mul2_results[0].i32);

    // Second call to phi_local must reuse the already-compiled code and
    // still take the other branch correctly.
    const phi_local_false = try aot_runtime_mod.callFuncScalar(
        inst,
        phi_local_idx,
        &.{ .i32, .i32 },
        &.{.i32},
        &.{ .{ .i32 = 40 }, .{ .i32 = 0 } },
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 42), phi_local_false[0].i32);

    // add1, phi_local, and mul2 are no longer pending; whichever
    // local index corresponds to `never_called` (the one we never
    // invoked) must still be pending -- proving its compilation was
    // genuinely skipped, not merely done in a different order.
    var still_pending: usize = 0;
    for (inst.lazy_jit.pending) |p| {
        if (p) still_pending += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), still_pending);
    try std.testing.expect(inst.lazy_jit.pending[3]);

    // Second call to add1 must reuse the already-compiled code (not
    // recompile) -- verified indirectly: pending stays false and the
    // result is still correct.
    const add1_again = try aot_runtime_mod.callFuncScalar(
        inst,
        add1_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 99 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 100), add1_again[0].i32);
}

// 200 leaf functions (each `fN(x) = x + N`), no calls, no tables — a
// stand-in for "a module where most functions are never called".
// Generated via: wasm-tools parse lazy_bench.wat -o lazy_bench_fixture.wasm
const lazy_bench_wasm = @embedFile("lazy_bench_fixture_wasm");

test "#892 lazy-JIT: skipping passes plus codegen beats codegen-only lazy compile" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    // Take the minimum of 3 runs per mode to dampen clock noise on this
    // intentionally timing-based regression guard.
    const eager = try measureCompileMinNs(gpa, lazy_bench_wasm, .{});
    const lazy_codegen_only = try measureCompileMinNs(gpa, lazy_bench_wasm, .{
        .lazy_jit = true,
        .lazy_defer_passes = false,
    });
    const lazy_skip_passes = try measureCompileMinNs(gpa, lazy_bench_wasm, .{
        .lazy_jit = true,
        .lazy_defer_passes = true,
    });

    std.debug.print(
        "[#892] 200-leaf-fn compileCoreWasmCached (best-of-3): eager={d}us lazy_codegen_only={d}us lazy_skip_passes={d}us ({d} functions deferred)\n",
        .{
            eager.ns / 1000,
            lazy_codegen_only.ns / 1000,
            lazy_skip_passes.ns / 1000,
            lazy_skip_passes.deferred,
        },
    );

    try std.testing.expectEqual(@as(usize, 200), lazy_codegen_only.deferred);
    try std.testing.expectEqual(@as(usize, 200), lazy_skip_passes.deferred);
    // Loose regression guard (see jit_fast_preset_test.zig for the same
    // pattern/rationale): deferring more work must not regress compile
    // latency on this all-lazy synthetic benchmark.
    try std.testing.expect(lazy_skip_passes.ns <= lazy_codegen_only.ns);
    try std.testing.expect(lazy_codegen_only.ns <= eager.ns);
}
