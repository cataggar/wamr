//! Lazy-JIT regression coverage: the #862 leaf-functions-only spike, and
//! its #887 (non-leaf x86_64), #891 (JitCodeCache integration), and #894
//! (concurrent first-call thread-safety) follow-ups.
const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");

const config = wamr.config;
const component_aot_compile = wamr.component_aot_compile;
const aot_loader_mod = wamr.aot_loader;
const aot_runtime_mod = wamr.aot_runtime;

const can_exec_aot = switch (builtin.cpu.arch) {
    .x86_64 => true,
    else => false,
};

const ThreadCount = 8;

fn sleepNs(ns: u64) void {
    const sec: i64 = @intCast(ns / std.time.ns_per_s);
    const nsec: i64 = @intCast(ns % std.time.ns_per_s);
    const ts = std.posix.timespec{ .sec = sec, .nsec = nsec };
    _ = std.posix.system.nanosleep(&ts, null);
}

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

const SpinBarrier = struct {
    arrived: std.atomic.Value(u32) = .init(0),
    go: std.atomic.Value(bool) = .init(false),

    fn wait(self: *SpinBarrier, participants: u32) void {
        const seen = self.arrived.fetchAdd(1, .acq_rel) + 1;
        if (seen == participants) {
            self.go.store(true, .release);
            return;
        }

        var spins: u32 = 0;
        while (!self.go.load(.acquire)) {
            if (spins < 1024) {
                spins += 1;
                std.atomic.spinLoopHint();
            } else {
                spins = 0;
                std.Thread.yield() catch {};
            }
        }
    }
};

const LazyFixtureInstance = struct {
    gpa: std.mem.Allocator,
    cwasm: []u8,
    module: aot_loader_mod.AotModule,
    inst: *aot_runtime_mod.AotInstance,
    driver: *component_aot_compile.LazyCompileDriver,
    add1_idx: u32,
    mul2_idx: u32,

    fn deinit(self: *LazyFixtureInstance) void {
        aot_runtime_mod.destroy(self.inst);
        self.driver.deinit();
        aot_loader_mod.unload(&self.module, self.gpa);
        self.gpa.free(self.cwasm);
    }
};

fn setupLazyFixtureInstance(gpa: std.mem.Allocator) !LazyFixtureInstance {
    var lazy_out: component_aot_compile.LazyJitOut = .{};
    var lazy_out_ready = false;
    var lazy_out_owned_by_driver = false;
    errdefer if (lazy_out_ready and !lazy_out_owned_by_driver) {
        lazy_out.ir_module.deinit();
        if (lazy_out.lazy_local_indices.len > 0) gpa.free(lazy_out.lazy_local_indices);
    };

    const cwasm = try component_aot_compile.compileCoreWasmCached(
        gpa,
        &lazy_fixture_wasm,
        .{ .lazy_jit = true },
        .{ .lazy_jit_out = &lazy_out },
    );
    lazy_out_ready = true;
    errdefer gpa.free(cwasm);

    var module = try aot_loader_mod.load(cwasm, gpa);
    errdefer aot_loader_mod.unload(&module, gpa);

    const inst = try aot_runtime_mod.instantiate(&module, gpa);
    var driver: ?*component_aot_compile.LazyCompileDriver = null;
    errdefer {
        aot_runtime_mod.destroy(inst);
        if (driver) |d| d.deinit();
    }

    try aot_runtime_mod.mapCodeExecutable(inst);
    driver = try component_aot_compile.setupLazyJit(inst, lazy_out, gpa);
    lazy_out_owned_by_driver = true;

    const add1_idx = aot_runtime_mod.findExportFunc(inst, "add1") orelse return error.ExportNotFound;
    const mul2_idx = aot_runtime_mod.findExportFunc(inst, "mul2") orelse return error.ExportNotFound;

    return .{
        .gpa = gpa,
        .cwasm = cwasm,
        .module = module,
        .inst = inst,
        .driver = driver.?,
        .add1_idx = add1_idx,
        .mul2_idx = mul2_idx,
    };
}

fn countLazySlotsInState(
    inst: *const aot_runtime_mod.AotInstance,
    state: aot_runtime_mod.LazyJitState.SlotState,
) usize {
    var count: usize = 0;
    for (0..inst.lazy_jit.slot_states.len) |i| {
        if (inst.lazy_jit.slotState(i) == state) count += 1;
    }
    return count;
}

const TrackedCompileFn = struct {
    inner_ctx: *anyopaque,
    inner_fn: *const fn (ctx: *anyopaque, local_idx: u32) ?aot_runtime_mod.LazyCompiledFunc,
    tracked_local_idx: u32,
    tracked_entries: std.atomic.Value(u32) = .init(0),
    fail_first_attempt: bool = false,
    delay_ns: u64 = 0,

    fn compileFn(ctx_opaque: *anyopaque, local_idx: u32) ?aot_runtime_mod.LazyCompiledFunc {
        const self: *TrackedCompileFn = @ptrCast(@alignCast(ctx_opaque));
        if (local_idx == self.tracked_local_idx) {
            const attempt = self.tracked_entries.fetchAdd(1, .acq_rel);
            if (self.delay_ns != 0) sleepNs(self.delay_ns);
            if (self.fail_first_attempt and attempt == 0) return null;
        }
        return self.inner_fn(self.inner_ctx, local_idx);
    }
};

const LazyCallThreadCtx = struct {
    inst: *aot_runtime_mod.AotInstance,
    func_idx: u32,
    input: i32,
    barrier: *SpinBarrier,
};

const LazyCallThreadResult = struct {
    ok: bool = false,
    got: i32 = 0,
    want: i32 = 0,
    err_name: []const u8 = "",
};

fn callLazyExportThread(ctx: *const LazyCallThreadCtx, result: *LazyCallThreadResult) void {
    result.want = ctx.input + 1;
    ctx.barrier.wait(ThreadCount);

    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;
    const results = aot_runtime_mod.callFuncScalar(
        ctx.inst,
        ctx.func_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = ctx.input }},
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

// Generated via:
//   wasm-tools parse lazy_fixture.wat -o lazy_fixture.wasm
// (module
//   (func $add1 (export "add1") (param i32) (result i32)
//     local.get 0 i32.const 1 i32.add)
//   (func $mul2 (export "mul2") (param i32) (result i32)
//     local.get 0 i32.const 2 i32.mul)
//   (func $never_called (export "never_called") (param i32) (result i32)
//     local.get 0 i32.const 1 i32.sub))
const lazy_fixture_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x03, 0x04, 0x03, 0x00, 0x00, 0x00, 0x07, 0x1e,
    0x03, 0x04, 0x61, 0x64, 0x64, 0x31, 0x00, 0x00,
    0x04, 0x6d, 0x75, 0x6c, 0x32, 0x00, 0x01, 0x0c,
    0x6e, 0x65, 0x76, 0x65, 0x72, 0x5f, 0x63, 0x61,
    0x6c, 0x6c, 0x65, 0x64, 0x00, 0x02, 0x0a, 0x19,
    0x03, 0x07, 0x00, 0x20, 0x00, 0x41, 0x01, 0x6a,
    0x0b, 0x07, 0x00, 0x20, 0x00, 0x41, 0x02, 0x6c,
    0x0b, 0x07, 0x00, 0x20, 0x00, 0x41, 0x01, 0x6b,
    0x0b, 0x00, 0x22, 0x04, 0x6e, 0x61, 0x6d, 0x65,
    0x01, 0x1b, 0x03, 0x00, 0x04, 0x61, 0x64, 0x64,
    0x31, 0x01, 0x04, 0x6d, 0x75, 0x6c, 0x32, 0x02,
    0x0c, 0x6e, 0x65, 0x76, 0x65, 0x72, 0x5f, 0x63,
    0x61, 0x6c, 0x6c, 0x65, 0x64,
};
const eager_entry_local: usize = 0;
const lazy_add_local: usize = 1;
const eager_callee_local: usize = 2;
const lazy_to_eager_local: usize = 3;
const nested_callee_local: usize = 4;
const nested_tail_local: usize = 5;
const unused_leaf_local: usize = 6;
const unused_nonleaf_local: usize = 7;

const lazy_nonleaf_fixture_wasm = @embedFile("lazy_nonleaf_fixture.wasm");

fn localCodeLen(module: *const aot_loader_mod.AotModule, local_idx: usize) usize {
    const text = module.text_section orelse return 0;
    const start = module.func_offsets[local_idx];
    const end: u32 = if (local_idx + 1 < module.func_offsets.len)
        module.func_offsets[local_idx + 1]
    else
        @intCast(text.len);
    return @intCast(end - start);
}

fn callI32(inst: *aot_runtime_mod.AotInstance, func_idx: u32, arg: i32) !i32 {
    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;
    const results = try aot_runtime_mod.callFuncScalar(
        inst,
        func_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = arg }},
        &results_buf,
    );
    return results[0].i32;
}

test "#887 lazy-JIT: stable stubs and lazy-body indirect calls cover non-leaf direct-call graphs" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    var lazy_out: component_aot_compile.LazyJitOut = .{};
    const cwasm = try component_aot_compile.compileCoreWasmCached(
        gpa,
        lazy_nonleaf_fixture_wasm,
        .{ .lazy_jit = true },
        .{ .lazy_jit_out = &lazy_out },
    );
    defer gpa.free(cwasm);

    try std.testing.expectEqualSlices(
        u32,
        &.{ 1, 3, 4, 5, 6, 7 },
        lazy_out.lazy_local_indices,
    );

    var module = try aot_loader_mod.load(cwasm, gpa);
    defer aot_loader_mod.unload(&module, gpa);

    try std.testing.expect(module.text_section != null);
    for (lazy_out.lazy_local_indices) |local_idx| {
        try std.testing.expect(localCodeLen(&module, local_idx) > 0);
    }

    const inst = try aot_runtime_mod.instantiate(&module, gpa);
    try aot_runtime_mod.mapCodeExecutable(inst);

    const driver = try component_aot_compile.setupLazyJit(inst, lazy_out, gpa);
    defer {
        aot_runtime_mod.destroy(inst);
        driver.deinit();
    }

    try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.ready, inst.lazy_jit.slotState(eager_entry_local));
    try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.pending, inst.lazy_jit.slotState(lazy_add_local));
    try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.ready, inst.lazy_jit.slotState(eager_callee_local));
    try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.pending, inst.lazy_jit.slotState(lazy_to_eager_local));
    try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.pending, inst.lazy_jit.slotState(nested_callee_local));
    try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.pending, inst.lazy_jit.slotState(nested_tail_local));
    try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.pending, inst.lazy_jit.slotState(unused_leaf_local));
    try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.pending, inst.lazy_jit.slotState(unused_nonleaf_local));

    const eager_entry_idx = aot_runtime_mod.findExportFunc(inst, "eager_entry") orelse return error.ExportNotFound;
    const lazy_to_eager_idx = aot_runtime_mod.findExportFunc(inst, "lazy_to_eager") orelse return error.ExportNotFound;
    const nested_tail_idx = aot_runtime_mod.findExportFunc(inst, "nested_tail") orelse return error.ExportNotFound;

    try std.testing.expectEqual(@as(i32, 42), try callI32(inst, eager_entry_idx, 41));
    try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.ready, inst.lazy_jit.slotState(lazy_add_local));
    const lazy_add_addr = inst.lazy_jit.compiled[lazy_add_local].?.addr;
    try std.testing.expectEqual(
        @intFromPtr(lazy_add_addr),
        inst.funcptrs[@as(usize, module.import_function_count) + lazy_add_local],
    );
    try std.testing.expectEqual(@as(i32, 10), try callI32(inst, eager_entry_idx, 9));
    try std.testing.expectEqual(lazy_add_addr, inst.lazy_jit.compiled[lazy_add_local].?.addr);

    try std.testing.expectEqual(@as(i32, 42), try callI32(inst, lazy_to_eager_idx, 21));
    try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.ready, inst.lazy_jit.slotState(lazy_to_eager_local));
    try std.testing.expect(inst.lazy_jit.compiled[lazy_to_eager_local] != null);
    try std.testing.expect(inst.lazy_jit.compiled[eager_callee_local] == null);

    try std.testing.expectEqual(@as(i32, 42), try callI32(inst, nested_tail_idx, 37));
    try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.ready, inst.lazy_jit.slotState(nested_callee_local));
    try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.ready, inst.lazy_jit.slotState(nested_tail_local));
    const nested_callee_addr = inst.lazy_jit.compiled[nested_callee_local].?.addr;
    const nested_tail_addr = inst.lazy_jit.compiled[nested_tail_local].?.addr;
    try std.testing.expectEqual(@as(i32, 105), try callI32(inst, nested_tail_idx, 100));
    try std.testing.expectEqual(nested_callee_addr, inst.lazy_jit.compiled[nested_callee_local].?.addr);
    try std.testing.expectEqual(nested_tail_addr, inst.lazy_jit.compiled[nested_tail_local].?.addr);

    try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.pending, inst.lazy_jit.slotState(unused_leaf_local));
    try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.pending, inst.lazy_jit.slotState(unused_nonleaf_local));
    try std.testing.expect(inst.lazy_jit.compiled[unused_leaf_local] == null);
    try std.testing.expect(inst.lazy_jit.compiled[unused_nonleaf_local] == null);
}

test "#862 lazy-JIT spike: leaf functions are deferred and compile correctly on first call" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;
    var fixture = try setupLazyFixtureInstance(gpa);
    defer fixture.deinit();

    // None of the 3 functions should have real code in the emitted
    // `.cwasm` — each deferred function's text-section slice is empty,
    // proving codegen was genuinely skipped, not just reordered.
    for (fixture.module.func_offsets, 0..) |_, i| {
        _ = i;
    }
    try std.testing.expect(fixture.module.text_section == null or fixture.module.text_section.?.len == 0);

    // All 3 functions are leaf + uncalled + the module has no element
    // segments, so all 3 should be lazy-eligible and start pending.
    try std.testing.expectEqual(@as(usize, 3), fixture.driver.lazy_out.lazy_local_indices.len);
    try std.testing.expectEqual(@as(usize, 3), countLazySlotsInState(fixture.inst, .pending));

    const add1_local_idx = fixture.add1_idx - fixture.inst.module.import_function_count;
    const mul2_local_idx = fixture.mul2_idx - fixture.inst.module.import_function_count;

    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;

    // First call to add1: compiles on demand, returns the correct result.
    const add1_results = try aot_runtime_mod.callFuncScalar(
        fixture.inst,
        fixture.add1_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 41 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 42), add1_results[0].i32);

    // First call to mul2: compiles on demand, returns the correct result.
    const mul2_results = try aot_runtime_mod.callFuncScalar(
        fixture.inst,
        fixture.mul2_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 21 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 42), mul2_results[0].i32);

    // add1 and mul2's local indices are no longer pending; whichever
    // local index corresponds to `never_called` (the one we never
    // invoked) must still be pending -- proving its compilation was
    // genuinely skipped, not merely done in a different order.
    try std.testing.expectEqual(@as(usize, 1), countLazySlotsInState(fixture.inst, .pending));
    try std.testing.expectEqual(
        aot_runtime_mod.LazyJitState.SlotState.ready,
        fixture.inst.lazy_jit.slotState(@intCast(add1_local_idx)),
    );
    try std.testing.expectEqual(
        aot_runtime_mod.LazyJitState.SlotState.ready,
        fixture.inst.lazy_jit.slotState(@intCast(mul2_local_idx)),
    );

    // Second call to add1 must reuse the already-compiled code (not
    // recompile) -- verified indirectly: the slot stays `.ready` and
    // the result is still correct.
    const add1_again = try aot_runtime_mod.callFuncScalar(
        fixture.inst,
        fixture.add1_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 99 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 100), add1_again[0].i32);
    try std.testing.expectEqual(
        aot_runtime_mod.LazyJitState.SlotState.ready,
        fixture.inst.lazy_jit.slotState(@intCast(add1_local_idx)),
    );
}

test "#894 lazy-JIT: concurrent first calls to the same lazy export compile exactly once" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;
    var fixture = try setupLazyFixtureInstance(gpa);
    defer fixture.deinit();

    const add1_local_idx = fixture.add1_idx - fixture.inst.module.import_function_count;
    var tracked = TrackedCompileFn{
        .inner_ctx = fixture.inst.lazy_jit.compile_ctx.?,
        .inner_fn = fixture.inst.lazy_jit.compile_fn.?,
        .tracked_local_idx = add1_local_idx,
        .delay_ns = 1_000_000,
    };
    fixture.inst.lazy_jit.compile_ctx = &tracked;
    fixture.inst.lazy_jit.compile_fn = &TrackedCompileFn.compileFn;

    var barrier = SpinBarrier{};
    var contexts: [ThreadCount]LazyCallThreadCtx = undefined;
    var results: [ThreadCount]LazyCallThreadResult = [_]LazyCallThreadResult{.{}} ** ThreadCount;
    var threads: [ThreadCount]std.Thread = undefined;

    for (0..ThreadCount) |i| {
        contexts[i] = .{
            .inst = fixture.inst,
            .func_idx = fixture.add1_idx,
            .input = @intCast(i * 100),
            .barrier = &barrier,
        };
        threads[i] = try std.Thread.spawn(.{}, callLazyExportThread, .{ &contexts[i], &results[i] });
    }
    for (threads) |t| t.join();

    for (results, 0..) |r, i| {
        if (r.err_name.len > 0) {
            std.debug.print("thread {d} failed: {s}\n", .{ i, r.err_name });
        }
        try std.testing.expect(r.ok);
        try std.testing.expectEqual(r.want, r.got);
    }

    try std.testing.expectEqual(@as(u32, 1), tracked.tracked_entries.load(.acquire));
    try std.testing.expectEqual(
        aot_runtime_mod.LazyJitState.SlotState.ready,
        fixture.inst.lazy_jit.slotState(@intCast(add1_local_idx)),
    );
    try std.testing.expectEqual(@as(usize, 2), countLazySlotsInState(fixture.inst, .pending));

    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;
    const mul2_results = try aot_runtime_mod.callFuncScalar(
        fixture.inst,
        fixture.mul2_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 21 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 42), mul2_results[0].i32);
    try std.testing.expectEqual(@as(usize, 1), countLazySlotsInState(fixture.inst, .pending));
}

test "#894 lazy-JIT: failed contended compile resets the slot to pending so a waiter can retry" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;
    var fixture = try setupLazyFixtureInstance(gpa);
    defer fixture.deinit();

    const add1_local_idx = fixture.add1_idx - fixture.inst.module.import_function_count;
    var tracked = TrackedCompileFn{
        .inner_ctx = fixture.inst.lazy_jit.compile_ctx.?,
        .inner_fn = fixture.inst.lazy_jit.compile_fn.?,
        .tracked_local_idx = add1_local_idx,
        .fail_first_attempt = true,
        .delay_ns = 1_000_000,
    };
    fixture.inst.lazy_jit.compile_ctx = &tracked;
    fixture.inst.lazy_jit.compile_fn = &TrackedCompileFn.compileFn;

    var barrier = SpinBarrier{};
    var contexts: [ThreadCount]LazyCallThreadCtx = undefined;
    var results: [ThreadCount]LazyCallThreadResult = [_]LazyCallThreadResult{.{}} ** ThreadCount;
    var threads: [ThreadCount]std.Thread = undefined;

    for (0..ThreadCount) |i| {
        contexts[i] = .{
            .inst = fixture.inst,
            .func_idx = fixture.add1_idx,
            .input = @intCast(i * 10),
            .barrier = &barrier,
        };
        threads[i] = try std.Thread.spawn(.{}, callLazyExportThread, .{ &contexts[i], &results[i] });
    }
    for (threads) |t| t.join();

    var ok_count: usize = 0;
    var code_mapping_failed_count: usize = 0;
    for (results, 0..) |r, i| {
        if (r.ok) {
            ok_count += 1;
            try std.testing.expectEqual(r.want, r.got);
            continue;
        }
        if (std.mem.eql(u8, r.err_name, "CodeMappingFailed")) {
            code_mapping_failed_count += 1;
            continue;
        }
        std.debug.print("thread {d} failed unexpectedly: {s}\n", .{ i, r.err_name });
        return error.UnexpectedThreadFailure;
    }

    try std.testing.expectEqual(@as(usize, ThreadCount - 1), ok_count);
    try std.testing.expectEqual(@as(usize, 1), code_mapping_failed_count);
    try std.testing.expectEqual(@as(u32, 2), tracked.tracked_entries.load(.acquire));
    try std.testing.expectEqual(
        aot_runtime_mod.LazyJitState.SlotState.ready,
        fixture.inst.lazy_jit.slotState(@intCast(add1_local_idx)),
    );

    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;
    const add1_again = try aot_runtime_mod.callFuncScalar(
        fixture.inst,
        fixture.add1_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 99 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 100), add1_again[0].i32);
    try std.testing.expectEqual(@as(u32, 2), tracked.tracked_entries.load(.acquire));
}

test "#891 lazy-JIT: deferred mappings are tracked by JitCodeCache" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;
    const baseline_bytes = aot_runtime_mod.JitCodeCache.residentBytes();
    const baseline_count = aot_runtime_mod.JitCodeCache.mappingCount();

    {
        var fixture = try setupLazyFixtureInstance(gpa);
        defer fixture.deinit();

        const inst = fixture.inst;
        const add1_idx = fixture.add1_idx;
        const local_idx = add1_idx - inst.module.import_function_count;

        try std.testing.expectEqual(baseline_count, aot_runtime_mod.JitCodeCache.mappingCount());
        try std.testing.expectEqual(baseline_bytes, aot_runtime_mod.JitCodeCache.residentBytes());

        var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;
        const add1_results = try aot_runtime_mod.callFuncScalar(
            inst,
            add1_idx,
            &.{.i32},
            &.{.i32},
            &.{.{ .i32 = 41 }},
            &results_buf,
        );
        try std.testing.expectEqual(@as(i32, 42), add1_results[0].i32);

        const compiled = inst.lazy_jit.compiled[local_idx] orelse return error.FunctionNotFound;
        try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.ready, inst.lazy_jit.slotState(local_idx));
        try std.testing.expectEqual(baseline_count + 1, aot_runtime_mod.JitCodeCache.mappingCount());
        try std.testing.expectEqual(baseline_bytes + compiled.size, aot_runtime_mod.JitCodeCache.residentBytes());

        const add1_again = try aot_runtime_mod.callFuncScalar(
            inst,
            add1_idx,
            &.{.i32},
            &.{.i32},
            &.{.{ .i32 = 99 }},
            &results_buf,
        );
        try std.testing.expectEqual(@as(i32, 100), add1_again[0].i32);
        try std.testing.expectEqual(baseline_count + 1, aot_runtime_mod.JitCodeCache.mappingCount());
        try std.testing.expectEqual(baseline_bytes + compiled.size, aot_runtime_mod.JitCodeCache.residentBytes());
    }

    try std.testing.expectEqual(baseline_count, aot_runtime_mod.JitCodeCache.mappingCount());
    try std.testing.expectEqual(baseline_bytes, aot_runtime_mod.JitCodeCache.residentBytes());
}

test "#891 lazy-JIT: deferred mappings honor JitCodeCache budgets" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;
    const baseline_bytes = aot_runtime_mod.JitCodeCache.residentBytes();
    const baseline_count = aot_runtime_mod.JitCodeCache.mappingCount();

    const compiled_size = blk: {
        var fixture = try setupLazyFixtureInstance(gpa);
        defer fixture.deinit();

        const inst = fixture.inst;
        const add1_idx = fixture.add1_idx;
        const local_idx = add1_idx - inst.module.import_function_count;
        var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;

        const add1_results = try aot_runtime_mod.callFuncScalar(
            inst,
            add1_idx,
            &.{.i32},
            &.{.i32},
            &.{.{ .i32 = 41 }},
            &results_buf,
        );
        try std.testing.expectEqual(@as(i32, 42), add1_results[0].i32);

        const compiled = inst.lazy_jit.compiled[local_idx] orelse return error.FunctionNotFound;
        break :blk compiled.size;
    };

    try std.testing.expectEqual(baseline_count, aot_runtime_mod.JitCodeCache.mappingCount());
    try std.testing.expectEqual(baseline_bytes, aot_runtime_mod.JitCodeCache.residentBytes());

    const reject_budget = baseline_bytes + compiled_size - 1;
    try std.testing.expect(reject_budget != 0);
    aot_runtime_mod.JitCodeCache.budget_bytes = reject_budget;
    defer aot_runtime_mod.JitCodeCache.budget_bytes = 0;

    {
        var fixture = try setupLazyFixtureInstance(gpa);
        defer fixture.deinit();

        const inst = fixture.inst;
        const add1_idx = fixture.add1_idx;
        const local_idx = add1_idx - inst.module.import_function_count;
        const before_bytes = aot_runtime_mod.JitCodeCache.residentBytes();
        const before_count = aot_runtime_mod.JitCodeCache.mappingCount();
        var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;

        try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.pending, inst.lazy_jit.slotState(local_idx));
        try std.testing.expect(inst.lazy_jit.compiled[local_idx] == null);
        try std.testing.expectError(
            error.CodeBudgetExceeded,
            aot_runtime_mod.callFuncScalar(
                inst,
                add1_idx,
                &.{.i32},
                &.{.i32},
                &.{.{ .i32 = 41 }},
                &results_buf,
            ),
        );
        try std.testing.expectEqual(aot_runtime_mod.LazyJitState.SlotState.pending, inst.lazy_jit.slotState(local_idx));
        try std.testing.expect(inst.lazy_jit.compiled[local_idx] == null);
        try std.testing.expectEqual(before_count, aot_runtime_mod.JitCodeCache.mappingCount());
        try std.testing.expectEqual(before_bytes, aot_runtime_mod.JitCodeCache.residentBytes());
    }

    try std.testing.expectEqual(baseline_count, aot_runtime_mod.JitCodeCache.mappingCount());
    try std.testing.expectEqual(baseline_bytes, aot_runtime_mod.JitCodeCache.residentBytes());
}

// 200 leaf functions (each `fN(x) = x + N`), no calls, no tables — a
// stand-in for "a module where most functions are never called".
// Generated via: wasm-tools parse lazy_bench.wat -o lazy_bench_fixture.wasm
const lazy_bench_wasm = @embedFile("lazy_bench_fixture_wasm");

test "#862 lazy-JIT spike: skipping 199/200 unused leaf functions measurably reduces compile time" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    // Eager: compile every function.
    const eager_start = nowNs();
    const eager_cwasm = try component_aot_compile.compileCoreWasmCached(
        gpa,
        lazy_bench_wasm,
        .{},
        .{},
    );
    defer gpa.free(eager_cwasm);
    const eager_ns = nowNs() - eager_start;

    // Lazy: defer every eligible (leaf + uncalled) function -- with no
    // calls between any of these 200 functions and no tables, all 200
    // qualify.
    var lazy_out: component_aot_compile.LazyJitOut = .{};
    const lazy_start = nowNs();
    const lazy_cwasm = try component_aot_compile.compileCoreWasmCached(
        gpa,
        lazy_bench_wasm,
        .{ .lazy_jit = true },
        .{ .lazy_jit_out = &lazy_out },
    );
    defer gpa.free(lazy_cwasm);
    const lazy_ns = nowNs() - lazy_start;
    defer lazy_out.ir_module.deinit();
    defer gpa.free(lazy_out.lazy_local_indices);

    std.debug.print(
        "[#862] 200-leaf-fn module compileCoreWasmCached: eager={d}us lazy={d}us ({d} functions deferred)\n",
        .{ eager_ns / 1000, lazy_ns / 1000, lazy_out.lazy_local_indices.len },
    );

    try std.testing.expectEqual(@as(usize, 200), lazy_out.lazy_local_indices.len);
    // Loose regression guard (see jit_fast_preset_test.zig for the same
    // pattern/rationale): skipping codegen for 200 functions must not be
    // slower than compiling all of them.
    try std.testing.expect(lazy_ns <= eager_ns);
}
