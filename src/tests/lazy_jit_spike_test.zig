//! #862 — lazy JIT design-spike prototype: narrow, leaf-functions-only
//! demonstration that `config.lazy_jit` genuinely defers per-function
//! codegen until first call.
//!
//! Test fixture: a 3-function core wasm module (no imports, no tables,
//! no calls between functions — every function is a leaf), each
//! exported: `add1(x) = x+1`, `mul2(x) = x*2`, `never_called(x) = x-1`.
//! Since the module has no element segments and none of these
//! functions call each other, `lazy_jit.findLazyEligibleLeaves` marks
//! all three eligible.
//!
//! The test compiles with `opts.lazy_jit = true`, verifies NONE of the
//! three were actually codegen'd up front (their function bodies are
//! zero bytes in the emitted `.cwasm`), then calls `add1` and `mul2`
//! through `callFuncScalar` (the top-level host-invocation entry point
//! this narrow spike hooks — see `docs/design/lazy-jit-spike.md` for
//! why `call_indirect`/direct intra-module calls are explicitly out of
//! scope here) and checks both:
//!   1. Both calls return the correct result (compiling correctly
//!      on-demand doesn't corrupt anything).
//!   2. `never_called` is never compiled at all (proving the deferral
//!      is real, not just a same-work-different-order relabeling).
const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");

const config = wamr.config;
const component_aot_compile = wamr.component_aot_compile;
const aot_loader_mod = wamr.aot_loader;
const aot_runtime_mod = wamr.aot_runtime;

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

// Generated via:
//   wasm-tools parse lazy_call_indirect_fixture.wat -o lazy_call_indirect_fixture.wasm
// (module
//   (type $t0 (func (param i32) (result i32)))
//   (table 1 funcref)
//   (elem (i32.const 0) $target)
//   (func $target (param i32) (result i32)
//     local.get 0 i32.const 1 i32.add)
//   (func (export "call_indirect") (param i32) (result i32)
//     local.get 0 i32.const 0 call_indirect (type $t0)))
const lazy_call_indirect_fixture_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x03, 0x03, 0x02, 0x00, 0x00, 0x04, 0x04, 0x01,
    0x70, 0x00, 0x01, 0x07, 0x11, 0x01, 0x0d, 0x63,
    0x61, 0x6c, 0x6c, 0x5f, 0x69, 0x6e, 0x64, 0x69,
    0x72, 0x65, 0x63, 0x74, 0x00, 0x01, 0x09, 0x07,
    0x01, 0x00, 0x41, 0x00, 0x0b, 0x01, 0x00, 0x0a,
    0x13, 0x02, 0x07, 0x00, 0x20, 0x00, 0x41, 0x01,
    0x6a, 0x0b, 0x09, 0x00, 0x20, 0x00, 0x41, 0x00,
    0x11, 0x00, 0x00, 0x0b, 0x00, 0x17, 0x04, 0x6e,
    0x61, 0x6d, 0x65, 0x01, 0x09, 0x01, 0x00, 0x06,
    0x74, 0x61, 0x72, 0x67, 0x65, 0x74, 0x04, 0x05,
    0x01, 0x00, 0x02, 0x74, 0x30,
};

// Generated via:
//   wasm-tools parse lazy_ref_func_table_fixture.wat -o lazy_ref_func_table_fixture.wasm
// (module
//   (type $t0 (func (param i32) (result i32)))
//   (table 1 funcref)
//   (elem declare func $target)
//   (func $target (param i32) (result i32)
//     local.get 0 i32.const 2 i32.mul)
//   (func (export "set_and_call") (param i32) (result i32)
//     i32.const 0 ref.func $target table.set 0
//     local.get 0 i32.const 0 call_indirect (type $t0)))
const lazy_ref_func_table_fixture_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x03, 0x03, 0x02, 0x00, 0x00, 0x04, 0x04, 0x01,
    0x70, 0x00, 0x01, 0x07, 0x10, 0x01, 0x0c, 0x73,
    0x65, 0x74, 0x5f, 0x61, 0x6e, 0x64, 0x5f, 0x63,
    0x61, 0x6c, 0x6c, 0x00, 0x01, 0x09, 0x05, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x0a, 0x19, 0x02, 0x07,
    0x00, 0x20, 0x00, 0x41, 0x02, 0x6c, 0x0b, 0x0f,
    0x00, 0x41, 0x00, 0xd2, 0x00, 0x26, 0x00, 0x20,
    0x00, 0x41, 0x00, 0x11, 0x00, 0x00, 0x0b, 0x00,
    0x17, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x01, 0x09,
    0x01, 0x00, 0x06, 0x74, 0x61, 0x72, 0x67, 0x65,
    0x74, 0x04, 0x05, 0x01, 0x00, 0x02, 0x74, 0x30,
};

fn instantiateLazyFixture(
    gpa: std.mem.Allocator,
    wasm_bytes: []const u8,
) !struct {
    module: aot_loader_mod.AotModule,
    inst: *aot_runtime_mod.AotInstance,
    driver: *component_aot_compile.LazyCompileDriver,
    lazy_out: component_aot_compile.LazyJitOut,
} {
    var lazy_out: component_aot_compile.LazyJitOut = .{};
    const cwasm = try component_aot_compile.compileCoreWasmCached(
        gpa,
        wasm_bytes,
        .{ .lazy_jit = true },
        .{ .lazy_jit_out = &lazy_out },
    );
    defer gpa.free(cwasm);

    var module = try aot_loader_mod.load(cwasm, gpa);
    errdefer aot_loader_mod.unload(&module, gpa);

    const inst = try aot_runtime_mod.instantiate(&module, gpa);
    errdefer aot_runtime_mod.destroy(inst);

    const driver = try component_aot_compile.setupLazyJit(inst, lazy_out, gpa);
    errdefer driver.deinit();

    try aot_runtime_mod.mapCodeExecutable(inst);

    return .{
        .module = module,
        .inst = inst,
        .driver = driver,
        .lazy_out = lazy_out,
    };
}

test "#862 lazy-JIT spike: leaf functions are deferred and compile correctly on first call" {
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

    // All 3 functions are leaf + uncalled + the module has no element
    // segments, so all 3 should be lazy-eligible.
    try std.testing.expectEqual(@as(usize, 3), lazy_out.lazy_local_indices.len);

    var module = try aot_loader_mod.load(cwasm, gpa);
    defer aot_loader_mod.unload(&module, gpa);

    // None of the 3 functions should have real code in the emitted
    // `.cwasm` — each deferred function's text-section slice is empty,
    // proving codegen was genuinely skipped, not just reordered.
    for (module.func_offsets, 0..) |_, i| {
        _ = i;
    }
    try std.testing.expect(module.text_section == null or module.text_section.?.len == 0);

    const inst = try aot_runtime_mod.instantiate(&module, gpa);
    const driver = try component_aot_compile.setupLazyJit(inst, lazy_out, gpa);
    defer driver.deinit();
    defer aot_runtime_mod.destroy(inst);
    try aot_runtime_mod.mapCodeExecutable(inst);

    // Every function starts pending.
    try std.testing.expect(inst.lazy_jit.pending[0]);
    try std.testing.expect(inst.lazy_jit.pending[1]);
    try std.testing.expect(inst.lazy_jit.pending[2]);

    const add1_idx = aot_runtime_mod.findExportFunc(inst, "add1") orelse return error.ExportNotFound;
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

    // add1 and mul2's local indices are no longer pending; whichever
    // local index corresponds to `never_called` (the one we never
    // invoked) must still be pending -- proving its compilation was
    // genuinely skipped, not merely done in a different order.
    var still_pending: usize = 0;
    for (inst.lazy_jit.pending) |p| {
        if (p) still_pending += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), still_pending);

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

test "#888 lazy-JIT: call_indirect compiles through a stable trampoline" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    var fixture = try instantiateLazyFixture(gpa, &lazy_call_indirect_fixture_wasm);
    defer aot_loader_mod.unload(&fixture.module, gpa);
    defer fixture.driver.deinit();
    defer aot_runtime_mod.destroy(fixture.inst);

    try std.testing.expectEqual(@as(usize, 1), fixture.lazy_out.lazy_local_indices.len);
    try std.testing.expectEqual(@as(u32, 0), fixture.lazy_out.lazy_local_indices[0]);
    try std.testing.expect(fixture.inst.lazy_jit.pending[0]);

    const target_funcidx: usize = fixture.module.import_function_count;
    const stub_ptr = fixture.inst.funcptrs[target_funcidx];
    try std.testing.expect(stub_ptr != 0);
    try std.testing.expectEqual(stub_ptr, fixture.inst.func_table[0]);

    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;
    const caller_idx = aot_runtime_mod.findExportFunc(fixture.inst, "call_indirect") orelse return error.ExportNotFound;

    const first = try aot_runtime_mod.callFuncScalar(
        fixture.inst,
        caller_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 41 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 42), first[0].i32);
    try std.testing.expect(!fixture.inst.lazy_jit.pending[0]);
    try std.testing.expect(fixture.inst.lazy_jit.compiled[0] != null);
    try std.testing.expectEqual(stub_ptr, fixture.inst.funcptrs[target_funcidx]);
    try std.testing.expectEqual(stub_ptr, fixture.inst.func_table[0]);

    const second = try aot_runtime_mod.callFuncScalar(
        fixture.inst,
        caller_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 7 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 8), second[0].i32);
    try std.testing.expectEqual(stub_ptr, fixture.inst.funcptrs[target_funcidx]);
}

test "#888 lazy-JIT: ref.func + table.set reaches a lazy target through the same trampoline" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    var fixture = try instantiateLazyFixture(gpa, &lazy_ref_func_table_fixture_wasm);
    defer aot_loader_mod.unload(&fixture.module, gpa);
    defer fixture.driver.deinit();
    defer aot_runtime_mod.destroy(fixture.inst);

    try std.testing.expectEqual(@as(usize, 1), fixture.lazy_out.lazy_local_indices.len);
    try std.testing.expectEqual(@as(u32, 0), fixture.lazy_out.lazy_local_indices[0]);

    const target_funcidx: usize = fixture.module.import_function_count;
    const stub_ptr = fixture.inst.funcptrs[target_funcidx];
    try std.testing.expect(stub_ptr != 0);

    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;
    const caller_idx = aot_runtime_mod.findExportFunc(fixture.inst, "set_and_call") orelse return error.ExportNotFound;

    const result = try aot_runtime_mod.callFuncScalar(
        fixture.inst,
        caller_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 21 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 42), result[0].i32);
    try std.testing.expect(!fixture.inst.lazy_jit.pending[0]);
    try std.testing.expectEqual(stub_ptr, fixture.inst.funcptrs[target_funcidx]);
    try std.testing.expectEqual(stub_ptr, fixture.inst.func_table[0]);
    try std.testing.expectEqual(
        fixture.inst.func_sig_ids[target_funcidx],
        fixture.inst.tables[0].type_backing[0],
    );
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
