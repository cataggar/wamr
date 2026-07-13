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

// #879 M4.7 (phase 1) fixtures -- each proves `lazy_jit.
// rewriteLocalCallsToIndirect` produces functionally-identical results
// to the normal, direct-`.call` codegen path when routed through the
// real compile -> emit -> instantiate -> execute pipeline via
// `PrecompileOptions.force_indirect_calls`.

// Generated via: wasm-tools parse m47_caller_callee.wat -o ...wasm
// (module
//   (func $callee (param i32) (result i32)
//     local.get 0 i32.const 2 i32.mul)
//   (func $caller (export "caller") (param i32) (result i32)
//     local.get 0 call $callee i32.const 1 i32.add))
const m47_caller_callee_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x03, 0x03, 0x02, 0x00, 0x00, 0x07, 0x0a, 0x01,
    0x06, 0x63, 0x61, 0x6c, 0x6c, 0x65, 0x72, 0x00,
    0x01, 0x0a, 0x13, 0x02, 0x07, 0x00, 0x20, 0x00,
    0x41, 0x02, 0x6c, 0x0b, 0x09, 0x00, 0x20, 0x00,
    0x10, 0x00, 0x41, 0x01, 0x6a, 0x0b, 0x00, 0x18,
    0x04, 0x6e, 0x61, 0x6d, 0x65, 0x01, 0x11, 0x02,
    0x00, 0x06, 0x63, 0x61, 0x6c, 0x6c, 0x65, 0x65,
    0x01, 0x06, 0x63, 0x61, 0x6c, 0x6c, 0x65, 0x72,
};

// Generated via: wasm-tools parse m47_tail_call.wat -o ...wasm
// (module
//   (func $callee (param i32) (result i32)
//     local.get 0 i32.const 3 i32.add)
//   (func $caller (export "caller") (param i32) (result i32)
//     local.get 0 return_call $callee))
const m47_tail_call_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x03, 0x03, 0x02, 0x00, 0x00, 0x07, 0x0a, 0x01,
    0x06, 0x63, 0x61, 0x6c, 0x6c, 0x65, 0x72, 0x00,
    0x01, 0x0a, 0x10, 0x02, 0x07, 0x00, 0x20, 0x00,
    0x41, 0x03, 0x6a, 0x0b, 0x06, 0x00, 0x20, 0x00,
    0x12, 0x00, 0x0b, 0x00, 0x18, 0x04, 0x6e, 0x61,
    0x6d, 0x65, 0x01, 0x11, 0x02, 0x00, 0x06, 0x63,
    0x61, 0x6c, 0x6c, 0x65, 0x65, 0x01, 0x06, 0x63,
    0x61, 0x6c, 0x6c, 0x65, 0x72,
};

// Generated via: wasm-tools parse m47_recursive.wat -o ...wasm
// (module
//   (func $sumto (export "sumto") (param i32) (result i32)
//     local.get 0 i32.const 0 i32.le_s
//     if (result i32)
//       i32.const 0
//     else
//       local.get 0 local.get 0 i32.const 1 i32.sub call $sumto i32.add
//     end))
const m47_recursive_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00, 0x07, 0x09, 0x01, 0x05,
    0x73, 0x75, 0x6d, 0x74, 0x6f, 0x00, 0x00, 0x0a,
    0x19, 0x01, 0x17, 0x00, 0x20, 0x00, 0x41, 0x00,
    0x4c, 0x04, 0x7f, 0x41, 0x00, 0x05, 0x20, 0x00,
    0x20, 0x00, 0x41, 0x01, 0x6b, 0x10, 0x00, 0x6a,
    0x0b, 0x0b, 0x00, 0x0f, 0x04, 0x6e, 0x61, 0x6d,
    0x65, 0x01, 0x08, 0x01, 0x00, 0x05, 0x73, 0x75,
    0x6d, 0x74, 0x6f,
};

/// Compiles `wasm_bytes` (with `force_indirect_calls` on/off per
/// `opts`), instantiates, calls the single exported i32(i32) function
/// named `export_name` with `arg`, and returns the i32 result.
fn compileAndCallI32I32(
    allocator: std.mem.Allocator,
    wasm_bytes: []const u8,
    opts: component_aot_compile.PrecompileOptions,
    export_name: []const u8,
    arg: i32,
) !i32 {
    const cwasm = try component_aot_compile.compileCoreWasm(allocator, wasm_bytes, opts);
    defer allocator.free(cwasm);

    var module = try aot_loader_mod.load(cwasm, allocator);
    defer aot_loader_mod.unload(&module, allocator);

    const inst = try aot_runtime_mod.instantiate(&module, allocator);
    defer aot_runtime_mod.destroy(inst);
    try aot_runtime_mod.mapCodeExecutable(inst);

    const fn_idx = aot_runtime_mod.findExportFunc(inst, export_name) orelse return error.ExportNotFound;
    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;
    const results = try aot_runtime_mod.callFuncScalar(
        inst,
        fn_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = arg }},
        &results_buf,
    );
    return results[0].i32;
}

test "#879 M4.7 phase 1: rewriting a direct call to ref_func+call_ref preserves behaviour" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    const normal = try compileAndCallI32I32(gpa, &m47_caller_callee_wasm, .{}, "caller", 5);
    const rewritten = try compileAndCallI32I32(gpa, &m47_caller_callee_wasm, .{ .force_indirect_calls = true }, "caller", 5);

    try std.testing.expectEqual(@as(i32, 11), normal); // 5*2 + 1
    try std.testing.expectEqual(normal, rewritten);
}

test "#879 M4.7 phase 1: rewriting a tail call to ref_func+call_ref (real-tail codegen) preserves behaviour" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    const normal = try compileAndCallI32I32(gpa, &m47_tail_call_wasm, .{}, "caller", 10);
    const rewritten = try compileAndCallI32I32(gpa, &m47_tail_call_wasm, .{ .force_indirect_calls = true }, "caller", 10);

    try std.testing.expectEqual(@as(i32, 13), normal); // 10 + 3
    try std.testing.expectEqual(normal, rewritten);
}

test "#879 M4.7 phase 1: rewriting a self-recursive call to ref_func+call_ref preserves behaviour" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    const normal = try compileAndCallI32I32(gpa, &m47_recursive_wasm, .{}, "sumto", 5);
    const rewritten = try compileAndCallI32I32(gpa, &m47_recursive_wasm, .{ .force_indirect_calls = true }, "sumto", 5);

    try std.testing.expectEqual(@as(i32, 15), normal); // 5+4+3+2+1+0
    try std.testing.expectEqual(normal, rewritten);
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
    defer aot_runtime_mod.destroy(inst);

    // #879 M4.6 phase 2: setupLazyJit must run BEFORE mapCodeExecutable
    // so any table/ref.func-reachable deferred function's trampoline
    // address is ready in time to be installed into func_addrs -- see
    // setupLazyJit's doc comment. This fixture has none (no element
    // segments, nothing calls ref.func), so the ordering doesn't
    // change this particular test's outcome, but keeping every call
    // site consistent with the documented contract avoids this being
    // the one example that silently still works by accident.
    const driver = try component_aot_compile.setupLazyJit(inst, lazy_out, gpa);
    defer driver.deinit();
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
    defer gpa.free(lazy_out.needs_trampoline_indices);

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

// #879 M4.6 phase 2: a leaf, never-directly-called function placed in
// a table by an active element segment, called only through
// call_indirect from a second (eagerly-compiled, non-leaf) function.
// Generated via: wasm-tools parse lazy_call_indirect.wat -o lazy_call_indirect_fixture.wasm
//   (module
//     (type $unary (func (param i32) (result i32)))
//     (table 1 1 funcref)
//     (elem (i32.const 0) func $add1)
//     (func $add1 (param i32) (result i32)
//       local.get 0 i32.const 1 i32.add)
//     (func $caller (export "caller") (param i32 i32) (result i32)
//       local.get 1 local.get 0 call_indirect (type $unary)))
const lazy_call_indirect_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x0c, 0x02, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f, 0x03, 0x03,
    0x02, 0x00, 0x01, 0x04, 0x05, 0x01, 0x70, 0x01,
    0x01, 0x01, 0x07, 0x0a, 0x01, 0x06, 0x63, 0x61,
    0x6c, 0x6c, 0x65, 0x72, 0x00, 0x01, 0x09, 0x07,
    0x01, 0x00, 0x41, 0x00, 0x0b, 0x01, 0x00, 0x0a,
    0x13, 0x02, 0x07, 0x00, 0x20, 0x00, 0x41, 0x01,
    0x6a, 0x0b, 0x09, 0x00, 0x20, 0x01, 0x20, 0x00,
    0x11, 0x00, 0x00, 0x0b, 0x00, 0x20, 0x04, 0x6e,
    0x61, 0x6d, 0x65, 0x01, 0x0f, 0x02, 0x00, 0x04,
    0x61, 0x64, 0x64, 0x31, 0x01, 0x06, 0x63, 0x61,
    0x6c, 0x6c, 0x65, 0x72, 0x04, 0x08, 0x01, 0x00,
    0x05, 0x75, 0x6e, 0x61, 0x72, 0x79,
};

test "#879 M4.6 phase 2: call_indirect to a still-pending lazy function compiles it on demand via a native trampoline" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    var lazy_out: component_aot_compile.LazyJitOut = .{};
    const cwasm = try component_aot_compile.compileCoreWasmCached(
        gpa,
        &lazy_call_indirect_wasm,
        .{ .lazy_jit = true },
        .{ .lazy_jit_out = &lazy_out },
    );
    defer gpa.free(cwasm);

    // add1 (local index 0) is leaf and never a *direct* call target,
    // but IS placed in the table -- eligible, and needs a trampoline
    // (unlike every other fixture in this file). caller (local index
    // 1) contains call_indirect, so it's not a leaf and is compiled
    // eagerly as usual.
    try std.testing.expectEqual(@as(usize, 1), lazy_out.lazy_local_indices.len);
    try std.testing.expectEqual(@as(u32, 0), lazy_out.lazy_local_indices[0]);
    try std.testing.expectEqual(@as(usize, 1), lazy_out.needs_trampoline_indices.len);
    try std.testing.expectEqual(@as(u32, 0), lazy_out.needs_trampoline_indices[0]);

    var module = try aot_loader_mod.load(cwasm, gpa);
    defer aot_loader_mod.unload(&module, gpa);

    const inst = try aot_runtime_mod.instantiate(&module, gpa);
    defer aot_runtime_mod.destroy(inst);

    // Order matters -- see setupLazyJit's doc comment: it must run
    // before mapCodeExecutable so add1's trampoline address is ready
    // in time to be installed into the table.
    const driver = try component_aot_compile.setupLazyJit(inst, lazy_out, gpa);
    defer driver.deinit();
    try aot_runtime_mod.mapCodeExecutable(inst);

    try std.testing.expect(inst.lazy_jit.pending[0]); // add1 still pending

    const caller_idx = aot_runtime_mod.findExportFunc(inst, "caller") orelse return error.ExportNotFound;
    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;

    // caller(idx=0, x=41): call_indirect through table[0] -- add1's
    // trampoline -- must compile add1 on demand and return 41+1=42.
    const results = try aot_runtime_mod.callFuncScalar(
        inst,
        caller_idx,
        &.{ .i32, .i32 },
        &.{.i32},
        &.{ .{ .i32 = 0 }, .{ .i32 = 41 } },
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 42), results[0].i32);

    // add1 must now be resolved -- proving the call_indirect actually
    // went through the trampoline and triggered on-demand compilation,
    // not that this test is vacuously passing some other way.
    try std.testing.expect(!inst.lazy_jit.pending[0]);

    // A second call_indirect must reuse the already-compiled code
    // (the trampoline's own first call already patched func_addrs, so
    // this call doesn't even reach the trampoline anymore) and still
    // return the correct result.
    const results2 = try aot_runtime_mod.callFuncScalar(
        inst,
        caller_idx,
        &.{ .i32, .i32 },
        &.{.i32},
        &.{ .{ .i32 = 0 }, .{ .i32 = 99 } },
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 100), results2[0].i32);
}

// #879 M4.6 phase 2: a leaf, never-directly-called, never-tabled
// function reachable only via `ref.func` + `call_ref`. Needs a
// declarative element segment naming `$add1` -- wasm validation
// requires any `ref.func`-referenced function to be "declared"
// somewhere (an active/passive/declarative element segment all
// count); a declarative one contributes nothing to any live table.
// Generated via: wasm-tools parse lazy_call_ref.wat -o lazy_call_ref_fixture.wasm
//   (module
//     (type $unary (func (param i32) (result i32)))
//     (elem declare func $add1)
//     (func $add1 (param i32) (result i32)
//       local.get 0 i32.const 1 i32.add)
//     (func $caller (export "caller") (param i32) (result i32)
//       local.get 0 ref.func $add1 call_ref $unary))
const lazy_call_ref_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x03, 0x03, 0x02, 0x00, 0x00, 0x07, 0x0a, 0x01,
    0x06, 0x63, 0x61, 0x6c, 0x6c, 0x65, 0x72, 0x00,
    0x01, 0x09, 0x05, 0x01, 0x03, 0x00, 0x01, 0x00,
    0x0a, 0x12, 0x02, 0x07, 0x00, 0x20, 0x00, 0x41,
    0x01, 0x6a, 0x0b, 0x08, 0x00, 0x20, 0x00, 0xd2,
    0x00, 0x14, 0x00, 0x0b, 0x00, 0x20, 0x04, 0x6e,
    0x61, 0x6d, 0x65, 0x01, 0x0f, 0x02, 0x00, 0x04,
    0x61, 0x64, 0x64, 0x31, 0x01, 0x06, 0x63, 0x61,
    0x6c, 0x6c, 0x65, 0x72, 0x04, 0x08, 0x01, 0x00,
    0x05, 0x75, 0x6e, 0x61, 0x72, 0x79,
};

test "#879 M4.6 phase 2: ref.func + call_ref to a still-pending lazy function compiles it on demand via a native trampoline" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    var lazy_out: component_aot_compile.LazyJitOut = .{};
    const cwasm = try component_aot_compile.compileCoreWasmCached(
        gpa,
        &lazy_call_ref_wasm,
        .{ .lazy_jit = true },
        .{ .lazy_jit_out = &lazy_out },
    );
    defer gpa.free(cwasm);

    // #879 M4.7 phase 2: `caller` (local idx 1) now qualifies too --
    // its only outgoing "call" is `call_ref`, which was already
    // indirect and never needed the leaf restriction in the first
    // place (rule 1 relaxed to "no `call_indirect`"). It's never
    // itself a direct `.call` target (rule 2, unchanged), so both
    // functions are eligible now.
    try std.testing.expectEqual(@as(usize, 2), lazy_out.lazy_local_indices.len);
    try std.testing.expectEqual(@as(u32, 0), lazy_out.lazy_local_indices[0]);
    try std.testing.expectEqual(@as(u32, 1), lazy_out.lazy_local_indices[1]);
    // Only `add1` (local idx 0) is reachable via `ref.func` anywhere in
    // the module, so only it needs a trampoline -- `caller` is only
    // ever reached via a direct host `callFuncScalar` export call,
    // already covered by `LazyJitState.resolve` without one.
    try std.testing.expectEqual(@as(usize, 1), lazy_out.needs_trampoline_indices.len);
    try std.testing.expectEqual(@as(u32, 0), lazy_out.needs_trampoline_indices[0]);

    var module = try aot_loader_mod.load(cwasm, gpa);
    defer aot_loader_mod.unload(&module, gpa);

    const inst = try aot_runtime_mod.instantiate(&module, gpa);
    defer aot_runtime_mod.destroy(inst);

    const driver = try component_aot_compile.setupLazyJit(inst, lazy_out, gpa);
    defer driver.deinit();
    try aot_runtime_mod.mapCodeExecutable(inst);

    try std.testing.expect(inst.lazy_jit.pending[0]); // add1 still pending
    try std.testing.expect(inst.lazy_jit.pending[1]); // caller also still pending now

    const caller_idx = aot_runtime_mod.findExportFunc(inst, "caller") orelse return error.ExportNotFound;
    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;

    const results = try aot_runtime_mod.callFuncScalar(
        inst,
        caller_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 41 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 42), results[0].i32);
    try std.testing.expect(!inst.lazy_jit.pending[0]); // add1 compiled via the call_ref/trampoline dispatch
    try std.testing.expect(!inst.lazy_jit.pending[1]); // caller compiled via the callFuncScalar resolve() hook
}

// #879 M4.8 (phase 1): precompileComponentInMemory must produce one
// independently-computed LazyJitOut per core module, not a single
// shared/conflated one -- reuses the existing h1-compose fixture
// (see instance.zig's own "#156 H1" test): core 0 ($A) exports a
// single leaf function "f" (returns the constant 7, no calls at all
// -- fully lazy-eligible); core 1 ($B) imports "a"."f" and calls it
// from "g" (a `.call` to a cross-instance import).
//
// #879 M4.7 phase 2: "g"'s `.call` targets an IMPORT (func_idx <
// import_count), which was never actually unsafe to defer -- import
// calls already dispatch indirectly through `vmctx.host_functions[]`,
// regardless of where the caller's own code lives, so
// `rewriteLocalCallsToIndirect` deliberately leaves them untouched (no
// rewrite needed). Before this phase, the leaf check disqualified ANY
// `.call` (import or local) indiscriminately, so "g" was ineligible
// purely because of this overly-conservative rule, not because it was
// genuinely unsafe. "g" is eligible now too, independently proving
// each core's eligibility set is still computed independently (core
// 0's "f" and core 1's "g" are unrelated functions in unrelated
// modules).
test "#879 M4.8: precompileComponentInMemory produces one independently-eligible LazyJitOut per core" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;
    const data = @embedFile("h1_compose_wasm");

    var in_mem = try component_aot_compile.precompileComponentInMemory(gpa, data, .{ .lazy_jit = true });
    defer in_mem.deinit();

    try std.testing.expectEqual(@as(usize, 2), in_mem.lazy_jit_outs.len);
    try std.testing.expectEqual(@as(usize, 1), in_mem.lazy_jit_outs[0].lazy_local_indices.len);
    // #879 M4.7 phase 2: "g" (core 1's only function) is eligible too
    // now -- see the doc comment above.
    try std.testing.expectEqual(@as(usize, 1), in_mem.lazy_jit_outs[1].lazy_local_indices.len);

    // Exercise core 0's LazyJitOut end-to-end (consumed via
    // setupLazyJit, same as every single-core test above).
    var module = try aot_loader_mod.load(in_mem.pcs[0].cwasm_bytes, gpa);
    defer aot_loader_mod.unload(&module, gpa);
    const inst = try aot_runtime_mod.instantiate(&module, gpa);
    defer aot_runtime_mod.destroy(inst);

    const driver = try component_aot_compile.setupLazyJit(inst, in_mem.lazy_jit_outs[0], gpa);
    // Hollow the source slot out immediately after handing its
    // contents to setupLazyJit -- see InMemoryPrecompiled.lazy_jit_outs'
    // doc comment for why this makes in_mem.deinit() (deferred above)
    // safe to unconditionally clean up every entry.
    in_mem.lazy_jit_outs[0] = .{ .ir_module = wamr.ir.IrModule.init(gpa), .allocator = gpa };
    defer driver.deinit();
    try aot_runtime_mod.mapCodeExecutable(inst);

    try std.testing.expect(inst.lazy_jit.pending[0]); // f still pending

    const f_idx = aot_runtime_mod.findExportFunc(inst, "f") orelse return error.ExportNotFound;
    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;
    const results = try aot_runtime_mod.callFuncScalar(inst, f_idx, &.{}, &.{.i32}, &.{}, &results_buf);
    try std.testing.expectEqual(@as(i32, 7), results[0].i32);
    try std.testing.expect(!inst.lazy_jit.pending[0]);

    // Core 1's LazyJitOut was never consumed -- in_mem.deinit() (see
    // its deferred call above) now cleans it up automatically.
}

// #879 M4.8 phase 2: the same h1-compose fixture, but instantiated
// through the REAL component-instantiation path
// (`component_instance.instantiateWithOptions`, exactly like `wamr
// run`/`wamr serve` would use it) instead of directly driving a single
// core's `AotInstance` by hand. Proves the `PrecompiledCore.lazy_jit_setup`
// hook wired in `instance.zig`'s AOT core-instantiation block actually
// runs: core 0's leaf function "f" stays deferred until the
// component-level call to "g" (which cross-instance-calls "f")
// resolves it on demand.
test "#879 M4.8 phase 2: a real component instantiation defers a core function and compiles it on demand" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;
    const data = @embedFile("h1_compose_wasm");

    var in_mem = try component_aot_compile.precompileComponentInMemory(gpa, data, .{ .lazy_jit = true });
    defer in_mem.deinit();
    try std.testing.expectEqual(@as(usize, 1), in_mem.lazy_jit_outs[0].lazy_local_indices.len);

    // Independent parse of the SAME `data` buffer -- `precompileComponentInMemory`'s
    // own doc comment establishes this is a zero-copy, byte-identical
    // re-parse, which is what lets `findPrecompiled`/`findLazyJitSetup`
    // match `pcs[i].core_wasm` against this component's
    // `core_modules[i].data` by slice identity.
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var component = try wamr.component_loader.load(data, arena.allocator());

    const inst = try wamr.component_instance.instantiateWithOptions(&component, gpa, .{
        .precompiled_cores = in_mem.precompiledCores(),
        .aot_only = true,
    });
    defer inst.deinit();

    var providers: std.StringHashMapUnmanaged(wamr.component_instance.ImportBinding) = .empty;
    defer providers.deinit(gpa);
    try inst.linkImports(providers);

    // Core 0's AotInstance backs core-instance slot 0 (see the H1
    // fixture test in instance.zig) -- its "f" must still be pending
    // immediately after instantiation+linking, before "g" is ever called.
    const core0 = inst.core_instances[0].aot_inst orelse return error.ExpectedAotCore;
    try std.testing.expect(core0.lazy_jit.pending[0]);

    var args: [0]wamr.canonical_abi.InterfaceValue = .{};
    var results: [1]wamr.canonical_abi.InterfaceValue = undefined;
    try wamr.component_executor.callComponentFunc(inst, "g", &args, &results, gpa);
    try std.testing.expectEqual(@as(u32, 8), results[0].u32);

    // "f" must now be resolved -- proving the cross-instance call from
    // "g" genuinely went through the lazy-compile hook, not that this
    // test is vacuously passing some other way.
    try std.testing.expect(!core0.lazy_jit.pending[0]);
}
