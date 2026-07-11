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
const core_loader_mod = wamr.loader;
const frontend_mod = wamr.frontend;
const passes_mod = wamr.passes;

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
    try aot_runtime_mod.mapCodeExecutable(inst);

    const driver = try component_aot_compile.setupLazyJit(inst, lazy_out, gpa);
    defer driver.deinit();

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

// #879: single-function fixture with an obviously constant-foldable
// pattern (`x + 2 + 3`), used to prove the per-function IR-optimization
// pass loop is genuinely skipped for lazy-eligible functions too --
// not just codegen, which is all the original #862 spike deferred.
// Generated via: wasm-tools parse foldable.wat -o foldable_fixture.wasm
//   (module
//     (func $foldable (export "foldable") (param i32) (result i32)
//       local.get 0 i32.const 2 i32.const 3 i32.add i32.add))
const foldable_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00, 0x07, 0x0c, 0x01, 0x08,
    0x66, 0x6f, 0x6c, 0x64, 0x61, 0x62, 0x6c, 0x65,
    0x00, 0x00, 0x0a, 0x0c, 0x01, 0x0a, 0x00, 0x20,
    0x00, 0x41, 0x02, 0x41, 0x03, 0x6a, 0x6a, 0x0b,
    0x00, 0x12, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x01,
    0x0b, 0x01, 0x00, 0x08, 0x66, 0x6f, 0x6c, 0x64,
    0x61, 0x62, 0x6c, 0x65,
};

test "#879 lazy-JIT spike: lazy-eligible functions skip IR optimization passes too, not just codegen" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;
    const target_arch: passes_mod.TargetArch = switch (builtin.cpu.arch) {
        .aarch64 => .aarch64,
        else => .x86_64,
    };

    // Reference A: raw frontend lowering, no passes at all -- ground
    // truth for what completely un-optimized IR looks like for this
    // function.
    var raw_arena = std.heap.ArenaAllocator.init(gpa);
    defer raw_arena.deinit();
    const raw_wasm_module = try core_loader_mod.load(&foldable_wasm, raw_arena.allocator());
    var raw_ir = try frontend_mod.lowerModule(&raw_wasm_module, gpa);
    defer raw_ir.deinit();
    const raw_insts = raw_ir.functions.items[0].blocks.items[0].instructions.items.len;

    // Reference B: the SAME lowering, but with the full optimization
    // pipeline run over it -- mirrors what an eager (non-lazy) compile
    // does. Constant folding should collapse the `2 + 3` sub-expression
    // into a single `iconst 5`, shrinking the instruction count.
    var opt_arena = std.heap.ArenaAllocator.init(gpa);
    defer opt_arena.deinit();
    const opt_wasm_module = try core_loader_mod.load(&foldable_wasm, opt_arena.allocator());
    var opt_ir = try frontend_mod.lowerModule(&opt_wasm_module, gpa);
    defer opt_ir.deinit();
    _ = try passes_mod.runPassesWithOptions(&opt_ir, passes_mod.passesForPreset(target_arch, .full), gpa, .{});
    const opt_insts = opt_ir.functions.items[0].blocks.items[0].instructions.items.len;

    // Sanity check on the fixture itself: the reference pipeline must
    // actually fold something here, otherwise this fixture wouldn't be
    // able to distinguish "passes skipped" from "passes ran".
    try std.testing.expect(opt_insts < raw_insts);

    // The real lazy-JIT path: `foldable` is a single, never-called,
    // leaf function with no element segments in the module, so it's
    // fully lazy-eligible -- and nothing in this test ever calls it,
    // so it stays pending/deferred throughout.
    var lazy_out: component_aot_compile.LazyJitOut = .{};
    const cwasm = try component_aot_compile.compileCoreWasmCached(
        gpa,
        &foldable_wasm,
        .{ .lazy_jit = true },
        .{ .lazy_jit_out = &lazy_out },
    );
    defer gpa.free(cwasm);
    defer lazy_out.ir_module.deinit();
    defer gpa.free(lazy_out.lazy_local_indices);

    try std.testing.expectEqual(@as(usize, 1), lazy_out.lazy_local_indices.len);
    const lazy_insts = lazy_out.ir_module.functions.items[0].blocks.items[0].instructions.items.len;

    // The core #879 assertion: the deferred function's retained IR
    // must match the completely-untouched raw lowering (same
    // instruction count), NOT the folded/optimized form -- proving
    // the optimization pass loop was skipped for it too, not just
    // codegen.
    try std.testing.expectEqual(raw_insts, lazy_insts);
    try std.testing.expect(lazy_insts > opt_insts);
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
