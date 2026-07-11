//! #862 — lazy JIT design-spike eligibility analysis.
//!
//! Pure, compiler-side analysis: given a parsed core wasm module (pre-
//! lowering, for table/element-segment info) and its lowered IR (for
//! call-graph info), decide which LOCAL function indices are safe to
//! defer ("lazy-eligible") under the narrow leaf-functions-only spike
//! described in `docs/design/lazy-jit-spike.md`.
//!
//! A function is lazy-eligible only if ALL of the following hold:
//!
//!  1. **Leaf**: its own IR contains no `.call` or `.call_indirect`
//!     instruction (so deferring its compilation never blocks on some
//!     *other* not-yet-compiled function it would need to call).
//!  2. **Never a direct call target**: no other function's IR contains
//!     a `.call` whose `func_idx` names this function. Direct calls are
//!     resolved via a compile-time relative-branch patch computed from
//!     each function's final code offset (see `CallPatch` in both
//!     codegen backends) — deferring a function that's a direct-call
//!     target would leave that patch with nothing to point at.
//!  3. **Trampoline-compatible lowered signature**: the runtime-side
//!     native trampoline currently supports only the x86_64 scalar
//!     envelope used by `host_trampolines.genericDispatcher`: no retptr,
//!     no `v128`, at most one scalar result, and at most 9 user params
//!     after the hidden `vmctx`.
//!
//! Only local (non-imported) function indices are considered; the
//! returned indices are LOCAL indices (0-based, excluding imports),
//! matching `ir_module.functions.items`' own indexing.
const std = @import("std");
const ir = @import("ir/ir.zig");
const core_types = @import("../runtime/common/types.zig");

fn supportsLazyTrampolineValType(vt: core_types.ValType) bool {
    return switch (vt) {
        .i32, .i64, .f32, .f64, .funcref, .externref => true,
        else => false,
    };
}

fn supportsLazyTrampolineFuncType(ft: core_types.FuncType, target_arch: std.Target.Cpu.Arch) bool {
    if (target_arch != .x86_64) return false;
    if (ft.kind != .func) return false;
    if (ft.params.len > 9) return false;
    if (ft.results.len > 1) return false;
    for (ft.params) |pt| {
        if (!supportsLazyTrampolineValType(pt)) return false;
    }
    for (ft.results) |rt| {
        if (!supportsLazyTrampolineValType(rt)) return false;
    }
    return true;
}

/// Returns a caller-owned `[]bool` (indexed by LOCAL function index,
/// same length as `ir_module.functions.items`) where `true` marks a
/// lazy-eligible leaf function. Free with `allocator.free`.
pub fn findLazyEligibleLeaves(
    module: *const core_types.WasmModule,
    ir_module: *const ir.IrModule,
    target_arch: std.Target.Cpu.Arch,
    allocator: std.mem.Allocator,
) ![]bool {
    const n = ir_module.functions.items.len;
    const eligible = try allocator.alloc(bool, n);
    @memset(eligible, false);

    // Rule 2 precursor: collect every LOCAL func_idx that is the direct
    // target of some `.call` instruction anywhere in the module.
    var called: std.DynamicBitSetUnmanaged = try .initEmpty(allocator, n);
    defer called.deinit(allocator);

    for (ir_module.functions.items) |func| {
        for (func.blocks.items) |block| {
            for (block.instructions.items) |inst| {
                switch (inst.op) {
                    .call => |c| {
                        if (c.func_idx >= ir_module.import_count) {
                            const local_idx = c.func_idx - ir_module.import_count;
                            if (local_idx < n) called.set(local_idx);
                        }
                    },
                    else => {},
                }
            }
        }
    }

    for (ir_module.functions.items, 0..) |func, local_idx| {
        if (called.isSet(local_idx)) continue; // rule 2
        const wasm_func_idx: u32 = ir_module.import_count + @as(u32, @intCast(local_idx));
        const ft = module.getFuncType(wasm_func_idx) orelse continue;
        if (!supportsLazyTrampolineFuncType(ft, target_arch)) continue; // rule 3

        var is_leaf = true;
        outer: for (func.blocks.items) |block| {
            for (block.instructions.items) |inst| {
                switch (inst.op) {
                    .call, .call_indirect, .call_ref => {
                        is_leaf = false;
                        break :outer;
                    },
                    else => {},
                }
            }
        }
        if (is_leaf) eligible[local_idx] = true;
    }

    return eligible;
}

test "findLazyEligibleLeaves: leaf, uncalled, exported function is eligible" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    // Function 0: a trivial leaf (no calls at all).
    var f0 = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try f0.newBlock();
    try ir_module.functions.append(allocator, f0);

    const ft = core_types.FuncType{ .params = &.{}, .results = &.{.i32} };
    const module_functions = [_]core_types.WasmFunction{.{
        .type_idx = 0,
        .func_type = ft,
        .local_count = 0,
        .locals = &.{},
        .code = &.{},
    }};
    var module = core_types.WasmModule{};
    module.types = &.{ft};
    module.functions = &module_functions;
    module.elements = &.{};

    const eligible = try findLazyEligibleLeaves(&module, &ir_module, .x86_64, allocator);
    defer allocator.free(eligible);

    try std.testing.expect(eligible.len == 1);
    try std.testing.expect(eligible[0]);
}

test "findLazyEligibleLeaves: active element segments no longer blanket-disqualify leaf targets" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();
    var f0 = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try f0.newBlock();
    try ir_module.functions.append(allocator, f0);

    const ft = core_types.FuncType{ .params = &.{.i32}, .results = &.{.i32} };
    const module_functions = [_]core_types.WasmFunction{.{
        .type_idx = 0,
        .func_type = ft,
        .local_count = 0,
        .locals = &.{},
        .code = &.{},
    }};
    const elem_func_indices = [_]?u32{0};
    var module = core_types.WasmModule{};
    module.types = &.{ft};
    module.functions = &module_functions;
    const one_elem = [_]core_types.ElemSegment{.{
        .table_idx = 0,
        .offset = .{ .i32_const = 0 },
        .kind = .func_ref,
        .func_indices = &elem_func_indices,
        .nullable_elements = false,
    }};
    module.elements = &one_elem;

    const eligible = try findLazyEligibleLeaves(&module, &ir_module, .x86_64, allocator);
    defer allocator.free(eligible);

    try std.testing.expect(eligible[0]);
}

test "findLazyEligibleLeaves: a directly-called function is not eligible; a caller is not a leaf" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    // Function 0: leaf, but called by function 1 below -> not eligible (rule 2).
    var f0 = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try f0.newBlock();
    try ir_module.functions.append(allocator, f0);

    // Function 1: calls function 0 -> not eligible itself (rule 1, not a leaf).
    var f1 = ir.IrFunction.init(allocator, 0, 0, 0);
    const b1 = try f1.newBlock();
    try f1.getBlock(b1).append(.{ .op = .{ .call = .{ .func_idx = 0, .args = &.{} } } });
    try ir_module.functions.append(allocator, f1);

    // Function 2: leaf, uncalled -> eligible.
    var f2 = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try f2.newBlock();
    try ir_module.functions.append(allocator, f2);

    const ft = core_types.FuncType{ .params = &.{.i32}, .results = &.{.i32} };
    const module_functions = [_]core_types.WasmFunction{
        .{
            .type_idx = 0,
            .func_type = ft,
            .local_count = 0,
            .locals = &.{},
            .code = &.{},
        },
        .{
            .type_idx = 0,
            .func_type = ft,
            .local_count = 0,
            .locals = &.{},
            .code = &.{},
        },
        .{
            .type_idx = 0,
            .func_type = ft,
            .local_count = 0,
            .locals = &.{},
            .code = &.{},
        },
    };
    var module = core_types.WasmModule{};
    module.types = &.{ft};
    module.functions = &module_functions;
    module.elements = &.{};

    const eligible = try findLazyEligibleLeaves(&module, &ir_module, .x86_64, allocator);
    defer allocator.free(eligible);

    try std.testing.expect(!eligible[0]); // called directly
    try std.testing.expect(!eligible[1]); // not a leaf (calls fn 0)
    try std.testing.expect(eligible[2]); // leaf, uncalled
}

test "findLazyEligibleLeaves: call_ref callers are not leaves but ref.func targets remain eligible" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    var target = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try target.newBlock();
    try ir_module.functions.append(allocator, target);

    var caller = ir.IrFunction.init(allocator, 0, 0, 0);
    const b0 = try caller.newBlock();
    const fref = caller.newVReg();
    try caller.getBlock(b0).append(.{ .op = .{ .ref_func = 0 }, .dest = fref, .type = .i64 });
    try caller.getBlock(b0).append(.{
        .op = .{ .call_ref = .{ .type_idx = 0, .func_ref = fref, .args = &.{} } },
        .dest = caller.newVReg(),
        .type = .i32,
    });
    try ir_module.functions.append(allocator, caller);

    const ft = core_types.FuncType{ .params = &.{}, .results = &.{.i32} };
    const module_functions = [_]core_types.WasmFunction{
        .{
            .type_idx = 0,
            .func_type = ft,
            .local_count = 0,
            .locals = &.{},
            .code = &.{},
        },
        .{
            .type_idx = 0,
            .func_type = ft,
            .local_count = 0,
            .locals = &.{},
            .code = &.{},
        },
    };
    var module = core_types.WasmModule{};
    module.types = &.{ft};
    module.functions = &module_functions;

    const eligible = try findLazyEligibleLeaves(&module, &ir_module, .x86_64, allocator);
    defer allocator.free(eligible);

    try std.testing.expect(eligible[0]);
    try std.testing.expect(!eligible[1]);
}

test "findLazyEligibleLeaves: signatures outside the trampoline envelope stay eager" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    var f0 = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try f0.newBlock();
    try ir_module.functions.append(allocator, f0);

    const many_params = [_]core_types.ValType{
        .i32, .i32, .i32, .i32, .i32,
        .i32, .i32, .i32, .i32, .i32,
    };
    const ft = core_types.FuncType{ .params = &many_params, .results = &.{.i32} };
    const module_functions = [_]core_types.WasmFunction{.{
        .type_idx = 0,
        .func_type = ft,
        .local_count = 0,
        .locals = &.{},
        .code = &.{},
    }};
    var module = core_types.WasmModule{};
    module.types = &.{ft};
    module.functions = &module_functions;

    const eligible = try findLazyEligibleLeaves(&module, &ir_module, .x86_64, allocator);
    defer allocator.free(eligible);

    try std.testing.expect(!eligible[0]);
}
