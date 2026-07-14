//! Lazy-JIT eligibility analysis for x86_64 direct-call graphs (#887) and
//! table/`call_indirect`/`ref.func`-reachable leaf functions (#888).
//!
//! Given a parsed core wasm module (for table / element-segment info) and its
//! lowered IR (for call / funcref usage), decide which LOCAL function indices
//! are safe to defer under the current lazy-JIT implementation, and which of
//! the two runtime dispatch mechanisms each deferred function needs:
//!
//!  - **Stub mechanism** (#887): a small, always-resident entry stub is
//!    emitted at the function's normal text-section offset, so direct
//!    `.call`/tail-call patches (and eager callers) keep working exactly as
//!    if the function had been compiled eagerly; the stub compiles the real
//!    body on first entry and forwards through it. Available to any
//!    function that is never reachable through a table, `ref.func`, or
//!    `call_indirect`/`call_ref` — i.e. only ever entered via `callFuncScalar`
//!    or a direct intra-module call. Non-leaf functions are fine here.
//!
//!  - **Trampoline mechanism** (#888): a per-instance native trampoline
//!    (`host_trampolines.TrampolinePool`) is allocated up front and its
//!    stable address is published into `func_addrs`/`funcptrs`/tables
//!    wherever the runtime normally mirrors callable pointers — needed
//!    because `call_indirect`/`call_ref`/`ref.func` all assume a real,
//!    callable native address exists at load time, before any lazy body has
//!    been compiled. Only available to LEAF functions (the trampoline's
//!    `genericDispatcher` envelope can't itself make further outgoing
//!    calls) that are never a direct `.call` target and whose lowered
//!    signature fits the trampoline's scalar calling convention.
//!
//! A function that itself contains `.call_indirect` or `.call_ref` is never
//! lazy-eligible under either mechanism (out of scope for both spikes).
//!
//! The returned slices are indexed by LOCAL function index (0-based,
//! excluding imports), matching `ir_module.functions.items`.
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

fn markLocalIfPresent(
    marks: []bool,
    import_count: u32,
    func_idx: u32,
) void {
    if (func_idx < import_count) return;
    const local_idx = func_idx - import_count;
    if (local_idx < marks.len) marks[local_idx] = true;
}

/// Per-function lazy-eligibility result. `eligible[i]` is `true` iff local
/// function `i` may be deferred at all; `needs_trampoline[i]` (meaningful
/// only where `eligible[i]` is `true`) selects the native trampoline,
/// while `needs_stub[i]` selects a text-section entry stub.
pub const LazyEligibility = struct {
    eligible: []bool,
    needs_trampoline: []bool,
    needs_stub: []bool,

    pub fn deinit(self: *LazyEligibility, allocator: std.mem.Allocator) void {
        allocator.free(self.eligible);
        allocator.free(self.needs_trampoline);
        allocator.free(self.needs_stub);
    }
};

/// Returns caller-owned slices (see `LazyEligibility`). Free with
/// `result.deinit(allocator)`.
pub fn findLazyEligibleFunctions(
    module: *const core_types.WasmModule,
    ir_module: *const ir.IrModule,
    target_arch: std.Target.Cpu.Arch,
    allocator: std.mem.Allocator,
) !LazyEligibility {
    const n = ir_module.functions.items.len;

    const eligible = try allocator.alloc(bool, n);
    errdefer allocator.free(eligible);
    @memset(eligible, true);

    const needs_trampoline = try allocator.alloc(bool, n);
    errdefer allocator.free(needs_trampoline);
    @memset(needs_trampoline, false);
    const needs_stub = try allocator.alloc(bool, n);
    errdefer allocator.free(needs_stub);
    @memset(needs_stub, false);

    // Table-reachable functions need a stable pointer at load time, so they
    // can only be lazy via the trampoline mechanism.
    for (module.elements) |elem| {
        for (elem.func_indices) |maybe_func_idx| {
            if (maybe_func_idx) |func_idx| {
                markLocalIfPresent(needs_trampoline, ir_module.import_count, func_idx);
            }
        }
    }

    // `ref.func` also hands out a raw function pointer that can escape into
    // tables/funcref consumers, so its target needs the same stable pointer.
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
                    .ref_func => |func_idx| markLocalIfPresent(needs_trampoline, ir_module.import_count, func_idx),
                    else => {},
                }
            }
        }
    }

    for (ir_module.functions.items, 0..) |func, local_idx| {
        var is_leaf = true;
        outer: for (func.blocks.items) |block| {
            for (block.instructions.items) |inst| {
                switch (inst.op) {
                    .call_indirect, .call_ref => {
                        // A function that itself dispatches indirectly is
                        // out of scope for both mechanisms.
                        eligible[local_idx] = false;
                        is_leaf = false;
                        break :outer;
                    },
                    .call => is_leaf = false,
                    else => {},
                }
            }
        }
        if (!eligible[local_idx]) continue;

        if (needs_trampoline[local_idx]) {
            // Trampoline mechanism (#888): leaf only, never a direct call
            // target, and signature must fit the trampoline envelope.
            if (!is_leaf or called.isSet(local_idx)) {
                eligible[local_idx] = false;
                continue;
            }
            const wasm_func_idx: u32 = ir_module.import_count + @as(u32, @intCast(local_idx));
            const ft = module.getFuncType(wasm_func_idx) orelse {
                eligible[local_idx] = false;
                continue;
            };
            if (!supportsLazyTrampolineFuncType(ft, target_arch)) {
                eligible[local_idx] = false;
            }
        } else if (target_arch != .x86_64 and !is_leaf) {
            // Stub mechanism (#887): non-leaf lazy bodies rely on
            // `.local_call_lowering = .via_funcptrs`, an x86_64-only
            // codegen option (#890's aarch64 backend parity only covers
            // the original leaf-only stub emission, not indirect local-call
            // lowering) -- so non-x86_64 targets stay leaf-only here too.
            eligible[local_idx] = false;
        }
        // Root-only lazy functions resolve through callFuncScalar before
        // dispatch, so only direct-call targets need an eager text stub.
        if (eligible[local_idx] and !needs_trampoline[local_idx] and called.isSet(local_idx)) {
            needs_stub[local_idx] = true;
        }
    }

    return .{
        .eligible = eligible,
        .needs_trampoline = needs_trampoline,
        .needs_stub = needs_stub,
    };
}

test "findLazyEligibleFunctions: only direct callees need lazy stubs" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    var callee = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try callee.newBlock();
    try ir_module.functions.append(allocator, callee);

    var caller = ir.IrFunction.init(allocator, 0, 0, 0);
    const caller_block = try caller.newBlock();
    try caller.getBlock(caller_block).append(.{ .op = .{ .call = .{ .func_idx = 0 } } });
    try ir_module.functions.append(allocator, caller);

    var module = core_types.WasmModule{};
    module.elements = &.{};

    var result = try findLazyEligibleFunctions(&module, &ir_module, .x86_64, allocator);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), result.eligible.len);
    try std.testing.expect(result.eligible[0]);
    try std.testing.expect(result.eligible[1]);
    try std.testing.expect(!result.needs_trampoline[0]);
    try std.testing.expect(!result.needs_trampoline[1]);
    try std.testing.expect(result.needs_stub[0]);
    try std.testing.expect(!result.needs_stub[1]);
}

test "findLazyEligibleFunctions: ref.func makes the target trampoline-eligible, not disqualified" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    var target = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try target.newBlock();
    try ir_module.functions.append(allocator, target);

    var ref_user = ir.IrFunction.init(allocator, 0, 0, 0);
    const ref_block = try ref_user.newBlock();
    const ref_val = ref_user.newVReg();
    try ref_user.getBlock(ref_block).append(.{
        .op = .{ .ref_func = 0 },
        .dest = ref_val,
        .type = .i64,
    });
    try ir_module.functions.append(allocator, ref_user);

    var unrelated = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try unrelated.newBlock();
    try ir_module.functions.append(allocator, unrelated);

    const ft = core_types.FuncType{ .params = &.{}, .results = &.{.i32} };
    const module_functions = [_]core_types.WasmFunction{
        .{ .type_idx = 0, .func_type = ft, .local_count = 0, .locals = &.{}, .code = &.{} },
        .{ .type_idx = 0, .func_type = ft, .local_count = 0, .locals = &.{}, .code = &.{} },
        .{ .type_idx = 0, .func_type = ft, .local_count = 0, .locals = &.{}, .code = &.{} },
    };
    var module = core_types.WasmModule{};
    module.types = &.{ft};
    module.functions = &module_functions;
    module.elements = &.{};

    var result = try findLazyEligibleFunctions(&module, &ir_module, .x86_64, allocator);
    defer result.deinit(allocator);

    // `target` (fn 0) is a leaf, never directly called, trampoline-envelope
    // compatible -- eligible via the trampoline mechanism.
    try std.testing.expect(result.eligible[0]);
    try std.testing.expect(result.needs_trampoline[0]);
    // `ref_user` itself is a plain leaf, not table/ref.func-reachable ->
    // stub-eligible.
    try std.testing.expect(result.eligible[1]);
    try std.testing.expect(!result.needs_trampoline[1]);
    try std.testing.expect(result.eligible[2]);
    try std.testing.expect(!result.needs_trampoline[2]);
}

test "findLazyEligibleFunctions: call_indirect/call_ref callers are never eligible themselves" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    var elem_target = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try elem_target.newBlock();
    try ir_module.functions.append(allocator, elem_target);

    var indirect_user = ir.IrFunction.init(allocator, 0, 0, 0);
    const block = try indirect_user.newBlock();
    const elem_idx = indirect_user.newVReg();
    try indirect_user.getBlock(block).append(.{
        .op = .{ .iconst_32 = 0 },
        .dest = elem_idx,
        .type = .i32,
    });
    try indirect_user.getBlock(block).append(.{
        .op = .{ .call_indirect = .{ .type_idx = 0, .elem_idx = elem_idx } },
    });
    try ir_module.functions.append(allocator, indirect_user);

    var unrelated = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try unrelated.newBlock();
    try ir_module.functions.append(allocator, unrelated);

    const ft = core_types.FuncType{ .params = &.{.i32}, .results = &.{.i32} };
    const module_functions = [_]core_types.WasmFunction{
        .{ .type_idx = 0, .func_type = ft, .local_count = 0, .locals = &.{}, .code = &.{} },
        .{ .type_idx = 0, .func_type = ft, .local_count = 0, .locals = &.{}, .code = &.{} },
        .{ .type_idx = 0, .func_type = ft, .local_count = 0, .locals = &.{}, .code = &.{} },
    };
    var module = core_types.WasmModule{};
    module.types = &.{ft};
    module.functions = &module_functions;
    const elem_func_indices = [_]?u32{0};
    const one_elem = [_]core_types.ElemSegment{.{
        .table_idx = 0,
        .offset = null,
        .kind = .func_ref,
        .func_indices = &elem_func_indices,
    }};
    module.elements = &one_elem;

    var result = try findLazyEligibleFunctions(&module, &ir_module, .x86_64, allocator);
    defer result.deinit(allocator);

    // `elem_target` (fn 0) is table-reachable, leaf, never directly called,
    // envelope-compatible -> trampoline-eligible.
    try std.testing.expect(result.eligible[0]);
    try std.testing.expect(result.needs_trampoline[0]);
    // `indirect_user` (fn 1) performs `call_indirect` itself -> ineligible.
    try std.testing.expect(!result.eligible[1]);
    // `unrelated` (fn 2) -> stub-eligible.
    try std.testing.expect(result.eligible[2]);
    try std.testing.expect(!result.needs_trampoline[2]);
}

test "findLazyEligibleFunctions: a directly-called non-leaf function is stub-eligible" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    // Function 0: leaf, called by function 1 below -- still stub-eligible
    // (not table/ref.func-reachable).
    var f0 = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try f0.newBlock();
    try ir_module.functions.append(allocator, f0);

    // Function 1: calls function 0 -- non-leaf, but still stub-eligible.
    var f1 = ir.IrFunction.init(allocator, 0, 0, 0);
    const b1 = try f1.newBlock();
    try f1.getBlock(b1).append(.{ .op = .{ .call = .{ .func_idx = 0, .args = &.{} } } });
    try ir_module.functions.append(allocator, f1);

    var module = core_types.WasmModule{};
    module.elements = &.{};

    var result = try findLazyEligibleFunctions(&module, &ir_module, .x86_64, allocator);
    defer result.deinit(allocator);

    try std.testing.expect(result.eligible[0]);
    try std.testing.expect(!result.needs_trampoline[0]);
    try std.testing.expect(result.eligible[1]);
    try std.testing.expect(!result.needs_trampoline[1]);
    try std.testing.expect(result.needs_stub[0]);
    try std.testing.expect(!result.needs_stub[1]);
}

test "findLazyEligibleFunctions: call_ref callers are non-leaf but ref.func targets remain eligible" {
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
        .{ .type_idx = 0, .func_type = ft, .local_count = 0, .locals = &.{}, .code = &.{} },
        .{ .type_idx = 0, .func_type = ft, .local_count = 0, .locals = &.{}, .code = &.{} },
    };
    var module = core_types.WasmModule{};
    module.types = &.{ft};
    module.functions = &module_functions;
    module.elements = &.{};

    var result = try findLazyEligibleFunctions(&module, &ir_module, .x86_64, allocator);
    defer result.deinit(allocator);

    try std.testing.expect(result.eligible[0]);
    try std.testing.expect(result.needs_trampoline[0]);
    try std.testing.expect(!result.eligible[1]); // performs call_ref itself
}

test "findLazyEligibleFunctions: signatures outside the trampoline envelope stay eager when table-reachable" {
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
    const elem_func_indices = [_]?u32{0};
    var module = core_types.WasmModule{};
    module.types = &.{ft};
    module.functions = &module_functions;
    const one_elem = [_]core_types.ElemSegment{.{
        .table_idx = 0,
        .offset = null,
        .kind = .func_ref,
        .func_indices = &elem_func_indices,
    }};
    module.elements = &one_elem;

    var result = try findLazyEligibleFunctions(&module, &ir_module, .x86_64, allocator);
    defer result.deinit(allocator);

    // Table-reachable + too many params for the trampoline envelope -> not
    // eligible (would need the trampoline mechanism, but doesn't fit it).
    try std.testing.expect(!result.eligible[0]);
}
