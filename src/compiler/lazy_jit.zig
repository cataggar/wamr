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
//!  3. **Never reachable via `call_indirect`**: conservatively, if the
//!     module has ANY element segments at all, no function is
//!     considered eligible (see `elements_present` below). A function
//!     placed in *any* table by *any* active element segment could be
//!     invoked via `call_indirect`, which this narrow spike's runtime
//!     hook (`AotInstance`'s `callFuncScalar`-only interception, see
//!     `runtime.zig`) does not cover — only top-level host-invoked
//!     (`callFuncScalar`) calls trigger lazy compilation in this
//!     prototype. A real implementation would need `call_indirect`'s
//!     codegen and the `func_addrs` table itself to route through a
//!     native trampoline (see the design doc's "Deferred to follow-up"
//!     section) rather than relying on this narrower host-call-site
//!     interception.
//!
//! Only local (non-imported) function indices are considered; the
//! returned indices are LOCAL indices (0-based, excluding imports),
//! matching `ir_module.functions.items`' own indexing.
const std = @import("std");
const ir = @import("ir/ir.zig");
const core_types = @import("../runtime/common/types.zig");

/// Returns a caller-owned `[]bool` (indexed by LOCAL function index,
/// same length as `ir_module.functions.items`) where `true` marks a
/// lazy-eligible leaf function. Free with `allocator.free`.
pub fn findLazyEligibleLeaves(
    module: *const core_types.WasmModule,
    ir_module: *const ir.IrModule,
    allocator: std.mem.Allocator,
) ![]bool {
    const n = ir_module.functions.items.len;
    const eligible = try allocator.alloc(bool, n);
    @memset(eligible, false);

    // Rule 3: any element segment at all disqualifies every function in
    // this narrow spike (see doc comment above). Real wasm modules with
    // no `table`/`elem` sections (e.g. many small CLI-style utilities)
    // hit this trivially; anything using `call_indirect`/function
    // references does not qualify for this prototype.
    if (module.elements.len > 0) return eligible;

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

        var is_leaf = true;
        outer: for (func.blocks.items) |block| {
            for (block.instructions.items) |inst| {
                switch (inst.op) {
                    .call, .call_indirect => {
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

    var module = core_types.WasmModule{};
    module.elements = &.{};

    const eligible = try findLazyEligibleLeaves(&module, &ir_module, allocator);
    defer allocator.free(eligible);

    try std.testing.expect(eligible.len == 1);
    try std.testing.expect(eligible[0]);
}

test "findLazyEligibleLeaves: any element segment disqualifies everything" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();
    var f0 = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try f0.newBlock();
    try ir_module.functions.append(allocator, f0);

    var module = core_types.WasmModule{};
    const one_elem = [_]core_types.ElemSegment{.{
        .table_idx = 0,
        .offset = null,
        .kind = .func_ref,
        .func_indices = &.{},
    }};
    module.elements = &one_elem;

    const eligible = try findLazyEligibleLeaves(&module, &ir_module, allocator);
    defer allocator.free(eligible);

    try std.testing.expect(!eligible[0]);
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

    var module = core_types.WasmModule{};
    module.elements = &.{};

    const eligible = try findLazyEligibleLeaves(&module, &ir_module, allocator);
    defer allocator.free(eligible);

    try std.testing.expect(!eligible[0]); // called directly
    try std.testing.expect(!eligible[1]); // not a leaf (calls fn 0)
    try std.testing.expect(eligible[2]); // leaf, uncalled
}
