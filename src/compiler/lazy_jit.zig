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
//!  1. **Leaf**: its own IR contains no `.call`, `.call_indirect`, or
//!     `.call_ref` instruction (so deferring its compilation never
//!     blocks on some *other* not-yet-compiled function it would need
//!     to call).
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

/// #879 M4.6 (phase 1): result of `findLazyEligibleWithTrampoline` --
/// a strict superset of what `findLazyEligibleLeaves` finds, now also
/// admitting leaf, never-directly-called functions that ARE referenced
/// by an element segment and/or a `ref.func` instruction (rule 3's
/// "any element segment disqualifies everything" restriction, lifted).
pub const EligibilityResult = struct {
    /// Indexed by LOCAL function index, same as `findLazyEligibleLeaves`'s
    /// return. `true` marks a lazy-eligible function -- deferring its
    /// compilation is safe.
    eligible: []bool,
    /// Indexed the same way; only ever `true` where `eligible[i]` is
    /// also `true`. Marks a function that's reachable via
    /// `call_indirect` (present in some element segment's
    /// `func_indices`) and/or `ref.func`/`call_ref` (its index is the
    /// operand of some `.ref_func` instruction anywhere in the
    /// module). For these, `func_addrs[i]`/the table's native backing
    /// must be populated with a real callable trampoline (see
    /// `lazy_call_trampoline.zig`) rather than left as an
    /// unreachable-by-construction placeholder -- something *could*
    /// call through this address before the function is ever compiled.
    needs_trampoline: []bool,

    pub fn deinit(self: *EligibilityResult, allocator: std.mem.Allocator) void {
        allocator.free(self.eligible);
        allocator.free(self.needs_trampoline);
    }
};

/// #879 M4.6 (phase 1) -- see `EligibilityResult`'s doc comment.
///
/// NOT wired into `compileCoreWasmCached`/`mapCodeExecutable` yet
/// (phase 2, a separate follow-up): this is deliberately a standalone,
/// independently-testable analysis so the (already-large,
/// correctness-critical) `func_addrs`/table-population code in
/// `runtime.zig` only needs to change once, after the trampoline
/// mechanism itself (`lazy_call_trampoline.zig`) is proven correct in
/// isolation. See `docs/design/lazy-jit-spike.md`'s phase-split note.
pub fn findLazyEligibleWithTrampoline(
    module: *const core_types.WasmModule,
    ir_module: *const ir.IrModule,
    allocator: std.mem.Allocator,
) !EligibilityResult {
    const n = ir_module.functions.items.len;
    const eligible = try allocator.alloc(bool, n);
    errdefer allocator.free(eligible);
    @memset(eligible, false);
    const needs_trampoline = try allocator.alloc(bool, n);
    errdefer allocator.free(needs_trampoline);
    @memset(needs_trampoline, false);

    // Rule 2 precursor (same as `findLazyEligibleLeaves`): collect every
    // LOCAL func_idx directly targeted by some `.call`.
    var called: std.DynamicBitSetUnmanaged = try .initEmpty(allocator, n);
    defer called.deinit(allocator);

    // Table/funcref reachability: union of every element segment's
    // `func_indices` (active, passive, AND declarative -- all three
    // mean "something outside this function's own control chain can
    // obtain a callable reference to it", which is exactly the
    // condition `func_addrs[i]` must be safe against) and every
    // `.ref_func` instruction's operand anywhere in the module.
    var reachable: std.DynamicBitSetUnmanaged = try .initEmpty(allocator, n);
    defer reachable.deinit(allocator);

    for (module.elements) |seg| {
        for (seg.func_indices) |maybe_idx| {
            const idx = maybe_idx orelse continue;
            if (idx >= ir_module.import_count) {
                const local_idx = idx - ir_module.import_count;
                if (local_idx < n) reachable.set(local_idx);
            }
        }
    }

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
                    .ref_func => |fidx| {
                        if (fidx >= ir_module.import_count) {
                            const local_idx = fidx - ir_module.import_count;
                            if (local_idx < n) reachable.set(local_idx);
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
                    .call, .call_indirect, .call_ref => {
                        is_leaf = false;
                        break :outer;
                    },
                    else => {},
                }
            }
        }
        if (!is_leaf) continue;

        eligible[local_idx] = true;
        if (reachable.isSet(local_idx)) needs_trampoline[local_idx] = true;
    }

    return .{ .eligible = eligible, .needs_trampoline = needs_trampoline };
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

test "findLazyEligibleWithTrampoline: function placed in an active element segment needs a trampoline" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    // Function 0: leaf, never called directly -- but placed in table 0
    // by an active element segment below, so it's reachable via
    // call_indirect.
    var f0 = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try f0.newBlock();
    try ir_module.functions.append(allocator, f0);

    var module = core_types.WasmModule{};
    const one_elem = [_]core_types.ElemSegment{.{
        .table_idx = 0,
        .offset = null,
        .kind = .func_ref,
        .func_indices = &[_]?u32{0},
    }};
    module.elements = &one_elem;

    var result = try findLazyEligibleWithTrampoline(&module, &ir_module, allocator);
    defer result.deinit(allocator);

    try std.testing.expect(result.eligible[0]);
    try std.testing.expect(result.needs_trampoline[0]);
}

test "findLazyEligibleWithTrampoline: function named by ref.func needs a trampoline" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    // Function 0: leaf, never called directly.
    var f0 = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try f0.newBlock();
    try ir_module.functions.append(allocator, f0);

    // Function 1: obtains a reference to function 0 via ref.func (e.g.
    // to store it in a global or pass it to an import) -- function 0
    // is now reachable via call_ref even though nothing here directly
    // calls it.
    var f1 = ir.IrFunction.init(allocator, 0, 0, 0);
    const b1 = try f1.newBlock();
    try f1.getBlock(b1).append(.{ .op = .{ .ref_func = 0 } });
    try ir_module.functions.append(allocator, f1);

    var module = core_types.WasmModule{};
    module.elements = &.{};

    var result = try findLazyEligibleWithTrampoline(&module, &ir_module, allocator);
    defer result.deinit(allocator);

    try std.testing.expect(result.eligible[0]);
    try std.testing.expect(result.needs_trampoline[0]);
    // Function 1 itself has no calls of its own kind that disqualify it
    // (ref_func isn't `.call`/`.call_indirect`), isn't in any element
    // segment, and nothing calls it directly -- eligible, no trampoline
    // needed.
    try std.testing.expect(result.eligible[1]);
    try std.testing.expect(!result.needs_trampoline[1]);
}

test "findLazyEligibleWithTrampoline: unreferenced leaf function needs no trampoline (matches findLazyEligibleLeaves)" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();
    var f0 = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try f0.newBlock();
    try ir_module.functions.append(allocator, f0);

    var module = core_types.WasmModule{};
    module.elements = &.{};

    var result = try findLazyEligibleWithTrampoline(&module, &ir_module, allocator);
    defer result.deinit(allocator);

    try std.testing.expect(result.eligible[0]);
    try std.testing.expect(!result.needs_trampoline[0]);
}

test "findLazyEligibleWithTrampoline: a directly-called function stays ineligible even if also tabled" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    // Function 0: placed in the table AND directly called by function 1.
    var f0 = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try f0.newBlock();
    try ir_module.functions.append(allocator, f0);

    var f1 = ir.IrFunction.init(allocator, 0, 0, 0);
    const b1 = try f1.newBlock();
    try f1.getBlock(b1).append(.{ .op = .{ .call = .{ .func_idx = 0, .args = &.{} } } });
    try ir_module.functions.append(allocator, f1);

    var module = core_types.WasmModule{};
    const one_elem = [_]core_types.ElemSegment{.{
        .table_idx = 0,
        .offset = null,
        .kind = .func_ref,
        .func_indices = &[_]?u32{0},
    }};
    module.elements = &one_elem;

    var result = try findLazyEligibleWithTrampoline(&module, &ir_module, allocator);
    defer result.deinit(allocator);

    // Rule 2 (never a direct call target) still applies -- being
    // tabled doesn't override it.
    try std.testing.expect(!result.eligible[0]);
    try std.testing.expect(!result.needs_trampoline[0]);
}

test "findLazyEligibleWithTrampoline: a function containing call_ref is not a leaf" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    // Function 0: leaf, never called directly, never tabled -- eligible
    // (and reachable only via whatever obtains its funcref elsewhere).
    var f0 = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try f0.newBlock();
    try ir_module.functions.append(allocator, f0);

    // Function 1: performs a call_ref (e.g. to a funcref obtained from
    // a table.get or ref.func elsewhere) -- this IS an outgoing call,
    // so function 1 itself must NOT be treated as a leaf, regardless
    // of the fact call_ref isn't literally spelled `.call`/`.call_indirect`.
    var f1 = ir.IrFunction.init(allocator, 0, 1, 1);
    const b1 = try f1.newBlock();
    try f1.getBlock(b1).append(.{ .op = .{ .call_ref = .{ .type_idx = 0, .func_ref = 0, .args = &.{} } } });
    try ir_module.functions.append(allocator, f1);

    var module = core_types.WasmModule{};
    module.elements = &.{};

    var result = try findLazyEligibleWithTrampoline(&module, &ir_module, allocator);
    defer result.deinit(allocator);

    try std.testing.expect(result.eligible[0]);
    try std.testing.expect(!result.eligible[1]); // not a leaf (performs call_ref)
}

/// #879 M4.7 (phase 1): rewrites every `.call` instruction in `func`
/// that targets a LOCAL (non-imported) function into an equivalent
/// `.ref_func` + `.call_ref` pair, preserving `dest`/`type`/`args`/
/// `extra_results`/`tail` exactly.
///
/// Why this is needed: direct `.call` codegen patches a compile-time
/// PC-relative branch (`CallPatch`, resolved against each function's
/// final offset within one contiguous code blob) to reach its target.
/// A lazily-compiled function's machine code instead lives in its own,
/// separately-`mmap`'d region (see `mapExecutableCode` /
/// `lazy_call_trampoline.zig`) that can be gigabytes away from the
/// main blob -- a rel32 branch cannot reliably reach it, and the
/// offset isn't even a compile-time constant. `.ref_func`/`.call_ref`
/// already dispatch indirectly through `vmctx.funcptrs[idx]` (see both
/// backends' `.ref_func` codegen) -- the SAME array `mapCodeExecutable`
/// populates with either a function's real compiled address or a
/// trampoline address -- so this rewrite is correct regardless of
/// whether the callee is itself deferred, already compiled, or
/// deferred-but-reachable-only-through-a-trampoline. Self-recursive
/// calls are also safe: `funcptrs[F]` is read at call *execution*
/// time, not compile time, and is updated to F's real address before
/// F's own first-call compile returns control to any caller.
///
/// Import calls (`func_idx < import_count`) are left untouched: their
/// `.call` codegen already dispatches indirectly through
/// `vmctx.host_functions[]`, so rewriting them would add overhead for
/// no benefit.
///
/// `func_type_indices` is the module's function-index → wasm-type-index
/// table (`IrModule.func_type_indices.items`, imports first then
/// locals) -- used only to populate `.call_ref`'s `type_idx` field,
/// which every backend's `.call_ref` codegen treats as informational
/// (it does not affect dispatch: `.call_ref` targets are already
/// statically typed by validation, unlike `call_indirect`'s runtime
/// table-entry signature check).
///
/// This is a standalone, independently-testable IR transformation
/// (deliberately not yet wired into any real compile path -- that's a
/// later phase, mirroring M4.6/M4.8's phase split; see
/// `docs/design/lazy-jit-spike.md`).
///
/// Uses `func`'s own per-block `allocator` (recorded on each
/// `BasicBlock` at construction) to grow the instruction list -- no
/// separate allocator parameter is needed.
pub fn rewriteLocalCallsToIndirect(
    func: *ir.IrFunction,
    import_count: u32,
    func_type_indices: []const u32,
) !void {
    for (func.blocks.items) |*block| {
        var i: usize = 0;
        while (i < block.instructions.items.len) : (i += 1) {
            const call = switch (block.instructions.items[i].op) {
                .call => |c| c,
                else => continue,
            };
            if (call.func_idx < import_count) continue; // import: already indirect

            const dest = block.instructions.items[i].dest;
            const result_type = block.instructions.items[i].type;
            const type_idx = if (call.func_idx < func_type_indices.len) func_type_indices[call.func_idx] else 0;

            const ref_dest = func.newVReg();
            try block.instructions.insert(block.allocator, i, .{
                .op = .{ .ref_func = call.func_idx },
                .dest = ref_dest,
                .type = .i64,
            });
            i += 1;

            // Overwrite the original `.call` slot in place with the
            // equivalent `.call_ref`. `call.args` transfers ownership
            // as-is (same allocation, no dupe/free) -- see `Inst.
            // freeOwnedSlices`'s `.call`/`.call_ref` arms, which both
            // free `args` the same way.
            block.instructions.items[i] = .{
                .op = .{ .call_ref = .{
                    .type_idx = type_idx,
                    .func_ref = ref_dest,
                    .args = call.args,
                    .extra_results = call.extra_results,
                    .tail = call.tail,
                } },
                .dest = dest,
                .type = result_type,
            };
        }
    }
}

test "rewriteLocalCallsToIndirect: a call to a local function becomes ref_func + call_ref" {
    const allocator = std.testing.allocator;

    var func = ir.IrFunction.init(allocator, 1, 1, 1);
    defer func.deinit();
    const b0 = try func.newBlock();
    const args = try allocator.dupe(ir.VReg, &.{0});
    const dest = func.newVReg(); // vreg 1
    try func.getBlock(b0).append(.{
        .op = .{ .call = .{ .func_idx = 3, .args = args, .extra_results = 0, .tail = false } },
        .dest = dest,
        .type = .i32,
    });

    const func_type_indices = [_]u32{ 0, 0, 0, 7 };
    try rewriteLocalCallsToIndirect(&func, 1, &func_type_indices);

    const insts = func.getBlock(b0).instructions.items;
    try std.testing.expectEqual(@as(usize, 2), insts.len);

    try std.testing.expectEqual(ir.Inst.Op{ .ref_func = 3 }, insts[0].op);
    try std.testing.expectEqual(@as(ir.IrType, .i64), insts[0].type);
    const ref_dest = insts[0].dest.?;

    switch (insts[1].op) {
        .call_ref => |cr| {
            try std.testing.expectEqual(@as(u32, 7), cr.type_idx);
            try std.testing.expectEqual(ref_dest, cr.func_ref);
            try std.testing.expectEqual(@as(usize, 1), cr.args.len);
            try std.testing.expectEqual(@as(ir.VReg, 0), cr.args[0]);
            try std.testing.expectEqual(@as(u8, 0), cr.extra_results);
            try std.testing.expectEqual(false, cr.tail);
        },
        else => return error.TestUnexpectedResult,
    }
    try std.testing.expectEqual(dest, insts[1].dest.?);
    try std.testing.expectEqual(@as(ir.IrType, .i32), insts[1].type);
}

test "rewriteLocalCallsToIndirect: an import call is left untouched" {
    const allocator = std.testing.allocator;

    var func = ir.IrFunction.init(allocator, 1, 1, 1);
    defer func.deinit();
    const b0 = try func.newBlock();
    const args = try allocator.dupe(ir.VReg, &.{0});
    try func.getBlock(b0).append(.{
        .op = .{ .call = .{ .func_idx = 0, .args = args, .extra_results = 0, .tail = false } },
        .dest = func.newVReg(),
        .type = .i32,
    });

    try rewriteLocalCallsToIndirect(&func, 2, &.{});

    const insts = func.getBlock(b0).instructions.items;
    try std.testing.expectEqual(@as(usize, 1), insts.len);
    switch (insts[0].op) {
        .call => |c| try std.testing.expectEqual(@as(u32, 0), c.func_idx),
        else => return error.TestUnexpectedResult,
    }
}

test "rewriteLocalCallsToIndirect: preserves tail flag and multiple calls in one block" {
    const allocator = std.testing.allocator;

    var func = ir.IrFunction.init(allocator, 1, 1, 1);
    defer func.deinit();
    const b0 = try func.newBlock();
    try func.getBlock(b0).append(.{
        .op = .{ .call = .{ .func_idx = 1, .args = &.{}, .extra_results = 0, .tail = false } },
        .dest = func.newVReg(),
        .type = .i32,
    });
    try func.getBlock(b0).append(.{
        .op = .{ .call = .{ .func_idx = 2, .args = &.{}, .extra_results = 0, .tail = true } },
        .dest = null,
        .type = .i32,
    });

    try rewriteLocalCallsToIndirect(&func, 0, &.{ 5, 6, 8 });

    const insts = func.getBlock(b0).instructions.items;
    // Each of the 2 original `.call`s becomes a `.ref_func` + `.call_ref`
    // pair, so 2 calls -> 4 instructions.
    try std.testing.expectEqual(@as(usize, 4), insts.len);

    try std.testing.expectEqual(ir.Inst.Op{ .ref_func = 1 }, insts[0].op);
    switch (insts[1].op) {
        .call_ref => |cr| {
            try std.testing.expectEqual(@as(u32, 6), cr.type_idx);
            try std.testing.expectEqual(false, cr.tail);
        },
        else => return error.TestUnexpectedResult,
    }

    try std.testing.expectEqual(ir.Inst.Op{ .ref_func = 2 }, insts[2].op);
    switch (insts[3].op) {
        .call_ref => |cr| {
            try std.testing.expectEqual(@as(u32, 8), cr.type_idx);
            try std.testing.expectEqual(true, cr.tail); // tail flag preserved
        },
        else => return error.TestUnexpectedResult,
    }
}
