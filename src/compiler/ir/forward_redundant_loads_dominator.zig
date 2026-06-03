//! Cross-block / dominator-aware redundant-load forwarder (#391).
//!
//! Promotes the single-block `forwardRedundantLoads` pass to a
//! dominator-tree walk. The algorithm (per #391):
//!
//!   1. DFS the dominator tree from the entry block.
//!   2. On entry to each block, push a fresh `(LoadKey -> VReg)` map.
//!   3. While scanning instructions, lookups walk the stack from
//!      innermost to outermost frame. A hit means an ancestor (or this
//!      block) already loaded the same `(base, offset, size, sign_extend)`
//!      and no aliasing store / call / barrier has run since.
//!   4. Stores invalidate aliasing entries in every active frame (a
//!      store anywhere on a dominator path kills cached values seen by
//!      dominated blocks). Calls and barriers clear every frame fully.
//!   5. On exit from a block, pop its frame.
//!
//! Invalidation rules mirror `forward_redundant_loads.zig`; the `LoadKey`
//! type lives in `alias_class.zig` and is managed by the shared driver.
//!
//! Sibling-block invariance: stores in one branch of a diamond do not
//! propagate up into the dominator's frame (we only invalidate frames
//! that are on the current DFS path), so the unrelated sibling-then-
//! merge block still observes the dominator's load.
//!
//! Sibling-block BARRIER soundness (#719): a barrier (call / atomic /
//! bulk-memory) on one diamond branch DOES need to wipe ancestor
//! cached loads, because a merge block dominated by the diamond head
//! is reachable via the barrier-containing branch on some execution
//! path. We achieve this by walking children through
//! `passes.BarrierOrderedDomChildren`, which schedules barrier-containing
//! subtrees to be DFS-visited before their non-barrier siblings — so the
//! barrier's `clearAll` takes effect on ancestor frames before any non-barrier
//! subtree gets a chance to forward stale loads into a downstream merge.

const std = @import("std");
const ir = @import("ir.zig");
const passes = @import("passes.zig");
pub fn forwardRedundantLoadsDominator(
    func: *ir.IrFunction,
    allocator: std.mem.Allocator,
) !bool {
    const Visitor = struct {
        pub const forward_destless_loads = true;

        pub fn onInstruction(
            _: *@This(),
            _: *ir.IrFunction,
            _: ir.BlockId,
            _: usize,
            _: *ir.Inst,
            _: anytype,
        ) !passes.LoadForwardingDomWalkInstructionResult {
            return .unchanged;
        }
    };

    var visitor = Visitor{};
    return try passes.LoadForwardingDomWalk(Visitor).run(&visitor, func, allocator);
}

// ── Tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

/// Append a `br` terminator wiring `block.successors`-equivalent via the
/// only mechanism the IR actually uses: terminator ops.
fn terminateBr(blk: *ir.BasicBlock, target: ir.BlockId) !void {
    try blk.append(.{ .op = .{ .br = target }, .type = .i32 });
}

fn terminateBrIf(blk: *ir.BasicBlock, cond: ir.VReg, t: ir.BlockId, e: ir.BlockId) !void {
    try blk.append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = t, .else_block = e } }, .type = .i32 });
}

const FrlTestFuncMeta = struct {
    params: u32 = 0,
    results: u32 = 0,
    locals: u32 = 0,
};

fn makeFrlTestFunc(allocator: std.mem.Allocator, meta: FrlTestFuncMeta) ir.IrFunction {
    return ir.IrFunction.init(allocator, meta.params, meta.results, meta.locals);
}

fn expectFrlTestFuncMeta(func: *const ir.IrFunction, meta: FrlTestFuncMeta) !void {
    try testing.expectEqual(meta.params, func.param_count);
    try testing.expectEqual(meta.results, func.result_count);
    try testing.expectEqual(meta.locals, func.local_count);
}

test "dominator FRL: forwards through linear chain (entry → mid → tail)" {
    const allocator = testing.allocator;
    const meta = FrlTestFuncMeta{};
    var func = makeFrlTestFunc(allocator, meta);
    defer func.deinit();
    try expectFrlTestFuncMeta(&func, meta);

    const entry = try func.newBlock();
    const mid = try func.newBlock();
    const tail = try func.newBlock();

    const v_base = func.newVReg();
    const v_load1 = func.newVReg();
    const v_load2 = func.newVReg();

    {
        const blk = &func.blocks.items[entry];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_load1, .type = .i32 });
        try terminateBr(blk, mid);
    }
    {
        const blk = &func.blocks.items[mid];
        try terminateBr(blk, tail);
    }
    {
        const blk = &func.blocks.items[tail];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_load2, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = null }, .type = .i32 });
    }

    const changed = try forwardRedundantLoadsDominator(&func, allocator);
    try testing.expect(changed);

    // Tail's redundant load should be gone; only `ret` remains.
    const tail_blk = &func.blocks.items[tail];
    try testing.expectEqual(@as(usize, 1), tail_blk.instructions.items.len);
    try testing.expect(tail_blk.instructions.items[0].op == .ret);
}

test "dominator FRL: diamond merge — load forwarded from dominator" {
    const allocator = testing.allocator;
    const meta = FrlTestFuncMeta{};
    var func = makeFrlTestFunc(allocator, meta);
    defer func.deinit();
    try expectFrlTestFuncMeta(&func, meta);

    const entry = try func.newBlock();
    const left = try func.newBlock();
    const right = try func.newBlock();
    const merge = try func.newBlock();

    const v_base = func.newVReg();
    const v_cond = func.newVReg();
    const v_dom = func.newVReg();
    const v_merge_load = func.newVReg();

    {
        const blk = &func.blocks.items[entry];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_dom, .type = .i32 });
        try terminateBrIf(blk, v_cond, left, right);
    }
    {
        const blk = &func.blocks.items[left];
        try terminateBr(blk, merge);
    }
    {
        const blk = &func.blocks.items[right];
        try terminateBr(blk, merge);
    }
    {
        const blk = &func.blocks.items[merge];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_merge_load, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = null }, .type = .i32 });
    }

    const changed = try forwardRedundantLoadsDominator(&func, allocator);
    try testing.expect(changed);
    const m = &func.blocks.items[merge];
    try testing.expectEqual(@as(usize, 1), m.instructions.items.len);
    try testing.expect(m.instructions.items[0].op == .ret);
}

test "dominator FRL: sibling-block store does NOT invalidate dominator-cached value" {
    // entry: load v_dom@base; br_if to left/right
    // left:  store @base (aliases v_dom) → would invalidate its own frame,
    //        but must NOT touch the entry frame so `right` still forwards.
    // right: load @base (should be forwarded from entry's v_dom).
    const allocator = testing.allocator;
    const meta = FrlTestFuncMeta{};
    var func = makeFrlTestFunc(allocator, meta);
    defer func.deinit();
    try expectFrlTestFuncMeta(&func, meta);

    const entry = try func.newBlock();
    const left = try func.newBlock();
    const right = try func.newBlock();

    const v_base = func.newVReg();
    const v_cond = func.newVReg();
    const v_dom = func.newVReg();
    const v_store_val = func.newVReg();
    const v_right_load = func.newVReg();

    {
        const blk = &func.blocks.items[entry];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_dom, .type = .i32 });
        try terminateBrIf(blk, v_cond, left, right);
    }
    {
        const blk = &func.blocks.items[left];
        try blk.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v_store_val } }, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = null }, .type = .i32 });
    }
    {
        const blk = &func.blocks.items[right];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_right_load, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = null }, .type = .i32 });
    }

    const changed = try forwardRedundantLoadsDominator(&func, allocator);
    try testing.expect(changed);

    // Right block's load forwarded; only `ret` remains.
    const r = &func.blocks.items[right];
    try testing.expectEqual(@as(usize, 1), r.instructions.items.len);
    try testing.expect(r.instructions.items[0].op == .ret);

    // Left block's store survives.
    const l = &func.blocks.items[left];
    try testing.expect(l.instructions.items[0].op == .store);
}

test "dominator FRL: call between dominator-load and dominated-load clears scope" {
    const allocator = testing.allocator;
    const meta = FrlTestFuncMeta{};
    var func = makeFrlTestFunc(allocator, meta);
    defer func.deinit();
    try expectFrlTestFuncMeta(&func, meta);

    const entry = try func.newBlock();
    const tail = try func.newBlock();

    const v_base = func.newVReg();
    const v_dom = func.newVReg();
    const v_call_res = func.newVReg();
    const v_tail_load = func.newVReg();

    {
        const blk = &func.blocks.items[entry];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_dom, .type = .i32 });
        try blk.append(.{ .op = .{ .call = .{ .func_idx = 0 } }, .dest = v_call_res, .type = .i32 });
        try terminateBr(blk, tail);
    }
    {
        const blk = &func.blocks.items[tail];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_tail_load, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = null }, .type = .i32 });
    }

    const changed = try forwardRedundantLoadsDominator(&func, allocator);
    try testing.expect(!changed);
    // Tail's load is preserved (still 2 insts including ret).
    const t = &func.blocks.items[tail];
    try testing.expectEqual(@as(usize, 2), t.instructions.items.len);
    try testing.expect(t.instructions.items[0].op == .load);
}

test "dominator FRL: call in sibling branch invalidates merge-block forwarding (#719)" {
    // entry:  v_dom = load p[0]; br_if cond, sib, tail
    // sib:    call f; br tail
    // tail:   v_tail = load p[0]; ret
    //
    // idom(tail) = entry. The path entry → sib → tail crosses a
    // barrier, so the load in `tail` must NOT be forwarded from `v_dom`.
    //
    // Regression test mirroring the GVN-side bug (issue #719): without
    // dom-DFS reordering, `tail` is popped before `sib` and forwards
    // the stale ancestor value.
    const allocator = testing.allocator;
    const meta = FrlTestFuncMeta{};
    var func = makeFrlTestFunc(allocator, meta);
    defer func.deinit();
    try expectFrlTestFuncMeta(&func, meta);

    const entry = try func.newBlock();
    const sib = try func.newBlock();
    const tail = try func.newBlock();

    const v_base = func.newVReg();
    const cond = func.newVReg();
    const v_dom = func.newVReg();
    const v_call = func.newVReg();
    const v_tail = func.newVReg();

    {
        const blk = &func.blocks.items[entry];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_dom, .type = .i32 });
        try terminateBrIf(blk, cond, sib, tail);
    }
    {
        const blk = &func.blocks.items[sib];
        try blk.append(.{ .op = .{ .call = .{ .func_idx = 0 } }, .dest = v_call, .type = .i32 });
        try terminateBr(blk, tail);
    }
    {
        const blk = &func.blocks.items[tail];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_tail, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = null }, .type = .i32 });
    }

    _ = try forwardRedundantLoadsDominator(&func, allocator);
    const t = &func.blocks.items[tail];
    try testing.expectEqual(@as(usize, 2), t.instructions.items.len);
    try testing.expect(t.instructions.items[0].op == .load);
}

test "FRLDominator (cell C): call AFTER load in same block prevents later forwarding (#734)" {
    // entry: load p[0]; call barrier; load p[0]; ret second load.
    // FRL deletes fused loads, so both load instructions must remain.
    const allocator = testing.allocator;
    const meta = FrlTestFuncMeta{};
    var func = makeFrlTestFunc(allocator, meta);
    defer func.deinit();
    try expectFrlTestFuncMeta(&func, meta);

    const entry = try func.newBlock();
    const v_base = func.newVReg();
    const v_first = func.newVReg();
    const v_call = func.newVReg();
    const v_second = func.newVReg();

    const blk = &func.blocks.items[entry];
    try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_first, .type = .i32 });
    try blk.append(.{ .op = .{ .call = .{ .func_idx = 0 } }, .dest = v_call, .type = .i32 });
    try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_second, .type = .i32 });
    try blk.append(.{ .op = .{ .ret = v_second }, .type = .i32 });

    const changed = try forwardRedundantLoadsDominator(&func, allocator);
    try testing.expect(!changed);
    try testing.expectEqual(@as(usize, 4), blk.instructions.items.len);
    try testing.expect(blk.instructions.items[0].op == .load);
    try testing.expect(blk.instructions.items[2].op == .load);
    try testing.expectEqual(ir.Inst.Op{ .ret = v_second }, blk.instructions.items[3].op);
}

test "FRLDominator (cell F): call in sibling SUBTREE invalidates merge forwarding (#719,#734)" {
    // entry branches directly to tail or to a sibling subtree containing a call.
    // The merge load is reachable through that deeper barrier subtree.
    const allocator = testing.allocator;
    const meta = FrlTestFuncMeta{};
    var func = makeFrlTestFunc(allocator, meta);
    defer func.deinit();
    try expectFrlTestFuncMeta(&func, meta);

    const entry = try func.newBlock();
    const sib_top = try func.newBlock();
    const sib_call = try func.newBlock();
    const tail = try func.newBlock();

    const v_base = func.newVReg();
    const cond = func.newVReg();
    const v_dom = func.newVReg();
    const v_call = func.newVReg();
    const v_tail = func.newVReg();

    {
        const blk = &func.blocks.items[entry];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_dom, .type = .i32 });
        try terminateBrIf(blk, cond, sib_top, tail);
    }
    {
        const blk = &func.blocks.items[sib_top];
        try terminateBr(blk, sib_call);
    }
    {
        const blk = &func.blocks.items[sib_call];
        try blk.append(.{ .op = .{ .call = .{ .func_idx = 0 } }, .dest = v_call, .type = .i32 });
        try terminateBr(blk, tail);
    }
    {
        const blk = &func.blocks.items[tail];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_tail, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = v_tail }, .type = .i32 });
    }

    _ = try forwardRedundantLoadsDominator(&func, allocator);
    const t = &func.blocks.items[tail];
    try testing.expectEqual(@as(usize, 2), t.instructions.items.len);
    try testing.expect(t.instructions.items[0].op == .load);
    try testing.expectEqual(ir.Inst.Op{ .ret = v_tail }, t.instructions.items[1].op);
}

test "FRLDominator (cell G): call on loop back-edge invalidates dominated body forwarding (#734)" {
    // entry loads, header may take a call-bearing latch back-edge, then body loads.
    // The body load is reachable after the back-edge barrier and must survive.
    const allocator = testing.allocator;
    const meta = FrlTestFuncMeta{};
    var func = makeFrlTestFunc(allocator, meta);
    defer func.deinit();
    try expectFrlTestFuncMeta(&func, meta);

    const entry = try func.newBlock();
    const header = try func.newBlock();
    const latch = try func.newBlock();
    const body = try func.newBlock();

    const v_base = func.newVReg();
    const cond = func.newVReg();
    const v_dom = func.newVReg();
    const v_call = func.newVReg();
    const v_body = func.newVReg();

    {
        const blk = &func.blocks.items[entry];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_dom, .type = .i32 });
        try terminateBr(blk, header);
    }
    {
        const blk = &func.blocks.items[header];
        try terminateBrIf(blk, cond, latch, body);
    }
    {
        const blk = &func.blocks.items[latch];
        try blk.append(.{ .op = .{ .call = .{ .func_idx = 0 } }, .dest = v_call, .type = .i32 });
        try terminateBr(blk, header);
    }
    {
        const blk = &func.blocks.items[body];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_body, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = v_body }, .type = .i32 });
    }

    _ = try forwardRedundantLoadsDominator(&func, allocator);
    const b = &func.blocks.items[body];
    try testing.expectEqual(@as(usize, 2), b.instructions.items.len);
    try testing.expect(b.instructions.items[0].op == .load);
    try testing.expectEqual(ir.Inst.Op{ .ret = v_body }, b.instructions.items[1].op);
}

test "FRLDominator (cell H): barrier in then-branch, store in else-branch, merge load not fused (#734)" {
    // entry branches to a call in then or an aliasing store in else, then merges.
    // The merge load is reachable through the barrier branch and must survive.
    const allocator = testing.allocator;
    const meta = FrlTestFuncMeta{};
    var func = makeFrlTestFunc(allocator, meta);
    defer func.deinit();
    try expectFrlTestFuncMeta(&func, meta);

    const entry = try func.newBlock();
    const then_blk = try func.newBlock();
    const else_blk = try func.newBlock();
    const merge = try func.newBlock();

    const v_base = func.newVReg();
    const cond = func.newVReg();
    const v_dom = func.newVReg();
    const v_call = func.newVReg();
    const v_store_val = func.newVReg();
    const v_merge = func.newVReg();

    {
        const blk = &func.blocks.items[entry];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_dom, .type = .i32 });
        try terminateBrIf(blk, cond, then_blk, else_blk);
    }
    {
        const blk = &func.blocks.items[then_blk];
        try blk.append(.{ .op = .{ .call = .{ .func_idx = 0 } }, .dest = v_call, .type = .i32 });
        try terminateBr(blk, merge);
    }
    {
        const blk = &func.blocks.items[else_blk];
        try blk.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v_store_val } }, .type = .i32 });
        try terminateBr(blk, merge);
    }
    {
        const blk = &func.blocks.items[merge];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_merge, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = v_merge }, .type = .i32 });
    }

    _ = try forwardRedundantLoadsDominator(&func, allocator);
    const m = &func.blocks.items[merge];
    try testing.expectEqual(@as(usize, 2), m.instructions.items.len);
    try testing.expect(m.instructions.items[0].op == .load);
    try testing.expectEqual(ir.Inst.Op{ .ret = v_merge }, m.instructions.items[1].op);
}
