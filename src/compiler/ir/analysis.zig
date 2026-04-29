//! IR analysis passes for register allocation.
//!
//! Provides CFG construction, liveness analysis, and live range computation
//! needed by the linear scan register allocator.

const std = @import("std");
const ir = @import("ir.zig");

// ── CFG: Successor computation ──────────────────────────────────────────

/// Compute successor block IDs for each block by scanning branch instructions.
pub fn buildSuccessors(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) !std.AutoHashMap(ir.BlockId, []const ir.BlockId) {
    var successors = std.AutoHashMap(ir.BlockId, []const ir.BlockId).init(allocator);

    for (func.blocks.items, 0..) |block, idx| {
        var succs: std.ArrayList(ir.BlockId) = .empty;
        for (block.instructions.items) |inst| {
            switch (inst.op) {
                .br => |target| try succs.append(allocator, target),
                .br_if => |bi| {
                    try succs.append(allocator, bi.then_block);
                    try succs.append(allocator, bi.else_block);
                },
                .br_table => |bt| {
                    for (bt.targets) |t| try succs.append(allocator, t);
                    try succs.append(allocator, bt.default);
                },
                else => {},
            }
        }
        try successors.put(@intCast(idx), try succs.toOwnedSlice(allocator));
    }
    return successors;
}

// ── Liveness analysis ───────────────────────────────────────────────────

/// Per-block liveness sets.
pub const BlockLiveness = struct {
    /// VRegs live at the start of the block.
    live_in: std.AutoHashMap(ir.VReg, void),
    /// VRegs live at the end of the block.
    live_out: std.AutoHashMap(ir.VReg, void),
};

/// Compute liveness information for all blocks using backward dataflow analysis.
/// Returns a map from BlockId to BlockLiveness.
pub fn computeLiveness(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) !std.AutoHashMap(ir.BlockId, BlockLiveness) {
    const successors = try buildSuccessors(func, allocator);
    defer {
        var it = successors.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        @constCast(&successors).deinit();
    }

    var liveness = std.AutoHashMap(ir.BlockId, BlockLiveness).init(allocator);
    for (0..func.blocks.items.len) |idx| {
        try liveness.put(@intCast(idx), .{
            .live_in = std.AutoHashMap(ir.VReg, void).init(allocator),
            .live_out = std.AutoHashMap(ir.VReg, void).init(allocator),
        });
    }

    // Fixed-point iteration
    var changed = true;
    while (changed) {
        changed = false;

        // Process blocks in reverse order
        var block_idx: usize = func.blocks.items.len;
        while (block_idx > 0) {
            block_idx -= 1;
            const bid: ir.BlockId = @intCast(block_idx);
            const block = &func.blocks.items[block_idx];
            const bl = liveness.getPtr(bid).?;

            // live_out = ∪ live_in[succ]
            if (successors.get(bid)) |succs| {
                for (succs) |succ_id| {
                    if (liveness.getPtr(succ_id)) |succ_bl| {
                        var sit = succ_bl.live_in.iterator();
                        while (sit.next()) |entry| {
                            const result = try bl.live_out.getOrPut(entry.key_ptr.*);
                            if (!result.found_existing) {
                                result.value_ptr.* = {};
                                changed = true;
                            }
                        }
                    }
                }
            }

            // live_in = use[B] ∪ (live_out[B] - def[B])
            // Start with live_out, remove defs, add uses (backward through instructions)
            var live = std.AutoHashMap(ir.VReg, void).init(allocator);
            defer live.deinit();
            // Copy live_out into working set
            var lit = bl.live_out.iterator();
            while (lit.next()) |entry| try live.put(entry.key_ptr.*, {});

            // Walk instructions backward
            var inst_idx: usize = block.instructions.items.len;
            while (inst_idx > 0) {
                inst_idx -= 1;
                const inst = block.instructions.items[inst_idx];
                // Remove def
                if (inst.dest) |dest| _ = live.remove(dest);
                // Add uses
                addInstUses(&live, inst);
            }

            // Update live_in if changed
            var wit = live.iterator();
            while (wit.next()) |entry| {
                const result = try bl.live_in.getOrPut(entry.key_ptr.*);
                if (!result.found_existing) {
                    result.value_ptr.* = {};
                    changed = true;
                }
            }
        }
    }

    return liveness;
}

/// Add all VReg uses of an instruction to a live set.
fn addInstUses(live: *std.AutoHashMap(ir.VReg, void), inst: ir.Inst) void {
    switch (inst.op) {
        .iconst_32, .iconst_64, .fconst_32, .fconst_64 => {},
        .local_get, .global_get => {},
        .br, .@"unreachable", .atomic_fence => {},

        .add, .sub, .mul, .div_s, .div_u, .rem_s, .rem_u,
        .@"and", .@"or", .xor, .shl, .shr_s, .shr_u, .rotl, .rotr,
        .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u,
        .f_min, .f_max, .f_copysign,
        .f_eq, .f_ne, .f_lt, .f_gt, .f_le, .f_ge,
        => |bin| {
            live.put(bin.lhs, {}) catch {};
            live.put(bin.rhs, {}) catch {};
        },

        .clz, .ctz, .popcnt, .eqz, .wrap_i64, .extend_i32_s, .extend_i32_u,
        .extend8_s, .extend16_s, .extend32_s,
        .f_neg, .f_abs, .f_sqrt, .f_ceil, .f_floor, .f_trunc, .f_nearest,
        .trunc_f32_s, .trunc_f32_u, .trunc_f64_s, .trunc_f64_u,
        .convert_s, .convert_u, .convert_i32_s, .convert_i64_s, .convert_i32_u, .convert_i64_u, .demote_f64, .promote_f32, .reinterpret,
        .trunc_sat_f32_s, .trunc_sat_f32_u, .trunc_sat_f64_s, .trunc_sat_f64_u,
        => |vreg| live.put(vreg, {}) catch {},

        .local_set => |ls| live.put(ls.val, {}) catch {},
        .global_set => |gs| live.put(gs.val, {}) catch {},
        .load => |ld| live.put(ld.base, {}) catch {},
        .store => |st| {
            live.put(st.base, {}) catch {};
            live.put(st.val, {}) catch {};
        },
        .br_if => |bi| live.put(bi.cond, {}) catch {},
        .br_table => |bt| live.put(bt.index, {}) catch {},
        .ret => |maybe_vreg| if (maybe_vreg) |v| live.put(v, {}) catch {},
        .ret_multi => |vregs| {
            for (vregs) |v| live.put(v, {}) catch {};
        },
        .call_result => {},
        .call => |cl| {
            for (cl.args) |arg| live.put(arg, {}) catch {};
        },
        .call_indirect => |ci| {
            live.put(ci.elem_idx, {}) catch {};
            for (ci.args) |arg| live.put(arg, {}) catch {};
        },
        .call_ref => |cr| {
            live.put(cr.func_ref, {}) catch {};
            for (cr.args) |arg| live.put(arg, {}) catch {};
        },
        .select => |sel| {
            live.put(sel.cond, {}) catch {};
            live.put(sel.if_true, {}) catch {};
            live.put(sel.if_false, {}) catch {};
        },

        .atomic_load => |al| live.put(al.base, {}) catch {},
        .atomic_store => |ast| {
            live.put(ast.base, {}) catch {};
            live.put(ast.val, {}) catch {};
        },
        .atomic_rmw => |ar| {
            live.put(ar.base, {}) catch {};
            live.put(ar.val, {}) catch {};
        },
        .atomic_cmpxchg => |ac| {
            live.put(ac.base, {}) catch {};
            live.put(ac.expected, {}) catch {};
            live.put(ac.replacement, {}) catch {};
        },
        .atomic_notify => |an| {
            live.put(an.base, {}) catch {};
            live.put(an.count, {}) catch {};
        },
        .atomic_wait => |aw| {
            live.put(aw.base, {}) catch {};
            live.put(aw.expected, {}) catch {};
            live.put(aw.timeout, {}) catch {};
        },
        .memory_copy => |mc| {
            live.put(mc.dst, {}) catch {};
            live.put(mc.src, {}) catch {};
            live.put(mc.len, {}) catch {};
        },
        .memory_fill => |mf| {
            live.put(mf.dst, {}) catch {};
            live.put(mf.val, {}) catch {};
            live.put(mf.len, {}) catch {};
        },
        .memory_size => {},
        .memory_grow => |pages| {
            live.put(pages, {}) catch {};
        },
        .table_size => {},
        .table_get => |tg| {
            live.put(tg.idx, {}) catch {};
        },
        .table_set => |ts| {
            live.put(ts.idx, {}) catch {};
            live.put(ts.val, {}) catch {};
        },
        .table_grow => |tg| {
            live.put(tg.init, {}) catch {};
            live.put(tg.delta, {}) catch {};
        },
        .ref_func => {},
        .memory_init => |mi| {
            live.put(mi.dst, {}) catch {};
            live.put(mi.src, {}) catch {};
            live.put(mi.len, {}) catch {};
        },
        .data_drop => {},
        .table_init => |ti| {
            live.put(ti.dst, {}) catch {};
            live.put(ti.src, {}) catch {};
            live.put(ti.len, {}) catch {};
        },
        .elem_drop => {},
        .phi => |edges| {
            for (edges) |edge| live.put(edge.val, {}) catch {};
        },
    }
}

/// A live range interval for a VReg.
pub const LiveRange = struct {
    vreg: ir.VReg,
    start: u32, // global instruction index of definition
    end: u32, // global instruction index of last use
};

/// Compute live ranges for all VRegs in a function.
/// Uses global instruction numbering across all blocks.
pub fn computeLiveRanges(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) ![]LiveRange {
    return computeLiveRangesWithOrder(func, null, allocator);
}

/// Compute live ranges with instruction numbering following `block_order`.
/// When provided, this MUST match the codegen emission order so that the
/// register allocator's interval arithmetic is consistent with actual
/// code layout.
pub fn computeLiveRangesWithOrder(
    func: *const ir.IrFunction,
    block_order: ?[]const ir.BlockId,
    allocator: std.mem.Allocator,
) ![]LiveRange {
    const liveness = try computeLiveness(func, allocator);
    defer {
        var it = @constCast(&liveness).iterator();
        while (it.next()) |entry| {
            entry.value_ptr.live_in.deinit();
            entry.value_ptr.live_out.deinit();
        }
        @constCast(&liveness).deinit();
    }

    // Global instruction numbering — follows block_order if provided.
    var def_pos = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer def_pos.deinit();
    var last_use_pos = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer last_use_pos.deinit();

    // Build default sequential order if none provided.
    const nblocks = func.blocks.items.len;
    var owns_order = false;
    const effective_order: []const ir.BlockId = if (block_order) |bo| bo else blk: {
        const raw = try allocator.alloc(ir.BlockId, nblocks);
        for (raw, 0..) |*r, i| r.* = @intCast(i);
        owns_order = true;
        break :blk raw;
    };
    defer if (owns_order) allocator.free(effective_order);

    var global_idx: u32 = 0;
    for (effective_order) |bid| {
        const block = func.blocks.items[bid];

        // VRegs in live_in are used before defined in this block — extend their range
        if (liveness.getPtr(bid)) |bl| {
            var lit = bl.live_in.iterator();
            while (lit.next()) |entry| {
                const vreg = entry.key_ptr.*;
                // Extend last use to at least this block's start
                const existing = last_use_pos.get(vreg) orelse 0;
                try last_use_pos.put(vreg, @max(existing, global_idx));
            }
        }

        for (block.instructions.items) |inst| {
            // Record definition position
            if (inst.dest) |dest| {
                if (!def_pos.contains(dest)) {
                    try def_pos.put(dest, global_idx);
                }
            }
            // Record last use position
            updateLastUse(&last_use_pos, inst, global_idx);
            global_idx += 1;
        }

        // VRegs in live_out extend to end of block
        if (liveness.getPtr(bid)) |bl| {
            var lit = bl.live_out.iterator();
            while (lit.next()) |entry| {
                const vreg = entry.key_ptr.*;
                const existing = last_use_pos.get(vreg) orelse 0;
                try last_use_pos.put(vreg, @max(existing, global_idx -| 1));
            }
        }
    }

    // Build sorted live ranges
    var ranges: std.ArrayList(LiveRange) = .empty;
    var dit = def_pos.iterator();
    while (dit.next()) |entry| {
        const vreg = entry.key_ptr.*;
        const start = entry.value_ptr.*;
        const end = last_use_pos.get(vreg) orelse start;
        try ranges.append(allocator, .{ .vreg = vreg, .start = start, .end = @max(start, end) });
    }

    // Sort by start position
    std.mem.sort(LiveRange, ranges.items, {}, struct {
        fn lessThan(_: void, a: LiveRange, b: LiveRange) bool {
            return a.start < b.start;
        }
    }.lessThan);

    return try ranges.toOwnedSlice(allocator);
}

fn updateLastUse(last_use: *std.AutoHashMap(ir.VReg, u32), inst: ir.Inst, pos: u32) void {
    switch (inst.op) {
        .iconst_32, .iconst_64, .fconst_32, .fconst_64 => {},
        .local_get, .global_get => {},
        .br, .@"unreachable", .atomic_fence => {},

        .add, .sub, .mul, .div_s, .div_u, .rem_s, .rem_u,
        .@"and", .@"or", .xor, .shl, .shr_s, .shr_u, .rotl, .rotr,
        .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u,
        .f_min, .f_max, .f_copysign,
        .f_eq, .f_ne, .f_lt, .f_gt, .f_le, .f_ge,
        => |bin| {
            last_use.put(bin.lhs, pos) catch {};
            last_use.put(bin.rhs, pos) catch {};
        },

        .clz, .ctz, .popcnt, .eqz, .wrap_i64, .extend_i32_s, .extend_i32_u,
        .extend8_s, .extend16_s, .extend32_s,
        .f_neg, .f_abs, .f_sqrt, .f_ceil, .f_floor, .f_trunc, .f_nearest,
        .trunc_f32_s, .trunc_f32_u, .trunc_f64_s, .trunc_f64_u,
        .convert_s, .convert_u, .convert_i32_s, .convert_i64_s, .convert_i32_u, .convert_i64_u, .demote_f64, .promote_f32, .reinterpret,
        .trunc_sat_f32_s, .trunc_sat_f32_u, .trunc_sat_f64_s, .trunc_sat_f64_u,
        => |vreg| last_use.put(vreg, pos) catch {},

        .local_set => |ls| last_use.put(ls.val, pos) catch {},
        .global_set => |gs| last_use.put(gs.val, pos) catch {},
        .load => |ld| last_use.put(ld.base, pos) catch {},
        .store => |st| {
            last_use.put(st.base, pos) catch {};
            last_use.put(st.val, pos) catch {};
        },
        .br_if => |bi| last_use.put(bi.cond, pos) catch {},
        .br_table => |bt| last_use.put(bt.index, pos) catch {},
        .ret => |maybe_vreg| if (maybe_vreg) |v| last_use.put(v, pos) catch {},
        .ret_multi => |vregs| {
            for (vregs) |v| last_use.put(v, pos) catch {};
        },
        .call_result => {},
        .call => |cl| {
            for (cl.args) |arg| last_use.put(arg, pos) catch {};
        },
        .call_indirect => |ci| {
            last_use.put(ci.elem_idx, pos) catch {};
            for (ci.args) |arg| last_use.put(arg, pos) catch {};
        },
        .call_ref => |cr| {
            last_use.put(cr.func_ref, pos) catch {};
            for (cr.args) |arg| last_use.put(arg, pos) catch {};
        },
        .select => |sel| {
            last_use.put(sel.cond, pos) catch {};
            last_use.put(sel.if_true, pos) catch {};
            last_use.put(sel.if_false, pos) catch {};
        },

        .atomic_load => |al| last_use.put(al.base, pos) catch {},
        .atomic_store => |ast| {
            last_use.put(ast.base, pos) catch {};
            last_use.put(ast.val, pos) catch {};
        },
        .atomic_rmw => |ar| {
            last_use.put(ar.base, pos) catch {};
            last_use.put(ar.val, pos) catch {};
        },
        .atomic_cmpxchg => |ac| {
            last_use.put(ac.base, pos) catch {};
            last_use.put(ac.expected, pos) catch {};
            last_use.put(ac.replacement, pos) catch {};
        },
        .atomic_notify => |an| {
            last_use.put(an.base, pos) catch {};
            last_use.put(an.count, pos) catch {};
        },
        .atomic_wait => |aw| {
            last_use.put(aw.base, pos) catch {};
            last_use.put(aw.expected, pos) catch {};
            last_use.put(aw.timeout, pos) catch {};
        },
        .memory_copy => |mc| {
            last_use.put(mc.dst, pos) catch {};
            last_use.put(mc.src, pos) catch {};
            last_use.put(mc.len, pos) catch {};
        },
        .memory_fill => |mf| {
            last_use.put(mf.dst, pos) catch {};
            last_use.put(mf.val, pos) catch {};
            last_use.put(mf.len, pos) catch {};
        },
        .memory_size => {},
        .memory_grow => |pages| {
            last_use.put(pages, pos) catch {};
        },
        .table_size => {},
        .table_get => |tg| {
            last_use.put(tg.idx, pos) catch {};
        },
        .table_set => |ts| {
            last_use.put(ts.idx, pos) catch {};
            last_use.put(ts.val, pos) catch {};
        },
        .table_grow => |tg| {
            last_use.put(tg.init, pos) catch {};
            last_use.put(tg.delta, pos) catch {};
        },
        .ref_func => {},
        .memory_init => |mi| {
            last_use.put(mi.dst, pos) catch {};
            last_use.put(mi.src, pos) catch {};
            last_use.put(mi.len, pos) catch {};
        },
        .data_drop => {},
        .table_init => |ti| {
            last_use.put(ti.dst, pos) catch {};
            last_use.put(ti.src, pos) catch {};
            last_use.put(ti.len, pos) catch {};
        },
        .elem_drop => {},
        .phi => |edges| {
            for (edges) |edge| last_use.put(edge.val, pos) catch {};
        },
    }
}

// ── Dominator tree (Cooper-Harvey-Kennedy) ──────────────────────────────

/// Immediate-dominator tree for the function's CFG, rooted at the entry
/// block (block 0). Owns no block-ID storage — `idom`/`post_order` are
/// sized to `func.blocks.items.len`.
///
/// - `idom[b]` is the immediate dominator of block `b`. The entry block
///   dominates only itself, so `idom[entry] == entry`. Unreachable blocks
///   (not visited from entry) have `idom[b] == null`.
/// - `post_order` lists reachable blocks in DFS post-order from the entry.
///   Reverse post-order (RPO) is the natural iteration order for forward
///   dataflow and is used internally by this pass.
///
/// Algorithm: "A Simple, Fast Dominance Algorithm" — Cooper, Harvey,
/// Kennedy (2001). O(N²) worst-case on pathological CFGs, near-linear in
/// practice on structured wasm CFGs.
pub const DomTree = struct {
    idom: []?ir.BlockId,
    /// Post-order numbering for each block, or `null` if unreachable.
    /// Higher number ⇒ later in post-order ⇒ earlier in reverse post-order.
    post_num: []?u32,
    /// Reachable blocks in DFS post-order from entry (length ≤ nblocks).
    post_order: []ir.BlockId,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *DomTree) void {
        self.allocator.free(self.idom);
        self.allocator.free(self.post_num);
        self.allocator.free(self.post_order);
    }

    /// Returns true if `a` dominates `b` (reflexive: every block dominates
    /// itself). Unreachable blocks are dominated only by themselves.
    pub fn dominates(self: *const DomTree, a: ir.BlockId, b: ir.BlockId) bool {
        if (a == b) return true;
        if (self.idom[b] == null) return false;
        var cur: ir.BlockId = b;
        while (true) {
            const next = self.idom[cur] orelse return false;
            if (next == a) return true;
            // Entry's idom is itself; stop to avoid infinite loop.
            if (next == cur) return false;
            cur = next;
        }
    }
};

/// Compute predecessors for each block from `buildSuccessors`. The
/// `BasicBlock.predecessors` field is not guaranteed to be populated by
/// all IR producers, so passes that need predecessors should use this.
pub fn buildPredecessors(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) !std.AutoHashMap(ir.BlockId, []const ir.BlockId) {
    const successors = try buildSuccessors(func, allocator);
    defer {
        var it = successors.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        @constCast(&successors).deinit();
    }

    var lists = std.AutoHashMap(ir.BlockId, std.ArrayList(ir.BlockId)).init(allocator);
    defer {
        var it = lists.iterator();
        while (it.next()) |entry| entry.value_ptr.deinit(allocator);
        lists.deinit();
    }
    for (0..func.blocks.items.len) |idx| {
        try lists.put(@intCast(idx), .empty);
    }

    var sit = successors.iterator();
    while (sit.next()) |entry| {
        const from = entry.key_ptr.*;
        for (entry.value_ptr.*) |to| {
            const list_ptr = lists.getPtr(to).?;
            // Deduplicate (br_table may list a target more than once).
            var already = false;
            for (list_ptr.items) |p| {
                if (p == from) {
                    already = true;
                    break;
                }
            }
            if (!already) try list_ptr.append(allocator, from);
        }
    }

    var result = std.AutoHashMap(ir.BlockId, []const ir.BlockId).init(allocator);
    errdefer {
        var it = result.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        result.deinit();
    }
    var lit = lists.iterator();
    while (lit.next()) |entry| {
        const owned = try entry.value_ptr.toOwnedSlice(allocator);
        try result.put(entry.key_ptr.*, owned);
    }
    return result;
}

/// Compute the dominator tree for `func` rooted at block 0.
pub fn computeDominators(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) !DomTree {
    const nblocks = func.blocks.items.len;

    const successors = try buildSuccessors(func, allocator);
    defer {
        var sit = successors.iterator();
        while (sit.next()) |entry| allocator.free(entry.value_ptr.*);
        @constCast(&successors).deinit();
    }
    var predecessors = try buildPredecessors(func, allocator);
    defer {
        var pit = predecessors.iterator();
        while (pit.next()) |entry| allocator.free(entry.value_ptr.*);
        predecessors.deinit();
    }

    // ── Iterative DFS to produce post-order from entry (block 0) ──
    var post_order: std.ArrayList(ir.BlockId) = .empty;
    errdefer post_order.deinit(allocator);

    const post_num = try allocator.alloc(?u32, nblocks);
    errdefer allocator.free(post_num);
    @memset(post_num, null);

    if (nblocks > 0) {
        const StackEntry = struct { bid: ir.BlockId, next_succ: usize };
        var visited = try allocator.alloc(bool, nblocks);
        defer allocator.free(visited);
        @memset(visited, false);

        var stack: std.ArrayList(StackEntry) = .empty;
        defer stack.deinit(allocator);

        const entry: ir.BlockId = 0;
        visited[entry] = true;
        try stack.append(allocator, .{ .bid = entry, .next_succ = 0 });

        while (stack.items.len > 0) {
            const top = &stack.items[stack.items.len - 1];
            const succs = successors.get(top.bid) orelse &[_]ir.BlockId{};
            if (top.next_succ < succs.len) {
                const s = succs[top.next_succ];
                top.next_succ += 1;
                if (!visited[s]) {
                    visited[s] = true;
                    try stack.append(allocator, .{ .bid = s, .next_succ = 0 });
                }
            } else {
                const bid = top.bid;
                post_num[bid] = @intCast(post_order.items.len);
                try post_order.append(allocator, bid);
                _ = stack.pop();
            }
        }
    }

    // ── Cooper-Harvey-Kennedy iterative idom computation ──
    const idom = try allocator.alloc(?ir.BlockId, nblocks);
    errdefer allocator.free(idom);
    @memset(idom, null);

    if (nblocks > 0 and post_num[0] != null) {
        // Entry dominates itself.
        idom[0] = 0;

        // Reverse post-order excluding entry.
        const Intersect = struct {
            fn call(idom_slice: []const ?ir.BlockId, post: []const ?u32, b1_in: ir.BlockId, b2_in: ir.BlockId) ir.BlockId {
                var b1 = b1_in;
                var b2 = b2_in;
                while (b1 != b2) {
                    while (post[b1].? < post[b2].?) b1 = idom_slice[b1].?;
                    while (post[b2].? < post[b1].?) b2 = idom_slice[b2].?;
                }
                return b1;
            }
        };

        var changed = true;
        while (changed) {
            changed = false;
            // Iterate reverse post-order, skipping the entry node (last in post_order).
            var i: usize = post_order.items.len;
            while (i > 0) {
                i -= 1;
                const b = post_order.items[i];
                if (b == 0) continue;

                const preds = predecessors.get(b) orelse &[_]ir.BlockId{};
                // Pick first processed predecessor as the running idom.
                var new_idom_opt: ?ir.BlockId = null;
                var other_start: usize = 0;
                for (preds, 0..) |p, pi| {
                    if (idom[p] != null) {
                        new_idom_opt = p;
                        other_start = pi + 1;
                        break;
                    }
                }
                const new_idom_first = new_idom_opt orelse continue;
                var new_idom = new_idom_first;
                for (preds[other_start..]) |p| {
                    if (idom[p] != null) {
                        new_idom = Intersect.call(idom, post_num, p, new_idom);
                    }
                }
                if (idom[b] == null or idom[b].? != new_idom) {
                    idom[b] = new_idom;
                    changed = true;
                }
            }
        }
    }

    return .{
        .idom = idom,
        .post_num = post_num,
        .post_order = try post_order.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

// ── Dominance frontiers ─────────────────────────────────────────────────

/// Compute the dominance frontier for every block in `func`.
///
/// DF(b) = { y | ∃ pred of y that b dominates, but b does not strictly
///             dominate y }. Uses the efficient "bottom-up" algorithm
/// from Cooper, Harvey & Kennedy (2001), §4.2.
///
/// Caller owns the returned slices; call `freeDominanceFrontiers` to release.
pub fn computeDominanceFrontiers(
    dom: *const DomTree,
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) ![][]const ir.BlockId {
    const nblocks = func.blocks.items.len;

    var preds = try buildPredecessors(func, allocator);
    defer {
        var pit = preds.iterator();
        while (pit.next()) |entry| allocator.free(entry.value_ptr.*);
        preds.deinit();
    }

    // Accumulate DF sets as ArrayLists, then convert to owned slices.
    var df_lists = try allocator.alloc(std.ArrayList(ir.BlockId), nblocks);
    defer allocator.free(df_lists);
    for (df_lists) |*l| l.* = .empty;

    for (0..nblocks) |idx| {
        const b: ir.BlockId = @intCast(idx);
        const pred_list = preds.get(b) orelse continue;
        if (pred_list.len < 2) continue; // join point iff ≥2 preds

        for (pred_list) |p| {
            var runner = p;
            while (runner != (dom.idom[b] orelse break)) {
                // Add b to DF(runner) if not already present.
                var dup = false;
                for (df_lists[runner].items) |existing| {
                    if (existing == b) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) try df_lists[runner].append(allocator, b);
                const next = dom.idom[runner] orelse break;
                if (next == runner) break;
                runner = next;
            }
        }
    }

    // Convert to owned slices.
    const result = try allocator.alloc([]const ir.BlockId, nblocks);
    errdefer allocator.free(result);
    for (df_lists, 0..) |*l, i| {
        result[i] = try l.toOwnedSlice(allocator);
    }
    return result;
}

/// Free the slices returned by `computeDominanceFrontiers`.
pub fn freeDominanceFrontiers(df: [][]const ir.BlockId, allocator: std.mem.Allocator) void {
    for (df) |s| allocator.free(s);
    allocator.free(df);
}

// ── Natural-loop detection ──────────────────────────────────────────────

/// A natural loop identified by a single header and one or more latches
/// (blocks with back-edges to the header). Multiple back-edges to the
/// same header are merged into one loop, matching the standard
/// "natural loops of a flow graph" definition.
///
/// Invariants:
///   - `header ∈ blocks` and every `latch ∈ blocks`.
///   - `header` dominates every block in `blocks`.
///   - `blocks` and `latches` are sorted ascending, no duplicates.
pub const Loop = struct {
    header: ir.BlockId,
    latches: []ir.BlockId,
    blocks: []ir.BlockId,

    /// O(log N) membership test (blocks is sorted).
    pub fn containsBlock(self: *const Loop, bid: ir.BlockId) bool {
        var lo: usize = 0;
        var hi: usize = self.blocks.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const v = self.blocks[mid];
            if (v == bid) return true;
            if (v < bid) lo = mid + 1 else hi = mid;
        }
        return false;
    }
};

/// A forest of natural loops for a function. Each loop is identified by
/// its index in `loops`. `header_loop` maps a header block ID back to
/// its loop index, so callers can answer "is this block a loop header?"
/// and "what loop does this header start?" in O(1).
pub const LoopForest = struct {
    loops: []Loop,
    header_loop: std.AutoHashMap(ir.BlockId, u32),
    allocator: std.mem.Allocator,

    pub fn deinit(self: *LoopForest) void {
        for (self.loops) |*loop| {
            self.allocator.free(loop.latches);
            self.allocator.free(loop.blocks);
        }
        self.allocator.free(self.loops);
        self.header_loop.deinit();
    }

    /// Returns true if `bid` is the header of some loop in this forest.
    pub fn isHeader(self: *const LoopForest, bid: ir.BlockId) bool {
        return self.header_loop.contains(bid);
    }
};

/// Compute the natural-loop forest for `func` using the supplied
/// dominator tree. Only reachable back-edges contribute to loops;
/// unreachable subgraphs (null idom) are ignored.
pub fn computeLoops(
    func: *const ir.IrFunction,
    dom: *const DomTree,
    allocator: std.mem.Allocator,
) !LoopForest {
    const nblocks = func.blocks.items.len;

    const successors = try buildSuccessors(func, allocator);
    defer {
        var sit = successors.iterator();
        while (sit.next()) |entry| allocator.free(entry.value_ptr.*);
        @constCast(&successors).deinit();
    }
    var predecessors = try buildPredecessors(func, allocator);
    defer {
        var pit = predecessors.iterator();
        while (pit.next()) |entry| allocator.free(entry.value_ptr.*);
        predecessors.deinit();
    }

    // ── Collect back-edges per header ──
    // A back-edge is t → h where h dominates t. Both ends must be
    // reachable (have a non-null idom).
    var latches_by_header = std.AutoHashMap(ir.BlockId, std.ArrayList(ir.BlockId)).init(allocator);
    defer {
        var it = latches_by_header.iterator();
        while (it.next()) |entry| entry.value_ptr.deinit(allocator);
        latches_by_header.deinit();
    }

    var from_idx: usize = 0;
    while (from_idx < nblocks) : (from_idx += 1) {
        const from: ir.BlockId = @intCast(from_idx);
        if (dom.idom[from] == null) continue;
        const succs = successors.get(from) orelse continue;
        for (succs) |to| {
            if (dom.idom[to] == null) continue;
            if (!dom.dominates(to, from)) continue;
            const gop = try latches_by_header.getOrPut(to);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            // br_table / br_if may produce duplicate (from → to) edges.
            var dup = false;
            for (gop.value_ptr.items) |l| {
                if (l == from) {
                    dup = true;
                    break;
                }
            }
            if (!dup) try gop.value_ptr.append(allocator, from);
        }
    }

    // ── Materialize one Loop per header ──
    var loops: std.ArrayList(Loop) = .empty;
    errdefer {
        for (loops.items) |*loop| {
            allocator.free(loop.latches);
            allocator.free(loop.blocks);
        }
        loops.deinit(allocator);
    }

    var header_loop = std.AutoHashMap(ir.BlockId, u32).init(allocator);
    errdefer header_loop.deinit();

    // Scratch buffers reused across headers.
    var in_loop = try allocator.alloc(bool, nblocks);
    defer allocator.free(in_loop);
    var worklist: std.ArrayList(ir.BlockId) = .empty;
    defer worklist.deinit(allocator);

    var hit = latches_by_header.iterator();
    while (hit.next()) |entry| {
        const header = entry.key_ptr.*;
        const latch_list = entry.value_ptr.*;

        // Standard natural-loop body computation: start from header, add
        // each latch, then walk predecessors but never past the header.
        @memset(in_loop, false);
        in_loop[header] = true;
        worklist.clearRetainingCapacity();
        for (latch_list.items) |latch| {
            if (!in_loop[latch]) {
                in_loop[latch] = true;
                try worklist.append(allocator, latch);
            }
        }
        while (worklist.pop()) |bid| {
            const preds = predecessors.get(bid) orelse continue;
            for (preds) |p| {
                if (!in_loop[p]) {
                    in_loop[p] = true;
                    try worklist.append(allocator, p);
                }
            }
        }

        // Freeze in_loop → sorted blocks slice.
        var count: usize = 0;
        for (in_loop) |b| {
            if (b) count += 1;
        }
        const blocks_slice = try allocator.alloc(ir.BlockId, count);
        errdefer allocator.free(blocks_slice);
        var bi: usize = 0;
        for (in_loop, 0..) |b, idx| {
            if (b) {
                blocks_slice[bi] = @intCast(idx);
                bi += 1;
            }
        }

        // Sort latches ascending, no duplicates (duplicates already filtered above).
        const latches_slice = try allocator.dupe(ir.BlockId, latch_list.items);
        std.mem.sort(ir.BlockId, latches_slice, {}, std.sort.asc(ir.BlockId));

        try header_loop.put(header, @intCast(loops.items.len));
        try loops.append(allocator, .{
            .header = header,
            .latches = latches_slice,
            .blocks = blocks_slice,
        });
    }

    return .{
        .loops = try loops.toOwnedSlice(allocator),
        .header_loop = header_loop,
        .allocator = allocator,
    };
}

// ── Tests ───────────────────────────────────────────────────────────────

test "buildSuccessors: linear block" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const block0 = func.getBlock(b0);
    try block0.append(.{ .op = .{ .ret = null } });

    var succs = try buildSuccessors(&func, allocator);
    defer {
        var it = succs.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        succs.deinit();
    }

    try std.testing.expectEqual(@as(usize, 0), succs.get(b0).?.len);
}

test "buildSuccessors: branch block" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const v0 = func.newVReg();
    try block0.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try block0.append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    var succs = try buildSuccessors(&func, allocator);
    defer {
        var it = succs.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        succs.deinit();
    }

    try std.testing.expectEqual(@as(usize, 2), succs.get(b0).?.len);
    try std.testing.expectEqual(@as(usize, 0), succs.get(b1).?.len);
}

test "computeLiveness: simple def-use in one block" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block0.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try block0.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try block0.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try block0.append(.{ .op = .{ .ret = v2 } });

    var liveness = try computeLiveness(&func, allocator);
    defer {
        var it = liveness.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.live_in.deinit();
            entry.value_ptr.live_out.deinit();
        }
        liveness.deinit();
    }

    // Nothing should be live_in at the entry block
    try std.testing.expectEqual(@as(u32, 0), liveness.get(b0).?.live_in.count());
    // Nothing should be live_out (ret terminates)
    try std.testing.expectEqual(@as(u32, 0), liveness.get(b0).?.live_out.count());
}

test "computeLiveRanges: basic ranges" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block0.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 }); // pos 0
    try block0.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 }); // pos 1
    try block0.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 }); // pos 2
    try block0.append(.{ .op = .{ .ret = v2 } }); // pos 3

    const ranges = try computeLiveRanges(&func, allocator);
    defer allocator.free(ranges);

    // Should have 3 live ranges (v0, v1, v2)
    try std.testing.expectEqual(@as(usize, 3), ranges.len);
    // Sorted by start: v0 at 0, v1 at 1, v2 at 2
    try std.testing.expectEqual(@as(u32, 0), ranges[0].start);
    try std.testing.expectEqual(@as(u32, 2), ranges[0].end); // v0 used at pos 2 (add)
    try std.testing.expectEqual(@as(u32, 1), ranges[1].start);
    try std.testing.expectEqual(@as(u32, 2), ranges[1].end); // v1 used at pos 2 (add)
    try std.testing.expectEqual(@as(u32, 2), ranges[2].start);
    try std.testing.expectEqual(@as(u32, 3), ranges[2].end); // v2 used at pos 3 (ret)
}

test "computeLiveRanges: call with explicit args" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    const args = try allocator.alloc(ir.VReg, 2);
    defer allocator.free(args);
    args[0] = v0;
    args[1] = v1;
    try block0.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 }); // pos 0
    try block0.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 }); // pos 1
    try block0.append(.{ .op = .{ .call = .{ .func_idx = 0, .args = args } }, .dest = v2 }); // pos 2
    try block0.append(.{ .op = .{ .ret = v2 } }); // pos 3

    const ranges = try computeLiveRanges(&func, allocator);
    defer allocator.free(ranges);

    try std.testing.expectEqual(@as(usize, 3), ranges.len);
    // v0: defined at 0, used at 2 (call arg)
    try std.testing.expectEqual(@as(u32, 0), ranges[0].start);
    try std.testing.expectEqual(@as(u32, 2), ranges[0].end);
    // v1: defined at 1, used at 2 (call arg)
    try std.testing.expectEqual(@as(u32, 1), ranges[1].start);
    try std.testing.expectEqual(@as(u32, 2), ranges[1].end);
}

test "computeLiveness: cross-block value is live" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 1);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const block1 = func.getBlock(b1);

    const v0 = func.newVReg();
    try block0.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0 });
    try block0.append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v0 } } });
    try block0.append(.{ .op = .{ .br = b1 } });

    const v1 = func.newVReg();
    try block1.append(.{ .op = .{ .local_get = 0 }, .dest = v1 });
    try block1.append(.{ .op = .{ .ret = v1 } });

    var liveness = try computeLiveness(&func, allocator);
    defer {
        var it = liveness.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.live_in.deinit();
            entry.value_ptr.live_out.deinit();
        }
        liveness.deinit();
    }

    // Block 0 should have nothing live_in (entry block)
    try std.testing.expectEqual(@as(u32, 0), liveness.get(b0).?.live_in.count());
}

test "buildSuccessors: loop backedge" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const block1 = func.getBlock(b1);

    try block0.append(.{ .op = .{ .br = b1 } });
    const v0 = func.newVReg();
    try block1.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try block1.append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b0, .else_block = b1 } } });

    var succs = try buildSuccessors(&func, allocator);
    defer {
        var it = succs.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        succs.deinit();
    }

    // Block 0 has successor b1
    try std.testing.expectEqual(@as(usize, 1), succs.get(b0).?.len);
    // Block 1 has successors b0 and b1 (loop)
    try std.testing.expectEqual(@as(usize, 2), succs.get(b1).?.len);
}

test "buildPredecessors: diamond" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // entry → {b1, b2} → b3
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    var preds = try buildPredecessors(&func, allocator);
    defer {
        var it = preds.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        preds.deinit();
    }

    try std.testing.expectEqual(@as(usize, 0), preds.get(b0).?.len);
    try std.testing.expectEqual(@as(usize, 1), preds.get(b1).?.len);
    try std.testing.expectEqual(@as(usize, 1), preds.get(b2).?.len);
    try std.testing.expectEqual(@as(usize, 2), preds.get(b3).?.len);
}

test "computeDominators: linear chain" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b2 } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();

    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b0]);
    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b1]);
    try std.testing.expectEqual(@as(?ir.BlockId, b1), dom.idom[b2]);
    try std.testing.expect(dom.dominates(b0, b2));
    try std.testing.expect(dom.dominates(b1, b2));
    try std.testing.expect(!dom.dominates(b2, b0));
}

test "computeDominators: diamond idom is entry for merge" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();

    // Entry dominates itself.
    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b0]);
    // Sides' idom is entry.
    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b1]);
    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b2]);
    // Merge's idom is entry (neither side dominates the merge).
    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b3]);

    try std.testing.expect(dom.dominates(b0, b3));
    try std.testing.expect(!dom.dominates(b1, b3));
    try std.testing.expect(!dom.dominates(b2, b3));
}

test "computeDominators: simple loop" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // b0 → b1 → (self-loop or to b2)
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();

    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b0]);
    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b1]);
    // b1 is the only predecessor of b2 that is reachable; b1 dominates b2.
    try std.testing.expectEqual(@as(?ir.BlockId, b1), dom.idom[b2]);
    try std.testing.expect(dom.dominates(b0, b2));
    try std.testing.expect(dom.dominates(b1, b2));
}

test "computeDominators: unreachable block has null idom" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock(); // unreachable
    try func.getBlock(b0).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();

    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b0]);
    try std.testing.expectEqual(@as(?ir.BlockId, null), dom.idom[b1]);
    try std.testing.expect(!dom.dominates(b0, b1));
}

test "computeLoops: no loops in DAG" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();
    var lf = try computeLoops(&func, &dom, allocator);
    defer lf.deinit();

    try std.testing.expectEqual(@as(usize, 0), lf.loops.len);
    try std.testing.expect(!lf.isHeader(b0));
    try std.testing.expect(!lf.isHeader(b1));
}

test "computeLoops: self-loop" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock(); // self-loop
    const b2 = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();
    var lf = try computeLoops(&func, &dom, allocator);
    defer lf.deinit();

    try std.testing.expectEqual(@as(usize, 1), lf.loops.len);
    const loop = &lf.loops[0];
    try std.testing.expectEqual(b1, loop.header);
    try std.testing.expectEqual(@as(usize, 1), loop.latches.len);
    try std.testing.expectEqual(b1, loop.latches[0]);
    try std.testing.expect(loop.containsBlock(b1));
    try std.testing.expect(!loop.containsBlock(b0));
    try std.testing.expect(!loop.containsBlock(b2));
    try std.testing.expect(lf.isHeader(b1));
}

test "computeLoops: while-loop body" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // b0 → b1(header) → b2(body) → b1 (back-edge) | b3(exit)
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b2, .else_block = b3 } } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();
    var lf = try computeLoops(&func, &dom, allocator);
    defer lf.deinit();

    try std.testing.expectEqual(@as(usize, 1), lf.loops.len);
    const loop = &lf.loops[0];
    try std.testing.expectEqual(b1, loop.header);
    try std.testing.expectEqual(@as(usize, 2), loop.blocks.len);
    try std.testing.expect(loop.containsBlock(b1));
    try std.testing.expect(loop.containsBlock(b2));
    try std.testing.expect(!loop.containsBlock(b0));
    try std.testing.expect(!loop.containsBlock(b3));
    try std.testing.expectEqual(@as(usize, 1), loop.latches.len);
    try std.testing.expectEqual(b2, loop.latches[0]);
}

test "computeLoops: multiple latches share header" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // b0 → h; h branches to b2 or b3; both jump back to h; h has exit b4.
    const b0 = try func.newBlock();
    const h = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const b4 = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = h } });
    try func.getBlock(h).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(h).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b2, .else_block = b3 } } });
    try func.getBlock(b2).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v1 });
    try func.getBlock(b2).append(.{ .op = .{ .br_if = .{ .cond = v1, .then_block = h, .else_block = b4 } } });
    try func.getBlock(b3).append(.{ .op = .{ .br = h } });
    try func.getBlock(b4).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();
    var lf = try computeLoops(&func, &dom, allocator);
    defer lf.deinit();

    try std.testing.expectEqual(@as(usize, 1), lf.loops.len);
    const loop = &lf.loops[0];
    try std.testing.expectEqual(h, loop.header);
    try std.testing.expectEqual(@as(usize, 2), loop.latches.len);
    // Sorted ascending.
    try std.testing.expectEqual(b2, loop.latches[0]);
    try std.testing.expectEqual(b3, loop.latches[1]);
    try std.testing.expect(loop.containsBlock(h));
    try std.testing.expect(loop.containsBlock(b2));
    try std.testing.expect(loop.containsBlock(b3));
    try std.testing.expect(!loop.containsBlock(b0));
    try std.testing.expect(!loop.containsBlock(b4));
}

test "computeLoops: nested loops" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // outer: h_o → h_i → body → h_i (inner back-edge) | exit_i → h_o (outer back-edge) | exit_o
    const b0 = try func.newBlock();
    const h_o = try func.newBlock();
    const h_i = try func.newBlock();
    const body = try func.newBlock();
    const exit_i = try func.newBlock();
    const exit_o = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = h_o } });
    try func.getBlock(h_o).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(h_o).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = h_i, .else_block = exit_o } } });
    try func.getBlock(h_i).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v1 });
    try func.getBlock(h_i).append(.{ .op = .{ .br_if = .{ .cond = v1, .then_block = body, .else_block = exit_i } } });
    try func.getBlock(body).append(.{ .op = .{ .br = h_i } });
    try func.getBlock(exit_i).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v2 });
    try func.getBlock(exit_i).append(.{ .op = .{ .br_if = .{ .cond = v2, .then_block = h_o, .else_block = exit_o } } });
    try func.getBlock(exit_o).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();
    var lf = try computeLoops(&func, &dom, allocator);
    defer lf.deinit();

    try std.testing.expectEqual(@as(usize, 2), lf.loops.len);
    try std.testing.expect(lf.isHeader(h_o));
    try std.testing.expect(lf.isHeader(h_i));

    const outer = &lf.loops[lf.header_loop.get(h_o).?];
    const inner = &lf.loops[lf.header_loop.get(h_i).?];

    // Inner loop: h_i and body only.
    try std.testing.expectEqual(@as(usize, 2), inner.blocks.len);
    try std.testing.expect(inner.containsBlock(h_i));
    try std.testing.expect(inner.containsBlock(body));

    // Outer loop contains all inner blocks plus h_o and exit_i.
    try std.testing.expect(outer.containsBlock(h_o));
    try std.testing.expect(outer.containsBlock(h_i));
    try std.testing.expect(outer.containsBlock(body));
    try std.testing.expect(outer.containsBlock(exit_i));
    try std.testing.expect(!outer.containsBlock(exit_o));
    try std.testing.expect(!outer.containsBlock(b0));
}

test "computeLoops: irreducible-ish (no back-edge without dominator) produces no loops" {
    // Ensure we don't spuriously report a loop when CFG has a cycle but
    // no edge target dominates its source (dominator-based natural-loop
    // detection ignores irreducible cycles by design).
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // b0 → {b1, b2}; b1 → b2; b2 → b1. Neither b1 nor b2 dominates
    // the other because both are entered directly from b0.
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v1 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v1, .then_block = b2, .else_block = b0 } } });
    try func.getBlock(b2).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v2 });
    try func.getBlock(b2).append(.{ .op = .{ .br_if = .{ .cond = v2, .then_block = b1, .else_block = b0 } } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();
    var lf = try computeLoops(&func, &dom, allocator);
    defer lf.deinit();

    // Only b0 dominates b1 and b2. There's an edge b1→b0 and b2→b0
    // (b0 dominates both), so those are back-edges ⇒ one natural loop
    // headed at b0. The cycle b1↔b2 is irreducible and not reported.
    try std.testing.expectEqual(@as(usize, 1), lf.loops.len);
    try std.testing.expectEqual(b0, lf.loops[0].header);
    try std.testing.expectEqual(@as(usize, 3), lf.loops[0].blocks.len);
}
