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
        => |bin| {
            live.put(bin.lhs, {}) catch {};
            live.put(bin.rhs, {}) catch {};
        },

        .clz, .ctz, .popcnt, .eqz, .wrap_i64, .extend_i32_s, .extend_i32_u,
        .extend8_s, .extend16_s, .extend32_s,
        .f_neg, .f_abs, .f_sqrt, .f_ceil, .f_floor, .f_trunc, .f_nearest,
        .trunc_f32_s, .trunc_f32_u, .trunc_f64_s, .trunc_f64_u,
        .convert_s, .convert_u, .demote_f64, .promote_f32, .reinterpret,
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
        .ret => |maybe_vreg| if (maybe_vreg) |v| live.put(v, {}) catch {},
        .call => |cl| {
            for (cl.args) |arg| live.put(arg, {}) catch {};
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
    }
}

// ── Live range computation ──────────────────────────────────────────────

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
    const liveness = try computeLiveness(func, allocator);
    defer {
        var it = @constCast(&liveness).iterator();
        while (it.next()) |entry| {
            entry.value_ptr.live_in.deinit();
            entry.value_ptr.live_out.deinit();
        }
        @constCast(&liveness).deinit();
    }

    // Global instruction numbering
    var def_pos = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer def_pos.deinit();
    var last_use_pos = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer last_use_pos.deinit();

    var global_idx: u32 = 0;
    for (func.blocks.items, 0..) |block, block_idx| {
        const bid: ir.BlockId = @intCast(block_idx);

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
        => |bin| {
            last_use.put(bin.lhs, pos) catch {};
            last_use.put(bin.rhs, pos) catch {};
        },

        .clz, .ctz, .popcnt, .eqz, .wrap_i64, .extend_i32_s, .extend_i32_u,
        .extend8_s, .extend16_s, .extend32_s,
        .f_neg, .f_abs, .f_sqrt, .f_ceil, .f_floor, .f_trunc, .f_nearest,
        .trunc_f32_s, .trunc_f32_u, .trunc_f64_s, .trunc_f64_u,
        .convert_s, .convert_u, .demote_f64, .promote_f32, .reinterpret,
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
        .ret => |maybe_vreg| if (maybe_vreg) |v| last_use.put(v, pos) catch {},
        .call => |cl| {
            for (cl.args) |arg| last_use.put(arg, pos) catch {};
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
    }
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
