//! Conservative local scheduler for the AArch64 backend.
//!
//! This schedules a backend-local copy of each basic block's IR instructions.
//! The source IR is left untouched; codegen consumes the scheduled copies so
//! every position-sensitive scan can use the same order.

const std = @import("std");
const ir = @import("../../ir/ir.zig");

pub const Options = struct {
    enabled: bool = true,
    max_window_len: usize = 128,
};

pub const Class = enum {
    barrier,
    constant,
    alu,
    mul,
    compare,
    load,
    store,
};

pub const Metadata = struct {
    class: Class,
    latency: u8,
    barrier: bool,
    def: ?ir.VReg,
};

pub const FunctionSchedule = struct {
    allocator: std.mem.Allocator,
    blocks: []BlockSchedule,

    pub fn deinit(self: *FunctionSchedule) void {
        for (self.blocks) |*block| block.deinit(self.allocator);
        self.allocator.free(self.blocks);
    }

    pub fn instructions(self: *const FunctionSchedule, block_id: ir.BlockId) []const ir.Inst {
        return self.blocks[block_id].instructions;
    }
};

const BlockSchedule = struct {
    instructions: []ir.Inst = &.{},

    fn deinit(self: *BlockSchedule, allocator: std.mem.Allocator) void {
        allocator.free(self.instructions);
    }
};

const Node = struct {
    preds_left: u32 = 0,
    succs: std.ArrayListUnmanaged(usize) = .empty,
    latency: u8,
    height: u32 = 0,
    original_index: usize,

    fn deinit(self: *Node, allocator: std.mem.Allocator) void {
        self.succs.deinit(allocator);
    }
};

const EdgeCtx = struct {
    defs: *std.AutoHashMap(ir.VReg, usize),
    nodes: []Node,
    user: usize,
    allocator: std.mem.Allocator,
};

pub fn scheduleFunction(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
    options: Options,
) !FunctionSchedule {
    var blocks = try allocator.alloc(BlockSchedule, func.blocks.items.len);
    errdefer allocator.free(blocks);
    for (blocks) |*block| block.* = .{};

    var initialized: usize = 0;
    errdefer {
        for (blocks[0..initialized]) |*block| block.deinit(allocator);
    }

    for (func.blocks.items, 0..) |block, idx| {
        blocks[idx].instructions = try scheduleBlock(block.instructions.items, allocator, options);
        initialized += 1;
    }

    return .{
        .allocator = allocator,
        .blocks = blocks,
    };
}

pub fn scheduleBlock(
    insts: []const ir.Inst,
    allocator: std.mem.Allocator,
    options: Options,
) ![]ir.Inst {
    if (!options.enabled or insts.len <= 1) return allocator.dupe(ir.Inst, insts);

    var out: std.ArrayListUnmanaged(ir.Inst) = .empty;
    errdefer out.deinit(allocator);
    try out.ensureTotalCapacity(allocator, insts.len);

    var window_start: usize = 0;
    for (insts, 0..) |inst, idx| {
        const meta = metadata(inst);
        if (meta.barrier) {
            try appendScheduledWindow(insts[window_start..idx], &out, allocator, options);
            out.appendAssumeCapacity(inst);
            window_start = idx + 1;
        }
    }
    try appendScheduledWindow(insts[window_start..], &out, allocator, options);

    return out.toOwnedSlice(allocator);
}

fn appendScheduledWindow(
    window: []const ir.Inst,
    out: *std.ArrayListUnmanaged(ir.Inst),
    allocator: std.mem.Allocator,
    options: Options,
) !void {
    if (window.len == 0) return;
    if (window.len == 1 or window.len > options.max_window_len) {
        out.appendSliceAssumeCapacity(window);
        return;
    }

    var nodes = try allocator.alloc(Node, window.len);
    defer {
        for (nodes) |*node| node.deinit(allocator);
        allocator.free(nodes);
    }

    for (window, 0..) |inst, idx| {
        const meta = metadata(inst);
        std.debug.assert(!meta.barrier);
        nodes[idx] = .{
            .latency = meta.latency,
            .original_index = idx,
        };
    }

    var defs = std.AutoHashMap(ir.VReg, usize).init(allocator);
    defer defs.deinit();

    const addUseEdge = struct {
        fn f(ctx: *EdgeCtx, use: ir.VReg) !void {
            const pred = ctx.defs.get(use) orelse return;
            try ctx.nodes[pred].succs.append(ctx.allocator, ctx.user);
            ctx.nodes[ctx.user].preds_left += 1;
        }
    }.f;

    for (window, 0..) |inst, idx| {
        const meta = metadata(inst);
        var edge_ctx = EdgeCtx{
            .defs = &defs,
            .nodes = nodes,
            .user = idx,
            .allocator = allocator,
        };
        try forEachUse(inst, &edge_ctx, addUseEdge);
        if (meta.def) |dest| try defs.put(dest, idx);
    }

    var last_memory: ?usize = null;
    for (window, 0..) |inst, idx| {
        if (!isOrderedMemory(inst)) continue;
        if (last_memory) |pred| {
            try nodes[pred].succs.append(allocator, idx);
            nodes[idx].preds_left += 1;
        }
        last_memory = idx;
    }

    var rev_idx = nodes.len;
    while (rev_idx > 0) {
        rev_idx -= 1;
        var best_succ_height: u32 = 0;
        for (nodes[rev_idx].succs.items) |succ| {
            best_succ_height = @max(best_succ_height, nodes[succ].height);
        }
        nodes[rev_idx].height = @as(u32, nodes[rev_idx].latency) + best_succ_height;
    }

    var ready: std.ArrayListUnmanaged(usize) = .empty;
    defer ready.deinit(allocator);
    for (nodes, 0..) |node, idx| {
        if (node.preds_left == 0) try ready.append(allocator, idx);
    }

    var emitted: usize = 0;
    while (emitted < nodes.len) : (emitted += 1) {
        if (ready.items.len == 0) return error.ScheduleCycle;

        const ready_pos = bestReadyIndex(nodes, ready.items);
        const node_idx = ready.orderedRemove(ready_pos);
        out.appendAssumeCapacity(window[node_idx]);

        for (nodes[node_idx].succs.items) |succ| {
            nodes[succ].preds_left -= 1;
            if (nodes[succ].preds_left == 0) try ready.append(allocator, succ);
        }
    }
}

fn bestReadyIndex(nodes: []const Node, ready: []const usize) usize {
    var best_pos: usize = 0;
    for (ready[1..], 1..) |candidate, pos| {
        if (better(nodes[candidate], nodes[ready[best_pos]])) best_pos = pos;
    }
    return best_pos;
}

fn better(a: Node, b: Node) bool {
    if (a.height != b.height) return a.height > b.height;
    if (a.latency != b.latency) return a.latency > b.latency;
    return a.original_index < b.original_index;
}

pub fn metadata(inst: ir.Inst) Metadata {
    const def = inst.dest;
    const class: Class = switch (inst.op) {
        .iconst_32, .iconst_64 => if (def != null) .constant else .barrier,
        .v128_const => if (def != null) .constant else .barrier,
        .v128_load => if (def != null) .load else .barrier,
        .v128_store => .store,
        .v128_not,
        .v128_bitwise,
        .i32x4_binop,
        .i32x4_shift,
        .i32x4_splat,
        .i32x4_extract_lane,
        .i32x4_replace_lane,
        .i8x16_binop,
        .i8x16_splat,
        .i8x16_extract_lane,
        .i8x16_replace_lane,
        .i16x8_binop,
        .i16x8_shift,
        .i16x8_splat,
        .i16x8_extract_lane,
        .i16x8_replace_lane,
        => if (def != null) .alu else .barrier,

        .mul => if (def != null and isIntegerType(inst.type)) .mul else .barrier,

        .add,
        .sub,
        .@"and",
        .@"or",
        .xor,
        .shl,
        .shr_s,
        .shr_u,
        .rotl,
        .rotr,
        => if (def != null and isIntegerType(inst.type)) .alu else .barrier,

        .clz,
        .ctz,
        .popcnt,
        .eqz,
        .extend8_s,
        .extend16_s,
        .extend32_s,
        .extend_i32_s,
        .extend_i32_u,
        .wrap_i64,
        => if (def != null and isIntegerType(inst.type)) .alu else .barrier,

        .eq,
        .ne,
        .lt_s,
        .lt_u,
        .gt_s,
        .gt_u,
        .le_s,
        .le_u,
        .ge_s,
        .ge_u,
        => if (def != null and isIntegerType(inst.type)) .compare else .barrier,

        .load => if (def != null) .load else .barrier,

        else => .barrier,
    };

    const latency: u8 = switch (class) {
        .load => 4,
        .mul => 3,
        .constant, .alu, .compare, .store => 1,
        .barrier => 0,
    };

    return .{
        .class = class,
        .latency = latency,
        .barrier = class == .barrier,
        .def = switch (class) {
            .barrier, .store => null,
            else => def,
        },
    };
}

fn isIntegerType(ty: ir.IrType) bool {
    return ty == .i32 or ty == .i64;
}

fn isOrderedMemory(inst: ir.Inst) bool {
    return switch (inst.op) {
        .load, .v128_load, .v128_store => true,
        else => false,
    };
}

pub fn forEachUse(
    inst: ir.Inst,
    context: anytype,
    comptime visit: fn (@TypeOf(context), ir.VReg) anyerror!void,
) !void {
    switch (inst.op) {
        .add,
        .sub,
        .mul,
        .@"and",
        .@"or",
        .xor,
        .div_s,
        .div_u,
        .rem_s,
        .rem_u,
        .shl,
        .shr_s,
        .shr_u,
        .rotl,
        .rotr,
        .eq,
        .ne,
        .lt_s,
        .lt_u,
        .gt_s,
        .gt_u,
        .le_s,
        .le_u,
        .ge_s,
        .ge_u,
        .f_eq,
        .f_ne,
        .f_lt,
        .f_gt,
        .f_le,
        .f_ge,
        .f_min,
        .f_max,
        .f_copysign,
        => |b| {
            try visit(context, b.lhs);
            try visit(context, b.rhs);
        },
        .local_set => |ls| try visit(context, ls.val),
        .global_set => |gs| try visit(context, gs.val),
        .eqz,
        .ctz,
        .clz,
        .popcnt,
        .extend8_s,
        .extend16_s,
        .extend32_s,
        .extend_i32_s,
        .extend_i32_u,
        .wrap_i64,
        .f_neg,
        .f_abs,
        .f_sqrt,
        .f_ceil,
        .f_floor,
        .f_trunc,
        .f_nearest,
        .convert_s,
        .convert_u,
        .convert_i32_s,
        .convert_i32_u,
        .convert_i64_s,
        .convert_i64_u,
        .demote_f64,
        .promote_f32,
        .trunc_f32_s,
        .trunc_f32_u,
        .trunc_f64_s,
        .trunc_f64_u,
        .trunc_sat_f32_s,
        .trunc_sat_f32_u,
        .trunc_sat_f64_s,
        .trunc_sat_f64_u,
        .reinterpret,
        .memory_grow,
        => |v| try visit(context, v),
        .ret => |maybe_v| if (maybe_v) |v| try visit(context, v),
        .ret_multi => |vregs| for (vregs) |v| try visit(context, v),
        .load => |ld| try visit(context, ld.base),
        .store => |st| {
            try visit(context, st.base);
            try visit(context, st.val);
        },
        .atomic_load => |ald| try visit(context, ald.base),
        .atomic_store => |ast| {
            try visit(context, ast.base);
            try visit(context, ast.val);
        },
        .atomic_rmw => |arm| {
            try visit(context, arm.base);
            try visit(context, arm.val);
        },
        .atomic_cmpxchg => |acx| {
            try visit(context, acx.base);
            try visit(context, acx.expected);
            try visit(context, acx.replacement);
        },
        .atomic_notify => |an| {
            try visit(context, an.base);
            try visit(context, an.count);
        },
        .atomic_wait => |aw| {
            try visit(context, aw.base);
            try visit(context, aw.expected);
            try visit(context, aw.timeout);
        },
        .select => |sel| {
            try visit(context, sel.cond);
            try visit(context, sel.if_true);
            try visit(context, sel.if_false);
        },
        .br_if => |bi| try visit(context, bi.cond),
        .br_table => |bt| try visit(context, bt.index),
        .call => |cl| for (cl.args) |a| try visit(context, a),
        .call_indirect => |ci| {
            try visit(context, ci.elem_idx);
            for (ci.args) |a| try visit(context, a);
        },
        .call_ref => |cr| {
            try visit(context, cr.func_ref);
            for (cr.args) |a| try visit(context, a);
        },
        .memory_fill => |mf| {
            try visit(context, mf.dst);
            try visit(context, mf.val);
            try visit(context, mf.len);
        },
        .memory_copy => |mc| {
            try visit(context, mc.dst);
            try visit(context, mc.src);
            try visit(context, mc.len);
        },
        .memory_init => |mi| {
            try visit(context, mi.dst);
            try visit(context, mi.src);
            try visit(context, mi.len);
        },
        .table_init => |ti| {
            try visit(context, ti.dst);
            try visit(context, ti.src);
            try visit(context, ti.len);
        },
        .table_get => |tg| try visit(context, tg.idx),
        .table_set => |ts| {
            try visit(context, ts.idx);
            try visit(context, ts.val);
        },
        .table_grow => |tg| {
            try visit(context, tg.init);
            try visit(context, tg.delta);
        },
        .phi => |edges| for (edges) |edge| try visit(context, edge.val),
        .v128_not => |v| try visit(context, v),
        .v128_load => |ld| try visit(context, ld.base),
        .v128_store => |st| {
            try visit(context, st.base);
            try visit(context, st.val);
        },
        .v128_bitwise => |bin| {
            try visit(context, bin.lhs);
            try visit(context, bin.rhs);
        },
        .i32x4_binop => |bin| {
            try visit(context, bin.lhs);
            try visit(context, bin.rhs);
        },
        .i8x16_binop => |bin| {
            try visit(context, bin.lhs);
            try visit(context, bin.rhs);
        },
        .i16x8_binop => |bin| {
            try visit(context, bin.lhs);
            try visit(context, bin.rhs);
        },
        .i32x4_shift => |shift| {
            try visit(context, shift.vector);
            try visit(context, shift.count);
        },
        .i16x8_shift => |shift| {
            try visit(context, shift.vector);
            try visit(context, shift.count);
        },
        .i32x4_splat => |v| try visit(context, v),
        .i32x4_extract_lane => |lane| try visit(context, lane.vector),
        .i32x4_replace_lane => |lane| {
            try visit(context, lane.vector);
            try visit(context, lane.val);
        },
        .i8x16_splat => |v| try visit(context, v),
        .i8x16_extract_lane => |lane| try visit(context, lane.vector),
        .i8x16_replace_lane => |lane| {
            try visit(context, lane.vector);
            try visit(context, lane.val);
        },
        .i16x8_splat => |v| try visit(context, v),
        .i16x8_extract_lane => |lane| try visit(context, lane.vector),
        .i16x8_replace_lane => |lane| {
            try visit(context, lane.vector);
            try visit(context, lane.val);
        },
        else => {},
    }
}

test "metadata marks pure integer ops schedulable and hazards as barriers" {
    const c = ir.Inst{ .op = .{ .iconst_32 = 1 }, .dest = 1, .type = .i32 };
    try std.testing.expect(!metadata(c).barrier);
    try std.testing.expectEqual(Class.constant, metadata(c).class);

    const mul = ir.Inst{ .op = .{ .mul = .{ .lhs = 1, .rhs = 2 } }, .dest = 3, .type = .i64 };
    try std.testing.expect(!metadata(mul).barrier);
    try std.testing.expectEqual(@as(u8, 3), metadata(mul).latency);

    const load = ir.Inst{ .op = .{ .load = .{ .base = 1, .offset = 0, .size = 4 } }, .dest = 2, .type = .i32 };
    try std.testing.expect(!metadata(load).barrier);
    try std.testing.expectEqual(Class.load, metadata(load).class);

    const call = ir.Inst{ .op = .{ .call = .{ .func_idx = 0, .args = &.{1} } }, .dest = 2, .type = .i32 };
    try std.testing.expect(metadata(call).barrier);

    const br_if = ir.Inst{ .op = .{ .br_if = .{ .cond = 1, .then_block = 0, .else_block = 1 } } };
    try std.testing.expect(metadata(br_if).barrier);

    const trap = ir.Inst{ .op = .{ .@"unreachable" = {} } };
    try std.testing.expect(metadata(trap).barrier);

    const atomic = ir.Inst{ .op = .{ .atomic_load = .{ .base = 1, .offset = 0, .size = 4 } }, .dest = 2, .type = .i32 };
    try std.testing.expect(metadata(atomic).barrier);
}

test "metadata models supported v128 ops as schedulable" {
    const vconst = ir.Inst{ .op = .{ .v128_const = 0x0102030405060708090a0b0c0d0e0f10 }, .dest = 1, .type = .v128 };
    try std.testing.expect(!metadata(vconst).barrier);
    try std.testing.expectEqual(Class.constant, metadata(vconst).class);
    try std.testing.expectEqual(@as(?ir.VReg, 1), metadata(vconst).def);

    const load = ir.Inst{ .op = .{ .v128_load = .{ .base = 2, .offset = 0, .alignment = 4 } }, .dest = 3, .type = .v128 };
    try std.testing.expect(!metadata(load).barrier);
    try std.testing.expectEqual(Class.load, metadata(load).class);
    try std.testing.expectEqual(@as(u8, 4), metadata(load).latency);
    try std.testing.expectEqual(@as(?ir.VReg, 3), metadata(load).def);

    const store = ir.Inst{ .op = .{ .v128_store = .{ .base = 4, .offset = 16, .alignment = 4, .val = 3 } }, .type = .void };
    try std.testing.expect(!metadata(store).barrier);
    try std.testing.expectEqual(Class.store, metadata(store).class);
    try std.testing.expectEqual(@as(?ir.VReg, null), metadata(store).def);

    const not = ir.Inst{ .op = .{ .v128_not = 3 }, .dest = 5, .type = .v128 };
    try std.testing.expect(!metadata(not).barrier);
    try std.testing.expectEqual(Class.alu, metadata(not).class);

    const bitwise = ir.Inst{ .op = .{ .v128_bitwise = .{ .op = .xor, .lhs = 3, .rhs = 5 } }, .dest = 6, .type = .v128 };
    try std.testing.expect(!metadata(bitwise).barrier);
    try std.testing.expectEqual(Class.alu, metadata(bitwise).class);

    const binop = ir.Inst{ .op = .{ .i32x4_binop = .{ .op = .add, .lhs = 3, .rhs = 6 } }, .dest = 7, .type = .v128 };
    try std.testing.expect(!metadata(binop).barrier);
    try std.testing.expectEqual(Class.alu, metadata(binop).class);

    const splat = ir.Inst{ .op = .{ .i32x4_splat = 8 }, .dest = 9, .type = .v128 };
    try std.testing.expect(!metadata(splat).barrier);
    try std.testing.expectEqual(Class.alu, metadata(splat).class);

    const extract = ir.Inst{ .op = .{ .i32x4_extract_lane = .{ .vector = 7, .lane = 2 } }, .dest = 10, .type = .i32 };
    try std.testing.expect(!metadata(extract).barrier);
    try std.testing.expectEqual(Class.alu, metadata(extract).class);

    const replace = ir.Inst{ .op = .{ .i32x4_replace_lane = .{ .vector = 7, .val = 8, .lane = 1 } }, .dest = 11, .type = .v128 };
    try std.testing.expect(!metadata(replace).barrier);
    try std.testing.expectEqual(Class.alu, metadata(replace).class);

    const i16_binop = ir.Inst{ .op = .{ .i16x8_binop = .{ .op = .add, .lhs = 7, .rhs = 11 } }, .dest = 12, .type = .v128 };
    try std.testing.expect(!metadata(i16_binop).barrier);
    try std.testing.expectEqual(Class.alu, metadata(i16_binop).class);

    const i16_extract = ir.Inst{ .op = .{ .i16x8_extract_lane = .{ .vector = 12, .lane = 5, .sign = .unsigned } }, .dest = 13, .type = .i32 };
    try std.testing.expect(!metadata(i16_extract).barrier);
    try std.testing.expectEqual(Class.alu, metadata(i16_extract).class);
}

test "local scheduler prioritizes a long independent multiply chain" {
    const insts = [_]ir.Inst{
        .{ .op = .{ .iconst_32 = 10 }, .dest = 1, .type = .i32 },
        .{ .op = .{ .iconst_32 = 20 }, .dest = 2, .type = .i32 },
        .{ .op = .{ .add = .{ .lhs = 1, .rhs = 2 } }, .dest = 3, .type = .i32 },
        .{ .op = .{ .iconst_32 = 3 }, .dest = 4, .type = .i32 },
        .{ .op = .{ .iconst_32 = 4 }, .dest = 5, .type = .i32 },
        .{ .op = .{ .mul = .{ .lhs = 4, .rhs = 5 } }, .dest = 6, .type = .i32 },
        .{ .op = .{ .add = .{ .lhs = 6, .rhs = 3 } }, .dest = 7, .type = .i32 },
    };

    const scheduled = try scheduleBlock(&insts, std.testing.allocator, .{});
    defer std.testing.allocator.free(scheduled);

    try std.testing.expectEqual(@as(?ir.VReg, 4), scheduled[0].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 5), scheduled[1].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 6), scheduled[2].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 1), scheduled[3].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 2), scheduled[4].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 3), scheduled[5].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 7), scheduled[6].dest);
}

test "local scheduler keeps barriers fixed" {
    const insts = [_]ir.Inst{
        .{ .op = .{ .iconst_32 = 1 }, .dest = 1, .type = .i32 },
        .{ .op = .{ .local_get = 0 }, .dest = 2, .type = .i32 },
        .{ .op = .{ .iconst_32 = 2 }, .dest = 3, .type = .i32 },
        .{ .op = .{ .iconst_32 = 3 }, .dest = 4, .type = .i32 },
        .{ .op = .{ .mul = .{ .lhs = 3, .rhs = 4 } }, .dest = 5, .type = .i32 },
    };

    const scheduled = try scheduleBlock(&insts, std.testing.allocator, .{});
    defer std.testing.allocator.free(scheduled);

    try std.testing.expectEqual(@as(?ir.VReg, 1), scheduled[0].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 2), scheduled[1].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 3), scheduled[2].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 4), scheduled[3].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 5), scheduled[4].dest);
}

test "local scheduler can overlap load latency with independent ALU" {
    const insts = [_]ir.Inst{
        .{ .op = .{ .iconst_32 = 10 }, .dest = 1, .type = .i32 },
        .{ .op = .{ .iconst_32 = 20 }, .dest = 2, .type = .i32 },
        .{ .op = .{ .add = .{ .lhs = 1, .rhs = 2 } }, .dest = 3, .type = .i32 },
        .{ .op = .{ .iconst_32 = 0 }, .dest = 4, .type = .i32 },
        .{ .op = .{ .load = .{ .base = 4, .offset = 0, .size = 4 } }, .dest = 5, .type = .i32 },
        .{ .op = .{ .add = .{ .lhs = 5, .rhs = 3 } }, .dest = 6, .type = .i32 },
    };

    const scheduled = try scheduleBlock(&insts, std.testing.allocator, .{});
    defer std.testing.allocator.free(scheduled);

    try std.testing.expectEqual(@as(?ir.VReg, 4), scheduled[0].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 5), scheduled[1].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 1), scheduled[2].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 2), scheduled[3].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 3), scheduled[4].dest);
    try std.testing.expectEqual(@as(?ir.VReg, 6), scheduled[5].dest);
}

test "local scheduler preserves load trap order" {
    const insts = [_]ir.Inst{
        .{ .op = .{ .iconst_32 = 0 }, .dest = 1, .type = .i32 },
        .{ .op = .{ .load = .{ .base = 1, .offset = 0, .size = 4 } }, .dest = 2, .type = .i32 },
        .{ .op = .{ .iconst_32 = 4 }, .dest = 3, .type = .i32 },
        .{ .op = .{ .load = .{ .base = 3, .offset = 0, .size = 4 } }, .dest = 4, .type = .i32 },
        .{ .op = .{ .add = .{ .lhs = 4, .rhs = 2 } }, .dest = 5, .type = .i32 },
    };

    const scheduled = try scheduleBlock(&insts, std.testing.allocator, .{});
    defer std.testing.allocator.free(scheduled);

    var first_load_pos: ?usize = null;
    var second_load_pos: ?usize = null;
    for (scheduled, 0..) |inst, idx| {
        if (inst.dest == 2) first_load_pos = idx;
        if (inst.dest == 4) second_load_pos = idx;
    }

    try std.testing.expect(first_load_pos != null);
    try std.testing.expect(second_load_pos != null);
    try std.testing.expect(first_load_pos.? < second_load_pos.?);
}

fn findDest(insts: []const ir.Inst, dest: ir.VReg) ?usize {
    for (insts, 0..) |inst, idx| {
        if (inst.dest == dest) return idx;
    }
    return null;
}

fn findV128Store(insts: []const ir.Inst) ?usize {
    for (insts, 0..) |inst, idx| {
        switch (inst.op) {
            .v128_store => return idx,
            else => {},
        }
    }
    return null;
}

test "local scheduler models v128 dependencies while moving independent scalar work" {
    const insts = [_]ir.Inst{
        .{ .op = .{ .iconst_32 = 0 }, .dest = 1, .type = .i32 },
        .{ .op = .{ .v128_load = .{ .base = 1, .offset = 0, .alignment = 4 } }, .dest = 2, .type = .v128 },
        .{ .op = .{ .v128_not = 2 }, .dest = 3, .type = .v128 },
        .{ .op = .{ .i32x4_extract_lane = .{ .vector = 3, .lane = 0 } }, .dest = 4, .type = .i32 },
        .{ .op = .{ .iconst_32 = 10 }, .dest = 5, .type = .i32 },
        .{ .op = .{ .iconst_32 = 20 }, .dest = 6, .type = .i32 },
        .{ .op = .{ .mul = .{ .lhs = 5, .rhs = 6 } }, .dest = 7, .type = .i32 },
        .{ .op = .{ .add = .{ .lhs = 4, .rhs = 7 } }, .dest = 8, .type = .i32 },
    };

    const scheduled = try scheduleBlock(&insts, std.testing.allocator, .{});
    defer std.testing.allocator.free(scheduled);

    const load_pos = findDest(scheduled, 2).?;
    const not_pos = findDest(scheduled, 3).?;
    const extract_pos = findDest(scheduled, 4).?;
    const mul_pos = findDest(scheduled, 7).?;
    const add_pos = findDest(scheduled, 8).?;

    try std.testing.expect(load_pos < not_pos);
    try std.testing.expect(not_pos < extract_pos);
    try std.testing.expect(extract_pos < add_pos);
    try std.testing.expect(mul_pos < add_pos);
    try std.testing.expect(load_pos < mul_pos);
    try std.testing.expect(mul_pos < not_pos);
}

test "local scheduler preserves mixed scalar and v128 memory order" {
    const insts = [_]ir.Inst{
        .{ .op = .{ .iconst_32 = 0 }, .dest = 1, .type = .i32 },
        .{ .op = .{ .v128_load = .{ .base = 1, .offset = 0, .alignment = 4 } }, .dest = 2, .type = .v128 },
        .{ .op = .{ .iconst_32 = 16 }, .dest = 3, .type = .i32 },
        .{ .op = .{ .v128_store = .{ .base = 3, .offset = 0, .alignment = 4, .val = 2 } }, .type = .void },
        .{ .op = .{ .iconst_32 = 32 }, .dest = 4, .type = .i32 },
        .{ .op = .{ .load = .{ .base = 4, .offset = 0, .size = 4 } }, .dest = 5, .type = .i32 },
        .{ .op = .{ .add = .{ .lhs = 5, .rhs = 3 } }, .dest = 6, .type = .i32 },
    };

    const scheduled = try scheduleBlock(&insts, std.testing.allocator, .{});
    defer std.testing.allocator.free(scheduled);

    const v128_load_pos = findDest(scheduled, 2).?;
    const v128_store_pos = findV128Store(scheduled).?;
    const scalar_load_pos = findDest(scheduled, 5).?;

    try std.testing.expect(v128_load_pos < v128_store_pos);
    try std.testing.expect(v128_store_pos < scalar_load_pos);
}
