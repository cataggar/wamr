//! AOT compiler optimization passes.
//!
//! Operates on the SSA-form IR between frontend lowering and codegen.
//! Each pass transforms an IrFunction in-place and returns whether
//! it made any changes (for fixpoint iteration).

const std = @import("std");
const ir = @import("ir.zig");

// ── Use-Def Analysis ────────────────────────────────────────────────────────

/// Tracks which instructions define and use each VReg.
pub const UseDefInfo = struct {
    /// Index of the instruction that defines this VReg (in the block's instruction list).
    def_inst: ?usize = null,
    /// Number of instructions that use this VReg as an operand.
    use_count: u32 = 0,
};

/// Build use-def information for all VRegs in a function.
pub fn buildUseDef(func: *const ir.IrFunction, allocator: std.mem.Allocator) !std.AutoHashMap(ir.VReg, UseDefInfo) {
    var info = std.AutoHashMap(ir.VReg, UseDefInfo).init(allocator);

    for (func.blocks.items) |block| {
        for (block.instructions.items, 0..) |inst, idx| {
            // Record definition
            if (inst.dest) |dest| {
                const entry = try info.getOrPut(dest);
                if (!entry.found_existing) entry.value_ptr.* = .{};
                entry.value_ptr.def_inst = idx;
            }
            // Count uses from bounded list (most instructions)
            const used_vregs = getUsedVRegs(inst);
            for (used_vregs.slice()) |vreg| {
                const entry = try info.getOrPut(vreg);
                if (!entry.found_existing) entry.value_ptr.* = .{};
                entry.value_ptr.use_count += 1;
            }
            // Count uses from call args (unbounded)
            if (inst.op == .call) {
                for (inst.op.call.args) |vreg| {
                    const entry = try info.getOrPut(vreg);
                    if (!entry.found_existing) entry.value_ptr.* = .{};
                    entry.value_ptr.use_count += 1;
                }
            }
        }
    }
    return info;
}

/// Extract all VRegs used as operands by an instruction.
fn getUsedVRegs(inst: ir.Inst) BoundedVRegList {
    var list = BoundedVRegList{};
    switch (inst.op) {
        .iconst_32, .iconst_64, .fconst_32, .fconst_64 => {},
        .local_get, .global_get => {},
        .br, .@"unreachable" => {},

        // Binary ops
        .add, .sub, .mul, .div_s, .div_u, .rem_s, .rem_u,
        .@"and", .@"or", .xor, .shl, .shr_s, .shr_u, .rotl, .rotr,
        .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u,
        .f_min, .f_max, .f_copysign,
        => |bin| {
            list.append(bin.lhs);
            list.append(bin.rhs);
        },

        // Unary ops
        .clz, .ctz, .popcnt, .eqz, .wrap_i64, .extend_i32_s, .extend_i32_u,
        .extend8_s, .extend16_s, .extend32_s,
        .f_neg, .f_abs, .f_sqrt, .f_ceil, .f_floor, .f_trunc, .f_nearest,
        .trunc_f32_s, .trunc_f32_u, .trunc_f64_s, .trunc_f64_u,
        .convert_s, .convert_u, .demote_f64, .promote_f32, .reinterpret,
        .trunc_sat_f32_s, .trunc_sat_f32_u, .trunc_sat_f64_s, .trunc_sat_f64_u,
        => |vreg| list.append(vreg),

        .local_set => |ls| list.append(ls.val),
        .global_set => |gs| list.append(gs.val),
        .load => |ld| list.append(ld.base),
        .store => |st| {
            list.append(st.base);
            list.append(st.val);
        },
        .br_if => |bi| list.append(bi.cond),
        .br_table => |bt| list.append(bt.index),
        .ret => |maybe_vreg| if (maybe_vreg) |v| list.append(v),
        .call => {}, // call args handled separately in buildUseDef (unbounded)
        .call_indirect => {}, // same
        .select => |sel| {
            list.append(sel.cond);
            list.append(sel.if_true);
            list.append(sel.if_false);
        },

        // Atomic operations
        .atomic_fence => {},
        .atomic_load => |al| list.append(al.base),
        .atomic_store => |ast| {
            list.append(ast.base);
            list.append(ast.val);
        },
        .atomic_rmw => |ar| {
            list.append(ar.base);
            list.append(ar.val);
        },
        .atomic_cmpxchg => |ac| {
            list.append(ac.base);
            list.append(ac.expected);
            list.append(ac.replacement);
        },
        .atomic_notify => |an| {
            list.append(an.base);
            list.append(an.count);
        },
        .atomic_wait => |aw| {
            list.append(aw.base);
            list.append(aw.expected);
            list.append(aw.timeout);
        },
        .memory_copy => |mc| {
            list.append(mc.dst);
            list.append(mc.src);
            list.append(mc.len);
        },
        .memory_fill => |mf| {
            list.append(mf.dst);
            list.append(mf.val);
            list.append(mf.len);
        },
        .memory_size => {},
        .memory_grow => |pages| {
            list.append(pages);
        },
        .memory_init => |mi| {
            list.append(mi.dst);
            list.append(mi.src);
            list.append(mi.len);
        },
        .data_drop => {},
    }
    return list;
}

const BoundedVRegList = struct {
    items: [4]ir.VReg = undefined,
    len: u8 = 0,

    fn append(self: *BoundedVRegList, v: ir.VReg) void {
        if (self.len < 4) {
            self.items[self.len] = v;
            self.len += 1;
        }
    }

    fn slice(self: *const BoundedVRegList) []const ir.VReg {
        return self.items[0..self.len];
    }
};

/// Replace all uses of `old` VReg with `new` VReg in a function.
pub fn replaceVReg(func: *ir.IrFunction, old: ir.VReg, new: ir.VReg) void {
    for (func.blocks.items) |*block| {
        for (block.instructions.items) |*inst| {
            replaceInInst(inst, old, new);
        }
    }
}

fn replaceInInst(inst: *ir.Inst, old: ir.VReg, new: ir.VReg) void {
    switch (inst.op) {
        .iconst_32, .iconst_64, .fconst_32, .fconst_64,
        .local_get, .global_get, .br, .@"unreachable",
        => {},

        .add, .sub, .mul, .div_s, .div_u, .rem_s, .rem_u,
        .@"and", .@"or", .xor, .shl, .shr_s, .shr_u, .rotl, .rotr,
        .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u,
        .f_min, .f_max, .f_copysign,
        => |*bin| {
            if (bin.lhs == old) bin.lhs = new;
            if (bin.rhs == old) bin.rhs = new;
        },

        .clz, .ctz, .popcnt, .eqz, .wrap_i64, .extend_i32_s, .extend_i32_u,
        .extend8_s, .extend16_s, .extend32_s,
        .f_neg, .f_abs, .f_sqrt, .f_ceil, .f_floor, .f_trunc, .f_nearest,
        .trunc_f32_s, .trunc_f32_u, .trunc_f64_s, .trunc_f64_u,
        .convert_s, .convert_u, .demote_f64, .promote_f32, .reinterpret,
        .trunc_sat_f32_s, .trunc_sat_f32_u, .trunc_sat_f64_s, .trunc_sat_f64_u,
        => |*vreg| if (vreg.* == old) { vreg.* = new; },

        .local_set => |*ls| if (ls.val == old) { ls.val = new; },
        .global_set => |*gs| if (gs.val == old) { gs.val = new; },
        .load => |*ld| if (ld.base == old) { ld.base = new; },
        .store => |*st| {
            if (st.base == old) st.base = new;
            if (st.val == old) st.val = new;
        },
        .br_if => |*bi| if (bi.cond == old) { bi.cond = new; },
        .br_table => |*bt| if (bt.index == old) { bt.index = new; },
        .ret => |*maybe_vreg| if (maybe_vreg.*) |v| { if (v == old) maybe_vreg.* = new; },
        .call => |cl| {
            for (@constCast(cl.args)) |*arg| {
                if (arg.* == old) arg.* = new;
            }
        },
        .call_indirect => |ci| {
            if (ci.elem_idx == old) @constCast(&ci.elem_idx).* = new;
            for (@constCast(ci.args)) |*arg| {
                if (arg.* == old) arg.* = new;
            }
        },
        .select => |*sel| {
            if (sel.cond == old) sel.cond = new;
            if (sel.if_true == old) sel.if_true = new;
            if (sel.if_false == old) sel.if_false = new;
        },

        // Atomic operations
        .atomic_fence => {},
        .atomic_load => |*al| if (al.base == old) { al.base = new; },
        .atomic_store => |*ast| {
            if (ast.base == old) ast.base = new;
            if (ast.val == old) ast.val = new;
        },
        .atomic_rmw => |*ar| {
            if (ar.base == old) ar.base = new;
            if (ar.val == old) ar.val = new;
        },
        .atomic_cmpxchg => |*ac| {
            if (ac.base == old) ac.base = new;
            if (ac.expected == old) ac.expected = new;
            if (ac.replacement == old) ac.replacement = new;
        },
        .atomic_notify => |*an| {
            if (an.base == old) an.base = new;
            if (an.count == old) an.count = new;
        },
        .atomic_wait => |*aw| {
            if (aw.base == old) aw.base = new;
            if (aw.expected == old) aw.expected = new;
            if (aw.timeout == old) aw.timeout = new;
        },
        .memory_copy => |*mc| {
            if (mc.dst == old) mc.dst = new;
            if (mc.src == old) mc.src = new;
            if (mc.len == old) mc.len = new;
        },
        .memory_fill => |*mf| {
            if (mf.dst == old) mf.dst = new;
            if (mf.val == old) mf.val = new;
            if (mf.len == old) mf.len = new;
        },
        .memory_size => {},
        .memory_grow => |*pages| {
            if (pages.* == old) pages.* = new;
        },
        .memory_init => |*mi| {
            if (mi.dst == old) mi.dst = new;
            if (mi.src == old) mi.src = new;
            if (mi.len == old) mi.len = new;
        },
        .data_drop => {},
    }
}

// ── Constant Folding ────────────────────────────────────────────────────────

/// Evaluate operations on constant operands at compile time.
pub fn constantFold(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;
    var constants = std.AutoHashMap(ir.VReg, i64).init(allocator);
    defer constants.deinit();

    for (func.blocks.items) |*block| {
        for (block.instructions.items) |*inst| {
            switch (inst.op) {
                .iconst_32 => |v| if (inst.dest) |d| {
                    try constants.put(d, v);
                },
                .iconst_64 => |v| if (inst.dest) |d| {
                    try constants.put(d, v);
                },
                .add, .sub, .mul, .@"and", .@"or", .xor,
                .eq, .ne, .lt_s, .gt_s, .le_s, .ge_s,
                => |bin| {
                    const lhs = constants.get(bin.lhs) orelse continue;
                    const rhs = constants.get(bin.rhs) orelse continue;
                    const result = evalBinOp(inst.op, lhs, rhs) orelse continue;
                    if (inst.dest) |d| {
                        try constants.put(d, result);
                        if (inst.type == .i64) {
                            inst.op = .{ .iconst_64 = result };
                        } else {
                            inst.op = .{ .iconst_32 = @truncate(result) };
                        }
                        changed = true;
                    }
                },
                .eqz => |vreg| {
                    const val = constants.get(vreg) orelse continue;
                    const result: i64 = if (val == 0) 1 else 0;
                    if (inst.dest) |d| {
                        try constants.put(d, result);
                        inst.op = .{ .iconst_32 = @truncate(result) };
                        changed = true;
                    }
                },
                else => {},
            }
        }
    }
    return changed;
}

fn evalBinOp(op: ir.Inst.Op, lhs: i64, rhs: i64) ?i64 {
    return switch (op) {
        .add => lhs +% rhs,
        .sub => lhs -% rhs,
        .mul => lhs *% rhs,
        .@"and" => lhs & rhs,
        .@"or" => lhs | rhs,
        .xor => lhs ^ rhs,
        .eq => @intFromBool(lhs == rhs),
        .ne => @intFromBool(lhs != rhs),
        .lt_s => @intFromBool(lhs < rhs),
        .gt_s => @intFromBool(lhs > rhs),
        .le_s => @intFromBool(lhs <= rhs),
        .ge_s => @intFromBool(lhs >= rhs),
        else => null,
    };
}

// ── Dead Code Elimination ───────────────────────────────────────────────────

/// Remove instructions whose dest VReg is never used.
pub fn deadCodeElimination(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;
    var iterate = true;

    while (iterate) {
        iterate = false;
        var use_def = try buildUseDef(func, allocator);
        defer use_def.deinit();

        for (func.blocks.items) |*block| {
            var i: usize = 0;
            while (i < block.instructions.items.len) {
                const inst = block.instructions.items[i];
                if (inst.dest) |dest| {
                    if (!hasSideEffect(inst) and
                        (use_def.get(dest) orelse UseDefInfo{}).use_count == 0)
                    {
                        _ = block.instructions.orderedRemove(i);
                        changed = true;
                        iterate = true;
                        continue;
                    }
                }
                i += 1;
            }
        }
    }
    return changed;
}

fn hasSideEffect(inst: ir.Inst) bool {
    return switch (inst.op) {
        .store, .local_set, .global_set, .call, .call_indirect, .ret, .br, .br_if, .br_table, .@"unreachable",
        .atomic_fence, .atomic_load, .atomic_store, .atomic_rmw, .atomic_cmpxchg,
        .atomic_notify, .atomic_wait, .memory_copy, .memory_fill, .memory_grow,
        .memory_init, .data_drop,
        => true,
        else => false,
    };
}

// ── Common Subexpression Elimination ────────────────────────────────────────

/// Deduplicate identical pure operations.
pub fn commonSubexprElimination(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    _ = allocator;
    var changed = false;

    for (func.blocks.items) |*block| {
        var i: usize = 0;
        while (i < block.instructions.items.len) {
            const inst = &block.instructions.items[i];
            if (inst.dest == null or hasSideEffect(inst.*) or !isPure(inst.*)) {
                i += 1;
                continue;
            }

            // Look backwards for a matching instruction
            var j: usize = 0;
            while (j < i) : (j += 1) {
                const earlier = block.instructions.items[j];
                if (earlier.dest != null and sameOp(earlier, inst.*)) {
                    // Replace all uses of inst.dest with earlier.dest
                    replaceVReg(func, inst.dest.?, earlier.dest.?);
                    changed = true;
                    break;
                }
            }
            i += 1;
        }
    }
    return changed;
}

fn isPure(inst: ir.Inst) bool {
    return switch (inst.op) {
        .iconst_32, .iconst_64, .fconst_32, .fconst_64,
        .add, .sub, .mul, .div_s, .div_u, .rem_s, .rem_u,
        .@"and", .@"or", .xor, .shl, .shr_s, .shr_u, .rotl, .rotr,
        .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u,
        .clz, .ctz, .popcnt, .eqz,
        .wrap_i64, .extend_i32_s, .extend_i32_u,
        .extend8_s, .extend16_s, .extend32_s,
        .f_neg, .f_abs, .f_sqrt, .f_ceil, .f_floor, .f_trunc, .f_nearest,
        .f_min, .f_max, .f_copysign,
        .trunc_f32_s, .trunc_f32_u, .trunc_f64_s, .trunc_f64_u,
        .convert_s, .convert_u, .demote_f64, .promote_f32, .reinterpret,
        .trunc_sat_f32_s, .trunc_sat_f32_u, .trunc_sat_f64_s, .trunc_sat_f64_u,
        => true,
        else => false,
    };
}

fn sameOp(a: ir.Inst, b: ir.Inst) bool {
    const TagType = std.meta.Tag(ir.Inst.Op);
    if (@as(TagType, a.op) != @as(TagType, b.op)) return false;
    return switch (a.op) {
        .iconst_32 => |v| v == b.op.iconst_32,
        .iconst_64 => |v| v == b.op.iconst_64,
        .add => |bin| bin.lhs == b.op.add.lhs and bin.rhs == b.op.add.rhs,
        .sub => |bin| bin.lhs == b.op.sub.lhs and bin.rhs == b.op.sub.rhs,
        .mul => |bin| bin.lhs == b.op.mul.lhs and bin.rhs == b.op.mul.rhs,
        .@"and" => |bin| bin.lhs == b.op.@"and".lhs and bin.rhs == b.op.@"and".rhs,
        .@"or" => |bin| bin.lhs == b.op.@"or".lhs and bin.rhs == b.op.@"or".rhs,
        .xor => |bin| bin.lhs == b.op.xor.lhs and bin.rhs == b.op.xor.rhs,
        .shl => |bin| bin.lhs == b.op.shl.lhs and bin.rhs == b.op.shl.rhs,
        .shr_s => |bin| bin.lhs == b.op.shr_s.lhs and bin.rhs == b.op.shr_s.rhs,
        .shr_u => |bin| bin.lhs == b.op.shr_u.lhs and bin.rhs == b.op.shr_u.rhs,
        .eq => |bin| bin.lhs == b.op.eq.lhs and bin.rhs == b.op.eq.rhs,
        .ne => |bin| bin.lhs == b.op.ne.lhs and bin.rhs == b.op.ne.rhs,
        .eqz => |v| v == b.op.eqz,
        .clz => |v| v == b.op.clz,
        .ctz => |v| v == b.op.ctz,
        .popcnt => |v| v == b.op.popcnt,
        .extend8_s => |v| v == b.op.extend8_s,
        .extend16_s => |v| v == b.op.extend16_s,
        .extend32_s => |v| v == b.op.extend32_s,
        .f_neg => |v| v == b.op.f_neg,
        .f_abs => |v| v == b.op.f_abs,
        .f_sqrt => |v| v == b.op.f_sqrt,
        .f_ceil => |v| v == b.op.f_ceil,
        .f_floor => |v| v == b.op.f_floor,
        .f_trunc => |v| v == b.op.f_trunc,
        .f_nearest => |v| v == b.op.f_nearest,
        .wrap_i64 => |v| v == b.op.wrap_i64,
        .extend_i32_s => |v| v == b.op.extend_i32_s,
        .extend_i32_u => |v| v == b.op.extend_i32_u,
        .trunc_f32_s => |v| v == b.op.trunc_f32_s,
        .trunc_f32_u => |v| v == b.op.trunc_f32_u,
        .trunc_f64_s => |v| v == b.op.trunc_f64_s,
        .trunc_f64_u => |v| v == b.op.trunc_f64_u,
        .convert_s => |v| v == b.op.convert_s,
        .convert_u => |v| v == b.op.convert_u,
        .demote_f64 => |v| v == b.op.demote_f64,
        .promote_f32 => |v| v == b.op.promote_f32,
        .reinterpret => |v| v == b.op.reinterpret,
        .trunc_sat_f32_s => |v| v == b.op.trunc_sat_f32_s,
        .trunc_sat_f32_u => |v| v == b.op.trunc_sat_f32_u,
        .trunc_sat_f64_s => |v| v == b.op.trunc_sat_f64_s,
        .trunc_sat_f64_u => |v| v == b.op.trunc_sat_f64_u,
        else => false,
    };
}

// ── Pass Manager ────────────────────────────────────────────────────────────

pub const PassFn = *const fn (*ir.IrFunction, std.mem.Allocator) anyerror!bool;

/// Run a sequence of optimization passes on an IR module.
pub fn runPasses(module: *ir.IrModule, passes: []const PassFn, allocator: std.mem.Allocator) !u32 {
    var total_changes: u32 = 0;
    for (module.functions.items) |*func| {
        for (passes) |pass| {
            if (try pass(func, allocator)) total_changes += 1;
        }
    }
    return total_changes;
}

/// The default optimization pipeline.
/// Note: deadCodeElimination is excluded because the stack-based x86-64
/// backend does not track call argument VRegs, so DCE incorrectly removes
/// instructions that compute values consumed implicitly via the operand stack.
pub const default_passes: []const PassFn = &.{
    &constantFold,
    &commonSubexprElimination,
};

// ── Tests ───────────────────────────────────────────────────────────────────

test "constantFold: iconst + iconst + add → iconst" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 4 }, .dest = v1 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const changed = try constantFold(&func, allocator);
    try std.testing.expect(changed);

    // The add should now be iconst_32(7)
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 7 }, block.instructions.items[2].op);
}

test "constantFold: eqz on constant" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0 });
    try block.append(.{ .op = .{ .eqz = v0 }, .dest = v1 });

    const changed = try constantFold(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 1 }, block.instructions.items[1].op);
}

test "DCE: removes unused iconst" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v0 = func.newVReg();
    const v1 = func.newVReg(); // unused
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 99 }, .dest = v1 });
    try block.append(.{ .op = .{ .ret = v0 } });

    try std.testing.expectEqual(@as(usize, 3), block.instructions.items.len);
    const changed = try deadCodeElimination(&func, allocator);
    try std.testing.expect(changed);
    // v1 (iconst 99) should be removed
    try std.testing.expectEqual(@as(usize, 2), block.instructions.items.len);
}

test "DCE: preserves side-effect instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v0 });
    try block.append(.{ .op = .{ .global_set = .{ .idx = 0, .val = v0 } } }); // side effect
    try block.append(.{ .op = .{ .ret = null } });

    const changed = try deadCodeElimination(&func, allocator);
    try std.testing.expect(!changed); // nothing should be removed
    try std.testing.expectEqual(@as(usize, 3), block.instructions.items.len);
}

test "CSE: deduplicates identical add" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg(); // add(v0, v1)
    const v3 = func.newVReg(); // add(v0, v1) — duplicate
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v3 });
    try block.append(.{ .op = .{ .ret = v3 } });

    const changed = try commonSubexprElimination(&func, allocator);
    try std.testing.expect(changed);

    // The ret should now reference v2 instead of v3
    try std.testing.expectEqual(ir.Inst.Op{ .ret = v2 }, block.instructions.items[4].op);
}

test "combined pipeline: fold + DCE" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    const v3 = func.newVReg(); // unused
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 4 }, .dest = v1 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try block.append(.{ .op = .{ .iconst_32 = 999 }, .dest = v3 }); // dead
    try block.append(.{ .op = .{ .ret = v2 } });

    // Fold: add(3,4) → 7
    _ = try constantFold(&func, allocator);
    // DCE: remove unused v0, v1, v3
    _ = try deadCodeElimination(&func, allocator);

    // Should have: iconst_32(7); ret v2
    try std.testing.expectEqual(@as(usize, 2), block.instructions.items.len);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 7 }, block.instructions.items[0].op);
}

test "buildUseDef: counts uses correctly" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v0 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v0 } }, .dest = v1 }); // v0 used twice
    try block.append(.{ .op = .{ .ret = v1 } });

    var use_def = try buildUseDef(&func, allocator);
    defer use_def.deinit();

    try std.testing.expectEqual(@as(u32, 2), use_def.get(v0).?.use_count); // used twice in add
    try std.testing.expectEqual(@as(u32, 1), use_def.get(v1).?.use_count); // used once in ret
}

test "replaceVReg: updates all uses" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try block.append(.{ .op = .{ .ret = v2 } });

    replaceVReg(&func, v0, v1); // replace v0 with v1

    const add = block.instructions.items[2].op.add;
    try std.testing.expectEqual(v1, add.lhs); // was v0, now v1
    try std.testing.expectEqual(v1, add.rhs); // was already v1
}
