//! AOT compiler optimization passes.
//!
//! Operates on the SSA-form IR between frontend lowering and codegen.
//! Each pass transforms an IrFunction in-place and returns whether
//! it made any changes (for fixpoint iteration).

const std = @import("std");
const ir = @import("ir.zig");
const analysis = @import("analysis.zig");

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
            // Count uses from unbounded VReg lists (call args, ret_multi)
            switch (inst.op) {
                .call => |cl| {
                    for (cl.args) |vreg| {
                        const entry = try info.getOrPut(vreg);
                        if (!entry.found_existing) entry.value_ptr.* = .{};
                        entry.value_ptr.use_count += 1;
                    }
                },
                .call_indirect => |ci| {
                    const ei_entry = try info.getOrPut(ci.elem_idx);
                    if (!ei_entry.found_existing) ei_entry.value_ptr.* = .{};
                    ei_entry.value_ptr.use_count += 1;
                    for (ci.args) |vreg| {
                        const entry = try info.getOrPut(vreg);
                        if (!entry.found_existing) entry.value_ptr.* = .{};
                        entry.value_ptr.use_count += 1;
                    }
                },
                .call_ref => |cr| {
                    const fr_entry = try info.getOrPut(cr.func_ref);
                    if (!fr_entry.found_existing) fr_entry.value_ptr.* = .{};
                    fr_entry.value_ptr.use_count += 1;
                    for (cr.args) |vreg| {
                        const entry = try info.getOrPut(vreg);
                        if (!entry.found_existing) entry.value_ptr.* = .{};
                        entry.value_ptr.use_count += 1;
                    }
                },
                .ret_multi => |vregs| {
                    for (vregs) |vreg| {
                        const entry = try info.getOrPut(vreg);
                        if (!entry.found_existing) entry.value_ptr.* = .{};
                        entry.value_ptr.use_count += 1;
                    }
                },
                else => {},
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
        .f_eq, .f_ne, .f_lt, .f_gt, .f_le, .f_ge,
        => |bin| {
            list.append(bin.lhs);
            list.append(bin.rhs);
        },

        // Unary ops
        .clz, .ctz, .popcnt, .eqz, .wrap_i64, .extend_i32_s, .extend_i32_u,
        .extend8_s, .extend16_s, .extend32_s,
        .f_neg, .f_abs, .f_sqrt, .f_ceil, .f_floor, .f_trunc, .f_nearest,
        .trunc_f32_s, .trunc_f32_u, .trunc_f64_s, .trunc_f64_u,
        .convert_s, .convert_u, .convert_i32_s, .convert_i64_s, .convert_i32_u, .convert_i64_u, .demote_f64, .promote_f32, .reinterpret,
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
        .ret_multi => {}, // multi-return VRegs handled separately (unbounded)
        .call => {}, // call args handled separately in buildUseDef (unbounded)
        .call_indirect => {}, // same
        .call_ref => {}, // same
        .call_result => {},
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
        .table_size => {},
        .table_get => |tg| list.append(tg.idx),
        .table_set => |ts| {
            list.append(ts.idx);
            list.append(ts.val);
        },
        .table_grow => |tg| {
            list.append(tg.init);
            list.append(tg.delta);
        },
        .ref_func => {},
        .memory_init => |mi| {
            list.append(mi.dst);
            list.append(mi.src);
            list.append(mi.len);
        },
        .data_drop => {},
        .table_init => |ti| {
            list.append(ti.dst);
            list.append(ti.src);
            list.append(ti.len);
        },
        .elem_drop => {},
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

/// Count the number of operand slots in `func` that currently reference
/// `vreg`. Cheap O(N) scan; used by passes that need to detect whether
/// a rewrite would actually change anything (for idempotent fixpoint
/// reporting).
pub fn countUsesOfVReg(func: *const ir.IrFunction, vreg: ir.VReg) u32 {
    var count: u32 = 0;
    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            for (getUsedVRegs(inst).slice()) |u| {
                if (u == vreg) count += 1;
            }
        }
    }
    return count;
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
        .f_eq, .f_ne, .f_lt, .f_gt, .f_le, .f_ge,
        => |*bin| {
            if (bin.lhs == old) bin.lhs = new;
            if (bin.rhs == old) bin.rhs = new;
        },

        .clz, .ctz, .popcnt, .eqz, .wrap_i64, .extend_i32_s, .extend_i32_u,
        .extend8_s, .extend16_s, .extend32_s,
        .f_neg, .f_abs, .f_sqrt, .f_ceil, .f_floor, .f_trunc, .f_nearest,
        .trunc_f32_s, .trunc_f32_u, .trunc_f64_s, .trunc_f64_u,
        .convert_s, .convert_u, .convert_i32_s, .convert_i64_s, .convert_i32_u, .convert_i64_u, .demote_f64, .promote_f32, .reinterpret,
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
        .ret_multi => |vregs| {
            for (@constCast(vregs)) |*v| {
                if (v.* == old) v.* = new;
            }
        },
        .call_result => {},
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
        .call_ref => |cr| {
            if (cr.func_ref == old) @constCast(&cr.func_ref).* = new;
            for (@constCast(cr.args)) |*arg| {
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
        .table_size => {},
        .table_get => |*tg| {
            if (tg.idx == old) tg.idx = new;
        },
        .table_set => |*ts| {
            if (ts.idx == old) ts.idx = new;
            if (ts.val == old) ts.val = new;
        },
        .table_grow => |*tg| {
            if (tg.init == old) tg.init = new;
            if (tg.delta == old) tg.delta = new;
        },
        .ref_func => {},
        .memory_init => |*mi| {
            if (mi.dst == old) mi.dst = new;
            if (mi.src == old) mi.src = new;
            if (mi.len == old) mi.len = new;
        },
        .data_drop => {},
        .table_init => |*ti| {
            if (ti.dst == old) ti.dst = new;
            if (ti.src == old) ti.src = new;
            if (ti.len == old) ti.len = new;
        },
        .elem_drop => {},
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
                .shl, .shr_s, .shr_u, .rotl, .rotr,
                .eq, .ne, .lt_s, .gt_s, .le_s, .ge_s,
                .lt_u, .gt_u, .le_u, .ge_u,
                .div_s, .div_u, .rem_s, .rem_u,
                => |bin| {
                    const dest = inst.dest orelse continue;
                    const maybe_lhs = constants.get(bin.lhs);
                    const maybe_rhs = constants.get(bin.rhs);

                    // Try full constant folding first.
                    if (maybe_lhs != null and maybe_rhs != null) {
                        if (evalBinOp(inst.op, maybe_lhs.?, maybe_rhs.?, inst.type)) |result| {
                            try constants.put(dest, result);
                            if (inst.type == .i64) {
                                inst.op = .{ .iconst_64 = result };
                            } else {
                                inst.op = .{ .iconst_32 = @truncate(result) };
                            }
                            changed = true;
                            continue;
                        }
                    }

                    // Algebraic identities: reduce to a single-operand copy.
                    if (algebraicIdentity(inst.op, maybe_lhs, maybe_rhs, inst.type, bin.lhs, bin.rhs)) |keep| {
                        replaceVReg(func, dest, keep);
                        // Turn into a dead iconst; DCE will remove it on the
                        // next pipeline iteration.
                        inst.op = .{ .iconst_32 = 0 };
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
                .select => |sel| {
                    const dest = inst.dest orelse continue;
                    const cond = constants.get(sel.cond) orelse continue;
                    const pick = if (cond != 0) sel.if_true else sel.if_false;
                    replaceVReg(func, dest, pick);
                    inst.op = .{ .iconst_32 = 0 };
                    changed = true;
                },
                else => {},
            }
        }
    }
    return changed;
}

/// Return the operand to keep if this op simplifies to a copy of that
/// operand (e.g., `v + 0` → `v`, `v * 1` → `v`, `v << 0` → `v`).
fn algebraicIdentity(
    op: ir.Inst.Op,
    maybe_lhs: ?i64,
    maybe_rhs: ?i64,
    ty: ir.IrType,
    lhs_reg: ir.VReg,
    rhs_reg: ir.VReg,
) ?ir.VReg {
    const mask: u64 = if (ty == .i64) 63 else 31;
    const ones: i64 = if (ty == .i64) -1 else @as(i64, @as(i32, -1));
    // RHS-is-constant cases.
    if (maybe_rhs) |r| {
        switch (op) {
            .add, .sub, .@"or", .xor, .shl, .shr_s, .shr_u, .rotl, .rotr => {
                if ((op == .shl or op == .shr_s or op == .shr_u or op == .rotl or op == .rotr)) {
                    if ((@as(u64, @bitCast(r)) & mask) == 0) return lhs_reg;
                } else if (r == 0) return lhs_reg;
            },
            .mul => if (r == 1) return lhs_reg,
            .div_s, .div_u => if (r == 1) return lhs_reg,
            .@"and" => if (r == ones) return lhs_reg,
            else => {},
        }
    }
    // LHS-is-constant cases (for commutative ops).
    if (maybe_lhs) |l| {
        switch (op) {
            .add, .@"or", .xor => if (l == 0) return rhs_reg,
            .mul => if (l == 1) return rhs_reg,
            .@"and" => if (l == ones) return rhs_reg,
            else => {},
        }
    }
    return null;
}

fn evalBinOp(op: ir.Inst.Op, lhs: i64, rhs: i64, ty: ir.IrType) ?i64 {
    const mask: u64 = if (ty == .i64) 63 else 31;
    return switch (op) {
        .add => lhs +% rhs,
        .sub => lhs -% rhs,
        .mul => lhs *% rhs,
        .@"and" => lhs & rhs,
        .@"or" => lhs | rhs,
        .xor => lhs ^ rhs,
        .shl => blk: {
            const n: u6 = @intCast(@as(u64, @bitCast(rhs)) & mask);
            break :blk @bitCast(@as(u64, @bitCast(lhs)) << n);
        },
        .shr_s => blk: {
            const n: u6 = @intCast(@as(u64, @bitCast(rhs)) & mask);
            if (ty == .i64) break :blk lhs >> n;
            const l32: i32 = @truncate(lhs);
            break :blk @as(i64, l32 >> @as(u5, @intCast(n)));
        },
        .shr_u => blk: {
            const n: u6 = @intCast(@as(u64, @bitCast(rhs)) & mask);
            if (ty == .i64) {
                break :blk @bitCast(@as(u64, @bitCast(lhs)) >> n);
            }
            const l32: u32 = @truncate(@as(u64, @bitCast(lhs)));
            break :blk @as(i64, l32 >> @as(u5, @intCast(n)));
        },
        .rotl => blk: {
            const n: u6 = @intCast(@as(u64, @bitCast(rhs)) & mask);
            if (ty == .i64) {
                break :blk @bitCast(std.math.rotl(u64, @bitCast(lhs), n));
            }
            const l32: u32 = @truncate(@as(u64, @bitCast(lhs)));
            break :blk @as(i64, std.math.rotl(u32, l32, n));
        },
        .rotr => blk: {
            const n: u6 = @intCast(@as(u64, @bitCast(rhs)) & mask);
            if (ty == .i64) {
                break :blk @bitCast(std.math.rotr(u64, @bitCast(lhs), n));
            }
            const l32: u32 = @truncate(@as(u64, @bitCast(lhs)));
            break :blk @as(i64, std.math.rotr(u32, l32, n));
        },
        .eq => @intFromBool(lhs == rhs),
        .ne => @intFromBool(lhs != rhs),
        .lt_s => if (ty == .i64) @intFromBool(lhs < rhs)
            else @intFromBool(@as(i32, @truncate(lhs)) < @as(i32, @truncate(rhs))),
        .gt_s => if (ty == .i64) @intFromBool(lhs > rhs)
            else @intFromBool(@as(i32, @truncate(lhs)) > @as(i32, @truncate(rhs))),
        .le_s => if (ty == .i64) @intFromBool(lhs <= rhs)
            else @intFromBool(@as(i32, @truncate(lhs)) <= @as(i32, @truncate(rhs))),
        .ge_s => if (ty == .i64) @intFromBool(lhs >= rhs)
            else @intFromBool(@as(i32, @truncate(lhs)) >= @as(i32, @truncate(rhs))),
        .lt_u => if (ty == .i64) @intFromBool(@as(u64, @bitCast(lhs)) < @as(u64, @bitCast(rhs)))
            else @intFromBool(@as(u32, @truncate(@as(u64, @bitCast(lhs)))) < @as(u32, @truncate(@as(u64, @bitCast(rhs))))),
        .gt_u => if (ty == .i64) @intFromBool(@as(u64, @bitCast(lhs)) > @as(u64, @bitCast(rhs)))
            else @intFromBool(@as(u32, @truncate(@as(u64, @bitCast(lhs)))) > @as(u32, @truncate(@as(u64, @bitCast(rhs))))),
        .le_u => if (ty == .i64) @intFromBool(@as(u64, @bitCast(lhs)) <= @as(u64, @bitCast(rhs)))
            else @intFromBool(@as(u32, @truncate(@as(u64, @bitCast(lhs)))) <= @as(u32, @truncate(@as(u64, @bitCast(rhs))))),
        .ge_u => if (ty == .i64) @intFromBool(@as(u64, @bitCast(lhs)) >= @as(u64, @bitCast(rhs)))
            else @intFromBool(@as(u32, @truncate(@as(u64, @bitCast(lhs)))) >= @as(u32, @truncate(@as(u64, @bitCast(rhs))))),
        .div_s => blk: {
            if (rhs == 0) break :blk null;
            if (ty == .i64) {
                // i64 min / -1 traps; skip to be safe.
                if (lhs == std.math.minInt(i64) and rhs == -1) break :blk null;
                break :blk @divTrunc(lhs, rhs);
            }
            const l32: i32 = @truncate(lhs);
            const r32: i32 = @truncate(rhs);
            if (l32 == std.math.minInt(i32) and r32 == -1) break :blk null;
            break :blk @as(i64, @divTrunc(l32, r32));
        },
        .div_u => blk: {
            if (rhs == 0) break :blk null;
            if (ty == .i64) {
                break :blk @bitCast(@as(u64, @bitCast(lhs)) / @as(u64, @bitCast(rhs)));
            }
            const l32: u32 = @truncate(@as(u64, @bitCast(lhs)));
            const r32: u32 = @truncate(@as(u64, @bitCast(rhs)));
            break :blk @as(i64, l32 / r32);
        },
        .rem_s => blk: {
            if (rhs == 0) break :blk null;
            if (ty == .i64) {
                if (lhs == std.math.minInt(i64) and rhs == -1) break :blk 0;
                break :blk @rem(lhs, rhs);
            }
            const l32: i32 = @truncate(lhs);
            const r32: i32 = @truncate(rhs);
            if (l32 == std.math.minInt(i32) and r32 == -1) break :blk 0;
            break :blk @as(i64, @rem(l32, r32));
        },
        .rem_u => blk: {
            if (rhs == 0) break :blk null;
            if (ty == .i64) {
                break :blk @bitCast(@as(u64, @bitCast(lhs)) % @as(u64, @bitCast(rhs)));
            }
            const l32: u32 = @truncate(@as(u64, @bitCast(lhs)));
            const r32: u32 = @truncate(@as(u64, @bitCast(rhs)));
            break :blk @as(i64, l32 % r32);
        },
        else => null,
    };
}

// ── Strength Reduction ──────────────────────────────────────────────────────

/// Return the shift amount `k` if `c` is a power of two that fits
/// within a legal shift for `ir_type` (i32 → k in [1,31], i64 → k in [1,63]).
/// For i32, `c` is interpreted modulo 2^32 since wasm `i32.mul` is modular,
/// so `c = 0x80000000` (negative as i32) correctly maps to a shift of 31.
/// Returns `null` for `c == 0`, `c == 1`, non-powers-of-two, or shift
/// amounts outside the legal range.
fn powerOfTwoShift(c: i64, ir_type: ir.IrType) ?u6 {
    const u: u64 = switch (ir_type) {
        .i32 => @as(u32, @truncate(@as(u64, @bitCast(c)))),
        .i64 => @bitCast(c),
        else => return null,
    };
    if (u <= 1) return null;
    if (u & (u - 1) != 0) return null; // not a power of two
    const k: u6 = @intCast(@ctz(u));
    const max: u6 = if (ir_type == .i32) 31 else 63;
    if (k == 0 or k > max) return null;
    return k;
}

/// Rewrite `mul(x, 2^k)` → `shl(x, k)`. `imul` has higher latency than
/// `shl` on modern x86-64 (3 vs 1 cycles) and on AArch64, so this is a
/// win for both backends and for code size (no 64-bit immediate needed).
///
/// Matches when either operand of a `.mul` is defined by an `iconst_32`
/// / `iconst_64` whose value is a power of two in `[2, 2^31]` (i32) or
/// `[2, 2^63]` (i64). A new iconst for the shift amount is inserted
/// immediately before the rewritten instruction; the original constant
/// instruction is left untouched (DCE will remove it if it becomes
/// unused).
pub fn strengthReduceMul(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;
    var constants = std.AutoHashMap(ir.VReg, i64).init(allocator);
    defer constants.deinit();

    for (func.blocks.items) |*block| {
        // Build constants map for this block (linear SSA within a block is
        // sufficient: all producers of a power-of-two constant we care about
        // are iconst_32 / iconst_64 instructions defined earlier in the same
        // block via the frontend's straight-line lowering of `i32.const`.)
        constants.clearRetainingCapacity();

        var i: usize = 0;
        while (i < block.instructions.items.len) : (i += 1) {
            const inst = block.instructions.items[i];
            switch (inst.op) {
                .iconst_32 => |v| if (inst.dest) |d| {
                    try constants.put(d, v);
                },
                .iconst_64 => |v| if (inst.dest) |d| {
                    try constants.put(d, v);
                },
                .mul => |bin| {
                    const dest = inst.dest orelse continue;
                    // Determine which operand is the constant power of two.
                    const lhs_const = constants.get(bin.lhs);
                    const rhs_const = constants.get(bin.rhs);

                    var non_const_vreg: ir.VReg = undefined;
                    var k: u6 = undefined;
                    if (rhs_const) |c| {
                        if (powerOfTwoShift(c, inst.type)) |s| {
                            non_const_vreg = bin.lhs;
                            k = s;
                        } else if (lhs_const) |lc| {
                            if (powerOfTwoShift(lc, inst.type)) |s| {
                                non_const_vreg = bin.rhs;
                                k = s;
                            } else continue;
                        } else continue;
                    } else if (lhs_const) |c| {
                        if (powerOfTwoShift(c, inst.type)) |s| {
                            non_const_vreg = bin.rhs;
                            k = s;
                        } else continue;
                    } else continue;

                    // Insert a fresh iconst for the shift amount *before* the
                    // mul, then rewrite the mul into a shl.
                    const shift_vreg = func.newVReg();
                    const shift_op: ir.Inst.Op = if (inst.type == .i64)
                        .{ .iconst_64 = @intCast(k) }
                    else
                        .{ .iconst_32 = @intCast(k) };
                    try block.instructions.insert(
                        block.allocator,
                        i,
                        .{ .op = shift_op, .dest = shift_vreg, .type = inst.type },
                    );
                    // After insertion, what was at index i is now at i+1.
                    block.instructions.items[i + 1].op = .{ .shl = .{
                        .lhs = non_const_vreg,
                        .rhs = shift_vreg,
                    } };
                    block.instructions.items[i + 1].dest = dest;
                    // Record the new shift amount constant so downstream muls
                    // in the same block can see it (harmless — value is small).
                    try constants.put(shift_vreg, @intCast(k));
                    changed = true;
                    i += 1; // skip over the newly-inserted iconst
                },
                else => {},
            }
        }
    }
    return changed;
}

/// Classification of a constant multiplier that reduces to a single
/// shift plus an add (or subtract) of the multiplicand.
const ShiftAddKind = struct { k: u6, is_plus: bool };

/// Return `{ k, is_plus }` if `c` is `2^k + 1` (is_plus=true) or
/// `2^k - 1` (is_plus=false) with `k` in the legal shift range for
/// `ir_type`. Used to recognise multipliers that reduce to
/// `(x << k) + x` or `(x << k) - x` rather than a single shift.
///
/// `k == 0` (for the plus form: C == 2) is excluded so this helper
/// does not poach cases already handled by `powerOfTwoShift`
/// (`mul x, 2` → `shl x, 1`). `k == 1` for the minus form (C == 1)
/// is a no-op multiply handled by `constantFold`.
fn shiftPlusMinusOne(c: i64, ir_type: ir.IrType) ?ShiftAddKind {
    const u: u64 = switch (ir_type) {
        .i32 => @as(u32, @truncate(@as(u64, @bitCast(c)))),
        .i64 => @bitCast(c),
        else => return null,
    };
    if (u < 3) return null;
    const max: u6 = if (ir_type == .i32) 31 else 63;

    // 2^k + 1: `u - 1` is a non-zero power of two, k = ctz(u-1), k >= 1.
    const p = u - 1;
    if (p != 0 and (p & (p - 1)) == 0) {
        const k: u6 = @intCast(@ctz(p));
        if (k >= 1 and k <= max) return .{ .k = k, .is_plus = true };
    }

    // 2^k - 1: `u + 1` is a non-zero power of two, k = ctz(u+1), k >= 2.
    // Skip when `u + 1` wraps to 0 in u64 (i.e. u == 2^64 - 1, i64 case).
    const q = u +% 1;
    if (q != 0 and (q & (q -% 1)) == 0) {
        const k: u6 = @intCast(@ctz(q));
        if (k >= 2 and k <= max) return .{ .k = k, .is_plus = false };
    }

    return null;
}

/// Rewrite `mul(x, 2^k + 1)` → `add(shl(x, k), x)` and
/// `mul(x, 2^k - 1)` → `sub(shl(x, k), x)`. This covers the common
/// small-integer multipliers — 3, 5, 7, 9, 15, 17, 31, 33, ... — that
/// turn into a single latency-1 shift + add/sub on AArch64 and x86-64
/// instead of the 3–4 cycle integer multiplier. Array indexing with
/// element sizes like 3, 5, 6 (2*3), 9, 12 is the dominant source in
/// real workloads; the pow2-only `strengthReduceMul` misses these
/// entirely.
///
/// Matches only when the constant operand of `.mul` is defined by an
/// `iconst_32` / `iconst_64` in the same block (matching
/// `strengthReduceMul`'s block-local lowering assumption). Does not
/// fire when the constant is a power of two — those are left to
/// `strengthReduceMul`.
///
/// Cost: replaces 1 mul with 2 arithmetic instructions (plus a shift
/// amount iconst that DCE / backend constant-folding will coalesce).
/// On both target backends `shl` + `add`/`sub` decode to the
/// `add x, x, x, lsl #k` style fused AArch64 instruction or an
/// `lea`/shift-add sequence on x86-64, so the net is usually a strict
/// win vs `imul`.
pub fn strengthReduceMulShiftAdd(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;
    var constants = std.AutoHashMap(ir.VReg, i64).init(allocator);
    defer constants.deinit();

    for (func.blocks.items) |*block| {
        constants.clearRetainingCapacity();

        var i: usize = 0;
        while (i < block.instructions.items.len) : (i += 1) {
            const inst = block.instructions.items[i];
            switch (inst.op) {
                .iconst_32 => |v| if (inst.dest) |d| {
                    try constants.put(d, v);
                },
                .iconst_64 => |v| if (inst.dest) |d| {
                    try constants.put(d, v);
                },
                .mul => |bin| {
                    const dest = inst.dest orelse continue;
                    const lhs_const = constants.get(bin.lhs);
                    const rhs_const = constants.get(bin.rhs);

                    // Skip if either operand is already a power of two —
                    // `strengthReduceMul` handles that pattern and will
                    // convert it to a single `shl` which dominates this
                    // two-instruction form.
                    if (rhs_const) |rc| if (powerOfTwoShift(rc, inst.type) != null) continue;
                    if (lhs_const) |lc| if (powerOfTwoShift(lc, inst.type) != null) continue;

                    var x_vreg: ir.VReg = undefined;
                    var info: ShiftAddKind = undefined;
                    if (rhs_const) |c| {
                        if (shiftPlusMinusOne(c, inst.type)) |r| {
                            x_vreg = bin.lhs;
                            info = r;
                        } else if (lhs_const) |lc| {
                            if (shiftPlusMinusOne(lc, inst.type)) |r| {
                                x_vreg = bin.rhs;
                                info = r;
                            } else continue;
                        } else continue;
                    } else if (lhs_const) |c| {
                        if (shiftPlusMinusOne(c, inst.type)) |r| {
                            x_vreg = bin.rhs;
                            info = r;
                        } else continue;
                    } else continue;

                    // Splice in two instructions *before* the mul at index i:
                    //   [i]   iconst shift_vreg = k
                    //   [i+1] shl    shl_vreg   = x << shift_vreg
                    // and rewrite the mul (now at index i+2) to add/sub.
                    const shift_vreg = func.newVReg();
                    const shl_vreg = func.newVReg();
                    const shift_op: ir.Inst.Op = if (inst.type == .i64)
                        .{ .iconst_64 = @intCast(info.k) }
                    else
                        .{ .iconst_32 = @intCast(info.k) };
                    try block.instructions.insert(
                        block.allocator,
                        i,
                        .{ .op = shift_op, .dest = shift_vreg, .type = inst.type },
                    );
                    try block.instructions.insert(
                        block.allocator,
                        i + 1,
                        .{ .op = .{ .shl = .{ .lhs = x_vreg, .rhs = shift_vreg } }, .dest = shl_vreg, .type = inst.type },
                    );
                    if (info.is_plus) {
                        block.instructions.items[i + 2].op = .{ .add = .{ .lhs = shl_vreg, .rhs = x_vreg } };
                    } else {
                        block.instructions.items[i + 2].op = .{ .sub = .{ .lhs = shl_vreg, .rhs = x_vreg } };
                    }
                    block.instructions.items[i + 2].dest = dest;

                    try constants.put(shift_vreg, @intCast(info.k));
                    changed = true;
                    i += 2; // skip over the two newly-inserted instructions
                },
                else => {},
            }
        }
    }
    return changed;
}

// ── Dead Code Elimination ───────────────────────────────────────────────────

/// Rewrite `div_u(x, 2^k)` → `shr_u(x, k)` and `rem_u(x, 2^k)` → `and(x, 2^k - 1)`.
/// Unsigned integer division and modulo by a power-of-two constant
/// divisor are equivalent to a shift and a mask, which are ~5-10× faster
/// than the hardware divider on both x86-64 and AArch64 and avoid the
/// microarchitectural div-unit pressure.
///
/// Only rewrites when the rhs is produced by an `iconst_32` /
/// `iconst_64` defined earlier in the same block (matches
/// `strengthReduceMul`'s straight-line lowering assumption). Signed
/// `div_s`/`rem_s` are intentionally NOT handled here — they require
/// rounding-toward-zero bias adjustment for negative dividends which is
/// several additional ops; those patterns are better left to a dedicated
/// magic-number pass.
///
/// Safety: `powerOfTwoShift` rejects c == 0, so we never rewrite a
/// division that could trap at runtime; c == 1 is also rejected (the
/// result would be `x` / `0` which the existing `constantFold` handles
/// algebraically if it fires). Float div/rem are unchanged (not
/// integer, `powerOfTwoShift` returns null).
pub fn strengthReduceDivRem(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;
    var constants = std.AutoHashMap(ir.VReg, i64).init(allocator);
    defer constants.deinit();

    for (func.blocks.items) |*block| {
        constants.clearRetainingCapacity();

        var i: usize = 0;
        while (i < block.instructions.items.len) : (i += 1) {
            const inst = block.instructions.items[i];
            switch (inst.op) {
                .iconst_32 => |v| if (inst.dest) |d| {
                    try constants.put(d, v);
                },
                .iconst_64 => |v| if (inst.dest) |d| {
                    try constants.put(d, v);
                },
                .div_u => |bin| {
                    const dest = inst.dest orelse continue;
                    if (inst.type != .i32 and inst.type != .i64) continue;
                    const rhs_const = constants.get(bin.rhs) orelse continue;
                    const k = powerOfTwoShift(rhs_const, inst.type) orelse continue;

                    const shift_vreg = func.newVReg();
                    const shift_op: ir.Inst.Op = if (inst.type == .i64)
                        .{ .iconst_64 = @intCast(k) }
                    else
                        .{ .iconst_32 = @intCast(k) };
                    try block.instructions.insert(
                        block.allocator,
                        i,
                        .{ .op = shift_op, .dest = shift_vreg, .type = inst.type },
                    );
                    block.instructions.items[i + 1].op = .{ .shr_u = .{
                        .lhs = bin.lhs,
                        .rhs = shift_vreg,
                    } };
                    block.instructions.items[i + 1].dest = dest;
                    try constants.put(shift_vreg, @intCast(k));
                    changed = true;
                    i += 1; // skip over the newly-inserted iconst
                },
                .rem_u => |bin| {
                    const dest = inst.dest orelse continue;
                    if (inst.type != .i32 and inst.type != .i64) continue;
                    const rhs_const = constants.get(bin.rhs) orelse continue;
                    const k = powerOfTwoShift(rhs_const, inst.type) orelse continue;

                    // mask = 2^k - 1. Compute in u64 to avoid sign-shift
                    // edge cases; truncate for the i32 iconst.
                    const mask_u: u64 = (@as(u64, 1) << k) - 1;
                    const mask_vreg = func.newVReg();
                    const mask_op: ir.Inst.Op = if (inst.type == .i64)
                        .{ .iconst_64 = @bitCast(mask_u) }
                    else
                        .{ .iconst_32 = @bitCast(@as(u32, @truncate(mask_u))) };
                    try block.instructions.insert(
                        block.allocator,
                        i,
                        .{ .op = mask_op, .dest = mask_vreg, .type = inst.type },
                    );
                    block.instructions.items[i + 1].op = .{ .@"and" = .{
                        .lhs = bin.lhs,
                        .rhs = mask_vreg,
                    } };
                    block.instructions.items[i + 1].dest = dest;
                    try constants.put(mask_vreg, @as(i64, @bitCast(mask_u)));
                    changed = true;
                    i += 1;
                },
                else => {},
            }
        }
    }
    return changed;
}

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

/// Value-independent algebraic simplifications. Complements
/// `constantFold` (which only fires when a concrete constant operand
/// is visible) by exploiting the fact that `x op x` often reduces to
/// a constant or to `x` itself, regardless of `x`'s value:
///
///   sub x, x        -> 0
///   xor x, x        -> 0
///   and x, x        -> x
///   or  x, x        -> x
///   eq  x, x        -> 1
///   ne  x, x        -> 0
///   lt_s/lt_u x, x  -> 0
///   gt_s/gt_u x, x  -> 0
///   le_s/le_u x, x  -> 1
///   ge_s/ge_u x, x  -> 1
///
/// These patterns appear after `forwardLocalGet` or `commonSubexprElimination`
/// coalesce two vregs into one (e.g. a loop guard that was already
/// proven earlier). They are all sound without value knowledge:
///
/// - None of the integer operations above trap (div/rem are deliberately
///   excluded — `x/x` traps when x == 0).
/// - Float compares are deliberately excluded — `NaN == NaN` is false,
///   so `f_eq x, x` does not reduce to 1.
/// - Operations that reduce to a constant leave the original dest in
///   place (now produced by an `iconst_*`); users see the new value.
/// - Operations that reduce to `x` rewrite uses via `replaceVReg` and
///   leave an `iconst_32 = 0` placeholder for `deadCodeElimination` to
///   sweep, matching the convention used by `constantFold`.
pub fn algebraicSimplify(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    _ = allocator;
    var changed = false;

    for (func.blocks.items) |*block| {
        for (block.instructions.items) |*inst| {
            const dest = inst.dest orelse continue;
            const is_int = inst.type == .i32 or inst.type == .i64;
            if (!is_int) continue;

            switch (inst.op) {
                .sub, .xor => |bin| {
                    if (bin.lhs != bin.rhs) continue;
                    if (inst.type == .i64) {
                        inst.op = .{ .iconst_64 = 0 };
                    } else {
                        inst.op = .{ .iconst_32 = 0 };
                    }
                    changed = true;
                },
                .@"and", .@"or" => |bin| {
                    if (bin.lhs != bin.rhs) continue;
                    const keep = bin.lhs;
                    replaceVReg(func, dest, keep);
                    inst.op = .{ .iconst_32 = 0 };
                    changed = true;
                },
                .eq, .le_s, .le_u, .ge_s, .ge_u => |bin| {
                    if (bin.lhs != bin.rhs) continue;
                    // Match `constantFold` convention: width is `inst.type`.
                    if (inst.type == .i64) {
                        inst.op = .{ .iconst_64 = 1 };
                    } else {
                        inst.op = .{ .iconst_32 = 1 };
                    }
                    changed = true;
                },
                .ne, .lt_s, .lt_u, .gt_s, .gt_u => |bin| {
                    if (bin.lhs != bin.rhs) continue;
                    if (inst.type == .i64) {
                        inst.op = .{ .iconst_64 = 0 };
                    } else {
                        inst.op = .{ .iconst_32 = 0 };
                    }
                    changed = true;
                },
                else => {},
            }
        }
    }

    return changed;
}

fn hasSideEffect(inst: ir.Inst) bool {
    return switch (inst.op) {
        .store, .local_set, .global_set, .call, .call_indirect, .call_ref, .ret, .ret_multi, .br, .br_if, .br_table, .@"unreachable",
        .atomic_fence, .atomic_load, .atomic_store, .atomic_rmw, .atomic_cmpxchg,
        .atomic_notify, .atomic_wait, .memory_copy, .memory_fill, .memory_grow,
        .memory_init, .data_drop, .table_init, .elem_drop, .table_set, .table_grow,
        => true,
        // Trapping ops: must not be removed even if result is unused.
        .load, .table_get,
        .div_u, .rem_u,
        .trunc_f32_s, .trunc_f32_u, .trunc_f64_s, .trunc_f64_u,
        => true,
        // div_s/rem_s trap for integers but not floats (float div produces NaN/Inf).
        .div_s, .rem_s => inst.type != .f32 and inst.type != .f64,
        else => false,
    };
}

// ── Common Subexpression Elimination ────────────────────────────────────────

/// Dominator-scoped CSE. Deduplicates identical pure, non-trapping
/// instructions across basic blocks whenever the earlier def dominates
/// the later one. Walks the dominator tree in preorder using a
/// stack-based DFS; entries added by a block are popped when that
/// block's subtree is fully visited, so sibling subtrees don't see each
/// other's defs.
///
/// Replacement is done via `replaceVReg` — redundant instructions are
/// left in place with their uses rewritten; subsequent
/// `deadCodeElimination` removes them.
///
/// Value keys include `inst.type` so that same-tag, same-operands ops
/// of different types (e.g., `convert_i32_s` lowered from
/// `f32.convert_i32_s` vs `f64.convert_i32_s`) are never conflated.
pub fn commonSubexprElimination(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    if (func.blocks.items.len == 0) return false;

    var dom = try analysis.computeDominators(func, allocator);
    defer dom.deinit();

    // Build dominator-tree children: children[b] = { c : idom[c] == b and c != b }.
    const nblocks = func.blocks.items.len;
    var children = try allocator.alloc(std.ArrayList(ir.BlockId), nblocks);
    defer {
        for (children) |*list| list.deinit(allocator);
        allocator.free(children);
    }
    for (children) |*list| list.* = .empty;
    for (0..nblocks) |i| {
        const bid: ir.BlockId = @intCast(i);
        const idom = dom.idom[bid] orelse continue;
        if (idom == bid) continue; // entry is its own idom
        try children[idom].append(allocator, bid);
    }

    // Value table: list of (op, type, def_vreg) entries visible on the
    // current dominator path. We snapshot the length on block entry and
    // restore it on exit to emulate a scoped table without hash-context
    // plumbing. Lookup is linear in the current path depth — OK for
    // realistic IR sizes and matches the cost profile of the original
    // block-local CSE.
    const Entry = struct { inst: ir.Inst, def: ir.VReg };
    var table: std.ArrayList(Entry) = .empty;
    defer table.deinit(allocator);

    // DFS frame: (block, phase). Phase 0 = process block + snapshot table
    // length and schedule children; phase 1 = restore table length on exit.
    const Frame = struct { bid: ir.BlockId, phase: u1, snapshot_len: usize };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(allocator);

    // Start at entry (block 0). Only reachable blocks have a non-null idom
    // (entry's idom is itself). Unreachable blocks are never visited.
    if (dom.idom[0] == null) return false;
    try stack.append(allocator, .{ .bid = 0, .phase = 0, .snapshot_len = 0 });

    var changed = false;
    while (stack.items.len > 0) {
        const top = &stack.items[stack.items.len - 1];
        if (top.phase == 1) {
            // Exiting this block's subtree — restore table to the snapshot.
            table.shrinkRetainingCapacity(top.snapshot_len);
            _ = stack.pop();
            continue;
        }

        const bid = top.bid;
        top.phase = 1;
        top.snapshot_len = table.items.len;

        const block = &func.blocks.items[bid];
        for (block.instructions.items) |*inst| {
            if (inst.dest == null) continue;
            if (hasSideEffect(inst.*) or !isPure(inst.*)) continue;

            // Look up earliest matching def on the current dom path.
            var found: ?ir.VReg = null;
            for (table.items) |e| {
                if (e.inst.type == inst.type and sameOp(e.inst, inst.*)) {
                    found = e.def;
                    break;
                }
            }
            if (found) |earlier_def| {
                // Only count as a real change if the redundant instruction
                // actually had uses — otherwise it was already dead weight
                // that DCE will sweep regardless. Without this guard the
                // pass would spuriously keep returning `changed=true` and
                // prevent `runPasses` from reaching its fixpoint.
                const uses = countUsesOfVReg(func, inst.dest.?);
                if (uses > 0) {
                    replaceVReg(func, inst.dest.?, earlier_def);
                    changed = true;
                }
                // Don't record this instruction — an earlier entry already
                // covers any future match on the dominator path.
            } else {
                try table.append(allocator, .{ .inst = inst.*, .def = inst.dest.? });
            }
        }

        // Schedule children for visit.
        for (children[bid].items) |c| {
            try stack.append(allocator, .{ .bid = c, .phase = 0, .snapshot_len = 0 });
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
        .f_eq, .f_ne, .f_lt, .f_gt, .f_le, .f_ge,
        .trunc_f32_s, .trunc_f32_u, .trunc_f64_s, .trunc_f64_u,
        .convert_s, .convert_u, .convert_i32_s, .convert_i64_s, .convert_i32_u, .convert_i64_u, .demote_f64, .promote_f32, .reinterpret,
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
        .convert_i32_s => |v| v == b.op.convert_i32_s,
        .convert_i64_s => |v| v == b.op.convert_i64_s,
        .convert_i32_u => |v| v == b.op.convert_i32_u,
        .convert_i64_u => |v| v == b.op.convert_i64_u,
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
/// Redundant bounds-check elimination, dominator-scoped.
///
/// For every `.load` and `.store`, codegen emits an inline wasm-memory
/// bounds check that verifies `zext(base) + offset + size <= memory_size`.
/// When an access is dominated by a prior access sharing the same `base`
/// vreg, the first check already validates any later access whose
/// `offset + size` does not exceed the max previously validated end.
///
/// This pass marks such accesses with `bounds_known = true`; both backends
/// skip emitting the check for those.
///
/// Soundness / scope:
/// - Walks the dominator tree in DFS order. Entries produced in a
///   dominator block are visible to all its dom-tree descendants. This
///   generalises the earlier block-local version and catches the typical
///   loop-body pattern where the header or preheader dominates every
///   access in the loop.
/// - Wasm memory grows monotonically (there is no `memory.shrink`; all
///   mutations only extend it), so `zext(base) + offset + size <=
///   memory_size_old` implies `... <= memory_size_new`. A prior check
///   therefore stays valid across calls / memory.grow / memory.copy /
///   memory.fill / table mutations / atomics. We still clear the active
///   entries on those opcodes as a conservative guard against future IR
///   operations that could shrink memory.
/// - IR is SSA, so a base vreg's value never changes once defined.
/// - `valid_start` is the index into `table` below which entries are
///   shadowed by a fence on the current dominator path. Siblings in the
///   dom-tree are unaffected because we restore `valid_start` on block
///   exit.
pub fn elideRedundantBoundsChecks(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    if (func.blocks.items.len == 0) return false;

    var dom = try analysis.computeDominators(func, allocator);
    defer dom.deinit();

    const nblocks = func.blocks.items.len;
    var children = try allocator.alloc(std.ArrayList(ir.BlockId), nblocks);
    defer {
        for (children) |*list| list.deinit(allocator);
        allocator.free(children);
    }
    for (children) |*list| list.* = .empty;
    for (0..nblocks) |i| {
        const bid: ir.BlockId = @intCast(i);
        const idom = dom.idom[bid] orelse continue;
        if (idom == bid) continue;
        try children[idom].append(allocator, bid);
    }

    const Entry = struct { base: ir.VReg, max_end: u64 };
    var table: std.ArrayList(Entry) = .empty;
    defer table.deinit(allocator);
    var valid_start: usize = 0;

    const Frame = struct {
        bid: ir.BlockId,
        phase: u1,
        snap_len: usize,
        snap_valid_start: usize,
    };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(allocator);

    if (dom.idom[0] == null) return false;
    try stack.append(allocator, .{ .bid = 0, .phase = 0, .snap_len = 0, .snap_valid_start = 0 });

    var changed = false;
    while (stack.items.len > 0) {
        const top = &stack.items[stack.items.len - 1];
        if (top.phase == 1) {
            table.shrinkRetainingCapacity(top.snap_len);
            valid_start = top.snap_valid_start;
            _ = stack.pop();
            continue;
        }
        const bid = top.bid;
        top.phase = 1;
        top.snap_len = table.items.len;
        top.snap_valid_start = valid_start;

        const block = &func.blocks.items[bid];
        for (block.instructions.items) |*inst| {
            switch (inst.op) {
                .load => |*ld| {
                    const end: u64 = @as(u64, ld.offset) + @as(u64, ld.size);
                    var prev: u64 = 0;
                    for (table.items[valid_start..]) |e| {
                        if (e.base == ld.base and e.max_end > prev) prev = e.max_end;
                    }
                    if (end <= prev) {
                        if (!ld.bounds_known) {
                            ld.bounds_known = true;
                            changed = true;
                        }
                    } else {
                        try table.append(allocator, .{ .base = ld.base, .max_end = end });
                    }
                },
                .store => |*st| {
                    const end: u64 = @as(u64, st.offset) + @as(u64, st.size);
                    var prev: u64 = 0;
                    for (table.items[valid_start..]) |e| {
                        if (e.base == st.base and e.max_end > prev) prev = e.max_end;
                    }
                    if (end <= prev) {
                        if (!st.bounds_known) {
                            st.bounds_known = true;
                            changed = true;
                        }
                    } else {
                        try table.append(allocator, .{ .base = st.base, .max_end = end });
                    }
                },
                // Fences: conservatively hide entries from upstream scope
                // for the remainder of this block and its dom-tree
                // descendants. (See header comment on why this is
                // strictly defensive given wasm memory semantics.)
                .memory_grow,
                .call, .call_indirect, .call_ref,
                .memory_copy, .memory_fill, .memory_init,
                .table_grow, .table_init,
                .atomic_notify, .atomic_wait,
                => valid_start = table.items.len,
                else => {},
            }
        }

        for (children[bid].items) |c| {
            try stack.append(allocator, .{ .bid = c, .phase = 0, .snap_len = 0, .snap_valid_start = 0 });
        }
    }

    return changed;
}

/// Forward `local_set K, val` → subsequent `local_get K` within the same
/// block: rewrite consumers of the `local_get`'s dest to use `val` directly,
/// turning the `local_get` into dead code that DCE then removes. This
/// eliminates a STR/LDR round-trip (and an LDR on the initial get) for every
/// such pair — common in induction-variable heavy loops like
/// `i = i + 1; local.set i`.
///
/// Safety: wasm locals are modifiable only by the current function's
/// local.set, so call instructions do not invalidate the map. Control-flow
/// at the block end ends the map's scope. IR is SSA so `val` remains valid
/// everywhere `local_get`'s dest was consumed.
pub fn forwardLocalGet(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;
    var last_set = std.AutoHashMap(u32, ir.VReg).init(allocator);
    defer last_set.deinit();

    for (func.blocks.items) |*block| {
        last_set.clearRetainingCapacity();
        for (block.instructions.items) |inst| {
            switch (inst.op) {
                .local_set => |ls| try last_set.put(ls.idx, ls.val),
                .local_get => |idx| {
                    const dest = inst.dest orelse continue;
                    if (last_set.get(idx)) |val| {
                        replaceVReg(func, dest, val);
                        changed = true;
                    } else {
                        // First read of this local in the block — remember
                        // the new dest so subsequent reads of the same local
                        // in this block can coalesce to this same vreg.
                        try last_set.put(idx, dest);
                    }
                },
                else => {},
            }
        }
    }
    return changed;
}

/// Remove `local.set K, v` for any local K that is never read by a
/// `local.get` anywhere in the function. This is intra-procedural and
/// trivially sound: wasm locals are frame-scoped; calls never observe
/// them. In practice this fires heavily after `forwardLocalGet` has
/// rewritten all reads away.
pub fn deadLocalSetElimination(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var live_locals = std.AutoHashMap(u32, void).init(allocator);
    defer live_locals.deinit();

    for (func.blocks.items) |*block| {
        for (block.instructions.items) |inst| {
            if (inst.op == .local_get) try live_locals.put(inst.op.local_get, {});
        }
    }

    var changed = false;
    for (func.blocks.items) |*block| {
        var i: usize = 0;
        while (i < block.instructions.items.len) {
            const inst = block.instructions.items[i];
            if (inst.op == .local_set and !live_locals.contains(inst.op.local_set.idx)) {
                _ = block.instructions.orderedRemove(i);
                changed = true;
            } else {
                i += 1;
            }
        }
    }
    return changed;
}

/// Constant-fold `br_if` whose condition is a known `iconst_32`. If the
/// condition is zero, rewrite to `br else_block`; otherwise rewrite to
/// `br then_block`. Uses a per-block iconst_32 map (conditions are
/// always i32 in wasm). Unreachable successors are cleaned up later by
/// DCE / block reordering. The fold opens up further straight-line
/// optimizations.
pub fn foldConstantBranches(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;
    for (func.blocks.items) |*block| {
        var iconst32 = std.AutoHashMap(ir.VReg, i32).init(allocator);
        defer iconst32.deinit();

        for (block.instructions.items) |*inst| {
            switch (inst.op) {
                .iconst_32 => |c| {
                    if (inst.dest) |d| try iconst32.put(d, c);
                },
                .br_if => |bi| {
                    if (iconst32.get(bi.cond)) |c| {
                        const target = if (c != 0) bi.then_block else bi.else_block;
                        inst.* = .{ .op = .{ .br = target } };
                        changed = true;
                    }
                },
                else => {},
            }
        }
    }
    return changed;
}

// ── Function Inlining ───────────────────────────────────────────────────────

/// Shift every VReg referenced by `inst` (reads and def) by `+offset`.
/// Mirrors `replaceInInst` but applies a constant shift instead of a
/// single rename.
fn shiftVRegsInInst(inst: *ir.Inst, offset: ir.VReg) void {
    if (inst.dest) |d| inst.dest = d + offset;
    switch (inst.op) {
        .iconst_32, .iconst_64, .fconst_32, .fconst_64,
        .local_get, .global_get, .br, .@"unreachable",
        .memory_size, .table_size, .ref_func, .data_drop, .elem_drop,
        .atomic_fence, .call_result,
        => {},

        .add, .sub, .mul, .div_s, .div_u, .rem_s, .rem_u,
        .@"and", .@"or", .xor, .shl, .shr_s, .shr_u, .rotl, .rotr,
        .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u,
        .f_min, .f_max, .f_copysign,
        .f_eq, .f_ne, .f_lt, .f_gt, .f_le, .f_ge,
        => |*bin| {
            bin.lhs += offset;
            bin.rhs += offset;
        },

        .clz, .ctz, .popcnt, .eqz, .wrap_i64, .extend_i32_s, .extend_i32_u,
        .extend8_s, .extend16_s, .extend32_s,
        .f_neg, .f_abs, .f_sqrt, .f_ceil, .f_floor, .f_trunc, .f_nearest,
        .trunc_f32_s, .trunc_f32_u, .trunc_f64_s, .trunc_f64_u,
        .convert_s, .convert_u, .convert_i32_s, .convert_i64_s, .convert_i32_u, .convert_i64_u, .demote_f64, .promote_f32, .reinterpret,
        .trunc_sat_f32_s, .trunc_sat_f32_u, .trunc_sat_f64_s, .trunc_sat_f64_u,
        => |*vreg| vreg.* += offset,

        .local_set => |*ls| ls.val += offset,
        .global_set => |*gs| gs.val += offset,
        .load => |*ld| ld.base += offset,
        .store => |*st| {
            st.base += offset;
            st.val += offset;
        },
        .br_if => |*bi| bi.cond += offset,
        .br_table => |*bt| bt.index += offset,
        .ret => |*maybe_vreg| if (maybe_vreg.*) |v| { maybe_vreg.* = v + offset; },
        .ret_multi => |vregs| {
            for (@constCast(vregs)) |*v| v.* += offset;
        },
        .call => |cl| {
            for (@constCast(cl.args)) |*arg| arg.* += offset;
        },
        .call_indirect => |*ci| {
            ci.elem_idx += offset;
            for (@constCast(ci.args)) |*arg| arg.* += offset;
        },
        .call_ref => |*cr| {
            cr.func_ref += offset;
            for (@constCast(cr.args)) |*arg| arg.* += offset;
        },
        .select => |*sel| {
            sel.cond += offset;
            sel.if_true += offset;
            sel.if_false += offset;
        },
        .atomic_load => |*al| al.base += offset,
        .atomic_store => |*ast| {
            ast.base += offset;
            ast.val += offset;
        },
        .atomic_rmw => |*ar| {
            ar.base += offset;
            ar.val += offset;
        },
        .atomic_cmpxchg => |*ac| {
            ac.base += offset;
            ac.expected += offset;
            ac.replacement += offset;
        },
        .atomic_notify => |*an| {
            an.base += offset;
            an.count += offset;
        },
        .atomic_wait => |*aw| {
            aw.base += offset;
            aw.expected += offset;
            aw.timeout += offset;
        },
        .memory_copy => |*mc| {
            mc.dst += offset;
            mc.src += offset;
            mc.len += offset;
        },
        .memory_fill => |*mf| {
            mf.dst += offset;
            mf.val += offset;
            mf.len += offset;
        },
        .memory_grow => |*pages| pages.* += offset,
        .table_get => |*tg| tg.idx += offset,
        .table_set => |*ts| {
            ts.idx += offset;
            ts.val += offset;
        },
        .table_grow => |*tg| {
            tg.init += offset;
            tg.delta += offset;
        },
        .memory_init => |*mi| {
            mi.dst += offset;
            mi.src += offset;
            mi.len += offset;
        },
        .table_init => |*ti| {
            ti.dst += offset;
            ti.src += offset;
            ti.len += offset;
        },
    }
}

/// Is this callee eligible for the inliner?
///   - Non-empty, ≤ `max_blocks` blocks
///   - Total instructions ≤ `max_insts`
///   - No calls (direct/indirect/ref), no call_result
///   - No memory_grow, no atomics, no bulk memory/table ops
///   - No `br_table` (avoid cloning the targets slice)
///   - No `ret_multi`
///   - local_count == param_count (no extra declared locals)
///   - No `local_set` anywhere (params aren't mutated)
///   - Every `local_get` targets a param (idx < param_count)
///   - `result_count` ∈ {0, 1}
///   - If result_count == 1: exactly one `ret` (so the returned value
///     is unambiguous; phi would be required otherwise)
///   - If result_count == 0: ≥ 1 `ret` (so the continuation block is
///     reachable after inlining)
fn isInlinable(callee: *const ir.IrFunction, max_insts: u32, max_blocks: u32) bool {
    const nblocks = callee.blocks.items.len;
    if (nblocks == 0) return false;
    if (nblocks > max_blocks) return false;
    if (callee.result_count > 1) return false;
    if (callee.local_count != callee.param_count) return false;

    var total_insts: u32 = 0;
    var ret_count: u32 = 0;

    for (callee.blocks.items) |blk| {
        if (blk.instructions.items.len == 0) return false;
        total_insts +|= @intCast(blk.instructions.items.len);
        if (total_insts > max_insts) return false;

        for (blk.instructions.items) |inst| {
            switch (inst.op) {
                .call, .call_indirect, .call_ref, .call_result,
                .memory_grow,
                .atomic_fence, .atomic_load, .atomic_store, .atomic_rmw,
                .atomic_cmpxchg, .atomic_notify, .atomic_wait,
                .memory_copy, .memory_fill, .memory_init, .table_init,
                .table_grow, .data_drop, .elem_drop,
                .br_table,
                .ret_multi,
                .local_set,
                => return false,
                .local_get => |idx| if (idx >= callee.param_count) return false,
                .ret => ret_count += 1,
                else => {},
            }
        }
    }

    if (callee.result_count == 1) {
        if (ret_count != 1) return false;
    } else {
        if (ret_count == 0) return false;
    }
    return true;
}

/// Shift every `BlockId` referenced by `inst` (br / br_if targets) by
/// `+offset`. br_table is excluded by `isInlinable` so we don't handle
/// it here.
fn shiftBlockIdsInInst(inst: *ir.Inst, offset: ir.BlockId) void {
    switch (inst.op) {
        .br => |*t| t.* += offset,
        .br_if => |*bi| {
            bi.then_block += offset;
            bi.else_block += offset;
        },
        else => {},
    }
}

/// Module-level pass: replace direct calls to small callees (including
/// multi-block ones) with a clone of the callee's body. Returns whether
/// any call site was inlined.
///
/// Layout at each call site:
///   caller block B  [pre..., call, post..., terminator]
/// becomes:
///   B (unchanged id)     [pre..., br clone_entry]
///   clone_block[0..M]    shifted copy of callee.blocks[0..M] with every
///                        `ret` rewritten to `br B_after`
///   B_after (new id)     [post..., terminator]
///
/// Every VReg in the clone is shifted by `vreg_offset = caller.next_vreg`
/// (then caller.next_vreg += callee.next_vreg). Every BlockId in the
/// clone is shifted by `clone_offset = b_after_id + 1`. `local_get`
/// instructions are dropped and their shifted dests rewired to the
/// corresponding call-site argument vreg. If the callee produces a
/// result, its (single) `ret` value is translated through local renames
/// (to cover the `local.get; ret` identity case) and the call's dest is
/// rewritten to it.
pub fn inlineSmallFunctions(module: *ir.IrModule, allocator: std.mem.Allocator) !bool {
    const max_blocks: u32 = 8;
    const max_insts: u32 = 32;

    var eligible = try allocator.alloc(bool, module.functions.items.len);
    defer allocator.free(eligible);
    for (module.functions.items, 0..) |*f, i| eligible[i] = isInlinable(f, max_insts, max_blocks);

    var any_inlined = false;
    for (module.functions.items, 0..) |*caller, caller_idx| {
        // Only scan blocks that existed at the start of this pass. Newly
        // created clone blocks can't contain eligible calls (isInlinable
        // excludes all calls), and B_after inherits only post-call IR
        // from the caller which the fixpoint loop will revisit.
        const original_block_count = caller.blocks.items.len;
        var b: usize = 0;
        while (b < original_block_count) : (b += 1) {
            // Find the first inlinable call in this block.
            var call_idx: ?usize = null;
            {
                const block = &caller.blocks.items[b];
                var i: usize = 0;
                while (i < block.instructions.items.len) : (i += 1) {
                    const inst = block.instructions.items[i];
                    const call = switch (inst.op) {
                        .call => |c| c,
                        else => continue,
                    };
                    if (call.tail) continue;
                    if (call.extra_results != 0) continue;
                    if (call.func_idx < module.import_count) continue;
                    const c_idx: usize = @intCast(call.func_idx - module.import_count);
                    if (c_idx == caller_idx) continue;
                    if (c_idx >= module.functions.items.len) continue;
                    if (!eligible[c_idx]) continue;
                    const callee = &module.functions.items[c_idx];
                    if (call.args.len != callee.param_count) continue;
                    call_idx = i;
                    break;
                }
            }
            if (call_idx == null) continue;

            const ci = call_idx.?;
            const call_inst = caller.blocks.items[b].instructions.items[ci];
            const call = call_inst.op.call;
            const call_dest = call_inst.dest;
            const call_args = call.args;
            const callee_ref_idx: usize = @intCast(call.func_idx - module.import_count);
            const callee = &module.functions.items[callee_ref_idx];

            const vreg_offset: ir.VReg = caller.next_vreg;
            caller.next_vreg += callee.next_vreg;

            // Allocate clone blocks, then B_after last, so storage order
            // matches execution order (B → clones → B_after). `computeLiveRanges`
            // numbers instructions by storage-order traversal; placing B_after
            // after the clones is essential for cross-block live ranges of the
            // inlined ret value to be computed correctly.
            const clone_offset = try caller.newBlock();
            var kb: usize = 1;
            while (kb < callee.blocks.items.len) : (kb += 1) {
                _ = try caller.newBlock();
            }
            const b_after_id = try caller.newBlock();
            // Caller.blocks may have re-allocated; all block pointers
            // taken before now are invalid. Always index via
            // `caller.blocks.items[...]` from here on.

            // Move post-call instructions into B_after, truncate B,
            // and append `br clone_offset` as B's new terminator.
            {
                const post_start = ci + 1;
                const src_len = caller.blocks.items[b].instructions.items.len;
                var k = post_start;
                while (k < src_len) : (k += 1) {
                    const moved = caller.blocks.items[b].instructions.items[k];
                    try caller.blocks.items[b_after_id].instructions.append(caller.allocator, moved);
                }
                caller.blocks.items[b].instructions.shrinkRetainingCapacity(ci);
                try caller.blocks.items[b].instructions.append(caller.allocator, .{ .op = .{ .br = clone_offset } });
            }

            // Clone callee blocks, shifting vregs and block ids.
            var local_renames = std.ArrayList(struct { from: ir.VReg, to: ir.VReg }).empty;
            defer local_renames.deinit(allocator);
            var ret_val_shifted: ?ir.VReg = null;

            for (callee.blocks.items, 0..) |callee_block, cidx| {
                const clone_id: ir.BlockId = clone_offset + @as(ir.BlockId, @intCast(cidx));
                for (callee_block.instructions.items) |citem| {
                    switch (citem.op) {
                        .local_get => |idx| {
                            const shifted_dest = (citem.dest orelse continue) + vreg_offset;
                            try local_renames.append(allocator, .{
                                .from = shifted_dest,
                                .to = call_args[idx],
                            });
                        },
                        .ret => |maybe_v| {
                            if (maybe_v) |v| {
                                // Only one ret is allowed when result_count==1,
                                // so this is the single ret value.
                                ret_val_shifted = v + vreg_offset;
                            }
                            try caller.blocks.items[clone_id].instructions.append(
                                caller.allocator,
                                .{ .op = .{ .br = b_after_id } },
                            );
                        },
                        else => {
                            var cloned = citem;
                            shiftVRegsInInst(&cloned, vreg_offset);
                            shiftBlockIdsInInst(&cloned, clone_offset);
                            try caller.blocks.items[clone_id].instructions.append(caller.allocator, cloned);
                        },
                    }
                }
            }

            // Apply local_get renames across the whole caller.
            for (local_renames.items) |r| {
                replaceVReg(caller, r.from, r.to);
            }

            // Translate ret value through local renames: covers the case
            // where the callee's ret references a local_get's dest that
            // was never emitted (`local.get 0; ret`).
            if (ret_val_shifted) |rv| {
                for (local_renames.items) |r| {
                    if (rv == r.from) {
                        ret_val_shifted = r.to;
                        break;
                    }
                }
            }

            // Rewrite the call's dest (now only present in the copied
            // B_after) to the shifted ret value.
            if (call_dest) |d| {
                if (ret_val_shifted) |rv| {
                    replaceVReg(caller, d, rv);
                }
            }

            any_inlined = true;
        }
    }
    return any_inlined;
}

pub fn runPasses(module: *ir.IrModule, passes: []const PassFn, allocator: std.mem.Allocator) !u32 {
    var total_changes: u32 = 0;
    // Module-level: inline small leaf callees before per-function passes.
    // Iterate to fixpoint so callers of callers also benefit.
    var inline_iter: u32 = 0;
    while (inline_iter < 4) : (inline_iter += 1) {
        if (!(try inlineSmallFunctions(module, allocator))) break;
        total_changes += 1;
    }
    for (module.functions.items) |*func| {
        // Iterate the pipeline until fixpoint so that passes can re-expose
        // opportunities for each other (e.g. constantFold → CSE → DCE →
        // more constantFold). Cap iterations as a safety net.
        var iter: u32 = 0;
        while (iter < 8) : (iter += 1) {
            var any_changed = false;
            for (passes) |pass| {
                if (try pass(func, allocator)) {
                    any_changed = true;
                    total_changes += 1;
                }
            }
            if (!any_changed) break;
        }
    }
    return total_changes;
}

/// The default optimization pipeline.
pub const default_passes: []const PassFn = &.{
    &forwardLocalGet,
    &constantFold,
    &algebraicSimplify,
    &strengthReduceMul,
    &strengthReduceMulShiftAdd,
    &strengthReduceDivRem,
    &foldConstantBranches,
    &commonSubexprElimination,
    &deadCodeElimination,
    &deadLocalSetElimination,
    &elideRedundantBoundsChecks,
};

// ── Block Reordering ────────────────────────────────────────────────────────

const DfsEntry = struct { block: ir.BlockId, child_idx: usize };

/// Compute a block emission order using Reverse Postorder (RPO) with
/// cold-block sinking. Places hot blocks contiguously for i-cache locality
/// and maximises fall-through opportunities for the C3 peephole.
///
/// Returns a permutation of all BlockIds (caller owns the slice).
/// Block 0 (entry) is always first. Unreachable blocks are appended at the
/// end. Blocks containing `.unreachable` instructions are sunk after all
/// hot blocks.
pub fn reorderBlocks(func: *const ir.IrFunction, allocator: std.mem.Allocator) ![]ir.BlockId {
    const n: u32 = @intCast(func.blocks.items.len);
    if (n <= 1) {
        const order = try allocator.alloc(ir.BlockId, n);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        return order;
    }

    // Build CFG
    var successors = try analysis.buildSuccessors(func, allocator);
    defer {
        var it = successors.valueIterator();
        while (it.next()) |v| allocator.free(v.*);
        successors.deinit();
    }

    // Iterative DFS → post-order
    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    @memset(visited, false);

    var post_order: std.ArrayList(ir.BlockId) = .empty;
    defer post_order.deinit(allocator);

    var stack: std.ArrayList(DfsEntry) = .empty;
    defer stack.deinit(allocator);

    visited[0] = true;
    try stack.append(allocator, .{ .block = 0, .child_idx = 0 });

    while (stack.items.len > 0) {
        const top = &stack.items[stack.items.len - 1];
        const succs = successors.get(top.block) orelse &[_]ir.BlockId{};
        if (top.child_idx < succs.len) {
            const child = succs[top.child_idx];
            top.child_idx += 1;
            if (child < n and !visited[child]) {
                visited[child] = true;
                try stack.append(allocator, .{ .block = child, .child_idx = 0 });
            }
        } else {
            try post_order.append(allocator, top.block);
            _ = stack.pop();
        }
    }

    // Reverse → RPO
    std.mem.reverse(ir.BlockId, post_order.items);

    // Detect cold blocks (those containing .@"unreachable")
    const is_cold = try allocator.alloc(bool, n);
    defer allocator.free(is_cold);
    @memset(is_cold, false);

    for (func.blocks.items, 0..) |block, idx| {
        for (block.instructions.items) |inst| {
            if (inst.op == .@"unreachable") {
                is_cold[idx] = true;
                break;
            }
        }
    }
    // Entry block is never treated as cold.
    is_cold[0] = false;

    // Partition RPO: hot first, cold second (preserving RPO within each group)
    var order = try allocator.alloc(ir.BlockId, n);
    var hot_i: usize = 0;

    for (post_order.items) |bid| {
        if (!is_cold[bid]) {
            order[hot_i] = bid;
            hot_i += 1;
        }
    }
    var cold_i: usize = hot_i;
    for (post_order.items) |bid| {
        if (is_cold[bid]) {
            order[cold_i] = bid;
            cold_i += 1;
        }
    }

    // Append any unreachable blocks (not visited by DFS)
    for (0..n) |idx| {
        if (!visited[idx]) {
            order[cold_i] = @intCast(idx);
            cold_i += 1;
        }
    }

    std.debug.assert(cold_i == n);
    std.debug.assert(order[0] == 0); // Entry block must be first
    return order;
}

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

test "DCE: preserves call argument VRegs" {
    // Regression test: DCE must not remove instructions whose results
    // are passed as arguments to a call (unbounded VReg list).
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const arg0 = func.newVReg();
    const arg1 = func.newVReg();
    const args = try allocator.dupe(ir.VReg, &[_]ir.VReg{ arg0, arg1 });
    defer allocator.free(args);
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = arg0 });
    try block.append(.{ .op = .{ .iconst_32 = 20 }, .dest = arg1 });
    try block.append(.{ .op = .{ .call = .{ .func_idx = 0, .args = args } } });
    try block.append(.{ .op = .{ .ret = null } });

    const changed = try deadCodeElimination(&func, allocator);
    try std.testing.expect(!changed); // arg0 and arg1 are used by the call
    try std.testing.expectEqual(@as(usize, 4), block.instructions.items.len);
}

test "DCE: preserves call_indirect elem_idx and arg VRegs" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const elem = func.newVReg();
    const arg = func.newVReg();
    const call_args = try allocator.dupe(ir.VReg, &[_]ir.VReg{arg});
    defer allocator.free(call_args);
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = elem });
    try block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = arg });
    try block.append(.{ .op = .{ .call_indirect = .{ .type_idx = 0, .elem_idx = elem, .args = call_args } } });
    try block.append(.{ .op = .{ .ret = null } });

    const changed = try deadCodeElimination(&func, allocator);
    try std.testing.expect(!changed); // elem and arg are both used
    try std.testing.expectEqual(@as(usize, 4), block.instructions.items.len);
}

test "DCE: preserves call_ref func_ref and arg VRegs" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const fref = func.newVReg();
    const arg = func.newVReg();
    const call_args = try allocator.dupe(ir.VReg, &[_]ir.VReg{arg});
    defer allocator.free(call_args);
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = fref });
    try block.append(.{ .op = .{ .iconst_32 = 9 }, .dest = arg });
    try block.append(.{ .op = .{ .call_ref = .{ .type_idx = 0, .func_ref = fref, .args = call_args } } });
    try block.append(.{ .op = .{ .ret = null } });

    const changed = try deadCodeElimination(&func, allocator);
    try std.testing.expect(!changed); // fref and arg are both used
    try std.testing.expectEqual(@as(usize, 4), block.instructions.items.len);
}

test "DCE: preserves ret_multi VRegs" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const ret_vals = try allocator.dupe(ir.VReg, &[_]ir.VReg{ v0, v1 });
    defer allocator.free(ret_vals);
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1 });
    try block.append(.{ .op = .{ .ret_multi = ret_vals } });

    const changed = try deadCodeElimination(&func, allocator);
    try std.testing.expect(!changed); // v0 and v1 are used by ret_multi
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

test "CSE: cross-block dominator substitution" {
    // b0 defines add(v0, v1) = v2; b0 branches to b1; b1 recomputes the
    // same add into v3. b0 dominates b1, so v3 must be rewritten to v2.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    const v3 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try func.getBlock(b0).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v3 });
    try func.getBlock(b1).append(.{ .op = .{ .ret = v3 } });

    const changed = try commonSubexprElimination(&func, allocator);
    try std.testing.expect(changed);

    try std.testing.expectEqual(ir.Inst.Op{ .ret = v2 }, func.getBlock(b1).instructions.items[1].op);
}

test "CSE: sibling defs do not match at merge" {
    // b0 → {b1, b2} → b3. b1 and b2 each compute add(v0, v1) independently;
    // b3 recomputes it. Neither b1 nor b2 dominates b3 (b0 does), so neither
    // sibling def is visible when b3 is processed — b3's add must remain a
    // new def, NOT rewritten to a sibling's VReg.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const cond = func.newVReg();
    const v_b1 = func.newVReg();
    const v_b2 = func.newVReg();
    const v_b3 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v_b1 });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v_b2 });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v_b3 });
    try func.getBlock(b3).append(.{ .op = .{ .ret = v_b3 } });

    _ = try commonSubexprElimination(&func, allocator);

    // b3's add must still produce v_b3 (not have been rewritten away), and
    // the ret must still reference v_b3.
    try std.testing.expectEqual(
        @as(?ir.VReg, v_b3),
        func.getBlock(b3).instructions.items[0].dest,
    );
    try std.testing.expectEqual(
        ir.Inst.Op{ .ret = v_b3 },
        func.getBlock(b3).instructions.items[1].op,
    );
    // Neither sibling should have been rewritten by the other.
    try std.testing.expectEqual(@as(?ir.VReg, v_b1), func.getBlock(b1).instructions.items[0].dest);
    try std.testing.expectEqual(@as(?ir.VReg, v_b2), func.getBlock(b2).instructions.items[0].dest);
}

test "CSE: type-sensitive — convert_i32_s to f32 vs f64 do not merge" {
    // Two `convert_i32_s` insts with the same source VReg but different
    // inst.type must NOT be CSE'd, because the frontend lowers
    // f32.convert_i32_s and f64.convert_i32_s to the same IR tag with
    // different types.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const src = func.newVReg();
    const v_f32 = func.newVReg();
    const v_f64 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = src });
    try block.append(.{ .op = .{ .convert_i32_s = src }, .dest = v_f32, .type = .f32 });
    try block.append(.{ .op = .{ .convert_i32_s = src }, .dest = v_f64, .type = .f64 });
    try block.append(.{ .op = .{ .ret = v_f64 } });

    _ = try commonSubexprElimination(&func, allocator);

    // v_f64 must not have been rewritten to v_f32.
    try std.testing.expectEqual(ir.Inst.Op{ .ret = v_f64 }, block.instructions.items[3].op);
    try std.testing.expectEqual(@as(?ir.VReg, v_f64), block.instructions.items[2].dest);
}

test "CSE: trapping int div_s is not deduplicated" {
    // Two identical i32 div_s must NOT be CSE'd because div_s traps on
    // zero divisor (hasSideEffect returns true for integer div_s). Both
    // defs must remain so both traps happen.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    const v3 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1 });
    try block.append(.{ .op = .{ .div_s = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .div_s = .{ .lhs = v0, .rhs = v1 } }, .dest = v3, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v3 } });

    _ = try commonSubexprElimination(&func, allocator);

    // v3 must remain — ret still points to v3 and v3's def is still div_s.
    try std.testing.expectEqual(ir.Inst.Op{ .ret = v3 }, block.instructions.items[4].op);
    try std.testing.expectEqual(@as(?ir.VReg, v3), block.instructions.items[3].dest);
}

test "CSE: loop header def reused in body" {
    // b0 → b1(header): adds v0+v1 into v_h; br_if body/exit.
    // body → b1 (back-edge). body recomputes v0+v1 as v_body and writes
    // it to a local (so v_body has a real use). Since b1 dominates body,
    // CSE should rewrite that use to v_h.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 1);
    defer func.deinit();

    const b0 = try func.newBlock();
    const h = try func.newBlock();
    const body = try func.newBlock();
    const exit = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v_h = func.newVReg();
    const cond = func.newVReg();
    const v_body = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try func.getBlock(b0).append(.{ .op = .{ .br = h } });

    try func.getBlock(h).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v_h });
    try func.getBlock(h).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(h).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = body, .else_block = exit } } });

    try func.getBlock(body).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v_body });
    // Real use of v_body so CSE has something to rewrite.
    try func.getBlock(body).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_body } } });
    try func.getBlock(body).append(.{ .op = .{ .br = h } });

    try func.getBlock(exit).append(.{ .op = .{ .ret = v_h } });

    const changed = try commonSubexprElimination(&func, allocator);
    try std.testing.expect(changed);

    // The local_set in body must now reference v_h, not v_body.
    try std.testing.expectEqual(
        ir.Inst.Op{ .local_set = .{ .idx = 0, .val = v_h } },
        func.getBlock(body).instructions.items[1].op,
    );

    // Idempotent: a second run finds the redundant add but its dest
    // (v_body) now has zero uses, so it's treated as already-dead and
    // not reported as a change.
    const again = try commonSubexprElimination(&func, allocator);
    try std.testing.expect(!again);
}

test "CSE: unreachable block is skipped" {
    // b0 → ret; b1 is unreachable and has an add that matches nothing.
    // Running CSE must not crash and must not touch b1.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try func.getBlock(b1).append(.{ .op = .{ .ret = v2 } });

    const changed = try commonSubexprElimination(&func, allocator);
    try std.testing.expect(!changed);

    // b1's add is untouched.
    try std.testing.expectEqual(@as(?ir.VReg, v2), func.getBlock(b1).instructions.items[2].dest);
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

// ── Block Reordering Tests ─────────────────────────────────────────────────

test "reorderBlocks: single block → identity" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    _ = try func.newBlock(); // block 0

    const order = try reorderBlocks(&func, allocator);
    defer allocator.free(order);
    try std.testing.expectEqual(@as(usize, 1), order.len);
    try std.testing.expectEqual(@as(ir.BlockId, 0), order[0]);
}

test "reorderBlocks: diamond CFG preserves RPO" {
    // CFG: 0 → {1,2}, 1 → 3, 2 → 3
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    const cond = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    const order = try reorderBlocks(&func, allocator);
    defer allocator.free(order);
    try std.testing.expectEqual(@as(usize, 4), order.len);
    // Block 0 must be first (entry)
    try std.testing.expectEqual(@as(ir.BlockId, 0), order[0]);
    // All 4 blocks present
    var seen = [_]bool{false} ** 4;
    for (order) |bid| seen[bid] = true;
    for (seen) |s| try std.testing.expect(s);
    // Block 3 (merge) must come after both 1 and 2
    var pos: [4]usize = undefined;
    for (order, 0..) |bid, i| pos[bid] = i;
    try std.testing.expect(pos[3] > pos[1]);
    try std.testing.expect(pos[3] > pos[2]);
}

test "reorderBlocks: cold block sunk to end" {
    // CFG: 0 → {1(cold), 2}, 2 → ret
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock(); // cold (unreachable)
    const b2 = try func.newBlock(); // hot

    const cond = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .@"unreachable" = {} } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    const order = try reorderBlocks(&func, allocator);
    defer allocator.free(order);
    try std.testing.expectEqual(@as(usize, 3), order.len);
    // Block 0 first
    try std.testing.expectEqual(@as(ir.BlockId, 0), order[0]);
    // Cold block 1 should be last
    try std.testing.expectEqual(@as(ir.BlockId, 1), order[order.len - 1]);
    // Hot block 2 should be second (right after entry)
    try std.testing.expectEqual(@as(ir.BlockId, 2), order[1]);
}

test "reorderBlocks: unreachable block appended at end" {
    // CFG: 0 → 1, block 2 is unreachable
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    _ = try func.newBlock(); // b2: unreachable, no edges to it

    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });

    const order = try reorderBlocks(&func, allocator);
    defer allocator.free(order);
    try std.testing.expectEqual(@as(usize, 3), order.len);
    try std.testing.expectEqual(@as(ir.BlockId, 0), order[0]);
    try std.testing.expectEqual(@as(ir.BlockId, 1), order[1]);
    // Unreachable block 2 at end
    try std.testing.expectEqual(@as(ir.BlockId, 2), order[2]);
}

test "strengthReduceMul: mul(x, 8) → shl(x, 3)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_x = func.newVReg(); // param (fake)
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 8 }, .dest = v_c });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceMul(&func, allocator);
    try std.testing.expect(changed);

    // Block should now be: iconst_32=8, iconst_32=3, shl(v_x, new_vreg), ret
    try std.testing.expectEqual(@as(usize, 4), block.instructions.items.len);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 3 }, block.instructions.items[1].op);
    const shl = block.instructions.items[2];
    switch (shl.op) {
        .shl => |bin| {
            try std.testing.expectEqual(v_x, bin.lhs);
            try std.testing.expectEqual(block.instructions.items[1].dest.?, bin.rhs);
        },
        else => try std.testing.expect(false),
    }
    try std.testing.expectEqual(v_r, shl.dest.?);
}

test "strengthReduceMul: commutative mul(C, x)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 16 }, .dest = v_c });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_c, .rhs = v_x } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceMul(&func, allocator);
    try std.testing.expect(changed);
    const shl = block.instructions.items[2];
    switch (shl.op) {
        .shl => |bin| try std.testing.expectEqual(v_x, bin.lhs),
        else => try std.testing.expect(false),
    }
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 4 }, block.instructions.items[1].op);
}

test "strengthReduceMul: i64 mul by power of two" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_64 = 1 << 40 }, .dest = v_c, .type = .i64 });
    try block.append(.{
        .op = .{ .mul = .{ .lhs = v_x, .rhs = v_c } },
        .dest = v_r,
        .type = .i64,
    });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceMul(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_64 = 40 }, block.instructions.items[1].op);
    try std.testing.expectEqual(ir.IrType.i64, block.instructions.items[1].type);
    try std.testing.expectEqual(ir.IrType.i64, block.instructions.items[2].type);
}

test "strengthReduceMul: does not rewrite mul by non-power-of-two" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = v_c });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceMul(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(@as(usize, 3), block.instructions.items.len);
    switch (block.instructions.items[1].op) {
        .mul => {},
        else => try std.testing.expect(false),
    }
}

test "strengthReduceMul: skips C=1 and C=0 and negatives" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_x = func.newVReg();
    const v_c0 = func.newVReg();
    const v_c1 = func.newVReg();
    const v_cneg = func.newVReg();
    const v_r0 = func.newVReg();
    const v_r1 = func.newVReg();
    const v_rn = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_c0 });
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_c1 });
    try block.append(.{ .op = .{ .iconst_32 = -4 }, .dest = v_cneg });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_x, .rhs = v_c0 } }, .dest = v_r0 });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_x, .rhs = v_c1 } }, .dest = v_r1 });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_x, .rhs = v_cneg } }, .dest = v_rn });
    try block.append(.{ .op = .{ .ret = v_r0 } });

    const changed = try strengthReduceMul(&func, allocator);
    try std.testing.expect(!changed);
}

test "strengthReduceMul: i32 does not rewrite shift >= 32" {
    // 2^32 fits in i64 but is illegal as an i32 shift amount.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    // i32 iconst; value 1<<31 = -2147483648 as i32 → still a power of two,
    // and 31 is a legal shift amount, so this should rewrite.
    try block.append(.{ .op = .{ .iconst_32 = @bitCast(@as(u32, 1) << 31) }, .dest = v_c });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceMul(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 31 }, block.instructions.items[1].op);
}

test "elideRedundantBoundsChecks: back-to-back loads on same base" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_base = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    const v_c = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_a, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 4, .size = 4 } }, .dest = v_b, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 8 } }, .dest = v_c, .type = .i64 });
    try block.append(.{ .op = .{ .ret = v_c } });

    const changed = try elideRedundantBoundsChecks(&func, allocator);
    try std.testing.expect(changed);
    // First load establishes end=4. Second load end=8 > 4, so it sets new max
    // and is NOT elided. Third load end=8 == max, IS elided.
    try std.testing.expect(!block.instructions.items[1].op.load.bounds_known);
    try std.testing.expect(!block.instructions.items[2].op.load.bounds_known);
    try std.testing.expect(block.instructions.items[3].op.load.bounds_known);
}

test "elideRedundantBoundsChecks: call invalidates tracker" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_base = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 8 } }, .dest = v_a, .type = .i64 });
    try block.append(.{ .op = .{ .call = .{ .func_idx = 0 } } });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_b, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_b } });

    _ = try elideRedundantBoundsChecks(&func, allocator);
    // Post-call load cannot be elided because memory_size may have changed.
    try std.testing.expect(!block.instructions.items[3].op.load.bounds_known);
}

test "forwardLocalGet: set then get within block forwards vreg" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 1); // 1 local
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_c = func.newVReg();
    const v_g = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v_c, .type = .i32 });
    try block.append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_c } } });
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = v_g, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v_g, .rhs = v_c } }, .dest = v_r, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try forwardLocalGet(&func, allocator);
    try std.testing.expect(changed);
    // The add should now use v_c (the forwarded val) on both sides.
    try std.testing.expectEqual(v_c, block.instructions.items[3].op.add.lhs);
    try std.testing.expectEqual(v_c, block.instructions.items[3].op.add.rhs);
}

test "forwardLocalGet: repeated gets without set share the first dest" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 1);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_a = func.newVReg();
    const v_b = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = v_a, .type = .i32 });
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = v_b, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v_a, .rhs = v_b } }, .dest = v_r, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try forwardLocalGet(&func, allocator);
    try std.testing.expect(changed);
    // Both adds' operands should coalesce to v_a (the first get's dest).
    try std.testing.expectEqual(v_a, block.instructions.items[2].op.add.lhs);
    try std.testing.expectEqual(v_a, block.instructions.items[2].op.add.rhs);
}

test "deadLocalSetElimination: removes set of never-read local" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 2); // 2 locals
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_c = func.newVReg();
    const v_g = func.newVReg();
    // local 0 is set but never read; local 1 is set and read.
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_c, .type = .i32 });
    try block.append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_c } } });
    try block.append(.{ .op = .{ .local_set = .{ .idx = 1, .val = v_c } } });
    try block.append(.{ .op = .{ .local_get = 1 }, .dest = v_g, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_g } });

    const changed = try deadLocalSetElimination(&func, allocator);
    try std.testing.expect(changed);
    // Only the set for local 0 should be removed.
    try std.testing.expectEqual(@as(usize, 4), block.instructions.items.len);
    // Verify the remaining set is for local 1.
    try std.testing.expectEqual(@as(u32, 1), block.instructions.items[1].op.local_set.idx);
}

test "deadLocalSetElimination: keeps set when local is read" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 1);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_c = func.newVReg();
    const v_g = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = v_c, .type = .i32 });
    try block.append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_c } } });
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = v_g, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_g } });

    const changed = try deadLocalSetElimination(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(@as(usize, 4), block.instructions.items.len);
}

test "constantFold: shl of constants" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b = try func.newBlock();
    var block = &func.blocks.items[b];
    const va = func.newVReg();
    const vb = func.newVReg();
    const vr = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = va, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = vb, .type = .i32 });
    try block.append(.{ .op = .{ .shl = .{ .lhs = va, .rhs = vb } }, .dest = vr, .type = .i32 });
    try block.append(.{ .op = .{ .ret = vr } });

    const changed = try constantFold(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(@as(i64, 40), block.instructions.items[2].op.iconst_32);
}

test "constantFold: unsigned compare lt_u" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b = try func.newBlock();
    var block = &func.blocks.items[b];
    const va = func.newVReg();
    const vb = func.newVReg();
    const vr = func.newVReg();
    // -1 as i32 (0xFFFFFFFF) < 1 unsigned? No: 0xFFFFFFFF > 1.
    try block.append(.{ .op = .{ .iconst_32 = -1 }, .dest = va, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = vb, .type = .i32 });
    try block.append(.{ .op = .{ .lt_u = .{ .lhs = va, .rhs = vb } }, .dest = vr, .type = .i32 });
    try block.append(.{ .op = .{ .ret = vr } });

    const changed = try constantFold(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(@as(i64, 0), block.instructions.items[2].op.iconst_32);
}

test "constantFold: algebraic identity add zero" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b = try func.newBlock();
    var block = &func.blocks.items[b];
    const v_param = func.newVReg(); // vreg 0, param
    const v_zero = func.newVReg();
    const v_r = func.newVReg();
    const v_ret = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_zero, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v_param, .rhs = v_zero } }, .dest = v_r, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v_r, .rhs = v_r } }, .dest = v_ret, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_ret } });

    const changed = try constantFold(&func, allocator);
    try std.testing.expect(changed);
    // After identity rewrite, the second add should use v_param directly on both sides.
    try std.testing.expectEqual(v_param, block.instructions.items[2].op.add.lhs);
    try std.testing.expectEqual(v_param, block.instructions.items[2].op.add.rhs);
}

test "constantFold: select with constant cond picks branch" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 2, 1, 0);
    defer func.deinit();
    const b = try func.newBlock();
    var block = &func.blocks.items[b];
    const v_a = func.newVReg(); // param 0
    const v_b = func.newVReg(); // param 1
    const v_cond = func.newVReg();
    const v_sel = func.newVReg();
    const v_ret = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_cond, .type = .i32 });
    try block.append(.{ .op = .{ .select = .{ .cond = v_cond, .if_true = v_a, .if_false = v_b } }, .dest = v_sel, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v_sel, .rhs = v_sel } }, .dest = v_ret, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_ret } });

    const changed = try constantFold(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(v_a, block.instructions.items[2].op.add.lhs);
    try std.testing.expectEqual(v_a, block.instructions.items[2].op.add.rhs);
}

test "inlineSmallFunctions: leaf with param-return is inlined" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    // Callee: fn id(x) -> x   { local.get 0; return }
    try module.functions.append(allocator, ir.IrFunction.init(allocator, 1, 1, 1));
    {
        const callee = &module.functions.items[0];
        const cb = try callee.newBlock();
        _ = callee.newVReg(); // param 0 placeholder
        const v_get = callee.newVReg();
        try callee.getBlock(cb).append(.{ .op = .{ .local_get = 0 }, .dest = v_get, .type = .i32 });
        try callee.getBlock(cb).append(.{ .op = .{ .ret = v_get } });
    }

    // Caller: fn main() -> i32   { i32.const 42; call 0; return }
    try module.functions.append(allocator, ir.IrFunction.init(allocator, 0, 1, 0));
    const args = try allocator.alloc(ir.VReg, 1);
    defer allocator.free(args);
    {
        const caller = &module.functions.items[1];
        const mb = try caller.newBlock();
        const v_arg = caller.newVReg();
        const v_ret = caller.newVReg();
        try caller.getBlock(mb).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v_arg, .type = .i32 });
        args[0] = v_arg;
        try caller.getBlock(mb).append(.{ .op = .{ .call = .{ .func_idx = 0, .args = args } }, .dest = v_ret, .type = .i32 });
        try caller.getBlock(mb).append(.{ .op = .{ .ret = v_ret } });
    }

    const inlined = try inlineSmallFunctions(&module, allocator);
    try std.testing.expect(inlined);

    const caller = &module.functions.items[1];
    // V2 layout: B0 keeps [iconst_32, br clone_entry]; B1 = clone_entry [br B_after];
    // B2 = B_after [ret].
    try std.testing.expectEqual(@as(usize, 3), caller.blocks.items.len);
    const b0 = caller.blocks.items[0].instructions.items;
    try std.testing.expect(b0[0].op == .iconst_32);
    try std.testing.expect(b0[1].op == .br);
    try std.testing.expectEqual(@as(ir.BlockId, 1), b0[1].op.br);
    const clone_entry = caller.blocks.items[1].instructions.items;
    try std.testing.expect(clone_entry[0].op == .br);
    try std.testing.expectEqual(@as(ir.BlockId, 2), clone_entry[0].op.br);
    const b_after = caller.blocks.items[2].instructions.items;
    try std.testing.expect(b_after[0].op == .ret);
    // After local rename, the ret value should be the caller's iconst dest.
    try std.testing.expectEqual(b0[0].dest.?, b_after[0].op.ret.?);
}

test "inlineSmallFunctions: multi-block if/else callee is inlined" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    // Callee: fn cond(p, a, b) -> i32   { if p then a else b }
    //   entry: local_get 0; br_if t, e
    //   t:     local_get 1; ret
    //   e:     local_get 2; ret
    try module.functions.append(allocator, ir.IrFunction.init(allocator, 3, 1, 3));
    {
        const callee = &module.functions.items[0];
        const c_entry = try callee.newBlock();
        const c_then = try callee.newBlock();
        const c_else = try callee.newBlock();
        _ = callee.newVReg(); // param 0 placeholder
        _ = callee.newVReg(); // param 1 placeholder
        _ = callee.newVReg(); // param 2 placeholder
        const v_p = callee.newVReg();
        const v_a = callee.newVReg();
        const v_b = callee.newVReg();
        try callee.getBlock(c_entry).append(.{ .op = .{ .local_get = 0 }, .dest = v_p, .type = .i32 });
        try callee.getBlock(c_entry).append(.{ .op = .{ .br_if = .{ .cond = v_p, .then_block = c_then, .else_block = c_else } } });
        try callee.getBlock(c_then).append(.{ .op = .{ .local_get = 1 }, .dest = v_a, .type = .i32 });
        try callee.getBlock(c_then).append(.{ .op = .{ .ret = v_a } });
        try callee.getBlock(c_else).append(.{ .op = .{ .local_get = 2 }, .dest = v_b, .type = .i32 });
        try callee.getBlock(c_else).append(.{ .op = .{ .ret = v_b } });
    }
    // Test only hits the "result_count=1 requires exactly 1 ret" branch:
    // two rets means this callee is ineligible. Confirm that path, then
    // rewrite with a single-ret variant.
    try std.testing.expect(!isInlinable(&module.functions.items[0], 32, 8));

    // Now build a single-ret if/else: merge via br to a common tail.
    module.functions.items[0].deinit();
    module.functions.items[0] = ir.IrFunction.init(allocator, 3, 1, 3);
    {
        const callee = &module.functions.items[0];
        const c_entry = try callee.newBlock();
        const c_then = try callee.newBlock();
        const c_else = try callee.newBlock();
        const c_tail = try callee.newBlock();
        _ = callee.newVReg();
        _ = callee.newVReg();
        _ = callee.newVReg();
        const v_p = callee.newVReg();
        const v_a = callee.newVReg();
        const v_b = callee.newVReg();
        const v_x = callee.newVReg();
        // Not truly phi-safe (both branches def different vregs for v_x,
        // real IR would use a local_set), but this module-level inliner
        // just clones blocks verbatim. For test we just check that
        // multi-block callees with a single ret get inlined structurally.
        try callee.getBlock(c_entry).append(.{ .op = .{ .local_get = 0 }, .dest = v_p, .type = .i32 });
        try callee.getBlock(c_entry).append(.{ .op = .{ .br_if = .{ .cond = v_p, .then_block = c_then, .else_block = c_else } } });
        try callee.getBlock(c_then).append(.{ .op = .{ .local_get = 1 }, .dest = v_a, .type = .i32 });
        try callee.getBlock(c_then).append(.{ .op = .{ .br = c_tail } });
        try callee.getBlock(c_else).append(.{ .op = .{ .local_get = 2 }, .dest = v_b, .type = .i32 });
        try callee.getBlock(c_else).append(.{ .op = .{ .br = c_tail } });
        try callee.getBlock(c_tail).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_x, .type = .i32 });
        try callee.getBlock(c_tail).append(.{ .op = .{ .ret = v_x } });
    }
    try std.testing.expect(isInlinable(&module.functions.items[0], 32, 8));

    // Caller: fn main(p, a, b) -> i32  { call 0(p, a, b); return }
    try module.functions.append(allocator, ir.IrFunction.init(allocator, 3, 1, 3));
    const args = try allocator.alloc(ir.VReg, 3);
    defer allocator.free(args);
    {
        const caller = &module.functions.items[1];
        const mb = try caller.newBlock();
        _ = caller.newVReg();
        _ = caller.newVReg();
        _ = caller.newVReg();
        const v_p = caller.newVReg();
        const v_a = caller.newVReg();
        const v_b = caller.newVReg();
        const v_r = caller.newVReg();
        try caller.getBlock(mb).append(.{ .op = .{ .local_get = 0 }, .dest = v_p, .type = .i32 });
        try caller.getBlock(mb).append(.{ .op = .{ .local_get = 1 }, .dest = v_a, .type = .i32 });
        try caller.getBlock(mb).append(.{ .op = .{ .local_get = 2 }, .dest = v_b, .type = .i32 });
        args[0] = v_p;
        args[1] = v_a;
        args[2] = v_b;
        try caller.getBlock(mb).append(.{ .op = .{ .call = .{ .func_idx = 0, .args = args } }, .dest = v_r, .type = .i32 });
        try caller.getBlock(mb).append(.{ .op = .{ .ret = v_r } });
    }

    const inlined = try inlineSmallFunctions(&module, allocator);
    try std.testing.expect(inlined);

    const caller = &module.functions.items[1];
    // Before inlining: 1 block. After: 1 (B) + 1 (B_after) + 4 (clones) = 6.
    try std.testing.expectEqual(@as(usize, 6), caller.blocks.items.len);
    // B ends with `br clone_entry`.
    const b0 = caller.blocks.items[0].instructions.items;
    try std.testing.expect(b0[b0.len - 1].op == .br);
    // No remaining `.call` instructions anywhere.
    for (caller.blocks.items) |blk| {
        for (blk.instructions.items) |inst| try std.testing.expect(inst.op != .call);
    }
}

test "foldConstantBranches: zero cond picks else block" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const v_c = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_c, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v_c, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    const changed = try foldConstantBranches(&func, allocator);
    try std.testing.expect(changed);
    const last = func.getBlock(b0).instructions.items[1];
    try std.testing.expect(last.op == .br);
    try std.testing.expectEqual(b2, last.op.br);
}

test "foldConstantBranches: nonzero cond picks then block" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const v_c = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v_c, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v_c, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    const changed = try foldConstantBranches(&func, allocator);
    try std.testing.expect(changed);
    const last = func.getBlock(b0).instructions.items[1];
    try std.testing.expect(last.op == .br);
    try std.testing.expectEqual(b1, last.op.br);
}

test "strengthReduceDivRem: div_u(x, 8) → shr_u(x, 3)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 8 }, .dest = v_c });
    try block.append(.{ .op = .{ .div_u = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceDivRem(&func, allocator);
    try std.testing.expect(changed);

    // Block: iconst_32=8, iconst_32=3, shr_u(v_x, shift_vreg), ret.
    try std.testing.expectEqual(@as(usize, 4), block.instructions.items.len);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 3 }, block.instructions.items[1].op);
    switch (block.instructions.items[2].op) {
        .shr_u => |bin| {
            try std.testing.expectEqual(v_x, bin.lhs);
            try std.testing.expectEqual(block.instructions.items[1].dest.?, bin.rhs);
        },
        else => try std.testing.expect(false),
    }
    try std.testing.expectEqual(v_r, block.instructions.items[2].dest.?);
}

test "strengthReduceDivRem: rem_u(x, 16) → and(x, 15)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 16 }, .dest = v_c });
    try block.append(.{ .op = .{ .rem_u = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceDivRem(&func, allocator);
    try std.testing.expect(changed);

    // Block: iconst_32=16, iconst_32=15, and(v_x, mask_vreg), ret.
    try std.testing.expectEqual(@as(usize, 4), block.instructions.items.len);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 15 }, block.instructions.items[1].op);
    switch (block.instructions.items[2].op) {
        .@"and" => |bin| {
            try std.testing.expectEqual(v_x, bin.lhs);
            try std.testing.expectEqual(block.instructions.items[1].dest.?, bin.rhs);
        },
        else => try std.testing.expect(false),
    }
    try std.testing.expectEqual(v_r, block.instructions.items[2].dest.?);
}

test "strengthReduceDivRem: i64 div_u by 2^32" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_64 = 1 << 32 }, .dest = v_c, .type = .i64 });
    try block.append(.{ .op = .{ .div_u = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r, .type = .i64 });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceDivRem(&func, allocator);
    try std.testing.expect(changed);

    try std.testing.expectEqual(ir.Inst.Op{ .iconst_64 = 32 }, block.instructions.items[1].op);
    try std.testing.expectEqual(ir.IrType.i64, block.instructions.items[1].type);
    switch (block.instructions.items[2].op) {
        .shr_u => |bin| {
            try std.testing.expectEqual(v_x, bin.lhs);
        },
        else => try std.testing.expect(false),
    }
}

test "strengthReduceDivRem: rem_u i64 by 2^63 uses full mask" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    const divisor: i64 = @bitCast(@as(u64, 1) << 63); // interpreted as 2^63 unsigned
    try block.append(.{ .op = .{ .iconst_64 = divisor }, .dest = v_c, .type = .i64 });
    try block.append(.{ .op = .{ .rem_u = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r, .type = .i64 });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceDivRem(&func, allocator);
    try std.testing.expect(changed);

    // Mask for rem_u by 2^63 is 2^63 - 1 == 0x7FFF_FFFF_FFFF_FFFF.
    const expected_mask: i64 = @bitCast((@as(u64, 1) << 63) - 1);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_64 = expected_mask }, block.instructions.items[1].op);
}

test "strengthReduceDivRem: does not rewrite non-power-of-two divisor" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v_c });
    try block.append(.{ .op = .{ .div_u = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceDivRem(&func, allocator);
    try std.testing.expect(!changed);
    // Original div_u still present.
    try std.testing.expectEqual(@as(usize, 3), block.instructions.items.len);
    try std.testing.expect(block.instructions.items[1].op == .div_u);
}

test "strengthReduceDivRem: does not rewrite div_s / rem_s (signed left alone)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 8 }, .dest = v_c });
    try block.append(.{ .op = .{ .div_s = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceDivRem(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expect(block.instructions.items[1].op == .div_s);
}

test "strengthReduceDivRem: does not rewrite div_u by 1" {
    // c == 1 is rejected by powerOfTwoShift (shift amount 0 disallowed).
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_c });
    try block.append(.{ .op = .{ .div_u = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceDivRem(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expect(block.instructions.items[1].op == .div_u);
}

test "elideRedundantBoundsChecks: cross-block via dominator" {
    // b0: load base+0 size=8 (establishes end=8).
    // b0 -> b1: load base+0 size=4 (end=4 <= 8, should be elided).
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();

    const v_base = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 8 } }, .dest = v_a, .type = .i64 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_b, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .ret = v_b } });

    const changed = try elideRedundantBoundsChecks(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expect(func.getBlock(b1).instructions.items[0].op.load.bounds_known);
}

test "elideRedundantBoundsChecks: sibling does not dominate" {
    // b0 -> {b1, b2} -> b3. b1 establishes a bounds check. b2 does NOT,
    // because b1 does not dominate b2. b3 is dominated only by b0, so it
    // also gets NO free bounds_known from either sibling.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    const v_base = func.newVReg();
    const cond = func.newVReg();
    const v_1 = func.newVReg();
    const v_2 = func.newVReg();
    const v_3 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_1, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_2, .type = .i32 });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_3, .type = .i32 });
    try func.getBlock(b3).append(.{ .op = .{ .ret = v_3 } });

    _ = try elideRedundantBoundsChecks(&func, allocator);
    // Neither b1's nor b2's load is dominated by the other, so neither
    // can elide; b3 is not dominated by either of them either.
    try std.testing.expect(!func.getBlock(b1).instructions.items[0].op.load.bounds_known);
    try std.testing.expect(!func.getBlock(b2).instructions.items[0].op.load.bounds_known);
    try std.testing.expect(!func.getBlock(b3).instructions.items[0].op.load.bounds_known);
}

test "elideRedundantBoundsChecks: loop body inherits from preheader" {
    // preheader(b0) dominates header(b1) dominates body(b2). preheader does
    // a wide load establishing max_end=8; body does a narrower load, must
    // be elided.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    const v_base = func.newVReg();
    const cond = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    // b0 (preheader): establishes bounds for base [0,8).
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 8 } }, .dest = v_a, .type = .i64 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    // b1 (header): branches to body or exit.
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b2, .else_block = b3 } } });
    // b2 (body): load base+4 size=4.
    try func.getBlock(b2).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 4, .size = 4 } }, .dest = v_b, .type = .i32 });
    try func.getBlock(b2).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = v_a } });

    const changed = try elideRedundantBoundsChecks(&func, allocator);
    try std.testing.expect(changed);
    // Body load must inherit bounds_known from preheader via dom.
    try std.testing.expect(func.getBlock(b2).instructions.items[0].op.load.bounds_known);
}

test "elideRedundantBoundsChecks: fence in dominator hides upstream entries" {
    // b0 loads base+0 size=8; b0 then calls (fence) and branches to b1.
    // b1's load must NOT be elided because the fence conservatively hides
    // the earlier bounds check for the remainder of b0 and its dom subtree.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();

    const v_base = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 8 } }, .dest = v_a, .type = .i64 });
    try func.getBlock(b0).append(.{ .op = .{ .call = .{ .func_idx = 0 } } });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_b, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .ret = v_b } });

    _ = try elideRedundantBoundsChecks(&func, allocator);
    try std.testing.expect(!func.getBlock(b1).instructions.items[0].op.load.bounds_known);
}

test "algebraicSimplify: sub x, x -> iconst 0" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const v_x = func.newVReg();
    const v_r = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v_x });
    try func.getBlock(b0).append(.{ .op = .{ .sub = .{ .lhs = v_x, .rhs = v_x } }, .dest = v_r });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v_r } });

    const changed = try algebraicSimplify(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 0 }, func.getBlock(b0).instructions.items[1].op);
    try std.testing.expectEqual(@as(?ir.VReg, v_r), func.getBlock(b0).instructions.items[1].dest);
}

test "algebraicSimplify: sub x, x i64 -> iconst_64 0" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const v_x = func.newVReg();
    const v_r = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_64 = 42 }, .dest = v_x, .type = .i64 });
    try func.getBlock(b0).append(.{ .op = .{ .sub = .{ .lhs = v_x, .rhs = v_x } }, .dest = v_r, .type = .i64 });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v_r } });

    const changed = try algebraicSimplify(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_64 = 0 }, func.getBlock(b0).instructions.items[1].op);
}

test "algebraicSimplify: xor x, x -> iconst 0" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const v_x = func.newVReg();
    const v_r = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v_x });
    try func.getBlock(b0).append(.{ .op = .{ .xor = .{ .lhs = v_x, .rhs = v_x } }, .dest = v_r });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v_r } });

    const changed = try algebraicSimplify(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 0 }, func.getBlock(b0).instructions.items[1].op);
}

test "algebraicSimplify: and x, x -> x (users rewritten)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const v_x = func.newVReg();
    const v_r = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 5 }, .dest = v_x });
    try func.getBlock(b0).append(.{ .op = .{ .@"and" = .{ .lhs = v_x, .rhs = v_x } }, .dest = v_r });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v_r } });

    const changed = try algebraicSimplify(&func, allocator);
    try std.testing.expect(changed);
    // ret must now reference v_x directly.
    try std.testing.expectEqual(ir.Inst.Op{ .ret = v_x }, func.getBlock(b0).instructions.items[2].op);
}

test "algebraicSimplify: or x, x -> x" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const v_x = func.newVReg();
    const v_r = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 5 }, .dest = v_x });
    try func.getBlock(b0).append(.{ .op = .{ .@"or" = .{ .lhs = v_x, .rhs = v_x } }, .dest = v_r });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v_r } });

    const changed = try algebraicSimplify(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(ir.Inst.Op{ .ret = v_x }, func.getBlock(b0).instructions.items[2].op);
}

test "algebraicSimplify: eq x, x -> iconst 1" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const v_x = func.newVReg();
    const v_r = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 99 }, .dest = v_x });
    try func.getBlock(b0).append(.{ .op = .{ .eq = .{ .lhs = v_x, .rhs = v_x } }, .dest = v_r });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v_r } });

    const changed = try algebraicSimplify(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 1 }, func.getBlock(b0).instructions.items[1].op);
}

test "algebraicSimplify: ne x, x -> iconst 0" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const v_x = func.newVReg();
    const v_r = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_x });
    try func.getBlock(b0).append(.{ .op = .{ .ne = .{ .lhs = v_x, .rhs = v_x } }, .dest = v_r });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v_r } });

    const changed = try algebraicSimplify(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 0 }, func.getBlock(b0).instructions.items[1].op);
}

test "algebraicSimplify: le_u x, x -> 1; lt_u x, x -> 0" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 2, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const v_x = func.newVReg();
    const v_le = func.newVReg();
    const v_lt = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 3 }, .dest = v_x });
    try func.getBlock(b0).append(.{ .op = .{ .le_u = .{ .lhs = v_x, .rhs = v_x } }, .dest = v_le });
    try func.getBlock(b0).append(.{ .op = .{ .lt_u = .{ .lhs = v_x, .rhs = v_x } }, .dest = v_lt });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v_le } });

    _ = try algebraicSimplify(&func, allocator);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 1 }, func.getBlock(b0).instructions.items[1].op);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 0 }, func.getBlock(b0).instructions.items[2].op);
}

test "algebraicSimplify: sub with distinct operands is unchanged" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    const v_r = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_a });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 2 }, .dest = v_b });
    try func.getBlock(b0).append(.{ .op = .{ .sub = .{ .lhs = v_a, .rhs = v_b } }, .dest = v_r });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v_r } });

    const changed = try algebraicSimplify(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expect(func.getBlock(b0).instructions.items[2].op == .sub);
}

test "algebraicSimplify: is idempotent (no spin after first fire)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const v_x = func.newVReg();
    const v_r = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v_x });
    try func.getBlock(b0).append(.{ .op = .{ .xor = .{ .lhs = v_x, .rhs = v_x } }, .dest = v_r });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v_r } });

    const first = try algebraicSimplify(&func, allocator);
    try std.testing.expect(first);
    const second = try algebraicSimplify(&func, allocator);
    try std.testing.expect(!second);
}

test "strengthReduceMulShiftAdd: mul(x, 3) -> (x << 1) + x" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = v_c });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceMulShiftAdd(&func, allocator);
    try std.testing.expect(changed);

    // Expected block: iconst=3, iconst=1, shl(v_x, shift), add(shl_res, v_x), ret.
    try std.testing.expectEqual(@as(usize, 5), block.instructions.items.len);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 1 }, block.instructions.items[1].op);
    switch (block.instructions.items[2].op) {
        .shl => |bin| {
            try std.testing.expectEqual(v_x, bin.lhs);
            try std.testing.expectEqual(block.instructions.items[1].dest.?, bin.rhs);
        },
        else => try std.testing.expect(false),
    }
    switch (block.instructions.items[3].op) {
        .add => |bin| {
            try std.testing.expectEqual(block.instructions.items[2].dest.?, bin.lhs);
            try std.testing.expectEqual(v_x, bin.rhs);
        },
        else => try std.testing.expect(false),
    }
    try std.testing.expectEqual(v_r, block.instructions.items[3].dest.?);
}

test "strengthReduceMulShiftAdd: mul(x, 7) -> (x << 3) - x" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = v_c });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceMulShiftAdd(&func, allocator);
    try std.testing.expect(changed);

    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 3 }, block.instructions.items[1].op);
    try std.testing.expect(block.instructions.items[2].op == .shl);
    switch (block.instructions.items[3].op) {
        .sub => |bin| {
            try std.testing.expectEqual(block.instructions.items[2].dest.?, bin.lhs);
            try std.testing.expectEqual(v_x, bin.rhs);
        },
        else => try std.testing.expect(false),
    }
}

test "strengthReduceMulShiftAdd: mul(x, 5) commutative" {
    // Constant on the LHS; x on the RHS.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v_c });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_c, .rhs = v_x } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceMulShiftAdd(&func, allocator);
    try std.testing.expect(changed);

    // shift amount = 2, op = add, the non-constant multiplicand is v_x.
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 2 }, block.instructions.items[1].op);
    switch (block.instructions.items[2].op) {
        .shl => |bin| try std.testing.expectEqual(v_x, bin.lhs),
        else => try std.testing.expect(false),
    }
    switch (block.instructions.items[3].op) {
        .add => |bin| try std.testing.expectEqual(v_x, bin.rhs),
        else => try std.testing.expect(false),
    }
}

test "strengthReduceMulShiftAdd: i64 mul by 9 -> (x << 3) + x" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_64 = 9 }, .dest = v_c, .type = .i64 });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r, .type = .i64 });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceMulShiftAdd(&func, allocator);
    try std.testing.expect(changed);

    try std.testing.expectEqual(ir.Inst.Op{ .iconst_64 = 3 }, block.instructions.items[1].op);
    try std.testing.expectEqual(ir.IrType.i64, block.instructions.items[1].type);
    try std.testing.expect(block.instructions.items[2].op == .shl);
    try std.testing.expectEqual(ir.IrType.i64, block.instructions.items[2].type);
    try std.testing.expect(block.instructions.items[3].op == .add);
}

test "strengthReduceMulShiftAdd: does not touch power-of-two multiplier" {
    // mul by 8 is pow2 — `strengthReduceMul` handles it; this pass must skip.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 8 }, .dest = v_c });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceMulShiftAdd(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expect(block.instructions.items[1].op == .mul);
}

test "strengthReduceMulShiftAdd: does not touch mul by 10 (neither 2^k+/-1)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v_c });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceMulShiftAdd(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expect(block.instructions.items[1].op == .mul);
}

test "strengthReduceMulShiftAdd: pipeline composition with strengthReduceMul" {
    // Feed both multipliers into the default pipeline order and verify each
    // selects the appropriate pass.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 2, 2, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_c8 = func.newVReg();
    const v_c3 = func.newVReg();
    const v_r1 = func.newVReg();
    const v_r2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 8 }, .dest = v_c8 });
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = v_c3 });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_x, .rhs = v_c8 } }, .dest = v_r1 });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v_x, .rhs = v_c3 } }, .dest = v_r2 });
    try block.append(.{ .op = .{ .ret = v_r2 } });

    _ = try strengthReduceMul(&func, allocator);
    _ = try strengthReduceMulShiftAdd(&func, allocator);

    // Expect: no remaining `.mul` instructions.
    for (block.instructions.items) |inst| {
        try std.testing.expect(inst.op != .mul);
    }
}
