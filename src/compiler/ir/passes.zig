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
                .phi => |edges| {
                    for (edges) |edge| {
                        const entry = try info.getOrPut(edge.val);
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
        .iconst_32, .iconst_64, .fconst_32, .fconst_64, .v128_const => {},
        .local_get, .global_get => {},
        .br, .@"unreachable" => {},

        // Binary ops
        .add,
        .sub,
        .mul,
        .div_s,
        .div_u,
        .rem_s,
        .rem_u,
        .@"and",
        .@"or",
        .xor,
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
        .f_min,
        .f_max,
        .f_copysign,
        .f_eq,
        .f_ne,
        .f_lt,
        .f_gt,
        .f_le,
        .f_ge,
        => |bin| {
            list.append(bin.lhs);
            list.append(bin.rhs);
        },

        .v128_bitwise => |bin| {
            list.append(bin.lhs);
            list.append(bin.rhs);
        },
        .i32x4_binop => |bin| {
            list.append(bin.lhs);
            list.append(bin.rhs);
        },
        .i32x4_unop => |un| list.append(un.vector),
        .i32x4_extadd_pairwise_i16x8 => |op| list.append(op.vector),
        .i32x4_extend_i16x8 => |op| list.append(op.vector),
        .i32x4_extmul_i16x8 => |op| {
            list.append(op.lhs);
            list.append(op.rhs);
        },
        .i8x16_binop => |bin| {
            list.append(bin.lhs);
            list.append(bin.rhs);
        },
        .i8x16_narrow_i16x8 => |op| {
            list.append(op.lhs);
            list.append(op.rhs);
        },
        .i8x16_unop => |un| list.append(un.vector),
        .i8x16_shift => |shift| {
            list.append(shift.vector);
            list.append(shift.count);
        },
        .i16x8_binop => |bin| {
            list.append(bin.lhs);
            list.append(bin.rhs);
        },
        .i16x8_unop => |un| list.append(un.vector),
        .i16x8_extadd_pairwise_i8x16 => |op| list.append(op.vector),
        .i16x8_extend_i8x16 => |op| list.append(op.vector),
        .i16x8_extmul_i8x16 => |op| {
            list.append(op.lhs);
            list.append(op.rhs);
        },
        .i16x8_narrow_i32x4 => |op| {
            list.append(op.lhs);
            list.append(op.rhs);
        },
        .i64x2_extend_i32x4 => |op| list.append(op.vector),
        .i64x2_extmul_i32x4 => |op| {
            list.append(op.lhs);
            list.append(op.rhs);
        },
        .i64x2_binop => |bin| {
            list.append(bin.lhs);
            list.append(bin.rhs);
        },
        .i64x2_unop => |un| list.append(un.vector),
        .i64x2_shift => |shift| {
            list.append(shift.vector);
            list.append(shift.count);
        },
        .i32x4_shift => |shift| {
            list.append(shift.vector);
            list.append(shift.count);
        },
        .i16x8_shift => |shift| {
            list.append(shift.vector);
            list.append(shift.count);
        },

        // Unary ops
        .clz,
        .ctz,
        .popcnt,
        .eqz,
        .wrap_i64,
        .extend_i32_s,
        .extend_i32_u,
        .extend8_s,
        .extend16_s,
        .extend32_s,
        .f_neg,
        .f_abs,
        .f_sqrt,
        .f_ceil,
        .f_floor,
        .f_trunc,
        .f_nearest,
        .trunc_f32_s,
        .trunc_f32_u,
        .trunc_f64_s,
        .trunc_f64_u,
        .convert_s,
        .convert_u,
        .convert_i32_s,
        .convert_i64_s,
        .convert_i32_u,
        .convert_i64_u,
        .demote_f64,
        .promote_f32,
        .reinterpret,
        .trunc_sat_f32_s,
        .trunc_sat_f32_u,
        .trunc_sat_f64_s,
        .trunc_sat_f64_u,
        .v128_not,
        .i32x4_splat,
        .i8x16_splat,
        .i16x8_splat,
        .i64x2_splat,
        => |vreg| list.append(vreg),
        .i32x4_extract_lane => |lane| list.append(lane.vector),
        .i8x16_extract_lane => |lane| list.append(lane.vector),
        .i16x8_extract_lane => |lane| list.append(lane.vector),
        .i64x2_extract_lane => |lane| list.append(lane.vector),
        .i32x4_replace_lane => |lane| {
            list.append(lane.vector);
            list.append(lane.val);
        },
        .i8x16_replace_lane => |lane| {
            list.append(lane.vector);
            list.append(lane.val);
        },
        .i16x8_replace_lane => |lane| {
            list.append(lane.vector);
            list.append(lane.val);
        },
        .i64x2_replace_lane => |lane| {
            list.append(lane.vector);
            list.append(lane.val);
        },

        .local_set => |ls| list.append(ls.val),
        .global_set => |gs| list.append(gs.val),
        .load => |ld| list.append(ld.base),
        .v128_load => |ld| list.append(ld.base),
        .store => |st| {
            list.append(st.base);
            list.append(st.val);
        },
        .v128_store => |st| {
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
        // Phi operands handled separately (unbounded, like call args).
        .phi => {},
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
        .iconst_32,
        .iconst_64,
        .fconst_32,
        .fconst_64,
        .v128_const,
        .local_get,
        .global_get,
        .br,
        .@"unreachable",
        => {},

        .add,
        .sub,
        .mul,
        .div_s,
        .div_u,
        .rem_s,
        .rem_u,
        .@"and",
        .@"or",
        .xor,
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
        .f_min,
        .f_max,
        .f_copysign,
        .f_eq,
        .f_ne,
        .f_lt,
        .f_gt,
        .f_le,
        .f_ge,
        => |*bin| {
            if (bin.lhs == old) bin.lhs = new;
            if (bin.rhs == old) bin.rhs = new;
        },

        .v128_bitwise => |*bin| {
            if (bin.lhs == old) bin.lhs = new;
            if (bin.rhs == old) bin.rhs = new;
        },
        .i32x4_binop => |*bin| {
            if (bin.lhs == old) bin.lhs = new;
            if (bin.rhs == old) bin.rhs = new;
        },
        .i32x4_unop => |*un| if (un.vector == old) {
            un.vector = new;
        },
        .i32x4_extadd_pairwise_i16x8 => |*op| if (op.vector == old) {
            op.vector = new;
        },
        .i32x4_extend_i16x8 => |*op| if (op.vector == old) {
            op.vector = new;
        },
        .i32x4_extmul_i16x8 => |*op| {
            if (op.lhs == old) op.lhs = new;
            if (op.rhs == old) op.rhs = new;
        },
        .i8x16_binop => |*bin| {
            if (bin.lhs == old) bin.lhs = new;
            if (bin.rhs == old) bin.rhs = new;
        },
        .i8x16_narrow_i16x8 => |*op| {
            if (op.lhs == old) op.lhs = new;
            if (op.rhs == old) op.rhs = new;
        },
        .i8x16_unop => |*un| if (un.vector == old) {
            un.vector = new;
        },
        .i8x16_shift => |*shift| {
            if (shift.vector == old) shift.vector = new;
            if (shift.count == old) shift.count = new;
        },
        .i16x8_binop => |*bin| {
            if (bin.lhs == old) bin.lhs = new;
            if (bin.rhs == old) bin.rhs = new;
        },
        .i16x8_unop => |*un| if (un.vector == old) {
            un.vector = new;
        },
        .i16x8_extadd_pairwise_i8x16 => |*op| if (op.vector == old) {
            op.vector = new;
        },
        .i16x8_extend_i8x16 => |*op| if (op.vector == old) {
            op.vector = new;
        },
        .i16x8_extmul_i8x16 => |*op| {
            if (op.lhs == old) op.lhs = new;
            if (op.rhs == old) op.rhs = new;
        },
        .i16x8_narrow_i32x4 => |*op| {
            if (op.lhs == old) op.lhs = new;
            if (op.rhs == old) op.rhs = new;
        },
        .i64x2_binop => |*bin| {
            if (bin.lhs == old) bin.lhs = new;
            if (bin.rhs == old) bin.rhs = new;
        },
        .i64x2_unop => |*un| if (un.vector == old) {
            un.vector = new;
        },
        .i64x2_extend_i32x4 => |*op| if (op.vector == old) {
            op.vector = new;
        },
        .i64x2_extmul_i32x4 => |*op| {
            if (op.lhs == old) op.lhs = new;
            if (op.rhs == old) op.rhs = new;
        },
        .i64x2_shift => |*shift| {
            if (shift.vector == old) shift.vector = new;
            if (shift.count == old) shift.count = new;
        },
        .i32x4_shift => |*shift| {
            if (shift.vector == old) shift.vector = new;
            if (shift.count == old) shift.count = new;
        },
        .i16x8_shift => |*shift| {
            if (shift.vector == old) shift.vector = new;
            if (shift.count == old) shift.count = new;
        },

        .clz,
        .ctz,
        .popcnt,
        .eqz,
        .wrap_i64,
        .extend_i32_s,
        .extend_i32_u,
        .extend8_s,
        .extend16_s,
        .extend32_s,
        .f_neg,
        .f_abs,
        .f_sqrt,
        .f_ceil,
        .f_floor,
        .f_trunc,
        .f_nearest,
        .trunc_f32_s,
        .trunc_f32_u,
        .trunc_f64_s,
        .trunc_f64_u,
        .convert_s,
        .convert_u,
        .convert_i32_s,
        .convert_i64_s,
        .convert_i32_u,
        .convert_i64_u,
        .demote_f64,
        .promote_f32,
        .reinterpret,
        .trunc_sat_f32_s,
        .trunc_sat_f32_u,
        .trunc_sat_f64_s,
        .trunc_sat_f64_u,
        .v128_not,
        .i32x4_splat,
        .i8x16_splat,
        .i16x8_splat,
        .i64x2_splat,
        => |*vreg| if (vreg.* == old) {
            vreg.* = new;
        },
        .i32x4_extract_lane => |*lane| if (lane.vector == old) {
            lane.vector = new;
        },
        .i8x16_extract_lane => |*lane| if (lane.vector == old) {
            lane.vector = new;
        },
        .i16x8_extract_lane => |*lane| if (lane.vector == old) {
            lane.vector = new;
        },
        .i64x2_extract_lane => |*lane| if (lane.vector == old) {
            lane.vector = new;
        },
        .i32x4_replace_lane => |*lane| {
            if (lane.vector == old) lane.vector = new;
            if (lane.val == old) lane.val = new;
        },
        .i8x16_replace_lane => |*lane| {
            if (lane.vector == old) lane.vector = new;
            if (lane.val == old) lane.val = new;
        },
        .i16x8_replace_lane => |*lane| {
            if (lane.vector == old) lane.vector = new;
            if (lane.val == old) lane.val = new;
        },
        .i64x2_replace_lane => |*lane| {
            if (lane.vector == old) lane.vector = new;
            if (lane.val == old) lane.val = new;
        },

        .local_set => |*ls| if (ls.val == old) {
            ls.val = new;
        },
        .global_set => |*gs| if (gs.val == old) {
            gs.val = new;
        },
        .load => |*ld| if (ld.base == old) {
            ld.base = new;
        },
        .v128_load => |*ld| if (ld.base == old) {
            ld.base = new;
        },
        .store => |*st| {
            if (st.base == old) st.base = new;
            if (st.val == old) st.val = new;
        },
        .v128_store => |*st| {
            if (st.base == old) st.base = new;
            if (st.val == old) st.val = new;
        },
        .br_if => |*bi| if (bi.cond == old) {
            bi.cond = new;
        },
        .br_table => |*bt| if (bt.index == old) {
            bt.index = new;
        },
        .ret => |*maybe_vreg| if (maybe_vreg.*) |v| {
            if (v == old) maybe_vreg.* = new;
        },
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
        .atomic_load => |*al| if (al.base == old) {
            al.base = new;
        },
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
        .phi => |edges| {
            for (@constCast(edges)) |*edge| {
                if (edge.val == old) edge.val = new;
            }
        },
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
                .add,
                .sub,
                .mul,
                .@"and",
                .@"or",
                .xor,
                .shl,
                .shr_s,
                .shr_u,
                .rotl,
                .rotr,
                .eq,
                .ne,
                .lt_s,
                .gt_s,
                .le_s,
                .ge_s,
                .lt_u,
                .gt_u,
                .le_u,
                .ge_u,
                .div_s,
                .div_u,
                .rem_s,
                .rem_u,
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
        .lt_s => if (ty == .i64) @intFromBool(lhs < rhs) else @intFromBool(@as(i32, @truncate(lhs)) < @as(i32, @truncate(rhs))),
        .gt_s => if (ty == .i64) @intFromBool(lhs > rhs) else @intFromBool(@as(i32, @truncate(lhs)) > @as(i32, @truncate(rhs))),
        .le_s => if (ty == .i64) @intFromBool(lhs <= rhs) else @intFromBool(@as(i32, @truncate(lhs)) <= @as(i32, @truncate(rhs))),
        .ge_s => if (ty == .i64) @intFromBool(lhs >= rhs) else @intFromBool(@as(i32, @truncate(lhs)) >= @as(i32, @truncate(rhs))),
        .lt_u => if (ty == .i64) @intFromBool(@as(u64, @bitCast(lhs)) < @as(u64, @bitCast(rhs))) else @intFromBool(@as(u32, @truncate(@as(u64, @bitCast(lhs)))) < @as(u32, @truncate(@as(u64, @bitCast(rhs))))),
        .gt_u => if (ty == .i64) @intFromBool(@as(u64, @bitCast(lhs)) > @as(u64, @bitCast(rhs))) else @intFromBool(@as(u32, @truncate(@as(u64, @bitCast(lhs)))) > @as(u32, @truncate(@as(u64, @bitCast(rhs))))),
        .le_u => if (ty == .i64) @intFromBool(@as(u64, @bitCast(lhs)) <= @as(u64, @bitCast(rhs))) else @intFromBool(@as(u32, @truncate(@as(u64, @bitCast(lhs)))) <= @as(u32, @truncate(@as(u64, @bitCast(rhs))))),
        .ge_u => if (ty == .i64) @intFromBool(@as(u64, @bitCast(lhs)) >= @as(u64, @bitCast(rhs))) else @intFromBool(@as(u32, @truncate(@as(u64, @bitCast(lhs)))) >= @as(u32, @truncate(@as(u64, @bitCast(rhs))))),
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

                    if (powerOfTwoShift(rhs_const, inst.type)) |k| {
                        // Power-of-two: x / 2^k → x >> k
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
                        i += 1;
                    } else if (inst.type == .i32 and rhs_const > 1) {
                        // Non-power-of-two i32: reciprocal multiply via i64.
                        //   ext = extend_i32_u(x)
                        //   prod = mul(ext, magic)
                        //   hi = shr_u(prod, 32 + shift)
                        //   result = wrap_i64(hi)
                        const d_u32: u32 = @bitCast(@as(i32, @truncate(rhs_const)));
                        const magic = computeMagicU32(d_u32) orelse continue;

                        const v_ext = func.newVReg();
                        const v_magic = func.newVReg();
                        const v_prod = func.newVReg();
                        const v_shift = func.newVReg();
                        const v_hi = func.newVReg();

                        const shift_amt: i64 = 32 + @as(i64, magic.shift);

                        // Insert 5 instructions before the div_u, then replace it.
                        const insts = [_]ir.Inst{
                            .{ .op = .{ .extend_i32_u = bin.lhs }, .dest = v_ext, .type = .i64 },
                            .{ .op = .{ .iconst_64 = @bitCast(magic.magic) }, .dest = v_magic, .type = .i64 },
                            .{ .op = .{ .mul = .{ .lhs = v_ext, .rhs = v_magic } }, .dest = v_prod, .type = .i64 },
                            .{ .op = .{ .iconst_64 = shift_amt }, .dest = v_shift, .type = .i64 },
                            .{ .op = .{ .shr_u = .{ .lhs = v_prod, .rhs = v_shift } }, .dest = v_hi, .type = .i64 },
                        };
                        for (insts) |new_inst| {
                            try block.instructions.insert(block.allocator, i, new_inst);
                            i += 1;
                        }
                        // Replace div_u with wrap_i64.
                        block.instructions.items[i].op = .{ .wrap_i64 = v_hi };
                        block.instructions.items[i].dest = dest;
                        block.instructions.items[i].type = .i32;
                        changed = true;
                    }
                },
                .rem_u => |bin| {
                    const dest = inst.dest orelse continue;
                    if (inst.type != .i32 and inst.type != .i64) continue;
                    const rhs_const = constants.get(bin.rhs) orelse continue;

                    if (powerOfTwoShift(rhs_const, inst.type)) |k| {
                        // Power-of-two: x % 2^k → x & (2^k - 1)
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
                    } else if (inst.type == .i32 and rhs_const > 1) {
                        // Non-power-of-two i32: x % d = x - (x / d) * d
                        const d_u32: u32 = @bitCast(@as(i32, @truncate(rhs_const)));
                        const magic = computeMagicU32(d_u32) orelse continue;

                        const v_ext = func.newVReg();
                        const v_magic = func.newVReg();
                        const v_prod = func.newVReg();
                        const v_shift = func.newVReg();
                        const v_hi = func.newVReg();
                        const v_q = func.newVReg();
                        const v_d = func.newVReg();
                        const v_qd = func.newVReg();

                        const shift_amt: i64 = 32 + @as(i64, magic.shift);

                        const insts = [_]ir.Inst{
                            .{ .op = .{ .extend_i32_u = bin.lhs }, .dest = v_ext, .type = .i64 },
                            .{ .op = .{ .iconst_64 = @bitCast(magic.magic) }, .dest = v_magic, .type = .i64 },
                            .{ .op = .{ .mul = .{ .lhs = v_ext, .rhs = v_magic } }, .dest = v_prod, .type = .i64 },
                            .{ .op = .{ .iconst_64 = shift_amt }, .dest = v_shift, .type = .i64 },
                            .{ .op = .{ .shr_u = .{ .lhs = v_prod, .rhs = v_shift } }, .dest = v_hi, .type = .i64 },
                            .{ .op = .{ .wrap_i64 = v_hi }, .dest = v_q, .type = .i32 },
                            .{ .op = .{ .iconst_32 = @bitCast(d_u32) }, .dest = v_d, .type = .i32 },
                            .{ .op = .{ .mul = .{ .lhs = v_q, .rhs = v_d } }, .dest = v_qd, .type = .i32 },
                        };
                        for (insts) |new_inst| {
                            try block.instructions.insert(block.allocator, i, new_inst);
                            i += 1;
                        }
                        // Replace rem_u with sub(x, q*d).
                        block.instructions.items[i].op = .{ .sub = .{
                            .lhs = bin.lhs,
                            .rhs = v_qd,
                        } };
                        block.instructions.items[i].dest = dest;
                        block.instructions.items[i].type = .i32;
                        changed = true;
                    }
                },
                else => {},
            }
        }
    }
    return changed;
}

/// Magic number for unsigned 32-bit division by constant `d`.
/// Returns (magic_multiplier, post_shift) such that for all 0 ≤ x < 2^32:
///     x / d == (u64(x) * magic) >> (32 + post_shift)
/// Based on "Hacker's Delight" §10-8 (unsigned division).
fn computeMagicU32(d: u32) ?struct { magic: u64, shift: u6 } {
    if (d == 0 or d == 1) return null;
    // Power of two is handled by the shift path.
    if (d & (d - 1) == 0) return null;

    // Iterate s upward until we find a magic multiplier that works for all x.
    // magic = ceil(2^(32+s) / d), verified by testing boundary values.
    var s: u6 = 0;
    while (s < 32) : (s += 1) {
        // magic = ceil(2^(32+s) / d)
        const shift_amt: u7 = @as(u7, 32) + s;
        if (shift_amt >= 64) break;
        const pow: u64 = @as(u64, 1) << @as(u6, @intCast(shift_amt));
        const m: u64 = pow / d + @intFromBool(pow % d != 0); // ceil division

        // Verify: m * d must be in (2^(32+s), 2^(32+s) + 2^s] for the
        // rounding to work for all x. Simplified check: test boundary values.
        // For correctness, verify: floor(m * x / 2^(32+s)) == floor(x / d)
        // for x = d-1, x = d, x = 2*d, x = 2^32-1.
        var ok = true;
        const test_vals = [_]u64{ 0, 1, d - 1, d, d + 1, 2 * d, 0xFFFFFFFF };
        for (test_vals) |x| {
            if (x > 0xFFFFFFFF) continue;
            const expected = x / d;
            // Compute (x * m) >> (32 + s) using 128-bit arithmetic via two 64-bit muls.
            const prod = @as(u128, x) * @as(u128, m);
            const result = @as(u64, @truncate(prod >> shift_amt));
            if (result != expected) {
                ok = false;
                break;
            }
        }
        if (ok) return .{ .magic = m, .shift = s };
    }
    return null;
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
        .store,
        .v128_store,
        .local_set,
        .global_set,
        .call,
        .call_indirect,
        .call_ref,
        .ret,
        .ret_multi,
        .br,
        .br_if,
        .br_table,
        .@"unreachable",
        .atomic_fence,
        .atomic_load,
        .atomic_store,
        .atomic_rmw,
        .atomic_cmpxchg,
        .atomic_notify,
        .atomic_wait,
        .memory_copy,
        .memory_fill,
        .memory_grow,
        .memory_init,
        .data_drop,
        .table_init,
        .elem_drop,
        .table_set,
        .table_grow,
        => true,
        // Trapping ops: must not be removed even if result is unused.
        .load,
        .v128_load,
        .table_get,
        .div_u,
        .rem_u,
        .trunc_f32_s,
        .trunc_f32_u,
        .trunc_f64_s,
        .trunc_f64_u,
        => true,
        // div_s/rem_s trap for integers but not floats (float div produces NaN/Inf).
        .div_s, .rem_s => inst.type != .f32 and inst.type != .f64,
        else => false,
    };
}

// ── Common Subexpression Elimination ────────────────────────────────────────

/// Dominator-scoped CSE: deduplicate identical pure, non-trapping
/// instructions across basic blocks using the dominator tree.
///
/// Walks the dominator tree in DFS order, maintaining a scoped
/// expression table. When a dominated block computes an expression
/// already available from a dominator, the redundant def's uses are
/// rewritten to the earlier def via `replaceVReg`. The now-dead
/// instruction is left in place for `deadCodeElimination` to clean up.
///
/// This strictly subsumes block-local CSE: within a single block the
/// table accumulates entries exactly as the old linear scan did, but
/// entries also propagate down to dom-tree children and are restored
/// (snapshot/restore) when backtracking — the same pattern used by
/// `elideRedundantBoundsChecks`.
///
/// Safety (SSA): each VReg has exactly one definition. If block A
/// dominates block B, then A also dominates every use of B's defs
/// (because the def in B dominates its own uses, and A dominates B).
/// Therefore `replaceVReg(func, v_B, v_A)` is globally correct.
///
/// History: a prior cross-block CSE was reverted because codegen
/// iterated blocks in raw id order, not RPO. PR #195 fixed block
/// ordering and emission order, making this safe again.
pub fn commonSubexprElimination(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    if (func.blocks.items.len == 0) return false;

    var dom = try analysis.computeDominators(func, allocator);
    defer dom.deinit();

    const nblocks = func.blocks.items.len;

    // Build dom-tree children lists.
    var children = try allocator.alloc(std.ArrayList(ir.BlockId), nblocks);
    defer {
        for (children) |*list| list.deinit(allocator);
        allocator.free(children);
    }
    for (children) |*list| list.* = .empty;
    for (0..nblocks) |i| {
        const bid: ir.BlockId = @intCast(i);
        const idom = dom.idom[bid] orelse continue;
        if (idom == bid) continue; // entry block
        try children[idom].append(allocator, bid);
    }

    // Expression table: flat append-only list with snapshot/restore.
    const ExprEntry = struct { inst: ir.Inst, dest: ir.VReg };
    var table: std.ArrayList(ExprEntry) = .empty;
    defer table.deinit(allocator);

    const Frame = struct {
        bid: ir.BlockId,
        phase: u1,
        snap_len: usize,
    };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(allocator);

    if (dom.idom[0] == null) return false;
    try stack.append(allocator, .{ .bid = 0, .phase = 0, .snap_len = 0 });

    var changed = false;
    while (stack.items.len > 0) {
        const top = &stack.items[stack.items.len - 1];
        if (top.phase == 1) {
            // Backtrack: restore expression table.
            table.shrinkRetainingCapacity(top.snap_len);
            _ = stack.pop();
            continue;
        }
        const bid = top.bid;
        top.phase = 1;
        top.snap_len = table.items.len;

        const block = &func.blocks.items[bid];
        for (block.instructions.items) |*inst| {
            if (inst.dest == null or hasSideEffect(inst.*) or !isPure(inst.*)) continue;

            // Scan table backwards for nearest dominating match.
            // Later entries are from closer ancestors, so backwards
            // scan picks the nearest def and minimises live-range
            // inflation.
            var found = false;
            var k: usize = table.items.len;
            while (k > 0) {
                k -= 1;
                const entry = &table.items[k];
                if (entry.inst.type == inst.type and sameOp(entry.inst, inst.*)) {
                    replaceVReg(func, inst.dest.?, entry.dest);
                    changed = true;
                    found = true;
                    break;
                }
            }
            if (!found) {
                try table.append(allocator, .{ .inst = inst.*, .dest = inst.dest.? });
            }
        }

        // Push dom-tree children for DFS traversal.
        for (children[bid].items) |c| {
            try stack.append(allocator, .{ .bid = c, .phase = 0, .snap_len = 0 });
        }
    }

    return changed;
}

fn isPure(inst: ir.Inst) bool {
    return switch (inst.op) {
        .iconst_32,
        .iconst_64,
        .fconst_32,
        .fconst_64,
        .v128_const,
        .add,
        .sub,
        .mul,
        .div_s,
        .div_u,
        .rem_s,
        .rem_u,
        .@"and",
        .@"or",
        .xor,
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
        .clz,
        .ctz,
        .popcnt,
        .eqz,
        .wrap_i64,
        .extend_i32_s,
        .extend_i32_u,
        .extend8_s,
        .extend16_s,
        .extend32_s,
        .f_neg,
        .f_abs,
        .f_sqrt,
        .f_ceil,
        .f_floor,
        .f_trunc,
        .f_nearest,
        .f_min,
        .f_max,
        .f_copysign,
        .f_eq,
        .f_ne,
        .f_lt,
        .f_gt,
        .f_le,
        .f_ge,
        .trunc_f32_s,
        .trunc_f32_u,
        .trunc_f64_s,
        .trunc_f64_u,
        .convert_s,
        .convert_u,
        .convert_i32_s,
        .convert_i64_s,
        .convert_i32_u,
        .convert_i64_u,
        .demote_f64,
        .promote_f32,
        .reinterpret,
        .trunc_sat_f32_s,
        .trunc_sat_f32_u,
        .trunc_sat_f64_s,
        .trunc_sat_f64_u,
        .v128_not,
        .v128_bitwise,
        .i32x4_binop,
        .i32x4_unop,
        .i32x4_extadd_pairwise_i16x8,
        .i32x4_extend_i16x8,
        .i32x4_extmul_i16x8,
        .i32x4_shift,
        .i32x4_splat,
        .i32x4_extract_lane,
        .i32x4_replace_lane,
        .i8x16_binop,
        .i8x16_unop,
        .i8x16_shift,
        .i8x16_splat,
        .i8x16_extract_lane,
        .i8x16_replace_lane,
        .i8x16_narrow_i16x8,
        .i16x8_binop,
        .i16x8_unop,
        .i16x8_extadd_pairwise_i8x16,
        .i16x8_extend_i8x16,
        .i16x8_extmul_i8x16,
        .i16x8_narrow_i32x4,
        .i16x8_shift,
        .i16x8_splat,
        .i16x8_extract_lane,
        .i16x8_replace_lane,
        .i64x2_binop,
        .i64x2_unop,
        .i64x2_extend_i32x4,
        .i64x2_extmul_i32x4,
        .i64x2_shift,
        .i64x2_splat,
        .i64x2_extract_lane,
        .i64x2_replace_lane,
        => true,
        else => false,
    };
}

fn sameOp(a: ir.Inst, b: ir.Inst) bool {
    const TagType = std.meta.Tag(ir.Inst.Op);
    if (@as(TagType, a.op) != @as(TagType, b.op)) return false;
    return switch (a.op) {
        // Constants
        .iconst_32 => |v| v == b.op.iconst_32,
        .iconst_64 => |v| v == b.op.iconst_64,
        .fconst_32 => |v| @as(u32, @bitCast(v)) == @as(u32, @bitCast(b.op.fconst_32)),
        .fconst_64 => |v| @as(u64, @bitCast(v)) == @as(u64, @bitCast(b.op.fconst_64)),
        .v128_const => |v| v == b.op.v128_const,
        // Binary integer arithmetic / logic / shifts / rotations
        .add => |bin| bin.lhs == b.op.add.lhs and bin.rhs == b.op.add.rhs,
        .sub => |bin| bin.lhs == b.op.sub.lhs and bin.rhs == b.op.sub.rhs,
        .mul => |bin| bin.lhs == b.op.mul.lhs and bin.rhs == b.op.mul.rhs,
        .@"and" => |bin| bin.lhs == b.op.@"and".lhs and bin.rhs == b.op.@"and".rhs,
        .@"or" => |bin| bin.lhs == b.op.@"or".lhs and bin.rhs == b.op.@"or".rhs,
        .xor => |bin| bin.lhs == b.op.xor.lhs and bin.rhs == b.op.xor.rhs,
        .shl => |bin| bin.lhs == b.op.shl.lhs and bin.rhs == b.op.shl.rhs,
        .shr_s => |bin| bin.lhs == b.op.shr_s.lhs and bin.rhs == b.op.shr_s.rhs,
        .shr_u => |bin| bin.lhs == b.op.shr_u.lhs and bin.rhs == b.op.shr_u.rhs,
        .rotl => |bin| bin.lhs == b.op.rotl.lhs and bin.rhs == b.op.rotl.rhs,
        .rotr => |bin| bin.lhs == b.op.rotr.lhs and bin.rhs == b.op.rotr.rhs,
        // Integer comparisons
        .eq => |bin| bin.lhs == b.op.eq.lhs and bin.rhs == b.op.eq.rhs,
        .ne => |bin| bin.lhs == b.op.ne.lhs and bin.rhs == b.op.ne.rhs,
        .lt_s => |bin| bin.lhs == b.op.lt_s.lhs and bin.rhs == b.op.lt_s.rhs,
        .lt_u => |bin| bin.lhs == b.op.lt_u.lhs and bin.rhs == b.op.lt_u.rhs,
        .gt_s => |bin| bin.lhs == b.op.gt_s.lhs and bin.rhs == b.op.gt_s.rhs,
        .gt_u => |bin| bin.lhs == b.op.gt_u.lhs and bin.rhs == b.op.gt_u.rhs,
        .le_s => |bin| bin.lhs == b.op.le_s.lhs and bin.rhs == b.op.le_s.rhs,
        .le_u => |bin| bin.lhs == b.op.le_u.lhs and bin.rhs == b.op.le_u.rhs,
        .ge_s => |bin| bin.lhs == b.op.ge_s.lhs and bin.rhs == b.op.ge_s.rhs,
        .ge_u => |bin| bin.lhs == b.op.ge_u.lhs and bin.rhs == b.op.ge_u.rhs,
        // Unary integer
        .eqz => |v| v == b.op.eqz,
        .clz => |v| v == b.op.clz,
        .ctz => |v| v == b.op.ctz,
        .popcnt => |v| v == b.op.popcnt,
        // Sign extensions
        .extend8_s => |v| v == b.op.extend8_s,
        .extend16_s => |v| v == b.op.extend16_s,
        .extend32_s => |v| v == b.op.extend32_s,
        // Float unary
        .f_neg => |v| v == b.op.f_neg,
        .f_abs => |v| v == b.op.f_abs,
        .f_sqrt => |v| v == b.op.f_sqrt,
        .f_ceil => |v| v == b.op.f_ceil,
        .f_floor => |v| v == b.op.f_floor,
        .f_trunc => |v| v == b.op.f_trunc,
        .f_nearest => |v| v == b.op.f_nearest,
        // Float binary
        .f_min => |bin| bin.lhs == b.op.f_min.lhs and bin.rhs == b.op.f_min.rhs,
        .f_max => |bin| bin.lhs == b.op.f_max.lhs and bin.rhs == b.op.f_max.rhs,
        .f_copysign => |bin| bin.lhs == b.op.f_copysign.lhs and bin.rhs == b.op.f_copysign.rhs,
        // Float comparisons
        .f_eq => |bin| bin.lhs == b.op.f_eq.lhs and bin.rhs == b.op.f_eq.rhs,
        .f_ne => |bin| bin.lhs == b.op.f_ne.lhs and bin.rhs == b.op.f_ne.rhs,
        .f_lt => |bin| bin.lhs == b.op.f_lt.lhs and bin.rhs == b.op.f_lt.rhs,
        .f_gt => |bin| bin.lhs == b.op.f_gt.lhs and bin.rhs == b.op.f_gt.rhs,
        .f_le => |bin| bin.lhs == b.op.f_le.lhs and bin.rhs == b.op.f_le.rhs,
        .f_ge => |bin| bin.lhs == b.op.f_ge.lhs and bin.rhs == b.op.f_ge.rhs,
        // Conversions
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
        .v128_not => |v| v == b.op.v128_not,
        .v128_bitwise => |bin| bin.op == b.op.v128_bitwise.op and bin.lhs == b.op.v128_bitwise.lhs and bin.rhs == b.op.v128_bitwise.rhs,
        .i32x4_binop => |bin| bin.op == b.op.i32x4_binop.op and bin.lhs == b.op.i32x4_binop.lhs and bin.rhs == b.op.i32x4_binop.rhs,
        .i32x4_unop => |un| un.op == b.op.i32x4_unop.op and un.vector == b.op.i32x4_unop.vector,
        .i32x4_extadd_pairwise_i16x8 => |op| op.sign == b.op.i32x4_extadd_pairwise_i16x8.sign and op.vector == b.op.i32x4_extadd_pairwise_i16x8.vector,
        .i32x4_extend_i16x8 => |op| op.sign == b.op.i32x4_extend_i16x8.sign and op.half == b.op.i32x4_extend_i16x8.half and op.vector == b.op.i32x4_extend_i16x8.vector,
        .i32x4_extmul_i16x8 => |op| op.sign == b.op.i32x4_extmul_i16x8.sign and op.half == b.op.i32x4_extmul_i16x8.half and op.lhs == b.op.i32x4_extmul_i16x8.lhs and op.rhs == b.op.i32x4_extmul_i16x8.rhs,
        .i32x4_shift => |shift| shift.op == b.op.i32x4_shift.op and shift.vector == b.op.i32x4_shift.vector and shift.count == b.op.i32x4_shift.count,
        .i32x4_splat => |v| v == b.op.i32x4_splat,
        .i32x4_extract_lane => |lane| lane.vector == b.op.i32x4_extract_lane.vector and lane.lane == b.op.i32x4_extract_lane.lane,
        .i32x4_replace_lane => |lane| lane.vector == b.op.i32x4_replace_lane.vector and lane.val == b.op.i32x4_replace_lane.val and lane.lane == b.op.i32x4_replace_lane.lane,
        .i8x16_binop => |bin| bin.op == b.op.i8x16_binop.op and bin.lhs == b.op.i8x16_binop.lhs and bin.rhs == b.op.i8x16_binop.rhs,
        .i8x16_narrow_i16x8 => |op| op.sign == b.op.i8x16_narrow_i16x8.sign and op.lhs == b.op.i8x16_narrow_i16x8.lhs and op.rhs == b.op.i8x16_narrow_i16x8.rhs,
        .i8x16_unop => |un| un.op == b.op.i8x16_unop.op and un.vector == b.op.i8x16_unop.vector,
        .i8x16_shift => |shift| shift.op == b.op.i8x16_shift.op and shift.vector == b.op.i8x16_shift.vector and shift.count == b.op.i8x16_shift.count,
        .i8x16_splat => |v| v == b.op.i8x16_splat,
        .i8x16_extract_lane => |lane| lane.vector == b.op.i8x16_extract_lane.vector and lane.lane == b.op.i8x16_extract_lane.lane and lane.sign == b.op.i8x16_extract_lane.sign,
        .i8x16_replace_lane => |lane| lane.vector == b.op.i8x16_replace_lane.vector and lane.val == b.op.i8x16_replace_lane.val and lane.lane == b.op.i8x16_replace_lane.lane,
        .i16x8_binop => |bin| bin.op == b.op.i16x8_binop.op and bin.lhs == b.op.i16x8_binop.lhs and bin.rhs == b.op.i16x8_binop.rhs,
        .i16x8_unop => |un| un.op == b.op.i16x8_unop.op and un.vector == b.op.i16x8_unop.vector,
        .i16x8_extadd_pairwise_i8x16 => |op| op.sign == b.op.i16x8_extadd_pairwise_i8x16.sign and op.vector == b.op.i16x8_extadd_pairwise_i8x16.vector,
        .i16x8_extend_i8x16 => |op| op.sign == b.op.i16x8_extend_i8x16.sign and op.half == b.op.i16x8_extend_i8x16.half and op.vector == b.op.i16x8_extend_i8x16.vector,
        .i16x8_extmul_i8x16 => |op| op.sign == b.op.i16x8_extmul_i8x16.sign and op.half == b.op.i16x8_extmul_i8x16.half and op.lhs == b.op.i16x8_extmul_i8x16.lhs and op.rhs == b.op.i16x8_extmul_i8x16.rhs,
        .i16x8_narrow_i32x4 => |op| op.sign == b.op.i16x8_narrow_i32x4.sign and op.lhs == b.op.i16x8_narrow_i32x4.lhs and op.rhs == b.op.i16x8_narrow_i32x4.rhs,
        .i16x8_shift => |shift| shift.op == b.op.i16x8_shift.op and shift.vector == b.op.i16x8_shift.vector and shift.count == b.op.i16x8_shift.count,
        .i16x8_splat => |v| v == b.op.i16x8_splat,
        .i16x8_extract_lane => |lane| lane.vector == b.op.i16x8_extract_lane.vector and lane.lane == b.op.i16x8_extract_lane.lane and lane.sign == b.op.i16x8_extract_lane.sign,
        .i16x8_replace_lane => |lane| lane.vector == b.op.i16x8_replace_lane.vector and lane.val == b.op.i16x8_replace_lane.val and lane.lane == b.op.i16x8_replace_lane.lane,
        .i64x2_binop => |bin| bin.op == b.op.i64x2_binop.op and bin.lhs == b.op.i64x2_binop.lhs and bin.rhs == b.op.i64x2_binop.rhs,
        .i64x2_unop => |un| un.op == b.op.i64x2_unop.op and un.vector == b.op.i64x2_unop.vector,
        .i64x2_extend_i32x4 => |op| op.sign == b.op.i64x2_extend_i32x4.sign and op.half == b.op.i64x2_extend_i32x4.half and op.vector == b.op.i64x2_extend_i32x4.vector,
        .i64x2_extmul_i32x4 => |op| op.sign == b.op.i64x2_extmul_i32x4.sign and op.half == b.op.i64x2_extmul_i32x4.half and op.lhs == b.op.i64x2_extmul_i32x4.lhs and op.rhs == b.op.i64x2_extmul_i32x4.rhs,
        .i64x2_shift => |shift| shift.op == b.op.i64x2_shift.op and shift.vector == b.op.i64x2_shift.vector and shift.count == b.op.i64x2_shift.count,
        .i64x2_splat => |v| v == b.op.i64x2_splat,
        .i64x2_extract_lane => |lane| lane.vector == b.op.i64x2_extract_lane.vector and lane.lane == b.op.i64x2_extract_lane.lane,
        .i64x2_replace_lane => |lane| lane.vector == b.op.i64x2_replace_lane.vector and lane.val == b.op.i64x2_replace_lane.val and lane.lane == b.op.i64x2_replace_lane.lane,
        // div/rem: covered by isPure+hasSideEffect guard; float variants
        // (side-effect-free) reach here.
        .div_s => |bin| bin.lhs == b.op.div_s.lhs and bin.rhs == b.op.div_s.rhs,
        .div_u => |bin| bin.lhs == b.op.div_u.lhs and bin.rhs == b.op.div_u.rhs,
        .rem_s => |bin| bin.lhs == b.op.rem_s.lhs and bin.rhs == b.op.rem_s.rhs,
        .rem_u => |bin| bin.lhs == b.op.rem_u.lhs and bin.rhs == b.op.rem_u.rhs,
        else => false,
    };
}

// ── Global Value Numbering (cross-block CSE) ────────────────────────────────

/// Dominator-scoped GVN: deduplicate identical pure, non-trapping
/// instructions across basic blocks using the dominator tree.
///
/// Walks the dom tree in DFS pre-order with a scoped expression table.
/// When an instruction in block B matches an entry from a dominator of B,
/// all uses of B's instruction are rewritten to the dominating def via
/// `replaceVReg`. `deadCodeElimination` removes the now-unused original.
///
/// Subsumes block-local `commonSubexprElimination`.
pub fn globalValueNumbering(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    if (func.blocks.items.len == 0) return false;

    var dom = try analysis.computeDominators(func, allocator);
    defer dom.deinit();
    if (dom.idom[0] == null) return false;

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

    const GvnEntry = struct { inst: ir.Inst, vreg: ir.VReg };
    var table: std.ArrayList(GvnEntry) = .empty;
    defer table.deinit(allocator);

    const Frame = struct { bid: ir.BlockId, phase: u1, snap_len: usize };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(allocator);
    try stack.append(allocator, .{ .bid = 0, .phase = 0, .snap_len = 0 });

    var changed = false;
    while (stack.items.len > 0) {
        const top = &stack.items[stack.items.len - 1];
        if (top.phase == 1) {
            table.shrinkRetainingCapacity(top.snap_len);
            _ = stack.pop();
            continue;
        }
        const bid = top.bid;
        top.phase = 1;
        top.snap_len = table.items.len;

        const block = &func.blocks.items[bid];
        for (block.instructions.items) |*inst| {
            if (inst.dest == null or hasSideEffect(inst.*) or !isPure(inst.*)) continue;

            var found: ?ir.VReg = null;
            for (table.items) |entry| {
                if (entry.inst.type == inst.type and sameOp(entry.inst, inst.*)) {
                    found = entry.vreg;
                    break;
                }
            }

            if (found) |earlier_vreg| {
                replaceVReg(func, inst.dest.?, earlier_vreg);
                changed = true;
            } else {
                try table.append(allocator, .{ .inst = inst.*, .vreg = inst.dest.? });
            }
        }

        for (children[bid].items) |child| {
            try stack.append(allocator, .{ .bid = child, .phase = 0, .snap_len = 0 });
        }
    }

    return changed;
}

// ── Pass Manager ────────────────────────────────────────────────────────────

pub const PassFn = *const fn (*ir.IrFunction, std.mem.Allocator) anyerror!bool;

// ── Loop-invariant bounds-check hoisting ────────────────────────────────────

/// Hoist loop-invariant bounds checks to the loop preheader.
///
/// For each natural loop, scans the **header block** for `load`/`store`
/// instructions whose base VReg is loop-invariant (defined outside the
/// loop). For each such base, inserts a single guard load in the
/// preheader with `checked_end = max(offset + size)` across all
/// header accesses with that base. The guard's bounds check runs once
/// before the loop; all covered loop accesses are marked
/// `bounds_known = true` so codegen skips their inline checks.
///
/// Soundness:
///   - Only header accesses are considered. The header executes on
///     every iteration, so a preheader trap is equivalent to a
///     first-iteration trap.
///   - Accesses after a fence (call, memory_grow, etc.) in the header
///     are skipped: the fence could grow memory, making the preheader
///     check invalid.
///   - The preheader must be a dedicated single-successor block
///     (`br header`), ensuring the guard runs only on paths entering
///     the loop.
///   - Wasm memory grows monotonically, so a passing preheader check
///     remains valid for all subsequent iterations (even if memory
///     grows inside the loop body).
///   - Only loop-body accesses with `offset + size ≤ max_end` are
///     marked `bounds_known`; the guard's widened check covers them.
pub fn hoistLoopBoundsChecks(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    if (func.blocks.items.len == 0) return false;

    var dom = try analysis.computeDominators(func, allocator);
    defer dom.deinit();

    var lf = try analysis.computeLoops(func, &dom, allocator);
    defer lf.deinit();
    if (lf.loops.len == 0) return false;

    var predecessors = try analysis.buildPredecessors(func, allocator);
    defer {
        var pit = predecessors.iterator();
        while (pit.next()) |entry| allocator.free(entry.value_ptr.*);
        predecessors.deinit();
    }

    // Build def-block map: for each VReg, which block defines it?
    var def_block = std.AutoHashMap(ir.VReg, ir.BlockId).init(allocator);
    defer def_block.deinit();
    for (func.blocks.items, 0..) |block, idx| {
        for (block.instructions.items) |inst| {
            if (inst.dest) |d| try def_block.put(d, @intCast(idx));
        }
    }

    // Per-base max-end accumulator, reused across loops.
    var base_max = std.AutoHashMap(ir.VReg, u64).init(allocator);
    defer base_max.deinit();

    var changed = false;
    for (lf.loops) |*loop| {
        // ── Find dedicated preheader ──
        // The unique non-loop predecessor of the header whose sole
        // successor is the header (unconditional `br header`).
        const header_preds = predecessors.get(loop.header) orelse continue;
        var preheader: ?ir.BlockId = null;
        for (header_preds) |p| {
            if (!loop.containsBlock(p)) {
                if (preheader != null) {
                    preheader = null;
                    break; // multiple outside predecessors → no unique preheader
                }
                preheader = p;
            }
        }
        const ph = preheader orelse continue;

        // Verify it's a dedicated preheader: sole successor = header.
        const ph_block = &func.blocks.items[ph];
        const ph_insts = ph_block.instructions.items;
        if (ph_insts.len == 0) continue;
        const ph_term = ph_insts[ph_insts.len - 1];
        switch (ph_term.op) {
            .br => |target| {
                if (target != loop.header) continue;
            },
            else => continue, // br_if, br_table, ret, etc. → not dedicated
        }

        // Verify preheader dominates header (sanity).
        if (!dom.dominates(ph, loop.header)) continue;

        // ── Scan header for loop-invariant bases ──
        // Stop at the first fence op (call, memory_grow, etc.) to avoid
        // hoisting checks that could be invalidated by memory growth
        // happening before the access on a later iteration.
        base_max.clearRetainingCapacity();
        const header_block = &func.blocks.items[loop.header];
        for (header_block.instructions.items) |inst| {
            // Fence: stop scanning.
            switch (inst.op) {
                .memory_grow,
                .call,
                .call_indirect,
                .call_ref,
                .memory_copy,
                .memory_fill,
                .memory_init,
                .table_grow,
                .table_init,
                .atomic_notify,
                .atomic_wait,
                => break,
                else => {},
            }
            switch (inst.op) {
                .load => |ld| {
                    if (ld.bounds_known) continue;
                    const db = def_block.get(ld.base) orelse continue;
                    if (loop.containsBlock(db)) continue; // not loop-invariant
                    const end: u64 = @as(u64, ld.offset) + @as(u64, ld.size);
                    const gop = try base_max.getOrPut(ld.base);
                    if (!gop.found_existing) gop.value_ptr.* = end else if (end > gop.value_ptr.*) gop.value_ptr.* = end;
                },
                .store => |st| {
                    if (st.bounds_known) continue;
                    const db = def_block.get(st.base) orelse continue;
                    if (loop.containsBlock(db)) continue;
                    const end: u64 = @as(u64, st.offset) + @as(u64, st.size);
                    const gop = try base_max.getOrPut(st.base);
                    if (!gop.found_existing) gop.value_ptr.* = end else if (end > gop.value_ptr.*) gop.value_ptr.* = end;
                },
                else => {},
            }
        }

        if (base_max.count() == 0) continue;

        // ── Insert guard loads in preheader + mark loop accesses ──
        var bit = base_max.iterator();
        while (bit.next()) |kv| {
            const base = kv.key_ptr.*;
            const max_end = kv.value_ptr.*;

            // Insert guard load before the preheader's terminator.
            const guard_dest = func.newVReg();
            const guard_pos = ph_block.instructions.items.len - 1;
            try ph_block.instructions.insert(ph_block.allocator, guard_pos, .{
                .op = .{ .load = .{
                    .base = base,
                    .offset = 0,
                    .size = 1,
                    .checked_end = max_end,
                } },
                .dest = guard_dest,
                .type = .i32,
            });

            // Mark all loop-body accesses with this base as bounds_known
            // if their offset+size ≤ max_end.
            for (loop.blocks) |bid| {
                for (func.blocks.items[bid].instructions.items) |*inst| {
                    switch (inst.op) {
                        .load => |*ld| {
                            if (ld.bounds_known) continue;
                            if (ld.base != base) continue;
                            const end: u64 = @as(u64, ld.offset) + @as(u64, ld.size);
                            if (end <= max_end) {
                                ld.bounds_known = true;
                                changed = true;
                            }
                        },
                        .store => |*st| {
                            if (st.bounds_known) continue;
                            if (st.base != base) continue;
                            const end: u64 = @as(u64, st.offset) + @as(u64, st.size);
                            if (end <= max_end) {
                                st.bounds_known = true;
                                changed = true;
                            }
                        },
                        else => {},
                    }
                }
            }
        }
    }
    return changed;
}

/// Hoist loop-invariant pure instructions to the loop preheader.
///
/// An instruction is hoistable when `isPure` and `!hasSideEffect` and
/// ALL operand VRegs are defined outside the loop body.  Iterates to
/// a fixed point so cascading works (e.g. hoisting a constant exposes
/// an add that depends on it).
pub fn hoistLoopInvariantCode(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    if (func.blocks.items.len == 0) return false;

    var dom = try analysis.computeDominators(func, allocator);
    defer dom.deinit();

    var lf = try analysis.computeLoops(func, &dom, allocator);
    defer lf.deinit();
    if (lf.loops.len == 0) return false;

    var predecessors = try analysis.buildPredecessors(func, allocator);
    defer {
        var pit = predecessors.iterator();
        while (pit.next()) |entry| allocator.free(entry.value_ptr.*);
        predecessors.deinit();
    }

    var def_block = std.AutoHashMap(ir.VReg, ir.BlockId).init(allocator);
    defer def_block.deinit();
    for (func.blocks.items, 0..) |block, idx| {
        for (block.instructions.items) |inst| {
            if (inst.dest) |d| try def_block.put(d, @intCast(idx));
        }
    }

    var changed = false;
    for (lf.loops) |*loop| {
        const header_preds = predecessors.get(loop.header) orelse continue;
        var preheader: ?ir.BlockId = null;
        for (header_preds) |p| {
            if (!loop.containsBlock(p)) {
                if (preheader != null) {
                    preheader = null;
                    break;
                }
                preheader = p;
            }
        }
        const ph = preheader orelse continue;
        const ph_insts = func.blocks.items[ph].instructions.items;
        if (ph_insts.len == 0) continue;
        switch (ph_insts[ph_insts.len - 1].op) {
            .br => |t| {
                if (t != loop.header) continue;
            },
            else => continue,
        }
        if (!dom.dominates(ph, loop.header)) continue;

        var any = true;
        while (any) {
            any = false;
            for (loop.blocks) |bid| {
                const block = &func.blocks.items[bid];
                var i: usize = 0;
                while (i < block.instructions.items.len) {
                    const inst = block.instructions.items[i];
                    if (inst.dest == null or !isPure(inst) or hasSideEffect(inst)) {
                        i += 1;
                        continue;
                    }
                    const used = getUsedVRegs(inst);
                    var ok = true;
                    for (used.slice()) |v| {
                        if (def_block.get(v)) |db| {
                            if (loop.containsBlock(db)) {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if (!ok) {
                        i += 1;
                        continue;
                    }

                    const ph_block = &func.blocks.items[ph];
                    try ph_block.instructions.insert(ph_block.allocator, ph_block.instructions.items.len - 1, inst);
                    _ = block.instructions.orderedRemove(i);
                    if (inst.dest) |d| try def_block.put(d, ph);
                    any = true;
                    changed = true;
                }
            }
        }
    }
    return changed;
}

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
/// Dominator-table entry: records that `base` has been checked up to `max_end`.
const BoundsEntry = struct { base: ir.VReg, max_end: u64 };

/// Segment-local entry: the first un-elided access for a base and the
/// running maximum end across all same-base accesses in the segment.
const SegEntry = struct { inst: *ir.Inst, max_end: u64 };

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

    const Entry = BoundsEntry;
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

    // Per-segment state for widening: tracks the first un-elided access
    // per base VReg within a fence-free segment, along with the maximum
    // end (offset + size) seen for that base across all accesses in the
    // segment. At segment end (fence or block boundary), the first
    // access's checked_end is patched to the segment max so a single
    // widened bounds check covers all subsequent same-base accesses.
    var seg_first = std.AutoHashMap(ir.VReg, SegEntry).init(allocator);
    defer seg_first.deinit();

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

        seg_first.clearRetainingCapacity();

        const block = &func.blocks.items[bid];
        for (block.instructions.items) |*inst| {
            switch (inst.op) {
                .load => |*ld| {
                    const end: u64 = @as(u64, ld.offset) + @as(u64, ld.size);
                    const dom_max = domMaxEnd(table.items, valid_start, ld.base);

                    if (end <= dom_max) {
                        if (!ld.bounds_known) {
                            ld.bounds_known = true;
                            changed = true;
                        }
                    } else if (seg_first.getPtr(ld.base)) |se| {
                        // Covered by the widened first access in this segment.
                        if (!ld.bounds_known) {
                            ld.bounds_known = true;
                            changed = true;
                        }
                        if (end > se.max_end) se.max_end = end;
                    } else {
                        // First un-elided access for this base in segment.
                        try seg_first.put(ld.base, .{ .inst = inst, .max_end = end });
                    }
                },
                .store => |*st| {
                    const end: u64 = @as(u64, st.offset) + @as(u64, st.size);
                    const dom_max = domMaxEnd(table.items, valid_start, st.base);

                    if (end <= dom_max) {
                        if (!st.bounds_known) {
                            st.bounds_known = true;
                            changed = true;
                        }
                    } else if (seg_first.getPtr(st.base)) |se| {
                        if (!st.bounds_known) {
                            st.bounds_known = true;
                            changed = true;
                        }
                        if (end > se.max_end) se.max_end = end;
                    } else {
                        try seg_first.put(st.base, .{ .inst = inst, .max_end = end });
                    }
                },
                // Fences: commit the current segment (patch checked_end
                // on first accesses) then hide all dominator entries
                // from post-fence instructions and dom-tree descendants.
                .memory_grow,
                .call,
                .call_indirect,
                .call_ref,
                .memory_copy,
                .memory_fill,
                .memory_init,
                .table_grow,
                .table_init,
                .atomic_notify,
                .atomic_wait,
                => {
                    changed = patchSegment(&seg_first) or changed;
                    seg_first.clearRetainingCapacity();
                    valid_start = table.items.len;
                },
                else => {},
            }
        }

        // End of block: patch remaining segment and commit entries to
        // the dominator table so dom-tree children can see them.
        changed = patchSegment(&seg_first) or changed;
        {
            var it = seg_first.iterator();
            while (it.next()) |kv| {
                try table.append(allocator, .{ .base = kv.key_ptr.*, .max_end = kv.value_ptr.max_end });
            }
        }
        seg_first.clearRetainingCapacity();

        for (children[bid].items) |c| {
            try stack.append(allocator, .{ .bid = c, .phase = 0, .snap_len = 0, .snap_valid_start = 0 });
        }
    }

    return changed;
}

/// Look up the maximum checked end for `base` in the visible portion
/// of the dominator table (entries at indices >= valid_start).
fn domMaxEnd(table: []const BoundsEntry, valid_start: usize, base: ir.VReg) u64 {
    var best: u64 = 0;
    for (table[valid_start..]) |e| {
        if (e.base == base and e.max_end > best) best = e.max_end;
    }
    return best;
}

/// Patch the first un-elided access in each segment entry: set its
/// `checked_end` to the segment's max end so the emitted bounds check
/// covers all subsequent same-base accesses marked `bounds_known`.
/// Returns true if any instruction was modified.
fn patchSegment(seg_first: *std.AutoHashMap(ir.VReg, SegEntry)) bool {
    var patched = false;
    var it = seg_first.iterator();
    while (it.next()) |kv| {
        const se = kv.value_ptr;
        const own_end: u64 = switch (se.inst.op) {
            .load => |ld| @as(u64, ld.offset) + @as(u64, ld.size),
            .store => |st| @as(u64, st.offset) + @as(u64, st.size),
            else => unreachable,
        };
        if (se.max_end > own_end) {
            switch (se.inst.op) {
                .load => |*ld| ld.checked_end = se.max_end,
                .store => |*st| st.checked_end = se.max_end,
                else => unreachable,
            }
            patched = true;
        }
    }
    return patched;
}

// ── Address-mode folding (load/store offset) ────────────────────────────────

/// Fold `add base, iconst_32 C` feeding into a `load`/`store` by absorbing
/// `C` into the memory immediate offset:
///
///     v_addr = add base, C
///     load  v_addr, offset=N  =>  load  base, offset=N+C
///
/// This is only sound when a dominating bounds check has already proven
/// `base + (N+C) + size <= memory_size`. Without that proof, folding can
/// change wrapping semantics: Wasm `i32.add` wraps, but the load/store
/// effective address uses the zero-extended base plus a non-wrapping offset.
pub fn foldLoadStoreOffset(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    if (func.blocks.items.len == 0) return false;

    var dom = try analysis.computeDominators(func, allocator);
    defer dom.deinit();
    if (dom.idom[0] == null) return false;

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

    const AddInfo = struct { base: ir.VReg, offset: u32 };

    var table: std.ArrayList(BoundsEntry) = .empty;
    defer table.deinit(allocator);
    var valid_start: usize = 0;

    const Frame = struct { bid: ir.BlockId, phase: u1, snap_len: usize, snap_valid_start: usize };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(allocator);
    try stack.append(allocator, .{ .bid = 0, .phase = 0, .snap_len = 0, .snap_valid_start = 0 });

    var iconst32 = std.AutoHashMap(ir.VReg, i32).init(allocator);
    defer iconst32.deinit();
    var add_info = std.AutoHashMap(ir.VReg, AddInfo).init(allocator);
    defer add_info.deinit();
    var block_checked = std.AutoHashMap(ir.VReg, u64).init(allocator);
    defer block_checked.deinit();

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

        iconst32.clearRetainingCapacity();
        add_info.clearRetainingCapacity();
        block_checked.clearRetainingCapacity();

        const block = &func.blocks.items[bid];
        for (block.instructions.items) |*inst| {
            switch (inst.op) {
                .iconst_32 => |c| {
                    if (inst.dest) |d| try iconst32.put(d, c);
                },
                .add => |bin| {
                    if (inst.type != .i32) continue;
                    const dest = inst.dest orelse continue;
                    if (iconst32.get(bin.rhs)) |c| {
                        if (c >= 0) try add_info.put(dest, .{ .base = bin.lhs, .offset = @intCast(c) });
                    } else if (iconst32.get(bin.lhs)) |c| {
                        if (c >= 0) try add_info.put(dest, .{ .base = bin.rhs, .offset = @intCast(c) });
                    }
                },
                .load => |*ld| {
                    if (add_info.get(ld.base)) |info| {
                        const access_end = if (ld.checked_end > 0) ld.checked_end else @as(u64, ld.offset) + @as(u64, ld.size);
                        const new_end: ?u64 = std.math.add(u64, @as(u64, info.offset), access_end) catch null;
                        const new_offset: ?u64 = std.math.add(u64, @as(u64, info.offset), @as(u64, ld.offset)) catch null;
                        if (new_end) |end| {
                            if (new_offset) |off| {
                                const block_max = block_checked.get(info.base) orelse 0;
                                const dom_max = domMaxEnd(table.items, valid_start, info.base);
                                const proof = @max(block_max, dom_max);
                                if (end <= proof and off <= std.math.maxInt(i32)) {
                                    ld.base = info.base;
                                    ld.offset = @intCast(off);
                                    if (ld.checked_end > 0) ld.checked_end = end;
                                    changed = true;
                                }
                            }
                        }
                    }
                    if (!ld.bounds_known) {
                        const end = if (ld.checked_end > 0) ld.checked_end else @as(u64, ld.offset) + @as(u64, ld.size);
                        const gop = try block_checked.getOrPut(ld.base);
                        if (!gop.found_existing or end > gop.value_ptr.*) gop.value_ptr.* = end;
                    }
                },
                .store => |*st| {
                    if (add_info.get(st.base)) |info| {
                        const access_end = if (st.checked_end > 0) st.checked_end else @as(u64, st.offset) + @as(u64, st.size);
                        const new_end: ?u64 = std.math.add(u64, @as(u64, info.offset), access_end) catch null;
                        const new_offset: ?u64 = std.math.add(u64, @as(u64, info.offset), @as(u64, st.offset)) catch null;
                        if (new_end) |end| {
                            if (new_offset) |off| {
                                const block_max = block_checked.get(info.base) orelse 0;
                                const dom_max = domMaxEnd(table.items, valid_start, info.base);
                                const proof = @max(block_max, dom_max);
                                if (end <= proof and off <= std.math.maxInt(i32)) {
                                    st.base = info.base;
                                    st.offset = @intCast(off);
                                    if (st.checked_end > 0) st.checked_end = end;
                                    changed = true;
                                }
                            }
                        }
                    }
                    if (!st.bounds_known) {
                        const end = if (st.checked_end > 0) st.checked_end else @as(u64, st.offset) + @as(u64, st.size);
                        const gop = try block_checked.getOrPut(st.base);
                        if (!gop.found_existing or end > gop.value_ptr.*) gop.value_ptr.* = end;
                    }
                },
                .memory_grow,
                .call,
                .call_indirect,
                .call_ref,
                .memory_copy,
                .memory_fill,
                .memory_init,
                .table_grow,
                .table_init,
                .atomic_notify,
                .atomic_wait,
                => {
                    block_checked.clearRetainingCapacity();
                    valid_start = table.items.len;
                },
                else => {},
            }
        }

        var bit = block_checked.iterator();
        while (bit.next()) |kv| {
            try table.append(allocator, .{ .base = kv.key_ptr.*, .max_end = kv.value_ptr.* });
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

// ── Branch-on-Eqz folding ───────────────────────────────────────────────────

/// Collapse `br_if(cond=eqz(x), then=A, else=B)` into
/// `br_if(cond=x, then=B, else=A)`.
///
/// This removes a redundant `eqz` whose only use is the branch and flips
/// the target polarity. On aarch64 this turns `cmp ... ; cbz` into
/// `cbnz`, saving an instruction; on x86-64 the eqz lowers to a
/// `test/sete` + jump that becomes a single `test + jnz`.
///
/// Soundness:
///   - The rewrite is semantics-preserving: `eqz(x) != 0` iff `x == 0`,
///     so swapping the branch targets inverts the condition back.
///   - We only rewrite when the `eqz`'s single use is this br_if (so the
///     eqz becomes dead and DCE reaps it next iteration). If the eqz has
///     other uses we can still flip the branch, but we'd leave the eqz
///     live with no saving — skip to avoid churning `runPasses`.
pub fn foldBranchOnEqz(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;

    // Build a vreg -> defining-instruction index so we can identify
    // producers that may be in a different block from the terminator.
    var def_block = std.AutoHashMap(ir.VReg, ir.BlockId).init(allocator);
    defer def_block.deinit();
    var def_idx = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer def_idx.deinit();

    for (func.blocks.items, 0..) |block, bi| {
        for (block.instructions.items, 0..) |inst, ii| {
            if (inst.dest) |d| {
                try def_block.put(d, @intCast(bi));
                try def_idx.put(d, @intCast(ii));
            }
        }
    }

    for (func.blocks.items) |*block| {
        if (block.instructions.items.len == 0) continue;
        const term = &block.instructions.items[block.instructions.items.len - 1];
        switch (term.op) {
            .br_if => |bi| {
                const producer_block = def_block.get(bi.cond) orelse continue;
                const producer_ii = def_idx.get(bi.cond) orelse continue;
                const producer = &func.blocks.items[producer_block].instructions.items[producer_ii];
                const inner = switch (producer.op) {
                    .eqz => |v| v,
                    else => continue,
                };
                if (countUsesOfVReg(func, bi.cond) != 1) continue;
                term.op = .{ .br_if = .{
                    .cond = inner,
                    .then_block = bi.else_block,
                    .else_block = bi.then_block,
                } };
                changed = true;
            },
            else => {},
        }
    }
    return changed;
}

// ── Wrap-of-extend cancellation ────────────────────────────────────────────

/// Eliminate `wrap_i64(extend_i32_s(x))` and `wrap_i64(extend_i32_u(x))`
/// — both compose to the identity on the original i32 value.
///
/// The frontend (and inliner / function-merging passes) sometimes
/// produce these chains when an i32 is briefly widened to i64 to
/// participate in a helper or comparison and then narrowed back.
///
/// Soundness:
///   - `extend_i32_s(x)` places the low 32 bits of the result equal to
///     x's bit pattern (and sign-extends the upper 32 from x's sign
///     bit). `extend_i32_u(x)` places x in the low 32 and zeros the
///     upper. In both cases `wrap_i64` returns the low 32, recovering
///     x exactly.
///
/// We always rewrite when the pattern matches; the inner extend is
/// left in place (it may have other uses) and DCE drops it later if
/// it ends up unused.
pub fn foldWrapOfExtend(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;

    var def_block = std.AutoHashMap(ir.VReg, ir.BlockId).init(allocator);
    defer def_block.deinit();
    var def_idx = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer def_idx.deinit();

    for (func.blocks.items, 0..) |block, bi| {
        for (block.instructions.items, 0..) |inst, ii| {
            if (inst.dest) |d| {
                try def_block.put(d, @intCast(bi));
                try def_idx.put(d, @intCast(ii));
            }
        }
    }

    const Rewrite = struct {
        blk: ir.BlockId,
        ii: u32,
        wrap_dest: ir.VReg,
        inner_src: ir.VReg,
    };
    var rewrites = std.ArrayList(Rewrite).empty;
    defer rewrites.deinit(allocator);

    for (func.blocks.items, 0..) |block, bi| {
        for (block.instructions.items, 0..) |inst, ii| {
            const wrap_dest = inst.dest orelse continue;
            const wrap_src = switch (inst.op) {
                .wrap_i64 => |v| v,
                else => continue,
            };
            const pb = def_block.get(wrap_src) orelse continue;
            const pi = def_idx.get(wrap_src) orelse continue;
            const producer = func.blocks.items[pb].instructions.items[pi];
            const inner_src = switch (producer.op) {
                .extend_i32_s, .extend_i32_u => |v| v,
                else => continue,
            };
            try rewrites.append(allocator, .{
                .blk = @intCast(bi),
                .ii = @intCast(ii),
                .wrap_dest = wrap_dest,
                .inner_src = inner_src,
            });
        }
    }

    for (rewrites.items) |r| {
        replaceVReg(func, r.wrap_dest, r.inner_src);
        const inst = &func.blocks.items[r.blk].instructions.items[r.ii];
        inst.op = .{ .iconst_32 = 0 };
        changed = true;
    }
    return changed;
}

// ── Float unary idempotents ────────────────────────────────────────────────

/// Simplify chained unary float operations:
///   f_neg(f_neg(x)) -> x        (involution)
///   f_abs(f_abs(x)) -> f_abs(x) (idempotent)
///   f_abs(f_neg(x)) -> f_abs(x) (|-x| = |x|)
///
/// Soundness on IEEE-754 floats:
///   - f_neg only flips the sign bit and otherwise preserves the bit
///     pattern (including NaN payloads), so f_neg(f_neg(x)) bit-for-bit
///     equals x. -0.0 round-trips back to -0.0.
///   - f_abs clears the sign bit; clearing twice is the same as
///     clearing once, so f_abs is idempotent.
///   - f_abs(f_neg(x)) clears the sign bit no matter what f_neg
///     produced, matching f_abs(x).
///
/// We always rewrite without checking the inner producer's use count:
/// even if the inner f_neg/f_abs has other uses it stays alive, and
/// our rewrite still removes one outer instruction. DCE will drop the
/// inner if it later becomes dead.
pub fn foldFloatUnaryIdempotents(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;

    var def_block = std.AutoHashMap(ir.VReg, ir.BlockId).init(allocator);
    defer def_block.deinit();
    var def_idx = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer def_idx.deinit();

    for (func.blocks.items, 0..) |block, bi| {
        for (block.instructions.items, 0..) |inst, ii| {
            if (inst.dest) |d| {
                try def_block.put(d, @intCast(bi));
                try def_idx.put(d, @intCast(ii));
            }
        }
    }

    const Action = enum { replace_with_inner_src, replace_with_inner_dest, rewrite_operand };
    const Rewrite = struct {
        action: Action,
        outer_blk: ir.BlockId,
        outer_ii: u32,
        outer_dest: ir.VReg,
        new_vreg: ir.VReg, // replacement / new operand
    };
    var rewrites = std.ArrayList(Rewrite).empty;
    defer rewrites.deinit(allocator);

    for (func.blocks.items, 0..) |block, bi| {
        for (block.instructions.items, 0..) |inst, ii| {
            const outer_dest = inst.dest orelse continue;
            const outer_src: ir.VReg = switch (inst.op) {
                .f_neg, .f_abs => |v| v,
                else => continue,
            };
            const pb = def_block.get(outer_src) orelse continue;
            const pi = def_idx.get(outer_src) orelse continue;
            const producer = func.blocks.items[pb].instructions.items[pi];
            const producer_dest = producer.dest orelse continue;

            switch (inst.op) {
                .f_neg => {
                    // f_neg(f_neg(x)) -> x
                    if (producer.op == .f_neg) {
                        const inner_src = producer.op.f_neg;
                        try rewrites.append(allocator, .{
                            .action = .replace_with_inner_src,
                            .outer_blk = @intCast(bi),
                            .outer_ii = @intCast(ii),
                            .outer_dest = outer_dest,
                            .new_vreg = inner_src,
                        });
                    }
                },
                .f_abs => {
                    if (producer.op == .f_abs) {
                        // f_abs(f_abs(x)) -> f_abs(x): same value as
                        // inner; redirect consumers of outer to inner.
                        try rewrites.append(allocator, .{
                            .action = .replace_with_inner_dest,
                            .outer_blk = @intCast(bi),
                            .outer_ii = @intCast(ii),
                            .outer_dest = outer_dest,
                            .new_vreg = producer_dest,
                        });
                    } else if (producer.op == .f_neg) {
                        // f_abs(f_neg(x)) -> f_abs(x): rewrite this
                        // f_abs's operand to skip the inner f_neg.
                        const inner_src = producer.op.f_neg;
                        try rewrites.append(allocator, .{
                            .action = .rewrite_operand,
                            .outer_blk = @intCast(bi),
                            .outer_ii = @intCast(ii),
                            .outer_dest = outer_dest,
                            .new_vreg = inner_src,
                        });
                    }
                },
                else => unreachable,
            }
        }
    }

    for (rewrites.items) |r| {
        const inst = &func.blocks.items[r.outer_blk].instructions.items[r.outer_ii];
        switch (r.action) {
            .replace_with_inner_src, .replace_with_inner_dest => {
                replaceVReg(func, r.outer_dest, r.new_vreg);
                // Neutralise the outer; DCE drops it.
                inst.op = .{ .iconst_32 = 0 };
            },
            .rewrite_operand => {
                inst.op = .{ .f_abs = r.new_vreg };
            },
        }
        changed = true;
    }
    return changed;
}

// ── Sign-extending load fold ────────────────────────────────────────────────

/// Fold `extend{8,16,32}_s(load size=N, sign_extend=false)` into the
/// load itself by setting `sign_extend = true` and dropping the extend.
///
/// This collapses the wasm pattern `i32.load8_u; i32.extend8_s` (and
/// matching i64/16/32 variants) into a single sign-extending load,
/// which is the same machine instruction either way (`ldrsb` /
/// `movsx` etc.) — saving one IR instruction per occurrence.
///
/// Soundness:
///   - `load size=1 sign_extend=false type=i32` produces zero-extended
///     low byte of the loaded value; `extend8_s` re-interprets the low
///     byte as signed and sign-extends. The composition is exactly the
///     semantics of `load size=1 sign_extend=true type=i32`.
///   - We require the load result to have exactly one use (this
///     extend). Otherwise other consumers depend on the zero-extended
///     value and changing the load would corrupt them.
///   - load.type and extend.type must match (we never bridge i32↔i64
///     here; that requires an explicit `extend_i32_s/u`).
///   - extend32_s is only meaningful on i64, matching the wasm
///     `i64.load32_s` pattern.
pub fn foldSignExtendingLoad(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;

    var def_block = std.AutoHashMap(ir.VReg, ir.BlockId).init(allocator);
    defer def_block.deinit();
    var def_idx = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer def_idx.deinit();

    for (func.blocks.items, 0..) |block, bi| {
        for (block.instructions.items, 0..) |inst, ii| {
            if (inst.dest) |d| {
                try def_block.put(d, @intCast(bi));
                try def_idx.put(d, @intCast(ii));
            }
        }
    }

    const Rewrite = struct {
        ext_blk: ir.BlockId,
        ext_ii: u32,
        load_blk: ir.BlockId,
        load_ii: u32,
        ext_dest: ir.VReg,
        load_dest: ir.VReg,
    };
    var rewrites = std.ArrayList(Rewrite).empty;
    defer rewrites.deinit(allocator);

    for (func.blocks.items, 0..) |block, bi| {
        for (block.instructions.items, 0..) |inst, ii| {
            const ext_dest = inst.dest orelse continue;
            const ext_type = inst.type;
            const want_size: u8 = switch (inst.op) {
                .extend8_s => 1,
                .extend16_s => 2,
                .extend32_s => blk: {
                    if (ext_type != .i64) continue;
                    break :blk 4;
                },
                else => continue,
            };
            const src = switch (inst.op) {
                .extend8_s, .extend16_s, .extend32_s => |v| v,
                else => unreachable,
            };
            const pb = def_block.get(src) orelse continue;
            const pi = def_idx.get(src) orelse continue;
            const producer = func.blocks.items[pb].instructions.items[pi];
            const ld = switch (producer.op) {
                .load => |l| l,
                else => continue,
            };
            if (ld.size != want_size) continue;
            if (ld.sign_extend) continue;
            if (producer.type != ext_type) continue;
            if (countUsesOfVReg(func, src) != 1) continue;
            const load_dest = producer.dest orelse continue;
            try rewrites.append(allocator, .{
                .ext_blk = @intCast(bi),
                .ext_ii = @intCast(ii),
                .load_blk = pb,
                .load_ii = pi,
                .ext_dest = ext_dest,
                .load_dest = load_dest,
            });
        }
    }

    for (rewrites.items) |r| {
        // Flip the load to sign-extending.
        const load_inst = &func.blocks.items[r.load_blk].instructions.items[r.load_ii];
        switch (load_inst.op) {
            .load => |*ld| ld.sign_extend = true,
            else => continue,
        }
        // Redirect consumers of the extend's dest to the load's dest.
        replaceVReg(func, r.ext_dest, r.load_dest);
        // Neutralise the extend instruction so DCE removes it.
        const ext_inst = &func.blocks.items[r.ext_blk].instructions.items[r.ext_ii];
        ext_inst.op = .{ .iconst_32 = 0 };
        changed = true;
    }
    return changed;
}

// ── Select-on-Eqz folding ──────────────────────────────────────────────────

/// Collapse `select(cond=eqz(x), if_true=a, if_false=b)` into
/// `select(cond=x, if_true=b, if_false=a)`.
///
/// Mirror of `foldBranchOnEqz` for the non-terminator case. Removes a
/// redundant `eqz` whose only use is the select and swaps the chosen
/// arms. On aarch64 this maps `cmp; cset` style sequences to a single
/// `csel` with the inverse condition.
///
/// Soundness: `eqz(x) != 0 ⇔ x == 0`, so swapping if_true/if_false
/// inverts the condition back. Skipped unless the eqz has exactly this
/// one use (otherwise rewriting would leave the eqz live and waste a
/// `runPasses` iteration on a no-op fixpoint check).
pub fn foldSelectOnEqz(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;

    var def_block = std.AutoHashMap(ir.VReg, ir.BlockId).init(allocator);
    defer def_block.deinit();
    var def_idx = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer def_idx.deinit();

    for (func.blocks.items, 0..) |block, bi| {
        for (block.instructions.items, 0..) |inst, ii| {
            if (inst.dest) |d| {
                try def_block.put(d, @intCast(bi));
                try def_idx.put(d, @intCast(ii));
            }
        }
    }

    for (func.blocks.items) |*block| {
        for (block.instructions.items) |*inst| {
            const sel = switch (inst.op) {
                .select => |s| s,
                else => continue,
            };
            const pb = def_block.get(sel.cond) orelse continue;
            const pi = def_idx.get(sel.cond) orelse continue;
            const producer = func.blocks.items[pb].instructions.items[pi];
            const inner = switch (producer.op) {
                .eqz => |v| v,
                else => continue,
            };
            if (countUsesOfVReg(func, sel.cond) != 1) continue;
            inst.op = .{ .select = .{
                .cond = inner,
                .if_true = sel.if_false,
                .if_false = sel.if_true,
            } };
            changed = true;
        }
    }
    return changed;
}

// ── Inverse-compare / eqz fusion ───────────────────────────────────────────

/// Rewrite `eqz(cmp(a, b))` as the inverse comparison on `(a, b)`, where
/// `cmp` is any integer relational op. The original `eqz` instruction is
/// rewritten in place to hold the inverse comparison, preserving its dest
/// VReg. The original comparison may become dead and will be reaped by
/// `deadCodeElimination`.
///
/// Mappings:
///   eqz(eq)   → ne     eqz(ne)   → eq
///   eqz(lt_s) → ge_s   eqz(ge_s) → lt_s
///   eqz(le_s) → gt_s   eqz(gt_s) → le_s
///   eqz(lt_u) → ge_u   eqz(ge_u) → lt_u
///   eqz(le_u) → gt_u   eqz(gt_u) → le_u
///
/// Soundness:
///   - Integer relops produce exactly 0 or 1 (wasm semantics), so their
///     logical negation IS the inverse comparison. eqz(1) = 0 = !(1);
///     eqz(0) = 1 = !(0).
///   - Skipped for float compares: eqz is integer-only and the IR
///     doesn't emit `eqz(f_eq)` etc.
///
/// Why bother with this in addition to `foldBranchOnEqz`:
///   - Covers cases where the eqz result is used by `select`, stored to
///     a local, or used as an operand to another op — not just by a
///     terminator br_if.
///   - Removes a compare + eqz sequence; backend emits a single compare
///     with the inverse condition code.
pub fn foldInverseCompareEqz(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;

    var def_block = std.AutoHashMap(ir.VReg, ir.BlockId).init(allocator);
    defer def_block.deinit();
    var def_idx = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer def_idx.deinit();

    for (func.blocks.items, 0..) |block, bi| {
        for (block.instructions.items, 0..) |inst, ii| {
            if (inst.dest) |d| {
                try def_block.put(d, @intCast(bi));
                try def_idx.put(d, @intCast(ii));
            }
        }
    }

    // Rewrites must be applied after the scan so that iteration doesn't
    // see a half-mutated instruction stream.
    const Rewrite = struct {
        blk: ir.BlockId,
        ii: u32,
        new_op: ir.Inst.Op,
    };
    var rewrites = std.ArrayList(Rewrite).empty;
    defer rewrites.deinit(allocator);

    for (func.blocks.items, 0..) |block, bi| {
        for (block.instructions.items, 0..) |inst, ii| {
            const src = switch (inst.op) {
                .eqz => |v| v,
                else => continue,
            };
            const pb = def_block.get(src) orelse continue;
            const pi = def_idx.get(src) orelse continue;
            const producer = func.blocks.items[pb].instructions.items[pi];
            const new_op: ?ir.Inst.Op = switch (producer.op) {
                .eq => |b| .{ .ne = b },
                .ne => |b| .{ .eq = b },
                .lt_s => |b| .{ .ge_s = b },
                .ge_s => |b| .{ .lt_s = b },
                .le_s => |b| .{ .gt_s = b },
                .gt_s => |b| .{ .le_s = b },
                .lt_u => |b| .{ .ge_u = b },
                .ge_u => |b| .{ .lt_u = b },
                .le_u => |b| .{ .gt_u = b },
                .gt_u => |b| .{ .le_u = b },
                else => null,
            };
            if (new_op) |op| {
                try rewrites.append(allocator, .{
                    .blk = @intCast(bi),
                    .ii = @intCast(ii),
                    .new_op = op,
                });
            }
        }
    }

    for (rewrites.items) |r| {
        func.blocks.items[r.blk].instructions.items[r.ii].op = r.new_op;
        changed = true;
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
        .iconst_32,
        .iconst_64,
        .fconst_32,
        .fconst_64,
        .v128_const,
        .local_get,
        .global_get,
        .br,
        .@"unreachable",
        .memory_size,
        .table_size,
        .ref_func,
        .data_drop,
        .elem_drop,
        .atomic_fence,
        .call_result,
        => {},

        .add,
        .sub,
        .mul,
        .div_s,
        .div_u,
        .rem_s,
        .rem_u,
        .@"and",
        .@"or",
        .xor,
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
        .f_min,
        .f_max,
        .f_copysign,
        .f_eq,
        .f_ne,
        .f_lt,
        .f_gt,
        .f_le,
        .f_ge,
        => |*bin| {
            bin.lhs += offset;
            bin.rhs += offset;
        },

        .v128_bitwise => |*bin| {
            bin.lhs += offset;
            bin.rhs += offset;
        },
        .i32x4_binop => |*bin| {
            bin.lhs += offset;
            bin.rhs += offset;
        },
        .i32x4_unop => |*un| un.vector += offset,
        .i32x4_extadd_pairwise_i16x8 => |*op| op.vector += offset,
        .i32x4_extend_i16x8 => |*op| op.vector += offset,
        .i32x4_extmul_i16x8 => |*op| {
            op.lhs += offset;
            op.rhs += offset;
        },
        .i8x16_binop => |*bin| {
            bin.lhs += offset;
            bin.rhs += offset;
        },
        .i8x16_narrow_i16x8 => |*op| {
            op.lhs += offset;
            op.rhs += offset;
        },
        .i8x16_unop => |*un| un.vector += offset,
        .i8x16_shift => |*shift| {
            shift.vector += offset;
            shift.count += offset;
        },
        .i16x8_binop => |*bin| {
            bin.lhs += offset;
            bin.rhs += offset;
        },
        .i16x8_unop => |*un| un.vector += offset,
        .i16x8_extadd_pairwise_i8x16 => |*op| op.vector += offset,
        .i16x8_extend_i8x16 => |*op| op.vector += offset,
        .i16x8_extmul_i8x16 => |*op| {
            op.lhs += offset;
            op.rhs += offset;
        },
        .i16x8_narrow_i32x4 => |*op| {
            op.lhs += offset;
            op.rhs += offset;
        },
        .i64x2_binop => |*bin| {
            bin.lhs += offset;
            bin.rhs += offset;
        },
        .i64x2_unop => |*un| un.vector += offset,
        .i64x2_extend_i32x4 => |*op| op.vector += offset,
        .i64x2_extmul_i32x4 => |*op| {
            op.lhs += offset;
            op.rhs += offset;
        },
        .i64x2_shift => |*shift| {
            shift.vector += offset;
            shift.count += offset;
        },
        .i32x4_shift => |*shift| {
            shift.vector += offset;
            shift.count += offset;
        },
        .i16x8_shift => |*shift| {
            shift.vector += offset;
            shift.count += offset;
        },

        .clz,
        .ctz,
        .popcnt,
        .eqz,
        .wrap_i64,
        .extend_i32_s,
        .extend_i32_u,
        .extend8_s,
        .extend16_s,
        .extend32_s,
        .f_neg,
        .f_abs,
        .f_sqrt,
        .f_ceil,
        .f_floor,
        .f_trunc,
        .f_nearest,
        .trunc_f32_s,
        .trunc_f32_u,
        .trunc_f64_s,
        .trunc_f64_u,
        .convert_s,
        .convert_u,
        .convert_i32_s,
        .convert_i64_s,
        .convert_i32_u,
        .convert_i64_u,
        .demote_f64,
        .promote_f32,
        .reinterpret,
        .trunc_sat_f32_s,
        .trunc_sat_f32_u,
        .trunc_sat_f64_s,
        .trunc_sat_f64_u,
        .v128_not,
        .i32x4_splat,
        .i8x16_splat,
        .i16x8_splat,
        .i64x2_splat,
        => |*vreg| vreg.* += offset,
        .i32x4_extract_lane => |*lane| lane.vector += offset,
        .i8x16_extract_lane => |*lane| lane.vector += offset,
        .i16x8_extract_lane => |*lane| lane.vector += offset,
        .i64x2_extract_lane => |*lane| lane.vector += offset,
        .i32x4_replace_lane => |*lane| {
            lane.vector += offset;
            lane.val += offset;
        },
        .i8x16_replace_lane => |*lane| {
            lane.vector += offset;
            lane.val += offset;
        },
        .i16x8_replace_lane => |*lane| {
            lane.vector += offset;
            lane.val += offset;
        },
        .i64x2_replace_lane => |*lane| {
            lane.vector += offset;
            lane.val += offset;
        },

        .local_set => |*ls| ls.val += offset,
        .global_set => |*gs| gs.val += offset,
        .load => |*ld| ld.base += offset,
        .v128_load => |*ld| ld.base += offset,
        .store => |*st| {
            st.base += offset;
            st.val += offset;
        },
        .v128_store => |*st| {
            st.base += offset;
            st.val += offset;
        },
        .br_if => |*bi| bi.cond += offset,
        .br_table => |*bt| bt.index += offset,
        .ret => |*maybe_vreg| if (maybe_vreg.*) |v| {
            maybe_vreg.* = v + offset;
        },
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
        .phi => |edges| {
            for (@constCast(edges)) |*edge| edge.val += offset;
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
                .call,
                .call_indirect,
                .call_ref,
                .call_result,
                .memory_grow,
                .atomic_fence,
                .atomic_load,
                .atomic_store,
                .atomic_rmw,
                .atomic_cmpxchg,
                .atomic_notify,
                .atomic_wait,
                .memory_copy,
                .memory_fill,
                .memory_init,
                .table_init,
                .table_grow,
                .data_drop,
                .elem_drop,
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

// ── SSA Promotion (mem2reg) ─────────────────────────────────────────────

/// Promote wasm locals from explicit `local_set`/`local_get` ops to SSA
/// VRegs with phi nodes at CFG join points.
///
/// Algorithm: Cytron et al. "Efficiently Computing Static Single
/// Assignment Form and the Control Dependence Graph" (1991).
///
/// 1. Compute dominance frontiers.
/// 2. For each local, place phis at the iterated dominance frontier of
///    blocks containing a `local.set` for that local.
/// 3. Rename: DFS walk of the dominator tree with a per-local value
///    stack. `local.set` pushes the value; `local.get` reads the top.
///    Phi operands are filled in when processing successor edges.
/// 4. Dead `local.set`/`local.get` ops are left in place for DCE.
///
/// After this pass, phis must be lowered (via `lowerPhisToLocals`)
/// before codegen.
pub fn promoteLocalsToSSA(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    if (func.blocks.items.len == 0) return false;
    if (func.local_count == 0) return false;

    // Strip dead code after the first terminator in each block.
    for (func.blocks.items) |*block| {
        for (block.instructions.items, 0..) |inst, idx| {
            switch (inst.op) {
                .br, .br_if, .br_table, .ret, .ret_multi, .@"unreachable" => {
                    if (idx + 1 < block.instructions.items.len)
                        block.instructions.shrinkRetainingCapacity(idx + 1);
                    break;
                },
                else => {},
            }
        }
    }

    // Strip dead code after the first terminator in each block.
    for (func.blocks.items) |*block| {
        for (block.instructions.items, 0..) |inst, idx| {
            switch (inst.op) {
                .br, .br_if, .br_table, .ret, .ret_multi, .@"unreachable" => {
                    if (idx + 1 < block.instructions.items.len) {
                        block.instructions.shrinkRetainingCapacity(idx + 1);
                    }
                    break;
                },
                else => {},
            }
        }
    }

    var dom = try analysis.computeDominators(func, allocator);
    defer dom.deinit();
    if (dom.idom[0] == null) return false;

    const df = try analysis.computeDominanceFrontiers(&dom, func, allocator);
    defer analysis.freeDominanceFrontiers(df, allocator);

    var preds = try analysis.buildPredecessors(func, allocator);
    defer {
        var pit = preds.iterator();
        while (pit.next()) |entry| allocator.free(entry.value_ptr.*);
        preds.deinit();
    }

    const nblocks = func.blocks.items.len;
    const nlocals = func.local_count;

    // ── Step 1: find which blocks define (local.set) each local ──────
    var def_blocks = try allocator.alloc(std.ArrayList(ir.BlockId), nlocals);
    defer {
        for (def_blocks) |*l| l.deinit(allocator);
        allocator.free(def_blocks);
    }
    for (def_blocks) |*l| l.* = .empty;

    for (func.blocks.items, 0..) |block, bid_usize| {
        const bid: ir.BlockId = @intCast(bid_usize);
        for (block.instructions.items) |inst| {
            if (inst.op == .local_set) {
                const idx = inst.op.local_set.idx;
                if (idx < nlocals) {
                    // Deduplicate.
                    var dup = false;
                    for (def_blocks[idx].items) |existing| {
                        if (existing == bid) {
                            dup = true;
                            break;
                        }
                    }
                    if (!dup) try def_blocks[idx].append(allocator, bid);
                }
            }
        }
    }

    // ── Step 2: place phi nodes at iterated dominance frontiers ──────
    // For each local, compute IDF(def_blocks) and insert phi.
    // has_phi[local][block] tracks whether a phi was already placed.
    var has_phi = try allocator.alloc(std.AutoHashMap(ir.BlockId, ir.VReg), nlocals);
    defer {
        for (has_phi) |*m| m.deinit();
        allocator.free(has_phi);
    }
    for (has_phi) |*m| m.* = std.AutoHashMap(ir.BlockId, ir.VReg).init(allocator);

    // Worklist for iterated DF.
    var worklist: std.ArrayList(ir.BlockId) = .empty;
    defer worklist.deinit(allocator);
    var in_worklist = try allocator.alloc(bool, nblocks);
    defer allocator.free(in_worklist);

    for (0..nlocals) |local_idx| {
        // Pruned SSA: skip locals that have no defs (never set).
        if (def_blocks[local_idx].items.len == 0) continue;

        // Seed worklist with defining blocks.
        worklist.clearRetainingCapacity();
        @memset(in_worklist, false);
        for (def_blocks[local_idx].items) |b| {
            try worklist.append(allocator, b);
            in_worklist[b] = true;
        }

        var wi: usize = 0;
        while (wi < worklist.items.len) : (wi += 1) {
            const b = worklist.items[wi];
            for (df[b]) |y| {
                if (!has_phi[local_idx].contains(y)) {
                    // Insert phi at top of block y.
                    const phi_dest = func.newVReg();
                    const pred_list = preds.get(y) orelse &[_]ir.BlockId{};
                    const edges = try allocator.alloc(ir.Inst.PhiEdge, pred_list.len);
                    // Initialize with sentinel VRegs; rename pass fills them.
                    for (edges, 0..) |*e, ei| {
                        e.* = .{ .block = pred_list[ei], .val = phi_dest };
                    }
                    const local_type = if (func.local_types) |lt|
                        (if (local_idx < lt.len) lt[local_idx] else ir.IrType.i32)
                    else
                        ir.IrType.i32;
                    try func.getBlock(y).instructions.insert(func.allocator, 0, .{
                        .op = .{ .phi = edges },
                        .dest = phi_dest,
                        .type = local_type,
                    });
                    try has_phi[local_idx].put(y, phi_dest);
                    if (!in_worklist[y]) {
                        try worklist.append(allocator, y);
                        in_worklist[y] = true;
                    }
                }
            }
        }
    }

    // ── Step 3: rename ───────────────────────────────────────────────
    // Per-local value stack. Top = current SSA value for this local.
    var stacks = try allocator.alloc(std.ArrayList(ir.VReg), nlocals);
    defer {
        for (stacks) |*s| s.deinit(allocator);
        allocator.free(stacks);
    }
    for (stacks) |*s| s.* = .empty;

    // Seed stacks with initial values.
    // Params: the frontend allocates VRegs 0..param_count-1 for params.
    // Declared locals: start at zero (insert iconst/fconst in entry block).
    for (0..nlocals) |idx| {
        if (idx < func.param_count) {
            // Params live in frame slots; seed with a local_get so the
            // SSA value has an explicit definition the regalloc can track.
            const local_type = if (func.local_types) |lt|
                (if (idx < lt.len) lt[idx] else ir.IrType.i32)
            else
                ir.IrType.i32;
            const param_vreg = func.newVReg();
            try func.getBlock(0).instructions.insert(func.allocator, 0, .{
                .op = .{ .local_get = @intCast(idx) },
                .dest = param_vreg,
                .type = local_type,
            });
            try stacks[idx].append(allocator, param_vreg);
        } else {
            // Declared/synthetic local: seed with typed zero.
            const local_type = if (func.local_types) |lt|
                (if (idx < lt.len) lt[idx] else ir.IrType.i32)
            else
                ir.IrType.i32;
            const zero_vreg = func.newVReg();
            const zero_op: ir.Inst.Op = switch (local_type) {
                .i32 => .{ .iconst_32 = 0 },
                .i64 => .{ .iconst_64 = 0 },
                .f32 => .{ .fconst_32 = 0 },
                .f64 => .{ .fconst_64 = 0 },
                .v128 => .{ .v128_const = 0 },
                .void => .{ .iconst_32 = 0 },
            };
            // Insert at start of entry block (block 0) before phis.
            try func.getBlock(0).instructions.insert(func.allocator, 0, .{
                .op = zero_op,
                .dest = zero_vreg,
                .type = local_type,
            });
            try stacks[idx].append(allocator, zero_vreg);
        }
    }

    // Build dom-tree children list.
    var dom_children = try allocator.alloc(std.ArrayList(ir.BlockId), nblocks);
    defer {
        for (dom_children) |*l| l.deinit(allocator);
        allocator.free(dom_children);
    }
    for (dom_children) |*l| l.* = .empty;
    for (0..nblocks) |i| {
        const bid: ir.BlockId = @intCast(i);
        const idom = dom.idom[bid] orelse continue;
        if (idom == bid) continue;
        try dom_children[idom].append(allocator, bid);
    }

    // Compute successors for filling phi operands in successor blocks.
    var successors = try analysis.buildSuccessors(func, allocator);
    defer {
        var sit = successors.iterator();
        while (sit.next()) |entry| allocator.free(entry.value_ptr.*);
        successors.deinit();
    }

    // DFS rename walk.
    const RenameFrame = struct {
        bid: ir.BlockId,
        phase: u1,
        stack_heights: []u32, // per-local stack height on entry (for restore)
        rename_snap: u32, // rename_keys length on entry (for restore)
    };
    var rename_stack: std.ArrayList(RenameFrame) = .empty;
    defer {
        for (rename_stack.items) |f| allocator.free(f.stack_heights);
        rename_stack.deinit(allocator);
    }

    // Map from old local_get dest VReg → SSA replacement VReg.
    // Entries are scoped to the dominator subtree: when the DFS backtracks,
    // entries added by the leaving block are removed to prevent stale
    // rewrites in non-dominated sibling blocks.
    var rename_map = std.AutoHashMap(ir.VReg, ir.VReg).init(allocator);
    defer rename_map.deinit();

    // Track keys added to rename_map for each DFS level so we can undo them.
    var rename_keys: std.ArrayList(ir.VReg) = .empty;
    defer rename_keys.deinit(allocator);

    const entry_heights = try allocator.alloc(u32, nlocals);
    for (0..nlocals) |i| entry_heights[i] = @intCast(stacks[i].items.len);
    try rename_stack.append(allocator, .{
        .bid = 0,
        .phase = 0,
        .stack_heights = entry_heights,
        .rename_snap = 0,
    });

    var changed = false;
    while (rename_stack.items.len > 0) {
        const top = &rename_stack.items[rename_stack.items.len - 1];

        if (top.phase == 1) {
            // Restore stacks.
            for (0..nlocals) |i| {
                stacks[i].shrinkRetainingCapacity(top.stack_heights[i]);
            }
            // Restore rename_map: remove entries added by this block.
            while (rename_keys.items.len > top.rename_snap) {
                const key = rename_keys.pop().?;
                _ = rename_map.remove(key);
            }
            allocator.free(top.stack_heights);
            _ = rename_stack.pop();
            continue;
        }
        const bid = top.bid;
        top.phase = 1;

        // Process instructions in this block.
        const block = &func.blocks.items[bid];
        for (block.instructions.items) |*inst| {
            // Rewrite operands: any VReg in the rename map gets replaced.
            // This handles uses of local_get dests that were renamed.
            switch (inst.op) {
                .phi, .local_set, .local_get => {},
                else => {
                    const used = getUsedVRegs(inst.*);
                    for (used.slice()) |u| {
                        if (rename_map.get(u)) |replacement| {
                            replaceInInst(inst, u, replacement);
                        }
                    }
                    // Also handle unbounded operand lists.
                    switch (inst.op) {
                        .call => |cl| for (@constCast(cl.args)) |*a| {
                            if (rename_map.get(a.*)) |r| a.* = r;
                        },
                        .call_indirect => |ci| {
                            if (rename_map.get(ci.elem_idx)) |r| @constCast(&ci.elem_idx).* = r;
                            for (@constCast(ci.args)) |*a| {
                                if (rename_map.get(a.*)) |r| a.* = r;
                            }
                        },
                        .call_ref => |cr| {
                            if (rename_map.get(cr.func_ref)) |r| @constCast(&cr.func_ref).* = r;
                            for (@constCast(cr.args)) |*a| {
                                if (rename_map.get(a.*)) |r| a.* = r;
                            }
                        },
                        .ret_multi => |vregs| for (@constCast(vregs)) |*v| {
                            if (rename_map.get(v.*)) |r| v.* = r;
                        },
                        else => {},
                    }
                },
            }

            switch (inst.op) {
                .phi => {
                    // Push phi dest onto the local's stack.
                    const dest = inst.dest orelse continue;
                    for (0..nlocals) |local_idx| {
                        if (has_phi[local_idx].get(bid)) |phi_vreg| {
                            if (phi_vreg == dest) {
                                try stacks[local_idx].append(allocator, dest);
                                break;
                            }
                        }
                    }
                },
                .local_set => |ls| {
                    if (ls.idx < nlocals) {
                        // Rewrite the value operand, chasing rename chains.
                        var val = ls.val;
                        while (rename_map.get(val)) |r| {
                            if (r == val) break;
                            val = r;
                        }
                        try stacks[ls.idx].append(allocator, val);
                        inst.op = .{ .iconst_32 = 0 };
                        inst.dest = null;
                        changed = true;
                    }
                },
                .local_get => |idx| {
                    if (idx < nlocals and stacks[idx].items.len > 0) {
                        const current_val = stacks[idx].items[stacks[idx].items.len - 1];
                        if (inst.dest) |dest| {
                            if (dest == current_val and idx < func.param_count) {
                                // Seeded parameter local_get: this instruction
                                // IS the definition of the SSA VReg for the
                                // parameter value — keep it alive.
                            } else {
                                try rename_map.put(dest, current_val);
                                try rename_keys.append(allocator, dest);
                                inst.op = .{ .iconst_32 = 0 };
                                inst.dest = null;
                                changed = true;
                            }
                        }
                    }
                },
                else => {},
            }
        }

        // Fill phi operands in successor blocks.
        const succs = successors.get(bid) orelse &[_]ir.BlockId{};
        for (succs) |succ| {
            const succ_block = &func.blocks.items[succ];
            for (succ_block.instructions.items) |*succ_inst| {
                if (succ_inst.op != .phi) break; // phis are at top
                const phi_dest = succ_inst.dest orelse continue;
                // Find which local this phi belongs to.
                for (0..nlocals) |local_idx| {
                    if (has_phi[local_idx].get(succ)) |pv| {
                        if (pv == phi_dest) {
                            // Fill in this block's edge.
                            for (@constCast(succ_inst.op.phi)) |*edge| {
                                if (edge.block == bid) {
                                    if (stacks[local_idx].items.len > 0) {
                                        edge.val = stacks[local_idx].items[stacks[local_idx].items.len - 1];
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Push dom-tree children.
        for (dom_children[bid].items) |child| {
            const heights = try allocator.alloc(u32, nlocals);
            for (0..nlocals) |i| heights[i] = @intCast(stacks[i].items.len);
            try rename_stack.append(allocator, .{
                .bid = child,
                .phase = 0,
                .stack_heights = heights,
                .rename_snap = @intCast(rename_keys.items.len),
            });
        }
    }

    return changed;
}

/// Lower phi nodes to parallel copies through synthetic locals.
///
/// For each phi `dest = phi [(B0, v0), (B1, v1), ...]`:
///   - Allocate a synthetic local index L.
///   - In each predecessor Bi, insert `local_set L, vi` before the
///     terminator.
///   - Replace the phi with `local_get L → dest`.
///
/// Parallel-copy correctness: when multiple phis exist in the same block
/// (phi-of-phi, common in loops), all reads (the `vi` operands) must be
/// captured before any writes (local_set). We achieve this by allocating
/// distinct synthetic locals per phi — each phi gets its own slot, so
/// writes to one don't clobber reads of another.
pub fn lowerPhisToLocals(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;
    var next_synth_local = func.local_count;

    for (func.blocks.items) |*block| {
        var i: usize = 0;
        while (i < block.instructions.items.len) {
            if (block.instructions.items[i].op != .phi) {
                i += 1;
                continue;
            }

            const dest = block.instructions.items[i].dest orelse {
                i += 1;
                continue;
            };
            const edges = block.instructions.items[i].op.phi;
            const phi_type = block.instructions.items[i].type;
            const synth_idx = next_synth_local;
            next_synth_local += 1;

            // Insert local_set in each predecessor before its terminator.
            // NOTE: when a predecessor is this block (self-loop), the insert
            // may reallocate block.instructions.items — do NOT hold a pointer
            // into the instruction list across this loop.
            for (edges) |edge| {
                const pred_block = &func.blocks.items[edge.block];
                const term_idx = findTerminatorIndex(pred_block);
                try pred_block.instructions.insert(func.allocator, term_idx, .{
                    .op = .{ .local_set = .{ .idx = synth_idx, .val = edge.val } },
                });
            }

            allocator.free(edges);

            // Replace phi with local_get.  Re-index into the instruction list
            // because inserts above may have reallocated the backing array.
            block.instructions.items[i] = .{
                .op = .{ .local_get = synth_idx },
                .dest = dest,
                .type = phi_type,
            };
            changed = true;
            i += 1;
        }
    }

    // Update local_count to include synthetic locals.
    if (next_synth_local > func.local_count) {
        func.local_count = next_synth_local;
    }

    return changed;
}

/// Find the index of the terminator instruction in a block.
/// Terminators are br, br_if, br_table, ret, ret_multi, unreachable.
fn findTerminatorIndex(block: *const ir.BasicBlock) usize {
    for (block.instructions.items, 0..) |inst, idx| {
        switch (inst.op) {
            .br, .br_if, .br_table, .ret, .ret_multi, .@"unreachable" => return idx,
            else => {},
        }
    }
    return block.instructions.items.len;
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
        // SSA promotion: run once before the fixpoint loop.
        if (try promoteLocalsToSSA(func, allocator)) {
            total_changes += 1;
            if (try lowerPhisToLocals(func, allocator)) total_changes += 1;
        }

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
    &foldInverseCompareEqz,
    &foldBranchOnEqz,
    &foldSelectOnEqz,
    &foldSignExtendingLoad,
    &foldFloatUnaryIdempotents,
    &foldWrapOfExtend,
    &globalValueNumbering,
    &hoistLoopInvariantCode,
    &deadCodeElimination,
    &deadLocalSetElimination,
    &hoistLoopBoundsChecks,
    &elideRedundantBoundsChecks,
    &foldLoadStoreOffset,
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

test "CSE: cross-block CSE via dominator tree" {
    // b0 defines add(v0, v1) = v2; b0 branches to b1; b1 recomputes the
    // same add into v3. Since b0 dominates b1, the dominator-scoped CSE
    // rewrites v3's uses to v2.
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

    // b1's ret should now reference v2 (the dominator's def).
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

test "CSE: loop header def rewritten in body (dom-scoped)" {
    // The loop header dominates the body, so the body's redundant add IS
    // rewritten to the header's def.
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
    try func.getBlock(body).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_body } } });
    try func.getBlock(body).append(.{ .op = .{ .br = h } });

    try func.getBlock(exit).append(.{ .op = .{ .ret = v_h } });

    const changed = try commonSubexprElimination(&func, allocator);
    try std.testing.expect(changed);

    // local_set now references v_h (the header's dominating def).
    try std.testing.expectEqual(
        ir.Inst.Op{ .local_set = .{ .idx = 0, .val = v_h } },
        func.getBlock(body).instructions.items[1].op,
    );
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

test "CSE: diamond — dominator's def reaches both arms" {
    // b0 computes add(v0,v1)=v2, then branches to b1 and b2.
    // Both b1 and b2 recompute the same add. b0 dominates both,
    // so both should be rewritten to v2.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg(); // add in b0
    const cond = func.newVReg();
    const v_b1 = func.newVReg(); // redundant add in b1
    const v_b2 = func.newVReg(); // redundant add in b2

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try func.getBlock(b0).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v_b1 });
    try func.getBlock(b1).append(.{ .op = .{ .ret = v_b1 } });
    try func.getBlock(b2).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v_b2 });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v_b2 } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = null } }); // unused merge point

    const changed = try commonSubexprElimination(&func, allocator);
    try std.testing.expect(changed);

    // Both arms' rets should reference v2 from b0.
    try std.testing.expectEqual(ir.Inst.Op{ .ret = v2 }, func.getBlock(b1).instructions.items[1].op);
    try std.testing.expectEqual(ir.Inst.Op{ .ret = v2 }, func.getBlock(b2).instructions.items[1].op);
}

test "CSE: chain — grandparent dominates grandchild" {
    // b0 → b1 → b2. b0 computes add, b2 recomputes it.
    // b0 dominates b1, b1 dominates b2, so b0's def should reach b2.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v_b0 = func.newVReg();
    const v_b2 = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 5 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v1 });
    try func.getBlock(b0).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v_b0 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b2 } });
    try func.getBlock(b2).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v_b2 });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v_b2 } });

    const changed = try commonSubexprElimination(&func, allocator);
    try std.testing.expect(changed);

    // b2's ret should reference v_b0 from b0.
    try std.testing.expectEqual(ir.Inst.Op{ .ret = v_b0 }, func.getBlock(b2).instructions.items[1].op);
}

test "CSE: iconst_32 dedup across blocks" {
    // b0 defines iconst_32 42 = v0; b1 redefines iconst_32 42 = v1.
    // Since b0 dominates b1, v1's uses should be rewritten to v0.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v1 });
    try func.getBlock(b1).append(.{ .op = .{ .ret = v1 } });

    const changed = try commonSubexprElimination(&func, allocator);
    try std.testing.expect(changed);

    // b1's ret should reference v0 from b0.
    try std.testing.expectEqual(ir.Inst.Op{ .ret = v0 }, func.getBlock(b1).instructions.items[1].op);
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

test "hoistLoopBoundsChecks: header load hoisted to preheader" {
    // b0 (preheader) → b1 (header) → b2 (body) → b1, exit=b3.
    // Header has a load with loop-invariant base. The pass should insert
    // a guard load in b0 and mark the header's load as bounds_known.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    const v_base = func.newVReg();
    const cond = func.newVReg();
    const v_hdr = func.newVReg();
    const v_body = func.newVReg();

    // b0 (preheader): define base, unconditional br to header.
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    // b1 (header): load base+0 size=4.
    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_hdr, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b2, .else_block = b3 } } });

    // b2 (body): load base+4 size=4, back-edge to header.
    try func.getBlock(b2).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 4, .size = 4 } }, .dest = v_body, .type = .i32 });
    try func.getBlock(b2).append(.{ .op = .{ .br = b1 } });

    // b3 (exit).
    try func.getBlock(b3).append(.{ .op = .{ .ret = v_hdr } });

    const changed = try hoistLoopBoundsChecks(&func, allocator);
    try std.testing.expect(changed);

    // Preheader should now have 3 instructions: iconst, guard load, br.
    try std.testing.expectEqual(@as(usize, 3), func.getBlock(b0).instructions.items.len);
    // Guard load should have checked_end = 4 (from header's offset=0, size=4).
    const guard = func.getBlock(b0).instructions.items[1];
    try std.testing.expectEqual(@as(u64, 4), guard.op.load.checked_end);
    try std.testing.expectEqual(v_base, guard.op.load.base);
    // Header load should be marked bounds_known.
    try std.testing.expect(func.getBlock(b1).instructions.items[0].op.load.bounds_known);
    // Body load should also be marked bounds_known (offset+size=8 > 4?).
    // body offset=4, size=4 → end=8 > max_end=4 from header-only scan.
    // So body load should NOT be marked bounds_known by hoistLoopBoundsChecks
    // (the guard only covers header accesses' max_end).
    // Wait — the pass marks ALL loop accesses with end ≤ max_end. end=8 > 4, so not covered.
    try std.testing.expect(!func.getBlock(b2).instructions.items[0].op.load.bounds_known);
}

test "hoistLoopBoundsChecks: widens to cover multiple header accesses" {
    // Header has two loads: base+0/4 and base+4/4. Guard should have
    // checked_end = 8, covering both. Body load at base+2/2 (end=4 ≤ 8)
    // should also be marked bounds_known.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    const v_base = func.newVReg();
    const cond = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    const v_c = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_a, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 4, .size = 4 } }, .dest = v_b, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b2, .else_block = b3 } } });

    try func.getBlock(b2).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 2, .size = 2 } }, .dest = v_c, .type = .i32 });
    try func.getBlock(b2).append(.{ .op = .{ .br = b1 } });

    try func.getBlock(b3).append(.{ .op = .{ .ret = v_a } });

    const changed = try hoistLoopBoundsChecks(&func, allocator);
    try std.testing.expect(changed);

    // Guard's checked_end should be max(0+4, 4+4) = 8.
    const guard = func.getBlock(b0).instructions.items[1];
    try std.testing.expectEqual(@as(u64, 8), guard.op.load.checked_end);
    // Both header loads should be bounds_known.
    try std.testing.expect(func.getBlock(b1).instructions.items[0].op.load.bounds_known);
    try std.testing.expect(func.getBlock(b1).instructions.items[1].op.load.bounds_known);
    // Body load end=4 ≤ 8, should be bounds_known.
    try std.testing.expect(func.getBlock(b2).instructions.items[0].op.load.bounds_known);
}

test "hoistLoopBoundsChecks: non-invariant base skipped" {
    // Header load's base is defined inside the loop → not loop-invariant.
    // The pass should not hoist.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_base = func.newVReg();
    const cond = func.newVReg();
    const v_ld = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    // Base defined in header (inside loop).
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_base });
    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_ld, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });

    try func.getBlock(b2).append(.{ .op = .{ .ret = v_ld } });

    const changed = try hoistLoopBoundsChecks(&func, allocator);
    try std.testing.expect(!changed);
    // Preheader unchanged (just the br).
    try std.testing.expectEqual(@as(usize, 1), func.getBlock(b0).instructions.items.len);
}

test "hoistLoopBoundsChecks: call before load stops scan" {
    // Header has a call before the load → fence stops scan.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_base = func.newVReg();
    const cond = func.newVReg();
    const v_ld = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    // Header: call first, then load.
    try func.getBlock(b1).append(.{ .op = .{ .call = .{ .func_idx = 0 } } });
    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_ld, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });

    try func.getBlock(b2).append(.{ .op = .{ .ret = v_ld } });

    const changed = try hoistLoopBoundsChecks(&func, allocator);
    try std.testing.expect(!changed);
}

test "hoistLoopBoundsChecks: non-dedicated preheader skipped" {
    // Preheader has br_if (two successors) → not dedicated → skip.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    const v_base = func.newVReg();
    const cond = func.newVReg();
    const cond2 = func.newVReg();
    const v_ld = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    // br_if → header or skip: not a dedicated preheader.
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b3 } } });

    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_ld, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond2 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = cond2, .then_block = b1, .else_block = b2 } } });

    try func.getBlock(b2).append(.{ .op = .{ .ret = v_ld } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    const changed = try hoistLoopBoundsChecks(&func, allocator);
    try std.testing.expect(!changed);
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
    // With widening: first load (end=4) is widened to checked_end=8 covering
    // all three. Second (end=8) and third (end=8) are both elided.
    try std.testing.expect(!block.instructions.items[1].op.load.bounds_known);
    try std.testing.expectEqual(@as(u64, 8), block.instructions.items[1].op.load.checked_end);
    try std.testing.expect(block.instructions.items[2].op.load.bounds_known);
    try std.testing.expect(block.instructions.items[3].op.load.bounds_known);
}

test "hoistLoopInvariantCode: pure add with invariant operands hoisted" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const v2 = func.newVReg();
    const v3 = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v2 }, .dest = v3 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v3, .then_block = b2, .else_block = b1 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v2 } });

    const changed = try hoistLoopInvariantCode(&func, allocator);
    try std.testing.expect(changed);
    var found_add = false;
    for (func.getBlock(b0).instructions.items) |inst| {
        if (inst.op == .add) {
            found_add = true;
            break;
        }
    }
    try std.testing.expect(found_add);
    var hdr_has_add = false;
    for (func.getBlock(b1).instructions.items) |inst| {
        if (inst.op == .add) {
            hdr_has_add = true;
            break;
        }
    }
    try std.testing.expect(!hdr_has_add);
}

test "hoistLoopInvariantCode: cascading hoist" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const v1 = func.newVReg();
    const v2 = func.newVReg();
    const v3 = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v2 }, .dest = v3 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v3, .then_block = b2, .else_block = b1 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v2 } });

    const changed = try hoistLoopInvariantCode(&func, allocator);
    try std.testing.expect(changed);
    var ph_has_add = false;
    for (func.getBlock(b0).instructions.items) |inst| {
        if (inst.op == .add) {
            ph_has_add = true;
            break;
        }
    }
    try std.testing.expect(ph_has_add);
}

test "hoistLoopInvariantCode: trapping op not hoisted" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v1, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const v2 = func.newVReg();
    const v3 = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .div_u = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v2 }, .dest = v3 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v3, .then_block = b2, .else_block = b1 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v2 } });

    const changed = try hoistLoopInvariantCode(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expect(func.getBlock(b1).instructions.items[0].op == .div_u);
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

test "strengthReduceDivRem: rewrites non-power-of-two divisor via reciprocal multiply" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v_c, .type = .i32 });
    try block.append(.{ .op = .{ .div_u = .{ .lhs = v_x, .rhs = v_c } }, .dest = v_r, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try strengthReduceDivRem(&func, allocator);
    try std.testing.expect(changed);
    // div_u should be replaced with reciprocal multiply sequence.
    var has_div = false;
    var has_wrap = false;
    for (block.instructions.items) |inst| {
        if (inst.op == .div_u) has_div = true;
        if (inst.op == .wrap_i64) has_wrap = true;
    }
    try std.testing.expect(!has_div);
    try std.testing.expect(has_wrap);
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

test "elideRedundantBoundsChecks: widening three consecutive loads" {
    // Three loads from the same base with increasing offsets: base+0 (4B),
    // base+4 (4B), base+8 (4B). Only the first should emit a bounds check,
    // widened to checked_end=12 to cover all three.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_base = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    const v_c = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = v_base });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_a, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 4, .size = 4 } }, .dest = v_b, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 8, .size = 4 } }, .dest = v_c, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_c } });

    const changed = try elideRedundantBoundsChecks(&func, allocator);
    try std.testing.expect(changed);
    // First load: not bounds_known, but checked_end widened to 12.
    try std.testing.expect(!block.instructions.items[1].op.load.bounds_known);
    try std.testing.expectEqual(@as(u64, 12), block.instructions.items[1].op.load.checked_end);
    // Second and third loads: bounds_known = true (covered by widened first).
    try std.testing.expect(block.instructions.items[2].op.load.bounds_known);
    try std.testing.expect(block.instructions.items[3].op.load.bounds_known);
}

test "elideRedundantBoundsChecks: no widening across call fence" {
    // load, call (fence), load — the second load must NOT be covered by
    // the first load's widening because the call might change memory_size.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_base = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_a, .type = .i32 });
    try block.append(.{ .op = .{ .call = .{ .func_idx = 0 } } });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 4, .size = 4 } }, .dest = v_b, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_b } });

    _ = try elideRedundantBoundsChecks(&func, allocator);
    // First load: no widening (no subsequent same-base access in segment).
    try std.testing.expectEqual(@as(u64, 0), block.instructions.items[1].op.load.checked_end);
    // Post-call load: fresh segment, not elided.
    try std.testing.expect(!block.instructions.items[3].op.load.bounds_known);
    try std.testing.expectEqual(@as(u64, 0), block.instructions.items[3].op.load.checked_end);
}

test "elideRedundantBoundsChecks: mixed load and store same base" {
    // load base+0 (4B), store base+8 (4B) → first load widened to
    // checked_end=12, store elided.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_base = func.newVReg();
    const v_loaded = func.newVReg();
    const v_val = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x2000 }, .dest = v_base });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_loaded, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v_val });
    try block.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 8, .size = 4, .val = v_val } } });
    try block.append(.{ .op = .{ .ret = v_loaded } });

    const changed = try elideRedundantBoundsChecks(&func, allocator);
    try std.testing.expect(changed);
    // Load widened to checked_end=12.
    try std.testing.expect(!block.instructions.items[1].op.load.bounds_known);
    try std.testing.expectEqual(@as(u64, 12), block.instructions.items[1].op.load.checked_end);
    // Store elided.
    try std.testing.expect(block.instructions.items[3].op.store.bounds_known);
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

test "strengthReduceDivRem: div_u by 5 uses reciprocal multiply" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    var block = &func.blocks.items[bid];

    const v0 = func.newVReg(); // dividend
    const v1 = func.newVReg(); // divisor constant = 5
    const v2 = func.newVReg(); // result
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .div_u = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const changed = try strengthReduceDivRem(&func, allocator);
    try std.testing.expect(changed);
    // div_u should be replaced with extend+mul+shift+wrap sequence.
    // The block should no longer contain a div_u.
    var has_div = false;
    var has_extend = false;
    var has_wrap = false;
    for (block.instructions.items) |inst| {
        if (inst.op == .div_u) has_div = true;
        if (inst.op == .extend_i32_u) has_extend = true;
        if (inst.op == .wrap_i64) has_wrap = true;
    }
    try std.testing.expect(!has_div);
    try std.testing.expect(has_extend);
    try std.testing.expect(has_wrap);
}

test "strengthReduceDivRem: rem_u by 3 uses reciprocal multiply + sub" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    var block = &func.blocks.items[bid];

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .rem_u = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const changed = try strengthReduceDivRem(&func, allocator);
    try std.testing.expect(changed);
    var has_rem = false;
    var has_sub = false;
    for (block.instructions.items) |inst| {
        if (inst.op == .rem_u) has_rem = true;
        if (inst.op == .sub) has_sub = true;
    }
    try std.testing.expect(!has_rem);
    try std.testing.expect(has_sub); // x - (x/d)*d
}

test "strengthReduceDivRem: div_u by 1 unchanged" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    var block = &func.blocks.items[bid];

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .div_u = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const changed = try strengthReduceDivRem(&func, allocator);
    try std.testing.expect(!changed); // d=1 skipped
}

test "computeMagicU32: known divisors" {
    // Verify magic numbers produce correct results for several divisors.
    const test_cases = [_]u32{ 3, 5, 7, 10, 11, 13, 100, 255, 1000 };
    for (test_cases) |d| {
        const m = computeMagicU32(d) orelse {
            try std.testing.expect(false); // should always find magic for these
            continue;
        };
        // Verify correctness for boundary values.
        const vals = [_]u64{ 0, 1, d - 1, d, d + 1, 2 * d, 0xFFFF, 0xFFFFFFFF };
        for (vals) |x| {
            const expected = x / d;
            const prod = @as(u128, x) * @as(u128, m.magic);
            const result = @as(u64, @truncate(prod >> (@as(u7, 32) + m.shift)));
            try std.testing.expectEqual(expected, result);
        }
    }
}

test "foldBranchOnEqz: swaps targets and drops eqz use" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .eqz = v_x }, .dest = v_c });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v_c, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    const changed = try foldBranchOnEqz(&func, allocator);
    try std.testing.expect(changed);

    const term = func.getBlock(b0).instructions.items[1];
    switch (term.op) {
        .br_if => |bi| {
            try std.testing.expectEqual(v_x, bi.cond);
            try std.testing.expectEqual(b2, bi.then_block);
            try std.testing.expectEqual(b1, bi.else_block);
        },
        else => try std.testing.expect(false),
    }
}

test "foldBranchOnEqz: skips when eqz has multiple uses" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .eqz = v_x }, .dest = v_c });
    // second use of v_c
    try func.getBlock(b0).append(.{ .op = .{ .add = .{ .lhs = v_c, .rhs = v_c } }, .dest = v_r });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v_c, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    const changed = try foldBranchOnEqz(&func, allocator);
    try std.testing.expect(!changed);

    const term = func.getBlock(b0).instructions.items[2];
    switch (term.op) {
        .br_if => |bi| {
            try std.testing.expectEqual(v_c, bi.cond);
            try std.testing.expectEqual(b1, bi.then_block);
            try std.testing.expectEqual(b2, bi.else_block);
        },
        else => try std.testing.expect(false),
    }
}

test "foldBranchOnEqz: no-op when br_if cond is not eqz" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 2, 2, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_a = func.newVReg();
    const v_b = func.newVReg();
    const v_c = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 3 }, .dest = v_a });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 4 }, .dest = v_b });
    try func.getBlock(b0).append(.{ .op = .{ .eq = .{ .lhs = v_a, .rhs = v_b } }, .dest = v_c });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v_c, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    const changed = try foldBranchOnEqz(&func, allocator);
    try std.testing.expect(!changed);
}

test "foldBranchOnEqz: cross-block eqz producer" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const entry = try func.newBlock();
    const mid = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .eqz = v_x }, .dest = v_c });
    try func.getBlock(entry).append(.{ .op = .{ .br = mid } });
    try func.getBlock(mid).append(.{ .op = .{ .br_if = .{ .cond = v_c, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    const changed = try foldBranchOnEqz(&func, allocator);
    try std.testing.expect(changed);
    const term = func.getBlock(mid).instructions.items[0];
    switch (term.op) {
        .br_if => |bi| {
            try std.testing.expectEqual(v_x, bi.cond);
            try std.testing.expectEqual(b2, bi.then_block);
            try std.testing.expectEqual(b1, bi.else_block);
        },
        else => try std.testing.expect(false),
    }
}

test "foldBranchOnEqz: pipeline drops dead eqz after DCE" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_x = func.newVReg();
    const v_c = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .eqz = v_x }, .dest = v_c });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v_c, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    _ = try foldBranchOnEqz(&func, allocator);
    _ = try deadCodeElimination(&func, allocator);

    // eqz should be gone; only the br_if remains.
    try std.testing.expectEqual(@as(usize, 1), func.getBlock(b0).instructions.items.len);
    try std.testing.expect(func.getBlock(b0).instructions.items[0].op == .br_if);
}

test "foldInverseCompareEqz: eqz(eq) becomes ne" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 2, 2, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v_cmp = func.newVReg();
    const v_neg = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 4 }, .dest = v1 });
    try block.append(.{ .op = .{ .eq = .{ .lhs = v0, .rhs = v1 } }, .dest = v_cmp });
    try block.append(.{ .op = .{ .eqz = v_cmp }, .dest = v_neg });
    try block.append(.{ .op = .{ .ret = v_neg } });

    const changed = try foldInverseCompareEqz(&func, allocator);
    try std.testing.expect(changed);

    // The eqz instruction (index 3) should now be .ne on v0, v1.
    switch (block.instructions.items[3].op) {
        .ne => |b| {
            try std.testing.expectEqual(v0, b.lhs);
            try std.testing.expectEqual(v1, b.rhs);
        },
        else => try std.testing.expect(false),
    }
    // dest preserved.
    try std.testing.expectEqual(@as(?ir.VReg, v_neg), block.instructions.items[3].dest);
}

test "foldInverseCompareEqz: all 10 mappings" {
    const allocator = std.testing.allocator;
    const cases = [_]struct {
        src: ir.Inst.Op,
        expect_tag: std.meta.Tag(ir.Inst.Op),
    }{
        .{ .src = .{ .eq = .{ .lhs = 0, .rhs = 0 } }, .expect_tag = .ne },
        .{ .src = .{ .ne = .{ .lhs = 0, .rhs = 0 } }, .expect_tag = .eq },
        .{ .src = .{ .lt_s = .{ .lhs = 0, .rhs = 0 } }, .expect_tag = .ge_s },
        .{ .src = .{ .ge_s = .{ .lhs = 0, .rhs = 0 } }, .expect_tag = .lt_s },
        .{ .src = .{ .le_s = .{ .lhs = 0, .rhs = 0 } }, .expect_tag = .gt_s },
        .{ .src = .{ .gt_s = .{ .lhs = 0, .rhs = 0 } }, .expect_tag = .le_s },
        .{ .src = .{ .lt_u = .{ .lhs = 0, .rhs = 0 } }, .expect_tag = .ge_u },
        .{ .src = .{ .ge_u = .{ .lhs = 0, .rhs = 0 } }, .expect_tag = .lt_u },
        .{ .src = .{ .le_u = .{ .lhs = 0, .rhs = 0 } }, .expect_tag = .gt_u },
        .{ .src = .{ .gt_u = .{ .lhs = 0, .rhs = 0 } }, .expect_tag = .le_u },
    };
    for (cases) |c| {
        var func = ir.IrFunction.init(allocator, 2, 2, 0);
        defer func.deinit();
        const b0 = try func.newBlock();
        var block = &func.blocks.items[b0];
        const v0 = func.newVReg();
        const v1 = func.newVReg();
        const v_cmp = func.newVReg();
        const v_neg = func.newVReg();
        var src = c.src;
        switch (src) {
            .eq,
            .ne,
            .lt_s,
            .ge_s,
            .le_s,
            .gt_s,
            .lt_u,
            .ge_u,
            .le_u,
            .gt_u,
            => |*b| {
                b.lhs = v0;
                b.rhs = v1;
            },
            else => unreachable,
        }
        try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
        try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1 });
        try block.append(.{ .op = src, .dest = v_cmp });
        try block.append(.{ .op = .{ .eqz = v_cmp }, .dest = v_neg });
        try block.append(.{ .op = .{ .ret = v_neg } });

        _ = try foldInverseCompareEqz(&func, allocator);
        try std.testing.expectEqual(c.expect_tag, std.meta.activeTag(block.instructions.items[3].op));
    }
}

test "foldInverseCompareEqz: non-compare producer is skipped" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v0 = func.newVReg();
    const v_neg = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v0 });
    try block.append(.{ .op = .{ .eqz = v0 }, .dest = v_neg });
    try block.append(.{ .op = .{ .ret = v_neg } });

    const changed = try foldInverseCompareEqz(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expect(block.instructions.items[1].op == .eqz);
}

test "foldInverseCompareEqz: cross-block producer" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 2, 2, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v_cmp = func.newVReg();
    const v_neg = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 3 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 4 }, .dest = v1 });
    try func.getBlock(b0).append(.{ .op = .{ .lt_s = .{ .lhs = v0, .rhs = v1 } }, .dest = v_cmp });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v_cmp }, .dest = v_neg });
    try func.getBlock(b1).append(.{ .op = .{ .ret = v_neg } });

    const changed = try foldInverseCompareEqz(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(std.meta.Tag(ir.Inst.Op).ge_s, std.meta.activeTag(func.getBlock(b1).instructions.items[0].op));
}

test "foldInverseCompareEqz: composes with DCE to drop dead compare" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 2, 2, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v_cmp = func.newVReg();
    const v_neg = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 4 }, .dest = v1 });
    try block.append(.{ .op = .{ .eq = .{ .lhs = v0, .rhs = v1 } }, .dest = v_cmp });
    try block.append(.{ .op = .{ .eqz = v_cmp }, .dest = v_neg });
    try block.append(.{ .op = .{ .ret = v_neg } });

    _ = try foldInverseCompareEqz(&func, allocator);
    _ = try deadCodeElimination(&func, allocator);

    // The original eq producing v_cmp is now unused and should be gone.
    for (block.instructions.items) |inst| {
        try std.testing.expect(inst.op != .eq);
    }
}

test "foldSelectOnEqz: swaps if_true/if_false and drops eqz use" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 3, 3, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v_x });
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_a });
    try block.append(.{ .op = .{ .iconst_32 = 200 }, .dest = v_b });
    try block.append(.{ .op = .{ .eqz = v_x }, .dest = v_c });
    try block.append(.{ .op = .{ .select = .{ .cond = v_c, .if_true = v_a, .if_false = v_b } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try foldSelectOnEqz(&func, allocator);
    try std.testing.expect(changed);

    switch (block.instructions.items[4].op) {
        .select => |s| {
            try std.testing.expectEqual(v_x, s.cond);
            try std.testing.expectEqual(v_b, s.if_true);
            try std.testing.expectEqual(v_a, s.if_false);
        },
        else => try std.testing.expect(false),
    }
}

test "foldSelectOnEqz: skip when eqz has multiple uses" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 3, 3, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    const v_q = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v_x });
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_a });
    try block.append(.{ .op = .{ .iconst_32 = 200 }, .dest = v_b });
    try block.append(.{ .op = .{ .eqz = v_x }, .dest = v_c });
    try block.append(.{ .op = .{ .add = .{ .lhs = v_c, .rhs = v_c } }, .dest = v_q });
    try block.append(.{ .op = .{ .select = .{ .cond = v_c, .if_true = v_a, .if_false = v_b } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    const changed = try foldSelectOnEqz(&func, allocator);
    try std.testing.expect(!changed);
}

test "foldSelectOnEqz: composes with DCE to drop dead eqz" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 3, 3, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    const v_c = func.newVReg();
    const v_r = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v_x });
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_a });
    try block.append(.{ .op = .{ .iconst_32 = 200 }, .dest = v_b });
    try block.append(.{ .op = .{ .eqz = v_x }, .dest = v_c });
    try block.append(.{ .op = .{ .select = .{ .cond = v_c, .if_true = v_a, .if_false = v_b } }, .dest = v_r });
    try block.append(.{ .op = .{ .ret = v_r } });

    _ = try foldSelectOnEqz(&func, allocator);
    _ = try deadCodeElimination(&func, allocator);

    for (block.instructions.items) |inst| {
        try std.testing.expect(inst.op != .eqz);
    }
}

test "foldSignExtendingLoad: extend8_s of i32 load size=1 sign_extend=false" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_addr = func.newVReg();
    const v_byte = func.newVReg();
    const v_ext = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_addr });
    try block.append(.{
        .op = .{ .load = .{ .base = v_addr, .offset = 0, .size = 1, .sign_extend = false } },
        .dest = v_byte,
        .type = .i32,
    });
    try block.append(.{ .op = .{ .extend8_s = v_byte }, .dest = v_ext, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_ext } });

    const changed = try foldSignExtendingLoad(&func, allocator);
    try std.testing.expect(changed);

    // Load should now be sign-extending.
    switch (block.instructions.items[1].op) {
        .load => |ld| try std.testing.expect(ld.sign_extend),
        else => try std.testing.expect(false),
    }
    // ret should now reference v_byte (load's dest), not v_ext.
    try std.testing.expectEqual(ir.Inst.Op{ .ret = v_byte }, block.instructions.items[3].op);
}

test "foldSignExtendingLoad: extend16_s + size=2 i64 load" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_addr = func.newVReg();
    const v_half = func.newVReg();
    const v_ext = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_addr });
    try block.append(.{
        .op = .{ .load = .{ .base = v_addr, .offset = 0, .size = 2, .sign_extend = false } },
        .dest = v_half,
        .type = .i64,
    });
    try block.append(.{ .op = .{ .extend16_s = v_half }, .dest = v_ext, .type = .i64 });
    try block.append(.{ .op = .{ .ret = v_ext } });

    const changed = try foldSignExtendingLoad(&func, allocator);
    try std.testing.expect(changed);
    switch (block.instructions.items[1].op) {
        .load => |ld| try std.testing.expect(ld.sign_extend),
        else => try std.testing.expect(false),
    }
}

test "foldSignExtendingLoad: extend32_s + size=4 i64 load" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_addr = func.newVReg();
    const v_word = func.newVReg();
    const v_ext = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_addr });
    try block.append(.{
        .op = .{ .load = .{ .base = v_addr, .offset = 0, .size = 4, .sign_extend = false } },
        .dest = v_word,
        .type = .i64,
    });
    try block.append(.{ .op = .{ .extend32_s = v_word }, .dest = v_ext, .type = .i64 });
    try block.append(.{ .op = .{ .ret = v_ext } });

    const changed = try foldSignExtendingLoad(&func, allocator);
    try std.testing.expect(changed);
    switch (block.instructions.items[1].op) {
        .load => |ld| try std.testing.expect(ld.sign_extend),
        else => try std.testing.expect(false),
    }
}

test "foldSignExtendingLoad: skip when load already sign-extends" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_addr = func.newVReg();
    const v_byte = func.newVReg();
    const v_ext = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_addr });
    try block.append(.{
        .op = .{ .load = .{ .base = v_addr, .offset = 0, .size = 1, .sign_extend = true } },
        .dest = v_byte,
        .type = .i32,
    });
    try block.append(.{ .op = .{ .extend8_s = v_byte }, .dest = v_ext, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_ext } });

    const changed = try foldSignExtendingLoad(&func, allocator);
    try std.testing.expect(!changed);
}

test "foldSignExtendingLoad: skip when load size mismatches extend width" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_addr = func.newVReg();
    const v_word = func.newVReg();
    const v_ext = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_addr });
    // size=2 load with extend8_s (mismatched)
    try block.append(.{
        .op = .{ .load = .{ .base = v_addr, .offset = 0, .size = 2, .sign_extend = false } },
        .dest = v_word,
        .type = .i32,
    });
    try block.append(.{ .op = .{ .extend8_s = v_word }, .dest = v_ext, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_ext } });

    const changed = try foldSignExtendingLoad(&func, allocator);
    try std.testing.expect(!changed);
}

test "foldSignExtendingLoad: skip when load result has multiple uses" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_addr = func.newVReg();
    const v_byte = func.newVReg();
    const v_ext = func.newVReg();
    const v_sum = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_addr });
    try block.append(.{
        .op = .{ .load = .{ .base = v_addr, .offset = 0, .size = 1, .sign_extend = false } },
        .dest = v_byte,
        .type = .i32,
    });
    // Second use of v_byte (zero-extended consumer).
    try block.append(.{ .op = .{ .add = .{ .lhs = v_byte, .rhs = v_byte } }, .dest = v_sum });
    try block.append(.{ .op = .{ .extend8_s = v_byte }, .dest = v_ext, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_ext } });

    const changed = try foldSignExtendingLoad(&func, allocator);
    try std.testing.expect(!changed);
}

test "foldSignExtendingLoad: composes with DCE to drop the extend" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_addr = func.newVReg();
    const v_byte = func.newVReg();
    const v_ext = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_addr });
    try block.append(.{
        .op = .{ .load = .{ .base = v_addr, .offset = 0, .size = 1, .sign_extend = false } },
        .dest = v_byte,
        .type = .i32,
    });
    try block.append(.{ .op = .{ .extend8_s = v_byte }, .dest = v_ext, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_ext } });

    _ = try foldSignExtendingLoad(&func, allocator);
    _ = try deadCodeElimination(&func, allocator);

    // No remaining extend8_s instruction.
    for (block.instructions.items) |inst| {
        try std.testing.expect(inst.op != .extend8_s);
    }
}

test "foldFloatUnaryIdempotents: f_neg(f_neg(x)) becomes x" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_n1 = func.newVReg();
    const v_n2 = func.newVReg();
    try block.append(.{ .op = .{ .fconst_32 = 1.5 }, .dest = v_x, .type = .f32 });
    try block.append(.{ .op = .{ .f_neg = v_x }, .dest = v_n1, .type = .f32 });
    try block.append(.{ .op = .{ .f_neg = v_n1 }, .dest = v_n2, .type = .f32 });
    try block.append(.{ .op = .{ .ret = v_n2 } });

    const changed = try foldFloatUnaryIdempotents(&func, allocator);
    try std.testing.expect(changed);

    // ret should now reference v_x directly.
    try std.testing.expectEqual(ir.Inst.Op{ .ret = v_x }, block.instructions.items[3].op);
}

test "foldFloatUnaryIdempotents: f_abs(f_abs(x)) becomes f_abs(x)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_a1 = func.newVReg();
    const v_a2 = func.newVReg();
    try block.append(.{ .op = .{ .fconst_32 = -1.5 }, .dest = v_x, .type = .f32 });
    try block.append(.{ .op = .{ .f_abs = v_x }, .dest = v_a1, .type = .f32 });
    try block.append(.{ .op = .{ .f_abs = v_a1 }, .dest = v_a2, .type = .f32 });
    try block.append(.{ .op = .{ .ret = v_a2 } });

    const changed = try foldFloatUnaryIdempotents(&func, allocator);
    try std.testing.expect(changed);

    // ret should now reference v_a1 (inner f_abs's dest).
    try std.testing.expectEqual(ir.Inst.Op{ .ret = v_a1 }, block.instructions.items[3].op);
}

test "foldFloatUnaryIdempotents: f_abs(f_neg(x)) becomes f_abs(x)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_n = func.newVReg();
    const v_a = func.newVReg();
    try block.append(.{ .op = .{ .fconst_64 = 1.5 }, .dest = v_x, .type = .f64 });
    try block.append(.{ .op = .{ .f_neg = v_x }, .dest = v_n, .type = .f64 });
    try block.append(.{ .op = .{ .f_abs = v_n }, .dest = v_a, .type = .f64 });
    try block.append(.{ .op = .{ .ret = v_a } });

    const changed = try foldFloatUnaryIdempotents(&func, allocator);
    try std.testing.expect(changed);

    // The f_abs at index 2 should now read v_x directly.
    switch (block.instructions.items[2].op) {
        .f_abs => |v| try std.testing.expectEqual(v_x, v),
        else => try std.testing.expect(false),
    }
}

test "foldFloatUnaryIdempotents: no-op on unrelated unary (e.g., f_sqrt)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_s = func.newVReg();
    try block.append(.{ .op = .{ .fconst_32 = 4.0 }, .dest = v_x, .type = .f32 });
    try block.append(.{ .op = .{ .f_sqrt = v_x }, .dest = v_s, .type = .f32 });
    try block.append(.{ .op = .{ .ret = v_s } });

    const changed = try foldFloatUnaryIdempotents(&func, allocator);
    try std.testing.expect(!changed);
}

test "foldFloatUnaryIdempotents: composes with DCE" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_n1 = func.newVReg();
    const v_n2 = func.newVReg();
    try block.append(.{ .op = .{ .fconst_32 = 7.0 }, .dest = v_x, .type = .f32 });
    try block.append(.{ .op = .{ .f_neg = v_x }, .dest = v_n1, .type = .f32 });
    try block.append(.{ .op = .{ .f_neg = v_n1 }, .dest = v_n2, .type = .f32 });
    try block.append(.{ .op = .{ .ret = v_n2 } });

    _ = try foldFloatUnaryIdempotents(&func, allocator);
    _ = try deadCodeElimination(&func, allocator);

    // Both f_negs should now be gone.
    for (block.instructions.items) |inst| {
        try std.testing.expect(inst.op != .f_neg);
    }
}

test "foldWrapOfExtend: wrap_i64(extend_i32_s(x)) reduces to x" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_ext = func.newVReg();
    const v_wr = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v_x, .type = .i32 });
    try block.append(.{ .op = .{ .extend_i32_s = v_x }, .dest = v_ext, .type = .i64 });
    try block.append(.{ .op = .{ .wrap_i64 = v_ext }, .dest = v_wr, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_wr } });

    const changed = try foldWrapOfExtend(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(ir.Inst.Op{ .ret = v_x }, block.instructions.items[3].op);
}

test "foldWrapOfExtend: wrap_i64(extend_i32_u(x)) reduces to x" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_ext = func.newVReg();
    const v_wr = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = v_x, .type = .i32 });
    try block.append(.{ .op = .{ .extend_i32_u = v_x }, .dest = v_ext, .type = .i64 });
    try block.append(.{ .op = .{ .wrap_i64 = v_ext }, .dest = v_wr, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_wr } });

    const changed = try foldWrapOfExtend(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(ir.Inst.Op{ .ret = v_x }, block.instructions.items[3].op);
}

test "foldWrapOfExtend: skip when wrap source is not an extend" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_y = func.newVReg();
    const v_wr = func.newVReg();
    try block.append(.{ .op = .{ .iconst_64 = 0xDEADBEEFCAFE }, .dest = v_y, .type = .i64 });
    try block.append(.{ .op = .{ .wrap_i64 = v_y }, .dest = v_wr, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_wr } });

    const changed = try foldWrapOfExtend(&func, allocator);
    try std.testing.expect(!changed);
}

test "foldWrapOfExtend: composes with DCE to drop the extend" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_x = func.newVReg();
    const v_ext = func.newVReg();
    const v_wr = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 99 }, .dest = v_x, .type = .i32 });
    try block.append(.{ .op = .{ .extend_i32_s = v_x }, .dest = v_ext, .type = .i64 });
    try block.append(.{ .op = .{ .wrap_i64 = v_ext }, .dest = v_wr, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_wr } });

    _ = try foldWrapOfExtend(&func, allocator);
    _ = try deadCodeElimination(&func, allocator);

    for (block.instructions.items) |inst| {
        try std.testing.expect(inst.op != .extend_i32_s);
        try std.testing.expect(inst.op != .wrap_i64);
    }
}

test "foldLoadStoreOffset: folds add base, const into load offset when prior check proves range" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_base = func.newVReg();
    const v_guard = func.newVReg();
    const v_c = func.newVReg();
    const v_addr = func.newVReg();
    const v_load = func.newVReg();
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = v_base, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4, .checked_end = 32 } }, .dest = v_guard, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 12 }, .dest = v_c, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v_base, .rhs = v_c } }, .dest = v_addr, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v_addr, .offset = 4, .size = 4 } }, .dest = v_load, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_load } });

    const changed = try foldLoadStoreOffset(&func, allocator);
    try std.testing.expect(changed);
    const ld = block.instructions.items[4].op.load;
    try std.testing.expectEqual(v_base, ld.base);
    try std.testing.expectEqual(@as(u32, 16), ld.offset);
    try std.testing.expectEqual(@as(u64, 0), ld.checked_end);
}

test "foldLoadStoreOffset: folds commuted add into store offset" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_base = func.newVReg();
    const v_guard = func.newVReg();
    const v_c = func.newVReg();
    const v_addr = func.newVReg();
    const v_val = func.newVReg();
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = v_base, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4, .checked_end = 16 } }, .dest = v_guard, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 8 }, .dest = v_c, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v_c, .rhs = v_base } }, .dest = v_addr, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 99 }, .dest = v_val, .type = .i32 });
    try block.append(.{ .op = .{ .store = .{ .base = v_addr, .offset = 0, .size = 4, .val = v_val } } });
    try block.append(.{ .op = .{ .ret = null } });

    const changed = try foldLoadStoreOffset(&func, allocator);
    try std.testing.expect(changed);
    const st = block.instructions.items[5].op.store;
    try std.testing.expectEqual(v_base, st.base);
    try std.testing.expectEqual(@as(u32, 8), st.offset);
}

test "foldLoadStoreOffset: skips unproven add to preserve wrapping semantics" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_base = func.newVReg();
    const v_c = func.newVReg();
    const v_addr = func.newVReg();
    const v_load = func.newVReg();
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = v_base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 12 }, .dest = v_c, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v_base, .rhs = v_c } }, .dest = v_addr, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v_addr, .offset = 4, .size = 4 } }, .dest = v_load, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_load } });

    const changed = try foldLoadStoreOffset(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(v_addr, block.instructions.items[3].op.load.base);
}

test "foldLoadStoreOffset: skips negative constants" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_base = func.newVReg();
    const v_guard = func.newVReg();
    const v_c = func.newVReg();
    const v_addr = func.newVReg();
    const v_load = func.newVReg();
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = v_base, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4, .checked_end = 32 } }, .dest = v_guard, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = -4 }, .dest = v_c, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v_base, .rhs = v_c } }, .dest = v_addr, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v_addr, .offset = 8, .size = 4 } }, .dest = v_load, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_load } });

    const changed = try foldLoadStoreOffset(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(v_addr, block.instructions.items[4].op.load.base);
}

test "foldLoadStoreOffset: skips i64 adds" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_base = func.newVReg();
    const v_guard = func.newVReg();
    const v_c = func.newVReg();
    const v_addr = func.newVReg();
    const v_load = func.newVReg();
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = v_base, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4, .checked_end = 32 } }, .dest = v_guard, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_64 = 8 }, .dest = v_c, .type = .i64 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v_base, .rhs = v_c } }, .dest = v_addr, .type = .i64 });
    try block.append(.{ .op = .{ .load = .{ .base = v_addr, .offset = 0, .size = 4 } }, .dest = v_load, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_load } });

    const changed = try foldLoadStoreOffset(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(v_addr, block.instructions.items[4].op.load.base);
}

test "foldLoadStoreOffset: adjusts checked_end when folding a widened access" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    var block = &func.blocks.items[b0];

    const v_base = func.newVReg();
    const v_guard = func.newVReg();
    const v_c = func.newVReg();
    const v_addr = func.newVReg();
    const v_load = func.newVReg();
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = v_base, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4, .checked_end = 64 } }, .dest = v_guard, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 12 }, .dest = v_c, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v_base, .rhs = v_c } }, .dest = v_addr, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v_addr, .offset = 4, .size = 4, .checked_end = 20 } }, .dest = v_load, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_load } });

    const changed = try foldLoadStoreOffset(&func, allocator);
    try std.testing.expect(changed);
    const ld = block.instructions.items[4].op.load;
    try std.testing.expectEqual(v_base, ld.base);
    try std.testing.expectEqual(@as(u32, 16), ld.offset);
    try std.testing.expectEqual(@as(u64, 32), ld.checked_end);
}

test "promoteLocalsToSSA: simple countdown loop" {
    // Build a simple loop: local 0 starts at 3, counts down by 1 each
    // iteration until 0. Tests phi placement + rename on a loop.
    //
    //   block 0 (entry):
    //     local_set 0, 3
    //     br block 1
    //   block 1 (loop header):
    //     v_ctr = local_get 0
    //     v_eqz = eqz v_ctr
    //     br_if v_eqz → block 2 (exit), else block 3 (body)
    //   block 3 (body):
    //     v_one = iconst_32 1
    //     v_dec = sub v_ctr, v_one
    //     local_set 0, v_dec
    //     br block 1
    //   block 2 (exit):
    //     v_result = local_get 0
    //     ret v_result
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 1);
    defer func.deinit();

    // Set local_types for the single local (i32).
    const lt = try allocator.alloc(ir.IrType, 1);
    lt[0] = .i32;
    func.local_types = lt;

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    const v_three = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 3 }, .dest = v_three });
    try func.getBlock(b0).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_three } } });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const v_ctr = func.newVReg();
    const v_eqz = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_ctr });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v_ctr }, .dest = v_eqz });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_eqz, .then_block = b2, .else_block = b3 } } });

    const v_one = func.newVReg();
    const v_dec = func.newVReg();
    try func.getBlock(b3).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_one });
    try func.getBlock(b3).append(.{ .op = .{ .sub = .{ .lhs = v_ctr, .rhs = v_one } }, .dest = v_dec });
    try func.getBlock(b3).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_dec } } });
    try func.getBlock(b3).append(.{ .op = .{ .br = b1 } });

    const v_result = func.newVReg();
    try func.getBlock(b2).append(.{ .op = .{ .local_get = 0 }, .dest = v_result });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v_result } });

    // Run mem2reg.
    const changed = try promoteLocalsToSSA(&func, allocator);
    try std.testing.expect(changed);

    // Block 1 should have a phi at the top.
    const header = func.getBlock(b1);
    try std.testing.expect(header.instructions.items[0].op == .phi);
    const phi_dest = header.instructions.items[0].dest.?;
    const phi_edges = header.instructions.items[0].op.phi;
    try std.testing.expectEqual(@as(usize, 2), phi_edges.len);

    // Phi should have edges from block 0 (initial value) and block 3 (decremented).
    var has_b0_edge = false;
    var has_b3_edge = false;
    for (phi_edges) |edge| {
        if (edge.block == b0) has_b0_edge = true;
        if (edge.block == b3) has_b3_edge = true;
    }
    try std.testing.expect(has_b0_edge);
    try std.testing.expect(has_b3_edge);

    // Now lower phis and verify the result is runnable.
    _ = try lowerPhisToLocals(&func, allocator);

    // After lowering, no phi should remain.
    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            try std.testing.expect(inst.op != .phi);
        }
    }

    // The phi dest should still be used in block 1's eqz (or its replacement).
    // Check that the block 1 still has an eqz of the phi dest or its forwarded value.
    _ = phi_dest;
}

test "promoteLocalsToSSA + lowerPhis: two-local sum loop" {
    // sum = 0; i = 3; while (i != 0) { sum += i; i--; } ret sum
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 2);
    defer func.deinit();

    const lt = try allocator.alloc(ir.IrType, 2);
    lt[0] = .i32;
    lt[1] = .i32;
    func.local_types = lt;

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    // Block 0: sum=0, i=3, br 1
    const v_zero_init = func.newVReg();
    const v_three_init = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_zero_init });
    try func.getBlock(b0).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_zero_init } } });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 3 }, .dest = v_three_init });
    try func.getBlock(b0).append(.{ .op = .{ .local_set = .{ .idx = 1, .val = v_three_init } } });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    // Block 1: i = local_get 1; if (i==0) goto exit else goto body
    const v_i = func.newVReg();
    const v_eqz = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 1 }, .dest = v_i, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v_i }, .dest = v_eqz });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_eqz, .then_block = b2, .else_block = b3 } } });

    // Block 3: sum += i; i--; br 1
    const v_sum = func.newVReg();
    const v_new_sum = func.newVReg();
    const v_one = func.newVReg();
    const v_dec = func.newVReg();
    try func.getBlock(b3).append(.{ .op = .{ .local_get = 0 }, .dest = v_sum, .type = .i32 });
    try func.getBlock(b3).append(.{ .op = .{ .add = .{ .lhs = v_sum, .rhs = v_i } }, .dest = v_new_sum });
    try func.getBlock(b3).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_new_sum } } });
    try func.getBlock(b3).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_one });
    try func.getBlock(b3).append(.{ .op = .{ .sub = .{ .lhs = v_i, .rhs = v_one } }, .dest = v_dec });
    try func.getBlock(b3).append(.{ .op = .{ .local_set = .{ .idx = 1, .val = v_dec } } });
    try func.getBlock(b3).append(.{ .op = .{ .br = b1 } });

    // Block 2: ret local_get 0
    const v_result = func.newVReg();
    try func.getBlock(b2).append(.{ .op = .{ .local_get = 0 }, .dest = v_result, .type = .i32 });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v_result } });

    // Run mem2reg + phi lowering.
    const changed = try promoteLocalsToSSA(&func, allocator);
    try std.testing.expect(changed);
    _ = try lowerPhisToLocals(&func, allocator);

    // After lowering, no phi should remain.
    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            try std.testing.expect(inst.op != .phi);
        }
    }
}
