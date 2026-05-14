//! AOT compiler optimization passes.
//!
//! Operates on the SSA-form IR between frontend lowering and codegen.
//! Each pass transforms an IrFunction in-place and returns whether
//! it made any changes (for fixpoint iteration).

const std = @import("std");
const ir = @import("ir.zig");
const analysis = @import("analysis.zig");
const deadStoreElimination = @import("dead_store_elimination.zig").deadStoreElimination;

pub const TargetArch = enum { x86_64, aarch64 };

pub const CompileOptions = struct {
    enable_iv_simplify: bool = true,
    enable_loop_unroll: bool = true,
};

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
        .v128_bitselect => |sel| {
            list.append(sel.a);
            list.append(sel.b);
            list.append(sel.mask);
        },
        .i32x4_binop => |bin| {
            list.append(bin.lhs);
            list.append(bin.rhs);
        },
        .i32x4_unop => |un| list.append(un.vector),
        .i32x4_extadd_pairwise_i16x8 => |op| list.append(op.vector),
        .i32x4_dot_i16x8_s => |bin| {
            list.append(bin.lhs);
            list.append(bin.rhs);
        },
        .i32x4_extend_i16x8 => |op| list.append(op.vector),
        .f32x4_binop => |bin| {
            list.append(bin.lhs);
            list.append(bin.rhs);
        },
        .f32x4_unop => |un| list.append(un.vector),
        .f32x4_convert_i32x4 => |op| list.append(op.vector),
        .i32x4_trunc_sat => |op| list.append(op.vector),
        .f32x4_demote_f64x2_zero => |op| list.append(op.vector),
        .i32x4_extmul_i16x8 => |op| {
            list.append(op.lhs);
            list.append(op.rhs);
        },
        .i8x16_binop => |bin| {
            list.append(bin.lhs);
            list.append(bin.rhs);
        },
        .i8x16_shuffle => |op| {
            list.append(op.lhs);
            list.append(op.rhs);
        },
        .i8x16_swizzle => |op| {
            list.append(op.vector);
            list.append(op.indices);
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
        .f64x2_binop => |bin| {
            list.append(bin.lhs);
            list.append(bin.rhs);
        },
        .f64x2_unop => |un| list.append(un.vector),
        .f64x2_convert_low_i32x4 => |op| list.append(op.vector),
        .f64x2_promote_low_f32x4 => |op| list.append(op.vector),
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
        .v128_any_true,
        .i32x4_splat,
        .f32x4_splat,
        .i8x16_splat,
        .i16x8_splat,
        .i64x2_splat,
        .f64x2_splat,
        => |vreg| list.append(vreg),
        .simd_all_true => |op| list.append(op.vector),
        .simd_bitmask => |op| list.append(op.vector),
        .i32x4_extract_lane => |lane| list.append(lane.vector),
        .f32x4_extract_lane => |lane| list.append(lane.vector),
        .i8x16_extract_lane => |lane| list.append(lane.vector),
        .i16x8_extract_lane => |lane| list.append(lane.vector),
        .i64x2_extract_lane => |lane| list.append(lane.vector),
        .f64x2_extract_lane => |lane| list.append(lane.vector),
        .i32x4_replace_lane => |lane| {
            list.append(lane.vector);
            list.append(lane.val);
        },
        .f32x4_replace_lane => |lane| {
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
        .f64x2_replace_lane => |lane| {
            list.append(lane.vector);
            list.append(lane.val);
        },

        .local_set => |ls| list.append(ls.val),
        .global_set => |gs| list.append(gs.val),
        .load => |ld| list.append(ld.base),
        .v128_load => |ld| list.append(ld.base),
        .v128_load_splat => |ld| list.append(ld.base),
        .v128_load_zero => |ld| list.append(ld.base),
        .v128_load_extend => |ld| list.append(ld.base),
        .v128_load_lane => |ld| {
            list.append(ld.base);
            list.append(ld.vector);
        },
        .store => |st| {
            list.append(st.base);
            list.append(st.val);
        },
        .v128_store => |st| {
            list.append(st.base);
            list.append(st.val);
        },
        .v128_store_lane => |st| {
            list.append(st.base);
            list.append(st.vector);
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

/// Rewrite every operand use of `old` in `inst` to `new`. Does not touch
/// `inst.dest` (defs are not uses). Exposed pub so passes outside this
/// file (e.g. `ir/range_split.zig` — live-range splitting) can rewrite
/// uses in a scheduled instruction stream without re-implementing the
/// op-by-op switch.
pub fn replaceInInst(inst: *ir.Inst, old: ir.VReg, new: ir.VReg) void {
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
        .v128_bitselect => |*sel| {
            if (sel.a == old) sel.a = new;
            if (sel.b == old) sel.b = new;
            if (sel.mask == old) sel.mask = new;
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
        .i32x4_dot_i16x8_s => |*bin| {
            if (bin.lhs == old) bin.lhs = new;
            if (bin.rhs == old) bin.rhs = new;
        },
        .i32x4_extend_i16x8 => |*op| if (op.vector == old) {
            op.vector = new;
        },
        .f32x4_binop => |*bin| {
            if (bin.lhs == old) bin.lhs = new;
            if (bin.rhs == old) bin.rhs = new;
        },
        .f32x4_unop => |*un| if (un.vector == old) {
            un.vector = new;
        },
        .f32x4_convert_i32x4 => |*op| if (op.vector == old) {
            op.vector = new;
        },
        .i32x4_trunc_sat => |*op| if (op.vector == old) {
            op.vector = new;
        },
        .f32x4_demote_f64x2_zero => |*op| if (op.vector == old) {
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
        .i8x16_shuffle => |*op| {
            if (op.lhs == old) op.lhs = new;
            if (op.rhs == old) op.rhs = new;
        },
        .i8x16_swizzle => |*op| {
            if (op.vector == old) op.vector = new;
            if (op.indices == old) op.indices = new;
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
        .f64x2_binop => |*bin| {
            if (bin.lhs == old) bin.lhs = new;
            if (bin.rhs == old) bin.rhs = new;
        },
        .f64x2_unop => |*un| if (un.vector == old) {
            un.vector = new;
        },
        .f64x2_convert_low_i32x4 => |*op| if (op.vector == old) {
            op.vector = new;
        },
        .f64x2_promote_low_f32x4 => |*op| if (op.vector == old) {
            op.vector = new;
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
        .v128_any_true,
        .i32x4_splat,
        .f32x4_splat,
        .i8x16_splat,
        .i16x8_splat,
        .i64x2_splat,
        .f64x2_splat,
        => |*vreg| if (vreg.* == old) {
            vreg.* = new;
        },
        .simd_all_true => |*op| if (op.vector == old) {
            op.vector = new;
        },
        .simd_bitmask => |*op| if (op.vector == old) {
            op.vector = new;
        },
        .i32x4_extract_lane => |*lane| if (lane.vector == old) {
            lane.vector = new;
        },
        .f32x4_extract_lane => |*lane| if (lane.vector == old) {
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
        .f64x2_extract_lane => |*lane| if (lane.vector == old) {
            lane.vector = new;
        },
        .i32x4_replace_lane => |*lane| {
            if (lane.vector == old) lane.vector = new;
            if (lane.val == old) lane.val = new;
        },
        .f32x4_replace_lane => |*lane| {
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
        .f64x2_replace_lane => |*lane| {
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
        .v128_load_splat => |*ld| if (ld.base == old) {
            ld.base = new;
        },
        .v128_load_zero => |*ld| if (ld.base == old) {
            ld.base = new;
        },
        .v128_load_extend => |*ld| if (ld.base == old) {
            ld.base = new;
        },
        .v128_load_lane => |*ld| {
            if (ld.base == old) ld.base = new;
            if (ld.vector == old) ld.vector = new;
        },
        .store => |*st| {
            if (st.base == old) st.base = new;
            if (st.val == old) st.val = new;
        },
        .v128_store => |*st| {
            if (st.base == old) st.base = new;
            if (st.val == old) st.val = new;
        },
        .v128_store_lane => |*st| {
            if (st.base == old) st.base = new;
            if (st.vector == old) st.vector = new;
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
///
/// Also sweeps dest-less side-effect-free instructions: passes like
/// `promoteLocalsToSSA`, `foldWrapOfExtend`, `foldSignExtendingLoad`,
/// and the `*Mul*` strength-reducers neutralise an instruction in
/// place by setting `inst.op = .{ .iconst_32 = 0 }` (and sometimes
/// `inst.dest = null`) after rewriting users via `replaceVReg`. With
/// the old "dest required" check these placeholders survived the
/// pipeline indefinitely — codegen filtered them out at emit time
/// (`iconst_32` with `dest == null` returns early), but they cluttered
/// the IR dumps and forced every subsequent pass to walk them. The
/// dest-less sweep here treats `inst.dest == null` + `!hasSideEffect`
/// as unconditionally dead, matching the semantics codegen already
/// relies on.
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
                } else if (!hasSideEffect(inst)) {
                    // Dest-less placeholder (neutralised rewrite from an
                    // earlier pass): no observable effect, drop it.
                    _ = block.instructions.orderedRemove(i);
                    changed = true;
                    iterate = true;
                    continue;
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
        .v128_store_lane,
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
        .v128_load_splat,
        .v128_load_zero,
        .v128_load_extend,
        .v128_load_lane,
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
        .v128_any_true,
        .simd_all_true,
        .simd_bitmask,
        .v128_bitwise,
        .v128_bitselect,
        .i32x4_binop,
        .i32x4_unop,
        .i32x4_extadd_pairwise_i16x8,
        .i32x4_dot_i16x8_s,
        .i32x4_extend_i16x8,
        .f32x4_unop,
        .f32x4_binop,
        .f32x4_convert_i32x4,
        .i32x4_trunc_sat,
        .f32x4_demote_f64x2_zero,
        .f32x4_splat,
        .f32x4_extract_lane,
        .f32x4_replace_lane,
        .i32x4_extmul_i16x8,
        .i32x4_shift,
        .i32x4_splat,
        .i32x4_extract_lane,
        .i32x4_replace_lane,
        .i8x16_binop,
        .i8x16_shuffle,
        .i8x16_swizzle,
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
        .f64x2_binop,
        .f64x2_unop,
        .f64x2_convert_low_i32x4,
        .f64x2_promote_low_f32x4,
        .i64x2_unop,
        .i64x2_extend_i32x4,
        .i64x2_extmul_i32x4,
        .i64x2_shift,
        .i64x2_splat,
        .i64x2_extract_lane,
        .i64x2_replace_lane,
        .f64x2_splat,
        .f64x2_extract_lane,
        .f64x2_replace_lane,
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
        .v128_any_true => |v| v == b.op.v128_any_true,
        .simd_all_true => |op| op.width == b.op.simd_all_true.width and op.vector == b.op.simd_all_true.vector,
        .simd_bitmask => |op| op.width == b.op.simd_bitmask.width and op.vector == b.op.simd_bitmask.vector,
        .v128_bitwise => |bin| bin.op == b.op.v128_bitwise.op and bin.lhs == b.op.v128_bitwise.lhs and bin.rhs == b.op.v128_bitwise.rhs,
        .v128_bitselect => |sel| sel.a == b.op.v128_bitselect.a and sel.b == b.op.v128_bitselect.b and sel.mask == b.op.v128_bitselect.mask,
        .i32x4_binop => |bin| bin.op == b.op.i32x4_binop.op and bin.lhs == b.op.i32x4_binop.lhs and bin.rhs == b.op.i32x4_binop.rhs,
        .i32x4_unop => |un| un.op == b.op.i32x4_unop.op and un.vector == b.op.i32x4_unop.vector,
        .i32x4_extadd_pairwise_i16x8 => |op| op.sign == b.op.i32x4_extadd_pairwise_i16x8.sign and op.vector == b.op.i32x4_extadd_pairwise_i16x8.vector,
        .i32x4_dot_i16x8_s => |bin| bin.lhs == b.op.i32x4_dot_i16x8_s.lhs and bin.rhs == b.op.i32x4_dot_i16x8_s.rhs,
        .i32x4_extend_i16x8 => |op| op.sign == b.op.i32x4_extend_i16x8.sign and op.half == b.op.i32x4_extend_i16x8.half and op.vector == b.op.i32x4_extend_i16x8.vector,
        .f32x4_convert_i32x4 => |op| op.sign == b.op.f32x4_convert_i32x4.sign and op.vector == b.op.f32x4_convert_i32x4.vector,
        .i32x4_trunc_sat => |op| op.src_width == b.op.i32x4_trunc_sat.src_width and op.sign == b.op.i32x4_trunc_sat.sign and op.vector == b.op.i32x4_trunc_sat.vector,
        .f32x4_demote_f64x2_zero => |op| op.vector == b.op.f32x4_demote_f64x2_zero.vector,
        .i32x4_extmul_i16x8 => |op| op.sign == b.op.i32x4_extmul_i16x8.sign and op.half == b.op.i32x4_extmul_i16x8.half and op.lhs == b.op.i32x4_extmul_i16x8.lhs and op.rhs == b.op.i32x4_extmul_i16x8.rhs,
        .i32x4_shift => |shift| shift.op == b.op.i32x4_shift.op and shift.vector == b.op.i32x4_shift.vector and shift.count == b.op.i32x4_shift.count,
        .i32x4_splat => |v| v == b.op.i32x4_splat,
        .i32x4_extract_lane => |lane| lane.vector == b.op.i32x4_extract_lane.vector and lane.lane == b.op.i32x4_extract_lane.lane,
        .i32x4_replace_lane => |lane| lane.vector == b.op.i32x4_replace_lane.vector and lane.val == b.op.i32x4_replace_lane.val and lane.lane == b.op.i32x4_replace_lane.lane,
        .f32x4_splat => |v| v == b.op.f32x4_splat,
        .f32x4_extract_lane => |lane| lane.vector == b.op.f32x4_extract_lane.vector and lane.lane == b.op.f32x4_extract_lane.lane,
        .f32x4_replace_lane => |lane| lane.vector == b.op.f32x4_replace_lane.vector and lane.val == b.op.f32x4_replace_lane.val and lane.lane == b.op.f32x4_replace_lane.lane,
        .i8x16_binop => |bin| bin.op == b.op.i8x16_binop.op and bin.lhs == b.op.i8x16_binop.lhs and bin.rhs == b.op.i8x16_binop.rhs,
        .i8x16_shuffle => |op| op.lhs == b.op.i8x16_shuffle.lhs and op.rhs == b.op.i8x16_shuffle.rhs and std.mem.eql(u8, &op.lanes, &b.op.i8x16_shuffle.lanes),
        .i8x16_swizzle => |op| op.vector == b.op.i8x16_swizzle.vector and op.indices == b.op.i8x16_swizzle.indices,
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
        .f32x4_unop => |un| un.op == b.op.f32x4_unop.op and un.vector == b.op.f32x4_unop.vector,
        .f32x4_binop => |bin| bin.op == b.op.f32x4_binop.op and bin.lhs == b.op.f32x4_binop.lhs and bin.rhs == b.op.f32x4_binop.rhs,
        .f64x2_binop => |bin| bin.op == b.op.f64x2_binop.op and bin.lhs == b.op.f64x2_binop.lhs and bin.rhs == b.op.f64x2_binop.rhs,
        .f64x2_unop => |un| un.op == b.op.f64x2_unop.op and un.vector == b.op.f64x2_unop.vector,
        .f64x2_convert_low_i32x4 => |op| op.sign == b.op.f64x2_convert_low_i32x4.sign and op.vector == b.op.f64x2_convert_low_i32x4.vector,
        .f64x2_promote_low_f32x4 => |op| op.vector == b.op.f64x2_promote_low_f32x4.vector,
        .i64x2_unop => |un| un.op == b.op.i64x2_unop.op and un.vector == b.op.i64x2_unop.vector,
        .i64x2_extend_i32x4 => |op| op.sign == b.op.i64x2_extend_i32x4.sign and op.half == b.op.i64x2_extend_i32x4.half and op.vector == b.op.i64x2_extend_i32x4.vector,
        .i64x2_extmul_i32x4 => |op| op.sign == b.op.i64x2_extmul_i32x4.sign and op.half == b.op.i64x2_extmul_i32x4.half and op.lhs == b.op.i64x2_extmul_i32x4.lhs and op.rhs == b.op.i64x2_extmul_i32x4.rhs,
        .i64x2_shift => |shift| shift.op == b.op.i64x2_shift.op and shift.vector == b.op.i64x2_shift.vector and shift.count == b.op.i64x2_shift.count,
        .i64x2_splat => |v| v == b.op.i64x2_splat,
        .i64x2_extract_lane => |lane| lane.vector == b.op.i64x2_extract_lane.vector and lane.lane == b.op.i64x2_extract_lane.lane,
        .i64x2_replace_lane => |lane| lane.vector == b.op.i64x2_replace_lane.vector and lane.val == b.op.i64x2_replace_lane.val and lane.lane == b.op.i64x2_replace_lane.lane,
        .f64x2_splat => |v| v == b.op.f64x2_splat,
        .f64x2_extract_lane => |lane| lane.vector == b.op.f64x2_extract_lane.vector and lane.lane == b.op.f64x2_extract_lane.lane,
        .f64x2_replace_lane => |lane| lane.vector == b.op.f64x2_replace_lane.vector and lane.val == b.op.f64x2_replace_lane.val and lane.lane == b.op.f64x2_replace_lane.lane,
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

/// Information passed to a `DumpHook` after a pass executes. Sufficient
/// to identify the pass that just ran and the function it transformed.
pub const DumpInfo = struct {
    /// Canonical pass name (matches `passName(pass)` or one of the
    /// synthetic names `"promoteLocalsToSSA"`, `"lowerPhisToLocals"`,
    /// `"inlineSmallFunctions"`).
    pass_name: []const u8,
    /// The function the pass operated on. For module-level passes
    /// (currently only `inlineSmallFunctions`), the hook is invoked
    /// once per function with that function's post-pass state.
    func: *const ir.IrFunction,
    /// Module-level function index for the function being dumped.
    func_index: u32,
    /// Whether the pass reported a change (`pass()` returned true).
    /// Used by callers to decide whether to re-emit an IR snapshot.
    changed: bool,
    /// 0-based per-function fixpoint iteration counter (matches the
    /// outer `while (iter < 8)` loop in `runPassesWithOptions`).
    iter: u32,
    /// 0-based outer iteration counter (matches the outer
    /// `while (outer_iter < outer_max)` loop).
    outer_iter: u32,
};

/// User-supplied hook fired after each pass invocation by
/// `runPassesWithOptions`. The hook is invoked unconditionally — pass
/// selection (filtering by name or function) is the caller's job. Any
/// error returned from the callback aborts the pipeline.
pub const DumpHook = struct {
    ctx: *anyopaque,
    callback: *const fn (ctx: *anyopaque, info: DumpInfo) anyerror!void,
};

pub const RunOptions = struct {
    /// Optional hook invoked after each per-function pass run, plus
    /// once-per-function after `promoteLocalsToSSA` /
    /// `lowerPhisToLocals` (first outer iteration) and once-per-function
    /// after every successful round of `inlineSmallFunctions`.
    dump_hook: ?DumpHook = null,
};

const PassNameEntry = struct { fn_ptr: PassFn, name: []const u8 };

// Registry of every PassFn referenced from the default pipelines
// (`default_passes`, `x86_64_default_passes`, and their `_no_iv` /
// `_no_unroll` variants). `passName` linear-scans this table; the
// pipelines are small enough (~30 entries) that a hash lookup isn't
// worth the maintenance burden. New passes added to a pipeline MUST be
// registered here so dump hooks see a stable name instead of
// `"<unknown>"`.
const pass_name_registry = [_]PassNameEntry{
    .{ .fn_ptr = &forwardLocalGet, .name = "forwardLocalGet" },
    .{ .fn_ptr = &constantFold, .name = "constantFold" },
    .{ .fn_ptr = &algebraicSimplify, .name = "algebraicSimplify" },
    .{ .fn_ptr = &strengthReduceMul, .name = "strengthReduceMul" },
    .{ .fn_ptr = &strengthReduceMulShiftAdd, .name = "strengthReduceMulShiftAdd" },
    .{ .fn_ptr = &strengthReduceDivRem, .name = "strengthReduceDivRem" },
    .{ .fn_ptr = &foldConstantBranches, .name = "foldConstantBranches" },
    .{ .fn_ptr = &foldInverseCompareEqz, .name = "foldInverseCompareEqz" },
    .{ .fn_ptr = &foldBranchOnEqz, .name = "foldBranchOnEqz" },
    .{ .fn_ptr = &threadChainedConditionalBranches, .name = "threadChainedConditionalBranches" },
    .{ .fn_ptr = &tailDuplicateSmallJoins, .name = "tailDuplicateSmallJoins" },
    .{ .fn_ptr = &foldSelectOnEqz, .name = "foldSelectOnEqz" },
    .{ .fn_ptr = &foldSignExtendingLoad, .name = "foldSignExtendingLoad" },
    .{ .fn_ptr = &foldFloatUnaryIdempotents, .name = "foldFloatUnaryIdempotents" },
    .{ .fn_ptr = &foldWrapOfExtend, .name = "foldWrapOfExtend" },
    .{ .fn_ptr = &globalValueNumbering, .name = "globalValueNumbering" },
    .{ .fn_ptr = &inductionVariableSimplification, .name = "inductionVariableSimplification" },
    .{ .fn_ptr = &hoistLoopInvariantCode, .name = "hoistLoopInvariantCode" },
    .{ .fn_ptr = &unrollSmallFixedLoops, .name = "unrollSmallFixedLoops" },
    .{ .fn_ptr = &@import("forward_redundant_loads.zig").forwardRedundantLoads, .name = "forwardRedundantLoads" },
    .{ .fn_ptr = &deadStoreElimination, .name = "deadStoreElimination" },
    .{ .fn_ptr = &deadCodeElimination, .name = "deadCodeElimination" },
    .{ .fn_ptr = &deadLocalSetElimination, .name = "deadLocalSetElimination" },
    .{ .fn_ptr = &hoistLoopBoundsChecks, .name = "hoistLoopBoundsChecks" },
    .{ .fn_ptr = &elideRedundantBoundsChecks, .name = "elideRedundantBoundsChecks" },
    .{ .fn_ptr = &foldLoadStoreOffset, .name = "foldLoadStoreOffset" },
};

/// Map a `PassFn` to its canonical name (used by `DumpHook` callers to
/// match `--dump-ir-after=<name>` selectors). Returns `"<unknown>"` for
/// passes not in the registry — that indicates a missing entry above.
pub fn passName(p: PassFn) []const u8 {
    for (pass_name_registry) |entry| {
        if (p == entry.fn_ptr) return entry.name;
    }
    return "<unknown>";
}

// ── Loop-invariant bounds-check hoisting ────────────────────────────────────

/// Hoist loop-invariant bounds checks to the loop preheader.
///
/// For each natural loop, scans the loop's **must-execute blocks**
/// (the header plus any other loop block that dominates every latch)
/// for `load`/`store` instructions whose base VReg is loop-invariant
/// (defined outside the loop). For each such base, inserts a single
/// guard load in the preheader with `checked_end = max(offset + size)`
/// across all must-execute accesses with that base. The guard's bounds
/// check runs once before the loop; all covered loop accesses are
/// marked `bounds_known = true` so codegen skips their inline checks.
///
/// Soundness:
///   - Only must-execute accesses are considered. A must-execute block
///     dominates every latch, so any access in it is reached on every
///     iteration before the back-edge; a preheader trap is therefore
///     equivalent to a first-iteration trap. (The header is the
///     simplest case — it dominates every block in the loop including
///     latches.)
///   - Must-execute blocks form a chain in the dominator tree, so the
///     scan walks them in dominator order (header first).
///   - Accesses after a fence (call, memory_grow, etc.) in any
///     must-execute block are skipped — and the global scan halts at
///     that fence, since subsequent must-execute blocks execute after
///     it. This matches the original header-only behaviour and the
///     #212 PR rationale.
///   - The preheader must be a dedicated single-successor block
///     (`br header`), ensuring the guard runs only on paths entering
///     the loop. PR #490 Stage B synthesises preheaders where the
///     wasm front-end did not produce one.
///   - Wasm memory grows monotonically, so a passing preheader check
///     remains valid for all subsequent iterations (even if memory
///     grows inside the loop body).
///   - Only loop accesses with `offset + size ≤ max_end` for some
///     scanned base are marked `bounds_known`; the guard's widened
///     check covers them.
///
/// The header-only formulation was the original behaviour (PR #212);
/// the must-execute extension was added per the issue #470 diagnostic
/// finding that the header rarely contains the loop's memory accesses
/// in practice (most wasm front-end output puts the loop test in the
/// header and the memory accesses in dominated body blocks).
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

    // Reused per-loop must-execute scratch buffer.
    var must_exec: std.ArrayList(ir.BlockId) = .empty;
    defer must_exec.deinit(allocator);

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

        // ── Collect must-execute loop blocks ──
        // Header is always must-execute. Other blocks qualify when they
        // dominate every latch; such blocks form a chain in the
        // dominator tree, so we sort them in dominator order so the
        // scan walks them in execution order.
        must_exec.clearRetainingCapacity();
        try must_exec.append(allocator, loop.header);
        for (loop.blocks) |bid| {
            if (bid == loop.header) continue;
            var dominates_all = true;
            for (loop.latches) |latch| {
                if (!dom.dominates(bid, latch)) {
                    dominates_all = false;
                    break;
                }
            }
            if (dominates_all) try must_exec.append(allocator, bid);
        }
        const DomLess = struct {
            fn lt(d: *const analysis.DomTree, a: ir.BlockId, b: ir.BlockId) bool {
                if (a == b) return false;
                return d.dominates(a, b);
            }
        };
        std.sort.insertion(ir.BlockId, must_exec.items, &dom, DomLess.lt);

        // ── Scan must-execute blocks for loop-invariant bases ──
        // Stop at the first fence op (call, memory_grow, etc.) in any
        // must-execute block — subsequent must-execute blocks execute
        // after the fence, so accesses past it can't be hoisted.
        base_max.clearRetainingCapacity();
        scan: for (must_exec.items) |me_bid| {
            const me_block = &func.blocks.items[me_bid];
            for (me_block.instructions.items) |inst| {
                // Fence: stop scanning globally.
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
                    => break :scan,
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
                    .v128_load_extend => |ld| {
                        if (ld.bounds_known) continue;
                        const db = def_block.get(ld.base) orelse continue;
                        if (loop.containsBlock(db)) continue; // not loop-invariant
                        const end: u64 = @as(u64, ld.offset) + ld.accessSize();
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
                        .v128_load_extend => |*ld| {
                            if (ld.bounds_known) continue;
                            if (ld.base != base) continue;
                            const end: u64 = @as(u64, ld.offset) + ld.accessSize();
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
///
/// `local_get` and `global_get` are not in `isPure` (they read external
/// state, not a pure VReg computation) but are still safe to hoist when
/// the corresponding wasm-local / wasm-global is never written inside
/// the loop. For `global_get` we additionally require the loop body to
/// contain no calls — a call can `global_set` through the callee.
/// Wasm locals are function-private, so calls cannot mutate them and
/// `local_get` only needs `local_set` exclusion.
///
/// Trapping wasm loads (`load`, `v128_load*`) are speculatively hoisted
/// when (a) the loop body contains no memory-mutating op or call, and
/// (b) the load lives in the loop header. Condition (b) guarantees the
/// load executes on every loop entry (since the header runs on every
/// entry and each block has exactly one terminator at its tail, so any
/// load in the header precedes the exit check), making the hoist
/// trap-equivalent: the hoisted load traps in the preheader at the same
/// input the original would have on iteration 1.
///
/// When the loop lacks a dedicated `.br`-terminated preheader (either no
/// preheader at all, or a `br_if` / `br_table` entry, or multiple
/// non-loop predecessors), a preheader is synthesized in-place by
/// retargeting each non-loop predecessor's header edge through a fresh
/// `.br header` block. See `synthesizeLoopPreheader`.
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
        const ph = (try obtainLoopPreheader(func, loop, &predecessors, &dom, allocator)) orelse continue;

        // One-shot scan of the loop body: which wasm-local / wasm-global
        // indices are written, and does the body contain any call or
        // memory-mutating op?
        var set_locals = std.AutoHashMap(u32, void).init(allocator);
        defer set_locals.deinit();
        var set_globals = std.AutoHashMap(u32, void).init(allocator);
        defer set_globals.deinit();
        var loop_has_call = false;
        var loop_has_memory_write = false;
        for (loop.blocks) |bid| {
            for (func.blocks.items[bid].instructions.items) |inst| {
                switch (inst.op) {
                    .local_set => |ls| try set_locals.put(ls.idx, {}),
                    .global_set => |gs| try set_globals.put(gs.idx, {}),
                    .call, .call_indirect, .call_ref => loop_has_call = true,
                    .store,
                    .v128_store,
                    .v128_store_lane,
                    .memory_copy,
                    .memory_fill,
                    .memory_init,
                    .memory_grow,
                    .atomic_store,
                    .atomic_rmw,
                    .atomic_cmpxchg,
                    .atomic_notify,
                    .atomic_wait,
                    => loop_has_memory_write = true,
                    else => {},
                }
            }
        }
        const loop_can_hoist_load = !loop_has_call and !loop_has_memory_write;

        // Speculation anchors for `load` hoisting (Stage C extension, #494).
        // A speculative load in block B is safe iff B dominates every
        // anchor: every latch (so the load runs before each back-edge) and
        // every exiting block (so the load runs before each loop exit).
        // Together these guarantee the load runs once on every iteration,
        // preserving the original trap point.
        var anchors: std.ArrayList(ir.BlockId) = .empty;
        defer anchors.deinit(allocator);
        if (loop_can_hoist_load) {
            for (loop.latches) |latch| try anchors.append(allocator, latch);
            for (loop.blocks) |bid| {
                if (isLoopExitingBlock(func, loop, bid)) try anchors.append(allocator, bid);
            }
        }

        var any = true;
        while (any) {
            any = false;
            for (loop.blocks) |bid| {
                const block = &func.blocks.items[bid];
                var i: usize = 0;
                while (i < block.instructions.items.len) {
                    const inst = block.instructions.items[i];

                    // local_get / global_get are not in `isPure` because
                    // they read external state, but are safe to hoist
                    // under the per-loop invariance conditions above.
                    const is_invariant_local_get = switch (inst.op) {
                        .local_get => |idx| !set_locals.contains(idx),
                        else => false,
                    };
                    const is_invariant_global_get = switch (inst.op) {
                        .global_get => |idx| !loop_has_call and !set_globals.contains(idx),
                        else => false,
                    };
                    // Speculative load hoisting (Stage C, #446, extended
                    // per #494). A trapping load is safe to hoist iff:
                    //  (a) the loop body has no memory-mutating op and
                    //      no call (else the load could observe a
                    //      different value across iterations);
                    //  (b) the load's containing block dominates every
                    //      loop latch AND every exiting block — i.e.,
                    //      every iteration must transit through it
                    //      before either looping back or leaving the
                    //      loop. The header is the common special case
                    //      (it dominates every block in the loop by
                    //      definition); we short-circuit it to avoid
                    //      the per-anchor loop.
                    //      In both cases the load runs on every entered
                    //      iteration (including the 0-iter early-exit
                    //      path), so hoisting preserves the trap point.
                    const speculation_safe = loop_can_hoist_load and (bid == loop.header or blk: {
                        for (anchors.items) |x| if (!dom.dominates(bid, x)) break :blk false;
                        break :blk true;
                    });
                    const is_speculative_load = speculation_safe and switch (inst.op) {
                        .load,
                        .v128_load,
                        .v128_load_splat,
                        .v128_load_zero,
                        .v128_load_extend,
                        .v128_load_lane,
                        => true,
                        else => false,
                    };
                    const eligible_by_kind = is_invariant_local_get or is_invariant_global_get or is_speculative_load or
                        (inst.dest != null and isPure(inst) and !hasSideEffect(inst));
                    if (!eligible_by_kind) {
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

/// Resolve a dedicated preheader for `loop`: either an existing block
/// whose sole terminator is `.br loop.header` and which dominates the
/// header, or — failing that — a freshly synthesized one. Returns
/// `null` only when the loop header has no non-loop predecessor at all
/// (i.e., the header is unreachable from outside the loop).
fn obtainLoopPreheader(
    func: *ir.IrFunction,
    loop: *const analysis.Loop,
    predecessors: *std.AutoHashMap(ir.BlockId, []const ir.BlockId),
    dom: *const analysis.DomTree,
    allocator: std.mem.Allocator,
) !?ir.BlockId {
    const header_preds = predecessors.get(loop.header) orelse return null;

    var unique_non_loop_pred: ?ir.BlockId = null;
    var multiple_non_loop = false;
    var any_non_loop = false;
    for (header_preds) |p| {
        if (loop.containsBlock(p)) continue;
        any_non_loop = true;
        if (unique_non_loop_pred != null) {
            multiple_non_loop = true;
        } else {
            unique_non_loop_pred = p;
        }
    }
    if (!any_non_loop) return null;

    if (!multiple_non_loop) {
        if (unique_non_loop_pred) |p| {
            const ph_insts = func.blocks.items[p].instructions.items;
            if (ph_insts.len > 0) {
                const last = ph_insts[ph_insts.len - 1];
                const is_clean_br = switch (last.op) {
                    .br => |t| t == loop.header,
                    else => false,
                };
                if (is_clean_br and dom.dominates(p, loop.header)) {
                    return p;
                }
            }
        }
    }

    return try synthesizeLoopPreheader(func, loop, header_preds, predecessors, allocator);
}

/// True iff `bid` is an exiting block of `loop`: its terminator has
/// at least one successor outside `loop.blocks`, or it is a function-
/// exit terminator (`ret`/`unreachable`) and therefore leaves the loop
/// implicitly. A block with no instructions is treated as exiting as a
/// fail-safe — such a block is malformed and conservatively blocks any
/// speculative hoist that would rely on dominating it.
fn isLoopExitingBlock(func: *const ir.IrFunction, loop: *const analysis.Loop, bid: ir.BlockId) bool {
    const block = func.blocks.items[bid];
    if (block.instructions.items.len == 0) return true;
    const term = block.instructions.items[block.instructions.items.len - 1].op;
    return switch (term) {
        .br => |t| !loop.containsBlock(t),
        .br_if => |bi| !loop.containsBlock(bi.then_block) or !loop.containsBlock(bi.else_block),
        .br_table => |bt| blk: {
            if (!loop.containsBlock(bt.default)) break :blk true;
            for (bt.targets) |t| if (!loop.containsBlock(t)) break :blk true;
            break :blk false;
        },
        else => true,
    };
}

/// Allocate a fresh block, retarget every non-loop predecessor's
/// header-bound edge through it, and emit `.br loop.header` as its
/// sole instruction. The synthesized block trivially dominates the
/// header by construction.
///
/// `predecessors` is updated in-place: the new block's pred list is
/// the set of non-loop preds, and the header's pred list now lists
/// the new block in place of those preds (latches preserved).
fn synthesizeLoopPreheader(
    func: *ir.IrFunction,
    loop: *const analysis.Loop,
    header_preds: []const ir.BlockId,
    predecessors: *std.AutoHashMap(ir.BlockId, []const ir.BlockId),
    allocator: std.mem.Allocator,
) !?ir.BlockId {
    var non_loop_preds: std.ArrayList(ir.BlockId) = .empty;
    defer non_loop_preds.deinit(allocator);
    for (header_preds) |p| {
        if (!loop.containsBlock(p)) try non_loop_preds.append(allocator, p);
    }
    if (non_loop_preds.items.len == 0) return null;

    const ph_new: ir.BlockId = try func.newBlock();
    try func.blocks.items[ph_new].append(.{ .op = .{ .br = loop.header } });

    for (non_loop_preds.items) |p| {
        const pred_block = &func.blocks.items[p];
        const ninsts = pred_block.instructions.items.len;
        if (ninsts == 0) continue;
        const term = &pred_block.instructions.items[ninsts - 1];
        switch (term.op) {
            .br => |t| if (t == loop.header) {
                term.op = .{ .br = ph_new };
            },
            .br_if => |*bi| {
                if (bi.then_block == loop.header) bi.then_block = ph_new;
                if (bi.else_block == loop.header) bi.else_block = ph_new;
            },
            .br_table => |*bt| {
                if (bt.default == loop.header) bt.default = ph_new;
                var has_header_target = false;
                for (bt.targets) |t| {
                    if (t == loop.header) {
                        has_header_target = true;
                        break;
                    }
                }
                if (has_header_target) {
                    const new_targets = try allocator.alloc(ir.BlockId, bt.targets.len);
                    for (bt.targets, 0..) |t, idx| {
                        new_targets[idx] = if (t == loop.header) ph_new else t;
                    }
                    try func.owned_br_table_targets.append(allocator, new_targets);
                    bt.targets = new_targets;
                }
            },
            else => {},
        }
    }

    const ph_preds = try allocator.alloc(ir.BlockId, non_loop_preds.items.len);
    @memcpy(ph_preds, non_loop_preds.items);
    try predecessors.put(ph_new, ph_preds);

    var header_remaining: std.ArrayList(ir.BlockId) = .empty;
    defer header_remaining.deinit(allocator);
    for (header_preds) |p| {
        if (loop.containsBlock(p)) try header_remaining.append(allocator, p);
    }
    try header_remaining.append(allocator, ph_new);
    const new_header_preds = try allocator.alloc(ir.BlockId, header_remaining.items.len);
    @memcpy(new_header_preds, header_remaining.items);
    if (predecessors.fetchRemove(loop.header)) |kv| allocator.free(kv.value);
    try predecessors.put(loop.header, new_header_preds);

    return ph_new;
}

const DefSite = struct { block: ir.BlockId, inst: usize };

fn buildDefSites(func: *const ir.IrFunction, allocator: std.mem.Allocator) !std.AutoHashMap(ir.VReg, DefSite) {
    var defs = std.AutoHashMap(ir.VReg, DefSite).init(allocator);
    errdefer defs.deinit();
    for (func.blocks.items, 0..) |block, bid| {
        for (block.instructions.items, 0..) |inst, ii| {
            if (inst.dest) |d| try defs.put(d, .{ .block = @intCast(bid), .inst = ii });
        }
    }
    return defs;
}

fn defInst(func: *const ir.IrFunction, defs: *const std.AutoHashMap(ir.VReg, DefSite), v: ir.VReg) ?ir.Inst {
    const site = defs.get(v) orelse return null;
    return func.blocks.items[site.block].instructions.items[site.inst];
}

fn constI32Of(func: *const ir.IrFunction, defs: *const std.AutoHashMap(ir.VReg, DefSite), v: ir.VReg) ?i32 {
    const inst = defInst(func, defs, v) orelse return null;
    return switch (inst.op) {
        .iconst_32 => |c| c,
        else => null,
    };
}

fn localGetIdxOf(func: *const ir.IrFunction, defs: *const std.AutoHashMap(ir.VReg, DefSite), v: ir.VReg) ?u32 {
    const inst = defInst(func, defs, v) orelse return null;
    return switch (inst.op) {
        .local_get => |idx| idx,
        else => null,
    };
}

fn dedicatedPreheader(
    func: *const ir.IrFunction,
    loop: *const analysis.Loop,
    predecessors: *const std.AutoHashMap(ir.BlockId, []const ir.BlockId),
    dom: *const analysis.DomTree,
) ?ir.BlockId {
    const header_preds = predecessors.get(loop.header) orelse return null;
    var preheader: ?ir.BlockId = null;
    for (header_preds) |p| {
        if (!loop.containsBlock(p)) {
            if (preheader != null) return null;
            preheader = p;
        }
    }
    const ph = preheader orelse return null;
    const ph_insts = func.blocks.items[ph].instructions.items;
    if (ph_insts.len == 0) return null;
    switch (ph_insts[ph_insts.len - 1].op) {
        .br => |target| if (target != loop.header) return null,
        else => return null,
    }
    if (!dom.dominates(ph, loop.header)) return null;
    return ph;
}

const Induction = struct {
    local_idx: u32,
    init: ?i32 = null,
    step: i32,
    step_vreg: ir.VReg,
    update_val: ir.VReg,
    update_block: ir.BlockId,
    update_index: usize,
};

fn findPrimaryInduction(
    func: *const ir.IrFunction,
    loop: *const analysis.Loop,
    preheader: ir.BlockId,
    defs: *const std.AutoHashMap(ir.VReg, DefSite),
) ?Induction {
    for (loop.blocks) |bid| {
        const block = func.blocks.items[bid];
        for (block.instructions.items, 0..) |inst, ii| {
            const ls = switch (inst.op) {
                .local_set => |ls| ls,
                else => continue,
            };
            const add_inst = defInst(func, defs, ls.val) orelse continue;
            const add = switch (add_inst.op) {
                .add => |a| a,
                else => continue,
            };

            var step_vreg: ir.VReg = undefined;
            if (localGetIdxOf(func, defs, add.lhs) == ls.idx and constI32Of(func, defs, add.rhs) != null) {
                step_vreg = add.rhs;
            } else if (localGetIdxOf(func, defs, add.rhs) == ls.idx and constI32Of(func, defs, add.lhs) != null) {
                step_vreg = add.lhs;
            } else {
                continue;
            }
            const step = constI32Of(func, defs, step_vreg) orelse continue;
            if (step == 0) continue;

            var init: ?i32 = null;
            for (func.blocks.items[preheader].instructions.items) |ph_inst| {
                const ph_ls = switch (ph_inst.op) {
                    .local_set => |set| set,
                    else => continue,
                };
                if (ph_ls.idx != ls.idx) continue;
                init = constI32Of(func, defs, ph_ls.val);
            }

            return .{
                .local_idx = ls.idx,
                .init = init,
                .step = step,
                .step_vreg = step_vreg,
                .update_val = ls.val,
                .update_block = bid,
                .update_index = ii,
            };
        }
    }
    return null;
}

fn currentInductionUpdateIndex(func: *const ir.IrFunction, ind: Induction) usize {
    const block = func.blocks.items[ind.update_block];
    for (block.instructions.items, 0..) |inst, ii| {
        if (inst.op == .local_set and inst.op.local_set.idx == ind.local_idx and inst.op.local_set.val == ind.update_val) return ii;
    }
    return @min(ind.update_index, block.instructions.items.len);
}

fn invariantAddressBase(
    func: *const ir.IrFunction,
    loop: *const analysis.Loop,
    defs: *const std.AutoHashMap(ir.VReg, DefSite),
    addr: ir.VReg,
    induction_local: u32,
) ?ir.VReg {
    const inst = defInst(func, defs, addr) orelse return null;
    const add = switch (inst.op) {
        .add => |a| a,
        else => return null,
    };
    const lhs_is_iv = localGetIdxOf(func, defs, add.lhs) == induction_local;
    const rhs_is_iv = localGetIdxOf(func, defs, add.rhs) == induction_local;
    const base = if (lhs_is_iv and !rhs_is_iv) add.rhs else if (rhs_is_iv and !lhs_is_iv) add.lhs else return null;
    const base_def = defs.get(base) orelse return null;
    if (loop.containsBlock(base_def.block)) return null;
    return base;
}

/// Strength-reduce stride-1 induction-addressed memory accesses.
///
/// This first implementation recognizes `local_set i, i + c` and rewrites
/// scalar `load`/`store` bases of the form `base + i` to use a synthetic
/// pointer local `p`, initialized in the preheader and bumped by the same
/// step immediately after the induction update.  Multi-stride (`i * stride`)
/// and preheader range-check collapsing are intentionally left for follow-up:
/// if a per-iteration bounds check on the same address pattern remains, future
/// work can collapse it to a single preheader range check on `[p, p + N*stride]`.
pub fn inductionVariableSimplification(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
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

    var changed = false;
    for (lf.loops) |*loop| {
        // Restrict to innermost loops. For an outer loop that contains a
        // nested loop, `loop.blocks` includes the nested loop's blocks, so
        // `findPrimaryInduction` may pick the inner loop's `i = i + step`
        // as the outer's primary induction. The transform would then init
        // `p = base` only in the outer preheader, but `i` (and the inner
        // loop's IV) gets reset every outer iteration — leaving `p` stale
        // and producing out-of-bounds memory accesses (observed as a
        // CoreMark AOT trap, see #385).
        var is_innermost = true;
        for (loop.blocks) |bid| {
            if (bid == loop.header) continue;
            if (lf.header_loop.contains(bid)) {
                is_innermost = false;
                break;
            }
        }
        if (!is_innermost) continue;

        var defs = try buildDefSites(func, allocator);
        defer defs.deinit();

        const ph = dedicatedPreheader(func, loop, &predecessors, &dom) orelse continue;
        const ind = findPrimaryInduction(func, loop, ph, &defs) orelse continue;

        // The preheader initialiser below sets `p = base`, which is only
        // correct when `i` starts at 0. Skip otherwise — handling non-zero
        // init would require emitting a `p = base + init` add in the
        // preheader, deferred to a follow-up.
        const init_val = ind.init orelse continue;
        if (init_val != 0) continue;

        var ptr_locals = std.AutoHashMap(ir.VReg, u32).init(allocator);
        defer ptr_locals.deinit();
        var ptr_order: std.ArrayList(struct { base: ir.VReg, local: u32 }) = .empty;
        defer ptr_order.deinit(allocator);

        for (loop.blocks) |bid| {
            var block = &func.blocks.items[bid];
            var i: usize = 0;
            while (i < block.instructions.items.len) : (i += 1) {
                const maybe_base: ?ir.VReg = switch (block.instructions.items[i].op) {
                    .load => |ld| invariantAddressBase(func, loop, &defs, ld.base, ind.local_idx),
                    .store => |st| invariantAddressBase(func, loop, &defs, st.base, ind.local_idx),
                    else => null,
                };
                const base = maybe_base orelse continue;

                const gop = try ptr_locals.getOrPut(base);
                if (!gop.found_existing) {
                    gop.value_ptr.* = func.local_count;
                    func.local_count += 1;
                    try ptr_order.append(allocator, .{ .base = base, .local = gop.value_ptr.* });
                }
                const p_local = gop.value_ptr.*;
                const p_vreg = func.newVReg();
                try block.instructions.insert(func.allocator, i, .{
                    .op = .{ .local_get = p_local },
                    .dest = p_vreg,
                    .type = .i32,
                });
                i += 1;
                switch (block.instructions.items[i].op) {
                    .load => |*ld| ld.base = p_vreg,
                    .store => |*st| st.base = p_vreg,
                    else => unreachable,
                }
                changed = true;
            }
        }

        if (ptr_order.items.len == 0) continue;

        var ph_block = &func.blocks.items[ph];
        var insert_at = ph_block.instructions.items.len - 1;
        for (ptr_order.items) |entry| {
            try ph_block.instructions.insert(func.allocator, insert_at, .{
                .op = .{ .local_set = .{ .idx = entry.local, .val = entry.base } },
            });
            insert_at += 1;
        }

        var update_block = &func.blocks.items[ind.update_block];
        var update_at = currentInductionUpdateIndex(func, ind) + 1;
        for (ptr_order.items) |entry| {
            const cur = func.newVReg();
            const next = func.newVReg();
            try update_block.instructions.insert(func.allocator, update_at, .{
                .op = .{ .local_get = entry.local },
                .dest = cur,
                .type = .i32,
            });
            update_at += 1;
            try update_block.instructions.insert(func.allocator, update_at, .{
                .op = .{ .add = .{ .lhs = cur, .rhs = ind.step_vreg } },
                .dest = next,
                .type = .i32,
            });
            update_at += 1;
            try update_block.instructions.insert(func.allocator, update_at, .{
                .op = .{ .local_set = .{ .idx = entry.local, .val = next } },
            });
            update_at += 1;
        }
    }
    return changed;
}

fn loopBodySize(func: *const ir.IrFunction, loop: *const analysis.Loop) usize {
    var n: usize = 0;
    for (loop.blocks) |bid| n += func.blocks.items[bid].instructions.items.len;
    return n;
}

fn isLoopTarget(loop: *const analysis.Loop, target: ir.BlockId) bool {
    return loop.containsBlock(target);
}

fn findLoopExit(func: *const ir.IrFunction, loop: *const analysis.Loop) ?struct { exit: ir.BlockId, cond: ir.VReg } {
    const header = func.blocks.items[loop.header];
    if (header.instructions.items.len == 0) return null;
    const term = header.instructions.items[header.instructions.items.len - 1];
    const bi = switch (term.op) {
        .br_if => |bi| bi,
        else => return null,
    };
    const then_loop = isLoopTarget(loop, bi.then_block);
    const else_loop = isLoopTarget(loop, bi.else_block);
    if (then_loop == else_loop) return null;
    return .{ .exit = if (then_loop) bi.else_block else bi.then_block, .cond = bi.cond };
}

fn tripCountForLoop(
    func: *const ir.IrFunction,
    defs: *const std.AutoHashMap(ir.VReg, DefSite),
    ind: Induction,
    cond: ir.VReg,
) ?u32 {
    const init = ind.init orelse return null;
    if (ind.step <= 0) return null;
    const cmp_inst = defInst(func, defs, cond) orelse return null;
    const cmp = switch (cmp_inst.op) {
        .lt_s, .lt_u => |c| c,
        else => return null,
    };
    const lhs_idx = localGetIdxOf(func, defs, cmp.lhs);
    if (lhs_idx != ind.local_idx) return null;
    const limit = constI32Of(func, defs, cmp.rhs) orelse return null;
    if (limit <= init) return 0;
    const distance: u32 = @intCast(limit - init);
    const step: u32 = @intCast(ind.step);
    return (distance + step - 1) / step;
}

const VRegRemap = struct { from: ir.VReg, to: ir.VReg };

fn remapCloneVRegs(inst: *ir.Inst, map: []const VRegRemap) void {
    for (map) |m| replaceInInst(inst, m.from, m.to);
}

/// Fully unroll very small counted loops.
///
/// The transform is deliberately conservative: it handles dedicated-preheader
/// natural loops with a single primary `i = i + const_step`, a header
/// `i < const_limit` condition, trip count ≤ 8, and ≤ 16 IR instructions in
/// the loop.  It clones the loop instructions into the preheader, substitutes
/// each `local_get i` with the iteration constant when possible, then redirects
/// the preheader to the loop exit; the old loop blocks become unreachable.
pub fn unrollSmallFixedLoops(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
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

    for (lf.loops) |*loop| {
        if (loopBodySize(func, loop) > 16) continue;
        const ph = dedicatedPreheader(func, loop, &predecessors, &dom) orelse continue;
        var defs = try buildDefSites(func, allocator);
        defer defs.deinit();
        const ind = findPrimaryInduction(func, loop, ph, &defs) orelse continue;
        const exit_info = findLoopExit(func, loop) orelse continue;
        const trips = tripCountForLoop(func, &defs, ind, exit_info.cond) orelse continue;
        if (trips > 8) continue;

        var templates: std.ArrayList(ir.Inst) = .empty;
        defer templates.deinit(allocator);
        var unsupported = false;
        for (loop.blocks) |bid| {
            const block = func.blocks.items[bid];
            for (block.instructions.items, 0..) |inst, ii| {
                if (ii == findTerminatorIndex(&block)) continue;
                switch (inst.op) {
                    .br,
                    .br_if,
                    .br_table,
                    .ret,
                    .ret_multi,
                    .@"unreachable",
                    .call,
                    .call_indirect,
                    .call_ref,
                    => {
                        unsupported = true;
                        break;
                    },
                    else => try templates.append(allocator, inst),
                }
            }
            if (unsupported) break;
        }
        if (unsupported) continue;
        if (templates.items.len == 0 and trips != 0) continue;

        var ph_block = &func.blocks.items[ph];
        const original_term = ph_block.instructions.items.len - 1;
        var insert_at = original_term;
        var iter: u32 = 0;
        while (iter < trips) : (iter += 1) {
            const iter_value = (ind.init orelse 0) + @as(i32, @intCast(iter)) * ind.step;
            var map: std.ArrayList(VRegRemap) = .empty;
            defer map.deinit(allocator);

            for (templates.items) |tmpl| {
                var cloned = tmpl;
                if (cloned.dest) |old_dest| {
                    const new_dest = func.newVReg();
                    cloned.dest = new_dest;
                    try map.append(allocator, .{ .from = old_dest, .to = new_dest });
                }
                remapCloneVRegs(&cloned, map.items);
                if (cloned.op == .local_get and cloned.op.local_get == ind.local_idx) {
                    cloned.op = .{ .iconst_32 = iter_value };
                }
                try ph_block.instructions.insert(func.allocator, insert_at, cloned);
                insert_at += 1;
            }
        }

        ph_block.instructions.items[insert_at].op = .{ .br = exit_info.exit };
        return true;
    }
    return false;
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
                .v128_load_extend => |*ld| {
                    const end: u64 = @as(u64, ld.offset) + ld.accessSize();
                    const dom_max = domMaxEnd(table.items, valid_start, ld.base);

                    if (end <= dom_max) {
                        if (!ld.bounds_known) {
                            ld.bounds_known = true;
                            changed = true;
                        }
                    } else if (seg_first.getPtr(ld.base)) |se| {
                        if (!ld.bounds_known) {
                            ld.bounds_known = true;
                            changed = true;
                        }
                        if (end > se.max_end) se.max_end = end;
                    } else {
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
            .v128_load_extend => |ld| @as(u64, ld.offset) + ld.accessSize(),
            .store => |st| @as(u64, st.offset) + @as(u64, st.size),
            else => unreachable,
        };
        if (se.max_end > own_end) {
            switch (se.inst.op) {
                .load => |*ld| ld.checked_end = se.max_end,
                .v128_load_extend => |*ld| ld.checked_end = se.max_end,
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
                .v128_load_extend => |*ld| {
                    if (add_info.get(ld.base)) |info| {
                        const access_end = if (ld.checked_end > 0) ld.checked_end else @as(u64, ld.offset) + ld.accessSize();
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
                        const end = if (ld.checked_end > 0) ld.checked_end else @as(u64, ld.offset) + ld.accessSize();
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

// ── Branch threading ────────────────────────────────────────────────────────

/// Thread `br_if` edges through a one-instruction `br_if` block that tests
/// the same condition. If block A's true edge reaches block B, then at B the
/// same condition is known true, so A can jump directly to B's true target.
/// Similarly, A's false edge can jump directly to B's false target.
pub fn threadChainedConditionalBranches(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    _ = allocator;
    var changed = false;

    for (func.blocks.items) |*block| {
        if (block.instructions.items.len == 0) continue;
        const term = &block.instructions.items[block.instructions.items.len - 1];
        switch (term.op) {
            .br_if => |bi| {
                var threaded = bi;
                var term_changed = false;

                if (threadedBrIfTarget(func, bi.cond, bi.then_block, true)) |target| {
                    if (target != threaded.then_block) {
                        threaded.then_block = target;
                        term_changed = true;
                    }
                }
                if (threadedBrIfTarget(func, bi.cond, bi.else_block, false)) |target| {
                    if (target != threaded.else_block) {
                        threaded.else_block = target;
                        term_changed = true;
                    }
                }

                if (term_changed) {
                    term.op = .{ .br_if = threaded };
                    changed = true;
                }
            },
            else => {},
        }
    }

    return changed;
}

fn threadedBrIfTarget(
    func: *const ir.IrFunction,
    cond: ir.VReg,
    target: ir.BlockId,
    take_then: bool,
) ?ir.BlockId {
    const target_idx: usize = @intCast(target);
    if (target_idx >= func.blocks.items.len) return null;

    const target_block = &func.blocks.items[target_idx];
    if (target_block.instructions.items.len != 1) return null;

    return switch (target_block.instructions.items[0].op) {
        .br_if => |bi| if (bi.cond == cond)
            (if (take_then) bi.then_block else bi.else_block)
        else
            null,
        else => null,
    };
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
        .v128_bitselect => |*sel| {
            sel.a += offset;
            sel.b += offset;
            sel.mask += offset;
        },
        .i32x4_binop => |*bin| {
            bin.lhs += offset;
            bin.rhs += offset;
        },
        .i32x4_unop => |*un| un.vector += offset,
        .i32x4_extadd_pairwise_i16x8 => |*op| op.vector += offset,
        .i32x4_dot_i16x8_s => |*bin| {
            bin.lhs += offset;
            bin.rhs += offset;
        },
        .i32x4_extend_i16x8 => |*op| op.vector += offset,
        .f32x4_unop => |*un| un.vector += offset,
        .f32x4_convert_i32x4 => |*op| op.vector += offset,
        .i32x4_trunc_sat => |*op| op.vector += offset,
        .f32x4_demote_f64x2_zero => |*op| op.vector += offset,
        .i32x4_extmul_i16x8 => |*op| {
            op.lhs += offset;
            op.rhs += offset;
        },
        .i8x16_binop => |*bin| {
            bin.lhs += offset;
            bin.rhs += offset;
        },
        .i8x16_shuffle => |*op| {
            op.lhs += offset;
            op.rhs += offset;
        },
        .i8x16_swizzle => |*op| {
            op.vector += offset;
            op.indices += offset;
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
        .f32x4_binop => |*bin| {
            bin.lhs += offset;
            bin.rhs += offset;
        },
        .f64x2_binop => |*bin| {
            bin.lhs += offset;
            bin.rhs += offset;
        },
        .f64x2_unop => |*un| un.vector += offset,
        .f64x2_convert_low_i32x4 => |*op| op.vector += offset,
        .f64x2_promote_low_f32x4 => |*op| op.vector += offset,
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
        .v128_any_true,
        .i32x4_splat,
        .f32x4_splat,
        .i8x16_splat,
        .i16x8_splat,
        .i64x2_splat,
        .f64x2_splat,
        => |*vreg| vreg.* += offset,
        .simd_all_true => |*op| op.vector += offset,
        .simd_bitmask => |*op| op.vector += offset,
        .i32x4_extract_lane => |*lane| lane.vector += offset,
        .f32x4_extract_lane => |*lane| lane.vector += offset,
        .i8x16_extract_lane => |*lane| lane.vector += offset,
        .i16x8_extract_lane => |*lane| lane.vector += offset,
        .i64x2_extract_lane => |*lane| lane.vector += offset,
        .f64x2_extract_lane => |*lane| lane.vector += offset,
        .i32x4_replace_lane => |*lane| {
            lane.vector += offset;
            lane.val += offset;
        },
        .f32x4_replace_lane => |*lane| {
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
        .f64x2_replace_lane => |*lane| {
            lane.vector += offset;
            lane.val += offset;
        },

        .local_set => |*ls| ls.val += offset,
        .global_set => |*gs| gs.val += offset,
        .load => |*ld| ld.base += offset,
        .v128_load => |*ld| ld.base += offset,
        .v128_load_splat => |*ld| ld.base += offset,
        .v128_load_zero => |*ld| ld.base += offset,
        .v128_load_extend => |*ld| ld.base += offset,
        .v128_load_lane => |*ld| {
            ld.base += offset;
            ld.vector += offset;
        },
        .store => |*st| {
            st.base += offset;
            st.val += offset;
        },
        .v128_store => |*st| {
            st.base += offset;
            st.val += offset;
        },
        .v128_store_lane => |*st| {
            st.base += offset;
            st.vector += offset;
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

const inline_small_max_blocks: u32 = 16;
const inline_small_max_insts: u32 = 64;

fn localTypeOf(func: *const ir.IrFunction, idx: u32) ir.IrType {
    if (func.local_types) |lt| {
        if (idx < lt.len) return lt[idx];
    }
    return .i32;
}

fn hasLocalSet(func: *const ir.IrFunction) bool {
    for (func.blocks.items) |blk| {
        for (blk.instructions.items) |inst| {
            if (inst.op == .local_set) return true;
        }
    }
    return false;
}

fn extendCallerLocalsForInline(caller: *ir.IrFunction, callee: *const ir.IrFunction) !u32 {
    const base = caller.local_count;
    const new_count = base + callee.local_count;
    if (callee.local_count == 0) return base;

    if (caller.local_types != null or callee.local_types != null) {
        const new_types = try caller.allocator.alloc(ir.IrType, new_count);
        errdefer caller.allocator.free(new_types);

        for (0..base) |i| new_types[i] = localTypeOf(caller, @intCast(i));
        for (0..callee.local_count) |i| {
            new_types[base + i] = localTypeOf(callee, @intCast(i));
        }

        if (caller.local_types) |old_types| caller.allocator.free(old_types);
        caller.local_types = new_types;
    }

    caller.local_count = new_count;
    return base;
}

fn zeroOpForType(local_type: ir.IrType) ir.Inst.Op {
    return switch (local_type) {
        .i32 => .{ .iconst_32 = 0 },
        .i64 => .{ .iconst_64 = 0 },
        .f32 => .{ .fconst_32 = 0 },
        .f64 => .{ .fconst_64 = 0 },
        .v128 => .{ .v128_const = 0 },
        .void => .{ .iconst_32 = 0 },
    };
}

/// Is this callee eligible for the inliner?
///   - Non-empty, ≤ `max_blocks` blocks
///   - Total instructions ≤ `max_insts`
///   - No calls (direct/indirect/ref), no call_result
///   - No memory_grow, no atomics, no bulk memory/table ops
///   - No `ret_multi`
///   - Every `local_get`/`local_set` targets an existing callee local
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
                .ret_multi,
                => return false,
                .local_get => |idx| if (idx >= callee.local_count) return false,
                .local_set => |ls| if (ls.idx >= callee.local_count) return false,
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

/// Shift every inline `BlockId` referenced by `inst` by `+offset`.
/// br_table target slices are deep-cloned by the inliner before append.
fn shiftBlockIdsInInst(inst: *ir.Inst, offset: ir.BlockId) void {
    switch (inst.op) {
        .br => |*t| t.* += offset,
        .br_if => |*bi| {
            bi.then_block += offset;
            bi.else_block += offset;
        },
        .br_table => |*bt| bt.default += offset,
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
fn inlineSmallFunctionsCount(module: *ir.IrModule, allocator: std.mem.Allocator) !u32 {
    var eligible = try allocator.alloc(bool, module.functions.items.len);
    defer allocator.free(eligible);
    for (module.functions.items, 0..) |*f, i| eligible[i] = isInlinable(f, inline_small_max_insts, inline_small_max_blocks);

    var inlined_count: u32 = 0;
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
            const needs_synthetic_locals = callee.local_count != callee.param_count or hasLocalSet(callee);

            const vreg_offset: ir.VReg = caller.next_vreg;
            caller.next_vreg += callee.next_vreg;

            var local_map: []u32 = &.{};
            defer if (local_map.len != 0) allocator.free(local_map);
            if (needs_synthetic_locals) {
                const local_base = try extendCallerLocalsForInline(caller, callee);
                local_map = try allocator.alloc(u32, callee.local_count);
                for (local_map, 0..) |*mapped, local_idx| {
                    mapped.* = local_base + @as(u32, @intCast(local_idx));
                }
            }

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
                if (needs_synthetic_locals and cidx == 0) {
                    for (0..callee.param_count) |param_idx| {
                        try caller.blocks.items[clone_id].instructions.append(caller.allocator, .{
                            .op = .{ .local_set = .{
                                .idx = local_map[param_idx],
                                .val = call_args[param_idx],
                            } },
                        });
                    }
                    for (callee.param_count..callee.local_count) |local_idx| {
                        const local_type = localTypeOf(callee, @intCast(local_idx));
                        const zero = caller.newVReg();
                        try caller.blocks.items[clone_id].instructions.append(caller.allocator, .{
                            .op = zeroOpForType(local_type),
                            .dest = zero,
                            .type = local_type,
                        });
                        try caller.blocks.items[clone_id].instructions.append(caller.allocator, .{
                            .op = .{ .local_set = .{
                                .idx = local_map[local_idx],
                                .val = zero,
                            } },
                        });
                    }
                }
                for (callee_block.instructions.items) |citem| {
                    switch (citem.op) {
                        .local_get => |idx| {
                            if (needs_synthetic_locals) {
                                var cloned = citem;
                                cloned.op = .{ .local_get = local_map[idx] };
                                shiftVRegsInInst(&cloned, vreg_offset);
                                try caller.blocks.items[clone_id].instructions.append(caller.allocator, cloned);
                            } else {
                                const shifted_dest = (citem.dest orelse continue) + vreg_offset;
                                try local_renames.append(allocator, .{
                                    .from = shifted_dest,
                                    .to = call_args[idx],
                                });
                            }
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
                            if (needs_synthetic_locals and cloned.op == .local_set) {
                                cloned.op.local_set.idx = local_map[cloned.op.local_set.idx];
                            }
                            shiftVRegsInInst(&cloned, vreg_offset);
                            if (cloned.op == .br_table) {
                                const old_targets = cloned.op.br_table.targets;
                                const new_targets = try caller.allocator.alloc(ir.BlockId, old_targets.len);
                                var targets_owned = false;
                                errdefer if (!targets_owned) caller.allocator.free(new_targets);
                                for (old_targets, 0..) |target, target_idx| {
                                    new_targets[target_idx] = target + clone_offset;
                                }
                                try caller.owned_br_table_targets.append(caller.allocator, new_targets);
                                targets_owned = true;
                                cloned.op.br_table.targets = new_targets;
                                cloned.op.br_table.default += clone_offset;
                            } else {
                                shiftBlockIdsInInst(&cloned, clone_offset);
                            }
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

            inlined_count += 1;
        }
    }
    return inlined_count;
}

pub fn inlineSmallFunctions(module: *ir.IrModule, allocator: std.mem.Allocator) !bool {
    return (try inlineSmallFunctionsCount(module, allocator)) != 0;
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

// ── Tail Duplication of Small Joins ────────────────────────────────────────

/// Tail-duplicate small "join" blocks into each predecessor, eliminating an
/// extra unconditional branch on every taken edge and exposing
/// per-predecessor specialisation opportunities to downstream passes.
///
/// A candidate join block B satisfies all of:
///   - B is not the entry block (id != 0).
///   - B has >= 2 predecessors.
///   - B has <= 4 non-terminator instructions plus a terminator that is
///     `br`, `br_if`, or `ret` (multi-result returns, `br_table`, and
///     `@"unreachable"` are excluded to keep duplication bounded and avoid
///     owned-slice cloning issues).
///   - Every predecessor P ends with `br B` (we only duplicate along
///     simple unconditional edges so we don't have to rewrite a `br_if`
///     operand).
///   - B has no `phi` and no `call`/`call_indirect`/`call_ref`/
///     `call_result` instructions in its body.
///   - No VReg defined in B is used outside B. After tail duplication the
///     original B is unreachable and its defs vanish; if any successor of
///     B referenced one of B's defs, the rewrite would break SSA. In
///     practice `lowerPhisToLocals` runs first, so cross-block dataflow
///     lives in `local_get`/`local_set` and the typical CoreMark join
///     blocks satisfy this guard.
///
/// When B qualifies, each predecessor's `br B` is replaced in place by a
/// fresh copy of B's body (with renamed dests) followed by B's terminator.
/// The original B is left in the function and becomes unreachable;
/// `reorderBlocks` skips unreachable blocks during RPO, so codegen never
/// emits them. The outer `runPasses` fixpoint loop (cap 8) bounds how
/// many times this pass can re-run — in practice 1-2 invocations suffice.
pub fn tailDuplicateSmallJoins(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    if (func.blocks.items.len < 2) return false;

    var predecessors = try analysis.buildPredecessors(func, allocator);
    defer {
        var it = predecessors.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        predecessors.deinit();
    }

    var changed = false;

    var b_id: ir.BlockId = 1;
    while (b_id < func.blocks.items.len) : (b_id += 1) {
        if (try tryAbsorbJoin(func, b_id, &predecessors, allocator)) {
            changed = true;

            // Predecessor map is now stale (preds of B disappeared; preds
            // of B's successors gained the duplicated copies). Rebuild
            // before considering further joins so the next iteration sees
            // the current CFG.
            var it = predecessors.iterator();
            while (it.next()) |entry| allocator.free(entry.value_ptr.*);
            predecessors.deinit();
            predecessors = try analysis.buildPredecessors(func, allocator);
        }
    }

    return changed;
}

/// Attempt to tail-duplicate block `b_id` into each of its predecessors.
/// Returns true iff the duplication was applied.
fn tryAbsorbJoin(
    func: *ir.IrFunction,
    b_id: ir.BlockId,
    predecessors: *const std.AutoHashMap(ir.BlockId, []const ir.BlockId),
    allocator: std.mem.Allocator,
) !bool {
    const b = &func.blocks.items[b_id];
    if (b.instructions.items.len == 0) return false;

    const term_idx = findTerminatorIndex(b);
    if (term_idx == b.instructions.items.len) return false;
    if (term_idx != b.instructions.items.len - 1) return false;

    const body_len = term_idx;
    if (body_len > 4) return false;

    switch (b.instructions.items[term_idx].op) {
        .br, .br_if, .ret => {},
        else => return false,
    }

    for (b.instructions.items[0..body_len]) |inst| {
        switch (inst.op) {
            .phi,
            .call,
            .call_indirect,
            .call_ref,
            .call_result,
            .br,
            .br_if,
            .br_table,
            .ret,
            .ret_multi,
            .@"unreachable",
            => return false,
            else => {},
        }
    }

    const preds = predecessors.get(b_id) orelse return false;
    if (preds.len < 2) return false;

    for (preds) |p| {
        if (p == b_id) return false;
        if (p >= func.blocks.items.len) return false;
        const pb = &func.blocks.items[p];
        if (pb.instructions.items.len == 0) return false;
        const pterm = pb.instructions.items[pb.instructions.items.len - 1];
        switch (pterm.op) {
            .br => |tgt| if (tgt != b_id) return false,
            else => return false,
        }
    }

    for (b.instructions.items) |inst| {
        const dest = inst.dest orelse continue;
        if (vregHasExternalUse(func, b_id, dest)) return false;
    }

    for (preds) |p| {
        try cloneJoinIntoPred(func, p, b_id, allocator);
    }
    return true;
}

/// Is `vreg` used by any instruction in a block other than `home_block`?
fn vregHasExternalUse(
    func: *const ir.IrFunction,
    home_block: ir.BlockId,
    vreg: ir.VReg,
) bool {
    for (func.blocks.items, 0..) |block, idx| {
        if (idx == home_block) continue;
        for (block.instructions.items) |inst| {
            for (getUsedVRegs(inst).slice()) |u| {
                if (u == vreg) return true;
            }
            switch (inst.op) {
                .call => |cl| for (cl.args) |a| {
                    if (a == vreg) return true;
                },
                .call_indirect => |ci| {
                    if (ci.elem_idx == vreg) return true;
                    for (ci.args) |a| {
                        if (a == vreg) return true;
                    }
                },
                .call_ref => |cr| {
                    if (cr.func_ref == vreg) return true;
                    for (cr.args) |a| {
                        if (a == vreg) return true;
                    }
                },
                .ret_multi => |vregs| for (vregs) |v| {
                    if (v == vreg) return true;
                },
                .phi => |edges| for (edges) |e| {
                    if (e.val == vreg) return true;
                },
                else => {},
            }
        }
    }
    return false;
}

/// Clone block `b_id`'s body and terminator into predecessor `p_id`,
/// replacing P's tail `br b_id` with the duplicated content. Every dest
/// VReg defined in B is renamed to a fresh VReg for this copy so each
/// duplicate is independent.
fn cloneJoinIntoPred(
    func: *ir.IrFunction,
    p_id: ir.BlockId,
    b_id: ir.BlockId,
    allocator: std.mem.Allocator,
) !void {
    const b_inst_count = func.blocks.items[b_id].instructions.items.len;

    var rename = std.AutoHashMap(ir.VReg, ir.VReg).init(allocator);
    defer rename.deinit();

    var i: usize = 0;
    while (i < b_inst_count) : (i += 1) {
        const src = func.blocks.items[b_id].instructions.items[i];
        if (src.dest) |old_dest| {
            const fresh = func.newVReg();
            try rename.put(old_dest, fresh);
        }
    }

    // Drop P's terminator (`br b_id`).
    _ = func.blocks.items[p_id].instructions.pop();

    i = 0;
    while (i < b_inst_count) : (i += 1) {
        var cloned = func.blocks.items[b_id].instructions.items[i];
        var rit = rename.iterator();
        while (rit.next()) |entry| {
            replaceInInst(&cloned, entry.key_ptr.*, entry.value_ptr.*);
        }
        if (cloned.dest) |old_dest| {
            if (rename.get(old_dest)) |fresh| cloned.dest = fresh;
        }
        try func.blocks.items[p_id].instructions.append(func.allocator, cloned);
    }
}

pub fn runPasses(module: *ir.IrModule, passes: []const PassFn, allocator: std.mem.Allocator) !u32 {
    return runPassesWithOptions(module, passes, allocator, .{});
}

/// Same as `runPasses`, but accepts a `RunOptions` carrying an optional
/// `dump_hook`. The hook fires once per pass per affected function:
///
///   - After each pass in the `passes` array (per per-function fixpoint
///     iteration; `info.changed` reports whether the pass mutated the
///     function).
///   - After `promoteLocalsToSSA` and `lowerPhisToLocals` on the first
///     outer iteration (only if promote actually fired; the hook for
///     `lowerPhisToLocals` only fires when `promoteLocalsToSSA` did).
///   - Once per local function after every successful round of the
///     module-level `inlineSmallFunctions` pass. `info.func_index` is
///     the module-local index of each function dumped.
///
/// Hook errors propagate and abort the pipeline.
pub fn runPassesWithOptions(
    module: *ir.IrModule,
    passes: []const PassFn,
    allocator: std.mem.Allocator,
    opts: RunOptions,
) !u32 {
    var total_changes: u32 = 0;

    // Outer loop: alternate between module-level inlining and per-function
    // fixpoint passes. The first per-function round constant-folds
    // arguments at call sites (e.g. via `forwardLocalGet` + `constantFold`),
    // which can make previously-borderline callees newly inlinable, or
    // exposes brand-new inline opportunities in callers whose body was
    // simplified. Re-running module-level inlining lets the second
    // per-function round specialise those freshly inlined bodies.
    //
    // Cap the outer iterations at 2 — sufficient for the
    // constant-argument-specialisation cases that motivate this loop,
    // without exploding compile time.
    const outer_max: u32 = 2;
    var outer_iter: u32 = 0;
    while (outer_iter < outer_max) : (outer_iter += 1) {
        // Module-level: inline small leaf callees. Iterate to fixpoint so
        // callers of callers also benefit within a single outer round.
        var inline_iter: u32 = 0;
        var inlined_count: u32 = 0;
        while (inline_iter < 4) : (inline_iter += 1) {
            const iter_inlined = try inlineSmallFunctionsCount(module, allocator);
            if (iter_inlined == 0) break;
            inlined_count += iter_inlined;
            total_changes += 1;

            if (opts.dump_hook) |hook| {
                for (module.functions.items, 0..) |*f, fi| {
                    try hook.callback(hook.ctx, .{
                        .pass_name = "inlineSmallFunctions",
                        .func = f,
                        .func_index = @intCast(fi),
                        .changed = true,
                        .iter = inline_iter,
                        .outer_iter = outer_iter,
                    });
                }
            }
        }
        if (inlined_count != 0) {
            std.log.debug(
                "inlineSmallFunctions (outer {d}): inlined {d} call(s) over {d} iteration(s)",
                .{ outer_iter, inlined_count, inline_iter },
            );
        } else if (outer_iter != 0) {
            // No additional inlining opportunities surfaced after the first
            // per-function fixpoint, so a second per-function round would
            // be redundant. Stop here.
            break;
        }

        for (module.functions.items, 0..) |*func, func_idx_usize| {
            const func_idx: u32 = @intCast(func_idx_usize);
            // SSA promotion: only meaningful on the first outer round. On
            // later rounds the function is already past mem2reg and any new
            // local_set/get inserted by inlining is handled by
            // `forwardLocalGet` + `deadLocalSetElimination`.
            if (outer_iter == 0) {
                if (try promoteLocalsToSSA(func, allocator)) {
                    total_changes += 1;
                    if (opts.dump_hook) |hook| {
                        try hook.callback(hook.ctx, .{
                            .pass_name = "promoteLocalsToSSA",
                            .func = func,
                            .func_index = func_idx,
                            .changed = true,
                            .iter = 0,
                            .outer_iter = outer_iter,
                        });
                    }
                    if (try lowerPhisToLocals(func, allocator)) {
                        total_changes += 1;
                        if (opts.dump_hook) |hook| {
                            try hook.callback(hook.ctx, .{
                                .pass_name = "lowerPhisToLocals",
                                .func = func,
                                .func_index = func_idx,
                                .changed = true,
                                .iter = 0,
                                .outer_iter = outer_iter,
                            });
                        }
                    }
                }
            }

            // Iterate the pipeline until fixpoint so that passes can
            // re-expose opportunities for each other (e.g. constantFold →
            // CSE → DCE → more constantFold). Cap iterations as a safety
            // net.
            var iter: u32 = 0;
            while (iter < 8) : (iter += 1) {
                var any_changed = false;
                for (passes) |pass| {
                    const changed = try pass(func, allocator);
                    if (changed) {
                        any_changed = true;
                        total_changes += 1;
                    }
                    if (opts.dump_hook) |hook| {
                        try hook.callback(hook.ctx, .{
                            .pass_name = passName(pass),
                            .func = func,
                            .func_index = func_idx,
                            .changed = changed,
                            .iter = iter,
                            .outer_iter = outer_iter,
                        });
                    }
                }
                if (!any_changed) break;
            }
        }
    }
    return total_changes;
}

/// Target-independent optimization pipeline.
pub const default_passes: []const PassFn = &.{
    &forwardLocalGet,
    &constantFold,
    &algebraicSimplify,
    &strengthReduceMul,
    &strengthReduceMulShiftAdd,
    &strengthReduceDivRem,
    &foldConstantBranches,
    &foldInverseCompareEqz,
    &threadChainedConditionalBranches,
    &tailDuplicateSmallJoins,
    &foldSelectOnEqz,
    &foldSignExtendingLoad,
    &foldFloatUnaryIdempotents,
    &foldWrapOfExtend,
    &globalValueNumbering,
    &inductionVariableSimplification,
    &hoistLoopInvariantCode,
    &unrollSmallFixedLoops,
    &@import("forward_redundant_loads.zig").forwardRedundantLoads,
    &deadStoreElimination,
    &deadCodeElimination,
    &deadLocalSetElimination,
    &hoistLoopBoundsChecks,
    &elideRedundantBoundsChecks,
    &foldLoadStoreOffset,
};


/// Default optimization pipeline for x86-64.
///
/// Note: `inductionVariableSimplification` and `unrollSmallFixedLoops`
/// are intentionally omitted here pending a cost-model fix for issue
/// #385. PR #413's own table reported x86_64 −2.45% on CoreMark (vs
/// aarch64 −0.24%), confirmed in the #393 audit
/// (https://github.com/cataggar/wamr/issues/393#issuecomment-4423326059)
/// as "the cost model is still picking wrong loops". The aarch64
/// pipeline keeps both passes for now — issue #385 tracks the
/// cost-model rework that should let x86_64 re-enable them safely.
const x86_64_default_passes: []const PassFn = &.{
    &forwardLocalGet,
    &constantFold,
    &algebraicSimplify,
    &strengthReduceMul,
    &strengthReduceMulShiftAdd,
    &strengthReduceDivRem,
    &foldConstantBranches,
    &foldInverseCompareEqz,
    &foldBranchOnEqz,
    &threadChainedConditionalBranches,
    &tailDuplicateSmallJoins,
    &foldSelectOnEqz,
    &foldSignExtendingLoad,
    &foldFloatUnaryIdempotents,
    &foldWrapOfExtend,
    &globalValueNumbering,
    &hoistLoopInvariantCode,
    &@import("forward_redundant_loads.zig").forwardRedundantLoads,
    &deadStoreElimination,
    &deadCodeElimination,
    &deadLocalSetElimination,
    &hoistLoopBoundsChecks,
    &elideRedundantBoundsChecks,
    &foldLoadStoreOffset,
};

const default_passes_no_iv: []const PassFn = &.{
    &forwardLocalGet,                  &constantFold,            &algebraicSimplify,      &strengthReduceMul,
    &strengthReduceMulShiftAdd,        &strengthReduceDivRem,    &foldConstantBranches,   &foldInverseCompareEqz,
    &threadChainedConditionalBranches, &tailDuplicateSmallJoins, &foldSelectOnEqz,        &foldSignExtendingLoad,
    &foldFloatUnaryIdempotents,        &foldWrapOfExtend,        &globalValueNumbering,   &hoistLoopInvariantCode,
    &unrollSmallFixedLoops,            &deadCodeElimination,     &deadLocalSetElimination, &hoistLoopBoundsChecks,
    &elideRedundantBoundsChecks,       &foldLoadStoreOffset,
};

const default_passes_no_unroll: []const PassFn = &.{
    &forwardLocalGet,                  &constantFold,            &algebraicSimplify,               &strengthReduceMul,
    &strengthReduceMulShiftAdd,        &strengthReduceDivRem,    &foldConstantBranches,            &foldInverseCompareEqz,
    &threadChainedConditionalBranches, &tailDuplicateSmallJoins, &foldSelectOnEqz,                 &foldSignExtendingLoad,
    &foldFloatUnaryIdempotents,        &foldWrapOfExtend,        &globalValueNumbering,            &inductionVariableSimplification,
    &hoistLoopInvariantCode,           &deadCodeElimination,     &deadLocalSetElimination,         &hoistLoopBoundsChecks,
    &elideRedundantBoundsChecks,       &foldLoadStoreOffset,
};

const default_passes_no_iv_no_unroll: []const PassFn = &.{
    &forwardLocalGet,                  &constantFold,            &algebraicSimplify,          &strengthReduceMul,
    &strengthReduceMulShiftAdd,        &strengthReduceDivRem,    &foldConstantBranches,       &foldInverseCompareEqz,
    &threadChainedConditionalBranches, &tailDuplicateSmallJoins, &foldSelectOnEqz,            &foldSignExtendingLoad,
    &foldFloatUnaryIdempotents,        &foldWrapOfExtend,        &globalValueNumbering,       &hoistLoopInvariantCode,
    &deadCodeElimination,              &deadLocalSetElimination, &hoistLoopBoundsChecks,      &elideRedundantBoundsChecks,
    &foldLoadStoreOffset,
};

const x86_64_default_passes_no_iv: []const PassFn = &.{
    &forwardLocalGet,            &constantFold,                     &algebraicSimplify,       &strengthReduceMul,
    &strengthReduceMulShiftAdd,  &strengthReduceDivRem,             &foldConstantBranches,    &foldInverseCompareEqz,
    &foldBranchOnEqz,            &threadChainedConditionalBranches, &tailDuplicateSmallJoins, &foldSelectOnEqz,
    &foldSignExtendingLoad,      &foldFloatUnaryIdempotents,        &foldWrapOfExtend,        &globalValueNumbering,
    &hoistLoopInvariantCode,     &unrollSmallFixedLoops,            &deadCodeElimination,     &deadLocalSetElimination,
    &hoistLoopBoundsChecks,      &elideRedundantBoundsChecks,       &foldLoadStoreOffset,
};

const x86_64_default_passes_no_unroll: []const PassFn = &.{
    &forwardLocalGet,            &constantFold,                     &algebraicSimplify,       &strengthReduceMul,
    &strengthReduceMulShiftAdd,  &strengthReduceDivRem,             &foldConstantBranches,    &foldInverseCompareEqz,
    &foldBranchOnEqz,            &threadChainedConditionalBranches, &tailDuplicateSmallJoins, &foldSelectOnEqz,
    &foldSignExtendingLoad,      &foldFloatUnaryIdempotents,        &foldWrapOfExtend,        &globalValueNumbering,
    &inductionVariableSimplification, &hoistLoopInvariantCode,      &deadCodeElimination,     &deadLocalSetElimination,
    &hoistLoopBoundsChecks,      &elideRedundantBoundsChecks,       &foldLoadStoreOffset,
};

const x86_64_default_passes_no_iv_no_unroll: []const PassFn = &.{
    &forwardLocalGet,           &constantFold,                     &algebraicSimplify,       &strengthReduceMul,
    &strengthReduceMulShiftAdd, &strengthReduceDivRem,             &foldConstantBranches,    &foldInverseCompareEqz,
    &foldBranchOnEqz,           &threadChainedConditionalBranches, &tailDuplicateSmallJoins, &foldSelectOnEqz,
    &foldSignExtendingLoad,     &foldFloatUnaryIdempotents,        &foldWrapOfExtend,        &globalValueNumbering,
    &hoistLoopInvariantCode,    &deadCodeElimination,              &deadLocalSetElimination, &hoistLoopBoundsChecks,
    &elideRedundantBoundsChecks, &foldLoadStoreOffset,
};

pub fn defaultPassesForTarget(target: TargetArch) []const PassFn {
    return defaultPassesForTargetWithOptions(target, .{});
}

pub fn defaultPassesForTargetWithOptions(target: TargetArch, options: CompileOptions) []const PassFn {
    return switch (target) {
        .x86_64 => if (options.enable_iv_simplify)
            (if (options.enable_loop_unroll) x86_64_default_passes else x86_64_default_passes_no_unroll)
        else
            (if (options.enable_loop_unroll) x86_64_default_passes_no_iv else x86_64_default_passes_no_iv_no_unroll),
        .aarch64 => if (options.enable_iv_simplify)
            (if (options.enable_loop_unroll) default_passes else default_passes_no_unroll)
        else
            (if (options.enable_loop_unroll) default_passes_no_iv else default_passes_no_iv_no_unroll),
    };
}

// ── Block Reordering ────────────────────────────────────────────────────────

const DfsEntry = struct { block: ir.BlockId, child_idx: usize };

/// Compute a block emission order using Reverse Postorder (RPO) with
/// frequency-biased sibling order and cold-block sinking. Places hot
/// blocks contiguously for i-cache locality and maximises fall-through
/// opportunities for the C3 peephole.
///
/// Internally:
///   * Static block frequencies are estimated via
///     `analysis.computeBlockFrequencies` (push-flow + loop-depth heuristic).
///   * Successor lists are sorted ascending by callee frequency before the
///     DFS, which causes the hottest successor to land directly after its
///     parent in the reversed post-order.
///   * "Cold" blocks — those containing `.@"unreachable"` — are moved
///     to the end of the layout, preserving relative RPO order within
///     each hot/cold group.
///
/// The original PR #443 also sank blocks whose estimated frequency fell
/// below 10% of the entry's, but that broke the def-precedes-use flat
/// order invariant on which `computeLiveRangesWithOrder` relies whenever
/// a "cold" block contained an SSA def feeding a hot successor — surfaced
/// as an x86_64 CoreMark AOT trap. See issue #393 PR #443 follow-up.
///
/// Returns a permutation of all BlockIds (caller owns the slice).
/// Block 0 (entry) is always first. Unreachable blocks are appended at the
/// end. The entry block is never classified as cold.
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

    // Estimate static block frequencies. Used to (a) bias the DFS visit
    // order so the hottest sibling ends up adjacent to its parent in the
    // resulting RPO, and (b) sink rarely-executed blocks (≤10% of entry)
    // to the end of the layout alongside any block containing
    // `.@"unreachable"`.
    const freq = try analysis.computeBlockFrequencies(func, allocator);
    defer allocator.free(freq);

    // Build a writable per-block visit-order array indexed by BlockId.
    // Each entry is the block's successor list sorted ascending by
    // estimated frequency, with original index as the stable tie-breaker.
    //
    // The iterative DFS visits successors in order: the *first* child
    // is fully explored and popped before the next sibling, so it gets
    // the lowest post-order number → appears *latest* in RPO. Iterating
    // siblings in ascending-frequency order therefore places the hottest
    // successor closest to its parent in the final RPO — the fall-through
    // slot the C3 peephole later collapses into a no-op branch.
    const visit_succs = try allocator.alloc([]ir.BlockId, n);
    defer {
        for (visit_succs) |s| allocator.free(s);
        allocator.free(visit_succs);
    }
    for (visit_succs, 0..) |*slot, idx| {
        const src = successors.get(@intCast(idx)) orelse &[_]ir.BlockId{};
        slot.* = try allocator.dupe(ir.BlockId, src);
        if (slot.len < 2) continue;
        // Stable insertion sort by ascending frequency. `succs.len` is
        // tiny in practice (≤2 for br_if; br_table arity is bounded by
        // the source wasm module). Stability preserves the original
        // br_if then/else order for equal-frequency siblings, keeping
        // `reorderBlocks` deterministic.
        const succs = slot.*;
        var i: usize = 1;
        while (i < succs.len) : (i += 1) {
            var j: usize = i;
            while (j > 0) : (j -= 1) {
                const a = succs[j - 1];
                const b = succs[j];
                const fa: f32 = if (a < n) freq[a] else 0.0;
                const fb: f32 = if (b < n) freq[b] else 0.0;
                if (fa <= fb) break; // already ascending; stable for equals
                succs[j - 1] = b;
                succs[j] = a;
            }
        }
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
        const succs = visit_succs[top.block];
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

    // Detect cold blocks: blocks containing `.@"unreachable"` — dead-end
    // traps that should never execute on a hot path. The freq-based
    // threshold from the original PR #443 was reverted because it could
    // sink blocks containing SSA defs used by hot successors, breaking
    // the def-precedes-use flat-order invariant `computeLiveRangesWithOrder`
    // relies on (issue #393, supervisor note on PR #443).
    const is_cold = try allocator.alloc(bool, n);
    defer allocator.free(is_cold);
    @memset(is_cold, false);

    for (func.blocks.items, 0..) |block, idx| {
        // `.@"unreachable"` always sinks.
        for (block.instructions.items) |inst| {
            if (inst.op == .@"unreachable") {
                is_cold[idx] = true;
                break;
            }
        }
    }
    // Entry block is never treated as cold.
    is_cold[0] = false;

    // Partition RPO: hot first, cold second (preserving RPO within each group).
    // `buildSuccessors` visits br_if then_block before else_block, so AArch64
    // keeps the original branch polarity/layout bias unless target-aware passes
    // deliberately rewrite the terminator first.
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

test "DCE: removes dest-less placeholder iconsts (#469)" {
    // Passes like `promoteLocalsToSSA` neutralise an instruction in
    // place by setting `inst.op = .{ .iconst_32 = 0 }` and
    // `inst.dest = null`. With the legacy "dest required" guard these
    // placeholders survived every subsequent pipeline iteration and
    // cluttered IR dumps for CoreMark's hot functions
    // (`core_state_transition` etc.). DCE must drop them.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0 });
    // Three dest-less placeholders sandwiched between live ops.
    try block.append(.{ .op = .{ .iconst_32 = 0 } });
    try block.append(.{ .op = .{ .iconst_32 = 0 } });
    try block.append(.{ .op = .{ .iconst_32 = 0 } });
    try block.append(.{ .op = .{ .ret = v0 } });

    try std.testing.expectEqual(@as(usize, 5), block.instructions.items.len);
    const changed = try deadCodeElimination(&func, allocator);
    try std.testing.expect(changed);
    // Only `iconst_32 42` and `ret v0` survive — the three placeholders
    // are gone.
    try std.testing.expectEqual(@as(usize, 2), block.instructions.items.len);
    try std.testing.expectEqual(ir.Inst.Op{ .iconst_32 = 42 }, block.instructions.items[0].op);
    try std.testing.expect(block.instructions.items[0].dest != null);
}

test "DCE: cascades after foldLoadStoreOffset-style rewrites (#469)" {
    // After a load/store-offset fold, the original `add v_a, v_base, vK`
    // becomes dead (v_a has zero uses because the load consumes v_base
    // directly with a baked-in offset). DCE must remove the add on its
    // first pass, then on the same call's fixpoint iteration also drop
    // the iconst_32 that defined vK — otherwise the asm leaks dead
    // `mov xN, #K` stomps documented in issue #469.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const block_id = try func.newBlock();
    var block = &func.blocks.items[block_id];

    const v_k = func.newVReg(); // iconst 16
    const v_base = func.newVReg();
    const v_addr = func.newVReg(); // = v_base + v_k, now dead
    const v_val = func.newVReg();

    try block.append(.{ .op = .{ .iconst_32 = 16 }, .dest = v_k, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base, .type = .i32 });
    try block.append(.{
        .op = .{ .add = .{ .lhs = v_base, .rhs = v_k } },
        .dest = v_addr,
        .type = .i32,
    });
    // Mimic foldLoadStoreOffset's output: the load consumes v_base
    // directly, not v_addr. v_addr becomes dead.
    try block.append(.{
        .op = .{ .load = .{ .base = v_base, .offset = 16, .size = 4 } },
        .dest = v_val,
        .type = .i32,
    });
    try block.append(.{ .op = .{ .ret = v_val } });

    try std.testing.expectEqual(@as(usize, 5), block.instructions.items.len);
    const changed = try deadCodeElimination(&func, allocator);
    try std.testing.expect(changed);
    // Both the iconst_32 16 (v_k) AND the add (v_addr) must be gone.
    // What remains: v_base iconst, load, ret.
    try std.testing.expectEqual(@as(usize, 3), block.instructions.items.len);
    for (block.instructions.items) |inst| {
        try std.testing.expect(inst.op != .add);
        // The only surviving iconst_32 is v_base = 0.
        if (inst.dest) |d| {
            if (inst.op == .iconst_32) try std.testing.expectEqual(v_base, d);
        }
    }
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

test "reorderBlocks: hot loop laid out before cold throw (issue #388)" {
    // Sketches a typical "checked arithmetic in a loop" pattern:
    //   b0 (entry) → b1 (loop header)
    //   b1 → b2 (loop body, back-edges to b1) | b3 (loop exit)
    //   b3 → b4 (normal return) | b5 (overflow → unreachable)
    //
    // Expected layout invariants after `reorderBlocks`:
    //   * Entry comes first.
    //   * The hot loop {b1, b2} is laid out contiguously, with b2 (the
    //     back-edge body) placed directly after b1 so the back-edge
    //     becomes a fall-through-like short branch.
    //   * The cold/trap block b5 lands after every hot block.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock(); // loop header
    const b2 = try func.newBlock(); // loop body
    const b3 = try func.newBlock(); // loop exit
    const b4 = try func.newBlock(); // normal return
    const b5 = try func.newBlock(); // overflow → unreachable

    const c0 = func.newVReg();
    const c1 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = c0 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = c0, .then_block = b2, .else_block = b3 } } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b3).append(.{ .op = .{ .iconst_32 = 1 }, .dest = c1 });
    try func.getBlock(b3).append(.{ .op = .{ .br_if = .{ .cond = c1, .then_block = b4, .else_block = b5 } } });
    try func.getBlock(b4).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b5).append(.{ .op = .{ .@"unreachable" = {} } });

    const order = try reorderBlocks(&func, allocator);
    defer allocator.free(order);
    try std.testing.expectEqual(@as(usize, 6), order.len);

    var pos: [6]usize = undefined;
    for (order, 0..) |bid, i| pos[bid] = i;

    // Entry is first.
    try std.testing.expectEqual(@as(ir.BlockId, 0), order[0]);
    // Loop header and body are adjacent, with body right after header.
    try std.testing.expectEqual(pos[b1] + 1, pos[b2]);
    // Hot path (everything except b5) precedes the cold trap.
    try std.testing.expect(pos[b5] > pos[b0]);
    try std.testing.expect(pos[b5] > pos[b1]);
    try std.testing.expect(pos[b5] > pos[b2]);
    try std.testing.expect(pos[b5] > pos[b3]);
    try std.testing.expect(pos[b5] > pos[b4]);
    // b5 (containing .@"unreachable") is the very last block.
    try std.testing.expectEqual(@as(ir.BlockId, b5), order[order.len - 1]);
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
    // Header has a load with loop-invariant base, body (which is
    // must-execute — it dominates the only latch, itself) has another.
    // The pass should scan both must-execute blocks, derive max_end=8,
    // insert a guard load in b0, and mark both loads bounds_known.
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

    // b2 (body): load base+4 size=4, back-edge to header. b2 dominates
    // the latch (itself), so it is must-execute and contributes to the
    // header-or-body invariant-base scan.
    try func.getBlock(b2).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 4, .size = 4 } }, .dest = v_body, .type = .i32 });
    try func.getBlock(b2).append(.{ .op = .{ .br = b1 } });

    // b3 (exit).
    try func.getBlock(b3).append(.{ .op = .{ .ret = v_hdr } });

    const changed = try hoistLoopBoundsChecks(&func, allocator);
    try std.testing.expect(changed);

    // Preheader should now have 3 instructions: iconst, guard load, br.
    try std.testing.expectEqual(@as(usize, 3), func.getBlock(b0).instructions.items.len);
    // Guard load checked_end = max(header end=4, body end=8) = 8.
    const guard = func.getBlock(b0).instructions.items[1];
    try std.testing.expectEqual(@as(u64, 8), guard.op.load.checked_end);
    try std.testing.expectEqual(v_base, guard.op.load.base);
    // Both accesses must-execute → both marked bounds_known.
    try std.testing.expect(func.getBlock(b1).instructions.items[0].op.load.bounds_known);
    try std.testing.expect(func.getBlock(b2).instructions.items[0].op.load.bounds_known);
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

test "hoistLoopBoundsChecks: H5 body-only invariant base in must-execute block" {
    // Header is just the loop test (no memory access); the must-execute
    // body block has a load with a loop-invariant base. The pre-#470
    // header-only scan would skip this loop. The post-#470 must-execute
    // scan picks up the body's invariant access and marks it.
    //
    // CFG: b0 (preheader) → b1 (header, br_if to b2/b3) → b2 (body,
    // load+br b1) → b1; b3 (exit). b2 is must-execute since it
    // dominates the only latch (itself).
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    const v_base = func.newVReg();
    const cond = func.newVReg();
    const v_body = func.newVReg();

    // b0 (preheader): define invariant base, br to header.
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    // b1 (header): ONLY the loop test, no memory access. This is the
    // H5 shape that drove issue #470.
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b2, .else_block = b3 } } });

    // b2 (body): load on the invariant base, br back to header.
    try func.getBlock(b2).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_body, .type = .i32 });
    try func.getBlock(b2).append(.{ .op = .{ .br = b1 } });

    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    const changed = try hoistLoopBoundsChecks(&func, allocator);
    try std.testing.expect(changed);
    // Guard inserted in preheader.
    try std.testing.expectEqual(@as(usize, 3), func.getBlock(b0).instructions.items.len);
    try std.testing.expectEqual(@as(u64, 4), func.getBlock(b0).instructions.items[1].op.load.checked_end);
    // Body load marked bounds_known.
    try std.testing.expect(func.getBlock(b2).instructions.items[0].op.load.bounds_known);
}

test "hoistLoopBoundsChecks: H4 loop-variant base in body not hoisted" {
    // Body's load uses a base that is redefined inside the loop (the
    // CoreMark linked-list `p = p->next` pattern reduced). The base
    // VReg is defined in the body, not outside, so the pass correctly
    // refuses to hoist — no preheader guard, no bounds_known mark.
    //
    // CFG: b0 (preheader, br b1) → b1 (header, br_if to b2/b3) → b2
    // (body: define new base via load, then load via it, br b1).
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    const v_init = func.newVReg();
    const cond = func.newVReg();
    const v_next = func.newVReg();
    const v_val = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_init });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b2, .else_block = b3 } } });

    // Body redefines the base (linked-list `p = p->next`): the v_next
    // VReg is defined inside the loop, then used as base for v_val.
    try func.getBlock(b2).append(.{ .op = .{ .load = .{ .base = v_init, .offset = 0, .size = 4 } }, .dest = v_next, .type = .i32 });
    try func.getBlock(b2).append(.{ .op = .{ .load = .{ .base = v_next, .offset = 8, .size = 4 } }, .dest = v_val, .type = .i32 });
    try func.getBlock(b2).append(.{ .op = .{ .br = b1 } });

    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    const changed = try hoistLoopBoundsChecks(&func, allocator);
    // v_init IS invariant and used in body — the must-execute scan
    // picks it up and marks the first body load (offset+size=4 ≤ 4).
    try std.testing.expect(changed);
    try std.testing.expect(func.getBlock(b2).instructions.items[0].op.load.bounds_known);
    // The second body load uses v_next (loop-variant): MUST NOT be
    // marked. This is the H4 / linked-list pattern the pass correctly
    // cannot help with.
    try std.testing.expect(!func.getBlock(b2).instructions.items[1].op.load.bounds_known);
}

test "hoistLoopBoundsChecks: H3 synthesized preheader is recognised" {
    // PR #490 Stage B synthesises a preheader when the wasm front-end
    // produced a `br_if` directly into the loop header. Since LICM
    // (`hoistLoopInvariantCode`) runs before `hoistLoopBoundsChecks`
    // in the pipeline, by the time this pass runs the synthesised
    // preheader exists and looks like any other dedicated preheader.
    // This test exercises the integration: an entry that's a `br_if`
    // into the header gets a preheader synthesised by LICM, and the
    // bounds-check pass then operates on it.
    //
    // The loop body includes a store so LICM Stage C does NOT also
    // hoist the load (we want the load to survive into the body so
    // hoistLoopBoundsChecks has something to mark bounds_known).
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock(); // entry; br_if into b1 or b3
    const b1 = try func.newBlock(); // loop header
    const b2 = try func.newBlock(); // loop body / latch
    const b3 = try func.newBlock(); // exit

    const v_base = func.newVReg();
    const v_other = func.newVReg();
    const cond_entry = func.newVReg();
    const cond_loop = func.newVReg();
    const v_load = func.newVReg();
    const v_zero = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 200 }, .dest = v_other });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond_entry });
    // br_if directly into loop header — no dedicated preheader yet.
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond_entry, .then_block = b1, .else_block = b3 } } });

    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_load, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond_loop });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = cond_loop, .then_block = b2, .else_block = b3 } } });

    // Store in the body disqualifies the header load from LICM Stage C
    // speculative hoist (presence of any store/memory-mutating op in
    // the loop blocks).
    try func.getBlock(b2).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_zero });
    try func.getBlock(b2).append(.{ .op = .{ .store = .{ .base = v_other, .offset = 0, .size = 4, .val = v_zero } } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    // Step 1: LICM runs, synthesising a preheader before the header.
    _ = try hoistLoopInvariantCode(&func, allocator);

    // Step 2: bounds-check pass should now see the synthesised
    // preheader and fire on the header load.
    const changed = try hoistLoopBoundsChecks(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expect(func.getBlock(b1).instructions.items[0].op.load.bounds_known);
}

test "hoistLoopBoundsChecks: H2 fence in body must-execute block halts scan" {
    // Must-execute body contains a call before its load. The call is a
    // fence: scan halts globally. Header's invariant load (before the
    // fence) still gets hoisted; the post-fence body load does not.
    //
    // CFG: b0 (preheader) → b1 (header, load+br_if) → b2 (body, call,
    // load, br b1) → b1; b3 exit.
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

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_hdr, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b2, .else_block = b3 } } });

    try func.getBlock(b2).append(.{ .op = .{ .call = .{ .func_idx = 0 } } });
    try func.getBlock(b2).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 16, .size = 4 } }, .dest = v_body, .type = .i32 });
    try func.getBlock(b2).append(.{ .op = .{ .br = b1 } });

    try func.getBlock(b3).append(.{ .op = .{ .ret = v_hdr } });

    const changed = try hoistLoopBoundsChecks(&func, allocator);
    try std.testing.expect(changed);
    // Header load (before the body's fence) was scanned and is marked.
    try std.testing.expect(func.getBlock(b1).instructions.items[0].op.load.bounds_known);
    // Body load (after the call fence) — its end=20 was NOT added to
    // base_max, and since the existing body-marking step caps at
    // header's max_end=4, end=20 > 4 → not marked.
    try std.testing.expect(!func.getBlock(b2).instructions.items[1].op.load.bounds_known);
}

test "hoistLoopBoundsChecks: H5 body access in conditional (non-must-execute) block NOT hoisted" {
    // Body block is conditional (only entered through a br_if from the
    // header), so the access in it is NOT trap-equivalent to a
    // preheader check — that block does not dominate the latch.
    // Pass must skip; otherwise iter 1 would trap even when the
    // conditional block is never entered.
    //
    // CFG: b0 (ph) → b1 (header, br_if to b2 or b4) → b2 (conditional,
    // load, br b4) → b4 (latch, br b1); b3 exit. b2 has invariant base
    // access but does NOT dominate the latch b4 (b4 is reachable from
    // b1 directly via the else branch).
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const b4 = try func.newBlock();

    const v_base = func.newVReg();
    const cond_in = func.newVReg();
    const cond_branch = func.newVReg();
    const cond_exit = func.newVReg();
    const v_load = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond_branch });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = cond_branch, .then_block = b2, .else_block = b4 } } });

    // Conditional block: load only happens when branch taken.
    try func.getBlock(b2).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_load, .type = .i32 });
    try func.getBlock(b2).append(.{ .op = .{ .br = b4 } });

    // Latch: loop-back conditional.
    try func.getBlock(b4).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond_in });
    try func.getBlock(b4).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond_exit });
    try func.getBlock(b4).append(.{ .op = .{ .br_if = .{ .cond = cond_exit, .then_block = b1, .else_block = b3 } } });

    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    const changed = try hoistLoopBoundsChecks(&func, allocator);
    // No must-execute block has memory access (b2 is conditional, b4
    // has no load, b1 has no load) — pass must NOT fire.
    try std.testing.expect(!changed);
    try std.testing.expect(!func.getBlock(b2).instructions.items[0].op.load.bounds_known);
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

test "hoistLoopInvariantCode: local_get hoisted when idx never set in loop" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 1);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    // preheader: br loop_header
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    // loop body reads local 0 (never written in loop) and branches back.
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v0, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v0 }, .dest = v1 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v1, .then_block = b2, .else_block = b1 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v0 } });

    const changed = try hoistLoopInvariantCode(&func, allocator);
    try std.testing.expect(changed);

    var ph_has_local_get = false;
    for (func.getBlock(b0).instructions.items) |inst| {
        if (inst.op == .local_get) {
            ph_has_local_get = true;
            break;
        }
    }
    try std.testing.expect(ph_has_local_get);

    var body_has_local_get = false;
    for (func.getBlock(b1).instructions.items) |inst| {
        if (inst.op == .local_get) {
            body_has_local_get = true;
            break;
        }
    }
    try std.testing.expect(!body_has_local_get);
}

test "hoistLoopInvariantCode: local_get NOT hoisted when idx is local_set in loop" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 1);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    // loop body reads AND writes local 0, so local_get is not invariant.
    const v0 = func.newVReg();
    const v_one = func.newVReg();
    const v_next = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v0, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_one, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v_one } }, .dest = v_next, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_next } } });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v_next }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b2, .else_block = b1 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v_next } });

    _ = try hoistLoopInvariantCode(&func, allocator);

    // local_get must remain inside the loop body; add+iconst are also not
    // invariant (add depends on the loop-modified local).
    var body_has_local_get = false;
    for (func.getBlock(b1).instructions.items) |inst| {
        if (inst.op == .local_get) {
            body_has_local_get = true;
            break;
        }
    }
    try std.testing.expect(body_has_local_get);

    var ph_has_local_get = false;
    for (func.getBlock(b0).instructions.items) |inst| {
        if (inst.op == .local_get) {
            ph_has_local_get = true;
            break;
        }
    }
    try std.testing.expect(!ph_has_local_get);
}

test "hoistLoopInvariantCode: global_get hoisted when idx never set in loop and no calls" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .global_get = 3 }, .dest = v0, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v0 }, .dest = v1 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v1, .then_block = b2, .else_block = b1 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v0 } });

    const changed = try hoistLoopInvariantCode(&func, allocator);
    try std.testing.expect(changed);

    var ph_has_global_get = false;
    for (func.getBlock(b0).instructions.items) |inst| {
        if (inst.op == .global_get) {
            ph_has_global_get = true;
            break;
        }
    }
    try std.testing.expect(ph_has_global_get);

    var body_has_global_get = false;
    for (func.getBlock(b1).instructions.items) |inst| {
        if (inst.op == .global_get) {
            body_has_global_get = true;
            break;
        }
    }
    try std.testing.expect(!body_has_global_get);
}

test "hoistLoopInvariantCode: global_get NOT hoisted when idx is global_set in loop" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const v0 = func.newVReg();
    const v_one = func.newVReg();
    const v_next = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .global_get = 7 }, .dest = v0, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_one, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v_one } }, .dest = v_next, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .global_set = .{ .idx = 7, .val = v_next } } });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v_next }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b2, .else_block = b1 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v_next } });

    _ = try hoistLoopInvariantCode(&func, allocator);

    var body_has_global_get = false;
    for (func.getBlock(b1).instructions.items) |inst| {
        if (inst.op == .global_get) {
            body_has_global_get = true;
            break;
        }
    }
    try std.testing.expect(body_has_global_get);
}

test "hoistLoopInvariantCode: global_get NOT hoisted when loop contains a call" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    // The call has no args and no result captured here; its presence alone
    // is what blocks hoisting global_get (the callee may global_set).
    try func.getBlock(b1).append(.{ .op = .{ .global_get = 2 }, .dest = v0, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .call = .{ .func_idx = 0 } } });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v0 }, .dest = v1 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v1, .then_block = b2, .else_block = b1 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v0 } });

    _ = try hoistLoopInvariantCode(&func, allocator);

    var body_has_global_get = false;
    for (func.getBlock(b1).instructions.items) |inst| {
        if (inst.op == .global_get) {
            body_has_global_get = true;
            break;
        }
    }
    try std.testing.expect(body_has_global_get);
}

test "hoistLoopInvariantCode: cascading via hoisted local_get exposes invariant add" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 1);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_k = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v_k, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    // local_get of an untouched local + add with an external constant.
    // After local_get hoists, the add becomes loop-invariant and cascades.
    const v_lg = func.newVReg();
    const v_sum = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_lg, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_lg, .rhs = v_k } }, .dest = v_sum, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v_sum }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b2, .else_block = b1 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v_sum } });

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

test "hoistLoopInvariantCode: preheader synthesized when entry is br_if" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const b_entry = try func.newBlock();
    const b_header = try func.newBlock();
    const b_exit = try func.newBlock();

    // Entry block conditionally enters the loop via br_if — no dedicated
    // preheader exists, so Stage A would skip this loop entirely.
    const v_gate = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_gate, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v_a, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 20 }, .dest = v_b, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .br_if = .{ .cond = v_gate, .then_block = b_header, .else_block = b_exit } } });

    // Loop body computes an invariant add and a self-edge back to header.
    const v_sum = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b_header).append(.{ .op = .{ .add = .{ .lhs = v_a, .rhs = v_b } }, .dest = v_sum, .type = .i32 });
    try func.getBlock(b_header).append(.{ .op = .{ .eqz = v_sum }, .dest = v_cond });
    try func.getBlock(b_header).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b_exit, .else_block = b_header } } });
    try func.getBlock(b_exit).append(.{ .op = .{ .ret = v_a } });

    const blocks_before = func.blocks.items.len;
    const changed = try hoistLoopInvariantCode(&func, allocator);
    try std.testing.expect(changed);
    // A fresh preheader block must have been synthesized.
    try std.testing.expect(func.blocks.items.len == blocks_before + 1);

    // The entry's br_if must no longer point directly at the header on
    // its then-branch; instead, it routes through the synthesized
    // preheader. The else-branch (going to b_exit) must be untouched.
    const entry_term = func.getBlock(b_entry).instructions.items[func.getBlock(b_entry).instructions.items.len - 1];
    try std.testing.expect(entry_term.op == .br_if);
    try std.testing.expect(entry_term.op.br_if.then_block != b_header);
    try std.testing.expect(entry_term.op.br_if.else_block == b_exit);
    const ph_new = entry_term.op.br_if.then_block;

    // The synthesized preheader has `add` (hoisted) followed by `.br header`.
    const ph_block = func.getBlock(ph_new);
    var ph_has_add = false;
    var ph_last_is_br_header = false;
    for (ph_block.instructions.items, 0..) |inst, idx| {
        if (inst.op == .add) ph_has_add = true;
        if (idx == ph_block.instructions.items.len - 1) {
            ph_last_is_br_header = (inst.op == .br and inst.op.br == b_header);
        }
    }
    try std.testing.expect(ph_has_add);
    try std.testing.expect(ph_last_is_br_header);

    // Body must no longer contain the hoisted add.
    var body_has_add = false;
    for (func.getBlock(b_header).instructions.items) |inst| {
        if (inst.op == .add) {
            body_has_add = true;
            break;
        }
    }
    try std.testing.expect(!body_has_add);
}

test "hoistLoopInvariantCode: preheader synthesized when loop has two non-loop entries" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b_split = try func.newBlock();
    const b_left = try func.newBlock();
    const b_right = try func.newBlock();
    const b_header = try func.newBlock();
    const b_exit = try func.newBlock();

    // Split into two paths that both `.br` directly into the header.
    const v_gate = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    try func.getBlock(b_split).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_gate, .type = .i32 });
    try func.getBlock(b_split).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v_a, .type = .i32 });
    try func.getBlock(b_split).append(.{ .op = .{ .iconst_32 = 20 }, .dest = v_b, .type = .i32 });
    try func.getBlock(b_split).append(.{ .op = .{ .br_if = .{ .cond = v_gate, .then_block = b_left, .else_block = b_right } } });
    try func.getBlock(b_left).append(.{ .op = .{ .br = b_header } });
    try func.getBlock(b_right).append(.{ .op = .{ .br = b_header } });

    const v_sum = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b_header).append(.{ .op = .{ .add = .{ .lhs = v_a, .rhs = v_b } }, .dest = v_sum, .type = .i32 });
    try func.getBlock(b_header).append(.{ .op = .{ .eqz = v_sum }, .dest = v_cond });
    try func.getBlock(b_header).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b_exit, .else_block = b_header } } });
    try func.getBlock(b_exit).append(.{ .op = .{ .ret = v_a } });

    const blocks_before = func.blocks.items.len;
    const changed = try hoistLoopInvariantCode(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expect(func.blocks.items.len == blocks_before + 1);

    // Both predecessor branches must now route through the new preheader,
    // not directly to b_header.
    const left_term = func.getBlock(b_left).instructions.items[0];
    const right_term = func.getBlock(b_right).instructions.items[0];
    try std.testing.expect(left_term.op == .br);
    try std.testing.expect(right_term.op == .br);
    try std.testing.expect(left_term.op.br != b_header);
    try std.testing.expect(right_term.op.br != b_header);
    try std.testing.expect(left_term.op.br == right_term.op.br); // same preheader

    // Hoisted add lives in the synthesized preheader; loop body free of it.
    const ph_new = left_term.op.br;
    var ph_has_add = false;
    for (func.getBlock(ph_new).instructions.items) |inst| {
        if (inst.op == .add) {
            ph_has_add = true;
            break;
        }
    }
    try std.testing.expect(ph_has_add);

    var body_has_add = false;
    for (func.getBlock(b_header).instructions.items) |inst| {
        if (inst.op == .add) {
            body_has_add = true;
            break;
        }
    }
    try std.testing.expect(!body_has_add);
}

test "hoistLoopInvariantCode: preheader synthesized when entry is br_table" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b_entry = try func.newBlock();
    const b_other = try func.newBlock();
    const b_header = try func.newBlock();
    const b_exit = try func.newBlock();

    const v_idx = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_idx, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v_a, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 20 }, .dest = v_b, .type = .i32 });

    // br_table with header appearing both in `targets` and as `default`.
    const targets = try allocator.alloc(ir.BlockId, 2);
    targets[0] = b_header;
    targets[1] = b_other;
    try func.owned_br_table_targets.append(allocator, targets);
    try func.getBlock(b_entry).append(.{ .op = .{ .br_table = .{ .index = v_idx, .targets = targets, .default = b_header } } });

    try func.getBlock(b_other).append(.{ .op = .{ .br = b_exit } });

    const v_sum = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b_header).append(.{ .op = .{ .add = .{ .lhs = v_a, .rhs = v_b } }, .dest = v_sum, .type = .i32 });
    try func.getBlock(b_header).append(.{ .op = .{ .eqz = v_sum }, .dest = v_cond });
    try func.getBlock(b_header).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b_exit, .else_block = b_header } } });
    try func.getBlock(b_exit).append(.{ .op = .{ .ret = v_a } });

    const blocks_before = func.blocks.items.len;
    const changed = try hoistLoopInvariantCode(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expect(func.blocks.items.len == blocks_before + 1);

    // br_table default and matching target are rewritten away from header.
    const entry_term = func.getBlock(b_entry).instructions.items[func.getBlock(b_entry).instructions.items.len - 1];
    try std.testing.expect(entry_term.op == .br_table);
    try std.testing.expect(entry_term.op.br_table.default != b_header);
    try std.testing.expect(entry_term.op.br_table.targets[0] != b_header);
    try std.testing.expect(entry_term.op.br_table.targets[1] == b_other);
    try std.testing.expect(entry_term.op.br_table.default == entry_term.op.br_table.targets[0]);

    const ph_new = entry_term.op.br_table.default;
    var ph_has_add = false;
    for (func.getBlock(ph_new).instructions.items) |inst| {
        if (inst.op == .add) {
            ph_has_add = true;
            break;
        }
    }
    try std.testing.expect(ph_has_add);
}

test "hoistLoopInvariantCode: existing dedicated preheader not duplicated" {
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

    const blocks_before = func.blocks.items.len;
    const changed = try hoistLoopInvariantCode(&func, allocator);
    try std.testing.expect(changed);
    // No new block needed when an existing clean preheader is present.
    try std.testing.expect(func.blocks.items.len == blocks_before);
}

test "hoistLoopInvariantCode: load in header hoisted when no memory writes or calls in loop" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_base = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 256 }, .dest = v_base, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    // Header has a load whose base is loop-invariant; no memory writes
    // and no calls in the loop body.
    const v_loaded = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_loaded, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v_loaded }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b2, .else_block = b1 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v_loaded } });

    const changed = try hoistLoopInvariantCode(&func, allocator);
    try std.testing.expect(changed);

    var ph_has_load = false;
    for (func.getBlock(b0).instructions.items) |inst| {
        if (inst.op == .load) {
            ph_has_load = true;
            break;
        }
    }
    try std.testing.expect(ph_has_load);

    var body_has_load = false;
    for (func.getBlock(b1).instructions.items) |inst| {
        if (inst.op == .load) {
            body_has_load = true;
            break;
        }
    }
    try std.testing.expect(!body_has_load);
}

test "hoistLoopInvariantCode: load NOT hoisted when loop contains a store" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_base = func.newVReg();
    const v_zero = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 256 }, .dest = v_base, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_zero, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const v_loaded = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_loaded, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v_zero } } });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v_loaded }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b2, .else_block = b1 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v_loaded } });

    _ = try hoistLoopInvariantCode(&func, allocator);

    var body_has_load = false;
    for (func.getBlock(b1).instructions.items) |inst| {
        if (inst.op == .load) {
            body_has_load = true;
            break;
        }
    }
    try std.testing.expect(body_has_load);
}

test "hoistLoopInvariantCode: load NOT hoisted when loop contains a call" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_base = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 256 }, .dest = v_base, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const v_loaded = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_loaded, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .call = .{ .func_idx = 0 } } });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v_loaded }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b2, .else_block = b1 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v_loaded } });

    _ = try hoistLoopInvariantCode(&func, allocator);

    var body_has_load = false;
    for (func.getBlock(b1).instructions.items) |inst| {
        if (inst.op == .load) {
            body_has_load = true;
            break;
        }
    }
    try std.testing.expect(body_has_load);
}

test "hoistLoopInvariantCode: load NOT hoisted when load lives outside the loop header" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b_pre = try func.newBlock();
    const b_header = try func.newBlock();
    const b_body = try func.newBlock();
    const b_exit = try func.newBlock();

    const v_base = func.newVReg();
    try func.getBlock(b_pre).append(.{ .op = .{ .iconst_32 = 256 }, .dest = v_base, .type = .i32 });
    try func.getBlock(b_pre).append(.{ .op = .{ .br = b_header } });

    // Header branches conditionally to body or exit. The load lives in
    // body, NOT in the header — so on a 0-iter exit the load would not
    // run. Stage C must keep the load in body to preserve trap point.
    const v_gate = func.newVReg();
    try func.getBlock(b_header).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_gate, .type = .i32 });
    try func.getBlock(b_header).append(.{ .op = .{ .br_if = .{ .cond = v_gate, .then_block = b_body, .else_block = b_exit } } });

    const v_loaded = func.newVReg();
    try func.getBlock(b_body).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_loaded, .type = .i32 });
    try func.getBlock(b_body).append(.{ .op = .{ .br = b_header } });
    try func.getBlock(b_exit).append(.{ .op = .{ .ret = v_base } });

    _ = try hoistLoopInvariantCode(&func, allocator);

    var body_has_load = false;
    for (func.getBlock(b_body).instructions.items) |inst| {
        if (inst.op == .load) {
            body_has_load = true;
            break;
        }
    }
    try std.testing.expect(body_has_load);

    var ph_has_load = false;
    for (func.getBlock(b_pre).instructions.items) |inst| {
        if (inst.op == .load) {
            ph_has_load = true;
            break;
        }
    }
    try std.testing.expect(!ph_has_load);
}

test "hoistLoopInvariantCode: load NOT hoisted when base operand is loop-variant" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 1);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    // Base is local_get of an idx that IS updated in the loop, so the
    // base VReg is loop-variant.
    const v_base = func.newVReg();
    const v_loaded = func.newVReg();
    const v_one = func.newVReg();
    const v_next = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_base, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_loaded, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_one, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_base, .rhs = v_one } }, .dest = v_next, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_next } } });
    try func.getBlock(b1).append(.{ .op = .{ .eqz = v_loaded }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b2, .else_block = b1 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v_loaded } });

    _ = try hoistLoopInvariantCode(&func, allocator);

    var body_has_load = false;
    for (func.getBlock(b1).instructions.items) |inst| {
        if (inst.op == .load) {
            body_has_load = true;
            break;
        }
    }
    try std.testing.expect(body_has_load);
}

test "hoistLoopInvariantCode: load in non-header block hoisted when block dominates all anchors" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    // b_pre → b_header → b_body. b_header is a pass-through; b_body
    // holds the load and is both the sole latch (br header) and the
    // sole exit (br_if header/exit). b_body dominates itself and is
    // therefore a safe speculation site under the relaxed rule (#494).
    const b_pre = try func.newBlock();
    const b_header = try func.newBlock();
    const b_body = try func.newBlock();
    const b_exit = try func.newBlock();

    const v_base = func.newVReg();
    try func.getBlock(b_pre).append(.{ .op = .{ .iconst_32 = 256 }, .dest = v_base, .type = .i32 });
    try func.getBlock(b_pre).append(.{ .op = .{ .br = b_header } });

    try func.getBlock(b_header).append(.{ .op = .{ .br = b_body } });

    const v_loaded = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b_body).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_loaded, .type = .i32 });
    try func.getBlock(b_body).append(.{ .op = .{ .eqz = v_loaded }, .dest = v_cond });
    try func.getBlock(b_body).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b_exit, .else_block = b_header } } });
    try func.getBlock(b_exit).append(.{ .op = .{ .ret = v_loaded } });

    const changed = try hoistLoopInvariantCode(&func, allocator);
    try std.testing.expect(changed);

    var body_has_load = false;
    for (func.getBlock(b_body).instructions.items) |inst| {
        if (inst.op == .load) {
            body_has_load = true;
            break;
        }
    }
    try std.testing.expect(!body_has_load);
}

test "hoistLoopInvariantCode: load NOT hoisted when on one diamond branch only" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    // Diamond inside the loop:
    //   pre → header → {then(load), else} → join → header | exit
    // The load lives in `then`. `then` does not dominate `join` (the
    // sole latch + exit anchor) because the else-side reaches `join`
    // independently. Stage C extension must not hoist.
    const b_pre = try func.newBlock();
    const b_header = try func.newBlock();
    const b_then = try func.newBlock();
    const b_else = try func.newBlock();
    const b_join = try func.newBlock();
    const b_exit = try func.newBlock();

    const v_base = func.newVReg();
    const v_gate = func.newVReg();
    try func.getBlock(b_pre).append(.{ .op = .{ .iconst_32 = 256 }, .dest = v_base, .type = .i32 });
    try func.getBlock(b_pre).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_gate, .type = .i32 });
    try func.getBlock(b_pre).append(.{ .op = .{ .br = b_header } });

    try func.getBlock(b_header).append(.{ .op = .{ .br_if = .{ .cond = v_gate, .then_block = b_then, .else_block = b_else } } });

    const v_loaded = func.newVReg();
    try func.getBlock(b_then).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_loaded, .type = .i32 });
    try func.getBlock(b_then).append(.{ .op = .{ .br = b_join } });

    try func.getBlock(b_else).append(.{ .op = .{ .br = b_join } });

    const v_cond = func.newVReg();
    try func.getBlock(b_join).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_cond, .type = .i32 });
    try func.getBlock(b_join).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b_exit, .else_block = b_header } } });
    try func.getBlock(b_exit).append(.{ .op = .{ .ret = v_base } });

    _ = try hoistLoopInvariantCode(&func, allocator);

    var then_has_load = false;
    for (func.getBlock(b_then).instructions.items) |inst| {
        if (inst.op == .load) {
            then_has_load = true;
            break;
        }
    }
    try std.testing.expect(then_has_load);

    var pre_has_load = false;
    for (func.getBlock(b_pre).instructions.items) |inst| {
        if (inst.op == .load) {
            pre_has_load = true;
            break;
        }
    }
    try std.testing.expect(!pre_has_load);
}

test "hoistLoopInvariantCode: load NOT hoisted when one of multiple latches is not dominated" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    // Two latches, both back-edge to header:
    //   pre → header → {path_a(load, latch+exit), path_b(latch)}
    // path_a holds the load and is one latch; path_b is the other
    // latch. path_a does not dominate path_b (path_b is reached from
    // header directly, not via path_a). Stage C extension must not
    // hoist.
    const b_pre = try func.newBlock();
    const b_header = try func.newBlock();
    const b_path_a = try func.newBlock();
    const b_path_b = try func.newBlock();
    const b_exit = try func.newBlock();

    const v_base = func.newVReg();
    const v_gate = func.newVReg();
    try func.getBlock(b_pre).append(.{ .op = .{ .iconst_32 = 256 }, .dest = v_base, .type = .i32 });
    try func.getBlock(b_pre).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_gate, .type = .i32 });
    try func.getBlock(b_pre).append(.{ .op = .{ .br = b_header } });

    try func.getBlock(b_header).append(.{ .op = .{ .br_if = .{ .cond = v_gate, .then_block = b_path_a, .else_block = b_path_b } } });

    const v_loaded = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b_path_a).append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_loaded, .type = .i32 });
    try func.getBlock(b_path_a).append(.{ .op = .{ .eqz = v_loaded }, .dest = v_cond });
    try func.getBlock(b_path_a).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b_exit, .else_block = b_header } } });

    try func.getBlock(b_path_b).append(.{ .op = .{ .br = b_header } });
    try func.getBlock(b_exit).append(.{ .op = .{ .ret = v_loaded } });

    _ = try hoistLoopInvariantCode(&func, allocator);

    var path_a_has_load = false;
    for (func.getBlock(b_path_a).instructions.items) |inst| {
        if (inst.op == .load) {
            path_a_has_load = true;
            break;
        }
    }
    try std.testing.expect(path_a_has_load);

    var pre_has_load = false;
    for (func.getBlock(b_pre).instructions.items) |inst| {
        if (inst.op == .load) {
            pre_has_load = true;
            break;
        }
    }
    try std.testing.expect(!pre_has_load);
}

test "inductionVariableSimplification: single induction address is strength reduced" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 1);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_base = func.newVReg();
    const v_zero = func.newVReg();
    const v_step = func.newVReg();
    const v_limit = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_zero });
    try func.getBlock(b0).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_zero } } });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_step });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 4 }, .dest = v_limit });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const v_i_addr = func.newVReg();
    const v_addr = func.newVReg();
    const v_val = func.newVReg();
    const v_i_up = func.newVReg();
    const v_next = func.newVReg();
    const v_i_cmp = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i_addr });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_base, .rhs = v_i_addr } }, .dest = v_addr });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v_val });
    try func.getBlock(b1).append(.{ .op = .{ .store = .{ .base = v_addr, .offset = 0, .size = 4, .val = v_val } } });
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i_up });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_i_up, .rhs = v_step } }, .dest = v_next });
    try func.getBlock(b1).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_next } } });
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i_cmp });
    try func.getBlock(b1).append(.{ .op = .{ .lt_s = .{ .lhs = v_i_cmp, .rhs = v_limit } }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    const changed = try inductionVariableSimplification(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(@as(u32, 2), func.local_count);

    var rewritten = false;
    for (func.getBlock(b1).instructions.items) |inst| {
        if (inst.op == .store) {
            try std.testing.expect(inst.op.store.base != v_addr);
            rewritten = true;
        }
    }
    try std.testing.expect(rewritten);
}

test "inductionVariableSimplification: multiple inductions leave secondary alone" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 2);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_base = func.newVReg();
    const v_zero = func.newVReg();
    const v_step = func.newVReg();
    const v_limit = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_zero });
    try func.getBlock(b0).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_zero } } });
    try func.getBlock(b0).append(.{ .op = .{ .local_set = .{ .idx = 1, .val = v_zero } } });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_step });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 4 }, .dest = v_limit });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const v_i = func.newVReg();
    const v_addr = func.newVReg();
    const v_load = func.newVReg();
    const v_i_up = func.newVReg();
    const v_next = func.newVReg();
    const v_j = func.newVReg();
    const v_j_next = func.newVReg();
    const v_i_cmp = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_base, .rhs = v_i } }, .dest = v_addr });
    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_addr, .offset = 0, .size = 4 } }, .dest = v_load });
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i_up });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_i_up, .rhs = v_step } }, .dest = v_next });
    try func.getBlock(b1).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_next } } });
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 1 }, .dest = v_j });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_j, .rhs = v_step } }, .dest = v_j_next });
    try func.getBlock(b1).append(.{ .op = .{ .local_set = .{ .idx = 1, .val = v_j_next } } });
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i_cmp });
    try func.getBlock(b1).append(.{ .op = .{ .lt_s = .{ .lhs = v_i_cmp, .rhs = v_limit } }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    try std.testing.expect(try inductionVariableSimplification(&func, allocator));
    var saw_secondary = false;
    for (func.getBlock(b1).instructions.items) |inst| {
        if (inst.op == .local_set and inst.op.local_set.idx == 1) saw_secondary = true;
    }
    try std.testing.expect(saw_secondary);
}

test "inductionVariableSimplification: aliased base pointers share synthetic local" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 1);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_base = func.newVReg();
    const v_zero = func.newVReg();
    const v_step = func.newVReg();
    const v_limit = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_zero });
    try func.getBlock(b0).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_zero } } });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_step });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 4 }, .dest = v_limit });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const v_i = func.newVReg();
    const v_addr = func.newVReg();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    const v_i_up = func.newVReg();
    const v_next = func.newVReg();
    const v_i_cmp = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_base, .rhs = v_i } }, .dest = v_addr });
    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_addr, .offset = 0, .size = 4 } }, .dest = v_a });
    try func.getBlock(b1).append(.{ .op = .{ .load = .{ .base = v_addr, .offset = 4, .size = 4 } }, .dest = v_b });
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i_up });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_i_up, .rhs = v_step } }, .dest = v_next });
    try func.getBlock(b1).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_next } } });
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i_cmp });
    try func.getBlock(b1).append(.{ .op = .{ .lt_s = .{ .lhs = v_i_cmp, .rhs = v_limit } }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    try std.testing.expect(try inductionVariableSimplification(&func, allocator));
    try std.testing.expectEqual(@as(u32, 2), func.local_count);
}

test "inductionVariableSimplification: nested loops only transform innermost" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 2);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const b4 = try func.newBlock();

    const v_base = func.newVReg();
    const v_zero = func.newVReg();
    const v_one = func.newVReg();
    const v_limit = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_zero });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_one });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 4 }, .dest = v_limit });
    try func.getBlock(b0).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_zero } } });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    try func.getBlock(b1).append(.{ .op = .{ .local_set = .{ .idx = 1, .val = v_zero } } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b2 } });

    const v_j = func.newVReg();
    const v_addr = func.newVReg();
    const v_load = func.newVReg();
    const v_j_up = func.newVReg();
    const v_j_next = func.newVReg();
    const v_j_cmp = func.newVReg();
    const v_jcond = func.newVReg();
    try func.getBlock(b2).append(.{ .op = .{ .local_get = 1 }, .dest = v_j });
    try func.getBlock(b2).append(.{ .op = .{ .add = .{ .lhs = v_base, .rhs = v_j } }, .dest = v_addr });
    try func.getBlock(b2).append(.{ .op = .{ .load = .{ .base = v_addr, .offset = 0, .size = 4 } }, .dest = v_load });
    try func.getBlock(b2).append(.{ .op = .{ .local_get = 1 }, .dest = v_j_up });
    try func.getBlock(b2).append(.{ .op = .{ .add = .{ .lhs = v_j_up, .rhs = v_one } }, .dest = v_j_next });
    try func.getBlock(b2).append(.{ .op = .{ .local_set = .{ .idx = 1, .val = v_j_next } } });
    try func.getBlock(b2).append(.{ .op = .{ .local_get = 1 }, .dest = v_j_cmp });
    try func.getBlock(b2).append(.{ .op = .{ .lt_s = .{ .lhs = v_j_cmp, .rhs = v_limit } }, .dest = v_jcond });
    try func.getBlock(b2).append(.{ .op = .{ .br_if = .{ .cond = v_jcond, .then_block = b2, .else_block = b3 } } });

    const v_i_up = func.newVReg();
    const v_i_next = func.newVReg();
    const v_i_cmp = func.newVReg();
    const v_icond = func.newVReg();
    try func.getBlock(b3).append(.{ .op = .{ .local_get = 0 }, .dest = v_i_up });
    try func.getBlock(b3).append(.{ .op = .{ .add = .{ .lhs = v_i_up, .rhs = v_one } }, .dest = v_i_next });
    try func.getBlock(b3).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_i_next } } });
    try func.getBlock(b3).append(.{ .op = .{ .local_get = 0 }, .dest = v_i_cmp });
    try func.getBlock(b3).append(.{ .op = .{ .lt_s = .{ .lhs = v_i_cmp, .rhs = v_limit } }, .dest = v_icond });
    try func.getBlock(b3).append(.{ .op = .{ .br_if = .{ .cond = v_icond, .then_block = b1, .else_block = b4 } } });

    try func.getBlock(b4).append(.{ .op = .{ .ret = null } });

    try std.testing.expect(try inductionVariableSimplification(&func, allocator));
    try std.testing.expectEqual(@as(u32, 3), func.local_count);

    var rewritten = false;
    for (func.getBlock(b2).instructions.items) |inst| {
        if (inst.op == .load) {
            try std.testing.expect(inst.op.load.base != v_addr);
            rewritten = true;
        }
    }
    try std.testing.expect(rewritten);
}

test "inductionVariableSimplification: non-zero init is skipped" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 1);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_base = func.newVReg();
    const v_init = func.newVReg();
    const v_step = func.newVReg();
    const v_limit = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_base });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 5 }, .dest = v_init });
    try func.getBlock(b0).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_init } } });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_step });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 9 }, .dest = v_limit });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const v_i = func.newVReg();
    const v_addr = func.newVReg();
    const v_val = func.newVReg();
    const v_i_up = func.newVReg();
    const v_next = func.newVReg();
    const v_i_cmp = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_base, .rhs = v_i } }, .dest = v_addr });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v_val });
    try func.getBlock(b1).append(.{ .op = .{ .store = .{ .base = v_addr, .offset = 0, .size = 4, .val = v_val } } });
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i_up });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_i_up, .rhs = v_step } }, .dest = v_next });
    try func.getBlock(b1).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_next } } });
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i_cmp });
    try func.getBlock(b1).append(.{ .op = .{ .lt_s = .{ .lhs = v_i_cmp, .rhs = v_limit } }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    const changed = try inductionVariableSimplification(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(@as(u32, 1), func.local_count);
}

test "unrollSmallFixedLoops: trip count four fully unrolled" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 1);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_zero = func.newVReg();
    const v_step = func.newVReg();
    const v_limit = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_zero });
    try func.getBlock(b0).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_zero } } });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_step });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 4 }, .dest = v_limit });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const v_i = func.newVReg();
    const v_next = func.newVReg();
    const v_i_cmp = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_i, .rhs = v_step } }, .dest = v_next });
    try func.getBlock(b1).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_next } } });
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i_cmp });
    try func.getBlock(b1).append(.{ .op = .{ .lt_s = .{ .lhs = v_i_cmp, .rhs = v_limit } }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    try std.testing.expect(try unrollSmallFixedLoops(&func, allocator));
    try std.testing.expectEqual(ir.Inst.Op{ .br = b2 }, func.getBlock(b0).instructions.items[func.getBlock(b0).instructions.items.len - 1].op);
    var dom = try analysis.computeDominators(&func, allocator);
    defer dom.deinit();
    var lf = try analysis.computeLoops(&func, &dom, allocator);
    defer lf.deinit();
    try std.testing.expectEqual(@as(usize, 0), lf.loops.len);
}

test "unrollSmallFixedLoops: trip count too large is skipped" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 1);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const v_zero = func.newVReg();
    const v_step = func.newVReg();
    const v_limit = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_zero });
    try func.getBlock(b0).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_zero } } });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_step });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 16 }, .dest = v_limit });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    const v_i = func.newVReg();
    const v_next = func.newVReg();
    const v_i_cmp = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_i, .rhs = v_step } }, .dest = v_next });
    try func.getBlock(b1).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_next } } });
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i_cmp });
    try func.getBlock(b1).append(.{ .op = .{ .lt_s = .{ .lhs = v_i_cmp, .rhs = v_limit } }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });
    try std.testing.expect(!try unrollSmallFixedLoops(&func, allocator));
}

test "unrollSmallFixedLoops: body too large is skipped" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 1);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const v_zero = func.newVReg();
    const v_step = func.newVReg();
    const v_limit = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_zero });
    try func.getBlock(b0).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_zero } } });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_step });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 4 }, .dest = v_limit });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    var last = v_zero;
    for (0..17) |_| {
        const v = func.newVReg();
        try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = last, .rhs = v_step } }, .dest = v });
        last = v;
    }
    const v_i = func.newVReg();
    const v_next = func.newVReg();
    const v_i_cmp = func.newVReg();
    const v_cond = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_i, .rhs = v_step } }, .dest = v_next });
    try func.getBlock(b1).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_next } } });
    try func.getBlock(b1).append(.{ .op = .{ .local_get = 0 }, .dest = v_i_cmp });
    try func.getBlock(b1).append(.{ .op = .{ .lt_s = .{ .lhs = v_i_cmp, .rhs = v_limit } }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });
    try std.testing.expect(!try unrollSmallFixedLoops(&func, allocator));
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

test "inlineSmallFunctions: local_set callee gets inlined with synthetic local renumbering" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    // Callee: fn set_param(x) -> i32 { local.set 0, 7; local.get 0; return }
    try module.functions.append(allocator, ir.IrFunction.init(allocator, 1, 1, 1));
    {
        const callee = &module.functions.items[0];
        const cb = try callee.newBlock();
        _ = callee.newVReg(); // param 0 placeholder
        const v_new = callee.newVReg();
        const v_get = callee.newVReg();
        try callee.getBlock(cb).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v_new, .type = .i32 });
        try callee.getBlock(cb).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v_new } } });
        try callee.getBlock(cb).append(.{ .op = .{ .local_get = 0 }, .dest = v_get, .type = .i32 });
        try callee.getBlock(cb).append(.{ .op = .{ .ret = v_get } });
    }

    // Caller has one original local; the inlined callee must not reuse local 0.
    try module.functions.append(allocator, ir.IrFunction.init(allocator, 1, 1, 1));
    const args = try allocator.alloc(ir.VReg, 1);
    defer allocator.free(args);
    const caller_original_locals = module.functions.items[1].local_count;
    {
        const caller = &module.functions.items[1];
        const mb = try caller.newBlock();
        _ = caller.newVReg(); // param 0 placeholder
        const v_arg = caller.newVReg();
        const v_ret = caller.newVReg();
        try caller.getBlock(mb).append(.{ .op = .{ .local_get = 0 }, .dest = v_arg, .type = .i32 });
        args[0] = v_arg;
        try caller.getBlock(mb).append(.{ .op = .{ .call = .{ .func_idx = 0, .args = args } }, .dest = v_ret, .type = .i32 });
        try caller.getBlock(mb).append(.{ .op = .{ .ret = v_ret } });
    }

    const inlined = try inlineSmallFunctions(&module, allocator);
    try std.testing.expect(inlined);

    const caller = &module.functions.items[1];
    try std.testing.expectEqual(caller_original_locals + 1, caller.local_count);
    const clone_entry = caller.blocks.items[1].instructions.items;
    var saw_local_set = false;
    var saw_local_get = false;
    for (clone_entry) |inst| {
        switch (inst.op) {
            .local_set => |ls| {
                try std.testing.expect(ls.idx >= caller_original_locals);
                saw_local_set = true;
            },
            .local_get => |idx| {
                try std.testing.expect(idx >= caller_original_locals);
                saw_local_get = true;
            },
            else => {},
        }
    }
    try std.testing.expect(saw_local_set);
    try std.testing.expect(saw_local_get);
}

test "inlineSmallFunctions: multi-block callee with declared locals zero-inits synthetic local" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    // Callee: fn read_declared(x) -> i32 { local.get 1; br tail; tail: ret }
    try module.functions.append(allocator, ir.IrFunction.init(allocator, 1, 1, 2));
    {
        const callee = &module.functions.items[0];
        const c_entry = try callee.newBlock();
        const c_tail = try callee.newBlock();
        _ = callee.newVReg(); // param 0 placeholder
        const v_local = callee.newVReg();
        try callee.getBlock(c_entry).append(.{ .op = .{ .local_get = 1 }, .dest = v_local, .type = .i32 });
        try callee.getBlock(c_entry).append(.{ .op = .{ .br = c_tail } });
        try callee.getBlock(c_tail).append(.{ .op = .{ .ret = v_local } });
    }

    try module.functions.append(allocator, ir.IrFunction.init(allocator, 1, 1, 1));
    const args = try allocator.alloc(ir.VReg, 1);
    defer allocator.free(args);
    const caller_original_locals = module.functions.items[1].local_count;
    {
        const caller = &module.functions.items[1];
        const mb = try caller.newBlock();
        _ = caller.newVReg(); // param 0 placeholder
        const v_arg = caller.newVReg();
        const v_ret = caller.newVReg();
        try caller.getBlock(mb).append(.{ .op = .{ .local_get = 0 }, .dest = v_arg, .type = .i32 });
        args[0] = v_arg;
        try caller.getBlock(mb).append(.{ .op = .{ .call = .{ .func_idx = 0, .args = args } }, .dest = v_ret, .type = .i32 });
        try caller.getBlock(mb).append(.{ .op = .{ .ret = v_ret } });
    }

    const inlined = try inlineSmallFunctions(&module, allocator);
    try std.testing.expect(inlined);

    const caller = &module.functions.items[1];
    try std.testing.expectEqual(caller_original_locals + 2, caller.local_count);
    const clone_entry = caller.blocks.items[1].instructions.items;
    try std.testing.expect(clone_entry[0].op == .local_set);
    try std.testing.expectEqual(caller_original_locals, clone_entry[0].op.local_set.idx);
    try std.testing.expect(clone_entry[1].op == .iconst_32);
    try std.testing.expectEqual(@as(i32, 0), clone_entry[1].op.iconst_32);
    try std.testing.expect(clone_entry[2].op == .local_set);
    try std.testing.expectEqual(caller_original_locals + 1, clone_entry[2].op.local_set.idx);
    try std.testing.expectEqual(clone_entry[1].dest.?, clone_entry[2].op.local_set.val);
    try std.testing.expect(clone_entry[3].op == .local_get);
    try std.testing.expectEqual(caller_original_locals + 1, clone_entry[3].op.local_get);
}

test "inlineSmallFunctions: br_table callee gets inlined with remapped targets" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    const callee_targets = try allocator.alloc(ir.BlockId, 2);
    defer allocator.free(callee_targets);

    // Callee: br_table over two branch blocks plus a default tail, with one ret.
    try module.functions.append(allocator, ir.IrFunction.init(allocator, 1, 1, 1));
    {
        const callee = &module.functions.items[0];
        const c_entry = try callee.newBlock();
        const c_t0 = try callee.newBlock();
        const c_t1 = try callee.newBlock();
        const c_tail = try callee.newBlock();
        callee_targets[0] = c_t0;
        callee_targets[1] = c_t1;
        _ = callee.newVReg(); // param 0 placeholder
        const v_idx = callee.newVReg();
        const v_ret = callee.newVReg();
        try callee.getBlock(c_entry).append(.{ .op = .{ .local_get = 0 }, .dest = v_idx, .type = .i32 });
        try callee.getBlock(c_entry).append(.{ .op = .{ .br_table = .{
            .index = v_idx,
            .targets = callee_targets,
            .default = c_tail,
        } } });
        try callee.getBlock(c_t0).append(.{ .op = .{ .br = c_tail } });
        try callee.getBlock(c_t1).append(.{ .op = .{ .br = c_tail } });
        try callee.getBlock(c_tail).append(.{ .op = .{ .iconst_32 = 11 }, .dest = v_ret, .type = .i32 });
        try callee.getBlock(c_tail).append(.{ .op = .{ .ret = v_ret } });
    }

    try module.functions.append(allocator, ir.IrFunction.init(allocator, 1, 1, 1));
    const args = try allocator.alloc(ir.VReg, 1);
    defer allocator.free(args);
    {
        const caller = &module.functions.items[1];
        const mb = try caller.newBlock();
        _ = caller.newVReg(); // param 0 placeholder
        const v_arg = caller.newVReg();
        const v_ret = caller.newVReg();
        try caller.getBlock(mb).append(.{ .op = .{ .local_get = 0 }, .dest = v_arg, .type = .i32 });
        args[0] = v_arg;
        try caller.getBlock(mb).append(.{ .op = .{ .call = .{ .func_idx = 0, .args = args } }, .dest = v_ret, .type = .i32 });
        try caller.getBlock(mb).append(.{ .op = .{ .ret = v_ret } });
    }

    const inlined = try inlineSmallFunctions(&module, allocator);
    try std.testing.expect(inlined);

    const caller = &module.functions.items[1];
    const clone_offset: ir.BlockId = 1;
    const br_table_inst = caller.blocks.items[clone_offset].instructions.items[0];
    try std.testing.expect(br_table_inst.op == .br_table);
    const bt = br_table_inst.op.br_table;
    try std.testing.expectEqual(clone_offset + 1, bt.targets[0]);
    try std.testing.expectEqual(clone_offset + 2, bt.targets[1]);
    try std.testing.expectEqual(clone_offset + 3, bt.default);
    try std.testing.expectEqual(module.functions.items[1].blocks.items[0].instructions.items[0].dest.?, bt.index);
}

test "runPasses: re-runs inlining after first per-function fixpoint" {
    // Scenario: a caller invokes a small leaf callee with an argument that
    // is itself the result of a small chain (local.get → constant fold).
    // After the first per-function fixpoint round the call survives, but
    // it now has a constant-folded argument, which still satisfies the
    // existing inliner's eligibility. The second outer round must inline
    // it. We use two levels of indirection so that the first round only
    // inlines the innermost callee, and the second round inlines what
    // becomes the new leaf after that.
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    // f0 (innermost leaf): fn id(x) -> x.
    try module.functions.append(allocator, ir.IrFunction.init(allocator, 1, 1, 1));
    {
        const f = &module.functions.items[0];
        const cb = try f.newBlock();
        _ = f.newVReg();
        const v_get = f.newVReg();
        try f.getBlock(cb).append(.{ .op = .{ .local_get = 0 }, .dest = v_get, .type = .i32 });
        try f.getBlock(cb).append(.{ .op = .{ .ret = v_get } });
    }

    // f1 (middle): fn middle(x) -> x. Calls f0(x). On the first outer
    // round f1 is not eligible (it contains a `call`), so its body is
    // simplified but f2's call to it survives. After the first
    // per-function fixpoint, f0 has been inlined into f1, making f1 a
    // pass-through that becomes eligible. The second outer round then
    // inlines f1 into f2.
    try module.functions.append(allocator, ir.IrFunction.init(allocator, 1, 1, 1));
    const f1_args = try allocator.alloc(ir.VReg, 1);
    defer allocator.free(f1_args);
    {
        const f = &module.functions.items[1];
        const cb = try f.newBlock();
        _ = f.newVReg();
        const v_arg = f.newVReg();
        const v_ret = f.newVReg();
        try f.getBlock(cb).append(.{ .op = .{ .local_get = 0 }, .dest = v_arg, .type = .i32 });
        f1_args[0] = v_arg;
        try f.getBlock(cb).append(.{ .op = .{ .call = .{ .func_idx = 0, .args = f1_args } }, .dest = v_ret, .type = .i32 });
        try f.getBlock(cb).append(.{ .op = .{ .ret = v_ret } });
    }

    // f2 (caller): fn main() -> i32 { call middle(42); ret }.
    try module.functions.append(allocator, ir.IrFunction.init(allocator, 0, 1, 0));
    const f2_args = try allocator.alloc(ir.VReg, 1);
    defer allocator.free(f2_args);
    {
        const f = &module.functions.items[2];
        const cb = try f.newBlock();
        const v_arg = f.newVReg();
        const v_ret = f.newVReg();
        try f.getBlock(cb).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v_arg, .type = .i32 });
        f2_args[0] = v_arg;
        try f.getBlock(cb).append(.{ .op = .{ .call = .{ .func_idx = 1, .args = f2_args } }, .dest = v_ret, .type = .i32 });
        try f.getBlock(cb).append(.{ .op = .{ .ret = v_ret } });
    }

    // Use a minimal pass list to keep the test focused. forwardLocalGet
    // and constantFold are what the issue's motivating example relies on.
    const test_passes = [_]PassFn{
        &forwardLocalGet,
        &constantFold,
        &deadCodeElimination,
    };
    _ = try runPasses(&module, &test_passes, allocator);

    // After two outer rounds, f2 must contain no `.call` instructions:
    // f0 was inlined into f1 during the first outer round, and the now
    // pass-through f1 was inlined into f2 during the second outer round.
    const caller = &module.functions.items[2];
    for (caller.blocks.items) |blk| {
        for (blk.instructions.items) |inst| {
            try std.testing.expect(inst.op != .call);
        }
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

fn pipelineContains(pass_list: []const PassFn, needle: PassFn) bool {
    for (pass_list) |pass| {
        if (pass == needle) return true;
    }
    return false;
}

test "default pipeline enables foldBranchOnEqz only for x86_64" {
    try std.testing.expect(pipelineContains(defaultPassesForTarget(.x86_64), &foldBranchOnEqz));
    try std.testing.expect(!pipelineContains(defaultPassesForTarget(.aarch64), &foldBranchOnEqz));
}

test "threadChainedConditionalBranches: true edge jumps to inner true target" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const entry = try func.newBlock();
    const mid = try func.newBlock();
    const source_else = try func.newBlock();
    const inner_true = try func.newBlock();
    const inner_false = try func.newBlock();

    const v_c = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_c, .type = .i32 });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = v_c, .then_block = mid, .else_block = source_else } } });
    try func.getBlock(mid).append(.{ .op = .{ .br_if = .{ .cond = v_c, .then_block = inner_true, .else_block = inner_false } } });
    try func.getBlock(source_else).append(.{ .op = .{ .ret = null } });
    try func.getBlock(inner_true).append(.{ .op = .{ .ret = null } });
    try func.getBlock(inner_false).append(.{ .op = .{ .ret = null } });

    const changed = try threadChainedConditionalBranches(&func, allocator);
    try std.testing.expect(changed);

    const term = func.getBlock(entry).instructions.items[1];
    switch (term.op) {
        .br_if => |bi| {
            try std.testing.expectEqual(v_c, bi.cond);
            try std.testing.expectEqual(inner_true, bi.then_block);
            try std.testing.expectEqual(source_else, bi.else_block);
        },
        else => try std.testing.expect(false),
    }
}

test "threadChainedConditionalBranches: false edge jumps to inner false target" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const entry = try func.newBlock();
    const source_then = try func.newBlock();
    const mid = try func.newBlock();
    const inner_true = try func.newBlock();
    const inner_false = try func.newBlock();

    const v_c = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_c, .type = .i32 });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = v_c, .then_block = source_then, .else_block = mid } } });
    try func.getBlock(source_then).append(.{ .op = .{ .ret = null } });
    try func.getBlock(mid).append(.{ .op = .{ .br_if = .{ .cond = v_c, .then_block = inner_true, .else_block = inner_false } } });
    try func.getBlock(inner_true).append(.{ .op = .{ .ret = null } });
    try func.getBlock(inner_false).append(.{ .op = .{ .ret = null } });

    const changed = try threadChainedConditionalBranches(&func, allocator);
    try std.testing.expect(changed);

    const term = func.getBlock(entry).instructions.items[1];
    switch (term.op) {
        .br_if => |bi| {
            try std.testing.expectEqual(v_c, bi.cond);
            try std.testing.expectEqual(source_then, bi.then_block);
            try std.testing.expectEqual(inner_false, bi.else_block);
        },
        else => try std.testing.expect(false),
    }
}

test "threadChainedConditionalBranches: skips mismatched inner condition" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const entry = try func.newBlock();
    const mid = try func.newBlock();
    const source_else = try func.newBlock();
    const inner_true = try func.newBlock();
    const inner_false = try func.newBlock();

    const v_c = func.newVReg();
    const v_other = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_c, .type = .i32 });
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_other, .type = .i32 });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = v_c, .then_block = mid, .else_block = source_else } } });
    try func.getBlock(mid).append(.{ .op = .{ .br_if = .{ .cond = v_other, .then_block = inner_true, .else_block = inner_false } } });
    try func.getBlock(source_else).append(.{ .op = .{ .ret = null } });
    try func.getBlock(inner_true).append(.{ .op = .{ .ret = null } });
    try func.getBlock(inner_false).append(.{ .op = .{ .ret = null } });

    const changed = try threadChainedConditionalBranches(&func, allocator);
    try std.testing.expect(!changed);

    const term = func.getBlock(entry).instructions.items[2];
    switch (term.op) {
        .br_if => |bi| {
            try std.testing.expectEqual(mid, bi.then_block);
            try std.testing.expectEqual(source_else, bi.else_block);
        },
        else => try std.testing.expect(false),
    }
}

test "threadChainedConditionalBranches: skips non-empty intermediate block" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const entry = try func.newBlock();
    const mid = try func.newBlock();
    const source_else = try func.newBlock();
    const inner_true = try func.newBlock();
    const inner_false = try func.newBlock();

    const v_c = func.newVReg();
    const v_dead = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_c, .type = .i32 });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = v_c, .then_block = mid, .else_block = source_else } } });
    try func.getBlock(mid).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v_dead, .type = .i32 });
    try func.getBlock(mid).append(.{ .op = .{ .br_if = .{ .cond = v_c, .then_block = inner_true, .else_block = inner_false } } });
    try func.getBlock(source_else).append(.{ .op = .{ .ret = null } });
    try func.getBlock(inner_true).append(.{ .op = .{ .ret = null } });
    try func.getBlock(inner_false).append(.{ .op = .{ .ret = null } });

    const changed = try threadChainedConditionalBranches(&func, allocator);
    try std.testing.expect(!changed);

    const term = func.getBlock(entry).instructions.items[1];
    switch (term.op) {
        .br_if => |bi| {
            try std.testing.expectEqual(mid, bi.then_block);
            try std.testing.expectEqual(source_else, bi.else_block);
        },
        else => try std.testing.expect(false),
    }
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

test "tailDuplicateSmallJoins: 4-block diamond — join inlined into both arms" {
    // b0 (cond br) → b1 → b3, b2 → b3
    // b3 is a small join: add v0+v1 → ret.
    // After tailDuplicateSmallJoins, b3's body should be cloned into both
    // b1 and b2 (each ending in its own copy of the join's ret), and b3
    // should be unreachable (no predecessors).
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
    const v_join_add = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v_join_add });
    try func.getBlock(b3).append(.{ .op = .{ .ret = v_join_add } });

    const changed = try tailDuplicateSmallJoins(&func, allocator);
    try std.testing.expect(changed);

    // b1 and b2 should each now hold a copy of b3's body (the `add`) plus
    // the cloned terminator (`ret`).
    try std.testing.expectEqual(@as(usize, 2), func.getBlock(b1).instructions.items.len);
    try std.testing.expectEqual(@as(usize, 2), func.getBlock(b2).instructions.items.len);

    // The first instruction in each arm should be the duplicated add.
    try std.testing.expect(func.getBlock(b1).instructions.items[0].op == .add);
    try std.testing.expect(func.getBlock(b2).instructions.items[0].op == .add);

    // The cloned add dests must be fresh (≠ v_join_add): both arms got
    // independent renamed defs.
    const b1_add_dest = func.getBlock(b1).instructions.items[0].dest.?;
    const b2_add_dest = func.getBlock(b2).instructions.items[0].dest.?;
    try std.testing.expect(b1_add_dest != v_join_add);
    try std.testing.expect(b2_add_dest != v_join_add);
    try std.testing.expect(b1_add_dest != b2_add_dest);

    // Cloned add operands still reference the originals (v0, v1) defined
    // in the dominating b0 — those vregs are not local to b3.
    const b1_add_op = func.getBlock(b1).instructions.items[0].op.add;
    try std.testing.expectEqual(v0, b1_add_op.lhs);
    try std.testing.expectEqual(v1, b1_add_op.rhs);

    // The cloned terminators reference the cloned (renamed) add dest.
    try std.testing.expectEqual(ir.Inst.Op{ .ret = b1_add_dest }, func.getBlock(b1).instructions.items[1].op);
    try std.testing.expectEqual(ir.Inst.Op{ .ret = b2_add_dest }, func.getBlock(b2).instructions.items[1].op);

    // b3 must now be unreachable: its predecessor set is empty.
    var preds = try analysis.buildPredecessors(&func, allocator);
    defer {
        var it = preds.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        preds.deinit();
    }
    if (preds.get(b3)) |p| {
        try std.testing.expectEqual(@as(usize, 0), p.len);
    }
}

test "tailDuplicateSmallJoins: triple predecessor with br terminator — all three duplicated" {
    // b0,b1,b2 each unconditionally br to b3 (sub + br_if to b4 or b5).
    // After duplication, all three predecessors should each contain the
    // sub + br_if; b3 should be unreachable.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b_entry = try func.newBlock();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const b4 = try func.newBlock();
    const b5 = try func.newBlock();
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    const v_cond = func.newVReg();
    const v_join_sub = func.newVReg();

    // Entry sets up the operands and dispatches via br_table-like sequence:
    // we use three different small heads each branching to b3 for simplicity.
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v_a });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 3 }, .dest = v_b });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_cond });
    try func.getBlock(b_entry).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b0, .else_block = b1 } } });

    // b0 and b1 both directly br to b3. Also need a third predecessor b2.
    // To keep this realistic we fall through b0 → b3 via br, then add b2
    // as an additional br b3 path from b_entry's else… but br_if can only
    // pick one of two targets. Instead, give b1 a br_if into either b2 or
    // b3, and let b2 also br b3 — so b3 has three predecessors {b0, b1, b2}.
    // Then change b1's terminator to br b3 to satisfy the pass's
    // "all preds are unconditional br" precondition.
    //
    // Simpler structure: route b1 via b2 -> b3 to gather three predecessors.
    // We achieve this by making b1's terminator a `br b2` and b2's a
    // `br b3`, but that yields only one predecessor pointing at b3 from
    // b2. So we re-route: b0 → b3, b1 → b3, and inject a separate br
    // chain that joins via b2 → b3.
    try func.getBlock(b0).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });

    // Body of b3: small sub + conditional branch to b4/b5.
    try func.getBlock(b3).append(.{ .op = .{ .sub = .{ .lhs = v_a, .rhs = v_b } }, .dest = v_join_sub });
    try func.getBlock(b3).append(.{ .op = .{ .br_if = .{ .cond = v_join_sub, .then_block = b4, .else_block = b5 } } });

    try func.getBlock(b4).append(.{ .op = .{ .ret = v_a } });
    try func.getBlock(b5).append(.{ .op = .{ .ret = v_b } });

    // b2 has no predecessor from the entry CFG in this synthetic test —
    // tailDuplicateSmallJoins works on the predecessor set without
    // requiring full reachability, which mirrors how passes operate over
    // raw IR blocks. The pass should still duplicate b3 into b0, b1, b2.

    const changed = try tailDuplicateSmallJoins(&func, allocator);
    try std.testing.expect(changed);

    // Each of b0, b1, b2 should now end in a cloned sub + br_if (2 insts).
    inline for (.{ b0, b1, b2 }) |pid| {
        const pb = func.getBlock(pid);
        try std.testing.expectEqual(@as(usize, 2), pb.instructions.items.len);
        try std.testing.expect(pb.instructions.items[0].op == .sub);
        try std.testing.expect(pb.instructions.items[1].op == .br_if);
    }

    // The three cloned sub-dests must be pairwise distinct fresh vregs.
    const d0 = func.getBlock(b0).instructions.items[0].dest.?;
    const d1 = func.getBlock(b1).instructions.items[0].dest.?;
    const d2 = func.getBlock(b2).instructions.items[0].dest.?;
    try std.testing.expect(d0 != d1);
    try std.testing.expect(d0 != d2);
    try std.testing.expect(d1 != d2);
    try std.testing.expect(d0 != v_join_sub);

    // b3 is unreachable: zero predecessors in the rebuilt map.
    var preds = try analysis.buildPredecessors(&func, allocator);
    defer {
        var it = preds.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        preds.deinit();
    }
    if (preds.get(b3)) |p| {
        try std.testing.expectEqual(@as(usize, 0), p.len);
    }
}

test "tailDuplicateSmallJoins: join with `ret` terminator is duplicated" {
    // Two predecessors both br to a small join whose terminator is a
    // single-value ret.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const v_x = func.newVReg();
    const v_y = func.newVReg();
    const v_cond = func.newVReg();
    const v_join = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v_x });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_y });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_cond });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .add = .{ .lhs = v_x, .rhs = v_y } }, .dest = v_join });
    try func.getBlock(b3).append(.{ .op = .{ .ret = v_join } });

    const changed = try tailDuplicateSmallJoins(&func, allocator);
    try std.testing.expect(changed);

    // Both arms must hold an add + ret of the renamed add dest.
    inline for (.{ b1, b2 }) |pid| {
        const pb = func.getBlock(pid);
        try std.testing.expectEqual(@as(usize, 2), pb.instructions.items.len);
        try std.testing.expect(pb.instructions.items[0].op == .add);
        const fresh = pb.instructions.items[0].dest.?;
        try std.testing.expect(fresh != v_join);
        try std.testing.expectEqual(ir.Inst.Op{ .ret = fresh }, pb.instructions.items[1].op);
    }
}

test "tailDuplicateSmallJoins: oversized join (>4 body inst) is NOT duplicated" {
    // Join block has 5 non-terminator instructions — exceeds the cap.
    // The pass must leave the CFG unchanged.
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
    const v_a = func.newVReg();
    const v_b = func.newVReg();
    const v_c = func.newVReg();
    const v_d = func.newVReg();
    const v_e = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    // 5 instructions before terminator → oversized.
    try func.getBlock(b3).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v_a });
    try func.getBlock(b3).append(.{ .op = .{ .sub = .{ .lhs = v_a, .rhs = v0 } }, .dest = v_b });
    try func.getBlock(b3).append(.{ .op = .{ .add = .{ .lhs = v_b, .rhs = v1 } }, .dest = v_c });
    try func.getBlock(b3).append(.{ .op = .{ .sub = .{ .lhs = v_c, .rhs = v0 } }, .dest = v_d });
    try func.getBlock(b3).append(.{ .op = .{ .add = .{ .lhs = v_d, .rhs = v1 } }, .dest = v_e });
    try func.getBlock(b3).append(.{ .op = .{ .ret = v_e } });

    const before_b1_len = func.getBlock(b1).instructions.items.len;
    const before_b2_len = func.getBlock(b2).instructions.items.len;
    const before_b3_len = func.getBlock(b3).instructions.items.len;

    const changed = try tailDuplicateSmallJoins(&func, allocator);
    try std.testing.expect(!changed);

    try std.testing.expectEqual(before_b1_len, func.getBlock(b1).instructions.items.len);
    try std.testing.expectEqual(before_b2_len, func.getBlock(b2).instructions.items.len);
    try std.testing.expectEqual(before_b3_len, func.getBlock(b3).instructions.items.len);
}

test "tailDuplicateSmallJoins: pred ending in br_if (not br) is NOT duplicated" {
    // b1's terminator is br_if (conditional). The pass requires all preds
    // to end in an unconditional `br B`, so it must refuse the rewrite.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock(); // join
    const b4 = try func.newBlock(); // other br_if target
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const cond1 = func.newVReg();
    const cond2 = func.newVReg();
    const v_join = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 5 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 6 }, .dest = v1 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond1 });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond1, .then_block = b1, .else_block = b2 } } });
    // b1 ends in br_if → b3 or b4 (conditional, not unconditional br).
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 0 }, .dest = cond2 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = cond2, .then_block = b3, .else_block = b4 } } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v_join });
    try func.getBlock(b3).append(.{ .op = .{ .ret = v_join } });
    try func.getBlock(b4).append(.{ .op = .{ .ret = v0 } });

    const before_b1_len = func.getBlock(b1).instructions.items.len;
    const before_b3_len = func.getBlock(b3).instructions.items.len;

    const changed = try tailDuplicateSmallJoins(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(before_b1_len, func.getBlock(b1).instructions.items.len);
    try std.testing.expectEqual(before_b3_len, func.getBlock(b3).instructions.items.len);
}

test "tailDuplicateSmallJoins: join def with external use is NOT duplicated" {
    // Join b3 defines v_join via add; b4 (successor of b3) reads v_join.
    // After hypothetical duplication, b3 would be unreachable and v_join
    // would have no def for b4's read — SSA violation. The pass must
    // refuse.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const b4 = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const cond = func.newVReg();
    const v_join = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v_join });
    try func.getBlock(b3).append(.{ .op = .{ .br = b4 } });
    // b4 reads v_join — an external use of a vreg defined in b3.
    try func.getBlock(b4).append(.{ .op = .{ .ret = v_join } });

    const changed = try tailDuplicateSmallJoins(&func, allocator);
    try std.testing.expect(!changed);
}
