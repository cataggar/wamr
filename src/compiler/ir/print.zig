//! IR pretty-printer.
//!
//! Produces a human-readable text rendering of `IrFunction` /
//! `IrModule` values, used for debugging optimisation pipeline behaviour
//! (e.g. `wamrc --dump-ir-after=<pass>`) and unit tests.
//!
//! Output sketch:
//!
//!     func #5 my_func (params=2, results=1, locals=4, vregs=12) {
//!     b0:
//!       v0 = iconst_32.i32 42
//!       local_set local[2], v0
//!       v1 = local_get.i32 local[2]
//!       v2 = add.i32 v0, v1
//!       store base=v3, offset=8, size=4, val=v2
//!       br_if cond=v4, then=b1, else=b2
//!     b1  ; preds=[b0]:
//!       ret v2
//!     b2  ; preds=[b0]:
//!       unreachable
//!     }
//!
//! Goals:
//!   - Scalar Ops (load/store, local_get/set, br_*, call, phi, …) print
//!     with stable, grep-able field names — these are what we use to
//!     diagnose CoreMark reload spam (issue #467).
//!   - SIMD/exotic Ops fall back to `<tag>(vregs=[v…])` rendering so the
//!     printer covers the entire Op union without enumerating every
//!     SIMD variant.
//!
//! Format is NOT round-trippable — focus is diagnostic utility.

const std = @import("std");
const ir = @import("ir.zig");

pub const Error = std.Io.Writer.Error;

/// Pretty-print an entire module. Functions are numbered using their
/// module-local index (i.e. excluding imports — same indexing as
/// `IrModule.functions`).
pub fn formatModule(module: *const ir.IrModule, w: *std.Io.Writer) Error!void {
    try w.print("; module: {d} function(s), import_count={d}\n", .{
        module.functions.items.len,
        module.import_count,
    });
    for (module.functions.items, 0..) |*func, idx| {
        try formatFunc(func, @intCast(idx), w);
        try w.writeByte('\n');
    }
}

/// Pretty-print a single function. `func_index` is the caller-supplied
/// module-local index; printed in the header for cross-referencing.
pub fn formatFunc(func: *const ir.IrFunction, func_index: u32, w: *std.Io.Writer) Error!void {
    const name = func.name orelse "<unnamed>";
    try w.print("func #{d} {s} (params={d}, results={d}, locals={d}, vregs={d}) {{\n", .{
        func_index,
        name,
        func.param_count,
        func.result_count,
        func.local_count,
        func.next_vreg,
    });
    for (func.blocks.items) |*block| {
        try formatBlock(block, w);
    }
    try w.writeAll("}\n");
}

/// Pretty-print a single basic block (header + body, terminated by a
/// newline). The header includes a predecessor list when non-empty.
pub fn formatBlock(block: *const ir.BasicBlock, w: *std.Io.Writer) Error!void {
    try w.print("b{d}", .{block.id});
    if (block.predecessors.items.len > 0) {
        try w.writeAll("  ; preds=[");
        for (block.predecessors.items, 0..) |p, i| {
            if (i > 0) try w.writeAll(", ");
            try w.print("b{d}", .{p});
        }
        try w.writeByte(']');
    }
    try w.writeAll(":\n");
    for (block.instructions.items) |inst| {
        try w.writeAll("  ");
        try formatInst(inst, w);
        try w.writeByte('\n');
    }
}

/// Pretty-print a single instruction (no trailing newline).
pub fn formatInst(inst: ir.Inst, w: *std.Io.Writer) Error!void {
    if (inst.dest) |d| try w.print("v{d} = ", .{d});

    try w.writeAll(@tagName(inst.op));
    // Only emit the type suffix when the instruction produces a value
    // (i.e. has a dest VReg). For sinks/terminators/stores the `type`
    // field carries its default and printing it would be misleading.
    if (inst.dest != null and inst.type != .void) try w.print(".{s}", .{@tagName(inst.type)});

    try formatPayload(inst.op, w);
}

fn formatPayload(op: ir.Inst.Op, w: *std.Io.Writer) Error!void {
    switch (op) {
        // Constants
        .iconst_32 => |v| try w.print(" {d}", .{v}),
        .iconst_64 => |v| try w.print(" {d}", .{v}),
        .fconst_32 => |v| try w.print(" {d}", .{v}),
        .fconst_64 => |v| try w.print(" {d}", .{v}),
        .v128_const => |v| try w.print(" 0x{x:0>32}", .{v}),

        // Scalar binary ops (lhs/rhs)
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
        => |bin| try w.print(" v{d}, v{d}", .{ bin.lhs, bin.rhs }),

        // Scalar unary ops (single VReg payload)
        .clz,
        .ctz,
        .popcnt,
        .eqz,
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
        .wrap_i64,
        .extend_i32_s,
        .extend_i32_u,
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
        .memory_grow,
        => |v| try w.print(" v{d}", .{v}),

        // Locals / globals
        .local_get => |idx| try w.print(" local[{d}]", .{idx}),
        .local_set => |ls| try w.print(" local[{d}], v{d}", .{ ls.idx, ls.val }),
        .global_get => |idx| try w.print(" global[{d}]", .{idx}),
        .global_set => |gs| try w.print(" global[{d}], v{d}", .{ gs.idx, gs.val }),

        // Memory
        .load => |ld| {
            try w.print(" base=v{d}, offset={d}, size={d}", .{ ld.base, ld.offset, ld.size });
            if (ld.sign_extend) try w.writeAll(", sext");
            if (ld.bounds_known) try w.writeAll(", bounds_known");
            if (ld.checked_end != 0) try w.print(", checked_end={d}", .{ld.checked_end});
        },
        .store => |st| {
            try w.print(" base=v{d}, offset={d}, size={d}, val=v{d}", .{ st.base, st.offset, st.size, st.val });
            if (st.bounds_known) try w.writeAll(", bounds_known");
            if (st.checked_end != 0) try w.print(", checked_end={d}", .{st.checked_end});
        },

        // Control flow
        .br => |t| try w.print(" b{d}", .{t}),
        .br_if => |bi| try w.print(" cond=v{d}, then=b{d}, else=b{d}", .{ bi.cond, bi.then_block, bi.else_block }),
        .br_table => |bt| {
            try w.print(" index=v{d}, default=b{d}, targets=[", .{ bt.index, bt.default });
            for (bt.targets, 0..) |t, i| {
                if (i > 0) try w.writeAll(", ");
                try w.print("b{d}", .{t});
            }
            try w.writeByte(']');
        },
        .ret => |maybe| if (maybe) |v| try w.print(" v{d}", .{v}),
        .ret_multi => |vregs| {
            try w.writeAll(" [");
            for (vregs, 0..) |v, i| {
                if (i > 0) try w.writeAll(", ");
                try w.print("v{d}", .{v});
            }
            try w.writeByte(']');
        },
        .@"unreachable" => {},

        // Calls
        .call => |c| {
            try w.print(" func={d}", .{c.func_idx});
            if (c.tail) try w.writeAll(", tail");
            try w.writeAll(", args=[");
            for (c.args, 0..) |a, i| {
                if (i > 0) try w.writeAll(", ");
                try w.print("v{d}", .{a});
            }
            try w.writeByte(']');
            if (c.extra_results > 0) try w.print(", extra_results={d}", .{c.extra_results});
        },
        .call_indirect => |ci| {
            try w.print(" type={d}, elem_idx=v{d}, table={d}", .{ ci.type_idx, ci.elem_idx, ci.table_idx });
            if (ci.tail) try w.writeAll(", tail");
            try w.writeAll(", args=[");
            for (ci.args, 0..) |a, i| {
                if (i > 0) try w.writeAll(", ");
                try w.print("v{d}", .{a});
            }
            try w.writeByte(']');
            if (ci.extra_results > 0) try w.print(", extra_results={d}", .{ci.extra_results});
        },
        .call_ref => |cr| {
            try w.print(" type={d}, func_ref=v{d}", .{ cr.type_idx, cr.func_ref });
            if (cr.tail) try w.writeAll(", tail");
            try w.writeAll(", args=[");
            for (cr.args, 0..) |a, i| {
                if (i > 0) try w.writeAll(", ");
                try w.print("v{d}", .{a});
            }
            try w.writeByte(']');
            if (cr.extra_results > 0) try w.print(", extra_results={d}", .{cr.extra_results});
        },
        .call_result => |idx| try w.print(" #{d}", .{idx}),

        // Select / phi
        .select => |s| try w.print(" cond=v{d}, if_true=v{d}, if_false=v{d}", .{ s.cond, s.if_true, s.if_false }),
        .phi => |edges| {
            try w.writeAll(" [");
            for (edges, 0..) |e, i| {
                if (i > 0) try w.writeAll(", ");
                try w.print("b{d}:v{d}", .{ e.block, e.val });
            }
            try w.writeByte(']');
        },
        .parallel_copy => |pairs| {
            try w.writeAll(" [");
            for (pairs, 0..) |p, i| {
                if (i > 0) try w.writeAll(", ");
                try w.print("v{d}<-v{d}:{s}", .{ p.dst, p.src, @tagName(p.ty) });
            }
            try w.writeByte(']');
        },

        // Atomic
        .atomic_load => |a| try w.print(" base=v{d}, offset={d}, size={d}", .{ a.base, a.offset, a.size }),
        .atomic_store => |a| try w.print(" base=v{d}, offset={d}, size={d}, val=v{d}", .{ a.base, a.offset, a.size, a.val }),
        .atomic_rmw => |a| try w.print(" base=v{d}, offset={d}, size={d}, val=v{d}, op={s}", .{ a.base, a.offset, a.size, a.val, @tagName(a.op) }),
        .atomic_cmpxchg => |a| try w.print(" base=v{d}, offset={d}, size={d}, expected=v{d}, replacement=v{d}", .{ a.base, a.offset, a.size, a.expected, a.replacement }),
        .atomic_fence => {},
        .atomic_notify => |a| try w.print(" base=v{d}, offset={d}, count=v{d}", .{ a.base, a.offset, a.count }),
        .atomic_wait => |a| try w.print(" base=v{d}, offset={d}, expected=v{d}, timeout=v{d}, size={d}", .{ a.base, a.offset, a.expected, a.timeout, a.size }),

        // Bulk memory / tables
        .memory_copy => |m| try w.print(" dst=v{d}, src=v{d}, len=v{d}", .{ m.dst, m.src, m.len }),
        .memory_fill => |m| try w.print(" dst=v{d}, val=v{d}, len=v{d}", .{ m.dst, m.val, m.len }),
        .memory_init => |m| try w.print(" seg={d}, dst=v{d}, src=v{d}, len=v{d}", .{ m.seg_idx, m.dst, m.src, m.len }),
        .data_drop => |idx| try w.print(" #{d}", .{idx}),
        .memory_size => {},
        .table_size => |idx| try w.print(" table[{d}]", .{idx}),
        .table_get => |t| try w.print(" table[{d}], idx=v{d}", .{ t.table_idx, t.idx }),
        .table_set => |t| try w.print(" table[{d}], idx=v{d}, val=v{d}", .{ t.table_idx, t.idx, t.val }),
        .table_grow => |t| try w.print(" table[{d}], init=v{d}, delta=v{d}", .{ t.table_idx, t.init, t.delta }),
        .table_init => |t| try w.print(" seg={d}, table[{d}], dst=v{d}, src=v{d}, len=v{d}", .{ t.seg_idx, t.table_idx, t.dst, t.src, t.len }),
        .elem_drop => |idx| try w.print(" #{d}", .{idx}),
        .ref_func => |idx| try w.print(" #{d}", .{idx}),

        // SIMD — render via the generic operand list. This keeps coverage
        // broad without per-variant boilerplate; precise SIMD diagnostics
        // can grow later if a use case arises.
        .v128_not,
        .v128_any_true,
        .i32x4_splat,
        .f32x4_splat,
        .i8x16_splat,
        .i16x8_splat,
        .i64x2_splat,
        .f64x2_splat,
        => |v| try w.print(" v{d}", .{v}),

        .v128_load,
        .v128_load_splat,
        .v128_load_zero,
        .v128_load_extend,
        .v128_store,
        .v128_load_lane,
        .v128_store_lane,
        .v128_bitwise,
        .v128_bitselect,
        .simd_all_true,
        .simd_bitmask,
        .i8x16_shuffle,
        .i8x16_swizzle,
        .i32x4_binop,
        .i32x4_unop,
        .i32x4_extadd_pairwise_i16x8,
        .i32x4_dot_i16x8_s,
        .i32x4_extend_i16x8,
        .i32x4_extmul_i16x8,
        .i32x4_shift,
        .i32x4_extract_lane,
        .i32x4_replace_lane,
        .i32x4_trunc_sat,
        .i16x8_binop,
        .i16x8_unop,
        .i16x8_extadd_pairwise_i8x16,
        .i16x8_extend_i8x16,
        .i16x8_extmul_i8x16,
        .i16x8_narrow_i32x4,
        .i16x8_shift,
        .i16x8_extract_lane,
        .i16x8_replace_lane,
        .i8x16_binop,
        .i8x16_unop,
        .i8x16_shift,
        .i8x16_extract_lane,
        .i8x16_replace_lane,
        .i8x16_narrow_i16x8,
        .i64x2_binop,
        .i64x2_unop,
        .i64x2_extend_i32x4,
        .i64x2_extmul_i32x4,
        .i64x2_shift,
        .i64x2_extract_lane,
        .i64x2_replace_lane,
        .f32x4_binop,
        .f32x4_unop,
        .f32x4_convert_i32x4,
        .f32x4_demote_f64x2_zero,
        .f32x4_extract_lane,
        .f32x4_replace_lane,
        .f64x2_binop,
        .f64x2_unop,
        .f64x2_convert_low_i32x4,
        .f64x2_promote_low_f32x4,
        .f64x2_extract_lane,
        .f64x2_replace_lane,
        => try formatGenericSimdPayload(op, w),

        // #672 EH ops.
        .try_table_begin => |tb| {
            try w.print(" results={d} clauses={d}", .{ tb.result_arity, tb.clauses.len });
        },
        .try_table_end => {},
        .throw => |th| {
            try w.print(" tag={d} args=[", .{th.tag_idx});
            for (th.args, 0..) |a, i| {
                if (i != 0) try w.print(",", .{});
                try w.print("v{d}", .{a});
            }
            try w.print("]", .{});
        },
        .throw_ref => |v| try w.print(" v{d}", .{v}),
    }
}

/// Generic SIMD operand renderer — emits `(vregs=[v…])` listing the VReg
/// operands of the instruction. Used for variants whose explicit
/// pretty-printer would only repeat their tag-name without adding
/// readability. Mirrors the operand walk in `passes.getUsedVRegs` so
/// the listed VRegs match what use-def analysis sees.
fn formatGenericSimdPayload(op: ir.Inst.Op, w: *std.Io.Writer) Error!void {
    var operands: [4]ir.VReg = undefined;
    var n: usize = 0;
    switch (op) {
        inline .v128_load,
        .v128_load_splat,
        .v128_load_zero,
        .v128_load_extend,
        => |ld| {
            operands[n] = ld.base;
            n += 1;
        },
        .v128_store => |st| {
            operands[n] = st.base;
            n += 1;
            operands[n] = st.val;
            n += 1;
        },
        .v128_load_lane => |ld| {
            operands[n] = ld.base;
            n += 1;
            operands[n] = ld.vector;
            n += 1;
        },
        .v128_store_lane => |st| {
            operands[n] = st.base;
            n += 1;
            operands[n] = st.vector;
            n += 1;
        },
        .v128_bitwise => |bin| {
            operands[n] = bin.lhs;
            n += 1;
            operands[n] = bin.rhs;
            n += 1;
        },
        .v128_bitselect => |sel| {
            operands[n] = sel.a;
            n += 1;
            operands[n] = sel.b;
            n += 1;
            operands[n] = sel.mask;
            n += 1;
        },
        inline .simd_all_true, .simd_bitmask => |o| {
            operands[n] = o.vector;
            n += 1;
        },
        .i8x16_shuffle => |s| {
            operands[n] = s.lhs;
            n += 1;
            operands[n] = s.rhs;
            n += 1;
        },
        .i8x16_swizzle => |s| {
            operands[n] = s.vector;
            n += 1;
            operands[n] = s.indices;
            n += 1;
        },
        inline .i32x4_binop, .i16x8_binop, .i8x16_binop, .i64x2_binop, .f32x4_binop, .f64x2_binop => |b| {
            operands[n] = b.lhs;
            n += 1;
            operands[n] = b.rhs;
            n += 1;
        },
        inline .i32x4_unop, .i16x8_unop, .i8x16_unop, .i64x2_unop, .f32x4_unop, .f64x2_unop => |u| {
            operands[n] = u.vector;
            n += 1;
        },
        inline .i32x4_extadd_pairwise_i16x8,
        .i16x8_extadd_pairwise_i8x16,
        .i32x4_extend_i16x8,
        .i16x8_extend_i8x16,
        .i64x2_extend_i32x4,
        .f32x4_convert_i32x4,
        .i32x4_trunc_sat,
        .f32x4_demote_f64x2_zero,
        .f64x2_convert_low_i32x4,
        .f64x2_promote_low_f32x4,
        => |o| {
            operands[n] = o.vector;
            n += 1;
        },
        .i32x4_dot_i16x8_s => |b| {
            operands[n] = b.lhs;
            n += 1;
            operands[n] = b.rhs;
            n += 1;
        },
        inline .i32x4_extmul_i16x8, .i16x8_extmul_i8x16, .i64x2_extmul_i32x4 => |o| {
            operands[n] = o.lhs;
            n += 1;
            operands[n] = o.rhs;
            n += 1;
        },
        inline .i16x8_narrow_i32x4, .i8x16_narrow_i16x8 => |o| {
            operands[n] = o.lhs;
            n += 1;
            operands[n] = o.rhs;
            n += 1;
        },
        inline .i32x4_shift, .i16x8_shift, .i8x16_shift, .i64x2_shift => |s| {
            operands[n] = s.vector;
            n += 1;
            operands[n] = s.count;
            n += 1;
        },
        inline .i32x4_extract_lane,
        .i16x8_extract_lane,
        .i8x16_extract_lane,
        .i64x2_extract_lane,
        .f32x4_extract_lane,
        .f64x2_extract_lane,
        => |e| {
            operands[n] = e.vector;
            n += 1;
        },
        inline .i32x4_replace_lane,
        .i16x8_replace_lane,
        .i8x16_replace_lane,
        .i64x2_replace_lane,
        .f32x4_replace_lane,
        .f64x2_replace_lane,
        => |r| {
            operands[n] = r.vector;
            n += 1;
            operands[n] = r.val;
            n += 1;
        },
        else => {},
    }

    try w.writeAll(" (vregs=[");
    for (operands[0..n], 0..) |v, i| {
        if (i > 0) try w.writeAll(", ");
        try w.print("v{d}", .{v});
    }
    try w.writeAll("])");
}
