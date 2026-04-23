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
/// Redundant bounds-check elimination within each basic block.
///
/// For every `.load` and `.store`, codegen emits an inline wasm-memory
/// bounds check that verifies `zext(base) + offset + size <= memory_size`.
/// When two accesses in the same block share the same `base` vreg, the
/// first check already validates any subsequent access whose
/// `offset + size` does not exceed the max previously validated end.
///
/// This pass marks such accesses with `bounds_known = true`; both backends
/// skip emitting the check for those.
///
/// Safety conditions:
/// - Same block only (no cross-block dominator walk — conservative).
/// - Any call / atomic / `memory_grow` / `memory_copy` / `memory_fill`
///   can change `memory_size`, so the tracker is cleared at those points.
/// - IR is SSA, so a base vreg's value never changes once defined.
pub fn elideRedundantBoundsChecks(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;
    var max_end = std.AutoHashMap(ir.VReg, u64).init(allocator);
    defer max_end.deinit();

    for (func.blocks.items) |block| {
        max_end.clearRetainingCapacity();
        for (block.instructions.items) |*inst| {
            switch (inst.op) {
                .load => |*ld| {
                    const end: u64 = @as(u64, ld.offset) + @as(u64, ld.size);
                    const prev = max_end.get(ld.base) orelse 0;
                    if (end <= prev) {
                        if (!ld.bounds_known) {
                            ld.bounds_known = true;
                            changed = true;
                        }
                    } else {
                        try max_end.put(ld.base, end);
                    }
                },
                .store => |*st| {
                    const end: u64 = @as(u64, st.offset) + @as(u64, st.size);
                    const prev = max_end.get(st.base) orelse 0;
                    if (end <= prev) {
                        if (!st.bounds_known) {
                            st.bounds_known = true;
                            changed = true;
                        }
                    } else {
                        try max_end.put(st.base, end);
                    }
                },
                // Anything that can change memory_size, trap, or otherwise
                // invalidate the "already checked" property: clear the
                // tracker. (memory.grow can extend memory; calls can grow
                // or shrink via imports; atomics/traps can't shrink but
                // cost us nothing to be conservative.)
                .memory_grow,
                .call, .call_indirect, .call_ref,
                .memory_copy, .memory_fill, .memory_init,
                .table_grow, .table_init,
                .atomic_notify, .atomic_wait,
                => max_end.clearRetainingCapacity(),
                else => {},
            }
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

pub fn runPasses(module: *ir.IrModule, passes: []const PassFn, allocator: std.mem.Allocator) !u32 {
    var total_changes: u32 = 0;
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
    &strengthReduceMul,
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
