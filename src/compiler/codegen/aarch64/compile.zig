//! AArch64 IR Compiler
//!
//! Walks IR functions and emits AArch64 machine code via CodeBuffer.
//! Uses the same pattern as the x86-64 backend with a linear-scan
//! register allocator mapping VRegs to physical registers.

const std = @import("std");
const ir = @import("../../ir/ir.zig");
const emit = @import("emit.zig");

/// Simple VReg → physical register mapping.
const RegMap = struct {
    entries: std.AutoHashMap(ir.VReg, Location),
    reg_used: [scratch_regs.len]bool = [_]bool{false} ** scratch_regs.len,
    next_stack_offset: u32 = 0,

    const Location = union(enum) {
        reg: emit.Reg,
        stack: u32, // offset from FP
    };

    // Caller-saved scratch registers (AAPCS64: X0–X15 are caller-saved).
    // We reserve X16 (IP0) and X17 (IP1) as non-allocatable scratch for
    // codegen use (shift-count negation, rem via MSUB, etc.); X18 is
    // reserved platform register. Once calls land, caller-saved across
    // call boundaries will need spill/reload handling.
    const scratch_regs = [_]emit.Reg{
        .x0, .x1, .x2, .x3, .x4, .x5, .x6, .x7,
        .x8, .x9, .x10, .x11, .x12, .x13, .x14, .x15,
    };

    /// Non-allocatable scratch registers usable by any handler.
    /// Must never appear in `scratch_regs`.
    pub const tmp0: emit.Reg = .x16;
    pub const tmp1: emit.Reg = .x17;

    fn init(allocator: std.mem.Allocator) RegMap {
        return .{
            .entries = std.AutoHashMap(ir.VReg, Location).init(allocator),
        };
    }

    fn deinit(self: *RegMap) void {
        self.entries.deinit();
    }

    fn assign(self: *RegMap, vreg: ir.VReg) !Location {
        for (scratch_regs, 0..) |r, i| {
            if (!self.reg_used[i]) {
                self.reg_used[i] = true;
                const loc = Location{ .reg = r };
                try self.entries.put(vreg, loc);
                return loc;
            }
        }
        // Spill to stack
        const offset = self.next_stack_offset;
        self.next_stack_offset += 8;
        const loc = Location{ .stack = offset };
        try self.entries.put(vreg, loc);
        return loc;
    }

    fn get(self: *const RegMap, vreg: ir.VReg) ?Location {
        return self.entries.get(vreg);
    }
};

/// A pending branch that needs its 19/26-bit PC-relative offset patched once
/// the target block's offset is known.
const BranchPatch = struct {
    patch_offset: usize,
    target_block: ir.BlockId,
    kind: Kind,

    const Kind = enum { b_uncond, b_cond };
};

/// Compile an IR function to AArch64 machine code.
///
/// **AArch64 frame layout** (positive offsets from FP, which equals new SP):
/// ```
///   [fp + 0]            = saved FP
///   [fp + 8]            = saved LR
///   [fp + 16]           = VMContext pointer (spilled from x0)
///   [fp + (i+3)*8]      = local i (0 ≤ i < local_count)
///   [fp + frame_size]   = caller's stack (args beyond x7, etc.)
/// ```
///
/// **VMContext ABI**: The VMContext pointer is passed in `x0` (AAPCS64
/// first argument) as a hidden prefix to the wasm params. It is spilled
/// to `[fp + 16]` in the prologue and reloaded into a scratch register
/// (typically `x16`/`tmp0`) whenever needed.
///
/// **Wasm parameters**: passed in `x1..x7` (up to 7 params; more not yet
/// supported), spilled to their matching local slots in the prologue.
/// Declared locals beyond `param_count` are zero-initialized.
pub fn compileFunction(func: *const ir.IrFunction, allocator: std.mem.Allocator) ![]u8 {
    // Phase 1b: stack args beyond x7 aren't supported yet.
    if (func.param_count > 7) return error.TooManyParams;

    var code = emit.CodeBuffer.init(allocator);
    errdefer code.deinit();

    var reg_map = RegMap.init(allocator);
    defer reg_map.deinit();

    // Frame size: (local_count + 3) × 8 for saved FP/LR + vmctx + locals,
    // plus 256 bytes of spill headroom, aligned to 16 (AArch64 SP alignment).
    const raw_frame = (func.local_count + 3) * 8 + 256;
    const frame_size: u32 = (raw_frame + 15) & ~@as(u32, 15);
    try code.emitPrologue(frame_size);

    // Spill VMContext (x0) and wasm params (x1..x7) to their frame slots.
    try emitEntrySpill(&code, func.*);

    var block_offsets = try allocator.alloc(usize, func.blocks.items.len);
    defer allocator.free(block_offsets);
    @memset(block_offsets, 0);

    var patches: std.ArrayListUnmanaged(BranchPatch) = .empty;
    defer patches.deinit(allocator);

    var last_was_ret = false;
    for (func.blocks.items, 0..) |block, bi| {
        block_offsets[bi] = code.len();
        for (block.instructions.items) |inst| {
            last_was_ret = isRet(inst.op);
            try compileInst(&code, inst, &reg_map, frame_size, &patches, allocator);
        }
    }

    if (!last_was_ret) {
        try code.emitEpilogue(frame_size);
    }

    // Resolve branch patches.
    for (patches.items) |p| {
        const target_off = block_offsets[p.target_block];
        const delta_bytes: i64 = @as(i64, @intCast(target_off)) - @as(i64, @intCast(p.patch_offset));
        // All AArch64 PC-relative branches use word offsets.
        if (@mod(delta_bytes, 4) != 0) return error.BranchMisaligned;
        const word_off = @divExact(delta_bytes, 4);
        const existing = std.mem.readInt(u32, code.bytes.items[p.patch_offset..][0..4], .little);
        const new_word: u32 = switch (p.kind) {
            .b_uncond => blk: {
                // B imm26: bits 25-0
                if (word_off < -(@as(i64, 1) << 25) or word_off >= (@as(i64, 1) << 25))
                    return error.BranchOutOfRange;
                const imm26: u26 = @bitCast(@as(i26, @intCast(word_off)));
                break :blk (existing & 0xFC000000) | @as(u32, imm26);
            },
            .b_cond => blk: {
                // B.cond imm19: bits 23-5
                if (word_off < -(@as(i64, 1) << 18) or word_off >= (@as(i64, 1) << 18))
                    return error.BranchOutOfRange;
                const imm19: u19 = @bitCast(@as(i19, @intCast(word_off)));
                break :blk (existing & 0xFF00001F) | (@as(u32, imm19) << 5);
            },
        };
        code.patch32(p.patch_offset, new_word);
    }

    return code.bytes.toOwnedSlice(allocator);
}

/// FP-relative slot offset (in bytes) for the VMContext pointer.
const vmctx_slot_offset: u32 = 16;

/// FP-relative slot offset (in bytes, unsigned, positive) for local `idx`.
fn localSlotOffset(idx: u32) u32 {
    return (idx + 3) * 8;
}

/// ABI register holding wasm parameter `i` (0-based). Parameter 0 is in x1
/// because x0 carries the hidden VMContext pointer.
fn paramAbiReg(i: u32) emit.Reg {
    return switch (i) {
        0 => .x1, 1 => .x2, 2 => .x3, 3 => .x4,
        4 => .x5, 5 => .x6, 6 => .x7,
        else => unreachable,
    };
}

/// Spill x0 (VMContext) and x1..x7 (wasm params) to their frame slots, and
/// zero-initialize declared locals. Must be called right after emitPrologue
/// and before any vreg allocation so we don't clobber these ABI regs.
fn emitEntrySpill(code: *emit.CodeBuffer, func: ir.IrFunction) !void {
    // VMContext → [fp + 16]
    try code.strImm(.x0, .fp, vmctx_slot_offset / 8);

    // Wasm params → local slots [fp + (i+3)*8]
    var i: u32 = 0;
    while (i < func.param_count) : (i += 1) {
        const off_scaled: u12 = @intCast(localSlotOffset(i) / 8);
        try code.strImm(paramAbiReg(i), .fp, off_scaled);
    }

    // Zero-init declared locals (wasm spec requires non-param locals to be 0).
    // Use x16 (tmp0) as the zero source. Always 64-bit store to cover any type.
    if (func.local_count > func.param_count) {
        try code.movz(RegMap.tmp0, 0, 0);
        var j: u32 = func.param_count;
        while (j < func.local_count) : (j += 1) {
            const off_scaled: u12 = @intCast(localSlotOffset(j) / 8);
            try code.strImm(RegMap.tmp0, .fp, off_scaled);
        }
    }
}

fn isRet(op: ir.Inst.Op) bool {
    return switch (op) {
        .ret => true,
        else => false,
    };
}

fn compileInst(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    reg_map: *RegMap,
    frame_size: u32,
    patches: *std.ArrayListUnmanaged(BranchPatch),
    allocator: std.mem.Allocator,
) !void {
    switch (inst.op) {
        // ── Constants ────────────────────────────────────────────────
        .iconst_32 => |val| {
            const dest = inst.dest orelse return;
            if (try destReg(code, reg_map, dest)) |r| {
                try code.movImm32(r, val);
            }
        },
        .iconst_64 => |val| {
            const dest = inst.dest orelse return;
            if (try destReg(code, reg_map, dest)) |r| {
                try code.movImm64(r, @bitCast(val));
            }
        },
        .fconst_32 => |val| {
            const dest = inst.dest orelse return;
            if (try destReg(code, reg_map, dest)) |r| {
                try code.movImm32(r, @bitCast(val));
            }
        },
        .fconst_64 => |val| {
            const dest = inst.dest orelse return;
            if (try destReg(code, reg_map, dest)) |r| {
                try code.movImm64(r, @bitCast(val));
            }
        },

        .add => |bin| try emitBinOp(code, inst, bin, reg_map, .add),
        .sub => |bin| try emitBinOp(code, inst, bin, reg_map, .sub),
        .mul => |bin| try emitBinOp(code, inst, bin, reg_map, .mul),
        .@"and" => |bin| try emitBinOp(code, inst, bin, reg_map, .@"and"),
        .@"or" => |bin| try emitBinOp(code, inst, bin, reg_map, .@"or"),
        .xor => |bin| try emitBinOp(code, inst, bin, reg_map, .xor),

        // ── Shifts / rotates ─────────────────────────────────────────
        .shl => |bin| try emitShift(code, inst, bin, reg_map, .lsl),
        .shr_u => |bin| try emitShift(code, inst, bin, reg_map, .lsr),
        .shr_s => |bin| try emitShift(code, inst, bin, reg_map, .asr),
        .rotr => |bin| try emitShift(code, inst, bin, reg_map, .ror),
        .rotl => |bin| try emitRotl(code, inst, bin, reg_map),

        // ── Comparisons ──────────────────────────────────────────────
        inline .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u => |bin, tag| {
            try emitCmp(code, inst, bin, reg_map, comptime tagToCond(tag));
        },

        .eqz => |vreg| try emitEqz(code, inst, vreg, reg_map),

        // ── Unary bit ops ────────────────────────────────────────────
        .clz => |vreg| try emitClz(code, inst, vreg, reg_map),
        .ctz => |vreg| try emitCtz(code, inst, vreg, reg_map),
        // popcnt requires NEON (CNT on V-reg). Deferred.

        // ── Sign extension / wrap ────────────────────────────────────
        .extend8_s => |vreg| try emitExtendImpl(code, inst, vreg, reg_map, .b),
        .extend16_s => |vreg| try emitExtendImpl(code, inst, vreg, reg_map, .h),
        .extend32_s => |vreg| try emitExtendImpl(code, inst, vreg, reg_map, .w),
        .wrap_i64 => |vreg| try emitWrap(code, inst, vreg, reg_map),

        // ── Select (CSEL) ────────────────────────────────────────────
        .select => |sel| try emitSelect(code, inst, sel, reg_map),

        // ── Branches ─────────────────────────────────────────────────
        .br => |target| try emitBr(code, patches, target, allocator),
        .br_if => |br| try emitBrIf(code, reg_map, patches, br, allocator),

        // ── Locals (FP-relative frame slots) ─────────────────────────
        .local_get => |idx| {
            const dest = inst.dest orelse return;
            const dr = (try destReg(code, reg_map, dest)) orelse return;
            const off_scaled: u12 = @intCast(localSlotOffset(idx) / 8);
            // Always 64-bit load: the slot stores the full 8 bytes, and a
            // subsequent W-form op (or wrap_i64) will ignore the upper bits.
            try code.ldrImm(dr, .fp, off_scaled);
        },
        .local_set => |ls| {
            const src = useReg(reg_map, ls.val) orelse return;
            const off_scaled: u12 = @intCast(localSlotOffset(ls.idx) / 8);
            try code.strImm(src, .fp, off_scaled);
        },

        .ret => |maybe_val| {
            if (maybe_val) |val| {
                if (reg_map.get(val)) |loc| {
                    switch (loc) {
                        .reg => |r| {
                            if (r != .x0) try code.movRegReg(.x0, r);
                        },
                        .stack => {},
                    }
                }
            }
            try code.emitEpilogue(frame_size);
        },
        .@"unreachable" => try code.brk(0),
        else => {
            // Explicit failure for unimplemented ops. Previously this was a
            // silent no-op which produced incorrect code. Anything that lands
            // here needs a handler — track remaining work in issue #111.
            return error.UnimplementedOp;
        },
    }
}

fn tagToCond(comptime tag: std.meta.Tag(ir.Inst.Op)) emit.Cond {
    return switch (tag) {
        .eq => .eq,
        .ne => .ne,
        .lt_s => .lt,
        .lt_u => .lo,
        .gt_s => .gt,
        .gt_u => .hi,
        .le_s => .le,
        .le_u => .ls,
        .ge_s => .ge,
        .ge_u => .hs,
        else => @compileError("tagToCond: not a comparison tag"),
    };
}

/// Resolve the destination register, returning null if it spills to stack
/// (in which case the caller silently drops — stack-spill isn't wired up
/// yet for the new handlers).
fn destReg(code: *emit.CodeBuffer, reg_map: *RegMap, dest: ir.VReg) !?emit.Reg {
    _ = code;
    const loc = try reg_map.assign(dest);
    return switch (loc) {
        .reg => |r| r,
        .stack => null,
    };
}

fn useReg(reg_map: *const RegMap, vreg: ir.VReg) ?emit.Reg {
    const loc = reg_map.get(vreg) orelse return null;
    return switch (loc) {
        .reg => |r| r,
        .stack => null,
    };
}

const ExtendWidth = enum { b, h, w };

fn emitEqz(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = useReg(reg_map, vreg) orelse return;
    const dr = (try destReg(code, reg_map, dest)) orelse return;
    if (inst.type == .i32) {
        try code.cmpImm32(src, 0);
        try code.cset32(dr, .eq);
    } else {
        try code.cmpImm(src, 0);
        try code.cset(dr, .eq);
    }
}

fn emitCmp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    cond: emit.Cond,
) !void {
    const dest = inst.dest orelse return;
    const lhs = useReg(reg_map, bin.lhs) orelse return;
    const rhs = useReg(reg_map, bin.rhs) orelse return;
    const dr = (try destReg(code, reg_map, dest)) orelse return;
    if (inst.type == .i32) {
        try code.cmpRegReg32(lhs, rhs);
        try code.cset32(dr, cond);
    } else {
        try code.cmpRegReg(lhs, rhs);
        try code.cset(dr, cond);
    }
}

fn emitClz(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = useReg(reg_map, vreg) orelse return;
    const dr = (try destReg(code, reg_map, dest)) orelse return;
    if (inst.type == .i32) {
        try code.clzReg32(dr, src);
    } else {
        try code.clzReg(dr, src);
    }
}

fn emitCtz(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = useReg(reg_map, vreg) orelse return;
    const dr = (try destReg(code, reg_map, dest)) orelse return;
    // CTZ(x) = CLZ(RBIT(x)). Width matters: RBIT W/CLZ W gives 32 for zero;
    // RBIT X/CLZ X gives 64 — matching wasm semantics.
    if (inst.type == .i32) {
        try code.rbitReg32(dr, src);
        try code.clzReg32(dr, dr);
    } else {
        try code.rbitReg(dr, src);
        try code.clzReg(dr, dr);
    }
}

fn emitExtendImpl(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
    width: ExtendWidth,
) !void {
    const dest = inst.dest orelse return;
    const src = useReg(reg_map, vreg) orelse return;
    const dr = (try destReg(code, reg_map, dest)) orelse return;
    switch (width) {
        .b => try code.sxtb(dr, src),
        .h => try code.sxth(dr, src),
        .w => try code.sxtw(dr, src),
    }
}

fn emitWrap(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = useReg(reg_map, vreg) orelse return;
    const dr = (try destReg(code, reg_map, dest)) orelse return;
    // wasm i32.wrap_i64: take low 32 bits of i64. MOV Wd, Wn zero-extends.
    try code.uxtw(dr, src);
}

fn emitSelect(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    sel: @TypeOf(@as(ir.Inst.Op, undefined).select),
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const cond_r = useReg(reg_map, sel.cond) orelse return;
    const t_r = useReg(reg_map, sel.if_true) orelse return;
    const f_r = useReg(reg_map, sel.if_false) orelse return;
    const dr = (try destReg(code, reg_map, dest)) orelse return;
    // wasm: select picks if_true when cond != 0. Use W-form test since
    // cond is always i32.
    try code.cmpImm32(cond_r, 0);
    if (inst.type == .i32) {
        try code.csel32(dr, t_r, f_r, .ne);
    } else {
        try code.csel(dr, t_r, f_r, .ne);
    }
}

fn emitShift(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    op: emit.CodeBuffer.ShiftOp,
) !void {
    const dest = inst.dest orelse return;
    const lhs = useReg(reg_map, bin.lhs) orelse return;
    const rhs = useReg(reg_map, bin.rhs) orelse return;
    const dr = (try destReg(code, reg_map, dest)) orelse return;
    if (inst.type == .i32) {
        try code.shiftRegReg32(dr, lhs, rhs, op);
    } else {
        try code.shiftRegReg(dr, lhs, rhs, op);
    }
}

fn emitRotl(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const lhs = useReg(reg_map, bin.lhs) orelse return;
    const rhs = useReg(reg_map, bin.rhs) orelse return;
    const dr = (try destReg(code, reg_map, dest)) orelse return;
    // rotl(x, n) = ror(x, -n). AArch64 has no ROL, so negate count
    // into scratch, then RORV. Width-correct: W-form masks by 5, X by 6.
    if (inst.type == .i32) {
        try code.negReg32(RegMap.tmp0, rhs);
        try code.shiftRegReg32(dr, lhs, RegMap.tmp0, .ror);
    } else {
        try code.negReg(RegMap.tmp0, rhs);
        try code.shiftRegReg(dr, lhs, RegMap.tmp0, .ror);
    }
}

fn emitBr(
    code: *emit.CodeBuffer,
    patches: *std.ArrayListUnmanaged(BranchPatch),
    target: ir.BlockId,
    allocator: std.mem.Allocator,
) !void {
    const patch_off = code.len();
    try code.b(0); // placeholder
    try patches.append(allocator, .{
        .patch_offset = patch_off,
        .target_block = target,
        .kind = .b_uncond,
    });
}

fn emitBrIf(
    code: *emit.CodeBuffer,
    reg_map: *RegMap,
    patches: *std.ArrayListUnmanaged(BranchPatch),
    br: @TypeOf(@as(ir.Inst.Op, undefined).br_if),
    allocator: std.mem.Allocator,
) !void {
    const cond_r = useReg(reg_map, br.cond) orelse return;
    // wasm br_if: branch to then_block if cond != 0 (i32).
    try code.cmpImm32(cond_r, 0);
    const cond_patch = code.len();
    try code.bCond(.ne, 0); // placeholder
    try patches.append(allocator, .{
        .patch_offset = cond_patch,
        .target_block = br.then_block,
        .kind = .b_cond,
    });
    // Unconditional branch to else_block
    const else_patch = code.len();
    try code.b(0);
    try patches.append(allocator, .{
        .patch_offset = else_patch,
        .target_block = br.else_block,
        .kind = .b_uncond,
    });
}

const BinOpKind = enum { add, sub, mul, @"and", @"or", xor };

fn emitBinOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    kind: BinOpKind,
) !void {
    const dest = inst.dest orelse return;
    const lhs_loc = reg_map.get(bin.lhs) orelse return;
    const rhs_loc = reg_map.get(bin.rhs) orelse return;
    const dest_loc = try reg_map.assign(dest);

    const dest_reg = switch (dest_loc) { .reg => |r| r, .stack => return };
    const lhs_reg = switch (lhs_loc) { .reg => |r| r, .stack => return };
    const rhs_reg = switch (rhs_loc) { .reg => |r| r, .stack => return };

    switch (kind) {
        .add => try code.addRegReg(dest_reg, lhs_reg, rhs_reg),
        .sub => try code.subRegReg(dest_reg, lhs_reg, rhs_reg),
        .mul => try code.mulRegReg(dest_reg, lhs_reg, rhs_reg),
        .@"and" => try code.andRegReg(dest_reg, lhs_reg, rhs_reg),
        .@"or" => try code.orrRegReg(dest_reg, lhs_reg, rhs_reg),
        .xor => try code.eorRegReg(dest_reg, lhs_reg, rhs_reg),
    }
}

/// Result of compiling an IR module.
pub const CompileResult = struct {
    code: []u8,
    offsets: []u32,
};

/// Compile all functions in an IR module to AArch64 machine code.
pub fn compileModule(ir_module: *const ir.IrModule, allocator: std.mem.Allocator) !CompileResult {
    var all_code: std.ArrayListUnmanaged(u8) = .empty;
    errdefer all_code.deinit(allocator);
    var offsets: std.ArrayListUnmanaged(u32) = .empty;
    errdefer offsets.deinit(allocator);

    for (ir_module.functions.items) |func| {
        try offsets.append(allocator, @intCast(all_code.items.len));
        const func_code = try compileFunction(&func, allocator);
        defer allocator.free(func_code);
        try all_code.appendSlice(allocator, func_code);
    }

    return .{
        .code = try all_code.toOwnedSlice(allocator),
        .offsets = try offsets.toOwnedSlice(allocator),
    };
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "compileFunction: iconst_32 + ret" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Should produce: prologue + MOVZ + epilogue
    // All instructions are 4 bytes each
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0); // all 4-byte aligned
}

test "compileFunction: add two constants" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compileModule: records offsets" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    // Add two functions
    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    const b1 = try f1.newBlock();
    const v0 = f1.newVReg();
    try f1.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try f1.getBlock(b1).append(.{ .op = .{ .ret = v0 } });
    _ = try module.addFunction(f1);

    var f2 = ir.IrFunction.init(allocator, 0, 1, 0);
    const b2 = try f2.newBlock();
    const v1 = f2.newVReg();
    try f2.getBlock(b2).append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1 });
    try f2.getBlock(b2).append(.{ .op = .{ .ret = v1 } });
    _ = try module.addFunction(f2);

    const result = try compileModule(&module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    try std.testing.expectEqual(@as(usize, 2), result.offsets.len);
    try std.testing.expectEqual(@as(u32, 0), result.offsets[0]);
    try std.testing.expect(result.offsets[1] > 0); // second function starts after first
}

// ── New handler tests (Phase 1a) ─────────────────────────────────────────────

/// Build a single-block function with the given ops and verify codegen
/// succeeds and produces 4-byte-aligned output.
fn expectCompiles(insts: []const ir.Inst, param_count: u32, local_count: u32) !void {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, param_count, 1, local_count);
    defer func.deinit();
    const bid = try func.newBlock();
    const block = func.getBlock(bid);
    for (insts) |i| try block.append(i);
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compile: iconst_64 + ret" {
    var func = ir.IrFunction.init(std.testing.allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_64 = 0x1234_5678_9ABC_DEF0 }, .dest = v0, .type = .i64 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    const code = try compileFunction(&func, std.testing.allocator);
    defer std.testing.allocator.free(code);
    try std.testing.expect(code.len > 0);
}

test "compile: comparison (eq) produces CMP+CSET" {
    const v0: ir.VReg = 0;
    const v1: ir.VReg = 1;
    const v2: ir.VReg = 2;
    try expectCompiles(&.{
        .{ .op = .{ .iconst_32 = 5 }, .dest = v0, .type = .i32 },
        .{ .op = .{ .iconst_32 = 10 }, .dest = v1, .type = .i32 },
        .{ .op = .{ .eq = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 },
        .{ .op = .{ .ret = v2 } },
    }, 0, 0);
}

test "compile: eqz" {
    const v0: ir.VReg = 0;
    const v1: ir.VReg = 1;
    try expectCompiles(&.{
        .{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 },
        .{ .op = .{ .eqz = v0 }, .dest = v1, .type = .i32 },
        .{ .op = .{ .ret = v1 } },
    }, 0, 0);
}

test "compile: all comparisons" {
    // Each comparison returns i32 0/1.
    inline for (.{ "eq", "ne", "lt_s", "lt_u", "gt_s", "gt_u", "le_s", "le_u", "ge_s", "ge_u" }) |name| {
        const allocator = std.testing.allocator;
        var func = ir.IrFunction.init(allocator, 0, 1, 0);
        defer func.deinit();
        const bid = try func.newBlock();
        const v0 = func.newVReg();
        const v1 = func.newVReg();
        const v2 = func.newVReg();
        try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 3 }, .dest = v0, .type = .i32 });
        try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v1, .type = .i32 });
        const op = @unionInit(ir.Inst.Op, name, .{ .lhs = v0, .rhs = v1 });
        try func.getBlock(bid).append(.{ .op = op, .dest = v2, .type = .i32 });
        try func.getBlock(bid).append(.{ .op = .{ .ret = v2 } });
        const code = try compileFunction(&func, allocator);
        defer allocator.free(code);
        try std.testing.expect(code.len > 0);
    }
}

test "compile: shifts and rotates" {
    inline for (.{ "shl", "shr_u", "shr_s", "rotl", "rotr" }) |name| {
        const allocator = std.testing.allocator;
        var func = ir.IrFunction.init(allocator, 0, 1, 0);
        defer func.deinit();
        const bid = try func.newBlock();
        const v0 = func.newVReg();
        const v1 = func.newVReg();
        const v2 = func.newVReg();
        try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0, .type = .i32 });
        try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 3 }, .dest = v1, .type = .i32 });
        const op = @unionInit(ir.Inst.Op, name, .{ .lhs = v0, .rhs = v1 });
        try func.getBlock(bid).append(.{ .op = op, .dest = v2, .type = .i32 });
        try func.getBlock(bid).append(.{ .op = .{ .ret = v2 } });
        const code = try compileFunction(&func, allocator);
        defer allocator.free(code);
        try std.testing.expect(code.len > 0);
    }
}

test "compile: clz / ctz / extend / wrap" {
    inline for (.{ "clz", "ctz", "extend8_s", "extend16_s", "extend32_s", "wrap_i64" }) |name| {
        const allocator = std.testing.allocator;
        var func = ir.IrFunction.init(allocator, 0, 1, 0);
        defer func.deinit();
        const bid = try func.newBlock();
        const v0 = func.newVReg();
        const v1 = func.newVReg();
        try func.getBlock(bid).append(.{ .op = .{ .iconst_64 = 0x1234 }, .dest = v0, .type = .i64 });
        const op = @unionInit(ir.Inst.Op, name, v0);
        try func.getBlock(bid).append(.{ .op = op, .dest = v1, .type = if (std.mem.eql(u8, name, "wrap_i64")) .i32 else .i64 });
        try func.getBlock(bid).append(.{ .op = .{ .ret = v1 } });
        const code = try compileFunction(&func, allocator);
        defer allocator.free(code);
        try std.testing.expect(code.len > 0);
    }
}

test "compile: select via CSEL" {
    const v0: ir.VReg = 0;
    const v1: ir.VReg = 1;
    const vc: ir.VReg = 2;
    const vr: ir.VReg = 3;
    try expectCompiles(&.{
        .{ .op = .{ .iconst_32 = 10 }, .dest = v0, .type = .i32 },
        .{ .op = .{ .iconst_32 = 20 }, .dest = v1, .type = .i32 },
        .{ .op = .{ .iconst_32 = 1 }, .dest = vc, .type = .i32 },
        .{ .op = .{ .select = .{ .cond = vc, .if_true = v0, .if_false = v1 } }, .dest = vr, .type = .i32 },
        .{ .op = .{ .ret = vr } },
    }, 0, 0);
}

test "compile: br forward jump patches correctly" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = v0 } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compile: br_if with two blocks" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b_then = try func.newBlock();
    const b_else = try func.newBlock();
    const v_cond = func.newVReg();
    const v_t = func.newVReg();
    const v_f = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_cond, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b_then, .else_block = b_else } } });
    try func.getBlock(b_then).append(.{ .op = .{ .iconst_32 = 11 }, .dest = v_t, .type = .i32 });
    try func.getBlock(b_then).append(.{ .op = .{ .ret = v_t } });
    try func.getBlock(b_else).append(.{ .op = .{ .iconst_32 = 22 }, .dest = v_f, .type = .i32 });
    try func.getBlock(b_else).append(.{ .op = .{ .ret = v_f } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
}

// ── Phase 1b: VMContext ABI + locals tests ───────────────────────────────────

test "compile: entry spills vmctx to [fp+16]" {
    // Function with no params, no locals: prologue must still spill x0 → [fp+16].
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // First 2 words: prologue (STP + ADD fp, sp, #0)
    // Third word: STR x0, [fp, #16]
    //   opcode 0xF9000000 | (imm12=2 << 10) | (rn=29 << 5) | rt=0
    //   = 0xF9000000 | 0x800 | 0x3A0 | 0 = 0xF9000BA0
    const str_word = std.mem.readInt(u32, code[8..][0..4], .little);
    try std.testing.expectEqual(@as(u32, 0xF9000BA0), str_word);
}

test "compile: param spill — STR x1 into first local slot" {
    // 1 param (x1), 1 local (just the param), no body besides ret of local 0.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 1);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .local_get = 0 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // words: [0]=STP, [1]=ADD fp,sp,0, [2]=STR x0 vmctx, [3]=STR x1 local0
    //   STR x1, [fp, #24]: rt=1, rn=29, imm12=3
    //   = 0xF9000000 | (3<<10) | (29<<5) | 1 = 0xF9000FA1
    const str_param = std.mem.readInt(u32, code[12..][0..4], .little);
    try std.testing.expectEqual(@as(u32, 0xF9000FA1), str_param);
}

test "compile: zero-init of declared local (beyond params)" {
    // 0 params, 2 locals. After spilling x0 (vmctx), we MOVZ x16,#0 then
    // STR x16 to both local slots.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 2);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // words: STP, ADD fp, STR x0 vmctx, MOVZ x16 #0, STR x16→local0, STR x16→local1
    const movz_word = std.mem.readInt(u32, code[12..][0..4], .little);
    // MOVZ X16, #0, LSL #0 = 0xD2800000 | (0<<5) | 16 = 0xD2800010
    try std.testing.expectEqual(@as(u32, 0xD2800010), movz_word);
    const str0 = std.mem.readInt(u32, code[16..][0..4], .little);
    // STR x16, [fp, #24]: rt=16, rn=29, imm12=3 → 0xF9000000|(3<<10)|(29<<5)|16
    try std.testing.expectEqual(@as(u32, 0xF9000FB0), str0);
    const str1 = std.mem.readInt(u32, code[20..][0..4], .little);
    // STR x16, [fp, #32]: imm12=4 → 0xF9000000|(4<<10)|(29<<5)|16
    try std.testing.expectEqual(@as(u32, 0xF90013B0), str1);
}

test "compile: local_get + local_set round trip" {
    // local_set 0, const 42; local_get 0; ret.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 1);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v0 } } });
    try func.getBlock(bid).append(.{ .op = .{ .local_get = 0 }, .dest = v1, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v1 } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compile: rejects more than 7 params (Phase 1b limit)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 8, 1, 8);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    try std.testing.expectError(error.TooManyParams, compileFunction(&func, allocator));
}

test "compile: unimplemented op returns error.UnimplementedOp" {
    // .call is not yet implemented — must fail loudly, not silently drop.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .call = .{ .func_idx = 0, .args = &.{} } },
        .dest = v0,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    try std.testing.expectError(error.UnimplementedOp, compileFunction(&func, allocator));
}
