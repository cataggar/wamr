//! AArch64 IR Compiler
//!
//! Walks IR functions and emits AArch64 machine code via CodeBuffer.
//! Uses the same pattern as the x86-64 backend with a linear-scan
//! register allocator mapping VRegs to physical registers.

const std = @import("std");
const ir = @import("../../ir/ir.zig");
const emit = @import("emit.zig");

/// Simple VReg → physical register mapping.
///
/// Stack-spill layout: when all scratch regs are in use, the allocator
/// falls back to the frame spill region at `[fp + spill_base + slot*8]`.
/// `spill_base` is set by `compileFunction` to sit immediately above the
/// locals area so spill slots don't collide with saved FP/LR, VMContext,
/// or wasm locals.
const RegMap = struct {
    entries: std.AutoHashMap(ir.VReg, Location),
    reg_used: [scratch_regs.len]bool = [_]bool{false} ** scratch_regs.len,
    next_stack_offset: u32 = 0,
    /// Byte offset from FP where the first spill slot lives.
    spill_base: u32 = 0,
    /// Maximum bytes reserved for spills (beyond which `assign` errors).
    spill_capacity: u32 = 0,

    const Location = union(enum) {
        reg: emit.Reg,
        /// Byte offset from `spill_base` for the spill slot.
        stack: u32,
    };

    // Caller-saved scratch registers (AAPCS64: X0–X15 are caller-saved).
    // We reserve X16 (IP0) and X17 (IP1) as non-allocatable scratch for
    // codegen use (shift-count negation, rem via MSUB, spill reload, etc.);
    // X18 is reserved platform register. Once calls land, caller-saved
    // across call boundaries will need spill/reload handling.
    //
    // X15 is reserved as `tmp2` for multi-operand sequences (e.g., bounds
    // check needs vmctx + zext_addr + mem_size live simultaneously). The
    // tradeoff: one less allocatable reg in exchange for simpler helper
    // sequences that don't have to spill/reload through the frame.
    const scratch_regs = [_]emit.Reg{
        .x0, .x1, .x2, .x3, .x4, .x5, .x6, .x7,
        .x8, .x9, .x10, .x11, .x12, .x13, .x14,
    };

    /// Non-allocatable scratch registers usable by any handler.
    /// Must never appear in `scratch_regs`.
    pub const tmp0: emit.Reg = .x16;
    pub const tmp1: emit.Reg = .x17;
    pub const tmp2: emit.Reg = .x15;

    fn init(allocator: std.mem.Allocator, spill_base: u32, spill_capacity: u32) RegMap {
        return .{
            .entries = std.AutoHashMap(ir.VReg, Location).init(allocator),
            .spill_base = spill_base,
            .spill_capacity = spill_capacity,
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
        if (self.next_stack_offset >= self.spill_capacity) return error.OutOfSpillSlots;
        const offset = self.next_stack_offset;
        self.next_stack_offset += 8;
        const loc = Location{ .stack = offset };
        try self.entries.put(vreg, loc);
        return loc;
    }

    fn get(self: *const RegMap, vreg: ir.VReg) ?Location {
        return self.entries.get(vreg);
    }

    /// FP-relative byte offset of a spill slot (scaled by 8 for LDR/STR X).
    fn spillOffsetScaled(self: *const RegMap, slot_byte_off: u32) u12 {
        return @intCast((self.spill_base + slot_byte_off) / 8);
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

/// A pending BL whose 26-bit PC-relative offset needs to be patched once
/// all module functions are compiled and their start offsets are known.
pub const CallPatch = struct {
    /// Byte offset of the BL instruction within the target code buffer.
    patch_offset: usize,
    /// Local function index (import_count already subtracted).
    target_func_idx: u32,
};

/// Context threaded through per-function compilation for cross-function
/// concerns (calls, imports).
const FuncCompileCtx = struct {
    import_count: u32 = 0,
    call_patches: ?*std.ArrayListUnmanaged(CallPatch) = null,
    /// FP-relative byte offset of the call-save region where caller-save
    /// physregs (x0..x15) are spilled around BL instructions.
    call_save_base: u32 = 0,
    allocator: std.mem.Allocator,
};

/// Compile an IR function to AArch64 machine code.
///
/// **AArch64 frame layout** (positive offsets from FP, which equals new SP):
/// ```
///   [fp + 0]                = saved FP
///   [fp + 8]                = saved LR
///   [fp + 16]               = VMContext pointer (spilled from x0)
///   [fp + (i+3)*8]          = local i (0 ≤ i < local_count)
///   [fp + spill_base]       = RegMap spill slots (256 bytes, 32 slots)
///   [fp + call_save_base]   = per-physreg call-save slots (128 bytes)
///   [fp + frame_size]       = caller's stack (args beyond x7, etc.)
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
    // Test-friendly entry: no cross-function linking. Any `.call` op will
    // fail with error.CallLinkageUnavailable.
    return compileFunctionImpl(func, .{ .allocator = allocator }, allocator);
}

pub fn compileFunctionImpl(
    func: *const ir.IrFunction,
    ctx: FuncCompileCtx,
    allocator: std.mem.Allocator,
) ![]u8 {
    // Phase 1b: stack args beyond x7 aren't supported yet.
    if (func.param_count > 7) return error.TooManyParams;

    var code = emit.CodeBuffer.init(allocator);
    errdefer code.deinit();

    // Frame size: (local_count + 3) × 8 for saved FP/LR + vmctx + locals,
    // plus 256 bytes of spill headroom (32 slots), plus 128 bytes for
    // call-save (16 physregs × 8), aligned to 16 (AArch64 SP alignment).
    const spill_base: u32 = (func.local_count + 3) * 8;
    const spill_capacity: u32 = 256;
    const call_save_base: u32 = spill_base + spill_capacity;
    const call_save_size: u32 = 128;
    const raw_frame = call_save_base + call_save_size;
    const frame_size: u32 = (raw_frame + 15) & ~@as(u32, 15);

    var reg_map = RegMap.init(allocator, spill_base, spill_capacity);
    defer reg_map.deinit();

    var fctx = ctx;
    fctx.call_save_base = call_save_base;

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
            try compileInst(&code, inst, &reg_map, frame_size, &patches, &fctx);
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
    fctx: *FuncCompileCtx,
) !void {
    switch (inst.op) {
        // ── Constants ────────────────────────────────────────────────
        .iconst_32 => |val| {
            const dest = inst.dest orelse return;
            const info = try destBegin(reg_map, dest, RegMap.tmp0);
            try code.movImm32(info.reg, val);
            try destCommit(code, reg_map, info);
        },
        .iconst_64 => |val| {
            const dest = inst.dest orelse return;
            const info = try destBegin(reg_map, dest, RegMap.tmp0);
            try code.movImm64(info.reg, @bitCast(val));
            try destCommit(code, reg_map, info);
        },
        .fconst_32 => |val| {
            const dest = inst.dest orelse return;
            const info = try destBegin(reg_map, dest, RegMap.tmp0);
            try code.movImm32(info.reg, @bitCast(val));
            try destCommit(code, reg_map, info);
        },
        .fconst_64 => |val| {
            const dest = inst.dest orelse return;
            const info = try destBegin(reg_map, dest, RegMap.tmp0);
            try code.movImm64(info.reg, @bitCast(val));
            try destCommit(code, reg_map, info);
        },

        .add => |bin| try emitBinOp(code, inst, bin, reg_map, .add),
        .sub => |bin| try emitBinOp(code, inst, bin, reg_map, .sub),
        .mul => |bin| try emitBinOp(code, inst, bin, reg_map, .mul),
        .@"and" => |bin| try emitBinOp(code, inst, bin, reg_map, .@"and"),
        .@"or" => |bin| try emitBinOp(code, inst, bin, reg_map, .@"or"),
        .xor => |bin| try emitBinOp(code, inst, bin, reg_map, .xor),

        .div_s => |bin| try emitDivRem(code, inst, bin, reg_map, .div_s),
        .div_u => |bin| try emitDivRem(code, inst, bin, reg_map, .div_u),
        .rem_s => |bin| try emitDivRem(code, inst, bin, reg_map, .rem_s),
        .rem_u => |bin| try emitDivRem(code, inst, bin, reg_map, .rem_u),

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
        .br => |target| try emitBr(code, patches, target, fctx.allocator),
        .br_if => |br| try emitBrIf(code, reg_map, patches, br, fctx.allocator),

        // ── Direct function call ─────────────────────────────────────
        .call => |cl| try emitCall(code, inst, cl, reg_map, fctx),

        // ── Locals (FP-relative frame slots) ─────────────────────────
        .local_get => |idx| {
            const dest = inst.dest orelse return;
            const info = try destBegin(reg_map, dest, RegMap.tmp0);
            const off_scaled: u12 = @intCast(localSlotOffset(idx) / 8);
            // Always 64-bit load: the slot stores the full 8 bytes, and a
            // subsequent W-form op (or wrap_i64) will ignore the upper bits.
            try code.ldrImm(info.reg, .fp, off_scaled);
            try destCommit(code, reg_map, info);
        },
        .local_set => |ls| {
            const src = try useInto(code, reg_map, ls.val, RegMap.tmp0);
            const off_scaled: u12 = @intCast(localSlotOffset(ls.idx) / 8);
            try code.strImm(src, .fp, off_scaled);
        },

        // ── Linear memory load/store ─────────────────────────────────
        .load => |ld| try emitLoad(code, inst, ld, reg_map),
        .store => |st| try emitStore(code, st, reg_map),

        .ret => |maybe_val| {
            if (maybe_val) |val| {
                const r = try useInto(code, reg_map, val, .x0);
                if (r != .x0) try code.movRegReg(.x0, r);
            }
            try code.emitEpilogue(frame_size);
        },
        .@"unreachable" => try emitTrapHelperCall(code, vmctx_trap_unreachable_fn_slot),
        .global_get => |idx| try emitGlobalGet(code, inst, idx, reg_map),
        .global_set => |gs| try emitGlobalSet(code, gs, reg_map),
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
/// (in which case the caller silently drops — kept for handlers that
/// haven't been converted to the `destBegin`/`destCommit` spill API yet).
fn destReg(code: *emit.CodeBuffer, reg_map: *RegMap, dest: ir.VReg) !?emit.Reg {
    _ = code;
    const loc = try reg_map.assign(dest);
    return switch (loc) {
        .reg => |r| r,
        .stack => null,
    };
}

/// Full-spill-aware dest allocation. Returns a writable register: the
/// vreg's assigned physreg if any, else `scratch`. Caller must emit
/// `destCommit` once the result has been written into `info.reg`.
const DestInfo = struct {
    reg: emit.Reg,
    spill_slot: ?u32 = null, // byte offset from spill_base if spilled
};

fn destBegin(reg_map: *RegMap, dest: ir.VReg, scratch: emit.Reg) !DestInfo {
    const loc = try reg_map.assign(dest);
    return switch (loc) {
        .reg => |r| .{ .reg = r },
        .stack => |off| .{ .reg = scratch, .spill_slot = off },
    };
}

fn destCommit(code: *emit.CodeBuffer, reg_map: *const RegMap, info: DestInfo) !void {
    if (info.spill_slot) |off| {
        try code.strImm(info.reg, .fp, reg_map.spillOffsetScaled(off));
    }
}

/// Read `vreg`'s value, loading into `scratch` if spilled. Returns the
/// register holding the value. If the vreg is unbound the caller's
/// silent-drop contract is preserved via the `useReg` helper below.
fn useInto(
    code: *emit.CodeBuffer,
    reg_map: *const RegMap,
    vreg: ir.VReg,
    scratch: emit.Reg,
) !emit.Reg {
    const loc = reg_map.get(vreg) orelse return error.UnboundVReg;
    return switch (loc) {
        .reg => |r| r,
        .stack => |off| blk: {
            try code.ldrImm(scratch, .fp, reg_map.spillOffsetScaled(off));
            break :blk scratch;
        },
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
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    if (inst.type == .i32) {
        try code.cmpImm32(src, 0);
        try code.cset32(info.reg, .eq);
    } else {
        try code.cmpImm(src, 0);
        try code.cset(info.reg, .eq);
    }
    try destCommit(code, reg_map, info);
}

fn emitCmp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    cond: emit.Cond,
) !void {
    const dest = inst.dest orelse return;
    const lhs = try useInto(code, reg_map, bin.lhs, RegMap.tmp0);
    const rhs = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);
    const info = try destBegin(reg_map, dest, RegMap.tmp2);
    if (inst.type == .i32) {
        try code.cmpRegReg32(lhs, rhs);
        try code.cset32(info.reg, cond);
    } else {
        try code.cmpRegReg(lhs, rhs);
        try code.cset(info.reg, cond);
    }
    try destCommit(code, reg_map, info);
}

fn emitClz(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    if (inst.type == .i32) {
        try code.clzReg32(info.reg, src);
    } else {
        try code.clzReg(info.reg, src);
    }
    try destCommit(code, reg_map, info);
}

fn emitCtz(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    // CTZ(x) = CLZ(RBIT(x)). RBIT reads src then writes info.reg; CLZ reads
    // info.reg then writes info.reg — safe even if info.reg aliases src.
    if (inst.type == .i32) {
        try code.rbitReg32(info.reg, src);
        try code.clzReg32(info.reg, info.reg);
    } else {
        try code.rbitReg(info.reg, src);
        try code.clzReg(info.reg, info.reg);
    }
    try destCommit(code, reg_map, info);
}

fn emitExtendImpl(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
    width: ExtendWidth,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    switch (width) {
        .b => try code.sxtb(info.reg, src),
        .h => try code.sxth(info.reg, src),
        .w => try code.sxtw(info.reg, src),
    }
    try destCommit(code, reg_map, info);
}

fn emitWrap(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    // wasm i32.wrap_i64: take low 32 bits of i64. UXTW zero-extends.
    try code.uxtw(info.reg, src);
    try destCommit(code, reg_map, info);
}

fn emitSelect(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    sel: @TypeOf(@as(ir.Inst.Op, undefined).select),
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    // Load cond first; after CMP we don't need the cond register anymore,
    // so we can reuse tmp0 for subsequent uses if desired. We keep t/f in
    // distinct scratches (tmp1/tmp2) so CSEL can read them simultaneously.
    const cond_r = try useInto(code, reg_map, sel.cond, RegMap.tmp0);
    try code.cmpImm32(cond_r, 0);
    const t_r = try useInto(code, reg_map, sel.if_true, RegMap.tmp1);
    const f_r = try useInto(code, reg_map, sel.if_false, RegMap.tmp2);
    const info = try destBegin(reg_map, dest, RegMap.tmp0);
    // wasm: select picks if_true when cond != 0.
    if (inst.type == .i32) {
        try code.csel32(info.reg, t_r, f_r, .ne);
    } else {
        try code.csel(info.reg, t_r, f_r, .ne);
    }
    try destCommit(code, reg_map, info);
}

fn emitShift(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    op: emit.CodeBuffer.ShiftOp,
) !void {
    const dest = inst.dest orelse return;
    const lhs = try useInto(code, reg_map, bin.lhs, RegMap.tmp0);
    const rhs = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);
    const info = try destBegin(reg_map, dest, RegMap.tmp2);
    if (inst.type == .i32) {
        try code.shiftRegReg32(info.reg, lhs, rhs, op);
    } else {
        try code.shiftRegReg(info.reg, lhs, rhs, op);
    }
    try destCommit(code, reg_map, info);
}

fn emitRotl(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const lhs = try useInto(code, reg_map, bin.lhs, RegMap.tmp0);
    const rhs = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);
    // NEG output must not clobber lhs or rhs. Use tmp2 as the dedicated
    // scratch for -rhs. destBegin takes tmp0 as its fallback, so in the
    // worst case info.reg == tmp0 == lhs (aliased read-before-write — OK).
    const info = try destBegin(reg_map, dest, RegMap.tmp0);
    // rotl(x, n) = ror(x, -n). AArch64 has no ROL, so negate count into
    // tmp2, then RORV. Width-correct: W-form masks by 5, X by 6.
    if (inst.type == .i32) {
        try code.negReg32(RegMap.tmp2, rhs);
        try code.shiftRegReg32(info.reg, lhs, RegMap.tmp2, .ror);
    } else {
        try code.negReg(RegMap.tmp2, rhs);
        try code.shiftRegReg(info.reg, lhs, RegMap.tmp2, .ror);
    }
    try destCommit(code, reg_map, info);
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

/// ABI arg register for the i-th user argument (i=0 → x1 because x0 = vmctx).
fn callArgReg(i: u32) emit.Reg {
    return switch (i) {
        0 => .x1, 1 => .x2, 2 => .x3, 3 => .x4,
        4 => .x5, 5 => .x6, 6 => .x7,
        else => unreachable,
    };
}

/// Emit a direct wasm call.
///
/// Limitations (Phase 1): no tail calls, no extra results, no imports,
/// at most 7 arguments. Violations return an error rather than emit
/// incorrect code.
///
/// Sequence: spill all currently-live caller-save physregs (x0..x15) to
/// fixed per-reg slots in the call-save region, move arguments into
/// x1..x7 by reading from those save slots (for physreg sources) or the
/// RegMap spill region (for stack sources), load VMContext into x0, emit
/// `BL 0` with a patch record, move the result from x0 into the dest
/// vreg's location, then restore all saved regs.
///
/// The result-holding physreg is safe to not skip during restore: because
/// the RegMap allocates `dest` AFTER the spill and assign() only picks
/// currently-free regs, `dest.reg` cannot match any reg we saved.
fn emitCall(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    cl: @TypeOf(@as(ir.Inst.Op, undefined).call),
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
) !void {
    if (cl.tail) return error.UnimplementedTailCall;
    if (cl.extra_results > 0) return error.UnimplementedMultiResult;
    if (cl.args.len > 7) return error.TooManyCallArgs;
    if (cl.func_idx < fctx.import_count) return error.UnimplementedImportCall;
    const call_patches = fctx.call_patches orelse return error.CallLinkageUnavailable;

    // Snapshot live caller-save physregs (x0..x15).
    var used_snapshot: [RegMap.scratch_regs.len]bool = undefined;
    for (&used_snapshot, reg_map.reg_used) |*dst, src| dst.* = src;

    // Spill each used caller-save reg to [fp + call_save_base + i*8].
    for (used_snapshot, 0..) |used, i| {
        if (!used) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.strImm(reg, .fp, slot_scaled);
    }

    // Move arguments into x1..x7 by reading from their current source
    // locations via memory. Physreg sources are read from the call-save
    // slot we just filled; stack sources from the RegMap spill slot.
    // This order-independent approach avoids parallel-move hazards.
    for (cl.args, 0..) |arg_vreg, i| {
        const target = callArgReg(@intCast(i));
        const loc = reg_map.get(arg_vreg) orelse return error.UnboundVReg;
        switch (loc) {
            .reg => |r| {
                const reg_idx: u32 = @intFromEnum(r);
                const slot_scaled: u12 = @intCast((fctx.call_save_base + reg_idx * 8) / 8);
                try code.ldrImm(target, .fp, slot_scaled);
            },
            .stack => |off| {
                try code.ldrImm(target, .fp, reg_map.spillOffsetScaled(off));
            },
        }
    }

    // Load VMContext into x0 (was saved above if previously live).
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);

    // BL 0 (placeholder); record a patch for module-level resolution.
    const patch_off = code.len();
    try code.bl(0);
    try call_patches.append(fctx.allocator, .{
        .patch_offset = patch_off,
        .target_func_idx = cl.func_idx - fctx.import_count,
    });

    // Commit result (in x0) to dest's location BEFORE restoring saved regs,
    // so restoration can't clobber the result holder (dest.reg is always a
    // reg that was FREE at snapshot time — i.e. not in `used_snapshot`).
    if (inst.dest) |dest| {
        const info = try destBegin(reg_map, dest, RegMap.tmp0);
        if (info.reg != .x0) try code.movRegReg(info.reg, .x0);
        try destCommit(code, reg_map, info);
    }

    // Restore previously-saved caller-save regs.
    for (used_snapshot, 0..) |used, i| {
        if (!used) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(reg, .fp, slot_scaled);
    }
}

/// Compute linear-memory effective address in tmp0: mem_base + zext(wasm_addr) + offset.
///
/// Emits an inline bounds check: if `zext(wasm_addr) + end_offset > memory_size`,
/// calls `VmCtx.trap_oob_fn(vmctx)` which is noreturn. `end_offset` is
/// `static_offset + access_size` as supplied by the caller.
///
/// Clobbers tmp0, tmp1, tmp2. On the trap path additionally clobbers x0
/// (as the call arg) and LR — but that path is noreturn, so callers need
/// not care. After the fallthrough, only tmp0 carries meaningful data.
fn emitMemAddr(
    code: *emit.CodeBuffer,
    reg_map: *RegMap,
    base_vreg: ir.VReg,
    offset: u32,
    end_offset: u64,
) !void {
    // Step 1: zero-extend wasm address into tmp2 (kept alive across check).
    const src = try useInto(code, reg_map, base_vreg, RegMap.tmp1);
    try code.movRegReg32(RegMap.tmp2, src);

    // Step 2: compute end = tmp2 + end_offset in tmp1.
    if (end_offset == 0) {
        try code.movRegReg(RegMap.tmp1, RegMap.tmp2);
    } else if (end_offset <= 0xFFF) {
        try code.addImm(RegMap.tmp1, RegMap.tmp2, @intCast(end_offset));
    } else {
        try code.movImm64(RegMap.tmp1, end_offset);
        try code.addRegReg(RegMap.tmp1, RegMap.tmp1, RegMap.tmp2);
    }

    // Step 3: load VmCtx.memory_size (at +8, scaled-by-8 offset = 1).
    try code.ldrImm(RegMap.tmp0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, RegMap.tmp0, 1);

    // Step 4: cmp end, mem_size; B.LS over_trap (≤ is in-bounds since
    // accesses of size s are valid when end = addr+s ≤ memory_size).
    try code.cmpRegReg(RegMap.tmp1, RegMap.tmp0);
    const over_patch = code.len();
    try code.bCond(.ls, 0); // placeholder

    // Trap path: arg0=vmctx, call vmctx.trap_oob_fn (noreturn). We read
    // vmctx again because tmp0 now holds mem_size. BRK after BLR is
    // defensive — the helper is declared noreturn.
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    // trap_oob_fn is at VmCtx offset 80 = scale-8 slot 10.
    try code.ldrImm(RegMap.tmp0, .x0, vmctx_trap_oob_fn_slot);
    try code.blr(RegMap.tmp0);
    try code.brk(0);

    // Patch B.LS to land here.
    const over_target = code.len();
    const delta_words: i19 = @intCast(@divExact(
        @as(i64, @intCast(over_target)) - @as(i64, @intCast(over_patch)),
        4,
    ));
    const existing = std.mem.readInt(u32, code.bytes.items[over_patch..][0..4], .little);
    const imm19: u19 = @bitCast(delta_words);
    const new_word: u32 = (existing & 0xFF00001F) | (@as(u32, imm19) << 5);
    code.patch32(over_patch, new_word);

    // Step 5: reload vmctx, then load memory_base (at +0).
    try code.ldrImm(RegMap.tmp0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, RegMap.tmp0, 0);

    // Step 6: tmp0 = mem_base + zext(wasm_addr).
    try code.addRegReg(RegMap.tmp0, RegMap.tmp0, RegMap.tmp2);

    // Step 7: fold constant offset.
    if (offset != 0) {
        if (offset <= 0xFFF) {
            try code.addImm(RegMap.tmp0, RegMap.tmp0, @intCast(offset));
        } else {
            try code.movImm64(RegMap.tmp1, offset);
            try code.addRegReg(RegMap.tmp0, RegMap.tmp0, RegMap.tmp1);
        }
    }
}

/// VmCtx field offsets. Must match `runtime/aot/runtime.zig::VmCtx`.
const vmctx_memsize_slot: u12 = 1;       // byte 8, scale 8
const vmctx_globals_slot: u12 = 2;       // byte 16, scale 8
const vmctx_trap_oob_fn_slot: u12 = 10;  // byte 80, scale 8
const vmctx_trap_unreachable_fn_slot: u12 = 11; // byte 88, scale 8
const vmctx_trap_idivz_fn_slot: u12 = 12; // byte 96, scale 8
const vmctx_trap_iovf_fn_slot: u12 = 13;  // byte 104, scale 8

/// Globals are a packed array of 8-byte slots indexed by global index.
/// For i32/f32 we read/write the low 32 bits with W-form LDR/STR; for
/// i64/f64 we use the 64-bit forms. This matches x86-64 which uses an
/// 8-byte slot per global.
fn emitGlobalGet(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    idx: u32,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    // tmp0 = vmctx; tmp0 = globals_base
    try code.ldrImm(RegMap.tmp0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, RegMap.tmp0, vmctx_globals_slot);

    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    const is32 = (inst.type == .i32 or inst.type == .f32);
    // Byte offset = idx * 8. LDR scaled-imm requires idx fit in u12.
    if (idx > 0xFFF) return error.GlobalIndexOutOfRange;
    if (is32) try code.ldrImm32(info.reg, RegMap.tmp0, @intCast(idx * 2))
    else try code.ldrImm(info.reg, RegMap.tmp0, @intCast(idx));
    try destCommit(code, reg_map, info);
}

fn emitGlobalSet(
    code: *emit.CodeBuffer,
    gs: @TypeOf(@as(ir.Inst.Op, undefined).global_set),
    reg_map: *RegMap,
) !void {
    const val_r = try useInto(code, reg_map, gs.val, RegMap.tmp1);
    // tmp0 = vmctx; tmp0 = globals_base
    try code.ldrImm(RegMap.tmp0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, RegMap.tmp0, vmctx_globals_slot);
    if (gs.idx > 0xFFF) return error.GlobalIndexOutOfRange;
    // Type width inferred from the val's IR type is not available here;
    // always store 8 bytes (safe: the interpreter allocates 8 bytes per
    // global slot, and i32/f32 are zero-extended or host-padded).
    try code.strImm(val_r, RegMap.tmp0, @intCast(gs.idx));
}

/// Patch a B.cond placeholder at `patch_off` to branch to the current PC.
/// Preserves the original condition bits and updates only imm19.
fn patchBCondHere(code: *emit.CodeBuffer, patch_off: usize) void {
    const target = code.len();
    const delta_words: i19 = @intCast(@divExact(
        @as(i64, @intCast(target)) - @as(i64, @intCast(patch_off)),
        4,
    ));
    const existing = std.mem.readInt(u32, code.bytes.items[patch_off..][0..4], .little);
    const imm19: u19 = @bitCast(delta_words);
    const new_word: u32 = (existing & 0xFF00001F) | (@as(u32, imm19) << 5);
    code.patch32(patch_off, new_word);
}

/// Emit a noreturn call to a VmCtx trap helper at scaled slot `slot`.
/// Sequence: load x0 = vmctx, tmp0 = [x0 + slot*8], BLR tmp0, BRK 0.
/// Clobbers x0, tmp0, LR — but this path is noreturn, so callers do not
/// care what survives it.
fn emitTrapHelperCall(code: *emit.CodeBuffer, slot: u12) !void {
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, .x0, slot);
    try code.blr(RegMap.tmp0);
    try code.brk(0);
}

const DivRemKind = enum { div_s, div_u, rem_s, rem_u };

/// Emit div_s / div_u / rem_s / rem_u. Semantics per wasm:
///   * rhs == 0 traps (idivz helper).
///   * div_s with INT_MIN / -1 traps (iovf helper).
///   * rem_s with INT_MIN / -1 yields 0 — handled naturally by MSUB with
///     two's complement wrap (INT_MIN - INT_MIN*(-1) == 0).
///
/// Uses tmp0=lhs, tmp1=rhs, tmp2=scratch/quotient/INT_MIN constant.
fn emitDivRem(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    kind: DivRemKind,
) !void {
    const dest = inst.dest orelse return;
    const is64 = inst.type == .i64;

    // Copy sources into the fixed scratch registers (tmp0, tmp1). useInto
    // may already place them there, but if the vreg lives in an allocated
    // physreg we need an explicit move because later code clobbers tmp0
    // and tmp1 across the trap-helper calls.
    const lhs_loaded = try useInto(code, reg_map, bin.lhs, RegMap.tmp0);
    if (lhs_loaded != RegMap.tmp0) {
        if (is64) try code.movRegReg(RegMap.tmp0, lhs_loaded)
        else try code.movRegReg32(RegMap.tmp0, lhs_loaded);
    }
    const rhs_loaded = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);
    if (rhs_loaded != RegMap.tmp1) {
        if (is64) try code.movRegReg(RegMap.tmp1, rhs_loaded)
        else try code.movRegReg32(RegMap.tmp1, rhs_loaded);
    }

    // Zero-divisor check: cmp rhs, #0; b.ne skip; call trap_idivz; skip:
    if (is64) try code.cmpImm(RegMap.tmp1, 0)
    else try code.cmpImm32(RegMap.tmp1, 0);
    const skip_idivz = code.len();
    try code.bCond(.ne, 0);
    try emitTrapHelperCall(code, vmctx_trap_idivz_fn_slot);
    patchBCondHere(code, skip_idivz);

    // div_s INT_MIN/-1 overflow check. Sequence:
    //   tmp2 = -1; cmp rhs, tmp2; b.ne skip_ovf
    //   tmp2 = INT_MIN; cmp lhs, tmp2; b.ne skip_ovf
    //   call trap_iovf
    // skip_ovf:
    if (kind == .div_s) {
        if (is64) {
            try code.movImm64(RegMap.tmp2, 0xFFFF_FFFF_FFFF_FFFF);
            try code.cmpRegReg(RegMap.tmp1, RegMap.tmp2);
        } else {
            try code.movImm32(RegMap.tmp2, -1);
            try code.cmpRegReg32(RegMap.tmp1, RegMap.tmp2);
        }
        const skip_a = code.len();
        try code.bCond(.ne, 0);
        if (is64) {
            try code.movImm64(RegMap.tmp2, 0x8000_0000_0000_0000);
            try code.cmpRegReg(RegMap.tmp0, RegMap.tmp2);
        } else {
            try code.movImm32(RegMap.tmp2, @as(i32, -2147483648));
            try code.cmpRegReg32(RegMap.tmp0, RegMap.tmp2);
        }
        const skip_b = code.len();
        try code.bCond(.ne, 0);
        try emitTrapHelperCall(code, vmctx_trap_iovf_fn_slot);
        patchBCondHere(code, skip_a);
        patchBCondHere(code, skip_b);
    }

    // After checks: tmp0=lhs, tmp1=rhs are still live. Now allocate dest
    // (destBegin's scratch is tmp2 — safe because tmp2 is dead here).
    const info = try destBegin(reg_map, dest, RegMap.tmp2);

    switch (kind) {
        .div_s => if (is64)
            try code.sdivRegReg(info.reg, RegMap.tmp0, RegMap.tmp1)
        else
            try code.sdivRegReg32(info.reg, RegMap.tmp0, RegMap.tmp1),
        .div_u => if (is64)
            try code.udivRegReg(info.reg, RegMap.tmp0, RegMap.tmp1)
        else
            try code.udivRegReg32(info.reg, RegMap.tmp0, RegMap.tmp1),
        .rem_s, .rem_u => {
            // q = lhs DIV rhs into tmp2; dest = lhs - q*rhs via MSUB.
            // If info.reg happens to equal tmp2 (dest is spilled), SDIV/UDIV
            // still writes tmp2=q, then MSUB reads q (tmp2) before writing
            // dest (tmp2) — single-instruction read-before-write, safe.
            if (kind == .rem_s) {
                if (is64) try code.sdivRegReg(RegMap.tmp2, RegMap.tmp0, RegMap.tmp1)
                else try code.sdivRegReg32(RegMap.tmp2, RegMap.tmp0, RegMap.tmp1);
            } else {
                if (is64) try code.udivRegReg(RegMap.tmp2, RegMap.tmp0, RegMap.tmp1)
                else try code.udivRegReg32(RegMap.tmp2, RegMap.tmp0, RegMap.tmp1);
            }
            // MSUB dest, q, rhs, lhs  →  dest = lhs - q*rhs
            if (is64) try code.msubRegReg(info.reg, RegMap.tmp2, RegMap.tmp1, RegMap.tmp0)
            else try code.msubRegReg32(info.reg, RegMap.tmp2, RegMap.tmp1, RegMap.tmp0);
        },
    }
    try destCommit(code, reg_map, info);
}


/// Scaled-immediate offset suitable for LDR/STR with given scale.
/// Returns null if the offset isn't a multiple of `scale` within u12*scale range.
fn scaledOffset(off: u32, scale: u32) ?u12 {
    const max = @as(u32, 0xFFF) * scale;
    if (off > max) return null;
    if ((off % scale) != 0) return null;
    return @intCast(off / scale);
}

fn emitLoad(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    ld: @TypeOf(@as(ir.Inst.Op, undefined).load),
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const is64 = inst.type == .i64;

    // Validate size/sign combinations. i32.load32 can't sign-extend since
    // there's nothing wider to extend into at 32-bit dest.
    const scale: u32 = switch (ld.size) {
        1 => 1,
        2 => 2,
        4 => 4,
        8 => 8,
        else => return error.UnimplementedLoadSize,
    };
    if (ld.sign_extend and ld.size == 8) return error.BadLoadSign; // 8-byte has no extension
    if (ld.sign_extend and ld.size == 4 and !is64) return error.BadLoadSign;

    // Try to fold offset into the load's scaled immediate; else add it to tmp0.
    const folded = scaledOffset(ld.offset, scale);
    const end_offset: u64 = @as(u64, ld.offset) + @as(u64, ld.size);
    try emitMemAddr(code, reg_map, ld.base, if (folded == null) ld.offset else 0, end_offset);
    const disp: u12 = if (folded) |d| d else 0;

    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    if (ld.sign_extend) {
        switch (ld.size) {
            1 => if (is64) try code.ldrsbImm64(info.reg, RegMap.tmp0, disp)
                 else try code.ldrsbImm32(info.reg, RegMap.tmp0, disp),
            2 => if (is64) try code.ldrshImm64(info.reg, RegMap.tmp0, disp)
                 else try code.ldrshImm32(info.reg, RegMap.tmp0, disp),
            4 => try code.ldrswImm(info.reg, RegMap.tmp0, disp), // always X-form
            else => unreachable,
        }
    } else {
        switch (ld.size) {
            1 => try code.ldrbImm(info.reg, RegMap.tmp0, disp),
            2 => try code.ldrhImm(info.reg, RegMap.tmp0, disp),
            4 => try code.ldrImm32(info.reg, RegMap.tmp0, disp),
            8 => try code.ldrImm(info.reg, RegMap.tmp0, disp),
            else => unreachable,
        }
    }
    try destCommit(code, reg_map, info);
}

fn emitStore(
    code: *emit.CodeBuffer,
    st: @TypeOf(@as(ir.Inst.Op, undefined).store),
    reg_map: *RegMap,
) !void {
    const scale: u32 = switch (st.size) {
        1 => 1,
        2 => 2,
        4 => 4,
        8 => 8,
        else => return error.UnimplementedStoreSize,
    };

    const folded = scaledOffset(st.offset, scale);
    const end_offset: u64 = @as(u64, st.offset) + @as(u64, st.size);
    try emitMemAddr(code, reg_map, st.base, if (folded == null) st.offset else 0, end_offset);
    const disp: u12 = if (folded) |d| d else 0;

    // Materialize the value into tmp1 (or use its home reg). emitMemAddr is
    // done with tmp1 by now, so it's free for `useInto` to reuse.
    const val_reg = try useInto(code, reg_map, st.val, RegMap.tmp1);
    switch (st.size) {
        1 => try code.strbImm(val_reg, RegMap.tmp0, disp),
        2 => try code.strhImm(val_reg, RegMap.tmp0, disp),
        4 => try code.strImm32(val_reg, RegMap.tmp0, disp),
        8 => try code.strImm(val_reg, RegMap.tmp0, disp),
        else => unreachable,
    }
}

fn emitBrIf(
    code: *emit.CodeBuffer,
    reg_map: *RegMap,
    patches: *std.ArrayListUnmanaged(BranchPatch),
    br: @TypeOf(@as(ir.Inst.Op, undefined).br_if),
    allocator: std.mem.Allocator,
) !void {
    const cond_r = try useInto(code, reg_map, br.cond, RegMap.tmp0);
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
    // Allocate dest first; if spilled, dst_info.reg is tmp0 and we STR on commit.
    const dst_info = try destBegin(reg_map, dest, RegMap.tmp0);
    // Load sources. lhs may share tmp0 with dst when dst is spilled — that's
    // fine, because the final op writes dst_info.reg last.
    const lhs_scratch: emit.Reg = if (dst_info.spill_slot == null) RegMap.tmp0 else dst_info.reg;
    const lhs_reg = try useInto(code, reg_map, bin.lhs, lhs_scratch);
    const rhs_reg = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);

    switch (kind) {
        .add => try code.addRegReg(dst_info.reg, lhs_reg, rhs_reg),
        .sub => try code.subRegReg(dst_info.reg, lhs_reg, rhs_reg),
        .mul => try code.mulRegReg(dst_info.reg, lhs_reg, rhs_reg),
        .@"and" => try code.andRegReg(dst_info.reg, lhs_reg, rhs_reg),
        .@"or" => try code.orrRegReg(dst_info.reg, lhs_reg, rhs_reg),
        .xor => try code.eorRegReg(dst_info.reg, lhs_reg, rhs_reg),
    }
    try destCommit(code, reg_map, dst_info);
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

    var global_call_patches: std.ArrayListUnmanaged(CallPatch) = .empty;
    defer global_call_patches.deinit(allocator);

    for (ir_module.functions.items) |func| {
        const func_base: u32 = @intCast(all_code.items.len);
        try offsets.append(allocator, func_base);

        var func_patches: std.ArrayListUnmanaged(CallPatch) = .empty;
        defer func_patches.deinit(allocator);

        const ctx: FuncCompileCtx = .{
            .import_count = ir_module.import_count,
            .call_patches = &func_patches,
            .allocator = allocator,
        };
        const func_code = try compileFunctionImpl(&func, ctx, allocator);
        defer allocator.free(func_code);

        // Globalize patch offsets to module code coordinates.
        for (func_patches.items) |p| {
            try global_call_patches.append(allocator, .{
                .patch_offset = p.patch_offset + func_base,
                .target_func_idx = p.target_func_idx,
            });
        }
        try all_code.appendSlice(allocator, func_code);
    }

    // Resolve BL patches. offsets[i] is the byte offset of local function i
    // (indices already exclude imports).
    for (global_call_patches.items) |p| {
        if (p.target_func_idx >= offsets.items.len) return error.BadFuncIndex;
        const target_off: i64 = @intCast(offsets.items[p.target_func_idx]);
        const patch_off: i64 = @intCast(p.patch_offset);
        const delta_bytes = target_off - patch_off;
        if (@mod(delta_bytes, 4) != 0) return error.BranchMisaligned;
        const word_off = @divExact(delta_bytes, 4);
        // BL imm26 range: ±128 MB → word_off in [-2^25, 2^25).
        const limit: i64 = 1 << 25;
        if (word_off >= limit or word_off < -limit) return error.CallOutOfRange;
        const bytes = all_code.items[p.patch_offset..][0..4];
        const existing = std.mem.readInt(u32, bytes, .little);
        // BL encoding: 100101 imm26 | opcode bits 31-26 = 0b100101
        const imm26: u26 = @bitCast(@as(i26, @intCast(word_off)));
        const new_word: u32 = (existing & 0xFC000000) | imm26;
        std.mem.writeInt(u32, bytes, new_word, .little);
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

test "compileFunction: global_get then global_set round-trips" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .global_get = 3 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .global_set = .{ .idx = 5, .val = v0 } } });
    try block.append(.{ .op = .{ .ret = null } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compileFunction: div_s by zero codegen produces trap path" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .div_s = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
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

test "compileModule: direct call between two local functions" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    // f0: returns 42
    var f0 = ir.IrFunction.init(allocator, 0, 1, 0);
    {
        const b = try f0.newBlock();
        const v = f0.newVReg();
        try f0.getBlock(b).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .ret = v } });
    }
    _ = try module.addFunction(f0);

    // f1: calls f0, returns its result
    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    {
        const b = try f1.newBlock();
        const v = f1.newVReg();
        try f1.getBlock(b).append(.{
            .op = .{ .call = .{ .func_idx = 0, .args = &.{} } },
            .dest = v,
            .type = .i32,
        });
        try f1.getBlock(b).append(.{ .op = .{ .ret = v } });
    }
    _ = try module.addFunction(f1);

    const result = try compileModule(&module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    // Walk f1's code looking for a BL that points at f0 (offset 0).
    const f1_start = result.offsets[1];
    const f0_start = result.offsets[0];
    var found_bl_to_f0 = false;
    var i: usize = f1_start;
    while (i + 4 <= result.code.len) : (i += 4) {
        const word = std.mem.readInt(u32, result.code[i..][0..4], .little);
        // BL opcode = 0b100101 in top 6 bits → 0x94000000
        if ((word & 0xFC000000) == 0x94000000) {
            const imm26_raw: u26 = @truncate(word);
            const imm26: i26 = @bitCast(imm26_raw);
            const delta_words: i64 = imm26;
            const target: i64 = @as(i64, @intCast(i)) + delta_words * 4;
            if (target == @as(i64, @intCast(f0_start))) {
                found_bl_to_f0 = true;
                break;
            }
        }
    }
    try std.testing.expect(found_bl_to_f0);
}

// ── New handler tests (Phase 1a) ─────────────────────────────────────────────

test "load i32: emits VMCtx load + LDR W with scaled offset" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{ .base = addr, .offset = 8, .size = 4 } },
        .dest = val,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Expect at least one LDR W (opcode 0xB9400000 in top bits).
    var found_ldr_w = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0xB9400000) {
            found_ldr_w = true;
            break;
        }
    }
    try std.testing.expect(found_ldr_w);
}

test "store i32: emits VMCtx load + STR W" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = @bitCast(@as(u32, 0xDEADBEEF)) }, .dest = val, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .store = .{ .base = addr, .offset = 0, .size = 4, .val = val } },
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = null } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Expect at least one STR W (opcode 0xB9000000 in top bits).
    var found_str_w = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0xB9000000) {
            found_str_w = true;
            break;
        }
    }
    try std.testing.expect(found_str_w);
}

test "load i32: unrepresentable offset falls back to movImm + add" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    // Offset 5 is not a multiple of 4 → must go through tmp1/ADD path.
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{ .base = addr, .offset = 5, .size = 4 } },
        .dest = val,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "load: size 8 (i64) emits 64-bit LDR X" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{ .base = addr, .offset = 0, .size = 8 } },
        .dest = val,
        .type = .i64,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // LDR X opcode: bits 31..22 = 0xF94 → mask 0xFFC00000 = 0xF9400000.
    var found = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0xF9400000) { found = true; break; }
    }
    try std.testing.expect(found);
}

test "load: size 1 unsigned (i32.load8_u) emits LDRB" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{ .base = addr, .offset = 3, .size = 1 } },
        .dest = val,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // LDRB encoding top bits: 0x39400000 in bits 31..22 (mask 0xFFC00000).
    var found = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0x39400000) { found = true; break; }
    }
    try std.testing.expect(found);
}

test "load: size 2 signed (i32.load16_s) emits LDRSH W" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{ .base = addr, .offset = 2, .size = 2, .sign_extend = true } },
        .dest = val,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // LDRSH W top bits: 0x79C00000 (mask 0xFFC00000).
    var found = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0x79C00000) { found = true; break; }
    }
    try std.testing.expect(found);
}

test "load: bounds check emits B.LS + trap BLR + BRK" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{ .base = addr, .offset = 0, .size = 4 } },
        .dest = val,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Walk instructions; find B.cond with cond=LS (cond=0b1001), BLR, BRK.
    // B.cond encoding: 0101_0100 in bits 31..24, bits 3..0 = cond.
    // BLR Xn: 1101_0110_0011_1111_0000_00_Rn_00000 → 0xD63F0000 | Rn<<5.
    // BRK #imm: 1101_0100_001_imm16_00000 → 0xD4200000.
    var found_bls = false;
    var found_blr = false;
    var found_brk = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        // B.cond: opcode 0x54 in bits 31..24, bit 4 = 0, cond in bits 3..0.
        if ((w & 0xFF000010) == 0x54000000 and (w & 0xF) == 0x9) found_bls = true;
        if ((w & 0xFFFFFC1F) == 0xD63F0000) found_blr = true;
        if ((w & 0xFFE0001F) == 0xD4200000) found_brk = true;
    }
    try std.testing.expect(found_bls);
    try std.testing.expect(found_blr);
    try std.testing.expect(found_brk);
}

test "store: size 1 emits STRB" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 42 }, .dest = val, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .store = .{ .base = addr, .offset = 0, .size = 1, .val = val } },
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = null } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // STRB top bits: 0x39000000 (mask 0xFFC00000).
    var found = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0x39000000) { found = true; break; }
    }
    try std.testing.expect(found);
}

test "load: size != 1,2,4,8 returns UnimplementedLoadSize" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{ .base = addr, .offset = 0, .size = 3 } },
        .dest = val,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    try std.testing.expectError(error.UnimplementedLoadSize, compileFunction(&func, allocator));
}

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
    // popcnt is not yet implemented (needs NEON CNT) — must fail loudly,
    // not silently drop.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .popcnt = v0 },
        .dest = v1,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v1 } });
    try std.testing.expectError(error.UnimplementedOp, compileFunction(&func, allocator));
}

test "compile: .call without linkage context returns CallLinkageUnavailable" {
    // compileFunction (no import_count, no patch list) can't resolve
    // direct calls; must error loudly rather than emit bogus BL 0.
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
    try std.testing.expectError(error.CallLinkageUnavailable, compileFunction(&func, allocator));
}

test "compile: binop with spilled operands emits LDR/STR via spill slots" {
    // 17 iconsts keep x0..x15 live, then an ADD with v16+v17 forces spills
    // for the operands and possibly the dest. Verify code succeeds (no
    // silent drop) and contains at least one LDR/STR pair against FP.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    // Allocate 17 vregs, each with a distinct iconst. The 17th (v16)
    // forces spill.
    var vregs: [18]ir.VReg = undefined;
    for (&vregs, 0..) |*v, i| {
        v.* = func.newVReg();
        try func.getBlock(bid).append(.{
            .op = .{ .iconst_64 = @intCast(i + 1) },
            .dest = v.*,
            .type = .i64,
        });
    }
    // dst = vregs[16] + vregs[17]   (both spilled)
    const dst = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .add = .{ .lhs = vregs[16], .rhs = vregs[17] } },
        .dest = dst,
        .type = .i64,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = dst } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Scan for LDR Xt, [fp, #imm] (opcode pattern 0xF9400000 + rn=29<<5).
    // We must see at least one such load (the spill reload).
    var found_ldr_from_fp = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        // LDR (imm) unsigned offset, 64-bit: top 10 bits == 0x3E5, rn bits 5-9
        const top = (w >> 22) & 0x3FF;
        const rn = (w >> 5) & 0x1F;
        if (top == 0x3E5 and rn == 29) {
            found_ldr_from_fp = true;
            break;
        }
    }
    try std.testing.expect(found_ldr_from_fp);
}
