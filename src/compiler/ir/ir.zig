//! Compiler Intermediate Representation (IR).
//!
//! A lightweight SSA-form IR for compiling WebAssembly to native code.
//! Each Wasm function is lowered into a sequence of basic blocks containing
//! IR instructions that can be directly mapped to machine code.

const std = @import("std");

/// Virtual register index.
pub const VReg = u32;

/// Basic block index.
pub const BlockId = u32;

/// IR value types (maps closely to machine types).
pub const IrType = enum {
    i32,
    i64,
    f32,
    f64,
    v128,
    void,

    pub fn byteSize(self: IrType) u8 {
        return switch (self) {
            .i32, .f32 => 4,
            .i64, .f64 => 8,
            .v128 => 16,
            .void => 0,
        };
    }

    /// Number of 8-byte frame slots needed to spill this value.
    pub fn spillSlots64(self: IrType) u8 {
        return switch (self) {
            .v128 => 2,
            .void => 0,
            else => 1,
        };
    }

    /// Alignment required for stack storage, in 8-byte frame slots.
    pub fn spillAlignSlots64(self: IrType) u8 {
        return switch (self) {
            .v128 => 2,
            .void => 1,
            else => 1,
        };
    }
};

/// An IR instruction.
pub const Inst = struct {
    op: Op,
    dest: ?VReg = null,
    type: IrType = .i32,

    pub const Op = union(enum) {
        // Constants
        iconst_32: i32,
        iconst_64: i64,
        fconst_32: f32,
        fconst_64: f64,

        // SIMD/v128 foundation. These IR forms are intentionally narrow:
        // they cover the first AArch64 NEON slice without implying full
        // wasm SIMD coverage or exported v128 ABI support.
        v128_const: u128,
        v128_load: V128Mem,
        v128_store: V128Store,
        v128_not: VReg,
        v128_bitwise: V128Bitwise,
        i32x4_binop: I32x4BinOp,
        i32x4_shift: I32x4Shift,
        i32x4_splat: VReg,
        i32x4_extract_lane: I32x4ExtractLane,
        i32x4_replace_lane: I32x4ReplaceLane,
        i8x16_binop: I8x16BinOp,
        i8x16_shift: I8x16Shift,
        i8x16_splat: VReg,
        i8x16_extract_lane: I8x16ExtractLane,
        i8x16_replace_lane: I8x16ReplaceLane,
        i16x8_binop: I16x8BinOp,
        i16x8_shift: I16x8Shift,
        i16x8_splat: VReg,
        i16x8_extract_lane: I16x8ExtractLane,
        i16x8_replace_lane: I16x8ReplaceLane,
        i64x2_binop: I64x2BinOp,
        i64x2_splat: VReg,
        i64x2_extract_lane: I64x2ExtractLane,
        i64x2_replace_lane: I64x2ReplaceLane,

        // Binary arithmetic (dest = lhs op rhs)
        add: BinOp,
        sub: BinOp,
        mul: BinOp,
        div_s: BinOp,
        div_u: BinOp,
        rem_s: BinOp,
        rem_u: BinOp,
        @"and": BinOp,
        @"or": BinOp,
        xor: BinOp,
        shl: BinOp,
        shr_s: BinOp,
        shr_u: BinOp,
        rotl: BinOp,
        rotr: BinOp,

        // Unary
        clz: VReg,
        ctz: VReg,
        popcnt: VReg,
        eqz: VReg,

        // Comparisons (result is i32 0 or 1)
        eq: BinOp,
        ne: BinOp,
        lt_s: BinOp,
        lt_u: BinOp,
        gt_s: BinOp,
        gt_u: BinOp,
        le_s: BinOp,
        le_u: BinOp,
        ge_s: BinOp,
        ge_u: BinOp,

        // Local variable access
        local_get: u32,
        local_set: struct { idx: u32, val: VReg },

        // Memory
        // checked_end: when non-zero, codegen uses this (instead of offset+size)
        // for the bounds check, enabling a single widened check to cover multiple
        // accesses sharing the same base within a basic block segment.
        load: struct { base: VReg, offset: u32, size: u8, sign_extend: bool = false, bounds_known: bool = false, checked_end: u64 = 0 },
        store: struct { base: VReg, offset: u32, size: u8, val: VReg, bounds_known: bool = false, checked_end: u64 = 0 },

        // Control flow
        br: BlockId,
        br_if: struct { cond: VReg, then_block: BlockId, else_block: BlockId },
        br_table: struct { index: VReg, targets: []const BlockId, default: BlockId },
        ret: ?VReg,
        // Multi-value return: first VReg -> RAX, remaining written to
        // memory via the hidden return pointer passed by the caller.
        ret_multi: []const VReg,
        @"unreachable": void,

        // Function calls. `extra_results` is the number of additional
        // results beyond the one returned in RAX (i.e. callee.result_count - 1).
        // When > 0, the caller passes a hidden return pointer as an implicit
        // trailing argument; the callee writes extras into [hrp + i*8], and
        // the caller retrieves them via `.call_result` ops emitted right
        // after the call. The primary result is delivered in `inst.dest` as
        // before (via RAX).
        call: struct { func_idx: u32, args: []const VReg = &.{}, extra_results: u8 = 0, tail: bool = false },
        call_indirect: struct { type_idx: u32, elem_idx: VReg, args: []const VReg = &.{}, extra_results: u8 = 0, table_idx: u32 = 0, tail: bool = false },
        call_ref: struct { type_idx: u32, func_ref: VReg, args: []const VReg = &.{}, extra_results: u8 = 0, tail: bool = false },

        // Retrieve the i-th extra result (i is 0-based among extras; i=0 is
        // the callee's 2nd result). Must immediately follow the corresponding
        // `.call`/`.call_indirect`. `inst.dest` receives the value; codegen
        // reads it from the caller's pre-reserved scratch slot.
        call_result: u8,

        // Parametric
        select: struct { cond: VReg, if_true: VReg, if_false: VReg },

        // Global access
        global_get: u32,
        global_set: struct { idx: u32, val: VReg },

        // Sign extension
        extend8_s: VReg,
        extend16_s: VReg,
        extend32_s: VReg,

        // Float unary
        f_neg: VReg,
        f_abs: VReg,
        f_sqrt: VReg,
        f_ceil: VReg,
        f_floor: VReg,
        f_trunc: VReg,
        f_nearest: VReg,

        // Float binary
        f_min: BinOp,
        f_max: BinOp,
        f_copysign: BinOp,

        // Float comparisons (result is i32; operand type via inst.type .f32/.f64)
        f_eq: BinOp,
        f_ne: BinOp,
        f_lt: BinOp,
        f_gt: BinOp,
        f_le: BinOp,
        f_ge: BinOp,

        // Conversions
        wrap_i64: VReg,
        extend_i32_s: VReg,
        extend_i32_u: VReg,

        // Type conversions
        trunc_f32_s: VReg,
        trunc_f32_u: VReg,
        trunc_f64_s: VReg,
        trunc_f64_u: VReg,
        convert_s: VReg,
        convert_u: VReg,
        convert_i32_s: VReg,
        convert_i64_s: VReg,
        convert_i32_u: VReg,
        convert_i64_u: VReg,
        demote_f64: VReg,
        promote_f32: VReg,
        reinterpret: VReg,
        trunc_sat_f32_s: VReg,
        trunc_sat_f32_u: VReg,
        trunc_sat_f64_s: VReg,
        trunc_sat_f64_u: VReg,

        // Atomic operations
        atomic_load: struct { base: VReg, offset: u32, size: u8 },
        atomic_store: struct { base: VReg, offset: u32, size: u8, val: VReg },
        atomic_rmw: struct { base: VReg, offset: u32, size: u8, val: VReg, op: AtomicRmwOp },
        atomic_cmpxchg: struct { base: VReg, offset: u32, size: u8, expected: VReg, replacement: VReg },
        atomic_fence: void,
        atomic_notify: struct { base: VReg, offset: u32, count: VReg },
        atomic_wait: struct { base: VReg, offset: u32, expected: VReg, timeout: VReg, size: u8 },

        // Bulk memory operations
        memory_copy: struct { dst: VReg, src: VReg, len: VReg },
        memory_fill: struct { dst: VReg, val: VReg, len: VReg },
        memory_init: struct { seg_idx: u32, dst: VReg, src: VReg, len: VReg },
        data_drop: u32, // data segment index

        // Memory management
        memory_size: void,
        memory_grow: VReg,

        // Table operations
        table_size: u32, // table_idx
        table_get: struct { table_idx: u32, idx: VReg },
        table_set: struct { table_idx: u32, idx: VReg, val: VReg },
        table_grow: struct { table_idx: u32, init: VReg, delta: VReg }, // -> i32 (prev size or -1)
        table_init: struct { seg_idx: u32, table_idx: u32, dst: VReg, src: VReg, len: VReg },
        elem_drop: u32, // element segment index
        ref_func: u32, // funcidx -> native pointer loaded from vmctx.func_table[idx]

        // SSA phi: merges values from predecessor edges at join points.
        // Inserted by mem2reg and lowered before codegen.
        phi: []const PhiEdge,
    };

    pub const BinOp = struct {
        lhs: VReg,
        rhs: VReg,
    };

    pub const AtomicRmwOp = enum { add, sub, @"and", @"or", xor, xchg };

    pub const V128BitwiseOp = enum { @"and", andnot, @"or", xor };

    pub const I32x4Op = enum {
        add,
        sub,
        eq,
        ne,
        lt_s,
        lt_u,
        gt_s,
        gt_u,
        le_s,
        le_u,
        ge_s,
        ge_u,
        mul,
    };

    pub const I8x16Op = enum {
        add,
        sub,
        eq,
        ne,
        lt_s,
        lt_u,
        gt_s,
        gt_u,
        le_s,
        le_u,
        ge_s,
        ge_u,
    };

    pub const I16x8Op = enum {
        add,
        sub,
        eq,
        ne,
        lt_s,
        lt_u,
        gt_s,
        gt_u,
        le_s,
        le_u,
        ge_s,
        ge_u,
        mul,
    };

    pub const I64x2Op = enum {
        add,
        sub,
        eq,
        ne,
        lt_s,
        gt_s,
        le_s,
        ge_s,
    };

    pub const I32x4ShiftOp = enum {
        shl,
        shr_s,
        shr_u,
    };

    pub const I16x8ShiftOp = enum {
        shl,
        shr_s,
        shr_u,
    };

    pub const I8x16ShiftOp = enum {
        shl,
        shr_s,
        shr_u,
    };

    pub const V128Mem = struct {
        base: VReg,
        offset: u32,
        alignment: u32,
        bounds_known: bool = false,
        checked_end: u64 = 0,
    };

    pub const V128Store = struct {
        base: VReg,
        offset: u32,
        alignment: u32,
        val: VReg,
        bounds_known: bool = false,
        checked_end: u64 = 0,
    };

    pub const V128Bitwise = struct {
        op: V128BitwiseOp,
        lhs: VReg,
        rhs: VReg,
    };

    pub const I32x4BinOp = struct {
        op: I32x4Op,
        lhs: VReg,
        rhs: VReg,
    };

    pub const I8x16BinOp = struct {
        op: I8x16Op,
        lhs: VReg,
        rhs: VReg,
    };

    pub const I16x8BinOp = struct {
        op: I16x8Op,
        lhs: VReg,
        rhs: VReg,
    };

    pub const I64x2BinOp = struct {
        op: I64x2Op,
        lhs: VReg,
        rhs: VReg,
    };

    pub const I32x4Shift = struct {
        op: I32x4ShiftOp,
        vector: VReg,
        count: VReg,
    };

    pub const I16x8Shift = struct {
        op: I16x8ShiftOp,
        vector: VReg,
        count: VReg,
    };

    pub const I8x16Shift = struct {
        op: I8x16ShiftOp,
        vector: VReg,
        count: VReg,
    };

    pub const I32x4ExtractLane = struct {
        vector: VReg,
        lane: u2,
    };

    pub const I8x16LaneSign = enum { signed, unsigned };

    pub const I8x16ExtractLane = struct {
        vector: VReg,
        lane: u4,
        sign: I8x16LaneSign,
    };

    pub const I16x8LaneSign = enum { signed, unsigned };

    pub const I16x8ExtractLane = struct {
        vector: VReg,
        lane: u3,
        sign: I16x8LaneSign,
    };

    pub const I64x2ExtractLane = struct {
        vector: VReg,
        lane: u1,
    };

    pub const I32x4ReplaceLane = struct {
        vector: VReg,
        val: VReg,
        lane: u2,
    };

    pub const I8x16ReplaceLane = struct {
        vector: VReg,
        val: VReg,
        lane: u4,
    };

    pub const I16x8ReplaceLane = struct {
        vector: VReg,
        val: VReg,
        lane: u3,
    };

    pub const I64x2ReplaceLane = struct {
        vector: VReg,
        val: VReg,
        lane: u1,
    };

    pub const PhiEdge = struct {
        block: BlockId,
        val: VReg,
    };
};

/// A basic block — a sequence of instructions with a single entry point.
pub const BasicBlock = struct {
    id: BlockId,
    instructions: std.ArrayList(Inst) = .empty,
    /// Predecessor block IDs (for SSA analysis).
    predecessors: std.ArrayList(BlockId) = .empty,
    allocator: std.mem.Allocator,

    pub fn init(id: BlockId, allocator: std.mem.Allocator) BasicBlock {
        return .{
            .id = id,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BasicBlock) void {
        for (self.instructions.items) |inst| {
            if (inst.op == .phi) self.allocator.free(inst.op.phi);
        }
        self.instructions.deinit(self.allocator);
        self.predecessors.deinit(self.allocator);
    }

    pub fn append(self: *BasicBlock, inst: Inst) !void {
        try self.instructions.append(self.allocator, inst);
    }

    pub fn addPredecessor(self: *BasicBlock, pred_id: BlockId) !void {
        try self.predecessors.append(self.allocator, pred_id);
    }
};

/// An IR function — the compilation unit.
pub const IrFunction = struct {
    name: ?[]const u8 = null,
    param_count: u32,
    result_count: u32,
    local_count: u32,
    /// Per-local IR type (params first, then declared locals, then synthetic).
    /// Populated by the frontend; used by mem2reg for typed-zero seeding.
    local_types: ?[]const IrType = null,
    blocks: std.ArrayList(BasicBlock) = .empty,
    next_vreg: VReg = 0,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, param_count: u32, result_count: u32, local_count: u32) IrFunction {
        return .{
            .param_count = param_count,
            .result_count = result_count,
            .local_count = local_count,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *IrFunction) void {
        for (self.blocks.items) |*block| block.deinit();
        self.blocks.deinit(self.allocator);
        if (self.local_types) |lt| self.allocator.free(lt);
    }

    /// Allocate a new virtual register.
    pub fn newVReg(self: *IrFunction) VReg {
        const reg = self.next_vreg;
        self.next_vreg += 1;
        return reg;
    }

    /// Create a new basic block and return its ID.
    pub fn newBlock(self: *IrFunction) !BlockId {
        const id: BlockId = @intCast(self.blocks.items.len);
        try self.blocks.append(self.allocator, BasicBlock.init(id, self.allocator));
        return id;
    }

    /// Get a mutable reference to a block by ID.
    pub fn getBlock(self: *IrFunction, id: BlockId) *BasicBlock {
        return &self.blocks.items[id];
    }
};

/// An IR module — collection of functions.
pub const IrModule = struct {
    functions: std.ArrayList(IrFunction) = .empty,
    allocator: std.mem.Allocator,
    /// Number of imported functions. IR only contains local functions,
    /// but call instructions use module-level indices where
    /// indices < import_count refer to imports.
    import_count: u32 = 0,

    pub fn init(allocator: std.mem.Allocator) IrModule {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *IrModule) void {
        for (self.functions.items) |*func| func.deinit();
        self.functions.deinit(self.allocator);
    }

    pub fn addFunction(self: *IrModule, func: IrFunction) !u32 {
        const idx: u32 = @intCast(self.functions.items.len);
        try self.functions.append(self.allocator, func);
        return idx;
    }
};

// ── Tests ──────────────────────────────────────────────────────────

test "IrFunction: create block and append instructions" {
    const allocator = std.testing.allocator;

    var func = IrFunction.init(allocator, 2, 1, 3);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);

    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = func.newVReg(), .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = func.newVReg(), .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = 0, .rhs = 1 } }, .dest = func.newVReg(), .type = .i32 });
    try block.append(.{ .op = .{ .ret = 2 } });

    try std.testing.expectEqual(@as(usize, 1), func.blocks.items.len);
    try std.testing.expectEqual(@as(usize, 4), block.instructions.items.len);
    try std.testing.expectEqual(@as(u32, 0), block_id);
}

test "IrFunction: newVReg returns sequential values" {
    const allocator = std.testing.allocator;

    var func = IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    try std.testing.expectEqual(@as(VReg, 0), func.newVReg());
    try std.testing.expectEqual(@as(VReg, 1), func.newVReg());
    try std.testing.expectEqual(@as(VReg, 2), func.newVReg());
    try std.testing.expectEqual(@as(VReg, 3), func.newVReg());
    try std.testing.expectEqual(@as(VReg, 4), func.next_vreg);
}

test "IrType: v128 uses 16 bytes and two spill slots" {
    try std.testing.expectEqual(@as(u8, 16), IrType.v128.byteSize());
    try std.testing.expectEqual(@as(u8, 2), IrType.v128.spillSlots64());
    try std.testing.expectEqual(@as(u8, 2), IrType.v128.spillAlignSlots64());
    try std.testing.expectEqual(@as(u8, 1), IrType.i64.spillSlots64());
}

test "Inst: first v128 op family preserves operand shape" {
    const c = Inst{ .op = .{ .v128_const = 0x0011_2233_4455_6677_8899_AABB_CCDD_EEFF }, .dest = 1, .type = .v128 };
    try std.testing.expectEqual(IrType.v128, c.type);
    try std.testing.expectEqual(@as(u128, 0x0011_2233_4455_6677_8899_AABB_CCDD_EEFF), c.op.v128_const);

    const bit = Inst{
        .op = .{ .v128_bitwise = .{ .op = .xor, .lhs = 1, .rhs = 2 } },
        .dest = 3,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.V128BitwiseOp.xor, bit.op.v128_bitwise.op);
    try std.testing.expectEqual(@as(VReg, 1), bit.op.v128_bitwise.lhs);

    const lane = Inst{
        .op = .{ .i32x4_extract_lane = .{ .vector = 3, .lane = 2 } },
        .dest = 4,
        .type = .i32,
    };
    try std.testing.expectEqual(@as(u2, 2), lane.op.i32x4_extract_lane.lane);

    const splat = Inst{
        .op = .{ .i32x4_splat = 5 },
        .dest = 6,
        .type = .v128,
    };
    try std.testing.expectEqual(@as(VReg, 5), splat.op.i32x4_splat);

    const replace = Inst{
        .op = .{ .i32x4_replace_lane = .{ .vector = 6, .val = 7, .lane = 1 } },
        .dest = 8,
        .type = .v128,
    };
    try std.testing.expectEqual(@as(VReg, 6), replace.op.i32x4_replace_lane.vector);
    try std.testing.expectEqual(@as(VReg, 7), replace.op.i32x4_replace_lane.val);
    try std.testing.expectEqual(@as(u2, 1), replace.op.i32x4_replace_lane.lane);

    const shift = Inst{
        .op = .{ .i32x4_shift = .{ .op = .shr_u, .vector = 8, .count = 9 } },
        .dest = 10,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.I32x4ShiftOp.shr_u, shift.op.i32x4_shift.op);
    try std.testing.expectEqual(@as(VReg, 8), shift.op.i32x4_shift.vector);
    try std.testing.expectEqual(@as(VReg, 9), shift.op.i32x4_shift.count);

    const i8_bin = Inst{
        .op = .{ .i8x16_binop = .{ .op = .sub, .lhs = 11, .rhs = 12 } },
        .dest = 13,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.I8x16Op.sub, i8_bin.op.i8x16_binop.op);
    try std.testing.expectEqual(@as(VReg, 11), i8_bin.op.i8x16_binop.lhs);

    const i8_shift = Inst{
        .op = .{ .i8x16_shift = .{ .op = .shr_s, .vector = 13, .count = 14 } },
        .dest = 15,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.I8x16ShiftOp.shr_s, i8_shift.op.i8x16_shift.op);
    try std.testing.expectEqual(@as(VReg, 13), i8_shift.op.i8x16_shift.vector);
    try std.testing.expectEqual(@as(VReg, 14), i8_shift.op.i8x16_shift.count);

    const i8_extract = Inst{
        .op = .{ .i8x16_extract_lane = .{ .vector = 13, .lane = 15, .sign = .unsigned } },
        .dest = 14,
        .type = .i32,
    };
    try std.testing.expectEqual(@as(u4, 15), i8_extract.op.i8x16_extract_lane.lane);
    try std.testing.expectEqual(Inst.I8x16LaneSign.unsigned, i8_extract.op.i8x16_extract_lane.sign);

    const i8_replace = Inst{
        .op = .{ .i8x16_replace_lane = .{ .vector = 13, .val = 14, .lane = 13 } },
        .dest = 15,
        .type = .v128,
    };
    try std.testing.expectEqual(@as(VReg, 13), i8_replace.op.i8x16_replace_lane.vector);
    try std.testing.expectEqual(@as(VReg, 14), i8_replace.op.i8x16_replace_lane.val);
    try std.testing.expectEqual(@as(u4, 13), i8_replace.op.i8x16_replace_lane.lane);

    const i16_bin = Inst{
        .op = .{ .i16x8_binop = .{ .op = .mul, .lhs = 16, .rhs = 17 } },
        .dest = 18,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.I16x8Op.mul, i16_bin.op.i16x8_binop.op);
    try std.testing.expectEqual(@as(VReg, 16), i16_bin.op.i16x8_binop.lhs);

    const i16_shift = Inst{
        .op = .{ .i16x8_shift = .{ .op = .shr_s, .vector = 18, .count = 19 } },
        .dest = 20,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.I16x8ShiftOp.shr_s, i16_shift.op.i16x8_shift.op);
    try std.testing.expectEqual(@as(VReg, 18), i16_shift.op.i16x8_shift.vector);
    try std.testing.expectEqual(@as(VReg, 19), i16_shift.op.i16x8_shift.count);

    const i16_extract = Inst{
        .op = .{ .i16x8_extract_lane = .{ .vector = 18, .lane = 5, .sign = .signed } },
        .dest = 21,
        .type = .i32,
    };
    try std.testing.expectEqual(@as(u3, 5), i16_extract.op.i16x8_extract_lane.lane);
    try std.testing.expectEqual(Inst.I16x8LaneSign.signed, i16_extract.op.i16x8_extract_lane.sign);

    const i16_replace = Inst{
        .op = .{ .i16x8_replace_lane = .{ .vector = 18, .val = 21, .lane = 7 } },
        .dest = 22,
        .type = .v128,
    };
    try std.testing.expectEqual(@as(VReg, 18), i16_replace.op.i16x8_replace_lane.vector);
    try std.testing.expectEqual(@as(VReg, 21), i16_replace.op.i16x8_replace_lane.val);
    try std.testing.expectEqual(@as(u3, 7), i16_replace.op.i16x8_replace_lane.lane);

    const i64_bin = Inst{
        .op = .{ .i64x2_binop = .{ .op = .gt_s, .lhs = 23, .rhs = 24 } },
        .dest = 25,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.I64x2Op.gt_s, i64_bin.op.i64x2_binop.op);
    try std.testing.expectEqual(@as(VReg, 23), i64_bin.op.i64x2_binop.lhs);

    const i64_splat = Inst{
        .op = .{ .i64x2_splat = 26 },
        .dest = 27,
        .type = .v128,
    };
    try std.testing.expectEqual(@as(VReg, 26), i64_splat.op.i64x2_splat);

    const i64_extract = Inst{
        .op = .{ .i64x2_extract_lane = .{ .vector = 27, .lane = 1 } },
        .dest = 28,
        .type = .i64,
    };
    try std.testing.expectEqual(@as(u1, 1), i64_extract.op.i64x2_extract_lane.lane);

    const i64_replace = Inst{
        .op = .{ .i64x2_replace_lane = .{ .vector = 27, .val = 28, .lane = 1 } },
        .dest = 29,
        .type = .v128,
    };
    try std.testing.expectEqual(@as(VReg, 27), i64_replace.op.i64x2_replace_lane.vector);
    try std.testing.expectEqual(@as(VReg, 28), i64_replace.op.i64x2_replace_lane.val);
    try std.testing.expectEqual(@as(u1, 1), i64_replace.op.i64x2_replace_lane.lane);
}

test "IrModule: add multiple functions" {
    const allocator = std.testing.allocator;

    var module = IrModule.init(allocator);
    defer module.deinit();

    var f1 = IrFunction.init(allocator, 0, 1, 0);
    const b1 = try f1.newBlock();
    try f1.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = f1.newVReg(), .type = .i32 });

    var f2 = IrFunction.init(allocator, 2, 1, 2);
    const b2 = try f2.newBlock();
    try f2.getBlock(b2).append(.{ .op = .{ .iconst_64 = 99 }, .dest = f2.newVReg(), .type = .i64 });

    const idx1 = try module.addFunction(f1);
    const idx2 = try module.addFunction(f2);

    try std.testing.expectEqual(@as(u32, 0), idx1);
    try std.testing.expectEqual(@as(u32, 1), idx2);
    try std.testing.expectEqual(@as(usize, 2), module.functions.items.len);
}

test "IrFunction: deinit frees all blocks" {
    const allocator = std.testing.allocator;

    var func = IrFunction.init(allocator, 0, 0, 0);
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = func.newVReg() });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 2 }, .dest = func.newVReg() });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    try func.getBlock(b1).addPredecessor(b0);

    // deinit should free everything without leaks (testing.allocator checks)
    func.deinit();
}

test "IrModule: deinit frees all functions" {
    const allocator = std.testing.allocator;

    var module = IrModule.init(allocator);

    var f1 = IrFunction.init(allocator, 1, 1, 1);
    _ = try f1.newBlock();
    try f1.getBlock(0).append(.{ .op = .{ .iconst_32 = 42 }, .dest = f1.newVReg() });

    var f2 = IrFunction.init(allocator, 0, 0, 0);
    _ = try f2.newBlock();
    try f2.getBlock(0).append(.{ .op = .{ .@"unreachable" = {} } });

    _ = try module.addFunction(f1);
    _ = try module.addFunction(f2);

    // deinit should free everything without leaks
    module.deinit();
}

test "BasicBlock: predecessors tracking" {
    const allocator = std.testing.allocator;

    var func = IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    try func.getBlock(b2).addPredecessor(b0);
    try func.getBlock(b2).addPredecessor(b1);

    try std.testing.expectEqual(@as(usize, 2), func.getBlock(b2).predecessors.items.len);
    try std.testing.expectEqual(b0, func.getBlock(b2).predecessors.items[0]);
    try std.testing.expectEqual(b1, func.getBlock(b2).predecessors.items[1]);
}
