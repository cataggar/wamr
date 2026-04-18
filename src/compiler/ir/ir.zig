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
    void,
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
        load: struct { base: VReg, offset: u32, size: u8, sign_extend: bool = false },
        store: struct { base: VReg, offset: u32, size: u8, val: VReg },

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
        call: struct { func_idx: u32, args: []const VReg = &.{}, extra_results: u8 = 0 },
        call_indirect: struct { type_idx: u32, elem_idx: VReg, args: []const VReg = &.{}, extra_results: u8 = 0 },
        call_ref: struct { type_idx: u32, func_ref: VReg, args: []const VReg = &.{}, extra_results: u8 = 0 },

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
        table_size: void,
        table_get: VReg, // idx
        table_set: struct { idx: VReg, val: VReg },
        table_grow: struct { init: VReg, delta: VReg }, // -> i32 (prev size or -1)
        ref_func: u32, // funcidx -> native pointer loaded from vmctx.func_table[idx]
    };

    pub const BinOp = struct {
        lhs: VReg,
        rhs: VReg,
    };

    pub const AtomicRmwOp = enum { add, sub, @"and", @"or", xor, xchg };
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
