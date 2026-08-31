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

    /// Free heap slices owned directly by this instruction's `op`. Mirrors
    /// the per-op `free` arms in `BasicBlock.deinit` so callers that drop an
    /// instruction out of a block's list (e.g. the inliner discarding dead
    /// code after a terminator, #784) can release the same memory `deinit`
    /// would have. NOTE: `br_table` targets are intentionally NOT freed here
    /// — they live in `IrFunction.owned_br_table_targets` and are freed at
    /// function deinit; freeing them here would double-free.
    pub fn freeOwnedSlices(self: Inst, allocator: std.mem.Allocator) void {
        switch (self.op) {
            .phi => |edges| allocator.free(edges),
            .parallel_copy => |pairs| allocator.free(pairs),
            .call => |cl| if (cl.args.len > 0) allocator.free(cl.args),
            .call_indirect => |ci| if (ci.args.len > 0) allocator.free(ci.args),
            .call_ref => |cr| if (cr.args.len > 0) allocator.free(cr.args),
            .ret_multi => |vregs| if (vregs.len > 0) allocator.free(vregs),
            .try_table_begin => |tt| if (tt.clauses.len > 0) allocator.free(tt.clauses),
            .throw => |th| if (th.args.len > 0) allocator.free(th.args),
            else => {},
        }
    }

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
        v128_load_splat: V128LoadSplat,
        v128_load_zero: V128LoadZero,
        v128_load_extend: V128LoadExtend,
        v128_load_lane: V128LoadLane,
        v128_store: V128Store,
        v128_store_lane: V128StoreLane,
        v128_not: VReg,
        v128_bitwise: V128Bitwise,
        v128_bitselect: V128Bitselect,
        v128_any_true: VReg,
        simd_all_true: SimdAllTrue,
        simd_bitmask: SimdBitmask,
        i32x4_binop: I32x4BinOp,
        i32x4_unop: SimdUnary,
        i32x4_extadd_pairwise_i16x8: SimdExtAddPairwise,
        i32x4_dot_i16x8_s: BinOp,
        i32x4_extend_i16x8: SimdExtendHalf,
        i32x4_extmul_i16x8: SimdExtMul,
        f32x4_unop: F32x4UnOp,
        f32x4_binop: F32x4BinOp,
        f32x4_convert_i32x4: SimdIntToFloatConvert,
        i32x4_trunc_sat: SimdFloatToIntTruncSat,
        f32x4_demote_f64x2_zero: SimdFloatPrecisionConvert,
        f32x4_splat: VReg,
        f32x4_extract_lane: F32x4ExtractLane,
        f32x4_replace_lane: F32x4ReplaceLane,
        i32x4_shift: I32x4Shift,
        i32x4_splat: VReg,
        i32x4_extract_lane: I32x4ExtractLane,
        i32x4_replace_lane: I32x4ReplaceLane,
        i8x16_binop: I8x16BinOp,
        i8x16_shuffle: I8x16Shuffle,
        i8x16_swizzle: I8x16Swizzle,
        i8x16_unop: SimdUnary,
        i8x16_shift: I8x16Shift,
        i8x16_splat: VReg,
        i8x16_extract_lane: I8x16ExtractLane,
        i8x16_replace_lane: I8x16ReplaceLane,
        i8x16_narrow_i16x8: SimdNarrow,
        i16x8_binop: I16x8BinOp,
        i16x8_unop: SimdUnary,
        i16x8_extadd_pairwise_i8x16: SimdExtAddPairwise,
        i16x8_extend_i8x16: SimdExtendHalf,
        i16x8_extmul_i8x16: SimdExtMul,
        i16x8_narrow_i32x4: SimdNarrow,
        i16x8_shift: I16x8Shift,
        i16x8_splat: VReg,
        i16x8_extract_lane: I16x8ExtractLane,
        i16x8_replace_lane: I16x8ReplaceLane,
        i64x2_binop: I64x2BinOp,
        i64x2_unop: SimdUnary,
        i64x2_extend_i32x4: SimdExtendHalf,
        i64x2_extmul_i32x4: SimdExtMul,
        i64x2_shift: I64x2Shift,
        i64x2_splat: VReg,
        i64x2_extract_lane: I64x2ExtractLane,
        i64x2_replace_lane: I64x2ReplaceLane,
        f64x2_unop: F64x2UnOp,
        f64x2_binop: F64x2BinOp,
        f64x2_splat: VReg,
        f64x2_extract_lane: F64x2ExtractLane,
        f64x2_replace_lane: F64x2ReplaceLane,
        f64x2_convert_low_i32x4: SimdIntToFloatConvert,
        f64x2_promote_low_f32x4: SimdFloatPrecisionConvert,

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

        // Fused address computation: dest = base + index*scale + disp.
        // Produced by the x86-64 `foldCompoundLea` peephole (#543) so the
        // register allocator sees base/index as coincident operands of a
        // single instruction (correct liveness) and codegen emits one LEA.
        lea: Lea,

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
        // Multi-value return: first VReg -> the platform primary result
        // register, remaining values are written to memory via the hidden
        // return pointer passed by the caller.
        ret_multi: []const VReg,
        @"unreachable": void,

        // Function calls. `extra_results` is the number of additional
        // results beyond the platform primary result register
        // (i.e. callee.result_count - 1). When > 0, the caller passes a hidden
        // return pointer as an implicit trailing argument; the callee writes
        // extras into the target ABI's return area layout, and the caller
        // retrieves them via `.call_result` ops emitted right after the call.
        // The primary result is delivered in `inst.dest`.
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

        // Parallel register-to-register copy. Resolves phi-incoming values
        // at a predecessor's terminator without going through a frame
        // round-trip. All `src` reads happen "in parallel" before any
        // `dst` is written — the aarch64 emitter applies a 4-phase
        // parallel-copy algorithm to honor that semantics even when the
        // assigned physical registers form cycles.
        //
        // Owned: the backing slice is freed by `BasicBlock.deinit` and
        // tracked alongside `phi` in `owned_phi_edges` / similar.
        parallel_copy: []const ParallelCopy,

        // ── Exception handling (#672) ──────────────────────────────────
        // Modern Wasm 3.0 EH only: try_table / throw / throw_ref.
        // Legacy try/catch/rethrow/delegate/catch_all are NOT modelled.

        /// Open a try region (matches the Wasm `try_table` opcode). The
        /// op is emitted at the start of the try body and carries the
        /// list of catch clauses with the IR `BlockId` of each handler.
        ///
        /// Codegen attaches an unwind table entry [try_start_pc,
        /// try_end_pc, clauses_ptr] for the function; throws inside the
        /// region consult this table to dispatch.
        ///
        /// Commit 2 only: lowering decodes try_table into this op +
        /// normal block scaffolding; codegen still errors out.
        try_table_begin: TryTableBegin,

        /// Close the matching try region. Emitted at the `end` opcode
        /// terminating a try_table body. Pairs 1:1 with `try_table_begin`
        /// in well-formed wasm; codegen uses both to derive the
        /// try_start_pc / try_end_pc range stored in the unwind table.
        try_table_end: void,

        /// Throw an exception (Wasm `throw` opcode). Terminator.
        /// `tag_idx` indexes the *resolved* tag instance in the module's
        /// tag space (imports first, then declared). `args` are popped
        /// from the wasm operand stack at lowering time (count =
        /// tag's func_type.params.len).
        ///
        /// At runtime, control transfers either to a matching catch in
        /// an enclosing try region of the *same* function (commits 3-4)
        /// or, when commit 5 lands, an enclosing frame's try region.
        ///
        /// Owned: `args` is freed by `BasicBlock.deinit`.
        throw: Throw,

        /// Rethrow a caught exnref (Wasm `throw_ref` opcode). Terminator.
        /// `exnref` carries the exnref pushed by an earlier catch_ref /
        /// catch_all_ref clause.
        throw_ref: VReg,
    };

    /// One arm of a `try_table` instruction. `kind` maps directly to the
    /// 4 wasm catch-clause discriminants:
    ///   0 = catch tag        — handler receives the tag's params
    ///   1 = catch_ref tag    — handler receives params + an exnref
    ///   2 = catch_all        — handler receives no params
    ///   3 = catch_all_ref    — handler receives a single exnref
    pub const CatchClause = struct {
        kind: u8,
        tag_idx: ?u32 = null,
        handler: BlockId,
    };

    pub const TryTableBegin = struct {
        /// Number of results the try body yields on normal exit. Carried
        /// through so codegen can size the spill area for the handler
        /// resolution stub matching commit 5's cross-frame unwind.
        result_arity: u32,
        /// Owned slice freed by `BasicBlock.deinit` via
        /// `owned_catch_clauses`.
        clauses: []const CatchClause,
    };

    pub const Throw = struct {
        tag_idx: u32,
        /// Owned slice freed by `BasicBlock.deinit` via
        /// `owned_throw_args`.
        args: []const VReg,
    };

    pub const BinOp = struct {
        lhs: VReg,
        rhs: VReg,
    };

    /// Operands of a fused `lea` (x86-64 address-generation) instruction.
    /// Computes `base + index * scale + disp`. `scale` must be 1, 2, 4, or
    /// 8 (the legal x86-64 SIB scales); `disp` is a signed 32-bit
    /// displacement. The result wraps to `type`'s width (i32/i64), matching
    /// a chain of wasm `add`/`shl` with the same operands.
    pub const Lea = struct {
        base: VReg,
        index: VReg,
        scale: u8,
        disp: i32,
    };

    pub const AtomicRmwOp = enum { add, sub, @"and", @"or", xor, xchg };

    pub const V128BitwiseOp = enum { @"and", andnot, @"or", xor };
    pub const SimdUnaryOp = enum { abs, neg, popcnt };
    pub const SimdExtAddPairwiseSign = enum { signed, unsigned };
    pub const SimdExtendSign = enum { signed, unsigned };
    pub const SimdExtendHalfSelect = enum { low, high };
    pub const SimdIntToFloatSign = enum { signed, unsigned };
    pub const SimdFloatToIntSign = enum { signed, unsigned };
    pub const SimdFloatToIntSrcWidth = enum { f32x4, f64x2 };
    pub const SimdFloatPrecisionConvert = struct {
        vector: VReg,
    };

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
        min_s,
        min_u,
        max_s,
        max_u,
    };

    pub const I8x16Op = enum {
        add,
        sub,
        add_sat_s,
        add_sat_u,
        sub_sat_s,
        sub_sat_u,
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
        min_s,
        min_u,
        max_s,
        max_u,
        avgr_u,
    };

    pub const I16x8Op = enum {
        add,
        sub,
        add_sat_s,
        add_sat_u,
        sub_sat_s,
        sub_sat_u,
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
        q15mulr_sat_s,
        min_s,
        min_u,
        max_s,
        max_u,
        avgr_u,
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

    pub const I64x2ShiftOp = enum {
        shl,
        shr_s,
        shr_u,
    };

    pub const F64x2Op = enum {
        add,
        sub,
        mul,
        div,
        min,
        max,
        pmin,
        pmax,
        eq,
        ne,
        lt,
        gt,
        le,
        ge,
    };

    pub const F32x4Op = enum {
        add,
        sub,
        mul,
        div,
        min,
        max,
        pmin,
        pmax,
        eq,
        ne,
        lt,
        gt,
        le,
        ge,
    };

    pub const F32x4UnaryOp = enum {
        abs,
        neg,
        sqrt,
        ceil,
        floor,
        trunc,
        nearest,
    };

    pub const F64x2UnaryOp = enum {
        abs,
        neg,
        sqrt,
        ceil,
        floor,
        trunc,
        nearest,
    };

    pub const V128Mem = struct {
        base: VReg,
        offset: u32,
        alignment: u32,
        bounds_known: bool = false,
        checked_end: u64 = 0,
    };

    pub const V128LoadSplatWidth = enum {
        i8x16,
        i16x8,
        i32x4,
        i64x2,

        pub fn accessSize(self: V128LoadSplatWidth) u8 {
            return switch (self) {
                .i8x16 => 1,
                .i16x8 => 2,
                .i32x4 => 4,
                .i64x2 => 8,
            };
        }
    };

    pub const V128LoadSplat = struct {
        width: V128LoadSplatWidth,
        base: VReg,
        offset: u32,
        alignment: u32,
        bounds_known: bool = false,
        checked_end: u64 = 0,

        pub fn accessSize(self: V128LoadSplat) u8 {
            return self.width.accessSize();
        }
    };

    pub const V128LoadZeroWidth = enum {
        i32,
        i64,

        pub fn accessSize(self: V128LoadZeroWidth) u8 {
            return switch (self) {
                .i32 => 4,
                .i64 => 8,
            };
        }
    };

    pub const V128LoadZero = struct {
        width: V128LoadZeroWidth,
        base: VReg,
        offset: u32,
        alignment: u32,
        bounds_known: bool = false,
        checked_end: u64 = 0,

        pub fn accessSize(self: V128LoadZero) u8 {
            return self.width.accessSize();
        }
    };

    pub const V128LoadExtendWidth = enum {
        i8x8,
        i16x4,
        i32x2,

        pub fn accessSize(_: V128LoadExtendWidth) u8 {
            return 8;
        }
    };

    pub const V128LoadExtend = struct {
        src_width: V128LoadExtendWidth,
        sign: SimdExtendSign,
        base: VReg,
        offset: u32,
        alignment: u32,
        bounds_known: bool = false,
        checked_end: u64 = 0,

        pub fn accessSize(self: V128LoadExtend) u8 {
            return self.src_width.accessSize();
        }
    };

    pub const V128LaneWidth = enum {
        i8,
        i16,
        i32,
        i64,

        pub fn accessSize(self: V128LaneWidth) u8 {
            return switch (self) {
                .i8 => 1,
                .i16 => 2,
                .i32 => 4,
                .i64 => 8,
            };
        }

        pub fn laneCount(self: V128LaneWidth) u8 {
            return switch (self) {
                .i8 => 16,
                .i16 => 8,
                .i32 => 4,
                .i64 => 2,
            };
        }
    };

    pub const V128LoadLane = struct {
        width: V128LaneWidth,
        base: VReg,
        offset: u32,
        alignment: u32,
        vector: VReg,
        lane: u8,
        bounds_known: bool = false,
        checked_end: u64 = 0,

        pub fn accessSize(self: V128LoadLane) u8 {
            return self.width.accessSize();
        }
    };

    pub const V128Store = struct {
        base: VReg,
        offset: u32,
        alignment: u32,
        val: VReg,
        bounds_known: bool = false,
        checked_end: u64 = 0,
    };

    pub const V128StoreLane = struct {
        width: V128LaneWidth,
        base: VReg,
        offset: u32,
        alignment: u32,
        vector: VReg,
        lane: u8,
        bounds_known: bool = false,
        checked_end: u64 = 0,

        pub fn accessSize(self: V128StoreLane) u8 {
            return self.width.accessSize();
        }
    };

    pub const V128Bitwise = struct {
        op: V128BitwiseOp,
        lhs: VReg,
        rhs: VReg,
    };

    pub const V128Bitselect = struct {
        a: VReg,
        b: VReg,
        mask: VReg,
    };

    pub const SimdAllTrueWidth = enum { i8x16, i16x8, i32x4, i64x2 };

    pub const SimdAllTrue = struct {
        width: SimdAllTrueWidth,
        vector: VReg,
    };

    pub const SimdBitmaskWidth = enum { i8x16, i16x8, i32x4, i64x2 };

    pub const SimdBitmask = struct {
        width: SimdBitmaskWidth,
        vector: VReg,
    };

    pub const I8x16Shuffle = struct {
        lhs: VReg,
        rhs: VReg,
        lanes: [16]u8,
    };

    pub const I8x16Swizzle = struct {
        vector: VReg,
        indices: VReg,
    };

    pub const SimdUnary = struct {
        op: SimdUnaryOp,
        vector: VReg,
    };

    pub const SimdExtAddPairwise = struct {
        sign: SimdExtAddPairwiseSign,
        vector: VReg,
    };

    pub const SimdExtendHalf = struct {
        sign: SimdExtendSign,
        half: SimdExtendHalfSelect,
        vector: VReg,
    };

    pub const SimdExtMul = struct {
        sign: SimdExtendSign,
        half: SimdExtendHalfSelect,
        lhs: VReg,
        rhs: VReg,
    };

    pub const SimdIntToFloatConvert = struct {
        sign: SimdIntToFloatSign,
        vector: VReg,
    };

    pub const SimdFloatToIntTruncSat = struct {
        src_width: SimdFloatToIntSrcWidth,
        sign: SimdFloatToIntSign,
        vector: VReg,
    };

    pub const SimdNarrowSign = enum { signed, unsigned };

    pub const SimdNarrow = struct {
        sign: SimdNarrowSign,
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

    pub const F32x4BinOp = struct {
        op: F32x4Op,
        lhs: VReg,
        rhs: VReg,
    };

    pub const F32x4UnOp = struct {
        op: F32x4UnaryOp,
        vector: VReg,
    };

    pub const F64x2UnOp = struct {
        op: F64x2UnaryOp,
        vector: VReg,
    };

    pub const F64x2BinOp = struct {
        op: F64x2Op,
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

    pub const I64x2Shift = struct {
        op: I64x2ShiftOp,
        vector: VReg,
        count: VReg,
    };

    pub const I32x4ExtractLane = struct {
        vector: VReg,
        lane: u2,
    };

    pub const F32x4ExtractLane = struct {
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

    pub const F64x2ExtractLane = struct {
        vector: VReg,
        lane: u1,
    };

    pub const I32x4ReplaceLane = struct {
        vector: VReg,
        val: VReg,
        lane: u2,
    };

    pub const F32x4ReplaceLane = struct {
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

    pub const F64x2ReplaceLane = struct {
        vector: VReg,
        val: VReg,
        lane: u1,
    };

    pub const PhiEdge = struct {
        block: BlockId,
        val: VReg,
    };

    /// One (dst ← src) pair inside a `parallel_copy` instruction. The
    /// per-pair `ty` lets the codegen pick the right MOV width without
    /// looking up the destination vreg's type from a side table.
    pub const ParallelCopy = struct {
        dst: VReg,
        src: VReg,
        ty: IrType,
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
            inst.freeOwnedSlices(self.allocator);
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
    /// First synthetic-local index produced by `lowerPhisToLocals`, and
    /// the EXCLUSIVE end of the synthetic-local range at the time of that
    /// pass. Used by `coalescePhiLocalsToParallelCopy` (#540) to identify
    /// the bridge slots without conflating them with locals later
    /// inserted by passes like `inductionVariableSimplification` (which
    /// also bumps `local_count`). `null` means no phi lowering has run,
    /// or it ran but produced no synthetic locals.
    phi_synth_local_start: ?u32 = null,
    phi_synth_local_end: ?u32 = null,
    /// Per-local IR type (params first, then declared locals, then synthetic).
    /// Populated by the frontend; used by mem2reg for typed-zero seeding.
    local_types: ?[]const IrType = null,
    owned_br_table_targets: std.ArrayList([]const BlockId) = .empty,
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
        for (self.owned_br_table_targets.items) |targets| self.allocator.free(targets);
        self.owned_br_table_targets.deinit(self.allocator);
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

    /// Deep-copy this function into `allocator`.
    ///
    /// `IrFunction` owns several instruction payload slices with two
    /// different lifetimes: most are block-owned and freed by
    /// `BasicBlock.deinit`, while `br_table.targets` slices are tracked in
    /// `owned_br_table_targets` and freed by `IrFunction.deinit`. Preserve
    /// that split so clones can be mutated and destroyed independently.
    pub fn clone(self: *const IrFunction, allocator: std.mem.Allocator) !IrFunction {
        var out = IrFunction.init(allocator, self.param_count, self.result_count, self.local_count);
        errdefer out.deinit();

        out.name = self.name;
        out.phi_synth_local_start = self.phi_synth_local_start;
        out.phi_synth_local_end = self.phi_synth_local_end;
        out.next_vreg = self.next_vreg;
        if (self.local_types) |lt| {
            out.local_types = try allocator.dupe(IrType, lt);
        }

        try out.blocks.ensureTotalCapacity(allocator, self.blocks.items.len);
        for (self.blocks.items) |*block| {
            var copied = BasicBlock.init(block.id, allocator);
            errdefer copied.deinit();

            try copied.predecessors.appendSlice(allocator, block.predecessors.items);
            try copied.instructions.ensureTotalCapacity(allocator, block.instructions.items.len);
            for (block.instructions.items) |inst| {
                try copied.instructions.append(allocator, try cloneInst(inst, allocator, &out));
            }

            out.blocks.appendAssumeCapacity(copied);
        }

        return out;
    }
};

fn cloneInst(inst: Inst, allocator: std.mem.Allocator, owner: *IrFunction) !Inst {
    var out = inst;
    out.op = try cloneOp(inst.op, allocator, owner);
    return out;
}

fn cloneVRegSlice(allocator: std.mem.Allocator, values: []const VReg) ![]const VReg {
    if (values.len == 0) return &.{};
    return try allocator.dupe(VReg, values);
}

fn cloneOp(op: Inst.Op, allocator: std.mem.Allocator, owner: *IrFunction) !Inst.Op {
    return switch (op) {
        .br_table => |bt| blk: {
            const targets = try allocator.dupe(BlockId, bt.targets);
            errdefer allocator.free(targets);
            try owner.owned_br_table_targets.append(allocator, targets);
            break :blk .{ .br_table = .{
                .index = bt.index,
                .targets = targets,
                .default = bt.default,
            } };
        },
        .ret_multi => |vregs| .{ .ret_multi = try cloneVRegSlice(allocator, vregs) },
        .call => |cl| .{ .call = .{
            .func_idx = cl.func_idx,
            .args = try cloneVRegSlice(allocator, cl.args),
            .extra_results = cl.extra_results,
            .tail = cl.tail,
        } },
        .call_indirect => |ci| .{ .call_indirect = .{
            .type_idx = ci.type_idx,
            .elem_idx = ci.elem_idx,
            .args = try cloneVRegSlice(allocator, ci.args),
            .extra_results = ci.extra_results,
            .table_idx = ci.table_idx,
            .tail = ci.tail,
        } },
        .call_ref => |cr| .{ .call_ref = .{
            .type_idx = cr.type_idx,
            .func_ref = cr.func_ref,
            .args = try cloneVRegSlice(allocator, cr.args),
            .extra_results = cr.extra_results,
            .tail = cr.tail,
        } },
        .phi => |edges| .{ .phi = try allocator.dupe(Inst.PhiEdge, edges) },
        .parallel_copy => |pairs| .{ .parallel_copy = try allocator.dupe(Inst.ParallelCopy, pairs) },
        .try_table_begin => |tt| .{ .try_table_begin = .{
            .result_arity = tt.result_arity,
            .clauses = if (tt.clauses.len == 0) &.{} else try allocator.dupe(Inst.CatchClause, tt.clauses),
        } },
        .throw => |th| .{ .throw = .{
            .tag_idx = th.tag_idx,
            .args = try cloneVRegSlice(allocator, th.args),
        } },
        else => op,
    };
}

/// Function type metadata copied from the Wasm type section. Slices are owned
/// by the containing `IrModule`.
pub const IrFuncType = struct {
    params: []const IrType,
    results: []const IrType,
};

/// An IR module — collection of functions.
pub const IrModule = struct {
    functions: std.ArrayList(IrFunction) = .empty,
    allocator: std.mem.Allocator,
    /// Number of imported functions. IR only contains local functions,
    /// but call instructions use module-level indices where
    /// indices < import_count refer to imports.
    import_count: u32 = 0,
    /// Conservative module-level memory mode used by target codegen.
    /// Any memory64 or shared memory disables transforms that are only
    /// proven for the single pinned memory32 base.
    has_memory64: bool = false,
    has_shared_memory: bool = false,
    /// Wasm-flat global types (imported globals first, then local globals).
    /// Populated by the frontend so codegen can use the same byte offsets as
    /// the AOT runtime when globals include 16-byte v128 slots.
    global_types: ?[]const IrType = null,
    /// Byte offset for each wasm-flat global in the AOT globals storage.
    global_offsets: ?[]const u32 = null,
    /// Total byte size of the AOT globals storage.
    global_storage_size: u32 = 0,
    /// Wasm function type table. Codegen uses this to marshal direct and
    /// indirect calls whose parameter classes differ from scalar GPRs.
    func_types: std.ArrayList(IrFuncType) = .empty,
    /// Module function-index → type-index mapping, imports first, then locals.
    func_type_indices: std.ArrayList(u32) = .empty,

    pub fn init(allocator: std.mem.Allocator) IrModule {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *IrModule) void {
        for (self.functions.items) |*func| func.deinit();
        self.functions.deinit(self.allocator);
        for (self.func_types.items) |ft| {
            self.allocator.free(ft.params);
            self.allocator.free(ft.results);
        }
        self.func_types.deinit(self.allocator);
        self.func_type_indices.deinit(self.allocator);
        if (self.global_types) |gt| self.allocator.free(gt);
        if (self.global_offsets) |go| self.allocator.free(go);
    }

    pub fn addFuncType(self: *IrModule, params: []const IrType, results: []const IrType) !u32 {
        const owned_params = try self.allocator.dupe(IrType, params);
        errdefer self.allocator.free(owned_params);
        const owned_results = try self.allocator.dupe(IrType, results);
        errdefer self.allocator.free(owned_results);

        const idx: u32 = @intCast(self.func_types.items.len);
        try self.func_types.append(self.allocator, .{
            .params = owned_params,
            .results = owned_results,
        });
        return idx;
    }

    pub fn addFuncTypeIndex(self: *IrModule, type_idx: u32) !void {
        try self.func_type_indices.append(self.allocator, type_idx);
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

test "IrFunction: clone deep-copies owned slices" {
    const allocator = std.testing.allocator;

    var func = IrFunction.init(allocator, 2, 1, 3);
    defer func.deinit();
    func.name = "clone-source";
    func.local_types = try allocator.dupe(IrType, &.{ .i32, .i64, .i32, .i32, .i64 });
    func.phi_synth_local_start = 3;
    func.phi_synth_local_end = 5;
    _ = func.newVReg();
    _ = func.newVReg();
    _ = func.newVReg();

    const entry = try func.newBlock();
    const join = try func.newBlock();
    try func.getBlock(join).addPredecessor(entry);

    const targets = try allocator.dupe(BlockId, &.{join});
    try func.owned_br_table_targets.append(allocator, targets);
    const call_args = try allocator.dupe(VReg, &.{ 0, 1 });
    try func.getBlock(entry).append(.{
        .op = .{ .call = .{ .func_idx = 7, .args = call_args, .extra_results = 1, .tail = true } },
        .dest = 2,
        .type = .i32,
    });
    try func.getBlock(entry).append(.{
        .op = .{ .br_table = .{ .index = 0, .targets = targets, .default = join } },
        .type = .void,
    });

    const phi_edges = try allocator.dupe(Inst.PhiEdge, &.{.{ .block = entry, .val = 2 }});
    try func.getBlock(join).append(.{ .op = .{ .phi = phi_edges }, .dest = 3, .type = .i32 });
    const copies = try allocator.dupe(Inst.ParallelCopy, &.{.{ .dst = 4, .src = 3, .ty = .i32 }});
    try func.getBlock(join).append(.{ .op = .{ .parallel_copy = copies }, .type = .void });
    const ret_vals = try allocator.dupe(VReg, &.{ 3, 4 });
    try func.getBlock(join).append(.{ .op = .{ .ret_multi = ret_vals }, .type = .void });
    const clauses = try allocator.dupe(Inst.CatchClause, &.{.{ .kind = 0, .tag_idx = 9, .handler = join }});
    try func.getBlock(join).append(.{ .op = .{ .try_table_begin = .{ .result_arity = 0, .clauses = clauses } }, .type = .void });
    const throw_args = try allocator.dupe(VReg, &.{3});
    try func.getBlock(join).append(.{ .op = .{ .throw = .{ .tag_idx = 9, .args = throw_args } }, .type = .void });

    var copy = try func.clone(allocator);
    defer copy.deinit();

    try std.testing.expectEqualStrings(func.name.?, copy.name.?);
    try std.testing.expectEqual(func.param_count, copy.param_count);
    try std.testing.expectEqual(func.result_count, copy.result_count);
    try std.testing.expectEqual(func.local_count, copy.local_count);
    try std.testing.expectEqual(func.next_vreg, copy.next_vreg);
    try std.testing.expectEqual(func.phi_synth_local_start, copy.phi_synth_local_start);
    try std.testing.expectEqual(func.phi_synth_local_end, copy.phi_synth_local_end);
    try std.testing.expectEqualSlices(IrType, func.local_types.?, copy.local_types.?);
    try std.testing.expect(@intFromPtr(func.local_types.?.ptr) != @intFromPtr(copy.local_types.?.ptr));

    try std.testing.expectEqual(func.blocks.items.len, copy.blocks.items.len);
    try std.testing.expectEqualSlices(BlockId, func.getBlock(join).predecessors.items, copy.getBlock(join).predecessors.items);

    const src_call = func.getBlock(entry).instructions.items[0].op.call;
    const dst_call = copy.getBlock(entry).instructions.items[0].op.call;
    try std.testing.expectEqualSlices(VReg, src_call.args, dst_call.args);
    try std.testing.expect(@intFromPtr(src_call.args.ptr) != @intFromPtr(dst_call.args.ptr));

    const src_bt = func.getBlock(entry).instructions.items[1].op.br_table;
    const dst_bt = copy.getBlock(entry).instructions.items[1].op.br_table;
    try std.testing.expectEqualSlices(BlockId, src_bt.targets, dst_bt.targets);
    try std.testing.expect(@intFromPtr(src_bt.targets.ptr) != @intFromPtr(dst_bt.targets.ptr));
    try std.testing.expectEqual(@as(usize, 1), copy.owned_br_table_targets.items.len);
    try std.testing.expectEqual(@intFromPtr(dst_bt.targets.ptr), @intFromPtr(copy.owned_br_table_targets.items[0].ptr));

    const src_phi = func.getBlock(join).instructions.items[0].op.phi;
    const dst_phi = copy.getBlock(join).instructions.items[0].op.phi;
    try std.testing.expectEqualSlices(Inst.PhiEdge, src_phi, dst_phi);
    try std.testing.expect(@intFromPtr(src_phi.ptr) != @intFromPtr(dst_phi.ptr));

    const src_pc = func.getBlock(join).instructions.items[1].op.parallel_copy;
    const dst_pc = copy.getBlock(join).instructions.items[1].op.parallel_copy;
    try std.testing.expectEqualSlices(Inst.ParallelCopy, src_pc, dst_pc);
    try std.testing.expect(@intFromPtr(src_pc.ptr) != @intFromPtr(dst_pc.ptr));

    const src_ret = func.getBlock(join).instructions.items[2].op.ret_multi;
    const dst_ret = copy.getBlock(join).instructions.items[2].op.ret_multi;
    try std.testing.expectEqualSlices(VReg, src_ret, dst_ret);
    try std.testing.expect(@intFromPtr(src_ret.ptr) != @intFromPtr(dst_ret.ptr));

    const src_tt = func.getBlock(join).instructions.items[3].op.try_table_begin;
    const dst_tt = copy.getBlock(join).instructions.items[3].op.try_table_begin;
    try std.testing.expectEqualSlices(Inst.CatchClause, src_tt.clauses, dst_tt.clauses);
    try std.testing.expect(@intFromPtr(src_tt.clauses.ptr) != @intFromPtr(dst_tt.clauses.ptr));

    const src_throw = func.getBlock(join).instructions.items[4].op.throw;
    const dst_throw = copy.getBlock(join).instructions.items[4].op.throw;
    try std.testing.expectEqualSlices(VReg, src_throw.args, dst_throw.args);
    try std.testing.expect(@intFromPtr(src_throw.args.ptr) != @intFromPtr(dst_throw.args.ptr));
}

test "IrType: v128 uses 16 bytes and two spill slots" {
    try std.testing.expectEqual(@as(u8, 16), IrType.v128.byteSize());
    try std.testing.expectEqual(@as(u8, 2), IrType.v128.spillSlots64());
    try std.testing.expectEqual(@as(u8, 2), IrType.v128.spillAlignSlots64());
    try std.testing.expectEqual(@as(u8, 1), IrType.i64.spillSlots64());
}

test "Inst: first v128 op family preserves operand shape" {
    const bitselect = Inst{
        .op = .{ .v128_bitselect = .{ .a = 1, .b = 2, .mask = 3 } },
        .dest = 4,
        .type = .v128,
    };
    try std.testing.expectEqual(.v128_bitselect, std.meta.activeTag(bitselect.op));
    try std.testing.expectEqual(@as(u32, 1), bitselect.op.v128_bitselect.a);
    try std.testing.expectEqual(@as(u32, 2), bitselect.op.v128_bitselect.b);
    try std.testing.expectEqual(@as(u32, 3), bitselect.op.v128_bitselect.mask);

    const any_true = Inst{
        .op = .{ .v128_any_true = 5 },
        .dest = 6,
        .type = .i32,
    };
    try std.testing.expectEqual(@as(VReg, 5), any_true.op.v128_any_true);

    const all_true = Inst{
        .op = .{ .simd_all_true = .{ .width = .i16x8, .vector = 5 } },
        .dest = 7,
        .type = .i32,
    };
    try std.testing.expectEqual(Inst.SimdAllTrueWidth.i16x8, all_true.op.simd_all_true.width);
    try std.testing.expectEqual(@as(VReg, 5), all_true.op.simd_all_true.vector);

    const bitmask = Inst{
        .op = .{ .simd_bitmask = .{ .width = .i16x8, .vector = 7 } },
        .dest = 8,
        .type = .i32,
    };
    try std.testing.expectEqual(Inst.SimdBitmaskWidth.i16x8, bitmask.op.simd_bitmask.width);
    try std.testing.expectEqual(@as(VReg, 7), bitmask.op.simd_bitmask.vector);

    const load_splat = Inst{
        .op = .{ .v128_load_splat = .{
            .width = .i32x4,
            .base = 6,
            .offset = 12,
            .alignment = 2,
        } },
        .dest = 8,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.V128LoadSplatWidth.i32x4, load_splat.op.v128_load_splat.width);
    try std.testing.expectEqual(@as(VReg, 6), load_splat.op.v128_load_splat.base);
    try std.testing.expectEqual(@as(u32, 12), load_splat.op.v128_load_splat.offset);
    try std.testing.expectEqual(@as(u32, 2), load_splat.op.v128_load_splat.alignment);
    try std.testing.expectEqual(@as(u8, 4), load_splat.op.v128_load_splat.accessSize());

    const store_lane = Inst{
        .op = .{ .v128_store_lane = .{
            .width = .i16,
            .base = 9,
            .offset = 14,
            .alignment = 1,
            .vector = 10,
            .lane = 7,
        } },
        .type = .void,
    };
    try std.testing.expectEqual(Inst.V128LaneWidth.i16, store_lane.op.v128_store_lane.width);
    try std.testing.expectEqual(@as(VReg, 9), store_lane.op.v128_store_lane.base);
    try std.testing.expectEqual(@as(VReg, 10), store_lane.op.v128_store_lane.vector);
    try std.testing.expectEqual(@as(u8, 7), store_lane.op.v128_store_lane.lane);
    try std.testing.expectEqual(@as(u8, 2), store_lane.op.v128_store_lane.accessSize());
    try std.testing.expectEqual(@as(u8, 8), Inst.V128LaneWidth.i16.laneCount());

    const load_zero = Inst{
        .op = .{ .v128_load_zero = .{
            .width = .i64,
            .base = 7,
            .offset = 20,
            .alignment = 3,
        } },
        .dest = 9,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.V128LoadZeroWidth.i64, load_zero.op.v128_load_zero.width);
    try std.testing.expectEqual(@as(VReg, 7), load_zero.op.v128_load_zero.base);
    try std.testing.expectEqual(@as(u32, 20), load_zero.op.v128_load_zero.offset);
    try std.testing.expectEqual(@as(u32, 3), load_zero.op.v128_load_zero.alignment);
    try std.testing.expectEqual(@as(u8, 8), load_zero.op.v128_load_zero.accessSize());

    const load_extend = Inst{
        .op = .{ .v128_load_extend = .{
            .src_width = .i16x4,
            .sign = .signed,
            .base = 8,
            .offset = 24,
            .alignment = 3,
        } },
        .dest = 10,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.V128LoadExtendWidth.i16x4, load_extend.op.v128_load_extend.src_width);
    try std.testing.expectEqual(Inst.SimdExtendSign.signed, load_extend.op.v128_load_extend.sign);
    try std.testing.expectEqual(@as(VReg, 8), load_extend.op.v128_load_extend.base);
    try std.testing.expectEqual(@as(u32, 24), load_extend.op.v128_load_extend.offset);
    try std.testing.expectEqual(@as(u32, 3), load_extend.op.v128_load_extend.alignment);
    try std.testing.expectEqual(@as(u8, 8), load_extend.op.v128_load_extend.accessSize());

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

    const f32_splat = Inst{
        .op = .{ .f32x4_splat = 9 },
        .dest = 10,
        .type = .v128,
    };
    try std.testing.expectEqual(@as(VReg, 9), f32_splat.op.f32x4_splat);

    const f32_lane = Inst{
        .op = .{ .f32x4_extract_lane = .{ .vector = 10, .lane = 3 } },
        .dest = 11,
        .type = .f32,
    };
    try std.testing.expectEqual(@as(VReg, 10), f32_lane.op.f32x4_extract_lane.vector);
    try std.testing.expectEqual(@as(u2, 3), f32_lane.op.f32x4_extract_lane.lane);

    const f32_replace = Inst{
        .op = .{ .f32x4_replace_lane = .{ .vector = 10, .val = 11, .lane = 0 } },
        .dest = 12,
        .type = .v128,
    };
    try std.testing.expectEqual(@as(VReg, 10), f32_replace.op.f32x4_replace_lane.vector);
    try std.testing.expectEqual(@as(VReg, 11), f32_replace.op.f32x4_replace_lane.val);
    try std.testing.expectEqual(@as(u2, 0), f32_replace.op.f32x4_replace_lane.lane);

    const shift = Inst{
        .op = .{ .i32x4_shift = .{ .op = .shr_u, .vector = 8, .count = 9 } },
        .dest = 10,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.I32x4ShiftOp.shr_u, shift.op.i32x4_shift.op);
    try std.testing.expectEqual(@as(VReg, 8), shift.op.i32x4_shift.vector);
    try std.testing.expectEqual(@as(VReg, 9), shift.op.i32x4_shift.count);

    const i32_un = Inst{
        .op = .{ .i32x4_unop = .{ .op = .abs, .vector = 10 } },
        .dest = 11,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdUnaryOp.abs, i32_un.op.i32x4_unop.op);
    try std.testing.expectEqual(@as(VReg, 10), i32_un.op.i32x4_unop.vector);

    const i32_extadd = Inst{
        .op = .{ .i32x4_extadd_pairwise_i16x8 = .{ .sign = .unsigned, .vector = 11 } },
        .dest = 12,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdExtAddPairwiseSign.unsigned, i32_extadd.op.i32x4_extadd_pairwise_i16x8.sign);
    try std.testing.expectEqual(@as(VReg, 11), i32_extadd.op.i32x4_extadd_pairwise_i16x8.vector);

    const i32_extend = Inst{
        .op = .{ .i32x4_extend_i16x8 = .{ .sign = .signed, .half = .high, .vector = 13 } },
        .dest = 14,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdExtendSign.signed, i32_extend.op.i32x4_extend_i16x8.sign);
    try std.testing.expectEqual(Inst.SimdExtendHalfSelect.high, i32_extend.op.i32x4_extend_i16x8.half);
    try std.testing.expectEqual(@as(VReg, 13), i32_extend.op.i32x4_extend_i16x8.vector);

    const i8_bin = Inst{
        .op = .{ .i8x16_binop = .{ .op = .sub, .lhs = 11, .rhs = 12 } },
        .dest = 13,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.I8x16Op.sub, i8_bin.op.i8x16_binop.op);
    try std.testing.expectEqual(@as(VReg, 11), i8_bin.op.i8x16_binop.lhs);

    const i8_shuffle = Inst{
        .op = .{ .i8x16_shuffle = .{
            .lhs = 13,
            .rhs = 14,
            .lanes = .{ 0, 1, 2, 3, 4, 5, 6, 7, 31, 30, 29, 28, 27, 26, 25, 24 },
        } },
        .dest = 15,
        .type = .v128,
    };
    try std.testing.expectEqual(@as(VReg, 13), i8_shuffle.op.i8x16_shuffle.lhs);
    try std.testing.expectEqual(@as(VReg, 14), i8_shuffle.op.i8x16_shuffle.rhs);
    try std.testing.expectEqual(@as(u8, 31), i8_shuffle.op.i8x16_shuffle.lanes[8]);

    const i8_swizzle = Inst{
        .op = .{ .i8x16_swizzle = .{ .vector = 13, .indices = 14 } },
        .dest = 15,
        .type = .v128,
    };
    try std.testing.expectEqual(@as(VReg, 13), i8_swizzle.op.i8x16_swizzle.vector);
    try std.testing.expectEqual(@as(VReg, 14), i8_swizzle.op.i8x16_swizzle.indices);

    const i8_shift = Inst{
        .op = .{ .i8x16_shift = .{ .op = .shr_s, .vector = 13, .count = 14 } },
        .dest = 15,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.I8x16ShiftOp.shr_s, i8_shift.op.i8x16_shift.op);
    try std.testing.expectEqual(@as(VReg, 13), i8_shift.op.i8x16_shift.vector);
    try std.testing.expectEqual(@as(VReg, 14), i8_shift.op.i8x16_shift.count);

    const i8_un = Inst{
        .op = .{ .i8x16_unop = .{ .op = .neg, .vector = 15 } },
        .dest = 16,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdUnaryOp.neg, i8_un.op.i8x16_unop.op);
    try std.testing.expectEqual(@as(VReg, 15), i8_un.op.i8x16_unop.vector);

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

    const i16_un = Inst{
        .op = .{ .i16x8_unop = .{ .op = .abs, .vector = 20 } },
        .dest = 21,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdUnaryOp.abs, i16_un.op.i16x8_unop.op);
    try std.testing.expectEqual(@as(VReg, 20), i16_un.op.i16x8_unop.vector);

    const i16_extadd = Inst{
        .op = .{ .i16x8_extadd_pairwise_i8x16 = .{ .sign = .signed, .vector = 21 } },
        .dest = 22,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdExtAddPairwiseSign.signed, i16_extadd.op.i16x8_extadd_pairwise_i8x16.sign);
    try std.testing.expectEqual(@as(VReg, 21), i16_extadd.op.i16x8_extadd_pairwise_i8x16.vector);

    const i16_extend = Inst{
        .op = .{ .i16x8_extend_i8x16 = .{ .sign = .unsigned, .half = .low, .vector = 23 } },
        .dest = 24,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdExtendSign.unsigned, i16_extend.op.i16x8_extend_i8x16.sign);
    try std.testing.expectEqual(Inst.SimdExtendHalfSelect.low, i16_extend.op.i16x8_extend_i8x16.half);
    try std.testing.expectEqual(@as(VReg, 23), i16_extend.op.i16x8_extend_i8x16.vector);

    const i64_extend = Inst{
        .op = .{ .i64x2_extend_i32x4 = .{ .sign = .signed, .half = .low, .vector = 25 } },
        .dest = 26,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdExtendSign.signed, i64_extend.op.i64x2_extend_i32x4.sign);
    try std.testing.expectEqual(Inst.SimdExtendHalfSelect.low, i64_extend.op.i64x2_extend_i32x4.half);
    try std.testing.expectEqual(@as(VReg, 25), i64_extend.op.i64x2_extend_i32x4.vector);

    const f32_convert = Inst{
        .op = .{ .f32x4_convert_i32x4 = .{ .sign = .unsigned, .vector = 26 } },
        .dest = 27,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdIntToFloatSign.unsigned, f32_convert.op.f32x4_convert_i32x4.sign);
    try std.testing.expectEqual(@as(VReg, 26), f32_convert.op.f32x4_convert_i32x4.vector);

    const i32_trunc = Inst{
        .op = .{ .i32x4_trunc_sat = .{ .src_width = .f64x2, .sign = .signed, .vector = 27 } },
        .dest = 28,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdFloatToIntSrcWidth.f64x2, i32_trunc.op.i32x4_trunc_sat.src_width);
    try std.testing.expectEqual(Inst.SimdFloatToIntSign.signed, i32_trunc.op.i32x4_trunc_sat.sign);
    try std.testing.expectEqual(@as(VReg, 27), i32_trunc.op.i32x4_trunc_sat.vector);

    const f32_demote = Inst{
        .op = .{ .f32x4_demote_f64x2_zero = .{ .vector = 26 } },
        .dest = 28,
        .type = .v128,
    };
    try std.testing.expectEqual(@as(VReg, 26), f32_demote.op.f32x4_demote_f64x2_zero.vector);

    const f32_un = Inst{
        .op = .{ .f32x4_unop = .{ .op = .sqrt, .vector = 28 } },
        .dest = 29,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.F32x4UnaryOp.sqrt, f32_un.op.f32x4_unop.op);
    try std.testing.expectEqual(@as(VReg, 28), f32_un.op.f32x4_unop.vector);

    const f32_bin = Inst{
        .op = .{ .f32x4_binop = .{ .op = .pmax, .lhs = 29, .rhs = 30 } },
        .dest = 31,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.F32x4Op.pmax, f32_bin.op.f32x4_binop.op);
    try std.testing.expectEqual(@as(VReg, 29), f32_bin.op.f32x4_binop.lhs);
    try std.testing.expectEqual(@as(VReg, 30), f32_bin.op.f32x4_binop.rhs);

    const i16_extmul = Inst{
        .op = .{ .i16x8_extmul_i8x16 = .{ .sign = .signed, .half = .high, .lhs = 30, .rhs = 31 } },
        .dest = 32,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdExtendSign.signed, i16_extmul.op.i16x8_extmul_i8x16.sign);
    try std.testing.expectEqual(Inst.SimdExtendHalfSelect.high, i16_extmul.op.i16x8_extmul_i8x16.half);
    try std.testing.expectEqual(@as(VReg, 30), i16_extmul.op.i16x8_extmul_i8x16.lhs);
    try std.testing.expectEqual(@as(VReg, 31), i16_extmul.op.i16x8_extmul_i8x16.rhs);

    const i32_extmul = Inst{
        .op = .{ .i32x4_extmul_i16x8 = .{ .sign = .unsigned, .half = .low, .lhs = 30, .rhs = 31 } },
        .dest = 32,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdExtendSign.unsigned, i32_extmul.op.i32x4_extmul_i16x8.sign);
    try std.testing.expectEqual(Inst.SimdExtendHalfSelect.low, i32_extmul.op.i32x4_extmul_i16x8.half);

    const i64_extmul = Inst{
        .op = .{ .i64x2_extmul_i32x4 = .{ .sign = .signed, .half = .high, .lhs = 33, .rhs = 34 } },
        .dest = 35,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdExtendSign.signed, i64_extmul.op.i64x2_extmul_i32x4.sign);
    try std.testing.expectEqual(Inst.SimdExtendHalfSelect.high, i64_extmul.op.i64x2_extmul_i32x4.half);

    const i8_narrow = Inst{
        .op = .{ .i8x16_narrow_i16x8 = .{ .sign = .signed, .lhs = 36, .rhs = 37 } },
        .dest = 38,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdNarrowSign.signed, i8_narrow.op.i8x16_narrow_i16x8.sign);
    try std.testing.expectEqual(@as(VReg, 36), i8_narrow.op.i8x16_narrow_i16x8.lhs);
    try std.testing.expectEqual(@as(VReg, 37), i8_narrow.op.i8x16_narrow_i16x8.rhs);

    const i16_narrow = Inst{
        .op = .{ .i16x8_narrow_i32x4 = .{ .sign = .unsigned, .lhs = 39, .rhs = 40 } },
        .dest = 41,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdNarrowSign.unsigned, i16_narrow.op.i16x8_narrow_i32x4.sign);

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

    const i64_shift = Inst{
        .op = .{ .i64x2_shift = .{ .op = .shr_u, .vector = 22, .count = 23 } },
        .dest = 24,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.I64x2ShiftOp.shr_u, i64_shift.op.i64x2_shift.op);
    try std.testing.expectEqual(@as(VReg, 22), i64_shift.op.i64x2_shift.vector);
    try std.testing.expectEqual(@as(VReg, 23), i64_shift.op.i64x2_shift.count);

    const i64_un = Inst{
        .op = .{ .i64x2_unop = .{ .op = .neg, .vector = 24 } },
        .dest = 25,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdUnaryOp.neg, i64_un.op.i64x2_unop.op);
    try std.testing.expectEqual(@as(VReg, 24), i64_un.op.i64x2_unop.vector);

    const i64_bin = Inst{
        .op = .{ .i64x2_binop = .{ .op = .gt_s, .lhs = 23, .rhs = 24 } },
        .dest = 25,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.I64x2Op.gt_s, i64_bin.op.i64x2_binop.op);
    try std.testing.expectEqual(@as(VReg, 23), i64_bin.op.i64x2_binop.lhs);

    const f64_un = Inst{
        .op = .{ .f64x2_unop = .{ .op = .sqrt, .vector = 24 } },
        .dest = 26,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.F64x2UnaryOp.sqrt, f64_un.op.f64x2_unop.op);
    try std.testing.expectEqual(@as(VReg, 24), f64_un.op.f64x2_unop.vector);

    const f64_bin = Inst{
        .op = .{ .f64x2_binop = .{ .op = .div, .lhs = 24, .rhs = 25 } },
        .dest = 26,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.F64x2Op.div, f64_bin.op.f64x2_binop.op);
    try std.testing.expectEqual(@as(VReg, 24), f64_bin.op.f64x2_binop.lhs);

    const f64_convert = Inst{
        .op = .{ .f64x2_convert_low_i32x4 = .{ .sign = .signed, .vector = 26 } },
        .dest = 27,
        .type = .v128,
    };
    try std.testing.expectEqual(Inst.SimdIntToFloatSign.signed, f64_convert.op.f64x2_convert_low_i32x4.sign);
    try std.testing.expectEqual(@as(VReg, 26), f64_convert.op.f64x2_convert_low_i32x4.vector);

    const f64_promote = Inst{
        .op = .{ .f64x2_promote_low_f32x4 = .{ .vector = 26 } },
        .dest = 27,
        .type = .v128,
    };
    try std.testing.expectEqual(@as(VReg, 26), f64_promote.op.f64x2_promote_low_f32x4.vector);

    const f64_splat = Inst{
        .op = .{ .f64x2_splat = 28 },
        .dest = 29,
        .type = .v128,
    };
    try std.testing.expectEqual(@as(VReg, 28), f64_splat.op.f64x2_splat);

    const f64_extract = Inst{
        .op = .{ .f64x2_extract_lane = .{ .vector = 29, .lane = 1 } },
        .dest = 30,
        .type = .f64,
    };
    try std.testing.expectEqual(@as(VReg, 29), f64_extract.op.f64x2_extract_lane.vector);
    try std.testing.expectEqual(@as(u1, 1), f64_extract.op.f64x2_extract_lane.lane);

    const f64_replace = Inst{
        .op = .{ .f64x2_replace_lane = .{ .vector = 29, .val = 30, .lane = 0 } },
        .dest = 31,
        .type = .v128,
    };
    try std.testing.expectEqual(@as(VReg, 29), f64_replace.op.f64x2_replace_lane.vector);
    try std.testing.expectEqual(@as(VReg, 30), f64_replace.op.f64x2_replace_lane.val);
    try std.testing.expectEqual(@as(u1, 0), f64_replace.op.f64x2_replace_lane.lane);

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
