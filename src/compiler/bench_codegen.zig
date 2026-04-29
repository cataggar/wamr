//! x86-64 Codegen Microbenchmark
//!
//! Measures compilation throughput and generated code size for atomics,
//! arithmetic, branches, memory, calls, floats, register pressure, and
//! the effect of IR optimization passes.
//! Run with: zig build bench

const std = @import("std");
const ir = @import("ir/ir.zig");
const passes = @import("ir/passes.zig");
const compile = @import("codegen/x86_64/compile.zig");

/// Read the CPU timestamp counter (RDTSC) for cycle-accurate timing.
inline fn rdtsc() u64 {
    var lo: u32 = undefined;
    var hi: u32 = undefined;
    asm volatile ("rdtsc"
        : [lo] "={eax}" (lo),
          [hi] "={edx}" (hi),
    );
    return (@as(u64, hi) << 32) | lo;
}

const BenchResult = struct {
    name: []const u8,
    iterations: u64,
    total_cycles: u64,
    code_size: usize,

    fn cyclesPerOp(self: BenchResult) u64 {
        if (self.iterations == 0) return 0;
        return self.total_cycles / self.iterations;
    }
};

const BuildBodyFn = *const fn (*ir.IrFunction, *ir.BasicBlock) void;

fn buildTestFunc(
    allocator: std.mem.Allocator,
    buildBody: BuildBodyFn,
) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    const block_id = func.newBlock() catch unreachable;
    const block = func.getBlock(block_id);
    buildBody(&func, block);
    return func;
}

fn runBench(
    allocator: std.mem.Allocator,
    name: []const u8,
    buildBody: BuildBodyFn,
) !BenchResult {
    var func = try buildTestFunc(allocator, buildBody);
    defer func.deinit();

    const sample_result = try compile.compileFunctionRA(&func, 0, allocator);
    const code_size = sample_result.code.len;
    defer allocator.free(sample_result.code);
    defer allocator.free(sample_result.call_patches);

    // Warmup
    for (0..200) |_| {
        const r = try compile.compileFunctionRA(&func, 0, allocator);
        allocator.free(r.code);
        allocator.free(r.call_patches);
    }

    // Timed iterations (fixed count for consistency)
    const iterations: u64 = 10_000;
    const start = rdtsc();

    for (0..iterations) |_| {
        const r = try compile.compileFunctionRA(&func, 0, allocator);
        allocator.free(r.code);
        allocator.free(r.call_patches);
    }

    const end = rdtsc();

    return .{
        .name = name,
        .iterations = iterations,
        .total_cycles = end - start,
        .code_size = code_size,
    };
}

// ── Benchmark bodies ──────────────────────────────────────────────────

fn bodyFence(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    block.append(.{ .op = .{ .atomic_fence = {} } }) catch unreachable;
    _ = func;
}

fn bodyLoad32(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const loaded = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base }) catch unreachable;
    block.append(.{ .op = .{ .atomic_load = .{ .base = base, .offset = 0, .size = 4 } }, .dest = loaded }) catch unreachable;
    block.append(.{ .op = .{ .ret = loaded } }) catch unreachable;
}

fn bodyStore32(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const val = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = val }) catch unreachable;
    block.append(.{ .op = .{ .atomic_store = .{ .base = base, .offset = 0, .size = 4, .val = val } } }) catch unreachable;
}

fn bodyRmwAdd32(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const val = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = val }) catch unreachable;
    block.append(.{ .op = .{ .atomic_rmw = .{ .base = base, .offset = 0, .size = 4, .val = val, .op = .add } }, .dest = result }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

fn bodyRmwSub32(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const val = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = val }) catch unreachable;
    block.append(.{ .op = .{ .atomic_rmw = .{ .base = base, .offset = 0, .size = 4, .val = val, .op = .sub } }, .dest = result }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

fn bodyRmwAnd32(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const val = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 0xFF }, .dest = val }) catch unreachable;
    block.append(.{ .op = .{ .atomic_rmw = .{ .base = base, .offset = 0, .size = 4, .val = val, .op = .@"and" } }, .dest = result }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

fn bodyRmwXchg32(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const val = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 99 }, .dest = val }) catch unreachable;
    block.append(.{ .op = .{ .atomic_rmw = .{ .base = base, .offset = 0, .size = 4, .val = val, .op = .xchg } }, .dest = result }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

fn bodyCmpxchg32(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const expected = func.newVReg();
    const replacement = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = expected }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = replacement }) catch unreachable;
    block.append(.{ .op = .{ .atomic_cmpxchg = .{ .base = base, .offset = 0, .size = 4, .expected = expected, .replacement = replacement } }, .dest = result }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

fn bodyLoad64(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const loaded = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base }) catch unreachable;
    block.append(.{ .op = .{ .atomic_load = .{ .base = base, .offset = 0, .size = 8 } }, .dest = loaded, .type = .i64 }) catch unreachable;
    block.append(.{ .op = .{ .ret = loaded } }) catch unreachable;
}

fn bodyRmwAdd8(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const val = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = val }) catch unreachable;
    block.append(.{ .op = .{ .atomic_rmw = .{ .base = base, .offset = 0, .size = 1, .val = val, .op = .add } }, .dest = result }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

/// Body with a shl by a compile-time constant — exercises the shift-imm
/// fast path (C1/D1 form, no CL load). Issue #137.
fn bodyShlImm(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const x = func.newVReg();
    const k = func.newVReg();
    const r = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = x, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = k, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .shl = .{ .lhs = x, .rhs = k } }, .dest = r, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = r } }) catch unreachable;
}

/// Body with several dead intermediate values — used to demonstrate DCE effect.
fn bodyDeadIntermediates(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const a = func.newVReg();
    const b = func.newVReg();
    const dead1 = func.newVReg();
    const dead2 = func.newVReg();
    const dead3 = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = a }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 4 }, .dest = b }) catch unreachable;
    // These three are never used — DCE should remove them.
    block.append(.{ .op = .{ .iconst_32 = 999 }, .dest = dead1 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = a, .rhs = b } }, .dest = dead2 }) catch unreachable;
    block.append(.{ .op = .{ .mul = .{ .lhs = a, .rhs = b } }, .dest = dead3 }) catch unreachable;
    // Only this result is returned.
    block.append(.{ .op = .{ .add = .{ .lhs = a, .rhs = b } }, .dest = result }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

/// Body that multiplies a function-result placeholder by a power-of-two
/// constant (8). With the `strengthReduceMul` pass this becomes `shl x, 3`.
fn bodyMulByPow2(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const x = func.newVReg();
    const c = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = x }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 8 }, .dest = c }) catch unreachable;
    block.append(.{ .op = .{ .mul = .{ .lhs = x, .rhs = c } }, .dest = result }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

/// Three consecutive loads from the same base at offsets 0, 4, 8 (each i32).
/// With `elideRedundantBoundsChecks` the first load's check is widened to
/// end=12 and the second/third loads skip their checks entirely.
fn bodyConsecutiveLoads(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } }, .dest = v0, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = base, .offset = 4, .size = 4 } }, .dest = v1, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = base, .offset = 8, .size = 4 } }, .dest = v2, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = v2 } }) catch unreachable;
}

// ── Arithmetic benchmark bodies ───────────────────────────────────────

/// Chain of add + sub — exercises 2-operand register forms and LEA folding.
fn bodyAddSub(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const a = func.newVReg();
    const b = func.newVReg();
    const r1 = func.newVReg();
    const r2 = func.newVReg();
    const r3 = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = a, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = b, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = a, .rhs = b } }, .dest = r1, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .sub = .{ .lhs = r1, .rhs = b } }, .dest = r2, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = r2, .rhs = r1 } }, .dest = r3, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = r3 } }) catch unreachable;
}

/// mul + div_u pair — exercises rax/rdx fixed-register handling for division.
fn bodyMulDiv(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const a = func.newVReg();
    const b = func.newVReg();
    const product = func.newVReg();
    const quotient = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = a, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = b, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .mul = .{ .lhs = a, .rhs = b } }, .dest = product, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .div_u = .{ .lhs = product, .rhs = b } }, .dest = quotient, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = quotient } }) catch unreachable;
}

/// Bitwise chain: and + or + xor — exercises simple ALU instruction selection.
fn bodyBitwiseChain(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const a = func.newVReg();
    const b = func.newVReg();
    const r1 = func.newVReg();
    const r2 = func.newVReg();
    const r3 = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0xFF00 }, .dest = a, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 0x0FF0 }, .dest = b, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .@"and" = .{ .lhs = a, .rhs = b } }, .dest = r1, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .@"or" = .{ .lhs = a, .rhs = b } }, .dest = r2, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .xor = .{ .lhs = r1, .rhs = r2 } }, .dest = r3, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = r3 } }) catch unreachable;
}

/// div_u by constant 7 — exercises strength-reduction pass (magic multiply).
fn bodyDivByConst(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const x = func.newVReg();
    const d = func.newVReg();
    const q = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = x, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = d, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .div_u = .{ .lhs = x, .rhs = d } }, .dest = q, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = q } }) catch unreachable;
}

/// Prior checked base access followed by `add base, const` feeding a load.
/// With `foldLoadStoreOffset`, the second load uses `base` with a larger
/// immediate offset and the add becomes dead.
fn bodyAddIntoLoad(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const guard = func.newVReg();
    const c = func.newVReg();
    const addr = func.newVReg();
    const loaded = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = base, .offset = 0, .size = 4, .checked_end = 32 } }, .dest = guard, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 12 }, .dest = c, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = base, .rhs = c } }, .dest = addr, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = addr, .offset = 4, .size = 4 } }, .dest = loaded, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = loaded } }) catch unreachable;
}

// ── Branch benchmark bodies ───────────────────────────────────────────

/// compare + br_if diamond — exercises Jcc fusion and branch layout.
fn bodyBrIf(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const a = func.newVReg();
    const b = func.newVReg();
    const cond = func.newVReg();
    const r1 = func.newVReg();
    const r2 = func.newVReg();

    const b1 = func.newBlock() catch unreachable;
    const b2 = func.newBlock() catch unreachable;
    const block1 = func.getBlock(b1);
    const block2 = func.getBlock(b2);

    block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = a, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = b, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .lt_u = .{ .lhs = a, .rhs = b } }, .dest = cond, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } }) catch unreachable;

    block1.append(.{ .op = .{ .iconst_32 = 1 }, .dest = r1, .type = .i32 }) catch unreachable;
    block1.append(.{ .op = .{ .ret = r1 } }) catch unreachable;

    block2.append(.{ .op = .{ .iconst_32 = 0 }, .dest = r2, .type = .i32 }) catch unreachable;
    block2.append(.{ .op = .{ .ret = r2 } }) catch unreachable;
}

/// br_table with 4 targets — exercises jump table codegen.
fn bodyBrTable(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const idx = func.newVReg();
    const r0 = func.newVReg();
    const r1 = func.newVReg();
    const r2 = func.newVReg();
    const r3 = func.newVReg();

    const t0 = func.newBlock() catch unreachable;
    const t1 = func.newBlock() catch unreachable;
    const t2 = func.newBlock() catch unreachable;
    const t3 = func.newBlock() catch unreachable;

    block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = idx, .type = .i32 }) catch unreachable;
    const targets = &[_]ir.BlockId{ t0, t1, t2, t3 };
    block.append(.{ .op = .{ .br_table = .{ .index = idx, .targets = targets, .default = t0 } } }) catch unreachable;

    func.getBlock(t0).append(.{ .op = .{ .iconst_32 = 10 }, .dest = r0, .type = .i32 }) catch unreachable;
    func.getBlock(t0).append(.{ .op = .{ .ret = r0 } }) catch unreachable;
    func.getBlock(t1).append(.{ .op = .{ .iconst_32 = 20 }, .dest = r1, .type = .i32 }) catch unreachable;
    func.getBlock(t1).append(.{ .op = .{ .ret = r1 } }) catch unreachable;
    func.getBlock(t2).append(.{ .op = .{ .iconst_32 = 30 }, .dest = r2, .type = .i32 }) catch unreachable;
    func.getBlock(t2).append(.{ .op = .{ .ret = r2 } }) catch unreachable;
    func.getBlock(t3).append(.{ .op = .{ .iconst_32 = 40 }, .dest = r3, .type = .i32 }) catch unreachable;
    func.getBlock(t3).append(.{ .op = .{ .ret = r3 } }) catch unreachable;
}

/// select instructions — exercises conditional-move codegen.
fn bodySelectChain(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const a = func.newVReg();
    const b = func.newVReg();
    const c = func.newVReg();
    const cond = func.newVReg();
    const s1 = func.newVReg();
    const s2 = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = a, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = b, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = c, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .select = .{ .cond = cond, .if_true = a, .if_false = b } }, .dest = s1, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .select = .{ .cond = cond, .if_true = s1, .if_false = c } }, .dest = s2, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = s2 } }) catch unreachable;
}

// ── Memory benchmark bodies ──────────────────────────────────────────

/// load + compute + store — exercises address mode, bounds checking, and ALU.
fn bodyLoadStore(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const loaded = func.newVReg();
    const one = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } }, .dest = loaded, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = one, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = loaded, .rhs = one } }, .dest = result, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .store = .{ .base = base, .offset = 4, .size = 4, .val = result } } }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

/// Multiple loads at different offsets → sum → store back.
/// Exercises bounds-check elision across multiple accesses.
fn bodyLoadStoreMulti(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    const v3 = func.newVReg();
    const sum1 = func.newVReg();
    const sum2 = func.newVReg();
    const sum3 = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } }, .dest = v0, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = base, .offset = 4, .size = 4 } }, .dest = v1, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = base, .offset = 8, .size = 4 } }, .dest = v2, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = base, .offset = 12, .size = 4 } }, .dest = v3, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = sum1, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = sum1, .rhs = v2 } }, .dest = sum2, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = sum2, .rhs = v3 } }, .dest = sum3, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .store = .{ .base = base, .offset = 16, .size = 4, .val = sum3 } } }) catch unreachable;
    block.append(.{ .op = .{ .ret = sum3 } }) catch unreachable;
}

// ── Call benchmark bodies ────────────────────────────────────────────

/// call + ret — exercises call ABI, caller-saved save/restore.
fn bodyCallRet(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const result = func.newVReg();
    block.append(.{ .op = .{ .call = .{ .func_idx = 0, .args = &.{} } }, .dest = result, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

// ── Float benchmark bodies ───────────────────────────────────────────

/// f64 add + mul chain — exercises XMM register allocation and float codegen.
fn bodyFloatArith(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const a = func.newVReg();
    const b = func.newVReg();
    const sum = func.newVReg();
    const product = func.newVReg();
    block.append(.{ .op = .{ .fconst_64 = 3.14 }, .dest = a, .type = .f64 }) catch unreachable;
    block.append(.{ .op = .{ .fconst_64 = 2.71 }, .dest = b, .type = .f64 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = a, .rhs = b } }, .dest = sum, .type = .f64 }) catch unreachable;
    block.append(.{ .op = .{ .mul = .{ .lhs = sum, .rhs = b } }, .dest = product, .type = .f64 }) catch unreachable;
    block.append(.{ .op = .{ .ret = product } }) catch unreachable;
}

/// i32 → f64 convert + f64 add — exercises mixed int/float pipeline.
fn bodyIntToFloat(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const x = func.newVReg();
    const xf = func.newVReg();
    const bias = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = x, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .convert_i32_s = x }, .dest = xf, .type = .f64 }) catch unreachable;
    block.append(.{ .op = .{ .fconst_64 = 0.5 }, .dest = bias, .type = .f64 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = xf, .rhs = bias } }, .dest = result, .type = .f64 }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

// ── Register pressure benchmark bodies ───────────────────────────────

/// 10+ live values through adds — exercises register spilling.
fn bodyRegPressure(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    var vregs: [12]ir.VReg = undefined;
    for (&vregs, 0..) |*v, i| {
        v.* = func.newVReg();
        block.append(.{ .op = .{ .iconst_32 = @as(i32, @intCast(i + 1)) }, .dest = v.*, .type = .i32 }) catch unreachable;
    }
    // Chain additions that keep all values live until the end
    var acc = vregs[0];
    for (vregs[1..]) |v| {
        const next = func.newVReg();
        block.append(.{ .op = .{ .add = .{ .lhs = acc, .rhs = v } }, .dest = next, .type = .i32 }) catch unreachable;
        acc = next;
    }
    block.append(.{ .op = .{ .ret = acc } }) catch unreachable;
}

fn runBenchWithPasses(
    allocator: std.mem.Allocator,
    name: []const u8,
    buildBody: BuildBodyFn,
) !BenchResult {
    // Build function, run passes once, then time repeated codegen.
    var module = ir.IrModule.init(allocator);
    defer module.deinit();
    const func = try buildTestFunc(allocator, buildBody);
    _ = try module.addFunction(func);
    _ = try passes.runPasses(&module, passes.default_passes, allocator);

    const sample_result = try compile.compileFunctionRA(&module.functions.items[0], 0, allocator);
    const code_size = sample_result.code.len;
    defer allocator.free(sample_result.code);
    defer allocator.free(sample_result.call_patches);

    // Warmup
    for (0..200) |_| {
        const r = try compile.compileFunctionRA(&module.functions.items[0], 0, allocator);
        allocator.free(r.code);
        allocator.free(r.call_patches);
    }

    // Timed iterations
    const iterations: u64 = 10_000;
    const start = rdtsc();

    for (0..iterations) |_| {
        const r = try compile.compileFunctionRA(&module.functions.items[0], 0, allocator);
        allocator.free(r.code);
        allocator.free(r.call_patches);
    }

    const end = rdtsc();

    return .{
        .name = name,
        .iterations = iterations,
        .total_cycles = end - start,
        .code_size = code_size,
    };
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n", .{});
    std.debug.print("  x86-64 Codegen Benchmark (10,000 iterations each)\n", .{});
    std.debug.print("  ===================================================\n\n", .{});

    // ── Atomic operations (raw codegen, no passes) ──────────────────
    std.debug.print("  Atomic operations\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", "cycles/op", "code bytes" });
    std.debug.print("  {s:-<34} {s:->12} {s:->10}\n", .{ "", "", "" });

    const atomic_benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
        .{ .name = "atomic_fence", .body = &bodyFence },
        .{ .name = "atomic_load i32", .body = &bodyLoad32 },
        .{ .name = "atomic_load i64", .body = &bodyLoad64 },
        .{ .name = "atomic_store i32", .body = &bodyStore32 },
        .{ .name = "atomic_rmw add i32", .body = &bodyRmwAdd32 },
        .{ .name = "atomic_rmw sub i32", .body = &bodyRmwSub32 },
        .{ .name = "atomic_rmw and i32 (CAS)", .body = &bodyRmwAnd32 },
        .{ .name = "atomic_rmw xchg i32", .body = &bodyRmwXchg32 },
        .{ .name = "atomic_rmw add i8", .body = &bodyRmwAdd8 },
        .{ .name = "atomic_cmpxchg i32", .body = &bodyCmpxchg32 },
    };
    for (atomic_benchmarks) |b| {
        const result = try runBench(allocator, b.name, b.body);
        std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{ result.name, result.cyclesPerOp(), result.code_size });
    }

    // ── Arithmetic (raw codegen) ────────────────────────────────────
    std.debug.print("\n  Arithmetic\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", "cycles/op", "code bytes" });
    std.debug.print("  {s:-<34} {s:->12} {s:->10}\n", .{ "", "", "" });

    const arith_benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
        .{ .name = "add + sub chain", .body = &bodyAddSub },
        .{ .name = "mul + div_u", .body = &bodyMulDiv },
        .{ .name = "and + or + xor chain", .body = &bodyBitwiseChain },
        .{ .name = "shl i32 by const 3", .body = &bodyShlImm },
    };
    for (arith_benchmarks) |b| {
        const result = try runBench(allocator, b.name, b.body);
        std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{ result.name, result.cyclesPerOp(), result.code_size });
    }

    // ── Branches + control flow (raw codegen) ──────────────────────
    std.debug.print("\n  Branches + control flow\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", "cycles/op", "code bytes" });
    std.debug.print("  {s:-<34} {s:->12} {s:->10}\n", .{ "", "", "" });

    const branch_benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
        .{ .name = "cmp + br_if diamond", .body = &bodyBrIf },
        .{ .name = "br_table (4 targets)", .body = &bodyBrTable },
        .{ .name = "select chain (cmov)", .body = &bodySelectChain },
    };
    for (branch_benchmarks) |b| {
        const result = try runBench(allocator, b.name, b.body);
        std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{ result.name, result.cyclesPerOp(), result.code_size });
    }

    // ── Memory (raw codegen) ───────────────────────────────────────
    std.debug.print("\n  Memory\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", "cycles/op", "code bytes" });
    std.debug.print("  {s:-<34} {s:->12} {s:->10}\n", .{ "", "", "" });

    const memory_benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
        .{ .name = "load + add + store", .body = &bodyLoadStore },
        .{ .name = "4× load + sum + store", .body = &bodyLoadStoreMulti },
    };
    for (memory_benchmarks) |b| {
        const result = try runBench(allocator, b.name, b.body);
        std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{ result.name, result.cyclesPerOp(), result.code_size });
    }

    // ── Calls (raw codegen) ────────────────────────────────────────
    std.debug.print("\n  Calls\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", "cycles/op", "code bytes" });
    std.debug.print("  {s:-<34} {s:->12} {s:->10}\n", .{ "", "", "" });

    const call_benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
        .{ .name = "call + ret", .body = &bodyCallRet },
    };
    for (call_benchmarks) |b| {
        const result = try runBench(allocator, b.name, b.body);
        std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{ result.name, result.cyclesPerOp(), result.code_size });
    }

    // ── Float (raw codegen) ────────────────────────────────────────
    std.debug.print("\n  Float\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", "cycles/op", "code bytes" });
    std.debug.print("  {s:-<34} {s:->12} {s:->10}\n", .{ "", "", "" });

    const float_benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
        .{ .name = "f64 add + mul", .body = &bodyFloatArith },
        .{ .name = "i32→f64 convert + add", .body = &bodyIntToFloat },
    };
    for (float_benchmarks) |b| {
        const result = try runBench(allocator, b.name, b.body);
        std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{ result.name, result.cyclesPerOp(), result.code_size });
    }

    // ── Optimization passes (codegen after default_passes) ─────────
    std.debug.print("\n  Optimization passes (codegen after default_passes)\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", "cycles/op", "code bytes" });
    std.debug.print("  {s:-<34} {s:->12} {s:->10}\n", .{ "", "", "" });

    const pass_benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
        .{ .name = "dead intermediates (DCE)", .body = &bodyDeadIntermediates },
        .{ .name = "mul(x, 8) → shl(x, 3)", .body = &bodyMulByPow2 },
        .{ .name = "3× load same base (hoisted)", .body = &bodyConsecutiveLoads },
        .{ .name = "div_u by const 7 (magic mul)", .body = &bodyDivByConst },
        .{ .name = "add base,const → load offset", .body = &bodyAddIntoLoad },
        .{ .name = "4× load + sum (bounds elide)", .body = &bodyLoadStoreMulti },
        .{ .name = "reg pressure (12 live vals)", .body = &bodyRegPressure },
    };
    for (pass_benchmarks) |b| {
        const result = try runBenchWithPasses(allocator, b.name, b.body);
        std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{ result.name, result.cyclesPerOp(), result.code_size });
    }

    std.debug.print("\n", .{});
}
