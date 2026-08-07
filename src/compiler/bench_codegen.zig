//! Codegen Microbenchmark
//!
//! Measures compilation throughput and generated code size for x86-64
//! codegen, IR optimization passes, and AArch64 scheduler-focused kernels.
//! Run with: zig build bench

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("ir/ir.zig");
const passes = @import("ir/passes.zig");
const compile = @import("codegen/x86_64/compile.zig");
const aarch64_compile = @import("codegen/aarch64/compile.zig");

fn usesCycleCounter() bool {
    return switch (builtin.cpu.arch) {
        .x86, .x86_64 => true,
        else => false,
    };
}

fn sampleUnit() []const u8 {
    return if (comptime usesCycleCounter()) "cycles/op" else "ns/op";
}

/// Read the fastest available monotonic-ish sample source for benchmark timing.
/// Returns CPU cycles on x86 (via rdtsc) and nanoseconds on every other target.
inline fn sampleTicks() u64 {
    if (comptime usesCycleCounter()) {
        var lo: u32 = undefined;
        var hi: u32 = undefined;
        asm volatile ("rdtsc"
            : [lo] "={eax}" (lo),
              [hi] "={edx}" (hi),
        );
        return (@as(u64, hi) << 32) | lo;
    }
    return switch (comptime builtin.os.tag) {
        .linux => blk: {
            const linux = std.os.linux;
            var ts: linux.timespec = undefined;
            const rc = linux.clock_gettime(.MONOTONIC, &ts);
            if (rc != 0) @panic("clock_gettime(CLOCK_MONOTONIC) failed");
            break :blk @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
        .macos, .ios, .tvos, .watchos, .visionos => blk: {
            // Darwin: clock_gettime(CLOCK_MONOTONIC) lives in libSystem; the
            // bench module is built with link_libc=true on these targets.
            var ts: std.c.timespec = undefined;
            if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) {
                @panic("clock_gettime(CLOCK_MONOTONIC) failed");
            }
            break :blk @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
        .windows => blk: {
            // Windows: convert QueryPerformanceCounter ticks to nanoseconds so
            // the reported unit ("ns/op") is meaningful regardless of the
            // host's perf-counter frequency.
            const ntdll = std.os.windows.ntdll;
            var counter: std.os.windows.LARGE_INTEGER = undefined;
            var freq: std.os.windows.LARGE_INTEGER = undefined;
            _ = ntdll.RtlQueryPerformanceCounter(&counter);
            _ = ntdll.RtlQueryPerformanceFrequency(&freq);
            const ticks: u128 = @intCast(counter);
            const hz: u128 = @intCast(freq);
            break :blk @as(u64, @truncate(ticks * std.time.ns_per_s / hz));
        },
        else => @compileError("codegen benchmark needs a portable timer for this target"),
    };
}

const BenchResult = struct {
    name: []const u8,
    iterations: u64,
    total_ticks: u64,
    code_size: usize,

    fn ticksPerOp(self: BenchResult) u64 {
        if (self.iterations == 0) return 0;
        return self.total_ticks / self.iterations;
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
    const start = sampleTicks();

    for (0..iterations) |_| {
        const r = try compile.compileFunctionRA(&func, 0, allocator);
        allocator.free(r.code);
        allocator.free(r.call_patches);
    }

    const end = sampleTicks();

    return .{
        .name = name,
        .iterations = iterations,
        .total_ticks = end - start,
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

/// Compound address `base + index*4 + 16` fed by two opaque memory loads.
/// `foldCompoundLea` collapses the `shl` + inner `add` + outer `add` chain
/// into a single `lea` (#543). The base/index come from loads rather than
/// constants because `constantFold` would otherwise collapse the whole
/// expression before the LEA fold could fire.
fn bodyLeaCompound(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const mem = func.newVReg();
    const base = func.newVReg();
    const index = func.newVReg();
    const two = func.newVReg();
    const shifted = func.newVReg();
    const inner = func.newVReg();
    const c16 = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = mem, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = mem, .offset = 0, .size = 4 } }, .dest = base, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = mem, .offset = 4, .size = 4 } }, .dest = index, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = two, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .shl = .{ .lhs = index, .rhs = two } }, .dest = shifted, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = base, .rhs = shifted } }, .dest = inner, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 16 }, .dest = c16, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = inner, .rhs = c16 } }, .dest = result, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
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

/// Chained br_if on the same condition — exercises IR branch threading.
fn bodyChainedBrIfSameCond(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const cond = func.newVReg();
    const r_else = func.newVReg();
    const r_true = func.newVReg();
    const r_false = func.newVReg();

    const mid = func.newBlock() catch unreachable;
    const source_else = func.newBlock() catch unreachable;
    const inner_true = func.newBlock() catch unreachable;
    const inner_false = func.newBlock() catch unreachable;

    block.append(.{ .op = .memory_size, .dest = cond, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = mid, .else_block = source_else } } }) catch unreachable;

    func.getBlock(mid).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = inner_true, .else_block = inner_false } } }) catch unreachable;

    func.getBlock(source_else).append(.{ .op = .{ .iconst_32 = 0 }, .dest = r_else, .type = .i32 }) catch unreachable;
    func.getBlock(source_else).append(.{ .op = .{ .ret = r_else } }) catch unreachable;
    func.getBlock(inner_true).append(.{ .op = .{ .iconst_32 = 1 }, .dest = r_true, .type = .i32 }) catch unreachable;
    func.getBlock(inner_true).append(.{ .op = .{ .ret = r_true } }) catch unreachable;
    func.getBlock(inner_false).append(.{ .op = .{ .iconst_32 = 2 }, .dest = r_false, .type = .i32 }) catch unreachable;
    func.getBlock(inner_false).append(.{ .op = .{ .ret = r_false } }) catch unreachable;
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
    // `br_table.targets` must outlive this function: codegen reads it long
    // after `bodyBrTable` returns, and IR does not copy or free it (see the
    // "leaked by IR" note on the br_table compile test). A `&[_]…{…}` array
    // literal would dangle here, so allocate a slice that survives (leaked;
    // acceptable in a short-lived benchmark process).
    const targets = func.allocator.alloc(ir.BlockId, 4) catch unreachable;
    targets[0] = t0;
    targets[1] = t1;
    targets[2] = t2;
    targets[3] = t3;
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

fn bodyMemoryCopyFixed(func: *ir.IrFunction, block: *ir.BasicBlock, comptime len_value: i32) void {
    const dst = func.newVReg();
    const src = func.newVReg();
    const len = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = dst, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 0x1100 }, .dest = src, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = len_value }, .dest = len, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .memory_copy = .{ .dst = dst, .src = src, .len = len } } }) catch unreachable;
    block.append(.{ .op = .{ .ret = null } }) catch unreachable;
}

fn bodyMemoryCopy8(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    bodyMemoryCopyFixed(func, block, 8);
}

fn bodyMemoryCopy16(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    bodyMemoryCopyFixed(func, block, 16);
}

fn bodyMemoryCopy32(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    bodyMemoryCopyFixed(func, block, 32);
}

fn bodyMemoryCopy64(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    bodyMemoryCopyFixed(func, block, 64);
}

fn bodyMemoryFillFixed(func: *ir.IrFunction, block: *ir.BasicBlock, comptime len_value: i32) void {
    const dst = func.newVReg();
    const val = func.newVReg();
    const len = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = dst, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 0x5a }, .dest = val, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = len_value }, .dest = len, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .memory_fill = .{ .dst = dst, .val = val, .len = len } } }) catch unreachable;
    block.append(.{ .op = .{ .ret = null } }) catch unreachable;
}

fn bodyMemoryFill8(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    bodyMemoryFillFixed(func, block, 8);
}

fn bodyMemoryFill16(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    bodyMemoryFillFixed(func, block, 16);
}

fn bodyMemoryFill32(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    bodyMemoryFillFixed(func, block, 32);
}

fn bodyMemoryFill64(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    bodyMemoryFillFixed(func, block, 64);
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

// ── AArch64 scheduler benchmark bodies ───────────────────────────────

fn bodySchedLoadUse(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const loaded = func.newVReg();
    const a = func.newVReg();
    const b = func.newVReg();
    const mul = func.newVReg();
    const c = func.newVReg();
    const independent = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } }, .dest = loaded, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 13 }, .dest = a, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 17 }, .dest = b, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .mul = .{ .lhs = a, .rhs = b } }, .dest = mul, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 19 }, .dest = c, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = mul, .rhs = c } }, .dest = independent, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = loaded, .rhs = independent } }, .dest = result, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

fn bodySchedIndependentAlu(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const a0 = func.newVReg();
    const a1 = func.newVReg();
    const b0 = func.newVReg();
    const b1 = func.newVReg();
    const c0 = func.newVReg();
    const c1 = func.newVReg();
    const a2 = func.newVReg();
    const b2 = func.newVReg();
    const c2 = func.newVReg();
    const ab = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = a0, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = a1, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = b0, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 4 }, .dest = b1, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = c0, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 6 }, .dest = c1, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = a0, .rhs = a1 } }, .dest = a2, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .xor = .{ .lhs = b0, .rhs = b1 } }, .dest = b2, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .@"or" = .{ .lhs = c0, .rhs = c1 } }, .dest = c2, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = a2, .rhs = b2 } }, .dest = ab, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = ab, .rhs = c2 } }, .dest = result, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

fn bodySchedMulLatency(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const a = func.newVReg();
    const b = func.newVReg();
    const c = func.newVReg();
    const d = func.newVReg();
    const m0 = func.newVReg();
    const m1 = func.newVReg();
    const sum = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = a, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 11 }, .dest = b, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .mul = .{ .lhs = a, .rhs = b } }, .dest = m0, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 13 }, .dest = c, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 17 }, .dest = d, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .mul = .{ .lhs = c, .rhs = d } }, .dest = m1, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = m0, .rhs = m1 } }, .dest = sum, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .mul = .{ .lhs = sum, .rhs = b } }, .dest = result, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

fn bodySchedLoopCarriedLike(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const step = func.newVReg();
    const addr1 = func.newVReg();
    const addr2 = func.newVReg();
    const l0 = func.newVReg();
    const l1 = func.newVReg();
    const l2 = func.newVReg();
    const s0 = func.newVReg();
    const s1 = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 4 }, .dest = step, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = base, .rhs = step } }, .dest = addr1, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = addr1, .rhs = step } }, .dest = addr2, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } }, .dest = l0, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = addr1, .offset = 0, .size = 4 } }, .dest = l1, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = addr2, .offset = 0, .size = 4 } }, .dest = l2, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = l0, .rhs = l1 } }, .dest = s0, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = s0, .rhs = l2 } }, .dest = s1, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .mul = .{ .lhs = s1, .rhs = step } }, .dest = result, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

fn bodySchedStoreWithAlu(func: *ir.IrFunction, block: *ir.BasicBlock) void {
    const base = func.newVReg();
    const val = func.newVReg();
    const a = func.newVReg();
    const b = func.newVReg();
    const mul = func.newVReg();
    const load = func.newVReg();
    const result = func.newVReg();
    block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 23 }, .dest = val, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .store = .{ .base = base, .offset = 0, .size = 4, .val = val } }, .type = .void }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 29 }, .dest = a, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .iconst_32 = 31 }, .dest = b, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .mul = .{ .lhs = a, .rhs = b } }, .dest = mul, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .load = .{ .base = base, .offset = 4, .size = 4 } }, .dest = load, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .add = .{ .lhs = load, .rhs = mul } }, .dest = result, .type = .i32 }) catch unreachable;
    block.append(.{ .op = .{ .ret = result } }) catch unreachable;
}

fn runAarch64Bench(
    allocator: std.mem.Allocator,
    name: []const u8,
    buildBody: BuildBodyFn,
    enable_scheduler: bool,
) !BenchResult {
    var func = try buildTestFunc(allocator, buildBody);
    defer func.deinit();

    const options = aarch64_compile.CompileOptions{ .enable_scheduler = enable_scheduler };
    const sample_code = try aarch64_compile.compileFunctionWithOptions(&func, allocator, options);
    const code_size = sample_code.len;
    defer allocator.free(sample_code);

    for (0..100) |_| {
        const code = try aarch64_compile.compileFunctionWithOptions(&func, allocator, options);
        allocator.free(code);
    }

    const iterations: u64 = 5_000;
    const start = sampleTicks();
    for (0..iterations) |_| {
        const code = try aarch64_compile.compileFunctionWithOptions(&func, allocator, options);
        allocator.free(code);
    }
    const end = sampleTicks();

    return .{
        .name = name,
        .iterations = iterations,
        .total_ticks = end - start,
        .code_size = code_size,
    };
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
    _ = try passes.runPasses(&module, passes.defaultPassesForTarget(.x86_64), allocator);

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
    const start = sampleTicks();

    for (0..iterations) |_| {
        const r = try compile.compileFunctionRA(&module.functions.items[0], 0, allocator);
        allocator.free(r.code);
        allocator.free(r.call_patches);
    }

    const end = sampleTicks();

    return .{
        .name = name,
        .iterations = iterations,
        .total_ticks = end - start,
        .code_size = code_size,
    };
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const metric_label = sampleUnit();

    std.debug.print("\n", .{});
    std.debug.print("  x86-64 Codegen Benchmark (10,000 iterations each)\n", .{});
    std.debug.print("  ===================================================\n\n", .{});

    // ── Atomic operations (raw codegen, no passes) ──────────────────
    std.debug.print("  Atomic operations\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", metric_label, "code bytes" });
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
        std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{ result.name, result.ticksPerOp(), result.code_size });
    }

    // ── Arithmetic (raw codegen) ────────────────────────────────────
    std.debug.print("\n  Arithmetic\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", metric_label, "code bytes" });
    std.debug.print("  {s:-<34} {s:->12} {s:->10}\n", .{ "", "", "" });

    const arith_benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
        .{ .name = "add + sub chain", .body = &bodyAddSub },
        .{ .name = "mul + div_u", .body = &bodyMulDiv },
        .{ .name = "and + or + xor chain", .body = &bodyBitwiseChain },
        .{ .name = "shl i32 by const 3", .body = &bodyShlImm },
    };
    for (arith_benchmarks) |b| {
        const result = try runBench(allocator, b.name, b.body);
        std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{ result.name, result.ticksPerOp(), result.code_size });
    }

    // ── Branches + control flow (raw codegen) ──────────────────────
    std.debug.print("\n  Branches + control flow\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", metric_label, "code bytes" });
    std.debug.print("  {s:-<34} {s:->12} {s:->10}\n", .{ "", "", "" });

    const branch_benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
        .{ .name = "cmp + br_if diamond", .body = &bodyBrIf },
        .{ .name = "br_table (4 targets)", .body = &bodyBrTable },
        .{ .name = "select chain (cmov)", .body = &bodySelectChain },
    };
    for (branch_benchmarks) |b| {
        const result = try runBench(allocator, b.name, b.body);
        std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{ result.name, result.ticksPerOp(), result.code_size });
    }

    // ── Memory (raw codegen) ───────────────────────────────────────
    std.debug.print("\n  Memory\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", metric_label, "code bytes" });
    std.debug.print("  {s:-<34} {s:->12} {s:->10}\n", .{ "", "", "" });

    const memory_benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
        .{ .name = "load + add + store", .body = &bodyLoadStore },
        .{ .name = "4× load + sum + store", .body = &bodyLoadStoreMulti },
        .{ .name = "memory.copy fixed 8", .body = &bodyMemoryCopy8 },
        .{ .name = "memory.copy fixed 16", .body = &bodyMemoryCopy16 },
        .{ .name = "memory.copy fixed 32", .body = &bodyMemoryCopy32 },
        .{ .name = "memory.copy fixed 64", .body = &bodyMemoryCopy64 },
        .{ .name = "memory.fill fixed 8", .body = &bodyMemoryFill8 },
        .{ .name = "memory.fill fixed 16", .body = &bodyMemoryFill16 },
        .{ .name = "memory.fill fixed 32", .body = &bodyMemoryFill32 },
        .{ .name = "memory.fill fixed 64", .body = &bodyMemoryFill64 },
    };
    for (memory_benchmarks) |b| {
        const result = try runBench(allocator, b.name, b.body);
        std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{ result.name, result.ticksPerOp(), result.code_size });
    }

    // ── Calls (raw codegen) ────────────────────────────────────────
    std.debug.print("\n  Calls\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", metric_label, "code bytes" });
    std.debug.print("  {s:-<34} {s:->12} {s:->10}\n", .{ "", "", "" });

    const call_benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
        .{ .name = "call + ret", .body = &bodyCallRet },
    };
    for (call_benchmarks) |b| {
        const result = try runBench(allocator, b.name, b.body);
        std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{ result.name, result.ticksPerOp(), result.code_size });
    }

    // ── Float (raw codegen) ────────────────────────────────────────
    std.debug.print("\n  Float\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", metric_label, "code bytes" });
    std.debug.print("  {s:-<34} {s:->12} {s:->10}\n", .{ "", "", "" });

    const float_benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
        .{ .name = "f64 add + mul", .body = &bodyFloatArith },
        .{ .name = "i32→f64 convert + add", .body = &bodyIntToFloat },
    };
    for (float_benchmarks) |b| {
        const result = try runBench(allocator, b.name, b.body);
        std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{ result.name, result.ticksPerOp(), result.code_size });
    }

    // ── Optimization passes (codegen after default_passes) ─────────
    std.debug.print("\n  Optimization passes (codegen after default_passes)\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", metric_label, "code bytes" });
    std.debug.print("  {s:-<34} {s:->12} {s:->10}\n", .{ "", "", "" });

    const pass_benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
        .{ .name = "dead intermediates (DCE)", .body = &bodyDeadIntermediates },
        .{ .name = "mul(x, 8) → shl(x, 3)", .body = &bodyMulByPow2 },
        .{ .name = "3× load same base (hoisted)", .body = &bodyConsecutiveLoads },
        .{ .name = "div_u by const 7 (magic mul)", .body = &bodyDivByConst },
        .{ .name = "add base,const → load offset", .body = &bodyAddIntoLoad },
        .{ .name = "base+index*4+16 → lea (#543)", .body = &bodyLeaCompound },
        .{ .name = "chained br_if same cond", .body = &bodyChainedBrIfSameCond },
        .{ .name = "4× load + sum (bounds elide)", .body = &bodyLoadStoreMulti },
        .{ .name = "reg pressure (12 live vals)", .body = &bodyRegPressure },
    };
    for (pass_benchmarks) |b| {
        const result = try runBenchWithPasses(allocator, b.name, b.body);
        std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{ result.name, result.ticksPerOp(), result.code_size });
    }

    // ── AArch64 local scheduler (raw codegen, scheduler off/on) ─────
    std.debug.print("\n  AArch64 scheduler microbenchmarks ({s})\n", .{metric_label});
    std.debug.print("  {s:<34} {s:>12} {s:>12} {s:>10} {s:>10}\n", .{
        "operation",
        "sched off",
        "sched on",
        "off bytes",
        "on bytes",
    });
    std.debug.print("  {s:-<34} {s:->12} {s:->12} {s:->10} {s:->10}\n", .{ "", "", "", "", "" });

    const scheduler_benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
        .{ .name = "load/use + independent mul", .body = &bodySchedLoadUse },
        .{ .name = "independent ALU chains", .body = &bodySchedIndependentAlu },
        .{ .name = "independent mul latency", .body = &bodySchedMulLatency },
        .{ .name = "loop-carried-like loads", .body = &bodySchedLoopCarriedLike },
        .{ .name = "store/load + independent mul", .body = &bodySchedStoreWithAlu },
    };
    for (scheduler_benchmarks) |b| {
        const off = try runAarch64Bench(allocator, b.name, b.body, false);
        const on = try runAarch64Bench(allocator, b.name, b.body, true);
        std.debug.print("  {s:<34} {d:>12} {d:>12} {d:>10} {d:>10}\n", .{
            b.name,
            off.ticksPerOp(),
            on.ticksPerOp(),
            off.code_size,
            on.code_size,
        });
    }

    std.debug.print("\n", .{});
}
