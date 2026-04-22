//! x86-64 Atomic + DCE Codegen Microbenchmark
//!
//! Measures compilation throughput and generated code size for each
//! atomic instruction type and demonstrates dead code elimination.
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

    const sample_code = try compile.compileFunction(&func, allocator);
    const code_size = sample_code.len;
    defer allocator.free(sample_code);

    // Warmup
    for (0..200) |_| {
        const c = try compile.compileFunction(&func, allocator);
        allocator.free(c);
    }

    // Timed iterations (fixed count for consistency)
    const iterations: u64 = 10_000;
    const start = rdtsc();

    for (0..iterations) |_| {
        const c = try compile.compileFunction(&func, allocator);
        allocator.free(c);
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

    const sample_code = try compile.compileFunction(&module.functions.items[0], allocator);
    const code_size = sample_code.len;
    defer allocator.free(sample_code);

    // Warmup
    for (0..200) |_| {
        const c = try compile.compileFunction(&module.functions.items[0], allocator);
        allocator.free(c);
    }

    // Timed iterations
    const iterations: u64 = 10_000;
    const start = rdtsc();

    for (0..iterations) |_| {
        const c = try compile.compileFunction(&module.functions.items[0], allocator);
        allocator.free(c);
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
    std.debug.print("  x86-64 Atomic Codegen Benchmark (10,000 iterations each)\n", .{});
    std.debug.print("  =========================================================\n\n", .{});
    std.debug.print("  {s:<28} {s:>12} {s:>10}\n", .{ "operation", "cycles/op", "code bytes" });
    std.debug.print("  {s:-<28} {s:->12} {s:->10}\n", .{ "", "", "" });

    const benchmarks = [_]struct { name: []const u8, body: BuildBodyFn }{
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

    for (benchmarks) |b| {
        const result = try runBench(allocator, b.name, b.body);
        std.debug.print("  {s:<28} {d:>12} {d:>10}\n", .{
            result.name,
            result.cyclesPerOp(),
            result.code_size,
        });
    }

    std.debug.print("\n", .{});
    std.debug.print("  DCE benchmark: codegen after default passes (10,000 iterations each)\n", .{});
    std.debug.print("  -----------------------------------------------------------------------\n\n", .{});
    std.debug.print("  {s:<34} {s:>12} {s:>10}\n", .{ "operation", "cycles/op", "code bytes" });
    std.debug.print("  {s:-<34} {s:->12} {s:->10}\n", .{ "", "", "" });

    const dce_result = try runBenchWithPasses(allocator, "dead intermediates (DCE)", &bodyDeadIntermediates);
    std.debug.print("  {s:<34} {d:>12} {d:>10}\n", .{
        dce_result.name,
        dce_result.cyclesPerOp(),
        dce_result.code_size,
    });

    std.debug.print("\n", .{});
}
