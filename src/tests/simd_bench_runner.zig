//! SIMD benchmark/status runner for the Zig AOT backend.
//!
//! The runner builds small in-memory Wasm modules that use SIMD internally and
//! expose scalar `() -> i32` wrapper functions.  That keeps the benchmark usable
//! before the AOT runtime grows direct v128 parameter/result support.

const std = @import("std");
const builtin = @import("builtin");

const wamr = @import("wamr");
const loader_mod = wamr.loader;
const instance_mod = wamr.instance;
const interp = wamr.interp;
const ExecEnv = wamr.exec_env.ExecEnv;
const aot_runtime = wamr.aot_runtime;
const aot_harness = @import("aot_harness.zig");

const Allocator = std.mem.Allocator;

const BenchBuilder = *const fn (Allocator) anyerror![]u8;

const BenchCase = struct {
    name: []const u8,
    simd: bool,
    build: BenchBuilder,
};

const RunResult = struct {
    result: i32,
    run_ns: u64,
};

const BenchOptions = struct {
    iterations: u32 = 10_000,
    run_wasmtime: bool = false,
    wasmtime_path: []const u8 = "wasmtime",
    wasmtime_iterations: ?u32 = null,
};

const cases = [_]BenchCase{
    .{
        .name = "scalar_i32_add",
        .simd = false,
        .build = buildScalarAddModule,
    },
    .{
        .name = "simd_i32x4_add_lane0",
        .simd = true,
        .build = buildSimdI32x4AddLane0Module,
    },
    .{
        .name = "simd_v128_xor_lane0",
        .simd = true,
        .build = buildSimdV128XorLane0Module,
    },
    .{
        .name = "simd_i32x4_eq_lane0",
        .simd = true,
        .build = buildSimdI32x4EqLane0Module,
    },
    .{
        .name = "simd_i32x4_mul_lane0",
        .simd = true,
        .build = buildSimdI32x4MulLane0Module,
    },
    .{
        .name = "simd_i32x4_min_s_lane0",
        .simd = true,
        .build = buildSimdI32x4MinSLane0Module,
    },
    .{
        .name = "simd_i32x4_min_u_lane0",
        .simd = true,
        .build = buildSimdI32x4MinULane0Module,
    },
    .{
        .name = "simd_i32x4_max_s_lane0",
        .simd = true,
        .build = buildSimdI32x4MaxSLane0Module,
    },
    .{
        .name = "simd_i32x4_max_u_lane0",
        .simd = true,
        .build = buildSimdI32x4MaxULane0Module,
    },
    .{
        .name = "simd_i32x4_abs_lane0",
        .simd = true,
        .build = buildSimdI32x4AbsLane0Module,
    },
    .{
        .name = "simd_i32x4_neg_lane0",
        .simd = true,
        .build = buildSimdI32x4NegLane0Module,
    },
    .{
        .name = "simd_i32x4_ne_lane0",
        .simd = true,
        .build = buildSimdI32x4NeLane0Module,
    },
    .{
        .name = "simd_i32x4_gt_s_lane0",
        .simd = true,
        .build = buildSimdI32x4GtSLane0Module,
    },
    .{
        .name = "simd_i32x4_gt_u_lane0",
        .simd = true,
        .build = buildSimdI32x4GtULane0Module,
    },
    .{
        .name = "simd_i32x4_shl_lane0",
        .simd = true,
        .build = buildSimdI32x4ShlLane0Module,
    },
    .{
        .name = "simd_i32x4_shr_s_lane0",
        .simd = true,
        .build = buildSimdI32x4ShrSLane0Module,
    },
    .{
        .name = "simd_i32x4_shr_u_lane0",
        .simd = true,
        .build = buildSimdI32x4ShrULane0Module,
    },
    .{
        .name = "simd_i32x4_splat_lane0",
        .simd = true,
        .build = buildSimdI32x4SplatLane0Module,
    },
    .{
        .name = "simd_i32x4_replace_lane2",
        .simd = true,
        .build = buildSimdI32x4ReplaceLane2Module,
    },
    .{
        .name = "simd_i16x8_add_lane0",
        .simd = true,
        .build = buildSimdI16x8AddLane0Module,
    },
    .{
        .name = "simd_i16x8_mul_lane0",
        .simd = true,
        .build = buildSimdI16x8MulLane0Module,
    },
    .{
        .name = "simd_i16x8_q15mulr_sat_s_lane0",
        .simd = true,
        .build = buildSimdI16x8Q15MulrSatSLane0Module,
    },
    .{
        .name = "simd_i16x8_abs_lane0",
        .simd = true,
        .build = buildSimdI16x8AbsLane0Module,
    },
    .{
        .name = "simd_i16x8_neg_lane0",
        .simd = true,
        .build = buildSimdI16x8NegLane0Module,
    },
    .{
        .name = "simd_i16x8_add_sat_s_lane0",
        .simd = true,
        .build = buildSimdI16x8AddSatSLane0Module,
    },
    .{
        .name = "simd_i16x8_add_sat_u_lane0",
        .simd = true,
        .build = buildSimdI16x8AddSatULane0Module,
    },
    .{
        .name = "simd_i16x8_sub_sat_s_lane0",
        .simd = true,
        .build = buildSimdI16x8SubSatSLane0Module,
    },
    .{
        .name = "simd_i16x8_sub_sat_u_lane0",
        .simd = true,
        .build = buildSimdI16x8SubSatULane0Module,
    },
    .{
        .name = "simd_i16x8_min_s_lane0",
        .simd = true,
        .build = buildSimdI16x8MinSLane0Module,
    },
    .{
        .name = "simd_i16x8_min_u_lane0",
        .simd = true,
        .build = buildSimdI16x8MinULane0Module,
    },
    .{
        .name = "simd_i16x8_max_s_lane0",
        .simd = true,
        .build = buildSimdI16x8MaxSLane0Module,
    },
    .{
        .name = "simd_i16x8_max_u_lane0",
        .simd = true,
        .build = buildSimdI16x8MaxULane0Module,
    },
    .{
        .name = "simd_i16x8_avgr_u_lane0",
        .simd = true,
        .build = buildSimdI16x8AvgrULane0Module,
    },
    .{
        .name = "simd_i16x8_gt_s_lane0",
        .simd = true,
        .build = buildSimdI16x8GtSLane0Module,
    },
    .{
        .name = "simd_i16x8_gt_u_lane0",
        .simd = true,
        .build = buildSimdI16x8GtULane0Module,
    },
    .{
        .name = "simd_i16x8_splat_lane0",
        .simd = true,
        .build = buildSimdI16x8SplatLane0Module,
    },
    .{
        .name = "simd_i16x8_replace_lane5",
        .simd = true,
        .build = buildSimdI16x8ReplaceLane5Module,
    },
    .{
        .name = "simd_i16x8_shl_lane0",
        .simd = true,
        .build = buildSimdI16x8ShlLane0Module,
    },
    .{
        .name = "simd_i16x8_shr_s_lane0",
        .simd = true,
        .build = buildSimdI16x8ShrSLane0Module,
    },
    .{
        .name = "simd_i16x8_shr_u_lane0",
        .simd = true,
        .build = buildSimdI16x8ShrULane0Module,
    },
    .{
        .name = "simd_i8x16_add_lane0",
        .simd = true,
        .build = buildSimdI8x16AddLane0Module,
    },
    .{
        .name = "simd_i8x16_sub_lane0",
        .simd = true,
        .build = buildSimdI8x16SubLane0Module,
    },
    .{
        .name = "simd_i8x16_abs_lane0",
        .simd = true,
        .build = buildSimdI8x16AbsLane0Module,
    },
    .{
        .name = "simd_i8x16_neg_lane0",
        .simd = true,
        .build = buildSimdI8x16NegLane0Module,
    },
    .{
        .name = "simd_i8x16_add_sat_s_lane0",
        .simd = true,
        .build = buildSimdI8x16AddSatSLane0Module,
    },
    .{
        .name = "simd_i8x16_add_sat_u_lane0",
        .simd = true,
        .build = buildSimdI8x16AddSatULane0Module,
    },
    .{
        .name = "simd_i8x16_sub_sat_s_lane0",
        .simd = true,
        .build = buildSimdI8x16SubSatSLane0Module,
    },
    .{
        .name = "simd_i8x16_sub_sat_u_lane0",
        .simd = true,
        .build = buildSimdI8x16SubSatULane0Module,
    },
    .{
        .name = "simd_i8x16_min_s_lane0",
        .simd = true,
        .build = buildSimdI8x16MinSLane0Module,
    },
    .{
        .name = "simd_i8x16_min_u_lane0",
        .simd = true,
        .build = buildSimdI8x16MinULane0Module,
    },
    .{
        .name = "simd_i8x16_max_s_lane0",
        .simd = true,
        .build = buildSimdI8x16MaxSLane0Module,
    },
    .{
        .name = "simd_i8x16_max_u_lane0",
        .simd = true,
        .build = buildSimdI8x16MaxULane0Module,
    },
    .{
        .name = "simd_i8x16_avgr_u_lane0",
        .simd = true,
        .build = buildSimdI8x16AvgrULane0Module,
    },
    .{
        .name = "simd_i8x16_eq_lane0",
        .simd = true,
        .build = buildSimdI8x16EqLane0Module,
    },
    .{
        .name = "simd_i8x16_gt_s_lane0",
        .simd = true,
        .build = buildSimdI8x16GtSLane0Module,
    },
    .{
        .name = "simd_i8x16_gt_u_lane0",
        .simd = true,
        .build = buildSimdI8x16GtULane0Module,
    },
    .{
        .name = "simd_i8x16_splat_lane0",
        .simd = true,
        .build = buildSimdI8x16SplatLane0Module,
    },
    .{
        .name = "simd_i8x16_replace_lane13",
        .simd = true,
        .build = buildSimdI8x16ReplaceLane13Module,
    },
    .{
        .name = "simd_i8x16_shl_lane0",
        .simd = true,
        .build = buildSimdI8x16ShlLane0Module,
    },
    .{
        .name = "simd_i8x16_shr_s_lane0",
        .simd = true,
        .build = buildSimdI8x16ShrSLane0Module,
    },
    .{
        .name = "simd_i8x16_shr_u_lane0",
        .simd = true,
        .build = buildSimdI8x16ShrULane0Module,
    },
    .{
        .name = "simd_i64x2_add_lane0",
        .simd = true,
        .build = buildSimdI64x2AddLane0Module,
    },
    .{
        .name = "simd_i64x2_sub_lane0",
        .simd = true,
        .build = buildSimdI64x2SubLane0Module,
    },
    .{
        .name = "simd_i64x2_abs_lane0",
        .simd = true,
        .build = buildSimdI64x2AbsLane0Module,
    },
    .{
        .name = "simd_i64x2_neg_lane0",
        .simd = true,
        .build = buildSimdI64x2NegLane0Module,
    },
    .{
        .name = "simd_i64x2_eq_lane0",
        .simd = true,
        .build = buildSimdI64x2EqLane0Module,
    },
    .{
        .name = "simd_i64x2_gt_s_lane0",
        .simd = true,
        .build = buildSimdI64x2GtSLane0Module,
    },
    .{
        .name = "simd_i64x2_splat_lane0",
        .simd = true,
        .build = buildSimdI64x2SplatLane0Module,
    },
    .{
        .name = "simd_i64x2_replace_lane1",
        .simd = true,
        .build = buildSimdI64x2ReplaceLane1Module,
    },
    .{
        .name = "simd_i64x2_shl_lane0",
        .simd = true,
        .build = buildSimdI64x2ShlLane0Module,
    },
    .{
        .name = "simd_i64x2_shr_s_lane0",
        .simd = true,
        .build = buildSimdI64x2ShrSLane0Module,
    },
    .{
        .name = "simd_i64x2_shr_u_lane0",
        .simd = true,
        .build = buildSimdI64x2ShrULane0Module,
    },
    .{
        .name = "simd_v128_load_store_lane0",
        .simd = true,
        .build = buildSimdLoadStoreLane0Module,
    },
    .{
        .name = "scalar_i32_mem_add_4k_loop",
        .simd = false,
        .build = buildScalarI32MemoryAdd4kLoopModule,
    },
    .{
        .name = "simd_i32x4_mem_add_4k_loop",
        .simd = true,
        .build = buildSimdI32x4MemoryAdd4kLoopModule,
    },
    .{
        .name = "simd_i32x4_mem_sum8_4k_loop",
        .simd = true,
        .build = buildSimdI32x4MemorySum8_4kLoopModule,
    },
    .{
        .name = "simd_i32x4_shift_mix_4k_loop",
        .simd = true,
        .build = buildSimdI32x4ShiftMix4kLoopModule,
    },
    .{
        .name = "simd_i32x4_minmax_4k_loop",
        .simd = true,
        .build = buildSimdI32x4MinMax4kLoopModule,
    },
    .{
        .name = "simd_i16x8_mem_add_4k_loop",
        .simd = true,
        .build = buildSimdI16x8MemoryAdd4kLoopModule,
    },
    .{
        .name = "simd_i16x8_shift_mix_4k_loop",
        .simd = true,
        .build = buildSimdI16x8ShiftMix4kLoopModule,
    },
    .{
        .name = "simd_i16x8_arith_extra_4k_loop",
        .simd = true,
        .build = buildSimdI16x8ArithExtra4kLoopModule,
    },
    .{
        .name = "simd_i8x16_mem_add_4k_loop",
        .simd = true,
        .build = buildSimdI8x16MemoryAdd4kLoopModule,
    },
    .{
        .name = "simd_i8x16_shift_mix_4k_loop",
        .simd = true,
        .build = buildSimdI8x16ShiftMix4kLoopModule,
    },
    .{
        .name = "simd_i8x16_arith_extra_4k_loop",
        .simd = true,
        .build = buildSimdI8x16ArithExtra4kLoopModule,
    },
    .{
        .name = "simd_i64x2_mem_add_4k_loop",
        .simd = true,
        .build = buildSimdI64x2MemoryAdd4kLoopModule,
    },
    .{
        .name = "simd_i64x2_shift_mix_4k_loop",
        .simd = true,
        .build = buildSimdI64x2ShiftMix4kLoopModule,
    },
    .{
        .name = "simd_int_absneg_4k_loop",
        .simd = true,
        .build = buildSimdIntAbsNeg4kLoopModule,
    },
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parseOptions(args) catch |err| {
        std.debug.print("simd-bench-runner: invalid arguments: {s}\n", .{@errorName(err)});
        usage();
        std.process.exit(2);
    };

    if (!aot_harness.can_exec_aot) {
        std.debug.print(
            "simd-bench-runner: AOT execution is not supported on this target ({s}); interpreter rows will still run.\n",
            .{@tagName(builtin.cpu.arch)},
        );
    }

    for (cases) |case| {
        try runCase(allocator, init.io, case, options);
    }
}

fn usage() void {
    std.debug.print(
        \\usage: simd-bench-runner [--iterations N] [--wasmtime] [--wasmtime-path PATH] [--wasmtime-iterations N]
        \\
        \\Runs embedded SIMD microbench modules through the interpreter and,
        \\when supported by the host CPU, through the in-memory AOT pipeline.
        \\With --wasmtime, also invokes the Wasmtime CLI for a small external
        \\baseline. Wasmtime timings include CLI startup and compilation.
        \\Rows are tab-separated and prefixed with "bench".
        \\
    , .{});
}

fn parseOptions(args: []const []const u8) !BenchOptions {
    var options = BenchOptions{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--help") or std.mem.eql(u8, args[i], "-h")) {
            usage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, args[i], "--iterations")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            options.iterations = try std.fmt.parseUnsigned(u32, args[i], 10);
            if (options.iterations == 0) return error.InvalidIterationCount;
        } else if (std.mem.eql(u8, args[i], "--wasmtime")) {
            options.run_wasmtime = true;
        } else if (std.mem.eql(u8, args[i], "--wasmtime-path")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            options.wasmtime_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--wasmtime-iterations")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            options.wasmtime_iterations = try std.fmt.parseUnsigned(u32, args[i], 10);
            if (options.wasmtime_iterations.? == 0) return error.InvalidIterationCount;
        } else {
            return error.UnknownArgument;
        }
    }
    return options;
}

fn runCase(allocator: Allocator, io: std.Io, case: BenchCase, options: BenchOptions) !void {
    const wasm = try case.build(allocator);
    defer allocator.free(wasm);

    const interp_result = runInterpMany(allocator, wasm, "run", options.iterations) catch |err| {
        emitRow(case.name, "interp", "trap", null, null, null, options.iterations, wasm.len);
        std.debug.print("simd-bench-runner: interpreter failed for {s}: {s}\n", .{ case.name, @errorName(err) });
        return;
    };
    emitRow(case.name, "interp", "ok", interp_result.result, 0, interp_result.run_ns, options.iterations, wasm.len);

    if (options.run_wasmtime) {
        const wasmtime_iterations = options.wasmtime_iterations orelse @min(options.iterations, 10);
        const wasmtime_result = runWasmtimeMany(
            allocator,
            io,
            wasm,
            "run",
            case.name,
            wasmtime_iterations,
            options.wasmtime_path,
        ) catch |err| {
            const status = if (err == error.FileNotFound) "unsupported" else "trap";
            emitRow(case.name, "wasmtime", status, null, null, null, wasmtime_iterations, wasm.len);
            std.debug.print("simd-bench-runner: Wasmtime failed for {s}: {s}\n", .{ case.name, @errorName(err) });
            return;
        };
        const status = if (wasmtime_result.result == interp_result.result) "ok" else "mismatch";
        emitRow(case.name, "wasmtime", status, wasmtime_result.result, null, wasmtime_result.run_ns, wasmtime_iterations, wasm.len);
    }

    if (!aot_harness.can_exec_aot) {
        emitRow(case.name, "aot", "unsupported", null, null, null, options.iterations, null);
        return;
    }

    const compile_start = nowNs();
    const h = aot_harness.Harness.initWithOptions(
        allocator,
        wasm,
        null,
        .{ .invoke_start = false },
    ) catch |err| {
        const status = if (case.simd and err == error.CompileFailed) "unsupported" else "compile_failed";
        emitRow(case.name, "aot", status, null, elapsedSince(compile_start), null, options.iterations, null);
        return;
    };
    const compile_ns = elapsedSince(compile_start);
    defer h.deinit();

    const aot_result = runAotMany(h, "run", options.iterations) catch |err| {
        emitRow(case.name, "aot", "trap", null, compile_ns, null, options.iterations, h.aot_bin.len);
        std.debug.print("simd-bench-runner: AOT failed for {s}: {s}\n", .{ case.name, @errorName(err) });
        return;
    };
    const status = if (aot_result.result == interp_result.result) "ok" else "mismatch";
    emitRow(case.name, "aot", status, aot_result.result, compile_ns, aot_result.run_ns, options.iterations, h.aot_bin.len);
}

fn emitRow(
    case_name: []const u8,
    engine: []const u8,
    status: []const u8,
    result: ?i32,
    compile_ns: ?u64,
    run_ns: ?u64,
    iterations: u32,
    code_size: ?usize,
) void {
    std.debug.print("bench\t{s}\t{s}\t{s}\t", .{ case_name, engine, status });
    if (result) |v| {
        std.debug.print("{d}", .{v});
    } else {
        std.debug.print("-", .{});
    }
    std.debug.print("\t", .{});
    if (compile_ns) |v| {
        std.debug.print("{d}", .{v});
    } else {
        std.debug.print("-", .{});
    }
    std.debug.print("\t", .{});
    if (run_ns) |v| {
        std.debug.print("{d}", .{v});
    } else {
        std.debug.print("-", .{});
    }
    std.debug.print("\t{d}\t", .{iterations});
    if (code_size) |v| {
        std.debug.print("{d}", .{v});
    } else {
        std.debug.print("-", .{});
    }
    std.debug.print("\n", .{});
}

fn elapsedSince(start: u64) u64 {
    return nowNs() - start;
}

fn nowNs() u64 {
    return wamr.platform.timeGetBootUs() * std.time.ns_per_us;
}

fn runInterpMany(allocator: Allocator, wasm: []const u8, name: []const u8, iterations: u32) !RunResult {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const module = try loader_mod.load(wasm, arena.allocator());

    const inst = try instance_mod.instantiate(&module, allocator);
    defer instance_mod.destroy(inst);

    const exp = inst.module.findExport(name, .function) orelse return error.FunctionNotFound;

    var env = try ExecEnv.create(inst, 4096, allocator);
    defer env.destroy();

    var last: i32 = 0;
    const start = nowNs();
    for (0..iterations) |_| {
        try interp.executeFunction(env, exp.index);
        last = try env.popI32();
    }
    return .{
        .result = last,
        .run_ns = elapsedSince(start),
    };
}

fn runAotMany(h: *aot_harness.Harness, name: []const u8, iterations: u32) !RunResult {
    const func_idx = h.findFuncExport(name) orelse return error.FunctionNotFound;

    var results_buf: [1]aot_runtime.ScalarResult = undefined;
    var last: i32 = 0;
    const start = nowNs();
    for (0..iterations) |_| {
        const results = try h.callScalar(func_idx, &.{}, &results_buf);
        if (results.len != 1) return error.UnsupportedSignature;
        last = switch (results[0]) {
            .i32 => |v| v,
            else => return error.InvalidResultType,
        };
    }
    return .{
        .result = last,
        .run_ns = elapsedSince(start),
    };
}

fn runWasmtimeMany(
    allocator: Allocator,
    io: std.Io,
    wasm: []const u8,
    name: []const u8,
    case_name: []const u8,
    iterations: u32,
    wasmtime_path: []const u8,
) !RunResult {
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, ".zig-cache/simd-bench-wasmtime");
    const wasm_path = try std.fmt.allocPrint(allocator, ".zig-cache/simd-bench-wasmtime/{s}.wasm", .{case_name});
    defer allocator.free(wasm_path);
    defer cwd.deleteFile(io, wasm_path) catch {};

    try cwd.writeFile(io, .{ .sub_path = wasm_path, .data = wasm });

    var last: i32 = 0;
    const start = nowNs();
    var i: u32 = 0;
    while (i < iterations) : (i += 1) {
        const argv = [_][]const u8{ wasmtime_path, "--invoke", name, wasm_path };
        const result = try std.process.run(allocator, io, .{
            .argv = &argv,
            .stdout_limit = .limited(4096),
            .stderr_limit = .limited(4096),
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);

        switch (result.term) {
            .exited => |code| if (code != 0) return error.WasmtimeFailed,
            else => return error.WasmtimeFailed,
        }
        const trimmed = std.mem.trim(u8, result.stdout, " \t\r\n");
        last = try std.fmt.parseInt(i32, trimmed, 10);
    }
    return .{
        .result = last,
        .run_ns = elapsedSince(start),
    };
}

fn buildScalarAddModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try instr.append(allocator, 0x41); // i32.const
    try encodeSLEB128(&instr, allocator, 123);
    try instr.append(allocator, 0x41); // i32.const
    try encodeSLEB128(&instr, allocator, 456);
    try instr.append(allocator, 0x6A); // i32.add

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4AddLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI32x4(&instr, allocator, .{ 1, 2, 3, 4 });
    try appendV128ConstI32x4(&instr, allocator, .{ 5, 6, 7, 8 });
    try appendSimdOpcode(&instr, allocator, 0xAE); // i32x4.add
    try appendI32x4ExtractLane(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdV128XorLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI32x4(&instr, allocator, .{ 0x1357_9BDF, 2, 3, 4 });
    try appendV128ConstI32x4(&instr, allocator, .{ 0x0102_0304, 6, 7, 8 });
    try appendSimdOpcode(&instr, allocator, 0x51); // v128.xor
    try appendI32x4ExtractLane(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4EqLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI32x4(&instr, allocator, .{ 42, 2, 3, 4 });
    try appendV128ConstI32x4(&instr, allocator, .{ 42, 0, 3, 5 });
    try appendSimdOpcode(&instr, allocator, 0x37); // i32x4.eq
    try appendI32x4ExtractLane(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4MulLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI32x4(&instr, allocator, .{ 50_000, -7, 3, 4 });
    try appendV128ConstI32x4(&instr, allocator, .{ 50_000, 6, 7, 8 });
    try appendSimdOpcode(&instr, allocator, 0xB5); // i32x4.mul
    try appendI32x4ExtractLane(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4MinSLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI32x4(&instr, allocator, .{ std.math.minInt(i32), -7, 500, -1 });
    try appendV128ConstI32x4(&instr, allocator, .{ std.math.maxInt(i32), 6, -500, 1 });
    try appendSimdOpcode(&instr, allocator, 0xB6); // i32x4.min_s
    try appendI32x4ExtractLane(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4MinULane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI32x4(&instr, allocator, .{ std.math.minInt(i32), -7, 500, -1 });
    try appendV128ConstI32x4(&instr, allocator, .{ std.math.maxInt(i32), 6, -500, 1 });
    try appendSimdOpcode(&instr, allocator, 0xB7); // i32x4.min_u
    try appendI32x4ExtractLane(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4MaxSLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI32x4(&instr, allocator, .{ std.math.minInt(i32), -7, 500, -1 });
    try appendV128ConstI32x4(&instr, allocator, .{ std.math.maxInt(i32), 6, -500, 1 });
    try appendSimdOpcode(&instr, allocator, 0xB8); // i32x4.max_s
    try appendI32x4ExtractLane(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4MaxULane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI32x4(&instr, allocator, .{ std.math.minInt(i32), -7, 500, -1 });
    try appendV128ConstI32x4(&instr, allocator, .{ std.math.maxInt(i32), 6, -500, 1 });
    try appendSimdOpcode(&instr, allocator, 0xB9); // i32x4.max_u
    try appendI32x4ExtractLane(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4AbsLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI32x4(&instr, allocator, .{ -123_456_789, std.math.minInt(i32), -7, 42 });
    try appendSimdOpcode(&instr, allocator, 0xA0); // i32x4.abs
    try appendI32x4ExtractLane(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4NegLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI32x4(&instr, allocator, .{ 123_456_789, std.math.minInt(i32), -7, 42 });
    try appendSimdOpcode(&instr, allocator, 0xA1); // i32x4.neg
    try appendI32x4ExtractLane(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4NeLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI32x4(&instr, allocator, .{ 42, 2, 3, 4 });
    try appendV128ConstI32x4(&instr, allocator, .{ 7, 2, 0, 4 });
    try appendSimdOpcode(&instr, allocator, 0x38); // i32x4.ne
    try appendI32x4ExtractLane(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4GtSLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI32x4(&instr, allocator, .{ -1, 10, -5, 4 });
    try appendV128ConstI32x4(&instr, allocator, .{ 1, 9, -6, 4 });
    try appendSimdOpcode(&instr, allocator, 0x3B); // i32x4.gt_s
    try appendI32x4ExtractLane(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4GtULane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI32x4(&instr, allocator, .{ -1, 10, -5, 4 });
    try appendV128ConstI32x4(&instr, allocator, .{ 1, 9, -6, 4 });
    try appendSimdOpcode(&instr, allocator, 0x3C); // i32x4.gt_u
    try appendI32x4ExtractLane(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4ShlLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI32x4ShiftLane0Module(allocator, 0xAB, .{ 3, 7, 11, 13 });
}

fn buildSimdI32x4ShrSLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI32x4ShiftLane0Module(allocator, 0xAC, .{ -8, -1024, 1024, 7 });
}

fn buildSimdI32x4ShrULane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI32x4ShiftLane0Module(allocator, 0xAD, .{ -8, -1024, 1024, 7 });
}

fn buildSimdI32x4ShiftLane0Module(allocator: Allocator, opcode: u32, lanes: [4]i32) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    const counts = [_]i64{ 0, 1, 31, 32, 33 };
    for (counts, 0..) |count, idx| {
        try appendV128ConstI32x4(&instr, allocator, lanes);
        try appendI32Const(&instr, allocator, count);
        try appendSimdOpcode(&instr, allocator, opcode);
        try appendI32x4ExtractLane(&instr, allocator, 0);
        if (idx != 0) try appendI32Add(&instr, allocator);
    }

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4SplatLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try instr.append(allocator, 0x41); // i32.const
    try encodeSLEB128(&instr, allocator, 77);
    try appendI32x4Splat(&instr, allocator);
    try appendI32x4ExtractLane(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI32x4ReplaceLane2Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI32x4(&instr, allocator, .{ 1, 2, 3, 4 });
    try instr.append(allocator, 0x41); // i32.const
    try encodeSLEB128(&instr, allocator, 99);
    try appendI32x4ReplaceLane(&instr, allocator, 2);
    try appendI32x4ExtractLane(&instr, allocator, 2);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI16x8AddLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI16x8(&instr, allocator, .{ 1200, 2, 3, 4, 5, 6, 7, 8 });
    try appendV128ConstI16x8(&instr, allocator, .{ 3400, 6, 7, 8, 9, 10, 11, 12 });
    try appendSimdOpcode(&instr, allocator, 0x8E); // i16x8.add
    try appendI16x8ExtractLaneU(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI16x8MulLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI16x8(&instr, allocator, .{ 300, 2, 3, 4, 5, 6, 7, 8 });
    try appendV128ConstI16x8(&instr, allocator, .{ 300, 6, 7, 8, 9, 10, 11, 12 });
    try appendSimdOpcode(&instr, allocator, 0x95); // i16x8.mul
    try appendI16x8ExtractLaneU(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI16x8Q15MulrSatSLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI16x8BinaryLane0Module(
        allocator,
        0x82,
        .{ 0x8000, 0x4000, 0x7FFF, 0xC000, 1, 2, 3, 4 },
        .{ 0x8000, 0x4000, 0x7FFF, 0xC000, 5, 6, 7, 8 },
        .signed,
    );
}

fn buildSimdI16x8AbsLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI16x8(&instr, allocator, .{ 0xCFC7, 0x8000, 0xFFF9, 42, 1, 2, 3, 4 });
    try appendSimdOpcode(&instr, allocator, 0x80); // i16x8.abs
    try appendI16x8ExtractLaneS(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI16x8NegLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI16x8(&instr, allocator, .{ 12345, 0x8000, 0xFFF9, 42, 1, 2, 3, 4 });
    try appendSimdOpcode(&instr, allocator, 0x81); // i16x8.neg
    try appendI16x8ExtractLaneS(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI16x8AddSatSLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI16x8BinaryLane0Module(
        allocator,
        0x8F,
        .{ 0x7FFF, 0x8000, 30000, 0x9000, 1, 2, 3, 4 },
        .{ 1, 0xFFFF, 10000, 0x9000, 5, 6, 7, 8 },
        .signed,
    );
}

fn buildSimdI16x8AddSatULane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI16x8BinaryLane0Module(
        allocator,
        0x90,
        .{ 65530, 0, 32768, 65535, 1, 2, 3, 4 },
        .{ 20, 1, 32768, 1, 5, 6, 7, 8 },
        .unsigned,
    );
}

fn buildSimdI16x8SubSatSLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI16x8BinaryLane0Module(
        allocator,
        0x92,
        .{ 0x8000, 0x7FFF, 0x9000, 30000, 1, 2, 3, 4 },
        .{ 1, 0xFFFF, 30000, 0x9000, 5, 6, 7, 8 },
        .signed,
    );
}

fn buildSimdI16x8SubSatULane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI16x8BinaryLane0Module(
        allocator,
        0x93,
        .{ 3, 0, 32768, 65535, 1, 2, 3, 4 },
        .{ 7, 1, 32769, 1, 5, 6, 7, 8 },
        .unsigned,
    );
}

fn buildSimdI16x8MinSLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI16x8BinaryLane0Module(
        allocator,
        0x96,
        .{ 0x8000, 0x7FFF, 0xFFFF, 1, 2, 3, 4, 5 },
        .{ 0x7FFF, 0x8000, 1, 0xFFFF, 6, 7, 8, 9 },
        .signed,
    );
}

fn buildSimdI16x8MinULane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI16x8BinaryLane0Module(
        allocator,
        0x97,
        .{ 0x8000, 0x7FFF, 0xFFFF, 1, 2, 3, 4, 5 },
        .{ 0x7FFF, 0x8000, 1, 0xFFFF, 6, 7, 8, 9 },
        .unsigned,
    );
}

fn buildSimdI16x8MaxSLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI16x8BinaryLane0Module(
        allocator,
        0x98,
        .{ 0x8000, 0x7FFF, 0xFFFF, 1, 2, 3, 4, 5 },
        .{ 0x7FFF, 0x8000, 1, 0xFFFF, 6, 7, 8, 9 },
        .signed,
    );
}

fn buildSimdI16x8MaxULane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI16x8BinaryLane0Module(
        allocator,
        0x99,
        .{ 0x8000, 0x7FFF, 0xFFFF, 1, 2, 3, 4, 5 },
        .{ 0x7FFF, 0x8000, 1, 0xFFFF, 6, 7, 8, 9 },
        .unsigned,
    );
}

fn buildSimdI16x8AvgrULane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI16x8BinaryLane0Module(
        allocator,
        0x9B,
        .{ 5, 65534, 0, 1, 2, 3, 4, 5 },
        .{ 6, 65535, 1, 2, 6, 7, 8, 9 },
        .unsigned,
    );
}

fn buildSimdI16x8BinaryLane0Module(
    allocator: Allocator,
    opcode: u32,
    lhs: [8]u16,
    rhs: [8]u16,
    extract_sign: I16x8ExtractSign,
) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI16x8(&instr, allocator, lhs);
    try appendV128ConstI16x8(&instr, allocator, rhs);
    try appendSimdOpcode(&instr, allocator, opcode);
    switch (extract_sign) {
        .signed => try appendI16x8ExtractLaneS(&instr, allocator, 0),
        .unsigned => try appendI16x8ExtractLaneU(&instr, allocator, 0),
    }

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI16x8GtSLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI16x8(&instr, allocator, .{ 0xFFFF, 10, 0x8000, 4, 5, 6, 7, 8 });
    try appendV128ConstI16x8(&instr, allocator, .{ 0xFFFE, 9, 0x7FFF, 4, 5, 6, 7, 8 });
    try appendSimdOpcode(&instr, allocator, 0x31); // i16x8.gt_s
    try appendI16x8ExtractLaneS(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI16x8GtULane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI16x8(&instr, allocator, .{ 0xFFFF, 10, 0x8000, 4, 5, 6, 7, 8 });
    try appendV128ConstI16x8(&instr, allocator, .{ 0xFFFE, 9, 0x7FFF, 4, 5, 6, 7, 8 });
    try appendSimdOpcode(&instr, allocator, 0x32); // i16x8.gt_u
    try appendI16x8ExtractLaneU(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI16x8SplatLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0x1_2345);
    try appendI16x8Splat(&instr, allocator);
    try appendI16x8ExtractLaneU(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI16x8ReplaceLane5Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI16x8(&instr, allocator, .{ 1, 2, 3, 4, 5, 6, 7, 8 });
    try appendI32Const(&instr, allocator, 0x0000_FF80);
    try appendI16x8ReplaceLane(&instr, allocator, 5);
    try appendI16x8ExtractLaneS(&instr, allocator, 5);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI16x8ShlLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI16x8ShiftLane0Module(allocator, 0x8B, .{ 3, 7, 11, 13, 17, 19, 23, 29 }, .unsigned);
}

fn buildSimdI16x8ShrSLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI16x8ShiftLane0Module(allocator, 0x8C, .{ 0x8000, 0xFF00, 1024, 7, 5, 6, 7, 8 }, .signed);
}

fn buildSimdI16x8ShrULane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI16x8ShiftLane0Module(allocator, 0x8D, .{ 0x8000, 0xFF00, 1024, 7, 5, 6, 7, 8 }, .unsigned);
}

const I16x8ExtractSign = enum { signed, unsigned };

fn buildSimdI16x8ShiftLane0Module(
    allocator: Allocator,
    opcode: u32,
    lanes: [8]u16,
    extract_sign: I16x8ExtractSign,
) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    const counts = [_]i64{ 0, 1, 15, 16, 17 };
    for (counts, 0..) |count, idx| {
        try appendV128ConstI16x8(&instr, allocator, lanes);
        try appendI32Const(&instr, allocator, count);
        try appendSimdOpcode(&instr, allocator, opcode);
        switch (extract_sign) {
            .signed => try appendI16x8ExtractLaneS(&instr, allocator, 0),
            .unsigned => try appendI16x8ExtractLaneU(&instr, allocator, 0),
        }
        if (idx != 0) try appendI32Add(&instr, allocator);
    }

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI8x16AddLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI8x16(&instr, allocator, .{ 200, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    try appendV128ConstI8x16(&instr, allocator, .{ 100, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 });
    try appendSimdOpcode(&instr, allocator, 0x6E); // i8x16.add
    try appendI8x16ExtractLaneU(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI8x16SubLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI8x16(&instr, allocator, .{ 3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    try appendV128ConstI8x16(&instr, allocator, .{ 7, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 });
    try appendSimdOpcode(&instr, allocator, 0x71); // i8x16.sub
    try appendI8x16ExtractLaneU(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI8x16AbsLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI8x16(&instr, allocator, .{ 0x85, 0x80, 0xF9, 42, 1, 2, 3, 4, 0x85, 0x80, 5, 6, 7, 8, 9, 10 });
    try appendSimdOpcode(&instr, allocator, 0x60); // i8x16.abs
    try appendI8x16ExtractLaneS(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI8x16NegLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI8x16(&instr, allocator, .{ 123, 0x80, 0xF9, 42, 1, 2, 3, 4, 123, 0x80, 5, 6, 7, 8, 9, 10 });
    try appendSimdOpcode(&instr, allocator, 0x61); // i8x16.neg
    try appendI8x16ExtractLaneS(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI8x16AddSatSLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI8x16BinaryLane0Module(
        allocator,
        0x6F,
        .{ 0x7F, 0x80, 100, 0x90, 1, 2, 3, 4, 0x7F, 0x80, 5, 6, 7, 8, 9, 10 },
        .{ 1, 0xFF, 50, 0x90, 11, 12, 13, 14, 1, 0xFF, 15, 16, 17, 18, 19, 20 },
        .signed,
    );
}

fn buildSimdI8x16AddSatULane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI8x16BinaryLane0Module(
        allocator,
        0x70,
        .{ 250, 0, 128, 255, 1, 2, 3, 4, 250, 0, 5, 6, 7, 8, 9, 10 },
        .{ 20, 1, 128, 1, 11, 12, 13, 14, 20, 1, 15, 16, 17, 18, 19, 20 },
        .unsigned,
    );
}

fn buildSimdI8x16SubSatSLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI8x16BinaryLane0Module(
        allocator,
        0x72,
        .{ 0x80, 0x7F, 0x90, 100, 1, 2, 3, 4, 0x80, 0x7F, 5, 6, 7, 8, 9, 10 },
        .{ 1, 0xFF, 100, 0x90, 11, 12, 13, 14, 1, 0xFF, 15, 16, 17, 18, 19, 20 },
        .signed,
    );
}

fn buildSimdI8x16SubSatULane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI8x16BinaryLane0Module(
        allocator,
        0x73,
        .{ 3, 0, 128, 255, 1, 2, 3, 4, 3, 0, 5, 6, 7, 8, 9, 10 },
        .{ 7, 1, 129, 1, 11, 12, 13, 14, 7, 1, 15, 16, 17, 18, 19, 20 },
        .unsigned,
    );
}

fn buildSimdI8x16MinSLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI8x16BinaryLane0Module(
        allocator,
        0x76,
        .{ 0x80, 0x7F, 0xFF, 1, 2, 3, 4, 5, 0x80, 0x7F, 6, 7, 8, 9, 10, 11 },
        .{ 0x7F, 0x80, 1, 0xFF, 12, 13, 14, 15, 0x7F, 0x80, 16, 17, 18, 19, 20, 21 },
        .signed,
    );
}

fn buildSimdI8x16MinULane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI8x16BinaryLane0Module(
        allocator,
        0x77,
        .{ 0x80, 0x7F, 0xFF, 1, 2, 3, 4, 5, 0x80, 0x7F, 6, 7, 8, 9, 10, 11 },
        .{ 0x7F, 0x80, 1, 0xFF, 12, 13, 14, 15, 0x7F, 0x80, 16, 17, 18, 19, 20, 21 },
        .unsigned,
    );
}

fn buildSimdI8x16MaxSLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI8x16BinaryLane0Module(
        allocator,
        0x78,
        .{ 0x80, 0x7F, 0xFF, 1, 2, 3, 4, 5, 0x80, 0x7F, 6, 7, 8, 9, 10, 11 },
        .{ 0x7F, 0x80, 1, 0xFF, 12, 13, 14, 15, 0x7F, 0x80, 16, 17, 18, 19, 20, 21 },
        .signed,
    );
}

fn buildSimdI8x16MaxULane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI8x16BinaryLane0Module(
        allocator,
        0x79,
        .{ 0x80, 0x7F, 0xFF, 1, 2, 3, 4, 5, 0x80, 0x7F, 6, 7, 8, 9, 10, 11 },
        .{ 0x7F, 0x80, 1, 0xFF, 12, 13, 14, 15, 0x7F, 0x80, 16, 17, 18, 19, 20, 21 },
        .unsigned,
    );
}

fn buildSimdI8x16AvgrULane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI8x16BinaryLane0Module(
        allocator,
        0x7B,
        .{ 5, 254, 0, 1, 2, 3, 4, 5, 5, 254, 6, 7, 8, 9, 10, 11 },
        .{ 6, 255, 1, 2, 12, 13, 14, 15, 6, 255, 16, 17, 18, 19, 20, 21 },
        .unsigned,
    );
}

fn buildSimdI8x16BinaryLane0Module(
    allocator: Allocator,
    opcode: u32,
    lhs: [16]u8,
    rhs: [16]u8,
    extract_sign: I8x16ExtractSign,
) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI8x16(&instr, allocator, lhs);
    try appendV128ConstI8x16(&instr, allocator, rhs);
    try appendSimdOpcode(&instr, allocator, opcode);
    switch (extract_sign) {
        .signed => try appendI8x16ExtractLaneS(&instr, allocator, 0),
        .unsigned => try appendI8x16ExtractLaneU(&instr, allocator, 0),
    }

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI8x16EqLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI8x16(&instr, allocator, .{ 42, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    try appendV128ConstI8x16(&instr, allocator, .{ 42, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 });
    try appendSimdOpcode(&instr, allocator, 0x23); // i8x16.eq
    try appendI8x16ExtractLaneU(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI8x16GtSLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI8x16(&instr, allocator, .{ 0x7F, 10, 0x80, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    try appendV128ConstI8x16(&instr, allocator, .{ 0x80, 9, 0x7F, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    try appendSimdOpcode(&instr, allocator, 0x27); // i8x16.gt_s
    try appendI8x16ExtractLaneS(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI8x16GtULane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI8x16(&instr, allocator, .{ 0xFF, 10, 0x80, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    try appendV128ConstI8x16(&instr, allocator, .{ 0x01, 9, 0x7F, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    try appendSimdOpcode(&instr, allocator, 0x28); // i8x16.gt_u
    try appendI8x16ExtractLaneU(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI8x16SplatLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0x1_2334);
    try appendI8x16Splat(&instr, allocator);
    try appendI8x16ExtractLaneU(&instr, allocator, 0);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI8x16ReplaceLane13Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI8x16(&instr, allocator, .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    try appendI32Const(&instr, allocator, 0x0000_0080);
    try appendI8x16ReplaceLane(&instr, allocator, 13);
    try appendI8x16ExtractLaneS(&instr, allocator, 13);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI8x16ShlLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI8x16ShiftLane0Module(allocator, 0x6B, .{ 3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61 }, .unsigned);
}

fn buildSimdI8x16ShrSLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI8x16ShiftLane0Module(allocator, 0x6C, .{ 0x80, 0xFF, 0x7F, 0x40, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }, .signed);
}

fn buildSimdI8x16ShrULane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI8x16ShiftLane0Module(allocator, 0x6D, .{ 0x80, 0xFF, 0x7F, 0x40, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }, .unsigned);
}

const I8x16ExtractSign = enum { signed, unsigned };

fn buildSimdI8x16ShiftLane0Module(
    allocator: Allocator,
    opcode: u32,
    lanes: [16]u8,
    extract_sign: I8x16ExtractSign,
) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    const counts = [_]i64{ 0, 1, 7, 8, 9 };
    for (counts, 0..) |count, idx| {
        try appendV128ConstI8x16(&instr, allocator, lanes);
        try appendI32Const(&instr, allocator, count);
        try appendSimdOpcode(&instr, allocator, opcode);
        switch (extract_sign) {
            .signed => try appendI8x16ExtractLaneS(&instr, allocator, 0),
            .unsigned => try appendI8x16ExtractLaneU(&instr, allocator, 0),
        }
        if (idx != 0) try appendI32Add(&instr, allocator);
    }

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI64x2AddLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI64x2(&instr, allocator, .{ 1, 2 });
    try appendV128ConstI64x2(&instr, allocator, .{ 5, 6 });
    try appendSimdOpcode(&instr, allocator, 0xCE); // i64x2.add
    try appendI64x2ExtractLane(&instr, allocator, 0);
    try appendI32WrapI64(&instr, allocator);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI64x2SubLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI64x2(&instr, allocator, .{ 3, 2 });
    try appendV128ConstI64x2(&instr, allocator, .{ 7, 6 });
    try appendSimdOpcode(&instr, allocator, 0xD1); // i64x2.sub
    try appendI64x2ExtractLane(&instr, allocator, 0);
    try appendI32WrapI64(&instr, allocator);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI64x2AbsLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI64x2(&instr, allocator, .{ 0xFFFF_FFFE_FFFF_FF85, 0x8000_0000_0000_0000 });
    try appendSimdOpcode(&instr, allocator, 0xC0); // i64x2.abs
    try appendI64x2ExtractLane(&instr, allocator, 0);
    try appendI32WrapI64(&instr, allocator);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI64x2NegLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI64x2(&instr, allocator, .{ 0x0000_0001_0000_007B, 0x8000_0000_0000_0000 });
    try appendSimdOpcode(&instr, allocator, 0xC1); // i64x2.neg
    try appendI64x2ExtractLane(&instr, allocator, 0);
    try appendI32WrapI64(&instr, allocator);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI64x2EqLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI64x2(&instr, allocator, .{ 42, 2 });
    try appendV128ConstI64x2(&instr, allocator, .{ 42, 0 });
    try appendSimdOpcode(&instr, allocator, 0xD6); // i64x2.eq
    try appendI64x2ExtractLane(&instr, allocator, 0);
    try appendI32WrapI64(&instr, allocator);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI64x2GtSLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI64x2(&instr, allocator, .{ 0xffff_ffff_ffff_ffff, 4 });
    try appendV128ConstI64x2(&instr, allocator, .{ 0xffff_ffff_ffff_fffe, 5 });
    try appendSimdOpcode(&instr, allocator, 0xD9); // i64x2.gt_s
    try appendI64x2ExtractLane(&instr, allocator, 0);
    try appendI32WrapI64(&instr, allocator);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI64x2SplatLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI64Const(&instr, allocator, 77);
    try appendI64x2Splat(&instr, allocator);
    try appendI64x2ExtractLane(&instr, allocator, 0);
    try appendI32WrapI64(&instr, allocator);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI64x2ReplaceLane1Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendV128ConstI64x2(&instr, allocator, .{ 1, 2 });
    try appendI64Const(&instr, allocator, -128);
    try appendI64x2ReplaceLane(&instr, allocator, 1);
    try appendI64x2ExtractLane(&instr, allocator, 1);
    try appendI32WrapI64(&instr, allocator);

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdI64x2ShlLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI64x2ShiftLane0Module(allocator, 0xCB, .{ 3, 7 });
}

fn buildSimdI64x2ShrSLane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI64x2ShiftLane0Module(allocator, 0xCC, .{ 0xffff_ffff_ffff_fff8, 0x8000_0000_0000_0000 });
}

fn buildSimdI64x2ShrULane0Module(allocator: Allocator) ![]u8 {
    return buildSimdI64x2ShiftLane0Module(allocator, 0xCD, .{ 0xffff_ffff_ffff_fff8, 0x8000_0000_0000_0000 });
}

fn buildSimdI64x2ShiftLane0Module(allocator: Allocator, opcode: u32, lanes: [2]u64) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    const counts = [_]i64{ 0, 1, 63, 64, 65 };
    for (counts, 0..) |count, idx| {
        try appendV128ConstI64x2(&instr, allocator, lanes);
        try appendI32Const(&instr, allocator, count);
        try appendSimdOpcode(&instr, allocator, opcode);
        try appendI64x2ExtractLane(&instr, allocator, 0);
        try appendI32WrapI64(&instr, allocator);
        if (idx != 0) try appendI32Add(&instr, allocator);
    }

    return buildRunI32Module(allocator, instr.items, .{});
}

fn buildSimdLoadStoreLane0Module(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try instr.append(allocator, 0x41); // i32.const destination address
    try encodeSLEB128(&instr, allocator, 16);
    try instr.append(allocator, 0x41); // i32.const source address
    try encodeSLEB128(&instr, allocator, 0);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdMemOpcode(&instr, allocator, 0x0B, 4, 0); // v128.store
    try instr.append(allocator, 0x41); // i32.const destination address
    try encodeSLEB128(&instr, allocator, 16);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendI32x4ExtractLane(&instr, allocator, 0);

    var data: [16]u8 = undefined;
    writeI32Lane(data[0..4], 0x1122_3344);
    writeI32Lane(data[4..8], 0x5566_7788);
    writeI32Lane(data[8..12], @bitCast(@as(i32, -7)));
    writeI32Lane(data[12..16], 0x0102_0304);

    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data[0..],
    });
}

const memory_loop_lanes: u32 = 1024;
const memory_loop_bytes: u32 = memory_loop_lanes * @sizeOf(u32);
const memory_loop_a_base: u32 = 0;
const memory_loop_b_base: u32 = memory_loop_a_base + memory_loop_bytes;
const memory_loop_dst_base: u32 = memory_loop_b_base + memory_loop_bytes;

fn buildScalarI32MemoryAdd4kLoopModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 0);
    try appendBlock(&instr, allocator);
    try appendLoop(&instr, allocator);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, memory_loop_bytes);
    try appendI32GeU(&instr, allocator);
    try appendBrIf(&instr, allocator, 1);

    try appendI32Const(&instr, allocator, memory_loop_dst_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendI32Load(&instr, allocator, 2, 0);

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendI32Load(&instr, allocator, 2, 0);

    try appendI32Add(&instr, allocator);
    try appendI32Store(&instr, allocator, 2, 0);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, @sizeOf(u32));
    try appendI32Add(&instr, allocator);
    try appendLocalSet(&instr, allocator, 0);
    try appendBr(&instr, allocator, 0);

    try appendEnd(&instr, allocator);
    try appendEnd(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_dst_base + memory_loop_bytes - @sizeOf(u32));
    try appendI32Load(&instr, allocator, 2, 0);

    const data = try buildMemoryLoopData(allocator);
    defer allocator.free(data);
    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data,
        .local_i32_count = 1,
    });
}

fn buildSimdI32x4MemoryAdd4kLoopModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 0);
    try appendBlock(&instr, allocator);
    try appendLoop(&instr, allocator);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, memory_loop_bytes);
    try appendI32GeU(&instr, allocator);
    try appendBrIf(&instr, allocator, 1);

    try appendI32Const(&instr, allocator, memory_loop_dst_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendSimdOpcode(&instr, allocator, 0xAE); // i32x4.add
    try appendSimdMemOpcode(&instr, allocator, 0x0B, 4, 0); // v128.store

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 16);
    try appendI32Add(&instr, allocator);
    try appendLocalSet(&instr, allocator, 0);
    try appendBr(&instr, allocator, 0);

    try appendEnd(&instr, allocator);
    try appendEnd(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_dst_base + memory_loop_bytes - 16);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendI32x4ExtractLane(&instr, allocator, 3);

    const data = try buildMemoryLoopData(allocator);
    defer allocator.free(data);
    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data,
        .local_i32_count = 1,
    });
}

fn buildSimdI32x4MemorySum8_4kLoopModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 0);
    try appendBlock(&instr, allocator);
    try appendLoop(&instr, allocator);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, memory_loop_bytes);
    try appendI32GeU(&instr, allocator);
    try appendBrIf(&instr, allocator, 1);

    inline for (0..4) |chunk| {
        const byte_offset = chunk * 16;

        try appendI32Const(&instr, allocator, memory_loop_a_base);
        try appendLocalGet(&instr, allocator, 0);
        try appendI32Add(&instr, allocator);
        try appendSimdMemOpcode(&instr, allocator, 0x00, 4, byte_offset); // v128.load

        try appendI32Const(&instr, allocator, memory_loop_b_base);
        try appendLocalGet(&instr, allocator, 0);
        try appendI32Add(&instr, allocator);
        try appendSimdMemOpcode(&instr, allocator, 0x00, 4, byte_offset); // v128.load
    }

    inline for (0..7) |_| {
        try appendSimdOpcode(&instr, allocator, 0xAE); // i32x4.add
    }
    try appendI32x4ExtractLane(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 1);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 64);
    try appendI32Add(&instr, allocator);
    try appendLocalSet(&instr, allocator, 0);
    try appendBr(&instr, allocator, 0);

    try appendEnd(&instr, allocator);
    try appendEnd(&instr, allocator);

    try appendLocalGet(&instr, allocator, 1);

    const data = try buildMemoryLoopData(allocator);
    defer allocator.free(data);
    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data,
        .local_i32_count = 2,
    });
}

fn buildSimdI32x4ShiftMix4kLoopModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 0);
    try appendBlock(&instr, allocator);
    try appendLoop(&instr, allocator);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, memory_loop_bytes);
    try appendI32GeU(&instr, allocator);
    try appendBrIf(&instr, allocator, 1);

    try appendI32Const(&instr, allocator, memory_loop_dst_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendLocalGet(&instr, allocator, 0);
    try appendSimdOpcode(&instr, allocator, 0xAB); // i32x4.shl

    try appendLocalGet(&instr, allocator, 0);
    try appendSimdOpcode(&instr, allocator, 0xAD); // i32x4.shr_u

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendLocalGet(&instr, allocator, 0);
    try appendSimdOpcode(&instr, allocator, 0xAC); // i32x4.shr_s

    try appendSimdOpcode(&instr, allocator, 0xAE); // i32x4.add
    try appendSimdMemOpcode(&instr, allocator, 0x0B, 4, 0); // v128.store

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 16);
    try appendI32Add(&instr, allocator);
    try appendLocalSet(&instr, allocator, 0);
    try appendBr(&instr, allocator, 0);

    try appendEnd(&instr, allocator);
    try appendEnd(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_dst_base + memory_loop_bytes - 16);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendI32x4ExtractLane(&instr, allocator, 3);

    const data = try buildMemoryLoopData(allocator);
    defer allocator.free(data);
    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data,
        .local_i32_count = 1,
    });
}

fn buildSimdI32x4MinMax4kLoopModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 0);
    try appendBlock(&instr, allocator);
    try appendLoop(&instr, allocator);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, memory_loop_bytes);
    try appendI32GeU(&instr, allocator);
    try appendBrIf(&instr, allocator, 1);

    try appendI32Const(&instr, allocator, memory_loop_dst_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0xB6); // i32x4.min_s

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0xB9); // i32x4.max_u
    try appendSimdOpcode(&instr, allocator, 0xB8); // i32x4.max_s

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0xB7); // i32x4.min_u
    try appendSimdOpcode(&instr, allocator, 0xB9); // i32x4.max_u
    try appendSimdMemOpcode(&instr, allocator, 0x0B, 4, 0); // v128.store

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 16);
    try appendI32Add(&instr, allocator);
    try appendLocalSet(&instr, allocator, 0);
    try appendBr(&instr, allocator, 0);

    try appendEnd(&instr, allocator);
    try appendEnd(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_dst_base + memory_loop_bytes - 16);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendI32x4ExtractLane(&instr, allocator, 3);

    const data = try buildMemoryLoopDataI32MinMax(allocator);
    defer allocator.free(data);
    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data,
        .local_i32_count = 1,
    });
}

fn buildSimdI16x8MemoryAdd4kLoopModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 0);
    try appendBlock(&instr, allocator);
    try appendLoop(&instr, allocator);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, memory_loop_bytes);
    try appendI32GeU(&instr, allocator);
    try appendBrIf(&instr, allocator, 1);

    try appendI32Const(&instr, allocator, memory_loop_dst_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendSimdOpcode(&instr, allocator, 0x8E); // i16x8.add
    try appendSimdMemOpcode(&instr, allocator, 0x0B, 4, 0); // v128.store

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 16);
    try appendI32Add(&instr, allocator);
    try appendLocalSet(&instr, allocator, 0);
    try appendBr(&instr, allocator, 0);

    try appendEnd(&instr, allocator);
    try appendEnd(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_dst_base + memory_loop_bytes - 16);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendI16x8ExtractLaneU(&instr, allocator, 7);

    const data = try buildMemoryLoopDataI16(allocator);
    defer allocator.free(data);
    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data,
        .local_i32_count = 1,
    });
}

fn buildSimdI16x8ShiftMix4kLoopModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 0);
    try appendBlock(&instr, allocator);
    try appendLoop(&instr, allocator);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, memory_loop_bytes);
    try appendI32GeU(&instr, allocator);
    try appendBrIf(&instr, allocator, 1);

    try appendI32Const(&instr, allocator, memory_loop_dst_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 1);
    try appendI32ShrU(&instr, allocator);
    try appendSimdOpcode(&instr, allocator, 0x8B); // i16x8.shl

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 2);
    try appendI32ShrU(&instr, allocator);
    try appendSimdOpcode(&instr, allocator, 0x8D); // i16x8.shr_u

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 3);
    try appendI32ShrU(&instr, allocator);
    try appendSimdOpcode(&instr, allocator, 0x8C); // i16x8.shr_s

    try appendSimdOpcode(&instr, allocator, 0x8E); // i16x8.add
    try appendSimdMemOpcode(&instr, allocator, 0x0B, 4, 0); // v128.store

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 16);
    try appendI32Add(&instr, allocator);
    try appendLocalSet(&instr, allocator, 0);
    try appendBr(&instr, allocator, 0);

    try appendEnd(&instr, allocator);
    try appendEnd(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_dst_base + memory_loop_bytes - 16);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendI16x8ExtractLaneU(&instr, allocator, 6);

    const data = try buildMemoryLoopDataI16(allocator);
    defer allocator.free(data);
    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data,
        .local_i32_count = 1,
    });
}

fn buildSimdI16x8ArithExtra4kLoopModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 0);
    try appendBlock(&instr, allocator);
    try appendLoop(&instr, allocator);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, memory_loop_bytes);
    try appendI32GeU(&instr, allocator);
    try appendBrIf(&instr, allocator, 1);

    try appendI32Const(&instr, allocator, memory_loop_dst_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendSimdOpcode(&instr, allocator, 0x82); // i16x8.q15mulr_sat_s

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x8F); // i16x8.add_sat_s

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x90); // i16x8.add_sat_u

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x92); // i16x8.sub_sat_s

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x93); // i16x8.sub_sat_u

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x9B); // i16x8.avgr_u

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x96); // i16x8.min_s

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x97); // i16x8.min_u

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x98); // i16x8.max_s

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x99); // i16x8.max_u

    try appendSimdMemOpcode(&instr, allocator, 0x0B, 4, 0); // v128.store

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 16);
    try appendI32Add(&instr, allocator);
    try appendLocalSet(&instr, allocator, 0);
    try appendBr(&instr, allocator, 0);

    try appendEnd(&instr, allocator);
    try appendEnd(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_dst_base + memory_loop_bytes - 16);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendI16x8ExtractLaneU(&instr, allocator, 7);

    const data = try buildMemoryLoopDataI16ArithExtra(allocator);
    defer allocator.free(data);
    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data,
        .local_i32_count = 1,
    });
}

fn buildSimdI8x16MemoryAdd4kLoopModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 0);
    try appendBlock(&instr, allocator);
    try appendLoop(&instr, allocator);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, memory_loop_bytes);
    try appendI32GeU(&instr, allocator);
    try appendBrIf(&instr, allocator, 1);

    try appendI32Const(&instr, allocator, memory_loop_dst_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendSimdOpcode(&instr, allocator, 0x6E); // i8x16.add
    try appendSimdMemOpcode(&instr, allocator, 0x0B, 4, 0); // v128.store

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 16);
    try appendI32Add(&instr, allocator);
    try appendLocalSet(&instr, allocator, 0);
    try appendBr(&instr, allocator, 0);

    try appendEnd(&instr, allocator);
    try appendEnd(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_dst_base + memory_loop_bytes - 16);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendI8x16ExtractLaneU(&instr, allocator, 15);

    const data = try buildMemoryLoopDataI8(allocator);
    defer allocator.free(data);
    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data,
        .local_i32_count = 1,
    });
}

fn buildSimdI8x16ShiftMix4kLoopModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 0);
    try appendBlock(&instr, allocator);
    try appendLoop(&instr, allocator);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, memory_loop_bytes);
    try appendI32GeU(&instr, allocator);
    try appendBrIf(&instr, allocator, 1);

    try appendI32Const(&instr, allocator, memory_loop_dst_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 4);
    try appendI32ShrU(&instr, allocator);
    try appendI32Const(&instr, allocator, 9);
    try appendI32Add(&instr, allocator);
    try appendSimdOpcode(&instr, allocator, 0x6B); // i8x16.shl

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 4);
    try appendI32ShrU(&instr, allocator);
    try appendI32Const(&instr, allocator, 10);
    try appendI32Add(&instr, allocator);
    try appendSimdOpcode(&instr, allocator, 0x6D); // i8x16.shr_u

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 4);
    try appendI32ShrU(&instr, allocator);
    try appendI32Const(&instr, allocator, 11);
    try appendI32Add(&instr, allocator);
    try appendSimdOpcode(&instr, allocator, 0x6C); // i8x16.shr_s

    try appendSimdOpcode(&instr, allocator, 0x6E); // i8x16.add
    try appendSimdMemOpcode(&instr, allocator, 0x0B, 4, 0); // v128.store

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 16);
    try appendI32Add(&instr, allocator);
    try appendLocalSet(&instr, allocator, 0);
    try appendBr(&instr, allocator, 0);

    try appendEnd(&instr, allocator);
    try appendEnd(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_dst_base + memory_loop_bytes - 16);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendI8x16ExtractLaneU(&instr, allocator, 14);

    const data = try buildMemoryLoopDataI8(allocator);
    defer allocator.free(data);
    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data,
        .local_i32_count = 1,
    });
}

fn buildSimdI8x16ArithExtra4kLoopModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 0);
    try appendBlock(&instr, allocator);
    try appendLoop(&instr, allocator);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, memory_loop_bytes);
    try appendI32GeU(&instr, allocator);
    try appendBrIf(&instr, allocator, 1);

    try appendI32Const(&instr, allocator, memory_loop_dst_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendSimdOpcode(&instr, allocator, 0x70); // i8x16.add_sat_u

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x73); // i8x16.sub_sat_u

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x7B); // i8x16.avgr_u

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x6F); // i8x16.add_sat_s

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x72); // i8x16.sub_sat_s

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x79); // i8x16.max_u

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x76); // i8x16.min_s

    try appendSimdMemOpcode(&instr, allocator, 0x0B, 4, 0); // v128.store

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 16);
    try appendI32Add(&instr, allocator);
    try appendLocalSet(&instr, allocator, 0);
    try appendBr(&instr, allocator, 0);

    try appendEnd(&instr, allocator);
    try appendEnd(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_dst_base + memory_loop_bytes - 16);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendI8x16ExtractLaneU(&instr, allocator, 15);

    const data = try buildMemoryLoopDataI8(allocator);
    defer allocator.free(data);
    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data,
        .local_i32_count = 1,
    });
}

fn buildSimdI64x2MemoryAdd4kLoopModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 0);
    try appendBlock(&instr, allocator);
    try appendLoop(&instr, allocator);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, memory_loop_bytes);
    try appendI32GeU(&instr, allocator);
    try appendBrIf(&instr, allocator, 1);

    try appendI32Const(&instr, allocator, memory_loop_dst_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendSimdOpcode(&instr, allocator, 0xCE); // i64x2.add
    try appendSimdMemOpcode(&instr, allocator, 0x0B, 4, 0); // v128.store

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 16);
    try appendI32Add(&instr, allocator);
    try appendLocalSet(&instr, allocator, 0);
    try appendBr(&instr, allocator, 0);

    try appendEnd(&instr, allocator);
    try appendEnd(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_dst_base + memory_loop_bytes - 16);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendI64x2ExtractLane(&instr, allocator, 1);
    try appendI32WrapI64(&instr, allocator);

    const data = try buildMemoryLoopDataI64(allocator);
    defer allocator.free(data);
    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data,
        .local_i32_count = 1,
    });
}

fn buildSimdI64x2ShiftMix4kLoopModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 0);
    try appendBlock(&instr, allocator);
    try appendLoop(&instr, allocator);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, memory_loop_bytes);
    try appendI32GeU(&instr, allocator);
    try appendBrIf(&instr, allocator, 1);

    try appendI32Const(&instr, allocator, memory_loop_dst_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 3);
    try appendI32ShrU(&instr, allocator);
    try appendI32Const(&instr, allocator, 65);
    try appendI32Add(&instr, allocator);
    try appendSimdOpcode(&instr, allocator, 0xCB); // i64x2.shl

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 3);
    try appendI32ShrU(&instr, allocator);
    try appendI32Const(&instr, allocator, 66);
    try appendI32Add(&instr, allocator);
    try appendSimdOpcode(&instr, allocator, 0xCD); // i64x2.shr_u

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 3);
    try appendI32ShrU(&instr, allocator);
    try appendI32Const(&instr, allocator, 67);
    try appendI32Add(&instr, allocator);
    try appendSimdOpcode(&instr, allocator, 0xCC); // i64x2.shr_s

    try appendSimdOpcode(&instr, allocator, 0xCE); // i64x2.add
    try appendSimdMemOpcode(&instr, allocator, 0x0B, 4, 0); // v128.store

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 16);
    try appendI32Add(&instr, allocator);
    try appendLocalSet(&instr, allocator, 0);
    try appendBr(&instr, allocator, 0);

    try appendEnd(&instr, allocator);
    try appendEnd(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_dst_base + memory_loop_bytes - 16);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendI64x2ExtractLane(&instr, allocator, 1);
    try appendI32WrapI64(&instr, allocator);

    const data = try buildMemoryLoopDataI64(allocator);
    defer allocator.free(data);
    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data,
        .local_i32_count = 1,
    });
}

fn buildSimdIntAbsNeg4kLoopModule(allocator: Allocator) ![]u8 {
    var instr: std.ArrayList(u8) = .empty;
    defer instr.deinit(allocator);

    try appendI32Const(&instr, allocator, 0);
    try appendLocalSet(&instr, allocator, 0);
    try appendBlock(&instr, allocator);
    try appendLoop(&instr, allocator);

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, memory_loop_bytes);
    try appendI32GeU(&instr, allocator);
    try appendBrIf(&instr, allocator, 1);

    try appendI32Const(&instr, allocator, memory_loop_dst_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_a_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0x60); // i8x16.abs
    try appendSimdOpcode(&instr, allocator, 0x61); // i8x16.neg
    try appendSimdOpcode(&instr, allocator, 0x80); // i16x8.abs
    try appendSimdOpcode(&instr, allocator, 0x81); // i16x8.neg

    try appendI32Const(&instr, allocator, memory_loop_b_base);
    try appendLocalGet(&instr, allocator, 0);
    try appendI32Add(&instr, allocator);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendSimdOpcode(&instr, allocator, 0xA0); // i32x4.abs
    try appendSimdOpcode(&instr, allocator, 0xA1); // i32x4.neg
    try appendSimdOpcode(&instr, allocator, 0xC0); // i64x2.abs
    try appendSimdOpcode(&instr, allocator, 0xC1); // i64x2.neg

    try appendSimdOpcode(&instr, allocator, 0x51); // v128.xor
    try appendSimdMemOpcode(&instr, allocator, 0x0B, 4, 0); // v128.store

    try appendLocalGet(&instr, allocator, 0);
    try appendI32Const(&instr, allocator, 16);
    try appendI32Add(&instr, allocator);
    try appendLocalSet(&instr, allocator, 0);
    try appendBr(&instr, allocator, 0);

    try appendEnd(&instr, allocator);
    try appendEnd(&instr, allocator);

    try appendI32Const(&instr, allocator, memory_loop_dst_base + memory_loop_bytes - 16);
    try appendSimdMemOpcode(&instr, allocator, 0x00, 4, 0); // v128.load
    try appendI32x4ExtractLane(&instr, allocator, 3);

    const data = try buildMemoryLoopDataIntAbsNeg(allocator);
    defer allocator.free(data);
    return buildRunI32Module(allocator, instr.items, .{
        .memory_min = 1,
        .data = data,
        .local_i32_count = 1,
    });
}

fn buildMemoryLoopData(allocator: Allocator) ![]u8 {
    const data_len = memory_loop_b_base + memory_loop_bytes;
    const data = try allocator.alloc(u8, data_len);
    @memset(data, 0);

    var lane: u32 = 0;
    while (lane < memory_loop_lanes) : (lane += 1) {
        const lane_offset = lane * @sizeOf(u32);
        const a_pos: usize = @intCast(memory_loop_a_base + lane_offset);
        const b_pos: usize = @intCast(memory_loop_b_base + lane_offset);
        writeI32Lane(data[a_pos..][0..4], lane * 3 + 5);
        writeI32Lane(data[b_pos..][0..4], lane * 7 + 11);
    }

    return data;
}

fn buildMemoryLoopDataI32MinMax(allocator: Allocator) ![]u8 {
    const data_len = memory_loop_b_base + memory_loop_bytes;
    const data = try allocator.alloc(u8, data_len);
    @memset(data, 0);

    var lane: u32 = 0;
    while (lane < memory_loop_lanes) : (lane += 1) {
        const lane_offset = lane * @sizeOf(u32);
        const a_pos: usize = @intCast(memory_loop_a_base + lane_offset);
        const b_pos: usize = @intCast(memory_loop_b_base + lane_offset);
        const a_mag = lane * 3 + 5;
        const b_mag = lane * 7 + 11;
        const a_value = if ((lane & 1) == 0) 0x8000_0000 | a_mag else a_mag;
        const b_value = if ((lane & 1) == 0) b_mag else 0x8000_0000 | b_mag;
        writeI32Lane(data[a_pos..][0..4], a_value);
        writeI32Lane(data[b_pos..][0..4], b_value);
    }

    return data;
}

fn buildMemoryLoopDataIntAbsNeg(allocator: Allocator) ![]u8 {
    const data_len = memory_loop_b_base + memory_loop_bytes;
    const data = try allocator.alloc(u8, data_len);
    @memset(data, 0);

    var byte: u32 = 0;
    while (byte < memory_loop_bytes) : (byte += 1) {
        const a_pos: usize = @intCast(memory_loop_a_base + byte);
        const b_pos: usize = @intCast(memory_loop_b_base + byte);
        data[a_pos] = if ((byte & 0x0F) == 0)
            0x80
        else
            @intCast((byte * 13 + 0x81) & 0xFF);
        data[b_pos] = if ((byte & 0x0F) == 8)
            0x80
        else
            @intCast((byte * 7 + 0x41) & 0xFF);
    }

    return data;
}

fn buildMemoryLoopDataI16(allocator: Allocator) ![]u8 {
    const data_len = memory_loop_b_base + memory_loop_bytes;
    const data = try allocator.alloc(u8, data_len);
    @memset(data, 0);

    var lane: u32 = 0;
    while (lane < memory_loop_bytes / @sizeOf(u16)) : (lane += 1) {
        const lane_offset = lane * @sizeOf(u16);
        const a_pos: usize = @intCast(memory_loop_a_base + lane_offset);
        const b_pos: usize = @intCast(memory_loop_b_base + lane_offset);
        writeI16Lane(data[a_pos..][0..2], @intCast((lane * 3 + 5) & 0xFFFF));
        writeI16Lane(data[b_pos..][0..2], @intCast((lane * 7 + 11) & 0xFFFF));
    }

    return data;
}

fn buildMemoryLoopDataI16ArithExtra(allocator: Allocator) ![]u8 {
    const data_len = memory_loop_b_base + memory_loop_bytes;
    const data = try allocator.alloc(u8, data_len);
    @memset(data, 0);

    var lane: u32 = 0;
    while (lane < memory_loop_bytes / @sizeOf(u16)) : (lane += 1) {
        const lane_offset = lane * @sizeOf(u16);
        const a_pos: usize = @intCast(memory_loop_a_base + lane_offset);
        const b_pos: usize = @intCast(memory_loop_b_base + lane_offset);
        writeI16Lane(data[a_pos..][0..2], @intCast((lane * 1103 + 0x8000) & 0xFFFF));
        writeI16Lane(data[b_pos..][0..2], @intCast((lane * 1741 + 0x7FFF) & 0xFFFF));
    }

    return data;
}

fn buildMemoryLoopDataI8(allocator: Allocator) ![]u8 {
    const data_len = memory_loop_b_base + memory_loop_bytes;
    const data = try allocator.alloc(u8, data_len);
    @memset(data, 0);

    for (0..memory_loop_bytes) |offset| {
        data[memory_loop_a_base + offset] = @intCast((offset * 3 + 5) & 0xFF);
        data[memory_loop_b_base + offset] = @intCast((offset * 7 + 11) & 0xFF);
    }

    return data;
}

fn buildMemoryLoopDataI64(allocator: Allocator) ![]u8 {
    const data_len = memory_loop_b_base + memory_loop_bytes;
    const data = try allocator.alloc(u8, data_len);
    @memset(data, 0);

    var lane: u32 = 0;
    while (lane < memory_loop_bytes / @sizeOf(u64)) : (lane += 1) {
        const lane_offset = lane * @sizeOf(u64);
        const a_pos: usize = @intCast(memory_loop_a_base + lane_offset);
        const b_pos: usize = @intCast(memory_loop_b_base + lane_offset);
        const lane64: u64 = lane;
        writeI64Lane(data[a_pos..][0..8], lane64 * 3 + 5);
        writeI64Lane(data[b_pos..][0..8], lane64 * 7 + 11);
    }

    return data;
}

const ModuleExtras = struct {
    memory_min: ?u32 = null,
    data: ?[]const u8 = null,
    local_i32_count: u32 = 0,
};

fn buildRunI32Module(allocator: Allocator, instructions: []const u8, extras: ModuleExtras) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, &[_]u8{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 });

    var type_payload: std.ArrayList(u8) = .empty;
    defer type_payload.deinit(allocator);
    try type_payload.appendSlice(allocator, &[_]u8{
        0x01, // type count
        0x60, // func
        0x00, // param count
        0x01, // result count
        0x7F, // i32
    });
    try appendSection(&out, allocator, 0x01, type_payload.items);

    var func_payload: std.ArrayList(u8) = .empty;
    defer func_payload.deinit(allocator);
    try func_payload.appendSlice(allocator, &[_]u8{
        0x01, // function count
        0x00, // type index
    });
    try appendSection(&out, allocator, 0x03, func_payload.items);

    if (extras.memory_min) |min_pages| {
        var memory_payload: std.ArrayList(u8) = .empty;
        defer memory_payload.deinit(allocator);
        try memory_payload.append(allocator, 0x01); // memory count
        try memory_payload.append(allocator, 0x00); // limits: min only
        try encodeULEB128(&memory_payload, allocator, min_pages);
        try appendSection(&out, allocator, 0x05, memory_payload.items);
    }

    var export_payload: std.ArrayList(u8) = .empty;
    defer export_payload.deinit(allocator);
    try export_payload.append(allocator, 0x01); // export count
    try export_payload.append(allocator, 0x03); // name length
    try export_payload.appendSlice(allocator, "run");
    try export_payload.append(allocator, 0x00); // function export
    try export_payload.append(allocator, 0x00); // function index
    try appendSection(&out, allocator, 0x07, export_payload.items);

    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    if (extras.local_i32_count == 0) {
        try body.append(allocator, 0x00); // local decl count
    } else {
        try body.append(allocator, 0x01); // local decl group count
        try encodeULEB128(&body, allocator, extras.local_i32_count);
        try body.append(allocator, 0x7F); // i32
    }
    try body.appendSlice(allocator, instructions);
    try body.append(allocator, 0x0B); // end

    var code_payload: std.ArrayList(u8) = .empty;
    defer code_payload.deinit(allocator);
    try code_payload.append(allocator, 0x01); // function count
    try encodeULEB128(&code_payload, allocator, @intCast(body.items.len));
    try code_payload.appendSlice(allocator, body.items);
    try appendSection(&out, allocator, 0x0A, code_payload.items);

    if (extras.data) |data| {
        var data_payload: std.ArrayList(u8) = .empty;
        defer data_payload.deinit(allocator);
        try data_payload.append(allocator, 0x01); // segment count
        try data_payload.append(allocator, 0x00); // active segment for memory 0
        try data_payload.append(allocator, 0x41); // i32.const
        try encodeSLEB128(&data_payload, allocator, 0);
        try data_payload.append(allocator, 0x0B); // end offset expr
        try encodeULEB128(&data_payload, allocator, @intCast(data.len));
        try data_payload.appendSlice(allocator, data);
        try appendSection(&out, allocator, 0x0B, data_payload.items);
    }

    return out.toOwnedSlice(allocator);
}

fn appendSection(out: *std.ArrayList(u8), allocator: Allocator, id: u8, payload: []const u8) !void {
    try out.append(allocator, id);
    try encodeULEB128(out, allocator, @intCast(payload.len));
    try out.appendSlice(allocator, payload);
}

fn appendV128ConstI32x4(buf: *std.ArrayList(u8), allocator: Allocator, lanes: [4]i32) !void {
    try appendSimdOpcode(buf, allocator, 0x0C); // v128.const
    for (lanes) |lane| {
        var le = std.mem.nativeToLittle(u32, @bitCast(lane));
        try buf.appendSlice(allocator, std.mem.asBytes(&le));
    }
}

fn appendV128ConstI64x2(buf: *std.ArrayList(u8), allocator: Allocator, lanes: [2]u64) !void {
    try appendSimdOpcode(buf, allocator, 0x0C); // v128.const
    for (lanes) |lane| {
        var le = std.mem.nativeToLittle(u64, lane);
        try buf.appendSlice(allocator, std.mem.asBytes(&le));
    }
}

fn appendV128ConstI16x8(buf: *std.ArrayList(u8), allocator: Allocator, lanes: [8]u16) !void {
    try appendSimdOpcode(buf, allocator, 0x0C); // v128.const
    for (lanes) |lane| {
        var le = std.mem.nativeToLittle(u16, lane);
        try buf.appendSlice(allocator, std.mem.asBytes(&le));
    }
}

fn appendV128ConstI8x16(buf: *std.ArrayList(u8), allocator: Allocator, lanes: [16]u8) !void {
    try appendSimdOpcode(buf, allocator, 0x0C); // v128.const
    try buf.appendSlice(allocator, &lanes);
}

fn appendI32x4Splat(buf: *std.ArrayList(u8), allocator: Allocator) !void {
    try appendSimdOpcode(buf, allocator, 0x11); // i32x4.splat
}

fn appendI64x2Splat(buf: *std.ArrayList(u8), allocator: Allocator) !void {
    try appendSimdOpcode(buf, allocator, 0x12); // i64x2.splat
}

fn appendI16x8Splat(buf: *std.ArrayList(u8), allocator: Allocator) !void {
    try appendSimdOpcode(buf, allocator, 0x10); // i16x8.splat
}

fn appendI8x16Splat(buf: *std.ArrayList(u8), allocator: Allocator) !void {
    try appendSimdOpcode(buf, allocator, 0x0F); // i8x16.splat
}

fn appendI32x4ExtractLane(buf: *std.ArrayList(u8), allocator: Allocator, lane: u8) !void {
    try appendSimdOpcode(buf, allocator, 0x1B); // i32x4.extract_lane
    try buf.append(allocator, lane);
}

fn appendI64x2ExtractLane(buf: *std.ArrayList(u8), allocator: Allocator, lane: u8) !void {
    try appendSimdOpcode(buf, allocator, 0x1D); // i64x2.extract_lane
    try buf.append(allocator, lane);
}

fn appendI8x16ExtractLaneS(buf: *std.ArrayList(u8), allocator: Allocator, lane: u8) !void {
    try appendSimdOpcode(buf, allocator, 0x15); // i8x16.extract_lane_s
    try buf.append(allocator, lane);
}

fn appendI8x16ExtractLaneU(buf: *std.ArrayList(u8), allocator: Allocator, lane: u8) !void {
    try appendSimdOpcode(buf, allocator, 0x16); // i8x16.extract_lane_u
    try buf.append(allocator, lane);
}

fn appendI16x8ExtractLaneS(buf: *std.ArrayList(u8), allocator: Allocator, lane: u8) !void {
    try appendSimdOpcode(buf, allocator, 0x18); // i16x8.extract_lane_s
    try buf.append(allocator, lane);
}

fn appendI16x8ExtractLaneU(buf: *std.ArrayList(u8), allocator: Allocator, lane: u8) !void {
    try appendSimdOpcode(buf, allocator, 0x19); // i16x8.extract_lane_u
    try buf.append(allocator, lane);
}

fn appendI32x4ReplaceLane(buf: *std.ArrayList(u8), allocator: Allocator, lane: u8) !void {
    try appendSimdOpcode(buf, allocator, 0x1C); // i32x4.replace_lane
    try buf.append(allocator, lane);
}

fn appendI64x2ReplaceLane(buf: *std.ArrayList(u8), allocator: Allocator, lane: u8) !void {
    try appendSimdOpcode(buf, allocator, 0x1E); // i64x2.replace_lane
    try buf.append(allocator, lane);
}

fn appendI16x8ReplaceLane(buf: *std.ArrayList(u8), allocator: Allocator, lane: u8) !void {
    try appendSimdOpcode(buf, allocator, 0x1A); // i16x8.replace_lane
    try buf.append(allocator, lane);
}

fn appendI8x16ReplaceLane(buf: *std.ArrayList(u8), allocator: Allocator, lane: u8) !void {
    try appendSimdOpcode(buf, allocator, 0x17); // i8x16.replace_lane
    try buf.append(allocator, lane);
}

fn appendI32Const(buf: *std.ArrayList(u8), allocator: Allocator, value: i64) !void {
    try buf.append(allocator, 0x41); // i32.const
    try encodeSLEB128(buf, allocator, value);
}

fn appendI64Const(buf: *std.ArrayList(u8), allocator: Allocator, value: i64) !void {
    try buf.append(allocator, 0x42); // i64.const
    try encodeSLEB128(buf, allocator, value);
}

fn appendI32WrapI64(buf: *std.ArrayList(u8), allocator: Allocator) !void {
    try buf.append(allocator, 0xA7); // i32.wrap_i64
}

fn appendLocalGet(buf: *std.ArrayList(u8), allocator: Allocator, index: u32) !void {
    try buf.append(allocator, 0x20); // local.get
    try encodeULEB128(buf, allocator, index);
}

fn appendLocalSet(buf: *std.ArrayList(u8), allocator: Allocator, index: u32) !void {
    try buf.append(allocator, 0x21); // local.set
    try encodeULEB128(buf, allocator, index);
}

fn appendBlock(buf: *std.ArrayList(u8), allocator: Allocator) !void {
    try buf.appendSlice(allocator, &[_]u8{ 0x02, 0x40 }); // block void
}

fn appendLoop(buf: *std.ArrayList(u8), allocator: Allocator) !void {
    try buf.appendSlice(allocator, &[_]u8{ 0x03, 0x40 }); // loop void
}

fn appendEnd(buf: *std.ArrayList(u8), allocator: Allocator) !void {
    try buf.append(allocator, 0x0B); // end
}

fn appendBr(buf: *std.ArrayList(u8), allocator: Allocator, depth: u32) !void {
    try buf.append(allocator, 0x0C); // br
    try encodeULEB128(buf, allocator, depth);
}

fn appendBrIf(buf: *std.ArrayList(u8), allocator: Allocator, depth: u32) !void {
    try buf.append(allocator, 0x0D); // br_if
    try encodeULEB128(buf, allocator, depth);
}

fn appendI32Add(buf: *std.ArrayList(u8), allocator: Allocator) !void {
    try buf.append(allocator, 0x6A); // i32.add
}

fn appendI32ShrU(buf: *std.ArrayList(u8), allocator: Allocator) !void {
    try buf.append(allocator, 0x76); // i32.shr_u
}

fn appendI32GeU(buf: *std.ArrayList(u8), allocator: Allocator) !void {
    try buf.append(allocator, 0x4F); // i32.ge_u
}

fn appendI32Load(
    buf: *std.ArrayList(u8),
    allocator: Allocator,
    alignment: u32,
    offset: u32,
) !void {
    try buf.append(allocator, 0x28); // i32.load
    try encodeULEB128(buf, allocator, alignment);
    try encodeULEB128(buf, allocator, offset);
}

fn appendI32Store(
    buf: *std.ArrayList(u8),
    allocator: Allocator,
    alignment: u32,
    offset: u32,
) !void {
    try buf.append(allocator, 0x36); // i32.store
    try encodeULEB128(buf, allocator, alignment);
    try encodeULEB128(buf, allocator, offset);
}

fn appendSimdMemOpcode(
    buf: *std.ArrayList(u8),
    allocator: Allocator,
    opcode: u32,
    alignment: u32,
    offset: u32,
) !void {
    try appendSimdOpcode(buf, allocator, opcode);
    try encodeULEB128(buf, allocator, alignment);
    try encodeULEB128(buf, allocator, offset);
}

fn appendSimdOpcode(buf: *std.ArrayList(u8), allocator: Allocator, opcode: u32) !void {
    try buf.append(allocator, 0xFD);
    try encodeULEB128(buf, allocator, opcode);
}

fn writeI32Lane(dst: []u8, value: u32) void {
    var le = std.mem.nativeToLittle(u32, value);
    @memcpy(dst, std.mem.asBytes(&le));
}

fn writeI64Lane(dst: []u8, value: u64) void {
    var le = std.mem.nativeToLittle(u64, value);
    @memcpy(dst, std.mem.asBytes(&le));
}

fn writeI16Lane(dst: []u8, value: u16) void {
    var le = std.mem.nativeToLittle(u16, value);
    @memcpy(dst, std.mem.asBytes(&le));
}

fn encodeULEB128(buf: *std.ArrayList(u8), allocator: Allocator, value: u32) !void {
    var v = value;
    while (true) {
        var byte: u8 = @intCast(v & 0x7F);
        v >>= 7;
        if (v != 0) byte |= 0x80;
        try buf.append(allocator, byte);
        if (v == 0) break;
    }
}

fn encodeSLEB128(buf: *std.ArrayList(u8), allocator: Allocator, value: i64) !void {
    var v = value;
    var more = true;
    while (more) {
        const byte: u8 = @as(u8, @truncate(@as(u64, @bitCast(v)))) & 0x7F;
        v >>= 7;
        const sign_bit = byte & 0x40;
        if ((v == 0 and sign_bit == 0) or (v == -1 and sign_bit != 0)) {
            more = false;
            try buf.append(allocator, byte);
        } else {
            try buf.append(allocator, byte | 0x80);
        }
    }
}
