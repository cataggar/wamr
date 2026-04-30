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

fn appendI32x4Splat(buf: *std.ArrayList(u8), allocator: Allocator) !void {
    try appendSimdOpcode(buf, allocator, 0x11); // i32x4.splat
}

fn appendI32x4ExtractLane(buf: *std.ArrayList(u8), allocator: Allocator, lane: u8) !void {
    try appendSimdOpcode(buf, allocator, 0x1B); // i32x4.extract_lane
    try buf.append(allocator, lane);
}

fn appendI32x4ReplaceLane(buf: *std.ArrayList(u8), allocator: Allocator, lane: u8) !void {
    try appendSimdOpcode(buf, allocator, 0x1C); // i32x4.replace_lane
    try buf.append(allocator, lane);
}

fn appendI32Const(buf: *std.ArrayList(u8), allocator: Allocator, value: i64) !void {
    try buf.append(allocator, 0x41); // i32.const
    try encodeSLEB128(buf, allocator, value);
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
