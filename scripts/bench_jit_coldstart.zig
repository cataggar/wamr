//! Compare AOT, JIT, and Wasmtime cold start and CoreMark throughput.
const std = @import("std");
const process_control = @import("process_control.zig");
const fixed = @import("decimal_format.zig").fixed;
const ExactSum = @import("float_sum.zig").ExactSum;
const Allocator = std.mem.Allocator;

const usage =
    \\Compare cold start and CoreMark throughput across AOT, JIT, and Wasmtime.
    \\Run from the repository with:
    \\  zig run scripts/bench_jit_coldstart.zig -- [options]
    \\
    \\Options:
    \\  --samples N         timed cold-start invocations per mode (default: 20)
    \\  --warmup N          discarded cold-start invocations (default: 3)
    \\  --coremark-runs N   throughput runs per mode (default: 3)
    \\  --skip-coremark     omit throughput measurements
    \\  --wasmtime-path PATH  executable (default: PATH, then ~/.wasmtime/bin)
    \\  --out FILE          also write the Markdown report to FILE
    \\  --repo-root DIR     repository root (default: search from current directory)
    \\  -h, --help          show this help
    \\
;

const Options = struct {
    samples: usize = 20,
    warmup: usize = 3,
    coremark_runs: usize = 3,
    skip_coremark: bool = false,
    wasmtime: ?[]const u8 = null,
    output: ?[]const u8 = null,
    root: ?[]const u8 = null,
};

fn parseOptions(args: []const []const u8) !Options {
    var result: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const raw = args[i];
        if (std.mem.eql(u8, raw, "-h") or std.mem.eql(u8, raw, "--help")) return error.HelpRequested;
        if (std.mem.eql(u8, raw, "--skip-coremark")) {
            result.skip_coremark = true;
            continue;
        }
        const separator = std.mem.indexOfScalar(u8, raw, '=');
        const option = if (separator) |index| raw[0..index] else raw;
        const value = if (separator) |index| raw[index + 1 ..] else blk: {
            i += 1;
            if (i == args.len) return error.MissingOptionValue;
            break :blk args[i];
        };
        if (std.mem.eql(u8, option, "--samples")) {
            result.samples = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, option, "--warmup")) {
            result.warmup = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, option, "--coremark-runs")) {
            result.coremark_runs = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, option, "--wasmtime-path")) {
            result.wasmtime = value;
        } else if (std.mem.eql(u8, option, "--out")) {
            result.output = value;
        } else if (std.mem.eql(u8, option, "--repo-root")) {
            result.root = value;
        } else return error.UnknownOption;
    }
    if (result.samples == 0 or (!result.skip_coremark and result.coremark_runs == 0))
        return error.EmptyMeasurements;
    _ = std.math.add(usize, result.samples, result.warmup) catch return error.TooManyMeasurements;
    return result;
}

const Sample = struct {
    label: []const u8,
    median_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    max_ms: f64,
    n: usize,
};

fn summarize(label: []const u8, values: []f64) !Sample {
    if (values.len == 0) return error.EmptyMeasurements;
    std.mem.sort(f64, values, {}, std.sort.asc(f64));
    const midpoint = values.len / 2;
    const median = if (values.len % 2 == 0) (values[midpoint - 1] + values[midpoint]) / 2 else values[midpoint];
    const product = try std.math.mul(usize, values.len - 1, 19);
    var index = product / 20;
    const remainder = product % 20;
    if (remainder > 10 or (remainder == 10 and index % 2 != 0)) index += 1;
    return .{
        .label = label,
        .median_ms = median,
        .p95_ms = values[index],
        .min_ms = values[0],
        .max_ms = values[values.len - 1],
        .n = values.len,
    };
}

fn coremarkRate(output: []const u8) !f64 {
    const label = "Iterations/Sec";
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, output, from, label)) |start| {
        var pos = start + label.len;
        from = pos;
        while (pos < output.len and std.ascii.isWhitespace(output[pos])) : (pos += 1) {}
        if (pos == output.len or output[pos] != ':') continue;
        pos += 1;
        while (pos < output.len and std.ascii.isWhitespace(output[pos])) : (pos += 1) {}
        const begin = pos;
        while (pos < output.len and std.ascii.isDigit(output[pos])) : (pos += 1) {}
        if (pos == begin) continue;
        if (pos + 1 < output.len and output[pos] == '.' and std.ascii.isDigit(output[pos + 1])) {
            pos += 1;
            while (pos < output.len and std.ascii.isDigit(output[pos])) : (pos += 1) {}
        }
        return std.fmt.parseFloat(f64, output[begin..pos]);
    }
    return error.InvalidCoreMarkOutput;
}

fn runCommand(allocator: Allocator, io: std.Io, cwd: []const u8, env: *const std.process.Environ.Map, args: []const []const u8) ![]u8 {
    const result = try std.process.run(allocator, io, .{ .argv = args, .cwd = .{ .path = cwd }, .environ_map = env });
    defer allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("[harness] command failed ({f}):", .{result.term});
        for (args) |arg| std.debug.print(" {s}", .{arg});
        std.debug.print("\n[harness] --- stdout ---\n{s}\n[harness] --- stderr ---\n{s}\n", .{ result.stdout, result.stderr });
        allocator.free(result.stdout);
        return error.BenchmarkCommandFailed;
    }
    return result.stdout;
}

fn timeCommands(
    allocator: Allocator,
    io: std.Io,
    cwd: []const u8,
    env: *const std.process.Environ.Map,
    label: []const u8,
    batch: []const []const []const u8,
    samples: usize,
    warmup: usize,
) !Sample {
    const durations = try allocator.alloc(f64, samples);
    defer allocator.free(durations);
    for (0..samples + warmup) |i| {
        const start = std.Io.Clock.awake.now(io);
        for (batch) |args| {
            const output = try runCommand(allocator, io, cwd, env, args);
            allocator.free(output);
        }
        const elapsed = start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
        if (i >= warmup) durations[i - warmup] = @as(f64, @floatFromInt(elapsed)) / 1_000_000;
    }
    return summarize(label, durations);
}

const Throughput = struct { label: []const u8, mean: f64, min: f64, max: f64, n: usize };

fn measureCoreMark(
    allocator: Allocator,
    io: std.Io,
    cwd: []const u8,
    env: *const std.process.Environ.Map,
    label: []const u8,
    args: []const []const u8,
    runs: usize,
) !Throughput {
    var total: ExactSum = .{};
    defer total.deinit(allocator);
    var min: f64 = std.math.inf(f64);
    var max: f64 = -std.math.inf(f64);
    for (0..runs) |i| {
        const output = try runCommand(allocator, io, cwd, env, args);
        defer allocator.free(output);
        const value = coremarkRate(output) catch |err| {
            std.debug.print("[harness] could not parse CoreMark output:\n{s}\n", .{output});
            return err;
        };
        const formatted = try fixed(allocator, value, 1);
        defer allocator.free(formatted);
        std.debug.print("[harness]   run {d}/{d}: {s} iter/s\n", .{ i + 1, runs, formatted });
        try total.add(allocator, value);
        min = @min(min, value);
        max = @max(max, value);
    }
    return .{ .label = label, .mean = total.result() / @as(f64, @floatFromInt(runs)), .min = min, .max = max, .n = runs };
}

fn findRoot(allocator: Allocator, io: std.Io, requested: ?[]const u8) ![:0]u8 {
    const cwd = std.Io.Dir.cwd();
    var current = try cwd.realPathFileAlloc(io, requested orelse ".", allocator);
    errdefer allocator.free(current);
    while (true) {
        const marker = try std.fs.path.join(allocator, &.{ current, "build.zig.zon" });
        defer allocator.free(marker);
        if (cwd.access(io, marker, .{})) |_| {
            const fixture = try std.fs.path.join(allocator, &.{ current, "tests/coldstart/noop.wasm" });
            defer allocator.free(fixture);
            if (cwd.access(io, fixture, .{})) |_| return current else |err| switch (err) {
                error.FileNotFound => {},
                else => return err,
            }
        } else |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        }
        if (requested != null) return error.InvalidRepositoryRoot;
        const parent = std.fs.path.dirname(current) orelse return error.RepositoryRootNotFound;
        if (std.mem.eql(u8, current, parent)) return error.RepositoryRootNotFound;
        const next = try allocator.dupeSentinel(u8, parent, 0);
        allocator.free(current);
        current = next;
    }
}

fn findWasmtime(allocator: Allocator, io: std.Io, env: *const std.process.Environ.Map, requested: ?[]const u8) !?[]const u8 {
    if (requested) |value| if (value.len != 0) return value;
    if (env.get("PATH")) |path| {
        var directories = std.mem.splitScalar(u8, path, std.fs.path.delimiter);
        while (directories.next()) |directory| {
            const name = if (@import("builtin").os.tag == .windows) "wasmtime.exe" else "wasmtime";
            const candidate = try std.fs.path.join(allocator, &.{ if (directory.len == 0) "." else directory, name });
            defer allocator.free(candidate);
            if (std.Io.Dir.cwd().access(io, candidate, .{ .execute = true })) |_| {
                const stat = try std.Io.Dir.cwd().statFile(io, candidate, .{});
                if (stat.kind != .directory) return try allocator.dupe(u8, candidate);
            } else |err| switch (err) {
                error.FileNotFound, error.AccessDenied, error.PermissionDenied => {},
                else => return err,
            }
        }
    }
    if (env.get(if (@import("builtin").os.tag == .windows) "USERPROFILE" else "HOME")) |home| {
        const candidate = try std.fs.path.join(allocator, &.{ home, ".wasmtime", "bin", if (@import("builtin").os.tag == .windows) "wasmtime.exe" else "wasmtime" });
        defer allocator.free(candidate);
        if (std.Io.Dir.cwd().access(io, candidate, .{})) |_| return try allocator.dupe(u8, candidate) else |err| switch (err) {
            error.FileNotFound, error.AccessDenied, error.PermissionDenied => {},
            else => return err,
        }
    }
    return null;
}

const Scratch = struct {
    root: std.Io.Dir,
    name: []const u8,
    path: []const u8,

    fn create(allocator: Allocator, io: std.Io, repo: []const u8) !Scratch {
        const cache = try std.fs.path.join(allocator, &.{ repo, ".zig-cache" });
        try std.Io.Dir.cwd().createDirPath(io, cache);
        const root = try std.Io.Dir.cwd().openDir(io, cache, .{});
        errdefer root.close(io);
        for (0..10) |_| {
            var random: [16]u8 = undefined;
            io.random(&random);
            const name = try std.fmt.allocPrint(allocator, "bench-jit-coldstart-{s}", .{std.fmt.bytesToHex(random, .lower)});
            root.createDir(io, name, if (@import("builtin").os.tag == .windows) .default_dir else .fromMode(0o700)) catch |err| switch (err) {
                error.PathAlreadyExists => continue,
                else => return err,
            };
            errdefer root.deleteTree(io, name) catch |err| std.debug.print("[harness] cleanup failed: {s}\n", .{@errorName(err)});
            return .{ .root = root, .name = name, .path = try std.fs.path.join(allocator, &.{ cache, name }) };
        }
        return error.TemporaryNameCollisions;
    }

    fn destroy(self: Scratch, io: std.Io) !void {
        defer self.root.close(io);
        try self.root.deleteTree(io, self.name);
    }
};

fn probe(allocator: Allocator, io: std.Io, args: []const []const u8) !?[]u8 {
    const result = process_control.runTimed(allocator, io, .{ .argv = args }, 5) catch |err| switch (err) {
        error.FileNotFound, error.Timeout => {
            std.debug.print("[harness] metadata probe {s}: {s}\n", .{ args[0], @errorName(err) });
            return null;
        },
        else => return err,
    };
    defer allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("[harness] metadata probe {s} failed ({f}): {s}\n", .{ args[0], result.term, result.stderr });
        allocator.free(result.stdout);
        return null;
    }
    return result.stdout;
}

const SystemTime = extern struct {
    year: u16,
    month: u16,
    day_of_week: u16,
    day: u16,
    hour: u16,
    minute: u16,
    second: u16,
    milliseconds: u16,
};
extern "kernel32" fn GetLocalTime(*SystemTime) callconv(.winapi) void;

const Host = struct { date: []const u8, arch: []const u8, cpu_count: []const u8, model: []const u8 };

fn hostInfo(allocator: Allocator, io: std.Io, env: *const std.process.Environ.Map) !Host {
    if (@import("builtin").os.tag == .windows) {
        var date: SystemTime = undefined;
        GetLocalTime(&date);
        return .{
            .date = try std.fmt.allocPrint(allocator, "{d:0>4}-{d:0>2}-{d:0>2}", .{ date.year, date.month, date.day }),
            .arch = env.get("PROCESSOR_ARCHITECTURE") orelse "unknown-arch",
            .cpu_count = if (std.Thread.getCpuCount()) |count| try std.fmt.allocPrint(allocator, "{d}", .{count}) else |_| "?",
            .model = env.get("PROCESSOR_IDENTIFIER") orelse "unknown CPU",
        };
    }
    const date_raw = (try probe(allocator, io, &.{ "date", "+%F" })) orelse return error.LocalDateUnavailable;
    const machine = try probe(allocator, io, &.{ "uname", "-m" });
    const cpus = try probe(allocator, io, &.{ "getconf", "_NPROCESSORS_ONLN" });
    const lscpu = try probe(allocator, io, &.{"lscpu"});
    var model: ?[]const u8 = null;
    if (lscpu) |output| {
        var lines = std.mem.splitScalar(u8, output, '\n');
        while (lines.next()) |line| {
            const label = "model name:";
            if (line.len >= label.len and std.ascii.eqlIgnoreCase(line[0..label.len], label)) {
                const text = std.mem.trim(u8, line[label.len..], " \t\r\n");
                if (text.len != 0) model = text;
                break;
            }
        }
    }
    if (model == null) {
        const processor = try probe(allocator, io, &.{ "uname", "-p" });
        if (processor) |value| {
            const text = std.mem.trim(u8, value, " \t\r\n");
            if (text.len != 0) model = text;
        }
    }
    const count = if (cpus) |value| std.mem.trim(u8, value, " \t\r\n") else "?";
    return .{
        .date = std.mem.trim(u8, date_raw, " \t\r\n"),
        .arch = if (machine) |value| std.mem.trim(u8, value, " \t\r\n") else "unknown-arch",
        .cpu_count = if (count.len == 0) "?" else count,
        .model = model orelse "unknown CPU",
    };
}

fn appendFormat(allocator: Allocator, buffer: *std.ArrayList(u8), comptime format: []const u8, args: anytype) !void {
    const text = try std.fmt.allocPrint(allocator, format, args);
    defer allocator.free(text);
    try buffer.appendSlice(allocator, text);
}

fn renderReport(allocator: Allocator, host: Host, samples: []const Sample, rates: ?[]const Throughput) ![]u8 {
    var output: std.ArrayList(u8) = .empty;
    errdefer output.deinit(allocator);
    try appendFormat(allocator, &output,
        \\# JIT cold-start comparison — {s}
        \\
        \\_Host: arch `{s}` · {s} vCPU · `{s}`_
        \\
        \\## Cold-start (noop module, subprocess wall time)
        \\
        \\| Mode | Median | p95 | Min | Max | Samples |
        \\|---|---:|---:|---:|---:|---:|
        \\
    , .{ host.date, host.arch, host.cpu_count, host.model });
    for (samples) |sample| {
        const median = try fixed(allocator, sample.median_ms, 3);
        defer allocator.free(median);
        const p95 = try fixed(allocator, sample.p95_ms, 3);
        defer allocator.free(p95);
        const min = try fixed(allocator, sample.min_ms, 3);
        defer allocator.free(min);
        const max = try fixed(allocator, sample.max_ms, 3);
        defer allocator.free(max);
        try appendFormat(allocator, &output, "| {s} | {s} ms | {s} ms | {s} ms | {s} ms | {d} |\n", .{
            sample.label, median, p95, min, max, sample.n,
        });
    }
    try output.append(allocator, '\n');
    if (rates) |values| {
        try output.appendSlice(allocator,
            \\## Steady-state throughput (CoreMark)
            \\
            \\| Mode | Mean iter/s | Min | Max | Runs |
            \\|---|---:|---:|---:|---:|
            \\
        );
        for (values) |row| {
            const mean = try fixed(allocator, row.mean, 1);
            defer allocator.free(mean);
            const min = try fixed(allocator, row.min, 1);
            defer allocator.free(min);
            const max = try fixed(allocator, row.max, 1);
            defer allocator.free(max);
            try appendFormat(allocator, &output, "| {s} | {s} | {s} | {s} | {d} |\n", .{ row.label, mean, min, max, row.n });
        }
        try output.append(allocator, '\n');
    }
    return output.toOwnedSlice(allocator);
}

fn benchmark(init: std.process.Init, options: Options) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const root = try findRoot(allocator, io, options.root);
    const wasmtime = try findWasmtime(allocator, io, init.environ_map, options.wasmtime);
    if (wasmtime == null) std.debug.print("[harness] warning: wasmtime not found; wasmtime columns will be omitted\n", .{});
    var env = try init.environ_map.clone(allocator);
    defer env.deinit();
    _ = env.swapRemove("ZIG_LOCAL_CACHE_DIR");
    const cache = try std.fs.path.join(allocator, &.{ root, ".zig-global-cache" });
    try cwd.createDirPath(io, cache);
    try env.put("ZIG_GLOBAL_CACHE_DIR", cache);
    var full_env = try env.clone(allocator);
    defer full_env.deinit();
    try full_env.put("WAMR_JIT_FULL_OPT", "1");

    std.debug.print("[harness] building wamr/wamrc (-Djit=true -Doptimize=fast)\n", .{});
    const build_output = try runCommand(init.gpa, io, root, &env, &.{ "zig", "build", "-Djit=true", "-Doptimize=fast" });
    init.gpa.free(build_output);
    const ext = if (@import("builtin").os.tag == .windows) ".exe" else "";
    const wamr = try std.fmt.allocPrint(allocator, "{s}/zig-out/bin/wamr{s}", .{ root, ext });
    const wamrc = try std.fmt.allocPrint(allocator, "{s}/zig-out/bin/wamrc{s}", .{ root, ext });
    try cwd.access(io, wamr, .{});
    try cwd.access(io, wamrc, .{});

    const scratch = try Scratch.create(allocator, io, root);
    var cleanup_pending = true;
    defer if (cleanup_pending) scratch.destroy(io) catch |err| std.debug.print("[harness] cleanup failed: {s}\n", .{@errorName(err)});
    const noop_source = try std.fs.path.join(allocator, &.{ root, "tests/coldstart/noop.wasm" });
    const noop = try std.fs.path.join(allocator, &.{ scratch.path, "noop.wasm" });
    const compiled = try std.fs.path.join(allocator, &.{ scratch.path, "noop.cwasm" });
    const cold = try std.fs.path.join(allocator, &.{ scratch.path, "noop_cold.cwasm" });
    try cwd.copyFile(noop_source, cwd, noop, io, .{});
    const prepare_output = try runCommand(init.gpa, io, scratch.path, &env, &.{ wamrc, "compile", noop, "-o", compiled });
    init.gpa.free(prepare_output);
    std.debug.print("[harness] timing cold-start ({d} samples, {d} warmup)...\n", .{ options.samples, options.warmup });
    var samples: std.ArrayList(Sample) = .empty;
    try samples.append(allocator, try timeCommands(init.gpa, io, scratch.path, &env, "AOT cold (wamrc compile + wamr run, no cache)", &.{
        &.{ wamrc, "compile", noop, "-o", cold }, &.{ wamr, "run", cold },
    }, options.samples, options.warmup));
    try samples.append(allocator, try timeCommands(init.gpa, io, scratch.path, &env, "AOT warm (wamr run, precompiled .cwasm)", &.{
        &.{ wamr, "run", compiled },
    }, options.samples, options.warmup));
    try samples.append(allocator, try timeCommands(init.gpa, io, scratch.path, &env, "wamrc run (one-step compile+execute, no persistent cache)", &.{
        &.{ wamrc, "run", noop },
    }, options.samples, options.warmup));
    try samples.append(allocator, try timeCommands(init.gpa, io, scratch.path, &env, "in-process JIT, .fast preset (default)", &.{
        &.{ wamr, "run", noop },
    }, options.samples, options.warmup));
    try samples.append(allocator, try timeCommands(init.gpa, io, scratch.path, &full_env, "in-process JIT, .full preset (WAMR_JIT_FULL_OPT=1)", &.{
        &.{ wamr, "run", noop },
    }, options.samples, options.warmup));
    if (wasmtime) |exe| try samples.append(allocator, try timeCommands(init.gpa, io, scratch.path, &env, "wasmtime run (default JIT)", &.{
        &.{ exe, "run", noop },
    }, options.samples, options.warmup));

    var rates: std.ArrayList(Throughput) = .empty;
    if (!options.skip_coremark) {
        std.debug.print("[harness] timing CoreMark throughput ({d} runs per mode)...\n", .{options.coremark_runs});
        const wasm = try std.fs.path.join(allocator, &.{ root, "tests/benchmarks/coremark/coremark_wasi.wasm" });
        const artifact = try std.fs.path.join(allocator, &.{ scratch.path, "coremark.cwasm" });
        const output = try runCommand(init.gpa, io, scratch.path, &env, &.{ wamrc, "compile", wasm, "-o", artifact });
        init.gpa.free(output);
        try rates.append(allocator, try measureCoreMark(init.gpa, io, scratch.path, &env, "AOT (precompiled .cwasm)", &.{ wamr, "run", artifact }, options.coremark_runs));
        try rates.append(allocator, try measureCoreMark(init.gpa, io, scratch.path, &env, ".fast preset (in-process JIT default)", &.{ wamr, "run", wasm }, options.coremark_runs));
        try rates.append(allocator, try measureCoreMark(init.gpa, io, scratch.path, &full_env, ".full preset (in-process JIT, WAMR_JIT_FULL_OPT=1)", &.{ wamr, "run", wasm }, options.coremark_runs));
        if (wasmtime) |exe| try rates.append(allocator, try measureCoreMark(init.gpa, io, scratch.path, &env, "wasmtime (default JIT)", &.{ exe, "run", wasm }, options.coremark_runs));
    }
    const host = try hostInfo(allocator, io, &env);
    const report = try renderReport(allocator, host, samples.items, if (options.skip_coremark) null else rates.items);
    try std.Io.File.stdout().writeStreamingAll(io, report);
    if (options.output) |path| try cwd.writeFile(io, .{ .sub_path = path, .data = report });
    cleanup_pending = false;
    try scratch.destroy(io);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parseOptions(args[1..]) catch |err| {
        if (err == error.HelpRequested) {
            try std.Io.File.stdout().writeStreamingAll(init.io, usage);
            return;
        }
        std.debug.print("error: {s}\n{s}", .{ @errorName(err), usage });
        std.process.exit(2);
    };
    benchmark(init, options) catch |err| {
        std.debug.print("[harness] error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

test "benchmark options reject empty samples and retain explicit paths" {
    const options = try parseOptions(&.{ "--samples=11", "--warmup", "2", "--skip-coremark", "--coremark-runs", "0", "--wasmtime-path", "/tools/wasmtime", "--repo-root", "/repo" });
    try std.testing.expectEqual(@as(usize, 11), options.samples);
    try std.testing.expectEqual(@as(usize, 2), options.warmup);
    try std.testing.expectEqualStrings("/tools/wasmtime", options.wasmtime.?);
    try std.testing.expectError(error.EmptyMeasurements, parseOptions(&.{ "--samples", "0" }));
    try std.testing.expectError(error.EmptyMeasurements, parseOptions(&.{ "--coremark-runs", "0" }));
    try std.testing.expectError(error.MissingOptionValue, parseOptions(&.{"--out"}));
}

test "median and percentile preserve the benchmark's rounding rule" {
    var odd = [_]f64{ 5, 1, 3, 2, 4 };
    const a = try summarize("mode", &odd);
    try std.testing.expectEqual(@as(f64, 3), a.median_ms);
    try std.testing.expectEqual(@as(f64, 5), a.p95_ms);
    var even = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    const b = try summarize("mode", &even);
    try std.testing.expectEqual(@as(f64, 6), b.median_ms);
    try std.testing.expectEqual(@as(f64, 11), b.p95_ms);
    var tie: [31]f64 = undefined;
    for (&tie, 0..) |*value, index| value.* = @floatFromInt(index);
    try std.testing.expectEqual(@as(f64, 28), (try summarize("mode", &tie)).p95_ms);
    try std.testing.expectError(error.EmptyMeasurements, summarize("mode", &.{}));
}

test "CoreMark throughput accepts a numeric result and rejects absent values" {
    try std.testing.expectEqual(@as(f64, 1234.5), try coremarkRate("ignore\nIterations/Sec : 1234.5\n"));
    try std.testing.expectEqual(@as(f64, 42), try coremarkRate("Iterations/Sec: invalid\nIterations/Sec: 42\n"));
    try std.testing.expectError(error.InvalidCoreMarkOutput, coremarkRate("Iterations/Sec: NaN"));
}

test "CoreMark mean uses accurately rounded summation" {
    var sum: ExactSum = .{};
    defer sum.deinit(std.testing.allocator);
    for ([_]f64{ 8153.61, 8412.06, 6939.78 }) |value|
        try sum.add(std.testing.allocator, value);
    const formatted = try fixed(std.testing.allocator, sum.result() / 3, 1);
    defer std.testing.allocator.free(formatted);
    try std.testing.expectEqualStrings("7835.1", formatted);
}

test "fixed decimal rendering rounds the binary value to nearest even" {
    const allocator = std.testing.allocator;
    for ([_]struct { value: f64, expected: []const u8 }{
        .{ .value = 2.675, .expected = "2.67" },
        .{ .value = 1.125, .expected = "1.12" },
        .{ .value = 1.375, .expected = "1.38" },
        .{ .value = 0.0, .expected = "0.00" },
    }) |case| {
        const actual = try fixed(allocator, case.value, 2);
        defer allocator.free(actual);
        try std.testing.expectEqualStrings(case.expected, actual);
    }
}

test "benchmark runs all warmups and samples and propagates command failures" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(path);
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    const sample = try timeCommands(allocator, io, path, &env, "fixture", &.{
        &.{ "/bin/sh", "-c", "printf x >> count" },
    }, 3, 2);
    try std.testing.expectEqual(@as(usize, 3), sample.n);
    try std.testing.expect(sample.min_ms <= sample.median_ms and sample.median_ms <= sample.max_ms);
    const contents = try tmp.dir.readFileAlloc(io, "count", allocator, .unlimited);
    defer allocator.free(contents);
    try std.testing.expectEqualStrings("xxxxx", contents);
    try std.testing.expectError(error.BenchmarkCommandFailed, runCommand(allocator, io, path, &env, &.{ "/bin/sh", "-c", "exit 7" }));
    const throughput = try measureCoreMark(allocator, io, path, &env, "fixture", &.{ "/bin/sh", "-c", "printf 'Iterations/Sec : 1234.5\\n'" }, 2);
    try std.testing.expectEqual(@as(f64, 1234.5), throughput.mean);
}

test "rendered report retains benchmark headings, units, and rows" {
    const report = try renderReport(std.testing.allocator, .{
        .date = "2026-07-10",
        .arch = "x86_64",
        .cpu_count = "4",
        .model = "example",
    }, &.{.{
        .label = "AOT warm",
        .median_ms = 1.25,
        .p95_ms = 2,
        .min_ms = 1,
        .max_ms = 3,
        .n = 2,
    }}, &.{.{
        .label = "AOT",
        .mean = 10.25,
        .min = 9,
        .max = 11,
        .n = 2,
    }});
    defer std.testing.allocator.free(report);
    try std.testing.expectEqualStrings(
        \\# JIT cold-start comparison — 2026-07-10
        \\
        \\_Host: arch `x86_64` · 4 vCPU · `example`_
        \\
        \\## Cold-start (noop module, subprocess wall time)
        \\
        \\| Mode | Median | p95 | Min | Max | Samples |
        \\|---|---:|---:|---:|---:|---:|
        \\| AOT warm | 1.250 ms | 2.000 ms | 1.000 ms | 3.000 ms | 2 |
        \\
        \\## Steady-state throughput (CoreMark)
        \\
        \\| Mode | Mean iter/s | Min | Max | Runs |
        \\|---|---:|---:|---:|---:|
        \\| AOT | 10.2 | 9.0 | 11.0 | 2 |
        \\
        \\
    , report);
}
