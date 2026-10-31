//! Benchmark `wamr run --allow-precompiled` cold-start time against a baseline ref and Wasmtime.
const std = @import("std");
const optimize = @import("benchmark_optimize.zig");
const process = @import("benchmark_process.zig");
const fixed = @import("decimal_format.zig").fixed;
const integer_median = @import("integer_median.zig");
const temp_root = @import("temp_root.zig");
const Allocator = std.mem.Allocator;

var interrupted: std.atomic.Value(bool) = .init(false);

fn onSignal(_: std.posix.SIG) callconv(.c) void {
    interrupted.store(true, .release);
}

const usage =
    \\usage: bench-coldstart [--baseline REF] [--target REF] [--samples N]
    \\  [--warmup N] [--wasmtime] [--wasmtime-path PATH] [--include-jit]
    \\  [--budget-ms MS] [--regression-ratio RATIO] [--json PATH] [--out FILE]
    \\  [--emit markdown|github] [--repo DIR]
    \\  [--optimize ReleaseFast|ReleaseSafe|both]
    \\
;

pub const Tools = struct {
    git: []const u8 = "git",
    zig: []const u8 = "zig",
    zig016: []const u8 = "zig016",
    wasm_tools: ?[]const u8 = null,
};

pub const Options = struct {
    baseline: []const u8 = "origin/main",
    target: []const u8 = "HEAD",
    samples: usize = 30,
    warmup: usize = 3,
    wasmtime: bool = false,
    wasmtime_path: ?[]const u8 = null,
    include_jit: bool = false,
    budget_ms: ?f64 = null,
    regression_ratio: f64 = 1.5,
    json_path: ?[]const u8 = null,
    output: ?[]const u8 = null,
    emit: []const u8 = "markdown",
    repo: ?[]const u8 = null,
    mode: []const u8 = "ReleaseFast",
};

fn parseInteger(raw: []const u8, allow_zero: bool) !usize {
    const text = std.mem.trim(u8, raw, " \t\r\n\x0b\x0c");
    if (text.len == 0) return error.InvalidInteger;
    for (text, 0..) |byte, index| {
        if (byte >= 0x80) return error.InvalidInteger;
        if (byte == '_' and (index == 0 or index + 1 == text.len or
            !std.ascii.isDigit(text[index - 1]) or !std.ascii.isDigit(text[index + 1])))
            return error.InvalidInteger;
    }
    const value = std.fmt.parseInt(i128, text, 10) catch return error.InvalidInteger;
    if (allow_zero) {
        if (value < 0) return error.NegativeCount;
    } else if (value <= 0) {
        return error.NonpositiveCount;
    }
    return std.math.cast(usize, value) orelse error.IntegerTooLarge;
}

fn parsePositiveFloat(raw: []const u8) !f64 {
    const text = std.mem.trim(u8, raw, " \t\r\n\x0b\x0c");
    if (text.len == 0) return error.InvalidFloat;
    for (text, 0..) |byte, index| {
        if (byte >= 0x80) return error.InvalidFloat;
        if (byte == '_' and (index == 0 or index + 1 == text.len or
            !std.ascii.isDigit(text[index - 1]) or !std.ascii.isDigit(text[index + 1])))
            return error.InvalidFloat;
    }
    var clean: [128]u8 = undefined;
    var len: usize = 0;
    for (text) |byte| {
        if (byte == '_') continue;
        if (len >= clean.len) return error.InvalidFloat;
        clean[len] = byte;
        len += 1;
    }
    const value = std.fmt.parseFloat(f64, clean[0..len]) catch return error.InvalidFloat;
    if (!std.math.isFinite(value) or value <= 0) return error.NonpositiveFloat;
    return value;
}

pub fn parseOptions(args: []const []const u8) !Options {
    var result: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) return error.HelpRequested;
        if (std.mem.eql(u8, arg, "--wasmtime")) {
            result.wasmtime = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--include-jit")) {
            result.include_jit = true;
            continue;
        }
        const equal = std.mem.indexOfScalar(u8, arg, '=');
        const name = if (equal) |at| arg[0..at] else arg;
        const value = if (equal) |at| arg[at + 1 ..] else blk: {
            i += 1;
            if (i == args.len) return error.MissingOptionValue;
            break :blk args[i];
        };
        if (std.mem.eql(u8, name, "--baseline")) {
            result.baseline = value;
        } else if (std.mem.eql(u8, name, "--target")) {
            result.target = value;
        } else if (std.mem.eql(u8, name, "--samples")) {
            result.samples = try parseInteger(value, false);
        } else if (std.mem.eql(u8, name, "--warmup")) {
            result.warmup = try parseInteger(value, true);
        } else if (std.mem.eql(u8, name, "--wasmtime-path")) {
            result.wasmtime_path = value;
        } else if (std.mem.eql(u8, name, "--budget-ms")) {
            result.budget_ms = try parsePositiveFloat(value);
        } else if (std.mem.eql(u8, name, "--regression-ratio")) {
            result.regression_ratio = try parsePositiveFloat(value);
        } else if (std.mem.eql(u8, name, "--json")) {
            result.json_path = value;
        } else if (std.mem.eql(u8, name, "--out")) {
            result.output = value;
        } else if (std.mem.eql(u8, name, "--emit")) {
            result.emit = value;
        } else if (std.mem.eql(u8, name, "--repo")) {
            result.repo = value;
        } else if (std.mem.eql(u8, name, "--optimize")) {
            result.mode = value;
        } else return error.UnknownOption;
    }
    _ = try optimize.parseModes(result.mode);
    if (!std.mem.eql(u8, result.emit, "markdown") and !std.mem.eql(u8, result.emit, "github"))
        return error.InvalidOutputMode;
    _ = std.math.add(usize, result.samples, result.warmup) catch return error.IntegerTooLarge;
    return result;
}

pub const TimingStats = struct {
    label: []const u8,
    engine: []const u8,
    module: []const u8,
    samples: usize,
    warmup: usize,
    median_ns: u64,
    min_ns: u64,
    max_ns: u64,
    p95_ns: u64,
    stdev_ns: u64,
    raw_ns: []const u64,

    pub fn medianMs(self: TimingStats) f64 {
        return @as(f64, @floatFromInt(self.median_ns)) / 1e6;
    }

    pub fn minMs(self: TimingStats) f64 {
        return @as(f64, @floatFromInt(self.min_ns)) / 1e6;
    }

    pub fn maxMs(self: TimingStats) f64 {
        return @as(f64, @floatFromInt(self.max_ns)) / 1e6;
    }

    pub fn p95Ms(self: TimingStats) f64 {
        return @as(f64, @floatFromInt(self.p95_ns)) / 1e6;
    }

    pub fn stdevMs(self: TimingStats) f64 {
        return @as(f64, @floatFromInt(self.stdev_ns)) / 1e6;
    }
};

pub fn roundHalfToEven(value: f64) usize {
    if (value <= 0) return 0;
    const floored = @floor(value);
    const diff = value - floored;
    const base: usize = @intFromFloat(floored);
    if (diff > 0.5) return base + 1;
    if (diff < 0.5) return base;
    return if (base % 2 != 0) base + 1 else base;
}

pub fn sampleStdevNs(allocator: Allocator, values: []const u64) !u64 {
    if (values.len <= 1) return 0;
    var n_big: std.math.big.int.Managed = try .initSet(allocator, values.len);
    defer n_big.deinit();
    var n_minus_one: std.math.big.int.Managed = try .initSet(allocator, values.len - 1);
    defer n_minus_one.deinit();
    var sum: std.math.big.int.Managed = try .initSet(allocator, 0);
    defer sum.deinit();
    var sum_sq: std.math.big.int.Managed = try .initSet(allocator, 0);
    defer sum_sq.deinit();
    var term: std.math.big.int.Managed = try .init(allocator);
    defer term.deinit();
    var sq: std.math.big.int.Managed = try .init(allocator);
    defer sq.deinit();

    for (values) |v| {
        try term.set(v);
        try sum.add(&sum, &term);
        try sq.mul(&term, &term);
        try sum_sq.add(&sum_sq, &sq);
    }

    var numerator: std.math.big.int.Managed = try .init(allocator);
    defer numerator.deinit();
    try numerator.mul(&n_big, &sum_sq);
    try sq.mul(&sum, &sum);
    try numerator.sub(&numerator, &sq);

    var denominator: std.math.big.int.Managed = try .init(allocator);
    defer denominator.deinit();
    try denominator.mul(&n_big, &n_minus_one);

    const num_f = numerator.toFloat(f64, .nearest_even)[0];
    const den_f = denominator.toFloat(f64, .nearest_even)[0];
    const variance = num_f / den_f;
    if (!std.math.isFinite(variance) or variance <= 0) return 0;
    return @intFromFloat(@sqrt(variance));
}

pub fn summarizeRawNs(
    allocator: Allocator,
    label: []const u8,
    engine: []const u8,
    module_name: []const u8,
    warmup: usize,
    raw: []const u64,
) !TimingStats {
    if (raw.len == 0) return error.EmptyMeasurements;
    const raw_copy = try allocator.dupe(u64, raw);
    const sorted = try allocator.dupe(u64, raw);
    defer allocator.free(sorted);
    std.mem.sort(u64, sorted, {}, std.sort.asc(u64));
    const mid = sorted.len / 2;
    const median_ns: u64 = if (sorted.len % 2 == 1)
        sorted[mid]
    else
        @intFromFloat(integer_median.average(@intCast(sorted[mid - 1]), @intCast(sorted[mid])));
    const p95_idx = @min(sorted.len - 1, roundHalfToEven(0.95 * @as(f64, @floatFromInt(sorted.len - 1))));
    const stdev_ns = try sampleStdevNs(allocator, sorted);
    return .{
        .label = label,
        .engine = engine,
        .module = module_name,
        .samples = raw.len,
        .warmup = warmup,
        .median_ns = median_ns,
        .min_ns = sorted[0],
        .max_ns = sorted[sorted.len - 1],
        .p95_ns = sorted[p95_idx],
        .stdev_ns = stdev_ns,
        .raw_ns = raw_copy,
    };
}

fn isExecutableFile(io: std.Io, path: []const u8) bool {
    const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    if (stat.kind == .directory) return false;
    if (@import("builtin").os.tag == .windows) return true;
    std.Io.Dir.cwd().access(io, path, .{ .execute = true }) catch return false;
    return true;
}

fn findInPath(allocator: Allocator, io: std.Io, env: *const std.process.Environ.Map, name: []const u8) !?[]const u8 {
    const path_env = env.get("PATH") orelse return null;
    const exe_name = if (@import("builtin").os.tag == .windows and !std.mem.endsWith(u8, name, ".exe"))
        try std.fmt.allocPrint(allocator, "{s}.exe", .{name})
    else
        name;
    var iter = std.mem.splitScalar(u8, path_env, std.fs.path.delimiter);
    while (iter.next()) |dir| {
        const candidate = try std.fs.path.join(allocator, &.{ if (dir.len == 0) "." else dir, exe_name });
        if (isExecutableFile(io, candidate)) return candidate;
        allocator.free(candidate);
    }
    return null;
}

pub fn resolveWasmtime(
    allocator: Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    explicit_path: ?[]const u8,
) !?[]const u8 {
    if (explicit_path) |path| {
        if (!isExecutableFile(io, path)) {
            std.debug.print("error: --wasmtime-path '{s}' is not an executable file\n", .{path});
            return error.InvalidWasmtimePath;
        }
        return try std.Io.Dir.cwd().realPathFileAlloc(io, path, allocator);
    }
    if (try findInPath(allocator, io, env, "wasmtime")) |found| {
        defer allocator.free(found);
        return try std.Io.Dir.cwd().realPathFileAlloc(io, found, allocator);
    }
    const home_var = if (@import("builtin").os.tag == .windows) "USERPROFILE" else "HOME";
    if (env.get(home_var)) |home| {
        const exe_name = if (@import("builtin").os.tag == .windows) "wasmtime.exe" else "wasmtime";
        const candidate = try std.fs.path.join(allocator, &.{ home, ".wasmtime", "bin", exe_name });
        defer allocator.free(candidate);
        if (isExecutableFile(io, candidate)) {
            return try std.Io.Dir.cwd().realPathFileAlloc(io, candidate, allocator);
        }
    }
    return null;
}

pub const WorktreeBuild = struct {
    sha: []const u8,
    path: []const u8,
    wamr: []const u8,
    wamrc: []const u8,
};

pub fn removeWorktree(
    allocator: Allocator,
    io: std.Io,
    repo: []const u8,
    tools: Tools,
    wt_path: []const u8,
) !void {
    const stop: std.atomic.Value(bool) = .init(false);
    const out = try process.run(allocator, io, repo, null, &.{ tools.git, "worktree", "remove", "--force", wt_path }, &stop);
    allocator.free(out);
}

pub fn buildWorktree(
    allocator: Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    repo: []const u8,
    work_root: []const u8,
    label: []const u8,
    ref: []const u8,
    mode: []const u8,
    tools: Tools,
    stop: *const std.atomic.Value(bool),
) !WorktreeBuild {
    const rev_out = try process.run(allocator, io, repo, null, &.{ tools.git, "rev-parse", "--verify", ref }, stop);
    defer allocator.free(rev_out);
    const sha = try allocator.dupe(u8, std.mem.trim(u8, rev_out, " \t\r\n"));
    if (sha.len != 40 and sha.len != 64) return error.InvalidRevision;
    for (sha) |byte| if (!std.ascii.isHex(byte)) return error.InvalidRevision;
    const sha_short = sha[0..@min(12, sha.len)];
    const wt_name = try std.fmt.allocPrint(allocator, "{s}-{s}", .{ label, sha_short });
    const wt_path = try std.fs.path.join(allocator, &.{ work_root, wt_name });

    std.debug.print("[harness] creating worktree for {s} ({s} -> {s})\n", .{ label, ref, sha_short });
    const add_out = try process.run(allocator, io, repo, null, &.{ tools.git, "worktree", "add", "--detach", wt_path, sha }, stop);
    allocator.free(add_out);
    errdefer removeWorktree(allocator, io, repo, tools, wt_path) catch |err|
        std.debug.print("[harness] warning: failed to remove worktree {s}: {s}\n", .{ wt_path, @errorName(err) });

    var zig_env = try base_env.clone(allocator);
    defer zig_env.deinit();
    _ = zig_env.swapRemove("ZIG_LOCAL_CACHE_DIR");
    const global_cache = try std.fs.path.join(allocator, &.{ wt_path, ".zig-global-cache" });
    try std.Io.Dir.cwd().createDirPath(io, global_cache);
    try zig_env.put("ZIG_GLOBAL_CACHE_DIR", global_cache);

    var cmd = try optimize.buildCommand(allocator, io, wt_path, mode);
    cmd[0] = if (std.mem.eql(u8, cmd[0], "zig016")) tools.zig016 else tools.zig;
    std.debug.print("[harness] building wamr + wamrc in {s} ({s}: {s} {s} {s})\n", .{ label, mode, cmd[0], cmd[1], cmd[2] });
    const build_out = try process.run(allocator, io, wt_path, &zig_env, &cmd, stop);
    allocator.free(build_out);

    const wamr_name = if (@import("builtin").os.tag == .windows) "zig-out/bin/wamr.exe" else "zig-out/bin/wamr";
    const wamrc_name = if (@import("builtin").os.tag == .windows) "zig-out/bin/wamrc.exe" else "zig-out/bin/wamrc";
    const wamr = try std.fs.path.join(allocator, &.{ wt_path, wamr_name });
    const wamrc = try std.fs.path.join(allocator, &.{ wt_path, wamrc_name });
    if (!isExecutableFile(io, wamr)) return error.MissingBuiltBinary;
    if (!isExecutableFile(io, wamrc)) return error.MissingBuiltBinary;
    return .{ .sha = sha, .path = wt_path, .wamr = wamr, .wamrc = wamrc };
}

pub const PreparedModule = struct {
    name: []const u8,
    wasm: []const u8,
    wamr_cwasm: []const u8,
    wasmtime_cwasm: ?[]const u8,
};

fn buildAdd100Wat(allocator: Allocator) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    const w = &out.writer;
    try w.writeAll("(module\n");
    for (0..100) |i| {
        try w.print("  (func $f{d} (param i32) (result i32) (i32.add (local.get 0) (i32.const 1)))\n", .{i});
    }
    try w.writeAll("  (func (export \"_start\") (drop (call $f0 (i32.const 0))))\n");
    try w.writeAll(")\n");
    return out.toOwnedSlice();
}

pub fn prepareModules(
    allocator: Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    repo: []const u8,
    work_dir: []const u8,
    wamrc_bin: []const u8,
    wasmtime_bin: ?[]const u8,
    tools: Tools,
    stop: *const std.atomic.Value(bool),
) ![]const PreparedModule {
    try std.Io.Dir.cwd().createDirPath(io, work_dir);
    var modules: std.ArrayList(PreparedModule) = .empty;

    const noop_src = try std.fs.path.join(allocator, &.{ repo, "tests/coldstart/noop.wasm" });
    const noop_bytes = std.Io.Dir.cwd().readFileAlloc(io, noop_src, allocator, .unlimited) catch |err| {
        std.debug.print("error: missing fixture {s}: {s}\n", .{ noop_src, @errorName(err) });
        return error.MissingNoopFixture;
    };
    const noop_wasm = try std.fs.path.join(allocator, &.{ work_dir, "noop.wasm" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = noop_wasm, .data = noop_bytes });

    const noop_cwasm = try std.fs.path.join(allocator, &.{ work_dir, "noop.cwasm" });
    const compile_out = try process.run(allocator, io, work_dir, env, &.{ wamrc_bin, "compile", noop_wasm, "-o", noop_cwasm }, stop);
    allocator.free(compile_out);

    var noop_wt_cwasm: ?[]const u8 = null;
    if (wasmtime_bin) |wt| {
        const candidate = try std.fs.path.join(allocator, &.{ work_dir, "noop.wasmtime.cwasm" });
        if (process.run(allocator, io, work_dir, env, &.{ wt, "compile", noop_wasm, "-o", candidate }, stop)) |wt_out| {
            allocator.free(wt_out);
            noop_wt_cwasm = candidate;
        } else |err| {
            if (err == error.Interrupted) return err;
            std.debug.print("[harness] warning: wasmtime compile of noop.wasm failed\n", .{});
        }
    }
    try modules.append(allocator, .{
        .name = "noop",
        .wasm = noop_wasm,
        .wamr_cwasm = noop_cwasm,
        .wasmtime_cwasm = noop_wt_cwasm,
    });

    const wasm_tools = if (tools.wasm_tools) |explicit|
        (if (explicit.len == 0) null else explicit)
    else
        try findInPath(allocator, io, env, "wasm-tools");
    if (wasm_tools) |wt_tool| {
        const add_wat = try std.fs.path.join(allocator, &.{ work_dir, "add100.wat" });
        const add_wasm = try std.fs.path.join(allocator, &.{ work_dir, "add100.wasm" });
        const wat_text = try buildAdd100Wat(allocator);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = add_wat, .data = wat_text });
        if (process.run(allocator, io, work_dir, env, &.{ wt_tool, "parse", add_wat, "-o", add_wasm }, stop)) |parse_out| {
            allocator.free(parse_out);
            const add_cwasm = try std.fs.path.join(allocator, &.{ work_dir, "add100.cwasm" });
            const add_compile = try process.run(allocator, io, work_dir, env, &.{ wamrc_bin, "compile", add_wasm, "-o", add_cwasm }, stop);
            allocator.free(add_compile);
            var add_wt_cwasm: ?[]const u8 = null;
            if (wasmtime_bin) |wt| {
                const candidate = try std.fs.path.join(allocator, &.{ work_dir, "add100.wasmtime.cwasm" });
                if (process.run(allocator, io, work_dir, env, &.{ wt, "compile", add_wasm, "-o", candidate }, stop)) |wt_out| {
                    allocator.free(wt_out);
                    add_wt_cwasm = candidate;
                } else |err| {
                    if (err == error.Interrupted) return err;
                    add_wt_cwasm = null;
                }
            }
            try modules.append(allocator, .{
                .name = "add100",
                .wasm = add_wasm,
                .wamr_cwasm = add_cwasm,
                .wasmtime_cwasm = add_wt_cwasm,
            });
        } else |err| {
            if (err == error.Interrupted) return err;
            std.debug.print("[harness] warning: wasm-tools parse failed; skipping add100\n", .{});
        }
    }
    return modules.toOwnedSlice(allocator);
}

pub fn timeEngineModule(
    allocator: Allocator,
    io: std.Io,
    cwd: []const u8,
    env: *const std.process.Environ.Map,
    label: []const u8,
    engine: []const u8,
    module_name: []const u8,
    cmd: []const []const u8,
    warmup: usize,
    samples: usize,
    stop: *const std.atomic.Value(bool),
) !TimingStats {
    for (0..warmup) |_| {
        const out = try process.run(allocator, io, cwd, env, cmd, stop);
        allocator.free(out);
    }
    const raw = try allocator.alloc(u64, samples);
    defer allocator.free(raw);
    for (0..samples) |i| {
        const t0 = std.Io.Clock.awake.now(io);
        const out = try process.run(allocator, io, cwd, env, cmd, stop);
        const elapsed = t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
        allocator.free(out);
        raw[i] = if (elapsed > 0) @intCast(elapsed) else 0;
    }
    const stats = try summarizeRawNs(allocator, label, engine, module_name, warmup, raw);
    const med_s = try fixed(allocator, stats.medianMs(), 3);
    const min_s = try fixed(allocator, stats.minMs(), 3);
    const p95_s = try fixed(allocator, stats.p95Ms(), 3);
    const max_s = try fixed(allocator, stats.maxMs(), 3);
    const std_s = try fixed(allocator, stats.stdevMs(), 3);
    std.debug.print(
        "[harness] {s} {s}/{s}: median={s} ms (min={s} ms, p95={s} ms, max={s} ms, stdev={s} ms, n={d})\n",
        .{ label, engine, module_name, med_s, min_s, p95_s, max_s, std_s, samples },
    );
    return stats;
}

pub fn measureWorktree(
    allocator: Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    repo: []const u8,
    work_root: []const u8,
    label: []const u8,
    wt: WorktreeBuild,
    wasmtime_bin: ?[]const u8,
    warmup: usize,
    samples: usize,
    include_jit: bool,
    tools: Tools,
    stop: *const std.atomic.Value(bool),
) ![]const TimingStats {
    const mod_dir_name = try std.fmt.allocPrint(allocator, "modules-{s}", .{label});
    const work_dir = try std.fs.path.join(allocator, &.{ work_root, mod_dir_name });
    const modules = try prepareModules(allocator, io, env, repo, work_dir, wt.wamrc, wasmtime_bin, tools, stop);

    var results: std.ArrayList(TimingStats) = .empty;
    for (modules) |m| {
        try results.append(allocator, try timeEngineModule(
            allocator,
            io,
            wt.path,
            env,
            label,
            "wamr-aot",
            m.name,
            &.{ wt.wamr, "run", "--allow-precompiled", "--", m.wamr_cwasm },
            warmup,
            samples,
            stop,
        ));
        if (include_jit) {
            try results.append(allocator, try timeEngineModule(
                allocator,
                io,
                wt.path,
                env,
                label,
                "wamr-jit",
                m.name,
                &.{ wt.wamr, "run", "--jit", "--", m.wasm },
                warmup,
                samples,
                stop,
            ));
        }
        if (wasmtime_bin != null and m.wasmtime_cwasm != null) {
            try results.append(allocator, try timeEngineModule(
                allocator,
                io,
                wt.path,
                env,
                label,
                "wasmtime-aot",
                m.name,
                &.{ wasmtime_bin.?, "run", "--allow-precompiled", m.wasmtime_cwasm.? },
                warmup,
                samples,
                stop,
            ));
        }
    }
    return results.toOwnedSlice(allocator);
}

pub fn formatNsAsMs(allocator: Allocator, ns: ?u64) ![]const u8 {
    const value = ns orelse return "\xe2\x80\x94";
    const ms = try fixed(allocator, @as(f64, @floatFromInt(value)) / 1e6, 3);
    return std.fmt.allocPrint(allocator, "{s} ms", .{ms});
}

fn findRow(rows: []const TimingStats, engine: []const u8, module_name: []const u8) ?TimingStats {
    for (rows) |row| {
        if (std.mem.eql(u8, row.engine, engine) and std.mem.eql(u8, row.module, module_name))
            return row;
    }
    return null;
}

fn collectModules(allocator: Allocator, first: []const TimingStats, second: []const TimingStats) ![]const []const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    for ([_][]const TimingStats{ first, second }) |rows| {
        for (rows) |row| {
            var exists = false;
            for (names.items) |item| {
                if (std.mem.eql(u8, item, row.module)) {
                    exists = true;
                    break;
                }
            }
            if (!exists) try names.append(allocator, row.module);
        }
    }
    return names.toOwnedSlice(allocator);
}

fn collectEnginesForModule(
    allocator: Allocator,
    module_name: []const u8,
    first: []const TimingStats,
    second: []const TimingStats,
) ![]const []const u8 {
    var engines: std.ArrayList([]const u8) = .empty;
    for ([_][]const TimingStats{ first, second }) |rows| {
        for (rows) |row| {
            if (!std.mem.eql(u8, row.module, module_name)) continue;
            var exists = false;
            for (engines.items) |item| {
                if (std.mem.eql(u8, item, row.engine)) {
                    exists = true;
                    break;
                }
            }
            if (!exists) try engines.append(allocator, row.engine);
        }
    }
    return engines.toOwnedSlice(allocator);
}

pub const ComparisonReport = struct {
    markdown: []const u8,
    regressions: []const []const u8,
};

pub fn renderMarkdown(
    allocator: Allocator,
    baseline_ref: []const u8,
    baseline_sha: []const u8,
    baseline_rows: []const TimingStats,
    target_ref: []const u8,
    target_sha: []const u8,
    target_rows: []const TimingStats,
    budget_ms: ?f64,
    regression_ratio: f64,
    mode: []const u8,
) !ComparisonReport {
    var out: std.Io.Writer.Allocating = .init(allocator);
    const w = &out.writer;
    var regressions: std.ArrayList([]const u8) = .empty;

    const b_short = baseline_sha[0..@min(12, baseline_sha.len)];
    const t_short = target_sha[0..@min(12, target_sha.len)];
    const title_suffix = if (std.mem.eql(u8, mode, "ReleaseFast"))
        ""
    else
        try std.fmt.allocPrint(allocator, " ({s})", .{mode});
    try w.print("## CLI cold-start benchmark (--allow-precompiled){s}\n\n", .{title_suffix});
    try w.print("- Baseline: `{s}` (`{s}`)\n", .{ baseline_ref, b_short });
    try w.print("- Target:   `{s}` (`{s}`)\n", .{ target_ref, t_short });
    try w.print("- Optimize: `{s}`\n", .{mode});
    if (budget_ms) |budget| {
        try w.print("- Budget:   `{s} ms` median (target `wamr-aot`)\n", .{try fixed(allocator, budget, 2)});
    }
    try w.print("- Regression gate: target / baseline median ratio <= `{s}x`\n\n", .{try fixed(allocator, regression_ratio, 2)});
    try w.writeAll("| Module | Engine | Ref | Median | p95 | Min | Max | Stdev | N | vs baseline | vs wasmtime-aot | Status |\n");
    try w.writeAll("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");

    const modules = try collectModules(allocator, baseline_rows, target_rows);
    for (modules) |mod| {
        const wt_row = findRow(target_rows, "wasmtime-aot", mod) orelse findRow(baseline_rows, "wasmtime-aot", mod);
        const engines = try collectEnginesForModule(allocator, mod, baseline_rows, target_rows);
        for (engines) |eng| {
            const b = findRow(baseline_rows, eng, mod);
            const t = findRow(target_rows, eng, mod);
            if (b) |b_row| {
                const vs_wt = if (wt_row != null and wt_row.?.median_ns > 0 and !std.mem.eql(u8, eng, "wasmtime-aot"))
                    try std.fmt.allocPrint(allocator, "\xc3\x97{s}", .{try fixed(allocator, @as(f64, @floatFromInt(b_row.median_ns)) / @as(f64, @floatFromInt(wt_row.?.median_ns)), 2)})
                else
                    "\xe2\x80\x94";
                try w.print("| `{s}` | `{s}` | `{s}` (baseline) | {s} | {s} | {s} | {s} | {s} | {d} | \xe2\x80\x94 | {s} | \xe2\x80\x94 |\n", .{
                    mod,
                    eng,
                    baseline_ref,
                    try formatNsAsMs(allocator, b_row.median_ns),
                    try formatNsAsMs(allocator, b_row.p95_ns),
                    try formatNsAsMs(allocator, b_row.min_ns),
                    try formatNsAsMs(allocator, b_row.max_ns),
                    try formatNsAsMs(allocator, b_row.stdev_ns),
                    b_row.samples,
                    vs_wt,
                });
            }
            if (t) |t_row| {
                var vs_base: []const u8 = "\xe2\x80\x94";
                var status: []const u8 = "OK";
                if (b != null and b.?.median_ns > 0) {
                    const ratio = @as(f64, @floatFromInt(t_row.median_ns)) / @as(f64, @floatFromInt(b.?.median_ns));
                    vs_base = try std.fmt.allocPrint(allocator, "\xc3\x97{s}", .{try fixed(allocator, ratio, 2)});
                    if (std.mem.eql(u8, eng, "wamr-aot") and ratio > regression_ratio) {
                        status = try std.fmt.allocPrint(allocator, "**REGRESSION (>\xc3\x97{s})**", .{try fixed(allocator, regression_ratio, 2)});
                        try regressions.append(allocator, try std.fmt.allocPrint(
                            allocator,
                            "{s}/{s}: target median {s} vs baseline {s} (ratio {s}x > {s}x)",
                            .{
                                mod,
                                eng,
                                try formatNsAsMs(allocator, t_row.median_ns),
                                try formatNsAsMs(allocator, b.?.median_ns),
                                try fixed(allocator, ratio, 2),
                                try fixed(allocator, regression_ratio, 2),
                            },
                        ));
                    }
                }
                if (budget_ms != null and std.mem.eql(u8, eng, "wamr-aot") and t_row.medianMs() > budget_ms.?) {
                    status = try std.fmt.allocPrint(allocator, "**OVER BUDGET (>{s} ms)**", .{try fixed(allocator, budget_ms.?, 2)});
                    try regressions.append(allocator, try std.fmt.allocPrint(
                        allocator,
                        "{s}/{s}: target median {s} exceeds budget {s} ms",
                        .{ mod, eng, try formatNsAsMs(allocator, t_row.median_ns), try fixed(allocator, budget_ms.?, 2) },
                    ));
                }
                const vs_wt = if (wt_row != null and wt_row.?.median_ns > 0 and !std.mem.eql(u8, eng, "wasmtime-aot"))
                    try std.fmt.allocPrint(allocator, "\xc3\x97{s}", .{try fixed(allocator, @as(f64, @floatFromInt(t_row.median_ns)) / @as(f64, @floatFromInt(wt_row.?.median_ns)), 2)})
                else
                    "\xe2\x80\x94";
                try w.print("| `{s}` | `{s}` | `{s}` (target) | {s} | {s} | {s} | {s} | {s} | {d} | {s} | {s} | {s} |\n", .{
                    mod,
                    eng,
                    target_ref,
                    try formatNsAsMs(allocator, t_row.median_ns),
                    try formatNsAsMs(allocator, t_row.p95_ns),
                    try formatNsAsMs(allocator, t_row.min_ns),
                    try formatNsAsMs(allocator, t_row.max_ns),
                    try formatNsAsMs(allocator, t_row.stdev_ns),
                    t_row.samples,
                    vs_base,
                    vs_wt,
                    status,
                });
            }
        }
    }

    try w.writeAll("\nNotes:\n");
    try w.writeAll("- Each sample is one full OS process invocation (`fork`/`exec` through `_start` return and process exit).\n");
    try w.writeAll("- Precompilation (`wamrc compile` / `wasmtime compile`) runs once before timing; only `run --allow-precompiled` is timed.\n");
    return .{
        .markdown = try out.toOwnedSlice(),
        .regressions = try regressions.toOwnedSlice(allocator),
    };
}

pub fn renderOptimizeComparisonMarkdown(
    allocator: Allocator,
    target_ref: []const u8,
    fast_sha: []const u8,
    fast_rows: []const TimingStats,
    safe_sha: []const u8,
    safe_rows: []const TimingStats,
) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    const w = &out.writer;
    const f_short = fast_sha[0..@min(12, fast_sha.len)];
    const s_short = safe_sha[0..@min(12, safe_sha.len)];
    try w.writeAll("## CLI cold-start optimize-mode comparison (--allow-precompiled)\n\n");
    try w.print("- Target: `{s}` (ReleaseFast `{s}`, ReleaseSafe `{s}`)\n\n", .{ target_ref, f_short, s_short });
    try w.writeAll("| Module | Engine | Ref | Fast median | Safe median | Safe/Fast median | Fast p95 | Safe p95 | Safe/Fast p95 | N |\n");
    try w.writeAll("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n");

    const modules = try collectModules(allocator, fast_rows, safe_rows);
    for (modules) |mod| {
        const engines = try collectEnginesForModule(allocator, mod, fast_rows, safe_rows);
        for (engines) |eng| {
            const f = findRow(fast_rows, eng, mod);
            const s = findRow(safe_rows, eng, mod);
            const f_med: ?f64 = if (f) |row| @floatFromInt(row.median_ns) else null;
            const s_med: ?f64 = if (s) |row| @floatFromInt(row.median_ns) else null;
            const f_p95: ?f64 = if (f) |row| @floatFromInt(row.p95_ns) else null;
            const s_p95: ?f64 = if (s) |row| @floatFromInt(row.p95_ns) else null;
            const med_ratio = try optimize.formatRatio(allocator, s_med, f_med);
            const p95_ratio = try optimize.formatRatio(allocator, s_p95, f_p95);
            const n_str = if (f) |row|
                try std.fmt.allocPrint(allocator, "{d}", .{row.samples})
            else if (s) |row|
                try std.fmt.allocPrint(allocator, "{d}", .{row.samples})
            else
                "\xe2\x80\x94";
            try w.print("| `{s}` | `{s}` | `{s}` (target) | {s} | {s} | {s} | {s} | {s} | {s} | {s} |\n", .{
                mod,
                eng,
                target_ref,
                try formatNsAsMs(allocator, if (f) |row| row.median_ns else null),
                try formatNsAsMs(allocator, if (s) |row| row.median_ns else null),
                med_ratio,
                try formatNsAsMs(allocator, if (f) |row| row.p95_ns else null),
                try formatNsAsMs(allocator, if (s) |row| row.p95_ns else null),
                p95_ratio,
                n_str,
            });
        }
    }
    try w.writeAll("\nNotes:\n");
    try w.writeAll("- Safe/Fast ratios compare ReleaseSafe timing divided by ReleaseFast timing for the same target/module/engine.\n");
    return out.toOwnedSlice();
}

fn writeJsonStatsArray(jw: *std.json.Stringify, rows: []const TimingStats) !void {
    try jw.beginArray();
    for (rows) |row| {
        try jw.beginObject();
        try jw.objectField("label");
        try jw.write(row.label);
        try jw.objectField("engine");
        try jw.write(row.engine);
        try jw.objectField("module");
        try jw.write(row.module);
        try jw.objectField("samples");
        try jw.write(row.samples);
        try jw.objectField("warmup");
        try jw.write(row.warmup);
        try jw.objectField("median_ns");
        try jw.write(row.median_ns);
        try jw.objectField("min_ns");
        try jw.write(row.min_ns);
        try jw.objectField("max_ns");
        try jw.write(row.max_ns);
        try jw.objectField("p95_ns");
        try jw.write(row.p95_ns);
        try jw.objectField("stdev_ns");
        try jw.write(row.stdev_ns);
        try jw.objectField("raw_ns");
        try jw.beginArray();
        for (row.raw_ns) |v| try jw.write(v);
        try jw.endArray();
        try jw.endObject();
    }
    try jw.endArray();
}

pub fn renderComparisonJson(
    allocator: Allocator,
    mode: []const u8,
    baseline_ref: []const u8,
    baseline_sha: []const u8,
    baseline_rows: []const TimingStats,
    target_ref: []const u8,
    target_sha: []const u8,
    target_rows: []const TimingStats,
    budget_ms: ?f64,
    regression_ratio: f64,
    regressions: []const []const u8,
) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    var jw: std.json.Stringify = .{ .writer = &out.writer, .options = .{ .whitespace = .indent_2 } };
    try jw.beginObject();
    try jw.objectField("optimize");
    try jw.write(mode);
    try jw.objectField("baseline");
    try jw.beginObject();
    try jw.objectField("ref");
    try jw.write(baseline_ref);
    try jw.objectField("sha");
    try jw.write(baseline_sha);
    try jw.objectField("results");
    try writeJsonStatsArray(&jw, baseline_rows);
    try jw.endObject();
    try jw.objectField("target");
    try jw.beginObject();
    try jw.objectField("ref");
    try jw.write(target_ref);
    try jw.objectField("sha");
    try jw.write(target_sha);
    try jw.objectField("results");
    try writeJsonStatsArray(&jw, target_rows);
    try jw.endObject();
    try jw.objectField("budget_ms");
    if (budget_ms) |budget| try jw.write(budget) else try jw.write(null);
    try jw.objectField("regression_ratio");
    try jw.write(regression_ratio);
    try jw.objectField("regressions");
    try jw.beginArray();
    for (regressions) |item| try jw.write(item);
    try jw.endArray();
    try jw.endObject();
    try out.writer.writeByte('\n');
    return out.toOwnedSlice();
}

pub fn renderOptimizeJson(
    allocator: Allocator,
    target_ref: []const u8,
    fast_sha: []const u8,
    fast_rows: []const TimingStats,
    safe_sha: []const u8,
    safe_rows: []const TimingStats,
) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    var jw: std.json.Stringify = .{ .writer = &out.writer, .options = .{ .whitespace = .indent_2 } };
    try jw.beginObject();
    try jw.objectField("optimize");
    try jw.write("both");
    try jw.objectField("target");
    try jw.beginObject();
    try jw.objectField("ref");
    try jw.write(target_ref);
    try jw.objectField("ReleaseFast");
    try jw.beginObject();
    try jw.objectField("sha");
    try jw.write(fast_sha);
    try jw.objectField("results");
    try writeJsonStatsArray(&jw, fast_rows);
    try jw.endObject();
    try jw.objectField("ReleaseSafe");
    try jw.beginObject();
    try jw.objectField("sha");
    try jw.write(safe_sha);
    try jw.objectField("results");
    try writeJsonStatsArray(&jw, safe_rows);
    try jw.endObject();
    try jw.endObject();
    try jw.endObject();
    try out.writer.writeByte('\n');
    return out.toOwnedSlice();
}

fn findRepo(allocator: Allocator, io: std.Io, start: []const u8) !?[]const u8 {
    var directory: ?[]const u8 = start;
    while (directory) |path| {
        const marker = try std.fs.path.join(allocator, &.{ path, "tests/coldstart/noop.wasm" });
        std.Io.Dir.cwd().access(io, marker, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                directory = std.fs.path.dirname(path);
                continue;
            },
            else => return err,
        };
        return path;
    }
    return null;
}

fn resolveRepo(allocator: Allocator, io: std.Io, requested: ?[]const u8) ![]const u8 {
    if (requested) |path| return std.Io.Dir.cwd().realPathFileAlloc(io, path, allocator);
    const executable = try std.process.executablePathAlloc(io, allocator);
    if (std.fs.path.dirname(executable)) |directory| {
        if (try findRepo(allocator, io, directory)) |repo| return repo;
    }
    const cwd = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
    return (try findRepo(allocator, io, cwd)) orelse error.BenchmarkRepositoryNotFound;
}

fn createWorkRoot(allocator: Allocator, io: std.Io, env: *const std.process.Environ.Map) ![]const u8 {
    const parent = try temp_root.select(allocator, io, env);
    for (0..10) |_| {
        var random: [16]u8 = undefined;
        io.random(&random);
        const name = try std.fmt.allocPrint(allocator, "wamr-coldstart-{s}", .{std.fmt.bytesToHex(random, .lower)});
        const path = try std.fs.path.join(allocator, &.{ parent, name });
        std.Io.Dir.cwd().createDir(io, path, .default_dir) catch |err| switch (err) {
            error.PathAlreadyExists => continue,
            else => return err,
        };
        return path;
    }
    return error.TemporaryNameCollisions;
}

fn writeFileCreatingParents(io: std.Io, path: []const u8, data: []const u8) !void {
    if (std.fs.path.dirname(path)) |parent| {
        if (parent.len != 0) try std.Io.Dir.cwd().createDirPath(io, parent);
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
}

const Cleanup = struct {
    allocator: Allocator,
    io: std.Io,
    repo: []const u8,
    root: []const u8,
    tools: Tools,
    worktrees: [2]?[]const u8 = @splat(null),

    fn clean(self: *Cleanup) !void {
        var failure: ?anyerror = null;
        for (0..self.worktrees.len) |index| {
            if (self.worktrees[self.worktrees.len - index - 1]) |wt| {
                removeWorktree(self.allocator, self.io, self.repo, self.tools, wt) catch |err| {
                    std.debug.print("[harness] warning: failed to remove worktree {s}: {s}\n", .{ wt, @errorName(err) });
                    failure = err;
                };
            }
        }
        std.Io.Dir.cwd().deleteTree(self.io, self.root) catch |err| {
            if (failure == null) failure = err;
        };
        if (failure) |err| return err;
    }
};

pub fn execute(
    allocator: Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    opts: Options,
    tools: Tools,
) !void {
    const repo = try resolveRepo(allocator, io, opts.repo);
    var wasmtime_bin: ?[]const u8 = null;
    if (opts.wasmtime or opts.wasmtime_path != null) {
        wasmtime_bin = try resolveWasmtime(allocator, io, env, opts.wasmtime_path);
        if (wasmtime_bin) |wt| {
            std.debug.print("[harness] using wasmtime: {s}\n", .{wt});
        } else {
            std.debug.print("[harness] warning: --wasmtime requested, but wasmtime binary was not found on PATH or ~/.wasmtime/bin; skipping wasmtime comparison\n", .{});
        }
    }

    const work_root = try createWorkRoot(allocator, io, env);
    var cleanup: Cleanup = .{ .allocator = allocator, .io = io, .repo = repo, .root = work_root, .tools = tools };
    var needs_cleanup = true;
    defer if (needs_cleanup) cleanup.clean() catch |err|
        std.debug.print("[harness] warning: coldstart cleanup failed: {s}\n", .{@errorName(err)});

    var markdown: []const u8 = "";
    var regressions: []const []const u8 = &.{};
    var json_payload: ?[]const u8 = null;

    if (std.mem.eql(u8, opts.mode, "both")) {
        const fast_wt = try buildWorktree(allocator, io, env, repo, work_root, "target-fast", opts.target, "ReleaseFast", tools, &interrupted);
        cleanup.worktrees[0] = fast_wt.path;
        const fast_rows = try measureWorktree(
            allocator,
            io,
            env,
            repo,
            work_root,
            "target-fast",
            fast_wt,
            wasmtime_bin,
            opts.warmup,
            opts.samples,
            opts.include_jit,
            tools,
            &interrupted,
        );
        const safe_wt = try buildWorktree(allocator, io, env, repo, work_root, "target-safe", opts.target, "ReleaseSafe", tools, &interrupted);
        cleanup.worktrees[1] = safe_wt.path;
        const safe_rows = try measureWorktree(
            allocator,
            io,
            env,
            repo,
            work_root,
            "target-safe",
            safe_wt,
            wasmtime_bin,
            opts.warmup,
            opts.samples,
            opts.include_jit,
            tools,
            &interrupted,
        );
        markdown = try renderOptimizeComparisonMarkdown(allocator, opts.target, fast_wt.sha, fast_rows, safe_wt.sha, safe_rows);
        if (opts.json_path != null) {
            json_payload = try renderOptimizeJson(allocator, opts.target, fast_wt.sha, fast_rows, safe_wt.sha, safe_rows);
        }
    } else {
        const base_wt = try buildWorktree(allocator, io, env, repo, work_root, "baseline", opts.baseline, opts.mode, tools, &interrupted);
        cleanup.worktrees[0] = base_wt.path;
        const target_wt = try buildWorktree(allocator, io, env, repo, work_root, "target", opts.target, opts.mode, tools, &interrupted);
        cleanup.worktrees[1] = target_wt.path;
        const baseline_rows = try measureWorktree(
            allocator,
            io,
            env,
            repo,
            work_root,
            "baseline",
            base_wt,
            wasmtime_bin,
            opts.warmup,
            opts.samples,
            opts.include_jit,
            tools,
            &interrupted,
        );
        const target_rows = try measureWorktree(
            allocator,
            io,
            env,
            repo,
            work_root,
            "target",
            target_wt,
            wasmtime_bin,
            opts.warmup,
            opts.samples,
            opts.include_jit,
            tools,
            &interrupted,
        );
        const report = try renderMarkdown(
            allocator,
            opts.baseline,
            base_wt.sha,
            baseline_rows,
            opts.target,
            target_wt.sha,
            target_rows,
            opts.budget_ms,
            opts.regression_ratio,
            opts.mode,
        );
        markdown = report.markdown;
        regressions = report.regressions;
        if (opts.json_path != null) {
            json_payload = try renderComparisonJson(
                allocator,
                opts.mode,
                opts.baseline,
                base_wt.sha,
                baseline_rows,
                opts.target,
                target_wt.sha,
                target_rows,
                opts.budget_ms,
                opts.regression_ratio,
                regressions,
            );
        }
    }

    needs_cleanup = false;
    try cleanup.clean();
    if (interrupted.load(.acquire)) return error.Interrupted;

    const output_text = try std.fmt.allocPrint(allocator, "{s}\n", .{markdown});
    if (!@import("builtin").is_test) {
        try std.Io.File.stdout().writeStreamingAll(io, output_text);
    }
    if (opts.output) |out_path| {
        try writeFileCreatingParents(io, out_path, output_text);
    }
    if (std.mem.eql(u8, opts.emit, "github")) {
        if (env.get("GITHUB_STEP_SUMMARY")) |summary_path| {
            if (summary_path.len != 0) {
                const file = try std.Io.Dir.cwd().createFile(io, summary_path, .{ .truncate = false });
                defer file.close(io);
                const end = (try file.stat(io)).size;
                const summary_text = try std.fmt.allocPrint(allocator, "\n{s}\n", .{markdown});
                try file.writePositionalAll(io, summary_text, end);
            }
        }
    }
    if (opts.json_path) |json_path| {
        try writeFileCreatingParents(io, json_path, json_payload.?);
    }
    if (regressions.len != 0) {
        std.debug.print("\n[harness] FAIL: cold-start regression(s) detected:\n", .{});
        for (regressions) |item| {
            std.debug.print("  - {s}\n", .{item});
        }
        return error.ColdStartRegression;
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const opts = parseOptions(args[1..]) catch |err| {
        if (err == error.HelpRequested) {
            try std.Io.File.stdout().writeStreamingAll(init.io, usage);
            return;
        }
        std.debug.print("bench-coldstart: {s}\n{s}", .{ @errorName(err), usage });
        std.process.exit(2);
    };
    const signals = [_]std.posix.SIG{ .INT, .TERM };
    var old: [2]std.posix.Sigaction = undefined;
    const action: std.posix.Sigaction = .{ .handler = .{ .handler = onSignal }, .mask = std.posix.sigemptyset(), .flags = 0 };
    interrupted.store(false, .release);
    for (signals, 0..) |signal, index| std.posix.sigaction(signal, &action, &old[index]);
    defer for (signals, 0..) |signal, index| std.posix.sigaction(signal, &old[index], null);
    execute(allocator, init.io, init.environ_map, opts, .{}) catch |err| {
        if (err == error.Interrupted) std.process.exit(130);
        if (err == error.InvalidWasmtimePath) std.process.exit(2);
        if (err != error.ColdStartRegression) {
            std.debug.print("bench-coldstart: {s}\n", .{@errorName(err)});
        }
        std.process.exit(1);
    };
}

test "parseOptions handles defaults, flags, and validation errors" {
    const defaults = try parseOptions(&.{});
    try std.testing.expectEqualStrings("origin/main", defaults.baseline);
    try std.testing.expectEqualStrings("HEAD", defaults.target);
    try std.testing.expectEqual(@as(usize, 30), defaults.samples);
    try std.testing.expectEqual(@as(usize, 3), defaults.warmup);
    try std.testing.expect(!defaults.wasmtime);
    try std.testing.expect(!defaults.include_jit);
    try std.testing.expectEqual(@as(?f64, null), defaults.budget_ms);
    try std.testing.expectEqual(@as(f64, 1.5), defaults.regression_ratio);
    try std.testing.expectEqualStrings("ReleaseFast", defaults.mode);

    const parsed = try parseOptions(&.{
        "--baseline",         "base-ref",
        "--target=head-ref",  "--samples",
        "10",                 "--warmup",
        "0",                  "--wasmtime",
        "--wasmtime-path",    "/bin/wt",
        "--include-jit",      "--budget-ms",
        "5.25",               "--regression-ratio",
        "1.25",               "--json",
        "out.json",           "--out",
        "out.md",             "--emit",
        "github",             "--repo",
        "/tmp/repo",          "--optimize",
        "both",
    });
    try std.testing.expectEqualStrings("base-ref", parsed.baseline);
    try std.testing.expectEqualStrings("head-ref", parsed.target);
    try std.testing.expectEqual(@as(usize, 10), parsed.samples);
    try std.testing.expectEqual(@as(usize, 0), parsed.warmup);
    try std.testing.expect(parsed.wasmtime);
    try std.testing.expectEqualStrings("/bin/wt", parsed.wasmtime_path.?);
    try std.testing.expect(parsed.include_jit);
    try std.testing.expectEqual(@as(?f64, 5.25), parsed.budget_ms);
    try std.testing.expectEqual(@as(f64, 1.25), parsed.regression_ratio);
    try std.testing.expectEqualStrings("out.json", parsed.json_path.?);
    try std.testing.expectEqualStrings("out.md", parsed.output.?);
    try std.testing.expectEqualStrings("github", parsed.emit);
    try std.testing.expectEqualStrings("/tmp/repo", parsed.repo.?);
    try std.testing.expectEqualStrings("both", parsed.mode);

    try std.testing.expectError(error.NonpositiveCount, parseOptions(&.{ "--samples", "0" }));
    try std.testing.expectError(error.NegativeCount, parseOptions(&.{ "--warmup", "-1" }));
    try std.testing.expectError(error.NonpositiveFloat, parseOptions(&.{ "--budget-ms", "0" }));
    try std.testing.expectError(error.NonpositiveFloat, parseOptions(&.{ "--regression-ratio", "-1" }));
    try std.testing.expectError(error.InvalidOptimizeMode, parseOptions(&.{ "--optimize", "Debug" }));
    try std.testing.expectError(error.InvalidOutputMode, parseOptions(&.{ "--emit", "json" }));
}

test "summarizeRawNs matches median, p95, and sample stdev semantics" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const stats = try summarizeRawNs(
        allocator,
        "target",
        "wamr-aot",
        "noop",
        2,
        &.{ 1_500_000, 1_100_000, 1_300_000, 1_200_000 },
    );
    try std.testing.expectEqual(@as(u64, 1_250_000), stats.median_ns);
    try std.testing.expectEqual(@as(u64, 1_100_000), stats.min_ns);
    try std.testing.expectEqual(@as(u64, 1_500_000), stats.max_ns);
    try std.testing.expectEqual(@as(u64, 1_500_000), stats.p95_ns);
    try std.testing.expectEqual(@as(u64, 170_782), stats.stdev_ns);
}

test "renderMarkdown and renderOptimizeComparisonMarkdown format tables and detect regressions" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const base_rows = [_]TimingStats{
        .{
            .label = "baseline",
            .engine = "wamr-aot",
            .module = "noop",
            .samples = 5,
            .warmup = 1,
            .median_ns = 1_000_000,
            .min_ns = 900_000,
            .max_ns = 1_200_000,
            .p95_ns = 1_150_000,
            .stdev_ns = 100_000,
            .raw_ns = &.{1_000_000},
        },
        .{
            .label = "baseline",
            .engine = "wasmtime-aot",
            .module = "noop",
            .samples = 5,
            .warmup = 1,
            .median_ns = 2_000_000,
            .min_ns = 1_900_000,
            .max_ns = 2_200_000,
            .p95_ns = 2_150_000,
            .stdev_ns = 100_000,
            .raw_ns = &.{2_000_000},
        },
    };
    const target_rows = [_]TimingStats{
        .{
            .label = "target",
            .engine = "wamr-aot",
            .module = "noop",
            .samples = 5,
            .warmup = 1,
            .median_ns = 1_600_000,
            .min_ns = 1_500_000,
            .max_ns = 1_800_000,
            .p95_ns = 1_750_000,
            .stdev_ns = 100_000,
            .raw_ns = &.{1_600_000},
        },
        .{
            .label = "target",
            .engine = "wasmtime-aot",
            .module = "noop",
            .samples = 5,
            .warmup = 1,
            .median_ns = 2_000_000,
            .min_ns = 1_900_000,
            .max_ns = 2_200_000,
            .p95_ns = 2_150_000,
            .stdev_ns = 100_000,
            .raw_ns = &.{2_000_000},
        },
    };

    const report = try renderMarkdown(
        allocator,
        "origin/main",
        "1111111111111111111111111111111111111111",
        &base_rows,
        "HEAD",
        "2222222222222222222222222222222222222222",
        &target_rows,
        1.5,
        1.5,
        "ReleaseFast",
    );
    try std.testing.expect(std.mem.indexOf(u8, report.markdown, "## CLI cold-start benchmark (--allow-precompiled)") != null);
    try std.testing.expect(std.mem.indexOf(u8, report.markdown, "\xc3\x971.60") != null);
    try std.testing.expect(std.mem.indexOf(u8, report.markdown, "\xc3\x970.80") != null);
    try std.testing.expect(std.mem.indexOf(u8, report.markdown, "**OVER BUDGET (>1.50 ms)**") != null);
    try std.testing.expectEqual(@as(usize, 2), report.regressions.len);

    const opt_md = try renderOptimizeComparisonMarkdown(
        allocator,
        "HEAD",
        "1111111111111111111111111111111111111111",
        &base_rows,
        "2222222222222222222222222222222222222222",
        &target_rows,
    );
    try std.testing.expect(std.mem.indexOf(u8, opt_md, "## CLI cold-start optimize-mode comparison (--allow-precompiled)") != null);
    try std.testing.expect(std.mem.indexOf(u8, opt_md, "\xc3\x971.60") != null);
}

test "execute orchestrates worktrees, writes JSON and Markdown outputs, and cleans up" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    try temp.dir.createDirPath(io, "tests/coldstart");
    try temp.dir.writeFile(io, .{ .sub_path = "tests/coldstart/noop.wasm", .data = "\x00asm\x01\x00\x00\x00" });

    const actions = try std.fs.path.join(allocator, &.{ root, "actions.log" });
    const fake_git = try std.fs.path.join(allocator, &.{ root, "fake-git" });
    const fake_zig = try std.fs.path.join(allocator, &.{ root, "fake-zig" });
    const fake_wamr = try std.fs.path.join(allocator, &.{ root, "fake-wamr" });
    const fake_wamrc = try std.fs.path.join(allocator, &.{ root, "fake-wamrc" });

    try temp.dir.writeFile(io, .{ .sub_path = "fake-wamr", .data = "#!/bin/sh\nexit 0\n" });
    try temp.dir.writeFile(io, .{ .sub_path = "fake-wamrc", .data =
        \\#!/bin/sh
        \\printf 'cwasm' > "$4"
        \\
    });
    const git_script = try std.fmt.allocPrint(allocator,
        \\#!/bin/sh
        \\if [ "$1" = "rev-parse" ]; then
        \\  printf '1234567890abcdef1234567890abcdef12345678\n'
        \\  exit 0
        \\fi
        \\if [ "$1" = "worktree" ] && [ "$2" = "add" ]; then
        \\  mkdir -p "$4/zig-out/bin"
        \\  printf '.minimum_zig_version = "0.17.0"\n' > "$4/build.zig.zon"
        \\  cp "{s}" "$4/zig-out/bin/wamr"
        \\  cp "{s}" "$4/zig-out/bin/wamrc"
        \\  chmod +x "$4/zig-out/bin/wamr" "$4/zig-out/bin/wamrc"
        \\  printf 'add %%s\n' "$4" >> "{s}"
        \\  exit 0
        \\fi
        \\if [ "$1" = "worktree" ] && [ "$2" = "remove" ]; then
        \\  rm -rf "$4"
        \\  printf 'remove %%s\n' "$4" >> "{s}"
        \\  exit 0
        \\fi
        \\exit 9
        \\
    , .{ fake_wamr, fake_wamrc, actions, actions });
    try temp.dir.writeFile(io, .{ .sub_path = "fake-git", .data = git_script });
    try temp.dir.writeFile(io, .{ .sub_path = "fake-zig", .data =
        \\#!/bin/sh
        \\[ -z "$ZIG_LOCAL_CACHE_DIR" ] && [ "$ZIG_GLOBAL_CACHE_DIR" = "$PWD/.zig-global-cache" ]
        \\
    });

    for ([_][]const u8{ "fake-wamr", "fake-wamrc", "fake-git", "fake-zig" }) |name| {
        var file = try temp.dir.openFile(io, name, .{});
        defer file.close(io);
        try file.setPermissions(io, .fromMode(0o755));
    }

    const out_md = try std.fs.path.join(allocator, &.{ root, "report.md" });
    const out_json = try std.fs.path.join(allocator, &.{ root, "report.json" });
    const summary = try std.fs.path.join(allocator, &.{ root, "summary.md" });
    var env: std.process.Environ.Map = .init(allocator);
    try env.put("TMPDIR", root);
    try env.put("GITHUB_STEP_SUMMARY", summary);

    try execute(allocator, io, &env, .{
        .repo = root,
        .samples = 2,
        .warmup = 1,
        .output = out_md,
        .json_path = out_json,
        .emit = "github",
        .regression_ratio = 100.0,
    }, .{
        .git = fake_git,
        .zig = fake_zig,
        .wasm_tools = "",
    });

    const md_bytes = try std.Io.Dir.cwd().readFileAlloc(io, out_md, allocator, .unlimited);
    try std.testing.expect(std.mem.indexOf(u8, md_bytes, "## CLI cold-start benchmark (--allow-precompiled)") != null);
    const json_bytes = try std.Io.Dir.cwd().readFileAlloc(io, out_json, allocator, .unlimited);
    try std.testing.expect(std.mem.indexOf(u8, json_bytes, "\"optimize\": \"ReleaseFast\"") != null);
    const summary_bytes = try std.Io.Dir.cwd().readFileAlloc(io, summary, allocator, .unlimited);
    try std.testing.expect(std.mem.indexOf(u8, summary_bytes, "## CLI cold-start benchmark") != null);
    const log_bytes = try std.Io.Dir.cwd().readFileAlloc(io, actions, allocator, .unlimited);
    try std.testing.expect(std.mem.indexOf(u8, log_bytes, "add ") != null);
    try std.testing.expect(std.mem.indexOf(u8, log_bytes, "remove ") != null);
}
