//! Subprocess execution, build caching, AOT compilation, and paired measurement for WASI thread benchmarks.
const std = @import("std");
const builtin = @import("builtin");
const benchmark_schema = @import("benchmark_schema.zig");
const benchmark_stats = @import("benchmark_stats.zig");
const process_control = @import("process_control.zig");
const spec = @import("wasi_thread_bench_spec.zig");

pub const Allocator = spec.Allocator;
pub const Value = spec.Value;
pub const ObjectMap = spec.ObjectMap;
pub const Array = spec.Array;
pub const HarnessError = spec.HarnessError;
const fail = spec.fail;
const failData = spec.failData;
const require = spec.require;

pub const Build = struct {
    name: []const u8,
    mode: []const u8,
    threads_enabled: bool,
    prefix: []const u8,
    wamr: []const u8,
    wamrc: ?[]const u8,
    key: []const u8,
    command: []const []const u8,
    reused: bool,
};

pub fn runProcess(
    allocator: Allocator,
    io: std.Io,
    command: []const []const u8,
    cwd: ?[]const u8,
    timeout_seconds: f64,
    env_map: ?*const std.process.Environ.Map,
) HarnessError!struct { exit_code: i32, stdout: []const u8, stderr: []const u8 } {
    const secs: u32 = if (timeout_seconds > 0) @max(@as(u32, 1), @as(u32, @intFromFloat(@ceil(timeout_seconds)))) else 3600;
    const res = process_control.runTimed(allocator, io, .{
        .argv = command,
        .cwd = if (cwd) |dir| .{ .path = dir } else .inherit,
        .environ_map = env_map,
        .stdout_limit = .limited(64 * 1024 * 1024),
        .stderr_limit = .limited(64 * 1024 * 1024),
    }, secs) catch |err| switch (err) {
        error.Timeout => {
            const joined = std.mem.join(allocator, " ", command) catch "command";
            return fail("command timed out after {d}s: {s}", .{ timeout_seconds, joined });
        },
        else => {
            const joined = std.mem.join(allocator, " ", command) catch "command";
            return fail("failed to execute command: {s}", .{joined});
        },
    };
    if (!std.unicode.utf8ValidateSlice(res.stdout) or !std.unicode.utf8ValidateSlice(res.stderr)) {
        const joined = std.mem.join(allocator, " ", command) catch "command";
        return fail("command produced non-UTF-8 output: {s}", .{joined});
    }
    const code: i32 = switch (res.term) {
        .exited => |c| @intCast(c),
        .signal => |sig| -@as(i32, @intCast(@intFromEnum(sig))),
        else => -1,
    };
    return .{ .exit_code = code, .stdout = res.stdout, .stderr = res.stderr };
}

pub fn gitOutput(allocator: Allocator, io: std.Io, repo: []const u8, args: []const []const u8) HarnessError![]const u8 {
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.append(allocator, "git");
    try argv.appendSlice(allocator, args);
    const res = try runProcess(allocator, io, argv.items, repo, 30.0, null);
    if (res.exit_code != 0) {
        return fail("git command failed in {s}: {s}", .{ repo, res.stderr });
    }
    return allocator.dupe(u8, std.mem.trim(u8, res.stdout, " \t\r\n"));
}

pub fn sourceIdentity(allocator: Allocator, io: std.Io, repo: []const u8) HarnessError!ObjectMap {
    const commit = try gitOutput(allocator, io, repo, &.{ "rev-parse", "HEAD" });
    const diff_res = try runProcess(
        allocator,
        io,
        &.{ "git", "diff", "--binary", "HEAD", "--", "build.zig", "src" },
        repo,
        30.0,
        null,
    );
    if (diff_res.exit_code != 0) return fail("git diff failed in {s}", .{repo});
    const tracked_diff_sha256 = try spec.sha256Bytes(allocator, diff_res.stdout);

    const ls_res = try runProcess(
        allocator,
        io,
        &.{ "git", "ls-files", "-z", "--", "build.zig", "build.zig.zon", "src" },
        repo,
        30.0,
        null,
    );
    if (ls_res.exit_code != 0) return fail("git ls-files failed in {s}", .{repo});

    var content: std.ArrayList(u8) = .empty;
    var it = std.mem.splitScalar(u8, ls_res.stdout, 0);
    while (it.next()) |raw_path| {
        if (raw_path.len == 0) continue;
        try content.appendSlice(allocator, raw_path);
        try content.append(allocator, 0);
        const full_path = try std.fs.path.join(allocator, &.{ repo, raw_path });
        const file_bytes = std.Io.Dir.cwd().readFileAlloc(io, full_path, allocator, .limited(64 * 1024 * 1024)) catch {
            return fail("cannot read tracked source file {s}", .{full_path});
        };
        try content.appendSlice(allocator, file_bytes);
        try content.append(allocator, 0);
    }
    const build_source_sha256 = try spec.sha256Bytes(allocator, content.items);

    var out: ObjectMap = .empty;
    try out.put(allocator, "commit", .{ .string = commit });
    try out.put(allocator, "tracked_diff_sha256", .{ .string = tracked_diff_sha256 });
    try out.put(allocator, "build_source_sha256", .{ .string = build_source_sha256 });
    return out;
}

pub fn fixtureSetIdentity(allocator: Allocator, fixtures: ObjectMap) ![]const u8 {
    var root: ObjectMap = .empty;
    var it = fixtures.iterator();
    while (it.next()) |entry| {
        const item = entry.value_ptr.object;
        var sub: ObjectMap = .empty;
        try sub.put(allocator, "path", item.get("path").?);
        try sub.put(allocator, "sha256", item.get("sha256").?);
        try root.put(allocator, entry.key_ptr.*, .{ .object = sub });
    }
    return spec.cacheKey(allocator, .{ .object = root });
}

pub fn resolveFixtures(allocator: Allocator, io: std.Io, repo: []const u8) HarnessError!ObjectMap {
    var result: ObjectMap = .empty;
    for (spec.FIXTURES) |fx| {
        const full_path = try std.fs.path.join(allocator, &.{ repo, fx.path });
        const stat = std.Io.Dir.cwd().statFile(io, full_path, .{}) catch {
            return fail("missing fixture: {s}", .{full_path});
        };
        const digest = try spec.sha256File(allocator, io, full_path);
        if (!std.mem.eql(u8, digest, fx.sha256)) {
            return fail("{s} fixture checksum mismatch: expected {s}, got {s}", .{ fx.name, fx.sha256, digest });
        }
        var item: ObjectMap = .empty;
        try item.put(allocator, "path", .{ .string = fx.path });
        try item.put(allocator, "sha256", .{ .string = digest });
        try item.put(allocator, "size", .{ .integer = @intCast(stat.size) });
        try result.put(allocator, fx.name, .{ .object = item });
    }
    for (spec.FIXTURE_SOURCE_FILES) |source_name| {
        const rel_path = try std.fmt.allocPrint(allocator, "tests/benchmarks/wasi-threads/{s}", .{source_name});
        const full_path = try std.fs.path.join(allocator, &.{ repo, rel_path });
        const stat = std.Io.Dir.cwd().statFile(io, full_path, .{}) catch {
            return fail("missing fixture source: {s}", .{full_path});
        };
        const digest = try spec.sha256File(allocator, io, full_path);
        var item: ObjectMap = .empty;
        try item.put(allocator, "path", .{ .string = rel_path });
        try item.put(allocator, "sha256", .{ .string = digest });
        try item.put(allocator, "size", .{ .integer = @intCast(stat.size) });
        const key = try std.fmt.allocPrint(allocator, "source:{s}", .{source_name});
        try result.put(allocator, key, .{ .object = item });
    }
    return result;
}

pub fn resolveMeasurementFixtures(
    allocator: Allocator,
    io: std.Io,
    revision_repos: std.StringHashMapUnmanaged([]const u8),
) HarnessError!struct { fixture_repo: []const u8, fixtures: ObjectMap, fixture_set_sha256: []const u8 } {
    const fixture_repo = revision_repos.get("candidate") orelse return fail("measurement fixtures require a candidate checkout", .{});
    const fixtures = try resolveFixtures(allocator, io, fixture_repo);
    const sha = try fixtureSetIdentity(allocator, fixtures);
    return .{
        .fixture_repo = fixture_repo,
        .fixtures = fixtures,
        .fixture_set_sha256 = sha,
    };
}

pub fn hostPairIdentity(
    allocator: Allocator,
    platform_id: []const u8,
    host: ObjectMap,
    explicit: ?[]const u8,
) HarnessError!ObjectMap {
    const fp_val = host.get("host_fingerprint") orelse return failData("host.host_fingerprint", .{});
    try require(fp_val == .object, "host.host_fingerprint", .{});
    const sha_val = fp_val.object.get("sha256") orelse return failData("host.host_fingerprint.sha256", .{});
    try require(sha_val == .string and spec.isHexSha256(sha_val.string), "host.host_fingerprint.sha256", .{});
    const fingerprint_sha256 = sha_val.string;

    const run_id = if (host.get("github_run_id")) |v| (if (v == .string) v.string else "") else "";
    const run_attempt = if (host.get("github_run_attempt")) |v| (if (v == .string) v.string else "") else "";

    var pair_id = explicit;
    if (pair_id == null and run_id.len > 0) {
        const attempt = if (run_attempt.len > 0) run_attempt else "1";
        pair_id = try std.fmt.allocPrint(allocator, "github:{s}:{s}:{s}", .{ run_id, attempt, platform_id });
    }
    if (pair_id == null) {
        pair_id = try std.fmt.allocPrint(allocator, "local:{s}:{s}", .{ platform_id, fingerprint_sha256[0..16] });
    }
    const runner_env_val = host.get("runner_environment") orelse return failData("host.runner_environment", .{});
    var out: ObjectMap = .empty;
    try out.put(allocator, "id", .{ .string = try allocator.dupe(u8, pair_id.?) });
    try out.put(allocator, "runner_environment", runner_env_val);
    try out.put(allocator, "host_fingerprint_sha256", .{ .string = try allocator.dupe(u8, fingerprint_sha256) });
    return out;
}

pub fn hostQuiescenceDiagnostics(
    allocator: Allocator,
    io: std.Io,
    environ: std.process.Environ,
) !ObjectMap {
    var diag: ObjectMap = .empty;
    try diag.put(allocator, "collected_at", .{ .string = try benchmark_schema.collectedAt(allocator, io) });
    const lcpus: i64 = @intCast(std.Thread.getCpuCount() catch 1);
    try diag.put(allocator, "logical_cpus", .{ .integer = lcpus });

    var affinity_arr = Array.init(allocator);
    var avail_val: Value = .null;
    if (builtin.os.tag == .linux) {
        var mask: [128]u8 = @splat(0);
        const rc = std.os.linux.syscall3(.sched_getaffinity, 0, mask.len, @intFromPtr(&mask));
        const signed_rc: isize = @bitCast(rc);
        if (signed_rc > 0) {
            const bytes: usize = @intCast(signed_rc);
            for (mask[0..@min(bytes, mask.len)], 0..) |b, byte_idx| {
                if (b != 0) {
                    for (0..8) |bit| {
                        if ((b & (@as(u8, 1) << @intCast(bit))) != 0) {
                            try affinity_arr.append(.{ .integer = @intCast(byte_idx * 8 + bit) });
                        }
                    }
                }
            }
            avail_val = .{ .integer = @intCast(affinity_arr.items.len) };
        }
    }
    try diag.put(allocator, "available_cpu_count", avail_val);
    try diag.put(allocator, "cpu_affinity", .{ .array = affinity_arr });

    var proc_loadavg_val: Value = .null;
    var load_avg_val: Value = .null;
    if (std.Io.Dir.cwd().readFileAlloc(io, "/proc/loadavg", allocator, .limited(4096))) |raw| {
        const trimmed = std.mem.trim(u8, raw, " \t\r\n");
        proc_loadavg_val = .{ .string = try allocator.dupe(u8, trimmed) };
        var tok = std.mem.tokenizeAny(u8, trimmed, " \t");
        const l1 = tok.next();
        const l2 = tok.next();
        const l3 = tok.next();
        if (l1 != null and l2 != null and l3 != null) {
            const f1 = std.fmt.parseFloat(f64, l1.?) catch null;
            const f2 = std.fmt.parseFloat(f64, l2.?) catch null;
            const f3 = std.fmt.parseFloat(f64, l3.?) catch null;
            if (f1 != null and f2 != null and f3 != null) {
                var la = Array.init(allocator);
                try la.append(.{ .float = f1.? });
                try la.append(.{ .float = f2.? });
                try la.append(.{ .float = f3.? });
                load_avg_val = .{ .array = la };
            }
        }
    } else |_| {}
    try diag.put(allocator, "load_average", load_avg_val);
    try diag.put(allocator, "proc_loadavg", proc_loadavg_val);

    var cpu_pressure_val: Value = .null;
    if (std.Io.Dir.cwd().readFileAlloc(io, "/proc/pressure/cpu", allocator, .limited(4096))) |raw| {
        cpu_pressure_val = .{ .string = try allocator.dupe(u8, std.mem.trim(u8, raw, " \t\r\n")) };
    } else |_| {}
    try diag.put(allocator, "cpu_pressure", cpu_pressure_val);
    try diag.put(allocator, "runner_worker_process_count", .{ .integer = 0 });

    var rjob: ObjectMap = .empty;
    try rjob.put(allocator, "runner_name", .{ .string = environ.getPosix("RUNNER_NAME") orelse "" });
    try rjob.put(allocator, "github_run_id", .{ .string = environ.getPosix("GITHUB_RUN_ID") orelse "" });
    try rjob.put(allocator, "github_run_attempt", .{ .string = environ.getPosix("GITHUB_RUN_ATTEMPT") orelse "" });
    try rjob.put(allocator, "github_job", .{ .string = environ.getPosix("GITHUB_JOB") orelse "" });
    try rjob.put(allocator, "github_workflow", .{ .string = environ.getPosix("GITHUB_WORKFLOW") orelse "" });
    try diag.put(allocator, "runner_job", .{ .object = rjob });
    return diag;
}

pub fn maximumPreflightBarrierNs(minimum_interval_ns: i64) HarnessError!i64 {
    if (minimum_interval_ns <= 0) {
        return fail("minimum timed interval must be positive", .{});
    }
    return @divTrunc(minimum_interval_ns - 1, 99);
}

pub fn preflightSampleAccepted(
    timing_overhead_ns: i64,
    timed_interval_ns: i64,
    minimum_interval_ns: i64,
) bool {
    return timed_interval_ns >= minimum_interval_ns and
        timing_overhead_ns >= 0 and
        99 * @as(i128, timing_overhead_ns) < @as(i128, minimum_interval_ns);
}

fn fmtOptionalVal(allocator: Allocator, v: ?Value) ![]const u8 {
    if (v == null or v.? == .null) return "None";
    return switch (v.?) {
        .integer => |i| std.fmt.allocPrint(allocator, "{d}", .{i}),
        .float => |f| std.fmt.allocPrint(allocator, "{d}", .{f}),
        .string => |s| allocator.dupe(u8, s),
        .number_string => |s| allocator.dupe(u8, s),
        else => "None",
    };
}

pub fn failureDiagnosticMarkdown(allocator: Allocator, document: ObjectMap) ![]const u8 {
    const scenario = document.get("scenario").?.object;
    const host = document.get("host").?.object;
    const host_pair = document.get("host_pair").?.object;
    const preflight_samples = document.get("preflight_samples").?.array;

    const wl = if (scenario.get("workload")) |v| (if (v == .string) v.string else "") else "";
    const mode = if (scenario.get("mode")) |v| (if (v == .string) v.string else "") else "";
    const threads_s = try fmtOptionalVal(allocator, scenario.get("threads"));
    const barrier_s = try fmtOptionalVal(allocator, document.get("timing_overhead_ns"));
    const timed_s = try fmtOptionalVal(allocator, document.get("timed_interval_ns"));
    const ratio_s = try fmtOptionalVal(allocator, document.get("timing_overhead_ratio"));
    const ratio_min_s = try fmtOptionalVal(allocator, document.get("ratio_at_minimum_timed_interval"));
    const runner_name = if (host.get("runner_name")) |v| (if (v == .string) v.string else "") else "";
    const fp_sha = host_pair.get("host_fingerprint_sha256").?.string;

    return std.fmt.allocPrint(
        allocator,
        "# WASI threaded benchmark quality failure\n\n" ++
            "- Stage: `{s}`\n" ++
            "- Reason: `{s}`\n" ++
            "- Scenario: `{s}` / `{s}` / {s} threads\n" ++
            "- Barrier: `{s}` ns\n" ++
            "- Timed interval: `{s}` ns\n" ++
            "- Ratio: `{s}`\n" ++
            "- Ratio at minimum interval: `{s}`\n" ++
            "- Fixed limit: `< {d}`\n" ++
            "- Pilot clock-resolution minimum / corrected cap / host-wall cap: `{d}` / `{d}` / `{d}` ns\n" ++
            "- Corrected cap scope: `{s}`\n" ++
            "- Projected evidence minimum / benchmark limit / job reserve: `{d}` / `{d}` / `{d}` ns\n" ++
            "- Runner: `{s}`\n" ++
            "- Host fingerprint: `{s}`\n\n" ++
            "All {d} preflight samples are retained in `failure-diagnostic.json`.\n",
        .{
            document.get("stage").?.string,
            document.get("reason").?.string,
            wl,
            mode,
            threads_s,
            barrier_s,
            timed_s,
            ratio_s,
            ratio_min_s,
            document.get("timing_overhead_ratio_limit").?.float,
            document.get("pilot_clock_resolution_minimum_ns").?.integer,
            document.get("maximum_pilot_corrected_ns").?.integer,
            document.get("maximum_pilot_host_wall_ns").?.integer,
            document.get("maximum_pilot_corrected_scope").?.string,
            document.get("projected_evidence_minimum_ns").?.integer,
            document.get("projected_benchmark_limit_ns").?.integer,
            document.get("job_non_benchmark_reserve_ns").?.integer,
            runner_name,
            fp_sha,
            preflight_samples.items.len,
        },
    );
}

pub const FailureDiagnosticArgs = struct {
    output: []const u8,
    stage: []const u8,
    reason: []const u8,
    scenario: ObjectMap,
    timing_overhead_ns: ?i64 = null,
    timed_interval_ns: ?i64 = null,
    raw_elapsed_ns: ?i64 = null,
    timing_overhead_ppm: ?i64 = null,
    minimum_interval_ns: i64,
    host: ObjectMap,
    host_pair: ObjectMap,
    host_quiescence_at_start: ObjectMap,
    host_quiescence_at_failure: ObjectMap,
    preflight_samples: []const ObjectMap,
    message: []const u8,
    ratio_at_minimum_timed_interval: ?f64 = null,
};

pub fn writeFailureDiagnostic(
    allocator: Allocator,
    io: std.Io,
    args: FailureDiagnosticArgs,
) !ObjectMap {
    const ratio: ?f64 = if (args.timing_overhead_ns != null and args.raw_elapsed_ns != null and args.raw_elapsed_ns.? > 0)
        @as(f64, @floatFromInt(args.timing_overhead_ns.?)) / @as(f64, @floatFromInt(args.raw_elapsed_ns.?))
    else
        null;
    const ratio_at_min: ?f64 = if (args.ratio_at_minimum_timed_interval) |r|
        r
    else if (args.timing_overhead_ns) |ov|
        @as(f64, @floatFromInt(ov)) / @as(f64, @floatFromInt(args.minimum_interval_ns + ov))
    else
        null;

    var samples_arr = Array.init(allocator);
    for (args.preflight_samples) |s| {
        try samples_arr.append(.{ .object = s });
    }

    var doc: ObjectMap = .empty;
    try doc.put(allocator, "schema_version", .{ .integer = spec.REPORT_SCHEMA_VERSION });
    try doc.put(allocator, "kind", .{ .string = "wasi-thread-benchmark-quality-failure" });
    try doc.put(allocator, "collected_at", .{ .string = try benchmark_schema.collectedAt(allocator, io) });
    try doc.put(allocator, "stage", .{ .string = args.stage });
    try doc.put(allocator, "reason", .{ .string = args.reason });
    try doc.put(allocator, "message", .{ .string = args.message });
    try doc.put(allocator, "scenario", .{ .object = args.scenario });
    try doc.put(allocator, "timing_overhead_ns", if (args.timing_overhead_ns) |v| .{ .integer = v } else .null);
    try doc.put(allocator, "timed_interval_ns", if (args.timed_interval_ns) |v| .{ .integer = v } else .null);
    try doc.put(allocator, "raw_elapsed_ns", if (args.raw_elapsed_ns) |v| .{ .integer = v } else .null);
    try doc.put(allocator, "timing_overhead_ppm", if (args.timing_overhead_ppm) |v| .{ .integer = v } else .null);
    try doc.put(allocator, "timing_overhead_ratio", if (ratio) |v| .{ .float = v } else .null);
    try doc.put(allocator, "ratio_at_minimum_timed_interval", if (ratio_at_min) |v| .{ .float = v } else .null);
    try doc.put(allocator, "timing_overhead_ratio_limit", .{ .float = spec.TIMING_OVERHEAD_RATIO_LIMIT });
    try doc.put(allocator, "pilot_clock_resolution_minimum_ns", .{ .integer = spec.PILOT_CLOCK_RESOLUTION_MINIMUM_NS });
    try doc.put(allocator, "maximum_pilot_corrected_ns", .{ .integer = spec.MAXIMUM_PILOT_CORRECTED_NS });
    try doc.put(allocator, "maximum_pilot_corrected_scope", .{ .string = "per-process-monotonic-or-per-worker-process-cpu" });
    try doc.put(allocator, "maximum_pilot_host_wall_ns", .{ .integer = spec.MAXIMUM_PILOT_HOST_WALL_NS });
    try doc.put(allocator, "projected_evidence_minimum_ns", .{ .integer = spec.PROJECTED_EVIDENCE_MINIMUM_NS });
    try doc.put(allocator, "projected_benchmark_limit_ns", .{ .integer = spec.PROJECTED_BENCHMARK_LIMIT_NS });
    try doc.put(allocator, "job_non_benchmark_reserve_ns", .{ .integer = spec.JOB_NON_BENCHMARK_RESERVE_NS });
    try doc.put(allocator, "minimum_timed_interval_ns", .{ .integer = args.minimum_interval_ns });
    try doc.put(allocator, "maximum_preflight_barrier_ns", .{ .integer = try maximumPreflightBarrierNs(args.minimum_interval_ns) });
    try doc.put(allocator, "host", .{ .object = args.host });
    try doc.put(allocator, "host_pair", .{ .object = args.host_pair });
    try doc.put(allocator, "host_quiescence_at_start", .{ .object = args.host_quiescence_at_start });
    try doc.put(allocator, "host_quiescence_at_failure", .{ .object = args.host_quiescence_at_failure });
    try doc.put(allocator, "preflight_samples", .{ .array = samples_arr });

    const json_path = try std.fs.path.join(allocator, &.{ args.output, "failure-diagnostic.json" });
    const md_path = try std.fs.path.join(allocator, &.{ args.output, "failure-diagnostic.md" });
    try spec.atomicWriteJson(allocator, io, json_path, .{ .object = doc });
    const md = try failureDiagnosticMarkdown(allocator, doc);
    try spec.atomicWriteText(allocator, io, md_path, try std.fmt.allocPrint(allocator, "{s}\n", .{md}));
    return doc;
}

pub const TRANSIENT_ZIG_FETCH_ERROR_MARKERS = [_][]const u8{
    "NameServerFailure",
    "TemporaryNameServerFailure",
    "ConnectionResetByPeer",
    "ConnectionTimedOut",
    "unable to discover remote git server capabilities",
    "server connection failed",
};

pub fn isTransientZigFetchError(stderr: ?[]const u8) bool {
    const text = stderr orelse return false;
    if (text.len == 0) return false;
    for (TRANSIENT_ZIG_FETCH_ERROR_MARKERS) |marker| {
        if (std.mem.indexOf(u8, text, marker) != null) return true;
    }
    return false;
}

pub fn controlledEnv(
    allocator: Allocator,
    io: std.Io,
    environ: std.process.Environ,
    cache_root: []const u8,
) HarnessError!std.process.Environ.Map {
    var env_map = environ.createMap(allocator) catch return error.OutOfMemory;
    const parent = std.fs.path.dirname(cache_root) orelse cache_root;
    const inherited_global = std.mem.trim(u8, environ.getPosix("ZIG_GLOBAL_CACHE_DIR") orelse "", " \t\r\n");
    const global_cache = if (inherited_global.len > 0)
        try allocator.dupe(u8, inherited_global)
    else
        try std.fs.path.join(allocator, &.{ parent, "global" });
    const local_cache = try std.fs.path.join(allocator, &.{ cache_root, "local" });
    const temp = try std.fs.path.join(allocator, &.{ cache_root, "tmp" });
    for ([_][]const u8{ cache_root, global_cache, local_cache, temp }) |dir| {
        std.Io.Dir.cwd().createDirPath(io, dir) catch {};
    }
    try env_map.put("ZIG_GLOBAL_CACHE_DIR", global_cache);
    try env_map.put("ZIG_LOCAL_CACHE_DIR", local_cache);
    try env_map.put("TMPDIR", temp);
    try env_map.put("LANG", "C");
    try env_map.put("LC_ALL", "C");
    try env_map.put("TZ", "UTC");
    return env_map;
}

pub fn zigBuildCommand(allocator: Allocator, io: std.Io, repo: []const u8, optimize: []const u8) ![]const []const u8 {
    var cmd: std.ArrayList([]const u8) = .empty;
    try cmd.appendSlice(allocator, &.{
        "zig",
        "build",
        try std.fmt.allocPrint(allocator, "-Doptimize={s}", .{optimize}),
    });
    const build_zig_path = try std.fs.path.join(allocator, &.{ repo, "build.zig" });
    if (std.Io.Dir.cwd().readFileAlloc(io, build_zig_path, allocator, .limited(4 * 1024 * 1024))) |text| {
        if (std.mem.indexOf(u8, text, "\"benchmark-cpu\"") != null) {
            try cmd.append(allocator, "-Dbenchmark-cpu=baseline");
        }
    } else |_| {}
    return cmd.toOwnedSlice(allocator);
}

pub const BuildVariantArgs = struct {
    repo: []const u8,
    root: []const u8,
    mode: []const u8,
    threads_enabled: bool,
    optimize: []const u8,
    target: ?[]const u8,
    source: ObjectMap,
    rebuild: bool,
    compiler_toggle: bool,
};

pub fn buildVariant(
    allocator: Allocator,
    io: std.Io,
    environ: std.process.Environ,
    args: BuildVariantArgs,
) HarnessError!Build {
    const name = try std.fmt.allocPrint(
        allocator,
        "{s}-{s}",
        .{ if (args.threads_enabled) "enabled" else "disabled", args.mode },
    );
    const base_cmd = try zigBuildCommand(allocator, io, args.repo, args.optimize);
    const zig_ver = try benchmark_schema.commandIdentity(allocator, io, &.{ base_cmd[0], "version" });

    var parts: ObjectMap = .empty;
    try parts.put(allocator, "build_source_sha256", args.source.get("build_source_sha256").?);
    try parts.put(allocator, "mode", .{ .string = args.mode });
    try parts.put(allocator, "threads_enabled", .{ .bool = args.threads_enabled });
    try parts.put(allocator, "optimize", .{ .string = args.optimize });
    try parts.put(allocator, "target", .{ .string = args.target orelse "native" });
    try parts.put(allocator, "compiler_toggle", .{ .bool = args.compiler_toggle });
    try parts.put(allocator, "thread_manager_toggle", .{ .bool = args.threads_enabled });
    try parts.put(allocator, "zig", .{ .string = zig_ver });

    const key = try spec.cacheKey(allocator, .{ .object = parts });
    const prefix = try std.fs.path.join(
        allocator,
        &.{ args.root, "builds", try std.fmt.allocPrint(allocator, "{s}-{s}", .{ name, key[0..16] }) },
    );
    const marker = try std.fs.path.join(allocator, &.{ prefix, "benchmark-build.json" });
    const wamr = try std.fs.path.join(allocator, &.{ prefix, "bin", "wamr" });
    const wamrc: ?[]const u8 = if (std.mem.eql(u8, args.mode, "aot"))
        try std.fs.path.join(allocator, &.{ prefix, "bin", "wamrc" })
    else
        null;

    const wamr_exists = if (std.Io.Dir.cwd().statFile(io, wamr, .{})) |_| true else |_| false;
    const wamrc_exists = if (wamrc) |wc| (if (std.Io.Dir.cwd().statFile(io, wc, .{})) |_| true else |_| false) else true;

    if (!args.rebuild and wamr_exists and wamrc_exists) {
        if (std.Io.Dir.cwd().readFileAlloc(io, marker, allocator, .limited(1024 * 1024))) |marker_bytes| {
            if (std.json.parseFromSlice(Value, allocator, marker_bytes, .{})) |parsed| {
                if (parsed.value == .object) {
                    const st_obj = parsed.value.object;
                    const sv = st_obj.get("schema_version");
                    const sk = st_obj.get("key");
                    const sc = st_obj.get("command");
                    if (sv != null and sv.? == .integer and sv.?.integer == benchmark_schema.SCHEMA_VERSION and
                        sk != null and sk.? == .string and std.mem.eql(u8, sk.?.string, key) and
                        sc != null and sc.? == .array)
                    {
                        var cmd_list: std.ArrayList([]const u8) = .empty;
                        for (sc.?.array.items) |ci| {
                            if (ci == .string) try cmd_list.append(allocator, ci.string);
                        }
                        return .{
                            .name = name,
                            .mode = args.mode,
                            .threads_enabled = args.threads_enabled,
                            .prefix = prefix,
                            .wamr = wamr,
                            .wamrc = wamrc,
                            .key = key,
                            .command = try cmd_list.toOwnedSlice(allocator),
                            .reused = true,
                        };
                    }
                }
            } else |_| {}
        } else |_| {}
    }

    var cmd: std.ArrayList([]const u8) = .empty;
    try cmd.appendSlice(allocator, base_cmd[0..2]);
    if (args.target) |t| {
        try cmd.append(allocator, try std.fmt.allocPrint(allocator, "-Dtarget={s}", .{t}));
    }
    try cmd.appendSlice(allocator, base_cmd[2..]);
    try cmd.appendSlice(allocator, &.{
        if (args.threads_enabled) "-Dlib_wasi_threads=true" else "-Dlib_wasi_threads=false",
        if (std.mem.eql(u8, args.mode, "interpreter")) "-Dinterp=true" else "-Dinterp=false",
        if (std.mem.eql(u8, args.mode, "aot")) "-Daot=true" else "-Daot=false",
        if (std.mem.eql(u8, args.mode, "interpreter")) "-Dbenchmark-interp-fuel=4000000000" else "-Dbenchmark-interp-fuel=100000000",
        if (args.compiler_toggle) "-Dbenchmark-cancel-point-toggle=true" else "-Dbenchmark-cancel-point-toggle=false",
        if (args.threads_enabled) "-Dbenchmark-wasi-thread-manager-toggle=true" else "-Dbenchmark-wasi-thread-manager-toggle=false",
        "--prefix",
        prefix,
    });

    const cache_dir = try std.fs.path.join(allocator, &.{ args.root, "cache", name });
    var env_map = try controlledEnv(allocator, io, environ, cache_dir);
    defer env_map.deinit();
    std.Io.Dir.cwd().createDirPath(io, prefix) catch {};

    var attempt: usize = 1;
    while (attempt <= 3) : (attempt += 1) {
        const res = try runProcess(allocator, io, cmd.items, args.repo, 600.0, &env_map);
        if (res.exit_code == 0) break;
        if (res.stderr.len > 0) {
            std.debug.print("{s}\n", .{res.stderr});
        }
        if (attempt < 3 and isTransientZigFetchError(res.stderr)) {
            io.sleep(std.Io.Duration.fromSeconds(@intCast(2 * attempt)), .awake) catch {};
            continue;
        }
        return fail("build failed for {s}: exit {d}", .{ name, res.exit_code });
    }

    if (std.Io.Dir.cwd().statFile(io, wamr, .{})) |_| {} else |_| {
        return fail("build {s} did not produce expected binaries", .{name});
    }
    if (wamrc) |wc| {
        if (std.Io.Dir.cwd().statFile(io, wc, .{})) |_| {} else |_| {
            return fail("build {s} did not produce expected binaries", .{name});
        }
    }

    var cmd_arr = Array.init(allocator);
    for (cmd.items) |arg| try cmd_arr.append(.{ .string = arg });
    var marker_obj: ObjectMap = .empty;
    try marker_obj.put(allocator, "schema_version", .{ .integer = benchmark_schema.SCHEMA_VERSION });
    try marker_obj.put(allocator, "key", .{ .string = key });
    try marker_obj.put(allocator, "parts", .{ .object = parts });
    try marker_obj.put(allocator, "command", .{ .array = cmd_arr });
    try spec.atomicWriteJson(allocator, io, marker, .{ .object = marker_obj });

    return .{
        .name = name,
        .mode = args.mode,
        .threads_enabled = args.threads_enabled,
        .prefix = prefix,
        .wamr = wamr,
        .wamrc = wamrc,
        .key = key,
        .command = try cmd.toOwnedSlice(allocator),
        .reused = false,
    };
}

pub fn compileAotFixtures(
    allocator: Allocator,
    io: std.Io,
    environ: std.process.Environ,
    repo: []const u8,
    output: []const u8,
    compiler: Build,
    arch: []const u8,
) HarnessError!ObjectMap {
    const wamrc = compiler.wamrc orelse return fail("AOT compiler build is missing wamrc", .{});
    const artifacts_dir = try std.fs.path.join(allocator, &.{ output, "aot" });
    std.Io.Dir.cwd().createDirPath(io, artifacts_dir) catch {};

    const single_wasm = try std.fs.path.join(allocator, &.{ repo, spec.FIXTURES[0].path });
    const threaded_wasm = try std.fs.path.join(allocator, &.{ repo, spec.FIXTURES[1].path });
    const single_out = try std.fs.path.join(allocator, &.{ artifacts_dir, "single.cwasm" });
    const on_out = try std.fs.path.join(allocator, &.{ artifacts_dir, "threaded-polls-on.cwasm" });
    const off_out = try std.fs.path.join(allocator, &.{ artifacts_dir, "threaded-polls-off.cwasm" });

    const cache_dir = try std.fs.path.join(allocator, &.{ output, "cache", "compile-aot" });
    var env_map = try controlledEnv(allocator, io, environ, cache_dir);
    defer env_map.deinit();

    const jobs = [_]struct { name: []const u8, argv: []const []const u8, out: []const u8 }{
        .{
            .name = "single",
            .argv = &.{ wamrc, "compile", "--target", arch, single_wasm, "-o", single_out },
            .out = single_out,
        },
        .{
            .name = "threaded-polls-on",
            .argv = &.{ wamrc, "compile", "--target", arch, threaded_wasm, "-o", on_out },
            .out = on_out,
        },
        .{
            .name = "threaded-polls-off",
            .argv = &.{ wamrc, "compile", "--target", arch, "--benchmark-disable-cancel-points", threaded_wasm, "-o", off_out },
            .out = off_out,
        },
    };
    var out: ObjectMap = .empty;
    for (jobs) |job| {
        const res = try runProcess(allocator, io, job.argv, repo, 120.0, &env_map);
        if (res.exit_code != 0) {
            return fail("AOT fixture compilation failed for {s}", .{job.name});
        }
        try out.put(allocator, job.name, .{ .string = job.out });
    }
    return out;
}

pub fn aotTextSection(allocator: Allocator, io: std.Io, path: []const u8) HarnessError![]const u8 {
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024)) catch {
        return fail("not a WAMR AOT file: {s}", .{path});
    };
    if (data.len < 8 or !std.mem.eql(u8, data[0..4], "\x00aot")) {
        return fail("not a WAMR AOT file: {s}", .{path});
    }
    const version = std.mem.readInt(u32, data[4..8], .little);
    if (version != spec.AOT_VERSION) {
        return fail("unsupported WAMR AOT version {d} in {s}; expected {d}", .{ version, path, spec.AOT_VERSION });
    }
    var pos: usize = 8;
    while (pos + 8 <= data.len) {
        const sec_type = std.mem.readInt(u32, data[pos..][0..4], .little);
        const sec_size: usize = @intCast(std.mem.readInt(u32, data[pos + 4 ..][0..4], .little));
        pos += 8;
        if (pos + sec_size > data.len) {
            return fail("truncated AOT section: {s}", .{path});
        }
        if (sec_type == 2) {
            return allocator.dupe(u8, data[pos .. pos + sec_size]);
        }
        pos += sec_size;
    }
    return fail("AOT text section missing: {s}", .{path});
}

fn countSubslice(haystack: []const u8, needle: []const u8) usize {
    if (needle.len == 0 or haystack.len < needle.len) return 0;
    var count: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, haystack, pos, needle)) |found| {
        count += 1;
        pos = found + needle.len;
    }
    return count;
}

pub fn aotArtifactReport(
    allocator: Allocator,
    io: std.Io,
    artifacts: ObjectMap,
    arch: []const u8,
) HarnessError!ObjectMap {
    const signature = spec.cancelPollSignature(arch) orelse return fail("unsupported cancel-poll architecture: {s}", .{arch});
    var report: ObjectMap = .empty;
    var text_sections: std.StringHashMapUnmanaged([]const u8) = .empty;

    var it = artifacts.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const path = entry.value_ptr.string;
        const text = try aotTextSection(allocator, io, path);
        try text_sections.put(allocator, name, text);
        const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch return fail("missing AOT artifact {s}", .{path});
        const sha = try spec.sha256File(allocator, io, path);
        var item: ObjectMap = .empty;
        try item.put(allocator, "path", .{ .string = path });
        try item.put(allocator, "sha256", .{ .string = sha });
        try item.put(allocator, "file_bytes", .{ .integer = @intCast(stat.size) });
        try item.put(allocator, "text_bytes", .{ .integer = @intCast(text.len) });
        try report.put(allocator, name, .{ .object = item });
    }

    const on_text = text_sections.get("threaded-polls-on") orelse return fail("missing threaded-polls-on", .{});
    const off_text = text_sections.get("threaded-polls-off") orelse return fail("missing threaded-polls-off", .{});
    const sites_enabled: i64 = @intCast(countSubslice(on_text, signature));
    const sites_disabled: i64 = @intCast(countSubslice(off_text, signature));
    const on_bytes: i64 = @intCast(on_text.len);
    const off_bytes: i64 = @intCast(off_text.len);
    const delta = on_bytes - off_bytes;
    if (sites_enabled <= 0 or sites_disabled != 0) {
        return fail("cancel-poll signature count is invalid for {s}: enabled={d}, disabled={d}", .{ arch, sites_enabled, sites_disabled });
    }
    if (delta <= 0 or @rem(delta, sites_enabled) != 0) {
        return fail("cancel-poll text delta {d} cannot be attributed to {d} detected sites for {s}", .{ delta, sites_enabled, arch });
    }
    const poll_bytes = @divExact(delta, sites_enabled);
    var hex_buf = try allocator.alloc(u8, signature.len * 2);
    const hex_chars = "0123456789abcdef";
    for (signature, 0..) |b, idx| {
        hex_buf[idx * 2] = hex_chars[b >> 4];
        hex_buf[idx * 2 + 1] = hex_chars[b & 0x0F];
    }

    var cps: ObjectMap = .empty;
    try cps.put(allocator, "architecture", .{ .string = arch });
    try cps.put(allocator, "detection", .{ .string = "machine-code-signature" });
    try cps.put(allocator, "signature_hex", .{ .string = hex_buf });
    try cps.put(allocator, "bytes_per_site", .{ .integer = poll_bytes });
    try cps.put(allocator, "text_delta_bytes", .{ .integer = delta });
    try cps.put(allocator, "sites_enabled", .{ .integer = sites_enabled });
    try cps.put(allocator, "sites_disabled", .{ .integer = sites_disabled });
    try report.put(allocator, "cancel_poll_static", .{ .object = cps });
    return report;
}

pub fn buildToolReport(
    allocator: Allocator,
    io: std.Io,
    builds: []const Build,
    runner: []const []const u8,
) !ObjectMap {
    var report: ObjectMap = .empty;
    for (builds) |b| {
        const binary_runner: []const []const u8 = if (std.mem.eql(u8, b.name, "host-compiler")) &.{} else runner;
        var cmd_arr = Array.init(allocator);
        for (b.command) |arg| try cmd_arr.append(.{ .string = arg });

        var wamr_argv: std.ArrayList([]const u8) = .empty;
        try wamr_argv.appendSlice(allocator, binary_runner);
        try wamr_argv.appendSlice(allocator, &.{ b.wamr, "version" });
        const wamr_ver = try benchmark_schema.commandIdentity(allocator, io, wamr_argv.items);
        var wamr_obj: ObjectMap = .empty;
        try wamr_obj.put(allocator, "path", .{ .string = b.wamr });
        try wamr_obj.put(allocator, "sha256", .{ .string = try spec.sha256File(allocator, io, b.wamr) });
        try wamr_obj.put(allocator, "version", .{ .string = wamr_ver });

        var entry: ObjectMap = .empty;
        try entry.put(allocator, "mode", .{ .string = b.mode });
        try entry.put(allocator, "threads_enabled", .{ .bool = b.threads_enabled });
        try entry.put(allocator, "cache_key", .{ .string = b.key });
        try entry.put(allocator, "cache_reused", .{ .bool = b.reused });
        try entry.put(allocator, "build_command", .{ .array = cmd_arr });
        try entry.put(allocator, "wamr", .{ .object = wamr_obj });

        if (b.wamrc) |wc| {
            var wc_argv: std.ArrayList([]const u8) = .empty;
            try wc_argv.appendSlice(allocator, binary_runner);
            try wc_argv.appendSlice(allocator, &.{ wc, "version" });
            const wc_ver = try benchmark_schema.commandIdentity(allocator, io, wc_argv.items);
            var wc_obj: ObjectMap = .empty;
            try wc_obj.put(allocator, "path", .{ .string = wc });
            try wc_obj.put(allocator, "sha256", .{ .string = try spec.sha256File(allocator, io, wc) });
            try wc_obj.put(allocator, "version", .{ .string = wc_ver });
            try entry.put(allocator, "wamrc", .{ .object = wc_obj });
        }
        try report.put(allocator, b.name, .{ .object = entry });
    }
    return report;
}

pub const MeasureOnceArgs = struct {
    repo: []const u8,
    runner: []const []const u8,
    cpu_placement: ObjectMap,
    wamr_path: []const u8,
    module_path: []const u8,
    workload: []const u8,
    threads: i64,
    iterations: u64,
    timeout: f64,
    min_interval_ns: i64,
    record_fields: ObjectMap,
    enforce_timing_quality: bool = true,
    cpu_affinity_override: ?[]const i64 = null,
    runtime_args: []const []const u8 = &.{},
    guest_clock_id: ?[]const u8 = null,
};

pub fn measureOnce(
    allocator: Allocator,
    io: std.Io,
    args: MeasureOnceArgs,
) HarnessError!ObjectMap {
    const resolved_clock_id = if (args.guest_clock_id) |cid| cid else try spec.expectedGuestClockId(args.workload);
    var guest_args: std.ArrayList([]const u8) = .empty;
    if (std.mem.eql(u8, args.workload, "single-hot")) {
        try guest_args.append(allocator, try std.fmt.allocPrint(allocator, "{d}", .{args.iterations}));
    } else {
        try guest_args.appendSlice(allocator, &.{
            args.workload,
            try std.fmt.allocPrint(allocator, "{d}", .{args.threads}),
            try std.fmt.allocPrint(allocator, "{d}", .{args.iterations}),
        });
        if (std.mem.eql(u8, args.workload, "hot") or std.mem.eql(u8, args.workload, "atomic")) {
            if (std.mem.eql(u8, resolved_clock_id, spec.WASI_PROCESS_CPU_CLOCK_ID)) {
                try guest_args.append(allocator, spec.WASI_PROCESS_CPU_CLOCK_MODE);
            } else if (std.mem.eql(u8, resolved_clock_id, spec.WASI_MONOTONIC_CLOCK_ID)) {
                try guest_args.append(allocator, spec.WASI_MONOTONIC_CLOCK_MODE);
            } else {
                return fail("unsupported guest clock identity {s}", .{resolved_clock_id});
            }
        }
    }

    const cpu_affinity = if (args.cpu_affinity_override) |ov|
        try allocator.dupe(i64, ov)
    else
        try spec.cpuAffinityFor(allocator, args.cpu_placement, args.workload, args.threads);

    const allowed_arr = (args.cpu_placement.get("allowed_logical_cpus") orelse return fail("measurement CPU affinity is invalid", .{})).array;
    if (cpu_affinity.len == 0) return fail("measurement CPU affinity is invalid", .{});
    for (cpu_affinity, 0..) |cpu, idx| {
        for (cpu_affinity[0..idx]) |prev| {
            if (prev == cpu) return fail("measurement CPU affinity is invalid", .{});
        }
        var in_allowed = false;
        for (allowed_arr.items) |av| {
            if (av == .integer and av.integer == cpu) in_allowed = true;
        }
        if (!in_allowed) return fail("measurement CPU affinity is invalid", .{});
    }

    var cpu_strs: std.ArrayList([]const u8) = .empty;
    for (cpu_affinity) |c| {
        try cpu_strs.append(allocator, try std.fmt.allocPrint(allocator, "{d}", .{c}));
    }
    const cpu_list_str = try std.mem.join(allocator, ",", cpu_strs.items);

    var command: std.ArrayList([]const u8) = .empty;
    try command.appendSlice(allocator, &.{ "taskset", "--cpu-list", cpu_list_str });
    try command.appendSlice(allocator, args.runner);
    try command.appendSlice(allocator, &.{ args.wamr_path, "run" });
    try command.appendSlice(allocator, args.runtime_args);
    try command.append(allocator, args.module_path);
    try command.appendSlice(allocator, guest_args.items);

    const started: i64 = @intCast(std.Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds);
    const proc_res = runProcess(allocator, io, command.items, args.repo, args.timeout, null) catch |err| {
        const msg = spec.lastErrorMessage();
        if (std.mem.indexOf(u8, msg, "command timed out") != null) {
            const classification = if (std.mem.eql(u8, args.workload, "wait-notify"))
                "notification-loss"
            else
                "guest-watchdog-timeout";
            const dup_msg = try allocator.dupe(u8, msg);
            return fail("guest failure classification={s}: {s}", .{ classification, dup_msg });
        }
        return err;
    };
    const finished: i64 = @intCast(std.Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds);
    const host_wall_elapsed_ns = @max(@as(i64, 1), finished - started);

    if (proc_res.exit_code != 0) {
        const classification = try spec.classifyGuestFailure(allocator, proc_res.stderr, args.workload);
        const joined = try std.mem.join(allocator, " ", command.items);
        return fail("guest failure classification={s}; exit {d}: {s}\n{s}", .{
            classification,
            proc_res.exit_code,
            joined,
            proc_res.stderr,
        });
    }

    const expected = try spec.expectedResult(
        args.workload,
        args.threads,
        @as(i128, args.iterations),
        resolved_clock_id,
    );
    const guest = try spec.parseGuestResult(
        allocator,
        proc_res.stdout,
        expected,
        args.min_interval_ns,
        args.enforce_timing_quality,
    );

    const operations = spec.valueAsU64(guest.get("operations").?).?;
    const guest_elapsed_ns = guest.get("elapsed_ns").?.integer;
    const throughput = @as(f64, @floatFromInt(operations)) / (@as(f64, @floatFromInt(guest_elapsed_ns)) / 1e9);

    var cancel_polls_val: Value = .null;
    if (std.mem.eql(u8, args.workload, "single-hot") or std.mem.eql(u8, args.workload, "hot")) {
        if (args.record_fields.get("cancel_points")) |cp_v| {
            if (cp_v == .string) {
                if (std.mem.eql(u8, cp_v.string, "on")) {
                    cancel_polls_val = .{ .float = 1.0 };
                } else if (std.mem.eql(u8, cp_v.string, "off") or std.mem.eql(u8, cp_v.string, "not-applicable")) {
                    cancel_polls_val = .{ .float = 0.0 };
                }
            }
        }
    }

    var out: ObjectMap = .empty;
    var rf_it = args.record_fields.iterator();
    while (rf_it.next()) |entry| {
        try out.put(allocator, entry.key_ptr.*, entry.value_ptr.*);
    }

    var cmd_arr = Array.init(allocator);
    for (command.items) |arg| try cmd_arr.append(.{ .string = arg });
    var aff_arr = Array.init(allocator);
    for (cpu_affinity) |c| try aff_arr.append(.{ .integer = c });

    var correctness_obj: ObjectMap = .empty;
    try correctness_obj.put(allocator, "passed", .{ .bool = true });
    try correctness_obj.put(allocator, "expected", .{ .object = try expected.toObjectMap(allocator) });
    try correctness_obj.put(allocator, "actual", .{ .object = guest });

    try out.put(allocator, "command", .{ .array = cmd_arr });
    try out.put(allocator, "cpu_affinity", .{ .array = aff_arr });
    try out.put(allocator, "host_started_ns", .{ .integer = started });
    try out.put(allocator, "host_finished_ns", .{ .integer = finished });
    try out.put(allocator, "elapsed_ns", .{ .integer = guest_elapsed_ns });
    try out.put(allocator, "guest_elapsed_ns", .{ .integer = guest_elapsed_ns });
    try out.put(allocator, "raw_guest_elapsed_ns", guest.get("raw_elapsed_ns").?);
    try out.put(allocator, "timing_overhead_ns", guest.get("timing_overhead_ns").?);
    try out.put(allocator, "timing_overhead_ppm", guest.get("timing_overhead_ppm").?);
    try out.put(allocator, "host_wall_elapsed_ns", .{ .integer = host_wall_elapsed_ns });
    try out.put(allocator, "host_wall_over_guest", .{ .float = @as(f64, @floatFromInt(host_wall_elapsed_ns)) / @as(f64, @floatFromInt(guest_elapsed_ns)) });
    try out.put(allocator, "metric_kind", guest.get("metric_kind").?);
    try out.put(allocator, "cancel_polls_per_operation", cancel_polls_val);
    try spec.putU64Json(allocator, &out, "operations", operations);
    try out.put(allocator, "throughput_ops_per_second", .{ .float = throughput });
    try out.put(allocator, "per_thread_ops_per_second", .{ .float = throughput / @as(f64, @floatFromInt(args.threads)) });
    try out.put(allocator, "guest", .{ .object = guest });
    try out.put(allocator, "correct", .{ .bool = true });
    try out.put(allocator, "correctness", .{ .object = correctness_obj });
    try out.put(allocator, "stdout", .{ .string = proc_res.stdout });
    try out.put(allocator, "stderr", .{ .string = proc_res.stderr });
    return out;
}

const ConcurrentWorkerCtx = struct {
    io: std.Io,
    args: MeasureOnceArgs,
    result: ?ObjectMap = null,
    err: ?HarnessError = null,
    err_msg: [512]u8 = undefined,
    err_len: usize = 0,
    tq: spec.TimingQualityInfo = .{},

    fn run(self: *ConcurrentWorkerCtx) void {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        // Leak into page allocator or clone back in caller; we keep arena alive for process lifetime
        const alloc = arena.allocator();
        if (measureOnce(alloc, self.io, self.args)) |res| {
            self.result = res;
        } else |e| {
            self.err = e;
            const msg = spec.lastErrorMessage();
            const n = @min(msg.len, self.err_msg.len);
            @memcpy(self.err_msg[0..n], msg[0..n]);
            self.err_len = n;
            self.tq = spec.last_timing_quality;
        }
    }
};

pub fn measureConcurrently(
    allocator: Allocator,
    io: std.Io,
    invocations: []const MeasureOnceArgs,
) HarnessError![]ObjectMap {
    if (invocations.len != 2) {
        return fail("concurrent measurement requires exactly two invocations", .{});
    }
    var ctx0: ConcurrentWorkerCtx = .{ .io = io, .args = invocations[0] };
    var ctx1: ConcurrentWorkerCtx = .{ .io = io, .args = invocations[1] };

    const t0 = std.Thread.spawn(.{}, ConcurrentWorkerCtx.run, .{&ctx0}) catch {
        return fail("failed to spawn concurrent measurement thread 0", .{});
    };
    const t1 = std.Thread.spawn(.{}, ConcurrentWorkerCtx.run, .{&ctx1}) catch {
        t0.join();
        return fail("failed to spawn concurrent measurement thread 1", .{});
    };
    t0.join();
    t1.join();

    for ([_]*ConcurrentWorkerCtx{ &ctx0, &ctx1 }) |ctx| {
        if (ctx.err) |e| {
            if (e == error.TimingQualityError) {
                return spec.failTimingQuality(ctx.tq, "{s}", .{ctx.err_msg[0..ctx.err_len]});
            }
            return fail("{s}", .{ctx.err_msg[0..ctx.err_len]});
        }
    }

    var records = try allocator.alloc(ObjectMap, 2);
    records[0] = (try spec.jsonClone(allocator, .{ .object = ctx0.result.? })).object;
    records[1] = (try spec.jsonClone(allocator, .{ .object = ctx1.result.? })).object;

    const max_start = @max(records[0].get("host_started_ns").?.integer, records[1].get("host_started_ns").?.integer);
    const min_finish = @min(records[0].get("host_finished_ns").?.integer, records[1].get("host_finished_ns").?.integer);
    if (max_start >= min_finish) {
        return fail("concurrent measurement host intervals did not overlap", .{});
    }
    return records;
}

pub const TrustedBarrierPreflightArgs = struct {
    repo: []const u8,
    runner: []const []const u8,
    cpu_placement: ObjectMap,
    wamr_path: []const u8,
    module_path: []const u8,
    thread_counts: []const i64,
    iterations_by_thread: ObjectMap,
    timeout: f64,
    minimum_interval_ns: i64,
    static_cancel_poll_sites: i64,
};

pub var last_preflight_samples: []ObjectMap = &.{};
pub var last_preflight_scenario: ObjectMap = .empty;

pub fn runTrustedBarrierPreflight(
    allocator: Allocator,
    io: std.Io,
    args: TrustedBarrierPreflightArgs,
) HarnessError!ObjectMap {
    var samples: std.ArrayList(ObjectMap) = .empty;
    var overheads: std.ArrayList(f64) = .empty;
    var all_accepted = true;

    for (args.thread_counts) |threads| {
        const cpu_affinity = try spec.cpuAffinityFor(allocator, args.cpu_placement, "hot", threads);
        var buf: [32]u8 = undefined;
        const tc_str = std.fmt.bufPrint(&buf, "{d}", .{threads}) catch unreachable;
        const base_iters = spec.valueAsU64(args.iterations_by_thread.get(tc_str).?).?;
        const iterations = base_iters * @as(u64, @intCast(cpu_affinity.len));

        var probe_index: i64 = 0;
        while (probe_index < spec.TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD) : (probe_index += 1) {
            var rf: ObjectMap = .empty;
            try rf.put(allocator, "mode", .{ .string = "aot" });
            try rf.put(allocator, "threads_enabled", .{ .bool = true });
            try rf.put(allocator, "cancel_points", .{ .string = "on" });
            try rf.put(allocator, "static_cancel_poll_sites", .{ .integer = args.static_cancel_poll_sites });
            try rf.put(allocator, "workload", .{ .string = "hot" });
            try rf.put(allocator, "threads", .{ .integer = threads });
            try spec.putU64Json(allocator, &rf, "iterations", iterations);

            const measured = measureOnce(allocator, io, .{
                .repo = args.repo,
                .runner = args.runner,
                .cpu_placement = args.cpu_placement,
                .wamr_path = args.wamr_path,
                .module_path = args.module_path,
                .workload = "hot",
                .threads = threads,
                .iterations = iterations,
                .timeout = args.timeout,
                .min_interval_ns = args.minimum_interval_ns,
                .enforce_timing_quality = false,
                .guest_clock_id = spec.WASI_MONOTONIC_CLOCK_ID,
                .cpu_affinity_override = cpu_affinity,
                .record_fields = rf,
            }) catch {
                last_preflight_samples = samples.items;
                var scen: ObjectMap = .empty;
                try scen.put(allocator, "mode", .{ .string = "aot" });
                try scen.put(allocator, "workload", .{ .string = "hot" });
                try scen.put(allocator, "clock_id", .{ .string = spec.WASI_MONOTONIC_CLOCK_ID });
                try scen.put(allocator, "iteration_multiplier", .{ .string = spec.TRUSTED_BARRIER_PREFLIGHT_ITERATION_MULTIPLIER });
                try scen.put(allocator, "threads", .{ .integer = threads });
                try scen.put(allocator, "probe_index", .{ .integer = probe_index });
                last_preflight_scenario = scen;
                return error.PreflightProbeError;
            };

            const overhead = measured.get("timing_overhead_ns").?.integer;
            const timed = measured.get("guest_elapsed_ns").?.integer;
            const raw = measured.get("raw_guest_elapsed_ns").?.integer;
            const accepted = preflightSampleAccepted(overhead, timed, args.minimum_interval_ns);
            if (!accepted) all_accepted = false;
            try overheads.append(allocator, @floatFromInt(overhead));

            var sample: ObjectMap = .empty;
            try sample.put(allocator, "probe_index", .{ .integer = probe_index });
            try sample.put(allocator, "mode", .{ .string = "aot" });
            try sample.put(allocator, "workload", .{ .string = "hot" });
            try sample.put(allocator, "clock_id", measured.get("guest").?.object.get("clock_id").?);
            try sample.put(allocator, "threads", .{ .integer = threads });
            try spec.putU64Json(allocator, &sample, "iterations", iterations);
            try sample.put(allocator, "cpu_affinity", measured.get("cpu_affinity").?);
            try sample.put(allocator, "timing_overhead_ns", .{ .integer = overhead });
            try sample.put(allocator, "timed_interval_ns", .{ .integer = timed });
            try sample.put(allocator, "raw_elapsed_ns", .{ .integer = raw });
            try sample.put(allocator, "timing_overhead_ppm", measured.get("timing_overhead_ppm").?);
            try sample.put(allocator, "timing_overhead_ratio", .{ .float = @as(f64, @floatFromInt(overhead)) / @as(f64, @floatFromInt(raw)) });
            try sample.put(allocator, "ratio_at_minimum_timed_interval", .{
                .float = @as(f64, @floatFromInt(overhead)) / @as(f64, @floatFromInt(args.minimum_interval_ns + overhead)),
            });
            try sample.put(allocator, "accepted", .{ .bool = accepted });
            try samples.append(allocator, sample);
        }
    }

    const stats = benchmark_stats.summarize(allocator, overheads.items) catch return fail("failed to compute preflight stats", .{});

    var summary: ObjectMap = .empty;
    try summary.put(allocator, "minimum_barrier_ns", .{ .integer = @intFromFloat(stats.min) });
    if (@floor(stats.median) == stats.median) {
        try summary.put(allocator, "median_barrier_ns", .{ .integer = @intFromFloat(stats.median) });
    } else {
        try summary.put(allocator, "median_barrier_ns", .{ .float = stats.median });
    }
    try summary.put(allocator, "maximum_barrier_ns", .{ .integer = @intFromFloat(stats.max) });

    var tc_arr = Array.init(allocator);
    for (args.thread_counts) |tc| try tc_arr.append(.{ .integer = tc });
    var samp_arr = Array.init(allocator);
    for (samples.items) |s| try samp_arr.append(.{ .object = s });

    var out: ObjectMap = .empty;
    try out.put(allocator, "enabled", .{ .bool = true });
    try out.put(allocator, "status", .{ .string = if (all_accepted) "passed" else "failed" });
    try out.put(allocator, "probe_count", .{ .integer = @intCast(samples.items.len) });
    try out.put(allocator, "probes_per_thread", .{ .integer = spec.TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD });
    try out.put(allocator, "mode", .{ .string = "aot" });
    try out.put(allocator, "workload", .{ .string = "hot" });
    try out.put(allocator, "clock_id", .{ .string = spec.WASI_MONOTONIC_CLOCK_ID });
    try out.put(allocator, "iteration_multiplier", .{ .string = spec.TRUSTED_BARRIER_PREFLIGHT_ITERATION_MULTIPLIER });
    try out.put(allocator, "thread_counts", .{ .array = tc_arr });
    try out.put(allocator, "minimum_timed_interval_ns", .{ .integer = args.minimum_interval_ns });
    try out.put(allocator, "timing_overhead_ratio_limit", .{ .float = spec.TIMING_OVERHEAD_RATIO_LIMIT });
    try out.put(allocator, "maximum_accepted_barrier_ns", .{ .integer = try maximumPreflightBarrierNs(args.minimum_interval_ns) });
    try out.put(allocator, "acceptance_rule", .{ .string = spec.PREFLIGHT_ACCEPTANCE_RULE });
    try out.put(allocator, "summary", .{ .object = summary });
    try out.put(allocator, "samples", .{ .array = samp_arr });
    return out;
}
