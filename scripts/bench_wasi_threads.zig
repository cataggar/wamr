//! Native WASI threads benchmark harness and budget evaluator.
const std = @import("std");
const builtin = @import("builtin");
const benchmark_schema = @import("benchmark_schema.zig");
const benchmark_stats = @import("benchmark_stats.zig");
pub const spec = @import("wasi_thread_bench_spec.zig");
pub const sizing = @import("wasi_thread_bench_sizing.zig");
pub const runner = @import("wasi_thread_bench_runner.zig");
pub const report = @import("wasi_thread_bench_report.zig");

pub const Allocator = spec.Allocator;
pub const Value = spec.Value;
pub const ObjectMap = spec.ObjectMap;
pub const Array = spec.Array;
pub const HarnessError = spec.HarnessError;
const fail = spec.fail;
const require = spec.require;

pub const Args = struct {
    repo: ?[]const u8 = null,
    baseline_repo: ?[]const u8 = null,
    candidate_repo: ?[]const u8 = null,
    comparison_purpose: ?[]const u8 = null,
    output_dir: []const u8 = "zig-out/wasi-thread-bench",
    profile: []const u8 = "authoritative",
    warmups: ?i64 = null,
    samples: ?i64 = null,
    modes_arg: []const u8 = "both",
    modes: []const []const u8 = &spec.MODES,
    thread_counts: []const i64 = &spec.DEFAULT_THREAD_COUNTS,
    single_iterations: ?u64 = null,
    cancel_iterations: ?u64 = null,
    hot_iterations: ?u64 = null,
    atomic_iterations: ?u64 = null,
    atomic_total_iterations: ?u64 = null,
    wait_iterations: ?u64 = null,
    spawn_iterations: ?u64 = null,
    timeout: f64 = 90.0,
    min_interval_ms: f64 = spec.MIN_TIMED_INTERVAL_MS,
    platform_id: ?[]const u8 = null,
    runner_environment: ?[]const u8 = null,
    host_pair_id: ?[]const u8 = null,
    optimize: []const u8 = "ReleaseFast",
    target: ?[]const u8 = null,
    aot_target: ?[]const u8 = null,
    runner_prefix: []const u8 = "",
    budget: ?[]const u8 = null,
    no_budget: bool = false,
    rebuild: bool = false,
    trusted_calibration_preflight: bool = false,
    validate_report_path: ?[]const u8 = null,
    check_budget_path: ?[]const u8 = null,
};

fn parseThreadCounts(allocator: Allocator, raw: []const u8) HarnessError![]const i64 {
    var list: std.ArrayList(i64) = .empty;
    var it = std.mem.splitScalar(u8, raw, ',');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \t\r\n");
        if (trimmed.len == 0) continue;
        const val = std.fmt.parseInt(i64, trimmed, 10) catch {
            return fail("invalid thread count '{s}'", .{trimmed});
        };
        if (val < 1 or val > 32) {
            return fail("thread counts must be in 1..=32", .{});
        }
        for (list.items) |existing| {
            if (existing == val) return fail("thread counts must not contain duplicates", .{});
        }
        try list.append(allocator, val);
    }
    if (list.items.len == 0) return fail("thread counts must not be empty", .{});
    return list.toOwnedSlice(allocator);
}

fn parseModes(allocator: Allocator, raw: []const u8) HarnessError![]const []const u8 {
    if (std.mem.eql(u8, raw, "both")) return &spec.MODES;
    var list: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, raw, ',');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \t\r\n");
        if (trimmed.len == 0) continue;
        if (!std.mem.eql(u8, trimmed, "interpreter") and
            !std.mem.eql(u8, trimmed, "aot"))
        {
            return fail("unsupported mode '{s}'", .{trimmed});
        }
        var dup = false;
        for (list.items) |existing| {
            if (std.mem.eql(u8, existing, trimmed)) dup = true;
        }
        if (!dup) try list.append(allocator, trimmed);
    }
    if (list.items.len == 0) return fail("modes must not be empty", .{});
    return list.toOwnedSlice(allocator);
}

fn parsePositiveU64(raw: []const u8, flag: []const u8) HarnessError!u64 {
    const val = std.fmt.parseInt(i64, raw, 10) catch return fail("invalid {s}", .{flag});
    if (val <= 0) return fail("{s} must be > 0", .{flag});
    return @intCast(val);
}

pub fn parseArgs(allocator: Allocator, argv: []const []const u8) HarnessError!Args {
    var args: Args = .{};
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--repo")) {
            i += 1;
            if (i >= argv.len) return fail("--repo requires a value", .{});
            args.repo = argv[i];
        } else if (std.mem.eql(u8, arg, "--baseline-repo") or std.mem.eql(u8, arg, "--baseline-root")) {
            i += 1;
            if (i >= argv.len) return fail("{s} requires a value", .{arg});
            args.baseline_repo = argv[i];
        } else if (std.mem.eql(u8, arg, "--candidate-repo") or std.mem.eql(u8, arg, "--candidate-root")) {
            i += 1;
            if (i >= argv.len) return fail("{s} requires a value", .{arg});
            args.candidate_repo = argv[i];
        } else if (std.mem.eql(u8, arg, "--comparison-purpose")) {
            i += 1;
            if (i >= argv.len) return fail("--comparison-purpose requires a value", .{});
            if (!std.mem.eql(u8, argv[i], "candidate-evaluation") and
                !std.mem.eql(u8, argv[i], "noise-calibration"))
            {
                return fail("invalid --comparison-purpose '{s}'", .{argv[i]});
            }
            args.comparison_purpose = argv[i];
        } else if (std.mem.eql(u8, arg, "--output-dir") or std.mem.eql(u8, arg, "--work-dir")) {
            i += 1;
            if (i >= argv.len) return fail("{s} requires a value", .{arg});
            args.output_dir = argv[i];
        } else if (std.mem.eql(u8, arg, "--profile")) {
            i += 1;
            if (i >= argv.len) return fail("--profile requires a value", .{});
            if (!std.mem.eql(u8, argv[i], "smoke") and !std.mem.eql(u8, argv[i], "authoritative")) {
                return fail("invalid --profile '{s}'", .{argv[i]});
            }
            args.profile = argv[i];
        } else if (std.mem.eql(u8, arg, "--modes")) {
            i += 1;
            if (i >= argv.len) return fail("--modes requires a value", .{});
            args.modes_arg = argv[i];
            args.modes = try parseModes(allocator, argv[i]);
        } else if (std.mem.eql(u8, arg, "--thread-counts")) {
            i += 1;
            if (i >= argv.len) return fail("--thread-counts requires a value", .{});
            args.thread_counts = try parseThreadCounts(allocator, argv[i]);
        } else if (std.mem.eql(u8, arg, "--warmups")) {
            i += 1;
            if (i >= argv.len) return fail("--warmups requires a value", .{});
            args.warmups = std.fmt.parseInt(i64, argv[i], 10) catch return fail("invalid --warmups", .{});
        } else if (std.mem.eql(u8, arg, "--samples")) {
            i += 1;
            if (i >= argv.len) return fail("--samples requires a value", .{});
            args.samples = std.fmt.parseInt(i64, argv[i], 10) catch return fail("invalid --samples", .{});
        } else if (std.mem.eql(u8, arg, "--single-iterations")) {
            i += 1;
            if (i >= argv.len) return fail("--single-iterations requires a value", .{});
            args.single_iterations = try parsePositiveU64(argv[i], "--single-iterations");
        } else if (std.mem.eql(u8, arg, "--cancel-iterations")) {
            i += 1;
            if (i >= argv.len) return fail("--cancel-iterations requires a value", .{});
            args.cancel_iterations = try parsePositiveU64(argv[i], "--cancel-iterations");
        } else if (std.mem.eql(u8, arg, "--hot-iterations")) {
            i += 1;
            if (i >= argv.len) return fail("--hot-iterations requires a value", .{});
            args.hot_iterations = try parsePositiveU64(argv[i], "--hot-iterations");
        } else if (std.mem.eql(u8, arg, "--atomic-iterations")) {
            i += 1;
            if (i >= argv.len) return fail("--atomic-iterations requires a value", .{});
            args.atomic_iterations = try parsePositiveU64(argv[i], "--atomic-iterations");
        } else if (std.mem.eql(u8, arg, "--atomic-total-iterations")) {
            i += 1;
            if (i >= argv.len) return fail("--atomic-total-iterations requires a value", .{});
            args.atomic_total_iterations = try parsePositiveU64(argv[i], "--atomic-total-iterations");
        } else if (std.mem.eql(u8, arg, "--wait-iterations")) {
            i += 1;
            if (i >= argv.len) return fail("--wait-iterations requires a value", .{});
            args.wait_iterations = try parsePositiveU64(argv[i], "--wait-iterations");
        } else if (std.mem.eql(u8, arg, "--spawn-iterations")) {
            i += 1;
            if (i >= argv.len) return fail("--spawn-iterations requires a value", .{});
            args.spawn_iterations = try parsePositiveU64(argv[i], "--spawn-iterations");
        } else if (std.mem.eql(u8, arg, "--timeout")) {
            i += 1;
            if (i >= argv.len) return fail("--timeout requires a value", .{});
            args.timeout = std.fmt.parseFloat(f64, argv[i]) catch return fail("invalid --timeout", .{});
        } else if (std.mem.eql(u8, arg, "--min-interval-ms") or std.mem.eql(u8, arg, "--minimum-timed-interval-ms")) {
            i += 1;
            if (i >= argv.len) return fail("{s} requires a value", .{arg});
            args.min_interval_ms = std.fmt.parseFloat(f64, argv[i]) catch return fail("invalid {s}", .{arg});
        } else if (std.mem.eql(u8, arg, "--platform-id")) {
            i += 1;
            if (i >= argv.len) return fail("--platform-id requires a value", .{});
            args.platform_id = argv[i];
        } else if (std.mem.eql(u8, arg, "--runner-environment")) {
            i += 1;
            if (i >= argv.len) return fail("--runner-environment requires a value", .{});
            args.runner_environment = argv[i];
        } else if (std.mem.eql(u8, arg, "--host-pair-id")) {
            i += 1;
            if (i >= argv.len) return fail("--host-pair-id requires a value", .{});
            args.host_pair_id = argv[i];
        } else if (std.mem.eql(u8, arg, "--optimize")) {
            i += 1;
            if (i >= argv.len) return fail("--optimize requires a value", .{});
            args.optimize = argv[i];
        } else if (std.mem.eql(u8, arg, "--target")) {
            i += 1;
            if (i >= argv.len) return fail("--target requires a value", .{});
            args.target = argv[i];
        } else if (std.mem.eql(u8, arg, "--aot-target")) {
            i += 1;
            if (i >= argv.len) return fail("--aot-target requires a value", .{});
            if (!std.mem.eql(u8, argv[i], "x86_64") and !std.mem.eql(u8, argv[i], "aarch64")) {
                return fail("invalid --aot-target '{s}'", .{argv[i]});
            }
            args.aot_target = argv[i];
        } else if (std.mem.eql(u8, arg, "--runner")) {
            i += 1;
            if (i >= argv.len) return fail("--runner requires a value", .{});
            args.runner_prefix = argv[i];
        } else if (std.mem.eql(u8, arg, "--budget")) {
            i += 1;
            if (i >= argv.len) return fail("--budget requires a value", .{});
            args.budget = argv[i];
        } else if (std.mem.eql(u8, arg, "--no-budget")) {
            args.no_budget = true;
        } else if (std.mem.eql(u8, arg, "--rebuild")) {
            args.rebuild = true;
        } else if (std.mem.eql(u8, arg, "--trusted-calibration-preflight") or std.mem.eql(u8, arg, "--trusted-quality-preflight")) {
            args.trusted_calibration_preflight = true;
        } else if (std.mem.eql(u8, arg, "--validate-report")) {
            i += 1;
            if (i >= argv.len) return fail("--validate-report requires a value", .{});
            args.validate_report_path = argv[i];
        } else if (std.mem.eql(u8, arg, "--check-budget")) {
            i += 1;
            if (i >= argv.len) return fail("--check-budget requires a value", .{});
            args.check_budget_path = argv[i];
        } else {
            return fail("unknown argument '{s}'", .{arg});
        }
    }

    if (args.validate_report_path != null) return args;

    const defaults = spec.profileCounts(args.profile) orelse return fail("unknown profile '{s}'", .{args.profile});
    if (args.warmups == null) args.warmups = defaults.warmups;
    if (args.samples == null) args.samples = defaults.samples;
    if (args.warmups.? < 0 or args.samples.? <= 0) {
        return fail("--warmups must be >= 0 and --samples must be > 0", .{});
    }
    if (args.wait_iterations) |wi| {
        for (args.thread_counts) |tc| {
            if (@rem(wi, @as(u64, @intCast(tc))) != 0) {
                return fail("--wait-iterations must be divisible by every selected thread count", .{});
            }
        }
    }
    if (args.timeout <= 0.0) return fail("--timeout must be > 0", .{});
    if (args.min_interval_ms < spec.MIN_TIMED_INTERVAL_MS) {
        return fail("--min-interval-ms must be >= {d}", .{spec.MIN_TIMED_INTERVAL_MS});
    }
    if (args.budget != null and args.no_budget) {
        return fail("--budget and --no-budget are mutually exclusive", .{});
    }
    if ((args.baseline_repo == null) != (args.candidate_repo == null)) {
        return fail("--baseline-repo and --candidate-repo must be supplied together", .{});
    }
    const paired = args.baseline_repo != null;
    if (paired and args.comparison_purpose == null) {
        return fail("--comparison-purpose is required for paired revisions", .{});
    }
    if (!paired and args.comparison_purpose != null) {
        return fail("--comparison-purpose requires paired revisions", .{});
    }
    if (paired and @rem(args.samples.?, 4) != 0) {
        return fail("paired revision measurements require --samples divisible by 4", .{});
    }
    if (args.comparison_purpose != null and std.mem.eql(u8, args.comparison_purpose.?, "noise-calibration") and !args.no_budget) {
        return fail("noise calibration requires --no-budget", .{});
    }
    if (args.budget != null and (args.comparison_purpose == null or !std.mem.eql(u8, args.comparison_purpose.?, "candidate-evaluation"))) {
        return fail("budget enforcement requires paired candidate evaluation", .{});
    }
    if (args.host_pair_id) |hpi| {
        if (std.mem.trim(u8, hpi, " \t\r\n").len == 0) return fail("--host-pair-id must not be empty", .{});
    }
    if (args.runner_environment) |re| {
        if (std.mem.trim(u8, re, " \t\r\n").len == 0) return fail("--runner-environment must not be empty", .{});
    }
    if (args.trusted_calibration_preflight and (!paired or !std.mem.eql(u8, args.comparison_purpose.?, "noise-calibration"))) {
        return fail("--trusted-calibration-preflight requires paired noise calibration", .{});
    }
    return args;
}

fn executionArch(args: Args) []const u8 {
    if (args.aot_target) |at| return at;
    if (args.target) |t| {
        if (std.mem.startsWith(u8, t, "aarch64")) return "aarch64";
    }
    return if (builtin.cpu.arch == .aarch64) "aarch64" else "x86_64";
}

fn splitRunnerPrefix(allocator: Allocator, raw: []const u8) ![]const []const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeAny(u8, raw, " \t\r\n");
    while (it.next()) |tok| {
        try list.append(allocator, tok);
    }
    return list.toOwnedSlice(allocator);
}

fn resolvePath(allocator: Allocator, base: []const u8, p: []const u8) ![]const u8 {
    if (std.fs.path.isAbsolute(p)) return allocator.dupe(u8, p);
    return std.fs.path.join(allocator, &.{ base, p });
}

fn resolveRepoRoot(allocator: Allocator, io: std.Io, environ: std.process.Environ) ![]const u8 {
    if (environ.getAlloc(allocator, "WAMR_REPO_ROOT") catch null) |r| {
        if (r.len > 0) return r;
    }
    return std.process.currentPathAlloc(io, allocator);
}

const RevisionContext = struct {
    repo: []const u8,
    builds: std.StringHashMapUnmanaged(runner.Build),
    builds_list: []const runner.Build,
    aot_artifacts: ObjectMap,
    aot_artifacts_metadata: ObjectMap,
    single_wasm: []const u8,
    threaded_wasm: []const u8,
};

pub fn execute(allocator: Allocator, io: std.Io, environ: std.process.Environ, args: Args) HarnessError!u8 {
    if (args.validate_report_path) |vpath| {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, vpath, allocator, .limited(64 * 1024 * 1024)) catch {
            return fail("cannot read report {s}", .{vpath});
        };
        const parsed = std.json.parseFromSlice(Value, allocator, bytes, .{
            .duplicate_field_behavior = .@"error",
        }) catch return fail("invalid JSON in {s}", .{vpath});
        try report.validateReport(allocator, parsed.value);
        if (args.check_budget_path) |bpath| {
            const plat_budget = try report.loadBudget(allocator, io, bpath, parsed.value);
            const failures = try report.evaluateBudget(allocator, plat_budget, parsed.value);
            if (failures.len > 0) {
                for (failures) |f| std.debug.print("budget failure: {s}\n", .{f});
                return 1;
            }
        }
        return 0;
    }

    if (args.check_budget_path != null) {
        return fail("--check-budget requires --validate-report", .{});
    }

    const cwd = resolveRepoRoot(allocator, io, environ) catch return fail("cannot resolve repo root", .{});
    const warmups = args.warmups.?;
    const samples = args.samples.?;
    const minimum_interval_ns: i64 = @intFromFloat(args.min_interval_ms * 1_000_000.0);

    var revision_mode: []const u8 = "";
    var comparison_purpose: []const u8 = "";
    var revision_roles: []const []const u8 = &.{};
    var revision_repos: std.StringHashMapUnmanaged([]const u8) = .empty;
    var candidate_repo: []const u8 = "";

    if (args.baseline_repo == null) {
        candidate_repo = resolvePath(allocator, cwd, args.repo orelse cwd) catch return error.OutOfMemory;
        revision_mode = "single-revision-compatibility";
        comparison_purpose = "single-revision-compatibility";
        revision_roles = &spec.SINGLE_REVISION_ROLES;
        try revision_repos.put(allocator, "candidate", candidate_repo);
    } else {
        const baseline_repo = resolvePath(allocator, cwd, args.baseline_repo.?) catch return error.OutOfMemory;
        candidate_repo = resolvePath(allocator, cwd, args.candidate_repo.?) catch return error.OutOfMemory;
        if (std.mem.eql(u8, baseline_repo, candidate_repo)) {
            return fail("paired revisions must use distinct independently built checkout paths", .{});
        }
        revision_mode = "paired-revisions";
        comparison_purpose = args.comparison_purpose.?;
        revision_roles = &spec.REVISION_ROLES;
        try revision_repos.put(allocator, "baseline", baseline_repo);
        try revision_repos.put(allocator, "candidate", candidate_repo);
    }

    const output = resolvePath(allocator, candidate_repo, args.output_dir) catch return error.OutOfMemory;
    std.Io.Dir.cwd().createDirPath(io, output) catch return fail("cannot create output directory {s}", .{output});
    for ([_][]const u8{ "report.json", "report.md", "failure-diagnostic.json", "failure-diagnostic.md" }) |stale| {
        const stale_path = std.fs.path.join(allocator, &.{ output, stale }) catch return error.OutOfMemory;
        std.Io.Dir.cwd().deleteFile(io, stale_path) catch {};
    }

    var sources: std.StringHashMapUnmanaged(ObjectMap) = .empty;
    for (revision_roles) |role| {
        const r_repo = revision_repos.get(role).?;
        try sources.put(allocator, role, try runner.sourceIdentity(allocator, io, r_repo));
    }
    if (std.mem.eql(u8, comparison_purpose, "noise-calibration")) {
        const b_src = sources.get("baseline").?;
        const c_src = sources.get("candidate").?;
        for ([_][]const u8{ "commit", "tracked_diff_sha256", "build_source_sha256" }) |k| {
            if (!std.mem.eql(u8, b_src.get(k).?.string, c_src.get(k).?.string)) {
                return fail("noise calibration requires identical revision identities from distinct checkout paths", .{});
            }
        }
    }

    const resolved_fixtures = try runner.resolveMeasurementFixtures(allocator, io, revision_repos);
    const fixture_repo = resolved_fixtures.fixture_repo;
    const fixture_report = resolved_fixtures.fixtures;
    const fixture_set_sha256 = resolved_fixtures.fixture_set_sha256;

    const overrides: sizing.IterationOverrides = .{
        .single_iterations = if (args.single_iterations) |v| @intCast(v) else null,
        .cancel_iterations = if (args.cancel_iterations) |v| @intCast(v) else null,
        .hot_iterations = if (args.hot_iterations) |v| @intCast(v) else null,
        .atomic_iterations = if (args.atomic_iterations) |v| @intCast(v) else null,
        .atomic_total_iterations = if (args.atomic_total_iterations) |v| @intCast(v) else null,
        .wait_iterations = if (args.wait_iterations) |v| @intCast(v) else null,
        .spawn_iterations = if (args.spawn_iterations) |v| @intCast(v) else null,
    };
    const pilot_iteration_plan = try sizing.resolvedIterationPlan(allocator, args.modes, args.thread_counts, overrides);
    try sizing.validateIterationPlanRanges(pilot_iteration_plan, args.thread_counts);

    var has_aot = false;
    for (args.modes) |m| if (std.mem.eql(u8, m, "aot")) {
        has_aot = true;
    };
    if (args.trusted_calibration_preflight and !has_aot) {
        return fail("trusted calibration preflight requires the AOT runtime path", .{});
    }

    const pair_plan = try sizing.plannedPairSpecs(allocator, args.modes, args.thread_counts);
    const pilot_order = try sizing.pilotOrderForPlan(allocator, pair_plan, revision_roles, pilot_iteration_plan);
    const runner_cmd = splitRunnerPrefix(allocator, args.runner_prefix) catch return error.OutOfMemory;
    const cpu_placement = try spec.discoverCpuPlacement(allocator, io, args.thread_counts);
    try spec.validateCpuPlacement(allocator, .{ .object = cpu_placement }, args.thread_counts);

    const revision_artifact_strategy = if (std.mem.eql(u8, comparison_purpose, "noise-calibration"))
        "shared-exact-build"
    else if (std.mem.eql(u8, revision_mode, "paired-revisions"))
        "independent-builds"
    else
        "single-revision-build";
    const preflight_acceptance_rule = "every probe must have timed_interval_ns >= minimum_timed_interval_ns and 99 * timing_overhead_ns < minimum_timed_interval_ns";

    const host = try benchmark_schema.hostMetadata(allocator, io, environ, args.runner_environment);
    const default_platform_id = try std.fmt.allocPrint(
        allocator,
        "local-{s}-{s}",
        .{
            try std.ascii.allocLowerString(allocator, host.get("system").?.string),
            try std.ascii.allocLowerString(allocator, host.get("machine").?.string),
        },
    );
    const platform_id = args.platform_id orelse default_platform_id;
    const host_pair = try runner.hostPairIdentity(allocator, platform_id, host, args.host_pair_id);
    const host_quiescence_at_start = try runner.hostQuiescenceDiagnostics(allocator, io, environ);
    const exec_arch = executionArch(args);

    var contexts: std.StringHashMapUnmanaged(RevisionContext) = .empty;
    for (revision_roles) |role| {
        if (std.mem.eql(u8, revision_artifact_strategy, "shared-exact-build") and std.mem.eql(u8, role, "candidate")) {
            try contexts.put(allocator, role, contexts.get("baseline").?);
            continue;
        }
        const r_repo = revision_repos.get(role).?;
        const r_out = std.fs.path.join(allocator, &.{ output, "revisions", role }) catch return error.OutOfMemory;
        var builds: std.StringHashMapUnmanaged(runner.Build) = .empty;
        var builds_list: std.ArrayList(runner.Build) = .empty;
        for (args.modes) |mode| {
            const b = try runner.buildVariant(allocator, io, environ, .{
                .repo = r_repo,
                .root = r_out,
                .mode = mode,
                .threads_enabled = true,
                .optimize = args.optimize,
                .target = args.target,
                .source = sources.get(role).?,
                .rebuild = args.rebuild,
                .compiler_toggle = std.mem.eql(u8, mode, "aot"),
            });
            try builds.put(allocator, b.name, b);
            try builds_list.append(allocator, b);
        }
        var aot_artifacts: ObjectMap = .empty;
        var aot_artifacts_metadata: ObjectMap = .empty;
        if (has_aot) {
            const compiler = if (args.target != null) blk: {
                const hc = try runner.buildVariant(allocator, io, environ, .{
                    .repo = r_repo,
                    .root = r_out,
                    .mode = "aot",
                    .threads_enabled = true,
                    .optimize = args.optimize,
                    .target = null,
                    .source = sources.get(role).?,
                    .rebuild = args.rebuild,
                    .compiler_toggle = true,
                });
                try builds.put(allocator, "host-compiler", hc);
                try builds_list.append(allocator, hc);
                break :blk hc;
            } else builds.get("enabled-aot").?;
            aot_artifacts = try runner.compileAotFixtures(allocator, io, environ, fixture_repo, r_out, compiler, exec_arch);
            aot_artifacts_metadata = try runner.aotArtifactReport(allocator, io, aot_artifacts, exec_arch);
        }
        try contexts.put(allocator, role, .{
            .repo = r_repo,
            .builds = builds,
            .builds_list = try builds_list.toOwnedSlice(allocator),
            .aot_artifacts = aot_artifacts,
            .aot_artifacts_metadata = aot_artifacts_metadata,
            .single_wasm = std.fs.path.join(allocator, &.{ fixture_repo, spec.FIXTURES[0].path }) catch return error.OutOfMemory,
            .threaded_wasm = std.fs.path.join(allocator, &.{ fixture_repo, spec.FIXTURES[1].path }) catch return error.OutOfMemory,
        });
    }

    var quality_preflight: ObjectMap = .empty;
    try quality_preflight.put(allocator, "enabled", .{ .bool = false });
    try quality_preflight.put(allocator, "status", .{ .string = "not-requested" });
    try quality_preflight.put(allocator, "probe_count", .{ .integer = 0 });
    try quality_preflight.put(allocator, "probes_per_thread", .{ .integer = spec.TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD });
    try quality_preflight.put(allocator, "mode", .{ .string = "aot" });
    try quality_preflight.put(allocator, "workload", .{ .string = "hot" });
    try quality_preflight.put(allocator, "clock_id", .{ .string = spec.WASI_MONOTONIC_CLOCK_ID });
    try quality_preflight.put(allocator, "iteration_multiplier", .{ .string = spec.TRUSTED_BARRIER_PREFLIGHT_ITERATION_MULTIPLIER });
    var tc_arr_qp = Array.init(allocator);
    for (args.thread_counts) |tc| try tc_arr_qp.append(.{ .integer = tc });
    try quality_preflight.put(allocator, "thread_counts", .{ .array = tc_arr_qp });
    try quality_preflight.put(allocator, "minimum_timed_interval_ns", .{ .integer = minimum_interval_ns });
    try quality_preflight.put(allocator, "timing_overhead_ratio_limit", .{ .float = spec.TIMING_OVERHEAD_RATIO_LIMIT });
    try quality_preflight.put(allocator, "maximum_accepted_barrier_ns", .{ .integer = try runner.maximumPreflightBarrierNs(minimum_interval_ns) });
    try quality_preflight.put(allocator, "acceptance_rule", .{ .string = preflight_acceptance_rule });
    try quality_preflight.put(allocator, "summary", .null);
    try quality_preflight.put(allocator, "samples", .{ .array = Array.init(allocator) });
    try quality_preflight.put(allocator, "host_quiescence_at_start", .{ .object = host_quiescence_at_start });

    var pilot_records: std.ArrayList(ObjectMap) = .empty;
    _ = try sizing.pilotProgressBound(allocator, pilot_records.items, pilot_order, warmups, samples);

    for (pilot_order) |p_spec| {
        const rev_role = p_spec.get("revision").?.string;
        const context = contexts.get(rev_role).?;
        const mode = p_spec.get("mode").?.string;
        const sizing_wl = p_spec.get("workload").?.string;
        const guest_wl = if (std.mem.eql(u8, sizing_wl, "cancel-hot")) "hot" else sizing_wl;
        const build_key = try std.fmt.allocPrint(allocator, "enabled-{s}", .{mode});

        var selected: runner.Build = undefined;
        var module_path: []const u8 = "";
        var cancel_points: []const u8 = "";
        var static_sites: ?i64 = null;
        var runtime_args_buf: [1][]const u8 = undefined;
        var runtime_args: []const []const u8 = &.{};

        if (std.mem.eql(u8, sizing_wl, "single-hot")) {
            selected = context.builds.get(build_key).?;
            module_path = if (std.mem.eql(u8, mode, "interpreter")) context.single_wasm else context.aot_artifacts.get("single").?.string;
            cancel_points = "not-applicable";
            static_sites = 0;
            runtime_args_buf[0] = if (std.mem.eql(u8, p_spec.get("condition").?.string, "threads-disabled"))
                "--benchmark-wasi-thread-manager=disabled"
            else
                "--benchmark-wasi-thread-manager=enabled";
            runtime_args = runtime_args_buf[0..1];
        } else if (std.mem.eql(u8, sizing_wl, "cancel-hot")) {
            selected = context.builds.get("enabled-aot").?;
            cancel_points = if (std.mem.eql(u8, p_spec.get("condition").?.string, "cancel-points-off")) "off" else "on";
            const art_key = try std.fmt.allocPrint(allocator, "threaded-polls-{s}", .{cancel_points});
            module_path = context.aot_artifacts.get(art_key).?.string;
            static_sites = if (std.mem.eql(u8, cancel_points, "on"))
                context.aot_artifacts_metadata.get("cancel_poll_static").?.object.get("sites_enabled").?.integer
            else
                0;
        } else {
            selected = context.builds.get(build_key).?;
            module_path = if (std.mem.eql(u8, mode, "interpreter")) context.threaded_wasm else context.aot_artifacts.get("threaded-polls-on").?.string;
            cancel_points = if (std.mem.eql(u8, mode, "aot")) "on" else "interpreter-dispatch";
            static_sites = if (std.mem.eql(u8, mode, "aot"))
                context.aot_artifacts_metadata.get("cancel_poll_static").?.object.get("sites_enabled").?.integer
            else
                null;
        }

        var rec_fields = try spec.cloneObject(allocator, p_spec);
        try rec_fields.put(allocator, "phase", .{ .string = "pilot" });
        try rec_fields.put(allocator, "threads_enabled", .{ .bool = selected.threads_enabled });
        try rec_fields.put(allocator, "cancel_points", .{ .string = cancel_points });
        try rec_fields.put(allocator, "static_cancel_poll_sites", if (static_sites) |ss| .{ .integer = ss } else .null);

        const pilot = try runner.measureOnce(allocator, io, .{
            .repo = context.repo,
            .runner = runner_cmd,
            .cpu_placement = cpu_placement,
            .wamr_path = selected.wamr,
            .module_path = module_path,
            .workload = guest_wl,
            .threads = p_spec.get("threads").?.integer,
            .iterations = spec.valueAsU64(p_spec.get("iterations").?).?,
            .timeout = args.timeout,
            .min_interval_ns = spec.PILOT_CLOCK_RESOLUTION_MINIMUM_NS,
            .enforce_timing_quality = false,
            .runtime_args = runtime_args,
            .record_fields = rec_fields,
        });
        try pilot_records.append(allocator, pilot);
        try sizing.validateSizingPilot(allocator, pilot, p_spec);
        _ = try sizing.pilotProgressBound(allocator, pilot_records.items, pilot_order, warmups, samples);
    }

    const sizing_res = try sizing.resolveOneShotSizing(
        allocator,
        pilot_records.items,
        pilot_order,
        args.modes,
        args.thread_counts,
        warmups,
        samples,
        args.timeout,
    );
    const iteration_plan = sizing_res.plan;
    const sizing_resolution = sizing_res.resolution;

    var plan: ObjectMap = .empty;
    try plan.put(allocator, "profile", .{ .string = args.profile });
    try plan.put(allocator, "warmups", .{ .integer = warmups });
    try plan.put(allocator, "samples", .{ .integer = samples });
    try plan.put(allocator, "revision_mode", .{ .string = revision_mode });
    try plan.put(allocator, "comparison_purpose", .{ .string = comparison_purpose });
    var roles_arr = Array.init(allocator);
    for (revision_roles) |r| try roles_arr.append(.{ .string = r });
    try plan.put(allocator, "revision_roles", .{ .array = roles_arr });
    var modes_arr = Array.init(allocator);
    for (args.modes) |m| try modes_arr.append(.{ .string = m });
    try plan.put(allocator, "modes", .{ .array = modes_arr });
    var tc_arr = Array.init(allocator);
    for (args.thread_counts) |tc| try tc_arr.append(.{ .integer = tc });
    try plan.put(allocator, "thread_counts", .{ .array = tc_arr });
    try plan.put(allocator, "iterations", .{ .object = iteration_plan });

    var sizing_plan_obj: ObjectMap = .empty;
    try sizing_plan_obj.put(allocator, "algorithm", try sizing.sizingAlgorithmSpec(allocator, args.timeout));
    try sizing_plan_obj.put(allocator, "pilot_iterations", .{ .object = pilot_iteration_plan });
    var po_arr = Array.init(allocator);
    for (pilot_order) |po| try po_arr.append(.{ .object = po });
    try sizing_plan_obj.put(allocator, "pilot_order", .{ .array = po_arr });
    try sizing_plan_obj.put(allocator, "resolved", .{ .object = sizing_resolution });
    try plan.put(allocator, "sizing", .{ .object = sizing_plan_obj });
    try plan.put(allocator, "timeout_seconds", .{ .float = args.timeout });
    try plan.put(allocator, "minimum_timed_interval_ns", .{ .integer = minimum_interval_ns });
    try plan.put(allocator, "atomic_wait_preflight_runs", .{ .integer = spec.atomicWaitPreflightRuns(args.profile).? });
    try plan.put(allocator, "atomic_wait_preflight_iterations", .{ .integer = @intCast(spec.ATOMIC_WAIT_PREFLIGHT_ITERATIONS) });
    try plan.put(allocator, "atomic_wait_preflight_timing_quality", .{ .string = "correctness-only" });

    var sbp: ObjectMap = .empty;
    try sbp.put(allocator, "enabled", .{ .bool = args.trusted_calibration_preflight });
    try sbp.put(allocator, "mode", .{ .string = "aot" });
    try sbp.put(allocator, "workload", .{ .string = "hot" });
    try sbp.put(allocator, "clock_id", .{ .string = spec.WASI_MONOTONIC_CLOCK_ID });
    try sbp.put(allocator, "iteration_multiplier", .{ .string = spec.TRUSTED_BARRIER_PREFLIGHT_ITERATION_MULTIPLIER });
    try sbp.put(allocator, "probes_per_thread", .{ .integer = spec.TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD });
    try sbp.put(allocator, "probe_count", .{
        .integer = if (args.trusted_calibration_preflight)
            spec.TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD * @as(i64, @intCast(args.thread_counts.len))
        else
            0,
    });
    try sbp.put(allocator, "timing_overhead_ratio_limit", .{ .float = spec.TIMING_OVERHEAD_RATIO_LIMIT });
    try sbp.put(allocator, "target_barrier_ns", .{ .integer = spec.TARGET_BARRIER_NS });
    try sbp.put(allocator, "target_required_interval_ns", .{ .integer = spec.TARGET_BARRIER_REQUIRED_INTERVAL_NS });
    try sbp.put(allocator, "minimum_interval_headroom_ns", .{ .integer = minimum_interval_ns - spec.TARGET_BARRIER_REQUIRED_INTERVAL_NS });
    try sbp.put(allocator, "maximum_accepted_barrier_ns", .{ .integer = try runner.maximumPreflightBarrierNs(minimum_interval_ns) });
    try sbp.put(allocator, "acceptance_rule", .{ .string = preflight_acceptance_rule });
    try plan.put(allocator, "scheduler_barrier_preflight", .{ .object = sbp });
    try plan.put(allocator, "cpu_placement", try spec.cpuPlacementPolicyValue(allocator));
    try plan.put(allocator, "pair_execution", try spec.pairExecutionPolicyValue(allocator));
    try plan.put(allocator, "revision_artifact_policy", .{ .string = spec.REVISION_ARTIFACT_POLICY });
    try plan.put(allocator, "fixture_source_policy", .{ .string = spec.FIXTURE_SOURCE_POLICY });
    try plan.put(allocator, "guest_clock_policy", try spec.guestClockPolicyValue(allocator));
    try plan.put(allocator, "optimize", .{ .string = args.optimize });
    var pairs_arr = Array.init(allocator);
    for (pair_plan) |pp| try pairs_arr.append(.{ .object = pp });
    try plan.put(allocator, "pairs", .{ .array = pairs_arr });

    const plan_sha256 = try spec.cacheKey(allocator, .{ .object = plan });
    const measurement_plan_identity = try spec.measurementPlanSha256(allocator, .{ .object = plan });
    const checksum_preparation = try sizing.prepareExpectedResults(allocator, io, iteration_plan, args.modes, args.thread_counts);

    var revisions: ObjectMap = .empty;
    var revision_fields: std.StringHashMapUnmanaged(ObjectMap) = .empty;
    for (revision_roles) |role| {
        var r_obj = try spec.cloneObject(allocator, sources.get(role).?);
        try r_obj.put(allocator, "fixture_set_sha256", .{ .string = fixture_set_sha256 });
        try r_obj.put(allocator, "plan_sha256", .{ .string = plan_sha256 });
        try r_obj.put(allocator, "host_pair_id", host_pair.get("id").?);
        try r_obj.put(allocator, "host_fingerprint_sha256", host_pair.get("host_fingerprint_sha256").?);
        try revisions.put(allocator, role, .{ .object = r_obj });

        var rf: ObjectMap = .empty;
        try rf.put(allocator, "revision_commit", r_obj.get("commit").?);
        try rf.put(allocator, "revision_build_source_sha256", r_obj.get("build_source_sha256").?);
        try rf.put(allocator, "fixture_set_sha256", .{ .string = fixture_set_sha256 });
        try rf.put(allocator, "plan_sha256", .{ .string = plan_sha256 });
        try rf.put(allocator, "host_pair_id", host_pair.get("id").?);
        try rf.put(allocator, "host_fingerprint_sha256", host_pair.get("host_fingerprint_sha256").?);
        try revision_fields.put(allocator, role, rf);
    }

    if (args.trusted_calibration_preflight) {
        const cand_ctx = contexts.get("candidate").?;
        quality_preflight = try runner.runTrustedBarrierPreflight(allocator, io, .{
            .repo = cand_ctx.repo,
            .runner = runner_cmd,
            .cpu_placement = cpu_placement,
            .wamr_path = cand_ctx.builds.get("enabled-aot").?.wamr,
            .module_path = cand_ctx.aot_artifacts.get("threaded-polls-on").?.string,
            .thread_counts = args.thread_counts,
            .iterations_by_thread = iteration_plan.get("aot").?.object.get("hot").?.object,
            .timeout = args.timeout,
            .minimum_interval_ns = minimum_interval_ns,
            .static_cancel_poll_sites = cand_ctx.aot_artifacts_metadata.get("cancel_poll_static").?.object.get("sites_enabled").?.integer,
        });
        try quality_preflight.put(allocator, "host_quiescence_at_start", .{ .object = host_quiescence_at_start });
        if (!std.mem.eql(u8, quality_preflight.get("status").?.string, "passed")) {
            return fail("trusted scheduler/barrier preflight failed", .{});
        }
    }

    if (has_aot) {
        for (revision_roles) |role| {
            const ctx = contexts.get(role).?;
            const runs = spec.atomicWaitPreflightRuns(args.profile).?;
            var idx: i64 = 0;
            while (idx < runs) : (idx += 1) {
                var pf_fields = try spec.cloneObject(allocator, revision_fields.get(role).?);
                try pf_fields.put(allocator, "revision", .{ .string = role });
                try pf_fields.put(allocator, "pair_kind", .{ .string = "atomic-wait-preflight" });
                try pf_fields.put(allocator, "pair_key", .{ .string = "atomic-wait-preflight" });
                try pf_fields.put(allocator, "pair_index", .{ .integer = idx });
                try pf_fields.put(allocator, "phase", .{ .string = "preflight" });
                try pf_fields.put(allocator, "order", .{ .integer = 0 });
                try pf_fields.put(allocator, "condition", .{ .string = "aot" });
                try pf_fields.put(allocator, "pair_left", .{ .string = "aot" });
                try pf_fields.put(allocator, "pair_right", .{ .string = "aot" });
                try pf_fields.put(allocator, "mode", .{ .string = "aot" });
                try pf_fields.put(allocator, "threads_enabled", .{ .bool = true });
                try pf_fields.put(allocator, "cancel_points", .{ .string = "on" });
                try pf_fields.put(allocator, "static_cancel_poll_sites", ctx.aot_artifacts_metadata.get("cancel_poll_static").?.object.get("sites_enabled").?);
                try pf_fields.put(allocator, "workload", .{ .string = "atomic" });
                try pf_fields.put(allocator, "threads", .{ .integer = 1 });
                try pf_fields.put(allocator, "iterations", .{ .integer = @intCast(spec.ATOMIC_WAIT_PREFLIGHT_ITERATIONS) });

                _ = try runner.measureOnce(allocator, io, .{
                    .repo = ctx.repo,
                    .runner = runner_cmd,
                    .cpu_placement = cpu_placement,
                    .wamr_path = ctx.builds.get("enabled-aot").?.wamr,
                    .module_path = ctx.aot_artifacts.get("threaded-polls-on").?.string,
                    .workload = "atomic",
                    .threads = 1,
                    .iterations = @intCast(spec.ATOMIC_WAIT_PREFLIGHT_ITERATIONS),
                    .timeout = args.timeout,
                    .min_interval_ns = 1,
                    .enforce_timing_quality = false,
                    .record_fields = pf_fields,
                });
            }
        }
    }

    var records: std.ArrayList(ObjectMap) = .empty;
    const phases = [_]struct { name: []const u8, count: i64 }{
        .{ .name = "warmup", .count = warmups },
        .{ .name = "measure", .count = samples },
    };

    for (args.modes) |mode| {
        const pair_key = try std.fmt.allocPrint(allocator, "single-infrastructure/{s}", .{mode});
        const left = "threads-disabled";
        const right = "threads-enabled";
        for (phases) |ph| {
            var p_idx: i64 = 0;
            while (p_idx < ph.count) : (p_idx += 1) {
                const global_idx: usize = @intCast(if (std.mem.eql(u8, ph.name, "warmup")) p_idx else warmups + p_idx);
                const rev_order = if (revision_roles.len == 2)
                    benchmark_schema.alternatingPairOrder(global_idx, "baseline", "candidate")
                else
                    [2][]const u8{ "candidate", "" };
                const cond_order = spec.pairedConditionOrder(global_idx, "single-infrastructure", left, right);
                const build_key = try std.fmt.allocPrint(allocator, "enabled-{s}", .{mode});
                const sh_iters = try sizing.iterationCount(iteration_plan, mode, "single-hot", 1);

                var r_i: usize = 0;
                while (r_i < revision_roles.len) : (r_i += 1) {
                    const rev = rev_order[r_i];
                    const ctx = contexts.get(rev).?;
                    const mod_path = if (std.mem.eql(u8, mode, "interpreter")) ctx.single_wasm else ctx.aot_artifacts.get("single").?.string;

                    var invs: [2]runner.MeasureOnceArgs = undefined;
                    var rt_args_storage: [2][1][]const u8 = undefined;
                    for (cond_order, 0..) |cond, c_i| {
                        const mgr_enabled = std.mem.eql(u8, cond, "threads-enabled");
                        rt_args_storage[c_i][0] = if (mgr_enabled) "--benchmark-wasi-thread-manager=enabled" else "--benchmark-wasi-thread-manager=disabled";
                        var fields = try spec.cloneObject(allocator, revision_fields.get(rev).?);
                        try fields.put(allocator, "revision", .{ .string = rev });
                        try fields.put(allocator, "revision_order", .{ .integer = @intCast(r_i) });
                        try fields.put(allocator, "pair_kind", .{ .string = "single-infrastructure" });
                        try fields.put(allocator, "pair_execution", .{ .string = "concurrent-topology-paired-cpus" });
                        try fields.put(allocator, "pair_key", .{ .string = pair_key });
                        try fields.put(allocator, "pair_index", .{ .integer = p_idx });
                        try fields.put(allocator, "phase", .{ .string = ph.name });
                        try fields.put(allocator, "order", .{ .integer = @intCast(c_i) });
                        try fields.put(allocator, "condition", .{ .string = cond });
                        try fields.put(allocator, "pair_left", .{ .string = left });
                        try fields.put(allocator, "pair_right", .{ .string = right });
                        try fields.put(allocator, "mode", .{ .string = mode });
                        try fields.put(allocator, "threads_enabled", .{ .bool = true });
                        try fields.put(allocator, "thread_manager_enabled", .{ .bool = mgr_enabled });
                        try fields.put(allocator, "cancel_points", .{ .string = "not-applicable" });
                        try fields.put(allocator, "static_cancel_poll_sites", .{ .integer = 0 });
                        try fields.put(allocator, "workload", .{ .string = "single-hot" });
                        try fields.put(allocator, "threads", .{ .integer = 1 });
                        try fields.put(allocator, "iterations", .{ .integer = @intCast(sh_iters) });

                        invs[c_i] = .{
                            .repo = ctx.repo,
                            .runner = runner_cmd,
                            .cpu_placement = cpu_placement,
                            .wamr_path = ctx.builds.get(build_key).?.wamr,
                            .module_path = mod_path,
                            .workload = "single-hot",
                            .threads = 1,
                            .iterations = sh_iters,
                            .timeout = args.timeout,
                            .min_interval_ns = minimum_interval_ns,
                            .cpu_affinity_override = try spec.directPairCpuAffinity(allocator, cpu_placement, global_idx, cond, left, right),
                            .runtime_args = rt_args_storage[c_i][0..1],
                            .record_fields = fields,
                        };
                    }
                    const pair_res = try runner.measureConcurrently(allocator, io, &invs);
                    try records.append(allocator, pair_res[0]);
                    try records.append(allocator, pair_res[1]);
                }
            }
        }
    }

    const scenarios = try sizing.plannedScenarios(allocator, args.thread_counts);
    for (scenarios) |scenario| {
        const pair_kind = if (args.modes.len == 2) "runtime-parity" else "repeatability";
        const pair_key = if (args.modes.len == 2)
            try std.fmt.allocPrint(allocator, "runtime/{s}", .{scenario.key})
        else
            try std.fmt.allocPrint(allocator, "runtime/{s}/{s}", .{ scenario.key, args.modes[0] });
        const left = if (args.modes.len == 2) "interpreter" else try std.fmt.allocPrint(allocator, "{s}-a", .{args.modes[0]});
        const right = if (args.modes.len == 2) "aot" else try std.fmt.allocPrint(allocator, "{s}-b", .{args.modes[0]});

        for (phases) |ph| {
            var p_idx: i64 = 0;
            while (p_idx < ph.count) : (p_idx += 1) {
                const global_idx: usize = @intCast(if (std.mem.eql(u8, ph.name, "warmup")) p_idx else warmups + p_idx);
                const rev_order = if (revision_roles.len == 2)
                    benchmark_schema.alternatingPairOrder(global_idx, "baseline", "candidate")
                else
                    [2][]const u8{ "candidate", "" };
                const cond_order = spec.pairedConditionOrder(global_idx, pair_kind, left, right);
                const inv_order = try spec.pairedInvocationOrder(allocator, global_idx, pair_kind, left, right, revision_roles);

                for (inv_order) |inv| {
                    const rev = inv.revision;
                    const cond = inv.condition;
                    const mode = if (args.modes.len == 2) cond else args.modes[0];
                    const ctx = contexts.get(rev).?;
                    const build_key = try std.fmt.allocPrint(allocator, "enabled-{s}", .{mode});
                    const mod_path = if (std.mem.eql(u8, mode, "interpreter"))
                        ctx.threaded_wasm
                    else
                        ctx.aot_artifacts.get("threaded-polls-on").?.string;
                    const iters = try sizing.iterationCount(iteration_plan, mode, scenario.workload, scenario.threads);
                    const r_ord: i64 = if (std.mem.eql(u8, rev_order[0], rev)) 0 else 1;
                    const c_ord: i64 = if (std.mem.eql(u8, cond_order[0], cond)) 0 else 1;

                    var fields = try spec.cloneObject(allocator, revision_fields.get(rev).?);
                    try fields.put(allocator, "revision", .{ .string = rev });
                    try fields.put(allocator, "revision_order", .{ .integer = r_ord });
                    try fields.put(allocator, "pair_kind", .{ .string = pair_kind });
                    try fields.put(allocator, "pair_execution", .{ .string = "sequential-position-balanced" });
                    try fields.put(allocator, "pair_key", .{ .string = pair_key });
                    try fields.put(allocator, "pair_index", .{ .integer = p_idx });
                    try fields.put(allocator, "phase", .{ .string = ph.name });
                    try fields.put(allocator, "order", .{ .integer = c_ord });
                    try fields.put(allocator, "condition", .{ .string = cond });
                    try fields.put(allocator, "pair_left", .{ .string = left });
                    try fields.put(allocator, "pair_right", .{ .string = right });
                    try fields.put(allocator, "mode", .{ .string = mode });
                    try fields.put(allocator, "threads_enabled", .{ .bool = true });
                    try fields.put(allocator, "thread_manager_enabled", .{ .bool = true });
                    try fields.put(allocator, "cancel_points", .{ .string = if (std.mem.eql(u8, mode, "aot")) "on" else "interpreter-dispatch" });
                    try fields.put(allocator, "static_cancel_poll_sites", if (std.mem.eql(u8, mode, "aot"))
                        ctx.aot_artifacts_metadata.get("cancel_poll_static").?.object.get("sites_enabled").?
                    else
                        .null);
                    try fields.put(allocator, "workload", .{ .string = scenario.workload });
                    try fields.put(allocator, "threads", .{ .integer = scenario.threads });
                    try fields.put(allocator, "iterations", .{ .integer = @intCast(iters) });

                    const rec = try runner.measureOnce(allocator, io, .{
                        .repo = ctx.repo,
                        .runner = runner_cmd,
                        .cpu_placement = cpu_placement,
                        .wamr_path = ctx.builds.get(build_key).?.wamr,
                        .module_path = mod_path,
                        .workload = scenario.workload,
                        .threads = scenario.threads,
                        .iterations = iters,
                        .timeout = args.timeout,
                        .min_interval_ns = minimum_interval_ns,
                        .record_fields = fields,
                    });
                    try records.append(allocator, rec);
                }
            }
        }
    }

    if (has_aot) {
        for (args.thread_counts) |tc| {
            const iters = try sizing.iterationCount(iteration_plan, "aot", "cancel-hot", tc);
            const pair_kind = "cancel-point-cost";
            const pair_key = try std.fmt.allocPrint(allocator, "cancel-points/hot/{d}", .{tc});
            const left = "cancel-points-off";
            const right = "cancel-points-on";

            for (phases) |ph| {
                var p_idx: i64 = 0;
                while (p_idx < ph.count) : (p_idx += 1) {
                    const global_idx: usize = @intCast(if (std.mem.eql(u8, ph.name, "warmup")) p_idx else warmups + p_idx);
                    const rev_order = if (revision_roles.len == 2)
                        benchmark_schema.alternatingPairOrder(global_idx, "baseline", "candidate")
                    else
                        [2][]const u8{ "candidate", "" };
                    const cond_order = spec.pairedConditionOrder(global_idx, pair_kind, left, right);
                    const inv_order = try spec.pairedInvocationOrder(allocator, global_idx, pair_kind, left, right, revision_roles);

                    for (inv_order) |inv| {
                        const rev = inv.revision;
                        const cond = inv.condition;
                        const ctx = contexts.get(rev).?;
                        const polls = if (std.mem.eql(u8, cond, "cancel-points-off")) "off" else "on";
                        const art_key = try std.fmt.allocPrint(allocator, "threaded-polls-{s}", .{polls});
                        const mod_path = ctx.aot_artifacts.get(art_key).?.string;
                        const r_ord: i64 = if (std.mem.eql(u8, rev_order[0], rev)) 0 else 1;
                        const c_ord: i64 = if (std.mem.eql(u8, cond_order[0], cond)) 0 else 1;

                        var fields = try spec.cloneObject(allocator, revision_fields.get(rev).?);
                        try fields.put(allocator, "revision", .{ .string = rev });
                        try fields.put(allocator, "revision_order", .{ .integer = r_ord });
                        try fields.put(allocator, "pair_kind", .{ .string = pair_kind });
                        try fields.put(allocator, "pair_execution", .{ .string = "sequential-position-balanced" });
                        try fields.put(allocator, "pair_key", .{ .string = pair_key });
                        try fields.put(allocator, "pair_index", .{ .integer = p_idx });
                        try fields.put(allocator, "phase", .{ .string = ph.name });
                        try fields.put(allocator, "order", .{ .integer = c_ord });
                        try fields.put(allocator, "condition", .{ .string = cond });
                        try fields.put(allocator, "pair_left", .{ .string = left });
                        try fields.put(allocator, "pair_right", .{ .string = right });
                        try fields.put(allocator, "mode", .{ .string = "aot" });
                        try fields.put(allocator, "threads_enabled", .{ .bool = true });
                        try fields.put(allocator, "thread_manager_enabled", .{ .bool = true });
                        try fields.put(allocator, "cancel_points", .{ .string = polls });
                        try fields.put(allocator, "static_cancel_poll_sites", .{
                            .integer = if (std.mem.eql(u8, polls, "on"))
                                ctx.aot_artifacts_metadata.get("cancel_poll_static").?.object.get("sites_enabled").?.integer
                            else
                                0,
                        });
                        try fields.put(allocator, "workload", .{ .string = "hot" });
                        try fields.put(allocator, "threads", .{ .integer = tc });
                        try fields.put(allocator, "iterations", .{ .integer = @intCast(iters) });

                        const rec = try runner.measureOnce(allocator, io, .{
                            .repo = ctx.repo,
                            .runner = runner_cmd,
                            .cpu_placement = cpu_placement,
                            .wamr_path = ctx.builds.get("enabled-aot").?.wamr,
                            .module_path = mod_path,
                            .workload = "hot",
                            .threads = tc,
                            .iterations = iters,
                            .timeout = args.timeout,
                            .min_interval_ns = minimum_interval_ns,
                            .record_fields = fields,
                        });
                        try records.append(allocator, rec);
                    }
                }
            }
        }
    }

    const summaries = try report.summarize(allocator, records.items);
    const pairs = try report.pairedSummaries(allocator, records.items);
    const comparisons = try report.comparisonSummaries(allocator, records.items);
    const ratios = try report.ratioOfRatiosSummaries(allocator, records.items);
    const candidate_rev = revisions.get("candidate").?.object;

    var metadata: ObjectMap = .empty;
    try metadata.put(allocator, "commit", candidate_rev.get("commit").?);
    try metadata.put(allocator, "tracked_diff_sha256", candidate_rev.get("tracked_diff_sha256").?);
    try metadata.put(allocator, "build_source_sha256", candidate_rev.get("build_source_sha256").?);
    try metadata.put(allocator, "revisions", .{ .object = revisions });
    var checkouts_map: ObjectMap = .empty;
    for (revision_roles) |role| {
        try checkouts_map.put(allocator, role, .{ .string = revision_repos.get(role).? });
    }
    try metadata.put(allocator, "revision_checkouts", .{ .object = checkouts_map });
    var fs_map: ObjectMap = .empty;
    try fs_map.put(allocator, "policy", .{ .string = spec.FIXTURE_SOURCE_POLICY });
    try fs_map.put(allocator, "role", .{ .string = "candidate" });
    try fs_map.put(allocator, "checkout", .{ .string = fixture_repo });
    try metadata.put(allocator, "fixture_source", .{ .object = fs_map });
    try metadata.put(allocator, "collected_at", .{ .string = try benchmark_schema.collectedAt(allocator, io) });
    try metadata.put(allocator, "platform_id", .{ .string = platform_id });
    try metadata.put(allocator, "fixture_set_sha256", .{ .string = fixture_set_sha256 });
    try metadata.put(allocator, "plan_sha256", .{ .string = plan_sha256 });
    try metadata.put(allocator, "measurement_plan_version", .{ .integer = spec.MEASUREMENT_PLAN_IDENTITY_VERSION });
    try metadata.put(allocator, "measurement_plan_sha256", .{ .string = measurement_plan_identity });
    try metadata.put(allocator, "cpu_placement", .{ .object = cpu_placement });
    var ra_map: ObjectMap = .empty;
    try ra_map.put(allocator, "policy", .{ .string = spec.REVISION_ARTIFACT_POLICY });
    try ra_map.put(allocator, "strategy", .{ .string = revision_artifact_strategy });
    try ra_map.put(allocator, "source_role", if (std.mem.eql(u8, revision_artifact_strategy, "shared-exact-build")) .{ .string = "baseline" } else .null);
    try metadata.put(allocator, "revision_artifacts", .{ .object = ra_map });
    try metadata.put(allocator, "checksum_preparation", .{ .object = checksum_preparation });
    try metadata.put(allocator, "host", .{ .object = host });
    try metadata.put(allocator, "host_pair", .{ .object = host_pair });
    var exec_map: ObjectMap = .empty;
    try exec_map.put(allocator, "target", .{ .string = args.target orelse "native" });
    try exec_map.put(allocator, "aot_target", .{ .string = exec_arch });
    var r_arr = Array.init(allocator);
    for (runner_cmd) |rc| try r_arr.append(.{ .string = rc });
    try exec_map.put(allocator, "runner", .{ .array = r_arr });
    try metadata.put(allocator, "execution", .{ .object = exec_map });

    var tools_map: ObjectMap = .empty;
    var aot_art_map: ObjectMap = .empty;
    for (revision_roles) |role| {
        const ctx = contexts.get(role).?;
        try tools_map.put(allocator, role, .{ .object = try runner.buildToolReport(allocator, io, ctx.builds_list, runner_cmd) });
        try aot_art_map.put(allocator, role, .{ .object = ctx.aot_artifacts_metadata });
    }
    try metadata.put(allocator, "tools", .{ .object = tools_map });
    try metadata.put(allocator, "fixture_toolchain", try spec.wasiSdkValue(allocator));
    try metadata.put(allocator, "fixtures", .{ .object = fixture_report });
    try metadata.put(allocator, "aot_artifacts", .{ .object = aot_art_map });

    var document: ObjectMap = .empty;
    try document.put(allocator, "schema_version", .{ .integer = spec.REPORT_SCHEMA_VERSION });
    try document.put(allocator, "kind", .{ .string = spec.KIND });
    try document.put(allocator, "metadata", .{ .object = metadata });
    try document.put(allocator, "plan", .{ .object = plan });
    var rec_arr = Array.init(allocator);
    for (records.items) |r| try rec_arr.append(.{ .object = r });
    try document.put(allocator, "records", .{ .array = rec_arr });
    var sum_arr = Array.init(allocator);
    for (summaries) |s| try sum_arr.append(.{ .object = s });
    try document.put(allocator, "summaries", .{ .array = sum_arr });
    var ps_arr = Array.init(allocator);
    for (pairs) |p| try ps_arr.append(.{ .object = p });
    try document.put(allocator, "paired_summaries", .{ .array = ps_arr });
    var cs_arr = Array.init(allocator);
    for (comparisons) |c| try cs_arr.append(.{ .object = c });
    try document.put(allocator, "comparison_summaries", .{ .array = cs_arr });
    var rr_arr = Array.init(allocator);
    for (ratios) |r| try rr_arr.append(.{ .object = r });
    try document.put(allocator, "ratio_of_ratios_summaries", .{ .array = rr_arr });
    try document.put(allocator, "quality_preflight", .{ .object = quality_preflight });

    const resolved_budget_path = if (args.budget) |b|
        (resolvePath(allocator, cwd, b) catch return error.OutOfMemory)
    else
        null;
    var budget_obj: ObjectMap = .empty;
    try budget_obj.put(allocator, "status", .{ .string = if (args.no_budget) "disabled" else "not-selected" });
    try budget_obj.put(allocator, "path", if (resolved_budget_path) |bp| .{ .string = bp } else .null);
    try budget_obj.put(allocator, "failures", .{ .array = Array.init(allocator) });
    try document.put(allocator, "budget", .{ .object = budget_obj });

    try report.validateReport(allocator, .{ .object = document });
    var budget_failures: []const []const u8 = &.{};
    if (resolved_budget_path) |bp| {
        const plat_budget = try report.loadBudget(allocator, io, bp, .{ .object = document });
        budget_failures = try report.evaluateBudget(allocator, plat_budget, .{ .object = document });
        var updated_budget = document.get("budget").?.object;
        try updated_budget.put(allocator, "status", .{ .string = if (budget_failures.len == 0) "passed" else "failed" });
        var bf_arr = Array.init(allocator);
        for (budget_failures) |bf| try bf_arr.append(.{ .string = bf });
        try updated_budget.put(allocator, "failures", .{ .array = bf_arr });
        try document.put(allocator, "budget", .{ .object = updated_budget });
        try report.validateReport(allocator, .{ .object = document });
    }

    const report_json_path = std.fs.path.join(allocator, &.{ output, "report.json" }) catch return error.OutOfMemory;
    const report_md_path = std.fs.path.join(allocator, &.{ output, "report.md" }) catch return error.OutOfMemory;
    try spec.atomicWriteJson(allocator, io, report_json_path, .{ .object = document });
    const md = try report.renderMarkdown(allocator, document);
    const md_nl = try std.fmt.allocPrint(allocator, "{s}\n", .{md});
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = report_md_path, .data = md_nl }) catch return fail("cannot write {s}", .{report_md_path});

    if (budget_failures.len > 0) {
        const joined = try std.mem.join(allocator, "; ", budget_failures);
        return fail("budget failures: {s}", .{joined});
    }
    std.debug.print("{s}\n", .{md});
    return 0;
}

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = init.io;
    const raw_args = try init.minimal.args.toSlice(allocator);
    const args = parseArgs(allocator, raw_args[1..]) catch |err| {
        std.debug.print("error: {s}\n", .{spec.lastErrorMessage()});
        return err;
    };
    const code = execute(allocator, io, init.minimal.environ, args) catch |err| {
        std.debug.print("error: {s}\n", .{spec.lastErrorMessage()});
        return err;
    };
    if (code != 0) std.process.exit(code);
}

// ============================================================================
// Unit Tests
// ============================================================================

test "guest math: hotKernelJumpAhead matches iterative reference and expectedResult" {
    const seed = spec.workerSeed(0);
    var acc: u64 = seed;
    var counter: u64 = spec.HOT_KERNEL_COUNTER_BASE;
    const steps: u64 = 257;
    var i: u64 = 0;
    while (i < steps) : (i += 1) {
        acc = spec.rotateLeftU64(acc, 7) ^ counter;
        counter +%= 1;
    }
    const jumped = try spec.hotKernelJumpAhead(seed, @as(i128, steps));
    try std.testing.expectEqual(acc, jumped);

    const hot_exp = try spec.expectedResult("hot", 4, 128, null);
    try std.testing.expectEqual(@as(u64, 512), hot_exp.operations);
    try std.testing.expect(hot_exp.checksum != 0);

    const atomic_exp = try spec.expectedResult("atomic", 4, 64, null);
    try std.testing.expectEqual(@as(u64, 256), atomic_exp.operations);

    const wn_exp = try spec.expectedResult("wait-notify", 4, 32, null);
    try std.testing.expectEqual(@as(u64, 128), wn_exp.operations);

    const sj_exp = try spec.expectedResult("spawn-join", 4, 16, null);
    try std.testing.expectEqual(@as(u64, 64), sj_exp.operations);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const json_line = try std.fmt.allocPrint(
        allocator,
        \\{{"kind":"{s}","workload":"hot","clock_id":"{s}","metric_kind":"{s}","threads":4,"iterations":128,"operations":512,"checksum":{d},"timed_loop_backedges":{d},"clock_calls_in_timed_loop":0,"raw_elapsed_ns":50001200,"timing_overhead_ns":1200,"elapsed_ns":50000000,"timing_overhead_ppm":23}}
    ,
        .{ hot_exp.kind, hot_exp.clock_id, hot_exp.metric_kind, hot_exp.checksum, hot_exp.timed_loop_backedges },
    );
    const parsed = try spec.parseGuestResult(allocator, json_line, hot_exp, 50_000_000, true);
    try std.testing.expectEqualStrings("hot", parsed.get("workload").?.string);
    try std.testing.expectEqual(@as(i64, 4), parsed.get("threads").?.integer);
    try std.testing.expectEqual(@as(i64, 50_000_000), parsed.get("elapsed_ns").?.integer);
    try std.testing.expectEqual(@as(i64, 50_001_200), parsed.get("raw_elapsed_ns").?.integer);
    try std.testing.expectEqual(@as(i64, 1200), parsed.get("timing_overhead_ns").?.integer);
    try std.testing.expectEqual(@as(i64, 23), parsed.get("timing_overhead_ppm").?.integer);
    try std.testing.expectEqualStrings(hot_exp.clock_id, parsed.get("clock_id").?.string);
}

test "cpuPlacementFromTopology and validateCpuPlacement" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const allowed = [_]i64{ 0, 1, 2, 3 };
    const entries = [_]spec.TopologyEntry{
        .{ .cpu = 0, .package_id = 0, .core_id = 0 },
        .{ .cpu = 1, .package_id = 0, .core_id = 1 },
        .{ .cpu = 2, .package_id = 0, .core_id = 0 },
        .{ .cpu = 3, .package_id = 0, .core_id = 1 },
    };
    const thread_counts = [_]i64{ 1, 2, 4 };
    const placement = try spec.cpuPlacementFromTopology(allocator, &allowed, &entries, &thread_counts, "taskset from util-linux 2.39");
    try spec.validateCpuPlacement(allocator, .{ .object = placement }, &thread_counts);

    const ordered = placement.get("ordered_logical_cpus").?.array.items;
    try std.testing.expectEqual(@as(usize, 4), ordered.len);
    try std.testing.expectEqual(@as(i64, 1), ordered[0].integer);
    try std.testing.expectEqual(@as(i64, 0), ordered[1].integer);
    try std.testing.expectEqual(@as(i64, 3), ordered[2].integer);
    try std.testing.expectEqual(@as(i64, 2), ordered[3].integer);

    const aff0 = try spec.directPairCpuAffinity(allocator, placement, 0, "threads-disabled", "threads-disabled", "threads-enabled");
    const aff1 = try spec.directPairCpuAffinity(allocator, placement, 0, "threads-enabled", "threads-disabled", "threads-enabled");
    try std.testing.expectEqual(@as(i64, 1), aff0[0]);
    try std.testing.expectEqual(@as(i64, 3), aff1[0]);
}

test "one-shot sizing resolves and validates against sizing contract" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const modes = [_][]const u8{ "interpreter", "aot" };
    const thread_counts = [_]i64{ 1, 2, 4 };
    const warmups: i64 = 1;
    const samples: i64 = 4;
    const timeout_seconds: f64 = 90.0;

    const overrides: sizing.IterationOverrides = .{};
    const pilot_iters = try sizing.resolvedIterationPlan(allocator, &modes, &thread_counts, overrides);
    const pair_plan = try sizing.plannedPairSpecs(allocator, &modes, &thread_counts);
    const pilot_order = try sizing.pilotOrderForPlan(allocator, pair_plan, &spec.REVISION_ROLES, pilot_iters);

    var pilot_records: std.ArrayList(ObjectMap) = .empty;
    for (pilot_order) |p_spec| {
        const sizing_wl = p_spec.get("workload").?.string;
        const guest_wl = if (std.mem.eql(u8, sizing_wl, "cancel-hot")) "hot" else sizing_wl;
        const threads = p_spec.get("threads").?.integer;
        const iters = spec.valueAsU64(p_spec.get("iterations").?).?;
        const exp_res = try spec.expectedResult(guest_wl, threads, @as(i128, iters), null);

        var rec = try spec.cloneObject(allocator, p_spec);
        try rec.put(allocator, "guest_elapsed_ns", .{ .integer = 5_000_000_000 });
        try rec.put(allocator, "raw_guest_elapsed_ns", .{ .integer = 5_000_001_000 });
        try rec.put(allocator, "timing_overhead_ns", .{ .integer = 1_000 });
        try rec.put(allocator, "host_wall_elapsed_ns", .{ .integer = 5_005_000_000 });
        try rec.put(allocator, "operations", .{ .integer = @intCast(exp_res.operations) });
        try rec.put(allocator, "correct", .{ .bool = true });
        try sizing.validateSizingPilot(allocator, rec, p_spec);
        try pilot_records.append(allocator, rec);
    }

    const res = try sizing.resolveOneShotSizing(
        allocator,
        pilot_records.items,
        pilot_order,
        &modes,
        &thread_counts,
        warmups,
        samples,
        timeout_seconds,
    );

    var plan: ObjectMap = .empty;
    var modes_arr = Array.init(allocator);
    for (modes) |m| try modes_arr.append(.{ .string = m });
    var tc_arr = Array.init(allocator);
    for (thread_counts) |tc| try tc_arr.append(.{ .integer = tc });
    var roles_arr = Array.init(allocator);
    for (spec.REVISION_ROLES) |r| try roles_arr.append(.{ .string = r });
    try plan.put(allocator, "modes", .{ .array = modes_arr });
    try plan.put(allocator, "thread_counts", .{ .array = tc_arr });
    try plan.put(allocator, "revision_roles", .{ .array = roles_arr });
    try plan.put(allocator, "warmups", .{ .integer = warmups });
    try plan.put(allocator, "samples", .{ .integer = samples });
    try plan.put(allocator, "timeout_seconds", .{ .float = timeout_seconds });
    try plan.put(allocator, "iterations", .{ .object = res.plan });
    var sizing_obj: ObjectMap = .empty;
    try sizing_obj.put(allocator, "algorithm", try sizing.sizingAlgorithmSpec(allocator, timeout_seconds));
    try sizing_obj.put(allocator, "pilot_iterations", .{ .object = pilot_iters });
    var po_arr = Array.init(allocator);
    for (pilot_order) |po| try po_arr.append(.{ .object = po });
    try sizing_obj.put(allocator, "pilot_order", .{ .array = po_arr });
    try sizing_obj.put(allocator, "resolved", .{ .object = res.resolution });
    try plan.put(allocator, "sizing", .{ .object = sizing_obj });
    var pairs_arr = Array.init(allocator);
    for (pair_plan) |p| try pairs_arr.append(.{ .object = p });
    try plan.put(allocator, "pairs", .{ .array = pairs_arr });

    try sizing.validateSizingPlan(allocator, plan);
}

test "loadBudget, summarize, pairedSummaries, comparisonSummaries, ratioOfRatiosSummaries, and parseArgs" {
    _ = &execute;
    _ = &parseArgs;
    _ = &report.validateReport;
    _ = &report.evaluateBudget;
    _ = &report.renderMarkdown;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const parsed_args = try parseArgs(allocator, &.{
        "--profile",
        "smoke",
        "--no-budget",
        "--platform-id",
        "ubuntu-24.04-aarch64",
        "--runner-environment",
        "github-hosted",
    });
    try std.testing.expectEqualStrings("smoke", parsed_args.profile);
    try std.testing.expect(parsed_args.no_budget);

    // Exercise report.summarize, pairedSummaries, comparisonSummaries, ratioOfRatiosSummaries
    var rec0: ObjectMap = .empty;
    try rec0.put(allocator, "phase", .{ .string = "measure" });
    try rec0.put(allocator, "revision", .{ .string = "baseline" });
    try rec0.put(allocator, "pair_kind", .{ .string = "workload-scaling" });
    try rec0.put(allocator, "pair_key", .{ .string = "interpreter:hot:2-vs-1" });
    try rec0.put(allocator, "pair_index", .{ .integer = 0 });
    try rec0.put(allocator, "pair_left", .{ .string = "threads-1" });
    try rec0.put(allocator, "pair_right", .{ .string = "threads-2" });
    try rec0.put(allocator, "condition", .{ .string = "threads-1" });
    try rec0.put(allocator, "mode", .{ .string = "interpreter" });
    try rec0.put(allocator, "workload", .{ .string = "hot" });
    try rec0.put(allocator, "threads", .{ .integer = 1 });
    try rec0.put(allocator, "threads_enabled", .{ .bool = true });
    try rec0.put(allocator, "cancel_points", .{ .string = "on" });
    try rec0.put(allocator, "cancel_poll_sites", .{ .integer = 0 });
    try rec0.put(allocator, "cancel_polls_per_operation", .{ .float = 1.0 });
    try rec0.put(allocator, "iterations", .{ .integer = 100 });
    try rec0.put(allocator, "operations", .{ .integer = 100 });
    try rec0.put(allocator, "checksum", .{ .integer = 42 });
    try rec0.put(allocator, "clock_id", .{ .string = spec.WASI_PROCESS_CPU_CLOCK_ID });
    try rec0.put(allocator, "metric_kind", .{ .string = "kernel-throughput" });
    try rec0.put(allocator, "elapsed_ns", .{ .integer = 50_000_000 });
    try rec0.put(allocator, "guest_elapsed_ns", .{ .integer = 50_000_000 });
    try rec0.put(allocator, "raw_guest_elapsed_ns", .{ .integer = 50_001_000 });
    try rec0.put(allocator, "timing_overhead_ns", .{ .integer = 1_000 });
    try rec0.put(allocator, "timing_overhead_ppm", .{ .integer = 20 });
    try rec0.put(allocator, "host_wall_elapsed_ns", .{ .integer = 52_000_000 });
    try rec0.put(allocator, "throughput_ops_per_second", .{ .float = 2000.0 });
    try rec0.put(allocator, "per_thread_ops_per_second", .{ .float = 2000.0 });
    try rec0.put(allocator, "correct", .{ .bool = true });
    var cpu_arr = Array.init(allocator);
    try cpu_arr.append(.{ .integer = 1 });
    try rec0.put(allocator, "cpu_affinity", .{ .array = cpu_arr });

    var records: std.ArrayList(ObjectMap) = .empty;
    for (0..4) |pair_idx| {
        for ([_][]const u8{ "baseline", "candidate" }) |rev| {
            var r_left = try spec.cloneObject(allocator, rec0);
            try r_left.put(allocator, "revision", .{ .string = rev });
            try r_left.put(allocator, "pair_index", .{ .integer = @intCast(pair_idx) });
            try records.append(allocator, r_left);

            var r_right = try spec.cloneObject(allocator, rec0);
            try r_right.put(allocator, "revision", .{ .string = rev });
            try r_right.put(allocator, "pair_index", .{ .integer = @intCast(pair_idx) });
            try r_right.put(allocator, "condition", .{ .string = "threads-2" });
            try r_right.put(allocator, "threads", .{ .integer = 2 });
            try r_right.put(allocator, "operations", .{ .integer = 200 });
            try r_right.put(allocator, "throughput_ops_per_second", .{ .float = 3800.0 });
            try r_right.put(allocator, "per_thread_ops_per_second", .{ .float = 1900.0 });
            try records.append(allocator, r_right);
        }
    }

    const summary = try report.summarize(allocator, records.items);
    try std.testing.expectEqual(@as(usize, 4), summary.len);

    const paired = try report.pairedSummaries(allocator, records.items);
    try std.testing.expectEqual(@as(usize, 2), paired.len);
    const th_rol = paired[0].get("throughput_right_over_left").?.object;
    try std.testing.expectApproxEqAbs(@as(f64, 1.9), spec.valueAsF64(th_rol.get("median").?).?, 1e-6);

    const comps = try report.comparisonSummaries(allocator, records.items);
    try std.testing.expectEqual(@as(usize, 2), comps.len);

    const rors = try report.ratioOfRatiosSummaries(allocator, records.items);
    try std.testing.expectEqual(@as(usize, 1), rors.len);
    const th_ror = rors[0].get("throughput_ratio_of_ratios").?.object;
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), spec.valueAsF64(th_ror.get("median").?).?, 1e-6);

    _ = &execute;
    _ = &main;
}

test "real fixtures.sha256, calibrated budget.json, and cancelPollSignature validate against Zig implementation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const fixture_report = try runner.resolveFixtures(allocator, io, ".");
    const fixture_set_sha256 = try runner.fixtureSetIdentity(allocator, fixture_report);
    try std.testing.expect(spec.isHexSha256(fixture_set_sha256));

    const sig = spec.cancelPollSignature("x86_64").?;
    try std.testing.expectEqual(@as(usize, 9), sig.len);

    const budget_bytes = try std.Io.Dir.cwd().readFileAlloc(io, "tests/benchmarks/wasi-threads/budget.json", allocator, .limited(16 * 1024 * 1024));
    const parsed_budget = try std.json.parseFromSlice(Value, allocator, budget_bytes, .{});
    const budget_obj = parsed_budget.value.object;
    try std.testing.expect(budget_obj.get("calibrated").?.bool);
    try std.testing.expect(budget_obj.get("enforcement").?.bool);
    const prov = budget_obj.get("calibration_provenance").?.object;
    try std.testing.expectEqualStrings(fixture_set_sha256, prov.get("fixture_set_sha256").?.string);
    const platforms = budget_obj.get("platforms").?.object;
    try std.testing.expect(platforms.contains("ubuntu-22.04-x86_64"));
    try std.testing.expect(platforms.contains("ubuntu-24.04-aarch64"));

    // Verify calibration-compatible measurement plan identity matching
    var candidate_plan: ObjectMap = .empty;
    try candidate_plan.put(allocator, "comparison_purpose", .{ .string = "candidate-evaluation" });
    const iters_obj: ObjectMap = .empty;
    try candidate_plan.put(allocator, "iterations", .{ .object = iters_obj });
    var sizing_obj: ObjectMap = .empty;
    const resolved_obj: ObjectMap = .empty;
    try sizing_obj.put(allocator, "resolved", .{ .object = resolved_obj });
    try candidate_plan.put(allocator, "sizing", .{ .object = sizing_obj });
    var modes_arr = Array.init(allocator);
    try modes_arr.append(.{ .string = "interpreter" });
    try modes_arr.append(.{ .string = "aot" });
    try candidate_plan.put(allocator, "modes", .{ .array = modes_arr });
    var tc_arr = Array.init(allocator);
    for ([_]i64{ 1, 2, 4, 8 }) |tc| try tc_arr.append(.{ .integer = tc });
    try candidate_plan.put(allocator, "thread_counts", .{ .array = tc_arr });
    var pf_obj: ObjectMap = .empty;
    try pf_obj.put(allocator, "enabled", .{ .bool = false });
    try pf_obj.put(allocator, "probe_count", .{ .integer = 0 });
    try candidate_plan.put(allocator, "scheduler_barrier_preflight", .{ .object = pf_obj });

    var calibrated_plan = (try spec.jsonClone(allocator, .{ .object = candidate_plan })).object;
    var cal_pf = calibrated_plan.getPtr("scheduler_barrier_preflight").?;
    try cal_pf.object.put(allocator, "enabled", .{ .bool = true });
    try cal_pf.object.put(allocator, "probe_count", .{ .integer = spec.TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD * 4 });
    const calibrated_sha = try spec.measurementPlanSha256(allocator, .{ .object = calibrated_plan });
    try std.testing.expect(try spec.isCalibrationCompatibleMeasurementPlanSha256(allocator, .{ .object = candidate_plan }, calibrated_sha));
}
