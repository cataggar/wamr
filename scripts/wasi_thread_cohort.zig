//! Dispatch, download, validate, and derive budgets from an immutable paired WASI thread benchmark cohort.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const ObjectMap = std.json.ObjectMap;
const Array = std.json.Array;

const benchmark_schema = @import("benchmark_schema.zig");
const benchmark_stats = @import("benchmark_stats.zig");
const spec = @import("wasi_thread_bench_spec.zig");
const sizing = @import("wasi_thread_bench_sizing.zig");
const runner = @import("wasi_thread_bench_runner.zig");
const report = @import("wasi_thread_bench_report.zig");

pub const HarnessError = spec.HarnessError;
const fail = spec.fail;
const require = spec.require;

pub const DEFAULT_PLATFORMS = [_][]const u8{
    "ubuntu-22.04-x86_64",
    "ubuntu-24.04-aarch64",
};

pub const RUNNER_TARGETS = [_][]const u8{
    "github-hosted",
    "trusted-calibration",
};

pub const TRUSTED_X86_RUNNER_NAME = "vm31e-wamr-temp-20260906";
pub const TRUSTED_X86_RUNNER_LABEL = "wamr-temp-20260906";
pub const TRUSTED_X86_PLATFORM = "ubuntu-22.04-x86_64";
pub const DEFAULT_DISPATCH_TIMEOUT_SECONDS: f64 = 120.0 * 60.0 * 60.0;

pub const DERIVATION_POLICY_KIND = "wasi-thread-budget-derivation-policy";
pub const DERIVATION_EVIDENCE_KIND = "wasi-thread-budget-derivation-evidence";
pub const DIRECT_THREAD_DELTA_POLICY_LIMIT: f64 = 0.02;
pub const MINIMUM_TRAINING_REPORTS_PER_CPU_CLASS: i64 = 16;
pub const MINIMUM_HOLDOUT_REPORTS_PER_CPU_CLASS: i64 = 4;
pub const MAD_SCALE: f64 = 1.4826;
pub const ROBUST_MAD_MULTIPLIER: f64 = 6.0;

pub fn requireSha(value: []const u8, name: []const u8) HarnessError![]const u8 {
    if (!spec.isHexCommit(value)) {
        return fail("{s} must be an immutable 40-character lowercase hex commit", .{name});
    }
    return value;
}

pub fn isTrustedCalibrationTag(ref: []const u8) bool {
    const prefix = "wasi-thread-calibration-";
    if (!std.mem.startsWith(u8, ref, prefix)) return false;
    const suffix = ref[prefix.len..];
    if (suffix.len == 0) return false;
    if (!std.ascii.isAlphanumeric(suffix[0])) return false;
    for (suffix) |c| {
        if (!std.ascii.isAlphanumeric(c) and c != '.' and c != '_' and c != '-') return false;
    }
    return true;
}

pub fn isAllDigits(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s) |c| {
        if (!std.ascii.isDigit(c)) return false;
    }
    return true;
}

fn containsStr(haystack: []const []const u8, needle: []const u8) bool {
    for (haystack) |item| {
        if (std.mem.eql(u8, item, needle)) return true;
    }
    return false;
}

pub fn ghJson(allocator: Allocator, io: std.Io, command: []const []const u8, timeout_seconds: f64) HarnessError!Value {
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.append(allocator, "gh");
    try argv.appendSlice(allocator, command);
    const res = try runner.runProcess(allocator, io, argv.items, ".", timeout_seconds, null);
    if (res.exit_code != 0) {
        return fail("GitHub command failed: gh {s}: {s}", .{ std.mem.join(allocator, " ", command) catch "gh", res.stderr });
    }
    const parsed = std.json.parseFromSlice(Value, allocator, res.stdout, .{}) catch {
        return fail("GitHub command returned invalid JSON", .{});
    };
    return parsed.value;
}

pub fn validateCpuClassMapping(
    allocator: Allocator,
    value: Value,
    context: []const u8,
) HarnessError!std.StringHashMapUnmanaged([]const []const u8) {
    if (value != .object or value.object.count() != DEFAULT_PLATFORMS.len) {
        return fail("{s} must cover exactly the canonical platforms", .{context});
    }
    var out: std.StringHashMapUnmanaged([]const []const u8) = .empty;
    for (DEFAULT_PLATFORMS) |platform| {
        const classes_val = value.object.get(platform) orelse {
            return fail("{s} must cover exactly the canonical platforms", .{context});
        };
        if (classes_val != .array or classes_val.array.items.len == 0) {
            return fail("{s} has invalid CPU classes for {s}", .{ context, platform });
        }
        var list: std.ArrayList([]const u8) = .empty;
        for (classes_val.array.items, 0..) |item, idx| {
            if (item != .string or item.string.len == 0 or
                !std.mem.eql(u8, item.string, std.mem.trim(u8, item.string, " \t\r\n")))
            {
                return fail("{s} has invalid CPU classes for {s}", .{ context, platform });
            }
            if (idx > 0 and std.mem.order(u8, list.items[idx - 1], item.string) != .lt) {
                return fail("{s} has invalid CPU classes for {s}", .{ context, platform });
            }
            try list.append(allocator, item.string);
        }
        try out.put(allocator, platform, try list.toOwnedSlice(allocator));
    }
    return out;
}

pub fn parseCpuClassAssignments(
    allocator: Allocator,
    values: []const []const u8,
) HarnessError!std.StringHashMapUnmanaged([]const []const u8) {
    var lists: [DEFAULT_PLATFORMS.len]std.ArrayList([]const u8) = .{ .empty, .empty };
    for (values) |val| {
        const eq = std.mem.indexOfScalar(u8, val, '=') orelse {
            return fail("--accepted-cpu-class must use PLATFORM=EXACT_CPU_MODEL", .{});
        };
        const platform = val[0..eq];
        const cpu_class = val[eq + 1 ..];
        if (cpu_class.len == 0 or !std.mem.eql(u8, cpu_class, std.mem.trim(u8, cpu_class, " \t\r\n"))) {
            return fail("--accepted-cpu-class must use a canonical platform and exact non-empty CPU model", .{});
        }
        var found = false;
        for (DEFAULT_PLATFORMS, 0..) |p, idx| {
            if (std.mem.eql(u8, p, platform)) {
                try lists[idx].append(allocator, cpu_class);
                found = true;
                break;
            }
        }
        if (!found) {
            return fail("--accepted-cpu-class must use a canonical platform and exact non-empty CPU model", .{});
        }
    }
    var obj: ObjectMap = .empty;
    for (DEFAULT_PLATFORMS, 0..) |p, idx| {
        std.mem.sort([]const u8, lists[idx].items, {}, struct {
            fn lt(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.lt);
        var arr = Array.init(allocator);
        for (lists[idx].items) |item| try arr.append(.{ .string = item });
        try obj.put(allocator, p, .{ .array = arr });
    }
    return validateCpuClassMapping(allocator, .{ .object = obj }, "--accepted-cpu-class");
}

pub const DispatchArgs = struct {
    repository: []const u8 = "cataggar/wamr",
    workflow: []const u8 = "wasi-thread-bench.yml",
    workflow_ref: []const u8 = "main",
    baseline_sha: []const u8 = "",
    candidate_sha: []const u8 = "",
    purpose: []const u8 = "",
    profile: []const u8 = "authoritative",
    warmups: i64 = 2,
    samples: i64 = 12,
    runner_target: []const u8 = "trusted-calibration",
    accepted_cpu_class: []const []const u8 = &.{},
    runs: i64 = 20,
    training_runs: ?i64 = null,
    start_sequence: i64 = 1,
    end_sequence: ?i64 = null,
    max_in_flight: i64 = 2,
    timeout_seconds: f64 = DEFAULT_DISPATCH_TIMEOUT_SECONDS,
    poll_seconds: f64 = 60.0,
    lookup_attempts: i64 = 30,
    lookup_seconds: f64 = 2.0,
    output: []const u8 = "wasi-thread-cohort-dispatch.json",
};

pub const ValidatedDispatchOptions = struct {
    baseline_sha: []const u8,
    candidate_sha: []const u8,
    training_runs: i64,
    accepted_cpu_classes: std.StringHashMapUnmanaged([]const []const u8),
};

pub fn validateDispatchOptions(
    allocator: Allocator,
    args: DispatchArgs,
) HarnessError!ValidatedDispatchOptions {
    if (args.repository.len == 0) return fail("--repository must not be empty", .{});
    if (args.workflow.len == 0) return fail("--workflow must not be empty", .{});
    if (args.workflow_ref.len == 0) return fail("--workflow-ref must not be empty", .{});

    const baseline_sha = try requireSha(args.baseline_sha, "baseline SHA");
    const candidate_sha = try requireSha(args.candidate_sha, "candidate SHA");
    if (!containsStr(&spec.COMPARISON_PURPOSES, args.purpose)) {
        return fail("unsupported comparison purpose", .{});
    }
    if (spec.profileCounts(args.profile) == null) {
        return fail("unsupported measurement profile", .{});
    }
    if (!containsStr(&RUNNER_TARGETS, args.runner_target)) {
        return fail("unsupported runner target", .{});
    }
    if (args.warmups < 0 or args.samples <= 0) {
        return fail("warmups must be non-negative and samples must be positive", .{});
    }
    if (@rem(args.samples, 4) != 0) {
        return fail("paired revision samples must be divisible by 4", .{});
    }
    if (std.mem.eql(u8, args.purpose, "noise-calibration") and !std.mem.eql(u8, baseline_sha, candidate_sha)) {
        return fail("noise calibration requires identical target SHAs", .{});
    }
    if (std.mem.eql(u8, args.purpose, "candidate-evaluation") and std.mem.eql(u8, baseline_sha, candidate_sha)) {
        return fail("candidate evaluation requires distinct target SHAs", .{});
    }
    if (std.mem.eql(u8, args.runner_target, "trusted-calibration") and !std.mem.eql(u8, args.purpose, "noise-calibration")) {
        return fail("the trusted calibration runner accepts noise calibration only", .{});
    }
    if (std.mem.eql(u8, args.runner_target, "trusted-calibration") and !isTrustedCalibrationTag(args.workflow_ref)) {
        return fail("the trusted calibration runner requires an immutable wasi-thread-calibration-* workflow tag", .{});
    }
    if (args.runs <= 1 or args.max_in_flight <= 0 or args.poll_seconds < 0.0 or
        args.lookup_attempts <= 0 or args.lookup_seconds < 0.0 or
        !std.math.isFinite(args.timeout_seconds) or args.timeout_seconds <= 0.0)
    {
        return fail("--runs must exceed one; dispatch counts and timeout must be positive", .{});
    }
    if (std.mem.eql(u8, args.runner_target, "trusted-calibration") and args.max_in_flight > 10) {
        return fail("trusted calibration --max-in-flight cannot exceed 10", .{});
    }
    const training_runs = args.training_runs orelse @divFloor(args.runs, 2);
    if (training_runs <= 0 or training_runs >= args.runs) {
        return fail("--training-runs must be between one and runs minus one", .{});
    }
    const end_seq = args.end_sequence orelse args.runs;
    if (!(1 <= args.start_sequence and args.start_sequence <= end_seq and end_seq <= args.runs)) {
        return fail("--start-sequence and --end-sequence must satisfy 1 <= start-sequence <= end-sequence <= runs", .{});
    }
    const accepted_cpu_classes = try parseCpuClassAssignments(allocator, args.accepted_cpu_class);
    return .{
        .baseline_sha = baseline_sha,
        .candidate_sha = candidate_sha,
        .training_runs = training_runs,
        .accepted_cpu_classes = accepted_cpu_classes,
    };
}

pub const ValidatedDispatchState = struct {
    baseline_sha: []const u8,
    candidate_sha: []const u8,
    workflow_head_sha: []const u8,
    cohort_id: []const u8,
    purpose: []const u8,
    profile: []const u8,
    warmups: i64,
    samples: i64,
    runner_target: []const u8,
    accepted_cpu_classes: std.StringHashMapUnmanaged([]const []const u8),
    requested_runs: i64,
    requested_reports: i64,
    timeout_seconds: f64,
    run_by_id: std.StringHashMapUnmanaged(ObjectMap),
    split: ObjectMap,
};

pub fn validateDispatchState(
    allocator: Allocator,
    state: ObjectMap,
    required_platforms: []const []const u8,
    allow_partial: bool,
) HarnessError!ValidatedDispatchState {
    const sv = state.get("schema_version") orelse return fail("paired cohort requires a schema-v4 dispatch manifest", .{});
    const kind = state.get("kind") orelse return fail("paired cohort requires a schema-v4 dispatch manifest", .{});
    if (sv != .integer or sv.integer != spec.REPORT_SCHEMA_VERSION or
        kind != .string or !std.mem.eql(u8, kind.string, "wasi-thread-cohort-dispatch"))
    {
        return fail("paired cohort requires a schema-v4 dispatch manifest", .{});
    }
    if (required_platforms.len != DEFAULT_PLATFORMS.len) {
        return fail("cohort platforms must be exactly the canonical hosted set", .{});
    }
    const req_plats = state.get("required_platforms") orelse return fail("dispatch manifest platform order or identity changed", .{});
    if (req_plats != .array or req_plats.array.items.len != required_platforms.len) {
        return fail("dispatch manifest platform order or identity changed", .{});
    }
    for (required_platforms, 0..) |rp, idx| {
        const item = req_plats.array.items[idx];
        if (item != .string or !std.mem.eql(u8, item.string, rp)) {
            return fail("dispatch manifest platform order or identity changed", .{});
        }
    }
    for ([_][]const u8{ "repository", "workflow", "workflow_ref" }) |key| {
        const v = state.get(key) orelse return fail("dispatch manifest {s}", .{key});
        if (v != .string or v.string.len == 0) return fail("dispatch manifest {s}", .{key});
    }
    const cohort_id_v = state.get("cohort_id") orelse return fail("dispatch manifest cohort_id", .{});
    if (cohort_id_v != .string or !spec.isHexLen(cohort_id_v.string, 32)) {
        return fail("dispatch manifest cohort_id", .{});
    }
    const b_sha_v = state.get("baseline_sha") orelse return fail("baseline SHA must be an immutable 40-character lowercase hex commit", .{});
    const c_sha_v = state.get("candidate_sha") orelse return fail("candidate SHA must be an immutable 40-character lowercase hex commit", .{});
    const w_sha_v = state.get("workflow_head_sha") orelse return fail("workflow head SHA must be an immutable 40-character lowercase hex commit", .{});
    if (b_sha_v != .string or c_sha_v != .string or w_sha_v != .string) {
        return fail("dispatch manifest SHA fields must be strings", .{});
    }
    const baseline_sha = try requireSha(b_sha_v.string, "baseline SHA");
    const candidate_sha = try requireSha(c_sha_v.string, "candidate SHA");
    const workflow_head_sha = try requireSha(w_sha_v.string, "workflow head SHA");

    const purpose_v = state.get("comparison_purpose") orelse return fail("dispatch manifest comparison purpose", .{});
    const profile_v = state.get("profile") orelse return fail("dispatch manifest profile", .{});
    const rt_v = state.get("runner_target") orelse return fail("dispatch manifest runner target", .{});
    if (purpose_v != .string or !containsStr(&spec.COMPARISON_PURPOSES, purpose_v.string)) {
        return fail("dispatch manifest comparison purpose", .{});
    }
    if (profile_v != .string or spec.profileCounts(profile_v.string) == null) {
        return fail("dispatch manifest profile", .{});
    }
    if (rt_v != .string or !containsStr(&RUNNER_TARGETS, rt_v.string)) {
        return fail("dispatch manifest runner target", .{});
    }
    const acc_v = state.get("accepted_cpu_classes") orelse return fail("dispatch manifest accepted_cpu_classes", .{});
    const accepted_cpu_classes = try validateCpuClassMapping(allocator, acc_v, "dispatch manifest accepted_cpu_classes");

    const warmups_v = state.get("warmups") orelse return fail("dispatch manifest warmup/sample plan", .{});
    const samples_v = state.get("samples") orelse return fail("dispatch manifest warmup/sample plan", .{});
    if (warmups_v != .integer or warmups_v.integer < 0 or
        samples_v != .integer or samples_v.integer <= 0 or @rem(samples_v.integer, 4) != 0)
    {
        return fail("dispatch manifest warmup/sample plan", .{});
    }
    if (std.mem.eql(u8, purpose_v.string, "noise-calibration") and !std.mem.eql(u8, baseline_sha, candidate_sha)) {
        return fail("noise calibration dispatch has mixed target SHAs", .{});
    }
    if (std.mem.eql(u8, purpose_v.string, "candidate-evaluation") and std.mem.eql(u8, baseline_sha, candidate_sha)) {
        return fail("candidate evaluation dispatch has identical target SHAs", .{});
    }
    const req_runs_v = state.get("requested_runs") orelse return fail("dispatch manifest requested report count or timeout", .{});
    const req_reps_v = state.get("requested_reports") orelse return fail("dispatch manifest requested report count or timeout", .{});
    const timeout_v = state.get("timeout_seconds") orelse return fail("dispatch manifest requested report count or timeout", .{});
    const timeout_f = spec.valueAsF64(timeout_v) orelse return fail("dispatch manifest requested report count or timeout", .{});
    if (req_runs_v != .integer or req_runs_v.integer <= 1 or
        req_reps_v != .integer or req_reps_v.integer != req_runs_v.integer * @as(i64, @intCast(required_platforms.len)) or
        !std.math.isFinite(timeout_f) or timeout_f <= 0.0)
    {
        return fail("dispatch manifest requested report count or timeout", .{});
    }

    const split_v = state.get("split") orelse return fail("dispatch manifest training/holdout split", .{});
    if (split_v != .object) return fail("dispatch manifest training/holdout split", .{});
    const split = split_v.object;
    const method_v = split.get("method") orelse return fail("dispatch manifest training/holdout split", .{});
    const assigns_v = split.get("assignments") orelse return fail("dispatch manifest training/holdout split", .{});
    if (method_v != .string or !std.mem.eql(u8, method_v.string, "predeclared-sequence") or
        assigns_v != .array or assigns_v.array.items.len != @as(usize, @intCast(req_runs_v.integer)))
    {
        return fail("dispatch manifest training/holdout split", .{});
    }

    var assignment_by_seq: std.AutoHashMapUnmanaged(i64, []const u8) = .empty;
    var training_count: i64 = 0;
    var holdout_count: i64 = 0;
    for (assigns_v.array.items) |item| {
        if (item != .object) return fail("dispatch manifest split assignment", .{});
        const seq_v = item.object.get("sequence") orelse return fail("dispatch manifest split assignment", .{});
        const part_v = item.object.get("partition") orelse return fail("dispatch manifest split assignment", .{});
        if (seq_v != .integer or part_v != .string or seq_v.integer < 1 or seq_v.integer > req_runs_v.integer) {
            return fail("dispatch manifest split assignment", .{});
        }
        if (assignment_by_seq.contains(seq_v.integer)) return fail("dispatch manifest split assignment", .{});
        if (std.mem.eql(u8, part_v.string, "training")) {
            training_count += 1;
        } else if (std.mem.eql(u8, part_v.string, "holdout")) {
            holdout_count += 1;
        } else {
            return fail("dispatch manifest split assignment", .{});
        }
        try assignment_by_seq.put(allocator, seq_v.integer, part_v.string);
    }
    const tr_v = split.get("training_runs") orelse return fail("dispatch manifest split counts", .{});
    const hr_v = split.get("holdout_runs") orelse return fail("dispatch manifest split counts", .{});
    if (training_count == 0 or holdout_count == 0 or
        tr_v != .integer or tr_v.integer != training_count or
        hr_v != .integer or hr_v.integer != holdout_count)
    {
        return fail("dispatch manifest split counts", .{});
    }

    const runs_v = state.get("runs") orelse return fail("dispatch manifest run count", .{});
    if (runs_v != .array) return fail("dispatch manifest run count", .{});
    const n_runs: i64 = @intCast(runs_v.array.items.len);
    if (allow_partial) {
        if (n_runs < 1 or n_runs > req_runs_v.integer) return fail("dispatch manifest run count", .{});
    } else {
        if (n_runs != req_runs_v.integer) return fail("dispatch manifest run count", .{});
    }

    var run_by_id: std.StringHashMapUnmanaged(ObjectMap) = .empty;
    var seen_sequences: std.AutoHashMapUnmanaged(i64, void) = .empty;
    for (runs_v.array.items) |r_val| {
        if (r_val != .object) return fail("dispatch manifest run record", .{});
        const r_obj = r_val.object;
        const seq_v = r_obj.get("sequence") orelse return fail("dispatch manifest duplicate or invalid run identity", .{});
        const rid_v = r_obj.get("run_id") orelse return fail("dispatch manifest duplicate or invalid run identity", .{});
        if (seq_v != .integer or seen_sequences.contains(seq_v.integer) or !assignment_by_seq.contains(seq_v.integer)) {
            return fail("dispatch manifest duplicate or invalid run identity", .{});
        }
        const rid_str = switch (rid_v) {
            .integer => |i| if (i <= 0) return fail("dispatch manifest duplicate or invalid run identity", .{}) else try std.fmt.allocPrint(allocator, "{d}", .{i}),
            .string => |s| if (!isAllDigits(s)) return fail("dispatch manifest duplicate or invalid run identity", .{}) else s,
            else => return fail("dispatch manifest duplicate or invalid run identity", .{}),
        };
        if (run_by_id.contains(rid_str)) return fail("dispatch manifest duplicate or invalid run identity", .{});
        const exp_part = assignment_by_seq.get(seq_v.integer).?;
        const part_v = r_obj.get("partition") orelse return fail("dispatch manifest run split changed after dispatch", .{});
        if (part_v != .string or !std.mem.eql(u8, part_v.string, exp_part)) {
            return fail("dispatch manifest run split changed after dispatch", .{});
        }
        const exp_name = try std.fmt.allocPrint(allocator, "WASI thread cohort-{s}-{d}-{s}", .{ cohort_id_v.string, seq_v.integer, exp_part });
        const rname_v = r_obj.get("run_name") orelse return fail("dispatch manifest run name changed after dispatch", .{});
        if (rname_v != .string or !std.mem.eql(u8, rname_v.string, exp_name)) {
            return fail("dispatch manifest run name changed after dispatch", .{});
        }
        const st_v = r_obj.get("status") orelse return fail("workflow run {s} is partial or unsuccessful", .{rid_str});
        const conc_v = r_obj.get("conclusion") orelse return fail("workflow run {s} is partial or unsuccessful", .{rid_str});
        if (st_v != .string or !std.mem.eql(u8, st_v.string, "completed") or
            conc_v != .string or !std.mem.eql(u8, conc_v.string, "success"))
        {
            return fail("workflow run {s} is partial or unsuccessful", .{rid_str});
        }
        const wh_v = r_obj.get("workflow_head_sha") orelse return fail("workflow run {s} used a mixed workflow head", .{rid_str});
        if (wh_v != .string or !std.ascii.eqlIgnoreCase(wh_v.string, workflow_head_sha)) {
            return fail("workflow run {s} used a mixed workflow head", .{rid_str});
        }
        try seen_sequences.put(allocator, seq_v.integer, {});
        try run_by_id.put(allocator, rid_str, r_obj);
    }

    return .{
        .baseline_sha = baseline_sha,
        .candidate_sha = candidate_sha,
        .workflow_head_sha = workflow_head_sha,
        .cohort_id = cohort_id_v.string,
        .purpose = purpose_v.string,
        .profile = profile_v.string,
        .warmups = warmups_v.integer,
        .samples = samples_v.integer,
        .runner_target = rt_v.string,
        .accepted_cpu_classes = accepted_cpu_classes,
        .requested_runs = req_runs_v.integer,
        .requested_reports = req_reps_v.integer,
        .timeout_seconds = timeout_f,
        .run_by_id = run_by_id,
        .split = split,
    };
}

pub fn reportCalibrationMetrics(allocator: Allocator, document: ObjectMap) HarnessError!ObjectMap {
    const comp_arr = document.get("comparison_summaries").?.array;
    var comparisons = Array.init(allocator);
    for (comp_arr.items) |item_v| {
        const item = item_v.object;
        var entry: ObjectMap = .empty;
        try entry.put(allocator, "pair_kind", item.get("pair_kind").?);
        try entry.put(allocator, "pair_key", item.get("pair_key").?);
        try entry.put(allocator, "condition", item.get("condition").?);
        try entry.put(allocator, "metric_kind", item.get("metric_kind").?);
        try entry.put(allocator, "elapsed_candidate_over_baseline", item.get("elapsed_candidate_over_baseline").?.object.get("samples").?);
        try entry.put(allocator, "throughput_candidate_over_baseline", item.get("throughput_candidate_over_baseline").?.object.get("samples").?);
        try comparisons.append(.{ .object = entry });
    }

    const ror_arr = document.get("ratio_of_ratios_summaries").?.array;
    var ratio_of_ratios = Array.init(allocator);
    for (ror_arr.items) |item_v| {
        const item = item_v.object;
        var entry: ObjectMap = .empty;
        try entry.put(allocator, "pair_kind", item.get("pair_kind").?);
        try entry.put(allocator, "pair_key", item.get("pair_key").?);
        try entry.put(allocator, "left", item.get("left").?);
        try entry.put(allocator, "right", item.get("right").?);
        try entry.put(allocator, "elapsed_ratio_of_ratios", item.get("elapsed_ratio_of_ratios").?.object.get("samples").?);
        try entry.put(allocator, "throughput_ratio_of_ratios", item.get("throughput_ratio_of_ratios").?.object.get("samples").?);
        try ratio_of_ratios.append(.{ .object = entry });
    }

    const paired_arr = document.get("paired_summaries").?.array;
    var direct_candidate = Array.init(allocator);
    for (paired_arr.items) |item_v| {
        const item = item_v.object;
        if (std.mem.eql(u8, item.get("revision").?.string, "candidate") and
            std.mem.eql(u8, item.get("pair_kind").?.string, "single-infrastructure"))
        {
            var entry: ObjectMap = .empty;
            try entry.put(allocator, "pair_kind", item.get("pair_kind").?);
            try entry.put(allocator, "pair_key", item.get("pair_key").?);
            try entry.put(allocator, "left", item.get("left").?);
            try entry.put(allocator, "right", item.get("right").?);
            try entry.put(allocator, "elapsed_right_over_left", item.get("elapsed_right_over_left").?.object.get("samples").?);
            try entry.put(allocator, "throughput_right_over_left", item.get("throughput_right_over_left").?.object.get("samples").?);
            try direct_candidate.append(.{ .object = entry });
        }
    }
    if (direct_candidate.items.len == 0) {
        return fail("report cannot prove the direct candidate threads-enabled/disabled delta: candidate single-infrastructure raw ratios are missing", .{});
    }

    var out: ObjectMap = .empty;
    try out.put(allocator, "comparisons", .{ .array = comparisons });
    try out.put(allocator, "ratio_of_ratios", .{ .array = ratio_of_ratios });
    try out.put(allocator, "direct_candidate_single_infrastructure", .{ .array = direct_candidate });
    return out;
}

fn finiteNumber(value: Value, label: []const u8, positive: bool, nonnegative: bool) HarnessError!f64 {
    const num = spec.valueAsF64(value) orelse {
        const qual = if (positive) "positive " else if (nonnegative) "non-negative " else "";
        return fail("{s} must be a finite {s}number", .{ label, qual });
    };
    if ((positive and num <= 0.0) or (nonnegative and num < 0.0)) {
        const qual = if (positive) "positive " else if (nonnegative) "non-negative " else "";
        return fail("{s} must be a finite {s}number", .{ label, qual });
    }
    return num;
}

pub fn validateDerivationPolicy(allocator: Allocator, policy: ObjectMap) HarnessError!ObjectMap {
    const expected_keys = [_][]const u8{
        "schema_version",
        "kind",
        "minimum_reports_per_platform",
        "rounding_cushion_log",
        "engineering_policy_ceiling_log",
        "direct_candidate_single_infrastructure_max_delta_fraction",
    };
    if (policy.count() != expected_keys.len) {
        return fail("derivation policy fields are incomplete or unknown", .{});
    }
    for (expected_keys) |k| {
        if (!policy.contains(k)) return fail("derivation policy fields are incomplete or unknown", .{});
    }
    const sv = policy.get("schema_version").?;
    const kind = policy.get("kind").?;
    if (sv != .integer or sv.integer != 1 or kind != .string or !std.mem.eql(u8, kind.string, DERIVATION_POLICY_KIND)) {
        return fail("derivation policy schema/kind mismatch", .{});
    }
    const min_v = policy.get("minimum_reports_per_platform").?;
    if (min_v != .integer or min_v.integer < 20) {
        return fail("derivation policy minimum_reports_per_platform must be >= 20", .{});
    }
    const cushion = try finiteNumber(policy.get("rounding_cushion_log").?, "derivation policy rounding_cushion_log", false, true);

    const ceilings_v = policy.get("engineering_policy_ceiling_log").?;
    if (ceilings_v != .object) return fail("derivation policy engineering ceilings are incomplete or unknown", .{});
    const ceiling_names = [_][]const u8{
        "comparison_elapsed_upper",
        "comparison_throughput_lower",
        "ratio_of_ratios_elapsed_upper",
        "ratio_of_ratios_throughput_lower",
    };
    if (ceilings_v.object.count() != ceiling_names.len) {
        return fail("derivation policy engineering ceilings are incomplete or unknown", .{});
    }
    var norm_ceilings: ObjectMap = .empty;
    for (ceiling_names) |name| {
        const cv = ceilings_v.object.get(name) orelse {
            return fail("derivation policy engineering ceilings are incomplete or unknown", .{});
        };
        const c_num = try finiteNumber(
            cv,
            try std.fmt.allocPrint(allocator, "derivation policy ceiling {s}", .{name}),
            true,
            false,
        );
        if (cushion > c_num) {
            return fail("derivation policy rounding cushion exceeds an engineering ceiling", .{});
        }
        try norm_ceilings.put(allocator, name, .{ .float = c_num });
    }

    const direct_limit = try finiteNumber(
        policy.get("direct_candidate_single_infrastructure_max_delta_fraction").?,
        "direct candidate single-infrastructure delta limit",
        true,
        false,
    );
    if (direct_limit != DIRECT_THREAD_DELTA_POLICY_LIMIT) {
        return fail("direct candidate single-infrastructure delta limit must be exactly the issue policy value 0.02", .{});
    }

    var out: ObjectMap = .empty;
    try out.put(allocator, "schema_version", .{ .integer = 1 });
    try out.put(allocator, "kind", .{ .string = DERIVATION_POLICY_KIND });
    try out.put(allocator, "minimum_reports_per_platform", .{ .integer = min_v.integer });
    try out.put(allocator, "rounding_cushion_log", .{ .float = cushion });
    try out.put(allocator, "engineering_policy_ceiling_log", .{ .object = norm_ceilings });
    try out.put(allocator, "direct_candidate_single_infrastructure_max_delta_fraction", .{ .float = direct_limit });
    return out;
}

pub fn deriveOneSidedThreshold(
    allocator: Allocator,
    values: []const f64,
    direction: []const u8,
    rounding_cushion_log: f64,
    engineering_ceiling_log: f64,
    label: []const u8,
) HarnessError!ObjectMap {
    if (!std.mem.eql(u8, direction, "lower") and !std.mem.eql(u8, direction, "upper")) {
        return fail("{s}: invalid threshold direction", .{label});
    }
    if (values.len == 0) {
        return fail("{s}: no training observations", .{label});
    }
    var logs = try allocator.alloc(f64, values.len);
    var input_arr = Array.init(allocator);
    var logs_arr = Array.init(allocator);
    var min_log: f64 = std.math.inf(f64);
    var max_log: f64 = -std.math.inf(f64);
    for (values, 0..) |v, idx| {
        if (!std.math.isFinite(v) or v <= 0.0) {
            return fail("{s}[{d}] must be a finite positive number", .{ label, idx });
        }
        const lv = @log(v);
        logs[idx] = lv;
        if (lv < min_log) min_log = lv;
        if (lv > max_log) max_log = lv;
        try input_arr.append(.{ .float = v });
        try logs_arr.append(.{ .float = lv });
    }
    const log_stats = benchmark_stats.summarize(allocator, logs) catch return fail("{s}: failed to summarize logs", .{label});
    const median_log = log_stats.median;

    var abs_devs = try allocator.alloc(f64, logs.len);
    for (logs, 0..) |lv, idx| {
        abs_devs[idx] = @abs(lv - median_log);
    }
    const mad_stats = benchmark_stats.summarize(allocator, abs_devs) catch return fail("{s}: failed to summarize MAD", .{label});
    const mad_log = mad_stats.median;
    const robust_radius = ROBUST_MAD_MULTIPLIER * MAD_SCALE * mad_log;

    const worst_observed_deviation_log = if (std.mem.eql(u8, direction, "lower"))
        @max(0.0, -min_log)
    else
        @max(0.0, max_log);
    const robust_deviation_log = if (std.mem.eql(u8, direction, "lower"))
        @max(0.0, -(median_log - robust_radius))
    else
        @max(0.0, median_log + robust_radius);

    const selected_source = if (worst_observed_deviation_log >= robust_deviation_log)
        "worst-observed"
    else
        "median-plus-six-scaled-mad";
    const selected_noise_bound_log = @max(worst_observed_deviation_log, robust_deviation_log);
    const final_bound_log = selected_noise_bound_log + rounding_cushion_log;
    if (final_bound_log > engineering_ceiling_log) {
        return fail("{s}: derived log bound {d} exceeds engineering policy ceiling {d}", .{ label, final_bound_log, engineering_ceiling_log });
    }
    const threshold = @exp(if (std.mem.eql(u8, direction, "lower")) -final_bound_log else final_bound_log);
    const lower_diag = median_log - robust_radius;
    const upper_diag = median_log + robust_radius;

    var outlier_arr = Array.init(allocator);
    for (logs, 0..) |lv, idx| {
        if (lv < lower_diag or lv > upper_diag) {
            try outlier_arr.append(.{ .integer = @intCast(idx) });
        }
    }

    var out: ObjectMap = .empty;
    try out.put(allocator, "direction", .{ .string = direction });
    try out.put(allocator, "observation_count", .{ .integer = @intCast(values.len) });
    try out.put(allocator, "input_ratios", .{ .array = input_arr });
    try out.put(allocator, "input_log_ratios", .{ .array = logs_arr });
    try out.put(allocator, "median_log_ratio", .{ .float = median_log });
    try out.put(allocator, "mad_log_ratio", .{ .float = mad_log });
    try out.put(allocator, "mad_scale", .{ .float = MAD_SCALE });
    try out.put(allocator, "mad_multiplier", .{ .float = ROBUST_MAD_MULTIPLIER });
    try out.put(allocator, "worst_observed_deviation_log", .{ .float = worst_observed_deviation_log });
    try out.put(allocator, "robust_deviation_log", .{ .float = robust_deviation_log });
    try out.put(allocator, "selected_source", .{ .string = selected_source });
    try out.put(allocator, "selected_noise_bound_log", .{ .float = selected_noise_bound_log });
    try out.put(allocator, "rounding_cushion_log", .{ .float = rounding_cushion_log });
    try out.put(allocator, "final_bound_log", .{ .float = final_bound_log });
    try out.put(allocator, "engineering_policy_ceiling_log", .{ .float = engineering_ceiling_log });
    try out.put(allocator, "threshold_ratio", .{ .float = threshold });
    try out.put(allocator, "diagnostic_outlier_indices", .{ .array = outlier_arr });
    try out.put(allocator, "dropped_observations", .{ .array = Array.init(allocator) });
    return out;
}

pub fn loadJsonObject(allocator: Allocator, io: std.Io, path: []const u8, label: []const u8) HarnessError!ObjectMap {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024)) catch {
        return fail("cannot read {s}: {s}", .{ label, path });
    };
    const parsed = std.json.parseFromSlice(Value, allocator, bytes, .{
        .duplicate_field_behavior = .@"error",
    }) catch |err| switch (err) {
        error.DuplicateField => return fail("{s} contains duplicate key", .{label}),
        else => return fail("{s} is not valid JSON", .{label}),
    };
    if (parsed.value != .object) return fail("{s} must be a JSON object", .{label});
    return parsed.value.object;
}

fn sampleMedian(allocator: Allocator, values_val: Value, label: []const u8) HarnessError!f64 {
    if (values_val != .array or values_val.array.items.len == 0) {
        return fail("{s} is missing raw per-report metric samples", .{label});
    }
    var nums = try allocator.alloc(f64, values_val.array.items.len);
    for (values_val.array.items, 0..) |v, idx| {
        nums[idx] = try finiteNumber(v, label, true, false);
    }
    const stats = benchmark_stats.summarize(allocator, nums) catch return fail("{s}: failed to compute median", .{label});
    return stats.median;
}

fn positionBalancedSampleEstimate(allocator: Allocator, values_val: Value, label: []const u8) HarnessError!f64 {
    if (values_val != .array) {
        return fail("{s} is missing raw per-report metric samples", .{label});
    }
    var nums = try allocator.alloc(f64, values_val.array.items.len);
    for (values_val.array.items, 0..) |v, idx| {
        nums[idx] = try finiteNumber(v, label, true, false);
    }
    const stats_obj = spec.positionBalancedRatioStats(allocator, nums, "samples") catch {
        return fail("{s}: invalid position-balanced ratio samples", .{label});
    };
    return spec.valueAsF64(stats_obj.get("median").?).?;
}

pub const MetricKey = struct {
    k0: []const u8,
    k1: []const u8,
    k2: []const u8,

    pub fn formatKey(self: MetricKey, allocator: Allocator) Allocator.Error![]const u8 {
        return std.fmt.allocPrint(allocator, "{s}\x00{s}\x00{s}", .{ self.k0, self.k1, self.k2 });
    }

    pub fn lessThan(_: void, a: MetricKey, b: MetricKey) bool {
        const c0 = std.mem.order(u8, a.k0, b.k0);
        if (c0 != .eq) return c0 == .lt;
        const c1 = std.mem.order(u8, a.k1, b.k1);
        if (c1 != .eq) return c1 == .lt;
        return std.mem.order(u8, a.k2, b.k2) == .lt;
    }
};

fn extractMetricKey(item: ObjectMap, kind: []const u8) HarnessError!MetricKey {
    const keys: [3][]const u8 = if (std.mem.eql(u8, kind, "comparison"))
        .{ "pair_key", "condition", "metric_kind" }
    else
        .{ "pair_key", "left", "right" };
    var out: [3][]const u8 = undefined;
    for (keys, 0..) |k, idx| {
        const v = item.get(k) orelse return fail("{s} metric identity is incomplete", .{kind});
        if (v != .string or v.string.len == 0) return fail("{s} metric identity is incomplete", .{kind});
        out[idx] = v.string;
    }
    return .{ .k0 = out[0], .k1 = out[1], .k2 = out[2] };
}

pub const DerivedDocuments = struct {
    budget: ObjectMap,
    evidence: ObjectMap,
    markdown: []const u8,
};

fn deriveCpuClassMetric(
    allocator: Allocator,
    platform: []const u8,
    cpu_class: []const u8,
    observations: []const ObjectMap,
    collection: []const u8,
    key: MetricKey,
    throughput_field: []const u8,
    elapsed_field: []const u8,
    throughput_ceiling: f64,
    elapsed_ceiling: f64,
    cushion: f64,
    throughput_threshold_name: []const u8,
    elapsed_threshold_name: []const u8,
    holdout_failures: *std.ArrayList([]const u8),
) HarnessError!struct { threshold: ObjectMap, evidence: ObjectMap } {
    var training_tp: std.ArrayList(f64) = .empty;
    var training_el: std.ArrayList(f64) = .empty;
    var training_rids = Array.init(allocator);
    var holdout_obs: std.ArrayList(ObjectMap) = .empty;

    const k_str = try key.formatKey(allocator);
    for (observations) |obs| {
        const part = obs.get("partition").?.string;
        const coll = obs.get("metrics").?.object.get(collection).?.array;
        var matched: ?ObjectMap = null;
        for (coll.items) |m_val| {
            const mk = try extractMetricKey(m_val.object, if (std.mem.eql(u8, collection, "comparisons")) "comparison" else "ratio-of-ratios");
            if (std.mem.eql(u8, try mk.formatKey(allocator), k_str)) {
                matched = m_val.object;
                break;
            }
        }
        const m_obj = matched orelse return fail("missing metric {s} in {s}", .{ key.k0, collection });
        if (std.mem.eql(u8, part, "training")) {
            const tp = try positionBalancedSampleEstimate(allocator, m_obj.get(throughput_field).?, throughput_field);
            const el = try positionBalancedSampleEstimate(allocator, m_obj.get(elapsed_field).?, elapsed_field);
            try training_tp.append(allocator, tp);
            try training_el.append(allocator, el);
            try training_rids.append(obs.get("run_id").?);
        } else {
            try holdout_obs.append(allocator, obs);
        }
    }

    const label = try std.fmt.allocPrint(allocator, "{s} CPU class '{s}' {s} ({s}, {s}, {s})", .{
        platform, cpu_class, collection, key.k0, key.k1, key.k2,
    });
    const tp_rule = try deriveOneSidedThreshold(
        allocator,
        training_tp.items,
        "lower",
        cushion,
        throughput_ceiling,
        try std.fmt.allocPrint(allocator, "{s} throughput", .{label}),
    );
    const el_rule = try deriveOneSidedThreshold(
        allocator,
        training_el.items,
        "upper",
        cushion,
        elapsed_ceiling,
        try std.fmt.allocPrint(allocator, "{s} elapsed", .{label}),
    );
    const tp_thresh = tp_rule.get("threshold_ratio").?.float;
    const el_thresh = el_rule.get("threshold_ratio").?.float;

    var threshold: ObjectMap = .empty;
    try threshold.put(allocator, throughput_threshold_name, .{ .float = tp_thresh });
    try threshold.put(allocator, elapsed_threshold_name, .{ .float = el_thresh });

    var holdout_results = Array.init(allocator);
    for (holdout_obs.items) |obs| {
        const coll = obs.get("metrics").?.object.get(collection).?.array;
        var matched: ?ObjectMap = null;
        for (coll.items) |m_val| {
            const mk = try extractMetricKey(m_val.object, if (std.mem.eql(u8, collection, "comparisons")) "comparison" else "ratio-of-ratios");
            if (std.mem.eql(u8, try mk.formatKey(allocator), k_str)) {
                matched = m_val.object;
                break;
            }
        }
        const m_obj = matched.?;
        const tp = try positionBalancedSampleEstimate(allocator, m_obj.get(throughput_field).?, throughput_field);
        const el = try positionBalancedSampleEstimate(allocator, m_obj.get(elapsed_field).?, elapsed_field);
        const passed = (tp >= tp_thresh and el <= el_thresh);
        var hr: ObjectMap = .empty;
        try hr.put(allocator, "run_id", obs.get("run_id").?);
        try hr.put(allocator, "throughput_ratio", .{ .float = tp });
        try hr.put(allocator, "elapsed_ratio", .{ .float = el });
        try hr.put(allocator, "passed", .{ .bool = passed });
        try holdout_results.append(.{ .object = hr });
        if (!passed) {
            try holdout_failures.append(allocator, try std.fmt.allocPrint(
                allocator,
                "{s} run {s} failed: throughput={d} (min={d}), elapsed={d} (max={d})",
                .{ label, obs.get("run_id").?.string, tp, tp_thresh, el, el_thresh },
            ));
        }
    }

    var ev: ObjectMap = .empty;
    try ev.put(allocator, "training_run_ids", .{ .array = training_rids });
    try ev.put(allocator, "throughput_rule", .{ .object = tp_rule });
    try ev.put(allocator, "elapsed_rule", .{ .object = el_rule });
    try ev.put(allocator, "holdout", .{ .array = holdout_results });
    return .{ .threshold = threshold, .evidence = ev };
}

pub fn deriveBudgetDocuments(
    allocator: Allocator,
    cohort: ObjectMap,
    policy_doc: ObjectMap,
) HarnessError!DerivedDocuments {
    const policy = try validateDerivationPolicy(allocator, policy_doc);
    const sv = cohort.get("schema_version") orelse return fail("derivation requires a validated authoritative schema-v4 paired cohort", .{});
    const kind = cohort.get("kind") orelse return fail("derivation requires a validated authoritative schema-v4 paired cohort", .{});
    const auth = cohort.get("authoritative") orelse return fail("derivation requires a validated authoritative schema-v4 paired cohort", .{});
    if (sv != .integer or sv.integer != spec.REPORT_SCHEMA_VERSION or
        kind != .string or !std.mem.eql(u8, kind.string, "wasi-thread-paired-cohort") or
        auth != .bool or !auth.bool)
    {
        return fail("derivation requires a validated authoritative schema-v4 paired cohort", .{});
    }
    const excl = cohort.get("excluded_observations") orelse return fail("derivation forbids excluded observations or outlier drops", .{});
    if (excl != .array or excl.array.items.len != 0) {
        return fail("derivation forbids excluded observations or outlier drops", .{});
    }
    const identity_v = cohort.get("identity") orelse return fail("validated cohort is missing required provenance", .{});
    const dispatch_v = cohort.get("dispatch") orelse return fail("validated cohort is missing required provenance", .{});
    const split_v = cohort.get("split") orelse return fail("validated cohort is missing required provenance", .{});
    const platforms_v = cohort.get("platforms") orelse return fail("validated cohort is missing required provenance", .{});
    const observations_v = cohort.get("observations") orelse return fail("validated cohort is missing required provenance", .{});
    if (identity_v != .object or dispatch_v != .object or split_v != .object or platforms_v != .object or observations_v != .array) {
        return fail("validated cohort is missing required provenance", .{});
    }
    const identity = identity_v.object;
    const dispatch = dispatch_v.object;
    const platforms = platforms_v.object;
    const observations = observations_v.array.items;

    if (!std.mem.eql(u8, identity.get("profile").?.string, "authoritative")) {
        return fail("derivation rejects smoke or non-authoritative cohorts", .{});
    }
    if (!std.mem.eql(u8, identity.get("comparison_purpose").?.string, "noise-calibration")) {
        return fail("derivation requires noise-calibration purpose", .{});
    }
    const accepted_cpu_classes = try validateCpuClassMapping(
        allocator,
        dispatch.get("accepted_cpu_classes") orelse return fail("validated cohort accepted_cpu_classes", .{}),
        "validated cohort accepted_cpu_classes",
    );
    const minimum_reports = policy.get("minimum_reports_per_platform").?.integer;
    const cushion = policy.get("rounding_cushion_log").?.float;
    const ceilings = policy.get("engineering_policy_ceiling_log").?.object;
    const direct_limit = policy.get("direct_candidate_single_infrastructure_max_delta_fraction").?.float;

    if (observations.len == 0) return fail("validated cohort has no observations", .{});
    const first_metrics = observations[0].object.get("metrics").?.object;

    var comp_keys: std.ArrayList(MetricKey) = .empty;
    for (first_metrics.get("comparisons").?.array.items) |m_val| {
        try comp_keys.append(allocator, try extractMetricKey(m_val.object, "comparison"));
    }
    std.mem.sort(MetricKey, comp_keys.items, {}, MetricKey.lessThan);

    var ror_keys: std.ArrayList(MetricKey) = .empty;
    for (first_metrics.get("ratio_of_ratios").?.array.items) |m_val| {
        try ror_keys.append(allocator, try extractMetricKey(m_val.object, "ratio-of-ratios"));
    }
    std.mem.sort(MetricKey, ror_keys.items, {}, MetricKey.lessThan);

    var direct_keys: std.ArrayList(MetricKey) = .empty;
    for (first_metrics.get("direct_candidate_single_infrastructure").?.array.items) |m_val| {
        try direct_keys.append(allocator, try extractMetricKey(m_val.object, "direct candidate single-infrastructure"));
    }
    std.mem.sort(MetricKey, direct_keys.items, {}, MetricKey.lessThan);

    var platform_budgets: ObjectMap = .empty;
    var platform_evidence: ObjectMap = .empty;
    var report_count_by_platform: ObjectMap = .empty;
    var report_count_by_cpu_class: ObjectMap = .empty;
    var partition_report_count_by_cpu_class: ObjectMap = .empty;
    var holdout_failures: std.ArrayList([]const u8) = .empty;
    var direct_failures: std.ArrayList([]const u8) = .empty;

    for (DEFAULT_PLATFORMS) |platform| {
        var selected: std.ArrayList(ObjectMap) = .empty;
        for (observations) |obs_v| {
            if (std.mem.eql(u8, obs_v.object.get("platform").?.string, platform)) {
                try selected.append(allocator, obs_v.object);
            }
        }
        try report_count_by_platform.put(allocator, platform, .{ .integer = @intCast(selected.items.len) });

        const acc_classes = accepted_cpu_classes.get(platform).?;
        var class_counts_obj: ObjectMap = .empty;
        var class_part_counts_obj: ObjectMap = .empty;
        var class_obs_map: std.StringHashMapUnmanaged([]const ObjectMap) = .empty;

        for (acc_classes) |cpu_class| {
            var c_obs: std.ArrayList(ObjectMap) = .empty;
            var tr_cnt: i64 = 0;
            var ho_cnt: i64 = 0;
            for (selected.items) |obs| {
                if (std.mem.eql(u8, obs.get("cpu_class").?.string, cpu_class)) {
                    try c_obs.append(allocator, obs);
                    if (std.mem.eql(u8, obs.get("partition").?.string, "training")) {
                        tr_cnt += 1;
                    } else {
                        ho_cnt += 1;
                    }
                }
            }
            const total_cnt: i64 = @intCast(c_obs.items.len);
            if (total_cnt < minimum_reports or
                tr_cnt < MINIMUM_TRAINING_REPORTS_PER_CPU_CLASS or
                ho_cnt < MINIMUM_HOLDOUT_REPORTS_PER_CPU_CLASS)
            {
                return fail(
                    "validated cohort {s} CPU class '{s}' is undersampled: reports={d} (required >= {d}), training={d} (required >= {d}), holdout={d} (required >= {d})",
                    .{ platform, cpu_class, total_cnt, minimum_reports, tr_cnt, MINIMUM_TRAINING_REPORTS_PER_CPU_CLASS, ho_cnt, MINIMUM_HOLDOUT_REPORTS_PER_CPU_CLASS },
                );
            }
            try class_counts_obj.put(allocator, cpu_class, .{ .integer = total_cnt });
            var part_obj: ObjectMap = .empty;
            try part_obj.put(allocator, "training", .{ .integer = tr_cnt });
            try part_obj.put(allocator, "holdout", .{ .integer = ho_cnt });
            try class_part_counts_obj.put(allocator, cpu_class, .{ .object = part_obj });
            try class_obs_map.put(allocator, cpu_class, try c_obs.toOwnedSlice(allocator));
        }
        try report_count_by_cpu_class.put(allocator, platform, .{ .object = class_counts_obj });
        try partition_report_count_by_cpu_class.put(allocator, platform, .{ .object = class_part_counts_obj });

        var comp_budget = Array.init(allocator);
        var comp_ev = Array.init(allocator);
        for (comp_keys.items) |mk| {
            var min_tp: f64 = std.math.inf(f64);
            var max_el: f64 = -std.math.inf(f64);
            var cpu_classes_ev: ObjectMap = .empty;
            for (acc_classes) |cpu_class| {
                const res = try deriveCpuClassMetric(
                    allocator,
                    platform,
                    cpu_class,
                    class_obs_map.get(cpu_class).?,
                    "comparisons",
                    mk,
                    "throughput_candidate_over_baseline",
                    "elapsed_candidate_over_baseline",
                    ceilings.get("comparison_throughput_lower").?.float,
                    ceilings.get("comparison_elapsed_upper").?.float,
                    cushion,
                    "min_candidate_over_baseline_throughput_ratio",
                    "max_candidate_over_baseline_elapsed_ratio",
                    &holdout_failures,
                );
                const tp_t = res.threshold.get("min_candidate_over_baseline_throughput_ratio").?.float;
                const el_t = res.threshold.get("max_candidate_over_baseline_elapsed_ratio").?.float;
                if (tp_t < min_tp) min_tp = tp_t;
                if (el_t > max_el) max_el = el_t;
                try cpu_classes_ev.put(allocator, cpu_class, .{ .object = res.evidence });
            }
            var thresh: ObjectMap = .empty;
            try thresh.put(allocator, "pair_key", .{ .string = mk.k0 });
            try thresh.put(allocator, "condition", .{ .string = mk.k1 });
            try thresh.put(allocator, "metric_kind", .{ .string = mk.k2 });
            try thresh.put(allocator, "min_candidate_over_baseline_throughput_ratio", .{ .float = min_tp });
            try thresh.put(allocator, "max_candidate_over_baseline_elapsed_ratio", .{ .float = max_el });
            try comp_budget.append(.{ .object = thresh });

            var ev_entry: ObjectMap = .empty;
            try ev_entry.put(allocator, "pair_key", .{ .string = mk.k0 });
            try ev_entry.put(allocator, "condition", .{ .string = mk.k1 });
            try ev_entry.put(allocator, "metric_kind", .{ .string = mk.k2 });
            try ev_entry.put(allocator, "release_threshold", .{ .object = thresh });
            try ev_entry.put(allocator, "cpu_classes", .{ .object = cpu_classes_ev });
            try comp_ev.append(.{ .object = ev_entry });
        }

        var ror_budget = Array.init(allocator);
        var ror_ev = Array.init(allocator);
        for (ror_keys.items) |mk| {
            var min_tp: f64 = std.math.inf(f64);
            var max_el: f64 = -std.math.inf(f64);
            var cpu_classes_ev: ObjectMap = .empty;
            for (acc_classes) |cpu_class| {
                const res = try deriveCpuClassMetric(
                    allocator,
                    platform,
                    cpu_class,
                    class_obs_map.get(cpu_class).?,
                    "ratio_of_ratios",
                    mk,
                    "throughput_ratio_of_ratios",
                    "elapsed_ratio_of_ratios",
                    ceilings.get("ratio_of_ratios_throughput_lower").?.float,
                    ceilings.get("ratio_of_ratios_elapsed_upper").?.float,
                    cushion,
                    "min_candidate_over_baseline_throughput_ratio_of_ratios",
                    "max_candidate_over_baseline_elapsed_ratio_of_ratios",
                    &holdout_failures,
                );
                const tp_t = res.threshold.get("min_candidate_over_baseline_throughput_ratio_of_ratios").?.float;
                const el_t = res.threshold.get("max_candidate_over_baseline_elapsed_ratio_of_ratios").?.float;
                if (tp_t < min_tp) min_tp = tp_t;
                if (el_t > max_el) max_el = el_t;
                try cpu_classes_ev.put(allocator, cpu_class, .{ .object = res.evidence });
            }
            var thresh: ObjectMap = .empty;
            try thresh.put(allocator, "pair_key", .{ .string = mk.k0 });
            try thresh.put(allocator, "left", .{ .string = mk.k1 });
            try thresh.put(allocator, "right", .{ .string = mk.k2 });
            try thresh.put(allocator, "min_candidate_over_baseline_throughput_ratio_of_ratios", .{ .float = min_tp });
            try thresh.put(allocator, "max_candidate_over_baseline_elapsed_ratio_of_ratios", .{ .float = max_el });
            try ror_budget.append(.{ .object = thresh });

            var ev_entry: ObjectMap = .empty;
            try ev_entry.put(allocator, "pair_key", .{ .string = mk.k0 });
            try ev_entry.put(allocator, "left", .{ .string = mk.k1 });
            try ev_entry.put(allocator, "right", .{ .string = mk.k2 });
            try ev_entry.put(allocator, "release_threshold", .{ .object = thresh });
            try ev_entry.put(allocator, "cpu_classes", .{ .object = cpu_classes_ev });
            try ror_ev.append(.{ .object = ev_entry });
        }

        var direct_ev = Array.init(allocator);
        for (direct_keys.items) |mk| {
            const k_str = try mk.formatKey(allocator);
            var results = Array.init(allocator);
            for (selected.items) |obs| {
                const coll = obs.get("metrics").?.object.get("direct_candidate_single_infrastructure").?.array;
                var matched: ?ObjectMap = null;
                for (coll.items) |m_val| {
                    const item_mk = try extractMetricKey(m_val.object, "direct candidate single-infrastructure");
                    if (std.mem.eql(u8, try item_mk.formatKey(allocator), k_str)) {
                        matched = m_val.object;
                        break;
                    }
                }
                const m_obj = matched orelse return fail("missing direct candidate metric {s}", .{mk.k0});
                const tp = try sampleMedian(allocator, m_obj.get("throughput_right_over_left").?, "throughput_right_over_left");
                const el = try sampleMedian(allocator, m_obj.get("elapsed_right_over_left").?, "elapsed_right_over_left");
                const tp_delta = @abs(tp - 1.0);
                const el_delta = @abs(el - 1.0);
                const passed = (tp_delta < direct_limit and el_delta < direct_limit);
                var r_obj: ObjectMap = .empty;
                try r_obj.put(allocator, "run_id", obs.get("run_id").?);
                try r_obj.put(allocator, "partition", obs.get("partition").?);
                try r_obj.put(allocator, "cpu_class", obs.get("cpu_class").?);
                try r_obj.put(allocator, "throughput_ratio", .{ .float = tp });
                try r_obj.put(allocator, "elapsed_ratio", .{ .float = el });
                try r_obj.put(allocator, "absolute_throughput_delta_fraction", .{ .float = tp_delta });
                try r_obj.put(allocator, "absolute_elapsed_delta_fraction", .{ .float = el_delta });
                try r_obj.put(allocator, "passed", .{ .bool = passed });
                try results.append(.{ .object = r_obj });
                if (!passed) {
                    try direct_failures.append(allocator, try std.fmt.allocPrint(
                        allocator,
                        "{s} run {s} direct candidate {s} threads-enabled/disabled delta failed: throughput_delta={d}, elapsed_delta={d}, required both < {d}",
                        .{ platform, obs.get("run_id").?.string, mk.k0, tp_delta, el_delta, direct_limit },
                    ));
                }
            }
            var d_entry: ObjectMap = .empty;
            try d_entry.put(allocator, "pair_key", .{ .string = mk.k0 });
            try d_entry.put(allocator, "left", .{ .string = mk.k1 });
            try d_entry.put(allocator, "right", .{ .string = mk.k2 });
            try d_entry.put(allocator, "policy", .{ .string = "absolute median ratio delta must be strictly below 2%" });
            try d_entry.put(allocator, "limit_fraction", .{ .float = direct_limit });
            try d_entry.put(allocator, "observations", .{ .array = results });
            try direct_ev.append(.{ .object = d_entry });
        }

        const cp_machine = if (std.mem.eql(u8, platform, "ubuntu-22.04-x86_64")) "x86_64" else "aarch64";
        const plat_info = platforms.get(platform).?.object;
        const runner_target = dispatch.get("runner_target").?.string;
        const self_hosted_x86 = std.mem.eql(u8, runner_target, "trusted-calibration") and
            std.mem.eql(u8, platform, TRUSTED_X86_PLATFORM) and
            (plat_info.contains("trusted_runner_name") or plat_info.contains("host_fingerprint_sha256"));
        var acc_arr = Array.init(allocator);
        for (acc_classes) |c| try acc_arr.append(.{ .string = c });

        var pb: ObjectMap = .empty;
        try pb.put(allocator, "host_system", .{ .string = "Linux" });
        try pb.put(allocator, "host_machine", .{ .string = cp_machine });
        try pb.put(allocator, "runner_environment", .{ .string = if (self_hosted_x86) "self-hosted" else "github-hosted" });
        try pb.put(allocator, "accepted_cpu_classes", .{ .array = acc_arr });
        try pb.put(allocator, "comparisons", .{ .array = comp_budget });
        try pb.put(allocator, "ratio_of_ratios", .{ .array = ror_budget });
        try platform_budgets.put(allocator, platform, .{ .object = pb });

        var tr_rep_cnt: i64 = 0;
        var ho_rep_cnt: i64 = 0;
        for (selected.items) |obs| {
            if (std.mem.eql(u8, obs.get("partition").?.string, "training")) tr_rep_cnt += 1 else ho_rep_cnt += 1;
        }
        var pev: ObjectMap = .empty;
        try pev.put(allocator, "report_count", .{ .integer = @intCast(selected.items.len) });
        try pev.put(allocator, "training_report_count", .{ .integer = tr_rep_cnt });
        try pev.put(allocator, "holdout_report_count", .{ .integer = ho_rep_cnt });
        try pev.put(allocator, "cpu_class_report_counts", .{ .object = class_counts_obj });
        try pev.put(allocator, "cpu_class_partition_counts", .{ .object = class_part_counts_obj });
        try pev.put(allocator, "host_fingerprint_distribution", plat_info.get("host_fingerprint_distribution") orelse .{ .object = .empty });
        try pev.put(allocator, "cpu_distribution", plat_info.get("cpu_distribution") orelse .{ .object = .empty });
        try pev.put(allocator, "runner_image_distribution", plat_info.get("runner_image_distribution") orelse .{ .object = .empty });
        try pev.put(allocator, "selected_iteration_distribution", plat_info.get("selected_iteration_distribution") orelse .{ .object = .empty });
        try pev.put(allocator, "pilot_elapsed_ns_summary", plat_info.get("pilot_elapsed_ns_summary") orelse .{ .object = .empty });
        try pev.put(allocator, "comparisons", .{ .array = comp_ev });
        try pev.put(allocator, "ratio_of_ratios", .{ .array = ror_ev });
        try pev.put(allocator, "direct_candidate_single_infrastructure", .{ .array = direct_ev });
        try platform_evidence.put(allocator, platform, .{ .object = pev });
    }

    if (direct_failures.items.len > 0) {
        return fail("direct candidate single-infrastructure policy failed:\n{s}", .{
            std.mem.join(allocator, "\n", direct_failures.items) catch "direct failure",
        });
    }
    if (holdout_failures.items.len > 0) {
        return fail("untouched holdout validation failed:\n{s}", .{
            std.mem.join(allocator, "\n", holdout_failures.items) catch "holdout failure",
        });
    }

    const base_id = identity.get("baseline").?.object;
    var cal_rev: ObjectMap = .empty;
    try cal_rev.put(allocator, "commit", base_id.get("commit").?);
    try cal_rev.put(allocator, "build_source_sha256", base_id.get("build_source_sha256").?);

    var req_plats = Array.init(allocator);
    for (DEFAULT_PLATFORMS) |p| try req_plats.append(.{ .string = p });

    var cal_req: ObjectMap = .empty;
    try cal_req.put(allocator, "minimum_reports_per_platform", .{ .integer = minimum_reports });
    try cal_req.put(allocator, "minimum_reports_per_cpu_class", .{ .integer = minimum_reports });
    try cal_req.put(allocator, "minimum_training_reports_per_cpu_class", .{ .integer = MINIMUM_TRAINING_REPORTS_PER_CPU_CLASS });
    try cal_req.put(allocator, "minimum_holdout_reports_per_cpu_class", .{ .integer = MINIMUM_HOLDOUT_REPORTS_PER_CPU_CLASS });
    try cal_req.put(allocator, "required_profile", .{ .string = "authoritative" });
    try cal_req.put(allocator, "required_platforms", .{ .array = req_plats });

    var cal_prov: ObjectMap = .empty;
    try cal_prov.put(allocator, "baseline_revision", .{ .object = cal_rev });
    try cal_prov.put(allocator, "candidate_revision", .{ .object = cal_rev });
    try cal_prov.put(allocator, "comparison_purpose", .{ .string = "noise-calibration" });
    try cal_prov.put(allocator, "fixture_set_sha256", identity.get("fixture_set_sha256").?);
    try cal_prov.put(allocator, "measurement_plan_version", identity.get("measurement_plan_version").?);
    try cal_prov.put(allocator, "measurement_plan_sha256", identity.get("measurement_plan_sha256").?);
    try cal_prov.put(allocator, "profile", .{ .string = "authoritative" });
    try cal_prov.put(allocator, "report_count_by_platform", .{ .object = report_count_by_platform });
    try cal_prov.put(allocator, "accepted_cpu_classes", dispatch.get("accepted_cpu_classes").?);
    try cal_prov.put(allocator, "report_count_by_cpu_class", .{ .object = report_count_by_cpu_class });
    try cal_prov.put(allocator, "partition_report_count_by_cpu_class", .{ .object = partition_report_count_by_cpu_class });

    var budget: ObjectMap = .empty;
    try budget.put(allocator, "schema_version", .{ .integer = spec.REPORT_SCHEMA_VERSION });
    try budget.put(allocator, "kind", .{ .string = "wasi-thread-benchmark-budget" });
    try budget.put(allocator, "calibrated", .{ .bool = true });
    try budget.put(allocator, "enforcement", .{ .bool = false });
    try budget.put(allocator, "calibration_requirements", .{ .object = cal_req });
    try budget.put(allocator, "calibration_provenance", .{ .object = cal_prov });
    try budget.put(allocator, "platforms", .{ .object = platform_budgets });

    const source_cohort_sha_arr = benchmark_schema.cacheKey(allocator, .{ .object = cohort }) catch return error.BenchmarkDataError;
    const source_cohort_sha = try allocator.dupe(u8, &source_cohort_sha_arr);
    const candidate_budget_sha_arr = benchmark_schema.cacheKey(allocator, .{ .object = budget }) catch return error.BenchmarkDataError;
    const candidate_budget_sha = try allocator.dupe(u8, &candidate_budget_sha_arr);

    var method_obj: ObjectMap = .empty;
    try method_obj.put(allocator, "domain", .{ .string = "natural log of each per-report position-balanced estimate, derived independently within each exact CPU-model class" });
    try method_obj.put(allocator, "robust_rule", .{ .string = "For each one-sided metric, select the larger of the worst observed training deviation and median +/- 6*1.4826*MAD, then add the predeclared log rounding cushion." });
    try method_obj.put(allocator, "ceiling_rule", .{ .string = "Fail when the selected bound plus cushion exceeds its separately declared engineering policy ceiling." });
    try method_obj.put(allocator, "holdout_rule", .{ .string = "Every untouched holdout report must satisfy every threshold derived for its exact CPU-model class." });
    try method_obj.put(allocator, "release_rule", .{ .string = "Release throughput minima use the lowest class threshold; release elapsed maxima use the highest class threshold. Observations from different CPU classes are never pooled." });
    try method_obj.put(allocator, "outlier_rule", .{ .string = "Outliers are diagnostic only; no observation is dropped." });

    var evidence: ObjectMap = .empty;
    try evidence.put(allocator, "schema_version", .{ .integer = 1 });
    try evidence.put(allocator, "kind", .{ .string = DERIVATION_EVIDENCE_KIND });
    try evidence.put(allocator, "source_cohort_sha256", .{ .string = source_cohort_sha });
    try evidence.put(allocator, "candidate_budget_sha256", .{ .string = candidate_budget_sha });
    try evidence.put(allocator, "enforcement_enabled", .{ .bool = false });
    try evidence.put(allocator, "method", .{ .object = method_obj });
    try evidence.put(allocator, "policy", .{ .object = policy });
    try evidence.put(allocator, "calibration_provenance", .{ .object = cal_prov });
    try evidence.put(allocator, "platforms", .{ .object = platform_evidence });
    try evidence.put(allocator, "excluded_observations", .{ .array = Array.init(allocator) });
    try evidence.put(allocator, "holdout_status", .{ .string = "passed" });
    try evidence.put(allocator, "direct_candidate_single_infrastructure_status", .{ .string = "passed" });

    var md: std.ArrayList(u8) = .empty;
    try md.appendSlice(allocator, "# WASI thread budget derivation evidence\n\n");
    try md.appendSlice(allocator, "- Enforcement: **disabled** (a later proof/final PR must enable it).\n");
    try md.print(allocator, "- Source cohort SHA-256: `{s}`\n", .{source_cohort_sha});
    try md.print(allocator, "- Candidate budget SHA-256: `{s}`\n", .{candidate_budget_sha});
    try md.appendSlice(allocator, "- Formula: natural-log ratios; max(worst observed training deviation, median ± 6×1.4826×MAD), plus the predeclared log rounding cushion.\n");
    try md.appendSlice(allocator, "- Outliers: diagnostic only; zero observations dropped.\n");
    try md.appendSlice(allocator, "- Holdout: all untouched holdout reports passed.\n");
    try md.appendSlice(allocator, "- Direct candidate threads-enabled/disabled: every report on both architectures has absolute median throughput and elapsed ratio delta <2%.\n\n");
    try md.appendSlice(allocator, "## Platforms\n\n");
    try md.appendSlice(allocator, "| Platform | Reports | Training | Holdout | Comparisons | Ratio-of-ratios |\n");
    try md.appendSlice(allocator, "|---|---:|---:|---:|---:|---:|\n");
    for (DEFAULT_PLATFORMS) |platform| {
        const pev = platform_evidence.get(platform).?.object;
        try md.print(allocator, "| {s} | {d} | {d} | {d} | {d} | {d} |\n", .{
            platform,
            pev.get("report_count").?.integer,
            pev.get("training_report_count").?.integer,
            pev.get("holdout_report_count").?.integer,
            pev.get("comparisons").?.array.items.len,
            pev.get("ratio_of_ratios").?.array.items.len,
        });
    }
    try md.appendSlice(allocator, "\nThe machine-readable evidence JSON contains every formula input, intermediate bound, diagnostic outlier index, holdout result, workflow run ID, revision identity, and host/CPU/image distribution.\n");

    return .{
        .budget = budget,
        .evidence = evidence,
        .markdown = try md.toOwnedSlice(allocator),
    };
}

pub const DocEntry = struct {
    path: []const u8,
    doc: ObjectMap,
};

pub fn platformIdForHost(host: ObjectMap) HarnessError![]const u8 {
    const sys = host.get("system") orelse return fail("host is missing system", .{});
    const mach = host.get("machine") orelse return fail("host is missing machine", .{});
    if (sys != .string or mach != .string) return fail("host system/machine must be strings", .{});
    for (DEFAULT_PLATFORMS) |plat| {
        const cp = spec.canonicalPlatformHost(plat).?;
        if (std.mem.eql(u8, sys.string, cp.system) and std.mem.eql(u8, mach.string, cp.machine)) {
            return plat;
        }
    }
    return fail("unsupported host platform ({s}, {s})", .{ sys.string, mach.string });
}

pub fn validatePairedDocuments(
    allocator: Allocator,
    io: std.Io,
    documents: []const DocEntry,
    required_platforms: []const []const u8,
    dispatch_state: ObjectMap,
    allow_partial: bool,
) HarnessError!ObjectMap {
    const expected = try validateDispatchState(allocator, dispatch_state, required_platforms, allow_partial);
    const expected_report_count = @as(usize, @intCast(expected.run_by_id.count())) * required_platforms.len;
    if (documents.len != expected_report_count) {
        return fail("paired cohort requires exactly {d} reports ({d} runs x {d} platforms), found {d}", .{
            expected_report_count, expected.run_by_id.count(), required_platforms.len, documents.len,
        });
    }

    var observations = Array.init(allocator);
    for (documents) |entry| {
        const path = entry.path;
        const doc = entry.doc;
        report.validateReport(allocator, .{ .object = doc }) catch |err| {
            return fail("{s}: {s}", .{ path, @errorName(err) });
        };
        const metadata = doc.get("metadata").?.object;
        const plan = doc.get("plan").?.object;
        const host = metadata.get("host").?.object;
        const platform_id = try platformIdForHost(host);
        const run_id = host.get("workflow_run_id").?.string;
        const run_obj = expected.run_by_id.get(run_id) orelse {
            return fail("{s}: workflow run {s} is not in the dispatch manifest", .{ path, run_id });
        };
        const revisions = metadata.get("revisions").?.object;
        const fingerprint = host.get("host_fingerprint").?.object.get("sha256").?.string;
        const cpu = host.get("host_fingerprint").?.object.get("fields").?.object.get("cpu").?.string;
        const host_pair_id = metadata.get("host_pair").?.object.get("id").?.string;

        var id_obj: ObjectMap = .empty;
        try id_obj.put(allocator, "baseline", revisions.get("baseline").?);
        try id_obj.put(allocator, "candidate", revisions.get("candidate").?);
        try id_obj.put(allocator, "fixture_set_sha256", metadata.get("fixture_set_sha256").?);
        try id_obj.put(allocator, "plan_sha256", metadata.get("plan_sha256").?);
        try id_obj.put(allocator, "measurement_plan_version", metadata.get("measurement_plan_version").?);
        try id_obj.put(allocator, "measurement_plan_sha256", metadata.get("measurement_plan_sha256").?);
        try id_obj.put(allocator, "profile", plan.get("profile").?);
        try id_obj.put(allocator, "comparison_purpose", plan.get("comparison_purpose").?);

        var obs: ObjectMap = .empty;
        try obs.put(allocator, "sequence", run_obj.get("sequence").?);
        try obs.put(allocator, "partition", run_obj.get("partition").?);
        try obs.put(allocator, "run_id", .{ .string = run_id });
        try obs.put(allocator, "platform", .{ .string = platform_id });
        try obs.put(allocator, "path", .{ .string = path });
        try obs.put(allocator, "host_pair_id", .{ .string = host_pair_id });
        try obs.put(allocator, "host_fingerprint_sha256", .{ .string = fingerprint });
        try obs.put(allocator, "cpu_class", .{ .string = cpu });
        try obs.put(allocator, "records", .{ .integer = @intCast(doc.get("records").?.array.items.len) });
        const doc_sha_arr = benchmark_schema.cacheKey(allocator, .{ .object = doc }) catch return error.BenchmarkDataError;
        const doc_sha = try allocator.dupe(u8, &doc_sha_arr);
        try obs.put(allocator, "report_sha256", .{ .string = doc_sha });
        try obs.put(allocator, "identity", .{ .object = id_obj });
        try obs.put(allocator, "plan", .{ .object = plan });
        try obs.put(allocator, "metrics", .{ .object = try reportCalibrationMetrics(allocator, doc) });
        try observations.append(.{ .object = obs });
    }

    const first_doc = documents[0].doc;
    const first_meta = first_doc.get("metadata").?.object;
    const first_revs = first_meta.get("revisions").?.object;
    var identity: ObjectMap = .empty;
    try identity.put(allocator, "baseline", first_revs.get("baseline").?);
    try identity.put(allocator, "candidate", first_revs.get("candidate").?);
    try identity.put(allocator, "fixture_set_sha256", first_meta.get("fixture_set_sha256").?);
    try identity.put(allocator, "measurement_plan_version", first_meta.get("measurement_plan_version").?);
    try identity.put(allocator, "measurement_plan_sha256", first_meta.get("measurement_plan_sha256").?);
    try identity.put(allocator, "profile", .{ .string = expected.profile });
    try identity.put(allocator, "comparison_purpose", .{ .string = expected.purpose });
    try identity.put(allocator, "warmups", .{ .integer = expected.warmups });
    try identity.put(allocator, "samples", .{ .integer = expected.samples });

    var platforms_obj: ObjectMap = .empty;
    for (required_platforms) |plat| {
        var p_obj: ObjectMap = .empty;
        try p_obj.put(allocator, "reports", .{ .integer = @intCast(expected.run_by_id.count()) });
        try platforms_obj.put(allocator, plat, .{ .object = p_obj });
    }

    var out: ObjectMap = .empty;
    try out.put(allocator, "schema_version", .{ .integer = spec.REPORT_SCHEMA_VERSION });
    try out.put(allocator, "kind", .{ .string = "wasi-thread-paired-cohort" });
    try out.put(allocator, "authoritative", .{ .bool = @as(i64, @intCast(expected.run_by_id.count())) == expected.requested_runs });
    try out.put(allocator, "validated_at", .{ .string = try benchmark_schema.collectedAt(allocator, io) });
    try out.put(allocator, "dispatch", .{ .object = dispatch_state });
    try out.put(allocator, "identity", .{ .object = identity });
    try out.put(allocator, "split", .{ .object = expected.split });
    try out.put(allocator, "platforms", .{ .object = platforms_obj });
    try out.put(allocator, "observations", .{ .array = observations });
    try out.put(allocator, "excluded_observations", .{ .array = Array.init(allocator) });
    return out;
}

pub fn downloadCohortArtifacts(
    allocator: Allocator,
    io: std.Io,
    dispatch_state_path: []const u8,
    output_dir: []const u8,
    timeout_seconds: f64,
    allow_partial: bool,
) HarnessError!ObjectMap {
    const dispatch_state = try loadJsonObject(allocator, io, dispatch_state_path, "dispatch state");
    const repo_v = dispatch_state.get("repository") orelse return fail("dispatch state missing repository", .{});
    if (repo_v != .string or repo_v.string.len == 0) return fail("dispatch state missing repository", .{});
    const req_runs_v = dispatch_state.get("requested_runs") orelse dispatch_state.get("runs_requested") orelse return fail("dispatch state is missing valid requested_runs", .{});
    if (req_runs_v != .integer or req_runs_v.integer < 1) return fail("dispatch state is missing valid requested_runs", .{});
    const runs_v = dispatch_state.get("runs") orelse return fail("dispatch state has no runs to download", .{});
    if (runs_v != .array or runs_v.array.items.len == 0) return fail("dispatch state has no runs to download", .{});
    const n_runs: i64 = @intCast(runs_v.array.items.len);
    if (!allow_partial and n_runs != req_runs_v.integer) {
        return fail("dispatch state has {d} runs, expected {d}", .{ n_runs, req_runs_v.integer });
    }
    if (n_runs > req_runs_v.integer) {
        return fail("dispatch state has {d} runs, exceeding {d}", .{ n_runs, req_runs_v.integer });
    }
    if (!std.math.isFinite(timeout_seconds) or timeout_seconds <= 0.0) {
        return fail("--timeout-seconds must be positive", .{});
    }
    std.Io.Dir.cwd().createDirPath(io, output_dir) catch return fail("cannot create output dir: {s}", .{output_dir});
    var downloaded = Array.init(allocator);
    for (runs_v.array.items) |r_val| {
        if (r_val != .object) return fail("dispatch state run record must be an object", .{});
        const r_obj = r_val.object;
        const seq = r_obj.get("sequence").?.integer;
        const rid_v = r_obj.get("run_id").?;
        const rid_str = switch (rid_v) {
            .integer => |i| try std.fmt.allocPrint(allocator, "{d}", .{i}),
            .string => |s| s,
            else => return fail("dispatch state run has invalid sequence or run_id", .{}),
        };
        for (DEFAULT_PLATFORMS) |platform| {
            const artifact_name = try std.fmt.allocPrint(allocator, "wasi-thread-bench-{s}-{s}-1", .{ platform, rid_str });
            const subdir = try std.fmt.allocPrint(allocator, "seq-{d:0>2}-run-{s}-{s}", .{ @as(u64, @intCast(seq)), rid_str, platform });
            const target_dir = try std.fs.path.join(allocator, &.{ output_dir, subdir });
            const report_json = try std.fs.path.join(allocator, &.{ target_dir, "report.json" });
            const report_md = try std.fs.path.join(allocator, &.{ target_dir, "report.md" });
            const json_exists = if (std.Io.Dir.cwd().statFile(io, report_json, .{})) |_| true else |_| false;
            const md_exists = if (std.Io.Dir.cwd().statFile(io, report_md, .{})) |_| true else |_| false;
            var reused = false;
            if (json_exists and md_exists) {
                reused = true;
            } else {
                std.Io.Dir.cwd().createDirPath(io, target_dir) catch return fail("cannot create dir: {s}", .{target_dir});
                const res = try runner.runProcess(
                    allocator,
                    io,
                    &.{ "gh", "run", "download", rid_str, "--repo", repo_v.string, "--name", artifact_name, "--dir", target_dir },
                    ".",
                    timeout_seconds,
                    null,
                );
                if (res.exit_code != 0) {
                    return fail("failed downloading artifact {s}: {s}", .{ artifact_name, res.stderr });
                }
            }
            var rec: ObjectMap = .empty;
            try rec.put(allocator, "sequence", .{ .integer = seq });
            try rec.put(allocator, "partition", r_obj.get("partition") orelse .null);
            try rec.put(allocator, "run_id", .{ .integer = std.fmt.parseInt(i64, rid_str, 10) catch 0 });
            try rec.put(allocator, "platform", .{ .string = platform });
            try rec.put(allocator, "artifact_name", .{ .string = artifact_name });
            try rec.put(allocator, "directory", .{ .string = target_dir });
            try rec.put(allocator, "report_json", .{ .string = report_json });
            try rec.put(allocator, "report_md", .{ .string = report_md });
            try rec.put(allocator, "reused", .{ .bool = reused });
            try downloaded.append(.{ .object = rec });
        }
    }

    var manifest: ObjectMap = .empty;
    try manifest.put(allocator, "schema_version", .{ .integer = spec.REPORT_SCHEMA_VERSION });
    try manifest.put(allocator, "kind", .{ .string = "wasi-thread-cohort-download-manifest" });
    try manifest.put(allocator, "downloaded_at", .{ .string = try benchmark_schema.collectedAt(allocator, io) });
    try manifest.put(allocator, "repository", repo_v);
    try manifest.put(allocator, "cohort_id", dispatch_state.get("cohort_id") orelse .null);
    try manifest.put(allocator, "requested_runs", req_runs_v);
    try manifest.put(allocator, "completed_runs", .{ .integer = n_runs });
    try manifest.put(allocator, "partial", .{ .bool = n_runs < req_runs_v.integer });
    try manifest.put(allocator, "artifacts", .{ .array = downloaded });
    const manifest_path = try std.fs.path.join(allocator, &.{ output_dir, "manifest.json" });
    try spec.atomicWriteJson(allocator, io, manifest_path, .{ .object = manifest });
    return manifest;
}

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = init.io;
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_it.deinit();
    _ = args_it.next();

    const subcmd = args_it.next() orelse {
        std.debug.print("error: subcommand required (dispatch|download|validate|derive)\n", .{});
        std.process.exit(2);
    };
    if (std.mem.eql(u8, subcmd, "download")) {
        var dispatch_state_path: ?[]const u8 = null;
        var output_dir: ?[]const u8 = null;
        var timeout_seconds: f64 = 300.0;
        var allow_partial = false;
        while (args_it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--dispatch-state")) {
                dispatch_state_path = args_it.next();
            } else if (std.mem.eql(u8, arg, "--output-dir")) {
                output_dir = args_it.next();
            } else if (std.mem.eql(u8, arg, "--timeout-seconds")) {
                timeout_seconds = std.fmt.parseFloat(f64, args_it.next() orelse "") catch 0.0;
            } else if (std.mem.eql(u8, arg, "--allow-partial")) {
                allow_partial = true;
            }
        }
        const dsp = dispatch_state_path orelse {
            std.debug.print("error: --dispatch-state is required\n", .{});
            std.process.exit(2);
        };
        const od = output_dir orelse {
            std.debug.print("error: --output-dir is required\n", .{});
            std.process.exit(2);
        };
        _ = downloadCohortArtifacts(allocator, io, dsp, od, timeout_seconds, allow_partial) catch |err| {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
    } else if (std.mem.eql(u8, subcmd, "validate")) {
        var input_dir: ?[]const u8 = null;
        var dispatch_state_path: ?[]const u8 = null;
        var allow_partial = false;
        var output_path: []const u8 = "wasi-thread-cohort.json";
        while (args_it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--input-dir")) {
                input_dir = args_it.next();
            } else if (std.mem.eql(u8, arg, "--dispatch-state")) {
                dispatch_state_path = args_it.next();
            } else if (std.mem.eql(u8, arg, "--allow-partial")) {
                allow_partial = true;
            } else if (std.mem.eql(u8, arg, "--output")) {
                output_path = args_it.next() orelse "wasi-thread-cohort.json";
            } else if (std.mem.eql(u8, arg, "--minimum-reports")) {
                _ = args_it.next();
            }
        }
        const idir = input_dir orelse {
            std.debug.print("error: --input-dir is required\n", .{});
            std.process.exit(2);
        };
        const dsp = dispatch_state_path orelse {
            std.debug.print("error: paired authoritative reports require their exact schema-v4 dispatch manifest\n", .{});
            std.process.exit(2);
        };
        const dispatch_state = loadJsonObject(allocator, io, dsp, "dispatch state") catch |err| {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        var dir = std.Io.Dir.cwd().openDir(io, idir, .{ .iterate = true }) catch {
            std.debug.print("error: cannot open input dir {s}\n", .{idir});
            std.process.exit(2);
        };
        defer dir.close(io);
        var walker = dir.walk(allocator) catch {
            std.debug.print("error: cannot walk input dir {s}\n", .{idir});
            std.process.exit(2);
        };
        defer walker.deinit();
        var docs: std.ArrayList(DocEntry) = .empty;
        while (walker.next(io) catch null) |entry| {
            if (entry.kind == .file and std.mem.eql(u8, entry.basename, "report.json")) {
                const full = try std.fs.path.join(allocator, &.{ idir, entry.path });
                const doc = loadJsonObject(allocator, io, full, full) catch |err| {
                    std.debug.print("error: {s}\n", .{@errorName(err)});
                    std.process.exit(2);
                };
                try docs.append(allocator, .{ .path = full, .doc = doc });
            }
        }
        const validated = validatePairedDocuments(allocator, io, docs.items, &DEFAULT_PLATFORMS, dispatch_state, allow_partial) catch |err| {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        try spec.atomicWriteJson(allocator, io, output_path, .{ .object = validated });
    } else if (std.mem.eql(u8, subcmd, "derive")) {
        var cohort_path: ?[]const u8 = null;
        var policy_path: ?[]const u8 = null;
        var budget_output: []const u8 = "wasi-thread-budget.candidate.json";
        var evidence_json_output: []const u8 = "wasi-thread-budget-evidence.json";
        var evidence_md_output: []const u8 = "wasi-thread-budget-evidence.md";
        while (args_it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--cohort")) {
                cohort_path = args_it.next();
            } else if (std.mem.eql(u8, arg, "--policy")) {
                policy_path = args_it.next();
            } else if (std.mem.eql(u8, arg, "--budget-output")) {
                budget_output = args_it.next() orelse "";
            } else if (std.mem.eql(u8, arg, "--evidence-json-output")) {
                evidence_json_output = args_it.next() orelse "";
            } else if (std.mem.eql(u8, arg, "--evidence-markdown-output")) {
                evidence_md_output = args_it.next() orelse "";
            }
        }
        const cp = cohort_path orelse {
            std.debug.print("error: --cohort is required\n", .{});
            std.process.exit(2);
        };
        const pp = policy_path orelse {
            std.debug.print("error: --policy is required\n", .{});
            std.process.exit(2);
        };
        if (std.mem.eql(u8, budget_output, evidence_json_output) or
            std.mem.eql(u8, budget_output, evidence_md_output) or
            std.mem.eql(u8, evidence_json_output, evidence_md_output))
        {
            std.debug.print("error: derive output paths must be distinct\n", .{});
            std.process.exit(2);
        }
        const cohort = loadJsonObject(allocator, io, cp, "validated cohort") catch |err| {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        const policy = loadJsonObject(allocator, io, pp, "derivation policy") catch |err| {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        const derived = deriveBudgetDocuments(allocator, cohort, policy) catch |err| {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        try spec.atomicWriteJson(allocator, io, budget_output, .{ .object = derived.budget });
        try spec.atomicWriteJson(allocator, io, evidence_json_output, .{ .object = derived.evidence });
        try spec.atomicWriteText(allocator, io, evidence_md_output, derived.markdown);
    }
}

test "validateDerivationPolicy accepts production and synthetic policy files" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const prod = try loadJsonObject(allocator, io, "tests/benchmarks/wasi-threads/derivation-policy.production.json", "production policy");
    const validated_prod = try validateDerivationPolicy(allocator, prod);
    try std.testing.expectEqual(@as(i64, 20), validated_prod.get("minimum_reports_per_platform").?.integer);

    const synth = try loadJsonObject(allocator, io, "tests/benchmarks/wasi-threads/derivation-policy.synthetic.json", "synthetic policy");
    const validated_synth = try validateDerivationPolicy(allocator, synth);
    try std.testing.expectEqual(@as(i64, 20), validated_synth.get("minimum_reports_per_platform").?.integer);
}

test "deriveOneSidedThreshold computes lower and upper bounds and flags diagnostic outliers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const values = [_]f64{
        1.000, 0.999, 1.001, 1.000, 0.998, 1.002, 1.000, 0.999,
        1.001, 1.000, 0.999, 1.001, 1.000, 0.998, 1.002, 0.960,
    };
    const lower = try deriveOneSidedThreshold(allocator, &values, "lower", 0.005, 0.10, "test-lower");
    try std.testing.expectEqualStrings("lower", lower.get("direction").?.string);
    try std.testing.expectEqual(@as(i64, 16), lower.get("observation_count").?.integer);
    try std.testing.expect(lower.get("threshold_ratio").?.float < 0.960);
    try std.testing.expectEqual(@as(usize, 1), lower.get("diagnostic_outlier_indices").?.array.items.len);
    try std.testing.expectEqual(@as(i64, 15), lower.get("diagnostic_outlier_indices").?.array.items[0].integer);

    const upper = try deriveOneSidedThreshold(allocator, &values, "upper", 0.005, 0.10, "test-upper");
    try std.testing.expectEqualStrings("upper", upper.get("direction").?.string);
    try std.testing.expect(upper.get("threshold_ratio").?.float > 1.002);
}

test "validateDispatchOptions and validateDispatchState enforce cohort invariants" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const sha = "0123456789abcdef0123456789abcdef01234567";
    const opts = try validateDispatchOptions(allocator, .{
        .workflow_ref = "wasi-thread-calibration-v1",
        .baseline_sha = sha,
        .candidate_sha = sha,
        .purpose = "noise-calibration",
        .profile = "authoritative",
        .warmups = 2,
        .samples = 12,
        .runner_target = "trusted-calibration",
        .accepted_cpu_class = &.{
            "ubuntu-22.04-x86_64=AMD EPYC 7763 64-Core Processor",
            "ubuntu-24.04-aarch64=Neoverse-N2",
        },
        .runs = 2,
        .training_runs = 1,
    });
    try std.testing.expectEqual(@as(i64, 1), opts.training_runs);
    try std.testing.expectEqualStrings("AMD EPYC 7763 64-Core Processor", opts.accepted_cpu_classes.get("ubuntu-22.04-x86_64").?[0]);
    _ = &main;
}

test "deriveBudgetDocuments derives disabled candidate budget and holdout evidence" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const policy = try loadJsonObject(allocator, io, "tests/benchmarks/wasi-threads/derivation-policy.synthetic.json", "synthetic policy");

    const sha = "0123456789abcdef0123456789abcdef01234567";
    const hash64 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    var samples_arr = Array.init(allocator);
    for (0..4) |_| try samples_arr.append(.{ .float = 1.001 });

    var comp_entry: ObjectMap = .empty;
    try comp_entry.put(allocator, "pair_kind", .{ .string = "single-infrastructure" });
    try comp_entry.put(allocator, "pair_key", .{ .string = "interpreter/single-scalar-loop/1" });
    try comp_entry.put(allocator, "condition", .{ .string = "threads-enabled" });
    try comp_entry.put(allocator, "metric_kind", .{ .string = "single-thread-control" });
    try comp_entry.put(allocator, "elapsed_candidate_over_baseline", .{ .array = samples_arr });
    try comp_entry.put(allocator, "throughput_candidate_over_baseline", .{ .array = samples_arr });
    var comps = Array.init(allocator);
    try comps.append(.{ .object = comp_entry });

    var ror_entry: ObjectMap = .empty;
    try ror_entry.put(allocator, "pair_kind", .{ .string = "single-infrastructure" });
    try ror_entry.put(allocator, "pair_key", .{ .string = "interpreter/single-scalar-loop/1" });
    try ror_entry.put(allocator, "left", .{ .string = "threads-disabled" });
    try ror_entry.put(allocator, "right", .{ .string = "threads-enabled" });
    try ror_entry.put(allocator, "elapsed_ratio_of_ratios", .{ .array = samples_arr });
    try ror_entry.put(allocator, "throughput_ratio_of_ratios", .{ .array = samples_arr });
    var rors = Array.init(allocator);
    try rors.append(.{ .object = ror_entry });

    var dir_entry: ObjectMap = .empty;
    try dir_entry.put(allocator, "pair_kind", .{ .string = "single-infrastructure" });
    try dir_entry.put(allocator, "pair_key", .{ .string = "interpreter/single-scalar-loop/1" });
    try dir_entry.put(allocator, "left", .{ .string = "threads-disabled" });
    try dir_entry.put(allocator, "right", .{ .string = "threads-enabled" });
    try dir_entry.put(allocator, "elapsed_right_over_left", .{ .array = samples_arr });
    try dir_entry.put(allocator, "throughput_right_over_left", .{ .array = samples_arr });
    var dirs = Array.init(allocator);
    try dirs.append(.{ .object = dir_entry });

    var metrics: ObjectMap = .empty;
    try metrics.put(allocator, "comparisons", .{ .array = comps });
    try metrics.put(allocator, "ratio_of_ratios", .{ .array = rors });
    try metrics.put(allocator, "direct_candidate_single_infrastructure", .{ .array = dirs });

    var observations = Array.init(allocator);
    for (DEFAULT_PLATFORMS, 0..) |plat, p_idx| {
        const cpu = if (p_idx == 0) "CPU-X86" else "CPU-ARM";
        for (1..21) |seq| {
            var obs: ObjectMap = .empty;
            try obs.put(allocator, "sequence", .{ .integer = @intCast(seq) });
            try obs.put(allocator, "partition", .{ .string = if (seq <= 16) "training" else "holdout" });
            try obs.put(allocator, "run_id", .{ .string = try std.fmt.allocPrint(allocator, "{d}", .{1000 + seq}) });
            try obs.put(allocator, "platform", .{ .string = plat });
            try obs.put(allocator, "cpu_class", .{ .string = cpu });
            try obs.put(allocator, "metrics", .{ .object = metrics });
            try observations.append(.{ .object = obs });
        }
    }

    var acc_x86 = Array.init(allocator);
    try acc_x86.append(.{ .string = "CPU-X86" });
    var acc_arm = Array.init(allocator);
    try acc_arm.append(.{ .string = "CPU-ARM" });
    var acc_map: ObjectMap = .empty;
    try acc_map.put(allocator, "ubuntu-22.04-x86_64", .{ .array = acc_x86 });
    try acc_map.put(allocator, "ubuntu-24.04-aarch64", .{ .array = acc_arm });

    var dispatch: ObjectMap = .empty;
    try dispatch.put(allocator, "runner_target", .{ .string = "github-hosted" });
    try dispatch.put(allocator, "accepted_cpu_classes", .{ .object = acc_map });

    var rev: ObjectMap = .empty;
    try rev.put(allocator, "commit", .{ .string = sha });
    try rev.put(allocator, "tracked_diff_sha256", .{ .string = hash64 });
    try rev.put(allocator, "build_source_sha256", .{ .string = hash64 });

    var identity: ObjectMap = .empty;
    try identity.put(allocator, "profile", .{ .string = "authoritative" });
    try identity.put(allocator, "comparison_purpose", .{ .string = "noise-calibration" });
    try identity.put(allocator, "baseline", .{ .object = rev });
    try identity.put(allocator, "candidate", .{ .object = rev });
    try identity.put(allocator, "fixture_set_sha256", .{ .string = hash64 });
    try identity.put(allocator, "measurement_plan_version", .{ .integer = spec.MEASUREMENT_PLAN_IDENTITY_VERSION });
    try identity.put(allocator, "measurement_plan_sha256", .{ .string = hash64 });

    var platforms: ObjectMap = .empty;
    try platforms.put(allocator, "ubuntu-22.04-x86_64", .{ .object = .empty });
    try platforms.put(allocator, "ubuntu-24.04-aarch64", .{ .object = .empty });

    var cohort: ObjectMap = .empty;
    try cohort.put(allocator, "schema_version", .{ .integer = spec.REPORT_SCHEMA_VERSION });
    try cohort.put(allocator, "kind", .{ .string = "wasi-thread-paired-cohort" });
    try cohort.put(allocator, "authoritative", .{ .bool = true });
    try cohort.put(allocator, "excluded_observations", .{ .array = Array.init(allocator) });
    try cohort.put(allocator, "identity", .{ .object = identity });
    try cohort.put(allocator, "dispatch", .{ .object = dispatch });
    try cohort.put(allocator, "split", .{ .object = .empty });
    try cohort.put(allocator, "platforms", .{ .object = platforms });
    try cohort.put(allocator, "observations", .{ .array = observations });

    const derived = try deriveBudgetDocuments(allocator, cohort, policy);
    try std.testing.expectEqual(false, derived.budget.get("enforcement").?.bool);
    try std.testing.expectEqualStrings("passed", derived.evidence.get("holdout_status").?.string);
    try std.testing.expect(std.mem.indexOf(u8, derived.markdown, "# WASI thread budget derivation evidence") != null);
}
