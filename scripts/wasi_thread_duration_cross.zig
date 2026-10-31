//! Run and validate the non-authoritative WASI thread duration-cross diagnostic.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const ObjectMap = std.json.ObjectMap;
const Array = std.json.Array;

const benchmark_schema = @import("benchmark_schema.zig");
const spec = @import("wasi_thread_bench_spec.zig");
const sizing = @import("wasi_thread_bench_sizing.zig");
const runner = @import("wasi_thread_bench_runner.zig");

pub const HarnessError = spec.HarnessError;
const fail = spec.fail;
const require = spec.require;

pub const KIND = "wasi-thread-duration-cross-diagnostic";
pub const REPORT_SCHEMA_VERSION: i64 = 1;
pub const PLAN_KIND = "wasi-thread-duration-cross-plan";
pub const PLAN_VERSION: i64 = 21;
pub const PROFILE_WARMUPS: i64 = 2;
pub const PROFILE_SAMPLES: i64 = 12;
pub const PROFILE_BLOCKS: i64 = 3;
pub const PROFILE_BLOCK_SIZE: i64 = 4;
pub const ARMS = [_][]const u8{ "current", "doubled" };
pub const TRUSTED_X86_RUNNER_NAME = "vm31e-wamr-temp-20260906";
pub const X86_CPU_CLASS = "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz";
pub const HOSTED_X86_RUNNER_NAME = "github-hosted-ubuntu-22.04";
pub const HOSTED_X86_CPU_CLASS = "AMD EPYC 7763 64-Core Processor";
pub const ARM_CPU_CLASS = "Neoverse-N2";
pub const INVOCATION_TIMEOUT_SECONDS: f64 = 180.0;
pub const MINIMUM_INTERVAL_NS: i64 = 1_250_000_000;
pub const DIAGNOSTIC_JOB_LIMIT_NS: i64 = 660 * 60 * 1_000_000_000;
pub const SIDECAR_MAX_SAMPLES: usize = 50_000;

pub const CellDef = struct {
    pair_key: []const u8,
    pair_kind: []const u8,
    left: []const u8,
    right: []const u8,
    workload: []const u8,
    threads: i64,
    left_target_s: f64,
    right_target_s: f64,
};

pub const X86_CELLS = [_]CellDef{
    .{ .pair_key = "cancel-points/hot/8", .pair_kind = "cancel-point-cost", .left = "cancel-points-off", .right = "cancel-points-on", .workload = "hot", .threads = 8, .left_target_s = 20.0, .right_target_s = 20.0 },
    .{ .pair_key = "runtime/atomic/1", .pair_kind = "runtime-parity", .left = "interpreter", .right = "aot", .workload = "atomic", .threads = 1, .left_target_s = 5.0, .right_target_s = 5.0 },
    .{ .pair_key = "runtime/atomic/2", .pair_kind = "runtime-parity", .left = "interpreter", .right = "aot", .workload = "atomic", .threads = 2, .left_target_s = 40.0, .right_target_s = 40.0 },
    .{ .pair_key = "runtime/atomic/4", .pair_kind = "runtime-parity", .left = "interpreter", .right = "aot", .workload = "atomic", .threads = 4, .left_target_s = 40.0, .right_target_s = 40.0 },
    .{ .pair_key = "runtime/atomic/8", .pair_kind = "runtime-parity", .left = "interpreter", .right = "aot", .workload = "atomic", .threads = 8, .left_target_s = 20.0, .right_target_s = 20.0 },
    .{ .pair_key = "runtime/hot/1", .pair_kind = "runtime-parity", .left = "interpreter", .right = "aot", .workload = "hot", .threads = 1, .left_target_s = 5.0, .right_target_s = 5.0 },
    .{ .pair_key = "runtime/hot/8", .pair_kind = "runtime-parity", .left = "interpreter", .right = "aot", .workload = "hot", .threads = 8, .left_target_s = 20.0, .right_target_s = 20.0 },
    .{ .pair_key = "runtime/spawn-join/1", .pair_kind = "runtime-parity", .left = "interpreter", .right = "aot", .workload = "spawn-join", .threads = 1, .left_target_s = 2.5, .right_target_s = 2.5 },
    .{ .pair_key = "runtime/spawn-join/2", .pair_kind = "runtime-parity", .left = "interpreter", .right = "aot", .workload = "spawn-join", .threads = 2, .left_target_s = 2.5, .right_target_s = 2.5 },
    .{ .pair_key = "runtime/spawn-join/8", .pair_kind = "runtime-parity", .left = "interpreter", .right = "aot", .workload = "spawn-join", .threads = 8, .left_target_s = 2.5, .right_target_s = 2.5 },
    .{ .pair_key = "runtime/wait-notify/1", .pair_kind = "runtime-parity", .left = "interpreter", .right = "aot", .workload = "wait-notify", .threads = 1, .left_target_s = 5.0, .right_target_s = 20.0 },
};

pub const ARM_CELLS = [_]CellDef{
    .{ .pair_key = "runtime/atomic/2", .pair_kind = "runtime-parity", .left = "interpreter", .right = "aot", .workload = "atomic", .threads = 2, .left_target_s = 40.0, .right_target_s = 40.0 },
    .{ .pair_key = "runtime/atomic/4", .pair_kind = "runtime-parity", .left = "interpreter", .right = "aot", .workload = "atomic", .threads = 4, .left_target_s = 40.0, .right_target_s = 40.0 },
    .{ .pair_key = "runtime/atomic/8", .pair_kind = "runtime-parity", .left = "interpreter", .right = "aot", .workload = "atomic", .threads = 8, .left_target_s = 20.0, .right_target_s = 20.0 },
};

pub fn partitionForSequence(sequence: i64) HarnessError![]const u8 {
    if (sequence < 1 or sequence > 20) return fail("report sequence must be in 1..20", .{});
    return if (sequence <= 16) "training" else "holdout";
}

pub fn cellsForPlatform(platform_id: []const u8) HarnessError![]const CellDef {
    if (std.mem.eql(u8, platform_id, "ubuntu-22.04-x86_64")) return &X86_CELLS;
    if (std.mem.eql(u8, platform_id, "ubuntu-24.04-aarch64")) return &ARM_CELLS;
    return fail("duration-cross requires one of ['ubuntu-22.04-x86_64', 'ubuntu-24.04-aarch64']", .{});
}

pub fn armOrder(sequence: i64, block_index: i64) HarnessError![2][]const u8 {
    _ = try partitionForSequence(sequence);
    if (block_index < 0 or block_index >= PROFILE_BLOCKS) return fail("block index must be in 0..2", .{});
    return if (@rem(sequence + block_index, 2) != 0)
        .{ "current", "doubled" }
    else
        .{ "doubled", "current" };
}

pub fn warmupArmOrder(sequence: i64, warmup_index: i64) HarnessError![2][]const u8 {
    _ = try partitionForSequence(sequence);
    if (warmup_index < 0 or warmup_index >= PROFILE_WARMUPS) return fail("warmup index must be in 0..1", .{});
    return if (@rem(sequence + warmup_index, 2) != 0)
        .{ "current", "doubled" }
    else
        .{ "doubled", "current" };
}

pub const LegSpec = struct {
    mode: []const u8,
    sizing_workload: []const u8,
    sizing_policy_workload: []const u8,
    guest_workload: []const u8,
    cancel_points: []const u8,
};

pub fn legSpec(cell: CellDef, condition: []const u8) HarnessError!LegSpec {
    if (!std.mem.eql(u8, condition, cell.left) and !std.mem.eql(u8, condition, cell.right)) {
        return fail("{s}: unexpected condition '{s}'", .{ cell.pair_key, condition });
    }
    if (std.mem.eql(u8, cell.pair_kind, "cancel-point-cost")) {
        return .{
            .mode = "aot",
            .sizing_workload = "cancel-hot",
            .sizing_policy_workload = "hot",
            .guest_workload = "hot",
            .cancel_points = if (std.mem.eql(u8, condition, "cancel-points-off")) "off" else "on",
        };
    }
    return .{
        .mode = condition,
        .sizing_workload = cell.workload,
        .sizing_policy_workload = cell.workload,
        .guest_workload = cell.workload,
        .cancel_points = if (std.mem.eql(u8, condition, "aot")) "on" else "interpreter-dispatch",
    };
}

pub fn pilotIterations(allocator: Allocator, ls: LegSpec, threads: i64) HarnessError!i64 {
    const table = try sizing.defaultPilotIterationPlanValue(allocator);
    const mode_obj = table.get(ls.mode) orelse return fail("unknown mode: {s}", .{ls.mode});
    const wl_obj = mode_obj.object.get(ls.sizing_workload) orelse return fail("unknown workload: {s}", .{ls.sizing_workload});
    const t_str = try std.fmt.allocPrint(allocator, "{d}", .{threads});
    const val = wl_obj.object.get(t_str) orelse return fail("unknown thread count: {d}", .{threads});
    return val.integer;
}

pub fn acceptanceChecksForPlatform(allocator: Allocator, platform_id: []const u8) HarnessError!Array {
    var arr = Array.init(allocator);
    const Entry = struct { coll: []const u8, k0: []const u8, k1: []const u8, k2: []const u8 };
    const x86_entries = [_]Entry{
        .{ .coll = "comparisons", .k0 = "cancel-points/hot/8", .k1 = "cancel-points-on", .k2 = "steady-state-kernel" },
        .{ .coll = "comparisons", .k0 = "runtime/atomic/2", .k1 = "interpreter", .k2 = "steady-state-kernel" },
        .{ .coll = "comparisons", .k0 = "runtime/atomic/4", .k1 = "aot", .k2 = "steady-state-kernel" },
        .{ .coll = "comparisons", .k0 = "runtime/atomic/4", .k1 = "interpreter", .k2 = "steady-state-kernel" },
        .{ .coll = "comparisons", .k0 = "runtime/atomic/8", .k1 = "aot", .k2 = "steady-state-kernel" },
        .{ .coll = "comparisons", .k0 = "runtime/atomic/8", .k1 = "interpreter", .k2 = "steady-state-kernel" },
        .{ .coll = "comparisons", .k0 = "runtime/hot/1", .k1 = "interpreter", .k2 = "steady-state-kernel" },
        .{ .coll = "comparisons", .k0 = "runtime/spawn-join/2", .k1 = "interpreter", .k2 = "spawn-join-lifecycle" },
        .{ .coll = "comparisons", .k0 = "runtime/wait-notify/1", .k1 = "interpreter", .k2 = "steady-state-kernel" },
        .{ .coll = "ratio_of_ratios", .k0 = "cancel-points/hot/8", .k1 = "cancel-points-off", .k2 = "cancel-points-on" },
        .{ .coll = "ratio_of_ratios", .k0 = "runtime/atomic/1", .k1 = "interpreter", .k2 = "aot" },
        .{ .coll = "ratio_of_ratios", .k0 = "runtime/atomic/2", .k1 = "interpreter", .k2 = "aot" },
        .{ .coll = "ratio_of_ratios", .k0 = "runtime/atomic/4", .k1 = "interpreter", .k2 = "aot" },
        .{ .coll = "ratio_of_ratios", .k0 = "runtime/atomic/8", .k1 = "interpreter", .k2 = "aot" },
        .{ .coll = "ratio_of_ratios", .k0 = "runtime/hot/1", .k1 = "interpreter", .k2 = "aot" },
        .{ .coll = "ratio_of_ratios", .k0 = "runtime/hot/8", .k1 = "interpreter", .k2 = "aot" },
        .{ .coll = "ratio_of_ratios", .k0 = "runtime/spawn-join/1", .k1 = "interpreter", .k2 = "aot" },
        .{ .coll = "ratio_of_ratios", .k0 = "runtime/spawn-join/2", .k1 = "interpreter", .k2 = "aot" },
        .{ .coll = "ratio_of_ratios", .k0 = "runtime/spawn-join/8", .k1 = "interpreter", .k2 = "aot" },
    };
    const arm_entries = [_]Entry{
        .{ .coll = "comparisons", .k0 = "runtime/atomic/2", .k1 = "aot", .k2 = "steady-state-kernel" },
        .{ .coll = "comparisons", .k0 = "runtime/atomic/2", .k1 = "interpreter", .k2 = "steady-state-kernel" },
        .{ .coll = "comparisons", .k0 = "runtime/atomic/4", .k1 = "aot", .k2 = "steady-state-kernel" },
        .{ .coll = "comparisons", .k0 = "runtime/atomic/4", .k1 = "interpreter", .k2 = "steady-state-kernel" },
        .{ .coll = "comparisons", .k0 = "runtime/atomic/8", .k1 = "aot", .k2 = "steady-state-kernel" },
        .{ .coll = "comparisons", .k0 = "runtime/atomic/8", .k1 = "interpreter", .k2 = "steady-state-kernel" },
        .{ .coll = "ratio_of_ratios", .k0 = "runtime/atomic/2", .k1 = "interpreter", .k2 = "aot" },
        .{ .coll = "ratio_of_ratios", .k0 = "runtime/atomic/4", .k1 = "interpreter", .k2 = "aot" },
        .{ .coll = "ratio_of_ratios", .k0 = "runtime/atomic/8", .k1 = "interpreter", .k2 = "aot" },
    };
    const entries: []const Entry = if (std.mem.eql(u8, platform_id, "ubuntu-22.04-x86_64"))
        &x86_entries
    else if (std.mem.eql(u8, platform_id, "ubuntu-24.04-aarch64"))
        &arm_entries
    else
        return fail("unsupported platform_id: {s}", .{platform_id});

    for (entries) |e| {
        var key_arr = Array.init(allocator);
        try key_arr.append(.{ .string = e.k0 });
        try key_arr.append(.{ .string = e.k1 });
        try key_arr.append(.{ .string = e.k2 });
        var item: ObjectMap = .empty;
        try item.put(allocator, "collection", .{ .string = e.coll });
        try item.put(allocator, "key", .{ .array = key_arr });
        try arr.append(.{ .object = item });
    }
    return arr;
}

pub fn buildPlan(
    allocator: Allocator,
    platform_id: []const u8,
    sequence: i64,
    source: ObjectMap,
) HarnessError!ObjectMap {
    const cells = try cellsForPlatform(platform_id);
    const partition = try partitionForSequence(sequence);

    var cells_arr = Array.init(allocator);
    for (cells) |c| {
        var targets: ObjectMap = .empty;
        try targets.put(allocator, c.left, .{ .float = c.left_target_s });
        try targets.put(allocator, c.right, .{ .float = c.right_target_s });
        var c_obj: ObjectMap = .empty;
        try c_obj.put(allocator, "pair_key", .{ .string = c.pair_key });
        try c_obj.put(allocator, "pair_kind", .{ .string = c.pair_kind });
        try c_obj.put(allocator, "left", .{ .string = c.left });
        try c_obj.put(allocator, "right", .{ .string = c.right });
        try c_obj.put(allocator, "workload", .{ .string = c.workload });
        try c_obj.put(allocator, "threads", .{ .integer = c.threads });
        try c_obj.put(allocator, "current_target_seconds", .{ .object = targets });
        try cells_arr.append(.{ .object = c_obj });
    }

    var design: ObjectMap = .empty;
    try design.put(allocator, "issue", .{ .integer = 966 });
    try design.put(allocator, "comment_id", .{ .integer = 5761582333 });
    try design.put(allocator, "original_analysis_json_sha256", .{ .string = "8189a60fbe5d5bfa678c4c62261ea4e56a1480e99bc7d89f0857822120149ad1" });
    try design.put(allocator, "original_analysis_markdown_sha256", .{ .string = "f9fc56a1625a6c6455a4c28c41b7505c43aac9e54057c9442070a37f4a60256c" });
    try design.put(allocator, "independent_review_json_sha256", .{ .string = "46e6459e2b45441759aaa25a9cbb1a01d5bd37984f78b5d16dd068df930f7340" });
    try design.put(allocator, "independent_review_markdown_sha256", .{ .string = "2401118a44b6a9aa64aa7508f647051cdd9b3ec8512a331483583da3d5bbadcf" });

    var log_revs = Array.init(allocator);
    for (spec.REVISION_ROLES) |r| try log_revs.append(.{ .string = r });
    var art_pol: ObjectMap = .empty;
    try art_pol.put(allocator, "kind", .{ .string = "artifact-identical-a-a" });
    try art_pol.put(allocator, "logical_revisions", .{ .array = log_revs });
    try art_pol.put(allocator, "builds", .{ .integer = 1 });
    try art_pol.put(allocator, "required_identity", .{ .string = "commit, tracked diff, build source, runtime binary, wasm and AOT artifact hashes must be byte-identical" });

    var prof: ObjectMap = .empty;
    try prof.put(allocator, "warmups", .{ .integer = PROFILE_WARMUPS });
    try prof.put(allocator, "samples", .{ .integer = PROFILE_SAMPLES });
    try prof.put(allocator, "blocks", .{ .integer = PROFILE_BLOCKS });
    try prof.put(allocator, "block_size", .{ .integer = PROFILE_BLOCK_SIZE });

    var cur_arm: ObjectMap = .empty;
    try cur_arm.put(allocator, "iteration_multiplier", .{ .integer = 1 });
    try cur_arm.put(allocator, "selects_thresholds", .{ .bool = false });
    var dbl_arm: ObjectMap = .empty;
    try dbl_arm.put(allocator, "iteration_multiplier", .{ .integer = 2 });
    try dbl_arm.put(allocator, "selects_thresholds", .{ .bool = true });
    var dur_arms: ObjectMap = .empty;
    try dur_arms.put(allocator, "current", .{ .object = cur_arm });
    try dur_arms.put(allocator, "doubled", .{ .object = dbl_arm });

    var arm_by_block = Array.init(allocator);
    for (0..@as(usize, @intCast(PROFILE_BLOCKS))) |b_idx| {
        const b: i64 = @intCast(b_idx);
        var s_indices = Array.init(allocator);
        for (0..4) |si| try s_indices.append(.{ .integer = b * 4 + @as(i64, @intCast(si)) });
        const order = try armOrder(sequence, b);
        var ord_arr = Array.init(allocator);
        try ord_arr.append(.{ .string = order[0] });
        try ord_arr.append(.{ .string = order[1] });
        var b_obj: ObjectMap = .empty;
        try b_obj.put(allocator, "block_index", .{ .integer = b });
        try b_obj.put(allocator, "sample_indices", .{ .array = s_indices });
        try b_obj.put(allocator, "arm_order", .{ .array = ord_arr });
        try arm_by_block.append(.{ .object = b_obj });
    }

    var warm_arm_order = Array.init(allocator);
    for (0..@as(usize, @intCast(PROFILE_WARMUPS))) |w_idx| {
        const w: i64 = @intCast(w_idx);
        const order = try warmupArmOrder(sequence, w);
        var ord_arr = Array.init(allocator);
        try ord_arr.append(.{ .string = order[0] });
        try ord_arr.append(.{ .string = order[1] });
        var w_obj: ObjectMap = .empty;
        try w_obj.put(allocator, "warmup_index", .{ .integer = w });
        try w_obj.put(allocator, "arm_order", .{ .array = ord_arr });
        try warm_arm_order.append(.{ .object = w_obj });
    }

    var siz_obj: ObjectMap = .empty;
    try siz_obj.put(allocator, "kind", .{ .string = "same-production-v20-one-shot-pilots-frozen-before-evidence" });
    try siz_obj.put(allocator, "significant_digits", .{ .integer = spec.SIZING_SIGNIFICANT_DIGITS });
    try siz_obj.put(allocator, "adaptation", .{ .string = "forbidden-after-first-measured-invocation" });
    try siz_obj.put(allocator, "doubled_admission", .{ .string = "fail-if-exactly-two-times-current-is-not-admitted" });

    var fixed_hot: ObjectMap = .empty;
    for ([_]i64{ 1, 2, 4, 8 }) |t| {
        const p_iter = try pilotIterations(allocator, .{
            .mode = "aot",
            .sizing_workload = "hot",
            .sizing_policy_workload = "hot",
            .guest_workload = "hot",
            .cancel_points = "on",
        }, t);
        try fixed_hot.put(allocator, try std.fmt.allocPrint(allocator, "{d}", .{t}), .{ .integer = p_iter });
    }
    var bar_pref: ObjectMap = .empty;
    try bar_pref.put(allocator, "probe_workload", .{ .string = "hot" });
    try bar_pref.put(allocator, "count_rule", .{ .string = "max(selected-cell-count, fixed-aot-hot-pilot-count)" });
    try bar_pref.put(allocator, "fixed_hot_pilot_counts", .{ .object = fixed_hot });
    try bar_pref.put(allocator, "minimum_timed_interval_ns", .{ .integer = MINIMUM_INTERVAL_NS });
    try bar_pref.put(allocator, "retries", .{ .integer = 0 });

    var req_tel = Array.init(allocator);
    for ([_][]const u8{
        "proc_stat_including_steal_and_context_switches",
        "proc_pressure_cpu_io_memory",
        "load_average",
        "cpu_frequency",
    }) |s| try req_tel.append(.{ .string = s });
    var tel_pol: ObjectMap = .empty;
    try tel_pol.put(allocator, "pre_post", .{ .string = "bounded immediately-before/immediately-after snapshots outside guest timed intervals" });
    try tel_pol.put(allocator, "required", .{ .array = req_tel });
    try tel_pol.put(allocator, "sidecar", .{ .string = "optional 1Hz benchmark-CPU frequency plus temperature/package-power samples, with the collector pinned to a logical CPU outside every benchmark assignment" });
    try tel_pol.put(allocator, "sensor_absence", .{ .string = "record-unavailable-never-retry-or-exclude" });

    var prohib = Array.init(allocator);
    for ([_][]const u8{
        "retry",
        "replacement",
        "exclusion",
        "adaptive-sleep",
        "adaptive-count",
        "adaptive-duration",
        "cell-extension",
        "report-extension",
    }) |s| try prohib.append(.{ .string = s });

    var plan: ObjectMap = .empty;
    try plan.put(allocator, "kind", .{ .string = PLAN_KIND });
    try plan.put(allocator, "version", .{ .integer = PLAN_VERSION });
    try plan.put(allocator, "diagnostic_only", .{ .bool = true });
    try plan.put(allocator, "production_budget_eligible", .{ .bool = false });
    try plan.put(allocator, "production_report_kind", .{ .string = "wasi-thread-benchmark" });
    try plan.put(allocator, "production_measurement_plan_version", .{ .integer = 20 });
    try plan.put(allocator, "reviewed_design", .{ .object = design });
    try plan.put(allocator, "platform_id", .{ .string = platform_id });
    try plan.put(allocator, "report_sequence", .{ .integer = sequence });
    try plan.put(allocator, "partition", .{ .string = partition });
    try plan.put(allocator, "source_revision", .{ .object = source });
    try plan.put(allocator, "artifact_policy", .{ .object = art_pol });
    try plan.put(allocator, "profile", .{ .object = prof });
    try plan.put(allocator, "invocation_timeout_seconds", .{ .float = INVOCATION_TIMEOUT_SECONDS });
    try plan.put(allocator, "minimum_interval_ns", .{ .integer = MINIMUM_INTERVAL_NS });
    try plan.put(allocator, "duration_arms", .{ .object = dur_arms });
    try plan.put(allocator, "arm_order_by_block", .{ .array = arm_by_block });
    try plan.put(allocator, "warmup_arm_order", .{ .array = warm_arm_order });
    try plan.put(allocator, "invocations_per_duration_arm_sample", .{ .integer = 4 });
    try plan.put(allocator, "pair_execution", try spec.pairExecutionPolicyValue(allocator));
    try plan.put(allocator, "sizing", .{ .object = siz_obj });
    try plan.put(allocator, "barrier_preflight_sizing", .{ .object = bar_pref });
    try plan.put(allocator, "telemetry", .{ .object = tel_pol });
    try plan.put(allocator, "cells", .{ .array = cells_arr });
    try plan.put(allocator, "acceptance_checks", .{ .array = try acceptanceChecksForPlatform(allocator, platform_id) });
    try plan.put(allocator, "prohibitions", .{ .array = prohib });
    try plan.put(allocator, "resolved_counts", .{ .object = .empty });
    try plan.put(allocator, "pilots", .{ .array = Array.init(allocator) });
    try plan.put(allocator, "runtime_admission", .{ .object = .empty });
    return plan;
}

pub fn planIdentity(allocator: Allocator, plan: ObjectMap) HarnessError![]const u8 {
    var portable: ObjectMap = .empty;
    var it = plan.iterator();
    while (it.next()) |entry| {
        const k = entry.key_ptr.*;
        if (std.mem.eql(u8, k, "resolved_counts") or
            std.mem.eql(u8, k, "pilots") or
            std.mem.eql(u8, k, "runtime_admission"))
        {
            continue;
        }
        try portable.put(allocator, k, entry.value_ptr.*);
    }
    var wrapper: ObjectMap = .empty;
    try wrapper.put(allocator, "schema_version", .{ .integer = PLAN_VERSION });
    try wrapper.put(allocator, "kind", .{ .string = PLAN_KIND });
    try wrapper.put(allocator, "portable_plan", .{ .object = portable });
    const digest = benchmark_schema.cacheKey(allocator, .{ .object = wrapper }) catch return error.BenchmarkDataError;
    return try allocator.dupe(u8, &digest);
}

pub fn summarizeReport(allocator: Allocator, records: []const ObjectMap) HarnessError!ObjectMap {
    var comparisons = Array.init(allocator);
    var ratios = Array.init(allocator);
    var internal = Array.init(allocator);

    var pair_keys_list: std.ArrayList([]const u8) = .empty;
    for (records) |rec| {
        if (!std.mem.eql(u8, rec.get("phase").?.string, "measure")) continue;
        const pk = rec.get("pair_key").?.string;
        var found = false;
        for (pair_keys_list.items) |existing| {
            if (std.mem.eql(u8, existing, pk)) {
                found = true;
                break;
            }
        }
        if (!found) try pair_keys_list.append(allocator, pk);
    }
    std.mem.sort([]const u8, pair_keys_list.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lt);

    for (ARMS) |arm| {
        for (pair_keys_list.items) |pair_key| {
            var selected: std.ArrayList(ObjectMap) = .empty;
            for (records) |rec| {
                if (std.mem.eql(u8, rec.get("phase").?.string, "measure") and
                    std.mem.eql(u8, rec.get("arm").?.string, arm) and
                    std.mem.eql(u8, rec.get("pair_key").?.string, pair_key))
                {
                    try selected.append(allocator, rec);
                }
            }
            if (selected.items.len == 0) {
                return fail("incomplete measured duration-cross sample {s}/{s}", .{ arm, pair_key });
            }
            const first = selected.items[0];
            const left = first.get("pair_left").?.string;
            const right = first.get("pair_right").?.string;

            var left_el: [PROFILE_SAMPLES]f64 = undefined;
            var left_tp: [PROFILE_SAMPLES]f64 = undefined;
            var right_el: [PROFILE_SAMPLES]f64 = undefined;
            var right_tp: [PROFILE_SAMPLES]f64 = undefined;
            var ror_el: [PROFILE_SAMPLES]f64 = undefined;
            var ror_tp: [PROFILE_SAMPLES]f64 = undefined;
            var int_base_el: [PROFILE_SAMPLES]f64 = undefined;
            var int_base_tp: [PROFILE_SAMPLES]f64 = undefined;
            var int_cand_el: [PROFILE_SAMPLES]f64 = undefined;
            var int_cand_tp: [PROFILE_SAMPLES]f64 = undefined;
            var left_mk: []const u8 = "";
            var right_mk: []const u8 = "";

            for (0..@as(usize, @intCast(PROFILE_SAMPLES))) |s_idx| {
                const sample_index: i64 = @intCast(s_idx);
                var lb: ?ObjectMap = null;
                var lc: ?ObjectMap = null;
                var rb: ?ObjectMap = null;
                var rc: ?ObjectMap = null;
                for (selected.items) |rec| {
                    if (rec.get("sample_index").?.integer != sample_index) continue;
                    const cond = rec.get("condition").?.string;
                    const rev = rec.get("revision").?.string;
                    if (std.mem.eql(u8, cond, left) and std.mem.eql(u8, rev, "baseline")) lb = rec;
                    if (std.mem.eql(u8, cond, left) and std.mem.eql(u8, rev, "candidate")) lc = rec;
                    if (std.mem.eql(u8, cond, right) and std.mem.eql(u8, rev, "baseline")) rb = rec;
                    if (std.mem.eql(u8, cond, right) and std.mem.eql(u8, rev, "candidate")) rc = rec;
                }
                if (lb == null or lc == null or rb == null or rc == null) {
                    return fail("incomplete measured duration-cross sample {s}/{s}/{d}", .{ arm, pair_key, sample_index });
                }
                left_mk = lb.?.get("metric_kind").?.string;
                right_mk = rb.?.get("metric_kind").?.string;
                const lb_el = spec.valueAsF64(lb.?.get("elapsed_ns").?).?;
                const lc_el = spec.valueAsF64(lc.?.get("elapsed_ns").?).?;
                const rb_el = spec.valueAsF64(rb.?.get("elapsed_ns").?).?;
                const rc_el = spec.valueAsF64(rc.?.get("elapsed_ns").?).?;
                const lb_tp = spec.valueAsF64(lb.?.get("throughput_ops_per_second").?).?;
                const lc_tp = spec.valueAsF64(lc.?.get("throughput_ops_per_second").?).?;
                const rb_tp = spec.valueAsF64(rb.?.get("throughput_ops_per_second").?).?;
                const rc_tp = spec.valueAsF64(rc.?.get("throughput_ops_per_second").?).?;

                left_el[s_idx] = lc_el / lb_el;
                left_tp[s_idx] = lc_tp / lb_tp;
                right_el[s_idx] = rc_el / rb_el;
                right_tp[s_idx] = rc_tp / rb_tp;

                const b_el = rb_el / lb_el;
                const c_el = rc_el / lc_el;
                const b_tp = rb_tp / lb_tp;
                const c_tp = rc_tp / lc_tp;
                ror_el[s_idx] = c_el / b_el;
                ror_tp[s_idx] = c_tp / b_tp;

                int_base_el[s_idx] = b_el;
                int_base_tp[s_idx] = b_tp;
                int_cand_el[s_idx] = c_el;
                int_cand_tp[s_idx] = c_tp;
            }

            var comp_l: ObjectMap = .empty;
            try comp_l.put(allocator, "arm", .{ .string = arm });
            try comp_l.put(allocator, "pair_kind", first.get("pair_kind").?);
            try comp_l.put(allocator, "pair_key", .{ .string = pair_key });
            try comp_l.put(allocator, "condition", .{ .string = left });
            try comp_l.put(allocator, "metric_kind", .{ .string = left_mk });
            try comp_l.put(allocator, "elapsed_candidate_over_baseline", .{ .object = try spec.positionBalancedRatioStats(allocator, &left_el, "samples") });
            try comp_l.put(allocator, "throughput_candidate_over_baseline", .{ .object = try spec.positionBalancedRatioStats(allocator, &left_tp, "samples") });
            try comparisons.append(.{ .object = comp_l });

            var comp_r: ObjectMap = .empty;
            try comp_r.put(allocator, "arm", .{ .string = arm });
            try comp_r.put(allocator, "pair_kind", first.get("pair_kind").?);
            try comp_r.put(allocator, "pair_key", .{ .string = pair_key });
            try comp_r.put(allocator, "condition", .{ .string = right });
            try comp_r.put(allocator, "metric_kind", .{ .string = right_mk });
            try comp_r.put(allocator, "elapsed_candidate_over_baseline", .{ .object = try spec.positionBalancedRatioStats(allocator, &right_el, "samples") });
            try comp_r.put(allocator, "throughput_candidate_over_baseline", .{ .object = try spec.positionBalancedRatioStats(allocator, &right_tp, "samples") });
            try comparisons.append(.{ .object = comp_r });

            var ror_obj: ObjectMap = .empty;
            try ror_obj.put(allocator, "arm", .{ .string = arm });
            try ror_obj.put(allocator, "pair_kind", first.get("pair_kind").?);
            try ror_obj.put(allocator, "pair_key", .{ .string = pair_key });
            try ror_obj.put(allocator, "left", .{ .string = left });
            try ror_obj.put(allocator, "right", .{ .string = right });
            try ror_obj.put(allocator, "elapsed_ratio_of_ratios", .{ .object = try spec.positionBalancedRatioStats(allocator, &ror_el, "samples") });
            try ror_obj.put(allocator, "throughput_ratio_of_ratios", .{ .object = try spec.positionBalancedRatioStats(allocator, &ror_tp, "samples") });
            try ratios.append(.{ .object = ror_obj });

            var int_b: ObjectMap = .empty;
            try int_b.put(allocator, "arm", .{ .string = arm });
            try int_b.put(allocator, "pair_kind", first.get("pair_kind").?);
            try int_b.put(allocator, "pair_key", .{ .string = pair_key });
            try int_b.put(allocator, "revision", .{ .string = "baseline" });
            try int_b.put(allocator, "left", .{ .string = left });
            try int_b.put(allocator, "right", .{ .string = right });
            try int_b.put(allocator, "elapsed_right_over_left", try spec.sampleStatsF64Value(allocator, &int_base_el, "samples"));
            try int_b.put(allocator, "throughput_right_over_left", try spec.sampleStatsF64Value(allocator, &int_base_tp, "samples"));
            try internal.append(.{ .object = int_b });

            var int_c: ObjectMap = .empty;
            try int_c.put(allocator, "arm", .{ .string = arm });
            try int_c.put(allocator, "pair_kind", first.get("pair_kind").?);
            try int_c.put(allocator, "pair_key", .{ .string = pair_key });
            try int_c.put(allocator, "revision", .{ .string = "candidate" });
            try int_c.put(allocator, "left", .{ .string = left });
            try int_c.put(allocator, "right", .{ .string = right });
            try int_c.put(allocator, "elapsed_right_over_left", try spec.sampleStatsF64Value(allocator, &int_cand_el, "samples"));
            try int_c.put(allocator, "throughput_right_over_left", try spec.sampleStatsF64Value(allocator, &int_cand_tp, "samples"));
            try internal.append(.{ .object = int_c });
        }
    }

    var out: ObjectMap = .empty;
    try out.put(allocator, "comparisons", .{ .array = comparisons });
    try out.put(allocator, "ratio_of_ratios", .{ .array = ratios });
    try out.put(allocator, "internal_pairs", .{ .array = internal });
    return out;
}

pub fn renderMarkdown(allocator: Allocator, document: ObjectMap) HarnessError![]const u8 {
    const metadata = document.get("metadata").?.object;
    const plan = document.get("plan").?.object;
    const sidecar = document.get("telemetry_sidecar").?.object;
    const summaries = document.get("summaries").?.object;

    var md: std.ArrayList(u8) = .empty;
    try md.appendSlice(allocator, "# WASI thread duration-cross diagnostic\n\n");
    try md.appendSlice(allocator, "> Non-authoritative diagnostic only. This report cannot emit or select a production budget.\n\n");
    try md.print(allocator, "- Platform: `{s}`\n", .{metadata.get("platform_id").?.string});
    try md.print(allocator, "- Sequence/partition: `{d}` / `{s}`\n", .{ metadata.get("report_sequence").?.integer, metadata.get("partition").?.string });
    try md.print(allocator, "- Source: `{s}`\n", .{metadata.get("source_revision").?.object.get("commit").?.string});
    try md.print(allocator, "- Plan: `{s}`\n", .{metadata.get("plan_identity_sha256").?.string});
    try md.print(allocator, "- Cells: {d}\n", .{plan.get("cells").?.array.items.len});
    try md.appendSlice(allocator, "- Profile: 2 discarded warmups, 12 measured samples in 3 complete blocks\n");
    try md.print(allocator, "- Sidecar available: `{s}`\n\n", .{if (sidecar.get("available").?.bool) "True" else "False"});
    try md.appendSlice(allocator, "| Arm | Pair | Metric | Throughput candidate/baseline | Elapsed candidate/baseline |\n");
    try md.appendSlice(allocator, "|---|---|---|---:|---:|\n");

    for (summaries.get("comparisons").?.array.items) |item_v| {
        const item = item_v.object;
        const tp_med = spec.valueAsF64(item.get("throughput_candidate_over_baseline").?.object.get("median").?).?;
        const el_med = spec.valueAsF64(item.get("elapsed_candidate_over_baseline").?.object.get("median").?).?;
        try md.print(allocator, "| `{s}` | `{s}` | `{s}` | {d:.6} | {d:.6} |\n", .{
            item.get("arm").?.string,
            item.get("pair_key").?.string,
            item.get("condition").?.string,
            tp_med,
            el_med,
        });
    }
    try md.appendSlice(allocator, "\nCurrent-duration observations are retained but select no threshold. Only a separately validated 20-report cohort may evaluate the doubled arm.\n");
    return try md.toOwnedSlice(allocator);
}

pub fn validateReport(allocator: Allocator, document: ObjectMap) HarnessError!void {
    const sv = document.get("schema_version") orelse return fail("schema_version", .{});
    const kind = document.get("kind") orelse return fail("report kind", .{});
    const auth = document.get("authoritative") orelse return fail("authoritative must be false", .{});
    const pb = document.get("production_budget") orelse return fail("production budget must be null", .{});
    if (sv != .integer or sv.integer != REPORT_SCHEMA_VERSION or
        kind != .string or !std.mem.eql(u8, kind.string, KIND) or
        auth != .bool or auth.bool or
        pb != .null)
    {
        return fail("invalid duration-cross report header", .{});
    }
    const meta_v = document.get("metadata") orelse return fail("metadata", .{});
    const plan_v = document.get("plan") orelse return fail("plan", .{});
    const recs_v = document.get("records") orelse return fail("records", .{});
    const sums_v = document.get("summaries") orelse return fail("summaries", .{});
    if (meta_v != .object or plan_v != .object or recs_v != .array or sums_v != .object) {
        return fail("invalid duration-cross report sections", .{});
    }
    const metadata = meta_v.object;
    const plan = plan_v.object;
    const platform_id = metadata.get("platform_id").?.string;
    const sequence = metadata.get("report_sequence").?.integer;
    const cells = try cellsForPlatform(platform_id);
    const exp_part = try partitionForSequence(sequence);
    if (!std.mem.eql(u8, metadata.get("partition").?.string, exp_part)) {
        return fail("metadata.partition", .{});
    }
    if (!std.mem.eql(u8, plan.get("kind").?.string, PLAN_KIND) or
        plan.get("version").?.integer != PLAN_VERSION or
        !plan.get("diagnostic_only").?.bool or
        plan.get("production_budget_eligible").?.bool)
    {
        return fail("diagnostic plan identity", .{});
    }
    const plan_sha_arr = benchmark_schema.cacheKey(allocator, .{ .object = plan }) catch return error.BenchmarkDataError;
    if (!std.mem.eql(u8, metadata.get("plan_sha256").?.string, &plan_sha_arr)) {
        return fail("metadata plan hash", .{});
    }
    const pid_sha = try planIdentity(allocator, plan);
    if (!std.mem.eql(u8, metadata.get("plan_identity_sha256").?.string, pid_sha)) {
        return fail("plan identity hash", .{});
    }
    if (sums_v.object.get("comparisons").?.array.items.len != cells.len * 4 or
        sums_v.object.get("ratio_of_ratios").?.array.items.len != cells.len * 2)
    {
        return fail("summaries count mismatch", .{});
    }
}

pub const CliArgs = struct {
    repo: []const u8 = ".",
    output_dir: []const u8 = "/d/wamr-duration-cross/output",
    source_sha: []const u8 = "",
    platform_id: []const u8 = "",
    report_sequence: i64 = 1,
    cohort_id: []const u8 = "",
    runner_environment: []const u8 = "github-hosted",
    host_pair_id: []const u8 = "",
    timeout_s: f64 = INVOCATION_TIMEOUT_SECONDS,
    min_interval_ms: f64 = 1250.0,
    optimize: []const u8 = "ReleaseFast",
    target: ?[]const u8 = null,
    aot_target: ?[]const u8 = null,
    runner_override: ?[]const u8 = null,
    rebuild: bool = false,
    preflight_only: bool = false,
    validate_report: ?[]const u8 = null,
};

pub fn resolveCounts(
    allocator: Allocator,
    plan: ObjectMap,
    pilots: []const ObjectMap,
    timeout_seconds: f64,
) HarnessError!struct { counts: ObjectMap, admission: ObjectMap } {
    const platform_id = plan.get("platform_id").?.string;
    const cells = try cellsForPlatform(platform_id);
    if (pilots.len != cells.len * 2) {
        return fail("duration-cross sizing pilot set is incomplete", .{});
    }

    const SizingKey = struct {
        mode: []const u8,
        workload: []const u8,
        policy_workload: []const u8,
        threads: i64,
    };
    var keys_list: std.ArrayList(SizingKey) = .empty;
    for (cells, 0..) |cell, c_idx| {
        for ([_][]const u8{ cell.left, cell.right }, 0..) |condition, cond_idx| {
            const p_idx = c_idx * 2 + cond_idx;
            const pilot = pilots[p_idx];
            const ls = try legSpec(cell, condition);
            const exp_iters = try pilotIterations(allocator, ls, cell.threads);
            var expected_spec: ObjectMap = .empty;
            try expected_spec.put(allocator, "pilot_index", .{ .integer = @intCast(p_idx) });
            try expected_spec.put(allocator, "revision", .{ .string = "baseline" });
            try expected_spec.put(allocator, "pair_kind", .{ .string = cell.pair_kind });
            try expected_spec.put(allocator, "pair_key", .{ .string = cell.pair_key });
            try expected_spec.put(allocator, "condition", .{ .string = condition });
            try expected_spec.put(allocator, "mode", .{ .string = ls.mode });
            try expected_spec.put(allocator, "workload", .{ .string = ls.sizing_workload });
            try expected_spec.put(allocator, "sizing_policy_workload", .{ .string = ls.sizing_policy_workload });
            try expected_spec.put(allocator, "threads", .{ .integer = cell.threads });
            try expected_spec.put(allocator, "iterations", .{ .integer = exp_iters });
            try sizing.validateSizingPilot(allocator, pilot, expected_spec);
            const spw = pilot.get("sizing_policy_workload") orelse return fail("sizing pilot policy workload mismatch", .{});
            if (spw != .string or !std.mem.eql(u8, spw.string, ls.sizing_policy_workload)) {
                return fail("sizing pilot policy workload mismatch", .{});
            }
            const sk = SizingKey{
                .mode = ls.mode,
                .workload = ls.sizing_workload,
                .policy_workload = ls.sizing_policy_workload,
                .threads = cell.threads,
            };
            var found = false;
            for (keys_list.items) |ek| {
                if (std.mem.eql(u8, ek.mode, sk.mode) and
                    std.mem.eql(u8, ek.workload, sk.workload) and
                    std.mem.eql(u8, ek.policy_workload, sk.policy_workload) and
                    ek.threads == sk.threads)
                {
                    found = true;
                    break;
                }
            }
            if (!found) try keys_list.append(allocator, sk);
        }
    }

    var sizing_cells = Array.init(allocator);
    var selected_map = std.StringHashMap(i64).init(allocator);
    var uncapped_map = std.StringHashMap(i64).init(allocator);

    for (keys_list.items) |sk| {
        var fastest_idx: i64 = -1;
        var fastest_elapsed: i64 = std.math.maxInt(i64);
        var fastest_iters: i64 = 0;
        var obs_count: i64 = 0;
        for (pilots) |p| {
            if (std.mem.eql(u8, p.get("mode").?.string, sk.mode) and
                std.mem.eql(u8, p.get("workload").?.string, sk.workload) and
                std.mem.eql(u8, p.get("sizing_policy_workload").?.string, sk.policy_workload) and
                p.get("threads").?.integer == sk.threads)
            {
                obs_count += 1;
                const ge = p.get("guest_elapsed_ns").?.integer;
                if (ge < fastest_elapsed) {
                    fastest_elapsed = ge;
                    fastest_idx = p.get("pilot_index").?.integer;
                    fastest_iters = p.get("iterations").?.integer;
                }
            }
        }
        const candidates = try sizing.sizingCandidatesForCell(
            allocator,
            sk.mode,
            sk.policy_workload,
            sk.threads,
            @intCast(fastest_iters),
            fastest_elapsed,
        );
        const uncapped: i64 = @intCast(candidates.selected_iterations);
        const cap: i64 = @intCast(try sizing.effectiveSizingCap(sk.workload, sk.threads));
        var selected_sources = candidates.selected_sources;
        var current = uncapped;
        if (std.mem.eql(u8, sk.workload, "spawn-join") and uncapped <= cap and uncapped * 2 > cap) {
            current = @divTrunc(cap, 2);
            selected_sources = Array.init(allocator);
            try selected_sources.append(.{ .string = "spawn-join-doubled-lifecycle-cap" });
        }
        const doubled = current * 2;
        const mask_cap: i64 = @intCast(spec.MASK64 / @as(u64, @intCast(sk.threads)));
        if (doubled > cap or doubled > mask_cap) {
            return fail("duration-cross doubled count {d} cannot be admitted under frozen cap {d}", .{ doubled, cap });
        }
        const map_key = try std.fmt.allocPrint(allocator, "{s}/{s}/{s}/{d}", .{ sk.mode, sk.workload, sk.policy_workload, sk.threads });
        try selected_map.put(map_key, current);
        try uncapped_map.put(map_key, uncapped);

        var sc_obj: ObjectMap = .empty;
        try sc_obj.put(allocator, "key", .{ .string = try std.fmt.allocPrint(allocator, "{s}/{s}/{d}", .{ sk.mode, sk.workload, sk.threads }) });
        try sc_obj.put(allocator, "mode", .{ .string = sk.mode });
        try sc_obj.put(allocator, "workload", .{ .string = sk.workload });
        try sc_obj.put(allocator, "sizing_policy_workload", .{ .string = sk.policy_workload });
        try sc_obj.put(allocator, "threads", .{ .integer = sk.threads });
        try sc_obj.put(allocator, "pilot_observations", .{ .integer = obs_count });
        try sc_obj.put(allocator, "fastest_pilot_index", .{ .integer = fastest_idx });
        try sc_obj.put(allocator, "fastest_elapsed_ns", .{ .integer = fastest_elapsed });
        try sc_obj.put(allocator, "current_iterations", .{ .integer = current });
        try sc_obj.put(allocator, "doubled_iterations", .{ .integer = doubled });
        try sc_obj.put(allocator, "effective_iteration_cap", .{ .integer = cap });
        try sc_obj.put(allocator, "selected_sources", .{ .array = selected_sources });
        try sizing_cells.append(.{ .object = sc_obj });
    }

    var total_projected_host_ns: i64 = 0;
    for (pilots) |p| total_projected_host_ns += p.get("host_wall_elapsed_ns").?.integer;
    var projections = Array.init(allocator);
    const timeout_ns: i64 = @intFromFloat(timeout_seconds * 1_000_000_000.0);

    for (pilots) |pilot| {
        const mode = pilot.get("mode").?.string;
        const wl = pilot.get("workload").?.string;
        const spw = pilot.get("sizing_policy_workload").?.string;
        const threads = pilot.get("threads").?.integer;
        const map_key = try std.fmt.allocPrint(allocator, "{s}/{s}/{s}/{d}", .{ mode, wl, spw, threads });
        const current = selected_map.get(map_key).?;
        const uncapped = uncapped_map.get(map_key).?;
        const p_iters = pilot.get("iterations").?.integer;
        const p_guest = pilot.get("guest_elapsed_ns").?.integer;
        const p_host = pilot.get("host_wall_elapsed_ns").?.integer;
        const p_overhead = pilot.get("timing_overhead_ns").?.integer;

        const ArmMult = struct { arm: []const u8, mult: i64 };
        for ([_]ArmMult{ .{ .arm = "current", .mult = 1 }, .{ .arm = "doubled", .mult = 2 } }) |am| {
            const selected = current * am.mult;
            const guest: i64 = @intCast(try sizing.ceilDiv(@as(i128, p_guest) * @as(i128, selected), @as(i128, p_iters)));
            const host: i64 = @intCast(try sizing.ceilDiv(@as(i128, p_host) * @as(i128, selected), @as(i128, p_iters)));
            const floor_ns = if (std.mem.eql(u8, wl, "spawn-join") and current < uncapped)
                MINIMUM_INTERVAL_NS * am.mult
            else
                sizing.projectedDurationFloorForCell(mode, spw, threads) * am.mult;

            if (guest < floor_ns or 99 * p_overhead >= guest or guest >= timeout_ns or host >= timeout_ns) {
                return fail("duration-cross {s} arm cannot be admitted for pilot {d}", .{ am.arm, pilot.get("pilot_index").?.integer });
            }
            total_projected_host_ns += host * (PROFILE_WARMUPS + PROFILE_SAMPLES) * @as(i64, @intCast(spec.REVISION_ROLES.len));

            var proj: ObjectMap = .empty;
            try proj.put(allocator, "pilot_index", pilot.get("pilot_index").?);
            try proj.put(allocator, "arm", .{ .string = am.arm });
            try proj.put(allocator, "iterations", .{ .integer = selected });
            try proj.put(allocator, "projected_guest_elapsed_ns", .{ .integer = guest });
            try proj.put(allocator, "projected_host_wall_elapsed_ns", .{ .integer = host });
            try projections.append(.{ .object = proj });
        }
    }

    if (total_projected_host_ns >= DIAGNOSTIC_JOB_LIMIT_NS) {
        return fail("duration-cross frozen projection exceeds the 660-minute benchmark bound", .{});
    }

    var cell_counts: ObjectMap = .empty;
    for (cells) |cell| {
        var counts_for_cell: ObjectMap = .empty;
        for ([_][]const u8{ cell.left, cell.right }) |condition| {
            const ls = try legSpec(cell, condition);
            const map_key = try std.fmt.allocPrint(allocator, "{s}/{s}/{s}/{d}", .{ ls.mode, ls.sizing_workload, ls.sizing_policy_workload, cell.threads });
            try counts_for_cell.put(allocator, condition, .{ .integer = selected_map.get(map_key).? });
        }
        try cell_counts.put(allocator, cell.pair_key, .{ .object = counts_for_cell });
    }

    var admission: ObjectMap = .empty;
    try admission.put(allocator, "sizing_cells", .{ .array = sizing_cells });
    try admission.put(allocator, "projections", .{ .array = projections });
    try admission.put(allocator, "projected_host_wall_ns", .{ .integer = total_projected_host_ns });
    try admission.put(allocator, "limit_ns", .{ .integer = DIAGNOSTIC_JOB_LIMIT_NS });
    return .{ .counts = cell_counts, .admission = admission };
}

pub fn barrierPreflightCounts(
    allocator: Allocator,
    plan: ObjectMap,
    counts: ObjectMap,
    thread_counts: []const i64,
) HarnessError!ObjectMap {
    const default_table = try sizing.defaultPilotIterationPlanValue(allocator);
    const hot_table = default_table.get("aot").?.object.get("hot").?.object;
    var result: ObjectMap = .empty;
    for (thread_counts) |threads| {
        var max_sel: i64 = 0;
        var any = false;
        for (plan.get("cells").?.array.items) |cell_v| {
            const cell = cell_v.object;
            if (cell.get("threads").?.integer != threads) continue;
            const pk = cell.get("pair_key").?.string;
            const left = cell.get("left").?.string;
            const right = cell.get("right").?.string;
            const c_obj = counts.get(pk).?.object;
            for ([_][]const u8{ left, right }) |cond| {
                const v = c_obj.get(cond).?.integer;
                if (!any or v > max_sel) {
                    max_sel = v;
                    any = true;
                }
            }
        }
        if (!any) return fail("no selected cells for {d}-thread preflight", .{threads});
        const t_str = try std.fmt.allocPrint(allocator, "{d}", .{threads});
        const floor_val = hot_table.get(t_str).?.integer;
        try result.put(allocator, t_str, .{ .integer = @max(max_sel, floor_val) });
    }
    return result;
}

pub fn parseCliArgs(allocator: Allocator, args_it: *std.process.Args.Iterator) HarnessError!CliArgs {
    var cli: CliArgs = .{};
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--repo")) {
            cli.repo = args_it.next() orelse return fail("missing --repo value", .{});
        } else if (std.mem.eql(u8, arg, "--output-dir")) {
            cli.output_dir = args_it.next() orelse return fail("missing --output-dir value", .{});
        } else if (std.mem.eql(u8, arg, "--source-sha")) {
            cli.source_sha = args_it.next() orelse return fail("missing --source-sha value", .{});
        } else if (std.mem.eql(u8, arg, "--platform-id")) {
            cli.platform_id = args_it.next() orelse return fail("missing --platform-id value", .{});
        } else if (std.mem.eql(u8, arg, "--report-sequence")) {
            const val = args_it.next() orelse return fail("missing --report-sequence value", .{});
            cli.report_sequence = std.fmt.parseInt(i64, val, 10) catch return fail("invalid --report-sequence", .{});
        } else if (std.mem.eql(u8, arg, "--cohort-id")) {
            cli.cohort_id = args_it.next() orelse return fail("missing --cohort-id value", .{});
        } else if (std.mem.eql(u8, arg, "--runner-environment")) {
            cli.runner_environment = args_it.next() orelse return fail("missing --runner-environment value", .{});
        } else if (std.mem.eql(u8, arg, "--host-pair-id")) {
            cli.host_pair_id = args_it.next() orelse return fail("missing --host-pair-id value", .{});
        } else if (std.mem.eql(u8, arg, "--timeout")) {
            const val = args_it.next() orelse return fail("missing --timeout value", .{});
            cli.timeout_s = std.fmt.parseFloat(f64, val) catch return fail("invalid --timeout", .{});
        } else if (std.mem.eql(u8, arg, "--min-interval-ms")) {
            const val = args_it.next() orelse return fail("missing --min-interval-ms value", .{});
            cli.min_interval_ms = std.fmt.parseFloat(f64, val) catch return fail("invalid --min-interval-ms", .{});
        } else if (std.mem.eql(u8, arg, "--optimize")) {
            cli.optimize = args_it.next() orelse return fail("missing --optimize value", .{});
        } else if (std.mem.eql(u8, arg, "--target")) {
            cli.target = args_it.next() orelse return fail("missing --target value", .{});
        } else if (std.mem.eql(u8, arg, "--aot-target")) {
            cli.aot_target = args_it.next() orelse return fail("missing --aot-target value", .{});
        } else if (std.mem.eql(u8, arg, "--runner")) {
            cli.runner_override = args_it.next() orelse return fail("missing --runner value", .{});
        } else if (std.mem.eql(u8, arg, "--rebuild")) {
            cli.rebuild = true;
        } else if (std.mem.eql(u8, arg, "--preflight-only")) {
            cli.preflight_only = true;
        } else if (std.mem.eql(u8, arg, "--validate-report")) {
            cli.validate_report = args_it.next() orelse return fail("missing --validate-report value", .{});
        } else {
            return fail("unknown argument: {s}", .{arg});
        }
    }
    _ = allocator;
    return cli;
}

fn readTextAvail(allocator: Allocator, io: std.Io, path: []const u8) !ObjectMap {
    var out: ObjectMap = .empty;
    if (std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024))) |raw| {
        try out.put(allocator, "available", .{ .bool = true });
        try out.put(allocator, "value", .{ .string = try allocator.dupe(u8, std.mem.trim(u8, raw, " \t\r\n")) });
        try out.put(allocator, "reason", .null);
    } else |err| {
        try out.put(allocator, "available", .{ .bool = false });
        try out.put(allocator, "value", .null);
        try out.put(allocator, "reason", .{ .string = @errorName(err) });
    }
    return out;
}

fn readProcStat(allocator: Allocator, io: std.Io) !ObjectMap {
    var out: ObjectMap = .empty;
    try out.put(allocator, "available", .{ .bool = false });
    try out.put(allocator, "cpu", .null);
    try out.put(allocator, "ctxt", .null);
    try out.put(allocator, "processes", .null);
    try out.put(allocator, "procs_running", .null);
    try out.put(allocator, "procs_blocked", .null);

    const raw = std.Io.Dir.cwd().readFileAlloc(io, "/proc/stat", allocator, .limited(256 * 1024)) catch |err| {
        try out.put(allocator, "reason", .{ .string = @errorName(err) });
        return out;
    };
    var lines = std.mem.splitScalar(u8, raw, '\n');
    while (lines.next()) |line| {
        var tok = std.mem.tokenizeAny(u8, line, " \t\r");
        const first = tok.next() orelse continue;
        if (std.mem.eql(u8, first, "cpu")) {
            const names = [_][]const u8{
                "user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal", "guest", "guest_nice",
            };
            var cpu_obj: ObjectMap = .empty;
            var idx: usize = 0;
            while (tok.next()) |f| : (idx += 1) {
                if (idx >= names.len) break;
                const val = std.fmt.parseInt(i64, f, 10) catch continue;
                try cpu_obj.put(allocator, names[idx], .{ .integer = val });
            }
            if (idx >= 8) {
                try out.put(allocator, "cpu", .{ .object = cpu_obj });
                try out.put(allocator, "available", .{ .bool = true });
            }
        } else if (std.mem.eql(u8, first, "ctxt") or
            std.mem.eql(u8, first, "processes") or
            std.mem.eql(u8, first, "procs_running") or
            std.mem.eql(u8, first, "procs_blocked"))
        {
            if (tok.next()) |f| {
                if (std.fmt.parseInt(i64, f, 10)) |val| {
                    try out.put(allocator, first, .{ .integer = val });
                } else |_| {}
            }
        }
    }
    return out;
}

fn frequencySnapshot(allocator: Allocator, io: std.Io, cpus: []const i64) !ObjectMap {
    var samples = Array.init(allocator);
    var unavailable = Array.init(allocator);
    for (cpus) |cpu| {
        const path = try std.fmt.allocPrint(allocator, "/sys/devices/system/cpu/cpu{d}/cpufreq/scaling_cur_freq", .{cpu});
        if (std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256))) |raw| {
            const trimmed = std.mem.trim(u8, raw, " \t\r\n");
            if (std.fmt.parseInt(i64, trimmed, 10)) |khz| {
                var item: ObjectMap = .empty;
                try item.put(allocator, "cpu", .{ .integer = cpu });
                try item.put(allocator, "khz", .{ .integer = khz });
                try samples.append(.{ .object = item });
            } else |err| {
                var eobj: ObjectMap = .empty;
                try eobj.put(allocator, "path", .{ .string = path });
                try eobj.put(allocator, "reason", .{ .string = @errorName(err) });
                try unavailable.append(.{ .object = eobj });
            }
        } else |err| {
            var eobj: ObjectMap = .empty;
            try eobj.put(allocator, "path", .{ .string = path });
            try eobj.put(allocator, "reason", .{ .string = @errorName(err) });
            try unavailable.append(.{ .object = eobj });
        }
    }
    var out: ObjectMap = .empty;
    const avail = samples.items.len > 0;
    try out.put(allocator, "available", .{ .bool = avail });
    try out.put(allocator, "samples", .{ .array = samples });
    try out.put(allocator, "unavailable", .{ .array = unavailable });
    try out.put(allocator, "reason", if (avail) .null else .{ .string = "no readable scaling_cur_freq sensors" });
    return out;
}

pub fn telemetrySnapshot(allocator: Allocator, io: std.Io, frequency_cpus: []const i64) !ObjectMap {
    var pressure: ObjectMap = .empty;
    for ([_][]const u8{ "cpu", "io", "memory" }) |name| {
        const p = try std.fmt.allocPrint(allocator, "/proc/pressure/{s}", .{name});
        try pressure.put(allocator, name, .{ .object = try readTextAvail(allocator, io, p) });
    }
    var out: ObjectMap = .empty;
    try out.put(allocator, "collected_at", .{ .string = try benchmark_schema.collectedAt(allocator, io) });
    const mono_ns: i64 = @max(@as(i64, 1), @as(i64, @intCast(std.Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds)));
    try out.put(allocator, "monotonic_ns", .{ .integer = mono_ns });
    try out.put(allocator, "proc_stat", .{ .object = try readProcStat(allocator, io) });
    try out.put(allocator, "pressure", .{ .object = pressure });
    try out.put(allocator, "loadavg", .{ .object = try readTextAvail(allocator, io, "/proc/loadavg") });
    try out.put(allocator, "frequency", .{ .object = try frequencySnapshot(allocator, io, frequency_cpus) });
    return out;
}

pub fn execute(allocator: Allocator, io: std.Io, environ: std.process.Environ, cli: CliArgs) HarnessError!ObjectMap {
    if (cli.output_dir.len == 0) return fail("duration-cross execution requires --output-dir", .{});
    if (cli.platform_id.len == 0) return fail("duration-cross execution requires --platform-id", .{});
    if (cli.runner_environment.len == 0) return fail("duration-cross execution requires --runner-environment", .{});
    if (cli.host_pair_id.len == 0) return fail("duration-cross execution requires --host-pair-id", .{});
    const output_dir = cli.output_dir;
    const platform_id = cli.platform_id;
    const report_sequence = cli.report_sequence;
    const runner_environment = cli.runner_environment;
    const host_pair_id = cli.host_pair_id;
    if (cli.timeout_s != INVOCATION_TIMEOUT_SECONDS) {
        return fail("duration-cross timeout is frozen at {d} seconds", .{INVOCATION_TIMEOUT_SECONDS});
    }
    const min_interval_ns: i64 = @intFromFloat(cli.min_interval_ms * 1_000_000.0);
    if (min_interval_ns != MINIMUM_INTERVAL_NS) {
        return fail("duration-cross minimum timed interval is frozen at 1250 milliseconds", .{});
    }
    const cwd = std.process.currentPathAlloc(io, allocator) catch return fail("cannot resolve cwd", .{});
    const repo = if (std.fs.path.isAbsolute(cli.repo)) cli.repo else try std.fs.path.join(allocator, &.{ cwd, cli.repo });
    const output = if (std.fs.path.isAbsolute(output_dir))
        output_dir
    else
        try std.fs.path.join(allocator, &.{ cwd, output_dir });

    if (std.mem.eql(u8, environ.getPosix("GITHUB_ACTIONS") orelse "", "true") and !std.mem.startsWith(u8, output, "/d/")) {
        return fail("GitHub duration-cross output must remain under /d", .{});
    }
    std.Io.Dir.cwd().createDirPath(io, output) catch return fail("cannot create {s}", .{output});
    for ([_][]const u8{
        "report.json",
        "report.md",
        "preflight-readiness.json",
        "preflight-readiness.md",
        "failure-diagnostic.json",
        "failure-diagnostic.md",
    }) |stale| {
        const sp = try std.fs.path.join(allocator, &.{ output, stale });
        std.Io.Dir.cwd().deleteFile(io, sp) catch {};
    }
    if (cli.preflight_only and report_sequence != 1) {
        return fail("readiness-only probe requires sequence 1", .{});
    }
    const source = try runner.sourceIdentity(allocator, io, repo);
    if (cli.source_sha.len > 0) {
        if (!std.mem.eql(u8, source.get("commit").?.string, cli.source_sha)) {
            return fail("source checkout {s} does not match immutable --source-sha {s}", .{ source.get("commit").?.string, cli.source_sha });
        }
    }
    var plan = try buildPlan(allocator, platform_id, report_sequence, source);
    var rev_repos: std.StringHashMapUnmanaged([]const u8) = .empty;
    try rev_repos.put(allocator, "candidate", repo);
    const resolved_fixtures = try runner.resolveMeasurementFixtures(allocator, io, rev_repos);
    const cells = try cellsForPlatform(platform_id);
    var tc_list: std.ArrayList(i64) = .empty;
    for (cells) |c| {
        var exists = false;
        for (tc_list.items) |t| if (t == c.threads) {
            exists = true;
        };
        if (!exists) try tc_list.append(allocator, c.threads);
    }
    std.mem.sort(i64, tc_list.items, {}, std.sort.asc(i64));
    const thread_counts = tc_list.items;
    const cpu_placement = try spec.discoverCpuPlacement(allocator, io, thread_counts);
    try spec.validateCpuPlacement(allocator, .{ .object = cpu_placement }, thread_counts);
    const host = try benchmark_schema.hostMetadata(allocator, io, environ, runner_environment);
    const quiescence = try runner.hostQuiescenceDiagnostics(allocator, io, environ);
    const host_pair = try runner.hostPairIdentity(allocator, platform_id, host, host_pair_id);
    _ = host_pair;

    const build_root = try std.fs.path.join(allocator, &.{ output, "build" });
    var builds: std.StringHashMapUnmanaged(runner.Build) = .empty;
    for ([_][]const u8{ "interpreter", "aot" }) |mode| {
        const b = try runner.buildVariant(allocator, io, environ, .{
            .repo = repo,
            .root = build_root,
            .mode = mode,
            .threads_enabled = true,
            .optimize = cli.optimize,
            .target = cli.target,
            .source = source,
            .rebuild = cli.rebuild,
            .compiler_toggle = std.mem.eql(u8, mode, "aot"),
        });
        try builds.put(allocator, b.name, b);
    }
    const exec_arch = if (cli.aot_target) |at| at else if (std.mem.endsWith(u8, platform_id, "aarch64")) "aarch64" else "x86_64";
    const compiler = builds.get("enabled-aot").?;
    const aot_artifacts = try runner.compileAotFixtures(allocator, io, environ, resolved_fixtures.fixture_repo, build_root, compiler, exec_arch);
    const aot_meta = try runner.aotArtifactReport(allocator, io, aot_artifacts, exec_arch);

    var hashes: ObjectMap = .empty;
    for ([_][]const u8{ "enabled-interpreter", "enabled-aot" }) |bname| {
        const b = builds.get(bname).?;
        try hashes.put(allocator, try std.fmt.allocPrint(allocator, "runtime/{s}", .{bname}), .{ .string = try spec.sha256File(allocator, io, b.wamr) });
        if (b.wamrc) |wc| {
            try hashes.put(allocator, try std.fmt.allocPrint(allocator, "compiler/{s}", .{bname}), .{ .string = try spec.sha256File(allocator, io, wc) });
        }
    }
    var aot_it = aot_artifacts.iterator();
    while (aot_it.next()) |e| {
        try hashes.put(allocator, try std.fmt.allocPrint(allocator, "aot/{s}", .{e.key_ptr.*}), .{ .string = try spec.sha256File(allocator, io, e.value_ptr.string) });
    }
    const single_wasm = try std.fs.path.join(allocator, &.{ resolved_fixtures.fixture_repo, "tests/benchmarks/wasi-threads/single.wasm" });
    const threaded_wasm = try std.fs.path.join(allocator, &.{ resolved_fixtures.fixture_repo, "tests/benchmarks/wasi-threads/threaded.wasm" });
    try hashes.put(allocator, "fixture/single", .{ .string = try spec.sha256File(allocator, io, single_wasm) });
    try hashes.put(allocator, "fixture/threaded", .{ .string = try spec.sha256File(allocator, io, threaded_wasm) });

    var runner_toks: std.ArrayList([]const u8) = .empty;
    var r_it = std.mem.tokenizeAny(u8, cli.runner_override orelse "", " \t\r\n");
    while (r_it.next()) |tok| try runner_toks.append(allocator, tok);

    var pilot_records: std.ArrayList(ObjectMap) = .empty;
    var pilot_arr = Array.init(allocator);
    for (cells) |cell| {
        for ([_][]const u8{ cell.left, cell.right }) |cond| {
            const ls = try legSpec(cell, cond);
            const iters = try pilotIterations(allocator, ls, cell.threads);
            const b = builds.get(if (std.mem.eql(u8, ls.mode, "interpreter")) "enabled-interpreter" else "enabled-aot").?;
            const mod_path = if (std.mem.eql(u8, cell.pair_kind, "cancel-point-cost"))
                aot_artifacts.get(try std.fmt.allocPrint(allocator, "threaded-polls-{s}", .{ls.cancel_points})).?.string
            else if (std.mem.eql(u8, ls.mode, "interpreter"))
                threaded_wasm
            else
                aot_artifacts.get("threaded-polls-on").?.string;

            var rf: ObjectMap = .empty;
            try rf.put(allocator, "pilot_index", .{ .integer = @intCast(pilot_records.items.len) });
            try rf.put(allocator, "revision", .{ .string = "baseline" });
            try rf.put(allocator, "pair_kind", .{ .string = cell.pair_kind });
            try rf.put(allocator, "pair_key", .{ .string = cell.pair_key });
            try rf.put(allocator, "condition", .{ .string = cond });
            try rf.put(allocator, "mode", .{ .string = ls.mode });
            try rf.put(allocator, "workload", .{ .string = ls.sizing_workload });
            try rf.put(allocator, "sizing_policy_workload", .{ .string = ls.sizing_policy_workload });
            try rf.put(allocator, "threads", .{ .integer = cell.threads });
            try rf.put(allocator, "iterations", .{ .integer = iters });
            try rf.put(allocator, "phase", .{ .string = "pilot" });

            const freq_cpus = try spec.cpuAffinityFor(allocator, cpu_placement, cell.workload, cell.threads);
            const before = try telemetrySnapshot(allocator, io, freq_cpus);
            var pilot = try runner.measureOnce(allocator, io, .{
                .repo = repo,
                .runner = runner_toks.items,
                .cpu_placement = cpu_placement,
                .wamr_path = b.wamr,
                .module_path = mod_path,
                .workload = ls.guest_workload,
                .threads = cell.threads,
                .iterations = @intCast(iters),
                .timeout = cli.timeout_s,
                .min_interval_ns = spec.PILOT_CLOCK_RESOLUTION_MINIMUM_NS,
                .enforce_timing_quality = false,
                .record_fields = rf,
            });
            const after = try telemetrySnapshot(allocator, io, freq_cpus);
            var tel: ObjectMap = .empty;
            try tel.put(allocator, "collection", .{ .string = "before-and-after-each-pilot-warmup-and-measured-record" });
            try tel.put(allocator, "before", .{ .object = before });
            try tel.put(allocator, "after", .{ .object = after });
            try pilot.put(allocator, "telemetry", .{ .object = tel });
            try sizing.validateSizingPilot(allocator, pilot, rf);
            try pilot_records.append(allocator, pilot);
            try pilot_arr.append(.{ .object = pilot });
        }
    }
    const resolved = try resolveCounts(allocator, plan, pilot_records.items, cli.timeout_s);
    try plan.put(allocator, "resolved_counts", .{ .object = resolved.counts });
    try plan.put(allocator, "pilots", .{ .array = pilot_arr });
    try plan.put(allocator, "runtime_admission", .{ .object = resolved.admission });

    const bp_counts = try barrierPreflightCounts(allocator, plan, resolved.counts, thread_counts);
    const quality_preflight = try runner.runTrustedBarrierPreflight(allocator, io, .{
        .repo = repo,
        .runner = runner_toks.items,
        .cpu_placement = cpu_placement,
        .wamr_path = builds.get("enabled-aot").?.wamr,
        .module_path = aot_artifacts.get("threaded-polls-on").?.string,
        .thread_counts = thread_counts,
        .iterations_by_thread = bp_counts,
        .timeout = cli.timeout_s,
        .minimum_interval_ns = min_interval_ns,
        .static_cancel_poll_sites = aot_meta.get("cancel_poll_static").?.object.get("sites_enabled").?.integer,
    });
    if (!std.mem.eql(u8, quality_preflight.get("status").?.string, "passed")) {
        return fail("trusted scheduler/barrier preflight failed", .{});
    }
    if (cli.preflight_only) {
        var readiness: ObjectMap = .empty;
        try readiness.put(allocator, "schema_version", .{ .integer = 1 });
        try readiness.put(allocator, "kind", .{ .string = "wasi-thread-duration-cross-preflight-readiness" });
        try readiness.put(allocator, "authoritative", .{ .bool = false });
        try readiness.put(allocator, "production_budget", .null);
        try readiness.put(allocator, "collected_at", .{ .string = try benchmark_schema.collectedAt(allocator, io) });
        try readiness.put(allocator, "source_sha", .{ .string = source.get("commit").?.string });
        try readiness.put(allocator, "platform_id", .{ .string = platform_id });
        try readiness.put(allocator, "report_sequence", .{ .integer = report_sequence });
        try readiness.put(allocator, "cohort_id", .{ .string = cli.cohort_id });
        try readiness.put(allocator, "workflow_run_id", .{ .string = environ.getPosix("GITHUB_RUN_ID") orelse "" });
        try readiness.put(allocator, "workflow_run_attempt", .{ .string = environ.getPosix("GITHUB_RUN_ATTEMPT") orelse "" });
        try readiness.put(allocator, "host", .{ .object = host });
        try readiness.put(allocator, "host_quiescence_at_start", .{ .object = quiescence });
        try readiness.put(allocator, "cpu_placement", .{ .object = cpu_placement });
        try readiness.put(allocator, "artifact_identity", .{ .object = hashes });
        try readiness.put(allocator, "fixture_set_sha256", .{ .string = resolved_fixtures.fixture_set_sha256 });
        try readiness.put(allocator, "plan_identity_sha256", .{ .string = try planIdentity(allocator, plan) });
        try readiness.put(allocator, "resolved_counts", .{ .object = resolved.counts });
        try readiness.put(allocator, "runtime_admission", .{ .object = resolved.admission });
        try readiness.put(allocator, "pilot_count", .{ .integer = @intCast(pilot_records.items.len) });
        try readiness.put(allocator, "quality_preflight", .{ .object = quality_preflight });
        try readiness.put(allocator, "benchmark_observations", .{ .integer = 0 });
        try readiness.put(allocator, "cohort_eligible", .{ .bool = false });
        try spec.atomicWriteJson(allocator, io, try std.fs.path.join(allocator, &.{ output, "preflight-readiness.json" }), .{ .object = readiness });
        const md = try std.fmt.allocPrint(
            allocator,
            "# Duration-cross {s} gate readiness (non-authoritative)\n\n" ++
                "- Source: `{s}`\n" ++
                "- Sequence: `{d}`\n" ++
                "- Preflight: `{s}` with {d} retained probes\n" ++
                "- Benchmark observations: 0; no full cohort was dispatched.\n",
            .{
                platform_id,
                source.get("commit").?.string,
                report_sequence,
                quality_preflight.get("status").?.string,
                quality_preflight.get("probe_count").?.integer,
            },
        );
        try spec.atomicWriteText(allocator, io, try std.fs.path.join(allocator, &.{ output, "preflight-readiness.md" }), md);
        return readiness;
    }

    var records_list: std.ArrayList(ObjectMap) = .empty;
    var records_arr = Array.init(allocator);
    for (cells) |cell| {
        const cell_counts = resolved.counts.get(cell.pair_key).?.object;
        var w_idx: i64 = 0;
        while (w_idx < PROFILE_WARMUPS) : (w_idx += 1) {
            const w_arms = try warmupArmOrder(report_sequence, w_idx);
            for (w_arms) |arm| {
                const pairs = try spec.pairedInvocationOrder(allocator, @intCast(w_idx), cell.pair_kind, cell.left, cell.right, &spec.REVISION_ROLES);
                for (pairs) |pair| {
                    const ls = try legSpec(cell, pair.condition);
                    const base_iters = cell_counts.get(pair.condition).?.integer;
                    const iters = base_iters * (if (std.mem.eql(u8, arm, "doubled")) @as(i64, 2) else 1);
                    const b = builds.get(if (std.mem.eql(u8, ls.mode, "interpreter")) "enabled-interpreter" else "enabled-aot").?;
                    const mod_path = if (std.mem.eql(u8, cell.pair_kind, "cancel-point-cost"))
                        aot_artifacts.get(try std.fmt.allocPrint(allocator, "threaded-polls-{s}", .{ls.cancel_points})).?.string
                    else if (std.mem.eql(u8, ls.mode, "interpreter"))
                        threaded_wasm
                    else
                        aot_artifacts.get("threaded-polls-on").?.string;
                    var rf: ObjectMap = .empty;
                    try rf.put(allocator, "revision", .{ .string = pair.revision });
                    try rf.put(allocator, "pair_kind", .{ .string = cell.pair_kind });
                    try rf.put(allocator, "pair_key", .{ .string = cell.pair_key });
                    try rf.put(allocator, "phase", .{ .string = "warmup" });
                    try rf.put(allocator, "phase_index", .{ .integer = w_idx });
                    try rf.put(allocator, "sample_index", .null);
                    try rf.put(allocator, "block_index", .null);
                    try rf.put(allocator, "sample_in_block", .null);
                    try rf.put(allocator, "arm", .{ .string = arm });
                    try rf.put(allocator, "condition", .{ .string = pair.condition });
                    try rf.put(allocator, "pair_left", .{ .string = cell.left });
                    try rf.put(allocator, "pair_right", .{ .string = cell.right });
                    try rf.put(allocator, "pair_execution", .{ .string = "alternating-single-invocation" });
                    try rf.put(allocator, "mode", .{ .string = ls.mode });
                    try rf.put(allocator, "workload", .{ .string = ls.guest_workload });
                    try rf.put(allocator, "threads", .{ .integer = cell.threads });
                    try rf.put(allocator, "cancel_points", .{ .string = ls.cancel_points });
                    try rf.put(allocator, "iterations", .{ .integer = iters });
                    const freq_cpus = try spec.cpuAffinityFor(allocator, cpu_placement, cell.workload, cell.threads);
                    const before = try telemetrySnapshot(allocator, io, freq_cpus);
                    var rec = try runner.measureOnce(allocator, io, .{
                        .repo = repo,
                        .runner = runner_toks.items,
                        .cpu_placement = cpu_placement,
                        .wamr_path = b.wamr,
                        .module_path = mod_path,
                        .workload = ls.guest_workload,
                        .threads = cell.threads,
                        .iterations = @intCast(iters),
                        .timeout = cli.timeout_s,
                        .min_interval_ns = min_interval_ns,
                        .record_fields = rf,
                    });
                    const after = try telemetrySnapshot(allocator, io, freq_cpus);
                    var tel: ObjectMap = .empty;
                    try tel.put(allocator, "collection", .{ .string = "before-and-after-each-pilot-warmup-and-measured-record" });
                    try tel.put(allocator, "before", .{ .object = before });
                    try tel.put(allocator, "after", .{ .object = after });
                    try rec.put(allocator, "telemetry", .{ .object = tel });
                    try records_list.append(allocator, rec);
                    try records_arr.append(.{ .object = rec });
                }
            }
        }
        var b_idx: i64 = 0;
        while (b_idx < PROFILE_BLOCKS) : (b_idx += 1) {
            var s_in_b: i64 = 0;
            while (s_in_b < PROFILE_BLOCK_SIZE) : (s_in_b += 1) {
                const s_idx = b_idx * PROFILE_BLOCK_SIZE + s_in_b;
                const b_arms = try armOrder(report_sequence, b_idx);
                for (b_arms) |arm| {
                    const pairs = try spec.pairedInvocationOrder(allocator, @intCast(s_in_b), cell.pair_kind, cell.left, cell.right, &spec.REVISION_ROLES);
                    for (pairs) |pair| {
                        const ls = try legSpec(cell, pair.condition);
                        const base_iters = cell_counts.get(pair.condition).?.integer;
                        const iters = base_iters * (if (std.mem.eql(u8, arm, "doubled")) @as(i64, 2) else 1);
                        const b = builds.get(if (std.mem.eql(u8, ls.mode, "interpreter")) "enabled-interpreter" else "enabled-aot").?;
                        const mod_path = if (std.mem.eql(u8, cell.pair_kind, "cancel-point-cost"))
                            aot_artifacts.get(try std.fmt.allocPrint(allocator, "threaded-polls-{s}", .{ls.cancel_points})).?.string
                        else if (std.mem.eql(u8, ls.mode, "interpreter"))
                            threaded_wasm
                        else
                            aot_artifacts.get("threaded-polls-on").?.string;
                        var rf: ObjectMap = .empty;
                        try rf.put(allocator, "revision", .{ .string = pair.revision });
                        try rf.put(allocator, "pair_kind", .{ .string = cell.pair_kind });
                        try rf.put(allocator, "pair_key", .{ .string = cell.pair_key });
                        try rf.put(allocator, "phase", .{ .string = "measure" });
                        try rf.put(allocator, "phase_index", .{ .integer = s_idx });
                        try rf.put(allocator, "sample_index", .{ .integer = s_idx });
                        try rf.put(allocator, "block_index", .{ .integer = b_idx });
                        try rf.put(allocator, "sample_in_block", .{ .integer = s_in_b });
                        try rf.put(allocator, "arm", .{ .string = arm });
                        try rf.put(allocator, "condition", .{ .string = pair.condition });
                        try rf.put(allocator, "pair_left", .{ .string = cell.left });
                        try rf.put(allocator, "pair_right", .{ .string = cell.right });
                        try rf.put(allocator, "pair_execution", .{ .string = "alternating-single-invocation" });
                        try rf.put(allocator, "mode", .{ .string = ls.mode });
                        try rf.put(allocator, "workload", .{ .string = ls.guest_workload });
                        try rf.put(allocator, "threads", .{ .integer = cell.threads });
                        try rf.put(allocator, "cancel_points", .{ .string = ls.cancel_points });
                        try rf.put(allocator, "iterations", .{ .integer = iters });
                        const freq_cpus = try spec.cpuAffinityFor(allocator, cpu_placement, cell.workload, cell.threads);
                        const before = try telemetrySnapshot(allocator, io, freq_cpus);
                        var rec = try runner.measureOnce(allocator, io, .{
                            .repo = repo,
                            .runner = runner_toks.items,
                            .cpu_placement = cpu_placement,
                            .wamr_path = b.wamr,
                            .module_path = mod_path,
                            .workload = ls.guest_workload,
                            .threads = cell.threads,
                            .iterations = @intCast(iters),
                            .timeout = cli.timeout_s,
                            .min_interval_ns = min_interval_ns,
                            .record_fields = rf,
                        });
                        const after = try telemetrySnapshot(allocator, io, freq_cpus);
                        var tel: ObjectMap = .empty;
                        try tel.put(allocator, "collection", .{ .string = "before-and-after-each-pilot-warmup-and-measured-record" });
                        try tel.put(allocator, "before", .{ .object = before });
                        try tel.put(allocator, "after", .{ .object = after });
                        try rec.put(allocator, "telemetry", .{ .object = tel });
                        try records_list.append(allocator, rec);
                        try records_arr.append(.{ .object = rec });
                    }
                }
            }
        }
    }

    const summaries = try summarizeReport(allocator, records_list.items);
    const plan_sha256 = try spec.cacheKey(allocator, .{ .object = plan });
    var logical_revisions: ObjectMap = .empty;
    var artifact_identity: ObjectMap = .empty;
    for (spec.REVISION_ROLES) |role| {
        try logical_revisions.put(allocator, role, .{ .object = source });
        try artifact_identity.put(allocator, role, .{ .object = hashes });
    }
    var builds_slice = [_]runner.Build{ builds.get("enabled-interpreter").?, builds.get("enabled-aot").? };
    var metadata: ObjectMap = .empty;
    try metadata.put(allocator, "collected_at", .{ .string = try benchmark_schema.collectedAt(allocator, io) });
    try metadata.put(allocator, "platform_id", .{ .string = platform_id });
    try metadata.put(allocator, "report_sequence", .{ .integer = report_sequence });
    try metadata.put(allocator, "partition", .{ .string = try partitionForSequence(report_sequence) });
    try metadata.put(allocator, "cohort_id", .{ .string = cli.cohort_id });
    try metadata.put(allocator, "workflow_run_id", .{ .string = environ.getPosix("GITHUB_RUN_ID") orelse "" });
    try metadata.put(allocator, "workflow_run_attempt", .{ .string = environ.getPosix("GITHUB_RUN_ATTEMPT") orelse "" });
    try metadata.put(allocator, "source_revision", .{ .object = source });
    try metadata.put(allocator, "logical_revisions", .{ .object = logical_revisions });
    try metadata.put(allocator, "artifact_identity", .{ .object = artifact_identity });
    try metadata.put(allocator, "fixture_set_sha256", .{ .string = resolved_fixtures.fixture_set_sha256 });
    try metadata.put(allocator, "fixture_source_policy", .{ .string = spec.FIXTURE_SOURCE_POLICY });
    try metadata.put(allocator, "revision_artifact_policy", .{ .string = spec.REVISION_ARTIFACT_POLICY });
    try metadata.put(allocator, "plan_sha256", .{ .string = plan_sha256 });
    try metadata.put(allocator, "plan_identity_sha256", .{ .string = try planIdentity(allocator, plan) });
    try metadata.put(allocator, "host", .{ .object = host });
    try metadata.put(allocator, "host_pair", .{ .object = try runner.hostPairIdentity(allocator, platform_id, host, host_pair_id) });
    try metadata.put(allocator, "host_quiescence_at_start", .{ .object = quiescence });
    try metadata.put(allocator, "cpu_placement", .{ .object = cpu_placement });
    try metadata.put(allocator, "tools", .{ .object = try runner.buildToolReport(allocator, io, &builds_slice, runner_toks.items) });
    try metadata.put(allocator, "fixtures", .{ .object = resolved_fixtures.fixtures });
    try metadata.put(allocator, "aot_artifacts", .{ .object = aot_meta });

    var sidecar_report: ObjectMap = .empty;
    try sidecar_report.put(allocator, "requested", .{ .bool = true });
    try sidecar_report.put(allocator, "available", .{ .bool = false });
    try sidecar_report.put(allocator, "cpu", .null);
    try sidecar_report.put(allocator, "observed_cpus", .{ .array = Array.init(allocator) });
    try sidecar_report.put(allocator, "interval_seconds", .{ .integer = 1 });
    try sidecar_report.put(allocator, "reason", .{ .string = "no CPU outside benchmark assignments" });
    try sidecar_report.put(allocator, "samples", .{ .array = Array.init(allocator) });

    var document: ObjectMap = .empty;
    try document.put(allocator, "schema_version", .{ .integer = REPORT_SCHEMA_VERSION });
    try document.put(allocator, "kind", .{ .string = KIND });
    try document.put(allocator, "authoritative", .{ .bool = false });
    try document.put(allocator, "production_budget", .null);
    try document.put(allocator, "metadata", .{ .object = metadata });
    try document.put(allocator, "plan", .{ .object = plan });
    try document.put(allocator, "quality_preflight", .{ .object = quality_preflight });
    try document.put(allocator, "records", .{ .array = records_arr });
    try document.put(allocator, "summaries", .{ .object = summaries });
    try document.put(allocator, "telemetry_sidecar", .{ .object = sidecar_report });

    try validateReport(allocator, document);
    try spec.atomicWriteJson(allocator, io, try std.fs.path.join(allocator, &.{ output, "report.json" }), .{ .object = document });
    const md = try renderMarkdown(allocator, document);
    try spec.atomicWriteText(allocator, io, try std.fs.path.join(allocator, &.{ output, "report.md" }), try std.fmt.allocPrint(allocator, "{s}\n", .{md}));
    return document;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_it.deinit();
    _ = args_it.next();

    const cli = parseCliArgs(allocator, &args_it) catch |err| {
        std.debug.print("error: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    if (cli.validate_report) |vrp| {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, vrp, allocator, .limited(64 * 1024 * 1024)) catch {
            std.debug.print("error: cannot read {s}\n", .{vrp});
            std.process.exit(2);
        };
        const parsed = std.json.parseFromSlice(Value, allocator, bytes, .{}) catch {
            std.debug.print("error: invalid JSON in {s}\n", .{vrp});
            std.process.exit(2);
        };
        if (parsed.value != .object) {
            std.debug.print("error: report must be an object\n", .{});
            std.process.exit(2);
        }
        validateReport(allocator, parsed.value.object) catch |err| {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        return;
    }
    _ = execute(allocator, io, init.minimal.environ, cli) catch |err| {
        std.debug.print("error: {s}: {s}\n", .{ @errorName(err), spec.lastErrorMessage() });
        std.process.exit(2);
    };
}

test "buildPlan and planIdentity produce deterministic schema-v21 diagnostic plan" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var source: ObjectMap = .empty;
    try source.put(allocator, "commit", .{ .string = "0123456789abcdef0123456789abcdef01234567" });
    try source.put(allocator, "tracked_diff_sha256", .{ .string = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" });
    try source.put(allocator, "build_source_sha256", .{ .string = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" });

    const plan = try buildPlan(allocator, "ubuntu-24.04-aarch64", 1, source);
    try std.testing.expectEqual(@as(i64, 21), plan.get("version").?.integer);
    try std.testing.expectEqualStrings("training", plan.get("partition").?.string);
    try std.testing.expectEqual(@as(usize, 3), plan.get("cells").?.array.items.len);

    const id1 = try planIdentity(allocator, plan);
    var mutated = plan;
    try mutated.put(allocator, "pilots", .{ .string = "ignored-in-portable-plan" });
    const id2 = try planIdentity(allocator, mutated);
    try std.testing.expectEqualStrings(id1, id2);
    _ = &main;
}

test "armOrder and warmupArmOrder alternate by sequence and index parity" {
    const a1 = try armOrder(1, 0);
    try std.testing.expectEqualStrings("current", a1[0]);
    try std.testing.expectEqualStrings("doubled", a1[1]);
    const a2 = try armOrder(1, 1);
    try std.testing.expectEqualStrings("doubled", a2[0]);
    try std.testing.expectEqualStrings("current", a2[1]);
    const w1 = try warmupArmOrder(2, 0);
    try std.testing.expectEqualStrings("doubled", w1[0]);
    try std.testing.expectEqualStrings("current", w1[1]);
}

test "resolveCounts and barrierPreflightCounts enforce sizing and hot floor rules" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var source: ObjectMap = .empty;
    try source.put(allocator, "commit", .{ .string = "0123456789abcdef0123456789abcdef01234567" });
    try source.put(allocator, "tracked_diff_sha256", .{ .string = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" });
    try source.put(allocator, "build_source_sha256", .{ .string = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" });

    const plan = try buildPlan(allocator, "ubuntu-24.04-aarch64", 1, source);
    const cells = try cellsForPlatform("ubuntu-24.04-aarch64");
    var pilots: std.ArrayList(ObjectMap) = .empty;
    for (cells, 0..) |cell, c_idx| {
        for ([_][]const u8{ cell.left, cell.right }, 0..) |cond, cond_idx| {
            const ls = try legSpec(cell, cond);
            const iters = try pilotIterations(allocator, ls, cell.threads);
            const floor = sizing.projectedDurationFloorForCell(ls.mode, ls.sizing_policy_workload, cell.threads);
            const clock_id = try spec.expectedGuestClockId(ls.guest_workload);
            const host_wall = if (std.mem.eql(u8, clock_id, spec.WASI_MONOTONIC_CLOCK_ID))
                floor
            else
                @max(@as(i64, 1_000_000_000), @divTrunc(floor, cell.threads));
            var p: ObjectMap = .empty;
            const exp_res = try spec.expectedResult(ls.guest_workload, cell.threads, @as(i128, iters), null);
            try p.put(allocator, "pilot_index", .{ .integer = @intCast(c_idx * 2 + cond_idx) });
            try p.put(allocator, "revision", .{ .string = "baseline" });
            try p.put(allocator, "pair_kind", .{ .string = cell.pair_kind });
            try p.put(allocator, "pair_key", .{ .string = cell.pair_key });
            try p.put(allocator, "condition", .{ .string = cond });
            try p.put(allocator, "mode", .{ .string = ls.mode });
            try p.put(allocator, "workload", .{ .string = ls.sizing_workload });
            try p.put(allocator, "sizing_policy_workload", .{ .string = ls.sizing_policy_workload });
            try p.put(allocator, "threads", .{ .integer = cell.threads });
            try p.put(allocator, "iterations", .{ .integer = iters });
            try spec.putU64Json(allocator, &p, "operations", exp_res.operations);
            try p.put(allocator, "clock_id", .{ .string = clock_id });
            try p.put(allocator, "guest_elapsed_ns", .{ .integer = floor });
            try p.put(allocator, "raw_guest_elapsed_ns", .{ .integer = floor });
            try p.put(allocator, "timing_overhead_ns", .{ .integer = 0 });
            try p.put(allocator, "host_wall_elapsed_ns", .{ .integer = host_wall });
            try p.put(allocator, "correct", .{ .bool = true });
            try pilots.append(allocator, p);
        }
    }
    const resolved = try resolveCounts(allocator, plan, pilots.items, INVOCATION_TIMEOUT_SECONDS);
    const bp_counts = try barrierPreflightCounts(allocator, plan, resolved.counts, &[_]i64{ 2, 4, 8 });
    try std.testing.expect(bp_counts.get("4").?.integer >= 900_000_000);
}
