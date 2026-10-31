//! Plan, dispatch, download, validate, and analyze duration-cross diagnostic cohorts.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const ObjectMap = std.json.ObjectMap;
const Array = std.json.Array;

const benchmark_schema = @import("benchmark_schema.zig");
const spec = @import("wasi_thread_bench_spec.zig");
const cohort = @import("wasi_thread_cohort.zig");
const diag = @import("wasi_thread_duration_cross.zig");

pub const HarnessError = spec.HarnessError;
const fail = spec.fail;
const require = spec.require;

pub const DISPATCH_KIND = "wasi-thread-duration-cross-dispatch";
pub const COHORT_KIND = "wasi-thread-duration-cross-cohort";
pub const CONCLUSION_KIND = "wasi-thread-duration-cross-conclusion";
pub const DOWNLOAD_MANIFEST_KIND = "wasi-thread-duration-cross-download-manifest";
pub const COHORT_SCHEMA_VERSION: i64 = 1;
pub const WORKFLOW_RUN_COUNT: i64 = 20;
pub const ACCEPTANCE_CEILING_LOG: f64 = 0.10;
pub const WORKFLOW_NAME = "wasi-thread-duration-cross.yml";
pub const DEFAULT_POLICY_PATH = "tests/benchmarks/wasi-threads/derivation-policy.production.json";
pub const READINESS_ARTIFACTS = [_][]const u8{
    "wasi-thread-duration-cross-readiness-",
    "wasi-thread-duration-cross-readiness-arm-",
};

fn isHexLower(value: []const u8, expected_len: usize) bool {
    if (value.len != expected_len) return false;
    for (value) |ch| {
        if (!((ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f'))) return false;
    }
    return true;
}

fn requireHex(value: []const u8, expected_len: usize, label: []const u8) HarnessError![]const u8 {
    if (!isHexLower(value, expected_len)) {
        return fail("{s} must be {d} lowercase hexadecimal characters", .{ label, expected_len });
    }
    return value;
}

pub fn planCohort(
    allocator: Allocator,
    source_sha: []const u8,
    calibration_ref: []const u8,
    cohort_id: []const u8,
    readiness_runs_val: ?Value,
) HarnessError!ObjectMap {
    _ = try requireHex(source_sha, 40, "source_sha");
    _ = try requireHex(cohort_id, 32, "cohort_id");
    if (!std.mem.startsWith(u8, calibration_ref, "wasi-thread-duration-cross-")) {
        return fail("calibration_ref must start with 'wasi-thread-duration-cross-'", .{});
    }

    var readiness_obj: ObjectMap = .empty;
    if (readiness_runs_val) |rv| {
        if (rv != .object) return fail("readiness_runs must be an object", .{});
        readiness_obj = rv.object;
    } else {
        var r_arr = Array.init(allocator);
        try r_arr.append(.{ .string = "x86 self-hosted gate-readiness workflow run must succeed on source_sha" });
        try r_arr.append(.{ .string = "Arm gate-readiness workflow run must succeed on source_sha" });
        try readiness_obj.put(allocator, "required_before_dispatch", .{ .array = r_arr });
        try readiness_obj.put(allocator, "x86_run_id", .null);
        try readiness_obj.put(allocator, "arm_run_id", .null);
    }

    var runs_arr = Array.init(allocator);
    for (1..@as(usize, @intCast(WORKFLOW_RUN_COUNT + 1))) |seq_u| {
        const sequence: i64 = @intCast(seq_u);
        const partition = try diag.partitionForSequence(sequence);
        var exp_arts: ObjectMap = .empty;
        try exp_arts.put(allocator, "ubuntu-22.04-x86_64", .{ .string = "wasi-thread-duration-cross-ubuntu-22.04-x86_64-<run_id>-1" });
        try exp_arts.put(allocator, "ubuntu-24.04-aarch64", .{ .string = "wasi-thread-duration-cross-ubuntu-24.04-aarch64-<run_id>-1" });

        var run_obj: ObjectMap = .empty;
        try run_obj.put(allocator, "sequence", .{ .integer = sequence });
        try run_obj.put(allocator, "partition", .{ .string = partition });
        try run_obj.put(allocator, "status", .{ .string = "planned" });
        try run_obj.put(allocator, "run_id", .null);
        try run_obj.put(allocator, "dispatched_at", .null);
        try run_obj.put(allocator, "completed_at", .null);
        try run_obj.put(allocator, "conclusion", .null);
        try run_obj.put(allocator, "expected_artifacts", .{ .object = exp_arts });
        try runs_arr.append(.{ .object = run_obj });
    }

    var part_counts: ObjectMap = .empty;
    try part_counts.put(allocator, "training", .{ .integer = 16 });
    try part_counts.put(allocator, "holdout", .{ .integer = 4 });

    var out: ObjectMap = .empty;
    try out.put(allocator, "schema_version", .{ .integer = COHORT_SCHEMA_VERSION });
    try out.put(allocator, "kind", .{ .string = DISPATCH_KIND });
    try out.put(allocator, "authoritative", .{ .bool = false });
    try out.put(allocator, "production_budget_eligible", .{ .bool = false });
    try out.put(allocator, "workflow", .{ .string = WORKFLOW_NAME });
    try out.put(allocator, "cohort_id", .{ .string = cohort_id });
    try out.put(allocator, "source_sha", .{ .string = source_sha });
    try out.put(allocator, "calibration_ref", .{ .string = calibration_ref });
    try out.put(allocator, "run_count", .{ .integer = WORKFLOW_RUN_COUNT });
    try out.put(allocator, "partition_counts", .{ .object = part_counts });
    try out.put(allocator, "readiness_gate", .{ .object = readiness_obj });
    try out.put(allocator, "runs", .{ .array = runs_arr });
    return out;
}

pub fn validateDispatchDocument(state: ObjectMap) HarnessError!void {
    const sv = state.get("schema_version") orelse return fail("dispatch schema_version", .{});
    const kind = state.get("kind") orelse return fail("dispatch kind", .{});
    const auth = state.get("authoritative") orelse return fail("dispatch authoritative must be false", .{});
    const pbe = state.get("production_budget_eligible") orelse return fail("dispatch production_budget_eligible must be false", .{});
    const wf = state.get("workflow") orelse return fail("dispatch workflow", .{});
    if (sv != .integer or sv.integer != COHORT_SCHEMA_VERSION or
        kind != .string or !std.mem.eql(u8, kind.string, DISPATCH_KIND) or
        auth != .bool or auth.bool or
        pbe != .bool or pbe.bool or
        wf != .string or !std.mem.eql(u8, wf.string, WORKFLOW_NAME))
    {
        return fail("invalid dispatch document header", .{});
    }
    _ = try requireHex(state.get("cohort_id").?.string, 32, "cohort_id");
    _ = try requireHex(state.get("source_sha").?.string, 40, "source_sha");
    const cal_ref = state.get("calibration_ref").?.string;
    if (!std.mem.startsWith(u8, cal_ref, "wasi-thread-duration-cross-")) {
        return fail("calibration_ref", .{});
    }
    if (state.get("run_count").?.integer != WORKFLOW_RUN_COUNT) return fail("run_count", .{});
    const pc = state.get("partition_counts").?.object;
    if (pc.get("training").?.integer != 16 or pc.get("holdout").?.integer != 4) {
        return fail("partition_counts", .{});
    }
    const runs = state.get("runs").?.array.items;
    if (runs.len != @as(usize, @intCast(WORKFLOW_RUN_COUNT))) return fail("runs length", .{});
    for (runs, 1..) |run_v, expected_seq_u| {
        const expected_seq: i64 = @intCast(expected_seq_u);
        if (run_v != .object) return fail("run record must be an object", .{});
        const run_o = run_v.object;
        if (run_o.get("sequence").?.integer != expected_seq) return fail("run sequence", .{});
        const exp_part = try diag.partitionForSequence(expected_seq);
        if (!std.mem.eql(u8, run_o.get("partition").?.string, exp_part)) return fail("run partition", .{});
        const status = run_o.get("status").?.string;
        if (!std.mem.eql(u8, status, "planned") and !std.mem.eql(u8, status, "completed")) {
            return fail("run status", .{});
        }
        if (std.mem.eql(u8, status, "completed")) {
            const rid = run_o.get("run_id") orelse return fail("completed run_id", .{});
            if (rid != .integer or rid.integer <= 0) return fail("completed run_id", .{});
            const conc = run_o.get("conclusion") orelse return fail("completed conclusion", .{});
            if (conc != .string or !std.mem.eql(u8, conc.string, "success")) return fail("completed conclusion", .{});
        }
    }
}

pub fn validateRunReports(
    allocator: Allocator,
    dispatch_state: ObjectMap,
    reports_by_platform: ObjectMap,
) HarnessError!ObjectMap {
    try validateDispatchDocument(dispatch_state);
    const source_sha = dispatch_state.get("source_sha").?.string;
    const cohort_id = dispatch_state.get("cohort_id").?.string;

    var summary_platforms: ObjectMap = .empty;
    for (cohort.DEFAULT_PLATFORMS) |platform_id| {
        const p_reports_v = reports_by_platform.get(platform_id) orelse return fail("reports_by_platform must provide 20 reports per platform", .{});
        if (p_reports_v != .array or p_reports_v.array.items.len != @as(usize, @intCast(WORKFLOW_RUN_COUNT))) {
            return fail("reports_by_platform must provide 20 reports per platform", .{});
        }
        const platform_reports = p_reports_v.array.items;

        var seen_runs: std.ArrayList(i64) = .empty;
        var seen_host_pairs: std.ArrayList([]const u8) = .empty;
        var cpu_classes: std.ArrayList([]const u8) = .empty;
        var artifact_bundles: std.ArrayList([64]u8) = .empty;
        var sidecar_available_count: i64 = 0;

        for (dispatch_state.get("runs").?.array.items, platform_reports) |run_v, report_v| {
            const run_o = run_v.object;
            if (!std.mem.eql(u8, run_o.get("status").?.string, "completed")) {
                return fail("{s}: sequence {d} is not completed", .{ platform_id, run_o.get("sequence").?.integer });
            }
            if (report_v != .object) return fail("{s}: report must be an object", .{platform_id});
            const report = report_v.object;
            try diag.validateReport(allocator, report);
            const metadata = report.get("metadata").?.object;
            if (!std.mem.eql(u8, metadata.get("platform_id").?.string, platform_id)) return fail("{s}: platform_id mismatch", .{platform_id});
            if (metadata.get("report_sequence").?.integer != run_o.get("sequence").?.integer) return fail("{s}: report_sequence mismatch", .{platform_id});
            if (!std.mem.eql(u8, metadata.get("partition").?.string, run_o.get("partition").?.string)) return fail("{s}: partition mismatch", .{platform_id});
            if (!std.mem.eql(u8, metadata.get("cohort_id").?.string, cohort_id)) return fail("{s}: cohort_id mismatch", .{platform_id});
            if (!std.mem.eql(u8, metadata.get("source_revision").?.object.get("commit").?.string, source_sha)) return fail("{s}: source commit mismatch", .{platform_id});

            const wf_run = metadata.get("workflow_run").?.object;
            const run_id = wf_run.get("run_id").?.integer;
            const run_attempt = wf_run.get("run_attempt").?.integer;
            if (run_id != run_o.get("run_id").?.integer or run_attempt != 1) {
                return fail("{s}: workflow run/attempt mismatch", .{platform_id});
            }
            for (seen_runs.items) |sr| {
                if (sr == run_id) return fail("{s}: duplicate workflow run_id", .{platform_id});
            }
            try seen_runs.append(allocator, run_id);

            const hp_id = metadata.get("host_pair_id").?.string;
            for (seen_host_pairs.items) |shp| {
                if (std.mem.eql(u8, shp, hp_id)) return fail("{s}: duplicate host_pair_id", .{platform_id});
            }
            try seen_host_pairs.append(allocator, hp_id);

            const cpu_model = report.get("environment").?.object.get("cpu_model").?.string;
            var found_cpu = false;
            for (cpu_classes.items) |cc| {
                if (std.mem.eql(u8, cc, cpu_model)) {
                    found_cpu = true;
                    break;
                }
            }
            if (!found_cpu) try cpu_classes.append(allocator, cpu_model);

            var art_wrapper: ObjectMap = .empty;
            try art_wrapper.put(allocator, "runtime", report.get("runtime").?);
            try art_wrapper.put(allocator, "prepared_fixtures", report.get("prepared_fixtures").?);
            const bundle_sha = benchmark_schema.cacheKey(allocator, .{ .object = art_wrapper }) catch return error.BenchmarkDataError;
            var found_bundle = false;
            for (artifact_bundles.items) |ab| {
                if (std.mem.eql(u8, &ab, &bundle_sha)) {
                    found_bundle = true;
                    break;
                }
            }
            if (!found_bundle) try artifact_bundles.append(allocator, bundle_sha);

            if (report.get("telemetry_sidecar").?.object.get("available").?.bool) {
                sidecar_available_count += 1;
            }
        }

        if (cpu_classes.items.len != 1) return fail("{s}: mixed CPU classes", .{platform_id});
        if (artifact_bundles.items.len != 1) return fail("{s}: mixed runtime/prepared-fixture hashes", .{platform_id});

        var plat_sum: ObjectMap = .empty;
        try plat_sum.put(allocator, "report_count", .{ .integer = WORKFLOW_RUN_COUNT });
        try plat_sum.put(allocator, "training_count", .{ .integer = 16 });
        try plat_sum.put(allocator, "holdout_count", .{ .integer = 4 });
        try plat_sum.put(allocator, "cpu_model", .{ .string = cpu_classes.items[0] });
        try plat_sum.put(allocator, "artifact_bundle_sha256", .{ .string = try allocator.dupe(u8, &artifact_bundles.items[0]) });
        try plat_sum.put(allocator, "sidecar_available_reports", .{ .integer = sidecar_available_count });
        try summary_platforms.put(allocator, platform_id, .{ .object = plat_sum });
    }

    var out: ObjectMap = .empty;
    try out.put(allocator, "schema_version", .{ .integer = COHORT_SCHEMA_VERSION });
    try out.put(allocator, "kind", .{ .string = COHORT_KIND });
    try out.put(allocator, "authoritative", .{ .bool = false });
    try out.put(allocator, "production_budget_eligible", .{ .bool = false });
    try out.put(allocator, "cohort_id", .{ .string = cohort_id });
    try out.put(allocator, "source_sha", .{ .string = source_sha });
    try out.put(allocator, "calibration_ref", dispatch_state.get("calibration_ref").?);
    try out.put(allocator, "platforms", .{ .object = summary_platforms });
    return out;
}

fn findSummaryEntry(
    report: ObjectMap,
    arm: []const u8,
    collection: []const u8,
    k0: []const u8,
    k1: []const u8,
    k2: []const u8,
) HarnessError!struct { entry: ObjectMap, field: []const u8 } {
    const sums = report.get("summaries").?.object;
    const items = sums.get(collection).?.array.items;
    if (std.mem.eql(u8, collection, "comparisons")) {
        for (items) |it_v| {
            const it = it_v.object;
            if (std.mem.eql(u8, it.get("arm").?.string, arm) and
                std.mem.eql(u8, it.get("pair_key").?.string, k0) and
                std.mem.eql(u8, it.get("condition").?.string, k1) and
                std.mem.eql(u8, it.get("metric_kind").?.string, k2))
            {
                return .{ .entry = it, .field = "elapsed_candidate_over_baseline" };
            }
        }
    } else if (std.mem.eql(u8, collection, "ratio_of_ratios")) {
        for (items) |it_v| {
            const it = it_v.object;
            if (std.mem.eql(u8, it.get("arm").?.string, arm) and
                std.mem.eql(u8, it.get("pair_key").?.string, k0) and
                std.mem.eql(u8, it.get("left").?.string, k1) and
                std.mem.eql(u8, it.get("right").?.string, k2))
            {
                return .{ .entry = it, .field = "elapsed_ratio_of_ratios" };
            }
        }
    }
    return fail("missing {s} summary for {s} ({s}, {s}, {s})", .{ collection, arm, k0, k1, k2 });
}

fn evaluateArm(
    allocator: Allocator,
    platform_id: []const u8,
    arm: []const u8,
    reports: []const Value,
    policy: ObjectMap,
    select_thresholds: bool,
) HarnessError!ObjectMap {
    var training_reports: std.ArrayList(ObjectMap) = .empty;
    var holdout_reports: std.ArrayList(ObjectMap) = .empty;
    for (reports) |rv| {
        const rep = rv.object;
        const part = rep.get("metadata").?.object.get("partition").?.string;
        if (std.mem.eql(u8, part, "training")) {
            try training_reports.append(allocator, rep);
        } else if (std.mem.eql(u8, part, "holdout")) {
            try holdout_reports.append(allocator, rep);
        }
    }
    if (training_reports.items.len != 16 or holdout_reports.items.len != 4) {
        return fail("{s}: expected 16 training and 4 holdout reports", .{platform_id});
    }

    const checks_arr = try diag.acceptanceChecksForPlatform(allocator, platform_id);
    var checks_out = Array.init(allocator);
    var ceiling_failures = Array.init(allocator);
    var holdout_failures = Array.init(allocator);

    for (checks_arr.items) |chk_v| {
        const chk = chk_v.object;
        const coll = chk.get("collection").?.string;
        const key_arr = chk.get("key").?.array.items;
        const k0 = key_arr[0].string;
        const k1 = key_arr[1].string;
        const k2 = key_arr[2].string;

        var train_tp: [16]f64 = undefined;
        var train_el: [16]f64 = undefined;
        var hold_tp: [4]f64 = undefined;
        var hold_el: [4]f64 = undefined;
        const tp_field = if (std.mem.eql(u8, coll, "comparisons")) "throughput_candidate_over_baseline" else "throughput_ratio_of_ratios";
        const el_field = if (std.mem.eql(u8, coll, "comparisons")) "elapsed_candidate_over_baseline" else "elapsed_ratio_of_ratios";

        for (training_reports.items, 0..) |rep, idx| {
            const found = try findSummaryEntry(rep, arm, coll, k0, k1, k2);
            const tp_stats = found.entry.get(tp_field).?.object;
            const el_stats = found.entry.get(el_field).?.object;
            train_tp[idx] = spec.valueAsF64(tp_stats.get("median").?).?;
            train_el[idx] = spec.valueAsF64(el_stats.get("median").?).?;
        }
        for (holdout_reports.items, 0..) |rep, idx| {
            const found = try findSummaryEntry(rep, arm, coll, k0, k1, k2);
            const tp_stats = found.entry.get(tp_field).?.object;
            const el_stats = found.entry.get(el_field).?.object;
            hold_tp[idx] = spec.valueAsF64(tp_stats.get("median").?).?;
            hold_el[idx] = spec.valueAsF64(el_stats.get("median").?).?;
        }

        const cushion = spec.valueAsF64(policy.get("rounding_cushion_log").?).?;
        var tp_rule = try cohort.deriveOneSidedThreshold(allocator, &train_tp, "lower", cushion, 1e9, "throughput");
        var el_rule = try cohort.deriveOneSidedThreshold(allocator, &train_el, "upper", cushion, 1e9, "elapsed");
        const tp_bound = spec.valueAsF64(tp_rule.get("final_bound_log").?).?;
        const el_bound = spec.valueAsF64(el_rule.get("final_bound_log").?).?;
        const tp_thresh = spec.valueAsF64(tp_rule.get("threshold_ratio").?).?;
        const el_thresh = spec.valueAsF64(el_rule.get("threshold_ratio").?).?;

        const ceiling_passed = (tp_bound <= ACCEPTANCE_CEILING_LOG) and (el_bound <= ACCEPTANCE_CEILING_LOG);
        var holdout_passed = true;
        var holdout_max: f64 = 0.0;
        for (hold_tp, hold_el) |htp, hel| {
            if (htp < tp_thresh or hel > el_thresh) holdout_passed = false;
            const max_dev = @max(@abs(@log(htp)), @abs(@log(hel)));
            if (max_dev > holdout_max) holdout_max = max_dev;
        }
        try tp_rule.put(allocator, "engineering_policy_ceiling_log", .{ .float = ACCEPTANCE_CEILING_LOG });
        try tp_rule.put(allocator, "engineering_ceiling_passed", .{ .bool = tp_bound <= ACCEPTANCE_CEILING_LOG });
        try el_rule.put(allocator, "engineering_policy_ceiling_log", .{ .float = ACCEPTANCE_CEILING_LOG });
        try el_rule.put(allocator, "engineering_ceiling_passed", .{ .bool = el_bound <= ACCEPTANCE_CEILING_LOG });

        var rec: ObjectMap = .empty;
        try rec.put(allocator, "collection", .{ .string = coll });
        try rec.put(allocator, "key", chk.get("key").?);
        try rec.put(allocator, "selects_threshold", .{ .bool = select_thresholds });
        try rec.put(allocator, "holdout_max_abs_log", .{ .float = holdout_max });
        try rec.put(allocator, "acceptance_ceiling_log", .{ .float = ACCEPTANCE_CEILING_LOG });
        try rec.put(allocator, "throughput_rule", .{ .object = tp_rule });
        try rec.put(allocator, "elapsed_rule", .{ .object = el_rule });

        if (select_thresholds) {
            var d_obj: ObjectMap = .empty;
            try d_obj.put(allocator, "log_threshold", .{ .float = @max(tp_bound, el_bound) });
            try rec.put(allocator, "derived", .{ .object = d_obj });
            try rec.put(allocator, "ceiling_passed", .{ .bool = ceiling_passed });
            try rec.put(allocator, "holdout_passed", .{ .bool = holdout_passed });
            if (!ceiling_passed) try ceiling_failures.append(.{ .object = rec });
            if (!holdout_passed) try holdout_failures.append(.{ .object = rec });
        } else {
            try rec.put(allocator, "derived", .null);
            try rec.put(allocator, "ceiling_passed", .null);
            try rec.put(allocator, "holdout_passed", .null);
        }
        try checks_out.append(.{ .object = rec });
    }

    var out: ObjectMap = .empty;
    try out.put(allocator, "arm", .{ .string = arm });
    try out.put(allocator, "selects_thresholds", .{ .bool = select_thresholds });
    try out.put(allocator, "checks", .{ .array = checks_out });
    try out.put(allocator, "ceiling_failure_count", if (select_thresholds) .{ .integer = @intCast(ceiling_failures.items.len) } else .null);
    try out.put(allocator, "holdout_failure_count", if (select_thresholds) .{ .integer = @intCast(holdout_failures.items.len) } else .null);
    const passed_val: Value = if (select_thresholds)
        .{ .bool = (ceiling_failures.items.len == 0 and holdout_failures.items.len == 0) }
    else
        .null;
    try out.put(allocator, "passed", passed_val);
    return out;
}

pub fn analyzeCohort(
    allocator: Allocator,
    dispatch_state: ObjectMap,
    reports_by_platform: ObjectMap,
    policy: ObjectMap,
) HarnessError!ObjectMap {
    _ = try cohort.validateDerivationPolicy(allocator, policy);
    const cohort_summary = try validateRunReports(allocator, dispatch_state, reports_by_platform);

    var platform_results: ObjectMap = .empty;
    var all_passed = true;
    for (cohort.DEFAULT_PLATFORMS) |platform_id| {
        const reports = reports_by_platform.get(platform_id).?.array.items;
        const current = try evaluateArm(allocator, platform_id, "current", reports, policy, false);
        const doubled = try evaluateArm(allocator, platform_id, "doubled", reports, policy, true);
        const plat_passed = doubled.get("passed").?.bool;
        all_passed = all_passed and plat_passed;

        var pr_obj: ObjectMap = .empty;
        try pr_obj.put(allocator, "current_descriptive_only", .{ .object = current });
        try pr_obj.put(allocator, "doubled_acceptance", .{ .object = doubled });
        try pr_obj.put(allocator, "passed", .{ .bool = plat_passed });
        try platform_results.put(allocator, platform_id, .{ .object = pr_obj });
    }

    const decision = if (all_passed) "admit-doubled-durations" else "close-without-production-budget-activation";
    const next_action = if (all_passed)
        "Cut a separate production measurement-plan bump for doubled failing-cell durations and collect a fresh 20-pair production calibration cohort."
    else
        "Keep WASI thread benchmark budgets disabled (enforced=false) on current runner classes; do not activate production budgets from this diagnostic cohort.";

    var pol_info: ObjectMap = .empty;
    try pol_info.put(allocator, "policy_version", policy.get("policy_version").?);
    try pol_info.put(allocator, "estimator", policy.get("estimator").?);
    try pol_info.put(allocator, "acceptance_ceiling_log", .{ .float = ACCEPTANCE_CEILING_LOG });
    try pol_info.put(allocator, "selection_arm", .{ .string = "doubled" });
    try pol_info.put(allocator, "current_arm_role", .{ .string = "descriptive-only-never-selects-thresholds" });

    var out: ObjectMap = .empty;
    try out.put(allocator, "schema_version", .{ .integer = COHORT_SCHEMA_VERSION });
    try out.put(allocator, "kind", .{ .string = CONCLUSION_KIND });
    try out.put(allocator, "authoritative_for_production_budgets", .{ .bool = false });
    try out.put(allocator, "production_budget_Update", .{ .string = "forbidden" });
    _ = out.swapRemove("production_budget_Update");
    try out.put(allocator, "production_budget_update", .{ .string = "forbidden" });
    try out.put(allocator, "decision", .{ .string = decision });
    try out.put(allocator, "next_action", .{ .string = next_action });
    try out.put(allocator, "policy", .{ .object = pol_info });
    try out.put(allocator, "cohort", .{ .object = cohort_summary });
    try out.put(allocator, "platforms", .{ .object = platform_results });
    return out;
}

pub fn renderAnalysisMarkdown(allocator: Allocator, analysis: ObjectMap) HarnessError![]const u8 {
    var md: std.ArrayList(u8) = .empty;
    try md.appendSlice(allocator, "# WASI thread duration-cross cohort conclusion\n\n");
    try md.appendSlice(allocator, "> Non-authoritative diagnostic conclusion. This artifact does not activate or modify `tests/benchmarks/wasi-threads/budgets.json`.\n\n");
    try md.print(allocator, "- Decision: `{s}`\n", .{analysis.get("decision").?.string});
    try md.print(allocator, "- Next action: {s}\n", .{analysis.get("next_action").?.string});
    const coh = analysis.get("cohort").?.object;
    try md.print(allocator, "- Source SHA: `{s}`\n", .{coh.get("source_sha").?.string});
    try md.print(allocator, "- Cohort ID: `{s}`\n\n", .{coh.get("cohort_id").?.string});

    const platforms = analysis.get("platforms").?.object;
    for (cohort.DEFAULT_PLATFORMS) |platform_id| {
        const plat = platforms.get(platform_id).?.object;
        const doubled = plat.get("doubled_acceptance").?.object;
        try md.print(allocator, "## `{s}`\n\n", .{platform_id});
        try md.print(allocator, "- Doubled-arm acceptance passed: `{s}`\n", .{if (plat.get("passed").?.bool) "True" else "False"});
        try md.print(allocator, "- Ceiling failures (`log_threshold > 0.10`): `{d}`\n", .{doubled.get("ceiling_failure_count").?.integer});
        try md.print(allocator, "- Holdout failures (`holdout_max > log_threshold`): `{d}`\n\n", .{doubled.get("holdout_failure_count").?.integer});
        try md.appendSlice(allocator, "| Collection | Key | Derived log threshold | Holdout max abs log | Ceiling <= 0.10 | Holdout <= threshold |\n");
        try md.appendSlice(allocator, "|---|---|---:|---:|---|---|\n");
        for (doubled.get("checks").?.array.items) |chk_v| {
            const chk = chk_v.object;
            const derived = chk.get("derived").?.object;
            const key_items = chk.get("key").?.array.items;
            const log_thr = spec.valueAsF64(derived.get("log_threshold").?).?;
            const h_max = spec.valueAsF64(chk.get("holdout_max_abs_log").?).?;
            try md.print(allocator, "| `{s}` | `{s} / {s} / {s}` | {d:.6} | {d:.6} | `{s}` | `{s}` |\n", .{
                chk.get("collection").?.string,
                key_items[0].string,
                key_items[1].string,
                key_items[2].string,
                log_thr,
                h_max,
                if (chk.get("ceiling_passed").?.bool) "True" else "False",
                if (chk.get("holdout_passed").?.bool) "True" else "False",
            });
        }
        try md.appendSlice(allocator, "\n");
    }
    return try md.toOwnedSlice(allocator);
}

fn loadPlatformReportsFromDir(
    allocator: Allocator,
    io: std.Io,
    cohort_dir: []const u8,
) HarnessError!struct { dispatch_state: ObjectMap, reports_by_platform: ObjectMap } {
    const state_path = try std.fmt.allocPrint(allocator, "{s}/dispatch-state.json", .{cohort_dir});
    const dispatch_state = try cohort.loadJsonObject(allocator, io, state_path, "dispatch-state.json");
    try validateDispatchDocument(dispatch_state);

    var reports_by_platform: ObjectMap = .empty;
    for (cohort.DEFAULT_PLATFORMS) |platform_id| {
        var plat_reports = Array.init(allocator);
        for (dispatch_state.get("runs").?.array.items) |run_v| {
            const run_o = run_v.object;
            const seq: usize = @intCast(run_o.get("sequence").?.integer);
            const run_id = run_o.get("run_id").?.integer;
            const report_path = try std.fmt.allocPrint(
                allocator,
                "{s}/runs/{d:0>2}-{d}/wasi-thread-duration-cross-{s}-{d}-1/report.json",
                .{ cohort_dir, seq, run_id, platform_id, run_id },
            );
            const rep_obj = try cohort.loadJsonObject(allocator, io, report_path, "report.json");
            try plat_reports.append(.{ .object = rep_obj });
        }
        try reports_by_platform.put(allocator, platform_id, .{ .array = plat_reports });
    }
    return .{ .dispatch_state = dispatch_state, .reports_by_platform = reports_by_platform };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_it.deinit();
    _ = args_it.next();

    const subcmd = args_it.next() orelse {
        std.debug.print("usage: wasi_thread_duration_cross_cohort <plan|dispatch|download|validate|analyze> [options]\n", .{});
        std.process.exit(2);
    };

    if (std.mem.eql(u8, subcmd, "plan")) {
        var source_sha: []const u8 = "";
        var calibration_ref: []const u8 = "";
        var cohort_id: []const u8 = "";
        var output_path: []const u8 = "";
        var x86_readiness_run_id: ?i64 = null;
        var arm_readiness_run_id: ?i64 = null;
        while (args_it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--source-sha")) {
                source_sha = args_it.next() orelse std.process.exit(2);
            } else if (std.mem.eql(u8, arg, "--calibration-ref")) {
                calibration_ref = args_it.next() orelse std.process.exit(2);
            } else if (std.mem.eql(u8, arg, "--cohort-id")) {
                cohort_id = args_it.next() orelse std.process.exit(2);
            } else if (std.mem.eql(u8, arg, "--output")) {
                output_path = args_it.next() orelse std.process.exit(2);
            } else if (std.mem.eql(u8, arg, "--x86-readiness-run-id")) {
                x86_readiness_run_id = std.fmt.parseInt(i64, args_it.next() orelse std.process.exit(2), 10) catch std.process.exit(2);
            } else if (std.mem.eql(u8, arg, "--arm-readiness-run-id")) {
                arm_readiness_run_id = std.fmt.parseInt(i64, args_it.next() orelse std.process.exit(2), 10) catch std.process.exit(2);
            }
        }
        var readiness: ObjectMap = .empty;
        try readiness.put(allocator, "x86_run_id", if (x86_readiness_run_id) |id| .{ .integer = id } else .null);
        try readiness.put(allocator, "arm_run_id", if (arm_readiness_run_id) |id| .{ .integer = id } else .null);
        const doc = planCohort(allocator, source_sha, calibration_ref, cohort_id, .{ .object = readiness }) catch |err| {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        try spec.atomicWriteJson(allocator, io, output_path, .{ .object = doc });
    } else if (std.mem.eql(u8, subcmd, "validate")) {
        var cohort_dir: []const u8 = "";
        while (args_it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--cohort-dir")) {
                cohort_dir = args_it.next() orelse std.process.exit(2);
            }
        }
        const loaded = loadPlatformReportsFromDir(allocator, io, cohort_dir) catch |err| {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        const summary = validateRunReports(allocator, loaded.dispatch_state, loaded.reports_by_platform) catch |err| {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        const out_json = try benchmark_schema.canonicalJson(allocator, .{ .object = summary });
        var stdout_buf: [4096]u8 = undefined;
        var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);
        try stdout_writer.interface.print("{s}\n", .{out_json});
        try stdout_writer.interface.flush();
    } else if (std.mem.eql(u8, subcmd, "analyze")) {
        var cohort_dir: []const u8 = "";
        var policy_path: []const u8 = DEFAULT_POLICY_PATH;
        var output_json: ?[]const u8 = null;
        var output_md: ?[]const u8 = null;
        while (args_it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--cohort-dir")) {
                cohort_dir = args_it.next() orelse std.process.exit(2);
            } else if (std.mem.eql(u8, arg, "--policy")) {
                policy_path = args_it.next() orelse std.process.exit(2);
            } else if (std.mem.eql(u8, arg, "--output-json")) {
                output_json = args_it.next() orelse std.process.exit(2);
            } else if (std.mem.eql(u8, arg, "--output-md")) {
                output_md = args_it.next() orelse std.process.exit(2);
            }
        }
        const loaded = loadPlatformReportsFromDir(allocator, io, cohort_dir) catch |err| {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        const policy_obj = cohort.loadJsonObject(allocator, io, policy_path, "policy") catch |err| {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        const analysis = analyzeCohort(allocator, loaded.dispatch_state, loaded.reports_by_platform, policy_obj) catch |err| {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        const md = renderAnalysisMarkdown(allocator, analysis) catch |err| {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        const j_path = output_json orelse try std.fmt.allocPrint(allocator, "{s}/conclusion.json", .{cohort_dir});
        const m_path = output_md orelse try std.fmt.allocPrint(allocator, "{s}/conclusion.md", .{cohort_dir});
        try spec.atomicWriteJson(allocator, io, j_path, .{ .object = analysis });
        try spec.atomicWriteText(allocator, io, m_path, md);
    } else {
        std.debug.print("unsupported subcommand: {s}\n", .{subcmd});
        std.process.exit(2);
    }
}

test "planCohort and validateDispatchDocument enforce 16 training + 4 holdout runs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const doc = try planCohort(
        allocator,
        "0123456789abcdef0123456789abcdef01234567",
        "wasi-thread-duration-cross-20260417",
        "0123456789abcdef0123456789abcdef",
        null,
    );
    try validateDispatchDocument(doc);
    try std.testing.expectEqual(@as(usize, 20), doc.get("runs").?.array.items.len);
    _ = &main;
}
