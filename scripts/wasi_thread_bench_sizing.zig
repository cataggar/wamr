//! One-shot pilot sizing, iteration plan resolution, and pair specifications for WASI thread benchmarks.
const std = @import("std");
const spec = @import("wasi_thread_bench_spec.zig");

pub const Allocator = spec.Allocator;
pub const Value = spec.Value;
pub const ObjectMap = spec.ObjectMap;
pub const Array = spec.Array;
pub const HarnessError = spec.HarnessError;
const fail = spec.fail;
const failData = spec.failData;
const require = spec.require;

pub const SIZING_WORKLOAD_CAPS = struct {
    pub fn get(workload: []const u8) ?u64 {
        if (std.mem.eql(u8, workload, "single-hot")) return 16_000_000_000;
        if (std.mem.eql(u8, workload, "hot")) return 16_000_000_000;
        if (std.mem.eql(u8, workload, "cancel-hot")) return 16_000_000_000;
        if (std.mem.eql(u8, workload, "atomic")) return 8_000_000_000;
        if (std.mem.eql(u8, workload, "wait-notify")) return 100_000_000;
        if (std.mem.eql(u8, workload, "spawn-join")) return 1_000_000;
        return null;
    }

    pub fn toObjectMap(allocator: Allocator) !ObjectMap {
        var obj: ObjectMap = .empty;
        try obj.put(allocator, "single-hot", .{ .integer = 16_000_000_000 });
        try obj.put(allocator, "hot", .{ .integer = 16_000_000_000 });
        try obj.put(allocator, "cancel-hot", .{ .integer = 16_000_000_000 });
        try obj.put(allocator, "atomic", .{ .integer = 8_000_000_000 });
        try obj.put(allocator, "wait-notify", .{ .integer = 100_000_000 });
        try obj.put(allocator, "spawn-join", .{ .integer = 1_000_000 });
        return obj;
    }
};

pub const LEGACY_ITERATION_DEFAULTS = struct {
    pub const single: i64 = 224_000_000;
    pub const cancel: i64 = 224_000_000;
    pub const hot: i64 = 128_000_000;
    pub const atomic: i64 = 64_000_000;
    pub const atomic_total: i64 = 256_000_000;
    pub const wait: i64 = 512_000;
    pub const spawn: i64 = 3_000;
};

pub fn defaultPilotIterationPlanValue(allocator: Allocator) !ObjectMap {
    var root: ObjectMap = .empty;

    var interp: ObjectMap = .empty;
    try interp.put(allocator, "single-hot", .{ .integer = 30_000_000 });
    try interp.put(allocator, "hot", try threadCountMap(allocator, &.{ 28_000_000, 28_000_000, 20_000_000, 10_000_000 }));
    try interp.put(allocator, "atomic", try threadCountMap(allocator, &.{ 72_000_000, 40_000_000, 28_000_000, 14_000_000 }));
    try interp.put(allocator, "wait-notify", try threadCountMap(allocator, &.{ 128_000, 64_000, 32_000, 16_000 }));
    try interp.put(allocator, "spawn-join", try threadCountMap(allocator, &.{ 9_000, 4_500, 2_250, 1_250 }));
    try root.put(allocator, "interpreter", .{ .object = interp });

    var aot: ObjectMap = .empty;
    try aot.put(allocator, "single-hot", .{ .integer = 1_900_000_000 });
    try aot.put(allocator, "hot", try threadCountMap(allocator, &.{ 1_800_000_000, 1_800_000_000, 900_000_000, 450_000_000 }));
    try aot.put(allocator, "atomic", try threadCountMap(allocator, &.{ 850_000_000, 180_000_000, 64_000_000, 64_000_000 }));
    try aot.put(allocator, "wait-notify", try threadCountMap(allocator, &.{ 500_000, 64_000, 32_000, 16_000 }));
    try aot.put(allocator, "spawn-join", try threadCountMap(allocator, &.{ 10_000, 5_000, 2_500, 1_250 }));
    try aot.put(allocator, "cancel-hot", try threadCountMap(allocator, &.{ 1_900_000_000, 1_900_000_000, 950_000_000, 475_000_000 }));
    try root.put(allocator, "aot", .{ .object = aot });

    return root;
}

fn threadCountMap(allocator: Allocator, vals: *const [4]i64) !Value {
    var obj: ObjectMap = .empty;
    try obj.put(allocator, "1", .{ .integer = vals[0] });
    try obj.put(allocator, "2", .{ .integer = vals[1] });
    try obj.put(allocator, "4", .{ .integer = vals[2] });
    try obj.put(allocator, "8", .{ .integer = vals[3] });
    return .{ .object = obj };
}

pub const IterationOverrides = struct {
    single_iterations: ?i64 = null,
    cancel_iterations: ?i64 = null,
    hot_iterations: ?i64 = null,
    atomic_iterations: ?i64 = null,
    atomic_total_iterations: ?i64 = null,
    wait_iterations: ?i64 = null,
    spawn_iterations: ?i64 = null,
};

pub fn resolvedIterationPlan(
    allocator: Allocator,
    modes: []const []const u8,
    thread_counts: []const i64,
    overrides: IterationOverrides,
) !ObjectMap {
    const defaults = try defaultPilotIterationPlanValue(allocator);
    var plan: ObjectMap = .empty;

    for (modes) |mode| {
        const mode_defaults = defaults.get(mode).?.object;
        var mode_obj: ObjectMap = .empty;
        var it = mode_defaults.iterator();
        while (it.next()) |entry| {
            const wl = entry.key_ptr.*;
            const v = entry.value_ptr.*;
            if (v == .object) {
                var filtered: ObjectMap = .empty;
                for (thread_counts) |tc| {
                    const tc_str = try std.fmt.allocPrint(allocator, "{d}", .{tc});
                    if (v.object.get(tc_str)) |tv| {
                        try filtered.put(allocator, tc_str, tv);
                    }
                }
                try mode_obj.put(allocator, wl, .{ .object = filtered });
            } else {
                try mode_obj.put(allocator, wl, v);
            }
        }
        try plan.put(allocator, mode, .{ .object = mode_obj });
    }

    if (overrides.single_iterations) |single| {
        var it = plan.iterator();
        while (it.next()) |entry| {
            try entry.value_ptr.object.put(allocator, "single-hot", .{ .integer = single });
        }
    }
    if (overrides.hot_iterations) |hot| {
        var it = plan.iterator();
        while (it.next()) |entry| {
            var m: ObjectMap = .empty;
            for (thread_counts) |tc| {
                const tc_str = try std.fmt.allocPrint(allocator, "{d}", .{tc});
                try m.put(allocator, tc_str, .{ .integer = hot });
            }
            try entry.value_ptr.object.put(allocator, "hot", .{ .object = m });
        }
    }
    if (overrides.atomic_iterations != null or overrides.atomic_total_iterations != null) {
        const per_worker = overrides.atomic_iterations orelse LEGACY_ITERATION_DEFAULTS.atomic;
        const aggregate = overrides.atomic_total_iterations orelse LEGACY_ITERATION_DEFAULTS.atomic_total;
        var it = plan.iterator();
        while (it.next()) |entry| {
            var m: ObjectMap = .empty;
            for (thread_counts) |tc| {
                const tc_str = try std.fmt.allocPrint(allocator, "{d}", .{tc});
                try m.put(allocator, tc_str, .{ .integer = @max(per_worker, @divTrunc(aggregate, tc)) });
            }
            try entry.value_ptr.object.put(allocator, "atomic", .{ .object = m });
        }
    }
    if (overrides.wait_iterations) |wait| {
        var it = plan.iterator();
        while (it.next()) |entry| {
            var m: ObjectMap = .empty;
            for (thread_counts) |tc| {
                const tc_str = try std.fmt.allocPrint(allocator, "{d}", .{tc});
                try m.put(allocator, tc_str, .{ .integer = @divTrunc(wait, tc) });
            }
            try entry.value_ptr.object.put(allocator, "wait-notify", .{ .object = m });
        }
    }
    if (overrides.spawn_iterations) |spawn| {
        var it = plan.iterator();
        while (it.next()) |entry| {
            var m: ObjectMap = .empty;
            for (thread_counts) |tc| {
                const tc_str = try std.fmt.allocPrint(allocator, "{d}", .{tc});
                try m.put(allocator, tc_str, .{ .integer = spawn });
            }
            try entry.value_ptr.object.put(allocator, "spawn-join", .{ .object = m });
        }
    }
    if (plan.getPtr("aot")) |aot_val| {
        if (overrides.cancel_iterations != null or overrides.hot_iterations != null) {
            const cancel_total = overrides.cancel_iterations orelse LEGACY_ITERATION_DEFAULTS.cancel;
            const hot_floor = overrides.hot_iterations orelse LEGACY_ITERATION_DEFAULTS.hot;
            var m: ObjectMap = .empty;
            for (thread_counts) |tc| {
                const tc_str = try std.fmt.allocPrint(allocator, "{d}", .{tc});
                try m.put(allocator, tc_str, .{ .integer = @max(hot_floor, @divTrunc(cancel_total, tc)) });
            }
            try aot_val.object.put(allocator, "cancel-hot", .{ .object = m });
        }
    }
    return plan;
}

pub fn iterationCount(
    iteration_plan: ObjectMap,
    mode: []const u8,
    workload: []const u8,
    threads: i64,
) HarnessError!u64 {
    const mode_val = iteration_plan.get(mode) orelse return fail("missing mode {s} in iteration plan", .{mode});
    if (mode_val != .object) return fail("invalid mode {s} in iteration plan", .{mode});
    const wl_val = mode_val.object.get(workload) orelse return fail("missing workload {s} in iteration plan", .{workload});
    if (wl_val == .integer or wl_val == .number_string) {
        return spec.valueAsU64(wl_val) orelse return fail("invalid iteration count for {s}/{s}", .{ mode, workload });
    }
    if (wl_val == .object) {
        var buf: [32]u8 = undefined;
        const tc_str = std.fmt.bufPrint(&buf, "{d}", .{threads}) catch unreachable;
        const tv = wl_val.object.get(tc_str) orelse return fail("missing thread count {d} for {s}/{s}", .{ threads, mode, workload });
        return spec.valueAsU64(tv) orelse return fail("invalid iteration count for {s}/{s}/{d}", .{ mode, workload, threads });
    }
    return fail("invalid workload entry {s}/{s}", .{ mode, workload });
}

fn valueAsSignedI128(val: Value) ?i128 {
    return switch (val) {
        .integer => |i| @as(i128, i),
        .number_string => |s| std.fmt.parseInt(i128, s, 10) catch null,
        else => null,
    };
}

pub fn validateIterationPlanRanges(
    plan: ObjectMap,
    thread_counts: []const i64,
) HarnessError!void {
    var mode_it = plan.iterator();
    while (mode_it.next()) |entry| {
        const mode = entry.key_ptr.*;
        if (entry.value_ptr.* != .object) return fail("{s} iteration plan must be an object", .{mode});
        const workloads = entry.value_ptr.object;
        const single_val = workloads.get("single-hot") orelse return fail("{s} single-hot iterations missing", .{mode});
        const single = valueAsSignedI128(single_val) orelse return fail("{s} single-hot iterations must fit uint64", .{mode});
        if (single <= 0 or single > @as(i128, spec.MASK64)) {
            return fail("{s} single-hot iterations must fit uint64", .{mode});
        }
        const threaded_wls = [_][]const u8{ "hot", "atomic", "wait-notify", "spawn-join" };
        for (threaded_wls) |wl| {
            const wl_obj_v = workloads.get(wl) orelse return fail("{s} {s} missing", .{ mode, wl });
            if (wl_obj_v != .object) return fail("{s} {s} must be an object", .{ mode, wl });
            for (thread_counts) |threads| {
                var buf: [32]u8 = undefined;
                const tc_str = std.fmt.bufPrint(&buf, "{d}", .{threads}) catch unreachable;
                const iv = wl_obj_v.object.get(tc_str) orelse return fail("{s} {s}/{d} missing", .{ mode, wl, threads });
                const iters = valueAsSignedI128(iv) orelse return fail("{s} {s}/{d} operations overflow uint64", .{ mode, wl, threads });
                if (iters <= 0 or iters > @divTrunc(@as(i128, spec.MASK64), @as(i128, threads))) {
                    return fail("{s} {s}/{d} operations overflow uint64", .{ mode, wl, threads });
                }
                if (std.mem.eql(u8, wl, "wait-notify") and iters > @as(i128, spec.INT32_MAX)) {
                    return fail("{s} wait-notify/{d} iterations exceed int32", .{ mode, threads });
                }
                if (std.mem.eql(u8, wl, "spawn-join")) {
                    const tri = @divExact(@as(i128, threads) * @as(i128, threads + 1), 2);
                    if (iters > @divTrunc(@as(i128, spec.MASK64), tri)) {
                        return fail("{s} spawn-join/{d} checksum overflows uint64", .{ mode, threads });
                    }
                }
            }
        }
        if (std.mem.eql(u8, mode, "aot")) {
            const ch_obj_v = workloads.get("cancel-hot") orelse return fail("aot cancel-hot missing", .{});
            if (ch_obj_v != .object) return fail("aot cancel-hot must be an object", .{});
            for (thread_counts) |threads| {
                var buf: [32]u8 = undefined;
                const tc_str = std.fmt.bufPrint(&buf, "{d}", .{threads}) catch unreachable;
                const iv = ch_obj_v.object.get(tc_str) orelse return fail("aot cancel-hot/{d} missing", .{threads});
                const iters = valueAsSignedI128(iv) orelse return fail("aot cancel-hot/{d} operations overflow uint64", .{threads});
                if (iters <= 0 or iters > @divTrunc(@as(i128, spec.MASK64), @as(i128, threads))) {
                    return fail("aot cancel-hot/{d} operations overflow uint64", .{threads});
                }
            }
        }
    }
}

pub fn ceilDiv(numerator: i128, denominator: i128) HarnessError!u64 {
    if (numerator < 0 or denominator <= 0) {
        return fail("ceiling division requires non-negative/positive inputs", .{});
    }
    return @intCast(@divTrunc(numerator + denominator - 1, denominator));
}

pub fn roundUpSignificant(value: i128, significant_digits: usize) HarnessError!u64 {
    if (value <= 0 or significant_digits == 0) {
        return fail("sizing rounding inputs must be positive", .{});
    }
    const v_u64: u64 = @intCast(value);
    var buf: [64]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "{d}", .{v_u64}) catch unreachable;
    const digits = s.len;
    const exp: usize = if (digits > significant_digits) digits - significant_digits else 0;
    var quantum: i128 = 1;
    for (0..exp) |_| quantum *= 10;
    const q = try ceilDiv(value, quantum);
    return @intCast(@as(i128, q) * quantum);
}

pub fn sizingCellKey(allocator: Allocator, mode: []const u8, workload: []const u8, threads: i64) ![]const u8 {
    return std.fmt.allocPrint(allocator, "{s}/{s}/{d}", .{ mode, workload, threads });
}

pub const SizingCell = struct {
    mode: []const u8,
    workload: []const u8,
    threads: i64,
};

pub fn sizingCellForCondition(
    pair: ObjectMap,
    condition: []const u8,
) HarnessError!SizingCell {
    const left = (pair.get("left") orelse return fail("pair missing left", .{})).string;
    const right = (pair.get("right") orelse return fail("pair missing right", .{})).string;
    const pair_key = (pair.get("pair_key") orelse return fail("pair missing pair_key", .{})).string;
    const pair_kind = (pair.get("pair_kind") orelse return fail("pair missing pair_kind", .{})).string;
    if (!std.mem.eql(u8, condition, left) and !std.mem.eql(u8, condition, right)) {
        return fail("condition '{s}' is outside {s}", .{ condition, pair_key });
    }
    if (std.mem.eql(u8, pair_kind, "single-infrastructure")) {
        const slash = std.mem.lastIndexOfScalar(u8, pair_key, '/') orelse return fail("invalid pair_key {s}", .{pair_key});
        return .{ .mode = pair_key[slash + 1 ..], .workload = "single-hot", .threads = 1 };
    }
    if (std.mem.eql(u8, pair_kind, "cancel-point-cost")) {
        const slash = std.mem.lastIndexOfScalar(u8, pair_key, '/') orelse return fail("invalid pair_key {s}", .{pair_key});
        const threads = std.fmt.parseInt(i64, pair_key[slash + 1 ..], 10) catch return fail("invalid threads in {s}", .{pair_key});
        return .{ .mode = "aot", .workload = "cancel-hot", .threads = threads };
    }
    var it = std.mem.splitScalar(u8, pair_key, '/');
    _ = it.next(); // "runtime"
    const workload = it.next() orelse return fail("invalid pair_key {s}", .{pair_key});
    const raw_threads = it.next() orelse return fail("invalid pair_key {s}", .{pair_key});
    const threads = std.fmt.parseInt(i64, raw_threads, 10) catch return fail("invalid threads in {s}", .{pair_key});
    if (std.mem.eql(u8, pair_kind, "runtime-parity")) {
        return .{ .mode = condition, .workload = workload, .threads = threads };
    }
    if (std.mem.eql(u8, pair_kind, "repeatability")) {
        const mode = if (std.mem.endsWith(u8, left, "-a")) left[0 .. left.len - 2] else left;
        return .{ .mode = mode, .workload = workload, .threads = threads };
    }
    return fail("unsupported pair kind '{s}'", .{pair_kind});
}

fn makeBaseOverrideValue(
    allocator: Allocator,
    name: []const u8,
    workload: []const u8,
    threads: ?i64,
    safety_num: i64,
    safety_den: i64,
    rate_num: i64,
    rate_den: i64,
    projected_pilot_ns: i64,
) !Value {
    var obj: ObjectMap = .empty;
    try obj.put(allocator, "name", .{ .string = name });
    var sel: ObjectMap = .empty;
    try sel.put(allocator, "workload", .{ .string = workload });
    if (threads) |t| try sel.put(allocator, "threads", .{ .integer = t });
    try obj.put(allocator, "selector", .{ .object = sel });
    try obj.put(allocator, "target_duration_ns", .{ .integer = spec.SIZING_TARGET_NS });
    var sf: ObjectMap = .empty;
    try sf.put(allocator, "numerator", .{ .integer = safety_num });
    try sf.put(allocator, "denominator", .{ .integer = safety_den });
    try obj.put(allocator, "safety_factor", .{ .object = sf });
    var re: ObjectMap = .empty;
    try re.put(allocator, "numerator", .{ .integer = rate_num });
    try re.put(allocator, "denominator", .{ .integer = rate_den });
    try obj.put(allocator, "measurement_to_pilot_rate_envelope", .{ .object = re });
    try obj.put(allocator, "projected_pilot_duration_ns", .{ .integer = projected_pilot_ns });
    try obj.put(allocator, "formula", .{ .string = spec.SIZING_FORMULA });
    return .{ .object = obj };
}

pub fn sizingCellBaseOverridesValue(allocator: Allocator) !Value {
    var arr = Array.init(allocator);
    try arr.append(try makeBaseOverrideValue(allocator, "spawn-join-2x-rate-envelope", "spawn-join", null, 10, 7, 2, 1, 2_500_000_000));
    try arr.append(try makeBaseOverrideValue(allocator, "hot-8-20s-process-cpu-window", "hot", 8, 80, 7, 16, 1, 20_000_000_000));
    try arr.append(try makeBaseOverrideValue(allocator, "atomic-2-4-40s-contention-window", "atomic", 2, 160, 7, 32, 1, 40_000_000_000));
    try arr.append(try makeBaseOverrideValue(allocator, "atomic-2-4-40s-contention-window", "atomic", 4, 160, 7, 32, 1, 40_000_000_000));
    try arr.append(try makeBaseOverrideValue(allocator, "atomic-8-20s-contention-window", "atomic", 8, 80, 7, 16, 1, 20_000_000_000));
    return .{ .array = arr };
}

pub fn sizingCellEnvelopesValue(allocator: Allocator) !Value {
    var arr = Array.init(allocator);
    var env: ObjectMap = .empty;
    try env.put(allocator, "name", .{ .string = "aot-wait-notify-1-rate-acceleration" });
    var sel: ObjectMap = .empty;
    try sel.put(allocator, "mode", .{ .string = "aot" });
    try sel.put(allocator, "workload", .{ .string = "wait-notify" });
    try sel.put(allocator, "threads", .{ .integer = 1 });
    try env.put(allocator, "selector", .{ .object = sel });
    try env.put(allocator, "quality_floor_ns", .{ .integer = 1_250_000_000 });
    var re: ObjectMap = .empty;
    try re.put(allocator, "numerator", .{ .integer = 16 });
    try re.put(allocator, "denominator", .{ .integer = 1 });
    try env.put(allocator, "measurement_to_pilot_rate_envelope", .{ .object = re });
    try env.put(allocator, "projected_pilot_duration_ns", .{ .integer = 20_000_000_000 });
    try env.put(allocator, "formula", .{ .string = spec.SIZING_CELL_ENVELOPE_FORMULA });
    try arr.append(.{ .object = env });
    return .{ .array = arr };
}

pub fn sizingAlgorithmSpec(allocator: Allocator, timeout_seconds: f64) !Value {
    const timeout_ns: i64 = @intFromFloat(timeout_seconds * 1_000_000_000.0);
    var obj: ObjectMap = .empty;
    try obj.put(allocator, "version", .{ .integer = spec.SIZING_ALGORITHM_VERSION });
    try obj.put(allocator, "kind", .{ .string = spec.SIZING_ALGORITHM_KIND });
    try obj.put(allocator, "selection_rate", .{ .string = "fastest-valid-pilot-across-all-revisions-and-conditions" });
    try obj.put(allocator, "formula", .{ .string = spec.SIZING_FORMULA });
    try obj.put(allocator, "cell_selection", .{ .string = spec.SIZING_CELL_SELECTION });
    try obj.put(allocator, "cell_base_overrides", try sizingCellBaseOverridesValue(allocator));
    try obj.put(allocator, "cell_envelopes", try sizingCellEnvelopesValue(allocator));
    try obj.put(allocator, "target_duration_ns", .{ .integer = spec.SIZING_TARGET_NS });
    var sf: ObjectMap = .empty;
    try sf.put(allocator, "numerator", .{ .integer = spec.SIZING_SAFETY_NUMERATOR });
    try sf.put(allocator, "denominator", .{ .integer = spec.SIZING_SAFETY_DENOMINATOR });
    try obj.put(allocator, "safety_factor", .{ .object = sf });
    var rounding: ObjectMap = .empty;
    try rounding.put(allocator, "kind", .{ .string = "decimal-significant-digits-ceiling" });
    try rounding.put(allocator, "significant_digits", .{ .integer = @intCast(spec.SIZING_SIGNIFICANT_DIGITS) });
    try obj.put(allocator, "rounding", .{ .object = rounding });
    try obj.put(allocator, "pilot_clock_resolution_minimum_ns", .{ .integer = spec.PILOT_CLOCK_RESOLUTION_MINIMUM_NS });
    try obj.put(allocator, "maximum_pilot_corrected_ns", .{ .integer = spec.MAXIMUM_PILOT_CORRECTED_NS });
    try obj.put(allocator, "maximum_pilot_corrected_scope", .{ .string = "per-process-monotonic-or-per-worker-process-cpu" });
    try obj.put(allocator, "maximum_pilot_host_wall_ns", .{ .integer = spec.MAXIMUM_PILOT_HOST_WALL_NS });
    try obj.put(allocator, "projected_evidence_minimum_ns", .{ .integer = spec.PROJECTED_EVIDENCE_MINIMUM_NS });
    try obj.put(allocator, "timing_overhead_ratio_limit", .{ .float = spec.TIMING_OVERHEAD_RATIO_LIMIT });

    var limits: ObjectMap = .empty;
    try spec.putU64Json(allocator, &limits, "uint64_max", spec.MASK64);
    try limits.put(allocator, "wait_notify_int32_max", .{ .integer = spec.INT32_MAX });
    try limits.put(allocator, "spawn_join_thread_lifecycle_cap", .{ .integer = spec.SPAWN_JOIN_THREAD_LIFECYCLE_CAP });
    try limits.put(allocator, "workload_iteration_caps", .{ .object = try SIZING_WORKLOAD_CAPS.toObjectMap(allocator) });
    try limits.put(allocator, "per_invocation_timeout_ns", .{ .integer = timeout_ns });
    try limits.put(allocator, "workflow_job_timeout_ns", .{ .integer = spec.WORKFLOW_JOB_TIMEOUT_NS });
    try limits.put(allocator, "job_non_benchmark_reserve_ns", .{ .integer = spec.JOB_NON_BENCHMARK_RESERVE_NS });
    try limits.put(allocator, "projected_benchmark_limit_ns", .{ .integer = spec.PROJECTED_BENCHMARK_LIMIT_NS });
    try limits.put(allocator, "auxiliary_invocation_budget_ns", .{ .integer = spec.AUXILIARY_INVOCATION_BUDGET_NS });
    try obj.put(allocator, "limits", .{ .object = limits });
    try obj.put(allocator, "failure_policy", .{ .string = "one-shot-no-retry-no-discard" });
    return .{ .object = obj };
}

pub fn pilotOrderForPlan(
    allocator: Allocator,
    pairs: []const ObjectMap,
    revision_roles: []const []const u8,
    pilot_plan: ObjectMap,
) HarnessError![]ObjectMap {
    var order: std.ArrayList(ObjectMap) = .empty;
    for (pairs) |pair| {
        const pair_kind = pair.get("pair_kind").?.string;
        const pair_key = pair.get("pair_key").?.string;
        const left = pair.get("left").?.string;
        const right = pair.get("right").?.string;
        const conditions = [_][]const u8{ left, right };
        for (revision_roles) |revision| {
            for (conditions) |condition| {
                const cell = try sizingCellForCondition(pair, condition);
                const iters = try iterationCount(pilot_plan, cell.mode, cell.workload, cell.threads);
                var item: ObjectMap = .empty;
                try item.put(allocator, "pilot_index", .{ .integer = @intCast(order.items.len) });
                try item.put(allocator, "revision", .{ .string = revision });
                try item.put(allocator, "pair_kind", .{ .string = pair_kind });
                try item.put(allocator, "pair_key", .{ .string = pair_key });
                try item.put(allocator, "condition", .{ .string = condition });
                try item.put(allocator, "mode", .{ .string = cell.mode });
                try item.put(allocator, "workload", .{ .string = cell.workload });
                try item.put(allocator, "threads", .{ .integer = cell.threads });
                try spec.putU64Json(allocator, &item, "iterations", iters);
                try order.append(allocator, item);
            }
        }
    }
    return order.toOwnedSlice(allocator);
}

pub fn effectiveSizingCap(workload: []const u8, threads: i64) HarnessError!u64 {
    var cap = SIZING_WORKLOAD_CAPS.get(workload) orelse return fail("unknown workload {s}", .{workload});
    const threads_u64: u64 = @intCast(threads);
    cap = @min(cap, spec.MASK64 / threads_u64);
    if (std.mem.eql(u8, workload, "wait-notify")) {
        cap = @min(cap, @as(u64, @intCast(spec.INT32_MAX)));
    }
    if (std.mem.eql(u8, workload, "spawn-join")) {
        const tri: u64 = @intCast(@divExact(threads * (threads + 1), 2));
        cap = @min(cap, spec.MASK64 / tri);
        cap = @min(cap, @as(u64, @intCast(@divTrunc(spec.SPAWN_JOIN_THREAD_LIFECYCLE_CAP, threads))));
    }
    return cap;
}

pub fn selectedIterationsFromElapsed(
    pilot_iterations: i128,
    pilot_elapsed_ns: i128,
) HarnessError!struct { required: u64, rounded: u64 } {
    if (pilot_iterations <= 0 or pilot_elapsed_ns <= 0) {
        return fail("pilot sizing inputs must be positive", .{});
    }
    const required = try ceilDiv(
        pilot_iterations * @as(i128, spec.SIZING_TARGET_NS) * @as(i128, spec.SIZING_SAFETY_NUMERATOR),
        pilot_elapsed_ns * @as(i128, spec.SIZING_SAFETY_DENOMINATOR),
    );
    const rounded = try roundUpSignificant(@as(i128, required), spec.SIZING_SIGNIFICANT_DIGITS);
    return .{ .required = required, .rounded = rounded };
}

pub const BaseSizingPolicy = struct {
    name: []const u8,
    target_duration_ns: i64,
    safety_numerator: i64,
    safety_denominator: i64,
    projected_pilot_duration_ns: i64,
};

pub fn baseSizingPolicyForCell(
    _mode: []const u8,
    workload: []const u8,
    threads: i64,
) BaseSizingPolicy {
    _ = _mode;
    if (std.mem.eql(u8, workload, "spawn-join")) {
        return .{
            .name = "spawn-join-2x-rate-envelope",
            .target_duration_ns = spec.SIZING_TARGET_NS,
            .safety_numerator = 10,
            .safety_denominator = 7,
            .projected_pilot_duration_ns = 2_500_000_000,
        };
    }
    if (std.mem.eql(u8, workload, "hot") and threads == 8) {
        return .{
            .name = "hot-8-20s-process-cpu-window",
            .target_duration_ns = spec.SIZING_TARGET_NS,
            .safety_numerator = 80,
            .safety_denominator = 7,
            .projected_pilot_duration_ns = 20_000_000_000,
        };
    }
    if (std.mem.eql(u8, workload, "atomic") and (threads == 2 or threads == 4)) {
        return .{
            .name = "atomic-2-4-40s-contention-window",
            .target_duration_ns = spec.SIZING_TARGET_NS,
            .safety_numerator = 160,
            .safety_denominator = 7,
            .projected_pilot_duration_ns = 40_000_000_000,
        };
    }
    if (std.mem.eql(u8, workload, "atomic") and threads == 8) {
        return .{
            .name = "atomic-8-20s-contention-window",
            .target_duration_ns = spec.SIZING_TARGET_NS,
            .safety_numerator = 80,
            .safety_denominator = 7,
            .projected_pilot_duration_ns = 20_000_000_000,
        };
    }
    return .{
        .name = "general",
        .target_duration_ns = spec.SIZING_TARGET_NS,
        .safety_numerator = spec.SIZING_SAFETY_NUMERATOR,
        .safety_denominator = spec.SIZING_SAFETY_DENOMINATOR,
        .projected_pilot_duration_ns = spec.PROJECTED_EVIDENCE_MINIMUM_NS,
    };
}

pub fn projectedDurationFloorForCell(
    mode: []const u8,
    workload: []const u8,
    threads: i64,
) i64 {
    const base_policy = baseSizingPolicyForCell(mode, workload, threads);
    var floor = base_policy.projected_pilot_duration_ns;
    if (std.mem.eql(u8, mode, "aot") and std.mem.eql(u8, workload, "wait-notify") and threads == 1) {
        floor = @max(floor, 20_000_000_000);
    }
    return floor;
}

pub const SizingCandidates = struct {
    general_required_iterations: u64,
    general_rounded_iterations: u64,
    applicable_envelopes: Array,
    selected_required_iterations: u64,
    selected_iterations: u64,
    selected_sources: Array,
};

pub fn sizingCandidatesForCell(
    allocator: Allocator,
    mode: []const u8,
    workload: []const u8,
    threads: i64,
    pilot_iterations: u64,
    pilot_elapsed_ns: i64,
) HarnessError!SizingCandidates {
    if (pilot_iterations == 0 or pilot_elapsed_ns <= 0) {
        return fail("pilot sizing inputs must be positive", .{});
    }
    const base_policy = baseSizingPolicyForCell(mode, workload, threads);
    const general_required = try ceilDiv(
        @as(i128, pilot_iterations) * @as(i128, base_policy.target_duration_ns) * @as(i128, base_policy.safety_numerator),
        @as(i128, pilot_elapsed_ns) * @as(i128, base_policy.safety_denominator),
    );
    const general_rounded = try roundUpSignificant(@as(i128, general_required), spec.SIZING_SIGNIFICANT_DIGITS);

    var envelope_candidates = Array.init(allocator);
    var selected_required = general_required;
    var selected = general_rounded;

    if (std.mem.eql(u8, mode, "aot") and std.mem.eql(u8, workload, "wait-notify") and threads == 1) {
        const req = try ceilDiv(
            @as(i128, pilot_iterations) * 1_250_000_000 * 16,
            @as(i128, pilot_elapsed_ns) * 1,
        );
        const rnd = try roundUpSignificant(@as(i128, req), spec.SIZING_SIGNIFICANT_DIGITS);
        var env_obj: ObjectMap = .empty;
        try env_obj.put(allocator, "name", .{ .string = "aot-wait-notify-1-rate-acceleration" });
        try spec.putU64Json(allocator, &env_obj, "required_iterations", req);
        try spec.putU64Json(allocator, &env_obj, "rounded_iterations", rnd);
        try envelope_candidates.append(.{ .object = env_obj });
        selected_required = @max(selected_required, req);
        selected = @max(selected, rnd);
    }

    var selected_sources = Array.init(allocator);
    if (general_rounded == selected) {
        try selected_sources.append(.{ .string = base_policy.name });
    }
    for (envelope_candidates.items) |item| {
        const rnd = spec.valueAsU64(item.object.get("rounded_iterations").?).?;
        if (rnd == selected) {
            try selected_sources.append(item.object.get("name").?);
        }
    }
    if (std.mem.eql(u8, workload, "spawn-join")) {
        selected = @min(selected, @as(u64, @intCast(@divTrunc(spec.SPAWN_JOIN_THREAD_LIFECYCLE_CAP, threads))));
    }

    return .{
        .general_required_iterations = general_required,
        .general_rounded_iterations = general_rounded,
        .applicable_envelopes = envelope_candidates,
        .selected_required_iterations = selected_required,
        .selected_iterations = selected,
        .selected_sources = selected_sources,
    };
}

pub fn validateSizingPilot(
    allocator: Allocator,
    record: ObjectMap,
    expected: ObjectMap,
) HarnessError!void {
    const keys = [_][]const u8{
        "pilot_index",
        "revision",
        "pair_kind",
        "pair_key",
        "condition",
        "mode",
        "workload",
        "threads",
        "iterations",
    };
    for (keys) |k| {
        const rv = record.get(k) orelse return fail("sizing pilot order mismatch for {s}", .{k});
        const ev = expected.get(k).?;
        if (!(try spec.jsonEquals(allocator, rv, ev))) {
            return fail("sizing pilot order mismatch for {s}", .{k});
        }
    }
    const correct_val = record.get("correct") orelse return fail("sizing pilot correctness failed", .{});
    if (correct_val != .bool or !correct_val.bool) {
        return fail("sizing pilot correctness failed", .{});
    }
    const elapsed_val = record.get("guest_elapsed_ns") orelse return fail("sizing pilot clock resolution is insufficient", .{});
    if (elapsed_val != .integer or elapsed_val.integer < spec.PILOT_CLOCK_RESOLUTION_MINIMUM_NS) {
        return fail("sizing pilot clock resolution is insufficient", .{});
    }
    const elapsed = elapsed_val.integer;
    const expected_wl = expected.get("workload").?.string;
    const guest_workload = if (std.mem.eql(u8, expected_wl, "cancel-hot")) "hot" else expected_wl;
    const threads = expected.get("threads").?.integer;
    const clock_id = try spec.expectedGuestClockId(guest_workload);
    const mult: i64 = if (std.mem.eql(u8, clock_id, spec.WASI_PROCESS_CPU_CLOCK_ID) and
        (std.mem.eql(u8, guest_workload, "hot") or std.mem.eql(u8, guest_workload, "atomic")))
        threads
    else
        1;
    const maximum_corrected_ns = spec.MAXIMUM_PILOT_CORRECTED_NS * mult;
    if (elapsed > maximum_corrected_ns) {
        return fail("sizing pilot corrected duration exceeds 30 seconds", .{});
    }
    const host_wall_val = record.get("host_wall_elapsed_ns") orelse return fail("sizing pilot host-wall duration exceeds 33 seconds", .{});
    if (host_wall_val != .integer) {
        return fail("sizing pilot host-wall duration exceeds 33 seconds", .{});
    }
    const host_wall = host_wall_val.integer;
    if ((std.mem.eql(u8, clock_id, spec.WASI_MONOTONIC_CLOCK_ID) and host_wall < elapsed) or
        host_wall > spec.MAXIMUM_PILOT_HOST_WALL_NS)
    {
        return fail("sizing pilot host-wall duration exceeds 33 seconds", .{});
    }
    const overhead_val = record.get("timing_overhead_ns") orelse return fail("sizing pilot barrier diagnostics are invalid", .{});
    const raw_val = record.get("raw_guest_elapsed_ns") orelse return fail("sizing pilot barrier diagnostics are invalid", .{});
    if (overhead_val != .integer or overhead_val.integer < 0 or
        raw_val != .integer or raw_val.integer != elapsed + overhead_val.integer)
    {
        return fail("sizing pilot barrier diagnostics are invalid", .{});
    }
    const exp_iters = spec.valueAsU64(expected.get("iterations").?).?;
    const exp_res = try spec.expectedResult(guest_workload, threads, @as(i128, exp_iters), null);
    const rec_ops_val = record.get("operations") orelse return fail("sizing pilot operation count mismatch", .{});
    const rec_ops = spec.valueAsU64(rec_ops_val) orelse return fail("sizing pilot operation count mismatch", .{});
    if (rec_ops != exp_res.operations) {
        return fail("sizing pilot operation count mismatch", .{});
    }
}

pub fn pilotProgressBound(
    allocator: Allocator,
    pilot_records: []const ObjectMap,
    pilot_order: []const ObjectMap,
    warmups: i64,
    samples: i64,
) HarnessError!ObjectMap {
    if (pilot_records.len > pilot_order.len) {
        return fail("sizing pilot progress exceeds declared order", .{});
    }
    var completed_wall_ns: i64 = 0;
    var completed_elapsed_ns: i64 = 0;
    for (pilot_records) |rec| {
        completed_wall_ns += rec.get("host_wall_elapsed_ns").?.integer;
        completed_elapsed_ns += rec.get("guest_elapsed_ns").?.integer;
    }
    const remaining_pilots: i64 = @intCast(pilot_order.len - pilot_records.len);
    const remaining_pilot_bound_ns = remaining_pilots * spec.MAXIMUM_PILOT_HOST_WALL_NS;
    var floor_sum: i64 = 0;
    for (pilot_order) |p| {
        floor_sum += projectedDurationFloorForCell(
            p.get("mode").?.string,
            p.get("workload").?.string,
            p.get("threads").?.integer,
        );
    }
    const minimum_evidence_bound_ns = (warmups + samples) * floor_sum;
    const earliest_complete_bound_ns = completed_wall_ns +
        remaining_pilot_bound_ns +
        minimum_evidence_bound_ns +
        spec.AUXILIARY_INVOCATION_BUDGET_NS;
    if (earliest_complete_bound_ns >= spec.PROJECTED_BENCHMARK_LIMIT_NS) {
        return fail(
            "sizing pilot progress cannot fit the {d}-minute benchmark bound",
            .{@divTrunc(spec.PROJECTED_BENCHMARK_LIMIT_NS, 60 * 1_000_000_000)},
        );
    }
    var out: ObjectMap = .empty;
    try out.put(allocator, "completed_pilot_elapsed_ns", .{ .integer = completed_elapsed_ns });
    try out.put(allocator, "completed_pilot_host_wall_ns", .{ .integer = completed_wall_ns });
    try out.put(allocator, "remaining_pilot_bound_ns", .{ .integer = remaining_pilot_bound_ns });
    try out.put(allocator, "minimum_evidence_bound_ns", .{ .integer = minimum_evidence_bound_ns });
    try out.put(allocator, "earliest_complete_bound_ns", .{ .integer = earliest_complete_bound_ns });
    return out;
}

pub fn emptyIterationPlan(
    allocator: Allocator,
    modes: []const []const u8,
    thread_counts: []const i64,
) !ObjectMap {
    var result: ObjectMap = .empty;
    for (modes) |mode| {
        var m_obj: ObjectMap = .empty;
        try m_obj.put(allocator, "single-hot", .{ .integer = 0 });
        const wls = [_][]const u8{ "hot", "atomic", "wait-notify", "spawn-join" };
        for (wls) |wl| {
            var tc_obj: ObjectMap = .empty;
            for (thread_counts) |tc| {
                const tc_str = try std.fmt.allocPrint(allocator, "{d}", .{tc});
                try tc_obj.put(allocator, tc_str, .{ .integer = 0 });
            }
            try m_obj.put(allocator, wl, .{ .object = tc_obj });
        }
        if (std.mem.eql(u8, mode, "aot")) {
            var tc_obj: ObjectMap = .empty;
            for (thread_counts) |tc| {
                const tc_str = try std.fmt.allocPrint(allocator, "{d}", .{tc});
                try tc_obj.put(allocator, tc_str, .{ .integer = 0 });
            }
            try m_obj.put(allocator, "cancel-hot", .{ .object = tc_obj });
        }
        try result.put(allocator, mode, .{ .object = m_obj });
    }
    return result;
}

pub fn setIterationCount(
    allocator: Allocator,
    plan: *ObjectMap,
    mode: []const u8,
    workload: []const u8,
    threads: i64,
    value: u64,
) !void {
    var mode_obj = plan.getPtr(mode).?;
    if (std.mem.eql(u8, workload, "single-hot")) {
        try spec.putU64Json(allocator, &mode_obj.object, workload, value);
    } else {
        var wl_obj = mode_obj.object.getPtr(workload).?;
        const tc_str = try std.fmt.allocPrint(allocator, "{d}", .{threads});
        try spec.putU64Json(allocator, &wl_obj.object, tc_str, value);
    }
}

fn cellLessThan(_: void, a: SizingCell, b: SizingCell) bool {
    const m_cmp = std.mem.order(u8, a.mode, b.mode);
    if (m_cmp != .eq) return m_cmp == .lt;
    const w_cmp = std.mem.order(u8, a.workload, b.workload);
    if (w_cmp != .eq) return w_cmp == .lt;
    return a.threads < b.threads;
}

fn cellEq(a: SizingCell, b: SizingCell) bool {
    return std.mem.eql(u8, a.mode, b.mode) and
        std.mem.eql(u8, a.workload, b.workload) and
        a.threads == b.threads;
}

pub fn resolveOneShotSizing(
    allocator: Allocator,
    pilot_records: []const ObjectMap,
    pilot_order: []const ObjectMap,
    modes: []const []const u8,
    thread_counts: []const i64,
    warmups: i64,
    samples: i64,
    timeout_seconds: f64,
) HarnessError!struct { plan: ObjectMap, resolution: ObjectMap } {
    if (pilot_records.len != pilot_order.len) {
        return fail("sizing pilot set is incomplete", .{});
    }
    for (pilot_order, pilot_records) |expected, record| {
        try validateSizingPilot(allocator, record, expected);
    }

    var expected_cells: std.ArrayList(SizingCell) = .empty;
    for (modes) |mode| {
        try expected_cells.append(allocator, .{ .mode = mode, .workload = "single-hot", .threads = 1 });
        const wls = [_][]const u8{ "hot", "atomic", "wait-notify", "spawn-join" };
        for (wls) |wl| {
            for (thread_counts) |tc| {
                try expected_cells.append(allocator, .{ .mode = mode, .workload = wl, .threads = tc });
            }
        }
        if (std.mem.eql(u8, mode, "aot")) {
            for (thread_counts) |tc| {
                try expected_cells.append(allocator, .{ .mode = mode, .workload = "cancel-hot", .threads = tc });
            }
        }
    }
    std.mem.sort(SizingCell, expected_cells.items, {}, cellLessThan);

    for (pilot_order) |expected| {
        const cell: SizingCell = .{
            .mode = expected.get("mode").?.string,
            .workload = expected.get("workload").?.string,
            .threads = expected.get("threads").?.integer,
        };
        var found = false;
        for (expected_cells.items) |ec| {
            if (cellEq(cell, ec)) {
                found = true;
                break;
            }
        }
        if (!found) return fail("sizing pilots do not cover every selected cell", .{});
    }

    var selected_plan = try emptyIterationPlan(allocator, modes, thread_counts);
    var cells_arr = Array.init(allocator);
    var selected_by_cell = try allocator.alloc(u64, expected_cells.items.len);

    for (expected_cells.items, 0..) |cell, cell_idx| {
        var records_for_cell: std.ArrayList(ObjectMap) = .empty;
        for (pilot_order, pilot_records) |expected, rec| {
            const c: SizingCell = .{
                .mode = expected.get("mode").?.string,
                .workload = expected.get("workload").?.string,
                .threads = expected.get("threads").?.integer,
            };
            if (cellEq(c, cell)) {
                try records_for_cell.append(allocator, rec);
            }
        }
        if (records_for_cell.items.len == 0) {
            return fail("sizing pilots do not cover every selected cell", .{});
        }

        var fastest = records_for_cell.items[0];
        for (records_for_cell.items[1..]) |rec| {
            if (rec.get("guest_elapsed_ns").?.integer < fastest.get("guest_elapsed_ns").?.integer) {
                fastest = rec;
            }
        }

        var baseline_fastest_opt: ?ObjectMap = null;
        for (records_for_cell.items) |rec| {
            if (std.mem.eql(u8, rec.get("revision").?.string, "baseline")) {
                if (baseline_fastest_opt == null or
                    rec.get("guest_elapsed_ns").?.integer < baseline_fastest_opt.?.get("guest_elapsed_ns").?.integer)
                {
                    baseline_fastest_opt = rec;
                }
            }
        }
        const baseline_fastest = baseline_fastest_opt orelse fastest;

        const baseline_candidates = try sizingCandidatesForCell(
            allocator,
            cell.mode,
            cell.workload,
            cell.threads,
            spec.valueAsU64(baseline_fastest.get("iterations").?).?,
            baseline_fastest.get("guest_elapsed_ns").?.integer,
        );
        const fastest_candidates = try sizingCandidatesForCell(
            allocator,
            cell.mode,
            cell.workload,
            cell.threads,
            spec.valueAsU64(fastest.get("iterations").?).?,
            fastest.get("guest_elapsed_ns").?.integer,
        );
        const selected = fastest_candidates.selected_iterations;
        const cap = try effectiveSizingCap(cell.workload, cell.threads);
        if (selected > cap) {
            const ckey = try sizingCellKey(allocator, cell.mode, cell.workload, cell.threads);
            return fail("sizing cell {s} requires {d} iterations above cap {d}", .{ ckey, selected, cap });
        }
        try setIterationCount(allocator, &selected_plan, cell.mode, cell.workload, cell.threads, selected);
        selected_by_cell[cell_idx] = selected;

        var cell_obj: ObjectMap = .empty;
        try cell_obj.put(allocator, "key", .{ .string = try sizingCellKey(allocator, cell.mode, cell.workload, cell.threads) });
        try cell_obj.put(allocator, "mode", .{ .string = cell.mode });
        try cell_obj.put(allocator, "workload", .{ .string = cell.workload });
        try cell_obj.put(allocator, "threads", .{ .integer = cell.threads });
        try spec.putU64Json(allocator, &cell_obj, "pilot_count", spec.valueAsU64(fastest.get("iterations").?).?);
        try cell_obj.put(allocator, "pilot_observations", .{ .integer = @intCast(records_for_cell.items.len) });
        try cell_obj.put(allocator, "baseline_fastest_pilot_index", baseline_fastest.get("pilot_index").?);
        try cell_obj.put(allocator, "baseline_fastest_elapsed_ns", baseline_fastest.get("guest_elapsed_ns").?);
        try spec.putU64Json(allocator, &cell_obj, "baseline_required_iterations", baseline_candidates.selected_required_iterations);
        try spec.putU64Json(allocator, &cell_obj, "baseline_rounded_iterations", baseline_candidates.selected_iterations);
        try cell_obj.put(allocator, "fastest_pilot_index", fastest.get("pilot_index").?);
        try cell_obj.put(allocator, "fastest_elapsed_ns", fastest.get("guest_elapsed_ns").?);
        try spec.putU64Json(allocator, &cell_obj, "fastest_required_iterations", fastest_candidates.selected_required_iterations);
        try spec.putU64Json(allocator, &cell_obj, "general_required_iterations", fastest_candidates.general_required_iterations);
        try spec.putU64Json(allocator, &cell_obj, "general_rounded_iterations", fastest_candidates.general_rounded_iterations);
        try cell_obj.put(allocator, "applicable_envelopes", .{ .array = fastest_candidates.applicable_envelopes });
        try spec.putU64Json(allocator, &cell_obj, "selected_iterations", selected);
        try cell_obj.put(allocator, "selected_sources", .{ .array = fastest_candidates.selected_sources });
        try spec.putU64Json(allocator, &cell_obj, "effective_iteration_cap", cap);
        try cells_arr.append(.{ .object = cell_obj });
    }

    const timeout_ns: u64 = @intFromFloat(timeout_seconds * 1_000_000_000.0);
    const min_timed_ns: u64 = @intFromFloat(spec.MIN_TIMED_INTERVAL_MS * 1_000_000.0);
    var projections_arr = Array.init(allocator);
    var projected_evidence_ns: i64 = 0;

    for (pilot_records) |rec| {
        const cell: SizingCell = .{
            .mode = rec.get("mode").?.string,
            .workload = rec.get("workload").?.string,
            .threads = rec.get("threads").?.integer,
        };
        var selected: u64 = 0;
        for (expected_cells.items, 0..) |ec, cidx| {
            if (cellEq(cell, ec)) {
                selected = selected_by_cell[cidx];
                break;
            }
        }
        const rec_guest_ns = rec.get("guest_elapsed_ns").?.integer;
        const rec_host_ns = rec.get("host_wall_elapsed_ns").?.integer;
        const rec_iters = spec.valueAsU64(rec.get("iterations").?).?;
        const rec_overhead_ns = rec.get("timing_overhead_ns").?.integer;
        const pilot_idx = rec.get("pilot_index").?.integer;

        const projected_guest = try ceilDiv(@as(i128, rec_guest_ns) * @as(i128, selected), @as(i128, rec_iters));
        const projected_host = try ceilDiv(@as(i128, rec_host_ns) * @as(i128, selected), @as(i128, rec_iters));
        const projected_minimum: u64 = if (std.mem.eql(u8, cell.workload, "spawn-join") and
            selected == @as(u64, @intCast(@divTrunc(spec.SPAWN_JOIN_THREAD_LIFECYCLE_CAP, cell.threads))))
            min_timed_ns
        else
            @max(min_timed_ns, @as(u64, @intCast(projectedDurationFloorForCell(cell.mode, cell.workload, cell.threads))));

        if (projected_guest < projected_minimum) {
            return fail("sizing projection is below the evidence minimum for pilot {d}", .{pilot_idx});
        }
        if (99 * @as(i128, rec_overhead_ns) >= @as(i128, projected_guest)) {
            return fail("sizing projected barrier ratio is not below 1% for pilot {d}", .{pilot_idx});
        }
        if (projected_guest >= timeout_ns or projected_host >= timeout_ns) {
            return fail("sizing projection exceeds {d}s invocation timeout for pilot {d}", .{ timeout_seconds, pilot_idx });
        }
        projected_evidence_ns += @as(i64, @intCast(projected_host)) * (warmups + samples);

        var proj_obj: ObjectMap = .empty;
        try proj_obj.put(allocator, "pilot_index", .{ .integer = pilot_idx });
        try spec.putU64Json(allocator, &proj_obj, "selected_iterations", selected);
        try spec.putU64Json(allocator, &proj_obj, "projected_guest_elapsed_ns", projected_guest);
        try spec.putU64Json(allocator, &proj_obj, "projected_host_wall_elapsed_ns", projected_host);
        const ratio = @as(f64, @floatFromInt(rec_overhead_ns)) /
            (@as(f64, @floatFromInt(projected_guest)) + @as(f64, @floatFromInt(rec_overhead_ns)));
        try proj_obj.put(allocator, "projected_timing_overhead_ratio", .{ .float = ratio });
        try spec.putU64Json(allocator, &proj_obj, "projected_evidence_minimum_ns", projected_minimum);
        try projections_arr.append(.{ .object = proj_obj });
    }

    var pilot_host_ns: i64 = 0;
    var pilot_elapsed_ns: i64 = 0;
    var pilots_arr = Array.init(allocator);
    for (pilot_records) |rec| {
        pilot_host_ns += rec.get("host_wall_elapsed_ns").?.integer;
        pilot_elapsed_ns += rec.get("guest_elapsed_ns").?.integer;
        try pilots_arr.append(try spec.jsonClone(allocator, .{ .object = rec }));
    }
    const maximum_pre_admission_pilot_bound_ns: i64 =
        @as(i64, @intCast(pilot_order.len)) * spec.MAXIMUM_PILOT_HOST_WALL_NS;
    const projected_evidence_limit_ns =
        spec.PROJECTED_BENCHMARK_LIMIT_NS - pilot_host_ns - spec.AUXILIARY_INVOCATION_BUDGET_NS;
    if (projected_evidence_limit_ns <= 0) {
        return fail("declared pilot set leaves no evidence runtime budget", .{});
    }
    if (projected_evidence_ns >= projected_evidence_limit_ns) {
        return fail(
            "sizing evidence projection cannot fit the {d}-minute benchmark bound",
            .{@divTrunc(spec.PROJECTED_BENCHMARK_LIMIT_NS, 60 * 1_000_000_000)},
        );
    }
    const projected_benchmark_ns =
        pilot_host_ns + projected_evidence_ns + spec.AUXILIARY_INVOCATION_BUDGET_NS;
    if (projected_benchmark_ns >= spec.PROJECTED_BENCHMARK_LIMIT_NS) {
        return fail("sizing projection exceeds the benchmark share of the workflow timeout", .{});
    }
    try validateIterationPlanRanges(selected_plan, thread_counts);

    var resolution: ObjectMap = .empty;
    try resolution.put(allocator, "pilots", .{ .array = pilots_arr });
    try resolution.put(allocator, "cells", .{ .array = cells_arr });
    try resolution.put(allocator, "projections", .{ .array = projections_arr });
    try resolution.put(allocator, "pilot_corrected_elapsed_ns", .{ .integer = pilot_elapsed_ns });
    try resolution.put(allocator, "pilot_host_wall_elapsed_ns", .{ .integer = pilot_host_ns });
    try resolution.put(allocator, "maximum_pre_admission_pilot_bound_ns", .{ .integer = maximum_pre_admission_pilot_bound_ns });
    try resolution.put(allocator, "projected_evidence_host_wall_ns", .{ .integer = projected_evidence_ns });
    try resolution.put(allocator, "projected_evidence_limit_ns", .{ .integer = projected_evidence_limit_ns });
    try resolution.put(allocator, "auxiliary_invocation_budget_ns", .{ .integer = spec.AUXILIARY_INVOCATION_BUDGET_NS });
    try resolution.put(allocator, "projected_benchmark_ns", .{ .integer = projected_benchmark_ns });
    try resolution.put(allocator, "projected_benchmark_limit_ns", .{ .integer = spec.PROJECTED_BENCHMARK_LIMIT_NS });

    return .{ .plan = selected_plan, .resolution = resolution };
}

pub fn validateSizingPlan(allocator: Allocator, plan: ObjectMap) HarnessError!void {
    const sizing_val = plan.get("sizing") orelse return failData("plan.sizing", .{});
    try require(sizing_val == .object and sizing_val.object.count() == 4, "plan.sizing", .{});
    const sizing = sizing_val.object;
    for ([_][]const u8{ "algorithm", "pilot_iterations", "pilot_order", "resolved" }) |k| {
        try require(sizing.contains(k), "plan.sizing", .{});
    }
    const timeout_val = plan.get("timeout_seconds") orelse return failData("plan.sizing.algorithm", .{});
    const timeout_seconds = spec.valueAsF64(timeout_val) orelse return failData("plan.sizing.algorithm", .{});
    const expected_alg = try sizingAlgorithmSpec(allocator, timeout_seconds);
    try require(try spec.jsonEquals(allocator, sizing.get("algorithm").?, expected_alg), "plan.sizing.algorithm", .{});

    const modes_val = plan.get("modes") orelse return failData("plan.sizing.pilot_iterations", .{});
    try require(modes_val == .array, "plan.sizing.pilot_iterations", .{});
    var modes: std.ArrayList([]const u8) = .empty;
    for (modes_val.array.items) |m| {
        try require(m == .string, "plan.sizing.pilot_iterations", .{});
        try modes.append(allocator, m.string);
    }

    const pilot_iters_val = sizing.get("pilot_iterations").?;
    try require(pilot_iters_val == .object and pilot_iters_val.object.count() == modes.items.len, "plan.sizing.pilot_iterations", .{});
    for (modes.items) |m| {
        try require(pilot_iters_val.object.contains(m), "plan.sizing.pilot_iterations", .{});
    }

    const tc_val = plan.get("thread_counts") orelse return failData("plan.thread_counts", .{});
    try require(tc_val == .array, "plan.thread_counts", .{});
    var thread_counts: std.ArrayList(i64) = .empty;
    for (tc_val.array.items) |tv| {
        try require(tv == .integer, "plan.thread_counts", .{});
        try thread_counts.append(allocator, tv.integer);
    }

    validateIterationPlanRanges(pilot_iters_val.object, thread_counts.items) catch |err| {
        return if (err == error.InvalidHarness) error.BenchmarkDataError else err;
    };

    const roles_val = plan.get("revision_roles") orelse return failData("plan.revision_roles", .{});
    try require(roles_val == .array, "plan.revision_roles", .{});
    var roles: std.ArrayList([]const u8) = .empty;
    for (roles_val.array.items) |rv| {
        try require(rv == .string, "plan.revision_roles", .{});
        try roles.append(allocator, rv.string);
    }

    const pairs_val = plan.get("pairs") orelse return failData("plan.pairs", .{});
    try require(pairs_val == .array, "plan.pairs", .{});
    var pairs: std.ArrayList(ObjectMap) = .empty;
    for (pairs_val.array.items) |pv| {
        try require(pv == .object, "plan.pairs", .{});
        try pairs.append(allocator, pv.object);
    }

    const expected_pilot_order = pilotOrderForPlan(allocator, pairs.items, roles.items, pilot_iters_val.object) catch {
        return failData("plan.sizing.pilot_order", .{});
    };
    var expected_order_arr = Array.init(allocator);
    for (expected_pilot_order) |po| try expected_order_arr.append(.{ .object = po });
    try require(
        try spec.jsonEquals(allocator, sizing.get("pilot_order").?, .{ .array = expected_order_arr }),
        "plan.sizing.pilot_order",
        .{},
    );

    const resolved_val = sizing.get("resolved").?;
    try require(resolved_val == .object, "plan sizing resolution", .{});
    const pilots_val = resolved_val.object.get("pilots") orelse return failData("plan sizing resolution", .{});
    try require(pilots_val == .array, "plan sizing resolution", .{});
    var pilot_records: std.ArrayList(ObjectMap) = .empty;
    for (pilots_val.array.items) |pr| {
        try require(pr == .object, "plan sizing resolution", .{});
        try pilot_records.append(allocator, pr.object);
    }

    const warmups = plan.get("warmups").?.integer;
    const samples = plan.get("samples").?.integer;
    const recomputed = resolveOneShotSizing(
        allocator,
        pilot_records.items,
        expected_pilot_order,
        modes.items,
        thread_counts.items,
        warmups,
        samples,
        timeout_seconds,
    ) catch |err| {
        return if (err == error.InvalidHarness) error.BenchmarkDataError else err;
    };

    try require(
        try spec.jsonEquals(allocator, plan.get("iterations").?, .{ .object = recomputed.plan }),
        "plan resolved iterations",
        .{},
    );
    try require(
        try spec.jsonEquals(allocator, resolved_val, .{ .object = recomputed.resolution }),
        "plan sizing resolution",
        .{},
    );
}

pub fn plannedPairSpecs(
    allocator: Allocator,
    modes: []const []const u8,
    thread_counts: []const i64,
) ![]ObjectMap {
    var pairs: std.ArrayList(ObjectMap) = .empty;
    for (modes) |mode| {
        var p: ObjectMap = .empty;
        try p.put(allocator, "pair_kind", .{ .string = "single-infrastructure" });
        try p.put(allocator, "pair_key", .{ .string = try std.fmt.allocPrint(allocator, "single-infrastructure/{s}", .{mode}) });
        try p.put(allocator, "left", .{ .string = "threads-disabled" });
        try p.put(allocator, "right", .{ .string = "threads-enabled" });
        try pairs.append(allocator, p);
    }
    const wls = [_][]const u8{ "hot", "atomic", "wait-notify", "spawn-join" };
    for (wls) |workload| {
        for (thread_counts) |threads| {
            var p: ObjectMap = .empty;
            if (modes.len == 2) {
                try p.put(allocator, "pair_kind", .{ .string = "runtime-parity" });
                try p.put(allocator, "pair_key", .{ .string = try std.fmt.allocPrint(allocator, "runtime/{s}/{d}", .{ workload, threads }) });
                try p.put(allocator, "left", .{ .string = "interpreter" });
                try p.put(allocator, "right", .{ .string = "aot" });
            } else {
                const mode = modes[0];
                try p.put(allocator, "pair_kind", .{ .string = "repeatability" });
                try p.put(allocator, "pair_key", .{ .string = try std.fmt.allocPrint(allocator, "runtime/{s}/{d}/{s}", .{ workload, threads, mode }) });
                try p.put(allocator, "left", .{ .string = try std.fmt.allocPrint(allocator, "{s}-a", .{mode}) });
                try p.put(allocator, "right", .{ .string = try std.fmt.allocPrint(allocator, "{s}-b", .{mode}) });
            }
            try pairs.append(allocator, p);
        }
    }
    var has_aot = false;
    for (modes) |m| {
        if (std.mem.eql(u8, m, "aot")) has_aot = true;
    }
    if (has_aot) {
        for (thread_counts) |threads| {
            var p: ObjectMap = .empty;
            try p.put(allocator, "pair_kind", .{ .string = "cancel-point-cost" });
            try p.put(allocator, "pair_key", .{ .string = try std.fmt.allocPrint(allocator, "cancel-points/hot/{d}", .{threads}) });
            try p.put(allocator, "left", .{ .string = "cancel-points-off" });
            try p.put(allocator, "right", .{ .string = "cancel-points-on" });
            try pairs.append(allocator, p);
        }
    }
    return pairs.toOwnedSlice(allocator);
}

pub fn expectedPairSpecsForPlan(allocator: Allocator, plan: ObjectMap) HarnessError![]ObjectMap {
    const modes_val = plan.get("modes") orelse return failData("plan.modes", .{});
    const tc_val = plan.get("thread_counts") orelse return failData("plan.thread_counts", .{});
    const iters_val = plan.get("iterations") orelse return failData("plan.iterations", .{});
    try require(modes_val == .array and tc_val == .array and iters_val == .object, "plan.pairs", .{});

    var modes: std.ArrayList([]const u8) = .empty;
    for (modes_val.array.items) |mv| {
        try require(mv == .string, "plan.modes", .{});
        try modes.append(allocator, mv.string);
    }
    var thread_counts: std.ArrayList(i64) = .empty;
    for (tc_val.array.items) |tv| {
        try require(tv == .integer, "plan.thread_counts", .{});
        try thread_counts.append(allocator, tv.integer);
    }
    for ([_][]const u8{ "hot", "atomic", "wait-notify", "spawn-join" }) |wl| {
        for (modes.items) |m| {
            const m_obj = iters_val.object.get(m) orelse return failData("plan iterations missing {s}", .{wl});
            try require(m_obj == .object and m_obj.object.contains(wl), "plan iterations missing {s}", .{wl});
        }
    }
    return plannedPairSpecs(allocator, modes.items, thread_counts.items);
}

pub fn prepareExpectedResults(
    allocator: Allocator,
    io: std.Io,
    iteration_plan: ObjectMap,
    modes: []const []const u8,
    thread_counts: []const i64,
) HarnessError!ObjectMap {
    const Req = struct {
        workload: []const u8,
        threads: i64,
        iterations: u64,
    };
    var requests: std.ArrayList(Req) = .empty;
    const appendUnique = struct {
        fn call(alloc: Allocator, list: *std.ArrayList(Req), r: Req) !void {
            for (list.items) |existing| {
                if (std.mem.eql(u8, existing.workload, r.workload) and
                    existing.threads == r.threads and
                    existing.iterations == r.iterations)
                {
                    return;
                }
            }
            try list.append(alloc, r);
        }
    }.call;

    for (modes) |mode| {
        const single_iters = try iterationCount(iteration_plan, mode, "single-hot", 1);
        try appendUnique(allocator, &requests, .{ .workload = "single-hot", .threads = 1, .iterations = single_iters });
        for ([_][]const u8{ "hot", "atomic", "wait-notify", "spawn-join" }) |wl| {
            for (thread_counts) |tc| {
                const iters = try iterationCount(iteration_plan, mode, wl, tc);
                try appendUnique(allocator, &requests, .{ .workload = wl, .threads = tc, .iterations = iters });
            }
        }
    }
    for (modes) |mode| {
        if (std.mem.eql(u8, mode, "aot")) {
            for (thread_counts) |tc| {
                const iters = try iterationCount(iteration_plan, "aot", "cancel-hot", tc);
                try appendUnique(allocator, &requests, .{ .workload = "hot", .threads = tc, .iterations = iters });
            }
        }
    }

    const total_started = std.Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds;
    var worst_ns: i64 = -1;
    var worst_req: Req = requests.items[0];
    for (requests.items) |r| {
        const started = std.Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds;
        _ = try spec.expectedResult(r.workload, r.threads, @as(i128, r.iterations), null);
        const elapsed: i64 = @intCast(@max(@as(i96, 0), std.Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds - started));
        if (elapsed > worst_ns) {
            worst_ns = elapsed;
            worst_req = r;
        }
    }
    const total_ns: i64 = @intCast(@max(@as(i96, 0), std.Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds - total_started));

    var worst_key: ObjectMap = .empty;
    try worst_key.put(allocator, "workload", .{ .string = worst_req.workload });
    try worst_key.put(allocator, "threads", .{ .integer = worst_req.threads });
    try spec.putU64Json(allocator, &worst_key, "iterations", worst_req.iterations);

    var out: ObjectMap = .empty;
    try out.put(allocator, "algorithm", .{ .string = "64-residue-xor-jump-ahead" });
    try out.put(allocator, "complexity", .{ .string = "O(64 * threads), independent of iteration count" });
    try out.put(allocator, "unique_keys", .{ .integer = @intCast(requests.items.len) });
    try out.put(allocator, "total_ns", .{ .integer = total_ns });
    try out.put(allocator, "worst_ns", .{ .integer = worst_ns });
    try out.put(allocator, "worst_key", .{ .object = worst_key });
    return out;
}

pub const Scenario = struct {
    key: []const u8,
    workload: []const u8,
    threads: i64,
};

pub fn plannedScenarios(allocator: Allocator, thread_counts: []const i64) ![]Scenario {
    var list: std.ArrayList(Scenario) = .empty;
    for ([_][]const u8{ "hot", "atomic", "wait-notify", "spawn-join" }) |wl| {
        for (thread_counts) |tc| {
            try list.append(allocator, .{
                .key = try std.fmt.allocPrint(allocator, "{s}/{d}", .{ wl, tc }),
                .workload = wl,
                .threads = tc,
            });
        }
    }
    return list.toOwnedSlice(allocator);
}
