//! Shared specification, guest math, CPU placement, and identity helpers for WASI thread benchmarks.
const std = @import("std");
const builtin = @import("builtin");
const benchmark_schema = @import("benchmark_schema.zig");
const benchmark_stats = @import("benchmark_stats.zig");
const process_control = @import("process_control.zig");

pub const Allocator = std.mem.Allocator;
pub const Value = std.json.Value;
pub const ObjectMap = std.json.ObjectMap;
pub const Array = std.json.Array;

pub const KIND = "wasi-thread-benchmark";
pub const REPORT_SCHEMA_VERSION: i64 = 16;
pub const WASI_MONOTONIC_CLOCK_ID = "wasi-monotonic";
pub const WASI_PROCESS_CPU_CLOCK_ID = "wasi-process-cputime";
pub const WASI_MONOTONIC_CLOCK_MODE = "monotonic";
pub const WASI_PROCESS_CPU_CLOCK_MODE = "process-cpu";
pub const REVISION_ROLES = [_][]const u8{ "baseline", "candidate" };
pub const SINGLE_REVISION_ROLES = [_][]const u8{"candidate"};
pub const COMPARISON_PURPOSES = [_][]const u8{ "candidate-evaluation", "noise-calibration" };
pub const MODES = [_][]const u8{ "interpreter", "aot" };
pub const THREAD_COUNTS = [_]i64{ 1, 2, 4, 8 };
pub const DEFAULT_THREAD_COUNTS = THREAD_COUNTS;
pub const MEASUREMENT_PLAN_IDENTITY_VERSION: i64 = 20;
pub const MEASUREMENT_PLAN_IDENTITY_KIND = "wasi-thread-measurement-plan";

pub const CPU_PLACEMENT_VERSION: i64 = 3;
pub const CPU_PLACEMENT_KIND = "fixed-linux-physical-core-affinity";
pub const REVISION_ARTIFACT_POLICY = "reuse-exact-artifacts-for-identical-noise-calibration-revisions";
pub const FIXTURE_SOURCE_POLICY = "candidate-measurement-fixtures-for-all-revisions";

pub const SIZING_ALGORITHM_VERSION: i64 = 13;
pub const SIZING_ALGORITHM_KIND = "fastest-valid-one-shot-pilot";
pub const SIZING_FORMULA =
    "round_up_3_significant_digits(ceil(P*target_duration_ns*safety_numerator/(E*safety_denominator)))";
pub const SIZING_CELL_ENVELOPE_FORMULA =
    "round_up_3_significant_digits(ceil(P*quality_floor_ns*acceleration_numerator/(E*acceleration_denominator)))";
pub const SIZING_CELL_SELECTION =
    "max(general_rounded_iterations, applicable_envelope_rounded_iterations)";

pub const MIN_TIMED_INTERVAL_MS: f64 = 1_250.0;
pub const TIMING_OVERHEAD_RATIO_LIMIT: f64 = 0.01;
pub const TARGET_BARRIER_NS: i64 = 12_456_000;
pub const TARGET_BARRIER_REQUIRED_INTERVAL_NS: i64 = 99 * TARGET_BARRIER_NS;
pub const MINIMUM_INTERVAL_HEADROOM_NS: i64 =
    @as(i64, @intFromFloat(MIN_TIMED_INTERVAL_MS * 1_000_000.0)) - TARGET_BARRIER_REQUIRED_INTERVAL_NS;

pub const SIZING_TARGET_NS: i64 = 1_750_000_000;
pub const SIZING_SAFETY_NUMERATOR: i64 = 20;
pub const SIZING_SAFETY_DENOMINATOR: i64 = 7;
pub const SIZING_SIGNIFICANT_DIGITS: usize = 3;
pub const PROJECTED_EVIDENCE_MINIMUM_NS: i64 =
    @divTrunc(SIZING_TARGET_NS * SIZING_SAFETY_NUMERATOR + SIZING_SAFETY_DENOMINATOR - 1, SIZING_SAFETY_DENOMINATOR);
pub const PILOT_CLOCK_RESOLUTION_MINIMUM_NS: i64 = 1_000_000;
pub const MAXIMUM_PILOT_CORRECTED_NS: i64 = 30_000_000_000;
pub const MAXIMUM_PILOT_HOST_WALL_NS: i64 = 33_000_000_000;
pub const WORKFLOW_JOB_TIMEOUT_NS: i64 = 330 * 60 * 1_000_000_000;
pub const JOB_NON_BENCHMARK_RESERVE_NS: i64 = 69 * 60 * 1_000_000_000;
pub const PROJECTED_BENCHMARK_LIMIT_NS: i64 = WORKFLOW_JOB_TIMEOUT_NS - JOB_NON_BENCHMARK_RESERVE_NS;
pub const AUXILIARY_INVOCATION_BUDGET_NS: i64 = 10 * 60 * 1_000_000_000;
pub const INT32_MAX: i64 = (1 << 31) - 1;
pub const SPAWN_JOIN_THREAD_LIFECYCLE_CAP: i64 = 27_000;
pub const TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD: i64 = 4;
pub const TRUSTED_BARRIER_PREFLIGHT_ITERATION_MULTIPLIER = "cpu-bound-affinity-logical-cpu-count";
pub const PREFLIGHT_ACCEPTANCE_RULE =
    "every probe must have timed_interval_ns >= minimum_timed_interval_ns and 99 * timing_overhead_ns < minimum_timed_interval_ns";
pub const ATOMIC_WAIT_PREFLIGHT_ITERATIONS: i64 = 1_000_000;

pub const AOT_VERSION: u32 = 11;
pub const MASK64: u64 = 0xFFFFFFFFFFFFFFFF;
pub const HOT_KERNEL_COUNTER_BASE: u64 = 0xD1B54A32D192ED03;

pub const CANCEL_POLL_X86_64 = [_]u8{
    0x83, 0xbb, 0xb0, 0x01, 0x00, 0x00, 0x00, 0x74, 0x0c,
};

pub const CANCEL_POLL_AARCH64 = [_]u8{
    0x70, 0xb2, 0x41, 0xb9, // 0xB941B270
    0x90, 0x00, 0x00, 0x34, // 0x34000090
};

pub fn cancelPollSignature(arch: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, arch, "x86_64")) return &CANCEL_POLL_X86_64;
    if (std.mem.eql(u8, arch, "aarch64")) return &CANCEL_POLL_AARCH64;
    return null;
}

pub const FixtureSpec = struct {
    name: []const u8,
    path: []const u8,
    sha256: []const u8,
};

pub const FIXTURES = [_]FixtureSpec{
    .{
        .name = "single",
        .path = "tests/benchmarks/wasi-threads/single.wasm",
        .sha256 = "376d3a19168164f84f628a398fdad2dc743967ff6068bec357aa9d952d9306b5",
    },
    .{
        .name = "threaded",
        .path = "tests/benchmarks/wasi-threads/threaded.wasm",
        .sha256 = "93378ef0c2e0051bfd7616b4724cf8a574747f74f6215cec99ddc07988e975f0",
    },
};

pub const FIXTURE_SOURCE_FILES = [_][]const u8{
    "kernel.h",
    "output.h",
    "timing.h",
    "single.c",
    "threaded.c",
    "build-fixtures.sh",
};

pub fn profileCounts(profile: []const u8) ?struct { warmups: i64, samples: i64 } {
    if (std.mem.eql(u8, profile, "authoritative")) return .{ .warmups = 2, .samples = 12 };
    if (std.mem.eql(u8, profile, "smoke")) return .{ .warmups = 1, .samples = 4 };
    return null;
}

pub fn atomicWaitPreflightRuns(profile: []const u8) ?i64 {
    if (std.mem.eql(u8, profile, "authoritative")) return 64;
    if (std.mem.eql(u8, profile, "smoke")) return 8;
    return null;
}

pub fn canonicalPlatformHost(platform_id: []const u8) ?struct { system: []const u8, machine: []const u8 } {
    if (std.mem.eql(u8, platform_id, "ubuntu-22.04-x86_64")) return .{ .system = "Linux", .machine = "x86_64" };
    if (std.mem.eql(u8, platform_id, "ubuntu-24.04-aarch64")) return .{ .system = "Linux", .machine = "aarch64" };
    return null;
}

pub fn cpuPlacementPolicyValue(allocator: Allocator) !Value {
    var obj: ObjectMap = .empty;
    try obj.put(allocator, "version", .{ .integer = CPU_PLACEMENT_VERSION });
    try obj.put(allocator, "kind", .{ .string = CPU_PLACEMENT_KIND });
    try obj.put(allocator, "physical_core_order", .{ .string = "descending-package-core" });
    try obj.put(allocator, "logical_cpu_order", .{ .string = "one-lowest-logical-cpu-per-core-before-siblings" });
    try obj.put(allocator, "single_hot_logical_cpus", .{ .integer = 1 });
    try obj.put(allocator, "single_hot_pair_logical_cpus", .{ .integer = 2 });
    try obj.put(allocator, "single_hot_pair_preference", .{ .string = "smt-siblings-then-separate-physical-cores" });
    try obj.put(allocator, "cpu_bound_logical_cpus", .{ .string = "min(available,workers)" });
    try obj.put(allocator, "threaded_logical_cpus", .{ .string = "min(available,workers+controller)" });
    try obj.put(allocator, "failure_policy", .{ .string = "fail-closed" });
    return .{ .object = obj };
}

pub fn pairExecutionPolicyValue(allocator: Allocator) !Value {
    var obj: ObjectMap = .empty;
    try obj.put(allocator, "default", .{ .string = "sequential-position-balanced" });
    try obj.put(allocator, "default_order", .{ .string = "condition-major-adjacent-revisions-four-position-cycle" });
    try obj.put(allocator, "default_balance", .{ .string = "each-revision-condition-occupies-each-quartet-position-once-per-cycle" });
    try obj.put(allocator, "default_estimator", .{ .string = "median-of-four-position-geometric-means" });
    try obj.put(allocator, "default_estimator_block_size", .{ .integer = 4 });
    try obj.put(allocator, "single_infrastructure", .{ .string = "concurrent-topology-paired-cpus" });
    try obj.put(allocator, "single_infrastructure_artifact", .{ .string = "same-enabled-runtime-binary" });
    try obj.put(allocator, "single_infrastructure_condition", .{ .string = "benchmark-runtime-thread-manager-enabled-or-disabled" });
    try obj.put(allocator, "single_infrastructure_assignment", .{ .string = "condition-order-maps-to-cpu-order-and-swaps-each-sample" });
    try obj.put(allocator, "overlap_requirement", .{ .string = "both-guest-process-host-intervals-overlap" });
    return .{ .object = obj };
}

pub fn guestClockPolicyValue(allocator: Allocator) !Value {
    var obj: ObjectMap = .empty;
    const entries = [_]struct { k: []const u8, clock_id: []const u8, arg: ?[]const u8 }{
        .{ .k = "single-hot", .clock_id = WASI_PROCESS_CPU_CLOCK_ID, .arg = null },
        .{ .k = "hot", .clock_id = WASI_PROCESS_CPU_CLOCK_ID, .arg = WASI_PROCESS_CPU_CLOCK_MODE },
        .{ .k = "atomic", .clock_id = WASI_PROCESS_CPU_CLOCK_ID, .arg = WASI_PROCESS_CPU_CLOCK_MODE },
        .{ .k = "wait-notify", .clock_id = WASI_MONOTONIC_CLOCK_ID, .arg = null },
        .{ .k = "spawn-join", .clock_id = WASI_MONOTONIC_CLOCK_ID, .arg = null },
        .{ .k = "trusted-hot-preflight", .clock_id = WASI_MONOTONIC_CLOCK_ID, .arg = WASI_MONOTONIC_CLOCK_MODE },
    };
    for (entries) |e| {
        var sub: ObjectMap = .empty;
        try sub.put(allocator, "clock_id", .{ .string = e.clock_id });
        try sub.put(allocator, "argument", if (e.arg) |a| .{ .string = a } else .null);
        try obj.put(allocator, e.k, .{ .object = sub });
    }
    return .{ .object = obj };
}

pub fn wasiSdkValue(allocator: Allocator) !Value {
    var obj: ObjectMap = .empty;
    try obj.put(allocator, "version", .{ .string = "25.0" });
    try obj.put(allocator, "clang", .{ .string = "clang version 19.1.5-wasi-sdk" });
    try obj.put(allocator, "archive", .{ .string = "wasi-sdk-25.0-x86_64-linux.tar.gz" });
    try obj.put(allocator, "archive_sha256", .{ .string = "52640dde13599bf127a95499e61d6d640256119456d1af8897ab6725bcf3d89c" });
    try obj.put(allocator, "url", .{ .string = "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-25/wasi-sdk-25.0-x86_64-linux.tar.gz" });
    try obj.put(allocator, "wasi_libc_revision", .{ .string = "574b88da481569b65a237cb80daf9a2d5aeaf82d" });
    return .{ .object = obj };
}

// ============================================================================
// Error Handling & Diagnostics State
// ============================================================================

pub const HarnessError = error{
    InvalidHarness,
    BenchmarkDataError,
    TimingQualityError,
    PreflightProbeError,
    OutOfMemory,
};

pub const TimingQualityInfo = struct {
    raw_elapsed_ns: i64 = 0,
    timing_overhead_ns: i64 = 0,
    elapsed_ns: i64 = 0,
    timing_overhead_ppm: i64 = 0,
    reason: []const u8 = "",
};

var last_error_buf: [4096]u8 = undefined;
var last_error_len: usize = 0;
pub var last_timing_quality: TimingQualityInfo = .{};

pub fn fail(comptime fmt: []const u8, args: anytype) HarnessError {
    const slice = std.fmt.bufPrint(&last_error_buf, fmt, args) catch blk: {
        const msg = "wasi-thread benchmark error (message truncated)";
        @memcpy(last_error_buf[0..msg.len], msg);
        break :blk last_error_buf[0..msg.len];
    };
    last_error_len = slice.len;
    return error.InvalidHarness;
}

pub fn failData(comptime fmt: []const u8, args: anytype) HarnessError {
    _ = fail(fmt, args) catch {};
    return error.BenchmarkDataError;
}

pub fn failTimingQuality(
    info: TimingQualityInfo,
    comptime fmt: []const u8,
    args: anytype,
) HarnessError {
    last_timing_quality = info;
    _ = fail(fmt, args) catch {};
    return error.TimingQualityError;
}

pub fn require(cond: bool, comptime fmt: []const u8, args: anytype) HarnessError!void {
    if (!cond) return failData(fmt, args);
}

pub fn lastErrorMessage() []const u8 {
    return last_error_buf[0..last_error_len];
}

// ============================================================================
// JSON, Hashing, and File Utilities
// ============================================================================

pub fn sha256Bytes(allocator: Allocator, data: []const u8) ![]const u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    return allocator.dupe(u8, &hex);
}

pub fn sha256File(allocator: Allocator, io: std.Io, path: []const u8) ![]const u8 {
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024)) catch {
        return fail("missing or unreadable file: {s}", .{path});
    };
    defer allocator.free(data);
    return sha256Bytes(allocator, data);
}

pub fn cacheKey(allocator: Allocator, value: Value) HarnessError![]const u8 {
    const key = benchmark_schema.cacheKey(allocator, value) catch return fail("invalid cache key", .{});
    return allocator.dupe(u8, &key);
}

pub fn isHexLen(s: []const u8, expected_len: usize) bool {
    if (s.len != expected_len) return false;
    for (s) |c| {
        if (!((c >= '0' and c <= '9') or (c >= 'a' and c <= 'f'))) return false;
    }
    return true;
}

pub fn isHexSha256(s: []const u8) bool {
    return isHexLen(s, 64);
}

pub fn isHexCommit(s: []const u8) bool {
    return isHexLen(s, 40);
}

pub fn atomicWriteText(allocator: Allocator, io: std.Io, path: []const u8, text: []const u8) HarnessError!void {
    if (std.fs.path.dirname(path)) |parent| {
        if (parent.len > 0) std.Io.Dir.cwd().createDirPath(io, parent) catch {};
    }
    var rand_bytes: [8]u8 = undefined;
    io.random(&rand_bytes);
    const tmp_path = try std.fmt.allocPrint(allocator, "{s}.{x}.tmp", .{
        path,
        std.mem.readInt(u64, &rand_bytes, .little),
    });
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = tmp_path, .data = text }) catch {
        return fail("failed to write file: {s}", .{path});
    };
    std.Io.Dir.cwd().rename(tmp_path, std.Io.Dir.cwd(), path, io) catch {
        return fail("failed to rename file: {s}", .{path});
    };
}

pub fn atomicWriteJson(allocator: Allocator, io: std.Io, path: []const u8, value: Value) HarnessError!void {
    const canonical = benchmark_schema.canonicalJson(allocator, value) catch {
        return fail("failed to serialize JSON for {s}", .{path});
    };
    const payload = try std.fmt.allocPrint(allocator, "{s}\n", .{canonical});
    try atomicWriteText(allocator, io, path, payload);
}

pub fn jsonClone(allocator: Allocator, value: Value) !Value {
    switch (value) {
        .null => return .null,
        .bool => |b| return .{ .bool = b },
        .integer => |i| return .{ .integer = i },
        .float => |f| return .{ .float = f },
        .number_string => |s| return .{ .number_string = try allocator.dupe(u8, s) },
        .string => |s| return .{ .string = try allocator.dupe(u8, s) },
        .array => |arr| {
            var out = Array.init(allocator);
            try out.ensureTotalCapacity(arr.items.len);
            for (arr.items) |item| {
                try out.append(try jsonClone(allocator, item));
            }
            return .{ .array = out };
        },
        .object => |obj| {
            var out: ObjectMap = .empty;
            var it = obj.iterator();
            while (it.next()) |entry| {
                const k = try allocator.dupe(u8, entry.key_ptr.*);
                const v = try jsonClone(allocator, entry.value_ptr.*);
                try out.put(allocator, k, v);
            }
            return .{ .object = out };
        },
    }
}

pub fn cloneObject(allocator: Allocator, obj: ObjectMap) HarnessError!ObjectMap {
    const v = jsonClone(allocator, .{ .object = obj }) catch return error.OutOfMemory;
    return v.object;
}

pub fn jsonEquals(allocator: Allocator, a: Value, b: Value) HarnessError!bool {
    const ca = benchmark_schema.canonicalJson(allocator, a) catch return fail("invalid JSON value", .{});
    defer allocator.free(ca);
    const cb = benchmark_schema.canonicalJson(allocator, b) catch return fail("invalid JSON value", .{});
    defer allocator.free(cb);
    return std.mem.eql(u8, ca, cb);
}

pub fn valueAsF64(val: Value) ?f64 {
    return switch (val) {
        .integer => |i| @floatFromInt(i),
        .float => |f| if (std.math.isFinite(f)) f else null,
        .number_string => |s| blk: {
            const f = std.fmt.parseFloat(f64, s) catch break :blk null;
            break :blk if (std.math.isFinite(f)) f else null;
        },
        else => null,
    };
}

pub fn valueAsI64(val: Value) ?i64 {
    return switch (val) {
        .integer => |i| i,
        .number_string => |s| std.fmt.parseInt(i64, s, 10) catch null,
        else => null,
    };
}

pub fn valueAsU64(val: Value) ?u64 {
    return switch (val) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        .number_string => |s| std.fmt.parseInt(u64, s, 10) catch null,
        else => null,
    };
}

pub fn canonicalMeasurementPlan(allocator: Allocator, plan: Value) HarnessError!Value {
    try require(plan == .object, "measurement plan comparison_purpose", .{});
    const obj = plan.object;
    const sizing_val = obj.get("sizing") orelse return failData("measurement plan comparison_purpose", .{});
    try require(
        obj.contains("comparison_purpose") and
            obj.contains("iterations") and
            sizing_val == .object and
            sizing_val.object.contains("resolved"),
        "measurement plan comparison_purpose",
        .{},
    );
    var cloned = try jsonClone(allocator, plan);
    _ = cloned.object.swapRemove("comparison_purpose");
    _ = cloned.object.swapRemove("iterations");
    var sizing_obj = cloned.object.getPtr("sizing").?;
    _ = sizing_obj.object.swapRemove("resolved");
    return cloned;
}

pub fn measurementPlanSha256(allocator: Allocator, plan: Value) HarnessError![]const u8 {
    const canonical_plan = try canonicalMeasurementPlan(allocator, plan);
    var wrapper: ObjectMap = .empty;
    try wrapper.put(allocator, "schema_version", .{ .integer = MEASUREMENT_PLAN_IDENTITY_VERSION });
    try wrapper.put(allocator, "kind", .{ .string = MEASUREMENT_PLAN_IDENTITY_KIND });
    try wrapper.put(allocator, "canonical_plan", canonical_plan);
    return cacheKey(allocator, .{ .object = wrapper });
}

pub fn isCalibrationCompatibleMeasurementPlanSha256(
    allocator: Allocator,
    plan: Value,
    expected_sha256: []const u8,
) HarnessError!bool {
    const direct = try measurementPlanSha256(allocator, plan);
    if (std.mem.eql(u8, expected_sha256, direct)) return true;
    if (plan != .object) return false;
    const preflight_v = plan.object.get("scheduler_barrier_preflight") orelse return false;
    if (preflight_v != .object) return false;
    const enabled_v = preflight_v.object.get("enabled") orelse return false;
    if (enabled_v != .bool or enabled_v.bool) return false;
    const modes_v = plan.object.get("modes") orelse return false;
    if (modes_v != .array) return false;
    var has_aot = false;
    for (modes_v.array.items) |mv| {
        if (mv == .string and std.mem.eql(u8, mv.string, "aot")) {
            has_aot = true;
            break;
        }
    }
    if (!has_aot) return false;
    const tc_v = plan.object.get("thread_counts") orelse return false;
    if (tc_v != .array) return false;
    var calibrated_plan = try jsonClone(allocator, plan);
    const pf_ptr = calibrated_plan.object.getPtr("scheduler_barrier_preflight").?;
    try pf_ptr.object.put(allocator, "enabled", .{ .bool = true });
    try pf_ptr.object.put(
        allocator,
        "probe_count",
        .{ .integer = TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD * @as(i64, @intCast(tc_v.array.items.len)) },
    );
    const calibrated_sha = try measurementPlanSha256(allocator, calibrated_plan);
    return std.mem.eql(u8, expected_sha256, calibrated_sha);
}

// ============================================================================
// Guest Math & Verification
// ============================================================================

pub fn rotateLeftU64(value: u64, bits: u64) u64 {
    const shift: u6 = @intCast(bits & 63);
    if (shift == 0) return value;
    return std.math.rotl(u64, value, shift);
}

pub fn xorUpto(value: i128) u64 {
    if (value < 0) return 0;
    const v: u64 = @intCast(value);
    return switch (v & 3) {
        0 => v,
        1 => 1,
        2 => v +% 1,
        else => 0,
    };
}

pub fn xorWrappedU58Range(start: u64, count: i128) HarnessError!u64 {
    const modulus: i128 = @as(i128, 1) << 58;
    if (count < 0 or count > modulus) {
        return fail("hot-kernel grouped range exceeds uint64 iteration space", .{});
    }
    if (count == 0 or count == modulus) return 0;
    const s: i128 = @mod(@as(i128, start), modulus);
    const end: i128 = s + count;
    if (end <= modulus) {
        return xorUpto(end - 1) ^ xorUpto(s - 1);
    }
    return xorUpto(modulus - 1) ^ xorUpto(s - 1) ^ xorUpto(end - modulus - 1);
}

pub fn hotKernelJumpAhead(seed: u64, iterations: i128) HarnessError!u64 {
    if (iterations < 0 or iterations > @as(i128, MASK64)) {
        return fail("hot-kernel iterations must fit uint64", .{});
    }
    const iters_u64: u64 = @intCast(iterations);
    var value = rotateLeftU64(seed, 7 *% iters_u64);
    const limit: usize = @intCast(@min(@as(u64, 64), iters_u64));
    for (0..limit) |residue_usize| {
        const residue: u64 = @intCast(residue_usize);
        const terms: i128 = @divTrunc(@as(i128, iters_u64 - 1 - residue), 64) + 1;
        const first: u64 = HOT_KERNEL_COUNTER_BASE +% residue;
        var grouped_xor: u64 = (try xorWrappedU58Range(first >> 6, terms)) << 6;
        if ((terms & 1) != 0) {
            grouped_xor ^= first & 63;
        }
        value ^= rotateLeftU64(grouped_xor, 7 *% (iters_u64 - 1 - residue));
    }
    return value;
}

pub fn workerSeed(index: i64) u64 {
    const idx_u64: u64 = @intCast(index);
    return 0x243F6A8885A308D3 ^ (0x9E3779B97F4A7C15 *% (idx_u64 +% 1));
}

pub fn expectedGuestClockId(workload: []const u8) HarnessError![]const u8 {
    if (std.mem.eql(u8, workload, "single-hot") or
        std.mem.eql(u8, workload, "hot") or
        std.mem.eql(u8, workload, "atomic"))
    {
        return WASI_PROCESS_CPU_CLOCK_ID;
    }
    if (std.mem.eql(u8, workload, "wait-notify") or
        std.mem.eql(u8, workload, "spawn-join") or
        std.mem.eql(u8, workload, "trusted-hot-preflight"))
    {
        return WASI_MONOTONIC_CLOCK_ID;
    }
    return fail("unsupported workload clock policy {s}", .{workload});
}

pub const ExpectedResult = struct {
    kind: []const u8 = "wasi-thread-benchmark-result",
    workload: []const u8,
    threads: i64,
    iterations: u64,
    operations: u64,
    checksum: u64,
    clock_id: []const u8,
    metric_kind: []const u8,
    timed_loop_backedges: u64,
    clock_calls_in_timed_loop: i64 = 0,

    pub fn toObjectMap(self: ExpectedResult, allocator: Allocator) !ObjectMap {
        var obj: ObjectMap = .empty;
        try obj.put(allocator, "kind", .{ .string = self.kind });
        try obj.put(allocator, "workload", .{ .string = self.workload });
        try obj.put(allocator, "threads", .{ .integer = self.threads });
        try putU64Json(allocator, &obj, "iterations", self.iterations);
        try putU64Json(allocator, &obj, "operations", self.operations);
        try putU64Json(allocator, &obj, "checksum", self.checksum);
        try obj.put(allocator, "clock_id", .{ .string = self.clock_id });
        try obj.put(allocator, "metric_kind", .{ .string = self.metric_kind });
        try putU64Json(allocator, &obj, "timed_loop_backedges", self.timed_loop_backedges);
        try obj.put(allocator, "clock_calls_in_timed_loop", .{ .integer = self.clock_calls_in_timed_loop });
        return obj;
    }
};

pub fn putU64Json(allocator: Allocator, obj: *ObjectMap, key: []const u8, val: u64) !void {
    if (val <= std.math.maxInt(i64)) {
        try obj.put(allocator, key, .{ .integer = @intCast(val) });
    } else {
        try obj.put(allocator, key, .{
            .number_string = try std.fmt.allocPrint(allocator, "{d}", .{val}),
        });
    }
}

pub fn expectedResult(
    workload: []const u8,
    threads: i64,
    iterations: i128,
    clock_id_override: ?[]const u8,
) HarnessError!ExpectedResult {
    if (threads <= 0 or threads > 8 or iterations < 0 or iterations > @divTrunc(@as(i128, MASK64), @as(i128, threads))) {
        return fail("expected-result operations must fit uint64", .{});
    }
    const threads_u64: u64 = @intCast(threads);
    const iters_u64: u64 = @intCast(iterations);
    const operations: u64 = threads_u64 * iters_u64;
    const checksum: u64 = if (std.mem.eql(u8, workload, "single-hot"))
        try hotKernelJumpAhead(workerSeed(0), iterations)
    else if (std.mem.eql(u8, workload, "hot")) blk: {
        var sum: u64 = 0;
        var idx: i64 = 0;
        while (idx < threads) : (idx += 1) {
            sum +%= try hotKernelJumpAhead(workerSeed(idx), iterations);
        }
        break :blk sum;
    } else if (std.mem.eql(u8, workload, "atomic") or std.mem.eql(u8, workload, "wait-notify"))
        operations
    else if (std.mem.eql(u8, workload, "spawn-join")) blk: {
        const tri: i128 = @divExact(@as(i128, threads) * @as(i128, threads + 1), 2);
        const prod: i128 = iterations * tri;
        if (prod > @as(i128, MASK64)) {
            return fail("spawn-join checksum must fit uint64", .{});
        }
        break :blk @intCast(prod);
    } else return fail("unknown workload {s}", .{workload});

    const resolved_clock = if (clock_id_override) |cid| cid else try expectedGuestClockId(workload);
    const metric_kind = if (std.mem.eql(u8, workload, "spawn-join"))
        "spawn-join-lifecycle"
    else
        "steady-state-kernel";
    const backedges: u64 = if (std.mem.eql(u8, workload, "single-hot") or std.mem.eql(u8, workload, "hot"))
        operations
    else
        0;

    return .{
        .workload = workload,
        .threads = threads,
        .iterations = iters_u64,
        .operations = operations,
        .checksum = checksum,
        .clock_id = resolved_clock,
        .metric_kind = metric_kind,
        .timed_loop_backedges = backedges,
        .clock_calls_in_timed_loop = 0,
    };
}

const GUEST_ALL_KEYS = [_][]const u8{
    "kind",
    "workload",
    "threads",
    "iterations",
    "operations",
    "checksum",
    "clock_id",
    "metric_kind",
    "timed_loop_backedges",
    "clock_calls_in_timed_loop",
    "raw_elapsed_ns",
    "timing_overhead_ns",
    "elapsed_ns",
    "timing_overhead_ppm",
};

pub fn parseGuestResult(
    allocator: Allocator,
    stdout: []const u8,
    expected: ExpectedResult,
    min_interval_ns: i64,
    enforce_timing_quality: bool,
) HarnessError!ObjectMap {
    var non_empty_line: ?[]const u8 = null;
    var line_count: usize = 0;
    var it = std.mem.splitScalar(u8, stdout, '\n');
    while (it.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len > 0) {
            non_empty_line = line;
            line_count += 1;
        }
    }
    if (line_count != 1) {
        return fail("expected one guest JSON line, got {d}", .{line_count});
    }
    const parsed = std.json.parseFromSlice(Value, allocator, non_empty_line.?, .{
        .duplicate_field_behavior = .@"error",
        .parse_numbers = false,
    }) catch |err| switch (err) {
        error.DuplicateField => return fail("guest output contains duplicate key", .{}),
        else => return fail("guest output is not JSON: {s}", .{non_empty_line.?}),
    };
    if (parsed.value != .object) {
        return fail("guest output is not JSON: {s}", .{non_empty_line.?});
    }
    const raw_obj = parsed.value.object;
    if (raw_obj.count() != GUEST_ALL_KEYS.len) {
        return fail("guest timing fields mismatch", .{});
    }
    for (GUEST_ALL_KEYS) |k| {
        if (!raw_obj.contains(k)) {
            return fail("guest timing fields mismatch: missing {s}", .{k});
        }
    }

    const kind_val = raw_obj.get("kind").?;
    if (kind_val != .string or !std.mem.eql(u8, kind_val.string, expected.kind)) {
        return fail("guest result mismatch for kind", .{});
    }
    const wl_val = raw_obj.get("workload").?;
    if (wl_val != .string or !std.mem.eql(u8, wl_val.string, expected.workload)) {
        return fail("guest result mismatch for workload", .{});
    }
    const clock_val = raw_obj.get("clock_id").?;
    if (clock_val != .string or !std.mem.eql(u8, clock_val.string, expected.clock_id)) {
        return fail("guest result mismatch for clock_id", .{});
    }
    const metric_val = raw_obj.get("metric_kind").?;
    if (metric_val != .string or !std.mem.eql(u8, metric_val.string, expected.metric_kind)) {
        return fail("guest result mismatch for metric_kind", .{});
    }

    const guest_threads = valueAsI64(raw_obj.get("threads").?) orelse return fail("guest result mismatch for threads", .{});
    if (guest_threads != expected.threads) return fail("guest result mismatch for threads", .{});

    const guest_iters = valueAsU64(raw_obj.get("iterations").?) orelse return fail("guest result mismatch for iterations", .{});
    if (guest_iters != expected.iterations) return fail("guest result mismatch for iterations", .{});

    const guest_ops = valueAsU64(raw_obj.get("operations").?) orelse return fail("guest result mismatch for operations", .{});
    if (guest_ops != expected.operations) return fail("guest result mismatch for operations", .{});

    const guest_checksum = valueAsU64(raw_obj.get("checksum").?) orelse return fail("guest result mismatch for checksum", .{});
    if (guest_checksum != expected.checksum) return fail("guest result mismatch for checksum", .{});

    const guest_backedges = valueAsU64(raw_obj.get("timed_loop_backedges").?) orelse return fail("guest result mismatch for timed_loop_backedges", .{});
    if (guest_backedges != expected.timed_loop_backedges) return fail("guest result mismatch for timed_loop_backedges", .{});

    const guest_clock_calls = valueAsI64(raw_obj.get("clock_calls_in_timed_loop").?) orelse return fail("guest result mismatch for clock_calls_in_timed_loop", .{});
    if (guest_clock_calls != expected.clock_calls_in_timed_loop) return fail("guest result mismatch for clock_calls_in_timed_loop", .{});

    const raw = valueAsI64(raw_obj.get("raw_elapsed_ns").?) orelse return fail("guest timing raw_elapsed_ns must be an integer", .{});
    const overhead = valueAsI64(raw_obj.get("timing_overhead_ns").?) orelse return fail("guest timing timing_overhead_ns must be an integer", .{});
    const elapsed = valueAsI64(raw_obj.get("elapsed_ns").?) orelse return fail("guest timing elapsed_ns must be an integer", .{});
    const overhead_ppm = valueAsI64(raw_obj.get("timing_overhead_ppm").?) orelse return fail("guest timing timing_overhead_ppm must be an integer", .{});

    if (raw <= 0 or overhead < 0 or elapsed <= 0) {
        return fail("guest timing values must be positive", .{});
    }
    if (raw - overhead != elapsed) {
        return fail("guest elapsed_ns must equal raw_elapsed_ns - overhead", .{});
    }
    const expected_ppm: i64 = @intCast(@divTrunc(@as(i128, overhead) * 1_000_000, @as(i128, raw)));
    if (overhead_ppm != expected_ppm) {
        return fail("guest timing_overhead_ppm is inconsistent", .{});
    }
    if (enforce_timing_quality and 99 * @as(i128, overhead) >= @as(i128, elapsed)) {
        const pct = @as(f64, @floatFromInt(overhead)) / @as(f64, @floatFromInt(raw)) * 100.0;
        return failTimingQuality(.{
            .raw_elapsed_ns = raw,
            .timing_overhead_ns = overhead,
            .elapsed_ns = elapsed,
            .timing_overhead_ppm = overhead_ppm,
            .reason = "timing-overhead",
        }, "guest timing overhead {d:.3}% is not below 1%", .{pct});
    }
    if (enforce_timing_quality and elapsed < min_interval_ns) {
        return failTimingQuality(.{
            .raw_elapsed_ns = raw,
            .timing_overhead_ns = overhead,
            .elapsed_ns = elapsed,
            .timing_overhead_ppm = overhead_ppm,
            .reason = "minimum-timed-interval",
        }, "guest timed interval {d}ns is below required {d}ns", .{ elapsed, min_interval_ns });
    }

    var out = try expected.toObjectMap(allocator);
    try out.put(allocator, "raw_elapsed_ns", .{ .integer = raw });
    try out.put(allocator, "timing_overhead_ns", .{ .integer = overhead });
    try out.put(allocator, "elapsed_ns", .{ .integer = elapsed });
    try out.put(allocator, "timing_overhead_ppm", .{ .integer = overhead_ppm });
    return out;
}

pub fn classifyGuestFailure(allocator: Allocator, stderr: []const u8, workload: []const u8) ![]const u8 {
    if (std.mem.indexOf(u8, stderr, "outcome=backend-error") != null) return "atomic-wait-backend-error";
    if (std.mem.indexOf(u8, stderr, "outcome=unexpected-timeout") != null or std.mem.indexOf(u8, stderr, "failed: 11") != null) return "atomic-wait-unexpected-timeout";
    if (std.mem.indexOf(u8, stderr, "outcome=cancelled") != null) return "atomic-wait-cancelled";
    if (std.mem.indexOf(u8, stderr, "outcome=closed") != null) return "atomic-wait-closed";
    if (std.mem.indexOf(u8, stderr, "controller barrier failed:") != null) return "controller-barrier-failure";
    if (std.mem.indexOf(u8, stderr, "failed: 13") != null) return "barrier-value-mismatch";
    if (std.mem.indexOf(u8, stderr, "failed: 12") != null) return "atomic-wait-invalid-result";
    if (std.mem.indexOf(u8, stderr, "failed: 10") != null) return "barrier-peer-abort";
    if (std.mem.indexOf(u8, stderr, "worker[") != null and std.mem.indexOf(u8, stderr, "failed:") != null) return "barrier-unclassified-worker-failure";
    return std.fmt.allocPrint(allocator, "{s}-guest-failure", .{workload});
}

// ============================================================================
// CPU Placement & Pairing
// ============================================================================

pub const TopologyEntry = struct {
    cpu: i64,
    package_id: i64,
    core_id: i64,
};

const PhysicalCoreGroup = struct {
    package_id: i64,
    core_id: i64,
    logical_cpus: std.ArrayList(i64),
};

fn descCoreGroupLessThan(_: void, a: PhysicalCoreGroup, b: PhysicalCoreGroup) bool {
    if (a.package_id != b.package_id) return a.package_id > b.package_id;
    return a.core_id > b.core_id;
}

pub fn cpuPlacementFromTopology(
    allocator: Allocator,
    allowed_logical_cpus: []const i64,
    topology: []const TopologyEntry,
    thread_counts: []const i64,
    taskset_version: []const u8,
) HarnessError!ObjectMap {
    if (allowed_logical_cpus.len == 0 or topology.len != allowed_logical_cpus.len) {
        return fail("CPU placement requires a complete unique allowed CPU set", .{});
    }
    const allowed_sorted = try allocator.dupe(i64, allowed_logical_cpus);
    std.mem.sort(i64, allowed_sorted, {}, std.sort.asc(i64));
    for (allowed_sorted, 0..) |cpu, idx| {
        if (cpu < 0) return fail("CPU placement requires a complete unique allowed CPU set", .{});
        if (idx > 0 and allowed_sorted[idx - 1] == cpu) {
            return fail("CPU placement requires a complete unique allowed CPU set", .{});
        }
    }

    var topo_map: std.AutoHashMapUnmanaged(i64, struct { pkg: i64, core: i64 }) = .empty;
    defer topo_map.deinit(allocator);
    for (topology) |entry| {
        if (topo_map.contains(entry.cpu)) {
            return fail("CPU placement requires a complete unique allowed CPU set", .{});
        }
        var in_allowed = false;
        for (allowed_sorted) |c| {
            if (c == entry.cpu) in_allowed = true;
        }
        if (!in_allowed) {
            return fail("CPU placement requires a complete unique allowed CPU set", .{});
        }
        try topo_map.put(allocator, entry.cpu, .{ .pkg = entry.package_id, .core = entry.core_id });
    }
    if (topo_map.count() != allowed_sorted.len) {
        return fail("CPU placement requires a complete unique allowed CPU set", .{});
    }

    var groups: std.ArrayList(PhysicalCoreGroup) = .empty;
    for (allowed_logical_cpus) |cpu| {
        const tc = topo_map.get(cpu).?;
        var found = false;
        for (groups.items) |*g| {
            if (g.package_id == tc.pkg and g.core_id == tc.core) {
                try g.logical_cpus.append(allocator, cpu);
                found = true;
                break;
            }
        }
        if (!found) {
            var list: std.ArrayList(i64) = .empty;
            try list.append(allocator, cpu);
            try groups.append(allocator, .{
                .package_id = tc.pkg,
                .core_id = tc.core,
                .logical_cpus = list,
            });
        }
    }
    for (groups.items) |*g| {
        std.mem.sort(i64, g.logical_cpus.items, {}, std.sort.asc(i64));
    }
    std.mem.sort(PhysicalCoreGroup, groups.items, {}, descCoreGroupLessThan);

    var physical_core_groups_arr = Array.init(allocator);
    var ordered_logical_cpus: std.ArrayList(i64) = .empty;

    for (groups.items) |g| {
        try ordered_logical_cpus.append(allocator, g.logical_cpus.items[0]);
        var l_arr = Array.init(allocator);
        for (g.logical_cpus.items) |c| try l_arr.append(.{ .integer = c });
        var g_obj: ObjectMap = .empty;
        try g_obj.put(allocator, "physical_package_id", .{ .integer = g.package_id });
        try g_obj.put(allocator, "core_id", .{ .integer = g.core_id });
        try g_obj.put(allocator, "logical_cpus", .{ .array = l_arr });
        try physical_core_groups_arr.append(.{ .object = g_obj });
    }
    for (groups.items) |g| {
        for (g.logical_cpus.items[1..]) |c| {
            try ordered_logical_cpus.append(allocator, c);
        }
    }

    var direct_pair_assignment: ObjectMap = .empty;
    var found_sibling = false;
    for (groups.items) |g| {
        if (g.logical_cpus.items.len >= 2) {
            try direct_pair_assignment.put(allocator, "kind", .{ .string = "smt-siblings" });
            var pair_arr = Array.init(allocator);
            try pair_arr.append(.{ .integer = g.logical_cpus.items[0] });
            try pair_arr.append(.{ .integer = g.logical_cpus.items[1] });
            try direct_pair_assignment.put(allocator, "logical_cpus", .{ .array = pair_arr });
            found_sibling = true;
            break;
        }
    }
    if (!found_sibling) {
        if (groups.items.len >= 2) {
            try direct_pair_assignment.put(allocator, "kind", .{ .string = "separate-physical-cores" });
            var pair_arr = Array.init(allocator);
            try pair_arr.append(.{ .integer = groups.items[0].logical_cpus.items[0] });
            try pair_arr.append(.{ .integer = groups.items[1].logical_cpus.items[0] });
            try direct_pair_assignment.put(allocator, "logical_cpus", .{ .array = pair_arr });
        } else {
            return fail("CPU placement requires two logical CPUs for concurrent single-infrastructure pairs", .{});
        }
    }

    var allowed_arr = Array.init(allocator);
    for (allowed_sorted) |c| try allowed_arr.append(.{ .integer = c });

    var ordered_arr = Array.init(allocator);
    for (ordered_logical_cpus.items) |c| try ordered_arr.append(.{ .integer = c });

    var single_hot_arr = Array.init(allocator);
    try single_hot_arr.append(.{ .integer = ordered_logical_cpus.items[0] });

    var cpu_bound_obj: ObjectMap = .empty;
    var threaded_obj: ObjectMap = .empty;
    for (thread_counts) |threads| {
        const key_str = try std.fmt.allocPrint(allocator, "{d}", .{threads});
        const t_usize: usize = @intCast(threads);
        const cb_len = @min(ordered_logical_cpus.items.len, t_usize);
        var cb_arr = Array.init(allocator);
        for (ordered_logical_cpus.items[0..cb_len]) |c| try cb_arr.append(.{ .integer = c });
        try cpu_bound_obj.put(allocator, key_str, .{ .array = cb_arr });

        const th_len = @min(ordered_logical_cpus.items.len, t_usize + 1);
        var th_arr = Array.init(allocator);
        for (ordered_logical_cpus.items[0..th_len]) |c| try th_arr.append(.{ .integer = c });
        try threaded_obj.put(allocator, key_str, .{ .array = th_arr });
    }

    var assignments: ObjectMap = .empty;
    try assignments.put(allocator, "single-hot", .{ .array = single_hot_arr });
    try assignments.put(allocator, "single-hot-pair", .{ .object = direct_pair_assignment });
    try assignments.put(allocator, "cpu-bound", .{ .object = cpu_bound_obj });
    try assignments.put(allocator, "threaded", .{ .object = threaded_obj });

    var out: ObjectMap = .empty;
    try out.put(allocator, "policy", try cpuPlacementPolicyValue(allocator));
    try out.put(allocator, "taskset_version", .{ .string = try allocator.dupe(u8, taskset_version) });
    try out.put(allocator, "allowed_logical_cpus", .{ .array = allowed_arr });
    try out.put(allocator, "physical_core_groups", .{ .array = physical_core_groups_arr });
    try out.put(allocator, "ordered_logical_cpus", .{ .array = ordered_arr });
    try out.put(allocator, "assignments", .{ .object = assignments });
    return out;
}

fn readSysfsInt(io: std.Io, path: []const u8) ?i64 {
    var buf: [64]u8 = undefined;
    const slice = std.Io.Dir.cwd().readFile(io, path, &buf) catch return null;
    const trimmed = std.mem.trim(u8, slice, " \t\r\n");
    return std.fmt.parseInt(i64, trimmed, 10) catch null;
}

pub fn discoverCpuPlacement(
    allocator: Allocator,
    io: std.Io,
    thread_counts: []const i64,
) HarnessError!ObjectMap {
    if (builtin.os.tag != .linux) {
        return fail("fixed CPU placement requires Linux sched_getaffinity", .{});
    }
    var mask: [128]u8 = @splat(0);
    const rc = std.os.linux.syscall3(.sched_getaffinity, 0, mask.len, @intFromPtr(&mask));
    const signed_rc: isize = @bitCast(rc);
    if (signed_rc <= 0) {
        return fail("fixed CPU placement requires Linux sched_getaffinity", .{});
    }
    const bytes: usize = @intCast(signed_rc);
    var allowed: std.ArrayList(i64) = .empty;
    for (mask[0..@min(bytes, mask.len)], 0..) |b, byte_idx| {
        if (b != 0) {
            for (0..8) |bit| {
                if ((b & (@as(u8, 1) << @intCast(bit))) != 0) {
                    try allowed.append(allocator, @intCast(byte_idx * 8 + bit));
                }
            }
        }
    }
    if (allowed.items.len == 0) {
        return fail("fixed CPU placement requires Linux sched_getaffinity", .{});
    }
    const taskset_ver = benchmark_schema.commandIdentity(allocator, io, &.{ "taskset", "--version" }) catch {
        return fail("fixed CPU placement requires taskset", .{});
    };
    if (std.mem.eql(u8, taskset_ver, "unavailable") or taskset_ver.len == 0) {
        return fail("fixed CPU placement requires taskset", .{});
    }
    var topology: std.ArrayList(TopologyEntry) = .empty;
    for (allowed.items) |cpu| {
        const pkg_path = try std.fmt.allocPrint(allocator, "/sys/devices/system/cpu/cpu{d}/topology/physical_package_id", .{cpu});
        const core_path = try std.fmt.allocPrint(allocator, "/sys/devices/system/cpu/cpu{d}/topology/core_id", .{cpu});
        const pkg = readSysfsInt(io, pkg_path) orelse return fail("cannot resolve physical topology for allowed CPU {d}", .{cpu});
        const core = readSysfsInt(io, core_path) orelse return fail("cannot resolve physical topology for allowed CPU {d}", .{cpu});
        try topology.append(allocator, .{ .cpu = cpu, .package_id = pkg, .core_id = core });
    }
    return cpuPlacementFromTopology(allocator, allowed.items, topology.items, thread_counts, taskset_ver);
}

pub fn validateCpuPlacement(
    allocator: Allocator,
    cpu_placement_val: Value,
    thread_counts: []const i64,
) HarnessError!void {
    try require(cpu_placement_val == .object, "metadata.cpu_placement policy", .{});
    const cp = cpu_placement_val.object;
    const policy_val = cp.get("policy") orelse return failData("metadata.cpu_placement policy", .{});
    const expected_policy = try cpuPlacementPolicyValue(allocator);
    try require(try jsonEquals(allocator, policy_val, expected_policy), "metadata.cpu_placement policy", .{});
    const taskset_val = cp.get("taskset_version") orelse return failData("metadata.cpu_placement policy", .{});
    try require(taskset_val == .string and taskset_val.string.len > 0, "metadata.cpu_placement policy", .{});

    const allowed_val = cp.get("allowed_logical_cpus") orelse return failData("metadata.cpu_placement topology", .{});
    const groups_val = cp.get("physical_core_groups") orelse return failData("metadata.cpu_placement topology", .{});
    try require(
        allowed_val == .array and allowed_val.array.items.len > 0 and
            groups_val == .array and groups_val.array.items.len > 0,
        "metadata.cpu_placement topology",
        .{},
    );
    var allowed: std.ArrayList(i64) = .empty;
    var allowed_set: std.AutoHashMapUnmanaged(i64, void) = .empty;
    for (allowed_val.array.items, 0..) |item, idx| {
        try require(item == .integer and item.integer >= 0, "metadata.cpu_placement topology", .{});
        const c = item.integer;
        if (allowed_set.contains(c)) return failData("metadata.cpu_placement topology", .{});
        if (idx > 0 and allowed.items[idx - 1] > c) return failData("metadata.cpu_placement topology", .{});
        try allowed.append(allocator, c);
        try allowed_set.put(allocator, c, {});
    }

    var topology: std.ArrayList(TopologyEntry) = .empty;
    var seen_topo: std.AutoHashMapUnmanaged(i64, void) = .empty;
    for (groups_val.array.items) |g_val| {
        try require(g_val == .object and g_val.object.count() == 3, "metadata.cpu_placement physical core group", .{});
        const g_obj = g_val.object;
        const pkg_v = g_obj.get("physical_package_id") orelse return failData("metadata.cpu_placement physical core group", .{});
        const core_v = g_obj.get("core_id") orelse return failData("metadata.cpu_placement physical core group", .{});
        const lcpus_v = g_obj.get("logical_cpus") orelse return failData("metadata.cpu_placement physical core group", .{});
        try require(
            pkg_v == .integer and core_v == .integer and
                lcpus_v == .array and lcpus_v.array.items.len > 0,
            "metadata.cpu_placement physical core group",
            .{},
        );
        for (lcpus_v.array.items) |c_val| {
            try require(c_val == .integer, "metadata.cpu_placement logical CPU membership", .{});
            const c = c_val.integer;
            try require(allowed_set.contains(c) and !seen_topo.contains(c), "metadata.cpu_placement logical CPU membership", .{});
            try seen_topo.put(allocator, c, {});
            try topology.append(allocator, .{
                .cpu = c,
                .package_id = pkg_v.integer,
                .core_id = core_v.integer,
            });
        }
    }
    try require(seen_topo.count() == allowed.items.len, "metadata.cpu_placement complete topology", .{});
    const expected_cp = cpuPlacementFromTopology(
        allocator,
        allowed.items,
        topology.items,
        thread_counts,
        taskset_val.string,
    ) catch return failData("metadata.cpu_placement assignments", .{});
    try require(try jsonEquals(allocator, cpu_placement_val, .{ .object = expected_cp }), "metadata.cpu_placement assignments", .{});
}

pub fn cpuAffinityFor(
    allocator: Allocator,
    cpu_placement: ObjectMap,
    workload: []const u8,
    threads: i64,
) HarnessError![]i64 {
    const assignments_val = cpu_placement.get("assignments") orelse return fail("CPU placement has no assignments", .{});
    if (assignments_val != .object) return fail("CPU placement has no assignments", .{});
    const assignments = assignments_val.object;
    if (std.mem.eql(u8, workload, "single-hot")) {
        const sh = assignments.get("single-hot") orelse return fail("CPU placement has no single-hot assignment", .{});
        if (sh != .array) return fail("CPU placement has no single-hot assignment", .{});
        const out = try allocator.alloc(i64, sh.array.items.len);
        for (sh.array.items, 0..) |v, idx| {
            out[idx] = valueAsI64(v) orelse return fail("invalid single-hot CPU", .{});
        }
        return out;
    }
    const kind = if (std.mem.eql(u8, workload, "hot") or std.mem.eql(u8, workload, "atomic"))
        "cpu-bound"
    else
        "threaded";
    const kind_val = assignments.get(kind) orelse return fail("CPU placement has no assignment for {d} threads", .{threads});
    if (kind_val != .object) return fail("CPU placement has no assignment for {d} threads", .{threads});
    var buf: [32]u8 = undefined;
    const key_str = std.fmt.bufPrint(&buf, "{d}", .{threads}) catch unreachable;
    const arr_val = kind_val.object.get(key_str) orelse return fail("CPU placement has no assignment for {d} threads", .{threads});
    if (arr_val != .array) return fail("CPU placement has no assignment for {d} threads", .{threads});
    const out = try allocator.alloc(i64, arr_val.array.items.len);
    for (arr_val.array.items, 0..) |v, idx| {
        out[idx] = valueAsI64(v) orelse return fail("invalid CPU id", .{});
    }
    return out;
}

pub fn directPairCpuAffinity(
    allocator: Allocator,
    cpu_placement: ObjectMap,
    global_index: usize,
    condition: []const u8,
    left: []const u8,
    right: []const u8,
) HarnessError![]i64 {
    const conditions = benchmark_schema.alternatingPairOrder(global_index, left, right);
    const cond_idx: usize = if (std.mem.eql(u8, conditions[0], condition))
        0
    else if (std.mem.eql(u8, conditions[1], condition))
        1
    else
        return fail("CPU placement has no direct-pair assignment", .{});

    const assignments_val = cpu_placement.get("assignments") orelse return fail("CPU placement has no direct-pair assignment", .{});
    if (assignments_val != .object) return fail("CPU placement has no direct-pair assignment", .{});
    const shp = assignments_val.object.get("single-hot-pair") orelse return fail("CPU placement has no direct-pair assignment", .{});
    if (shp != .object) return fail("CPU placement has no direct-pair assignment", .{});
    const lcpus = shp.object.get("logical_cpus") orelse return fail("CPU placement has no direct-pair assignment", .{});
    if (lcpus != .array or cond_idx >= lcpus.array.items.len) return fail("CPU placement has no direct-pair assignment", .{});
    const cpu = valueAsI64(lcpus.array.items[cond_idx]) orelse return fail("CPU placement has no direct-pair assignment", .{});
    const out = try allocator.alloc(i64, 1);
    out[0] = cpu;
    return out;
}

pub fn pairedConditionOrder(
    global_index: usize,
    pair_kind: []const u8,
    left: []const u8,
    right: []const u8,
) [2][]const u8 {
    const order_index = if (std.mem.eql(u8, pair_kind, "single-infrastructure"))
        global_index
    else
        global_index / 2;
    return benchmark_schema.alternatingPairOrder(order_index, left, right);
}

pub const InvocationPair = struct {
    revision: []const u8,
    condition: []const u8,
};

pub fn pairedInvocationOrder(
    allocator: Allocator,
    global_index: usize,
    pair_kind: []const u8,
    left: []const u8,
    right: []const u8,
    revision_roles: []const []const u8,
) ![]InvocationPair {
    const is_paired = revision_roles.len == 2 and
        std.mem.eql(u8, revision_roles[0], "baseline") and
        std.mem.eql(u8, revision_roles[1], "candidate");
    const revisions_pair = if (is_paired)
        benchmark_schema.alternatingPairOrder(global_index, "baseline", "candidate")
    else
        [2][]const u8{ "candidate", "" };
    const revisions: []const []const u8 = if (is_paired) revisions_pair[0..2] else revisions_pair[0..1];
    const conditions = pairedConditionOrder(global_index, pair_kind, left, right);

    var out: std.ArrayList(InvocationPair) = .empty;
    if (std.mem.eql(u8, pair_kind, "single-infrastructure")) {
        for (revisions) |rev| {
            for (conditions) |cond| {
                try out.append(allocator, .{ .revision = rev, .condition = cond });
            }
        }
    } else {
        for (conditions) |cond| {
            for (revisions) |rev| {
                try out.append(allocator, .{ .revision = rev, .condition = cond });
            }
        }
    }
    return out.toOwnedSlice(allocator);
}

pub fn positionBalancedRatioStats(
    allocator: Allocator,
    values: []const f64,
    sample_key: []const u8,
) HarnessError!ObjectMap {
    for (values, 0..) |v, idx| {
        if (!std.math.isFinite(v) or v <= 0.0) {
            return failData("{s}[{d}] must be a finite positive ratio", .{ sample_key, idx });
        }
    }
    const block_size: usize = 4;
    if (values.len == 0 or values.len % block_size != 0) {
        return failData("{s} must contain complete four-position blocks", .{sample_key});
    }
    const num_blocks = values.len / block_size;
    var block_estimates = try allocator.alloc(f64, num_blocks);
    var block_arr = Array.init(allocator);
    for (0..num_blocks) |b| {
        var log_sum: f64 = 0.0;
        for (values[b * block_size .. (b + 1) * block_size]) |v| {
            log_sum += @log(v);
        }
        const est = @exp(log_sum / @as(f64, @floatFromInt(block_size)));
        block_estimates[b] = est;
        try block_arr.append(.{ .float = est });
    }
    const stats = benchmark_stats.summarize(allocator, values) catch return failData("failed to compute stats", .{});
    const block_stats = benchmark_stats.summarize(allocator, block_estimates) catch return failData("failed to compute block stats", .{});

    var out: ObjectMap = .empty;
    var samp_arr = Array.init(allocator);
    for (values) |v| try samp_arr.append(.{ .float = v });
    try out.put(allocator, sample_key, .{ .array = samp_arr });
    try out.put(allocator, "runs", .{ .integer = @intCast(stats.runs) });
    try out.put(allocator, "mean", .{ .float = stats.mean });
    try out.put(allocator, "median", .{ .float = block_stats.median });
    try out.put(allocator, "min", .{ .float = stats.min });
    try out.put(allocator, "max", .{ .float = stats.max });
    try out.put(allocator, "range", .{ .float = stats.range });
    try out.put(allocator, "estimator", .{ .string = "median-of-four-position-geometric-means" });
    try out.put(allocator, "position_block_size", .{ .integer = @intCast(block_size) });
    try out.put(allocator, "position_block_estimates", .{ .array = block_arr });
    return out;
}

pub fn sampleStatsF64Value(allocator: Allocator, values: []const f64, sample_key: []const u8) !Value {
    const stats = benchmark_stats.summarize(allocator, values) catch return failData("failed to compute stats", .{});
    var out: ObjectMap = .empty;
    var samp_arr = Array.init(allocator);
    for (values) |v| try samp_arr.append(.{ .float = v });
    try out.put(allocator, sample_key, .{ .array = samp_arr });
    try out.put(allocator, "runs", .{ .integer = @intCast(stats.runs) });
    try out.put(allocator, "mean", .{ .float = stats.mean });
    try out.put(allocator, "median", .{ .float = stats.median });
    try out.put(allocator, "min", .{ .float = stats.min });
    try out.put(allocator, "max", .{ .float = stats.max });
    try out.put(allocator, "range", .{ .float = stats.range });
    return .{ .object = out };
}

pub fn sampleStatsI64Value(allocator: Allocator, values: []const i64, sample_key: []const u8) !Value {
    var floats = try allocator.alloc(f64, values.len);
    for (values, 0..) |v, idx| floats[idx] = @floatFromInt(v);
    const stats = benchmark_stats.summarize(allocator, floats) catch return failData("failed to compute stats", .{});
    var out: ObjectMap = .empty;
    var samp_arr = Array.init(allocator);
    for (values) |v| try samp_arr.append(.{ .integer = v });
    try out.put(allocator, sample_key, .{ .array = samp_arr });
    try out.put(allocator, "runs", .{ .integer = @intCast(stats.runs) });
    try out.put(allocator, "mean", .{ .float = stats.mean });
    if (@floor(stats.median) == stats.median) {
        try out.put(allocator, "median", .{ .integer = @intFromFloat(stats.median) });
    } else {
        try out.put(allocator, "median", .{ .float = stats.median });
    }
    try out.put(allocator, "min", .{ .integer = @intFromFloat(stats.min) });
    try out.put(allocator, "max", .{ .integer = @intFromFloat(stats.max) });
    try out.put(allocator, "range", .{ .integer = @intFromFloat(stats.range) });
    return .{ .object = out };
}
