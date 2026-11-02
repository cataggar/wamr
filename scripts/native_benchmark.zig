//! Native guest adapter for CoreMark and loop-pass benchmarks.
const std = @import("std");
const builtin = @import("builtin");
const benchmark_schema = @import("benchmark_schema.zig");
const benchmark_stats = @import("benchmark_stats.zig");
const process_control = @import("process_control.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const version: i64 = 2;
pub const phase_contract = "wamr-embedding-v2";
pub const unikraft_native_sdk_commit = "bb24e3e3f0ce5b998c0b24f6c06007d9a8ad7d20";
pub const post_wave12_unikraft_native_sdk_commit = "578b5619c61c93742e9bcba0af550c0fe675acb7";
pub const current_unikraft_native_sdk_commit = "370e8e83ac8298b6faa2b288714ffcc9b8798c4b";
pub const historical_unikraft_native_sdk_commit = "a53205d77be3b880eb8f8b96679512ba58e2331a";
pub const supported_unikraft_native_sdk_commits = [_][]const u8{
    unikraft_native_sdk_commit,
    post_wave12_unikraft_native_sdk_commit,
    current_unikraft_native_sdk_commit,
    historical_unikraft_native_sdk_commit,
};
pub const unikraft_tiny_correctness_profile = "qcow2-derived-vhd";
pub const unikraft_coremark_correctness_profile = "coremark-qcow2-derived-vhd";
pub const unikraft_local_boot_modes = [_][]const u8{
    "raw-x2apic",
    "raw-legacy-apic",
    "qcow2-x2apic",
    "qcow2-legacy-apic",
    "vpc-x2apic",
    "vpc-legacy-apic",
};
pub const coremark_short_crc_iterations: i64 = 100;
pub const coremark_short_crc_arguments = [_][]const u8{ "coremark", "0", "0", "0", "100", "0" };
pub const coremark_guest_args = [_][]const u8{ "0", "0", "0", "400000", "0" };
pub const expected_iterations: i64 = 400000;
pub const default_fixture_sha256 = "f4b7591296ead10264e0f101f355bdf848865c31329325594e66fbabefec235b";
pub const prefix = "WAMR_BENCH_RESULT=";
pub const coremark_ticks_per_second: i64 = 1000;

pub const FixtureInfo = struct {
    name: []const u8,
    path: []const u8,
    sha256: []const u8,
};

pub const fixtures = [_]FixtureInfo{
    .{
        .name = "coremark",
        .path = "tests/benchmarks/coremark/coremark_wasi.wasm",
        .sha256 = default_fixture_sha256,
    },
    .{
        .name = "coremark-nofp",
        .path = "tests/benchmarks/coremark/coremark_wasi_nofp.wasm",
        .sha256 = "24c0cc1bd52b641cf9e8ae74d1be188cba38d74cdb7ac18378de47382aab9541",
    },
    .{
        .name = "compute",
        .path = "tests/benchmarks/loop-passes/unroll4.wasm",
        .sha256 = "6870b3373e4098117c82b6736d0ca7cbcc7d8d747fe87ca5b7d1ebf0e4d12890",
    },
    .{
        .name = "memory",
        .path = "tests/benchmarks/loop-passes/iv_store.wasm",
        .sha256 = "b1979dd330c14d5f898b8a7c8c313e58f6521ad6eb7db9a6e7d7e78788ac6e72",
    },
};

pub const required_safety = [_][]const u8{ "bounds_checks", "wx_enforced", "import_checks", "trap_isolation" };
pub const option_keys = [_][]const u8{
    "bounds_checks", "wx_enforced",  "import_checks", "trap_isolation",
    "optimize",      "stack_checks", "simd",          "threads",
    "memory64",
};
pub const platform_keys = [_][]const u8{ "arch", "cpu_model", "active_cpu_count", "azure_sku", "azure_region" };
pub const source_keys = [_][]const u8{ "commit", "tree_sha256", "tracked_diff_sha256" };
pub const phase_keys = [_][]const u8{
    "compile_ticks",         "load_ticks",             "instantiate_ticks",
    "lifecycle_setup_ticks", "first_invocation_ticks", "steady_state_ticks",
};
pub const outcomes = [_][]const u8{ "success", "trap", "timeout", "abort", "error" };
pub const warm_reset_scope = [_][]const u8{ "invocation-state", "stdout-capture" };
pub const snapshot_reset_scope = [_][]const u8{
    "globals",                         "invocation-output",
    "linear-memory-access-protection", "linear-memory-contents",
    "linear-memory-logical-size",      "passive-segment-drop-state",
    "table-entries-signatures",        "wasi-context",
};
pub const measurement_errors = [_][]const u8{ "pending-output", "post-call-clock", "stdout-copy" };

pub fn findFixture(name: []const u8) ?FixtureInfo {
    for (fixtures) |f| {
        if (std.mem.eql(u8, f.name, name)) return f;
    }
    return null;
}

pub fn profileCounts(profile: []const u8) ?struct { warmups: i64, runs: i64 } {
    if (std.mem.eql(u8, profile, "authoritative")) return .{ .warmups = 2, .runs = 10 };
    if (std.mem.eql(u8, profile, "ci")) return .{ .warmups = 0, .runs = 3 };
    return null;
}

pub fn require(ok: bool) !void {
    if (!ok) return error.ValidationFailed;
}

pub fn requireKeys(value: Value, expected: []const []const u8) !std.json.ObjectMap {
    if (value != .object) return error.ValidationFailed;
    if (value.object.count() != expected.len) return error.ValidationFailed;
    for (expected) |key| {
        if (!value.object.contains(key)) return error.ValidationFailed;
    }
    return value.object;
}

pub fn requireHeader(obj: std.json.ObjectMap, kind: []const u8) !void {
    const sv = obj.get("schema_version") orelse return error.ValidationFailed;
    if (sv != .integer or sv.integer != version) return error.ValidationFailed;
    const k = obj.get("kind") orelse return error.ValidationFailed;
    if (k != .string or !std.mem.eql(u8, k.string, kind)) return error.ValidationFailed;
}

pub fn encoded(allocator: Allocator, value: Value) ![]u8 {
    return benchmark_schema.canonicalJson(allocator, value);
}

pub fn sha256Bytes(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

pub fn sha256BytesAlloc(allocator: Allocator, bytes: []const u8) ![]u8 {
    const hex = sha256Bytes(bytes);
    return allocator.dupe(u8, &hex);
}

pub fn sha256FileAlloc(allocator: Allocator, io: std.Io, path: []const u8) ![]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024));
    defer allocator.free(bytes);
    return sha256BytesAlloc(allocator, bytes);
}

pub fn cacheKeyAlloc(allocator: Allocator, value: Value) ![]u8 {
    const key = try benchmark_schema.cacheKey(allocator, value);
    return allocator.dupe(u8, &key);
}

pub fn identical(allocator: Allocator, left: Value, right: Value) !bool {
    const l_bytes = try encoded(allocator, left);
    defer allocator.free(l_bytes);
    const r_bytes = try encoded(allocator, right);
    defer allocator.free(r_bytes);
    return std.mem.eql(u8, l_bytes, r_bytes);
}

pub fn requireDigest(value: Value, length: usize) ![]const u8 {
    if (value != .string) return error.ValidationFailed;
    return requireDigestSlice(value.string, length);
}

pub fn requireDigestSlice(text: []const u8, length: usize) ![]const u8 {
    if (text.len != length) return error.ValidationFailed;
    for (text) |ch| {
        if (!((ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f'))) return error.ValidationFailed;
    }
    return text;
}

pub fn requirePublicToken(value: Value) ![]const u8 {
    if (value != .string) return error.ValidationFailed;
    const s = value.string;
    if (s.len == 0 or s.len > 128) return error.ValidationFailed;
    if (!std.ascii.isAlphanumeric(s[0])) return error.ValidationFailed;
    for (s[1..]) |ch| {
        if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == ':' or ch == '+' or ch == '-')) {
            return error.ValidationFailed;
        }
    }
    return s;
}

pub fn requireAvailableToken(value: Value) ![]const u8 {
    const s = try requirePublicToken(value);
    if (std.ascii.eqlIgnoreCase(s, "unknown") or
        std.ascii.eqlIgnoreCase(s, "placeholder") or
        std.ascii.eqlIgnoreCase(s, "unspecified"))
    {
        return error.ValidationFailed;
    }
    return s;
}

pub fn requireIntMin(value: Value, minimum: i64) !i64 {
    switch (value) {
        .integer => |v| {
            if (v < minimum) return error.ValidationFailed;
            return v;
        },
        .number_string => |s| {
            const parsed = std.fmt.parseInt(u64, s, 10) catch return error.ValidationFailed;
            if (parsed < @as(u64, @intCast(@max(0, minimum)))) return error.ValidationFailed;
            return std.math.cast(i64, parsed) orelse std.math.maxInt(i64);
        },
        else => return error.ValidationFailed,
    }
}

pub fn requireNumberMin(value: Value, minimum: f64) !f64 {
    switch (value) {
        .integer => |v| {
            const f: f64 = @floatFromInt(v);
            if (f < minimum or !std.math.isFinite(f)) return error.ValidationFailed;
            return f;
        },
        .float => |v| {
            if (v < minimum or v > @as(f64, @floatFromInt(std.math.maxInt(u64))) or !std.math.isFinite(v)) {
                return error.ValidationFailed;
            }
            return v;
        },
        .number_string => |s| {
            if (std.mem.indexOfAny(u8, s, ".eE") == null) {
                const parsed = std.fmt.parseInt(u64, s, 10) catch return error.ValidationFailed;
                const f: f64 = @floatFromInt(parsed);
                if (f < minimum) return error.ValidationFailed;
                return f;
            }
            const parsed = std.fmt.parseFloat(f64, s) catch return error.ValidationFailed;
            if (parsed < minimum or parsed > @as(f64, @floatFromInt(std.math.maxInt(u64))) or !std.math.isFinite(parsed)) {
                return error.ValidationFailed;
            }
            return parsed;
        },
        else => return error.ValidationFailed,
    }
}

pub fn strictJson(allocator: Allocator, text: []const u8) !Value {
    if (!std.unicode.utf8ValidateSlice(text)) return error.ValidationFailed;
    const parsed = std.json.parseFromSliceLeaky(Value, allocator, text, .{
        .duplicate_field_behavior = .@"error",
        .allocate = .alloc_always,
    }) catch return error.ValidationFailed;
    try checkFiniteNumbers(parsed);
    return parsed;
}

fn checkFiniteNumbers(value: Value) !void {
    switch (value) {
        .float => |v| if (!std.math.isFinite(v)) return error.ValidationFailed,
        .array => |arr| for (arr.items) |item| try checkFiniteNumbers(item),
        .object => |obj| {
            var it = obj.iterator();
            while (it.next()) |entry| try checkFiniteNumbers(entry.value_ptr.*);
        },
        else => {},
    }
}

pub fn readJson(allocator: Allocator, io: std.Io, path: []const u8) !Value {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024));
    return strictJson(allocator, bytes);
}

fn daysFromCivil(y_in: i64, m_in: i64, d_in: i64) i64 {
    const y = if (m_in <= 2) y_in - 1 else y_in;
    const era = @divFloor(if (y >= 0) y else y - 399, 400);
    const yoe = y - era * 400;
    const mp = m_in + (if (m_in > 2) @as(i64, -3) else 9);
    const doy = @divFloor(153 * mp + 2, 5) + d_in - 1;
    const doe = yoe * 365 + @divFloor(yoe, 4) - @divFloor(yoe, 100) + doy;
    return era * 146097 + doe - 719468;
}

fn isLeapYear(year: i64) bool {
    return (@mod(year, 4) == 0 and @mod(year, 100) != 0) or (@mod(year, 400) == 0);
}

fn daysInMonth(year: i64, month: i64) i64 {
    return switch (month) {
        1, 3, 5, 7, 8, 10, 12 => 31,
        4, 6, 9, 11 => 30,
        2 => if (isLeapYear(year)) 29 else 28,
        else => 0,
    };
}

pub fn parseUtcTimestampNs(value: Value) !i128 {
    if (value != .string) return error.ValidationFailed;
    return parseUtcTimestampSliceNs(value.string);
}

pub fn parseUtcTimestampSliceNs(s: []const u8) !i128 {
    if (s.len < 20) return error.ValidationFailed;
    if (s[4] != '-' or s[7] != '-' or (s[10] != 'T' and s[10] != ' ')) return error.ValidationFailed;
    if (s[13] != ':' or s[16] != ':') return error.ValidationFailed;
    const year = std.fmt.parseInt(i64, s[0..4], 10) catch return error.ValidationFailed;
    const month = std.fmt.parseInt(i64, s[5..7], 10) catch return error.ValidationFailed;
    const day = std.fmt.parseInt(i64, s[8..10], 10) catch return error.ValidationFailed;
    const hour = std.fmt.parseInt(i64, s[11..13], 10) catch return error.ValidationFailed;
    const minute = std.fmt.parseInt(i64, s[14..16], 10) catch return error.ValidationFailed;
    const second = std.fmt.parseInt(i64, s[17..19], 10) catch return error.ValidationFailed;
    if (month < 1 or month > 12) return error.ValidationFailed;
    if (day < 1 or day > daysInMonth(year, month)) return error.ValidationFailed;
    if (hour < 0 or hour > 23 or minute < 0 or minute > 59 or second < 0 or second > 59) return error.ValidationFailed;

    var pos: usize = 19;
    var nanos: i128 = 0;
    if (pos < s.len and s[pos] == '.') {
        pos += 1;
        const frac_start = pos;
        while (pos < s.len and std.ascii.isDigit(s[pos])) : (pos += 1) {}
        const frac_len = pos - frac_start;
        if (frac_len == 0 or frac_len > 9) return error.ValidationFailed;
        var parsed_frac = std.fmt.parseInt(i128, s[frac_start..pos], 10) catch return error.ValidationFailed;
        var scale: usize = frac_len;
        while (scale < 9) : (scale += 1) parsed_frac *= 10;
        nanos = parsed_frac;
    }
    const tz = s[pos..];
    if (!std.mem.eql(u8, tz, "Z") and !std.mem.eql(u8, tz, "+00:00") and !std.mem.eql(u8, tz, "-00:00")) {
        return error.ValidationFailed;
    }
    const days: i128 = daysFromCivil(year, month, day);
    const total_secs: i128 = ((days * 24 + hour) * 60 + minute) * 60 + second;
    return total_secs * std.time.ns_per_s + nanos;
}

pub fn formatUtcTimestamp(allocator: Allocator, epoch_ns: i128) ![]u8 {
    const secs = @divFloor(epoch_ns, std.time.ns_per_s);
    const rem_ns: u64 = @intCast(@mod(epoch_ns, std.time.ns_per_s));
    const micros = rem_ns / 1000;
    const days = @divFloor(secs, 86400);
    const day_secs: u64 = @intCast(@mod(secs, 86400));
    const hour = day_secs / 3600;
    const minute = (day_secs % 3600) / 60;
    const second = day_secs % 60;

    const z = days + 719468;
    const era = @divFloor(if (z >= 0) z else z - 146096, 146097);
    const doe = z - era * 146097;
    const yoe = @divFloor(doe - @divFloor(doe, 1460) + @divFloor(doe, 36524) - @divFloor(doe, 146096), 365);
    const y = yoe + era * 400;
    const doy = doe - (365 * yoe + @divFloor(yoe, 4) - @divFloor(yoe, 100));
    const mp = @divFloor(5 * doy + 2, 153);
    const d = doy - @divFloor(153 * mp + 2, 5) + 1;
    const m = mp + (if (mp < 10) @as(i64, 3) else -9);
    const year: u32 = @intCast(y + (if (m <= 2) @as(i128, 1) else 0));

    if (micros == 0) {
        return std.fmt.allocPrint(
            allocator,
            "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}+00:00",
            .{ year, @as(u8, @intCast(m)), @as(u8, @intCast(d)), hour, minute, second },
        );
    }
    return std.fmt.allocPrint(
        allocator,
        "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}.{d:0>6}+00:00",
        .{ year, @as(u8, @intCast(m)), @as(u8, @intCast(d)), hour, minute, second, micros },
    );
}

pub fn setPrivateDirPermissions(io: std.Io, path: []const u8) !void {
    if (builtin.os.tag == .windows) return;
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    try dir.setPermissions(io, .fromMode(0o700));
}

pub fn privateWrite(io: std.Io, path: []const u8, data: []const u8) !void {
    if (std.fs.path.dirname(path)) |parent| {
        if (parent.len != 0) {
            try std.Io.Dir.cwd().createDirPath(io, parent);
            try setPrivateDirPermissions(io, parent);
        }
    }
    const file = try std.Io.Dir.cwd().createFile(io, path, .{
        .exclusive = true,
        .permissions = if (builtin.os.tag == .windows) .default_file else .fromMode(0o600),
    });
    defer file.close(io);
    try file.writeStreamingAll(io, data);
}

pub fn artifact(allocator: Allocator, io: std.Io, path: []const u8) !Value {
    const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch return error.ValidationFailed;
    if (stat.kind != .file or stat.size == 0) return error.ValidationFailed;
    const sha_hex = try sha256FileAlloc(allocator, io, path);
    var map: std.json.ObjectMap = .empty;
    try map.put(allocator, "sha256", .{ .string = sha_hex });
    try map.put(allocator, "bytes", .{ .integer = @intCast(stat.size) });
    return .{ .object = map };
}

pub fn validateArtifact(value: Value) !void {
    const obj = try requireKeys(value, &.{ "sha256", "bytes" });
    _ = try requireDigest(obj.get("sha256").?, 64);
    _ = try requireIntMin(obj.get("bytes").?, 1);
}

pub fn validateSource(source: Value) !void {
    const obj = try requireKeys(source, &source_keys);
    _ = try requireDigest(obj.get("commit").?, 40);
    _ = try requireDigest(obj.get("tree_sha256").?, 64);
    _ = try requireDigest(obj.get("tracked_diff_sha256").?, 64);
}

pub fn validateOptions(options: Value) !void {
    const obj = try requireKeys(options, &option_keys);
    const opt = obj.get("optimize").?;
    try require(opt == .string and (std.mem.eql(u8, opt.string, "ReleaseFast") or
        std.mem.eql(u8, opt.string, "ReleaseSafe") or
        std.mem.eql(u8, opt.string, "ReleaseSmall") or
        std.mem.eql(u8, opt.string, "Debug")));
    for (option_keys) |k| {
        if (std.mem.eql(u8, k, "optimize")) continue;
        try require(obj.get(k).? == .bool);
    }
    for (required_safety) |k| {
        try require(obj.get(k).?.bool);
    }
}

pub fn validateMemoryPolicy(policy: Value) !void {
    const obj = try requireKeys(policy, &.{ "allocator_by_phase", "page_policy" });
    const alloc_obj = try requireKeys(obj.get("allocator_by_phase").?, &.{ "load", "instantiate", "first", "steady" });
    const page_obj = try requireKeys(obj.get("page_policy").?, &.{ "reservation", "commitment", "release" });
    for ([_][]const u8{ "load", "instantiate", "first", "steady" }) |k| {
        _ = try requireAvailableToken(alloc_obj.get(k).?);
    }
    for ([_][]const u8{ "reservation", "commitment", "release" }) |k| {
        _ = try requireAvailableToken(page_obj.get(k).?);
    }
}

pub fn validateLifecycle(lifecycle: Value) !void {
    const obj = try requireKeys(lifecycle, &.{ "mode", "reset_policy", "reset_before", "reset_timing", "reset_scope" });
    const mode = obj.get("mode").?;
    try require(mode == .string and (std.mem.eql(u8, mode.string, "same-instance-warm") or
        std.mem.eql(u8, mode.string, "snapshot-replay")));
    const warm = std.mem.eql(u8, mode.string, "same-instance-warm");
    const policy = obj.get("reset_policy").?;
    try require(policy == .string and std.mem.eql(
        u8,
        policy.string,
        if (warm) "invocation-state-only" else "restore-post-start-snapshot",
    ));
    const before = obj.get("reset_before").?;
    try require(before == .string and std.mem.eql(u8, before.string, "each-steady-invocation"));
    const timing = obj.get("reset_timing").?;
    try require(timing == .string and std.mem.eql(u8, timing.string, "excluded-from-invocation"));
    const scope = obj.get("reset_scope").?;
    try require(scope == .array);
    const expected_scope: []const []const u8 = if (warm) &warm_reset_scope else &snapshot_reset_scope;
    try require(scope.array.items.len == expected_scope.len);
    for (scope.array.items, expected_scope) |item, exp| {
        try require(item == .string and std.mem.eql(u8, item.string, exp));
    }
}

pub fn validatePlatform(platform_val: Value) !void {
    const obj = try requireKeys(platform_val, &platform_keys);
    _ = try requireIntMin(obj.get("active_cpu_count").?, 1);
    for (platform_keys) |k| {
        if (std.mem.eql(u8, k, "active_cpu_count")) continue;
        _ = try requireAvailableToken(obj.get(k).?);
    }
    const arch = obj.get("arch").?.string;
    try require(std.mem.eql(u8, arch, "x86_64") or std.mem.eql(u8, arch, "aarch64"));
}

pub fn validateImageReceipt(allocator: Allocator, receipt: Value, target: Value, evidence_kind: []const u8) !void {
    const obj = try requireKeys(receipt, &.{
        "schema_version",          "kind",                "evidence_kind",   "os",
        "image",                   "runtime",             "source",          "target_abi",
        "platform",                "options",             "compiler",        "compile_profile",
        "configured_vm_ram_bytes", "compiler_embedded",   "runtime_linkage", "memory_policy",
        "aot_modules",             "execution_lifecycle",
    });
    try requireHeader(obj, "wamr-native-image-receipt");
    const ek = obj.get("evidence_kind").?;
    try require(ek == .string and std.mem.eql(u8, ek.string, evidence_kind));
    const target_obj = target.object;
    for ([_][]const u8{
        "os",                  "image",   "runtime",  "source",          "target_abi",
        "platform",            "options", "compiler", "compile_profile", "memory_policy",
        "execution_lifecycle",
    }) |key| {
        try require(try identical(allocator, obj.get(key).?, target_obj.get(key).?));
    }
    try require(obj.get("compiler_embedded").? == .bool and !obj.get("compiler_embedded").?.bool);
    const linkage = obj.get("runtime_linkage").?;
    try require(linkage == .string and std.mem.eql(u8, linkage.string, "static"));
    _ = try requireIntMin(obj.get("configured_vm_ram_bytes").?, 1);
    try require(try identical(allocator, obj.get("aot_modules").?, target_obj.get("aot_modules").?));
}

pub fn makeSchedule(allocator: Allocator, manifest: Value) !Value {
    const obj = manifest.object;
    const warmups = try requireIntMin(obj.get("warmups").?, 0);
    const runs = try requireIntMin(obj.get("runs").?, 1);
    const workloads_val = obj.get("workloads").?;
    try require(workloads_val == .object and workloads_val.object.count() > 0);
    const wl_count: i64 = @intCast(workloads_val.object.count());
    try require(2 * wl_count * (warmups + runs) <= 10000);

    const sorted_wls = try allocator.alloc([]const u8, workloads_val.object.count());
    for (sorted_wls, workloads_val.object.keys()) |*dst, k| dst.* = k;
    std.mem.sort([]const u8, sorted_wls, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);

    var schedule: std.json.Array = .init(allocator);
    for ([_]struct { phase: []const u8, count: i64 }{
        .{ .phase = "warmup", .count = warmups },
        .{ .phase = "measured", .count = runs },
    }) |p| {
        for (sorted_wls) |wl| {
            var rep: i64 = 0;
            while (rep < p.count) : (rep += 1) {
                const pair = if (@mod(rep, 2) == 0)
                    [_][]const u8{ "linux", "unikraft" }
                else
                    [_][]const u8{ "unikraft", "linux" };
                for (pair) |target_name| {
                    const pos: i64 = @intCast(schedule.items.len + 1);
                    var entry: std.json.ObjectMap = .empty;
                    try entry.put(allocator, "run_id", .{ .string = try std.fmt.allocPrint(allocator, "run-{d:0>4}", .{@as(u32, @intCast(pos))}) });
                    try entry.put(allocator, "target", .{ .string = target_name });
                    try entry.put(allocator, "workload", .{ .string = wl });
                    try entry.put(allocator, "phase", .{ .string = p.phase });
                    try entry.put(allocator, "attempt", .{ .integer = 1 });
                    try entry.put(allocator, "position", .{ .integer = pos });
                    try schedule.append(.{ .object = entry });
                }
            }
        }
    }
    return .{ .array = schedule };
}

fn validateUuidString(value: Value) !void {
    if (value != .string) return error.ValidationFailed;
    const s = value.string;
    if (s.len != 36) return error.ValidationFailed;
    for (s, 0..) |ch, i| {
        if (i == 8 or i == 13 or i == 18 or i == 23) {
            if (ch != '-') return error.ValidationFailed;
        } else {
            if (!((ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f'))) return error.ValidationFailed;
        }
    }
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (haystack.len < needle.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

pub fn validateManifest(allocator: Allocator, manifest: Value, allow_synthetic: bool) !void {
    const obj = try requireKeys(manifest, &.{
        "schema_version", "kind",               "evidence_kind", "campaign_id",
        "created_at",     "expires_at",         "profile",       "warmups",
        "runs",           "steady_invocations", "targets",       "workloads",
        "schedule",       "abi_compatibility",  "producer",
    });
    try requireHeader(obj, "wamr-native-benchmark-plan");
    const ek = obj.get("evidence_kind").?;
    try require(ek == .string and (std.mem.eql(u8, ek.string, "measurement") or
        (allow_synthetic and std.mem.eql(u8, ek.string, "synthetic"))));
    try validateUuidString(obj.get("campaign_id").?);
    const created_ns = try parseUtcTimestampNs(obj.get("created_at").?);
    const expires_ns = try parseUtcTimestampNs(obj.get("expires_at").?);
    try require(created_ns < expires_ns);
    try require(expires_ns - created_ns <= 7 * 24 * 3600 * @as(i128, std.time.ns_per_s));
    const prof = obj.get("profile").?;
    try require(prof == .string and profileCounts(prof.string) != null);
    const warmups = try requireIntMin(obj.get("warmups").?, 0);
    const runs = try requireIntMin(obj.get("runs").?, 1);
    const steady = try requireIntMin(obj.get("steady_invocations").?, 1);
    const prod = try requireKeys(obj.get("producer").?, &.{ "source", "script_sha256", "helper_sha256", "schema_sha256" });
    try validateSource(prod.get("source").?);
    for ([_][]const u8{ "script_sha256", "helper_sha256", "schema_sha256" }) |k| {
        _ = try requireDigest(prod.get(k).?, 64);
    }

    const workloads_val = obj.get("workloads").?;
    try require(workloads_val == .object and workloads_val.object.count() > 0);
    const wl_count: i64 = @intCast(workloads_val.object.count());
    try require(2 * wl_count * (warmups + runs) <= 10000 and steady <= 10000);
    var wl_it = workloads_val.object.iterator();
    while (wl_it.next()) |entry| {
        const name = entry.key_ptr.*;
        const fix = findFixture(name) orelse return error.ValidationFailed;
        const wl_obj = try requireKeys(entry.value_ptr.*, &.{ "path", "sha256", "export", "args" });
        try require(wl_obj.get("path").? == .string and std.mem.eql(u8, wl_obj.get("path").?.string, fix.path));
        try require(wl_obj.get("sha256").? == .string and std.mem.eql(u8, wl_obj.get("sha256").?.string, fix.sha256));
        try require(wl_obj.get("export").? == .string and std.mem.eql(u8, wl_obj.get("export").?.string, "_start"));
        const args = wl_obj.get("args").?;
        try require(args == .array);
        if (std.mem.startsWith(u8, name, "coremark")) {
            try require(args.array.items.len == coremark_guest_args.len);
            for (args.array.items, coremark_guest_args) |a, exp| {
                try require(a == .string and std.mem.eql(u8, a.string, exp));
            }
        } else {
            try require(args.array.items.len == 0);
        }
    }

    const targets_obj = try requireKeys(obj.get("targets").?, &.{ "linux", "unikraft" });
    for ([_][]const u8{ "linux", "unikraft" }) |os_name| {
        const target = targets_obj.get(os_name).?;
        const t_obj = try requireKeys(target, &.{
            "os",            "runtime",              "compiler",      "source",
            "target_abi",    "platform",             "options",       "image",
            "image_receipt", "image_receipt_sha256", "aot_modules",   "mode",
            "jit_preset",    "compile_profile",      "memory_policy", "execution_lifecycle",
        });
        try require(t_obj.get("os").? == .string and std.mem.eql(u8, t_obj.get("os").?.string, os_name));
        try require(t_obj.get("mode").? == .string and std.mem.eql(u8, t_obj.get("mode").?.string, "aot") and
            t_obj.get("jit_preset").? == .null);
        try validateSource(t_obj.get("source").?);
        try validatePlatform(t_obj.get("platform").?);
        if (std.mem.eql(u8, ek.string, "measurement")) {
            var p_it = t_obj.get("platform").?.object.iterator();
            while (p_it.next()) |pe| {
                if (pe.value_ptr.* == .string) {
                    try require(!containsIgnoreCase(pe.value_ptr.*.string, "synthetic"));
                }
            }
        }
        try validateOptions(t_obj.get("options").?);
        try validateMemoryPolicy(t_obj.get("memory_policy").?);
        try validateLifecycle(t_obj.get("execution_lifecycle").?);
        _ = try requirePublicToken(t_obj.get("target_abi").?);
        const cp = t_obj.get("compile_profile").?;
        try require(cp == .null or (cp == .string and std.mem.eql(u8, cp.string, "unikraft-x86_64")));
        if (std.mem.eql(u8, os_name, "unikraft")) {
            try require(cp == .string and std.mem.eql(u8, cp.string, "unikraft-x86_64"));
        }
        if (cp == .string and std.mem.eql(u8, cp.string, "unikraft-x86_64")) {
            try require(std.mem.eql(u8, t_obj.get("platform").?.object.get("arch").?.string, "x86_64"));
        }
        try validateArtifact(t_obj.get("image").?);
        try validateArtifact(t_obj.get("runtime").?);
        const comp_obj = try requireKeys(t_obj.get("compiler").?, &.{ "binary", "source", "version" });
        try validateArtifact(comp_obj.get("binary").?);
        try validateSource(comp_obj.get("source").?);
        _ = try requirePublicToken(comp_obj.get("version").?);
        try require(try identical(allocator, comp_obj.get("source").?, t_obj.get("source").?));
        const aot_mods = try requireKeys(t_obj.get("aot_modules").?, workloads_val.object.keys());
        var am_it = aot_mods.iterator();
        while (am_it.next()) |am| try validateArtifact(am.value_ptr.*);
        _ = try requireDigest(t_obj.get("image_receipt_sha256").?, 64);
        try validateImageReceipt(allocator, t_obj.get("image_receipt").?, target, ek.string);
    }

    const linux = targets_obj.get("linux").?.object;
    const unikraft = targets_obj.get("unikraft").?.object;
    for ([_][]const u8{ "source", "compiler", "options", "platform", "execution_lifecycle" }) |k| {
        try require(try identical(allocator, linux.get(k).?, unikraft.get(k).?));
    }
    if (std.mem.eql(u8, ek.string, "measurement")) {
        const commit = unikraft.get("source").?.object.get("commit").?.string;
        var ok_commit = false;
        for (supported_unikraft_native_sdk_commits) |sc| {
            if (std.mem.eql(u8, commit, sc)) ok_commit = true;
        }
        try require(ok_commit);
    }
    const compat = try requireKeys(obj.get("abi_compatibility").?, workloads_val.object.keys());
    var c_it = compat.iterator();
    while (c_it.next()) |ce| {
        const wl = ce.key_ptr.*;
        const same = try identical(
            allocator,
            linux.get("aot_modules").?.object.get(wl).?,
            unikraft.get("aot_modules").?.object.get(wl).?,
        );
        const decl = try requireKeys(ce.value_ptr.*, &.{ "strategy", "proof_sha256" });
        const strat = decl.get("strategy").?;
        try require(strat == .string and std.mem.eql(u8, strat.string, if (same) "identical" else "target-specific"));
        if (same) {
            _ = try requireDigest(decl.get("proof_sha256").?, 64);
            try require(try identical(allocator, linux.get("compile_profile").?, unikraft.get("compile_profile").?));
        } else {
            try require(decl.get("proof_sha256").? == .null);
            try require(!std.mem.eql(u8, linux.get("target_abi").?.string, unikraft.get("target_abi").?.string));
        }
    }
    const expected_sched = try makeSchedule(allocator, manifest);
    try require(try identical(allocator, obj.get("schedule").?, expected_sched));
}

pub fn runConfiguration(allocator: Allocator, manifest: Value, run_id: []const u8) !Value {
    const obj = manifest.object;
    const schedule = obj.get("schedule").?.array;
    var found: ?Value = null;
    for (schedule.items) |item| {
        if (std.mem.eql(u8, item.object.get("run_id").?.string, run_id)) {
            if (found != null) return error.ValidationFailed;
            found = item;
        }
    }
    const run = found orelse return error.ValidationFailed;
    const target_name = run.object.get("target").?.string;
    const workload_name = run.object.get("workload").?.string;
    const target = obj.get("targets").?.object.get(target_name).?;
    var cfg: std.json.ObjectMap = .empty;
    try cfg.put(allocator, "campaign_id", obj.get("campaign_id").?);
    try cfg.put(allocator, "run", run);
    try cfg.put(allocator, "target", target);
    try cfg.put(allocator, "workload", obj.get("workloads").?.object.get(workload_name).?);
    try cfg.put(allocator, "steady_invocations", obj.get("steady_invocations").?);
    try cfg.put(allocator, "execution_lifecycle", target.object.get("execution_lifecycle").?);
    try cfg.put(allocator, "phase_contract", .{ .string = phase_contract });
    return .{ .object = cfg };
}

pub fn createRequest(allocator: Allocator, manifest: Value, run_id: []const u8, allow_synthetic: bool) !Value {
    try validateManifest(allocator, manifest, allow_synthetic);
    const cfg = try runConfiguration(allocator, manifest, run_id);
    var doc: std.json.ObjectMap = .empty;
    try doc.put(allocator, "config", cfg);
    try doc.put(allocator, "config_sha256", .{ .string = try cacheKeyAlloc(allocator, cfg) });
    return .{ .object = doc };
}

fn generateUuidV4(io: std.Io, allocator: Allocator) ![]u8 {
    var bytes: [16]u8 = undefined;
    io.random(&bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = std.fmt.bytesToHex(bytes, .lower);
    return std.fmt.allocPrint(allocator, "{s}-{s}-{s}-{s}-{s}", .{
        hex[0..8], hex[8..12], hex[12..16], hex[16..20], hex[20..32],
    });
}

fn resolveRefSha(allocator: Allocator, io: std.Io, repo: []const u8, ref: []const u8) ![]u8 {
    const res = try std.process.run(allocator, io, .{
        .argv = &.{ "git", "rev-parse", ref },
        .cwd = .{ .path = repo },
    });
    if (res.term != .exited or res.term.exited != 0) return error.ValidationFailed;
    return try allocator.dupe(u8, std.mem.trim(u8, res.stdout, " \t\r\n"));
}

fn hashFirstExisting(allocator: Allocator, io: std.Io, repo: []const u8, candidates: []const []const u8) ![]u8 {
    for (candidates) |rel| {
        const full = try std.fs.path.join(allocator, &.{ repo, rel });
        if (sha256FileAlloc(allocator, io, full)) |digest| return digest else |_| {}
    }
    return error.ValidationFailed;
}

fn resolveGitRepoDir(io: std.Io, repo_arg: []const u8) []const u8 {
    if (std.Io.Dir.cwd().openDir(io, repo_arg, .{})) |dir_val| {
        var dir = dir_val;
        dir.close(io);
        return repo_arg;
    } else |_| {
        return ".";
    }
}

fn resolveSourceFromConfig(allocator: Allocator, io: std.Io, cfg: std.json.ObjectMap) !Value {
    if (cfg.get("source")) |source_val| {
        try validateSource(source_val);
        if (cfg.get("source_commit")) |sc| {
            try require(sc == .string and std.mem.eql(u8, sc.string, source_val.object.get("commit").?.string));
        }
        if (cfg.get("source_tree_sha256")) |st| {
            try require(st == .string and std.mem.eql(u8, st.string, source_val.object.get("tree_sha256").?.string));
        }
        if (cfg.get("source_tracked_diff_sha256")) |sd| {
            try require(sd == .string and std.mem.eql(u8, sd.string, source_val.object.get("tracked_diff_sha256").?.string));
        }
        return source_val;
    }

    const repo_dir: ?[]const u8 = if (cfg.get("source_repo")) |sr| blk: {
        try require(sr == .string and sr.string.len > 0);
        break :blk resolveGitRepoDir(io, sr.string);
    } else null;

    const commit_str: []const u8 = if (cfg.get("source_commit")) |sc|
        try requireDigest(sc, 40)
    else if (repo_dir) |rdir|
        try resolveRefSha(allocator, io, rdir, "HEAD")
    else
        return error.ValidationFailed;

    const tree_sha: []const u8 = if (cfg.get("source_tree_sha256")) |st|
        try requireDigest(st, 64)
    else if (repo_dir) |rdir| blk: {
        const res = try std.process.run(allocator, io, .{
            .argv = &.{ "git", "ls-tree", "-r", commit_str },
            .cwd = .{ .path = rdir },
        });
        if (res.term != .exited or res.term.exited != 0) return error.ValidationFailed;
        break :blk try sha256BytesAlloc(allocator, res.stdout);
    } else return error.ValidationFailed;

    const diff_sha: []const u8 = if (cfg.get("source_tracked_diff_sha256")) |sd|
        try requireDigest(sd, 64)
    else if (repo_dir) |rdir| blk: {
        const res = try std.process.run(allocator, io, .{
            .argv = &.{ "git", "diff", "--binary", commit_str },
            .cwd = .{ .path = rdir },
        });
        if (res.term != .exited or res.term.exited != 0) return error.ValidationFailed;
        break :blk try sha256BytesAlloc(allocator, res.stdout);
    } else return error.ValidationFailed;

    var source_map: std.json.ObjectMap = .empty;
    try source_map.put(allocator, "commit", .{ .string = commit_str });
    try source_map.put(allocator, "tree_sha256", .{ .string = tree_sha });
    try source_map.put(allocator, "tracked_diff_sha256", .{ .string = diff_sha });
    const source_val: Value = .{ .object = source_map };
    try validateSource(source_val);
    return source_val;
}

pub fn createAbiProof(
    allocator: Allocator,
    io: std.Io,
    config: Value,
    allow_synthetic: bool,
) !Value {
    try require(config == .object);
    const cfg = config.object;
    const allowed_keys = [_][]const u8{
        "aot_path",
        "aot_sha256",
        "source",
        "source_repo",
        "source_commit",
        "source_tree_sha256",
        "source_tracked_diff_sha256",
        "target_abis",
        "linux_target_abi",
        "unikraft_target_abi",
        "compatible",
        "evidence_kind",
    };
    var it = cfg.iterator();
    while (it.next()) |entry| {
        var known = false;
        for (allowed_keys) |ak| {
            if (std.mem.eql(u8, entry.key_ptr.*, ak)) {
                known = true;
                break;
            }
        }
        try require(known);
    }

    const ek_str: []const u8 = if (cfg.get("evidence_kind")) |ek_val| blk: {
        try require(ek_val == .string);
        if (allow_synthetic) {
            try require(std.mem.eql(u8, ek_val.string, "measurement") or std.mem.eql(u8, ek_val.string, "synthetic"));
        } else {
            try require(std.mem.eql(u8, ek_val.string, "measurement"));
        }
        break :blk ek_val.string;
    } else if (allow_synthetic) "synthetic" else "measurement";

    if (cfg.get("compatible")) |compat_val| {
        try require(compat_val == .bool and compat_val.bool);
    }

    const source_val = try resolveSourceFromConfig(allocator, io, cfg);

    var linux_abi_opt: ?[]const u8 = null;
    var unikraft_abi_opt: ?[]const u8 = null;
    if (cfg.get("target_abis")) |abis_val| {
        try require(abis_val == .array and abis_val.array.items.len == 2);
        linux_abi_opt = try requirePublicToken(abis_val.array.items[0]);
        unikraft_abi_opt = try requirePublicToken(abis_val.array.items[1]);
    }
    if (cfg.get("linux_target_abi")) |labi| {
        const s = try requirePublicToken(labi);
        if (linux_abi_opt) |existing| try require(std.mem.eql(u8, existing, s));
        linux_abi_opt = s;
    }
    if (cfg.get("unikraft_target_abi")) |uabi| {
        const s = try requirePublicToken(uabi);
        if (unikraft_abi_opt) |existing| try require(std.mem.eql(u8, existing, s));
        unikraft_abi_opt = s;
    }
    const linux_abi = linux_abi_opt orelse return error.ValidationFailed;
    const unikraft_abi = unikraft_abi_opt orelse return error.ValidationFailed;

    var aot_sha_opt: ?[]const u8 = null;
    if (cfg.get("aot_path")) |aot_path_val| {
        try require(aot_path_val == .string and aot_path_val.string.len > 0);
        const aot_art = try artifact(allocator, io, aot_path_val.string);
        aot_sha_opt = aot_art.object.get("sha256").?.string;
    }
    if (cfg.get("aot_sha256")) |aot_sha_val| {
        const expected = try requireDigest(aot_sha_val, 64);
        if (aot_sha_opt) |actual| try require(std.mem.eql(u8, actual, expected));
        aot_sha_opt = expected;
    }
    const aot_sha = aot_sha_opt orelse return error.ValidationFailed;

    var abis: std.json.Array = .init(allocator);
    try abis.append(.{ .string = linux_abi });
    try abis.append(.{ .string = unikraft_abi });

    var proof: std.json.ObjectMap = .empty;
    try proof.put(allocator, "schema_version", .{ .integer = version });
    try proof.put(allocator, "kind", .{ .string = "wamr-aot-abi-proof" });
    try proof.put(allocator, "evidence_kind", .{ .string = ek_str });
    try proof.put(allocator, "compatible", .{ .bool = true });
    try proof.put(allocator, "source", source_val);
    try proof.put(allocator, "target_abis", .{ .array = abis });
    try proof.put(allocator, "aot_sha256", .{ .string = aot_sha });
    return .{ .object = proof };
}

pub fn createImageReceipt(
    allocator: Allocator,
    io: std.Io,
    config: Value,
    allow_synthetic: bool,
) !Value {
    try require(config == .object);
    const cfg = config.object;
    const required_keys = [_][]const u8{
        "image_path",
        "runtime_path",
        "compiler_path",
        "compiler_version",
        "source",
        "target_abi",
        "platform",
        "options",
        "compile_profile",
        "memory_policy",
        "execution_lifecycle",
        "configured_vm_ram_bytes",
        "aot_paths",
    };
    const optional_keys = [_][]const u8{
        "os",
        "compiler_embedded",
        "runtime_linkage",
        "evidence_kind",
    };
    for (required_keys) |rk| {
        try require(cfg.contains(rk));
    }
    var it = cfg.iterator();
    while (it.next()) |entry| {
        var known = false;
        for (required_keys) |rk| {
            if (std.mem.eql(u8, entry.key_ptr.*, rk)) {
                known = true;
                break;
            }
        }
        if (!known) {
            for (optional_keys) |ok| {
                if (std.mem.eql(u8, entry.key_ptr.*, ok)) {
                    known = true;
                    break;
                }
            }
        }
        try require(known);
    }

    const os_str: []const u8 = if (cfg.get("os")) |os_val| blk: {
        try require(os_val == .string and (std.mem.eql(u8, os_val.string, "linux") or std.mem.eql(u8, os_val.string, "unikraft")));
        break :blk os_val.string;
    } else "linux";

    const ek_str: []const u8 = if (cfg.get("evidence_kind")) |ek_val| blk: {
        try require(ek_val == .string);
        if (allow_synthetic) {
            try require(std.mem.eql(u8, ek_val.string, "measurement") or std.mem.eql(u8, ek_val.string, "synthetic"));
        } else {
            try require(std.mem.eql(u8, ek_val.string, "measurement"));
        }
        break :blk ek_val.string;
    } else if (allow_synthetic) "synthetic" else "measurement";

    const compiler_embedded: bool = if (cfg.get("compiler_embedded")) |ce_val| blk: {
        try require(ce_val == .bool and !ce_val.bool);
        break :blk ce_val.bool;
    } else false;

    const runtime_linkage: []const u8 = if (cfg.get("runtime_linkage")) |rl_val| blk: {
        try require(rl_val == .string and std.mem.eql(u8, rl_val.string, "static"));
        break :blk rl_val.string;
    } else "static";

    const source_val = cfg.get("source").?;
    try validateSource(source_val);
    if (std.mem.eql(u8, ek_str, "measurement") and std.mem.eql(u8, os_str, "unikraft")) {
        const commit = source_val.object.get("commit").?.string;
        var ok_commit = false;
        for (supported_unikraft_native_sdk_commits) |sc| {
            if (std.mem.eql(u8, commit, sc)) ok_commit = true;
        }
        try require(ok_commit);
    }

    const platform_val = cfg.get("platform").?;
    try validatePlatform(platform_val);
    if (std.mem.eql(u8, ek_str, "measurement")) {
        var p_it = platform_val.object.iterator();
        while (p_it.next()) |pe| {
            if (pe.value_ptr.* == .string) {
                try require(!containsIgnoreCase(pe.value_ptr.*.string, "synthetic"));
            }
        }
    }

    const options_val = cfg.get("options").?;
    try validateOptions(options_val);
    const memory_policy_val = cfg.get("memory_policy").?;
    try validateMemoryPolicy(memory_policy_val);
    const lifecycle_val = cfg.get("execution_lifecycle").?;
    try validateLifecycle(lifecycle_val);
    const target_abi_val = cfg.get("target_abi").?;
    _ = try requirePublicToken(target_abi_val);
    const compiler_version_val = cfg.get("compiler_version").?;
    _ = try requirePublicToken(compiler_version_val);

    const cp_val = cfg.get("compile_profile").?;
    try require(cp_val == .null or (cp_val == .string and std.mem.eql(u8, cp_val.string, "unikraft-x86_64")));
    if (std.mem.eql(u8, os_str, "unikraft")) {
        try require(cp_val == .string and std.mem.eql(u8, cp_val.string, "unikraft-x86_64"));
    }
    if (cp_val == .string and std.mem.eql(u8, cp_val.string, "unikraft-x86_64")) {
        try require(std.mem.eql(u8, platform_val.object.get("arch").?.string, "x86_64"));
    }

    const ram_val = cfg.get("configured_vm_ram_bytes").?;
    _ = try requireIntMin(ram_val, 1);

    const image_path_val = cfg.get("image_path").?;
    const runtime_path_val = cfg.get("runtime_path").?;
    const compiler_path_val = cfg.get("compiler_path").?;
    try require(image_path_val == .string and runtime_path_val == .string and compiler_path_val == .string);

    const image_art = try artifact(allocator, io, image_path_val.string);
    const runtime_art = try artifact(allocator, io, runtime_path_val.string);
    const compiler_bin_art = try artifact(allocator, io, compiler_path_val.string);

    const aot_paths_val = cfg.get("aot_paths").?;
    try require(aot_paths_val == .object and aot_paths_val.object.count() > 0);
    var aot_modules: std.json.ObjectMap = .empty;
    var aot_it = aot_paths_val.object.iterator();
    while (aot_it.next()) |entry| {
        const wname = entry.key_ptr.*;
        try require(findFixture(wname) != null);
        try require(entry.value_ptr.* == .string);
        try aot_modules.put(allocator, wname, try artifact(allocator, io, entry.value_ptr.*.string));
    }

    var comp_map: std.json.ObjectMap = .empty;
    try comp_map.put(allocator, "binary", compiler_bin_art);
    try comp_map.put(allocator, "source", source_val);
    try comp_map.put(allocator, "version", compiler_version_val);
    const compiler_val: Value = .{ .object = comp_map };

    var receipt: std.json.ObjectMap = .empty;
    try receipt.put(allocator, "schema_version", .{ .integer = version });
    try receipt.put(allocator, "kind", .{ .string = "wamr-native-image-receipt" });
    try receipt.put(allocator, "evidence_kind", .{ .string = ek_str });
    try receipt.put(allocator, "os", .{ .string = os_str });
    try receipt.put(allocator, "image", image_art);
    try receipt.put(allocator, "runtime", runtime_art);
    try receipt.put(allocator, "source", source_val);
    try receipt.put(allocator, "target_abi", target_abi_val);
    try receipt.put(allocator, "platform", platform_val);
    try receipt.put(allocator, "options", options_val);
    try receipt.put(allocator, "compiler", compiler_val);
    try receipt.put(allocator, "compile_profile", cp_val);
    try receipt.put(allocator, "configured_vm_ram_bytes", ram_val);
    try receipt.put(allocator, "compiler_embedded", .{ .bool = compiler_embedded });
    try receipt.put(allocator, "runtime_linkage", .{ .string = runtime_linkage });
    try receipt.put(allocator, "memory_policy", memory_policy_val);
    try receipt.put(allocator, "aot_modules", .{ .object = aot_modules });
    try receipt.put(allocator, "execution_lifecycle", lifecycle_val);
    const receipt_val: Value = .{ .object = receipt };

    var target_map: std.json.ObjectMap = .empty;
    try target_map.put(allocator, "os", .{ .string = os_str });
    try target_map.put(allocator, "image", image_art);
    try target_map.put(allocator, "runtime", runtime_art);
    try target_map.put(allocator, "source", source_val);
    try target_map.put(allocator, "target_abi", target_abi_val);
    try target_map.put(allocator, "platform", platform_val);
    try target_map.put(allocator, "options", options_val);
    try target_map.put(allocator, "compiler", compiler_val);
    try target_map.put(allocator, "compile_profile", cp_val);
    try target_map.put(allocator, "memory_policy", memory_policy_val);
    try target_map.put(allocator, "execution_lifecycle", lifecycle_val);
    try target_map.put(allocator, "aot_modules", .{ .object = aot_modules });

    try validateImageReceipt(allocator, receipt_val, .{ .object = target_map }, ek_str);
    return receipt_val;
}

pub fn createPlan(
    allocator: Allocator,
    io: std.Io,
    config: Value,
    repo: []const u8,
    now_ns_opt: ?i128,
    allow_synthetic: bool,
) !Value {
    const cfg = try requireKeys(config, &.{
        "profile", "warmups",   "runs",       "steady_invocations", "valid_hours",
        "targets", "workloads", "abi_proofs", "producer_source",
    });
    const prof = cfg.get("profile").?;
    try require(prof == .string and profileCounts(prof.string) != null);
    const valid_hours = try requireNumberMin(cfg.get("valid_hours").?, 0.01);
    try require(valid_hours <= 168.0);
    try validateSource(cfg.get("producer_source").?);
    if (!allow_synthetic) {
        const head_sha = try resolveRefSha(allocator, io, repo, "HEAD");
        try require(std.mem.eql(u8, cfg.get("producer_source").?.object.get("commit").?.string, head_sha));
    }
    const now_ns: i128 = now_ns_opt orelse @as(i128, @intCast(std.Io.Clock.real.now(io).nanoseconds));
    const delta_ns: i128 = @intFromFloat(valid_hours * 3600.0 * @as(f64, @floatFromInt(std.time.ns_per_s)));

    var producer_map: std.json.ObjectMap = .empty;
    try producer_map.put(allocator, "source", cfg.get("producer_source").?);
    try producer_map.put(allocator, "script_sha256", .{
        .string = try hashFirstExisting(allocator, io, repo, &.{ "scripts/bench_coremark.zig", "scripts/native_benchmark.zig" }),
    });
    try producer_map.put(allocator, "helper_sha256", .{
        .string = try hashFirstExisting(allocator, io, repo, &.{"scripts/native_benchmark.zig"}),
    });
    try producer_map.put(allocator, "schema_sha256", .{
        .string = try hashFirstExisting(allocator, io, repo, &.{"scripts/benchmark_schema.zig"}),
    });

    const wl_arr = cfg.get("workloads").?;
    try require(wl_arr == .array and wl_arr.array.items.len > 0);
    const wl_names = try allocator.alloc([]const u8, wl_arr.array.items.len);
    for (wl_arr.array.items, 0..) |item, i| {
        try require(item == .string);
        for (wl_names[0..i]) |prev| try require(!std.mem.eql(u8, prev, item.string));
        wl_names[i] = item.string;
    }
    std.mem.sort([]const u8, wl_names, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);

    var workloads_map: std.json.ObjectMap = .empty;
    for (wl_names) |name| {
        const fix = findFixture(name) orelse return error.ValidationFailed;
        const full_fix = try std.fs.path.join(allocator, &.{ repo, fix.path });
        const actual_sha = try sha256FileAlloc(allocator, io, full_fix);
        try require(std.mem.eql(u8, actual_sha, fix.sha256));
        var args_json: std.json.Array = .init(allocator);
        if (std.mem.startsWith(u8, name, "coremark")) {
            for (coremark_guest_args) |a| try args_json.append(.{ .string = a });
        }
        var wl_obj: std.json.ObjectMap = .empty;
        try wl_obj.put(allocator, "path", .{ .string = fix.path });
        try wl_obj.put(allocator, "sha256", .{ .string = fix.sha256 });
        try wl_obj.put(allocator, "export", .{ .string = "_start" });
        try wl_obj.put(allocator, "args", .{ .array = args_json });
        try workloads_map.put(allocator, name, .{ .object = wl_obj });
    }

    const cfg_targets = try requireKeys(cfg.get("targets").?, &.{ "linux", "unikraft" });
    var targets_map: std.json.ObjectMap = .empty;
    for ([_][]const u8{ "linux", "unikraft" }) |os_name| {
        const spec = try requireKeys(cfg_targets.get(os_name).?, &.{
            "runtime_path",        "compiler_path", "compiler_version", "source",
            "target_abi",          "platform",      "options",          "image_path",
            "image_receipt_path",  "aot_paths",     "compile_profile",  "memory_policy",
            "execution_lifecycle",
        });
        const aot_paths = try requireKeys(spec.get("aot_paths").?, wl_names);
        const receipt_path = spec.get("image_receipt_path").?.string;
        var aot_modules: std.json.ObjectMap = .empty;
        for (wl_names) |wname| {
            try aot_modules.put(allocator, wname, try artifact(allocator, io, aot_paths.get(wname).?.string));
        }
        var comp_map: std.json.ObjectMap = .empty;
        try comp_map.put(allocator, "binary", try artifact(allocator, io, spec.get("compiler_path").?.string));
        try comp_map.put(allocator, "source", spec.get("source").?);
        try comp_map.put(allocator, "version", spec.get("compiler_version").?);

        var t_map: std.json.ObjectMap = .empty;
        try t_map.put(allocator, "os", .{ .string = os_name });
        try t_map.put(allocator, "mode", .{ .string = "aot" });
        try t_map.put(allocator, "jit_preset", .null);
        try t_map.put(allocator, "runtime", try artifact(allocator, io, spec.get("runtime_path").?.string));
        try t_map.put(allocator, "source", spec.get("source").?);
        try t_map.put(allocator, "compiler", .{ .object = comp_map });
        try t_map.put(allocator, "compile_profile", spec.get("compile_profile").?);
        try t_map.put(allocator, "memory_policy", spec.get("memory_policy").?);
        try t_map.put(allocator, "execution_lifecycle", spec.get("execution_lifecycle").?);
        try t_map.put(allocator, "target_abi", spec.get("target_abi").?);
        try t_map.put(allocator, "platform", spec.get("platform").?);
        try t_map.put(allocator, "options", spec.get("options").?);
        try t_map.put(allocator, "image", try artifact(allocator, io, spec.get("image_path").?.string));
        try t_map.put(allocator, "image_receipt", try readJson(allocator, io, receipt_path));
        try t_map.put(allocator, "image_receipt_sha256", .{ .string = try sha256FileAlloc(allocator, io, receipt_path) });
        try t_map.put(allocator, "aot_modules", .{ .object = aot_modules });
        try targets_map.put(allocator, os_name, .{ .object = t_map });
    }

    const abi_proofs = try requireKeys(cfg.get("abi_proofs").?, wl_names);
    var compat_map: std.json.ObjectMap = .empty;
    const linux_t = targets_map.get("linux").?.object;
    const unikraft_t = targets_map.get("unikraft").?.object;
    const ek_str = if (allow_synthetic) "synthetic" else "measurement";
    for (wl_names) |wname| {
        const proof_val = abi_proofs.get(wname).?;
        const same = try identical(
            allocator,
            linux_t.get("aot_modules").?.object.get(wname).?,
            unikraft_t.get("aot_modules").?.object.get(wname).?,
        );
        if (same) {
            try require(proof_val == .string);
            const proof = try readJson(allocator, io, proof_val.string);
            var abis: std.json.Array = .init(allocator);
            try abis.append(linux_t.get("target_abi").?);
            try abis.append(unikraft_t.get("target_abi").?);
            var exp_proof: std.json.ObjectMap = .empty;
            try exp_proof.put(allocator, "schema_version", .{ .integer = version });
            try exp_proof.put(allocator, "kind", .{ .string = "wamr-aot-abi-proof" });
            try exp_proof.put(allocator, "evidence_kind", .{ .string = ek_str });
            try exp_proof.put(allocator, "compatible", .{ .bool = true });
            try exp_proof.put(allocator, "source", linux_t.get("source").?);
            try exp_proof.put(allocator, "target_abis", .{ .array = abis });
            try exp_proof.put(allocator, "aot_sha256", linux_t.get("aot_modules").?.object.get(wname).?.object.get("sha256").?);
            try require(try identical(allocator, proof, .{ .object = exp_proof }));
        } else {
            try require(proof_val == .null);
        }
        var decl: std.json.ObjectMap = .empty;
        try decl.put(allocator, "strategy", .{ .string = if (same) "identical" else "target-specific" });
        try decl.put(allocator, "proof_sha256", if (same) .{ .string = try sha256FileAlloc(allocator, io, proof_val.string) } else .null);
        try compat_map.put(allocator, wname, .{ .object = decl });
    }

    var manifest_map: std.json.ObjectMap = .empty;
    try manifest_map.put(allocator, "schema_version", .{ .integer = version });
    try manifest_map.put(allocator, "kind", .{ .string = "wamr-native-benchmark-plan" });
    try manifest_map.put(allocator, "evidence_kind", .{ .string = ek_str });
    try manifest_map.put(allocator, "campaign_id", .{ .string = try generateUuidV4(io, allocator) });
    try manifest_map.put(allocator, "created_at", .{ .string = try formatUtcTimestamp(allocator, now_ns) });
    try manifest_map.put(allocator, "expires_at", .{ .string = try formatUtcTimestamp(allocator, now_ns + delta_ns) });
    try manifest_map.put(allocator, "profile", prof);
    try manifest_map.put(allocator, "warmups", cfg.get("warmups").?);
    try manifest_map.put(allocator, "runs", cfg.get("runs").?);
    try manifest_map.put(allocator, "steady_invocations", cfg.get("steady_invocations").?);
    try manifest_map.put(allocator, "targets", .{ .object = targets_map });
    try manifest_map.put(allocator, "workloads", .{ .object = workloads_map });
    try manifest_map.put(allocator, "abi_compatibility", .{ .object = compat_map });
    try manifest_map.put(allocator, "producer", .{ .object = producer_map });
    const sched = try makeSchedule(allocator, .{ .object = manifest_map });
    try manifest_map.put(allocator, "schedule", sched);

    const manifest_val: Value = .{ .object = manifest_map };
    try validateManifest(allocator, manifest_val, allow_synthetic);
    return manifest_val;
}

pub fn parseResultStream(allocator: Allocator, raw: []const u8) !Value {
    var results: std.ArrayList(Value) = .empty;
    var i: usize = 0;
    while (i < raw.len) {
        var end = i;
        while (end < raw.len and raw[end] != '\n' and raw[end] != '\r') : (end += 1) {}
        const line = raw[i..end];
        if (end < raw.len and raw[end] == '\r' and end + 1 < raw.len and raw[end + 1] == '\n') {
            i = end + 2;
        } else {
            i = @min(raw.len, end + 1);
        }
        if (std.mem.indexOf(u8, line, prefix) != null) {
            try require(std.mem.startsWith(u8, line, prefix));
            try results.append(allocator, try strictJson(allocator, line[prefix.len..]));
        }
    }
    try require(results.items.len == 1);
    return results.items[0];
}

fn containsString(items: []const Value, target: []const u8) bool {
    for (items) |item| {
        if (item == .string and std.mem.eql(u8, item.string, target)) return true;
    }
    return false;
}

pub fn validateMemory(allocator: Allocator, memory: Value, target: Value, complete: bool) !void {
    const obj = try requireKeys(memory, &.{
        "image_sha256", "configured_vm_ram_bytes", "coverage",
        "method",       "covered_regions",         "omitted_regions",
        "samples",      "policy",
    });
    const t_obj = target.object;
    const img_sha = obj.get("image_sha256").?;
    try require(img_sha == .string and std.mem.eql(u8, img_sha.string, t_obj.get("image").?.object.get("sha256").?.string));
    const ram = try requireIntMin(obj.get("configured_vm_ram_bytes").?, 1);
    try require(ram == t_obj.get("image_receipt").?.object.get("configured_vm_ram_bytes").?.integer);
    try require(try identical(allocator, obj.get("policy").?, t_obj.get("memory_policy").?));
    const cov = obj.get("coverage").?;
    try require(cov == .string and (std.mem.eql(u8, cov.string, "complete-guest") or
        std.mem.eql(u8, cov.string, "partial-guest")));
    _ = try requirePublicToken(obj.get("method").?);

    for ([_][]const u8{ "covered_regions", "omitted_regions" }) |rname| {
        const arr = obj.get(rname).?;
        try require(arr == .array);
        for (arr.array.items, 0..) |item, idx| {
            const s = try requirePublicToken(item);
            try require(!containsString(arr.array.items[0..idx], s));
        }
    }
    const covered = obj.get("covered_regions").?.array.items;
    const omitted = obj.get("omitted_regions").?.array.items;
    try require(covered.len > 0);
    for (covered) |c| try require(!containsString(omitted, c.string));
    try require(std.mem.eql(u8, cov.string, "complete-guest") == (omitted.len == 0));

    const samples = obj.get("samples").?;
    try require(samples == .array);
    if (complete) {
        try require(samples.array.items.len == 3);
    } else {
        try require(samples.array.items.len >= 1 and samples.array.items.len <= 3);
    }
    const stages = [_][]const u8{ "after_instantiation", "after_first", "after_steady" };
    for (samples.array.items, 0..) |sample, idx| {
        const s_obj = try requireKeys(sample, &.{ "stage", "reserved_address_bytes", "committed_bytes" });
        try require(s_obj.get("stage").? == .string and std.mem.eql(u8, s_obj.get("stage").?.string, stages[idx]));
        const res_b = try requireIntMin(s_obj.get("reserved_address_bytes").?, 1);
        const com_b = try requireIntMin(s_obj.get("committed_bytes").?, 1);
        try require(com_b <= res_b);
    }
}

pub fn decodeStdout(allocator: Allocator, encoded_val: Value) ![]u8 {
    if (encoded_val != .string) return error.ValidationFailed;
    const s = encoded_val.string;
    const dec_len = std.base64.standard.Decoder.calcSizeForSlice(s) catch return error.ValidationFailed;
    const raw = try allocator.alloc(u8, dec_len);
    std.base64.standard.Decoder.decode(raw, s) catch return error.ValidationFailed;
    const enc_len = std.base64.standard.Encoder.calcSize(raw.len);
    const reencoded = try allocator.alloc(u8, enc_len);
    _ = std.base64.standard.Encoder.encode(reencoded, raw);
    try require(std.mem.eql(u8, reencoded, s));
    return raw;
}

fn isDigitsOnly(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s) |ch| if (!std.ascii.isDigit(ch)) return false;
    return true;
}

fn isFixed6Dp(s: []const u8) bool {
    const dot = std.mem.indexOfScalar(u8, s, '.') orelse return false;
    if (dot == 0 or s.len - dot - 1 != 6) return false;
    return isDigitsOnly(s[0..dot]) and isDigitsOnly(s[dot + 1 ..]);
}

pub fn validateCoremarkShortCrcOutput(allocator: Allocator, stdout: []const u8, workload: []const u8) !Value {
    try require(std.mem.eql(u8, workload, "coremark") or std.mem.eql(u8, workload, "coremark-nofp"));
    try require(stdout.len > 0 and stdout[stdout.len - 1] == '\n');

    const req_fields = [_]struct { k: []const u8, v: []const u8 }{
        .{ .k = "Iterations", .v = "100" },
        .{ .k = "seedcrc", .v = "0xe9f5" },
        .{ .k = "[0]crclist", .v = "0xe714" },
        .{ .k = "[0]crcmatrix", .v = "0x1fd7" },
        .{ .k = "[0]crcstate", .v = "0x8e3a" },
        .{ .k = "[0]crcfinal", .v = "0x988c" },
    };
    const meta_fields = [_][]const u8{
        "CoreMark Size",    "Total ticks",    "Total time (secs)", "Iterations/Sec",
        "Compiler version", "Compiler flags", "Memory location",
    };
    const markers = [_][]const u8{
        "2K performance run parameters for coremark.",
        "ERROR! Must execute for at least 10 secs for a valid result!",
        "Errors detected",
    };

    var seen_req: [req_fields.len]bool = @splat(false);
    var seen_meta: [meta_fields.len]?[]const u8 = @splat(null);
    var seen_markers: [markers.len]bool = @splat(false);

    var line_it = std.mem.splitScalar(u8, stdout[0 .. stdout.len - 1], '\n');
    while (line_it.next()) |raw| {
        for (raw) |ch| {
            try require((ch >= 32 and ch <= 126) or ch == '\t' or ch == '\r');
        }
        const without_cr = if (std.mem.endsWith(u8, raw, "\r")) raw[0 .. raw.len - 1] else raw;
        const line = std.mem.trim(u8, without_cr, " \t");
        if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
            const key = std.mem.trim(u8, line[0..colon], " \t");
            const val = std.mem.trim(u8, line[colon + 1 ..], " \t");
            var matched = false;
            for (req_fields, 0..) |rf, idx| {
                if (std.mem.eql(u8, key, rf.k)) {
                    try require(!seen_req[idx] and std.mem.eql(u8, val, rf.v));
                    seen_req[idx] = true;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                for (meta_fields, 0..) |mf, idx| {
                    if (std.mem.eql(u8, key, mf)) {
                        try require(seen_meta[idx] == null);
                        seen_meta[idx] = val;
                        matched = true;
                        break;
                    }
                }
            }
            try require(matched);
        } else {
            var matched = std.mem.eql(u8, line, "(e.g. code in flash, data on heap etc)");
            for (markers, 0..) |m, idx| {
                if (std.mem.eql(u8, line, m)) {
                    try require(!seen_markers[idx]);
                    seen_markers[idx] = true;
                    matched = true;
                    break;
                }
            }
            try require(matched);
        }
    }
    for (seen_req) |s| try require(s);
    for (seen_markers) |s| try require(s);

    const nofp = std.mem.eql(u8, workload, "coremark-nofp");
    if (seen_meta[2]) |total_time| {
        try require(if (nofp) isDigitsOnly(total_time) else isFixed6Dp(total_time));
    }
    if (seen_meta[3]) |iter_sec| {
        try require(if (nofp) isDigitsOnly(iter_sec) else isFixed6Dp(iter_sec));
    }
    if (seen_meta[1]) |total_ticks| {
        try require(isDigitsOnly(total_ticks));
    }

    var crc: std.json.ObjectMap = .empty;
    try crc.put(allocator, "seedcrc", .{ .string = "e9f5" });
    try crc.put(allocator, "crclist", .{ .string = "e714" });
    try crc.put(allocator, "crcmatrix", .{ .string = "1fd7" });
    try crc.put(allocator, "crcstate", .{ .string = "8e3a" });
    try crc.put(allocator, "crcfinal", .{ .string = "988c" });

    var res: std.json.ObjectMap = .empty;
    try res.put(allocator, "qualification", .{ .string = "correctness-only-not-performance" });
    try res.put(allocator, "profile", .{ .string = unikraft_coremark_correctness_profile });
    try res.put(allocator, "workload", .{ .string = workload });
    try res.put(allocator, "iterations", .{ .integer = coremark_short_crc_iterations });
    try res.put(allocator, "crc", .{ .object = crc });
    try res.put(allocator, "time_format", .{ .string = if (nofp) "integer-truncated" else "float-6dp" });
    try res.put(allocator, "minimum_timing_met", .{ .bool = false });
    try res.put(allocator, "compliance", .{ .string = "not-certified" });
    return .{ .object = res };
}

fn countSubstring(haystack: []const u8, needle: []const u8) usize {
    var count: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, haystack, pos, needle)) |idx| {
        count += 1;
        pos = idx + needle.len;
    }
    return count;
}

pub const ParsedCoreMark = struct {
    throughput: f64,
    iterations: i64,
};

pub fn parseCoremarkOutput(output: []const u8, exp_iters: i64) !ParsedCoreMark {
    try require(countSubstring(output, "Correct operation validated.") == 1);
    try require(std.mem.indexOf(u8, output, "ERROR!") == null);
    try require(countSubstring(output, "2K performance run parameters for coremark.") == 1);

    var throughput_val: ?f64 = null;
    var iterations_val: ?i64 = null;
    var line_it = std.mem.splitScalar(u8, output, '\n');
    while (line_it.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (std.mem.indexOf(u8, line, "Iterations/Sec") != null) {
            try require(throughput_val == null);
            try require(std.mem.startsWith(u8, line, "Iterations/Sec"));
            const rest = std.mem.trimStart(u8, line["Iterations/Sec".len..], " \t");
            try require(rest.len > 0 and rest[0] == ':');
            const num_str = std.mem.trim(u8, rest[1..], " \t");
            if (std.mem.indexOfScalar(u8, num_str, '.')) |dot| {
                try require(dot > 0 and isDigitsOnly(num_str[0..dot]) and isDigitsOnly(num_str[dot + 1 ..]));
            } else {
                try require(isDigitsOnly(num_str));
            }
            throughput_val = std.fmt.parseFloat(f64, num_str) catch return error.ValidationFailed;
        } else if (std.mem.startsWith(u8, line, "Iterations")) {
            const rest = std.mem.trimStart(u8, line["Iterations".len..], " \t");
            if (rest.len > 0 and rest[0] == ':') {
                try require(iterations_val == null);
                const num_str = std.mem.trim(u8, rest[1..], " \t");
                try require(isDigitsOnly(num_str));
                iterations_val = std.fmt.parseInt(i64, num_str, 10) catch return error.ValidationFailed;
            }
        }
    }
    const tp = throughput_val orelse return error.ValidationFailed;
    const iters = iterations_val orelse return error.ValidationFailed;
    try require(iters == exp_iters);
    return .{ .throughput = tp, .iterations = iters };
}

fn extractFieldOnce(stdout: []const u8, label: []const u8, expected_key: []const u8) ![]const u8 {
    var matched_line: ?[]const u8 = null;
    var line_it = std.mem.splitScalar(u8, stdout, '\n');
    while (line_it.next()) |raw_line| {
        const line = if (std.mem.endsWith(u8, raw_line, "\r")) raw_line[0 .. raw_line.len - 1] else raw_line;
        if (std.mem.indexOf(u8, line, label) != null) {
            try require(matched_line == null);
            matched_line = std.mem.trim(u8, line, " \t");
        }
    }
    const line = matched_line orelse return error.ValidationFailed;
    try require(std.mem.startsWith(u8, line, expected_key));
    const after_key = std.mem.trimStart(u8, line[expected_key.len..], " \t");
    try require(after_key.len > 0 and after_key[0] == ':');
    return std.mem.trim(u8, after_key[1..], " \t");
}

fn ulp(x: f64) f64 {
    const abs_x = @abs(x);
    if (!std.math.isFinite(abs_x)) return std.math.nan(f64);
    const bits: u64 = @bitCast(abs_x);
    const next: f64 = @bitCast(bits + 1);
    return next - abs_x;
}

pub fn coremarkCorrectness(
    allocator: Allocator,
    stdout: []const u8,
    invocation_seconds: ?f64,
    workload: []const u8,
    clock_resolution_seconds: f64,
) !Value {
    try require(std.mem.eql(u8, workload, "coremark") or std.mem.eql(u8, workload, "coremark-nofp"));
    if (invocation_seconds) |inv_s| {
        try require(inv_s >= 0.0 and std.math.isFinite(inv_s));
    }
    try require(clock_resolution_seconds >= 0.0 and clock_resolution_seconds <= 1.0 and std.math.isFinite(clock_resolution_seconds));
    const parsed = try parseCoremarkOutput(stdout, expected_iterations);

    var crc_map: std.json.ObjectMap = .empty;
    for ([_]struct { name: []const u8, expected: ?[]const u8 }{
        .{ .name = "seedcrc", .expected = "e9f5" },
        .{ .name = "crclist", .expected = "e714" },
        .{ .name = "crcmatrix", .expected = "1fd7" },
        .{ .name = "crcstate", .expected = "8e3a" },
        .{ .name = "crcfinal", .expected = null },
    }) |item| {
        const marker = if (std.mem.eql(u8, item.name, "seedcrc")) "seedcrc" else try std.fmt.allocPrint(allocator, "[0]{s}", .{item.name});
        const raw_val = try extractFieldOnce(stdout, item.name, marker);
        try require(raw_val.len == 6 and std.mem.startsWith(u8, raw_val, "0x"));
        const lower = try allocator.alloc(u8, 4);
        for (raw_val[2..], 0..) |ch, idx| {
            try require(std.ascii.isHex(ch));
            lower[idx] = std.ascii.toLower(ch);
        }
        if (item.expected) |exp| try require(std.mem.eql(u8, lower, exp));
        try crc_map.put(allocator, item.name, .{ .string = lower });
    }

    const nofp = std.mem.eql(u8, workload, "coremark-nofp");
    const seconds_text = try extractFieldOnce(stdout, "Total time (secs)", "Total time (secs)");
    try require(if (nofp) isDigitsOnly(seconds_text) else isFixed6Dp(seconds_text));
    const iter_sec_text = try extractFieldOnce(stdout, "Iterations/Sec", "Iterations/Sec");
    try require(if (nofp) isDigitsOnly(iter_sec_text) else isFixed6Dp(iter_sec_text));
    const ticks_text = try extractFieldOnce(stdout, "Total ticks", "Total ticks");
    try require(isDigitsOnly(ticks_text));

    const ticks = std.fmt.parseInt(i64, ticks_text, 10) catch return error.ValidationFailed;
    try require(ticks >= 1);
    const seconds_f = std.fmt.parseFloat(f64, seconds_text) catch return error.ValidationFailed;
    try require(seconds_f > 0.0 and std.math.isFinite(seconds_f));
    try require(std.math.isFinite(parsed.throughput) and parsed.throughput > 0.0);

    const tick_seconds: f64 = @as(f64, @floatFromInt(ticks)) / @as(f64, @floatFromInt(coremark_ticks_per_second));
    const expected_seconds: f64 = if (nofp)
        @floatFromInt(@divFloor(ticks, coremark_ticks_per_second))
    else
        tick_seconds;
    const seconds_tolerance: f64 = if (nofp) 0.0 else 0.5e-6 + ulp(tick_seconds);
    try require(@abs(seconds_f - expected_seconds) <= seconds_tolerance);

    const expected_rate: f64 = if (nofp)
        @floatFromInt(@divFloor(expected_iterations, @as(i64, @intFromFloat(expected_seconds))))
    else
        @as(f64, @floatFromInt(expected_iterations)) / tick_seconds;
    const rate_tolerance: f64 = if (nofp) 0.0 else 0.5e-6 + ulp(expected_rate);
    try require(@abs(parsed.throughput - expected_rate) <= rate_tolerance);

    if (invocation_seconds) |inv_s| {
        const enclosing_tol = (1.0 / @as(f64, @floatFromInt(coremark_ticks_per_second))) +
            clock_resolution_seconds + ulp(tick_seconds) + ulp(inv_s);
        try require(tick_seconds <= inv_s + enclosing_tol);
    }

    var res: std.json.ObjectMap = .empty;
    try res.put(allocator, "crc", .{ .object = crc_map });
    try res.put(allocator, "iterations", .{ .integer = parsed.iterations });
    try res.put(allocator, "iterations_per_second", .{ .float = parsed.throughput });
    try res.put(allocator, "reported_seconds", if (nofp) .{ .integer = @intFromFloat(seconds_f) } else .{ .float = seconds_f });
    try res.put(allocator, "reported_ticks", .{ .integer = ticks });
    try res.put(allocator, "reported_ticks_per_second", .{ .integer = coremark_ticks_per_second });
    try res.put(allocator, "seconds_from_ticks", .{ .float = tick_seconds });
    try res.put(allocator, "time_format", .{ .string = if (nofp) "integer-truncated" else "float-6dp" });
    try res.put(allocator, "invocation_timing_available", .{ .bool = invocation_seconds != null });
    try res.put(allocator, "minimum_timing_met", .{
        .bool = invocation_seconds != null and seconds_f >= 10.0 and tick_seconds >= 10.0 and invocation_seconds.? >= 10.0,
    });
    try res.put(allocator, "compliance", .{ .string = "not-certified" });
    return .{ .object = res };
}

pub fn validateResetEvents(result: Value, config: Value, steady_count: usize) !i64 {
    const r_obj = result.object;
    const events_val = r_obj.get("reset_events").?;
    try require(events_val == .array);
    const events = events_val.array.items;
    try require(events.len >= steady_count and events.len <= steady_count + 1);
    const max_steady: usize = @intCast(config.object.get("steady_invocations").?.integer);
    try require(events.len <= max_steady);
    const invs = r_obj.get("invocations").?.array.items;
    try require(events.len == 0 or invs.len > 0);
    if (std.mem.eql(u8, r_obj.get("outcome").?.string, "success")) {
        try require(events.len == steady_count);
    }
    var total: i64 = 0;
    for (events, 0..) |ev, idx| {
        const ev_obj = try requireKeys(ev, &.{ "before_invocation", "outcome", "elapsed_ticks" });
        const bi = ev_obj.get("before_invocation").?;
        try require(bi == .integer and bi.integer == @as(i64, @intCast(idx + 2)));
        const out = ev_obj.get("outcome").?;
        try require(out == .string and (std.mem.eql(u8, out.string, "completed") or std.mem.eql(u8, out.string, "error")));
        const el = ev_obj.get("elapsed_ticks").?;
        if (el != .null) {
            const ticks = try requireIntMin(el, 0);
            total +|= ticks;
        }
        if (idx < steady_count) {
            try require(std.mem.eql(u8, out.string, "completed") and el != .null);
        }
    }
    return total;
}

pub fn validateResult(allocator: Allocator, result: Value, manifest: Value, run_id: []const u8) !std.json.Array {
    const config = try runConfiguration(allocator, manifest, run_id);
    const target = config.object.get("target").?;
    const t_obj = target.object;
    const r_obj = try requireKeys(result, &.{
        "schema_version", "kind",          "evidence_kind",        "campaign_id",
        "run_id",         "config_sha256", "image_receipt_sha256", "observed",
        "outcome",        "exit_code",     "clock",                "phase_contract",
        "phases",         "invocations",   "memory",               "execution_lifecycle",
        "reset_events",
    });
    try requireHeader(r_obj, "wamr-native-benchmark-result");
    const m_obj = manifest.object;
    try require(r_obj.get("evidence_kind").? == .string and
        std.mem.eql(u8, r_obj.get("evidence_kind").?.string, m_obj.get("evidence_kind").?.string));
    try require(r_obj.get("campaign_id").? == .string and
        std.mem.eql(u8, r_obj.get("campaign_id").?.string, m_obj.get("campaign_id").?.string));
    try require(r_obj.get("run_id").? == .string and std.mem.eql(u8, r_obj.get("run_id").?.string, run_id));
    const cfg_sha = try cacheKeyAlloc(allocator, config);
    try require(r_obj.get("config_sha256").? == .string and std.mem.eql(u8, r_obj.get("config_sha256").?.string, cfg_sha));
    try require(r_obj.get("image_receipt_sha256").? == .string and
        std.mem.eql(u8, r_obj.get("image_receipt_sha256").?.string, t_obj.get("image_receipt_sha256").?.string));

    const wl_name = config.object.get("run").?.object.get("workload").?.string;
    var exp_obs: std.json.ObjectMap = .empty;
    try exp_obs.put(allocator, "image_sha256", t_obj.get("image").?.object.get("sha256").?);
    try exp_obs.put(allocator, "runtime_sha256", t_obj.get("runtime").?.object.get("sha256").?);
    try exp_obs.put(allocator, "aot_sha256", t_obj.get("aot_modules").?.object.get(wl_name).?.object.get("sha256").?);
    try exp_obs.put(allocator, "wasm_sha256", config.object.get("workload").?.object.get("sha256").?);
    try exp_obs.put(allocator, "platform", t_obj.get("platform").?);
    try exp_obs.put(allocator, "options", t_obj.get("options").?);
    try exp_obs.put(allocator, "mode", .{ .string = "aot" });
    try exp_obs.put(allocator, "jit_preset", .null);
    try exp_obs.put(allocator, "compile_profile", t_obj.get("compile_profile").?);
    try require(try identical(allocator, r_obj.get("observed").?, .{ .object = exp_obs }));

    const outcome_val = r_obj.get("outcome").?;
    try require(outcome_val == .string);
    var valid_outcome = false;
    for (outcomes) |o| if (std.mem.eql(u8, outcome_val.string, o)) {
        valid_outcome = true;
    };
    try require(valid_outcome);

    const exit_code = r_obj.get("exit_code").?;
    try require(exit_code == .null or (exit_code == .integer and exit_code.integer >= 0 and exit_code.integer <= 0xffffffff));
    try require(!std.mem.eql(u8, outcome_val.string, "trap") or exit_code == .null);
    try require(r_obj.get("phase_contract").? == .string and std.mem.eql(u8, r_obj.get("phase_contract").?.string, phase_contract));
    try require(try identical(allocator, r_obj.get("execution_lifecycle").?, t_obj.get("execution_lifecycle").?));

    const phases = try requireKeys(r_obj.get("phases").?, &phase_keys);
    try require(phases.get("compile_ticks").? == .null);
    const clock_val = r_obj.get("clock").?;
    if (clock_val == .null) {
        try require(!std.mem.eql(u8, outcome_val.string, "success"));
        for ([_][]const u8{ "load_ticks", "instantiate_ticks", "lifecycle_setup_ticks", "first_invocation_ticks" }) |pname| {
            try require(phases.get(pname).? == .null);
        }
        try require(phases.get("steady_state_ticks").? == .array and phases.get("steady_state_ticks").?.array.items.len == 0);
        try require(r_obj.get("invocations").? == .array and r_obj.get("invocations").?.array.items.len == 0);
        try require(r_obj.get("reset_events").? == .array and r_obj.get("reset_events").?.array.items.len == 0);
        try require(r_obj.get("memory").? == .null and exit_code == .null);
        return std.json.Array.init(allocator);
    }

    const clock = try requireKeys(clock_val, &.{ "source", "unit", "ticks_per_second", "resolution_ticks" });
    _ = try requirePublicToken(clock.get("source").?);
    const unit = clock.get("unit").?;
    try require(unit == .string);
    const exp_tps: i64 = if (std.mem.eql(u8, unit.string, "ns"))
        1_000_000_000
    else if (std.mem.eql(u8, unit.string, "us"))
        1_000_000
    else if (std.mem.eql(u8, unit.string, "ms"))
        1000
    else
        return error.ValidationFailed;
    const tps_val = clock.get("ticks_per_second").?;
    try require(tps_val == .integer and tps_val.integer == exp_tps);
    const res_ticks = try requireIntMin(clock.get("resolution_ticks").?, 1);
    try require(res_ticks <= exp_tps);

    const success = std.mem.eql(u8, outcome_val.string, "success");
    for ([_][]const u8{ "load_ticks", "instantiate_ticks", "first_invocation_ticks" }) |pname| {
        const pv = phases.get(pname).?;
        try require(!success or pv != .null);
        if (pv != .null) _ = try requireIntMin(pv, 0);
    }
    const snapshot = std.mem.eql(u8, r_obj.get("execution_lifecycle").?.object.get("mode").?.string, "snapshot-replay");
    const setup_pv = phases.get("lifecycle_setup_ticks").?;
    if (snapshot) {
        try require(!success or setup_pv != .null);
        if (setup_pv != .null) _ = try requireIntMin(setup_pv, 0);
    } else {
        try require(setup_pv == .null);
    }

    const steady_arr = phases.get("steady_state_ticks").?;
    try require(steady_arr == .array);
    const max_steady: usize = @intCast(config.object.get("steady_invocations").?.integer);
    try require(steady_arr.array.items.len <= max_steady);
    if (success) try require(steady_arr.array.items.len == max_steady);
    for (steady_arr.array.items) |sv| {
        if (sv != .null) {
            _ = try requireIntMin(sv, 0);
        } else {
            try require(!success);
        }
    }

    try require(phases.get("instantiate_ticks").? == .null or phases.get("load_ticks").? != .null);
    try require(phases.get("first_invocation_ticks").? == .null or phases.get("instantiate_ticks").? != .null);
    const invs_val = r_obj.get("invocations").?;
    try require(invs_val == .array);
    const invs = invs_val.array.items;
    try require(invs.len > 0 or (phases.get("first_invocation_ticks").? == .null and steady_arr.array.items.len == 0));

    var durations: std.ArrayList(?i64) = .empty;
    if (invs.len > 0) {
        const ft = phases.get("first_invocation_ticks").?;
        try durations.append(allocator, if (ft == .null) null else try requireIntMin(ft, 0));
        for (steady_arr.array.items) |sv| {
            try durations.append(allocator, if (sv == .null) null else try requireIntMin(sv, 0));
        }
    }
    try require(invs.len == durations.items.len);
    try require(durations.items.len == 0 or phases.get("instantiate_ticks").? != .null);
    try require(!snapshot or durations.items.len == 0 or setup_pv != .null);
    try require(setup_pv == .null or phases.get("instantiate_ticks").? != .null);

    const reset_ticks = try validateResetEvents(result, config, steady_arr.array.items.len);
    var timed_ticks: i128 = reset_ticks;
    for (durations.items) |d| {
        if (d) |v| timed_ticks += v;
    }
    for ([_][]const u8{ "load_ticks", "instantiate_ticks", "lifecycle_setup_ticks" }) |pname| {
        const pv = phases.get(pname).?;
        if (pv != .null) timed_ticks += pv.integer;
    }
    const created_ns = try parseUtcTimestampNs(m_obj.get("created_at").?);
    const expires_ns = try parseUtcTimestampNs(m_obj.get("expires_at").?);
    const campaign_seconds: f64 = @as(f64, @floatFromInt(expires_ns - created_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));
    try require(@as(f64, @floatFromInt(timed_ticks)) <= campaign_seconds * @as(f64, @floatFromInt(exp_tps)));

    var checks: std.json.Array = .init(allocator);
    for (invs, durations.items, 0..) |inv, ticks_opt, idx| {
        const inv_obj = try requireKeys(inv, &.{
            "phase", "outcome", "exit_code", "stdout_base64", "stdout_complete", "measurement_errors",
        });
        const p_str = inv_obj.get("phase").?;
        try require(p_str == .string and std.mem.eql(u8, p_str.string, if (idx == 0) "first" else "steady"));
        const i_out = inv_obj.get("outcome").?;
        try require(i_out == .string and (std.mem.eql(u8, i_out.string, "returned") or
            std.mem.eql(u8, i_out.string, "proc_exit") or
            std.mem.eql(u8, i_out.string, "trap") or
            std.mem.eql(u8, i_out.string, "error")));
        const i_exit = inv_obj.get("exit_code").?;
        try require(i_exit == .null or (i_exit == .integer and i_exit.integer >= 0 and i_exit.integer <= 0xffffffff));
        try require(!std.mem.eql(u8, i_out.string, "proc_exit") or i_exit != .null);
        try require(std.mem.eql(u8, i_out.string, "proc_exit") or i_exit == .null);

        const errs_val = inv_obj.get("measurement_errors").?;
        try require(errs_val == .array);
        for (errs_val.array.items, 0..) |ev, e_idx| {
            try require(ev == .string);
            var allowed = false;
            for (measurement_errors) |me| if (std.mem.eql(u8, ev.string, me)) {
                allowed = true;
            };
            try require(allowed);
            if (e_idx > 0) {
                try require(std.mem.lessThan(u8, errs_val.array.items[e_idx - 1].string, ev.string));
            }
        }
        const stdout_complete = inv_obj.get("stdout_complete").?;
        try require(stdout_complete == .bool);
        try require(stdout_complete.bool == !containsString(errs_val.array.items, "stdout-copy"));
        try require((ticks_opt == null) == containsString(errs_val.array.items, "post-call-clock"));

        const stdout_bytes = try decodeStdout(allocator, inv_obj.get("stdout_base64").?);
        const good = std.mem.eql(u8, i_out.string, "returned") or
            (std.mem.eql(u8, i_out.string, "proc_exit") and i_exit.integer == 0);
        const continues = idx + 1 < durations.items.len or r_obj.get("reset_events").?.array.items.len > idx;
        try require(!continues or (good and errs_val.array.items.len == 0));
        if (std.mem.eql(u8, i_out.string, "proc_exit") and i_exit.integer != 0) {
            try require(exit_code == .integer and exit_code.integer == i_exit.integer);
        }
        try require(!success or (good and errs_val.array.items.len == 0));

        if (!stdout_complete.bool) {
            var chk: std.json.ObjectMap = .empty;
            try chk.put(allocator, "self_check", .{ .string = "unavailable" });
            try chk.put(allocator, "failure", .{ .string = "stdout-copy" });
            try checks.append(.{ .object = chk });
        } else if (good and std.mem.startsWith(u8, wl_name, "coremark")) {
            for (stdout_bytes) |b| {
                if (b >= 128) {
                    if (success) return error.ValidationFailed;
                    var chk: std.json.ObjectMap = .empty;
                    try chk.put(allocator, "self_check", .{ .string = "failed" });
                    try chk.put(allocator, "failure", .{ .string = "coremark-validation" });
                    try checks.append(.{ .object = chk });
                    break;
                }
            } else {
                const inv_s: ?f64 = if (ticks_opt) |t| @as(f64, @floatFromInt(t)) / @as(f64, @floatFromInt(exp_tps)) else null;
                const res_s: f64 = @as(f64, @floatFromInt(res_ticks)) / @as(f64, @floatFromInt(exp_tps));
                if (coremarkCorrectness(allocator, stdout_bytes, inv_s, wl_name, res_s)) |cm_chk| {
                    try checks.append(cm_chk);
                } else |_| {
                    if (success) return error.ValidationFailed;
                    var chk: std.json.ObjectMap = .empty;
                    try chk.put(allocator, "self_check", .{ .string = "failed" });
                    try chk.put(allocator, "failure", .{ .string = "coremark-validation" });
                    try checks.append(.{ .object = chk });
                }
            }
        } else if (good) {
            try require(std.mem.eql(u8, i_out.string, "returned") and stdout_bytes.len == 0);
            var chk: std.json.ObjectMap = .empty;
            try chk.put(allocator, "self_check", .{ .string = "returned-without-trap" });
            try checks.append(.{ .object = chk });
        } else {
            var chk: std.json.ObjectMap = .empty;
            try chk.put(allocator, "self_check", .{ .string = "failed" });
            try checks.append(.{ .object = chk });
        }
    }

    const expected_exit: Value = if (invs.len > 0) invs[invs.len - 1].object.get("exit_code").? else .null;
    try require(try identical(allocator, exit_code, expected_exit));
    const mem_val = r_obj.get("memory").?;
    if (success) {
        try validateMemory(allocator, mem_val, target, true);
    } else if (mem_val != .null) {
        try validateMemory(allocator, mem_val, target, false);
    }
    if (mem_val != .null) {
        const reached: usize = @as(usize, @intFromBool(phases.get("instantiate_ticks").? != .null)) +
            @as(usize, @intFromBool(invs.len > 0)) +
            @as(usize, @intFromBool(invs.len > 1));
        try require(mem_val.object.get("samples").?.array.items.len <= reached);
    }
    return checks;
}

pub const ObservationOptions = struct {
    started_at: []const u8,
    completed_at: []const u8,
    observation_seconds: f64,
    control_plane_seconds: ?f64 = null,
    transport_outcome: []const u8 = "exited",
    process_returncode: ?i64 = 0,
};

pub fn persistObservation(
    allocator: Allocator,
    io: std.Io,
    manifest: Value,
    run_id: []const u8,
    output: []const u8,
    stdout: []const u8,
    stderr: []const u8,
    opts: ObservationOptions,
) !void {
    const config = try runConfiguration(allocator, manifest, run_id);
    const stdout_path = try std.fs.path.join(allocator, &.{ output, "stdout.bin" });
    const stderr_path = try std.fs.path.join(allocator, &.{ output, "stderr.bin" });
    try privateWrite(io, stdout_path, stdout);
    try privateWrite(io, stderr_path, stderr);

    var rec: std.json.ObjectMap = .empty;
    try rec.put(allocator, "schema_version", .{ .integer = version });
    try rec.put(allocator, "kind", .{ .string = "wamr-native-host-observation" });
    try rec.put(allocator, "evidence_kind", manifest.object.get("evidence_kind").?);
    try rec.put(allocator, "campaign_id", manifest.object.get("campaign_id").?);
    try rec.put(allocator, "run_id", .{ .string = run_id });
    try rec.put(allocator, "config_sha256", .{ .string = try cacheKeyAlloc(allocator, config) });
    try rec.put(allocator, "started_at", .{ .string = opts.started_at });
    try rec.put(allocator, "completed_at", .{ .string = opts.completed_at });
    try rec.put(allocator, "observation_seconds", .{ .float = opts.observation_seconds });
    try rec.put(allocator, "control_plane_seconds", if (opts.control_plane_seconds) |c| .{ .float = c } else .null);
    try rec.put(allocator, "transport_outcome", .{ .string = opts.transport_outcome });
    try rec.put(allocator, "process_returncode", if (opts.process_returncode) |rc| .{ .integer = rc } else .null);
    try rec.put(allocator, "stdout_sha256", .{ .string = try sha256FileAlloc(allocator, io, stdout_path) });
    try rec.put(allocator, "stderr_sha256", .{ .string = try sha256FileAlloc(allocator, io, stderr_path) });

    const obs_path = try std.fs.path.join(allocator, &.{ output, "observation.json" });
    const obs_bytes = try std.fmt.allocPrint(allocator, "{s}\n", .{try encoded(allocator, .{ .object = rec })});
    try privateWrite(io, obs_path, obs_bytes);
}

pub fn capture(
    allocator: Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
    manifest: Value,
    run_id: []const u8,
    command: []const []const u8,
    output: []const u8,
    timeout_s: f64,
    allow_synthetic: bool,
) !bool {
    try validateManifest(allocator, manifest, allow_synthetic);
    const now_ns: i128 = @intCast(std.Io.Clock.real.now(io).nanoseconds);
    const created_ns = try parseUtcTimestampNs(manifest.object.get("created_at").?);
    const expires_ns = try parseUtcTimestampNs(manifest.object.get("expires_at").?);
    try require(created_ns <= now_ns and now_ns < expires_ns);

    const config = try runConfiguration(allocator, manifest, run_id);
    try require(std.mem.eql(u8, config.object.get("run").?.object.get("target").?.string, "linux"));
    try require(command.len > 0);
    try require(timeout_s >= 0.001 and std.math.isFinite(timeout_s));
    const rem_s: f64 = @as(f64, @floatFromInt(expires_ns - now_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));
    try require(timeout_s <= rem_s);

    if (std.Io.Dir.cwd().statFile(io, output, .{})) |_| return error.ValidationFailed else |_| {}
    if (std.fs.path.dirname(output)) |parent| {
        if (parent.len != 0) try std.Io.Dir.cwd().createDirPath(io, parent);
    }
    try std.Io.Dir.cwd().createDir(io, output, if (builtin.os.tag == .windows) .default_dir else .fromMode(0o700));

    const cfg_sha = try cacheKeyAlloc(allocator, config);
    var req_map: std.json.ObjectMap = .empty;
    try req_map.put(allocator, "config", config);
    try req_map.put(allocator, "config_sha256", .{ .string = cfg_sha });
    const req_path = try std.fs.path.join(allocator, &.{ output, "request.json" });
    try privateWrite(io, req_path, try std.fmt.allocPrint(allocator, "{s}\n", .{try encoded(allocator, .{ .object = req_map })}));
    const abs_req = try std.Io.Dir.cwd().realPathFileAlloc(io, req_path, allocator);

    var child_env = try environ_map.clone(allocator);
    try child_env.put("WAMR_BENCH_REQUEST", abs_req);
    try child_env.put("WAMR_BENCH_CONFIG_SHA256", cfg_sha);

    const start_mono = std.Io.Clock.awake.now(io).nanoseconds;
    var status: []const u8 = "exited";
    var returncode: ?i64 = null;
    var stdout_bytes: []const u8 = "";
    var stderr_bytes: []const u8 = "";

    const timeout_ms: i64 = @max(1, @as(i64, @intFromFloat(timeout_s * 1000.0)));
    if (runTimedMs(allocator, io, .{ .argv = command, .environ_map = &child_env }, timeout_ms)) |res| {
        stdout_bytes = res.stdout;
        stderr_bytes = res.stderr;
        returncode = switch (res.term) {
            .exited => |c| @intCast(c),
            .signal => |s| -@as(i64, @intCast(@backingInt(s))),
            else => 1,
        };
    } else |err| {
        if (err == error.Timeout) {
            status = "timeout";
        } else {
            status = "launch_error";
            stderr_bytes = try std.fmt.allocPrint(allocator, "{s}", .{@errorName(err)});
        }
    }
    const end_mono = std.Io.Clock.awake.now(io).nanoseconds;
    const elapsed_s: f64 = @as(f64, @floatFromInt(@max(0, end_mono - start_mono))) / @as(f64, @floatFromInt(std.time.ns_per_s));
    const completed_ns: i128 = @intCast(std.Io.Clock.real.now(io).nanoseconds);

    try persistObservation(allocator, io, manifest, run_id, output, stdout_bytes, stderr_bytes, .{
        .started_at = try formatUtcTimestamp(allocator, now_ns),
        .completed_at = try formatUtcTimestamp(allocator, completed_ns),
        .observation_seconds = elapsed_s,
        .transport_outcome = status,
        .process_returncode = returncode,
    });
    if (!std.mem.eql(u8, status, "exited")) return false;
    const result = try parseResultStream(allocator, stdout_bytes);
    _ = try validateResult(allocator, result, manifest, run_id);
    const out_str = result.object.get("outcome").?.string;
    try require(returncode.? == 0 or !std.mem.eql(u8, out_str, "success"));
    return std.mem.eql(u8, out_str, "success");
}

fn runTimedMs(allocator: Allocator, io: std.Io, options: std.process.RunOptions, timeout_ms: i64) !std.process.RunResult {
    var child = try std.process.spawn(io, .{
        .argv = options.argv,
        .cwd = options.cwd,
        .environ_map = options.environ_map,
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .pipe,
    });
    defer if (child.id != null) {
        process_control.forceKill(&child) catch {};
        child.kill(io);
    };
    const deadline: std.Io.Timeout = .{ .deadline = .fromNow(io, .{ .clock = .awake, .raw = .fromMilliseconds(timeout_ms) }) };
    var buffer: std.Io.File.MultiReader.Buffer(2) = undefined;
    var reader: std.Io.File.MultiReader = undefined;
    reader.init(allocator, io, buffer.toStreams(), &.{ child.stdout.?, child.stderr.? });
    defer reader.deinit();
    while (reader.fill(options.reserve_amount, deadline)) |_| {} else |err| switch (err) {
        error.EndOfStream => {},
        else => return err,
    }
    try reader.checkAnyError();
    const term = while (true) {
        if (try process_control.pollChild(&child, io)) |result| break result;
        if (deadline.deadline.durationFromNow(io).raw.nanoseconds <= 0) return error.Timeout;
        try io.sleep(.fromMilliseconds(2), .awake);
    };
    const stdout = try reader.toOwnedSlice(0);
    errdefer allocator.free(stdout);
    const stderr = try reader.toOwnedSlice(1);
    return .{ .term = term, .stdout = stdout, .stderr = stderr };
}

pub fn consumeCapture(allocator: Allocator, io: std.Io, manifest: Value, directory: []const u8, allow_synthetic: bool) !Value {
    const obs_path = try std.fs.path.join(allocator, &.{ directory, "observation.json" });
    const observation = try readJson(allocator, io, obs_path);
    const obs = try requireKeys(observation, &.{
        "schema_version",      "kind",                  "evidence_kind",     "campaign_id",
        "run_id",              "config_sha256",         "started_at",        "completed_at",
        "observation_seconds", "control_plane_seconds", "transport_outcome", "process_returncode",
        "stdout_sha256",       "stderr_sha256",
    });
    try requireHeader(obs, "wamr-native-host-observation");
    const m_obj = manifest.object;
    const ek = obs.get("evidence_kind").?;
    try require(ek == .string and std.mem.eql(u8, ek.string, m_obj.get("evidence_kind").?.string) and
        (std.mem.eql(u8, ek.string, "measurement") or allow_synthetic));
    const run_id = obs.get("run_id").?;
    try require(run_id == .string);
    const config = try runConfiguration(allocator, manifest, run_id.string);
    const cfg_sha = try cacheKeyAlloc(allocator, config);
    try require(obs.get("campaign_id").? == .string and
        std.mem.eql(u8, obs.get("campaign_id").?.string, m_obj.get("campaign_id").?.string) and
        obs.get("config_sha256").? == .string and
        std.mem.eql(u8, obs.get("config_sha256").?.string, cfg_sha));

    const start_ns = try parseUtcTimestampNs(obs.get("started_at").?);
    const end_ns = try parseUtcTimestampNs(obs.get("completed_at").?);
    const created_ns = try parseUtcTimestampNs(m_obj.get("created_at").?);
    const expires_ns = try parseUtcTimestampNs(m_obj.get("expires_at").?);
    try require(created_ns <= start_ns and start_ns <= end_ns and end_ns <= expires_ns);
    _ = try requireNumberMin(obs.get("observation_seconds").?, 0.0);
    if (obs.get("control_plane_seconds").? != .null) {
        _ = try requireNumberMin(obs.get("control_plane_seconds").?, 0.0);
    }
    const to = obs.get("transport_outcome").?;
    try require(to == .string and (std.mem.eql(u8, to.string, "exited") or
        std.mem.eql(u8, to.string, "timeout") or
        std.mem.eql(u8, to.string, "launch_error")));
    const prc = obs.get("process_returncode").?;
    try require(prc == .null or prc == .integer);
    try require(std.mem.eql(u8, to.string, "exited") == (prc != .null));

    const stdout_path = try std.fs.path.join(allocator, &.{ directory, "stdout.bin" });
    const stderr_path = try std.fs.path.join(allocator, &.{ directory, "stderr.bin" });
    try require(obs.get("stdout_sha256").? == .string and
        std.mem.eql(u8, obs.get("stdout_sha256").?.string, try sha256FileAlloc(allocator, io, stdout_path)));
    try require(obs.get("stderr_sha256").? == .string and
        std.mem.eql(u8, obs.get("stderr_sha256").?.string, try sha256FileAlloc(allocator, io, stderr_path)));

    const raw = try std.Io.Dir.cwd().readFileAlloc(io, stdout_path, allocator, .limited(64 * 1024 * 1024));
    var result_val: Value = .null;
    var checks: std.json.Array = .init(allocator);
    if (!std.mem.eql(u8, to.string, "exited")) {
        try require(std.mem.indexOf(u8, raw, prefix) == null);
    } else {
        result_val = try parseResultStream(allocator, raw);
        checks = try validateResult(allocator, result_val, manifest, run_id.string);
        try require(prc.integer == 0 or !std.mem.eql(u8, result_val.object.get("outcome").?.string, "success"));
    }
    var rec: std.json.ObjectMap = .empty;
    try rec.put(allocator, "run", config.object.get("run").?);
    try rec.put(allocator, "observation", observation);
    try rec.put(allocator, "result", result_val);
    try rec.put(allocator, "correctness", .{ .array = checks });
    return .{ .object = rec };
}

pub fn importCapture(
    allocator: Allocator,
    io: std.Io,
    manifest: Value,
    run_id: []const u8,
    directory: []const u8,
    stdout: []const u8,
    stderr: []const u8,
    opts: ObservationOptions,
    allow_synthetic: bool,
) !Value {
    try validateManifest(allocator, manifest, allow_synthetic);
    if (std.Io.Dir.cwd().statFile(io, directory, .{})) |_| return error.ValidationFailed else |_| {}
    try persistObservation(allocator, io, manifest, run_id, directory, stdout, stderr, opts);
    return try consumeCapture(allocator, io, manifest, directory, allow_synthetic);
}

fn sampleStatsValue(allocator: Allocator, values: []const f64) !Value {
    if (values.len == 0) return .null;
    const stats = try benchmark_stats.summarize(allocator, values);
    var vals_arr: std.json.Array = .init(allocator);
    for (values) |v| try vals_arr.append(.{ .float = v });
    var map: std.json.ObjectMap = .empty;
    try map.put(allocator, "values", .{ .array = vals_arr });
    try map.put(allocator, "runs", .{ .integer = @intCast(stats.runs) });
    try map.put(allocator, "mean", .{ .float = stats.mean });
    try map.put(allocator, "median", .{ .float = stats.median });
    try map.put(allocator, "min", .{ .float = stats.min });
    try map.put(allocator, "max", .{ .float = stats.max });
    try map.put(allocator, "range", .{ .float = stats.range });
    return .{ .object = map };
}

pub fn buildReport(
    allocator: Allocator,
    io: std.Io,
    manifest: Value,
    directories: []const []const u8,
    allow_synthetic: bool,
) !Value {
    try validateManifest(allocator, manifest, allow_synthetic);
    const m_obj = manifest.object;
    const schedule = m_obj.get("schedule").?.array.items;
    try require(directories.len == schedule.len);

    const records = try allocator.alloc(Value, directories.len);
    for (directories, 0..) |dir, idx| {
        records[idx] = try consumeCapture(allocator, io, manifest, dir, allow_synthetic);
        const rid = records[idx].object.get("run").?.object.get("run_id").?.string;
        for (records[0..idx]) |prev| {
            try require(!std.mem.eql(u8, prev.object.get("run").?.object.get("run_id").?.string, rid));
        }
    }
    std.mem.sort(Value, records, {}, struct {
        fn lessThan(_: void, a: Value, b: Value) bool {
            return a.object.get("run").?.object.get("position").?.integer <
                b.object.get("run").?.object.get("position").?.integer;
        }
    }.lessThan);
    for (records, schedule) |rec, sched| {
        try require(std.mem.eql(
            u8,
            rec.object.get("run").?.object.get("run_id").?.string,
            sched.object.get("run_id").?.string,
        ));
    }

    const workloads_obj = m_obj.get("workloads").?.object;
    for (workloads_obj.keys()) |wname| {
        var final_crc: ?[]const u8 = null;
        for (records) |rec| {
            if (!std.mem.eql(u8, rec.object.get("run").?.object.get("workload").?.string, wname)) continue;
            for (rec.object.get("correctness").?.array.items) |chk| {
                if (chk.object.get("crc")) |crc_val| {
                    const cf = crc_val.object.get("crcfinal").?.string;
                    if (final_crc) |prev| {
                        try require(std.mem.eql(u8, prev, cf));
                    } else {
                        final_crc = cf;
                    }
                }
            }
        }
    }

    var all_success = true;
    for (records) |rec| {
        const res = rec.object.get("result").?;
        if (res == .null or !std.mem.eql(u8, res.object.get("outcome").?.string, "success")) {
            all_success = false;
        }
    }

    var summary_map: std.json.ObjectMap = .empty;
    for ([_][]const u8{ "linux", "unikraft" }) |target_name| {
        var target_summary: std.json.ObjectMap = .empty;
        const lifecycle = m_obj.get("targets").?.object.get(target_name).?.object.get("execution_lifecycle").?;
        const lmode = lifecycle.object.get("mode").?.string;
        for (workloads_obj.keys()) |wname| {
            var selected: std.ArrayList(Value) = .empty;
            for (records) |rec| {
                const run_obj = rec.object.get("run").?.object;
                const res = rec.object.get("result").?;
                if (std.mem.eql(u8, run_obj.get("target").?.string, target_name) and
                    std.mem.eql(u8, run_obj.get("workload").?.string, wname) and
                    std.mem.eql(u8, run_obj.get("phase").?.string, "measured") and
                    res != .null and
                    std.mem.eql(u8, res.object.get("outcome").?.string, "success"))
                {
                    try selected.append(allocator, rec);
                }
            }
            var row: std.json.ObjectMap = .empty;
            try row.put(allocator, "successful_runs", .{ .integer = @intCast(selected.items.len) });
            try row.put(allocator, "planned_runs", m_obj.get("runs").?);
            try row.put(allocator, "execution_lifecycle", lifecycle);

            for ([_]struct { tick_key: []const u8, sec_key: []const u8 }{
                .{ .tick_key = "load_ticks", .sec_key = "load_seconds" },
                .{ .tick_key = "instantiate_ticks", .sec_key = "instantiate_seconds" },
                .{ .tick_key = "first_invocation_ticks", .sec_key = "first_invocation_seconds" },
            }) |pk| {
                var vals: std.ArrayList(f64) = .empty;
                for (selected.items) |rec| {
                    const res_obj = rec.object.get("result").?.object;
                    const t = res_obj.get("phases").?.object.get(pk.tick_key).?.integer;
                    const tps = res_obj.get("clock").?.object.get("ticks_per_second").?.integer;
                    try vals.append(allocator, @as(f64, @floatFromInt(t)) / @as(f64, @floatFromInt(tps)));
                }
                try row.put(allocator, pk.sec_key, try sampleStatsValue(allocator, vals.items));
            }

            var repeat_vals: std.ArrayList(f64) = .empty;
            var reset_vals: std.ArrayList(f64) = .empty;
            var combined_vals: std.ArrayList(f64) = .empty;
            var setup_vals: std.ArrayList(f64) = .empty;
            for (selected.items) |rec| {
                const res_obj = rec.object.get("result").?.object;
                const tps: f64 = @floatFromInt(res_obj.get("clock").?.object.get("ticks_per_second").?.integer);
                const steady_items = res_obj.get("phases").?.object.get("steady_state_ticks").?.array.items;
                const reset_items = res_obj.get("reset_events").?.array.items;
                for (steady_items) |st| {
                    try repeat_vals.append(allocator, @as(f64, @floatFromInt(st.integer)) / tps);
                }
                for (reset_items) |ev| {
                    try reset_vals.append(allocator, @as(f64, @floatFromInt(ev.object.get("elapsed_ticks").?.integer)) / tps);
                }
                for (reset_items, steady_items) |ev, st| {
                    const sum_ticks = ev.object.get("elapsed_ticks").?.integer + st.integer;
                    try combined_vals.append(allocator, @as(f64, @floatFromInt(sum_ticks)) / tps);
                }
                const setup_t = res_obj.get("phases").?.object.get("lifecycle_setup_ticks").?;
                if (setup_t != .null) {
                    try setup_vals.append(allocator, @as(f64, @floatFromInt(setup_t.integer)) / tps);
                }
            }
            const repeats = try sampleStatsValue(allocator, repeat_vals.items);
            try row.put(allocator, "repeated_invocation_seconds", repeats);
            try row.put(allocator, "steady_state_seconds", if (std.mem.eql(u8, lmode, "same-instance-warm")) repeats else .null);
            try row.put(allocator, "snapshot_replay_seconds", if (std.mem.eql(u8, lmode, "snapshot-replay")) repeats else .null);
            try row.put(allocator, "reset_seconds", try sampleStatsValue(allocator, reset_vals.items));
            try row.put(allocator, "reset_and_invocation_seconds", try sampleStatsValue(allocator, combined_vals.items));
            try row.put(allocator, "lifecycle_setup_seconds", try sampleStatsValue(allocator, setup_vals.items));
            try target_summary.put(allocator, wname, .{ .object = row });
        }
        try summary_map.put(allocator, target_name, .{ .object = target_summary });
    }

    var timing_met = true;
    for (records) |rec| {
        for (rec.object.get("correctness").?.array.items) |chk| {
            if (chk.object.get("minimum_timing_met")) |mtm| {
                if (!mtm.bool) timing_met = false;
            }
        }
    }
    var has_coremark = false;
    for (workloads_obj.keys()) |wname| {
        if (std.mem.startsWith(u8, wname, "coremark")) has_coremark = true;
    }

    var public_records: std.json.Array = .init(allocator);
    for (records) |rec| {
        const rec_obj = rec.object;
        const res = rec_obj.get("result").?;
        var pub_res: Value = .null;
        if (res != .null) {
            var res_copy: std.json.ObjectMap = .empty;
            var r_it = res.object.iterator();
            while (r_it.next()) |re| {
                if (std.mem.eql(u8, re.key_ptr.*, "invocations")) continue;
                try res_copy.put(allocator, re.key_ptr.*, re.value_ptr.*);
            }
            var pub_invs: std.json.Array = .init(allocator);
            for (res.object.get("invocations").?.array.items) |inv| {
                var inv_copy: std.json.ObjectMap = .empty;
                var i_it = inv.object.iterator();
                while (i_it.next()) |ie| {
                    if (std.mem.eql(u8, ie.key_ptr.*, "stdout_base64")) continue;
                    try inv_copy.put(allocator, ie.key_ptr.*, ie.value_ptr.*);
                }
                const decoded = try decodeStdout(allocator, inv.object.get("stdout_base64").?);
                try inv_copy.put(allocator, "stdout_sha256", .{ .string = try sha256BytesAlloc(allocator, decoded) });
                try pub_invs.append(.{ .object = inv_copy });
            }
            try res_copy.put(allocator, "invocations", .{ .array = pub_invs });
            pub_res = .{ .object = res_copy };
        }
        var pub_rec: std.json.ObjectMap = .empty;
        try pub_rec.put(allocator, "run", rec_obj.get("run").?);
        try pub_rec.put(allocator, "observation", rec_obj.get("observation").?);
        try pub_rec.put(allocator, "result", pub_res);
        try pub_rec.put(allocator, "correctness", rec_obj.get("correctness").?);
        try public_records.append(.{ .object = pub_rec });
    }

    const pcounts = profileCounts(m_obj.get("profile").?.string).?;
    const counts_match = m_obj.get("warmups").?.integer == pcounts.warmups and
        m_obj.get("runs").?.integer == pcounts.runs;

    var status_map: std.json.ObjectMap = .empty;
    try status_map.put(allocator, "all_attempts_successful", .{ .bool = all_success });
    try status_map.put(
        allocator,
        "execution_lifecycle",
        m_obj.get("targets").?.object.get("linux").?.object.get("execution_lifecycle").?.object.get("mode").?,
    );
    try status_map.put(allocator, "profile_counts_match", .{ .bool = counts_match });
    try status_map.put(
        allocator,
        "coremark_minimum_timing_met",
        if (has_coremark) .{ .bool = timing_met and all_success } else .null,
    );
    try status_map.put(
        allocator,
        "coremark_compliance",
        .{ .string = if (has_coremark) "not-certified" else "not-applicable" },
    );
    try status_map.put(
        allocator,
        "paired_measurement_complete",
        .{ .bool = all_success and std.mem.eql(u8, m_obj.get("evidence_kind").?.string, "measurement") },
    );
    try status_map.put(allocator, "identity_assurance", .{ .string = "observations-and-receipt-attestations" });
    try status_map.put(allocator, "hardware_qualification", .{ .string = "requires-independent-deployment-evidence" });
    try status_map.put(allocator, "os_attribution", .{ .string = "requires-experimental-review" });
    try status_map.put(allocator, "performance_threshold", .null);

    const now_ns: i128 = @intCast(std.Io.Clock.real.now(io).nanoseconds);
    var out_map: std.json.ObjectMap = .empty;
    try out_map.put(allocator, "schema_version", .{ .integer = version });
    try out_map.put(allocator, "kind", .{ .string = "wamr-native-matched-comparison" });
    try out_map.put(allocator, "evidence_kind", m_obj.get("evidence_kind").?);
    try out_map.put(allocator, "generated_at", .{ .string = try formatUtcTimestamp(allocator, now_ns) });
    try out_map.put(allocator, "manifest_sha256", .{ .string = try cacheKeyAlloc(allocator, manifest) });
    try out_map.put(allocator, "plan", manifest);
    try out_map.put(allocator, "records", .{ .array = public_records });
    try out_map.put(allocator, "summary", .{ .object = summary_map });
    try out_map.put(allocator, "status", .{ .object = status_map });
    return .{ .object = out_map };
}

pub const ReportSummary = struct {
    document: Value,
    schema_version: i64,
    comparable: bool,
    sdk_commit: []const u8,
};

pub fn createReport(
    allocator: Allocator,
    io: std.Io,
    manifest: Value,
    directories: []const []const u8,
    allow_synthetic: bool,
) !ReportSummary {
    const doc = try buildReport(allocator, io, manifest, directories, allow_synthetic);
    const status_obj = doc.object.get("status").?.object;
    const plan_obj = doc.object.get("plan").?.object;
    const unikraft_commit = plan_obj.get("targets").?.object.get("unikraft").?.object.get("source").?.object.get("commit").?.string;
    return .{
        .document = doc,
        .schema_version = doc.object.get("schema_version").?.integer,
        .comparable = status_obj.get("paired_measurement_complete").?.bool,
        .sdk_commit = unikraft_commit,
    };
}

fn writeJsonDocument(allocator: Allocator, io: std.Io, path: []const u8, doc: Value) !void {
    if (std.Io.Dir.cwd().statFile(io, path, .{})) |_| return error.ValidationFailed else |_| {}
    if (std.fs.path.dirname(path)) |parent| {
        if (parent.len != 0) try std.Io.Dir.cwd().createDirPath(io, parent);
    }
    const payload = try std.fmt.allocPrint(allocator, "{s}\n", .{try encoded(allocator, doc)});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = payload });
}

pub fn runCli(
    allocator: Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
    argv: []const []const u8,
) u8 {
    return runCliInner(allocator, io, environ_map, argv) catch |err| {
        if (!builtin.is_test) std.debug.print("native benchmark: {s}\n", .{@errorName(err)});
        return 2;
    };
}

fn runCliInner(
    allocator: Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
    argv: []const []const u8,
) !u8 {
    if (argv.len == 0) return error.InvalidCliArguments;
    const cmd = argv[0];
    const rest = argv[1..];

    if (std.mem.eql(u8, cmd, "native-abi-proof")) {
        var config_path: ?[]const u8 = null;
        var aot_path: ?[]const u8 = null;
        var aot_sha256: ?[]const u8 = null;
        var source_repo: ?[]const u8 = null;
        var source_commit: ?[]const u8 = null;
        var source_tree_sha256: ?[]const u8 = null;
        var source_tracked_diff_sha256: ?[]const u8 = null;
        var linux_target_abi: ?[]const u8 = null;
        var unikraft_target_abi: ?[]const u8 = null;
        var out_path: ?[]const u8 = null;
        var allow_synthetic = false;
        var i: usize = 0;
        while (i < rest.len) : (i += 1) {
            const arg = rest[i];
            if (std.mem.eql(u8, arg, "--allow-synthetic")) {
                allow_synthetic = true;
                continue;
            }
            if (i + 1 >= rest.len) return error.InvalidCliArguments;
            i += 1;
            const val = rest[i];
            if (std.mem.eql(u8, arg, "--config")) {
                config_path = val;
            } else if (std.mem.eql(u8, arg, "--aot") or std.mem.eql(u8, arg, "--aot-path")) {
                aot_path = val;
            } else if (std.mem.eql(u8, arg, "--aot-sha256")) {
                aot_sha256 = val;
            } else if (std.mem.eql(u8, arg, "--source-repo")) {
                source_repo = val;
            } else if (std.mem.eql(u8, arg, "--source-commit")) {
                source_commit = val;
            } else if (std.mem.eql(u8, arg, "--source-tree-sha256")) {
                source_tree_sha256 = val;
            } else if (std.mem.eql(u8, arg, "--source-tracked-diff-sha256")) {
                source_tracked_diff_sha256 = val;
            } else if (std.mem.eql(u8, arg, "--linux-target-abi")) {
                linux_target_abi = val;
            } else if (std.mem.eql(u8, arg, "--unikraft-target-abi")) {
                unikraft_target_abi = val;
            } else if (std.mem.eql(u8, arg, "--out")) {
                out_path = val;
            } else return error.InvalidCliArguments;
        }
        var cfg_map: std.json.ObjectMap = if (config_path) |cpath| blk: {
            const parsed = try readJson(allocator, io, cpath);
            try require(parsed == .object);
            break :blk parsed.object;
        } else .empty;
        if (aot_path) |v| try cfg_map.put(allocator, "aot_path", .{ .string = v });
        if (aot_sha256) |v| try cfg_map.put(allocator, "aot_sha256", .{ .string = v });
        if (source_repo) |v| try cfg_map.put(allocator, "source_repo", .{ .string = v });
        if (source_commit) |v| try cfg_map.put(allocator, "source_commit", .{ .string = v });
        if (source_tree_sha256) |v| try cfg_map.put(allocator, "source_tree_sha256", .{ .string = v });
        if (source_tracked_diff_sha256) |v| try cfg_map.put(allocator, "source_tracked_diff_sha256", .{ .string = v });
        if (linux_target_abi) |v| try cfg_map.put(allocator, "linux_target_abi", .{ .string = v });
        if (unikraft_target_abi) |v| try cfg_map.put(allocator, "unikraft_target_abi", .{ .string = v });
        const doc = try createAbiProof(allocator, io, .{ .object = cfg_map }, allow_synthetic);
        try writeJsonDocument(allocator, io, out_path orelse return error.InvalidCliArguments, doc);
        return 0;
    } else if (std.mem.eql(u8, cmd, "native-linux-receipt")) {
        var config_path: ?[]const u8 = null;
        var out_path: ?[]const u8 = null;
        var allow_synthetic = false;
        var i: usize = 0;
        while (i < rest.len) : (i += 1) {
            const arg = rest[i];
            if (std.mem.eql(u8, arg, "--allow-synthetic")) {
                allow_synthetic = true;
                continue;
            }
            if (i + 1 >= rest.len) return error.InvalidCliArguments;
            i += 1;
            const val = rest[i];
            if (std.mem.eql(u8, arg, "--config")) {
                config_path = val;
            } else if (std.mem.eql(u8, arg, "--out")) {
                out_path = val;
            } else return error.InvalidCliArguments;
        }
        const doc = try createImageReceipt(
            allocator,
            io,
            try readJson(allocator, io, config_path orelse return error.InvalidCliArguments),
            allow_synthetic,
        );
        try writeJsonDocument(allocator, io, out_path orelse return error.InvalidCliArguments, doc);
        return 0;
    } else if (std.mem.eql(u8, cmd, "native-plan")) {
        var config_path: ?[]const u8 = null;
        var repo_path: []const u8 = ".";
        var out_path: ?[]const u8 = null;
        var allow_synthetic = false;
        var i: usize = 0;
        while (i < rest.len) : (i += 1) {
            const arg = rest[i];
            if (std.mem.eql(u8, arg, "--allow-synthetic")) {
                allow_synthetic = true;
                continue;
            }
            if (std.mem.eql(u8, arg, "--config")) {
                i += 1;
                if (i >= rest.len) return error.InvalidCliArguments;
                config_path = rest[i];
            } else if (std.mem.eql(u8, arg, "--repo")) {
                i += 1;
                if (i >= rest.len) return error.InvalidCliArguments;
                repo_path = rest[i];
            } else if (std.mem.eql(u8, arg, "--out")) {
                i += 1;
                if (i >= rest.len) return error.InvalidCliArguments;
                out_path = rest[i];
            } else return error.InvalidCliArguments;
        }
        const doc = try createPlan(
            allocator,
            io,
            try readJson(allocator, io, config_path orelse return error.InvalidCliArguments),
            repo_path,
            null,
            allow_synthetic,
        );
        try writeJsonDocument(allocator, io, out_path orelse return error.InvalidCliArguments, doc);
        return 0;
    } else if (std.mem.eql(u8, cmd, "native-request")) {
        var manifest_path: ?[]const u8 = null;
        var run_id: ?[]const u8 = null;
        var out_path: ?[]const u8 = null;
        var i: usize = 0;
        while (i < rest.len) : (i += 1) {
            const arg = rest[i];
            if (std.mem.eql(u8, arg, "--manifest")) {
                i += 1;
                if (i >= rest.len) return error.InvalidCliArguments;
                manifest_path = rest[i];
            } else if (std.mem.eql(u8, arg, "--run-id")) {
                i += 1;
                if (i >= rest.len) return error.InvalidCliArguments;
                run_id = rest[i];
            } else if (std.mem.eql(u8, arg, "--out")) {
                i += 1;
                if (i >= rest.len) return error.InvalidCliArguments;
                out_path = rest[i];
            } else return error.InvalidCliArguments;
        }
        const manifest = try readJson(allocator, io, manifest_path orelse return error.InvalidCliArguments);
        const doc = try createRequest(allocator, manifest, run_id orelse return error.InvalidCliArguments, false);
        try writeJsonDocument(allocator, io, out_path orelse return error.InvalidCliArguments, doc);
        return 0;
    } else if (std.mem.eql(u8, cmd, "native-capture")) {
        var manifest_path: ?[]const u8 = null;
        var run_id: ?[]const u8 = null;
        var out_path: ?[]const u8 = null;
        var timeout_s: ?f64 = null;
        var producer: []const []const u8 = &.{};
        var i: usize = 0;
        while (i < rest.len) : (i += 1) {
            const arg = rest[i];
            if (std.mem.eql(u8, arg, "--manifest")) {
                i += 1;
                if (i >= rest.len) return error.InvalidCliArguments;
                manifest_path = rest[i];
            } else if (std.mem.eql(u8, arg, "--run-id")) {
                i += 1;
                if (i >= rest.len) return error.InvalidCliArguments;
                run_id = rest[i];
            } else if (std.mem.eql(u8, arg, "--out")) {
                i += 1;
                if (i >= rest.len) return error.InvalidCliArguments;
                out_path = rest[i];
            } else if (std.mem.eql(u8, arg, "--timeout")) {
                i += 1;
                if (i >= rest.len) return error.InvalidCliArguments;
                timeout_s = std.fmt.parseFloat(f64, rest[i]) catch return error.InvalidCliArguments;
            } else {
                producer = if (std.mem.eql(u8, arg, "--")) rest[i + 1 ..] else rest[i..];
                break;
            }
        }
        const manifest = try readJson(allocator, io, manifest_path orelse return error.InvalidCliArguments);
        try validateManifest(allocator, manifest, false);
        const ok = try capture(
            allocator,
            io,
            environ_map,
            manifest,
            run_id orelse return error.InvalidCliArguments,
            producer,
            out_path orelse return error.InvalidCliArguments,
            timeout_s orelse return error.InvalidCliArguments,
            false,
        );
        return if (ok) 0 else 1;
    } else if (std.mem.eql(u8, cmd, "native-import")) {
        var manifest_path: ?[]const u8 = null;
        var run_id: ?[]const u8 = null;
        var stdout_path: ?[]const u8 = null;
        var stderr_path: ?[]const u8 = null;
        var started_at: ?[]const u8 = null;
        var completed_at: ?[]const u8 = null;
        var obs_s: ?f64 = null;
        var cp_s: ?f64 = null;
        var transport_outcome: []const u8 = "exited";
        var proc_rc: ?i64 = null;
        var out_path: ?[]const u8 = null;
        var i: usize = 0;
        while (i < rest.len) : (i += 1) {
            const arg = rest[i];
            if (i + 1 >= rest.len) return error.InvalidCliArguments;
            i += 1;
            const val = rest[i];
            if (std.mem.eql(u8, arg, "--manifest")) manifest_path = val else if (std.mem.eql(u8, arg, "--run-id")) run_id = val else if (std.mem.eql(u8, arg, "--stdout")) stdout_path = val else if (std.mem.eql(u8, arg, "--stderr")) stderr_path = val else if (std.mem.eql(u8, arg, "--started-at")) started_at = val else if (std.mem.eql(u8, arg, "--completed-at")) completed_at = val else if (std.mem.eql(u8, arg, "--observation-seconds")) obs_s = std.fmt.parseFloat(f64, val) catch return error.InvalidCliArguments else if (std.mem.eql(u8, arg, "--control-plane-seconds")) cp_s = std.fmt.parseFloat(f64, val) catch return error.InvalidCliArguments else if (std.mem.eql(u8, arg, "--transport-outcome")) transport_outcome = val else if (std.mem.eql(u8, arg, "--process-returncode")) proc_rc = std.fmt.parseInt(i64, val, 10) catch return error.InvalidCliArguments else if (std.mem.eql(u8, arg, "--out")) out_path = val else return error.InvalidCliArguments;
        }
        const manifest = try readJson(allocator, io, manifest_path orelse return error.InvalidCliArguments);
        const out = out_path orelse return error.InvalidCliArguments;
        const stdout_b = try std.Io.Dir.cwd().readFileAlloc(io, stdout_path orelse return error.InvalidCliArguments, allocator, .limited(64 * 1024 * 1024));
        const stderr_b = try std.Io.Dir.cwd().readFileAlloc(io, stderr_path orelse return error.InvalidCliArguments, allocator, .limited(64 * 1024 * 1024));
        const rec = try importCapture(allocator, io, manifest, run_id orelse return error.InvalidCliArguments, out, stdout_b, stderr_b, .{
            .started_at = started_at orelse return error.InvalidCliArguments,
            .completed_at = completed_at orelse return error.InvalidCliArguments,
            .observation_seconds = obs_s orelse return error.InvalidCliArguments,
            .control_plane_seconds = cp_s,
            .transport_outcome = transport_outcome,
            .process_returncode = proc_rc,
        }, false);
        const res = rec.object.get("result").?;
        return if (res != .null and std.mem.eql(u8, res.object.get("outcome").?.string, "success")) 0 else 1;
    } else if (std.mem.eql(u8, cmd, "native-report")) {
        var manifest_path: ?[]const u8 = null;
        var captures: std.ArrayList([]const u8) = .empty;
        var out_path: ?[]const u8 = null;
        var i: usize = 0;
        while (i < rest.len) : (i += 1) {
            const arg = rest[i];
            if (i + 1 >= rest.len) return error.InvalidCliArguments;
            i += 1;
            const val = rest[i];
            if (std.mem.eql(u8, arg, "--manifest")) manifest_path = val else if (std.mem.eql(u8, arg, "--capture")) try captures.append(allocator, val) else if (std.mem.eql(u8, arg, "--out")) out_path = val else return error.InvalidCliArguments;
        }
        if (captures.items.len == 0) return error.InvalidCliArguments;
        const manifest = try readJson(allocator, io, manifest_path orelse return error.InvalidCliArguments);
        const rep = try createReport(allocator, io, manifest, captures.items, false);
        try writeJsonDocument(allocator, io, out_path orelse return error.InvalidCliArguments, rep.document);
        return if (rep.document.object.get("status").?.object.get("all_attempts_successful").?.bool) 0 else 1;
    }
    return error.InvalidCliArguments;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const code = runCli(allocator, init.io, init.environ_map, if (args.len > 1) args[1..] else &.{});
    if (code != 0) std.process.exit(code);
}

// ============================================================================
// Tests: NativeGuestProtocolTests, NativeBenchmarkTests, NativeLinuxProducerTests
// ============================================================================

fn testCloneValue(allocator: Allocator, val: Value) !Value {
    const json_bytes = try encoded(allocator, val);
    return strictJson(allocator, json_bytes);
}

fn testEncodeBase64(allocator: Allocator, raw: []const u8) ![]u8 {
    const len = std.base64.standard.Encoder.calcSize(raw.len);
    const buf = try allocator.alloc(u8, len);
    _ = std.base64.standard.Encoder.encode(buf, raw);
    return buf;
}

const sample_coremark_stdout =
    "2K performance run parameters for coremark.\n" ++
    "CoreMark Size    : 666\n" ++
    "Total ticks      : 12500\n" ++
    "Total time (secs): 12.500000\n" ++
    "Iterations/Sec   : 32000.000000\n" ++
    "Iterations       : 400000\n" ++
    "Compiler version : Clang\n" ++
    "Compiler flags   : -O3\n" ++
    "Memory location  : STACK\n" ++
    "seedcrc          : 0xe9f5\n" ++
    "[0]crclist       : 0xe714\n" ++
    "[0]crcmatrix     : 0x1fd7\n" ++
    "[0]crcstate      : 0x8e3a\n" ++
    "[0]crcfinal      : 0x5275\n" ++
    "Correct operation validated. See README.txt for run and reporting rules.\n" ++
    "CoreMark 1.0 : 32000.000000 / Clang -O3 / STACK\n";

const sample_coremark_short_crc_stdout =
    "2K performance run parameters for coremark.\n" ++
    "CoreMark Size    : 666\n" ++
    "Total ticks      : 6\n" ++
    "Total time (secs): 0.006000\n" ++
    "Iterations/Sec   : 16666.666667\n" ++
    "ERROR! Must execute for at least 10 secs for a valid result!\n" ++
    "Iterations       : 100\n" ++
    "Compiler version : GCCClang 19.1.7\n" ++
    "Compiler flags   : -O3 -DPERFORMANCE_RUN=1  -lrt\n" ++
    "Memory location  : Please put data memory location here\n" ++
    "\t\t\t(e.g. code in flash, data on heap etc)\n" ++
    "seedcrc          : 0xe9f5\n" ++
    "[0]crclist       : 0xe714\n" ++
    "[0]crcmatrix     : 0x1fd7\n" ++
    "[0]crcstate      : 0x8e3a\n" ++
    "[0]crcfinal      : 0x988c\n" ++
    "Errors detected\n";

fn testLifecycleSpec(allocator: Allocator, mode: []const u8) !Value {
    const warm = std.mem.eql(u8, mode, "same-instance-warm");
    var scope: std.json.Array = .init(allocator);
    const items = if (warm) &warm_reset_scope else &snapshot_reset_scope;
    for (items) |s| try scope.append(.{ .string = s });
    var map: std.json.ObjectMap = .empty;
    try map.put(allocator, "mode", .{ .string = mode });
    try map.put(allocator, "reset_policy", .{ .string = if (warm) "invocation-state-only" else "restore-post-start-snapshot" });
    try map.put(allocator, "reset_before", .{ .string = "each-steady-invocation" });
    try map.put(allocator, "reset_timing", .{ .string = "excluded-from-invocation" });
    try map.put(allocator, "reset_scope", .{ .array = scope });
    return .{ .object = map };
}

fn testMemoryPolicySpec(allocator: Allocator, os_name: []const u8) !Value {
    var alloc_map: std.json.ObjectMap = .empty;
    try alloc_map.put(allocator, "load", .{ .string = "wamr-Pool_Allocator" });
    try alloc_map.put(allocator, "instantiate", .{ .string = if (std.mem.eql(u8, os_name, "linux")) "host-mmap+wamr-Pool_Allocator" else "ukalloc+wamr-Pool_Allocator" });
    try alloc_map.put(allocator, "first", .{ .string = "preallocated-instance-memory" });
    try alloc_map.put(allocator, "steady", .{ .string = "reused-instance-memory" });

    var page_map: std.json.ObjectMap = .empty;
    try page_map.put(allocator, "reservation", .{ .string = "bounds-check-guard-reservation" });
    try page_map.put(allocator, "commitment", .{ .string = if (std.mem.eql(u8, os_name, "linux")) "demand-paged" else "eager-boot-mapped" });
    try page_map.put(allocator, "release", .{ .string = "retained-until-instance-teardown" });

    var map: std.json.ObjectMap = .empty;
    try map.put(allocator, "allocator_by_phase", .{ .object = alloc_map });
    try map.put(allocator, "page_policy", .{ .object = page_map });
    return .{ .object = map };
}

const SyntheticPlanFixture = struct {
    manifest: Value,
    config: Value,
    config_path: []const u8,
};

fn makeSyntheticPlanFixture(
    allocator: Allocator,
    io: std.Io,
    root: []const u8,
    opts: struct {
        profile: []const u8 = "ci",
        warmups: i64 = 0,
        runs: i64 = 1,
        steady_invocations: i64 = 2,
        workloads: []const []const u8 = &.{"coremark"},
        same_aot: bool = true,
        lifecycle_mode: []const u8 = "same-instance-warm",
        commit: []const u8 = unikraft_native_sdk_commit,
        allow_synthetic: bool = false,
    },
) !SyntheticPlanFixture {
    for (opts.workloads) |wname| {
        const fix = findFixture(wname).?;
        const full_fix = try std.fs.path.join(allocator, &.{ root, fix.path });
        if (std.Io.Dir.cwd().statFile(io, full_fix, .{})) |_| {} else |_| {
            try std.Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(full_fix).?);
            const src_bytes = try std.Io.Dir.cwd().readFileAlloc(io, fix.path, allocator, .limited(64 * 1024 * 1024));
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = full_fix, .data = src_bytes });
        }
    }
    for ([_][]const u8{
        "scripts/native_benchmark.zig",
        "scripts/benchmark_schema.zig",
    }) |rel| {
        const full = try std.fs.path.join(allocator, &.{ root, rel });
        if (std.Io.Dir.cwd().statFile(io, full, .{})) |_| {} else |_| {
            try std.Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(full).?);
            if (std.Io.Dir.cwd().readFileAlloc(io, rel, allocator, .limited(64 * 1024 * 1024))) |src_bytes| {
                try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = full, .data = src_bytes });
            } else |_| {
                try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = full, .data = "stub\n" });
            }
        }
    }

    var source_map: std.json.ObjectMap = .empty;
    try source_map.put(allocator, "commit", .{ .string = opts.commit });
    try source_map.put(allocator, "tree_sha256", .{ .string = "2222222222222222222222222222222222222222222222222222222222222222" });
    try source_map.put(allocator, "tracked_diff_sha256", .{ .string = "0000000000000000000000000000000000000000000000000000000000000000" });
    const source_val: Value = .{ .object = source_map };

    var options_map: std.json.ObjectMap = .empty;
    try options_map.put(allocator, "optimize", .{ .string = "ReleaseFast" });
    try options_map.put(allocator, "bounds_checks", .{ .bool = true });
    try options_map.put(allocator, "wx_enforced", .{ .bool = true });
    try options_map.put(allocator, "import_checks", .{ .bool = true });
    try options_map.put(allocator, "trap_isolation", .{ .bool = true });
    try options_map.put(allocator, "stack_checks", .{ .bool = true });
    try options_map.put(allocator, "simd", .{ .bool = true });
    try options_map.put(allocator, "threads", .{ .bool = false });
    try options_map.put(allocator, "memory64", .{ .bool = false });
    const options_val: Value = .{ .object = options_map };

    var platform_map: std.json.ObjectMap = .empty;
    try platform_map.put(allocator, "arch", .{ .string = "x86_64" });
    try platform_map.put(allocator, "cpu_model", .{ .string = "Intel_Xeon_Platinum_8370C" });
    try platform_map.put(allocator, "active_cpu_count", .{ .integer = 2 });
    try platform_map.put(allocator, "azure_sku", .{ .string = "Standard_D2s_v5" });
    try platform_map.put(allocator, "azure_region", .{ .string = "eastus" });
    const platform_val: Value = .{ .object = platform_map };

    const lifecycle_val = try testLifecycleSpec(allocator, opts.lifecycle_mode);
    const compiler_path = try std.fs.path.join(allocator, &.{ root, "wamrc" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = compiler_path, .data = "compiler-binary" });

    var targets_cfg: std.json.ObjectMap = .empty;
    for ([_][]const u8{ "linux", "unikraft" }) |os_name| {
        const runtime_path = try std.fs.path.join(allocator, &.{ root, try std.fmt.allocPrint(allocator, "{s}-runtime.a", .{os_name}) });
        const image_path = try std.fs.path.join(allocator, &.{ root, try std.fmt.allocPrint(allocator, "{s}.img", .{os_name}) });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = runtime_path, .data = try std.fmt.allocPrint(allocator, "{s}-runtime", .{os_name}) });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = image_path, .data = try std.fmt.allocPrint(allocator, "{s}-image", .{os_name}) });

        var aot_paths_map: std.json.ObjectMap = .empty;
        for (opts.workloads) |wname| {
            const suffix = if (opts.same_aot) "shared" else os_name;
            const aot_path = try std.fs.path.join(allocator, &.{ root, try std.fmt.allocPrint(allocator, "{s}-{s}.aot", .{ wname, suffix }) });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = aot_path, .data = try std.fmt.allocPrint(allocator, "aot-{s}-{s}", .{ wname, suffix }) });
            try aot_paths_map.put(allocator, wname, .{ .string = aot_path });
        }

        const target_abi: []const u8 = if (!opts.same_aot and std.mem.eql(u8, os_name, "linux"))
            "x86_64-unknown-linux-gnu"
        else
            "x86_64-unknown-none-elf";
        const compile_profile: Value = if (opts.same_aot or std.mem.eql(u8, os_name, "unikraft"))
            .{ .string = "unikraft-x86_64" }
        else
            .null;
        const mem_policy = try testMemoryPolicySpec(allocator, os_name);

        var receipt_cfg: std.json.ObjectMap = .empty;
        try receipt_cfg.put(allocator, "os", .{ .string = os_name });
        try receipt_cfg.put(allocator, "image_path", .{ .string = image_path });
        try receipt_cfg.put(allocator, "runtime_path", .{ .string = runtime_path });
        try receipt_cfg.put(allocator, "compiler_path", .{ .string = compiler_path });
        try receipt_cfg.put(allocator, "compiler_version", .{ .string = "wamrc-2.4.1" });
        try receipt_cfg.put(allocator, "source", source_val);
        try receipt_cfg.put(allocator, "target_abi", .{ .string = target_abi });
        try receipt_cfg.put(allocator, "platform", platform_val);
        try receipt_cfg.put(allocator, "options", options_val);
        try receipt_cfg.put(allocator, "compile_profile", compile_profile);
        try receipt_cfg.put(allocator, "configured_vm_ram_bytes", .{ .integer = 536870912 });
        try receipt_cfg.put(allocator, "memory_policy", mem_policy);
        try receipt_cfg.put(allocator, "aot_paths", .{ .object = aot_paths_map });
        try receipt_cfg.put(allocator, "execution_lifecycle", lifecycle_val);
        const receipt_val = try createImageReceipt(allocator, io, .{ .object = receipt_cfg }, opts.allow_synthetic);

        const receipt_path = try std.fs.path.join(allocator, &.{ root, try std.fmt.allocPrint(allocator, "{s}-receipt.json", .{os_name}) });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = receipt_path, .data = try encoded(allocator, receipt_val) });

        var t_spec: std.json.ObjectMap = .empty;
        try t_spec.put(allocator, "runtime_path", .{ .string = runtime_path });
        try t_spec.put(allocator, "compiler_path", .{ .string = compiler_path });
        try t_spec.put(allocator, "compiler_version", .{ .string = "wamrc-2.4.1" });
        try t_spec.put(allocator, "source", source_val);
        try t_spec.put(allocator, "target_abi", .{ .string = target_abi });
        try t_spec.put(allocator, "platform", platform_val);
        try t_spec.put(allocator, "options", options_val);
        try t_spec.put(allocator, "image_path", .{ .string = image_path });
        try t_spec.put(allocator, "image_receipt_path", .{ .string = receipt_path });
        try t_spec.put(allocator, "aot_paths", .{ .object = aot_paths_map });
        try t_spec.put(allocator, "compile_profile", compile_profile);
        try t_spec.put(allocator, "memory_policy", mem_policy);
        try t_spec.put(allocator, "execution_lifecycle", lifecycle_val);
        try targets_cfg.put(allocator, os_name, .{ .object = t_spec });
    }

    var proofs_map: std.json.ObjectMap = .empty;
    for (opts.workloads) |wname| {
        if (opts.same_aot) {
            const proof_path = try std.fs.path.join(allocator, &.{ root, try std.fmt.allocPrint(allocator, "{s}-abi-proof.json", .{wname}) });
            const aot_path = targets_cfg.get("linux").?.object.get("aot_paths").?.object.get(wname).?;
            var proof_cfg: std.json.ObjectMap = .empty;
            try proof_cfg.put(allocator, "aot_path", aot_path);
            try proof_cfg.put(allocator, "source", source_val);
            try proof_cfg.put(allocator, "linux_target_abi", targets_cfg.get("linux").?.object.get("target_abi").?);
            try proof_cfg.put(allocator, "unikraft_target_abi", targets_cfg.get("unikraft").?.object.get("target_abi").?);
            const proof_val = try createAbiProof(allocator, io, .{ .object = proof_cfg }, opts.allow_synthetic);
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = proof_path, .data = try encoded(allocator, proof_val) });
            try proofs_map.put(allocator, wname, .{ .string = proof_path });
        } else {
            try proofs_map.put(allocator, wname, .null);
        }
    }

    var wl_json: std.json.Array = .init(allocator);
    for (opts.workloads) |wname| try wl_json.append(.{ .string = wname });

    var producer_source_map: std.json.ObjectMap = .empty;
    try producer_source_map.put(allocator, "commit", .{ .string = try resolveRefSha(allocator, io, ".", "HEAD") });
    try producer_source_map.put(allocator, "tree_sha256", .{ .string = "3333333333333333333333333333333333333333333333333333333333333333" });
    try producer_source_map.put(allocator, "tracked_diff_sha256", .{ .string = "0000000000000000000000000000000000000000000000000000000000000000" });

    var config_map: std.json.ObjectMap = .empty;
    try config_map.put(allocator, "profile", .{ .string = opts.profile });
    try config_map.put(allocator, "warmups", .{ .integer = opts.warmups });
    try config_map.put(allocator, "runs", .{ .integer = opts.runs });
    try config_map.put(allocator, "steady_invocations", .{ .integer = opts.steady_invocations });
    try config_map.put(allocator, "valid_hours", .{ .float = 24.0 });
    try config_map.put(allocator, "targets", .{ .object = targets_cfg });
    try config_map.put(allocator, "workloads", .{ .array = wl_json });
    try config_map.put(allocator, "abi_proofs", .{ .object = proofs_map });
    try config_map.put(allocator, "producer_source", .{ .object = producer_source_map });
    const config_val: Value = .{ .object = config_map };

    const config_path = try std.fs.path.join(allocator, &.{ root, "plan-config.json" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = config_path, .data = try encoded(allocator, config_val) });
    const now_ns: i128 = @intCast(std.Io.Clock.real.now(io).nanoseconds);
    const manifest = try createPlan(allocator, io, config_val, root, now_ns - 60 * @as(i128, std.time.ns_per_s), opts.allow_synthetic);
    return .{ .manifest = manifest, .config = config_val, .config_path = config_path };
}

fn makeSyntheticResult(
    allocator: Allocator,
    manifest: Value,
    run_id: []const u8,
    opts: struct {
        stdout_text: ?[]const u8 = null,
        outcome: []const u8 = "success",
        steady_ticks: []const i64 = &.{ 13_000_000_000, 12_900_000_000 },
        reset_ticks: ?[]const i64 = null,
    },
) !Value {
    const config = try runConfiguration(allocator, manifest, run_id);
    const target = config.object.get("target").?;
    const t_obj = target.object;
    const wl_name = config.object.get("run").?.object.get("workload").?.string;
    const stdout_raw = opts.stdout_text orelse (if (std.mem.startsWith(u8, wl_name, "coremark")) sample_coremark_stdout else "");
    const encoded_stdout = try testEncodeBase64(allocator, stdout_raw);
    const lifecycle = t_obj.get("execution_lifecycle").?;
    const snapshot = std.mem.eql(u8, lifecycle.object.get("mode").?.string, "snapshot-replay");

    var invs: std.json.Array = .init(allocator);
    for (0..(1 + opts.steady_ticks.len)) |idx| {
        var inv: std.json.ObjectMap = .empty;
        try inv.put(allocator, "phase", .{ .string = if (idx == 0) "first" else "steady" });
        try inv.put(allocator, "outcome", .{ .string = "returned" });
        try inv.put(allocator, "exit_code", .null);
        try inv.put(allocator, "stdout_base64", .{ .string = encoded_stdout });
        try inv.put(allocator, "stdout_complete", .{ .bool = true });
        try inv.put(allocator, "measurement_errors", .{ .array = .init(allocator) });
        try invs.append(.{ .object = inv });
    }

    var reset_events: std.json.Array = .init(allocator);
    for (0..opts.steady_ticks.len) |idx| {
        const ticks: i64 = if (opts.reset_ticks) |rt| rt[idx] else @intCast((idx + 1) * 5000);
        var ev: std.json.ObjectMap = .empty;
        try ev.put(allocator, "before_invocation", .{ .integer = @intCast(idx + 2) });
        try ev.put(allocator, "outcome", .{ .string = "completed" });
        try ev.put(allocator, "elapsed_ticks", .{ .integer = ticks });
        try reset_events.append(.{ .object = ev });
    }

    var obs: std.json.ObjectMap = .empty;
    try obs.put(allocator, "image_sha256", t_obj.get("image").?.object.get("sha256").?);
    try obs.put(allocator, "runtime_sha256", t_obj.get("runtime").?.object.get("sha256").?);
    try obs.put(allocator, "aot_sha256", t_obj.get("aot_modules").?.object.get(wl_name).?.object.get("sha256").?);
    try obs.put(allocator, "wasm_sha256", config.object.get("workload").?.object.get("sha256").?);
    try obs.put(allocator, "platform", t_obj.get("platform").?);
    try obs.put(allocator, "options", t_obj.get("options").?);
    try obs.put(allocator, "mode", .{ .string = "aot" });
    try obs.put(allocator, "jit_preset", .null);
    try obs.put(allocator, "compile_profile", t_obj.get("compile_profile").?);

    var clock: std.json.ObjectMap = .empty;
    try clock.put(allocator, "source", .{ .string = if (std.mem.eql(u8, t_obj.get("os").?.string, "linux")) "CLOCK_MONOTONIC_RAW" else "ukplat_monotonic_clock" });
    try clock.put(allocator, "unit", .{ .string = "ns" });
    try clock.put(allocator, "ticks_per_second", .{ .integer = 1_000_000_000 });
    try clock.put(allocator, "resolution_ticks", .{ .integer = 1 });

    var steady_arr: std.json.Array = .init(allocator);
    for (opts.steady_ticks) |st| try steady_arr.append(.{ .integer = st });
    var phases: std.json.ObjectMap = .empty;
    try phases.put(allocator, "compile_ticks", .null);
    try phases.put(allocator, "load_ticks", .{ .integer = 100_000 });
    try phases.put(allocator, "instantiate_ticks", .{ .integer = 200_000 });
    try phases.put(allocator, "lifecycle_setup_ticks", if (snapshot) .{ .integer = 50_000 } else .null);
    try phases.put(allocator, "first_invocation_ticks", .{ .integer = 13_100_000_000 });
    try phases.put(allocator, "steady_state_ticks", .{ .array = steady_arr });

    var covered: std.json.Array = .init(allocator);
    try covered.append(.{ .string = "wamr_heap" });
    try covered.append(.{ .string = "linear_memory" });
    var samples: std.json.Array = .init(allocator);
    for ([_][]const u8{ "after_instantiation", "after_first", "after_steady" }) |stage| {
        var s: std.json.ObjectMap = .empty;
        try s.put(allocator, "stage", .{ .string = stage });
        try s.put(allocator, "reserved_address_bytes", .{ .integer = 8_388_608 });
        try s.put(allocator, "committed_bytes", .{ .integer = 2_097_152 });
        try samples.append(.{ .object = s });
    }
    var memory: std.json.ObjectMap = .empty;
    try memory.put(allocator, "image_sha256", t_obj.get("image").?.object.get("sha256").?);
    try memory.put(allocator, "configured_vm_ram_bytes", t_obj.get("image_receipt").?.object.get("configured_vm_ram_bytes").?);
    try memory.put(allocator, "coverage", .{ .string = "complete-guest" });
    try memory.put(allocator, "method", .{ .string = if (std.mem.eql(u8, t_obj.get("os").?.string, "linux")) "proc_self_smaps_rollup" else "ukalloc_region_walk" });
    try memory.put(allocator, "covered_regions", .{ .array = covered });
    try memory.put(allocator, "omitted_regions", .{ .array = .init(allocator) });
    try memory.put(allocator, "samples", .{ .array = samples });
    try memory.put(allocator, "policy", t_obj.get("memory_policy").?);

    var res: std.json.ObjectMap = .empty;
    try res.put(allocator, "schema_version", .{ .integer = version });
    try res.put(allocator, "kind", .{ .string = "wamr-native-benchmark-result" });
    try res.put(allocator, "evidence_kind", manifest.object.get("evidence_kind").?);
    try res.put(allocator, "campaign_id", manifest.object.get("campaign_id").?);
    try res.put(allocator, "run_id", .{ .string = run_id });
    try res.put(allocator, "config_sha256", .{ .string = try cacheKeyAlloc(allocator, config) });
    try res.put(allocator, "image_receipt_sha256", t_obj.get("image_receipt_sha256").?);
    try res.put(allocator, "observed", .{ .object = obs });
    try res.put(allocator, "outcome", .{ .string = opts.outcome });
    try res.put(allocator, "exit_code", .null);
    try res.put(allocator, "clock", .{ .object = clock });
    try res.put(allocator, "phase_contract", .{ .string = phase_contract });
    try res.put(allocator, "phases", .{ .object = phases });
    try res.put(allocator, "invocations", .{ .array = invs });
    try res.put(allocator, "memory", .{ .object = memory });
    try res.put(allocator, "execution_lifecycle", lifecycle);
    try res.put(allocator, "reset_events", .{ .array = reset_events });
    return .{ .object = res };
}

fn writeSyntheticCapture(
    allocator: Allocator,
    io: std.Io,
    root: []const u8,
    manifest: Value,
    run_id: []const u8,
    result_opt: ?Value,
    transport_outcome: []const u8,
    process_returncode: ?i64,
    raw_stdout_opt: ?[]const u8,
) ![]const u8 {
    const out = try std.fs.path.join(allocator, &.{ root, try std.fmt.allocPrint(allocator, "capture-{s}", .{run_id}) });
    const raw_stdout = if (raw_stdout_opt) |r|
        r
    else blk: {
        const res = result_opt orelse try makeSyntheticResult(allocator, manifest, run_id, .{});
        break :blk try std.fmt.allocPrint(allocator, "boot banner\n{s}{s}\n", .{ prefix, try encoded(allocator, res) });
    };
    const created_ns = try parseUtcTimestampNs(manifest.object.get("created_at").?);
    const started = try formatUtcTimestamp(allocator, created_ns + std.time.ns_per_s);
    const completed = try formatUtcTimestamp(allocator, created_ns + 2 * std.time.ns_per_s);
    try persistObservation(allocator, io, manifest, run_id, out, raw_stdout, "", .{
        .started_at = started,
        .completed_at = completed,
        .observation_seconds = 1.0,
        .control_plane_seconds = 0.25,
        .transport_outcome = transport_outcome,
        .process_returncode = process_returncode,
    });
    return out;
}

test "NativeGuestProtocolTests" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const binary_z = std.c.getenv("WAMR_NATIVE_GUEST_TEST") orelse return error.SkipZigTest;
    const binary = std.mem.span(binary_z);
    var argv: std.ArrayList([]const u8) = .empty;
    if (std.c.getenv("WAMR_NATIVE_GUEST_TEST_RUNNER")) |runner_z| {
        var tok = std.mem.tokenizeAny(u8, std.mem.span(runner_z), " \t");
        while (tok.next()) |part| try argv.append(allocator, part);
    }
    try argv.append(allocator, binary);

    const res = try std.process.run(allocator, io, .{ .argv = argv.items });
    try std.testing.expect(res.term == .exited and res.term.exited == 0);

    var lines: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, std.mem.trim(u8, res.stdout, " \t\r\n"), '\n');
    while (it.next()) |raw_line| {
        const line = if (std.mem.endsWith(u8, raw_line, "\r")) raw_line[0 .. raw_line.len - 1] else raw_line;
        if (line.len > 0) try lines.append(allocator, line);
    }
    try std.testing.expectEqual(@as(usize, 12), lines.items.len);

    const manifest_prefix = "WAMR_NATIVE_TEST_MANIFEST=";
    const check_prefix = "WAMR_NATIVE_CHECK_RESULT=";
    var results: std.ArrayList(Value) = .empty;
    var index: usize = 0;
    while (index < lines.items.len) : (index += 2) {
        const m_line = lines.items[index];
        const r_line = lines.items[index + 1];
        try std.testing.expect(std.mem.startsWith(u8, m_line, manifest_prefix));
        try std.testing.expect(std.mem.startsWith(u8, r_line, check_prefix));
        const manifest = try strictJson(allocator, m_line[manifest_prefix.len..]);
        const result = try strictJson(allocator, r_line[check_prefix.len..]);
        try std.testing.expect(std.mem.indexOf(u8, manifest.object.get("campaign_id").?.string, "synthetic") != null);
        for ([_][]const u8{ "created_at", "expires_at" }) |field| {
            const stamp = manifest.object.get(field).?;
            _ = try parseUtcTimestampNs(stamp);
            try std.testing.expect(std.mem.endsWith(u8, stamp.string, "+00:00"));
        }
        _ = try validateResult(allocator, result, manifest, "run-0001");
        try std.testing.expectError(error.ValidationFailed, parseResultStream(allocator, r_line));
        try results.append(allocator, result);
    }

    const expected_outcomes = [_][]const u8{ "success", "error", "error", "error", "error", "error" };
    const expected_inv_lens = [_]usize{ 3, 1, 1, 1, 2, 1 };
    for (results.items, expected_outcomes, expected_inv_lens) |r, exp_out, exp_len| {
        try std.testing.expectEqualStrings(exp_out, r.object.get("outcome").?.string);
        try std.testing.expectEqual(exp_len, r.object.get("invocations").?.array.items.len);
    }
    try std.testing.expect(results.items[1].object.get("phases").?.object.get("first_invocation_ticks").? == .null);
    try std.testing.expectEqualStrings(
        "error",
        results.items[2].object.get("reset_events").?.array.items[0].object.get("outcome").?.string,
    );
    try std.testing.expectEqualStrings(
        "completed",
        results.items[3].object.get("reset_events").?.array.items[0].object.get("outcome").?.string,
    );
    try std.testing.expectEqual(@as(usize, 1), results.items[4].object.get("phases").?.object.get("steady_state_ticks").?.array.items.len);
    try std.testing.expect(results.items[4].object.get("phases").?.object.get("steady_state_ticks").?.array.items[0] == .null);
    try std.testing.expectEqual(@as(i64, 0xffffffff), results.items[5].object.get("exit_code").?.integer);
    try std.testing.expectEqualStrings(
        "proc_exit",
        results.items[5].object.get("invocations").?.array.items[0].object.get("outcome").?.string,
    );
    try std.testing.expectEqualStrings(
        "AP/DKAoA",
        results.items[5].object.get("invocations").?.array.items[0].object.get("stdout_base64").?.string,
    );
    try std.testing.expect(std.mem.indexOf(u8, res.stderr, "WAMR_NATIVE_INVOCATION=") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stderr, "method=caller-requested-bytes live=0") != null);
}

test "NativeBenchmarkTests" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);

    // 1. Plan schedule alternation, receipts, and ABI compatibility
    {
        const sub = try std.fs.path.join(allocator, &.{ root, "t1" });
        try std.Io.Dir.cwd().createDirPath(io, sub);
        const fix = try makeSyntheticPlanFixture(allocator, io, sub, .{
            .warmups = 1,
            .runs = 2,
            .workloads = &.{ "coremark", "compute" },
        });
        const sched = fix.manifest.object.get("schedule").?.array.items;
        try std.testing.expectEqual(@as(usize, 12), sched.len);
        try std.testing.expectEqualStrings("linux", sched[0].object.get("target").?.string);
        try std.testing.expectEqualStrings("unikraft", sched[1].object.get("target").?.string);
        try std.testing.expectEqualStrings("unikraft", sched[6].object.get("target").?.string);
        try std.testing.expectEqualStrings("linux", sched[7].object.get("target").?.string);

        const sub_diff = try std.fs.path.join(allocator, &.{ root, "t1_diff" });
        try std.Io.Dir.cwd().createDirPath(io, sub_diff);
        const diff_fix = try makeSyntheticPlanFixture(allocator, io, sub_diff, .{ .same_aot = false });
        try std.testing.expectEqualStrings(
            "target-specific",
            diff_fix.manifest.object.get("abi_compatibility").?.object.get("coremark").?.object.get("strategy").?.string,
        );

        var broken_same = try testCloneValue(allocator, fix.manifest);
        try broken_same.object.getPtr("abi_compatibility").?.object.getPtr("coremark").?.object.put(allocator, "proof_sha256", .null);
        try std.testing.expectError(error.ValidationFailed, validateManifest(allocator, broken_same, false));

        var broken_receipt = try testCloneValue(allocator, fix.manifest);
        try broken_receipt.object.getPtr("targets").?.object.getPtr("unikraft").?.object.getPtr("image_receipt").?.object.put(
            allocator,
            "compiler_embedded",
            .{ .bool = true },
        );
        try std.testing.expectError(error.ValidationFailed, validateManifest(allocator, broken_receipt, false));

        const sub_hist = try std.fs.path.join(allocator, &.{ root, "t1_hist" });
        try std.Io.Dir.cwd().createDirPath(io, sub_hist);
        const hist_fix = try makeSyntheticPlanFixture(allocator, io, sub_hist, .{
            .commit = historical_unikraft_native_sdk_commit,
        });
        try std.testing.expectEqualStrings(
            historical_unikraft_native_sdk_commit,
            hist_fix.manifest.object.get("targets").?.object.get("unikraft").?.object.get("source").?.object.get("commit").?.string,
        );

        const sub_post_wave12 = try std.fs.path.join(allocator, &.{ root, "t1_post_wave12" });
        try std.Io.Dir.cwd().createDirPath(io, sub_post_wave12);
        const post_wave12_fix = try makeSyntheticPlanFixture(allocator, io, sub_post_wave12, .{
            .commit = post_wave12_unikraft_native_sdk_commit,
        });
        try std.testing.expectEqualStrings(
            post_wave12_unikraft_native_sdk_commit,
            post_wave12_fix.manifest.object.get("targets").?.object.get("unikraft").?.object.get("source").?.object.get("commit").?.string,
        );

        const sub_current = try std.fs.path.join(allocator, &.{ root, "t1_current" });
        try std.Io.Dir.cwd().createDirPath(io, sub_current);
        const current_fix = try makeSyntheticPlanFixture(allocator, io, sub_current, .{
            .commit = current_unikraft_native_sdk_commit,
        });
        try std.testing.expectEqualStrings(
            current_unikraft_native_sdk_commit,
            current_fix.manifest.object.get("targets").?.object.get("unikraft").?.object.get("source").?.object.get("commit").?.string,
        );

        var wrong_commit = try testCloneValue(allocator, fix.manifest);
        for ([_][]const u8{ "linux", "unikraft" }) |os_name| {
            var t_ptr = wrong_commit.object.getPtr("targets").?.object.getPtr(os_name).?;
            try t_ptr.object.getPtr("source").?.object.put(allocator, "commit", .{ .string = "1111111111111111111111111111111111111111" });
            try t_ptr.object.getPtr("compiler").?.object.getPtr("source").?.object.put(allocator, "commit", .{ .string = "1111111111111111111111111111111111111111" });
            try t_ptr.object.getPtr("image_receipt").?.object.getPtr("source").?.object.put(allocator, "commit", .{ .string = "1111111111111111111111111111111111111111" });
            try t_ptr.object.getPtr("image_receipt").?.object.getPtr("compiler").?.object.getPtr("source").?.object.put(allocator, "commit", .{ .string = "1111111111111111111111111111111111111111" });
        }
        try std.testing.expectError(error.ValidationFailed, validateManifest(allocator, wrong_commit, false));
    }

    // 2. Execution lifecycles and reset events
    {
        const sub_warm = try std.fs.path.join(allocator, &.{ root, "t2_warm" });
        const sub_snap = try std.fs.path.join(allocator, &.{ root, "t2_snap" });
        try std.Io.Dir.cwd().createDirPath(io, sub_warm);
        try std.Io.Dir.cwd().createDirPath(io, sub_snap);
        const warm_fix = try makeSyntheticPlanFixture(allocator, io, sub_warm, .{ .lifecycle_mode = "same-instance-warm" });
        const snap_fix = try makeSyntheticPlanFixture(allocator, io, sub_snap, .{ .lifecycle_mode = "snapshot-replay" });
        const warm_res = try makeSyntheticResult(allocator, warm_fix.manifest, "run-0001", .{});
        const snap_res = try makeSyntheticResult(allocator, snap_fix.manifest, "run-0001", .{});
        _ = try validateResult(allocator, warm_res, warm_fix.manifest, "run-0001");
        _ = try validateResult(allocator, snap_res, snap_fix.manifest, "run-0001");

        var warm_with_setup = try testCloneValue(allocator, warm_res);
        try warm_with_setup.object.getPtr("phases").?.object.put(allocator, "lifecycle_setup_ticks", .{ .integer = 1 });
        try std.testing.expectError(error.ValidationFailed, validateResult(allocator, warm_with_setup, warm_fix.manifest, "run-0001"));

        var snap_no_setup = try testCloneValue(allocator, snap_res);
        try snap_no_setup.object.getPtr("phases").?.object.put(allocator, "lifecycle_setup_ticks", .null);
        try std.testing.expectError(error.ValidationFailed, validateResult(allocator, snap_no_setup, snap_fix.manifest, "run-0001"));

        var missing_reset = try testCloneValue(allocator, warm_res);
        _ = missing_reset.object.getPtr("reset_events").?.array.pop();
        try std.testing.expectError(error.ValidationFailed, validateResult(allocator, missing_reset, warm_fix.manifest, "run-0001"));
    }

    // 3. Strict stream framing and schema validation
    {
        const sub = try std.fs.path.join(allocator, &.{ root, "t3" });
        try std.Io.Dir.cwd().createDirPath(io, sub);
        const fix = try makeSyntheticPlanFixture(allocator, io, sub, .{});
        const res = try makeSyntheticResult(allocator, fix.manifest, "run-0001", .{});
        const raw_line = try std.fmt.allocPrint(allocator, "{s}{s}\n", .{ prefix, try encoded(allocator, res) });
        const framed = try std.fmt.allocPrint(allocator, "binary boot \xff\xfe\n{s}\x80\x00shutdown\n", .{raw_line});
        const parsed = try parseResultStream(allocator, framed);
        try std.testing.expectEqual(@as(usize, 3), (try validateResult(allocator, parsed, fix.manifest, "run-0001")).items.len);

        try std.testing.expectError(error.ValidationFailed, parseResultStream(allocator, "no sentinel\n"));
        try std.testing.expectError(error.ValidationFailed, parseResultStream(allocator, try std.fmt.allocPrint(allocator, "noise {s}", .{raw_line})));
        try std.testing.expectError(error.ValidationFailed, parseResultStream(allocator, try std.fmt.allocPrint(allocator, "{s}{s}", .{ raw_line, raw_line })));
        try std.testing.expectError(error.ValidationFailed, strictJson(allocator, "{\"a\":1,\"a\":2}"));
    }

    // 4. CoreMark and non-CoreMark correctness checks + short CRC output
    {
        const sub = try std.fs.path.join(allocator, &.{ root, "t4" });
        try std.Io.Dir.cwd().createDirPath(io, sub);
        const fix = try makeSyntheticPlanFixture(allocator, io, sub, .{
            .workloads = &.{ "coremark", "coremark-nofp", "compute" },
        });
        var rid_by_wl: std.StringHashMap([]const u8) = .init(allocator);
        for (fix.manifest.object.get("schedule").?.array.items) |item| {
            if (std.mem.eql(u8, item.object.get("target").?.string, "linux")) {
                try rid_by_wl.put(item.object.get("workload").?.string, item.object.get("run_id").?.string);
            }
        }
        const nofp_stdout = blk: {
            const s1 = try std.mem.replaceOwned(u8, allocator, sample_coremark_stdout, "Total ticks      : 12500", "Total ticks      : 12000");
            const s2 = try std.mem.replaceOwned(u8, allocator, s1, "Total time (secs): 12.500000", "Total time (secs): 12");
            break :blk try std.mem.replaceOwned(u8, allocator, s2, "Iterations/Sec   : 32000.000000", "Iterations/Sec   : 33333");
        };
        const nofp_res = try makeSyntheticResult(allocator, fix.manifest, rid_by_wl.get("coremark-nofp").?, .{ .stdout_text = nofp_stdout });
        const nofp_checks = try validateResult(allocator, nofp_res, fix.manifest, rid_by_wl.get("coremark-nofp").?);
        try std.testing.expectEqualStrings("integer-truncated", nofp_checks.items[0].object.get("time_format").?.string);

        const compute_res = try makeSyntheticResult(allocator, fix.manifest, rid_by_wl.get("compute").?, .{});
        const compute_checks = try validateResult(allocator, compute_res, fix.manifest, rid_by_wl.get("compute").?);
        try std.testing.expectEqualStrings("returned-without-trap", compute_checks.items[0].object.get("self_check").?.string);

        const short_summary = try validateCoremarkShortCrcOutput(allocator, sample_coremark_short_crc_stdout, "coremark");
        try std.testing.expectEqualStrings("correctness-only-not-performance", short_summary.object.get("qualification").?.string);
        try std.testing.expectEqual(@as(i64, 100), short_summary.object.get("iterations").?.integer);
        try std.testing.expect(!short_summary.object.get("minimum_timing_met").?.bool);

        const bad_short = try std.mem.replaceOwned(u8, allocator, sample_coremark_short_crc_stdout, "0x988c", "0x5275");
        try std.testing.expectError(error.ValidationFailed, validateCoremarkShortCrcOutput(allocator, bad_short, "coremark"));
    }

    // 5. Report aggregation for warm and snapshot modes
    {
        for ([_][]const u8{ "same-instance-warm", "snapshot-replay" }) |lmode| {
            const sub = try std.fs.path.join(allocator, &.{ root, try std.fmt.allocPrint(allocator, "t5_{s}", .{lmode}) });
            try std.Io.Dir.cwd().createDirPath(io, sub);
            const fix = try makeSyntheticPlanFixture(allocator, io, sub, .{
                .warmups = 1,
                .runs = 3,
                .workloads = &.{ "coremark", "compute" },
                .lifecycle_mode = lmode,
            });
            var dirs: std.ArrayList([]const u8) = .empty;
            for (fix.manifest.object.get("schedule").?.array.items) |entry| {
                const rid = entry.object.get("run_id").?.string;
                try dirs.append(allocator, try writeSyntheticCapture(allocator, io, sub, fix.manifest, rid, null, "exited", 0, null));
            }
            const rep = try buildReport(allocator, io, fix.manifest, dirs.items, false);
            try std.testing.expect(rep.object.get("status").?.object.get("all_attempts_successful").?.bool);
            try std.testing.expectEqualStrings(lmode, rep.object.get("status").?.object.get("execution_lifecycle").?.string);
            const cm_row = rep.object.get("summary").?.object.get("linux").?.object.get("coremark").?.object;
            try std.testing.expectEqual(@as(i64, 3), cm_row.get("successful_runs").?.integer);
            try std.testing.expectEqual(@as(i64, 6), cm_row.get("repeated_invocation_seconds").?.object.get("runs").?.integer);
            if (std.mem.eql(u8, lmode, "same-instance-warm")) {
                try std.testing.expect(cm_row.get("steady_state_seconds").? != .null);
                try std.testing.expect(cm_row.get("snapshot_replay_seconds").? == .null);
                try std.testing.expect(cm_row.get("lifecycle_setup_seconds").? == .null);
            } else {
                try std.testing.expect(cm_row.get("steady_state_seconds").? == .null);
                try std.testing.expect(cm_row.get("snapshot_replay_seconds").? != .null);
                try std.testing.expectEqual(@as(i64, 3), cm_row.get("lifecycle_setup_seconds").?.object.get("runs").?.integer);
            }
        }
    }

    // 6. Deterministic native-abi-proof and native-linux-receipt CLI + round-trip into native-plan
    {
        const sub = try std.fs.path.join(allocator, &.{ root, "t6_cli_roundtrip" });
        try std.Io.Dir.cwd().createDirPath(io, sub);
        const fix = try makeSyntheticPlanFixture(allocator, io, sub, .{
            .workloads = &.{ "coremark", "compute" },
            .same_aot = true,
            .lifecycle_mode = "snapshot-replay",
        });

        var empty_env: std.process.Environ.Map = .init(allocator);
        const linux_spec = fix.config.object.get("targets").?.object.get("linux").?.object;
        const unikraft_spec = fix.config.object.get("targets").?.object.get("unikraft").?.object;

        var linux_rcfg: std.json.ObjectMap = .empty;
        for ([_][]const u8{
            "runtime_path", "compiler_path",   "compiler_version", "source",
            "target_abi",   "platform",        "options",          "image_path",
            "aot_paths",    "compile_profile", "memory_policy",    "execution_lifecycle",
        }) |k| {
            try linux_rcfg.put(allocator, k, linux_spec.get(k).?);
        }
        try linux_rcfg.put(allocator, "configured_vm_ram_bytes", .{ .integer = 536870912 });
        const linux_rcfg_path = try std.fs.path.join(allocator, &.{ sub, "cli-linux-rcfg.json" });
        const cli_linux_receipt = try std.fs.path.join(allocator, &.{ sub, "cli-linux-receipt.json" });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = linux_rcfg_path, .data = try encoded(allocator, .{ .object = linux_rcfg }) });
        try std.testing.expectEqual(
            @as(u8, 0),
            runCli(allocator, io, &empty_env, &.{ "native-linux-receipt", "--config", linux_rcfg_path, "--out", cli_linux_receipt }),
        );
        // Existing output file must be rejected
        try std.testing.expectEqual(
            @as(u8, 2),
            runCli(allocator, io, &empty_env, &.{ "native-linux-receipt", "--config", linux_rcfg_path, "--out", cli_linux_receipt }),
        );

        var unikraft_rcfg = try testCloneValue(allocator, .{ .object = linux_rcfg });
        try unikraft_rcfg.object.put(allocator, "os", .{ .string = "unikraft" });
        for ([_][]const u8{ "runtime_path", "image_path", "target_abi", "compile_profile", "memory_policy", "aot_paths" }) |k| {
            try unikraft_rcfg.object.put(allocator, k, unikraft_spec.get(k).?);
        }
        const unikraft_rcfg_path = try std.fs.path.join(allocator, &.{ sub, "cli-unikraft-rcfg.json" });
        const cli_unikraft_receipt = try std.fs.path.join(allocator, &.{ sub, "cli-unikraft-receipt.json" });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = unikraft_rcfg_path, .data = try encoded(allocator, unikraft_rcfg) });
        try std.testing.expectEqual(
            @as(u8, 0),
            runCli(allocator, io, &empty_env, &.{ "native-linux-receipt", "--config", unikraft_rcfg_path, "--out", cli_unikraft_receipt }),
        );

        // Generate ABI proofs via CLI: one using --config JSON, one using direct CLI flags
        const cm_aot = linux_spec.get("aot_paths").?.object.get("coremark").?.string;
        const comp_aot = linux_spec.get("aot_paths").?.object.get("compute").?.string;
        const src_obj = linux_spec.get("source").?.object;

        var cm_pcfg: std.json.ObjectMap = .empty;
        try cm_pcfg.put(allocator, "aot_path", .{ .string = cm_aot });
        try cm_pcfg.put(allocator, "source", linux_spec.get("source").?);
        var abis_arr: std.json.Array = .init(allocator);
        try abis_arr.append(linux_spec.get("target_abi").?);
        try abis_arr.append(unikraft_spec.get("target_abi").?);
        try cm_pcfg.put(allocator, "target_abis", .{ .array = abis_arr });
        const cm_pcfg_path = try std.fs.path.join(allocator, &.{ sub, "cli-cm-pcfg.json" });
        const cli_cm_proof = try std.fs.path.join(allocator, &.{ sub, "cli-cm-proof.json" });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = cm_pcfg_path, .data = try encoded(allocator, .{ .object = cm_pcfg }) });
        try std.testing.expectEqual(
            @as(u8, 0),
            runCli(allocator, io, &empty_env, &.{ "native-abi-proof", "--config", cm_pcfg_path, "--out", cli_cm_proof }),
        );

        const cli_comp_proof = try std.fs.path.join(allocator, &.{ sub, "cli-comp-proof.json" });
        try std.testing.expectEqual(
            @as(u8, 0),
            runCli(allocator, io, &empty_env, &.{
                "native-abi-proof",
                "--aot",
                comp_aot,
                "--source-repo",
                "cataggar/wamr",
                "--source-commit",
                src_obj.get("commit").?.string,
                "--source-tree-sha256",
                src_obj.get("tree_sha256").?.string,
                "--source-tracked-diff-sha256",
                src_obj.get("tracked_diff_sha256").?.string,
                "--linux-target-abi",
                linux_spec.get("target_abi").?.string,
                "--unikraft-target-abi",
                unikraft_spec.get("target_abi").?.string,
                "--out",
                cli_comp_proof,
            }),
        );

        // Round-trip both CLI-generated receipts and CLI-generated ABI proofs into native-plan
        var plan_cfg = try testCloneValue(allocator, fix.config);
        try plan_cfg.object.getPtr("targets").?.object.getPtr("linux").?.object.put(allocator, "image_receipt_path", .{ .string = cli_linux_receipt });
        try plan_cfg.object.getPtr("targets").?.object.getPtr("unikraft").?.object.put(allocator, "image_receipt_path", .{ .string = cli_unikraft_receipt });
        try plan_cfg.object.getPtr("abi_proofs").?.object.put(allocator, "coremark", .{ .string = cli_cm_proof });
        try plan_cfg.object.getPtr("abi_proofs").?.object.put(allocator, "compute", .{ .string = cli_comp_proof });
        const cli_plan_cfg_path = try std.fs.path.join(allocator, &.{ sub, "cli-plan-config.json" });
        const cli_plan_out = try std.fs.path.join(allocator, &.{ sub, "cli-plan-manifest.json" });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = cli_plan_cfg_path, .data = try encoded(allocator, plan_cfg) });
        try std.testing.expectEqual(
            @as(u8, 0),
            runCli(allocator, io, &empty_env, &.{ "native-plan", "--config", cli_plan_cfg_path, "--repo", ".", "--out", cli_plan_out }),
        );
        const parsed_plan = try readJson(allocator, io, cli_plan_out);
        try validateManifest(allocator, parsed_plan, false);

        // Negative validation tests for createAbiProof and createImageReceipt
        var bad_proof_cfg = try testCloneValue(allocator, .{ .object = cm_pcfg });
        try bad_proof_cfg.object.put(allocator, "aot_sha256", .{ .string = "0000000000000000000000000000000000000000000000000000000000000000" });
        try std.testing.expectError(error.ValidationFailed, createAbiProof(allocator, io, bad_proof_cfg, false));

        var bad_receipt_cfg = try testCloneValue(allocator, .{ .object = linux_rcfg });
        try bad_receipt_cfg.object.put(allocator, "compiler_embedded", .{ .bool = true });
        try std.testing.expectError(error.ValidationFailed, createImageReceipt(allocator, io, bad_receipt_cfg, false));
    }

    // 7. Full v2 producer-to-report chain with current_unikraft_native_sdk_commit
    {
        const sub = try std.fs.path.join(allocator, &.{ root, "t7_full_v2_chain" });
        try std.Io.Dir.cwd().createDirPath(io, sub);
        const fix = try makeSyntheticPlanFixture(allocator, io, sub, .{
            .profile = "ci",
            .warmups = 0,
            .runs = 1,
            .steady_invocations = 2,
            .workloads = &.{ "coremark", "compute" },
            .same_aot = true,
            .lifecycle_mode = "snapshot-replay",
            .commit = current_unikraft_native_sdk_commit,
            .allow_synthetic = false,
        });

        const linux_spec = fix.config.object.get("targets").?.object.get("linux").?.object;
        const unikraft_spec = fix.config.object.get("targets").?.object.get("unikraft").?.object;

        var proofs_map: std.json.ObjectMap = .empty;
        for ([_][]const u8{ "coremark", "compute" }) |wname| {
            var proof_cfg: std.json.ObjectMap = .empty;
            try proof_cfg.put(allocator, "aot_path", linux_spec.get("aot_paths").?.object.get(wname).?);
            try proof_cfg.put(allocator, "source", linux_spec.get("source").?);
            try proof_cfg.put(allocator, "linux_target_abi", linux_spec.get("target_abi").?);
            try proof_cfg.put(allocator, "unikraft_target_abi", unikraft_spec.get("target_abi").?);
            const proof_doc = try createAbiProof(allocator, io, .{ .object = proof_cfg }, false);
            const proof_path = try std.fs.path.join(allocator, &.{ sub, try std.fmt.allocPrint(allocator, "chain-{s}-proof.json", .{wname}) });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = proof_path, .data = try encoded(allocator, proof_doc) });
            try proofs_map.put(allocator, wname, .{ .string = proof_path });
        }

        var plan_cfg = try testCloneValue(allocator, fix.config);
        try plan_cfg.object.put(allocator, "abi_proofs", .{ .object = proofs_map });
        for ([_][]const u8{ "linux", "unikraft" }) |os_name| {
            const t_spec = if (std.mem.eql(u8, os_name, "linux")) linux_spec else unikraft_spec;
            var rcfg: std.json.ObjectMap = .empty;
            try rcfg.put(allocator, "os", .{ .string = os_name });
            for ([_][]const u8{
                "runtime_path", "compiler_path",   "compiler_version", "source",
                "target_abi",   "platform",        "options",          "image_path",
                "aot_paths",    "compile_profile", "memory_policy",    "execution_lifecycle",
            }) |k| {
                try rcfg.put(allocator, k, t_spec.get(k).?);
            }
            try rcfg.put(allocator, "configured_vm_ram_bytes", .{ .integer = 536870912 });
            const receipt_doc = try createImageReceipt(allocator, io, .{ .object = rcfg }, false);
            const receipt_path = try std.fs.path.join(allocator, &.{ sub, try std.fmt.allocPrint(allocator, "chain-{s}-receipt.json", .{os_name}) });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = receipt_path, .data = try encoded(allocator, receipt_doc) });
            try plan_cfg.object.getPtr("targets").?.object.getPtr(os_name).?.object.put(allocator, "image_receipt_path", .{ .string = receipt_path });
        }

        const now_ns: i128 = @intCast(std.Io.Clock.real.now(io).nanoseconds);
        const manifest = try createPlan(allocator, io, plan_cfg, sub, now_ns - 60 * @as(i128, std.time.ns_per_s), false);
        const created_ns = try parseUtcTimestampNs(manifest.object.get("created_at").?);
        const started = try formatUtcTimestamp(allocator, created_ns + std.time.ns_per_s);
        const completed = try formatUtcTimestamp(allocator, created_ns + 2 * std.time.ns_per_s);

        var capture_dirs: std.ArrayList([]const u8) = .empty;
        for (manifest.object.get("schedule").?.array.items) |entry| {
            const rid = entry.object.get("run_id").?.string;
            const req = try createRequest(allocator, manifest, rid, false);
            const res = try makeSyntheticResult(allocator, manifest, rid, .{});
            try std.testing.expectEqualStrings(
                req.object.get("config_sha256").?.string,
                res.object.get("config_sha256").?.string,
            );
            const raw_stdout = try std.fmt.allocPrint(allocator, "boot banner\n{s}{s}\n", .{ prefix, try encoded(allocator, res) });
            const cap_dir = try std.fs.path.join(allocator, &.{ sub, try std.fmt.allocPrint(allocator, "chain-capture-{s}", .{rid}) });
            const imported = try importCapture(allocator, io, manifest, rid, cap_dir, raw_stdout, "", .{
                .started_at = started,
                .completed_at = completed,
                .observation_seconds = 1.0,
                .control_plane_seconds = 0.25,
                .transport_outcome = "exited",
                .process_returncode = 0,
            }, false);
            try std.testing.expectEqualStrings("success", imported.object.get("result").?.object.get("outcome").?.string);
            try capture_dirs.append(allocator, cap_dir);
        }

        const report = try createReport(allocator, io, manifest, capture_dirs.items, false);
        try std.testing.expect(report.comparable == true);
        try std.testing.expectEqual(@as(i64, 2), report.schema_version);
        try std.testing.expectEqualStrings("370e8e83ac8298b6faa2b288714ffcc9b8798c4b", report.sdk_commit);
        try std.testing.expect(report.document.object.get("status").?.object.get("paired_measurement_complete").?.bool);
    }
}

fn checkProducer(
    allocator: Allocator,
    io: std.Io,
    prod_cmd: []const []const u8,
    fixture_dir: []const u8,
    workload: []const u8,
    repeats: usize,
) !Value {
    const cwasm_path = try std.fs.path.join(allocator, &.{ fixture_dir, try std.fmt.allocPrint(allocator, "{s}.cwasm", .{workload}) });
    var argv: std.ArrayList([]const u8) = .empty;
    for (prod_cmd) |p| try argv.append(allocator, p);
    try argv.appendSlice(allocator, &.{
        "--check",
        cwasm_path,
        "100",
        try std.fmt.allocPrint(allocator, "{d}", .{repeats}),
        workload,
    });
    const res = try std.process.run(allocator, io, .{ .argv = argv.items });
    try std.testing.expect(res.term == .exited and res.term.exited == 0);
    const check_prefix = "WAMR_NATIVE_CHECK_RESULT=";
    const trimmed_out = std.mem.trim(u8, res.stdout, " \t\r\n");
    try std.testing.expect(std.mem.indexOfScalar(u8, trimmed_out, '\n') == null);
    try std.testing.expect(std.mem.startsWith(u8, trimmed_out, check_prefix));
    try std.testing.expect(std.mem.indexOf(u8, res.stderr, "WAMR_NATIVE_ALLOCATOR stage=after_release method=caller-requested-bytes live=0 peak=") != null);

    const rec = try strictJson(allocator, trimmed_out[check_prefix.len..]);
    const obj = rec.object;
    try std.testing.expectEqualStrings("correctness-only-not-performance", obj.get("qualification").?.string);
    try std.testing.expectEqualStrings("same-instance-full-snapshot-reset", obj.get("reset_semantics").?.string);
    try std.testing.expectEqualStrings("success", obj.get("outcome").?.string);
    try std.testing.expectEqualStrings(try sha256FileAlloc(allocator, io, cwasm_path), obj.get("artifact_sha256").?.string);
    try std.testing.expectEqual(repeats, obj.get("invocations").?.array.items.len);
    try std.testing.expect(obj.get("clock_resolution_ns").?.integer > 0);
    try std.testing.expect(obj.get("phases").?.object.get("load_ticks").?.integer > 0);
    try std.testing.expect(obj.get("phases").?.object.get("instantiate_ticks").?.integer > 0);
    try std.testing.expect(obj.get("phases").?.object.get("lifecycle_setup_ticks").?.integer > 0);
    try std.testing.expectEqual(repeats - 1, obj.get("phases").?.object.get("steady_state_ticks").?.array.items.len);
    try std.testing.expectEqualStrings("snapshot-replay", obj.get("execution_lifecycle").?.object.get("mode").?.string);
    try std.testing.expectEqual(repeats - 1, obj.get("reset_events").?.array.items.len);
    for (obj.get("reset_events").?.array.items, 2..) |ev, ordinal| {
        try std.testing.expectEqual(@as(i64, @intCast(ordinal)), ev.object.get("before_invocation").?.integer);
        try std.testing.expectEqualStrings("completed", ev.object.get("outcome").?.string);
        try std.testing.expect(ev.object.get("elapsed_ticks").?.integer > 0);
    }
    for (obj.get("invocations").?.array.items) |inv| {
        try std.testing.expect(inv.object.get("stdout_complete").?.bool);
        try std.testing.expectEqual(@as(usize, 0), inv.object.get("measurement_errors").?.array.items.len);
    }
    return rec;
}

fn makeSyntheticEnvelope(allocator: Allocator, manifest: Value, run: Value, record: Value) !Value {
    const run_id = run.object.get("run_id").?.string;
    var env = try makeSyntheticResult(allocator, manifest, run_id, .{});
    const rec_obj = record.object;
    try env.object.put(allocator, "outcome", rec_obj.get("outcome").?);
    try env.object.put(allocator, "phases", rec_obj.get("phases").?);
    try env.object.put(allocator, "execution_lifecycle", rec_obj.get("execution_lifecycle").?);
    try env.object.put(allocator, "reset_events", rec_obj.get("reset_events").?);

    var clock: std.json.ObjectMap = .empty;
    try clock.put(allocator, "source", .{ .string = "linux-clock-monotonic" });
    try clock.put(allocator, "unit", .{ .string = "ns" });
    try clock.put(allocator, "ticks_per_second", .{ .integer = 1_000_000_000 });
    try clock.put(allocator, "resolution_ticks", rec_obj.get("clock_resolution_ns").?);
    try env.object.put(allocator, "clock", .{ .object = clock });

    var invs: std.json.Array = .init(allocator);
    for (rec_obj.get("invocations").?.array.items) |inv| {
        var inv_map: std.json.ObjectMap = .empty;
        for ([_][]const u8{ "phase", "outcome", "exit_code", "stdout_base64", "stdout_complete", "measurement_errors" }) |k| {
            try inv_map.put(allocator, k, inv.object.get(k).?);
        }
        try invs.append(.{ .object = inv_map });
    }
    try env.object.put(allocator, "invocations", .{ .array = invs });

    const resets = rec_obj.get("reset_events").?.array.items;
    if (resets.len > 0 and std.mem.eql(u8, resets[resets.len - 1].object.get("outcome").?.string, "error")) {
        try env.object.put(allocator, "memory", .null);
    } else {
        var mem = env.object.getPtr("memory").?;
        try mem.object.put(allocator, "samples", rec_obj.get("memory_samples").?);
        try mem.object.put(allocator, "method", .{ .string = "linux-mmap-retained-commit-high-water" });
        var covered: std.json.Array = .init(allocator);
        try covered.append(.{ .string = "native-code" });
        try covered.append(.{ .string = "wasm-linear-memory" });
        var omitted: std.json.Array = .init(allocator);
        try omitted.append(.{ .string = "allocator" });
        try omitted.append(.{ .string = "kernel" });
        try omitted.append(.{ .string = "process" });
        try mem.object.put(allocator, "covered_regions", .{ .array = covered });
        try mem.object.put(allocator, "omitted_regions", .{ .array = omitted });
        try mem.object.put(allocator, "coverage", .{ .string = "partial-guest" });
    }
    return env;
}

fn findLinuxRun(manifest: Value, workload: []const u8) !Value {
    for (manifest.object.get("schedule").?.array.items) |entry| {
        if (std.mem.eql(u8, entry.object.get("target").?.string, "linux") and
            std.mem.eql(u8, entry.object.get("workload").?.string, workload))
        {
            return entry;
        }
    }
    return error.ValidationFailed;
}

test "NativeLinuxProducerTests" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const producer_z = std.c.getenv("WAMR_NATIVE_PRODUCER") orelse return error.SkipZigTest;
    const fixture_dir_z = std.c.getenv("WAMR_NATIVE_FIXTURE_DIR") orelse return error.SkipZigTest;
    const producer = std.mem.span(producer_z);
    const fixture_dir = std.mem.span(fixture_dir_z);

    var prod_cmd_list: std.ArrayList([]const u8) = .empty;
    if (std.c.getenv("WAMR_NATIVE_PRODUCER_RUNNER")) |runner_z| {
        var tok = std.mem.tokenizeAny(u8, std.mem.span(runner_z), " \t");
        while (tok.next()) |part| try prod_cmd_list.append(allocator, part);
    }
    try prod_cmd_list.append(allocator, producer);
    const prod_cmd = prod_cmd_list.items;

    var base_env: std.process.Environ.Map = .init(allocator);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);

    // 1. Both real CoreMark variants CRC after snapshot reset
    for ([_][]const u8{ "coremark", "coremark-nofp" }) |wname| {
        const rec = try checkProducer(allocator, io, prod_cmd, fixture_dir, wname, 3);
        for (rec.object.get("invocations").?.array.items) |inv| {
            try std.testing.expect(inv.object.get("crc_valid").?.bool);
            try std.testing.expect(inv.object.get("exit_code").? == .null);
            const stdout = try decodeStdout(allocator, inv.object.get("stdout_base64").?);
            try std.testing.expect(std.mem.indexOf(u8, stdout, "[0]crcfinal      : 0x988c") != null);
            const summary = try validateCoremarkShortCrcOutput(allocator, stdout, wname);
            try std.testing.expectEqualStrings("correctness-only-not-performance", summary.object.get("qualification").?.string);
            try std.testing.expectEqualStrings(unikraft_coremark_correctness_profile, summary.object.get("profile").?.string);
            try std.testing.expectEqualStrings("988c", summary.object.get("crc").?.object.get("crcfinal").?.string);
            const elapsed_s: f64 = @as(f64, @floatFromInt(inv.object.get("elapsed_ns").?.integer)) / 1e9;
            try std.testing.expectError(error.ValidationFailed, coremarkCorrectness(allocator, stdout, elapsed_s, wname, 0.0));
        }
    }

    // 2. Closed four-workload fixtures and end-to-end replay report + compute/memory real phases
    {
        for (fixtures) |fix| {
            try std.testing.expectEqualStrings(fix.sha256, try sha256FileAlloc(allocator, io, fix.path));
            const cwasm = try std.fs.path.join(allocator, &.{ fixture_dir, try std.fmt.allocPrint(allocator, "{s}.cwasm", .{fix.name}) });
            const st = try std.Io.Dir.cwd().statFile(io, cwasm, .{});
            try std.testing.expect(st.kind == .file and st.size > 0);
        }
        const sub = try std.fs.path.join(allocator, &.{ root, "e2e" });
        try std.Io.Dir.cwd().createDirPath(io, sub);
        const harness = try makeSyntheticPlanFixture(allocator, io, sub, .{
            .workloads = &.{ "coremark", "coremark-nofp", "compute", "memory" },
            .lifecycle_mode = "snapshot-replay",
            .allow_synthetic = true,
        });

        var real_recs: std.StringHashMap(Value) = .init(allocator);
        for ([_][]const u8{ "compute", "memory", "coremark", "coremark-nofp" }) |wname| {
            try real_recs.put(wname, try checkProducer(allocator, io, prod_cmd, fixture_dir, wname, 3));
        }

        for ([_][]const u8{ "compute", "memory" }) |wname| {
            const run = try findLinuxRun(harness.manifest, wname);
            const env = try makeSyntheticEnvelope(allocator, harness.manifest, run, real_recs.get(wname).?);
            const checks = try validateResult(allocator, env, harness.manifest, run.object.get("run_id").?.string);
            try std.testing.expectEqual(@as(usize, 3), checks.items.len);
            for (checks.items) |chk| {
                try std.testing.expectEqualStrings("returned-without-trap", chk.object.get("self_check").?.string);
            }
        }

        var captures: std.ArrayList([]const u8) = .empty;
        const created_ns = try parseUtcTimestampNs(harness.manifest.object.get("created_at").?);
        const now_str = try formatUtcTimestamp(allocator, created_ns + std.time.ns_per_s);
        for (harness.manifest.object.get("schedule").?.array.items) |run| {
            const wname = run.object.get("workload").?.string;
            const rid = run.object.get("run_id").?.string;
            const out_dir = try std.fs.path.join(allocator, &.{ sub, try std.fmt.allocPrint(allocator, "e2e-{s}", .{rid}) });
            var env = try makeSyntheticEnvelope(allocator, harness.manifest, run, real_recs.get(wname).?);
            if (std.mem.eql(u8, wname, "compute") or std.mem.eql(u8, wname, "memory")) {
                if (std.mem.eql(u8, run.object.get("target").?.string, "unikraft")) {
                    var clk: std.json.ObjectMap = .empty;
                    try clk.put(allocator, "source", .{ .string = "unikraft-hyperv-reference-tsc" });
                    try clk.put(allocator, "unit", .{ .string = "ns" });
                    try clk.put(allocator, "ticks_per_second", .{ .integer = 1_000_000_000 });
                    try clk.put(allocator, "resolution_ticks", .{ .integer = 100 });
                    try env.object.put(allocator, "clock", .{ .object = clk });
                    try env.object.getPtr("memory").?.object.put(allocator, "method", .{ .string = "unikraft-ukvmem-page-frame-high-water" });
                }
            } else {
                try std.testing.expectError(error.ValidationFailed, validateResult(allocator, env, harness.manifest, rid));
                try env.object.put(allocator, "outcome", .{ .string = "error" });
            }
            const payload = try std.fmt.allocPrint(allocator, "{s}{s}\n", .{ prefix, try encoded(allocator, env) });
            try persistObservation(allocator, io, harness.manifest, rid, out_dir, payload, "", .{
                .started_at = now_str,
                .completed_at = now_str,
                .observation_seconds = 2.0,
                .control_plane_seconds = if (std.mem.eql(u8, run.object.get("target").?.string, "unikraft")) 0.5 else null,
            });
            try captures.append(allocator, out_dir);
        }
        const rep = try buildReport(allocator, io, harness.manifest, captures.items, true);
        try std.testing.expectEqualStrings("wamr-native-matched-comparison", rep.object.get("kind").?.string);
        try std.testing.expectEqualStrings("synthetic", rep.object.get("evidence_kind").?.string);
        try std.testing.expect(!rep.object.get("status").?.object.get("paired_measurement_complete").?.bool);
    }

    // 3. Real v2 failed calls, capture fail-closed, empty input, unqualified deployment, incomplete stdout
    {
        const sub = try std.fs.path.join(allocator, &.{ root, "faults" });
        try std.Io.Dir.cwd().createDirPath(io, sub);
        const harness = try makeSyntheticPlanFixture(allocator, io, sub, .{
            .workloads = &.{"coremark"},
            .lifecycle_mode = "snapshot-replay",
            .allow_synthetic = true,
        });
        const run = try findLinuxRun(harness.manifest, "coremark");
        const rid = run.object.get("run_id").?.string;
        const cwasm = try std.fs.path.join(allocator, &.{ fixture_dir, "coremark.cwasm" });

        for ([_][]const u8{
            "opening-clock",        "closing-clock",         "backwards-clock", "output-limit",
            "reset-clock",          "reset-backwards-clock", "reset-protect",   "steady-opening-clock",
            "steady-closing-clock", "steady-output-limit",
        }) |fault| {
            var argv: std.ArrayList([]const u8) = .empty;
            for (prod_cmd) |p| try argv.append(allocator, p);
            try argv.appendSlice(allocator, &.{ "--check", cwasm, "100", "3", "coremark", fault });
            const res = try std.process.run(allocator, io, .{ .argv = argv.items });
            try std.testing.expect(res.term != .exited or res.term.exited != 0);
            const check_prefix = "WAMR_NATIVE_CHECK_RESULT=";
            const trimmed = std.mem.trim(u8, res.stdout, " \t\r\n");
            try std.testing.expect(std.mem.startsWith(u8, trimmed, check_prefix));
            const rec = try strictJson(allocator, trimmed[check_prefix.len..]);
            const env = try makeSyntheticEnvelope(allocator, harness.manifest, run, rec);
            const checks = try validateResult(allocator, env, harness.manifest, rid);
            try std.testing.expectEqual(rec.object.get("invocations").?.array.items.len, checks.items.len);
            try std.testing.expectEqualStrings("error", env.object.get("outcome").?.string);
        }

        // Capture does not mislabel correctness run as measurement
        var det_cmd: std.ArrayList([]const u8) = .empty;
        for (prod_cmd) |p| try det_cmd.append(allocator, p);
        try det_cmd.appendSlice(allocator, &.{
            "--check",
            try std.fs.path.join(allocator, &.{ fixture_dir, "deterministic.cwasm" }),
            "100",
            "2",
            "deterministic",
        });
        const det_out = try std.fs.path.join(allocator, &.{ sub, "actual-correctness-capture" });
        try std.testing.expectError(
            error.ValidationFailed,
            capture(allocator, io, &base_env, harness.manifest, rid, det_cmd.items, det_out, 30.0, true),
        );

        // Unqualified deployment rejected without a result
        const dep_path = try std.fs.path.join(allocator, &.{ sub, "deployment.json" });
        try std.Io.Dir.cwd().writeFile(io, .{
            .sub_path = dep_path,
            .data = "{\"schema_version\":1,\"kind\":\"wamr-linux-producer-deployment\",\"qualification\":\"correctness-only\"}\n",
        });
        var dep_cmd: std.ArrayList([]const u8) = .empty;
        for (prod_cmd) |p| try dep_cmd.append(allocator, p);
        try dep_cmd.appendSlice(allocator, &.{ "--deployment", dep_path });
        const dep_res = try std.process.run(allocator, io, .{ .argv = dep_cmd.items });
        try std.testing.expect(dep_res.term != .exited or dep_res.term.exited != 0);
        try std.testing.expect(std.mem.indexOf(u8, dep_res.stderr, "InvalidConfiguration") != null);
        try std.testing.expect(std.mem.indexOf(u8, dep_res.stdout, prefix) == null);
    }

    // 4. Deterministic native-abi-proof and native-linux-receipt CLI round-trip with real producer & .cwasm fixtures
    {
        const sub = try std.fs.path.join(allocator, &.{ root, "receipt_proof_roundtrip" });
        try std.Io.Dir.cwd().createDirPath(io, sub);
        const fix = try makeSyntheticPlanFixture(allocator, io, sub, .{
            .workloads = &.{"coremark"},
            .lifecycle_mode = "snapshot-replay",
        });

        const cwasm = try std.fs.path.join(allocator, &.{ fixture_dir, "coremark.cwasm" });
        const linux_spec = fix.config.object.get("targets").?.object.get("linux").?.object;
        const unikraft_spec = fix.config.object.get("targets").?.object.get("unikraft").?.object;
        const src_obj = linux_spec.get("source").?.object;

        var aot_paths: std.json.ObjectMap = .empty;
        try aot_paths.put(allocator, "coremark", .{ .string = cwasm });

        var linux_rcfg: std.json.ObjectMap = .empty;
        for ([_][]const u8{
            "compiler_path",   "compiler_version", "source",
            "target_abi",      "platform",         "options",
            "compile_profile", "memory_policy",    "execution_lifecycle",
        }) |k| {
            try linux_rcfg.put(allocator, k, linux_spec.get(k).?);
        }
        try linux_rcfg.put(allocator, "image_path", .{ .string = producer });
        try linux_rcfg.put(allocator, "runtime_path", .{ .string = producer });
        try linux_rcfg.put(allocator, "aot_paths", .{ .object = aot_paths });
        try linux_rcfg.put(allocator, "configured_vm_ram_bytes", .{ .integer = 536870912 });

        const linux_rcfg_path = try std.fs.path.join(allocator, &.{ sub, "real-linux-rcfg.json" });
        const linux_receipt_out = try std.fs.path.join(allocator, &.{ sub, "real-linux-receipt.json" });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = linux_rcfg_path, .data = try encoded(allocator, .{ .object = linux_rcfg }) });
        try std.testing.expectEqual(
            @as(u8, 0),
            runCli(allocator, io, &base_env, &.{ "native-linux-receipt", "--config", linux_rcfg_path, "--out", linux_receipt_out }),
        );

        var unikraft_rcfg = try testCloneValue(allocator, .{ .object = linux_rcfg });
        try unikraft_rcfg.object.put(allocator, "os", .{ .string = "unikraft" });
        for ([_][]const u8{ "runtime_path", "image_path", "target_abi", "compile_profile", "memory_policy" }) |k| {
            try unikraft_rcfg.object.put(allocator, k, unikraft_spec.get(k).?);
        }
        const unikraft_rcfg_path = try std.fs.path.join(allocator, &.{ sub, "real-unikraft-rcfg.json" });
        const unikraft_receipt_out = try std.fs.path.join(allocator, &.{ sub, "real-unikraft-receipt.json" });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = unikraft_rcfg_path, .data = try encoded(allocator, unikraft_rcfg) });
        try std.testing.expectEqual(
            @as(u8, 0),
            runCli(allocator, io, &base_env, &.{ "native-linux-receipt", "--config", unikraft_rcfg_path, "--out", unikraft_receipt_out }),
        );

        const abi_proof_out = try std.fs.path.join(allocator, &.{ sub, "real-coremark-abi-proof.json" });
        try std.testing.expectEqual(
            @as(u8, 0),
            runCli(allocator, io, &base_env, &.{
                "native-abi-proof",
                "--aot",
                cwasm,
                "--source-repo",
                "cataggar/wamr",
                "--source-commit",
                src_obj.get("commit").?.string,
                "--source-tree-sha256",
                src_obj.get("tree_sha256").?.string,
                "--source-tracked-diff-sha256",
                src_obj.get("tracked_diff_sha256").?.string,
                "--linux-target-abi",
                linux_spec.get("target_abi").?.string,
                "--unikraft-target-abi",
                unikraft_spec.get("target_abi").?.string,
                "--out",
                abi_proof_out,
            }),
        );

        var plan_cfg = try testCloneValue(allocator, fix.config);
        var l_target = plan_cfg.object.getPtr("targets").?.object.getPtr("linux").?;
        try l_target.object.put(allocator, "image_path", .{ .string = producer });
        try l_target.object.put(allocator, "runtime_path", .{ .string = producer });
        try l_target.object.put(allocator, "aot_paths", .{ .object = aot_paths });
        try l_target.object.put(allocator, "image_receipt_path", .{ .string = linux_receipt_out });

        var u_target = plan_cfg.object.getPtr("targets").?.object.getPtr("unikraft").?;
        try u_target.object.put(allocator, "aot_paths", .{ .object = aot_paths });
        try u_target.object.put(allocator, "image_receipt_path", .{ .string = unikraft_receipt_out });

        try plan_cfg.object.getPtr("abi_proofs").?.object.put(allocator, "coremark", .{ .string = abi_proof_out });
        const plan_cfg_path = try std.fs.path.join(allocator, &.{ sub, "real-plan-config.json" });
        const plan_out = try std.fs.path.join(allocator, &.{ sub, "real-plan.json" });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = plan_cfg_path, .data = try encoded(allocator, plan_cfg) });
        try std.testing.expectEqual(
            @as(u8, 0),
            runCli(allocator, io, &base_env, &.{ "native-plan", "--config", plan_cfg_path, "--repo", ".", "--out", plan_out }),
        );
        const plan_doc = try readJson(allocator, io, plan_out);
        try validateManifest(allocator, plan_doc, false);
    }
}
