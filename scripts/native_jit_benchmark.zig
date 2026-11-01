//! Explicit matched JIT experiment and qualification validator.
const std = @import("std");
const builtin = @import("builtin");
const schema = @import("benchmark_schema.zig");
const process_control = @import("process_control.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const modes = [_][]const u8{ "aot", "fast", "full" };
pub const prefix = "WAMR_JIT_SAMPLE=";
pub const lifecycle = "same-instance-workload-initializes-memory-in-timed-call";
pub const max_record_bytes: usize = 8192;
pub const max_serial_bytes: usize = 16 * 1024 * 1024;
pub const max_receipt_bytes: usize = 128 * 1024;

const sample_keys = [_][]const u8{
    "schema_version",          "kind",                "qualification",
    "request_sha256",          "mode",                "compiler_embedded",
    "wasm_sha256",             "wasm_bytes",          "cwasm_sha256",
    "cwasm_bytes",             "workload",            "expected",
    "lifecycle",               "clock_resolution_ns", "compile_ns",
    "compiler_phases_ns",      "compiler_peak_bytes", "compiler_retained_bytes",
    "compiler_polls",          "load_ns",             "instantiate_ns",
    "start_ns",                "growth_ns",           "growth_previous_pages",
    "fuel_per_invocation",     "invocations",         "memory_before",
    "memory_after",            "caller_peak_bytes",   "caller_live_after_teardown",
    "reserved_after_teardown", "failure_stage",       "failure",
};

const memory_keys = [_][]const u8{
    "heap_live_bytes",     "heap_peak_bytes",       "code_bytes",
    "code_reserved_bytes", "linear_reserved_bytes", "linear_committed_bytes",
};

pub fn expected() u32 {
    var cells: [256]u32 = undefined;
    for (0..256) |i| cells[i] = @intCast(i * 17 + 3);
    var result: u32 = 0;
    for (0..2000) |i| {
        const index = i & 255;
        const step: u32 = @intCast(i);
        cells[index] = if (i & 1 == 0)
            cells[index] *% 3 +% step
        else
            cells[index] ^ (step *% 7);
        result +%= cells[index];
    }
    return result;
}

pub fn sha(data: []const u8) [64]u8 {
    var digest_bytes: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest_bytes, .{});
    return std.fmt.bytesToHex(digest_bytes, .lower);
}

pub fn shaAlloc(allocator: Allocator, data: []const u8) ![]u8 {
    const hex = sha(data);
    return allocator.dupe(u8, &hex);
}

pub fn sha256File(allocator: Allocator, io: std.Io, path: []const u8) ![]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(bytes);
    return shaAlloc(allocator, bytes);
}

pub fn encoded(allocator: Allocator, value: Value) ![]u8 {
    return schema.canonicalJson(allocator, value);
}

pub fn identical(allocator: Allocator, left: Value, right: Value) !bool {
    const l = try encoded(allocator, left);
    defer allocator.free(l);
    const r = try encoded(allocator, right);
    defer allocator.free(r);
    return std.mem.eql(u8, l, r);
}

pub fn strictJson(allocator: Allocator, text: []const u8) !Value {
    if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidJsonUtf8;
    const parsed = std.json.parseFromSliceLeaky(Value, allocator, text, .{
        .duplicate_field_behavior = .@"error",
        .allocate = .alloc_always,
    }) catch return error.InvalidStrictJson;
    try validateFiniteNumbers(parsed);
    return parsed;
}

fn validateFiniteNumbers(value: Value) !void {
    switch (value) {
        .float => |v| if (!std.math.isFinite(v)) return error.NonFiniteJsonNumber,
        .number_string => |s| {
            const parsed = std.fmt.parseFloat(f64, s) catch return error.InvalidStrictJson;
            if (!std.math.isFinite(parsed)) return error.NonFiniteJsonNumber;
        },
        .array => |arr| for (arr.items) |item| try validateFiniteNumbers(item),
        .object => |obj| {
            var it = obj.iterator();
            while (it.next()) |entry| try validateFiniteNumbers(entry.value_ptr.*);
        },
        else => {},
    }
}

pub fn readJson(allocator: Allocator, io: std.Io, path: []const u8) !Value {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    return strictJson(allocator, bytes);
}

pub fn require(ok: bool) !void {
    if (!ok) return error.ValidationFailed;
}

pub fn requireKeys(value: Value, expected_list: []const []const u8) !std.json.ObjectMap {
    if (value != .object) return error.ValidationFailed;
    const obj = value.object;
    if (obj.count() != expected_list.len) return error.ValidationFailed;
    for (expected_list) |key| {
        if (!obj.contains(key)) return error.ValidationFailed;
    }
    return obj;
}

pub fn requireDigest(value: Value, length: usize) ![]const u8 {
    if (value != .string) return error.ValidationFailed;
    const s = value.string;
    if (s.len != length) return error.ValidationFailed;
    for (s) |ch| {
        const ok = (ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f');
        if (!ok) return error.ValidationFailed;
    }
    return s;
}

pub fn requireDigestSlice(text: ?[]const u8, length: usize) ![]const u8 {
    const s = text orelse return error.ValidationFailed;
    if (s.len != length) return error.ValidationFailed;
    for (s) |ch| {
        const ok = (ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f');
        if (!ok) return error.ValidationFailed;
    }
    return s;
}

pub fn publicToken(value: Value) ![]const u8 {
    if (value != .string) return error.ValidationFailed;
    const s = value.string;
    if (s.len == 0 or s.len > 128) return error.ValidationFailed;
    if (!std.ascii.isAlphanumeric(s[0])) return error.ValidationFailed;
    for (s[1..]) |ch| {
        const ok = std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == ':' or ch == '+' or ch == '-';
        if (!ok) return error.ValidationFailed;
    }
    return s;
}

pub fn availableToken(value: Value) ![]const u8 {
    const s = try publicToken(value);
    if (std.ascii.eqlIgnoreCase(s, "unknown") or
        std.ascii.eqlIgnoreCase(s, "unspecified") or
        std.ascii.eqlIgnoreCase(s, "placeholder"))
    {
        return error.ValidationFailed;
    }
    return s;
}

pub fn requireInt(value: Value, minimum: u64) !u64 {
    switch (value) {
        .integer => |v| {
            if (v < 0) return error.ValidationFailed;
            const u: u64 = @intCast(v);
            if (u < minimum) return error.ValidationFailed;
            return u;
        },
        .number_string => |s| {
            const u = std.fmt.parseInt(u64, s, 10) catch return error.ValidationFailed;
            if (u < minimum) return error.ValidationFailed;
            return u;
        },
        else => return error.ValidationFailed,
    }
}

pub fn requireExactInt(value: Value, target: i64) !void {
    if (value != .integer or value.integer != target) return error.ValidationFailed;
}

pub fn requireExactString(value: Value, target: []const u8) !void {
    if (value != .string or !std.mem.eql(u8, value.string, target)) return error.ValidationFailed;
}

pub fn requireExactBool(value: Value, target: bool) !void {
    if (value != .bool or value.bool != target) return error.ValidationFailed;
}

pub fn validateUuid(value: Value) !void {
    if (value != .string) return error.ValidationFailed;
    const s = value.string;
    if (s.len != 36) return error.ValidationFailed;
    for (s, 0..) |ch, idx| {
        if (idx == 8 or idx == 13 or idx == 18 or idx == 23) {
            if (ch != '-') return error.ValidationFailed;
        } else {
            const ok = (ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f');
            if (!ok) return error.ValidationFailed;
        }
    }
}

pub fn generateUuid(allocator: Allocator, io: std.Io) ![]u8 {
    var bytes: [16]u8 = undefined;
    io.random(&bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = std.fmt.bytesToHex(bytes, .lower);
    return std.fmt.allocPrint(allocator, "{s}-{s}-{s}-{s}-{s}", .{
        hex[0..8],
        hex[8..12],
        hex[12..16],
        hex[16..20],
        hex[20..32],
    });
}

fn isLeapYear(year: i64) bool {
    return (@mod(year, 4) == 0 and @mod(year, 100) != 0) or (@mod(year, 400) == 0);
}

fn daysInMonth(year: i64, month: u8) u8 {
    return switch (month) {
        1, 3, 5, 7, 8, 10, 12 => 31,
        4, 6, 9, 11 => 30,
        2 => if (isLeapYear(year)) 29 else 28,
        else => 0,
    };
}

fn daysFromCivil(year: i64, month: u8, day: u8) i64 {
    const y = year - @as(i64, if (month <= 2) 1 else 0);
    const era = @divFloor(if (y >= 0) y else y - 399, 400);
    const yoe = y - era * 400;
    const m: i64 = month;
    const doy = @divFloor(153 * (m + (if (m > 2) @as(i64, -3) else 9)) + 2, 5) + day - 1;
    const doe = yoe * 365 + @divFloor(yoe, 4) - @divFloor(yoe, 100) + doy;
    return era * 146097 + doe - 719468;
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
    const month = std.fmt.parseInt(u8, s[5..7], 10) catch return error.ValidationFailed;
    const day = std.fmt.parseInt(u8, s[8..10], 10) catch return error.ValidationFailed;
    const hour = std.fmt.parseInt(u8, s[11..13], 10) catch return error.ValidationFailed;
    const minute = std.fmt.parseInt(u8, s[14..16], 10) catch return error.ValidationFailed;
    const second = std.fmt.parseInt(u8, s[17..19], 10) catch return error.ValidationFailed;
    if (month < 1 or month > 12) return error.ValidationFailed;
    const max_day = daysInMonth(year, month);
    if (day < 1 or day > max_day) return error.ValidationFailed;
    if (hour > 23 or minute > 59 or second > 59) return error.ValidationFailed;

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

pub fn createPrivateDir(io: std.Io, path: []const u8) !void {
    if (std.fs.path.dirname(path)) |parent| {
        if (parent.len != 0) try std.Io.Dir.cwd().createDirPath(io, parent);
    }
    try std.Io.Dir.cwd().createDir(io, path, if (builtin.os.tag == .windows) .default_dir else .fromMode(0o700));
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

pub fn artifactValue(allocator: Allocator, sha_hex: []const u8, size: u64) !Value {
    var obj: std.json.ObjectMap = .empty;
    try obj.put(allocator, "sha256", .{ .string = try allocator.dupe(u8, sha_hex) });
    try obj.put(allocator, "bytes", .{ .integer = @intCast(size) });
    return .{ .object = obj };
}

pub fn artifact(allocator: Allocator, io: std.Io, path: []const u8) !Value {
    const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch return error.ValidationFailed;
    if (stat.kind != .file or stat.size == 0) return error.ValidationFailed;
    const digest_str = try sha256File(allocator, io, path);
    return artifactValue(allocator, digest_str, stat.size);
}

pub fn streamArtifact(allocator: Allocator, io: std.Io, path: []const u8) !Value {
    const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch return error.ValidationFailed;
    if (stat.kind != .file) return error.ValidationFailed;
    const digest_str = try sha256File(allocator, io, path);
    return artifactValue(allocator, digest_str, stat.size);
}

pub fn validateArtifact(value: Value) !void {
    const obj = try requireKeys(value, &.{ "sha256", "bytes" });
    _ = try requireDigest(obj.get("sha256").?, 64);
    _ = try requireInt(obj.get("bytes").?, 1);
}

pub fn validatePlatform(value: Value) !void {
    const obj = try requireKeys(value, &.{ "arch", "cpu_model", "active_cpu_count", "azure_sku", "azure_region" });
    _ = try requireInt(obj.get("active_cpu_count").?, 1);
    for ([_][]const u8{ "arch", "cpu_model", "azure_sku", "azure_region" }) |name| {
        _ = try availableToken(obj.get(name).?);
    }
    const arch = obj.get("arch").?.string;
    try require(std.mem.eql(u8, arch, "x86_64") or std.mem.eql(u8, arch, "aarch64"));
}

fn validateSafety(value: Value) !void {
    const obj = try requireKeys(value, &.{ "bounds_checks", "checked_imports", "checked_traps", "wx" });
    for ([_][]const u8{ "bounds_checks", "checked_imports", "checked_traps", "wx" }) |name| {
        try requireExactBool(obj.get(name).?, true);
    }
}

fn validateCompilerEmbedded(value: Value) !void {
    const obj = try requireKeys(value, &.{ "aot", "jit" });
    try requireExactBool(obj.get("aot").?, false);
    try requireExactBool(obj.get("jit").?, true);
}

fn validateArtifactPair(
    allocator: Allocator,
    receipt_field: Value,
    actual: Value,
) !struct { aot_sha: []const u8, jit_sha: []const u8 } {
    const rec_obj = try requireKeys(receipt_field, &.{ "aot", "jit" });
    const act_obj = try requireKeys(actual, &.{ "aot", "jit" });
    try validateArtifact(rec_obj.get("aot").?);
    try validateArtifact(rec_obj.get("jit").?);
    try validateArtifact(act_obj.get("aot").?);
    try validateArtifact(act_obj.get("jit").?);
    try require(try identical(allocator, receipt_field, actual));
    try require(!try identical(allocator, act_obj.get("aot").?, act_obj.get("jit").?));
    return .{
        .aot_sha = act_obj.get("aot").?.object.get("sha256").?.string,
        .jit_sha = act_obj.get("jit").?.object.get("sha256").?.string,
    };
}

pub fn validateReceipt(
    allocator: Allocator,
    receipt: Value,
    executables: Value,
    images: Value,
) !void {
    const obj = try requireKeys(receipt, &.{
        "schema_version",    "kind",                      "issuer",
        "source_commit",     "deployment_receipt_sha256", "os",
        "arch",              "hardware_execution",        "platform",
        "safety",            "images",                    "executables",
        "compiler_embedded", "runtime_linkage",           "lifecycle",
        "allocator",         "page_policy",               "wasm",
        "aot_module",
    });
    try requireExactInt(obj.get("schema_version").?, 1);
    try requireExactString(obj.get("kind").?, "wamr-jit-independent-image-deployment-receipt");
    _ = try publicToken(obj.get("issuer").?);
    _ = try requireDigest(obj.get("source_commit").?, 40);
    _ = try requireDigest(obj.get("deployment_receipt_sha256").?, 64);
    try requireExactString(obj.get("os").?, "linux");
    try requireExactString(obj.get("arch").?, "x86_64");
    try requireExactBool(obj.get("hardware_execution").?, true);
    try validatePlatform(obj.get("platform").?);
    try requireExactString(obj.get("platform").?.object.get("arch").?, "x86_64");
    try validateSafety(obj.get("safety").?);
    try validateCompilerEmbedded(obj.get("compiler_embedded").?);
    try requireExactString(obj.get("runtime_linkage").?, "static");
    try requireExactString(obj.get("lifecycle").?, lifecycle);
    try validateArtifact(obj.get("wasm").?);
    try validateArtifact(obj.get("aot_module").?);
    _ = try availableToken(obj.get("allocator").?);
    _ = try availableToken(obj.get("page_policy").?);
    const exe_shas = try validateArtifactPair(allocator, obj.get("executables").?, executables);
    const img_shas = try validateArtifactPair(allocator, obj.get("images").?, images);
    try require(!std.mem.eql(u8, img_shas.aot_sha, exe_shas.aot_sha));
    try require(!std.mem.eql(u8, img_shas.jit_sha, exe_shas.jit_sha));
}

pub fn validateSample(
    sample: Value,
    mode: []const u8,
    request_sha: []const u8,
) !Value {
    const obj = try requireKeys(sample, &sample_keys);
    try requireExactInt(obj.get("schema_version").?, 1);
    try requireExactString(obj.get("kind").?, "wamr-native-jit-sample");
    try requireExactString(obj.get("qualification").?, "requires-independent-image-and-deployment-evidence");
    try requireExactString(obj.get("request_sha256").?, request_sha);
    try requireExactString(obj.get("mode").?, mode);
    const is_jit = !std.mem.eql(u8, mode, "aot");
    try requireExactBool(obj.get("compiler_embedded").?, is_jit);
    try require(obj.get("failure").? == .null and obj.get("failure_stage").? == .null);
    try requireExactString(obj.get("workload").?, "volatile-compute-memory-2000");
    try requireExactInt(obj.get("expected").?, @intCast(expected()));
    try requireExactString(obj.get("lifecycle").?, lifecycle);
    _ = try requireDigest(obj.get("wasm_sha256").?, 64);
    _ = try requireDigest(obj.get("cwasm_sha256").?, 64);
    for ([_][]const u8{ "wasm_bytes", "cwasm_bytes", "clock_resolution_ns", "caller_peak_bytes" }) |field| {
        _ = try requireInt(obj.get(field).?, 1);
    }
    for ([_][]const u8{ "load_ns", "instantiate_ns", "start_ns", "growth_ns" }) |field| {
        _ = try requireInt(obj.get(field).?, 0);
    }
    try requireExactInt(obj.get("growth_previous_pages").?, 2);
    try require(try requireInt(obj.get("caller_live_after_teardown").?, 0) == 0);
    try require(try requireInt(obj.get("reserved_after_teardown").?, 0) == 0);

    const invs = obj.get("invocations").?;
    try require(invs == .array and invs.array.items.len == 4);
    for (invs.array.items) |call| {
        const call_obj = try requireKeys(call, &.{ "ns", "outcome", "value", "diagnostic" });
        _ = try requireInt(call_obj.get("ns").?, 0);
        try requireExactString(call_obj.get("outcome").?, "returned");
        try requireExactInt(call_obj.get("value").?, @intCast(expected()));
        try require(call_obj.get("diagnostic").? == .null);
    }

    for ([_][]const u8{ "memory_before", "memory_after" }) |stage| {
        const mem_obj = try requireKeys(obj.get(stage).?, &memory_keys);
        for (memory_keys) |field| {
            _ = try requireInt(mem_obj.get(field).?, 1);
        }
        const live = try requireInt(mem_obj.get("heap_live_bytes").?, 1);
        const peak = try requireInt(mem_obj.get("heap_peak_bytes").?, 1);
        const code = try requireInt(mem_obj.get("code_bytes").?, 1);
        const code_res = try requireInt(mem_obj.get("code_reserved_bytes").?, 1);
        const lin_res = try requireInt(mem_obj.get("linear_reserved_bytes").?, 1);
        try require(live <= peak);
        try require(peak <= 16 * 1024 * 1024);
        try require(code <= 4 * 1024 * 1024);
        try require(code_res + lin_res <= 8 * 1024 * 1024);
    }

    const before = obj.get("memory_before").?.object;
    const after = obj.get("memory_after").?.object;
    try require(try requireInt(before.get("linear_committed_bytes").?, 1) == 2 * 65536);
    try require(try requireInt(after.get("linear_committed_bytes").?, 1) == 3 * 65536);
    try require(try requireInt(before.get("linear_reserved_bytes").?, 1) == 8 * 65536);
    try require(try requireInt(after.get("linear_reserved_bytes").?, 1) == 8 * 65536);

    if (!is_jit) {
        try require(obj.get("compile_ns").? == .null);
        try require(obj.get("compiler_phases_ns").? == .null);
        try require(obj.get("fuel_per_invocation").? == .null);
        try require(try requireInt(obj.get("compiler_peak_bytes").?, 0) == 0);
        try require(try requireInt(obj.get("compiler_retained_bytes").?, 0) == 0);
        try require(try requireInt(obj.get("compiler_polls").?, 0) == 0);
    } else {
        const compile_ns = try requireInt(obj.get("compile_ns").?, 0);
        const phases = try requireKeys(obj.get("compiler_phases_ns").?, &.{ "parse", "lower", "optimize", "codegen", "emit" });
        var phase_sum: u64 = 0;
        for ([_][]const u8{ "parse", "lower", "optimize", "codegen", "emit" }) |phase| {
            phase_sum += try requireInt(phases.get(phase).?, 0);
        }
        try require(phase_sum <= compile_ns);
        const peak = try requireInt(obj.get("compiler_peak_bytes").?, 1);
        const retained = try requireInt(obj.get("compiler_retained_bytes").?, 1);
        const polls = try requireInt(obj.get("compiler_polls").?, 1);
        try require(retained <= peak and peak <= 64 * 1024 * 1024);
        try require(polls <= 100000);
        try requireExactInt(obj.get("fuel_per_invocation").?, 100000);
    }
    return sample;
}

pub fn report(
    allocator: Allocator,
    io: std.Io,
    directory: []const u8,
    trusted_receipt_sha256: ?[]const u8,
) !Value {
    const req_path = try std.fs.path.join(allocator, &.{ directory, "request.json" });
    const req_bytes = try std.Io.Dir.cwd().readFileAlloc(io, req_path, allocator, .unlimited);
    const req = try strictJson(allocator, req_bytes);
    const req_obj = try requireKeys(req, &.{
        "schema_version", "kind",    "evidence_kind",  "nonce",
        "executables",    "receipt", "receipt_sha256", "images",
        "runner",
    });
    try requireExactInt(req_obj.get("schema_version").?, 1);
    try requireExactString(req_obj.get("kind").?, "wamr-jit-comparison-request");
    try validateUuid(req_obj.get("nonce").?);
    const ev_kind_val = req_obj.get("evidence_kind").?;
    try require(ev_kind_val == .string and
        (std.mem.eql(u8, ev_kind_val.string, "correctness-only") or
            std.mem.eql(u8, ev_kind_val.string, "measurement")));
    const exes = try requireKeys(req_obj.get("executables").?, &.{ "aot", "jit" });
    try validateArtifact(exes.get("aot").?);
    try validateArtifact(exes.get("jit").?);
    try require(!try identical(allocator, exes.get("aot").?, exes.get("jit").?));

    if (std.mem.eql(u8, ev_kind_val.string, "measurement")) {
        const runner_val = req_obj.get("runner").?;
        try require(runner_val == .array and runner_val.array.items.len == 0);
        const trusted = try requireDigestSlice(trusted_receipt_sha256, 64);
        try requireExactString(req_obj.get("receipt_sha256").?, trusted);
        const rec_bytes = try encoded(allocator, req_obj.get("receipt").?);
        const rec_sha = sha(rec_bytes);
        try require(std.mem.eql(u8, &rec_sha, trusted));
        try validateReceipt(allocator, req_obj.get("receipt").?, req_obj.get("executables").?, req_obj.get("images").?);
    } else {
        try require(req_obj.get("receipt").? == .null);
        try require(req_obj.get("receipt_sha256").? == .null);
        try require(req_obj.get("images").? == .null);
    }

    const req_sha = try shaAlloc(allocator, req_bytes);
    var samples_map: std.json.ObjectMap = .empty;
    for (modes) |mode| {
        const status_path = try std.fs.path.join(allocator, &.{ directory, try std.fmt.allocPrint(allocator, "{s}.status.json", .{mode}) });
        const status = try readJson(allocator, io, status_path);
        const status_obj = try requireKeys(status, &.{ "returncode", "error", "stdout", "stderr" });
        for ([_][]const u8{ "stdout", "stderr" }) |stream| {
            const stream_path = try std.fs.path.join(allocator, &.{ directory, try std.fmt.allocPrint(allocator, "{s}.{s}", .{ mode, stream }) });
            const actual_stream = try streamArtifact(allocator, io, stream_path);
            try require(try identical(allocator, status_obj.get(stream).?, actual_stream));
        }
        try requireExactInt(status_obj.get("returncode").?, 0);
        try require(status_obj.get("error").? == .null);

        const stdout_path = try std.fs.path.join(allocator, &.{ directory, try std.fmt.allocPrint(allocator, "{s}.stdout", .{mode}) });
        const stdout_bytes = try std.Io.Dir.cwd().readFileAlloc(io, stdout_path, allocator, .unlimited);
        const trimmed = std.mem.trimEnd(u8, stdout_bytes, "\r\n");
        try require(std.mem.indexOfScalar(u8, trimmed, '\n') == null);
        try require(std.mem.startsWith(u8, trimmed, prefix));
        const sample = try validateSample(try strictJson(allocator, trimmed[prefix.len..]), mode, req_sha);
        try samples_map.put(allocator, mode, sample);
    }

    const aot_sample = samples_map.get("aot").?.object;
    for (modes[1..]) |mode| {
        const s = samples_map.get(mode).?.object;
        try require(std.mem.eql(u8, aot_sample.get("wasm_sha256").?.string, s.get("wasm_sha256").?.string));
        try require(aot_sample.get("wasm_bytes").?.integer == s.get("wasm_bytes").?.integer);
    }

    if (req_obj.get("receipt").? != .null) {
        const rec_obj = req_obj.get("receipt").?.object;
        const expected_wasm = try artifactValue(
            allocator,
            aot_sample.get("wasm_sha256").?.string,
            @intCast(aot_sample.get("wasm_bytes").?.integer),
        );
        const expected_cwasm = try artifactValue(
            allocator,
            aot_sample.get("cwasm_sha256").?.string,
            @intCast(aot_sample.get("cwasm_bytes").?.integer),
        );
        try require(try identical(allocator, rec_obj.get("wasm").?, expected_wasm));
        try require(try identical(allocator, rec_obj.get("aot_module").?, expected_cwasm));
    }

    var peak_growth: std.json.ObjectMap = .empty;
    const aot_peak = aot_sample.get("caller_peak_bytes").?.integer;
    for (modes[1..]) |mode| {
        const mode_peak = samples_map.get(mode).?.object.get("caller_peak_bytes").?.integer;
        try peak_growth.put(allocator, mode, .{ .integer = mode_peak - aot_peak });
    }

    var out: std.json.ObjectMap = .empty;
    try out.put(allocator, "schema_version", .{ .integer = 1 });
    try out.put(allocator, "kind", .{ .string = "wamr-jit-comparison" });
    try out.put(allocator, "request_sha256", .{ .string = req_sha });
    try out.put(allocator, "evidence_kind", ev_kind_val);
    try out.put(allocator, "samples", .{ .object = samples_map });
    try out.put(allocator, "executable_bytes_growth", .{
        .integer = exes.get("jit").?.object.get("bytes").?.integer - exes.get("aot").?.object.get("bytes").?.integer,
    });
    if (req_obj.get("images").? == .null) {
        try out.put(allocator, "complete_image_bytes_growth", .null);
    } else {
        const imgs = req_obj.get("images").?.object;
        try out.put(allocator, "complete_image_bytes_growth", .{
            .integer = imgs.get("jit").?.object.get("bytes").?.integer - imgs.get("aot").?.object.get("bytes").?.integer,
        });
    }
    try out.put(allocator, "caller_requested_peak_growth", .{ .object = peak_growth });
    try out.put(allocator, "memory_scope", .{ .string = "requested-allocator-bytes-and-logical-page-commitment-not-RSS" });
    try out.put(allocator, "qualification", .{
        .string = if (std.mem.eql(u8, ev_kind_val.string, "correctness-only"))
            "correctness-only-no-performance-claim"
        else
            "bound-to-independently-supplied-attestation-not-hardware-proof",
    });
    return .{ .object = out };
}

pub const CaptureOptions = struct {
    runner: []const []const u8 = &.{},
    timeout_s: u64 = 120,
    receipt: ?Value = null,
    trusted_receipt_sha256: ?[]const u8 = null,
    images: ?Value = null,
    host_os: std.Target.Os.Tag = builtin.os.tag,
    host_arch: std.Target.Cpu.Arch = builtin.cpu.arch,
};

pub fn capture(
    allocator: Allocator,
    io: std.Io,
    aot: []const u8,
    jit: []const u8,
    output: []const u8,
    opts: CaptureOptions,
) !Value {
    try require(opts.timeout_s >= 1 and opts.timeout_s <= 600);
    var executables_map: std.json.ObjectMap = .empty;
    try executables_map.put(allocator, "aot", try artifact(allocator, io, aot));
    try executables_map.put(allocator, "jit", try artifact(allocator, io, jit));
    const executables: Value = .{ .object = executables_map };
    try require(!try identical(allocator, executables_map.get("aot").?, executables_map.get("jit").?));

    var receipt_hash: ?[]const u8 = null;
    if (opts.receipt) |rec| {
        try require(opts.runner.len == 0 and opts.host_os == .linux and opts.host_arch == .x86_64);
        const trusted = try requireDigestSlice(opts.trusted_receipt_sha256, 64);
        const rec_sha = try shaAlloc(allocator, try encoded(allocator, rec));
        try require(std.mem.eql(u8, rec_sha, trusted));
        receipt_hash = rec_sha;
        try validateReceipt(allocator, rec, executables, opts.images orelse return error.ValidationFailed);
    } else {
        try require(opts.images == null and opts.trusted_receipt_sha256 == null);
    }

    var runner_arr: std.json.Array = .init(allocator);
    for (opts.runner) |item| try runner_arr.append(.{ .string = item });

    var req_map: std.json.ObjectMap = .empty;
    try req_map.put(allocator, "schema_version", .{ .integer = 1 });
    try req_map.put(allocator, "kind", .{ .string = "wamr-jit-comparison-request" });
    try req_map.put(allocator, "evidence_kind", .{ .string = if (opts.receipt != null) "measurement" else "correctness-only" });
    try req_map.put(allocator, "nonce", .{ .string = try generateUuid(allocator, io) });
    try req_map.put(allocator, "executables", executables);
    try req_map.put(allocator, "receipt", opts.receipt orelse .null);
    try req_map.put(allocator, "receipt_sha256", if (receipt_hash) |h| .{ .string = h } else .null);
    try req_map.put(allocator, "images", opts.images orelse .null);
    try req_map.put(allocator, "runner", .{ .array = runner_arr });

    if (std.fs.path.dirname(output)) |parent| {
        if (parent.len != 0) try std.Io.Dir.cwd().createDirPath(io, parent);
    }
    try std.Io.Dir.cwd().createDir(io, output, .default_dir);

    const req_bytes = try encoded(allocator, .{ .object = req_map });
    const req_sha = try shaAlloc(allocator, req_bytes);
    const req_path = try std.fs.path.join(allocator, &.{ output, "request.json" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = req_path, .data = req_bytes });

    for (modes) |mode| {
        const binary = if (std.mem.eql(u8, mode, "aot")) aot else jit;
        const resolved = try std.Io.Dir.cwd().realPathFileAlloc(io, binary, allocator);
        var argv: std.ArrayList([]const u8) = .empty;
        for (opts.runner) |r| try argv.append(allocator, r);
        try argv.appendSlice(allocator, &.{ resolved, mode, req_sha });

        const stdout_path = try std.fs.path.join(allocator, &.{ output, try std.fmt.allocPrint(allocator, "{s}.stdout", .{mode}) });
        const stderr_path = try std.fs.path.join(allocator, &.{ output, try std.fmt.allocPrint(allocator, "{s}.stderr", .{mode}) });
        var returncode: Value = .null;
        var err_val: Value = .null;

        if (process_control.runTimed(allocator, io, .{ .argv = argv.items }, @intCast(opts.timeout_s))) |res| {
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = stdout_path, .data = res.stdout });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = stderr_path, .data = res.stderr });
            returncode = .{ .integer = switch (res.term) {
                .exited => |code| @intCast(code),
                .signal => |sig| -@as(i64, @intCast(@backingInt(sig))),
                else => 1,
            } };
        } else |err| {
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = stdout_path, .data = "" });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = stderr_path, .data = "" });
            err_val = .{ .string = if (err == error.Timeout)
                "TimeoutExpired: process timed out"
            else
                try std.fmt.allocPrint(allocator, "OSError: {s}", .{@errorName(err)}) };
        }

        var status_map: std.json.ObjectMap = .empty;
        try status_map.put(allocator, "returncode", returncode);
        try status_map.put(allocator, "error", err_val);
        try status_map.put(allocator, "stdout", try streamArtifact(allocator, io, stdout_path));
        try status_map.put(allocator, "stderr", try streamArtifact(allocator, io, stderr_path));
        const status_path = try std.fs.path.join(allocator, &.{ output, try std.fmt.allocPrint(allocator, "{s}.status.json", .{mode}) });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = status_path, .data = try encoded(allocator, .{ .object = status_map }) });
    }

    var after_exes: std.json.ObjectMap = .empty;
    try after_exes.put(allocator, "aot", try artifact(allocator, io, aot));
    try after_exes.put(allocator, "jit", try artifact(allocator, io, jit));
    try require(try identical(allocator, executables, .{ .object = after_exes }));

    const result = try report(allocator, io, output, opts.trusted_receipt_sha256);
    const comp_path = try std.fs.path.join(allocator, &.{ output, "comparison.json" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = comp_path, .data = try encoded(allocator, result) });
    return result;
}

fn validateNativeTarget(value: Value) !void {
    const obj = try requireKeys(value, &.{ "triple", "abi", "contract_version", "aot_profile", "jit_profile" });
    try requireExactString(obj.get("triple").?, "x86_64-freestanding-none");
    try requireExactString(obj.get("abi").?, "sysv");
    try requireExactInt(obj.get("contract_version").?, 1);
    try requireExactString(obj.get("aot_profile").?, "0x554b0001");
    try requireExactString(obj.get("jit_profile").?, "0x554b0002");
}

fn validateNativeOptions(value: Value) !void {
    const obj = try requireKeys(value, &.{ "presets", "verify", "compiler", "runtime", "rounds", "invocations", "growth_pages" });
    const presets = obj.get("presets").?;
    try require(presets == .array and presets.array.items.len == 2);
    try requireExactString(presets.array.items[0], "fast");
    try requireExactString(presets.array.items[1], "full");
    try requireExactString(obj.get("verify").?, "after_each_pass");
    try requireExactInt(obj.get("rounds").?, 2000);
    try requireExactInt(obj.get("invocations").?, 4);
    try requireExactInt(obj.get("growth_pages").?, 1);

    const comp = try requireKeys(obj.get("compiler").?, &.{
        "max_input_bytes", "max_compiler_bytes",      "max_code_bytes",
        "max_polls",       "max_functions",           "max_function_bytes",
        "max_locals",      "max_blocks_per_function", "max_instructions_per_function",
    });
    try requireExactInt(comp.get("max_input_bytes").?, 1024 * 1024);
    try requireExactInt(comp.get("max_compiler_bytes").?, 64 * 1024 * 1024);
    try requireExactInt(comp.get("max_code_bytes").?, 4 * 1024 * 1024);
    try requireExactInt(comp.get("max_polls").?, 100000);
    try requireExactInt(comp.get("max_functions").?, 1024);
    try requireExactInt(comp.get("max_function_bytes").?, 65536);
    try requireExactInt(comp.get("max_locals").?, 4096);
    try requireExactInt(comp.get("max_blocks_per_function").?, 2048);
    try requireExactInt(comp.get("max_instructions_per_function").?, 32768);

    const rt = try requireKeys(obj.get("runtime").?, &.{
        "max_heap_bytes",          "max_reserved_bytes", "max_code_bytes",
        "max_memory_pages",        "max_table_elements", "jit_fuel_per_invocation",
        "aot_fuel_per_invocation",
    });
    try requireExactInt(rt.get("max_heap_bytes").?, 16 * 1024 * 1024);
    try requireExactInt(rt.get("max_reserved_bytes").?, 8 * 1024 * 1024);
    try requireExactInt(rt.get("max_code_bytes").?, 4 * 1024 * 1024);
    try requireExactInt(rt.get("max_memory_pages").?, 8);
    try requireExactInt(rt.get("max_table_elements").?, 16);
    try requireExactInt(rt.get("jit_fuel_per_invocation").?, 100000);
    try require(rt.get("aot_fuel_per_invocation").? == .null);
}

fn validateNativeBuildOptions(value: Value) !void {
    const obj = try requireKeys(value, &.{
        "pic",           "single_threaded", "red_zone",
        "stack_check",   "stack_protector", "unwind_tables",
        "error_tracing", "link_libc",       "optimize",
    });
    try requireExactBool(obj.get("pic").?, true);
    try requireExactBool(obj.get("single_threaded").?, true);
    try requireExactBool(obj.get("red_zone").?, false);
    try requireExactBool(obj.get("stack_check").?, false);
    try requireExactBool(obj.get("stack_protector").?, false);
    try requireExactString(obj.get("unwind_tables").?, "none");
    try requireExactBool(obj.get("error_tracing").?, false);
    try requireExactBool(obj.get("link_libc").?, false);
    const opt = obj.get("optimize").?;
    try require(opt == .string and
        (std.mem.eql(u8, opt.string, "Debug") or
            std.mem.eql(u8, opt.string, "ReleaseSafe") or
            std.mem.eql(u8, opt.string, "ReleaseFast") or
            std.mem.eql(u8, opt.string, "ReleaseSmall")));
}

pub fn validateNativeReceipt(
    allocator: Allocator,
    receipt: Value,
    executables: Value,
    images: Value,
) !void {
    const obj = try requireKeys(receipt, &.{
        "schema_version", "kind",              "evidence_kind",
        "issuer",         "source_commit",     "deployment_receipt_sha256",
        "os",             "arch",              "hardware_execution",
        "platform",       "safety",            "images",
        "executables",    "compiler_embedded", "runtime_linkage",
        "lifecycle",      "allocator",         "page_policy",
        "wasm",           "aot_module",        "target",
        "options",        "build_options",     "adapter",
        "clock",          "native_stack",      "memory_observer",
    });
    try requireExactInt(obj.get("schema_version").?, 2);
    try requireExactString(obj.get("kind").?, "wamr-jit-independent-image-deployment-receipt");
    try requireExactString(obj.get("evidence_kind").?, "measurement");
    try requireExactString(obj.get("os").?, "unikraft");
    try requireExactString(obj.get("arch").?, "x86_64");
    try requireExactBool(obj.get("hardware_execution").?, true);
    _ = try availableToken(obj.get("issuer").?);
    _ = try requireDigest(obj.get("source_commit").?, 40);
    _ = try requireDigest(obj.get("deployment_receipt_sha256").?, 64);
    try validatePlatform(obj.get("platform").?);
    try requireExactString(obj.get("platform").?.object.get("arch").?, "x86_64");
    try validateNativeTarget(obj.get("target").?);
    try validateNativeOptions(obj.get("options").?);
    try validateNativeBuildOptions(obj.get("build_options").?);
    try validateSafety(obj.get("safety").?);
    try validateCompilerEmbedded(obj.get("compiler_embedded").?);
    try requireExactString(obj.get("runtime_linkage").?, "static");
    try requireExactString(obj.get("lifecycle").?, lifecycle);
    _ = try availableToken(obj.get("allocator").?);
    _ = try availableToken(obj.get("page_policy").?);
    try validateArtifact(obj.get("wasm").?);
    try validateArtifact(obj.get("aot_module").?);
    try require(try requireInt(obj.get("wasm").?.object.get("bytes").?, 1) <= 1024 * 1024);

    const exe_shas = try validateArtifactPair(allocator, obj.get("executables").?, executables);
    const img_shas = try validateArtifactPair(allocator, obj.get("images").?, images);
    try require(!std.mem.eql(u8, img_shas.aot_sha, exe_shas.aot_sha));
    try require(!std.mem.eql(u8, img_shas.jit_sha, exe_shas.jit_sha));

    const adapter = try requireKeys(obj.get("adapter").?, &.{ "name", "source_commit", "qualification_receipt_sha256" });
    _ = try availableToken(adapter.get("name").?);
    _ = try requireDigest(adapter.get("source_commit").?, 40);
    _ = try requireDigest(adapter.get("qualification_receipt_sha256").?, 64);

    const clock = try requireKeys(obj.get("clock").?, &.{ "method", "resolution_ns", "scope" });
    _ = try availableToken(clock.get("method").?);
    _ = try requireInt(clock.get("resolution_ns").?, 1);
    try requireExactString(clock.get("scope").?, "guest-monotonic-execution");

    const stack = try requireKeys(obj.get("native_stack").?, &.{
        "generated_frames_bytes",
        "compiler_embedder_callbacks_bytes",
        "provisioned_bytes",
    });
    const gen_frames = try requireInt(stack.get("generated_frames_bytes").?, 256 * 1024);
    const cb_bytes = try requireInt(stack.get("compiler_embedder_callbacks_bytes").?, 1);
    const prov_bytes = try requireInt(stack.get("provisioned_bytes").?, 1);
    try require(gen_frames + cb_bytes <= prov_bytes);

    const observer = try requireKeys(obj.get("memory_observer").?, &.{
        "method",   "quantity", "coverage",
        "excludes", "sampling", "interval_ns",
    });
    _ = try availableToken(observer.get("method").?);
    try requireExactString(observer.get("quantity").?, "physical-backing-bytes");
    const cov = observer.get("coverage").?;
    try require(cov == .string and
        (std.mem.eql(u8, cov.string, "whole-guest") or
            std.mem.eql(u8, cov.string, "caller-allocator-and-native-pages")));
    const excludes = observer.get("excludes").?;
    try require(excludes == .array);
    const expected_excludes: []const []const u8 = if (std.mem.eql(u8, cov.string, "whole-guest"))
        &.{"hypervisor"}
    else
        &.{ "image", "native-stack", "other-kernel-allocations", "page-tables" };
    try require(excludes.array.items.len == expected_excludes.len);
    for (excludes.array.items, expected_excludes) |item, exp| {
        try requireExactString(item, exp);
    }
    const sampling = observer.get("sampling").?;
    try require(sampling == .string and
        (std.mem.eql(u8, sampling.string, "continuous-high-water") or
            std.mem.eql(u8, sampling.string, "periodic") or
            std.mem.eql(u8, sampling.string, "phase-boundaries")));
    if (std.mem.eql(u8, sampling.string, "periodic")) {
        _ = try requireInt(observer.get("interval_ns").?, 1);
    } else {
        try require(observer.get("interval_ns").? == .null);
    }
}

pub fn validateNativeRequest(
    allocator: Allocator,
    request: Value,
    trusted_receipt_sha256: ?[]const u8,
) !void {
    const obj = try requireKeys(request, &.{
        "schema_version", "kind",           "transport",
        "evidence_kind",  "nonce",          "created_at",
        "expires_at",     "executables",    "images",
        "receipt",        "receipt_sha256",
    });
    try requireExactInt(obj.get("schema_version").?, 2);
    try requireExactString(obj.get("kind").?, "wamr-jit-comparison-request");
    try requireExactString(obj.get("transport").?, "external-native");
    try requireExactString(obj.get("evidence_kind").?, "measurement");
    try validateUuid(obj.get("nonce").?);
    const start = try parseUtcTimestampNs(obj.get("created_at").?);
    const end = try parseUtcTimestampNs(obj.get("expires_at").?);
    const diff = end - start;
    try require(diff > 0 and diff <= 24 * 3600 * @as(i128, std.time.ns_per_s));
    const trusted = try requireDigestSlice(trusted_receipt_sha256, 64);
    try requireExactString(obj.get("receipt_sha256").?, trusted);
    const rec_sha = sha(try encoded(allocator, obj.get("receipt").?));
    try require(std.mem.eql(u8, &rec_sha, trusted));
    try validateNativeReceipt(allocator, obj.get("receipt").?, obj.get("executables").?, obj.get("images").?);
}

pub fn prepareNative(
    allocator: Allocator,
    io: std.Io,
    aot: []const u8,
    jit: []const u8,
    output: []const u8,
    images: Value,
    receipt: Value,
    trusted_receipt_sha256: ?[]const u8,
) !Value {
    var executables_map: std.json.ObjectMap = .empty;
    try executables_map.put(allocator, "aot", try artifact(allocator, io, aot));
    try executables_map.put(allocator, "jit", try artifact(allocator, io, jit));
    const executables: Value = .{ .object = executables_map };

    const now_ns = std.Io.Clock.real.now(io).nanoseconds;
    const day_ns: i128 = 24 * 3600 * @as(i128, std.time.ns_per_s);
    const rec_sha = try shaAlloc(allocator, try encoded(allocator, receipt));

    var req_map: std.json.ObjectMap = .empty;
    try req_map.put(allocator, "schema_version", .{ .integer = 2 });
    try req_map.put(allocator, "kind", .{ .string = "wamr-jit-comparison-request" });
    try req_map.put(allocator, "transport", .{ .string = "external-native" });
    try req_map.put(allocator, "evidence_kind", .{ .string = "measurement" });
    try req_map.put(allocator, "nonce", .{ .string = try generateUuid(allocator, io) });
    try req_map.put(allocator, "created_at", .{ .string = try formatUtcTimestamp(allocator, now_ns) });
    try req_map.put(allocator, "expires_at", .{ .string = try formatUtcTimestamp(allocator, now_ns + day_ns) });
    try req_map.put(allocator, "executables", executables);
    try req_map.put(allocator, "images", images);
    try req_map.put(allocator, "receipt", receipt);
    try req_map.put(allocator, "receipt_sha256", .{ .string = rec_sha });

    const request: Value = .{ .object = req_map };
    try validateNativeRequest(allocator, request, trusted_receipt_sha256);
    try createPrivateDir(io, output);
    const req_path = try std.fs.path.join(allocator, &.{ output, "request.json" });
    try privateWrite(io, req_path, try encoded(allocator, request));
    return request;
}

pub fn nativeSampleRecord(allocator: Allocator, raw: []const u8) !Value {
    try require(raw.len <= max_serial_bytes);
    const first = std.mem.indexOf(u8, raw, prefix) orelse return error.ValidationFailed;
    if (std.mem.indexOfPos(u8, raw, first + prefix.len, prefix) != null) return error.ValidationFailed;
    try require(first == 0 or raw[first - 1] == '\n' or raw[first - 1] == '\r');
    const end = std.mem.indexOfScalarPos(u8, raw, first, '\n') orelse return error.ValidationFailed;
    try require(end + 1 - first <= max_record_bytes);
    var line = raw[first + prefix.len .. end];
    if (std.mem.endsWith(u8, line, "\r")) line = line[0 .. line.len - 1];
    return strictJson(allocator, line);
}

fn readBoundedPrefix(allocator: Allocator, io: std.Io, path: []const u8, limit: usize) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const buf = try allocator.alloc(u8, limit + 1);
    var total: usize = 0;
    while (total < buf.len) {
        var iov = [_][]u8{buf[total..]};
        const n = file.readStreaming(io, &iov) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        if (n == 0) break;
        total += n;
    }
    return buf[0..total];
}

fn boundedNativeBytes(allocator: Allocator, io: std.Io, path: []const u8, limit: usize) ![]u8 {
    const data = try readBoundedPrefix(allocator, io, path, limit);
    try require(data.len <= limit);
    return data;
}

pub fn nativeReport(
    allocator: Allocator,
    io: std.Io,
    directory: []const u8,
    trusted_receipt_sha256: ?[]const u8,
    trusted_capture_sha256: ?[]const u8,
) !Value {
    const req_path = try std.fs.path.join(allocator, &.{ directory, "request.json" });
    const req_bytes = try boundedNativeBytes(allocator, io, req_path, max_receipt_bytes);
    const request = try strictJson(allocator, req_bytes);
    try validateNativeRequest(allocator, request, trusted_receipt_sha256);
    const req_obj = request.object;

    const cap_path = try std.fs.path.join(allocator, &.{ directory, "capture.json" });
    const cap_bytes = try boundedNativeBytes(allocator, io, cap_path, max_receipt_bytes);
    const cap = try strictJson(allocator, cap_bytes);
    const trusted_cap = try requireDigestSlice(trusted_capture_sha256, 64);
    const cap_sha = sha(try encoded(allocator, cap));
    try require(std.mem.eql(u8, &cap_sha, trusted_cap));

    const cap_obj = try requireKeys(cap, &.{
        "schema_version",     "kind",                 "evidence_kind",
        "request_sha256",     "image_receipt_sha256", "adapter_qualification_sha256",
        "hardware_execution", "records",
    });
    try requireExactInt(cap_obj.get("schema_version").?, 1);
    try requireExactString(cap_obj.get("kind").?, "wamr-jit-native-capture-receipt");
    try requireExactString(cap_obj.get("evidence_kind").?, "measurement");
    try requireExactBool(cap_obj.get("hardware_execution").?, true);

    const req_sha = try shaAlloc(allocator, req_bytes);
    const trusted_rec = trusted_receipt_sha256.?;
    const rec_obj = req_obj.get("receipt").?.object;
    const adapter_qual = rec_obj.get("adapter").?.object.get("qualification_receipt_sha256").?.string;
    try requireExactString(cap_obj.get("request_sha256").?, req_sha);
    try requireExactString(cap_obj.get("image_receipt_sha256").?, trusted_rec);
    try requireExactString(cap_obj.get("adapter_qualification_sha256").?, adapter_qual);

    const records = try requireKeys(cap_obj.get("records").?, &modes);
    const req_start = try parseUtcTimestampNs(req_obj.get("created_at").?);
    const req_end = try parseUtcTimestampNs(req_obj.get("expires_at").?);

    var samples_map: std.json.ObjectMap = .empty;
    var memory_map: std.json.ObjectMap = .empty;

    for (modes) |mode| {
        const record = try requireKeys(records.get(mode).?, &.{
            "image",        "executable",       "serial",
            "outcome",      "capture_complete", "started_at",
            "completed_at", "memory",
        });
        const img_key = if (std.mem.eql(u8, mode, "aot")) "aot" else "jit";
        try require(try identical(allocator, record.get("image").?, req_obj.get("images").?.object.get(img_key).?));
        try require(try identical(allocator, record.get("executable").?, req_obj.get("executables").?.object.get(img_key).?));
        try requireExactString(record.get("outcome").?, "success");
        try requireExactBool(record.get("capture_complete").?, true);

        const start = try parseUtcTimestampNs(record.get("started_at").?);
        const end = try parseUtcTimestampNs(record.get("completed_at").?);
        try require(req_start <= start and start <= end and end <= req_end);

        const raw_path = try std.fs.path.join(allocator, &.{ directory, try std.fmt.allocPrint(allocator, "{s}.serial", .{mode}) });
        const raw = try boundedNativeBytes(allocator, io, raw_path, max_serial_bytes);
        const raw_art = try artifactValue(allocator, try shaAlloc(allocator, raw), raw.len);
        try require(try identical(allocator, record.get("serial").?, raw_art));

        const sample = try validateSample(try nativeSampleRecord(allocator, raw), mode, req_sha);
        const s_obj = sample.object;
        const expected_res = rec_obj.get("clock").?.object.get("resolution_ns").?.integer;
        try requireExactInt(s_obj.get("clock_resolution_ns").?, expected_res);

        const sample_wasm = try artifactValue(allocator, s_obj.get("wasm_sha256").?.string, @intCast(s_obj.get("wasm_bytes").?.integer));
        try require(try identical(allocator, sample_wasm, rec_obj.get("wasm").?));
        if (std.mem.eql(u8, mode, "aot")) {
            const sample_cwasm = try artifactValue(allocator, s_obj.get("cwasm_sha256").?.string, @intCast(s_obj.get("cwasm_bytes").?.integer));
            try require(try identical(allocator, sample_cwasm, rec_obj.get("aot_module").?));
        } else {
            try require(s_obj.get("compiler_retained_bytes").?.integer == s_obj.get("cwasm_bytes").?.integer);
        }

        const mem_before = s_obj.get("memory_before").?.object;
        const mem_after = s_obj.get("memory_after").?.object;
        for ([_][]const u8{ "code_bytes", "code_reserved_bytes", "linear_reserved_bytes" }) |field| {
            try require(mem_before.get(field).?.integer == mem_after.get(field).?.integer);
        }
        try require(mem_after.get("heap_peak_bytes").?.integer >= mem_before.get("heap_peak_bytes").?.integer);

        const obs_val = record.get("memory").?;
        const obs = try requireKeys(obs_val, &.{ "before_bytes", "observed_max_bytes", "after_teardown_bytes", "observation_count" });
        const before_b = try requireInt(obs.get("before_bytes").?, 0);
        const max_b = try requireInt(obs.get("observed_max_bytes").?, 1);
        const after_b = try requireInt(obs.get("after_teardown_bytes").?, 0);
        _ = try requireInt(obs.get("observation_count").?, 2);
        try require(before_b <= max_b and after_b <= max_b);

        try samples_map.put(allocator, mode, sample);
        try memory_map.put(allocator, mode, obs_val);
    }

    const aot_peak = samples_map.get("aot").?.object.get("caller_peak_bytes").?.integer;
    var peak_growth: std.json.ObjectMap = .empty;
    for (modes[1..]) |mode| {
        const p = samples_map.get(mode).?.object.get("caller_peak_bytes").?.integer;
        try peak_growth.put(allocator, mode, .{ .integer = p - aot_peak });
    }

    const aot_obs_max = memory_map.get("aot").?.object.get("observed_max_bytes").?.integer;
    var obs_growth: std.json.ObjectMap = .empty;
    for (modes[1..]) |mode| {
        const m = memory_map.get(mode).?.object.get("observed_max_bytes").?.integer;
        try obs_growth.put(allocator, mode, .{ .integer = m - aot_obs_max });
    }

    const imgs = req_obj.get("images").?.object;
    const exes = req_obj.get("executables").?.object;
    var out: std.json.ObjectMap = .empty;
    try out.put(allocator, "schema_version", .{ .integer = 2 });
    try out.put(allocator, "kind", .{ .string = "wamr-jit-comparison" });
    try out.put(allocator, "transport", .{ .string = "external-native" });
    try out.put(allocator, "request_sha256", .{ .string = req_sha });
    try out.put(allocator, "evidence_kind", .{ .string = "measurement" });
    try out.put(allocator, "samples", .{ .object = samples_map });
    try out.put(allocator, "image_receipt_sha256", .{ .string = trusted_rec });
    try out.put(allocator, "capture_receipt_sha256", .{ .string = trusted_cap });
    try out.put(allocator, "images", req_obj.get("images").?);
    try out.put(allocator, "executables", req_obj.get("executables").?);
    try out.put(allocator, "target", rec_obj.get("target").?);
    try out.put(allocator, "options", rec_obj.get("options").?);
    try out.put(allocator, "build_options", rec_obj.get("build_options").?);
    try out.put(allocator, "lifecycle", .{ .string = lifecycle });
    try out.put(allocator, "complete_image_bytes_growth", .{
        .integer = imgs.get("jit").?.object.get("bytes").?.integer - imgs.get("aot").?.object.get("bytes").?.integer,
    });
    try out.put(allocator, "executable_bytes_growth", .{
        .integer = exes.get("jit").?.object.get("bytes").?.integer - exes.get("aot").?.object.get("bytes").?.integer,
    });
    try out.put(allocator, "caller_requested_peak_growth", .{ .object = peak_growth });
    try out.put(allocator, "memory_scope", .{ .string = "requested-allocator-bytes-and-logical-page-commitment-not-RSS" });
    try out.put(allocator, "native_memory_observer", rec_obj.get("memory_observer").?);
    try out.put(allocator, "native_memory_observations", .{ .object = memory_map });
    try out.put(allocator, "native_observed_max_bytes_growth", .{ .object = obs_growth });
    try out.put(allocator, "timing_scope", .{ .string = "guest-clock-only-not-serial-arrival-or-collector-latency" });
    try out.put(allocator, "qualification", .{ .string = "bound-to-independently-supplied-attestations-not-hardware-proof" });
    return .{ .object = out };
}

pub fn importNative(
    allocator: Allocator,
    io: std.Io,
    directory: []const u8,
    external: []const u8,
    aot: []const u8,
    jit: []const u8,
    images: Value,
    trusted_receipt_sha256: ?[]const u8,
    trusted_capture_sha256: ?[]const u8,
) !Value {
    const req_path = try std.fs.path.join(allocator, &.{ directory, "request.json" });
    const req_bytes = try boundedNativeBytes(allocator, io, req_path, max_receipt_bytes);
    const request = try strictJson(allocator, req_bytes);
    try validateNativeRequest(allocator, request, trusted_receipt_sha256);

    var current_exes: std.json.ObjectMap = .empty;
    try current_exes.put(allocator, "aot", try artifact(allocator, io, aot));
    try current_exes.put(allocator, "jit", try artifact(allocator, io, jit));
    try require(try identical(allocator, request.object.get("executables").?, .{ .object = current_exes }));
    try require(try identical(allocator, request.object.get("images").?, images));

    const status_path = try std.fs.path.join(allocator, &.{ directory, "import.status.json" });
    if (std.Io.Dir.cwd().access(io, status_path, .{})) |_| {
        return error.ImportAlreadyAttempted;
    } else |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    }

    try setPrivateDirPermissions(io, directory);
    var archive_errors: std.json.ObjectMap = .empty;
    const ArchiveItem = struct { name: []const u8, limit: usize };
    const items = [_]ArchiveItem{
        .{ .name = "capture.json", .limit = max_receipt_bytes },
        .{ .name = "aot.serial", .limit = max_serial_bytes },
        .{ .name = "fast.serial", .limit = max_serial_bytes },
        .{ .name = "full.serial", .limit = max_serial_bytes },
    };

    for (items) |item| {
        const src_path = try std.fs.path.join(allocator, &.{ external, item.name });
        const dst_path = try std.fs.path.join(allocator, &.{ directory, item.name });
        const data = readBoundedPrefix(allocator, io, src_path, item.limit) catch |err| {
            const msg = if (err == error.FileNotFound)
                "FileNotFoundError: missing archive input"
            else
                try std.fmt.allocPrint(allocator, "OSError: {s}", .{@errorName(err)});
            try archive_errors.put(allocator, item.name, .{ .string = msg });
            continue;
        };
        privateWrite(io, dst_path, data) catch |err| {
            const msg = try std.fmt.allocPrint(allocator, "OSError: {s}", .{@errorName(err)});
            try archive_errors.put(allocator, item.name, .{ .string = msg });
            continue;
        };
        if (data.len > item.limit) {
            try archive_errors.put(allocator, item.name, .{ .string = "exceeds archive bound; bounded prefix retained" });
        }
    }

    if (archive_errors.count() != 0) {
        var status_obj: std.json.ObjectMap = .empty;
        try status_obj.put(allocator, "success", .{ .bool = false });
        try status_obj.put(allocator, "error", .{ .string = "ValueError: native archival failed; available bounded evidence retained" });
        try status_obj.put(allocator, "archive_errors", .{ .object = archive_errors });
        try privateWrite(io, status_path, try encoded(allocator, .{ .object = status_obj }));
        return error.NativeArchivalFailed;
    }

    const result = nativeReport(allocator, io, directory, trusted_receipt_sha256, trusted_capture_sha256) catch |err| {
        var status_obj: std.json.ObjectMap = .empty;
        try status_obj.put(allocator, "success", .{ .bool = false });
        try status_obj.put(allocator, "error", .{ .string = try std.fmt.allocPrint(allocator, "ValueError: {s}", .{@errorName(err)}) });
        try status_obj.put(allocator, "archive_errors", .{ .object = archive_errors });
        try privateWrite(io, status_path, try encoded(allocator, .{ .object = status_obj }));
        return err;
    };

    const comp_path = try std.fs.path.join(allocator, &.{ directory, "comparison.json" });
    try privateWrite(io, comp_path, try encoded(allocator, result));
    var ok_status: std.json.ObjectMap = .empty;
    try ok_status.put(allocator, "success", .{ .bool = true });
    try ok_status.put(allocator, "error", .null);
    try ok_status.put(allocator, "archive_errors", .{ .object = archive_errors });
    try privateWrite(io, status_path, try encoded(allocator, .{ .object = ok_status }));
    return result;
}

pub const CliOptions = struct {
    aot: ?[]const u8 = null,
    jit: ?[]const u8 = null,
    output: ?[]const u8 = null,
    runner: []const u8 = "",
    timeout_s: u64 = 120,
    receipt: ?[]const u8 = null,
    trusted_receipt_sha256: ?[]const u8 = null,
    aot_image: ?[]const u8 = null,
    jit_image: ?[]const u8 = null,
    native_prepare: bool = false,
    native_import: ?[]const u8 = null,
    trusted_capture_sha256: ?[]const u8 = null,
};

fn splitWords(allocator: Allocator, text: []const u8) ![]const []const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    var iter = std.mem.tokenizeAny(u8, text, " \t\r\n");
    while (iter.next()) |tok| try list.append(allocator, tok);
    return list.toOwnedSlice(allocator);
}

pub fn runCli(allocator: Allocator, io: std.Io, args: []const []const u8) !void {
    var opts: CliOptions = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--native-prepare")) {
            if (opts.native_import != null) return error.MutuallyExclusiveAction;
            opts.native_prepare = true;
            continue;
        }
        const eq = std.mem.indexOfScalar(u8, arg, '=');
        const name = if (eq) |at| arg[0..at] else arg;
        const val = if (eq) |at| arg[at + 1 ..] else blk: {
            i += 1;
            if (i == args.len) return error.MissingOptionValue;
            break :blk args[i];
        };
        if (std.mem.eql(u8, name, "--aot")) {
            opts.aot = val;
        } else if (std.mem.eql(u8, name, "--jit")) {
            opts.jit = val;
        } else if (std.mem.eql(u8, name, "--output")) {
            opts.output = val;
        } else if (std.mem.eql(u8, name, "--runner")) {
            opts.runner = val;
        } else if (std.mem.eql(u8, name, "--timeout")) {
            opts.timeout_s = try std.fmt.parseInt(u64, val, 10);
        } else if (std.mem.eql(u8, name, "--receipt")) {
            opts.receipt = val;
        } else if (std.mem.eql(u8, name, "--trusted-receipt-sha256")) {
            opts.trusted_receipt_sha256 = val;
        } else if (std.mem.eql(u8, name, "--aot-image")) {
            opts.aot_image = val;
        } else if (std.mem.eql(u8, name, "--jit-image")) {
            opts.jit_image = val;
        } else if (std.mem.eql(u8, name, "--native-import")) {
            if (opts.native_prepare) return error.MutuallyExclusiveAction;
            opts.native_import = val;
        } else if (std.mem.eql(u8, name, "--trusted-capture-sha256")) {
            opts.trusted_capture_sha256 = val;
        } else return error.UnknownOption;
    }

    const aot = opts.aot orelse return error.MissingRequiredOption;
    const jit = opts.jit orelse return error.MissingRequiredOption;
    const output = opts.output orelse return error.MissingRequiredOption;

    const any_evidence = opts.aot_image != null or opts.jit_image != null or opts.receipt != null or opts.trusted_receipt_sha256 != null;
    const all_evidence = opts.aot_image != null and opts.jit_image != null and opts.receipt != null and opts.trusted_receipt_sha256 != null;
    var images: ?Value = null;
    var receipt: ?Value = null;
    if (any_evidence) {
        if (!all_evidence) {
            std.debug.print("error: measurement needs both complete images, independent receipt, and its trusted hash\n", .{});
            return error.InvalidCliArguments;
        }
        var imgs: std.json.ObjectMap = .empty;
        try imgs.put(allocator, "aot", try artifact(allocator, io, opts.aot_image.?));
        try imgs.put(allocator, "jit", try artifact(allocator, io, opts.jit_image.?));
        images = .{ .object = imgs };
        receipt = try readJson(allocator, io, opts.receipt.?);
    }

    if (opts.native_prepare or opts.native_import != null) {
        if (!all_evidence or opts.runner.len != 0) {
            if (!builtin.is_test) std.debug.print("error: native transport needs complete images and independent receipt/hash, without a runner\n", .{});
            return error.InvalidCliArguments;
        }
        if (opts.native_import) |ext_dir| {
            if (opts.trusted_capture_sha256 == null) {
                if (!builtin.is_test) std.debug.print("error: native import needs an independently trusted capture receipt hash\n", .{});
                return error.InvalidCliArguments;
            }
            const rec_sha = sha(try encoded(allocator, receipt.?));
            try require(std.mem.eql(u8, &rec_sha, opts.trusted_receipt_sha256.?));
            _ = try importNative(
                allocator,
                io,
                output,
                ext_dir,
                aot,
                jit,
                images.?,
                opts.trusted_receipt_sha256,
                opts.trusted_capture_sha256,
            );
        } else {
            if (opts.trusted_capture_sha256 != null) {
                if (!builtin.is_test) std.debug.print("error: capture trust is supplied only after a native capture exists\n", .{});
                return error.InvalidCliArguments;
            }
            _ = try prepareNative(
                allocator,
                io,
                aot,
                jit,
                output,
                images.?,
                receipt.?,
                opts.trusted_receipt_sha256,
            );
        }
    } else {
        if (opts.trusted_capture_sha256 != null) {
            if (!builtin.is_test) std.debug.print("error: native capture receipts cannot qualify a Linux capture\n", .{});
            return error.InvalidCliArguments;
        }
        _ = try capture(allocator, io, aot, jit, output, .{
            .runner = try splitWords(allocator, opts.runner),
            .timeout_s = opts.timeout_s,
            .images = images,
            .receipt = receipt,
            .trusted_receipt_sha256 = opts.trusted_receipt_sha256,
        });
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    runCli(allocator, init.io, args[1..]) catch |err| {
        std.debug.print("native-jit-benchmark: {s}\n", .{@errorName(err)});
        std.process.exit(if (err == error.InvalidCliArguments) 2 else 1);
    };
}

fn cloneValue(allocator: Allocator, value: Value) !Value {
    const bytes = try encoded(allocator, value);
    return strictJson(allocator, bytes);
}

fn makeLegacyFixtureReceipt(allocator: Allocator, executables: Value, images: Value) !Value {
    const json_text =
        \\{
        \\  "schema_version": 1,
        \\  "kind": "wamr-jit-independent-image-deployment-receipt",
        \\  "issuer": "unit-test-not-evidence",
        \\  "source_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        \\  "deployment_receipt_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        \\  "os": "linux",
        \\  "arch": "x86_64",
        \\  "hardware_execution": true,
        \\  "safety": {"bounds_checks": true, "checked_imports": true, "checked_traps": true, "wx": true},
        \\  "platform": {"arch": "x86_64", "cpu_model": "fixture-cpu", "active_cpu_count": 1, "azure_sku": "fixture-sku", "azure_region": "fixture-region"},
        \\  "compiler_embedded": {"aot": false, "jit": true},
        \\  "runtime_linkage": "static",
        \\  "lifecycle": "same-instance-workload-initializes-memory-in-timed-call",
        \\  "allocator": "caller-init-gpa",
        \\  "page_policy": "mmap-mprotect-munmap"
        \\}
    ;
    var val = try strictJson(allocator, json_text);
    try val.object.put(allocator, "executables", try cloneValue(allocator, executables));
    try val.object.put(allocator, "images", try cloneValue(allocator, images));
    try val.object.put(allocator, "wasm", try cloneValue(allocator, executables.object.get("aot").?));
    try val.object.put(allocator, "aot_module", try cloneValue(allocator, executables.object.get("jit").?));
    return val;
}

fn makeNativeFixtureReceipt(allocator: Allocator, executables: Value, images: Value) !Value {
    const json_text =
        \\{
        \\  "schema_version": 2,
        \\  "kind": "wamr-jit-independent-image-deployment-receipt",
        \\  "evidence_kind": "measurement",
        \\  "issuer": "unit-test-not-evidence",
        \\  "source_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        \\  "deployment_receipt_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        \\  "os": "unikraft",
        \\  "arch": "x86_64",
        \\  "hardware_execution": true,
        \\  "safety": {"bounds_checks": true, "checked_imports": true, "checked_traps": true, "wx": true},
        \\  "platform": {"arch": "x86_64", "cpu_model": "fixture-cpu", "active_cpu_count": 1, "azure_sku": "fixture-sku", "azure_region": "fixture-region"},
        \\  "compiler_embedded": {"aot": false, "jit": true},
        \\  "runtime_linkage": "static",
        \\  "lifecycle": "same-instance-workload-initializes-memory-in-timed-call",
        \\  "allocator": "unit-test-allocator",
        \\  "page_policy": "unit-test-page-policy",
        \\  "target": {"triple": "x86_64-freestanding-none", "abi": "sysv", "contract_version": 1, "aot_profile": "0x554b0001", "jit_profile": "0x554b0002"},
        \\  "options": {
        \\    "presets": ["fast", "full"],
        \\    "verify": "after_each_pass",
        \\    "compiler": {
        \\      "max_input_bytes": 1048576,
        \\      "max_compiler_bytes": 67108864,
        \\      "max_code_bytes": 4194304,
        \\      "max_polls": 100000,
        \\      "max_functions": 1024,
        \\      "max_function_bytes": 65536,
        \\      "max_locals": 4096,
        \\      "max_blocks_per_function": 2048,
        \\      "max_instructions_per_function": 32768
        \\    },
        \\    "runtime": {
        \\      "max_heap_bytes": 16777216,
        \\      "max_reserved_bytes": 8388608,
        \\      "max_code_bytes": 4194304,
        \\      "max_memory_pages": 8,
        \\      "max_table_elements": 16,
        \\      "jit_fuel_per_invocation": 100000,
        \\      "aot_fuel_per_invocation": null
        \\    },
        \\    "rounds": 2000,
        \\    "invocations": 4,
        \\    "growth_pages": 1
        \\  },
        \\  "build_options": {
        \\    "pic": true,
        \\    "single_threaded": true,
        \\    "red_zone": false,
        \\    "stack_check": false,
        \\    "stack_protector": false,
        \\    "unwind_tables": "none",
        \\    "error_tracing": false,
        \\    "link_libc": false,
        \\    "optimize": "ReleaseSafe"
        \\  },
        \\  "adapter": {
        \\    "name": "unit-test-not-qualified",
        \\    "source_commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        \\    "qualification_receipt_sha256": "9999999999999999999999999999999999999999999999999999999999999999"
        \\  },
        \\  "clock": {"method": "unit-test-clock", "resolution_ns": 1, "scope": "guest-monotonic-execution"},
        \\  "native_stack": {
        \\    "generated_frames_bytes": 262144,
        \\    "compiler_embedder_callbacks_bytes": 1048576,
        \\    "provisioned_bytes": 2097152
        \\  },
        \\  "memory_observer": {
        \\    "method": "unit-test-not-an-observation",
        \\    "quantity": "physical-backing-bytes",
        \\    "coverage": "caller-allocator-and-native-pages",
        \\    "excludes": ["image", "native-stack", "other-kernel-allocations", "page-tables"],
        \\    "sampling": "continuous-high-water",
        \\    "interval_ns": null
        \\  }
        \\}
    ;
    var val = try strictJson(allocator, json_text);
    try val.object.put(allocator, "executables", try cloneValue(allocator, executables));
    try val.object.put(allocator, "images", try cloneValue(allocator, images));
    try val.object.put(allocator, "wasm", try cloneValue(allocator, executables.object.get("aot").?));
    try val.object.put(allocator, "aot_module", try cloneValue(allocator, executables.object.get("jit").?));
    return val;
}

fn makeSyntheticSample(
    allocator: Allocator,
    mode: []const u8,
    request_sha: []const u8,
    wasm_art: Value,
    cwasm_art: Value,
) !Value {
    const compiler = !std.mem.eql(u8, mode, "aot");
    const exp = expected();
    const json_text = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "schema_version": 1,
        \\  "kind": "wamr-native-jit-sample",
        \\  "qualification": "requires-independent-image-and-deployment-evidence",
        \\  "request_sha256": "{s}",
        \\  "mode": "{s}",
        \\  "compiler_embedded": {s},
        \\  "wasm_sha256": "{s}",
        \\  "wasm_bytes": {d},
        \\  "cwasm_sha256": "{s}",
        \\  "cwasm_bytes": {d},
        \\  "workload": "volatile-compute-memory-2000",
        \\  "expected": {d},
        \\  "lifecycle": "same-instance-workload-initializes-memory-in-timed-call",
        \\  "clock_resolution_ns": 1,
        \\  "compile_ns": {s},
        \\  "compiler_phases_ns": {s},
        \\  "compiler_peak_bytes": {d},
        \\  "compiler_retained_bytes": {d},
        \\  "compiler_polls": {d},
        \\  "load_ns": 10,
        \\  "instantiate_ns": 10,
        \\  "start_ns": 10,
        \\  "growth_ns": 10,
        \\  "growth_previous_pages": 2,
        \\  "fuel_per_invocation": {s},
        \\  "invocations": [
        \\    {{"ns": 10, "outcome": "returned", "value": {d}, "diagnostic": null}},
        \\    {{"ns": 10, "outcome": "returned", "value": {d}, "diagnostic": null}},
        \\    {{"ns": 10, "outcome": "returned", "value": {d}, "diagnostic": null}},
        \\    {{"ns": 10, "outcome": "returned", "value": {d}, "diagnostic": null}}
        \\  ],
        \\  "memory_before": {{
        \\    "heap_live_bytes": 4096,
        \\    "heap_peak_bytes": 8192,
        \\    "code_bytes": 1024,
        \\    "code_reserved_bytes": 4096,
        \\    "linear_reserved_bytes": 524288,
        \\    "linear_committed_bytes": 131072
        \\  }},
        \\  "memory_after": {{
        \\    "heap_live_bytes": 4096,
        \\    "heap_peak_bytes": 8192,
        \\    "code_bytes": 1024,
        \\    "code_reserved_bytes": 4096,
        \\    "linear_reserved_bytes": 524288,
        \\    "linear_committed_bytes": 196608
        \\  }},
        \\  "caller_peak_bytes": 100000,
        \\  "caller_live_after_teardown": 0,
        \\  "reserved_after_teardown": 0,
        \\  "failure_stage": null,
        \\  "failure": null
        \\}}
    , .{
        request_sha,
        mode,
        if (compiler) "true" else "false",
        wasm_art.object.get("sha256").?.string,
        wasm_art.object.get("bytes").?.integer,
        cwasm_art.object.get("sha256").?.string,
        if (compiler) @as(i64, 4096) else cwasm_art.object.get("bytes").?.integer,
        exp,
        if (compiler) "100" else "null",
        if (compiler) "{\"parse\":10,\"lower\":10,\"optimize\":10,\"codegen\":10,\"emit\":10}" else "null",
        if (compiler) @as(u64, 65536) else 0,
        if (compiler) @as(u64, 4096) else 0,
        if (compiler) @as(u64, 100) else 0,
        if (compiler) "100000" else "null",
        exp,
        exp,
        exp,
        exp,
    });
    return strictJson(allocator, json_text);
}

test "QualificationTests: receipt bindings, emulation rejection, and timeout preservation" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);

    const aot_path = try std.fs.path.join(allocator, &.{ root, "aot" });
    const jit_path = try std.fs.path.join(allocator, &.{ root, "jit" });
    const aot_img_path = try std.fs.path.join(allocator, &.{ root, "aot-img" });
    const jit_img_path = try std.fs.path.join(allocator, &.{ root, "jit-img" });
    try temp.dir.writeFile(io, .{ .sub_path = "aot", .data = "#!/bin/sh\nsleep 5\n" });
    try temp.dir.writeFile(io, .{ .sub_path = "jit", .data = "#!/bin/sh\nsleep 6\n" });
    try temp.dir.writeFile(io, .{ .sub_path = "aot-img", .data = "aot-image-bytes" });
    try temp.dir.writeFile(io, .{ .sub_path = "jit-img", .data = "jit-image-bytes" });
    for ([_][]const u8{ "aot", "jit" }) |name| {
        var f = try temp.dir.openFile(io, name, .{});
        defer f.close(io);
        try f.setPermissions(io, .fromMode(0o755));
    }

    var exes_map: std.json.ObjectMap = .empty;
    try exes_map.put(allocator, "aot", try artifact(allocator, io, aot_path));
    try exes_map.put(allocator, "jit", try artifact(allocator, io, jit_path));
    const executables: Value = .{ .object = exes_map };

    var imgs_map: std.json.ObjectMap = .empty;
    try imgs_map.put(allocator, "aot", try artifact(allocator, io, aot_img_path));
    try imgs_map.put(allocator, "jit", try artifact(allocator, io, jit_img_path));
    const images: Value = .{ .object = imgs_map };

    const receipt = try makeLegacyFixtureReceipt(allocator, executables, images);
    try validateReceipt(allocator, receipt, executables, images);

    for ([_]struct { key: []const u8, val: Value }{
        .{ .key = "runtime_linkage", .val = .{ .string = "dynamic" } },
        .{ .key = "hardware_execution", .val = .{ .bool = false } },
        .{ .key = "arch", .val = .{ .string = "aarch64" } },
        .{ .key = "os", .val = .{ .string = "unikraft" } },
        .{ .key = "lifecycle", .val = .{ .string = "snapshot-replay" } },
        .{ .key = "executables", .val = images },
        .{ .key = "images", .val = executables },
        .{ .key = "allocator", .val = .{ .string = "unknown" } },
    }) |mutation| {
        var altered = try cloneValue(allocator, receipt);
        try altered.object.put(allocator, mutation.key, mutation.val);
        try std.testing.expectError(error.ValidationFailed, validateReceipt(allocator, altered, executables, images));
    }

    // Executable files cannot be relabelled as complete images
    var relabelled = try cloneValue(allocator, receipt);
    try relabelled.object.put(allocator, "images", executables);
    try std.testing.expectError(error.ValidationFailed, validateReceipt(allocator, relabelled, executables, executables));

    // Emulated measurement fails before process launch
    const rec_sha = try shaAlloc(allocator, try encoded(allocator, receipt));
    try std.testing.expectError(error.ValidationFailed, capture(allocator, io, aot_path, jit_path, try std.fs.path.join(allocator, &.{ root, "unused1" }), .{
        .runner = &.{"qemu-x86_64"},
        .receipt = receipt,
        .images = images,
        .trusted_receipt_sha256 = rec_sha,
        .host_os = .linux,
        .host_arch = .x86_64,
    }));

    // Untrusted receipt fails before process launch
    try std.testing.expectError(error.ValidationFailed, capture(allocator, io, aot_path, jit_path, try std.fs.path.join(allocator, &.{ root, "unused2" }), .{
        .receipt = receipt,
        .images = images,
        .trusted_receipt_sha256 = "0000000000000000000000000000000000000000000000000000000000000000",
        .host_os = .linux,
        .host_arch = .x86_64,
    }));

    // Timeout preserves failed raw capture
    const cap_dir = try std.fs.path.join(allocator, &.{ root, "timeout-capture" });
    try std.testing.expectError(error.ValidationFailed, capture(allocator, io, aot_path, jit_path, cap_dir, .{
        .timeout_s = 1,
    }));
    for (modes) |mode| {
        const status = try readJson(allocator, io, try std.fs.path.join(allocator, &.{ cap_dir, try std.fmt.allocPrint(allocator, "{s}.status.json", .{mode}) }));
        try std.testing.expect(status.object.get("returncode").? == .null);
        try std.testing.expect(std.mem.indexOf(u8, status.object.get("error").?.string, "TimeoutExpired") != null);
        try std.testing.expectEqual(@as(i64, 0), status.object.get("stdout").?.object.get("bytes").?.integer);
    }
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, try std.fs.path.join(allocator, &.{ cap_dir, "comparison.json" }), .{}));
}

test "NativeExternalTransportTests: external import, receipt/record mutations, and archive bounds" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);

    const aot_path = try std.fs.path.join(allocator, &.{ root, "aot" });
    const jit_path = try std.fs.path.join(allocator, &.{ root, "jit" });
    const aot_img_path = try std.fs.path.join(allocator, &.{ root, "aot-image" });
    const jit_img_path = try std.fs.path.join(allocator, &.{ root, "jit-image" });
    try temp.dir.writeFile(io, .{ .sub_path = "aot", .data = "UNIT TEST ONLY: aot" });
    try temp.dir.writeFile(io, .{ .sub_path = "jit", .data = "UNIT TEST ONLY: jit" });
    try temp.dir.writeFile(io, .{ .sub_path = "aot-image", .data = "UNIT TEST ONLY: aot-image" });
    try temp.dir.writeFile(io, .{ .sub_path = "jit-image", .data = "UNIT TEST ONLY: jit-image" });

    var exes_map: std.json.ObjectMap = .empty;
    try exes_map.put(allocator, "aot", try artifact(allocator, io, aot_path));
    try exes_map.put(allocator, "jit", try artifact(allocator, io, jit_path));
    const executables: Value = .{ .object = exes_map };

    var imgs_map: std.json.ObjectMap = .empty;
    try imgs_map.put(allocator, "aot", try artifact(allocator, io, aot_img_path));
    try imgs_map.put(allocator, "jit", try artifact(allocator, io, jit_img_path));
    const images: Value = .{ .object = imgs_map };

    const receipt = try makeNativeFixtureReceipt(allocator, executables, images);
    const receipt_sha = try shaAlloc(allocator, try encoded(allocator, receipt));

    // Native receipt cannot relabel Linux or software metrics
    try std.testing.expectError(error.ValidationFailed, validateReceipt(allocator, receipt, executables, images));
    for ([_]struct { key: []const u8, val: Value }{
        .{ .key = "schema_version", .val = .{ .integer = 1 } },
        .{ .key = "os", .val = .{ .string = "linux" } },
        .{ .key = "evidence_kind", .val = .{ .string = "correctness-only" } },
        .{ .key = "hardware_execution", .val = .{ .bool = false } },
    }) |mutation| {
        var altered = try cloneValue(allocator, receipt);
        try altered.object.put(allocator, mutation.key, mutation.val);
        try std.testing.expectError(error.ValidationFailed, validateNativeReceipt(allocator, altered, executables, images));
    }

    const priv_dir = try std.fs.path.join(allocator, &.{ root, "private" });
    const req = try prepareNative(allocator, io, aot_path, jit_path, priv_dir, images, receipt, receipt_sha);
    const req_bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(allocator, &.{ priv_dir, "request.json" }), allocator, .unlimited);
    const req_sha = try shaAlloc(allocator, req_bytes);

    const ext_dir = try std.fs.path.join(allocator, &.{ root, "external" });
    try std.Io.Dir.cwd().createDirPath(io, ext_dir);

    var records_map: std.json.ObjectMap = .empty;
    for (modes) |mode| {
        const sample = try makeSyntheticSample(allocator, mode, req_sha, receipt.object.get("wasm").?, receipt.object.get("aot_module").?);
        const sample_bytes = try encoded(allocator, sample);
        const raw = try std.mem.concat(allocator, u8, &.{ "\xff\xfeUNIT TEST BOOT NOISE\r\n", prefix, sample_bytes, "\r\n\x80noise\n" });
        const serial_path = try std.fs.path.join(allocator, &.{ ext_dir, try std.fmt.allocPrint(allocator, "{s}.serial", .{mode}) });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = serial_path, .data = raw });

        const img_key = if (std.mem.eql(u8, mode, "aot")) "aot" else "jit";
        var mem_obs: std.json.ObjectMap = .empty;
        try mem_obs.put(allocator, "before_bytes", .{ .integer = 4096 });
        try mem_obs.put(allocator, "observed_max_bytes", .{ .integer = 2 * 1024 * 1024 });
        try mem_obs.put(allocator, "after_teardown_bytes", .{ .integer = 4096 });
        try mem_obs.put(allocator, "observation_count", .{ .integer = 3 });

        var rec_entry: std.json.ObjectMap = .empty;
        try rec_entry.put(allocator, "image", images.object.get(img_key).?);
        try rec_entry.put(allocator, "executable", executables.object.get(img_key).?);
        try rec_entry.put(allocator, "serial", try streamArtifact(allocator, io, serial_path));
        try rec_entry.put(allocator, "outcome", .{ .string = "success" });
        try rec_entry.put(allocator, "capture_complete", .{ .bool = true });
        try rec_entry.put(allocator, "started_at", req.object.get("created_at").?);
        try rec_entry.put(allocator, "completed_at", req.object.get("created_at").?);
        try rec_entry.put(allocator, "memory", .{ .object = mem_obs });
        try records_map.put(allocator, mode, .{ .object = rec_entry });
    }

    var cap_map: std.json.ObjectMap = .empty;
    try cap_map.put(allocator, "schema_version", .{ .integer = 1 });
    try cap_map.put(allocator, "kind", .{ .string = "wamr-jit-native-capture-receipt" });
    try cap_map.put(allocator, "evidence_kind", .{ .string = "measurement" });
    try cap_map.put(allocator, "request_sha256", .{ .string = req_sha });
    try cap_map.put(allocator, "image_receipt_sha256", .{ .string = receipt_sha });
    try cap_map.put(allocator, "hardware_execution", .{ .bool = true });
    try cap_map.put(allocator, "adapter_qualification_sha256", receipt.object.get("adapter").?.object.get("qualification_receipt_sha256").?);
    try cap_map.put(allocator, "records", .{ .object = records_map });
    const cap_val: Value = .{ .object = cap_map };
    const cap_bytes = try encoded(allocator, cap_val);
    const cap_sha = try shaAlloc(allocator, cap_bytes);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(allocator, &.{ ext_dir, "capture.json" }), .data = cap_bytes });

    const imported = try importNative(allocator, io, priv_dir, ext_dir, aot_path, jit_path, images, receipt_sha, cap_sha);
    try std.testing.expectEqualStrings("external-native", imported.object.get("transport").?.string);
    try std.testing.expect(try identical(allocator, imported, try nativeReport(allocator, io, priv_dir, receipt_sha, cap_sha)));
    try std.testing.expectError(error.ImportAlreadyAttempted, importNative(allocator, io, priv_dir, ext_dir, aot_path, jit_path, images, receipt_sha, cap_sha));

    // Duplicate, truncated, non-UTF8, and oversized serial records fail
    for ([_][]const u8{
        "WAMR_JIT_SAMPLE={}\nWAMR_JIT_SAMPLE={}\n",
        "WAMR_JIT_SAMPLE={}",
        "noise\nWAMR_JIT_SAMPLE=\xff\n",
        "not-a-line:WAMR_JIT_SAMPLE={}\n",
        "WAMR_JIT_SAMPLE={\"mode\":\"fast\",\"mode\":\"fast\"}\n",
        "boot noise only\n",
    }) |bad_raw| {
        try std.testing.expectError(error.ValidationFailed, nativeSampleRecord(allocator, bad_raw) catch error.ValidationFailed);
    }

    // Missing first serial retains metadata and later serials
    const priv_dir2 = try std.fs.path.join(allocator, &.{ root, "private2" });
    _ = try prepareNative(allocator, io, aot_path, jit_path, priv_dir2, images, receipt, receipt_sha);
    try std.Io.Dir.cwd().deleteFile(io, try std.fs.path.join(allocator, &.{ ext_dir, "aot.serial" }));
    try std.testing.expectError(error.NativeArchivalFailed, importNative(allocator, io, priv_dir2, ext_dir, aot_path, jit_path, images, receipt_sha, cap_sha));
    const status2 = try readJson(allocator, io, try std.fs.path.join(allocator, &.{ priv_dir2, "import.status.json" }));
    try std.testing.expect(!status2.object.get("success").?.bool);
    try std.testing.expect(status2.object.get("archive_errors").?.object.contains("aot.serial"));
    try std.Io.Dir.cwd().access(io, try std.fs.path.join(allocator, &.{ priv_dir2, "fast.serial" }), .{});
    try std.Io.Dir.cwd().access(io, try std.fs.path.join(allocator, &.{ priv_dir2, "full.serial" }), .{});

    // CLI validation checks
    try std.testing.expectError(error.InvalidCliArguments, runCli(allocator, io, &.{
        "--aot",            aot_path,
        "--jit",            jit_path,
        "--output",         try std.fs.path.join(allocator, &.{ root, "cli1" }),
        "--native-prepare",
    }));
    try std.testing.expectError(error.InvalidCliArguments, runCli(allocator, io, &.{
        "--aot",                    aot_path,
        "--jit",                    jit_path,
        "--output",                 try std.fs.path.join(allocator, &.{ root, "cli2" }),
        "--trusted-capture-sha256", "0000000000000000000000000000000000000000000000000000000000000000",
    }));
}

fn getTestEnv(allocator: Allocator, name: [*:0]const u8) !?[]u8 {
    if (builtin.link_libc) {
        const val = std.c.getenv(name) orelse return null;
        const slice = std.mem.span(val);
        if (slice.len == 0) return null;
        return try allocator.dupe(u8, slice);
    }
    return null;
}

test "ActualMatchedSamplerTests: real matched fast/full/AOT sampling when binaries are provided" {
    if (builtin.os.tag == .windows or !builtin.link_libc) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const aot = (try getTestEnv(allocator, "WAMR_JIT_BENCH_AOT")) orelse return error.SkipZigTest;
    const jit = (try getTestEnv(allocator, "WAMR_JIT_BENCH_JIT")) orelse return error.SkipZigTest;
    const runner_raw = (try getTestEnv(allocator, "WAMR_JIT_BENCH_RUNNER")) orelse "";
    const runner = try splitWords(allocator, runner_raw);

    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    const cap_dir = try std.fs.path.join(allocator, &.{ root, "capture" });

    const result = try capture(allocator, io, aot, jit, cap_dir, .{ .runner = runner });
    try std.testing.expectEqualStrings("correctness-only", result.object.get("evidence_kind").?.string);
    try std.testing.expect(result.object.get("complete_image_bytes_growth").? == .null);

    const samples = result.object.get("samples").?.object;
    for (modes) |mode| {
        const s = samples.get(mode).?.object;
        try std.testing.expectEqual(@as(usize, 4), s.get("invocations").?.array.items.len);
        try std.testing.expectEqual(@as(i64, 3 * 65536), s.get("memory_after").?.object.get("linear_committed_bytes").?.integer);
        try std.testing.expectEqual(@as(i64, 0), s.get("caller_live_after_teardown").?.integer);
        try std.testing.expectEqual(@as(i64, 0), s.get("reserved_after_teardown").?.integer);
        try std.testing.expectEqual(std.mem.eql(u8, mode, "aot"), s.get("compile_ns").? == .null);
    }

    // Check comparator has no compiler symbols via nm -a
    const nm_aot = try process_control.runTimed(allocator, io, .{ .argv = &.{ "nm", "-a", aot } }, 30);
    const nm_jit = try process_control.runTimed(allocator, io, .{ .argv = &.{ "nm", "-a", jit } }, 30);
    try std.testing.expect(nm_aot.term.success() and nm_jit.term.success());
    const markers = [_][]const u8{
        "aot_compile.compileCoreWasm",
        "api.jit.compile",
        "compiler.frontend.",
        "compiler.codegen.x86_64.compile.",
    };
    var jit_has_marker = false;
    for (markers) |marker| {
        try std.testing.expect(std.mem.indexOf(u8, nm_aot.stdout, marker) == null);
        if (std.mem.indexOf(u8, nm_jit.stdout, marker) != null) jit_has_marker = true;
    }
    try std.testing.expect(jit_has_marker);

    // Raw capture revalidation detects changes
    const full_stdout = try std.fs.path.join(allocator, &.{ cap_dir, "full.stdout" });
    const saved = try std.Io.Dir.cwd().readFileAlloc(io, full_stdout, allocator, .unlimited);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = full_stdout, .data = try std.mem.concat(allocator, u8, &.{ saved, "\n" }) });
    try std.testing.expectError(error.ValidationFailed, report(allocator, io, cap_dir, null));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = full_stdout, .data = saved });
    try std.testing.expect(try identical(allocator, try report(allocator, io, cap_dir, null), result));

    // Compiler-free binary rejects JIT mode without fallback
    var reject_argv: std.ArrayList([]const u8) = .empty;
    for (runner) |r| try reject_argv.append(allocator, r);
    try reject_argv.appendSlice(allocator, &.{ aot, "fast", "0000000000000000000000000000000000000000000000000000000000000000" });
    const reject_res = try process_control.runTimed(allocator, io, .{ .argv = reject_argv.items }, 30);
    try std.testing.expect(!reject_res.term.success());
    try std.testing.expect(std.mem.indexOf(u8, reject_res.stdout, prefix) == null);
}

test "NativeEmbeddingArchiveTests: embedding archives do not define memory intrinsics" {
    if (builtin.os.tag == .windows or !builtin.link_libc) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const dir = (try getTestEnv(allocator, "WAMR_JIT_NATIVE_ARCHIVES")) orelse return error.SkipZigTest;
    for ([_][]const u8{ "libwamr-jit.a", "libwamr-jit-aot-sample.a" }) |name| {
        const archive_path = try std.fs.path.join(allocator, &.{ dir, name });
        const nm_res = try process_control.runTimed(allocator, io, .{ .argv = &.{ "nm", "-g", "--defined-only", archive_path } }, 30);
        try std.testing.expect(nm_res.term.success());
        var lines = std.mem.splitScalar(u8, nm_res.stdout, '\n');
        while (lines.next()) |line| {
            var toks = std.mem.tokenizeAny(u8, line, " \t\r");
            var last: ?[]const u8 = null;
            while (toks.next()) |tok| last = tok;
            if (last) |sym| {
                try std.testing.expect(!std.mem.eql(u8, sym, "memcpy"));
                try std.testing.expect(!std.mem.eql(u8, sym, "memset"));
                try std.testing.expect(!std.mem.eql(u8, sym, "memmove"));
            }
        }
    }
}
