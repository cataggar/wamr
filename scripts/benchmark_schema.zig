//! Shared deterministic benchmark cache-key encoding.
const std = @import("std");
const float_text = @import("float_text.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

fn writeString(writer: *std.Io.Writer, text: []const u8) !void {
    if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidBenchmarkString;
    try writer.writeByte('"');
    var index: usize = 0;
    while (index < text.len) {
        const width = try std.unicode.utf8ByteSequenceLength(text[index]);
        const point = try std.unicode.utf8Decode(text[index..][0..width]);
        index += width;
        switch (point) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\x08' => try writer.writeAll("\\b"),
            '\x09' => try writer.writeAll("\\t"),
            '\x0a' => try writer.writeAll("\\n"),
            '\x0c' => try writer.writeAll("\\f"),
            '\x0d' => try writer.writeAll("\\r"),
            0x20...0x21, 0x23...0x5b, 0x5d...0x7e => try writer.writeByte(@intCast(point)),
            0...0x07, 0x0b, 0x0e...0x1f, 0x7f...0xffff => try writer.print("\\u{x:0>4}", .{point}),
            else => {
                const adjusted = point - 0x10000;
                try writer.print("\\u{x:0>4}\\u{x:0>4}", .{
                    0xd800 + (adjusted >> 10), 0xdc00 + (adjusted & 0x3ff),
                });
            },
        }
    }
    try writer.writeByte('"');
}

fn writeFloat(allocator: Allocator, writer: *std.Io.Writer, value: f64) !void {
    if (std.math.isNan(value)) return writer.writeAll("NaN");
    if (std.math.isInf(value)) return writer.writeAll(if (value < 0) "-Infinity" else "Infinity");
    const text = try float_text.format(allocator, value);
    defer allocator.free(text);
    try writer.writeAll(text);
}

fn keyLessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.lessThan(u8, lhs, rhs);
}

fn writeValue(allocator: Allocator, writer: *std.Io.Writer, value: Value, depth: usize) anyerror!void {
    if (depth > 256) return error.BenchmarkJsonTooDeep;
    switch (value) {
        .null => try writer.writeAll("null"),
        .bool => |v| try writer.writeAll(if (v) "true" else "false"),
        .integer => |v| try writer.print("{d}", .{v}),
        .float => |v| try writeFloat(allocator, writer, v),
        .number_string => |v| {
            if (v.len == 0 or (v[0] != '-' and !std.ascii.isDigit(v[0])) or !try std.json.validate(allocator, v))
                return error.InvalidBenchmarkNumber;
            if (std.mem.indexOfAny(u8, v, ".eE") != null) {
                try writeFloat(allocator, writer, std.fmt.parseFloat(f64, v) catch return error.InvalidBenchmarkNumber);
            } else {
                try writer.writeAll(if (std.mem.eql(u8, v, "-0")) "0" else v);
            }
        },
        .string => |v| try writeString(writer, v),
        .array => |v| {
            try writer.writeByte('[');
            for (v.items, 0..) |item, index| {
                if (index != 0) try writer.writeByte(',');
                try writeValue(allocator, writer, item, depth + 1);
            }
            try writer.writeByte(']');
        },
        .object => |v| {
            const keys = try allocator.alloc([]const u8, v.count());
            defer allocator.free(keys);
            for (keys, v.keys()) |*key, value_key| key.* = value_key;
            std.mem.sort([]const u8, keys, {}, keyLessThan);
            try writer.writeByte('{');
            for (keys, 0..) |key, index| {
                if (index != 0) try writer.writeByte(',');
                try writeString(writer, key);
                try writer.writeByte(':');
                try writeValue(allocator, writer, v.get(key).?, depth + 1);
            }
            try writer.writeByte('}');
        },
    }
}

pub fn canonicalJson(allocator: Allocator, value: Value) ![]u8 {
    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();
    try writeValue(allocator, &output.writer, value, 0);
    return output.toOwnedSlice();
}

pub const SCHEMA_VERSION: i64 = 3;
pub const HOST_FINGERPRINT_FIELDS = [_][]const u8{
    "system",
    "machine",
    "cpu",
    "logical_cpus",
    "runner_environment",
    "runner_image",
    "runner_os",
    "runner_arch",
};

pub fn alternatingPairOrder(pair_index: usize, left: []const u8, right: []const u8) [2][]const u8 {
    return if (pair_index % 2 == 0) .{ left, right } else .{ right, left };
}

pub fn commandIdentity(allocator: Allocator, io: std.Io, argv: []const []const u8) Allocator.Error![]const u8 {
    const res = std.process.run(allocator, io, .{
        .argv = argv,
        .stdout_limit = .limited(1024 * 1024),
        .stderr_limit = .limited(1024 * 1024),
    }) catch {
        return allocator.dupe(u8, "unavailable");
    };
    const code: u32 = switch (res.term) {
        .exited => |c| c,
        else => 255,
    };
    const text = if (res.stdout.len > 0) res.stdout else res.stderr;
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) {
        return std.fmt.allocPrint(allocator, "exit {d}", .{code});
    }
    var it = std.mem.splitScalar(u8, trimmed, '\n');
    return allocator.dupe(u8, std.mem.trimEnd(u8, it.first(), "\r"));
}

pub fn collectedAt(allocator: Allocator, io: std.Io) Allocator.Error![]const u8 {
    const now_ns: i128 = @intCast(std.Io.Clock.real.now(io).nanoseconds);
    const secs: u64 = @intCast(@max(@divFloor(now_ns, std.time.ns_per_s), 0));
    const epoch_secs = std.time.epoch.EpochSeconds{ .secs = secs };
    const epoch_day = epoch_secs.getEpochDay();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_secs = epoch_secs.getDaySeconds();
    return std.fmt.allocPrint(
        allocator,
        "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}+00:00",
        .{
            year_day.year,
            month_day.month.numeric(),
            month_day.day_index + 1,
            day_secs.getHoursIntoDay(),
            day_secs.getMinutesIntoHour(),
            day_secs.getSecondsIntoMinute(),
        },
    );
}

fn cpuModel(allocator: Allocator, io: std.Io, fallback: []const u8) Allocator.Error![]const u8 {
    const res = std.process.run(allocator, io, .{
        .argv = &.{"lscpu"},
        .stdout_limit = .limited(1024 * 1024),
        .stderr_limit = .limited(1024 * 1024),
    }) catch return allocator.dupe(u8, fallback);
    if (res.term == .exited and res.term.exited == 0) {
        var line_it = std.mem.splitScalar(u8, res.stdout, '\n');
        while (line_it.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (std.ascii.startsWithIgnoreCase(trimmed, "model name:")) {
                if (std.mem.indexOfScalar(u8, trimmed, ':')) |colon| {
                    const val = std.mem.trim(u8, trimmed[colon + 1 ..], " \t\r");
                    if (val.len > 0) return allocator.dupe(u8, val);
                }
            }
        }
    }
    return allocator.dupe(u8, fallback);
}

pub fn hostMetadata(
    allocator: Allocator,
    io: std.Io,
    environ: std.process.Environ,
    runner_environment_opt: ?[]const u8,
) Allocator.Error!std.json.ObjectMap {
    const env_val = if (runner_environment_opt) |r|
        r
    else
        (environ.getPosix("RUNNER_ENVIRONMENT") orelse "");
    const environment = if (env_val.len > 0)
        env_val
    else if (std.ascii.eqlIgnoreCase(environ.getPosix("GITHUB_ACTIONS") orelse "", "true"))
        "github-actions-unspecified"
    else
        "local";

    const system = try commandIdentity(allocator, io, &.{ "uname", "-s" });
    const release = try commandIdentity(allocator, io, &.{ "uname", "-r" });
    const version = try commandIdentity(allocator, io, &.{ "uname", "-v" });
    const machine = try commandIdentity(allocator, io, &.{ "uname", "-m" });
    const processor = try commandIdentity(allocator, io, &.{ "uname", "-p" });
    const cpu = try cpuModel(allocator, io, machine);
    const logical_cpus: i64 = @intCast(std.Thread.getCpuCount() catch 1);

    const runner_name = try allocator.dupe(u8, environ.getPosix("RUNNER_NAME") orelse "");
    const runner_image = try allocator.dupe(u8, environ.getPosix("ImageOS") orelse "");
    const runner_os = try allocator.dupe(u8, environ.getPosix("RUNNER_OS") orelse "");
    const runner_arch = try allocator.dupe(u8, environ.getPosix("RUNNER_ARCH") orelse "");
    const github_run_id = try allocator.dupe(u8, environ.getPosix("GITHUB_RUN_ID") orelse "");
    const github_run_attempt = try allocator.dupe(u8, environ.getPosix("GITHUB_RUN_ATTEMPT") orelse "");
    const github_workflow = try allocator.dupe(u8, environ.getPosix("GITHUB_WORKFLOW") orelse "");

    var fp_fields: std.json.ObjectMap = .empty;
    try fp_fields.put(allocator, "system", .{ .string = system });
    try fp_fields.put(allocator, "machine", .{ .string = machine });
    try fp_fields.put(allocator, "cpu", .{ .string = cpu });
    try fp_fields.put(allocator, "logical_cpus", .{ .integer = logical_cpus });
    try fp_fields.put(allocator, "runner_environment", .{ .string = try allocator.dupe(u8, environment) });
    try fp_fields.put(allocator, "runner_image", .{ .string = runner_image });
    try fp_fields.put(allocator, "runner_os", .{ .string = runner_os });
    try fp_fields.put(allocator, "runner_arch", .{ .string = runner_arch });

    const key_hex = cacheKey(allocator, .{ .object = fp_fields }) catch return error.OutOfMemory;
    var fp_obj: std.json.ObjectMap = .empty;
    try fp_obj.put(allocator, "sha256", .{ .string = try allocator.dupe(u8, &key_hex) });
    try fp_obj.put(allocator, "fields", .{ .object = fp_fields });

    var meta: std.json.ObjectMap = .empty;
    try meta.put(allocator, "system", .{ .string = system });
    try meta.put(allocator, "release", .{ .string = release });
    try meta.put(allocator, "version", .{ .string = version });
    try meta.put(allocator, "machine", .{ .string = machine });
    try meta.put(allocator, "processor", .{ .string = processor });
    try meta.put(allocator, "cpu", .{ .string = cpu });
    try meta.put(allocator, "logical_cpus", .{ .integer = logical_cpus });
    try meta.put(allocator, "runner_name", .{ .string = runner_name });
    try meta.put(allocator, "runner_image", .{ .string = runner_image });
    try meta.put(allocator, "runner_os", .{ .string = runner_os });
    try meta.put(allocator, "runner_arch", .{ .string = runner_arch });
    try meta.put(allocator, "runner_environment", .{ .string = try allocator.dupe(u8, environment) });
    try meta.put(allocator, "github_run_id", .{ .string = github_run_id });
    try meta.put(allocator, "github_run_attempt", .{ .string = github_run_attempt });
    try meta.put(allocator, "github_workflow", .{ .string = github_workflow });
    try meta.put(allocator, "host_fingerprint", .{ .object = fp_obj });
    return meta;
}

pub fn cacheKey(allocator: Allocator, parts: Value) ![64]u8 {
    if (parts != .object) return error.InvalidBenchmarkCacheKey;
    const bytes = try canonicalJson(allocator, parts);
    defer allocator.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

test "canonical cache keys sort nested keys and escape non-ASCII text" {
    const allocator = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(Value, allocator,
        \\{"z":[true,null,-0.0,1e-7,1e20,"🦀 café\n"],"á":"\u0001","a":{"b":2,"A":1}}
    , .{});
    defer parsed.deinit();
    const canonical = try canonicalJson(allocator, parsed.value);
    defer allocator.free(canonical);
    try std.testing.expectEqualStrings(
        \\{"a":{"A":1,"b":2},"z":[true,null,-0.0,1e-07,1e+20,"\ud83e\udd80 caf\u00e9\n"],"\u00e1":"\u0001"}
    , canonical);
    const key = try cacheKey(allocator, parsed.value);
    try std.testing.expectEqualStrings("351ace1c0a7fa3a9470e6c09e89895d5bb7987b1fdbf86e5db459d9050bba88b", &key);
}

test "canonical values retain large integers and special float spelling" {
    const allocator = std.testing.allocator;
    for ([_]struct { value: Value, expected: []const u8 }{
        .{ .value = .{ .number_string = "18446744073709551616" }, .expected = "18446744073709551616" },
        .{ .value = .{ .number_string = "-0" }, .expected = "0" },
        .{ .value = .{ .number_string = "1.0" }, .expected = "1.0" },
        .{ .value = .{ .number_string = "1e400" }, .expected = "Infinity" },
        .{ .value = .{ .float = std.math.inf(f64) }, .expected = "Infinity" },
        .{ .value = .{ .float = -std.math.inf(f64) }, .expected = "-Infinity" },
        .{ .value = .{ .float = std.math.nan(f64) }, .expected = "NaN" },
    }) |case| {
        const actual = try canonicalJson(allocator, case.value);
        defer allocator.free(actual);
        try std.testing.expectEqualStrings(case.expected, actual);
    }
    try std.testing.expectError(error.InvalidBenchmarkNumber, canonicalJson(allocator, .{ .number_string = "01" }));
    try std.testing.expectError(error.InvalidBenchmarkString, canonicalJson(allocator, .{ .string = "\xff" }));
    try std.testing.expectError(error.InvalidBenchmarkCacheKey, cacheKey(allocator, .null));
}
