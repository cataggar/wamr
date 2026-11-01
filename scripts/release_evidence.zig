//! Validate release smoke-test and Windows signature evidence.
const std = @import("std");
const binary = @import("release_binary.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const Expected = struct {
    target: binary.Target,
    version: []const u8,
    fixture_sha256: [2][]const u8,
    stdout_sha256: [2][]const u8,
    windows_binary_sha256: [2][]const u8,
};

fn object(value: Value) !std.json.ObjectMap {
    return if (value == .object) value.object else error.InvalidReleaseEvidence;
}

fn array(value: Value) ![]const Value {
    return if (value == .array) value.array.items else error.InvalidReleaseEvidence;
}

fn field(map: std.json.ObjectMap, name: []const u8) !Value {
    return map.get(name) orelse error.InvalidReleaseEvidence;
}

fn text(value: Value) ![]const u8 {
    return if (value == .string) value.string else error.InvalidReleaseEvidence;
}

fn equals(map: std.json.ObjectMap, name: []const u8, expected: []const u8) !bool {
    return std.mem.eql(u8, try text(try field(map, name)), expected);
}

fn isNative(host: []const Value, target: binary.Target) !bool {
    if (host.len != 2) return false;
    return std.mem.eql(u8, try text(host[0]), @tagName(target.system)) and
        std.mem.eql(u8, try text(host[1]), @tagName(target.arch));
}

fn validateSmoke(allocator: Allocator, root: std.json.ObjectMap, expected: Expected) !void {
    const smoke = try object(try field(root, "native_smoke"));
    const host = try array(try field(root, "host"));
    const native = try isNative(host, expected.target);
    if (try equals(smoke, "status", "passed")) {
        if (!native) return error.NonNativeSmokeClaim;
        const versions = try object(try field(smoke, "versions"));
        if (versions.count() != 3) return error.InvalidSmokeVersions;
        const wamr = try std.fmt.allocPrint(allocator, "wamr {s}\n", .{expected.version});
        defer allocator.free(wamr);
        const wamrc = try std.fmt.allocPrint(allocator, "wamrc {s}\n", .{expected.version});
        defer allocator.free(wamrc);
        if (!try equals(versions, "wamr version", wamr) or
            !try equals(versions, "wamr --version", wamr) or
            !try equals(versions, "wamrc version", wamrc))
            return error.InvalidSmokeVersions;

        const cases = try array(try field(smoke, "cases"));
        if (cases.len != 2) return error.InvalidSmokeCases;
        const fixtures = [_][]const u8{
            "tests/coldstart/noop.wasm",
            "tests/regressions/1008-aot-magic-u32/printf-decimal.wasm",
        };
        for (cases, 0..) |case, index| {
            const data = try object(case);
            if (data.count() != 3 or
                !try equals(data, "fixture", fixtures[index]) or
                !try equals(data, "fixture_sha256", expected.fixture_sha256[index]) or
                !try equals(data, "stdout_sha256", expected.stdout_sha256[index]))
                return error.InvalidSmokeCases;
        }
    } else {
        if (expected.target.required_native or native or smoke.count() != 2 or
            !try equals(smoke, "status", "not-run") or
            !try equals(smoke, "reason", "host-target-mismatch"))
            return error.MissingNativeSmoke;
    }
}

fn validateSignatures(root: std.json.ObjectMap, expected: Expected) !void {
    if (expected.target.system != .windows) return;
    const signatures = try array(try field(root, "signatures"));
    if (signatures.len != 2) return error.InvalidWindowsSignatures;
    var seen: [2]bool = @splat(false);
    for (signatures) |value| {
        const signature = try object(value);
        const name = try text(try field(signature, "name"));
        const index: usize = if (std.mem.eql(u8, name, "wamr.exe"))
            0
        else if (std.mem.eql(u8, name, "wamrc.exe"))
            1
        else
            return error.InvalidWindowsSignatures;
        if (seen[index] or
            !try equals(signature, "status", "Valid") or
            !try equals(signature, "sha256", expected.windows_binary_sha256[index]) or
            (try text(try field(signature, "signer"))).len == 0 or
            (try text(try field(signature, "timestamp_signer"))).len == 0)
            return error.InvalidWindowsSignatures;
        seen[index] = true;
    }
}

pub fn validate(allocator: Allocator, report: Value, expected: Expected) !void {
    const root = try object(report);
    try validateSmoke(allocator, root, expected);
    try validateSignatures(root, expected);
}

test "native smoke evidence requires exact versions, fixture hashes, and host" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const source =
        \\{"host":["linux","x86_64"],"native_smoke":{"status":"passed",
        \\"versions":{"wamr version":"wamr 1.2.3\n","wamr --version":"wamr 1.2.3\n","wamrc version":"wamrc 1.2.3\n"},
        \\"cases":[{"fixture":"tests/coldstart/noop.wasm","fixture_sha256":"a","stdout_sha256":"b"},
        \\{"fixture":"tests/regressions/1008-aot-magic-u32/printf-decimal.wasm","fixture_sha256":"c","stdout_sha256":"d"}]}}
    ;
    const report = try std.json.parseFromSliceLeaky(Value, allocator, source, .{});
    var expected: Expected = .{
        .target = binary.target("linux-x64").?,
        .version = "1.2.3",
        .fixture_sha256 = .{ "a", "c" },
        .stdout_sha256 = .{ "b", "d" },
        .windows_binary_sha256 = .{ "", "" },
    };
    try validate(allocator, report, expected);
    expected.version = "1.2.4";
    try std.testing.expectError(error.InvalidSmokeVersions, validate(allocator, report, expected));
    expected.version = "1.2.3";
    expected.fixture_sha256[1] = "wrong";
    try std.testing.expectError(error.InvalidSmokeCases, validate(allocator, report, expected));
    expected.fixture_sha256[1] = "c";
    expected.target = binary.target("linux-arm64").?;
    try std.testing.expectError(error.NonNativeSmokeClaim, validate(allocator, report, expected));
}

test "required native smoke and timestamped Windows signatures cannot be omitted" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const source =
        \\{"host":["linux","x86_64"],"native_smoke":{"status":"not-run","reason":"host-target-mismatch"},
        \\"signatures":[{"name":"wamr.exe","status":"Valid","sha256":"a","signer":"s","timestamp_signer":"t"},
        \\{"name":"wamrc.exe","status":"Valid","sha256":"b","signer":"s","timestamp_signer":"t"}]}
    ;
    var report = try std.json.parseFromSliceLeaky(Value, allocator, source, .{});
    var expected: Expected = .{
        .target = binary.target("windows-arm64").?,
        .version = "1.2.3",
        .fixture_sha256 = .{ "", "" },
        .stdout_sha256 = .{ "", "" },
        .windows_binary_sha256 = .{ "a", "b" },
    };
    try validate(allocator, report, expected);
    expected.windows_binary_sha256[1] = "wrong";
    try std.testing.expectError(error.InvalidWindowsSignatures, validate(allocator, report, expected));
    expected.windows_binary_sha256[1] = "b";
    const signatures = report.object.getPtr("signatures").?.array.items;
    try signatures[0].object.put(allocator, "timestamp_signer", .{ .string = "" });
    try std.testing.expectError(error.InvalidWindowsSignatures, validate(allocator, report, expected));
    try signatures[0].object.put(allocator, "timestamp_signer", .{ .string = "t" });
    try signatures[1].object.put(allocator, "name", .{ .string = "wamr.exe" });
    try std.testing.expectError(error.InvalidWindowsSignatures, validate(allocator, report, expected));
    try signatures[1].object.put(allocator, "name", .{ .string = "wamrc.exe" });
    expected.target = binary.target("windows-x64").?;
    try std.testing.expectError(error.MissingNativeSmoke, validate(allocator, report, expected));
}
