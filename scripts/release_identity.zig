//! Validate release report identity against inspected archive contents.
const std = @import("std");
const archive = @import("release_archive.zig");
const binary = @import("release_binary.zig");
const version = @import("release_version.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

fn object(value: Value) !std.json.ObjectMap {
    return if (value == .object) value.object else error.InvalidReleaseIdentity;
}

fn field(map: std.json.ObjectMap, name: []const u8) !Value {
    return map.get(name) orelse error.InvalidReleaseIdentity;
}

fn textEquals(map: std.json.ObjectMap, name: []const u8, expected: []const u8) !bool {
    const value = try field(map, name);
    return value == .string and std.mem.eql(u8, value.string, expected);
}

fn hash(data: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

/// The inspector must have received and validated every archive member.
pub fn validateReport(
    allocator: Allocator,
    report: Value,
    inspected: *const archive.Inspector,
    release_version: []const u8,
    source_sha: []const u8,
    archive_name: []const u8,
    archive_sha256: []const u8,
) !void {
    try inspected.finish();
    const package_name = try std.fmt.allocPrint(allocator, "wamr-{s}-{s}", .{ release_version, inspected.target.name });
    defer allocator.free(package_name);
    const expected_archive = try std.fmt.allocPrint(allocator, "{s}{s}", .{
        package_name, if (inspected.target.system == .windows) ".zip" else ".tar.gz",
    });
    defer allocator.free(expected_archive);
    if (!std.mem.eql(u8, inspected.root, package_name) or !std.mem.eql(u8, archive_name, expected_archive))
        return error.InvalidReleaseIdentity;
    const root = try object(report);
    const schema = try field(root, "schema_version");
    const pep440 = try version.toPep440(allocator, release_version);
    defer if (pep440.ptr != release_version.ptr) allocator.free(pep440);
    if (schema != .integer or schema.integer != 1 or
        !try textEquals(root, "version", release_version) or
        !try textEquals(root, "pep440", pep440) or
        !try textEquals(root, "source_sha", source_sha) or
        !try textEquals(root, "platform", inspected.target.name) or
        !try textEquals(root, "archive", archive_name) or
        !try textEquals(root, "archive_sha256", archive_sha256))
        return error.InvalidReleaseIdentity;

    const files = try object(try field(root, "files"));
    if (files.count() != inspected.files.len) return error.InvalidReleaseIdentity;
    for (inspected.files) |optional| {
        const member = optional.?;
        const metadata = try object(try field(files, member.name));
        const size = try field(metadata, "size");
        const digest = hash(member.data);
        if (metadata.count() != 2 or size != .integer or size.integer < 0 or
            @as(u64, @intCast(size.integer)) != member.data.len or
            !try textEquals(metadata, "sha256", &digest))
            return error.InvalidReleaseIdentity;
    }
}

pub fn validateSpdx(document: Value) !void {
    const root = try object(document);
    const namespace = try field(root, "documentNamespace");
    if (!try textEquals(root, "spdxVersion", "SPDX-2.3") or
        !try textEquals(root, "SPDXID", "SPDXRef-DOCUMENT") or
        namespace != .string or namespace.string.len == 0)
        return error.InvalidSpdxDocument;
}

/// Generated outputs may already exist, but every input asset must be present.
pub fn validateAssetNames(allocator: Allocator, names: []const []const u8, release_version: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const scratch = arena.allocator();
    var required: std.StringHashMap(void) = .init(scratch);
    var seen: std.StringHashMap(void) = .init(scratch);
    try required.put(try std.fmt.allocPrint(scratch, "wamr-{s}.tar.xz", .{release_version}), {});
    for (binary.targets) |target| {
        const stem = try std.fmt.allocPrint(scratch, "wamr-{s}-{s}", .{ release_version, target.name });
        for ([_][]const u8{
            if (target.system == .windows) ".zip" else ".tar.gz",
            ".verification.json",
            ".sbom.spdx.json",
        }) |suffix| {
            try required.put(try std.fmt.allocPrint(scratch, "{s}{s}", .{ stem, suffix }), {});
        }
    }
    for (names) |name| {
        const entry = try seen.getOrPut(name);
        if (entry.found_existing) return error.DuplicateReleaseAsset;
        if (required.remove(name)) continue;
        if (std.mem.eql(u8, name, "release-manifest.json") or
            std.mem.eql(u8, name, "SHA256SUMS") or
            std.mem.eql(u8, name, "RELEASE_NOTES.md")) continue;
        return error.UnexpectedReleaseAsset;
    }
    if (required.count() != 0) return error.MissingReleaseAsset;
}

test "report identity binds all inspected files, source, and archive digest" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var inspector = try archive.Inspector.init(allocator, "3.0.0-dev.14", binary.target("linux-x64").?, "license\n", "readme\n");
    defer inspector.deinit();
    try inspector.add("wamr-3.0.0-dev.14-linux-x64/LICENSE", .regular, 0o644, "license\n");
    try inspector.add("wamr-3.0.0-dev.14-linux-x64/README.md", .regular, 0o644, "readme\n");
    var elf: [64]u8 = @splat(0);
    @memcpy(elf[0..6], "\x7fELF\x02\x01");
    std.mem.writeInt(u16, elf[16..18], 2, .little);
    std.mem.writeInt(u16, elf[18..20], 62, .little);
    try inspector.add("wamr-3.0.0-dev.14-linux-x64/bin/wamr", .regular, 0o755, &elf);
    try inspector.add("wamr-3.0.0-dev.14-linux-x64/bin/wamrc", .regular, 0o755, &elf);

    var files: std.json.ObjectMap = .empty;
    for (inspector.files) |optional| {
        const member = optional.?;
        var metadata: std.json.ObjectMap = .empty;
        try metadata.put(allocator, "size", .{ .integer = @intCast(member.data.len) });
        const digest = hash(member.data);
        try metadata.put(allocator, "sha256", .{ .string = try allocator.dupe(u8, &digest) });
        try files.put(allocator, member.name, .{ .object = metadata });
    }
    var root: std.json.ObjectMap = .empty;
    try root.put(allocator, "schema_version", .{ .integer = 1 });
    try root.put(allocator, "version", .{ .string = "3.0.0-dev.14" });
    try root.put(allocator, "pep440", .{ .string = "3.0.0.dev14" });
    try root.put(allocator, "source_sha", .{ .string = "source" });
    try root.put(allocator, "platform", .{ .string = "linux-x64" });
    try root.put(allocator, "archive", .{ .string = "wamr-3.0.0-dev.14-linux-x64.tar.gz" });
    try root.put(allocator, "archive_sha256", .{ .string = "archive-hash" });
    try root.put(allocator, "files", .{ .object = files });
    const report: Value = .{ .object = root };
    const name = "wamr-3.0.0-dev.14-linux-x64.tar.gz";
    try validateReport(allocator, report, &inspector, "3.0.0-dev.14", "source", name, "archive-hash");
    try std.testing.expectError(error.InvalidReleaseIdentity, validateReport(allocator, report, &inspector, "3.0.0-dev.14", "wrong", name, "archive-hash"));
    try std.testing.expectError(error.InvalidReleaseIdentity, validateReport(allocator, report, &inspector, "3.0.0-dev.14", "source", name, "wrong"));
    try std.testing.expectError(error.InvalidReleaseIdentity, validateReport(allocator, report, &inspector, "3.0.0-dev.14", "source", "other.tar.gz", "archive-hash"));
    const metadata = report.object.getPtr("files").?.object.getPtr(inspector.files[0].?.name).?;
    try metadata.object.put(allocator, "size", .{ .integer = 1 });
    try std.testing.expectError(error.InvalidReleaseIdentity, validateReport(allocator, report, &inspector, "3.0.0-dev.14", "source", name, "archive-hash"));
    try metadata.object.put(allocator, "size", .{ .integer = @intCast(inspector.files[0].?.data.len) });
    try metadata.object.put(allocator, "sha256", .{ .string = "wrong" });
    try std.testing.expectError(error.InvalidReleaseIdentity, validateReport(allocator, report, &inspector, "3.0.0-dev.14", "source", name, "archive-hash"));
}

test "inventory requires all expected assets and rejects extras and duplicates" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var names: std.ArrayList([]const u8) = .empty;
    try names.append(allocator, "wamr-1.2.3.tar.xz");
    for (binary.targets) |target| {
        const stem = try std.fmt.allocPrint(allocator, "wamr-1.2.3-{s}", .{target.name});
        for ([_][]const u8{
            if (target.system == .windows) ".zip" else ".tar.gz",
            ".verification.json",
            ".sbom.spdx.json",
        }) |suffix| try names.append(allocator, try std.fmt.allocPrint(allocator, "{s}{s}", .{ stem, suffix }));
    }
    try validateAssetNames(allocator, names.items, "1.2.3");
    try std.testing.expectError(error.MissingReleaseAsset, validateAssetNames(allocator, names.items[1..], "1.2.3"));
    try names.append(allocator, "release-manifest.json");
    try validateAssetNames(allocator, names.items, "1.2.3");
    try names.append(allocator, "extra.tar.gz");
    try std.testing.expectError(error.UnexpectedReleaseAsset, validateAssetNames(allocator, names.items, "1.2.3"));
    names.items[names.items.len - 1] = "SHA256SUMS";
    try names.append(allocator, "SHA256SUMS");
    try std.testing.expectError(error.DuplicateReleaseAsset, validateAssetNames(allocator, names.items, "1.2.3"));
}

test "SPDX identity and namespace are required" {
    const allocator = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(Value, allocator,
        \\{"spdxVersion":"SPDX-2.3","SPDXID":"SPDXRef-DOCUMENT","documentNamespace":"https://example.invalid/release"}
    , .{});
    defer parsed.deinit();
    try validateSpdx(parsed.value);
    parsed.value.object.getPtr("documentNamespace").?.* = .{ .string = "" };
    try std.testing.expectError(error.InvalidSpdxDocument, validateSpdx(parsed.value));
}
