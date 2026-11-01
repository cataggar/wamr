//! Compiler-generation handling shared by benchmark worktree builds.
const std = @import("std");
const fixed = @import("decimal_format.zig").fixed;
const Allocator = std.mem.Allocator;

const Compiler = enum { zig016, zig };

pub fn parseModes(value: []const u8) ![]const []const u8 {
    if (std.mem.eql(u8, value, "both")) return &.{ "ReleaseFast", "ReleaseSafe" };
    if (std.mem.eql(u8, value, "ReleaseFast")) return &.{"ReleaseFast"};
    if (std.mem.eql(u8, value, "ReleaseSafe")) return &.{"ReleaseSafe"};
    return error.InvalidOptimizeMode;
}

pub fn slug(allocator: Allocator, value: []const u8) ![]u8 {
    const text = if (std.mem.startsWith(u8, value, "Release")) value["Release".len..] else value;
    for (text) |ch| if (ch >= 0x80) return error.UnsupportedOptimizeName;
    const result = try allocator.alloc(u8, text.len);
    return std.ascii.lowerString(result, text);
}

pub fn formatRatio(allocator: Allocator, numerator: ?f64, denominator: ?f64) ![]u8 {
    const top = numerator orelse return allocator.dupe(u8, "\xe2\x80\x94");
    const bottom = denominator orelse return allocator.dupe(u8, "\xe2\x80\x94");
    if (bottom == 0) return allocator.dupe(u8, "\xe2\x80\x94");
    const ratio = top / bottom;
    if (std.math.isNan(ratio)) return allocator.dupe(u8, "\xc3\x97nan");
    if (std.math.isInf(ratio)) return allocator.dupe(u8, if (ratio < 0) "\xc3\x97-inf" else "\xc3\x97inf");
    const magnitude = try fixed(allocator, @abs(ratio), 2);
    defer allocator.free(magnitude);
    const negative = @as(u64, @bitCast(ratio)) >> 63 != 0;
    return std.fmt.allocPrint(allocator, "\xc3\x97{s}{s}", .{ if (negative) "-" else "", magnitude });
}

fn skipWhitespace(text: []const u8, start: usize) usize {
    var index = start;
    while (index < text.len) {
        const width = std.unicode.utf8ByteSequenceLength(text[index]) catch unreachable;
        const codepoint = std.unicode.utf8Decode(text[index..][0..width]) catch unreachable;
        const whitespace = switch (codepoint) {
            0x09...0x0d, 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
            else => false,
        };
        if (!whitespace) break;
        index += width;
    }
    return index;
}

fn compiler(manifest: []const u8) !Compiler {
    if (!std.unicode.utf8ValidateSlice(manifest)) return error.InvalidManifestEncoding;
    const key = ".minimum_zig_version";
    var start: usize = 0;
    while (std.mem.indexOfPos(u8, manifest, start, key)) |index| {
        start = index + key.len;
        var pos = skipWhitespace(manifest, start);
        if (pos >= manifest.len or manifest[pos] != '=') continue;
        pos = skipWhitespace(manifest, pos + 1);
        if (pos >= manifest.len or manifest[pos] != '"') continue;
        pos += 1;
        const version = manifest[pos..];
        const tool: Compiler = if (std.mem.startsWith(u8, version, "0.16."))
            .zig016
        else if (std.mem.startsWith(u8, version, "0.17."))
            .zig
        else
            continue;
        if (std.mem.indexOfScalar(u8, version, '"') != null) return tool;
    }
    return error.UnsupportedZigVersion;
}

pub fn buildCommand(allocator: Allocator, io: std.Io, worktree: []const u8, optimize: []const u8) ![3][]const u8 {
    const path = try std.fs.path.join(allocator, &.{ worktree, "build.zig.zon" });
    defer allocator.free(path);
    const manifest = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(manifest);
    const tool = try compiler(manifest);
    const mode = if (tool == .zig016) optimize else if (std.mem.eql(u8, optimize, "Debug"))
        "debug"
    else if (std.mem.eql(u8, optimize, "ReleaseFast"))
        "fast"
    else if (std.mem.eql(u8, optimize, "ReleaseSafe"))
        "safe"
    else if (std.mem.eql(u8, optimize, "ReleaseSmall"))
        "small"
    else
        return error.InvalidOptimizeMode;
    return .{ if (tool == .zig016) "zig016" else "zig", "build", try std.fmt.allocPrint(allocator, "-Doptimize={s}", .{mode}) };
}

test "benchmark worktrees use the compiler declared by each reference" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    try temp.dir.writeFile(io, .{ .sub_path = "build.zig.zon", .data = ".minimum_zig_version = \"0.16.0\"" });
    const old = try buildCommand(allocator, io, root, "ReleaseFast");
    defer allocator.free(old[2]);
    try std.testing.expectEqualStrings("zig016", old[0]);
    try std.testing.expectEqualStrings("-Doptimize=ReleaseFast", old[2]);
    try temp.dir.writeFile(io, .{ .sub_path = "build.zig.zon", .data = ".minimum_zig_version = \"0.17.0\"" });
    for ([_][2][]const u8{ .{ "Debug", "debug" }, .{ "ReleaseFast", "fast" }, .{ "ReleaseSafe", "safe" }, .{ "ReleaseSmall", "small" } }) |pair| {
        const result = try buildCommand(allocator, io, root, pair[0]);
        defer allocator.free(result[2]);
        const expected = try std.fmt.allocPrint(allocator, "-Doptimize={s}", .{pair[1]});
        defer allocator.free(expected);
        try std.testing.expectEqualStrings("zig", result[0]);
        try std.testing.expectEqualStrings(expected, result[2]);
    }
    try std.testing.expectError(error.InvalidOptimizeMode, buildCommand(allocator, io, root, "other"));
    try temp.dir.writeFile(io, .{ .sub_path = "build.zig.zon", .data = ".minimum_zig_version = \"0.18.0\"" });
    try std.testing.expectError(error.UnsupportedZigVersion, buildCommand(allocator, io, root, "ReleaseFast"));
}

test "manifest matching skips invalid declarations and accepts reference whitespace" {
    try std.testing.expectEqual(Compiler.zig, try compiler(".minimum_zig_version = \"0.18.0\"\n.minimum_zig_version\xc2\xa0=\t\"0.17.dev\""));
    try std.testing.expectError(error.UnsupportedZigVersion, compiler(".minimum_zig_version = \"0.17.0"));
    try std.testing.expectError(error.InvalidManifestEncoding, compiler("\xff"));
}

test "benchmark mode selection, slugs, and ratio rendering" {
    const allocator = std.testing.allocator;
    const modes = try parseModes("both");
    try std.testing.expectEqualStrings("ReleaseFast", modes[0]);
    try std.testing.expectEqualStrings("ReleaseSafe", modes[1]);
    try std.testing.expectError(error.InvalidOptimizeMode, parseModes("Debug"));
    const suffix = try slug(allocator, "ReleaseFast");
    defer allocator.free(suffix);
    try std.testing.expectEqualStrings("fast", suffix);
    for ([_]struct { numerator: ?f64, denominator: ?f64, expected: []const u8 }{
        .{ .numerator = null, .denominator = 1, .expected = "\xe2\x80\x94" },
        .{ .numerator = 1, .denominator = 0, .expected = "\xe2\x80\x94" },
        .{ .numerator = 1, .denominator = 2, .expected = "\xc3\x970.50" },
        .{ .numerator = -0.0, .denominator = 1, .expected = "\xc3\x97-0.00" },
        .{ .numerator = -1, .denominator = 8, .expected = "\xc3\x97-0.12" },
        .{ .numerator = std.math.inf(f64), .denominator = 1, .expected = "\xc3\x97inf" },
        .{ .numerator = std.math.nan(f64), .denominator = 1, .expected = "\xc3\x97nan" },
    }) |item| {
        const actual = try formatRatio(allocator, item.numerator, item.denominator);
        defer allocator.free(actual);
        try std.testing.expectEqualStrings(item.expected, actual);
    }
}
