const std = @import("std");

fn isDigits(value: []const u8) bool {
    if (value.len == 0) return false;
    for (value) |char| if (!std.ascii.isDigit(char)) {
        return false;
    };
    return true;
}

fn validCore(value: []const u8) bool {
    var fields = std.mem.splitScalar(u8, value, '.');
    var count: usize = 0;
    while (fields.next()) |field| {
        if (!isDigits(field) or (field.len > 1 and field[0] == '0')) return false;
        count += 1;
    }
    return count == 3;
}

fn validIdentifiers(value: []const u8, prerelease: bool) bool {
    var fields = std.mem.splitScalar(u8, value, '.');
    while (fields.next()) |field| {
        if (field.len == 0) return false;
        for (field) |char| {
            if (!std.ascii.isAlphanumeric(char) and char != '-') return false;
        }
        if (prerelease and isDigits(field) and field.len > 1 and field[0] == '0') return false;
    }
    return true;
}

pub fn releaseVersion(raw: []const u8) ![]const u8 {
    const version = if (std.mem.startsWith(u8, raw, "v")) raw[1..] else raw;
    const plus = std.mem.indexOfScalar(u8, version, '+') orelse version.len;
    if (plus < version.len and !validIdentifiers(version[plus + 1 ..], false)) return error.InvalidReleaseVersion;
    const before_build = version[0..plus];
    const dash = std.mem.indexOfScalar(u8, before_build, '-') orelse before_build.len;
    if (!validCore(before_build[0..dash])) return error.InvalidReleaseVersion;
    if (dash < before_build.len and !validIdentifiers(before_build[dash + 1 ..], true)) return error.InvalidReleaseVersion;
    return version;
}

pub fn toPep440(allocator: std.mem.Allocator, version: []const u8) ![]const u8 {
    const marker = std.mem.indexOf(u8, version, "-dev.") orelse return version;
    const number = version[marker + "-dev.".len ..];
    if (!validCore(version[0..marker]) or !isDigits(number)) return version;
    return std.fmt.allocPrint(allocator, "{s}.dev{s}", .{ version[0..marker], number });
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 2) {
        std.debug.print("usage: release-version VERSION\n", .{});
        std.process.exit(1);
    }
    const version = releaseVersion(args[1]) catch {
        std.debug.print("invalid release version: {s}\n", .{args[1]});
        std.process.exit(1);
    };
    const pep440 = try toPep440(allocator, version);
    const output = try std.fmt.allocPrint(allocator, "version={s}\npep440={s}\n", .{ version, pep440 });
    try std.Io.File.stdout().writeStreamingAll(init.io, output);
}

test "release versions retain SemVer validation and PEP 440 conversion" {
    const version = try releaseVersion("v3.0.0-dev.14");
    try std.testing.expectEqualStrings("3.0.0-dev.14", version);
    const pep = try toPep440(std.testing.allocator, version);
    defer std.testing.allocator.free(pep);
    try std.testing.expectEqualStrings("3.0.0.dev14", pep);
    for ([_][]const u8{ "3.0.0", "3.0.0-rc.1", "3.0.0+build.4", "3.0.0-rc.1+build.04", "3.0.0-rc-dev.1" }) |valid| {
        try std.testing.expectEqualStrings(valid, try releaseVersion(valid));
        try std.testing.expectEqualStrings(valid, try toPep440(std.testing.allocator, valid));
    }
    for ([_][]const u8{ "", "main", "../3.0.0", "3.0.0\n", "3.0.0;echo", "03.0.0", "3.0.0-dev.01", "3.0.0/evil", "vv3.0.0", "3.0.0+", "3.0.0-rc..1" }) |invalid| {
        try std.testing.expectError(error.InvalidReleaseVersion, releaseVersion(invalid));
    }
}
