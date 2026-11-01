// Copyright (C) 2019 Intel Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

const std = @import("std");

const Version = [3][]const u8;

fn digits(text: []const u8) ?[]const u8 {
    var len: usize = 0;
    while (len < text.len and std.ascii.isDigit(text[len])) : (len += 1) {}
    return if (len == 0) null else text[0..len];
}

fn versionFromHeader(header: []const u8) !Version {
    const names = [3][]const u8{
        "WAMR_VERSION_MAJOR ",
        "WAMR_VERSION_MINOR ",
        "WAMR_VERSION_PATCH ",
    };
    var values: [3]?[]const u8 = .{ null, null, null };
    var lines = std.mem.splitScalar(u8, header, '\n');
    while (lines.next()) |line| {
        for (names, 0..) |name, i| {
            if (std.mem.indexOf(u8, line, name)) |at| {
                if (digits(line[at + name.len ..])) |value| values[i] = value;
            }
        }
    }
    return .{
        values[0] orelse return error.MissingVersion,
        values[1] orelse return error.MissingVersion,
        values[2] orelse return error.MissingVersion,
    };
}

fn versionFromTag(tag: []const u8) !Version {
    if (!std.mem.startsWith(u8, tag, "WAMR-")) return error.InvalidTag;
    var tail = tag["WAMR-".len..];
    var version: Version = undefined;
    for (0..3) |i| {
        version[i] = digits(tail) orelse return error.InvalidTag;
        tail = tail[version[i].len..];
        if (i < 2) {
            if (tail.len == 0 or tail[0] != '.') return error.InvalidTag;
            tail = tail[1..];
        }
    }
    return version;
}

fn compareNumber(a: []const u8, b: []const u8) std.math.Order {
    const left = std.mem.trimStart(u8, a, "0");
    const right = std.mem.trimStart(u8, b, "0");
    if (left.len != right.len) return std.math.order(left.len, right.len);
    return std.mem.order(u8, left, right);
}

fn compareVersion(a: Version, b: Version) std.math.Order {
    for (a, b) |left, right| {
        const order = compareNumber(left, right);
        if (order != .eq) return order;
    }
    return .eq;
}

fn result(allocator: std.mem.Allocator, definition: Version, tag: Version) ![]const u8 {
    if (compareVersion(definition, tag) != .gt) return ",patch_change\n";
    const major_minor_changed = compareNumber(definition[0], tag[0]) != .eq or
        compareNumber(definition[1], tag[1]) != .eq;
    return std.fmt.allocPrint(allocator, "{s}.{s}.{s},{s}\n", .{
        definition[0],
        definition[1],
        definition[2],
        if (major_minor_changed) "major_minor_change" else "patch_change",
    });
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const header = try std.Io.Dir.cwd().readFileAlloc(init.io, "core/version.h", allocator, .limited(1024 * 1024));
    const definition = try versionFromHeader(header);
    const command = try std.process.run(allocator, init.io, .{
        .argv = &.{ "git", "describe", "--tags", "--abbrev=0", "HEAD" },
    });
    if (command.term != .exited or command.term.exited != 0) {
        std.debug.print("git describe failed: {s}\n", .{command.stderr});
        return error.GitDescribeFailed;
    }
    const tag = try versionFromTag(std.mem.trim(u8, command.stdout, " \t\r\n"));
    try std.Io.File.stdout().writeStreamingAll(init.io, try result(allocator, definition, tag));
}

test "header and tag parsing" {
    const header = "#define WAMR_VERSION_MAJOR 01\n#define WAMR_VERSION_MINOR 2\n#define WAMR_VERSION_PATCH 34\n";
    const parsed = try versionFromHeader(header);
    try std.testing.expectEqualStrings("01", parsed[0]);
    try std.testing.expectEqualStrings("2", parsed[1]);
    try std.testing.expectEqualStrings("34", parsed[2]);
    try std.testing.expectError(error.MissingVersion, versionFromHeader(header[0..30]));
    try std.testing.expectError(error.InvalidTag, versionFromTag("v1.2.3"));
    try std.testing.expectEqualStrings("3", (try versionFromTag("WAMR-1.2.3-rc1"))[2]);
}

test "version output preserves comparison and change type" {
    const allocator = std.testing.allocator;
    const patch = try result(allocator, .{ "1", "2", "10" }, .{ "1", "2", "9" });
    defer allocator.free(patch);
    try std.testing.expectEqualStrings("1.2.10,patch_change\n", patch);
    const minor = try result(allocator, .{ "1", "3", "0" }, .{ "1", "2", "99" });
    defer allocator.free(minor);
    try std.testing.expectEqualStrings("1.3.0,major_minor_change\n", minor);
    try std.testing.expectEqualStrings(",patch_change\n", try result(allocator, .{ "01", "2", "3" }, .{ "1", "02", "3" }));
    try std.testing.expectEqual(.gt, compareNumber("1000000000000000000000000", "9"));
}
