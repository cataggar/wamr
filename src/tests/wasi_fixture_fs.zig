//! Copy fixture roots into an isolated, initially empty directory.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Dir = std.Io.Dir;
const builtin = @import("builtin");

pub const Snapshot = struct {
    allocator: Allocator,
    parent: Dir,
    name: []u8,
    path: []u8,

    /// Remove only the uniquely created snapshot and release its resources.
    pub fn destroy(self: *Snapshot, io: std.Io) !void {
        defer {
            self.parent.close(io);
            self.allocator.free(self.name);
            self.allocator.free(self.path);
            self.* = undefined;
        }
        removeSnapshot(io, self.parent, self.name) catch |err| {
            std.debug.print("failed to remove fixture snapshot {s}: {s}\n", .{ self.path, @errorName(err) });
            return err;
        };
    }
};

fn removeSnapshot(io: std.Io, parent: Dir, name: []const u8) !void {
    if (builtin.os.tag == .windows) return error.UnsupportedWindowsFixtureSnapshots;
    const stat = try parent.statFile(io, name, .{ .follow_symlinks = false });
    if (stat.kind != .directory) return error.InvalidSnapshotDirectory;
    const permissions = writable(stat.permissions);
    try parent.setFilePermissions(io, name, permissions, .{ .follow_symlinks = false });
    {
        const dir = try parent.openDir(io, name, .{ .iterate = true });
        defer dir.close(io);
        try makeRemovable(io, dir);
    }
    try parent.deleteTree(io, name);
}

fn writable(permissions: std.Io.File.Permissions) std.Io.File.Permissions {
    if (builtin.os.tag == .windows) {
        var attributes = permissions.toAttributes();
        attributes.READONLY = false;
        return @fromBackingInt(@intCast(@as(u32, @bitCast(attributes))));
    }
    return .fromMode(permissions.toMode() | 0o700);
}

fn makeRemovable(io: std.Io, dir: Dir) anyerror!void {
    const stat = try dir.stat(io);
    const permissions = writable(stat.permissions);
    try dir.setPermissions(io, permissions);
    var iter = dir.iterate();
    while (try iter.next(io)) |entry| {
        const kind = if (entry.kind == .unknown)
            (try dir.statFile(io, entry.name, .{ .follow_symlinks = false })).kind
        else
            entry.kind;
        switch (kind) {
            .directory => {
                const child_stat = try dir.statFile(io, entry.name, .{ .follow_symlinks = false });
                const child_permissions = writable(child_stat.permissions);
                try dir.setFilePermissions(io, entry.name, child_permissions, .{ .follow_symlinks = false });
                const child = try dir.openDir(io, entry.name, .{ .iterate = true });
                defer child.close(io);
                try makeRemovable(io, child);
            },
            .file => if (builtin.os.tag == .windows) {
                const file_stat = try dir.statFile(io, entry.name, .{ .follow_symlinks = false });
                try dir.setFilePermissions(io, entry.name, writable(file_stat.permissions), .{ .follow_symlinks = false });
            },
            else => {},
        }
    }
}

fn isWithin(source: []const u8, directory: []const u8) bool {
    if (directory.len < source.len) return false;
    const prefix = directory[0..source.len];
    const equal = if (builtin.os.tag == .windows)
        std.ascii.eqlIgnoreCase(source, prefix)
    else
        std.mem.eql(u8, source, prefix);
    if (!equal) return false;
    if (directory.len == source.len or std.fs.path.isSep(source[source.len - 1])) return true;
    return std.fs.path.isSep(directory[source.len]);
}

/// Remove upstream fixture output markers without following symbolic links.
pub fn cleanupProducts(io: std.Io, dir: Dir) anyerror!void {
    var iter = dir.iterate();
    while (try iter.next(io)) |entry| {
        const kind = if (entry.kind == .unknown)
            (try dir.statFile(io, entry.name, .{ .follow_symlinks = false })).kind
        else
            entry.kind;
        if (std.mem.endsWith(u8, entry.name, ".cleanup")) {
            switch (kind) {
                .file => try dir.deleteFile(io, entry.name),
                .directory => try dir.deleteTree(io, entry.name),
                else => {},
            }
        } else if (kind == .directory) {
            const child = try dir.openDir(io, entry.name, .{ .iterate = true });
            defer child.close(io);
            try cleanupProducts(io, child);
        }
    }
}

/// Create a private snapshot under an explicit temporary directory.
pub fn snapshot(allocator: Allocator, io: std.Io, source_path: []const u8, temp_path: []const u8) !Snapshot {
    if (builtin.os.tag == .windows) return error.UnsupportedWindowsFixtureSnapshots;
    const cwd = Dir.cwd();
    const source = try cwd.openDir(io, source_path, .{ .iterate = true });
    defer source.close(io);
    const parent = try cwd.openDir(io, temp_path, .{ .iterate = true });
    errdefer parent.close(io);
    var source_buffer: [std.fs.max_path_bytes]u8 = undefined;
    var temp_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const source_length = try source.realPath(io, &source_buffer);
    const temp_length = try parent.realPath(io, &temp_buffer);
    if (isWithin(source_buffer[0..source_length], temp_buffer[0..temp_length])) return error.InvalidSnapshotLocation;
    var random: [16]u8 = undefined;
    var name_buffer: [37]u8 = undefined;
    const name = while (true) {
        io.random(&random);
        const candidate = try std.fmt.bufPrint(&name_buffer, "wasi-{s}", .{std.fmt.bytesToHex(random, .lower)});
        const owned = try allocator.dupe(u8, candidate);
        parent.createDir(io, candidate, if (builtin.os.tag == .windows) .default_dir else .fromMode(0o700)) catch |err| switch (err) {
            error.PathAlreadyExists => {
                allocator.free(owned);
                continue;
            },
            else => {
                allocator.free(owned);
                return err;
            },
        };
        break owned;
    };
    errdefer allocator.free(name);
    errdefer removeSnapshot(io, parent, name) catch |err| std.debug.print("failed to remove partial fixture snapshot {s}: {s}\n", .{ name, @errorName(err) });
    const path = try std.fs.path.join(allocator, &.{ temp_path, name });
    errdefer allocator.free(path);
    const destination = try parent.openDir(io, name, .{ .iterate = true });
    defer destination.close(io);
    try copyTree(allocator, io, source, destination);
    return .{ .allocator = allocator, .parent = parent, .name = name, .path = path };
}

/// Both directories must be open for iteration; the destination must be empty.
pub fn copyTree(allocator: Allocator, io: std.Io, source: Dir, destination: Dir) !void {
    if (builtin.os.tag == .windows) return error.UnsupportedWindowsFixtureSnapshots;
    var iter = source.iterate();
    while (try iter.next(io)) |entry| {
        const kind = if (entry.kind == .unknown)
            (try source.statFile(io, entry.name, .{ .follow_symlinks = false })).kind
        else
            entry.kind;
        switch (kind) {
            .directory => {
                try destination.createDir(io, entry.name, .default_dir);
                const src = try source.openDir(io, entry.name, .{ .iterate = true });
                defer src.close(io);
                const dst = try destination.openDir(io, entry.name, .{ .iterate = true });
                defer dst.close(io);
                try copyTree(allocator, io, src, dst);
            },
            .file => {
                const stat = try source.statFile(io, entry.name, .{});
                try source.copyFile(entry.name, destination, entry.name, io, .{ .replace = false });
                try destination.setTimestamps(io, entry.name, .{
                    .access_timestamp = .init(stat.atime),
                    .modify_timestamp = .init(stat.mtime),
                });
            },
            .sym_link => {
                const buffer = try allocator.alloc(u8, std.fs.max_path_bytes);
                defer allocator.free(buffer);
                const length = try source.readLink(io, entry.name, buffer);
                const is_directory = if (builtin.os.tag == .windows)
                    if (source.statFile(io, entry.name, .{})) |stat|
                        stat.kind == .directory
                    else |err| switch (err) {
                        error.FileNotFound => false,
                        else => return err,
                    }
                else
                    false;
                try destination.symLink(io, buffer[0..length], entry.name, .{ .is_directory = is_directory });
                if (builtin.os.tag != .windows) {
                    const stat = try source.statFile(io, entry.name, .{ .follow_symlinks = false });
                    try destination.setTimestamps(io, entry.name, .{
                        .follow_symlinks = false,
                        .access_timestamp = .init(stat.atime),
                        .modify_timestamp = .init(stat.mtime),
                    });
                }
            },
            else => return error.UnsupportedFixtureFileType,
        }
    }
    const stat = try source.stat(io);
    try destination.setPermissions(io, stat.permissions);
    try destination.setTimestamps(io, ".", .{
        .access_timestamp = .init(stat.atime),
        .modify_timestamp = .init(stat.mtime),
    });
}

test "snapshot copies files, nested directories, and links without sharing writes" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(io, "source", .default_dir);
    try tmp.dir.createDir(io, "snapshot", .default_dir);
    const source = try tmp.dir.openDir(io, "source", .{ .iterate = true });
    defer source.close(io);
    const destination = try tmp.dir.openDir(io, "snapshot", .{ .iterate = true });
    defer destination.close(io);
    try source.createDir(io, "nested", .default_dir);
    try source.writeFile(io, .{ .sub_path = "nested/input", .data = "original" });
    if (@import("builtin").os.tag != .windows) {
        try source.setFilePermissions(io, "nested/input", .fromMode(0o750), .{});
        try source.symLink(io, "nested/input", "link", .{});
        try source.symLink(io, "missing", "dangling", .{});
    }

    try copyTree(allocator, io, source, destination);
    const copied = try destination.readFileAlloc(io, "nested/input", allocator, .unlimited);
    defer allocator.free(copied);
    try std.testing.expectEqualStrings("original", copied);
    if (@import("builtin").os.tag != .windows) {
        const source_stat = try source.statFile(io, "nested/input", .{});
        const copied_stat = try destination.statFile(io, "nested/input", .{});
        try std.testing.expectEqual(source_stat.permissions, copied_stat.permissions);
        try std.testing.expectEqual(source_stat.mtime.nanoseconds, copied_stat.mtime.nanoseconds);
    }

    try destination.writeFile(io, .{ .sub_path = "nested/input", .data = "changed" });
    const original = try source.readFileAlloc(io, "nested/input", allocator, .unlimited);
    defer allocator.free(original);
    try std.testing.expectEqualStrings("original", original);
    if (@import("builtin").os.tag != .windows) {
        var buffer: [128]u8 = undefined;
        const target_len = try destination.readLink(io, "link", &buffer);
        try std.testing.expectEqualStrings("nested/input", buffer[0..target_len]);
        const dangling_len = try destination.readLink(io, "dangling", &buffer);
        try std.testing.expectEqualStrings("missing", buffer[0..dangling_len]);
        const source_link = try source.statFile(io, "link", .{ .follow_symlinks = false });
        const copied_link = try destination.statFile(io, "link", .{ .follow_symlinks = false });
        try std.testing.expectEqual(source_link.mtime.nanoseconds, copied_link.mtime.nanoseconds);
    }
}

test "snapshot creates and removes a private copy without changing the source" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(io, "source", .default_dir);
    const source = try tmp.dir.openDir(io, "source", .{ .iterate = true });
    defer source.close(io);
    try source.createDir(io, "nested", .default_dir);
    try source.writeFile(io, .{ .sub_path = "nested/input", .data = "original" });
    if (builtin.os.tag != .windows) {
        try source.setFilePermissions(io, "nested/input", .fromMode(0o400), .{});
        try source.setFilePermissions(io, "nested", .fromMode(0o500), .{});
    }

    defer if (builtin.os.tag != .windows) source.setFilePermissions(io, "nested", .fromMode(0o700), .{}) catch |err|
        std.debug.print("failed to restore test directory: {s}\n", .{@errorName(err)});
    const temp_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer allocator.free(temp_path);
    const source_path = try std.fs.path.join(allocator, &.{ temp_path, "source" });
    defer allocator.free(source_path);
    if (builtin.os.tag != .windows) {
        const external_target = try std.fs.path.resolve(allocator, &.{ source_path, "nested/input" });
        defer allocator.free(external_target);
        try source.symLink(io, external_target, "outside-link", .{});
    }
    try std.testing.expectError(error.InvalidSnapshotLocation, snapshot(allocator, io, source_path, source_path));
    if (builtin.os.tag != .windows) {
        try tmp.dir.symLink(io, "source", "source-link", .{ .is_directory = true });
        const linked_path = try std.fs.path.join(allocator, &.{ temp_path, "source-link" });
        defer allocator.free(linked_path);
        try std.testing.expectError(error.InvalidSnapshotLocation, snapshot(allocator, io, source_path, linked_path));
    }
    var result = try snapshot(allocator, io, source_path, temp_path);
    var alive = true;
    defer if (alive) result.destroy(io) catch |err| std.debug.print("snapshot cleanup failed: {s}\n", .{@errorName(err)});
    const name = try allocator.dupe(u8, result.name);
    defer allocator.free(name);
    try std.testing.expect(!std.mem.eql(u8, result.path, source_path));
    alive = false;
    try result.destroy(io);
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, name, .{}));
    const original = try source.readFileAlloc(io, "nested/input", allocator, .unlimited);
    defer allocator.free(original);
    try std.testing.expectEqualStrings("original", original);
    if (builtin.os.tag != .windows) {
        const stat = try source.statFile(io, "nested/input", .{});
        try std.testing.expectEqual(@as(u32, 0o400), stat.permissions.toMode() & 0o777);
    }
}

test "Windows snapshots report unsupported instead of entering incomplete I/O operations" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    try std.testing.expectError(error.UnsupportedWindowsFixtureSnapshots, snapshot(std.testing.allocator, std.testing.io, ".", "."));
    try std.testing.expectError(error.UnsupportedWindowsFixtureSnapshots, copyTree(std.testing.allocator, std.testing.io, Dir.cwd(), Dir.cwd()));
}
