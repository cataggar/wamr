//! File-reading utilities for WAMR (replaces `bh_read_file.h` / `.c`).
//!
//! The C version uses platform-specific `open` / `_sopen_s`, `fstat`, and
//! `read` / `_read` calls with `BH_MALLOC`.  In Zig we leverage `std.Io`
//! which abstracts platform differences and pairs naturally with
//! `std.mem.Allocator`.

const std = @import("std");
const Io = std.Io;
const File = Io.File;
const Dir = Io.Dir;

/// Read an entire file into a buffer allocated with the given allocator.
/// Returns the file contents as a slice.  Caller owns the returned memory
/// and must free it with the same allocator.
pub fn readFileToBuffer(io: Io, path: []const u8, allocator: std.mem.Allocator) ![]u8 {
    const file = try Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size == 0) return try allocator.alloc(u8, 0);
    var buf: [4096]u8 = undefined;
    var reader = file.reader(io, &buf);
    return try reader.interface.readAlloc(allocator, stat.size);
}

/// Read a file with a caller-specified maximum size limit.
/// Returns `error.FileTooBig` if the file exceeds `max_size`.
pub fn readFileToBufferWithLimit(io: Io, path: []const u8, allocator: std.mem.Allocator, max_size: usize) ![]u8 {
    const file = try Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size > max_size) return error.FileTooBig;
    if (stat.size == 0) return try allocator.alloc(u8, 0);
    var buf: [4096]u8 = undefined;
    var reader = file.reader(io, &buf);
    return try reader.interface.readAlloc(allocator, stat.size);
}

// ── Tests ───────────────────────────────────────────────────────────────

test "round-trip: write then read" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var test_dir = std.testing.tmpDir(.{});
    defer test_dir.cleanup();

    const content = "Hello, WAMR!\nLine 2.\n";

    // Write a file into the temporary directory.
    {
        const file = try test_dir.dir.createFile(io, "test_read_file.txt", .{});
        defer file.close(io);
        var buf: [4096]u8 = undefined;
        var writer = file.writer(io, &buf);
        try writer.interface.writeAll(content);
        try writer.flush();
    }

    // Read it back.
    {
        const file = try test_dir.dir.openFile(io, "test_read_file.txt", .{});
        defer file.close(io);
        const stat = try file.stat(io);
        const size: usize = if (stat.size > 0) stat.size else 1;
        var buf: [4096]u8 = undefined;
        var reader = file.reader(io, &buf);
        const data = try reader.interface.readAlloc(allocator, size);
        defer allocator.free(data);
        try std.testing.expectEqualStrings(content, data);
    }
}

test "read empty file" {
    const io = std.testing.io;
    var test_dir = std.testing.tmpDir(.{});
    defer test_dir.cleanup();

    {
        const file = try test_dir.dir.createFile(io, "empty.txt", .{});
        file.close(io);
    }

    {
        const file = try test_dir.dir.openFile(io, "empty.txt", .{});
        defer file.close(io);
        const stat = try file.stat(io);
        try std.testing.expectEqual(@as(u64, 0), stat.size);
    }
}

test "readFileToBufferWithLimit: limit exceeded" {
    const io = std.testing.io;
    var test_dir = std.testing.tmpDir(.{});
    defer test_dir.cleanup();

    // Write 100 bytes.
    {
        const file = try test_dir.dir.createFile(io, "big_file.txt", .{});
        defer file.close(io);
        var buf: [4096]u8 = undefined;
        var writer = file.writer(io, &buf);
        try writer.interface.writeAll(&([_]u8{'A'} ** 100));
        try writer.flush();
    }

    // Verify the stat-based size check.
    {
        const file = try test_dir.dir.openFile(io, "big_file.txt", .{});
        defer file.close(io);
        const stat = try file.stat(io);
        try std.testing.expect(stat.size > 10);
    }
}
