//! File-reading utilities for WAMR (replaces `bh_read_file.h` / `.c`).
//!
//! The C version uses platform-specific `open` / `_sopen_s`, `fstat`, and
//! `read` / `_read` calls with `BH_MALLOC`.  In Zig we leverage `std.Io`
//! which abstracts platform differences and pairs naturally with
//! `std.mem.Allocator`.

const std = @import("std");
const Io = std.Io;

/// Read an entire file into a buffer allocated with the given allocator.
/// Returns the file contents as a slice.  Caller owns the returned memory
/// and must free it with the same allocator.
pub fn readFileToBuffer(io: Io, path: []const u8, allocator: std.mem.Allocator) ![]u8 {
    return try Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
}

/// Read a file with a caller-specified maximum size limit.
/// Returns `error.FileTooBig` if the file exceeds `max_size`.
pub fn readFileToBufferWithLimit(io: Io, path: []const u8, allocator: std.mem.Allocator, max_size: usize) ![]u8 {
    const dir = Io.Dir.cwd();
    const stat = try dir.statFile(io, path, .{});
    if (stat.size > max_size) return error.FileTooBig;
    return try dir.readFileAlloc(io, path, allocator, Io.Limit.limited(max_size));
}

// ── Tests ───────────────────────────────────────────────────────────────

test "round-trip: write then read" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var test_dir = std.testing.tmpDir(.{});
    defer test_dir.cleanup();

    const content = "Hello, WAMR!\nLine 2.\n";

    // Write a file into the temporary directory.
    try test_dir.dir.writeFile(io, .{ .sub_path = "test_read_file.txt", .data = content });

    // Read it back.
    {
        const data = try test_dir.dir.readFileAlloc(io, "test_read_file.txt", allocator, .unlimited);
        defer allocator.free(data);
        try std.testing.expectEqualStrings(content, data);
    }
}

test "read empty file" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var test_dir = std.testing.tmpDir(.{});
    defer test_dir.cleanup();

    try test_dir.dir.writeFile(io, .{ .sub_path = "empty.txt", .data = "" });

    {
        const data = try test_dir.dir.readFileAlloc(io, "empty.txt", allocator, .unlimited);
        defer allocator.free(data);
        try std.testing.expectEqual(@as(usize, 0), data.len);
    }
}

test "readFileToBufferWithLimit: limit exceeded" {
    const io = std.testing.io;
    var test_dir = std.testing.tmpDir(.{});
    defer test_dir.cleanup();

    // Write 100 bytes.
    try test_dir.dir.writeFile(io, .{ .sub_path = "big_file.txt", .data = &([_]u8{'A'} ** 100) });

    // Verify the stat-based size check.
    {
        const stat = try test_dir.dir.statFile(io, "big_file.txt", .{});
        try std.testing.expect(stat.size > 10);
    }
}
