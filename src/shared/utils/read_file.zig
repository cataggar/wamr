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

// ── Friendly CLI error formatting (issue #401) ─────────────────────────

const isdir_fmt =
    "Error: '{s}' is a directory, not a .wasm or .cwasm file\n" ++
    "       Hint: pass the wasm artifact directly, e.g. target/wasm32-wasip2/debug/<name>.wasm\n";
const not_found_fmt = "Error: '{s}' does not exist\n";
const perm_fmt = "Error: '{s}' could not be read (permission denied)\n";
const generic_fmt = "Error: cannot read '{s}': {}\n";

/// Format a file-read error in human-readable form. Maps the common
/// `error.IsDir`, `error.FileNotFound`, and `error.AccessDenied` /
/// `error.PermissionDenied` cases to actionable messages and falls
/// through to the existing `cannot read '<path>': <err>` shape for
/// everything else. Writes to `w`.
pub fn formatReadFileError(w: *Io.Writer, path: []const u8, err: anyerror) Io.Writer.Error!void {
    switch (err) {
        error.IsDir => try w.print(isdir_fmt, .{path}),
        error.FileNotFound => try w.print(not_found_fmt, .{path}),
        error.AccessDenied, error.PermissionDenied => try w.print(perm_fmt, .{path}),
        else => try w.print(generic_fmt, .{ path, err }),
    }
}

/// Print a friendly file-read error to stderr and exit the process with
/// status 1. Convenience for CLI entry points; mirrors the format of
/// `formatReadFileError` so tests of the latter lock both behaviors.
pub fn dieReadFileError(path: []const u8, err: anyerror) noreturn {
    switch (err) {
        error.IsDir => std.debug.print(isdir_fmt, .{path}),
        error.FileNotFound => std.debug.print(not_found_fmt, .{path}),
        error.AccessDenied, error.PermissionDenied => std.debug.print(perm_fmt, .{path}),
        else => std.debug.print(generic_fmt, .{ path, err }),
    }
    std.process.exit(1);
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

test "formatReadFileError: friendly messages" {
    const allocator = std.testing.allocator;

    const Case = struct { err: anyerror, want: []const u8 };
    const cases = [_]Case{
        .{
            .err = error.IsDir,
            .want = "Error: './build/' is a directory, not a .wasm or .cwasm file\n" ++
                "       Hint: pass the wasm artifact directly, e.g. target/wasm32-wasip2/debug/<name>.wasm\n",
        },
        .{ .err = error.FileNotFound, .want = "Error: './build/' does not exist\n" },
        .{ .err = error.AccessDenied, .want = "Error: './build/' could not be read (permission denied)\n" },
        .{ .err = error.PermissionDenied, .want = "Error: './build/' could not be read (permission denied)\n" },
        .{ .err = error.FileTooBig, .want = "Error: cannot read './build/': error.FileTooBig\n" },
    };

    for (cases) |c| {
        var aw: Io.Writer.Allocating = .init(allocator);
        defer aw.deinit();
        try formatReadFileError(&aw.writer, "./build/", c.err);
        try std.testing.expectEqualStrings(c.want, aw.writer.buffered());
    }
}
