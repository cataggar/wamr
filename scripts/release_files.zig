//! Shared release hashing, source identity, and JSON helpers.
const std = @import("std");
const process_control = @import("process_control.zig");
const Allocator = std.mem.Allocator;

pub fn digest(data: []const u8) [64]u8 {
    var bytes: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &bytes, .{});
    return std.fmt.bytesToHex(bytes, .lower);
}

pub fn fileDigest(io: std.Io, path: []const u8) ![64]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader_buffer: [8192]u8 = undefined;
    var reader = file.reader(io, &reader_buffer);
    var buffer: [64 * 1024]u8 = undefined;
    var state = std.crypto.hash.sha2.Sha256.init(.{});
    while (true) {
        const n = try reader.interface.readSliceShort(&buffer);
        if (n == 0) break;
        state.update(buffer[0..n]);
    }
    var bytes: [32]u8 = undefined;
    state.final(&bytes);
    return std.fmt.bytesToHex(bytes, .lower);
}

pub fn checkedSource(allocator: Allocator, io: std.Io, repo: []const u8, sha: []const u8) !void {
    const result = try process_control.runTimed(allocator, io, .{ .argv = &.{ "git", "-C", repo, "rev-parse", "HEAD" } }, 120);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("[release] git rev-parse failed ({f}): {s}\n", .{ result.term, result.stderr });
        return error.SourceRevisionUnavailable;
    }
    if (!std.mem.eql(u8, std.mem.trim(u8, result.stdout, " \t\r\n"), sha)) return error.SourceRevisionMismatch;
}

pub fn jsonValue(allocator: Allocator, value: anytype) !std.json.Value {
    const text = try std.json.Stringify.valueAlloc(allocator, value, .{});
    return std.json.parseFromSliceLeaky(std.json.Value, allocator, text, .{});
}
