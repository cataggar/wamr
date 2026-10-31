//! Decompress release inputs with external tools that verify stream checksums.
const std = @import("std");
const process_control = @import("process_control.zig");

pub fn decompress(allocator: std.mem.Allocator, io: std.Io, tool: []const u8, path: []const u8, max_bytes: usize, timeout_seconds: u32) ![]u8 {
    const result = process_control.runTimed(allocator, io, .{
        .argv = &.{ tool, "--decompress", "--stdout", "--", path },
        .stdout_limit = .limited(max_bytes),
        .stderr_limit = .limited(64 * 1024),
    }, timeout_seconds) catch |err| {
        std.debug.print("[release] {s} failed: {s}\n", .{ tool, @errorName(err) });
        return err;
    };
    defer allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("[release] {s} failed ({f}): {s}\n", .{ tool, result.term, result.stderr });
        allocator.free(result.stdout);
        return error.DecompressionFailed;
    }
    return result.stdout;
}
