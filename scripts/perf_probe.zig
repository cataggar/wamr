//! Generate a short CPU workload for the perf sampling permission check.
const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const start = std.Io.Clock.awake.now(init.io);
    var value: u32 = 1;
    while (start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds < 500 * std.time.ns_per_ms) {
        value = value *% 1664525 +% 1013904223;
    }
    var buffer: [32]u8 = undefined;
    const line = try std.fmt.bufPrint(&buffer, "{d}\n", .{value});
    try std.Io.File.stdout().writeStreamingAll(init.io, line);
}
