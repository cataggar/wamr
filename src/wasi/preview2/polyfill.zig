//! WASIp1 polyfill — map WASI Preview 1 calls to Preview 2 interfaces.
//!
//! Allows WASI Preview 1 modules to run on a Preview 2 host by
//! translating p1 function calls (fd_write, fd_read, etc.) to
//! their Preview 2 equivalents (output-stream.write, etc.).

const std = @import("std");
const core = @import("core.zig");
const streams = @import("streams.zig");

/// Polyfill context that bridges p1 file descriptors to p2 streams.
pub const WasiP1Polyfill = struct {
    /// Map from p1 fd number to p2 stream.
    stdin: streams.InputStream,
    stdout: streams.OutputStream,
    stderr: streams.OutputStream,
    env: core.Environment,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) WasiP1Polyfill {
        return .{
            .stdin = streams.InputStream.fromBuffer(""),
            .stdout = streams.OutputStream.toBuffer(),
            .stderr = streams.OutputStream.toBuffer(),
            .env = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *WasiP1Polyfill) void {
        self.stdout.deinit(self.allocator);
        self.stderr.deinit(self.allocator);
    }

    /// Polyfill for fd_write (p1) → output-stream.write (p2).
    /// Returns number of bytes written.
    pub fn fdWrite(self: *WasiP1Polyfill, fd: u32, data: []const u8) !u32 {
        const stream = switch (fd) {
            1 => &self.stdout,
            2 => &self.stderr,
            else => return error.BadFd,
        };
        const result = stream.write(data, self.allocator);
        return switch (result) {
            .ok => |n| @intCast(n),
            .closed => error.BadFd,
            .err => error.IoError,
        };
    }

    /// Polyfill for fd_read (p1) → input-stream.read (p2).
    pub fn fdRead(self: *WasiP1Polyfill, fd: u32, buf: []u8) !u32 {
        if (fd != 0) return error.BadFd;
        const result = self.stdin.read(buf);
        return switch (result) {
            .ok => |n| @intCast(n),
            .closed => 0,
            .err => error.IoError,
        };
    }

    /// Polyfill for clock_time_get (p1) → monotonic-clock.now (p2).
    pub fn clockTimeGet(clock_id: u32) u64 {
        return switch (clock_id) {
            0 => blk: {
                const dt = core.WallClock.now();
                break :blk dt.seconds * 1_000_000_000 + dt.nanoseconds;
            },
            1 => core.MonotonicClock.now(),
            else => 0,
        };
    }

    /// Polyfill for random_get (p1) → random.get-random-bytes (p2).
    pub fn randomGet(buf: []u8) void {
        core.Random.getRandomBytes(buf);
    }

    /// Polyfill for args_sizes_get (p1) → environment.get-arguments (p2).
    pub fn argsSizesGet(self: *const WasiP1Polyfill) struct { count: u32, buf_size: u32 } {
        const args = self.env.getArguments();
        var buf_size: u32 = 0;
        for (args) |arg| buf_size += @as(u32, @intCast(arg.len)) + 1;
        return .{ .count = @intCast(args.len), .buf_size = buf_size };
    }
};

// ── Errors ──────────────────────────────────────────────────────────────────

const PolyfillError = error{
    BadFd,
    IoError,
};

// ── Tests ───────────────────────────────────────────────────────────────────

test "WasiP1Polyfill: fd_write to stdout" {
    var pf = WasiP1Polyfill.init(std.testing.allocator);
    defer pf.deinit();

    const n = try pf.fdWrite(1, "hello");
    try std.testing.expectEqual(@as(u32, 5), n);
    try std.testing.expectEqualSlices(u8, "hello", pf.stdout.getBufferContents());
}

test "WasiP1Polyfill: fd_write to stderr" {
    var pf = WasiP1Polyfill.init(std.testing.allocator);
    defer pf.deinit();

    const n = try pf.fdWrite(2, "err");
    try std.testing.expectEqual(@as(u32, 3), n);
}

test "WasiP1Polyfill: fd_write bad fd" {
    var pf = WasiP1Polyfill.init(std.testing.allocator);
    defer pf.deinit();

    try std.testing.expectError(error.BadFd, pf.fdWrite(99, "data"));
}

test "WasiP1Polyfill: clock_time_get" {
    const ns = WasiP1Polyfill.clockTimeGet(1); // monotonic
    try std.testing.expect(ns > 0);
}

test "WasiP1Polyfill: random_get" {
    var buf = [_]u8{0} ** 8;
    WasiP1Polyfill.randomGet(&buf);
    // At least one byte should be non-zero
    var non_zero = false;
    for (buf) |b| if (b != 0) {
        non_zero = true;
        break;
    };
    try std.testing.expect(non_zero);
}
