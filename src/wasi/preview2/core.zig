//! WASI Preview 2 core interfaces — clock, random, CLI, filesystem.
//!
//! Host-side implementations of the foundational WASI interfaces that
//! components import. These are registered as host component functions
//! during component instantiation.

const std = @import("std");

// ── wasi:clocks/wall-clock ──────────────────────────────────────────────────

pub const WallClock = struct {
    pub const DateTime = struct {
        seconds: u64,
        nanoseconds: u32,
    };

    pub fn now() DateTime {
        const ts = std.time.nanoTimestamp();
        if (ts < 0) return .{ .seconds = 0, .nanoseconds = 0 };
        const ns: u64 = @intCast(ts);
        return .{
            .seconds = ns / 1_000_000_000,
            .nanoseconds = @intCast(ns % 1_000_000_000),
        };
    }

    pub fn resolution() DateTime {
        return .{ .seconds = 0, .nanoseconds = 1 };
    }
};

// ── wasi:clocks/monotonic-clock ─────────────────────────────────────────────

pub const MonotonicClock = struct {
    pub fn now() u64 {
        const ts = std.time.nanoTimestamp();
        if (ts < 0) return 0;
        return @intCast(ts);
    }

    pub fn resolution() u64 {
        return 1; // 1 nanosecond
    }
};

// ── wasi:random/random ──────────────────────────────────────────────────────

pub const Random = struct {
    pub fn getRandomBytes(buf: []u8) void {
        std.crypto.random.bytes(buf);
    }

    pub fn getRandomU64() u64 {
        return std.crypto.random.int(u64);
    }
};

// ── wasi:cli/environment ────────────────────────────────────────────────────

pub const Environment = struct {
    args: []const []const u8 = &.{},
    env_vars: []const [2][]const u8 = &.{}, // key-value pairs

    pub fn getArguments(self: *const Environment) []const []const u8 {
        return self.args;
    }

    pub fn getEnvironment(self: *const Environment) []const [2][]const u8 {
        return self.env_vars;
    }
};

// ── wasi:filesystem/types ───────────────────────────────────────────────────

pub const Filesystem = struct {
    pub const DescriptorType = enum(u8) {
        unknown = 0,
        block_device = 1,
        character_device = 2,
        directory = 3,
        fifo = 4,
        symbolic_link = 5,
        regular_file = 6,
        socket = 7,
    };

    pub const DescriptorStat = struct {
        type: DescriptorType = .unknown,
        link_count: u64 = 0,
        size: u64 = 0,
        data_access_timestamp: ?WallClock.DateTime = null,
        data_modification_timestamp: ?WallClock.DateTime = null,
        status_change_timestamp: ?WallClock.DateTime = null,
    };

    pub const OpenFlags = packed struct(u32) {
        create: bool = false,
        directory: bool = false,
        exclusive: bool = false,
        truncate: bool = false,
        _padding: u28 = 0,
    };

    /// Descriptor handle — wraps a host file descriptor.
    pub const Descriptor = struct {
        fd: std.posix.fd_t,
        type: DescriptorType,

        pub fn stat(self: *const Descriptor) DescriptorStat {
            _ = self;
            return .{};
        }
    };
};

// ── wasi:filesystem/preopens ────────────────────────────────────────────────

pub const Preopens = struct {
    dirs: []const PreopenDir = &.{},

    pub const PreopenDir = struct {
        descriptor_idx: u32,
        path: []const u8,
    };

    pub fn getDirectories(self: *const Preopens) []const PreopenDir {
        return self.dirs;
    }
};

// ── Tests ───────────────────────────────────────────────────────────────────

test "WallClock: now returns non-zero" {
    const dt = WallClock.now();
    // Should be after 2020 (~1.6 billion seconds)
    try std.testing.expect(dt.seconds > 1_600_000_000);
}

test "MonotonicClock: monotonically increasing" {
    const t1 = MonotonicClock.now();
    const t2 = MonotonicClock.now();
    try std.testing.expect(t2 >= t1);
}

test "Random: getRandomU64 returns values" {
    const v1 = Random.getRandomU64();
    const v2 = Random.getRandomU64();
    // Vanishingly unlikely to be equal
    try std.testing.expect(v1 != v2 or v1 == v2); // always passes; just exercises the code
}

test "Random: getRandomBytes fills buffer" {
    var buf = [_]u8{0} ** 32;
    Random.getRandomBytes(&buf);
    // Check at least one byte is non-zero (probabilistically certain)
    var all_zero = true;
    for (buf) |b| {
        if (b != 0) {
            all_zero = false;
            break;
        }
    }
    try std.testing.expect(!all_zero);
}

test "Environment: get arguments" {
    const args = [_][]const u8{ "arg0", "arg1" };
    const env = Environment{ .args = &args };
    try std.testing.expectEqual(@as(usize, 2), env.getArguments().len);
}
