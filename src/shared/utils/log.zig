//! Logging subsystem for WAMR (replaces `bh_log.h` / `bh_log.c`).
//!
//! The C version supports wrapping multiple outputs into a single log message
//! without extra memory buffers – useful for resource-constrained systems.
//! This Zig port keeps the same five log levels and the compile-time gating
//! via `config.log`, while leveraging `std.io` for output.

const std = @import("std");
const config = @import("../../config.zig");

/// Log severity levels, matching the C `LogLevel` enum values exactly.
pub const Level = enum(u3) {
    fatal = 0,
    err = 1,
    warning = 2,
    debug = 3,
    verbose = 4,
};

/// The runtime verbose level.  Only messages whose level is `<=` this value
/// are emitted.  Default mirrors the C code (`BH_LOG_LEVEL_WARNING`).
var verbose_level: Level = .warning;

pub fn setVerboseLevel(level: Level) void {
    verbose_level = level;
}

pub fn getVerboseLevel() Level {
    return verbose_level;
}

/// Log a message if the current verbose level is >= the given level.
///
/// When `config.log` is `false` the entire function body is compiled away.
pub fn log(level: Level, comptime fmt: []const u8, args: anytype) void {
    if (!config.log) return;
    if (@intFromEnum(level) > @intFromEnum(verbose_level)) return;

    const prefix: []const u8 = switch (level) {
        .fatal => "[FATAL] ",
        .err => "[ERROR] ",
        .warning => "[WARN]  ",
        .debug => "[DEBUG] ",
        .verbose => "[VERB]  ",
    };

    var buffer: [256]u8 = undefined;
    const stderr = std.debug.lockStderr(&buffer);
    defer std.debug.unlockStderr();
    const w = &stderr.file_writer.interface;
    w.writeAll(prefix) catch {};
    w.print(fmt, args) catch {};
    w.writeAll("\n") catch {};
}

// ── Convenience wrappers ────────────────────────────────────────────────

pub fn fatal(comptime fmt: []const u8, args: anytype) void {
    log(.fatal, fmt, args);
}
pub fn err(comptime fmt: []const u8, args: anytype) void {
    log(.err, fmt, args);
}
pub fn warn(comptime fmt: []const u8, args: anytype) void {
    log(.warning, fmt, args);
}
pub fn debug(comptime fmt: []const u8, args: anytype) void {
    log(.debug, fmt, args);
}
pub fn verbose(comptime fmt: []const u8, args: anytype) void {
    log(.verbose, fmt, args);
}

// ── Tests ───────────────────────────────────────────────────────────────

test "default verbose level is warning" {
    const saved = verbose_level;
    defer verbose_level = saved;

    try std.testing.expectEqual(Level.warning, getVerboseLevel());
}

test "setVerboseLevel / getVerboseLevel round-trips" {
    const saved = verbose_level;
    defer verbose_level = saved;

    inline for (.{ Level.fatal, Level.err, Level.warning, Level.debug, Level.verbose }) |lvl| {
        setVerboseLevel(lvl);
        try std.testing.expectEqual(lvl, getVerboseLevel());
    }
}

test "level enum ordinals match C LogLevel values" {
    try std.testing.expectEqual(@as(u3, 0), @intFromEnum(Level.fatal));
    try std.testing.expectEqual(@as(u3, 1), @intFromEnum(Level.err));
    try std.testing.expectEqual(@as(u3, 2), @intFromEnum(Level.warning));
    try std.testing.expectEqual(@as(u3, 3), @intFromEnum(Level.debug));
    try std.testing.expectEqual(@as(u3, 4), @intFromEnum(Level.verbose));
}
