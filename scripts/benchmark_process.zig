//! Interruptible subprocess execution for native benchmark tools.
const std = @import("std");
const process_control = @import("process_control.zig");
const Allocator = std.mem.Allocator;

fn signalGroup(pid: std.posix.pid_t, signal: std.posix.SIG) !void {
    std.posix.kill(-pid, signal) catch |err| switch (err) {
        error.ProcessNotFound => {},
        else => return err,
    };
}

/// Captures stdout followed by stderr, matching the existing benchmark tools.
pub fn run(
    allocator: Allocator,
    io: std.Io,
    cwd: []const u8,
    env: ?*const std.process.Environ.Map,
    args: []const []const u8,
    interrupted: *const std.atomic.Value(bool),
) ![]u8 {
    if (interrupted.load(.acquire)) return error.Interrupted;
    var child = try std.process.spawn(io, .{
        .argv = args,
        .cwd = .{ .path = cwd },
        .environ_map = env,
        .stdout = .pipe,
        .stderr = .pipe,
        .pgid = 0,
    });
    const pid = child.id.?;
    const stdout = child.stdout.?;
    const stderr = child.stderr.?;
    child.stdout = null;
    child.stderr = null;
    defer stdout.close(io);
    defer stderr.close(io);
    var active = true;
    defer if (active) {
        signalGroup(pid, .KILL) catch |err|
            std.debug.print("[harness] failed to terminate command group: {s}\n", .{@errorName(err)});
        if (child.id != null) child.kill(io);
    };
    var buffer: std.Io.File.MultiReader.Buffer(2) = undefined;
    var reader: std.Io.File.MultiReader = undefined;
    reader.init(allocator, io, buffer.toStreams(), &.{ stdout, stderr });
    defer reader.deinit();
    var term: ?std.process.Child.Term = null;
    var eof = false;
    var stopping = false;
    var killed = false;
    var stop_started: std.Io.Timestamp = undefined;
    while (true) {
        const now = std.Io.Clock.awake.now(io);
        if (!stopping and interrupted.load(.acquire)) {
            try signalGroup(pid, .TERM);
            stopping = true;
            stop_started = now;
        }
        if (stopping and !killed and stop_started.durationTo(now).nanoseconds >= 5 * std.time.ns_per_s) {
            try signalGroup(pid, .KILL);
            killed = true;
        }
        if (term == null) term = try process_control.pollChild(&child, io);
        if (term != null and eof) break;
        if (eof) {
            try io.sleep(.fromMilliseconds(20), .awake);
        } else {
            const deadline: std.Io.Timeout = .{ .deadline = .fromNow(io, .{ .clock = .awake, .raw = .fromMilliseconds(20) }) };
            reader.fill(1, deadline) catch |err| switch (err) {
                error.Timeout => {},
                error.EndOfStream => eof = true,
                else => return err,
            };
            try reader.checkAnyError();
        }
    }
    active = false;
    if (interrupted.load(.acquire)) return error.Interrupted;
    const output = try reader.toOwnedSlice(0);
    defer allocator.free(output);
    const errors = try reader.toOwnedSlice(1);
    defer allocator.free(errors);
    if (!term.?.success()) {
        std.debug.print("[harness] command failed ({f}):", .{term.?});
        for (args) |argument| std.debug.print(" {s}", .{argument});
        std.debug.print("\n[harness] cwd: {s}\n[harness] --- stdout ---\n{s}\n[harness] --- stderr ---\n{s}\n", .{ cwd, output, errors });
        return error.BenchmarkCommandFailed;
    }
    return std.mem.concat(allocator, u8, &.{ output, errors });
}

test "benchmark subprocesses preserve both streams and failure status" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const interrupted: std.atomic.Value(bool) = .init(false);
    const result = try run(allocator, io, ".", null, &.{ "/bin/sh", "-c", "printf out; printf err >&2" }, &interrupted);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("outerr", result);
    try std.testing.expectError(error.BenchmarkCommandFailed, run(allocator, io, ".", null, &.{ "/bin/sh", "-c", "exit 7" }, &interrupted));
}
