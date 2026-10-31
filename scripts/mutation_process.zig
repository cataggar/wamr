//! Bounded process-group execution for the mutation-testing helper.
const std = @import("std");
const process_control = @import("process_control.zig");
const Allocator = std.mem.Allocator;

pub const Result = struct {
    exit_code: i64,
    output: []u8,
    timed_out: bool,
};

fn signalGroup(pid: std.posix.pid_t, signal: std.posix.SIG) !void {
    std.posix.kill(-pid, signal) catch |err| switch (err) {
        error.ProcessNotFound => {},
        else => return err,
    };
}

/// Run a shell command with merged output and a dedicated process group.
/// An interrupt requests group termination and is returned after cleanup.
pub fn run(allocator: Allocator, io: std.Io, command: []const u8, timeout_ns: ?i96, interrupted: *const std.atomic.Value(bool)) !Result {
    const builtin = @import("builtin");
    if (builtin.os.tag != .linux and builtin.os.tag != .macos) return error.UnsupportedMutationPlatform;
    var fds: [2]std.posix.fd_t = undefined;
    if (std.posix.errno(std.c.pipe(&fds)) != .SUCCESS) return error.MutationPipeFailed;
    const read_file: std.Io.File = .{ .handle = fds[0], .flags = .{ .nonblocking = false } };
    const write_file: std.Io.File = .{ .handle = fds[1], .flags = .{ .nonblocking = false } };
    defer read_file.close(io);
    var write_open = true;
    defer if (write_open) write_file.close(io);
    for (fds) |fd| {
        if (std.posix.errno(std.posix.system.fcntl(fd, std.posix.F.SETFD, @as(usize, 1))) != .SUCCESS)
            return error.MutationPipeFailed;
    }

    var child = try std.process.spawn(io, .{
        .argv = &.{ "bash", "-o", "pipefail", "-c", command },
        .stdout = .{ .file = write_file },
        .stderr = .{ .file = write_file },
        .pgid = 0,
    });
    write_file.close(io);
    write_open = false;
    const pid = child.id.?;
    var group_active = true;
    defer if (group_active) {
        signalGroup(pid, .KILL) catch |err| std.debug.print("mutation command: group termination failed: {s}\n", .{@errorName(err)});
        if (child.id != null) child.kill(io);
    };
    var storage: std.Io.File.MultiReader.Buffer(1) = undefined;
    var reader: std.Io.File.MultiReader = undefined;
    reader.init(allocator, io, storage.toStreams(), &.{read_file});
    defer reader.deinit();
    const start = std.Io.Clock.awake.now(io);
    var term: ?std.process.Child.Term = null;
    var eof = false;
    var timed_out = false;
    var stopping = false;
    var killed = false;
    var stop_started: std.Io.Timestamp = undefined;
    while (true) {
        const now = std.Io.Clock.awake.now(io);
        const elapsed = start.durationTo(now).nanoseconds;
        if (!stopping and (interrupted.load(.acquire) or
            (timeout_ns != null and elapsed >= timeout_ns.?)))
        {
            timed_out = !interrupted.load(.acquire);
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
        const poll_ns: i96 = if (!stopping and timeout_ns != null)
            @intCast(@min(20 * std.time.ns_per_ms, @max(0, @as(i128, timeout_ns.?) - elapsed)))
        else
            20 * std.time.ns_per_ms;
        if (!eof) {
            const deadline: std.Io.Timeout = .{ .deadline = .fromNow(io, .{ .clock = .awake, .raw = .fromNanoseconds(poll_ns) }) };
            reader.fill(1, deadline) catch |err| switch (err) {
                error.Timeout => {},
                error.EndOfStream => eof = true,
                else => return err,
            };
            try reader.checkAnyError();
        } else {
            try io.sleep(.fromNanoseconds(poll_ns), .awake);
        }
    }
    if (interrupted.load(.acquire)) return error.Interrupted;
    const exit_code: i64 = if (timed_out) 124 else switch (term.?) {
        .exited => |code| code,
        .signal => |signal| -@as(i64, @intCast(@backingInt(signal))),
        else => return error.UnexpectedChildStatus,
    };
    const output = try reader.toOwnedSlice(0);
    group_active = false;
    return .{ .exit_code = exit_code, .output = output, .timed_out = timed_out };
}

test "merged output and pipeline status match shell semantics" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const interrupted: std.atomic.Value(bool) = .init(false);
    const result = try run(allocator, std.testing.io, "printf out; printf err >&2; false | true", 5 * std.time.ns_per_s, &interrupted);
    defer allocator.free(result.output);
    try std.testing.expectEqualStrings("outerr", result.output);
    try std.testing.expectEqual(@as(i64, 1), result.exit_code);
    try std.testing.expect(!result.timed_out);
}

test "timeout terminates a command and its child process group" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const interrupted: std.atomic.Value(bool) = .init(false);
    const result = try run(allocator, std.testing.io, "printf started; sleep 30 & wait", 100 * std.time.ns_per_ms, &interrupted);
    defer allocator.free(result.output);
    try std.testing.expectEqualStrings("started", result.output);
    try std.testing.expectEqual(@as(i64, 124), result.exit_code);
    try std.testing.expect(result.timed_out);
}

test "timeout escalates when the process group ignores termination" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const interrupted: std.atomic.Value(bool) = .init(false);
    const result = try run(allocator, std.testing.io, "trap '' TERM; printf stuck; sleep 30 & wait", 200 * std.time.ns_per_ms, &interrupted);
    defer allocator.free(result.output);
    try std.testing.expectEqualStrings("stuck", result.output);
    try std.testing.expectEqual(@as(i64, 124), result.exit_code);
    try std.testing.expect(result.timed_out);
}

test "interrupt terminates the command rather than returning success" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const interrupted: std.atomic.Value(bool) = .init(true);
    try std.testing.expectError(error.Interrupted, run(std.testing.allocator, std.testing.io, "sleep 30", null, &interrupted));
}
