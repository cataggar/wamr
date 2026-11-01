//! Portable forced termination and nonblocking child status checks.
const std = @import("std");
const builtin = @import("builtin");

pub fn forceKill(child: *std.process.Child) !void {
    const id = child.id orelse return;
    if (builtin.os.tag == .windows) {
        const windows = std.os.windows;
        switch (windows.ntdll.NtTerminateProcess(id, @fromBackingInt(1))) {
            .SUCCESS, .PROCESS_IS_TERMINATING => {},
            else => return error.ProcessTerminationFailed,
        }
    } else {
        try std.posix.kill(id, .KILL);
    }
}

pub fn pollChild(child: *std.process.Child, io: std.Io) !?std.process.Child.Term {
    if (builtin.os.tag == .windows) {
        const windows = std.os.windows;
        const immediate: windows.LARGE_INTEGER = 0;
        return switch (windows.ntdll.NtWaitForSingleObject(child.id.?, .FALSE, &immediate)) {
            .WAIT_0 => try child.wait(io),
            .TIMEOUT => null,
            else => error.ProcessWaitFailed,
        };
    } else {
        var status: if (builtin.link_libc) c_int else i32 = undefined;
        const rc = std.posix.system.wait4(child.id.?, &status, std.posix.W.NOHANG, null);
        switch (std.posix.errno(rc)) {
            .SUCCESS => {},
            .INTR => return null,
            else => return error.ProcessWaitFailed,
        }

        if (rc == 0) return null;
        if (child.stdin) |file| file.close(io);
        if (child.stdout) |file| file.close(io);
        if (child.stderr) |file| file.close(io);
        child.stdin = null;
        child.stdout = null;
        child.stderr = null;
        child.id = null;
        const bits: u32 = @bitCast(status);
        return if (std.posix.W.IFEXITED(bits))
            .{ .exited = std.posix.W.EXITSTATUS(bits) }
        else if (std.posix.W.IFSIGNALED(bits))
            .{ .signal = std.posix.W.TERMSIG(bits) }
        else if (std.posix.W.IFSTOPPED(bits))
            .{ .stopped = std.posix.W.STOPSIG(bits) }
        else
            .{ .unknown = bits };
    }
}

/// Collect a child's output and enforce a deadline, including forced cleanup.
pub fn runTimed(allocator: std.mem.Allocator, io: std.Io, options: std.process.RunOptions, seconds: u32) !std.process.RunResult {
    var child = try std.process.spawn(io, .{
        .exe = options.exe,
        .argv = options.argv,
        .cwd = options.cwd,
        .environ_map = options.environ_map,
        .expand_arg0 = options.expand_arg0,
        .progress_node = options.progress_node,
        .create_no_window = options.create_no_window,
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .pipe,
    });
    defer if (child.id != null) {
        forceKill(&child) catch |err| std.debug.print("failed to terminate runner: {s}\n", .{@errorName(err)});
        child.kill(io);
    };
    const deadline: std.Io.Timeout = .{ .deadline = .fromNow(io, .{ .clock = .awake, .raw = .fromSeconds(seconds) }) };
    var buffer: std.Io.File.MultiReader.Buffer(2) = undefined;
    var reader: std.Io.File.MultiReader = undefined;
    reader.init(allocator, io, buffer.toStreams(), &.{ child.stdout.?, child.stderr.? });
    defer reader.deinit();
    while (reader.fill(options.reserve_amount, deadline)) |_| {
        if (options.stdout_limit.toInt()) |limit| if (reader.reader(0).buffered().len > limit) return error.StreamTooLong;
        if (options.stderr_limit.toInt()) |limit| if (reader.reader(1).buffered().len > limit) return error.StreamTooLong;
    } else |err| switch (err) {
        error.EndOfStream => {},
        else => return err,
    }
    try reader.checkAnyError();
    const term = while (true) {
        if (try pollChild(&child, io)) |result| break result;
        if (deadline.deadline.durationFromNow(io).raw.nanoseconds <= 0) return error.Timeout;
        try io.sleep(.fromMilliseconds(10), .awake);
    };
    const stdout = try reader.toOwnedSlice(0);
    errdefer allocator.free(stdout);
    const stderr = try reader.toOwnedSlice(1);
    return .{ .term = term, .stdout = stdout, .stderr = stderr };
}
