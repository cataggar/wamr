//! Experimental native launcher for macOS sampling profiles.
const std = @import("std");
const process_control = @import("process_control.zig");
const Allocator = std.mem.Allocator;

extern "c" fn fork() std.posix.pid_t;
extern "c" fn execv(file: [*:0]const u8, argv: [*:null]const ?[*:0]const u8) c_int;
extern "c" fn _exit(status: c_int) noreturn;

var pending_signals: std.atomic.Value(u8) = .init(0);

fn forwardSignal(signal: std.posix.SIG) callconv(.c) void {
    const bit: u8 = switch (signal) {
        .INT => 1,
        .TERM => 2,
        .HUP => 4,
        else => return,
    };
    _ = pending_signals.fetchOr(bit, .monotonic);
}

fn waitPid(pid: std.posix.pid_t, options: c_int) !?u32 {
    while (true) {
        var status: c_int = 0;
        const result = std.c.waitpid(pid, &status, options);
        switch (std.posix.errno(result)) {
            .SUCCESS => return if (result == 0) null else @as(u32, @bitCast(status)),
            .INTR => continue,
            else => return error.SampleTargetWaitFailed,
        }
    }
}

fn forwardPending(pid: std.posix.pid_t) !void {
    const pending = pending_signals.swap(0, .monotonic);
    for ([_]struct { bit: u8, signal: std.posix.SIG }{
        .{ .bit = 1, .signal = .INT },
        .{ .bit = 2, .signal = .TERM },
        .{ .bit = 4, .signal = .HUP },
    }) |item| {
        if (pending & item.bit != 0) {
            std.posix.kill(pid, item.signal) catch |err| switch (err) {
                error.ProcessNotFound => {},
                else => return err,
            };
        }
    }
}

fn waitTarget(io: std.Io, pid: std.posix.pid_t) !u32 {
    while (true) {
        // Only forward while the child is unreaped, so its PID cannot be reused.
        try forwardPending(pid);
        if (try waitPid(pid, std.posix.W.NOHANG)) |status| return status;
        try io.sleep(.fromMilliseconds(10), .awake);
    }
}

fn cleanTarget(pid: std.posix.pid_t) void {
    std.posix.kill(pid, .KILL) catch |err| switch (err) {
        error.ProcessNotFound => {},
        else => std.debug.print("sample launcher: failed to kill target: {s}\n", .{@errorName(err)}),
    };
    _ = waitPid(pid, 0) catch |err| std.debug.print("sample launcher: failed to reap target: {s}\n", .{@errorName(err)});
}

fn waitSampler(io: std.Io, sampler: *std.process.Child, timeout_ms: u32) !void {
    const start = std.Io.Clock.awake.now(io);
    while (true) {
        if (try process_control.pollChild(sampler, io)) |_| return;
        if (start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds >= @as(i128, timeout_ms) * std.time.ns_per_ms) return error.Timeout;
        try io.sleep(.fromMilliseconds(10), .awake);
    }
}

/// `sampler_prefix` is followed by PID, duration, interval, and output arguments.
pub fn capture(allocator: Allocator, io: std.Io, command: []const []const u8, output: []const u8, log: []const u8, sampler_prefix: []const []const u8, timeout_ms: u32) !u8 {
    if (command.len == 0 or sampler_prefix.len == 0) return error.MissingCommand;
    if (std.fs.path.dirname(output)) |parent| try std.Io.Dir.cwd().createDirPath(io, parent);

    const argv = try allocator.allocSentinel(?[*:0]const u8, command.len, null);
    defer allocator.free(argv);
    var initialized: usize = 0;
    defer for (argv[0..initialized]) |argument| allocator.free(std.mem.span(argument.?));
    for (command, 0..) |arg, i| {
        argv[i] = (try allocator.dupeSentinel(u8, arg, 0)).ptr;
        initialized += 1;
    }
    var candidates: std.ArrayList([:0]u8) = .empty;
    defer {
        for (candidates.items) |candidate| allocator.free(candidate);
        candidates.deinit(allocator);
    }
    if (std.mem.indexOfScalar(u8, command[0], '/') != null) {
        const candidate = try allocator.dupeSentinel(u8, command[0], 0);
        candidates.append(allocator, candidate) catch |err| {
            allocator.free(candidate);
            return err;
        };
    } else {
        const path = if (std.c.getenv("PATH")) |value| std.mem.span(value) else "/bin:/usr/bin";
        var directories = std.mem.splitScalar(u8, path, ':');
        while (directories.next()) |directory| {
            const candidate = if (directory.len == 0)
                try allocator.dupeSentinel(u8, command[0], 0)
            else
                try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ directory, command[0] }, 0);
            candidates.append(allocator, candidate) catch |err| {
                allocator.free(candidate);
                return err;
            };
        }
    }

    var old_actions: [3]std.posix.Sigaction = undefined;
    const signals = [_]std.posix.SIG{ .INT, .TERM, .HUP };
    const action: std.posix.Sigaction = .{
        .handler = .{ .handler = forwardSignal },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    pending_signals.store(0, .monotonic);
    for (signals, 0..) |signal, i| std.posix.sigaction(signal, &action, &old_actions[i]);
    defer for (signals, 0..) |signal, i| std.posix.sigaction(signal, &old_actions[i], null);

    const pid = fork();
    switch (std.posix.errno(pid)) {
        .SUCCESS => {},
        else => return error.SampleTargetForkFailed,
    }
    if (pid == 0) {
        for (signals, 0..) |signal, i| std.posix.sigaction(signal, &old_actions[i], null);
        if (std.posix.system.kill(std.posix.system.getpid(), .STOP) != 0) _exit(125);
        for (candidates.items) |candidate| _ = execv(candidate.ptr, argv.ptr);
        const message = "sample launcher: target exec failed\n";
        _ = std.c.write(2, message.ptr, message.len);
        _exit(1);
    }
    var target_alive = true;
    defer if (target_alive) cleanTarget(pid);
    const stopped = (try waitPid(pid, std.posix.W.UNTRACED)) orelse return error.SampleTargetWaitFailed;
    if (!std.posix.W.IFSTOPPED(stopped)) {
        target_alive = false;
        return 125;
    }

    const log_file = try std.Io.Dir.cwd().createFile(io, log, .{});
    defer log_file.close(io);
    const pid_text = try std.fmt.allocPrint(allocator, "{d}", .{pid});
    defer allocator.free(pid_text);
    const sampler_args = try allocator.alloc([]const u8, sampler_prefix.len + 5);
    defer allocator.free(sampler_args);
    @memcpy(sampler_args[0..sampler_prefix.len], sampler_prefix);
    @memcpy(sampler_args[sampler_prefix.len..], &[_][]const u8{ pid_text, "300", "1", "-file", output });
    var sampler = try std.process.spawn(io, .{ .argv = sampler_args, .stdout = .{ .file = log_file }, .stderr = .{ .file = log_file } });
    defer if (sampler.id != null) {
        process_control.forceKill(&sampler) catch |err| switch (err) {
            error.ProcessNotFound => {},
            else => std.debug.print("sample launcher: failed to kill sampler: {s}\n", .{@errorName(err)}),
        };
        sampler.kill(io);
    };
    try io.sleep(.fromMilliseconds(250), .awake);
    try forwardPending(pid);
    try std.posix.kill(pid, .CONT);
    const status = try waitTarget(io, pid);
    target_alive = false;

    waitSampler(io, &sampler, timeout_ms) catch |err| switch (err) {
        error.Timeout => {
            std.posix.kill(sampler.id.?, .TERM) catch |kill_err| switch (kill_err) {
                error.ProcessNotFound => {},
                else => return kill_err,
            };
            waitSampler(io, &sampler, timeout_ms) catch |second| switch (second) {
                error.Timeout => return error.SampleProfilerDidNotTerminate,
                else => return second,
            };
        },
        else => return err,
    };
    const stat = std.Io.Dir.cwd().statFile(io, output, .{}) catch |err| switch (err) {
        error.FileNotFound => return 125,
        else => return err,
    };
    if (stat.kind != .file or stat.size == 0) return 125;
    if (std.posix.W.IFEXITED(status)) return std.posix.W.EXITSTATUS(status);
    if (std.posix.W.IFSIGNALED(status)) return @intCast(128 + @backingInt(std.posix.W.TERMSIG(status)));
    return 125;
}

pub fn main(init: std.process.Init) !void {
    if (@import("builtin").os.tag != .macos) {
        std.debug.print("sample launcher: macOS is required\n", .{});
        std.process.exit(2);
    }
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    var output: ?[]const u8 = null;
    var log: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len and !std.mem.eql(u8, args[i], "--")) {
        if (i + 1 >= args.len) break;
        if (std.mem.eql(u8, args[i], "--output")) output = args[i + 1] else if (std.mem.eql(u8, args[i], "--log")) log = args[i + 1] else break;
        i += 2;
    }
    if (output == null or log == null or i >= args.len or !std.mem.eql(u8, args[i], "--") or i + 1 >= args.len) {
        std.debug.print("usage: wasi-p3-sample-launch --output FILE --log FILE -- COMMAND [ARG ...]\n", .{});
        std.process.exit(2);
    }
    const status = capture(allocator, init.io, args[i + 1 ..], output.?, log.?, &.{"/usr/bin/sample"}, 10_000) catch |err| {
        std.debug.print("sample launcher: {s}\n", .{@errorName(err)});
        std.process.exit(125);
    };
    std.process.exit(status);
}

test "capture samples a stopped target and preserves its exit status" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.txt", .{tmp.sub_path});
    defer allocator.free(output);
    const log = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.log", .{tmp.sub_path});
    defer allocator.free(log);
    const marker = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/target-started", .{tmp.sub_path});
    defer allocator.free(marker);
    const status = try capture(allocator, io, &.{ "sh", "-c", "printf started > \"$1\"; exit 7", "target", marker }, output, log, &.{
        "/bin/sh", "-c", "state=$(ps -o state= -p \"$2\"); case \"$state\" in *T*) ;; *) exit 99;; esac; [ ! -e \"$1\" ] || exit 98; printf profile > \"$6\"; printf log", "sampler", marker,
    }, 5000);
    try std.testing.expectEqual(@as(u8, 7), status);
    const contents = try tmp.dir.readFileAlloc(io, "sample.txt", allocator, .unlimited);
    defer allocator.free(contents);
    try std.testing.expectEqualStrings("profile", contents);
    const log_contents = try tmp.dir.readFileAlloc(io, "sample.log", allocator, .unlimited);
    defer allocator.free(log_contents);
    try std.testing.expectEqualStrings("log", log_contents);
}

test "capture preserves a target signal status and terminates an overdue sampler" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.txt", .{tmp.sub_path});
    defer allocator.free(output);
    const log = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.log", .{tmp.sub_path});
    defer allocator.free(log);
    const status = try capture(allocator, io, &.{ "/bin/sh", "-c", "kill -TERM $$" }, output, log, &.{
        "/bin/sh", "-c", "trap 'printf profile > \"$5\"; exit 0' TERM; while :; do :; done", "sampler",
    }, 1000);
    try std.testing.expectEqual(@as(u8, 143), status);
}

test "capture forwards a signal received while the target is running" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.txt", .{tmp.sub_path});
    defer allocator.free(output);
    const log = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.log", .{tmp.sub_path});
    defer allocator.free(log);
    const marker = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/ready", .{tmp.sub_path});
    defer allocator.free(marker);
    const Sender = struct {
        io: std.Io,
        dir: std.Io.Dir,
        done: std.atomic.Value(bool) = .init(false),
        failure: ?anyerror = null,

        fn run(self: *@This()) void {
            while (!self.done.load(.acquire)) {
                if (self.dir.statFile(self.io, "ready", .{})) |_| {
                    std.posix.kill(std.posix.system.getpid(), .TERM) catch |err| {
                        self.failure = err;
                    };
                    return;
                } else |err| switch (err) {
                    error.FileNotFound => {},
                    else => {
                        self.failure = err;
                        return;
                    },
                }
                self.io.sleep(.fromMilliseconds(5), .awake) catch |err| {
                    self.failure = err;
                    return;
                };
            }
        }
    };
    var sender: Sender = .{ .io = io, .dir = tmp.dir };
    const thread = try std.Thread.spawn(.{}, Sender.run, .{&sender});
    const status = capture(allocator, io, &.{
        "/bin/sh", "-c", "trap 'exit 42' TERM; printf ready > \"$1\"; while :; do :; done", "target", marker,
    }, output, log, &.{
        "/bin/sh", "-c", "printf profile > \"$5\"", "sampler",
    }, 1000) catch |err| {
        sender.done.store(true, .release);
        thread.join();
        return err;
    };
    sender.done.store(true, .release);
    thread.join();
    if (sender.failure) |err| return err;
    try std.testing.expectEqual(@as(u8, 42), status);
}

test "capture forwards signals before the stopped target executes" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.txt", .{tmp.sub_path});
    defer allocator.free(output);
    const log = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.log", .{tmp.sub_path});
    defer allocator.free(log);
    const ready = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/ready", .{tmp.sub_path});
    defer allocator.free(ready);
    const started = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/started", .{tmp.sub_path});
    defer allocator.free(started);
    const Sender = struct {
        io: std.Io,
        dir: std.Io.Dir,
        done: std.atomic.Value(bool) = .init(false),
        failure: ?anyerror = null,

        fn run(self: *@This()) void {
            while (!self.done.load(.acquire)) {
                if (self.dir.statFile(self.io, "ready", .{})) |_| {
                    std.posix.kill(std.posix.system.getpid(), .TERM) catch |err| {
                        self.failure = err;
                    };
                    return;
                } else |err| switch (err) {
                    error.FileNotFound => {},
                    else => {
                        self.failure = err;
                        return;
                    },
                }
                self.io.sleep(.fromMilliseconds(1), .awake) catch |err| {
                    self.failure = err;
                    return;
                };
            }
        }
    };
    var sender: Sender = .{ .io = io, .dir = tmp.dir };
    const thread = try std.Thread.spawn(.{}, Sender.run, .{&sender});
    const status = capture(allocator, io, &.{ "/bin/sh", "-c", "printf started > \"$1\"", "target", started }, output, log, &.{
        "/bin/sh", "-c", "printf profile > \"$6\"; printf ready > \"$1\"", "sampler", ready,
    }, 1000) catch |err| {
        sender.done.store(true, .release);
        thread.join();
        return err;
    };
    sender.done.store(true, .release);
    thread.join();
    if (sender.failure) |err| return err;
    try std.testing.expectEqual(@as(u8, 143), status);
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "started", .{}));
}

test "capture fails when no sample was produced" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/missing.txt", .{tmp.sub_path});
    defer allocator.free(output);
    const log = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.log", .{tmp.sub_path});
    defer allocator.free(log);
    const status = try capture(allocator, io, &.{"true"}, output, log, &.{"/bin/true"}, 5000);
    try std.testing.expectEqual(@as(u8, 125), status);
}

test "capture preserves target exec failure status when sampling succeeds" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.txt", .{tmp.sub_path});
    defer allocator.free(output);
    const log = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.log", .{tmp.sub_path});
    defer allocator.free(log);
    const missing = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/missing-target", .{tmp.sub_path});
    defer allocator.free(missing);
    const status = try capture(allocator, io, &.{missing}, output, log, &.{
        "/bin/sh", "-c", "printf profile > \"$5\"", "sampler",
    }, 1000);
    try std.testing.expectEqual(@as(u8, 1), status);
}

test "capture cleans up a stopped target when the sampler cannot start" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.txt", .{tmp.sub_path});
    defer allocator.free(output);
    const log = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.log", .{tmp.sub_path});
    defer allocator.free(log);
    const marker = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/started", .{tmp.sub_path});
    defer allocator.free(marker);
    const missing = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/missing-sampler", .{tmp.sub_path});
    defer allocator.free(missing);
    try std.testing.expectError(error.FileNotFound, capture(allocator, io, &.{
        "/bin/sh", "-c", "printf started > \"$1\"", "target", marker,
    }, output, log, &.{missing}, 100));
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "started", .{}));
}

test "capture fails when the sampler ignores termination" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const output = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.txt", .{tmp.sub_path});
    defer allocator.free(output);
    const log = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/sample.log", .{tmp.sub_path});
    defer allocator.free(log);
    try std.testing.expectError(error.SampleProfilerDidNotTerminate, capture(allocator, io, &.{"/bin/true"}, output, log, &.{
        "/bin/sh", "-c", "trap '' TERM; while :; do :; done", "sampler",
    }, 100));
}
