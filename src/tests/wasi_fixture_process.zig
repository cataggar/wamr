//! Bounded subprocess streams for the WASI fixture runner.
const std = @import("std");
const process_control = @import("process_control");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

pub const Stream = enum { stdout, stderr };

pub const Process = struct {
    io: std.Io,
    child: std.process.Child,
    buffer: std.Io.File.MultiReader.Buffer(2),
    reader: std.Io.File.MultiReader,

    /// `self` must stay at the same address until `deinit`.
    pub fn start(self: *Process, allocator: Allocator, io: std.Io, options: std.process.SpawnOptions) !void {
        var spawn_options = options;
        spawn_options.stdout = .pipe;
        spawn_options.stderr = .pipe;
        self.io = io;
        if (builtin.os.tag == .windows) {
            const pipe = try windowsStdinPipe();
            var parent_end: ?std.Io.File = pipe.writer;
            defer if (parent_end) |file| file.close(io);
            defer pipe.reader.close(io);
            spawn_options.stdin = .{ .file = pipe.reader };
            self.child = try std.process.spawn(io, spawn_options);
            self.child.stdin = parent_end;
            parent_end = null;
        } else {
            spawn_options.stdin = .pipe;
            self.child = try std.process.spawn(io, spawn_options);
        }
        errdefer self.child.kill(io);
        if (builtin.os.tag != .windows) try setNonblocking(self.child.stdin.?);
        self.reader.init(allocator, io, self.buffer.toStreams(), &.{ self.child.stdout.?, self.child.stderr.? });
    }

    pub fn deinit(self: *Process) void {
        self.reader.deinit();
        if (self.child.id != null) {
            process_control.forceKill(&self.child) catch |err| std.debug.print("failed to terminate fixture process: {s}\n", .{@errorName(err)});
            self.child.kill(self.io);
        }
        self.* = undefined;
    }

    fn deadline(self: *Process, milliseconds: u32) std.Io.Timeout {
        return .{ .deadline = .fromNow(self.io, .{ .clock = .awake, .raw = .fromMilliseconds(milliseconds) }) };
    }

    /// The result is borrowed until the next stream operation. EOF may return fewer bytes.
    pub fn read(self: *Process, stream: Stream, count: usize, milliseconds: u32) ![]const u8 {
        const index: usize = @backingInt(stream);
        const until = self.deadline(milliseconds);
        const input = self.reader.reader(index);
        while (input.bufferedLen() < count) {
            if (self.reader.streams.contexts()[index].err) |err| {
                if (err == error.EndOfStream) break;
                return err;
            }
            self.reader.fill(1, until) catch |err| switch (err) {
                error.EndOfStream => break,
                else => return err,
            };
        }
        try self.reader.checkAnyError();
        const length = @min(count, input.bufferedLen());
        const bytes = input.buffered()[0..length];
        input.toss(length);
        return bytes;
    }

    /// Read Unicode characters with Python's universal-newline stream semantics.
    pub fn readText(self: *Process, stream: Stream, count: usize, milliseconds: u32) ![]u8 {
        return self.readTextUntil(stream, count, self.deadline(milliseconds));
    }

    /// Read one line, including its newline, under a single deadline.
    pub fn readLine(self: *Process, stream: Stream, milliseconds: u32) ![]u8 {
        return self.readTextUntil(stream, null, self.deadline(milliseconds));
    }

    fn readTextUntil(self: *Process, stream: Stream, count: ?usize, until: std.Io.Timeout) ![]u8 {
        const index: usize = @backingInt(stream);
        const input = self.reader.reader(index);
        var result: std.ArrayList(u8) = .empty;
        errdefer result.deinit(self.reader.gpa);
        var chars: usize = 0;
        var eof = false;
        while (count == null or chars < count.?) {
            const buffered = input.buffered();
            if (buffered.len == 0) {
                if (eof) break;
                eof = !try self.fillStream(index, until);
                continue;
            }
            if (buffered[0] == '\r') {
                if (buffered.len == 1 and !eof) {
                    eof = !try self.fillStream(index, until);
                    continue;
                }
                const consumed: usize = if (buffered.len > 1 and buffered[1] == '\n') 2 else 1;
                try result.append(self.reader.gpa, '\n');
                input.toss(consumed);
                chars += 1;
                if (count == null) break;
                continue;
            }
            const length = try std.unicode.utf8ByteSequenceLength(buffered[0]);
            if (buffered.len < length) {
                if (eof) return error.InvalidUtf8;
                eof = !try self.fillStream(index, until);
                continue;
            }
            _ = try std.unicode.utf8Decode(buffered[0..length]);
            const newline = buffered[0] == '\n';
            try result.appendSlice(self.reader.gpa, buffered[0..length]);
            input.toss(length);
            chars += 1;
            if (count == null and newline) break;
        }
        return result.toOwnedSlice(self.reader.gpa);
    }

    fn fillStream(self: *Process, index: usize, until: std.Io.Timeout) !bool {
        if (self.reader.streams.contexts()[index].err) |err| {
            if (err == error.EndOfStream) return false;
            return err;
        }
        self.reader.fill(1, until) catch |err| switch (err) {
            error.EndOfStream => return false,
            else => return err,
        };
        try self.reader.checkAnyError();
        return true;
    }

    pub fn write(self: *Process, bytes: []const u8, milliseconds: u32) !void {
        const stdin = self.child.stdin orelse return error.StdinClosed;
        const until = self.deadline(milliseconds);
        var storage: [1]std.Io.Operation.Storage = undefined;
        var batch: std.Io.Batch = .init(&storage);
        defer batch.cancel(self.io);
        var written: usize = 0;
        while (written < bytes.len) {
            const data = [_][]const u8{bytes[written..]};
            batch.addAt(0, .{ .file_write_streaming = .{ .file = stdin, .data = &data } });
            while (true) {
                batch.awaitConcurrent(self.io, self.deadline(1)) catch |err| switch (err) {
                    error.Timeout => {
                        if (until.deadline.durationFromNow(self.io).raw.nanoseconds <= 0) return error.Timeout;
                        self.reader.fill(1, self.deadline(1)) catch |fill_err| switch (fill_err) {
                            error.Timeout, error.EndOfStream => {},
                            else => return fill_err,
                        };
                        try self.reader.checkAnyError();
                        continue;
                    },
                    else => return err,
                };
                break;
            }
            const operation = batch.next() orelse return error.WriteDidNotComplete;
            const count = operation.result.file_write_streaming catch |err| switch (err) {
                error.WouldBlock => {
                    if (until.deadline.durationFromNow(self.io).raw.nanoseconds <= 0) return error.Timeout;
                    self.reader.fill(1, self.deadline(1)) catch |fill_err| switch (fill_err) {
                        error.Timeout, error.EndOfStream => {},
                        else => return fill_err,
                    };
                    try self.reader.checkAnyError();
                    continue;
                },
                else => return err,
            };
            if (count == 0) return error.WriteDidNotProgress;
            written += count;
        }
    }

    pub fn signal(self: *Process, name: []const u8) !void {
        if (!@import("wasi_fixture_config.zig").validSignal(name)) return error.InvalidSignal;
        if (builtin.os.tag == .windows) {
            try process_control.forceKill(&self.child);
        } else {
            const sig = lookupSignal(name) orelse return error.UnsupportedSignal;
            try std.posix.kill(self.child.id orelse return error.ProcessNotRunning, sig);
        }
    }

    pub fn terminate(self: *Process) !void {
        try process_control.forceKill(&self.child);
    }

    /// Close stdin, collect unconsumed output, and reap even if both pipes close early.
    pub fn wait(self: *Process, milliseconds: u32) !std.process.RunResult {
        if (self.child.id == null) return error.ProcessNotRunning;
        if (self.child.stdin) |stdin| {
            stdin.close(self.io);
            self.child.stdin = null;
        }
        const until = self.deadline(milliseconds);
        try self.reader.fillRemaining(until);
        try self.reader.checkAnyError();
        const term = while (true) {
            if (try process_control.pollChild(&self.child, self.io)) |status| break status;
            if (until.deadline.durationFromNow(self.io).raw.nanoseconds <= 0) return error.Timeout;
            try self.io.sleep(.fromMilliseconds(10), .awake);
        };

        const stdout = try self.takeText(0);
        errdefer self.reader.gpa.free(stdout);
        const stderr = try self.takeText(1);
        return .{ .term = term, .stdout = stdout, .stderr = stderr };
    }

    fn takeText(self: *Process, index: usize) ![]u8 {
        const allocator = self.reader.gpa;
        const input = try self.reader.toOwnedSlice(index);
        errdefer allocator.free(input);
        if (!std.unicode.utf8ValidateSlice(input)) return error.InvalidUtf8;
        var src: usize = 0;
        var dst: usize = 0;
        while (src < input.len) {
            if (input[src] == '\r') {
                input[dst] = '\n';
                src += 1;
                if (src < input.len and input[src] == '\n') src += 1;
            } else {
                input[dst] = input[src];
                src += 1;
            }
            dst += 1;
        }
        return allocator.realloc(input, dst);
    }
};

fn windowsStdinPipe() !struct { reader: std.Io.File, writer: std.Io.File } {
    var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{});
    defer threaded.deinit();
    const handles = try threaded.windowsCreatePipe(.{
        .server = .{ .mode = .{ .IO = .ASYNCHRONOUS } },
        .client = .{ .mode = .{ .IO = .SYNCHRONOUS_NONALERT } },
        .outbound = true,
    });
    return .{
        .writer = .{ .handle = handles[0], .flags = .{ .nonblocking = true } },
        .reader = .{ .handle = handles[1], .flags = .{ .nonblocking = false } },
    };
}

fn setNonblocking(file: std.Io.File) !void {
    const posix = std.posix;
    const flags: usize = while (true) {
        const rc = posix.system.fcntl(file.handle, posix.F.GETFL, @as(usize, 0));
        switch (posix.errno(rc)) {
            .SUCCESS => break @intCast(rc),
            .INTR => continue,
            else => |err| return posix.unexpectedErrno(err),
        }
    };
    const nonblocking = flags | (@as(usize, 1) << @bitOffsetOf(posix.O, "NONBLOCK"));
    while (true) {
        switch (posix.errno(posix.system.fcntl(file.handle, posix.F.SETFL, nonblocking))) {
            .SUCCESS => return,
            .INTR => continue,
            else => |err| return posix.unexpectedErrno(err),
        }
    }
}

fn lookupSignal(name: []const u8) ?std.posix.SIG {
    const suffix = name[3..];
    if (builtin.os.tag == .linux and builtin.link_libc) {
        if (std.mem.eql(u8, suffix, "RTMIN")) return @fromBackingInt(std.c.sigrtmin());
        if (std.mem.eql(u8, suffix, "RTMAX")) return @fromBackingInt(std.c.sigrtmax());
    }
    if (std.meta.stringToEnum(std.posix.SIG, suffix)) |signal_value| return signal_value;
    inline for (@typeInfo(std.posix.SIG).@"enum".decl_names) |decl| {
        if (@TypeOf(@field(std.posix.SIG, decl)) == std.posix.SIG and std.mem.eql(u8, suffix, decl)) return @field(std.posix.SIG, decl);
    }
    return null;
}

test "process API compiles for the target" {
    std.testing.refAllDecls(Process);
}

test "wait decodes and normalizes unconsumed output like text-mode subprocesses" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var child: Process = undefined;
    try child.start(allocator, std.testing.io, .{ .argv = &.{
        "/bin/sh", "-c", "printf 'a\\r\\nb\\r'; printf 'c\\rd\\r\\n' >&2",
    } });
    defer child.deinit();
    const result = try child.wait(5000);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try std.testing.expectEqualStrings("a\nb\n", result.stdout);
    try std.testing.expectEqualStrings("c\nd\n", result.stderr);
}

test "invalid output reports an error without polling a reaped child" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var child: Process = undefined;
    try child.start(std.testing.allocator, std.testing.io, .{ .argv = &.{
        "/bin/sh", "-c", "printf '\\377'",
    } });
    defer child.deinit();
    try std.testing.expectError(error.InvalidUtf8, child.wait(5000));
    try std.testing.expectError(error.ProcessNotRunning, child.wait(5000));
}

test "read, write, and collect remaining stdout and stderr" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var process: Process = undefined;
    try process.start(allocator, io, .{ .argv = &.{ "/bin/sh", "-c", "printf hello; read -r line; printf '%s' \"$line\"; printf error >&2; exit 7" } });
    defer process.deinit();
    try std.testing.expectEqualStrings("hello", try process.read(.stdout, 5, 5000));
    try process.write("world\n", 5000);
    const result = try process.wait(5000);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try std.testing.expectEqualStrings("world", result.stdout);
    try std.testing.expectEqualStrings("error", result.stderr);
    try std.testing.expectEqual(@as(u8, 7), result.term.exited);
}

test "read returns a short result at EOF" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var process: Process = undefined;
    try process.start(allocator, io, .{ .argv = &.{ "/bin/sh", "-c", "printf hi" } });
    defer process.deinit();
    try std.testing.expectEqualStrings("hi", try process.read(.stdout, 5, 5000));
    const result = try process.wait(5000);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try std.testing.expect(result.term.success());
}

test "text reads count Unicode characters and normalize split CRLF" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var process: Process = undefined;
    try process.start(allocator, io, .{ .argv = &.{
        "/bin/sh", "-c", "printf '\\303\\251\\r'; sleep 0.05; printf '\\nx\\r'",
    } });
    defer process.deinit();
    const first = try process.readText(.stdout, 2, 5000);
    defer allocator.free(first);
    try std.testing.expectEqualStrings("\xc3\xa9\n", first);
    const remaining = try process.readText(.stdout, 4, 5000);
    defer allocator.free(remaining);
    try std.testing.expectEqualStrings("x\n", remaining);
    const result = try process.wait(5000);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try std.testing.expect(result.term.success());
}

test "line reads preserve following output and normalize newlines" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var process: Process = undefined;
    try process.start(allocator, std.testing.io, .{ .argv = &.{
        "/bin/sh", "-c", "printf '\\303\\251\\r'; sleep 0.05; printf '\\nnext\\nlast'",
    } });
    defer process.deinit();
    const first = try process.readLine(.stdout, 5000);
    defer allocator.free(first);
    try std.testing.expectEqualStrings("\xc3\xa9\n", first);
    const second = try process.readLine(.stdout, 5000);
    defer allocator.free(second);
    try std.testing.expectEqualStrings("next\n", second);
    const third = try process.readLine(.stdout, 5000);
    defer allocator.free(third);
    try std.testing.expectEqualStrings("last", third);
    const result = try process.wait(5000);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try std.testing.expect(result.term.success());
}

test "line reads use one deadline even when output continues" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var process: Process = undefined;
    try process.start(std.testing.allocator, std.testing.io, .{ .argv = &.{
        "/bin/sh", "-c", "printf a; sleep 0.08; printf b; sleep 0.12; printf '\\n'",
    } });
    defer process.deinit();
    try std.testing.expectError(error.Timeout, process.readLine(.stdout, 150));
}

test "timeouts terminate processes which ignore SIGTERM after closing pipes" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var process: Process = undefined;
    try process.start(allocator, io, .{ .argv = &.{ "/bin/sh", "-c", "trap '' TERM; exec 1>&- 2>&-; while :; do :; done" } });
    const pid = process.child.id.?;
    try std.testing.expectError(error.Timeout, process.wait(100));
    process.deinit();
    try std.testing.expectError(error.ProcessNotFound, std.posix.kill(pid, @fromBackingInt(@intCast(0))));
}

test "blocked reads honor deadlines" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var process: Process = undefined;
    try process.start(allocator, io, .{ .argv = &.{ "/bin/sh", "-c", "while :; do :; done" } });
    defer process.deinit();
    try std.testing.expectError(error.Timeout, process.read(.stdout, 1, 100));
}

test "signal delivery and invalid signal rejection" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var process: Process = undefined;
    try process.start(allocator, io, .{ .argv = &.{
        "/bin/sh", "-c", "trap 'exit 9' TERM; printf ready; while :; do :; done",
    } });
    defer process.deinit();
    try std.testing.expectEqualStrings("ready", try process.read(.stdout, 5, 5000));
    try std.testing.expectError(error.InvalidSignal, process.signal("SIGBOGUS"));
    try process.signal("SIGTERM");
    const result = try process.wait(5000);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try std.testing.expectEqual(@as(u8, 9), result.term.exited);
}

test "Linux real-time signal delivery uses libc signal values" {
    if (builtin.os.tag != .linux or !builtin.link_libc) return error.SkipZigTest;
    for ([_][]const u8{ "SIGRTMIN", "SIGRTMAX" }) |name| {
        const signal_value = lookupSignal(name).?;
        var process: Process = undefined;
        try process.start(std.testing.allocator, std.testing.io, .{ .argv = &.{
            "/bin/sh", "-c", "printf ready; while :; do :; done",
        } });
        defer process.deinit();
        try std.testing.expectEqualStrings("ready", try process.read(.stdout, 5, 5000));
        try process.signal(name);
        const result = try process.wait(5000);
        defer std.testing.allocator.free(result.stdout);
        defer std.testing.allocator.free(result.stderr);
        try std.testing.expectEqual(signal_value, result.term.signal);
    }
}

test "blocked writes honor deadlines" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var process: Process = undefined;
    try process.start(allocator, io, .{ .argv = &.{ "/bin/sh", "-c", "while :; do :; done" } });
    defer process.deinit();
    const bytes = try allocator.alloc(u8, 1024 * 1024);
    defer allocator.free(bytes);
    @memset(bytes, 'x');
    try std.testing.expectError(error.Timeout, process.write(bytes, 100));
}

test "writes drain simultaneous output to avoid pipe deadlock" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var process: Process = undefined;
    try process.start(allocator, io, .{ .argv = &.{
        "/bin/sh", "-c", "dd if=/dev/zero bs=1024 count=128 2>/dev/null; read -r line; printf done",
    } });
    defer process.deinit();
    const bytes = try allocator.alloc(u8, 128 * 1024 + 1);
    defer allocator.free(bytes);
    @memset(bytes, 'x');
    bytes[bytes.len - 1] = '\n';
    try process.write(bytes, 5000);
    const result = try process.wait(5000);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try std.testing.expect(result.term.success());
    try std.testing.expectEqual(@as(usize, 128 * 1024 + 4), result.stdout.len);
    try std.testing.expectEqualStrings("done", result.stdout[result.stdout.len - 4 ..]);
}

test "Windows stdin and bounded writes" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    {
        var process: Process = undefined;
        try process.start(allocator, io, .{ .argv = &.{
            "cmd.exe", "/V:ON", "/C", "set /p X=& echo !X!",
        } });
        defer process.deinit();
        try process.write("hello\r\n", 5000);
        const output = try process.readText(.stdout, 6, 5000);
        defer allocator.free(output);
        try std.testing.expectEqualStrings("hello\n", output);
        const result = try process.wait(5000);
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        try std.testing.expect(result.term.success());
    }
    {
        var process: Process = undefined;
        try process.start(allocator, io, .{ .argv = &.{
            "cmd.exe", "/C", "for /L %i in (1,0,2) do @ver >nul",
        } });
        defer process.deinit();
        const bytes = try allocator.alloc(u8, 1024 * 1024);
        defer allocator.free(bytes);
        @memset(bytes, 'x');
        try std.testing.expectError(error.Timeout, process.write(bytes, 100));
    }
}
