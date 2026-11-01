//! Fixture execution for process, text-stream, TCP, and HTTP operations.
const std = @import("std");
const fixture = @import("wasi_fixture_config.zig");
const Process = @import("wasi_fixture_process.zig").Process;
const network = @import("wasi_fixture_network.zig");
const http = @import("wasi_fixture_http.zig");
const profile = @import("wasi_fixture_profile.zig");
const Allocator = std.mem.Allocator;

pub const FailureKind = enum { expectation, unexpected };

pub const Result = struct {
    argv: []const []const u8,
    failures: []const []const u8,
    failure_kinds: []const FailureKind,

    pub fn deinit(self: *Result, allocator: Allocator) void {
        for (self.failures) |failure| allocator.free(failure);
        allocator.free(self.failures);
        allocator.free(self.failure_kinds);
        self.* = undefined;
    }
};

pub const Options = struct {
    operation_timeout_ms: u32,
    wait_timeout_ms: u32,
    environ_map: ?*const std.process.Environ.Map = null,
    timing: ?profile.Execution = null,
};

pub const CommandProvider = struct {
    context: *anyopaque,
    command: *const fn (context: *anyopaque, index: usize, run: fixture.Run) anyerror![]const []const u8,
};

const Commands = union(enum) {
    fixed: []const []const []const u8,
    provider: CommandProvider,
};

fn appendFailure(allocator: Allocator, failures: *std.ArrayList([]const u8), comptime format: []const u8, args: anytype) !void {
    const message = try std.fmt.allocPrint(allocator, format, args);
    errdefer allocator.free(message);
    try failures.append(allocator, message);
}

/// One precomputed command is required for each `run` operation.
/// Unsupported redirect features and ambiguous text encodings are rejected.
pub fn execute(
    allocator: Allocator,
    io: std.Io,
    operations: []const fixture.Operation,
    commands: []const []const []const u8,
    timeout_ms: u32,
) !Result {
    return executeWithOptions(allocator, io, operations, commands, .{
        .operation_timeout_ms = timeout_ms,
        .wait_timeout_ms = timeout_ms,
    });
}

/// The caller can supply a copied environment with DEBUG=true for debug fixtures.
pub fn executeWithOptions(
    allocator: Allocator,
    io: std.Io,
    operations: []const fixture.Operation,
    commands: []const []const []const u8,
    options: Options,
) !Result {
    return executeCore(allocator, io, operations, .{ .fixed = commands }, options);
}

/// Commands are resolved only when each run operation is reached.
/// The returned argv must remain alive until the result is no longer used.
pub fn executeWithProvider(
    allocator: Allocator,
    io: std.Io,
    operations: []const fixture.Operation,
    provider: CommandProvider,
    options: Options,
) !Result {
    return executeCore(allocator, io, operations, .{ .provider = provider }, options);
}

fn executeCore(
    allocator: Allocator,
    io: std.Io,
    operations: []const fixture.Operation,
    commands: Commands,
    options: Options,
) !Result {
    const timeout_ms = options.operation_timeout_ms;
    var required_commands: usize = 0;
    for (operations) |op| switch (op) {
        .run => required_commands += 1,
        .wait, .read, .write, .kill, .connect, .send, .recv, .request => {},
    };
    if (commands == .fixed and required_commands != commands.fixed.len) return error.CommandCountMismatch;
    try fixture.validate(allocator, operations);

    var failures: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (failures.items) |failure| allocator.free(failure);
        failures.deinit(allocator);
    }
    var process: ?*Process = null;
    defer if (process) |p| {
        p.deinit();
        allocator.destroy(p);
    };
    var sockets: std.StringHashMap(std.Io.net.Stream) = .init(allocator);
    defer {
        var iter = sockets.valueIterator();
        while (iter.next()) |socket| socket.close(io);
        sockets.deinit();
    }
    var http_server: ?[]const u8 = null;
    defer if (http_server) |server| allocator.free(server);
    var command_index: usize = 0;
    var last_argv: []const []const u8 = &.{};
    var windows_terminated = false;
    var failure_kind: FailureKind = .unexpected;
    var process_started: std.Io.Timestamp = undefined;

    for (operations) |op| {
        switch (op) {
            .run => |run| {
                last_argv = switch (commands) {
                    .fixed => |list| list[command_index],
                    .provider => |provider| try provider.command(provider.context, command_index, run),
                };
                command_index += 1;
                const child = try allocator.create(Process);
                child.start(allocator, io, .{ .argv = last_argv, .environ_map = options.environ_map }) catch |err| {
                    allocator.destroy(child);
                    try appendFailure(allocator, &failures, "Failed to start process: {s}", .{@errorName(err)});
                    break;
                };
                process = child;
                if (options.timing != null) process_started = std.Io.Clock.awake.now(io);
                windows_terminated = false;
            },
            .read => |read| {
                const stream: @import("wasi_fixture_process.zig").Stream =
                    if (std.mem.eql(u8, read.id, "stdout")) .stdout else .stderr;
                const count = try std.unicode.utf8CountCodepoints(read.payload);
                const actual = process.?.readText(stream, count, timeout_ms) catch |err| {
                    try appendFailure(allocator, &failures, "Failed reading {s}: {s}", .{ read.id, @errorName(err) });
                    break;
                };
                defer allocator.free(actual);
                if (!std.mem.eql(u8, actual, read.payload)) {
                    failure_kind = .expectation;
                    try appendFailure(allocator, &failures, "Read {s} failed: expected {s}, got {s}", .{ read.id, read.payload, actual });
                    break;
                }
            },
            .write => |write| {
                process.?.write(write.payload, timeout_ms) catch |err| {
                    try appendFailure(allocator, &failures, "Failed writing {s}: {s}", .{ write.id, @errorName(err) });
                    break;
                };
            },
            .connect => |id| {
                const deadline: std.Io.Timeout = .{ .deadline = .fromNow(io, .{
                    .clock = .awake,
                    .raw = .fromMilliseconds(timeout_ms),
                }) };
                const line = process.?.readLine(.stdout, timeout_ms) catch |err| {
                    try appendFailure(allocator, &failures, "Failed reading server address: {s}", .{@errorName(err)});
                    break;
                };
                defer allocator.free(line);
                const address = std.mem.trim(u8, line, " \t\r\n\x0b\x0c");
                var parts = std.mem.splitScalar(u8, address, ':');
                const host = parts.next().?;
                const port_text = parts.next() orelse {
                    try appendFailure(allocator, &failures, "Invalid server address: {s}", .{address});
                    break;
                };
                if (parts.next() != null or port_text.len == 0) {
                    try appendFailure(allocator, &failures, "Invalid server address: {s}", .{address});
                    break;
                }
                for (port_text) |c| {
                    if (!std.ascii.isDigit(c)) {
                        try appendFailure(allocator, &failures, "Invalid server address: {s}", .{address});
                        break;
                    }
                }
                if (failures.items.len != 0) break;
                const port = std.fmt.parseInt(u16, port_text, 10) catch {
                    try appendFailure(allocator, &failures, "Invalid server address: {s}", .{address});
                    break;
                };
                const remaining_ns = deadline.deadline.durationFromNow(io).raw.nanoseconds;
                if (remaining_ns <= 0) {
                    try appendFailure(allocator, &failures, "Failed connecting to {s}: Timeout", .{address});
                    break;
                }
                const remaining_ms: u32 = @intCast(@min(timeout_ms, @divTrunc(remaining_ns + 999_999, 1_000_000)));
                const socket = network.connect(io, host, port, remaining_ms) catch |err| {
                    try appendFailure(allocator, &failures, "Failed connecting to {s}: {s}", .{ address, @errorName(err) });
                    break;
                };
                const entry = sockets.getOrPut(id) catch |err| {
                    socket.close(io);
                    return err;
                };
                if (entry.found_existing) entry.value_ptr.close(io);
                entry.value_ptr.* = socket;
            },
            .send => |send| {
                network.send(io, sockets.get(send.id).?, send.payload, timeout_ms) catch |err| {
                    try appendFailure(allocator, &failures, "Failed sending {s}: {s}", .{ send.id, @errorName(err) });
                    break;
                };
            },
            .recv => |recv| {
                const count = try std.unicode.utf8CountCodepoints(recv.payload);
                const buffer = try allocator.alloc(u8, count);
                defer allocator.free(buffer);
                const received = network.recv(io, sockets.get(recv.id).?, buffer, timeout_ms) catch |err| {
                    try appendFailure(allocator, &failures, "Failed receiving {s}: {s}", .{ recv.id, @errorName(err) });
                    break;
                };
                const actual = buffer[0..received];
                if (!std.unicode.utf8ValidateSlice(actual)) {
                    try appendFailure(allocator, &failures, "Failed decoding {s}: InvalidUtf8", .{recv.id});
                    break;
                }
                if (!std.mem.eql(u8, actual, recv.payload)) {
                    try appendFailure(allocator, &failures, "Receive {s} failed: expected {s}, got {s}", .{ recv.id, recv.payload, actual });
                    break;
                }
            },
            .request => |request| {
                const deadline: std.Io.Timeout = .{ .deadline = .fromNow(io, .{
                    .clock = .awake,
                    .raw = .fromMilliseconds(timeout_ms),
                }) };
                if (http_server == null) {
                    const line = process.?.readLine(.stderr, timeout_ms) catch |err| {
                        try appendFailure(allocator, &failures, "Failed reading HTTP server address: {s}", .{@errorName(err)});
                        break;
                    };
                    defer allocator.free(line);
                    const start = std.mem.indexOf(u8, line, "http://") orelse {
                        try appendFailure(allocator, &failures, "HTTP server address not found: {s}", .{line});
                        break;
                    };
                    const address = line[start..];
                    var end: usize = 0;
                    while (end < address.len and !std.ascii.isWhitespace(address[end])) : (end += 1) {}
                    http_server = try allocator.dupe(u8, address[0..end]);
                }
                const base = std.mem.trimEnd(u8, http_server.?, "/");
                const url = try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{
                    base,
                    if (std.mem.startsWith(u8, request.path, "/")) "" else "/",
                    request.path,
                });
                defer allocator.free(url);
                const remaining_ns = deadline.deadline.durationFromNow(io).raw.nanoseconds;
                if (remaining_ns <= 0) {
                    try appendFailure(allocator, &failures, "HTTP request failed: Timeout", .{});
                    break;
                }
                const remaining_ms: u32 = @intCast(@min(5000, @divTrunc(remaining_ns + 999_999, 1_000_000)));
                var response = http.request(io, url, request.method, remaining_ms) catch |err| {
                    try appendFailure(allocator, &failures, "HTTP request failed: {s}", .{@errorName(err)});
                    break;
                };
                defer response.deinit();
                const mismatch = http.compare(&response, request) catch |err| {
                    try appendFailure(allocator, &failures, "HTTP response could not be compared: {s}", .{@errorName(err)});
                    break;
                };
                if (mismatch) |difference| {
                    switch (difference) {
                        .status => |value| try appendFailure(allocator, &failures, "HTTP status: expected {d}, got {d}", .{ value.expected, value.actual }),
                        .missing_header => |name| try appendFailure(allocator, &failures, "HTTP response missing header {s}", .{name}),
                        .header => |value| try appendFailure(allocator, &failures, "HTTP header {s}: expected {s}, got {s}", .{ value.name, value.expected, value.actual }),
                        .body => |value| try appendFailure(allocator, &failures, "HTTP body: expected {s}, got {s}", .{ value.expected, value.actual }),
                    }
                    break;
                }
            },
            .kill => |signal| {
                process.?.signal(signal) catch |err| {
                    try appendFailure(allocator, &failures, "Failed sending {s}: {s}", .{ signal, @errorName(err) });
                    break;
                };
                windows_terminated = @import("builtin").os.tag == .windows;
            },
            .wait => |expected| {
                const child = process.?;
                const waited = child.wait(options.wait_timeout_ms);
                if (options.timing) |timing| {
                    profile.executionEvent(io, timing, process_started) catch |err| {
                        if (waited) |value| {
                            allocator.free(value.stdout);
                            allocator.free(value.stderr);
                        } else |_| {}
                        return err;
                    };
                }
                const result = waited catch |err| {
                    if (err == error.Timeout) failure_kind = .expectation;
                    if (err == error.Timeout) {
                        try appendFailure(allocator, &failures, "Wait(exit_code={d}) failed: timeout expired", .{expected});
                    } else {
                        try appendFailure(allocator, &failures, "Wait failed: {s}", .{@errorName(err)});
                    }
                    break;
                };
                defer allocator.free(result.stdout);
                defer allocator.free(result.stderr);
                const raw_code: ?i64 = switch (result.term) {
                    .exited => |n| n,
                    .signal => |signal| -@as(i64, @intCast(@backingInt(signal))),
                    else => null,
                };
                const code: ?i64 = if (windows_terminated and expected == 0 and raw_code == 1) 0 else raw_code;
                if (code == null or code.? != expected) {
                    failure_kind = .expectation;
                    const actual = if (code) |n|
                        try std.fmt.allocPrint(allocator, "{d}", .{n})
                    else
                        try std.fmt.allocPrint(allocator, "{f}", .{result.term});
                    defer allocator.free(actual);
                    try appendFailure(
                        allocator,
                        &failures,
                        "Wait(exit_code={d}) failed: expected {d}, got {s}{s}{s}{s}{s}",
                        .{
                            expected,                                               expected,      actual,
                            if (result.stdout.len != 0) "\n\n==STDOUT==\n" else "", result.stdout, if (result.stderr.len != 0) "\n\n==STDERR==\n" else "",
                            result.stderr,
                        },
                    );
                }
                child.deinit();
                allocator.destroy(child);
                process = null;
                if (failures.items.len != 0) break;
            },
        }
    }
    const original_failures = failures.items.len;
    if (process) |child| {
        child.terminate() catch |err| {
            try appendFailure(allocator, &failures, "Failed terminating fixture process: {s}", .{@errorName(err)});
        };
        const waited = child.wait(5000);
        if (options.timing) |timing| {
            profile.executionEvent(io, timing, process_started) catch |err| {
                if (waited) |value| {
                    allocator.free(value.stdout);
                    allocator.free(value.stderr);
                } else |_| {}
                return err;
            };
        }
        if (waited) |result| {
            defer allocator.free(result.stdout);
            defer allocator.free(result.stderr);
            try appendFailure(allocator, &failures, "{s}{s}{s}{s}", .{
                if (result.stdout.len != 0) "\n\n==STDOUT==\n" else "",
                result.stdout,
                if (result.stderr.len != 0) "\n\n==STDERR==\n" else "",
                result.stderr,
            });
        } else |err| {
            try appendFailure(allocator, &failures, "Cleanup failed: {s}", .{@errorName(err)});
        }
        child.deinit();
        allocator.destroy(child);
        process = null;
    }
    const failure_kinds = try allocator.alloc(FailureKind, failures.items.len);
    errdefer allocator.free(failure_kinds);
    for (failure_kinds, 0..) |*kind, index| kind.* = if (index < original_failures) failure_kind else .unexpected;
    return .{
        .argv = last_argv,
        .failures = try failures.toOwnedSlice(allocator),
        .failure_kinds = failure_kinds,
    };
}

test "executor API compiles for the target" {
    std.testing.refAllDecls(@This());
    _ = @import("wasi_fixture_network.zig");
    _ = @import("wasi_fixture_http.zig");
}

test "execute legacy output and exit expectations" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const config = try fixture.parse(allocator, "{\"stdout\":\"hello\\n\",\"exit_code\":3}");
    const success = try execute(allocator, std.testing.io, config.operations, &.{&.{ "/bin/sh", "-c", "printf 'hello\\r\\n'; exit 3" }}, 5000);
    try std.testing.expectEqual(@as(usize, 0), success.failures.len);
    const mismatch = try execute(allocator, std.testing.io, config.operations, &.{&.{ "/bin/sh", "-c", "printf diagnostic >&2; printf 'wrong\\n'; exit 3" }}, 5000);
    try std.testing.expectEqual(@as(usize, 2), mismatch.failures.len);
    try std.testing.expect(std.mem.indexOf(u8, mismatch.failures[0], "Read stdout failed") != null);
    try std.testing.expect(std.mem.indexOf(u8, mismatch.failures[1], "==STDERR==\ndiagnostic") != null);
    try std.testing.expectEqualSlices(FailureKind, &.{ .expectation, .unexpected }, mismatch.failure_kinds);
}

test "result releases failure messages" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const operations = [_]fixture.Operation{
        .{ .run = .{} },
        .{ .read = .{ .id = "stdout", .payload = "expected" } },
        .{ .wait = 0 },
    };
    var result = try execute(std.testing.allocator, std.testing.io, &operations, &.{&.{ "/bin/sh", "-c", "printf actual" }}, 5000);
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2), result.failures.len);
}

test "timing sink failures propagate without leaking completed process output" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "fixture.wasm", .data = "\x00asm\x01\x00\x00\x00" });
    const path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/fixture.wasm", .{tmp.sub_path});
    defer allocator.free(path);
    const Sink = struct {
        fn emit(_: *anyopaque, _: profile.Event) !void {
            return error.ProfileWriteFailed;
        }
    };
    var context: u8 = 0;
    const operations = [_]fixture.Operation{ .{ .run = .{} }, .{ .wait = 0 } };
    try std.testing.expectError(error.ProfileWriteFailed, executeWithOptions(allocator, io, &operations, &.{
        &.{ "/bin/sh", "-c", "printf out; printf err >&2" },
    }, .{
        .operation_timeout_ms = 5000,
        .wait_timeout_ms = 5000,
        .timing = .{ .sink = .{ .context = &context, .emit = Sink.emit }, .fixture_path = path },
    }));
}

test "execute interactive stdin and check command counts" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const config = try fixture.parse(allocator,
        \\{"operations":[{"type":"run"},{"type":"write","id":"stdin","payload":"hi\n"},
        \\{"type":"read","payload":"hi\n"},{"type":"wait"}]}
    );
    const result = try execute(allocator, std.testing.io, config.operations, &.{&.{ "/bin/sh", "-c", "read -r value; printf '%s\\n' \"$value\"" }}, 5000);
    try std.testing.expectEqual(@as(usize, 0), result.failures.len);
    try std.testing.expectError(error.CommandCountMismatch, execute(allocator, std.testing.io, config.operations, &.{}, 100));
}

test "runtime commands are prepared only when their run is reached" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const marker = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/marker", .{tmp.sub_path});
    defer allocator.free(marker);
    const first = [_][]const u8{ "/bin/sh", "-c", "printf ready > \"$1\"", "sh", marker };
    const second = [_][]const u8{ "/bin/sh", "-c", "printf done" };
    const Context = struct {
        io: std.Io,
        dir: std.Io.Dir,
        calls: usize = 0,
        first: []const []const u8,
        second: []const []const u8,
        fn command(raw: *anyopaque, index: usize, run: fixture.Run) ![]const []const u8 {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.calls += 1;
            if (index == 0) return self.first;
            if (index != 1 or run.args.len != 1 or !std.mem.eql(u8, run.args[0], "second")) return error.UnexpectedRun;
            _ = try self.dir.statFile(self.io, "marker", .{});
            return self.second;
        }
    };
    var context: Context = .{ .io = io, .dir = tmp.dir, .first = &first, .second = &second };
    const operations = [_]fixture.Operation{
        .{ .run = .{} },
        .{ .wait = 0 },
        .{ .run = .{ .args = &.{"second"} } },
        .{ .read = .{ .id = "stdout", .payload = "done" } },
        .{ .wait = 0 },
    };
    const provider: CommandProvider = .{ .context = &context, .command = Context.command };
    var result = try executeWithProvider(allocator, io, &operations, provider, .{
        .operation_timeout_ms = 5000,
        .wait_timeout_ms = 5000,
    });
    defer result.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), result.failures.len);
    try std.testing.expectEqual(@as(usize, 2), context.calls);

    context.calls = 0;
    var failing_operations = operations;
    failing_operations[1] = .{ .wait = 7 };
    var failure = try executeWithProvider(allocator, io, &failing_operations, provider, .{
        .operation_timeout_ms = 5000,
        .wait_timeout_ms = 5000,
    });
    defer failure.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), failure.failures.len);
    try std.testing.expectEqual(@as(usize, 1), context.calls);
}

test "execute signal and expected exit operations" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const config = try fixture.parse(allocator,
        \\{"operations":[{"type":"run"},{"type":"read","payload":"ready"},
        \\{"type":"kill","signal":"SIGTERM"},{"type":"wait","exit_code":9}]}
    );
    const result = try execute(allocator, std.testing.io, config.operations, &.{&.{
        "/bin/sh", "-c", "trap 'exit 9' TERM; printf ready; while :; do :; done",
    }}, 5000);
    try std.testing.expectEqual(@as(usize, 0), result.failures.len);
}

test "execution reports timeout and exit-code failures" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const config = try fixture.defaultConfig(allocator);
    const timed = try execute(allocator, std.testing.io, config.operations, &.{&.{ "/bin/sh", "-c", "while :; do :; done" }}, 100);
    try std.testing.expectEqual(@as(usize, 2), timed.failures.len);
    try std.testing.expectEqualStrings("Wait(exit_code=0) failed: timeout expired", timed.failures[0]);
    const exited = try execute(allocator, std.testing.io, config.operations, &.{&.{ "/bin/sh", "-c", "printf out; printf err >&2; exit 5" }}, 5000);
    try std.testing.expectEqual(@as(usize, 1), exited.failures.len);
    try std.testing.expectEqualStrings("Wait(exit_code=0) failed: expected 0, got 5\n\n==STDOUT==\nout\n\n==STDERR==\nerr", exited.failures[0]);
    const stderr_only = try execute(allocator, std.testing.io, config.operations, &.{&.{ "/bin/sh", "-c", "printf err >&2; exit 5" }}, 5000);
    try std.testing.expectEqualStrings("Wait(exit_code=0) failed: expected 0, got 5\n\n==STDERR==\nerr", stderr_only.failures[0]);
    const signaled = try execute(allocator, std.testing.io, config.operations, &.{&.{ "/bin/sh", "-c", "kill -TERM $$" }}, 5000);
    try std.testing.expectEqualStrings("Wait(exit_code=0) failed: expected 0, got -15", signaled.failures[0]);
}

test "operation and wait deadlines are independent and the child receives its environment" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var environment = std.process.Environ.Map.init(allocator);
    defer environment.deinit();
    try environment.put("DEBUG", "true");
    const operations = [_]fixture.Operation{
        .{ .run = .{} },
        .{ .read = .{ .id = "stdout", .payload = "true" } },
        .{ .wait = 0 },
    };
    const options: Options = .{
        .operation_timeout_ms = 5000,
        .wait_timeout_ms = 100,
        .environ_map = &environment,
    };
    var wait_failure = try executeWithOptions(allocator, std.testing.io, &operations, &.{&.{
        "/bin/sh", "-c", "printf %s \"$DEBUG\"; while :; do :; done",
    }}, options);
    defer wait_failure.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), wait_failure.failures.len);
    try std.testing.expectEqualStrings("Wait(exit_code=0) failed: timeout expired", wait_failure.failures[0]);
    try std.testing.expectEqualSlices(FailureKind, &.{ .expectation, .unexpected }, wait_failure.failure_kinds);

    var read_failure = try executeWithOptions(allocator, std.testing.io, &operations, &.{&.{
        "/bin/sh", "-c", "while :; do :; done",
    }}, .{ .operation_timeout_ms = 100, .wait_timeout_ms = 5000, .environ_map = &environment });
    defer read_failure.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), read_failure.failures.len);
    try std.testing.expect(std.mem.indexOf(u8, read_failure.failures[0], "Failed reading stdout: Timeout") != null);
    try std.testing.expectEqualSlices(FailureKind, &.{ .unexpected, .unexpected }, read_failure.failure_kinds);
}

fn testEchoServer(io: std.Io, server: *std.Io.net.Server) !void {
    const peer = try server.accept(io);
    defer peer.close(io);
    var buffer: [32]u8 = undefined;
    const count = try network.recv(io, peer, &buffer, 5000);
    try std.testing.expectEqualStrings("hello", buffer[0..count]);
    try network.send(io, peer, "world", 5000);
}

fn testShortServer(io: std.Io, server: *std.Io.net.Server) !void {
    const peer = try server.accept(io);
    defer peer.close(io);
    try network.send(io, peer, "wo", 5000);
}

fn testHttpServer(io: std.Io, server: *std.Io.net.Server) !void {
    for (0..2) |_| {
        const peer = try server.accept(io);
        defer peer.close(io);
        var buffer: [4096]u8 = undefined;
        var count: usize = 0;
        while (std.mem.indexOf(u8, buffer[0..count], "\r\n\r\n") == null) {
            if (count == buffer.len) return error.HttpRequestTooLong;
            const n = try network.recv(io, peer, buffer[count..], 5000);
            if (n == 0) return error.UnexpectedEndOfStream;
            count += n;
        }
        try std.testing.expect(std.mem.startsWith(u8, buffer[0..count], "GET /test "));
        try network.send(io, peer, "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok", 5000);
    }
}

test "execute TCP fixture operations" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
    var server = try address.listen(io, .{});
    defer server.deinit(io);
    var task = try io.concurrent(testEchoServer, .{ io, &server });
    var awaited = false;
    defer if (!awaited) {
        task.cancel(io) catch |err| {
            if (err != error.Canceled) std.debug.print("test server failed: {s}\n", .{@errorName(err)});
        };
    };
    const script = try std.fmt.allocPrint(allocator, "printf '127.0.0.1:{d}\\n'; read -r line || :", .{server.socket.address.getPort()});
    defer allocator.free(script);
    const operations = [_]fixture.Operation{
        .{ .run = .{} },
        .{ .connect = "server" },
        .{ .send = .{ .id = "server", .payload = "hello" } },
        .{ .recv = .{ .id = "server", .payload = "world" } },
        .{ .wait = 0 },
    };
    var result = try execute(allocator, io, &operations, &.{&.{ "/bin/sh", "-c", script }}, 5000);
    defer result.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), result.failures.len);
    try task.await(io);
    awaited = true;
}

test "execute rejects malformed server addresses" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const operations = [_]fixture.Operation{ .{ .run = .{} }, .{ .connect = "server" }, .{ .wait = 0 } };
    var result = try execute(std.testing.allocator, std.testing.io, &operations, &.{
        &.{ "/bin/sh", "-c", "printf 'not-an-address\\n'" },
    }, 5000);
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2), result.failures.len);
    try std.testing.expect(std.mem.indexOf(u8, result.failures[0], "Invalid server address") != null);
}

test "execute reports short TCP responses" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
    var server = try address.listen(io, .{});
    defer server.deinit(io);
    var task = try io.concurrent(testShortServer, .{ io, &server });
    var awaited = false;
    defer if (!awaited) {
        task.cancel(io) catch |err| {
            if (err != error.Canceled) std.debug.print("test server failed: {s}\n", .{@errorName(err)});
        };
    };
    const script = try std.fmt.allocPrint(allocator, "printf '127.0.0.1:{d}\\n'; read -r line || :", .{server.socket.address.getPort()});
    defer allocator.free(script);
    const operations = [_]fixture.Operation{
        .{ .run = .{} },
        .{ .connect = "server" },
        .{ .recv = .{ .id = "server", .payload = "world" } },
        .{ .wait = 0 },
    };
    var result = try execute(allocator, io, &operations, &.{&.{ "/bin/sh", "-c", script }}, 5000);
    defer result.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), result.failures.len);
    try std.testing.expect(std.mem.indexOf(u8, result.failures[0], "expected world, got w") != null);
    try task.await(io);
    awaited = true;
}

test "execute HTTP requests reuse the discovered server address" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
    var server = try address.listen(io, .{});
    defer server.deinit(io);
    var task = try io.concurrent(testHttpServer, .{ io, &server });
    var awaited = false;
    defer if (!awaited) {
        task.cancel(io) catch |err| {
            if (err != error.Canceled) std.debug.print("test server failed: {s}\n", .{@errorName(err)});
        };
    };
    const script = try std.fmt.allocPrint(allocator, "printf 'Listening on http://127.0.0.1:{d}/\\n' >&2; read -r line || :", .{server.socket.address.getPort()});
    defer allocator.free(script);
    const request: fixture.Request = .{
        .path = "test",
        .headers = &.{.{ .name = "content-type", .value = "text/plain" }},
        .body = "ok",
    };
    const operations = [_]fixture.Operation{
        .{ .run = .{} },
        .{ .request = request },
        .{ .request = request },
        .{ .wait = 0 },
    };
    var result = try execute(allocator, io, &operations, &.{&.{ "/bin/sh", "-c", script }}, 5000);
    defer result.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), result.failures.len);
    try task.await(io);
    awaited = true;
}
