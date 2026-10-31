//! Timing events collected by the experimental native fixture runner.
const std = @import("std");
const runtime = @import("wasi_fixture_runtime.zig");

pub const Phase = enum { core_precompile, component_precompile, fixture_execution };

pub const Event = struct {
    fixture: []const u8,
    phase: Phase,
    artifact_kind: runtime.ArtifactKind,
    cache: runtime.Cache,
    duration_ns: u64,
};

pub const Sink = struct {
    context: *anyopaque,
    emit: *const fn (context: *anyopaque, event: Event) anyerror!void,
};

pub const Execution = struct {
    sink: Sink,
    fixture_path: []const u8,
};

pub const JsonlWriter = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    run_id: []const u8,
    mode: []const u8,
    pid: u32,

    pub fn emit(raw: *anyopaque, event: Event) !void {
        const self: *JsonlWriter = @ptrCast(@alignCast(raw));
        if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos)
            return error.ProfilingAppendUnsupported;
        const json = try std.json.Stringify.valueAlloc(self.allocator, .{
            .artifact_kind = @tagName(event.artifact_kind),
            .cache = if (event.cache == .not_applicable) "n/a" else @tagName(event.cache),
            .duration_ns = event.duration_ns,
            .event = "phase_timing",
            .fixture = event.fixture,
            .mode = self.mode,
            .phase = @tagName(event.phase),
            .pid = self.pid,
            .run_id = self.run_id,
            .schema_version = @as(u8, 1),
        }, .{});
        defer self.allocator.free(json);
        const line = try std.fmt.allocPrint(self.allocator, "{s}\n", .{json});
        defer self.allocator.free(line);
        const fd = try std.posix.openat(std.posix.AT.FDCWD, self.path, .{
            .ACCMODE = .WRONLY,
            .APPEND = true,
            .CREAT = true,
            .CLOEXEC = true,
        }, 0o644);
        const file: std.Io.File = .{ .handle = fd, .flags = .{ .nonblocking = false } };
        defer file.close(self.io);
        const written = try file.writeStreaming(self.io, &.{}, &.{line}, 1);
        if (written != line.len) return error.ShortProfileWrite;
    }
};

pub fn elapsed(io: std.Io, start: std.Io.Timestamp) u64 {
    const nanoseconds = start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
    return @intCast(@max(0, nanoseconds));
}

pub fn executionEvent(io: std.Io, timing: Execution, start: std.Io.Timestamp) !void {
    const duration_ns = elapsed(io, start);
    const file = try std.Io.Dir.cwd().openFile(io, timing.fixture_path, .{});
    defer file.close(io);
    var header: [8]u8 = undefined;
    const count = try file.readPositionalAll(io, &header, 0);
    try timing.sink.emit(timing.sink.context, .{
        .fixture = std.fs.path.stem(timing.fixture_path),
        .phase = .fixture_execution,
        .artifact_kind = if (count == 8 and std.mem.eql(u8, header[4..8], "\x0d\x00\x01\x00")) .component else .core,
        .cache = .not_applicable,
        .duration_ns = duration_ns,
    });
}

test "timing JSONL uses the upstream schema and appends complete events" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/timings.jsonl", .{tmp.sub_path});
    defer allocator.free(path);
    var writer: JsonlWriter = .{ .allocator = allocator, .io = io, .path = path, .run_id = "run\"1", .mode = "jit", .pid = 42 };
    const event: Event = .{
        .fixture = "fixture",
        .phase = .fixture_execution,
        .artifact_kind = .component,
        .cache = .not_applicable,
        .duration_ns = 123,
    };
    try JsonlWriter.emit(&writer, event);
    try JsonlWriter.emit(&writer, event);
    const contents = try tmp.dir.readFileAlloc(io, "timings.jsonl", allocator, .unlimited);
    defer allocator.free(contents);
    var lines = std.mem.splitScalar(u8, contents, '\n');
    for (0..2) |_| {
        const line = lines.next().?;
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, line, .{});
        defer parsed.deinit();
        const object = parsed.value.object;
        try std.testing.expectEqual(@as(usize, 10), object.count());
        try std.testing.expectEqualStrings("phase_timing", object.get("event").?.string);
        try std.testing.expectEqualStrings("run\"1", object.get("run_id").?.string);
        try std.testing.expectEqualStrings("jit", object.get("mode").?.string);
        try std.testing.expectEqualStrings("fixture", object.get("fixture").?.string);
        try std.testing.expectEqualStrings("fixture_execution", object.get("phase").?.string);
        try std.testing.expectEqualStrings("component", object.get("artifact_kind").?.string);
        try std.testing.expectEqualStrings("n/a", object.get("cache").?.string);
        try std.testing.expectEqual(@as(i64, 123), object.get("duration_ns").?.integer);
        try std.testing.expectEqual(@as(i64, 42), object.get("pid").?.integer);
        try std.testing.expectEqual(@as(i64, 1), object.get("schema_version").?.integer);
    }

    try std.testing.expectEqualStrings("", lines.next().?);
    try std.testing.expect(lines.next() == null);
}

test "concurrent timing writers append intact JSON lines" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/concurrent.jsonl", .{tmp.sub_path});
    defer allocator.free(path);
    const Worker = struct {
        writer: JsonlWriter,
        index: usize,
        failure: ?anyerror = null,
        fn run(self: *@This()) void {
            for (0..32) |i| {
                JsonlWriter.emit(&self.writer, .{
                    .fixture = "fixture",
                    .phase = .core_precompile,
                    .artifact_kind = .core,
                    .cache = .hit,
                    .duration_ns = self.index * 32 + i,
                }) catch |err| {
                    self.failure = err;
                    return;
                };
            }
        }
    };
    var workers: [4]Worker = undefined;
    var threads: [4]std.Thread = undefined;
    var started: usize = 0;
    defer for (threads[0..started]) |thread| thread.join();
    for (&workers, &threads, 0..) |*worker, *thread, index| {
        worker.* = .{
            .writer = .{ .allocator = std.heap.page_allocator, .io = io, .path = path, .run_id = "", .mode = "aot", .pid = 1 },
            .index = index,
        };
        thread.* = try std.Thread.spawn(.{}, Worker.run, .{worker});
        started += 1;
    }
    for (threads) |thread| thread.join();
    started = 0;
    for (workers) |worker| if (worker.failure) |err| return err;
    const contents = try tmp.dir.readFileAlloc(io, "concurrent.jsonl", allocator, .unlimited);
    defer allocator.free(contents);
    var seen: [128]bool = @splat(false);
    var lines = std.mem.tokenizeScalar(u8, contents, '\n');
    while (lines.next()) |line| {
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, line, .{});
        defer parsed.deinit();
        const index: usize = @intCast(parsed.value.object.get("duration_ns").?.integer);
        try std.testing.expect(index < seen.len and !seen[index]);
        seen[index] = true;
    }
    for (seen) |present| try std.testing.expect(present);
}
