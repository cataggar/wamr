//! Execute individual mutations while restoring the target after every run.
const std = @import("std");
const candidates = @import("mutation_candidates.zig");
const target_mod = @import("mutation_target.zig");
const process = @import("mutation_process.zig");
const report = @import("mutation_report.zig");
const fixed = @import("decimal_format.zig").fixed;
const Allocator = std.mem.Allocator;

fn tail(allocator: Allocator, raw: []const u8) ![]const u8 {
    if (!std.unicode.utf8ValidateSlice(raw)) return error.InvalidCommandOutputEncoding;
    var lines: std.ArrayList([]const u8) = .empty;
    var index: usize = 0;
    var start: usize = 0;
    while (index < raw.len) {
        const width = try std.unicode.utf8ByteSequenceLength(raw[index]);
        const point = try std.unicode.utf8Decode(raw[index..][0..width]);
        if (switch (point) {
            '\n', '\r', 0x0b, 0x0c, 0x1c...0x1e, 0x85, 0x2028, 0x2029 => true,
            else => false,
        }) {
            try lines.append(allocator, raw[start..index]);
            index += width;
            if (point == '\r' and index < raw.len and raw[index] == '\n') index += 1;
            start = index;
        } else {
            index += width;
        }
    }
    if (start < raw.len) try lines.append(allocator, raw[start..]);
    var out: std.Io.Writer.Allocating = .init(allocator);
    for (lines.items[lines.items.len - @min(5, lines.items.len) ..], 0..) |line, i| {
        if (i != 0) try out.writer.writeByte('\n');
        try out.writer.writeAll(line);
    }
    return out.toOwnedSlice();
}

/// Use an arena for results and candidate strings. `target` remains open.
pub fn execute(
    allocator: Allocator,
    io: std.Io,
    target: *target_mod.Target,
    selected: []const candidates.Mutant,
    command: []const u8,
    timeout_ns: ?i96,
    timeout_text: []const u8,
    interrupted: *const std.atomic.Value(bool),
    progress: *std.Io.Writer,
) ![]const report.Result {
    var results: std.ArrayList(report.Result) = .empty;
    for (selected, 0..) |mutant, index| {
        if (interrupted.load(.acquire)) return error.Interrupted;
        try target.apply(mutant);
        const start = std.Io.Clock.awake.now(io);
        const executed = process.run(allocator, io, command, timeout_ns, interrupted) catch |err| {
            target.restore() catch |restore_err| {
                std.debug.print("mutation target restoration failed after {s}: {s}\n", .{ @errorName(err), @errorName(restore_err) });
                return restore_err;
            };
            return err;
        };
        const seconds = @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds)) / std.time.ns_per_s;
        try target.restore();
        const output = if (executed.timed_out)
            try std.fmt.allocPrint(allocator, "{s}\nmutation-run: timed out after {s} seconds", .{ executed.output, timeout_text })
        else
            executed.output;
        const result: report.Result = .{
            .mutant = mutant,
            .survived = executed.exit_code == 0,
            .exit_code = executed.exit_code,
            .seconds = seconds,
            .output_tail = try tail(allocator, output),
        };
        try results.append(allocator, result);
        try progress.print("[{d}/{d}] {s} line {d}: {s} ({s}s)\n", .{
            index + 1, selected.len, if (result.survived) "SURVIVED" else "KILLED", mutant.line_no, mutant.mutator, try fixed(allocator, seconds, 1),
        });
        try progress.flush();
    }
    return results.toOwnedSlice(allocator);
}

test "output tail follows text splitlines and keeps the last five" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    try std.testing.expectEqualStrings("b\nc\nd\ne\nf", try tail(allocator, "a\r\nb\rc\x0bd\xc2\x85e\xe2\x80\xa8f\n"));
    try std.testing.expectEqualStrings("", try tail(allocator, ""));
    try std.testing.expectError(error.InvalidCommandOutputEncoding, tail(allocator, "\xff"));
}

test "execution restores every mutation on success and interruption" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.writeFile(io, .{ .sub_path = "test.zig", .data = "x = 0;\n" });
    const path = try temp.dir.realPathFileAlloc(io, "test.zig", allocator);
    var target = try target_mod.Target.open(allocator, io, path);
    defer target.close() catch unreachable;
    const mutants = try candidates.collect(allocator, target.source, &.{});
    try std.testing.expectEqual(@as(usize, 1), mutants.len);
    var interrupted: std.atomic.Value(bool) = .init(false);
    var progress: std.Io.Writer.Allocating = .init(allocator);
    const results = try execute(allocator, io, &target, mutants, "printf 'one\\ntwo\\n'; exit 3", 5 * std.time.ns_per_s, "5.0", &interrupted, &progress.writer);
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqual(@as(i64, 3), results[0].exit_code);
    try std.testing.expectEqualStrings("one\ntwo", results[0].output_tail);
    try std.testing.expectEqualStrings("x = 0;\n", try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited));
    const timed = try execute(allocator, io, &target, mutants, "printf started; sleep 30", 100 * std.time.ns_per_ms, "0.1", &interrupted, &progress.writer);
    try std.testing.expectEqual(@as(i64, 124), timed[0].exit_code);
    try std.testing.expect(std.mem.startsWith(u8, timed[0].output_tail, "started\n"));
    try std.testing.expect(std.mem.endsWith(u8, timed[0].output_tail, ": timed out after 0.1 seconds"));
    try std.testing.expectEqualStrings("x = 0;\n", try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited));
    interrupted.store(true, .release);
    try std.testing.expectError(error.Interrupted, execute(allocator, io, &target, mutants, "exit 0", null, "", &interrupted, &progress.writer));
    try std.testing.expectEqualStrings("x = 0;\n", try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited));
}

test "interruption during a running mutation restores the target" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.writeFile(io, .{ .sub_path = "test.zig", .data = "x = 0;\n" });
    const path = try temp.dir.realPathFileAlloc(io, "test.zig", allocator);
    const marker = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/started", .{temp.sub_path});
    const command = try std.fmt.allocPrint(allocator, "printf ready > {s}; sleep 30", .{marker});
    var target = try target_mod.Target.open(allocator, io, path);
    defer target.close() catch unreachable;
    const mutants = try candidates.collect(allocator, target.source, &.{});
    var interrupted: std.atomic.Value(bool) = .init(false);
    const Sender = struct {
        io: std.Io,
        dir: std.Io.Dir,
        interrupted: *std.atomic.Value(bool),
        done: std.atomic.Value(bool) = .init(false),
        failure: ?anyerror = null,

        fn run(self: *@This()) void {
            while (!self.done.load(.acquire)) {
                if (self.dir.statFile(self.io, "started", .{})) |_| {
                    self.interrupted.store(true, .release);
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
    var sender: Sender = .{ .io = io, .dir = temp.dir, .interrupted = &interrupted };
    const thread = try std.Thread.spawn(.{}, Sender.run, .{&sender});
    var progress: std.Io.Writer.Allocating = .init(allocator);
    const result = execute(allocator, io, &target, mutants, command, 5 * std.time.ns_per_s, "5.0", &interrupted, &progress.writer);
    sender.done.store(true, .release);
    thread.join();
    if (sender.failure) |err| return err;
    try std.testing.expectError(error.Interrupted, result);
    try std.testing.expectEqualStrings("x = 0;\n", try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited));
}
