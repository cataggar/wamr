//! End-to-end smoke coverage for the native mutation CLI.
const std = @import("std");
const process_control = @import("process_control.zig");

test "CLI writes a report and restores the original file" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const executable = if (std.c.getenv("MUTATION_RUNNER")) |path| std.mem.span(path) else return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const target = try std.fs.path.join(allocator, &.{ root, "test.zig" });
    defer allocator.free(target);
    const output = try std.fs.path.join(allocator, &.{ root, "report.md" });
    defer allocator.free(output);
    try temp.dir.writeFile(io, .{ .sub_path = "test.zig", .data = "x = 0;\n" });
    const run = try std.process.run(allocator, io, .{
        .argv = &.{ executable, target, "--limit", "1", "--test-cmd", "printf diagnostic >&2; exit 7", "--out", output },
        .timeout = .{ .deadline = .fromNow(io, .{ .clock = .awake, .raw = .fromSeconds(30) }) },
    });
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    try std.testing.expectEqual(@as(u8, 0), run.term.exited);
    try std.testing.expect(std.mem.indexOf(u8, run.stdout, "summary: 0 survived / 1 killed / 1 executed") != null);
    try std.testing.expectEqualStrings("", run.stderr);
    const original = try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited);
    defer allocator.free(original);
    try std.testing.expectEqualStrings("x = 0;\n", original);
    const markdown = try temp.dir.readFileAlloc(io, "report.md", allocator, .unlimited);
    defer allocator.free(markdown);
    try std.testing.expect(std.mem.indexOf(u8, markdown, "| KILLED | 1 | integer 0 -&gt; 1 at column 5 | `x = 0;` | 7 | ") != null);
    try std.testing.expect(std.mem.indexOf(u8, markdown, "| `diagnostic` |") != null);
}

test "CLI refuses to overwrite the mutation target with a report" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const executable = if (std.c.getenv("MUTATION_RUNNER")) |path| std.mem.span(path) else return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.writeFile(io, .{ .sub_path = "test.zig", .data = "x = 0;\n" });
    const target = try temp.dir.realPathFileAlloc(io, "test.zig", allocator);
    defer allocator.free(target);
    const run = try std.process.run(allocator, io, .{
        .argv = &.{ executable, target, "--out", target, "--test-cmd", "exit 0" },
        .timeout = .{ .deadline = .fromNow(io, .{ .clock = .awake, .raw = .fromSeconds(30) }) },
    });
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    try std.testing.expectEqual(@as(u8, 2), run.term.exited);
    try std.testing.expect(std.mem.indexOf(u8, run.stderr, "OutputWouldOverwriteTarget") != null);
    const actual = try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited);
    defer allocator.free(actual);
    try std.testing.expectEqualStrings("x = 0;\n", actual);
}

test "CLI reports arbitrary-size negative limits" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const executable = if (std.c.getenv("MUTATION_RUNNER")) |path| std.mem.span(path) else return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.writeFile(io, .{ .sub_path = "test.zig", .data = "x = 0;\n" });
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const target = try std.fs.path.join(allocator, &.{ root, "test.zig" });
    defer allocator.free(target);
    const output = try std.fs.path.join(allocator, &.{ root, "report.md" });
    defer allocator.free(output);
    const result = try std.process.run(allocator, io, .{
        .argv = &.{ executable, target, "--limit", "-999999999999999999999999999999", "--out", output },
        .timeout = .{ .deadline = .fromNow(io, .{ .clock = .awake, .raw = .fromSeconds(30) }) },
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try std.testing.expectEqual(@as(u8, 0), result.term.exited);
    try std.testing.expect(std.mem.indexOf(u8, result.stdout, "summary: 0 survived / 0 killed / 0 executed") != null);
    const markdown = try temp.dir.readFileAlloc(io, "report.md", allocator, .unlimited);
    defer allocator.free(markdown);
    try std.testing.expect(std.mem.indexOf(u8, markdown, "- Limit: -999999999999999999999999999999\n") != null);
    const original = try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited);
    defer allocator.free(original);
    try std.testing.expectEqualStrings("x = 0;\n", original);
}

test "CLI retains a sub-millisecond timeout in its report" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const executable = if (std.c.getenv("MUTATION_RUNNER")) |path| std.mem.span(path) else return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.writeFile(io, .{ .sub_path = "test.zig", .data = "x = 0;\n" });
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const target = try std.fs.path.join(allocator, &.{ root, "test.zig" });
    defer allocator.free(target);
    const output = try std.fs.path.join(allocator, &.{ root, "report.md" });
    defer allocator.free(output);
    const result = try std.process.run(allocator, io, .{
        .argv = &.{ executable, target, "--timeout", "0.0000001", "--test-cmd", "sleep 2", "--out", output },
        .timeout = .{ .deadline = .fromNow(io, .{ .clock = .awake, .raw = .fromSeconds(30) }) },
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try std.testing.expectEqual(@as(u8, 0), result.term.exited);
    try std.testing.expect(std.mem.indexOf(u8, result.stdout, "summary: 0 survived / 1 killed / 1 executed") != null);
    const markdown = try temp.dir.readFileAlloc(io, "report.md", allocator, .unlimited);
    defer allocator.free(markdown);
    try std.testing.expect(std.mem.indexOf(u8, markdown, "timed out after 1e-07 seconds") != null);
    const original = try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited);
    defer allocator.free(original);
    try std.testing.expectEqualStrings("x = 0;\n", original);
}

test "CLI restores an active mutation on termination" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const executable = if (std.c.getenv("MUTATION_RUNNER")) |path| std.mem.span(path) else return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.writeFile(io, .{ .sub_path = "test.zig", .data = "x = 0;\n" });
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const target = try std.fs.path.join(allocator, &.{ root, "test.zig" });
    defer allocator.free(target);
    var child = try std.process.spawn(io, .{
        .argv = &.{ executable, target, "--test-cmd", "printf ready > marker; exec sleep 2" },
        .cwd = .{ .path = root },
        .stdout = .ignore,
        .stderr = .ignore,
    });
    defer if (child.id != null) child.kill(io);
    const start = std.Io.Clock.awake.now(io);
    while (true) {
        if (temp.dir.statFile(io, "marker", .{})) |_| {
            break;
        } else |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        }
        if (start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds >= 5 * std.time.ns_per_s)
            return error.MutationCommandDidNotStart;
        try io.sleep(.fromMilliseconds(10), .awake);
    }
    const changed = try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited);
    defer allocator.free(changed);
    try std.testing.expectEqualStrings("x = 1;\n", changed);
    try std.posix.kill(child.id.?, .TERM);
    const terminated = while (true) {
        if (try process_control.pollChild(&child, io)) |term| break term;
        if (start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds >= 10 * std.time.ns_per_s)
            return error.MutationCommandDidNotStop;
        try io.sleep(.fromMilliseconds(10), .awake);
    };
    try std.testing.expectEqual(@as(u8, 130), terminated.exited);
    const restored = try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited);
    defer allocator.free(restored);
    try std.testing.expectEqualStrings("x = 0;\n", restored);
}
