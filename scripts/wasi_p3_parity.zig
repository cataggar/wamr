//! Run both Preview 3 suites and classify their reports even if a runner fails.
const std = @import("std");
const diff = @import("diff_testsuite_reports.zig");
const Allocator = std.mem.Allocator;

const Options = struct {
    output_dir: []const u8,
    root: []const u8 = ".",
    runner: ?[]const u8 = null,
    strict: bool = false,
};

fn parseOptions(args: []const []const u8) !Options {
    var result: Options = .{ .output_dir = "" };
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const raw = args[i];
        const equals = std.mem.indexOfScalar(u8, raw, '=');
        const option = if (equals) |index| raw[0..index] else raw;
        if (std.mem.eql(u8, option, "--strict")) {
            if (equals != null) return error.UnknownOption;
            result.strict = true;
            continue;
        }
        const value = if (equals) |index| raw[index + 1 ..] else blk: {
            i += 1;
            if (i == args.len) return error.MissingOptionValue;
            break :blk args[i];
        };
        if (std.mem.eql(u8, option, "--output-dir")) {
            result.output_dir = value;
        } else if (std.mem.eql(u8, option, "--root")) {
            result.root = value;
        } else if (std.mem.eql(u8, option, "--runner")) {
            result.runner = value;
        } else return error.UnknownOption;
    }
    if (result.output_dir.len == 0) return error.MissingOutputDirectory;
    return result;
}

fn defaultRunner(allocator: Allocator, root: []const u8) ![]const u8 {
    return std.fs.path.resolve(allocator, &.{
        root,
        "zig-out/bin",
        if (@import("builtin").os.tag == .windows) "wasi-suite-runner.exe" else "wasi-suite-runner",
    });
}

fn suiteCommand(allocator: Allocator, options: Options, adapter: []const u8, report: []const u8) ![]const []const u8 {
    const runner = options.runner orelse try defaultRunner(allocator, options.root);
    return allocator.dupe([]const u8, &.{
        runner,
        "--test-suite",
        try std.fs.path.resolve(allocator, &.{ options.root, "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip3" }),
        "--runtime-adapter",
        adapter,
        "--expectations",
        try std.fs.path.resolve(allocator, &.{ options.root, "tests/wasi-p3-testsuite-expectations.toml" }),
        "--json-output-location",
        report,
    });
}

fn runSuite(init: std.process.Init, options: Options, label: []const u8, adapter: []const u8, report: []const u8) !std.process.Child.Term {
    const allocator = init.arena.allocator();
    const command = try suiteCommand(allocator, options, adapter, report);
    const out = std.Io.File.stdout();
    try out.writeStreamingAll(init.io, try std.fmt.allocPrint(allocator, "\n=== {s} ===\n", .{label}));
    for (command, 0..) |arg, i| {
        if (i > 0) try out.writeStreamingAll(init.io, " ");
        try out.writeStreamingAll(init.io, arg);
    }
    try out.writeStreamingAll(init.io, "\n");
    var child = try std.process.spawn(init.io, .{ .argv = command });
    defer child.kill(init.io);
    const term = try child.wait(init.io);
    try out.writeStreamingAll(init.io, try std.fmt.allocPrint(allocator, "=== {s} exit={f} ===\n", .{ label, term }));
    return term;
}

fn missingReportStatus(term: std.process.Child.Term) u8 {
    return switch (term) {
        .exited => |code| if (code == 0) 2 else code,
        else => 2,
    };
}

fn run(init: std.process.Init, options: Options) !u8 {
    const allocator = init.arena.allocator();
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(init.io, options.output_dir);
    const wamr = try std.fs.path.join(allocator, &.{ options.output_dir, "wamr-p3.json" });
    const wasmtime = try std.fs.path.join(allocator, &.{ options.output_dir, "wasmtime-p3.json" });
    const summary = try std.fs.path.join(allocator, &.{ options.output_dir, "wasi-p3-parity.json" });
    const skip = try std.fs.path.resolve(allocator, &.{ options.root, "tests/wasi-p3-parity-skip.json" });

    const wamr_term = try runSuite(init, options, "wamr", "wamr-zig", wamr);
    const wasmtime_term = try runSuite(init, options, "wasmtime", "wasmtime", wasmtime);
    cwd.access(init.io, wamr, .{}) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("error: wamr report missing at {s} (runner exit {f}); aborting parity diff.\n", .{ wamr, wamr_term });
            return missingReportStatus(wamr_term);
        },
        else => return err,
    };
    cwd.access(init.io, wasmtime, .{}) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("error: wasmtime report missing at {s} (runner exit {f}); aborting parity diff.\n", .{ wasmtime, wasmtime_term });
            return missingReportStatus(wasmtime_term);
        },
        else => return err,
    };
    try std.Io.File.stdout().writeStreamingAll(init.io, "\n=== diff ===\n");
    const failed = try diff.run(init, .{
        .report_a = wamr,
        .report_b = wasmtime,
        .skip = skip,
        .output = summary,
        .strict = options.strict,
    });
    return if (failed) 1 else 0;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try std.Io.File.stdout().writeStreamingAll(init.io, "usage: wasi-p3-parity --output-dir DIR [--root REPO_ROOT] [--runner RUNNER] [--strict]\n");
            return;
        }
    }
    const options = parseOptions(args[1..]) catch |err| {
        std.debug.print("error: {s}\nusage: wasi-p3-parity --output-dir DIR [--root REPO_ROOT] [--runner RUNNER] [--strict]\n", .{@errorName(err)});
        std.process.exit(2);
    };
    const status = run(init, options) catch |err| {
        std.debug.print("error: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    if (status != 0) std.process.exit(status);
}

test "options and missing report exit status" {
    const options = try parseOptions(&.{ "--output-dir", "reports", "--strict", "--root", "/repo", "--runner", "/bin/wasi-suite-runner" });
    try std.testing.expectEqualStrings("reports", options.output_dir);
    try std.testing.expectEqualStrings("/repo", options.root);
    try std.testing.expectEqualStrings("/bin/wasi-suite-runner", options.runner.?);
    try std.testing.expect(options.strict);
    try std.testing.expectError(error.MissingOutputDirectory, parseOptions(&.{}));
    try std.testing.expectError(error.MissingOptionValue, parseOptions(&.{"--output-dir"}));
    try std.testing.expectError(error.UnknownOption, parseOptions(&.{ "--other", "x" }));
    try std.testing.expectEqualStrings("reports", (try parseOptions(&.{"--output-dir=reports"})).output_dir);
    try std.testing.expectError(error.UnknownOption, parseOptions(&.{ "--output-dir=reports", "--strict=true" }));
    try std.testing.expectEqual(@as(u8, 2), missingReportStatus(.{ .exited = 0 }));
    try std.testing.expectEqual(@as(u8, 7), missingReportStatus(.{ .exited = 7 }));
}
