//! Experimental native mutation-testing CLI.
const std = @import("std");
const list = @import("mutation_list.zig");
const candidates = @import("mutation_candidates.zig");
const target_mod = @import("mutation_target.zig");
const execution = @import("mutation_execution.zig");
const report = @import("mutation_report.zig");
const float_text = @import("float_text.zig");
const Allocator = std.mem.Allocator;

const default_command = "unset ZIG_LOCAL_CACHE_DIR; export ZIG_GLOBAL_CACHE_DIR=\"$PWD/.zig-global-cache\"; zig build test --summary all 2>&1 | tail -5";
var interrupted: std.atomic.Value(bool) = .init(false);

fn onSignal(_: std.posix.SIG) callconv(.c) void {
    interrupted.store(true, .release);
}

const Options = struct {
    listing: list.Options,
    is_list: bool,
    seed_label: ?[]const u8,
    command: []const u8,
    output: ?[]const u8,
    timeout_ns: ?i96,
    timeout_text: []const u8,
};

fn seedLabel(allocator: Allocator, value: []const u8) ![]const u8 {
    var text = std.mem.trim(u8, value, " \t\r\n\x0b\x0c");
    const negative = text.len != 0 and text[0] == '-';
    if (text.len != 0 and (text[0] == '-' or text[0] == '+')) text = text[1..];
    var digits: std.ArrayList(u8) = .empty;
    for (text) |ch| if (ch != '_') try digits.append(allocator, ch);
    var first: usize = 0;
    while (first + 1 < digits.items.len and digits.items[first] == '0') : (first += 1) {}
    const value_digits = digits.items[first..];
    return std.fmt.allocPrint(allocator, "{s}{s}", .{ if (negative and !std.mem.eql(u8, value_digits, "0")) "-" else "", value_digits });
}

fn parseOptions(allocator: Allocator, args: []const []const u8) !Options {
    var base: std.ArrayList([]const u8) = .empty;
    try base.append(allocator, "--list");
    var command: []const u8 = default_command;
    var output: ?[]const u8 = null;
    var timeout_ns: ?i96 = null;
    var timeout_text: []const u8 = "";
    var seed_label: ?[]const u8 = null;
    var listing = false;
    var positional = false;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (!positional and (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")))
            return error.HelpRequested;
        if (!positional and std.mem.eql(u8, arg, "--")) {
            positional = true;
            try base.append(allocator, arg);
            continue;
        }
        if (!positional and std.mem.eql(u8, arg, "--list")) listing = true;
        const eq = if (!positional) std.mem.indexOfScalar(u8, arg, '=') else null;
        const name = if (eq) |index| arg[0..index] else arg;
        if (!positional and (std.mem.eql(u8, name, "--test-cmd") or std.mem.eql(u8, name, "--out") or
            std.mem.eql(u8, name, "--timeout") or std.mem.eql(u8, name, "--seed")))
        {
            const value = if (eq) |index| arg[index + 1 ..] else blk: {
                i += 1;
                if (i >= args.len) return error.MissingOptionValue;
                break :blk args[i];
            };
            if (std.mem.eql(u8, name, "--test-cmd")) {
                command = value;
            } else if (std.mem.eql(u8, name, "--out")) {
                output = value;
            } else if (std.mem.eql(u8, name, "--timeout")) {
                const seconds = try std.fmt.parseFloat(f64, value);
                if (!std.math.isFinite(seconds)) return error.InvalidTimeout;
                const nanoseconds = @ceil(@max(seconds, 0) * std.time.ns_per_s);
                if (!std.math.isFinite(nanoseconds) or nanoseconds >= 0x1p95) return error.InvalidTimeout;
                timeout_ns = @intFromFloat(nanoseconds);
                timeout_text = try float_text.format(allocator, seconds);
            } else {
                seed_label = try seedLabel(allocator, value);
                try base.appendSlice(allocator, &.{ "--seed", value });
            }
            continue;
        }
        try base.append(allocator, arg);
    }
    const base_options = try list.parseOptions(allocator, base.items);
    return .{ .listing = base_options, .is_list = listing, .seed_label = seed_label, .command = command, .output = output, .timeout_ns = timeout_ns, .timeout_text = timeout_text };
}

fn generatedAt(allocator: Allocator, nanos: i96) ![]const u8 {
    if (nanos < 0 or nanos >= 253_402_300_800 * std.time.ns_per_s) return error.InvalidTimestamp;
    const epoch: std.time.epoch.EpochSeconds = .{ .secs = @intCast(@divTrunc(nanos, std.time.ns_per_s)) };
    const date = epoch.getEpochDay().calculateYearDay();
    const day = date.calculateMonthDay();
    const time = epoch.getDaySeconds();
    return std.fmt.allocPrint(allocator, "{d:0>4}-{d:0>2}-{d:0>2} {d:0>2}:{d:0>2}:{d:0>2}Z", .{
        date.year, day.month.numeric(), @as(u8, day.day_index) + 1, time.getHoursIntoDay(), time.getMinutesIntoHour(), time.getSecondsIntoMinute(),
    });
}

fn run(init: std.process.Init, options: Options) !u8 {
    const allocator = init.arena.allocator();
    const io = init.io;
    var target = try target_mod.Target.open(allocator, io, options.listing.target);
    var closed = false;
    defer if (!closed) target.close() catch |err| std.debug.print("mutation target restoration failed: {s}\n", .{@errorName(err)});
    if (!std.mem.eql(u8, std.fs.path.extension(target.path), ".zig")) return error.InvalidTargetExtension;
    if (options.is_list) {
        const text = try list.listingText(allocator, target.original, options.listing);
        try target.close();
        closed = true;
        try std.Io.File.stdout().writeStreamingAll(io, text);
        return 0;
    }
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.UnsupportedMutationPlatform;
    if (options.output) |path| {
        if (std.Io.Dir.cwd().realPathFileAlloc(io, path, allocator)) |resolved| {
            if (std.mem.eql(u8, resolved, target.path)) return error.OutputWouldOverwriteTarget;
        } else |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        }
    }
    const all = try candidates.collect(allocator, target.source, options.listing.ranges);
    const selected = list.selected(all, options.listing);
    const sigs = [_]std.posix.SIG{ .INT, .TERM };
    var old: [2]std.posix.Sigaction = undefined;
    const action: std.posix.Sigaction = .{ .handler = .{ .handler = onSignal }, .mask = std.posix.sigemptyset(), .flags = 0 };
    interrupted.store(false, .release);
    for (sigs, 0..) |sig, index| std.posix.sigaction(sig, &action, &old[index]);
    defer for (sigs, 0..) |sig, index| std.posix.sigaction(sig, &old[index], null);
    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buffer);
    const results = execution.execute(allocator, io, &target, selected, options.command, options.timeout_ns, options.timeout_text, &interrupted, &stdout.interface) catch |err| {
        try target.close();
        closed = true;
        if (err == error.Interrupted) return 130;
        return err;
    };
    const cwd = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
    const report_target = if (std.mem.startsWith(u8, target.path, cwd) and target.path.len > cwd.len and target.path[cwd.len] == '/')
        target.path[cwd.len + 1 ..]
    else
        target.path;
    const limit = if (options.listing.limit) |value| value.label else null;
    const text = try report.render(allocator, .{
        .target = report_target,
        .generated = try generatedAt(allocator, std.Io.Clock.real.now(io).nanoseconds),
        .command = options.command,
        .seed = options.seed_label,
        .limit = limit,
        .ranges = options.listing.ranges,
        .total_available = all.len,
        .results = results,
    });
    try target.close();
    closed = true;
    if (options.output) |path| {
        if (std.fs.path.dirname(path)) |parent| try std.Io.Dir.cwd().createDirPath(io, parent);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text });
        try stdout.interface.print("wrote {s}\n", .{path});
    } else {
        try stdout.interface.print("{s}\n", .{text});
    }
    var survived: usize = 0;
    for (results) |result| survived += @intFromBool(result.survived);
    try stdout.interface.print("summary: {d} survived / {d} killed / {d} executed\n", .{ survived, results.len - survived, results.len });
    try stdout.interface.flush();
    return if (interrupted.load(.acquire)) 130 else 0;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parseOptions(init.arena.allocator(), args[1..]) catch |err| {
        if (err == error.HelpRequested) {
            try std.Io.File.stdout().writeStreamingAll(init.io,
                \\usage: mutation-run TARGET [--seed N] [--limit N] [--test-cmd COMMAND]
                \\                    [--out FILE] [--timeout SECONDS]
                \\                    [--line-range START:END] [--list]
                \\
                \\Run one-shot mutation tests for a Zig source file.
                \\The native implementation is experimental.
                \\
            );
            return;
        }
        std.debug.print("mutation-run: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    const status = run(init, options) catch |err| {
        std.debug.print("mutation-run: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    if (status != 0) std.process.exit(status);
}

test "CLI execution imports compile" {
    std.testing.refAllDecls(@This());
    _ = execution;
    _ = target_mod;
}

test "CLI retains signed seed, fractional timeout, and listing options" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const options = try parseOptions(allocator, &.{ "file.zig", "--seed=-00_737", "--limit", "3", "--timeout", "0.125", "--test-cmd", "false", "--out", "report.md" });
    try std.testing.expectEqualStrings("-737", options.seed_label.?);
    try std.testing.expectEqual(@as(?i96, 125 * std.time.ns_per_ms), options.timeout_ns);
    try std.testing.expectEqualStrings("0.125", options.timeout_text);
    try std.testing.expectEqualStrings("false", options.command);
    try std.testing.expectEqualStrings("report.md", options.output.?);
    try std.testing.expect(!options.is_list);
    try std.testing.expectError(error.InvalidTimeout, parseOptions(allocator, &.{ "file.zig", "--timeout", "inf" }));
    try std.testing.expectError(error.HelpRequested, parseOptions(allocator, &.{"--help"}));
    const short = try parseOptions(allocator, &.{ "file.zig", "--timeout", "0.0000001" });
    try std.testing.expectEqual(@as(?i96, 100), short.timeout_ns);
    const negative = try parseOptions(allocator, &.{ "file.zig", "--timeout", "-0.1" });
    try std.testing.expectEqual(@as(?i96, 0), negative.timeout_ns);
}
