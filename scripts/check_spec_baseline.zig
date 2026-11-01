const std = @import("std");
const process_control = @import("process_control.zig");

const Allocator = std.mem.Allocator;
const Counts = struct { passed: u64, failed: u64, skipped: u64 };
const Result = struct {
    name: []const u8,
    source: []const u8,
    workdir: []const u8,
    counts: ?Counts = null,
    status: []const u8 = "not run",
    exit_code: ?u8 = null,
};

const Options = struct {
    runner: []const u8 = "zig-out/bin/spec-test-runner",
    suite: []const u8 = "third_party/testsuite",
    baseline: []const u8 = "tests/spec-baseline.tsv",
    jobs: usize = 8,
    update: bool = false,
};

fn parseOptions(args: []const []const u8) !Options {
    var opts: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const raw = args[i];
        const equals = std.mem.indexOfScalar(u8, raw, '=');
        const arg = if (equals) |index| raw[0..index] else raw;
        if (std.mem.eql(u8, arg, "--update")) {
            if (equals != null) return error.InvalidOption;
            opts.update = true;
        } else if (std.mem.eql(u8, arg, "--runner") or std.mem.eql(u8, arg, "--suite") or
            std.mem.eql(u8, arg, "--baseline") or std.mem.eql(u8, arg, "--jobs"))
        {
            const value = if (equals) |index| raw[index + 1 ..] else blk: {
                i += 1;
                if (i == args.len) return error.MissingOptionValue;
                break :blk args[i];
            };
            if (std.mem.eql(u8, arg, "--runner")) {
                opts.runner = value;
            } else if (std.mem.eql(u8, arg, "--suite")) {
                opts.suite = value;
            } else if (std.mem.eql(u8, arg, "--baseline")) {
                opts.baseline = value;
            } else {
                opts.jobs = try std.fmt.parseInt(usize, value, 10);
                if (opts.jobs == 0) return error.InvalidJobCount;
            }
        } else {
            std.debug.print("unknown option: {s}\n", .{arg});
            return error.InvalidOption;
        }
    }
    return opts;
}

fn numberAt(text: []const u8, pos: *usize) ?u64 {
    const start = pos.*;
    while (pos.* < text.len and std.ascii.isDigit(text[pos.*])) : (pos.* += 1) {}
    if (pos.* == start) return null;
    return std.fmt.parseInt(u64, text[start..pos.*], 10) catch null;
}

fn parseCounts(text: []const u8) ?Counts {
    var search: usize = 0;
    while (search < text.len) {
        const offset = std.mem.indexOf(u8, text[search..], "pass=") orelse return null;
        var pos = search + offset + "pass=".len;
        search = pos;
        const passed = numberAt(text, &pos) orelse continue;
        const before_space = pos;
        while (pos < text.len and std.ascii.isWhitespace(text[pos])) : (pos += 1) {}
        if (pos == before_space or !std.mem.startsWith(u8, text[pos..], "fail=")) continue;
        pos += "fail=".len;
        const failed = numberAt(text, &pos) orelse continue;
        const before_skip = pos;
        while (pos < text.len and std.ascii.isWhitespace(text[pos])) : (pos += 1) {}
        if (pos == before_skip or !std.mem.startsWith(u8, text[pos..], "skip=")) continue;
        pos += "skip=".len;
        const skipped = numberAt(text, &pos) orelse continue;
        return .{ .passed = passed, .failed = failed, .skipped = skipped };
    }
    return null;
}

fn readBaseline(allocator: Allocator, text: []const u8) !std.StringHashMap(Counts) {
    var map: std.StringHashMap(Counts) = .init(allocator);
    errdefer map.deinit();
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, "\r");
        if (std.mem.trim(u8, line, " \t\r").len == 0 or line[0] == '#') continue;
        var fields = std.mem.splitScalar(u8, line, '\t');
        const name = fields.next().?;
        if (name.len == 0) return error.InvalidBaseline;
        const passed = try std.fmt.parseInt(u64, fields.next() orelse return error.InvalidBaseline, 10);
        const failed = try std.fmt.parseInt(u64, fields.next() orelse return error.InvalidBaseline, 10);
        const skipped = try std.fmt.parseInt(u64, fields.next() orelse return error.InvalidBaseline, 10);
        if (fields.next() != null) return error.InvalidBaseline;
        const entry = try map.getOrPut(name);
        if (entry.found_existing) return error.DuplicateBaselineEntry;
        entry.value_ptr.* = .{ .passed = passed, .failed = failed, .skipped = skipped };
    }
    if (map.count() == 0) return error.EmptyBaseline;
    return map;
}

fn hasDrift(actual: Counts, baseline: Counts) bool {
    return actual.passed < baseline.passed or actual.failed > baseline.failed or actual.skipped > baseline.skipped;
}

const Work = struct {
    io: std.Io,
    runner: []const u8,
    results: []Result,
    next: std.atomic.Value(usize) = .init(0),

    fn run(self: *Work) void {
        while (true) {
            const index = self.next.fetchAdd(1, .monotonic);
            if (index >= self.results.len) return;
            self.runOne(&self.results[index]);
        }
    }

    fn runOne(self: *Work, item: *Result) void {
        const allocator = std.heap.page_allocator;
        std.Io.Dir.cwd().createDirPath(self.io, item.workdir) catch |err| {
            item.status = @errorName(err);
            return;
        };
        const output = process_control.runTimed(allocator, self.io, .{
            .argv = &.{ self.runner, item.source },
            .cwd = .{ .path = item.workdir },
            .stdout_limit = .limited(64 * 1024 * 1024),
            .stderr_limit = .limited(64 * 1024 * 1024),
        }, 600) catch |err| {
            item.status = @errorName(err);
            return;
        };
        defer allocator.free(output.stdout);
        defer allocator.free(output.stderr);
        switch (output.term) {
            .exited => |code| if (code != 0) {
                item.status = "EXIT";
                item.exit_code = code;
                return;
            },
            .signal => {
                item.status = "SIGNAL";
                return;
            },
            else => {
                item.status = "unexpected termination";
                return;
            },
        }
        item.counts = parseCounts(output.stdout) orelse parseCounts(output.stderr) orelse {
            item.status = "NO-OUTPUT";
            return;
        };
        item.status = "OK";
    }
};

const header =
    \\# Spec-test baseline. Each line lists the expected pass/fail/skip counts
    \\# for one .wast file under third_party/testsuite/, as produced by
    \\# zig-out/bin/spec-test-runner. CI compares live output against this file
    \\# and fails on any drift. Regenerate with:
    \\#     zig build update-spec-baseline
    \\
;

fn run(init: std.process.Init, opts: Options) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const runner = cwd.realPathFileAlloc(io, opts.runner, allocator) catch |err| {
        std.debug.print("error: runner not found: {s}: {s}\n", .{ opts.runner, @errorName(err) });
        return err;
    };
    const suite = try cwd.realPathFileAlloc(io, opts.suite, allocator);
    var suite_dir = try cwd.openDir(io, suite, .{ .iterate = true });
    defer suite_dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var iter = suite_dir.iterate();
    while (try iter.next(io)) |entry| {
        if (std.mem.endsWith(u8, entry.name, ".wast")) {
            if (entry.kind != .file and entry.kind != .sym_link) {
                std.debug.print("error: invalid suite entry: {s}\n", .{entry.name});
                return error.InvalidSuiteEntry;
            }
            try names.append(allocator, try allocator.dupe(u8, entry.name));
        }
    }
    if (names.items.len == 0) {
        std.debug.print("error: no .wast files under {s}\n", .{opts.suite});
        return error.EmptySuite;
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.less);

    try cwd.createDirPath(io, ".zig-cache");
    var random: [16]u8 = undefined;
    io.random(&random);
    const scratch = try std.fmt.allocPrint(allocator, ".zig-cache/spec-baseline-{s}", .{std.fmt.bytesToHex(random, .lower)});
    try cwd.createDir(io, scratch, .default_dir);
    defer cwd.deleteTree(io, scratch) catch |err| {
        std.debug.print("error cleaning {s}: {s}\n", .{ scratch, @errorName(err) });
    };
    const results = try allocator.alloc(Result, names.items.len);
    for (names.items, 0..) |name, i| {
        results[i] = .{
            .name = name,
            .source = try std.fs.path.join(allocator, &.{ suite, name }),
            .workdir = try std.fmt.allocPrint(allocator, "{s}/{d}", .{ scratch, i }),
        };
    }
    var work: Work = .{ .io = io, .runner = runner, .results = results };
    const count = @min(opts.jobs, results.len);
    const workers = try allocator.alloc(std.Thread, count);
    var started: usize = 0;
    defer for (workers[0..started]) |thread| thread.join();
    while (started < count) : (started += 1) {
        workers[started] = try std.Thread.spawn(.{}, Work.run, .{&work});
    }
    for (workers[0..started]) |thread| thread.join();
    started = 0;

    if (opts.update) {
        var output: std.ArrayList(u8) = .empty;
        try output.appendSlice(allocator, header);
        for (results) |item| {
            const counts = item.counts orelse {
                if (item.exit_code) |code| {
                    std.debug.print("error: refusing to baseline {s}: EXIT{d}\n", .{ item.name, code });
                } else {
                    std.debug.print("error: refusing to baseline {s}: {s}\n", .{ item.name, item.status });
                }
                return error.RunFailed;
            };
            const line = try std.fmt.allocPrint(allocator, "{s}\t{d}\t{d}\t{d}\n", .{ item.name, counts.passed, counts.failed, counts.skipped });
            try output.appendSlice(allocator, line);
        }
        try cwd.writeFile(io, .{ .sub_path = opts.baseline, .data = output.items });
        var total: u64 = 0;
        for (results) |item| total += item.counts.?.passed;
        try std.Io.File.stdout().writeStreamingAll(io, try std.fmt.allocPrint(allocator, "wrote {s} ({d} files, {d} passing assertions)\n", .{ opts.baseline, results.len, total }));
        return;
    }

    const text = try cwd.readFileAlloc(io, opts.baseline, allocator, .limited(16 * 1024 * 1024));
    var baseline = try readBaseline(allocator, text);
    var total_pass: u64 = 0;
    var total_base: u64 = 0;
    for (results) |item| if (item.counts) |counts| {
        total_pass += counts.passed;
    };
    var baseline_values = baseline.valueIterator();
    while (baseline_values.next()) |counts| total_base += counts.passed;
    var failed = false;
    for (results) |item| {
        const expected = baseline.fetchRemove(item.name) orelse {
            std.debug.print("{s}: in testsuite but missing from baseline\n", .{item.name});
            failed = true;
            continue;
        };
        const actual = item.counts orelse {
            if (item.exit_code) |code| {
                std.debug.print("{s}: EXIT{d}\n", .{ item.name, code });
            } else {
                std.debug.print("{s}: {s}\n", .{ item.name, item.status });
            }
            failed = true;
            continue;
        };
        if (hasDrift(actual, expected.value)) {
            std.debug.print("{s}: pass/fail/skip {d}/{d}/{d} vs baseline {d}/{d}/{d}\n", .{
                item.name,             actual.passed,         actual.failed,          actual.skipped,
                expected.value.passed, expected.value.failed, expected.value.skipped,
            });
            failed = true;
        }
    }
    var remaining = baseline.iterator();
    while (remaining.next()) |entry| {
        std.debug.print("{s}: in baseline but missing from testsuite\n", .{entry.key_ptr.*});
        failed = true;
    }
    try std.Io.File.stdout().writeStreamingAll(io, try std.fmt.allocPrint(allocator, "{d} files, {d} passing assertions (baseline {d})\n", .{ results.len, total_pass, total_base }));
    if (failed) return error.BaselineDrift;
    try std.Io.File.stdout().writeStreamingAll(io, "no drift\n");
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try std.Io.File.stdout().writeStreamingAll(init.io, "Usage: check-spec-baseline [--runner PATH] [--suite PATH] [--baseline PATH] [--jobs N] [--update]\n");
            return;
        }
    }
    const options = parseOptions(args[1..]) catch |err| {
        std.debug.print("error parsing options: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
    run(init, options) catch |err| {
        switch (err) {
            error.BaselineDrift, error.RunFailed => {},
            else => std.debug.print("error: {s}\n", .{@errorName(err)}),
        }
        std.process.exit(1);
    };
}

test "counts are parsed only from complete result lines" {
    try std.testing.expectEqual(Counts{ .passed = 12, .failed = 3, .skipped = 4 }, parseCounts("bad pass=x fail=0 skip=0\npass=12  fail=3 skip=4 total=19").?);
    try std.testing.expect(parseCounts("pass=12 fail=3") == null);
}

test "baseline rejects malformed and duplicate entries" {
    var valid = try readBaseline(std.testing.allocator, "# header\na.wast\t2\t0\t1\n");
    defer valid.deinit();
    try std.testing.expectEqual(@as(u64, 2), valid.get("a.wast").?.passed);
    try std.testing.expectError(error.DuplicateBaselineEntry, readBaseline(std.testing.allocator, "a\t1\t0\t0\na\t1\t0\t0\n"));
    try std.testing.expectError(error.InvalidBaseline, readBaseline(std.testing.allocator, "a\t1\t0\n"));
}

test "job count must be positive" {
    try std.testing.expectError(error.InvalidJobCount, parseOptions(&.{ "--jobs", "0" }));
    try std.testing.expectEqual(@as(usize, 2), (try parseOptions(&.{ "--jobs", "2" })).jobs);
    try std.testing.expectEqual(@as(usize, 3), (try parseOptions(&.{"--jobs=3"})).jobs);
}

test "baseline permits improvements but rejects each regression" {
    const baseline: Counts = .{ .passed = 10, .failed = 2, .skipped = 3 };
    try std.testing.expect(!hasDrift(.{ .passed = 11, .failed = 1, .skipped = 2 }, baseline));
    try std.testing.expect(hasDrift(.{ .passed = 9, .failed = 2, .skipped = 3 }, baseline));
    try std.testing.expect(hasDrift(.{ .passed = 10, .failed = 3, .skipped = 3 }, baseline));
    try std.testing.expect(hasDrift(.{ .passed = 10, .failed = 2, .skipped = 4 }, baseline));
}

test "runner output and failure status are captured" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.writeFile(io, .{ .sub_path = "pass", .data = "printf 'pass=4 fail=1 skip=2\\n' >&2\n" });
    try temp.dir.writeFile(io, .{ .sub_path = "fail", .data = "exit 7\n" });
    const pass = try temp.dir.realPathFileAlloc(io, "pass", allocator);
    defer allocator.free(pass);
    const fail = try temp.dir.realPathFileAlloc(io, "fail", allocator);
    defer allocator.free(fail);
    const dir = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(dir);
    const workdir = try std.fs.path.join(allocator, &.{ dir, "work" });
    defer allocator.free(workdir);
    var work: Work = .{ .io = io, .runner = "/bin/sh", .results = &.{} };
    var item: Result = .{ .name = "pass", .source = pass, .workdir = workdir };
    work.runOne(&item);
    try std.testing.expectEqual(Counts{ .passed = 4, .failed = 1, .skipped = 2 }, item.counts.?);
    item = .{ .name = "fail", .source = fail, .workdir = workdir };
    work.runOne(&item);
    try std.testing.expect(item.counts == null);
    try std.testing.expectEqual(@as(?u8, 7), item.exit_code);
}

test "timeout includes a child that closes its output before exiting" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    try std.testing.expectError(error.Timeout, process_control.runTimed(std.testing.allocator, std.testing.io, .{
        .argv = &.{ "/bin/sh", "-c", "exec 1>&- 2>&-; exec sleep 30" },
    }, 1));
}

test "timed out children are terminated even if they ignore SIGTERM" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const path = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(path);
    for ([_][]const u8{
        "trap '' TERM; exec 1>&- 2>&-; sleep 2; touch escaped",
        "trap '' TERM; sleep 2; touch escaped",
    }) |script| {
        try std.testing.expectError(error.Timeout, process_control.runTimed(allocator, io, .{
            .argv = &.{ "/bin/sh", "-c", script },
            .cwd = .{ .path = path },
        }, 1));
        try io.sleep(.fromSeconds(3), .awake);
        try std.testing.expectError(error.FileNotFound, temp.dir.access(io, "escaped", .{}));
    }
}
