//! Run the complete Preview 3 corpus and assert its unfiltered contract.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Dir = std.Io.Dir;
const Value = std.json.Value;
const expected_fixtures = 41;

const Contract = struct {
    fixtures: usize,
    executed: usize,
    failures: []const []const u8,
    passed: ?i64,
};

const Outcome = enum { wrong_count, failures, runner_failed, pass };

fn outcome(contract: Contract, term: std.process.Child.Term) Outcome {
    if (contract.fixtures != expected_fixtures or contract.executed != expected_fixtures) return .wrong_count;
    if (contract.failures.len != 0) return .failures;
    if (term != .exited or term.exited != 0) return .runner_failed;
    return .pass;
}

fn parseContract(allocator: Allocator, source: []const u8) !Contract {
    const root = try std.json.parseFromSliceLeaky(Value, allocator, source, .{});
    if (root != .object) return error.InvalidReport;
    const suites = root.object.get("results") orelse return error.InvalidSuiteCount;
    if (suites != .array or suites.array.items.len != 1) return error.InvalidSuiteCount;
    const suite = suites.array.items[0];
    if (suite != .object) return error.InvalidReport;
    const tests = suite.object.get("tests") orelse return error.InvalidReport;
    if (tests != .array) return error.InvalidReport;
    var executed: usize = 0;
    var failures: std.StringHashMap(void) = .init(allocator);
    for (tests.array.items) |test_value| {
        if (test_value != .object) return error.InvalidReport;
        const test_object = test_value.object;
        const is_executed = test_object.get("executed") orelse continue;
        if (is_executed != .bool) return error.InvalidReport;
        if (!is_executed.bool) continue;
        executed += 1;
        const errors = test_object.get("failures") orelse continue;
        if (errors == .null) continue;
        if (errors != .array) return error.InvalidReport;
        if (errors.array.items.len != 0) {
            const name = test_object.get("name") orelse return error.InvalidReport;
            if (name != .string) return error.InvalidReport;
            try failures.put(name.string, {});
        }
    }
    const names = try allocator.alloc([]const u8, failures.count());
    var it = failures.keyIterator();
    var index: usize = 0;
    while (it.next()) |name| : (index += 1) names[index] = name.*;
    std.mem.sort([]const u8, names, {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.less);
    const passed_value = suite.object.get("passed") orelse .null;
    const passed = switch (passed_value) {
        .integer => |value| value,
        .null => null,
        else => return error.InvalidReport,
    };
    return .{ .fixtures = tests.array.items.len, .executed = executed, .failures = names, .passed = passed };
}

fn isSidecar(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".cwasm") or std.mem.endsWith(u8, name, ".cwasm.json");
}

fn removeJitSidecars(io: std.Io, suite_path: []const u8) !void {
    const suite = Dir.cwd().openDir(io, suite_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    defer suite.close(io);
    var iterator = suite.iterate();
    while (try iterator.next(io)) |entry| {
        if (isSidecar(entry.name)) try suite.deleteFile(io, entry.name);
    }
    iterator = suite.iterate();
    while (try iterator.next(io)) |entry| {
        if (isSidecar(entry.name)) {
            std.debug.print("error: JIT P3 run has component sidecar: {s}\n", .{entry.name});
            return error.JitSidecarsRemain;
        }
    }
}

fn missingReportStatus(term: std.process.Child.Term) u8 {
    return switch (term) {
        .exited => |code| if (code == 0) 2 else code,
        .signal => |signal| @truncate(@as(u32, 256) -% @as(u32, @intCast(@backingInt(signal)))),
        else => 2,
    };
}

fn run(init: std.process.Init, root_path: []const u8, runner_override: ?[]const u8) !u8 {
    const allocator = init.arena.allocator();
    const io = init.io;
    const cwd = Dir.cwd();
    const root = try cwd.realPathFileAlloc(io, root_path, allocator);
    const suite = try std.fs.path.resolve(allocator, &.{ root, "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip3" });
    const jit = if (init.environ_map.get("WAMR_JIT_TESTSUITE")) |value| value.len != 0 else false;
    if (jit) try removeJitSidecars(io, suite);
    const override = init.environ_map.get("WAMR_P3_REPORT");
    const report = if (override != null and override.?.len != 0) override.? else try std.fs.path.join(allocator, &.{ root, "zig-out", if (jit) "wasi-p3-unfiltered-jit.json" else "wasi-p3-unfiltered-aot.json" });
    try cwd.createDirPath(io, std.fs.path.dirname(report) orelse ".");
    cwd.deleteFile(io, report) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };
    const default_runner = try std.fs.path.join(allocator, &.{
        root,
        "zig-out/bin",
        if (@import("builtin").os.tag == .windows) "wasi-suite-runner.exe" else "wasi-suite-runner",
    });
    const runner = runner_override orelse default_runner;
    const args = [_][]const u8{
        runner,
        "--test-suite",
        suite,
        "--runtime-adapter",
        "wamr-zig",
        "--json-output-location",
        report,
    };
    const out = std.Io.File.stdout();
    for (args, 0..) |arg, index| {
        if (index != 0) try out.writeStreamingAll(io, " ");
        try out.writeStreamingAll(io, arg);
    }
    try out.writeStreamingAll(io, "\n");
    var child = try std.process.spawn(io, .{ .argv = &args });
    defer child.kill(io);
    const term = try child.wait(io);
    const info = cwd.statFile(io, report, .{}) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => {
            std.debug.print("error: P3 runner did not create {s}\n", .{report});
            return missingReportStatus(term);
        },
        else => return err,
    };
    if (info.kind != .file) {
        std.debug.print("error: P3 runner did not create {s}\n", .{report});
        return missingReportStatus(term);
    }
    const source = try cwd.readFileAlloc(io, report, allocator, .unlimited);
    const contract = try parseContract(allocator, source);
    switch (outcome(contract, term)) {
        .wrong_count => {
            std.debug.print("error: unfiltered P3 contract violated: fixtures={d}, executed={d}, expected={d}\n", .{
                contract.fixtures, contract.executed, expected_fixtures,
            });
            return 2;
        },
        .failures => {
            std.debug.print("error: unfiltered P3 failures:", .{});
            for (contract.failures) |failure| std.debug.print(" {s}", .{failure});
            std.debug.print("\n", .{});
            return 2;
        },
        .runner_failed => {
            std.debug.print("error: P3 runner exited with status {f}\n", .{term});
            return 2;
        },
        .pass => {},
    }
    if (contract.passed) |passed| {
        try out.writeStreamingAll(io, try std.fmt.allocPrint(allocator, "unfiltered P3 contract: executed={d}/{d}, passed={d}\n", .{ contract.executed, expected_fixtures, passed }));
    } else {
        try out.writeStreamingAll(io, try std.fmt.allocPrint(allocator, "unfiltered P3 contract: executed={d}/{d}, passed=None\n", .{ contract.executed, expected_fixtures }));
    }
    return 0;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2 and args.len != 3) {
        std.debug.print("usage: wasi-p3-unfiltered REPOSITORY_ROOT [RUNNER]\n", .{});
        std.process.exit(2);
    }
    const runner_override = if (args.len == 3) args[2] else null;
    const status = run(init, args[1], runner_override) catch |err| {
        std.debug.print("error: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    if (status != 0) std.process.exit(status);
}

test "unfiltered report distinguishes skipped and failing fixtures" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const contract = try parseContract(allocator,
        \\{"results":[{"passed":1,"tests":[
        \\{"name":"ok","executed":true,"failures":[]},
        \\{"name":"skip","executed":false,"failures":[]},
        \\{"name":"z","executed":true,"failures":["bad"]},
        \\{"name":"a","executed":true,"failures":["bad"]},
        \\{"name":"a","executed":true,"failures":["also bad"]}
        \\]}]}
    );
    try std.testing.expectEqual(@as(usize, 5), contract.fixtures);
    try std.testing.expectEqual(@as(usize, 4), contract.executed);
    try std.testing.expectEqualDeep(&[_][]const u8{ "a", "z" }, contract.failures);
    try std.testing.expectEqual(@as(?i64, 1), contract.passed);
    try std.testing.expectError(error.InvalidSuiteCount, parseContract(allocator, "{\"results\":[]}"));
    try std.testing.expectError(error.InvalidReport, parseContract(allocator, "{\"results\":[{\"tests\":[{\"executed\":\"yes\"}]}]}"));
    try std.testing.expectError(error.InvalidReport, parseContract(allocator, "{\"results\":[{\"tests\":[{\"executed\":true,\"failures\":[\"bad\"]}]}]}"));
    var valid: Contract = .{ .fixtures = 41, .executed = 41, .failures = &.{}, .passed = 41 };
    try std.testing.expectEqual(Outcome.pass, outcome(valid, .{ .exited = 0 }));
    try std.testing.expectEqual(Outcome.runner_failed, outcome(valid, .{ .exited = 1 }));
    valid.failures = &.{"failure"};
    try std.testing.expectEqual(Outcome.failures, outcome(valid, .{ .exited = 0 }));
    valid.executed = 40;
    try std.testing.expectEqual(Outcome.wrong_count, outcome(valid, .{ .exited = 0 }));
}

test "JIT cleanup removes only direct sidecars" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "test.cwasm", .data = "" });
    try tmp.dir.writeFile(io, .{ .sub_path = "test.cwasm.json", .data = "" });
    try tmp.dir.writeFile(io, .{ .sub_path = "test.wasm", .data = "" });
    try tmp.dir.createDirPath(io, "nested");
    try tmp.dir.writeFile(io, .{ .sub_path = "nested/test.cwasm", .data = "" });
    var buffer: [std.fs.max_path_bytes]u8 = undefined;
    const length = try tmp.dir.realPath(io, &buffer);
    try removeJitSidecars(io, buffer[0..length]);
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, "test.cwasm", .{}));
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, "test.cwasm.json", .{}));
    try tmp.dir.access(io, "test.wasm", .{});
    try tmp.dir.access(io, "nested/test.cwasm", .{});
    try std.testing.expectEqual(@as(u8, 2), missingReportStatus(.{ .exited = 0 }));
    try std.testing.expectEqual(@as(u8, 7), missingReportStatus(.{ .exited = 7 }));
}
