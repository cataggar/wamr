//! Execution of discovered WASI suites using the in-tree runtime adapters.
const std = @import("std");
const suite = @import("wasi_fixture_suite.zig");
const runtime = @import("wasi_fixture_runtime.zig");
const meta = @import("wasi_runtime_meta.zig");
const prepare = @import("wasi_runtime_prepare.zig");
const execute = @import("wasi_fixture_execute.zig");
const expectations = @import("wasi_expectations.zig");
const report = @import("wasi_fixture_report.zig");
const profile = @import("wasi_fixture_profile.zig");
const sampling = @import("wasi_fixture_sampling.zig");
const Allocator = std.mem.Allocator;

pub const Progress = struct {
    context: *anyopaque,
    report: *const fn (context: *anyopaque, result: report.FixtureResult, argv: []const []const u8, failure_kinds: []const execute.FailureKind) anyerror!void,
};

pub const Options = struct {
    adapter: runtime.Runtime,
    executable: []const []const u8,
    compiler: []const []const u8 = &.{},
    compiler_mtime: i128 = 0,
    compiler_timeout_ms: u32 = 600_000,
    operation_timeout_ms: u32,
    wait_timeout_ms: u32,
    temp_path: []const u8,
    environment: *const std.process.Environ.Map,
    jit: bool = false,
    progress: ?Progress = null,
    timing: ?profile.Sink = null,
    sampling: ?sampling.Sampling = null,
};

fn seconds(io: std.Io, start: std.Io.Timestamp) f64 {
    const elapsed = start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
    return @as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0;
}

/// All returned data lives in the caller's allocator, normally an arena.
pub fn run(
    allocator: Allocator,
    io: std.Io,
    discovered: suite.Suite,
    runtime_version: []const u8,
    filters: []const expectations.Expectations,
    options: Options,
) !report.SuiteResult {
    const started = std.Io.Clock.awake.now(io);
    const results = try allocator.alloc(report.FixtureResult, discovered.cases.len);
    for (discovered.cases, 0..) |case, index| {
        const filter = expectations.combine(filters, discovered.manifest.name, case.name);
        if (filter.skip or !meta.supports(options.adapter, discovered.manifest.version, case.config.world)) {
            results[index] = .init(case.name, false, false, 0, &.{});
            if (options.progress) |progress| try progress.report(progress.context, results[index], &.{}, &.{});
            continue;
        }

        const fixture_started = std.Io.Clock.awake.now(io);
        var debug_environment: ?std.process.Environ.Map = null;
        defer if (debug_environment) |*env| env.deinit();
        if (case.config.debug) {
            debug_environment = try options.environment.clone(allocator);
            try debug_environment.?.put("DEBUG", "true");
        }
        var preparation: prepare.Preparation = .{
            .allocator = allocator,
            .io = io,
            .adapter = options.adapter,
            .executable = options.executable,
            .compiler = options.compiler,
            .compiler_mtime = options.compiler_mtime,
            .compiler_timeout_ms = options.compiler_timeout_ms,
            .fixture_path = case.path,
            .temp_path = options.temp_path,
            .config = case.config,
            .version = discovered.manifest.version,
            .jit = options.jit,
            .timing = options.timing,
            .sampling = options.sampling,
        };
        var alive = true;
        defer if (alive) preparation.deinit() catch |err|
            std.debug.print("failed to clean up fixture preparation: {s}\n", .{@errorName(err)});
        const result = try execute.executeWithProvider(allocator, io, case.config.operations, .{
            .context = &preparation,
            .command = prepare.Preparation.command,
        }, .{
            .operation_timeout_ms = options.operation_timeout_ms,
            .wait_timeout_ms = options.wait_timeout_ms,
            .environ_map = if (debug_environment) |*env| env else options.environment,
            .timing = if (options.timing) |sink| .{ .sink = sink, .fixture_path = case.path } else null,
        });
        alive = false;
        try preparation.deinit();
        results[index] = .init(case.name, true, filter.expected_fail, seconds(io, fixture_started), result.failures);
        if (options.progress) |progress| try progress.report(progress.context, results[index], result.argv, result.failure_kinds);
    }
    return .init(discovered.manifest.name, .{
        .name = meta.name(options.adapter),
        .version = runtime_version,
    }, seconds(io, started), results);
}

test "suite execution combines expectations, debug environment, and failure outcomes" {
    _ = @import("wasi_runner_options.zig");
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const fixture = @import("wasi_fixture_config.zig");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var environment = std.process.Environ.Map.init(allocator);
    defer environment.deinit();
    try environment.put("DEBUG", "original");
    const Recorder = struct {
        outcomes: [4]suite.Outcome = undefined,
        count: usize = 0,
        fn record(raw: *anyopaque, result: report.FixtureResult, argv: []const []const u8, failure_kinds: []const execute.FailureKind) !void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            if (self.count == self.outcomes.len) return error.TooManyResults;
            if (result.outcome == .skip) try std.testing.expectEqual(@as(usize, 0), argv.len) else try std.testing.expect(argv.len != 0);
            if (result.outcome == .xfail) {
                try std.testing.expectEqual(execute.FailureKind.expectation, failure_kinds[0]);
            } else {
                try std.testing.expectEqual(@as(usize, 0), failure_kinds.len);
            }

            self.outcomes[self.count] = result.outcome;
            self.count += 1;
        }
    };
    var recorder: Recorder = .{};
    const success_ops = [_]fixture.Operation{
        .{ .run = .{} },
        .{ .read = .{ .id = "stdout", .payload = "true" } },
        .{ .wait = 0 },
    };
    const failure_ops = [_]fixture.Operation{
        .{ .run = .{} },
        .{ .read = .{ .id = "stdout", .payload = "wrong" } },
        .{ .wait = 0 },
    };
    const tests = [_]suite.Case{
        .{ .name = "pass", .path = "pass.wasm", .config = .{ .operations = &success_ops, .debug = true } },
        .{ .name = "skip", .path = "skip.wasm", .config = .{ .operations = &success_ops } },
        .{ .name = "xfail", .path = "xfail.wasm", .config = .{ .operations = &failure_ops } },
        .{ .name = "xpass", .path = "xpass.wasm", .config = .{ .operations = &success_ops, .debug = true } },
    };
    var filters = try expectations.Expectations.parse(allocator,
        \\[[suite]]
        \\name = 'suite'
        \\[[suite.test]]
        \\name = 'skip'
        \\action = 'skip'
        \\[[suite.test]]
        \\name = 'xfail'
        \\expected = 'fail'
        \\[[suite.test]]
        \\name = 'xpass'
        \\expected = 'fail'
    );
    defer filters.deinit();
    const result = try run(allocator, io, .{
        .manifest = .{ .name = "suite", .version = .wasip1 },
        .cases = &tests,
    }, "1", &.{filters}, .{
        .adapter = .wasmtime,
        .executable = &.{ "/bin/sh", "-c", "printf %s \"$DEBUG\"", "sh" },
        .operation_timeout_ms = 5000,
        .wait_timeout_ms = 5000,
        .temp_path = ".zig-cache/tmp",
        .environment = &environment,
        .progress = .{ .context = &recorder, .report = Recorder.record },
    });
    try std.testing.expectEqual(@as(usize, 2), result.passed);
    try std.testing.expectEqual(@as(usize, 1), result.failed);
    try std.testing.expectEqual(@as(usize, 1), result.skipped);
    try std.testing.expectEqual(suite.Outcome.pass, result.tests[0].outcome);
    try std.testing.expectEqual(suite.Outcome.skip, result.tests[1].outcome);
    try std.testing.expectEqual(suite.Outcome.xfail, result.tests[2].outcome);
    try std.testing.expectEqual(suite.Outcome.xpass, result.tests[3].outcome);
    try std.testing.expectEqualSlices(suite.Outcome, &.{ .pass, .skip, .xfail, .xpass }, recorder.outcomes[0..recorder.count]);
    try std.testing.expectEqualStrings("original", environment.get("DEBUG").?);

    const unsupported = try run(allocator, io, .{
        .manifest = .{ .name = "suite", .version = .wasip2 },
        .cases = &tests,
    }, "1", &.{}, .{
        .adapter = .wasmtime,
        .executable = &.{"nonexistent"},
        .operation_timeout_ms = 5000,
        .wait_timeout_ms = 5000,
        .temp_path = ".zig-cache/tmp",
        .environment = &environment,
    });
    try std.testing.expectEqual(@as(usize, 4), unsupported.skipped);
    try std.testing.expectEqual(@as(usize, 0), unsupported.failed);
}

test "timing records AOT cache decisions and fixture waits" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const fixture = @import("wasi_fixture_config.zig");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "fixture.wasm", .data = "\x00asm\x0d\x00\x01\x00" });
    const path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/fixture.wasm", .{tmp.sub_path});
    const Recorder = struct {
        events: [10]profile.Event = undefined,
        count: usize = 0,
        fn emit(raw: *anyopaque, event: profile.Event) !void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            if (self.count == self.events.len) return error.TooManyEvents;
            self.events[self.count] = event;
            self.count += 1;
        }
    };
    var recorder: Recorder = .{};
    var environment = std.process.Environ.Map.init(allocator);
    defer environment.deinit();
    const operations = [_]fixture.Operation{ .{ .run = .{} }, .{ .wait = 0 } };
    const cases = [_]suite.Case{.{ .name = "fixture", .path = path, .config = .{ .operations = &operations } }};
    const discovered: suite.Suite = .{ .manifest = .{ .name = "suite", .version = .wasip1 }, .cases = &cases };
    var options: Options = .{
        .adapter = .wamr,
        .executable = &.{"/bin/true"},
        .compiler = &.{"/bin/true"},
        .operation_timeout_ms = 5000,
        .wait_timeout_ms = 5000,
        .temp_path = ".zig-cache/tmp",
        .environment = &environment,
        .jit = true,
        .timing = .{ .context = &recorder, .emit = Recorder.emit },
    };
    _ = try run(allocator, io, discovered, "1", &.{}, options);
    options.jit = false;
    _ = try run(allocator, io, discovered, "1", &.{}, options);
    try tmp.dir.writeFile(io, .{ .sub_path = "fixture.cwasm.json", .data = "{}" });
    _ = try run(allocator, io, discovered, "1", &.{}, options);

    try std.testing.expectEqual(@as(usize, 6), recorder.count);
    const caches = [_]runtime.Cache{ .bypass, .miss, .hit };
    for (caches, 0..) |cache, i| {
        const compile_event = recorder.events[2 * i];
        const execute_event = recorder.events[2 * i + 1];
        try std.testing.expectEqual(profile.Phase.component_precompile, compile_event.phase);
        try std.testing.expectEqual(cache, compile_event.cache);
        try std.testing.expectEqual(profile.Phase.fixture_execution, execute_event.phase);
        try std.testing.expectEqual(runtime.Cache.not_applicable, execute_event.cache);
        try std.testing.expectEqual(runtime.ArtifactKind.component, execute_event.artifact_kind);
        try std.testing.expectEqualStrings("fixture", execute_event.fixture);
    }
    try std.testing.expectEqual(@as(u64, 0), recorder.events[0].duration_ns);
    options.jit = true;
    options.executable = &.{ "/bin/sh", "-c", "while :; do :; done", "sh" };
    options.wait_timeout_ms = 50;
    const timed_out = try run(allocator, io, discovered, "1", &.{}, options);
    try std.testing.expectEqual(@as(usize, 1), timed_out.failed);
    try std.testing.expectEqual(@as(usize, 9), recorder.count);
    try std.testing.expectEqual(profile.Phase.fixture_execution, recorder.events[7].phase);
    try std.testing.expectEqual(profile.Phase.fixture_execution, recorder.events[8].phase);
    try std.testing.expect(recorder.events[8].duration_ns >= recorder.events[7].duration_ns);
}
