//! Native WASI conformance test suite runner.
const std = @import("std");
const options_parser = @import("wasi_runner_options.zig");
const fixture_suite = @import("wasi_fixture_suite.zig");
const suite_runner = @import("wasi_fixture_suite_runner.zig");
const runtime = @import("wasi_fixture_runtime.zig");
const runtime_env = @import("wasi_runtime_env.zig");
const temp_root = @import("temp_root");
const runtime_meta = @import("wasi_runtime_meta.zig");
const execute = @import("wasi_fixture_execute.zig");
const expectations = @import("wasi_expectations.zig");
const report = @import("wasi_fixture_report.zig");
const profile = @import("wasi_fixture_profile.zig");
const sampling = @import("wasi_fixture_sampling.zig");
const Allocator = std.mem.Allocator;

const usage = "usage: wasi-suite-runner -t SUITE [SUITE ...] -r ADAPTER [-e EXPECTATIONS ...] [--json-output-location FILE] [--disable-colors]\n";

fn adapterKind(path: []const u8) !runtime.Runtime {
    if (std.mem.eql(u8, path, "wamr") or std.mem.eql(u8, path, "wamr-zig")) return .wamr;
    if (std.mem.eql(u8, path, "wasmtime")) return .wasmtime;
    return error.UnsupportedRuntimeAdapter;
}

fn command(allocator: Allocator, value: []const u8) ![]const []const u8 {
    const parts = try runtime_env.splitCommand(allocator, value);
    if (parts.len == 0) return error.MissingExecutable;
    const result = try allocator.alloc([]const u8, parts.len);
    for (parts, result) |part, *dest| dest.* = part;
    return result;
}

fn compilerCommand(allocator: Allocator, io: std.Io, env: *const std.process.Environ.Map, executable: []const []const u8) ![]const []const u8 {
    if (env.get("WAMRC")) |value| if (value.len != 0) return command(allocator, value);
    const path = try std.fs.path.resolve(allocator, &.{executable[0]});
    const sibling = try std.fs.path.join(allocator, &.{
        std.fs.path.dirname(path) orelse ".",
        if (@import("builtin").os.tag == .windows) "wamrc.exe" else "wamrc",
    });
    const exists = if (std.Io.Dir.cwd().statFile(io, sibling, .{})) |_|
        true
    else |err| switch (err) {
        error.FileNotFound, error.NotDir => false,
        else => return err,
    };
    return allocator.dupe([]const u8, &.{if (exists) sibling else "wamrc"});
}

fn compilerMtime(io: std.Io, args: []const []const u8) !i128 {
    var newest: i128 = 0;
    for (args) |arg| {
        const stat = std.Io.Dir.cwd().statFile(io, arg, .{}) catch |err| switch (err) {
            error.FileNotFound, error.NotDir => continue,
            else => return err,
        };
        newest = @max(newest, stat.mtime.nanoseconds);
    }
    return newest;
}

const Console = struct {
    allocator: Allocator,
    io: std.Io,
    color: bool,

    fn write(self: *Console, comptime format: []const u8, args: anytype) !void {
        const text = try std.fmt.allocPrint(self.allocator, format, args);
        try std.Io.File.stdout().writeStreamingAll(self.io, text);
    }

    fn progress(raw: *anyopaque, result: report.FixtureResult, _: []const []const u8, kinds: []const execute.FailureKind) !void {
        const self: *Console = @ptrCast(@alignCast(raw));
        const color = if (!self.color) "" else switch (result.outcome) {
            .pass => "\x1b[32m",
            .fail, .xpass => "\x1b[31m",
            .skip, .xfail => "\x1b[90m",
        };
        const reset = if (self.color) "\x1b[39m" else "";
        const label = switch (result.outcome) {
            .pass => "passed",
            .fail => "failed",
            .skip => "skipped",
            .xfail => "failed as expected (xfail)",
            .xpass => "passed but was expected to fail (xpass)",
        };
        try self.write("{s}Test {s} {s}{s}\n", .{ color, result.name, label, reset });
        if (result.outcome == .fail) {
            for (result.failures, kinds) |failure, kind| {
                try self.write("{s}  [{s}] {s}{s}\n", .{
                    color, if (kind == .expectation) "Expectation mismatch" else "Unexpected", failure, reset,
                });
            }
        }
    }
};

fn run(init: std.process.Init, options: options_parser.Options) !bool {
    const allocator = init.arena.allocator();
    const io = init.io;
    const env = init.environ_map;
    const jit = if (env.get("WAMR_JIT_TESTSUITE")) |value| value.len != 0 else false;
    var timing_writer: ?profile.JsonlWriter = null;
    if (env.get("WAMR_PROFILE_TIMINGS")) |path| {
        if (path.len != 0) {
            if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos)
                return error.ProfilingAppendUnsupported;
            timing_writer = .{
                .allocator = allocator,
                .io = io,
                .path = path,
                .run_id = env.get("WAMR_PROFILE_RUN_ID") orelse "",
                .mode = env.get("WAMR_PROFILE_MODE") orelse if (jit) "jit" else "aot",
                .pid = @intCast(std.posix.system.getpid()),
            };
        }
    }
    const start = try report.timestamp(allocator, std.Io.Clock.real.now(io).nanoseconds);
    const adapter = try adapterKind(options.adapter);
    var sampler: ?sampling.Sampling = null;
    const selection = env.get("WAMR_PROFILE_SELECTION") orelse "";
    const output_dir = env.get("WAMR_PROFILE_OUTPUT_DIR") orelse "";
    if (adapter == .wamr and selection.len != 0 and output_dir.len != 0) {
        if (@import("builtin").os.tag != .macos) return error.SamplingRequiresMacOS;
        const launcher = env.get("WAMR_PROFILE_LAUNCHER") orelse return error.MissingSampleLauncher;
        if (launcher.len == 0) return error.MissingSampleLauncher;
        sampler = .{
            .selection = selection,
            .output_dir = output_dir,
            .mode = env.get("WAMR_PROFILE_MODE") orelse "aot",
            .launcher = launcher,
        };
    }
    const raw = switch (adapter) {
        .wamr => env.get("WAMR") orelse "wamr",
        .wasmtime => env.get("WASMTIME") orelse "wasmtime",
    };
    const executable = try command(allocator, raw);
    const compiler = if (adapter == .wamr) try compilerCommand(allocator, io, env, executable) else &.{};
    const mtime = if (adapter == .wamr) try compilerMtime(io, compiler) else 0;
    const timeouts = try runtime_env.timeouts(env, adapter);
    const version = try runtime_meta.version(allocator, io, adapter, executable, timeouts.wait_ms);
    const temp_path = try temp_root.select(allocator, io, env);
    var filters: std.ArrayList(expectations.Expectations) = .empty;
    defer for (filters.items) |*filter| filter.deinit();
    for (options.expectations) |path| {
        const source = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        var parsed = try expectations.Expectations.parse(allocator, source);
        filters.append(allocator, parsed) catch |err| {
            parsed.deinit();
            return err;
        };
    }
    const results = try allocator.alloc(report.SuiteResult, options.suites.len);
    var console: Console = .{
        .allocator = allocator,
        .io = io,
        .color = !options.disable_colors and try std.Io.File.stdout().supportsAnsiEscapeCodes(io),
    };
    var failed: usize = 0;
    var passed: usize = 0;
    var skipped: usize = 0;
    for (options.suites, 0..) |path, index| {
        const discovered = try fixture_suite.discover(allocator, io, path);
        if (discovered.cases.len != 0) try console.write("Running test suite {s} with {s} {s}\n", .{
            discovered.manifest.name, runtime_meta.name(adapter), version,
        });
        results[index] = try suite_runner.run(allocator, io, discovered, version, filters.items, .{
            .adapter = adapter,
            .executable = executable,
            .compiler = compiler,
            .compiler_mtime = mtime,
            .compiler_timeout_ms = timeouts.compile_ms,
            .operation_timeout_ms = timeouts.operation_ms,
            .wait_timeout_ms = timeouts.wait_ms,
            .temp_path = temp_path,
            .environment = env,
            .jit = jit,
            .progress = .{ .context = &console, .report = Console.progress },
            .timing = if (timing_writer) |*writer| .{ .context = writer, .emit = profile.JsonlWriter.emit } else null,
            .sampling = sampler,
        });
        try console.write("\n", .{});
        failed += results[index].failed;
        passed += results[index].passed;
        skipped += results[index].skipped;
    }
    try console.write("===== Test results =====\n{s} {s}: ", .{ runtime_meta.name(adapter), version });
    if (failed != 0) {
        try console.write("FAIL: {d}/{d} tests failed", .{ failed, passed + failed });
    } else if (passed != 0) {
        try console.write("PASS: {d} tests passed", .{passed});
    } else {
        try console.write("SKIP: all tests skipped", .{});
    }
    if (skipped != 0) try console.write(" ({d} skipped)", .{skipped});
    try console.write("\n", .{});
    if (options.json_output) |path| {
        const result: report.Report = .{
            .execution = .{
                .start_timestamp = start,
                .finish_timestamp = try report.timestamp(allocator, std.Io.Clock.real.now(io).nanoseconds),
            },
            .results = results,
        };
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = try result.json(allocator) });
    }
    return failed != 0;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = options_parser.parse(init.arena.allocator(), args[1..]) catch |err| {
        if (err == error.HelpRequested) {
            try std.Io.File.stdout().writeStreamingAll(init.io, usage);
            return;
        }
        std.debug.print("error: {s}\n{s}", .{ @errorName(err), usage });
        std.process.exit(2);
    };
    const failed = run(init, options) catch |err| {
        std.debug.print("wasi-suite-runner: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    if (failed) std.process.exit(1);
}

test "runner selects supported runtime adapters and splits runtime commands" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    try std.testing.expectEqual(runtime.Runtime.wamr, try adapterKind("wamr"));
    try std.testing.expectEqual(runtime.Runtime.wamr, try adapterKind("wamr-zig"));
    try std.testing.expectEqual(runtime.Runtime.wasmtime, try adapterKind("wasmtime"));
    try std.testing.expectError(error.UnsupportedRuntimeAdapter, adapterKind("README.md"));
    try std.testing.expectEqualDeep(&[_][]const u8{ "/path with spaces/wasmtime", "-Wflag" }, try command(allocator, "'/path with spaces/wasmtime' -Wflag"));
    try std.testing.expectError(error.MissingExecutable, command(allocator, " "));
}
