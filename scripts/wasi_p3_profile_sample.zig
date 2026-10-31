//! Execute one experimental native WASI P3 timing sample.
const std = @import("std");
const collect = @import("wasi_p3_profile_collect.zig");
const process_control = @import("process_control.zig");
const Allocator = std.mem.Allocator;

pub const Options = struct {
    repo: []const u8,
    suite: []const u8,
    output_dir: []const u8,
    runner: []const []const u8,
    mode: collect.Mode,
    temperature: collect.Temperature,
    index: usize,
    timeout_ns: ?u64 = null,
    /// Python-style `g` formatting of the timeout in seconds.
    timeout_seconds_text: ?[]const u8 = null,
    grace_ns: u64 = 10 * std.time.ns_per_s,
    interrupted: ?*const std.atomic.Value(bool) = null,
};

pub const Sample = struct {
    id: []const u8,
    temperature: []const u8,
    index: usize,
    valid: bool,
    errors: []const []const u8,
    timed_out: bool,
    returncode: i64,
    suite_duration_ns: u64,
    counts: collect.Counts,
    events: []const std.json.Value,
    raw_timing: []const u8,
    report: []const u8,
    log: []const u8,
};

pub fn removeIfPresent(io: std.Io, path: []const u8) !void {
    std.Io.Dir.cwd().deleteFile(io, path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };
}

fn isSidecar(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".cwasm") or std.mem.endsWith(u8, name, ".cwasm.json");
}

pub fn removeSidecars(io: std.Io, path: []const u8, verify: bool) !void {
    const suite = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer suite.close(io);
    var entries = suite.iterate();
    while (try entries.next(io)) |entry| {
        if (isSidecar(entry.name)) try suite.deleteFile(io, entry.name);
    }
    if (verify) {
        entries = suite.iterate();
        while (try entries.next(io)) |entry| {
            if (isSidecar(entry.name)) return error.JitSidecarsRemain;
        }
    }
}

pub fn signalGroup(pid: std.posix.pid_t, signal: std.posix.SIG) !void {
    std.posix.kill(-pid, signal) catch |err| switch (err) {
        error.ProcessNotFound => {},
        else => return err,
    };
}

pub fn waitUntil(io: std.Io, child: *std.process.Child, duration_ns: u64) !?std.process.Child.Term {
    const started = std.Io.Clock.awake.now(io);
    while (true) {
        if (try process_control.pollChild(child, io)) |term| return term;
        const elapsed = started.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
        if (elapsed >= duration_ns) return null;
        const remaining: u64 = @intCast(@as(i128, duration_ns) - elapsed);
        try io.sleep(.fromNanoseconds(@intCast(@min(remaining, 10 * std.time.ns_per_ms))), .awake);
    }
}

pub fn runCommand(
    allocator: Allocator,
    io: std.Io,
    argv: []const []const u8,
    cwd: []const u8,
    env: *const std.process.Environ.Map,
    log: std.Io.File,
    timeout_ns: ?u64,
    grace_ns: u64,
    interrupted: ?*const std.atomic.Value(bool),
) !struct { term: std.process.Child.Term, timed_out: bool, duration_ns: u64 } {
    if (interrupted) |flag| if (flag.load(.acquire)) return error.Interrupted;
    var fds: [2]std.posix.fd_t = undefined;
    if (std.posix.errno(std.c.pipe(&fds)) != .SUCCESS) return error.ProfilePipeFailed;
    const read_file: std.Io.File = .{ .handle = fds[0], .flags = .{ .nonblocking = false } };
    const write_file: std.Io.File = .{ .handle = fds[1], .flags = .{ .nonblocking = false } };
    defer read_file.close(io);
    var write_open = true;
    defer if (write_open) write_file.close(io);
    for (fds) |fd| {
        if (std.posix.errno(std.posix.system.fcntl(fd, std.posix.F.SETFD, @as(usize, 1))) != .SUCCESS)
            return error.ProfilePipeFailed;
    }

    const started = std.Io.Clock.awake.now(io);
    var child = try std.process.spawn(io, .{
        .argv = argv,
        .cwd = .{ .path = cwd },
        .environ_map = env,
        .stdout = .{ .file = write_file },
        .stderr = .{ .file = write_file },
        .pgid = 0,
    });
    write_file.close(io);
    write_open = false;
    const pid = child.id.?;
    defer if (child.id != null) {
        signalGroup(pid, .KILL) catch |err|
            std.debug.print("profile sample: failed to terminate process group: {s}\n", .{@errorName(err)});
        child.kill(io);
    };
    var storage: std.Io.File.MultiReader.Buffer(1) = undefined;
    var reader: std.Io.File.MultiReader = undefined;
    reader.init(allocator, io, storage.toStreams(), &.{read_file});
    defer reader.deinit();
    var eof = false;
    var timed_out = false;
    var stopping = false;
    var killed = false;
    var stop_started: std.Io.Timestamp = undefined;
    while (!eof) {
        const now = std.Io.Clock.awake.now(io);
        const elapsed: u64 = @intCast(@max(0, started.durationTo(now).nanoseconds));
        const was_interrupted = if (interrupted) |flag| flag.load(.acquire) else false;
        if (!stopping and (was_interrupted or (timeout_ns != null and elapsed >= timeout_ns.?))) {
            timed_out = !was_interrupted;
            try signalGroup(pid, .TERM);
            stopping = true;
            stop_started = now;
        }
        if (stopping and !killed and stop_started.durationTo(now).nanoseconds >= @as(i128, grace_ns)) {
            try signalGroup(pid, .KILL);
            killed = true;
        }
        if (stopping and killed and stop_started.durationTo(now).nanoseconds >= @as(i128, grace_ns) + 10 * std.time.ns_per_s)
            return error.ProfileProcessDidNotTerminate;
        const deadline: std.Io.Timeout = .{ .deadline = .fromNow(io, .{ .clock = .awake, .raw = .fromMilliseconds(20) }) };
        reader.fill(4096, deadline) catch |err| switch (err) {
            error.Timeout => {},
            error.EndOfStream => eof = true,
            else => return err,
        };
        try reader.checkAnyError();
        const buffered = reader.reader(0).buffered();
        if (buffered.len != 0) {
            try log.writeStreamingAll(io, buffered);
            reader.reader(0).toss(buffered.len);
        }
    }
    // Keep the leader unreaped until pipe EOF and final group kill so its PGID cannot be reused.
    if (stopping and !killed) try signalGroup(pid, .KILL);
    const term = (try waitUntil(io, &child, 10 * std.time.ns_per_s)) orelse return error.ProfileProcessDidNotTerminate;
    if (interrupted) |flag| if (flag.load(.acquire)) return error.Interrupted;
    const elapsed = started.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
    return .{ .term = term, .timed_out = timed_out, .duration_ns = @intCast(@max(0, elapsed)) };
}

pub fn exitCode(term: std.process.Child.Term) !i64 {
    return switch (term) {
        .exited => |code| code,
        .signal => |signal| -@as(i64, @intCast(@backingInt(signal))),
        else => error.InvalidProfileProcessStatus,
    };
}

/// Use an arena for the sample and all parsed timing data.
pub fn run(allocator: Allocator, io: std.Io, base_env: *const std.process.Environ.Map, opts: Options) !Sample {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.UnsupportedProfilePlatform;
    if (opts.runner.len == 0 or opts.index == 0) return error.InvalidProfileSample;
    if ((opts.timeout_ns == null) != (opts.timeout_seconds_text == null)) return error.InvalidProfileTimeout;
    const root = std.Io.Dir.cwd();
    try root.createDirPath(io, opts.output_dir);
    const output = try root.realPathFileAlloc(io, opts.output_dir, allocator);
    for ([_][]const u8{ "raw", "reports", "logs" }) |directory|
        try root.createDirPath(io, try std.fs.path.join(allocator, &.{ output, directory }));
    const id = try std.fmt.allocPrint(allocator, "{s}-{s}-{d:0>2}", .{ @tagName(opts.mode), @tagName(opts.temperature), opts.index });
    const timing_rel = try std.fmt.allocPrint(allocator, "raw/{s}.jsonl", .{id});
    const report_rel = try std.fmt.allocPrint(allocator, "reports/{s}.json", .{id});
    const log_rel = try std.fmt.allocPrint(allocator, "logs/{s}.log", .{id});
    const timing = try std.fs.path.join(allocator, &.{ output, timing_rel });
    const report = try std.fs.path.join(allocator, &.{ output, report_rel });
    const log_path = try std.fs.path.join(allocator, &.{ output, log_rel });
    try removeIfPresent(io, timing);
    try removeIfPresent(io, report);
    if (opts.mode == .jit or opts.temperature == .cold) try removeSidecars(io, opts.suite, opts.mode == .jit);
    var env = try base_env.clone(allocator);
    defer env.deinit();
    try env.put("WAMR_PROFILE_TIMINGS", timing);
    try env.put("WAMR_PROFILE_RUN_ID", id);
    try env.put("WAMR_PROFILE_MODE", @tagName(opts.mode));
    try env.put("WAMR_P3_REPORT", report);
    if (opts.mode == .jit) try env.put("WAMR_JIT_TESTSUITE", "1") else _ = env.swapRemove("WAMR_JIT_TESTSUITE");
    const argv = try allocator.alloc([]const u8, opts.runner.len + 2);
    @memcpy(argv[0..opts.runner.len], opts.runner);
    argv[opts.runner.len] = "--json-output-location";
    argv[opts.runner.len + 1] = report;
    const log_file = try root.createFile(io, log_path, .{});
    defer log_file.close(io);
    const outcome = try runCommand(allocator, io, argv, opts.repo, &env, log_file, opts.timeout_ns, opts.grace_ns, opts.interrupted);
    const code = try exitCode(outcome.term);
    const report_bytes = root.readFileAlloc(io, report, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    const counts: collect.Counts = if (report_bytes) |bytes| try collect.reportCounts(try std.json.parseFromSliceLeaky(std.json.Value, allocator, bytes, .{ .duplicate_field_behavior = .use_last })) else .{
        .fixtures = 0,
        .executed = 0,
        .passed = 0,
        .failed = 0,
    };
    const timing_bytes = root.readFileAlloc(io, timing, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    const events = if (timing_bytes) |bytes| try collect.parseEvents(allocator, timing, bytes) else &.{};
    const errors = if (outcome.timed_out)
        try allocator.dupe([]const u8, &.{try std.fmt.allocPrint(allocator, "sample exceeded {s}s and was terminated", .{opts.timeout_seconds_text.?})})
    else
        try collect.sampleErrors(allocator, opts.mode, opts.temperature, code, counts, events);
    return .{
        .id = id,
        .temperature = @tagName(opts.temperature),
        .index = opts.index,
        .valid = errors.len == 0,
        .errors = errors,
        .timed_out = outcome.timed_out,
        .returncode = code,
        .suite_duration_ns = outcome.duration_ns,
        .counts = counts,
        .events = events,
        .raw_timing = timing_rel,
        .report = report_rel,
        .log = log_rel,
    };
}

test "sample execution records validated results and resets cold sidecars" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    const suite = try std.fs.path.join(allocator, &.{ root, "suite" });
    const output = try std.fs.path.join(allocator, &.{ root, "output" });
    const runner = try std.fs.path.join(allocator, &.{ root, "runner" });
    try temp.dir.createDir(io, "suite", .default_dir);
    try temp.dir.writeFile(io, .{ .sub_path = "suite/stale.cwasm", .data = "" });
    try temp.dir.writeFile(io, .{ .sub_path = "suite/stale.cwasm.json", .data = "" });
    try temp.dir.writeFile(io, .{ .sub_path = "suite/keep.wasm", .data = "" });
    try temp.dir.writeFile(io, .{ .sub_path = "runner", .data =
        \\#!/bin/sh
        \\[ "$1" = --json-output-location ] && [ "$2" = "$WAMR_P3_REPORT" ] || exit 8
        \\case "$WAMR_PROFILE_MODE:$WAMR_JIT_TESTSUITE" in aot:) cache=miss;; jit:1) cache=bypass;; *) exit 9;; esac
        \\printf '{"results":[{"tests":[' > "$WAMR_P3_REPORT"
        \\i=0
        \\while [ "$i" -lt 41 ]; do
        \\  [ "$i" -eq 0 ] || printf ',' >> "$WAMR_P3_REPORT"
        \\  printf '{"executed":true,"failures":[]}' >> "$WAMR_P3_REPORT"
        \\  for phase in fixture_execution core_precompile; do
        \\    c="$cache"; [ "$phase" != fixture_execution ] || c=n/a
        \\    printf '{"schema_version":1,"event":"phase_timing","mode":"%s","fixture":"f%s","phase":"%s","artifact_kind":"core","cache":"%s","duration_ns":1}\n' "$WAMR_PROFILE_MODE" "$i" "$phase" "$c" >> "$WAMR_PROFILE_TIMINGS"
        \\  done
        \\  i=$((i + 1))
        \\done
        \\printf ']}]}' >> "$WAMR_P3_REPORT"
        \\printf output
        \\printf error >&2
        \\
    });
    try temp.dir.setFilePermissions(io, "runner", .executable_file, .{});
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("PATH", "/usr/bin:/bin");
    try env.put("WAMR_JIT_TESTSUITE", "stale");
    var opts: Options = .{ .repo = root, .suite = suite, .output_dir = output, .runner = &.{runner}, .mode = .aot, .temperature = .cold, .index = 1 };
    const aot = try run(allocator, io, &env, opts);
    try std.testing.expect(aot.valid and !aot.timed_out);
    try std.testing.expectEqualStrings("aot-cold-01", aot.id);
    try std.testing.expectEqual(@as(usize, 82), aot.events.len);
    try std.testing.expectEqual(@as(usize, 41), aot.counts.passed);
    try std.testing.expectError(error.FileNotFound, temp.dir.statFile(io, "suite/stale.cwasm", .{}));
    try std.testing.expectError(error.FileNotFound, temp.dir.statFile(io, "suite/stale.cwasm.json", .{}));
    try temp.dir.access(io, "suite/keep.wasm", .{});
    const log = try temp.dir.readFileAlloc(io, "output/logs/aot-cold-01.log", allocator, .unlimited);
    try std.testing.expectEqualStrings("outputerror", log);
    try temp.dir.writeFile(io, .{ .sub_path = "suite/stale.cwasm", .data = "" });
    opts.mode = .jit;
    opts.temperature = .warm;
    opts.index = 2;
    const jit = try run(allocator, io, &env, opts);
    try std.testing.expect(jit.valid and !jit.timed_out);
    try std.testing.expectEqualStrings("jit-warm-02", jit.id);
    try std.testing.expectError(error.FileNotFound, temp.dir.statFile(io, "suite/stale.cwasm", .{}));
}

test "sample timeout terminates cooperative and TERM-resistant descendants" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    const output = try std.fs.path.join(allocator, &.{ root, "output" });
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("PROFILE_TEST_ROOT", root);
    const cooperative = try run(allocator, io, &env, .{
        .repo = root,
        .suite = root,
        .output_dir = output,
        .runner = &.{
            "/bin/sh", "-c",
            \\ /bin/sh -c 'trap '\''printf stopped > "$PROFILE_TEST_ROOT/stopped"; exit 0'\'' TERM; printf ready > "$PROFILE_TEST_ROOT/ready"; while :; do /bin/sleep 1; done' &
            \\ while [ ! -f "$PROFILE_TEST_ROOT/ready" ]; do /bin/sleep 0.01; done
            \\ wait
        },
        .mode = .jit,
        .temperature = .cold,
        .index = 1,
        .timeout_ns = 200 * std.time.ns_per_ms,
        .timeout_seconds_text = "0.2",
    });
    try std.testing.expect(cooperative.timed_out and !cooperative.valid);
    try std.testing.expectEqual(-@as(i64, @intCast(@backingInt(std.posix.SIG.TERM))), cooperative.returncode);
    try std.testing.expectEqualStrings("sample exceeded 0.2s and was terminated", cooperative.errors[0]);
    try std.testing.expect(cooperative.suite_duration_ns < 3 * std.time.ns_per_s);
    try temp.dir.access(io, "stopped", .{});

    const sample = try run(allocator, io, &env, .{
        .repo = root,
        .suite = root,
        .output_dir = output,
        .runner = &.{
            "/bin/sh", "-c",
            \\ /bin/sh -c 'trap "" TERM; printf ready > "$PROFILE_TEST_ROOT/resistant-ready"; /bin/sleep 1; printf escaped > "$PROFILE_TEST_ROOT/escaped"' &
            \\ while [ ! -f "$PROFILE_TEST_ROOT/resistant-ready" ]; do /bin/sleep 0.01; done
            \\ /bin/sleep 10
        },
        .mode = .jit,
        .temperature = .cold,
        .index = 2,
        .timeout_ns = 200 * std.time.ns_per_ms,
        .timeout_seconds_text = "0.2",
        .grace_ns = 300 * std.time.ns_per_ms,
    });
    try std.testing.expect(sample.timed_out and !sample.valid);
    try std.testing.expectEqual(-@as(i64, @intCast(@backingInt(std.posix.SIG.TERM))), sample.returncode);
    try std.testing.expectEqualStrings("sample exceeded 0.2s and was terminated", sample.errors[0]);
    try std.testing.expectEqual(@as(usize, 0), sample.counts.fixtures);
    try std.testing.expect(sample.suite_duration_ns < 5 * std.time.ns_per_s);
    try io.sleep(.fromMilliseconds(600), .awake);
    try std.testing.expectError(error.FileNotFound, temp.dir.statFile(io, "escaped", .{}));

    var interrupted: std.atomic.Value(bool) = .init(false);
    const interrupter = try std.Thread.spawn(.{}, struct {
        fn trigger(io_ctx: std.Io, root_dir: []const u8, flag: *std.atomic.Value(bool)) void {
            const path = std.fs.path.join(std.heap.page_allocator, &.{ root_dir, "interrupt-ready" }) catch return;
            defer std.heap.page_allocator.free(path);
            for (0..200) |_| {
                if (std.Io.Dir.cwd().access(io_ctx, path, .{})) |_| break else |_| {}
                io_ctx.sleep(.fromMilliseconds(10), .awake) catch return;
            }
            flag.store(true, .release);
        }
    }.trigger, .{ io, root, &interrupted });
    defer interrupter.join();
    try std.testing.expectError(error.Interrupted, run(allocator, io, &env, .{
        .repo = root,
        .suite = root,
        .output_dir = output,
        .runner = &.{
            "/bin/sh", "-c",
            \\ /bin/sh -c 'trap '\''printf interrupted > "$PROFILE_TEST_ROOT/interrupted"; exit 0'\'' TERM; printf ready > "$PROFILE_TEST_ROOT/interrupt-ready"; while :; do /bin/sleep 1; done' &
            \\ wait
        },
        .mode = .jit,
        .temperature = .cold,
        .index = 3,
        .interrupted = &interrupted,
    }));
    try temp.dir.access(io, "interrupted", .{});
}

test "failed samples cannot reuse stale reports or timing events" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    const output = try std.fs.path.join(allocator, &.{ root, "output" });
    try temp.dir.createDirPath(io, "output/reports");
    try temp.dir.createDirPath(io, "output/raw");
    try temp.dir.writeFile(io, .{ .sub_path = "output/reports/jit-warm-01.json", .data = "stale" });
    try temp.dir.writeFile(io, .{ .sub_path = "output/raw/jit-warm-01.jsonl", .data = "stale" });
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    const opts: Options = .{
        .repo = root,
        .suite = root,
        .output_dir = output,
        .runner = &.{ "/bin/sh", "-c", "exit 7" },
        .mode = .jit,
        .temperature = .warm,
        .index = 1,
    };
    const sample = try run(allocator, io, &env, opts);
    try std.testing.expect(!sample.valid and !sample.timed_out);
    try std.testing.expectEqual(@as(i64, 7), sample.returncode);
    try std.testing.expectEqual(@as(usize, 5), sample.errors.len);
    try std.testing.expectEqualStrings("runner exit 7", sample.errors[0]);
    try std.testing.expectEqual(@as(usize, 0), sample.events.len);
    try std.testing.expectEqual(@as(usize, 0), sample.counts.fixtures);
    try std.testing.expectError(error.FileNotFound, temp.dir.statFile(io, "output/reports/jit-warm-01.json", .{}));
    try std.testing.expectError(error.FileNotFound, temp.dir.statFile(io, "output/raw/jit-warm-01.jsonl", .{}));
}

fn realCorpusTest(mode: collect.Mode) !void {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const repo = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    const source = try std.fs.path.join(allocator, &.{ repo, "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip3" });
    const suite = try std.fs.path.join(allocator, &.{ root, "wasm32-wasip3" });
    const copy = try std.process.run(allocator, io, .{ .argv = &.{ "cp", "-al", source, suite } });
    if (!copy.term.success()) {
        std.debug.print("fixture copy failed: {s}\n", .{copy.stderr});
        return error.ProfileFixtureCopyFailed;
    }
    const runner = try std.fs.path.join(allocator, &.{ repo, "zig-out/bin/wasi-suite-runner" });
    const bin = if (mode == .jit) ".zig-cache/tmp/wamr-jit/bin" else "zig-out/bin";
    const wamr = try std.fs.path.join(allocator, &.{ repo, bin, "wamr" });
    const wamrc = try std.fs.path.join(allocator, &.{ repo, bin, "wamrc" });
    const output = try std.fs.path.join(allocator, &.{ root, "output" });
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("PATH", "/usr/bin:/bin");
    try env.put("HOME", root);
    try env.put("XDG_CACHE_HOME", root);
    try env.put("TMPDIR", root);
    try env.put("WAMR", wamr);
    try env.put("WAMRC", wamrc);
    for (if (mode == .aot) &[_]collect.Temperature{ .cold, .warm } else &[_]collect.Temperature{.cold}) |temperature| {
        const sample = try run(allocator, io, &env, .{
            .repo = repo,
            .suite = suite,
            .output_dir = output,
            .runner = &.{ runner, "-t", suite, "-r", "wamr-zig", "--disable-colors" },
            .mode = mode,
            .temperature = temperature,
            .index = 1,
            .timeout_ns = 600 * std.time.ns_per_s,
            .timeout_seconds_text = "600",
        });
        if (!sample.valid) {
            for (sample.errors) |message| std.debug.print("profile sample: {s}\n", .{message});
            const log = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(allocator, &.{ output, sample.log }), allocator, .limited(32 * 1024));
            std.debug.print("profile sample log:\n{s}\n", .{log});
            return error.RealProfileSampleFailed;
        }
        try std.testing.expectEqual(@as(usize, 41), sample.counts.passed);
        try std.testing.expectEqual(@as(usize, 82), sample.events.len);
    }
}

test "real P3 JIT sample executes every fixture" {
    if (std.c.getenv("WAMR_PROFILE_REAL_TEST") == null) return error.SkipZigTest;
    try realCorpusTest(.jit);
}

test "real P3 AOT cold and warm samples execute every fixture" {
    if (std.c.getenv("WAMR_PROFILE_REAL_AOT_TEST") == null) return error.SkipZigTest;
    try realCorpusTest(.aot);
}
