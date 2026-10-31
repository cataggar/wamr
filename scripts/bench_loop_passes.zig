//! Compare focused IV-simplification and loop-unroll AOT microbenchmarks.
const std = @import("std");
const core = @import("bench_loop_passes_core.zig");
const worktree = @import("bench_loop_worktree.zig");
const cli = @import("bench_loop_options.zig");
const fixed = @import("decimal_format.zig").fixed;
const Allocator = std.mem.Allocator;

const usage =
    \\usage: bench-loop-passes [--baseline REF] [--target REF] [--repo DIR]
    \\  [--profile authoritative|ci] [--warmups N] [--runs N] [--optimize MODE]
    \\  [--out FILE] [--emit markdown|github] [--min-speedup CASE=PCT]
    \\
;

fn probe(allocator: Allocator, io: std.Io, args: []const []const u8) !?[]const u8 {
    const result = std.process.run(allocator, io, .{ .argv = args, .timeout = .{ .duration = .{ .raw = .fromSeconds(5), .clock = .awake } } }) catch |err| switch (err) {
        error.FileNotFound, error.Timeout => {
            std.debug.print("[harness] host probe {s} failed: {s}\n", .{ args[0], @errorName(err) });
            return null;
        },
        else => return err,
    };
    defer allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("[harness] host probe {s} failed: {f}\n", .{ args[0], result.term });
        allocator.free(result.stdout);
        return null;
    }
    return std.mem.trim(u8, result.stdout, " \t\r\n");
}

const Host = struct { architecture: []const u8, cpus: []const u8, model: []const u8 };

fn hostInfo(allocator: Allocator, io: std.Io) !Host {
    const architecture = (try probe(allocator, io, &.{ "uname", "-m" })) orelse "unknown-arch";
    const detected_cpus = (try probe(allocator, io, &.{ "getconf", "_NPROCESSORS_ONLN" })) orelse "";
    const cpus = if (detected_cpus.len == 0) "?" else detected_cpus;
    const output = try probe(allocator, io, &.{"lscpu"});
    var model: ?[]const u8 = null;
    if (output) |text| {
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |line| {
            const label = "model name:";
            if (line.len >= label.len and std.ascii.eqlIgnoreCase(line[0..label.len], label)) {
                model = std.mem.trim(u8, line[label.len..], " \t\r\n");
                break;
            }
        }
    }
    if (model == null) {
        const processor = try probe(allocator, io, &.{ "uname", "-p" });
        if (processor) |value| {
            if (value.len != 0 and !std.mem.eql(u8, value, "unknown")) model = value;
        }
    }
    return .{ .architecture = architecture, .cpus = cpus, .model = model orelse "unknown CPU" };
}

fn createWorkRoot(allocator: Allocator, io: std.Io, repo: []const u8) ![]const u8 {
    const parent = try std.fs.path.join(allocator, &.{ repo, ".bench-loop-passes" });
    try std.Io.Dir.cwd().createDirPath(io, parent);
    for (0..10) |_| {
        var random: [16]u8 = undefined;
        io.random(&random);
        const name = try std.fmt.allocPrint(allocator, "run-{s}", .{std.fmt.bytesToHex(random, .lower)});
        const path = try std.fs.path.join(allocator, &.{ parent, name });
        std.Io.Dir.cwd().createDir(io, path, .default_dir) catch |err| switch (err) {
            error.PathAlreadyExists => continue,
            else => return err,
        };
        return path;
    }
    return error.TemporaryNameCollisions;
}

const Cleanup = struct {
    allocator: Allocator,
    io: std.Io,
    repo: []const u8,
    root: []const u8,
    tools: worktree.Tools,
    baseline: ?worktree.Build = null,
    target: ?worktree.Build = null,

    fn clean(self: *Cleanup) !void {
        var failure: ?anyerror = null;
        for ([_]?worktree.Build{ self.target, self.baseline }) |maybe_build| {
            if (maybe_build) |built| {
                worktree.remove(self.allocator, self.io, self.repo, self.tools, built.path) catch |err| {
                    std.debug.print("[harness] warning: could not remove {s}: {s}\n", .{ built.path, @errorName(err) });
                    failure = err;
                };
            }
        }
        if (failure) |err| return err;
        try std.Io.Dir.cwd().deleteDir(self.io, self.root);
        const parent = std.fs.path.dirname(self.root) orelse return error.InvalidBenchmarkDirectory;
        std.Io.Dir.cwd().deleteDir(self.io, parent) catch |err| switch (err) {
            error.DirNotEmpty, error.FileNotFound => {},
            else => return err,
        };
    }
};

fn percentMagnitude(allocator: Allocator, value: f64) ![]const u8 {
    if (std.math.isNan(value)) return "nan";
    if (std.math.isInf(value)) return "inf";
    return fixed(allocator, @abs(value), 2);
}

fn execute(allocator: Allocator, io: std.Io, env: *const std.process.Environ.Map, opts: cli.Options, tools: worktree.Tools, host: Host) !u8 {
    const repo = try std.Io.Dir.cwd().realPathFileAlloc(io, opts.repo orelse ".", allocator);
    try core.validateFixtures(allocator, io, repo);
    const counts = try core.counts(opts.profile, opts.warmups, opts.runs);
    const warmups = std.math.cast(usize, counts.warmups) orelse return error.TooManyMeasurements;
    const runs = std.math.cast(usize, counts.runs) orelse return error.TooManyMeasurements;
    const root = try createWorkRoot(allocator, io, repo);
    var cleanup: Cleanup = .{ .allocator = allocator, .io = io, .repo = repo, .root = root, .tools = tools };
    var needs_cleanup = true;
    defer if (needs_cleanup) cleanup.clean() catch |err|
        std.debug.print("[harness] warning: benchmark cleanup failed: {s}\n", .{@errorName(err)});
    cleanup.baseline = try worktree.build(allocator, io, env, repo, root, "baseline", opts.baseline, opts.optimize, tools);
    cleanup.target = try worktree.build(allocator, io, env, repo, root, "target", opts.target, opts.optimize, tools);
    const baseline = cleanup.baseline.?;
    const target = cleanup.target.?;
    var measurements: [core.fixtures.len][2][]const f64 = undefined;
    for (core.fixtures, 0..) |fixture, index| {
        const result = try core.measureFixture(allocator, io, .{ .cwd = baseline.path, .wamr = baseline.wamr, .artifact = baseline.artifacts[index] }, .{ .cwd = target.path, .wamr = target.wamr, .artifact = target.artifacts[index] }, fixture.name, warmups, runs);
        measurements[index] = .{ result.baseline, result.target };
    }
    const report = try core.renderReport(allocator, .{
        .baseline = .{ .ref = baseline.reference, .sha = baseline.sha, .sizes = baseline.sizes },
        .target = .{ .ref = target.reference, .sha = target.sha, .sizes = target.sizes },
        .measurements = measurements,
        .profile = opts.profile,
        .sample_counts = counts,
        .optimize = opts.optimize,
        .architecture = host.architecture,
        .cpu_count = host.cpus,
        .cpu_model = host.model,
    });
    const line = try std.fmt.allocPrint(allocator, "{s}\n", .{report});
    if (!@import("builtin").is_test) {
        try std.Io.File.stdout().writeStreamingAll(io, line);
    }
    if (opts.output) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = line });
    if (std.mem.eql(u8, opts.emit, "github")) {
        if (env.get("GITHUB_STEP_SUMMARY")) |path| {
            if (path.len != 0) {
                const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = false });
                defer file.close(io);
                try file.writePositionalAll(io, line, (try file.stat(io)).size);
            }
        }
    }
    var status: u8 = 0;
    var failures: std.Io.Writer.Allocating = .init(allocator);
    for (opts.minimum_order[0..opts.minimum_count]) |index| {
        const minimum = opts.minimum[index].?;
        const actual = try core.speedupPct(allocator, measurements[index][0], measurements[index][1]);
        if (actual < minimum) {
            const actual_text = try percentMagnitude(allocator, actual);
            const required_text = try percentMagnitude(allocator, minimum);
            if (status != 0) try failures.writer.writeAll("; ");
            try failures.writer.print("{s}: {s}{s}% < required {s}{s}%", .{
                core.fixtures[index].name,
                if (@as(u64, @bitCast(actual)) >> 63 != 0) "-" else "+",
                actual_text,
                if (@as(u64, @bitCast(minimum)) >> 63 != 0) "-" else "+",
                required_text,
            });
            status = 1;
        }
    }
    if (status != 0) std.debug.print("[harness] speedup threshold failed: {s}\n", .{failures.written()});
    needs_cleanup = false;
    try cleanup.clean();
    return status;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const opts = cli.parse(args[1..]) catch |err| {
        if (err == error.HelpRequested) {
            try std.Io.File.stdout().writeStreamingAll(init.io, usage);
            return;
        }
        std.debug.print("error: {s}\n{s}", .{ @errorName(err), usage });
        std.process.exit(2);
    };
    const host = try hostInfo(allocator, init.io);
    const status = execute(allocator, init.io, init.environ_map, opts, .{}, host) catch |err| {
        std.debug.print("[harness] error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
    if (status != 0) std.process.exit(status);
}

test "benchmark CLI builds, measures, writes reports, and enforces thresholds" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    try temp.dir.createDirPath(io, "tests/benchmarks/loop-passes");
    for (core.fixtures) |fixture| {
        const content = try std.Io.Dir.cwd().readFileAlloc(io, fixture.path, allocator, .unlimited);
        try temp.dir.writeFile(io, .{ .sub_path = fixture.path, .data = content });
    }
    try temp.dir.writeFile(io, .{ .sub_path = "fake-git", .data =
        \\#!/bin/sh
        \\if [ "$1" = rev-parse ]; then
        \\  if [ "$2" = old ]; then printf '1111111111111111111111111111111111111111\n'; else printf '2222222222222222222222222222222222222222\n'; fi
        \\  exit
        \\fi
        \\if [ "$2" = add ]; then
        \\  mkdir -p "$4/zig-out/bin"
        \\  printf '.minimum_zig_version = "0.17.0"' > "$4/build.zig.zon"
        \\  cp fake-wamrc fake-wamr "$4/zig-out/bin/"
        \\  mv "$4/zig-out/bin/fake-wamrc" "$4/zig-out/bin/wamrc"
        \\  mv "$4/zig-out/bin/fake-wamr" "$4/zig-out/bin/wamr"
        \\  printf 'add\n' >> actions
        \\  exit
        \\fi
        \\if [ "$2" = remove ]; then
        \\  if [ -f fail-remove ]; then exit 13; fi
        \\  mv "$4" "$PWD/removed-$(basename "$(dirname "$4")")-$(basename "$4")"
        \\  printf 'remove\n' >> actions
        \\  exit
        \\fi
        \\exit 9
        \\
    });
    try temp.dir.writeFile(io, .{ .sub_path = "fake-zig", .data =
        \\#!/bin/sh
        \\[ -z "$ZIG_LOCAL_CACHE_DIR" ] && [ "$ZIG_GLOBAL_CACHE_DIR" = "$PWD/.zig-global-cache" ]
        \\
    });
    try temp.dir.writeFile(io, .{ .sub_path = "fake-wamrc", .data =
        \\#!/bin/sh
        \\printf '\000aot\001\000\000\000\002\000\000\000\001\000\000\000x' > "$4"
        \\
    });
    try temp.dir.writeFile(io, .{ .sub_path = "fake-wamr", .data = "#!/bin/sh\nexit 0\n" });
    for ([_][]const u8{ "fake-git", "fake-zig", "fake-wamrc", "fake-wamr" }) |name| {
        const file = try temp.dir.openFile(io, name, .{});
        defer file.close(io);
        try file.setPermissions(io, .executable_file);
    }
    try temp.dir.writeFile(io, .{ .sub_path = "summary.md", .data = "existing\n" });
    const git = try std.fs.path.join(allocator, &.{ root, "fake-git" });
    const zig = try std.fs.path.join(allocator, &.{ root, "fake-zig" });
    const output = try std.fs.path.join(allocator, &.{ root, "report.md" });
    const summary = try std.fs.path.join(allocator, &.{ root, "summary.md" });
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("PATH", "/usr/bin:/bin");
    try env.put("ZIG_LOCAL_CACHE_DIR", "stale");
    try env.put("GITHUB_STEP_SUMMARY", summary);
    var opts: cli.Options = .{ .repo = root, .baseline = "old", .target = "new", .profile = "ci", .runs = 1, .output = output, .emit = "github" };
    const tools: worktree.Tools = .{ .git = git, .zig = zig, .zig016 = zig };
    const host: Host = .{ .architecture = "arm64", .cpus = "4", .model = "test CPU" };
    try std.testing.expectEqual(@as(u8, 0), try execute(allocator, io, &env, opts, tools, host));
    const report = try temp.dir.readFileAlloc(io, "report.md", allocator, .unlimited);
    const summary_text = try temp.dir.readFileAlloc(io, "summary.md", allocator, .unlimited);
    try std.testing.expect(std.mem.indexOf(u8, report, "| `iv_store` | `old (baseline, 111111111111)`") != null);
    try std.testing.expectEqualStrings(try std.fmt.allocPrint(allocator, "existing\n{s}", .{report}), summary_text);
    opts.minimum[0] = std.math.inf(f64);
    opts.minimum_order[0] = 0;
    opts.minimum_count = 1;
    try std.testing.expectEqual(@as(u8, 1), try execute(allocator, io, &env, opts, tools, host));
    const actions = try temp.dir.readFileAlloc(io, "actions", allocator, .unlimited);
    try std.testing.expectEqualStrings("add\nadd\nremove\nremove\nadd\nadd\nremove\nremove\n", actions);
    try temp.dir.writeFile(io, .{ .sub_path = "fail-remove", .data = "" });
    try temp.dir.writeFile(io, .{ .sub_path = "fake-zig", .data = "#!/bin/sh\nexit 7\n" });
    try std.testing.expectError(error.BenchmarkCommandFailed, execute(allocator, io, &env, opts, tools, host));
    try temp.dir.access(io, ".bench-loop-passes", .{});
}
