//! Experimental native SIMD benchmark comparison.
const std = @import("std");
const options = @import("bench_simd_options.zig");
const report = @import("bench_simd_report.zig");
const simd_worktree = @import("bench_simd_worktree.zig");
const worktree = @import("bench_loop_worktree.zig");
const temp_root = @import("temp_root.zig");
const Allocator = std.mem.Allocator;
var interrupted: std.atomic.Value(bool) = .init(false);

fn onSignal(_: std.posix.SIG) callconv(.c) void {
    interrupted.store(true, .release);
}

const usage =
    \\usage: bench-simd [--baseline REF] [--target REF] [--runs N]
    \\  [--iterations N] [--wasmtime] [--wasmtime-path PATH]
    \\  [--wasmtime-iterations N] [--repo DIR] [--out FILE]
    \\  [--emit markdown|github] [--optimize ReleaseFast|ReleaseSafe|both]
    \\
;

fn createRoot(allocator: Allocator, io: std.Io, env: *const std.process.Environ.Map) ![]const u8 {
    const parent = try temp_root.select(allocator, io, env);
    for (0..10) |_| {
        var random: [16]u8 = undefined;
        io.random(&random);
        const name = try std.fmt.allocPrint(allocator, "bench-simd-{s}", .{std.fmt.bytesToHex(random, .lower)});
        const path = try std.fs.path.join(allocator, &.{ parent, name });
        std.Io.Dir.cwd().createDir(io, path, .default_dir) catch |err| switch (err) {
            error.PathAlreadyExists => continue,
            else => return err,
        };
        return path;
    }
    return error.TemporaryNameCollisions;
}

fn findRepo(allocator: Allocator, io: std.Io, start: []const u8) !?[]const u8 {
    var directory: ?[]const u8 = start;
    while (directory) |path| {
        const marker = try std.fs.path.join(allocator, &.{ path, "scripts/bench_simd.zig" });
        std.Io.Dir.cwd().access(io, marker, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                directory = std.fs.path.dirname(path);
                continue;
            },
            else => return err,
        };
        return path;
    }
    return null;
}

fn resolveRepo(allocator: Allocator, io: std.Io, requested: ?[]const u8) ![]const u8 {
    if (requested) |path| return std.Io.Dir.cwd().realPathFileAlloc(io, path, allocator);
    const executable = try std.process.executablePathAlloc(io, allocator);
    if (std.fs.path.dirname(executable)) |directory| {
        if (try findRepo(allocator, io, directory)) |repo| return repo;
    }
    const cwd = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
    return (try findRepo(allocator, io, cwd)) orelse error.BenchmarkRepositoryNotFound;
}

const Cleanup = struct {
    allocator: Allocator,
    io: std.Io,
    repo: []const u8,
    root: []const u8,
    tools: worktree.Tools,
    paths: [2]?[]const u8 = @splat(null),

    fn clean(self: *Cleanup) !void {
        var failure: ?anyerror = null;
        for (0..self.paths.len) |index| {
            if (self.paths[self.paths.len - index - 1]) |path| {
                worktree.remove(self.allocator, self.io, self.repo, self.tools, path) catch |err| {
                    std.debug.print("[harness] warning: could not remove {s}: {s}\n", .{ path, @errorName(err) });
                    failure = err;
                };
            }
        }
        if (failure) |err| return err;
        try std.Io.Dir.cwd().deleteDir(self.io, self.root);
    }
};

fn execute(allocator: Allocator, io: std.Io, env: *const std.process.Environ.Map, opts: options.Options, tools: worktree.Tools) !void {
    const repo = try resolveRepo(allocator, io, opts.repo);
    const root = try createRoot(allocator, io, env);
    var cleanup: Cleanup = .{ .allocator = allocator, .io = io, .repo = repo, .root = root, .tools = tools };
    var needs_cleanup = true;
    defer if (needs_cleanup) cleanup.clean() catch |err|
        std.debug.print("[harness] warning: benchmark cleanup failed: {s}\n", .{@errorName(err)});
    const table = if (std.mem.eql(u8, opts.mode, "both")) blk: {
        const fast = try simd_worktree.runRef(allocator, io, env, repo, root, "target-fast", opts.target, "ReleaseFast", opts, tools, &interrupted);
        cleanup.paths[0] = fast.path;
        const safe = try simd_worktree.runRef(allocator, io, env, repo, root, "target-safe", opts.target, "ReleaseSafe", opts, tools, &interrupted);
        cleanup.paths[1] = safe.path;
        break :blk try report.renderOptimizeComparison(allocator, opts.target, fast.measurements, safe.measurements);
    } else blk: {
        const baseline = try simd_worktree.prepareRef(allocator, io, repo, root, "baseline", opts.baseline, tools, &interrupted);
        cleanup.paths[0] = baseline.path;
        const target = try simd_worktree.prepareRef(allocator, io, repo, root, "target", opts.target, tools, &interrupted);
        cleanup.paths[1] = target.path;
        const baseline_rows = try simd_worktree.measureRef(allocator, io, env, repo, baseline, opts.mode, opts, tools, &interrupted);
        const target_rows = try simd_worktree.measureRef(allocator, io, env, repo, target, opts.mode, opts, tools, &interrupted);
        break :blk try report.renderComparison(allocator, opts.baseline, baseline_rows, opts.target, target_rows, opts.mode);
    };
    needs_cleanup = false;
    try cleanup.clean();
    if (interrupted.load(.acquire)) return error.Interrupted;
    const text = try std.fmt.allocPrint(allocator, "{s}\n", .{table});
    try std.Io.File.stdout().writeStreamingAll(io, text);
    if (opts.output) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text });
    if (std.mem.eql(u8, opts.emit, "github")) {
        if (env.get("GITHUB_STEP_SUMMARY")) |path| {
            if (path.len != 0) {
                const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = false });
                defer file.close(io);
                try file.writePositionalAll(io, text, (try file.stat(io)).size);
            }
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const opts = options.parse(args[1..]) catch |err| {
        if (err == error.HelpRequested) {
            try std.Io.File.stdout().writeStreamingAll(init.io, usage);
            return;
        }
        std.debug.print("bench-simd: {s}\n{s}", .{ @errorName(err), usage });
        std.process.exit(2);
    };
    const signals = [_]std.posix.SIG{ .INT, .TERM };
    var old: [2]std.posix.Sigaction = undefined;
    const action: std.posix.Sigaction = .{ .handler = .{ .handler = onSignal }, .mask = std.posix.sigemptyset(), .flags = 0 };
    interrupted.store(false, .release);
    for (signals, 0..) |signal, index| std.posix.sigaction(signal, &action, &old[index]);
    defer for (signals, 0..) |signal, index| std.posix.sigaction(signal, &old[index], null);
    execute(allocator, init.io, init.environ_map, opts, .{}) catch |err| {
        if (err == error.Interrupted) std.process.exit(130);
        std.debug.print("bench-simd: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

test "SIMD CLI orchestration builds both modes, writes report, and cleans worktrees" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    try temp.dir.createDirPath(io, "src/tests");
    for ([_][]const u8{ "build.zig", "src/tests/aot_harness.zig", "src/tests/simd_bench_runner.zig" }) |path|
        try temp.dir.writeFile(io, .{ .sub_path = path, .data = "overlay" });
    try temp.dir.writeFile(io, .{ .sub_path = "fake-git", .data =
        \\#!/bin/sh
        \\if [ "$1" = rev-parse ]; then printf '1234567890123456789012345678901234567890\n'; exit; fi
        \\if [ "$2" = add ]; then
        \\  mkdir -p "$4/zig-out/bin"
        \\  printf '.minimum_zig_version = "0.17.0"' > "$4/build.zig.zon"
        \\  printf 'add\n' >> actions
        \\  printf 'add\n' >> order
        \\  exit
        \\fi
        \\if [ "$2" = remove ]; then
        \\  mv "$4" "$PWD/removed-$(basename "$4")"
        \\  printf 'remove\n' >> actions
        \\  exit
        \\fi
        \\exit 9
        \\
    });
    try temp.dir.writeFile(io, .{ .sub_path = "fake-zig", .data =
        \\#!/bin/sh
        \\[ -z "$ZIG_LOCAL_CACHE_DIR" ] && [ "$ZIG_GLOBAL_CACHE_DIR" = "$PWD/.zig-global-cache" ] || exit 3
        \\printf 'build\n' >> "$SIMD_TEST_REPO/order"
        \\printf '#!/bin/sh\n[ "$1" = "--iterations" ] && [ "$2" = "4" ] && [ "$3" = "--wasmtime" ] && [ "$4" = "--wasmtime-path" ] && [ "$5" = "test-wt" ] && [ "$6" = "--wasmtime-iterations" ] && [ "$7" = "2" ] || exit 8\nprintf "bench\\tcase\\taot\\tok\\t1\\t2\\t3\\t4\\t5\\n"\n' > zig-out/bin/simd-bench-runner
        \\chmod +x zig-out/bin/simd-bench-runner
        \\
    });
    for ([_][]const u8{ "fake-git", "fake-zig" }) |path| {
        const file = try temp.dir.openFile(io, path, .{});
        defer file.close(io);
        try file.setPermissions(io, .executable_file);
    }
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("TMPDIR", root);
    try env.put("ZIG_LOCAL_CACHE_DIR", "stale");
    try env.put("SIMD_TEST_REPO", root);
    const tools: worktree.Tools = .{
        .git = try std.fs.path.join(allocator, &.{ root, "fake-git" }),
        .zig = try std.fs.path.join(allocator, &.{ root, "fake-zig" }),
    };
    const output = try std.fs.path.join(allocator, &.{ root, "report.md" });
    const opts: options.Options = .{ .repo = root, .runs = 1, .iterations = 4, .wasmtime = true, .wasmtime_path = "test-wt", .wasmtime_iterations = 2, .mode = "both", .output = output };
    try execute(allocator, io, &env, opts, tools);
    const text = try temp.dir.readFileAlloc(io, "report.md", allocator, .unlimited);
    try std.testing.expect(std.mem.indexOf(u8, text, "| `case` | `aot` | `HEAD` (target) | ok | ok |") != null);
    const actions = try temp.dir.readFileAlloc(io, "actions", allocator, .unlimited);
    try std.testing.expectEqualStrings("add\nadd\nremove\nremove\n", actions);
    const both_order = try temp.dir.readFileAlloc(io, "order", allocator, .unlimited);
    try std.testing.expectEqualStrings("add\nbuild\nadd\nbuild\n", both_order);
    try temp.dir.writeFile(io, .{ .sub_path = "order", .data = "" });
    try execute(allocator, io, &env, .{ .repo = root, .runs = 1, .iterations = 4, .wasmtime = true, .wasmtime_path = "test-wt", .wasmtime_iterations = 2 }, tools);
    const comparison_order = try temp.dir.readFileAlloc(io, "order", allocator, .unlimited);
    try std.testing.expectEqualStrings("add\nadd\nbuild\nbuild\n", comparison_order);
    try temp.dir.writeFile(io, .{ .sub_path = "fake-zig", .data =
        \\#!/bin/sh
        \\case "$PWD" in */target-*) exit 7;; esac
        \\printf '#!/bin/sh\nprintf "bench\\tcase\\taot\\tok\\t1\\t2\\t3\\t4\\t5\\n"\n' > zig-out/bin/simd-bench-runner
        \\chmod +x zig-out/bin/simd-bench-runner
        \\
    });
    try std.testing.expectError(error.BenchmarkCommandFailed, execute(allocator, io, &env, .{ .repo = root, .runs = 1 }, tools));
    const after_failure = try temp.dir.readFileAlloc(io, "actions", allocator, .unlimited);
    try std.testing.expectEqualStrings("add\nadd\nremove\nremove\nadd\nadd\nremove\nremove\nadd\nadd\nremove\nremove\n", after_failure);
}
