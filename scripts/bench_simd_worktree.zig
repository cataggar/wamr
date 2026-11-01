//! Build and measure a SIMD benchmark reference in an isolated worktree.
const std = @import("std");
const shared = @import("bench_loop_worktree.zig");
const optimize = @import("benchmark_optimize.zig");
const options = @import("bench_simd_options.zig");
const parser = @import("bench_simd_parse.zig");
const process = @import("benchmark_process.zig");
const Allocator = std.mem.Allocator;

const overlays = [_][]const u8{
    "build.zig",
    "src/tests/aot_harness.zig",
    "src/tests/simd_bench_runner.zig",
};

pub const Run = struct {
    path: []const u8,
    measurements: []const parser.Measurement,
};

pub const Prepared = struct {
    name: []const u8,
    path: []const u8,
};

/// Use an arena for the returned name and path.
pub fn prepareRef(
    allocator: Allocator,
    io: std.Io,
    repo: []const u8,
    root: []const u8,
    label: []const u8,
    reference: []const u8,
    tools: shared.Tools,
    interrupted: *const std.atomic.Value(bool),
) !Prepared {
    const raw_sha = try process.run(allocator, io, repo, null, &.{ tools.git, "rev-parse", reference }, interrupted);
    const sha = std.mem.trim(u8, raw_sha, " \t\r\n");
    if (sha.len != 40 and sha.len != 64) return error.InvalidRevision;
    for (sha) |ch| if (!std.ascii.isHex(ch)) return error.InvalidRevision;
    const name = try std.fmt.allocPrint(allocator, "{s}-{s}", .{ label, sha[0..12] });
    const path = try std.fs.path.join(allocator, &.{ root, name });
    _ = process.run(allocator, io, repo, null, &.{ tools.git, "worktree", "add", "--detach", path, sha }, interrupted) catch |err| {
        const path_exists = if (std.Io.Dir.cwd().access(io, path, .{})) true else |access_err| switch (access_err) {
            error.FileNotFound => false,
            else => blk: {
                std.debug.print("[harness] warning: cannot inspect {s}: {s}\n", .{ path, @errorName(access_err) });
                break :blk true;
            },
        };
        // Git may register a worktree before creating its destination.
        if (path_exists or err == error.Interrupted) {
            shared.remove(allocator, io, repo, tools, path) catch |cleanup_err|
                std.debug.print("[harness] warning: could not remove {s}: {s}\n", .{ path, @errorName(cleanup_err) });
        }
        return err;
    };
    return .{ .name = name, .path = path };
}

/// Use an arena: returned measurements borrow captured command output.
pub fn measureRef(
    allocator: Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    repo: []const u8,
    prepared: Prepared,
    mode: []const u8,
    opts: options.Options,
    tools: shared.Tools,
    interrupted: *const std.atomic.Value(bool),
) ![]const parser.Measurement {
    const path = prepared.path;
    var env = try base_env.clone(allocator);
    defer env.deinit();
    _ = env.swapRemove("ZIG_LOCAL_CACHE_DIR");
    const cache = try std.fs.path.join(allocator, &.{ path, ".zig-global-cache" });
    try std.Io.Dir.cwd().createDirPath(io, cache);
    try env.put("ZIG_GLOBAL_CACHE_DIR", cache);
    for (overlays) |relative| {
        const source = try std.fs.path.join(allocator, &.{ repo, relative });
        const dest = try std.fs.path.join(allocator, &.{ path, relative });
        try std.Io.Dir.cwd().copyFile(source, .cwd(), dest, io, .{ .make_path = true });
    }

    var command = try optimize.buildCommand(allocator, io, path, mode);
    command[0] = if (std.mem.eql(u8, command[0], "zig016")) tools.zig016 else tools.zig;
    std.debug.print("[harness] building {s} ({s})\n", .{ prepared.name, mode });
    _ = try process.run(allocator, io, path, &env, &command, interrupted);
    const runner = try std.fs.path.join(allocator, &.{ path, "zig-out/bin/simd-bench-runner" });
    try std.Io.Dir.cwd().access(io, runner, .{});

    var measurements: std.ArrayList(parser.Measurement) = .empty;
    for (0..opts.runs) |index| {
        std.debug.print("[harness] running {s} ({d}/{d}, iterations={d})\n", .{ prepared.name, index + 1, opts.runs, opts.iterations });
        var argv: std.ArrayList([]const u8) = .empty;
        try argv.appendSlice(allocator, &.{ runner, "--iterations", try std.fmt.allocPrint(allocator, "{d}", .{opts.iterations}) });
        if (opts.wasmtime) {
            try argv.appendSlice(allocator, &.{ "--wasmtime", "--wasmtime-path", opts.wasmtime_path });
            if (opts.wasmtime_iterations) |count| try argv.appendSlice(allocator, &.{ "--wasmtime-iterations", try std.fmt.allocPrint(allocator, "{d}", .{count}) });
        }
        const output = try process.run(allocator, io, path, &env, argv.items, interrupted);
        const rows = try parser.parse(allocator, output, index + 1);
        try measurements.appendSlice(allocator, rows);
    }
    return measurements.toOwnedSlice(allocator);
}

/// Prepare and measure one reference, removing it if measurement fails.
pub fn runRef(
    allocator: Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    repo: []const u8,
    root: []const u8,
    label: []const u8,
    reference: []const u8,
    mode: []const u8,
    opts: options.Options,
    tools: shared.Tools,
    interrupted: *const std.atomic.Value(bool),
) !Run {
    const prepared = try prepareRef(allocator, io, repo, root, label, reference, tools, interrupted);
    errdefer shared.remove(allocator, io, repo, tools, prepared.path) catch |err|
        std.debug.print("[harness] warning: could not remove {s}: {s}\n", .{ prepared.path, @errorName(err) });
    return .{
        .path = prepared.path,
        .measurements = try measureRef(allocator, io, base_env, repo, prepared, mode, opts, tools, interrupted),
    };
}

test "SIMD reference builds overlay and run with isolated cache" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    try temp.dir.createDirPath(io, "src/tests");
    for (overlays) |path| try temp.dir.writeFile(io, .{ .sub_path = path, .data = "overlay" });
    try temp.dir.writeFile(io, .{ .sub_path = "fake-git", .data =
        \\#!/bin/sh
        \\if [ "$1" = rev-parse ]; then printf '1234567890123456789012345678901234567890\n'; exit; fi
        \\if [ "$2" = add ]; then
        \\  mkdir -p "$4/zig-out/bin"
        \\  printf '.minimum_zig_version = "0.17.0"' > "$4/build.zig.zon"
        \\  exit
        \\fi
        \\if [ "$2" = remove ]; then printf 'remove\n' >> actions; exit; fi
        \\exit 9
        \\
    });
    try temp.dir.writeFile(io, .{ .sub_path = "fake-zig", .data =
        \\#!/bin/sh
        \\[ -z "$ZIG_LOCAL_CACHE_DIR" ] && [ "$ZIG_GLOBAL_CACHE_DIR" = "$PWD/.zig-global-cache" ] || exit 3
        \\[ "$(cat build.zig)" = overlay ] && [ "$(cat src/tests/aot_harness.zig)" = overlay ] || exit 4
        \\printf '#!/bin/sh\nprintf "bench\\tcase\\taot\\tok\\t1\\t2\\t3\\t4\\t5\\n"\n' > zig-out/bin/simd-bench-runner
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
    try env.put("ZIG_LOCAL_CACHE_DIR", "stale");
    const tools: shared.Tools = .{
        .git = try std.fs.path.join(allocator, &.{ root, "fake-git" }),
        .zig = try std.fs.path.join(allocator, &.{ root, "fake-zig" }),
    };
    const interrupted: std.atomic.Value(bool) = .init(false);
    const result = try runRef(allocator, io, &env, root, root, "target", "HEAD", "ReleaseFast", .{ .runs = 2 }, tools, &interrupted);
    try std.testing.expectEqual(@as(usize, 2), result.measurements.len);
    try std.testing.expectEqual(@as(usize, 2), result.measurements[1].run_index);
    try std.testing.expectEqualStrings("case", result.measurements[0].case);
    try shared.remove(allocator, io, root, tools, result.path);
}
