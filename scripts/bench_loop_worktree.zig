//! Build benchmark references in independently configured Git worktrees.
const std = @import("std");
const core = @import("bench_loop_passes_core.zig");
const optimize = @import("benchmark_optimize.zig");
const Allocator = std.mem.Allocator;

pub const Tools = struct {
    git: []const u8 = "git",
    zig: []const u8 = "zig",
    zig016: []const u8 = "zig016",
};

pub const Build = struct {
    label: []const u8,
    reference: []const u8,
    sha: []const u8,
    path: []const u8,
    wamr: []const u8,
    artifacts: [core.fixtures.len][]const u8,
    sizes: [core.fixtures.len]core.AotSize,

    pub fn deinit(self: Build, allocator: Allocator) void {
        allocator.free(self.sha);
        allocator.free(self.path);
        allocator.free(self.wamr);
        for (self.artifacts) |artifact| allocator.free(artifact);
    }
};

pub fn run(allocator: Allocator, io: std.Io, cwd: []const u8, env: ?*const std.process.Environ.Map, args: []const []const u8) ![]u8 {
    const result = try std.process.run(allocator, io, .{ .argv = args, .cwd = .{ .path = cwd }, .environ_map = env });
    defer allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("[harness] command failed ({f}):", .{result.term});
        for (args) |argument| std.debug.print(" {s}", .{argument});
        std.debug.print("\n[harness] cwd: {s}\n[harness] --- stdout ---\n{s}\n[harness] --- stderr ---\n{s}\n", .{ cwd, result.stdout, result.stderr });
        allocator.free(result.stdout);
        return error.BenchmarkCommandFailed;
    }
    defer allocator.free(result.stdout);
    return std.mem.concat(allocator, u8, &.{ result.stdout, result.stderr });
}

pub fn remove(allocator: Allocator, io: std.Io, repo: []const u8, tools: Tools, build_path: []const u8) !void {
    const result = try run(allocator, io, repo, null, &.{ tools.git, "worktree", "remove", "--force", build_path });
    allocator.free(result);
}

pub fn build(
    allocator: Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    repo: []const u8,
    work_root: []const u8,
    label: []const u8,
    reference: []const u8,
    mode: []const u8,
    tools: Tools,
) !Build {
    const sha_output = try run(allocator, io, repo, null, &.{ tools.git, "rev-parse", reference });
    defer allocator.free(sha_output);
    const sha = try allocator.dupe(u8, std.mem.trim(u8, sha_output, " \t\r\n"));
    errdefer allocator.free(sha);
    if (sha.len != 40 and sha.len != 64) return error.InvalidRevision;
    for (sha) |byte| if (!std.ascii.isHex(byte)) return error.InvalidRevision;
    const directory = try std.fmt.allocPrint(allocator, "{s}-{s}", .{ label, sha[0..@min(12, sha.len)] });
    defer allocator.free(directory);
    const path = try std.fs.path.join(allocator, &.{ work_root, directory });
    errdefer allocator.free(path);
    const added = try run(allocator, io, repo, null, &.{ tools.git, "worktree", "add", "--detach", path, sha });
    allocator.free(added);
    errdefer remove(allocator, io, repo, tools, path) catch |err|
        std.debug.print("[harness] warning: could not remove {s}: {s}\n", .{ path, @errorName(err) });

    var env = try base_env.clone(allocator);
    defer env.deinit();
    _ = env.swapRemove("ZIG_LOCAL_CACHE_DIR");
    const cache = try std.fs.path.join(allocator, &.{ path, ".zig-global-cache" });
    defer allocator.free(cache);
    try std.Io.Dir.cwd().createDirPath(io, cache);
    try env.put("ZIG_GLOBAL_CACHE_DIR", cache);
    var command = try optimize.buildCommand(allocator, io, path, mode);
    defer allocator.free(command[2]);
    command[0] = if (std.mem.eql(u8, command[0], "zig016")) tools.zig016 else tools.zig;
    std.debug.print("[harness] building {s} {s} ({s}, {s})\n", .{ label, reference, sha[0..@min(12, sha.len)], mode });
    const build_output = try run(allocator, io, path, &env, &command);
    allocator.free(build_output);

    const wamr = try std.fs.path.join(allocator, &.{ path, "zig-out/bin/wamr" });
    errdefer allocator.free(wamr);
    const wamrc = try std.fs.path.join(allocator, &.{ path, "zig-out/bin/wamrc" });
    defer allocator.free(wamrc);
    var artifacts: [core.fixtures.len][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer for (artifacts[0..initialized]) |artifact| allocator.free(artifact);
    var sizes: [core.fixtures.len]core.AotSize = undefined;
    for (core.fixtures, 0..) |fixture, index| {
        const fixture_path = try std.fs.path.join(allocator, &.{ repo, fixture.path });
        defer allocator.free(fixture_path);
        const artifact_name = try std.fmt.allocPrint(allocator, ".bench-loop-{s}.cwasm", .{fixture.name});
        defer allocator.free(artifact_name);
        const artifact = try std.fs.path.join(allocator, &.{ path, artifact_name });
        errdefer allocator.free(artifact);
        const output = try run(allocator, io, path, &env, &.{ wamrc, "compile", fixture_path, "-o", artifact });
        allocator.free(output);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, artifact, allocator, .unlimited);
        defer allocator.free(bytes);
        sizes[index] = try core.parseAotSize(bytes);
        artifacts[index] = artifact;
        initialized += 1;
    }
    return .{ .label = label, .reference = reference, .sha = sha, .path = path, .wamr = wamr, .artifacts = artifacts, .sizes = sizes };
}

test "builds each reference in an isolated worktree and cleans up failures" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const fake_git = try std.fs.path.join(allocator, &.{ root, "fake-git" });
    defer allocator.free(fake_git);
    const fake_zig = try std.fs.path.join(allocator, &.{ root, "fake-zig" });
    defer allocator.free(fake_zig);
    const fake_zig016 = try std.fs.path.join(allocator, &.{ root, "fake-zig016" });
    defer allocator.free(fake_zig016);
    try temp.dir.writeFile(io, .{ .sub_path = "fake-git", .data =
        \\#!/bin/sh
        \\if [ "$1" = rev-parse ]; then
        \\  if [ "$2" = invalid ]; then printf '../escape\n'; else printf '1234567890123456789012345678901234567890\n'; fi
        \\  exit
        \\fi
        \\if [ "$2" = add ]; then
        \\  mkdir -p "$4/zig-out/bin"
        \\  if [ -f generation016 ]; then
        \\    printf '.minimum_zig_version = "0.16.0"' > "$4/build.zig.zon"
        \\  else
        \\    printf '.minimum_zig_version = "0.17.0"' > "$4/build.zig.zon"
        \\  fi
        \\  cp fake-wamrc "$4/zig-out/bin/wamrc"
        \\  printf 'add\n' >> actions
        \\  exit
        \\fi
        \\if [ "$2" = remove ]; then printf 'remove\n' >> actions; exit; fi
        \\exit 9
        \\
    });
    try temp.dir.writeFile(io, .{ .sub_path = "fake-zig", .data =
        \\#!/bin/sh
        \\[ -z "$ZIG_LOCAL_CACHE_DIR" ] && [ "$ZIG_GLOBAL_CACHE_DIR" = "$PWD/.zig-global-cache" ]
        \\
    });
    try temp.dir.writeFile(io, .{ .sub_path = "fake-zig016", .data =
        \\#!/bin/sh
        \\[ "$2" = "-Doptimize=ReleaseFast" ] && [ -z "$ZIG_LOCAL_CACHE_DIR" ] && [ "$ZIG_GLOBAL_CACHE_DIR" = "$PWD/.zig-global-cache" ]
        \\
    });
    try temp.dir.writeFile(io, .{ .sub_path = "fake-wamrc", .data =
        \\#!/bin/sh
        \\printf '\000aot\001\000\000\000\002\000\000\000\001\000\000\000x' > "$4"
        \\
    });
    for ([_][]const u8{ "fake-git", "fake-zig", "fake-zig016", "fake-wamrc" }) |name| {
        const file = try temp.dir.openFile(io, name, .{});
        defer file.close(io);
        try file.setPermissions(io, .executable_file);
    }
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("ZIG_LOCAL_CACHE_DIR", "stale");
    const tools: Tools = .{ .git = fake_git, .zig = fake_zig, .zig016 = fake_zig016 };
    try std.testing.expectError(error.InvalidRevision, build(allocator, io, &env, root, root, "invalid", "invalid", "ReleaseFast", tools));
    const built = try build(allocator, io, &env, root, root, "baseline", "main", "ReleaseFast", tools);
    defer built.deinit(allocator);
    try std.testing.expectEqualStrings("1234567890123456789012345678901234567890", built.sha);
    try std.testing.expect(std.mem.endsWith(u8, built.path, "baseline-123456789012"));
    try std.testing.expectEqual(@as(u32, 1), built.sizes[0].text_bytes);
    try std.testing.expectEqual(@as(u32, 1), built.sizes[1].text_bytes);
    try remove(allocator, io, root, tools, built.path);
    try temp.dir.writeFile(io, .{ .sub_path = "generation016", .data = "" });
    const old = try build(allocator, io, &env, root, root, "old", "old-ref", "ReleaseFast", tools);
    defer old.deinit(allocator);
    try remove(allocator, io, root, tools, old.path);
    try temp.dir.deleteFile(io, "generation016");
    try temp.dir.writeFile(io, .{ .sub_path = "fake-zig", .data = "#!/bin/sh\nexit 7\n" });
    try std.testing.expectError(error.BenchmarkCommandFailed, build(allocator, io, &env, root, root, "failed", "main", "ReleaseFast", tools));
    const actions = try temp.dir.readFileAlloc(io, "actions", allocator, .unlimited);
    defer allocator.free(actions);
    try std.testing.expectEqualStrings("add\nremove\nadd\nremove\nadd\nremove\n", actions);
}
