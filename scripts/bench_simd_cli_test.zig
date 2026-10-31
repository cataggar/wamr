//! Signal and worktree-cleanup coverage for the native SIMD benchmark CLI.
const std = @import("std");
const process_control = @import("process_control.zig");

test "CLI terminates interrupted commands and removes their worktrees" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const executable = if (std.c.getenv("SIMD_BENCH")) |path| std.mem.span(path) else return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    for ([_]std.posix.SIG{ .INT, .TERM }) |signal| for ([_][]const u8{ "add-metadata", "add", "build", "build-ignore-term", "orphan-ignore-term" }) |stage| {
        if (signal == .INT and std.mem.eql(u8, stage, "orphan-ignore-term")) continue;
        var temp = std.testing.tmpDir(.{});
        defer temp.cleanup();
        const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
        defer allocator.free(root);
        try temp.dir.createDirPath(io, "src/tests");
        for ([_][]const u8{ "build.zig", "src/tests/aot_harness.zig", "src/tests/simd_bench_runner.zig" }) |path|
            try temp.dir.writeFile(io, .{ .sub_path = path, .data = "overlay" });
        try temp.dir.writeFile(io, .{ .sub_path = "git", .data =
            \\#!/bin/sh
            \\if [ "$1" = rev-parse ]; then printf '1234567890123456789012345678901234567890\n'; exit; fi
            \\if [ "$2" = add ]; then
            \\  if [ "$SIMD_TEST_STAGE" != add-metadata ]; then
            \\    mkdir -p "$4/zig-out/bin"
            \\    printf '.minimum_zig_version = "0.17.0"' > "$4/build.zig.zon"
            \\  fi
            \\  printf 'add\n' >> actions
            \\  case "$SIMD_TEST_STAGE" in add*)
            \\    printf '%s' "$$" > "$SIMD_TEST_ROOT/marker.tmp"
            \\    mv "$SIMD_TEST_ROOT/marker.tmp" "$SIMD_TEST_ROOT/marker"
            \\    exec /bin/sleep 30
            \\  esac
            \\  exit
            \\fi
            \\if [ "$2" = remove ]; then
            \\  if [ -e "$4" ]; then mv "$4" "$PWD/removed-$(basename "$4")"; fi
            \\  printf 'remove\n' >> actions
            \\  exit
            \\fi
            \\exit 9
            \\
        });
        try temp.dir.writeFile(io, .{ .sub_path = "zig", .data =
            \\#!/bin/sh
            \\if [ "$SIMD_TEST_STAGE" = build-ignore-term ]; then
            \\  trap '' TERM
            \\  printf '%s' "$$" > "$SIMD_TEST_ROOT/marker.tmp"
            \\  mv "$SIMD_TEST_ROOT/marker.tmp" "$SIMD_TEST_ROOT/marker"
            \\  exec /bin/sleep 30
            \\fi
            \\if [ "$SIMD_TEST_STAGE" = orphan-ignore-term ]; then
            \\  /bin/sh -c 'trap "" TERM; printf ready > "$SIMD_TEST_ROOT/descendant-ready"; exec /bin/sleep 30' &
            \\  while [ ! -f "$SIMD_TEST_ROOT/descendant-ready" ]; do /bin/sleep 0.01; done
            \\  printf '%s' "$$" > "$SIMD_TEST_ROOT/marker.tmp"
            \\  mv "$SIMD_TEST_ROOT/marker.tmp" "$SIMD_TEST_ROOT/marker"
            \\  exit 0
            \\fi
            \\/bin/sh -c 'trap '\''printf stopped > "$SIMD_TEST_ROOT/descendant-stopped"; exit 0'\'' TERM; printf ready > "$SIMD_TEST_ROOT/descendant-ready"; while :; do /bin/sleep 1; done' &
            \\while [ ! -f "$SIMD_TEST_ROOT/descendant-ready" ]; do /bin/sleep 0.01; done
            \\printf '%s' "$$" > "$SIMD_TEST_ROOT/marker.tmp"
            \\mv "$SIMD_TEST_ROOT/marker.tmp" "$SIMD_TEST_ROOT/marker"
            \\wait
            \\
        });
        for ([_][]const u8{ "git", "zig" }) |path| {
            const file = try temp.dir.openFile(io, path, .{});
            defer file.close(io);
            try file.setPermissions(io, .executable_file);
        }
        var env = std.process.Environ.Map.init(allocator);
        defer env.deinit();
        const system_path = if (std.c.getenv("PATH")) |value| std.mem.span(value) else "/usr/bin:/bin";
        const path = try std.fmt.allocPrint(allocator, "{s}:{s}", .{ root, system_path });
        defer allocator.free(path);
        try env.put("PATH", path);
        try env.put("TMPDIR", root);
        try env.put("SIMD_TEST_ROOT", root);
        try env.put("SIMD_TEST_STAGE", stage);
        var child = try std.process.spawn(io, .{
            .argv = &.{ executable, "--repo", root, "--runs", "1" },
            .environ_map = &env,
            .stdout = .ignore,
            .stderr = .ignore,
        });
        defer if (child.id != null) child.kill(io);
        const start = std.Io.Clock.awake.now(io);
        while (true) {
            if (temp.dir.statFile(io, "marker", .{})) |_| {
                break;
            } else |err| switch (err) {
                error.FileNotFound => {},
                else => return err,
            }
            if (start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds >= 15 * std.time.ns_per_s)
                return error.SimdCommandDidNotStart;
            try io.sleep(.fromMilliseconds(10), .awake);
        }
        const pid_text = try temp.dir.readFileAlloc(io, "marker", allocator, .unlimited);
        defer allocator.free(pid_text);
        const pid = try std.fmt.parseInt(std.posix.pid_t, pid_text, 10);
        if (std.mem.eql(u8, stage, "orphan-ignore-term")) {
            while (std.posix.errno(std.c.kill(pid, @fromBackingInt(@intCast(0)))) != .SRCH) {
                if (start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds >= 15 * std.time.ns_per_s)
                    return error.SimdCommandDidNotExit;
                try io.sleep(.fromMilliseconds(10), .awake);
            }
        }
        try std.posix.kill(child.id.?, signal);
        const term = while (true) {
            if (try process_control.pollChild(&child, io)) |result| break result;
            if (start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds >= 25 * std.time.ns_per_s)
                return error.SimdCommandDidNotStop;
            try io.sleep(.fromMilliseconds(10), .awake);
        };
        try std.testing.expectEqual(@as(u8, 130), term.exited);
        try std.testing.expectEqual(std.posix.E.SRCH, std.posix.errno(std.c.kill(pid, @fromBackingInt(@intCast(0)))));
        const actions = try temp.dir.readFileAlloc(io, "actions", allocator, .unlimited);
        defer allocator.free(actions);
        const expected_actions = if (std.mem.startsWith(u8, stage, "add"))
            "add\nremove\n"
        else
            "add\nadd\nremove\nremove\n";
        try std.testing.expectEqualStrings(expected_actions, actions);
        if (std.mem.eql(u8, stage, "build")) {
            const stopped = try temp.dir.readFileAlloc(io, "descendant-stopped", allocator, .unlimited);
            defer allocator.free(stopped);
            try std.testing.expectEqualStrings("stopped", stopped);
        }
        const directory = try temp.dir.openDir(io, ".", .{ .iterate = true });
        defer directory.close(io);
        var entries = directory.iterate();
        while (try entries.next(io)) |entry|
            try std.testing.expect(!std.mem.startsWith(u8, entry.name, "bench-simd-"));
    };
}

test "CLI removes a real worktree interrupted during checkout" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const executable = if (std.c.getenv("SIMD_BENCH")) |path| std.mem.span(path) else return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const repo = try std.fs.path.join(allocator, &.{ root, "repo" });
    defer allocator.free(repo);
    try temp.dir.createDirPath(io, "repo/src/tests");
    for ([_][]const u8{ "build.zig", "src/tests/aot_harness.zig", "src/tests/simd_bench_runner.zig" }) |path| {
        const file_path = try std.fs.path.join(allocator, &.{ "repo", path });
        defer allocator.free(file_path);
        try temp.dir.writeFile(io, .{ .sub_path = file_path, .data = "overlay" });
    }
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    const system_path = if (std.c.getenv("PATH")) |value| std.mem.span(value) else "/usr/bin:/bin";
    try env.put("PATH", system_path);
    try env.put("HOME", root);
    try env.put("XDG_CONFIG_HOME", root);
    try env.put("TMPDIR", root);
    try env.put("SIMD_TEST_ROOT", root);
    try env.put("GIT_CONFIG_NOSYSTEM", "1");
    const commands = [_][]const []const u8{
        &.{ "git", "init", "--quiet", repo },
        &.{ "git", "-C", repo, "add", "." },
        &.{ "git", "-C", repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture" },
    };
    for (commands) |command| {
        const result = try std.process.run(allocator, io, .{ .argv = command, .environ_map = &env });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        if (!result.term.success()) {
            std.debug.print("fixture command failed: {s}\n", .{result.stderr});
            return error.GitFixtureFailed;
        }
    }
    const hook = "repo/.git/hooks/post-checkout";
    try temp.dir.writeFile(io, .{ .sub_path = hook, .data =
        \\#!/bin/sh
        \\printf '%s' "$$" > "$SIMD_TEST_ROOT/marker.tmp"
        \\mv "$SIMD_TEST_ROOT/marker.tmp" "$SIMD_TEST_ROOT/marker"
        \\exec /bin/sleep 30
        \\
    });
    const hook_file = try temp.dir.openFile(io, hook, .{});
    defer hook_file.close(io);
    try hook_file.setPermissions(io, .executable_file);
    var child = try std.process.spawn(io, .{
        .argv = &.{ executable, "--repo", repo, "--baseline", "HEAD", "--runs", "1" },
        .environ_map = &env,
        .stdout = .ignore,
        .stderr = .ignore,
    });
    defer if (child.id != null) child.kill(io);
    const start = std.Io.Clock.awake.now(io);
    while (true) {
        if (temp.dir.statFile(io, "marker", .{})) |_| break else |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        }
        if (start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds >= 15 * std.time.ns_per_s)
            return error.GitCheckoutDidNotStart;
        try io.sleep(.fromMilliseconds(10), .awake);
    }
    try std.posix.kill(child.id.?, .TERM);
    const term = while (true) {
        if (try process_control.pollChild(&child, io)) |result| break result;
        if (start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds >= 25 * std.time.ns_per_s)
            return error.GitCheckoutDidNotStop;
        try io.sleep(.fromMilliseconds(10), .awake);
    };
    try std.testing.expectEqual(@as(u8, 130), term.exited);
    const listing = try std.process.run(allocator, io, .{ .argv = &.{ "git", "-C", repo, "worktree", "list", "--porcelain" }, .environ_map = &env });
    defer allocator.free(listing.stdout);
    defer allocator.free(listing.stderr);
    try std.testing.expect(listing.term.success());
    var lines = std.mem.splitScalar(u8, listing.stdout, '\n');
    var count: usize = 0;
    while (lines.next()) |line| if (std.mem.startsWith(u8, line, "worktree ")) {
        count += 1;
    };
    try std.testing.expectEqual(@as(usize, 1), count);
    const directory = try temp.dir.openDir(io, ".", .{ .iterate = true });
    defer directory.close(io);
    var entries = directory.iterate();
    while (try entries.next(io)) |entry|
        try std.testing.expect(!std.mem.startsWith(u8, entry.name, "bench-simd-"));
}
