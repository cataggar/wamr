//! Native release smoke checks; archive inspection must happen before extraction.
const std = @import("std");
const builtin = @import("builtin");
const binary = @import("release_binary.zig");
const process_control = @import("process_control.zig");
const Allocator = std.mem.Allocator;

pub const fixtures = [_]struct { path: []const u8, artifact: []const u8, output: []const u8 }{
    .{ .path = "tests/coldstart/noop.wasm", .artifact = "noop.cwasm", .output = "" },
    .{
        .path = "tests/regressions/1008-aot-magic-u32/printf-decimal.wasm",
        .artifact = "printf-decimal.cwasm",
        .output = "{\"checksum\":13856768990818897060,\"cases\":" ++ "[[1385676899,138567689,9],[2147483648,214748364,8]," ++ "[3000000008,300000000,8]]}\n",
    },
};

pub const Case = struct {
    fixture: []const u8,
    fixture_sha256: []const u8,
    stdout_sha256: []const u8,
};

pub const Smoke = struct {
    status: []const u8 = "passed",
    versions: struct {
        @"wamr version": []const u8,
        @"wamr --version": []const u8,
        @"wamrc version": []const u8,
    },
    cases: [fixtures.len]Case,

    pub fn deinit(self: *Smoke, allocator: Allocator) void {
        allocator.free(self.versions.@"wamr version");
        allocator.free(self.versions.@"wamrc version");
        for (self.cases) |case| {
            allocator.free(case.fixture_sha256);
            allocator.free(case.stdout_sha256);
        }
        self.* = undefined;
    }
};

fn digest(allocator: Allocator, bytes: []const u8) ![]const u8 {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    const hex = std.fmt.bytesToHex(hash, .lower);
    return allocator.dupe(u8, &hex);
}

fn checkedCommand(allocator: Allocator, io: std.Io, args: []const []const u8, expected: ?[]const u8, timeout: u32) !void {
    const result = process_control.runTimed(allocator, io, .{ .argv = args }, timeout) catch |err| {
        std.debug.print("[release-smoke] failed to execute {s}: {s}\n", .{ args[0], @errorName(err) });
        return err;
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    if (!result.term.success() or
        (expected != null and (!std.mem.eql(u8, result.stdout, expected.?) or result.stderr.len != 0)))
    {
        std.debug.print("[release-smoke] command failed ({f}):", .{result.term});
        for (args) |arg| std.debug.print(" {s}", .{arg});
        std.debug.print("\nstdout:\n{s}\nstderr:\n{s}\n", .{ result.stdout, result.stderr });
        return error.ReleaseSmokeCommandFailed;
    }
}

fn requireNative(target: binary.Target) !void {
    const system: binary.System = switch (builtin.os.tag) {
        .linux => .linux,
        .macos => .macos,
        .windows => .windows,
        else => return error.UnsupportedSmokeHost,
    };
    const arch: binary.Arch = switch (builtin.cpu.arch) {
        .x86_64 => .x86_64,
        .aarch64 => .aarch64,
        else => return error.UnsupportedSmokeHost,
    };
    if (system != target.system or arch != target.arch) return error.NonNativeSmokeTarget;
}

/// The returned strings belong to allocator. `work_dir` must already exist.
pub fn run(allocator: Allocator, io: std.Io, repo: []const u8, bin_dir: []const u8, work_dir: []const u8, target: binary.Target, version: []const u8) !Smoke {
    try requireNative(target);
    const extension = if (target.system == .windows) ".exe" else "";
    const wamr_name = try std.fmt.allocPrint(allocator, "wamr{s}", .{extension});
    defer allocator.free(wamr_name);
    const wamrc_name = try std.fmt.allocPrint(allocator, "wamrc{s}", .{extension});
    defer allocator.free(wamrc_name);
    const wamr = try std.fs.path.join(allocator, &.{ bin_dir, wamr_name });
    defer allocator.free(wamr);
    const wamrc = try std.fs.path.join(allocator, &.{ bin_dir, wamrc_name });
    defer allocator.free(wamrc);
    const wamr_version = try std.fmt.allocPrint(allocator, "wamr {s}\n", .{version});
    errdefer allocator.free(wamr_version);
    const wamrc_version = try std.fmt.allocPrint(allocator, "wamrc {s}\n", .{version});
    errdefer allocator.free(wamrc_version);
    try checkedCommand(allocator, io, &.{ wamr, "version" }, wamr_version, 120);
    try checkedCommand(allocator, io, &.{ wamrc, "version" }, wamrc_version, 120);
    try checkedCommand(allocator, io, &.{ wamr, "--version" }, wamr_version, 120);
    var cases: [fixtures.len]Case = undefined;
    var completed: usize = 0;
    errdefer for (cases[0..completed]) |case| {
        allocator.free(case.fixture_sha256);
        allocator.free(case.stdout_sha256);
    };
    for (fixtures, 0..) |fixture, index| {
        const path = try std.fs.path.join(allocator, &.{ repo, fixture.path });
        defer allocator.free(path);
        const artifact = try std.fs.path.join(allocator, &.{ work_dir, fixture.artifact });
        defer allocator.free(artifact);
        try checkedCommand(allocator, io, &.{ wamrc, "compile", path, "--target", @tagName(target.arch), "-o", artifact }, null, 120);
        try checkedCommand(allocator, io, &.{ wamr, "run", artifact }, fixture.output, 120);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(bytes);
        const fixture_digest = try digest(allocator, bytes);
        errdefer allocator.free(fixture_digest);
        cases[index] = .{
            .fixture = fixture.path,
            .fixture_sha256 = fixture_digest,
            .stdout_sha256 = try digest(allocator, fixture.output),
        };
        completed += 1;
    }
    return .{
        .versions = .{
            .@"wamr version" = wamr_version,
            .@"wamr --version" = wamr_version,
            .@"wamrc version" = wamrc_version,
        },
        .cases = cases,
    };
}

test "smoke commands require successful execution and exact clean output" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try checkedCommand(allocator, io, &.{ "/bin/sh", "-c", "printf ok" }, "ok", 5);
    try std.testing.expectError(error.ReleaseSmokeCommandFailed, checkedCommand(allocator, io, &.{ "/bin/sh", "-c", "printf wrong" }, "ok", 5));
    try std.testing.expectError(error.ReleaseSmokeCommandFailed, checkedCommand(allocator, io, &.{ "/bin/sh", "-c", "printf ok; printf warning >&2" }, "ok", 5));
    try std.testing.expectError(error.ReleaseSmokeCommandFailed, checkedCommand(allocator, io, &.{ "/bin/sh", "-c", "exit 7" }, null, 5));
}

test "smoke rejects non-native targets before executing any commands" {
    const host = switch (builtin.os.tag) {
        .linux => binary.System.linux,
        .macos => binary.System.macos,
        .windows => binary.System.windows,
        else => return error.SkipZigTest,
    };
    const other: binary.System = if (host == .windows) .linux else .windows;
    try std.testing.expectError(error.NonNativeSmokeTarget, run(std.testing.allocator, std.testing.io, ".", ".", ".", .{ .name = "other", .system = other, .arch = switch (builtin.cpu.arch) {
        .x86_64 => .x86_64,
        .aarch64 => .aarch64,
        else => return error.SkipZigTest,
    } }, "dev"));
}

test "native smoke verifies versions, fixture execution, and report hashes" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const system: binary.System = switch (builtin.os.tag) {
        .linux => .linux,
        .macos => .macos,
        else => return error.SkipZigTest,
    };
    const arch: binary.Arch = switch (builtin.cpu.arch) {
        .x86_64 => .x86_64,
        .aarch64 => .aarch64,
        else => return error.SkipZigTest,
    };
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    try temp.dir.createDirPath(io, "bin");
    try temp.dir.createDirPath(io, "work");
    try temp.dir.createDirPath(io, "tests/coldstart");
    try temp.dir.createDirPath(io, "tests/regressions/1008-aot-magic-u32");
    try temp.dir.writeFile(io, .{ .sub_path = fixtures[0].path, .data = "noop fixture" });
    try temp.dir.writeFile(io, .{ .sub_path = fixtures[1].path, .data = "decimal fixture" });
    try temp.dir.writeFile(io, .{
        .sub_path = "bin/wamr",
        .data =
        \\#!/bin/sh
        \\case "$1" in
        \\  version|--version) printf 'wamr test\n' ;;
        \\  run) case "$2" in
        \\    *printf-decimal.cwasm) printf '%s\n' '{"checksum":13856768990818897060,"cases":[[1385676899,138567689,9],[2147483648,214748364,8],[3000000008,300000000,8]]}' ;;
        \\  esac ;;
        \\  *) exit 7 ;;
        \\esac
        \\
        ,
    });
    try temp.dir.writeFile(io, .{
        .sub_path = "bin/wamrc",
        .data =
        \\#!/bin/sh
        \\case "$1" in
        \\  version) printf 'wamrc test\n' ;;
        \\  compile) : > "$6" ;;
        \\  *) exit 7 ;;
        \\esac
        \\
        ,
    });
    const bin_dir = try std.fs.path.join(allocator, &.{ root, "bin" });
    defer allocator.free(bin_dir);
    const work_dir = try std.fs.path.join(allocator, &.{ root, "work" });
    defer allocator.free(work_dir);
    const wamr = try std.fs.path.join(allocator, &.{ bin_dir, "wamr" });
    defer allocator.free(wamr);
    const wamrc = try std.fs.path.join(allocator, &.{ bin_dir, "wamrc" });
    defer allocator.free(wamrc);
    try checkedCommand(allocator, io, &.{ "chmod", "+x", wamr, wamrc }, null, 5);
    const target: binary.Target = .{ .name = "native", .system = system, .arch = arch };
    var result = try run(allocator, io, root, bin_dir, work_dir, target, "test");
    defer result.deinit(allocator);
    try std.testing.expectEqualStrings("passed", result.status);
    try std.testing.expectEqualStrings("wamr test\n", result.versions.@"wamr --version");
    try std.testing.expectEqualStrings("af032c2dc4829dbf4f3279b3e5e42a606c2e15c9e7aa36e86258d66d33be3a2e", result.cases[0].fixture_sha256);
    try std.testing.expectEqualStrings("370e99eddab791eff8d4619c0fcf243117be05753a070d87a61cad217866f38a", result.cases[1].fixture_sha256);
    try std.testing.expectEqualStrings("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", result.cases[0].stdout_sha256);
    try std.testing.expectEqualStrings("b30c92c5b7361b922e5b213b4214b83cf8c0d10089c54d73e5808d659d636c05", result.cases[1].stdout_sha256);
    try std.testing.expectError(error.ReleaseSmokeCommandFailed, run(allocator, io, root, bin_dir, work_dir, target, "wrong"));
}
