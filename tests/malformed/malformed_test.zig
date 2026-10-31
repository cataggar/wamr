// Copyright (C) 2019 Intel Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

fn isFixture(name: []const u8) bool {
    var parts = std.mem.splitScalar(u8, name, '.');
    _ = parts.next();
    return std.mem.eql(u8, parts.next() orelse return false, "wasm");
}

fn runtimeOption(args: []const []const u8) ![]const u8 {
    var runtime: []const u8 = "iwasm";
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--")) break;
        if (std.mem.eql(u8, arg, "-r") or std.mem.eql(u8, arg, "--run")) {
            i += 1;
            if (i == args.len) return error.MissingRuntime;
            runtime = args[i];
        } else if (std.mem.startsWith(u8, arg, "--run=")) {
            runtime = arg["--run=".len..];
        } else if (std.mem.startsWith(u8, arg, "-r") and arg.len > 2) {
            runtime = arg[2..];
        } else if (std.mem.startsWith(u8, arg, "-")) {
            return error.UnknownOption;
        }
    }
    return runtime;
}

fn invoke(allocator: Allocator, io: std.Io, runtime: []const u8, path: []const u8) ![]const u8 {
    const command = if (builtin.os.tag == .windows)
        try std.fmt.allocPrint(allocator, "({s} \"{s}\") 2>&1", .{ runtime, path })
    else
        try std.fmt.allocPrint(allocator, "{{ {s} \"$1\"; }} 2>&1", .{runtime});
    defer allocator.free(command);
    const argv: []const []const u8 = if (builtin.os.tag == .windows)
        &.{ "cmd.exe", "/d", "/s", "/c", command }
    else
        &.{ "/bin/sh", "-c", command, "malformed-test", path };
    const result = try std.process.run(allocator, io, .{
        .argv = argv,
        .stdout_limit = .unlimited,
        .stderr_limit = .unlimited,
    });
    allocator.free(result.stderr);
    return result.stdout;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(allocator);
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try std.Io.File.stdout().writeStreamingAll(io, "Usage: malformed-test [-r RUNTIME]\n");
            return;
        }
    }
    const runtime = try runtimeOption(args[1..]);
    var dir = try std.Io.Dir.cwd().openDir(io, ".", .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(allocator);
    defer walker.deinit();
    var passed: usize = 0;
    var failed: usize = 0;
    const started = std.Io.Clock.awake.now(io);
    while (try walker.next(io)) |entry| {
        if (entry.kind == .directory or !isFixture(entry.basename)) continue;
        const path = try std.fmt.allocPrint(allocator, ".{c}{s}", .{ std.fs.path.sep, entry.path });
        const output = try invoke(allocator, io, runtime, path);
        const crashed = std.mem.startsWith(u8, output, "Segmentation fault");
        if (crashed) failed += 1 else passed += 1;
        const label = try std.fmt.allocPrint(allocator, "test {s: <40} ........ [{s}]\n", .{ path, if (crashed) "FAILED" else "PASSED" });
        try std.Io.File.stdout().writeStreamingAll(io, label);
        if (crashed) {
            try std.Io.File.stdout().writeStreamingAll(io, std.mem.trimEnd(u8, output, "\n"));
            try std.Io.File.stdout().writeStreamingAll(io, "\n\n");
        }
    }
    const elapsed: f64 = @as(f64, @floatFromInt(started.durationTo(std.Io.Clock.awake.now(io)).nanoseconds)) / std.time.ns_per_s;
    const summary = try std.fmt.allocPrint(allocator,
        \\
        \\#####################  MALFORMED TEST  ########################
        \\run {d} test cases in {d}ms, {d} passed, {d} failed
        \\
    , .{ passed + failed, elapsed, passed, failed });
    try std.Io.File.stdout().writeStreamingAll(io, summary);
}

test "fixture selection and runtime option match the original interface" {
    try std.testing.expect(isFixture("one.wasm"));
    try std.testing.expect(isFixture("one.wasm.more"));
    try std.testing.expect(!isFixture("one.test.wasm"));
    try std.testing.expect(!isFixture("one"));
    try std.testing.expectEqualStrings("iwasm", try runtimeOption(&.{}));
    try std.testing.expectEqualStrings("wasmer run", try runtimeOption(&.{ "-r", "wasmer run" }));
    try std.testing.expectEqualStrings("wasmtime", try runtimeOption(&.{"--run=wasmtime"}));
    try std.testing.expectError(error.MissingRuntime, runtimeOption(&.{"-r"}));
}

test "fixture output is captured including shell diagnostics" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const output = try invoke(std.testing.allocator, std.testing.io, "printf 'Segmentation fault\\n' >&2; true", "file with spaces.wasm");
    defer std.testing.allocator.free(output);
    try std.testing.expect(std.mem.startsWith(u8, output, "Segmentation fault"));
}
