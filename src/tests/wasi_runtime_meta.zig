//! Metadata queries for the in-tree WASI runtime adapters.
const std = @import("std");
const runtime = @import("wasi_fixture_runtime.zig");
const suite = @import("wasi_fixture_suite.zig");
const fixture = @import("wasi_fixture_config.zig");
const Process = @import("wasi_fixture_process.zig").Process;
const Allocator = std.mem.Allocator;

pub fn name(adapter: runtime.Runtime) []const u8 {
    return switch (adapter) {
        .wamr => "wamr-zig",
        .wasmtime => "wasmtime",
    };
}

pub fn supports(_: runtime.Runtime, wasi_version: suite.Version, _: fixture.World) bool {
    return wasi_version == .wasip1 or wasi_version == .wasip3;
}

fn versionArgs(allocator: Allocator, adapter: runtime.Runtime, executable: []const []const u8) ![]const []const u8 {
    if (executable.len == 0) return error.MissingExecutable;
    const prefix = if (adapter == .wasmtime) executable[0..1] else executable;
    const result = try allocator.alloc([]const u8, prefix.len + 1);
    @memcpy(result[0..prefix.len], prefix);
    result[prefix.len] = if (adapter == .wamr) "version" else "--version";
    return result;
}

fn parseVersion(output: []const u8) ![]const u8 {
    if (!std.unicode.utf8ValidateSlice(output)) return error.InvalidRuntimeVersion;
    const line_end = std.mem.indexOfAny(u8, output, "\r\n") orelse output.len;
    const line = output[0..line_end];
    const separator = std.mem.indexOfScalar(u8, line, ' ') orelse return error.InvalidRuntimeVersion;
    const rest = line[separator + 1 ..];
    const end = std.mem.indexOfScalar(u8, rest, ' ') orelse rest.len;
    return rest[0..end];
}

/// Returns an owned version string. The child is forcibly reaped on timeout.
pub fn version(allocator: Allocator, io: std.Io, adapter: runtime.Runtime, executable: []const []const u8, timeout_ms: u32) ![]u8 {
    const args = try versionArgs(allocator, adapter, executable);
    defer allocator.free(args);
    var child: Process = undefined;
    try child.start(allocator, io, .{ .argv = args });
    defer child.deinit();
    const result = try child.wait(timeout_ms);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("runtime version command failed ({f}): {s}\n", .{ result.term, result.stderr });
        return error.RuntimeVersionCommandFailed;
    }
    return allocator.dupe(u8, try parseVersion(result.stdout));
}

test "version commands and supported WASI versions match the adapters" {
    _ = @import("wasi_fixture_compile.zig");
    _ = @import("wasi_fixture_fs.zig");
    _ = @import("wasi_runtime_env.zig");
    _ = @import("wasi_runtime_prepare.zig");
    const allocator = std.testing.allocator;
    const wamr = try versionArgs(allocator, .wamr, &.{ "wamr", "--flag" });
    defer allocator.free(wamr);
    try std.testing.expectEqualDeep(&[_][]const u8{ "wamr", "--flag", "version" }, wamr);
    const wasmtime = try versionArgs(allocator, .wasmtime, &.{ "wasmtime", "-Wflag" });
    defer allocator.free(wasmtime);
    try std.testing.expectEqualDeep(&[_][]const u8{ "wasmtime", "--version" }, wasmtime);
    try std.testing.expect(supports(.wamr, .wasip3, .http_service));
    try std.testing.expect(!supports(.wasmtime, .wasip2, .cli_command));
    try std.testing.expectEqualStrings("50.0.1", try parseVersion("wasmtime 50.0.1 (build)\n"));
    try std.testing.expectError(error.InvalidRuntimeVersion, parseVersion("malformed"));
    try std.testing.expectError(error.InvalidRuntimeVersion, parseVersion("wamr \xff"));
}

test "runtime version command is bounded and returns its version" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const result = try version(allocator, io, .wamr, &.{ "/bin/sh", "-c", "printf 'wamr 1.2.3\\n'" }, 5000);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("1.2.3", result);
    try std.testing.expectError(error.Timeout, version(allocator, io, .wamr, &.{ "/bin/sh", "-c", "while :; do :; done" }, 100));
}
