//! AOT cache inspection and bounded fixture compilation.
const std = @import("std");
const runtime = @import("wasi_fixture_runtime.zig");
const Process = @import("wasi_fixture_process.zig").Process;
const Allocator = std.mem.Allocator;

/// The returned plan uses caller-owned allocation, like `runtime.compilePlan`.
pub fn prepare(
    allocator: Allocator,
    io: std.Io,
    compiler: []const []const u8,
    source: []const u8,
    jit: bool,
    compiler_mtime: i128,
) !runtime.CompilePlan {
    if (!std.mem.endsWith(u8, source, ".wasm")) {
        return runtime.compilePlan(allocator, compiler, source, .core, jit, 0, compiler_mtime, null);
    }
    const cwd = std.Io.Dir.cwd();
    const input = try cwd.openFile(io, source, .{});
    defer input.close(io);
    var header: [8]u8 = undefined;
    const length = try input.readPositionalAll(io, &header, 0);
    const kind = runtime.artifactKind(header[0..length]);
    if (jit) return runtime.compilePlan(allocator, compiler, source, kind, true, 0, compiler_mtime, null);
    const input_mtime = (try input.stat(io)).mtime.nanoseconds;
    const artifact = try std.fmt.allocPrint(allocator, "{s}{s}", .{
        source[0 .. source.len - ".wasm".len],
        if (kind == .component) ".cwasm.json" else ".cwasm",
    });
    defer allocator.free(artifact);
    const artifact_mtime: ?i128 = if (cwd.statFile(io, artifact, .{})) |stat|
        stat.mtime.nanoseconds
    else |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    return runtime.compilePlan(allocator, compiler, source, kind, false, input_mtime, compiler_mtime, artifact_mtime);
}

/// Executes cache misses; failures and timeouts are surfaced to the suite.
pub fn execute(allocator: Allocator, io: std.Io, plan: runtime.CompilePlan, timeout_ms: u32) !void {
    if (plan.cache != .miss) return;
    const args = plan.argv orelse return error.MissingCompilerCommand;
    var child: Process = undefined;
    try child.start(allocator, io, .{ .argv = args });
    defer child.deinit();
    const result = try child.wait(timeout_ms);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("fixture compilation failed ({f})\nstdout: {s}\nstderr: {s}\n", .{ result.term, result.stdout, result.stderr });
        return error.FixtureCompilationFailed;
    }
}

test "cache preparation identifies misses, hits, and JIT bypass" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "fixture.wasm", .data = "\x00asm\x0d\x00\x01\x00" });
    const source = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/fixture.wasm", .{tmp.sub_path});
    const miss = try prepare(allocator, io, &.{"wamrc"}, source, false, 0);
    try std.testing.expectEqual(runtime.Cache.miss, miss.cache);
    try std.testing.expectEqual(runtime.ArtifactKind.component, miss.kind);
    try tmp.dir.writeFile(io, .{ .sub_path = "fixture.cwasm.json", .data = "{}" });
    const hit = try prepare(allocator, io, &.{"wamrc"}, source, false, 0);
    try std.testing.expectEqual(runtime.Cache.hit, hit.cache);
    const stale = try prepare(allocator, io, &.{"wamrc"}, source, false, std.math.maxInt(i128));
    try std.testing.expectEqual(runtime.Cache.miss, stale.cache);
    const jit = try prepare(allocator, io, &.{}, source, true, 0);
    try std.testing.expectEqual(runtime.Cache.bypass, jit.cache);
}

test "compiler invocation is bounded" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const plan: runtime.CompilePlan = .{
        .input = "fixture",
        .output = "artifact",
        .kind = .core,
        .cache = .miss,
        .argv = &.{ "/bin/sh", "-c", "exit 0" },
    };
    try execute(allocator, io, plan, 5000);
    var stalled = plan;
    stalled.argv = &.{ "/bin/sh", "-c", "while :; do :; done" };
    try std.testing.expectError(error.Timeout, execute(allocator, io, stalled, 100));
}
