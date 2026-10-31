//! WASI fixture runtime command construction. The caller precompiles and
//! isolates the fixture root before passing the resulting paths here.
const std = @import("std");
const fixture = @import("wasi_fixture_config.zig");
const suite = @import("wasi_fixture_suite.zig");
const Allocator = std.mem.Allocator;

pub const Runtime = enum { wamr, wasmtime };
pub const ArtifactKind = enum { core, component };
pub const Cache = enum { bypass, hit, miss, not_applicable };
pub const CompilePlan = struct {
    input: []const u8,
    output: ?[]const u8,
    kind: ArtifactKind,
    cache: Cache,
    argv: ?[]const []const u8,
};

pub fn artifactKind(header: []const u8) ArtifactKind {
    return if (header.len >= 8 and std.mem.eql(u8, header[0..8], "\x00asm\x0d\x00\x01\x00")) .component else .core;
}

/// Modification times use the same units; `artifact_time == null` means absent.
pub fn compilePlan(
    allocator: Allocator,
    compiler: []const []const u8,
    source: []const u8,
    kind: ArtifactKind,
    jit: bool,
    source_time: i128,
    compiler_time: i128,
    artifact_time: ?i128,
) !CompilePlan {
    if (jit) return .{ .input = source, .output = null, .kind = kind, .cache = .bypass, .argv = null };
    if (!std.mem.endsWith(u8, source, ".wasm")) return .{ .input = source, .output = null, .kind = kind, .cache = .not_applicable, .argv = null };
    const output = try std.fmt.allocPrint(allocator, "{s}{s}", .{
        source[0 .. source.len - ".wasm".len],
        if (kind == .component) ".cwasm.json" else ".cwasm",
    });
    const input = if (kind == .component) source else output;
    if (artifact_time) |time| {
        if (time >= source_time and time >= compiler_time) return .{ .input = input, .output = output, .kind = kind, .cache = .hit, .argv = null };
    }
    if (compiler.len == 0) return error.MissingCompiler;
    var args: std.ArrayList([]const u8) = .empty;
    try args.appendSlice(allocator, compiler);
    if (kind == .component) {
        try args.appendSlice(allocator, &.{ "compile-component", "-o", output, source });
    } else {
        try args.appendSlice(allocator, &.{ "compile", "--no-verify-ir", "-o", output, source });
    }
    return .{ .input = input, .output = output, .kind = kind, .cache = .miss, .argv = try args.toOwnedSlice(allocator) };
}

pub fn command(
    allocator: Allocator,
    runtime: Runtime,
    executable: []const []const u8,
    artifact: []const u8,
    run: fixture.Run,
    isolated_root: ?[]const u8,
    proposals: []const fixture.Proposal,
    world: fixture.World,
    version: suite.Version,
) ![]const []const u8 {
    if (executable.len == 0) return error.MissingExecutable;
    var args: std.ArrayList([]const u8) = .empty;
    try args.appendSlice(allocator, executable);
    if (runtime == .wamr) try args.append(allocator, if (world == .http_service) "serve" else "run");
    for (run.env) |entry| {
        try args.append(allocator, "--env");
        try args.append(allocator, try std.fmt.allocPrint(allocator, "{s}={s}", .{ entry.name, entry.value }));
    }
    if (world != .http_service or runtime == .wasmtime) {
        if (isolated_root) |root| {
            if (root.len != 0) {
                try args.append(allocator, if (runtime == .wamr) "--map-dir" else "--dir");
                try args.append(allocator, try std.fmt.allocPrint(allocator, "{s}::/", .{root}));
            }
        }
    }
    var http = false;
    var sockets = false;
    for (proposals) |proposal| switch (proposal) {
        .http => http = true,
        .sockets => sockets = true,
    };
    if (runtime == .wamr) {
        if (world != .http_service and sockets) try args.appendSlice(allocator, &.{
            "--allow-net", "127.0.0.0/8", "--allow-net", "::1/128",
        });
        if (world == .http_service) try args.appendSlice(allocator, &.{ "--addr", "127.0.0.1:0" });
    }
    try args.append(allocator, artifact);
    try args.appendSlice(allocator, run.args);

    if (runtime == .wasmtime) {
        var index = executable.len;
        while (index > 1 and std.mem.startsWith(u8, executable[index - 1], "-")) : (index -= 1) {}
        if (world == .http_service) {
            try args.insertSlice(allocator, index, &.{ "serve", "-Scli", "--addr=127.0.0.1:0" });
            index += 1;
        }
        if (version == .wasip3) {
            const flags = try std.fmt.allocPrint(allocator, "-Sp3{s}{s}", .{
                if (http) ",http" else "",
                if (sockets) ",inherit-network" else "",
            });
            try args.insertSlice(allocator, index, &.{ "-Wcomponent-model-async", flags });
        }
    }
    return args.toOwnedSlice(allocator);
}

test "WAMR and Wasmtime commands preserve environment, root and proposal flags" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const run: fixture.Run = .{
        .args = &.{"argument"},
        .env = &.{.{ .name = "K", .value = "V" }},
    };
    const wamr = try command(a, .wamr, &.{"wamr"}, "test.cwasm", run, "/isolated", &.{.sockets}, .cli_command, .wasip1);
    try std.testing.expectEqualDeep(&[_][]const u8{
        "wamr",        "run",         "--env",       "K=V",     "--map-dir",  "/isolated::/",
        "--allow-net", "127.0.0.0/8", "--allow-net", "::1/128", "test.cwasm", "argument",
    }, wamr);
    const wasmtime = try command(a, .wasmtime, &.{ "wasmtime", "-Woverride" }, "test.wasm", run, "/isolated", &.{ .http, .sockets }, .http_service, .wasip3);
    try std.testing.expectEqualDeep(&[_][]const u8{
        "wasmtime", "serve",              "-Wcomponent-model-async", "-Sp3,http,inherit-network",
        "-Scli",    "--addr=127.0.0.1:0", "-Woverride",              "--env",
        "K=V",      "--dir",              "/isolated::/",            "test.wasm",
        "argument",
    }, wasmtime);
    const service = try command(a, .wamr, &.{"wamr"}, "service.wasm", .{}, "/ignored", &.{.sockets}, .http_service, .wasip3);
    try std.testing.expectEqualDeep(&[_][]const u8{ "wamr", "serve", "--addr", "127.0.0.1:0", "service.wasm" }, service);
    try std.testing.expectError(error.MissingExecutable, command(a, .wamr, &.{}, "test.wasm", .{}, null, &.{}, .cli_command, .wasip1));
}

test "AOT cache is invalidated by source or compiler changes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqual(ArtifactKind.component, artifactKind("\x00asm\x0d\x00\x01\x00"));
    try std.testing.expectEqual(ArtifactKind.core, artifactKind("\x00asm\x01\x00\x00\x00"));
    try std.testing.expectEqual(ArtifactKind.core, artifactKind("\x00asm"));
    const miss = try compilePlan(a, &.{"wamrc"}, "dir/test.wasm", .core, false, 10, 20, 19);
    try std.testing.expectEqual(Cache.miss, miss.cache);
    try std.testing.expectEqualStrings("dir/test.cwasm", miss.input);
    try std.testing.expectEqualDeep(&[_][]const u8{ "wamrc", "compile", "--no-verify-ir", "-o", "dir/test.cwasm", "dir/test.wasm" }, miss.argv.?);
    const source_stale = try compilePlan(a, &.{"wamrc"}, "dir/test.wasm", .core, false, 21, 20, 20);
    try std.testing.expectEqual(Cache.miss, source_stale.cache);
    const hit = try compilePlan(a, &.{"wamrc"}, "dir/test.wasm", .component, false, 10, 20, 20);
    try std.testing.expectEqual(Cache.hit, hit.cache);
    try std.testing.expectEqualStrings("dir/test.wasm", hit.input);
    try std.testing.expectEqualStrings("dir/test.cwasm.json", hit.output.?);
    const component_miss = try compilePlan(a, &.{ "wrapper", "wamrc" }, "dir/test.wasm", .component, false, 10, 20, null);
    try std.testing.expectEqualDeep(&[_][]const u8{ "wrapper", "wamrc", "compile-component", "-o", "dir/test.cwasm.json", "dir/test.wasm" }, component_miss.argv.?);
    const bypass = try compilePlan(a, &.{}, "dir/test.wasm", .core, true, 10, 20, null);
    try std.testing.expectEqual(Cache.bypass, bypass.cache);
    try std.testing.expectEqualStrings("dir/test.wasm", bypass.input);
    try std.testing.expectError(error.MissingCompiler, compilePlan(a, &.{}, "dir/test.wasm", .core, false, 10, 20, null));
}
