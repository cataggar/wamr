//! Lazy adapter command preparation for individual fixture runs.
const std = @import("std");
const fixture = @import("wasi_fixture_config.zig");
const suite = @import("wasi_fixture_suite.zig");
const runtime = @import("wasi_fixture_runtime.zig");
const compile = @import("wasi_fixture_compile.zig");
const fs = @import("wasi_fixture_fs.zig");
const profile = @import("wasi_fixture_profile.zig");
const sampling = @import("wasi_fixture_sampling.zig");
const Allocator = std.mem.Allocator;

fn cleanSnapshot(allocator: Allocator, io: std.Io, source: []const u8, temp_path: []const u8) !fs.Snapshot {
    var snapshot = try fs.snapshot(allocator, io, source, temp_path);
    errdefer snapshot.destroy(io) catch |err| std.debug.print("failed to remove fixture snapshot: {s}\n", .{@errorName(err)});
    const dir = try std.Io.Dir.cwd().openDir(io, snapshot.path, .{ .iterate = true });
    defer dir.close(io);
    try fs.cleanupProducts(io, dir);
    return snapshot;
}

/// The allocator must outlive all commands returned by this object.
pub const Preparation = struct {
    allocator: Allocator,
    io: std.Io,
    adapter: runtime.Runtime,
    executable: []const []const u8,
    compiler: []const []const u8 = &.{},
    compiler_mtime: i128 = 0,
    compiler_timeout_ms: u32 = 600_000,
    fixture_path: []const u8,
    temp_path: []const u8,
    config: fixture.Config,
    version: suite.Version,
    jit: bool = false,
    timing: ?profile.Sink = null,
    sampling: ?sampling.Sampling = null,
    snapshots: std.ArrayList(fs.Snapshot) = .empty,

    pub fn command(raw: *anyopaque, _: usize, run: fixture.Run) ![]const []const u8 {
        const self: *Preparation = @ptrCast(@alignCast(raw));
        var root: ?[]const u8 = null;
        if (self.adapter != .wamr or self.config.world != .http_service) {
            if (run.root) |path| {
                if (path.len != 0) {
                    const parent = std.fs.path.dirname(self.fixture_path) orelse ".";
                    const source = if (std.fs.path.isAbsolute(path))
                        try self.allocator.dupe(u8, path)
                    else
                        try std.fs.path.join(self.allocator, &.{ parent, path });
                    const stat = std.Io.Dir.cwd().statFile(self.io, source, .{}) catch |err| switch (err) {
                        error.FileNotFound, error.NotDir => null,
                        else => return err,
                    };
                    if (stat != null and stat.?.kind == .directory) {
                        var snapshot = try cleanSnapshot(self.allocator, self.io, source, self.temp_path);
                        self.snapshots.append(self.allocator, snapshot) catch |err| {
                            snapshot.destroy(self.io) catch |cleanup_err| std.debug.print("failed to remove fixture snapshot: {s}\n", .{@errorName(cleanup_err)});
                            return err;
                        };
                        root = snapshot.path;
                    } else {
                        root = source;
                    }
                }
            }
        }

        var artifact = self.fixture_path;
        if (self.adapter == .wamr) {
            const started = if (self.timing != null) std.Io.Clock.awake.now(self.io) else undefined;
            var plan = try compile.prepare(self.allocator, self.io, self.compiler, self.fixture_path, self.jit, self.compiler_mtime);
            if (self.sampling) |sampler| {
                if (plan.cache == .miss) {
                    plan.argv = try sampler.command(self.allocator, self.io, self.fixture_path, if (plan.kind == .component) .component_precompile else .core_precompile, plan.argv orelse return error.MissingCompilerCommand);
                }
            }
            try compile.execute(self.allocator, self.io, plan, self.compiler_timeout_ms);
            if (self.timing) |sink| {
                if (plan.cache != .not_applicable) {
                    try sink.emit(sink.context, .{
                        .fixture = std.fs.path.stem(self.fixture_path),
                        .phase = if (plan.kind == .component) .component_precompile else .core_precompile,
                        .artifact_kind = plan.kind,
                        .cache = plan.cache,
                        .duration_ns = if (plan.cache == .bypass) 0 else profile.elapsed(self.io, started),
                    });
                }
            }
            artifact = plan.input;
        }
        const args = try runtime.command(self.allocator, self.adapter, self.executable, artifact, run, root, self.config.proposals, self.config.world, self.version);
        if (self.sampling) |sampler| {
            return sampler.command(self.allocator, self.io, self.fixture_path, .fixture_execution, args);
        }
        return args;
    }

    /// Attempt all snapshot removals, reporting every failure.
    pub fn deinit(self: *Preparation) !void {
        var first_error: ?anyerror = null;
        for (self.snapshots.items) |*snapshot| {
            snapshot.destroy(self.io) catch |err| {
                if (first_error == null) first_error = err;
            };
        }
        self.snapshots.deinit(self.allocator);
        if (first_error) |err| return err;
    }
};

test "each run receives an independent snapshot and cleanup leaves the source intact" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(io, "root", .default_dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "root/input", .data = "original" });
    try tmp.dir.writeFile(io, .{ .sub_path = "root/stale.cleanup", .data = "old" });
    try tmp.dir.createDir(io, "root/nested", .default_dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "root/nested/stale.cleanup", .data = "old" });
    try tmp.dir.createDir(io, "root/dir.cleanup", .default_dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "root/dir.cleanup/file", .data = "old" });
    try tmp.dir.symLink(io, "input", "root/link.cleanup", .{});
    const temp_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const fixture_path = try std.fs.path.join(allocator, &.{ temp_path, "test.wasm" });
    var preparation: Preparation = .{
        .allocator = allocator,
        .io = io,
        .adapter = .wasmtime,
        .executable = &.{"wasmtime"},
        .fixture_path = fixture_path,
        .temp_path = temp_path,
        .config = .{ .operations = &.{} },
        .version = .wasip1,
    };
    var alive = true;
    defer if (alive) preparation.deinit() catch |err| std.debug.print("cleanup failed: {s}\n", .{@errorName(err)});
    const run: fixture.Run = .{ .root = "root" };
    const first = try Preparation.command(&preparation, 0, run);
    try std.testing.expectEqualStrings("--dir", first[1]);
    const first_path = preparation.snapshots.items[0].path;
    const first_dir = try std.Io.Dir.cwd().openDir(io, first_path, .{});
    defer first_dir.close(io);
    try std.testing.expectError(error.FileNotFound, first_dir.statFile(io, "stale.cleanup", .{}));
    try std.testing.expectError(error.FileNotFound, first_dir.statFile(io, "nested/stale.cleanup", .{}));
    try std.testing.expectError(error.FileNotFound, first_dir.statFile(io, "dir.cleanup", .{}));
    try std.testing.expectEqual(std.Io.File.Kind.sym_link, (try first_dir.statFile(io, "link.cleanup", .{ .follow_symlinks = false })).kind);
    try first_dir.writeFile(io, .{ .sub_path = "input", .data = "changed" });
    const second = try Preparation.command(&preparation, 1, run);
    try std.testing.expectEqualStrings("--dir", second[1]);
    const second_path = preparation.snapshots.items[1].path;
    try std.testing.expect(!std.mem.eql(u8, first_path, second_path));
    const second_dir = try std.Io.Dir.cwd().openDir(io, second_path, .{});
    defer second_dir.close(io);
    const contents = try second_dir.readFileAlloc(io, "input", allocator, .unlimited);
    try std.testing.expectEqualStrings("original", contents);
    const first_name = try allocator.dupe(u8, preparation.snapshots.items[0].name);
    const second_name = try allocator.dupe(u8, preparation.snapshots.items[1].name);
    alive = false;
    try preparation.deinit();
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, first_name, .{}));
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, second_name, .{}));
    const original = try tmp.dir.readFileAlloc(io, "root/input", allocator, .unlimited);
    try std.testing.expectEqualStrings("original", original);
    const stale = try tmp.dir.readFileAlloc(io, "root/stale.cleanup", allocator, .unlimited);
    try std.testing.expectEqualStrings("old", stale);
}

test "executor prepares a fresh isolated root on each run" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const execute = @import("wasi_fixture_execute.zig");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(io, "root", .default_dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "root/input", .data = "original\n" });
    const temp_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const fixture_path = try std.fs.path.join(allocator, &.{ temp_path, "test.wasm" });
    const operations = [_]fixture.Operation{
        .{ .run = .{ .root = "root" } },
        .{ .read = .{ .id = "stdout", .payload = "original" } },
        .{ .wait = 0 },
        .{ .run = .{ .root = "root" } },
        .{ .read = .{ .id = "stdout", .payload = "original" } },
        .{ .wait = 0 },
    };
    var preparation: Preparation = .{
        .allocator = allocator,
        .io = io,
        .adapter = .wasmtime,
        .executable = &.{
            "/bin/sh",
            "-c",
            "dir=${2%::/}; read -r value < \"$dir/input\"; printf %s \"$value\"; printf changed > \"$dir/input\"",
            "sh",
        },
        .fixture_path = fixture_path,
        .temp_path = temp_path,
        .config = .{ .operations = &operations },
        .version = .wasip1,
    };
    defer preparation.deinit() catch |err| std.debug.print("cleanup failed: {s}\n", .{@errorName(err)});
    const result = try execute.executeWithProvider(allocator, io, &operations, .{
        .context = &preparation,
        .command = Preparation.command,
    }, .{ .operation_timeout_ms = 5000, .wait_timeout_ms = 5000 });
    try std.testing.expectEqual(@as(usize, 0), result.failures.len);
    try std.testing.expectEqual(@as(usize, 2), preparation.snapshots.items.len);
    const original = try tmp.dir.readFileAlloc(io, "root/input", allocator, .unlimited);
    try std.testing.expectEqualStrings("original\n", original);
}

test "sampling wraps compilation cache misses and fixture execution" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const base = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const fixture_path = try std.fs.path.join(allocator, &.{ base, "test.wasm" });
    const launcher = try std.fs.path.join(allocator, &.{ base, "launcher" });
    const selection = try std.fs.path.join(allocator, &.{ base, "selection.json" });
    const marker = try std.fs.path.join(allocator, &.{ base, "compiler-ran" });
    const launcher_log = try std.fs.path.join(allocator, &.{ base, "launcher-log" });
    try tmp.dir.writeFile(io, .{ .sub_path = "test.wasm", .data = "\x00asm\x01\x00\x00\x00" });
    try tmp.dir.writeFile(io, .{ .sub_path = "selection.json", .data =
        \\{"profiles":[{"mode":"aot","fixture":"test","phase":"core_precompile"},
        \\{"mode":"aot","fixture":"test","phase":"fixture_execution"},
        \\{"mode":"jit","fixture":"test","phase":"fixture_execution"}]}
    });
    const script = try std.fmt.allocPrint(allocator,
        \\#!/bin/sh
        \\printf '%s\n' "$2" >> '{s}'
        \\shift 5
        \\exec "$@"
        \\
    , .{launcher_log});
    try tmp.dir.writeFile(io, .{ .sub_path = "launcher", .data = script });
    try tmp.dir.setFilePermissions(io, "launcher", .fromMode(0o755), .{});
    var preparation: Preparation = .{
        .allocator = allocator,
        .io = io,
        .adapter = .wamr,
        .executable = &.{ "/bin/sh", "-c", "printf fixture", "runtime" },
        .compiler = &.{ "/bin/sh", "-c", "printf compiled > \"$1\"", "compiler", marker },
        .fixture_path = fixture_path,
        .temp_path = base,
        .config = .{ .operations = &.{} },
        .version = .wasip1,
        .sampling = .{ .selection = selection, .output_dir = "profiles", .mode = "aot", .launcher = launcher },
    };
    defer preparation.deinit() catch |err| std.debug.print("cleanup failed: {s}\n", .{@errorName(err)});
    const first = try Preparation.command(&preparation, 0, .{});
    try std.testing.expectEqualStrings(launcher, first[0]);
    try std.testing.expectEqualStrings("profiles/test-fixture_execution-aot.sample.txt", first[2]);
    const compiled = try tmp.dir.readFileAlloc(io, "compiler-ran", allocator, .unlimited);
    try std.testing.expectEqualStrings("compiled", compiled);
    const Process = @import("wasi_fixture_process.zig").Process;
    var child: Process = undefined;
    try child.start(allocator, io, .{ .argv = first });
    defer child.deinit();
    const result = try child.wait(5000);
    try std.testing.expect(result.term.success());
    try std.testing.expectEqualStrings("fixture", result.stdout);
    try tmp.dir.writeFile(io, .{ .sub_path = "test.cwasm", .data = "cached" });
    const second = try Preparation.command(&preparation, 1, .{});
    try std.testing.expectEqualStrings(launcher, second[0]);
    preparation.jit = true;
    preparation.sampling.?.mode = "jit";
    const jit = try Preparation.command(&preparation, 2, .{});
    try std.testing.expectEqualStrings("profiles/test-fixture_execution-jit.sample.txt", jit[2]);
    try std.testing.expectEqualStrings(fixture_path, jit[jit.len - 1]);
    const log = try tmp.dir.readFileAlloc(io, "launcher-log", allocator, .unlimited);
    try std.testing.expectEqualStrings(
        "profiles/test-core_precompile-aot.sample.txt\nprofiles/test-fixture_execution-aot.sample.txt\n",
        log,
    );
}
