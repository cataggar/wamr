//! Discovery and manifest parsing for wasi-testsuite fixtures.
const std = @import("std");
const fixture = @import("wasi_fixture_config.zig");
const Allocator = std.mem.Allocator;

pub const Version = enum {
    wasip1,
    wasip2,
    wasip3,

    fn parse(name: []const u8) ?Version {
        if (std.mem.eql(u8, name, "wasm32-wasip1")) return .wasip1;
        if (std.mem.eql(u8, name, "wasm32-wasip2")) return .wasip2;
        if (std.mem.eql(u8, name, "wasm32-wasip3")) return .wasip3;
        return null;
    }
};

pub const Manifest = struct { name: []const u8, version: Version };
pub const Case = struct { name: []const u8, path: []const u8, config: fixture.Config };
pub const Suite = struct { manifest: Manifest, cases: []const Case };

pub const Outcome = enum {
    pass,
    fail,
    skip,
    xfail,
    xpass,

    pub fn evaluate(executed: bool, failed: bool, expected_fail: bool) Outcome {
        if (!executed) return .skip;
        if (expected_fail) return if (failed) .xfail else .xpass;
        return if (failed) .fail else .pass;
    }

    pub fn isFailure(self: Outcome) bool {
        return self == .fail or self == .xpass;
    }
};

pub const Counts = struct {
    total: usize = 0,
    passed: usize = 0,
    failed: usize = 0,
    skipped: usize = 0,
    xfail: usize = 0,
    xpass: usize = 0,

    pub fn add(self: *Counts, outcome: Outcome) void {
        self.total += 1;
        switch (outcome) {
            .pass => self.passed += 1,
            .fail => self.failed += 1,
            .skip => self.skipped += 1,
            .xfail => {
                self.passed += 1;
                self.xfail += 1;
            },
            .xpass => {
                self.failed += 1;
                self.xpass += 1;
            },
        }
    }
};

fn manifest(allocator: Allocator, path: []const u8, contents: ?[]const u8) !Manifest {
    const trimmed = std.mem.trimEnd(u8, path, "/\\");
    const base = std.fs.path.basename(trimmed);
    const path_version = Version.parse(base);
    var result: Manifest = .{
        .name = if (path_version != null) std.fs.path.dirname(trimmed) orelse "." else trimmed,
        .version = path_version orelse .wasip1,
    };
    const source = contents orelse return result;
    const value = try std.json.parseFromSliceLeaky(std.json.Value, allocator, source, .{ .allocate = .alloc_always });
    if (value != .object) return error.InvalidManifest;
    var it = value.object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const item = entry.value_ptr.*;
        if (std.mem.eql(u8, key, "name")) {
            if (item != .string) return error.InvalidManifest;
            result.name = item.string;
        } else if (std.mem.eql(u8, key, "version")) {
            if (item != .string) return error.InvalidManifest;
            result.version = Version.parse(item.string) orelse return error.UnsupportedWasiVersion;
        } else {
            return error.UnknownManifestOption;
        }
    }
    return result;
}

/// The caller owns `allocator` and must retain it for the lifetime of the result.
pub fn discover(allocator: Allocator, io: std.Io, path: []const u8) !Suite {
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    const contents: ?[]const u8 = dir.readFileAlloc(io, "manifest.json", allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    const suite_manifest = try manifest(allocator, path, contents);
    var cases: std.ArrayList(Case) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".wasm")) continue;
        const name = try allocator.dupe(u8, entry.name[0 .. entry.name.len - ".wasm".len]);
        const case_path = try std.fs.path.join(allocator, &.{ path, entry.name });
        const config_name = try std.fmt.allocPrint(allocator, "{s}.json", .{name});
        const config_source: ?[]const u8 = dir.readFileAlloc(io, config_name, allocator, .unlimited) catch |err| switch (err) {
            error.FileNotFound => null,
            else => return err,
        };
        const config = if (config_source) |source|
            try fixture.parse(allocator, source)
        else
            try fixture.defaultConfig(allocator);
        try cases.append(allocator, .{ .name = name, .path = case_path, .config = config });
    }
    std.mem.sort(Case, cases.items, {}, struct {
        fn less(_: void, a: Case, b: Case) bool {
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.less);
    return .{ .manifest = suite_manifest, .cases = try cases.toOwnedSlice(allocator) };
}

test "manifest defaults and strict overrides" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const defaults = try manifest(a, "test/suite/wasm32-wasip3", null);
    try std.testing.expectEqualStrings("test/suite", defaults.name);
    try std.testing.expectEqual(Version.wasip3, defaults.version);
    const override = try manifest(a, "somewhere", "{\"name\":\"custom\",\"version\":\"wasm32-wasip2\"}");
    try std.testing.expectEqualStrings("custom", override.name);
    try std.testing.expectEqual(Version.wasip2, override.version);
    try std.testing.expectError(error.UnknownManifestOption, manifest(a, ".", "{\"unexpected\":1}"));
    try std.testing.expectError(error.UnsupportedWasiVersion, manifest(a, ".", "{\"version\":\"wasm32-unknown\"}"));
    try std.testing.expectError(error.InvalidManifest, manifest(a, ".", "{\"name\":123}"));
}

test "expectation outcomes and counts preserve xfail and xpass semantics" {
    try std.testing.expectEqual(Outcome.skip, Outcome.evaluate(false, true, true));
    try std.testing.expectEqual(Outcome.pass, Outcome.evaluate(true, false, false));
    try std.testing.expectEqual(Outcome.fail, Outcome.evaluate(true, true, false));
    try std.testing.expectEqual(Outcome.xfail, Outcome.evaluate(true, true, true));
    try std.testing.expectEqual(Outcome.xpass, Outcome.evaluate(true, false, true));
    var counts: Counts = .{};
    inline for (.{ Outcome.pass, .fail, .skip, .xfail, .xpass }) |outcome| counts.add(outcome);
    try std.testing.expectEqualDeep(Counts{ .total = 5, .passed = 2, .failed = 2, .skipped = 1, .xfail = 1, .xpass = 1 }, counts);
    try std.testing.expect(Outcome.fail.isFailure());
    try std.testing.expect(Outcome.xpass.isFailure());
    try std.testing.expect(!Outcome.xfail.isFailure());
}

test "discovery loads sidecars and default configurations" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "manifest.json", .data = "{\"name\":\"suite\",\"version\":\"wasm32-wasip3\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "b.wasm", .data = "" });
    try tmp.dir.writeFile(io, .{ .sub_path = "a.wasm", .data = "" });
    try tmp.dir.writeFile(io, .{ .sub_path = "a.json", .data = "{\"stdout\":\"ok\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "ignored.json", .data = "invalid" });
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const suite = try discover(a, io, path);
    try std.testing.expectEqualStrings("suite", suite.manifest.name);
    try std.testing.expectEqual(Version.wasip3, suite.manifest.version);
    try std.testing.expectEqual(@as(usize, 2), suite.cases.len);
    try std.testing.expectEqualStrings("a", suite.cases[0].name);
    try std.testing.expectEqualStrings("ok", suite.cases[0].config.operations[1].read.payload);
    try std.testing.expectEqualStrings("b", suite.cases[1].name);
    try std.testing.expectEqual(@as(usize, 2), suite.cases[1].config.operations.len);
}

test "pinned fixture manifests and unfiltered counts" {
    const io = std.testing.io;
    var upstream = std.Io.Dir.cwd().openDir(io, "tests/wasi-testsuite/test-runner", .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    upstream.close(io);
    const roots = [_]struct { path: []const u8, name: []const u8, version: Version, count: usize }{
        .{ .path = "tests/wasi-testsuite/tests/c/testsuite/wasm32-wasip1", .name = "WASI C tests [wasm32-wasip1]", .version = .wasip1, .count = 14 },
        .{ .path = "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip1", .name = "WASI Rust tests [wasm32-wasip1]", .version = .wasip1, .count = 46 },
        .{ .path = "tests/wasi-testsuite/tests/assemblyscript/testsuite/wasm32-wasip1", .name = "WASI AssemblyScript tests [wasm32-wasip1]", .version = .wasip1, .count = 12 },
        .{ .path = "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip3", .name = "WASI Rust tests [wasm32-wasip3]", .version = .wasip3, .count = 41 },
    };
    for (roots) |root| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const suite = try discover(arena.allocator(), io, root.path);
        try std.testing.expectEqualStrings(root.name, suite.manifest.name);
        try std.testing.expectEqual(root.version, suite.manifest.version);
        try std.testing.expectEqual(root.count, suite.cases.len);
    }
}
