//! Selection and command construction for opt-in fixture sampling.
const std = @import("std");
const profile = @import("wasi_fixture_profile.zig");
const Allocator = std.mem.Allocator;

pub const Sampling = struct {
    selection: []const u8,
    output_dir: []const u8,
    mode: []const u8,
    launcher: []const u8,

    /// Selection is read for each operation, as in the Python adapter.
    pub fn command(
        self: Sampling,
        allocator: Allocator,
        io: std.Io,
        fixture_path: []const u8,
        phase: profile.Phase,
        argv: []const []const u8,
    ) ![]const []const u8 {
        const contents = try std.Io.Dir.cwd().readFileAlloc(io, self.selection, allocator, .unlimited);
        defer allocator.free(contents);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, contents, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidProfileSelection;
        const entries = parsed.value.object.get("profiles") orelse return argv;
        if (entries != .array) return error.InvalidProfileSelection;
        const fixture = std.fs.path.stem(fixture_path);
        var selected = false;
        for (entries.array.items) |entry| {
            if (entry != .object) return error.InvalidProfileSelection;
            if (matches(entry.object, "mode", self.mode) and
                matches(entry.object, "fixture", fixture) and
                matches(entry.object, "phase", @tagName(phase)))
            {
                selected = true;
                break;
            }
        }
        if (!selected) return argv;
        if (std.mem.indexOfAny(u8, self.mode, "/\\") != null or self.mode.len == 0)
            return error.InvalidProfileMode;
        const output = try std.fmt.allocPrint(allocator, "{s}-{s}-{s}.sample.txt", .{ fixture, @tagName(phase), self.mode });
        const log = try std.fmt.allocPrint(allocator, "{s}-{s}-{s}.sample.log", .{ fixture, @tagName(phase), self.mode });
        const result = try allocator.alloc([]const u8, argv.len + 6);
        result[0..6].* = .{
            self.launcher,
            "--output",
            try std.fs.path.join(allocator, &.{ self.output_dir, output }),
            "--log",
            try std.fs.path.join(allocator, &.{ self.output_dir, log }),
            "--",
        };
        @memcpy(result[6..], argv);
        return result;
    }
};

fn matches(map: std.json.ObjectMap, name: []const u8, expected: []const u8) bool {
    const value = map.get(name) orelse return false;
    return value == .string and std.mem.eql(u8, value.string, expected);
}

test "selection wraps only the matching fixture, phase, and mode" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "selection.json", .data =
        \\{"profiles":[{"mode":"aot","fixture":"foo","phase":"component_precompile"},
        \\{"mode":"jit","fixture":"foo","phase":"fixture_execution"}]}
    });
    const selection = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/selection.json", .{tmp.sub_path});
    const sampler: Sampling = .{ .selection = selection, .output_dir = "profiles", .mode = "aot", .launcher = "launcher" };
    const command_line: []const []const u8 = &.{ "wamrc", "compile", "foo.wasm" };
    try std.testing.expectEqualDeep(command_line, try sampler.command(allocator, io, "fixtures/foo.wasm", .fixture_execution, command_line));
    try std.testing.expectEqualDeep(command_line, try sampler.command(allocator, io, "fixtures/bar.wasm", .component_precompile, command_line));
    try std.testing.expectEqualDeep(&[_][]const u8{
        "launcher", "--output",                                         "profiles/foo-component_precompile-aot.sample.txt",
        "--log",    "profiles/foo-component_precompile-aot.sample.log", "--",
        "wamrc",    "compile",                                          "foo.wasm",
    }, try sampler.command(allocator, io, "fixtures/foo.wasm", .component_precompile, command_line));
    var jit = sampler;
    jit.mode = "jit";
    const jit_command = try jit.command(allocator, io, "fixtures/foo.wasm", .fixture_execution, command_line);
    try std.testing.expectEqualStrings("profiles/foo-fixture_execution-jit.sample.txt", jit_command[2]);
    try std.testing.expectEqualDeep(command_line, try jit.command(allocator, io, "fixtures/foo.wasm", .component_precompile, command_line));

    try tmp.dir.writeFile(io, .{ .sub_path = "selection.json", .data = "{\"profiles\":[]}" });
    try std.testing.expectEqualDeep(command_line, try sampler.command(allocator, io, "fixtures/foo.wasm", .component_precompile, command_line));
    try tmp.dir.writeFile(io, .{ .sub_path = "selection.json", .data = "{\"profiles\":{}}" });
    try std.testing.expectError(error.InvalidProfileSelection, sampler.command(allocator, io, "fixtures/foo.wasm", .component_precompile, command_line));
}
