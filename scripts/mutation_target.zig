//! Atomic mutation and restoration of one source file.
const std = @import("std");
const candidates = @import("mutation_candidates.zig");
const Mutant = candidates.Mutant;
const Allocator = std.mem.Allocator;

pub const Target = struct {
    allocator: Allocator,
    io: std.Io,
    path: [:0]const u8,
    original: []const u8,
    source: []const u8,
    permissions: std.Io.File.Permissions,
    active: ?[]const u8 = null,

    pub fn open(allocator: Allocator, io: std.Io, requested: []const u8) !Target {
        const path = try std.Io.Dir.cwd().realPathFileAlloc(io, requested, allocator);
        errdefer allocator.free(path);
        const original = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        errdefer allocator.free(original);
        const permissions = (try std.Io.Dir.cwd().statFile(io, path, .{})).permissions;
        const source = try candidates.normalizeNewlines(allocator, original);
        return .{ .allocator = allocator, .io = io, .path = path, .original = original, .source = source, .permissions = permissions };
    }

    fn assertContents(self: *Target, expected: []const u8) !void {
        const current = try std.Io.Dir.cwd().readFileAlloc(self.io, self.path, self.allocator, .unlimited);
        defer self.allocator.free(current);
        if (!std.mem.eql(u8, current, expected)) return error.TargetChanged;
        if ((try std.Io.Dir.cwd().statFile(self.io, self.path, .{})).permissions != self.permissions)
            return error.TargetChanged;
    }

    fn writeAtomically(self: *Target, bytes: []const u8) !void {
        const parent = std.fs.path.dirname(self.path) orelse return error.InvalidTargetPath;
        const basename = std.fs.path.basename(self.path);
        const dir = try std.Io.Dir.cwd().openDir(self.io, parent, .{});
        defer dir.close(self.io);
        for (0..10) |_| {
            var random: [16]u8 = undefined;
            self.io.random(&random);
            const name = try std.fmt.allocPrint(self.allocator, ".{s}.mutation-{s}.tmp", .{ basename, std.fmt.bytesToHex(random, .lower) });
            defer self.allocator.free(name);
            const file = dir.createFile(self.io, name, .{ .exclusive = true, .permissions = self.permissions }) catch |err| switch (err) {
                error.PathAlreadyExists => continue,
                else => return err,
            };
            errdefer dir.deleteFile(self.io, name) catch |err| {
                std.debug.print("mutation target: cannot remove temporary file {s}: {s}\n", .{ name, @errorName(err) });
            };
            {
                defer file.close(self.io);
                try file.writeStreamingAll(self.io, bytes);
                try file.setPermissions(self.io, self.permissions);
            }
            try dir.rename(name, dir, basename, self.io);
            return;
        }
        return error.TemporaryNameCollisions;
    }

    pub fn apply(self: *Target, mutant: Mutant) !void {
        if (self.active != null) return error.MutationAlreadyApplied;
        var start: usize = 0;
        var line_no: usize = 1;
        while (start < self.source.len) : (line_no += 1) {
            const end = try candidates.nextLineEnd(self.source, start);
            if (line_no == mutant.line_no) {
                if (!std.mem.eql(u8, std.mem.trimEnd(u8, self.source[start..end], "\n"), mutant.original_line))
                    return error.StaleMutant;
                const changed = try std.fmt.allocPrint(self.allocator, "{s}{s}{s}", .{ self.source[0..start], mutant.mutated_line, self.source[end..] });
                errdefer self.allocator.free(changed);
                try self.assertContents(self.original);
                try self.writeAtomically(changed);
                self.active = changed;
                return;
            }
            start = end;
        }
        return error.StaleMutant;
    }

    pub fn restore(self: *Target) !void {
        const changed = self.active orelse return;
        try self.assertContents(changed);
        try self.writeAtomically(self.original);
        self.allocator.free(changed);
        self.active = null;
    }

    /// If restoration fails, the target remains open so the caller can retry.
    pub fn close(self: *Target) !void {
        try self.restore();
        self.allocator.free(self.source);
        self.allocator.free(self.original);
        self.allocator.free(self.path);
        self.* = undefined;
    }
};

test "atomic mutation restores the original and detects external edits" {
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try temp.dir.writeFile(io, .{ .sub_path = "test.zig", .data = "first\r\nx = 0;\nlast" });
    const path = try temp.dir.realPathFileAlloc(io, "test.zig", allocator);
    defer allocator.free(path);
    var target = try Target.open(allocator, io, path);
    const mutant: Mutant = .{ .line_no = 2, .original_line = "x = 0;", .mutated_line = "x = 1;\n", .mutator = "", .detail = "" };
    try target.apply(mutant);
    try std.testing.expectError(error.MutationAlreadyApplied, target.apply(mutant));
    const changed = try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited);
    defer allocator.free(changed);
    try std.testing.expectEqualStrings("first\nx = 1;\nlast", changed);
    try temp.dir.writeFile(io, .{ .sub_path = "test.zig", .data = "external edit" });
    try std.testing.expectError(error.TargetChanged, target.restore());
    const external = try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited);
    defer allocator.free(external);
    try std.testing.expectEqualStrings("external edit", external);
    try temp.dir.writeFile(io, .{ .sub_path = "test.zig", .data = changed });
    try target.restore();
    const restored = try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited);
    defer allocator.free(restored);
    try std.testing.expectEqualStrings("first\r\nx = 0;\nlast", restored);
    try target.close();
}

test "stale or externally changed sources are never overwritten" {
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try temp.dir.writeFile(io, .{ .sub_path = "test.zig", .data = "x = 0;\n" });
    const path = try temp.dir.realPathFileAlloc(io, "test.zig", allocator);
    defer allocator.free(path);
    var target = try Target.open(allocator, io, path);
    defer target.close() catch unreachable;
    const bad: Mutant = .{ .line_no = 1, .original_line = "different", .mutated_line = "x = 1;\n", .mutator = "", .detail = "" };
    try std.testing.expectError(error.StaleMutant, target.apply(bad));
    try temp.dir.writeFile(io, .{ .sub_path = "test.zig", .data = "user changes\n" });
    const mutant: Mutant = .{ .line_no = 1, .original_line = "x = 0;", .mutated_line = "x = 1;\n", .mutator = "", .detail = "" };
    try std.testing.expectError(error.TargetChanged, target.apply(mutant));
}

test "mutants after Unicode line separators restore original bytes" {
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    const original = "first\xe2\x80\xa8x = 0;\r\n";
    try temp.dir.writeFile(io, .{ .sub_path = "test.zig", .data = original });
    const path = try temp.dir.realPathFileAlloc(io, "test.zig", allocator);
    var target = try Target.open(allocator, io, path);
    defer target.close() catch unreachable;
    const found = try candidates.collect(allocator, target.source, &.{});
    try std.testing.expectEqual(@as(usize, 1), found.len);
    try std.testing.expectEqual(@as(usize, 2), found[0].line_no);
    try target.apply(found[0]);
    try std.testing.expectEqualStrings("first\xe2\x80\xa8x = 1;\n", try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited));
    try target.restore();
    try std.testing.expectEqualStrings(original, try temp.dir.readFileAlloc(io, "test.zig", allocator, .unlimited));
}

test "mutation preserves permissions and detects external mode changes" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try temp.dir.writeFile(io, .{ .sub_path = "test.zig", .data = "x = 0;\n" });
    const file = try temp.dir.openFile(io, "test.zig", .{});
    try file.setPermissions(io, .fromMode(0o751));
    file.close(io);
    const path = try temp.dir.realPathFileAlloc(io, "test.zig", allocator);
    defer allocator.free(path);
    var target = try Target.open(allocator, io, path);
    const mutant: Mutant = .{ .line_no = 1, .original_line = "x = 0;", .mutated_line = "x = 1;\n", .mutator = "", .detail = "" };
    try target.apply(mutant);
    try std.testing.expectEqual(@as(std.posix.mode_t, 0o751), (try temp.dir.statFile(io, "test.zig", .{})).permissions.toMode() & 0o7777);
    const changed = try temp.dir.openFile(io, "test.zig", .{});
    try changed.setPermissions(io, .fromMode(0o700));
    try std.testing.expectError(error.TargetChanged, target.restore());
    try changed.setPermissions(io, .fromMode(0o751));
    changed.close(io);
    try target.restore();
    try std.testing.expectEqual(@as(std.posix.mode_t, 0o751), (try temp.dir.statFile(io, "test.zig", .{})).permissions.toMode() & 0o7777);
    try target.close();
}
