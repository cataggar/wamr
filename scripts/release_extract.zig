//! Extract only members already accepted by the release archive inspector.
const std = @import("std");
const builtin = @import("builtin");
const archive = @import("release_archive.zig");
const binary = @import("release_binary.zig");
const version = @import("release_version.zig");

/// Returns the extracted root path, owned by allocator.
pub fn extract(allocator: std.mem.Allocator, io: std.Io, inspector: *const archive.Inspector, release_version: []const u8, work_dir: []const u8) ![:0]u8 {
    try inspector.finish();
    const checked = try version.releaseVersion(release_version);
    if (!std.mem.eql(u8, checked, release_version)) return error.InvalidReleaseVersion;
    const target = binary.target(inspector.target.name) orelse return error.InvalidReleaseTarget;
    if (target.system != inspector.target.system or target.arch != inspector.target.arch)
        return error.InvalidReleaseTarget;
    const expected_root = try std.fmt.allocPrint(allocator, "wamr-{s}-{s}", .{ release_version, inspector.target.name });
    defer allocator.free(expected_root);
    if (!std.mem.eql(u8, inspector.root, expected_root)) return error.InvalidReleaseIdentity;
    if (builtin.os.tag == .windows and inspector.target.system != .windows)
        return error.UnsupportedExtractionHost;

    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, work_dir);
    const work = try cwd.openDir(io, work_dir, .{});
    defer work.close(io);
    try work.createDir(io, inspector.root, .default_dir);
    const root = try work.openDir(io, inspector.root, .{ .follow_symlinks = false });
    defer root.close(io);
    try root.createDir(io, "bin", .default_dir);
    const bin = try root.openDir(io, "bin", .{ .follow_symlinks = false });
    defer bin.close(io);

    const ext = if (inspector.target.system == .windows) ".exe" else "";
    const wamr = try std.fmt.allocPrint(allocator, "wamr{s}", .{ext});
    defer allocator.free(wamr);
    const wamrc = try std.fmt.allocPrint(allocator, "wamrc{s}", .{ext});
    defer allocator.free(wamrc);
    const names = [_][]const u8{ "LICENSE", "README.md", wamr, wamrc };
    for (inspector.files, names, 0..) |optional, name, index| {
        const member = optional.?;
        const dir = if (index >= 2) bin else root;
        const file = try dir.createFile(io, name, .{ .exclusive = true });
        defer file.close(io);
        try file.writeStreamingAll(io, member.data);
        if (inspector.target.system != .windows) {
            if (builtin.os.tag == .windows) unreachable;
            try file.setPermissions(io, .fromMode(@intCast(member.mode & 0o777)));
        }
    }
    return root.realPathFileAlloc(io, ".", allocator);
}

test "extraction uses validated members and refuses an existing root" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const target = binary.target("linux-x64").?;
    var inspector = try archive.Inspector.init(allocator, "1.2.3", target, "license\n", "readme\n");
    defer inspector.deinit();
    try inspector.add("wamr-1.2.3-linux-x64/LICENSE", .regular, 0o644, "license\n");
    try inspector.add("wamr-1.2.3-linux-x64/README.md", .regular, 0o644, "readme\n");
    var elf: [64]u8 = @splat(0);
    @memcpy(elf[0..6], "\x7fELF\x02\x01");
    std.mem.writeInt(u16, elf[16..18], 2, .little);
    std.mem.writeInt(u16, elf[18..20], 62, .little);
    try inspector.add("wamr-1.2.3-linux-x64/bin/wamr", .regular, 0o751, &elf);
    try inspector.add("wamr-1.2.3-linux-x64/bin/wamrc", .regular, 0o755, &elf);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const extracted = try extract(allocator, io, &inspector, "1.2.3", root);
    defer allocator.free(extracted);
    const file = try std.fs.path.join(allocator, &.{ extracted, "bin", "wamr" });
    defer allocator.free(file);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, file, allocator, .unlimited);
    defer allocator.free(bytes);
    try std.testing.expectEqualSlices(u8, &elf, bytes);
    const stat = try std.Io.Dir.cwd().statFile(io, file, .{});
    try std.testing.expectEqual(@as(u32, 0o751), stat.permissions.toMode() & 0o777);
    try std.testing.expectError(error.PathAlreadyExists, extract(allocator, io, &inspector, "1.2.3", root));
    try std.testing.expectError(error.InvalidReleaseVersion, extract(allocator, io, &inspector, "../bad", root));
}
