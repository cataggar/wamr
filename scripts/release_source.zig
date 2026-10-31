//! Validate the uncompressed Git source archive before accepting release inventory.
const std = @import("std");
const process_control = @import("process_control.zig");
const compression = @import("release_compression.zig");
const tar_format = @import("release_tar.zig");

const block_size = 512;
const source_names = [_][]const u8{ "LICENSE", "README.md", "build.zig" };

/// Input must have been fully decompressed with its XZ integrity checks verified.
/// The three expected contents must come from the checkout being released.
pub fn inspect(tar: []const u8, version: []const u8, source_sha: []const u8, expected: [source_names.len][]const u8) !void {
    var root_buffer: [512]u8 = undefined;
    const root = std.fmt.bufPrint(&root_buffer, "wamr-{s}/", .{version}) catch return error.ReleaseVersionTooLong;
    var iterator = tar_format.Iterator.init(tar);
    var actual: [source_names.len]?[]const u8 = @splat(null);
    var valid_kind: [source_names.len]bool = @splat(false);
    while (try iterator.next()) |entry| {
        if (std.mem.startsWith(u8, entry.name, root)) {
            const relative = entry.name[root.len..];
            for (source_names, 0..) |source, index| {
                if (!std.mem.eql(u8, relative, source)) continue;
                actual[index] = entry.data;
                valid_kind[index] = entry.kind == 0 or entry.kind == '0' or entry.kind == '7';
            }
        }
    }
    const comment = iterator.comment() orelse return error.SourceCommitMismatch;
    if (!std.mem.eql(u8, comment, source_sha)) return error.SourceCommitMismatch;
    for (actual, valid_kind, expected) |optional, is_file, content| {
        const bytes = optional orelse return error.MissingSourceMember;
        if (!is_file or !std.mem.eql(u8, bytes, content)) return error.SourceMemberMismatch;
    }
}

/// XZ is also used to create the release source archive. Require a caller-supplied
/// output limit so a corrupt or hostile archive cannot exhaust unbounded memory.
pub fn inspectXz(allocator: std.mem.Allocator, io: std.Io, path: []const u8, version: []const u8, source_sha: []const u8, expected: [source_names.len][]const u8, max_bytes: usize, timeout_seconds: u32) !void {
    const tar = try compression.decompress(allocator, io, "xz", path, max_bytes, timeout_seconds);
    defer allocator.free(tar);
    try inspect(tar, version, source_sha, expected);
}

fn addTestPax(w: *std.Io.Writer, kind: u8, data: []const u8) !void {
    var header: std.tar.Writer.Header = .init(.regular);
    const raw: *[block_size]u8 = @ptrCast(&header);
    raw[156] = kind;
    try header.setSize(data.len);
    try header.write(w);
    try w.writeAll(data);
    try w.splatByteAll(0, (block_size - data.len % block_size) % block_size);
}

test "source archive requires global commit and exact source members" {
    const allocator = std.testing.allocator;
    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();
    try addTestPax(&output.writer, 'g', "15 comment=abc\n");
    var tar: std.tar.Writer = .{ .underlying_writer = &output.writer };
    try tar.writeFileBytes("wamr-1/LICENSE", "license\n", .{});
    try tar.writeFileBytes("wamr-1/README.md", "readme\n", .{});
    try tar.writeFileBytes("wamr-1/build.zig", "build\n", .{});
    try tar.finishPedantically();
    const contents = [_][]const u8{ "license\n", "readme\n", "build\n" };
    try inspect(output.written(), "1", "abc", contents);
    try std.testing.expectError(error.SourceCommitMismatch, inspect(output.written(), "1", "other", contents));
    try std.testing.expectError(error.SourceMemberMismatch, inspect(output.written(), "1", "abc", .{ "wrong", "readme\n", "build\n" }));
    try std.testing.expectError(error.MissingSourceMember, inspect(output.written(), "2", "abc", contents));
    const bad = try allocator.dupe(u8, output.written());
    defer allocator.free(bad);
    bad[block_size * 2] ^= 1;
    try std.testing.expectError(error.InvalidTarChecksum, inspect(bad, "1", "abc", contents));
}

test "local PAX path overrides and duplicate source members follow archive order" {
    const allocator = std.testing.allocator;
    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();
    try addTestPax(&output.writer, 'g', "15 comment=abc\n");
    var tar: std.tar.Writer = .{ .underlying_writer = &output.writer };
    try tar.writeFileBytes("wamr-1/LICENSE", "wrong", .{});
    try addTestPax(&output.writer, 'x', "23 path=wamr-1/LICENSE\n");
    try tar.writeFileBytes("placeholder", "license", .{});
    try tar.writeFileBytes("wamr-1/README.md", "readme", .{});
    try tar.writeFileBytes("wamr-1/build.zig", "build", .{});
    try tar.finishPedantically();
    try inspect(output.written(), "1", "abc", .{ "license", "readme", "build" });
    try std.testing.expectError(error.SourceMemberMismatch, inspect(output.written(), "1", "abc", .{ "wrong", "readme", "build" }));
    try std.testing.expectError(error.TruncatedTar, inspect(output.written()[0 .. block_size * 3 + 5], "1", "abc", .{ "license", "readme", "build" }));
    try tar.writeDir("wamr-1/LICENSE/", .{});
    // Replace the previous end markers so that the later directory is visible.
    const written = output.written();
    @memcpy(written[written.len - block_size * 3 ..][0..block_size], written[written.len - block_size ..]);
    try std.testing.expectError(error.SourceMemberMismatch, inspect(written[0 .. written.len - block_size * 2], "1", "abc", .{ "license", "readme", "build" }));
}

test "XZ source archive integrity and output limits are enforced" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();
    try addTestPax(&output.writer, 'g', "15 comment=abc\n");
    var tar: std.tar.Writer = .{ .underlying_writer = &output.writer };
    try tar.writeFileBytes("wamr-1/LICENSE", "license", .{});
    try tar.writeFileBytes("wamr-1/README.md", "readme", .{});
    try tar.writeFileBytes("wamr-1/build.zig", "build", .{});
    try tar.finishPedantically();
    try temp.dir.writeFile(io, .{ .sub_path = "source.tar", .data = output.written() });
    const tar_path = try std.fs.path.join(allocator, &.{ root, "source.tar" });
    defer allocator.free(tar_path);
    const xz_path = try std.fs.path.join(allocator, &.{ root, "source.tar.xz" });
    defer allocator.free(xz_path);
    const compressed = try process_control.runTimed(allocator, io, .{ .argv = &.{ "xz", "--keep", "--", tar_path } }, 30);
    defer allocator.free(compressed.stdout);
    defer allocator.free(compressed.stderr);
    try std.testing.expect(compressed.term.success());
    const expected = [_][]const u8{ "license", "readme", "build" };
    try inspectXz(allocator, io, xz_path, "1", "abc", expected, 1024 * 1024, 30);
    try std.testing.expectError(error.StreamTooLong, inspectXz(allocator, io, xz_path, "1", "abc", expected, 100, 30));
    const bytes = try temp.dir.readFileAlloc(io, "source.tar.xz", allocator, .unlimited);
    defer allocator.free(bytes);
    bytes[bytes.len - 5] ^= 1;
    try temp.dir.writeFile(io, .{ .sub_path = "corrupt.tar.xz", .data = bytes });
    const corrupt = try std.fs.path.join(allocator, &.{ root, "corrupt.tar.xz" });
    defer allocator.free(corrupt);
    try std.testing.expectError(error.DecompressionFailed, inspectXz(allocator, io, corrupt, "1", "abc", expected, 1024 * 1024, 30));
}
