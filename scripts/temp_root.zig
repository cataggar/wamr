//! Select a writable temporary directory for automation tools.
const std = @import("std");
const Allocator = std.mem.Allocator;

fn usable(io: std.Io, path: []const u8) !bool {
    if (path.len == 0) return false;
    const dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound, error.NotDir, error.AccessDenied, error.PermissionDenied, error.SymLinkLoop, error.NetworkNotFound => return false,
        else => return err,
    };
    defer dir.close(io);
    for (0..10) |_| {
        var random: [16]u8 = undefined;
        io.random(&random);
        const hex = std.fmt.bytesToHex(random, .lower);
        var buffer: [48]u8 = undefined;
        const name = try std.fmt.bufPrint(&buffer, ".temp-probe-{s}", .{hex});
        const file = dir.createFile(io, name, .{
            .exclusive = true,
            .permissions = if (@import("builtin").os.tag == .windows) .default_file else .fromMode(0o600),
        }) catch |err| switch (err) {
            error.PathAlreadyExists => continue,
            error.FileNotFound, error.NotDir, error.AccessDenied, error.PermissionDenied, error.NoSpaceLeft, error.ReadOnlyFileSystem => return false,
            else => return err,
        };
        const write_result = file.writeStreamingAll(io, "x");
        file.close(io);
        try dir.deleteFile(io, name);
        write_result catch |err| switch (err) {
            error.NoSpaceLeft, error.DiskQuota, error.AccessDenied => return false,
            else => return err,
        };
        return true;
    }
    return error.TemporaryNameCollisions;
}

/// Select a writable temporary directory in the same order as Python tempfile.
pub fn select(allocator: Allocator, io: std.Io, env: *const std.process.Environ.Map) ![]const u8 {
    var cwd_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_len = try std.Io.Dir.cwd().realPath(io, &cwd_buffer);
    const cwd = cwd_buffer[0..cwd_len];
    for ([_][]const u8{ "TMPDIR", "TEMP", "TMP" }) |key| {
        if (env.get(key)) |candidate| {
            if (try usable(io, candidate))
                return std.fs.path.resolve(allocator, &.{ cwd, candidate });
        }
    }
    const defaults: []const []const u8 = if (@import("builtin").os.tag == .windows)
        &.{ "C:\\TEMP", "C:\\TMP", "\\TEMP", "\\TMP", cwd }
    else
        &.{ "/tmp", "/var/tmp", "/usr/tmp", cwd };
    for (defaults) |candidate| {
        if (try usable(io, candidate))
            return std.fs.path.resolve(allocator, &.{ cwd, candidate });
    }
    return error.NoUsableTemporaryDirectory;
}

test "temporary selection skips empty or missing environment paths" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buffer: [std.fs.max_path_bytes]u8 = undefined;
    const length = try tmp.dir.realPath(io, &buffer);
    var environment = std.process.Environ.Map.init(allocator);
    defer environment.deinit();
    try environment.put("TMPDIR", "");
    try environment.put("TEMP", "a-missing-wasi-temp-directory");
    try environment.put("TMP", buffer[0..length]);
    const chosen = try select(allocator, io, &environment);
    defer allocator.free(chosen);
    try std.testing.expectEqualStrings(buffer[0..length], chosen);
    const directory = try tmp.dir.openDir(io, ".", .{ .iterate = true });
    defer directory.close(io);
    var iter = directory.iterate();
    try std.testing.expectEqual(@as(?std.Io.Dir.Entry, null), try iter.next(io));
}
