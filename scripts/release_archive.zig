//! Validation of members in a release binary archive.
const std = @import("std");
const binary = @import("release_binary.zig");
const tar_format = @import("release_tar.zig");
const compression = @import("release_compression.zig");
const Allocator = std.mem.Allocator;

pub const Kind = enum { directory, regular, other };
pub const Member = struct {
    name: []const u8,
    data: []const u8,
    mode: u32,
};

/// Archive readers must pass every member, including directories and links.
pub const Inspector = struct {
    allocator: Allocator,
    target: binary.Target,
    root: []const u8,
    license: []const u8,
    readme: []const u8,
    seen: std.StringHashMap(void),
    files: [4]?Member = @splat(null),

    pub fn init(allocator: Allocator, version: []const u8, target: binary.Target, license: []const u8, readme: []const u8) !Inspector {
        return .{
            .allocator = allocator,
            .target = target,
            .root = try std.fmt.allocPrint(allocator, "wamr-{s}-{s}", .{ version, target.name }),
            .license = license,
            .readme = readme,
            .seen = std.StringHashMap(void).init(allocator),
        };
    }

    pub fn deinit(self: *Inspector) void {
        var keys = self.seen.keyIterator();
        while (keys.next()) |key| self.allocator.free(key.*);
        self.seen.deinit();
        for (self.files) |member| if (member) |value| self.allocator.free(value.data);
        self.allocator.free(self.root);
        self.* = undefined;
    }

    pub fn add(self: *Inspector, raw_name: []const u8, kind: Kind, mode: u32, contents: []const u8) !void {
        const normalized = try self.allocator.dupe(u8, raw_name);
        defer self.allocator.free(normalized);
        if (self.target.system == .windows) {
            for (normalized) |*byte| if (byte.* == '\\') {
                byte.* = '/';
            };
        }
        const key = try self.allocator.dupe(u8, if (kind == .directory) std.mem.trimEnd(u8, normalized, "/") else normalized);
        var owned = false;
        errdefer if (!owned) self.allocator.free(key);
        if (self.seen.contains(key)) return error.DuplicateArchiveMember;
        try self.seen.put(key, {});
        owned = true;

        if (kind == .directory and std.mem.eql(u8, key, self.root)) return;
        const suffix = if (std.mem.startsWith(u8, key, self.root) and key.len > self.root.len and key[self.root.len] == '/')
            key[self.root.len + 1 ..]
        else
            return error.UnexpectedArchiveMember;
        if (kind == .directory and std.mem.eql(u8, suffix, "bin")) return;
        if (kind != .regular) return error.UnexpectedArchiveMember;
        const i: usize = if (std.mem.eql(u8, suffix, "LICENSE"))
            0
        else if (std.mem.eql(u8, suffix, "README.md"))
            1
        else if (std.mem.eql(u8, suffix, if (self.target.system == .windows) "bin/wamr.exe" else "bin/wamr"))
            2
        else if (std.mem.eql(u8, suffix, if (self.target.system == .windows) "bin/wamrc.exe" else "bin/wamrc"))
            3
        else
            return error.UnexpectedArchiveMember;
        if (contents.len == 0) return error.EmptyArchiveMember;
        if (i >= 2) {
            try binary.checkBinary(contents, self.target.system, self.target.arch);
            if (self.target.system != .windows and mode & 0o111 == 0) return error.NonExecutableArchiveMember;
        } else if (!equalNormalized(contents, if (i == 0) self.license else self.readme)) {
            return error.ArchiveDocumentationMismatch;
        }
        self.files[i] = .{ .name = key, .data = try self.allocator.dupe(u8, contents), .mode = mode };
    }

    pub fn finish(self: *const Inspector) !void {
        for (self.files) |member| if (member == null) return error.MissingArchiveMember;
    }
};

fn equalNormalized(left: []const u8, right: []const u8) bool {
    var i: usize = 0;
    var j: usize = 0;
    while (i < left.len and j < right.len) {
        if (left[i] == '\r' and i + 1 < left.len and left[i + 1] == '\n') i += 1;
        if (right[j] == '\r' and j + 1 < right.len and right[j + 1] == '\n') j += 1;
        if (left[i] != right[j]) return false;
        i += 1;
        j += 1;
    }
    return i == left.len and j == right.len;
}

/// Inspect every member; callers must discard the inspector after an error.
pub fn inspectTar(inspector: *Inspector, tar: []const u8) !void {
    if (inspector.target.system == .windows) return error.InvalidArchiveFormat;
    var iterator = tar_format.Iterator.init(tar);
    while (try iterator.next()) |entry| {
        const kind: Kind = switch (entry.kind) {
            0, '0', '7' => .regular,
            '5' => .directory,
            else => .other,
        };
        try inspector.add(entry.name, kind, entry.mode, entry.data);
    }
    try inspector.finish();
}

/// Caller provides a maximum expanded size and a decompression deadline.
pub fn inspectGzip(allocator: Allocator, io: std.Io, inspector: *Inspector, path: []const u8, max_bytes: usize, timeout_seconds: u32) !void {
    const tar = try compression.decompress(allocator, io, "gzip", path, max_bytes, timeout_seconds);
    defer allocator.free(tar);
    try inspectTar(inspector, tar);
}

/// Inspect ZIP entries without extracting archive-controlled paths to disk.
pub fn inspectZip(allocator: Allocator, io: std.Io, inspector: *Inspector, path: []const u8, max_bytes: usize) !void {
    if (inspector.target.system != .windows) return error.InvalidArchiveFormat;
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var file_buffer: [8192]u8 = undefined;
    var reader = file.reader(io, &file_buffer);
    var iterator = try std.zip.Iterator.init(&reader);
    const file_size = try reader.getSize();
    if (iterator.cd_zip_offset > file_size or iterator.cd_size > file_size - iterator.cd_zip_offset or
        iterator.cd_record_count > iterator.cd_size / @sizeOf(std.zip.CentralDirectoryFileHeader))
        return error.InvalidZipDirectory;
    var total: usize = 0;
    while (iterator.cd_record_index < iterator.cd_record_count) {
        if (iterator.cd_record_offset > iterator.cd_size or
            iterator.cd_size - iterator.cd_record_offset < @sizeOf(std.zip.CentralDirectoryFileHeader))
            return error.InvalidZipDirectory;
        const entry = (try iterator.next()) orelse return error.InvalidZipDirectory;
        if (iterator.cd_record_offset > iterator.cd_size) return error.InvalidZipDirectory;
        try reader.seekTo(entry.header_zip_offset);
        const central = try reader.interface.takeStruct(std.zip.CentralDirectoryFileHeader, .little);
        const name = try allocator.alloc(u8, entry.filename_len);
        defer allocator.free(name);
        try reader.interface.readSliceAll(name);
        if (name.len == 0 or std.mem.indexOfScalar(u8, name, 0) != null) return error.InvalidZipName;
        try reader.seekTo(entry.file_offset);
        const local = try reader.interface.takeStruct(std.zip.LocalFileHeader, .little);
        if (!std.mem.eql(u8, &local.signature, &std.zip.local_file_header_sig) or
            local.filename_len != name.len or local.compression_method != entry.compression_method)
            return error.InvalidZipLocalHeader;
        const local_name = try allocator.alloc(u8, local.filename_len);
        defer allocator.free(local_name);
        try reader.interface.readSliceAll(local_name);
        if (!std.mem.eql(u8, name, local_name)) return error.InvalidZipLocalHeader;
        const offset = try std.math.add(u64, entry.file_offset, @sizeOf(std.zip.LocalFileHeader) + @as(u64, local.filename_len) + local.extra_len);
        const end = try std.math.add(u64, offset, entry.compressed_size);
        if (end > iterator.cd_zip_offset) return error.InvalidZipLocalHeader;
        const mode = central.external_file_attributes >> 16;
        const typ = mode & 0o170000;
        const directory = name[name.len - 1] == '/' or name[name.len - 1] == '\\';
        const kind: Kind = if (directory and (typ == 0 or typ == 0o040000))
            .directory
        else if (!directory and (typ == 0 or typ == 0o100000))
            .regular
        else
            .other;
        if (kind != .regular) {
            try inspector.add(name, kind, mode, "");
            continue;
        }
        const size = std.math.cast(usize, entry.uncompressed_size) orelse return error.StreamTooLong;
        total = std.math.add(usize, total, size) catch return error.StreamTooLong;
        if (total > max_bytes) return error.StreamTooLong;
        const compressed_size = std.math.cast(usize, entry.compressed_size) orelse return error.StreamTooLong;
        if (compressed_size > max_bytes +| (1024 * 1024)) return error.StreamTooLong;
        const compressed = try allocator.alloc(u8, compressed_size);
        defer allocator.free(compressed);
        try reader.seekTo(offset);
        try reader.interface.readSliceAll(compressed);
        const data = switch (entry.compression_method) {
            .store => try allocator.dupe(u8, compressed),
            .deflate => blk: {
                var input: std.Io.Reader = .fixed(compressed);
                var window: [std.compress.flate.max_window_len]u8 = undefined;
                var decompressor: std.compress.flate.Decompress = .init(&input, .raw, &window);
                const decoded = decompressor.reader.allocRemaining(allocator, .limited(size)) catch |err| switch (err) {
                    error.ReadFailed => return decompressor.err orelse err,
                    else => return err,
                };
                errdefer allocator.free(decoded);
                if (input.seek != compressed.len)
                    return error.InvalidZipCompressedData;
                break :blk decoded;
            },
            else => return error.UnsupportedCompressionMethod,
        };
        defer allocator.free(data);
        if (data.len != size or std.hash.Crc32.hash(data) != entry.crc32) return error.InvalidZipChecksum;
        var output: std.Io.Writer.Allocating = .init(allocator);
        defer output.deinit();
        try entry.extractTo(&reader, &output.writer);
        if (!std.mem.eql(u8, data, output.written())) return error.InvalidZipCompressedData;
        try inspector.add(name, kind, mode, data);
    }
    if (try iterator.next() != null) return error.InvalidZipDirectory;
    try inspector.finish();
}

test "archive member validation enforces structure, contents, and executable identity" {
    const allocator = std.testing.allocator;
    var inspector = try Inspector.init(allocator, "1.2.3", binary.target("linux-x64").?, "license\n", "readme\n");
    defer inspector.deinit();
    try inspector.add("wamr-1.2.3-linux-x64/", .directory, 0, "");
    try inspector.add("wamr-1.2.3-linux-x64/bin", .directory, 0, "");
    try inspector.add("wamr-1.2.3-linux-x64/LICENSE", .regular, 0o644, "license\r\n");
    try inspector.add("wamr-1.2.3-linux-x64/README.md", .regular, 0o644, "readme\n");
    var elf: [64]u8 = @splat(0);
    @memcpy(elf[0..6], "\x7fELF\x02\x01");
    std.mem.writeInt(u16, elf[16..18], 3, .little);
    std.mem.writeInt(u16, elf[18..20], 62, .little);
    try inspector.add("wamr-1.2.3-linux-x64/bin/wamr", .regular, 0o755, &elf);
    try std.testing.expectError(error.MissingArchiveMember, inspector.finish());
    try inspector.add("wamr-1.2.3-linux-x64/bin/wamrc", .regular, 0o755, &elf);
    try inspector.finish();
    try std.testing.expectError(error.DuplicateArchiveMember, inspector.add("wamr-1.2.3-linux-x64/LICENSE", .regular, 0, "license\n"));
    try std.testing.expectError(error.UnexpectedArchiveMember, inspector.add("../outside", .regular, 0, "x"));
}

test "windows names normalize separators and reject links and collisions" {
    const allocator = std.testing.allocator;
    var inspector = try Inspector.init(allocator, "1.2.3", binary.target("windows-x64").?, "license\n", "readme\n");
    defer inspector.deinit();
    try inspector.add("wamr-1.2.3-windows-x64\\", .directory, 0, "");
    try inspector.add("wamr-1.2.3-windows-x64\\LICENSE", .regular, 0, "license\r\n");
    try std.testing.expectError(error.DuplicateArchiveMember, inspector.add("wamr-1.2.3-windows-x64/LICENSE", .regular, 0, "license\n"));
    try std.testing.expectError(error.UnexpectedArchiveMember, inspector.add("wamr-1.2.3-windows-x64\\bin\\", .other, 0, ""));
    try std.testing.expectError(error.UnexpectedArchiveMember, inspector.add("C:\\outside", .regular, 0, "x"));
}

test "reject invalid documentation, empty files, wrong binaries and modes" {
    const allocator = std.testing.allocator;
    var inspector = try Inspector.init(allocator, "1", binary.target("linux-x64").?, "license\n", "readme\n");
    defer inspector.deinit();
    try std.testing.expectError(error.ArchiveDocumentationMismatch, inspector.add("wamr-1-linux-x64/LICENSE", .regular, 0, "other\n"));
    try std.testing.expectError(error.EmptyArchiveMember, inspector.add("wamr-1-linux-x64/README.md", .regular, 0, ""));
    try std.testing.expectError(error.InvalidExecutableIdentity, inspector.add("wamr-1-linux-x64/bin/wamr", .regular, 0o755, "not an executable"));
    var elf: [64]u8 = @splat(0);
    @memcpy(elf[0..6], "\x7fELF\x02\x01");
    std.mem.writeInt(u16, elf[16..18], 2, .little);
    std.mem.writeInt(u16, elf[18..20], 62, .little);
    try std.testing.expectError(error.NonExecutableArchiveMember, inspector.add("wamr-1-linux-x64/bin/wamrc", .regular, 0o644, &elf));
}

fn testTar(allocator: Allocator) ![]u8 {
    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();
    var tar: std.tar.Writer = .{ .underlying_writer = &output.writer };
    try tar.writeDir("wamr-1-linux-x64", .{});
    try tar.writeDir("wamr-1-linux-x64/bin", .{});
    try tar.writeFileBytes("wamr-1-linux-x64/LICENSE", "license\n", .{});
    try tar.writeFileBytes("wamr-1-linux-x64/README.md", "readme\n", .{});
    var elf: [64]u8 = @splat(0);
    @memcpy(elf[0..6], "\x7fELF\x02\x01");
    std.mem.writeInt(u16, elf[16..18], 2, .little);
    std.mem.writeInt(u16, elf[18..20], 62, .little);
    try tar.writeFileBytes("wamr-1-linux-x64/bin/wamr", &elf, .{ .mode = 0o755 });
    try tar.writeFileBytes("wamr-1-linux-x64/bin/wamrc", &elf, .{ .mode = 0o755 });
    try tar.finishPedantically();
    return allocator.dupe(u8, output.written());
}

test "TAR inspection validates every member and rejects corrupted headers" {
    const allocator = std.testing.allocator;
    const tar = try testTar(allocator);
    defer allocator.free(tar);
    var inspector = try Inspector.init(allocator, "1", binary.target("linux-x64").?, "license\n", "readme\n");
    defer inspector.deinit();
    try inspectTar(&inspector, tar);
    try std.testing.expectEqualStrings("license\n", inspector.files[0].?.data);

    const bad = try allocator.dupe(u8, tar);
    defer allocator.free(bad);
    bad[0] ^= 1;
    var invalid = try Inspector.init(allocator, "1", binary.target("linux-x64").?, "license\n", "readme\n");
    defer invalid.deinit();
    try std.testing.expectError(error.InvalidTarChecksum, inspectTar(&invalid, bad));
}

test "TAR inspection rejects links and unexpected members" {
    const allocator = std.testing.allocator;
    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();
    var tar: std.tar.Writer = .{ .underlying_writer = &output.writer };
    try tar.writeLink("wamr-1-linux-x64/bin/wamr", "/outside", .{});
    var inspector = try Inspector.init(allocator, "1", binary.target("linux-x64").?, "license\n", "readme\n");
    defer inspector.deinit();
    try std.testing.expectError(error.UnexpectedArchiveMember, inspectTar(&inspector, output.written()));
}

test "gzip inspection validates the compressed stream before accepting members" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const tar = try testTar(allocator);
    defer allocator.free(tar);
    try temp.dir.writeFile(io, .{ .sub_path = "release.tar", .data = tar });
    const tar_path = try std.fs.path.join(allocator, &.{ root, "release.tar" });
    defer allocator.free(tar_path);
    const gzip_path = try std.fs.path.join(allocator, &.{ root, "release.tar.gz" });
    defer allocator.free(gzip_path);
    const result = try @import("process_control.zig").runTimed(allocator, io, .{ .argv = &.{ "gzip", "--keep", "--", tar_path } }, 30);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try std.testing.expect(result.term.success());
    var inspector = try Inspector.init(allocator, "1", binary.target("linux-x64").?, "license\n", "readme\n");
    defer inspector.deinit();
    try inspectGzip(allocator, io, &inspector, gzip_path, 1024 * 1024, 30);

    const bytes = try temp.dir.readFileAlloc(io, "release.tar.gz", allocator, .unlimited);
    defer allocator.free(bytes);
    bytes[bytes.len - 8] ^= 1;
    try temp.dir.writeFile(io, .{ .sub_path = "corrupt.gz", .data = bytes });
    const corrupt = try std.fs.path.join(allocator, &.{ root, "corrupt.gz" });
    defer allocator.free(corrupt);
    var invalid = try Inspector.init(allocator, "1", binary.target("linux-x64").?, "license\n", "readme\n");
    defer invalid.deinit();
    try std.testing.expectError(error.DecompressionFailed, inspectGzip(allocator, io, &invalid, corrupt, 1024 * 1024, 30));
}

pub const ZipFixtureEntry = struct { name: []const u8, data: []const u8, mode: u32, compressed: ?[]const u8 = null };

pub fn testZip(allocator: Allocator, entries: []const ZipFixtureEntry) ![]u8 {
    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();
    const w = &output.writer;
    const offsets = try allocator.alloc(u32, entries.len);
    defer allocator.free(offsets);
    for (entries, 0..) |entry, index| {
        offsets[index] = @intCast(output.written().len);
        const crc = std.hash.Crc32.hash(entry.data);
        const compressed = entry.compressed orelse entry.data;
        try w.writeAll("PK\x03\x04");
        try w.writeInt(u16, 20, .little);
        try w.writeInt(u16, 0, .little);
        try w.writeInt(u16, if (entry.compressed == null) 0 else 8, .little);
        try w.writeInt(u16, 0, .little);
        try w.writeInt(u16, 0, .little);
        try w.writeInt(u32, crc, .little);
        try w.writeInt(u32, @intCast(compressed.len), .little);
        try w.writeInt(u32, @intCast(entry.data.len), .little);
        try w.writeInt(u16, @intCast(entry.name.len), .little);
        try w.writeInt(u16, 0, .little);
        try w.writeAll(entry.name);
        try w.writeAll(compressed);
    }
    const cd_start: u32 = @intCast(output.written().len);
    for (entries, offsets) |entry, offset| {
        const compressed = entry.compressed orelse entry.data;
        try w.writeAll("PK\x01\x02");
        try w.writeInt(u16, 0x0314, .little);
        try w.writeInt(u16, 20, .little);
        try w.writeInt(u16, 0, .little);
        try w.writeInt(u16, if (entry.compressed == null) 0 else 8, .little);
        try w.writeInt(u16, 0, .little);
        try w.writeInt(u16, 0, .little);
        try w.writeInt(u32, std.hash.Crc32.hash(entry.data), .little);
        try w.writeInt(u32, @intCast(compressed.len), .little);
        try w.writeInt(u32, @intCast(entry.data.len), .little);
        try w.writeInt(u16, @intCast(entry.name.len), .little);
        try w.writeInt(u16, 0, .little);
        try w.writeInt(u16, 0, .little);
        try w.writeInt(u16, 0, .little);
        try w.writeInt(u16, 0, .little);
        try w.writeInt(u32, entry.mode << 16, .little);
        try w.writeInt(u32, offset, .little);
        try w.writeAll(entry.name);
    }
    const cd_size: u32 = @as(u32, @intCast(output.written().len)) - cd_start;
    try w.writeAll("PK\x05\x06");
    try w.writeInt(u16, 0, .little);
    try w.writeInt(u16, 0, .little);
    try w.writeInt(u16, @intCast(entries.len), .little);
    try w.writeInt(u16, @intCast(entries.len), .little);
    try w.writeInt(u32, cd_size, .little);
    try w.writeInt(u32, cd_start, .little);
    try w.writeInt(u16, 0, .little);
    return allocator.dupe(u8, output.written());
}

test "ZIP inspection validates names, modes, checksums, and expanded size" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const path = try std.fs.path.join(allocator, &.{ root, "release.zip" });
    defer allocator.free(path);
    var pe: [128]u8 = @splat(0);
    @memcpy(pe[0..2], "MZ");
    std.mem.writeInt(u32, pe[60..64], 64, .little);
    @memcpy(pe[64..68], "PE\x00\x00");
    std.mem.writeInt(u16, pe[68..70], 0x8664, .little);
    std.mem.writeInt(u16, pe[88..90], 0x20b, .little);
    var entries = [_]ZipFixtureEntry{
        .{ .name = "wamr-1-windows-x64\\LICENSE", .data = "license\n", .mode = 0o100644, .compressed = "\xcb\xc9\x4c\x4e\xcd\x2b\x4e\xe5\x02\x00" },
        .{ .name = "wamr-1-windows-x64\\README.md", .data = "readme\n", .mode = 0o100644 },
        .{ .name = "wamr-1-windows-x64\\bin\\wamr.exe", .data = &pe, .mode = 0o100755 },
        .{ .name = "wamr-1-windows-x64\\bin\\wamrc.exe", .data = &pe, .mode = 0o100755 },
    };
    const zip = try testZip(allocator, &entries);
    defer allocator.free(zip);
    try temp.dir.writeFile(io, .{ .sub_path = "release.zip", .data = zip });
    var inspector = try Inspector.init(allocator, "1", binary.target("windows-x64").?, "license\n", "readme\n");
    defer inspector.deinit();
    try inspectZip(allocator, io, &inspector, path, 1024);
    try std.testing.expectEqualStrings("license\n", inspector.files[0].?.data);

    var limited = try Inspector.init(allocator, "1", binary.target("windows-x64").?, "license\n", "readme\n");
    defer limited.deinit();
    try std.testing.expectError(error.StreamTooLong, inspectZip(allocator, io, &limited, path, 1));
    const readme_data = 30 + entries[0].name.len + entries[0].compressed.?.len + 30 + entries[1].name.len;
    zip[readme_data] ^= 1;
    try temp.dir.writeFile(io, .{ .sub_path = "release.zip", .data = zip });
    var corrupt = try Inspector.init(allocator, "1", binary.target("windows-x64").?, "license\n", "readme\n");
    defer corrupt.deinit();
    try std.testing.expectError(error.InvalidZipChecksum, inspectZip(allocator, io, &corrupt, path, 1024));
    zip[readme_data] ^= 1;
    zip[30] ^= 1;
    try temp.dir.writeFile(io, .{ .sub_path = "release.zip", .data = zip });
    var mismatch = try Inspector.init(allocator, "1", binary.target("windows-x64").?, "license\n", "readme\n");
    defer mismatch.deinit();
    try std.testing.expectError(error.InvalidZipLocalHeader, inspectZip(allocator, io, &mismatch, path, 1024));
    entries[2].mode = 0o120777;
    const link_zip = try testZip(allocator, &entries);
    defer allocator.free(link_zip);
    try temp.dir.writeFile(io, .{ .sub_path = "release.zip", .data = link_zip });
    var link = try Inspector.init(allocator, "1", binary.target("windows-x64").?, "license\n", "readme\n");
    defer link.deinit();
    try std.testing.expectError(error.UnexpectedArchiveMember, inspectZip(allocator, io, &link, path, 1024));
    entries[2].mode = 0o100755;
    entries[0].compressed = "\xcb\xc9\x4c\x4e\xcd\x2b\x4e\xe5\x02\x00\x00";
    const trailing = try testZip(allocator, &entries);
    defer allocator.free(trailing);
    try temp.dir.writeFile(io, .{ .sub_path = "release.zip", .data = trailing });
    var invalid_deflate = try Inspector.init(allocator, "1", binary.target("windows-x64").?, "license\n", "readme\n");
    defer invalid_deflate.deinit();
    try std.testing.expectError(error.InvalidZipCompressedData, inspectZip(allocator, io, &invalid_deflate, path, 1024));

    entries[0].compressed = "\xcb\xc9\x4c\x4e\xcd\x2b\x4e\xe5\x02";
    const truncated = try testZip(allocator, &entries);
    defer allocator.free(truncated);
    try temp.dir.writeFile(io, .{ .sub_path = "release.zip", .data = truncated });
    var truncated_deflate = try Inspector.init(allocator, "1", binary.target("windows-x64").?, "license\n", "readme\n");
    defer truncated_deflate.deinit();
    try std.testing.expectError(error.EndOfStream, inspectZip(allocator, io, &truncated_deflate, path, 1024));
}
