//! Synthetic nine-platform inventory and failure-path integration test.
const std = @import("std");
const builtin = @import("builtin");
const archive = @import("release_archive.zig");
const binary = @import("release_binary.zig");
const files = @import("release_files.zig");
const manifest = @import("release_manifest.zig");
const process_control = @import("process_control.zig");
const smoke = @import("release_smoke.zig");
const Value = std.json.Value;
const Allocator = std.mem.Allocator;

fn binaryFixture(target: binary.Target) [160]u8 {
    var data: [160]u8 = @splat(0);
    switch (target.system) {
        .linux => {
            @memcpy(data[0..6], "\x7fELF\x02\x01");
            std.mem.writeInt(u16, data[16..18], 2, .little);
            std.mem.writeInt(u16, data[18..20], switch (target.arch) {
                .x86_64 => 62,
                .aarch64 => 183,
                .riscv64 => 243,
            }, .little);
        },
        .macos => {
            @memcpy(data[0..4], "\xcf\xfa\xed\xfe");
            std.mem.writeInt(u32, data[4..8], if (target.arch == .x86_64) 0x1000007 else 0x100000c, .little);
            std.mem.writeInt(u32, data[12..16], 2, .little);
        },
        .windows => {
            @memcpy(data[0..2], "MZ");
            std.mem.writeInt(u32, data[60..64], 64, .little);
            @memcpy(data[64..68], "PE\x00\x00");
            std.mem.writeInt(u16, data[68..70], if (target.arch == .x86_64) 0x8664 else 0xaa64, .little);
            std.mem.writeInt(u16, data[88..90], 0x20b, .little);
        },
    }
    return data;
}

fn compress(allocator: Allocator, io: std.Io, tmp: std.Io.Dir, tmp_path: []const u8, tool: []const u8, input: []const u8) ![]u8 {
    try tmp.writeFile(io, .{ .sub_path = "input.tar", .data = input });
    const path = try std.fs.path.join(allocator, &.{ tmp_path, "input.tar" });
    const result = try process_control.runTimed(allocator, io, .{
        .argv = &.{ tool, "--stdout", "--", path },
        .stdout_limit = .limited(4 * 1024 * 1024),
    }, 30);
    if (!result.term.success()) {
        std.debug.print("fixture {s} failed: {s}\n", .{ tool, result.stderr });
        return error.FixtureCompressionFailed;
    }
    return result.stdout;
}

fn paxComment(w: *std.Io.Writer, sha: []const u8, allocator: Allocator) !void {
    const body = try std.fmt.allocPrint(allocator, "comment={s}\n", .{sha});
    const size = body.len + 3;
    const record = try std.fmt.allocPrint(allocator, "{d} {s}", .{ size, body });
    if (record.len != size) return error.InvalidFixture;
    var header: std.tar.Writer.Header = .init(.regular);
    const raw: *[512]u8 = @ptrCast(&header);
    raw[156] = 'g';
    try header.setSize(record.len);
    try header.write(w);
    try w.writeAll(record);
    try w.splatByteAll(0, (512 - record.len % 512) % 512);
}

test "complete synthetic release inventory is generated and rejects tampering" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const temp_path = try temp.dir.realPathFileAlloc(io, ".", allocator);
    const repo = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
    const revision = try process_control.runTimed(allocator, io, .{ .argv = &.{ "git", "-C", repo, "rev-parse", "HEAD" } }, 30);
    try std.testing.expect(revision.term.success());
    const sha = std.mem.trim(u8, revision.stdout, " \t\r\n");
    try temp.dir.createDirPath(io, "assets");
    const asset_dir = try std.fs.path.join(allocator, &.{ temp_path, "assets" });
    const cwd = std.Io.Dir.cwd();
    const license = try cwd.readFileAlloc(io, "LICENSE", allocator, .unlimited);
    const readme = try cwd.readFileAlloc(io, "README.md", allocator, .unlimited);
    const build = try cwd.readFileAlloc(io, "build.zig", allocator, .unlimited);
    var fixture_hashes: [2][64]u8 = undefined;
    var output_hashes: [2][64]u8 = undefined;
    for (smoke.fixtures, 0..) |fixture, index| {
        fixture_hashes[index] = try files.fileDigest(io, try std.fs.path.join(allocator, &.{ repo, fixture.path }));
        output_hashes[index] = files.digest(fixture.output);
    }
    for (binary.targets) |target| {
        const stem = try std.fmt.allocPrint(allocator, "wamr-1.2.3-{s}", .{target.name});
        const extension = if (target.system == .windows) ".exe" else "";
        const names = [_][]const u8{
            try std.fmt.allocPrint(allocator, "{s}/LICENSE", .{stem}),
            try std.fmt.allocPrint(allocator, "{s}/README.md", .{stem}),
            try std.fmt.allocPrint(allocator, "{s}/bin/wamr{s}", .{ stem, extension }),
            try std.fmt.allocPrint(allocator, "{s}/bin/wamrc{s}", .{ stem, extension }),
        };
        const exe = binaryFixture(target);
        const data = [_][]const u8{ license, readme, &exe, &exe };
        var archive_bytes: []const u8 = undefined;
        if (std.mem.eql(u8, target.name, "windows-x64")) {
            try temp.dir.createDirPath(io, try std.fmt.allocPrint(allocator, "{s}/bin", .{stem}));
            for (names, data) |name, bytes| try temp.dir.writeFile(io, .{ .sub_path = name, .data = bytes });
            const zip_name = try std.fmt.allocPrint(allocator, "{s}.zip", .{stem});
            const zipped = try process_control.runTimed(allocator, io, .{
                .argv = &.{ "zip", "-r", zip_name, stem },
                .cwd = .{ .path = temp_path },
            }, 30);
            if (!zipped.term.success()) {
                std.debug.print("fixture zip failed: {s}\n", .{zipped.stderr});
                return error.FixtureCompressionFailed;
            }
            archive_bytes = try temp.dir.readFileAlloc(io, zip_name, allocator, .unlimited);
        } else if (target.system == .windows) {
            var entries: [4]archive.ZipFixtureEntry = undefined;
            for (&entries, 0..) |*entry, index| entry.* = .{ .name = names[index], .data = data[index], .mode = 0o100755 };
            archive_bytes = try archive.testZip(allocator, &entries);
        } else {
            var out: std.Io.Writer.Allocating = .init(allocator);
            var tar: std.tar.Writer = .{ .underlying_writer = &out.writer };
            for (names, data, 0..) |name, bytes, index|
                try tar.writeFileBytes(name, bytes, .{ .mode = if (index >= 2) 0o755 else 0o644 });
            try tar.finishPedantically();
            archive_bytes = try compress(allocator, io, temp.dir, temp_path, "gzip", out.written());
        }
        const archive_name = try std.fmt.allocPrint(allocator, "{s}{s}", .{ stem, if (target.system == .windows) ".zip" else ".tar.gz" });
        try cwd.writeFile(io, .{ .sub_path = try std.fs.path.join(allocator, &.{ asset_dir, archive_name }), .data = archive_bytes });
        var members: std.json.ObjectMap = .empty;
        for (names, data) |name, bytes| {
            const hash = files.digest(bytes);
            try members.put(allocator, name, try files.jsonValue(allocator, .{ .sha256 = &hash, .size = bytes.len }));
        }
        const exe_hash = files.digest(&exe);
        const sig = if (target.system == .windows) try files.jsonValue(allocator, [_]struct {
            name: []const u8,
            sha256: []const u8,
            status: []const u8,
            signer: []const u8,
            timestamp_signer: []const u8,
        }{
            .{ .name = "wamr.exe", .sha256 = &exe_hash, .status = "Valid", .signer = "S", .timestamp_signer = "T" },
            .{ .name = "wamrc.exe", .sha256 = &exe_hash, .status = "Valid", .signer = "S", .timestamp_signer = "T" },
        }) else try files.jsonValue(allocator, [_]u8{});
        var cases: [2]smoke.Case = undefined;
        for (smoke.fixtures, 0..) |fixture, index| cases[index] = .{
            .fixture = fixture.path,
            .fixture_sha256 = &fixture_hashes[index],
            .stdout_sha256 = &output_hashes[index],
        };
        const passed = try files.jsonValue(allocator, .{
            .status = "passed",
            .versions = .{ .@"wamr version" = "wamr 1.2.3\n", .@"wamr --version" = "wamr 1.2.3\n", .@"wamrc version" = "wamrc 1.2.3\n" },
            .cases = cases,
        });
        const archive_hash = files.digest(archive_bytes);
        var report: std.json.ObjectMap = .empty;
        try report.put(allocator, "schema_version", .{ .integer = 1 });
        try report.put(allocator, "version", .{ .string = "1.2.3" });
        try report.put(allocator, "pep440", .{ .string = "1.2.3" });
        try report.put(allocator, "source_sha", .{ .string = sha });
        try report.put(allocator, "platform", .{ .string = target.name });
        try report.put(allocator, "host", try files.jsonValue(allocator, [_][]const u8{ @tagName(target.system), @tagName(target.arch) }));
        try report.put(allocator, "archive", .{ .string = archive_name });
        try report.put(allocator, "archive_sha256", .{ .string = try allocator.dupe(u8, &archive_hash) });
        try report.put(allocator, "files", .{ .object = members });
        try report.put(allocator, "native_smoke", passed);
        try report.put(allocator, "signatures", sig);
        const report_name = try std.fmt.allocPrint(allocator, "{s}.verification.json", .{stem});
        const sbom_name = try std.fmt.allocPrint(allocator, "{s}.sbom.spdx.json", .{stem});
        try cwd.writeFile(io, .{ .sub_path = try std.fs.path.join(allocator, &.{ asset_dir, report_name }), .data = try std.json.Stringify.valueAlloc(allocator, Value{ .object = report }, .{}) });
        try cwd.writeFile(io, .{ .sub_path = try std.fs.path.join(allocator, &.{ asset_dir, sbom_name }), .data = "{\"spdxVersion\":\"SPDX-2.3\",\"SPDXID\":\"SPDXRef-DOCUMENT\",\"documentNamespace\":\"https://example.invalid\"}" });
    }
    var source_output: std.Io.Writer.Allocating = .init(allocator);
    try paxComment(&source_output.writer, sha, allocator);
    var source_tar: std.tar.Writer = .{ .underlying_writer = &source_output.writer };
    try source_tar.writeFileBytes("wamr-1.2.3/LICENSE", license, .{});
    try source_tar.writeFileBytes("wamr-1.2.3/README.md", readme, .{});
    try source_tar.writeFileBytes("wamr-1.2.3/build.zig", build, .{});
    try source_tar.finishPedantically();
    const source_bytes = try compress(allocator, io, temp.dir, temp_path, "xz", source_output.written());
    const source_path = try std.fs.path.join(allocator, &.{ asset_dir, "wamr-1.2.3.tar.xz" });
    try cwd.writeFile(io, .{ .sub_path = source_path, .data = source_bytes });
    var environment: std.process.Environ.Map = .init(allocator);
    const opts: manifest.Options = .{ .repo = repo, .directory = asset_dir, .version = "1.2.3", .source_sha = sha, .max_expanded_bytes = 4 * 1024 * 1024 };
    try manifest.write(allocator, io, &environment, opts);
    const manifest_path = try std.fs.path.join(allocator, &.{ asset_dir, "release-manifest.json" });
    const result = try cwd.readFileAlloc(io, manifest_path, allocator, .unlimited);
    const json = try std.json.parseFromSliceLeaky(Value, allocator, result, .{});
    try std.testing.expectEqual(@as(usize, binary.targets.len), json.object.get("platforms").?.object.count());
    try std.testing.expectEqual(@as(usize, binary.targets.len * 3 + 2), json.object.get("assets").?.array.items.len);
    for (json.object.get("assets").?.array.items) |asset| {
        const info = asset.object;
        const path = try std.fs.path.join(allocator, &.{ asset_dir, info.get("name").?.string });
        const stat = try cwd.statFile(io, path, .{});
        try std.testing.expectEqual(stat.size, @as(u64, @intCast(info.get("size").?.integer)));
        const hash = try files.fileDigest(io, path);
        try std.testing.expectEqualStrings(&hash, info.get("sha256").?.string);
    }
    const checksums = try cwd.readFileAlloc(io, try std.fs.path.join(allocator, &.{ asset_dir, "SHA256SUMS" }), allocator, .unlimited);
    var lines = std.mem.splitScalar(u8, checksums, '\n');
    var checksum_count: usize = 0;
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        try std.testing.expect(line.len > 66);
        try std.testing.expectEqualStrings("  ", line[64..66]);
        const hash = try files.fileDigest(io, try std.fs.path.join(allocator, &.{ asset_dir, line[66..] }));
        try std.testing.expectEqualStrings(&hash, line[0..64]);
        checksum_count += 1;
    }
    try std.testing.expectEqual(@as(usize, binary.targets.len * 3 + 3), checksum_count);
    const corrupt = try allocator.dupe(u8, source_bytes);
    corrupt[corrupt.len - 5] ^= 1;
    try cwd.writeFile(io, .{ .sub_path = source_path, .data = corrupt });
    try std.testing.expectError(error.DecompressionFailed, manifest.write(allocator, io, &environment, opts));
    try std.testing.expectEqualStrings(result, try cwd.readFileAlloc(io, manifest_path, allocator, .unlimited));
}

test "workflow publication boundary and matrix" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    const workflow = try cwd.readFileAlloc(io, ".github/workflows/release.yml", allocator, .limited(4 * 1024 * 1024));
    try std.testing.expect(std.mem.indexOf(u8, workflow, "workflow_dispatch:") != null);
    const create_idx = std.mem.indexOf(u8, workflow, "- name: Create release\n") orelse return error.TestExpectedEqual;
    const publish = workflow[create_idx..];
    try std.testing.expect(std.mem.indexOf(u8, publish, "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')") != null);
    try std.testing.expect(std.mem.indexOf(u8, publish, "make_latest:") != null);
    try std.testing.expect(std.mem.indexOf(u8, workflow, "upload-release-assets: false") != null);
    for (binary.targets) |t| {
        const needle = try std.fmt.allocPrint(allocator, "name: {s}\n", .{t.name});
        try std.testing.expect(std.mem.indexOf(u8, workflow, needle) != null);
    }
    for ([_][]const u8{ ".github/workflows/pypi.yml", ".github/workflows/pypi-wamrc.yml" }) |wf| {
        try std.testing.expectError(error.FileNotFound, cwd.statFile(io, wf, .{}));
    }
}
