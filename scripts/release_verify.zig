//! Assemble a release verification report from inspected and extracted archives.
const std = @import("std");
const builtin = @import("builtin");
const archive = @import("release_archive.zig");
const binary = @import("release_binary.zig");
const extract = @import("release_extract.zig");
const files_util = @import("release_files.zig");
const process_control = @import("process_control.zig");
const signatures = @import("release_signatures.zig");
const smoke = @import("release_smoke.zig");
const release_version = @import("release_version.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const Options = struct {
    repo: []const u8,
    work_dir: []const u8,
    archive_path: []const u8,
    version: []const u8,
    source_sha: []const u8,
    target: binary.Target,
    require_smoke: bool = false,
    require_signature: bool = false,
    max_expanded_bytes: usize,
    timeout_seconds: u32 = 120,
};

/// Returns a complete schema-v1 report as UTF-8 JSON with a trailing newline.
pub fn verify(allocator: Allocator, io: std.Io, environment: *const std.process.Environ.Map, opts: Options) ![]u8 {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const scratch = arena.allocator();
    const normalized = try release_version.releaseVersion(opts.version);
    if (!std.mem.eql(u8, normalized, opts.version)) return error.InvalidReleaseVersion;
    const target = binary.target(opts.target.name) orelse return error.InvalidReleaseTarget;
    if (target.system != opts.target.system or target.arch != opts.target.arch or
        target.required_native != opts.target.required_native) return error.InvalidReleaseTarget;
    try files_util.checkedSource(scratch, io, opts.repo, opts.source_sha);
    const cwd = std.Io.Dir.cwd();
    const license_path = try std.fs.path.join(scratch, &.{ opts.repo, "LICENSE" });
    const readme_path = try std.fs.path.join(scratch, &.{ opts.repo, "README.md" });
    const license = try cwd.readFileAlloc(io, license_path, scratch, .unlimited);
    const readme = try cwd.readFileAlloc(io, readme_path, scratch, .unlimited);
    var inspector = try archive.Inspector.init(scratch, opts.version, opts.target, license, readme);
    defer inspector.deinit();
    if (opts.target.system == .windows) {
        try archive.inspectZip(scratch, io, &inspector, opts.archive_path, opts.max_expanded_bytes);
    } else {
        try archive.inspectGzip(scratch, io, &inspector, opts.archive_path, opts.max_expanded_bytes, opts.timeout_seconds);
    }
    const system = switch (builtin.os.tag) {
        .linux => "linux",
        .macos => "macos",
        .windows => "windows",
        else => @tagName(builtin.os.tag),
    };
    const arch = @tagName(builtin.cpu.arch);
    const native = std.mem.eql(u8, system, @tagName(opts.target.system)) and std.mem.eql(u8, arch, @tagName(opts.target.arch));
    if (opts.require_smoke and !native) return error.NonNativeSmokeTarget;
    if (opts.require_signature and (builtin.os.tag != .windows or opts.target.system != .windows))
        return error.WindowsSignaturesRequireWindows;
    const extracted = try extract.extract(scratch, io, &inspector, opts.version, opts.work_dir);
    const bin_dir = try std.fs.path.join(scratch, &.{ extracted, "bin" });

    var smoke_value: Value = undefined;
    if (native) {
        var result = try smoke.run(scratch, io, opts.repo, bin_dir, opts.work_dir, opts.target, opts.version);
        defer result.deinit(scratch);
        smoke_value = try files_util.jsonValue(scratch, result);
    } else {
        smoke_value = try files_util.jsonValue(scratch, .{ .status = "not-run", .reason = "host-target-mismatch" });
    }
    var signature_value = try files_util.jsonValue(scratch, [_]signatures.Signature{});
    if (opts.require_signature) {
        var result = try signatures.collect(scratch, io, environment, bin_dir, &inspector);
        defer result.deinit();
        signature_value = try files_util.jsonValue(scratch, result.value);
    }
    var files: std.json.ObjectMap = .empty;
    for (inspector.files) |member_optional| {
        const member = member_optional.?;
        const member_digest = files_util.digest(member.data);
        try files.put(scratch, member.name, try files_util.jsonValue(scratch, .{
            .sha256 = &member_digest,
            .size = member.data.len,
        }));
    }
    const archive_digest = try files_util.fileDigest(io, opts.archive_path);
    const pep440 = try release_version.toPep440(scratch, opts.version);
    var root: std.json.ObjectMap = .empty;
    try root.put(scratch, "schema_version", .{ .integer = 1 });
    try root.put(scratch, "version", .{ .string = opts.version });
    try root.put(scratch, "pep440", .{ .string = pep440 });
    try root.put(scratch, "source_sha", .{ .string = opts.source_sha });
    try root.put(scratch, "platform", .{ .string = opts.target.name });
    try root.put(scratch, "host", try files_util.jsonValue(scratch, [_][]const u8{ system, arch }));
    try root.put(scratch, "archive", .{ .string = std.fs.path.basename(opts.archive_path) });
    try root.put(scratch, "archive_sha256", .{ .string = &archive_digest });
    try root.put(scratch, "files", .{ .object = files });
    try root.put(scratch, "native_smoke", smoke_value);
    try root.put(scratch, "signatures", signature_value);
    const encoded = try std.json.Stringify.valueAlloc(scratch, Value{ .object = root }, .{ .whitespace = .indent_2 });
    return std.fmt.allocPrint(allocator, "{s}\n", .{encoded});
}

test "verification report binds archive, checkout, and non-native evidence" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const temp_path = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(temp_path);
    const repo = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
    defer allocator.free(repo);
    const revision = try process_control.runTimed(allocator, io, .{ .argv = &.{ "git", "-C", repo, "rev-parse", "HEAD" } }, 30);
    defer allocator.free(revision.stdout);
    defer allocator.free(revision.stderr);
    try std.testing.expect(revision.term.success());
    const sha = std.mem.trim(u8, revision.stdout, " \t\r\n");
    const target = binary.target(if (builtin.cpu.arch == .x86_64 and builtin.os.tag == .linux) "linux-arm64" else "linux-x64").?;
    const stem = try std.fmt.allocPrint(allocator, "wamr-1.2.3-{s}", .{target.name});
    defer allocator.free(stem);
    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();
    var tar: std.tar.Writer = .{ .underlying_writer = &output.writer };
    const license = try std.Io.Dir.cwd().readFileAlloc(io, "LICENSE", allocator, .unlimited);
    defer allocator.free(license);
    const readme = try std.Io.Dir.cwd().readFileAlloc(io, "README.md", allocator, .unlimited);
    defer allocator.free(readme);
    const license_name = try std.fmt.allocPrint(allocator, "{s}/LICENSE", .{stem});
    defer allocator.free(license_name);
    const readme_name = try std.fmt.allocPrint(allocator, "{s}/README.md", .{stem});
    defer allocator.free(readme_name);
    const wamr_name = try std.fmt.allocPrint(allocator, "{s}/bin/wamr", .{stem});
    defer allocator.free(wamr_name);
    const wamrc_name = try std.fmt.allocPrint(allocator, "{s}/bin/wamrc", .{stem});
    defer allocator.free(wamrc_name);
    try tar.writeFileBytes(license_name, license, .{});
    try tar.writeFileBytes(readme_name, readme, .{});
    var elf: [64]u8 = @splat(0);
    @memcpy(elf[0..6], "\x7fELF\x02\x01");
    std.mem.writeInt(u16, elf[16..18], 2, .little);
    std.mem.writeInt(u16, elf[18..20], if (target.arch == .aarch64) 183 else 62, .little);
    try tar.writeFileBytes(wamr_name, &elf, .{ .mode = 0o755 });
    try tar.writeFileBytes(wamrc_name, &elf, .{ .mode = 0o755 });
    try tar.finishPedantically();
    try temp.dir.writeFile(io, .{ .sub_path = "release.tar", .data = output.written() });
    const tar_path = try std.fs.path.join(allocator, &.{ temp_path, "release.tar" });
    defer allocator.free(tar_path);
    const compressed = try process_control.runTimed(allocator, io, .{ .argv = &.{ "gzip", "--keep", "--", tar_path } }, 30);
    defer allocator.free(compressed.stdout);
    defer allocator.free(compressed.stderr);
    try std.testing.expect(compressed.term.success());
    const archive_path = try std.fs.path.join(allocator, &.{ temp_path, "release.tar.gz" });
    defer allocator.free(archive_path);
    const work_dir = try std.fs.path.join(allocator, &.{ temp_path, "work" });
    defer allocator.free(work_dir);
    var env: std.process.Environ.Map = .init(allocator);
    defer env.deinit();
    var options: Options = .{
        .repo = repo,
        .work_dir = work_dir,
        .archive_path = archive_path,
        .version = "1.2.3",
        .source_sha = "wrong",
        .target = target,
        .max_expanded_bytes = 1024 * 1024,
    };
    try std.testing.expectError(error.SourceRevisionMismatch, verify(allocator, io, &env, options));
    options.source_sha = sha;
    options.require_smoke = true;
    try std.testing.expectError(error.NonNativeSmokeTarget, verify(allocator, io, &env, options));
    options.require_smoke = false;
    const report = try verify(allocator, io, &env, options);
    defer allocator.free(report);
    var parsed = try std.json.parseFromSlice(Value, allocator, report, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqualStrings(sha, root.get("source_sha").?.string);
    try std.testing.expectEqualStrings("not-run", root.get("native_smoke").?.object.get("status").?.string);
    const metadata = root.get("files").?.object.get(wamr_name).?.object;
    const expected_digest = files_util.digest(&elf);
    try std.testing.expectEqualStrings(&expected_digest, metadata.get("sha256").?.string);
    try std.testing.expectEqual(@as(i64, elf.len), metadata.get("size").?.integer);
    const expected_archive_digest = try files_util.fileDigest(io, archive_path);
    try std.testing.expectEqualStrings(&expected_archive_digest, root.get("archive_sha256").?.string);
}
