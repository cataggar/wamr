//! Validate release inputs and generate the source-bound release inventory.
const std = @import("std");
const archive = @import("release_archive.zig");
const binary = @import("release_binary.zig");
const evidence = @import("release_evidence.zig");
const files_util = @import("release_files.zig");
const identity = @import("release_identity.zig");
const smoke = @import("release_smoke.zig");
const source = @import("release_source.zig");
const release_version = @import("release_version.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const Options = struct {
    repo: []const u8,
    directory: []const u8,
    version: []const u8,
    source_sha: []const u8,
    max_expanded_bytes: usize,
    timeout_seconds: u32 = 120,
};

fn join(allocator: Allocator, directory: []const u8, name: []const u8) ![]u8 {
    return std.fs.path.join(allocator, &.{ directory, name });
}

fn readJson(allocator: Allocator, io: std.Io, directory: []const u8, name: []const u8) !Value {
    const path = try join(allocator, directory, name);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024));
    return std.json.parseFromSliceLeaky(Value, allocator, bytes, .{});
}

fn addAsset(allocator: Allocator, io: std.Io, directory: []const u8, name: []const u8) !Value {
    const path = try join(allocator, directory, name);
    const stat = try std.Io.Dir.cwd().statFile(io, path, .{});
    if (stat.kind != .file) return error.InvalidReleaseAsset;
    const sha = try files_util.fileDigest(io, path);
    return files_util.jsonValue(allocator, .{ .name = name, .size = stat.size, .sha256 = &sha });
}

fn envValue(environment: *const std.process.Environ.Map, name: []const u8) Value {
    return if (environment.get(name)) |value| .{ .string = value } else .null;
}

fn less(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

/// All input assets are checked before any generated outputs are written.
pub fn write(allocator: Allocator, io: std.Io, environment: *const std.process.Environ.Map, opts: Options) !void {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const scratch = arena.allocator();
    const checked = try release_version.releaseVersion(opts.version);
    if (!std.mem.eql(u8, checked, opts.version)) return error.InvalidReleaseVersion;
    try files_util.checkedSource(scratch, io, opts.repo, opts.source_sha);
    const cwd = std.Io.Dir.cwd();
    const dir = try cwd.openDir(io, opts.directory, .{ .iterate = true });
    defer dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var walker = dir.iterate();
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) return error.InvalidReleaseAsset;
        try names.append(scratch, try scratch.dupe(u8, entry.name));
    }
    try identity.validateAssetNames(scratch, names.items, opts.version);

    const license = try cwd.readFileAlloc(io, try join(scratch, opts.repo, "LICENSE"), scratch, .unlimited);
    const readme = try cwd.readFileAlloc(io, try join(scratch, opts.repo, "README.md"), scratch, .unlimited);
    const build = try cwd.readFileAlloc(io, try join(scratch, opts.repo, "build.zig"), scratch, .unlimited);
    var fixture_hashes: [smoke.fixtures.len][64]u8 = undefined;
    var output_hashes: [smoke.fixtures.len][64]u8 = undefined;
    for (smoke.fixtures, 0..) |fixture, index| {
        fixture_hashes[index] = try files_util.fileDigest(io, try join(scratch, opts.repo, fixture.path));
        output_hashes[index] = files_util.digest(fixture.output);
    }
    var platforms: std.json.ObjectMap = .empty;
    var inventory: std.ArrayList([]const u8) = .empty;
    for (binary.targets) |target| {
        const stem = try std.fmt.allocPrint(scratch, "wamr-{s}-{s}", .{ opts.version, target.name });
        const archive_name = try std.fmt.allocPrint(scratch, "{s}{s}", .{ stem, if (target.system == .windows) ".zip" else ".tar.gz" });
        const report_name = try std.fmt.allocPrint(scratch, "{s}.verification.json", .{stem});
        const sbom_name = try std.fmt.allocPrint(scratch, "{s}.sbom.spdx.json", .{stem});
        const archive_path = try join(scratch, opts.directory, archive_name);
        const report = try readJson(scratch, io, opts.directory, report_name);
        var target_arena: std.heap.ArenaAllocator = .init(allocator);
        defer target_arena.deinit();
        const target_alloc = target_arena.allocator();
        var inspector = try archive.Inspector.init(target_alloc, opts.version, target, license, readme);
        defer inspector.deinit();
        if (target.system == .windows) {
            try archive.inspectZip(target_alloc, io, &inspector, archive_path, opts.max_expanded_bytes);
        } else {
            try archive.inspectGzip(target_alloc, io, &inspector, archive_path, opts.max_expanded_bytes, opts.timeout_seconds);
        }
        const archive_digest = try files_util.fileDigest(io, archive_path);
        try identity.validateReport(target_alloc, report, &inspector, opts.version, opts.source_sha, archive_name, &archive_digest);
        const wamr_sha = files_util.digest(inspector.files[2].?.data);
        const wamrc_sha = files_util.digest(inspector.files[3].?.data);
        try evidence.validate(target_alloc, report, .{
            .target = target,
            .version = opts.version,
            .fixture_sha256 = .{ &fixture_hashes[0], &fixture_hashes[1] },
            .stdout_sha256 = .{ &output_hashes[0], &output_hashes[1] },
            .windows_binary_sha256 = .{ &wamr_sha, &wamrc_sha },
        });
        try identity.validateSpdx(try readJson(scratch, io, opts.directory, sbom_name));
        try platforms.put(scratch, target.name, report);
        try inventory.appendSlice(scratch, &.{ archive_name, report_name, sbom_name });
    }
    const source_name = try std.fmt.allocPrint(scratch, "wamr-{s}.tar.xz", .{opts.version});
    const source_path = try join(scratch, opts.directory, source_name);
    try source.inspectXz(scratch, io, source_path, opts.version, opts.source_sha, .{ license, readme, build }, opts.max_expanded_bytes, opts.timeout_seconds);
    try inventory.append(scratch, source_name);

    const draft_path = try std.fmt.allocPrint(scratch, "{s}/docs/releases/v{s}.md", .{ opts.repo, opts.version });
    const draft_stat = cwd.statFile(io, draft_path, .{}) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    const draft = if (draft_stat != null and draft_stat.?.kind == .file)
        try cwd.readFileAlloc(io, draft_path, scratch, .unlimited)
    else
        try std.fmt.allocPrint(scratch, "# wamr {s}\n", .{opts.version});
    const notes = try std.fmt.allocPrint(scratch,
        \\{s}
        \\Source: `{s}`
        \\
        \\Install: `ghr install cataggar/wamr@v{s}`
        \\
        \\See [INSTALL.md](https://github.com/cataggar/wamr/blob/{s}/INSTALL.md) for other installation options.
        \\
    , .{ draft, opts.source_sha, opts.version, opts.source_sha });
    try cwd.writeFile(io, .{ .sub_path = try join(scratch, opts.directory, "RELEASE_NOTES.md"), .data = notes });
    try inventory.append(scratch, "RELEASE_NOTES.md");
    std.mem.sort([]const u8, inventory.items, {}, less);
    var assets: std.array_list.Managed(Value) = .init(scratch);
    for (inventory.items) |name| try assets.append(try addAsset(scratch, io, opts.directory, name));
    var workflow: std.json.ObjectMap = .empty;
    try workflow.put(scratch, "event", envValue(environment, "GITHUB_EVENT_NAME"));
    try workflow.put(scratch, "run_id", envValue(environment, "GITHUB_RUN_ID"));
    try workflow.put(scratch, "run_attempt", envValue(environment, "GITHUB_RUN_ATTEMPT"));
    var manifest: std.json.ObjectMap = .empty;
    try manifest.put(scratch, "schema_version", .{ .integer = 1 });
    try manifest.put(scratch, "version", .{ .string = opts.version });
    try manifest.put(scratch, "pep440", .{ .string = try release_version.toPep440(scratch, opts.version) });
    try manifest.put(scratch, "source_sha", .{ .string = opts.source_sha });
    try manifest.put(scratch, "platforms", .{ .object = platforms });
    try manifest.put(scratch, "workflow", .{ .object = workflow });
    try manifest.put(scratch, "assets", .{ .array = assets });
    const json = try std.json.Stringify.valueAlloc(scratch, Value{ .object = manifest }, .{ .whitespace = .indent_2 });
    try cwd.writeFile(io, .{
        .sub_path = try join(scratch, opts.directory, "release-manifest.json"),
        .data = try std.fmt.allocPrint(scratch, "{s}\n", .{json}),
    });
    try inventory.append(scratch, "release-manifest.json");
    std.mem.sort([]const u8, inventory.items, {}, less);
    var sums: std.ArrayList(u8) = .empty;
    for (inventory.items) |name| {
        const sha = try files_util.fileDigest(io, try join(scratch, opts.directory, name));
        try sums.appendSlice(scratch, try std.fmt.allocPrint(scratch, "{s}  {s}\n", .{ &sha, name }));
    }
    try cwd.writeFile(io, .{ .sub_path = try join(scratch, opts.directory, "SHA256SUMS"), .data = sums.items });
}

test "manifest refuses missing or unexpected release assets before writing outputs" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const repo = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
    defer allocator.free(repo);
    const result = try @import("process_control.zig").runTimed(allocator, io, .{ .argv = &.{ "git", "-C", repo, "rev-parse", "HEAD" } }, 30);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    try std.testing.expect(result.term.success());
    const sha = std.mem.trim(u8, result.stdout, " \t\r\n");
    var env: std.process.Environ.Map = .init(allocator);
    defer env.deinit();
    const opts: Options = .{ .repo = repo, .directory = root, .version = "1.2.3", .source_sha = sha, .max_expanded_bytes = 1024 * 1024 };
    try std.testing.expectError(error.MissingReleaseAsset, write(allocator, io, &env, opts));
    try temp.dir.writeFile(io, .{ .sub_path = "unexpected", .data = "" });
    try std.testing.expectError(error.UnexpectedReleaseAsset, write(allocator, io, &env, opts));
    try std.testing.expectError(error.FileNotFound, temp.dir.access(io, "release-manifest.json", .{}));
}
