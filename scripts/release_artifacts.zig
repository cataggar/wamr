//! Experimental native release verification; the Python release gate remains authoritative.
const std = @import("std");
const binary = @import("release_binary.zig");
const version = @import("release_version.zig");
const verify = @import("release_verify.zig");
const manifest = @import("release_manifest.zig");

const usage =
    \\usage: release-artifacts version VERSION
    \\       release-artifacts verify --version VERSION --source-sha SHA --platform NAME
    \\           --archive FILE --work-dir DIR --output FILE [--require-smoke] [--require-signature]
    \\       release-artifacts manifest --version VERSION --source-sha SHA --directory DIR
    \\Options: --repo-root DIR (default: current directory)
    \\         --max-expanded-bytes N (default: 536870912)
    \\
;

const Command = enum { version, verify, manifest };
const Args = struct {
    command: Command,
    raw_version: ?[]const u8 = null,
    source_sha: ?[]const u8 = null,
    platform: ?[]const u8 = null,
    archive_path: ?[]const u8 = null,
    work_dir: ?[]const u8 = null,
    output: ?[]const u8 = null,
    directory: ?[]const u8 = null,
    repo_root: []const u8 = ".",
    max_expanded_bytes: usize = 512 * 1024 * 1024,
    require_smoke: bool = false,
    require_signature: bool = false,
};

fn parse(args: []const []const u8) !Args {
    if (args.len == 0) return error.MissingCommand;
    if (std.mem.eql(u8, args[0], "-h") or std.mem.eql(u8, args[0], "--help")) return error.HelpRequested;
    const command: Command = if (std.mem.eql(u8, args[0], "version"))
        .version
    else if (std.mem.eql(u8, args[0], "verify"))
        .verify
    else if (std.mem.eql(u8, args[0], "manifest"))
        .manifest
    else
        return error.UnknownCommand;
    var result: Args = .{ .command = command };
    if (command == .version) {
        if (args.len != 2) return error.MissingRequiredArgument;
        result.raw_version = args[1];
        return result;
    }
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const raw = args[i];
        if (std.mem.eql(u8, raw, "-h") or std.mem.eql(u8, raw, "--help")) return error.HelpRequested;
        if (std.mem.eql(u8, raw, "--require-smoke") and command == .verify) {
            result.require_smoke = true;
            continue;
        }
        if (std.mem.eql(u8, raw, "--require-signature") and command == .verify) {
            result.require_signature = true;
            continue;
        }
        const split = std.mem.indexOfScalar(u8, raw, '=');
        const option = if (split) |index| raw[0..index] else raw;
        const value = if (split) |index| raw[index + 1 ..] else blk: {
            i += 1;
            if (i == args.len) return error.MissingOptionValue;
            break :blk args[i];
        };
        if (value.len == 0) return error.EmptyOptionValue;
        if (std.mem.eql(u8, option, "--version")) {
            result.raw_version = value;
        } else if (std.mem.eql(u8, option, "--source-sha")) {
            result.source_sha = value;
        } else if (std.mem.eql(u8, option, "--repo-root")) {
            result.repo_root = value;
        } else if (std.mem.eql(u8, option, "--max-expanded-bytes")) {
            result.max_expanded_bytes = try std.fmt.parseInt(usize, value, 10);
            if (result.max_expanded_bytes == 0) return error.InvalidExpandedLimit;
        } else if (command == .verify and std.mem.eql(u8, option, "--platform")) {
            result.platform = value;
        } else if (command == .verify and std.mem.eql(u8, option, "--archive")) {
            result.archive_path = value;
        } else if (command == .verify and std.mem.eql(u8, option, "--work-dir")) {
            result.work_dir = value;
        } else if (command == .verify and std.mem.eql(u8, option, "--output")) {
            result.output = value;
        } else if (command == .manifest and std.mem.eql(u8, option, "--directory")) {
            result.directory = value;
        } else return error.UnknownOption;
    }
    if (result.raw_version == null or result.source_sha == null) return error.MissingRequiredArgument;
    if (command == .verify and (result.platform == null or result.archive_path == null or result.work_dir == null or result.output == null))
        return error.MissingRequiredArgument;
    if (command == .manifest and result.directory == null) return error.MissingRequiredArgument;
    return result;
}

fn execute(init: std.process.Init, args: Args) !void {
    const allocator = init.arena.allocator();
    const checked_version = try version.releaseVersion(args.raw_version.?);
    if (args.command == .version) {
        const pep = try version.toPep440(allocator, checked_version);
        const output = try std.fmt.allocPrint(allocator, "version={s}\npep440={s}\n", .{ checked_version, pep });
        try std.Io.File.stdout().writeStreamingAll(init.io, output);
        return;
    }
    const repo = try std.Io.Dir.cwd().realPathFileAlloc(init.io, args.repo_root, allocator);
    if (args.command == .verify) {
        const target = binary.target(args.platform.?) orelse return error.InvalidReleaseTarget;
        const report = try verify.verify(init.gpa, init.io, init.environ_map, .{
            .repo = repo,
            .work_dir = args.work_dir.?,
            .archive_path = args.archive_path.?,
            .version = checked_version,
            .source_sha = args.source_sha.?,
            .target = target,
            .require_smoke = args.require_smoke,
            .require_signature = args.require_signature,
            .max_expanded_bytes = args.max_expanded_bytes,
        });
        defer init.gpa.free(report);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args.output.?, .data = report });
    } else {
        try manifest.write(init.gpa, init.io, init.environ_map, .{
            .repo = repo,
            .directory = args.directory.?,
            .version = checked_version,
            .source_sha = args.source_sha.?,
            .max_expanded_bytes = args.max_expanded_bytes,
        });
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parse(args[1..]) catch |err| {
        if (err == error.HelpRequested) {
            try std.Io.File.stdout().writeStreamingAll(init.io, usage);
            return;
        }
        std.debug.print("[release] {s}\n{s}", .{ @errorName(err), usage });
        std.process.exit(2);
    };
    execute(init, options) catch |err| {
        std.debug.print("[release] {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

test "release command parsing rejects missing and invalid arguments" {
    const opts = try parse(&.{ "verify", "--version", "v1.2.3", "--source-sha=a", "--platform", "linux-x64", "--archive", "a", "--work-dir", "b", "--output", "c", "--require-smoke" });
    try std.testing.expect(opts.require_smoke);
    try std.testing.expectEqualStrings("linux-x64", opts.platform.?);
    try std.testing.expectError(error.MissingRequiredArgument, parse(&.{ "manifest", "--version", "1.2.3" }));
    try std.testing.expectError(error.UnknownOption, parse(&.{ "manifest", "--require-smoke", "x" }));
    try std.testing.expectError(error.InvalidExpandedLimit, parse(&.{ "manifest", "--max-expanded-bytes", "0" }));
}
