const std = @import("std");

pub fn setPath(run: *std.Build.Step.Run, key: []const u8, path: std.Build.LazyPath) void {
    const b = run.step.owner;
    const helper = b.addExecutable(.{
        .name = "wamr-build-environment",
        .root_module = b.createModule(.{
            .root_source_file = b.path("build/environment.zig"),
            .target = b.graph.host,
            .optimize = .safe,
        }),
    });
    const command = run.argv.items;
    run.argv = .empty;
    run.addArtifactArg(helper);
    run.addArg(key);
    const file_path: std.Build.LazyPath = switch (path) {
        .relative => |relative| if (relative.base == .install_bin)
            .{ .relative = .{
                .base = .install_bin,
                .sub_path = b.fmt("{s}{s}", .{ relative.sub_path, b.graph.host.result.exeFileExt() }),
            } }
        else
            path,
        else => path,
    };
    run.addFileArg(file_path);
    run.addArg("--");
    run.argv.appendSlice(b.allocator, command) catch @panic("OOM");
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len < 5 or !std.mem.eql(u8, args[3], "--")) return error.InvalidArguments;
    var environ = try init.environ_map.clone(allocator);
    defer environ.deinit();
    try environ.put(args[1], args[2]);
    var child = try std.process.spawn(init.io, .{
        .argv = args[4..],
        .environ_map = &environ,
        .stdin = .inherit,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    const term = try child.wait(init.io);
    std.process.exit(switch (term) {
        .exited => |code| code,
        .signal => |signal| @intCast(@min(255, 128 + @backingInt(signal))),
        else => return error.UnexpectedTermination,
    });
}
