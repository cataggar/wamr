const std = @import("std");

pub fn build(b: *std.Build) void {
    const wamr = b.dependency("wamr", .{
        .target = b.standardTargetOptions(.{}),
        .optimize = b.standardOptimizeOption(.{}),
    });

    if (wamr.artifact("wamr").kind != .lib)
        @panic("wamr artifact must resolve to the library");
    if (wamr.artifact("wamr-exe").kind != .exe)
        @panic("wamr-exe artifact must resolve to the executable");
}
