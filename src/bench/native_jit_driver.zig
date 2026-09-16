//! Explicit Linux sampler; qualification and process capture belong to the host.
const std = @import("std");
const linux = @import("native_linux.zig");
const fixture = @import("jit_bench_fixture");

pub fn main(comptime with_compiler: bool, init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3 or args[2].len != 64) return error.InvalidArguments;
    for (args[2]) |c| if (!std.ascii.isHex(c)) return error.InvalidArguments;
    const mode = args[1];
    if (with_compiler) {
        if (!std.mem.eql(u8, mode, "fast") and !std.mem.eql(u8, mode, "full")) return error.InvalidArguments;
    } else if (!std.mem.eql(u8, mode, "aot")) return error.InvalidArguments;
    const guest = if (with_compiler) @import("native_jit_guest.zig") else @import("native_jit_aot_guest.zig");
    var capture: guest.Capture = .{};
    var pages: linux.Pages = .{};
    const resolution = try linux.resolution(.MONOTONIC);
    if (with_compiler) {
        guest.run(init.gpa, pages.platform(), fixture.wasm, if (std.mem.eql(u8, mode, "fast")) .fast else .full, args[2], resolution, &capture) catch {};
    } else {
        guest.run(init.gpa, pages.platform(), fixture.wasm, fixture.aot, args[2], resolution, &capture) catch {};
    }
    if (capture.report.caller_live_after_teardown != 0 or pages.reserved() != 0) return error.IncompleteTeardown;
    var buffer: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(init.io, &buffer);
    try capture.writeRecord(&out.interface);
    try out.interface.flush();
    if (capture.report.failure != null) return error.SampleFailed;
}
