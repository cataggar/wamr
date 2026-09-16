const std = @import("std");
const wabt = @import("wabt");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) return error.ExpectedInputAndOutput;
    const cwd = std.Io.Dir.cwd();
    const source = try cwd.readFileAlloc(init.io, args[1], allocator, .limited(wabt.max_input_file_size));
    defer allocator.free(source);
    var module = try wabt.text.Parser.parseModule(allocator, source);
    defer module.deinit();
    // The pinned WABT validator rejects its own null-funcref sentinel.
    // The matching wamrc loader validates the emitted wasm instead.
    const bytes = try wabt.binary.writer.writeModule(allocator, &module);
    defer allocator.free(bytes);
    try cwd.writeFile(init.io, .{ .sub_path = args[2], .data = bytes });
}
