const std = @import("std");
const format = @import("runtime/aot/native_format.zig");
const fixture = @import("native_fixture").bytes;

test "native platform: saturated monotonic clocks return an explicit error" {
    const Clock = struct {
        ns: u64 = 100,
        fn read(context: *anyopaque) @import("platform/unikraft.zig").Error!u64 {
            const self: *@This() = @ptrCast(@alignCast(context));
            return self.ns;
        }
    };
    var clock: Clock = .{};
    const platform: @import("platform/unikraft.zig").Platform = .{
        .context = &clock,
        .reserve = undefined,
        .commit = undefined,
        .protect = undefined,
        .unmap = undefined,
        .monotonic_ns = Clock.read,
    };
    try std.testing.expectEqual(@as(u64, 100), try platform.monotonicNs());
    clock.ns = std.math.maxInt(u64);
    try std.testing.expectError(error.ClockFailed, platform.monotonicNs());
}

test "native format: matching host-wamrc fixture metadata is checked" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const module = try format.load(fixture, arena.allocator(), format.abi.cpu_features);
    try std.testing.expect(module.functions.len >= 10);
    try std.testing.expectEqual(@as(usize, 3), module.imports.len);
    try std.testing.expectEqual(@as(u32, 2), module.memory.?.min);
    try std.testing.expectEqual(@as(?u32, 8), module.memory.?.max);
    var found_add = false;
    for (module.exports) |exported| {
        if (std.mem.eql(u8, exported.name, "add")) {
            const sig = try module.signature(exported.index);
            try std.testing.expectEqualSlices(format.ValType, &.{ .i32, .i32 }, sig.params);
            try std.testing.expectEqualSlices(format.ValType, &.{.i32}, sig.results);
            found_add = true;
        }
    }
    try std.testing.expect(found_add);
}
test "native format: null funcref sentinel is not an out of range function index" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var module = try format.load(@import("native_fixture").tables, arena.allocator(), format.abi.cpu_features);
    try std.testing.expectEqual(@as(usize, 2), module.elements.len);
    try std.testing.expectEqualSlices(u32, &.{ 0, format.null_function_index }, module.elements[0].indices);
    try std.testing.expectEqualSlices(u32, &.{ format.null_function_index, 0 }, module.elements[1].indices);
    try std.testing.expect(!module.elements[0].passive);
    try std.testing.expect(module.elements[1].passive);
    var invalid = module.elements[0];
    invalid.indices = &.{@intCast(module.functions.len + module.imports.len)};
    module.elements = &.{invalid};
    try std.testing.expectError(error.InvalidIndex, module.validate());
}
test "native format: format runtime ABI target and CPU rejection" {
    const cases = [_]struct { offset: usize, value: u8, expected: anyerror }{
        .{ .offset = 0, .value = 1, .expected = error.InvalidMagic },
        .{ .offset = 4, .value = 0, .expected = error.InvalidVersion },
        .{ .offset = 16, .value = 6, .expected = error.UnsupportedTarget },
        .{ .offset = 18, .value = 3, .expected = error.UnsupportedTarget },
        .{ .offset = 22, .value = 0xb7, .expected = error.UnsupportedTarget },
        .{ .offset = 24, .value = 0, .expected = error.UnsupportedTarget },
        .{ .offset = 28, .value = 0, .expected = error.UnsupportedTarget },
        .{ .offset = 32, .value = 'a', .expected = error.UnsupportedTarget },
        .{ .offset = 48, .value = 0, .expected = error.UnsupportedTarget },
    };
    for (cases) |case| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const bytes = try arena.allocator().dupe(u8, fixture);
        bytes[case.offset] = case.value;
        try std.testing.expectError(case.expected, format.load(bytes, arena.allocator(), format.abi.cpu_features));
    }
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.UnsupportedTarget, format.load(fixture, arena.allocator(), 0));
}
test "native format: section-local bounds duplicate sections and invalid indices" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try a.dupe(u8, fixture);
    // A target section is exactly 40 bytes, never reads into the next section.
    std.mem.writeInt(u32, bytes[12..16], 39, .little);
    try std.testing.expectError(error.InvalidSection, format.load(bytes, a, format.abi.cpu_features));
    @memcpy(bytes, fixture);
    // Force the second section to repeat target_info.
    std.mem.writeInt(u32, bytes[56..60], 0, .little);
    try std.testing.expectError(error.InvalidSection, format.load(bytes, a, format.abi.cpu_features));
    @memcpy(bytes, fixture);
    var pos: usize = 8;
    while (pos < bytes.len) {
        const id = std.mem.readInt(u32, bytes[pos..][0..4], .little);
        const len = std.mem.readInt(u32, bytes[pos + 4 ..][0..4], .little);
        if (id == 3) {
            std.mem.writeInt(u32, bytes[pos + 12 ..][0..4], std.math.maxInt(u32), .little);
            break;
        }
        pos += 8 + len;
    }
    try std.testing.expectError(error.InvalidIndex, format.load(bytes, a, format.abi.cpu_features));
}
test "native format: truncated sections never overread and arena rollback is deterministic" {
    // Test every interior byte truncation (section boundaries may be complete
    // valid containers, because optional metadata sections can be omitted).
    var section: usize = 8;
    while (section < fixture.len) {
        const length = std.mem.readInt(u32, fixture[section + 4 ..][0..4], .little);
        for (section + 1..section + 8 + length) |end| {
            var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
            defer arena.deinit();
            if (format.load(fixture[0..end], arena.allocator(), format.abi.cpu_features)) |_| {
                return error.ExpectedError;
            } else |_| {}
        }
        section += 8 + length;
    }
    try std.testing.checkAllAllocationFailures(std.testing.allocator, parseWithArena, .{});
}
fn parseWithArena(allocator: std.mem.Allocator) !void {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    _ = try format.load(fixture, arena.allocator(), format.abi.cpu_features);
}
