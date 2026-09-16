var cells: [257]u32 = undefined;

export fn compute(seed: u32) u32 {
    var value = seed;
    for (0..4096) |i| value = (value *% 1664525 +% 1013904223) ^ @as(u32, @intCast(i));
    return value;
}

export fn memory_checksum(seed: u32) u32 {
    for (&cells, 0..) |*cell, i| cell.* = seed +% @as(u32, @intCast(i * 17));
    var checksum: u32 = 0;
    for (&cells, 0..) |*cell, i| {
        if (cell.* != seed +% @as(u32, @intCast(i * 17))) unreachable;
        checksum +%= cell.*;
    }
    return checksum;
}

export fn _start() void {
    if (compute(42) != 0x3001802a or memory_checksum(42) != 570026) unreachable;
}
