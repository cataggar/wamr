var cells: [256]u32 = @splat(0);

export fn workload(rounds: u32) u32 {
    for (&cells, 0..) |*cell, i| {
        const p: *volatile u32 = cell;
        p.* = @as(u32, @intCast(i)) *% 17 +% 3;
    }
    var sum: u32 = 0;
    var i: u32 = 0;
    while (i < rounds) : (i += 1) {
        const p: *volatile u32 = &cells[i & 255];
        const old = p.*;
        const value = if ((i & 1) == 0) old *% 3 +% i else old ^ (i *% 7);
        p.* = value;
        sum +%= value;
    }
    return sum;
}

export fn grow(pages: u32) i32 {
    return @bitCast(@wasmMemoryGrow(0, pages));
}

export fn size() u32 {
    return @wasmMemorySize(0);
}

export fn load(address: u32) u32 {
    return @as(*align(1) volatile u32, @ptrFromInt(address)).*;
}

export fn spin() void {
    while (true) {}
}
