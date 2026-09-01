const std = @import("std");
const wabt = @import("wabt");

const AtomicMarkers = struct {
    loads: usize,
    stores: usize,
    rmw_adds: usize,
    cmpxchgs: usize,
    wait32s: usize,
};

fn replaceAll(
    allocator: std.mem.Allocator,
    input: []const u8,
    needle: []const u8,
    replacement: []const u8,
) ![]u8 {
    var output: std.ArrayListUnmanaged(u8) = .empty;
    errdefer output.deinit(allocator);
    var cursor: usize = 0;
    while (std.mem.indexOfPos(u8, input, cursor, needle)) |found| {
        try output.appendSlice(allocator, input[cursor..found]);
        try output.appendSlice(allocator, replacement);
        cursor = found + needle.len;
    }
    try output.appendSlice(allocator, input[cursor..]);
    return output.toOwnedSlice(allocator);
}

fn rewriteAtomicText(
    allocator: std.mem.Allocator,
    source: []const u8,
) !struct { source: []u8, markers: AtomicMarkers } {
    const markers = AtomicMarkers{
        .loads = std.mem.count(u8, source, "i32.atomic.load"),
        .stores = std.mem.count(u8, source, "i32.atomic.store"),
        .rmw_adds = std.mem.count(u8, source, "i32.atomic.rmw.add"),
        .cmpxchgs = std.mem.count(u8, source, "i32.atomic.rmw.cmpxchg"),
        .wait32s = std.mem.count(u8, source, "memory.atomic.wait32"),
    };
    // `memory.atomic.wait32` consumes [addr:i32, expected:i32, timeout:i64]
    // and produces i32; `drop drop` has the same stack effect and leaves a
    // recognisable byte pattern.
    var rewritten = try replaceAll(
        allocator,
        source,
        "memory.atomic.wait32",
        "nop nop nop drop drop",
    );
    errdefer allocator.free(rewritten);
    var next = try replaceAll(
        allocator,
        rewritten,
        "i32.atomic.rmw.cmpxchg",
        "drop drop i32.load",
    );
    allocator.free(rewritten);
    rewritten = next;
    next = try replaceAll(
        allocator,
        rewritten,
        "i32.atomic.rmw.add",
        "nop nop nop i32.rotr",
    );
    allocator.free(rewritten);
    rewritten = next;
    next = try replaceAll(
        allocator,
        rewritten,
        "i32.atomic.load",
        "nop i32.load",
    );
    allocator.free(rewritten);
    rewritten = next;
    next = try replaceAll(
        allocator,
        rewritten,
        "i32.atomic.store",
        "nop nop i32.store",
    );
    allocator.free(rewritten);
    return .{ .source = next, .markers = markers };
}

fn lowerAtomicMarkers(
    allocator: std.mem.Allocator,
    module: *wabt.Module.Module,
    expected: AtomicMarkers,
) !void {
    var actual = AtomicMarkers{
        .loads = 0,
        .stores = 0,
        .rmw_adds = 0,
        .cmpxchgs = 0,
        .wait32s = 0,
    };
    for (module.funcs.items) |*function| {
        if (function.is_import or function.code_bytes.len == 0) continue;
        var output: std.ArrayListUnmanaged(u8) = .empty;
        errdefer output.deinit(allocator);
        var i: usize = 0;
        while (i < function.code_bytes.len) : (i += 1) {
            const remaining = function.code_bytes[i..];
            if (std.mem.startsWith(u8, remaining, &.{ 0x01, 0x01, 0x01, 0x1A, 0x1A })) {
                // memory.atomic.wait32 align=2 (4-byte) offset=0
                try output.appendSlice(allocator, &.{ 0xfe, 0x01, 0x02, 0x00 });
                i += 4;
                actual.wait32s += 1;
            } else if (std.mem.startsWith(u8, remaining, &.{ 0x01, 0x01, 0x01, 0x78 })) {
                try output.appendSlice(allocator, &.{ 0xfe, 0x1e, 0x02, 0x00 });
                i += 3;
                actual.rmw_adds += 1;
            } else if (std.mem.startsWith(u8, remaining, &.{ 0x1a, 0x1a, 0x28, 0x02, 0x00 })) {
                try output.appendSlice(allocator, &.{ 0xfe, 0x48, 0x02, 0x00 });
                i += 4;
                actual.cmpxchgs += 1;
            } else if (std.mem.startsWith(u8, remaining, &.{ 0x01, 0x01, 0x36, 0x02 })) {
                try output.appendSlice(allocator, &.{ 0xfe, 0x17, 0x02 });
                i += 3;
                actual.stores += 1;
            } else if (std.mem.startsWith(u8, remaining, &.{ 0x01, 0x28, 0x02 })) {
                try output.appendSlice(allocator, &.{ 0xfe, 0x10, 0x02 });
                i += 2;
                actual.loads += 1;
            } else {
                try output.append(allocator, function.code_bytes[i]);
            }
        }
        if (function.owns_code_bytes) allocator.free(function.code_bytes);
        function.code_bytes = try output.toOwnedSlice(allocator);
        function.owns_code_bytes = true;
    }
    if (actual.loads != expected.loads or
        actual.stores != expected.stores or
        actual.rmw_adds != expected.rmw_adds or
        actual.cmpxchgs != expected.cmpxchgs or
        actual.wait32s != expected.wait32s)
    {
        std.debug.print(
            "atomic marker mismatch: expected {d}/{d}/{d}/{d}/{d}, found {d}/{d}/{d}/{d}/{d}\n",
            .{
                expected.loads,
                expected.stores,
                expected.rmw_adds,
                expected.cmpxchgs,
                expected.wait32s,
                actual.loads,
                actual.stores,
                actual.rmw_adds,
                actual.cmpxchgs,
                actual.wait32s,
            },
        );
        return error.AtomicMarkerMismatch;
    }
}

pub fn main(init: std.process.Init) !u8 {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3 or args.len % 2 == 0) {
        std.debug.print(
            "usage: generate-wasi-thread-fixtures <input.wat> <output.wasm> [...]\n",
            .{},
        );
        return 2;
    }

    const cwd = std.Io.Dir.cwd();
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        const source = try cwd.readFileAlloc(
            init.io,
            args[i],
            allocator,
            std.Io.Limit.limited(wabt.max_input_file_size),
        );
        defer allocator.free(source);

        const rewritten = try rewriteAtomicText(allocator, source);
        defer allocator.free(rewritten.source);
        var module = wabt.text.Parser.parseModule(allocator, rewritten.source) catch |err| {
            std.debug.print("{s}: parse failed: {s}\n", .{ args[i], @errorName(err) });
            return 1;
        };
        defer module.deinit();

        // cataggar/wabt currently parses thread opcodes but does not accept
        // the WAT `shared` limits keyword. Keep the checked-in source readable
        // and set the corresponding binary-format bit before validation.
        for (module.memories.items) |*memory| {
            memory.type.limits.is_shared = true;
            if (!memory.type.limits.has_max) return error.SharedMemoryNeedsMaximum;
        }
        for (module.imports.items) |*import| {
            if (import.memory) |*memory| {
                memory.limits.is_shared = true;
                if (!memory.limits.has_max) return error.SharedMemoryNeedsMaximum;
            }
        }
        wabt.Validator.validate(&module, .{}) catch |err| {
            std.debug.print(
                "{s}: placeholder validation failed: {s}\n",
                .{ args[i], @errorName(err) },
            );
            return 1;
        };
        try lowerAtomicMarkers(allocator, &module, rewritten.markers);
        const wasm = wabt.binary.writer.writeModule(allocator, &module) catch |err| {
            std.debug.print("{s}: write failed: {s}\n", .{ args[i], @errorName(err) });
            return 1;
        };
        defer allocator.free(wasm);
        try cwd.writeFile(init.io, .{ .sub_path = args[i + 1], .data = wasm });
    }
    return 0;
}
