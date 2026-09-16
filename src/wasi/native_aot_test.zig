const std = @import("std");
const minimal = @import("minimal-wasi");
const adapter = @import("native_aot.zig");
const testing = std.testing;

// This contract fixture tests marshalling, not native stack unwinding or AOT
// execution. The real backend must separately test its pending-exit dispatcher.
const NativeContract = struct {
    pub const ValType = enum { i32, i64, f32, f64 };
    pub const Value = union(ValType) { i32: i32, i64: i64, f32: f32, f64: f64 };
    pub const HostError = error{ Unsupported, InvalidArgument, Io, OutOfMemory };
    pub const HostImport = struct {
        module: []const u8,
        name: []const u8,
        params: []const ValType,
        results: []const ValType,
        context: ?*anyopaque = null,
        callback: *const fn (?*anyopaque, *HostContext, []const Value, []Value) HostError!void,
    };
    pub const HostContext = struct {
        bytes: []u8,
        pending_exit: ?u32 = null,
        memory_lookups: usize = 0,
        termination_requests: usize = 0,

        pub fn memory(self: *HostContext) []u8 {
            self.memory_lookups += 1;
            return self.bytes;
        }

        pub fn terminate(self: *HostContext, code: u32) void {
            self.termination_requests += 1;
            if (self.pending_exit == null) self.pending_exit = code;
        }
    };
};

const Bindings = adapter.Adapter(NativeContract);

fn context() minimal.Context {
    return minimal.Context.init(.{ .args = &.{"coremark"}, .environment = &.{"A=b"}, .output = null, .clock = null }) catch unreachable;
}

fn binding(imports: []const NativeContract.HostImport, function: minimal.Function) NativeContract.HostImport {
    return imports[@intFromEnum(function)];
}

fn run(imports: []const NativeContract.HostImport, host: *NativeContract.HostContext, function: minimal.Function, args: []const NativeContract.Value) !i32 {
    const entry = binding(imports, function);
    var results = [_]NativeContract.Value{.{ .i32 = -1 }};
    try entry.callback(entry.context, host, args, &results);
    return results[0].i32;
}

test "native WASI adapter binds all exact snapshot-0 signatures" {
    var ctx = context();
    const imports = Bindings.imports(&ctx);
    try testing.expectEqual(@as(usize, 12), imports.len);
    for (imports, minimal.imports) |actual, expected| {
        try testing.expectEqualStrings(expected.namespace, actual.module);
        try testing.expectEqualStrings(expected.name, actual.name);
        try testing.expectEqual(@as(?*anyopaque, &ctx), actual.context);
        try testing.expectEqual(expected.params.len, actual.params.len);
        for (expected.params, actual.params) |byte, kind|
            try testing.expectEqual(if (byte == 0x7f) NativeContract.ValType.i32 else NativeContract.ValType.i64, kind);
        try testing.expectEqual(expected.results.len, actual.results.len);
        for (actual.results) |kind| try testing.expectEqual(NativeContract.ValType.i32, kind);
    }
}

test "native WASI adapter validates value shapes before touching guest memory" {
    var ctx = context();
    const imports = Bindings.imports(&ctx);
    var mem = [_]u8{0xaa} ** 32;
    var host: NativeContract.HostContext = .{ .bytes = &mem };
    const entry = binding(&imports, .args_sizes_get);
    var results = [_]NativeContract.Value{.{ .i32 = -1 }};
    try testing.expectError(error.InvalidArgument, entry.callback(null, &host, &.{ .{ .i32 = 0 }, .{ .i32 = 4 } }, &results));
    try testing.expectError(error.InvalidArgument, entry.callback(entry.context, &host, &.{.{ .i32 = 0 }}, &results));
    try testing.expectError(error.InvalidArgument, entry.callback(entry.context, &host, &.{ .{ .f32 = 0 }, .{ .i32 = 4 } }, &results));
    try testing.expectError(error.InvalidArgument, entry.callback(entry.context, &host, &.{ .{ .i32 = 0 }, .{ .i32 = 4 } }, &.{}));
    try testing.expectError(error.InvalidArgument, run(&imports, &host, .fd_seek, &.{ .{ .i32 = 1 }, .{ .i32 = 0 }, .{ .i32 = 2 }, .{ .i32 = 0 } }));
    try testing.expectEqual(@as(usize, 0), host.memory_lookups);
    for (mem) |byte| try testing.expectEqual(@as(u8, 0xaa), byte);
    try testing.expectEqual(@as(i32, -1), results[0].i32);
}

test "native WASI adapter refreshes memory and preserves scalar bit patterns" {
    var ctx = context();
    const imports = Bindings.imports(&ctx);
    var small = [_]u8{0xaa} ** 4;
    var grown = [_]u8{0xaa} ** 64;
    var host: NativeContract.HostContext = .{ .bytes = &small };
    const size_args = [_]NativeContract.Value{ .{ .i32 = 0 }, .{ .i32 = 4 } };
    try testing.expectEqual(@as(i32, 21), try run(&imports, &host, .args_sizes_get, &size_args));
    host.bytes = &grown;
    try testing.expectEqual(@as(i32, 0), try run(&imports, &host, .args_sizes_get, &size_args));
    try testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, grown[0..4], .little));
    try testing.expectEqual(@as(u32, 9), std.mem.readInt(u32, grown[4..8], .little));
    try testing.expectEqual(@as(i32, 0), try run(&imports, &host, .args_get, &.{ .{ .i32 = 8 }, .{ .i32 = 16 } }));
    try testing.expectEqualStrings("coremark\x00", grown[16..25]);
    try testing.expectEqual(@as(i32, 0), try run(&imports, &host, .environ_sizes_get, &size_args));
    try testing.expectEqual(@as(u32, 4), std.mem.readInt(u32, grown[4..8], .little));
    try testing.expectEqual(@as(i32, 0), try run(&imports, &host, .environ_get, &.{ .{ .i32 = 8 }, .{ .i32 = 16 } }));
    try testing.expectEqualStrings("A=b\x00", grown[16..20]);
    try testing.expectEqual(@as(i32, 8), try run(&imports, &host, .fd_close, &.{.{ .i32 = -1 }}));
    try testing.expectEqual(@as(i32, 8), try run(&imports, &host, .fd_prestat_get, &.{ .{ .i32 = 3 }, .{ .i32 = -1 } }));
    try testing.expectEqual(@as(i32, 8), try run(&imports, &host, .fd_prestat_dir_name, &.{ .{ .i32 = 3 }, .{ .i32 = -1 }, .{ .i32 = -1 } }));
    try testing.expectEqual(@as(i32, 21), try run(&imports, &host, .fd_fdstat_get, &.{ .{ .i32 = 1 }, .{ .i32 = -1 } }));
    try testing.expectEqual(@as(i32, 76), try run(&imports, &host, .fd_seek, &.{ .{ .i32 = 1 }, .{ .i64 = -1 }, .{ .i32 = 2 }, .{ .i32 = 32 } }));
    try testing.expectEqual(@as(i32, 76), try run(&imports, &host, .fd_write, &.{ .{ .i32 = 1 }, .{ .i32 = 0 }, .{ .i32 = 0 }, .{ .i32 = 32 } }));
    try testing.expectEqual(@as(u32, 0xaaaa_aaaa), std.mem.readInt(u32, &small, .little));
}

const PrecisionClock = struct {
    precision: u64 = 0,

    fn read(raw: ?*anyopaque, id: minimal.ClockId, precision: u64) minimal.ClockResult {
        const self: *PrecisionClock = @ptrCast(@alignCast(raw.?));
        std.debug.assert(id == .monotonic);
        self.precision = precision;
        return .{ .timestamp_ns = 1234 };
    }
};

test "native WASI adapter preserves 64-bit clock precision" {
    var ctx = context();
    var clock: PrecisionClock = .{};
    ctx.clock = .{ .userdata = &clock, .resolution_ns = .{ 0, 100, 0, 0 }, .read = PrecisionClock.read };
    const imports = Bindings.imports(&ctx);
    var mem = [_]u8{0xaa} ** 8;
    var host: NativeContract.HostContext = .{ .bytes = &mem };
    try testing.expectEqual(@as(i32, 0), try run(&imports, &host, .clock_time_get, &.{ .{ .i32 = 1 }, .{ .i64 = -1 }, .{ .i32 = 0 } }));
    try testing.expectEqual(std.math.maxInt(u64), clock.precision);
    try testing.expectEqual(@as(u64, 1234), std.mem.readInt(u64, &mem, .little));
}

const Capture = struct {
    bytes: [16]u8 = undefined,
    count: usize = 0,

    fn write(raw: ?*anyopaque, fd: u32, bytes: []const u8) minimal.WriteResult {
        const self: *Capture = @ptrCast(@alignCast(raw.?));
        std.debug.assert(fd == 2);
        const count = @min(bytes.len, 2);
        @memcpy(self.bytes[0..count], bytes[0..count]);
        self.count = count;
        return .{ .written = count, .errno = .pipe };
    }
};

test "native WASI adapter preserves actual partial output and callback error" {
    var ctx = context();
    var capture: Capture = .{};
    ctx.output = .{ .userdata = &capture, .write = Capture.write };
    const imports = Bindings.imports(&ctx);
    var mem = [_]u8{0xaa} ** 32;
    std.mem.writeInt(u32, mem[0..4], 16, .little);
    std.mem.writeInt(u32, mem[4..8], 4, .little);
    @memcpy(mem[16..20], "a\x00bc");
    var host: NativeContract.HostContext = .{ .bytes = &mem };
    try testing.expectEqual(@as(i32, 64), try run(&imports, &host, .fd_write, &.{ .{ .i32 = 2 }, .{ .i32 = 0 }, .{ .i32 = 1 }, .{ .i32 = 8 } }));
    try testing.expectEqualStrings("a\x00", capture.bytes[0..capture.count]);
    try testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, mem[8..12], .little));
    try testing.expectEqual(null, host.pending_exit);
}

test "native WASI adapter requests zero and nonzero exits then returns without errno" {
    for ([_]i32{ 0, 7, -1 }) |code| {
        var ctx = context();
        const imports = Bindings.imports(&ctx);
        var mem = [_]u8{0xaa} ** 8;
        var host: NativeContract.HostContext = .{ .bytes = &mem };
        const entry = binding(&imports, .proc_exit);
        var callback_returned = false;
        {
            defer callback_returned = true;
            try entry.callback(entry.context, &host, &.{.{ .i32 = code }}, &.{});
        }
        try testing.expect(callback_returned);
        try testing.expectEqual(@as(u32, @bitCast(code)), host.pending_exit.?);
        try testing.expectEqual(@as(usize, 1), host.termination_requests);
        try testing.expectEqual(@as(i32, -1), try run(&imports, &host, .fd_close, &.{.{ .i32 = 1 }}));
        try testing.expect(ctx.descriptors[1].open);
        try testing.expectEqual(@as(u32, @bitCast(code)), host.pending_exit.?);
    }
}
