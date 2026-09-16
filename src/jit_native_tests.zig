const std = @import("std");
const jit = @import("api/jit.zig");
const aot = jit.aot;
const wasm = @import("jit_fixture").bytes;
const equal = std.testing.expectEqual;
const expect = std.testing.expect;

const Pages = struct {
    live: usize = 0,
    rx: usize = 0,
    fn platform(self: *Pages) aot.Platform {
        return .{ .context = self, .reserve = reserve, .commit = commit, .protect = protect, .unmap = unmap, .monotonic_ns = clock };
    }
    fn reserve(ctx: *anyopaque, size: usize) aot.PlatformError![*]align(4096) u8 {
        const self: *Pages = @ptrCast(@alignCast(ctx));
        const bytes = std.posix.mmap(null, size, .{}, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0) catch return error.OutOfMemory;
        self.live += size;
        return @alignCast(bytes.ptr);
    }
    fn commit(_: *anyopaque, address: [*]align(4096) u8, size: usize) aot.PlatformError!void {
        if (std.posix.errno(std.posix.system.mprotect(address, size, .{ .READ = true, .WRITE = true })) != .SUCCESS) return error.OutOfMemory;
    }
    fn protect(ctx: *anyopaque, address: [*]align(4096) u8, size: usize, permission: aot.platform.Protection) aot.PlatformError!void {
        const self: *Pages = @ptrCast(@alignCast(ctx));
        if (permission != .read_execute) return error.ProtectionFailed;
        if (std.posix.errno(std.posix.system.mprotect(address, size, .{ .READ = true, .EXEC = true })) != .SUCCESS) return error.ProtectionFailed;
        self.rx += 1;
    }
    fn unmap(ctx: *anyopaque, address: [*]align(4096) u8, size: usize) void {
        const self: *Pages = @ptrCast(@alignCast(ctx));
        std.posix.munmap(address[0..size]);
        self.live -= size;
    }
    fn clock(_: *anyopaque) aot.PlatformError!u64 {
        var ts: std.os.linux.timespec = undefined;
        if (std.os.linux.clock_gettime(.MONOTONIC, &ts) != 0) return error.ClockFailed;
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }
};

fn compilerClock(_: ?*anyopaque) error{ClockFailed}!u64 {
    var unused: u8 = 0;
    return Pages.clock(&unused) catch return error.ClockFailed;
}

fn call(inst: *aot.Instance, name: []const u8, args: []const aot.Value) !i32 {
    var result: [1]aot.Value = undefined;
    try equal(@as(usize, 1), (try inst.call(name, args, &result)).returned);
    return result[0].i32;
}

fn reference(rounds: u32) u32 {
    var cells: [256]u32 = undefined;
    for (&cells, 0..) |*cell, i| cell.* = @as(u32, @intCast(i)) *% 17 +% 3;
    var sum: u32 = 0;
    for (0..rounds) |index| {
        const i: u32 = @intCast(index);
        const old = cells[i & 255];
        const value = if ((i & 1) == 0) old *% 3 +% i else old ^ (i *% 7);
        cells[i & 255] = value;
        sum +%= value;
    }
    return sum;
}

test "native JIT fast/full compile load run growth traps and repeated teardown" {
    for ([_]jit.PassPreset{ .fast, .full }) |preset| {
        var artifact = try jit.compile(std.testing.allocator, wasm, .{ .preset = preset, .monotonic_ns = compilerClock });
        defer artifact.deinit();
        try expect(artifact.metrics.parse_ns != null);
        try expect(artifact.metrics.lower_ns != null);
        try expect(artifact.metrics.optimize_ns != null);
        try expect(artifact.metrics.codegen_ns != null);
        try expect(artifact.metrics.emit_ns != null);
        try expect(artifact.metrics.code_bytes > 100);
        try expect(artifact.compiler_peak_bytes > artifact.bytes.len);
        try equal(artifact.bytes.len, artifact.compiler_retained_bytes);
        for (0..3) |_| {
            var pages: Pages = .{};
            {
                const inst = try aot.Instance.load(std.testing.allocator, pages.platform(), artifact.bytes, &.{}, jit.runtime_options);
                defer inst.deinit();
                try equal(@as(usize, 1), pages.rx);
                const stats = inst.memoryStats();
                try expect(stats.heap_peak_bytes <= jit.runtime_options.max_heap_bytes.?);
                try expect(stats.code_reserved_bytes + stats.linear_reserved_bytes <= jit.runtime_options.max_reserved_bytes.?);
                try equal(artifact.metrics.code_bytes, stats.code_bytes);
                try equal(@as(usize, 0), (try inst.start()).returned);
                try equal(@as(i32, @bitCast(reference(2000))), try call(inst, "workload", &.{.{ .i32 = 2000 }}));
                try equal(@as(i32, @bitCast(reference(2000))), try call(inst, "workload", &.{.{ .i32 = 2000 }}));
                const base = inst.memory().ptr;
                const old = try call(inst, "size", &.{});
                try equal(old, try call(inst, "grow", &.{.{ .i32 = 1 }}));
                try equal(base, inst.memory().ptr);
                try equal(@as(i32, 0), try call(inst, "load", &.{.{ .i32 = old * 65536 }}));
                try equal(@as(i32, -1), try call(inst, "grow", &.{.{ .i32 = 1000 }}));
                var result: [1]aot.Value = undefined;
                try equal(aot.Trap.out_of_bounds_memory, (try inst.call("load", &.{.{ .i32 = @intCast(inst.memory().len - 1) }}, &result)).trap);
            }

            try equal(@as(usize, 0), pages.live);
        }
    }
}

test "native JIT allocation-free infinite loop exhausts entry/backedge fuel" {
    for ([_]jit.PassPreset{ .fast, .full }) |preset| {
        var artifact = try jit.compile(std.testing.allocator, wasm, .{ .preset = preset });
        defer artifact.deinit();
        var pages: Pages = .{};
        var options = jit.runtime_options;
        options.max_run_fuel = 25;
        const inst = try aot.Instance.load(std.testing.allocator, pages.platform(), artifact.bytes, &.{}, options);
        defer inst.deinit();
        try equal(aot.Trap.fuel_exhausted, (try inst.call("spin", &.{}, &.{})).trap);
        try equal(@as(u32, 0), inst.vmctx.cancel_flag);
        try equal(@as(i32, 2), try call(inst, "size", &.{}));
        try equal(aot.Trap.fuel_exhausted, (try inst.call("spin", &.{}, &.{})).trap);
    }
}

test "native JIT loader rejects missing budgets and memory caps before execution" {
    var artifact = try jit.compile(std.testing.allocator, wasm, .{});
    defer artifact.deinit();
    var pages: Pages = .{};
    try std.testing.expectError(error.UnsupportedRunBudget, aot.Instance.load(std.testing.allocator, pages.platform(), artifact.bytes, &.{}, .{}));
    var options = jit.runtime_options;
    options.max_run_fuel = 0;
    try std.testing.expectError(error.InvalidLimits, aot.Instance.load(std.testing.allocator, pages.platform(), artifact.bytes, &.{}, options));
    options = jit.runtime_options;
    options.max_reserved_bytes = 1;
    try std.testing.expectError(error.MemoryLimitExceeded, aot.Instance.load(std.testing.allocator, pages.platform(), artifact.bytes, &.{}, options));
    options = jit.runtime_options;
    options.max_heap_bytes = @sizeOf(aot.Instance) + 1;
    try std.testing.expectError(error.OutOfMemory, aot.Instance.load(std.testing.allocator, pages.platform(), artifact.bytes, &.{}, options));
    try equal(@as(usize, 0), pages.live);
}

fn cancelled(_: ?*anyopaque) bool {
    return true;
}

test "native JIT budgets fail explicitly with cleanup" {
    try std.testing.expectError(error.InputLimitExceeded, jit.compile(std.testing.allocator, wasm, .{ .max_input_bytes = 8 }));
    try std.testing.expectError(error.CompilerMemoryLimitExceeded, jit.compile(std.testing.allocator, wasm, .{ .max_compiler_bytes = 32 }));
    try std.testing.expectError(error.CompileLimitExceeded, jit.compile(std.testing.allocator, wasm, .{ .max_polls = 1 }));
    try std.testing.expectError(error.CompileCancelled, jit.compile(std.testing.allocator, wasm, .{ .cancelled = cancelled }));
    try std.testing.expectError(error.InvalidLimits, jit.compile(std.testing.allocator, wasm, .{ .deadline_ns = 1 }));
    try std.testing.expectError(error.CompileCancelled, jit.compile(std.testing.allocator, wasm, .{ .deadline_ns = 0, .monotonic_ns = compilerClock }));
    try std.testing.expectError(error.CodeLimitExceeded, jit.compile(std.testing.allocator, wasm, .{ .max_code_bytes = 1 }));
    try std.testing.expectError(error.CompileLimitExceeded, jit.compile(std.testing.allocator, wasm, .{ .max_blocks_per_function = 1 }));
    try std.testing.expectError(error.CompileLimitExceeded, jit.compile(std.testing.allocator, wasm, .{ .max_function_bytes = 1 }));
}

fn compileAllocationProbe(allocator: std.mem.Allocator) !void {
    var artifact = try jit.compile(allocator, wasm, .{});
    defer artifact.deinit();
}

test "native JIT reclaims scratch at every allocator failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, compileAllocationProbe, .{});
}

const DelayedCancellation = struct {
    count: usize = 0,
    fn poll(ctx: ?*anyopaque) bool {
        const self: *DelayedCancellation = @ptrCast(@alignCast(ctx.?));
        self.count += 1;
        return self.count == 30;
    }
};

test "native JIT cooperative cancellation interrupts an active compilation" {
    var state: DelayedCancellation = .{};
    try std.testing.expectError(error.CompileCancelled, jit.compile(std.testing.allocator, wasm, .{ .context = &state, .cancelled = DelayedCancellation.poll }));
    try equal(@as(usize, 30), state.count);
}

fn add(_: ?*anyopaque, _: *aot.HostContext, args: []const aot.Value, results: []aot.Value) aot.HostError!void {
    results[0] = .{ .i32 = args[0].i32 +% args[1].i32 };
}

test "native JIT preserves checked scalar imports under both presets" {
    const imported =
        "\x00asm\x01\x00\x00\x00" ++
        "\x01\x07\x01\x60\x02\x7f\x7f\x01\x7f" ++
        "\x02\x10\x01\x03env\x08host_add\x00\x00" ++
        "\x03\x02\x01\x00" ++
        "\x07\x07\x01\x03add\x00\x01" ++
        "\x0a\x0a\x01\x08\x00\x20\x00\x20\x01\x10\x00\x0b";
    const imports = [_]aot.HostImport{.{
        .module = "env",
        .name = "host_add",
        .params = &.{ .i32, .i32 },
        .results = &.{.i32},
        .callback = add,
    }};
    for ([_]jit.PassPreset{ .fast, .full }) |preset| {
        var artifact = try jit.compile(std.testing.allocator, imported, .{ .preset = preset });
        defer artifact.deinit();
        var pages: Pages = .{};
        try std.testing.expectError(error.MissingImport, aot.Instance.load(std.testing.allocator, pages.platform(), artifact.bytes, &.{}, jit.runtime_options));
        const inst = try aot.Instance.load(std.testing.allocator, pages.platform(), artifact.bytes, &imports, jit.runtime_options);
        defer inst.deinit();
        try equal(@as(i32, 42), try call(inst, "add", &.{ .{ .i32 = 20 }, .{ .i32 = 22 } }));
    }
}

test "native JIT rejects an oversized generated native frame" {
    const large_frame =
        "\x00asm\x01\x00\x00\x00" ++
        "\x01\x04\x01\x60\x00\x00" ++
        "\x03\x02\x01\x00" ++
        "\x0a\x07\x01\x05\x01\xd0\x0f\x7f\x0b";
    try std.testing.expectError(error.UnsupportedNativeFeature, jit.compile(std.testing.allocator, large_frame, .{}));
}

test "native JIT bounded load preserves shared phase instrumentation" {
    var artifact = try jit.compile(std.testing.allocator, wasm, .{ .monotonic_ns = compilerClock });
    defer artifact.deinit();
    var pages: Pages = .{};
    var timings: aot.LoadTimings = .{};
    var options = jit.runtime_options;
    options.timings = &timings;
    {
        const inst = try aot.Instance.load(std.testing.allocator, pages.platform(), artifact.bytes, &.{}, options);
        defer inst.deinit();
        try equal(@as(u32, 3), timings.completed);
        try expect(inst.memoryStats().heap_peak_bytes <= options.max_heap_bytes.?);
        try equal(@as(i32, @bitCast(reference(2000))), try call(inst, "workload", &.{.{ .i32 = 2000 }}));
    }
    try equal(@as(usize, 0), pages.live);
    options.max_run_fuel = null;
    try std.testing.expectError(error.UnsupportedRunBudget, aot.Instance.load(std.testing.allocator, pages.platform(), artifact.bytes, &.{}, options));
    try equal(@as(u32, 0), timings.completed);
    try equal(@as(usize, 0), pages.live);
}
