const std = @import("std");
const api = @import("api/aot.zig");
const c_api = @import("aot_native.zig");
const fixture = @import("native_fixture").bytes;
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

const Pages = struct {
    live: usize = 0,
    reservations: usize = 0,
    commits: usize = 0,
    protects: usize = 0,
    fail_reserve: ?usize = null,
    fail_commit: ?usize = null,
    fail_protect: bool = false,
    code_executable: bool = false,
    clock_value: u64 = 123456789,
    clock_step: u64 = 0,
    clock_reads: usize = 0,
    clock_fail_on: ?usize = null,
    clock_rewind_after_first: bool = false,
    check_phase_boundaries: bool = false,

    fn platform(self: *Pages) api.Platform {
        return .{ .context = self, .reserve = reserve, .commit = commit, .protect = protect, .unmap = unmap, .monotonic_ns = clock };
    }
    fn reserve(context: *anyopaque, size: usize) api.PlatformError![*]align(4096) u8 {
        const self: *Pages = @ptrCast(@alignCast(context));
        const index = self.reservations;
        self.reservations += 1;
        if (self.fail_reserve == index) return error.OutOfMemory;
        const slice = std.posix.mmap(null, size, .{}, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0) catch return error.OutOfMemory;
        self.live += 1;
        return @alignCast(slice.ptr);
    }
    fn commit(context: *anyopaque, address: [*]align(4096) u8, size: usize) api.PlatformError!void {
        const self: *Pages = @ptrCast(@alignCast(context));
        const index = self.commits;
        self.commits += 1;
        if (self.fail_commit == index) return error.OutOfMemory;
        if (std.posix.errno(std.posix.system.mprotect(address, size, .{ .READ = true, .WRITE = true })) != .SUCCESS) return error.OutOfMemory;
    }
    fn protect(context: *anyopaque, address: [*]align(4096) u8, size: usize, permission: api.platform.Protection) api.PlatformError!void {
        const self: *Pages = @ptrCast(@alignCast(context));
        self.protects += 1;
        if (self.fail_protect) return error.ProtectionFailed;
        const prot: std.posix.PROT = switch (permission) {
            .none => .{},
            .read_write => .{ .READ = true, .WRITE = true },
            .read_execute => .{ .READ = true, .EXEC = true },
        };
        if (std.posix.errno(std.posix.system.mprotect(address, size, prot)) != .SUCCESS) return error.ProtectionFailed;
        if (permission == .read_execute) self.code_executable = true;
    }
    fn unmap(context: *anyopaque, address: [*]align(4096) u8, size: usize) void {
        const self: *Pages = @ptrCast(@alignCast(context));
        std.posix.munmap(address[0..size]);
        self.live -= 1;
    }
    fn clock(context: *anyopaque) api.PlatformError!u64 {
        const self: *Pages = @ptrCast(@alignCast(context));
        const index = self.clock_reads;
        self.clock_reads += 1;
        if (self.clock_fail_on == index) return error.ClockFailed;
        if (self.check_phase_boundaries) {
            if (index < 2 and self.reservations != 0) return error.ClockFailed;
            if (index == 2 and (self.reservations != 2 or self.protects != 1)) return error.ClockFailed;
        }
        const value = self.clock_value;
        self.clock_value = if (index == 0 and self.clock_rewind_after_first) value - 1 else value + self.clock_step;
        return value;
    }
};
const imports = [_]api.HostImport{
    .{ .module = "env", .name = "host_add", .params = &.{ .i32, .i32 }, .results = &.{.i32}, .callback = hostAdd },
    .{ .module = "env", .name = "host_exit", .params = &.{.i32}, .results = &.{}, .callback = hostExit },
    .{ .module = "env", .name = "host_fail", .params = &.{}, .results = &.{}, .callback = hostFail },
};
fn hostAdd(_: ?*anyopaque, ctx: *api.HostContext, args: []const api.Value, results: []api.Value) api.HostError!void {
    if ((ctx.monotonicNs() catch return error.Io) != 123456789) return error.Io;
    results[0] = .{ .i32 = args[0].i32 +% args[1].i32 };
}
fn hostExit(_: ?*anyopaque, context: *api.HostContext, args: []const api.Value, _: []api.Value) api.HostError!void {
    defer std.mem.writeInt(u32, context.memory()[68..72], 0xdefe, .little);
    context.terminate(@bitCast(args[0].i32));
}
fn hostFail(_: ?*anyopaque, _: *api.HostContext, _: []const api.Value, _: []api.Value) api.HostError!void {
    return error.Io;
}
fn create(allocator: std.mem.Allocator, pages: *Pages) !*api.Instance {
    return api.Instance.load(allocator, pages.platform(), fixture, &imports, .{ .max_memory_pages = 8 });
}
fn callI32(inst: *api.Instance, name: []const u8, args: []const api.Value) !i32 {
    var result: [1]api.Value = undefined;
    try equal(@as(usize, 1), (try inst.call(name, args, &result)).returned);
    return result[0].i32;
}

test "native AOT real API: add noop host import and checked signatures" {
    var pages: Pages = .{};
    const inst = try create(std.testing.allocator, &pages);
    defer inst.deinit();
    try expect(pages.code_executable);
    try equal(@as(usize, 1), pages.protects);
    try equal(@as(usize, 0), (try inst.start()).returned);
    try equal(@as(i32, 42), try callI32(inst, "add", &.{ .{ .i32 = 19 }, .{ .i32 = 23 } }));
    try equal(@as(i32, 42), try callI32(inst, "call_host", &.{ .{ .i32 = 20 }, .{ .i32 = 22 } }));
    try equal(@as(usize, 0), (try inst.call("noop", &.{}, &.{})).returned);
    var result: [1]api.Value = undefined;
    try std.testing.expectError(error.ArgumentCountMismatch, inst.call("add", &.{}, &result));
    try std.testing.expectError(error.ArgumentTypeMismatch, inst.call("add", &.{ .{ .i64 = 1 }, .{ .i32 = 2 } }, &result));
    try std.testing.expectError(error.ResultBufferTooSmall, inst.call("add", &.{ .{ .i32 = 1 }, .{ .i32 = 2 } }, &.{}));
    try std.testing.expectError(error.FunctionNotFound, inst.call("missing", &.{}, &.{}));
    try equal(@as(usize, 1), (try inst.call("add64", &.{ .{ .i64 = 0x100000000 }, .{ .i64 = 42 } }, &result)).returned);
    try equal(@as(i64, 0x10000002a), result[0].i64);
    try equal(@as(usize, 1), (try inst.call("add_float", &.{ .{ .f64 = 1.25 }, .{ .f64 = 2.5 } }, &result)).returned);
    try equal(@as(f64, 3.75), result[0].f64);
    try equal(@as(i32, 42), try callI32(inst, "indirect", &.{ .{ .i32 = 0 }, .{ .i32 = 21 } }));
    try equal(@as(i32, 42), try callI32(inst, "indirect", &.{ .{ .i32 = 1 }, .{ .i32 = 14 } }));
}
test "native AOT real API: memory bounds stable growth and deterministic traps" {
    var pages: Pages = .{};
    const inst = try create(std.testing.allocator, &pages);
    defer inst.deinit();
    const original = inst.memory().ptr;
    const old = try callI32(inst, "size", &.{});
    try equal(@as(usize, 0), (try inst.call("store", &.{ .{ .i32 = 64 }, .{ .i32 = 0x12345678 } }, &.{})).returned);
    try equal(@as(i32, 0x12345678), try callI32(inst, "load", &.{.{ .i32 = 64 }}));
    try equal(old, try callI32(inst, "grow", &.{.{ .i32 = 1 }}));
    try equal(original, inst.memory().ptr);
    try equal(@as(i32, 0x12345678), try callI32(inst, "load", &.{.{ .i32 = 64 }}));
    try equal(@as(i32, 0), try callI32(inst, "load", &.{.{ .i32 = old * 65536 }}));
    try equal(@as(i32, -1), try callI32(inst, "grow", &.{.{ .i32 = -1 }}));
    pages.fail_commit = pages.commits;
    try equal(@as(i32, -1), try callI32(inst, "grow", &.{.{ .i32 = 1 }}));
    try equal(old + 1, try callI32(inst, "size", &.{}));
    var result: [1]api.Value = undefined;
    try equal(api.Trap.out_of_bounds_memory, (try inst.call("load", &.{.{ .i32 = @intCast(inst.memory().len - 3) }}, &result)).trap);
    try equal(api.Trap.out_of_bounds_memory, (try inst.call("store", &.{ .{ .i32 = -1 }, .{ .i32 = 1 } }, &.{})).trap);
    try equal(api.Trap.unreachable_instruction, (try inst.call("trap", &.{}, &.{})).trap);
    try equal(api.Trap.integer_divide_by_zero, (try inst.call("divide", &.{ .{ .i32 = 4 }, .{ .i32 = 0 } }, &result)).trap);
    try equal(api.Trap.integer_overflow, (try inst.call("divide", &.{ .{ .i32 = std.math.minInt(i32) }, .{ .i32 = -1 } }, &result)).trap);
    // A trap does not poison subsequent calls or leak the call frame.
    try equal(@as(i32, 3), try callI32(inst, "add", &.{ .{ .i32 = 1 }, .{ .i32 = 2 } }));
}
test "native AOT integer division preserves normal results and every trap ABI" {
    var pages: Pages = .{};
    const inst = try create(std.testing.allocator, &pages);
    defer inst.deinit();
    try equal(@as(i32, -2), try callI32(inst, "divide", &.{ .{ .i32 = -7 }, .{ .i32 = 3 } }));
    try equal(@as(i32, 0x7fffffff), try callI32(inst, "divide_unsigned", &.{ .{ .i32 = -1 }, .{ .i32 = 2 } }));
    try equal(@as(i32, -1), try callI32(inst, "remainder", &.{ .{ .i32 = -7 }, .{ .i32 = 3 } }));
    try equal(@as(i32, 0), try callI32(inst, "remainder", &.{ .{ .i32 = std.math.minInt(i32) }, .{ .i32 = -1 } }));
    try equal(@as(i32, 5), try callI32(inst, "remainder_unsigned", &.{ .{ .i32 = -1 }, .{ .i32 = 10 } }));
    var result: [1]api.Value = undefined;
    for ([_][]const u8{ "divide", "divide_unsigned", "remainder", "remainder_unsigned" }) |name| {
        try equal(api.Trap.integer_divide_by_zero, (try inst.call(name, &.{ .{ .i32 = 4 }, .{ .i32 = 0 } }, &result)).trap);
    }
    try equal(@as(usize, 1), (try inst.call("divide64", &.{ .{ .i64 = -7 }, .{ .i64 = 3 } }, &result)).returned);
    try equal(@as(i64, -2), result[0].i64);
    try equal(@as(usize, 1), (try inst.call("remainder64", &.{ .{ .i64 = std.math.minInt(i64) }, .{ .i64 = -1 } }, &result)).returned);
    try equal(@as(i64, 0), result[0].i64);
    try equal(api.Trap.integer_overflow, (try inst.call("divide64", &.{ .{ .i64 = std.math.minInt(i64) }, .{ .i64 = -1 } }, &result)).trap);
    try equal(api.Trap.integer_divide_by_zero, (try inst.call("divide64", &.{ .{ .i64 = 4 }, .{ .i64 = 0 } }, &result)).trap);
}

test "native AOT real API: host exit unwinds after host cleanup and errors are terminal" {
    var pages: Pages = .{};
    const inst = try create(std.testing.allocator, &pages);
    defer inst.deinit();
    try equal(@as(u32, 17), (try inst.call("exit", &.{.{ .i32 = 17 }}, &.{})).exit);
    try equal(@as(u32, 0), std.mem.readInt(u32, inst.memory()[64..68], .little));
    try equal(@as(u32, 0xdefe), std.mem.readInt(u32, inst.memory()[68..72], .little));
    try equal(@as(u32, 0), (try inst.call("exit", &.{.{ .i32 = 0 }}, &.{})).exit);
    try equal(@as(u32, 0), std.mem.readInt(u32, inst.memory()[64..68], .little));
    try equal(error.Io, (try inst.call("fail", &.{}, &.{})).host_error);
    try equal(@as(u32, 0), std.mem.readInt(u32, inst.memory()[64..68], .little));
}
test "native AOT rejects incompatible artifact and unresolved imports before page allocation" {
    var pages: Pages = .{};
    try std.testing.expectError(error.MissingImport, api.Instance.load(std.testing.allocator, pages.platform(), fixture, &.{}, .{}));
    var bad_imports = imports;
    bad_imports[0].params = &.{.i64};
    try std.testing.expectError(error.ImportSignatureMismatch, api.Instance.load(std.testing.allocator, pages.platform(), fixture, &bad_imports, .{}));
    try std.testing.expectError(error.UnsupportedTarget, api.Instance.load(std.testing.allocator, pages.platform(), fixture, &imports, .{ .cpu_feature_mask = 0 }));
    const corrupt = try std.testing.allocator.dupe(u8, fixture);
    defer std.testing.allocator.free(corrupt);
    // Ordinary SysV/Linux emitter has neither native-profile flag nor contract.
    @memset(corrupt[24..32], 0);
    try std.testing.expectError(error.UnsupportedTarget, api.Instance.load(std.testing.allocator, pages.platform(), corrupt, &imports, .{}));
    @memcpy(corrupt, fixture);
    corrupt[4] = 0;
    try std.testing.expectError(error.InvalidVersion, api.Instance.load(std.testing.allocator, pages.platform(), corrupt, &imports, .{}));
    try equal(@as(usize, 0), pages.reservations);
}
test "native AOT rollback covers every allocation and page transition failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationLifecycle, .{});
    for (0..2) |index| {
        var pages: Pages = .{ .fail_reserve = index };
        try std.testing.expectError(error.OutOfMemory, create(std.testing.allocator, &pages));
        try equal(@as(usize, 0), pages.live);
    }

    for (0..2) |index| {
        var pages: Pages = .{ .fail_commit = index };
        try std.testing.expectError(error.OutOfMemory, create(std.testing.allocator, &pages));
        try equal(@as(usize, 0), pages.live);
    }
    var pages: Pages = .{ .fail_protect = true };
    try std.testing.expectError(error.ProtectionFailed, create(std.testing.allocator, &pages));
    try equal(@as(usize, 0), pages.live);
}
test "native AOT call boundary does not allocate after instantiation" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var pages: Pages = .{};
    const inst = try create(failing.allocator(), &pages);
    defer inst.deinit();
    const allocations = failing.alloc_index;
    const resizes = failing.resize_index;
    failing.fail_index = allocations;
    failing.resize_fail_index = resizes;
    for (0..32) |_| {
        try equal(@as(i32, 42), try callI32(inst, "add", &.{ .{ .i32 = 20 }, .{ .i32 = 22 } }));
        try equal(@as(i32, 42), try callI32(inst, "call_host", &.{ .{ .i32 = 20 }, .{ .i32 = 22 } }));
        try equal(@as(usize, 0), (try inst.call("noop", &.{}, &.{})).returned);
    }
    try equal(api.Trap.unreachable_instruction, (try inst.call("trap", &.{}, &.{})).trap);
    const old_pages: i32 = @intCast(inst.vmctx.memory_pages);
    const commits = pages.commits;
    try equal(old_pages, try callI32(inst, "grow", &.{.{ .i32 = 1 }}));
    try equal(commits + 1, pages.commits);
    try equal(allocations, failing.alloc_index);
    try equal(resizes, failing.resize_index);
    try expect(!failing.has_induced_failure);
}

test "native AOT phase timings measure real boundaries and roll back clock failures" {
    var pages: Pages = .{ .clock_value = 100, .clock_step = 100, .check_phase_boundaries = true };
    var timings: api.LoadTimings = .{};
    const inst = try api.Instance.load(std.testing.allocator, pages.platform(), fixture, &imports, .{ .max_memory_pages = 8, .timings = &timings });
    try equal(@as(u32, 3), timings.completed);
    try equal(@as(u64, 100), timings.load_ns);
    try equal(@as(u64, 100), timings.instantiate_ns);
    try equal(@as(usize, 3), pages.clock_reads);
    try equal(@as(i32, 42), try callI32(inst, "add", &.{ .{ .i32 = 20 }, .{ .i32 = 22 } }));
    inst.deinit();
    try equal(@as(usize, 0), pages.live);

    for (0..3) |index| {
        pages = .{ .clock_value = 100, .clock_step = 100, .clock_fail_on = index };
        try std.testing.expectError(error.ClockFailed, api.Instance.load(std.testing.allocator, pages.platform(), fixture, &imports, .{ .timings = &timings }));
        try equal(@as(u32, if (index == 2) 1 else 0), timings.completed);
        try equal(@as(usize, 0), pages.live);
    }
    pages = .{ .clock_value = 100, .clock_rewind_after_first = true };
    try std.testing.expectError(error.ClockFailed, api.Instance.load(std.testing.allocator, pages.platform(), fixture, &imports, .{ .timings = &timings }));
    try equal(@as(u32, 0), timings.completed);
    try equal(@as(usize, 0), pages.live);
}

fn allocationLifecycle(allocator: std.mem.Allocator) !void {
    var pages: Pages = .{};
    defer std.debug.assert(pages.live == 0);
    const inst = try create(allocator, &pages);
    inst.deinit();
}

const CState = struct {
    allocator: std.mem.Allocator,
    pages: Pages = .{},
    fn allocate(context: ?*anyopaque, size: usize, alignment: usize) callconv(.c) ?*anyopaque {
        const self: *CState = @ptrCast(@alignCast(context.?));
        return self.allocator.rawAlloc(size, .fromByteUnits(alignment), @returnAddress());
    }
    fn free(context: ?*anyopaque, pointer: *anyopaque, size: usize, alignment: usize) callconv(.c) void {
        const self: *CState = @ptrCast(@alignCast(context.?));
        self.allocator.rawFree(@as([*]u8, @ptrCast(pointer))[0..size], .fromByteUnits(alignment), @returnAddress());
    }
    fn reserve(context: ?*anyopaque, size: usize) callconv(.c) ?*anyopaque {
        const self: *CState = @ptrCast(@alignCast(context.?));
        return Pages.reserve(&self.pages, size) catch null;
    }
    fn commit(context: ?*anyopaque, pointer: *anyopaque, size: usize) callconv(.c) c_int {
        const self: *CState = @ptrCast(@alignCast(context.?));
        Pages.commit(&self.pages, @ptrCast(@alignCast(pointer)), size) catch return -1;
        return 0;
    }
    fn protect(context: ?*anyopaque, pointer: *anyopaque, size: usize, prot: u32) callconv(.c) c_int {
        const self: *CState = @ptrCast(@alignCast(context.?));
        Pages.protect(&self.pages, @ptrCast(@alignCast(pointer)), size, @enumFromInt(prot)) catch return -1;
        return 0;
    }
    fn unmap(context: ?*anyopaque, pointer: *anyopaque, size: usize) callconv(.c) void {
        const self: *CState = @ptrCast(@alignCast(context.?));
        Pages.unmap(&self.pages, @ptrCast(@alignCast(pointer)), size);
    }
    fn clock(context: ?*anyopaque, result: *u64) callconv(.c) c_int {
        const self: *CState = @ptrCast(@alignCast(context.?));
        result.* = Pages.clock(&self.pages) catch return -1;
        return 0;
    }
    fn config(self: *CState) c_api.Config {
        return .{ .context = self, .alloc = allocate, .free = free, .reserve = reserve, .commit = commit, .protect = protect, .unmap = unmap, .monotonic_ns = clock, .max_memory_pages = 8, .max_table_elements = 16 };
    }
};
fn cHost(_: ?*anyopaque, _: *api.HostContext, args: [*]const c_api.CValue, count: usize, results: [*]c_api.CValue, capacity: usize) callconv(.c) u32 {
    if (count == 2 and capacity == 1) {
        results[0] = .{ .kind = 0x7f, .bits = args[0].bits +% args[1].bits };
        return 0;
    }
    return 1;
}
test "native C ABI executes exact typed result and retains copied artifact ownership" {
    var state: CState = .{ .allocator = std.testing.allocator };
    const config = state.config();
    var c_imports: [3]c_api.CImport = undefined;
    for (imports, &c_imports) |imp, *c_imp| c_imp.* = .{
        .module = imp.module.ptr,
        .module_len = imp.module.len,
        .name = imp.name.ptr,
        .name_len = imp.name.len,
        .params = @ptrCast(imp.params.ptr),
        .param_count = imp.params.len,
        .results = @ptrCast(imp.results.ptr),
        .result_count = imp.results.len,
        .context = null,
        .callback = cHost,
    };
    const owned = try std.testing.allocator.dupe(u8, fixture);
    defer std.testing.allocator.free(owned);
    var handle: ?*c_api.Handle = null;
    try equal(@as(u32, 0), c_api.wamr_aot_load(&config, owned.ptr, owned.len, &c_imports, 3, &handle).kind);
    @memset(owned, 0);
    const args = [_]c_api.CValue{ .{ .kind = 0x7f, .bits = 40 }, .{ .kind = 0x7f, .bits = 2 } };
    var results: [1]c_api.CValue = undefined;
    const returned = c_api.wamr_aot_call(handle.?, "call_host", 9, &args, 2, &results, 1);
    try equal(@as(u32, 0), returned.kind);
    try equal(@as(usize, 1), returned.count);
    try equal(@as(u32, 0x7f), results[0].kind);
    try equal(@as(u64, 42), results[0].bits);
    c_api.wamr_aot_destroy(handle.?);
    try equal(@as(usize, 0), state.pages.live);
    state.pages.clock_step = 100;
    var timings: api.LoadTimings = .{};
    try equal(@as(u32, 0), c_api.wamr_aot_load_timed(&config, fixture.ptr, fixture.len, &c_imports, 3, &handle, &timings).kind);
    try equal(@as(u32, 3), timings.completed);
    try equal(@as(u64, 100), timings.load_ns);
    try equal(@as(u64, 100), timings.instantiate_ns);
    c_api.wamr_aot_destroy(handle.?);
    try equal(@as(usize, 0), state.pages.live);
}
