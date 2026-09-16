//! Native library root and C ABI. The Zig embedding API is `aot`.
const std = @import("std");
pub const aot = @import("api/aot.zig");

pub const Config = extern struct {
    context: ?*anyopaque,
    alloc: *const fn (?*anyopaque, usize, usize) callconv(.c) ?*anyopaque,
    free: *const fn (?*anyopaque, *anyopaque, usize, usize) callconv(.c) void,
    reserve: *const fn (?*anyopaque, usize) callconv(.c) ?*anyopaque,
    commit: *const fn (?*anyopaque, *anyopaque, usize) callconv(.c) c_int,
    protect: *const fn (?*anyopaque, *anyopaque, usize, u32) callconv(.c) c_int,
    unmap: *const fn (?*anyopaque, *anyopaque, usize) callconv(.c) void,
    monotonic_ns: *const fn (?*anyopaque, *u64) callconv(.c) c_int,
    max_memory_pages: u32,
    max_table_elements: u32,
};
pub const CValue = extern struct { kind: u32, reserved: u32 = 0, bits: u64 };
pub const CImport = extern struct {
    module: [*]const u8,
    module_len: usize,
    name: [*]const u8,
    name_len: usize,
    params: [*]const u8,
    param_count: usize,
    results: [*]const u8,
    result_count: usize,
    context: ?*anyopaque,
    callback: *const fn (?*anyopaque, *aot.HostContext, [*]const CValue, usize, [*]CValue, usize) callconv(.c) u32,
};
pub const Result = extern struct {
    /// 0 returned, 1 trap, 2 guest exit, 3 host error, 4 API/load error.
    kind: u32 = 0,
    detail: u32 = 0,
    count: usize = 0,
    /// Static, NUL-terminated name for API/load and host errors.
    error_name: ?[*:0]const u8 = null,
};
pub const Handle = struct { instance: *aot.Instance, config: *const Config };

pub export fn wamr_aot_contract_version() u32 {
    return @import("runtime/aot/native_abi.zig").contract_version;
}

/// Config and CImport descriptors (including strings/signatures/contexts) must
/// remain alive until destroy. Bytes are copied. On failure out is null.
pub export fn wamr_aot_load(config: *const Config, bytes: [*]const u8, length: usize, imports: [*]const CImport, import_count: usize, out: *?*Handle) Result {
    out.* = null;
    const allocator = makeAllocator(config);
    const handle = allocator.create(Handle) catch |err| return failure(err);
    const host_imports = allocator.alloc(aot.HostImport, import_count) catch |err| {
        allocator.destroy(handle);
        return failure(err);
    };
    defer allocator.free(host_imports);
    for (imports[0..import_count], host_imports) |*imp, *host| {
        if (imp.param_count > 5 or imp.result_count > 1) {
            allocator.destroy(handle);
            return failure(error.UnsupportedFeature);
        }
        for (imp.params[0..imp.param_count]) |t| {
            _ = std.enums.fromInt(aot.ValType, t) orelse {
                allocator.destroy(handle);
                return failure(error.UnsupportedFeature);
            };
        }
        for (imp.results[0..imp.result_count]) |t| {
            _ = std.enums.fromInt(aot.ValType, t) orelse {
                allocator.destroy(handle);
                return failure(error.UnsupportedFeature);
            };
        }
        host.* = .{
            .module = imp.module[0..imp.module_len],
            .name = imp.name[0..imp.name_len],
            .params = @ptrCast(imp.params[0..imp.param_count]),
            .results = @ptrCast(imp.results[0..imp.result_count]),
            .context = @ptrCast(@constCast(imp)),
            .callback = hostCall,
        };
    }
    const native: aot.Platform = .{
        .context = @ptrCast(@constCast(config)),
        .reserve = reserve,
        .commit = commit,
        .protect = protect,
        .unmap = unmap,
        .monotonic_ns = clock,
    };
    handle.* = .{
        .config = config,
        .instance = aot.Instance.load(allocator, native, bytes[0..length], host_imports, .{
            .max_memory_pages = config.max_memory_pages,
            .max_table_elements = config.max_table_elements,
        }) catch |err| {
            allocator.destroy(handle);
            return failure(err);
        },
    };
    out.* = handle;
    return .{};
}

pub export fn wamr_aot_destroy(handle: *Handle) void {
    const allocator = makeAllocator(handle.config);
    handle.instance.deinit();
    allocator.destroy(handle);
}
pub export fn wamr_aot_start(handle: *Handle) Result {
    return outcome(handle.instance.start() catch |err| return failure(err));
}
pub export fn wamr_aot_call(handle: *Handle, name: [*]const u8, name_len: usize, args: [*]const CValue, arg_count: usize, results: [*]CValue, result_capacity: usize) Result {
    if (arg_count > 16) return failure(error.ArgumentCountMismatch);
    var values: [16]aot.Value = undefined;
    for (args[0..arg_count], 0..) |arg, i| {
        const t = std.enums.fromInt(aot.ValType, std.math.cast(u8, arg.kind) orelse return failure(error.ArgumentTypeMismatch)) orelse return failure(error.ArgumentTypeMismatch);
        values[i] = aot.Value.fromRaw(t, arg.bits);
    }
    var returned: [1]aot.Value = undefined;
    const result = handle.instance.call(name[0..name_len], values[0..arg_count], returned[0..@min(result_capacity, 1)]) catch |err| return failure(err);
    if (result == .returned and result.returned == 1) results[0] = .{ .kind = @intFromEnum(std.meta.activeTag(returned[0])), .bits = returned[0].raw() };
    return outcome(result);
}
pub export fn wamr_aot_memory(handle: *Handle, length: *usize) [*]u8 {
    const bytes = handle.instance.memory();
    length.* = bytes.len;
    return bytes.ptr;
}
pub export fn wamr_aot_host_memory(context: *aot.HostContext, length: *usize) [*]u8 {
    const bytes = context.memory();
    length.* = bytes.len;
    return bytes.ptr;
}
pub export fn wamr_aot_host_clock(context: *aot.HostContext, ns: *u64) c_int {
    ns.* = context.monotonicNs() catch return -1;
    return 0;
}
pub export fn wamr_aot_host_exit(context: *aot.HostContext, code: u32) void {
    context.terminate(code);
}
fn failure(err: anyerror) Result {
    return .{ .kind = 4, .error_name = @errorName(err).ptr };
}
fn outcome(result: aot.Outcome) Result {
    return switch (result) {
        .returned => |count| .{ .count = count },
        .trap => |trap| .{ .kind = 1, .detail = @intFromEnum(trap) },
        .exit => |code| .{ .kind = 2, .detail = code },
        .host_error => |err| .{ .kind = 3, .error_name = @errorName(err).ptr },
    };
}
fn hostCall(userdata: ?*anyopaque, context: *aot.HostContext, args: []const aot.Value, results: []aot.Value) aot.HostError!void {
    const imp: *const CImport = @ptrCast(@alignCast(userdata.?));
    var raw_args: [5]CValue = undefined;
    var raw_result: [1]CValue = undefined;
    for (args, 0..) |arg, i| raw_args[i] = .{ .kind = @intFromEnum(std.meta.activeTag(arg)), .bits = arg.raw() };
    if (results.len == 1) raw_result[0] = .{ .kind = @intFromEnum(std.meta.activeTag(results[0])), .bits = 0 };
    const status = imp.callback(imp.context, context, &raw_args, args.len, &raw_result, results.len);
    switch (status) {
        0 => {},
        1 => return error.Unsupported,
        2 => return error.InvalidArgument,
        3 => return error.Io,
        4 => return error.OutOfMemory,
        else => return error.InvalidArgument,
    }
    if (results.len == 1) {
        if (raw_result[0].kind != @intFromEnum(std.meta.activeTag(results[0]))) return error.InvalidArgument;
        results[0] = aot.Value.fromRaw(std.meta.activeTag(results[0]), raw_result[0].bits);
    }
}
fn configOf(ptr: *anyopaque) *const Config {
    return @ptrCast(@alignCast(ptr));
}
fn makeAllocator(config: *const Config) std.mem.Allocator {
    return .{ .ptr = @ptrCast(@constCast(config)), .vtable = &allocator_vtable };
}
const allocator_vtable: std.mem.Allocator.VTable = .{
    .alloc = allocate,
    .resize = std.mem.Allocator.noResize,
    .remap = std.mem.Allocator.noRemap,
    .free = release,
};
fn allocate(ptr: *anyopaque, len: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
    const c = configOf(ptr);
    const p = c.alloc(c.context, len, alignment.toByteUnits()) orelse return null;
    if (@intFromPtr(p) % alignment.toByteUnits() != 0) {
        c.free(c.context, p, len, alignment.toByteUnits());
        return null;
    }
    return @ptrCast(p);
}
fn release(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, _: usize) void {
    const c = configOf(ptr);
    c.free(c.context, memory.ptr, memory.len, alignment.toByteUnits());
}
fn reserve(ptr: *anyopaque, size: usize) aot.PlatformError![*]align(4096) u8 {
    const c = configOf(ptr);
    const p = c.reserve(c.context, size) orelse return error.OutOfMemory;
    if (@intFromPtr(p) % 4096 != 0) {
        c.unmap(c.context, p, size);
        return error.InvalidMapping;
    }
    return @ptrCast(@alignCast(p));
}
fn commit(ptr: *anyopaque, base: [*]align(4096) u8, size: usize) aot.PlatformError!void {
    const c = configOf(ptr);
    if (c.commit(c.context, base, size) != 0) return error.OutOfMemory;
}
fn protect(ptr: *anyopaque, base: [*]align(4096) u8, size: usize, protection: aot.platform.Protection) aot.PlatformError!void {
    const c = configOf(ptr);
    if (c.protect(c.context, base, size, @intFromEnum(protection)) != 0) return error.ProtectionFailed;
}
fn unmap(ptr: *anyopaque, base: [*]align(4096) u8, size: usize) void {
    const c = configOf(ptr);
    c.unmap(c.context, base, size);
}
fn clock(ptr: *anyopaque) aot.PlatformError!u64 {
    const c = configOf(ptr);
    var ns: u64 = undefined;
    if (c.monotonic_ns(c.context, &ns) != 0) return error.ClockFailed;
    return ns;
}
