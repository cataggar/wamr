//! Compiler-free, single-threaded x86_64 SysV embedding of trusted .cwasm.
//! The caller owns allocation and native page/clock capabilities. No hosted
//! runtime, compiler, interpreter, component, CLI or thread modules are imported.
const std = @import("std");
const builtin = @import("builtin");
const format = @import("../runtime/aot/native_format.zig");
const abi = format.abi;
const jump = @import("../runtime/aot/trap_jmp.zig");
pub const platform = @import("../platform/unikraft.zig");
pub const Platform = platform.Platform;
pub const PlatformError = platform.Error;
pub const ValType = format.ValType;
pub const Value = format.Value;
pub const HostError = error{ Unsupported, InvalidArgument, Io, OutOfMemory };
pub const Trap = enum { out_of_bounds_memory, out_of_bounds_table, unreachable_instruction, integer_divide_by_zero, integer_overflow, invalid_conversion, unsupported_operation, bad_host_result };
pub const Outcome = union(enum) { returned: usize, trap: Trap, exit: u32, host_error: HostError };
pub const Error = format.Error || PlatformError || error{ MissingImport, ImportSignatureMismatch, TooManyImports, FunctionNotFound, ArgumentCountMismatch, ArgumentTypeMismatch, ResultBufferTooSmall, Busy, StartRequired, StartFailed, InvalidContinuation, MemoryLimitExceeded, TableLimitExceeded, InitializerOutOfBounds };

pub const HostImport = struct {
    module: []const u8,
    name: []const u8,
    params: []const ValType,
    results: []const ValType,
    context: ?*anyopaque = null,
    callback: *const fn (?*anyopaque, *HostContext, []const Value, []Value) HostError!void,
};

pub const HostContext = struct {
    instance: *Instance,

    /// This slice is borrowed for the call. Its base remains stable across grow.
    pub fn memory(self: *HostContext) []u8 {
        return self.instance.memory();
    }
    pub fn monotonicNs(self: *HostContext) PlatformError!u64 {
        const p = self.instance.native;
        return p.monotonicNs();
    }
    /// Host adapters must return normally from callbacks, allowing their defers
    /// to run. The dispatcher unwinds guest frames after observing this request.
    pub fn terminate(self: *HostContext, code: u32) void {
        if (self.instance.pending == null) self.instance.pending = .{ .exit = code };
    }
};

pub const Options = struct {
    /// Guest reserve limit; an explicit wasm maximum still takes precedence.
    max_memory_pages: u32 = 256,
    max_table_elements: u32 = 65536,
    /// Restrict available CPU features for qualification. Cannot enable a
    /// feature CPUID does not report.
    cpu_feature_mask: u64 = std.math.maxInt(u64),
    /// Opt-in measurements from the caller's monotonic clock. Bit 0 of
    /// completed marks load_ns valid, bit 1 marks instantiate_ns valid.
    /// No clock reads or timing claims are made when this is null.
    timings: ?*LoadTimings = null,
};

pub const LoadTimings = extern struct {
    load_ns: u64 = 0,
    instantiate_ns: u64 = 0,
    completed: u32 = 0,
    reserved: u32 = 0,
};

const Mapping = struct { base: [*]align(4096) u8, size: usize };
const TableStorage = struct { pointers: []usize, signatures: []u32, size: u32, max: u32 };

pub const Instance = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    native: Platform,
    module: format.Module,
    vmctx: abi.VmCtx = .{},
    code: ?Mapping = null,
    linear: ?Mapping = null,
    hosts: []HostImport = &.{},
    host_pointers: []usize = &.{},
    functions: []usize = &.{},
    globals: []u64 = &.{},
    signatures: []u32 = &.{},
    function_signatures: []u32 = &.{},
    tables: []TableStorage = &.{},
    table_descriptors: []abi.Table = &.{},
    dropped_data: []bool = &.{},
    dropped_elements: []bool = &.{},
    active: bool = false,
    started: bool = false,
    start_failed: bool = false,
    pending: ?Outcome = null,
    continuation: jump.JmpBuf = undefined,

    /// Copies input and import descriptors. Callback contexts and platform
    /// context must outlive this instance. Start functions are not executed
    /// while loading: call start() explicitly and inspect its terminal result.
    pub fn load(allocator: std.mem.Allocator, native: Platform, bytes: []const u8, imports: []const HostImport, options: Options) Error!*Instance {
        if (options.timings) |timings| timings.* = .{};
        if (comptime builtin.cpu.arch != .x86_64 or builtin.os.tag == .windows) return error.UnsupportedTarget;
        try native.validate();
        if (options.max_memory_pages > 65536) return error.InvalidLimits;
        const load_start = if (options.timings != null) try native.monotonicNs() else 0;
        const self = try allocator.create(Instance);
        self.* = .{ .allocator = allocator, .arena = .init(allocator), .native = native, .module = .{} };
        errdefer self.deinit();
        const a = self.arena.allocator();
        const owned = try a.dupe(u8, bytes);
        self.module = try format.load(owned, a, platform.detectedCpuFeatures() & options.cpu_feature_mask);
        if (self.module.imports.len > max_imports) return error.TooManyImports;
        self.hosts = try a.alloc(HostImport, self.module.imports.len);
        self.host_pointers = try a.alloc(usize, self.hosts.len);
        for (self.module.imports, 0..) |required, i| {
            const signature = self.module.signatures[required.type_index];
            if (signature.params.len > 5) return error.UnsupportedFeature;
            var found: ?HostImport = null;
            for (imports) |candidate| {
                if (std.mem.eql(u8, candidate.module, required.module) and std.mem.eql(u8, candidate.name, required.name)) {
                    if (found != null) return error.ImportSignatureMismatch;
                    found = candidate;
                }
            }
            var host = found orelse return error.MissingImport;
            if (!std.mem.eql(ValType, host.params, signature.params) or !std.mem.eql(ValType, host.results, signature.results))
                return error.ImportSignatureMismatch;
            host.module = required.module;
            host.name = required.name;
            host.params = signature.params;
            host.results = signature.results;
            self.hosts[i] = host;
            self.host_pointers[i] = @intFromPtr(import_pointers[i]);
        }
        // Metadata and required-import validation are complete. Everything
        // below allocates/initializes the executable instance, not the loader.
        const instantiate_start = if (options.timings) |timings| blk: {
            const now = try native.monotonicNs();
            timings.load_ns = std.math.sub(u64, now, load_start) catch return error.ClockFailed;
            timings.completed = 1;
            break :blk now;
        } else 0;
        self.globals = try a.alloc(u64, self.module.globals.len);
        for (self.module.globals, self.globals) |g, *bits| bits.* = g.bits;
        self.signatures = try a.alloc(u32, self.module.signatures.len);
        for (self.module.signatures, 0..) |sig, i| {
            self.signatures[i] = @intCast(i + 1);
            for (self.module.signatures[0..i], 0..) |prior, j| {
                if (std.mem.eql(ValType, sig.params, prior.params) and std.mem.eql(ValType, sig.results, prior.results)) {
                    self.signatures[i] = self.signatures[j];
                    break;
                }
            }
        }
        self.functions = try a.alloc(usize, self.hosts.len + self.module.functions.len);
        self.function_signatures = try a.alloc(u32, self.functions.len);
        for (self.module.imports, 0..) |imp, i| self.function_signatures[i] = self.signatures[imp.type_index];
        for (self.module.functions, self.hosts.len..) |f, i| self.function_signatures[i] = self.signatures[f.type_index];
        self.tables = try a.alloc(TableStorage, self.module.tables.len);
        self.table_descriptors = try a.alloc(abi.Table, self.tables.len);
        for (self.module.tables, self.tables) |t, *storage| {
            const max = @min(t.max orelse options.max_table_elements, options.max_table_elements);
            if (t.min > max) return error.TableLimitExceeded;
            storage.* = .{ .pointers = try a.alloc(usize, max), .signatures = try a.alloc(u32, max), .size = t.min, .max = max };
            @memset(storage.pointers, 0);
            @memset(storage.signatures, 0);
        }
        if (self.module.memory) |m| {
            const max = @min(m.max orelse options.max_memory_pages, options.max_memory_pages);
            if (m.min > max) return error.MemoryLimitExceeded;
            if (max != 0) {
                const size = @as(usize, max) * 65536;
                // Publish ownership only after the fallible reservation succeeds.
                const base = try native.reserve(native.context, size);
                self.linear = .{ .base = base, .size = size };
                const initial_size = @as(usize, m.min) * 65536;
                if (initial_size != 0) {
                    try native.commit(native.context, self.linear.?.base, initial_size);
                    @memset(self.linear.?.base[0..initial_size], 0);
                }
                self.vmctx.memory_base = @intFromPtr(self.linear.?.base);
                self.vmctx.memory_max_size = size;
                self.vmctx.memory_size = initial_size;
                self.vmctx.memory_pages = m.min;
            }
        }
        self.dropped_data = try a.alloc(bool, self.module.data.len);
        for (self.module.data, self.dropped_data) |d, *dropped| {
            dropped.* = d.kind != 2;
            if (d.kind == 2) continue;
            if (!inBounds(d.offset, d.bytes.len, self.vmctx.memory_size)) return error.InitializerOutOfBounds;
            @memcpy(self.memory()[d.offset..][0..d.bytes.len], d.bytes);
        }
        self.dropped_elements = try a.alloc(bool, self.module.elements.len);
        for (self.module.elements, self.dropped_elements) |e, *dropped| {
            dropped.* = !e.passive;
            if (!e.passive and !inBounds(e.offset, e.indices.len, self.tables[e.table].size)) return error.InitializerOutOfBounds;
        }
        const code_size = try native.rounded(self.module.text.len);
        const code_base = try native.reserve(native.context, code_size);
        self.code = .{ .base = code_base, .size = code_size };
        try native.commit(native.context, self.code.?.base, code_size);
        @memcpy(self.code.?.base[0..self.module.text.len], self.module.text);
        try native.protect(native.context, self.code.?.base, code_size, .read_execute);
        @memcpy(self.functions[0..self.hosts.len], self.host_pointers);
        for (self.module.functions, self.hosts.len..) |f, i| self.functions[i] = @intFromPtr(self.code.?.base + f.offset);
        for (self.module.elements) |e| {
            if (e.passive) continue;
            for (e.indices, e.offset..) |function, index| {
                const pointer = if (function == format.null_function_index) 0 else self.functions[function];
                self.setTableEntry(e.table, @intCast(index), pointer);
            }
        }
        self.vmctx.instance_ptr = @intFromPtr(self);
        self.vmctx.globals_ptr = @intFromPtr(self.globals.ptr);
        self.vmctx.globals_count = @intCast(self.globals.len);
        self.vmctx.host_functions_ptr = @intFromPtr(self.host_pointers.ptr);
        self.vmctx.host_functions_count = @intCast(self.hosts.len);
        self.vmctx.funcptrs_ptr = @intFromPtr(self.functions.ptr);
        self.vmctx.sig_table_ptr = @intFromPtr(self.signatures.ptr);
        self.vmctx.func_sig_ids_ptr = @intFromPtr(self.function_signatures.ptr);
        self.refreshTables();
        self.vmctx.mem_grow_fn = @intFromPtr(&memoryGrow);
        self.vmctx.mem_fill_fn = @intFromPtr(&memoryFill);
        self.vmctx.mem_copy_fn = @intFromPtr(&memoryCopy);
        self.vmctx.memory_init_fn = @intFromPtr(&memoryInit);
        self.vmctx.data_drop_fn = @intFromPtr(&dataDrop);
        self.vmctx.table_grow_fn = @intFromPtr(&tableGrow);
        self.vmctx.table_set_fn = @intFromPtr(&tableSet);
        self.vmctx.table_init_fn = @intFromPtr(&tableInit);
        self.vmctx.elem_drop_fn = @intFromPtr(&elementDrop);
        self.vmctx.trap_oob_fn = @intFromPtr(&trapOutOfBounds);
        self.vmctx.trap_unreachable_fn = @intFromPtr(&trapUnreachable);
        self.vmctx.trap_idivz_fn = @intFromPtr(&trapDivideZero);
        self.vmctx.trap_iovf_fn = @intFromPtr(&trapOverflow);
        self.vmctx.trap_ivc_fn = @intFromPtr(&trapConversion);
        self.vmctx.futex_wait32_fn = @intFromPtr(&trapUnsupported);
        self.vmctx.futex_wait64_fn = @intFromPtr(&trapUnsupported);
        self.vmctx.futex_notify_fn = @intFromPtr(&trapUnsupported);
        self.vmctx.aot_throw_uncaught_fn = @intFromPtr(&trapUnsupported);
        self.vmctx.lazy_compile_fn = @intFromPtr(&trapUnsupported);
        self.vmctx.trap_unaligned_fn = @intFromPtr(&trapUnsupported);
        self.vmctx.cancel_point_fn = @intFromPtr(&trapUnsupported);
        if (options.timings) |timings| {
            const ready = try native.monotonicNs();
            timings.instantiate_ns = std.math.sub(u64, ready, instantiate_start) catch return error.ClockFailed;
            timings.completed = 3;
        }
        return self;
    }

    /// The caller must serialize access and must not destroy an active instance.
    pub fn deinit(self: *Instance) void {
        std.debug.assert(!self.active);
        if (self.code) |m| self.native.unmap(self.native.context, m.base, m.size);
        if (self.linear) |m| self.native.unmap(self.native.context, m.base, m.size);
        const allocator = self.allocator;
        self.arena.deinit();
        allocator.destroy(self);
    }

    pub fn memory(self: *Instance) []u8 {
        if (self.linear) |m| return m.base[0..self.vmctx.memory_size];
        return &.{};
    }

    pub fn grow(self: *Instance, delta: u32) ?u32 {
        const old = self.vmctx.memory_pages;
        if (self.module.memory == null) return null;
        if (delta == 0) return old;
        const new = std.math.add(u32, old, delta) catch return null;
        const bytes = @as(usize, new) * 65536;
        if (bytes > self.vmctx.memory_max_size) return null;
        const begin: [*]align(4096) u8 = @alignCast(self.linear.?.base + self.vmctx.memory_size);
        const added = bytes - self.vmctx.memory_size;
        self.native.commit(self.native.context, begin, added) catch return null;
        @memset(begin[0..added], 0);
        self.vmctx.memory_pages = new;
        self.vmctx.memory_size = bytes;
        return old;
    }

    pub fn start(self: *Instance) Error!Outcome {
        if (self.active) return error.Busy;
        if (self.started) return error.Busy;
        self.started = true;
        if (self.module.start) |index| {
            const result = try self.invoke(index, &.{}, &.{});
            self.start_failed = result != .returned;
            return result;
        }
        return .{ .returned = 0 };
    }

    pub fn call(self: *Instance, name: []const u8, args: []const Value, results: []Value) Error!Outcome {
        if (self.start_failed) return error.StartFailed;
        if (self.module.start != null and !self.started) return error.StartRequired;
        for (self.module.exports) |e| {
            if (e.kind == 0 and std.mem.eql(u8, name, e.name)) return self.invoke(e.index, args, results);
        }
        return error.FunctionNotFound;
    }

    fn invoke(self: *Instance, index: u32, args: []const Value, results: []Value) Error!Outcome {
        if (self.active) return error.Busy;
        const sig = try self.module.signature(index);
        if (args.len != sig.params.len) return error.ArgumentCountMismatch;
        if (results.len < sig.results.len) return error.ResultBufferTooSmall;
        var raw: [16]u64 = @splat(0);
        for (args, sig.params, 0..) |value, expected, i| {
            if (std.meta.activeTag(value) != expected) return error.ArgumentTypeMismatch;
            raw[i] = value.raw();
        }
        self.pending = null;
        self.active = true;
        defer self.active = false;
        if (jump.capture(&self.continuation) != 0) {
            // The assembly continuation is not an ordinary Zig return edge.
            // Force a reload of state written by a deeper, now-unwound frame.
            const terminal: *volatile ?Outcome = &self.pending;
            return terminal.* orelse error.InvalidContinuation;
        }
        const bits = invokeRaw(self.functions[index], &self.vmctx, &raw);
        if (sig.results.len == 1) results[0] = Value.fromRaw(sig.results[0], bits);
        return .{ .returned = sig.results.len };
    }

    fn stop(self: *Instance, outcome: Outcome) noreturn {
        const terminal: *volatile ?Outcome = &self.pending;
        terminal.* = outcome;
        jump.restore(&self.continuation, 1);
    }

    fn setTableEntry(self: *Instance, table: u32, index: u32, pointer: usize) void {
        var sig: u32 = 0;
        for (self.functions, self.function_signatures) |candidate, id| {
            if (pointer == candidate) {
                sig = id;
                break;
            }
        }
        if (pointer != 0 and sig == 0) self.stop(.{ .trap = .out_of_bounds_table });
        self.tables[table].pointers[index] = pointer;
        self.tables[table].signatures[index] = sig;
    }

    fn refreshTables(self: *Instance) void {
        for (self.tables, self.table_descriptors) |t, *d| d.* = .{ .ptr = @intFromPtr(t.pointers.ptr), .len = t.size, .type_backing_ptr = @intFromPtr(t.signatures.ptr) };
        self.vmctx.tables_info_ptr = @intFromPtr(self.table_descriptors.ptr);
        if (self.tables.len != 0) {
            self.vmctx.func_table_ptr = self.table_descriptors[0].ptr;
            self.vmctx.func_table_len = self.tables[0].size;
        }
    }
};

fn owner(ctx: *abi.VmCtx) *Instance {
    return @ptrFromInt(ctx.instance_ptr);
}
fn inBounds(offset: usize, length: usize, size: usize) bool {
    return offset <= size and length <= size - offset;
}
fn trapOutOfBounds(ctx: *abi.VmCtx) callconv(.c) noreturn {
    owner(ctx).stop(.{ .trap = .out_of_bounds_memory });
}
fn trapUnreachable(ctx: *abi.VmCtx) callconv(.c) noreturn {
    owner(ctx).stop(.{ .trap = .unreachable_instruction });
}
fn trapDivideZero(ctx: *abi.VmCtx) callconv(.c) noreturn {
    owner(ctx).stop(.{ .trap = .integer_divide_by_zero });
}
fn trapOverflow(ctx: *abi.VmCtx) callconv(.c) noreturn {
    owner(ctx).stop(.{ .trap = .integer_overflow });
}
fn trapConversion(ctx: *abi.VmCtx) callconv(.c) noreturn {
    owner(ctx).stop(.{ .trap = .invalid_conversion });
}
fn trapUnsupported(ctx: *abi.VmCtx) callconv(.c) noreturn {
    owner(ctx).stop(.{ .trap = .unsupported_operation });
}
fn memoryGrow(ctx: *abi.VmCtx, delta: i32) callconv(.c) i32 {
    return @bitCast(owner(ctx).grow(@bitCast(delta)) orelse return -1);
}
fn memoryFill(ctx: *abi.VmCtx, dst: u32, val: u32, len: u32) callconv(.c) void {
    if (!inBounds(dst, len, ctx.memory_size)) trapOutOfBounds(ctx);
    @memset(owner(ctx).memory()[dst..][0..len], @truncate(val));
}
fn memoryCopy(ctx: *abi.VmCtx, dst: u32, src: u32, len: u32) callconv(.c) void {
    if (!inBounds(dst, len, ctx.memory_size) or !inBounds(src, len, ctx.memory_size)) trapOutOfBounds(ctx);
    const mem = owner(ctx).memory();
    if (dst <= src) std.mem.copyForwards(u8, mem[dst..][0..len], mem[src..][0..len]) else std.mem.copyBackwards(u8, mem[dst..][0..len], mem[src..][0..len]);
}
fn memoryInit(ctx: *abi.VmCtx, segment_memory: u64, dst_src: u64, len: u32) callconv(.c) void {
    const self = owner(ctx);
    const segment: u32 = @truncate(segment_memory);
    const memory_index = segment_memory >> 32;
    const dst: u32 = @truncate(dst_src);
    const src: u32 = @truncate(dst_src >> 32);
    if (memory_index != 0 or segment >= self.module.data.len) trapOutOfBounds(ctx);
    const bytes = if (self.dropped_data[segment]) &.{} else self.module.data[segment].bytes;
    if (!inBounds(dst, len, ctx.memory_size) or !inBounds(src, len, bytes.len)) trapOutOfBounds(ctx);
    @memcpy(self.memory()[dst..][0..len], bytes[src..][0..len]);
}
fn dataDrop(ctx: *abi.VmCtx, index: u32) callconv(.c) void {
    const self = owner(ctx);
    if (index >= self.dropped_data.len) trapOutOfBounds(ctx);
    self.dropped_data[index] = true;
}
fn tableSet(ctx: *abi.VmCtx, table: u32, index: u32, value: usize) callconv(.c) void {
    const self = owner(ctx);
    if (table >= self.tables.len or index >= self.tables[table].size) self.stop(.{ .trap = .out_of_bounds_table });
    self.setTableEntry(table, index, value);
}
fn tableGrow(ctx: *abi.VmCtx, value: i64, delta: i32, table: u32) callconv(.c) i32 {
    const self = owner(ctx);
    if (table >= self.tables.len) self.stop(.{ .trap = .out_of_bounds_table });
    const t = &self.tables[table];
    const old = t.size;
    const next = std.math.add(u32, old, @bitCast(delta)) catch return -1;
    if (next > t.max) return -1;
    for (old..next) |i| self.setTableEntry(table, @intCast(i), @bitCast(value));
    t.size = next;
    self.refreshTables();
    return @bitCast(old);
}
fn tableInit(ctx: *abi.VmCtx, segment_table: u64, dst_src: u64, len: u32) callconv(.c) void {
    const self = owner(ctx);
    const segment: u32 = @truncate(segment_table);
    const table: u32 = @truncate(segment_table >> 32);
    const dst: u32 = @truncate(dst_src);
    const src: u32 = @truncate(dst_src >> 32);
    if (table >= self.tables.len or segment >= self.module.elements.len) self.stop(.{ .trap = .out_of_bounds_table });
    const indices = if (self.dropped_elements[segment]) &.{} else self.module.elements[segment].indices;
    if (!inBounds(dst, len, self.tables[table].size) or !inBounds(src, len, indices.len)) self.stop(.{ .trap = .out_of_bounds_table });
    for (indices[src..][0..len], dst..) |index, offset| {
        const pointer = if (index == format.null_function_index) 0 else self.functions[index];
        self.setTableEntry(table, @intCast(offset), pointer);
    }
}
fn elementDrop(ctx: *abi.VmCtx, index: u32) callconv(.c) void {
    const self = owner(ctx);
    if (index >= self.dropped_elements.len) self.stop(.{ .trap = .out_of_bounds_table });
    self.dropped_elements[index] = true;
}

const max_imports = 64;
const ImportFunction = *const fn (*abi.VmCtx, u64, u64, u64, u64, u64) callconv(.c) u64;
const import_pointers = blk: {
    var pointers: [max_imports]ImportFunction = undefined;
    for (0..max_imports) |i| pointers[i] = &ImportThunk(i).call;
    break :blk pointers;
};
fn ImportThunk(comptime index: usize) type {
    return struct {
        fn call(ctx: *abi.VmCtx, a: u64, b: u64, c: u64, d: u64, e: u64) callconv(.c) u64 {
            const self = owner(ctx);
            const host = self.hosts[index];
            const raw = [_]u64{ a, b, c, d, e };
            var args: [5]Value = undefined;
            var results: [1]Value = .{.{ .i32 = 0 }};
            for (host.params, 0..) |t, i| args[i] = Value.fromRaw(t, raw[i]);
            if (host.results.len == 1) results[0] = Value.fromRaw(host.results[0], 0);
            var context: HostContext = .{ .instance = self };
            host.callback(host.context, &context, args[0..host.params.len], results[0..host.results.len]) catch |err| {
                self.stop(self.pending orelse .{ .host_error = err });
            };
            if (self.pending) |outcome| self.stop(outcome);
            if (host.results.len == 0) return 0;
            if (std.meta.activeTag(results[0]) != host.results[0]) self.stop(.{ .trap = .bad_host_result });
            return results[0].raw();
        }
    };
}

fn invokeRaw(address: usize, ctx: *abi.VmCtx, a: *const [16]u64) u64 {
    // The generated SysV entry ABI ignores trailing scalar slots. Passing all
    // slots also handles stack arguments without executable bridge generation.
    const F = *const fn (*abi.VmCtx, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
    const f: F = @ptrFromInt(address);
    return f(ctx, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
}
