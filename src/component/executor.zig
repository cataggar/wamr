//! Component function executor — bridge between component-level calls and core Wasm.
//!
//! Implements the Canonical ABI's lift/lower flow for calling component functions:
//! 1. Look up the exported function and its canon lift options
//! 2. Lower component-level args into core Wasm values (or linear memory)
//! 3. Execute the core function via the interpreter
//! 4. Lift core results back to component-level values
//! 5. Execute post-return if defined
//!
//! See: https://github.com/WebAssembly/component-model/blob/main/design/mvp/CanonicalABI.md

const std = @import("std");
const ctypes = @import("types.zig");
const abi = @import("canonical_abi.zig");
const instance_mod = @import("instance.zig");
const core_types = @import("../runtime/common/types.zig");
const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
const interp = @import("../runtime/interpreter/interp.zig");
const Allocator = std.mem.Allocator;

const ComponentInstance = instance_mod.ComponentInstance;
const InterfaceValue = abi.InterfaceValue;
const TypeRegistry = abi.TypeRegistry;

pub const MAX_FLAT_PARAMS: u32 = 16;
pub const MAX_FLAT_RESULTS: u32 = 1;

// ── Error types ─────────────────────────────────────────────────────────────

pub const ExecutionError = error{
    FunctionNotFound,
    CoreInstanceNotAvailable,
    MemoryNotAvailable,
    ReallocNotAvailable,
    InvalidFuncType,
    ReallocFailed,
    TrapInCoreFunction,
    StackOverflow,
    StackUnderflow,
    OutOfMemory,
    PostReturnFailed,
    LiftError,
    LowerError,
};

// ── Lift options parsed from CanonOpt array ─────────────────────────────────

/// Parsed canonical options for a lifted function.
pub const LiftOptions = struct {
    memory_idx: ?u32 = null,
    realloc_idx: ?u32 = null,
    post_return_idx: ?u32 = null,
    string_encoding: ctypes.StringEncoding = .utf8,

    pub fn fromOpts(opts: []const ctypes.CanonOpt) LiftOptions {
        var lo = LiftOptions{};
        for (opts) |opt| {
            switch (opt) {
                .memory => |idx| lo.memory_idx = idx,
                .realloc => |idx| lo.realloc_idx = idx,
                .post_return => |idx| lo.post_return_idx = idx,
                .string_encoding => |enc| lo.string_encoding = enc,
            }
        }
        return lo;
    }
};

// ── Realloc ─────────────────────────────────────────────────────────────────

/// Call the core module's realloc function: (old_ptr, old_size, align, new_size) -> ptr.
pub fn callRealloc(
    env: *ExecEnv,
    realloc_idx: u32,
    old_ptr: u32,
    old_size: u32,
    align_val: u32,
    new_size: u32,
) ExecutionError!u32 {
    // Push the 4 i32 arguments
    env.pushI32(@bitCast(old_ptr)) catch return error.StackOverflow;
    env.pushI32(@bitCast(old_size)) catch return error.StackOverflow;
    env.pushI32(@bitCast(align_val)) catch return error.StackOverflow;
    env.pushI32(@bitCast(new_size)) catch return error.StackOverflow;

    // Call the realloc function
    interp.executeFunction(env, realloc_idx) catch return error.ReallocFailed;

    // Pop the i32 result
    const result = env.popI32() catch return error.StackUnderflow;
    return @bitCast(result);
}

// ── Core function calling ───────────────────────────────────────────────────

/// Call a component-exported function by name.
///
/// This implements the `canon lift` flow:
/// 1. Look up the export and its canonical options
/// 2. Get the component function type to know param/result types
/// 3. Lower args: flatten params, if > MAX_FLAT_PARAMS spill to memory
/// 4. Call the core Wasm function
/// 5. Lift results: if > MAX_FLAT_RESULTS, read from memory pointer
/// 6. Call post-return if defined
pub fn callComponentFunc(
    comp_inst: *const ComponentInstance,
    func_name: []const u8,
    args: []const InterfaceValue,
    out_results: []InterfaceValue,
    allocator: Allocator,
) ExecutionError!void {
    // 1. Look up the exported function
    const exported = comp_inst.getExport(func_name) orelse return error.FunctionNotFound;

    // Get the core module instance
    if (exported.core_instance_idx >= comp_inst.core_instances.len)
        return error.CoreInstanceNotAvailable;
    const core_entry = comp_inst.core_instances[exported.core_instance_idx];
    const module_inst = core_entry.module_inst orelse return error.CoreInstanceNotAvailable;

    // Parse canonical options
    const lift_opts = LiftOptions.fromOpts(exported.opts);

    // Get the type registry
    const registry = TypeRegistry.init(comp_inst.component);

    // Resolve the function type
    const func_type = blk: {
        if (exported.func_type_idx < comp_inst.component.types.len) {
            const td = comp_inst.component.types[exported.func_type_idx];
            switch (td) {
                .func => |ft| break :blk ft,
                else => return error.InvalidFuncType,
            }
        }
        return error.InvalidFuncType;
    };

    // 2. Compute flat counts for params and results
    const param_types = getParamValTypes(func_type, allocator) catch return error.OutOfMemory;
    defer allocator.free(param_types);
    const result_types = getResultValTypes(func_type, allocator) catch return error.OutOfMemory;
    defer allocator.free(result_types);

    const flat_param_count = countFlatTypes(registry, param_types);
    const flat_result_count = countFlatTypes(registry, result_types);

    // 3. Create an ExecEnv for the core call
    const env = ExecEnv.create(module_inst, 4096, allocator) catch return error.OutOfMemory;
    defer env.destroy();

    // Get memory if needed
    const memory: ?[]u8 = if (lift_opts.memory_idx) |mem_idx|
        if (module_inst.getMemory(mem_idx)) |mem| mem.data else null
    else
        null;

    // 4. Lower args onto the core stack
    if (flat_param_count <= MAX_FLAT_PARAMS) {
        // Flatten each arg and push as core values
        for (args, param_types) |arg, pt| {
            pushInterfaceValue(env, arg, pt, registry) catch return error.LowerError;
        }
    } else {
        // Spill to memory: allocate space via realloc, store tuple, push ptr
        const mem = memory orelse return error.MemoryNotAvailable;
        const realloc_idx = lift_opts.realloc_idx orelse return error.ReallocNotAvailable;
        const tuple_size = computeTupleSize(registry, param_types);
        const tuple_align = computeTupleAlign(registry, param_types);
        const ptr = try callRealloc(env, realloc_idx, 0, 0, tuple_align, tuple_size);

        // Store each arg at its offset in the tuple
        var offset: u32 = 0;
        for (args, param_types) |arg, pt| {
            const al = typeAlign(registry, pt);
            offset = abi.alignUp(offset, al);
            storeInterfaceValue(mem, offset, arg, pt, registry);
            offset += typeSize(registry, pt);
        }

        env.pushI32(@bitCast(ptr)) catch return error.StackOverflow;
    }

    // 5. Call the core function
    interp.executeFunction(env, exported.core_func_idx) catch return error.TrapInCoreFunction;

    // 6. Lift results
    if (result_types.len == 0) {
        // No results — nothing to lift
    } else if (flat_result_count <= MAX_FLAT_RESULTS) {
        // Results are on the stack as flat values
        for (result_types, 0..) |rt, i| {
            out_results[i] = popInterfaceValue(env, rt, registry, allocator) catch return error.LiftError;
        }
    } else {
        // Results were stored in memory; core returned a pointer
        const result_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
        const mem = memory orelse return error.MemoryNotAvailable;

        var offset: u32 = result_ptr;
        for (result_types, 0..) |rt, i| {
            const al = typeAlign(registry, rt);
            offset = abi.alignUp(offset, al);
            out_results[i] = loadInterfaceValue(mem, offset, rt, registry, allocator) catch return error.LiftError;
            offset += typeSize(registry, rt);
        }
    }

    // 7. Post-return callback
    if (lift_opts.post_return_idx) |pr_idx| {
        // Push the flat results back for post_return to consume
        if (flat_result_count <= MAX_FLAT_RESULTS) {
            for (out_results[0..result_types.len], result_types) |r, rt| {
                pushInterfaceValue(env, r, rt, registry) catch {};
            }
        } else {
            // For spilled results, post_return receives the same pointer
            // (already consumed from stack, but we can re-push it)
        }
        interp.executeFunction(env, pr_idx) catch {};
    }
}

// ── Helper: extract ValType arrays from FuncType ────────────────────────────

fn getParamValTypes(ft: ctypes.FuncType, allocator: Allocator) ![]ctypes.ValType {
    const types = try allocator.alloc(ctypes.ValType, ft.params.len);
    for (ft.params, 0..) |p, i| types[i] = p.type;
    return types;
}

fn getResultValTypes(ft: ctypes.FuncType, allocator: Allocator) ![]ctypes.ValType {
    return switch (ft.results) {
        .unnamed => |t| {
            const types = try allocator.alloc(ctypes.ValType, 1);
            types[0] = t;
            return types;
        },
        .named => |named| {
            const types = try allocator.alloc(ctypes.ValType, named.len);
            for (named, 0..) |n, i| types[i] = n.type;
            return types;
        },
    };
}

// ── Helper: count flat core values for a set of types ───────────────────────

fn countFlatTypes(registry: TypeRegistry, types: []const ctypes.ValType) u32 {
    var count: u32 = 0;
    for (types) |t| {
        count += abi.flattenCountDef(registry, t);
    }
    return count;
}

// ── Helper: compute tuple layout ────────────────────────────────────────────

fn computeTupleSize(registry: TypeRegistry, types: []const ctypes.ValType) u32 {
    var size: u32 = 0;
    var max_align: u32 = 1;
    for (types) |t| {
        const al = typeAlign(registry, t);
        size = abi.alignUp(size, al);
        size += typeSize(registry, t);
        if (al > max_align) max_align = al;
    }
    return abi.alignUp(size, max_align);
}

fn computeTupleAlign(registry: TypeRegistry, types: []const ctypes.ValType) u32 {
    var max_align: u32 = 1;
    for (types) |t| {
        const al = typeAlign(registry, t);
        if (al > max_align) max_align = al;
    }
    return max_align;
}

/// Type alignment, using registry for compounds.
fn typeAlign(registry: TypeRegistry, t: ctypes.ValType) u32 {
    const a = abi.alignment(t);
    if (a > 0) return a;
    return abi.alignOfType(registry, t);
}

/// Type size, using registry for compounds.
fn typeSize(registry: TypeRegistry, t: ctypes.ValType) u32 {
    const s = abi.elemSize(t);
    if (s > 0) return s;
    return abi.sizeOfType(registry, t);
}

// ── Helper: push/pop interface values as core stack values ──────────────────

fn pushInterfaceValue(env: *ExecEnv, val: InterfaceValue, t: ctypes.ValType, registry: TypeRegistry) !void {
    _ = registry;
    switch (t) {
        .bool => try env.pushI32(if (val.bool) 1 else 0),
        .s8 => try env.pushI32(@as(i32, val.s8)),
        .u8 => try env.pushI32(@as(i32, @intCast(val.u8))),
        .s16 => try env.pushI32(@as(i32, val.s16)),
        .u16 => try env.pushI32(@as(i32, @intCast(val.u16))),
        .s32 => try env.pushI32(val.s32),
        .u32, .char => try env.pushI32(@bitCast(val.u32)),
        .s64 => try env.pushI64(val.s64),
        .u64 => try env.pushI64(@bitCast(val.u64)),
        .f32 => try env.push(.{ .f32 = @bitCast(val.f32) }),
        .f64 => try env.push(.{ .f64 = @bitCast(val.f64) }),
        .own, .borrow => try env.pushI32(@bitCast(val.handle)),
        .string => {
            try env.pushI32(@bitCast(val.string.ptr));
            try env.pushI32(@bitCast(val.string.len));
        },
        .list => {
            try env.pushI32(@bitCast(val.list.ptr));
            try env.pushI32(@bitCast(val.list.len));
        },
        // Compound types that need flattening are handled by the spill path
        .record, .variant, .tuple, .flags, .enum_, .option, .result, .type_idx => {
            return error.CompoundNeedsRegistry;
        },
    }
}

fn popInterfaceValue(env: *ExecEnv, t: ctypes.ValType, registry: TypeRegistry, allocator: Allocator) !InterfaceValue {
    _ = registry;
    _ = allocator;
    return switch (t) {
        .bool => .{ .bool = (try env.popI32()) != 0 },
        .s8 => .{ .s8 = @truncate(try env.popI32()) },
        .u8 => .{ .u8 = @truncate(@as(u32, @bitCast(try env.popI32()))) },
        .s16 => .{ .s16 = @truncate(try env.popI32()) },
        .u16 => .{ .u16 = @truncate(@as(u32, @bitCast(try env.popI32()))) },
        .s32 => .{ .s32 = try env.popI32() },
        .u32, .char => .{ .u32 = @bitCast(try env.popI32()) },
        .s64 => .{ .s64 = try env.popI64() },
        .u64 => .{ .u64 = @bitCast(try env.popI64()) },
        .f32 => blk: {
            const v = try env.pop();
            break :blk .{ .f32 = switch (v) {
                .f32 => |f| @bitCast(f),
                .i32 => |i| @bitCast(i),
                else => 0,
            } };
        },
        .f64 => blk: {
            const v = try env.pop();
            break :blk .{ .f64 = switch (v) {
                .f64 => |f| @bitCast(f),
                .i64 => |i| @bitCast(i),
                else => 0,
            } };
        },
        .own, .borrow => .{ .handle = @bitCast(try env.popI32()) },
        .string => .{ .string = .{
            .len = @bitCast(try env.popI32()),
            .ptr = @bitCast(try env.popI32()),
        } },
        .list => .{ .list = .{
            .len = @bitCast(try env.popI32()),
            .ptr = @bitCast(try env.popI32()),
        } },
        .record, .variant, .tuple, .flags, .enum_, .option, .result, .type_idx => error.CompoundNeedsRegistry,
    };
}

// ── Helper: load/store interface values from/to linear memory ───────────────

fn loadInterfaceValue(
    mem: []const u8,
    ptr: u32,
    t: ctypes.ValType,
    registry: TypeRegistry,
    allocator: Allocator,
) !InterfaceValue {
    // Try primitive first
    const prim = abi.loadVal(mem, ptr, t) catch |err| switch (err) {
        error.CompoundNeedsRegistry => {
            // Use registry-aware compound loading
            return abi.loadValReg(mem, ptr, t, registry, allocator);
        },
        else => return error.LiftError,
    };
    return prim;
}

fn storeInterfaceValue(
    mem: []u8,
    ptr: u32,
    val: InterfaceValue,
    t: ctypes.ValType,
    registry: TypeRegistry,
) void {
    abi.storeVal(mem, ptr, t, val) catch {
        // Compound type — use registry-aware store
        abi.storeValReg(mem, ptr, t, val, registry) catch {};
    };
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "LiftOptions: parse from CanonOpt array" {
    const opts = [_]ctypes.CanonOpt{
        .{ .memory = 0 },
        .{ .realloc = 1 },
        .{ .string_encoding = .utf16 },
        .{ .post_return = 2 },
    };
    const lo = LiftOptions.fromOpts(&opts);
    try std.testing.expectEqual(@as(?u32, 0), lo.memory_idx);
    try std.testing.expectEqual(@as(?u32, 1), lo.realloc_idx);
    try std.testing.expectEqual(@as(?u32, 2), lo.post_return_idx);
    try std.testing.expectEqual(ctypes.StringEncoding.utf16, lo.string_encoding);
}

test "LiftOptions: defaults" {
    const lo = LiftOptions.fromOpts(&.{});
    try std.testing.expectEqual(@as(?u32, null), lo.memory_idx);
    try std.testing.expectEqual(@as(?u32, null), lo.realloc_idx);
    try std.testing.expectEqual(@as(?u32, null), lo.post_return_idx);
    try std.testing.expectEqual(ctypes.StringEncoding.utf8, lo.string_encoding);
}

test "countFlatTypes: primitives" {
    const reg = TypeRegistry.fromTypes(&.{});
    const types = [_]ctypes.ValType{ .s32, .s32, .f64 };
    try std.testing.expectEqual(@as(u32, 3), countFlatTypes(reg, &types));
}

test "countFlatTypes: string is 2 flat values" {
    const reg = TypeRegistry.fromTypes(&.{});
    const types = [_]ctypes.ValType{ .string, .s32 };
    try std.testing.expectEqual(@as(u32, 3), countFlatTypes(reg, &types));
}

test "computeTupleSize and align" {
    const reg = TypeRegistry.fromTypes(&.{});
    // (i32, i64) → size = align(4, 8) + 8 = 16, align = 8
    const types = [_]ctypes.ValType{ .s32, .s64 };
    try std.testing.expectEqual(@as(u32, 16), computeTupleSize(reg, &types));
    try std.testing.expectEqual(@as(u32, 8), computeTupleAlign(reg, &types));
}

test "getParamValTypes and getResultValTypes" {
    const allocator = std.testing.allocator;
    const ft = ctypes.FuncType{
        .params = &[_]ctypes.NamedValType{
            .{ .name = "a", .type = .s32 },
            .{ .name = "b", .type = .f64 },
        },
        .results = .{ .unnamed = .s32 },
    };

    const param_types = try getParamValTypes(ft, allocator);
    defer allocator.free(param_types);
    try std.testing.expectEqual(@as(usize, 2), param_types.len);
    try std.testing.expectEqual(ctypes.ValType.s32, param_types[0]);
    try std.testing.expectEqual(ctypes.ValType.f64, param_types[1]);

    const result_types = try getResultValTypes(ft, allocator);
    defer allocator.free(result_types);
    try std.testing.expectEqual(@as(usize, 1), result_types.len);
    try std.testing.expectEqual(ctypes.ValType.s32, result_types[0]);
}

test "InterfaceValue.deinit: primitives are no-op" {
    const allocator = std.testing.allocator;
    const v = InterfaceValue{ .s32 = 42 };
    v.deinit(allocator); // should not crash
}

test "InterfaceValue.deinit: record" {
    const allocator = std.testing.allocator;
    const fields = try allocator.alloc(InterfaceValue, 2);
    fields[0] = .{ .s32 = 1 };
    fields[1] = .{ .u32 = 2 };
    const v = InterfaceValue{ .record_val = fields };
    v.deinit(allocator); // frees the slice
}

test "InterfaceValue.deinit: nested record" {
    const allocator = std.testing.allocator;
    // Inner record
    const inner = try allocator.alloc(InterfaceValue, 1);
    inner[0] = .{ .bool = true };
    // Outer record containing inner
    const outer = try allocator.alloc(InterfaceValue, 2);
    outer[0] = .{ .s32 = 42 };
    outer[1] = .{ .record_val = inner };
    const v = InterfaceValue{ .record_val = outer };
    v.deinit(allocator); // frees both inner and outer
}
