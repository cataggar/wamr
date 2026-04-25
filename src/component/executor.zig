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
        const td = registry.get(exported.func_type_idx) orelse return error.InvalidFuncType;
        switch (td) {
            .func => |ft| break :blk ft,
            else => return error.InvalidFuncType,
        }
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
    var result_ptr_for_post_return: u32 = 0;
    if (result_types.len == 0) {
        // No results — nothing to lift
    } else if (flat_result_count <= MAX_FLAT_RESULTS) {
        // Results are on the stack as flat values
        for (result_types, 0..) |rt, i| {
            out_results[i] = popInterfaceValue(env, rt, registry, allocator) catch return error.LiftError;
        }
    } else {
        // Results were stored in memory; core returned a pointer
        result_ptr_for_post_return = @bitCast(env.popI32() catch return error.StackUnderflow);
        const mem = memory orelse return error.MemoryNotAvailable;

        var offset: u32 = result_ptr_for_post_return;
        for (result_types, 0..) |rt, i| {
            const al = typeAlign(registry, rt);
            offset = abi.alignUp(offset, al);
            out_results[i] = loadInterfaceValue(mem, offset, rt, registry, allocator) catch return error.LiftError;
            offset += typeSize(registry, rt);
        }
    }

    // 7. Post-return callback
    if (lift_opts.post_return_idx) |pr_idx| {
        // Per spec: post_return receives the flat result value(s).
        // For inline results (≤ MAX_FLAT_RESULTS): re-push the flat values.
        // For spilled results: re-push the result pointer as i32.
        if (flat_result_count <= MAX_FLAT_RESULTS) {
            for (out_results[0..result_types.len], result_types) |r, rt| {
                pushInterfaceValue(env, r, rt, registry) catch {};
            }
        } else {
            // Spilled results: post_return receives the result pointer.
            // We've already read the ptr above; push it back for post_return.
            env.pushI32(@bitCast(result_ptr_for_post_return)) catch {};
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
        .none => try allocator.alloc(ctypes.ValType, 0),
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
        count += abi.flattenCount(registry, t);
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
        // result<T, E>: flat repr is `[i32 disc] ++ join(flatten(T), flatten(E))`,
        // where the per-slot join takes the wider of the two arms (treated as
        // i32 here since stdio-echo's variants land on i32-only payloads).
        // Inactive slots are zero-filled; payload values for the active arm
        // are recursively pushed and any remaining slots are then zero-filled.
        // A future slice can extend this to mixed i32/i64/f32/f64 joins.
        .result => |idx| {
            const td = registry.get(idx) orelse return error.CompoundNeedsRegistry;
            const r = switch (td) {
                .result => |rt| rt,
                else => return error.CompoundNeedsRegistry,
            };
            const total_payload_slots = abi.flattenCount(registry, t) - 1;
            const disc: i32 = if (val.result_val.is_ok) 0 else 1;
            try env.pushI32(disc);

            const arm_type: ?ctypes.ValType = if (val.result_val.is_ok) r.ok else r.err;
            var pushed: u32 = 0;
            if (arm_type) |at| {
                if (val.result_val.payload) |p| {
                    try pushInterfaceValue(env, p.*, at, registry);
                    pushed = abi.flattenCount(registry, at);
                }
            }
            while (pushed < total_payload_slots) : (pushed += 1) {
                try env.pushI32(0);
            }
        },
        .record, .variant, .tuple, .flags, .enum_, .option, .type_idx => {
            return error.CompoundNeedsRegistry;
        },
    }
}

fn popInterfaceValue(env: *ExecEnv, t: ctypes.ValType, registry: TypeRegistry, allocator: Allocator) !InterfaceValue {
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
        // result<T, E>: pop the discriminant, then pop and discard all
        // remaining payload slots (caller currently doesn't inspect the
        // typed payload; a future slice can lift it through `loadValReg`-
        // style logic). See pushInterfaceValue for the join rationale.
        .result => |idx| blk: {
            const td = registry.get(idx) orelse return error.CompoundNeedsRegistry;
            _ = switch (td) {
                .result => |rt| rt,
                else => return error.CompoundNeedsRegistry,
            };
            const total_payload_slots = abi.flattenCount(registry, t) - 1;
            // Payload slots are pushed last, so they pop first.
            var i: u32 = 0;
            while (i < total_payload_slots) : (i += 1) {
                _ = try env.popI32();
            }
            const disc = try env.popI32();
            break :blk .{ .result_val = .{ .is_ok = disc == 0, .payload = null } };
        },
        .record, .variant, .tuple, .flags, .enum_, .option, .type_idx => error.CompoundNeedsRegistry,
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
        inline else => |e| return e,
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

// ── Canonical built-in functions ─────────────────────────────────────────────

const ResourceTable = instance_mod.ResourceTable;

/// Execute `resource.new(rep) → handle`: allocate a new resource handle.
pub fn canonResourceNew(
    resource_table: *ResourceTable,
    representation: u32,
    allocator: Allocator,
) ExecutionError!u32 {
    return resource_table.new(representation, true, allocator) catch return error.OutOfMemory;
}

/// Execute `resource.drop(handle)`: deallocate a resource handle.
/// Returns the representation for the caller to invoke the destructor.
pub fn canonResourceDrop(
    resource_table: *ResourceTable,
    handle: u32,
    allocator: Allocator,
) ?u32 {
    return resource_table.drop(handle, allocator);
}

/// Execute `resource.rep(handle) → rep`: get the representation for a handle.
pub fn canonResourceRep(
    resource_table: *const ResourceTable,
    handle: u32,
) ?u32 {
    return resource_table.rep(handle);
}

/// Dispatch a canonical built-in function call. Used when the canon section
/// references resource.new/drop/rep instead of lift/lower.
pub fn dispatchCanonBuiltin(
    comp_inst: *ComponentInstance,
    canon: ctypes.Canon,
    env: *ExecEnv,
    allocator: Allocator,
) ExecutionError!void {
    switch (canon) {
        .resource_new => |resource_idx| {
            const rt = comp_inst.getOrCreateResourceTable(resource_idx) catch
                return error.FunctionNotFound;
            const rep_val: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const handle = try canonResourceNew(rt, rep_val, allocator);
            env.pushI32(@bitCast(handle)) catch return error.StackOverflow;
        },
        .resource_drop => |resource_idx| {
            const rt = comp_inst.getOrCreateResourceTable(resource_idx) catch
                return error.FunctionNotFound;
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            _ = canonResourceDrop(rt, handle, allocator);
        },
        .resource_rep => |resource_idx| {
            const rt = comp_inst.getOrCreateResourceTable(resource_idx) catch
                return error.FunctionNotFound;
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const rep_val = canonResourceRep(rt, handle) orelse 0;
            env.pushI32(@bitCast(rep_val)) catch return error.StackOverflow;
        },
        .lift, .lower => {}, // Handled by callComponentFunc
    }
}

// ── Async execution ─────────────────────────────────────────────────────────

const async_mod = @import("async.zig");
const async_canon = @import("async_canon.zig");

/// Start an async component function call. Returns a subtask handle
/// that the caller can poll via the waitable set.
///
/// Unlike `callComponentFunc`, this does NOT block — it creates a task,
/// starts it, and returns immediately. The caller polls the waitable set
/// to discover when results are available.
///
/// Note: In the current single-threaded implementation, the core function
/// is still executed synchronously and results are stored in the task.
/// True cooperative scheduling will require runtime loop integration.
pub fn callComponentFuncAsync(
    comp_inst: *const ComponentInstance,
    func_name: []const u8,
    args: []const InterfaceValue,
    task_manager: *async_mod.TaskManager,
    waitable_set: ?*async_mod.WaitableSet,
    allocator: Allocator,
) ExecutionError!u32 {
    // Create the subtask
    const lift_result = async_canon.asyncLift(.{
        .waitable_set = waitable_set,
        .task_manager = task_manager,
        .allocator = allocator,
    }) catch return error.OutOfMemory;
    const handle = lift_result.subtask_handle;

    // Look up the exported function to determine result count
    const exported = comp_inst.getExport(func_name) orelse {
        task_manager.cancelTask(handle);
        return error.FunctionNotFound;
    };

    // Get function type for result count
    const result_count: usize = blk: {
        const reg = TypeRegistry.init(comp_inst.component);
        if (reg.get(exported.func_type_idx)) |td| {
            switch (td) {
                .func => |ft| {
                    switch (ft.results) {
                        .none => break :blk 0,
                        .unnamed => break :blk 1,
                        .named => |named| break :blk named.len,
                    }
                },
                else => break :blk 0,
            }
        }
        break :blk 0;
    };

    // Execute synchronously (current impl — no real cooperative scheduling)
    const results = allocator.alloc(InterfaceValue, result_count) catch {
        task_manager.cancelTask(handle);
        return error.OutOfMemory;
    };

    callComponentFunc(comp_inst, func_name, args, results, allocator) catch |e| {
        allocator.free(results);
        task_manager.cancelTask(handle);
        return e;
    };

    // Store flat results in the task (as u32 representation)
    const flat_results = allocator.alloc(u32, result_count) catch {
        for (results) |r| r.deinit(allocator);
        allocator.free(results);
        task_manager.cancelTask(handle);
        return error.OutOfMemory;
    };
    for (results, 0..) |r, i| {
        flat_results[i] = switch (r) {
            .s32 => |v| @bitCast(v),
            .u32 => |v| v,
            .bool => |v| @intFromBool(v),
            else => 0,
        };
        r.deinit(allocator);
    }
    allocator.free(results);

    async_canon.asyncReturn(task_manager, handle, flat_results);

    return handle;
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

test "canonResourceNew and canonResourceRep" {
    const allocator = std.testing.allocator;
    var table = ResourceTable{};
    defer table.deinit(allocator);

    const handle = try canonResourceNew(&table, 42, allocator);
    try std.testing.expectEqual(@as(?u32, 42), canonResourceRep(&table, handle));
}

test "canonResourceDrop" {
    const allocator = std.testing.allocator;
    var table = ResourceTable{};
    defer table.deinit(allocator);

    const handle = try canonResourceNew(&table, 99, allocator);
    const rep = canonResourceDrop(&table, handle, allocator);
    try std.testing.expectEqual(@as(?u32, 99), rep);
    // After drop, rep returns null
    try std.testing.expectEqual(@as(?u32, null), canonResourceRep(&table, handle));
}

test "canonResourceDrop: double drop returns null" {
    const allocator = std.testing.allocator;
    var table = ResourceTable{};
    defer table.deinit(allocator);

    const handle = try canonResourceNew(&table, 7, allocator);
    _ = canonResourceDrop(&table, handle, allocator);
    // Second drop should return null
    try std.testing.expectEqual(@as(?u32, null), canonResourceDrop(&table, handle, allocator));
}

// ── Async tests ─────────────────────────────────────────────────────────────

test "callComponentFuncAsync: function not found cancels task" {
    const allocator = std.testing.allocator;
    var tm = async_mod.TaskManager{};
    defer tm.deinit(allocator);
    var ws = async_mod.WaitableSet{};
    defer ws.deinit(allocator);

    const comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };

    var inst = try instance_mod.instantiate(&comp, allocator);
    defer inst.deinit();

    const result = callComponentFuncAsync(
        inst,
        "nonexistent",
        &.{},
        &tm,
        &ws,
        allocator,
    );
    try std.testing.expectError(error.FunctionNotFound, result);

    // Task should have been created and then cancelled
    try std.testing.expectEqual(@as(usize, 1), tm.tasks.items.len);
    try std.testing.expectEqual(async_mod.TaskState.cancelled, tm.getState(0).?);
}

test "async poll flow: lift then return then poll" {
    const allocator = std.testing.allocator;
    var tm = async_mod.TaskManager{};
    defer tm.deinit(allocator);
    var ws = async_mod.WaitableSet{};
    defer ws.deinit(allocator);

    // Simulate the async flow manually
    const lift_result = try async_canon.asyncLift(.{
        .waitable_set = &ws,
        .task_manager = &tm,
        .allocator = allocator,
    });

    // Task should be started
    try std.testing.expectEqual(async_mod.TaskState.started, tm.getState(lift_result.subtask_handle).?);

    // Poll before return — should get null
    try std.testing.expect(async_canon.asyncPollResult(&tm, lift_result.subtask_handle) == null);

    // Return values
    var vals = [_]u32{ 42, 99 };
    async_canon.asyncReturn(&tm, lift_result.subtask_handle, &vals);

    // Now poll — should get results
    const ret = async_canon.asyncPollResult(&tm, lift_result.subtask_handle);
    try std.testing.expect(ret != null);
    try std.testing.expectEqual(@as(u32, 42), ret.?[0]);
    try std.testing.expectEqual(@as(u32, 99), ret.?[1]);
}

test "async cancel flow: lift then cancel then poll" {
    const allocator = std.testing.allocator;
    var tm = async_mod.TaskManager{};
    defer tm.deinit(allocator);

    const lift_result = try async_canon.asyncLift(.{
        .task_manager = &tm,
        .allocator = allocator,
    });

    async_canon.asyncCancel(&tm, lift_result.subtask_handle);
    try std.testing.expectEqual(async_mod.TaskState.cancelled, tm.getState(lift_result.subtask_handle).?);
    try std.testing.expect(async_canon.asyncPollResult(&tm, lift_result.subtask_handle) == null);
}

test "async waitable set: multiple subtasks" {
    const allocator = std.testing.allocator;
    var tm = async_mod.TaskManager{};
    defer tm.deinit(allocator);
    var ws = async_mod.WaitableSet{};
    defer ws.deinit(allocator);

    // Create two subtasks
    const r1 = try async_canon.asyncLift(.{
        .waitable_set = &ws,
        .task_manager = &tm,
        .allocator = allocator,
    });
    const r2 = try async_canon.asyncLift(.{
        .waitable_set = &ws,
        .task_manager = &tm,
        .allocator = allocator,
    });

    // Both registered
    try std.testing.expectEqual(@as(usize, 2), ws.items.items.len);

    // Complete first one
    var vals1 = [_]u32{10};
    async_canon.asyncReturn(&tm, r1.subtask_handle, &vals1);

    // First should be ready, second not
    try std.testing.expect(async_canon.asyncPollResult(&tm, r1.subtask_handle) != null);
    try std.testing.expect(async_canon.asyncPollResult(&tm, r2.subtask_handle) == null);

    // Complete second
    var vals2 = [_]u32{20};
    async_canon.asyncReturn(&tm, r2.subtask_handle, &vals2);
    try std.testing.expect(async_canon.asyncPollResult(&tm, r2.subtask_handle) != null);
}

// ── Canon-lower host trampoline ─────────────────────────────────────────────

const core_runtime_types = @import("../runtime/common/types.zig");
const HostFunc = instance_mod.HostFunc;

/// Lower options carved out of the canon.lower opts array. Mirrors
/// `LiftOptions` but is owned by the trampoline context so it's cheap to
/// resolve once at instantiation time instead of on every call.
pub const LowerOptions = struct {
    memory_idx: ?u32 = null,
    realloc_idx: ?u32 = null,
    string_encoding: ctypes.StringEncoding = .utf8,

    pub fn fromOpts(opts: []const ctypes.CanonOpt) LowerOptions {
        var lo = LowerOptions{};
        for (opts) |opt| {
            switch (opt) {
                .memory => |idx| lo.memory_idx = idx,
                .realloc => |idx| lo.realloc_idx = idx,
                .post_return => {},
                .string_encoding => |enc| lo.string_encoding = enc,
            }
        }
        return lo;
    }
};

/// Per-import-slot context for the canon-lower trampoline. Owned by the
/// `ComponentInstance` that installs the trampoline onto a core
/// ModuleInstance's `host_func_entries`.
pub const ComponentTrampolineCtx = struct {
    comp_inst: *ComponentInstance,
    host_func: HostFunc,
    /// Component-level function index this trampoline is lowering. Kept so
    /// `ComponentInstance.linkImports` can re-bind the host_func after the
    /// caller supplies providers.
    component_func_idx: u32 = 0,
    /// Component-level parameter types, cached so `trampoline` doesn't have
    /// to re-walk the FuncType on every call.
    param_types: []const ctypes.ValType,
    /// Component-level result types, same rationale.
    result_types: []const ctypes.ValType,
    lower_opts: LowerOptions,

    pub fn deinit(self: *ComponentTrampolineCtx, allocator: Allocator) void {
        allocator.free(self.param_types);
        allocator.free(self.result_types);
    }
};

/// Shared trampoline entry point installed on every lowered core import.
/// Reads the core arguments off the ExecEnv stack, lifts them into
/// `InterfaceValue`s, invokes the bound `HostFunc`, and lowers the result
/// values back onto the core stack.
///
/// Trampoline executed when a core wasm function imported by `canon.lower`
/// is called. Pops args from the core stack, lifts them to component-level
/// `InterfaceValue`s, invokes the bound `HostFunc`, and lowers the result
/// values back onto the core stack.
///
/// Stack discipline mirrors the rest of the interpreter: args were pushed
/// in natural order by the caller, so the last flat core value is on top.
///
/// Param/result spill follows the canon ABI for `lower`:
/// - `flat_params <= MAX_FLAT_PARAMS`: each param is on the stack as flat
///   core values (back-to-front pop order).
/// - `flat_params > MAX_FLAT_PARAMS`: a single i32 ptr on the stack points
///   at a tuple of params laid out per the canon ABI.
/// - `flat_results <= MAX_FLAT_RESULTS`: results are pushed back as flat
///   core values.
/// - `flat_results > MAX_FLAT_RESULTS`: an additional i32 ptr was pushed by
///   the caller (after the params or param-ptr) into which results must be
///   stored. The trampoline returns nothing.
///
/// Either spill path requires `lower_opts.memory_idx` to resolve the linear
/// memory; if absent the trampoline traps.
pub fn componentTrampoline(env_opaque: *anyopaque, ctx_opaque: ?*anyopaque) core_runtime_types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const ctx: *ComponentTrampolineCtx = @ptrCast(@alignCast(ctx_opaque.?));
    const allocator = ctx.comp_inst.allocator;
    const registry = TypeRegistry.init(ctx.comp_inst.component);

    const flat_params = countFlatTypes(registry, ctx.param_types);
    const flat_results = countFlatTypes(registry, ctx.result_types);
    const params_spill = flat_params > MAX_FLAT_PARAMS;
    const results_spill = flat_results > MAX_FLAT_RESULTS;

    // Resolve linear memory for either spill path. The memory option is
    // mandatory whenever spilling occurs, and we need a non-empty
    // core_instances list to actually own a memory.
    if (params_spill or results_spill) {
        if (ctx.lower_opts.memory_idx == null) return error.Trap;
        if (ctx.comp_inst.core_instances.len == 0) return error.Trap;
    }

    // Pop result-destination pointer first if results spill (it was pushed
    // last by the caller).
    var result_dest_ptr: u32 = 0;
    if (results_spill) {
        result_dest_ptr = @bitCast(env.popI32() catch return error.Trap);
    }

    // Lift args.
    const args = allocator.alloc(InterfaceValue, ctx.param_types.len) catch return error.Trap;
    defer allocator.free(args);
    if (params_spill) {
        const params_ptr: u32 = @bitCast(env.popI32() catch return error.Trap);
        const mem_idx = ctx.lower_opts.memory_idx.?;
        const mem = ctx.comp_inst.resolveTopLevelMemory(mem_idx) orelse return error.Trap;
        var offset: u32 = params_ptr;
        for (ctx.param_types, 0..) |pt, i| {
            const al = typeAlign(registry, pt);
            offset = abi.alignUp(offset, al);
            args[i] = loadInterfaceValue(mem.data, offset, pt, registry, allocator) catch return error.Trap;
            offset += typeSize(registry, pt);
        }
    } else {
        // Walk param_types back-to-front so the last flat core value
        // (which is on top of stack) becomes the last arg.
        var i: usize = ctx.param_types.len;
        while (i > 0) {
            i -= 1;
            args[i] = popInterfaceValue(env, ctx.param_types[i], registry, allocator) catch return error.Trap;
        }
    }

    // Invoke host. Host owns allocation of any compound result values via
    // `allocator`; we deinit each result after lowering so payloads
    // (e.g. `.result_val.payload` for input-stream.blocking-read) don't leak.
    const results = allocator.alloc(InterfaceValue, ctx.result_types.len) catch return error.Trap;
    defer {
        for (results) |r| r.deinit(allocator);
        allocator.free(results);
    }
    const call = ctx.host_func.call orelse return error.Trap;
    call(ctx.host_func.context, ctx.comp_inst, args, results, allocator) catch return error.Trap;

    // Lower results.
    if (results_spill) {
        const mem_idx = ctx.lower_opts.memory_idx.?;
        const mem = ctx.comp_inst.resolveTopLevelMemory(mem_idx) orelse return error.Trap;
        var offset: u32 = result_dest_ptr;
        for (results, ctx.result_types) |r, t| {
            const al = typeAlign(registry, t);
            offset = abi.alignUp(offset, al);
            storeInterfaceValue(mem.data, offset, r, t, registry);
            offset += typeSize(registry, t);
        }
    } else {
        for (results, ctx.result_types) |r, t| {
            pushInterfaceValue(env, r, t, registry) catch return error.Trap;
        }
    }
}

// ── Trampoline tests ────────────────────────────────────────────────────────

test "componentTrampoline: flat i32 host func with per-slot ctx" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    // Fake a minimal core module with one imported function (i32, i32) -> i32.
    const imports = [_]core_types_mod.ImportDesc{
        .{ .module_name = "host", .field_name = "sub", .kind = .function, .func_type_idx = 0 },
    };
    const func_types = [_]core_types_mod.FuncType{
        .{ .params = &.{ .i32, .i32 }, .results = &.{.i32} },
    };
    var module = core_types_mod.WasmModule{
        .imports = &imports,
        .import_function_count = 1,
        .types = &func_types,
    };
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);

    // Build a minimal component whose HostFunc computes a - b (non-commutative).
    const Component = ctypes.Component;
    var component = Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    const comp_inst = try instance_mod.instantiate(&component, testing.allocator);
    defer comp_inst.deinit();

    const Host = struct {
        fn sub(
            _: ?*anyopaque,
            _: *ComponentInstance,
            in: []const InterfaceValue,
            out: []InterfaceValue,
            _: Allocator,
        ) anyerror!void {
            out[0] = .{ .s32 = in[0].s32 - in[1].s32 };
        }
    };

    const param_types = try testing.allocator.alloc(ctypes.ValType, 2);
    param_types[0] = .s32;
    param_types[1] = .s32;
    const result_types = try testing.allocator.alloc(ctypes.ValType, 1);
    result_types[0] = .s32;

    var tctx = ComponentTrampolineCtx{
        .comp_inst = comp_inst,
        .host_func = .{ .call = &Host.sub },
        .param_types = param_types,
        .result_types = result_types,
        .lower_opts = .{},
    };
    defer tctx.deinit(testing.allocator);

    const entries = try testing.allocator.alloc(?core_types_mod.HostFnEntry, 1);
    entries[0] = .{ .func = &componentTrampoline, .ctx = @ptrCast(&tctx) };
    inst_mod_core.attachHostFuncEntries(core_inst, entries);

    // Call via the interpreter's normal dispatch path: push args (7, 2),
    // executeFunction on import slot 0, expect 7 - 2 = 5.
    const env = try ExecEnv.create(core_inst, 256, testing.allocator);
    defer env.destroy();
    try env.pushI32(7);
    try env.pushI32(2);
    try @import("../runtime/interpreter/interp.zig").executeFunction(env, 0);
    try testing.expectEqual(@as(i32, 5), try env.popI32());
}

test "componentTrampoline: traps on spill without memory option" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    // Build a core module with one import taking 17 i32 params (1 over MAX).
    var param_kinds: [17]core_types_mod.ValType = undefined;
    for (&param_kinds) |*p| p.* = .i32;
    const imports = [_]core_types_mod.ImportDesc{
        .{ .module_name = "host", .field_name = "many", .kind = .function, .func_type_idx = 0 },
    };
    const func_types = [_]core_types_mod.FuncType{
        .{ .params = &param_kinds, .results = &.{} },
    };
    var module = core_types_mod.WasmModule{
        .imports = &imports,
        .import_function_count = 1,
        .types = &func_types,
    };
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);

    var component = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    const comp_inst = try instance_mod.instantiate(&component, testing.allocator);
    defer comp_inst.deinit();

    const Host = struct {
        fn noop(
            _: ?*anyopaque,
            _: *ComponentInstance,
            _: []const InterfaceValue,
            _: []InterfaceValue,
            _: Allocator,
        ) anyerror!void {}
    };

    const param_types = try testing.allocator.alloc(ctypes.ValType, 17);
    for (param_types) |*p| p.* = .s32;
    const result_types = try testing.allocator.alloc(ctypes.ValType, 0);

    var tctx = ComponentTrampolineCtx{
        .comp_inst = comp_inst,
        .host_func = .{ .call = &Host.noop },
        .param_types = param_types,
        .result_types = result_types,
        .lower_opts = .{},
    };
    defer tctx.deinit(testing.allocator);

    const entries = try testing.allocator.alloc(?core_types_mod.HostFnEntry, 1);
    entries[0] = .{ .func = &componentTrampoline, .ctx = @ptrCast(&tctx) };
    inst_mod_core.attachHostFuncEntries(core_inst, entries);

    const env = try ExecEnv.create(core_inst, 256, testing.allocator);
    defer env.destroy();
    var n: i32 = 0;
    while (n < 17) : (n += 1) try env.pushI32(n);
    // executeFunction wraps HostFnError -> error.Unreachable for the trap.
    try std.testing.expectError(error.Unreachable, @import("../runtime/interpreter/interp.zig").executeFunction(env, 0));
}

test "componentTrampoline: param spill loads tuple from memory" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    // Core module with one memory and a host import that takes a single i32
    // (the spilled-params ptr) and returns nothing.
    const imports = [_]core_types_mod.ImportDesc{
        .{ .module_name = "host", .field_name = "many", .kind = .function, .func_type_idx = 0 },
    };
    const func_types = [_]core_types_mod.FuncType{
        .{ .params = &.{.i32}, .results = &.{} },
    };
    const memories = [_]core_types_mod.MemoryType{
        .{ .limits = .{ .min = 1, .max = 1 } },
    };
    var module = core_types_mod.WasmModule{
        .imports = &imports,
        .import_function_count = 1,
        .types = &func_types,
        .memories = &memories,
    };
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    // ownership of core_inst is transferred to comp_inst.core_instances[0]
    // below; comp_inst.deinit will destroy it.

    var component = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},
        .imports = &.{},          .exports = &.{},
    };
    const comp_inst = try instance_mod.instantiate(&component, testing.allocator);
    defer comp_inst.deinit();

    // Inject the core instance into core_instances[0] so the trampoline can
    // find the memory through it.
    const cis = try testing.allocator.alloc(ComponentInstance.CoreInstanceEntry, 1);
    cis[0] = .{ .module_inst = core_inst };
    comp_inst.core_instances = cis;

    // Host fn: capture args into a buffer the test owns.
    const Captured = struct {
        var seen: [17]i32 = undefined;
        var count: usize = 0;
        fn cb(
            _: ?*anyopaque,
            _: *ComponentInstance,
            in: []const InterfaceValue,
            _: []InterfaceValue,
            _: Allocator,
        ) anyerror!void {
            count = in.len;
            for (in, 0..) |v, i| seen[i] = v.s32;
        }
    };
    Captured.count = 0;

    const param_types = try testing.allocator.alloc(ctypes.ValType, 17);
    for (param_types) |*p| p.* = .s32;
    const result_types = try testing.allocator.alloc(ctypes.ValType, 0);

    var tctx = ComponentTrampolineCtx{
        .comp_inst = comp_inst,
        .host_func = .{ .call = &Captured.cb },
        .param_types = param_types,
        .result_types = result_types,
        .lower_opts = .{ .memory_idx = 0 },
    };
    defer tctx.deinit(testing.allocator);

    // Layout 17 s32 args at offset 64 in linear memory.
    const mem = core_inst.getMemory(0).?;
    const base: u32 = 64;
    var k: u32 = 0;
    while (k < 17) : (k += 1) {
        const off = base + k * 4;
        const v: i32 = @as(i32, @intCast(k)) + 100;
        std.mem.writeInt(i32, mem.data[off..][0..4], v, .little);
    }

    const env = try ExecEnv.create(core_inst, 256, testing.allocator);
    defer env.destroy();
    try env.pushI32(@intCast(base));
    try componentTrampoline(env, @ptrCast(&tctx));

    try testing.expectEqual(@as(usize, 17), Captured.count);
    var j: usize = 0;
    while (j < 17) : (j += 1) {
        try testing.expectEqual(@as(i32, @intCast(j)) + 100, Captured.seen[j]);
    }
}

test "componentTrampoline: result spill stores tuple into memory" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    // Imported host fn signature for canon.lower with results spill: takes
    // an i32 dest_ptr (no params) and returns nothing.
    const imports = [_]core_types_mod.ImportDesc{
        .{ .module_name = "host", .field_name = "many_results", .kind = .function, .func_type_idx = 0 },
    };
    const func_types = [_]core_types_mod.FuncType{
        .{ .params = &.{.i32}, .results = &.{} },
    };
    const memories = [_]core_types_mod.MemoryType{
        .{ .limits = .{ .min = 1, .max = 1 } },
    };
    var module = core_types_mod.WasmModule{
        .imports = &imports,
        .import_function_count = 1,
        .types = &func_types,
        .memories = &memories,
    };
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    // ownership transferred to comp_inst.core_instances[0] below.

    var component = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},
        .imports = &.{},          .exports = &.{},
    };
    const comp_inst = try instance_mod.instantiate(&component, testing.allocator);
    defer comp_inst.deinit();

    const cis = try testing.allocator.alloc(ComponentInstance.CoreInstanceEntry, 1);
    cis[0] = .{ .module_inst = core_inst };
    comp_inst.core_instances = cis;

    // Host fn: produce two s32 results.
    const Host = struct {
        fn pair(
            _: ?*anyopaque,
            _: *ComponentInstance,
            _: []const InterfaceValue,
            out: []InterfaceValue,
            _: Allocator,
        ) anyerror!void {
            out[0] = .{ .s32 = 0xCAFE };
            out[1] = .{ .s32 = 0xBEEF };
        }
    };

    const param_types = try testing.allocator.alloc(ctypes.ValType, 0);
    const result_types = try testing.allocator.alloc(ctypes.ValType, 2);
    result_types[0] = .s32;
    result_types[1] = .s32;

    var tctx = ComponentTrampolineCtx{
        .comp_inst = comp_inst,
        .host_func = .{ .call = &Host.pair },
        .param_types = param_types,
        .result_types = result_types,
        .lower_opts = .{ .memory_idx = 0 },
    };
    defer tctx.deinit(testing.allocator);

    const dest_ptr: u32 = 128;
    const env = try ExecEnv.create(core_inst, 256, testing.allocator);
    defer env.destroy();
    try env.pushI32(@intCast(dest_ptr));
    try componentTrampoline(env, @ptrCast(&tctx));

    const mem = core_inst.getMemory(0).?;
    const r0 = std.mem.readInt(i32, mem.data[dest_ptr..][0..4], .little);
    const r1 = std.mem.readInt(i32, mem.data[dest_ptr + 4 ..][0..4], .little);
    try testing.expectEqual(@as(i32, 0xCAFE), r0);
    try testing.expectEqual(@as(i32, 0xBEEF), r1);
}

test "pushInterfaceValue/popInterfaceValue: result<_, primitive> roundtrip (#155)" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    // result<_, u32>: ok arm empty, err arm flat = [i32]; total = 2 i32s.
    const type_defs = [_]ctypes.TypeDef{
        .{ .result = .{ .ok = null, .err = .u32 } },
    };
    var component = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &type_defs,      .canons = &.{},
        .imports = &.{},          .exports = &.{},
    };
    const registry = TypeRegistry.init(&component);
    const t: ctypes.ValType = .{ .result = 0 };

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    // Push ok arm: should produce [i32 0, i32 0] (zero-filled payload slot).
    try pushInterfaceValue(env, .{ .result_val = .{ .is_ok = true, .payload = null } }, t, registry);
    const lifted_ok = try popInterfaceValue(env, t, registry, testing.allocator);
    try testing.expect(lifted_ok.result_val.is_ok);

    // Push err arm with payload u32=0xCAFEu: produces [i32 1, i32 0xCAFE].
    const err_payload: InterfaceValue = .{ .u32 = 0xCAFE };
    try pushInterfaceValue(
        env,
        .{ .result_val = .{ .is_ok = false, .payload = &err_payload } },
        t,
        registry,
    );
    // Verify the underlying core stack layout: top = payload (0xCAFE), below = disc (1).
    const payload_slot: u32 = @bitCast(try env.popI32());
    const disc_slot = try env.popI32();
    try testing.expectEqual(@as(u32, 0xCAFE), payload_slot);
    try testing.expectEqual(@as(i32, 1), disc_slot);
}
