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
const HostTrapInfo = @import("../runtime/common/exec_env.zig").HostTrapInfo;
const interp = @import("../runtime/interpreter/interp.zig");
const indexspace = @import("indexspace.zig");
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
    /// Whether this lift uses the async ABI (Binary.md `canonopt 0x06`).
    /// Async lifts return a packed status immediately; results are
    /// delivered via `task.return`. (#478 sub-PR 2.)
    is_async: bool = false,
    /// Optional resumption callback core funcidx (Binary.md `canonopt 0x07`).
    /// Only meaningful when `is_async`; the single-threaded poll-cycle
    /// dispatcher invokes it once after each yield. (#478 sub-PR 2.)
    callback_idx: ?u32 = null,

    pub fn fromOpts(opts: []const ctypes.CanonOpt) LiftOptions {
        var lo = LiftOptions{};
        for (opts) |opt| {
            switch (opt) {
                .memory => |idx| lo.memory_idx = idx,
                .realloc => |idx| lo.realloc_idx = idx,
                .post_return => |idx| lo.post_return_idx = idx,
                .string_encoding => |enc| lo.string_encoding = enc,
                .async_lift => lo.is_async = true,
                .callback => |idx| lo.callback_idx = idx,
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
/// Context for `forwardingHostFnCall` — a `HostFunc` adapter that
/// dispatches a child component's import to a parent (or peer)
/// component's lifted export. Sub-component imports declared as
/// `(with "<name>" (func K))` or `(with "<name>" (instance J))` are
/// wired during the parent's `linkImports` by constructing one of
/// these contexts and storing it in a `ImportBinding.host_func` (or as
/// the member of a synthetic `HostInstance`). See
/// `instance.zig:wireSubComponentImports` (issue #355).
///
/// The forwarded target is given as a fully-resolved `(owner, local)`
/// pair so dispatch does not depend on the target being registered as
/// a top-level export under any name. The `local` value is the
/// owner-relative `ExportedFunc.Local` produced by
/// `flattenForwardedChain` — or built directly from a `canon.lift`
/// when the parent's component-func-index resolves to `.lifted`.
pub const ForwardingHostFnCtx = struct {
    owner: *const ComponentInstance,
    local: ComponentInstance.ExportedFunc.Local,
};

/// `HostFunc.call` adapter for forwarding contexts. Ignores the
/// trampoline's `ci` (the child whose core call invoked the import)
/// and dispatches against the recorded owner so the canonical-ABI
/// lift uses the owner's type registry and core instances.
pub fn forwardingHostFnCall(
    ctx_opaque: ?*anyopaque,
    _: *ComponentInstance,
    args: []const InterfaceValue,
    out_results: []InterfaceValue,
    allocator: Allocator,
) anyerror!void {
    const ctx: *const ForwardingHostFnCtx = @ptrCast(@alignCast(ctx_opaque orelse return error.FunctionNotFound));
    return callComponentFuncByLocal(ctx.owner, ctx.local, args, out_results, allocator);
}

pub fn callComponentFunc(
    comp_inst: *const ComponentInstance,
    func_name: []const u8,
    args: []const InterfaceValue,
    out_results: []InterfaceValue,
    allocator: Allocator,
) ExecutionError!void {
    const flat = flattenForwardedChain(comp_inst, func_name) orelse return error.FunctionNotFound;
    return callComponentFuncByLocal(flat.owner, flat.local, args, out_results, allocator);
}

/// Walk an `ExportedFunc.forwarded` chain starting from `(comp_inst,
/// func_name)` to the bottoming `(owner, .local)` pair. Returns null if
/// the name does not resolve, or if the chain exceeds 16 hops (which is
/// treated as a resolver bug since flattening at registration time is
/// expected to keep chains shallow).
pub const FlattenedExport = struct {
    owner: *const ComponentInstance,
    local: ComponentInstance.ExportedFunc.Local,
};

pub fn flattenForwardedChain(
    comp_inst: *const ComponentInstance,
    func_name: []const u8,
) ?FlattenedExport {
    const root = comp_inst.getExport(func_name) orelse return null;
    var owner_inst = comp_inst;
    var cursor = root;
    var hops: u8 = 0;
    while (true) : (hops += 1) {
        if (hops > 16) return null;
        switch (cursor) {
            .local => |l| return .{ .owner = owner_inst, .local = l },
            .forwarded => |f| {
                owner_inst = f.owner;
                cursor = f.owner.getExport(f.owner_export_name) orelse return null;
            },
        }
    }
}

/// Invoke a component-level lifted export, given the owning instance and
/// its already-resolved owner-relative indices. This is the common path
/// shared between name-keyed dispatch (`callComponentFunc`) and the
/// cross-component forwarding `HostFunc` adapter (#355): the latter holds
/// an `ExportedFunc.Local` directly because the parent's `.lifted`
/// component-funcs are not necessarily registered as top-level exports
/// under any name.
pub fn callComponentFuncByLocal(
    owner_inst: *const ComponentInstance,
    exported: ComponentInstance.ExportedFunc.Local,
    args: []const InterfaceValue,
    out_results: []InterfaceValue,
    allocator: Allocator,
) ExecutionError!void {
    // Get the core module instance
    if (exported.core_instance_idx >= owner_inst.core_instances.len)
        return error.CoreInstanceNotAvailable;
    const core_entry = owner_inst.core_instances[exported.core_instance_idx];
    const module_inst = core_entry.module_inst orelse return error.CoreInstanceNotAvailable;

    // Parse canonical options
    const lift_opts = LiftOptions.fromOpts(exported.opts);

    // Get the type registry — must come from the owner so component-
    // level type indices resolve in the owner's type-indexspace.
    const registry = TypeRegistry.init(owner_inst.component);

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
    interp.executeFunction(env, exported.core_func_idx) catch {
        if (env.host_trap) |ht| {
            // Suppress the diagnostic when this trap is actually a
            // `wasi:cli/exit.{exit, exit-with-code}` unwind — that's
            // normal control flow (the host code is already stashed on
            // `WasiCliAdapter.exit_code`), not a real error (issue
            // #436 / #448).
            const is_wasi_exit = std.mem.eql(u8, ht.err_name, "WasiExit");
            if (!is_wasi_exit) {
                std.debug.print("[component trap] core_func_idx={d}", .{ht.core_func_idx});
                if (ht.component_func_idx != std.math.maxInt(u32))
                    std.debug.print(" component_func_idx={d}", .{ht.component_func_idx});
                if (ht.import_module_name.len > 0 or ht.import_field_name.len > 0)
                    std.debug.print(
                        " import='{s}.{s}'",
                        .{ ht.import_module_name, ht.import_field_name },
                    );
                std.debug.print(
                    " stage={s} error={s}\n",
                    .{ @tagName(ht.stage), ht.err_name },
                );
            }
        }
        return error.TrapInCoreFunction;
    };

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

/// Async-lifted variant of `callComponentFuncByLocal`. Lifts args and
/// drives the core wasm body the same way, but the callee delivers its
/// results via `canon task.return` (#478 sub-PR 2). On return from the
/// core function, `out_status` is set to the i32 the core fn left on
/// the stack — for an async lift with no callback the spec says this
/// is always `0` (the "task returned synchronously" sentinel); with a
/// callback set it's the packed status that selects the callback's
/// follow-up action. The actual results land in
/// `task_manager.tasks[handle].return_values` via `dispatchCanonBuiltin`.
pub fn callComponentFuncByLocalAsyncLifted(
    owner_inst: *const ComponentInstance,
    exported: ComponentInstance.ExportedFunc.Local,
    args: []const InterfaceValue,
    out_status: *u32,
    allocator: Allocator,
) ExecutionError!void {
    out_status.* = 0;

    if (exported.core_instance_idx >= owner_inst.core_instances.len)
        return error.CoreInstanceNotAvailable;
    const core_entry = owner_inst.core_instances[exported.core_instance_idx];
    const module_inst = core_entry.module_inst orelse return error.CoreInstanceNotAvailable;

    const lift_opts = LiftOptions.fromOpts(exported.opts);
    if (!lift_opts.is_async) return error.InvalidFuncType;

    const registry = TypeRegistry.init(owner_inst.component);

    const func_type = blk: {
        const td = registry.get(exported.func_type_idx) orelse return error.InvalidFuncType;
        switch (td) {
            .func => |ft| break :blk ft,
            else => return error.InvalidFuncType,
        }
    };

    const param_types = getParamValTypes(func_type, allocator) catch return error.OutOfMemory;
    defer allocator.free(param_types);

    const flat_param_count = countFlatTypes(registry, param_types);

    const env = ExecEnv.create(module_inst, 4096, allocator) catch return error.OutOfMemory;
    defer env.destroy();

    const memory: ?[]u8 = if (lift_opts.memory_idx) |mem_idx|
        if (module_inst.getMemory(mem_idx)) |mem| mem.data else null
    else
        null;

    // Lower args — same logic as the sync path.
    if (flat_param_count <= MAX_FLAT_PARAMS) {
        for (args, param_types) |arg, pt| {
            pushInterfaceValue(env, arg, pt, registry) catch return error.LowerError;
        }
    } else {
        const mem = memory orelse return error.MemoryNotAvailable;
        const realloc_idx = lift_opts.realloc_idx orelse return error.ReallocNotAvailable;
        const tuple_size = computeTupleSize(registry, param_types);
        const tuple_align = computeTupleAlign(registry, param_types);
        const ptr = try callRealloc(env, realloc_idx, 0, 0, tuple_align, tuple_size);

        var offset: u32 = 0;
        for (args, param_types) |arg, pt| {
            const al = typeAlign(registry, pt);
            offset = abi.alignUp(offset, al);
            storeInterfaceValue(mem, offset, arg, pt, registry);
            offset += typeSize(registry, pt);
        }
        env.pushI32(@bitCast(ptr)) catch return error.StackOverflow;
    }

    // Drive the core body. It is the callee's responsibility to invoke
    // `canon task.return` before returning — which deposits the lifted
    // results onto the task via `dispatchCanonBuiltin`.
    interp.executeFunction(env, exported.core_func_idx) catch {
        return error.TrapInCoreFunction;
    };

    // Spec: with `callback` set, the core fn leaves a packed status i32
    // on the stack; otherwise (stackful async) it returns no value. We
    // probe optimistically: if the core fn returned an i32, peel it
    // off; otherwise leave status at 0 (the default).
    if (lift_opts.callback_idx != null) {
        out_status.* = @bitCast(env.popI32() catch 0);
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
        .own, .borrow, .future, .stream, .error_context => try env.pushI32(@bitCast(val.handle)),
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
        .record, .variant, .tuple, .flags, .enum_, .option => {
            return error.CompoundNeedsRegistry;
        },
        // Resolve `.type_idx` through the registry and re-dispatch on
        // the reified ValType. The registry may have been extended with
        // an instance-type body's local types (#156 H4a/H4b), so the
        // index can land on a `.val v` wrapper, a resource definition,
        // or any compound TypeDef.
        .type_idx => |idx| {
            const td = registry.get(idx) orelse return error.CompoundNeedsRegistry;
            const reified: ctypes.ValType = switch (td) {
                .val => |inner| inner,
                .list => .{ .list = idx },
                .record => .{ .record = idx },
                .tuple => .{ .tuple = idx },
                .variant => .{ .variant = idx },
                .flags => .{ .flags = idx },
                .enum_ => .{ .enum_ = idx },
                .option => .{ .option = idx },
                .result => .{ .result = idx },
                .resource => .{ .own = idx },
                else => return error.CompoundNeedsRegistry,
            };
            try pushInterfaceValue(env, val, reified, registry);
        },
    }
}

/// Resolve a `.type_idx` ValType chain to a more concrete form so that
/// flat lift/lower in `popInterfaceValue` / `liftFlat` can handle the
/// common `result<own<R>, …>` / `option<own<R>>` shapes wabt emits
/// (where R lives under `.type_idx -> .resource`).
fn resolveArmType(t: ctypes.ValType, registry: TypeRegistry) ctypes.ValType {
    var resolved = t;
    while (resolved == .type_idx) {
        const td = registry.get(resolved.type_idx) orelse return resolved;
        resolved = switch (td) {
            .val => |v| v,
            .resource => return .{ .own = resolved.type_idx },
            else => return resolved,
        };
    }
    return resolved;
}

fn popInterfaceValue(env: *ExecEnv, t: ctypes.ValType, registry: TypeRegistry, allocator: Allocator) !InterfaceValue {
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
        .own, .borrow, .future, .stream, .error_context => .{ .handle = @bitCast(try env.popI32()) },
        .string => .{ .string = .{
            .len = @bitCast(try env.popI32()),
            .ptr = @bitCast(try env.popI32()),
        } },
        .list => .{ .list = .{
            .len = @bitCast(try env.popI32()),
            .ptr = @bitCast(try env.popI32()),
        } },
        // `result<T, E>`: pop all payload slots into a scratch buffer
        // (we don't know the active arm's typing until we've popped the
        // discriminant), then pop the disc, then re-lift the active arm
        // from the buffered slots via `liftFlat`. For simple arm types
        // (own/borrow handles, primitives, string, list), `liftFlat`
        // succeeds; complex arms (variant, record, etc. — e.g. the
        // canonical `error-code` variant) currently leave `payload =
        // null`, which matches the pre-existing behaviour and is fine
        // for host imports (like `httpOutgoingBodyFinish`) that don't
        // inspect the err arm.
        .result => |idx| blk: {
            const td = registry.get(idx) orelse return error.CompoundNeedsRegistry;
            const rt = switch (td) {
                .result => |r| r,
                else => return error.CompoundNeedsRegistry,
            };
            const total_payload_slots = abi.flattenCount(registry, t) - 1;
            var slot_buf: [16]u32 = undefined;
            if (total_payload_slots > slot_buf.len) return error.CompoundNeedsRegistry;
            // Pop in stack order: top of stack is the last-pushed slot,
            // so it lands in slot_buf[total-1] first.
            var i: u32 = total_payload_slots;
            while (i > 0) {
                i -= 1;
                slot_buf[i] = @bitCast(try env.popI32());
            }
            const disc = try env.popI32();
            const is_ok = disc == 0;
            const arm_type: ?ctypes.ValType = if (is_ok) rt.ok else rt.err;
            var payload: ?*InterfaceValue = null;
            if (arm_type) |at| {
                // Resolve through `type_idx` hops so a resource handle
                // referenced as `.type_idx -> .resource` is treated as a
                // direct `own<R>` for the purpose of lifting.
                const resolved = resolveArmType(at, registry);
                const arm_slots = abi.flattenCount(registry, resolved);
                if (abi.liftFlat(slot_buf[0..arm_slots], resolved)) |lifted| {
                    const p = try allocator.create(InterfaceValue);
                    p.* = lifted;
                    payload = p;
                } else |_| {
                    // Compound arm type — leave payload null. Host
                    // imports that need the typed payload should ship
                    // its data through a different surface (e.g. raw
                    // handle args).
                }
            }
            break :blk .{ .result_val = .{ .is_ok = is_ok, .payload = payload } };
        },
        // `option<T>`: symmetric to `.result` above. Pop payload slots,
        // pop disc; if `is_some`, lift the buffered slots via `liftFlat`
        // for simple inner types.
        .option => |idx| blk: {
            const td = registry.get(idx) orelse return error.CompoundNeedsRegistry;
            const inner_type: ctypes.ValType = switch (td) {
                .option => |o| o.inner,
                else => return error.CompoundNeedsRegistry,
            };
            const total_payload_slots = abi.flattenCount(registry, t) - 1;
            var slot_buf: [16]u32 = undefined;
            if (total_payload_slots > slot_buf.len) return error.CompoundNeedsRegistry;
            var i: u32 = total_payload_slots;
            while (i > 0) {
                i -= 1;
                slot_buf[i] = @bitCast(try env.popI32());
            }
            const disc = try env.popI32();
            const is_some = disc != 0;
            var payload: ?*InterfaceValue = null;
            if (is_some) {
                const resolved = resolveArmType(inner_type, registry);
                const inner_slots = abi.flattenCount(registry, resolved);
                if (abi.liftFlat(slot_buf[0..inner_slots], resolved)) |lifted| {
                    const p = try allocator.create(InterfaceValue);
                    p.* = lifted;
                    payload = p;
                } else |_| {
                    // Compound inner — leave payload null. Same caveat
                    // as the `.result` branch above.
                }
            }
            break :blk .{ .option_val = .{ .is_some = is_some, .payload = payload } };
        },
        .record, .variant, .tuple, .flags, .enum_ => error.CompoundNeedsRegistry,
        // Mirror `pushInterfaceValue`: resolve `.type_idx` and re-pop on
        // the reified ValType.
        .type_idx => |idx| blk: {
            const td = registry.get(idx) orelse return error.CompoundNeedsRegistry;
            const reified: ctypes.ValType = switch (td) {
                .val => |inner| inner,
                .list => .{ .list = idx },
                .record => .{ .record = idx },
                .tuple => .{ .tuple = idx },
                .variant => .{ .variant = idx },
                .flags => .{ .flags = idx },
                .enum_ => .{ .enum_ = idx },
                .option => .{ .option = idx },
                .result => .{ .result = idx },
                .resource => .{ .own = idx },
                else => return error.CompoundNeedsRegistry,
            };
            break :blk try popInterfaceValue(env, reified, registry, allocator);
        },
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
/// references resource.new/drop/rep, task.yield, or context.{get,set}
/// instead of lift/lower.
///
/// `task_manager` is the async runtime state for this dispatch (nullable
/// because synchronous canon-lift paths don't construct one). When null,
/// `task.yield` is a no-op resume and `context.{get,set}` operate on
/// `comp_inst.implicit_task_context`, matching Wasmtime's per-instance
/// fallback for sync calls. (#478 sub-PR 1.)
pub fn dispatchCanonBuiltin(
    comp_inst: *ComponentInstance,
    canon: ctypes.Canon,
    env: *ExecEnv,
    task_manager: ?*async_mod.TaskManager,
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
        .task_yield => |info| {
            // `canon thread.yield cancel?` — Binary.md tag 0x0c. Pushes
            // an i32 discriminant: 0 on normal resume, 1 if the task was
            // cancelled while parked (cancellable-only).
            const outcome: u32 = if (task_manager) |tm| blk: {
                const handle = tm.current_task orelse break :blk 0;
                break :blk @intFromEnum(async_canon.taskYield(tm, handle, info.cancellable, allocator));
            } else 0;
            env.pushI32(@bitCast(outcome)) catch return error.StackOverflow;
        },
        .context_get => |info| {
            // Sub-PR 1 only admits i32; the loader rejects others. Defend
            // here too so a hand-constructed Canon doesn't bypass the
            // limit silently.
            if (info.val_type != .i32) return error.FunctionNotFound;
            const value: u32 = blk: {
                if (task_manager) |tm| {
                    if (tm.current_task) |handle| {
                        if (tm.getContextSlot(handle, info.slot)) |v| break :blk v;
                        // Slot out of range on a known task → trap.
                        return error.StackUnderflow;
                    }
                }
                if (info.slot >= async_mod.N_CONTEXT_SLOTS) return error.StackUnderflow;
                break :blk comp_inst.implicit_task_context[info.slot];
            };
            env.pushI32(@bitCast(value)) catch return error.StackOverflow;
        },
        .context_set => |info| {
            if (info.val_type != .i32) return error.FunctionNotFound;
            const value: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            if (task_manager) |tm| {
                if (tm.current_task) |handle| {
                    if (!tm.setContextSlot(handle, info.slot, value)) return error.StackUnderflow;
                    return;
                }
            }
            if (info.slot >= async_mod.N_CONTEXT_SLOTS) return error.StackUnderflow;
            comp_inst.implicit_task_context[info.slot] = value;
        },
        .task_return => |info| {
            // `canon task.return rs:<resultlist> opts:<opts>` — Binary.md
            // tag 0x09. Pops the lifted-callee's results off the env stack
            // (flat i32 representation per the Canonical ABI), stores them
            // on the current task, and notifies the parent waitable.
            // Caller without a task manager is malformed (a sync lift
            // never emits task.return).
            const tm = task_manager orelse return error.FunctionNotFound;
            const handle = tm.current_task orelse return error.FunctionNotFound;
            const flat_count: usize = switch (info.results) {
                .none => 0,
                .unnamed => 1,
                .named => |named| named.len,
            };
            const flat = allocator.alloc(u32, flat_count) catch return error.OutOfMemory;
            // The Canonical ABI pushes results left-to-right, so the last
            // result is on top — pop in reverse.
            var i: usize = flat_count;
            while (i > 0) {
                i -= 1;
                flat[i] = @bitCast(env.popI32() catch {
                    allocator.free(flat);
                    return error.StackUnderflow;
                });
            }
            async_canon.asyncReturn(tm, handle, flat);
        },
        .async_canon => |op| try dispatchAsyncCanon(comp_inst, op, env, task_manager, allocator),
        .lift, .lower => {}, // Handled by callComponentFunc
    }
}

/// Dispatch the WASIp3 async-canon surface (#478 sub-PR 3): subtask /
/// future / stream / error-context / waitable-set / waitable.join.
///
/// **Scope**: this is the minimum dispatch needed for the conformance
/// suite to stop failing at load. Every `.new`-flavoured op allocates
/// a per-instance handle; every `.drop` releases it. The read / write /
/// cancel-* ops trap with `error.FunctionNotFound` — they require
/// real fiber-based scheduling integration that lands as a follow-up.
fn dispatchAsyncCanon(
    comp_inst: *ComponentInstance,
    op: ctypes.AsyncCanonOp,
    env: *ExecEnv,
    task_manager: ?*async_mod.TaskManager,
    allocator: Allocator,
) ExecutionError!void {
    switch (op) {
        .subtask_drop => {
            // Pop the subtask handle. We simply discard — the task's
            // memory belongs to the TaskManager whose lifetime exceeds
            // any single subtask.drop call.
            _ = env.popI32() catch return error.StackUnderflow;
        },
        .subtask_cancel => |info| {
            _ = info;
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            if (task_manager) |tm| tm.cancelTask(handle);
            // `async?` info is ignored: cancellation is idempotent and
            // synchronous in the single-threaded runtime.
        },

        // ── Stream handles ──────────────────────────────────────────────
        .stream_new => |info| {
            _ = info;
            const handle = comp_inst.allocAsyncHandle();
            comp_inst.streams.put(comp_inst.allocator, handle, .{}) catch return error.OutOfMemory;
            // Spec packs read+write handles into one i64; in our
            // single-handle prototype we publish the same idx for both
            // ends to keep the wire format compatible with i64 pops.
            const packed_handles: u64 = (@as(u64, handle) << 32) | @as(u64, handle);
            env.pushI64(@bitCast(packed_handles)) catch return error.StackOverflow;
        },
        .stream_drop_readable => |info| {
            _ = info;
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            if (comp_inst.streams.fetchRemove(handle)) |kv| {
                var s = kv.value;
                s.deinit(comp_inst.allocator);
            }
        },
        .stream_drop_writable => |info| {
            _ = info;
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            if (comp_inst.streams.fetchRemove(handle)) |kv| {
                var s = kv.value;
                s.deinit(comp_inst.allocator);
            }
        },
        .stream_read, .stream_write, .stream_cancel_read, .stream_cancel_write => return error.FunctionNotFound,

        // ── Future handles ──────────────────────────────────────────────
        .future_new => |info| {
            const handle = comp_inst.allocAsyncHandle();
            comp_inst.futures.put(comp_inst.allocator, handle, .{
                .elem_type_idx = info.type_idx,
            }) catch return error.OutOfMemory;
            const packed_handles: u64 = (@as(u64, handle) << 32) | @as(u64, handle);
            env.pushI64(@bitCast(packed_handles)) catch return error.StackOverflow;
        },
        .future_read => |info| {
            // Stack: (handle, ptr) → status. The destination pointer
            // is on top; `handle` is below it.
            const guest_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);

            const fut = comp_inst.futures.getPtr(handle) orelse {
                env.pushI32(@bitCast(async_canon.packStatus(.cancelled, 0))) catch
                    return error.StackOverflow;
                return;
            };

            // Writer dropped without ever delivering a value → cancelled.
            if (fut.write_closed and fut.payload == null) {
                env.pushI32(@bitCast(async_canon.packStatus(.cancelled, 0))) catch
                    return error.StackOverflow;
                return;
            }

            // Writer already buffered — deliver and wake any read waitable.
            if (fut.payload) |buf| {
                const dst = comp_inst.writableGuestBytes(guest_ptr, @intCast(buf.len)) orelse
                    return error.MemoryNotAvailable;
                @memcpy(dst, buf);
                comp_inst.allocator.free(buf);
                fut.payload = null;
                fut.state = .ready;
                if (fut.waitable_set) |ws| if (fut.read_waitable_idx) |idx|
                    ws.setReady(idx, allocator);
                env.pushI32(@bitCast(async_canon.packStatus(.returned, 1))) catch
                    return error.StackOverflow;
                return;
            }

            // No writer yet — park. `info.type_idx` is captured into the
            // table entry at `future.new` time; we only need the guest
            // ptr here.
            _ = info;
            fut.pending_read = .{ .guest_ptr = guest_ptr };
            env.pushI32(@bitCast(async_canon.packStatus(.starting, 0))) catch
                return error.StackOverflow;
        },
        .future_write => |info| {
            // Stack: (handle, ptr) → status.
            const guest_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);

            const fut = comp_inst.futures.getPtr(handle) orelse {
                env.pushI32(@bitCast(async_canon.packStatus(.cancelled, 0))) catch
                    return error.StackOverflow;
                return;
            };

            // Reader already dropped → writer is cancelled.
            if (fut.read_closed) {
                env.pushI32(@bitCast(async_canon.packStatus(.cancelled, 0))) catch
                    return error.StackOverflow;
                return;
            }

            // Already buffered (double-write); reject with cancelled. Spec
            // forbids two writes on a one-shot future.
            if (fut.payload != null) {
                env.pushI32(@bitCast(async_canon.packStatus(.cancelled, 0))) catch
                    return error.StackOverflow;
                return;
            }

            const registry = TypeRegistry.init(comp_inst.component);
            const elem_size = abi.sizeOfType(registry, ctypes.ValType{ .type_idx = info.type_idx });
            if (elem_size == 0) return error.LowerError;

            const src = comp_inst.readGuestBytes(guest_ptr, elem_size) orelse
                return error.MemoryNotAvailable;

            // Reader parked first: copy straight into its destination,
            // skip allocating a heap buffer.
            if (fut.pending_read) |pr| {
                const dst = comp_inst.writableGuestBytes(pr.guest_ptr, elem_size) orelse
                    return error.MemoryNotAvailable;
                @memcpy(dst, src);
                fut.pending_read = null;
                fut.state = .ready;
                if (fut.waitable_set) |ws| if (fut.read_waitable_idx) |idx|
                    ws.setReady(idx, allocator);
                env.pushI32(@bitCast(async_canon.packStatus(.returned, 1))) catch
                    return error.StackOverflow;
                return;
            }

            // No reader yet — buffer the payload on the heap. Reader's
            // future arrival will memcpy out of `payload` and free it.
            const heap_buf = comp_inst.allocator.alloc(u8, elem_size) catch
                return error.OutOfMemory;
            @memcpy(heap_buf, src);
            fut.payload = heap_buf;
            fut.state = .ready;
            if (fut.waitable_set) |ws| if (fut.read_waitable_idx) |idx|
                ws.setReady(idx, allocator);
            env.pushI32(@bitCast(async_canon.packStatus(.returned, 1))) catch
                return error.StackOverflow;
        },
        .future_cancel_read => |info| {
            _ = info;
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const fut = comp_inst.futures.getPtr(handle) orelse {
                env.pushI32(@bitCast(async_canon.packStatus(.cancelled, 0))) catch
                    return error.StackOverflow;
                return;
            };
            // If a value was already delivered before cancel arrived,
            // surface RETURNED so the caller still observes the transfer.
            if (fut.state == .ready and fut.payload == null) {
                env.pushI32(@bitCast(async_canon.packStatus(.returned, 1))) catch
                    return error.StackOverflow;
                return;
            }
            fut.pending_read = null;
            env.pushI32(@bitCast(async_canon.packStatus(.cancelled, 0))) catch
                return error.StackOverflow;
        },
        .future_cancel_write => |info| {
            _ = info;
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const fut = comp_inst.futures.getPtr(handle) orelse {
                env.pushI32(@bitCast(async_canon.packStatus(.cancelled, 0))) catch
                    return error.StackOverflow;
                return;
            };
            // If the reader already consumed the buffered payload, the
            // write completed before cancel; report RETURNED.
            if (fut.state == .ready and fut.payload == null and fut.pending_read == null) {
                env.pushI32(@bitCast(async_canon.packStatus(.returned, 1))) catch
                    return error.StackOverflow;
                return;
            }
            // Otherwise reclaim any unconsumed buffered payload so the
            // future returns to the empty state.
            if (fut.payload) |buf| {
                comp_inst.allocator.free(buf);
                fut.payload = null;
            }
            env.pushI32(@bitCast(async_canon.packStatus(.cancelled, 0))) catch
                return error.StackOverflow;
        },
        .future_drop_readable => |info| {
            _ = info;
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            if (comp_inst.futures.getPtr(handle)) |fut| {
                fut.read_closed = true;
                // Wake a parked writer so it can observe CANCELLED.
                if (fut.waitable_set) |ws| if (fut.write_waitable_idx) |idx|
                    ws.setReady(idx, allocator);
                if (fut.read_closed and fut.write_closed) {
                    fut.deinit(comp_inst.allocator);
                    _ = comp_inst.futures.remove(handle);
                }
            }
        },
        .future_drop_writable => |info| {
            _ = info;
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            if (comp_inst.futures.getPtr(handle)) |fut| {
                fut.write_closed = true;
                // Wake a parked reader so it can observe CANCELLED.
                if (fut.waitable_set) |ws| if (fut.read_waitable_idx) |idx|
                    ws.setReady(idx, allocator);
                if (fut.read_closed and fut.write_closed) {
                    fut.deinit(comp_inst.allocator);
                    _ = comp_inst.futures.remove(handle);
                }
            }
        },

        // ── error-context ───────────────────────────────────────────────
        .error_context_new => |info| {
            _ = info;
            // Per spec: pops (ptr, len) for the debug-message string and
            // returns a new error-context handle. We don't yet eagerly
            // copy the message from guest memory — that requires
            // memory/realloc options threading. Store an empty string;
            // `debug-message` returns it as-is.
            _ = env.popI32() catch return error.StackUnderflow; // len
            _ = env.popI32() catch return error.StackUnderflow; // ptr
            const handle = comp_inst.allocAsyncHandle();
            const empty = allocator.alloc(u8, 0) catch return error.OutOfMemory;
            comp_inst.error_contexts.put(comp_inst.allocator, handle, empty) catch {
                allocator.free(empty);
                return error.OutOfMemory;
            };
            env.pushI32(@bitCast(handle)) catch return error.StackOverflow;
        },
        .error_context_debug_message => |info| {
            _ = info;
            // Pops error-context handle + (ptr, length) of the
            // caller-allocated buffer; writes the debug message into
            // guest memory. We don't yet bridge into guest memory, so
            // pop and no-op for now.
            _ = env.popI32() catch return error.StackUnderflow;
            _ = env.popI32() catch return error.StackUnderflow;
            _ = env.popI32() catch return error.StackUnderflow;
        },
        .error_context_drop => {
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            if (comp_inst.error_contexts.fetchRemove(handle)) |kv| {
                comp_inst.allocator.free(kv.value);
            }
        },

        // ── Waitable-set ────────────────────────────────────────────────
        .waitable_set_new => {
            const handle = comp_inst.allocAsyncHandle();
            comp_inst.waitable_sets.put(comp_inst.allocator, handle, .{}) catch return error.OutOfMemory;
            env.pushI32(@bitCast(handle)) catch return error.StackOverflow;
        },
        .waitable_set_drop => {
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            if (comp_inst.waitable_sets.fetchRemove(handle)) |kv| {
                var ws = kv.value;
                ws.deinit(comp_inst.allocator);
            }
        },
        .waitable_set_wait, .waitable_set_poll => {
            // wait/poll expect a (ws_handle, out_ptr) and write a 2-tuple
            // (event-kind, payload) into guest memory. Without
            // guest-memory bridging we can't honour that contract — pop
            // arguments and push a zero status to keep the core stack
            // balanced for the conformance "load + don't crash" target.
            _ = env.popI32() catch return error.StackUnderflow; // ws handle
            _ = env.popI32() catch return error.StackUnderflow; // out ptr
            env.pushI32(0) catch return error.StackOverflow;
        },

        .waitable_join => {
            // Pops (waitable_handle, ws_handle). We simply drop both —
            // the join becomes a no-op until real WaitableItem-typed
            // registration lands alongside the read/write canon ops.
            _ = env.popI32() catch return error.StackUnderflow;
            _ = env.popI32() catch return error.StackUnderflow;
        },
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

    // Look up the exported function to determine result count.
    // Resolve forwarded entries against the owning component so the
    // type-registry lookup hits the correct types[] (#355).
    const flat = flattenForwardedChain(comp_inst, func_name) orelse {
        task_manager.cancelTask(handle);
        return error.FunctionNotFound;
    };
    const owner_for_type = flat.owner;
    const exported_local = flat.local;

    // Inspect the lift options to choose between the async-lifted ABI
    // (callee invokes `task.return`) and the legacy sync-wrapped-in-task
    // path (host lifts the results and stashes them on the task).
    const lift_opts = LiftOptions.fromOpts(exported_local.opts);

    // Make the just-created subtask discoverable to context.{get,set},
    // task.yield, and task.return invoked from inside the core body.
    // Restored on return regardless of success/failure. (#478 sub-PR 1/2.)
    const saved_current_task = task_manager.current_task;
    task_manager.current_task = handle;
    defer task_manager.current_task = saved_current_task;

    if (lift_opts.is_async) {
        // Async-lifted ABI: drive the core fn and let `task.return`
        // populate task.return_values on its own.
        var status: u32 = 0;
        _ = &status;
        callComponentFuncByLocalAsyncLifted(owner_for_type, exported_local, args, &status, allocator) catch |e| {
            task_manager.cancelTask(handle);
            return e;
        };
        // If a callback is configured, sub-PR 2's poll-cycle stub would
        // invoke it once after each yield. Real future/stream-driven
        // polling lands in sub-PR 3; for now we surface the status by
        // ignoring it (the caller can re-inspect it via the task).
        return handle;
    }

    // Legacy sync-lift path: lift results, populate the task manually.
    const result_count: usize = blk: {
        const reg = TypeRegistry.init(owner_for_type.component);
        if (reg.get(exported_local.func_type_idx)) |td| {
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
    try std.testing.expectEqual(false, lo.is_async);
    try std.testing.expectEqual(@as(?u32, null), lo.callback_idx);
}

test "LiftOptions: async + callback (#478 sub-PR 2)" {
    const opts = [_]ctypes.CanonOpt{
        .{ .memory = 0 },
        .async_lift,
        .{ .callback = 7 },
    };
    const lo = LiftOptions.fromOpts(&opts);
    try std.testing.expectEqual(@as(?u32, 0), lo.memory_idx);
    try std.testing.expectEqual(true, lo.is_async);
    try std.testing.expectEqual(@as(?u32, 7), lo.callback_idx);
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

// ── dispatchCanonBuiltin: async ABI built-ins (#478 sub-PR 1) ────────────────

test "dispatchCanonBuiltin: task.yield pushes resumed=0 with no task manager" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    var comp = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},         .imports = &.{},
        .exports = &.{},
    };
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();

    try dispatchCanonBuiltin(
        inst,
        .{ .task_yield = .{ .cancellable = false } },
        env,
        null,
        testing.allocator,
    );
    try testing.expectEqual(@as(i32, 0), try env.popI32());
}

test "dispatchCanonBuiltin: task.yield observes cancellation via TaskManager" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    var comp = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},         .imports = &.{},
        .exports = &.{},
    };
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();

    var tm = async_mod.TaskManager{};
    defer tm.deinit(testing.allocator);

    const lift = try async_canon.asyncLift(.{
        .task_manager = &tm,
        .allocator = testing.allocator,
    });
    tm.current_task = lift.subtask_handle;
    async_canon.asyncCancel(&tm, lift.subtask_handle);

    // Non-cancellable yield: opaque, always reports resumed=0.
    try dispatchCanonBuiltin(
        inst,
        .{ .task_yield = .{ .cancellable = false } },
        env,
        &tm,
        testing.allocator,
    );
    try testing.expectEqual(@as(i32, 0), try env.popI32());

    // Cancellable yield: surfaces the pending cancellation as 1.
    try dispatchCanonBuiltin(
        inst,
        .{ .task_yield = .{ .cancellable = true } },
        env,
        &tm,
        testing.allocator,
    );
    try testing.expectEqual(@as(i32, 1), try env.popI32());
}

test "dispatchCanonBuiltin: context set+get round-trip on implicit task" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    var comp = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},         .imports = &.{},
        .exports = &.{},
    };
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();

    // Push 0x1234 then `context.set i32 0` → stored on implicit task.
    try env.pushI32(@bitCast(@as(u32, 0x1234)));
    try dispatchCanonBuiltin(
        inst,
        .{ .context_set = .{ .val_type = .i32, .slot = 0 } },
        env,
        null,
        testing.allocator,
    );
    try testing.expectEqual(@as(u32, 0x1234), inst.implicit_task_context[0]);

    // `context.get i32 0` → pushes the value back onto the stack.
    try dispatchCanonBuiltin(
        inst,
        .{ .context_get = .{ .val_type = .i32, .slot = 0 } },
        env,
        null,
        testing.allocator,
    );
    try testing.expectEqual(@as(i32, 0x1234), try env.popI32());
}

test "dispatchCanonBuiltin: context set+yield+get round-trip on async task" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    var comp = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},         .imports = &.{},
        .exports = &.{},
    };
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();

    var tm = async_mod.TaskManager{};
    defer tm.deinit(testing.allocator);
    const lift = try async_canon.asyncLift(.{
        .task_manager = &tm,
        .allocator = testing.allocator,
    });
    tm.current_task = lift.subtask_handle;

    // `context.set i32 1`(value=0xDEAD_BEEF) → task.yield → `context.get i32 1`
    try env.pushI32(@bitCast(@as(u32, 0xDEAD_BEEF)));
    try dispatchCanonBuiltin(
        inst,
        .{ .context_set = .{ .val_type = .i32, .slot = 1 } },
        env,
        &tm,
        testing.allocator,
    );

    try dispatchCanonBuiltin(
        inst,
        .{ .task_yield = .{ .cancellable = false } },
        env,
        &tm,
        testing.allocator,
    );
    // task.yield pushes its outcome — pop and discard.
    try testing.expectEqual(@as(i32, 0), try env.popI32());

    try dispatchCanonBuiltin(
        inst,
        .{ .context_get = .{ .val_type = .i32, .slot = 1 } },
        env,
        &tm,
        testing.allocator,
    );
    try testing.expectEqual(@as(u32, 0xDEAD_BEEF), @as(u32, @bitCast(try env.popI32())));

    // Slot stored on the task, NOT on the instance's implicit context.
    try testing.expectEqual(@as(u32, 0), inst.implicit_task_context[1]);
    try testing.expectEqual(@as(?u32, 0xDEAD_BEEF), tm.getContextSlot(lift.subtask_handle, 1));
}

test "dispatchCanonBuiltin: context.get out-of-range slot traps" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    var comp = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},         .imports = &.{},
        .exports = &.{},
    };
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();

    const oob_slot: u32 = async_mod.N_CONTEXT_SLOTS;
    const result = dispatchCanonBuiltin(
        inst,
        .{ .context_get = .{ .val_type = .i32, .slot = oob_slot } },
        env,
        null,
        testing.allocator,
    );
    try testing.expectError(error.StackUnderflow, result);
}

test "dispatchCanonBuiltin: task.return delivers results to the current task (#478 sub-PR 2)" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    var comp = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},         .imports = &.{},
        .exports = &.{},
    };
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();

    var tm = async_mod.TaskManager{};
    defer tm.deinit(testing.allocator);
    const lift = try async_canon.asyncLift(.{
        .task_manager = &tm,
        .allocator = testing.allocator,
    });
    tm.current_task = lift.subtask_handle;

    // Push the single i32 result on the env stack, then dispatch task.return.
    try env.pushI32(0x1234_5678);
    try dispatchCanonBuiltin(
        inst,
        .{ .task_return = .{
            .results = .{ .unnamed = .s32 },
            .opts = &.{},
        } },
        env,
        &tm,
        testing.allocator,
    );

    // Task should now be in .returned with the value visible to pollers.
    try testing.expectEqual(async_mod.TaskState.returned, tm.getState(lift.subtask_handle).?);
    const ret = async_canon.asyncPollResult(&tm, lift.subtask_handle).?;
    defer testing.allocator.free(ret);
    try testing.expectEqual(@as(usize, 1), ret.len);
    try testing.expectEqual(@as(u32, 0x1234_5678), ret[0]);
}

test "dispatchCanonBuiltin: task.return without a task manager is malformed" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    var comp = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},         .imports = &.{},
        .exports = &.{},
    };
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();

    const result = dispatchCanonBuiltin(
        inst,
        .{ .task_return = .{ .results = .none, .opts = &.{} } },
        env,
        null,
        testing.allocator,
    );
    try testing.expectError(error.FunctionNotFound, result);
}

// ── Sub-PR 3: async_canon smoke tests ────────────────────────────────────────

test "dispatchCanonBuiltin: waitable-set.new allocates a fresh handle; drop frees it" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    var comp = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},         .imports = &.{},
        .exports = &.{},
    };
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .waitable_set_new },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @bitCast(try env.popI32());
    try testing.expect(handle > 0);
    try testing.expectEqual(@as(u32, 1), inst.waitable_sets.count());

    // Drop releases it from the table.
    try env.pushI32(@bitCast(handle));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .waitable_set_drop },
        env,
        null,
        testing.allocator,
    );
    try testing.expectEqual(@as(u32, 0), inst.waitable_sets.count());
}

test "dispatchCanonBuiltin: future.new + drop both ends round-trip" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    var comp = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},         .imports = &.{},
        .exports = &.{},
    };
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    // future.new pushes i64 packed handle pair — pop and unpack.
    const packed_handles: u64 = @bitCast(try env.popI64());
    const r_idx: u32 = @truncate(packed_handles >> 32);
    const w_idx: u32 = @truncate(packed_handles & 0xFFFF_FFFF);
    try testing.expectEqual(r_idx, w_idx); // sub-PR 3 stub uses the same idx for both ends
    try testing.expect(r_idx > 0);
    try testing.expectEqual(@as(u32, 1), inst.futures.count());

    // Drop-readable alone retains the table entry — only marks read_closed.
    try env.pushI32(@bitCast(r_idx));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_drop_readable = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    try testing.expectEqual(@as(u32, 1), inst.futures.count());

    // Dropping the second end removes the table entry.
    try env.pushI32(@bitCast(w_idx));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_drop_writable = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    try testing.expectEqual(@as(u32, 0), inst.futures.count());
}

// ── #478 sub-PR 3a: future.read/write rendezvous tests ──────────────────────
//
// Each test wires a minimal `ComponentInstance` whose component-type-index
// space carries a single `(type u32)` entry; `future.new t=0` therefore
// names a `future<u32>`. `enableTestMem` provides a 4 KiB backing buffer
// for the lift/lower memcpy on each side of the rendezvous.

fn newFutureU32Inst(testing_allocator: std.mem.Allocator) !*instance_mod.ComponentInstance {
    // Static lifetimes — the `Component` and its types live as long as
    // the instance, so we leak them deliberately. Tests run in their
    // own process; the test allocator releases bookkeeping on exit.
    const FutureTypeFixture = struct {
        var types_array = [_]ctypes.TypeDef{.{ .val = .u32 }};
        var comp: ctypes.Component = .{
            .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
            .components = &.{},       .instances = &.{},      .aliases = &.{},
            .types = &.{},            .canons = &.{},         .imports = &.{},
            .exports = &.{},
        };
    };
    FutureTypeFixture.comp.types = &FutureTypeFixture.types_array;
    const inst = try instance_mod.instantiate(&FutureTypeFixture.comp, testing_allocator);
    try inst.enableTestMem(testing_allocator, 4096);
    return inst;
}

fn destroyFutureInst(inst: *instance_mod.ComponentInstance) void {
    inst.disableTestMem();
    inst.deinit();
}

test "future.new: returns packed read|write handles" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newFutureU32Inst(testing.allocator);
    defer destroyFutureInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );

    const packed_handles: u64 = @bitCast(try env.popI64());
    const r_idx: u32 = @truncate(packed_handles >> 32);
    const w_idx: u32 = @truncate(packed_handles & 0xFFFF_FFFF);
    // Single-handle prototype: read+write share the same idx.
    try testing.expectEqual(r_idx, w_idx);
    try testing.expect(r_idx > 0);
    try testing.expectEqual(@as(u32, 1), inst.futures.count());

    const fut = inst.futures.getPtr(r_idx).?;
    try testing.expectEqual(@as(u32, 0), fut.elem_type_idx);
    try testing.expect(fut.payload == null);
    try testing.expect(fut.pending_read == null);
}

test "future.write then future.read: round-trips a u32" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newFutureU32Inst(testing.allocator);
    defer destroyFutureInst(inst);

    // Allocate the handle.
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    // Stage the source u32 in test memory at offset 0; destination at 32.
    const src_ptr: u32 = 0;
    const dst_ptr: u32 = 32;
    const src_bytes = inst.writableGuestBytes(src_ptr, 4).?;
    std.mem.writeInt(u32, src_bytes[0..4], 0xDEAD_BEEF, .little);

    // Issue future.write — should buffer the bytes (no reader parked yet).
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(src_ptr));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_write = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const write_status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.returned, 1), write_status);
    try testing.expect(inst.futures.getPtr(handle).?.payload != null);

    // Issue future.read — should copy buffered bytes into dst.
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(dst_ptr));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const read_status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.returned, 1), read_status);

    const dst_bytes = inst.writableGuestBytes(dst_ptr, 4).?;
    try testing.expectEqual(@as(u32, 0xDEAD_BEEF), std.mem.readInt(u32, dst_bytes[0..4], .little));
    // Payload drained.
    try testing.expect(inst.futures.getPtr(handle).?.payload == null);
}

test "future.read parks then future.write wakes a waitable" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newFutureU32Inst(testing.allocator);
    defer destroyFutureInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    // Wire a waitable set + register the future's read side. This is
    // the manual plumbing the eventual `waitable.join` arm will set up;
    // the rendezvous logic only checks that the slots are populated.
    var ws = async_mod.WaitableSet{};
    defer ws.deinit(testing.allocator);
    {
        const fut = inst.futures.getPtr(handle).?;
        fut.waitable_set = &ws;
        const idx = try ws.register(.{ .kind = .future_read, .handle = handle }, testing.allocator);
        fut.read_waitable_idx = idx;
    }

    const src_ptr: u32 = 0;
    const dst_ptr: u32 = 16;
    const src_bytes = inst.writableGuestBytes(src_ptr, 4).?;
    std.mem.writeInt(u32, src_bytes[0..4], 0xCAFE_F00D, .little);

    // Reader arrives first — must park with STARTING and remember the dst.
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(dst_ptr));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const read_status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.starting, 0), read_status);
    try testing.expectEqual(@as(u32, dst_ptr), inst.futures.getPtr(handle).?.pending_read.?.guest_ptr);

    // Writer arrives — copies straight into the parked dst, returns RETURNED.
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(src_ptr));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_write = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const write_status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.returned, 1), write_status);

    // Destination memory updated, payload not buffered, waitable woken.
    const dst_bytes = inst.writableGuestBytes(dst_ptr, 4).?;
    try testing.expectEqual(@as(u32, 0xCAFE_F00D), std.mem.readInt(u32, dst_bytes[0..4], .little));
    try testing.expect(inst.futures.getPtr(handle).?.payload == null);
    try testing.expect(inst.futures.getPtr(handle).?.pending_read == null);
    var ready: [4]u32 = undefined;
    try testing.expectEqual(@as(u32, 1), ws.pollReady(&ready));
}

test "future.cancel-read on empty future returns CANCELLED" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newFutureU32Inst(testing.allocator);
    defer destroyFutureInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    // Park a reader so cancel has something to clear.
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(@as(u32, 0)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    _ = try env.popI32(); // discard STARTING
    try testing.expect(inst.futures.getPtr(handle).?.pending_read != null);

    // Cancel — should clear pending_read and return CANCELLED.
    try env.pushI32(@bitCast(handle));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_cancel_read = .{ .type_idx = 0, .is_async = false } } },
        env,
        null,
        testing.allocator,
    );
    const status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.cancelled, 0), status);
    try testing.expect(inst.futures.getPtr(handle).?.pending_read == null);
}

test "future.drop-writable while reader parked: subsequent read is CANCELLED" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newFutureU32Inst(testing.allocator);
    defer destroyFutureInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const packed_handles: u64 = @bitCast(try env.popI64());
    const r_idx: u32 = @truncate(packed_handles >> 32);
    const w_idx: u32 = @truncate(packed_handles & 0xFFFF_FFFF);

    // Park a reader first so the drop-writable code path observes it.
    try env.pushI32(@bitCast(r_idx));
    try env.pushI32(@bitCast(@as(u32, 16)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    _ = try env.popI32(); // discard STARTING

    // Writer drops without ever writing.
    try env.pushI32(@bitCast(w_idx));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_drop_writable = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    // Entry remains until the reader also drops.
    try testing.expectEqual(@as(u32, 1), inst.futures.count());
    try testing.expect(inst.futures.getPtr(r_idx).?.write_closed);

    // Clear the parked-reader slot and re-issue read — it should observe
    // CANCELLED because the writer is closed and no payload was buffered.
    inst.futures.getPtr(r_idx).?.pending_read = null;
    try env.pushI32(@bitCast(r_idx));
    try env.pushI32(@bitCast(@as(u32, 16)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.cancelled, 0), status);
}

test "future.drop both ends: table entry is freed" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newFutureU32Inst(testing.allocator);
    defer destroyFutureInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const packed_handles: u64 = @bitCast(try env.popI64());
    const r_idx: u32 = @truncate(packed_handles >> 32);
    const w_idx: u32 = @truncate(packed_handles & 0xFFFF_FFFF);

    // Buffer a payload to verify drop frees it (no leak diagnostic from
    // the test allocator).
    const src_ptr: u32 = 0;
    const src_bytes = inst.writableGuestBytes(src_ptr, 4).?;
    std.mem.writeInt(u32, src_bytes[0..4], 0x1234_5678, .little);
    try env.pushI32(@bitCast(r_idx));
    try env.pushI32(@bitCast(src_ptr));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_write = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    _ = try env.popI32(); // discard RETURNED

    try testing.expectEqual(@as(u32, 1), inst.futures.count());
    try testing.expect(inst.futures.getPtr(r_idx).?.payload != null);

    try env.pushI32(@bitCast(w_idx));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_drop_writable = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    try testing.expectEqual(@as(u32, 1), inst.futures.count());

    try env.pushI32(@bitCast(r_idx));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_drop_readable = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    try testing.expectEqual(@as(u32, 0), inst.futures.count());
}

test "dispatchCanonBuiltin: error-context.new + drop round-trip" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    var comp = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},         .imports = &.{},
        .exports = &.{},
    };
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();

    // error-context.new pops (ptr, len) for the debug-message — push
    // synthetic values; the sub-PR 3 stub doesn't actually read from
    // guest memory yet.
    try env.pushI32(0); // ptr
    try env.pushI32(0); // len
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .error_context_new = .{ .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @bitCast(try env.popI32());
    try testing.expect(handle > 0);
    try testing.expectEqual(@as(u32, 1), inst.error_contexts.count());

    try env.pushI32(@bitCast(handle));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .error_context_drop },
        env,
        null,
        testing.allocator,
    );
    try testing.expectEqual(@as(u32, 0), inst.error_contexts.count());
}

test "dispatchCanonBuiltin: stream.read traps with FunctionNotFound (sub-PR 3 scope)" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    var comp = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},         .imports = &.{},
        .exports = &.{},
    };
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();

    const result = dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    try testing.expectError(error.FunctionNotFound, result);
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
                // `async` and `callback` only meaningful on the lift side
                // (Binary.md grammar reuses one `opts` vec for both, so
                // we must accept them here without effect).
                .async_lift, .callback => {},
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

    /// Per-trampoline extension to the component's type indexspace, used
    /// to resolve param/result `.type_idx` references that point into an
    /// instance-type body's local type space. Empty when the FuncType
    /// for this trampoline came from the component-level type indexspace
    /// directly (e.g. hand-authored fixtures).
    extended_types: []const ctypes.TypeDef = &.{},
    extended_indexspace: []const ?u32 = &.{},

    pub fn deinit(self: *ComponentTrampolineCtx, allocator: Allocator) void {
        allocator.free(self.param_types);
        allocator.free(self.result_types);
        if (self.extended_types.len > 0) {
            // The extension TypeDefs are deep-copies built by the
            // trampoline construction site (instance.zig); record /
            // tuple / variant payloads have their own allocations.
            for (self.extended_types) |td| switch (td) {
                .record => |rec| allocator.free(rec.fields),
                .tuple => |tup| allocator.free(tup.fields),
                .variant => |v| allocator.free(v.cases),
                else => {},
            };
            allocator.free(self.extended_types);
        }
        if (self.extended_indexspace.len > 0) allocator.free(self.extended_indexspace);
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
    const registry = if (ctx.extended_types.len > 0)
        TypeRegistry.fromExtended(ctx.comp_inst.component, ctx.extended_types, ctx.extended_indexspace)
    else
        TypeRegistry.init(ctx.comp_inst.component);

    const flat_params = countFlatTypes(registry, ctx.param_types);
    const flat_results = countFlatTypes(registry, ctx.result_types);
    const params_spill = flat_params > MAX_FLAT_PARAMS;
    const results_spill = flat_results > MAX_FLAT_RESULTS;

    // Resolve linear memory for either spill path. The memory option is
    // mandatory whenever spilling occurs, and we need a non-empty
    // core_instances list to actually own a memory.
    if (params_spill or results_spill) {
        if (ctx.lower_opts.memory_idx == null) return trampolineTrap(env, ctx, error.MemoryNotAvailable, .memory_resolve);
        if (ctx.comp_inst.core_instances.len == 0) return trampolineTrap(env, ctx, error.MemoryNotAvailable, .memory_resolve);
    }

    // Pop result-destination pointer first if results spill (it was pushed
    // last by the caller).
    var result_dest_ptr: u32 = 0;
    if (results_spill) {
        result_dest_ptr = @bitCast(env.popI32() catch |err| return trampolineTrap(env, ctx, err, .lift_args));
    }

    // Lift args.
    const args = allocator.alloc(InterfaceValue, ctx.param_types.len) catch |err|
        return trampolineTrap(env, ctx, err, .lift_args);
    defer allocator.free(args);
    if (params_spill) {
        const params_ptr: u32 = @bitCast(env.popI32() catch |err| return trampolineTrap(env, ctx, err, .lift_args));
        const mem_idx = ctx.lower_opts.memory_idx.?;
        const mem = ctx.comp_inst.resolveTopLevelMemory(mem_idx) orelse
            return trampolineTrap(env, ctx, error.MemoryNotAvailable, .memory_resolve);
        var offset: u32 = params_ptr;
        for (ctx.param_types, 0..) |pt, i| {
            const al = typeAlign(registry, pt);
            offset = abi.alignUp(offset, al);
            args[i] = loadInterfaceValue(mem.data, offset, pt, registry, allocator) catch |err|
                return trampolineTrap(env, ctx, err, .lift_args);
            offset += typeSize(registry, pt);
        }
    } else {
        // Walk param_types back-to-front so the last flat core value
        // (which is on top of stack) becomes the last arg.
        var i: usize = ctx.param_types.len;
        while (i > 0) {
            i -= 1;
            args[i] = popInterfaceValue(env, ctx.param_types[i], registry, allocator) catch |err|
                return trampolineTrap(env, ctx, err, .lift_args);
        }
    }

    // Invoke host. Host owns allocation of any compound result values via
    // `allocator`; we deinit each result after lowering so payloads
    // (e.g. `.result_val.payload` for input-stream.blocking-read) don't leak.
    const results = allocator.alloc(InterfaceValue, ctx.result_types.len) catch |err|
        return trampolineTrap(env, ctx, err, .lift_args);
    defer {
        for (results) |r| r.deinit(allocator);
        allocator.free(results);
    }
    const call = ctx.host_func.call orelse {
        return trampolineTrap(env, ctx, error.HostFuncNotBound, .host_call);
    };
    call(ctx.host_func.context, ctx.comp_inst, args, results, allocator) catch |err| {
        return trampolineTrap(env, ctx, err, .host_call);
    };

    // Lower results.
    if (results_spill) {
        const mem_idx = ctx.lower_opts.memory_idx.?;
        const mem_via_resolve = ctx.comp_inst.resolveTopLevelMemory(mem_idx);
        const mem = mem_via_resolve orelse
            return trampolineTrap(env, ctx, error.MemoryNotAvailable, .memory_resolve);
        var offset: u32 = result_dest_ptr;
        for (results, ctx.result_types) |r, t| {
            const al = typeAlign(registry, t);
            offset = abi.alignUp(offset, al);
            storeInterfaceValue(mem.data, offset, r, t, registry);
            offset += typeSize(registry, t);
        }
    } else {
        for (results, ctx.result_types) |r, t| {
            pushInterfaceValue(env, r, t, registry) catch |err| {
                return trampolineTrap(env, ctx, err, .lower_results);
            };
        }
    }
}

/// Record `HostTrapInfo` on the env and return `error.Trap` so the canon-lower
/// trampoline collapses to a single trap-shaped failure for the interp loop.
/// First-write-wins so the deepest captured site survives `recordHostTrap`'s
/// later top-up of `core_func_idx`.
fn trampolineTrap(
    env: *ExecEnv,
    ctx: *const ComponentTrampolineCtx,
    err: anyerror,
    stage: HostTrapInfo.Stage,
) error{Trap} {
    if (env.host_trap == null) {
        env.host_trap = .{
            .component_func_idx = ctx.component_func_idx,
            .err_name = @errorName(err),
            .stage = stage,
        };
    }
    return error.Trap;
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
