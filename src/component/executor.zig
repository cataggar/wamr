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
const aot_runtime = @import("../runtime/aot/runtime.zig");
const host_trampolines = @import("../runtime/aot/host_trampolines.zig");
const core_backend = @import("core_backend.zig");
const call_frame_mod = @import("call_frame.zig");
const debugAotEnabled = core_backend.debugAotEnabled;
const Allocator = std.mem.Allocator;

const ComponentInstance = instance_mod.ComponentInstance;
const InterfaceValue = abi.InterfaceValue;
const TypeRegistry = abi.TypeRegistry;
pub const CallFrame = call_frame_mod.CallFrame;
pub const InterpFrame = call_frame_mod.InterpFrame;
pub const AotFrame = call_frame_mod.AotFrame;

pub const MAX_FLAT_PARAMS: u32 = 16;
pub const MAX_FLAT_RESULTS: u32 = 1;

/// Component-model `MAX_FLAT_PARAMS_ASYNC` per the spec's
/// canonical-ABI rules for `canon.lower (async)`: when a lifted async
/// func has more than 4 flat-params, the caller spills the entire
/// param block to memory and passes a single `params_ptr` instead.
/// wit-bindgen ≥ 0.45 emits shim trampolines that respect this
/// threshold — `[async-lower][method]descriptor.open-at` (6 flat
/// params) lowers to `(params_ptr, retptr) -> status` (2 i32 args),
/// while `create-directory-at` (3 flat params) lowers to
/// `(self, path_ptr, path_len, retptr) -> status` (4 i32 args). (#564.)
pub const MAX_FLAT_PARAMS_ASYNC: u32 = 4;

/// Encode a host-side resource representation (slot index from
/// `pushSocket` / `pushFsDescriptor` / `pushNetwork` / 0-based) into a
/// canon-ABI wire handle. Wit-bindgen Rust's `Resource<T>` constructor
/// asserts `handle != 0 && handle != u32::MAX`, so the wire value
/// must be non-zero. We adopt the convention `wire = rep + 1`. Reps
/// of `u32::MAX` are guarded (they would overflow); upper-layer code
/// should never produce that. (#520 wave 2)
pub fn encodeResourceWire(rep: u32) u32 {
    if (rep == std.math.maxInt(u32)) return std.math.maxInt(u32);
    return rep +% 1;
}

/// Decode a canon-ABI wire handle back into a host-side representation
/// (0-based slot index). Wire `0` decodes to `0` (the host's
/// `lookupSocket(0)` etc. would return null for an out-of-range
/// slot anyway, so the round-trip is safe). (#520 wave 2)
pub fn decodeResourceWire(wire: u32) u32 {
    if (wire == 0) return 0;
    return wire -% 1;
}

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
    /// AOT-backed core hit a canon-ABI shape this PR doesn't yet
    /// support — only scalar primitive params and a single scalar
    /// result land on the fast path; compound types, memory-spilled
    /// params, and post-return on AOT cores are deferred to a
    /// follow-up (see issue #625 phase 3 notes).
    AotPathUnsupported,
    /// `Options.aot_only` was set on this instantiation but at least
    /// one core has an import (or instantiation step) the AOT runtime
    /// cannot satisfy yet. The matching log line carries the import
    /// `module.field` and kind. Surfaced to `wamr run`'s CLI exit
    /// path as a clear error; library callers (tests, embedders)
    /// that don't set `aot_only` keep the silent interp fallback.
    /// See issue #644.
    AotImportUnresolvable,
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
    frame: *CallFrame,
    realloc_idx: u32,
    old_ptr: u32,
    old_size: u32,
    align_val: u32,
    new_size: u32,
) ExecutionError!u32 {
    return frame.realloc(realloc_idx, old_ptr, old_size, align_val, new_size) catch |err| switch (err) {
        error.StackOverflow => error.StackOverflow,
        error.StackUnderflow => error.StackUnderflow,
        error.ReallocFailed => error.ReallocFailed,
        error.OutOfMemory => error.OutOfMemory,
        else => error.ReallocFailed,
    };
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
    if (exported.core_instance_idx >= owner_inst.core_instances.len)
        return error.CoreInstanceNotAvailable;
    const core_entry = owner_inst.core_instances[exported.core_instance_idx];

    // Build a backend-agnostic CallFrame: InterpFrame for the interp
    // core, AotFrame for the AOT core. Both backends are driven through
    // the same canon-lift body below (issue #650). The handful of
    // ABI divergences (spilled-result mode, trap diagnostics) are
    // factored as `switch (frame.*)` predicates rather than separate
    // call sites.
    var frame: CallFrame = blk: {
        if (core_entry.module_inst) |mi| {
            const e = ExecEnv.create(mi, 4096, allocator) catch return error.OutOfMemory;
            break :blk .{ .interp = InterpFrame.init(e) };
        }
        if (core_entry.aot_inst) |ai| {
            if (debugAotEnabled()) {
                std.debug.print(
                    "[aot-debug] callComponentFuncByLocal -> AOT core_inst_idx={d} core_func_idx={d} func_type_idx={d}\n",
                    .{ exported.core_instance_idx, exported.core_func_idx, exported.func_type_idx },
                );
            }
            break :blk .{ .aot = AotFrame.init(ai, allocator) };
        }
        return error.CoreInstanceNotAvailable;
    };
    defer {
        switch (frame) {
            .interp => |f| f.env.destroy(),
            .aot => {},
        }
        frame.deinit();
    }
    const is_aot = switch (frame) { .aot => true, .interp => false };

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

    // Get memory (default to memory 0 when no explicit memory_idx is
    // bound in lift opts — matches canon ABI's default-memory rule and
    // the prior AOT/interp behaviour). Re-fetch around any realloc /
    // executeCore call since `memory.grow` may relocate the backing
    // slice.
    const default_mem_idx: u32 = lift_opts.memory_idx orelse 0;
    var memory: ?[]u8 = frame.memory(default_mem_idx);

    // 4. Lower args onto the core stack
    if (flat_param_count <= MAX_FLAT_PARAMS) {
        // Flatten each arg and push as core values
        for (args, param_types) |arg, pt| {
            pushInterfaceValue(&frame, arg, pt, registry) catch return error.LowerError;
        }
    } else {
        // Spill to memory: allocate space via realloc, store tuple, push ptr
        const realloc_idx = lift_opts.realloc_idx orelse return error.ReallocNotAvailable;
        const tuple_size = computeTupleSize(registry, param_types);
        const tuple_align = computeTupleAlign(registry, param_types);
        const ptr = try callRealloc(&frame, realloc_idx, 0, 0, tuple_align, tuple_size);
        // Re-fetch memory: realloc may have grown it (relocating the slice).
        memory = frame.memory(default_mem_idx);
        const mem = memory orelse return error.MemoryNotAvailable;

        // Store each arg at its offset in the tuple
        var offset: u32 = 0;
        for (args, param_types) |arg, pt| {
            const al = typeAlign(registry, pt);
            offset = abi.alignUp(offset, al);
            storeInterfaceValue(mem, offset, arg, pt, registry);
            offset += typeSize(registry, pt);
        }

        frame.pushSlot(.{ .i32 = @bitCast(ptr) }) catch return error.StackOverflow;
    }

    // 4b. Spilled-result mode on AOT: caller-allocates retptr and
    // passes it as the trailing core arg. The core function returns
    // void in this mode. (Canon ABI v1 spec; matches the convention
    // wit-bindgen emits on AOT-compiled guests.)
    //
    // The interp path uses callee-allocates: the core returns the
    // retptr it allocated itself, popped after executeCore below.
    var aot_retptr: ?u32 = null;
    if (is_aot and flat_result_count > MAX_FLAT_RESULTS and result_types.len > 0) {
        const realloc_idx = lift_opts.realloc_idx orelse return error.ReallocNotAvailable;
        const ret_size = computeTupleSize(registry, result_types);
        const ret_align = computeTupleAlign(registry, result_types);
        const rp = try callRealloc(&frame, realloc_idx, 0, 0, ret_align, ret_size);
        // Re-fetch memory after realloc.
        memory = frame.memory(default_mem_idx);
        frame.pushSlot(.{ .i32 = @bitCast(rp) }) catch return error.StackOverflow;
        aot_retptr = rp;
    }

    // 5. Compute core result types for executeCore. Advisory for
    // interp (signature comes from the module); load-bearing for AOT
    // since callFuncScalar needs an accurate signature. Only compute
    // it for AOT — for interp we pass an empty slice and let
    // `executeFunction` read the core sig from the module.
    var core_rt_buf: [1]core_types.ValType = undefined;
    const core_result_types: []const core_types.ValType = if (!is_aot)
        &.{}
    else if (flat_result_count > MAX_FLAT_RESULTS or result_types.len == 0)
        &.{}
    else blk: {
        core_rt_buf[0] = coreFlatSlotType(result_types[0], registry) catch
            return error.AotPathUnsupported;
        break :blk core_rt_buf[0..1];
    };

    // 5b. Call the core function
    frame.executeCore(exported.core_func_idx, &.{}, core_result_types) catch {
        switch (frame) {
            .interp => |f| if (f.env.host_trap) |ht| {
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
            },
            .aot => {},
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
            out_results[i] = popInterfaceValue(&frame, rt, registry, allocator) catch return error.LiftError;
        }
    } else {
        // Spilled-result path — read tuple from linear memory.
        // Backend split: AOT pre-allocated the retptr (caller-
        // allocates); interp expects the callee to return it.
        if (aot_retptr) |rp| {
            result_ptr_for_post_return = rp;
        } else {
            const popped = frame.popSlot(.i32) catch return error.StackUnderflow;
            result_ptr_for_post_return = @bitCast(popped.i32);
        }
        // Re-fetch memory after executeCore — the core function may
        // have grown linear memory, invalidating the captured slice.
        memory = frame.memory(default_mem_idx);
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
                pushInterfaceValue(&frame, r, rt, registry) catch {};
            }
        } else {
            // Spilled results: post_return receives the result pointer.
            frame.pushSlot(.{ .i32 = @bitCast(result_ptr_for_post_return) }) catch {};
        }
        frame.executeCore(pr_idx, &.{}, &.{}) catch {};
    }
}

/// Map a single-flat-slot interface result type to its core wasm
/// value type. Only valid when the interface type flattens to exactly
/// one core slot (i.e. `flat_result_count <= MAX_FLAT_RESULTS=1`).
/// Multi-slot results spill to memory and the core function returns
/// void instead — see the `aot_retptr` / `flat_result_count >
/// MAX_FLAT_RESULTS` branch in `callComponentFuncByLocal`.
fn coreFlatSlotType(t: ctypes.ValType, registry: TypeRegistry) !core_types.ValType {
    return switch (t) {
        .bool, .s8, .u8, .s16, .u16, .s32, .u32, .char => .i32,
        .s64, .u64 => .i64,
        .f32 => .f32,
        .f64 => .f64,
        .enum_ => .i32,
        .own, .borrow, .future, .stream, .error_context => .i32,
        .result, .variant, .option, .flags, .tuple, .record => blk: {
            if (abi.flattenCount(registry, t) != 1) return error.AotPathUnsupported;
            break :blk .i32;
        },
        .type_idx => |idx| blk: {
            const td = registry.get(idx) orelse return error.AotPathUnsupported;
            // Forward aggregate TypeDefs to the matching inline ValType so
            // the recursion lands in the inline arm above (which enforces
            // the `flattenCount == 1 → .i32` guard) or the direct `.enum_`
            // arm. The `u32` payload of each ValType aggregate variant is
            // itself a typeidx, and `flattenCount` collapses `type_idx`
            // indirection identically to the inline form (see
            // `canonical_abi.flattenCount` / `flattenCountDef`). So this
            // is a pure reify-table extension — no semantic change.
            // Without it, `wasi:cli/run.run`'s `result<unit,unit>`
            // declared as a stand-alone TypeDef traps every AOT-backed
            // export with `AotPathUnsupported` (issue #683).
            const reified: ctypes.ValType = switch (td) {
                .val => |inner| inner,
                .resource => .{ .own = idx },
                .result => .{ .result = idx },
                .variant => .{ .variant = idx },
                .option => .{ .option = idx },
                .flags => .{ .flags = idx },
                .enum_ => .{ .enum_ = idx },
                .tuple => .{ .tuple = idx },
                .record => .{ .record = idx },
                else => return error.AotPathUnsupported,
            };
            break :blk try coreFlatSlotType(reified, registry);
        },
        else => error.AotPathUnsupported,
    };
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

    var frame: CallFrame = .{ .interp = InterpFrame.init(env) };
    defer frame.deinit();

    const memory: ?[]u8 = if (lift_opts.memory_idx) |mem_idx|
        frame.memory(mem_idx)
    else
        null;

    // Lower args — same logic as the sync path.
    if (flat_param_count <= MAX_FLAT_PARAMS) {
        for (args, param_types) |arg, pt| {
            pushInterfaceValue(&frame, arg, pt, registry) catch return error.LowerError;
        }
    } else {
        const mem = memory orelse return error.MemoryNotAvailable;
        const realloc_idx = lift_opts.realloc_idx orelse return error.ReallocNotAvailable;
        const tuple_size = computeTupleSize(registry, param_types);
        const tuple_align = computeTupleAlign(registry, param_types);
        const ptr = try callRealloc(&frame, realloc_idx, 0, 0, tuple_align, tuple_size);

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
    //
    // On trap, surface diagnostic info from `env.host_trap` so the
    // failure mode is visible to the operator. Mirrors the sync-call
    // path in `callComponentFunc` (added in #520 wave 1 / PR #532).
    // The `WasiExit` case is normal control flow (the exit code is
    // already stashed on `WasiCliAdapter.exit_code`) and is suppressed
    // to avoid spurious diagnostics on a successful `wasi:cli/exit`.
    interp.executeFunction(env, exported.core_func_idx) catch {
        if (env.host_trap) |ht| {
            const is_wasi_exit = std.mem.eql(u8, ht.err_name, "WasiExit");
            if (!is_wasi_exit) {
                std.debug.print("[async-lifted trap] core_func_idx={d}", .{ht.core_func_idx});
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

fn pushInterfaceValue(frame: *CallFrame, val: InterfaceValue, t: ctypes.ValType, registry: TypeRegistry) !void {
    switch (t) {
        .bool => try frame.pushSlot(.{ .i32 = if (val.bool) 1 else 0 }),
        .s8 => try frame.pushSlot(.{ .i32 = @as(i32, val.s8) }),
        .u8 => try frame.pushSlot(.{ .i32 = @as(i32, @intCast(val.u8)) }),
        .s16 => try frame.pushSlot(.{ .i32 = @as(i32, val.s16) }),
        .u16 => try frame.pushSlot(.{ .i32 = @as(i32, @intCast(val.u16)) }),
        .s32 => try frame.pushSlot(.{ .i32 = val.s32 }),
        .u32, .char => try frame.pushSlot(.{ .i32 = @bitCast(val.u32) }),
        .s64 => try frame.pushSlot(.{ .i64 = val.s64 }),
        .u64 => try frame.pushSlot(.{ .i64 = @bitCast(val.u64) }),
        .f32 => try frame.pushSlot(.{ .f32 = @bitCast(val.f32) }),
        .f64 => try frame.pushSlot(.{ .f64 = @bitCast(val.f64) }),
        .own, .borrow => try frame.pushSlot(.{ .i32 = @bitCast(encodeResourceWire(val.handle)) }),
        .future, .stream, .error_context => try frame.pushSlot(.{ .i32 = @bitCast(val.handle) }),
        .string => {
            try frame.pushSlot(.{ .i32 = @bitCast(val.string.ptr) });
            try frame.pushSlot(.{ .i32 = @bitCast(val.string.len) });
        },
        .list => {
            try frame.pushSlot(.{ .i32 = @bitCast(val.list.ptr) });
            try frame.pushSlot(.{ .i32 = @bitCast(val.list.len) });
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
            try frame.pushSlot(.{ .i32 = disc });

            const arm_type: ?ctypes.ValType = if (val.result_val.is_ok) r.ok else r.err;
            var pushed: u32 = 0;
            if (arm_type) |at| {
                if (val.result_val.payload) |p| {
                    try pushInterfaceValue(frame, p.*, at, registry);
                    pushed = abi.flattenCount(registry, at);
                }
            }
            while (pushed < total_payload_slots) : (pushed += 1) {
                try frame.pushSlot(.{ .i32 = 0 });
            }
        },
        // Compound types — lower into a scratch buffer via
        // `lowerFlatReg`, then push the flat i32 slots onto the stack
        // in canonical order. The variant / record / tuple / flags /
        // enum_ / option shapes show up in sockets/filesystem fixture
        // results (e.g. a host that returns `result<_, error-code>`
        // with an `error-code` variant payload). (#520 wave 2)
        .record, .variant, .tuple, .flags, .enum_, .option => {
            const total_slots = abi.flattenCount(registry, t);
            var slot_buf: [32]u32 = undefined;
            if (total_slots > slot_buf.len) return error.CompoundNeedsRegistry;
            const written = try lowerFlatReg(slot_buf[0..total_slots], val, t, registry);
            // Pad any unused tail with zero (variant / option / result
            // join behaviour: shorter arms zero-fill the remaining
            // slots so the join's flat width is constant).
            for (written..total_slots) |k| slot_buf[k] = 0;
            try frame.pushSlotsU32(slot_buf[0..total_slots]);
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
            try pushInterfaceValue(frame, val, reified, registry);
        },
    }
}

/// Lower a single value into a flat slot buffer (canonical order),
/// returning the number of slots written. The buffer must be at least
/// `flattenCount(registry, t)` slots long; the caller zero-pads any
/// remaining tail (variant/option/result join behaviour). (#520 wave 2)
fn lowerFlatReg(
    out: []u32,
    val: InterfaceValue,
    t: ctypes.ValType,
    registry: TypeRegistry,
) error{ CompoundNeedsRegistry, InvalidTypeIndex, InvalidValue, BufferTooSmall }!u32 {
    if (out.len == 0) return error.BufferTooSmall;
    switch (t) {
        .bool => {
            out[0] = if (val.bool) 1 else 0;
            return 1;
        },
        .s8 => {
            out[0] = @as(u32, @intCast(@as(u8, @bitCast(val.s8))));
            return 1;
        },
        .u8 => {
            out[0] = val.u8;
            return 1;
        },
        .s16 => {
            out[0] = @as(u32, @intCast(@as(u16, @bitCast(val.s16))));
            return 1;
        },
        .u16 => {
            out[0] = val.u16;
            return 1;
        },
        .s32 => {
            out[0] = @bitCast(val.s32);
            return 1;
        },
        .u32, .char => {
            out[0] = val.u32;
            return 1;
        },
        .s64 => {
            if (out.len < 2) return error.BufferTooSmall;
            const bits: u64 = @bitCast(val.s64);
            out[0] = @truncate(bits);
            out[1] = @truncate(bits >> 32);
            return 2;
        },
        .u64 => {
            if (out.len < 2) return error.BufferTooSmall;
            out[0] = @truncate(val.u64);
            out[1] = @truncate(val.u64 >> 32);
            return 2;
        },
        .f32 => {
            out[0] = val.f32;
            return 1;
        },
        .f64 => {
            if (out.len < 2) return error.BufferTooSmall;
            out[0] = @truncate(val.f64);
            out[1] = @truncate(val.f64 >> 32);
            return 2;
        },
        .own, .borrow => {
            out[0] = encodeResourceWire(val.handle);
            return 1;
        },
        .future, .stream, .error_context => {
            out[0] = val.handle;
            return 1;
        },
        .string => {
            if (out.len < 2) return error.BufferTooSmall;
            out[0] = val.string.ptr;
            out[1] = val.string.len;
            return 2;
        },
        .list => {
            if (out.len < 2) return error.BufferTooSmall;
            out[0] = val.list.ptr;
            out[1] = val.list.len;
            return 2;
        },
        .enum_ => |idx| {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
            if (td != .enum_) return error.InvalidTypeIndex;
            out[0] = val.enum_val;
            return 1;
        },
        .flags => |idx| {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
            if (td != .flags) return error.InvalidTypeIndex;
            const n_words: u32 = @intCast(@max(@as(usize, 1), (td.flags.names.len + 31) / 32));
            if (out.len < n_words) return error.BufferTooSmall;
            const words = val.flags_val;
            for (0..n_words) |k| out[k] = if (k < words.len) words[k] else 0;
            return n_words;
        },
        .record => |idx| {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
            if (td != .record) return error.InvalidTypeIndex;
            const fields_meta = td.record.fields;
            const fields_val = val.record_val;
            if (fields_val.len != fields_meta.len) return error.InvalidValue;
            var off: u32 = 0;
            for (fields_meta, 0..) |f, i| {
                const ft = resolveArmType(f.type, registry);
                const w = try lowerFlatReg(out[off..], fields_val[i], ft, registry);
                off += w;
            }
            return off;
        },
        .tuple => |idx| {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
            if (td != .tuple) return error.InvalidTypeIndex;
            const fields_meta = td.tuple.fields;
            const fields_val = val.tuple_val;
            if (fields_val.len != fields_meta.len) return error.InvalidValue;
            var off: u32 = 0;
            for (fields_meta, 0..) |f, i| {
                const ft = resolveArmType(f, registry);
                const w = try lowerFlatReg(out[off..], fields_val[i], ft, registry);
                off += w;
            }
            return off;
        },
        .variant => |idx| {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
            if (td != .variant) return error.InvalidTypeIndex;
            const cases = td.variant.cases;
            const v = val.variant_val;
            if (v.discriminant >= cases.len) return error.InvalidValue;
            out[0] = v.discriminant;
            var off: u32 = 1;
            if (cases[v.discriminant].type) |ct| {
                if (v.payload) |p| {
                    const resolved = resolveArmType(ct, registry);
                    off += try lowerFlatReg(out[off..], p.*, resolved, registry);
                }
            }
            // Caller zero-pads tail to total flat width.
            return off;
        },
        .option => |idx| {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
            if (td != .option) return error.InvalidTypeIndex;
            const o = val.option_val;
            out[0] = if (o.is_some) 1 else 0;
            var off: u32 = 1;
            if (o.is_some) {
                if (o.payload) |p| {
                    const inner = resolveArmType(td.option.inner, registry);
                    off += try lowerFlatReg(out[off..], p.*, inner, registry);
                }
            }
            return off;
        },
        .result => |idx| {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
            if (td != .result) return error.InvalidTypeIndex;
            const r = val.result_val;
            out[0] = if (r.is_ok) 0 else 1;
            var off: u32 = 1;
            const arm_type: ?ctypes.ValType = if (r.is_ok) td.result.ok else td.result.err;
            if (arm_type) |at| {
                if (r.payload) |p| {
                    const resolved = resolveArmType(at, registry);
                    off += try lowerFlatReg(out[off..], p.*, resolved, registry);
                }
            }
            return off;
        },
        .type_idx => |idx| {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
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
                else => return error.InvalidTypeIndex,
            };
            return try lowerFlatReg(out, val, reified, registry);
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

fn popInterfaceValue(frame: *CallFrame, t: ctypes.ValType, registry: TypeRegistry, allocator: Allocator) !InterfaceValue {
    return switch (t) {
        .bool => .{ .bool = (try frame.popSlot(.i32)).i32 != 0 },
        .s8 => .{ .s8 = @truncate((try frame.popSlot(.i32)).i32) },
        .u8 => .{ .u8 = @truncate(@as(u32, @bitCast((try frame.popSlot(.i32)).i32))) },
        .s16 => .{ .s16 = @truncate((try frame.popSlot(.i32)).i32) },
        .u16 => .{ .u16 = @truncate(@as(u32, @bitCast((try frame.popSlot(.i32)).i32))) },
        .s32 => .{ .s32 = (try frame.popSlot(.i32)).i32 },
        .u32, .char => .{ .u32 = @bitCast((try frame.popSlot(.i32)).i32) },
        .s64 => .{ .s64 = (try frame.popSlot(.i64)).i64 },
        .u64 => .{ .u64 = @bitCast((try frame.popSlot(.i64)).i64) },
        .f32 => blk: {
            const v = try frame.popSlot(.f32);
            break :blk .{ .f32 = switch (v) {
                .f32 => |f| @bitCast(f),
                .i32 => |i| @bitCast(i),
                else => 0,
            } };
        },
        .f64 => blk: {
            const v = try frame.popSlot(.f64);
            break :blk .{ .f64 = switch (v) {
                .f64 => |f| @bitCast(f),
                .i64 => |i| @bitCast(i),
                else => 0,
            } };
        },
        .own, .borrow => .{ .handle = decodeResourceWire(@bitCast((try frame.popSlot(.i32)).i32)) },
        .future, .stream, .error_context => .{ .handle = @bitCast((try frame.popSlot(.i32)).i32) },
        .string => .{ .string = .{
            .len = @bitCast((try frame.popSlot(.i32)).i32),
            .ptr = @bitCast((try frame.popSlot(.i32)).i32),
        } },
        .list => .{ .list = .{
            .len = @bitCast((try frame.popSlot(.i32)).i32),
            .ptr = @bitCast((try frame.popSlot(.i32)).i32),
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
            // Pop payload slots in canonical forward order (CallFrame
            // hides the interp's LIFO stack reversal — `popSlotsU32`
            // writes back-to-front for the interp backend, and reads
            // straight from the AOT results buffer for AOT).
            try frame.popSlotsU32(slot_buf[0..total_payload_slots]);
            const disc = (try frame.popSlot(.i32)).i32;
            const is_ok = disc == 0;
            const arm_type: ?ctypes.ValType = if (is_ok) rt.ok else rt.err;
            var payload: ?*InterfaceValue = null;
            if (arm_type) |at| {
                // Resolve through `type_idx` hops so a resource handle
                // referenced as `.type_idx -> .resource` is treated as a
                // direct `own<R>` for the purpose of lifting.
                const resolved = resolveArmType(at, registry);
                const arm_slots = abi.flattenCount(registry, resolved);
                // Prefer the registry-aware lift so compound arms
                // (variant / option / result / enum) get a proper
                // payload (#552). Falls back to `liftFlat` only when
                // both fail.
                if (abi.liftFlatReg(slot_buf[0..arm_slots], resolved, registry, allocator)) |lifted| {
                    const p = try allocator.create(InterfaceValue);
                    p.* = lifted;
                    payload = p;
                } else |_| {
                    if (abi.liftFlat(slot_buf[0..arm_slots], resolved)) |lifted| {
                        const p = try allocator.create(InterfaceValue);
                        p.* = lifted;
                        payload = p;
                    } else |_| {}
                }
            }
            break :blk .{ .result_val = .{ .is_ok = is_ok, .payload = payload } };
        },
        // `option<T>`: symmetric to `.result` above. Pop payload slots,
        // pop disc; if `is_some`, lift the buffered slots — preferring
        // the registry-aware lifter so compound inners (variant /
        // option / result) keep their payload (#552).
        .option => |idx| blk: {
            const td = registry.get(idx) orelse return error.CompoundNeedsRegistry;
            const inner_type: ctypes.ValType = switch (td) {
                .option => |o| o.inner,
                else => return error.CompoundNeedsRegistry,
            };
            const total_payload_slots = abi.flattenCount(registry, t) - 1;
            var slot_buf: [16]u32 = undefined;
            if (total_payload_slots > slot_buf.len) return error.CompoundNeedsRegistry;
            try frame.popSlotsU32(slot_buf[0..total_payload_slots]);
            const disc = (try frame.popSlot(.i32)).i32;
            const is_some = disc != 0;
            var payload: ?*InterfaceValue = null;
            if (is_some) {
                const resolved = resolveArmType(inner_type, registry);
                const inner_slots = abi.flattenCount(registry, resolved);
                if (abi.liftFlatReg(slot_buf[0..inner_slots], resolved, registry, allocator)) |lifted| {
                    const p = try allocator.create(InterfaceValue);
                    p.* = lifted;
                    payload = p;
                } else |_| {
                    if (abi.liftFlat(slot_buf[0..inner_slots], resolved)) |lifted| {
                        const p = try allocator.create(InterfaceValue);
                        p.* = lifted;
                        payload = p;
                    } else |_| {}
                }
            }
            break :blk .{ .option_val = .{ .is_some = is_some, .payload = payload } };
        },
        // Compound types — pop all flat slots into a scratch buffer
        // (canonical order: slot[0]..slot[N-1]) and recursively lift
        // through the registry. The variant / record / tuple / flags /
        // enum_ shapes all show up in sockets and filesystem fixture
        // arguments (e.g. `ip-socket-address` is a variant of records
        // of u16+tuple). Supersedes the variant-only path landed in
        // PR #559 (#552). (#520 wave 2)
        .record, .variant, .tuple, .flags, .enum_ => |idx| blk: {
            _ = idx;
            const total_slots = abi.flattenCount(registry, t);
            break :blk try popFlatCompound(frame, t, total_slots, registry, allocator);
        },
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
            break :blk try popInterfaceValue(frame, reified, registry, allocator);
        },
    };
}

/// Pop `total_slots` flat i32 core values off the stack (top-of-stack is
/// the last-pushed = highest-index slot) into a scratch buffer in
/// canonical order (`slot[0]..slot[N-1]`), then lift the compound value
/// from the buffer via `liftFlatReg`. Used by `popInterfaceValue` for
/// `record` / `variant` / `tuple` / `flags` / `enum_` types whose flat
/// layout is determined by walking the type registry.
///
/// The `total_slots <= 32` cap matches the canon ABI's MAX_FLAT_PARAMS
/// (16) doubled to accommodate a return-result tuple of equal size; any
/// compound exceeding this would have spilled to memory anyway, in
/// which case the `loadInterfaceValue` (memory-spill) path is used
/// instead. (#520 wave 2)
fn popFlatCompound(
    frame: *CallFrame,
    t: ctypes.ValType,
    total_slots: u32,
    registry: TypeRegistry,
    allocator: Allocator,
) !InterfaceValue {
    var slot_buf: [32]u32 = undefined;
    if (total_slots > slot_buf.len) return error.CompoundNeedsRegistry;
    try frame.popSlotsU32(slot_buf[0..total_slots]);
    const lifted = try liftFlatReg(slot_buf[0..total_slots], t, registry, allocator);
    return lifted.val;
}

/// Lift a single value from a flat slot buffer (canonical order). Returns
/// the lifted `InterfaceValue` and the number of slots consumed. Handles
/// every shape the canon-lower trampoline can encounter at the import
/// boundary: primitives, handles, string/list ptr+len pairs, and the
/// registry-bound compounds (record/variant/tuple/flags/enum_/option/
/// result/type_idx). Slots are treated as i32 (canon ABI flat repr for
/// the i32-only-join case used by sockets / filesystem fixtures); i64
/// joins read consecutive low+high u32 slots. (#520 wave 2)
fn liftFlatReg(
    slots: []const u32,
    t: ctypes.ValType,
    registry: TypeRegistry,
    allocator: Allocator,
) error{ OutOfMemory, InvalidDiscriminant, InvalidTypeIndex, EmptySlots, CompoundNeedsRegistry }!struct { val: InterfaceValue, used: u32 } {
    if (slots.len == 0) return error.EmptySlots;
    return switch (t) {
        .bool => .{ .val = .{ .bool = slots[0] != 0 }, .used = 1 },
        .s8 => .{ .val = .{ .s8 = @bitCast(@as(u8, @truncate(slots[0]))) }, .used = 1 },
        .u8 => .{ .val = .{ .u8 = @truncate(slots[0]) }, .used = 1 },
        .s16 => .{ .val = .{ .s16 = @bitCast(@as(u16, @truncate(slots[0]))) }, .used = 1 },
        .u16 => .{ .val = .{ .u16 = @truncate(slots[0]) }, .used = 1 },
        .s32 => .{ .val = .{ .s32 = @bitCast(slots[0]) }, .used = 1 },
        .u32, .char => .{ .val = .{ .u32 = slots[0] }, .used = 1 },
        .s64 => blk: {
            const lo: u64 = slots[0];
            const hi: u64 = if (slots.len > 1) slots[1] else 0;
            break :blk .{ .val = .{ .s64 = @bitCast(hi << 32 | lo) }, .used = 2 };
        },
        .u64 => blk: {
            const lo: u64 = slots[0];
            const hi: u64 = if (slots.len > 1) slots[1] else 0;
            break :blk .{ .val = .{ .u64 = hi << 32 | lo }, .used = 2 };
        },
        .f32 => .{ .val = .{ .f32 = slots[0] }, .used = 1 },
        .f64 => blk: {
            const lo: u64 = slots[0];
            const hi: u64 = if (slots.len > 1) slots[1] else 0;
            break :blk .{ .val = .{ .f64 = hi << 32 | lo }, .used = 2 };
        },
        .own, .borrow => .{ .val = .{ .handle = decodeResourceWire(slots[0]) }, .used = 1 },
        .future, .stream, .error_context => .{ .val = .{ .handle = slots[0] }, .used = 1 },
        .string => .{ .val = .{ .string = .{ .ptr = slots[0], .len = if (slots.len > 1) slots[1] else 0 } }, .used = 2 },
        .list => .{ .val = .{ .list = .{ .ptr = slots[0], .len = if (slots.len > 1) slots[1] else 0 } }, .used = 2 },
        .enum_ => |idx| blk: {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
            if (td != .enum_) return error.InvalidTypeIndex;
            const disc = slots[0];
            if (disc >= td.enum_.names.len) return error.InvalidDiscriminant;
            break :blk .{ .val = .{ .enum_val = disc }, .used = 1 };
        },
        .flags => |idx| blk: {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
            if (td != .flags) return error.InvalidTypeIndex;
            const n_words: u32 = @intCast(@max(@as(usize, 1), (td.flags.names.len + 31) / 32));
            if (slots.len < n_words) return error.EmptySlots;
            const words = try allocator.alloc(u32, n_words);
            for (0..n_words) |k| words[k] = slots[k];
            break :blk .{ .val = .{ .flags_val = words }, .used = n_words };
        },
        .record => |idx| blk: {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
            if (td != .record) return error.InvalidTypeIndex;
            const fields = td.record.fields;
            const vals = try allocator.alloc(InterfaceValue, fields.len);
            errdefer {
                for (vals) |v| v.deinit(allocator);
                allocator.free(vals);
            }
            var used: u32 = 0;
            for (fields, 0..) |f, i| {
                const ft = resolveArmType(f.type, registry);
                const r = try liftFlatReg(slots[used..], ft, registry, allocator);
                vals[i] = r.val;
                used += r.used;
            }
            break :blk .{ .val = .{ .record_val = vals }, .used = used };
        },
        .tuple => |idx| blk: {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
            if (td != .tuple) return error.InvalidTypeIndex;
            const fields = td.tuple.fields;
            const vals = try allocator.alloc(InterfaceValue, fields.len);
            errdefer {
                for (vals) |v| v.deinit(allocator);
                allocator.free(vals);
            }
            var used: u32 = 0;
            for (fields, 0..) |f, i| {
                const ft = resolveArmType(f, registry);
                const r = try liftFlatReg(slots[used..], ft, registry, allocator);
                vals[i] = r.val;
                used += r.used;
            }
            break :blk .{ .val = .{ .tuple_val = vals }, .used = used };
        },
        .variant => |idx| blk: {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
            if (td != .variant) return error.InvalidTypeIndex;
            const cases = td.variant.cases;
            const disc = slots[0];
            if (disc >= cases.len) return error.InvalidDiscriminant;
            // Discriminant + max-of-payload-flatten-counts (i32-only join).
            const total_payload = abi.flattenCount(registry, t) - 1;
            var payload: ?*InterfaceValue = null;
            if (cases[disc].type) |ct| {
                const resolved = resolveArmType(ct, registry);
                const arm_slots = abi.flattenCount(registry, resolved);
                if (1 + arm_slots > slots.len) return error.EmptySlots;
                const r = try liftFlatReg(slots[1 .. 1 + arm_slots], resolved, registry, allocator);
                const p = try allocator.create(InterfaceValue);
                p.* = r.val;
                payload = p;
            }
            break :blk .{
                .val = .{ .variant_val = .{ .discriminant = disc, .payload = payload } },
                .used = 1 + total_payload,
            };
        },
        .option => |idx| blk: {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
            if (td != .option) return error.InvalidTypeIndex;
            const disc = slots[0];
            if (disc > 1) return error.InvalidDiscriminant;
            const total_payload = abi.flattenCount(registry, t) - 1;
            if (disc == 0) {
                break :blk .{
                    .val = .{ .option_val = .{ .is_some = false, .payload = null } },
                    .used = 1 + total_payload,
                };
            }
            const inner = resolveArmType(td.option.inner, registry);
            const r = try liftFlatReg(slots[1..], inner, registry, allocator);
            const p = try allocator.create(InterfaceValue);
            p.* = r.val;
            break :blk .{
                .val = .{ .option_val = .{ .is_some = true, .payload = p } },
                .used = 1 + total_payload,
            };
        },
        .result => |idx| blk: {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
            if (td != .result) return error.InvalidTypeIndex;
            const disc = slots[0];
            if (disc > 1) return error.InvalidDiscriminant;
            const is_ok = disc == 0;
            const arm_type: ?ctypes.ValType = if (is_ok) td.result.ok else td.result.err;
            const total_payload = abi.flattenCount(registry, t) - 1;
            var payload: ?*InterfaceValue = null;
            if (arm_type) |at| {
                const resolved = resolveArmType(at, registry);
                const arm_slots = abi.flattenCount(registry, resolved);
                if (1 + arm_slots > slots.len) return error.EmptySlots;
                const r = try liftFlatReg(slots[1 .. 1 + arm_slots], resolved, registry, allocator);
                const p = try allocator.create(InterfaceValue);
                p.* = r.val;
                payload = p;
            }
            break :blk .{
                .val = .{ .result_val = .{ .is_ok = is_ok, .payload = payload } },
                .used = 1 + total_payload,
            };
        },
        .type_idx => |idx| blk: {
            const td = registry.get(idx) orelse return error.InvalidTypeIndex;
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
                else => return error.InvalidTypeIndex,
            };
            break :blk try liftFlatReg(slots, reified, registry, allocator);
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
/// Resolve the per-element byte size for `stream.{read,write}` /
/// `future.{read,write}` from the canon's `type_idx` immediate.
///
/// The component-model canon `stream.read t` / `future.read t` carries
/// the typeidx of the *stream/future type def* — the element type sits
/// one level inside, encoded by the loader as
/// `TypeDef.val(.stream{inner})` / `TypeDef.val(.future{inner})` where
/// `inner` is either a typeidx or a sentinel-encoded primitive (see
/// `types.zig::decodeStreamFutureInner`).
///
/// To preserve compatibility with the hand-crafted test fixtures from
/// #478 sub-PR 3 — which pass an *element* typeidx directly — this helper
/// falls back to `abi.sizeOfType(.type_idx = type_idx)` when the typeidx
/// does NOT resolve to a wrapping stream/future deftype.
fn streamFutureElemSize(reg: TypeRegistry, type_idx: u32) u32 {
    if (reg.resolve(ctypes.ValType{ .type_idx = type_idx })) |td| {
        switch (td) {
            .val => |v| switch (v) {
                .stream => |inner| return decodeAndSize(reg, inner),
                .future => |inner| return decodeAndSize(reg, inner),
                else => {},
            },
            else => {},
        }
    }
    // Test-fixture / element-typeidx fallback path.
    return abi.sizeOfType(reg, ctypes.ValType{ .type_idx = type_idx });
}

fn decodeAndSize(reg: TypeRegistry, inner: u32) u32 {
    switch (ctypes.decodeStreamFutureInner(inner)) {
        .empty => return 0,
        .primitive => |prim| return abi.sizeOfType(reg, prim),
        .typeidx => |idx| return abi.sizeOfType(reg, ctypes.ValType{ .type_idx = idx }),
    }
}

pub fn dispatchCanonBuiltin(
    comp_inst: *ComponentInstance,
    canon: ctypes.Canon,
    env: *ExecEnv,
    task_manager: ?*async_mod.TaskManager,
    allocator: Allocator,
) ExecutionError!void {
    return dispatchCanonBuiltinWithCtx(comp_inst, canon, null, env, task_manager, allocator);
}

/// `dispatchCanonBuiltin` variant that threads the trampoline ctx
/// through to dispatch handlers that need it (today: `task.return`,
/// which needs the importing module's flat-param count to drain the
/// guest's flat-lowered results when the result type's inner variant
/// can't be flattened from the parent type pool). The ctx may be
/// null for call-sites that synthesize a `Canon` decl without going
/// through the import-link path (unit tests). (#570)
pub fn dispatchCanonBuiltinWithCtx(
    comp_inst: *ComponentInstance,
    canon: ctypes.Canon,
    ctx: ?*const CanonBuiltinTrampolineCtx,
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
            // Notify the host adapter so it can release any kernel-
            // side state attached to the dropped handle. WAMR's host
            // fns return reps directly as wire handles (no automatic
            // `canon resource.new` wrapping), so the per-type
            // resource table at `resource_idx` is typically empty
            // here; we still forward the raw `(resource_idx, handle)`
            // pair so adapters that maintain their own rep tables can
            // clean up synchronously. (#575)
            if (comp_inst.on_resource_drop) |hook| {
                hook(comp_inst.on_resource_drop_ctx, comp_inst, resource_idx, handle);
            }
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
            // (flat i32/i64/f32/f64 representation per the Canonical ABI),
            // stores them on the current task, and notifies the parent
            // waitable. Caller without a task manager is malformed (a
            // sync lift never emits task.return).
            const tm = task_manager orelse return error.FunctionNotFound;
            const handle = tm.current_task orelse return error.FunctionNotFound;
            // Prefer the link-time-snapshotted core wasm import flat
            // param count (`ctx.core_flat_param_count`) — this is the
            // authoritative truth of how many typed slots the guest
            // pushed onto the operand stack, regardless of whether the
            // canon ABI result type's inner variant cases can be
            // flattened from the parent component's local type pool.
            // (Today's loader only materialises primitive aliased types;
            // a `result<own<response>, error-code>` whose `error-code`
            // is alias-imported from a sub-component falls through to
            // `flattenCount` → 1, which would under-drain the stack —
            // see #570.) Fall back to `flattenCount` for hand-authored
            // dispatch tests that synthesize a `Canon.task_return`
            // without going through `linkImports`.
            const registry = TypeRegistry.init(comp_inst.component);
            const flat_count: usize = blk: {
                if (ctx) |c| if (c.core_flat_param_count) |n| break :blk n;
                break :blk switch (info.results) {
                    .none => 0,
                    .unnamed => |vt| abi.flattenCount(registry, vt),
                    .named => |named| nb: {
                        var n: u32 = 0;
                        for (named) |nv| n += abi.flattenCount(registry, nv.type);
                        break :nb n;
                    },
                };
            };
            const flat = allocator.alloc(u32, flat_count) catch return error.OutOfMemory;
            // Pop `flat_count` typed wasm operand slots in reverse
            // (Canonical ABI pushes left-to-right). Each slot is one
            // `Value` on the operand stack — popI32 silently truncates
            // i64/f64 to their low 32 bits, which is sufficient for
            // our `[]u32` storage shape (i64 result slots in async
            // result types are only used for payload-bearing variant
            // cases the http-service fixture never produces). Long-term
            // a typed flat storage will let us preserve i64 fidelity
            // round-trip; for now we just need to keep the stack
            // balanced and the first two i32 slots (disc + first-arm
            // handle) intact.
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
        .task_cancel => {
            // `canon task.cancel` — Binary.md tag 0x05. Cancels the
            // **currently executing** task (no handle on the stack,
            // distinct from `subtask.cancel`). Core signature is
            // `[] -> []`: no pops, no pushes. Traps when there is no
            // async task manager or no current task — guest code that
            // issues `task.cancel` from a non-async context is
            // malformed by the spec.
            const tm = task_manager orelse return error.FunctionNotFound;
            const handle = tm.current_task orelse return error.FunctionNotFound;
            tm.cancelTask(handle);
            // Propagate the cancellation to host-side waitables owned by
            // the cancelled task. Specifically, this aborts any pending
            // `wasi:clocks` `wait-for`/`wait-until` timer future so the
            // wait settles with the cancel disposition the next time the
            // guest issues `waitable-set.{wait,poll}`. (#551 / multi-clock-wait.)
            if (comp_inst.async_cancel_driver) |drv| {
                drv(comp_inst.async_event_driver_ctx, comp_inst, handle, allocator);
            }
        },

        // ── Stream handles ──────────────────────────────────────────────
        .stream_new => |info| {
            const handle = comp_inst.allocAsyncHandle();
            comp_inst.streams.put(comp_inst.allocator, handle, .{
                .elem_type_idx = info.type_idx,
            }) catch return error.OutOfMemory;
            // Spec packs read+write handles into one i64; in our
            // single-handle prototype we publish the same idx for both
            // ends to keep the wire format compatible with i64 pops.
            const packed_handles: u64 = (@as(u64, handle) << 32) | @as(u64, handle);
            env.pushI64(@bitCast(packed_handles)) catch return error.StackOverflow;
        },
        .stream_read => |info| {
            // Stack: (handle, ptr, max_count) → status. `max_count` is on
            // top, then `ptr`, then `handle`.
            const max_count: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const guest_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);

            const s = comp_inst.streams.getPtr(handle) orelse {
                // Unknown handle ⇒ treat as "other end gone" so wit-bindgen
                // surfaces `ReturnCode::Dropped(0)` rather than an unknown
                // sentinel that would trap.
                env.pushI32(@bitCast(async_canon.packStatus(.dropped, 0))) catch
                    return error.StackOverflow;
                return;
            };

            const registry = TypeRegistry.init(comp_inst.component);
            // Host eager-lowering producers (e.g.
            // `fsDescriptorReadDirectoryP3`) stash the byte stride
            // they actually appended on `elem_size_hint` so we can
            // drain in the correct stride even if `info.type_idx`
            // doesn't resolve to a sized type in this component's
            // registry (cross-instance element types lose their
            // resolution path). (#571.)
            const elem_size: u32 = if (s.elem_size_hint) |hint|
                hint
            else
                streamFutureElemSize(registry, info.type_idx);
            if (elem_size == 0) return error.LowerError;

            // Zero-length read: synchronous no-op completion.
            if (max_count == 0) {
                env.pushI32(@bitCast(async_canon.packStatus(.completed, 0))) catch
                    return error.StackOverflow;
                return;
            }

            // Host-driven streams (#535) get a chance to top up the FIFO
            // before we drain. Loop on `progressed` so a chunky reader can
            // keep filling until `would_block` / `eof`. The loop is bounded
            // by the driver itself — production drivers do a single
            // non-blocking syscall per invocation and return `would_block`
            // on the second pass.
            //
            // #583 B2 — when the driver exposes `on_read_into`, we
            // borrow a slice of guest linmem and let the driver write
            // bytes directly into it. This skips the
            // `stream.buffer.appendSlice` allocation + the second
            // `@memcpy` into guest linmem that the legacy `on_read`
            // path incurs. `comp_inst.writableGuestBytes` validates
            // `guest_ptr + max_bytes ≤ memory.size` synchronously
            // before the host call, and the executor never yields to
            // a `memory.grow` between that check and the driver
            // return — the slice stays valid for the call duration.
            //
            // Three preconditions for the zero-copy fast path; failing
            // any one falls back to the legacy `on_read` path below:
            //   * `elem_size`-aligned `guest_ptr` (the spec's
            //     "alignment permits" gate);
            //   * `max_count * elem_size` doesn't overflow `u32`;
            //   * the resulting byte range is fully inside guest
            //     `memory.size` (the "length permits" gate).
            if (s.buffer.items.len == 0 and !s.write_closed) {
                if (s.host_driver) |drv| zero_copy: {
                    const cb = drv.on_read_into orelse break :zero_copy;
                    if (elem_size > 1 and (guest_ptr % elem_size) != 0) break :zero_copy;
                    const max_bytes_u64: u64 = @as(u64, max_count) * @as(u64, elem_size);
                    if (max_bytes_u64 > std.math.maxInt(u32)) break :zero_copy;
                    const max_bytes: u32 = @intCast(max_bytes_u64);
                    const dst_slice = comp_inst.writableGuestBytes(guest_ptr, max_bytes) orelse
                        break :zero_copy;
                    var iters: u8 = 0;
                    var total_bytes: u32 = 0;
                    while (iters < 32 and total_bytes < max_bytes) : (iters += 1) {
                        const r = cb(drv.context, dst_slice[total_bytes..max_bytes]);
                        switch (r.action) {
                            .progressed => {
                                if (r.bytes_written == 0) break;
                                std.debug.assert(r.bytes_written <= max_bytes - total_bytes);
                                total_bytes += r.bytes_written;
                            },
                            .would_block => break,
                            .eof, .err => {
                                s.write_closed = true;
                                break;
                            },
                        }
                    }
                    if (total_bytes > 0) {
                        const got_count = total_bytes / elem_size;
                        // A sub-element trailing fragment (rare for
                        // `stream<u8>` where `elem_size == 1`) can't
                        // be delivered on this op without crossing
                        // an element boundary in the guest's dst;
                        // stash it in the FIFO so the next read picks
                        // it up. The bytes are already in guest
                        // linmem so we copy them out — alternatively
                        // we could leak the partial element, but the
                        // FIFO stash keeps semantics identical to the
                        // legacy on_read path.
                        const tail = total_bytes - got_count * elem_size;
                        if (tail != 0) {
                            const tail_off = got_count * elem_size;
                            s.buffer.appendSlice(comp_inst.allocator, dst_slice[tail_off..total_bytes]) catch
                                return error.OutOfMemory;
                        }
                        env.pushI32(@bitCast(async_canon.packStatus(.completed, got_count))) catch
                            return error.StackOverflow;
                        return;
                    }
                    // Driver returned `would_block` or `eof` with no
                    // bytes; fall through. If `eof` flipped
                    // `write_closed`, the post-drain branch surfaces
                    // `dropped(0)`. If `would_block`, we'll park.
                }
                if (s.host_driver) |drv| {
                    if (drv.on_read) |cb| {
                        var driver_iters: u8 = 0;
                        while (driver_iters < 32) : (driver_iters += 1) {
                            const action = cb(drv.context, s, comp_inst.allocator);
                            switch (action) {
                                .progressed => {
                                    if (s.buffer.items.len > 0) break;
                                    // Driver claimed progress but appended
                                    // nothing — treat as would_block to
                                    // avoid spinning.
                                    break;
                                },
                                .would_block => break,
                                .eof, .err => {
                                    s.write_closed = true;
                                    break;
                                },
                            }
                        }
                    }
                }
            }

            // Drain buffered data first.
            const buffered_elems: u32 = @intCast(s.buffer.items.len / elem_size);
            if (buffered_elems > 0) {
                const take = @min(max_count, buffered_elems);
                const take_bytes = take * elem_size;
                const dst = comp_inst.writableGuestBytes(guest_ptr, take_bytes) orelse
                    return error.MemoryNotAvailable;
                @memcpy(dst, s.buffer.items[0..take_bytes]);
                // Slide FIFO forward; O(n) is fine until we swap to a ring buffer.
                std.mem.copyForwards(
                    u8,
                    s.buffer.items[0 .. s.buffer.items.len - take_bytes],
                    s.buffer.items[take_bytes..],
                );
                s.buffer.items.len -= take_bytes;
                env.pushI32(@bitCast(async_canon.packStatus(.completed, take))) catch
                    return error.StackOverflow;
                return;
            }

            // Writer dropped and buffer drained → reader observes the
            // drop with zero additional elements transferred.
            if (s.write_closed) {
                env.pushI32(@bitCast(async_canon.packStatus(.dropped, 0))) catch
                    return error.StackOverflow;
                return;
            }

            // Host-attached source (#537): the host installed a
            // synchronous producer callback (e.g.
            // `wasi:cli/stdin.read-via-stream`). Read directly into the
            // guest's destination buffer instead of parking.
            if (s.host_handler) |h| if (h.on_read) |reader| {
                const max_bytes = max_count * elem_size;
                const dst = comp_inst.writableGuestBytes(guest_ptr, max_bytes) orelse
                    return error.MemoryNotAvailable;
                const got = reader(h.ctx, dst);
                if (got < 0) {
                    env.pushI32(@bitCast(async_canon.packStatus(.dropped, 0))) catch
                        return error.StackOverflow;
                    return;
                }
                if (got == 0) {
                    // EOF — flag write_closed so subsequent reads
                    // observe the drop without re-invoking the host.
                    s.write_closed = true;
                    env.pushI32(@bitCast(async_canon.packStatus(.dropped, 0))) catch
                        return error.StackOverflow;
                    return;
                }
                const got_count: u32 = @as(u32, @intCast(got)) / elem_size;
                env.pushI32(@bitCast(async_canon.packStatus(.completed, got_count))) catch
                    return error.StackOverflow;
                return;
            };

            // No data yet — park; signal BLOCKED (post-#541) so the
            // guest's `WaitableOperation` waits for the completion
            // callback rather than treating us as cancelled.
            s.pending_read = .{ .guest_ptr = guest_ptr, .max_count = max_count };
            env.pushI32(@bitCast(async_canon.BLOCKED_STATUS)) catch
                return error.StackOverflow;
        },
        .stream_write => |info| {
            // Stack: (handle, ptr, count) → status. `count` is on top,
            // then `ptr`, then `handle`.
            const count: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const guest_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);

            const s = comp_inst.streams.getPtr(handle) orelse {
                env.pushI32(@bitCast(async_canon.packStatus(.dropped, 0))) catch
                    return error.StackOverflow;
                return;
            };

            // Reader end already dropped → writer observes the drop.
            if (s.read_closed) {
                env.pushI32(@bitCast(async_canon.packStatus(.dropped, 0))) catch
                    return error.StackOverflow;
                return;
            }

            const registry = TypeRegistry.init(comp_inst.component);
            // Mirror the `stream.read t` arm (#571): honour the
            // per-stream `elem_size_hint` so a host-installed driver
            // can pin the byte stride even when the executor's
            // `info.type_idx` doesn't resolve cleanly in this
            // component's `TypeRegistry` (cross-instance element
            // types lose their resolution path).
            const elem_size: u32 = if (s.elem_size_hint) |hint|
                hint
            else
                streamFutureElemSize(registry, info.type_idx);
            if (elem_size == 0) return error.LowerError;

            // Zero-length write: synchronous no-op completion.
            if (count == 0) {
                env.pushI32(@bitCast(async_canon.packStatus(.completed, 0))) catch
                    return error.StackOverflow;
                return;
            }

            const byte_len: u32 = elem_size * count;
            const src = comp_inst.readGuestBytes(guest_ptr, byte_len) orelse
                return error.MemoryNotAvailable;

            // Host-attached sink (#537): the host installed a synchronous
            // drain callback (e.g. `wasi:cli/stdout.write-via-stream`).
            // Forward directly to the sink instead of buffering — this is
            // what keeps WAMR's single-threaded model in sync with guests
            // that use `futures::join!` to interleave a host I/O await
            // with a writer task.
            if (s.host_handler) |h| if (h.on_write) |writer| {
                if (!writer(h.ctx, src)) {
                    env.pushI32(@bitCast(async_canon.packStatus(.dropped, 0))) catch
                        return error.StackOverflow;
                    return;
                }
                env.pushI32(@bitCast(async_canon.packStatus(.completed, count))) catch
                    return error.StackOverflow;
                return;
            };

            // Reader parked first: fulfil it directly (no buffering).
            if (s.pending_read) |pr| {
                const transfer_count = @min(count, pr.max_count);
                const transfer_bytes = transfer_count * elem_size;
                const dst = comp_inst.writableGuestBytes(pr.guest_ptr, transfer_bytes) orelse
                    return error.MemoryNotAvailable;
                @memcpy(dst, src[0..transfer_bytes]);

                // Any tail past `transfer_count` overflows the parked
                // reader's request; buffer it for the next reader.
                if (count > transfer_count) {
                    s.buffer.appendSlice(comp_inst.allocator, src[transfer_bytes..byte_len]) catch
                        return error.OutOfMemory;
                }

                s.pending_read = null;
                if (s.waitable_set) |ws| if (s.read_waitable_idx) |idx|
                    ws.setReady(idx, allocator, async_canon.packStatus(.completed, transfer_count));
                env.pushI32(@bitCast(async_canon.packStatus(.completed, transfer_count))) catch
                    return error.StackOverflow;
                return;
            }

            // Host-driven sink (#535): the guest writes are forwarded
            // straight to the host (e.g. a connected TCP fd) instead of
            // accumulating in the FIFO. We only invoke the driver after
            // confirming there's no parked reader, since the reader path
            // is the canonical wakeup mechanism for in-component pairs.
            //
            // #583 B2 follow-up — when the driver exposes
            // `on_write_from`, we prefer it over the legacy `on_write`.
            // The bounds check has already happened above
            // (`readGuestBytes(guest_ptr, byte_len)` validated
            // `ptr + len <= memory.size`), and the executor never
            // yields to a `memory.grow` between that check and the
            // driver return — the borrowed slice stays valid for the
            // call duration. The thinner signature drops the unused
            // `*AsyncStream` / `Allocator` parameters; semantics are
            // otherwise identical to `on_write`.
            if (s.host_driver) |drv| {
                if (drv.on_write_from) |cb| {
                    const action = cb(drv.context, src);
                    switch (action) {
                        .progressed => {
                            env.pushI32(@bitCast(async_canon.packStatus(.completed, count))) catch
                                return error.StackOverflow;
                            return;
                        },
                        .eof, .err => {
                            s.read_closed = true;
                            env.pushI32(@bitCast(async_canon.packStatus(.dropped, 0))) catch
                                return error.StackOverflow;
                            return;
                        },
                        .would_block => {
                            // Fall through to FIFO buffering — a later
                            // driver invocation (or `cancel_write`) can
                            // drain it.
                        },
                    }
                } else if (drv.on_write) |cb| {
                    const action = cb(drv.context, s, src, comp_inst.allocator);
                    switch (action) {
                        .progressed => {
                            env.pushI32(@bitCast(async_canon.packStatus(.completed, count))) catch
                                return error.StackOverflow;
                            return;
                        },
                        .eof, .err => {
                            s.read_closed = true;
                            env.pushI32(@bitCast(async_canon.packStatus(.dropped, 0))) catch
                                return error.StackOverflow;
                            return;
                        },
                        .would_block => {
                            // Fall through to FIFO buffering — a later
                            // driver invocation (or `cancel_write`) can
                            // drain it.
                        },
                    }
                }
            }

            // No reader yet — append to FIFO.
            s.buffer.appendSlice(comp_inst.allocator, src) catch
                return error.OutOfMemory;
            if (s.waitable_set) |ws| if (s.read_waitable_idx) |idx|
                ws.setReady(idx, allocator, async_canon.packStatus(.completed, count));
            env.pushI32(@bitCast(async_canon.packStatus(.completed, count))) catch
                return error.StackOverflow;
        },
        .stream_cancel_read => |info| {
            _ = info;
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const s = comp_inst.streams.getPtr(handle) orelse {
                env.pushI32(@bitCast(async_canon.packStatus(.cancelled, 0))) catch
                    return error.StackOverflow;
                return;
            };
            s.pending_read = null;
            env.pushI32(@bitCast(async_canon.packStatus(.cancelled, 0))) catch
                return error.StackOverflow;
        },
        .stream_cancel_write => |info| {
            _ = info;
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const s = comp_inst.streams.getPtr(handle) orelse {
                env.pushI32(@bitCast(async_canon.packStatus(.cancelled, 0))) catch
                    return error.StackOverflow;
                return;
            };
            s.pending_write = null;
            env.pushI32(@bitCast(async_canon.packStatus(.cancelled, 0))) catch
                return error.StackOverflow;
        },
        .stream_drop_readable => |info| {
            _ = info;
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            if (comp_inst.streams.getPtr(handle)) |s| {
                s.read_closed = true;
                // Wake a parked writer so it can observe CANCELLED.
                if (s.waitable_set) |ws| if (s.write_waitable_idx) |idx|
                    ws.setReady(idx, allocator, async_canon.packStatus(.dropped, 0));
                if (s.read_closed and s.write_closed) {
                    if (s.host_handler) |h| if (h.on_destroy) |cb|
                        cb(h.ctx);
                    s.deinit(comp_inst.allocator);
                    _ = comp_inst.streams.remove(handle);
                }
            }
        },
        .stream_drop_writable => |info| {
            _ = info;
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            if (comp_inst.streams.getPtr(handle)) |s| {
                s.write_closed = true;
                // Notify any host-attached sink that the guest has
                // closed its writer end (#537). The host handler is
                // responsible for settling its companion future.
                if (s.host_handler) |h| if (h.on_drop_writable) |cb| {
                    cb(h.ctx);
                    // A host-attached sink only needs the read end open
                    // long enough to receive synchronous `stream.write`
                    // forwards. After the writer drops, the host is
                    // done — close the read end so the stream entry is
                    // freed (and `on_destroy` reclaims `ctx`).
                    s.read_closed = true;
                };
                // Wake a parked reader so it can observe CANCELLED.
                if (s.waitable_set) |ws| if (s.read_waitable_idx) |idx|
                    ws.setReady(idx, allocator, async_canon.packStatus(.dropped, 0));
                if (s.read_closed and s.write_closed) {
                    if (s.host_handler) |h| if (h.on_destroy) |cb|
                        cb(h.ctx);
                    s.deinit(comp_inst.allocator);
                    _ = comp_inst.streams.remove(handle);
                }
            }
        },

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
                env.pushI32(@bitCast(async_canon.packStatus(.dropped, 0))) catch
                    return error.StackOverflow;
                return;
            };

            // Unit-type fast-path (#487): when a writer marks the future
            // `.ready` with no payload (e.g. `future<()>` or
            // `future<result<_,_>>` with absent payload), `future.read`
            // succeeds immediately with zero bytes copied. Must precede
            // the writer-dropped check below so a host-completed unit
            // future is not mis-reported as cancelled.
            if (fut.state == .ready and fut.payload == null) {
                env.pushI32(@bitCast(async_canon.packStatus(.completed, 0))) catch
                    return error.StackOverflow;
                return;
            }

            // Writer dropped without ever delivering a value → DROPPED
            // (post-#541 spec).
            if (fut.write_closed and fut.payload == null) {
                env.pushI32(@bitCast(async_canon.packStatus(.dropped, 0))) catch
                    return error.StackOverflow;
                return;
            }

            // Unit-type (`future<()>`) fast-path (#483): a `wait-for` /
            // `wait-until` timer fired and set `state = .ready` with no
            // payload — there are zero bytes to copy. Distinct from the
            // writer-dropped case above because `write_closed`
            // is false here. `count = 1` reports one unit element.
            if (fut.state == .ready and fut.payload == null and !fut.write_closed) {
                fut.state = .closed;
                if (fut.waitable_set) |ws| if (fut.read_waitable_idx) |idx|
                    ws.setReady(idx, allocator, async_canon.packStatus(.completed, 0));
                env.pushI32(@bitCast(async_canon.packStatus(.completed, 0))) catch
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
                    ws.setReady(idx, allocator, async_canon.packStatus(.completed, 0));
                env.pushI32(@bitCast(async_canon.packStatus(.completed, 0))) catch
                    return error.StackOverflow;
                return;
            }

            // No writer yet — park. `info.type_idx` is captured into the
            // table entry at `future.new` time; we only need the guest
            // ptr here.
            _ = info;
            fut.pending_read = .{ .guest_ptr = guest_ptr };
            env.pushI32(@bitCast(async_canon.BLOCKED_STATUS)) catch
                return error.StackOverflow;
        },
        .future_write => |info| {
            // Stack: (handle, ptr) → status.
            const guest_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);

            const fut = comp_inst.futures.getPtr(handle) orelse {
                env.pushI32(@bitCast(async_canon.packStatus(.dropped, 0))) catch
                    return error.StackOverflow;
                return;
            };

            // Reader already dropped → writer observes the drop.
            if (fut.read_closed) {
                env.pushI32(@bitCast(async_canon.packStatus(.dropped, 0))) catch
                    return error.StackOverflow;
                return;
            }

            // Already buffered (double-write); reject. Spec forbids two
            // writes on a one-shot future — surface as DROPPED so the
            // guest's WaitableOperation observes the rejection.
            if (fut.payload != null) {
                env.pushI32(@bitCast(async_canon.packStatus(.dropped, 0))) catch
                    return error.StackOverflow;
                return;
            }

            const registry = TypeRegistry.init(comp_inst.component);
            const elem_size = streamFutureElemSize(registry, info.type_idx);
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
                    ws.setReady(idx, allocator, async_canon.packStatus(.completed, 0));
                env.pushI32(@bitCast(async_canon.packStatus(.completed, 0))) catch
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
                ws.setReady(idx, allocator, async_canon.packStatus(.completed, 0));
            env.pushI32(@bitCast(async_canon.packStatus(.completed, 0))) catch
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
            // surface COMPLETED so the caller still observes the transfer.
            if (fut.state == .ready and fut.payload == null) {
                env.pushI32(@bitCast(async_canon.packStatus(.completed, 0))) catch
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
            // write completed before cancel; report COMPLETED.
            if (fut.state == .ready and fut.payload == null and fut.pending_read == null) {
                env.pushI32(@bitCast(async_canon.packStatus(.completed, 0))) catch
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
                    ws.setReady(idx, allocator, async_canon.packStatus(.dropped, 0));
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
                    ws.setReady(idx, allocator, async_canon.packStatus(.dropped, 0));
                if (fut.read_closed and fut.write_closed) {
                    fut.deinit(comp_inst.allocator);
                    _ = comp_inst.futures.remove(handle);
                }
            }
        },

        // ── error-context ───────────────────────────────────────────────
        .error_context_new => |info| {
            _ = info; // opts.memory_idx / string_encoding ignored — we
            // use the canonical-memory shim that every other async-canon
            // arm uses (`readGuestBytes` / `hostAllocAndWrite`). Making
            // this arm opts-aware is a follow-up across the whole
            // dispatchAsyncCanon switch.
            const len: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);

            const copy: []u8 = blk: {
                if (len == 0) break :blk comp_inst.allocator.alloc(u8, 0) catch
                    return error.OutOfMemory;
                const bytes = comp_inst.readGuestBytes(ptr, len) orelse
                    return error.MemoryNotAvailable;
                break :blk comp_inst.allocator.dupe(u8, bytes) catch
                    return error.OutOfMemory;
            };

            const handle = comp_inst.allocAsyncHandle();
            comp_inst.error_contexts.put(comp_inst.allocator, handle, copy) catch {
                comp_inst.allocator.free(copy);
                return error.OutOfMemory;
            };
            env.pushI32(@bitCast(handle)) catch return error.StackOverflow;
        },
        .error_context_debug_message => |info| {
            _ = info; // opts.realloc_idx implicit via hostAllocAndWrite
            // WIT signature `func(borrow<error-context>) -> string`
            // lowers to `[i32] -> [i32 i32]`: pop the handle, push the
            // (ptr, len) of the debug-message materialized into guest
            // memory.
            const handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);

            const stored = comp_inst.error_contexts.get(handle) orelse {
                // Unknown handle (e.g. borrow already dropped on the
                // other end of a misbehaving guest): return an empty
                // string rather than trapping.
                env.pushI32(0) catch return error.StackOverflow;
                env.pushI32(0) catch return error.StackOverflow;
                return;
            };

            if (stored.len == 0) {
                env.pushI32(0) catch return error.StackOverflow;
                env.pushI32(0) catch return error.StackOverflow;
                return;
            }

            const guest_ptr = comp_inst.hostAllocAndWrite(stored) orelse
                return error.OutOfMemory;
            env.pushI32(@bitCast(guest_ptr)) catch return error.StackOverflow;
            env.pushI32(@bitCast(@as(u32, @intCast(stored.len)))) catch
                return error.StackOverflow;
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
            // wait/poll surface the oldest ready item registered in the
            // waitable-set as `(event-kind: i32, payload: (handle: u32, code: u32))`:
            // the event-kind is returned on the operand stack and the
            // (handle, code) pair is written at the guest out-pointer
            // popped off the stack. Mirrors the canonical-abi.py
            // `canon_waitable_set_{wait,poll}` semantics so wit-bindgen's
            // wakeup loop can decode the result via `EventCode + ReturnCode`.
            //
            // For the `wait` variant we additionally drive the host
            // async-event driver (typically
            // `WasiCliAdapter.driveAsyncEvents`, #551) so the host can
            // advance its monotonic clock and settle any pending
            // `wasi:clocks` timer-futures before we consult the queue —
            // otherwise a guest that wakes only on host-produced
            // subtask events (`wait-for` / `wait-until`) would spin on
            // EVENT_NONE forever.
            const out_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const ws_handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);

            const ws = comp_inst.waitable_sets.getPtr(ws_handle) orelse {
                env.pushI32(0) catch return error.StackOverflow;
                return;
            };

            const is_wait = op == .waitable_set_wait;
            // Bounded drive loop. Each iteration first asks the host
            // async-event driver to advance time / drain host I/O,
            // then re-checks `popReadyEvent`. Caps total iterations
            // to keep the runtime responsive when the guest has
            // non-host work (e.g. `futures::join!` peer rendezvous
            // — the cli-stdio-roundtrip path).
            const max_iters: usize = if (is_wait) 256 else 1;
            var iter: usize = 0;
            while (iter < max_iters) : (iter += 1) {
                if (ws.popReadyEvent()) |item| {
                    if (out_ptr != 0) {
                        if (comp_inst.writableGuestBytes(out_ptr, 8)) |bytes| {
                            std.mem.writeInt(u32, bytes[0..4], item.handle, .little);
                            std.mem.writeInt(u32, bytes[4..8], item.code, .little);
                        }
                    }
                    env.pushI32(@bitCast(async_canon.eventCodeForKind(item.kind))) catch
                        return error.StackOverflow;
                    return;
                }
                if (!is_wait) break;
                const driver = comp_inst.async_event_driver orelse break;
                // ~10ms-per-iteration budget: short enough to stay
                // responsive, long enough to coalesce many timers.
                // The driver is allowed to ignore the hint when it
                // has due work to deliver immediately.
                _ = driver(comp_inst.async_event_driver_ctx, comp_inst, 10 * std.time.ns_per_ms, allocator);
            }

            // No ready waitable — `none` event-code. Caller (wit-bindgen
            // reactor) reschedules.
            env.pushI32(@intFromEnum(async_canon.EventCode.none)) catch
                return error.StackOverflow;
        },

        .waitable_join => {
            // `canon waitable.join` — canonical-abi.py
            // `canon_waitable_join(wi, si)` declares params in
            // `(waitable, set)` order, so the runtime stack at call
            // time is (bottom→top) `[waitable, set]`. We therefore
            // pop **set** first (top) and **waitable** second
            // (under). A zero `set` means "remove from any current
            // set" — we currently no-op that because removal isn't
            // observable until the waitable becomes ready, and the
            // only producer of a zero-set join is wit-bindgen's drop
            // path which shortly drops the waitable entirely.
            const ws_handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
            const waitable_handle: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);

            if (ws_handle == 0) return;
            const ws = comp_inst.waitable_sets.getPtr(ws_handle) orelse return;

            // Determine kind by inspecting which end of the
            // future/stream is currently parked. The guest calls
            // `waitable.join` immediately after the corresponding
            // `{future,stream}.{read,write}` returned `BLOCKED`, so
            // exactly one of `pending_read` / `pending_write` is set
            // (futures only park readers in our impl; streams can
            // park either end). Fall back to `_read` for futures and
            // `_write` for streams — the common case is a guest
            // awaiting a host-produced future end (see #537
            // `write-via-stream` / `read-via-stream`).
            if (comp_inst.futures.getPtr(waitable_handle)) |fut| {
                if (fut.waitable_set != null) return; // already joined
                // #551: a future minted by a canon-lower-of-async-func
                // host adapter (`wasi:clocks` `wait-for` / `wait-until`)
                // carries a `subtask_managed` flag — wire it as a
                // `.subtask` waitable so `waitable-set.wait` surfaces
                // EVENT_SUBTASK with a STATUS_* code (matching the
                // wit-bindgen `Subtask` runtime decoder), not as
                // `.future_read` (which uses the future/stream
                // `ReturnCode` decoder and would mis-interpret
                // `STATUS_RETURNED=2` as `Cancelled(0)`).
                if (fut.subtask_managed) {
                    const idx = ws.register(.{ .kind = .subtask, .handle = waitable_handle }, allocator) catch
                        return error.OutOfMemory;
                    fut.waitable_set = ws;
                    fut.read_waitable_idx = idx;
                    // If the timer already fired before the guest
                    // could join, surface the settled state at join
                    // time so the next `waitable-set.wait` delivers
                    // EVENT_SUBTASK rather than spinning on NONE.
                    if (fut.state == .closed and fut.write_closed) {
                        ws.setReady(idx, allocator, STATUS_STARTED_CANCELLED);
                    } else if (fut.state == .ready or fut.state == .closed) {
                        ws.setReady(idx, allocator, STATUS_RETURNED);
                    }
                    return;
                }

                // Futures default to `future_read` because their only
                // blocking arm is the reader (`future.read` parks via
                // `pending_read`). A `pending_write` slot doesn't exist
                // in our model — `future.write` always completes
                // synchronously. We still treat `read_closed` as a
                // hint that the join is for the write end so the
                // dropped-peer wake fires the right slot.
                const kind: async_mod.WaitableSet.WaitableItem.Kind =
                    if (fut.read_closed) .future_write else .future_read;
                const idx = ws.register(.{ .kind = kind, .handle = waitable_handle }, allocator) catch
                    return error.OutOfMemory;
                fut.waitable_set = ws;
                if (kind == .future_read) {
                    fut.read_waitable_idx = idx;
                } else {
                    fut.write_waitable_idx = idx;
                }
                // If the corresponding end is already settled (the
                // peer fired its drop / write before the join arrived
                // — common when stdout's `on_drop_writable` runs
                // synchronously inside the same `futures::join!` arm),
                // mark the slot ready immediately so the very next
                // `waitable-set.{wait,poll}` surfaces the event.
                if (kind == .future_read) {
                    if (fut.payload != null or (fut.state == .ready and !fut.write_closed)) {
                        ws.setReady(idx, allocator, async_canon.packStatus(.completed, 0));
                    } else if (fut.write_closed and fut.payload == null) {
                        ws.setReady(idx, allocator, async_canon.packStatus(.dropped, 0));
                    }
                } else { // future_write
                    if (fut.read_closed) {
                        ws.setReady(idx, allocator, async_canon.packStatus(.dropped, 0));
                    }
                }
                return;
            }

            if (comp_inst.streams.getPtr(waitable_handle)) |s| {
                if (s.waitable_set != null) return; // already joined
                const kind: async_mod.WaitableSet.WaitableItem.Kind =
                    if (s.pending_read != null) .stream_read else .stream_write;
                const idx = ws.register(.{ .kind = kind, .handle = waitable_handle }, allocator) catch
                    return error.OutOfMemory;
                s.waitable_set = ws;
                if (kind == .stream_read) {
                    s.read_waitable_idx = idx;
                    // Buffered bytes available or peer closed →
                    // surface readiness immediately.
                    if (s.buffer.items.len > 0) {
                        ws.setReady(idx, allocator, async_canon.packStatus(.completed, 0));
                    } else if (s.write_closed) {
                        ws.setReady(idx, allocator, async_canon.packStatus(.dropped, 0));
                    }
                } else {
                    s.write_waitable_idx = idx;
                    if (s.read_closed) {
                        ws.setReady(idx, allocator, async_canon.packStatus(.dropped, 0));
                    }
                }
                return;
            }

            // Unknown waitable handle — silently no-op. Spec says trap,
            // but the conformance suite occasionally joins a
            // just-dropped handle on the cancellation path; matching
            // wasmtime's tolerant behaviour avoids spurious traps.
        },
    }
}

/// On the canon-lower-of-async-func path (#551) `componentTrampoline` packs
/// the host's future handle as `(handle << 4) | STATUS` and pushes a single
/// i32 status word — the lifted result is delivered later when the timer
/// fires (`WasiCliAdapter.completeDueTimerFutures`).

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

    // Publish the active TaskManager on the owning instance so the
    // canon-builtin host trampolines (`canonBuiltinTrampoline`)
    // installed during instantiation can dispatch into the right task
    // state. Restored to its prior value on return to support nested
    // async dispatches. (#520)
    const owner_for_tm: *ComponentInstance = @constCast(owner_for_type);
    const saved_tm = owner_for_tm.current_task_manager;
    owner_for_tm.current_task_manager = task_manager;
    defer owner_for_tm.current_task_manager = saved_tm;

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

test "LowerOptions: async opt flips is_async (#551 canon-lower-of-async-func)" {
    const opts_async = [_]ctypes.CanonOpt{ .{ .memory = 0 }, .async_lift };
    const lo_async = LowerOptions.fromOpts(&opts_async);
    try std.testing.expectEqual(true, lo_async.is_async);

    const opts_sync = [_]ctypes.CanonOpt{ .{ .memory = 0 } };
    const lo_sync = LowerOptions.fromOpts(&opts_sync);
    try std.testing.expectEqual(false, lo_sync.is_async);

    // `callback` opt on the lower side is accepted as a no-op (Binary.md
    // canon opt vec is shared with canon.lift); it must NOT flip is_async.
    const opts_cb = [_]ctypes.CanonOpt{ .{ .callback = 3 } };
    const lo_cb = LowerOptions.fromOpts(&opts_cb);
    try std.testing.expectEqual(false, lo_cb.is_async);
}

test "packAsyncLowerStatus: pending future → (handle << 4) | STATUS_STARTED (#551)" {
    const testing = std.testing;
    var comp = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},         .imports = &.{},
        .exports = &.{},
    };
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();

    // Allocate a pending future and verify the packed status carries the
    // handle in the high bits + STATUS_STARTED (=1) in the low nibble.
    const fh = inst.allocAsyncHandle();
    try inst.futures.put(testing.allocator, fh, .{ .elem_type_idx = 0, .state = .pending });
    const packed_status = packAsyncLowerStatus(inst, fh);
    try testing.expectEqual(STATUS_STARTED, packed_status & 0xf);
    try testing.expectEqual(fh, packed_status >> 4);

    // A ready future collapses to STATUS_RETURNED with no waitable
    // handle in the high bits — the guest reads zero results and exits
    // the WaitableOperation immediately.
    inst.futures.getPtr(fh).?.state = .ready;
    try testing.expectEqual(STATUS_RETURNED, packAsyncLowerStatus(inst, fh));

    // Unknown handles degrade to STATUS_RETURNED rather than trap, so a
    // host fn that forgot to populate the phantom slot doesn't poison
    // the call.
    try testing.expectEqual(STATUS_RETURNED, packAsyncLowerStatus(inst, 0));
    try testing.expectEqual(STATUS_RETURNED, packAsyncLowerStatus(inst, 99));
}

test "dispatchCanonBuiltin: waitable.join wires a subtask_managed future as .subtask (#551)" {
    // Canon-lower-of-async-func returns `(handle << 4) | STATUS_STARTED`;
    // the guest follows up with `[waitable-join]`. Verify the join
    // hooks the timer-future into the WaitableSet as a `.subtask`
    // waitable (not `.future_read`) so the wait/poll arm surfaces
    // EVENT_SUBTASK with a `STATUS_*` payload2 the wit-bindgen
    // `Subtask` decoder expects.
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

    const fh = inst.allocAsyncHandle();
    try inst.futures.put(testing.allocator, fh, .{ .elem_type_idx = 0, .state = .pending, .subtask_managed = true });
    const ws_handle = inst.allocAsyncHandle();
    try inst.waitable_sets.put(testing.allocator, ws_handle, .{});

    // Stack: (waitable, set) per canonical-abi.py order.
    try env.pushI32(@bitCast(fh));
    try env.pushI32(@bitCast(ws_handle));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .waitable_join },
        env,
        null,
        testing.allocator,
    );

    const ws = inst.waitable_sets.getPtr(ws_handle).?;
    try testing.expectEqual(@as(usize, 1), ws.items.items.len);
    try testing.expectEqual(fh, ws.items.items[0].handle);
    try testing.expectEqual(
        @import("async.zig").WaitableSet.WaitableItem.Kind.subtask,
        ws.items.items[0].kind,
    );
}

test "dispatchCanonBuiltin: waitable.join on a non-subtask future keeps the future_read shape (#551)" {
    // Cli-stdio-roundtrip regression guard: a `future.read`-driven
    // (non-subtask_managed) future must continue to be registered as
    // `.future_read` so `waitable-set.wait` surfaces a
    // future/stream `ReturnCode`-shaped payload2. Wiring it as
    // `.subtask` would mis-decode `STATUS_RETURNED=2` as
    // `ReturnCode::Cancelled(0)` per the wit-bindgen ≥ 0.53
    // future_support runtime, panicking the guest.
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

    const fh = inst.allocAsyncHandle();
    // subtask_managed left at default `false` — plain future.read shape.
    try inst.futures.put(testing.allocator, fh, .{ .elem_type_idx = 0, .state = .pending });
    const ws_handle = inst.allocAsyncHandle();
    try inst.waitable_sets.put(testing.allocator, ws_handle, .{});

    try env.pushI32(@bitCast(fh));
    try env.pushI32(@bitCast(ws_handle));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .waitable_join },
        env,
        null,
        testing.allocator,
    );

    const ws = inst.waitable_sets.getPtr(ws_handle).?;
    try testing.expectEqual(@as(usize, 1), ws.items.items.len);
    try testing.expectEqual(
        @import("async.zig").WaitableSet.WaitableItem.Kind.future_read,
        ws.items.items[0].kind,
    );
}

test "dispatchCanonBuiltin: waitable.join late-arrival on an already-fired subtask surfaces immediately (#551)" {
    // Regression guard: if the host completed the timer between
    // `canon.lower (async)` returning STATUS_STARTED and the guest
    // calling `[waitable-join]`, the join must observe the readiness
    // synchronously so the very next `waitable-set.wait` delivers
    // EVENT_SUBTASK rather than spinning on NONE.
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

    const fh = inst.allocAsyncHandle();
    try inst.futures.put(testing.allocator, fh, .{ .elem_type_idx = 0, .state = .ready, .subtask_managed = true });
    const ws_handle = inst.allocAsyncHandle();
    try inst.waitable_sets.put(testing.allocator, ws_handle, .{});

    try env.pushI32(@bitCast(fh));
    try env.pushI32(@bitCast(ws_handle));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .waitable_join },
        env,
        null,
        testing.allocator,
    );

    const ws = inst.waitable_sets.getPtr(ws_handle).?;
    try testing.expectEqual(@as(usize, 1), ws.items.items.len);
    try testing.expect(ws.items.items[0].ready);
    try testing.expectEqual(STATUS_RETURNED, ws.items.items[0].code);
}

test "dispatchCanonBuiltin: waitable.join late-arrival on a cancelled subtask carries STATUS_STARTED_CANCELLED (#551)" {
    // task.cancel during a clock wait flips the timer-future to
    // `.closed` + `write_closed=true`. A subsequent `[waitable-join]`
    // by the not-yet-aware guest still needs to surface the cancel
    // disposition so the wit-bindgen `Subtask` decoder transitions to
    // `STARTED_CANCELLED`.
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

    const fh = inst.allocAsyncHandle();
    try inst.futures.put(testing.allocator, fh, .{
        .elem_type_idx = 0,
        .state = .closed,
        .write_closed = true,
        .subtask_managed = true,
    });
    const ws_handle = inst.allocAsyncHandle();
    try inst.waitable_sets.put(testing.allocator, ws_handle, .{});

    try env.pushI32(@bitCast(fh));
    try env.pushI32(@bitCast(ws_handle));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .waitable_join },
        env,
        null,
        testing.allocator,
    );

    const ws = inst.waitable_sets.getPtr(ws_handle).?;
    try testing.expectEqual(@as(usize, 1), ws.items.items.len);
    try testing.expect(ws.items.items[0].ready);
    try testing.expectEqual(STATUS_STARTED_CANCELLED, ws.items.items[0].code);
}

test "coreFlatSlotType: TypeDef indirection to single-slot result reifies to .i32 (#683)" {
    // type 0: result<unit, unit> — discriminant only, flattens to 1 slot.
    const result_unit = ctypes.TypeDef{ .result = .{ .ok = null, .err = null } };
    const reg = TypeRegistry.fromTypes(&.{result_unit});
    const slot = try coreFlatSlotType(.{ .type_idx = 0 }, reg);
    try std.testing.expectEqual(core_types.ValType.i32, slot);
}

test "coreFlatSlotType: TypeDef indirection to multi-slot variant traps AotPathUnsupported (#683)" {
    // type 0: variant { a, b(s64) } — 1 disc + 1 payload = 2 slots.
    const variant_two_slot = ctypes.TypeDef{ .variant = .{ .cases = &.{
        .{ .name = "a", .type = null },
        .{ .name = "b", .type = .s64 },
    } } };
    const reg = TypeRegistry.fromTypes(&.{variant_two_slot});
    const got = coreFlatSlotType(.{ .type_idx = 0 }, reg);
    try std.testing.expectError(error.AotPathUnsupported, got);
}

test "coreFlatSlotType: TypeDef indirection to single-field record reifies to .i32 (#683)" {
    // type 0: record { x: u32 } — flattens to 1 slot.
    const record_one_slot = ctypes.TypeDef{ .record = .{ .fields = &.{
        .{ .name = "x", .type = .u32 },
    } } };
    const reg = TypeRegistry.fromTypes(&.{record_one_slot});
    const slot = try coreFlatSlotType(.{ .type_idx = 0 }, reg);
    try std.testing.expectEqual(core_types.ValType.i32, slot);
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

// ── dispatchCanonBuiltin: task.cancel (Binary.md tag 0x05, issue #488) ──

test "dispatchCanonBuiltin: task.cancel without task manager traps with FunctionNotFound" {
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
        .{ .async_canon = .task_cancel },
        env,
        null,
        testing.allocator,
    );
    try testing.expectError(error.FunctionNotFound, result);
    // Spec `[] -> []`: nothing popped, nothing pushed even on the trap path.
    try testing.expectEqual(@as(u32, 0), env.sp);
}

test "dispatchCanonBuiltin: task.cancel without current_task traps with FunctionNotFound" {
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

    // TaskManager present but `current_task` is null (default).
    var tm = async_mod.TaskManager{};
    defer tm.deinit(testing.allocator);

    const result = dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .task_cancel },
        env,
        &tm,
        testing.allocator,
    );
    try testing.expectError(error.FunctionNotFound, result);
    try testing.expectEqual(@as(u32, 0), env.sp);
}

test "dispatchCanonBuiltin: task.cancel flips current task to .cancelled" {
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
    const h = try tm.createTask(testing.allocator);
    tm.current_task = h;
    try testing.expect(tm.getState(h).? != .cancelled);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .task_cancel },
        env,
        &tm,
        testing.allocator,
    );
    try testing.expectEqual(async_mod.TaskState.cancelled, tm.getState(h).?);
    // Core signature is `[] -> []`: nothing pushed.
    try testing.expectEqual(@as(u32, 0), env.sp);
}

test "dispatchCanonBuiltin: task.cancel is idempotent" {
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
    const h = try tm.createTask(testing.allocator);
    tm.current_task = h;

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .task_cancel },
        env,
        &tm,
        testing.allocator,
    );
    try testing.expectEqual(async_mod.TaskState.cancelled, tm.getState(h).?);

    // Second call must be a no-op: state stays `.cancelled`, no trap,
    // no stack churn.
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .task_cancel },
        env,
        &tm,
        testing.allocator,
    );
    try testing.expectEqual(async_mod.TaskState.cancelled, tm.getState(h).?);
    try testing.expectEqual(@as(u32, 0), env.sp);
}

test "dispatchCanonBuiltin: task.cancel + cancellable task.yield observes cancellation" {
    // End-to-end propagation flow: a host invocation of `task.cancel`
    // followed by a guest `task.yield cancel?=1` must surface the
    // cancellation as discriminant `1`. Mirrors the existing
    // `task.yield observes cancellation via TaskManager` test, but
    // exercises the new `task.cancel` dispatch path instead of the
    // direct `async_canon.asyncCancel` helper.
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
    const h = try tm.createTask(testing.allocator);
    tm.current_task = h;

    // Cancel the currently-executing task via the new dispatch arm.
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .task_cancel },
        env,
        &tm,
        testing.allocator,
    );
    try testing.expectEqual(async_mod.TaskState.cancelled, tm.getState(h).?);

    // Non-cancellable yield: opaque, always reports resumed=0 even
    // when the task has been cancelled.
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
    try testing.expectEqual(async_canon.packStatus(.completed, 0), write_status);
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
    try testing.expectEqual(async_canon.packStatus(.completed, 0), read_status);

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
    try testing.expectEqual(async_canon.BLOCKED_STATUS, read_status);
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
    try testing.expectEqual(async_canon.packStatus(.completed, 0), write_status);

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

test "future.drop-writable while reader parked: subsequent read is DROPPED" {
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
    try testing.expectEqual(async_canon.packStatus(.dropped, 0), status);
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

// ── #478 sub-PR 3b: stream.read/write rendezvous tests ──────────────────────
//
// Mirror of the future suite above. Each test wires a minimal
// `ComponentInstance` whose component-type-index space carries a single
// `(type u32)` entry, so `stream.new t=0` names a `stream<u32>` (4-byte
// elements). `enableTestMem` provides a 4 KiB backing buffer for the
// lift/lower memcpy on each side of the rendezvous.

fn newStreamU32Inst(testing_allocator: std.mem.Allocator) !*instance_mod.ComponentInstance {
    // Static lifetimes — the `Component` and its types outlive the
    // instance, so we leak them deliberately. Tests run in their own
    // process; the test allocator releases bookkeeping on exit.
    const StreamTypeFixture = struct {
        var types_array = [_]ctypes.TypeDef{.{ .val = .u32 }};
        var comp: ctypes.Component = .{
            .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
            .components = &.{},       .instances = &.{},      .aliases = &.{},
            .types = &.{},            .canons = &.{},         .imports = &.{},
            .exports = &.{},
        };
    };
    StreamTypeFixture.comp.types = &StreamTypeFixture.types_array;
    const inst = try instance_mod.instantiate(&StreamTypeFixture.comp, testing_allocator);
    try inst.enableTestMem(testing_allocator, 4096);
    return inst;
}

fn destroyStreamInst(inst: *instance_mod.ComponentInstance) void {
    inst.disableTestMem();
    inst.deinit();
}

test "stream.new: returns packed read|write handles and records elem_type_idx" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newStreamU32Inst(testing.allocator);
    defer destroyStreamInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
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
    try testing.expectEqual(@as(u32, 1), inst.streams.count());

    const s = inst.streams.getPtr(r_idx).?;
    try testing.expectEqual(@as(u32, 0), s.elem_type_idx);
    try testing.expectEqual(@as(usize, 0), s.buffer.items.len);
    try testing.expect(s.pending_read == null);
}

test "stream.write then stream.read: round-trips 3×u32" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newStreamU32Inst(testing.allocator);
    defer destroyStreamInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    // Stage three u32s in test memory at offset 0; reader buffer at 64.
    const src_ptr: u32 = 0;
    const dst_ptr: u32 = 64;
    const src_bytes = inst.writableGuestBytes(src_ptr, 12).?;
    std.mem.writeInt(u32, src_bytes[0..4], 0x1111_1111, .little);
    std.mem.writeInt(u32, src_bytes[4..8], 0x2222_2222, .little);
    std.mem.writeInt(u32, src_bytes[8..12], 0x3333_3333, .little);

    // stream.write with count=3 — no reader parked yet, buffers all 12 bytes.
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(src_ptr));
    try env.pushI32(@bitCast(@as(u32, 3)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_write = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const write_status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.completed, 3), write_status);
    try testing.expectEqual(@as(usize, 12), inst.streams.getPtr(handle).?.buffer.items.len);

    // stream.read with max_count=3 — drains the FIFO.
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(dst_ptr));
    try env.pushI32(@bitCast(@as(u32, 3)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const read_status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.completed, 3), read_status);

    const dst_bytes = inst.writableGuestBytes(dst_ptr, 12).?;
    try testing.expectEqual(@as(u32, 0x1111_1111), std.mem.readInt(u32, dst_bytes[0..4], .little));
    try testing.expectEqual(@as(u32, 0x2222_2222), std.mem.readInt(u32, dst_bytes[4..8], .little));
    try testing.expectEqual(@as(u32, 0x3333_3333), std.mem.readInt(u32, dst_bytes[8..12], .little));
    try testing.expectEqual(@as(usize, 0), inst.streams.getPtr(handle).?.buffer.items.len);
}

test "stream.read parks; stream.write delivers and wakes waitable" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newStreamU32Inst(testing.allocator);
    defer destroyStreamInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    // Wire a waitable set + register the read side. This is the manual
    // plumbing the eventual `waitable.join` arm will perform.
    var ws = async_mod.WaitableSet{};
    defer ws.deinit(testing.allocator);
    {
        const s = inst.streams.getPtr(handle).?;
        s.waitable_set = &ws;
        const idx = try ws.register(.{ .kind = .stream_read, .handle = handle }, testing.allocator);
        s.read_waitable_idx = idx;
    }

    const src_ptr: u32 = 0;
    const dst_ptr: u32 = 32;
    const src_bytes = inst.writableGuestBytes(src_ptr, 8).?;
    std.mem.writeInt(u32, src_bytes[0..4], 0xAAAA_AAAA, .little);
    std.mem.writeInt(u32, src_bytes[4..8], 0xBBBB_BBBB, .little);

    // Reader arrives first with max_count=2 — must park with STARTING.
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(dst_ptr));
    try env.pushI32(@bitCast(@as(u32, 2)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const read_status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.BLOCKED_STATUS, read_status);
    try testing.expectEqual(@as(u32, dst_ptr), inst.streams.getPtr(handle).?.pending_read.?.guest_ptr);
    try testing.expectEqual(@as(u32, 2), inst.streams.getPtr(handle).?.pending_read.?.max_count);

    // Writer arrives with count=2 — copies straight into the parked dst.
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(src_ptr));
    try env.pushI32(@bitCast(@as(u32, 2)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_write = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const write_status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.completed, 2), write_status);

    const dst_bytes = inst.writableGuestBytes(dst_ptr, 8).?;
    try testing.expectEqual(@as(u32, 0xAAAA_AAAA), std.mem.readInt(u32, dst_bytes[0..4], .little));
    try testing.expectEqual(@as(u32, 0xBBBB_BBBB), std.mem.readInt(u32, dst_bytes[4..8], .little));
    try testing.expect(inst.streams.getPtr(handle).?.pending_read == null);
    // No tail to buffer — exact-fit transfer.
    try testing.expectEqual(@as(usize, 0), inst.streams.getPtr(handle).?.buffer.items.len);
    var ready: [4]u32 = undefined;
    try testing.expectEqual(@as(u32, 1), ws.pollReady(&ready));
}

test "stream.write count > pending reader max: extra bytes buffer for next read" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newStreamU32Inst(testing.allocator);
    defer destroyStreamInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    const src_ptr: u32 = 0;
    const dst_ptr: u32 = 64;
    const src_bytes = inst.writableGuestBytes(src_ptr, 12).?;
    std.mem.writeInt(u32, src_bytes[0..4], 0xAAAA_0001, .little);
    std.mem.writeInt(u32, src_bytes[4..8], 0xAAAA_0002, .little);
    std.mem.writeInt(u32, src_bytes[8..12], 0xAAAA_0003, .little);

    // Park a reader with max_count=1.
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(dst_ptr));
    try env.pushI32(@bitCast(@as(u32, 1)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    _ = try env.popI32(); // discard STARTING

    // Writer pushes count=3 — reader takes 1, the rest is buffered.
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(src_ptr));
    try env.pushI32(@bitCast(@as(u32, 3)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_write = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const write_status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.completed, 1), write_status);

    // Reader got the first element; remaining 2 elements (8 bytes)
    // sit in the FIFO awaiting the next reader.
    const dst_bytes = inst.writableGuestBytes(dst_ptr, 4).?;
    try testing.expectEqual(@as(u32, 0xAAAA_0001), std.mem.readInt(u32, dst_bytes[0..4], .little));
    try testing.expect(inst.streams.getPtr(handle).?.pending_read == null);
    try testing.expectEqual(@as(usize, 8), inst.streams.getPtr(handle).?.buffer.items.len);

    // A subsequent read with max_count=2 drains the buffered tail.
    const dst2_ptr: u32 = 128;
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(dst2_ptr));
    try env.pushI32(@bitCast(@as(u32, 2)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const read2_status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.completed, 2), read2_status);

    const dst2_bytes = inst.writableGuestBytes(dst2_ptr, 8).?;
    try testing.expectEqual(@as(u32, 0xAAAA_0002), std.mem.readInt(u32, dst2_bytes[0..4], .little));
    try testing.expectEqual(@as(u32, 0xAAAA_0003), std.mem.readInt(u32, dst2_bytes[4..8], .little));
    try testing.expectEqual(@as(usize, 0), inst.streams.getPtr(handle).?.buffer.items.len);
}

test "stream.cancel-read on parked reader returns CANCELLED and clears slot" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newStreamU32Inst(testing.allocator);
    defer destroyStreamInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    // Park a reader so cancel has something to clear.
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(@as(u32, 0)));
    try env.pushI32(@bitCast(@as(u32, 4)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    _ = try env.popI32(); // discard STARTING
    try testing.expect(inst.streams.getPtr(handle).?.pending_read != null);

    // Cancel — clears pending_read and reports CANCELLED.
    try env.pushI32(@bitCast(handle));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_cancel_read = .{ .type_idx = 0, .is_async = false } } },
        env,
        null,
        testing.allocator,
    );
    const status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.cancelled, 0), status);
    try testing.expect(inst.streams.getPtr(handle).?.pending_read == null);
}

test "stream.drop-writable while buffer drained: subsequent read returns DROPPED (EOF)" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newStreamU32Inst(testing.allocator);
    defer destroyStreamInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const packed_handles: u64 = @bitCast(try env.popI64());
    const r_idx: u32 = @truncate(packed_handles >> 32);
    const w_idx: u32 = @truncate(packed_handles & 0xFFFF_FFFF);

    // Write 1×u32 then drain it via a matching read; buffer is now empty.
    const src_ptr: u32 = 0;
    const dst_ptr: u32 = 32;
    const src_bytes = inst.writableGuestBytes(src_ptr, 4).?;
    std.mem.writeInt(u32, src_bytes[0..4], 0xDEAD_BEEF, .little);
    try env.pushI32(@bitCast(r_idx));
    try env.pushI32(@bitCast(src_ptr));
    try env.pushI32(@bitCast(@as(u32, 1)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_write = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    _ = try env.popI32(); // discard RETURNED 1

    try env.pushI32(@bitCast(r_idx));
    try env.pushI32(@bitCast(dst_ptr));
    try env.pushI32(@bitCast(@as(u32, 1)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    _ = try env.popI32(); // discard RETURNED 1
    try testing.expectEqual(@as(usize, 0), inst.streams.getPtr(r_idx).?.buffer.items.len);

    // Writer drops without further writes.
    try env.pushI32(@bitCast(w_idx));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_drop_writable = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    // Reader end still open — entry remains until dual-close.
    try testing.expectEqual(@as(u32, 1), inst.streams.count());
    try testing.expect(inst.streams.getPtr(r_idx).?.write_closed);

    // A subsequent read observes EOF as DROPPED.
    try env.pushI32(@bitCast(r_idx));
    try env.pushI32(@bitCast(dst_ptr));
    try env.pushI32(@bitCast(@as(u32, 1)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.dropped, 0), status);
}

// ── #583 B2: zero-copy `on_read_into` host_driver specialisation ────────

fn newStreamU8Inst(testing_allocator: std.mem.Allocator) !*instance_mod.ComponentInstance {
    const StreamTypeFixture = struct {
        var types_array = [_]ctypes.TypeDef{.{ .val = .u8 }};
        var comp: ctypes.Component = .{
            .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
            .components = &.{},       .instances = &.{},      .aliases = &.{},
            .types = &.{},            .canons = &.{},         .imports = &.{},
            .exports = &.{},
        };
    };
    StreamTypeFixture.comp.types = &StreamTypeFixture.types_array;
    const inst = try instance_mod.instantiate(&StreamTypeFixture.comp, testing_allocator);
    try inst.enableTestMem(testing_allocator, 4096);
    return inst;
}

/// Synthetic driver state shared by the zero-copy tests below: a fixed
/// payload that the driver writes / appends and a per-invocation
/// counter so the tests can assert which callback was invoked.
const ZeroCopyTestDriver = struct {
    payload: []const u8,
    on_read_into_calls: u32 = 0,
    on_read_calls: u32 = 0,

    fn intoCb(
        opaque_ctx: ?*anyopaque,
        dst: []u8,
    ) async_mod.HostStreamReadInto {
        const self: *ZeroCopyTestDriver = @ptrCast(@alignCast(opaque_ctx.?));
        self.on_read_into_calls += 1;
        const n = @min(self.payload.len, dst.len);
        if (n == 0) return .{ .action = .would_block };
        @memcpy(dst[0..n], self.payload[0..n]);
        return .{ .action = .progressed, .bytes_written = @intCast(n) };
    }

    fn fallbackCb(
        opaque_ctx: ?*anyopaque,
        stream: *async_mod.AsyncStream,
        allocator: std.mem.Allocator,
    ) async_mod.HostStreamAction {
        const self: *ZeroCopyTestDriver = @ptrCast(@alignCast(opaque_ctx.?));
        self.on_read_calls += 1;
        stream.buffer.appendSlice(allocator, self.payload) catch return .err;
        return .progressed;
    }
};

test "stream.read zero-copy: aligned dst → driver writes into guest linmem with no scratch alloc (#583 B2)" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newStreamU8Inst(testing.allocator);
    defer destroyStreamInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    // Install zero-copy driver (both callbacks set so the test can
    // assert which one fired). `payload` is the byte sequence the
    // driver will deposit into the guest dst slice.
    var driver_state = ZeroCopyTestDriver{ .payload = "hello-zc" };
    {
        const s = inst.streams.getPtr(handle).?;
        s.host_driver = .{
            .context = &driver_state,
            .on_read = &ZeroCopyTestDriver.fallbackCb,
            .on_read_into = &ZeroCopyTestDriver.intoCb,
        };
    }

    // u8 stream: any guest_ptr is naturally elem-aligned. Read 8 bytes
    // into dst at offset 16.
    const dst_ptr: u32 = 16;
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(dst_ptr));
    try env.pushI32(@bitCast(@as(u32, 8)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.completed, 8), status);

    // Zero-copy callback fired exactly once; legacy `on_read` never
    // ran (no scratch FIFO allocation either way).
    try testing.expectEqual(@as(u32, 1), driver_state.on_read_into_calls);
    try testing.expectEqual(@as(u32, 0), driver_state.on_read_calls);
    try testing.expectEqual(@as(usize, 0), inst.streams.getPtr(handle).?.buffer.items.len);
    try testing.expectEqual(@as(usize, 0), inst.streams.getPtr(handle).?.buffer.capacity);

    // Output bytes correct.
    const written = inst.writableGuestBytes(dst_ptr, 8).?;
    try testing.expectEqualStrings("hello-zc", written);
}

test "stream.read zero-copy: misaligned dst on stream<u32> falls back to scratch `on_read` (#583 B2)" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    // u32 stream → elem_size = 4 / required alignment = 4.
    const inst = try newStreamU32Inst(testing.allocator);
    defer destroyStreamInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    // Driver state — eight bytes of payload that lower as two u32s.
    var driver_state = ZeroCopyTestDriver{ .payload = "\x11\x11\x11\x11\x22\x22\x22\x22" };
    {
        const s = inst.streams.getPtr(handle).?;
        s.host_driver = .{
            .context = &driver_state,
            .on_read = &ZeroCopyTestDriver.fallbackCb,
            .on_read_into = &ZeroCopyTestDriver.intoCb,
        };
    }

    // `guest_ptr = 1` is 1-byte misaligned for a 4-byte element type;
    // the executor must reject the zero-copy fast path and use the
    // legacy `on_read` (scratch FIFO) callback instead.
    const dst_ptr: u32 = 1;
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(dst_ptr));
    try env.pushI32(@bitCast(@as(u32, 2)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.completed, 2), status);

    // The fallback path fired (not the zero-copy intoCb).
    try testing.expectEqual(@as(u32, 0), driver_state.on_read_into_calls);
    try testing.expectEqual(@as(u32, 1), driver_state.on_read_calls);

    // Output bytes still correct — fallback memcpys two u32s into the
    // misaligned guest dst. (The legacy path uses byte memcpy so
    // unaligned destinations work; the issue here is downstream
    // guest load semantics, which is the guest's problem.)
    const written = inst.writableGuestBytes(dst_ptr, 8).?;
    try testing.expectEqualSlices(u8, "\x11\x11\x11\x11\x22\x22\x22\x22", written);
}

test "stream.read zero-copy: dst extending past memory.size falls back (no UAF of guest linmem) (#583 B2)" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    // u8 stream — alignment isn't the gating factor here; we want to
    // stress the bounds check on `writableGuestBytes(guest_ptr, len)`.
    // `newStreamU8Inst` allocates a 4 KiB test_mem.
    const inst = try newStreamU8Inst(testing.allocator);
    defer destroyStreamInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    var driver_state = ZeroCopyTestDriver{ .payload = "ok" };
    {
        const s = inst.streams.getPtr(handle).?;
        s.host_driver = .{
            .context = &driver_state,
            .on_read = &ZeroCopyTestDriver.fallbackCb,
            .on_read_into = &ZeroCopyTestDriver.intoCb,
        };
    }

    // `dst_ptr = 4090` + `max_count = 64` overshoots the 4096-byte
    // test_mem. The would-be borrowed slice straddles the synthetic
    // memory.size — the spec safety check (the "cross-page write
    // because guest memory could grow mid-call" rule) forces the
    // executor onto the scratch-FIFO path, where the legacy
    // `on_read` callback can still service the request.
    const dst_ptr: u32 = 4090;
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(dst_ptr));
    try env.pushI32(@bitCast(@as(u32, 64)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const status: u32 = @bitCast(try env.popI32());
    // `on_read` payload is "ok" (2 bytes), which fits inside the 6
    // bytes remaining at `dst_ptr = 4090..4096`. The executor's FIFO
    // drain memcpys those 2 bytes there and reports completed(2).
    try testing.expectEqual(async_canon.packStatus(.completed, 2), status);

    try testing.expectEqual(@as(u32, 0), driver_state.on_read_into_calls);
    try testing.expectEqual(@as(u32, 1), driver_state.on_read_calls);

    const written = inst.writableGuestBytes(dst_ptr, 2).?;
    try testing.expectEqualStrings("ok", written);
}

// ── #583 B2 follow-up: zero-copy `on_write_from` host_driver specialisation ──

/// Synthetic driver state shared by the zero-copy `stream.write`
/// tests below: an arena for the bytes the driver "consumes" and a
/// per-invocation counter so the tests can assert which callback was
/// invoked.
const ZeroCopyWriteTestDriver = struct {
    sink: std.ArrayListUnmanaged(u8) = .empty,
    on_write_from_calls: u32 = 0,
    on_write_calls: u32 = 0,
    next_action: async_mod.HostStreamAction = .progressed,

    fn fromCb(
        opaque_ctx: ?*anyopaque,
        src: []const u8,
    ) async_mod.HostStreamAction {
        const self: *ZeroCopyWriteTestDriver = @ptrCast(@alignCast(opaque_ctx.?));
        self.on_write_from_calls += 1;
        if (self.next_action != .progressed) return self.next_action;
        self.sink.appendSlice(std.testing.allocator, src) catch return .err;
        return .progressed;
    }

    fn legacyCb(
        opaque_ctx: ?*anyopaque,
        _: *async_mod.AsyncStream,
        bytes: []const u8,
        _: std.mem.Allocator,
    ) async_mod.HostStreamAction {
        const self: *ZeroCopyWriteTestDriver = @ptrCast(@alignCast(opaque_ctx.?));
        self.on_write_calls += 1;
        if (self.next_action != .progressed) return self.next_action;
        self.sink.appendSlice(std.testing.allocator, bytes) catch return .err;
        return .progressed;
    }

    fn deinit(self: *ZeroCopyWriteTestDriver) void {
        self.sink.deinit(std.testing.allocator);
    }
};

test "stream.write zero-copy: driver receives borrowed guest linmem slice via on_write_from (#583 B2 follow-up)" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newStreamU8Inst(testing.allocator);
    defer destroyStreamInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    // Seed the guest source bytes into linmem at src_ptr.
    const src_ptr: u32 = 32;
    const payload = "hello-write-zc";
    const src_bytes = inst.writableGuestBytes(src_ptr, payload.len).?;
    @memcpy(src_bytes, payload);

    // Install zero-copy driver (only `on_write_from` set so the test
    // can assert which callback fired and that the executor uses the
    // thinner shape when only it is present).
    var driver_state = ZeroCopyWriteTestDriver{};
    defer driver_state.deinit();
    {
        const s = inst.streams.getPtr(handle).?;
        s.host_driver = .{
            .context = &driver_state,
            .on_write_from = &ZeroCopyWriteTestDriver.fromCb,
        };
    }

    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(src_ptr));
    try env.pushI32(@bitCast(@as(u32, payload.len)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_write = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.completed, payload.len), status);

    // Zero-copy callback fired exactly once; legacy `on_write` never
    // ran (and the FIFO never grew because the driver consumed all
    // bytes synchronously).
    try testing.expectEqual(@as(u32, 1), driver_state.on_write_from_calls);
    try testing.expectEqual(@as(u32, 0), driver_state.on_write_calls);
    try testing.expectEqual(@as(usize, 0), inst.streams.getPtr(handle).?.buffer.items.len);
    try testing.expectEqualStrings(payload, driver_state.sink.items);
}

test "stream.write zero-copy: on_write_from preferred over on_write when both installed (#583 B2 follow-up)" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newStreamU8Inst(testing.allocator);
    defer destroyStreamInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    const src_ptr: u32 = 64;
    const payload = "prefer-zc";
    const src_bytes = inst.writableGuestBytes(src_ptr, payload.len).?;
    @memcpy(src_bytes, payload);

    // Install BOTH callbacks. The executor must prefer the zero-copy
    // `on_write_from` and never invoke the legacy `on_write`. This
    // pins the preference contract — installing both shapes is the
    // production pattern (matches `tcpSendStream` / `fsWriteViaStream`
    // after this PR).
    var driver_state = ZeroCopyWriteTestDriver{};
    defer driver_state.deinit();
    {
        const s = inst.streams.getPtr(handle).?;
        s.host_driver = .{
            .context = &driver_state,
            .on_write = &ZeroCopyWriteTestDriver.legacyCb,
            .on_write_from = &ZeroCopyWriteTestDriver.fromCb,
        };
    }

    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(src_ptr));
    try env.pushI32(@bitCast(@as(u32, payload.len)));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_write = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.completed, payload.len), status);

    try testing.expectEqual(@as(u32, 1), driver_state.on_write_from_calls);
    try testing.expectEqual(@as(u32, 0), driver_state.on_write_calls);
    try testing.expectEqualStrings(payload, driver_state.sink.items);
}

test "stream.write: cross-page write (ptr + len > memory.size) is rejected before any host_driver call (#583 B2 follow-up)" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newStreamU8Inst(testing.allocator);
    defer destroyStreamInst(inst);

    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    // Install both callbacks. Neither should fire — `readGuestBytes`
    // rejects the (ptr, len) pair before the executor consults the
    // driver.
    var driver_state = ZeroCopyWriteTestDriver{};
    defer driver_state.deinit();
    {
        const s = inst.streams.getPtr(handle).?;
        s.host_driver = .{
            .context = &driver_state,
            .on_write = &ZeroCopyWriteTestDriver.legacyCb,
            .on_write_from = &ZeroCopyWriteTestDriver.fromCb,
        };
    }

    // `src_ptr = 4090` + `count = 64` overshoots the 4096-byte
    // test_mem; the executor's `readGuestBytes` returns null and we
    // trap with `error.MemoryNotAvailable` — the safety guarantee
    // mirrors PR #599's read-side cross-page test (the borrowed
    // slice must NEVER extend past `memory.size` before reaching
    // a host driver).
    const src_ptr: u32 = 4090;
    try env.pushI32(@bitCast(handle));
    try env.pushI32(@bitCast(src_ptr));
    try env.pushI32(@bitCast(@as(u32, 64)));
    try testing.expectError(error.MemoryNotAvailable, dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_write = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    ));
    try testing.expectEqual(@as(u32, 0), driver_state.on_write_from_calls);
    try testing.expectEqual(@as(u32, 0), driver_state.on_write_calls);
    try testing.expectEqual(@as(usize, 0), driver_state.sink.items.len);
}

// ── #550: waitable.join + waitable-set.{wait,poll} event delivery ─────────

test "waitable.join + waitable-set.wait: stream-write event delivered to a parked write waitable (#550)" {
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    const inst = try newStreamU32Inst(testing.allocator);
    defer destroyStreamInst(inst);

    // Allocate a stream + a waitable-set.
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const stream_handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    try dispatchCanonBuiltin(inst, .{ .async_canon = .waitable_set_new }, env, null, testing.allocator);
    const ws_handle: u32 = @bitCast(try env.popI32());

    // Park a stream.write by pre-flagging `pending_write` so the
    // join arm classifies the join as `stream_write` (matches the
    // post-#541 BLOCKED-then-join sequence wit-bindgen emits when
    // an out-of-band sink can't accept bytes immediately).
    {
        const s = inst.streams.getPtr(stream_handle).?;
        s.pending_write = .{ .guest_ptr = 0, .count = 4 };
    }

    // `canon waitable.join : [waitable, set]` — wasm pop order is set
    // (top), waitable (under), so push waitable first then set.
    try env.pushI32(@bitCast(stream_handle));
    try env.pushI32(@bitCast(ws_handle));
    try dispatchCanonBuiltin(inst, .{ .async_canon = .waitable_join }, env, null, testing.allocator);

    try testing.expectEqual(@as(usize, 1), inst.waitable_sets.getPtr(ws_handle).?.items.items.len);
    try testing.expectEqual(
        async_mod.WaitableSet.WaitableItem.Kind.stream_write,
        inst.waitable_sets.getPtr(ws_handle).?.items.items[0].kind,
    );
    try testing.expectEqual(@as(u32, 0), inst.streams.getPtr(stream_handle).?.write_waitable_idx.?);

    // Drop the readable end — the parked writer must wake with a
    // `dropped` event so the guest can re-issue `stream.write` and
    // observe the closed peer synchronously.
    try env.pushI32(@bitCast(stream_handle));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_drop_readable = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );

    // `canon waitable-set.wait : [set, out_ptr] -> [event]` —
    // out_ptr at top of stack.
    const out_ptr: u32 = 0x100;
    try env.pushI32(@bitCast(ws_handle));
    try env.pushI32(@bitCast(out_ptr));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .waitable_set_wait = .{ .cancellable = false, .memory = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const event: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(@intFromEnum(async_canon.EventCode.stream_write), event);

    // Payload at out_ptr: (handle, packed_status).
    const ev_bytes = inst.writableGuestBytes(out_ptr, 8).?;
    try testing.expectEqual(stream_handle, std.mem.readInt(u32, ev_bytes[0..4], .little));
    try testing.expectEqual(async_canon.packStatus(.dropped, 0), std.mem.readInt(u32, ev_bytes[4..8], .little));
}

test "waitable-set.poll: settled future surfaces FUTURE_READ event with the right handle/code (#550)" {
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

    // Allocate a future + waitable-set. The future stands in for the
    // `future<result<_,error-code>>` returned by `write-via-stream`.
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_new = .{ .type_idx = 0 } } },
        env,
        null,
        testing.allocator,
    );
    const future_handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    try dispatchCanonBuiltin(inst, .{ .async_canon = .waitable_set_new }, env, null, testing.allocator);
    const ws_handle: u32 = @bitCast(try env.popI32());

    // Park a reader via `future.read` — returns BLOCKED so the join
    // arm can detect `pending_read != null` and classify as future_read.
    const guest_ptr: u32 = 0x80;
    try env.pushI32(@bitCast(future_handle));
    try env.pushI32(@bitCast(guest_ptr));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .future_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    try testing.expectEqual(async_canon.BLOCKED_STATUS, @as(u32, @bitCast(try env.popI32())));

    // Join the parked future to the waitable-set.
    try env.pushI32(@bitCast(future_handle));
    try env.pushI32(@bitCast(ws_handle));
    try dispatchCanonBuiltin(inst, .{ .async_canon = .waitable_join }, env, null, testing.allocator);

    // poll before settlement → NONE.
    const out_ptr: u32 = 0x200;
    try env.pushI32(@bitCast(ws_handle));
    try env.pushI32(@bitCast(out_ptr));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .waitable_set_poll = .{ .cancellable = false, .memory = 0 } } },
        env,
        null,
        testing.allocator,
    );
    try testing.expectEqual(@intFromEnum(async_canon.EventCode.none), @as(u32, @bitCast(try env.popI32())));

    // Simulate host settling the future the way
    // `writeViaStreamOnDropWritable` does: directly populate the
    // parked reader's destination + set state ready + fire setReady.
    {
        const fut = inst.futures.getPtr(future_handle).?;
        const dst = inst.writableGuestBytes(fut.pending_read.?.guest_ptr, 1).?;
        dst[0] = 0;
        fut.pending_read = null;
        fut.state = .ready;
        fut.write_closed = true;
        if (fut.waitable_set) |ws| if (fut.read_waitable_idx) |idx|
            ws.setReady(idx, testing.allocator, async_canon.packStatus(.completed, 0));
    }

    // poll after settlement → FUTURE_READ event with the future handle.
    try env.pushI32(@bitCast(ws_handle));
    try env.pushI32(@bitCast(out_ptr));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .waitable_set_poll = .{ .cancellable = false, .memory = 0 } } },
        env,
        null,
        testing.allocator,
    );
    try testing.expectEqual(@intFromEnum(async_canon.EventCode.future_read), @as(u32, @bitCast(try env.popI32())));
    const ev_bytes = inst.writableGuestBytes(out_ptr, 8).?;
    try testing.expectEqual(future_handle, std.mem.readInt(u32, ev_bytes[0..4], .little));
    try testing.expectEqual(async_canon.packStatus(.completed, 0), std.mem.readInt(u32, ev_bytes[4..8], .little));

    // The lifted Ok discriminant is at `guest_ptr` already (written
    // synchronously above) so wit-bindgen's lift-from-original-ptr
    // contract is satisfied — no re-issued `future.read` needed.
    try testing.expectEqual(@as(u8, 0), inst.writableGuestBytes(guest_ptr, 1).?[0]);
}

test "waitable.join on already-settled future: marks ready synchronously so the next wait surfaces it (#550)" {
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

    // Allocate a future and pre-settle it (state=.ready, payload set)
    // — this mirrors a `read-via-stream` companion future that the
    // host creates already-Ok.
    const future_handle = inst.allocAsyncHandle();
    const payload = try testing.allocator.alloc(u8, 1);
    payload[0] = 0;
    try inst.futures.put(testing.allocator, future_handle, .{
        .elem_type_idx = 0,
        .state = .ready,
        .payload = payload,
    });

    try dispatchCanonBuiltin(inst, .{ .async_canon = .waitable_set_new }, env, null, testing.allocator);
    const ws_handle: u32 = @bitCast(try env.popI32());

    // Join — the future is already ready, so `waitable.join` must
    // synchronously mark the slot ready (otherwise the guest would
    // wait forever for an event that's already past).
    try env.pushI32(@bitCast(future_handle));
    try env.pushI32(@bitCast(ws_handle));
    try dispatchCanonBuiltin(inst, .{ .async_canon = .waitable_join }, env, null, testing.allocator);

    const ws = inst.waitable_sets.getPtr(ws_handle).?;
    try testing.expectEqual(@as(usize, 1), ws.ready_queue.items.len);

    // Wait drains the event.
    const out_ptr: u32 = 0x300;
    try env.pushI32(@bitCast(ws_handle));
    try env.pushI32(@bitCast(out_ptr));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .waitable_set_wait = .{ .cancellable = false, .memory = 0 } } },
        env,
        null,
        testing.allocator,
    );
    try testing.expectEqual(@intFromEnum(async_canon.EventCode.future_read), @as(u32, @bitCast(try env.popI32())));
    const ev_bytes = inst.writableGuestBytes(out_ptr, 8).?;
    try testing.expectEqual(future_handle, std.mem.readInt(u32, ev_bytes[0..4], .little));
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

// ── #480: error-context.new / debug-message copy through guest memory ──────

test "dispatchCanonBuiltin: error_context.new captures debug-message bytes (#480)" {
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
    try inst.enableTestMem(testing.allocator, 4096);
    defer inst.disableTestMem();

    const msg = "hello error";
    const ptr = inst.hostAllocAndWrite(msg).?;

    try env.pushI32(@bitCast(ptr));
    try env.pushI32(@intCast(msg.len));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .error_context_new = .{ .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );

    const handle: u32 = @bitCast(try env.popI32());
    try testing.expect(handle > 0);
    const stored = inst.error_contexts.get(handle).?;
    try testing.expectEqualStrings(msg, stored);
}

test "dispatchCanonBuiltin: error_context.debug_message returns stored bytes via guest memory (#480)" {
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
    try inst.enableTestMem(testing.allocator, 4096);
    defer inst.disableTestMem();

    // Pre-populate the error-context table directly so the test
    // exercises only the .debug_message arm.
    const stored = try testing.allocator.dupe(u8, "boom: out of resources");
    const handle = inst.allocAsyncHandle();
    try inst.error_contexts.put(testing.allocator, handle, stored);

    try env.pushI32(@bitCast(handle));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .error_context_debug_message = .{ .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );

    // (ptr, len) pushed in that order → pop len first, then ptr.
    const out_len: u32 = @bitCast(try env.popI32());
    const out_ptr: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(@as(u32, @intCast(stored.len)), out_len);
    const slice = inst.test_mem.?.buffer[out_ptr..][0..out_len];
    try testing.expectEqualStrings(stored, slice);
}

test "dispatchCanonBuiltin: error_context.new + drop + new produces distinct handles, no leak (#480)" {
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
    try inst.enableTestMem(testing.allocator, 4096);
    defer inst.disableTestMem();

    const msg1 = "first failure";
    const ptr1 = inst.hostAllocAndWrite(msg1).?;
    try env.pushI32(@bitCast(ptr1));
    try env.pushI32(@intCast(msg1.len));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .error_context_new = .{ .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const h1: u32 = @bitCast(try env.popI32());

    try env.pushI32(@bitCast(h1));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .error_context_drop },
        env,
        null,
        testing.allocator,
    );
    try testing.expectEqual(@as(u32, 0), inst.error_contexts.count());

    const msg2 = "second failure";
    const ptr2 = inst.hostAllocAndWrite(msg2).?;
    try env.pushI32(@bitCast(ptr2));
    try env.pushI32(@intCast(msg2.len));
    try dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .error_context_new = .{ .opts = &.{} } } },
        env,
        null,
        testing.allocator,
    );
    const h2: u32 = @bitCast(try env.popI32());

    try testing.expect(h1 != h2);
    try testing.expectEqual(@as(u32, 1), inst.error_contexts.count());
    try testing.expectEqualStrings(msg2, inst.error_contexts.get(h2).?);
    // testing.allocator (GPA) fails the test on a leak; ComponentInstance.deinit
    // frees the surviving entry's bytes.
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
    /// `canon.lower (async) (func)` — Binary.md `canonopt 0x06` shared with
    /// the lift side. When set, the canon-lower trampoline returns a packed
    /// status word `(handle << 4) | STATUS` per the component-model spec
    /// instead of the lifted result values. The host function is allowed
    /// to populate a phantom return slot with the async waitable handle
    /// (currently a `future<()>` handle for `wasi:clocks@0.3.x`
    /// `wait-for` / `wait-until`); the trampoline inspects that future's
    /// state and packs `STATUS_STARTED`/`STATUS_RETURNED` accordingly. (#551.)
    is_async: bool = false,

    pub fn fromOpts(opts: []const ctypes.CanonOpt) LowerOptions {
        var lo = LowerOptions{};
        for (opts) |opt| {
            switch (opt) {
                .memory => |idx| lo.memory_idx = idx,
                .realloc => |idx| lo.realloc_idx = idx,
                .post_return => {},
                .string_encoding => |enc| lo.string_encoding = enc,
                // `async` flips into the async-lower path (#551). The
                // canonopt is shared between lift and lower; the binary
                // grammar reuses one `opts` vec across both kinds.
                .async_lift => lo.is_async = true,
                // `callback` is only meaningful on the lift side; accept
                // here as a no-op.
                .callback => {},
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
    /// Optional component-level dispatch target used by the AOT host-import
    /// trampoline pool, which has no ExecEnv and therefore calls straight into
    /// `callComponentFuncByLocal` instead of the host-func path.
    lift_target: ?ComponentInstance.ExportedFunc.Local = null,
    lower_opts: LowerOptions,

    /// Per-trampoline extension to the component's type indexspace, used
    /// to resolve param/result `.type_idx` references that point into an
    /// instance-type body's local type space. Empty when the FuncType
    /// for this trampoline came from the component-level type indexspace
    /// directly (e.g. hand-authored fixtures).
    extended_types: []const ctypes.TypeDef = &.{},
    extended_indexspace: []const ?u32 = &.{},

    /// `true` when the underlying lifted `FuncType` is declared with the
    /// async-functype tag (`0x43`, loader records this on `FuncType.is_async`).
    /// Mirrored here so `componentTrampoline` can route through the
    /// canon-lower-of-async-func path even when the canon decl itself
    /// did NOT carry the `async_lift` canon-opt (the common case for
    /// wit-bindgen ≥ 0.45 `async func` lowers, which produce a plain
    /// `canon.lower (func $f)` decl and rely on the FuncType-level
    /// async-ness to drive the `(handle << 4) | STATUS_*` packed-status
    /// return shape). The legacy `lower_opts.is_async` (from the
    /// `async_lift` canon-opt) is still respected — either flag flips
    /// the trampoline into the async path. (#564.)
    is_async_func: bool = false,

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

    var frame: CallFrame = .{ .interp = InterpFrame.init(env) };
    defer frame.deinit();

    const flat_params = countFlatTypes(registry, ctx.param_types);
    const flat_results = countFlatTypes(registry, ctx.result_types);

    // `canon.lower (async)` uses a tighter param-spill threshold
    // (MAX_FLAT_PARAMS_ASYNC = 4 vs sync's MAX_FLAT_PARAMS = 16) per
    // the component-model spec — any flatten count exceeding 4 forces
    // the caller to spill the param block to memory and pass a single
    // `params_ptr` instead. wit-bindgen-emitted shims confirm this:
    // `[async-lower][method]descriptor.open-at` with 6 flat-params
    // lowers to `(params_ptr, retptr) -> status` (2 i32 args), while
    // `create-directory-at` with 3 flat-params lowers to
    // `(self, path_ptr, path_len, retptr) -> status` (4 i32 args).
    // The result side is uniform: a single `retptr` if the lifted
    // result type is non-empty. (#564.)
    //
    // The "async-lower" flag is the OR of the canon-opt-level
    // `is_async` (`async_lift` opt explicitly set on the canon.lower
    // decl) and the FuncType-level `is_async` (parsed from the `0x43`
    // functype tag — wit-bindgen ≥ 0.45 emits `(canon lower ... async)`
    // for async funcs; the FuncType-level flag also flips when the
    // canon decl forgot the opt for any reason).
    const is_async_lower = ctx.lower_opts.is_async or ctx.is_async_func;
    const params_spill_threshold: u32 = if (is_async_lower) MAX_FLAT_PARAMS_ASYNC else MAX_FLAT_PARAMS;
    const params_spill = flat_params > params_spill_threshold;
    const results_spill = flat_results > MAX_FLAT_RESULTS;

    // `canon.lower (async)` of an async-func WITH at least one lifted
    // result has the core signature `(P-flat..., retptr) -> i32 status`
    // regardless of `flat_results`. The retptr is always the last param
    // pushed by the caller — we pop it the same way `results_spill`
    // does, but the trigger is the canon-async-with-result shape, not
    // the flat-results overflow.
    const async_with_result_retptr = is_async_lower and ctx.result_types.len > 0;
    const has_retptr = (!is_async_lower and results_spill) or async_with_result_retptr;

    // Resolve linear memory for either spill path. The memory option is
    // mandatory whenever spilling occurs, and we need a non-empty
    // core_instances list to actually own a memory.
    if (params_spill or has_retptr) {
        if (ctx.lower_opts.memory_idx == null) return trampolineTrap(env, ctx, error.MemoryNotAvailable, .memory_resolve);
        if (ctx.comp_inst.core_instances.len == 0) return trampolineTrap(env, ctx, error.MemoryNotAvailable, .memory_resolve);
    }

    // Pop result-destination pointer first if results spill (it was pushed
    // last by the caller).
    var result_dest_ptr: u32 = 0;
    if (has_retptr) {
        result_dest_ptr = @bitCast(env.popI32() catch |err| return trampolineTrap(env, ctx, err, .lift_args));
    }

    // Lift args. Stack-buffer the common ≤8-param case so high-volume
    // host calls (the http-fields fixture sweeps ≥10k calls through
    // this path) don't pay the heap-alloc-and-free price every time
    // (#552).
    var args_stack_buf: [8]InterfaceValue = undefined;
    const args_heap: ?[]InterfaceValue = if (ctx.param_types.len <= args_stack_buf.len) null else (allocator.alloc(InterfaceValue, ctx.param_types.len) catch |err|
        return trampolineTrap(env, ctx, err, .lift_args));
    defer if (args_heap) |h| allocator.free(h);
    const args: []InterfaceValue = args_heap orelse args_stack_buf[0..ctx.param_types.len];
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
            args[i] = popInterfaceValue(&frame, ctx.param_types[i], registry, allocator) catch |err|
                return trampolineTrap(env, ctx, err, .lift_args);
        }
    }

    // Invoke host. Host owns allocation of any compound result values via
    // `allocator`; we deinit each result after lowering so payloads
    // (e.g. `.result_val.payload` for input-stream.blocking-read) don't leak.
    //
    // Async-lower (#551, #564): when `canon.lower (async)` of an async
    // func is invoked, the host fn delivers a future/subtask handle in
    // `results[0]` instead of a lifted value. We then pack
    // `(handle << 4) | STATUS_*` and push it as the i32 status word per
    // the component-model spec.
    //
    //   * Funcs with NO lifted result — the `wasi:clocks@0.3.x`
    //     `wait-for` / `wait-until` shape — get a single phantom slot
    //     for the host's `future<()>` handle. (#551.)
    //   * Funcs WITH a lifted result — the `wasi:filesystem` /
    //     `wasi:sockets@0.3.x` async-func shape — write the host's
    //     `future<R>` handle into `results[0]`, and the trampoline
    //     copies the future's pre-lowered canonical-ABI `payload` bytes
    //     into the caller's `retptr` (settled-synchronously fast path
    //     used by every current P3 adapter). If the future is `.pending`
    //     the trampoline returns `STATUS_STARTED` with the handle in
    //     the high bits so the guest can `waitable.join` + wait on it.
    //     (#564.)
    //
    // The `(handle << 4) | STATUS` encoding is the same shape required
    // by wit-bindgen ≥ 0.53's `wit_bindgen::Subtask` runtime; the
    // bridging future on `ComponentInstance.futures` carries
    // `subtask_managed = true` (set here on first observation) so the
    // guest's subsequent `waitable.join` routes it through the
    // `.subtask` waitable kind (see `executor.joinWaitable`).
    const is_async_no_result_lower = is_async_lower and ctx.result_types.len == 0;
    const host_result_len: usize = if (is_async_no_result_lower) 1 else ctx.result_types.len;
    var results_stack_buf: [4]InterfaceValue = undefined;
    const results_heap: ?[]InterfaceValue = if (host_result_len <= results_stack_buf.len) null else (allocator.alloc(InterfaceValue, host_result_len) catch |err|
        return trampolineTrap(env, ctx, err, .lift_args));
    const results: []InterfaceValue = results_heap orelse results_stack_buf[0..host_result_len];
    defer {
        for (results) |r| r.deinit(allocator);
        if (results_heap) |h| allocator.free(h);
    }
    if (is_async_no_result_lower) {
        // Phantom slot for the host's future/subtask handle. Initialise
        // to a sentinel so a host fn that forgets to write triggers a
        // clean STATUS_RETURNED with handle=0 (rather than reading
        // uninitialised memory).
        results[0] = .{ .u32 = 0 };
    }
    const call = ctx.host_func.call orelse {
        return trampolineTrap(env, ctx, error.HostFuncNotBound, .host_call);
    };
    call(ctx.host_func.context, ctx.comp_inst, args, results, allocator) catch |err| {
        return trampolineTrap(env, ctx, err, .host_call);
    };

    // Async-lower trampoline result path (#551, #564). Pack
    // `(handle << 4) | STATUS` and push it as the i32 status word. For
    // the with-result shape we also copy the future's pre-lowered
    // payload bytes into `mem[retptr..]` so the guest observes the
    // canonical-ABI result synchronously (every current P3 host fn
    // mints `.ready` futures with pre-populated payload bytes).
    if (is_async_lower) {
        const handle: u32 = if (results.len == 0) 0 else switch (results[0]) {
            .handle => |h| h,
            .u32 => |v| v,
            else => 0,
        };

        // Flag the bridging future as subtask-managed so a subsequent
        // `waitable.join` on the guest side routes the handle through
        // the `.subtask` waitable kind (delivering EVENT_SUBTASK +
        // STATUS_* per the wit-bindgen `Subtask` decoder). This is a
        // no-op for `wasi:clocks` timer-futures (already true) and
        // unconditional for the freshly-minted filesystem/sockets
        // futures from the P3 adapter. (#564.)
        if (handle != 0) {
            if (ctx.comp_inst.futures.getPtr(handle)) |fut| {
                if (!fut.subtask_managed) fut.subtask_managed = true;

                // For the with-result async-lower shape, write the
                // future's pre-lowered canonical-ABI bytes to the
                // caller's retptr. This is the sync-completion fast
                // path used by every current `wasi:filesystem` /
                // `wasi:sockets` P3 adapter (each `[method]descriptor.*`
                // and the sockets-P3 `connect` / `bind` / etc. mint
                // already-`.ready` futures via `spawnReadyFsFuture` /
                // `socketReadyResultFuture`). Pending futures leave
                // retptr untouched here; the deferred-completion path
                // is wired separately via `Future.async_lower_retptr`
                // and the relevant settle hooks.
                if (async_with_result_retptr) {
                    if (fut.state == .ready or fut.state == .closed) {
                        if (fut.payload) |bytes| {
                            const mem_idx = ctx.lower_opts.memory_idx.?;
                            const mem = ctx.comp_inst.resolveTopLevelMemory(mem_idx) orelse
                                return trampolineTrap(env, ctx, error.MemoryNotAvailable, .memory_resolve);
                            if (@as(u64, result_dest_ptr) + bytes.len > mem.data.len) {
                                return trampolineTrap(env, ctx, error.MemoryNotAvailable, .lower_results);
                            }
                            @memcpy(mem.data[result_dest_ptr .. result_dest_ptr + bytes.len], bytes);
                        }
                    } else {
                        // Pending: stash retptr so the host settle path
                        // can complete the lower-results copy on
                        // future-state transition. (#564 forward-compat
                        // for genuinely-async host bodies.)
                        fut.async_lower_retptr = result_dest_ptr;
                    }
                }
            }
        }

        const packed_status = packAsyncLowerStatus(ctx.comp_inst, handle);
        env.pushI32(@bitCast(packed_status)) catch |err| {
            return trampolineTrap(env, ctx, err, .lower_results);
        };
        return;
    }

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
            pushInterfaceValue(&frame, r, t, registry) catch |err| {
                return trampolineTrap(env, ctx, err, .lower_results);
            };
        }
    }
}

pub export fn wamrAotDispatchComponentTrampoline(
    ctx_opaque: *anyopaque,
    lowered_sig: *const host_trampolines.LoweredSig,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    a7: u64,
    a8: u64,
) callconv(.c) host_trampolines.DispatchResult {
    const ctx: *const ComponentTrampolineCtx = @ptrCast(@alignCast(ctx_opaque));
    const result = dispatchAotComponentTrampoline(ctx, lowered_sig.*, .{ a0, a1, a2, a3, a4, a5, a6, a7, a8 }) catch |err| {
        if (debugAotEnabled()) {
            std.debug.print(
                "[aot-dispatch] canon.lower trampoline failed: {s}\n",
                .{@errorName(err)},
            );
        }
        return .{ .status = 1, .value = 0 };
    };
    return .{ .status = 0, .value = result };
}

/// AOT-codegen-flavoured canon.lower dispatcher (#687). The trampoline pool
/// stub shifts caller regs right by one to inject `slot` as the first
/// C-ABI arg, so when the AOT codegen calls a host import as
/// `host_fn(vmctx, arg0, arg1, …)`, this dispatcher receives
/// `(slot, a0=vmctx, a1=arg0, a2=arg1, …, a8=arg7)`. We discard `a0`
/// (importer's vmctx) and re-issue `dispatchAotComponentTrampoline` over
/// `a1..a8`, matching the lowered wasm-arg shape the host trampoline
/// already expects. Widened from a0..a5 in #689 so WASIp2 filesystem
/// methods (`link-at` = 7 wasm params, etc.) fit.
pub export fn wamrAotDispatchComponentTrampolineAot(
    ctx_opaque: *anyopaque,
    lowered_sig: *const host_trampolines.LoweredSig,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    a7: u64,
    a8: u64,
) callconv(.c) host_trampolines.DispatchResult {
    _ = a0;
    const ctx: *const ComponentTrampolineCtx = @ptrCast(@alignCast(ctx_opaque));
    const result = dispatchAotComponentTrampoline(ctx, lowered_sig.*, .{ a1, a2, a3, a4, a5, a6, a7, a8, 0 }) catch |err| {
        if (debugAotEnabled()) {
            std.debug.print(
                "[aot-dispatch] canon.lower(aot) trampoline failed: {s}\n",
                .{@errorName(err)},
            );
        }
        return .{ .status = 1, .value = 0 };
    };
    return .{ .status = 0, .value = result };
}

/// Per-import context for the cross-instance core-to-core thunk (#662).
/// Owned by `ComponentInstance.cross_instance_thunk_ctxs`; the trampoline
/// pool slot stores `*CrossInstanceThunkCtx` as its `ctx`.
pub const CrossInstanceThunkCtx = struct {
    /// Sibling AOT instance whose `func_idx` we will re-issue the call into.
    target_ai: *aot_runtime.AotInstance,
    /// Index into the sibling's function-index space.
    target_func_idx: u32,
    /// Cached param-type slice (core ValTypes) for `callFuncScalar`. Slices
    /// are owned by the component-instance arena.
    param_types: []const core_types.ValType,
    result_types: []const core_types.ValType,
    /// Human-readable label used in trap-shaped error messages.
    label: []const u8,
};

/// Cross-instance core-to-core dispatcher (#662). The trampoline pool stub
/// shifts caller regs right by one to inject `slot` as the first C-ABI arg,
/// so when the AOT codegen calls a host import as
/// `host_fn(vmctx, arg0, arg1, …)`, the dispatcher receives
/// `(slot, a0=vmctx, a1=arg0, a2=arg1, …, a8=arg7)`. We ignore `a0`
/// (importer's vmctx — the sibling AotInstance builds its own vmctx
/// internally in `callFuncScalar`) and use `a1..a8` as lowered wasm args.
/// Calls outside the trampoline pool's 8-arg-in-regs envelope return a
/// failing status; richer signatures land in a follow-up. The 5 → 8 widening
/// in #689 covers WASIp2 filesystem methods like `link-at` (7 wasm params).
pub export fn wamrAotDispatchCrossInstance(
    ctx_opaque: *anyopaque,
    lowered_sig: *const host_trampolines.LoweredSig,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    a7: u64,
    a8: u64,
) callconv(.c) host_trampolines.DispatchResult {
    _ = a0; // importer's vmctx; the sibling AotInstance builds its own.
    const ctx: *const CrossInstanceThunkCtx = @ptrCast(@alignCast(ctx_opaque));
    const result = dispatchAotCrossInstance(ctx, lowered_sig.*, .{ a1, a2, a3, a4, a5, a6, a7, a8, 0 }) catch |err| {
        if (debugAotEnabled()) {
            std.debug.print(
                "[aot-dispatch] cross-instance thunk '{s}' failed: {s}\n",
                .{ ctx.label, @errorName(err) },
            );
        }
        return .{ .status = 1, .value = 0 };
    };
    return .{ .status = 0, .value = result };
}

fn dispatchAotCrossInstance(
    ctx: *const CrossInstanceThunkCtx,
    lowered_sig: host_trampolines.LoweredSig,
    arg_regs: [9]u64,
) !u64 {
    // We support up to 8 args in registers (a1..a8 in the dispatcher's
    // frame, since a0 is the importer's vmctx). Spilled-arg / spilled-result
    // shapes route through the lift trampoline pathway, not this one.
    if (lowered_sig.has_retptr) return error.UnsupportedSignature;
    if (lowered_sig.param_types.len > 8) return error.UnsupportedSignature;
    if (lowered_sig.result_types.len > 1) return error.UnsupportedSignature;
    if (lowered_sig.param_types.len != ctx.param_types.len) return error.UnsupportedSignature;
    if (lowered_sig.result_types.len != ctx.result_types.len) return error.UnsupportedSignature;
    for (lowered_sig.param_types, ctx.param_types) |a, b| if (a != b) return error.UnsupportedSignature;
    for (lowered_sig.result_types, ctx.result_types) |a, b| if (a != b) return error.UnsupportedSignature;

    var args_buf: [8]core_types.Value = undefined;
    for (ctx.param_types, 0..) |pt, i| {
        args_buf[i] = switch (pt) {
            .i32 => .{ .i32 = @bitCast(@as(u32, @truncate(arg_regs[i]))) },
            .i64 => .{ .i64 = @bitCast(arg_regs[i]) },
            .f32 => .{ .f32 = @bitCast(@as(u32, @truncate(arg_regs[i]))) },
            .f64 => .{ .f64 = @bitCast(arg_regs[i]) },
            else => return error.UnsupportedSignature,
        };
    }

    var results_buf: [1]aot_runtime.ScalarResult = .{.{ .i32 = 0 }};
    const results = aot_runtime.callFuncScalar(
        ctx.target_ai,
        ctx.target_func_idx,
        ctx.param_types,
        ctx.result_types,
        args_buf[0..ctx.param_types.len],
        &results_buf,
    ) catch return error.TrapInCoreFunction;

    if (ctx.result_types.len == 0) return 0;
    return switch (ctx.result_types[0]) {
        .i32 => @as(u64, @intCast(@as(u32, @bitCast(results[0].i32)))),
        .i64 => @bitCast(results[0].i64),
        .f32 => @as(u64, @intCast(@as(u32, @bitCast(results[0].f32)))),
        .f64 => @bitCast(results[0].f64),
        else => error.UnsupportedSignature,
    };
}

/// Trap-on-call stub dispatcher (#662 follow-up). The trampoline pool
/// installs this dispatcher for non-WASI fn imports the AOT bridge has
/// no wiring for yet (canon.lower / canon-builtin). Instantiation
/// succeeds, but any actual call returns a failing `DispatchResult` so
/// the AOT codegen's host-call site sees a clean trap rather than a
/// segfault through a null host slot.
pub export fn wamrAotDispatchTrapStub(
    ctx_opaque: *anyopaque,
    lowered_sig: *const host_trampolines.LoweredSig,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    a7: u64,
    a8: u64,
) callconv(.c) host_trampolines.DispatchResult {
    _ = lowered_sig;
    _ = a0;
    _ = a1;
    _ = a2;
    _ = a3;
    _ = a4;
    _ = a5;
    _ = a6;
    _ = a7;
    _ = a8;
    if (debugAotEnabled()) {
        const label_ptr: *const [*:0]const u8 = @ptrCast(@alignCast(ctx_opaque));
        std.debug.print("[aot-dispatch] trap-stub fired for unbridged import '{s}' (#662 follow-up)\n", .{label_ptr.*});
    }
    return .{ .status = 1, .value = 0 };
}


fn dispatchAotComponentTrampoline(
    ctx: *const ComponentTrampolineCtx,
    lowered_sig: host_trampolines.LoweredSig,
    regs: [9]u64,
) !u64 {
    if (ctx.lower_opts.is_async or ctx.is_async_func) return error.UnsupportedSignature;
    if (lowered_sig.param_types.len + @intFromBool(lowered_sig.has_retptr) > regs.len)
        return error.UnsupportedSignature;
    if (lowered_sig.has_retptr and lowered_sig.result_types.len != 0)
        return error.UnsupportedSignature;

    const allocator = ctx.comp_inst.allocator;
    const registry = if (ctx.extended_types.len > 0)
        TypeRegistry.fromExtended(ctx.comp_inst.component, ctx.extended_types, ctx.extended_indexspace)
    else
        TypeRegistry.init(ctx.comp_inst.component);

    if (countFlatTypes(registry, ctx.param_types) > MAX_FLAT_PARAMS) return error.UnsupportedSignature;

    var args_stack_buf: [8]InterfaceValue = undefined;
    const args_heap: ?[]InterfaceValue = if (ctx.param_types.len <= args_stack_buf.len)
        null
    else
        try allocator.alloc(InterfaceValue, ctx.param_types.len);
    defer if (args_heap) |h| allocator.free(h);
    const args: []InterfaceValue = args_heap orelse args_stack_buf[0..ctx.param_types.len];

    var reg_index: usize = 0;
    for (ctx.param_types, 0..) |pt, i| {
        args[i] = try liftAotDispatcherArg(pt, lowered_sig.param_types, &reg_index, &regs, registry, allocator);
    }

    var result_dest_ptr: u32 = 0;
    if (lowered_sig.has_retptr) {
        if (reg_index >= regs.len) return error.UnsupportedSignature;
        result_dest_ptr = @truncate(regs[reg_index]);
        reg_index += 1;
    }
    if (reg_index != lowered_sig.param_types.len + @intFromBool(lowered_sig.has_retptr))
        return error.UnsupportedSignature;

    var results_stack_buf: [4]InterfaceValue = undefined;
    const results_heap: ?[]InterfaceValue = if (ctx.result_types.len <= results_stack_buf.len)
        null
    else
        try allocator.alloc(InterfaceValue, ctx.result_types.len);
    const results: []InterfaceValue = results_heap orelse results_stack_buf[0..ctx.result_types.len];
    defer {
        for (results) |r| r.deinit(allocator);
        if (results_heap) |h| allocator.free(h);
    }

    if (ctx.lift_target) |target| {
        try callComponentFuncByLocal(ctx.comp_inst, target, args, results, allocator);
    } else {
        const call = ctx.host_func.call orelse return error.HostFuncNotBound;
        try call(ctx.host_func.context, ctx.comp_inst, args, results, allocator);
    }

    if (lowered_sig.has_retptr) {
        const mem_idx = ctx.lower_opts.memory_idx orelse return error.MemoryNotAvailable;
        const mem = ctx.comp_inst.resolveTopLevelMemory(mem_idx) orelse return error.MemoryNotAvailable;
        var offset: u32 = result_dest_ptr;
        for (results, ctx.result_types) |r, t| {
            const al = typeAlign(registry, t);
            offset = abi.alignUp(offset, al);
            storeInterfaceValue(mem.data, offset, r, t, registry);
            offset += typeSize(registry, t);
        }
        return 0;
    }

    if (ctx.result_types.len == 0) {
        if (lowered_sig.result_types.len != 0) return error.UnsupportedSignature;
        return 0;
    }
    if (ctx.result_types.len != 1 or lowered_sig.result_types.len != 1)
        return error.UnsupportedSignature;
    return lowerAotDispatcherResult(results[0], ctx.result_types[0], lowered_sig.result_types[0], registry);
}

fn liftAotDispatcherArg(
    t: ctypes.ValType,
    lowered_param_types: []const core_types.ValType,
    reg_index: *usize,
    regs: *const [9]u64,
    registry: TypeRegistry,
    allocator: Allocator,
) !InterfaceValue {
    switch (t) {
        .bool, .s8, .u8, .s16, .u16, .s32, .u32, .char, .own, .borrow, .future, .stream, .error_context, .enum_ => {
            if (reg_index.* >= lowered_param_types.len or lowered_param_types[reg_index.*] != .i32)
                return error.UnsupportedSignature;
            const slots = [_]u32{@truncate(regs[reg_index.*])};
            reg_index.* += 1;
            return abi.liftFlatReg(slots[0..], t, registry, allocator);
        },
        .s64, .u64 => {
            if (reg_index.* >= lowered_param_types.len or lowered_param_types[reg_index.*] != .i64)
                return error.UnsupportedSignature;
            const raw = regs[reg_index.*];
            const slots = [_]u32{ @truncate(raw), @truncate(raw >> 32) };
            reg_index.* += 1;
            return abi.liftFlatReg(slots[0..], t, registry, allocator);
        },
        .string, .list => {
            if (reg_index.* + 1 >= lowered_param_types.len or
                lowered_param_types[reg_index.*] != .i32 or
                lowered_param_types[reg_index.* + 1] != .i32)
                return error.UnsupportedSignature;
            const slots = [_]u32{
                @truncate(regs[reg_index.*]),
                @truncate(regs[reg_index.* + 1]),
            };
            reg_index.* += 2;
            return abi.liftFlatReg(slots[0..], t, registry, allocator);
        },
        else => return error.UnsupportedSignature,
    }
}

fn lowerAotDispatcherResult(
    val: InterfaceValue,
    t: ctypes.ValType,
    lowered_ty: core_types.ValType,
    registry: TypeRegistry,
) !u64 {
    if (lowered_ty == .f32 or lowered_ty == .f64) return error.UnsupportedSignature;
    // Single-flat-slot interface results only — multi-slot shapes
    // (string/list/multi-slot compounds) spill via the retptr path.
    const lifted_core = coreFlatSlotType(t, registry) catch return error.UnsupportedSignature;
    if (lifted_core != lowered_ty) return error.UnsupportedSignature;
    return switch (t) {
        .bool => @as(u64, @intCast(@as(u32, if (val.bool) 1 else 0))),
        .s8 => @as(u64, @intCast(@as(u32, @bitCast(@as(i32, val.s8))))),
        .u8 => @as(u64, val.u8),
        .s16 => @as(u64, @intCast(@as(u32, @bitCast(@as(i32, val.s16))))),
        .u16 => @as(u64, val.u16),
        .s32 => @as(u64, @intCast(@as(u32, @bitCast(val.s32)))),
        .u32, .char => @as(u64, val.u32),
        .s64 => @as(u64, @bitCast(val.s64)),
        .u64 => val.u64,
        .enum_ => @as(u64, val.enum_val),
        .own, .borrow => @as(u64, encodeResourceWire(val.handle)),
        .future, .stream, .error_context => @as(u64, val.handle),
        .result => @as(u64, if (val.result_val.is_ok) 0 else 1),
        .variant => @as(u64, val.variant_val.discriminant),
        .option => @as(u64, if (val.option_val.is_some) 1 else 0),
        else => error.UnsupportedSignature,
    };
}

/// Async-lower (`canon.lower (async)`) status-word encoding (#551). Given
/// the host-returned waitable handle (currently a `future<()>` handle for
/// `wasi:clocks@0.3.x` `wait-for` / `wait-until`), look up the future's
/// state and return the spec-shaped packed i32:
///
///   * `(0 << 4) | STATUS_RETURNED` — `future.state == .ready` already;
///     the call resolved synchronously (e.g. a `wait-for(0)` /
///     `wait-until(now)` past-deadline shortcut).
///   * `(handle << 4) | STATUS_STARTED` — pending; the guest must wait
///     for completion via `waitable-set.{wait,poll}` after a
///     `waitable.join(handle, ws_handle)`.
///
/// Status-bit encoding mirrors `wit-bindgen ≥ 0.53`'s
/// `crates/guest-rust/src/rt/async_support.rs` constants:
///   `STATUS_STARTING=0`, `STATUS_STARTED=1`, `STATUS_RETURNED=2`,
///   `STATUS_STARTED_CANCELLED=3`, `STATUS_RETURNED_CANCELLED=4`.
pub const STATUS_STARTING: u32 = 0;
pub const STATUS_STARTED: u32 = 1;
pub const STATUS_RETURNED: u32 = 2;
pub const STATUS_STARTED_CANCELLED: u32 = 3;
pub const STATUS_RETURNED_CANCELLED: u32 = 4;

pub fn packAsyncLowerStatus(comp_inst: *const ComponentInstance, handle: u32) u32 {
    if (handle == 0) return STATUS_RETURNED;
    const fut = comp_inst.futures.getPtr(handle) orelse {
        // No future entry — degenerate "already done" (e.g. host returned
        // an unallocated handle); treat as STATUS_RETURNED with no
        // waitable so the guest skips waitable.join.
        return STATUS_RETURNED;
    };
    if (fut.state == .ready or fut.state == .closed) {
        return STATUS_RETURNED;
    }
    return (handle << 4) | STATUS_STARTED;
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

// ── Canon-builtin host trampoline (#520) ────────────────────────────────────
//
// `canon.lower` is one of several canons that contribute to the core-func
// index space; the others are `context.{get,set}`, `task.{yield,return}`,
// `resource.{new,drop,rep}`, and the async ABI builtins (stream/future
// new/read/write/cancel/drop, subtask.{cancel,drop}, waitable-set ops,
// error-context). When a core wasm module imports one of these via the
// component's `(core instance (instantiate $main (with "x" (func $cidx))))`
// wiring, the import must dispatch to the matching canon-builtin semantics
// — not to a host function. `componentTrampoline` only handles canon.lower
// (host-bound funcs); this trampoline routes everything else through
// `dispatchCanonBuiltin`, which already implements the per-canon logic.

/// Per-import-slot context for the canon-builtin trampoline. Stores the
/// canon decl itself plus the owning component instance — enough to call
/// `dispatchCanonBuiltin` with the right state on every invocation.
pub const CanonBuiltinTrampolineCtx = struct {
    comp_inst: *ComponentInstance,
    canon: ctypes.Canon,
    /// Number of flat (typed) wasm params declared by the importing
    /// module for this canon-builtin slot. For `canon task.return`,
    /// this is the count of i32/i64/f32/f64 values the guest pushes
    /// onto the operand stack before invoking the host trampoline —
    /// which is exactly the canon-ABI flat lowering of the result
    /// type. We snapshot this at link time because flattening a
    /// `result<own<response>, error-code>` via `TypeRegistry` requires
    /// the inner variant payload types to be materialised in the
    /// parent component's type pool — the loader only materialises
    /// primitives today, so a runtime-only `flattenCount` would
    /// underflow the stack pop. (#570)
    core_flat_param_count: ?u32 = null,
};

/// Trampoline entry-point installed on a core wasm import that was
/// resolved to a canon builtin (context.{get,set}, task.{yield,return},
/// resource.{new,drop,rep}, async ABI). Routes the call through
/// `dispatchCanonBuiltin`, passing the instance's current TaskManager
/// (set up by `callComponentFuncAsync` for the duration of an
/// async-lifted dispatch; null on the sync-call path).
pub fn canonBuiltinTrampoline(env_opaque: *anyopaque, ctx_opaque: ?*anyopaque) core_runtime_types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const ctx: *CanonBuiltinTrampolineCtx = @ptrCast(@alignCast(ctx_opaque.?));
    dispatchCanonBuiltinWithCtx(
        ctx.comp_inst,
        ctx.canon,
        ctx,
        env,
        ctx.comp_inst.current_task_manager,
        ctx.comp_inst.allocator,
    ) catch |err| {
        env.host_trap = .{
            .component_func_idx = 0,
            .err_name = @errorName(err),
            .stage = .host_call,
        };
        return error.Trap;
    };
}

// ── Trampoline tests ────────────────────────────────────────────────────────

test "componentTrampoline: async-lower no-result (clocks-style) packs (handle << 4) | STATUS_STARTED (#564)" {
    // Regression test for the wasi:clocks `wait-for` / `wait-until`
    // shape — async-lower of a func with zero lifted results. The
    // trampoline allocates a phantom slot for the host's future
    // handle, packs `(handle << 4) | STATUS_*`, and pushes the i32
    // status word in place of the lifted results. Mirrors the
    // pre-#564 wave-3 behaviour; included here as a guard against
    // the generic path regressing the clocks special case.
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    const imports = [_]core_types_mod.ImportDesc{
        .{ .module_name = "host", .field_name = "wait", .kind = .function, .func_type_idx = 0 },
    };
    const func_types = [_]core_types_mod.FuncType{
        .{ .params = &.{}, .results = &.{.i32} },
    };
    var module = core_types_mod.WasmModule{
        .imports = &imports,
        .import_function_count = 1,
        .types = &func_types,
    };
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);

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

    // Allocate a pending future and inject its handle through the
    // host fn's phantom result slot.
    const fh = comp_inst.allocAsyncHandle();
    try comp_inst.futures.put(comp_inst.allocator, fh, .{ .elem_type_idx = 0, .state = .pending });

    const Host = struct {
        var captured_handle: u32 = 0;
        fn wait(
            _: ?*anyopaque,
            _: *ComponentInstance,
            _: []const InterfaceValue,
            out: []InterfaceValue,
            _: Allocator,
        ) anyerror!void {
            out[0] = .{ .handle = captured_handle };
        }
    };
    Host.captured_handle = fh;

    const param_types = try testing.allocator.alloc(ctypes.ValType, 0);
    const result_types = try testing.allocator.alloc(ctypes.ValType, 0);

    var tctx = ComponentTrampolineCtx{
        .comp_inst = comp_inst,
        .host_func = .{ .call = &Host.wait },
        .param_types = param_types,
        .result_types = result_types,
        .lower_opts = .{},
        .is_async_func = true,
    };
    defer tctx.deinit(testing.allocator);

    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();
    try componentTrampoline(env, @ptrCast(&tctx));

    const status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(STATUS_STARTED, status & 0xf);
    try testing.expectEqual(fh, status >> 4);
    // Pending future should now carry `subtask_managed = true` so the
    // guest's subsequent `waitable.join` routes it as a `.subtask`
    // waitable.
    try testing.expect(comp_inst.futures.getPtr(fh).?.subtask_managed);
}

test "componentTrampoline: async-lower with-result spilled-params writes payload to retptr (#564)" {
    // Generic async-lower path with N flat params > MAX_FLAT_PARAMS_ASYNC
    // (so params spill to memory) and one lifted result — the
    // wasi:filesystem `[method]descriptor.open-at`-style shape. The
    // trampoline:
    //   1. Pops retptr from the stack (last core arg pushed by caller).
    //   2. Pops params_ptr (penultimate core arg).
    //   3. Lifts each arg from `mem[params_ptr..]` via canon-ABI layout.
    //   4. Calls host, which deposits a `.ready` future handle in
    //      `results[0]`.
    //   5. Copies `fut.payload` bytes to `mem[retptr..]`.
    //   6. Pushes STATUS_RETURNED as the i32 status word.
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    // 5 flat params (5 × u32) → 5 > 4 (async limit) → caller passes a
    // single i32 params_ptr instead.
    const imports = [_]core_types_mod.ImportDesc{
        .{ .module_name = "host", .field_name = "spilled", .kind = .function, .func_type_idx = 0 },
    };
    const func_types = [_]core_types_mod.FuncType{
        .{ .params = &.{ .i32, .i32 }, .results = &.{.i32} },
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

    // Pre-stage a ready future with a 4-byte payload (the canonical-ABI
    // bytes the trampoline will copy into mem[retptr..]).
    const fh = comp_inst.allocAsyncHandle();
    const payload = try comp_inst.allocator.alloc(u8, 4);
    payload[0] = 0xCA;
    payload[1] = 0xFE;
    payload[2] = 0xBA;
    payload[3] = 0xBE;
    try comp_inst.futures.put(comp_inst.allocator, fh, .{
        .elem_type_idx = 0,
        .payload = payload,
        .state = .ready,
        .write_closed = true,
    });

    const mem = core_inst.getMemory(0).?;
    // Lay out the spilled params: 5 × u32 at offsets 0, 4, 8, 12, 16.
    const params_ptr: u32 = 32;
    std.mem.writeInt(u32, mem.data[params_ptr..][0..4], 0x11111111, .little);
    std.mem.writeInt(u32, mem.data[params_ptr + 4 ..][0..4], 0x22222222, .little);
    std.mem.writeInt(u32, mem.data[params_ptr + 8 ..][0..4], 0x33333333, .little);
    std.mem.writeInt(u32, mem.data[params_ptr + 12 ..][0..4], 0x44444444, .little);
    std.mem.writeInt(u32, mem.data[params_ptr + 16 ..][0..4], 0x55555555, .little);

    const Host = struct {
        var captured_handle: u32 = 0;
        var captured_args: [5]u32 = .{ 0, 0, 0, 0, 0 };
        fn body(
            _: ?*anyopaque,
            _: *ComponentInstance,
            args: []const InterfaceValue,
            out: []InterfaceValue,
            _: Allocator,
        ) anyerror!void {
            for (args, 0..) |a, i| captured_args[i] = a.u32;
            out[0] = .{ .handle = captured_handle };
        }
    };
    Host.captured_handle = fh;
    Host.captured_args = .{ 0, 0, 0, 0, 0 };

    const param_types = try testing.allocator.alloc(ctypes.ValType, 5);
    for (param_types) |*p| p.* = .u32;
    const result_types = try testing.allocator.alloc(ctypes.ValType, 1);
    result_types[0] = .u32;

    var tctx = ComponentTrampolineCtx{
        .comp_inst = comp_inst,
        .host_func = .{ .call = &Host.body },
        .param_types = param_types,
        .result_types = result_types,
        .lower_opts = .{ .memory_idx = 0, .is_async = true },
        .is_async_func = true,
    };
    defer tctx.deinit(testing.allocator);

    const env = try ExecEnv.create(core_inst, 256, testing.allocator);
    defer env.destroy();
    // Caller pushed (params_ptr, retptr) in canonical order — flat=5 > 4
    // so the params are spilled to memory.
    const retptr: u32 = 96;
    try env.pushI32(@bitCast(params_ptr));
    try env.pushI32(@bitCast(retptr));
    try componentTrampoline(env, @ptrCast(&tctx));

    // Host saw all 5 args lifted from mem[params_ptr..].
    try testing.expectEqual(@as(u32, 0x11111111), Host.captured_args[0]);
    try testing.expectEqual(@as(u32, 0x22222222), Host.captured_args[1]);
    try testing.expectEqual(@as(u32, 0x33333333), Host.captured_args[2]);
    try testing.expectEqual(@as(u32, 0x44444444), Host.captured_args[3]);
    try testing.expectEqual(@as(u32, 0x55555555), Host.captured_args[4]);

    // Trampoline pushed STATUS_RETURNED (the future was already ready).
    const status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(STATUS_RETURNED, status);

    // And copied the future's payload to mem[retptr..].
    try testing.expectEqual(@as(u8, 0xCA), mem.data[retptr]);
    try testing.expectEqual(@as(u8, 0xFE), mem.data[retptr + 1]);
    try testing.expectEqual(@as(u8, 0xBA), mem.data[retptr + 2]);
    try testing.expectEqual(@as(u8, 0xBE), mem.data[retptr + 3]);
}

test "componentTrampoline: async-lower resource handle arg uses encodeResourceWire decode (#564)" {
    // The canon-ABI wire format for `own<R>` / `borrow<R>` adds a
    // `slot + 1` offset (PR #560 wave 2) so wit-bindgen's
    // `Resource::from_handle` assertion (`handle != 0 && handle !=
    // u32::MAX`) sees a non-zero wire value. The trampoline lifts the
    // arg via `popInterfaceValue` → `decodeResourceWire`, which undoes
    // the `+1`. Verify a guest pushing wire=1 reaches the host as slot=0.
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    const imports = [_]core_types_mod.ImportDesc{
        .{ .module_name = "host", .field_name = "with_borrow", .kind = .function, .func_type_idx = 0 },
    };
    const func_types = [_]core_types_mod.FuncType{
        // (borrow<R>) → status
        .{ .params = &.{.i32}, .results = &.{.i32} },
    };
    var module = core_types_mod.WasmModule{
        .imports = &imports,
        .import_function_count = 1,
        .types = &func_types,
    };
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);

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

    const fh = comp_inst.allocAsyncHandle();
    try comp_inst.futures.put(comp_inst.allocator, fh, .{ .elem_type_idx = 0, .state = .ready });

    const Host = struct {
        var captured_slot: u32 = 0xFFFF_FFFF;
        var captured_handle: u32 = 0;
        fn body(
            _: ?*anyopaque,
            _: *ComponentInstance,
            args: []const InterfaceValue,
            out: []InterfaceValue,
            _: Allocator,
        ) anyerror!void {
            captured_slot = args[0].handle;
            out[0] = .{ .handle = captured_handle };
        }
    };
    Host.captured_handle = fh;

    const param_types = try testing.allocator.alloc(ctypes.ValType, 1);
    param_types[0] = .{ .borrow = 0 };
    const result_types = try testing.allocator.alloc(ctypes.ValType, 0);

    var tctx = ComponentTrampolineCtx{
        .comp_inst = comp_inst,
        .host_func = .{ .call = &Host.body },
        .param_types = param_types,
        .result_types = result_types,
        .lower_opts = .{},
        .is_async_func = true,
    };
    defer tctx.deinit(testing.allocator);

    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();
    // Guest pushes wire=1 (encoded form of slot=0).
    try env.pushI32(@bitCast(@as(u32, 1)));
    try componentTrampoline(env, @ptrCast(&tctx));

    // Host received the decoded slot=0.
    try testing.expectEqual(@as(u32, 0), Host.captured_slot);
    // Trampoline pushed (fh << 4) | STATUS_RETURNED (the future is .ready).
    const status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(STATUS_RETURNED, status);
}

test "componentTrampoline: async-lower routing fires for FuncType.is_async even without canon-opt (#564)" {
    // wit-bindgen ≥ 0.45 emits `(canon lower ... async)` for async
    // funcs (the `lower_opts.is_async` path), but the FuncType-level
    // `is_async` flag set by the `0x43` functype tag must independently
    // flip the trampoline into the async-lower routing even when the
    // canon decl is missing the explicit `async_lift` opt. Confirm the
    // `ctx.is_async_func` standalone trigger produces the packed
    // status return shape.
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    const imports = [_]core_types_mod.ImportDesc{
        .{ .module_name = "host", .field_name = "f", .kind = .function, .func_type_idx = 0 },
    };
    const func_types = [_]core_types_mod.FuncType{
        .{ .params = &.{}, .results = &.{.i32} },
    };
    var module = core_types_mod.WasmModule{
        .imports = &imports,
        .import_function_count = 1,
        .types = &func_types,
    };
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);

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

    const fh = comp_inst.allocAsyncHandle();
    try comp_inst.futures.put(comp_inst.allocator, fh, .{ .elem_type_idx = 0, .state = .pending });

    const Host = struct {
        var captured_handle: u32 = 0;
        fn body(_: ?*anyopaque, _: *ComponentInstance, _: []const InterfaceValue, out: []InterfaceValue, _: Allocator) anyerror!void {
            out[0] = .{ .handle = captured_handle };
        }
    };
    Host.captured_handle = fh;

    const param_types = try testing.allocator.alloc(ctypes.ValType, 0);
    const result_types = try testing.allocator.alloc(ctypes.ValType, 0);

    // Note: `lower_opts.is_async = false` (no `async_lift` canon-opt).
    // The trampoline should still take the async-lower path because
    // `is_async_func = true` mirrors the loader-level FuncType tag.
    var tctx = ComponentTrampolineCtx{
        .comp_inst = comp_inst,
        .host_func = .{ .call = &Host.body },
        .param_types = param_types,
        .result_types = result_types,
        .lower_opts = .{},
        .is_async_func = true,
    };
    defer tctx.deinit(testing.allocator);

    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();
    try componentTrampoline(env, @ptrCast(&tctx));

    const status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(STATUS_STARTED, status & 0xf);
    try testing.expectEqual(fh, status >> 4);
}

test "canonBuiltinTrampoline: context.{set,get} round-trip through implicit fallback (#520)" {
    // The CLI-side `wamr run` dispatch installs `canonBuiltinTrampoline`
    // on every core import that resolves to a canon builtin (context.set,
    // context.get, task.return, etc.). Confirm a context.set followed by
    // context.get round-trips through `dispatchCanonBuiltin`, which falls
    // back to `comp_inst.implicit_task_context` when no TaskManager is
    // active (sync-call path).
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
    try std.testing.expect(inst.current_task_manager == null);

    var set_ctx = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .context_set = .{ .val_type = .i32, .slot = 0 } },
    };
    var get_ctx = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .context_get = .{ .val_type = .i32, .slot = 0 } },
    };

    try env.pushI32(@bitCast(@as(u32, 0xCAFE_F00D)));
    try canonBuiltinTrampoline(@ptrCast(env), @ptrCast(&set_ctx));
    try testing.expectEqual(@as(u32, 0xCAFE_F00D), inst.implicit_task_context[0]);

    try canonBuiltinTrampoline(@ptrCast(env), @ptrCast(&get_ctx));
    try testing.expectEqual(@as(i32, @bitCast(@as(u32, 0xCAFE_F00D))), try env.popI32());
}

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

    var frame: CallFrame = .{ .interp = InterpFrame.init(env) };
    defer frame.deinit();

    // Push ok arm: should produce [i32 0, i32 0] (zero-filled payload slot).
    try pushInterfaceValue(&frame, .{ .result_val = .{ .is_ok = true, .payload = null } }, t, registry);
    const lifted_ok = try popInterfaceValue(&frame, t, registry, testing.allocator);
    try testing.expect(lifted_ok.result_val.is_ok);

    // Push err arm with payload u32=0xCAFEu: produces [i32 1, i32 0xCAFE].
    const err_payload: InterfaceValue = .{ .u32 = 0xCAFE };
    try pushInterfaceValue(
        &frame,
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

test "popInterfaceValue: lift variant<record<u16, tuple<u8x4>>> from flat stack (#520 wave 2)" {
    // Mirrors the wasi:sockets@0.3.0 `ip-socket-address` ipv4 arm shape
    // sent by wit-bindgen when the guest calls `tcp-socket.bind`.
    // Before wave-2 #520, `popInterfaceValue` returned
    // `error.CompoundNeedsRegistry` for `.variant` / `.record` /
    // `.tuple`, blocking the entire sockets fixture bucket at the
    // canon-lower trampoline boundary.
    const testing = std.testing;
    const core_types_mod = @import("../runtime/common/types.zig");
    const inst_mod_core = @import("../runtime/interpreter/instance.zig");

    const tup_fields = [_]ctypes.ValType{ .u8, .u8, .u8, .u8 };
    const rec_fields = [_]ctypes.Field{
        .{ .name = "port", .type = .u16 },
        .{ .name = "address", .type = .{ .tuple = 0 } },
    };
    const cases = [_]ctypes.Case{
        .{ .name = "ipv4", .type = .{ .record = 1 }, .refines = null },
    };
    const type_defs = [_]ctypes.TypeDef{
        .{ .tuple = .{ .fields = &tup_fields } }, // 0
        .{ .record = .{ .fields = &rec_fields } }, // 1
        .{ .variant = .{ .cases = &cases } }, // 2
    };
    var component = ctypes.Component{
        .core_modules = &.{}, .core_instances = &.{}, .core_types = &.{},
        .components = &.{},   .instances = &.{},      .aliases = &.{},
        .types = &type_defs,  .canons = &.{},
        .imports = &.{},      .exports = &.{},
    };
    const registry = TypeRegistry.init(&component);
    const t: ctypes.ValType = .{ .variant = 2 };

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, testing.allocator);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, testing.allocator);
    defer env.destroy();

    // Flat repr: [disc=0, port=0xBEEF, oct[0]=127, oct[1]=0, oct[2]=0, oct[3]=1].
    try env.pushI32(0);
    try env.pushI32(0xBEEF);
    try env.pushI32(127);
    try env.pushI32(0);
    try env.pushI32(0);
    try env.pushI32(1);

    var frame: CallFrame = .{ .interp = InterpFrame.init(env) };
    defer frame.deinit();

    const lifted = try popInterfaceValue(&frame, t, registry, testing.allocator);
    defer lifted.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 0), lifted.variant_val.discriminant);
    const rec = lifted.variant_val.payload.?;
    try testing.expectEqual(@as(u16, 0xBEEF), rec.record_val[0].u16);
    const oct = rec.record_val[1].tuple_val;
    try testing.expectEqual(@as(u8, 127), oct[0].u8);
    try testing.expectEqual(@as(u8, 0), oct[1].u8);
    try testing.expectEqual(@as(u8, 0), oct[2].u8);
    try testing.expectEqual(@as(u8, 1), oct[3].u8);
}

test "encode/decodeResourceWire round-trip (#520 wave 2)" {
    // Re-export of the canon-ABI helper for use by host code that
    // bypasses the lift/lower layer (e.g. `fsGetDirectories` writing
    // descriptor handles straight into linear memory).
    const testing = std.testing;
    try testing.expectEqual(@as(u32, 1), encodeResourceWire(0));
    try testing.expectEqual(@as(u32, 2), encodeResourceWire(1));
    try testing.expectEqual(@as(u32, 0), decodeResourceWire(1));
    try testing.expectEqual(@as(u32, 0), decodeResourceWire(0));
    try testing.expectEqual(@as(u32, 1), decodeResourceWire(2));
    // Round-trip stability for arbitrary slot indices.
    for (0..100) |i| {
        const slot: u32 = @intCast(i);
        try testing.expectEqual(slot, decodeResourceWire(encodeResourceWire(slot)));
    }
}
