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
const config = @import("../config.zig");
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
// `wasi_cli_adapter` is imported solely for the
// `takePendingWasiExitCode` handoff used by the AOT host-import
// dispatchers below (#760). The two modules are otherwise mutually
// dependent (`wasi_cli_adapter` already imports `executor`), but
// neither side names the other at comptime, so the cyclic `@import`
// resolves cleanly.
const wasi_cli_adapter = @import("wasi_cli_adapter.zig");
const debugAotEnabled = core_backend.debugAotEnabled;
const Allocator = std.mem.Allocator;

const ComponentInstance = instance_mod.ComponentInstance;
const InterfaceValue = abi.InterfaceValue;
const TypeRegistry = abi.TypeRegistry;
const lifted_result_invariant_violated: [:0]const u8 = "lifted-result-invariant-violated";
pub const CallFrame = call_frame_mod.CallFrame;
pub const InterpFrame = call_frame_mod.InterpFrame;
pub const AotFrame = call_frame_mod.AotFrame;

pub const MAX_FLAT_PARAMS: u32 = 16;
pub const MAX_FLAT_RESULTS: u32 = 1;

// ── Core function index newtypes ────────────────────────────────────────────

/// A core-funcidx that lives in the **component**'s function index space
/// (as carried in `canon lift` / `canon lower` opts). Must be translated
/// through `ComponentInstance.resolveTopLevelCoreFuncAny` before it can
/// be invoked on a `CallFrame`. Distinct from `CoreFuncIdxLocal` so the
/// compiler rejects the (#719) bug class where a component-level idx
/// was passed straight through to a frame call. Zero runtime cost.
pub const CoreFuncIdxComponent = enum(u32) {
    _,
    pub inline fn from(raw: u32) CoreFuncIdxComponent {
        return @enumFromInt(raw);
    }
    pub inline fn value(self: CoreFuncIdxComponent) u32 {
        return @intFromEnum(self);
    }
};

/// A core-funcidx in the **owning module instance**'s local function
/// index space — directly callable via `CallFrame.realloc` /
/// `CallFrame.executeCore`. Produced by translating a
/// `CoreFuncIdxComponent` through `resolveTopLevelCoreFuncAny`, or
/// minted by lookups like `aot_runtime.findExportFunc` /
/// `ModuleInstance.getExportFunc`. Re-exported from `call_frame.zig`
/// where the frame methods consume it.
pub const CoreFuncIdxLocal = call_frame_mod.CoreFuncIdxLocal;

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
    /// Component-level core-funcidx. Translate via
    /// `ComponentInstance.resolveTopLevelCoreFuncAny` before calling.
    realloc_idx: ?CoreFuncIdxComponent = null,
    /// Component-level core-funcidx. Translate via
    /// `ComponentInstance.resolveTopLevelCoreFuncAny` before calling.
    post_return_idx: ?CoreFuncIdxComponent = null,
    string_encoding: ctypes.StringEncoding = .utf8,
    /// Whether this lift uses the async ABI (Binary.md `canonopt 0x06`).
    /// Async lifts return a packed status immediately; results are
    /// delivered via `task.return`. (#478 sub-PR 2.)
    is_async: bool = false,
    /// Optional resumption callback core funcidx (Binary.md `canonopt 0x07`).
    /// Only meaningful when `is_async`; the single-threaded poll-cycle
    /// dispatcher invokes it once after each yield. (#478 sub-PR 2.)
    callback_idx: ?CoreFuncIdxComponent = null,

    pub fn fromOpts(opts: []const ctypes.CanonOpt) LiftOptions {
        var lo = LiftOptions{};
        for (opts) |opt| {
            switch (opt) {
                .memory => |idx| lo.memory_idx = idx,
                .realloc => |idx| lo.realloc_idx = CoreFuncIdxComponent.from(idx),
                .post_return => |idx| lo.post_return_idx = CoreFuncIdxComponent.from(idx),
                .string_encoding => |enc| lo.string_encoding = enc,
                .async_lift => lo.is_async = true,
                .callback => |idx| lo.callback_idx = CoreFuncIdxComponent.from(idx),
            }
        }
        return lo;
    }
};

// ── Realloc ─────────────────────────────────────────────────────────────────

/// Call the core module's realloc function: (old_ptr, old_size, align, new_size) -> ptr.
pub fn callRealloc(
    frame: *CallFrame,
    realloc_idx: CoreFuncIdxLocal,
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
///
/// `param_types` / `result_types` snapshot the **child-side** func
/// signature so the cross-memory marshaler can drive its per-element
/// list walks (#726). Both slices are owned by `child.allocator`; free
/// them via `deinit`. `registry` is a value-type view over
/// `child.component` and outlives the ctx.
pub const ForwardingHostFnCtx = struct {
    owner: *const ComponentInstance,
    local: ComponentInstance.ExportedFunc.Local,
    param_types: []const ctypes.ValType = &.{},
    result_types: []const ctypes.ValType = &.{},
    registry: TypeRegistry = .{ .types = &.{} },
    /// Per-trampoline extended TypeRegistry slot table. Used when the
    /// FuncType being forwarded was lifted out of an instance-type body
    /// whose nested `.record`/`.result`/etc. references use indices
    /// LOCAL to that body's decl space. Empty for top-level imports.
    /// Allocated by `buildInstanceTypeExtension`; freed in `deinit`
    /// with a deep walk of `extended_types`.
    extended_types: []const ctypes.TypeDef = &.{},
    extended_indexspace: []const ?u32 = &.{},

    pub fn deinit(self: *ForwardingHostFnCtx, allocator: Allocator) void {
        if (self.param_types.len > 0) allocator.free(self.param_types);
        if (self.result_types.len > 0) allocator.free(self.result_types);
        if (self.extended_types.len > 0) {
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

/// Build a `ForwardingHostFnCtx` populated with the child-side func
/// signature so the cross-memory marshaler can drive typed walks of
/// `list<compound>` payloads (#726). Caller owns the returned pointer
/// and must release via `destroyForwardingHostFnCtx`.
pub fn buildForwardingHostFnCtx(
    allocator: Allocator,
    owner: *const ComponentInstance,
    local: ComponentInstance.ExportedFunc.Local,
    ft: ctypes.FuncType,
    registry: TypeRegistry,
) !*ForwardingHostFnCtx {
    const ctx = try allocator.create(ForwardingHostFnCtx);
    errdefer allocator.destroy(ctx);
    const params = try getParamValTypes(ft, allocator);
    errdefer allocator.free(params);
    const results = try getResultValTypes(ft, allocator);
    errdefer allocator.free(results);
    ctx.* = .{
        .owner = owner,
        .local = local,
        .param_types = params,
        .result_types = results,
        .registry = registry,
    };
    return ctx;
}

pub fn destroyForwardingHostFnCtx(ctx: *ForwardingHostFnCtx, allocator: Allocator) void {
    ctx.deinit(allocator);
    allocator.destroy(ctx);
}

/// Like `buildForwardingHostFnCtx`, but takes pre-rewritten ValType
/// slices for `param_types` / `result_types` plus an instance-type
/// extension. Used when the FuncType being forwarded was lifted from
/// an instance-type body whose local type indexspace must be merged
/// into the registry via `TypeRegistry.fromExtended` so canon-ABI
/// resolution sees through `.result` / `.record` / etc. references.
///
/// Ownership: the ctx takes ownership of all four passed slices and
/// frees them via `deinit`.
pub fn buildForwardingHostFnCtxWithExtension(
    allocator: Allocator,
    owner: *const ComponentInstance,
    local: ComponentInstance.ExportedFunc.Local,
    param_types: []const ctypes.ValType,
    result_types: []const ctypes.ValType,
    registry: TypeRegistry,
    extended_types: []const ctypes.TypeDef,
    extended_indexspace: []const ?u32,
) !*ForwardingHostFnCtx {
    const ctx = try allocator.create(ForwardingHostFnCtx);
    ctx.* = .{
        .owner = owner,
        .local = local,
        .param_types = param_types,
        .result_types = result_types,
        .registry = registry,
        .extended_types = extended_types,
        .extended_indexspace = extended_indexspace,
    };
    return ctx;
}

/// `HostFunc.call` adapter for forwarding contexts. Ignores the
/// trampoline's `ci` (the child whose core call invoked the import)
/// and dispatches against the recorded owner so the canonical-ABI
/// lift uses the owner's type registry and core instances.
pub fn forwardingHostFnCall(
    ctx_opaque: ?*anyopaque,
    ci: *ComponentInstance,
    args: []const InterfaceValue,
    out_results: []InterfaceValue,
    allocator: Allocator,
) anyerror!void {
    const ctx: *const ForwardingHostFnCtx = @ptrCast(@alignCast(ctx_opaque orelse return error.FunctionNotFound));
    // Cross-component forwarding (issue #719): string/list args lifted
    // from the CALLER's guest memory carry (ptr, len) tuples whose
    // `ptr` is only valid in the caller's address space. The callee's
    // `canon lift` pushes those slots straight onto its own stack, so
    // without an explicit copy step the callee would dereference a
    // caller-memory pointer that is out of bounds (or worse: points
    // at unrelated data) inside its own linear memory and trap.
    //
    // Materialise top-level `string` / `list<u8>` args into the
    // callee's guest memory before forwarding: read source bytes
    // through `ci.readGuestBytes`, allocate destination bytes via
    // `ctx.owner.hostAllocGuest` (which routes through the callee's
    // `cabi_realloc`), copy, and rewrite the PtrLen.
    //
    // Recurses through compound shapes (option / result / variant /
    // record / tuple) so a `result<string, error-code>` or a record
    // field of type `string` is translated, not just top-level
    // `.string` / `.list` args. The rewritten tree is owned by a
    // single arena so cleanup is `arena.deinit()`. (#719 Bug B path 2.)
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const have_types = ctx.param_types.len == args.len;
    var needs_rewrite = false;
    if (have_types) {
        for (ctx.param_types) |pt| {
            if (valTypeHasPtrLen(ctx.registry, pt)) {
                needs_rewrite = true;
                break;
            }
        }
    } else {
        for (args) |a| {
            if (interfaceValueContainsPtrLen(a)) {
                needs_rewrite = true;
                break;
            }
        }
    }
    var effective_args: []const InterfaceValue = args;
    if (needs_rewrite and ctx.owner != ci) {
        const owner_mut: *ComponentInstance = @constCast(ctx.owner);
        const arena_alloc = arena.allocator();
        const buf = try arena_alloc.alloc(InterfaceValue, args.len);
        for (args, 0..) |a, i| {
            buf[i] = if (have_types)
                try marshalValueAcrossMemoryTyped(ci, owner_mut, a, ctx.param_types[i], ctx.registry, arena_alloc)
            else
                try marshalValueAcrossMemory(ci, owner_mut, a, arena_alloc);
        }
        effective_args = buf;
    }
    try callComponentFuncByLocal(ctx.owner, ctx.local, effective_args, out_results, allocator);
    if (have_types) {
        try rewriteResultsBackToCallerTyped(ci, @constCast(ctx.owner), out_results, ctx.result_types, ctx.registry, allocator);
    } else {
        try rewriteResultsBackToCaller(ci, @constCast(ctx.owner), out_results, allocator);
    }
}

/// Mirror of the arg-side marshaling block in `forwardingHostFnCall`,
/// applied to the results returned by `callComponentFuncByLocal`.
///
/// Background (#719 Bug B, Path 3): a sibling/parent-component's lifted
/// callee returns `out_results` whose `.string` / `.list` PtrLens point
/// into the CALLEE's linear memory. When control returns to the child
/// component that originally invoked the import, the caller's
/// trampoline lowers those PtrLens back onto its own core stack (or
/// stores them at `retptr` in its own linear memory). At that point
/// the i32 ptr is reinterpreted at the same numeric offset inside the
/// CALLER's memory — wrong memory, silent corruption (or a clean
/// `LiftedResultInvariantViolated` trap when `validateCanonPtrLenValue`
/// catches it).
///
/// This helper walks the returned tree and translates every nested
/// PtrLen from callee-memory (`callee`) back into caller-memory
/// (`caller`) via `caller.hostAllocGuest` + bytewise copy. After it
/// runs, the caller's downstream lowering writes pointers that are
/// valid in its own memory.
///
/// Bytes copied during arg-side marshaling (`marshalValueAcrossMemory`
/// allocated into callee memory) and any extra callee-side allocations
/// done by the lifted call body remain live in the callee — there is
/// no general way to free them here without a per-component
/// post-return hook. This matches the pre-existing arg-side
/// asymmetry: cross-component byte-copies leak into the dst memory
/// until that core instance is torn down. Out of scope for this PR;
/// tracked separately.
fn rewriteResultsBackToCaller(
    caller: *ComponentInstance,
    callee: *ComponentInstance,
    out_results: []InterfaceValue,
    allocator: Allocator,
) anyerror!void {
    if (caller == callee) return;
    var needs_rewrite = false;
    for (out_results) |r| {
        if (interfaceValueContainsPtrLen(r)) {
            needs_rewrite = true;
            break;
        }
    }
    if (!needs_rewrite) return;

    // The translated tree is owned by an arena that the caller's
    // downstream lowering does NOT free (it only reads the
    // InterfaceValue's flat tags). We have to keep that storage alive
    // for the duration of the caller's lowering. The simplest contract
    // that doesn't require changing `out_results`' allocator is: free
    // any compound payloads previously allocated against `allocator`
    // (via `InterfaceValue.deinit`), then overwrite each slot in place
    // with a fresh deep-copy whose compound payloads are re-allocated
    // against `allocator`. This keeps the existing deinit-by-allocator
    // ownership contract intact at every call site.
    for (out_results) |*slot| {
        const original = slot.*;
        const translated = try rewriteValueBackToCallerOwned(caller, callee, original, allocator);
        original.deinit(allocator);
        slot.* = translated;
    }
}

/// Typed companion to `rewriteResultsBackToCaller`. Used when
/// `forwardingHostFnCall` has populated `ctx.result_types` so
/// `list<compound>` results can be element-walked per #726.
fn rewriteResultsBackToCallerTyped(
    caller: *ComponentInstance,
    callee: *ComponentInstance,
    out_results: []InterfaceValue,
    result_types: []const ctypes.ValType,
    reg: TypeRegistry,
    allocator: Allocator,
) anyerror!void {
    if (caller == callee) return;
    var needs_rewrite = false;
    for (result_types) |rt| {
        if (valTypeHasPtrLen(reg, rt)) {
            needs_rewrite = true;
            break;
        }
    }
    if (!needs_rewrite) return;
    const n = @min(out_results.len, result_types.len);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const original = out_results[i];
        const translated = try rewriteValueBackToCallerOwnedTyped(caller, callee, original, result_types[i], reg, allocator);
        original.deinit(allocator);
        out_results[i] = translated;
    }
}

/// Deep-copy `val` and translate every nested `.string` / `.list`
/// PtrLen from `callee` to `caller`. The returned tree owns its
/// allocations via `allocator` so the caller's existing
/// `InterfaceValue.deinit` cleanup contract continues to hold.
fn rewriteValueBackToCallerOwned(
    caller: *ComponentInstance,
    callee: *ComponentInstance,
    val: InterfaceValue,
    allocator: Allocator,
) anyerror!InterfaceValue {
    switch (val) {
        .string => |pl| {
            if (pl.len == 0) return .{ .string = .{ .ptr = 0, .len = 0 } };
            const bytes = callee.readGuestBytes(pl.ptr, pl.len) orelse
                return error.MemoryNotAvailable;
            const new_ptr = caller.hostAllocGuest(pl.len, 1) orelse
                return error.ReallocFailed;
            const dest = caller.writableGuestBytes(new_ptr, pl.len) orelse
                return error.MemoryNotAvailable;
            @memcpy(dest, bytes);
            return .{ .string = .{ .ptr = new_ptr, .len = pl.len } };
        },
        .list => |pl| {
            if (pl.len == 0) return .{ .list = .{ .ptr = 0, .len = 0 } };
            const bytes = callee.readGuestBytes(pl.ptr, pl.len) orelse
                return error.MemoryNotAvailable;
            const new_ptr = caller.hostAllocGuest(pl.len, 1) orelse
                return error.ReallocFailed;
            const dest = caller.writableGuestBytes(new_ptr, pl.len) orelse
                return error.MemoryNotAvailable;
            @memcpy(dest, bytes);
            return .{ .list = .{ .ptr = new_ptr, .len = pl.len } };
        },
        .record_val => |fields| {
            const out = try allocator.alloc(InterfaceValue, fields.len);
            errdefer allocator.free(out);
            var produced: usize = 0;
            errdefer for (out[0..produced]) |o| o.deinit(allocator);
            for (fields, 0..) |f, i| {
                out[i] = try rewriteValueBackToCallerOwned(caller, callee, f, allocator);
                produced = i + 1;
            }
            return .{ .record_val = out };
        },
        .tuple_val => |fields| {
            const out = try allocator.alloc(InterfaceValue, fields.len);
            errdefer allocator.free(out);
            var produced: usize = 0;
            errdefer for (out[0..produced]) |o| o.deinit(allocator);
            for (fields, 0..) |f, i| {
                out[i] = try rewriteValueBackToCallerOwned(caller, callee, f, allocator);
                produced = i + 1;
            }
            return .{ .tuple_val = out };
        },
        .variant_val => |v| {
            const new_payload: ?*const InterfaceValue = if (v.payload) |p| blk: {
                const nv = try allocator.create(InterfaceValue);
                errdefer allocator.destroy(nv);
                nv.* = try rewriteValueBackToCallerOwned(caller, callee, p.*, allocator);
                break :blk nv;
            } else null;
            return .{ .variant_val = .{ .discriminant = v.discriminant, .payload = new_payload } };
        },
        .option_val => |o| {
            const new_payload: ?*const InterfaceValue = if (o.payload) |p| blk: {
                const nv = try allocator.create(InterfaceValue);
                errdefer allocator.destroy(nv);
                nv.* = try rewriteValueBackToCallerOwned(caller, callee, p.*, allocator);
                break :blk nv;
            } else null;
            return .{ .option_val = .{ .is_some = o.is_some, .payload = new_payload } };
        },
        .result_val => |r| {
            const new_payload: ?*const InterfaceValue = if (r.payload) |p| blk: {
                const nv = try allocator.create(InterfaceValue);
                errdefer allocator.destroy(nv);
                nv.* = try rewriteValueBackToCallerOwned(caller, callee, p.*, allocator);
                break :blk nv;
            } else null;
            return .{ .result_val = .{ .is_ok = r.is_ok, .payload = new_payload } };
        },
        .flags_val => |words| {
            const copy = try allocator.alloc(u32, words.len);
            @memcpy(copy, words);
            return .{ .flags_val = copy };
        },
        // Pure scalars / handles / enums: no nested PtrLen, no owned heap.
        else => return val,
    }
}

/// Cheap pre-flight: does `val` contain any `.string` / `.list` PtrLen,
/// possibly nested inside compound shapes? Used by
/// `forwardingHostFnCall` to skip the marshaling arena when there is
/// nothing to translate (pure-scalar sigs are the common case).
fn interfaceValueContainsPtrLen(val: InterfaceValue) bool {
    return switch (val) {
        .string, .list => true,
        .record_val => |fields| blk: {
            for (fields) |f| if (interfaceValueContainsPtrLen(f)) break :blk true;
            break :blk false;
        },
        .tuple_val => |fields| blk: {
            for (fields) |f| if (interfaceValueContainsPtrLen(f)) break :blk true;
            break :blk false;
        },
        .variant_val => |v| if (v.payload) |p| interfaceValueContainsPtrLen(p.*) else false,
        .option_val => |o| if (o.payload) |p| interfaceValueContainsPtrLen(p.*) else false,
        .result_val => |r| if (r.payload) |p| interfaceValueContainsPtrLen(p.*) else false,
        else => false,
    };
}

/// Recursively materialise every `.string` / `.list` PtrLen in `val`
/// into `dst`-side guest memory, copying source bytes through `src`.
/// Compound shapes (record / tuple / variant / option / result) are
/// walked so nested PtrLens — `option<string>`, `result<string, e>`,
/// `record { name: string }`, etc. — are translated too. The
/// rewritten tree is allocated inside `arena`; on success the caller
/// hands it to the callee and frees the entire tree by tearing the
/// arena down.
///
/// Limitations:
/// * `.list` is byte-copied opaque (no element walk). Lists whose
///   element type itself contains a PtrLen (e.g. `list<string>`,
///   `list<list<u8>>`) will leave their inner pointers pointing into
///   the source memory. This matches the pre-existing
///   `forwardingHostFnCall` behaviour for top-level lists and the
///   limitation that prompted that helper's "list<u8> only" comment.
///   Element-size-aware re-lifting would require type info from the
///   FuncType — out of scope for this helper, which intentionally
///   operates on the InterfaceValue tree alone.
fn marshalValueAcrossMemory(
    src: *ComponentInstance,
    dst: *ComponentInstance,
    val: InterfaceValue,
    arena: std.mem.Allocator,
) anyerror!InterfaceValue {
    switch (val) {
        .string => |pl| {
            if (pl.len == 0) return .{ .string = .{ .ptr = 0, .len = 0 } };
            const bytes = src.readGuestBytes(pl.ptr, pl.len) orelse
                return error.MemoryNotAvailable;
            const new_ptr = dst.hostAllocGuest(pl.len, 1) orelse
                return error.ReallocFailed;
            const dest = dst.writableGuestBytes(new_ptr, pl.len) orelse
                return error.MemoryNotAvailable;
            @memcpy(dest, bytes);
            return .{ .string = .{ .ptr = new_ptr, .len = pl.len } };
        },
        .list => |pl| {
            if (pl.len == 0) return .{ .list = .{ .ptr = 0, .len = 0 } };
            const bytes = src.readGuestBytes(pl.ptr, pl.len) orelse
                return error.MemoryNotAvailable;
            const new_ptr = dst.hostAllocGuest(pl.len, 1) orelse
                return error.ReallocFailed;
            const dest = dst.writableGuestBytes(new_ptr, pl.len) orelse
                return error.MemoryNotAvailable;
            @memcpy(dest, bytes);
            return .{ .list = .{ .ptr = new_ptr, .len = pl.len } };
        },
        .record_val => |fields| {
            const out = try arena.alloc(InterfaceValue, fields.len);
            for (fields, 0..) |f, i| out[i] = try marshalValueAcrossMemory(src, dst, f, arena);
            return .{ .record_val = out };
        },
        .tuple_val => |fields| {
            const out = try arena.alloc(InterfaceValue, fields.len);
            for (fields, 0..) |f, i| out[i] = try marshalValueAcrossMemory(src, dst, f, arena);
            return .{ .tuple_val = out };
        },
        .variant_val => |v| {
            const new_payload: ?*const InterfaceValue = if (v.payload) |p| blk: {
                const nv = try arena.create(InterfaceValue);
                nv.* = try marshalValueAcrossMemory(src, dst, p.*, arena);
                break :blk nv;
            } else null;
            return .{ .variant_val = .{ .discriminant = v.discriminant, .payload = new_payload } };
        },
        .option_val => |o| {
            const new_payload: ?*const InterfaceValue = if (o.payload) |p| blk: {
                const nv = try arena.create(InterfaceValue);
                nv.* = try marshalValueAcrossMemory(src, dst, p.*, arena);
                break :blk nv;
            } else null;
            return .{ .option_val = .{ .is_some = o.is_some, .payload = new_payload } };
        },
        .result_val => |r| {
            const new_payload: ?*const InterfaceValue = if (r.payload) |p| blk: {
                const nv = try arena.create(InterfaceValue);
                nv.* = try marshalValueAcrossMemory(src, dst, p.*, arena);
                break :blk nv;
            } else null;
            return .{ .result_val = .{ .is_ok = r.is_ok, .payload = new_payload } };
        },
        // Pure scalars / handles / flags / enums: no nested PtrLen.
        else => return val,
    }
}

// ── #726: typed cross-memory marshaling for `list<compound>` ────────────────
//
// `marshalValueAcrossMemory` / `rewriteValueBackToCallerOwned`
// (PR #725 / #727) walk the InterfaceValue tree alone, so `.list` is
// byte-copied opaque. For `list<string>`, `list<record { ...: string }>`,
// `list<list<u8>>`, `list<option<string>>`, `list<result<string, _>>`,
// `list<variant ... with string arms>` and `list<tuple<u32, string>>`
// the inner (ptr, len) tuples stay pointing into the source memory —
// the same class of bug PR #725 fixed for top-level compounds.
//
// The typed variants below take a `(t, reg)` pair and, when `.list`
// elements themselves contain a PtrLen, re-lift / re-lower each
// element through `canonical_abi.loadValReg` / `storeValReg` against
// the source/destination memories so element-internal pointers are
// translated correctly. Pure-scalar element types still take the
// existing opaque-bytes fast path.

/// Recursive: does the canonical-ABI representation of `t` carry any
/// `.string` / `.list` payload that needs cross-memory translation?
/// Lookups stop at unresolved/recursive types and conservatively
/// return false (no translation), matching the pre-#726 behaviour for
/// untyped values.
fn valTypeHasPtrLen(reg: TypeRegistry, t: ctypes.ValType) bool {
    switch (t) {
        .string, .list => return true,
        .bool, .s8, .u8, .s16, .u16, .s32, .u32, .s64, .u64, .f32, .f64, .char, .own, .borrow, .future, .stream, .error_context => return false,
        else => {},
    }
    const td = reg.resolve(t) orelse return false;
    return switch (td) {
        .record => |r| blk: {
            for (r.fields) |f| if (valTypeHasPtrLen(reg, f.type)) break :blk true;
            break :blk false;
        },
        .tuple => |tup| blk: {
            for (tup.fields) |f| if (valTypeHasPtrLen(reg, f)) break :blk true;
            break :blk false;
        },
        .variant => |v| blk: {
            for (v.cases) |c| if (c.type) |ct| if (valTypeHasPtrLen(reg, ct)) break :blk true;
            break :blk false;
        },
        .option => |o| valTypeHasPtrLen(reg, o.inner),
        .result => |r| (if (r.ok) |ok| valTypeHasPtrLen(reg, ok) else false) or
            (if (r.err) |er| valTypeHasPtrLen(reg, er) else false),
        .list => |l| valTypeHasPtrLen(reg, l.element) or true, // .list itself is a PtrLen
        .flags, .enum_, .resource => false,
        .val => |v| valTypeHasPtrLen(reg, v),
        .func, .component, .instance => false,
    };
}

/// Per-element list stride (size rounded up to element alignment).
fn listElementStride(reg: TypeRegistry, elem_t: ctypes.ValType) u32 {
    const sz = abi.sizeOfType(reg, elem_t);
    const al = abi.alignOfType(reg, elem_t);
    return abi.alignUp(sz, al);
}

/// Typed companion to `marshalValueAcrossMemory`. Drives per-element
/// translation for `.list` when the element type contains a PtrLen;
/// otherwise reuses the opaque-bytes fast path.
fn marshalValueAcrossMemoryTyped(
    src: *ComponentInstance,
    dst: *ComponentInstance,
    val: InterfaceValue,
    t: ctypes.ValType,
    reg: TypeRegistry,
    arena: std.mem.Allocator,
) anyerror!InterfaceValue {
    switch (val) {
        .string => |pl| {
            if (pl.len == 0) return .{ .string = .{ .ptr = 0, .len = 0 } };
            const bytes = src.readGuestBytes(pl.ptr, pl.len) orelse
                return error.MemoryNotAvailable;
            const new_ptr = dst.hostAllocGuest(pl.len, 1) orelse
                return error.ReallocFailed;
            const dest = dst.writableGuestBytes(new_ptr, pl.len) orelse
                return error.MemoryNotAvailable;
            @memcpy(dest, bytes);
            return .{ .string = .{ .ptr = new_ptr, .len = pl.len } };
        },
        .list => |pl| {
            if (pl.len == 0) return .{ .list = .{ .ptr = 0, .len = 0 } };
            const elem_t = blk: {
                const td = reg.resolve(t) orelse break :blk null;
                break :blk switch (td) {
                    .list => |l| l.element,
                    else => null,
                };
            };
            // Without a resolvable element type we can't compute
            // stride/load/store; fall back to the opaque byte-copy.
            // This is also the right behaviour for primitive-element
            // lists where the bytewise copy already does the job.
            if (elem_t == null or !valTypeHasPtrLen(reg, elem_t.?)) {
                // Bytewise copy with correct stride: pl.len is the
                // element count, not the byte size. For primitive
                // elements stride == elemSize so this also handles
                // alignment. For untyped fallback (elem_t == null)
                // we have no stride info, so we delegate to the
                // untyped helper which preserves the legacy behaviour
                // for test-only paths that don't supply a registry.
                if (elem_t == null) {
                    return try marshalValueAcrossMemory(src, dst, val, arena);
                }
                const et2 = elem_t.?;
                const stride2 = listElementStride(reg, et2);
                const elem_align2 = abi.alignOfType(reg, et2);
                const total_bytes2 = std.math.mul(u32, pl.len, stride2) catch
                    return error.MemoryNotAvailable;
                const bytes = src.readGuestBytes(pl.ptr, total_bytes2) orelse
                    return error.MemoryNotAvailable;
                const new_ptr = dst.hostAllocGuest(total_bytes2, elem_align2) orelse
                    return error.ReallocFailed;
                const dest = dst.writableGuestBytes(new_ptr, total_bytes2) orelse
                    return error.MemoryNotAvailable;
                @memcpy(dest, bytes);
                return .{ .list = .{ .ptr = new_ptr, .len = pl.len } };
            }
            // Slow path: per-element re-lift / re-lower.
            const et = elem_t.?;
            const stride = listElementStride(reg, et);
            const elem_align = abi.alignOfType(reg, et);
            const total_bytes = std.math.mul(u32, pl.len, stride) catch
                return error.MemoryNotAvailable;
            const src_bytes = src.readGuestBytes(pl.ptr, total_bytes) orelse
                return error.MemoryNotAvailable;
            const new_ptr = dst.hostAllocGuest(total_bytes, elem_align) orelse
                return error.ReallocFailed;
            // `loadValReg` reads ptrs out of src_bytes (so element
            // PtrLens still address src memory); the translated value
            // is then stored into dst_bytes via `storeValReg`.
            var i: u32 = 0;
            while (i < pl.len) : (i += 1) {
                const off: u32 = i * stride;
                const lifted = try abi.loadValReg(src_bytes, off, et, reg, arena);
                const translated = try marshalValueAcrossMemoryTyped(src, dst, lifted, et, reg, arena);
                // Re-read writable dst_bytes inside the loop in case
                // `dst.hostAllocGuest` invalidated the previous view
                // (memory.grow during a nested element translation).
                const dst_bytes = dst.writableGuestBytes(new_ptr, total_bytes) orelse
                    return error.MemoryNotAvailable;
                try abi.storeValReg(dst_bytes, off, et, translated, reg);
            }
            return .{ .list = .{ .ptr = new_ptr, .len = pl.len } };
        },
        .record_val => |fields| {
            const td = reg.resolve(t);
            const field_types: ?[]const ctypes.Field = if (td) |d| switch (d) {
                .record => |r| r.fields,
                else => null,
            } else null;
            const out = try arena.alloc(InterfaceValue, fields.len);
            for (fields, 0..) |f, i| {
                if (field_types) |ft| if (i < ft.len) {
                    out[i] = try marshalValueAcrossMemoryTyped(src, dst, f, ft[i].type, reg, arena);
                    continue;
                };
                out[i] = try marshalValueAcrossMemory(src, dst, f, arena);
            }
            return .{ .record_val = out };
        },
        .tuple_val => |fields| {
            const td = reg.resolve(t);
            const field_types: ?[]const ctypes.ValType = if (td) |d| switch (d) {
                .tuple => |tup| tup.fields,
                else => null,
            } else null;
            const out = try arena.alloc(InterfaceValue, fields.len);
            for (fields, 0..) |f, i| {
                if (field_types) |ft| if (i < ft.len) {
                    out[i] = try marshalValueAcrossMemoryTyped(src, dst, f, ft[i], reg, arena);
                    continue;
                };
                out[i] = try marshalValueAcrossMemory(src, dst, f, arena);
            }
            return .{ .tuple_val = out };
        },
        .variant_val => |v| {
            const td = reg.resolve(t);
            const case_t: ?ctypes.ValType = if (td) |d| switch (d) {
                .variant => |vt| if (v.discriminant < vt.cases.len) vt.cases[v.discriminant].type else null,
                else => null,
            } else null;
            const new_payload: ?*const InterfaceValue = if (v.payload) |p| blk: {
                const nv = try arena.create(InterfaceValue);
                nv.* = if (case_t) |ct|
                    try marshalValueAcrossMemoryTyped(src, dst, p.*, ct, reg, arena)
                else
                    try marshalValueAcrossMemory(src, dst, p.*, arena);
                break :blk nv;
            } else null;
            return .{ .variant_val = .{ .discriminant = v.discriminant, .payload = new_payload } };
        },
        .option_val => |o| {
            const td = reg.resolve(t);
            const inner_t: ?ctypes.ValType = if (td) |d| switch (d) {
                .option => |ot| ot.inner,
                else => null,
            } else null;
            const new_payload: ?*const InterfaceValue = if (o.payload) |p| blk: {
                const nv = try arena.create(InterfaceValue);
                nv.* = if (inner_t) |it|
                    try marshalValueAcrossMemoryTyped(src, dst, p.*, it, reg, arena)
                else
                    try marshalValueAcrossMemory(src, dst, p.*, arena);
                break :blk nv;
            } else null;
            return .{ .option_val = .{ .is_some = o.is_some, .payload = new_payload } };
        },
        .result_val => |r| {
            const td = reg.resolve(t);
            const arm_t: ?ctypes.ValType = if (td) |d| switch (d) {
                .result => |rt| if (r.is_ok) rt.ok else rt.err,
                else => null,
            } else null;
            const new_payload: ?*const InterfaceValue = if (r.payload) |p| blk: {
                const nv = try arena.create(InterfaceValue);
                nv.* = if (arm_t) |at|
                    try marshalValueAcrossMemoryTyped(src, dst, p.*, at, reg, arena)
                else
                    try marshalValueAcrossMemory(src, dst, p.*, arena);
                break :blk nv;
            } else null;
            return .{ .result_val = .{ .is_ok = r.is_ok, .payload = new_payload } };
        },
        else => return val,
    }
}

/// Typed companion to `rewriteValueBackToCallerOwned`. Mirror of
/// `marshalValueAcrossMemoryTyped` for the result direction: deep-copy
/// into `allocator` so the existing `InterfaceValue.deinit` ownership
/// contract continues to hold at the call site.
fn rewriteValueBackToCallerOwnedTyped(
    caller: *ComponentInstance,
    callee: *ComponentInstance,
    val: InterfaceValue,
    t: ctypes.ValType,
    reg: TypeRegistry,
    allocator: Allocator,
) anyerror!InterfaceValue {
    switch (val) {
        .string => |pl| {
            if (pl.len == 0) return .{ .string = .{ .ptr = 0, .len = 0 } };
            const bytes = callee.readGuestBytes(pl.ptr, pl.len) orelse
                return error.MemoryNotAvailable;
            const new_ptr = caller.hostAllocGuest(pl.len, 1) orelse
                return error.ReallocFailed;
            const dest = caller.writableGuestBytes(new_ptr, pl.len) orelse
                return error.MemoryNotAvailable;
            @memcpy(dest, bytes);
            return .{ .string = .{ .ptr = new_ptr, .len = pl.len } };
        },
        .list => |pl| {
            if (pl.len == 0) return .{ .list = .{ .ptr = 0, .len = 0 } };
            const elem_t = blk: {
                const td = reg.resolve(t) orelse break :blk null;
                break :blk switch (td) {
                    .list => |l| l.element,
                    else => null,
                };
            };
            if (elem_t == null or !valTypeHasPtrLen(reg, elem_t.?)) {
                if (elem_t == null) {
                    return try rewriteValueBackToCallerOwned(caller, callee, val, allocator);
                }
                const et2 = elem_t.?;
                const stride2 = listElementStride(reg, et2);
                const elem_align2 = abi.alignOfType(reg, et2);
                const total_bytes2 = std.math.mul(u32, pl.len, stride2) catch
                    return error.MemoryNotAvailable;
                const bytes = callee.readGuestBytes(pl.ptr, total_bytes2) orelse
                    return error.MemoryNotAvailable;
                const new_ptr = caller.hostAllocGuest(total_bytes2, elem_align2) orelse
                    return error.ReallocFailed;
                const dest = caller.writableGuestBytes(new_ptr, total_bytes2) orelse
                    return error.MemoryNotAvailable;
                @memcpy(dest, bytes);
                return .{ .list = .{ .ptr = new_ptr, .len = pl.len } };
            }
            const et = elem_t.?;
            const stride = listElementStride(reg, et);
            const elem_align = abi.alignOfType(reg, et);
            const total_bytes = std.math.mul(u32, pl.len, stride) catch
                return error.MemoryNotAvailable;
            const src_bytes = callee.readGuestBytes(pl.ptr, total_bytes) orelse
                return error.MemoryNotAvailable;
            const new_ptr = caller.hostAllocGuest(total_bytes, elem_align) orelse
                return error.ReallocFailed;
            var i: u32 = 0;
            while (i < pl.len) : (i += 1) {
                const off: u32 = i * stride;
                // Use a transient arena for the lifted intermediate so
                // we don't leak callee-side allocations into the
                // caller-owned tree's allocator. The intermediate is
                // freed by `arena.deinit` at end of element.
                var arena = std.heap.ArenaAllocator.init(allocator);
                defer arena.deinit();
                const arena_alloc = arena.allocator();
                const lifted = try abi.loadValReg(src_bytes, off, et, reg, arena_alloc);
                // Translate into caller memory through the typed
                // marshaler (writes new ptrs through caller.hostAllocGuest).
                const translated = try marshalValueAcrossMemoryTyped(callee, caller, lifted, et, reg, arena_alloc);
                const dst_bytes = caller.writableGuestBytes(new_ptr, total_bytes) orelse
                    return error.MemoryNotAvailable;
                try abi.storeValReg(dst_bytes, off, et, translated, reg);
            }
            return .{ .list = .{ .ptr = new_ptr, .len = pl.len } };
        },
        .record_val => |fields| {
            const td = reg.resolve(t);
            const field_types: ?[]const ctypes.Field = if (td) |d| switch (d) {
                .record => |r| r.fields,
                else => null,
            } else null;
            const out = try allocator.alloc(InterfaceValue, fields.len);
            errdefer allocator.free(out);
            var produced: usize = 0;
            errdefer for (out[0..produced]) |o| o.deinit(allocator);
            for (fields, 0..) |f, i| {
                out[i] = if (field_types) |ft|
                    (if (i < ft.len)
                        try rewriteValueBackToCallerOwnedTyped(caller, callee, f, ft[i].type, reg, allocator)
                    else
                        try rewriteValueBackToCallerOwned(caller, callee, f, allocator))
                else
                    try rewriteValueBackToCallerOwned(caller, callee, f, allocator);
                produced = i + 1;
            }
            return .{ .record_val = out };
        },
        .tuple_val => |fields| {
            const td = reg.resolve(t);
            const field_types: ?[]const ctypes.ValType = if (td) |d| switch (d) {
                .tuple => |tup| tup.fields,
                else => null,
            } else null;
            const out = try allocator.alloc(InterfaceValue, fields.len);
            errdefer allocator.free(out);
            var produced: usize = 0;
            errdefer for (out[0..produced]) |o| o.deinit(allocator);
            for (fields, 0..) |f, i| {
                out[i] = if (field_types) |ft|
                    (if (i < ft.len)
                        try rewriteValueBackToCallerOwnedTyped(caller, callee, f, ft[i], reg, allocator)
                    else
                        try rewriteValueBackToCallerOwned(caller, callee, f, allocator))
                else
                    try rewriteValueBackToCallerOwned(caller, callee, f, allocator);
                produced = i + 1;
            }
            return .{ .tuple_val = out };
        },
        .variant_val => |v| {
            const td = reg.resolve(t);
            const case_t: ?ctypes.ValType = if (td) |d| switch (d) {
                .variant => |vt| if (v.discriminant < vt.cases.len) vt.cases[v.discriminant].type else null,
                else => null,
            } else null;
            const new_payload: ?*const InterfaceValue = if (v.payload) |p| blk: {
                const nv = try allocator.create(InterfaceValue);
                errdefer allocator.destroy(nv);
                nv.* = if (case_t) |ct|
                    try rewriteValueBackToCallerOwnedTyped(caller, callee, p.*, ct, reg, allocator)
                else
                    try rewriteValueBackToCallerOwned(caller, callee, p.*, allocator);
                break :blk nv;
            } else null;
            return .{ .variant_val = .{ .discriminant = v.discriminant, .payload = new_payload } };
        },
        .option_val => |o| {
            const td = reg.resolve(t);
            const inner_t: ?ctypes.ValType = if (td) |d| switch (d) {
                .option => |ot| ot.inner,
                else => null,
            } else null;
            const new_payload: ?*const InterfaceValue = if (o.payload) |p| blk: {
                const nv = try allocator.create(InterfaceValue);
                errdefer allocator.destroy(nv);
                nv.* = if (inner_t) |it|
                    try rewriteValueBackToCallerOwnedTyped(caller, callee, p.*, it, reg, allocator)
                else
                    try rewriteValueBackToCallerOwned(caller, callee, p.*, allocator);
                break :blk nv;
            } else null;
            return .{ .option_val = .{ .is_some = o.is_some, .payload = new_payload } };
        },
        .result_val => |r| {
            const td = reg.resolve(t);
            const arm_t: ?ctypes.ValType = if (td) |d| switch (d) {
                .result => |rt| if (r.is_ok) rt.ok else rt.err,
                else => null,
            } else null;
            const new_payload: ?*const InterfaceValue = if (r.payload) |p| blk: {
                const nv = try allocator.create(InterfaceValue);
                errdefer allocator.destroy(nv);
                nv.* = if (arm_t) |at|
                    try rewriteValueBackToCallerOwnedTyped(caller, callee, p.*, at, reg, allocator)
                else
                    try rewriteValueBackToCallerOwned(caller, callee, p.*, allocator);
                break :blk nv;
            } else null;
            return .{ .result_val = .{ .is_ok = r.is_ok, .payload = new_payload } };
        },
        .flags_val => |words| {
            const copy = try allocator.alloc(u32, words.len);
            @memcpy(copy, words);
            return .{ .flags_val = copy };
        },
        else => return val,
    }
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

// ── Component → module-local funcidx translation ────────────────────────────

/// Translate a component-level core-funcidx into a `CoreFuncIdxLocal`
/// inside `core_entry`'s owning module instance. Returns `null` when
/// `comp_idx` is `null`. The fallback is the pre-#719 behaviour for
/// hand-authored test fixtures that do not register an `aliases[]`
/// entry — we interpret the component-level index as module-local.
/// Real wit-component output always carries the alias and hits the
/// translation path. Returns `error.ReallocNotAvailable` when the
/// alias resolves to a *different* core instance than the one
/// currently executing the lift, which Canon ABI v1 forbids
/// (`(realloc $f)` must live on the same core module as the lift's
/// memory).
fn translateComponentFuncIdx(
    owner_inst: *const ComponentInstance,
    core_entry: ComponentInstance.CoreInstanceEntry,
    comp_idx: ?CoreFuncIdxComponent,
) ExecutionError!?CoreFuncIdxLocal {
    const cidx = comp_idx orelse return null;
    const target = owner_inst.resolveTopLevelCoreFuncAny(cidx.value()) orelse
        return CoreFuncIdxLocal.from(cidx.value());
    switch (target) {
        .aot => |t| {
            if (core_entry.aot_inst) |ai| {
                if (ai != t.ai) return error.ReallocNotAvailable;
                return t.local_idx;
            }
            return error.ReallocNotAvailable;
        },
        .interp => |t| {
            if (core_entry.module_inst) |mi| {
                if (mi != t.mi) return error.ReallocNotAvailable;
                return t.local_idx;
            }
            return error.ReallocNotAvailable;
        },
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
    const is_aot = switch (frame) {
        .aot => true,
        .interp => false,
    };

    // Parse canonical options
    const lift_opts = LiftOptions.fromOpts(exported.opts);

    // Translate the component-level `realloc` and `post_return`
    // core-funcidx (as carried in `canon lift` opts) into function
    // indices local to the lifted core function's own AOT module /
    // interp module instance. Without this translation,
    // `AotFrame.realloc` / `InterpFrame.realloc` (or `executeCore` for
    // post_return) would dispatch against the WRONG wasm function —
    // sometimes landing inside an unrelated once-init guard whose body
    // starts with `unreachable` (#719: tcgc.compile trapping at
    // `local_func[26]+0x23f` because component-level realloc_idx 81
    // was passed straight through as a module-local funcidx). The
    // `CoreFuncIdxComponent` / `CoreFuncIdxLocal` newtype enums make
    // this conversion explicit and compiler-enforced.
    //
    // Per Canon ABI v1, `(realloc $f)` and `(post-return $f)` must
    // live on the same core module as the lift's memory; we treat a
    // cross-instance resolve as a malformed component. Hand-authored
    // test fixtures that do not register an `aliases[]` entry for
    // their realloc/post_return export fall back to interpreting the
    // index as module-local (the pre-#719 behaviour) — real
    // wit-component output always carries a corresponding alias and
    // hits the translation path.
    const resolved_realloc_idx: ?CoreFuncIdxLocal =
        try translateComponentFuncIdx(owner_inst, core_entry, lift_opts.realloc_idx);
    const resolved_post_return_idx: ?CoreFuncIdxLocal =
        try translateComponentFuncIdx(owner_inst, core_entry, lift_opts.post_return_idx);

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
        const realloc_idx = resolved_realloc_idx orelse return error.ReallocNotAvailable;
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
            storeInterfaceValue(mem, offset, arg, pt, registry) catch return error.LowerError;
            offset += typeSize(registry, pt);
        }

        frame.pushSlot(.{ .i32 = @bitCast(ptr) }) catch return error.StackOverflow;
    }

    // 4b. Spilled-result mode: per canon-ABI v1 spec for canon.lift,
    // when `flat_count(results) > MAX_FLAT_RESULTS` the lifted core
    // function uses CALLEE-allocates — it allocates the result buffer
    // (via its own realloc) and returns the pointer as a single i32
    // result. The caller does NOT pass a retptr. This matches what
    // wit-bindgen / rust-wit-bindgen emit on both interp and AOT.
    //
    // Earlier this path had a backend split (AOT caller-allocates,
    // interp callee-allocates), which broke real components: tcgc.compile
    // is emitted as `(param i32 i32 i32 i32) (result i32)` and the
    // AOT-pushed extra retptr arg was simply ignored, so the result
    // pointer was never read — see #719.

    // 5. Compute core result types for executeCore. Advisory for
    // interp (signature comes from the module); load-bearing for AOT
    // since callFuncScalar needs an accurate signature. Only compute
    // it for AOT — for interp we pass an empty slice and let
    // `executeFunction` read the core sig from the module.
    //
    // Spilled-result lift returns the retptr as a single i32 (callee-
    // allocates); other cases return the (single) flat result.
    var core_rt_buf: [1]core_types.ValType = undefined;
    const core_result_types: []const core_types.ValType = if (!is_aot)
        &.{}
    else if (result_types.len == 0)
        &.{}
    else if (flat_result_count > MAX_FLAT_RESULTS) blk: {
        core_rt_buf[0] = .i32;
        break :blk core_rt_buf[0..1];
    } else blk: {
        core_rt_buf[0] = coreFlatSlotType(result_types[0], registry) catch
            return error.AotPathUnsupported;
        break :blk core_rt_buf[0..1];
    };

    // 5b. Call the core function
    frame.executeCore(CoreFuncIdxLocal.from(exported.core_func_idx), &.{}, core_result_types) catch {
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
        // Spilled-result path — callee allocated the buffer and
        // returned its pointer as a single i32 (canon-ABI v1 lift
        // convention). Pop it and read the tuple from linear memory.
        const popped = frame.popSlot(.i32) catch return error.StackUnderflow;
        result_ptr_for_post_return = @bitCast(popped.i32);
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
    if (resolved_post_return_idx) |pr_idx| {
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
/// Multi-slot results spill to memory: the lifted core function
/// callee-allocates the buffer (via realloc) and returns its pointer
/// as a single i32 (canon-ABI v1 spilled-result convention) — see
/// the `flat_result_count > MAX_FLAT_RESULTS` branch in
/// `callComponentFuncByLocal`.
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

fn resolveAotCoreFuncResults(
    ai: *const aot_runtime.AotInstance,
    func_idx: u32,
) ?[]const core_types.ValType {
    if (func_idx < ai.module.import_function_count) {
        var imported_func_idx: u32 = 0;
        for (ai.module.imports) |imp| {
            if (imp.kind != .function) continue;
            if (imported_func_idx == func_idx) {
                if (imp.func_type_idx >= ai.module.func_types.len) return null;
                return ai.module.func_types[imp.func_type_idx].results;
            }
            imported_func_idx += 1;
        }
        return null;
    }

    const local_idx = func_idx - ai.module.import_function_count;
    if (local_idx >= ai.module.local_func_type_indices.len) return null;
    const type_idx = ai.module.local_func_type_indices[local_idx];
    if (type_idx >= ai.module.func_types.len) return null;
    return ai.module.func_types[type_idx].results;
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

    var frame: CallFrame = blk: {
        if (core_entry.module_inst) |mi| {
            const env = ExecEnv.create(mi, 4096, allocator) catch return error.OutOfMemory;
            break :blk .{ .interp = InterpFrame.init(env) };
        }
        if (core_entry.aot_inst) |ai| {
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

    const memory_idx = lift_opts.memory_idx orelse 0;
    var memory = frame.memory(memory_idx);

    // Lower args — same logic as the sync path.
    if (flat_param_count <= MAX_FLAT_PARAMS) {
        for (args, param_types) |arg, pt| {
            pushInterfaceValue(&frame, arg, pt, registry) catch return error.LowerError;
        }
    } else {
        if (memory == null) return error.MemoryNotAvailable;
        const realloc_idx = try translateComponentFuncIdx(owner_inst, core_entry, lift_opts.realloc_idx) orelse
            return error.ReallocNotAvailable;
        const tuple_size = computeTupleSize(registry, param_types);
        const tuple_align = computeTupleAlign(registry, param_types);
        const ptr = try callRealloc(&frame, realloc_idx, 0, 0, tuple_align, tuple_size);
        memory = frame.memory(memory_idx);
        const refreshed_mem = memory orelse return error.MemoryNotAvailable;

        var offset: u32 = 0;
        for (args, param_types) |arg, pt| {
            const al = typeAlign(registry, pt);
            offset = abi.alignUp(offset, al);
            storeInterfaceValue(refreshed_mem, offset, arg, pt, registry) catch return error.LowerError;
            offset += typeSize(registry, pt);
        }
        frame.pushSlot(.{ .i32 = @bitCast(ptr) }) catch return error.StackOverflow;
    }

    // Drive the core body. It is the callee's responsibility to invoke
    // `canon task.return` before returning — which deposits the lifted
    // results onto the task via `dispatchCanonBuiltin`.
    const core_result_types: []const core_types.ValType = switch (frame) {
        .interp => &.{},
        .aot => |f| resolveAotCoreFuncResults(f.ai, exported.core_func_idx) orelse
            return error.InvalidFuncType,
    };
    frame.executeCore(CoreFuncIdxLocal.from(exported.core_func_idx), &.{}, core_result_types) catch {
        switch (frame) {
            .interp => |f| if (f.env.host_trap) |ht| {
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
            },
            .aot => {},
        }
        return error.TrapInCoreFunction;
    };

    // Spec: with `callback` set, the core fn leaves a packed status i32
    // on the stack; otherwise (stackful async) it returns no value. We
    // probe optimistically: if the core fn returned an i32, peel it
    // off; otherwise leave status at 0 (the default).
    if (lift_opts.callback_idx != null) {
        const status = frame.popSlot(.i32) catch return error.StackUnderflow;
        out_status.* = @bitCast(status.i32);
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

fn strictCanonMemory(ctx: *const ComponentTrampolineCtx) ![]const u8 {
    const mem_idx = ctx.lower_opts.memory_idx orelse 0;
    const mem = ctx.comp_inst.resolveTopLevelMemory(mem_idx) orelse return error.MemoryNotAvailable;
    return mem.bytes();
}

fn validateCanonPtrLenValue(mem: []const u8, val: InterfaceValue, t: ctypes.ValType, registry: TypeRegistry, context: []const u8) !void {
    if (comptime !config.wamr_strict_canon) return;
    try abi.validatePtrLenValue(mem.len, val, t, registry, context);
}

fn typeContainsPtrLen(t: ctypes.ValType, registry: TypeRegistry) bool {
    if (comptime !config.wamr_strict_canon) return false;
    return switch (t) {
        .string, .list => true,
        .record => |idx| if (registry.get(idx)) |td| blk: {
            for (td.record.fields) |field| if (typeContainsPtrLen(field.type, registry)) break :blk true;
            break :blk false;
        } else false,
        .tuple => |idx| if (registry.get(idx)) |td| blk: {
            for (td.tuple.fields) |field_t| if (typeContainsPtrLen(field_t, registry)) break :blk true;
            break :blk false;
        } else false,
        .variant => |idx| if (registry.get(idx)) |td| blk: {
            for (td.variant.cases) |case| if (case.type) |payload_t| if (typeContainsPtrLen(payload_t, registry)) break :blk true;
            break :blk false;
        } else false,
        .option => |idx| if (registry.get(idx)) |td| typeContainsPtrLen(td.option.inner, registry) else false,
        .result => |idx| if (registry.get(idx)) |td| ((td.result.ok != null and typeContainsPtrLen(td.result.ok.?, registry)) or (td.result.err != null and typeContainsPtrLen(td.result.err.?, registry))) else false,
        .type_idx => |idx| if (registry.get(idx)) |td| typeDefContainsPtrLen(td, registry) else false,
        else => false,
    };
}

fn typeDefContainsPtrLen(td: ctypes.TypeDef, registry: TypeRegistry) bool {
    if (comptime !config.wamr_strict_canon) return false;
    return switch (td) {
        .val => |inner| typeContainsPtrLen(inner, registry),
        .list => true,
        .record => |record| blk: {
            for (record.fields) |field| if (typeContainsPtrLen(field.type, registry)) break :blk true;
            break :blk false;
        },
        .tuple => |tuple| blk: {
            for (tuple.fields) |field_t| if (typeContainsPtrLen(field_t, registry)) break :blk true;
            break :blk false;
        },
        .variant => |variant| blk: {
            for (variant.cases) |case| if (case.type) |payload_t| if (typeContainsPtrLen(payload_t, registry)) break :blk true;
            break :blk false;
        },
        .option => |option| typeContainsPtrLen(option.inner, registry),
        .result => |result| (result.ok != null and typeContainsPtrLen(result.ok.?, registry)) or (result.err != null and typeContainsPtrLen(result.err.?, registry)),
        else => false,
    };
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
            out[0] = switch (val) {
                .enum_val => |e| e,
                .variant_val => |v| v.discriminant,
                else => return error.InvalidValue,
            };
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
) abi.StoreError!void {
    abi.storeVal(mem, ptr, t, val) catch |err| switch (err) {
        error.CompoundNeedsRegistry => {
            try abi.storeValReg(mem, ptr, t, val, registry);
        },
        inline else => |e| return e,
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
    stack: anytype,
    task_manager: ?*async_mod.TaskManager,
    allocator: Allocator,
) ExecutionError!void {
    const env = stack;
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
    env: anytype,
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
            s.pending_read = .{
                .guest_ptr = guest_ptr,
                .max_count = max_count,
                .elem_size = elem_size,
            };
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
                if (s.host_driver) |driver| if (driver.on_drop_readable) |cb|
                    cb(driver.context);
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
                if (s.host_driver) |driver| if (driver.on_drop_writable) |cb|
                    cb(driver.context);
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

            const guest_ptr = comp_inst.hostAllocAndWrite(stored, 1) orelse
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

/// Advance host-backed streams whose guest reader is parked in
/// `stream.read`. A waitable event is only ready after the original read's
/// destination has been populated, so this completes the pending transfer
/// before waking wit-bindgen's callback reactor.
pub fn drivePendingHostStreamReads(
    comp_inst: *ComponentInstance,
    allocator: Allocator,
) bool {
    var delivered = false;
    var streams = comp_inst.streams.iterator();
    while (streams.next()) |entry| {
        const stream = entry.value_ptr;
        const pending = stream.pending_read orelse continue;
        const ws = stream.waitable_set orelse continue;
        const waitable_idx = stream.read_waitable_idx orelse continue;
        const driver = stream.host_driver orelse continue;

        if (stream.buffer.items.len == 0 and !stream.write_closed) {
            if (driver.on_read_into) |read_into| {
                const max_bytes_u64 =
                    @as(u64, pending.max_count) * @as(u64, pending.elem_size);
                if (max_bytes_u64 <= std.math.maxInt(u32)) {
                    const max_bytes: u32 = @intCast(max_bytes_u64);
                    if (comp_inst.writableGuestBytes(pending.guest_ptr, max_bytes)) |dst| {
                        const result = read_into(driver.context, dst);
                        switch (result.action) {
                            .progressed => {
                                const bytes_written = @min(result.bytes_written, max_bytes);
                                const count = bytes_written / pending.elem_size;
                                const tail = bytes_written - count * pending.elem_size;
                                if (tail != 0) {
                                    const tail_off = count * pending.elem_size;
                                    stream.buffer.appendSlice(
                                        comp_inst.allocator,
                                        dst[tail_off..bytes_written],
                                    ) catch {
                                        stream.write_closed = true;
                                    };
                                }
                                if (count > 0) {
                                    stream.pending_read = null;
                                    ws.setReady(
                                        waitable_idx,
                                        allocator,
                                        async_canon.packStatus(.completed, count),
                                    );
                                    delivered = true;
                                    continue;
                                }
                            },
                            .would_block => {},
                            .eof, .err => stream.write_closed = true,
                        }
                    } else {
                        stream.write_closed = true;
                    }
                } else {
                    stream.write_closed = true;
                }
            } else if (driver.on_read) |read| {
                const action = read(driver.context, stream, comp_inst.allocator);
                switch (action) {
                    .progressed, .would_block => {},
                    .eof, .err => stream.write_closed = true,
                }
            }
        }

        const buffered_count: u32 =
            @intCast(stream.buffer.items.len / pending.elem_size);
        if (buffered_count > 0) {
            const count = @min(pending.max_count, buffered_count);
            const byte_count = count * pending.elem_size;
            if (comp_inst.writableGuestBytes(pending.guest_ptr, byte_count)) |dst| {
                @memcpy(dst, stream.buffer.items[0..byte_count]);
                std.mem.copyForwards(
                    u8,
                    stream.buffer.items[0 .. stream.buffer.items.len - byte_count],
                    stream.buffer.items[byte_count..],
                );
                stream.buffer.items.len -= byte_count;
                stream.pending_read = null;
                ws.setReady(
                    waitable_idx,
                    allocator,
                    async_canon.packStatus(.completed, count),
                );
                delivered = true;
                continue;
            }
            stream.write_closed = true;
        }

        if (stream.write_closed) {
            stream.pending_read = null;
            ws.setReady(
                waitable_idx,
                allocator,
                async_canon.packStatus(.dropped, 0),
            );
            delivered = true;
        }
    }
    return delivered;
}

/// Whether the event driver should keep polling for a host-backed stream read.
pub fn hasPendingHostStreamReads(comp_inst: *ComponentInstance) bool {
    var streams = comp_inst.streams.iterator();
    while (streams.next()) |entry| {
        const stream = entry.value_ptr;
        if (stream.pending_read != null and
            stream.waitable_set != null and
            stream.read_waitable_idx != null and
            stream.host_driver != null)
        {
            return true;
        }
    }
    return false;
}

const AsyncLiftCallbackAction = union(enum) {
    exit,
    yield,
    wait: u32,
};

fn decodeAsyncLiftCallbackStatus(status: u32) ExecutionError!AsyncLiftCallbackAction {
    return switch (status & 0xf) {
        0 => .exit,
        1 => .yield,
        2 => .{ .wait = status >> 4 },
        else => error.InvalidFuncType,
    };
}

const AsyncLiftCallbackEvent = struct {
    kind: u32,
    handle: u32,
    code: u32,
};

fn executeAsyncLiftCallback(
    owner_inst: *const ComponentInstance,
    fallback_core_entry: ComponentInstance.CoreInstanceEntry,
    callback_idx: CoreFuncIdxComponent,
    event: AsyncLiftCallbackEvent,
    allocator: Allocator,
) ExecutionError!u32 {
    var owned_env: ?*ExecEnv = null;
    var callback_local: CoreFuncIdxLocal = undefined;
    var frame: CallFrame = blk: {
        if (owner_inst.resolveTopLevelCoreFuncAny(callback_idx.value())) |target| {
            switch (target) {
                .interp => |t| {
                    const env = ExecEnv.create(t.mi, 4096, allocator) catch
                        return error.OutOfMemory;
                    owned_env = env;
                    callback_local = t.local_idx;
                    break :blk .{ .interp = InterpFrame.init(env) };
                },
                .aot => |t| {
                    callback_local = t.local_idx;
                    break :blk .{ .aot = AotFrame.init(t.ai, allocator) };
                },
            }
        }

        // Hand-authored fixtures may omit the alias that maps a component
        // core-funcidx into its module-local indexspace.
        callback_local = CoreFuncIdxLocal.from(callback_idx.value());
        if (fallback_core_entry.module_inst) |mi| {
            const env = ExecEnv.create(mi, 4096, allocator) catch
                return error.OutOfMemory;
            owned_env = env;
            break :blk .{ .interp = InterpFrame.init(env) };
        }
        if (fallback_core_entry.aot_inst) |ai| {
            break :blk .{ .aot = AotFrame.init(ai, allocator) };
        }
        return error.CoreInstanceNotAvailable;
    };
    defer {
        if (owned_env) |env| env.destroy();
        frame.deinit();
    }

    frame.pushSlot(.{ .i32 = @bitCast(event.kind) }) catch
        return error.StackOverflow;
    frame.pushSlot(.{ .i32 = @bitCast(event.handle) }) catch
        return error.StackOverflow;
    frame.pushSlot(.{ .i32 = @bitCast(event.code) }) catch
        return error.StackOverflow;

    const result_types: []const core_types.ValType = switch (frame) {
        .interp => &.{},
        .aot => |f| resolveAotCoreFuncResults(f.ai, callback_local.value()) orelse
            return error.InvalidFuncType,
    };
    frame.executeCore(callback_local, &.{}, result_types) catch
        return error.TrapInCoreFunction;
    const status = frame.popSlot(.i32) catch return error.StackUnderflow;
    return @bitCast(status.i32);
}

fn driveAsyncLiftCallbacks(
    owner_inst: *const ComponentInstance,
    fallback_core_entry: ComponentInstance.CoreInstanceEntry,
    callback_idx: CoreFuncIdxComponent,
    initial_status: u32,
    task_manager: *async_mod.TaskManager,
    task_handle: u32,
    allocator: Allocator,
) ExecutionError!void {
    var status = initial_status;
    while (task_handle < task_manager.tasks.items.len and
        task_manager.tasks.items[task_handle].state == .started)
    {
        const event: AsyncLiftCallbackEvent = switch (try decodeAsyncLiftCallbackStatus(status)) {
            .exit => return,
            .yield => blk: {
                if (owner_inst.async_event_driver) |driver| {
                    _ = driver(
                        owner_inst.async_event_driver_ctx,
                        @constCast(owner_inst),
                        null,
                        allocator,
                    );
                }
                break :blk .{ .kind = 0, .handle = 0, .code = 0 };
            },
            .wait => |waitable_set_handle| blk: {
                const mutable_inst = @constCast(owner_inst);
                const ws = mutable_inst.waitable_sets.getPtr(waitable_set_handle) orelse
                    return error.TrapInCoreFunction;
                while (true) {
                    if (ws.popReadyEvent()) |ready| {
                        break :blk .{
                            .kind = async_canon.eventCodeForKind(ready.kind),
                            .handle = ready.handle,
                            .code = ready.code,
                        };
                    }
                    const driver = owner_inst.async_event_driver orelse
                        return error.TrapInCoreFunction;
                    _ = driver(
                        owner_inst.async_event_driver_ctx,
                        mutable_inst,
                        10 * std.time.ns_per_ms,
                        allocator,
                    );
                }
            },
        };
        status = try executeAsyncLiftCallback(
            owner_inst,
            fallback_core_entry,
            callback_idx,
            event,
            allocator,
        );
    }
}

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
        callComponentFuncByLocalAsyncLifted(owner_for_type, exported_local, args, &status, allocator) catch |e| {
            task_manager.cancelTask(handle);
            return e;
        };
        if (lift_opts.callback_idx) |callback_idx| {
            driveAsyncLiftCallbacks(
                owner_for_type,
                owner_for_type.core_instances[exported_local.core_instance_idx],
                callback_idx,
                status,
                task_manager,
                handle,
                allocator,
            ) catch |e| {
                task_manager.cancelTask(handle);
                return e;
            };
        }
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
    try std.testing.expectEqual(@as(?CoreFuncIdxComponent, CoreFuncIdxComponent.from(1)), lo.realloc_idx);
    try std.testing.expectEqual(@as(?CoreFuncIdxComponent, CoreFuncIdxComponent.from(2)), lo.post_return_idx);
    try std.testing.expectEqual(ctypes.StringEncoding.utf16, lo.string_encoding);
}

test "LiftOptions: defaults" {
    const lo = LiftOptions.fromOpts(&.{});
    try std.testing.expectEqual(@as(?u32, null), lo.memory_idx);
    try std.testing.expectEqual(@as(?CoreFuncIdxComponent, null), lo.realloc_idx);
    try std.testing.expectEqual(@as(?CoreFuncIdxComponent, null), lo.post_return_idx);
    try std.testing.expectEqual(ctypes.StringEncoding.utf8, lo.string_encoding);
    try std.testing.expectEqual(false, lo.is_async);
    try std.testing.expectEqual(@as(?CoreFuncIdxComponent, null), lo.callback_idx);
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
    try std.testing.expectEqual(@as(?CoreFuncIdxComponent, CoreFuncIdxComponent.from(7)), lo.callback_idx);
}

test "async lift callback status decodes canonical callback protocol" {
    try std.testing.expectEqual(
        AsyncLiftCallbackAction.exit,
        try decodeAsyncLiftCallbackStatus(0),
    );
    try std.testing.expectEqual(
        AsyncLiftCallbackAction.yield,
        try decodeAsyncLiftCallbackStatus(1),
    );
    try std.testing.expectEqual(
        AsyncLiftCallbackAction{ .wait = 37 },
        try decodeAsyncLiftCallbackStatus((37 << 4) | 2),
    );
    try std.testing.expectError(
        error.InvalidFuncType,
        decodeAsyncLiftCallbackStatus(3),
    );
}

test "LowerOptions: async opt flips is_async (#551 canon-lower-of-async-func)" {
    const opts_async = [_]ctypes.CanonOpt{ .{ .memory = 0 }, .async_lift };
    const lo_async = LowerOptions.fromOpts(&opts_async);
    try std.testing.expectEqual(true, lo_async.is_async);

    const opts_sync = [_]ctypes.CanonOpt{.{ .memory = 0 }};
    const lo_sync = LowerOptions.fromOpts(&opts_sync);
    try std.testing.expectEqual(false, lo_sync.is_async);

    // `callback` opt on the lower side is accepted as a no-op (Binary.md
    // canon opt vec is shared with canon.lift); it must NOT flip is_async.
    const opts_cb = [_]ctypes.CanonOpt{.{ .callback = 3 }};
    const lo_cb = LowerOptions.fromOpts(&opts_cb);
    try std.testing.expectEqual(false, lo_cb.is_async);
}

test "packAsyncLowerStatus: pending future → (handle << 4) | STATUS_STARTED (#551)" {
    const testing = std.testing;
    var comp = ctypes.Component{
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
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();
    try inst.enableTestMem(testing.allocator, 4096);
    defer inst.disableTestMem();

    const msg = "hello error";
    const ptr = inst.hostAllocAndWrite(msg, 1).?;

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
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();
    try inst.enableTestMem(testing.allocator, 4096);
    defer inst.disableTestMem();

    const msg1 = "first failure";
    const ptr1 = inst.hostAllocAndWrite(msg1, 1).?;
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
    const ptr2 = inst.hostAllocAndWrite(msg2, 1).?;
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
    /// Component-level core-funcidx. Translate via
    /// `ComponentInstance.resolveTopLevelCoreFuncAny` before calling.
    realloc_idx: ?CoreFuncIdxComponent = null,
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
                .realloc => |idx| lo.realloc_idx = CoreFuncIdxComponent.from(idx),
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

/// Resolve a canon-lower's `(memory $m)` / `(realloc $f)` opts into a
/// directly-usable `CanonLowerCallCtx`. Called by the trampoline once
/// per host import dispatch and stashed on `comp_inst` so
/// `hostAllocAndWrite` honors the lowerer's pinned memory + realloc
/// (e.g. wit-component preview1 adapter's `cabi_import_realloc`
/// routing to per-call `temporary_data`). (#715.)
fn resolveLowerCallCtx(
    comp_inst: *ComponentInstance,
    lower_opts: LowerOptions,
) ?ComponentInstance.CanonLowerCallCtx {
    if (lower_opts.memory_idx == null and lower_opts.realloc_idx == null) return null;
    var cctx: ComponentInstance.CanonLowerCallCtx = .{};
    if (lower_opts.memory_idx) |midx| {
        cctx.memory = comp_inst.resolveTopLevelMemory(midx);
    }
    if (lower_opts.realloc_idx) |ridx| {
        cctx.realloc = comp_inst.resolveTopLevelCoreFuncAny(ridx.value());
    }
    return cctx;
}

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
    canon_lower_idx: u32 = 0,
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
            args[i] = loadInterfaceValue(mem.bytes(), offset, pt, registry, allocator) catch |err|
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
    var strict_mem: ?[]const u8 = null;
    for (args, ctx.param_types) |arg, pt| {
        if (typeContainsPtrLen(pt, registry)) {
            if (strict_mem == null) strict_mem = strictCanonMemory(ctx) catch |err| return trampolineTrap(env, ctx, err, .memory_resolve);
            validateCanonPtrLenValue(strict_mem.?, arg, pt, registry, "canon-lift") catch |err| return trampolineTrap(env, ctx, err, .lift_args);
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
    const saved_lower_ctx = ctx.comp_inst.current_lower_call_ctx;
    ctx.comp_inst.current_lower_call_ctx = resolveLowerCallCtx(ctx.comp_inst, ctx.lower_opts);
    defer ctx.comp_inst.current_lower_call_ctx = saved_lower_ctx;
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
                            if (@as(u64, result_dest_ptr) + bytes.len > mem.byteLen()) {
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
            if (typeContainsPtrLen(t, registry)) {
                validateCanonPtrLenValue(mem.bytes(), r, t, registry, "canon-lift") catch |err| return trampolineTrap(env, ctx, err, .lower_results);
            }
            const al = typeAlign(registry, t);
            offset = abi.alignUp(offset, al);
            storeInterfaceValue(mem.bytes(), offset, r, t, registry) catch |err|
                return trampolineTrap(env, ctx, err, .lower_results);
            offset += typeSize(registry, t);
        }
    } else {
        for (results, ctx.result_types) |r, t| {
            if (typeContainsPtrLen(t, registry)) {
                if (strict_mem == null) strict_mem = strictCanonMemory(ctx) catch |err| return trampolineTrap(env, ctx, err, .memory_resolve);
                validateCanonPtrLenValue(strict_mem.?, r, t, registry, "canon-lift") catch |err| return trampolineTrap(env, ctx, err, .lower_results);
            }
            pushInterfaceValue(&frame, r, t, registry) catch |err| {
                return trampolineTrap(env, ctx, err, .lower_results);
            };
        }
    }
}

/// #760: special-case `error.WasiExit` from the AOT host-import
/// dispatchers' catch arms.
///
/// `wasi:cli/exit.{exit,exit-with-code}` raises `error.WasiExit` after
/// pinning the requested code in `wasi_cli_adapter.pending_wasi_exit_code`.
/// On the interpreter path that error unwinds Zig-style to
/// `runLoadedComponent`'s catch arm (`wasi_cli_adapter.zig:24145`), which
/// reads `adapter.exit_code` and produces a normal `RunOutcome`. On the
/// AOT path the C-ABI dispatcher used to translate that into a generic
/// `status=1` failure — `genericDispatcher` then wrote the post-#714
/// `0xdeaddeaddeaddead` sentinel into the guest's return slot, the
/// wit-bindgen WASIp1 → preview2 adapter saw a value where `exit` was
/// supposed to have unwound, and `unreachable executed at adapter line
/// 2335: host exit implementation didn't exit!` SIGSEGV'd the host
/// after a successful run.
///
/// Match `aotProcExit`'s precedent at `src/runtime/aot/host_bridge.zig:464`
/// for raw-WASIp1 `proc_exit`: when the TLS is set, terminate the host
/// process with the requested code directly. When it isn't set (which
/// would mean a contract violation in some future `error.WasiExit`
/// raiser), fall through to the existing sentinel/warn-once path so
/// the bug surfaces rather than masquerading as a silent exit-0 —
/// returning `true` tells the caller "we handled it" only when the
/// process is about to be replaced anyway (i.e. never).
fn handleWasiExitFromAotDispatch() void {
    if (wasi_cli_adapter.takePendingWasiExitCode()) |code| {
        std.process.exit(@intCast(code & 0xff));
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
    a9: u64,
) callconv(.c) host_trampolines.DispatchResult {
    const ctx: *const ComponentTrampolineCtx = @ptrCast(@alignCast(ctx_opaque));
    const result = dispatchAotComponentTrampoline(ctx, lowered_sig.*, &.{ a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 }, null) catch |err| {
        if (err == error.WasiExit) handleWasiExitFromAotDispatch();
        if (debugAotEnabled()) {
            std.debug.print(
                "[aot-dispatch] canon.lower trampoline failed: {s}\n",
                .{@errorName(err)},
            );
        }
        const err_name = if (err == error.LiftedResultInvariantViolated) lifted_result_invariant_violated.ptr else @errorName(err).ptr;
        return .{ .status = 1, .value = 0, .err_name = err_name };
    };
    return .{ .status = 0, .value = result };
}

/// AOT-codegen-flavoured canon.lower dispatcher (#687). The trampoline pool
/// stub shifts caller regs right by one to inject `slot` as the first
/// C-ABI arg, so when the AOT codegen calls a host import as
/// `host_fn(vmctx, arg0, arg1, …)`, this dispatcher receives
/// `(slot, a0=vmctx, a1=arg0, a2=arg1, …, a9=arg8)`. We discard `a0`
/// (importer's vmctx) and re-issue `dispatchAotComponentTrampoline` over
/// `a1..a9`, matching the lowered wasm-arg shape the host trampoline
/// already expects. Widened from a0..a5 in #689, then a0..a8 in #700 so
/// `wasi_snapshot_preview1.path_open` (9 wasm params) fits.
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
    a9: u64,
) callconv(.c) host_trampolines.DispatchResult {
    const ctx: *const ComponentTrampolineCtx = @ptrCast(@alignCast(ctx_opaque));
    const result = dispatchAotComponentTrampoline(ctx, lowered_sig.*, &.{ a1, a2, a3, a4, a5, a6, a7, a8, a9 }, a0) catch |err| {
        if (err == error.WasiExit) handleWasiExitFromAotDispatch();
        if (debugAotEnabled()) {
            std.debug.print(
                "[aot-dispatch] canon.lower(aot) trampoline failed: {s}\n",
                .{@errorName(err)},
            );
        }
        const err_name = if (err == error.LiftedResultInvariantViolated) lifted_result_invariant_violated.ptr else @errorName(err).ptr;
        return .{ .status = 1, .value = 0, .err_name = err_name };
    };
    return .{ .status = 0, .value = result };
}

/// Wide AOT canon-lower relay for signatures with 10–15 wasm-level ABI
/// slots. The source core's stack arguments arrive intact from the native
/// trampoline, so record/variant lowerings do not silently lose their tail.
pub export fn wamrAotDispatchComponentTrampolineAotWide(
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
    a9: u64,
    a10: u64,
    a11: u64,
    a12: u64,
    a13: u64,
    a14: u64,
    a15: u64,
) callconv(.c) host_trampolines.DispatchResult {
    const ctx: *const ComponentTrampolineCtx = @ptrCast(@alignCast(ctx_opaque));
    const result = dispatchAotComponentTrampoline(
        ctx,
        lowered_sig.*,
        &.{ a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 },
        a0,
    ) catch |err| {
        if (err == error.WasiExit) handleWasiExitFromAotDispatch();
        if (debugAotEnabled()) {
            std.debug.print(
                "[aot-dispatch] wide canon.lower(aot) trampoline failed: {s}\n",
                .{@errorName(err)},
            );
        }
        const err_name = if (err == error.LiftedResultInvariantViolated) lifted_result_invariant_violated.ptr else @errorName(err).ptr;
        return .{ .status = 1, .value = 0, .err_name = err_name };
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
    /// True once we've already complained about a caller / target memory
    /// mismatch for this import. The fast thunk forwards raw core args
    /// between sibling AOT instances with no canon.lower / canon.lift
    /// marshaling, so any i32 that the caller intended as a *pointer into
    /// its own linear memory* gets passed to the target unchanged and
    /// silently dereferences whatever (unrelated) bytes happen to live at
    /// the same numeric offset in the target's memory — corrupting
    /// dlmalloc bookkeeping or worse. We can't distinguish "ptr i32" from
    /// "scalar i32" at this layer, but if both source and target share the
    /// same memory the forwarding is a no-op and safe, so the mismatch
    /// itself is a strong red flag worth surfacing once per import.
    /// (#719 Bug B follow-up.)
    cross_memory_warned: bool = false,
};

/// Cross-instance core-to-core dispatcher (#662). The trampoline pool stub
/// shifts caller regs right by one to inject `slot` as the first C-ABI arg,
/// so when the AOT codegen calls a host import as
/// `host_fn(vmctx, arg0, arg1, …)`, the dispatcher receives
/// `(slot, a0=vmctx, a1=arg0, a2=arg1, …, a9=arg8)`. `a0` is the *importer's*
/// vmctx — the sibling AotInstance builds its own vmctx internally in
/// `callFuncScalar`, but we peek at the importer's `memory_base` for the
/// cross-memory guard (#719 Bug B); `a1..a9` are the lowered wasm args.
/// Imports with 10–15 scalar args use `wamrAotDispatchCrossInstanceWide`;
/// both paths preserve the same target ownership, recursion, and trap
/// semantics.
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
    a9: u64,
) callconv(.c) host_trampolines.DispatchResult {
    const ctx: *CrossInstanceThunkCtx = @ptrCast(@alignCast(ctx_opaque));
    const result = dispatchAotCrossInstance(ctx, lowered_sig.*, a0, &.{ a1, a2, a3, a4, a5, a6, a7, a8, a9 }) catch |err| {
        if (debugAotEnabled()) {
            std.debug.print(
                "[aot-dispatch] cross-instance thunk '{s}' failed: {s}\n",
                .{ ctx.label, @errorName(err) },
            );
        }
        return .{ .status = 1, .value = 0, .err_name = @errorName(err).ptr };
    };
    return .{ .status = 0, .value = result };
}

/// Wide counterpart to `wamrAotDispatchCrossInstance`, reached through the
/// trampoline pool's 16-arg relay when a real sibling core export has 10–15
/// scalar wasm params (notably socket-address record lowerings).
pub export fn wamrAotDispatchCrossInstanceWide(
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
    a9: u64,
    a10: u64,
    a11: u64,
    a12: u64,
    a13: u64,
    a14: u64,
    a15: u64,
) callconv(.c) host_trampolines.DispatchResult {
    const ctx: *CrossInstanceThunkCtx = @ptrCast(@alignCast(ctx_opaque));
    const result = dispatchAotCrossInstance(
        ctx,
        lowered_sig.*,
        a0,
        &.{ a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 },
    ) catch |err| {
        if (debugAotEnabled()) {
            std.debug.print(
                "[aot-dispatch] wide cross-instance thunk '{s}' failed: {s}\n",
                .{ ctx.label, @errorName(err) },
            );
        }
        return .{ .status = 1, .value = 0, .err_name = @errorName(err).ptr };
    };
    return .{ .status = 0, .value = result };
}

fn dispatchAotCrossInstance(
    ctx: *CrossInstanceThunkCtx,
    lowered_sig: host_trampolines.LoweredSig,
    caller_vmctx: u64,
    arg_regs: []const u64,
) !u64 {
    // The narrow relay carries nine wasm args; the wide relay carries 15.
    // Spilled-result shapes still need canonical lift/lower and are not
    // valid raw cross-instance forwards.
    if (lowered_sig.has_retptr) return error.UnsupportedSignature;
    if (lowered_sig.param_types.len > arg_regs.len) return error.UnsupportedSignature;
    if (lowered_sig.result_types.len > 1) return error.UnsupportedSignature;
    if (lowered_sig.param_types.len != ctx.param_types.len) return error.UnsupportedSignature;
    if (lowered_sig.result_types.len != ctx.result_types.len) return error.UnsupportedSignature;
    for (lowered_sig.param_types, ctx.param_types) |a, b| if (a != b) return error.UnsupportedSignature;
    for (lowered_sig.result_types, ctx.result_types) |a, b| if (a != b) return error.UnsupportedSignature;

    // Cross-memory guard (#719 Bug B follow-up). The fast thunk forwards
    // raw core args between sibling AOT instances with no canon.lower /
    // canon.lift marshaling. If the caller's i32s denote pointers into
    // *its own* linear memory and the target has a different memory,
    // they dereference garbage on the target side — typically corrupting
    // the target's dlmalloc bookkeeping with a static-data address that
    // happens to live at the same numeric offset. We can't distinguish
    // "ptr i32" from "scalar i32" without component-level type info, but
    // the memory mismatch *itself* is the strong signal worth surfacing.
    // One-shot per ctx to keep the log readable.
    if (!ctx.cross_memory_warned and caller_vmctx != 0 and hasPotentialPtr(ctx.param_types)) {
        const vm: *const aot_runtime.VmCtx = @ptrFromInt(@as(usize, @intCast(caller_vmctx)));
        const caller_mb = vm.memory_base;
        if (ctx.target_ai.memories.len > 0) {
            const target_mb = @intFromPtr(ctx.target_ai.memories[0].data.ptr);
            if (caller_mb != 0 and caller_mb != target_mb) {
                ctx.cross_memory_warned = true;
                std.log.warn(
                    "[aot cross-instance] '{s}': caller memory_base=0x{x} != target memory_base=0x{x}; sig has {d} i32 param(s) that may be pointers. " ++
                        "If this import takes string/list args, the call will silently corrupt the target's heap; set WAMR_TRAP_CROSS_MEMORY_THUNK=1 to convert this into a trap.",
                    .{ ctx.label, caller_mb, target_mb, countI32Params(ctx.param_types) },
                );
                if (trapCrossMemoryEnabled()) return error.CrossMemoryFastThunkUnsupported;
            }
        }
    }

    var args_buf: [15]core_types.Value = undefined;
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

fn hasPotentialPtr(types: []const core_types.ValType) bool {
    for (types) |t| if (t == .i32) return true;
    return false;
}

fn countI32Params(types: []const core_types.ValType) usize {
    var n: usize = 0;
    for (types) |t| if (t == .i32) {
        n += 1;
    };
    return n;
}

fn trapCrossMemoryEnabled() bool {
    return core_backend.trapCrossMemoryEnabled();
}

/// Fixed-capacity operand stack for an AOT canon-builtin import. The AOT
/// trampoline hands us raw scalar ABI values rather than an interpreter
/// `ExecEnv`, but the canonical-builtin implementation only needs stack
/// operations. Keeping this adapter deliberately small lets AOT and interp
/// calls share the exact same canonical-ABI state machine.
const AotCanonBuiltinFrame = struct {
    const max_slots = 10;

    values: [max_slots]core_types.Value = undefined,
    sp: usize = 0,

    fn init(
        lowered_sig: host_trampolines.LoweredSig,
        regs: [10]u64,
    ) !AotCanonBuiltinFrame {
        const has_multi_result = lowered_sig.result_types.len > 1;
        if (lowered_sig.has_retptr != has_multi_result) return error.UnsupportedSignature;
        if (lowered_sig.param_types.len + @intFromBool(lowered_sig.has_retptr) > regs.len)
            return error.UnsupportedSignature;
        if (lowered_sig.param_types.len > max_slots or lowered_sig.result_types.len > max_slots)
            return error.UnsupportedSignature;

        var frame = AotCanonBuiltinFrame{};
        for (lowered_sig.param_types, 0..) |ty, i| {
            frame.values[i] = switch (ty) {
                .i32 => .{ .i32 = @bitCast(@as(u32, @truncate(regs[i]))) },
                .i64 => .{ .i64 = @bitCast(regs[i]) },
                .f32 => .{ .f32 = @bitCast(@as(u32, @truncate(regs[i]))) },
                .f64 => .{ .f64 = @bitCast(regs[i]) },
                else => return error.UnsupportedSignature,
            };
        }
        frame.sp = lowered_sig.param_types.len;
        return frame;
    }

    fn pop(self: *AotCanonBuiltinFrame) !core_types.Value {
        if (self.sp == 0) return error.StackUnderflow;
        self.sp -= 1;
        return self.values[self.sp];
    }

    pub fn popI32(self: *AotCanonBuiltinFrame) !i32 {
        const value = try self.pop();
        return switch (value) {
            .i32 => |i| i,
            .f32 => |f| @bitCast(f),
            .funcref, .nonfuncref => 0,
            .externref, .nonexternref => 0,
            .i64 => |i| @as(i32, @bitCast(@as(u32, @truncate(@as(u64, @bitCast(i)))))),
            .f64 => |f| @as(i32, @bitCast(@as(u32, @truncate(@as(u64, @bitCast(f)))))),
            else => 0,
        };
    }

    fn push(self: *AotCanonBuiltinFrame, value: core_types.Value) !void {
        if (self.sp >= self.values.len) return error.StackOverflow;
        self.values[self.sp] = value;
        self.sp += 1;
    }

    pub fn pushI32(self: *AotCanonBuiltinFrame, value: i32) !void {
        try self.push(.{ .i32 = value });
    }

    pub fn pushI64(self: *AotCanonBuiltinFrame, value: i64) !void {
        try self.push(.{ .i64 = value });
    }

    fn rawValue(value: core_types.Value, ty: core_types.ValType) !u64 {
        return switch (ty) {
            .i32 => switch (value) {
                .i32 => |v| @as(u64, @intCast(@as(u32, @bitCast(v)))),
                else => error.UnsupportedSignature,
            },
            .i64 => switch (value) {
                .i64 => |v| @bitCast(v),
                else => error.UnsupportedSignature,
            },
            .f32 => switch (value) {
                .f32 => |v| @as(u64, @intCast(@as(u32, @bitCast(v)))),
                else => error.UnsupportedSignature,
            },
            .f64 => switch (value) {
                .f64 => |v| @bitCast(v),
                else => error.UnsupportedSignature,
            },
            else => error.UnsupportedSignature,
        };
    }

    fn finish(
        self: *const AotCanonBuiltinFrame,
        lowered_sig: host_trampolines.LoweredSig,
        regs: [10]u64,
    ) !u64 {
        if (self.sp != lowered_sig.result_types.len) return error.UnsupportedSignature;

        var raw_results: [max_slots]u64 = undefined;
        for (lowered_sig.result_types, 0..) |ty, i| {
            raw_results[i] = try rawValue(self.values[i], ty);
        }

        if (lowered_sig.has_retptr) {
            const retptr_raw = regs[lowered_sig.param_types.len];
            if (retptr_raw == 0) return error.UnsupportedSignature;
            const retptr: [*]u64 = @ptrFromInt(@as(usize, @intCast(retptr_raw)));
            for (raw_results[1..lowered_sig.result_types.len], 0..) |raw, i| {
                retptr[i] = raw;
            }
        }

        if (lowered_sig.result_types.len == 0) return 0;
        return raw_results[0];
    }
};

fn canonBuiltinOpts(canon: ctypes.Canon) []const ctypes.CanonOpt {
    return switch (canon) {
        .task_return => |info| info.opts,
        .async_canon => |op| switch (op) {
            .stream_read => |info| info.opts,
            .stream_write => |info| info.opts,
            .future_read => |info| info.opts,
            .future_write => |info| info.opts,
            .error_context_new => |info| info.opts,
            .error_context_debug_message => |info| info.opts,
            else => &.{},
        },
        else => &.{},
    };
}

fn canonBuiltinMemoryIdx(canon: ctypes.Canon) ?u32 {
    return switch (canon) {
        .async_canon => |op| switch (op) {
            .waitable_set_wait => |info| info.memory,
            .waitable_set_poll => |info| info.memory,
            else => null,
        },
        else => null,
    };
}

fn resolveAotCanonBuiltinCallCtx(
    ctx: *const CanonBuiltinTrampolineCtx,
    caller_vmctx: u64,
) ?ComponentInstance.CanonLowerCallCtx {
    var lower_opts = LowerOptions.fromOpts(canonBuiltinOpts(ctx.canon));
    if (canonBuiltinMemoryIdx(ctx.canon)) |memory_idx| {
        lower_opts.memory_idx = memory_idx;
    }
    var call_ctx = resolveLowerCallCtx(ctx.comp_inst, lower_opts);

    // The importer's vmctx is authoritative for raw pointer arguments. It
    // matters for stream/future/error-context operations because their
    // pointers are in the importing AOT core's memory, not necessarily the
    // first component memory. Keep a resolved realloc from canon options,
    // while pinning memory to the actual caller.
    if (aotCallerMemory(caller_vmctx)) |memory| {
        if (call_ctx) |*existing| {
            existing.memory = memory;
        } else {
            call_ctx = .{ .memory = memory };
        }
    }
    return call_ctx;
}

/// Canon-builtin dispatcher for AOT-compiled core modules. Mirrors
/// `wamrAotDispatchCrossInstance`'s vmctx-as-first-arg convention: `a0` is
/// the importing core's vmctx and `a1..a9` are lowered wasm args (plus an
/// optional hidden multi-result pointer). All supported canon-builtin source
/// kinds share `dispatchCanonBuiltinWithCtx`, preserving interpreter
/// semantics for context, task, waitable, stream, future, and resource state.
pub export fn wamrAotDispatchCanonBuiltin(
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
    a9: u64,
) callconv(.c) host_trampolines.DispatchResult {
    const ctx: *const CanonBuiltinTrampolineCtx = @ptrCast(@alignCast(ctx_opaque));
    const result = dispatchAotCanonBuiltin(
        ctx,
        lowered_sig.*,
        a0,
        .{ a1, a2, a3, a4, a5, a6, a7, a8, a9, 0 },
    ) catch |err| {
        if (err == error.WasiExit) handleWasiExitFromAotDispatch();
        if (debugAotEnabled()) {
            std.debug.print(
                "[aot-dispatch] canon-builtin '{s}' failed: {s}\n",
                .{ @tagName(ctx.canon), @errorName(err) },
            );
        }
        return .{ .status = 1, .value = 0, .err_name = @errorName(err).ptr };
    };
    return .{ .status = 0, .value = result };
}

fn dispatchAotCanonBuiltin(
    ctx: *const CanonBuiltinTrampolineCtx,
    lowered_sig: host_trampolines.LoweredSig,
    caller_vmctx: u64,
    regs: [10]u64,
) !u64 {
    var frame = try AotCanonBuiltinFrame.init(lowered_sig, regs);
    const saved_call_ctx = ctx.comp_inst.current_lower_call_ctx;
    ctx.comp_inst.current_lower_call_ctx = resolveAotCanonBuiltinCallCtx(ctx, caller_vmctx);
    defer ctx.comp_inst.current_lower_call_ctx = saved_call_ctx;

    try dispatchCanonBuiltinWithCtx(
        ctx.comp_inst,
        ctx.canon,
        ctx,
        &frame,
        ctx.comp_inst.current_task_manager,
        ctx.comp_inst.allocator,
    );
    return frame.finish(lowered_sig, regs);
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
    a9: u64,
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
    _ = a9;
    if (debugAotEnabled()) {
        const label_ptr: *const [*:0]const u8 = @ptrCast(@alignCast(ctx_opaque));
        std.debug.print("[aot-dispatch] trap-stub fired for unbridged import '{s}' (#662 follow-up)\n", .{label_ptr.*});
    }
    return .{ .status = 1, .value = 0 };
}

fn traceCanonValue(mem: ?[]const u8, val: InterfaceValue) void {
    switch (val) {
        .string => |pl| {
            std.debug.print("string(ptr=0x{x},len={d},hex=", .{ pl.ptr, pl.len });
            if (mem) |m| {
                const n: usize = @min(@as(usize, @intCast(pl.len)), @as(usize, 64));
                const start: usize = @intCast(pl.ptr);
                if (start <= m.len and n <= m.len - start) {
                    for (m[start .. start + n]) |b| std.debug.print("{x:0>2}", .{b});
                    if (pl.len > 64) std.debug.print("…", .{});
                } else std.debug.print("<oob>", .{});
            } else std.debug.print("<no-mem>", .{});
            std.debug.print(")", .{});
        },
        .list => |pl| std.debug.print("list(ptr=0x{x},len={d})", .{ pl.ptr, pl.len }),
        .bool => |v| std.debug.print("{any}", .{v}),
        .s8 => |v| std.debug.print("{d}", .{v}),
        .u8 => |v| std.debug.print("{d}", .{v}),
        .s16 => |v| std.debug.print("{d}", .{v}),
        .u16 => |v| std.debug.print("{d}", .{v}),
        .s32 => |v| std.debug.print("{d}", .{v}),
        .u32 => |v| std.debug.print("{d}", .{v}),
        .s64 => |v| std.debug.print("{d}", .{v}),
        .u64 => |v| std.debug.print("{d}", .{v}),
        .f32 => |v| std.debug.print("f32bits=0x{x}", .{v}),
        .f64 => |v| std.debug.print("f64bits=0x{x}", .{v}),
        .char => |v| std.debug.print("char=0x{x}", .{v}),
        .handle => |v| std.debug.print("handle={d}", .{v}),
        else => std.debug.print("…", .{}),
    }
}

fn traceCanonSlice(mem: ?[]const u8, vals: []const InterfaceValue) void {
    std.debug.print("[", .{});
    for (vals, 0..) |v, i| {
        if (i != 0) std.debug.print(",", .{});
        if (i >= 8) {
            std.debug.print("…", .{});
            break;
        }
        traceCanonValue(mem, v);
    }
    std.debug.print("]", .{});
}

fn traceCanonLowerCall(ctx: *const ComponentTrampolineCtx, lowered_sig: host_trampolines.LoweredSig, mem: ?[]const u8, mem_idx: u32, args: []const InterfaceValue, results: []const InterfaceValue, result_dest_ptr: u32) void {
    if (!debugAotEnabled()) return;
    std.debug.print("[canon-lower] slot={d} cfi={d} mem_idx={d} mem_size={d} params=", .{ lowered_sig.slot, ctx.canon_lower_idx, mem_idx, if (mem) |m| m.len else 0 });
    traceCanonSlice(mem, args);
    std.debug.print(" result_ptr=0x{x} result=", .{result_dest_ptr});
    traceCanonSlice(mem, results);
    std.debug.print("\n", .{});
}

fn aotCallerMemory(caller_vmctx: ?u64) ?*core_types.MemoryInstance {
    const raw_vmctx = caller_vmctx orelse return null;
    if (raw_vmctx == 0 or raw_vmctx % @alignOf(aot_runtime.VmCtx) != 0) return null;
    const vmctx: *const aot_runtime.VmCtx = @ptrFromInt(@as(usize, @intCast(raw_vmctx)));
    if (vmctx.instance_ptr == 0 or vmctx.instance_ptr % @alignOf(aot_runtime.AotInstance) != 0)
        return null;
    const caller: *const aot_runtime.AotInstance = @ptrFromInt(vmctx.instance_ptr);
    if (caller.memories.len == 0) return null;
    return caller.memories[0];
}

fn resolveAotLowerCallCtx(
    ctx: *const ComponentTrampolineCtx,
    caller_vmctx: ?u64,
) ?ComponentInstance.CanonLowerCallCtx {
    var call_ctx = resolveLowerCallCtx(ctx.comp_inst, ctx.lower_opts);
    if (aotCallerMemory(caller_vmctx)) |memory| {
        if (call_ctx) |*existing| {
            existing.memory = memory;
        } else {
            call_ctx = .{ .memory = memory };
        }
    }
    return call_ctx;
}

fn dispatchAotAsyncComponentTrampoline(
    ctx: *const ComponentTrampolineCtx,
    lowered_sig: host_trampolines.LoweredSig,
    regs: []const u64,
    caller_vmctx: ?u64,
) !u64 {
    // Async canon.lower always returns exactly one i32 status word. Its
    // optional retptr carries the eventual component result payload, not the
    // AOT ABI's multi-result HRP contract.
    if (lowered_sig.result_types.len != 1 or lowered_sig.result_types[0] != .i32)
        return error.UnsupportedSignature;
    const async_with_result_retptr = ctx.result_types.len > 0;
    if (lowered_sig.has_retptr != async_with_result_retptr)
        return error.UnsupportedSignature;
    if (lowered_sig.param_types.len + @intFromBool(lowered_sig.has_retptr) > regs.len)
        return error.UnsupportedSignature;

    const allocator = ctx.comp_inst.allocator;
    const registry = if (ctx.extended_types.len > 0)
        TypeRegistry.fromExtended(ctx.comp_inst.component, ctx.extended_types, ctx.extended_indexspace)
    else
        TypeRegistry.init(ctx.comp_inst.component);
    const flat_params = countFlatTypes(registry, ctx.param_types);
    const params_spill = flat_params > MAX_FLAT_PARAMS_ASYNC;
    const caller_memory = aotCallerMemory(caller_vmctx) orelse
        ctx.comp_inst.resolveTopLevelMemory(ctx.lower_opts.memory_idx orelse 0);

    if (params_spill or async_with_result_retptr) {
        if (ctx.lower_opts.memory_idx == null) return error.MemoryNotAvailable;
        if (caller_memory == null) return error.MemoryNotAvailable;
    }

    var args_stack_buf: [8]InterfaceValue = undefined;
    const args_heap: ?[]InterfaceValue = if (ctx.param_types.len <= args_stack_buf.len)
        null
    else
        try allocator.alloc(InterfaceValue, ctx.param_types.len);
    defer if (args_heap) |args| allocator.free(args);
    const args: []InterfaceValue = args_heap orelse args_stack_buf[0..ctx.param_types.len];

    var reg_index: usize = 0;
    if (params_spill) {
        if (lowered_sig.param_types.len != 1 or lowered_sig.param_types[0] != .i32)
            return error.UnsupportedSignature;
        const params_ptr: u32 = @truncate(regs[0]);
        reg_index = 1;
        const memory = caller_memory orelse return error.MemoryNotAvailable;
        var offset = params_ptr;
        for (ctx.param_types, 0..) |param_type, i| {
            offset = abi.alignUp(offset, typeAlign(registry, param_type));
            args[i] = try loadInterfaceValue(memory.bytes(), offset, param_type, registry, allocator);
            offset += typeSize(registry, param_type);
        }
    } else {
        for (ctx.param_types, 0..) |param_type, i| {
            args[i] = try liftAotDispatcherArg(
                param_type,
                lowered_sig.param_types,
                &reg_index,
                regs,
                registry,
                allocator,
            );
        }
    }

    var result_dest_ptr: u32 = 0;
    if (async_with_result_retptr) {
        if (reg_index >= regs.len) return error.UnsupportedSignature;
        result_dest_ptr = @truncate(regs[reg_index]);
        reg_index += 1;
    }
    if (reg_index != lowered_sig.param_types.len + @intFromBool(async_with_result_retptr))
        return error.UnsupportedSignature;

    var strict_mem: ?[]const u8 = null;
    for (args, ctx.param_types) |arg, param_type| {
        if (!typeContainsPtrLen(param_type, registry)) continue;
        if (strict_mem == null) {
            const memory = caller_memory orelse return error.MemoryNotAvailable;
            strict_mem = memory.bytes();
        }
        try validateCanonPtrLenValue(strict_mem.?, arg, param_type, registry, "canon-lift");
    }

    // Async lower uses a future/subtask handle in result slot zero even when
    // the lifted component function has no declared result.
    const host_result_len: usize = if (ctx.result_types.len == 0) 1 else ctx.result_types.len;
    var results_stack_buf: [4]InterfaceValue = undefined;
    const results_heap: ?[]InterfaceValue = if (host_result_len <= results_stack_buf.len)
        null
    else
        try allocator.alloc(InterfaceValue, host_result_len);
    const results: []InterfaceValue = results_heap orelse results_stack_buf[0..host_result_len];
    var results_filled: usize = 0;
    defer {
        for (results[0..results_filled]) |result| result.deinit(allocator);
        if (results_heap) |allocated| allocator.free(allocated);
    }
    if (ctx.result_types.len == 0) {
        results[0] = .{ .u32 = 0 };
        results_filled = 1;
    }

    const saved_lower_ctx = ctx.comp_inst.current_lower_call_ctx;
    ctx.comp_inst.current_lower_call_ctx = resolveAotLowerCallCtx(ctx, caller_vmctx);
    defer ctx.comp_inst.current_lower_call_ctx = saved_lower_ctx;
    if (ctx.lift_target) |target| {
        try callComponentFuncByLocal(ctx.comp_inst, target, args, results, allocator);
    } else {
        const call = ctx.host_func.call orelse return error.HostFuncNotBound;
        try call(ctx.host_func.context, ctx.comp_inst, args, results, allocator);
    }
    results_filled = results.len;

    const handle: u32 = switch (results[0]) {
        .handle => |value| value,
        .u32 => |value| value,
        else => 0,
    };
    if (handle != 0) {
        if (ctx.comp_inst.futures.getPtr(handle)) |future| {
            future.subtask_managed = true;
            if (async_with_result_retptr) {
                if (future.state == .ready or future.state == .closed) {
                    if (future.payload) |bytes| {
                        const memory = caller_memory orelse return error.MemoryNotAvailable;
                        if (@as(u64, result_dest_ptr) + bytes.len > memory.byteLen())
                            return error.MemoryNotAvailable;
                        @memcpy(memory.data[result_dest_ptr .. result_dest_ptr + bytes.len], bytes);
                    }
                } else {
                    future.async_lower_retptr = result_dest_ptr;
                }
            }
        }
    }

    return @as(u64, packAsyncLowerStatus(ctx.comp_inst, handle));
}

fn dispatchAotComponentTrampoline(
    ctx: *const ComponentTrampolineCtx,
    lowered_sig: host_trampolines.LoweredSig,
    regs: []const u64,
    caller_vmctx: ?u64,
) !u64 {
    if (ctx.lower_opts.is_async or ctx.is_async_func)
        return dispatchAotAsyncComponentTrampoline(ctx, lowered_sig, regs, caller_vmctx);
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

    var strict_mem: ?[]const u8 = null;
    var reg_index: usize = 0;
    for (ctx.param_types, 0..) |pt, i| {
        args[i] = try liftAotDispatcherArg(pt, lowered_sig.param_types, &reg_index, regs, registry, allocator);
        if (typeContainsPtrLen(pt, registry)) {
            if (strict_mem == null) strict_mem = try strictCanonMemory(ctx);
            try validateCanonPtrLenValue(strict_mem.?, args[i], pt, registry, "canon-lift");
        }
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
    // `host_func.call` only initialises the prefix of `results` it
    // successfully fills before returning an error; the remainder is
    // backed by `undefined` stack/heap memory. We track how many
    // entries are valid and only `deinit` that prefix. Without this
    // a failing host (e.g. `getEnvironment` -> `error.IoError` when
    // `cabi_realloc` is unavailable) used to dereference an
    // `undefined` tag in `InterfaceValue.deinit`'s switch and panic
    // with "switch on corrupt value".
    var results_filled: usize = 0;
    defer {
        for (results[0..results_filled]) |r| r.deinit(allocator);
        if (results_heap) |h| allocator.free(h);
    }

    if (ctx.lift_target) |target| {
        try callComponentFuncByLocal(ctx.comp_inst, target, args, results, allocator);
    } else {
        const call = ctx.host_func.call orelse return error.HostFuncNotBound;
        const saved_lower_ctx = ctx.comp_inst.current_lower_call_ctx;
        ctx.comp_inst.current_lower_call_ctx = resolveLowerCallCtx(ctx.comp_inst, ctx.lower_opts);
        defer ctx.comp_inst.current_lower_call_ctx = saved_lower_ctx;
        try call(ctx.host_func.context, ctx.comp_inst, args, results, allocator);
    }
    results_filled = results.len;
    if (debugAotEnabled()) {
        const mem_idx = ctx.lower_opts.memory_idx orelse 0;
        const trace_mem = if (ctx.comp_inst.resolveTopLevelMemory(mem_idx)) |mem| mem.bytes() else null;
        traceCanonLowerCall(ctx, lowered_sig, trace_mem, mem_idx, args, results, result_dest_ptr);
    }

    if (lowered_sig.has_retptr) {
        // Default to memory 0 when canon-lower has no explicit `.memory` opt,
        // mirroring the canon-lift path (see line ~321). The component's
        // single core memory is what the AOT caller's retptr points into.
        const mem_idx = ctx.lower_opts.memory_idx orelse 0;
        const mem = ctx.comp_inst.resolveTopLevelMemory(mem_idx) orelse return error.MemoryNotAvailable;
        var offset: u32 = result_dest_ptr;
        for (results, ctx.result_types) |r, t| {
            if (typeContainsPtrLen(t, registry)) {
                validateCanonPtrLenValue(mem.bytes(), r, t, registry, "canon-lift") catch return error.LiftedResultInvariantViolated;
            }
            const al = typeAlign(registry, t);
            offset = abi.alignUp(offset, al);
            try storeInterfaceValue(mem.bytes(), offset, r, t, registry);
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
    if (typeContainsPtrLen(ctx.result_types[0], registry)) {
        const mem = strict_mem orelse try strictCanonMemory(ctx);
        validateCanonPtrLenValue(mem, results[0], ctx.result_types[0], registry, "canon-lift") catch return error.LiftedResultInvariantViolated;
    }
    return lowerAotDispatcherResult(results[0], ctx.result_types[0], lowered_sig.result_types[0], registry);
}

fn liftAotDispatcherArg(
    t: ctypes.ValType,
    lowered_param_types: []const core_types.ValType,
    reg_index: *usize,
    regs: []const u64,
    registry: TypeRegistry,
    allocator: Allocator,
) !InterfaceValue {
    // Generic slot-driven lift: derive the wasm flat-slot count from the
    // canonical ABI's `flattenCount`, then walk that many lowered core
    // slots converting each into the u32-cell encoding expected by
    // `abi.liftFlatReg` (i32/f32 -> 1 cell; i64/f64 -> 2 cells lo|hi).
    // This covers every primitive the narrow per-type switch used to
    // enumerate (bool/sN/uN/char/own/borrow/future/stream/error_context/
    // enum_/s64/u64/string/list) plus the compound shapes the canonical
    // ABI flattens into multiple slots (type_idx, record, tuple, variant,
    // option, result, flags). Without the compound coverage every
    // canon.lower import carrying a compound param lands in the old
    // `else` arm and rejects with `UnsupportedSignature` — silently
    // collapsed to `return 0` by `genericDispatcher`
    // (`host_trampolines.zig:175`). Issue #707.
    const slot_count = abi.flattenCount(registry, t);
    if (slot_count == 0) return error.UnsupportedSignature;
    if (reg_index.* + slot_count > lowered_param_types.len) return error.UnsupportedSignature;

    var cell_buf: [MAX_FLAT_PARAMS * 2]u32 = undefined;
    var cells_used: usize = 0;
    var k: usize = 0;
    while (k < slot_count) : (k += 1) {
        const lpt = lowered_param_types[reg_index.* + k];
        const raw = regs[reg_index.* + k];
        switch (lpt) {
            .i32, .f32 => {
                if (cells_used >= cell_buf.len) return error.UnsupportedSignature;
                cell_buf[cells_used] = @truncate(raw);
                cells_used += 1;
            },
            .i64, .f64 => {
                if (cells_used + 2 > cell_buf.len) return error.UnsupportedSignature;
                cell_buf[cells_used] = @truncate(raw);
                cell_buf[cells_used + 1] = @truncate(raw >> 32);
                cells_used += 2;
            },
            else => return error.UnsupportedSignature,
        }
    }
    reg_index.* += slot_count;
    // Use the local `liftFlatReg` (above in this file) rather than
    // `abi.liftFlatReg`: the local copy has full compound coverage —
    // `.record`, `.tuple`, `.flags`, plus all `.type_idx` reifications
    // (including `.list`). The `canonical_abi.liftFlatReg` copy falls
    // back to `liftFlat` for `.record/.tuple/.flags` and returns
    // `CompoundNeedsRegistry` — which would silently zero-return to
    // the guest through `genericDispatcher`. Issue #707.
    const r = liftFlatReg(cell_buf[0..cells_used], t, registry, allocator) catch return error.UnsupportedSignature;
    return r.val;
}

fn lowerAotDispatcherResult(
    val: InterfaceValue,
    t: ctypes.ValType,
    lowered_ty: core_types.ValType,
    registry: TypeRegistry,
) error{UnsupportedSignature}!u64 {
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
        .enum_ => @as(u64, switch (val) {
            .enum_val => |e| e,
            .variant_val => |v| v.discriminant,
            else => return error.UnsupportedSignature,
        }),
        .own, .borrow => @as(u64, encodeResourceWire(val.handle)),
        .future, .stream, .error_context => @as(u64, val.handle),
        .result => @as(u64, if (val.result_val.is_ok) 0 else 1),
        .variant => @as(u64, switch (val) {
            .variant_val => |v| v.discriminant,
            .enum_val => |e| e,
            else => return error.UnsupportedSignature,
        }),
        .option => @as(u64, if (val.option_val.is_some) 1 else 0),
        // Resolve a typedef-encoded result through the registry and recurse.
        // The shape is unwrapped exactly as `coreFlatSlotType` does for
        // `.type_idx`, so it lands on one of the inline arms above (which
        // are already guarded by the `flattenCount == 1` check inside
        // `coreFlatSlotType`). Without this arm any canon.lower import
        // returning a typedef-bound compound (e.g. an exported
        // `result<unit, error-code>` alias) rejects with
        // `UnsupportedSignature` and silently zero-returns to the guest
        // via `genericDispatcher` (`host_trampolines.zig:175`). Issue #707.
        .type_idx => |idx| blk: {
            const td = registry.get(idx) orelse return error.UnsupportedSignature;
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
                else => return error.UnsupportedSignature,
            };
            break :blk try lowerAotDispatcherResult(val, reified, lowered_ty, registry);
        },
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
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &type_defs,
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
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
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &type_defs,
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
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

// ── #701: AOT canon-builtin dispatcher ──────────────────────────────────────

test "wamrAotDispatchCanonBuiltin: resource.new/rep/drop round-trip + on_resource_drop hook (#701)" {
    // End-to-end test of the AOT-side canon-builtin dispatcher added in
    // #701. Drives a `*ComponentInstance` through `wamrAotDispatchCanonBuiltin`
    // for each of the three in-scope canon kinds and verifies the resource
    // table mutates correctly + the `on_resource_drop` hook fires.
    const testing = std.testing;
    const allocator = testing.allocator;

    var comp = ctypes.Component{
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
    const inst = try instance_mod.instantiate(&comp, allocator);
    defer inst.deinit();

    // Wire a drop-hook so we can assert it fires with the right
    // (resource_idx, handle) pair when `.resource_drop` dispatches.
    const HookState = struct {
        var fires: u32 = 0;
        var last_resource_idx: u32 = 0xFFFF_FFFF;
        var last_handle: u32 = 0xFFFF_FFFF;
        fn reset() void {
            fires = 0;
            last_resource_idx = 0xFFFF_FFFF;
            last_handle = 0xFFFF_FFFF;
        }
        fn cb(_: ?*anyopaque, _: *ComponentInstance, ridx: u32, h: u32) void {
            fires += 1;
            last_resource_idx = ridx;
            last_handle = h;
        }
    };
    HookState.reset();
    inst.on_resource_drop = &HookState.cb;
    inst.on_resource_drop_ctx = null;

    const resource_idx: u32 = 7;
    const new_ctx = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .resource_new = resource_idx },
    };
    const drop_ctx = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .resource_drop = resource_idx },
    };
    const rep_ctx = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .resource_rep = resource_idx },
    };

    const i32_one = [_]core_types.ValType{.i32};
    const sig_one_to_one = host_trampolines.LoweredSig{
        .param_types = &i32_one,
        .result_types = &i32_one,
    };
    const sig_one_to_void = host_trampolines.LoweredSig{
        .param_types = &i32_one,
        .result_types = &.{},
    };

    // resource.new(rep=123) → handle.  Convention: a0 is importer's
    // vmctx (ignored), a1 is the wasm arg.
    const new_res = wamrAotDispatchCanonBuiltin(
        @ptrCast(@constCast(&new_ctx)),
        &sig_one_to_one,
        0xDEAD,
        123,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    try testing.expectEqual(@as(u32, 0), new_res.status);
    const handle: u32 = @intCast(new_res.value);
    // ResourceTable.new returns the slot index, starting at 0 for a fresh
    // table — don't assume handle != 0; round-trip via rep / drop instead.

    // resource.rep(handle) → 123. Resource table is keyed by the
    // canon's `resource_idx` immediate, so use the same idx as new.
    const rep_res = wamrAotDispatchCanonBuiltin(
        @ptrCast(@constCast(&rep_ctx)),
        &sig_one_to_one,
        0,
        handle,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    try testing.expectEqual(@as(u32, 0), rep_res.status);
    try testing.expectEqual(@as(u64, 123), rep_res.value);

    // resource.drop(handle). Status=0, value=0 (drop returns no
    // i32 result on the wasm side). Hook must fire exactly once
    // with (resource_idx, handle).
    const drop_res = wamrAotDispatchCanonBuiltin(
        @ptrCast(@constCast(&drop_ctx)),
        &sig_one_to_void,
        0,
        handle,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    try testing.expectEqual(@as(u32, 0), drop_res.status);
    try testing.expectEqual(@as(u64, 0), drop_res.value);
    try testing.expectEqual(@as(u32, 1), HookState.fires);
    try testing.expectEqual(resource_idx, HookState.last_resource_idx);
    try testing.expectEqual(handle, HookState.last_handle);

    // After drop, resource.rep on the freed handle returns 0 (canonResourceRep
    // returns null → dispatcher coerces to 0).
    const rep_after = wamrAotDispatchCanonBuiltin(
        @ptrCast(@constCast(&rep_ctx)),
        &sig_one_to_one,
        0,
        handle,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    try testing.expectEqual(@as(u32, 0), rep_after.status);
    try testing.expectEqual(@as(u64, 0), rep_after.value);
}

test "wamrAotDispatchCanonBuiltin: async context, task, waitable, future, and multi-result ABI (#881)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const Invoke = struct {
        fn call(
            ctx: *CanonBuiltinTrampolineCtx,
            sig: *const host_trampolines.LoweredSig,
            args: [9]u64,
        ) host_trampolines.DispatchResult {
            return wamrAotDispatchCanonBuiltin(
                @ptrCast(ctx),
                sig,
                0,
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
                args[5],
                args[6],
                args[7],
                args[8],
            );
        }
    };

    var comp = ctypes.Component{
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
    const inst = try instance_mod.instantiate(&comp, allocator);
    defer inst.deinit();
    try inst.enableTestMem(allocator, 256);
    defer inst.disableTestMem();

    const no_slots = [_]core_types.ValType{};
    const one_i32 = [_]core_types.ValType{.i32};
    const two_i32 = [_]core_types.ValType{ .i32, .i32 };
    const one_i64 = [_]core_types.ValType{.i64};
    const no_to_i32 = host_trampolines.LoweredSig{
        .param_types = &no_slots,
        .result_types = &one_i32,
    };
    const no_to_i64 = host_trampolines.LoweredSig{
        .param_types = &no_slots,
        .result_types = &one_i64,
    };
    const one_to_none = host_trampolines.LoweredSig{
        .param_types = &one_i32,
        .result_types = &no_slots,
    };
    const two_to_none = host_trampolines.LoweredSig{
        .param_types = &two_i32,
        .result_types = &no_slots,
    };
    const two_to_i32 = host_trampolines.LoweredSig{
        .param_types = &two_i32,
        .result_types = &one_i32,
    };

    var context_set = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .context_set = .{ .val_type = .i32, .slot = 0 } },
    };
    var context_get = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .context_get = .{ .val_type = .i32, .slot = 0 } },
    };
    const set_result = Invoke.call(&context_set, &one_to_none, .{ 0xCAFE_F00D, 0, 0, 0, 0, 0, 0, 0, 0 });
    try testing.expectEqual(@as(u32, 0), set_result.status);
    const get_result = Invoke.call(&context_get, &no_to_i32, .{ 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    try testing.expectEqual(@as(u32, 0), get_result.status);
    try testing.expectEqual(@as(u64, 0xCAFE_F00D), get_result.value);

    var waitable_new = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .async_canon = .waitable_set_new },
    };
    const ws_result = Invoke.call(&waitable_new, &no_to_i32, .{ 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    try testing.expectEqual(@as(u32, 0), ws_result.status);
    const waitable_set_handle: u32 = @truncate(ws_result.value);

    var future_new = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .async_canon = .{ .future_new = .{ .type_idx = 0 } } },
    };
    const future_result = Invoke.call(&future_new, &no_to_i64, .{ 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    try testing.expectEqual(@as(u32, 0), future_result.status);
    const future_handle: u32 = @truncate(future_result.value);

    var waitable_join = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .async_canon = .waitable_join },
    };
    const join_result = Invoke.call(&waitable_join, &two_to_none, .{ future_handle, waitable_set_handle, 0, 0, 0, 0, 0, 0, 0 });
    try testing.expectEqual(@as(u32, 0), join_result.status);
    try testing.expect(inst.futures.getPtr(future_handle).?.waitable_set != null);

    var waitable_poll = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .async_canon = .{ .waitable_set_poll = .{ .cancellable = false, .memory = 0 } } },
    };
    const poll_result = Invoke.call(&waitable_poll, &two_to_i32, .{ waitable_set_handle, 0, 0, 0, 0, 0, 0, 0, 0 });
    try testing.expectEqual(@as(u32, 0), poll_result.status);
    try testing.expectEqual(@as(u64, @intFromEnum(async_canon.EventCode.none)), poll_result.value);

    var task_manager = async_mod.TaskManager{};
    defer task_manager.deinit(allocator);
    const task = try task_manager.createTask(allocator);
    task_manager.startTask(task);
    task_manager.current_task = task;
    inst.current_task_manager = &task_manager;
    defer inst.current_task_manager = null;

    var task_cancel = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .async_canon = .task_cancel },
    };
    const cancel_result = Invoke.call(&task_cancel, &host_trampolines.LoweredSig{
        .param_types = &no_slots,
        .result_types = &no_slots,
    }, .{ 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    try testing.expectEqual(@as(u32, 0), cancel_result.status);
    try testing.expectEqual(async_mod.TaskState.cancelled, task_manager.getState(task).?);

    const message = "aot async builtin";
    try inst.error_contexts.put(allocator, 77, try allocator.dupe(u8, message));
    var error_debug = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .async_canon = .{ .error_context_debug_message = .{ .opts = &.{} } } },
    };
    const two_result_sig = host_trampolines.LoweredSig{
        .param_types = &one_i32,
        .result_types = &two_i32,
        .has_retptr = true,
    };
    var result_tail: [1]u64 = .{0};
    const debug_result = Invoke.call(
        &error_debug,
        &two_result_sig,
        .{ 77, @intFromPtr(&result_tail), 0, 0, 0, 0, 0, 0, 0 },
    );
    try testing.expectEqual(@as(u32, 0), debug_result.status);
    const message_ptr: u32 = @truncate(debug_result.value);
    const message_len: u32 = @truncate(result_tail[0]);
    try testing.expectEqual(@as(u32, message.len), message_len);
    try testing.expectEqualStrings(message, inst.readGuestBytes(message_ptr, message_len).?);
}

test "wamrAotDispatchCanonBuiltin: rejects malformed lowered_sig (#701)" {
    // Unsupported raw source shapes must return a failing DispatchResult
    // instead of reading an invalid register slot or silently falling back
    // to a trap stub.
    const testing = std.testing;
    const allocator = testing.allocator;

    var comp = ctypes.Component{
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
    const inst = try instance_mod.instantiate(&comp, allocator);
    defer inst.deinit();

    const drop_ctx = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .resource_drop = 0 },
    };

    // Wrong shape: drop with a result; reject.
    const i32_one = [_]core_types.ValType{.i32};
    const bad_sig = host_trampolines.LoweredSig{
        .param_types = &i32_one,
        .result_types = &i32_one,
    };
    const res = wamrAotDispatchCanonBuiltin(
        @ptrCast(@constCast(&drop_ctx)),
        &bad_sig,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    try testing.expectEqual(@as(u32, 1), res.status);

    // Non-scalar source args are not valid for the native AOT ABI relay.
    const ctx_get = CanonBuiltinTrampolineCtx{
        .comp_inst = inst,
        .canon = .{ .context_get = .{ .val_type = .i32, .slot = 0 } },
    };
    const v128_one = [_]core_types.ValType{.v128};
    const unsupported_source_sig = host_trampolines.LoweredSig{
        .param_types = &v128_one,
        .result_types = &i32_one,
    };
    const res2 = wamrAotDispatchCanonBuiltin(
        @ptrCast(@constCast(&ctx_get)),
        &unsupported_source_sig,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    try testing.expectEqual(@as(u32, 1), res2.status);
}

test "marshalValueAcrossMemory: nested string in result<string, _> is translated (#719 Bug B)" {
    const testing = std.testing;

    var comp = ctypes.Component{
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
    const src = try instance_mod.instantiate(&comp, testing.allocator);
    defer src.deinit();
    try src.enableTestMem(testing.allocator, 4096);
    defer src.disableTestMem();

    const dst = try instance_mod.instantiate(&comp, testing.allocator);
    defer dst.deinit();
    try dst.enableTestMem(testing.allocator, 4096);
    defer dst.disableTestMem();

    // Plant the source string at a known offset; bump the dst's
    // allocator a bit first so the translated ptr can't accidentally
    // equal the source ptr.
    _ = dst.hostAllocGuest(64, 1).?;
    const msg = "hello cross-memory";
    const src_ptr = src.hostAllocAndWrite(msg, 1).?;

    // Build result<string, _> with the ok arm carrying our string.
    const inner: InterfaceValue = .{ .string = .{ .ptr = src_ptr, .len = @intCast(msg.len) } };
    const val: InterfaceValue = .{ .result_val = .{ .is_ok = true, .payload = &inner } };

    try testing.expect(interfaceValueContainsPtrLen(val));

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const out = try marshalValueAcrossMemory(src, dst, val, arena.allocator());

    try testing.expect(out.result_val.is_ok);
    const new_pl = out.result_val.payload.?.*;
    switch (new_pl) {
        .string => |pl| {
            try testing.expect(pl.ptr != src_ptr);
            try testing.expectEqual(@as(u32, msg.len), pl.len);
            const dst_bytes = dst.readGuestBytes(pl.ptr, pl.len).?;
            try testing.expectEqualStrings(msg, dst_bytes);
        },
        else => return error.UnexpectedTag,
    }
}

test "marshalValueAcrossMemory: string inside record_val is translated (#719 Bug B)" {
    const testing = std.testing;

    var comp = ctypes.Component{
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
    const src = try instance_mod.instantiate(&comp, testing.allocator);
    defer src.deinit();
    try src.enableTestMem(testing.allocator, 4096);
    defer src.disableTestMem();

    const dst = try instance_mod.instantiate(&comp, testing.allocator);
    defer dst.deinit();
    try dst.enableTestMem(testing.allocator, 4096);
    defer dst.disableTestMem();

    _ = dst.hostAllocGuest(128, 1).?;
    const name = "bug-b";
    const name_ptr = src.hostAllocAndWrite(name, 1).?;

    // record { name: string, age: u32 }
    const fields = [_]InterfaceValue{
        .{ .string = .{ .ptr = name_ptr, .len = @intCast(name.len) } },
        .{ .u32 = 42 },
    };
    const val: InterfaceValue = .{ .record_val = &fields };

    try testing.expect(interfaceValueContainsPtrLen(val));

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const out = try marshalValueAcrossMemory(src, dst, val, arena.allocator());

    const out_fields = out.record_val;
    try testing.expectEqual(@as(usize, 2), out_fields.len);
    const name_field = out_fields[0];
    switch (name_field) {
        .string => |pl| {
            try testing.expect(pl.ptr != name_ptr);
            try testing.expectEqual(@as(u32, name.len), pl.len);
            const dst_bytes = dst.readGuestBytes(pl.ptr, pl.len).?;
            try testing.expectEqualStrings(name, dst_bytes);
        },
        else => return error.UnexpectedTag,
    }
    try testing.expectEqual(@as(u32, 42), out_fields[1].u32);
}

test "interfaceValueContainsPtrLen: returns false for pure scalars" {
    const testing = std.testing;
    try testing.expect(!interfaceValueContainsPtrLen(.{ .u32 = 1 }));
    try testing.expect(!interfaceValueContainsPtrLen(.{ .s64 = -1 }));
    try testing.expect(!interfaceValueContainsPtrLen(.{ .bool = true }));
    try testing.expect(!interfaceValueContainsPtrLen(.{ .enum_val = 3 }));
    // result<_, _> with no payload also has no PtrLen.
    try testing.expect(!interfaceValueContainsPtrLen(.{
        .result_val = .{ .is_ok = true, .payload = null },
    }));
    // option<u32> with payload is still scalar inside.
    const u32_payload: InterfaceValue = .{ .u32 = 7 };
    try testing.expect(!interfaceValueContainsPtrLen(.{
        .option_val = .{ .is_some = true, .payload = &u32_payload },
    }));
}

test "rewriteResultsBackToCaller: nested string in result<string, _> translated callee→caller (#719 Path 3)" {
    const testing = std.testing;

    var comp = ctypes.Component{
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
    const caller = try instance_mod.instantiate(&comp, testing.allocator);
    defer caller.deinit();
    try caller.enableTestMem(testing.allocator, 4096);
    defer caller.disableTestMem();

    const callee = try instance_mod.instantiate(&comp, testing.allocator);
    defer callee.deinit();
    try callee.enableTestMem(testing.allocator, 4096);
    defer callee.disableTestMem();

    // Bump the caller's bump-allocator first so the freshly-translated
    // ptr can't accidentally equal the callee ptr by sheer luck.
    _ = caller.hostAllocGuest(64, 1).?;
    const msg = "callee-allocated";
    const callee_ptr = callee.hostAllocAndWrite(msg, 1).?;

    // Build a result<string, _> as it would arrive from the callee.
    const inner_str: InterfaceValue = .{ .string = .{ .ptr = callee_ptr, .len = @intCast(msg.len) } };
    const inner_copy = try testing.allocator.create(InterfaceValue);
    inner_copy.* = inner_str;
    var out_results = [_]InterfaceValue{
        .{ .result_val = .{ .is_ok = true, .payload = inner_copy } },
    };
    defer for (out_results) |r| r.deinit(testing.allocator);

    try rewriteResultsBackToCaller(caller, callee, &out_results, testing.allocator);

    try testing.expect(out_results[0].result_val.is_ok);
    const new_pl = out_results[0].result_val.payload.?.*;
    switch (new_pl) {
        .string => |pl| {
            try testing.expect(pl.ptr != callee_ptr);
            try testing.expectEqual(@as(u32, msg.len), pl.len);
            const caller_bytes = caller.readGuestBytes(pl.ptr, pl.len).?;
            try testing.expectEqualStrings(msg, caller_bytes);
        },
        else => return error.UnexpectedTag,
    }
}

test "rewriteResultsBackToCaller: no-op when caller == callee" {
    const testing = std.testing;

    var comp = ctypes.Component{
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
    const inst = try instance_mod.instantiate(&comp, testing.allocator);
    defer inst.deinit();
    try inst.enableTestMem(testing.allocator, 4096);
    defer inst.disableTestMem();

    const msg = "intra-component";
    const ptr = inst.hostAllocAndWrite(msg, 1).?;
    var out_results = [_]InterfaceValue{
        .{ .string = .{ .ptr = ptr, .len = @intCast(msg.len) } },
    };

    try rewriteResultsBackToCaller(inst, inst, &out_results, testing.allocator);

    // No rewrite, ptr unchanged.
    try testing.expectEqual(ptr, out_results[0].string.ptr);
    try testing.expectEqual(@as(u32, msg.len), out_results[0].string.len);
}

test "valTypeHasPtrLen: scalars vs strings vs lists vs nested compounds (#726)" {
    const testing = std.testing;

    var comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{
            // 0: list<string>
            .{ .list = .{ .element = .string } },
            // 1: list<u32>
            .{ .list = .{ .element = .u32 } },
            // 2: record { name: string, age: u32 }
            .{ .record = .{ .fields = &.{
                .{ .name = "name", .type = .string },
                .{ .name = "age", .type = .u32 },
            } } },
            // 3: record { id: u32, name: string }
            // 4: tuple<u32, u32>
            .{ .tuple = .{ .fields = &.{ .u32, .u32 } } },
        },
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    const reg = abi.TypeRegistry.init(&comp);

    try testing.expect(!valTypeHasPtrLen(reg, .u32));
    try testing.expect(!valTypeHasPtrLen(reg, .u64));
    try testing.expect(!valTypeHasPtrLen(reg, .bool));
    try testing.expect(valTypeHasPtrLen(reg, .string));
    try testing.expect(valTypeHasPtrLen(reg, .{ .list = 0 })); // list<string>
    try testing.expect(valTypeHasPtrLen(reg, .{ .list = 1 })); // list<u32> — still a PtrLen itself
    try testing.expect(valTypeHasPtrLen(reg, .{ .record = 2 })); // record has string field
    try testing.expect(!valTypeHasPtrLen(reg, .{ .tuple = 4 })); // pure u32 tuple
}

test "marshalValueAcrossMemoryTyped: list<string> element ptrs translated to dst memory (#726)" {
    const testing = std.testing;

    var comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{
            // 0: list<string>
            .{ .list = .{ .element = .string } },
        },
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    const src = try instance_mod.instantiate(&comp, testing.allocator);
    defer src.deinit();
    try src.enableTestMem(testing.allocator, 4096);
    defer src.disableTestMem();

    const dst = try instance_mod.instantiate(&comp, testing.allocator);
    defer dst.deinit();
    try dst.enableTestMem(testing.allocator, 4096);
    defer dst.disableTestMem();

    // Skew dst's bump so dst pointers can't accidentally equal src pointers.
    _ = dst.hostAllocGuest(128, 1).?;

    // Plant the three element string payloads in src.
    const msgs = [_][]const u8{ "alpha", "beta", "gamma-longer" };
    var src_ptrs: [3]u32 = undefined;
    for (msgs, 0..) |m, i| {
        src_ptrs[i] = src.hostAllocAndWrite(m, 1).?;
    }

    // Plant the (ptr, len) array (3 elements × 8 bytes = 24 bytes) in src.
    const list_ptr = src.hostAllocGuest(24, 4).?;
    {
        const list_bytes = src.writableGuestBytes(list_ptr, 24).?;
        for (msgs, 0..) |m, i| {
            const off = i * 8;
            std.mem.writeInt(u32, list_bytes[off..][0..4], src_ptrs[i], .little);
            std.mem.writeInt(u32, list_bytes[off + 4 ..][0..4], @intCast(m.len), .little);
        }
    }

    const val: InterfaceValue = .{ .list = .{ .ptr = list_ptr, .len = @intCast(msgs.len) } };

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const reg = abi.TypeRegistry.init(&comp);
    const out = try marshalValueAcrossMemoryTyped(src, dst, val, .{ .list = 0 }, reg, arena.allocator());

    // Outer list ptr must have moved into dst.
    try testing.expect(out.list.ptr != list_ptr);
    try testing.expectEqual(@as(u32, msgs.len), out.list.len);

    // Read the translated (ptr, len) array out of dst and verify each
    // element ptr is in dst (not in src) and payload bytes match.
    const dst_list_bytes = dst.readGuestBytes(out.list.ptr, 24).?;
    for (msgs, 0..) |m, i| {
        const off = i * 8;
        const new_elem_ptr = std.mem.readInt(u32, dst_list_bytes[off..][0..4], .little);
        const new_elem_len = std.mem.readInt(u32, dst_list_bytes[off + 4 ..][0..4], .little);
        try testing.expect(new_elem_ptr != src_ptrs[i]);
        try testing.expectEqual(@as(u32, @intCast(m.len)), new_elem_len);
        const new_bytes = dst.readGuestBytes(new_elem_ptr, new_elem_len).?;
        try testing.expectEqualStrings(m, new_bytes);
    }
}

test "marshalValueAcrossMemoryTyped: list<u32> uses fast bytewise copy (#726)" {
    const testing = std.testing;

    var comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{
            // 0: list<u32>
            .{ .list = .{ .element = .u32 } },
        },
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    const src = try instance_mod.instantiate(&comp, testing.allocator);
    defer src.deinit();
    try src.enableTestMem(testing.allocator, 4096);
    defer src.disableTestMem();

    const dst = try instance_mod.instantiate(&comp, testing.allocator);
    defer dst.deinit();
    try dst.enableTestMem(testing.allocator, 4096);
    defer dst.disableTestMem();

    _ = dst.hostAllocGuest(64, 1).?;

    const values = [_]u32{ 0x11111111, 0x22222222, 0x33333333, 0x44444444 };
    const total_bytes: u32 = @intCast(values.len * 4);
    const src_ptr = src.hostAllocGuest(total_bytes, 4).?;
    {
        const sb = src.writableGuestBytes(src_ptr, total_bytes).?;
        for (values, 0..) |v, i| {
            std.mem.writeInt(u32, sb[i * 4 ..][0..4], v, .little);
        }
    }

    const val: InterfaceValue = .{ .list = .{ .ptr = src_ptr, .len = @intCast(values.len) } };
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const reg = abi.TypeRegistry.init(&comp);
    const out = try marshalValueAcrossMemoryTyped(src, dst, val, .{ .list = 0 }, reg, arena.allocator());

    try testing.expect(out.list.ptr != src_ptr);
    try testing.expectEqual(@as(u32, values.len), out.list.len);
    const dst_bytes = dst.readGuestBytes(out.list.ptr, total_bytes).?;
    for (values, 0..) |v, i| {
        const got = std.mem.readInt(u32, dst_bytes[i * 4 ..][0..4], .little);
        try testing.expectEqual(v, got);
    }
}

test "rewriteValueBackToCallerOwnedTyped: list<string> translated callee→caller (#726)" {
    const testing = std.testing;

    var comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{
            .{ .list = .{ .element = .string } },
        },
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    const caller = try instance_mod.instantiate(&comp, testing.allocator);
    defer caller.deinit();
    try caller.enableTestMem(testing.allocator, 4096);
    defer caller.disableTestMem();

    const callee = try instance_mod.instantiate(&comp, testing.allocator);
    defer callee.deinit();
    try callee.enableTestMem(testing.allocator, 4096);
    defer callee.disableTestMem();

    _ = caller.hostAllocGuest(96, 1).?;

    // Plant a list<string> with two elements in callee memory.
    const msgs = [_][]const u8{ "first-result", "second" };
    var callee_ptrs: [2]u32 = undefined;
    for (msgs, 0..) |m, i| {
        callee_ptrs[i] = callee.hostAllocAndWrite(m, 1).?;
    }
    const list_ptr = callee.hostAllocGuest(16, 4).?;
    {
        const lb = callee.writableGuestBytes(list_ptr, 16).?;
        for (msgs, 0..) |m, i| {
            std.mem.writeInt(u32, lb[i * 8 ..][0..4], callee_ptrs[i], .little);
            std.mem.writeInt(u32, lb[i * 8 + 4 ..][0..4], @intCast(m.len), .little);
        }
    }

    const original: InterfaceValue = .{ .list = .{ .ptr = list_ptr, .len = @intCast(msgs.len) } };
    const reg = abi.TypeRegistry.init(&comp);
    const out = try rewriteValueBackToCallerOwnedTyped(caller, callee, original, .{ .list = 0 }, reg, testing.allocator);
    defer out.deinit(testing.allocator);

    try testing.expect(out.list.ptr != list_ptr);
    try testing.expectEqual(@as(u32, msgs.len), out.list.len);
    const cb = caller.readGuestBytes(out.list.ptr, 16).?;
    for (msgs, 0..) |m, i| {
        const np = std.mem.readInt(u32, cb[i * 8 ..][0..4], .little);
        const nl = std.mem.readInt(u32, cb[i * 8 + 4 ..][0..4], .little);
        try testing.expect(np != callee_ptrs[i]);
        try testing.expectEqual(@as(u32, @intCast(m.len)), nl);
        try testing.expectEqualStrings(m, caller.readGuestBytes(np, nl).?);
    }
}
