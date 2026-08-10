//! Component instance — runtime state for an instantiated component.
//!
//! Manages resource tables, canonical function wrappers, and links
//! between component instances and their underlying core module instances.

const std = @import("std");
const ctypes = @import("types.zig");
const core_types = @import("../runtime/common/types.zig");
const executor_mod = @import("executor.zig");
const indexspace = @import("indexspace.zig");
const async_mod = @import("async.zig");
const core_backend = @import("core_backend.zig");
const aot_loader = @import("../runtime/aot/loader.zig");
const aot_runtime = @import("../runtime/aot/runtime.zig");
const host_trampolines = @import("../runtime/aot/host_trampolines.zig");
const call_frame_mod = @import("call_frame.zig");
const CoreFuncIdxLocal = call_frame_mod.CoreFuncIdxLocal;

const aot_host_bridge = @import("../runtime/aot/host_bridge.zig");

pub const Options = core_backend.Options;
pub const PrecompiledCore = core_backend.PrecompiledCore;
pub const CoreInstanceBackend = core_backend.CoreInstanceBackend;

// ── Resource Table ──────────────────────────────────────────────────────────

/// A resource table maps integer handles to host-side representations.
/// Each component instance has its own resource table per resource type.
pub const ResourceTable = struct {
    /// Slot in the resource table.
    const Slot = struct {
        /// The host-side representation value.
        rep: u32,
        /// Whether this handle is currently valid.
        active: bool = true,
        /// Borrow depth — number of outstanding borrows of this handle.
        borrow_count: u32 = 0,
        /// Whether this is an owned handle (vs borrowed).
        owned: bool = true,
    };

    slots: std.ArrayListUnmanaged(Slot) = .empty,
    /// Free list of slot indices for reuse.
    free_list: std.ArrayListUnmanaged(u32) = .empty,

    /// Allocate a new handle for a representation. Returns the handle index.
    pub fn new(self: *ResourceTable, representation: u32, owned: bool, allocator: std.mem.Allocator) !u32 {
        if (self.free_list.items.len > 0) {
            const idx = self.free_list.items[self.free_list.items.len - 1];
            self.free_list.items.len -= 1;
            self.slots.items[idx] = .{ .rep = representation, .owned = owned };
            return idx;
        }
        const idx: u32 = @intCast(self.slots.items.len);
        try self.slots.append(allocator, .{ .rep = representation, .owned = owned });
        return idx;
    }

    /// Get the representation for a handle. Returns null if invalid.
    pub fn rep(self: *const ResourceTable, handle: u32) ?u32 {
        if (handle >= self.slots.items.len) return null;
        const slot = self.slots.items[handle];
        if (!slot.active) return null;
        return slot.rep;
    }

    /// Drop a handle, marking it inactive. Returns the rep for destructor call.
    /// Returns null if the handle was already dropped or is a borrow with outstanding refs.
    pub fn drop(self: *ResourceTable, handle: u32, allocator: std.mem.Allocator) ?u32 {
        if (handle >= self.slots.items.len) return null;
        const slot = &self.slots.items[handle];
        if (!slot.active) return null;
        if (slot.borrow_count > 0) return null; // can't drop with outstanding borrows
        const r = slot.rep;
        slot.active = false;
        // Add to free list for reuse
        self.free_list.append(allocator, handle) catch {};
        return r;
    }

    /// Increment borrow count for a handle.
    pub fn borrow(self: *ResourceTable, handle: u32) bool {
        if (handle >= self.slots.items.len) return false;
        const slot = &self.slots.items[handle];
        if (!slot.active) return false;
        slot.borrow_count += 1;
        return true;
    }

    /// Decrement borrow count for a handle.
    pub fn returnBorrow(self: *ResourceTable, handle: u32) void {
        if (handle >= self.slots.items.len) return;
        const slot = &self.slots.items[handle];
        if (slot.borrow_count > 0) slot.borrow_count -= 1;
    }

    pub fn deinit(self: *ResourceTable, allocator: std.mem.Allocator) void {
        self.slots.deinit(allocator);
        self.free_list.deinit(allocator);
    }
};

// ── Component Instance ──────────────────────────────────────────────────────

/// Host-facing type of an interface value passed across a component boundary.
pub const InterfaceValue = @import("canonical_abi.zig").InterfaceValue;

/// A host-provided function that satisfies a component `func` import.
///
/// Phase 2A will invoke this from a canon-lowered core trampoline, after
/// the trampoline has decoded the core ABI representation into component-level
/// `InterfaceValue`s. For Phase 1B the field is captured but never called.
///
/// Memory ownership for compound result values (strings, lists, records)
/// follows the standard canonical-ABI convention: the callee allocates
/// into `allocator`; the trampoline that invoked the host func owns the
/// resulting values and is responsible for lifting them back into core memory.
pub const HostFunc = struct {
    /// Opaque context pointer forwarded to `call`.
    context: ?*anyopaque = null,
    /// Host-side implementation. Null is legal in tests where the call path
    /// is not exercised; `linkImports` does not require it to be set.
    call: ?*const fn (
        ctx: ?*anyopaque,
        ci: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: std.mem.Allocator,
    ) anyerror!void = null,
    /// Optional component-level function type index, for Phase 2B validation.
    type_idx: ?u32 = null,
};

/// A member of a host-provided instance binding.
pub const HostInstanceMember = union(enum) {
    func: HostFunc,
    /// Placeholder for host-side resource-type identity. For Phase 1B we
    /// carry only the raw component resource-type index the host claims to
    /// implement; real opaque identity / lift/lower glue lands in Phase 2A.
    resource_type: u32,
};

/// A host binding for an instance-typed component import. The members map
/// names (e.g. `"[method]output-stream.write"`) to host implementations.
///
/// `ComponentInstance` stores these by borrowed pointer — callers must keep
/// the `HostInstance` alive for at least the lifetime of every
/// `ComponentInstance` whose imports point at it.
pub const HostInstance = struct {
    members: std.StringHashMapUnmanaged(HostInstanceMember) = .empty,

    pub fn deinit(self: *HostInstance, allocator: std.mem.Allocator) void {
        self.members.deinit(allocator);
    }
};

/// Binding for a component import — either a host-provided callback,
/// a host-provided instance (member map), or a reference to another
/// component instance's export.
pub const ImportBinding = union(enum) {
    /// A host-provided function.
    host_func: HostFunc,
    /// A host-provided instance (WASI p2 top-level imports are always
    /// instance-typed; this is the common Phase 2B path).
    host_instance: *const HostInstance,
    /// A reference to another ComponentInstance's exported function.
    component_export: struct {
        instance: *const ComponentInstance,
        func_name: []const u8,
    },
};

/// Backing buffer used in unit tests that don't have a real core module
/// instance with a `cabi_realloc` export. Tests opt-in via
/// `ComponentInstance.enableTestMem`; runtime code never sets this.
///
/// Provides a bump-allocated `[]u8` that `hostAllocGuest` /
/// `hostAllocAndWrite` write into and `readGuestBytes` reads from, so
/// adapter host functions can lower lists / strings into "guest"
/// memory under test without needing a live realloc.
pub const TestGuestMem = struct {
    buffer: []u8,
    bump: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, size: usize) !TestGuestMem {
        const buf = try allocator.alloc(u8, size);
        @memset(buf, 0);
        return .{ .buffer = buf };
    }

    pub fn deinit(self: *TestGuestMem, allocator: std.mem.Allocator) void {
        allocator.free(self.buffer);
    }

    /// Bump-allocate `size` bytes aligned to `align_`. Returns the
    /// guest-side offset, or null if the buffer is exhausted.
    pub fn alloc(self: *TestGuestMem, size: u32, align_: u32) ?u32 {
        const a = if (align_ == 0) 1 else align_;
        const start = std.mem.alignForward(u32, self.bump, a);
        const end = std.math.add(u32, start, size) catch return null;
        if (end > self.buffer.len) return null;
        self.bump = end;
        return start;
    }
};

/// A runtime component instance — the result of instantiating a Component.
pub const ComponentInstance = struct {
    /// The parsed component this instance was created from.
    component: *const ctypes.Component,
    /// Core module instances created during instantiation.
    core_instances: []CoreInstanceEntry,
    /// Test-only guest-memory shim. When non-null, `hostAllocGuest`,
    /// `hostAllocAndWrite`, and `readGuestBytes` operate on this buffer
    /// instead of looking for a real `cabi_realloc` / canonical memory.
    /// Always null in production paths.
    test_mem: ?*TestGuestMem = null,
    /// Resource tables, keyed by the raw component resource type index
    /// (as referenced by `canon resource.{new,drop,rep}`). Allocated
    /// lazily on first access so the dense `[]ResourceTable` layout (which
    /// silently assumed resource indices were dense over locally declared
    /// resources) no longer corrupts on aliased or imported resources.
    resource_tables: std.AutoHashMapUnmanaged(u32, ResourceTable),
    /// Exported functions (component-level func index → core func index + instance).
    exported_funcs: std.StringHashMapUnmanaged(ExportedFunc),
    /// Resolved imports keyed by import name.
    imports: std.StringHashMapUnmanaged(ImportBinding),
    /// Arena used for core-module loader allocations. Core wasm binaries
    /// parsed from this component have their types/imports/exports/etc.
    /// arrays allocated into this arena, which is destroyed as one unit on
    /// `deinit`. Mirrors the pattern `api/wamr.zig` uses for top-level
    /// module loads.
    module_arena: std.heap.ArenaAllocator,
    /// Canon-lower trampoline contexts owned by this instance. Each entry
    /// is referenced as the `ctx` of an installed `HostFnEntry` on some core
    /// module instance; we keep the slice here so lifetimes are tied to the
    /// `ComponentInstance` and freed together on `deinit`.
    trampoline_ctxs: std.ArrayListUnmanaged(*executor_mod.ComponentTrampolineCtx) = .empty,
    /// Canon-builtin trampoline contexts (context.{get,set}, task.{yield,
    /// return}, resource.{new,drop,rep}, async ABI). Same ownership model
    /// as `trampoline_ctxs` — each is referenced from an installed
    /// `HostFnEntry` and freed when the instance tears down. (#520)
    canon_builtin_ctxs: std.ArrayListUnmanaged(*executor_mod.CanonBuiltinTrampolineCtx) = .empty,
    /// Memoisation map for canon-builtin trampoline contexts, keyed by the
    /// component canon-def index that produced the context. Lets multiple
    /// core import slots that resolve to the *same* canon-def-id (e.g. two
    /// imports of `context.get` from different interfaces) share a single
    /// `*CanonBuiltinTrampolineCtx` rather than allocating one per slot.
    /// The pointers in this map are non-owning aliases of entries already
    /// in `canon_builtin_ctxs`; that list remains the sole owner and is
    /// what `deinit` walks to free the contexts exactly once. (#533)
    canon_builtin_ctx_by_canon_idx: std.AutoHashMapUnmanaged(u32, *executor_mod.CanonBuiltinTrampolineCtx) = .empty,
    /// Per-(component-instance) host-trampoline pool. Lazily created the
    /// first time an AOT core needs a cross-instance fn-import thunk or a
    /// trap-on-call stub (#662 Phase C). Slots own the executable shim
    /// memory; the `cross_instance_thunk_ctxs` ArrayList owns the per-slot
    /// `CrossInstanceThunkCtx` payloads they reference.
    aot_trampoline_pool: ?*host_trampolines.TrampolinePool = null,
    /// Per-(component-instance) ownership of cross-instance thunk
    /// contexts (`*executor_mod.CrossInstanceThunkCtx`). Each entry is
    /// referenced as the `ctx` of an `aot_trampoline_pool` slot installed
    /// in a sibling AOT instance's `host_functions[]` table; freed
    /// together on `deinit` (#662 Phase C).
    cross_instance_thunk_ctxs: std.ArrayListUnmanaged(*executor_mod.CrossInstanceThunkCtx) = .empty,
    /// Per-(component-instance) ownership of trap-stub label C strings
    /// (`*const [*:0]const u8` payloads). Each entry is referenced from a
    /// `aot_trampoline_pool` trap slot's `ctx` so a debug-mode log line
    /// can name the un-bridged import (#662 follow-up).
    trap_stub_labels: std.ArrayListUnmanaged([*:0]const u8) = .empty,
    /// Pending core-module start functions whose execution was deferred
    /// during `instantiate` so canon-lower trampoline `host_funcs` can be
    /// bound by `linkImports` first. Drained by `linkImports` in core-instance
    /// order; see `runDeferredCoreStarts` (issue #308).
    pending_core_starts: std.ArrayListUnmanaged(*core_types.ModuleInstance) = .empty,
    /// Pending AOT-instantiated core-module start functions whose execution
    /// was deferred during `instantiate` so cross-instance imports (resolved
    /// against sibling cores) are wired before the start runs. Drained by
    /// `linkImports` after `pending_core_starts`. Mirrors #308 for the AOT
    /// path. Without this, AOT modules with start sections — e.g. the
    /// wasi-libc adapter's `path_unlink_file` (`State::new` cabi_realloc) —
    /// never initialize `__stack_pointer` / `allocation_state`, and any
    /// subsequent guest call into the adapter traps on State magic mismatch.
    pending_aot_starts: std.ArrayListUnmanaged(struct {
        inst: *aot_runtime.AotInstance,
        start_idx: u32,
    }) = .empty,
    /// Whether the start function has been executed.
    started: bool = false,
    /// Caller-supplied instantiation options (e.g. precompiled-core
    /// artifacts). Captured at instantiate-time and consulted by the
    /// section-aware loop when it reaches each `.instantiate` expr
    /// (#625). Default `.{}` matches the pre-#625 behaviour: every
    /// core is loaded via `runtime/interpreter/loader.zig`.
    options: Options = .{},
    /// Allocator for instance lifetime.
    allocator: std.mem.Allocator,
    /// Child component instances created when the parent component
    /// declares `(instance N (instantiate <subcomp> ...))` expressions.
    /// Indexed by the *local* component-instance idx (i.e. by index into
    /// `component.instances[]`), NOT by the full component-instance
    /// index space (which mixes imports / locals / aliases). A `null`
    /// slot means either:
    ///   - the matching `instances[i]` is an `.exports` inline bundle
    ///     (no runtime sub-instance is needed), or
    ///   - the matching `instances[i]` is an `.instantiate` of a
    ///     wit-bindgen "imported-func re-export" shim, which the
    ///     existing `registerInstanceExport.instantiate` path handles
    ///     without a real runtime instance.
    /// Children are allocated with `self.allocator` (NOT
    /// `module_arena.allocator()`) so each retains a deterministic
    /// `deinit()` path. `ComponentInstance.deinit` walks this slice
    /// before tearing down its own state. (Issue #355.)
    sub_instances: []?*ComponentInstance = &.{},
    /// Forwarding-`HostFunc` contexts owned by THIS instance — i.e. the
    /// instance whose `imports` map borrows them. Created during
    /// `linkImports` when wiring sub-component imports back to a parent
    /// (or peer) `ExportedFunc.Local`. The borrower-owns invariant
    /// keeps lifetimes tight: when this instance is deinitialized, its
    /// import map vanishes along with the contexts it referenced.
    /// (Issue #355.)
    forwarding_ctxs: std.ArrayListUnmanaged(*executor_mod.ForwardingHostFnCtx) = .empty,
    /// Synthetic `HostInstance`s constructed during `linkImports` to
    /// satisfy `.instance`-typed sub-component imports whose source is
    /// a parent / peer component (i.e. whose members must be wrapped
    /// as forwarding host funcs). Owned by the instance whose `imports`
    /// references them, freed in `deinit`. (Issue #355.)
    synthetic_host_instances: std.ArrayListUnmanaged(*HostInstance) = .empty,
    /// Whether `linkImports` has fully completed against this instance.
    /// Surfacing a partial link (e.g. a child link that failed mid-way)
    /// as `.poisoned` lets `deinit` and external callers reject reuse
    /// without depending on hashmap occupancy. (Issue #355.)
    link_state: LinkState = .unlinked,
    /// Per-instance context slots for `canon context.{get,set}` invoked
    /// outside any async task (the synchronous canon-lift call path).
    /// Mirrors Wasmtime's implicit-task fallback: when no async task is
    /// on the dispatch stack, context.get/set still works, scoped to
    /// the instance rather than to any caller task. Initialised to all
    /// zeros — the spec doesn't define a non-zero initial value.
    /// (#478 sub-PR 1.)
    implicit_task_context: [async_mod.N_CONTEXT_SLOTS]u32 = [_]u32{0} ** async_mod.N_CONTEXT_SLOTS,

    /// Per-instance async-handle tables for the WASIp3 canonical-ABI
    /// surface (#478 sub-PR 3). Each table maps a u32 handle to the
    /// host-side state allocated by the corresponding `.new` canon op:
    /// futures, streams, error-contexts, and waitable-sets. Handles are
    /// drawn from `next_async_handle` (starting at 1; zero is the spec
    /// sentinel meaning "no value yet").
    futures: std.AutoHashMapUnmanaged(u32, async_mod.Future) = .empty,
    streams: std.AutoHashMapUnmanaged(u32, async_mod.AsyncStream) = .empty,
    error_contexts: std.AutoHashMapUnmanaged(u32, []u8) = .empty,
    waitable_sets: std.AutoHashMapUnmanaged(u32, async_mod.WaitableSet) = .empty,
    next_async_handle: u32 = 1,

    /// TaskManager driving the currently-active async-lifted call into
    /// this instance, or null when no async call is on the stack. Set
    /// by `callComponentFuncAsync` for the duration of an async-lifted
    /// dispatch, and restored to its prior value on return so nested
    /// dispatches work. Used by the canon-builtin host trampoline (see
    /// `executor.canonBuiltinTrampoline`) to route `context.{get,set}`,
    /// `task.{yield,return}`, and the async ABI canons through
    /// `dispatchCanonBuiltin` with the right task state. When null,
    /// canon builtins fall back to the instance-scoped
    /// `implicit_task_context` (Wasmtime parity, sync-call path). (#520)
    current_task_manager: ?*async_mod.TaskManager = null,

    /// Host-driven async-event driver hook (#551). Installed by
    /// `wasi:cli`-style host adapters that own background timers or other
    /// futures whose completion is not produced by guest code. Invoked
    /// from `waitable-set.{wait,poll}` to advance time / drain host I/O
    /// before consulting `WaitableSet.ready_queue`. Returns `true` if any
    /// host-side event fired (i.e. some waitable in `futures`/`streams`
    /// transitioned to `.ready`).
    ///
    /// `wait_for_ns_hint` is the number of nanoseconds the executor is
    /// willing to block waiting for a host event before returning. The
    /// adapter may sleep up to that amount or, when servicing a short
    /// queue of due timers, return immediately. A `null` hint means the
    /// caller is polling (non-blocking): the driver MUST NOT sleep.
    /// (See `wasi_cli_adapter.WasiCliAdapter.driveAsyncEvents`.)
    async_event_driver: ?*const fn (
        ctx: ?*anyopaque,
        ci: *ComponentInstance,
        wait_for_ns_hint: ?u64,
        allocator: std.mem.Allocator,
    ) bool = null,
    async_event_driver_ctx: ?*anyopaque = null,
    /// Companion hook to `async_event_driver` invoked from the
    /// `task.cancel` canon-builtin path (#551). Lets the host abort any
    /// timer futures owned by the currently-cancelled task — the wait
    /// must settle with the cancel disposition so the guest sees
    /// `STATUS_STARTED_CANCELLED` on its next `waitable-set.wait`. The
    /// driver iterates its pending-timer table and, for every entry that
    /// belongs to `task_handle` (or all entries when `task_handle` is
    /// null), removes the timer and transitions the backing future to
    /// `.closed` with `write_closed = true` so the executor lowers
    /// `STATUS_STARTED_CANCELLED` for the corresponding subtask.
    async_cancel_driver: ?*const fn (
        ctx: ?*anyopaque,
        ci: *ComponentInstance,
        task_handle: ?u32,
        allocator: std.mem.Allocator,
    ) void = null,

    /// Companion hook to `async_cancel_driver` invoked when the guest
    /// drops the readable end of a future (#616 A7).
    ///
    /// Dropping the readable end means the guest has abandoned the
    /// result: nothing will ever observe it. For a host operation
    /// still in flight — an outbound HTTP fetch, say — that is a
    /// cancellation request, and the only signal the host gets, since
    /// no `task.cancel` is issued on this path. The driver looks the
    /// handle up in its pending-operation tables and, on a match,
    /// raises that operation's cancel flag so the worker unblocks
    /// instead of running the request to completion and discarding
    /// the answer.
    ///
    /// Called with the handle still present in `ci.futures`, before
    /// any drop bookkeeping, so the driver can correlate on it.
    async_future_drop_driver: ?*const fn (
        ctx: ?*anyopaque,
        ci: *ComponentInstance,
        future_handle: u32,
    ) void = null,

    /// Optional host hook invoked from the `canon resource.drop`
    /// canon-builtin (`dispatchCanonBuiltin.resource_drop`) after the
    /// per-type resource table entry is removed. Host adapters that
    /// keep kernel-side state for imported resources (e.g. POSIX fds
    /// behind a `wasi:sockets/types.tcp-socket` handle) install this
    /// hook so the kernel fd is released as soon as the guest drops
    /// the wit-resource — matching wit-bindgen's expectation that
    /// `[resource-drop]X` runs synchronously on guest drop. Without
    /// the hook, host fds linger until the adapter's `deinit`, which
    /// breaks tests like `sockets-tcp-bind::test_reuseaddr` that
    /// rebind to the freshly-released ephemeral port. The hook
    /// receives the same `(resource_idx, handle)` pair the canon
    /// builtin was invoked with so the adapter can route to the
    /// correct per-type cleanup. (#575)
    on_resource_drop: ?*const fn (
        ctx: ?*anyopaque,
        ci: *ComponentInstance,
        resource_idx: u32,
        handle: u32,
    ) void = null,
    on_resource_drop_ctx: ?*anyopaque = null,

    /// Cached `ExecEnv` for `cabi_realloc` (#538). The wasi:http@0.3.0
    /// fixtures allocate guest-side scratch buffers many thousand
    /// times per `wamr run`; recreating a 96 KiB `ExecEnv` on every
    /// `hostAllocGuest` pushed Debug-build `http-fields` past the
    /// wasi-p3-testsuite runner's 5-second wait timeout. Lazy-created
    /// on first use and freed in `deinit`. `realloc_env_owner` keeps
    /// the cache aligned with the core module that exposes the
    /// `cabi_realloc` export — re-creating if the realloc owner ever
    /// shifts (e.g. when sub-instances become the realloc target).
    realloc_env: ?*@import("../runtime/common/exec_env.zig").ExecEnv = null,
    realloc_env_owner: ?*core_types.ModuleInstance = null,

    /// Canon-lower opts of the in-flight host import call, set by
    /// `componentTrampoline` / `dispatchAotComponentTrampoline` before
    /// dispatching into the host method and cleared after. Lets
    /// `hostAllocAndWrite` honor the lowerer's chosen `(realloc $f)` +
    /// `(memory $m)` instead of falling back to the canonical
    /// `cabi_realloc` search.
    ///
    /// Critical for wit-component's `wasi_snapshot_preview1` adapter:
    /// the adapter imports WASIp2 via `canon.lower (realloc
    /// $cabi_import_realloc)`, where `cabi_import_realloc` routes
    /// through a per-call `import_alloc` cell that pins string
    /// allocations to the adapter's own `temporary_data` buffer.
    /// `fd_readdir` (in the adapter) asserts the returned `name` ptr
    /// equals `temporary_data` — calling the main module's
    /// `cabi_realloc` instead returns a ptr in the wrong memory and
    /// trips that assertion. (#715.)
    current_lower_call_ctx: ?CanonLowerCallCtx = null,

    /// Resolved canon-lower opts for an in-flight host import call.
    /// `memory` and `realloc` are pre-resolved from the canon-lower
    /// opts' top-level indices so `hostAllocAndWrite` does not redo
    /// the indexspace lookups per byte. Either field may be null when
    /// the canon-lower decl omitted the corresponding opt.
    pub const CanonLowerCallCtx = struct {
        memory: ?*core_types.MemoryInstance = null,
        realloc: ?ReallocTarget = null,
    };

    /// A directly-callable realloc, resolved from a top-level core-func
    /// index in canon-lower opts. Backend-tagged because realloc can
    /// live on either a fully-AOT-precompiled core or an interp core.
    pub const ReallocTarget = union(enum) {
        interp: struct {
            mi: *core_types.ModuleInstance,
            local_idx: CoreFuncIdxLocal,
        },
        aot: struct {
            ai: *aot_runtime.AotInstance,
            local_idx: CoreFuncIdxLocal,
        },
    };

    /// Allocate a fresh async-handle (#478 sub-PR 3). Used by every
    /// `.new`-flavoured canon-builtin to mint a unique key into the
    /// per-instance future / stream / error-context / waitable-set
    /// tables.
    pub fn allocAsyncHandle(self: *ComponentInstance) u32 {
        const h = self.next_async_handle;
        self.next_async_handle += 1;
        return h;
    }

    pub const LinkState = enum { unlinked, linking, linked, poisoned };

    pub const CoreInstanceEntry = struct {
        module_inst: ?*core_types.ModuleInstance = null,
        /// AOT-backed runnable form of this core instance. Mutually
        /// exclusive with `module_inst` — set only when `instantiate`
        /// found a `PrecompiledCore` artifact for this slot's
        /// `module_idx` in `Options.precompiled_cores` (#625).
        ///
        /// Phase 1: AOT instances are loaded + own memory/tables/
        /// globals (visible to cross-instance import wiring via the
        /// shared `MemoryInstance`/`TableInstance`/`GlobalInstance`
        /// types), but the canon-ABI lift call path still requires
        /// `module_inst`; lift against an AOT-only core errors out
        /// until phase 3 wires `aot_runtime.callFunc` into
        /// `executor.callComponentFuncByLocal`.
        aot_inst: ?*aot_runtime.AotInstance = null,
        /// Optional lazy-JIT driver attached to `aot_inst` for
        /// in-memory component precompile results (#889). Destroy only
        /// after `aot_inst` itself is torn down because the runtime may
        /// call back into the driver up until `AotInstance.destroy()`.
        lazy_driver: ?core_backend.LazyJitHandle = null,
        /// When this entry corresponds to a `CoreInstanceExpr.exports` (an
        /// inline instance bundling named core items rather than an actual
        /// core-module instantiation), the named items live here. `module_inst`
        /// is null in that case.
        inline_exports: []const ctypes.CoreInlineExport = &.{},

        /// Phase-1 helper: return whichever backend is populated. Null
        /// for inline-exports-only entries.
        pub fn backend(self: CoreInstanceEntry) ?CoreInstanceBackend {
            if (self.module_inst) |mi| return .{ .interp = mi };
            if (self.aot_inst) |ai| return .{ .aot = ai };
            return null;
        }
    };

    /// An exported component function. Two flavours:
    ///   * `.local` — backed by a `canon.lift` inside this very
    ///     component, executable directly against `core_instances[…]`.
    ///   * `.forwarded` — a re-publication of another
    ///     `ComponentInstance`'s export, used when wamr instantiates a
    ///     sub-component (e.g. inside a `wasm-tools compose`d wrapper)
    ///     and then exposes the child's `wasi:cli/run` instance through
    ///     an `(alias export …)` chain. Indices in the `Local` variant
    ///     are owner-relative; copying them into a parent would
    ///     mis-index into the parent's `core_instances`. Forwarding by
    ///     `(owner, name)` keeps execution dispatching against the
    ///     correct component (issue #355).
    pub const ExportedFunc = union(enum) {
        local: Local,
        forwarded: Forwarded,

        pub const Local = struct {
            core_instance_idx: u32,
            core_func_idx: u32,
            /// Component-level function type index (into component.types).
            func_type_idx: u32 = 0,
            /// Canonical options from the canon lift definition.
            opts: []const ctypes.CanonOpt = &.{},
        };

        pub const Forwarded = struct {
            owner: *ComponentInstance,
            /// Key into `owner.exported_funcs`. Forwarded entries are
            /// flattened during registration so this always points at
            /// a `.local` (or another `.forwarded` whose owner chain
            /// also ultimately bottoms out at `.local`).
            owner_export_name: []const u8,
        };
    };

    /// Look up an exported function by name.
    pub fn getExport(self: *const ComponentInstance, name: []const u8) ?ExportedFunc {
        return self.exported_funcs.get(name);
    }

    /// Look up a resolved import by name.
    pub fn getImport(self: *const ComponentInstance, name: []const u8) ?ImportBinding {
        return self.imports.get(name);
    }

    /// Get-or-create the resource table for a given component resource type
    /// index. Resource tables are lazily allocated on first use.
    pub fn getOrCreateResourceTable(self: *ComponentInstance, type_idx: u32) !*ResourceTable {
        const gop = try self.resource_tables.getOrPut(self.allocator, type_idx);
        if (!gop.found_existing) gop.value_ptr.* = .{};
        return gop.value_ptr;
    }

    /// Find the first core instance entry with a real `ModuleInstance`.
    /// The component's "canonical" memory always lives on a real
    /// instance — the inline-exports entries used to wire imports never
    /// own a module. This is the lookup both `componentTrampoline` and
    /// `readGuestBytes` use to resolve guest memory.
    pub fn firstModuleInst(self: *const ComponentInstance) ?*core_types.ModuleInstance {
        for (self.core_instances) |entry| {
            if (entry.module_inst) |mi| return mi;
        }
        return null;
    }

    /// Backend-agnostic variant of `firstModuleInst` for callers that
    /// only need a `*MemoryInstance` and don't care whether the core
    /// runs on the interpreter or the AOT runtime. Searches each
    /// `core_instances[]` entry in order and returns memory index 0 on
    /// the first backend that exposes one (#625).
    ///
    /// Phase 1: callers that genuinely need a `*ModuleInstance` (e.g.
    /// to drive interp-only execution paths) keep using
    /// `firstModuleInst`; only the read-only memory probe paths
    /// (`readGuestBytes`, `componentTrampoline` memory lookup) should
    /// migrate to this helper.
    pub fn firstBackendMemory(self: *const ComponentInstance) ?*core_types.MemoryInstance {
        for (self.core_instances) |entry| {
            const be = entry.backend() orelse continue;
            if (be.memory(0)) |m| return m;
        }
        return null;
    }

    /// Resolve a top-level core memory indexspace index to the underlying
    /// `*MemoryInstance`. Used by `componentTrampoline` and host-side
    /// helpers to find the memory referenced by a `(memory N)` canonical
    /// option, where `N` is the component-level core memory index — which
    /// may be contributed by an `alias core export` decl pointing at a
    /// memory exported by a different core instance than the "first" one.
    ///
    /// Resolution order:
    ///   1. If `N` is contributed by an `alias core export`, follow it
    ///      through the source core instance to the underlying memory.
    ///   2. Otherwise, fall back to `firstModuleInst().getMemory(N)` —
    ///      preserves behavior for hand-authored fixtures with a single
    ///      core module (where the local memory idx matches N).
    pub fn resolveTopLevelMemory(self: *const ComponentInstance, idx: u32) ?*core_types.MemoryInstance {
        const ref = indexspace.resolveCoreMemory(self.component, idx) orelse {
            if (self.firstModuleInst()) |mi| return mi.getMemory(idx);
            // All-AOT components have no interp `module_inst` on any core
            // instance; fall through to the backend-agnostic probe so
            // canon-lower(aot) retptr stores and `canonicalMemory()`
            // succeed against the AOT core's memory. Issue #707.
            return self.firstBackendMemory();
        };
        const ie = self.component.aliases[ref.aliased].instance_export;
        if (ie.instance_idx >= self.core_instances.len) return null;
        const src_entry = self.core_instances[ie.instance_idx];
        if (src_entry.module_inst) |src_mi| {
            const exp = src_mi.module.findExport(ie.name, .memory) orelse return null;
            if (exp.index >= src_mi.memories.len) return null;
            return src_mi.memories[exp.index];
        }
        // AOT source — go via backend-agnostic memory probe. The AOT
        // runtime synthesises a single canonical memory; we look up
        // by name through `findExportMemory`, falling back to memory
        // 0 when the AOT export table doesn't carry memory entries
        // by name. Issue #707.
        if (src_entry.aot_inst) |ai| {
            if (aot_runtime.findExportMemory(ai, ie.name)) |mi| return mi;
            if (src_entry.backend()) |be| return be.memory(0);
        }
        return null;
    }

    /// Resolve a top-level core func indexspace index to a callable
    /// `(*ModuleInstance, local_func_idx)` pair, suitable for invoking
    /// from a host context (e.g. calling `cabi_realloc`). Only the
    /// alias-core-export contributors yield a directly-callable function;
    /// canon.lower / resource.* canons are imports and return null.
    pub fn resolveTopLevelCoreFunc(
        self: *const ComponentInstance,
        idx: u32,
    ) ?struct { mi: *core_types.ModuleInstance, local_idx: u32 } {
        const ref = indexspace.resolveCoreFunc(self.component, idx) orelse return null;
        switch (ref) {
            .lowered,
            .resource_drop,
            .resource_new,
            .resource_rep,
            .task_yield,
            .context_get,
            .context_set,
            .task_return,
            .async_canon,
            => return null,
            .aliased => |alias_idx| {
                const ie = self.component.aliases[alias_idx].instance_export;
                if (ie.instance_idx >= self.core_instances.len) return null;
                const mi = self.core_instances[ie.instance_idx].module_inst orelse return null;
                const local = mi.getExportFunc(ie.name) orelse return null;
                return .{ .mi = mi, .local_idx = local };
            },
        }
    }

    /// AOT-aware sibling of `resolveTopLevelCoreFunc`. Returns either an
    /// AOT or interp `ReallocTarget` for the named export of the
    /// aliased core instance. Used by the canon-lower trampoline to
    /// pre-resolve `(realloc $f)` before stashing the call ctx on
    /// `current_lower_call_ctx`. (#715.)
    pub fn resolveTopLevelCoreFuncAny(
        self: *const ComponentInstance,
        idx: u32,
    ) ?ReallocTarget {
        const ref = indexspace.resolveCoreFunc(self.component, idx) orelse return null;
        switch (ref) {
            .lowered,
            .resource_drop,
            .resource_new,
            .resource_rep,
            .task_yield,
            .context_get,
            .context_set,
            .task_return,
            .async_canon,
            => return null,
            .aliased => |alias_idx| {
                const ie = self.component.aliases[alias_idx].instance_export;
                if (ie.instance_idx >= self.core_instances.len) return null;
                const entry = self.core_instances[ie.instance_idx];
                if (entry.module_inst) |mi| {
                    const local = mi.getExportFunc(ie.name) orelse return null;
                    return .{ .interp = .{ .mi = mi, .local_idx = CoreFuncIdxLocal.from(local) } };
                }
                if (entry.aot_inst) |ai| {
                    const local = aot_runtime.findExportFunc(ai, ie.name) orelse return null;
                    return .{ .aot = .{ .ai = ai, .local_idx = CoreFuncIdxLocal.from(local) } };
                }
                return null;
            },
        }
    }

    /// Read `len` bytes starting at guest linear-memory offset `ptr` from the
    /// canonical memory of this component. Used by host adapter callbacks
    /// invoked from `componentTrampoline` to materialize `list<u8>` /
    /// `string` arguments whose flat representation is `(ptr, len)` into
    /// guest memory.
    ///
    /// Phase 2B narrow assumption: the canonical memory lives on the
    /// first core instance with a real `module_inst` at memory index 0.
    /// Returns null if no such instance is present, the memory is missing,
    /// or the slice is out of bounds.
    pub fn readGuestBytes(self: *const ComponentInstance, ptr: u32, len: u32) ?[]const u8 {
        if (self.test_mem) |tm| {
            const end = @as(usize, ptr) + @as(usize, len);
            if (end > tm.buffer.len) return null;
            return tm.buffer[ptr..end];
        }
        // Mirror `writableGuestBytes`: when the in-flight canon-lower
        // pinned a specific `(memory $m)`, read from THAT memory so
        // string/list args lifted from the lowerer (e.g. the
        // wit-component preview1 adapter's per-page memory) resolve
        // correctly. Falls back to the canonical memory otherwise.
        // (#715.)
        if (self.current_lower_call_ctx) |cctx| {
            if (cctx.memory) |mem| {
                const end = @as(usize, ptr) + @as(usize, len);
                if (end > mem.byteLen()) return null;
                return mem.data[ptr..end];
            }
        }
        const mem = self.canonicalMemory() orelse return null;
        const end = @as(usize, ptr) + @as(usize, len);
        if (end > mem.byteLen()) return null;
        return mem.data[ptr..end];
    }

    /// Return the "canonical" guest memory: the one that
    /// `cabi_realloc` allocates into and that lift/lower of compound
    /// types reads/writes through. Prefers a top-level core memory 0
    /// (which under wit-component output is `alias core export $main
    /// "memory"`); falls back to the first module instance's memory[0]
    /// for legacy hand-authored fixtures.
    pub fn canonicalMemory(self: *const ComponentInstance) ?*core_types.MemoryInstance {
        if (self.resolveTopLevelMemory(0)) |m| return m;
        const mi = self.firstModuleInst() orelse return null;
        return mi.getMemory(0);
    }

    /// Locate the module instance that owns `cabi_realloc`. wit-component
    /// emits `cabi_realloc` on the same core module that owns the
    /// canonical memory (`$main` in stdio-echo); legacy fixtures put it
    /// on the only module instance.
    fn reallocOwner(self: *const ComponentInstance) ?*core_types.ModuleInstance {
        // First try: walk all core instances looking for one that exports
        // `cabi_realloc`. This is robust regardless of which instance the
        // canonical memory aliases through.
        for (self.core_instances) |entry| {
            const mi = entry.module_inst orelse continue;
            if (mi.getExportFunc("cabi_realloc") != null) return mi;
        }
        return null;
    }

    /// AOT analogue of `reallocOwner`. Walks `core_instances` looking
    /// for an `aot_inst` that exports `cabi_realloc`. Required because
    /// fully-precompiled components have no interp `module_inst` on any
    /// `core_instances` entry, and `hostAllocGuest`'s interp fallback
    /// otherwise returns null — surfacing as `error.IoError` from
    /// `getEnvironment` / `read-via-stream` / any other adapter that
    /// must materialise a host-built `list` / `string` into guest
    /// memory before lowering. Pairs with the canon-lower-aot
    /// dispatcher widening in this PR (issue #707).
    fn reallocOwnerAot(self: *const ComponentInstance) ?*aot_runtime.AotInstance {
        for (self.core_instances) |entry| {
            const ai = entry.aot_inst orelse continue;
            if (aot_runtime.findExportFunc(ai, "cabi_realloc") != null) return ai;
        }
        return null;
    }

    /// Allocate `size` bytes inside the canonical guest linear memory
    /// (or the test_mem shim, if installed) aligned to `align_`. Returns
    /// the guest-side pointer, or null on failure.
    ///
    /// Used by host-side adapter callbacks that must materialize
    /// host-constructed lists / strings into guest memory before the
    /// canonical ABI sees a `(ptr, len)` PtrLen.
    pub fn hostAllocGuest(self: *ComponentInstance, size: u32, align_: u32) ?u32 {
        if (self.test_mem) |tm| return tm.alloc(size, align_);
        const a: u32 = if (align_ == 0) 1 else align_;
        // Honor the in-flight canon-lower's `(realloc $f)` when set.
        // This is required for wit-component's preview1 adapter, which
        // imports WASIp2 via `canon.lower (realloc $cabi_import_realloc)`;
        // `cabi_import_realloc` routes through a per-call `import_alloc`
        // cell that pins return-string allocations to the adapter's
        // internal `temporary_data` buffer. Falling back to the main
        // module's `cabi_realloc` here returns a ptr in the wrong
        // memory and trips an assertion inside the adapter's
        // `fd_readdir` handler. (#715.)
        if (self.current_lower_call_ctx) |cctx| {
            if (cctx.realloc) |target| {
                const executor = @import("executor.zig");
                switch (target) {
                    .aot => |t| {
                        var frame: executor.CallFrame = .{
                            .aot = call_frame_mod.AotFrame.init(t.ai, self.allocator),
                        };
                        defer frame.deinit();
                        return executor.callRealloc(&frame, t.local_idx, 0, 0, a, size) catch null;
                    },
                    .interp => |t| {
                        const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
                        if (self.realloc_env_owner != t.mi) {
                            if (self.realloc_env) |old| old.destroy();
                            self.realloc_env = ExecEnv.create(t.mi, 1024, self.allocator) catch null;
                            self.realloc_env_owner = if (self.realloc_env != null) t.mi else null;
                        }
                        const env = self.realloc_env orelse return null;
                        var frame: executor.CallFrame = .{ .interp = executor.InterpFrame.init(env) };
                        defer frame.deinit();
                        return executor.callRealloc(&frame, t.local_idx, 0, 0, a, size) catch null;
                    },
                }
            }
        }
        // Prefer the AOT path when a precompiled core owns `cabi_realloc`:
        // it dispatches straight into native code via `aot_runtime.callFuncScalar`,
        // matching the dispatch path the rest of the all-AOT component
        // already uses. The interp path below only runs when no AOT
        // core exports `cabi_realloc`. (#707 surfaced this — every
        // canon-lower import returning a `list`/`string`/multi-slot
        // compound goes through `hostAllocGuest`, and all-AOT
        // components previously failed here with `error.IoError`.)
        if (self.reallocOwnerAot()) |ai| {
            const realloc_idx = aot_runtime.findExportFunc(ai, "cabi_realloc") orelse return null;
            const executor = @import("executor.zig");
            const aot_call_frame = @import("call_frame.zig");
            var frame: executor.CallFrame = .{ .aot = aot_call_frame.AotFrame.init(ai, self.allocator) };
            defer frame.deinit();
            return executor.callRealloc(&frame, CoreFuncIdxLocal.from(realloc_idx), 0, 0, a, size) catch null;
        }
        const realloc_owner = self.reallocOwner() orelse return null;
        const realloc_local = realloc_owner.getExportFunc("cabi_realloc") orelse return null;
        const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
        const executor = @import("executor.zig");
        // Reuse a cached `ExecEnv` keyed on the realloc owner — the
        // wasi:http@0.3.0 testsuite fixtures hit `hostAllocGuest`
        // thousands of times per run and a fresh `ExecEnv.create` is
        // ~96 KiB of allocator churn each (#538). `realloc_env` is
        // discarded + re-created if the owner ever shifts.
        if (self.realloc_env_owner != realloc_owner) {
            if (self.realloc_env) |old| old.destroy();
            self.realloc_env = ExecEnv.create(realloc_owner, 1024, self.allocator) catch null;
            self.realloc_env_owner = if (self.realloc_env != null) realloc_owner else null;
        }
        const env = self.realloc_env orelse return null;
        var frame: executor.CallFrame = .{ .interp = executor.InterpFrame.init(env) };
        defer frame.deinit();
        return executor.callRealloc(&frame, CoreFuncIdxLocal.from(realloc_local), 0, 0, a, size) catch null;
    }

    /// Return a writable slice into the canonical guest memory (or the
    /// test_mem shim) covering `[ptr, ptr+len)`. Returns null if the
    /// range is out of bounds.
    pub fn writableGuestBytes(self: *ComponentInstance, ptr: u32, len: u32) ?[]u8 {
        if (self.test_mem) |tm| {
            const end = @as(usize, ptr) + @as(usize, len);
            if (end > tm.buffer.len) return null;
            return tm.buffer[ptr..end];
        }
        // When the in-flight canon-lower pinned a specific `(memory $m)`,
        // write into THAT memory — the `hostAllocGuest` ptr above came
        // from the lowerer's realloc, which allocates inside the same
        // memory. Mismatching here would write into the wrong module's
        // memory and silently corrupt unrelated guest state. (#715.)
        if (self.current_lower_call_ctx) |cctx| {
            if (cctx.memory) |mem| {
                const end = @as(usize, ptr) + @as(usize, len);
                if (end > mem.byteLen()) return null;
                return mem.data[ptr..end];
            }
        }
        const mem = self.canonicalMemory() orelse return null;
        const end = @as(usize, ptr) + @as(usize, len);
        if (end > mem.byteLen()) return null;
        return mem.data[ptr..end];
    }

    /// Allocate `bytes.len` bytes in guest memory at `alignment` byte
    /// alignment and copy `bytes` into them. Returns the guest-side
    /// pointer or null on failure (no `cabi_realloc` export, OOM, or
    /// invocation error).
    ///
    /// Used by host-side callbacks (e.g. `wasi:io/streams.[method]
    /// input-stream.blocking-read`) that must materialize a `list<T>`
    /// or `string` value into guest memory before the canonical ABI
    /// stores its `(ptr, len)` representation in a spilled result tuple.
    ///
    /// `alignment` MUST match the canonical-ABI alignment of the list's
    /// element type. Strings and `list<u8>` use `1`; lists of records
    /// containing `u32`/pointer fields (e.g. `list<tuple<string,
    /// string>>` from `get-environment`) require `4`. Passing the wrong
    /// alignment leaves the returned ptr unaligned and the guest's
    /// canon-lift rejects it (#719).
    ///
    /// Convention: wit-bindgen emits a single `cabi_realloc` export on
    /// the main core module; we call it with `(0, 0, alignment, len)`.
    pub fn hostAllocAndWrite(self: *ComponentInstance, bytes: []const u8, alignment: u32) ?u32 {
        const ptr = self.hostAllocGuest(@intCast(bytes.len), alignment) orelse return null;
        const dst = self.writableGuestBytes(ptr, @intCast(bytes.len)) orelse return null;
        @memcpy(dst, bytes);
        return ptr;
    }

    /// Test-only: enable a backing `TestGuestMem` so `hostAllocGuest`,
    /// `hostAllocAndWrite`, `writableGuestBytes`, and `readGuestBytes`
    /// operate on a self-contained buffer. Sets `test_mem` and
    /// `allocator`; other fields are left untouched (callers using
    /// `var ci: ComponentInstance = undefined` keep their UB-on-read
    /// invariant for fields they don't initialize).
    pub fn enableTestMem(self: *ComponentInstance, allocator: std.mem.Allocator, size: usize) !void {
        const tm = try allocator.create(TestGuestMem);
        errdefer allocator.destroy(tm);
        tm.* = try TestGuestMem.init(allocator, size);
        self.test_mem = tm;
        self.allocator = allocator;
    }

    /// Test-only: tear down a `TestGuestMem` previously installed via
    /// `enableTestMem`.
    pub fn disableTestMem(self: *ComponentInstance) void {
        if (self.test_mem) |tm| {
            tm.deinit(self.allocator);
            self.allocator.destroy(tm);
            self.test_mem = null;
        }
    }

    /// Kind-classify a top-level component import. Pure-type imports (those
    /// whose sole purpose is to introduce a type index) do not need a runtime
    /// binding; every other kind must be satisfied by `linkImports`.
    fn importIsRuntime(imp: ctypes.ImportDecl) bool {
        return switch (imp.desc) {
            .type => false,
            .module, .func, .value, .component, .instance => true,
        };
    }

    /// Validate that `binding` is compatible with `imp.desc`'s kind. Returns
    /// `error.ImportKindMismatch` for an outright mismatch. Cross-component
    /// wiring via `component_export` is accepted for any runtime kind and is
    /// validated more thoroughly at call time.
    fn importKindMatches(imp: ctypes.ImportDecl, binding: ImportBinding) bool {
        return switch (binding) {
            .component_export => true,
            .host_func => imp.desc == .func,
            .host_instance => imp.desc == .instance,
        };
    }

    /// Link imports against a set of provided bindings.
    ///
    /// Every runtime (non-type-only) top-level import must have a provider;
    /// missing ones fail with `error.MissingImport`. Kind mismatches (e.g.
    /// binding a `host_func` to an instance-typed import) fail with
    /// `error.ImportKindMismatch`. Pure `.type` imports are satisfied by the
    /// type system and require no runtime binding.
    ///
    /// Side effect: after binding, this drains `pending_core_starts` and
    /// runs each deferred core-module `(start ...)` directive in original
    /// declaration order. This means `linkImports` can return guest-trap
    /// errors (surfaced as `error.StartFunctionFailed` plus a
    /// `[component init trap] ...` line on stderr), and a failure leaves
    /// the instance only partially started — callers should treat such an
    /// instance as poisoned and `deinit` it. The deferral exists so that
    /// trampoline `host_funcs` are bound before any `(start)` runs;
    /// otherwise wasi-using `_initialize` traps with `HostFuncNotBound`
    /// (issue #308).
    pub fn linkImports(
        self: *ComponentInstance,
        providers: std.StringHashMapUnmanaged(ImportBinding),
    ) !void {
        if (self.link_state == .poisoned) return error.LinkAlreadyPoisoned;
        self.link_state = .linking;
        errdefer self.link_state = .poisoned;

        // Phase 1 — bind everything (parent + recursive children) WITHOUT
        // running any deferred core starts. This means all forwarding
        // host_funcs and trampoline contexts are wired before any wasm
        // start function runs, so the start can call across the
        // composition boundary in either direction without seeing
        // `error.HostFuncNotBound`.
        try self.bindImports(providers);
        try self.bindChildren();

        // Phase 2 — drain deferred core starts post-order so a parent's
        // start (which may call into child lifts after parent core
        // modules are initialized) runs before any child start that
        // might invoke parent lifts. Issue #308 invariant for nested
        // composed components.
        try self.drainAllStartsPostOrder();

        self.link_state = .linked;
    }

    /// Bind only this instance's imports and resolve trampoline
    /// `host_func`s. Does NOT run any deferred core starts. Used as the
    /// first phase of `linkImports` so child instances can be wired
    /// before any wasm start runs.
    fn bindImports(
        self: *ComponentInstance,
        providers: std.StringHashMapUnmanaged(ImportBinding),
    ) !void {
        for (self.component.imports) |imp| {
            const maybe_binding = providers.get(imp.name);
            if (maybe_binding) |binding| {
                if (!importKindMatches(imp, binding)) return error.ImportKindMismatch;
                self.imports.put(self.allocator, imp.name, binding) catch
                    return error.OutOfMemory;
            } else if (importIsRuntime(imp)) {
                return error.MissingImport;
            }
        }

        // Fill in trampoline host_funcs now that bindings are in place.
        // Each trampoline records the component func index it lowers; we
        // walk the component's imports to find the matching host binding.
        //
        // Note: `resolveComponentFuncToHostFunc` only resolves bindings of
        // kind `.host_func` and `.host_instance.members.func`. Imports bound
        // via `.component_export` (cross-component composition) leave the
        // trampoline `host_func` unset, which would surface as
        // `error.HostFuncNotBound` if a deferred core start calls them.
        // Cross-component composition with deferred starts is not currently
        // exercised by any caller; revisit when that combination is wired.
        for (self.trampoline_ctxs.items) |ctx| {
            if (resolveComponentFuncToHostFunc(self, self.component, ctx.component_func_idx)) |hf| {
                ctx.host_func = hf;
            } else {}
        }
    }

    /// Recursively bind sub-component imports. For each non-null entry
    /// in `sub_instances`, build a `ImportBinding` provider map by
    /// enumerating the child component's import declarators and
    /// matching them against the parent's `(with "<name>" <sortidx>)`
    /// arguments. `.func` arguments are wrapped in forwarding
    /// `host_func` bindings; `.instance` arguments are either reused
    /// (when the parent's binding is itself a `*const HostInstance`,
    /// e.g. inherited WASI imports) or synthesized into a per-child
    /// `*HostInstance` whose member funcs are forwarding adapters.
    /// (Issue #355.)
    fn bindChildren(self: *ComponentInstance) !void {
        if (self.sub_instances.len == 0) return;
        for (self.sub_instances, 0..) |maybe_child, local_inst_idx| {
            const child = maybe_child orelse continue;
            try self.wireSubComponentImports(child, @intCast(local_inst_idx));
            try child.bindChildren();
        }
    }

    /// Drain deferred core starts in post-order: this instance first,
    /// then each non-null sub-instance recursively. See `linkImports`
    /// for ordering rationale (issue #355).
    fn drainAllStartsPostOrder(self: *ComponentInstance) !void {
        try self.runDeferredCoreStarts();
        for (self.sub_instances) |maybe_child| {
            const child = maybe_child orelse continue;
            try child.drainAllStartsPostOrder();
        }
    }

    /// Build a child `ImportBinding` provider map for the sub-instance
    /// at `local_inst_idx` and link the child. The `.instantiate` AST
    /// node carries the parent's `with` arguments; the child's import
    /// list comes from the child's parsed component. Resolution of
    /// each `with` argument flows through the parent's index-space
    /// resolvers and yields either a forwarding `host_func`, a
    /// `host_instance` (synthesized or inherited), or a pass-through
    /// of the parent's existing binding (the WASI inheritance case).
    fn wireSubComponentImports(
        self: *ComponentInstance,
        child: *ComponentInstance,
        local_inst_idx: u32,
    ) !void {
        const parent_comp = self.component;
        if (local_inst_idx >= parent_comp.instances.len)
            return error.SubComponentLinkFailed;
        const expr = parent_comp.instances[local_inst_idx];
        const ie = switch (expr) {
            .instantiate => |x| x,
            .exports => return, // synthesized from inline exports — no runtime import wiring needed
        };
        if (ie.component_idx >= parent_comp.components.len)
            return error.SubComponentLinkFailed;
        const subcomp = parent_comp.components[ie.component_idx];

        var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
        defer providers.deinit(self.allocator);

        for (subcomp.imports) |child_imp| {
            const arg = findInstantiateArg(ie.args, child_imp.name);
            if (arg == null) {
                // No matching `with` arg. Type imports are non-runtime
                // and need no binding; everything else is required.
                if (importIsRuntime(child_imp)) return error.SubComponentLinkFailed;
                continue;
            }
            const binding = self.resolveSubComponentArg(child, child_imp, arg.?) catch |e| {
                return e;
            };
            providers.put(self.allocator, child_imp.name, binding) catch
                return error.OutOfMemory;
        }

        try child.bindImports(providers);
    }

    /// Resolve a single sub-component import to an `ImportBinding`,
    /// dispatching by the child import's declared kind. Forwarding
    /// `host_func`s and synthetic `host_instance`s are allocated from
    /// `child.allocator` and tracked in the child's owned-resources
    /// lists so their lifetime ends with the child (issue #355).
    fn resolveSubComponentArg(
        self: *ComponentInstance,
        child: *ComponentInstance,
        child_imp: ctypes.ImportDecl,
        arg: ctypes.InstantiateArg,
    ) !ImportBinding {
        switch (child_imp.desc) {
            .type => return .{ .host_func = .{} }, // never inspected; satisfies importKindMatches loosely
            .func => {
                if (arg.sort_idx.sort != .func) return error.SubComponentLinkFailed;
                const flat = self.flattenComponentFunc(arg.sort_idx.idx) catch
                    return error.SubComponentLinkFailed;
                const child_type_idx: u32 = switch (child_imp.desc) {
                    .func => |t| t,
                    else => unreachable,
                };
                const child_func_td = resolveTypeDef(child.component, child_type_idx) orelse
                    return error.SubComponentLinkFailed;
                const child_ft = switch (child_func_td) {
                    .func => |f| f,
                    else => return error.SubComponentLinkFailed,
                };
                const reg = @import("canonical_abi.zig").TypeRegistry.init(child.component);
                const ctx = try executor_mod.buildForwardingHostFnCtx(
                    child.allocator,
                    flat.owner,
                    flat.local,
                    child_ft,
                    reg,
                );
                errdefer executor_mod.destroyForwardingHostFnCtx(ctx, child.allocator);
                try child.forwarding_ctxs.append(child.allocator, ctx);
                return .{ .host_func = .{
                    .context = ctx,
                    .call = executor_mod.forwardingHostFnCall,
                    .type_idx = child_type_idx,
                } };
            },
            .instance => {
                if (arg.sort_idx.sort != .instance) return error.SubComponentLinkFailed;
                // The child's import descriptor carries the type
                // index of the expected instance type — its body
                // enumerates the member funcs we must satisfy.
                const child_inst_type_idx: u32 = switch (child_imp.desc) {
                    .instance => |t| t,
                    else => unreachable,
                };
                const td = resolveTypeDef(child.component, child_inst_type_idx) orelse
                    return error.SubComponentLinkFailed;
                const decls = switch (td) {
                    .instance => |it| it.decls,
                    else => return error.SubComponentLinkFailed,
                };

                // Resolve the parent-side instance the alias chain
                // bottoms out at.
                const target = try self.resolveInstanceArgTarget(arg.sort_idx.idx);
                switch (target) {
                    .host_instance => |hi| return .{ .host_instance = hi },
                    .runtime_instance => |target_inst| {
                        // Synthesize a HostInstance whose members
                        // forward to `target_inst` named exports.
                        // The member name in the runtime export map
                        // is `<arg-name-on-target>/<member>` for
                        // exports of an instance member, but here
                        // the target instance is the runtime
                        // ComponentInstance whose `exported_funcs`
                        // is keyed by the names we already publish.
                        // For each member func in the child's
                        // expected instance type, we look up
                        // `target_inst.exported_funcs.get(<member-name-as-published-on-target>)`.
                        // Per current registration, the parent
                        // instance publishes instance-typed exports
                        // dotted under `<instance-name>/<member>`.
                        // The arg's `sort_idx.idx` references that
                        // instance in the parent's index space —
                        // the canonical published name for its
                        // members on the parent equals the export
                        // name under which the parent is *exporting*
                        // that instance, but we don't have that name
                        // here. So instead we register names directly
                        // off of `target_inst.exported_funcs` keyed
                        // by member name alone, which matches what
                        // the issue's compose-shaped components emit
                        // (see registerInstanceExport: it publishes
                        // bare member names for a specific subset).
                        //
                        // For the wasm-tools-compose pattern at hand
                        // — parent passes `(with "docs:adder/add@…"
                        // (instance 9))` to the child — the
                        // resolution lands on a parent sub_instance
                        // whose own `<instance-name>/<member>` keys
                        // are derivable from the parent instance's
                        // export decl that backs that arg. We
                        // recover the instance-name-on-target by
                        // walking the parent instance expression to
                        // find the export name.
                        //
                        // Concretely, for the arg referencing parent
                        // index `idx` resolving to a
                        // sub_instance N, we look at how that
                        // sub_instance was registered on the parent
                        // (i.e. find the parent's export with
                        // `sort_idx.idx == idx and sort == .instance`)
                        // — but that doesn't always exist, e.g. when
                        // an alias produces an intermediate instance.
                        // Implement a focused walk that handles the
                        // case where `idx` resolves to an alias-export
                        // off another runtime instance: the alias
                        // name IS the prefix we need.
                        const member_prefix = self.lookupInstancePublishedPrefix(arg.sort_idx.idx);

                        const hi = try child.allocator.create(HostInstance);
                        errdefer child.allocator.destroy(hi);
                        hi.* = .{};
                        errdefer hi.deinit(child.allocator);

                        for (decls) |d| {
                            const exp = switch (d) {
                                .@"export" => |e| e,
                                else => continue,
                            };
                            const member_func_type: u32 = switch (exp.desc) {
                                .func => |t| t,
                                else => continue,
                            };
                            // Look up the published key on target_inst
                            // for this member. Try (in order):
                            //   1. <prefix>/<member>
                            //   2. bare <member>
                            //   3. <member-name>
                            const candidate1 = if (member_prefix) |p|
                                std.fmt.allocPrint(child.module_arena.allocator(), "{s}/{s}", .{ p, exp.name }) catch return error.OutOfMemory
                            else
                                null;

                            var resolved: ?executor_mod.FlattenedExport = null;
                            if (candidate1) |k1| {
                                resolved = executor_mod.flattenForwardedChain(target_inst, k1);
                            }
                            if (resolved == null) {
                                resolved = executor_mod.flattenForwardedChain(target_inst, exp.name);
                            }
                            const flat = resolved orelse continue;

                            // `member_func_type` is the type index inside the
                            // instance body's local type space — NOT the
                            // child component's outer type indexspace.
                            // The previous code called
                            // `resolveTypeDef(child.component, …)`, which
                            // only worked when the local index happened to
                            // alias an outer func type. Components emitted
                            // by jco — including the keyvault repro's
                            // `azure:codegen/tcgc` instance import — declare
                            // their func types inline inside the instance
                            // body, and surfaced as `error.SubComponentLinkFailed`
                            // because the outer-space lookup returned a
                            // non-func TypeDef (in this case the next
                            // outer-space instance type at the same index).
                            // (#719 cross-component composition path.)
                            const member_func_td = resolveInstanceTypeLocal(decls, member_func_type) orelse
                                return error.SubComponentLinkFailed;
                            const member_ft = switch (member_func_td) {
                                .func => |f| f,
                                else => return error.SubComponentLinkFailed,
                            };

                            // The FuncType lives inside the instance-type body,
                            // so its param/result `.record`/`.result`/`.list`/etc.
                            // type-index references are LOCAL to that body's decl
                            // space — NOT the child's outer component
                            // type-indexspace. If we hand `child.component`'s
                            // registry to the forwarding ctx as-is, the cross-
                            // memory rewriter's `valTypeHasPtrLen` walks the
                            // outer indexspace at the local index and gets back
                            // garbage (e.g. the keyvault repro's tcgc.compile
                            // result type — `result<string,string>` — resolved
                            // through the OUTER registry to an `.instance`
                            // TypeDef and `needs_rewrite` came back `false`).
                            // That caused the lifted string ptr — which points
                            // into TCGC's 31 MiB memory — to never get
                            // translated into the caller's (codegen-cli's)
                            // 22 MiB memory; canon-lift validation then tripped
                            // and the AOT dispatcher returned a sentinel that
                            // the guest tried to JSON.parse as a `len`, leading
                            // to the `unreachable` crash. (#743 / sibling of
                            // #719 / #729 — same instance-type-local index bug.)
                            //
                            // Build a per-trampoline extension that materializes
                            // the instance body's local type space at an absolute
                            // offset, rewrite each ValType to the absolute index,
                            // and hand the ctx a `fromExtended` registry that
                            // can resolve both spaces.
                            const ext_base: u32 = if (child.component.type_indexspace.len > 0)
                                @intCast(child.component.type_indexspace.len)
                            else
                                @intCast(child.component.types.len);
                            const ext = buildInstanceTypeExtension(child.allocator, decls, ext_base, child.component) catch
                                return error.OutOfMemory;
                            errdefer ext.deinit(child.allocator, true);

                            const member_params = try child.allocator.alloc(ctypes.ValType, member_ft.params.len);
                            errdefer child.allocator.free(member_params);
                            for (member_ft.params, 0..) |p, i| {
                                member_params[i] = rewriteValTypeAbsolute(ext_base, p.type);
                            }
                            const member_results: []ctypes.ValType = switch (member_ft.results) {
                                .none => try child.allocator.alloc(ctypes.ValType, 0),
                                .unnamed => |t| blk: {
                                    const r = try child.allocator.alloc(ctypes.ValType, 1);
                                    r[0] = rewriteValTypeAbsolute(ext_base, t);
                                    break :blk r;
                                },
                                .named => |named| blk: {
                                    const r = try child.allocator.alloc(ctypes.ValType, named.len);
                                    for (named, 0..) |n, i| r[i] = rewriteValTypeAbsolute(ext_base, n.type);
                                    break :blk r;
                                },
                            };
                            errdefer child.allocator.free(member_results);

                            const member_reg = @import("canonical_abi.zig").TypeRegistry.fromExtended(
                                child.component,
                                ext.extension_types,
                                ext.extension_indexspace,
                            );
                            const ctx = try executor_mod.buildForwardingHostFnCtxWithExtension(
                                child.allocator,
                                flat.owner,
                                flat.local,
                                member_params,
                                member_results,
                                member_reg,
                                ext.extension_types,
                                ext.extension_indexspace,
                            );
                            errdefer executor_mod.destroyForwardingHostFnCtx(ctx, child.allocator);
                            try child.forwarding_ctxs.append(child.allocator, ctx);
                            try hi.members.put(child.allocator, exp.name, .{ .func = .{
                                .context = ctx,
                                .call = executor_mod.forwardingHostFnCall,
                                .type_idx = member_func_type,
                            } });
                        }
                        try child.synthetic_host_instances.append(child.allocator, hi);
                        return .{ .host_instance = hi };
                    },
                }
            },
            .module, .value, .component => return error.UnsupportedSubComponentImportKind,
        }
    }

    /// Find the published name on this instance under which the
    /// instance referenced by `comp_inst_idx` (parent's component-instance
    /// index space) was exported. Used to build the
    /// `<published-name>/<member>` lookup key when forwarding instance
    /// members across a `with` boundary. Returns null when the index
    /// does not correspond to a top-level export (e.g. the arg refers
    /// to another sub_instance directly without going through an
    /// outer-level export name).
    fn lookupInstancePublishedPrefix(
        self: *const ComponentInstance,
        comp_inst_idx: u32,
    ) ?[]const u8 {
        // First try: parent has a top-level `(export <name> (instance N))`
        // referencing this comp-instance idx. Common case for
        // hand-authored components.
        for (self.component.exports) |exp| {
            const si = exp.sort_idx orelse continue;
            if (si.sort != .instance) continue;
            if (si.idx != comp_inst_idx) continue;
            return exp.name;
        }
        // Second try: the comp-instance idx is an alias of the form
        // `(alias export <inner-inst-idx> "<name>")`. The alias name
        // IS the key under which the inner instance's child published
        // its member exports. (This matches the
        // `wasm-tools compose` pattern: outer aliases child[k]'s
        // exported instance, then re-passes that alias to a sibling
        // child via `with`.)
        const ref = indexspace.resolveInstanceExpr(self.component, comp_inst_idx) catch return null;
        const got = ref orelse return null;
        return switch (got) {
            .sub_export => |se| se.name,
            else => null,
        };
    }

    pub const SubArgInstanceTarget = union(enum) {
        host_instance: *const HostInstance,
        runtime_instance: *const ComponentInstance,
    };

    /// Resolve a parent-side `(instance N)` reference (as passed to a
    /// sub-component via `with`) to the runtime instance the alias
    /// chain bottoms out at. Three possibilities: an imported
    /// host_instance (return as `.host_instance` for zero-copy
    /// pass-through); a parent-local sub_instance; or — for an alias
    /// hop — the named instance export of either of the above.
    fn resolveInstanceArgTarget(
        self: *const ComponentInstance,
        comp_inst_idx: u32,
    ) !SubArgInstanceTarget {
        const ref = indexspace.resolveInstanceExpr(self.component, comp_inst_idx) catch
            return error.SubComponentLinkFailed;
        const got = ref orelse return error.SubComponentLinkFailed;
        switch (got) {
            .imported => |imp_idx| {
                if (imp_idx >= self.component.imports.len) return error.SubComponentLinkFailed;
                const name = self.component.imports[imp_idx].name;
                const binding = self.imports.get(name) orelse return error.SubComponentLinkFailed;
                switch (binding) {
                    .host_instance => |hi| return .{ .host_instance = hi },
                    else => return error.SubComponentLinkFailed,
                }
            },
            .local => |li| {
                if (li >= self.sub_instances.len) return error.SubComponentLinkFailed;
                const child = self.sub_instances[li] orelse return error.SubComponentLinkFailed;
                return .{ .runtime_instance = child };
            },
            .sub_export => |se| switch (se.source) {
                .imported => |imp_idx| {
                    if (imp_idx >= self.component.imports.len) return error.SubComponentLinkFailed;
                    const name = self.component.imports[imp_idx].name;
                    const binding = self.imports.get(name) orelse return error.SubComponentLinkFailed;
                    switch (binding) {
                        .host_instance => |hi| return .{ .host_instance = hi },
                        else => return error.SubComponentLinkFailed,
                    }
                },
                .local => |li| {
                    if (li >= self.sub_instances.len) return error.SubComponentLinkFailed;
                    const child = self.sub_instances[li] orelse return error.SubComponentLinkFailed;
                    return .{ .runtime_instance = child };
                },
            },
        }
    }

    /// Resolve a parent component-func index to the bottoming
    /// `(owner, ExportedFunc.Local)`. Imported funcs are rejected (a
    /// parent can't forward an unbound import — it must be host-bound
    /// first); aliased funcs walk to the producing instance's named
    /// export; lifted funcs build a `Local` directly off the canon
    /// definition without requiring a top-level export name.
    fn flattenComponentFunc(
        self: *const ComponentInstance,
        comp_func_idx: u32,
    ) !executor_mod.FlattenedExport {
        const ref = indexspace.resolveCompFunc(self.component, comp_func_idx) orelse
            return error.SubComponentLinkFailed;
        switch (ref) {
            .imported => return error.SubComponentLinkFailed,
            .lifted => |canon_idx| {
                if (canon_idx >= self.component.canons.len)
                    return error.SubComponentLinkFailed;
                const lift = switch (self.component.canons[canon_idx]) {
                    .lift => |l| l,
                    else => return error.SubComponentLinkFailed,
                };
                const resolved = resolveLiftedCoreFunc(self, self.component, lift.core_func_idx);
                return .{ .owner = self, .local = .{
                    .core_instance_idx = if (resolved) |r| r.core_instance_idx else 0,
                    .core_func_idx = if (resolved) |r| r.local_func_idx else lift.core_func_idx,
                    .func_type_idx = lift.type_idx,
                    .opts = lift.opts,
                } };
            },
            .aliased => |alias_idx| {
                if (alias_idx >= self.component.aliases.len)
                    return error.SubComponentLinkFailed;
                const ie = switch (self.component.aliases[alias_idx]) {
                    .instance_export => |x| x,
                    .outer => return error.SubComponentLinkFailed,
                };
                if (ie.sort != .func) return error.SubComponentLinkFailed;
                const target = try self.resolveInstanceArgTarget(ie.instance_idx);
                switch (target) {
                    .host_instance => return error.SubComponentLinkFailed, // no host-instance forwarding
                    .runtime_instance => |inst| {
                        return executor_mod.flattenForwardedChain(inst, ie.name) orelse
                            error.SubComponentLinkFailed;
                    },
                }
            },
        }
    }

    /// Execute any core-module `(start ...)` directives whose dispatch was
    /// deferred during `instantiate`, in the original core-instance order.
    /// Drains `pending_core_starts`. Surfaces the underlying trap as
    /// `error.StartFunctionFailed` (with the diagnostic on `env.host_trap`
    /// preserved on the per-start `ExecEnv` printed by `callComponentFunc`
    /// downstream — but for instantiation-time failures the diagnostic is
    /// printed here so a failed `_initialize` doesn't masquerade as a
    /// later runtime error).
    fn runDeferredCoreStarts(self: *ComponentInstance) !void {
        const inst_mod = @import("../runtime/interpreter/instance.zig");
        const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
        const interp = @import("../runtime/interpreter/interp.zig");
        defer self.pending_core_starts.clearRetainingCapacity();
        defer self.pending_aot_starts.clearRetainingCapacity();

        for (self.pending_core_starts.items) |mi| {
            const start_idx = mi.module.start_function orelse continue;
            const env = ExecEnv.create(mi, 4096, self.allocator) catch return error.OutOfMemory;
            defer env.destroy();
            interp.executeFunction(env, start_idx) catch {
                if (env.host_trap) |ht| {
                    std.debug.print(
                        "[component init trap] core_func_idx={d} import='{s}.{s}' stage={s} error={s}\n",
                        .{ ht.core_func_idx, ht.import_module_name, ht.import_field_name, @tagName(ht.stage), ht.err_name },
                    );
                }
                return inst_mod.InstantiationError.StartFunctionFailed;
            };
        }

        for (self.pending_aot_starts.items) |entry| {
            aot_runtime.callFunc(entry.inst, entry.start_idx, void) catch |err| {
                std.debug.print(
                    "[component init trap] aot start_func={d} error={s}\n",
                    .{ entry.start_idx, @errorName(err) },
                );
                return inst_mod.InstantiationError.StartFunctionFailed;
            };
        }
    }

    /// Execute the component's start function if one is defined and not yet run.
    pub fn executeStart(self: *ComponentInstance) !void {
        if (self.started) return;
        self.started = true;

        const start = self.component.start orelse return;

        // The start function references a canon index which should be
        // a canon lift that we've already mapped to an exported func.
        // Walk exports to find the matching canon func.
        if (start.func_idx < self.component.canons.len) {
            const canon = self.component.canons[start.func_idx];
            switch (canon) {
                .lift => |lift| {
                    if (self.core_instances.len > 0) {
                        if (self.core_instances[0].module_inst) |mod_inst| {
                            const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
                            const interp = @import("../runtime/interpreter/interp.zig");
                            const env = ExecEnv.create(mod_inst, 8192, self.allocator) catch return;
                            defer env.destroy();
                            interp.executeFunction(env, lift.core_func_idx) catch return;
                        }
                    }
                },
                else => {},
            }
        }
    }

    pub fn deinit(self: *ComponentInstance) void {
        // The cached `cabi_realloc` ExecEnv must be freed before the
        // core instances it points at (#538).
        if (self.realloc_env) |env| {
            env.destroy();
            self.realloc_env = null;
            self.realloc_env_owner = null;
        }
        // Children are allocated independently of `module_arena` so we
        // can give each a deterministic deinit before the parent tears
        // down core instances / imports / arena. Walk in declaration
        // order; per-child deinit handles its own grandchildren.
        if (self.sub_instances.len > 0) {
            for (self.sub_instances) |maybe_child| {
                if (maybe_child) |child| child.deinit();
            }
            self.allocator.free(self.sub_instances);
            self.sub_instances = &.{};
        }
        // Forwarding contexts / synthetic host instances are owned by
        // this instance because its `imports` map borrows them. Free
        // them before tearing down `imports` itself.
        for (self.forwarding_ctxs.items) |ctx| executor_mod.destroyForwardingHostFnCtx(ctx, self.allocator);
        self.forwarding_ctxs.deinit(self.allocator);
        for (self.synthetic_host_instances.items) |hi| {
            hi.deinit(self.allocator);
            self.allocator.destroy(hi);
        }
        self.synthetic_host_instances.deinit(self.allocator);

        var rt_it = self.resource_tables.valueIterator();
        while (rt_it.next()) |rt| rt.deinit(self.allocator);
        self.resource_tables.deinit(self.allocator);

        // Tear down per-instance async-handle tables (#478 sub-PR 3).
        // Streams and waitable-sets own heap memory; futures buffer
        // their lowered payload (#478 sub-PR 3a) and error-contexts
        // store `[]u8` debug strings which we free below.
        var fut_it = self.futures.valueIterator();
        while (fut_it.next()) |f| f.deinit(self.allocator);
        self.futures.deinit(self.allocator);
        var stream_it = self.streams.valueIterator();
        while (stream_it.next()) |s| s.deinit(self.allocator);
        self.streams.deinit(self.allocator);
        var ec_it = self.error_contexts.valueIterator();
        while (ec_it.next()) |msg| self.allocator.free(msg.*);
        self.error_contexts.deinit(self.allocator);
        var ws_it = self.waitable_sets.valueIterator();
        while (ws_it.next()) |ws| ws.deinit(self.allocator);
        self.waitable_sets.deinit(self.allocator);
        for (self.trampoline_ctxs.items) |ctx| {
            ctx.deinit(self.allocator);
            self.allocator.destroy(ctx);
        }
        self.trampoline_ctxs.deinit(self.allocator);
        for (self.canon_builtin_ctxs.items) |ctx| {
            self.allocator.destroy(ctx);
        }
        self.canon_builtin_ctxs.deinit(self.allocator);
        self.canon_builtin_ctx_by_canon_idx.deinit(self.allocator);
        for (self.cross_instance_thunk_ctxs.items) |ctx| {
            self.allocator.free(ctx.param_types);
            self.allocator.free(ctx.result_types);
            self.allocator.free(ctx.label);
            self.allocator.destroy(ctx);
        }
        self.cross_instance_thunk_ctxs.deinit(self.allocator);
        for (self.trap_stub_labels.items) |label_ptr| {
            // Each label was allocated as a null-terminated `[]u8` via
            // `allocator.alloc(u8, n)`; recover the slice (including the
            // trailing nul) and free it.
            const label_z: [*:0]const u8 = label_ptr;
            const len = std.mem.len(label_z);
            const slice = label_z[0 .. len + 1];
            self.allocator.free(slice);
        }
        self.trap_stub_labels.deinit(self.allocator);
        if (self.aot_trampoline_pool) |pool| {
            pool.deinit(self.allocator);
            self.allocator.destroy(pool);
        }
        self.pending_core_starts.deinit(self.allocator);
        self.pending_aot_starts.deinit(self.allocator);
        if (self.core_instances.len > 0) {
            const inst_mod = @import("../runtime/interpreter/instance.zig");
            for (self.core_instances) |*entry| {
                if (entry.module_inst) |mi| inst_mod.destroy(mi);
                if (entry.aot_inst) |ai| {
                    aot_runtime.destroy(ai);
                    if (entry.lazy_driver) |driver| driver.deinit();
                }
            }
            self.allocator.free(self.core_instances);
        }
        self.module_arena.deinit();
        self.exported_funcs.deinit(self.allocator);
        self.imports.deinit(self.allocator);
        self.allocator.destroy(self);
    }
};

// ── Component Instantiation ─────────────────────────────────────────────────

pub const InstantiationError = error{
    OutOfMemory,
    InvalidComponent,
    CoreModuleLoadFailed,
    CoreModuleInstantiateFailed,
    ImportResolutionFailed,
    MissingImport,
    ImportKindMismatch,
    /// A sub-component referenced by `(instance N (instantiate ...))`
    /// failed to instantiate. Distinct from `CoreModuleInstantiateFailed`
    /// so the diagnostic line points at the right layer (issue #355).
    SubComponentInstantiateFailed,
    /// `resolveInstanceExpr` hit a multi-hop alias chain that the
    /// current implementation does not flatten. Real wasm-tools
    /// compose output never produces this; surface a clear error
    /// rather than silently dropping later exports.
    AliasChainTooComplex,
    /// A child sub-component import could not be wired during the
    /// parent's `linkImports` (e.g. a required `with` arg was missing,
    /// or its target index space slot did not resolve). Distinct from
    /// `MissingImport` (which is raised against the *parent's* own
    /// imports) so diagnostics name the right layer (issue #355).
    SubComponentLinkFailed,
    /// A sub-component declared an import of a kind we do not yet
    /// know how to forward across `with` boundaries (`.module`,
    /// `.value`, `.component`). Real wasm-tools-compose output for
    /// the composed-command shape only produces `.func` / `.instance`
    /// / `.type` imports; if we ever encounter these others, the
    /// diagnostic must surface (issue #355).
    UnsupportedSubComponentImportKind,
    /// `linkImports` was called against an instance that already
    /// failed a previous link. Callers should `deinit` instead.
    LinkAlreadyPoisoned,
    /// `Options.aot_only` was set, but at least one core has an
    /// import or instantiation step the AOT runtime cannot satisfy
    /// today. The matching `[aot reject] ...` log line names the
    /// failing core + import. Surfaced by `wamr run` as a clear
    /// "AOT cannot run this component" error. Library callers
    /// that don't set `aot_only` keep the silent interp fallback.
    /// See issue #644.
    AotImportUnresolvable,
    /// An in-memory precompiled core needed a lazy-JIT sidecar attach
    /// during AOT instantiation (#889), but the sidecar was missing or
    /// otherwise could not be attached.
    LazyJitAttachFailed,
};

/// Instantiate a parsed component, producing a runnable ComponentInstance.
///
/// When `component.core_instances` is populated, each expression is
/// processed in order:
///   - `.exports` contributes an inline instance whose named members are
///     recorded on the ComponentInstance entry (no core module instantiation).
///   - `.instantiate { module_idx, args }` loads and instantiates the
///     referenced core module, resolving each of its imports against the
///     prior core-instance exports named by `args`. Whenever an imported
///     core function is satisfied by a `canon.lower`, the runtime installs
///     a `componentTrampoline` + per-slot `ComponentTrampolineCtx` on that
///     import slot so future core calls bridge back to the host `HostFunc`.
///
/// For legacy callers with `core_instances.len == 0`, falls back to the
/// pre-2A behaviour of instantiating each `core_module` exactly once.
pub fn instantiate(
    component: *const ctypes.Component,
    allocator: std.mem.Allocator,
) InstantiationError!*ComponentInstance {
    return instantiateWithOptions(component, allocator, .{});
}

/// `instantiate` variant that accepts caller-supplied options (#625).
///
/// Options primarily carry `precompiled_cores` — a slice of
/// `(module_idx, cwasm_bytes)` pairs that opt individual embedded core
/// modules into the AOT runtime. For in-memory component precompile
/// results they may also carry an internal lazy-JIT attach hook used to
/// consume per-core sidecars at AOT instantiation time (#889). Cores
/// not covered by the slice continue to load through
/// `runtime/interpreter/loader.zig`. Bytes are borrowed and must
/// outlive the returned `ComponentInstance`.
pub fn instantiateWithOptions(
    component: *const ctypes.Component,
    allocator: std.mem.Allocator,
    options: Options,
) InstantiationError!*ComponentInstance {
    const inst = allocator.create(ComponentInstance) catch return error.OutOfMemory;

    inst.* = .{
        .component = component,
        .module_arena = std.heap.ArenaAllocator.init(allocator),
        .core_instances = &.{},
        .resource_tables = .empty,
        .exported_funcs = .{},
        .imports = .{},
        .options = options,
        .allocator = allocator,
    };
    // From here on, `inst.deinit()` is the single owner of partial-init
    // cleanup. The struct fields above are all in trivially-deinitable
    // states (empty maps, freshly-init arena, &.{} slices) so deinit on
    // partial state is safe before any further work.
    errdefer inst.deinit();

    const loader = @import("../runtime/interpreter/loader.zig");
    const inst_mod = @import("../runtime/interpreter/instance.zig");

    if (component.core_instances.len > 0) {
        // Section-aware path: walk core_instances expressions in order.
        const cis = allocator.alloc(ComponentInstance.CoreInstanceEntry, component.core_instances.len) catch return error.OutOfMemory;
        for (cis) |*entry| entry.* = .{};
        inst.core_instances = cis;

        // #644: AOT path can't yet wire cross-instance imports
        // (functions/memory/tables) between cores running on
        // different backends. If even ONE core in this component
        // would have to fall back to the interpreter, fall back
        // ALL of them — otherwise an interp core that imports
        // from an AOT sibling (or vice-versa) will silently see
        // a null host_functions slot and trap on first call. The
        // common case in P2 wit-bindgen output is exactly this:
        // the rust main core needs interp (canon.lower imports)
        // but the tiny shim core has no imports and looks AOT-
        // safe in isolation.
        //
        // When `options.aot_only` is set (the `wamr run` CLI
        // mode), the silent fallback is OFF: any AOT-unresolvable
        // core surfaces as `error.AotImportUnresolvable` so the
        // caller can produce a clear "AOT cannot run this
        // component" error instead of running on the interpreter.
        // Library callers (tests, embedders) leave `aot_only` at
        // its default (`false`) and keep today's behaviour.
        const aot_only = inst.options.aot_only;
        const force_all_interp = blk: {
            if (aot_only) break :blk false;
            for (component.core_instances) |expr2| {
                const ie = switch (expr2) {
                    .instantiate => |x| x,
                    else => continue,
                };
                if (ie.module_idx >= component.core_modules.len) continue;
                const cwasm_bytes = inst.options.findPrecompiled(component.core_modules[ie.module_idx].data, ie.module_idx) orelse continue;
                const mod_alloc = inst.module_arena.allocator();
                const probe_mod = mod_alloc.create(aot_loader.AotModule) catch break :blk true;
                probe_mod.* = aot_loader.load(cwasm_bytes, mod_alloc) catch break :blk true;
                if (firstUnsupportedAotImport(probe_mod)) |bad| {
                    if (core_backend.debugAotEnabled()) {
                        std.debug.print(
                            "[aot-debug] core_module={d} has unresolvable import '{s}.{s}' (kind={s}); forcing ALL cores to interpreter to avoid cross-backend wiring gaps\n",
                            .{ ie.module_idx, bad.module_name, bad.field_name, @tagName(bad.kind) },
                        );
                    }
                    std.log.warn(
                        "aot core {d} has import '{s}.{s}' ({s}) that AOT cannot resolve yet; forcing entire component to interpreter",
                        .{ ie.module_idx, bad.module_name, bad.field_name, @tagName(bad.kind) },
                    );
                    break :blk true;
                }
            }
            break :blk false;
        };

        // When `aot_only` is set, pre-flight every core's AOT-
        // feasibility BEFORE attempting any partial instantiation.
        // This way the typed error fires before we mutate
        // `inst.core_instances` and before any side effects on
        // the host side (preopens, etc.).
        if (aot_only) {
            for (component.core_instances) |expr2| {
                const ie = switch (expr2) {
                    .instantiate => |x| x,
                    else => continue,
                };
                if (ie.module_idx >= component.core_modules.len) continue;
                const cwasm_bytes = inst.options.findPrecompiled(component.core_modules[ie.module_idx].data, ie.module_idx) orelse {
                    std.log.warn(
                        "[aot reject] core module {d} has no precompiled artifact; `wamr` is AOT-only (#644)",
                        .{ie.module_idx},
                    );
                    return error.AotImportUnresolvable;
                };
                const mod_alloc = inst.module_arena.allocator();
                const probe_mod = mod_alloc.create(aot_loader.AotModule) catch return error.OutOfMemory;
                probe_mod.* = aot_loader.load(cwasm_bytes, mod_alloc) catch |err| {
                    std.log.warn(
                        "[aot reject] core module {d} failed AOT load: {s}",
                        .{ ie.module_idx, @errorName(err) },
                    );
                    return error.AotImportUnresolvable;
                };
                if (firstUnsupportedAotImport(probe_mod)) |bad| {
                    std.log.warn(
                        "[aot reject] core module {d} import '{s}.{s}' ({s}) cannot be resolved by AOT yet (#644)",
                        .{ ie.module_idx, bad.module_name, bad.field_name, @tagName(bad.kind) },
                    );
                    return error.AotImportUnresolvable;
                }
            }
        }

        for (component.core_instances, 0..) |expr, ci_idx| {
            switch (expr) {
                .exports => |inline_exports| {
                    cis[ci_idx] = .{ .inline_exports = inline_exports };
                },
                .instantiate => |ie| {
                    if (ie.module_idx >= component.core_modules.len) continue;
                    const core_mod = component.core_modules[ie.module_idx];

                    // #644: AOT fast-path — only viable when every
                    // import this core declares can be satisfied by
                    // the AOT runtime (WASI/spectest names in
                    // host_bridge, no canon.lower trampolines, no
                    // cross-instance funcs/memory/tables/globals).
                    // Anything else — and in particular wit-bindgen-
                    // emitted P2 cores whose imports are component-
                    // level WIT interface names like
                    // `wasi:io/streams@0.2.0` resolved by the
                    // component's canon.lower defs — has no AOT-side
                    // wiring, so the very first call into the import
                    // jumps through a null host_functions slot and
                    // segfaults. Probe feasibility before committing
                    // to AOT; on any unsupported import, log a
                    // warning + fall through to the interp path that
                    // already knows how to wire these.
                    if (!force_all_interp) blk_aot_try: {
                        const precompiled_entry = inst.options.findPrecompiledEntry(core_mod.data, ie.module_idx) orelse break :blk_aot_try;
                        const cwasm_bytes = precompiled_entry.cwasm_bytes;
                        aot_blk: {
                            const mod_alloc = inst.module_arena.allocator();
                            const aot_module_ptr = mod_alloc.create(aot_loader.AotModule) catch break :aot_blk;
                            aot_module_ptr.* = aot_loader.load(cwasm_bytes, mod_alloc) catch |err| {
                                std.log.warn("aot core load failed for module {d}: {s}", .{ ie.module_idx, @errorName(err) });
                                if (aot_only) return error.AotImportUnresolvable;
                                break :aot_blk;
                            };
                            if (firstUnsupportedAotImport(aot_module_ptr)) |bad| {
                                if (core_backend.debugAotEnabled()) {
                                    std.debug.print(
                                        "[aot-debug] skipping AOT for core_module={d}: unresolvable import '{s}.{s}' (kind={s}); falling back to interpreter\n",
                                        .{ ie.module_idx, bad.module_name, bad.field_name, @tagName(bad.kind) },
                                    );
                                }
                                std.log.warn(
                                    "aot core {d} has import '{s}.{s}' ({s}) that AOT cannot resolve yet; running on interpreter",
                                    .{ ie.module_idx, bad.module_name, bad.field_name, @tagName(bad.kind) },
                                );
                                if (aot_only) return error.AotImportUnresolvable;
                                break :aot_blk;
                            }
                            const imported_table_overrides_opt = resolveAotImportedTableOverrides(
                                allocator,
                                inst,
                                component,
                                cis,
                                ci_idx,
                                ie.args,
                                ie.module_idx,
                                aot_module_ptr,
                            ) catch {
                                if (aot_only) {
                                    std.log.warn("[aot reject] core module {d}: imported table override resolution failed", .{ie.module_idx});
                                    return error.AotImportUnresolvable;
                                }
                                break :aot_blk;
                            };
                            const imported_table_overrides = imported_table_overrides_opt orelse {
                                if (aot_only) {
                                    std.log.warn("[aot reject] core module {d}: cross-instance table wiring unresolved (see preceding [aot reject] line)", .{ie.module_idx});
                                    return error.AotImportUnresolvable;
                                }
                                break :aot_blk;
                            };
                            defer if (imported_table_overrides.len > 0) allocator.free(imported_table_overrides);

                            const imported_memory_overrides_opt = resolveAotImportedMemoryOverrides(
                                allocator,
                                inst,
                                component,
                                cis,
                                ci_idx,
                                ie.args,
                                ie.module_idx,
                                aot_module_ptr,
                            ) catch {
                                if (aot_only) {
                                    std.log.warn("[aot reject] core module {d}: imported memory override resolution failed", .{ie.module_idx});
                                    return error.AotImportUnresolvable;
                                }
                                break :aot_blk;
                            };
                            const imported_memory_overrides = imported_memory_overrides_opt orelse {
                                if (aot_only) {
                                    std.log.warn("[aot reject] core module {d}: cross-instance memory wiring unresolved (see preceding [aot reject] line)", .{ie.module_idx});
                                    return error.AotImportUnresolvable;
                                }
                                break :aot_blk;
                            };
                            defer if (imported_memory_overrides.len > 0) allocator.free(imported_memory_overrides);

                            const imported_global_overrides_opt = resolveAotImportedGlobalOverrides(
                                allocator,
                                inst,
                                component,
                                cis,
                                ci_idx,
                                ie.args,
                                ie.module_idx,
                                aot_module_ptr,
                            ) catch {
                                if (aot_only) {
                                    std.log.warn("[aot reject] core module {d}: imported global override resolution failed", .{ie.module_idx});
                                    return error.AotImportUnresolvable;
                                }
                                break :aot_blk;
                            };
                            const imported_global_overrides = imported_global_overrides_opt orelse {
                                if (aot_only) {
                                    std.log.warn("[aot reject] core module {d}: cross-instance global wiring unresolved (see preceding [aot reject] line)", .{ie.module_idx});
                                    return error.AotImportUnresolvable;
                                }
                                break :aot_blk;
                            };
                            defer if (imported_global_overrides.len > 0) allocator.free(imported_global_overrides);

                            const imported_function_overrides = resolveAotImportedFunctionOverrides(
                                allocator,
                                inst,
                                component,
                                cis,
                                ci_idx,
                                ie.args,
                                ie.module_idx,
                                aot_module_ptr,
                            ) catch |err| {
                                std.log.warn("[aot reject] core module {d}: imported function override resolution failed: {s}", .{ ie.module_idx, @errorName(err) });
                                if (aot_only) return error.AotImportUnresolvable;
                                break :aot_blk;
                            };
                            defer if (imported_function_overrides.len > 0) allocator.free(imported_function_overrides);

                            const imported_tag_overrides_opt = resolveAotImportedTagOverrides(
                                allocator,
                                cis,
                                ci_idx,
                                ie.args,
                                aot_module_ptr,
                            ) catch {
                                if (aot_only) {
                                    std.log.warn("[aot reject] core module {d}: imported tag override resolution failed", .{ie.module_idx});
                                    return error.AotImportUnresolvable;
                                }
                                break :aot_blk;
                            };
                            const imported_tag_overrides = imported_tag_overrides_opt orelse {
                                if (aot_only) {
                                    std.log.warn("[aot reject] core module {d}: cross-instance tag wiring failed (#670)", .{ie.module_idx});
                                    return error.AotImportUnresolvable;
                                }
                                break :aot_blk;
                            };
                            defer if (imported_tag_overrides.len > 0) allocator.free(imported_tag_overrides);

                            const aot_inst_ptr = aot_runtime.instantiateWithOverrides(aot_module_ptr, inst.allocator, imported_table_overrides, imported_memory_overrides, imported_global_overrides, imported_function_overrides, imported_tag_overrides) catch |err| {
                                std.log.warn("aot core instantiate failed for module {d}: {s}", .{ ie.module_idx, @errorName(err) });
                                if (aot_only) return error.AotImportUnresolvable;
                                break :aot_blk;
                            };
                            aot_runtime.mapCodeExecutable(aot_inst_ptr) catch |err| {
                                std.log.warn("aot core code-map failed for module {d}: {s}", .{ ie.module_idx, @errorName(err) });
                                aot_runtime.destroy(aot_inst_ptr);
                                if (aot_only) return error.AotImportUnresolvable;
                                break :aot_blk;
                            };
                            var lazy_driver: ?core_backend.LazyJitHandle = null;
                            lazy_driver = inst.options.attachLazyJit(precompiled_entry, aot_inst_ptr, allocator) catch |err| switch (err) {
                                error.OutOfMemory => {
                                    aot_runtime.destroy(aot_inst_ptr);
                                    return error.OutOfMemory;
                                },
                                error.LazyJitSidecarUnavailable => {
                                    std.log.warn(
                                        "aot core {d} lazy-JIT sidecar unavailable; {s}",
                                        .{ ie.module_idx, if (aot_only) @as([]const u8, "failing instantiation") else "falling back to interpreter" },
                                    );
                                    aot_runtime.destroy(aot_inst_ptr);
                                    if (aot_only) return error.LazyJitAttachFailed;
                                    break :aot_blk;
                                },
                            };
                            // Defer the AOT core module's `(start ...)` until
                            // `runDeferredCoreStarts`, after all sibling cores
                            // are wired and trampolines are bound (#308 AOT
                            // analogue).
                            if (aot_module_ptr.start_function) |start_idx| {
                                inst.pending_aot_starts.append(allocator, .{
                                    .inst = aot_inst_ptr,
                                    .start_idx = start_idx,
                                }) catch {
                                    if (lazy_driver) |driver| driver.deinit();
                                    aot_runtime.destroy(aot_inst_ptr);
                                    if (aot_only) return error.AotImportUnresolvable;
                                    break :aot_blk;
                                };
                            }
                            cis[ci_idx] = .{
                                .aot_inst = aot_inst_ptr,
                                .lazy_driver = lazy_driver,
                            };
                            // AOT cross-instance overrides were resolved
                            // above before instantiation; unresolved imports
                            // either fall back to interp or use a trap stub.
                            continue;
                        }
                    }

                    const mod_alloc = inst.module_arena.allocator();
                    const loaded = loader.load(core_mod.data, mod_alloc) catch continue;
                    const module_ptr = mod_alloc.create(core_types.WasmModule) catch continue;
                    module_ptr.* = loaded;

                    // Resolve every import against `with` args BEFORE
                    // instantiation. Three backends:
                    //   * Cross-instance: the source `with`-arg is a real core
                    //     module instance; wire the appropriate per-kind slot
                    //     (`import_functions` / `memories` / `tables` /
                    //     `globals`) so the interpreter dispatches into the
                    //     source's body / shares the same MemoryInstance etc.
                    //   * Canon.lower (function-only): the source is an
                    //     inline-exports bundle pointing at a `canon lower`
                    //     core func; install a `componentTrampoline` on
                    //     `host_func_entries[i]`.
                    //   * Unresolved: function slots fall through to a no-op
                    //     stub; non-function unresolved imports trap at
                    //     instantiation (current `instantiateWithImports`
                    //     contract — caller must satisfy them).
                    const import_func_count = module_ptr.import_function_count;
                    const import_mem_count = module_ptr.import_memory_count;
                    const import_tbl_count = module_ptr.import_table_count;
                    const import_glob_count = module_ptr.import_global_count;
                    var entries: []?core_types.HostFnEntry = &.{};
                    var imps_buf: []core_types.ImportedFunction = &.{};
                    var is_cross: []bool = &.{};
                    var mems_buf: []*core_types.MemoryInstance = &.{};
                    var tbls_buf: []*core_types.TableInstance = &.{};
                    var globs_buf: []*core_types.GlobalInstance = &.{};
                    var first_cross_src: ?*core_types.ModuleInstance = null;
                    var has_imports_resolved = false;
                    if (import_func_count > 0) {
                        entries = allocator.alloc(?core_types.HostFnEntry, import_func_count) catch continue;
                        @memset(entries, null);
                        imps_buf = allocator.alloc(core_types.ImportedFunction, import_func_count) catch {
                            allocator.free(entries);
                            continue;
                        };
                        is_cross = allocator.alloc(bool, import_func_count) catch {
                            allocator.free(entries);
                            allocator.free(imps_buf);
                            continue;
                        };
                        @memset(is_cross, false);
                    }
                    // Per-kind resolution buffers; allocated lazily so a module
                    // with e.g. zero memory imports never touches the allocator.
                    if (import_mem_count > 0) {
                        mems_buf = allocator.alloc(*core_types.MemoryInstance, import_mem_count) catch {
                            if (entries.len > 0) allocator.free(entries);
                            if (imps_buf.len > 0) allocator.free(imps_buf);
                            if (is_cross.len > 0) allocator.free(is_cross);
                            continue;
                        };
                    }
                    if (import_tbl_count > 0) {
                        tbls_buf = allocator.alloc(*core_types.TableInstance, import_tbl_count) catch {
                            if (entries.len > 0) allocator.free(entries);
                            if (imps_buf.len > 0) allocator.free(imps_buf);
                            if (is_cross.len > 0) allocator.free(is_cross);
                            if (mems_buf.len > 0) allocator.free(mems_buf);
                            continue;
                        };
                    }
                    if (import_glob_count > 0) {
                        globs_buf = allocator.alloc(*core_types.GlobalInstance, import_glob_count) catch {
                            if (entries.len > 0) allocator.free(entries);
                            if (imps_buf.len > 0) allocator.free(imps_buf);
                            if (is_cross.len > 0) allocator.free(is_cross);
                            if (mems_buf.len > 0) allocator.free(mems_buf);
                            if (tbls_buf.len > 0) allocator.free(tbls_buf);
                            continue;
                        };
                    }

                    // Walk imports once, dispatching by kind. Per-kind running
                    // index counters track the position within the per-kind
                    // import sequence — the interpreter's `ImportContext`
                    // fields are indexed by these (not by the global import
                    // index).
                    var imp_func_idx: u32 = 0;
                    var imp_mem_idx: u32 = 0;
                    var imp_tbl_idx: u32 = 0;
                    var imp_glob_idx: u32 = 0;
                    for (module_ptr.imports) |imp| {
                        // Find the `with` arg whose name matches this import's
                        // wasm "module" string (component-model `with` keys).
                        const source_inst_idx: u32 = arg_blk: {
                            for (ie.args) |arg| {
                                if (std.mem.eql(u8, arg.name, imp.module_name)) break :arg_blk arg.instance_idx;
                            }
                            break :arg_blk std.math.maxInt(u32);
                        };

                        switch (imp.kind) {
                            .function => {
                                defer imp_func_idx += 1;
                                if (source_inst_idx == std.math.maxInt(u32)) continue;
                                if (source_inst_idx >= ci_idx) continue;
                                const source_entry = cis[source_inst_idx];

                                if (source_entry.module_inst) |src_mi| {
                                    const target_func_idx = src_mi.getExportFunc(imp.field_name) orelse continue;
                                    imps_buf[imp_func_idx] = .{ .module_inst = src_mi, .func_idx = target_func_idx };
                                    is_cross[imp_func_idx] = true;
                                    if (first_cross_src == null) first_cross_src = src_mi;
                                    has_imports_resolved = true;
                                    continue;
                                }

                                // Inline-exports source — member is a core func that is either:
                                //   (a) a `canon.lower` (host trampoline), or
                                //   (b) an `alias core export` of another core instance's
                                //       func (cross-instance wiring, e.g. shim's exports
                                //       routed into $main via wit-component's shim/fixup
                                //       pattern).
                                var member_sort_idx: ?ctypes.CoreSort = null;
                                var member_idx: u32 = 0;
                                for (source_entry.inline_exports) |mem| {
                                    if (std.mem.eql(u8, mem.name, imp.field_name)) {
                                        member_sort_idx = mem.sort_idx.sort;
                                        member_idx = mem.sort_idx.idx;
                                        break;
                                    }
                                }
                                if (member_sort_idx == null) continue;
                                if (member_sort_idx.? != .func) continue;

                                const cfref = indexspace.resolveCoreFunc(component, member_idx) orelse continue;
                                // Aliased core func — resolve to the underlying
                                // {module_inst, func_idx} pair from a previously
                                // instantiated core instance.
                                switch (cfref) {
                                    .aliased => |alias_idx| {
                                        const al = component.aliases[alias_idx];
                                        const ie_al = switch (al) {
                                            .instance_export => |x| x,
                                            else => continue,
                                        };
                                        if (ie_al.instance_idx >= ci_idx) continue;
                                        const al_src = cis[ie_al.instance_idx];
                                        const al_mi = al_src.module_inst orelse continue;
                                        const al_func_idx = al_mi.getExportFunc(ie_al.name) orelse continue;
                                        imps_buf[imp_func_idx] = .{ .module_inst = al_mi, .func_idx = al_func_idx };
                                        is_cross[imp_func_idx] = true;
                                        if (first_cross_src == null) first_cross_src = al_mi;
                                        has_imports_resolved = true;
                                        continue;
                                    },
                                    .lowered => {},
                                    // ── Canon-builtin imports (#520) ───────
                                    // The wit-bindgen-emitted P3 core modules
                                    // import context.{get,set}, task.{yield,
                                    // return}, resource.{new,drop,rep}, and
                                    // async ABI canons via `(core instance
                                    // (instantiate $main (with "x" (func
                                    // $canon))))`. Install
                                    // `canonBuiltinTrampoline` for each so
                                    // dispatch routes into
                                    // `dispatchCanonBuiltin` at call time.
                                    .context_get,
                                    .context_set,
                                    .task_yield,
                                    .task_return,
                                    .resource_drop,
                                    .resource_new,
                                    .resource_rep,
                                    .async_canon,
                                    => {
                                        const canon_idx_b: u32 = switch (cfref) {
                                            .context_get => |i| i,
                                            .context_set => |i| i,
                                            .task_yield => |i| i,
                                            .task_return => |i| i,
                                            .resource_drop => |i| i,
                                            .resource_new => |i| i,
                                            .resource_rep => |i| i,
                                            .async_canon => |i| i,
                                            .aliased, .lowered => unreachable,
                                        };
                                        if (canon_idx_b >= component.canons.len) continue;
                                        // Share one `*CanonBuiltinTrampolineCtx` across every
                                        // import slot that resolves to the same canon-def-id
                                        // (e.g. two imports of `context.get` from different
                                        // interfaces in a wit-bindgen P3 module). The Canon
                                        // payload — including any memory/realloc opts —
                                        // is fully determined by the canon-def-id within
                                        // a given component, and the trampoline only needs
                                        // `{comp_inst, canon}` to dispatch correctly, so the
                                        // ctx is safe to alias. (#533)
                                        const ctx_b = ctx_blk: {
                                            if (inst.canon_builtin_ctx_by_canon_idx.get(canon_idx_b)) |existing| {
                                                break :ctx_blk existing;
                                            }
                                            const new_ctx = allocator.create(executor_mod.CanonBuiltinTrampolineCtx) catch continue;
                                            new_ctx.* = .{
                                                .comp_inst = inst,
                                                .canon = component.canons[canon_idx_b],
                                            };
                                            inst.canon_builtin_ctxs.append(allocator, new_ctx) catch {
                                                allocator.destroy(new_ctx);
                                                continue;
                                            };
                                            // Failing to record the memoisation entry is
                                            // non-fatal: `canon_builtin_ctxs` still owns the
                                            // allocation, so it will be freed on `deinit`.
                                            // Subsequent duplicate slots will allocate their
                                            // own ctx (graceful degradation under OOM).
                                            inst.canon_builtin_ctx_by_canon_idx.put(allocator, canon_idx_b, new_ctx) catch {};
                                            break :ctx_blk new_ctx;
                                        };
                                        // Snapshot the core wasm import's flat param count
                                        // so the canon-builtin dispatcher (specifically
                                        // `task.return`) knows how many typed stack slots
                                        // the guest pushed before invoking us — independent
                                        // of whether the canon result type's inner variant
                                        // can be flattened from the parent type pool. (#570)
                                        if (ctx_b.core_flat_param_count == null) {
                                            if (imp.func_type_idx) |fti| {
                                                if (fti < module_ptr.types.len) {
                                                    const ft = module_ptr.types[fti];
                                                    ctx_b.core_flat_param_count = @intCast(ft.params.len);
                                                }
                                            }
                                        }
                                        entries[imp_func_idx] = .{
                                            .func = &executor_mod.canonBuiltinTrampoline,
                                            .ctx = @ptrCast(ctx_b),
                                        };
                                        continue;
                                    },
                                }
                                const canon_idx = cfref.lowered;
                                const canon = component.canons[canon_idx];
                                const lower = switch (canon) {
                                    .lower => |l| l,
                                    else => continue,
                                };

                                const ctx_ptr = allocator.create(executor_mod.ComponentTrampolineCtx) catch continue;
                                // Prefer name-based lookup (correct for real components).
                                // Fall back to direct types[lower.func_idx] indexing for
                                // hand-authored fixtures that put the FuncType at that
                                // top-level type slot without a nested instance-type body.
                                const rft_opt: ?ResolvedFuncType = resolveCompFuncType(component, lower.func_idx) orelse blk: {
                                    if (lower.func_idx >= component.types.len) break :blk null;
                                    break :blk switch (component.types[lower.func_idx]) {
                                        .func => |f| ResolvedFuncType{ .ft = f },
                                        else => null,
                                    };
                                };
                                const rft = rft_opt orelse {
                                    allocator.destroy(ctx_ptr);
                                    continue;
                                };
                                // When the FuncType came from an instance-type body, its
                                // param/result `.type_idx` references — and any nested
                                // structural type indices — are local to that body's
                                // type indexspace. Build a per-trampoline TypeRegistry
                                // extension that materializes the local type space at
                                // an absolute offset, then rebase param/result ValTypes
                                // to absolute indices that the registry can resolve.
                                const ext_base: u32 = if (component.type_indexspace.len > 0)
                                    @intCast(component.type_indexspace.len)
                                else
                                    @intCast(component.types.len);
                                const ext: InstanceTypeExtension = if (rft.decls) |decls|
                                    buildInstanceTypeExtension(allocator, decls, ext_base, component) catch {
                                        allocator.destroy(ctx_ptr);
                                        continue;
                                    }
                                else
                                    InstanceTypeExtension.empty();
                                const ft: ctypes.FuncType = rft.ft;
                                const params = allocator.alloc(ctypes.ValType, ft.params.len) catch {
                                    ext.deinit(allocator, true);
                                    allocator.destroy(ctx_ptr);
                                    continue;
                                };
                                for (ft.params, 0..) |p, i| {
                                    params[i] = if (rft.decls != null)
                                        rewriteValTypeAbsolute(ext_base, p.type)
                                    else
                                        p.type;
                                }
                                const results = switch (ft.results) {
                                    .none => allocator.alloc(ctypes.ValType, 0) catch {
                                        allocator.free(params);
                                        ext.deinit(allocator, true);
                                        allocator.destroy(ctx_ptr);
                                        continue;
                                    },
                                    .unnamed => |t| blk2: {
                                        const r = allocator.alloc(ctypes.ValType, 1) catch {
                                            allocator.free(params);
                                            ext.deinit(allocator, true);
                                            allocator.destroy(ctx_ptr);
                                            continue;
                                        };
                                        r[0] = if (rft.decls != null) rewriteValTypeAbsolute(ext_base, t) else t;
                                        break :blk2 r;
                                    },
                                    .named => |named| blk2: {
                                        const r = allocator.alloc(ctypes.ValType, named.len) catch {
                                            allocator.free(params);
                                            ext.deinit(allocator, true);
                                            allocator.destroy(ctx_ptr);
                                            continue;
                                        };
                                        for (named, 0..) |n, i| {
                                            r[i] = if (rft.decls != null)
                                                rewriteValTypeAbsolute(ext_base, n.type)
                                            else
                                                n.type;
                                        }
                                        break :blk2 r;
                                    },
                                };
                                ctx_ptr.* = .{
                                    .comp_inst = inst,
                                    .host_func = .{},
                                    .component_func_idx = lower.func_idx,
                                    .param_types = params,
                                    .result_types = results,
                                    .lower_opts = executor_mod.LowerOptions.fromOpts(lower.opts),
                                    .extended_types = ext.extension_types,
                                    .extended_indexspace = ext.extension_indexspace,
                                    .is_async_func = ft.is_async,
                                };
                                inst.trampoline_ctxs.append(allocator, ctx_ptr) catch {
                                    ctx_ptr.deinit(allocator);
                                    allocator.destroy(ctx_ptr);
                                    continue;
                                };

                                entries[imp_func_idx] = .{
                                    .func = &executor_mod.componentTrampoline,
                                    .ctx = @ptrCast(ctx_ptr),
                                };
                            },
                            .memory => {
                                defer imp_mem_idx += 1;
                                if (source_inst_idx == std.math.maxInt(u32)) continue;
                                if (source_inst_idx >= ci_idx) continue;
                                const source_entry = cis[source_inst_idx];
                                if (source_entry.module_inst) |src_mi| {
                                    const exp = src_mi.module.findExport(imp.field_name, .memory) orelse continue;
                                    if (exp.index >= src_mi.memories.len) continue;
                                    mems_buf[imp_mem_idx] = src_mi.memories[exp.index];
                                    if (first_cross_src == null) first_cross_src = src_mi;
                                    has_imports_resolved = true;
                                    continue;
                                }
                                // Inline-exports source: member references a
                                // top-level core memory contributed by an
                                // `alias core export` decl. Follow the alias
                                // back to the original module instance.
                                for (source_entry.inline_exports) |mem| {
                                    if (!std.mem.eql(u8, mem.name, imp.field_name)) continue;
                                    if (mem.sort_idx.sort != .memory) break;
                                    const mi_ptr = resolveCoreMemoryToMI(inst, component, mem.sort_idx.idx) orelse break;
                                    mems_buf[imp_mem_idx] = mi_ptr;
                                    has_imports_resolved = true;
                                    break;
                                }
                            },
                            .table => {
                                defer imp_tbl_idx += 1;
                                if (source_inst_idx == std.math.maxInt(u32)) continue;
                                if (source_inst_idx >= ci_idx) continue;
                                const source_entry = cis[source_inst_idx];
                                const t_ptr = resolveCoreInstanceTableExport(inst, component, source_entry, imp.field_name) orelse continue;
                                tbls_buf[imp_tbl_idx] = t_ptr;
                                if (first_cross_src == null) {
                                    if (source_entry.module_inst) |src_mi| first_cross_src = src_mi;
                                }
                                has_imports_resolved = true;
                            },
                            .global => {
                                defer imp_glob_idx += 1;
                                if (source_inst_idx == std.math.maxInt(u32)) continue;
                                if (source_inst_idx >= ci_idx) continue;
                                const source_entry = cis[source_inst_idx];
                                if (source_entry.module_inst) |src_mi| {
                                    const exp = src_mi.module.findExport(imp.field_name, .global) orelse continue;
                                    if (exp.index >= src_mi.globals.len) continue;
                                    globs_buf[imp_glob_idx] = src_mi.globals[exp.index];
                                    if (first_cross_src == null) first_cross_src = src_mi;
                                    has_imports_resolved = true;
                                    continue;
                                }
                                for (source_entry.inline_exports) |mem| {
                                    if (!std.mem.eql(u8, mem.name, imp.field_name)) continue;
                                    if (mem.sort_idx.sort != .global) break;
                                    const g_ptr = resolveCoreGlobalToMI(inst, component, mem.sort_idx.idx) orelse break;
                                    globs_buf[imp_glob_idx] = g_ptr;
                                    has_imports_resolved = true;
                                    break;
                                }
                            },
                            else => {},
                        }
                    }

                    // Instantiate, optionally seeding `import_functions` for
                    // cross-instance wiring. Slots that aren't cross-instance
                    // get a safe placeholder pointing at any cross-source we
                    // saw — interp dispatch never reaches them because their
                    // `host_func_entries[i]` is non-null (canon.lower) or the
                    // import is unresolved (caught by the no-op stub before
                    // `import_functions` is consulted).
                    const mi = blk: {
                        if (has_imports_resolved or first_cross_src != null) {
                            if (first_cross_src) |placeholder| {
                                for (imps_buf, 0..) |*slot, i| {
                                    if (!is_cross[i]) {
                                        slot.* = .{ .module_inst = placeholder, .func_idx = 0 };
                                    }
                                }
                            }
                            const ctx = inst_mod.ImportContext{
                                .functions = imps_buf,
                                .memories = mems_buf,
                                .tables = tbls_buf,
                                .globals = globs_buf,
                                .cross_instance_mask = is_cross,
                            };
                            // Defer core `(start ...)` execution so canon-lower
                            // trampoline `host_funcs` are bound by `linkImports`
                            // before any start runs (issue #308).
                            break :blk inst_mod.instantiateWithOptions(module_ptr, allocator, .{
                                .import_ctx = ctx,
                                .defer_start = true,
                            }) catch {
                                if (entries.len > 0) allocator.free(entries);
                                if (imps_buf.len > 0) allocator.free(imps_buf);
                                if (is_cross.len > 0) allocator.free(is_cross);
                                if (mems_buf.len > 0) allocator.free(mems_buf);
                                if (tbls_buf.len > 0) allocator.free(tbls_buf);
                                if (globs_buf.len > 0) allocator.free(globs_buf);
                                continue;
                            };
                        }
                        break :blk inst_mod.instantiateWithOptions(module_ptr, allocator, .{
                            .defer_start = true,
                        }) catch {
                            if (entries.len > 0) allocator.free(entries);
                            if (imps_buf.len > 0) allocator.free(imps_buf);
                            if (is_cross.len > 0) allocator.free(is_cross);
                            if (mems_buf.len > 0) allocator.free(mems_buf);
                            if (tbls_buf.len > 0) allocator.free(tbls_buf);
                            if (globs_buf.len > 0) allocator.free(globs_buf);
                            continue;
                        };
                    };
                    cis[ci_idx] = .{ .module_inst = mi };

                    if (entries.len > 0) inst_mod.attachHostFuncEntries(mi, entries);
                    if (module_ptr.start_function != null) {
                        inst.pending_core_starts.append(allocator, mi) catch {
                            // OOM here is fatal — without the start running,
                            // the component cannot reach a usable state.
                            if (imps_buf.len > 0) allocator.free(imps_buf);
                            if (is_cross.len > 0) allocator.free(is_cross);
                            if (mems_buf.len > 0) allocator.free(mems_buf);
                            if (tbls_buf.len > 0) allocator.free(tbls_buf);
                            if (globs_buf.len > 0) allocator.free(globs_buf);
                            return error.OutOfMemory;
                        };
                    }
                    {
                        var nset: u32 = 0;
                        for (entries) |e| if (e != null) {
                            nset += 1;
                        };
                    }
                    if (imps_buf.len > 0) allocator.free(imps_buf);
                    if (is_cross.len > 0) allocator.free(is_cross);
                    if (mems_buf.len > 0) allocator.free(mems_buf);
                    if (tbls_buf.len > 0) allocator.free(tbls_buf);
                    if (globs_buf.len > 0) allocator.free(globs_buf);
                },
            }
        }
    } else if (component.core_modules.len > 0) {
        // Legacy path: one core instance per core module.
        const cis = allocator.alloc(ComponentInstance.CoreInstanceEntry, component.core_modules.len) catch return error.OutOfMemory;
        for (component.core_modules, 0..) |core_mod, i| {
            cis[i] = .{};
            const mod_alloc = inst.module_arena.allocator();
            const module = loader.load(core_mod.data, mod_alloc) catch continue;
            const module_ptr = mod_alloc.create(core_types.WasmModule) catch continue;
            module_ptr.* = module;
            const module_inst = inst_mod.instantiate(module_ptr, allocator) catch continue;
            cis[i].module_inst = module_inst;
        }
        inst.core_instances = cis;
    }

    // ── Sub-component instantiation (issue #355) ───────────────────────────
    //
    // `wasm-tools compose` produces wrapper components that hold the actual
    // command/library code inside nested sub-components and expose
    // `wasi:cli/run` via `(alias export <local-instance> "wasi:cli/run@…")`.
    // Walk `component.instances` and, for each `.instantiate { component_idx
    // … }`, build a child `ComponentInstance` so its `canon.lift`s are
    // backed by real core instances. Linking + deferred starts run later
    // during the parent's `linkImports` so child WASI imports inherited
    // through `with`-args see fully-bound parent bindings (#308 invariant
    // for nested instances).
    if (component.instances.len > 0) {
        const subs = allocator.alloc(?*ComponentInstance, component.instances.len) catch
            return error.OutOfMemory;
        @memset(subs, null);
        inst.sub_instances = subs;

        for (component.instances, 0..) |expr, i| {
            switch (expr) {
                // Inline-export bundles are satisfied lexically by the
                // existing `registerInstanceExport.exports` arm; no
                // runtime sub-instance is needed.
                .exports => continue,
                .instantiate => |ie| {
                    if (ie.component_idx >= component.components.len) continue;
                    const subcomp = component.components[ie.component_idx];
                    if (isImportFuncReExportShim(subcomp)) {
                        // The wit-bindgen "0.2.0-shim" pattern — the
                        // sub-component is purely an imported-func
                        // re-export wrapper. The existing
                        // `registerInstanceExport.instantiate` arm
                        // resolves these via parent-side `with`-arg
                        // matching without ever instantiating the
                        // sub-component. Leave the slot null so that
                        // path keeps handling them.
                        continue;
                    }
                    // Propagate the parent's instantiation options
                    // (precompiled_cores, aot_only) into the sub-
                    // component so composed components produced by
                    // `wabt component compose -d` — whose actual cores
                    // live inside nested sub-components — pick up the
                    // precompiled artifacts that `wamr run` emits for
                    // them. `PrecompiledCore.component` scopes each
                    // entry to a specific (sub-)component pointer so
                    // sibling sub-components don't collide on shared
                    // local `module_idx` values. (#662 phase D)
                    subs[i] = instantiateWithOptions(subcomp, allocator, inst.options) catch
                        return error.SubComponentInstantiateFailed;
                },
            }
        }
    }

    // Build export map: walk top-level component exports, resolve each
    // `.func` export through the component-func index space to its backing
    // `canon.lift`, and register an entry in `exported_funcs` keyed by the
    // export name.
    //
    // Top-level *instance* exports (e.g. `wasi:cli/run@0.2.6`) are
    // handled by walking the locally-instantiated instance's inline-exports
    // and registering each member func under both the dotted name
    // (`<instance-name>/<member>`) and — for the canonical `wasi:cli/run`
    // shape — the bare member name so the existing `runComponent` adapter
    // can locate `"run"` without knowing the version suffix.
    for (component.exports) |exp| {
        const si = exp.sort_idx orelse continue;
        switch (si.sort) {
            .func => registerLiftedExport(inst, component, allocator, exp.name, si.idx),
            .instance => registerInstanceExport(inst, component, allocator, exp.name, si.idx),
            else => {},
        }
    }

    return inst;
}

/// Detect the wit-bindgen "imported-func re-export" sub-component
/// shape: zero core modules, every `.func`-sort export resolves to a
/// component-func import. Composed components built by `wasm-tools
/// compose` may contain such shims as part of the `wasi:cli/run`
/// wiring; their lifts live in the parent (or sibling sub-components),
/// not inside the shim. The existing `registerInstanceExport.instantiate`
/// path resolves them via `with`-arg name matching without needing a
/// runtime instance, so leave `sub_instances[i]` null for these.
fn isImportFuncReExportShim(subcomp: *const ctypes.Component) bool {
    if (subcomp.core_modules.len != 0) return false;
    if (subcomp.exports.len == 0) return false;
    var saw_func_export = false;
    for (subcomp.exports) |exp| {
        const si = exp.sort_idx orelse continue;
        if (si.sort != .func) continue;
        saw_func_export = true;
        const ref = indexspace.resolveCompFunc(subcomp, si.idx) orelse return false;
        switch (ref) {
            .imported => {},
            else => return false,
        }
    }
    return saw_func_export;
}

/// Find an `(with "<name>" <sortidx>)` argument in an instantiate
/// expression's arg list. Returns null when no arg matches the
/// requested name. Used by `wireSubComponentImports` (issue #355).
fn findInstantiateArg(args: []const ctypes.InstantiateArg, name: []const u8) ?ctypes.InstantiateArg {
    for (args) |arg| {
        if (std.mem.eql(u8, arg.name, name)) return arg;
    }
    return null;
}

fn registerLiftedExport(
    inst: *ComponentInstance,
    component: *const ctypes.Component,
    allocator: std.mem.Allocator,
    name: []const u8,
    func_idx: u32,
) void {
    const ref = indexspace.resolveCompFunc(component, func_idx) orelse return;
    const canon_idx = switch (ref) {
        .lifted => |i| i,
        else => return,
    };
    switch (component.canons[canon_idx]) {
        .lift => |lift| {
            const resolved = resolveLiftedCoreFunc(inst, component, lift.core_func_idx);
            inst.exported_funcs.put(allocator, name, .{ .local = .{
                .core_instance_idx = if (resolved) |r| r.core_instance_idx else 0,
                .core_func_idx = if (resolved) |r| r.local_func_idx else lift.core_func_idx,
                .func_type_idx = lift.type_idx,
                .opts = lift.opts,
            } }) catch {};
        },
        else => {},
    }
}

fn registerInstanceExport(
    inst: *ComponentInstance,
    component: *const ctypes.Component,
    allocator: std.mem.Allocator,
    instance_name: []const u8,
    instance_idx: u32,
) void {
    const ier_or_err = indexspace.resolveInstanceExpr(component, instance_idx);
    const ier = (ier_or_err catch return) orelse return;
    switch (ier) {
        .imported => return,
        .local => |local_idx| registerInstanceExportLocal(
            inst,
            component,
            allocator,
            instance_name,
            local_idx,
        ),
        .sub_export => |se| {
            switch (se.source) {
                // Aliased export of an imported instance — host instance,
                // no parent-side lifts to re-publish.
                .imported => return,
                .local => |local_idx| registerInstanceExportSubExport(
                    inst,
                    component,
                    allocator,
                    instance_name,
                    local_idx,
                    se.name,
                ),
            }
        },
    }
}

/// Register members of a parent-local instance expression as exports.
/// This is the original (pre-#355) path: `expr` is either an inline
/// `.exports` bundle or a parent-level `.instantiate` of a (typically
/// shim) sub-component.
fn registerInstanceExportLocal(
    inst: *ComponentInstance,
    component: *const ctypes.Component,
    allocator: std.mem.Allocator,
    instance_name: []const u8,
    local_idx: u32,
) void {
    if (local_idx >= component.instances.len) return;
    const expr = component.instances[local_idx];
    const expose_bare = isWasiCliRunName(instance_name);

    switch (expr) {
        .exports => |inline_exports| {
            for (inline_exports) |mem| {
                if (mem.sort_idx.sort != .func) continue;
                const dotted = std.fmt.allocPrint(inst.module_arena.allocator(), "{s}/{s}", .{ instance_name, mem.name }) catch continue;
                registerLiftedExport(inst, component, allocator, dotted, mem.sort_idx.idx);
                if (expose_bare and std.mem.eql(u8, mem.name, "run")) {
                    registerLiftedExport(inst, component, allocator, "run", mem.sort_idx.idx);
                }
            }
        },
        .instantiate => |inst_expr| {
            // The wit-bindgen "0.2.0-shim" pattern: the wasi:cli/run
            // export is an instance produced by instantiating a tiny
            // sub-component whose only purpose is to re-export an
            // imported func ("import-func-run") under the canonical
            // member name "run". We resolve such an instance member by:
            //   1. looking up the sub-component's named export,
            //   2. mapping its func sort_idx into the sub-component's
            //      func index space (it must be an imported func),
            //   3. matching that import's name against the parent
            //      `with` arg list, and
            //   4. resolving the parent's argument through the parent's
            //      indexspace. The parent func ref is the one we
            //      register under `<instance>/<member>`.
            if (inst_expr.component_idx >= component.components.len) return;
            const subcomp = component.components[inst_expr.component_idx];

            // If we instantiated this sub-component as a real runtime
            // instance (i.e. it is NOT an imported-func re-export
            // shim), publish forwarded exports against it directly
            // rather than using the imported-func re-export shim path.
            if (local_idx < inst.sub_instances.len) {
                if (inst.sub_instances[local_idx]) |child| {
                    publishChildInstanceMembers(
                        inst,
                        allocator,
                        instance_name,
                        child,
                        instance_name, // child's published name == parent export name
                    );
                    return;
                }
            }

            for (subcomp.exports) |sub_exp| {
                if (sub_exp.desc != .func) continue;
                const sub_si = sub_exp.sort_idx orelse continue;
                if (sub_si.sort != .func) continue;
                const sub_ref = indexspace.resolveCompFunc(subcomp, sub_si.idx) orelse continue;
                const sub_imp_idx: u32 = switch (sub_ref) {
                    .imported => |i| i,
                    else => continue,
                };
                if (sub_imp_idx >= subcomp.imports.len) continue;
                const import_name = subcomp.imports[sub_imp_idx].name;
                // Find matching `with` arg in the parent instantiate.
                const parent_func_idx: u32 = blk: {
                    for (inst_expr.args) |arg| {
                        if (arg.sort_idx.sort != .func) continue;
                        if (std.mem.eql(u8, arg.name, import_name)) {
                            break :blk arg.sort_idx.idx;
                        }
                    }
                    continue;
                };
                const dotted = std.fmt.allocPrint(inst.module_arena.allocator(), "{s}/{s}", .{ instance_name, sub_exp.name }) catch continue;
                registerLiftedExport(inst, component, allocator, dotted, parent_func_idx);
                if (expose_bare and std.mem.eql(u8, sub_exp.name, "run")) {
                    registerLiftedExport(inst, component, allocator, "run", parent_func_idx);
                }
            }
        },
    }
}

/// Register members of a child sub-instance's named instance export
/// (`(alias export <child-local-idx> "<sub_name>")`) as parent exports
/// keyed by `<top-level-name>/<member>` (and bare `<member>` for the
/// `wasi:cli/run` shape). Each member becomes a `.forwarded` entry
/// pointing at the child's pre-registered `<sub_name>/<member>` key.
/// (Issue #355.)
fn registerInstanceExportSubExport(
    inst: *ComponentInstance,
    component: *const ctypes.Component,
    allocator: std.mem.Allocator,
    instance_name: []const u8,
    local_idx: u32,
    sub_name: []const u8,
) void {
    if (local_idx >= inst.sub_instances.len) return;
    const child = inst.sub_instances[local_idx] orelse {
        // Sub-instance is the wit-bindgen shim — no runtime instance,
        // so re-publication via forwarding is impossible. Fall back
        // to the existing local-instantiate path against the
        // `(local_idx -> instances[i])` AST, matching `with` arg
        // names against shim imports.
        registerInstanceExportLocal(inst, component, allocator, instance_name, local_idx);
        return;
    };

    publishChildInstanceMembers(inst, allocator, instance_name, child, sub_name);
}

/// Publish each function member of the child's `<sub_name>` instance
/// export under the parent's `<instance_name>/<member>` (and bare
/// `<member>` for the `wasi:cli/run` shape) as `.forwarded` entries.
/// Member names come from the child component's instance-type body
/// for the matching export — not from prefix-iterating the runtime
/// hashmap (which is order-unstable and may include non-instance
/// keys).
fn publishChildInstanceMembers(
    inst: *ComponentInstance,
    allocator: std.mem.Allocator,
    instance_name: []const u8,
    child: *ComponentInstance,
    sub_name: []const u8,
) void {
    // Walk the child's already-registered runtime exports for keys
    // shaped `<sub_name>/<member>`. We can't recover member names
    // from the child's static export type body alone — top-level
    // exports of sort `.instance` may omit their externdesc in the
    // binary (the loader infers a placeholder type idx of 0), so
    // `resolveTypeDef` would land on the wrong instance type
    // signature. The child's `exported_funcs` is the authoritative
    // post-registration view: it was populated by the child's own
    // `registerInstanceExport` walk over its `(export <name>
    // (instance ...))` decls.
    const expose_bare = isWasiCliRunName(instance_name);
    const arena = inst.module_arena.allocator();

    // Two-key prefix match: `<sub_name>/<member>` and the bare
    // `<member>` form. We only forward the dotted form here; the
    // bare alias is handled below for the `wasi:cli/run` shape.
    var prefix_buf = std.ArrayListUnmanaged(u8).empty;
    defer prefix_buf.deinit(arena);
    prefix_buf.appendSlice(arena, sub_name) catch return;
    prefix_buf.append(arena, '/') catch return;
    const prefix = prefix_buf.items;

    var it = child.exported_funcs.iterator();
    while (it.next()) |entry| {
        const child_key = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, child_key, prefix)) continue;
        const member = child_key[prefix.len..];
        // Skip nested-dotted keys (would only arise from grand-child
        // forwarded re-publication; not applicable here).
        if (std.mem.indexOfScalar(u8, member, '/') != null) continue;

        const parent_key = std.fmt.allocPrint(arena, "{s}/{s}", .{ instance_name, member }) catch continue;
        // Store an owned copy of `child_key` to insulate against
        // future child-side hashmap rehashes invalidating its
        // internal slice. Owned by `inst.module_arena`.
        const owned_child_key = arena.dupe(u8, child_key) catch continue;
        inst.exported_funcs.put(allocator, parent_key, .{ .forwarded = .{
            .owner = child,
            .owner_export_name = owned_child_key,
        } }) catch continue;
        if (expose_bare and std.mem.eql(u8, member, "run")) {
            inst.exported_funcs.put(allocator, "run", .{ .forwarded = .{
                .owner = child,
                .owner_export_name = owned_child_key,
            } }) catch continue;
        }
    }
}

/// Match `wasi:cli/run` and `wasi:cli/run@<version>` instance export names.
fn isWasiCliRunName(name: []const u8) bool {
    const prefix = "wasi:cli/run";
    if (!std.mem.startsWith(u8, name, prefix)) return false;
    const rest = name[prefix.len..];
    return rest.len == 0 or rest[0] == '@';
}

/// Probe whether every import declared by an AOT-loaded core module
/// has a wire-up the AOT runtime can satisfy. Today the AOT host
/// bridge knows preview1 `wasi`/`wasi_snapshot_preview1`/
/// `wasi_unstable` and `spectest` directly; any other function import
/// is satisfied by `resolveAotImportedFunctionOverrides` which
/// installs either a cross-instance core-to-core thunk (when the
/// import resolves to a sibling core's exported func) or a trap-on-
/// call stub (#662 Phase C). Mutable globals borrow the exporter's
/// retained `GlobalInstance` and AOT call exit now writes mutable
/// imported slab slots back to that canonical value (#660 item 2).
/// Tag imports remain out of scope and still surface here as
/// unsupported.
/// Returns the first unsupported import, or null when the core is
/// safe to commit to the AOT backend (#644). A null `imports` slice
/// (no imports at all) is always supported.
fn firstUnsupportedAotImport(module: *const aot_loader.AotModule) ?aot_loader.AotImportDesc {
    for (module.imports) |imp| {
        switch (imp.kind) {
            // Function imports route through `resolveAotImportedFunctionOverrides`
            // at instantiation time. WASI / spectest land in the host bridge;
            // everything else gets a cross-instance thunk or a trap-on-call
            // stub. Instantiation always succeeds; calls into unbridged
            // imports surface as a clean trap rather than a null-slot
            // segfault (#662 Phase C).
            .function => continue,
            // Tables and memories support cross-instance borrowing via
            // `instantiateWithOverrides`'s `imported_table_overrides` /
            // `imported_memory_overrides` slices. The caller's
            // `resolveAotImported{Table,Memory}Overrides` will surface a
            // null and fall back to interp if a `with` arg can't be
            // satisfied, so we don't need to re-check feasibility here.
            .table, .memory => continue,
            // Global imports flow through `imported_global_overrides`: the
            // runtime borrows the exporter's retained `GlobalInstance` and
            // seeds each AOT call's globals slab from it. Mutable imports are
            // accepted because call exit flushes the slab slot back to that
            // canonical value (#660 item 2).
            .global => continue,
            // Tag imports are wired the same way as globals/tables/memories:
            // a sibling core's exported `TagInstance` is passed in via
            // `imported_tag_overrides` so AOT cores can `throw` and `catch`
            // a shared tag identity. Closes #670.
            .tag => continue,
        }
    }
    return null;
}

// ── Index-space helpers ─────────────────────────────────────────────────────
//
// These resolvers implement the narrow subset needed for the Phase 2A.2b
// hand-authored fixture: no core aliases, no imported core funcs, every
// core func in the core-func-index-space comes from a `canon.lower`.
// A later slice will replace them with a section-order-aware resolver that
// handles arbitrary component layouts including `stdio-echo.wasm`.

/// Map a core-func-index-space index back to the canon.lower that
/// produced it. Returns null when the index does not point to a lower
/// (e.g. when it refers to an aliased core func).
fn resolveCoreFuncLower(component: *const ctypes.Component, core_func_idx: u32) ?u32 {
    return switch (indexspace.resolveCoreFunc(component, core_func_idx) orelse return null) {
        .lowered => |i| i,
        .resource_drop,
        .resource_new,
        .resource_rep,
        .task_yield,
        .context_get,
        .context_set,
        .task_return,
        .async_canon,
        .aliased,
        => null,
    };
}

/// Find which `ComponentInstance.core_instances[i]` hosts the core function
/// referenced by `core_func_idx`, by searching each inline-exports instance
/// for a `func`-sort export of that idx. Falls back to 0 on miss so the
/// legacy single-module layout continues to work.
fn resolveCoreFuncToInstance(component: *const ctypes.Component, core_func_idx: u32) ?u32 {
    // A canon.lift's core_func_idx typically references a function exposed
    // by the main `.instantiate` core instance (not an inline instance
    // wrapper). For now we pick the last `.instantiate` expression and
    // fall through to 0 otherwise.
    _ = core_func_idx;
    var i: usize = component.core_instances.len;
    while (i > 0) {
        i -= 1;
        switch (component.core_instances[i]) {
            .instantiate => return @intCast(i),
            else => {},
        }
    }
    return null;
}

/// Resolve a top-level core memory index to the underlying source
/// `MemoryInstance` it aliases. Only `alias core export` is currently
/// modeled.
fn resolveCoreMemoryToMI(
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    core_mem_idx: u32,
) ?*core_types.MemoryInstance {
    const ref = indexspace.resolveCoreMemory(component, core_mem_idx) orelse return null;
    const ie = component.aliases[ref.aliased].instance_export;
    if (ie.instance_idx >= inst.core_instances.len) return null;
    return resolveCoreInstanceMemoryExport(inst, component, inst.core_instances[ie.instance_idx], ie.name);
}

fn resolveCoreInstanceTableExport(
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    entry: ComponentInstance.CoreInstanceEntry,
    export_name: []const u8,
) ?*core_types.TableInstance {
    if (entry.module_inst) |src_mi| {
        const exp = src_mi.module.findExport(export_name, .table) orelse return null;
        if (exp.index >= src_mi.tables.len) return null;
        return src_mi.tables[exp.index];
    }
    if (entry.aot_inst) |src_ai| {
        const exp = src_ai.module.findExport(export_name, .table) orelse return null;
        if (exp.index >= src_ai.tables.len) return null;
        return src_ai.tables[exp.index];
    }
    for (entry.inline_exports) |mem| {
        if (!std.mem.eql(u8, mem.name, export_name)) continue;
        if (mem.sort_idx.sort != .table) break;
        return resolveCoreTableToMI(inst, component, mem.sort_idx.idx);
    }
    return null;
}

fn resolveCoreTableToMI(
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    core_tbl_idx: u32,
) ?*core_types.TableInstance {
    const ref = indexspace.resolveCoreTable(component, core_tbl_idx) orelse return null;
    const ie = component.aliases[ref.aliased].instance_export;
    if (ie.instance_idx >= inst.core_instances.len) return null;
    return resolveCoreInstanceTableExport(inst, component, inst.core_instances[ie.instance_idx], ie.name);
}

fn resolveAotImportedTableOverrides(
    allocator: std.mem.Allocator,
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    cis: []const ComponentInstance.CoreInstanceEntry,
    ci_idx: usize,
    args: []const ctypes.CoreInstantiateArg,
    module_idx: u32,
    module: *const aot_loader.AotModule,
) error{OutOfMemory}!?[]?*core_types.TableInstance {
    const imported_tables = module.importedTables();
    if (imported_tables.len == 0) return &.{};

    const overrides = try allocator.alloc(?*core_types.TableInstance, imported_tables.len);
    errdefer allocator.free(overrides);

    for (imported_tables, 0..) |imp_tbl, i| {
        const source_inst_idx: u32 = arg_blk: {
            for (args) |arg| {
                if (std.mem.eql(u8, arg.name, imp_tbl.module_name)) break :arg_blk arg.instance_idx;
            }
            break :arg_blk std.math.maxInt(u32);
        };
        if (source_inst_idx == std.math.maxInt(u32)) {
            std.log.warn(
                "[aot reject] core module {d}: imported table '{s}.{s}' — instantiate arg '{s}' not provided to core_instance ci_idx={d}",
                .{ module_idx, imp_tbl.module_name, imp_tbl.name, imp_tbl.module_name, ci_idx },
            );
            return null;
        }
        if (source_inst_idx >= ci_idx) {
            std.log.warn(
                "[aot reject] core module {d}: imported table '{s}.{s}' — source core_instance idx={d} is a forward reference (ci_idx={d})",
                .{ module_idx, imp_tbl.module_name, imp_tbl.name, source_inst_idx, ci_idx },
            );
            return null;
        }
        overrides[i] = resolveCoreInstanceTableExport(inst, component, cis[source_inst_idx], imp_tbl.name) orelse {
            std.log.warn(
                "[aot reject] core module {d}: imported table '{s}.{s}' — source core_instance idx={d} does not export a table named '{s}'",
                .{ module_idx, imp_tbl.module_name, imp_tbl.name, source_inst_idx, imp_tbl.name },
            );
            return null;
        };
    }

    return overrides;
}

fn resolveCoreInstanceMemoryExport(
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    entry: ComponentInstance.CoreInstanceEntry,
    export_name: []const u8,
) ?*core_types.MemoryInstance {
    if (entry.module_inst) |src_mi| {
        const exp = src_mi.module.findExport(export_name, .memory) orelse return null;
        if (exp.index >= src_mi.memories.len) return null;
        return src_mi.memories[exp.index];
    }
    if (entry.aot_inst) |src_ai| {
        const exp = src_ai.module.findExport(export_name, .memory) orelse return null;
        if (exp.index >= src_ai.memories.len) return null;
        return src_ai.memories[exp.index];
    }
    for (entry.inline_exports) |mem| {
        if (!std.mem.eql(u8, mem.name, export_name)) continue;
        if (mem.sort_idx.sort != .memory) break;
        return resolveCoreMemoryToMI(inst, component, mem.sort_idx.idx);
    }
    return null;
}

fn resolveAotImportedMemoryOverrides(
    allocator: std.mem.Allocator,
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    cis: []const ComponentInstance.CoreInstanceEntry,
    ci_idx: usize,
    args: []const ctypes.CoreInstantiateArg,
    module_idx: u32,
    module: *const aot_loader.AotModule,
) error{OutOfMemory}!?[]?*core_types.MemoryInstance {
    const imported_memories = module.importedMemories();
    if (imported_memories.len == 0) return &.{};

    const overrides = try allocator.alloc(?*core_types.MemoryInstance, imported_memories.len);
    errdefer allocator.free(overrides);

    for (imported_memories, 0..) |imp_mem, i| {
        const source_inst_idx: u32 = arg_blk: {
            for (args) |arg| {
                if (std.mem.eql(u8, arg.name, imp_mem.module_name)) break :arg_blk arg.instance_idx;
            }
            break :arg_blk std.math.maxInt(u32);
        };
        if (source_inst_idx == std.math.maxInt(u32)) {
            std.log.warn(
                "[aot reject] core module {d}: imported memory '{s}.{s}' — instantiate arg '{s}' not provided to core_instance ci_idx={d}",
                .{ module_idx, imp_mem.module_name, imp_mem.name, imp_mem.module_name, ci_idx },
            );
            return null;
        }
        if (source_inst_idx >= ci_idx) {
            std.log.warn(
                "[aot reject] core module {d}: imported memory '{s}.{s}' — source core_instance idx={d} is a forward reference (ci_idx={d})",
                .{ module_idx, imp_mem.module_name, imp_mem.name, source_inst_idx, ci_idx },
            );
            return null;
        }
        overrides[i] = resolveCoreInstanceMemoryExport(inst, component, cis[source_inst_idx], imp_mem.name) orelse {
            std.log.warn(
                "[aot reject] core module {d}: imported memory '{s}.{s}' — source core_instance idx={d} does not export a memory named '{s}'",
                .{ module_idx, imp_mem.module_name, imp_mem.name, source_inst_idx, imp_mem.name },
            );
            return null;
        };
    }

    return overrides;
}

fn resolveCoreInstanceGlobalExport(
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    entry: ComponentInstance.CoreInstanceEntry,
    export_name: []const u8,
) ?*core_types.GlobalInstance {
    if (entry.module_inst) |src_mi| {
        const exp = src_mi.module.findExport(export_name, .global) orelse return null;
        if (exp.index >= src_mi.globals.len) return null;
        return src_mi.globals[exp.index];
    }
    if (entry.aot_inst) |src_ai| {
        const exp = src_ai.module.findExport(export_name, .global) orelse return null;
        if (exp.index >= src_ai.globals.len) return null;
        return src_ai.globals[exp.index];
    }
    for (entry.inline_exports) |g| {
        if (!std.mem.eql(u8, g.name, export_name)) continue;
        if (g.sort_idx.sort != .global) break;
        return resolveCoreGlobalToMI(inst, component, g.sort_idx.idx);
    }
    return null;
}

fn resolveAotImportedGlobalOverrides(
    allocator: std.mem.Allocator,
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    cis: []const ComponentInstance.CoreInstanceEntry,
    ci_idx: usize,
    args: []const ctypes.CoreInstantiateArg,
    module_idx: u32,
    module: *const aot_loader.AotModule,
) error{OutOfMemory}!?[]?*core_types.GlobalInstance {
    const imported_globals = module.importedGlobals();
    if (imported_globals.len == 0) return &.{};

    const overrides = try allocator.alloc(?*core_types.GlobalInstance, imported_globals.len);
    errdefer allocator.free(overrides);

    for (imported_globals, 0..) |imp_global, i| {
        const source_inst_idx: u32 = arg_blk: {
            for (args) |arg| {
                if (std.mem.eql(u8, arg.name, imp_global.module_name)) break :arg_blk arg.instance_idx;
            }
            break :arg_blk std.math.maxInt(u32);
        };
        if (source_inst_idx == std.math.maxInt(u32)) {
            std.log.warn(
                "[aot reject] core module {d}: imported global '{s}.{s}' — instantiate arg '{s}' not provided to core_instance ci_idx={d}",
                .{ module_idx, imp_global.module_name, imp_global.name, imp_global.module_name, ci_idx },
            );
            return null;
        }
        if (source_inst_idx >= ci_idx) {
            std.log.warn(
                "[aot reject] core module {d}: imported global '{s}.{s}' — source core_instance idx={d} is a forward reference (ci_idx={d})",
                .{ module_idx, imp_global.module_name, imp_global.name, source_inst_idx, ci_idx },
            );
            return null;
        }
        overrides[i] = resolveCoreInstanceGlobalExport(inst, component, cis[source_inst_idx], imp_global.name) orelse {
            std.log.warn(
                "[aot reject] core module {d}: imported global '{s}.{s}' — source core_instance idx={d} does not export a global named '{s}'",
                .{ module_idx, imp_global.module_name, imp_global.name, source_inst_idx, imp_global.name },
            );
            return null;
        };
    }

    return overrides;
}

/// #672 commit 5: resolve a sibling core instance's exported tag by name.
/// Both interp module-instances and AOT instances expose tags through
/// `module.findExport(name, .tag)` indexed into a `tags: []*TagInstance`
/// slot, so the lookup shape mirrors `resolveCoreInstanceGlobalExport`.
fn resolveCoreInstanceTagExport(
    entry: ComponentInstance.CoreInstanceEntry,
    export_name: []const u8,
) ?*core_types.TagInstance {
    if (entry.module_inst) |src_mi| {
        const exp = src_mi.module.findExport(export_name, .tag) orelse return null;
        if (exp.index >= src_mi.tags.len) return null;
        return src_mi.tags[exp.index];
    }
    if (entry.aot_inst) |src_ai| {
        const exp = src_ai.module.findExport(export_name, .tag) orelse return null;
        if (exp.index >= src_ai.tags.len) return null;
        return src_ai.tags[exp.index];
    }
    // Inline-export bundles don't expose tag instances today (no fixture
    // exercises that path); fall through as unresolved so the caller can
    // either reject AOT or surface a clear error.
    return null;
}

/// #672 commit 5: build the `imported_tag_overrides[]` slice for an AOT
/// core's `instantiateWithOverrides` call. Each entry borrows a sibling
/// instance's exported `TagInstance` so throw / catch see the same
/// identity across cores in the same component (closes #670).
fn resolveAotImportedTagOverrides(
    allocator: std.mem.Allocator,
    cis: []const ComponentInstance.CoreInstanceEntry,
    ci_idx: usize,
    args: []const ctypes.CoreInstantiateArg,
    module: *const aot_loader.AotModule,
) error{OutOfMemory}!?[]?*core_types.TagInstance {
    const imported_tags = module.importedTags();
    if (imported_tags.len == 0) return &.{};

    const overrides = try allocator.alloc(?*core_types.TagInstance, imported_tags.len);
    errdefer allocator.free(overrides);

    for (imported_tags, 0..) |imp_tag, i| {
        const source_inst_idx: u32 = arg_blk: {
            for (args) |arg| {
                if (std.mem.eql(u8, arg.name, imp_tag.module_name)) break :arg_blk arg.instance_idx;
            }
            break :arg_blk std.math.maxInt(u32);
        };
        if (source_inst_idx == std.math.maxInt(u32)) return null;
        if (source_inst_idx >= ci_idx) return null;
        overrides[i] = resolveCoreInstanceTagExport(cis[source_inst_idx], imp_tag.name) orelse return null;
    }

    return overrides;
}

/// Lazily allocate the component-instance trampoline pool used to hand
/// AOT cores executable thunks for non-WASI fn imports (#662 Phase C).
fn ensureAotTrampolinePool(inst: *ComponentInstance) !*host_trampolines.TrampolinePool {
    if (inst.aot_trampoline_pool) |pool| return pool;
    const pool = try inst.allocator.create(host_trampolines.TrampolinePool);
    errdefer inst.allocator.destroy(pool);
    pool.* = try host_trampolines.TrampolinePool.init(inst.allocator);
    inst.aot_trampoline_pool = pool;
    return pool;
}

/// Build the `imported_function_overrides[]` slice for an AOT core's
/// `instantiateWithOverrides` call. WASI / spectest imports are left as
/// null (the AOT runtime fills them from `host_bridge` at resolve time).
/// Every other function import either resolves to a sibling core's
/// exported func (→ cross-instance thunk via the trampoline pool) or is
/// left wired as a trap-on-call stub so instantiation succeeds but a
/// later call surfaces a clean trap rather than a segfault through a
/// null host slot. (#662 Phase C).
fn resolveAotImportedFunctionOverrides(
    allocator: std.mem.Allocator,
    inst: *ComponentInstance,
    component: *const ctypes.Component,
    cis: []const ComponentInstance.CoreInstanceEntry,
    ci_idx: usize,
    args: []const ctypes.CoreInstantiateArg,
    module_idx: u32,
    module: *const aot_loader.AotModule,
) ![]const ?*const anyopaque {
    if (module.import_function_count == 0) return &.{};

    const overrides = try allocator.alloc(?*const anyopaque, module.import_function_count);
    errdefer allocator.free(overrides);
    @memset(overrides, null);

    var func_idx: u32 = 0;
    for (module.imports) |imp| {
        if (imp.kind != .function) continue;
        defer func_idx += 1;

        // Set the trampoline-pool import context so an
        // `OutOfTrampolineSlots` exhaustion error names the import
        // that tripped the cap (#756). Read inside `reserveSlot`
        // on the first refusal; ignored on the success path.
        if (inst.aot_trampoline_pool) |pool| {
            pool.setNextImportContext(imp.module_name, imp.field_name);
            // Pool already exhausted by an earlier import in this
            // walk — stop trying. The structured error already
            // fired ONCE on first refusal; emitting one
            // `[aot reject] ...` warning per remaining import
            // produces 30+ lines of noise (the pre-#756 failure
            // shape) without changing the outcome (the guest will
            // hit a null function pointer the moment it tries to
            // call any import). Better: let the override stay
            // null and leave the failure mode as "pool exhausted,
            // see error above" instead of "pool exhausted +
            // mystery segfault".
            if (pool.exhausted) {
                overrides[func_idx] = null;
                continue;
            }
        }

        // Look up the `with` arg that names this import's wasm module
        // FIRST — before any WASI / spectest short-circuit. For
        // component-embedded WASIp1 cores, `wasm-tools component new
        // --adapt …` rewrites the WASIp1 imports to point at a sibling
        // **adapter** core, but keeps `imp.module_name ==
        // "wasi_snapshot_preview1"`. If we let the WASI guard fire
        // first, the adapter wiring is discarded and the runtime falls
        // back to `host_bridge.aot*` with no `WasiCtx` on the inner
        // core's vmctx — guest sees empty argv/env (#698).
        const source_inst_idx: u32 = arg_blk: {
            for (args) |arg| {
                if (std.mem.eql(u8, arg.name, imp.module_name)) break :arg_blk arg.instance_idx;
            }
            break :arg_blk std.math.maxInt(u32);
        };

        // No `with` arg → standalone-style WASIp1 / spectest import.
        // Leave the override `null` so the AOT runtime fills it from
        // `host_bridge` at resolve time (matches pre-#662 behaviour;
        // `aot_inst.wasi_ctx` gets wired by `main.runRun` for the
        // standalone `wamr run hello.wasm` path).
        if (source_inst_idx == std.math.maxInt(u32)) {
            if (aot_host_bridge.isWasiModule(imp.module_name)) continue;
            if (aot_host_bridge.isSpectestModule(imp.module_name)) continue;
        }

        var thunk: ?*const anyopaque = null;
        if (source_inst_idx != std.math.maxInt(u32) and source_inst_idx < ci_idx) {
            const src_entry = cis[source_inst_idx];

            // Wit-component adapter pattern: the sibling is an inline-
            // exports synthetic core instance whose `imp.field_name`
            // member is a `canon.lower` contributor bridging a parent
            // WASIp2 import. Build a ComponentTrampolineCtx and route
            // through the canon-lower dispatcher rather than the
            // sibling-AOT cross-instance path (which only handles
            // alias-of-AOT-export sources).
            if (resolveInlineExportCanonRef(component, src_entry, imp.field_name)) |canon_ref| {
                switch (canon_ref) {
                    .lowered => |cl_idx| {
                        thunk = installCanonLowerBackedCrossInstanceThunk(
                            allocator,
                            inst,
                            component,
                            cl_idx,
                            imp,
                            module,
                        ) catch |err| blk: {
                            if (err != error.OutOfTrampolineSlots) {
                                std.log.warn(
                                    "[aot reject] core module {d}: canon-lower thunk for '{s}.{s}' failed ({s}); falling back to cross-instance / trap-stub",
                                    .{ module_idx, imp.module_name, imp.field_name, @errorName(err) },
                                );
                            }
                            break :blk null;
                        };
                    },
                    // Canon-builtin contributors must stay in component
                    // runtime space rather than being treated as sibling
                    // AOT exports. Their source is an inline canon def, not
                    // an AOT instance/function pair; route all supported
                    // canonical builtins through the shared AOT dispatcher.
                    .context_get,
                    .context_set,
                    .task_yield,
                    .task_return,
                    .resource_drop,
                    .resource_new,
                    .resource_rep,
                    .async_canon,
                    => {
                        const canon_idx_b: u32 = switch (canon_ref) {
                            .context_get => |i| i,
                            .context_set => |i| i,
                            .task_yield => |i| i,
                            .task_return => |i| i,
                            .resource_drop => |i| i,
                            .resource_new => |i| i,
                            .resource_rep => |i| i,
                            .async_canon => |i| i,
                            else => unreachable,
                        };
                        thunk = installCanonBuiltinBackedCrossInstanceThunk(
                            allocator,
                            inst,
                            component,
                            canon_idx_b,
                            imp,
                            module,
                        ) catch |err| blk: {
                            if (err != error.OutOfTrampolineSlots) {
                                std.log.warn(
                                    "[aot reject] core module {d}: canon-builtin thunk for '{s}.{s}' failed ({s}); falling back to cross-instance / trap-stub",
                                    .{ module_idx, imp.module_name, imp.field_name, @errorName(err) },
                                );
                            }
                            break :blk null;
                        };
                    },
                    else => {},
                }
            }

            if (thunk == null) {
                thunk = installCrossInstanceThunk(allocator, inst, component, src_entry, imp, module) catch |err| blk: {
                    // Suppress the per-import warning when the pool
                    // is exhausted: the structured `[aot trampoline-
                    // pool] exhausted ...` error already fired once
                    // from `reserveSlot`, and the next-loop-iteration
                    // `pool.exhausted` short-circuit kicks in
                    // immediately so the user sees one diagnostic
                    // instead of a 30-line warning storm (#756).
                    if (err != error.OutOfTrampolineSlots) {
                        std.log.warn(
                            "[aot reject] core module {d}: cross-instance thunk for '{s}.{s}' failed ({s}); installing trap-on-call stub",
                            .{ module_idx, imp.module_name, imp.field_name, @errorName(err) },
                        );
                    }
                    break :blk null;
                };
            }
        }

        if (thunk == null) {
            // No sibling-core wiring available — install a trap-on-call
            // stub so instantiation succeeds. Any actual call surfaces a
            // clean trap via `wamrAotDispatchTrapStub`.
            thunk = installTrapStub(allocator, inst, imp, module_idx) catch |err| blk: {
                if (err != error.OutOfTrampolineSlots) {
                    std.log.warn(
                        "[aot reject] core module {d}: failed to install trap stub for '{s}.{s}': {s}",
                        .{ module_idx, imp.module_name, imp.field_name, @errorName(err) },
                    );
                }
                break :blk null;
            };
        }

        overrides[func_idx] = thunk;
    }

    return overrides;
}

/// Resolve a `(core_instance_entry, export_name)` reference to a
/// concrete `(target_ai, target_func_idx)` pair for the cross-instance
/// thunk. Walks through inline-exports + alias chains the same way
/// `resolveCoreInstanceMemoryExport` does for memories.
fn resolveCoreInstanceFuncToAi(
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    entry: ComponentInstance.CoreInstanceEntry,
    export_name: []const u8,
) ?struct { ai: *aot_runtime.AotInstance, func_idx: u32 } {
    if (entry.aot_inst) |src_ai| {
        const exp = src_ai.module.findExport(export_name, .function) orelse return null;
        return .{ .ai = src_ai, .func_idx = exp.index };
    }
    if (entry.module_inst != null) return null; // sibling on interp backend
    for (entry.inline_exports) |mem| {
        if (!std.mem.eql(u8, mem.name, export_name)) continue;
        if (mem.sort_idx.sort != .func) break;
        // Resolve through aliases to find the underlying aot_inst.
        const cf = indexspace.resolveCoreFunc(component, mem.sort_idx.idx) orelse return null;
        switch (cf) {
            .aliased => |alias_idx| {
                if (alias_idx >= component.aliases.len) return null;
                const al = component.aliases[alias_idx];
                const ie_al = switch (al) {
                    .instance_export => |x| x,
                    else => return null,
                };
                if (ie_al.instance_idx >= inst.core_instances.len) return null;
                return resolveCoreInstanceFuncToAi(inst, component, inst.core_instances[ie_al.instance_idx], ie_al.name);
            },
            else => return null,
        }
    }
    return null;
}

fn installCrossInstanceThunk(
    allocator: std.mem.Allocator,
    inst: *ComponentInstance,
    component: *const ctypes.Component,
    source_entry: ComponentInstance.CoreInstanceEntry,
    imp: aot_loader.AotImportDesc,
    module: *const aot_loader.AotModule,
) !*const anyopaque {
    // Only support sibling AOT cores today. Sibling interp cores fall
    // through to a trap stub (calling across backends without a richer
    // bridge would produce the same null-slot segfault we are trying to
    // avoid).
    const resolved = resolveCoreInstanceFuncToAi(inst, component, source_entry, imp.field_name) orelse return error.UnsupportedCrossInstanceSource;
    const target_ai = resolved.ai;
    const target_func_idx = resolved.func_idx;

    if (imp.func_type_idx >= module.func_types.len) return error.InvalidFuncType;
    const ft = module.func_types[imp.func_type_idx];

    // Scalar signatures use either the normal nine-arg relay or the
    // dedicated 10–15 arg relay. The latter is required for lowered socket
    // address records (`tcp/udp-socket.bind` has 14 i32 slots); preserving
    // every slot is essential because a truncated address silently changes
    // the socket operation rather than trapping at the call boundary.
    if (ft.params.len > 15) return error.SignatureTooWide;
    if (ft.results.len > 1) return error.MultipleResultsUnsupported;
    for (ft.params) |p| switch (p) {
        .i32, .i64, .f32, .f64 => {},
        else => return error.UnsupportedParamType,
    };
    for (ft.results) |r| switch (r) {
        .i32, .i64, .f32, .f64 => {},
        else => return error.UnsupportedResultType,
    };

    const param_types = try allocator.alloc(core_types.ValType, ft.params.len);
    errdefer allocator.free(param_types);
    for (ft.params, 0..) |p, i| param_types[i] = p;
    const result_types = try allocator.alloc(core_types.ValType, ft.results.len);
    errdefer allocator.free(result_types);
    for (ft.results, 0..) |r, i| result_types[i] = r;

    const label = try std.fmt.allocPrint(allocator, "{s}.{s}", .{ imp.module_name, imp.field_name });
    errdefer allocator.free(label);

    const ctx = try allocator.create(executor_mod.CrossInstanceThunkCtx);
    errdefer allocator.destroy(ctx);
    ctx.* = .{
        .target_ai = target_ai,
        .target_func_idx = target_func_idx,
        .param_types = param_types,
        .result_types = result_types,
        .label = label,
    };
    try inst.cross_instance_thunk_ctxs.append(allocator, ctx);
    errdefer _ = inst.cross_instance_thunk_ctxs.pop();

    const pool = try ensureAotTrampolinePool(inst);
    const stub = try pool.allocCrossInstanceSlot(@ptrCast(ctx), .{
        .param_types = param_types,
        .result_types = result_types,
        .has_retptr = false,
    });
    return @ptrCast(stub);
}

/// Walk a sibling core instance's `inline_exports` and, if `export_name`
/// names a member backed directly by a canon contributor (rather than an
/// alias of a sibling-instance export), return that canon-func reference.
/// Returns `null` for non-canon backings (the caller should fall back to
/// the alias-walking path in `resolveCoreInstanceFuncToAi`).
///
/// wit-component adapter outputs use this pattern: the synthetic
/// `(core instance (exports ...))` instance the adapter produces names
/// each parent-component WASIp2 import via a `canon.lower` contributor.
fn resolveInlineExportCanonRef(
    component: *const ctypes.Component,
    entry: ComponentInstance.CoreInstanceEntry,
    export_name: []const u8,
) ?indexspace.CoreFuncRef {
    if (entry.module_inst != null) return null;
    if (entry.aot_inst != null) return null;
    for (entry.inline_exports) |mem| {
        if (!std.mem.eql(u8, mem.name, export_name)) continue;
        if (mem.sort_idx.sort != .func) return null;
        return indexspace.resolveCoreFunc(component, mem.sort_idx.idx);
    }
    return null;
}

/// Build a `ComponentTrampolineCtx` for a `canon.lower` decl, mirroring the
/// interp path's ctx-population at the bottom of `instantiateWithOptions`'s
/// inline-exports walk. The resulting ctx has `host_func = .{}` (empty);
/// `linkImports` rebinds it later by walking `inst.trampoline_ctxs`.
///
/// The ctx is registered on `inst.trampoline_ctxs` (which owns destruction)
/// before returning. On error, the partially-built ctx is fully cleaned up.
fn buildLoweredComponentTrampolineCtx(
    allocator: std.mem.Allocator,
    inst: *ComponentInstance,
    component: *const ctypes.Component,
    canon_lower_idx: u32,
) !*executor_mod.ComponentTrampolineCtx {
    if (canon_lower_idx >= component.canons.len) return error.UnsupportedCrossInstanceSource;
    const lower = switch (component.canons[canon_lower_idx]) {
        .lower => |l| l,
        else => return error.UnsupportedCrossInstanceSource,
    };

    const ctx_ptr = try allocator.create(executor_mod.ComponentTrampolineCtx);
    errdefer allocator.destroy(ctx_ptr);

    const rft_opt: ?ResolvedFuncType = resolveCompFuncType(component, lower.func_idx) orelse blk: {
        if (lower.func_idx >= component.types.len) break :blk null;
        break :blk switch (component.types[lower.func_idx]) {
            .func => |f| ResolvedFuncType{ .ft = f },
            else => null,
        };
    };
    const rft = rft_opt orelse return error.UnsupportedCrossInstanceSource;

    const ext_base: u32 = if (component.type_indexspace.len > 0)
        @intCast(component.type_indexspace.len)
    else
        @intCast(component.types.len);

    const ext: InstanceTypeExtension = if (rft.decls) |decls|
        try buildInstanceTypeExtension(allocator, decls, ext_base, component)
    else
        InstanceTypeExtension.empty();
    // Once `ctx_ptr.*` is initialized with `extended_types/indexspace`,
    // the ComponentTrampolineCtx.deinit path owns those slices; before
    // that, the local errdefer below frees them.
    var ext_owned_by_ctx = false;
    errdefer if (!ext_owned_by_ctx) ext.deinit(allocator, true);

    const ft = rft.ft;
    const params = try allocator.alloc(ctypes.ValType, ft.params.len);
    errdefer allocator.free(params);
    for (ft.params, 0..) |p, i| {
        params[i] = if (rft.decls != null) rewriteValTypeAbsolute(ext_base, p.type) else p.type;
    }
    const results = switch (ft.results) {
        .none => try allocator.alloc(ctypes.ValType, 0),
        .unnamed => |t| blk: {
            const r = try allocator.alloc(ctypes.ValType, 1);
            r[0] = if (rft.decls != null) rewriteValTypeAbsolute(ext_base, t) else t;
            break :blk r;
        },
        .named => |named| blk: {
            const r = try allocator.alloc(ctypes.ValType, named.len);
            for (named, 0..) |n, i| {
                r[i] = if (rft.decls != null) rewriteValTypeAbsolute(ext_base, n.type) else n.type;
            }
            break :blk r;
        },
    };
    errdefer allocator.free(results);

    ctx_ptr.* = .{
        .comp_inst = inst,
        .host_func = .{},
        .component_func_idx = lower.func_idx,
        .canon_lower_idx = canon_lower_idx,
        .param_types = params,
        .result_types = results,
        .lower_opts = executor_mod.LowerOptions.fromOpts(lower.opts),
        .extended_types = ext.extension_types,
        .extended_indexspace = ext.extension_indexspace,
        .is_async_func = ft.is_async,
    };
    ext_owned_by_ctx = true;

    try inst.trampoline_ctxs.append(allocator, ctx_ptr);
    return ctx_ptr;
}

/// Install a cross-instance thunk for an AOT core module's import whose
/// sibling source is a `canon.lower` contributor (the wit-component
/// adapter pattern). Builds a `ComponentTrampolineCtx` keyed off the
/// canon-lower's component-func index and allocates a pool slot that
/// dispatches via `wamrAotDispatchComponentTrampoline` →
/// `dispatchAotComponentTrampoline`. The dispatcher invokes the bound
/// `HostFunc` directly, so the standard WASIp2 host implementations work
/// without any further bridge.
///
/// The `HostFunc` is left empty here; `linkImports` later walks
/// `inst.trampoline_ctxs` and binds via `resolveComponentFuncToHostFunc`.
fn installCanonLowerBackedCrossInstanceThunk(
    allocator: std.mem.Allocator,
    inst: *ComponentInstance,
    component: *const ctypes.Component,
    canon_lower_idx: u32,
    imp: aot_loader.AotImportDesc,
    module: *const aot_loader.AotModule,
) !*const anyopaque {
    // Probe the pool first so failures on platforms that don't support
    // RWX trampoline pages (Windows, macOS aarch64, non-x86_64/aarch64
    // arches) bail before we allocate or publish the ctx — the caller
    // installs a trap stub instead. Doing this up front also keeps the
    // failure-cleanup path trivially correct.
    const pool = try ensureAotTrampolinePool(inst);

    const ctx_ptr = try buildLoweredComponentTrampolineCtx(allocator, inst, component, canon_lower_idx);
    // `buildLoweredComponentTrampolineCtx` publishes `ctx_ptr` on
    // `inst.trampoline_ctxs` as its final step, so a failure below has
    // to pop the entry AND release the ctx's owned memory (param/result
    // type slices + extension storage). `ComponentInstance.deinit` is
    // the only other place that does this teardown.
    errdefer {
        _ = inst.trampoline_ctxs.pop();
        ctx_ptr.deinit(allocator);
        allocator.destroy(ctx_ptr);
    }

    // Attempt to bind the HostFunc eagerly so calls that fire before
    // `linkImports` (e.g. AOT start-section that calls back through
    // a canon.lower import) still dispatch correctly. linkImports's
    // later sweep is idempotent.
    if (resolveComponentFuncToHostFunc(inst, component, ctx_ptr.component_func_idx)) |hf| {
        ctx_ptr.host_func = hf;
    }

    // Lowered (core-level) signature for the trampoline pool slot. We
    // take the AOT importer's view of the import's signature — that's
    // what the AOT codegen's call site emits, and it must match what
    // `dispatchAotComponentTrampoline` lifts off the register file.
    //
    // `has_retptr` must be derived from the component-level result
    // flatten count, not hardcoded: per the canonical ABI, results
    // whose joined flatten count exceeds `MAX_FLAT_RESULTS` (1 for
    // canon.lower) are returned via a caller-allocated buffer whose
    // pointer is appended as the last i32 param of the wasm core
    // signature. The AOT codegen emits exactly that shape, so the
    // last `ft.params` slot is the retptr — `lowered_params` must
    // omit it and `has_retptr` must be `true`. Without this fix every
    // canon.lower import returning a compound (`result<…>`, `list<…>`,
    // `option<…>`, multi-field record/tuple) lands in
    // `dispatchAotComponentTrampoline`'s
    // `reg_index != lp.len + has_retptr` check and rejects with
    // `UnsupportedSignature` — silently degraded by `genericDispatcher`
    // (`host_trampolines.zig:175`) to a `return 0`. Issue #707.
    if (imp.func_type_idx >= module.func_types.len) return error.InvalidFuncType;
    const ft = module.func_types[imp.func_type_idx];

    const canonical_abi = @import("canonical_abi.zig");
    const registry = if (ctx_ptr.extended_types.len > 0)
        canonical_abi.TypeRegistry.fromExtended(ctx_ptr.comp_inst.component, ctx_ptr.extended_types, ctx_ptr.extended_indexspace)
    else
        canonical_abi.TypeRegistry.init(ctx_ptr.comp_inst.component);
    var flat_result_count: u32 = 0;
    for (ctx_ptr.result_types) |rt| flat_result_count += canonical_abi.flattenCount(registry, rt);
    const is_async_lower = ctx_ptr.lower_opts.is_async or ctx_ptr.is_async_func;
    // Async canon.lower returns an i32 status word and carries every
    // non-empty lifted result through an explicit trailing retptr,
    // including a single-flat-slot result. Sync canon.lower only needs
    // that trailing pointer when its joined result flattening spills.
    const has_retptr = if (is_async_lower)
        ctx_ptr.result_types.len > 0
    else
        ctx_ptr.result_types.len > 0 and flat_result_count > canonical_abi.MAX_FLAT_RESULTS;

    const arena = inst.module_arena.allocator();
    const lowered_param_len: usize = if (has_retptr) blk: {
        if (ft.params.len == 0) return error.InvalidFuncType;
        break :blk ft.params.len - 1;
    } else ft.params.len;
    const lowered_params = try arena.alloc(core_types.ValType, lowered_param_len);
    for (ft.params[0..lowered_param_len], 0..) |p, i| lowered_params[i] = p;
    const lowered_results = try arena.alloc(core_types.ValType, ft.results.len);
    for (ft.results, 0..) |r, i| lowered_results[i] = r;

    const stub = try pool.allocCanonLowerAotSlot(@ptrCast(ctx_ptr), .{
        .param_types = lowered_params,
        .result_types = lowered_results,
        .has_retptr = has_retptr,
    });
    if (core_backend.debugAotEnabled()) {
        const realloc_dbg: ?u32 = if (ctx_ptr.lower_opts.realloc_idx) |r| r.value() else null;
        std.debug.print(
            "[install canon-lower(aot)] slot={d} module='{s}' field='{s}' cfi={d} flat_results={d} has_retptr={} mem_opt={?} realloc_opt={?}\n",
            .{ pool.next_slot - 1, imp.module_name, imp.field_name, ctx_ptr.component_func_idx, flat_result_count, has_retptr, ctx_ptr.lower_opts.memory_idx, realloc_dbg },
        );
    }
    return @ptrCast(stub);
}

/// Bridge an AOT core module's import that resolves through a sibling
/// inline-export to a canon-builtin contributor. Unlike a cross-instance
/// core export, the source is a component-runtime canonical definition, so
/// the slot must dispatch through `CanonBuiltinTrampolineCtx` rather than
/// `resolveCoreInstanceFuncToAi`.
fn installCanonBuiltinBackedCrossInstanceThunk(
    allocator: std.mem.Allocator,
    inst: *ComponentInstance,
    component: *const ctypes.Component,
    canon_idx: u32,
    imp: aot_loader.AotImportDesc,
    module: *const aot_loader.AotModule,
) !*const anyopaque {
    if (canon_idx >= component.canons.len) return error.InvalidCanonIdx;
    const canon = component.canons[canon_idx];
    switch (canon) {
        .context_get,
        .context_set,
        .task_yield,
        .task_return,
        .resource_drop,
        .resource_new,
        .resource_rep,
        .async_canon,
        => {},
        else => return error.UnsupportedCanonKind,
    }

    // The native stub forwards nine wasm-level C-ABI slots after the
    // importer's vmctx. Multi-value results consume one of those slots for
    // the hidden return pointer. Reject wider or non-scalar source types
    // explicitly rather than accidentally treating a canonical builtin as a
    // sibling AOT function and installing a trap stub.
    if (imp.func_type_idx >= module.func_types.len) return error.InvalidFuncType;
    const ft = module.func_types[imp.func_type_idx];
    const has_retptr = ft.results.len > 1;
    if (ft.params.len + @intFromBool(has_retptr) > 9) return error.SignatureTooWide;
    if (ft.results.len > 10) return error.MultipleResultsUnsupported;
    for (ft.params) |ty| switch (ty) {
        .i32, .i64, .f32, .f64 => {},
        else => return error.UnsupportedParamType,
    };
    for (ft.results) |ty| switch (ty) {
        .i32, .i64, .f32, .f64 => {},
        else => return error.UnsupportedResultType,
    };

    // Probe the pool first so failures on platforms without RWX trampoline
    // pages bail before we publish a new ctx — the caller installs a trap
    // stub instead.
    const pool = try ensureAotTrampolinePool(inst);

    // Reuse / build the per-canon-def-id ctx. Same memoisation map the
    // interp `linkImports` sweep at instance.zig:1939 uses, so a single
    // canon-def shared by interp and AOT imports points at one ctx.
    const ctx_ptr = ctx_blk: {
        if (inst.canon_builtin_ctx_by_canon_idx.get(canon_idx)) |existing| {
            break :ctx_blk existing;
        }
        const new_ctx = try allocator.create(executor_mod.CanonBuiltinTrampolineCtx);
        errdefer allocator.destroy(new_ctx);
        new_ctx.* = .{
            .comp_inst = inst,
            .canon = canon,
        };
        try inst.canon_builtin_ctxs.append(allocator, new_ctx);
        errdefer _ = inst.canon_builtin_ctxs.pop();
        // Failing to record the memoisation entry is non-fatal — the
        // canon_builtin_ctxs list still owns the allocation, so it gets
        // freed on `deinit`. Subsequent duplicate slots will allocate
        // their own ctx (graceful degradation under OOM).
        inst.canon_builtin_ctx_by_canon_idx.put(allocator, canon_idx, new_ctx) catch {};
        break :ctx_blk new_ctx;
    };
    // `task.return` must drain the importing core's declared flat result
    // slots, not a best-effort reconstruction from component type metadata.
    // This is shared with the interp link path and is intentionally
    // first-writer-wins for a canon def reused by multiple imports.
    if (ctx_ptr.core_flat_param_count == null) {
        ctx_ptr.core_flat_param_count = @intCast(ft.params.len);
    }

    const arena = inst.module_arena.allocator();
    const lowered_params = try arena.alloc(core_types.ValType, ft.params.len);
    for (ft.params, 0..) |p, i| lowered_params[i] = p;
    const lowered_results = try arena.alloc(core_types.ValType, ft.results.len);
    for (ft.results, 0..) |r, i| lowered_results[i] = r;

    const stub = try pool.allocCanonBuiltinAotSlot(@ptrCast(ctx_ptr), .{
        .param_types = lowered_params,
        .result_types = lowered_results,
        .has_retptr = has_retptr,
    });
    return @ptrCast(stub);
}

fn installTrapStub(
    allocator: std.mem.Allocator,
    inst: *ComponentInstance,
    imp: aot_loader.AotImportDesc,
    module_idx: u32,
) !*const anyopaque {
    // Allocate a null-terminated label so the dispatcher can `printf` it
    // under WAMR_AOT_DEBUG without copying.
    const len = imp.module_name.len + 1 + imp.field_name.len;
    const buf = try allocator.alloc(u8, len + 1);
    errdefer allocator.free(buf);
    @memcpy(buf[0..imp.module_name.len], imp.module_name);
    buf[imp.module_name.len] = '.';
    @memcpy(buf[imp.module_name.len + 1 ..][0..imp.field_name.len], imp.field_name);
    buf[len] = 0;

    const label_z: [*:0]const u8 = @ptrCast(buf.ptr);
    try inst.trap_stub_labels.append(allocator, label_z);
    errdefer _ = inst.trap_stub_labels.pop();

    if (core_backend.debugAotEnabled()) {
        std.debug.print(
            "[aot-bridge] core module {d}: installing trap-on-call stub for un-bridged import '{s}.{s}' (#662 follow-up)\n",
            .{ module_idx, imp.module_name, imp.field_name },
        );
    }

    const pool = try ensureAotTrampolinePool(inst);
    // The trap dispatcher reads `ctx_opaque` as a `*const [*:0]const u8`
    // when debug logging is enabled, so we hand it a pointer to the
    // owned label entry in the ArrayList (stable because the list never
    // shrinks before `deinit`).
    const slot_ptr = &inst.trap_stub_labels.items[inst.trap_stub_labels.items.len - 1];
    const stub = try pool.allocTrapSlot(@ptrCast(slot_ptr));
    return @ptrCast(stub);
}

fn resolveCoreGlobalToMI(
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    core_glob_idx: u32,
) ?*core_types.GlobalInstance {
    const ref = indexspace.resolveCoreGlobal(component, core_glob_idx) orelse return null;
    const ie = component.aliases[ref.aliased].instance_export;
    if (ie.instance_idx >= inst.core_instances.len) return null;
    return resolveCoreInstanceGlobalExport(inst, component, inst.core_instances[ie.instance_idx], ie.name);
}

/// Walk `component.type_indexspace[idx]` (loader-populated) to find the
/// local entry in `component.types`. Falls back to direct indexing for
/// hand-authored fixtures that bypass the loader.
fn resolveTypeDef(component: *const ctypes.Component, type_idx: u32) ?ctypes.TypeDef {
    if (component.type_indexspace.len > 0) {
        if (type_idx >= component.type_indexspace.len) return null;
        const local = component.type_indexspace[type_idx] orelse return null;
        if (local >= component.types.len) return null;
        return component.types[local];
    }
    if (type_idx >= component.types.len) return null;
    return component.types[type_idx];
}

/// Resolve a component-func index to its function type. Handles imports,
/// aliases of imported-instance members, and `canon.lift` entries.
/// Inside an instance-type body, type indices are local. This helper
/// walks decls in order, resolving the Nth type-producing declarator
/// to its concrete `TypeDef`. Used by the canonical-ABI trampoline to
/// dereference param/result `.type_idx` references.
fn resolveInstanceTypeLocal(decls: []const ctypes.Decl, idx: u32) ?ctypes.TypeDef {
    var n: u32 = 0;
    for (decls) |d| switch (d) {
        .type => |td| {
            if (n == idx) return td;
            n += 1;
        },
        .alias => {
            // Aliases of types produce a binding but we don't track the
            // outer-resolved type here; return null to surface a fallback.
            if (n == idx) return null;
            n += 1;
        },
        .@"export" => |e| {
            if (e.desc == .type) {
                // Resource exports / type-eq exports — caller treats as
                // an opaque resource handle.
                if (n == idx) return null;
                n += 1;
            }
        },
        else => {},
    };
    return null;
}

/// Lower a possibly-local `ValType` referring into an instance-type body
/// to a `ValType` that the trampoline can lower without a TypeRegistry.
/// Superseded by `rewriteValTypeAbsolute` + `buildInstanceTypeExtension`
/// for the canon.lower trampoline path; kept for tests / external callers.
fn rewriteInstanceTypeValType(decls: []const ctypes.Decl, vt: ctypes.ValType) ctypes.ValType {
    _ = decls;
    return vt;
}

/// Rewrite an instance-type-local `ValType` so that any local type index
/// becomes an absolute index in the per-trampoline extended indexspace.
///
/// `base` is the offset where the extension starts in the absolute
/// indexspace (i.e. `component.type_indexspace.len`, or `types.len` when
/// the component has no indexspace).
///
/// Resource handle indices (`.own` / `.borrow`) carry resource identity
/// rather than structural type info and are left unchanged.
fn rewriteValTypeAbsolute(base: u32, vt: ctypes.ValType) ctypes.ValType {
    return switch (vt) {
        // Resource identity: do not rewrite.
        .own, .borrow => vt,
        // Structural compound refs: rebase index into extension.
        .record => |i| .{ .record = base + i },
        .variant => |i| .{ .variant = base + i },
        .list => |i| .{ .list = base + i },
        .tuple => |i| .{ .tuple = base + i },
        .flags => |i| .{ .flags = base + i },
        .enum_ => |i| .{ .enum_ = base + i },
        .option => |i| .{ .option = base + i },
        .result => |i| .{ .result = base + i },
        .type_idx => |i| .{ .type_idx = base + i },
        else => vt,
    };
}

/// Deep-copy a `TypeDef` from an instance-type body, rewriting all nested
/// `ValType` references through `rewriteValTypeAbsolute`. Allocations are
/// owned by `allocator` and freed when the trampoline ctx tears down.
fn rewriteTypeDefAbsolute(
    allocator: std.mem.Allocator,
    base: u32,
    td: ctypes.TypeDef,
) !ctypes.TypeDef {
    return switch (td) {
        .val => |v| .{ .val = rewriteValTypeAbsolute(base, v) },
        .list => |l| .{ .list = .{ .element = rewriteValTypeAbsolute(base, l.element) } },
        .option => |o| .{ .option = .{ .inner = rewriteValTypeAbsolute(base, o.inner) } },
        .result => |r| .{ .result = .{
            .ok = if (r.ok) |ok| rewriteValTypeAbsolute(base, ok) else null,
            .err = if (r.err) |er| rewriteValTypeAbsolute(base, er) else null,
        } },
        .record => |rec| blk: {
            const new_fields = try allocator.alloc(ctypes.Field, rec.fields.len);
            for (rec.fields, 0..) |f, i| {
                new_fields[i] = .{ .name = f.name, .type = rewriteValTypeAbsolute(base, f.type) };
            }
            break :blk .{ .record = .{ .fields = new_fields } };
        },
        .tuple => |tup| blk: {
            const new_fields = try allocator.alloc(ctypes.ValType, tup.fields.len);
            for (tup.fields, 0..) |f, i| new_fields[i] = rewriteValTypeAbsolute(base, f);
            break :blk .{ .tuple = .{ .fields = new_fields } };
        },
        .variant => |v| blk: {
            const new_cases = try allocator.alloc(ctypes.Case, v.cases.len);
            for (v.cases, 0..) |c, i| {
                new_cases[i] = .{
                    .name = c.name,
                    .type = if (c.type) |ct| rewriteValTypeAbsolute(base, ct) else null,
                    .refines = c.refines,
                };
            }
            break :blk .{ .variant = .{ .cases = new_cases } };
        },
        // Primitives + `.flags` / `.enum_` / `.resource` / func / component / instance
        // carry no nested ValType refs that need rewriting at this layer.
        else => td,
    };
}

const InstanceTypeExtension = struct {
    extension_types: []const ctypes.TypeDef,
    extension_indexspace: []const ?u32,

    pub fn empty() InstanceTypeExtension {
        return .{ .extension_types = &.{}, .extension_indexspace = &.{} };
    }

    pub fn deinit(self: InstanceTypeExtension, allocator: std.mem.Allocator, deep: bool) void {
        if (deep) {
            // Free the per-typedef allocations made by rewriteTypeDefAbsolute.
            for (self.extension_types) |td| switch (td) {
                .record => |rec| allocator.free(rec.fields),
                .tuple => |tup| allocator.free(tup.fields),
                .variant => |v| allocator.free(v.cases),
                else => {},
            };
        }
        if (self.extension_types.len > 0) allocator.free(self.extension_types);
        if (self.extension_indexspace.len > 0) allocator.free(self.extension_indexspace);
    }
};

/// True iff `a` is a single-hop outer alias of sort `.type` whose
/// parent target slot has been resolved (i.e. `type_indexspace[idx]`
/// is non-null). Used to size the extension's type buffer for #534.
fn canResolveOuterTypeAlias(a: ctypes.Alias, component: *const ctypes.Component) bool {
    return resolveOuterTypeAliasToParent(a, component) != null;
}

/// Resolve a single-hop `.alias outer 1 M (type)` against the parent
/// component's `type_indexspace`. Returns the parent's `TypeDef` or
/// `null` when the alias shape is unsupported (multi-level outer,
/// non-type sort, target slot still unresolved).
///
/// `#534` originally limited the return shape to `TypeDef.val` because
/// that covers `type duration = u64` (the clock fixtures). `#571`
/// (filesystem-stat) requires the same path for `.record` / `.variant`
/// / `.tuple` / `.flags` / `.enum_` / `.option` / `.result` / `.list`
/// because wit-bindgen emits the `wasi:filesystem/types` instance-type
/// body with `(alias outer 1 M (type))` decls pointing at top-level
/// records (`datetime`) and variants (`new-timestamp` references
/// `datetime` via this alias). The returned `TypeDef` is structurally
/// shared with the parent — `buildInstanceTypeExtension` deep-copies
/// it through `rewriteTypeDefAbsolute(allocator, 0, ...)` so the
/// extension owns its own field/case/tuple slice and `deinit` can free
/// safely without double-freeing the parent's slices.
fn resolveOuterTypeAliasToParent(
    a: ctypes.Alias,
    component: *const ctypes.Component,
) ?ctypes.TypeDef {
    const outer = switch (a) {
        .outer => |o| o,
        else => return null,
    };
    if (outer.sort != .type) return null;
    // Only single-hop outers — i.e. the immediate enclosing component.
    if (outer.outer_count != 1) return null;
    if (component.type_indexspace.len == 0) return null;
    if (outer.idx >= component.type_indexspace.len) return null;
    const local = component.type_indexspace[outer.idx] orelse return null;
    if (local >= component.types.len) return null;
    const td = component.types[local];
    return switch (td) {
        .val, .record, .variant, .tuple, .flags, .enum_, .option, .result, .list => td,
        else => null,
    };
}

/// Materialize the per-trampoline TypeRegistry extension covering an
/// instance-type body's local type space. Walks `decls` in declaration
/// order, mirroring `resolveInstanceTypeLocal`'s slot-counting rules:
/// `.type`, `.alias`, and `.@"export"`-with-type each contribute one
/// indexspace slot. `.type` slots materialize a structural typedef;
/// `.alias outer 1 M (type)` slots resolve through the parent
/// component's `type_indexspace` (#534); other alias shapes map to
/// `null` (the trampoline path for them was already null-fallback
/// under the prior local-only walker).
///
/// The caller is responsible for `deinit`'ing the returned extension.
fn buildInstanceTypeExtension(
    allocator: std.mem.Allocator,
    decls: []const ctypes.Decl,
    base: u32,
    component: *const ctypes.Component,
) !InstanceTypeExtension {
    // First pass: count slots and type entries. Reserve a type slot for
    // `(alias outer 1 M (type))` decls too — we materialize the parent
    // type into the extension when possible (#534).
    var slot_count: u32 = 0;
    var type_count: u32 = 0;
    for (decls) |d| switch (d) {
        .type => {
            slot_count += 1;
            type_count += 1;
        },
        .alias => |a| {
            slot_count += 1;
            // `.alias outer count=1 idx=M (type)` may resolve to a
            // parent type — reserve a type slot for the copy. Other
            // alias shapes (inner instance_export aliases, deeper outer
            // hops) stay unresolved.
            if (canResolveOuterTypeAlias(a, component)) type_count += 1;
        },
        .@"export" => |e| if (e.desc == .type) {
            slot_count += 1;
        },
        else => {},
    };

    if (slot_count == 0) return InstanceTypeExtension.empty();

    const types_buf = try allocator.alloc(ctypes.TypeDef, type_count);
    errdefer allocator.free(types_buf);
    const idxspace_buf = try allocator.alloc(?u32, slot_count);
    errdefer allocator.free(idxspace_buf);

    var slot_i: u32 = 0;
    var type_i: u32 = 0;
    var rewrite_failed: bool = false;
    for (decls) |d| switch (d) {
        .type => |td| {
            const rewritten = rewriteTypeDefAbsolute(allocator, base, td) catch {
                rewrite_failed = true;
                break;
            };
            types_buf[type_i] = rewritten;
            idxspace_buf[slot_i] = type_i;
            type_i += 1;
            slot_i += 1;
        },
        .alias => |a| {
            // `.alias outer count=1 idx=M (type)` — when M points at a
            // parent type-indexspace slot that has been resolved (e.g.
            // by the loader's #534 top-level type-alias resolution),
            // materialize the parent's TypeDef in the extension so
            // canon-ABI lift/lower of values whose declared type goes
            // through this slot can look up the concrete shape.
            //
            // For `.record` / `.tuple` / `.variant` we deep-copy via
            // `rewriteTypeDefAbsolute(allocator, 0, ...)` so the
            // extension owns its own fields/cases slice (`deinit`
            // would otherwise double-free shared parent slices). The
            // `base=0` argument leaves nested `.type_idx` refs at the
            // parent's absolute index, which is correct: those refs
            // are < `ext_base`, so the registry's `get()` falls into
            // the parent-component lookup path. (`#571`.)
            //
            // For leaf shapes (`.val` / `.flags` / `.enum_` / `.list`
            // / `.option` / `.result`) deinit doesn't free heap
            // payload, so shallow sharing is safe.
            if (resolveOuterTypeAliasToParent(a, component)) |parent_td| {
                const copied: ctypes.TypeDef = switch (parent_td) {
                    .record, .tuple, .variant => rewriteTypeDefAbsolute(allocator, 0, parent_td) catch {
                        rewrite_failed = true;
                        break;
                    },
                    else => parent_td,
                };
                types_buf[type_i] = copied;
                idxspace_buf[slot_i] = type_i;
                type_i += 1;
            } else {
                idxspace_buf[slot_i] = null;
            }
            slot_i += 1;
        },
        .@"export" => |e| if (e.desc == .type) {
            // `(export "X" (type ...))` adds a new type-indexspace slot
            // whose semantics depend on the bound:
            //   * `.eq{N}`: the new slot is alias-equal to local slot N.
            //     Per the component-model spec (Binary.md `exportdecl`,
            //     Explainer.md "Type definitions"), references to this
            //     slot must resolve to whatever N resolves to. Without
            //     this, canon-ABI lower/lift of values whose declared
            //     type goes through such an export trips
            //     `CompoundNeedsRegistry` (issue #310, e.g. TinyGo
            //     `_initialize` calling `wasi:clocks/wall-clock.now`
            //     whose result type is `(export "datetime" (type (eq 0)))`).
            //   * `.sub_resource`: introduces a fresh resource type.
            //     Resource handles (`.own`/`.borrow`) push i32 directly
            //     without consulting the registry, so leaving this null
            //     is correct.
            switch (e.desc.type) {
                .eq => |target| {
                    // The .eq target must be a prior slot (type indexspace
                    // is built in declaration order). If the target slot
                    // itself was unresolved (.alias / .sub_resource), we
                    // inherit null transitively — the right answer.
                    std.debug.assert(target < slot_i);
                    idxspace_buf[slot_i] = idxspace_buf[target];
                },
                .sub_resource => idxspace_buf[slot_i] = null,
            }
            slot_i += 1;
        },
        else => {},
    };

    if (rewrite_failed) {
        // Free any nested allocations made before the failure.
        var i: u32 = 0;
        while (i < type_i) : (i += 1) switch (types_buf[i]) {
            .record => |rec| allocator.free(rec.fields),
            .tuple => |tup| allocator.free(tup.fields),
            .variant => |v| allocator.free(v.cases),
            else => {},
        };
        allocator.free(types_buf);
        allocator.free(idxspace_buf);
        return error.OutOfMemory;
    }

    return .{
        .extension_types = types_buf,
        .extension_indexspace = idxspace_buf,
    };
}

/// Same as `rewriteInstanceTypeValType` but rewrites the params and
/// results of a `FuncType` and allocates a fresh slice via `allocator`.
/// (Superseded by `buildInstanceTypeExtension` + `rewriteValTypeAbsolute`.
/// Retained for any callers outside the canon.lower trampoline path.)
fn rewriteInstanceFuncType(
    allocator: std.mem.Allocator,
    decls: []const ctypes.Decl,
    ft: ctypes.FuncType,
) !ctypes.FuncType {
    _ = allocator;
    _ = decls;
    return ft;
}

const ResolvedFuncType = struct {
    ft: ctypes.FuncType,
    /// When the FuncType came from an instance-type body, its
    /// param/result `.type_idx` references are local to that body's
    /// type space. Callers must rewrite via `rewriteInstanceTypeValType`
    /// using this `decls` list before consuming the types.
    decls: ?[]const ctypes.Decl = null,
};

fn resolveCompFuncType(component: *const ctypes.Component, func_idx: u32) ?ResolvedFuncType {
    const ref = indexspace.resolveCompFunc(component, func_idx) orelse {
        return null;
    };
    switch (ref) {
        .imported => |imp_idx| {
            const imp = component.imports[imp_idx];
            const tidx = switch (imp.desc) {
                .func => |t| t,
                else => return null,
            };
            const td = resolveTypeDef(component, tidx) orelse return null;
            return switch (td) {
                .func => |ft| .{ .ft = ft },
                else => null,
            };
        },
        .aliased => |alias_idx| {
            const ie = component.aliases[alias_idx].instance_export;
            const inst_ref = indexspace.resolveCompInstance(component, ie.instance_idx) orelse {
                return null;
            };
            const inst_type_idx: u32 = switch (inst_ref) {
                .imported => |i| switch (component.imports[i].desc) {
                    .instance => |t| t,
                    else => return null,
                },
                else => {
                    return null;
                },
            };
            const inst_td = resolveTypeDef(component, inst_type_idx) orelse {
                return null;
            };
            const decls = switch (inst_td) {
                .instance => |it| it.decls,
                else => return null,
            };
            for (decls) |d| switch (d) {
                .@"export" => |e| {
                    if (!std.mem.eql(u8, e.name, ie.name)) continue;
                    const tidx = switch (e.desc) {
                        .func => |t| t,
                        else => return null,
                    };
                    var n: u32 = 0;
                    var found: ?ctypes.FuncType = null;
                    for (decls) |d2| {
                        switch (d2) {
                            .type => |td2| {
                                if (n == tidx) {
                                    found = switch (td2) {
                                        .func => |ft| ft,
                                        else => null,
                                    };
                                }
                                n += 1;
                            },
                            .alias => n += 1,
                            .@"export" => |e2| {
                                if (e2.desc == .type) n += 1;
                            },
                            else => {},
                        }
                        if (found != null) break;
                    }
                    if (found) |ft| return .{ .ft = ft, .decls = decls };
                    return null;
                },
                else => {},
            };
            return null;
        },
        .lifted => |canon_idx| {
            const lift = switch (component.canons[canon_idx]) {
                .lift => |l| l,
                else => return null,
            };
            const td = resolveTypeDef(component, lift.type_idx) orelse return null;
            return switch (td) {
                .func => |ft| .{ .ft = ft },
                else => null,
            };
        },
    }
}

/// Resolve a `canon.lift.core_func_idx` (component core-func-index-space) to
/// the (core_instance_idx, local_func_idx) pair where the function actually
/// lives, so the executor can call it via `interp.executeFunction` with the
/// module-local index.
///
/// Phase 2A.2c layout assumption: the core-func-index-space is built up by
/// (in this order) all `canon.lower` entries, then all `Alias.instance_export`
/// entries with `sort = .core(.func)`. A full section-order-aware resolver
/// will replace this once the loader emits ordered per-index-space streams.
fn resolveLiftedCoreFunc(
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    core_func_idx: u32,
) ?struct { core_instance_idx: u32, local_func_idx: u32 } {
    const ref = indexspace.resolveCoreFunc(component, core_func_idx) orelse return null;
    switch (ref) {
        // Canon entries (lowers, resource.{new,drop,rep}, async builtins)
        // all produce imports/host-bound core funcs — not exported
        // callables — so a canon.lift pointing at one is malformed.
        .lowered,
        .resource_drop,
        .resource_new,
        .resource_rep,
        .task_yield,
        .context_get,
        .context_set,
        .task_return,
        .async_canon,
        => return null,
        .aliased => |alias_idx| {
            const a = component.aliases[alias_idx];
            const ie = a.instance_export;
            if (ie.instance_idx >= inst.core_instances.len) return null;
            const target = inst.core_instances[ie.instance_idx];
            if (target.module_inst) |mi| {
                const local = mi.getExportFunc(ie.name) orelse return null;
                return .{
                    .core_instance_idx = ie.instance_idx,
                    .local_func_idx = local,
                };
            }
            if (target.aot_inst) |ai| {
                const exp = ai.module.findExport(ie.name, .function) orelse return null;
                return .{
                    .core_instance_idx = ie.instance_idx,
                    .local_func_idx = exp.index,
                };
            }
            return null;
        },
    }
}

/// Resolve a component-level func index to a bound `HostFunc`.
///
/// The component-level func index space is contributed to by, in section
/// order: `import (kind=func)` decls, then component-level aliases of
/// `Sort.func` exports of imported component instances, then canon.lifts,
/// and so on. Phase 2B narrow assumption: the prefix we care about is
/// `imports (kind=func)` followed by `aliases (Sort.func, instance_export)`
/// pointing into imported component instances. canon.lifts come later in
/// the index space and are not host-bound.
fn resolveComponentFuncToHostFunc(
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    func_idx: u32,
) ?HostFunc {
    const ref = indexspace.resolveCompFunc(component, func_idx) orelse return null;
    switch (ref) {
        .imported => |imp_idx| {
            const imp = component.imports[imp_idx];
            const binding = inst.imports.get(imp.name) orelse return null;
            return switch (binding) {
                .host_func => |hf| hf,
                else => null,
            };
        },
        .aliased => |alias_idx| {
            const ie = component.aliases[alias_idx].instance_export;
            const inst_ref = indexspace.resolveCompInstance(component, ie.instance_idx) orelse return null;
            // Only aliases pointing at imported component instances are
            // host-bound. Aliases of locally-instantiated instances are
            // resolved to their underlying canon.lift / canon.lower elsewhere.
            const imp_decl = switch (inst_ref) {
                .imported => |i| component.imports[i],
                else => return null,
            };
            const binding = inst.imports.get(imp_decl.name) orelse return null;
            const host_inst = switch (binding) {
                .host_instance => |hi| hi,
                else => return null,
            };
            const member = host_inst.members.get(ie.name) orelse return null;
            return switch (member) {
                .func => |hf| hf,
                .resource_type => null,
            };
        },
        // canon.lifts are not host-bound — they are component-defined
        // funcs implemented by core code.
        .lifted => return null,
    }
}

/// Resolve a component-level instance index to its `ImportDecl` if it
/// refers to an imported component instance. Returns null for locally
/// instantiated or aliased instances (callers wanting the broader form
/// should use `indexspace.resolveCompInstance` directly).
fn resolveImportedInstance(
    component: *const ctypes.Component,
    instance_idx: u32,
) ?ctypes.ImportDecl {
    const ref = indexspace.resolveCompInstance(component, instance_idx) orelse return null;
    return switch (ref) {
        .imported => |i| component.imports[i],
        else => null,
    };
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "ResourceTable: new and rep" {
    const allocator = std.testing.allocator;
    var rt = ResourceTable{};
    defer rt.deinit(allocator);

    const h0 = try rt.new(42, true, allocator);
    const h1 = try rt.new(99, true, allocator);

    try std.testing.expectEqual(@as(u32, 0), h0);
    try std.testing.expectEqual(@as(u32, 1), h1);
    try std.testing.expectEqual(@as(?u32, 42), rt.rep(h0));
    try std.testing.expectEqual(@as(?u32, 99), rt.rep(h1));
}

test "ResourceTable: drop and reuse" {
    const allocator = std.testing.allocator;
    var rt = ResourceTable{};
    defer rt.deinit(allocator);

    const h0 = try rt.new(10, true, allocator);
    const dropped_rep = rt.drop(h0, allocator);
    try std.testing.expectEqual(@as(?u32, 10), dropped_rep);
    try std.testing.expectEqual(@as(?u32, null), rt.rep(h0));

    // Reuse the slot
    const h2 = try rt.new(20, true, allocator);
    try std.testing.expectEqual(h0, h2); // reused slot
    try std.testing.expectEqual(@as(?u32, 20), rt.rep(h2));
}

test "ResourceTable: borrow prevents drop" {
    const allocator = std.testing.allocator;
    var rt = ResourceTable{};
    defer rt.deinit(allocator);

    const h = try rt.new(55, true, allocator);
    try std.testing.expect(rt.borrow(h));
    try std.testing.expectEqual(@as(?u32, null), rt.drop(h, allocator)); // can't drop

    rt.returnBorrow(h);
    try std.testing.expectEqual(@as(?u32, 55), rt.drop(h, allocator)); // now it works
}

test "ResourceTable: double drop returns null" {
    const allocator = std.testing.allocator;
    var rt = ResourceTable{};
    defer rt.deinit(allocator);

    const h = try rt.new(77, true, allocator);
    _ = rt.drop(h, allocator);
    try std.testing.expectEqual(@as(?u32, null), rt.drop(h, allocator));
}

test "ImportBinding: host func creation" {
    const binding = ImportBinding{ .host_func = .{ .context = null } };
    try std.testing.expect(binding == .host_func);
}

test "ComponentInstance: linkImports resolves known imports" {
    const allocator = std.testing.allocator;

    // Create a minimal component with one import
    const imports = [_]ctypes.ImportDecl{
        .{ .name = "my-import", .desc = .{ .func = 0 } },
    };
    const component = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &imports,
        .exports = &.{},
    };

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    // Provide a binding for the import
    var providers: std.StringHashMapUnmanaged(ImportBinding) = .{};
    defer providers.deinit(allocator);
    try providers.put(allocator, "my-import", .{ .host_func = .{ .context = null } });

    try inst.linkImports(providers);

    // Verify the import was resolved
    const resolved = inst.getImport("my-import");
    try std.testing.expect(resolved != null);
    try std.testing.expect(resolved.? == .host_func);
}

test "ComponentInstance: getImport returns null for unknown" {
    const allocator = std.testing.allocator;

    const component = ctypes.Component{
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

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    try std.testing.expectEqual(@as(?ImportBinding, null), inst.getImport("nonexistent"));
}

test "ComponentInstance: executeStart is idempotent" {
    const allocator = std.testing.allocator;

    // Component with no start function — executeStart should be a no-op
    const component = ctypes.Component{
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

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    try inst.executeStart(); // first call
    try std.testing.expect(inst.started);
    try inst.executeStart(); // second call — should be idempotent
    try std.testing.expect(inst.started);
}

test "linkImports: missing runtime import returns MissingImport" {
    const allocator = std.testing.allocator;

    const imports = [_]ctypes.ImportDecl{
        .{ .name = "my-func", .desc = .{ .func = 0 } },
    };
    const component = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &imports,
        .exports = &.{},
    };

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .{};
    defer providers.deinit(allocator);
    try std.testing.expectError(error.MissingImport, inst.linkImports(providers));
}

test "linkImports: kind mismatch returns ImportKindMismatch" {
    const allocator = std.testing.allocator;

    // Instance-typed import must be satisfied with host_instance, not host_func.
    const imports = [_]ctypes.ImportDecl{
        .{ .name = "wasi:io/streams", .desc = .{ .instance = 0 } },
    };
    const component = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &imports,
        .exports = &.{},
    };

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .{};
    defer providers.deinit(allocator);
    try providers.put(allocator, "wasi:io/streams", .{ .host_func = .{} });
    try std.testing.expectError(error.ImportKindMismatch, inst.linkImports(providers));
}

test "linkImports: host_instance binding satisfies instance import" {
    const allocator = std.testing.allocator;

    const imports = [_]ctypes.ImportDecl{
        .{ .name = "wasi:io/streams", .desc = .{ .instance = 0 } },
    };
    const component = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &imports,
        .exports = &.{},
    };

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    var host: HostInstance = .{};
    defer host.deinit(allocator);

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .{};
    defer providers.deinit(allocator);
    try providers.put(allocator, "wasi:io/streams", .{ .host_instance = &host });

    try inst.linkImports(providers);
    const resolved = inst.getImport("wasi:io/streams") orelse return error.TestUnexpectedResult;
    try std.testing.expect(resolved == .host_instance);
    try std.testing.expectEqual(&host, resolved.host_instance);
}

test "linkImports: type import needs no binding" {
    const allocator = std.testing.allocator;

    const imports = [_]ctypes.ImportDecl{
        .{ .name = "T", .desc = .{ .type = .sub_resource } },
    };
    const component = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &imports,
        .exports = &.{},
    };

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .{};
    defer providers.deinit(allocator);
    try inst.linkImports(providers); // no error despite empty providers
}

test "ComponentInstance: resource tables are lazy and keyed by typeidx" {
    const allocator = std.testing.allocator;

    const component = ctypes.Component{
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

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    // No tables allocated up front.
    try std.testing.expectEqual(@as(u32, 0), inst.resource_tables.count());

    // Sparse resource type indices are fine.
    const rt_a = try inst.getOrCreateResourceTable(5);
    const rt_b = try inst.getOrCreateResourceTable(42);
    try std.testing.expect(rt_a != rt_b);
    try std.testing.expectEqual(@as(u32, 2), inst.resource_tables.count());

    // Repeated access returns the same table.
    const rt_a2 = try inst.getOrCreateResourceTable(5);
    try std.testing.expectEqual(rt_a, rt_a2);
}

test "instantiate: canon.lower wires host func into core import (2A.2b)" {

    // Minimal core module:
    //   (type (func (param i32 i32) (result i32)))
    //   (type (func (result i32)))
    //   (import "host" "sub" (func $sub (type 0)))
    //   (func $run (type 1) i32.const 7 i32.const 2 call $sub)
    //   (export "run" (func $run))
    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section
        0x01, 0x0b, 0x02, 0x60, 0x02, 0x7f, 0x7f, 0x01,
        0x7f, 0x60, 0x00, 0x01, 0x7f,
        // import section: host.sub (func type 0)
        0x02, 0x0c, 0x01,
        0x04, 'h',  'o',  's',  't',  0x03, 's',  'u',
        'b',  0x00, 0x00,
        // function section: 1 local fn, type 1
        0x03, 0x02, 0x01, 0x01,
        // export section: "run" -> func 1
        0x07,
        0x07, 0x01, 0x03, 'r',  'u',  'n',  0x00, 0x01,
        // code section
        0x0a, 0x0a, 0x01, 0x08, 0x00,
        0x41, 0x07, // i32.const 7
        0x41, 0x02, // i32.const 2
        0x10, 0x00, // call 0 (imported sub)
        0x0b, // end
    };

    const core_modules = [_]ctypes.CoreModule{.{ .data = &core_wasm }};

    // Component func type 0: (s32, s32) -> s32
    const params = [_]ctypes.NamedValType{
        .{ .name = "a", .type = .s32 },
        .{ .name = "b", .type = .s32 },
    };
    const type_defs = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &params, .results = .{ .unnamed = .s32 } } },
    };
    const imports_decl = [_]ctypes.ImportDecl{
        .{ .name = "host-sub", .desc = .{ .func = 0 } },
    };
    const canons = [_]ctypes.Canon{
        .{ .lower = .{ .func_idx = 0, .opts = &.{} } },
    };
    const inline_exports = [_]ctypes.CoreInlineExport{
        .{ .name = "sub", .sort_idx = .{ .sort = .func, .idx = 0 } },
    };
    const inst_args = [_]ctypes.CoreInstantiateArg{
        .{ .name = "host", .instance_idx = 0 },
    };
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .exports = &inline_exports },
        .{ .instantiate = .{ .module_idx = 0, .args = &inst_args } },
    };

    const component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &type_defs,
        .canons = &canons,
        .imports = &imports_decl,
        .exports = &.{},
    };

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    // Register a host sub: returns a - b. Non-commutative to catch argument
    // order reversal in the trampoline.
    const Host = struct {
        fn sub(
            _: ?*anyopaque,
            _: *ComponentInstance,
            in: []const InterfaceValue,
            out: []InterfaceValue,
            _: std.mem.Allocator,
        ) anyerror!void {
            out[0] = .{ .s32 = in[0].s32 - in[1].s32 };
        }
    };
    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    try providers.put(std.testing.allocator, "host-sub", .{
        .host_func = .{ .call = &Host.sub },
    });
    try inst.linkImports(providers);

    // After linkImports, trampolines should be wired to the host fn.
    try std.testing.expect(inst.core_instances.len == 2);
    const mi = inst.core_instances[1].module_inst orelse return error.TestFailed;
    try std.testing.expect(mi.host_func_entries.len >= 1);
    try std.testing.expect(inst.trampoline_ctxs.items.len == 1);
    try std.testing.expect(inst.trampoline_ctxs.items[0].host_func.call != null);

    // Invoke the exported "run" core function.
    const run_idx = mi.getExportFunc("run") orelse return error.TestFailed;
    const env = try @import("../runtime/common/exec_env.zig").ExecEnv.create(mi, 512, std.testing.allocator);
    defer env.destroy();
    try @import("../runtime/interpreter/interp.zig").executeFunction(env, run_idx);
    try std.testing.expectEqual(@as(i32, 5), try env.popI32());
}

test "callComponentFunc: invokes lifted export through alias (2A.2c)" {
    const executor = @import("executor.zig");
    const abi_mod = @import("canonical_abi.zig");

    // Same shape as the 2A.2b fixture, but `run` takes the args instead of
    // hard-coding constants. A canon.lift then exposes it as a component
    // export, and callComponentFunc invokes it through the lift.
    //
    //   (module
    //     (type (func (param i32 i32) (result i32)))
    //     (import "host" "sub" (func (type 0)))
    //     (func (type 0)
    //       local.get 0 local.get 1 call 0)
    //     (export "run" (func 1)))
    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section (1 type)
        0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01,
        0x7f,
        // import section (1 import)
        0x02, 0x0c, 0x01, 0x04, 'h',  'o',  's',
        't',  0x03, 's',  'u',  'b',  0x00, 0x00,
        // function section (1 local fn)
        0x03,
        0x02, 0x01, 0x00,
        // export section: "run" -> func 1
        0x07, 0x07, 0x01, 0x03, 'r',
        'u',  'n',  0x00, 0x01,
        // code section (1 body, 8 bytes)
        0x0a, 0x0a, 0x01, 0x08,
        0x00, 0x20, 0x00, 0x20, 0x01, 0x10, 0x00, 0x0b,
    };

    const core_modules = [_]ctypes.CoreModule{.{ .data = &core_wasm }};

    const params = [_]ctypes.NamedValType{
        .{ .name = "a", .type = .s32 },
        .{ .name = "b", .type = .s32 },
    };
    const type_defs = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &params, .results = .{ .unnamed = .s32 } } },
    };
    const imports_decl = [_]ctypes.ImportDecl{
        .{ .name = "host-sub", .desc = .{ .func = 0 } },
    };
    // Canon order: lower first (core-func 0), then lift (component func 1).
    const canons = [_]ctypes.Canon{
        .{ .lower = .{ .func_idx = 0, .opts = &.{} } },
        .{ .lift = .{ .core_func_idx = 1, .type_idx = 0, .opts = &.{} } },
    };
    const inline_exports = [_]ctypes.CoreInlineExport{
        .{ .name = "sub", .sort_idx = .{ .sort = .func, .idx = 0 } },
    };
    const inst_args = [_]ctypes.CoreInstantiateArg{
        .{ .name = "host", .instance_idx = 0 },
    };
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .exports = &inline_exports },
        .{ .instantiate = .{ .module_idx = 0, .args = &inst_args } },
    };
    const aliases_decl = [_]ctypes.Alias{
        .{ .instance_export = .{
            .sort = .{ .core = .func },
            .instance_idx = 1,
            .name = "run",
        } },
    };
    const exports_decl = [_]ctypes.ExportDecl{
        .{ .name = "run", .desc = .{ .func = 1 }, .sort_idx = .{ .sort = .func, .idx = 1 } },
    };

    const component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &aliases_decl,
        .types = &type_defs,
        .canons = &canons,
        .imports = &imports_decl,
        .exports = &exports_decl,
    };

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    const Host = struct {
        fn sub(
            _: ?*anyopaque,
            _: *ComponentInstance,
            in: []const abi_mod.InterfaceValue,
            out: []abi_mod.InterfaceValue,
            _: std.mem.Allocator,
        ) anyerror!void {
            out[0] = .{ .s32 = in[0].s32 - in[1].s32 };
        }
    };
    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    try providers.put(std.testing.allocator, "host-sub", .{
        .host_func = .{ .call = &Host.sub },
    });
    try inst.linkImports(providers);

    // Confirm export resolution found the right (instance, local) pair.
    const exported = inst.getExport("run") orelse return error.TestFailed;
    try std.testing.expect(exported == .local);
    try std.testing.expectEqual(@as(u32, 1), exported.local.core_instance_idx);
    try std.testing.expectEqual(@as(u32, 1), exported.local.core_func_idx);

    var args = [_]abi_mod.InterfaceValue{
        .{ .s32 = 7 },
        .{ .s32 = 2 },
    };
    var results: [1]abi_mod.InterfaceValue = undefined;
    try executor.callComponentFunc(inst, "run", &args, &results, std.testing.allocator);
    try std.testing.expectEqual(@as(i32, 5), results[0].s32);
}

test "instantiate: H1 micro-fixture — multi-core-module composition with cross-instance call (#156 H1)" {
    const loader_mod = @import("loader.zig");
    const executor = @import("executor.zig");
    const abi_mod = @import("canonical_abi.zig");

    // Two-module composition (see fixtures/h1-compose.wat):
    //   $A exports func "f" returning 7.
    //   $B imports "a"."f", exports "g" returning f()+1 (==8).
    //   The component aliases $b's "g" and lifts it as export "g" : u32.
    // Exercises:
    //   * `(core instance (instantiate $B (with "a" (instance $a))))` where
    //     the source instance is a real module_inst, not an inline-exports
    //     bundle — currently the resolver only consults inline_exports.
    //   * `(alias core export $b "g")` driving the lifted export through
    //     the second core instance.
    const data = @embedFile("fixtures/h1-compose.wasm");
    // The loader has no Component.deinit yet (see #142 Phase 1B); allocate
    // its small slices into an arena so the test doesn't leak.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const component_owned = try loader_mod.load(data, arena.allocator());
    var component = component_owned;

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    // No host imports needed — both core modules are self-contained.
    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    try inst.linkImports(providers);

    var args: [0]abi_mod.InterfaceValue = .{};
    var results: [1]abi_mod.InterfaceValue = undefined;
    try executor.callComponentFunc(inst, "g", &args, &results, std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 8), results[0].u32);
}

test "instantiate: H1.2 micro-fixture — cross-instance memory wiring (#156 H1.2)" {
    const loader_mod = @import("loader.zig");
    const executor = @import("executor.zig");
    const abi_mod = @import("canonical_abi.zig");

    // Two-module composition (see fixtures/h1-mem.wat):
    //   $A exports memory "mem".
    //   $B imports "a"."mem", stores 42 at offset 0 then loads it.
    //   Component lifts $b's "g" as export "g" : u32 → must be 42.
    // Exercises `(with NAME (instance N))` matching against a real source
    // instance's *memory* export, populating ImportContext.memories so the
    // shared MemoryInstance is seen by both core modules.
    const data = @embedFile("fixtures/h1-mem.wasm");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const component_owned = try loader_mod.load(data, arena.allocator());
    var component = component_owned;

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    try inst.linkImports(providers);

    // Both core instances must share the same MemoryInstance.
    try std.testing.expect(inst.core_instances.len == 2);
    const mi_a = inst.core_instances[0].module_inst orelse return error.TestFailed;
    const mi_b = inst.core_instances[1].module_inst orelse return error.TestFailed;
    try std.testing.expect(mi_a.memories.len >= 1);
    try std.testing.expect(mi_b.memories.len >= 1);
    try std.testing.expectEqual(mi_a.memories[0], mi_b.memories[0]);

    var args: [0]abi_mod.InterfaceValue = .{};
    var results: [1]abi_mod.InterfaceValue = undefined;
    try executor.callComponentFunc(inst, "g", &args, &results, std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 42), results[0].u32);
}

test "instantiate: H1.3 micro-fixture — alias core export of memory through inline-exports (#156 H1.3)" {
    const loader_mod = @import("loader.zig");
    const executor = @import("executor.zig");
    const abi_mod = @import("canonical_abi.zig");

    // Three-step composition (see fixtures/h1-alias.wat):
    //   $A exports memory "mem" and func "init" (writes 7 at addr 99).
    //   `(alias core export $a "mem" (core memory))` → top-level core mem 0.
    //   `(core instance $args (export "mem" (memory 0)))` — inline-exports
    //   bundle that re-exports the aliased memory via the SortIdx path.
    //   $B imports "src"."mem" and exports "read" (loads addr 99).
    //   $b instantiated `(with "src" (instance $args))`.
    // Exercises:
    //   * `aliasContributesTo` for `.core_memory`.
    //   * `resolveCoreMemory` ordering.
    //   * Memory import resolution against an inline-exports source whose
    //     member's SortIdx points at a top-level core memory contributed by
    //     an alias-core-export — the path stdio-echo's `$fixup` takes for
    //     the lifted `$main.memory`.
    const data = @embedFile("fixtures/h1-alias.wasm");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const component_owned = try loader_mod.load(data, arena.allocator());
    var component = component_owned;

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    try inst.linkImports(providers);

    // Both real module instances must see the same MemoryInstance even
    // though the wiring goes through an inline-exports bundle.
    try std.testing.expect(inst.core_instances.len >= 3);
    const mi_a = inst.core_instances[0].module_inst orelse return error.TestFailed;
    // core_instances[1] is the inline-exports `$args` bundle (no module_inst).
    const mi_b = inst.core_instances[2].module_inst orelse return error.TestFailed;
    try std.testing.expect(mi_a.memories.len >= 1);
    try std.testing.expect(mi_b.memories.len >= 1);
    try std.testing.expectEqual(mi_a.memories[0], mi_b.memories[0]);

    // Run init via $A so memory has 7 at offset 99, then read via $B.
    var no_args: [0]abi_mod.InterfaceValue = .{};
    var no_results: [0]abi_mod.InterfaceValue = .{};
    try executor.callComponentFunc(inst, "init", &no_args, &no_results, std.testing.allocator);

    var results: [1]abi_mod.InterfaceValue = undefined;
    try executor.callComponentFunc(inst, "read", &no_args, &results, std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 7), results[0].u32);
}

test "instantiate: H2 micro-fixture — table.set + call_indirect via canon.lower trampoline (#156 H2)" {
    const loader_mod = @import("loader.zig");
    const executor = @import("executor.zig");
    const abi_mod = @import("canonical_abi.zig");

    // Three-module composition (see fixtures/h2-trampoline.wat):
    //   $A exports table "t" (1 funcref) and func "call0" which call_indirect's
    //     element 0 with the i32 arg passed in.
    //   `(alias core export $a "t" (core table))` → top-level core table 0.
    //   `(canon lower (func $dbl))` produces a core func bound via trampoline
    //     to host_func host:double (HostFunc.call doubles its u32 arg).
    //   `(core instance $args (export "t" (table 0)) (export "f" (func ...)))`
    //   $B imports the table and the lowered func; its `start` runs
    //     `i32.const 0  ref.func $f  table.set 0`, i.e. installs the
    //     trampoline-backed funcref into the imported table at offset 0.
    // After instantiation the lifted `call0(x)` exercises:
    //   * cross-module call_indirect against an imported, post-instantiation
    //     populated table;
    //   * funcref dispatch into a host_func_entries[]-backed canon.lower
    //     trampoline;
    //   * host return-value lift back into the calling core module.
    const data = @embedFile("fixtures/h2-trampoline.wasm");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const component_owned = try loader_mod.load(data, arena.allocator());
    var component = component_owned;

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    const Host = struct {
        fn double(
            _: ?*anyopaque,
            _: *ComponentInstance,
            args: []const abi_mod.InterfaceValue,
            results: []abi_mod.InterfaceValue,
            _: std.mem.Allocator,
        ) anyerror!void {
            results[0] = .{ .u32 = args[0].u32 *% 2 };
        }
    };

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    try providers.put(std.testing.allocator, "my:host/double", .{ .host_func = .{ .call = &Host.double } });
    try inst.linkImports(providers);

    var args: [1]abi_mod.InterfaceValue = .{.{ .u32 = 21 }};
    var results: [1]abi_mod.InterfaceValue = undefined;
    try executor.callComponentFunc(inst, "call0", &args, &results, std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 42), results[0].u32);
}

test "instantiate: registers nested wasi:cli/run instance member as 'run' (#151)" {
    // Hand-authored component:
    //   - 1 core module exporting "run" (no imports, no host calls)
    //   - 1 core instance instantiating it
    //   - 1 alias of the core "run" → core-func-idx 0
    //   - 1 canon.lift over that core func → comp-func-idx 0
    //   - 1 local component-instance bundling { "run": comp-func 0 }
    //     (instance-idx 0)
    //   - 1 top-level export "wasi:cli/run@0.2.6" → instance 0
    //
    // After instantiate(), inst.getExport("run") and
    // inst.getExport("wasi:cli/run@0.2.6/run") must both resolve to
    // the lifted core export.
    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: () -> ()
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
        // function section: 1 fn of type 0
        0x03, 0x02,
        0x01, 0x00,
        // export section: "run" -> func 0
        0x07, 0x07, 0x01, 0x03, 'r',  'u',
        'n',  0x00, 0x00,
        // code section: empty body
        0x0a, 0x04, 0x01, 0x02, 0x00,
        0x0b,
    };

    const core_modules = [_]ctypes.CoreModule{.{ .data = &core_wasm }};
    const type_defs = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &.{}, .results = .none } },
    };
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .instantiate = .{ .module_idx = 0, .args = &.{} } },
    };
    const aliases_decl = [_]ctypes.Alias{
        .{ .instance_export = .{
            .sort = .{ .core = .func },
            .instance_idx = 0,
            .name = "run",
        } },
    };
    const canons = [_]ctypes.Canon{
        .{ .lift = .{ .core_func_idx = 0, .type_idx = 0, .opts = &.{} } },
    };
    const inline_exp = [_]ctypes.InlineExport{
        .{ .name = "run", .sort_idx = .{ .sort = .func, .idx = 0 } },
    };
    const instances = [_]ctypes.InstanceExpr{
        .{ .exports = &inline_exp },
    };
    const exports_decl = [_]ctypes.ExportDecl{
        .{
            .name = "wasi:cli/run@0.2.6",
            .desc = .{ .instance = 0 },
            .sort_idx = .{ .sort = .instance, .idx = 0 },
        },
    };

    const component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &instances,
        .aliases = &aliases_decl,
        .types = &type_defs,
        .canons = &canons,
        .imports = &.{},
        .exports = &exports_decl,
    };

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    const bare = inst.getExport("run") orelse return error.TestFailed;
    try std.testing.expect(bare == .local);
    try std.testing.expectEqual(@as(u32, 0), bare.local.core_instance_idx);
    try std.testing.expectEqual(@as(u32, 0), bare.local.core_func_idx);

    const dotted = inst.getExport("wasi:cli/run@0.2.6/run") orelse return error.TestFailed;
    try std.testing.expect(dotted == .local);
    try std.testing.expectEqual(@as(u32, 0), dotted.local.core_instance_idx);
    try std.testing.expectEqual(@as(u32, 0), dotted.local.core_func_idx);
}

test "instantiate: core (start ...) calling canon-lowered host import sees bound host_func (#308)" {
    const loader_mod = @import("loader.zig");
    const abi_mod = @import("canonical_abi.zig");

    // Regression for #308. The fixture has a core module whose
    // `(start ...)` directive calls a canon-lowered host import.
    //   (component
    //     (import "host:nop/run" (func $run (param "x" u32)))
    //     (core module $A
    //       (import "host" "f" (func $f (param i32)))
    //       (start $start)
    //       (func $start  i32.const 42  call $f))
    //     (core func $f_low (canon lower (func $run)))
    //     (core instance $args (export "f" (func $f_low)))
    //     (core instance $a (instantiate $A (with "host" (instance $args)))))
    //
    // Before the deferred-start fix the (start) ran during instantiate(),
    // before linkImports() bound the trampoline `host_func` — and trapped
    // with `HostFuncNotBound`. After the fix, instantiate() defers the
    // start; linkImports() binds the trampoline and then drains the
    // pending starts, so the host fn is invoked with 42 exactly once.
    const data = @embedFile("fixtures/h3-start-host-call.wasm");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const component_owned = try loader_mod.load(data, arena.allocator());
    var component = component_owned;

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    // ─ Before linkImports: the deferred start has NOT run yet.
    try std.testing.expect(inst.pending_core_starts.items.len == 1);

    const Host = struct {
        var calls: u32 = 0;
        var last_arg: u32 = 0;
        fn run(
            _: ?*anyopaque,
            _: *ComponentInstance,
            args: []const abi_mod.InterfaceValue,
            _: []abi_mod.InterfaceValue,
            _: std.mem.Allocator,
        ) anyerror!void {
            calls += 1;
            last_arg = args[0].u32;
        }
    };
    Host.calls = 0;
    Host.last_arg = 0;

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    try providers.put(std.testing.allocator, "host:nop/run", .{ .host_func = .{ .call = &Host.run } });

    // linkImports binds the trampoline AND drains pending starts.
    try inst.linkImports(providers);

    try std.testing.expectEqual(@as(u32, 1), Host.calls);
    try std.testing.expectEqual(@as(u32, 42), Host.last_arg);
    try std.testing.expectEqual(@as(usize, 0), inst.pending_core_starts.items.len);
}

test "instantiate: core (start ...) calling host import without linkImports leaves start un-run (#308)" {
    const loader_mod = @import("loader.zig");

    // Same fixture — but the caller never invokes linkImports, so the
    // pending start remains queued. We just verify that instantiate()
    // alone does NOT trap and that the deferred start is observable.
    // This is the contract `WasiCliAdapter.runLoadedComponent` relies on
    // (instantiate must succeed even when the core start would call a
    // not-yet-bound host import).
    const data = @embedFile("fixtures/h3-start-host-call.wasm");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const component_owned = try loader_mod.load(data, arena.allocator());
    var component = component_owned;

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    try std.testing.expectEqual(@as(usize, 1), inst.pending_core_starts.items.len);
}

test "instantiate: instance-type body export-of-type .eq aliases prior local slot (#310)" {
    const loader_mod = @import("loader.zig");
    const executor = @import("executor.zig");
    const abi_mod = @import("canonical_abi.zig");

    // Regression for #310. Fixture imports a host instance whose
    // instance-type body uses `(export "instant" (type (eq 0)))` to
    // alias the prior `(type u64)`. Pre-fix, `buildInstanceTypeExtension`
    // wrote `null` for the export-of-type slot, so when the canon-lower
    // trampoline lowered `now()`'s `instant`-typed result through
    // `pushInterfaceValue`'s `.type_idx => |idx|` arm, `registry.get(idx)`
    // returned null and tripped `CompoundNeedsRegistry`. Post-fix, the
    // export-of-type slot resolves to the same type_i as slot 0, so the
    // u64 round-trips end-to-end.
    //
    // This is the synthetic shape of `wasi:clocks/wall-clock`,
    // `monotonic-clock`, and similar wasi:io interfaces TinyGo's
    // `_initialize` calls during startup.
    const data = @embedFile("fixtures/h4-export-of-type-alias.wasm");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const component_owned = try loader_mod.load(data, arena.allocator());
    var component = component_owned;

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    const Host = struct {
        const expected: u64 = 0xDEADBEEFCAFEBABE;
        var calls: u32 = 0;
        fn now(
            _: ?*anyopaque,
            _: *ComponentInstance,
            _: []const abi_mod.InterfaceValue,
            results: []abi_mod.InterfaceValue,
            _: std.mem.Allocator,
        ) anyerror!void {
            calls += 1;
            results[0] = .{ .u64 = expected };
        }
    };
    Host.calls = 0;

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    var clock_iface = HostInstance{};
    defer clock_iface.members.deinit(std.testing.allocator);
    try clock_iface.members.put(std.testing.allocator, "now", .{
        .func = .{ .call = &Host.now },
    });
    try providers.put(std.testing.allocator, "host:test/clock", .{ .host_instance = &clock_iface });

    try inst.linkImports(providers);

    var args: [0]abi_mod.InterfaceValue = .{};
    var results: [1]abi_mod.InterfaceValue = undefined;
    try executor.callComponentFunc(inst, "call-now", &args, &results, std.testing.allocator);
    try std.testing.expectEqual(@as(u64, Host.expected), results[0].u64);
    try std.testing.expectEqual(@as(u32, 1), Host.calls);
}

test "instantiate: aliased instance export of sub-component publishes 'run' (#355)" {
    const executor = @import("executor.zig");
    const abi_mod = @import("canonical_abi.zig");

    // Hand-authored composed-shape fixture (issue #355):
    //   (component   (component                                 ;; sub-component
    //                  (core module $A (func $r (export "run") nop))
    //                  (core instance $a (instantiate $A))
    //                  (alias core export $a "run" (core func $cr))
    //                  (func $lr (canon lift (core func $cr)))
    //                  (instance $i0 (export "run" (func $lr)))
    //                  (export "wasi:cli/run@0.2.6" (instance $i0))
    //                )
    //     (instance $sub (instantiate 0))                        ;; comp inst 0
    //     (alias export $sub "wasi:cli/run@0.2.6" (instance $r))  ;; comp inst 1
    //     (export "wasi:cli/run@0.2.6" (instance $r))
    //   )
    //
    // After instantiate(), inst.getExport("run") must resolve to a
    // `.forwarded` whose owner is the child sub-instance and whose
    // owner_export_name is `wasi:cli/run@0.2.6/run`. Calling it
    // through `executor.callComponentFunc` must reach the child's
    // lifted core func 0 in core instance 0.
    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: () -> ()
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
        // function section: 1 fn of type 0
        0x03, 0x02,
        0x01, 0x00,
        // export section: "run" -> func 0
        0x07, 0x07, 0x01, 0x03, 'r',  'u',
        'n',  0x00, 0x00,
        // code section: empty body
        0x0a, 0x04, 0x01, 0x02, 0x00,
        0x0b,
    };

    const sub_core_modules = [_]ctypes.CoreModule{.{ .data = &core_wasm }};
    const sub_type_defs = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &.{}, .results = .none } },
    };
    const sub_core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .instantiate = .{ .module_idx = 0, .args = &.{} } },
    };
    const sub_aliases = [_]ctypes.Alias{
        .{ .instance_export = .{
            .sort = .{ .core = .func },
            .instance_idx = 0,
            .name = "run",
        } },
    };
    const sub_canons = [_]ctypes.Canon{
        .{ .lift = .{ .core_func_idx = 0, .type_idx = 0, .opts = &.{} } },
    };
    const sub_inline_exp = [_]ctypes.InlineExport{
        .{ .name = "run", .sort_idx = .{ .sort = .func, .idx = 0 } },
    };
    const sub_instances = [_]ctypes.InstanceExpr{
        .{ .exports = &sub_inline_exp },
    };
    const sub_exports = [_]ctypes.ExportDecl{
        .{
            .name = "wasi:cli/run@0.2.6",
            .desc = .{ .instance = 0 },
            .sort_idx = .{ .sort = .instance, .idx = 0 },
        },
    };
    const sub_component = ctypes.Component{
        .core_modules = &sub_core_modules,
        .core_instances = &sub_core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &sub_instances,
        .aliases = &sub_aliases,
        .types = &sub_type_defs,
        .canons = &sub_canons,
        .imports = &.{},
        .exports = &sub_exports,
    };

    // Outer component.
    const sub_components = [_]*const ctypes.Component{&sub_component};
    const outer_instances = [_]ctypes.InstanceExpr{
        .{ .instantiate = .{ .component_idx = 0, .args = &.{} } },
    };
    const outer_aliases = [_]ctypes.Alias{
        .{ .instance_export = .{
            .sort = .instance,
            .instance_idx = 0,
            .name = "wasi:cli/run@0.2.6",
        } },
    };
    const outer_exports = [_]ctypes.ExportDecl{
        .{
            .name = "wasi:cli/run@0.2.6",
            .desc = .{ .instance = 0 },
            .sort_idx = .{ .sort = .instance, .idx = 1 },
        },
    };
    // Section-order indexspace: instance[0]=comp_inst 0, alias[0]=comp_inst 1.
    const outer_comp_inst_indexspace = [_]ctypes.CompInstanceContributor{
        .{ .instance = 0 },
        .{ .alias = 0 },
    };
    const outer_component = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = @ptrCast(&sub_components),
        .instances = &outer_instances,
        .aliases = &outer_aliases,
        .types = &.{},
        .canons = &.{},
        .imports = &.{},
        .exports = &outer_exports,
        .comp_instance_indexspace = &outer_comp_inst_indexspace,
    };

    const inst = try instantiate(&outer_component, std.testing.allocator);
    defer inst.deinit();

    // Bare 'run' must be present and forwarded.
    const bare = inst.getExport("run") orelse return error.TestFailed;
    try std.testing.expect(bare == .forwarded);
    // The dotted parent key must also be present.
    try std.testing.expect(inst.getExport("wasi:cli/run@0.2.6/run") != null);

    // Flattening should bottom out on the child's local lift over
    // its core instance 0, core func 0.
    const flat = executor.flattenForwardedChain(inst, "run") orelse return error.TestFailed;
    try std.testing.expect(flat.owner != inst);
    try std.testing.expectEqual(@as(u32, 0), flat.local.core_instance_idx);
    try std.testing.expectEqual(@as(u32, 0), flat.local.core_func_idx);

    // End-to-end: invoke "run" via the executor — should not trap.
    var args_buf: [0]abi_mod.InterfaceValue = .{};
    var results_buf: [0]abi_mod.InterfaceValue = .{};
    try executor.callComponentFunc(inst, "run", &args_buf, &results_buf, std.testing.allocator);
}

test "instantiate: two-deep alias chain to sub-component instance export (#355)" {
    // Same sub-component as the previous test, but the outer adds a
    // second alias hop:
    //   alias[0] = export 0 "wasi:cli/run@0.2.6"     (comp inst 1)
    //   alias[1] = export 1 "run"                    -- (THIS would alias a func, not an instance)
    // Instead, do a real two-hop alias of an instance:
    //   instance[0] = (instantiate 0)                                 (comp inst 0)
    //   alias[0]    = export 0 "wasi:cli/run@0.2.6" (instance ;; cmp 1)
    //   alias[1]    = export 1 ???   -- aliasing the instance member of an alias of an instance
    //
    // The composed shape only contains single-hop aliases of instance
    // members in practice (each alias section emits one hop). Two-hop
    // chains arise when an alias-of-instance is itself aliased again.
    // We model this by adding a passthrough alias hop:
    //   alias[1] = export 1 ... -- but instance 1 doesn't have nested instance exports here.
    //
    // The current resolveInstanceExpr API supports only single-hop in
    // the body; multi-hop returns MultiHopAliasUnsupported. We assert
    // that explicitly so that future relaxation is intentional.
    const ier = indexspace.InstanceExprRef;
    _ = ier;

    // Build a minimal component with an alias-of-an-alias and verify
    // that resolution returns `error.MultiHopAliasUnsupported`. This
    // pins current behaviour; if the resolver later grows multi-hop
    // support, this test should be flipped to verify the resolved
    // chain lands on the underlying sub-export.
    const dummy_imports = [_]ctypes.ImportDecl{};
    const aliases = [_]ctypes.Alias{
        // alias[0]: from comp_inst 0 (an import) — but with no
        // imports declared, we instead use a synthetic instance.
        .{ .instance_export = .{
            .sort = .instance,
            .instance_idx = 0,
            .name = "first-hop",
        } },
        // alias[1]: alias of the previous alias
        .{ .instance_export = .{
            .sort = .instance,
            .instance_idx = 1,
            .name = "second-hop",
        } },
    };
    const instances = [_]ctypes.InstanceExpr{
        .{ .exports = &.{} },
    };
    const idx_space = [_]ctypes.CompInstanceContributor{
        .{ .instance = 0 }, // comp inst 0 -> instance[0]
        .{ .alias = 0 }, // comp inst 1 -> alias[0]
        .{ .alias = 1 }, // comp inst 2 -> alias[1]
    };
    const component = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &instances,
        .aliases = &aliases,
        .types = &.{},
        .canons = &.{},
        .imports = &dummy_imports,
        .exports = &.{},
        .comp_instance_indexspace = &idx_space,
    };

    // Single-hop (alias[0] of instance[0]) must resolve.
    const r1 = try indexspace.resolveInstanceExpr(&component, 1);
    try std.testing.expect(r1 != null);
    try std.testing.expect(r1.? == .sub_export);

    // Two-hop (alias[1] of alias[0]) must surface MultiHopAliasUnsupported.
    const r2 = indexspace.resolveInstanceExpr(&component, 2);
    try std.testing.expectError(error.MultiHopAliasUnsupported, r2);
}

// ── #533: canon-builtin trampoline ctx sharing ──────────────────────────────
//
// The next three tests cover the memoisation contract added for #533:
// duplicate import slots that resolve to the SAME canon-def-id share one
// `*CanonBuiltinTrampolineCtx`, while slots resolving to DIFFERENT canon
// definitions still get distinct contexts. All three exercise the
// canon-builtin registration path in `instantiate` via a section-aware
// core_instances composition (no loader; section-order fallback in
// `indexspace.resolveCoreFunc` is what wires canons → core-func indices).

// Minimal core module used by the #533 tests:
//   (module
//     (type (func (result i32)))            ;; type 0 — context.get shape
//     (type (func (param i32)))             ;; type 1 — context.set shape
//     (import "x" "f0" (func (type 0)))     ;; context.get
//     (import "x" "f1" (func (type 0)))     ;; context.get (duplicate)
//     (import "x" "f2" (func (type 1))))    ;; context.set
//
// No function/code/export sections — we only need the import slots to be
// resolved against the canon-builtin trampoline path.
const ctx_share_core_wasm_533 = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    // type section (id=1, body=9): 2 types.
    0x01, 0x09, 0x02,
    0x60, 0x00, 0x01, 0x7f, // () -> i32
    0x60, 0x01, 0x7f, 0x00, // (i32) -> ()
    // import section (id=2, body=22): 3 imports.
    0x02, 0x16, 0x03,
    // host.f0 : func (type 0)
    0x01,
    'x',  0x02, 'f',  '0',
    0x00, 0x00,
    // host.f1 : func (type 0)
    0x01, 'x',
    0x02, 'f',  '1',  0x00,
    0x00,
    // host.f2 : func (type 1)
    0x01, 'x',  0x02,
    'f',  '2',  0x00, 0x01,
};

// Shared fixture builder for the #533 tests. Returns a Component descriptor
// whose `core_instances[0]` is an inline-exports bundle mapping "f0", "f1"
// to canon-def-id 0 (`context.get`) and "f2" to canon-def-id 1
// (`context.set`), and whose `core_instances[1]` instantiates the core
// module above with `with "x" (instance 0)`.
fn build533CtxShareComponent() ctypes.Component {
    const S = struct {
        const core_modules = [_]ctypes.CoreModule{.{ .data = &ctx_share_core_wasm_533 }};
        const canons = [_]ctypes.Canon{
            .{ .context_get = .{ .val_type = .i32, .slot = 0 } }, // canon-def-id 0 -> core-func-idx 0
            .{ .context_set = .{ .val_type = .i32, .slot = 0 } }, // canon-def-id 1 -> core-func-idx 1
        };
        const inline_exports = [_]ctypes.CoreInlineExport{
            .{ .name = "f0", .sort_idx = .{ .sort = .func, .idx = 0 } }, // → canon 0 (context.get)
            .{ .name = "f1", .sort_idx = .{ .sort = .func, .idx = 0 } }, // → canon 0 (duplicate)
            .{ .name = "f2", .sort_idx = .{ .sort = .func, .idx = 1 } }, // → canon 1 (context.set)
        };
        const inst_args = [_]ctypes.CoreInstantiateArg{
            .{ .name = "x", .instance_idx = 0 },
        };
        const core_insts = [_]ctypes.CoreInstanceExpr{
            .{ .exports = &inline_exports },
            .{ .instantiate = .{ .module_idx = 0, .args = &inst_args } },
        };
    };
    return ctypes.Component{
        .core_modules = &S.core_modules,
        .core_instances = &S.core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &S.canons,
        .imports = &.{},
        .exports = &.{},
    };
}

test "instantiate: duplicate canon-builtin imports share one trampoline ctx (#533)" {
    // Allocation-count regression: a component with two imports of
    // `context.get` (same canon-def-id) must allocate exactly ONE
    // `*CanonBuiltinTrampolineCtx`, not one per import slot.
    const component = build533CtxShareComponent();
    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    const mi = inst.core_instances[1].module_inst orelse return error.TestFailed;
    try std.testing.expectEqual(@as(usize, 3), mi.host_func_entries.len);

    // Three slots: f0, f1 (both → canon-def 0), f2 (→ canon-def 1).
    // Two distinct canon-def-ids ⇒ two distinct trampoline ctxs total
    // (NOT three — that would mean the duplicate slot allocated its own).
    try std.testing.expectEqual(@as(usize, 2), inst.canon_builtin_ctxs.items.len);
    try std.testing.expectEqual(@as(u32, 2), inst.canon_builtin_ctx_by_canon_idx.count());
}

test "instantiate: same canon-def-id yields identical trampoline ctx pointer (#533)" {
    // Identity test: two import slots that both resolve to canon-def-id 0
    // must be wired with the *same* `*CanonBuiltinTrampolineCtx`, so the
    // dispatch-time `ctx_opaque` is pointer-equal between them.
    const component = build533CtxShareComponent();
    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    const mi = inst.core_instances[1].module_inst orelse return error.TestFailed;
    try std.testing.expect(mi.host_func_entries[0] != null);
    try std.testing.expect(mi.host_func_entries[1] != null);

    const ctx0 = mi.host_func_entries[0].?.ctx.?;
    const ctx1 = mi.host_func_entries[1].?.ctx.?;
    try std.testing.expectEqual(ctx0, ctx1);

    // Both must alias the single ctx recorded in the memoisation map for
    // canon-def-id 0.
    const memoised = inst.canon_builtin_ctx_by_canon_idx.get(0) orelse return error.TestFailed;
    try std.testing.expectEqual(@as(*anyopaque, @ptrCast(memoised)), ctx0);
}

test "instantiate: distinct canon-def-ids yield distinct trampoline ctx pointers (#533)" {
    // Negative: two import slots resolving to DIFFERENT canon-def-ids
    // (`context.get` vs `context.set`) must NOT share a trampoline ctx.
    // Memoisation is keyed by canon-def-id; different keys ⇒ different
    // contexts, otherwise the dispatched `canon` payload would be wrong.
    const component = build533CtxShareComponent();
    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    const mi = inst.core_instances[1].module_inst orelse return error.TestFailed;
    try std.testing.expect(mi.host_func_entries[0] != null);
    try std.testing.expect(mi.host_func_entries[2] != null);

    const ctx0 = mi.host_func_entries[0].?.ctx.?;
    const ctx2 = mi.host_func_entries[2].?.ctx.?;
    try std.testing.expect(ctx0 != ctx2);

    // And the underlying Canon payloads must differ — sanity-check the
    // memoisation didn't accidentally alias across distinct tags.
    const ctx2_typed: *executor_mod.CanonBuiltinTrampolineCtx = @ptrCast(@alignCast(ctx2));
    try std.testing.expect(ctx2_typed.canon == .context_set);
    const ctx0_typed: *executor_mod.CanonBuiltinTrampolineCtx = @ptrCast(@alignCast(ctx0));
    try std.testing.expect(ctx0_typed.canon == .context_get);
}

test "instantiate: dropping component frees shared canon-builtin ctx exactly once (#533)" {
    // Memory cleanup: with N>1 import slots sharing one ctx via
    // memoisation, `deinit` must free that ctx exactly once — not N times
    // (use-after-free) and not zero times (leak). `std.testing.allocator`
    // is leak-detecting and panics on double-free; passing this test with
    // `defer inst.deinit()` exercises both failure modes.
    const component = build533CtxShareComponent();
    const inst = try instantiate(&component, std.testing.allocator);

    // Sanity: 3 import slots, 2 unique canon-def-ids ⇒ 2 owned ctxs.
    try std.testing.expectEqual(@as(usize, 2), inst.canon_builtin_ctxs.items.len);
    const mi = inst.core_instances[1].module_inst orelse return error.TestFailed;
    try std.testing.expectEqual(mi.host_func_entries[0].?.ctx, mi.host_func_entries[1].?.ctx);

    // Drop the instance — the testing allocator panics on leak or
    // double-free. If memoisation had handed out the same `*Context`
    // pointer to multiple `canon_builtin_ctxs.append` calls, the
    // `destroy()` loop in `deinit` would free it twice and trip the
    // safety check here.
    inst.deinit();
}

// ─── #625 phase 1: AOT-backed core instance load smoke test ────────────────
// (See `src/tests/component_aot_smoke_test.zig`. The test lives in its
// own module/test step rather than here because `aot_harness.zig` is
// owned by the test-runner module and Zig 0.16 rejects a file existing
// in two modules; importing it from `instance.zig` would pull
// `aot_harness.zig` into the `wamr` lib module and break the
// `differential.zig` and `run_spec_tests.zig` test runners that also
// import it.)
