//! Thread manager for WASI-threads.
//!
//! Manages thread IDs, thread lifecycle, and coordinates thread
//! spawning/termination for the WASI-threads proposal.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../runtime/common/types.zig");
const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
const execution_context = @import("../runtime/common/execution_context.zig");
const termination = @import("../runtime/common/termination.zig");
const platform = @import("../platform/platform.zig");
const parking_lot = @import("../platform/parking_lot.zig");
const windows_poll = @import("../platform/windows_poll.zig");
const config = @import("config");

/// Simple spinlock mutex (Zig 0.16 moved std.Thread.Mutex behind Io).
const Mutex = struct {
    state: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),
    pub const init: Mutex = .{ .state = std.atomic.Value(u8).init(0) };
    pub fn lock(self: *Mutex) void {
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null)
            std.atomic.spinLoopHint();
    }
    pub fn unlock(self: *Mutex) void {
        self.state.store(0, .release);
    }
};

/// Default auxiliary stack size per thread (bytes).
const DEFAULT_AUX_STACK_SIZE: u32 = 8192;

/// Auxiliary stack pool — manages per-thread stack regions in shared linear memory.
/// Stacks are allocated from a reserved region at the top of linear memory.
pub const AuxStackPool = struct {
    /// Stack size per thread (bytes).
    stack_size: u32 = DEFAULT_AUX_STACK_SIZE,
    /// Free stack offsets (top-of-stack addresses in linear memory).
    free_stacks: std.ArrayListUnmanaged(u32) = .empty,
    /// All allocated stacks (for cleanup).
    all_stacks: std.ArrayListUnmanaged(u32) = .empty,
    capacity: usize = 0,
    in_use: usize = 0,
    next_stack_top: u32 = 0,
    reserved_end: u32 = 0,
    memory: ?*types.MemoryInstance = null,
    allocator: std.mem.Allocator = undefined,
    configured: bool = false,
    mutex: Mutex = .init,

    /// Pre-allocate N auxiliary stacks starting at `base_offset` in linear memory.
    pub fn init(self: *AuxStackPool, count: u32, base_offset: u32, allocator: std.mem.Allocator) !void {
        std.debug.assert(!self.configured);
        std.debug.assert(self.free_stacks.items.len == 0);
        std.debug.assert(self.all_stacks.items.len == 0);

        var free_stacks: std.ArrayListUnmanaged(u32) = .empty;
        errdefer free_stacks.deinit(allocator);
        var all_stacks: std.ArrayListUnmanaged(u32) = .empty;
        errdefer all_stacks.deinit(allocator);
        try free_stacks.ensureTotalCapacity(allocator, count);
        try all_stacks.ensureTotalCapacity(allocator, count);

        var offset = base_offset;
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            const stack_top = std.math.add(u32, offset, self.stack_size) catch
                return error.StackAddressOverflow;
            free_stacks.appendAssumeCapacity(stack_top);
            all_stacks.appendAssumeCapacity(stack_top);
            offset = stack_top;
        }

        self.free_stacks = free_stacks;
        self.all_stacks = all_stacks;
        self.capacity = count;
        self.next_stack_top = offset;
        self.reserved_end = offset;
        self.allocator = allocator;
        self.configured = true;
    }

    /// Configure a lazily committed stack region. Unlike `init`, this does not
    /// materialize every stack up front; the shared memory grows only far
    /// enough for each newly assigned stack.
    pub fn initGrowing(
        self: *AuxStackPool,
        count: u32,
        base_offset: u32,
        memory: *types.MemoryInstance,
        allocator: std.mem.Allocator,
    ) !void {
        std.debug.assert(!self.configured);
        std.debug.assert(self.free_stacks.items.len == 0);
        std.debug.assert(self.all_stacks.items.len == 0);

        const region_size = std.math.mul(u32, count, self.stack_size) catch
            return error.StackAddressOverflow;
        const reserved_end = std.math.add(u32, base_offset, region_size) catch
            return error.StackAddressOverflow;

        var free_stacks: std.ArrayListUnmanaged(u32) = .empty;
        errdefer free_stacks.deinit(allocator);
        var all_stacks: std.ArrayListUnmanaged(u32) = .empty;
        errdefer all_stacks.deinit(allocator);
        try free_stacks.ensureTotalCapacity(allocator, count);
        try all_stacks.ensureTotalCapacity(allocator, count);

        self.free_stacks = free_stacks;
        self.all_stacks = all_stacks;
        self.capacity = count;
        self.next_stack_top = base_offset;
        self.reserved_end = reserved_end;
        self.memory = memory;
        self.allocator = allocator;
        self.configured = true;
    }

    pub fn deinit(self: *AuxStackPool, allocator: std.mem.Allocator) void {
        if (!self.configured) return;
        std.debug.assert(self.in_use == 0);
        std.debug.assert(self.free_stacks.items.len == self.all_stacks.items.len);
        self.free_stacks.deinit(allocator);
        self.all_stacks.deinit(allocator);
        self.free_stacks = .empty;
        self.all_stacks = .empty;
        self.capacity = 0;
        self.in_use = 0;
        self.next_stack_top = 0;
        self.reserved_end = 0;
        self.memory = null;
        self.configured = false;
    }

    /// Allocate a stack for a new thread. Returns the top-of-stack offset, or null.
    pub fn allocate(self: *AuxStackPool) ?u32 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const items = self.free_stacks.items;
        if (items.len != 0) {
            const val = items[items.len - 1];
            self.free_stacks.items.len -= 1;
            self.in_use += 1;
            return val;
        }
        if (self.all_stacks.items.len == self.capacity) return null;

        const stack_top = std.math.add(u32, self.next_stack_top, self.stack_size) catch
            return null;
        if (stack_top > self.reserved_end) return null;
        if (self.memory) |memory| {
            const required_pages_u64 =
                (@as(u64, stack_top) + types.MemoryInstance.page_size - 1) /
                types.MemoryInstance.page_size;
            const required_pages: u32 = @intCast(required_pages_u64);
            const current_pages = memory.pageCount();
            if (required_pages > current_pages) {
                _ = memory.grow(required_pages - current_pages, self.allocator) catch
                    return null;
            }
        }

        self.all_stacks.appendAssumeCapacity(stack_top);
        self.next_stack_top = stack_top;
        self.in_use += 1;
        return stack_top;
    }

    /// Return a stack to the pool.
    pub fn release(self: *AuxStackPool, stack_top: u32) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        std.debug.assert(std.mem.indexOfScalar(u32, self.all_stacks.items, stack_top) != null);
        std.debug.assert(std.mem.indexOfScalar(u32, self.free_stacks.items, stack_top) == null);
        std.debug.assert(self.free_stacks.items.len < self.free_stacks.capacity);
        std.debug.assert(self.in_use > 0);
        self.free_stacks.appendAssumeCapacity(stack_top);
        self.in_use -= 1;
    }

    pub fn availableCount(self: *AuxStackPool) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.capacity - self.in_use;
    }

    pub fn totalCount(self: *AuxStackPool) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.capacity;
    }

    pub fn isConfigured(self: *AuxStackPool) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.configured;
    }
};

pub const ThreadOutcome = enum {
    completed,
    trapped,
};

pub const SpawnError = error{
    ThreadFeatureDisabled,
    ThreadGroupShuttingDown,
    ThreadIdExhausted,
    MissingThreadStart,
    InvalidThreadStartSignature,
    AuxStackExhausted,
    ChildInitializationFailed,
    ThreadSpawnFailed,
    StartGateFailed,
    OutOfMemory,
};

/// Type-erased backend contract used by the lifecycle manager.
///
/// The manager owns the context returned by `create` from that point until
/// rollback or join. Backends keep their runtime-specific types out of this
/// module, avoiding an AOT runtime ↔ host bridge ↔ thread manager import
/// cycle while preserving one publication/join implementation.
pub const ThreadBackendOps = struct {
    create: *const fn (
        parent: *anyopaque,
        allocator: std.mem.Allocator,
    ) SpawnError!*anyopaque,
    configure: *const fn (
        child: *anyopaque,
        manager: *ThreadManager,
        tid: i32,
        start_arg: u32,
        auxiliary_stack: ?execution_context.AuxiliaryStack,
    ) SpawnError!void,
    run: *const fn (child: *anyopaque) ThreadOutcome,
    destroy: *const fn (child: *anyopaque) void,
    uses_auxiliary_stack: bool = false,
};

pub const PrepareError = error{
    ThreadFeatureDisabled,
    AlreadyPrepared,
    SharedMemoryRequired,
    InvalidHeapBase,
    AuxStackExhausted,
    OutOfMemory,
};

pub const BindError = error{
    WindowsCancelEventUnavailable,
};

pub const JoinError = error{
    InvalidThreadId,
    UnknownThread,
    StaleThreadId,
    ThreadAlreadyJoining,
    ThreadAlreadyJoined,
    ThreadGroupShuttingDown,
};

pub const JoinSummary = struct {
    joined: usize = 0,
    trapped: usize = 0,
};

/// Backend hook used to publish group cancellation into compiled code.
pub const CancelBroadcast = struct {
    ctx: *anyopaque,
    broadcast: *const fn (*anyopaque) void,
};

/// Result of a bounded group teardown.
pub const TerminationSummary = struct {
    joined: usize = 0,
    trapped: usize = 0,
    /// Records still executing guest code when the deadline expired. The
    /// manager keeps every shared resource alive for them; the embedder
    /// decides whether to keep waiting or to end the host process.
    unfinished: usize = 0,
    timed_out: bool = false,
};

/// Default bound on sibling teardown. Cooperative siblings unwind as soon as
/// they observe the terminal outcome (interpreter poll, wait cancellation,
/// or a blocking-I/O slice boundary); this only bounds the pathological case
/// of an AOT sibling spinning without any interruptible host call.
pub const default_termination_timeout_ns: u64 = 2 * std.time.ns_per_s;

pub const ThreadStats = struct {
    active: usize,
    completed: usize,
    retained: usize,
    spawning: usize,
    joining: usize,
    slots: usize,
    shutting_down: bool,
};

/// Deterministic failure and destruction hooks used by lifecycle tests.
const TestHooks = struct {
    fail_child_initialization: bool = false,
    fail_native_spawn: bool = false,
    fail_start_gate: bool = false,
    fail_windows_cancel_event: bool = false,
    native_threads_started: ?*std.atomic.Value(usize) = null,
    native_threads_joined: ?*std.atomic.Value(usize) = null,
    records_destroyed: ?*std.atomic.Value(usize) = null,
};

const tid_slot_bits = 16;
const tid_generation_bits = 13;
// WASI reserves positive TIDs below 2^29. Reuse advances the generation;
// exhausted generations retire their slot instead of ever repeating an ID.
const tid_slot_mask: u32 = (1 << tid_slot_bits) - 1;
const max_thread_slots: usize = tid_slot_mask;
const max_tid_generation: u16 = (1 << tid_generation_bits) - 1;
const join_batch_size = 32;

const GateState = enum(u8) {
    closed,
    run,
    abort,
};

const StartGate = struct {
    state: std.atomic.Value(u8) = std.atomic.Value(u8).init(@intFromEnum(GateState.closed)),

    fn open(self: *StartGate) void {
        std.debug.assert(self.state.load(.monotonic) == @intFromEnum(GateState.closed));
        self.state.store(@intFromEnum(GateState.run), .release);
    }

    fn abort(self: *StartGate) void {
        self.state.store(@intFromEnum(GateState.abort), .release);
    }

    fn wait(self: *StartGate) bool {
        var spins: usize = 0;
        while (true) {
            switch (@as(GateState, @enumFromInt(self.state.load(.acquire)))) {
                .closed => {
                    if (spins < 128) {
                        spins += 1;
                        std.atomic.spinLoopHint();
                    } else {
                        std.Thread.yield() catch {};
                    }
                },
                .run => return true,
                .abort => return false,
            }
        }
    }
};

const ExecutionState = enum(u8) {
    pending,
    completed,
    trapped,
    start_aborted,
};

const ThreadRecord = struct {
    manager: *ThreadManager,
    tid: i32 = 0,
    thread: ?std.Thread = null,
    backend_context: *anyopaque,
    backend_ops: *const ThreadBackendOps,
    aux_stack_top: ?u32,
    start_gate: StartGate = .{},
    execution: std.atomic.Value(u8) =
        std.atomic.Value(u8).init(@intFromEnum(ExecutionState.pending)),
};

const InterpThreadContext = struct {
    instance: *types.ModuleInstance,
    env: *ExecEnv,
    func_idx: u32,
    allocator: std.mem.Allocator,
};

fn createInterpThreadContext(
    parent_opaque: *anyopaque,
    allocator: std.mem.Allocator,
) SpawnError!*anyopaque {
    const parent: *types.ModuleInstance = @ptrCast(@alignCast(parent_opaque));
    const func_idx = parent.getExportFunc("wasi_thread_start") orelse
        return error.MissingThreadStart;
    const func_type = parent.module.getFuncType(func_idx) orelse
        return error.InvalidThreadStartSignature;
    if (!isWasiThreadStartType(func_type))
        return error.InvalidThreadStartSignature;
    const child = parent.cloneForThread(allocator) catch return error.OutOfMemory;
    errdefer child.destroyThreadClone();
    const env = ExecEnv.create(child, 4096, allocator) catch return error.OutOfMemory;
    errdefer env.destroy();
    const context = allocator.create(InterpThreadContext) catch
        return error.OutOfMemory;
    context.* = .{
        .instance = child,
        .env = env,
        .func_idx = func_idx,
        .allocator = allocator,
    };
    return @ptrCast(context);
}

fn configureInterpThreadContext(
    child_opaque: *anyopaque,
    manager: *ThreadManager,
    tid: i32,
    start_arg: u32,
    auxiliary_stack: ?execution_context.AuxiliaryStack,
) SpawnError!void {
    const child: *InterpThreadContext = @ptrCast(@alignCast(child_opaque));
    if (auxiliary_stack) |stack| {
        if (child.instance.module.findExport("__stack_pointer", .global)) |exp| {
            if (exp.index < child.instance.globals.len) {
                child.instance.globals[exp.index].value = .{ .i32 = @bitCast(stack.top) };
            }
        }
    }
    child.env.setThreadManager(manager);
    child.env.configureWasiThread(tid, start_arg, auxiliary_stack);
    child.env.pushI32(tid) catch return error.ChildInitializationFailed;
    child.env.pushI32(@bitCast(start_arg)) catch
        return error.ChildInitializationFailed;
}

fn runInterpThreadContext(child_opaque: *anyopaque) ThreadOutcome {
    const child: *InterpThreadContext = @ptrCast(@alignCast(child_opaque));
    const interp = @import("../runtime/interpreter/interp.zig");
    interp.executeFunction(child.env, child.func_idx) catch return .trapped;
    return .completed;
}

fn destroyInterpThreadContext(child_opaque: *anyopaque) void {
    const child: *InterpThreadContext = @ptrCast(@alignCast(child_opaque));
    const allocator = child.allocator;
    child.env.destroy();
    child.instance.destroyThreadClone();
    allocator.destroy(child);
}

const interp_thread_ops = ThreadBackendOps{
    .create = createInterpThreadContext,
    .configure = configureInterpThreadContext,
    .run = runInterpThreadContext,
    .destroy = destroyInterpThreadContext,
    .uses_auxiliary_stack = true,
};

const SlotState = enum {
    live,
    joining,
    free,
    retired,
};

const ThreadSlot = struct {
    generation: u16,
    state: SlotState,
    record: ?*ThreadRecord,
};

const ParsedTid = struct {
    slot_index: usize,
    generation: u16,
};

const JoinClaim = struct {
    slot_index: usize,
    generation: u16,
    record: *ThreadRecord,
};

/// Thread manager for coordinating WASI threads.
///
/// After first use the manager is address-stable. The host owner must keep it
/// alive until `shutdown`/`deinit` completes. Shutdown closes the group to new
/// spawns and joins every record; it does not detach or preempt guest code.
pub const ThreadManager = struct {
    slots: std.ArrayList(ThreadSlot),
    mutex: Mutex = .init,
    trap_flag: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    allocator: std.mem.Allocator,
    aux_stack_pool: AuxStackPool = .{},
    shutting_down: bool = false,
    spawning_count: usize = 0,
    joining_count: usize = 0,
    test_hooks: ?*const TestHooks = null,
    /// The group's first-wins terminal outcome, owned by the shared process
    /// state. Bound by the embedder; when null every termination path stays
    /// a manager-local interrupt flag exactly as before.
    termination: ?*termination.State = null,
    /// Shared linear memory backing guest futexes, recorded by
    /// `prepareSharedMemory` so termination can cancel every waiter.
    shared_memory: ?*types.MemoryInstance = null,
    /// Backend hook that publishes the group-cancel word read by compiled
    /// code. The AOT runtime installs it so a child spinning in guest code
    /// still reaches an interruption point; the interpreter needs none
    /// because its dispatch loop polls the manager directly.
    cancel_broadcast: ?CancelBroadcast = null,
    /// Lazily-created manual-reset event included in every Windows host wait
    /// set. It stays signalled after the first group interruption.
    windows_cancel: windows_poll.CancelEvent,

    pub fn init(allocator: std.mem.Allocator) ThreadManager {
        return .{
            .slots = .empty,
            .allocator = allocator,
            .windows_cancel = windows_poll.CancelEvent.init(),
        };
    }

    /// Reserve a guest-memory auxiliary-stack region before any guest entry
    /// point runs. `__heap_base` is advanced past the reserved address range;
    /// individual pages are committed lazily by `AuxStackPool.allocate`.
    pub fn prepareInterpreterInstance(
        self: *ThreadManager,
        parent_inst: *types.ModuleInstance,
    ) PrepareError!void {
        if (comptime !config.lib_wasi_threads or !config.thread_mgr or builtin.single_threaded)
            return error.ThreadFeatureDisabled;
        if (parent_inst.memories.len == 0) return error.SharedMemoryRequired;

        const memory = parent_inst.memories[0];
        const heap_global: ?*types.GlobalInstance = if (parent_inst.module.findExport(
            "__heap_base",
            .global,
        )) |heap_export| blk: {
            if (heap_export.index >= parent_inst.globals.len)
                return error.InvalidHeapBase;
            break :blk parent_inst.globals[heap_export.index];
        } else null;
        return self.prepareSharedMemory(memory, heap_global);
    }

    /// Configure the auxiliary-stack pool for a backend that exposes the
    /// group's shared memory and optional `__heap_base` global directly.
    pub fn prepareSharedMemory(
        self: *ThreadManager,
        memory: *types.MemoryInstance,
        heap_global: ?*types.GlobalInstance,
    ) PrepareError!void {
        if (comptime !config.lib_wasi_threads or !config.thread_mgr or builtin.single_threaded)
            return error.ThreadFeatureDisabled;
        if (self.aux_stack_pool.isConfigured() or self.slots.items.len != 0)
            return error.AlreadyPrepared;
        if (memory.shared_control == null) return error.SharedMemoryRequired;
        const heap_base: u32 = if (heap_global) |global|
            switch (global.value) {
                .i32 => |value| @bitCast(value),
                else => return error.InvalidHeapBase,
            }
        else
            std.math.cast(u32, memory.byteLen()) orelse
                return error.InvalidHeapBase;

        const aligned_base_u64 = (@as(u64, heap_base) + 15) & ~@as(u64, 15);
        const max_memory_bytes = @as(u64, memory.max_pages) * types.MemoryInstance.page_size;
        const max_address = @min(max_memory_bytes, @as(u64, std.math.maxInt(u32)));
        if (aligned_base_u64 >= max_address) return error.AuxStackExhausted;

        // Keep at least half of the declared address range available to the
        // guest allocator. The reserved stack window is address space only;
        // shared-memory pages are committed lazily as threads are spawned.
        const available_stacks =
            ((max_address - aligned_base_u64) / 2) / self.aux_stack_pool.stack_size;
        const stack_count_u64 = @min(@as(u64, max_thread_slots), available_stacks);
        if (stack_count_u64 == 0) return error.AuxStackExhausted;

        const stack_count: u32 = @intCast(stack_count_u64);
        const aligned_base: u32 = @intCast(aligned_base_u64);
        self.aux_stack_pool.initGrowing(
            stack_count,
            aligned_base,
            memory,
            self.allocator,
        ) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.StackAddressOverflow => return error.InvalidHeapBase,
        };

        const reserved_end = aligned_base +
            stack_count * self.aux_stack_pool.stack_size;
        if (heap_global) |global| {
            global.value = .{ .i32 = @bitCast(reserved_end) };
        }
        self.shared_memory = memory;
    }

    fn initWithTestHooks(allocator: std.mem.Allocator, hooks: *const TestHooks) ThreadManager {
        var manager = init(allocator);
        manager.test_hooks = hooks;
        return manager;
    }

    pub fn deinit(self: *ThreadManager) void {
        self.unbindTermination();
        self.shutdown();
        const current = self.stats();
        std.debug.assert(current.retained == 0);
        std.debug.assert(current.spawning == 0);
        std.debug.assert(current.joining == 0);
        self.slots.deinit(self.allocator);
        self.aux_stack_pool.deinit(self.allocator);
        self.shared_memory = null;
        self.cancel_broadcast = null;
        self.windows_cancel.deinit();
    }

    /// Bind the group to the shared process state's terminal-outcome record.
    ///
    /// Termination claimed anywhere — including `proc_exit` reached through
    /// the process state without ever touching this manager — then wakes
    /// every sibling through `interrupt`.
    pub fn bindTermination(self: *ThreadManager, state: *termination.State) BindError!void {
        if (comptime config.lib_wasi_threads) {
            const fail_for_test = if (self.test_hooks) |hooks|
                hooks.fail_windows_cancel_event
            else
                false;
            self.windows_cancel.ensureInitialized(fail_for_test) catch
                return error.WindowsCancelEventUnavailable;
        }
        self.termination = state;
        state.bindWindowsCancelHandle(self.windows_cancel.opaqueHandle());
        state.bindWake(.{ .ctx = @ptrCast(self), .wake = wakeFromTermination });
    }

    /// Drop the wakeup hook. Required before the manager's storage dies,
    /// since the process state can outlive it.
    pub fn unbindTermination(self: *ThreadManager) void {
        const state = self.termination orelse return;
        state.unbindWake(@ptrCast(self));
        state.bindWindowsCancelHandle(null);
        self.termination = null;
    }

    fn wakeFromTermination(raw: *anyopaque) void {
        const self: *ThreadManager = @ptrCast(@alignCast(raw));
        self.interrupt();
    }

    /// Signal all threads to stop (trap propagation).
    ///
    /// The first terminating thread also publishes a trap as the group's
    /// terminal outcome; a `proc_exit` that already won keeps its status.
    pub fn signalTrap(self: *ThreadManager) void {
        if (self.termination) |state|
            _ = state.claimTrap(termination.generic_trap_code);
        self.interrupt();
    }

    /// Install the backend hook that publishes cancellation into compiled
    /// code. Replaying it on every `interrupt` keeps threads that were
    /// spawned mid-teardown from missing the signal.
    pub fn bindCancelBroadcast(self: *ThreadManager, hook: CancelBroadcast) void {
        self.cancel_broadcast = hook;
        if (self.isTerminating()) hook.broadcast(hook.ctx);
    }

    /// Interrupt every sibling: raise the polled interrupt flag, publish the
    /// compiled-code cancel word, and cancel guest futex waits so blocked
    /// threads unwind instead of hanging. Idempotent, safe from any thread.
    pub fn interrupt(self: *ThreadManager) void {
        self.trap_flag.store(true, .release);
        _ = self.windows_cancel.signal();
        if (self.cancel_broadcast) |hook| hook.broadcast(hook.ctx);
        if (self.shared_memory) |memory| _ = memory.cancelWaiters() catch {};
    }

    /// True once the group's terminal outcome is claimed, or once a trap has
    /// been signalled locally. Polled by the interpreter loop and checked at
    /// every interruptible host blocking point.
    pub fn isTerminating(self: *ThreadManager) bool {
        if (self.trap_flag.load(.acquire)) return true;
        const state = self.termination orelse return false;
        return state.isTerminating();
    }

    /// Terminal outcome of the group, when one has been claimed.
    pub fn terminalOutcome(self: *ThreadManager) ?termination.Outcome {
        const state = self.termination orelse return null;
        return state.outcome();
    }

    /// Check if a trap has been signaled.
    pub fn hasTrap(self: *ThreadManager) bool {
        return self.isTerminating();
    }

    /// Join every retained record in bounded batches. Completed native handles
    /// stay owned by their records until this function or `joinOne` claims them.
    pub fn joinAll(self: *ThreadManager) void {
        _ = self.joinAllWithSummary();
    }

    pub fn joinAllWithSummary(self: *ThreadManager) JoinSummary {
        var summary = JoinSummary{};
        while (true) {
            var batch: [join_batch_size]JoinClaim = undefined;
            var count: usize = 0;

            self.mutex.lock();
            for (self.slots.items, 0..) |*slot, slot_index| {
                if (count == batch.len) break;
                if (slot.state != .live) continue;
                const record = slot.record.?;
                slot.state = .joining;
                self.joining_count += 1;
                batch[count] = .{
                    .slot_index = slot_index,
                    .generation = slot.generation,
                    .record = record,
                };
                count += 1;
            }
            const joins_in_flight = self.joining_count;
            const spawns_in_flight = self.spawning_count;
            self.mutex.unlock();

            for (batch[0..count]) |claim| {
                const outcome = self.joinClaimed(claim);
                summary.joined += 1;
                if (outcome == .trapped) summary.trapped += 1;
            }
            if (count != 0) continue;
            if (joins_in_flight == 0 and spawns_in_flight == 0) return summary;
            yieldForLifecycle();
        }
    }

    pub fn joinOne(self: *ThreadManager, tid: i32) JoinError!ThreadOutcome {
        self.mutex.lock();
        const claim = self.claimOneLocked(tid) catch |err| {
            self.mutex.unlock();
            return err;
        };
        self.mutex.unlock();
        return self.joinClaimed(claim);
    }

    /// Stop accepting new records, then take ownership of joining all existing
    /// records. Running guest code is allowed to finish cooperatively.
    pub fn shutdown(self: *ThreadManager) void {
        _ = self.shutdownWithSummary();
    }

    pub fn shutdownWithSummary(self: *ThreadManager) JoinSummary {
        self.mutex.lock();
        self.shutting_down = true;
        self.mutex.unlock();
        self.waitForSpawnsToDrain();
        return self.joinAllWithSummary();
    }

    /// Bounded group teardown for a claimed terminal outcome.
    ///
    /// Closes the group to new spawns, wakes every sibling, and then joins
    /// only the records that have already stopped executing guest code, so
    /// no single join can block. The loop re-arms the interrupt each round —
    /// that closes the race where a sibling enters a futex wait just after
    /// the first wake sweep — and gives up after `timeout_ns`, reporting the
    /// siblings still running. Nothing shared is released while a record is
    /// unfinished, so there is no use-after-free and no double free.
    pub fn terminateAndJoin(
        self: *ThreadManager,
        timeout_ns: u64,
    ) TerminationSummary {
        self.mutex.lock();
        self.shutting_down = true;
        self.mutex.unlock();

        var summary = TerminationSummary{};
        const deadline = monotonicNowNs() +| timeout_ns;
        while (true) {
            self.interrupt();

            const drained = self.joinFinishedRecords(&summary);
            const current = self.stats();
            if (current.retained == 0 and
                current.spawning == 0 and
                current.joining == 0) return summary;
            if (drained) continue;
            if (monotonicNowNs() >= deadline) {
                summary.unfinished = current.retained;
                summary.timed_out = true;
                return summary;
            }
            platform.usleep(200);
        }
    }

    /// Join every record whose guest thread has already stopped. Returns
    /// true when at least one record was reclaimed.
    fn joinFinishedRecords(
        self: *ThreadManager,
        summary: *TerminationSummary,
    ) bool {
        var reclaimed = false;
        while (true) {
            var batch: [join_batch_size]JoinClaim = undefined;
            var count: usize = 0;

            self.mutex.lock();
            for (self.slots.items, 0..) |*slot, slot_index| {
                if (count == batch.len) break;
                if (slot.state != .live) continue;
                const record = slot.record.?;
                if (executionState(record) == .pending) continue;
                slot.state = .joining;
                self.joining_count += 1;
                batch[count] = .{
                    .slot_index = slot_index,
                    .generation = slot.generation,
                    .record = record,
                };
                count += 1;
            }
            self.mutex.unlock();

            if (count == 0) return reclaimed;
            for (batch[0..count]) |claim| {
                const outcome = self.joinClaimed(claim);
                summary.joined += 1;
                if (outcome == .trapped) summary.trapped += 1;
            }
            reclaimed = true;
        }
    }

    pub fn isShuttingDown(self: *ThreadManager) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.shutting_down;
    }

    pub fn activeCount(self: *ThreadManager) usize {
        return self.stats().active;
    }

    pub fn retainedCount(self: *ThreadManager) usize {
        return self.stats().retained;
    }

    pub fn completedCount(self: *ThreadManager) usize {
        return self.stats().completed;
    }

    pub fn stats(self: *ThreadManager) ThreadStats {
        var active: usize = 0;
        var completed: usize = 0;
        var retained: usize = 0;

        self.mutex.lock();
        defer self.mutex.unlock();
        for (self.slots.items) |slot| {
            switch (slot.state) {
                .live, .joining => {
                    retained += 1;
                    const record = slot.record orelse continue;
                    switch (executionState(record)) {
                        .pending => active += 1,
                        .completed, .trapped, .start_aborted => completed += 1,
                    }
                },
                .free, .retired => {},
            }
        }
        return .{
            .active = active,
            .completed = completed,
            .retained = retained,
            .spawning = self.spawning_count,
            .joining = self.joining_count,
            .slots = self.slots.items.len,
            .shutting_down = self.shutting_down,
        };
    }

    pub fn threadOutcome(self: *ThreadManager, tid: i32) JoinError!?ThreadOutcome {
        const parsed = parseTid(tid) orelse return error.InvalidThreadId;
        self.mutex.lock();
        defer self.mutex.unlock();
        if (parsed.slot_index >= self.slots.items.len) return error.UnknownThread;
        const slot = &self.slots.items[parsed.slot_index];
        if (slot.generation != parsed.generation) return error.StaleThreadId;
        switch (slot.state) {
            .joining => return error.ThreadAlreadyJoining,
            .free, .retired => return error.ThreadAlreadyJoined,
            .live => {
                return switch (executionState(slot.record.?)) {
                    .pending => null,
                    .completed => .completed,
                    .trapped => .trapped,
                    .start_aborted => unreachable,
                };
            },
        }
    }

    /// Spawn a new thread with a cloned module instance.
    /// The new thread calls the exported `wasi_thread_start(tid, start_arg)` function.
    pub fn spawnThread(self: *ThreadManager, parent_inst: *types.ModuleInstance, start_arg: i32) SpawnError!i32 {
        return self.spawnWithBackend(
            @ptrCast(parent_inst),
            start_arg,
            &interp_thread_ops,
        );
    }

    /// Spawn a backend-specific guest thread under the same publication,
    /// rollback, generation-safe TID, join, and shutdown contract.
    pub fn spawnWithBackend(
        self: *ThreadManager,
        parent: *anyopaque,
        start_arg: i32,
        backend_ops: *const ThreadBackendOps,
    ) SpawnError!i32 {
        if (comptime !config.lib_wasi_threads or !config.thread_mgr or builtin.single_threaded)
            return error.ThreadFeatureDisabled;
        if (!self.beginSpawn()) return error.ThreadGroupShuttingDown;
        defer self.endSpawn();

        const backend_context = try backend_ops.create(parent, self.allocator);
        var backend_owned_directly = true;
        defer if (backend_owned_directly) backend_ops.destroy(backend_context);

        const aux_stack_top = if (backend_ops.uses_auxiliary_stack)
            self.aux_stack_pool.allocate()
        else
            null;
        if (backend_ops.uses_auxiliary_stack and
            aux_stack_top == null and
            self.aux_stack_pool.isConfigured())
            return error.AuxStackExhausted;
        var stack_owned_directly = aux_stack_top != null;
        defer if (stack_owned_directly) self.aux_stack_pool.release(aux_stack_top.?);

        if (self.test_hooks) |hooks| {
            if (hooks.fail_child_initialization)
                return error.ChildInitializationFailed;
        }

        const record = self.allocator.create(ThreadRecord) catch return error.OutOfMemory;
        record.* = .{
            .manager = self,
            .backend_context = backend_context,
            .backend_ops = backend_ops,
            .aux_stack_top = aux_stack_top,
        };
        backend_owned_directly = false;
        stack_owned_directly = false;
        var record_owned_locally = true;
        defer if (record_owned_locally) self.destroyRecord(record);

        self.mutex.lock();
        if (self.shutting_down) {
            self.mutex.unlock();
            return error.ThreadGroupShuttingDown;
        }

        const tid = self.publishRecordLocked(record) catch |err| {
            self.mutex.unlock();
            return err;
        };
        record.tid = tid;
        backend_ops.configure(
            backend_context,
            self,
            tid,
            @bitCast(start_arg),
            if (aux_stack_top) |top|
                execution_context.AuxiliaryStack.fromTop(top, self.aux_stack_pool.stack_size)
            else
                null,
        ) catch |err| {
            self.rollbackPublishedLocked(tid);
            self.mutex.unlock();
            return err;
        };

        const fail_native_spawn = if (self.test_hooks) |hooks|
            hooks.fail_native_spawn
        else
            false;
        const maybe_thread: ?std.Thread = if (fail_native_spawn)
            null
        else
            std.Thread.spawn(.{}, threadEntry, .{record}) catch null;
        const thread = maybe_thread orelse {
            self.rollbackPublishedLocked(tid);
            self.mutex.unlock();
            return error.ThreadSpawnFailed;
        };
        record.thread = thread;

        const fail_start_gate = if (self.test_hooks) |hooks|
            hooks.fail_start_gate
        else
            false;
        if (fail_start_gate) {
            self.rollbackPublishedLocked(tid);
            record.start_gate.abort();
            record.thread = null;
            self.mutex.unlock();
            thread.join();
            self.noteNativeJoin();
            return error.StartGateFailed;
        }

        record.start_gate.open();
        self.mutex.unlock();
        record_owned_locally = false;
        return tid;
    }

    fn publishRecordLocked(self: *ThreadManager, record: *ThreadRecord) error{
        OutOfMemory,
        ThreadIdExhausted,
    }!i32 {
        for (self.slots.items, 0..) |*slot, slot_index| {
            if (slot.state != .free) continue;
            if (slot.generation == max_tid_generation) {
                slot.state = .retired;
                continue;
            }
            slot.generation += 1;
            slot.state = .live;
            slot.record = record;
            return makeTid(slot_index, slot.generation);
        }

        if (self.slots.items.len == max_thread_slots)
            return error.ThreadIdExhausted;
        const slot_index = self.slots.items.len;
        try self.slots.append(self.allocator, .{
            .generation = 0,
            .state = .live,
            .record = record,
        });
        return makeTid(slot_index, 0);
    }

    fn beginSpawn(self: *ThreadManager) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.shutting_down) return false;
        self.spawning_count += 1;
        return true;
    }

    fn endSpawn(self: *ThreadManager) void {
        self.mutex.lock();
        std.debug.assert(self.spawning_count > 0);
        self.spawning_count -= 1;
        self.mutex.unlock();
    }

    fn waitForSpawnsToDrain(self: *ThreadManager) void {
        while (true) {
            self.mutex.lock();
            const spawning = self.spawning_count;
            self.mutex.unlock();
            if (spawning == 0) return;
            yieldForLifecycle();
        }
    }

    fn rollbackPublishedLocked(self: *ThreadManager, tid: i32) void {
        const parsed = parseTid(tid).?;
        const slot = &self.slots.items[parsed.slot_index];
        std.debug.assert(slot.generation == parsed.generation);
        std.debug.assert(slot.state == .live);
        slot.record = null;
        slot.state = .free;
    }

    fn claimOneLocked(self: *ThreadManager, tid: i32) JoinError!JoinClaim {
        const parsed = parseTid(tid) orelse return error.InvalidThreadId;
        if (parsed.slot_index >= self.slots.items.len) return error.UnknownThread;
        const slot = &self.slots.items[parsed.slot_index];
        if (slot.generation != parsed.generation) return error.StaleThreadId;
        switch (slot.state) {
            .joining => return error.ThreadAlreadyJoining,
            .free, .retired => return error.ThreadAlreadyJoined,
            .live => {
                if (self.shutting_down) return error.ThreadGroupShuttingDown;
                slot.state = .joining;
                self.joining_count += 1;
                return .{
                    .slot_index = parsed.slot_index,
                    .generation = parsed.generation,
                    .record = slot.record.?,
                };
            },
        }
    }

    fn joinClaimed(self: *ThreadManager, claim: JoinClaim) ThreadOutcome {
        const record = claim.record;
        const thread = record.thread orelse unreachable;
        record.thread = null;
        thread.join();
        self.noteNativeJoin();

        const outcome: ThreadOutcome = switch (executionState(record)) {
            .completed => .completed,
            .trapped => .trapped,
            .pending, .start_aborted => unreachable,
        };

        self.mutex.lock();
        var slot = &self.slots.items[claim.slot_index];
        std.debug.assert(slot.generation == claim.generation);
        std.debug.assert(slot.state == .joining);
        std.debug.assert(slot.record == record);
        slot.record = null;
        self.mutex.unlock();

        self.destroyRecord(record);

        self.mutex.lock();
        slot = &self.slots.items[claim.slot_index];
        std.debug.assert(slot.generation == claim.generation);
        std.debug.assert(slot.state == .joining);
        std.debug.assert(slot.record == null);
        slot.state = .free;
        std.debug.assert(self.joining_count > 0);
        self.joining_count -= 1;
        self.mutex.unlock();
        return outcome;
    }

    fn destroyRecord(self: *ThreadManager, record: *ThreadRecord) void {
        record.backend_ops.destroy(record.backend_context);
        if (record.aux_stack_top) |stack_top| self.aux_stack_pool.release(stack_top);
        self.noteCounter(if (self.test_hooks) |hooks| hooks.records_destroyed else null);
        self.allocator.destroy(record);
    }

    fn noteNativeJoin(self: *ThreadManager) void {
        self.noteCounter(if (self.test_hooks) |hooks| hooks.native_threads_joined else null);
    }

    fn noteThreadStarted(self: *ThreadManager) void {
        self.noteCounter(if (self.test_hooks) |hooks| hooks.native_threads_started else null);
    }

    fn noteCounter(_: *ThreadManager, counter: ?*std.atomic.Value(usize)) void {
        if (counter) |value| _ = value.fetchAdd(1, .monotonic);
    }
};

/// The legacy ABI only specifies that failures are negative. wasi-libc maps
/// every negative result to `EAGAIN`, and the existing host surface returned
/// `-1`, so keep that stable for every internal failure class.
pub fn spawnFailureResult(_: SpawnError) i32 {
    return -1;
}

fn isWasiThreadStartType(func_type: types.FuncType) bool {
    return func_type.params.len == 2 and
        func_type.params[0] == .i32 and
        func_type.params[1] == .i32 and
        func_type.results.len == 0;
}

fn makeTid(slot_index: usize, generation: u16) i32 {
    std.debug.assert(slot_index < max_thread_slots);
    std.debug.assert(generation <= max_tid_generation);
    const raw = (@as(u32, generation) << tid_slot_bits) |
        (@as(u32, @intCast(slot_index)) + 1);
    std.debug.assert(raw < (1 << 29));
    return @intCast(raw);
}

fn parseTid(tid: i32) ?ParsedTid {
    if (tid <= 0) return null;
    const raw: u32 = @intCast(tid);
    if (raw >= (1 << 29)) return null;
    const encoded_slot = raw & tid_slot_mask;
    if (encoded_slot == 0) return null;
    return .{
        .slot_index = @intCast(encoded_slot - 1),
        .generation = @intCast(raw >> tid_slot_bits),
    };
}

fn executionState(record: *const ThreadRecord) ExecutionState {
    return @enumFromInt(record.execution.load(.acquire));
}

fn yieldForLifecycle() void {
    std.Thread.yield() catch std.atomic.spinLoopHint();
}

fn monotonicNowNs() u64 {
    return std.math.mul(u64, platform.timeGetBootUs(), std.time.ns_per_us) catch
        std.math.maxInt(u64);
}

fn threadEntry(record: *ThreadRecord) void {
    const manager = record.manager;
    manager.noteThreadStarted();
    if (!record.start_gate.wait()) {
        record.execution.store(@intFromEnum(ExecutionState.start_aborted), .release);
        return;
    }

    // The gate opens while publication still holds the manager lock. Taking
    // that lock once guarantees guest code cannot begin until publication ends.
    manager.mutex.lock();
    manager.mutex.unlock();

    const outcome = record.backend_ops.run(record.backend_context);
    if (outcome == .trapped) {
        manager.signalTrap();
        record.execution.store(@intFromEnum(ExecutionState.trapped), .release);
        return;
    }
    record.execution.store(@intFromEnum(ExecutionState.completed), .release);
}

// ── Tests ────────────────────────────────────────────────────────────────

test "ThreadManager: TIDs encode slot generations within the spec range" {
    const first = makeTid(0, 0);
    const reused = makeTid(0, 1);
    const last = makeTid(max_thread_slots - 1, max_tid_generation);

    try std.testing.expectEqual(@as(i32, 1), first);
    try std.testing.expect(reused != first);
    try std.testing.expect(last > 0);
    try std.testing.expect(last < (1 << 29));
    try std.testing.expectEqual(@as(usize, 0), parseTid(reused).?.slot_index);
    try std.testing.expectEqual(@as(u16, 1), parseTid(reused).?.generation);
    try std.testing.expect(parseTid(0) == null);
    try std.testing.expect(parseTid(-1) == null);
}

test "ThreadManager: trap flag" {
    var tm = ThreadManager.init(std.testing.allocator);
    defer tm.deinit();

    try std.testing.expect(!tm.hasTrap());
    tm.signalTrap();
    try std.testing.expect(tm.hasTrap());
}

test "ModuleInstance: cloneForThread shares memory" {
    const allocator = std.testing.allocator;

    // Create a parent module with shared memory
    var module = types.WasmModule{};
    const mem_inst = try types.MemoryInstance.createShared(.{
        .limits = .{ .min = 1, .max = 4 },
        .is_shared = true,
    }, allocator);
    defer mem_inst.release(allocator);
    mem_inst.data[0] = 42;

    var mem_ptrs = [_]*types.MemoryInstance{mem_inst};
    var globals = [_]*types.GlobalInstance{};
    var parent = try allocator.create(types.ModuleInstance);
    parent.* = .{
        .module = &module,
        .memories = &mem_ptrs,
        .tables = &.{},
        .globals = &globals,
        .allocator = allocator,
    };
    // Don't destroy parent — we manage manually

    // Clone for a child thread
    const child = try parent.cloneForThread(allocator);
    defer child.destroyThreadClone();

    // Child should see the same memory
    try std.testing.expectEqual(@as(u8, 42), child.memories[0].data[0]);

    // Write through child — parent should see it (shared)
    child.memories[0].data[1] = 99;
    try std.testing.expectEqual(@as(u8, 99), parent.memories[0].data[1]);

    // Verify ref count was incremented
    try std.testing.expectEqual(@as(u32, 2), mem_inst.shared_control.?.referenceCount());

    allocator.destroy(parent);
}

test "AuxStackPool: allocate and release" {
    const allocator = std.testing.allocator;

    var pool = AuxStackPool{};
    try pool.init(4, 0, allocator);
    defer pool.deinit(allocator);

    // Should allocate 4 stacks
    const s1 = pool.allocate();
    const s2 = pool.allocate();
    const s3 = pool.allocate();
    const s4 = pool.allocate();
    try std.testing.expect(s1 != null);
    try std.testing.expect(s2 != null);
    try std.testing.expect(s3 != null);
    try std.testing.expect(s4 != null);

    // Pool is empty now
    try std.testing.expect(pool.allocate() == null);

    // Release one, then allocate again
    pool.release(s1.?);
    const s5 = pool.allocate();
    try std.testing.expect(s5 != null);
    try std.testing.expectEqual(s1.?, s5.?);
    pool.release(s2.?);
    pool.release(s3.?);
    pool.release(s4.?);
    pool.release(s5.?);
    try std.testing.expectEqual(@as(usize, 4), pool.availableCount());
}

test "AuxStackPool: stack addresses are correct" {
    const allocator = std.testing.allocator;

    var pool = AuxStackPool{ .stack_size = 1024 };
    try pool.init(3, 4096, allocator);
    defer pool.deinit(allocator);

    // Allocate all 3 — they are returned in LIFO order
    const s1 = pool.allocate().?;
    const s2 = pool.allocate().?;
    const s3 = pool.allocate().?;

    // Stack tops should be base + (i+1)*stack_size
    // But LIFO: last pushed = first popped, so s1 is the last one pushed = 4096+3*1024
    try std.testing.expect(s1 == 4096 + 3 * 1024);
    try std.testing.expect(s2 == 4096 + 2 * 1024);
    try std.testing.expect(s3 == 4096 + 1 * 1024);
    pool.release(s1);
    pool.release(s2);
    pool.release(s3);
}

// ── Integration tests ───────────────────────────────────────────────────────
// These tests exercise the full thread lifecycle: spawn → execute → join.
// Each builds a WasmModule with an exported wasi_thread_start function,
// creates a ModuleInstance with shared memory, and spawns real threads.
// Gated on multi-threaded targets (can't spawn threads on wasm32-wasi).

const ThreadTestCtx = struct {
    module: *types.WasmModule,
    mem_inst: *types.MemoryInstance,
    inst: *types.ModuleInstance,
};

/// Build a test module with a wasi_thread_start export.
/// `code` is the function body bytecode (excluding the end opcode).
fn buildThreadTestModule(
    func_code: []const u8,
    allocator: std.mem.Allocator,
) !ThreadTestCtx {
    return buildThreadTestModuleWithExport(func_code, "wasi_thread_start", allocator);
}

fn buildThreadTestModuleWithExport(
    func_code: []const u8,
    export_name: []const u8,
    allocator: std.mem.Allocator,
) !ThreadTestCtx {
    const module = try allocator.create(types.WasmModule);

    // Build code with end opcode
    const full_code = try allocator.alloc(u8, func_code.len + 1);
    @memcpy(full_code[0..func_code.len], func_code);
    full_code[func_code.len] = 0x0B; // end

    const func_types = try allocator.alloc(types.FuncType, 1);
    func_types[0] = .{
        .params = &.{ .i32, .i32 },
        .results = &.{},
    };
    const functions = try allocator.alloc(types.WasmFunction, 1);
    functions[0] = .{
        .type_idx = 0,
        .func_type = func_types[0],
        .local_count = 2,
        .locals = &.{},
        .code = full_code,
    };
    const exports = try allocator.alloc(types.ExportDesc, 1);
    exports[0] = .{
        .name = export_name,
        .kind = .function,
        .index = 0,
    };
    const memories = try allocator.alloc(types.MemoryType, 1);
    memories[0] = .{ .limits = .{ .min = 1, .max = 4 }, .is_shared = true };

    module.* = .{
        .types = func_types,
        .functions = functions,
        .exports = exports,
        .memories = memories,
    };

    // Create shared memory instance
    const mem_inst = try types.MemoryInstance.createShared(.{
        .limits = .{ .min = 1, .max = 4 },
        .is_shared = true,
    }, allocator);
    var mem_ptrs = try allocator.alloc(*types.MemoryInstance, 1);
    mem_ptrs[0] = mem_inst;

    const inst = try allocator.create(types.ModuleInstance);
    inst.* = .{
        .module = module,
        .memories = mem_ptrs,
        .tables = &.{},
        .globals = &.{},
        .allocator = allocator,
    };

    return .{ .module = module, .mem_inst = mem_inst, .inst = inst };
}

fn cleanupThreadTest(
    ctx: ThreadTestCtx,
    allocator: std.mem.Allocator,
) void {
    // Memory may have been retained by clones — just release our ref
    ctx.mem_inst.release(allocator);
    allocator.free(ctx.inst.memories);
    allocator.destroy(ctx.inst);
    allocator.free(@constCast(ctx.module.functions[0].code));
    allocator.free(ctx.module.types);
    allocator.free(ctx.module.functions);
    allocator.free(ctx.module.exports);
    allocator.free(ctx.module.memories);
    allocator.destroy(ctx.module);
}

const nop_thread_code = [_]u8{0x01};
const increment_thread_code = [_]u8{
    0x41, 0x00,
    0x41, 0x01,
    0xFE, 0x1E,
    0x02, 0x00,
    0x1A,
};
const add_start_arg_thread_code = [_]u8{
    0x41, 0x00,
    0x20, 0x01,
    0xFE, 0x1E,
    0x02, 0x00,
    0x1A,
};

/// Publish readiness at address 0, then park forever on address 16.
/// Only group termination can release this thread.
const futex_wait_thread_code = [_]u8{
    0x41, 0x00, // i32.const 0
    0x41, 0x01, // i32.const 1
    0xFE, 0x1E, 0x02, 0x00, // i32.atomic.rmw.add
    0x1A, // drop
    0x41, 0x10, // i32.const 16
    0x41, 0x00, // i32.const 0 (expected)
    0x42, 0x7F, // i64.const -1 (no timeout)
    0xFE, 0x01, 0x02, 0x00, // memory.atomic.wait32
    0x1A, // drop
};

/// Publish readiness, then spin forever. Only the interpreter's cross-thread
/// interrupt poll can stop this thread.
const spin_forever_thread_code = [_]u8{
    0x41, 0x00, // i32.const 0
    0x41, 0x01, // i32.const 1
    0xFE, 0x1E, 0x02, 0x00, // i32.atomic.rmw.add
    0x1A, // drop
    0x03, 0x40, // loop
    0x0C, 0x00, // br 0
    0x0B, // end
};

fn requireThreadLifecycle() !void {
    if (builtin.single_threaded or !config.lib_wasi_threads or !config.thread_mgr)
        return error.SkipZigTest;
}

fn waitForCompleted(manager: *ThreadManager, expected: usize) !void {
    var attempts: usize = 0;
    while (attempts < 1_000_000) : (attempts += 1) {
        const current = manager.stats();
        if (current.active == 0 and current.completed == expected) return;
        yieldForLifecycle();
    }
    return error.ThreadCompletionTimeout;
}

fn atomicCount(value: *const std.atomic.Value(usize)) usize {
    return value.load(.acquire);
}

test "thread lifecycle: disabled manager rejects spawn without publishing a record" {
    if (config.lib_wasi_threads) return error.SkipZigTest;
    var manager = ThreadManager.init(std.testing.allocator);
    defer manager.deinit();

    const unused_parent: *types.ModuleInstance = undefined;
    try std.testing.expectError(
        error.ThreadFeatureDisabled,
        manager.spawnThread(unused_parent, 0),
    );
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
}

test "thread lifecycle: immediate exits retain more than 256 handles until batched joinAll" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const child_count = 300;
    const ctx = try buildThreadTestModule(&nop_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var started = std.atomic.Value(usize).init(0);
    var joined = std.atomic.Value(usize).init(0);
    var destroyed = std.atomic.Value(usize).init(0);
    const hooks = TestHooks{
        .native_threads_started = &started,
        .native_threads_joined = &joined,
        .records_destroyed = &destroyed,
    };
    var manager = ThreadManager.initWithTestHooks(allocator, &hooks);
    defer manager.deinit();
    ctx.inst.thread_manager = &manager;

    for (0..child_count) |_| _ = try manager.spawnThread(ctx.inst, 0);
    try waitForCompleted(&manager, child_count);
    const before_join = manager.stats();
    try std.testing.expectEqual(@as(usize, 0), before_join.active);
    try std.testing.expectEqual(@as(usize, child_count), before_join.completed);
    try std.testing.expectEqual(@as(usize, child_count), before_join.retained);
    try std.testing.expectEqual(@as(usize, 0), atomicCount(&joined));
    try std.testing.expectEqual(@as(usize, 0), atomicCount(&destroyed));

    const summary = manager.joinAllWithSummary();
    try std.testing.expectEqual(@as(usize, child_count), summary.joined);
    try std.testing.expectEqual(@as(usize, 0), summary.trapped);
    try std.testing.expectEqual(@as(usize, child_count), atomicCount(&started));
    try std.testing.expectEqual(@as(usize, child_count), atomicCount(&joined));
    try std.testing.expectEqual(@as(usize, child_count), atomicCount(&destroyed));
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
}

test "thread lifecycle: joinOne is exact and reused slots reject stale generations" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&nop_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    ctx.inst.thread_manager = &manager;

    try std.testing.expectError(error.InvalidThreadId, manager.joinOne(-1));
    try std.testing.expectError(error.UnknownThread, manager.joinOne(makeTid(5, 0)));

    const first_tid = try manager.spawnThread(ctx.inst, 0);
    try waitForCompleted(&manager, 1);
    try std.testing.expectEqual(
        @as(?ThreadOutcome, .completed),
        try manager.threadOutcome(first_tid),
    );
    try std.testing.expectEqual(ThreadOutcome.completed, try manager.joinOne(first_tid));
    try std.testing.expectError(error.ThreadAlreadyJoined, manager.joinOne(first_tid));

    const second_tid = try manager.spawnThread(ctx.inst, 0);
    try std.testing.expect(second_tid != first_tid);
    try std.testing.expectError(error.StaleThreadId, manager.joinOne(first_tid));
    try std.testing.expectEqual(ThreadOutcome.completed, try manager.joinOne(second_tid));
}

test "thread lifecycle: child traps report an outcome and missing exports fail synchronously" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const trap_code = [_]u8{0x00};
    const trap_ctx = try buildThreadTestModule(&trap_code, allocator);
    defer cleanupThreadTest(trap_ctx, allocator);
    const missing_ctx = try buildThreadTestModuleWithExport(
        &nop_thread_code,
        "not_wasi_thread_start",
        allocator,
    );
    defer cleanupThreadTest(missing_ctx, allocator);
    const invalid_signature_ctx = try buildThreadTestModule(&nop_thread_code, allocator);
    defer cleanupThreadTest(invalid_signature_ctx, allocator);
    @constCast(invalid_signature_ctx.module.types)[0] = .{
        .params = &.{.i32},
        .results = &.{},
    };

    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    trap_ctx.inst.thread_manager = &manager;
    missing_ctx.inst.thread_manager = &manager;
    invalid_signature_ctx.inst.thread_manager = &manager;

    const trap_tid = try manager.spawnThread(trap_ctx.inst, 0);
    try std.testing.expectEqual(ThreadOutcome.trapped, try manager.joinOne(trap_tid));
    try std.testing.expect(manager.hasTrap());
    try std.testing.expectError(
        error.MissingThreadStart,
        manager.spawnThread(missing_ctx.inst, 0),
    );
    try std.testing.expectError(
        error.InvalidThreadStartSignature,
        manager.spawnThread(invalid_signature_ctx.inst, 0),
    );
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
}

test "ThreadManager: every spawn failure maps to the stable negative ABI result" {
    const failures = [_]SpawnError{
        error.ThreadFeatureDisabled,
        error.ThreadGroupShuttingDown,
        error.ThreadIdExhausted,
        error.MissingThreadStart,
        error.InvalidThreadStartSignature,
        error.AuxStackExhausted,
        error.ChildInitializationFailed,
        error.ThreadSpawnFailed,
        error.StartGateFailed,
        error.OutOfMemory,
    };
    for (failures) |failure| {
        try std.testing.expectEqual(@as(i32, -1), spawnFailureResult(failure));
    }
}

test "thread lifecycle: child initialization failure returns stack and clone ownership" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&nop_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);
    const hooks = TestHooks{ .fail_child_initialization = true };
    var manager = ThreadManager.initWithTestHooks(allocator, &hooks);
    defer manager.deinit();
    ctx.inst.thread_manager = &manager;
    try manager.aux_stack_pool.init(1, 32768, allocator);

    try std.testing.expectError(
        error.ChildInitializationFailed,
        manager.spawnThread(ctx.inst, 0),
    );
    try std.testing.expectEqual(@as(usize, 1), manager.aux_stack_pool.availableCount());
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
    try std.testing.expectEqual(@as(usize, 1), ctx.mem_inst.referenceCount());
}

test "thread lifecycle: auxiliary stack exhaustion is reversible" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&nop_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);
    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    ctx.inst.thread_manager = &manager;
    try manager.aux_stack_pool.init(1, 32768, allocator);

    const tid = try manager.spawnThread(ctx.inst, 0);
    try std.testing.expectError(
        error.AuxStackExhausted,
        manager.spawnThread(ctx.inst, 0),
    );
    try std.testing.expectEqual(@as(usize, 1), manager.retainedCount());
    try std.testing.expectEqual(ThreadOutcome.completed, try manager.joinOne(tid));
    try std.testing.expectEqual(@as(usize, 1), manager.aux_stack_pool.availableCount());
    try std.testing.expectEqual(@as(usize, 1), ctx.mem_inst.referenceCount());
}

test "thread lifecycle: injected native spawn failure rolls back the published record" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&nop_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);
    var started = std.atomic.Value(usize).init(0);
    var joined = std.atomic.Value(usize).init(0);
    var destroyed = std.atomic.Value(usize).init(0);
    const hooks = TestHooks{
        .fail_native_spawn = true,
        .native_threads_started = &started,
        .native_threads_joined = &joined,
        .records_destroyed = &destroyed,
    };
    var manager = ThreadManager.initWithTestHooks(allocator, &hooks);
    defer manager.deinit();
    ctx.inst.thread_manager = &manager;
    try manager.aux_stack_pool.init(1, 32768, allocator);

    try std.testing.expectError(
        error.ThreadSpawnFailed,
        manager.spawnThread(ctx.inst, 0),
    );
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
    try std.testing.expectEqual(@as(usize, 1), manager.aux_stack_pool.availableCount());
    try std.testing.expectEqual(@as(usize, 0), atomicCount(&started));
    try std.testing.expectEqual(@as(usize, 0), atomicCount(&joined));
    try std.testing.expectEqual(@as(usize, 1), atomicCount(&destroyed));
    try std.testing.expectEqual(@as(usize, 1), ctx.mem_inst.referenceCount());
}

test "thread lifecycle: injected start gate failure aborts and joins before cleanup" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&nop_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);
    var started = std.atomic.Value(usize).init(0);
    var joined = std.atomic.Value(usize).init(0);
    var destroyed = std.atomic.Value(usize).init(0);
    const hooks = TestHooks{
        .fail_start_gate = true,
        .native_threads_started = &started,
        .native_threads_joined = &joined,
        .records_destroyed = &destroyed,
    };
    var manager = ThreadManager.initWithTestHooks(allocator, &hooks);
    defer manager.deinit();
    ctx.inst.thread_manager = &manager;
    try manager.aux_stack_pool.init(1, 32768, allocator);

    try std.testing.expectError(
        error.StartGateFailed,
        manager.spawnThread(ctx.inst, 0),
    );
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
    try std.testing.expectEqual(@as(usize, 1), manager.aux_stack_pool.availableCount());
    try std.testing.expectEqual(@as(usize, 1), atomicCount(&started));
    try std.testing.expectEqual(@as(usize, 1), atomicCount(&joined));
    try std.testing.expectEqual(@as(usize, 1), atomicCount(&destroyed));
    try std.testing.expectEqual(@as(usize, 1), ctx.mem_inst.referenceCount());
}

fn exerciseSpawnAllocationRollback(
    allocator: std.mem.Allocator,
    parent: *types.ModuleInstance,
) !void {
    const hooks = TestHooks{ .fail_native_spawn = true };
    var manager = ThreadManager.initWithTestHooks(allocator, &hooks);
    defer manager.deinit();
    parent.thread_manager = &manager;
    defer parent.thread_manager = null;
    try manager.aux_stack_pool.init(1, 32768, allocator);

    if (manager.spawnThread(parent, 0)) |_| {
        return error.ExpectedNativeSpawnFailure;
    } else |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.ThreadSpawnFailed => {},
        else => return err,
    }
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
    try std.testing.expectEqual(@as(usize, 1), manager.aux_stack_pool.availableCount());
}

test "thread lifecycle: every allocation failure rolls back clone stack env record and slot" {
    try requireThreadLifecycle();
    const ctx = try buildThreadTestModule(&nop_thread_code, std.testing.allocator);
    defer cleanupThreadTest(ctx, std.testing.allocator);

    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        exerciseSpawnAllocationRollback,
        .{ctx.inst},
    );
    try std.testing.expectEqual(@as(usize, 1), ctx.mem_inst.referenceCount());
}

const ConcurrentLifecycleCtx = struct {
    manager: *ThreadManager,
    parent: *types.ModuleInstance,
    iterations: usize,
    failed: *std.atomic.Value(bool),
};

fn concurrentSpawnAndJoin(ctx: *const ConcurrentLifecycleCtx) void {
    for (0..ctx.iterations) |_| {
        const tid = ctx.manager.spawnThread(ctx.parent, 0) catch {
            ctx.failed.store(true, .release);
            return;
        };
        const outcome = ctx.manager.joinOne(tid) catch {
            ctx.failed.store(true, .release);
            return;
        };
        if (outcome != .completed) {
            ctx.failed.store(true, .release);
            return;
        }
    }
}

test "thread lifecycle: concurrent spawn and exact joins destroy every resource once" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const worker_count = 4;
    const iterations = 32;
    const expected = worker_count * iterations;
    const ctx = try buildThreadTestModule(&increment_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);
    var started = std.atomic.Value(usize).init(0);
    var joined = std.atomic.Value(usize).init(0);
    var destroyed = std.atomic.Value(usize).init(0);
    var failed = std.atomic.Value(bool).init(false);
    const hooks = TestHooks{
        .native_threads_started = &started,
        .native_threads_joined = &joined,
        .records_destroyed = &destroyed,
    };
    var manager = ThreadManager.initWithTestHooks(allocator, &hooks);
    defer manager.deinit();
    ctx.inst.thread_manager = &manager;
    const worker_ctx = ConcurrentLifecycleCtx{
        .manager = &manager,
        .parent = ctx.inst,
        .iterations = iterations,
        .failed = &failed,
    };

    var workers: [worker_count]std.Thread = undefined;
    for (&workers) |*worker| {
        worker.* = try std.Thread.spawn(.{}, concurrentSpawnAndJoin, .{&worker_ctx});
    }
    for (workers) |worker| worker.join();

    try std.testing.expect(!failed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
    try std.testing.expectEqual(@as(usize, expected), atomicCount(&started));
    try std.testing.expectEqual(@as(usize, expected), atomicCount(&joined));
    try std.testing.expectEqual(@as(usize, expected), atomicCount(&destroyed));
    const counter = std.mem.readInt(u32, ctx.mem_inst.data[0..4], .little);
    try std.testing.expectEqual(@as(u32, expected), counter);
}

test "thread lifecycle: start_arg contract remains intact" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&add_start_arg_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);
    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    ctx.inst.thread_manager = &manager;

    const first = try manager.spawnThread(ctx.inst, 10);
    const second = try manager.spawnThread(ctx.inst, 20);
    try std.testing.expectEqual(ThreadOutcome.completed, try manager.joinOne(first));
    try std.testing.expectEqual(ThreadOutcome.completed, try manager.joinOne(second));
    const counter = std.mem.readInt(u32, ctx.mem_inst.data[0..4], .little);
    try std.testing.expectEqual(@as(u32, 30), counter);
}

test "thread lifecycle: child record retains process state after parent release" {
    try requireThreadLifecycle();
    const Tracker = struct {
        refs: usize = 1,

        fn retain(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.refs += 1;
        }

        fn release(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            std.debug.assert(self.refs > 0);
            self.refs -= 1;
        }
    };
    const ops = execution_context.ProcessStateOps{
        .retain = Tracker.retain,
        .release = Tracker.release,
    };
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&nop_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);
    var tracker = Tracker{};
    const root_ref = execution_context.ProcessStateRef.init(@ptrCast(&tracker), &ops);
    ctx.inst.attachProcessState(root_ref);

    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    ctx.inst.thread_manager = &manager;
    try manager.aux_stack_pool.init(1, 32768, allocator);

    const start_arg: u32 = 0xF1234567;
    const tid = try manager.spawnThread(ctx.inst, @bitCast(start_arg));
    try waitForCompleted(&manager, 1);
    try std.testing.expectEqual(@as(usize, 4), tracker.refs);

    const parsed = parseTid(tid).?;
    const record = manager.slots.items[parsed.slot_index].record.?;
    const child: *InterpThreadContext =
        @ptrCast(@alignCast(record.backend_context));
    try std.testing.expectEqual(
        @as(*Tracker, @ptrCast(@alignCast(child.env.thread_context.process_state.?.ptr))),
        &tracker,
    );
    try std.testing.expectEqual(tid, child.env.thread_context.tid);
    try std.testing.expectEqual(start_arg, child.env.thread_context.start_arg);
    try std.testing.expectEqual(@as(u32, 32768), child.env.thread_context.auxiliary_stack.?.bottom);
    try std.testing.expectEqual(@as(u32, 40960), child.env.thread_context.auxiliary_stack.?.top);
    try std.testing.expect(child.env.thread_context.tls_base == null);

    ctx.inst.detachProcessState();
    root_ref.release();
    try std.testing.expectEqual(@as(usize, 2), tracker.refs);
    try std.testing.expectEqual(ThreadOutcome.completed, try manager.joinOne(tid));
    try std.testing.expectEqual(@as(usize, 0), tracker.refs);
}

test "thread lifecycle: parent teardown owns and joins all unclaimed records" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const child_count = 48;
    const ctx = try buildThreadTestModule(&nop_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);
    var started = std.atomic.Value(usize).init(0);
    var joined = std.atomic.Value(usize).init(0);
    var destroyed = std.atomic.Value(usize).init(0);
    const hooks = TestHooks{
        .native_threads_started = &started,
        .native_threads_joined = &joined,
        .records_destroyed = &destroyed,
    };
    var manager = ThreadManager.initWithTestHooks(allocator, &hooks);
    var manager_live = true;
    defer if (manager_live) manager.deinit();
    ctx.inst.thread_manager = &manager;

    for (0..child_count) |_| _ = try manager.spawnThread(ctx.inst, 0);
    manager.deinit();
    manager_live = false;
    ctx.inst.thread_manager = null;

    try std.testing.expectEqual(@as(usize, child_count), atomicCount(&started));
    try std.testing.expectEqual(@as(usize, child_count), atomicCount(&joined));
    try std.testing.expectEqual(@as(usize, child_count), atomicCount(&destroyed));
    try std.testing.expectEqual(@as(usize, 1), ctx.mem_inst.referenceCount());
}

const ShutdownSpawnCtx = struct {
    manager: *ThreadManager,
    parent: *types.ModuleInstance,
    succeeded: *std.atomic.Value(usize),
    failed: *std.atomic.Value(bool),
};

fn spawnUntilShutdown(ctx: *const ShutdownSpawnCtx) void {
    for (0..128) |_| {
        _ = ctx.manager.spawnThread(ctx.parent, 0) catch |err| switch (err) {
            error.ThreadGroupShuttingDown => return,
            else => {
                ctx.failed.store(true, .release);
                return;
            },
        };
        _ = ctx.succeeded.fetchAdd(1, .release);
    }
}

test "thread lifecycle: shutdown drains concurrent spawn publication before teardown" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&nop_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);
    var succeeded = std.atomic.Value(usize).init(0);
    var failed = std.atomic.Value(bool).init(false);
    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    ctx.inst.thread_manager = &manager;
    const spawn_ctx = ShutdownSpawnCtx{
        .manager = &manager,
        .parent = ctx.inst,
        .succeeded = &succeeded,
        .failed = &failed,
    };

    const spawner = try std.Thread.spawn(.{}, spawnUntilShutdown, .{&spawn_ctx});
    while (succeeded.load(.acquire) == 0 and !failed.load(.acquire))
        yieldForLifecycle();
    const summary = manager.shutdownWithSummary();
    spawner.join();

    try std.testing.expect(!failed.load(.acquire));
    try std.testing.expectEqual(succeeded.load(.acquire), summary.joined);
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
}

test "thread lifecycle: shutdown closes group ownership without detaching children" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const child_count = 24;
    const ctx = try buildThreadTestModule(&nop_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);
    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    ctx.inst.thread_manager = &manager;

    const first_tid = try manager.spawnThread(ctx.inst, 0);
    for (1..child_count) |_| _ = try manager.spawnThread(ctx.inst, 0);
    const summary = manager.shutdownWithSummary();
    try std.testing.expectEqual(@as(usize, child_count), summary.joined);
    try std.testing.expect(manager.isShuttingDown());
    try std.testing.expectError(error.ThreadAlreadyJoined, manager.joinOne(first_tid));
    try std.testing.expectError(
        error.ThreadGroupShuttingDown,
        manager.spawnThread(ctx.inst, 0),
    );
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
}

// ── First-wins group termination (#616) ─────────────────────────────────

fn readyCount(ctx: ThreadTestCtx) u32 {
    const cell: *align(4) const u32 = @ptrCast(@alignCast(ctx.mem_inst.data[0..4]));
    return @atomicLoad(u32, cell, .acquire);
}

fn waitForReady(ctx: ThreadTestCtx, expected: u32) !void {
    const deadline = monotonicNowNs() +| (10 * std.time.ns_per_s);
    while (readyCount(ctx) < expected) {
        if (monotonicNowNs() >= deadline) return error.ThreadReadinessTimeout;
        yieldForLifecycle();
    }
}

test "group termination: the winning proc_exit is not masked by later terminations" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&futex_wait_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var state = termination.State{};
    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    try manager.bindTermination(&state);
    try manager.prepareSharedMemory(ctx.mem_inst, null);
    ctx.inst.thread_manager = &manager;

    const child_count = 4;
    for (0..child_count) |_| _ = try manager.spawnThread(ctx.inst, 0);
    try waitForReady(ctx, child_count);

    try std.testing.expect(state.claimExit(9));
    // A racing trap loses; the embedder still observes the exit status.
    manager.signalTrap();
    try std.testing.expectEqual(@as(?u32, 9), state.exitCode());
    try std.testing.expectEqual(
        termination.Kind.exit,
        manager.terminalOutcome().?.kind,
    );

    const summary = manager.terminateAndJoin(default_termination_timeout_ns);
    try std.testing.expect(!summary.timed_out);
    try std.testing.expectEqual(@as(usize, 0), summary.unfinished);
    try std.testing.expectEqual(@as(usize, child_count), summary.joined);
    try std.testing.expectEqual(@as(usize, child_count), summary.trapped);
    try std.testing.expectEqual(@as(?u32, 9), state.exitCode());
}

test "group termination: a winning trap is never overwritten by a later proc_exit" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const trap_code = [_]u8{0x00};
    const ctx = try buildThreadTestModule(&trap_code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var state = termination.State{};
    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    try manager.bindTermination(&state);
    ctx.inst.thread_manager = &manager;

    const tid = try manager.spawnThread(ctx.inst, 0);
    try std.testing.expectEqual(ThreadOutcome.trapped, try manager.joinOne(tid));
    try std.testing.expectEqual(
        termination.Kind.trap,
        state.outcome().?.kind,
    );

    // `proc_exit(0)` arriving after the trap must not report success.
    try std.testing.expect(!state.claimExit(0));
    try std.testing.expect(state.exitCode() == null);
}

test "group termination: futex waiters wake and join within the teardown bound" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&futex_wait_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var destroyed = std.atomic.Value(usize).init(0);
    var joined = std.atomic.Value(usize).init(0);
    const hooks = TestHooks{
        .native_threads_joined = &joined,
        .records_destroyed = &destroyed,
    };
    var state = termination.State{};
    var manager = ThreadManager.initWithTestHooks(allocator, &hooks);
    defer manager.deinit();
    try manager.bindTermination(&state);
    try manager.prepareSharedMemory(ctx.mem_inst, null);
    ctx.inst.thread_manager = &manager;

    const child_count = 6;
    for (0..child_count) |_| _ = try manager.spawnThread(ctx.inst, 0);
    try waitForReady(ctx, child_count);
    // Every child is parked on the guest futex; nothing will ever notify it.
    try std.testing.expectEqual(@as(usize, child_count), manager.stats().active);

    const started_ns = monotonicNowNs();
    _ = state.claimExit(3);
    const summary = manager.terminateAndJoin(default_termination_timeout_ns);
    const elapsed_ns = monotonicNowNs() - started_ns;

    try std.testing.expect(!summary.timed_out);
    try std.testing.expectEqual(@as(usize, child_count), summary.joined);
    try std.testing.expectEqual(@as(usize, child_count), summary.trapped);
    try std.testing.expect(elapsed_ns < std.time.ns_per_s);
    // Each guest stack, clone, and record is reclaimed exactly once.
    try std.testing.expectEqual(@as(usize, child_count), atomicCount(&joined));
    try std.testing.expectEqual(@as(usize, child_count), atomicCount(&destroyed));
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
    try std.testing.expectEqual(
        manager.aux_stack_pool.totalCount(),
        manager.aux_stack_pool.availableCount(),
    );
    try std.testing.expectEqual(@as(u32, 1), ctx.mem_inst.referenceCount());
}

test "group termination: a spinning sibling is interrupted by the interpreter poll" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&spin_forever_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var state = termination.State{};
    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    try manager.bindTermination(&state);
    ctx.inst.thread_manager = &manager;

    const child_count = 3;
    for (0..child_count) |_| _ = try manager.spawnThread(ctx.inst, 0);
    try waitForReady(ctx, child_count);

    const started_ns = monotonicNowNs();
    _ = state.claimExit(0);
    const summary = manager.terminateAndJoin(default_termination_timeout_ns);
    const elapsed_ns = monotonicNowNs() - started_ns;

    try std.testing.expect(!summary.timed_out);
    try std.testing.expectEqual(@as(usize, child_count), summary.trapped);
    try std.testing.expect(elapsed_ns < std.time.ns_per_s);
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
}

test "group termination: joining a terminated sibling reports its trap" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&futex_wait_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var state = termination.State{};
    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    try manager.bindTermination(&state);
    try manager.prepareSharedMemory(ctx.mem_inst, null);
    ctx.inst.thread_manager = &manager;

    const tid = try manager.spawnThread(ctx.inst, 0);
    try waitForReady(ctx, 1);
    _ = state.claimExit(2);

    // The join is exact and bounded: the sibling was woken by the claim.
    try std.testing.expectEqual(ThreadOutcome.trapped, try manager.joinOne(tid));
    try std.testing.expectError(error.ThreadAlreadyJoined, manager.joinOne(tid));
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
}

/// Backend that ignores every interrupt for a fixed duration, standing in for
/// an AOT sibling spinning in guest code with no interruptible host call.
const StubbornBackend = struct {
    hold_ns: u64,
    running: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    fn create(parent: *anyopaque, _: std.mem.Allocator) SpawnError!*anyopaque {
        return parent;
    }

    fn configure(
        _: *anyopaque,
        _: *ThreadManager,
        _: i32,
        _: u32,
        _: ?execution_context.AuxiliaryStack,
    ) SpawnError!void {}

    fn run(child: *anyopaque) ThreadOutcome {
        const self: *StubbornBackend = @ptrCast(@alignCast(child));
        self.running.store(true, .release);
        const deadline = monotonicNowNs() +| self.hold_ns;
        while (monotonicNowNs() < deadline) platform.usleep(200);
        return .completed;
    }

    fn destroy(_: *anyopaque) void {}

    const ops = ThreadBackendOps{
        .create = create,
        .configure = configure,
        .run = run,
        .destroy = destroy,
    };
};

test "group termination: an uncooperative sibling bounds teardown instead of deadlocking" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;

    var backend = StubbornBackend{ .hold_ns = 400 * std.time.ns_per_ms };
    var destroyed = std.atomic.Value(usize).init(0);
    const hooks = TestHooks{ .records_destroyed = &destroyed };
    var state = termination.State{};
    var manager = ThreadManager.initWithTestHooks(allocator, &hooks);
    var manager_live = true;
    defer if (manager_live) manager.deinit();
    try manager.bindTermination(&state);

    _ = try manager.spawnWithBackend(
        @ptrCast(&backend),
        0,
        &StubbornBackend.ops,
    );
    while (!backend.running.load(.acquire)) yieldForLifecycle();

    const started_ns = monotonicNowNs();
    _ = state.claimTrap(termination.generic_trap_code);
    const summary = manager.terminateAndJoin(20 * std.time.ns_per_ms);
    const elapsed_ns = monotonicNowNs() - started_ns;

    // Teardown returns on its own deadline rather than waiting for the
    // sibling, and nothing it still owns has been released.
    try std.testing.expect(summary.timed_out);
    try std.testing.expectEqual(@as(usize, 1), summary.unfinished);
    try std.testing.expectEqual(@as(usize, 0), summary.joined);
    try std.testing.expect(elapsed_ns < 300 * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(usize, 0), atomicCount(&destroyed));

    // The record is still owned by the manager, so the ordinary shutdown
    // path reclaims it exactly once once the sibling finally returns.
    manager.deinit();
    manager_live = false;
    try std.testing.expectEqual(@as(usize, 1), atomicCount(&destroyed));
}

test "group termination: a late binder still wakes an already terminated group" {
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&futex_wait_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var state = termination.State{};
    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    try manager.prepareSharedMemory(ctx.mem_inst, null);
    ctx.inst.thread_manager = &manager;

    _ = try manager.spawnThread(ctx.inst, 0);
    try waitForReady(ctx, 1);
    // Claimed before the manager is bound: binding must replay the wakeup.
    _ = state.claimExit(4);
    try manager.bindTermination(&state);

    const summary = manager.terminateAndJoin(default_termination_timeout_ns);
    try std.testing.expect(!summary.timed_out);
    try std.testing.expectEqual(@as(usize, 1), summary.trapped);
}

test "group termination: the cancel broadcast reaches compiled code on every interrupt" {
    // The AOT backend publishes cancellation into each thread's VmCtx through
    // this hook; the manager must replay it for late binders and on every
    // teardown round so a thread that started after the first sweep still
    // observes the flag (#616).
    const Probe = struct {
        calls: usize = 0,

        fn broadcast(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.calls += 1;
        }
    };

    var probe = Probe{};
    var state = termination.State{};
    var manager = ThreadManager.init(std.testing.allocator);
    defer manager.deinit();
    try manager.bindTermination(&state);
    manager.bindCancelBroadcast(.{
        .ctx = @ptrCast(&probe),
        .broadcast = Probe.broadcast,
    });
    try std.testing.expectEqual(@as(usize, 0), probe.calls);

    manager.interrupt();
    try std.testing.expect(probe.calls >= 1);

    const after_interrupt = probe.calls;
    _ = manager.terminateAndJoin(default_termination_timeout_ns);
    try std.testing.expect(probe.calls > after_interrupt);

    // Binding after the group already terminated publishes immediately.
    var late = Probe{};
    var terminated_state = termination.State{};
    var late_manager = ThreadManager.init(std.testing.allocator);
    defer late_manager.deinit();
    try late_manager.bindTermination(&terminated_state);
    _ = terminated_state.claimExit(1);
    late_manager.bindCancelBroadcast(.{
        .ctx = @ptrCast(&late),
        .broadcast = Probe.broadcast,
    });
    try std.testing.expect(late.calls >= 1);
}

test "ThreadManager: Windows cancel event is lazy for unused managers" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    const hooks = TestHooks{ .fail_windows_cancel_event = true };
    var manager = ThreadManager.initWithTestHooks(std.testing.allocator, &hooks);
    defer manager.deinit();
    try std.testing.expect(manager.windows_cancel.opaqueHandle() == null);
}

test "ThreadManager: Windows cancel event creation failure is explicit" {
    if (builtin.os.tag != .windows or !config.lib_wasi_threads)
        return error.SkipZigTest;

    const hooks = TestHooks{ .fail_windows_cancel_event = true };
    var manager = ThreadManager.initWithTestHooks(std.testing.allocator, &hooks);
    defer manager.deinit();
    var state = termination.State{};
    try std.testing.expectError(
        error.WindowsCancelEventUnavailable,
        manager.bindTermination(&state),
    );
    try std.testing.expect(manager.termination == null);
    try std.testing.expect(state.windowsCancelHandle() == null);
}

/// Parks the calling thread on the group's shared memory with an infinite
/// timeout, on its own thread so the harness keeps a bounded escape hatch.
const EmbedderParkCtx = struct {
    memory: *types.MemoryInstance,
    offset: usize,
    result: types.MemoryInstance.SharedWaitError!parking_lot.WaitResult = .not_equal,
    finished: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    fn run(self: *EmbedderParkCtx) void {
        self.result = self.memory.wait32(self.offset, 0, -1);
        self.finished.store(true, .release);
    }

    fn awaitFinished(self: *EmbedderParkCtx, timeout_ns: u64) bool {
        const deadline = monotonicNowNs() +| timeout_ns;
        while (!self.finished.load(.acquire)) {
            if (monotonicNowNs() >= deadline) return false;
            platform.usleep(200);
        }
        return true;
    }
};

test "group termination: the embedder parking after the last sweep is still cancelled" {
    // The reported #955 hang: the embedder thread checks `isTerminating()`,
    // a child claims the terminal outcome and exits (running the last sweep
    // over an empty queue), and only then does the embedder park with an
    // infinite timeout. Nothing sweeps again — `terminateAndJoin` is never
    // even entered — so the wait must be refused rather than woken.
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&nop_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var state = termination.State{};
    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    try manager.bindTermination(&state);
    try manager.prepareSharedMemory(ctx.mem_inst, null);
    ctx.inst.thread_manager = &manager;

    // A child that completes on its own; its `threadEntry` outcome plus the
    // claim below produce the only two sweeps in the run.
    _ = try manager.spawnThread(ctx.inst, 0);
    try waitForCompleted(&manager, 1);
    _ = state.claimExit(0);
    try std.testing.expect(manager.isTerminating());

    var park = EmbedderParkCtx{ .memory = ctx.mem_inst, .offset = 16 };
    const thread = try std.Thread.spawn(.{}, EmbedderParkCtx.run, .{&park});
    const finished = park.awaitFinished(5 * std.time.ns_per_s);
    if (!finished) _ = ctx.mem_inst.cancelWaiters() catch 0;
    thread.join();

    try std.testing.expect(finished);
    try std.testing.expectEqual(
        parking_lot.WaitResult.cancelled,
        try park.result,
    );
    _ = manager.terminateAndJoin(default_termination_timeout_ns);
}

test "group termination: an interpreter child that parks after termination traps" {
    // Same shape as the embedder case but through the real interpreter
    // `memory.atomic.wait32` path with an infinite timeout: the group is
    // already terminated when the child reaches its wait, so the child must
    // trap out instead of parking behind the sweep.
    try requireThreadLifecycle();
    const allocator = std.testing.allocator;
    const ctx = try buildThreadTestModule(&futex_wait_thread_code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var state = termination.State{};
    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    try manager.bindTermination(&state);
    try manager.prepareSharedMemory(ctx.mem_inst, null);
    ctx.inst.thread_manager = &manager;

    _ = state.claimExit(3);
    try std.testing.expect(ctx.mem_inst.shared_control.?.parking_lot.isCancelled());

    const tid = try manager.spawnThread(ctx.inst, 0);
    const started_ns = monotonicNowNs();
    try std.testing.expectEqual(ThreadOutcome.trapped, try manager.joinOne(tid));
    try std.testing.expect(monotonicNowNs() - started_ns < std.time.ns_per_s);
    try std.testing.expectEqual(@as(?u32, 3), state.exitCode());
}
