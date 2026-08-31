//! Thread manager for WASI-threads.
//!
//! Manages thread IDs, thread lifecycle, and coordinates thread
//! spawning/termination for the WASI-threads proposal.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../runtime/common/types.zig");
const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
const execution_context = @import("../runtime/common/execution_context.zig");
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
    mutex: Mutex = .init,

    /// Pre-allocate N auxiliary stacks starting at `base_offset` in linear memory.
    pub fn init(self: *AuxStackPool, count: u32, base_offset: u32, allocator: std.mem.Allocator) !void {
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
    }

    pub fn deinit(self: *AuxStackPool, allocator: std.mem.Allocator) void {
        std.debug.assert(self.free_stacks.items.len == self.all_stacks.items.len);
        self.free_stacks.deinit(allocator);
        self.all_stacks.deinit(allocator);
        self.free_stacks = .empty;
        self.all_stacks = .empty;
    }

    /// Allocate a stack for a new thread. Returns the top-of-stack offset, or null.
    pub fn allocate(self: *AuxStackPool) ?u32 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const items = self.free_stacks.items;
        if (items.len == 0) return null;
        const val = items[items.len - 1];
        self.free_stacks.items.len -= 1;
        return val;
    }

    /// Return a stack to the pool.
    pub fn release(self: *AuxStackPool, stack_top: u32) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        std.debug.assert(std.mem.indexOfScalar(u32, self.all_stacks.items, stack_top) != null);
        std.debug.assert(std.mem.indexOfScalar(u32, self.free_stacks.items, stack_top) == null);
        std.debug.assert(self.free_stacks.items.len < self.free_stacks.capacity);
        self.free_stacks.appendAssumeCapacity(stack_top);
    }

    pub fn availableCount(self: *AuxStackPool) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.free_stacks.items.len;
    }

    pub fn totalCount(self: *AuxStackPool) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.all_stacks.items.len;
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

    pub fn init(allocator: std.mem.Allocator) ThreadManager {
        return .{
            .slots = .empty,
            .allocator = allocator,
        };
    }

    fn initWithTestHooks(allocator: std.mem.Allocator, hooks: *const TestHooks) ThreadManager {
        var manager = init(allocator);
        manager.test_hooks = hooks;
        return manager;
    }

    pub fn deinit(self: *ThreadManager) void {
        self.shutdown();
        const current = self.stats();
        std.debug.assert(current.retained == 0);
        std.debug.assert(current.spawning == 0);
        std.debug.assert(current.joining == 0);
        self.slots.deinit(self.allocator);
        self.aux_stack_pool.deinit(self.allocator);
    }

    /// Signal all threads to stop (trap propagation).
    pub fn signalTrap(self: *ThreadManager) void {
        self.trap_flag.store(true, .release);
    }

    /// Check if a trap has been signaled.
    pub fn hasTrap(self: *ThreadManager) bool {
        return self.trap_flag.load(.acquire);
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
        if (comptime !config.thread_mgr or builtin.single_threaded)
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
            self.aux_stack_pool.totalCount() != 0)
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

fn requireThreadLifecycle() !void {
    if (builtin.single_threaded or !config.thread_mgr) return error.SkipZigTest;
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
    if (config.thread_mgr) return error.SkipZigTest;
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

    var manager = ThreadManager.init(allocator);
    defer manager.deinit();
    trap_ctx.inst.thread_manager = &manager;
    missing_ctx.inst.thread_manager = &manager;

    const trap_tid = try manager.spawnThread(trap_ctx.inst, 0);
    try std.testing.expectEqual(ThreadOutcome.trapped, try manager.joinOne(trap_tid));
    try std.testing.expect(manager.hasTrap());
    try std.testing.expectError(
        error.MissingThreadStart,
        manager.spawnThread(missing_ctx.inst, 0),
    );
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
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
