//! Thread manager for WASI-threads.
//!
//! Manages thread IDs, thread lifecycle, and coordinates thread
//! spawning/termination for the WASI-threads proposal.

const std = @import("std");
const types = @import("../runtime/common/types.zig");

/// Thread handle tracking a spawned thread and its module instance.
pub const ThreadHandle = struct {
    tid: i32,
    thread: std.Thread,
    instance: *types.ModuleInstance,
};

/// Thread manager for coordinating WASI threads.
pub const ThreadManager = struct {
    /// Next TID to allocate (atomic for thread safety).
    next_tid: std.atomic.Value(i32) = std.atomic.Value(i32).init(1),
    /// Active threads (protected by mutex).
    threads: std.ArrayList(ThreadHandle),
    mutex: std.Thread.Mutex = .{},
    /// Global trap flag — set when any thread traps.
    trap_flag: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ThreadManager {
        return .{
            .threads = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ThreadManager) void {
        // Join all remaining threads
        self.joinAll();
        self.threads.deinit(self.allocator);
    }

    /// Allocate a new thread ID. Range: 1 to 2^29-1.
    pub fn allocateTid(self: *ThreadManager) i32 {
        const tid = self.next_tid.fetchAdd(1, .monotonic);
        // Wrap around if we exceed 2^29-1 (keep bits 30-31 zero per spec)
        if (tid >= (1 << 29)) {
            self.next_tid.store(2, .monotonic);
            return 1;
        }
        return tid;
    }

    /// Register a spawned thread.
    pub fn registerThread(self: *ThreadManager, handle: ThreadHandle) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.threads.append(self.allocator, handle);
    }

    /// Remove a thread from the registry (called when thread exits).
    pub fn unregisterThread(self: *ThreadManager, tid: i32) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        for (self.threads.items, 0..) |t, i| {
            if (t.tid == tid) {
                _ = self.threads.orderedRemove(i);
                break;
            }
        }
    }

    /// Signal all threads to stop (trap propagation).
    pub fn signalTrap(self: *ThreadManager) void {
        self.trap_flag.store(true, .release);
    }

    /// Check if a trap has been signaled.
    pub fn hasTrap(self: *ThreadManager) bool {
        return self.trap_flag.load(.acquire);
    }

    /// Join all active threads, waiting for them to complete.
    pub fn joinAll(self: *ThreadManager) void {
        self.mutex.lock();
        const threads = self.threads.items;
        // Copy handles to join outside the lock
        var handles: [256]ThreadHandle = undefined;
        const count = @min(threads.len, handles.len);
        @memcpy(handles[0..count], threads[0..count]);
        self.threads.clearRetainingCapacity();
        self.mutex.unlock();

        for (handles[0..count]) |h| {
            h.thread.join();
        }
    }

    /// Get the number of active threads.
    pub fn activeCount(self: *ThreadManager) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.threads.items.len;
    }

    /// Spawn a new thread with a cloned module instance.
    /// The new thread calls the exported `wasi_thread_start(tid, start_arg)` function.
    /// Returns the TID on success, or a negative errno on failure.
    pub fn spawnThread(self: *ThreadManager, parent_inst: *types.ModuleInstance) !i32 {
        const tid = self.allocateTid();

        // Clone the parent instance (shared memory, independent globals)
        const child_inst = parent_inst.cloneForThread(self.allocator) catch return error.OutOfMemory;

        // Spawn the native thread
        const thread = std.Thread.spawn(.{}, threadEntry, .{ self, child_inst, tid }) catch {
            // Clean up cloned instance on spawn failure
            destroyClonedInstance(child_inst);
            return error.ThreadSpawnFailed;
        };

        try self.registerThread(.{
            .tid = tid,
            .thread = thread,
            .instance = child_inst,
        });

        return tid;
    }

    fn threadEntry(self: *ThreadManager, inst: *types.ModuleInstance, tid: i32) void {
        defer {
            self.unregisterThread(tid);
            destroyClonedInstance(inst);
        }

        // Find the exported wasi_thread_start function
        const func_idx = inst.getExportFunc("wasi_thread_start") orelse return;

        // Create execution environment for this thread
        const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
        var env = ExecEnv.create(inst, 4096, self.allocator) catch return;
        defer env.destroy();
        env.thread_manager = self;
        env.tid = tid;

        // Push arguments: tid and start_arg (0 for now)
        env.pushI32(tid) catch return;
        env.pushI32(0) catch return;

        // Execute
        const interp = @import("../runtime/interpreter/interp.zig");
        interp.executeFunction(env, func_idx) catch {
            // Thread trapped — signal all other threads
            self.signalTrap();
        };
    }

    fn destroyClonedInstance(inst: *types.ModuleInstance) void {
        const allocator = inst.allocator;
        // Release shared memories/tables
        for (inst.memories) |m| m.release(allocator);
        if (inst.memories.len > 0) allocator.free(inst.memories);
        for (inst.tables) |t| t.release(allocator);
        if (inst.tables.len > 0) allocator.free(inst.tables);
        // Destroy cloned globals
        for (inst.globals) |g| allocator.destroy(g);
        if (inst.globals.len > 0) allocator.free(inst.globals);
        if (inst.dropped_elems.len > 0) allocator.free(inst.dropped_elems);
        allocator.destroy(inst);
    }
};

// ── Tests ────────────────────────────────────────────────────────────────

test "ThreadManager: allocate TIDs" {
    var tm = ThreadManager.init(std.testing.allocator);
    defer tm.deinit();

    const tid1 = tm.allocateTid();
    const tid2 = tm.allocateTid();
    const tid3 = tm.allocateTid();

    try std.testing.expect(tid1 >= 1);
    try std.testing.expect(tid2 > tid1);
    try std.testing.expect(tid3 > tid2);
}

test "ThreadManager: trap flag" {
    var tm = ThreadManager.init(std.testing.allocator);
    defer tm.deinit();

    try std.testing.expect(!tm.hasTrap());
    tm.signalTrap();
    try std.testing.expect(tm.hasTrap());
}
