//! Thread manager for WASI-threads.
//!
//! Manages thread IDs, thread lifecycle, and coordinates thread
//! spawning/termination for the WASI-threads proposal.

const std = @import("std");
const types = @import("../runtime/common/types.zig");
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
    pool_allocator: std.mem.Allocator = undefined,

    /// Pre-allocate N auxiliary stacks starting at `base_offset` in linear memory.
    pub fn init(self: *AuxStackPool, count: u32, base_offset: u32, allocator: std.mem.Allocator) !void {
        self.pool_allocator = allocator;
        try self.free_stacks.ensureTotalCapacity(allocator, count);
        try self.all_stacks.ensureTotalCapacity(allocator, count);

        var offset = base_offset;
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            // Stack grows downward; top-of-stack is at offset + stack_size
            const stack_top = offset + self.stack_size;
            self.free_stacks.appendAssumeCapacity(stack_top);
            self.all_stacks.appendAssumeCapacity(stack_top);
            offset += self.stack_size;
        }
    }

    pub fn deinit(self: *AuxStackPool, allocator: std.mem.Allocator) void {
        self.free_stacks.deinit(allocator);
        self.all_stacks.deinit(allocator);
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
        self.free_stacks.append(self.pool_allocator, stack_top) catch {};
    }
};

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
    mutex: Mutex = .init,
    /// Global trap flag — set when any thread traps.
    trap_flag: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    allocator: std.mem.Allocator,
    /// Per-thread auxiliary stack pool.
    aux_stack_pool: AuxStackPool = .{},

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
        self.aux_stack_pool.deinit(self.allocator);
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
    pub fn spawnThread(self: *ThreadManager, parent_inst: *types.ModuleInstance, start_arg: i32) !i32 {
        const tid = self.allocateTid();

        // Clone the parent instance (shared memory, independent globals)
        const child_inst = parent_inst.cloneForThread(self.allocator) catch return error.OutOfMemory;

        // Set up per-thread auxiliary stack if available
        var aux_stack_top: ?u32 = null;
        if (self.aux_stack_pool.allocate()) |stack_top| {
            aux_stack_top = stack_top;
            // Find and set __stack_pointer global in the cloned instance
            if (child_inst.module.findExport("__stack_pointer", .global)) |exp| {
                if (exp.index < child_inst.globals.len) {
                    child_inst.globals[exp.index].value = .{ .i32 = @bitCast(stack_top) };
                }
            }
        }

        // Spawn the native thread
        const thread = std.Thread.spawn(.{}, threadEntry, .{ self, child_inst, tid, start_arg, aux_stack_top }) catch {
            // Return aux stack on failure
            if (aux_stack_top) |st| self.aux_stack_pool.release(st);
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

    fn threadEntry(self: *ThreadManager, inst: *types.ModuleInstance, tid: i32, start_arg: i32, aux_stack_top: ?u32) void {
        defer {
            // Return aux stack to pool
            if (aux_stack_top) |st| self.aux_stack_pool.release(st);
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

        // Push arguments: tid and start_arg
        env.pushI32(tid) catch return;
        env.pushI32(start_arg) catch return;

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

test "ModuleInstance: cloneForThread shares memory" {
    const allocator = std.testing.allocator;

    // Create a parent module with shared memory
    var module = types.WasmModule{};
    const mem_data = try allocator.alloc(u8, 65536);
    defer allocator.free(mem_data);
    @memset(mem_data, 0);
    mem_data[0] = 42; // write a value

    var mem_inst = try allocator.create(types.MemoryInstance);
    defer {
        mem_inst.ref_count -= 1; // balance the clone's retain
        allocator.destroy(mem_inst);
    }
    mem_inst.* = .{
        .memory_type = .{ .limits = .{ .min = 1 }, .is_shared = true },
        .data = mem_data,
        .current_pages = 1,
        .max_pages = 4,
    };

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
    defer {
        // Release shared memory (retain was called in clone)
        for (child.memories) |m| m.release(allocator);
        allocator.free(child.memories);
        for (child.globals) |g| allocator.destroy(g);
        allocator.destroy(child);
    }

    // Child should see the same memory
    try std.testing.expectEqual(@as(u8, 42), child.memories[0].data[0]);

    // Write through child — parent should see it (shared)
    child.memories[0].data[1] = 99;
    try std.testing.expectEqual(@as(u8, 99), parent.memories[0].data[1]);

    // Verify ref count was incremented
    try std.testing.expectEqual(@as(u32, 2), mem_inst.ref_count);

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
}

test "WaiterQueue: notify with no waiters" {
    const allocator = std.testing.allocator;
    var wq = try allocator.create(types.WaiterQueue);
    defer wq.deinit(allocator);
    wq.* = .{};

    // Notify on empty queue should return 0
    const woken = wq.notify(0, 10);
    try std.testing.expectEqual(@as(u32, 0), woken);
}

test "WaiterQueue: wait with immediate timeout" {
    const allocator = std.testing.allocator;
    var wq = try allocator.create(types.WaiterQueue);
    defer wq.deinit(allocator);
    wq.* = .{};

    // Wait with 0 timeout should return 2 (timed out) quickly
    const result = wq.wait(100, 0, allocator);
    // 0 timeout = immediate timeout or woken (could be either depending on timing)
    try std.testing.expect(result == 0 or result == 2);
}

// ── Integration tests ───────────────────────────────────────────────────────
// These tests exercise the full thread lifecycle: spawn → execute → join.
// Each builds a WasmModule with an exported wasi_thread_start function,
// creates a ModuleInstance with shared memory, and spawns real threads.
// Gated on multi-threaded targets (can't spawn threads on wasm32-wasi).

const builtin = @import("builtin");

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
        .name = "wasi_thread_start",
        .kind = .function,
        .index = 0,
    };
    const memories = try allocator.alloc(types.MemoryType, 1);
    memories[0] = .{ .limits = .{ .min = 1 }, .is_shared = true };

    module.* = .{
        .types = func_types,
        .functions = functions,
        .exports = exports,
        .memories = memories,
    };

    // Create shared memory instance
    const mem_data = try allocator.alloc(u8, 65536);
    @memset(mem_data, 0);
    const mem_inst = try allocator.create(types.MemoryInstance);
    mem_inst.* = .{
        .memory_type = .{ .limits = .{ .min = 1 }, .is_shared = true },
        .data = mem_data,
        .current_pages = 1,
        .max_pages = 4,
    };
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

test "integration: spawn threads incrementing atomic counter" {
    if (builtin.single_threaded) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // wasi_thread_start body: i32.const 0; i32.const 1; i32.atomic.rmw.add 2 0; drop
    const code = [_]u8{
        0x41, 0x00, // i32.const 0 (address)
        0x41, 0x01, // i32.const 1 (value to add)
        0xFE, 0x1E, 0x02, 0x00, // i32.atomic.rmw.add align=2 offset=0
        0x1A, // drop
    };
    const ctx = try buildThreadTestModule(&code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var tm = ThreadManager.init(allocator);
    defer tm.deinit();
    ctx.inst.thread_manager = &tm;

    // Spawn 2 threads
    const tid1 = try tm.spawnThread(ctx.inst, 0);
    const tid2 = try tm.spawnThread(ctx.inst, 0);
    try std.testing.expect(tid1 > 0);
    try std.testing.expect(tid2 > 0);
    try std.testing.expect(tid1 != tid2);

    // Wait for completion
    tm.joinAll();

    // Read shared memory — should have been incremented twice
    const counter = std.mem.readInt(u32, ctx.mem_inst.data[0..4], .little);
    try std.testing.expectEqual(@as(u32, 2), counter);
}

test "integration: trap in child signals trap flag" {
    if (builtin.single_threaded) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // wasi_thread_start body: unreachable
    const code = [_]u8{0x00}; // unreachable
    const ctx = try buildThreadTestModule(&code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var tm = ThreadManager.init(allocator);
    defer tm.deinit();
    ctx.inst.thread_manager = &tm;

    try std.testing.expect(!tm.hasTrap());

    const tid = try tm.spawnThread(ctx.inst, 0);
    try std.testing.expect(tid > 0);

    // Wait for the thread to finish (it will trap)
    tm.joinAll();

    // Trap flag should be set
    try std.testing.expect(tm.hasTrap());
}

test "integration: stress spawn 4 threads" {
    if (builtin.single_threaded) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // wasi_thread_start body: i32.const 0; i32.const 1; i32.atomic.rmw.add 2 0; drop
    const code = [_]u8{
        0x41, 0x00, // i32.const 0
        0x41, 0x01, // i32.const 1
        0xFE, 0x1E, 0x02, 0x00, // i32.atomic.rmw.add align=2 offset=0
        0x1A, // drop
    };
    const ctx = try buildThreadTestModule(&code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var tm = ThreadManager.init(allocator);
    defer tm.deinit();
    ctx.inst.thread_manager = &tm;

    // Spawn 4 threads
    var tids: [4]i32 = undefined;
    for (&tids) |*tid| {
        tid.* = try tm.spawnThread(ctx.inst, 0);
        try std.testing.expect(tid.* > 0);
    }

    // All TIDs should be unique
    for (tids, 0..) |a, i| {
        for (tids[i + 1 ..]) |b| {
            try std.testing.expect(a != b);
        }
    }

    tm.joinAll();

    // Counter should equal 4
    const counter = std.mem.readInt(u32, ctx.mem_inst.data[0..4], .little);
    try std.testing.expectEqual(@as(u32, 4), counter);
    try std.testing.expect(!tm.hasTrap());
}

test "integration: threads receive start_arg" {
    if (builtin.single_threaded) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // wasi_thread_start(tid, start_arg) body:
    //   i32.const 0 (address); local.get 1 (start_arg); i32.atomic.rmw.add 2 0; drop
    const code = [_]u8{
        0x41, 0x00, // i32.const 0 (address)
        0x20, 0x01, // local.get 1 (start_arg)
        0xFE, 0x1E, 0x02, 0x00, // i32.atomic.rmw.add align=2 offset=0
        0x1A, // drop
    };
    const ctx = try buildThreadTestModule(&code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var tm = ThreadManager.init(allocator);
    defer tm.deinit();
    ctx.inst.thread_manager = &tm;

    // Spawn with start_arg=10 and start_arg=20
    _ = try tm.spawnThread(ctx.inst, 10);
    _ = try tm.spawnThread(ctx.inst, 20);
    tm.joinAll();

    // Counter should be 10 + 20 = 30
    const counter = std.mem.readInt(u32, ctx.mem_inst.data[0..4], .little);
    try std.testing.expectEqual(@as(u32, 30), counter);
}

test "integration: aux stack pool allocates distinct stacks" {
    if (builtin.single_threaded) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // Simple nop body — we just verify stacks are allocated/freed
    const code = [_]u8{0x01}; // nop
    const ctx = try buildThreadTestModule(&code, allocator);
    defer cleanupThreadTest(ctx, allocator);

    var tm = ThreadManager.init(allocator);
    defer tm.deinit();
    ctx.inst.thread_manager = &tm;

    // Pre-allocate aux stacks
    try tm.aux_stack_pool.init(2, 32768, allocator);

    // Spawn 2 threads — each should get a distinct stack
    _ = try tm.spawnThread(ctx.inst, 0);
    _ = try tm.spawnThread(ctx.inst, 0);
    tm.joinAll();

    // Both stacks should be returned to the pool
    try std.testing.expectEqual(@as(usize, 2), tm.aux_stack_pool.free_stacks.items.len);
    try std.testing.expect(!tm.hasTrap());
}
