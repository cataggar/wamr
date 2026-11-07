//! Stackful Coroutine & Shadow-Stack Runtime for WASIp3 Async Tasks (#1134).
//!
//! Provides `ShadowStack`, `ShadowStackPool`, `StackPointerBinding`, and
//! `TaskContext` compatible with `wit-zigar`'s `async_runtime.zig` while
//! integrating directly with WAMR's `ExecEnv` and `ModuleInstance`
//! `__stack_pointer` globals.

const std = @import("std");
const builtin = @import("builtin");

/// Opaque handle to a WAMR `ExecEnv` (`runtime/common/exec_env.zig`).
/// Kept type-erased at the module boundary so `coro.zig` can also be
/// unit-tested standalone via `zig test src/component/coro.zig`.
pub const ExecEnv = anyopaque;

pub const DEFAULT_SHADOW_STACK_SIZE: usize = 64 * 1024;
pub const SHADOW_STACK_ALIGNMENT: std.mem.Alignment = std.mem.Alignment.@"16";
pub const STACK_CANARY: u64 = 0xDEADBEEF_CAFEBABE;
pub const stack_canary: u64 = STACK_CANARY;

threadlocal var simulated_wasm_sp: usize = 0;
threadlocal var active_task_tls: ?*TaskContext = null;

pub inline fn getWasmStackPointer() usize {
    if (builtin.cpu.arch == .wasm32) {
        return asm volatile (
            \\global.get __stack_pointer
            : [ret] "=r" (-> usize),
        );
    } else if (builtin.cpu.arch == .wasm64) {
        return asm volatile (
            \\global.get __stack_pointer
            : [ret] "=r" (-> usize),
        );
    } else {
        return simulated_wasm_sp;
    }
}

pub inline fn setWasmStackPointer(sp: usize) void {
    if (builtin.cpu.arch == .wasm32) {
        asm volatile (
            \\local.get %[sp]
            \\global.set __stack_pointer
            :
            : [sp] "r" (sp),
        );
    } else if (builtin.cpu.arch == .wasm64) {
        asm volatile (
            \\local.get %[sp]
            \\global.set __stack_pointer
            :
            : [sp] "r" (sp),
        );
    } else {
        simulated_wasm_sp = sp;
    }
}

pub const getStackPointer = getWasmStackPointer;
pub const setStackPointer = setWasmStackPointer;

pub inline fn getActiveTask() ?*TaskContext {
    return active_task_tls;
}

pub inline fn setActiveTask(task: ?*TaskContext) void {
    active_task_tls = task;
}

/// Abstracts reading and writing a guest's linear-memory stack pointer
/// (`__stack_pointer`).
pub const StackPointerBinding = union(enum) {
    /// Direct pointer to a host-accessible stack-pointer slot (`usize`).
    host_slot: *usize,
    /// Callback pair for reading/writing a guest module's `__stack_pointer`.
    callbacks: struct {
        get_fn: *const fn (?*anyopaque) usize,
        set_fn: *const fn (?*anyopaque, usize) void,
        ctx: ?*anyopaque = null,
    },
    /// Direct pointer to a WAMR 32-bit `GlobalInstance.value.i32` slot.
    wasm_global_i32: *i32,
    /// Direct pointer to a WAMR 64-bit `GlobalInstance.value.i64` slot.
    wasm_global_i64: *i64,
    /// Uses inline Wasm `global.get/set __stack_pointer` on wasm32/wasm64,
    /// or a thread-local simulated stack pointer on native targets.
    wasm_global: void,

    pub const default: StackPointerBinding = .{ .wasm_global = {} };

    pub fn fromHostSlot(slot: *usize) StackPointerBinding {
        return .{ .host_slot = slot };
    }

    pub fn fromCallbacks(
        get_fn: *const fn (?*anyopaque) usize,
        set_fn: *const fn (?*anyopaque, usize) void,
        ctx: ?*anyopaque,
    ) StackPointerBinding {
        return .{
            .callbacks = .{
                .get_fn = get_fn,
                .set_fn = set_fn,
                .ctx = ctx,
            },
        };
    }

    /// Look up the `"__stack_pointer"` global export on `mi` (`*core_types.ModuleInstance`)
    /// and bind directly to its `GlobalInstance` value slot (`i32` or `i64`).
    /// Falls back to `.wasm_global` if the module does not export `"__stack_pointer"`.
    pub fn fromModuleInstance(mi: anytype) StackPointerBinding {
        if (mi.module.findExport("__stack_pointer", .global)) |exp| {
            if (exp.index < mi.globals.len) {
                switch (mi.globals[exp.index].value) {
                    .i32 => |*slot| return .{ .wasm_global_i32 = slot },
                    .i64 => |*slot| return .{ .wasm_global_i64 = slot },
                    else => {},
                }
            }
        }
        return .{ .wasm_global = {} };
    }

    pub inline fn get(self: StackPointerBinding) usize {
        return switch (self) {
            .host_slot => |slot| slot.*,
            .callbacks => |cb| cb.get_fn(cb.ctx),
            .wasm_global_i32 => |slot| @as(u32, @bitCast(slot.*)),
            .wasm_global_i64 => |slot| @intCast(@as(u64, @bitCast(slot.*))),
            .wasm_global => getWasmStackPointer(),
        };
    }

    pub inline fn set(self: StackPointerBinding, sp: usize) void {
        switch (self) {
            .host_slot => |slot| slot.* = sp,
            .callbacks => |cb| cb.set_fn(cb.ctx, sp),
            .wasm_global_i32 => |slot| slot.* = @bitCast(@as(u32, @truncate(sp))),
            .wasm_global_i64 => |slot| slot.* = @bitCast(@as(u64, @intCast(sp))),
            .wasm_global => setWasmStackPointer(sp),
        }
    }

    pub inline fn read(self: StackPointerBinding) u64 {
        return @intCast(self.get());
    }

    pub inline fn write(self: StackPointerBinding, sp: u64) void {
        self.set(@intCast(sp));
    }
};

/// A 16-byte aligned linear-memory shadow stack for an async task.
/// Stores a 64-bit canary (`0xDEADBEEF_CAFEBABE`) at the bottom (lowest
/// address) of the allocation to detect stack overflow.
pub const ShadowStack = struct {
    pub const CANARY_SIZE: usize = 8;

    bytes: []align(16) u8,
    from_scratch: bool = false,
    guest_base_offset: ?u32 = null,

    pub fn init(bytes: []align(16) u8) ShadowStack {
        std.debug.assert(bytes.len >= 32);
        std.debug.assert((bytes.len & 15) == 0);
        const s = ShadowStack{ .bytes = bytes };
        s.writeCanary();
        return s;
    }

    pub fn initGuest(bytes: []align(16) u8, guest_base_offset: u32) ShadowStack {
        std.debug.assert(bytes.len >= 32);
        std.debug.assert((bytes.len & 15) == 0);
        const s = ShadowStack{
            .bytes = bytes,
            .guest_base_offset = guest_base_offset,
        };
        s.writeCanary();
        return s;
    }

    pub fn writeCanary(self: ShadowStack) void {
        std.mem.writeInt(u64, self.bytes[0..8], STACK_CANARY, .little);
    }

    pub fn checkCanary(self: ShadowStack) bool {
        return std.mem.readInt(u64, self.bytes[0..8], .little) == STACK_CANARY;
    }

    /// Returns the 16-byte aligned top-of-stack address (stack grows downward).
    pub fn initialTop(self: ShadowStack) usize {
        if (self.guest_base_offset) |base_off| {
            return (@as(usize, base_off) + self.bytes.len) & ~@as(usize, 15);
        }
        return (@intFromPtr(self.bytes.ptr) + self.bytes.len) & ~@as(usize, 15);
    }

    /// Returns the lowest valid stack pointer address above the 16-byte
    /// canary guard region.
    pub fn usableBase(self: ShadowStack) usize {
        if (self.guest_base_offset) |base_off| {
            return @as(usize, base_off) + 16;
        }
        return @intFromPtr(self.bytes.ptr) + 16;
    }

    /// Returns the 16-byte aligned top offset relative to `base_addr`
    /// (for example when `bytes` is a subslice of a simulated linear memory).
    pub fn relativeInitialTop(self: ShadowStack, base_addr: usize) usize {
        const offset = @intFromPtr(self.bytes.ptr) - base_addr;
        return (offset + self.bytes.len) & ~@as(usize, 15);
    }

    /// Validates both the bottom canary word and that `sp` lies within the
    /// stack's valid range `[base + CANARY_SIZE, initialTop]`.
    pub fn checkCanaryAndBounds(self: ShadowStack, sp: usize) bool {
        if (!self.checkCanary()) return false;
        const base = if (self.guest_base_offset) |base_off|
            @as(usize, base_off) + CANARY_SIZE
        else
            @intFromPtr(self.bytes.ptr) + CANARY_SIZE;
        const top = self.initialTop();
        return sp >= base and sp <= top;
    }
};

/// Pool of reusable 16-byte aligned shadow stacks.
pub const ShadowStackPool = struct {
    pub const MAX_POOLED_STACKS: usize = 16;

    pooled: [16]ShadowStack = undefined,
    pooled_count: usize = 0,
    stack_size: usize = DEFAULT_SHADOW_STACK_SIZE,
    scratch_region: ?[]align(16) u8 = null,
    scratch_cursor: usize = 0,

    pub fn init(stack_size: usize) ShadowStackPool {
        const aligned_size = std.mem.alignForward(usize, @max(stack_size, 64), 16);
        return .{ .stack_size = aligned_size };
    }

    pub fn initWithScratch(scratch: []align(16) u8, stack_size: usize) ShadowStackPool {
        const aligned_size = std.mem.alignForward(usize, @max(stack_size, 64), 16);
        return .{
            .stack_size = aligned_size,
            .scratch_region = scratch,
            .scratch_cursor = 0,
        };
    }

    pub fn acquire(self: *ShadowStackPool, allocator: std.mem.Allocator) !ShadowStack {
        if (self.pooled_count > 0) {
            self.pooled_count -= 1;
            const s = self.pooled[self.pooled_count];
            s.writeCanary();
            return s;
        }
        if (self.scratch_region) |scratch| {
            if (self.scratch_cursor + self.stack_size <= scratch.len) {
                const slice: []align(16) u8 = @alignCast(scratch[self.scratch_cursor .. self.scratch_cursor + self.stack_size]);
                self.scratch_cursor += self.stack_size;
                var s = ShadowStack.init(slice);
                s.from_scratch = true;
                return s;
            }
        }
        const bytes = try allocator.alignedAlloc(u8, SHADOW_STACK_ALIGNMENT, self.stack_size);
        return ShadowStack.init(bytes);
    }

    pub fn release(self: *ShadowStackPool, allocator: std.mem.Allocator, stack: ShadowStack) error{StackOverflow}!void {
        const canary_ok = stack.checkCanary();
        if (stack.from_scratch) {
            if (canary_ok and self.pooled_count < MAX_POOLED_STACKS and stack.bytes.len == self.stack_size) {
                self.pooled[self.pooled_count] = stack;
                self.pooled_count += 1;
            }
        } else if (canary_ok and self.pooled_count < MAX_POOLED_STACKS and stack.bytes.len == self.stack_size) {
            self.pooled[self.pooled_count] = stack;
            self.pooled_count += 1;
        } else {
            allocator.free(stack.bytes);
        }
        if (!canary_ok) return error.StackOverflow;
    }

    pub fn deinit(self: *ShadowStackPool, allocator: std.mem.Allocator) void {
        for (self.pooled[0..self.pooled_count]) |s| {
            if (!s.from_scratch) {
                allocator.free(s.bytes);
            }
        }
        self.pooled_count = 0;
        self.scratch_cursor = 0;
    }
};

/// Per-task coroutine execution context and shadow-stack switcher.
pub const TaskContext = struct {
    pub const State = enum(u8) {
        initial,
        running,
        suspended,
        completed,
    };
    pub const Status = State;

    shadow_stack: ShadowStack,
    saved_task_sp: usize,
    caller_sp: usize = 0,
    sp_binding: StackPointerBinding = StackPointerBinding.default,
    state: State = .initial,
    context_slots: [2]usize = .{ 0, 0 },
    prev_task: ?*TaskContext = null,
    exec_env: ?*ExecEnv = null,
    waiting_on_set: ?u32 = null,
    waiting_out_ptr: u32 = 0,

    pub fn init(stack: ShadowStack, sp_binding: StackPointerBinding) TaskContext {
        const top = stack.initialTop();
        return .{
            .shadow_stack = stack,
            .saved_task_sp = top,
            .caller_sp = 0,
            .sp_binding = sp_binding,
            .state = .initial,
        };
    }

    pub fn initRelative(stack: ShadowStack, sp_binding: StackPointerBinding, base_addr: usize) TaskContext {
        const top = stack.relativeInitialTop(base_addr);
        return .{
            .shadow_stack = stack,
            .saved_task_sp = top,
            .caller_sp = 0,
            .sp_binding = sp_binding,
            .state = .initial,
        };
    }

    pub fn enter(self: *TaskContext) void {
        std.debug.assert(self.state == .initial);
        self.caller_sp = self.sp_binding.get();
        self.sp_binding.set(self.saved_task_sp);
        self.prev_task = getActiveTask();
        setActiveTask(self);
        self.state = .running;
    }

    pub fn suspendTask(self: *TaskContext) void {
        std.debug.assert(self.state == .running);
        self.saved_task_sp = self.sp_binding.get();
        self.sp_binding.set(self.caller_sp);
        setActiveTask(self.prev_task);
        self.prev_task = null;
        self.state = .suspended;
    }

    pub fn resumeTask(self: *TaskContext) void {
        std.debug.assert(self.state == .suspended);
        self.caller_sp = self.sp_binding.get();
        self.sp_binding.set(self.saved_task_sp);
        self.prev_task = getActiveTask();
        setActiveTask(self);
        self.state = .running;
    }

    pub fn complete(self: *TaskContext) error{StackOverflow}!void {
        const final_task_sp = self.sp_binding.get();
        self.saved_task_sp = final_task_sp;
        self.sp_binding.set(self.caller_sp);
        setActiveTask(self.prev_task);
        self.prev_task = null;
        self.state = .completed;
        if (!self.shadow_stack.checkCanary()) {
            return error.StackOverflow;
        }
    }

    pub fn contextGet(self: *const TaskContext, slot: u32) error{InvalidContextSlot}!usize {
        if (slot >= self.context_slots.len) return error.InvalidContextSlot;
        return self.context_slots[slot];
    }

    pub fn contextSet(self: *TaskContext, slot: u32, val: usize) error{InvalidContextSlot}!void {
        if (slot >= self.context_slots.len) return error.InvalidContextSlot;
        self.context_slots[slot] = val;
    }

    pub fn run(self: *TaskContext, comptime func: anytype, args: anytype) error{StackOverflow}!@TypeOf(@call(.auto, func, args)) {
        self.enter();
        const result = @call(.auto, func, args);
        try self.complete();
        return result;
    }

    pub fn runStep(self: *TaskContext, comptime func: anytype, args: anytype) error{StackOverflow}!State {
        switch (self.state) {
            .initial => self.enter(),
            .suspended => self.resumeTask(),
            .running, .completed => return self.state,
        }
        _ = @call(.auto, func, args);
        if (self.state == .running) {
            try self.complete();
        }
        return self.state;
    }
};

// ── Unit Tests ───────────────────────────────────────────────────────────────

test "coro: ShadowStackPool acquire, reuse, scratch region, and canary overflow detection" {
    const allocator = std.testing.allocator;
    var pool = ShadowStackPool.init(4096);
    defer pool.deinit(allocator);

    const stack1 = try pool.acquire(allocator);
    try std.testing.expectEqual(@as(usize, 4096), stack1.bytes.len);
    try std.testing.expectEqual(@as(usize, 0), stack1.initialTop() & 15);
    try std.testing.expect(stack1.checkCanary());
    try std.testing.expect(stack1.checkCanaryAndBounds(stack1.initialTop()));
    try std.testing.expect(!stack1.checkCanaryAndBounds(stack1.initialTop() + 16));

    const stack1_ptr = @intFromPtr(stack1.bytes.ptr);
    try pool.release(allocator, stack1);
    try std.testing.expectEqual(@as(usize, 1), pool.pooled_count);

    // Re-acquiring should reuse the pooled allocation and refresh its canary.
    const stack2 = try pool.acquire(allocator);
    try std.testing.expectEqual(stack1_ptr, @intFromPtr(stack2.bytes.ptr));
    try std.testing.expect(stack2.checkCanary());

    // Corrupt canary to verify StackOverflow detection on release.
    std.mem.writeInt(u64, stack2.bytes[0..8], 0x12345678_9ABCDEF0, .little);
    try std.testing.expect(!stack2.checkCanary());
    try std.testing.expectError(error.StackOverflow, pool.release(allocator, stack2));

    // Scratch-backed pool test.
    var scratch_buf: [1024]u8 align(16) = @splat(0);
    var scratch_pool = ShadowStackPool.initWithScratch(&scratch_buf, 256);
    defer scratch_pool.deinit(allocator);

    const s_a = try scratch_pool.acquire(allocator);
    const s_b = try scratch_pool.acquire(allocator);
    try std.testing.expect(s_a.from_scratch);
    try std.testing.expect(s_b.from_scratch);
    try std.testing.expect(@intFromPtr(s_a.bytes.ptr) != @intFromPtr(s_b.bytes.ptr));
    try scratch_pool.release(allocator, s_a);
    try scratch_pool.release(allocator, s_b);
}

test "coro: TaskContext suspend, resume, nested tasks, and context slots" {
    const allocator = std.testing.allocator;
    var pool = ShadowStackPool.init(4096);
    defer pool.deinit(allocator);

    const stack_a = try pool.acquire(allocator);
    defer pool.release(allocator, stack_a) catch {};
    const stack_b = try pool.acquire(allocator);
    defer pool.release(allocator, stack_b) catch {};

    const initial_caller_sp: usize = 0x7FFF_0000;
    var sp_slot: usize = initial_caller_sp;
    const binding = StackPointerBinding.fromHostSlot(&sp_slot);

    var task_a = TaskContext.init(stack_a, binding);
    var task_b = TaskContext.init(stack_b, binding);

    // Enter task A
    task_a.enter();
    try std.testing.expectEqual(TaskContext.State.running, task_a.state);
    try std.testing.expectEqual(@as(?*TaskContext, &task_a), getActiveTask());
    try std.testing.expectEqual(stack_a.initialTop(), sp_slot);
    try task_a.contextSet(0, 0xAAAA);
    try task_a.contextSet(1, 0xBBBB);
    try std.testing.expectError(error.InvalidContextSlot, task_a.contextSet(2, 1));

    // Simulate stack usage in task A and suspend
    const task_a_frame_sp = stack_a.initialTop() - 128;
    sp_slot = task_a_frame_sp;
    task_a.suspendTask();
    try std.testing.expectEqual(TaskContext.State.suspended, task_a.state);
    try std.testing.expectEqual(@as(?*TaskContext, null), getActiveTask());
    try std.testing.expectEqual(initial_caller_sp, sp_slot);

    // Enter task B while task A is suspended
    task_b.enter();
    try std.testing.expectEqual(@as(?*TaskContext, &task_b), getActiveTask());
    try std.testing.expectEqual(stack_b.initialTop(), sp_slot);
    const task_b_frame_sp = stack_b.initialTop() - 64;
    sp_slot = task_b_frame_sp;
    task_b.suspendTask();
    try std.testing.expectEqual(initial_caller_sp, sp_slot);

    // Resume task A and verify its stack pointer and context slots are preserved
    task_a.resumeTask();
    try std.testing.expectEqual(TaskContext.State.running, task_a.state);
    try std.testing.expectEqual(task_a_frame_sp, sp_slot);
    try std.testing.expectEqual(@as(usize, 0xAAAA), try task_a.contextGet(0));
    try std.testing.expectEqual(@as(usize, 0xBBBB), try task_a.contextGet(1));
    try task_a.complete();
    try std.testing.expectEqual(TaskContext.State.completed, task_a.state);
    try std.testing.expectEqual(initial_caller_sp, sp_slot);

    // Resume task B and complete
    task_b.resumeTask();
    try std.testing.expectEqual(task_b_frame_sp, sp_slot);
    try task_b.complete();
    try std.testing.expectEqual(TaskContext.State.completed, task_b.state);
    try std.testing.expectEqual(initial_caller_sp, sp_slot);
}

test "coro: ModuleInstance __stack_pointer global binding and canary overflow on complete" {
    const MockExportKind = enum { func, table, memory, global };
    const MockExport = struct {
        name: []const u8,
        kind: MockExportKind,
        index: u32,
    };
    const MockModule = struct {
        exports: []const MockExport,
        pub fn findExport(self: *const @This(), name: []const u8, kind: MockExportKind) ?MockExport {
            for (self.exports) |exp| {
                if (exp.kind == kind and std.mem.eql(u8, exp.name, name)) return exp;
            }
            return null;
        }
    };
    const MockValue = union(enum) {
        i32: i32,
        i64: i64,
        f32: f32,
    };
    const MockGlobal = struct {
        value: MockValue,
    };
    const MockModuleInstance = struct {
        module: *const MockModule,
        globals: []MockGlobal,
    };

    const exports = [_]MockExport{.{
        .name = "__stack_pointer",
        .kind = .global,
        .index = 0,
    }};
    const wasm_mod = MockModule{ .exports = &exports };
    var globals = [_]MockGlobal{.{
        .value = .{ .i32 = 0x10000 },
    }};
    var mod_inst = MockModuleInstance{
        .module = &wasm_mod,
        .globals = &globals,
    };

    const binding = StackPointerBinding.fromModuleInstance(&mod_inst);
    try std.testing.expectEqual(@as(usize, 0x10000), binding.get());

    var stack_buf: [1024]u8 align(16) = @splat(0);
    const stack = ShadowStack.initGuest(&stack_buf, 0x20000);
    var task = TaskContext.init(stack, binding);

    task.enter();
    try std.testing.expectEqual(@as(i32, @intCast(0x20000 + 1024)), mod_inst.globals[0].value.i32);

    // Simulate guest frame push and suspend
    mod_inst.globals[0].value.i32 = 0x20000 + 512;
    task.suspendTask();
    try std.testing.expectEqual(@as(i32, 0x10000), mod_inst.globals[0].value.i32);

    // Resume and corrupt the canary to verify complete() detects overflow
    task.resumeTask();
    try std.testing.expectEqual(@as(i32, 0x20000 + 512), mod_inst.globals[0].value.i32);
    std.mem.writeInt(u64, stack_buf[0..8], 0, .little);
    try std.testing.expectError(error.StackOverflow, task.complete());
    // Caller SP is still restored even when overflow is detected on completion.
    try std.testing.expectEqual(@as(i32, 0x10000), mod_inst.globals[0].value.i32);
}
