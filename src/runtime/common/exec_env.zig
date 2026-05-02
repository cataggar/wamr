//! Execution environment for the WebAssembly interpreter.
//!
//! Each thread of execution gets its own ExecEnv, which contains:
//! - An operand stack for Wasm values
//! - A call frame stack tracking function calls
//! - A reference to the current module instance

const std = @import("std");
const types = @import("types.zig");

/// A stored exception reference (for catch_ref / throw_ref).
pub const ExceptionRef = struct {
    tag: ?*types.TagInstance = null,
    params: [16]types.Value = undefined,
    param_count: u32 = 0,
};

/// Diagnostic info for a host-fn / trampoline trap. Populated lazily on
/// the trap path; consumers should treat empty / sentinel fields as "unknown".
/// All string fields are static (e.g. `@errorName(err)` or slices borrowed
/// from the WasmModule's imports list, which outlives the ExecEnv), so this
/// struct is freely copyable and never owns memory.
pub const HostTrapInfo = struct {
    /// Core import-function index that was being dispatched, or
    /// `std.math.maxInt(u32)` if not recorded by the interpreter loop.
    core_func_idx: u32 = std.math.maxInt(u32),
    /// Component-level function index for the canon-lower trampoline that
    /// failed, or `std.math.maxInt(u32)` if not from a trampoline.
    component_func_idx: u32 = std.math.maxInt(u32),
    /// `@errorName(err)` of the original (unsquashed) error. Static.
    err_name: []const u8 = "",
    /// Core import module name (e.g. "wasi:io/streams"), borrowed from the
    /// WasmModule's import list. Empty if not resolvable.
    import_module_name: []const u8 = "",
    /// Core import field name (e.g. "[method]output-stream.blocking-write-and-flush").
    import_field_name: []const u8 = "",
    /// Site within the canon-lower trampoline at which the trap occurred,
    /// or `.host_dispatch` for the bare interpreter dispatch path.
    stage: Stage = .host_dispatch,

    pub const Stage = enum {
        /// Host fn dispatched from the interp loop returned an error and
        /// no inner trampoline recorded a more specific stage.
        host_dispatch,
        /// Failed before the host call: lifting args from the core stack
        /// (pop, memory load, alloc).
        lift_args,
        /// The bound `HostFunc.call` returned an error.
        host_call,
        /// Failed after the host call while lowering results back onto the
        /// core stack or into linear memory.
        lower_results,
        /// Spill path needed memory but `lower_opts.memory_idx` was unset
        /// or the indexed memory could not be resolved.
        memory_resolve,
    };
};

/// A single call frame on the call stack.
pub const CallFrame = struct {
    /// Index of the function being executed.
    func_idx: u32,
    /// Instruction pointer (offset into the function's bytecode).
    ip: u32 = 0,
    /// Base index into the operand stack for this frame's locals.
    stack_base: u32,
    /// Number of local variables (including parameters).
    local_count: u32,
    /// Number of values to return.
    return_arity: u32,
    /// Previous stack pointer (for restoring on return).
    prev_sp: u32,
};

/// The execution environment — per-thread interpreter state.
pub const ExecEnv = struct {
    /// The module instance this environment is bound to.
    module_inst: *types.ModuleInstance,

    /// Operand stack (fixed-size, allocated at creation).
    operand_stack: []types.Value,
    /// Current operand stack pointer (index into operand_stack).
    sp: u32 = 0,

    /// Call frame stack.
    call_stack: []CallFrame,
    /// Current call frame depth.
    call_depth: u32 = 0,

    /// Maximum call depth (configurable, default 1024).
    max_call_depth: u32 = 1024,

    /// Exception/trap message (null if no trap).
    exception: ?[]const u8 = null,

    /// Pending uncaught exception tag (set by throw when no handler found).
    pending_exception_tag: ?*types.TagInstance = null,
    /// Parameter values for the pending exception.
    pending_exception_params: [16]types.Value = undefined,
    /// Number of valid entries in pending_exception_params.
    pending_exception_param_count: u32 = 0,

    /// Exception reference pool for exnref values (catch_ref / throw_ref).
    exception_refs: [8]ExceptionRef = undefined,
    /// Number of allocated exception refs.
    exception_ref_count: u32 = 0,

    /// Allocator used for this env's memory.
    allocator: std.mem.Allocator,

    /// Thread manager for cross-thread trap propagation (null for single-threaded).
    thread_manager: ?*@import("../../wasi/thread_manager.zig").ThreadManager = null,

    /// Thread ID (0 for main thread).
    tid: i32 = 0,

    /// Diagnostic info for the most recent host-fn or trampoline trap. Set by
    /// the interpreter dispatch loop and the canon-lower trampoline before
    /// they squash the underlying error to `error.Unreachable` / `error.Trap`.
    /// First-write-wins: the deepest site (typically the `WasiCliAdapter`
    /// host call inside the trampoline) records the most informative value,
    /// and outer layers only fill in fields that were left unset.
    /// Surfaces in `callComponentFunc` so a `TrapInCoreFunction` is no longer
    /// an opaque "Unreachable" — see issue #308.
    host_trap: ?HostTrapInfo = null,

    /// Create a new execution environment.
    pub fn create(
        module_inst: *types.ModuleInstance,
        stack_size: u32,
        allocator: std.mem.Allocator,
    ) !*ExecEnv {
        const self = try allocator.create(ExecEnv);
        errdefer allocator.destroy(self);

        const stack = try allocator.alloc(types.Value, stack_size);
        errdefer allocator.free(stack);

        const frames = try allocator.alloc(CallFrame, 1024);
        errdefer allocator.free(frames);

        self.* = .{
            .module_inst = module_inst,
            .operand_stack = stack,
            .call_stack = frames,
            .allocator = allocator,
        };
        return self;
    }

    /// Destroy the execution environment and free all memory.
    pub fn destroy(self: *ExecEnv) void {
        self.allocator.free(self.operand_stack);
        self.allocator.free(self.call_stack);
        self.allocator.destroy(self);
    }

    // -- Stack operations --

    /// Push a value onto the operand stack.
    pub fn push(self: *ExecEnv, val: types.Value) !void {
        if (self.sp >= self.operand_stack.len) return error.StackOverflow;
        self.operand_stack[self.sp] = val;
        self.sp += 1;
    }

    /// Pop a value from the operand stack.
    pub fn pop(self: *ExecEnv) !types.Value {
        if (self.sp == 0) return error.StackUnderflow;
        self.sp -= 1;
        return self.operand_stack[self.sp];
    }

    /// Peek at the top value without popping.
    pub fn peek(self: *const ExecEnv) !types.Value {
        if (self.sp == 0) return error.StackUnderflow;
        return self.operand_stack[self.sp - 1];
    }

    /// Push typed values
    pub fn pushI32(self: *ExecEnv, val: i32) !void {
        try self.push(.{ .i32 = val });
    }
    pub fn pushI64(self: *ExecEnv, val: i64) !void {
        try self.push(.{ .i64 = val });
    }
    pub fn pushF32(self: *ExecEnv, val: f32) !void {
        try self.push(.{ .f32 = val });
    }
    pub fn pushF64(self: *ExecEnv, val: f64) !void {
        try self.push(.{ .f64 = val });
    }

    /// Pop typed values
    pub fn popI32(self: *ExecEnv) !i32 {
        const v = try self.pop();
        return switch (v) {
            .i32 => |i| i,
            .f32 => |f| @bitCast(f),
            .funcref, .nonfuncref => 0, // null ref as 0
            .externref, .nonexternref => 0,
            .i64 => |i| @as(i32, @bitCast(@as(u32, @truncate(@as(u64, @bitCast(i)))))),
            .f64 => |f| @as(i32, @bitCast(@as(u32, @truncate(@as(u64, @bitCast(f)))))),
            else => 0,
        };
    }
    pub fn popI64(self: *ExecEnv) !i64 {
        const v = try self.pop();
        return switch (v) {
            .i64 => |i| i,
            .f64 => |f| @bitCast(f),
            .i32 => |i| @as(i64, i),
            .f32 => |f| @as(i64, @as(i32, @bitCast(f))),
            else => 0,
        };
    }
    pub fn popF32(self: *ExecEnv) !f32 {
        const v = try self.pop();
        return switch (v) {
            .f32 => |f| f,
            .i32 => |i| @bitCast(i),
            .i64 => |i| @bitCast(@as(u32, @truncate(@as(u64, @bitCast(i))))),
            .f64 => |f| @floatCast(f),
            else => 0.0,
        };
    }
    pub fn popF64(self: *ExecEnv) !f64 {
        const v = try self.pop();
        return switch (v) {
            .f64 => |f| f,
            .i64 => |i| @bitCast(i),
            .i32 => |i| @bitCast(@as(i64, i)),
            .f32 => |f| @floatCast(f),
            else => 0.0,
        };
    }

    // -- Call frame operations --

    /// Push a new call frame.
    pub fn pushFrame(self: *ExecEnv, frame: CallFrame) !void {
        if (self.call_depth >= self.call_stack.len) return error.CallStackOverflow;
        if (self.call_depth >= self.max_call_depth) return error.CallStackOverflow;
        self.call_stack[self.call_depth] = frame;
        self.call_depth += 1;
    }

    /// Pop the current call frame.
    pub fn popFrame(self: *ExecEnv) !CallFrame {
        if (self.call_depth == 0) return error.CallStackUnderflow;
        self.call_depth -= 1;
        return self.call_stack[self.call_depth];
    }

    /// Get the current (top) call frame.
    pub fn currentFrame(self: *const ExecEnv) ?*const CallFrame {
        if (self.call_depth == 0) return null;
        return &self.call_stack[self.call_depth - 1];
    }

    /// Get a mutable reference to the current call frame.
    pub fn currentFrameMut(self: *ExecEnv) ?*CallFrame {
        if (self.call_depth == 0) return null;
        return &self.call_stack[self.call_depth - 1];
    }

    // -- Local variable access --

    /// Get a local variable value.
    pub fn getLocal(self: *const ExecEnv, frame: *const CallFrame, idx: u32) !types.Value {
        if (idx >= frame.local_count) return error.StackOverflow;
        const abs = frame.stack_base + idx;
        if (abs >= self.operand_stack.len) return error.StackOverflow;
        return self.operand_stack[abs];
    }

    /// Set a local variable value.
    pub fn setLocal(self: *ExecEnv, frame: *const CallFrame, idx: u32, val: types.Value) !void {
        if (idx >= frame.local_count) return error.StackOverflow;
        const abs = frame.stack_base + idx;
        if (abs >= self.operand_stack.len) return error.StackOverflow;
        self.operand_stack[abs] = val;
    }

    // -- Trap handling --

    /// Set an exception/trap message.
    pub fn setException(self: *ExecEnv, msg: []const u8) void {
        self.exception = msg;
    }

    /// Clear any pending exception.
    pub fn clearException(self: *ExecEnv) void {
        self.exception = null;
    }

    /// Check if there's a pending exception.
    pub fn hasException(self: *const ExecEnv) bool {
        return self.exception != null;
    }
};

// ── Tests ──────────────────────────────────────────────────────────────────

const TestEnv = struct {
    env: *ExecEnv,
    inst: *types.ModuleInstance,
    module: *const types.WasmModule,

    fn deinit(self: TestEnv) void {
        const allocator = std.testing.allocator;
        self.env.destroy();
        allocator.destroy(@constCast(self.module));
        allocator.destroy(self.inst);
    }
};

fn createTestEnv(stack_size: u32) !TestEnv {
    const allocator = std.testing.allocator;
    const module = try allocator.create(types.WasmModule);
    module.* = .{};
    const inst = try allocator.create(types.ModuleInstance);
    inst.* = .{
        .module = module,
        .memories = &.{},
        .tables = &.{},
        .globals = &.{},
        .allocator = allocator,
    };
    const env = try ExecEnv.create(inst, stack_size, allocator);
    return .{ .env = env, .inst = inst, .module = module };
}

test "create and destroy lifecycle" {
    const t = try createTestEnv(64);
    defer t.deinit();
    const env = t.env;

    try std.testing.expectEqual(@as(u32, 0), env.sp);
    try std.testing.expectEqual(@as(u32, 0), env.call_depth);
    try std.testing.expectEqual(@as(u32, 1024), env.max_call_depth);
    try std.testing.expect(env.exception == null);
    try std.testing.expectEqual(@as(usize, 64), env.operand_stack.len);
}

test "push and pop i32" {
    const t = try createTestEnv(16);
    defer t.deinit();
    const env = t.env;

    try env.pushI32(42);
    try std.testing.expectEqual(@as(u32, 1), env.sp);

    const val = try env.popI32();
    try std.testing.expectEqual(@as(i32, 42), val);
    try std.testing.expectEqual(@as(u32, 0), env.sp);
}

test "push and pop i64" {
    const t = try createTestEnv(16);
    defer t.deinit();
    const env = t.env;

    try env.pushI64(0x1_0000_0000);
    const val = try env.popI64();
    try std.testing.expectEqual(@as(i64, 0x1_0000_0000), val);
}

test "push and pop f32" {
    const t = try createTestEnv(16);
    defer t.deinit();
    const env = t.env;

    try env.pushF32(3.14);
    const val = try env.popF32();
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), val, 0.001);
}

test "push and pop f64" {
    const t = try createTestEnv(16);
    defer t.deinit();
    const env = t.env;

    try env.pushF64(2.718281828);
    const val = try env.popF64();
    try std.testing.expectApproxEqAbs(@as(f64, 2.718281828), val, 0.000001);
}

test "peek does not consume" {
    const t = try createTestEnv(16);
    defer t.deinit();
    const env = t.env;

    try env.pushI32(99);
    const peeked = try env.peek();
    try std.testing.expectEqual(@as(i32, 99), peeked.i32);
    try std.testing.expectEqual(@as(u32, 1), env.sp);
}

test "stack overflow detection" {
    const t = try createTestEnv(2);
    defer t.deinit();
    const env = t.env;

    try env.pushI32(1);
    try env.pushI32(2);
    try std.testing.expectError(error.StackOverflow, env.pushI32(3));
}

test "stack underflow detection" {
    const t = try createTestEnv(16);
    defer t.deinit();
    const env = t.env;

    try std.testing.expectError(error.StackUnderflow, env.pop());
    try std.testing.expectError(error.StackUnderflow, env.peek());
}

test "push and pop call frames" {
    const t = try createTestEnv(64);
    defer t.deinit();
    const env = t.env;

    const frame = CallFrame{
        .func_idx = 5,
        .ip = 0,
        .stack_base = 0,
        .local_count = 3,
        .return_arity = 1,
        .prev_sp = 0,
    };
    try env.pushFrame(frame);
    try std.testing.expectEqual(@as(u32, 1), env.call_depth);

    const cur = env.currentFrame().?;
    try std.testing.expectEqual(@as(u32, 5), cur.func_idx);

    const popped = try env.popFrame();
    try std.testing.expectEqual(@as(u32, 5), popped.func_idx);
    try std.testing.expectEqual(@as(u32, 0), env.call_depth);
    try std.testing.expect(env.currentFrame() == null);
}

test "call stack overflow detection" {
    const t = try createTestEnv(16);
    defer t.deinit();
    const env = t.env;

    // Set a low max for testing.
    env.max_call_depth = 2;

    const frame = CallFrame{
        .func_idx = 0,
        .stack_base = 0,
        .local_count = 0,
        .return_arity = 0,
        .prev_sp = 0,
    };
    try env.pushFrame(frame);
    try env.pushFrame(frame);
    try std.testing.expectError(error.CallStackOverflow, env.pushFrame(frame));
}

test "call stack underflow detection" {
    const t = try createTestEnv(16);
    defer t.deinit();
    const env = t.env;

    try std.testing.expectError(error.CallStackUnderflow, env.popFrame());
}

test "local variable get and set" {
    const t = try createTestEnv(64);
    defer t.deinit();
    const env = t.env;

    // Reserve space for locals on the operand stack.
    try env.pushI32(0); // local 0 at stack_base=0
    try env.pushI32(0); // local 1
    try env.pushI32(0); // local 2

    const frame = CallFrame{
        .func_idx = 0,
        .stack_base = 0,
        .local_count = 3,
        .return_arity = 0,
        .prev_sp = 0,
    };
    try env.pushFrame(frame);

    const cur = env.currentFrame().?;
    try env.setLocal(cur, 1, .{ .i32 = 777 });
    const val = try env.getLocal(cur, 1);
    try std.testing.expectEqual(@as(i32, 777), val.i32);
}

test "currentFrameMut allows modification" {
    const t = try createTestEnv(16);
    defer t.deinit();
    const env = t.env;

    const frame = CallFrame{
        .func_idx = 0,
        .stack_base = 0,
        .local_count = 0,
        .return_arity = 0,
        .prev_sp = 0,
    };
    try env.pushFrame(frame);

    const cur = env.currentFrameMut().?;
    cur.ip = 42;
    try std.testing.expectEqual(@as(u32, 42), env.currentFrame().?.ip);
}

test "exception set, has, clear" {
    const t = try createTestEnv(16);
    defer t.deinit();
    const env = t.env;

    try std.testing.expect(!env.hasException());

    env.setException("integer overflow");
    try std.testing.expect(env.hasException());
    try std.testing.expectEqualStrings("integer overflow", env.exception.?);

    env.clearException();
    try std.testing.expect(!env.hasException());
}
