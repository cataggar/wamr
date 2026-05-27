//! CallFrame — backend-agnostic execution context for canonical-ABI
//! lift/lower.
//!
//! Wraps either an interpreter `ExecEnv` (`InterpFrame`) or an AOT
//! `AotInstance` (`AotFrame`) so that `pushInterfaceValue`,
//! `popInterfaceValue`, `callRealloc`, and the core-function dispatch in
//! `callComponentFuncByLocal` can be written once and target both
//! engines (issue #650).
//!
//! The abstraction is a *tagged union*, not a vtable: Zig's `switch`
//! over `CallFrame` resolves at the call site, so each backend keeps
//! its own struct layout and there is no `*anyopaque` plumbing.
//!
//! Op contract — five methods, mirroring exactly what canon-ABI
//! lift/lower needs from a core engine:
//!
//!   * `pushSlot(v)`        — push one core value in canonical order.
//!   * `popSlot(t)`         — pop one core value, coerced to type `t`.
//!   * `popSlotsU32(out)`   — pop `out.len` slots into `out` in canonical
//!                            forward order (interp's stack is LIFO, so
//!                            it writes back-to-front into `out`).
//!   * `pushSlotsU32(in)`   — push i32-flat slots in canonical order.
//!   * `memory(idx)`        — borrow the active linear memory (may be
//!                            invalidated by any subsequent realloc /
//!                            executeCore on the same frame; callers
//!                            re-fetch after those points).
//!   * `realloc(idx, …)`    — run the core's realloc with the standard
//!                            `(old_ptr, old_size, align, new_size) -> ptr`
//!                            signature.
//!   * `executeCore(idx)`   — drive the core function `idx` with the
//!                            args already pushed (interp: walks the
//!                            operand stack; aot: passes arg buffer to
//!                            `aot_runtime.callFuncScalar`).
//!
//! `AotFrame` is structurally complete in this commit but most ops
//! return `error.AotPathUnsupported`: the AOT path still routes
//! through the parallel `callComponentFuncByLocalAot` helper. The
//! AOT side is fully wired up in the follow-up commit; this commit
//! introduces the abstraction and migrates the interp path onto it,
//! without behaviour change.

const std = @import("std");
const core_types = @import("../runtime/common/types.zig");
const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
const interp = @import("../runtime/interpreter/interp.zig");
const aot_runtime = @import("../runtime/aot/runtime.zig");
const Allocator = std.mem.Allocator;

pub const FrameError = error{
    StackOverflow,
    StackUnderflow,
    MemoryNotAvailable,
    ReallocFailed,
    TrapInCoreFunction,
    OutOfMemory,
    AotPathUnsupported,
};

/// Interp-backed frame. Threads the existing `ExecEnv` operand stack
/// and `interp.executeFunction` into the abstraction.
pub const InterpFrame = struct {
    env: *ExecEnv,

    pub fn init(env: *ExecEnv) InterpFrame {
        return .{ .env = env };
    }

    pub fn pushSlot(self: *InterpFrame, v: core_types.Value) FrameError!void {
        self.env.push(v) catch return error.StackOverflow;
    }

    pub fn popSlot(self: *InterpFrame, t: core_types.ValType) FrameError!core_types.Value {
        return switch (t) {
            .i32 => .{ .i32 = self.env.popI32() catch return error.StackUnderflow },
            .i64 => .{ .i64 = self.env.popI64() catch return error.StackUnderflow },
            .f32 => .{ .f32 = self.env.popF32() catch return error.StackUnderflow },
            .f64 => .{ .f64 = self.env.popF64() catch return error.StackUnderflow },
            else => self.env.pop() catch return error.StackUnderflow,
        };
    }

    /// Fills `out` with the next `out.len` core slots in **canonical
    /// forward order** — i.e. `out[0]` is the slot that was pushed
    /// first (lowest stack index of the window). The interp stack is
    /// LIFO so we pop back-to-front into `out`.
    pub fn popSlotsU32(self: *InterpFrame, out: []u32) FrameError!void {
        var i = out.len;
        while (i > 0) {
            i -= 1;
            const v = self.env.popI32() catch return error.StackUnderflow;
            out[i] = @bitCast(v);
        }
    }

    pub fn pushSlotsU32(self: *InterpFrame, in: []const u32) FrameError!void {
        for (in) |s| self.env.pushI32(@bitCast(s)) catch return error.StackOverflow;
    }

    pub fn memory(self: *InterpFrame, memory_idx: u32) ?[]u8 {
        const mi = self.env.module_inst.getMemory(memory_idx) orelse return null;
        return mi.data;
    }

    pub fn realloc(
        self: *InterpFrame,
        realloc_idx: u32,
        old_ptr: u32,
        old_size: u32,
        alignment: u32,
        new_size: u32,
    ) FrameError!u32 {
        self.env.pushI32(@bitCast(old_ptr)) catch return error.StackOverflow;
        self.env.pushI32(@bitCast(old_size)) catch return error.StackOverflow;
        self.env.pushI32(@bitCast(alignment)) catch return error.StackOverflow;
        self.env.pushI32(@bitCast(new_size)) catch return error.StackOverflow;
        interp.executeFunction(self.env, realloc_idx) catch return error.ReallocFailed;
        const result = self.env.popI32() catch return error.StackUnderflow;
        return @bitCast(result);
    }

    /// Drive the core function `func_idx` with whatever args have
    /// already been pushed via `pushSlot` / `pushSlotsU32`. Results
    /// are left on the stack for the caller to pop via `popSlot` /
    /// `popSlotsU32`. `param_types` and `result_types` are advisory
    /// for the interp (the real signature is read from the module);
    /// they are kept in the API to keep the two impls symmetric.
    pub fn executeCore(
        self: *InterpFrame,
        func_idx: u32,
        param_types: []const core_types.ValType,
        result_types: []const core_types.ValType,
    ) FrameError!void {
        _ = param_types;
        _ = result_types;
        interp.executeFunction(self.env, func_idx) catch return error.TrapInCoreFunction;
    }
};

/// AOT-backed frame. Structurally present so callers can build a
/// `CallFrame` for either backend; the bodies are stubs in this commit
/// (returning `error.AotPathUnsupported`) and get filled in when the
/// AOT path is ported off the parallel `callComponentFuncByLocalAot`
/// helper in the follow-up commit.
pub const AotFrame = struct {
    ai: *aot_runtime.AotInstance,
    /// Args buffered by `pushSlot` / `pushSlotsU32`, consumed by the
    /// next `executeCore`. Caller-owned allocator.
    args: std.ArrayList(core_types.Value),
    arg_types: std.ArrayList(core_types.ValType),
    /// Results from the most recent `executeCore`, consumed by
    /// `popSlot` / `popSlotsU32`.
    results: []core_types.Value = &.{},
    result_cursor: usize = 0,
    allocator: Allocator,

    pub fn init(ai: *aot_runtime.AotInstance, allocator: Allocator) AotFrame {
        return .{
            .ai = ai,
            .args = std.ArrayList(core_types.Value){},
            .arg_types = std.ArrayList(core_types.ValType){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AotFrame) void {
        self.args.deinit(self.allocator);
        self.arg_types.deinit(self.allocator);
        if (self.results.len > 0) self.allocator.free(self.results);
    }

    pub fn pushSlot(self: *AotFrame, v: core_types.Value) FrameError!void {
        self.args.append(self.allocator, v) catch return error.OutOfMemory;
        self.arg_types.append(self.allocator, std.meta.activeTag(v)) catch return error.OutOfMemory;
    }

    pub fn popSlot(self: *AotFrame, t: core_types.ValType) FrameError!core_types.Value {
        _ = self;
        _ = t;
        return error.AotPathUnsupported;
    }

    pub fn popSlotsU32(self: *AotFrame, out: []u32) FrameError!void {
        _ = self;
        _ = out;
        return error.AotPathUnsupported;
    }

    pub fn pushSlotsU32(self: *AotFrame, in: []const u32) FrameError!void {
        for (in) |s| try self.pushSlot(.{ .i32 = @bitCast(s) });
    }

    pub fn memory(self: *AotFrame, memory_idx: u32) ?[]u8 {
        if (memory_idx >= self.ai.memories.len) return null;
        return self.ai.memories[memory_idx].data;
    }

    pub fn realloc(
        self: *AotFrame,
        realloc_idx: u32,
        old_ptr: u32,
        old_size: u32,
        alignment: u32,
        new_size: u32,
    ) FrameError!u32 {
        _ = self;
        _ = realloc_idx;
        _ = old_ptr;
        _ = old_size;
        _ = alignment;
        _ = new_size;
        return error.AotPathUnsupported;
    }

    pub fn executeCore(
        self: *AotFrame,
        func_idx: u32,
        param_types: []const core_types.ValType,
        result_types: []const core_types.ValType,
    ) FrameError!void {
        _ = self;
        _ = func_idx;
        _ = param_types;
        _ = result_types;
        return error.AotPathUnsupported;
    }
};

/// Tagged-union backend-agnostic execution context. See module
/// doc-comment for the op contract.
pub const CallFrame = union(enum) {
    interp: InterpFrame,
    aot: AotFrame,

    pub fn deinit(self: *CallFrame) void {
        switch (self.*) {
            .interp => {},
            .aot => |*f| f.deinit(),
        }
    }

    pub fn pushSlot(self: *CallFrame, v: core_types.Value) FrameError!void {
        return switch (self.*) {
            .interp => |*f| f.pushSlot(v),
            .aot => |*f| f.pushSlot(v),
        };
    }

    pub fn popSlot(self: *CallFrame, t: core_types.ValType) FrameError!core_types.Value {
        return switch (self.*) {
            .interp => |*f| f.popSlot(t),
            .aot => |*f| f.popSlot(t),
        };
    }

    pub fn popSlotsU32(self: *CallFrame, out: []u32) FrameError!void {
        return switch (self.*) {
            .interp => |*f| f.popSlotsU32(out),
            .aot => |*f| f.popSlotsU32(out),
        };
    }

    pub fn pushSlotsU32(self: *CallFrame, in: []const u32) FrameError!void {
        return switch (self.*) {
            .interp => |*f| f.pushSlotsU32(in),
            .aot => |*f| f.pushSlotsU32(in),
        };
    }

    pub fn memory(self: *CallFrame, memory_idx: u32) ?[]u8 {
        return switch (self.*) {
            .interp => |*f| f.memory(memory_idx),
            .aot => |*f| f.memory(memory_idx),
        };
    }

    pub fn realloc(
        self: *CallFrame,
        realloc_idx: u32,
        old_ptr: u32,
        old_size: u32,
        alignment: u32,
        new_size: u32,
    ) FrameError!u32 {
        return switch (self.*) {
            .interp => |*f| f.realloc(realloc_idx, old_ptr, old_size, alignment, new_size),
            .aot => |*f| f.realloc(realloc_idx, old_ptr, old_size, alignment, new_size),
        };
    }

    pub fn executeCore(
        self: *CallFrame,
        func_idx: u32,
        param_types: []const core_types.ValType,
        result_types: []const core_types.ValType,
    ) FrameError!void {
        return switch (self.*) {
            .interp => |*f| f.executeCore(func_idx, param_types, result_types),
            .aot => |*f| f.executeCore(func_idx, param_types, result_types),
        };
    }
};

// Integration is exercised through the existing canon-lift round-trip
// tests in `canonical_abi.zig` and the component-aot canon-lift tests
// in `tests/component_aot_canonlift_test.zig` once the interp and AOT
// paths route through `CallFrame`. No standalone unit tests here yet:
// the AOT-side ops are stubbed in this commit, and instantiating a
// real `ExecEnv` requires a full module pipeline.
