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
//! `AotFrame` and `InterpFrame` ops are fully implemented; the
//! parallel AOT helpers (`callComponentFuncByLocalAot`,
//! `lowerFlatRecur`, `lowerScalarArg`, `liftScalarResult`) were
//! deleted once `callComponentFuncByLocal` started using this
//! abstraction. Every canon-ABI shape that the interp path
//! supports — record/tuple/variant/option/result/flags/list/string
//! and multi-slot scalars — is supported on the AOT path through
//! the same `pushInterfaceValue` / `popInterfaceValue` walks.

const std = @import("std");
const core_types = @import("../runtime/common/types.zig");
const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
const interp = @import("../runtime/interpreter/interp.zig");
const aot_runtime = @import("../runtime/aot/runtime.zig");
const Allocator = std.mem.Allocator;

/// A core-funcidx in the owning module instance's local function index
/// space — directly callable via `CallFrame.realloc` /
/// `CallFrame.executeCore`. Distinct from `executor.CoreFuncIdxComponent`
/// so the compiler rejects passing a component-level idx straight
/// through to a frame call (the #719 bug class). Zero runtime cost.
pub const CoreFuncIdxLocal = enum(u32) {
    _,
    pub inline fn from(raw: u32) CoreFuncIdxLocal {
        return @enumFromInt(raw);
    }
    pub inline fn value(self: CoreFuncIdxLocal) u32 {
        return @intFromEnum(self);
    }
};


pub const FrameError = error{
    StackOverflow,
    StackUnderflow,
    MemoryNotAvailable,
    ReallocFailed,
    TrapInCoreFunction,
    OutOfMemory,
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
        realloc_idx: CoreFuncIdxLocal,
        old_ptr: u32,
        old_size: u32,
        alignment: u32,
        new_size: u32,
    ) FrameError!u32 {
        self.env.pushI32(@bitCast(old_ptr)) catch return error.StackOverflow;
        self.env.pushI32(@bitCast(old_size)) catch return error.StackOverflow;
        self.env.pushI32(@bitCast(alignment)) catch return error.StackOverflow;
        self.env.pushI32(@bitCast(new_size)) catch return error.StackOverflow;
        interp.executeFunction(self.env, realloc_idx.value()) catch return error.ReallocFailed;
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
        func_idx: CoreFuncIdxLocal,
        param_types: []const core_types.ValType,
        result_types: []const core_types.ValType,
    ) FrameError!void {
        _ = param_types;
        _ = result_types;
        interp.executeFunction(self.env, func_idx.value()) catch return error.TrapInCoreFunction;
    }
};

/// AOT-backed frame. Buffers args lowered via `pushSlot` /
/// `pushSlotsU32`, then drives `aot_runtime.callFuncScalar` on
/// `executeCore`. Result slots from the core function are surfaced
/// in canonical forward order via `popSlot` (cursor-advance into
/// `results`), matching the abstraction the interp side gets from
/// the operand stack.
pub const AotFrame = struct {
    ai: *aot_runtime.AotInstance,
    /// Args buffered by `pushSlot` / `pushSlotsU32`, consumed by the
    /// next `executeCore`. Caller-owned allocator.
    args: std.ArrayList(core_types.Value),
    arg_types: std.ArrayList(core_types.ValType),
    /// Results from the most recent `executeCore`, consumed by
    /// `popSlot` / `popSlotsU32`.
    results: []aot_runtime.ScalarResult = &.{},
    result_cursor: usize = 0,
    allocator: Allocator,

    pub fn init(ai: *aot_runtime.AotInstance, allocator: Allocator) AotFrame {
        return .{
            .ai = ai,
            .args = .empty,
            .arg_types = .empty,
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
        if (self.result_cursor == 0) return error.StackUnderflow;
        self.result_cursor -= 1;
        const sr = self.results[self.result_cursor];
        return switch (t) {
            .i32 => switch (sr) {
                .i32 => |x| .{ .i32 = x },
                .i64 => |x| .{ .i32 = @truncate(x) },
                .funcref => |fr| .{ .i32 = @bitCast(@as(u32, @truncate(fr orelse 0))) },
                .externref => |er| .{ .i32 = @bitCast(@as(u32, @truncate(er orelse 0))) },
                else => return error.StackUnderflow,
            },
            .i64 => switch (sr) {
                .i64 => |x| .{ .i64 = x },
                .i32 => |x| .{ .i64 = @as(i64, x) },
                else => return error.StackUnderflow,
            },
            .f32 => switch (sr) {
                .f32 => |x| .{ .f32 = @bitCast(x) },
                .i32 => |x| .{ .f32 = @bitCast(x) },
                else => return error.StackUnderflow,
            },
            .f64 => switch (sr) {
                .f64 => |x| .{ .f64 = @bitCast(x) },
                .i64 => |x| .{ .f64 = @bitCast(x) },
                else => return error.StackUnderflow,
            },
            else => return error.StackUnderflow,
        };
    }

    pub fn popSlotsU32(self: *AotFrame, out: []u32) FrameError!void {
        var i = out.len;
        while (i > 0) {
            i -= 1;
            const v = try self.popSlot(.i32);
            out[i] = @bitCast(v.i32);
        }
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
        realloc_idx: CoreFuncIdxLocal,
        old_ptr: u32,
        old_size: u32,
        alignment: u32,
        new_size: u32,
    ) FrameError!u32 {
        const args = [_]core_types.Value{
            .{ .i32 = @bitCast(old_ptr) },
            .{ .i32 = @bitCast(old_size) },
            .{ .i32 = @bitCast(alignment) },
            .{ .i32 = @bitCast(new_size) },
        };
        const param_types = [_]core_types.ValType{ .i32, .i32, .i32, .i32 };
        const result_types = [_]core_types.ValType{.i32};
        var rbuf: [1]aot_runtime.ScalarResult = .{.{ .i32 = 0 }};
        const rres = aot_runtime.callFuncScalar(
            self.ai,
            realloc_idx.value(),
            &param_types,
            &result_types,
            &args,
            &rbuf,
        ) catch return error.ReallocFailed;
        return switch (rres[0]) {
            .i32 => |x| @bitCast(x),
            else => error.ReallocFailed,
        };
    }

    pub fn executeCore(
        self: *AotFrame,
        func_idx: CoreFuncIdxLocal,
        param_types: []const core_types.ValType,
        result_types: []const core_types.ValType,
    ) FrameError!void {
        // Free the previous result buffer (if any) before running the
        // next core call; popSlot consumed up to result_cursor but the
        // backing slice itself is owned by the frame.
        if (self.results.len > 0) {
            self.allocator.free(self.results);
            self.results = &.{};
            self.result_cursor = 0;
        }
        // `arg_types` is the canonical list of types we recorded as args
        // were pushed via `pushSlot`. The advisory `param_types` from
        // the caller is ignored — the active source of truth is what
        // the canon-ABI lowering actually emitted.
        _ = param_types;

        const results_out = self.allocator.alloc(aot_runtime.ScalarResult, result_types.len) catch
            return error.OutOfMemory;
        errdefer self.allocator.free(results_out);

        const got = aot_runtime.callFuncScalar(
            self.ai,
            func_idx.value(),
            self.arg_types.items,
            result_types,
            self.args.items,
            results_out,
        ) catch return error.TrapInCoreFunction;

        // Reset args for the next call (e.g. a subsequent post_return).
        self.args.clearRetainingCapacity();
        self.arg_types.clearRetainingCapacity();

        // Convert the (slice-into-out) view into an owned slice of the
        // same allocator. `aot_runtime.callFuncScalar` returns a slice
        // pointing into `results_out`; we keep the whole `results_out`
        // buffer to back popSlot.
        std.debug.assert(got.ptr == results_out.ptr);
        self.results = results_out;
        self.result_cursor = got.len;
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
        realloc_idx: CoreFuncIdxLocal,
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
        func_idx: CoreFuncIdxLocal,
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
