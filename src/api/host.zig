//! Host function registration for embedding WAMR in Zig applications.
//!
//! Provides a comptime-typed API for registering native Zig functions as
//! WebAssembly imports. Generates adapter glue for both interpreter (stack-
//! based) and AOT (C-calling-convention) dispatch automatically.
//!
//! Usage:
//!   const host = @import("wamr").host;
//!
//!   fn add(ctx: host.HostContext, a: i32, b: i32) i32 {
//!       return a + b;
//!   }
//!
//!   const imports = host.HostImports(.{
//!       .{ "env", "add", add },
//!   });

const std = @import("std");
const types = @import("../runtime/common/types.zig");

/// Context passed to host functions — provides access to wasm linear memory.
/// Constructed automatically by the adapter from either ExecEnv (interpreter)
/// or VmCtx (AOT). Host functions should not store this beyond the call.
pub const HostContext = struct {
    mem_base: ?[*]u8,
    mem_len: usize,

    /// Get a mutable slice over the entire wasm linear memory.
    pub fn memory(self: HostContext) ?[]u8 {
        const base = self.mem_base orelse return null;
        return base[0..self.mem_len];
    }

    /// Read a value of type T from linear memory at `offset`.
    pub fn read(self: HostContext, comptime T: type, offset: u32) ?T {
        const size = @sizeOf(T);
        if (offset + size > self.mem_len) return null;
        const base = self.mem_base orelse return null;
        return std.mem.readInt(T, @as(*const [@sizeOf(T)]u8, @ptrCast(base + offset)), .little);
    }

    /// Write a value of type T to linear memory at `offset`.
    pub fn write(self: HostContext, comptime T: type, offset: u32, value: T) bool {
        const size = @sizeOf(T);
        if (offset + size > self.mem_len) return false;
        const base = self.mem_base orelse return false;
        std.mem.writeInt(T, @as(*[@sizeOf(T)]u8, @ptrCast(base + offset)), value, .little);
        return true;
    }

    /// Build a HostContext from an interpreter ExecEnv.
    pub fn fromExecEnv(env_opaque: *anyopaque) HostContext {
        const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
        const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
        const mem = if (env.module_inst.memories.len > 0) env.module_inst.memories[0] else null;
        return .{
            .mem_base = if (mem) |m| m.data.ptr else null,
            .mem_len = if (mem) |m| m.data.len else 0,
        };
    }

    /// Build a HostContext from an AOT VmCtx pointer.
    pub fn fromVmCtx(vmctx: anytype) HostContext {
        return .{
            .mem_base = if (vmctx.memory_base != 0) @ptrFromInt(vmctx.memory_base) else null,
            .mem_len = vmctx.memory_size,
        };
    }
};

/// Valid wasm value types for host function parameters and returns.
fn isWasmType(comptime T: type) bool {
    return T == i32 or T == u32 or T == i64 or T == u64;
}

/// A resolved host import entry.
pub const ImportEntry = struct {
    module_name: []const u8,
    field_name: []const u8,
    interp_fn: types.HostFn,
    aot_fn: *const anyopaque,
};

/// Build a comptime host import table from a tuple of (module, name, function) descriptors.
///
/// Each descriptor is a .{ "module_name", "field_name", zig_function } tuple.
/// The zig function must have signature: fn(HostContext, ...wasm_params) ReturnType
/// where ReturnType is i32, i64, or void, and params are i32/i64.
///
/// Returns a type with:
///   - `entries: [N]ImportEntry` — the resolved import table
///   - `resolve(module, field) ?ImportEntry` — lookup by name
pub fn HostImports(comptime descriptors: anytype) type {
    const N = descriptors.len;

    // Validate all descriptors at comptime
    comptime {
        for (descriptors) |desc| {
            const FnType = @TypeOf(desc[2]);
            const info = @typeInfo(FnType).@"fn";

            // First param must be HostContext
            if (info.params.len == 0)
                @compileError("Host function must take HostContext as first parameter");
            if (info.params[0].type.? != HostContext)
                @compileError("First parameter of host function must be HostContext, got " ++
                    @typeName(info.params[0].type.?));

            // Remaining params must be wasm types
            for (info.params[1..]) |p| {
                if (!isWasmType(p.type.?))
                    @compileError("Host function parameter must be i32, u32, i64, or u64, got " ++
                        @typeName(p.type.?));
            }

            // Return type must be void or a wasm type
            const RetT = info.return_type.?;
            if (RetT != void and !isWasmType(RetT))
                @compileError("Host function must return void, i32, u32, i64, or u64, got " ++
                    @typeName(RetT));
        }
    }

    return struct {
        pub const count = N;

        pub const entries: [N]ImportEntry = blk: {
            var e: [N]ImportEntry = undefined;
            for (descriptors, 0..) |desc, i| {
                e[i] = .{
                    .module_name = desc[0],
                    .field_name = desc[1],
                    .interp_fn = makeInterpAdapter(desc[2]),
                    .aot_fn = makeAotAdapter(desc[2]),
                };
            }
            break :blk e;
        };

        /// Look up a host function by import module + field name.
        pub fn resolve(module_name: []const u8, field_name: []const u8) ?ImportEntry {
            for (entries) |entry| {
                if (std.mem.eql(u8, entry.module_name, module_name) and
                    std.mem.eql(u8, entry.field_name, field_name))
                    return entry;
            }
            return null;
        }
    };
}

/// Generate an interpreter adapter that bridges stack-based dispatch to a
/// typed Zig function. The adapter pops arguments, builds a HostContext,
/// calls the real function, and pushes the result.
fn makeInterpAdapter(comptime func: anytype) types.HostFn {
    const FnType = @TypeOf(func);
    const info = @typeInfo(FnType).@"fn";
    const wasm_param_count = info.params.len - 1; // exclude HostContext
    const RetT = info.return_type.?;

    const S = struct {
        fn adapter(env_opaque: *anyopaque) types.HostFnError!void {
            const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
            const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));

            // Pop arguments in reverse order (stack is LIFO)
            var args: std.meta.ArgsTuple(FnType) = undefined;
            args[0] = HostContext.fromExecEnv(env_opaque);

            comptime var i = wasm_param_count;
            inline while (i > 0) {
                i -= 1;
                const ParamT = info.params[i + 1].type.?;
                args[i + 1] = if (ParamT == i64 or ParamT == u64)
                    @bitCast(env.popI64() catch return error.StackUnderflow)
                else
                    @bitCast(env.popI32() catch return error.StackUnderflow);
            }

            const result = @call(.auto, func, args);

            if (RetT != void) {
                if (RetT == i64 or RetT == u64) {
                    env.pushI64(@bitCast(result)) catch return error.StackOverflow;
                } else {
                    env.pushI32(@bitCast(result)) catch return error.StackOverflow;
                }
            }
        }
    };
    return S.adapter;
}

/// Generate an AOT adapter that bridges C-calling-convention dispatch to a
/// typed Zig function. The AOT codegen calls imported functions with VmCtx
/// as the hidden first parameter and wasm values in subsequent registers.
fn makeAotAdapter(comptime func: anytype) *const anyopaque {
    const VmCtx = @import("../runtime/aot/runtime.zig").VmCtx;
    const FnType = @TypeOf(func);
    const info = @typeInfo(FnType).@"fn";
    const wasm_param_count = info.params.len - 1;
    const RetT = info.return_type.?;
    const AbiRet = if (RetT == void) void else if (RetT == i64 or RetT == u64) i64 else i32;

    const S = struct {
        fn adapter(vmctx: *VmCtx, p0: i64, p1: i64, p2: i64, p3: i64, p4: i64, p5: i64) callconv(.c) AbiRet {
            const ctx = HostContext.fromVmCtx(vmctx);
            const raw = [_]i64{ p0, p1, p2, p3, p4, p5 };
            var args: std.meta.ArgsTuple(FnType) = undefined;
            args[0] = ctx;
            inline for (0..wasm_param_count) |i| {
                const ParamT = info.params[i + 1].type.?;
                args[i + 1] = if (ParamT == i64 or ParamT == u64)
                    @bitCast(raw[i])
                else
                    @as(ParamT, @bitCast(@as(i32, @truncate(raw[i]))));
            }
            const result = @call(.auto, func, args);
            if (RetT == void) return;
            return @bitCast(result);
        }
    };
    return @ptrCast(&S.adapter);
}

// ── Tests ──────────────────────────────────────────────────────────────────

const testing = std.testing;

fn testAdd(_: HostContext, a: i32, b: i32) i32 {
    return a + b;
}

fn testVoid(_: HostContext) void {}

fn testOneParam(_: HostContext, x: i32) i32 {
    return x * 2;
}

fn testI64(_: HostContext, a: i64, b: i64) i64 {
    return a + b;
}

test "HostImports: basic compile and resolve" {
    const imports = HostImports(.{
        .{ "env", "add", testAdd },
        .{ "env", "nop", testVoid },
        .{ "math", "double", testOneParam },
    });

    try testing.expectEqual(@as(usize, 3), imports.count);

    const add_entry = imports.resolve("env", "add");
    try testing.expect(add_entry != null);
    try testing.expectEqualStrings("env", add_entry.?.module_name);
    try testing.expectEqualStrings("add", add_entry.?.field_name);

    const nop_entry = imports.resolve("env", "nop");
    try testing.expect(nop_entry != null);

    const dbl_entry = imports.resolve("math", "double");
    try testing.expect(dbl_entry != null);

    try testing.expect(imports.resolve("env", "missing") == null);
    try testing.expect(imports.resolve("missing", "add") == null);
}

test "HostImports: i64 parameters" {
    const imports = HostImports(.{
        .{ "env", "add64", testI64 },
    });
    try testing.expectEqual(@as(usize, 1), imports.count);
    try testing.expect(imports.resolve("env", "add64") != null);
}

test "HostContext: memory access" {
    var buf = [_]u8{ 0x42, 0x00, 0x00, 0x00, 0xFE, 0xFF, 0xFF, 0xFF };
    const ctx = HostContext{ .mem_base = &buf, .mem_len = buf.len };

    const mem = ctx.memory();
    try testing.expect(mem != null);
    try testing.expectEqual(@as(usize, 8), mem.?.len);

    try testing.expectEqual(@as(i32, 0x42), ctx.read(i32, 0).?);
    try testing.expectEqual(@as(i32, -2), ctx.read(i32, 4).?);

    // Out of bounds
    try testing.expect(ctx.read(i32, 6) == null);

    // Write
    try testing.expect(ctx.write(i32, 0, 99));
    try testing.expectEqual(@as(i32, 99), ctx.read(i32, 0).?);

    // Write OOB
    try testing.expect(!ctx.write(i32, 6, 0));
}

test "HostContext: null memory" {
    const ctx = HostContext{ .mem_base = null, .mem_len = 0 };
    try testing.expect(ctx.memory() == null);
    try testing.expect(ctx.read(i32, 0) == null);
    try testing.expect(!ctx.write(i32, 0, 42));
}
