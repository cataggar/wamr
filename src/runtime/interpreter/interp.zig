//! WebAssembly bytecode interpreter.
//!
//! Executes Wasm functions using a switch-based dispatch loop.
//! Supports MVP i32 arithmetic, comparisons, locals, globals,
//! memory load/store, and parametric instructions.

const std = @import("std");
const types = @import("../common/types.zig");
const ExecEnv = @import("../common/exec_env.zig").ExecEnv;
const CallFrame = @import("../common/exec_env.zig").CallFrame;
const Opcode = @import("opcode.zig").Opcode;

pub const TrapError = error{
    Unreachable,
    IntegerOverflow,
    IntegerDivisionByZero,
    InvalidConversionToInteger,
    StackOverflow,
    StackUnderflow,
    CallStackOverflow,
    CallStackUnderflow,
    OutOfBoundsMemoryAccess,
    UnknownFunction,
    UnknownOpcode,
};

// ── LEB128 helpers for bytecode ──────────────────────────────────────────

fn readU32(code: []const u8, ip: *usize) u32 {
    var result: u32 = 0;
    var shift: u5 = 0;
    while (true) {
        if (ip.* >= code.len) return result;
        const byte = code[ip.*];
        ip.* += 1;
        result |= @as(u32, byte & 0x7F) << shift;
        if (byte & 0x80 == 0) break;
        if (shift >= 28) break;
        shift +|= 7;
    }
    return result;
}

fn readI32(code: []const u8, ip: *usize) i32 {
    var result: u32 = 0;
    var shift: u5 = 0;
    var byte: u8 = 0;
    while (true) {
        if (ip.* >= code.len) return @bitCast(result);
        byte = code[ip.*];
        ip.* += 1;
        result |= @as(u32, byte & 0x7F) << shift;
        if (byte & 0x80 == 0) break;
        if (shift >= 28) break;
        shift +|= 7;
    }
    // sign-extend
    if (shift < 32 and (byte & 0x40) != 0) {
        result |= @as(u32, 0xFFFFFFFF) << shift;
    }
    return @bitCast(result);
}

fn readI64(code: []const u8, ip: *usize) i64 {
    var result: u64 = 0;
    var shift: u6 = 0;
    var byte: u8 = 0;
    while (true) {
        if (ip.* >= code.len) return @bitCast(result);
        byte = code[ip.*];
        ip.* += 1;
        result |= @as(u64, byte & 0x7F) << shift;
        if (byte & 0x80 == 0) break;
        if (shift >= 63) break;
        shift +|= 7;
    }
    if (shift < 64 and (byte & 0x40) != 0) {
        result |= @as(u64, 0xFFFFFFFFFFFFFFFF) << shift;
    }
    return @bitCast(result);
}

// ── Public API ───────────────────────────────────────────────────────────

/// Execute a function by index within the module instance.
pub fn executeFunction(env: *ExecEnv, func_idx: u32) TrapError!void {
    const module = env.module_inst.module;
    if (func_idx < module.import_function_count) return error.UnknownFunction;
    const local_idx = func_idx - module.import_function_count;
    if (local_idx >= module.functions.len) return error.UnknownFunction;

    const func = &module.functions[local_idx];
    const func_type = module.types[func.type_idx];
    const param_count: u32 = @intCast(func_type.params.len);

    var total_locals: u32 = param_count;
    for (func.locals) |local| total_locals += local.count;

    const stack_base = env.sp - param_count;
    try env.pushFrame(.{
        .func_idx = func_idx,
        .ip = 0,
        .stack_base = stack_base,
        .local_count = total_locals,
        .return_arity = @intCast(func_type.results.len),
        .prev_sp = env.sp,
    });

    // Initialize non-param locals to zero.
    var i: u32 = 0;
    while (i < total_locals - param_count) : (i += 1) {
        try env.push(.{ .i32 = 0 });
    }

    try dispatchLoop(env, func.code);
}

// ── Dispatch loop ────────────────────────────────────────────────────────

fn dispatchLoop(env: *ExecEnv, code: []const u8) TrapError!void {
    var ip: usize = 0;

    while (ip < code.len) {
        const byte = code[ip];
        ip += 1;
        const op: Opcode = @enumFromInt(byte);

        switch (op) {
            // ── Control ──
            .@"unreachable" => return error.Unreachable,
            .nop => {},
            .end, .@"return" => return,

            // ── Constants ──
            .i32_const => try env.pushI32(readI32(code, &ip)),
            .i64_const => try env.pushI64(readI64(code, &ip)),
            .f32_const => {
                if (ip + 4 > code.len) return error.OutOfBoundsMemoryAccess;
                const v: f32 = @bitCast(std.mem.readInt(u32, code[ip..][0..4], .little));
                ip += 4;
                try env.pushF32(v);
            },
            .f64_const => {
                if (ip + 8 > code.len) return error.OutOfBoundsMemoryAccess;
                const v: f64 = @bitCast(std.mem.readInt(u64, code[ip..][0..8], .little));
                ip += 8;
                try env.pushF64(v);
            },

            // ── Locals ──
            .local_get => {
                const idx = readU32(code, &ip);
                const frame = env.currentFrame() orelse return error.CallStackUnderflow;
                try env.push(env.getLocal(frame, idx));
            },
            .local_set => {
                const idx = readU32(code, &ip);
                const frame = env.currentFrame() orelse return error.CallStackUnderflow;
                env.setLocal(frame, idx, try env.pop());
            },
            .local_tee => {
                const idx = readU32(code, &ip);
                const frame = env.currentFrame() orelse return error.CallStackUnderflow;
                env.setLocal(frame, idx, try env.peek());
            },

            // ── Globals ──
            .global_get => {
                const idx = readU32(code, &ip);
                if (idx >= env.module_inst.globals.len) return error.OutOfBoundsMemoryAccess;
                try env.push(env.module_inst.globals[idx].value);
            },
            .global_set => {
                const idx = readU32(code, &ip);
                if (idx >= env.module_inst.globals.len) return error.OutOfBoundsMemoryAccess;
                env.module_inst.globals[idx].value = try env.pop();
            },

            // ── Parametric ──
            .drop => _ = try env.pop(),
            .select => {
                const c = try env.popI32();
                const b = try env.pop();
                const a = try env.pop();
                try env.push(if (c != 0) a else b);
            },

            // ── i32 comparison ──
            .i32_eqz => try env.pushI32(@intFromBool(try env.popI32() == 0)),
            .i32_eq => {
                const b = try env.popI32();
                try env.pushI32(@intFromBool(try env.popI32() == b));
            },
            .i32_ne => {
                const b = try env.popI32();
                try env.pushI32(@intFromBool(try env.popI32() != b));
            },
            .i32_lt_s => {
                const b = try env.popI32();
                try env.pushI32(@intFromBool(try env.popI32() < b));
            },
            .i32_lt_u => {
                const b: u32 = @bitCast(try env.popI32());
                const a: u32 = @bitCast(try env.popI32());
                try env.pushI32(@intFromBool(a < b));
            },
            .i32_gt_s => {
                const b = try env.popI32();
                try env.pushI32(@intFromBool(try env.popI32() > b));
            },
            .i32_gt_u => {
                const b: u32 = @bitCast(try env.popI32());
                const a: u32 = @bitCast(try env.popI32());
                try env.pushI32(@intFromBool(a > b));
            },
            .i32_le_s => {
                const b = try env.popI32();
                try env.pushI32(@intFromBool(try env.popI32() <= b));
            },
            .i32_le_u => {
                const b: u32 = @bitCast(try env.popI32());
                const a: u32 = @bitCast(try env.popI32());
                try env.pushI32(@intFromBool(a <= b));
            },
            .i32_ge_s => {
                const b = try env.popI32();
                try env.pushI32(@intFromBool(try env.popI32() >= b));
            },
            .i32_ge_u => {
                const b: u32 = @bitCast(try env.popI32());
                const a: u32 = @bitCast(try env.popI32());
                try env.pushI32(@intFromBool(a >= b));
            },

            // ── i32 unary ──
            .i32_clz => {
                const a: u32 = @bitCast(try env.popI32());
                try env.pushI32(@bitCast(@as(u32, @clz(a))));
            },
            .i32_ctz => {
                const a: u32 = @bitCast(try env.popI32());
                try env.pushI32(@bitCast(@as(u32, @ctz(a))));
            },
            .i32_popcnt => {
                const a: u32 = @bitCast(try env.popI32());
                try env.pushI32(@bitCast(@as(u32, @popCount(a))));
            },

            // ── i32 arithmetic ──
            .i32_add => { const b = try env.popI32(); try env.pushI32((try env.popI32()) +% b); },
            .i32_sub => { const b = try env.popI32(); try env.pushI32((try env.popI32()) -% b); },
            .i32_mul => { const b = try env.popI32(); try env.pushI32((try env.popI32()) *% b); },
            .i32_div_s => {
                const b = try env.popI32();
                const a = try env.popI32();
                if (b == 0) return error.IntegerDivisionByZero;
                if (a == std.math.minInt(i32) and b == -1) return error.IntegerOverflow;
                try env.pushI32(@divTrunc(a, b));
            },
            .i32_div_u => {
                const b: u32 = @bitCast(try env.popI32());
                const a: u32 = @bitCast(try env.popI32());
                if (b == 0) return error.IntegerDivisionByZero;
                try env.pushI32(@bitCast(a / b));
            },
            .i32_rem_s => {
                const b = try env.popI32();
                const a = try env.popI32();
                if (b == 0) return error.IntegerDivisionByZero;
                try env.pushI32(if (a == std.math.minInt(i32) and b == -1) 0 else @rem(a, b));
            },
            .i32_rem_u => {
                const b: u32 = @bitCast(try env.popI32());
                const a: u32 = @bitCast(try env.popI32());
                if (b == 0) return error.IntegerDivisionByZero;
                try env.pushI32(@bitCast(a % b));
            },
            .i32_and => { const b = try env.popI32(); try env.pushI32((try env.popI32()) & b); },
            .i32_or => { const b = try env.popI32(); try env.pushI32((try env.popI32()) | b); },
            .i32_xor => { const b = try env.popI32(); try env.pushI32((try env.popI32()) ^ b); },
            .i32_shl => {
                const b: u32 = @bitCast(try env.popI32());
                const a: u32 = @bitCast(try env.popI32());
                try env.pushI32(@bitCast(a << @intCast(b % 32)));
            },
            .i32_shr_s => {
                const b: u32 = @bitCast(try env.popI32());
                const a = try env.popI32();
                try env.pushI32(a >> @intCast(b % 32));
            },
            .i32_shr_u => {
                const b: u32 = @bitCast(try env.popI32());
                const a: u32 = @bitCast(try env.popI32());
                try env.pushI32(@bitCast(a >> @intCast(b % 32)));
            },
            .i32_rotl => {
                const b: u32 = @bitCast(try env.popI32());
                const a: u32 = @bitCast(try env.popI32());
                try env.pushI32(@bitCast(std.math.rotl(u32, a, b % 32)));
            },
            .i32_rotr => {
                const b: u32 = @bitCast(try env.popI32());
                const a: u32 = @bitCast(try env.popI32());
                try env.pushI32(@bitCast(std.math.rotr(u32, a, b % 32)));
            },

            // ── Memory (i32 only) ──
            .i32_load => {
                _ = readU32(code, &ip); // alignment hint
                const offset = readU32(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + offset;
                const mem = env.module_inst.getMemory(0) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 4 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                try env.pushI32(std.mem.readInt(i32, mem.data[a..][0..4], .little));
            },
            .i32_store => {
                _ = readU32(code, &ip);
                const offset = readU32(code, &ip);
                const val = try env.popI32();
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + offset;
                const mem = env.module_inst.getMemory(0) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 4 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                std.mem.writeInt(i32, mem.data[a..][0..4], val, .little);
            },

            // ── Conversions (subset) ──
            .i32_wrap_i64 => try env.pushI32(@truncate(try env.popI64())),

            else => return error.UnknownOpcode,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

const testing = std.testing;

/// Helper: run bytecode directly via dispatchLoop, returning stack top.
fn runCode(code: []const u8) !i32 {
    const alloc = testing.allocator;
    // Minimal module/instance for testing
    var dummy_module = types.WasmModule{};
    var memories = [_]types.MemoryInstance{};
    var globals = [_]types.GlobalInstance{};
    var dummy_inst = types.ModuleInstance{
        .module = &dummy_module,
        .memories = &memories,
        .tables = &.{},
        .globals = &globals,
        .allocator = alloc,
    };
    var env = ExecEnv{
        .module_inst = &dummy_inst,
        .operand_stack = try alloc.alloc(types.Value, 256),
        .call_stack = try alloc.alloc(CallFrame, 64),
        .allocator = alloc,
    };
    defer alloc.free(env.operand_stack);
    defer alloc.free(env.call_stack);

    // Push a dummy frame so locals/currentFrame works
    try env.pushFrame(.{ .func_idx = 0, .ip = 0, .stack_base = 0, .local_count = 0, .return_arity = 1, .prev_sp = 0 });

    try dispatchLoop(&env, code);
    return env.popI32();
}

fn runCodeExpectTrap(code: []const u8, expected: TrapError) !void {
    const result = runCode(code);
    try testing.expectError(expected, result);
}

test "interp: i32.const" {
    // i32.const 42; end
    const result = try runCode(&.{ 0x41, 42, 0x0B });
    try testing.expectEqual(@as(i32, 42), result);
}

test "interp: i32.const negative" {
    // i32.const -1 (LEB128: 0x7F); end
    const result = try runCode(&.{ 0x41, 0x7F, 0x0B });
    try testing.expectEqual(@as(i32, -1), result);
}

test "interp: i32.add" {
    // i32.const 3; i32.const 4; i32.add; end
    const result = try runCode(&.{ 0x41, 3, 0x41, 4, 0x6A, 0x0B });
    try testing.expectEqual(@as(i32, 7), result);
}

test "interp: i32.sub" {
    const result = try runCode(&.{ 0x41, 10, 0x41, 3, 0x6B, 0x0B });
    try testing.expectEqual(@as(i32, 7), result);
}

test "interp: i32.mul" {
    const result = try runCode(&.{ 0x41, 6, 0x41, 7, 0x6C, 0x0B });
    try testing.expectEqual(@as(i32, 42), result);
}

test "interp: i32.div_s" {
    const result = try runCode(&.{ 0x41, 10, 0x41, 3, 0x6D, 0x0B });
    try testing.expectEqual(@as(i32, 3), result);
}

test "interp: i32.div_u by zero traps" {
    try runCodeExpectTrap(&.{ 0x41, 1, 0x41, 0, 0x6E, 0x0B }, error.IntegerDivisionByZero);
}

test "interp: i32.eqz true" {
    const result = try runCode(&.{ 0x41, 0, 0x45, 0x0B });
    try testing.expectEqual(@as(i32, 1), result);
}

test "interp: i32.eqz false" {
    const result = try runCode(&.{ 0x41, 5, 0x45, 0x0B });
    try testing.expectEqual(@as(i32, 0), result);
}

test "interp: unreachable traps" {
    try runCodeExpectTrap(&.{0x00}, error.Unreachable);
}

test "interp: drop" {
    // i32.const 50; i32.const 42; drop; end  → 50
    // Note: 50 = 0x32, bit6 clear so no sign-extend issue
    const result = try runCode(&.{ 0x41, 50, 0x41, 42, 0x1A, 0x0B });
    try testing.expectEqual(@as(i32, 50), result);
}

test "interp: select true" {
    // i32.const 10; i32.const 20; i32.const 1; select; end  → 10
    const result = try runCode(&.{ 0x41, 10, 0x41, 20, 0x41, 1, 0x1B, 0x0B });
    try testing.expectEqual(@as(i32, 10), result);
}

test "interp: select false" {
    // i32.const 10; i32.const 20; i32.const 0; select; end  → 20
    const result = try runCode(&.{ 0x41, 10, 0x41, 20, 0x41, 0, 0x1B, 0x0B });
    try testing.expectEqual(@as(i32, 20), result);
}

test "interp: i32.and/or/xor" {
    // 0xFF & 0x0F = 0x0F
    const and_result = try runCode(&.{ 0x41, 0xFF, 0x00, 0x41, 0x0F, 0x71, 0x0B });
    try testing.expectEqual(@as(i32, 0x0F), and_result);
}

test "interp: nop" {
    const result = try runCode(&.{ 0x41, 42, 0x01, 0x0B });
    try testing.expectEqual(@as(i32, 42), result);
}
