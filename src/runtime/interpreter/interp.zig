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
    var shift: u32 = 0;
    var byte: u8 = 0;
    while (true) {
        if (ip.* >= code.len) return @bitCast(result);
        byte = code[ip.*];
        ip.* += 1;
        result |= @as(u32, byte & 0x7F) << @intCast(shift);
        if (byte & 0x80 == 0) break;
        shift += 7;
        if (shift >= 35) break;
    }
    // Sign-extend from the bit above the last data septet.
    const sign_shift = shift + 7;
    if (sign_shift < 32 and (byte & 0x40) != 0) {
        result |= ~@as(u32, 0) << @intCast(sign_shift);
    }
    return @bitCast(result);
}

fn readI64(code: []const u8, ip: *usize) i64 {
    var result: u64 = 0;
    var shift: u32 = 0;
    var byte: u8 = 0;
    while (true) {
        if (ip.* >= code.len) return @bitCast(result);
        byte = code[ip.*];
        ip.* += 1;
        result |= @as(u64, byte & 0x7F) << @intCast(shift);
        if (byte & 0x80 == 0) break;
        shift += 7;
        if (shift >= 70) break;
    }
    // Sign-extend from the bit above the last data septet.
    const sign_shift = shift + 7;
    if (sign_shift < 64 and (byte & 0x40) != 0) {
        result |= ~@as(u64, 0) << @intCast(sign_shift);
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

// ── Block-target helpers ─────────────────────────────────────────────────

/// Skip past a single LEB128-encoded integer in `code` starting at `pos`.
fn skipLeb128(code: []const u8, pos: usize) usize {
    var p = pos;
    while (p < code.len) {
        const b = code[p];
        p += 1;
        if (b & 0x80 == 0) break;
    }
    return p;
}

/// Read a u32 from `code` at `pos` (static helper for pre-scan).
fn readU32Static(code: []const u8, pos: *usize) u32 {
    var result: u32 = 0;
    var shift: u5 = 0;
    while (true) {
        if (pos.* >= code.len) return result;
        const byte = code[pos.*];
        pos.* += 1;
        result |= @as(u32, byte & 0x7F) << shift;
        if (byte & 0x80 == 0) break;
        if (shift >= 28) break;
        shift +|= 7;
    }
    return result;
}

/// Starting at `start` (which must be right after a block/loop/if opcode
/// and its block-type byte), scan forward to find the position just AFTER
/// the matching `end` opcode.
fn findBlockEnd(code: []const u8, start: usize) usize {
    var depth: u32 = 1;
    var pos = start;
    while (pos < code.len) {
        const b = code[pos];
        pos += 1;
        switch (@as(Opcode, @enumFromInt(b))) {
            .block, .loop, .@"if" => depth += 1,
            .end => {
                depth -= 1;
                if (depth == 0) return pos;
            },
            // Skip LEB128 immediates
            .br, .br_if, .local_get, .local_set, .local_tee,
            .global_get, .global_set, .i32_const, .call,
            => {
                pos = skipLeb128(code, pos);
            },
            .i64_const => {
                pos = skipLeb128(code, pos);
            },
            .i32_load, .i32_store => {
                pos = skipLeb128(code, pos);
                pos = skipLeb128(code, pos);
            },
            .f32_const => pos += 4,
            .f64_const => pos += 8,
            .br_table => {
                var cnt = readU32Static(code, &pos);
                cnt += 1; // +1 for the default label
                var i: u32 = 0;
                while (i < cnt) : (i += 1) pos = skipLeb128(code, pos);
            },
            else => {},
        }
    }
    return pos;
}

/// Starting at `start` (right after the if block-type byte), scan forward
/// to find the `else` opcode at the same nesting depth. Returns the
/// position just AFTER the `else` opcode, or null if there is no else.
fn findElse(code: []const u8, start: usize) ?usize {
    var depth: u32 = 1;
    var pos = start;
    while (pos < code.len) {
        const b = code[pos];
        pos += 1;
        switch (@as(Opcode, @enumFromInt(b))) {
            .block, .loop, .@"if" => depth += 1,
            .end => {
                depth -= 1;
                if (depth == 0) return null; // hit end without else
            },
            .@"else" => {
                if (depth == 1) return pos;
            },
            .br, .br_if, .local_get, .local_set, .local_tee,
            .global_get, .global_set, .i32_const, .call,
            => {
                pos = skipLeb128(code, pos);
            },
            .i64_const => {
                pos = skipLeb128(code, pos);
            },
            .i32_load, .i32_store => {
                pos = skipLeb128(code, pos);
                pos = skipLeb128(code, pos);
            },
            .f32_const => pos += 4,
            .f64_const => pos += 8,
            .br_table => {
                var cnt = readU32Static(code, &pos);
                cnt += 1;
                var i: u32 = 0;
                while (i < cnt) : (i += 1) pos = skipLeb128(code, pos);
            },
            else => {},
        }
    }
    return null;
}

/// Parse a block type byte and return the result arity.
fn blockArity(bt: u8) u32 {
    return switch (bt) {
        0x40 => 0, // void
        0x7F, 0x7E, 0x7D, 0x7C => 1, // i32, i64, f32, f64
        else => 0,
    };
}

// ── Label for structured control flow ────────────────────────────────────

const Label = struct {
    kind: Kind,
    /// IP to branch to (after `end` for block/if, loop header for loop).
    target_ip: usize,
    /// Operand-stack height when this label was entered.
    stack_height: u32,
    /// Expected result arity of this block.
    arity: u32,

    const Kind = enum { block, loop, @"if" };
};

const MAX_LABELS = 256;

// ── Dispatch loop ────────────────────────────────────────────────────────

fn dispatchLoop(env: *ExecEnv, code: []const u8) TrapError!void {
    var ip: usize = 0;
    var labels: [MAX_LABELS]Label = undefined;
    var label_sp: u32 = 0;

    while (ip < code.len) {
        const byte = code[ip];
        ip += 1;
        const op: Opcode = @enumFromInt(byte);

        switch (op) {
            // ── Control ──
            .@"unreachable" => return error.Unreachable,
            .nop => {},

            .block => {
                const bt = code[ip];
                ip += 1;
                const arity = blockArity(bt);
                const end_ip = findBlockEnd(code, ip);
                if (label_sp >= MAX_LABELS) return error.StackOverflow;
                labels[label_sp] = .{
                    .kind = .block,
                    .target_ip = end_ip,
                    .stack_height = env.sp,
                    .arity = arity,
                };
                label_sp += 1;
            },

            .loop => {
                const bt = code[ip];
                ip += 1;
                _ = bt;
                // loop branches go back to the header; loop's br arity is 0
                if (label_sp >= MAX_LABELS) return error.StackOverflow;
                labels[label_sp] = .{
                    .kind = .loop,
                    .target_ip = ip, // right after the block-type byte
                    .stack_height = env.sp,
                    .arity = 0, // br to loop does not yield results
                };
                label_sp += 1;
            },

            .@"if" => {
                const bt = code[ip];
                ip += 1;
                const arity = blockArity(bt);
                const cond = try env.popI32();
                const end_ip = findBlockEnd(code, ip);
                if (cond != 0) {
                    // true branch: push label and execute body
                    if (label_sp >= MAX_LABELS) return error.StackOverflow;
                    labels[label_sp] = .{
                        .kind = .@"if",
                        .target_ip = end_ip,
                        .stack_height = env.sp,
                        .arity = arity,
                    };
                    label_sp += 1;
                } else {
                    // false branch: skip to else or end
                    if (findElse(code, ip)) |else_ip| {
                        if (label_sp >= MAX_LABELS) return error.StackOverflow;
                        labels[label_sp] = .{
                            .kind = .@"if",
                            .target_ip = end_ip,
                            .stack_height = env.sp,
                            .arity = arity,
                        };
                        label_sp += 1;
                        ip = else_ip;
                    } else {
                        // no else clause — skip to end
                        ip = end_ip;
                    }
                }
            },

            .@"else" => {
                // Reached else from the true branch — jump to end
                if (label_sp == 0) return error.StackUnderflow;
                const label = labels[label_sp - 1];
                ip = label.target_ip;
                // Don't pop the label; `end` will pop it
            },

            .end => {
                if (label_sp == 0) return; // function-level end
                label_sp -= 1;
            },

            .@"return" => return,

            .br => {
                const depth = readU32(code, &ip);
                if (depth >= label_sp) return error.StackUnderflow;
                const label = labels[label_sp - 1 - depth];
                // Preserve arity values on top of stack, unwind
                if (label.kind == .loop) {
                    env.sp = label.stack_height;
                } else {
                    var results: [16]types.Value = undefined;
                    var i: u32 = 0;
                    while (i < label.arity) : (i += 1) {
                        results[label.arity - 1 - i] = try env.pop();
                    }
                    env.sp = label.stack_height;
                    i = 0;
                    while (i < label.arity) : (i += 1) {
                        try env.push(results[i]);
                    }
                }
                // Pop labels up to and including the target
                label_sp -= depth + 1;
                if (label.kind == .loop) {
                    // Re-push the loop label and jump to loop header
                    labels[label_sp] = label;
                    label_sp += 1;
                }
                ip = label.target_ip;
            },

            .br_if => {
                const depth = readU32(code, &ip);
                const cond = try env.popI32();
                if (cond != 0) {
                    if (depth >= label_sp) return error.StackUnderflow;
                    const label = labels[label_sp - 1 - depth];
                    if (label.kind == .loop) {
                        env.sp = label.stack_height;
                    } else {
                        var results: [16]types.Value = undefined;
                        var i: u32 = 0;
                        while (i < label.arity) : (i += 1) {
                            results[label.arity - 1 - i] = try env.pop();
                        }
                        env.sp = label.stack_height;
                        i = 0;
                        while (i < label.arity) : (i += 1) {
                            try env.push(results[i]);
                        }
                    }
                    label_sp -= depth + 1;
                    if (label.kind == .loop) {
                        labels[label_sp] = label;
                        label_sp += 1;
                    }
                    ip = label.target_ip;
                }
            },

            .br_table => {
                const count = readU32(code, &ip);
                const save_ip = ip;
                const idx: u32 = @bitCast(try env.popI32());
                // Read all labels (count + 1 for default)
                var target_depth: u32 = 0;
                ip = save_ip;
                if (idx < count) {
                    // Skip to the idx-th label
                    var skip: u32 = 0;
                    while (skip < idx) : (skip += 1) _ = readU32(code, &ip);
                    target_depth = readU32(code, &ip);
                    // Skip remaining labels + default
                    var remain: u32 = idx + 1;
                    while (remain <= count) : (remain += 1) _ = readU32(code, &ip);
                } else {
                    // Use default label (last one)
                    var skip: u32 = 0;
                    while (skip <= count) : (skip += 1) {
                        target_depth = readU32(code, &ip);
                    }
                }
                // Now branch to target_depth
                if (target_depth >= label_sp) return error.StackUnderflow;
                const label = labels[label_sp - 1 - target_depth];
                if (label.kind == .loop) {
                    env.sp = label.stack_height;
                } else {
                    var results: [16]types.Value = undefined;
                    var i: u32 = 0;
                    while (i < label.arity) : (i += 1) {
                        results[label.arity - 1 - i] = try env.pop();
                    }
                    env.sp = label.stack_height;
                    i = 0;
                    while (i < label.arity) : (i += 1) {
                        try env.push(results[i]);
                    }
                }
                label_sp -= target_depth + 1;
                if (label.kind == .loop) {
                    labels[label_sp] = label;
                    label_sp += 1;
                }
                ip = label.target_ip;
            },

            .call => {
                const func_idx = readU32(code, &ip);
                const frame = env.currentFrameMut() orelse return error.CallStackUnderflow;
                frame.ip = @intCast(ip);
                try executeFunction(env, func_idx);
            },

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
    // 0xFF & 0x0F = 0x0F  (255 as signed LEB128 = 0xFF, 0x01)
    const and_result = try runCode(&.{ 0x41, 0xFF, 0x01, 0x41, 0x0F, 0x71, 0x0B });
    try testing.expectEqual(@as(i32, 0x0F), and_result);
}

test "interp: nop" {
    const result = try runCode(&.{ 0x41, 42, 0x01, 0x0B });
    try testing.expectEqual(@as(i32, 42), result);
}

test "interp: i32.div_s traps on overflow" {
    // INT32_MIN / -1 overflows.
    // INT32_MIN as signed LEB128: 0x80, 0x80, 0x80, 0x80, 0x78
    // -1 as signed LEB128: 0x7F
    try runCodeExpectTrap(&.{
        0x41, 0x80, 0x80, 0x80, 0x80, 0x78, // i32.const -2147483648
        0x41, 0x7F, // i32.const -1
        0x6D, // i32.div_s
    }, error.IntegerOverflow);
}

test "interp: local.get and local.set" {
    const alloc = testing.allocator;
    var dummy_module = types.WasmModule{};
    var dummy_inst = types.ModuleInstance{
        .module = &dummy_module,
        .memories = &.{},
        .tables = &.{},
        .globals = &.{},
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

    // Reserve 2 locals on the stack and push a frame.
    try env.pushI32(0); // local 0
    try env.pushI32(0); // local 1
    try env.pushFrame(.{
        .func_idx = 0,
        .ip = 0,
        .stack_base = 0,
        .local_count = 2,
        .return_arity = 0,
        .prev_sp = 2,
    });

    // i32.const 42; local.set 1; local.get 1; end
    try dispatchLoop(&env, &.{
        0x41, 42, // i32.const 42
        0x21, 1, // local.set 1
        0x20, 1, // local.get 1
        0x0B, // end
    });

    _ = try env.popFrame();
    try testing.expectEqual(@as(i32, 42), try env.popI32());
}

test "interp: i32.rem_s" {
    // 10 % 3 = 1
    const result = try runCode(&.{ 0x41, 10, 0x41, 3, 0x6F, 0x0B });
    try testing.expectEqual(@as(i32, 1), result);
}

test "interp: i32.rem_u traps on zero" {
    try runCodeExpectTrap(&.{ 0x41, 10, 0x41, 0, 0x70, 0x0B }, error.IntegerDivisionByZero);
}

test "interp: i32.shl" {
    // 1 << 4 = 16
    const result = try runCode(&.{ 0x41, 1, 0x41, 4, 0x74, 0x0B });
    try testing.expectEqual(@as(i32, 16), result);
}

test "interp: unknown opcode traps" {
    // 0x06 = try opcode (not implemented)
    try runCodeExpectTrap(&.{0x06}, error.UnknownOpcode);
}

test "LEB128: readI32 decodes -129 correctly" {
    // -129 as signed LEB128 = 0xFF, 0x7E
    var ip: usize = 0;
    const code = [_]u8{ 0xFF, 0x7E };
    try testing.expectEqual(@as(i32, -129), readI32(&code, &ip));
    try testing.expectEqual(@as(usize, 2), ip);
}

test "LEB128: readI32 decodes INT32_MIN correctly" {
    // -2147483648 as signed LEB128 = 0x80, 0x80, 0x80, 0x80, 0x78
    var ip: usize = 0;
    const code = [_]u8{ 0x80, 0x80, 0x80, 0x80, 0x78 };
    try testing.expectEqual(@as(i32, -2147483648), readI32(&code, &ip));
}

test "LEB128: readU32 single byte" {
    var ip: usize = 0;
    const code = [_]u8{42};
    try testing.expectEqual(@as(u32, 42), readU32(&code, &ip));
    try testing.expectEqual(@as(usize, 1), ip);
}

test "LEB128: readU32 multi-byte 128" {
    var ip: usize = 0;
    const code = [_]u8{ 0x80, 0x01 };
    try testing.expectEqual(@as(u32, 128), readU32(&code, &ip));
}

test "LEB128: readI64 single byte" {
    var ip: usize = 0;
    const code = [_]u8{0x01};
    try testing.expectEqual(@as(i64, 1), readI64(&code, &ip));
}

test "LEB128: readI64 negative" {
    var ip: usize = 0;
    // -1 as signed LEB128 i64 = 0x7F
    const code = [_]u8{0x7F};
    try testing.expectEqual(@as(i64, -1), readI64(&code, &ip));
}

// ── Control flow tests ──────────────────────────────────────────────────

test "interp: simple block" {
    // (block (result i32) (i32.const 42) end) end
    const result = try runCode(&.{
        0x02, 0x7F, // block (result i32)
        0x41, 42, //   i32.const 42
        0x0B, //   end (block)
        0x0B, //   end (function)
    });
    try testing.expectEqual(@as(i32, 42), result);
}

test "interp: block with br skips remaining code" {
    // (block (result i32) (i32.const 42) (br 0) (i32.const 99) end) end
    const result = try runCode(&.{
        0x02, 0x7F, // block (result i32)
        0x41, 42, //   i32.const 42
        0x0C, 0x00, // br 0
        0x41, 99, //   i32.const 99 (unreachable)
        0x0B, //   end (block)
        0x0B, //   end (function)
    });
    try testing.expectEqual(@as(i32, 42), result);
}

test "interp: loop with br_if counts down" {
    // A loop that counts from 10 down to 0:
    //   (local i32)
    //   local.set 0 = 10
    //   (block (loop
    //     local.get 0
    //     i32.eqz
    //     br_if 1        ;; if counter == 0, break to outer block
    //     local.get 0
    //     i32.const 1
    //     i32.sub
    //     local.set 0
    //     br 0            ;; continue loop
    //   ))
    //   local.get 0
    //   end
    const alloc = testing.allocator;
    var dummy_module = types.WasmModule{};
    var dummy_inst = types.ModuleInstance{
        .module = &dummy_module,
        .memories = &.{},
        .tables = &.{},
        .globals = &.{},
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

    // One local (counter), initialized to 0 on the stack
    try env.pushI32(0);
    try env.pushFrame(.{ .func_idx = 0, .ip = 0, .stack_base = 0, .local_count = 1, .return_arity = 1, .prev_sp = 1 });

    const code = [_]u8{
        0x41, 10, //   i32.const 10
        0x21, 0, //   local.set 0
        0x02, 0x40, // block (void)
        0x03, 0x40, //   loop (void)
        0x20, 0, //     local.get 0
        0x45, //     i32.eqz
        0x0D, 0x01, //   br_if 1  (break to outer block)
        0x20, 0, //     local.get 0
        0x41, 1, //     i32.const 1
        0x6B, //     i32.sub
        0x21, 0, //     local.set 0
        0x0C, 0x00, //   br 0  (continue loop)
        0x0B, //   end (loop)
        0x0B, //   end (block)
        0x20, 0, //   local.get 0
        0x0B, //   end (function)
    };

    try dispatchLoop(&env, &code);
    _ = try env.popFrame();
    const result = try env.popI32();
    try testing.expectEqual(@as(i32, 0), result);
}

test "interp: if true branch" {
    // (if (result i32) (i32.const 1) (then (i32.const 10)) (else (i32.const 20))) end
    const result = try runCode(&.{
        0x41, 1, //   i32.const 1 (condition)
        0x04, 0x7F, // if (result i32)
        0x41, 10, //   i32.const 10
        0x05, //   else
        0x41, 20, //   i32.const 20
        0x0B, //   end (if)
        0x0B, //   end (function)
    });
    try testing.expectEqual(@as(i32, 10), result);
}

test "interp: if false branch" {
    // Same as above but condition is 0 → else branch
    const result = try runCode(&.{
        0x41, 0, //   i32.const 0 (condition)
        0x04, 0x7F, // if (result i32)
        0x41, 10, //   i32.const 10
        0x05, //   else
        0x41, 20, //   i32.const 20
        0x0B, //   end (if)
        0x0B, //   end (function)
    });
    try testing.expectEqual(@as(i32, 20), result);
}

test "interp: if without else (true)" {
    // Push 42, if true -> drop 42, push 55. Result = 55.
    const result = try runCode(&.{
        0x41, 42, //   i32.const 42
        0x41, 1, //   i32.const 1 (condition)
        0x04, 0x40, // if (void)
        0x1A, //     drop (drops 42)
        0x41, 55, //   i32.const 55
        0x0B, //   end (if)
        0x0B, //   end (function)
    });
    try testing.expectEqual(@as(i32, 55), result);
}

test "interp: if without else (false)" {
    // (i32.const 42) (if (i32.const 0) (then (drop) (i32.const 99))) end → 42
    const result = try runCode(&.{
        0x41, 42, //   i32.const 42
        0x41, 0, //   i32.const 0 (condition)
        0x04, 0x40, // if (void)
        0x1A, //     drop
        0x41, 99, //   i32.const 99
        0x0B, //   end (if)
        0x0B, //   end (function)
    });
    try testing.expectEqual(@as(i32, 42), result);
}

test "interp: nested blocks with br 1 skips outer" {
    // (block (result i32)
    //   (block (result i32)
    //     (i32.const 42)
    //     (br 1)            ;; branch to outer block, carrying 42
    //     (i32.const 99)    ;; unreachable
    //   )
    //   (i32.const 77)      ;; unreachable (br 1 skips past outer end)
    // ) end
    const result = try runCode(&.{
        0x02, 0x7F, // block (result i32) — outer
        0x02, 0x7F, //   block (result i32) — inner
        0x41, 42, //     i32.const 42
        0x0C, 0x01, //   br 1 (to outer)
        0x41, 99, //     i32.const 99 (unreachable)
        0x0B, //   end (inner)
        0x41, 77, //   i32.const 77 (unreachable)
        0x0B, //   end (outer)
        0x0B, //   end (function)
    });
    try testing.expectEqual(@as(i32, 42), result);
}

test "interp: call another function" {
    const alloc = testing.allocator;

    // Function 0 (called): returns i32.const 42
    const callee_code = [_]u8{
        0x41, 42, // i32.const 42
        0x0B, //   end
    };
    const callee_type = types.FuncType{ .params = &.{}, .results = &.{.i32} };
    var functions = [_]types.WasmFunction{
        .{
            .type_idx = 0,
            .func_type = callee_type,
            .local_count = 0,
            .locals = &.{},
            .code = &callee_code,
        },
    };
    var func_types = [_]types.FuncType{callee_type};
    var dummy_module = types.WasmModule{
        .functions = &functions,
        .types = &func_types,
    };
    var dummy_inst = types.ModuleInstance{
        .module = &dummy_module,
        .memories = &.{},
        .tables = &.{},
        .globals = &.{},
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

    try env.pushFrame(.{ .func_idx = 0, .ip = 0, .stack_base = 0, .local_count = 0, .return_arity = 1, .prev_sp = 0 });

    // Caller code: call 0; end
    const caller_code = [_]u8{
        0x10, 0x00, // call 0
        0x0B, //       end
    };
    try dispatchLoop(&env, &caller_code);
    _ = try env.popFrame();
    const result = try env.popI32();
    try testing.expectEqual(@as(i32, 42), result);
}

test "interp: br_table dispatch" {
    // Uses void blocks + a local to track which branch was taken.
    // (local $r i32)
    // (block $L2 (block $L1 (block $L0
    //   br_table $L0 $L1 $L2 <idx>
    // ) i32.const 10; local.set 0; br 1)
    //   i32.const 20; local.set 0; br 0)
    // local.get 0; end

    const alloc = testing.allocator;
    var dummy_module = types.WasmModule{};
    var dummy_inst = types.ModuleInstance{
        .module = &dummy_module,
        .memories = &.{},
        .tables = &.{},
        .globals = &.{},
        .allocator = alloc,
    };

    // Helper to run the br_table test with a given index byte
    const T = struct {
        fn run(inst: *types.ModuleInstance, idx_byte: u8) !i32 {
            var env = ExecEnv{
                .module_inst = inst,
                .operand_stack = try testing.allocator.alloc(types.Value, 256),
                .call_stack = try testing.allocator.alloc(CallFrame, 64),
                .allocator = testing.allocator,
            };
            defer testing.allocator.free(env.operand_stack);
            defer testing.allocator.free(env.call_stack);

            try env.pushI32(0); // local 0 (result)
            try env.pushFrame(.{ .func_idx = 0, .ip = 0, .stack_base = 0, .local_count = 1, .return_arity = 1, .prev_sp = 1 });

            const code = [_]u8{
                0x02, 0x40, // block (void) — L2
                0x02, 0x40, //   block (void) — L1
                0x02, 0x40, //     block (void) — L0
                0x41, idx_byte, // i32.const <idx>
                0x0E, //       br_table
                0x02, //         count=2
                0x00, 0x01, 0x02, // labels: 0, 1, default=2
                0x0B, //     end L0
                0x41, 10, //   i32.const 10
                0x21, 0x00, // local.set 0
                0x0C, 0x01, // br 1 (to L2)
                0x0B, //   end L1
                0x41, 20, //   i32.const 20
                0x21, 0x00, // local.set 0
                0x0C, 0x00, // br 0 (to L2)
                0x0B, //   end L2
                0x20, 0x00, // local.get 0
                0x0B, //   end (function)
            };

            try dispatchLoop(&env, &code);
            _ = try env.popFrame();
            return env.popI32();
        }
    };

    // idx=0 → br to L0 → falls through to "i32.const 10; local.set; br" → 10
    try testing.expectEqual(@as(i32, 10), try T.run(&dummy_inst, 0));
    // idx=1 → br to L1 → falls through to "i32.const 20; local.set; br" → 20
    try testing.expectEqual(@as(i32, 20), try T.run(&dummy_inst, 1));
    // idx=2 (default) → br to L2 → local stays 0
    try testing.expectEqual(@as(i32, 0), try T.run(&dummy_inst, 2));
    // idx=50 (out of range → default) → br to L2 → local stays 0
    try testing.expectEqual(@as(i32, 0), try T.run(&dummy_inst, 50));
}
