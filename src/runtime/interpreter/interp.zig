//! WebAssembly bytecode interpreter.
//!
//! Executes Wasm functions using a switch-based dispatch loop.
//! Supports MVP i32/i64/f32/f64 arithmetic, comparisons, conversions,
//! locals, globals, memory load/store, and parametric instructions.

const std = @import("std");
const types = @import("../common/types.zig");
const ExecEnv = @import("../common/exec_env.zig").ExecEnv;
const CallFrame = @import("../common/exec_env.zig").CallFrame;

// NaN canonicalization per Wasm spec
inline fn canonF32(val: f32) f32 {
    return if (std.math.isNan(val)) @as(f32, @bitCast(@as(u32, 0x7FC00000))) else val;
}
inline fn canonF64(val: f64) f64 {
    return if (std.math.isNan(val)) @as(f64, @bitCast(@as(u64, 0x7FF8000000000000))) else val;
}

// Wasm-spec-compliant min/max that handle NaN propagation and signed zeros.
// Wasm requires: min(-0, +0) = -0, max(-0, +0) = +0.
inline fn wasmMinF32(a: f32, b: f32) f32 {
    if (std.math.isNan(a) or std.math.isNan(b)) return canonF32(std.math.nan(f32));
    if (a == b) return @bitCast(@as(u32, @bitCast(a)) | @as(u32, @bitCast(b)));
    return @min(a, b);
}
inline fn wasmMaxF32(a: f32, b: f32) f32 {
    if (std.math.isNan(a) or std.math.isNan(b)) return canonF32(std.math.nan(f32));
    if (a == b) return @bitCast(@as(u32, @bitCast(a)) & @as(u32, @bitCast(b)));
    return @max(a, b);
}
inline fn wasmMinF64(a: f64, b: f64) f64 {
    if (std.math.isNan(a) or std.math.isNan(b)) return canonF64(std.math.nan(f64));
    if (a == b) return @bitCast(@as(u64, @bitCast(a)) | @as(u64, @bitCast(b)));
    return @min(a, b);
}
inline fn wasmMaxF64(a: f64, b: f64) f64 {
    if (std.math.isNan(a) or std.math.isNan(b)) return canonF64(std.math.nan(f64));
    if (a == b) return @bitCast(@as(u64, @bitCast(a)) & @as(u64, @bitCast(b)));
    return @max(a, b);
}

// Wasm nearest: round to nearest integer, ties to even.
// Zig's @round uses ties-away-from-zero, so we use the add-subtract trick
// which relies on hardware FP default rounding mode (ties-to-even).
inline fn wasmNearestF32(x: f32) f32 {
    if (std.math.isNan(x)) return canonF32(x);
    const mag = @abs(x);
    if (mag == 0.0 or mag >= 0x1.0p23) return x;
    const magic: f32 = 0x1.0p23;
    const result = (mag + magic) - magic;
    return std.math.copysign(result, x);
}
inline fn wasmNearestF64(x: f64) f64 {
    if (std.math.isNan(x)) return canonF64(x);
    const mag = @abs(x);
    if (mag == 0.0 or mag >= 0x1.0p52) return x;
    const magic: f64 = 0x1.0p52;
    const result = (mag + magic) - magic;
    return std.math.copysign(result, x);
}

const Opcode = @import("opcode.zig").Opcode;
const simd = @import("simd.zig");

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
    OutOfBoundsTableAccess,
    UninitializedElement,
    IndirectCallTypeMismatch,
    UnknownFunction,
    UnknownOpcode,
    UnalignedAtomicAccess,
};

/// Result from dispatchLoop indicating how the function exited.
const DispatchResult = enum(u32) {
    /// Normal return (.end at function level or .return).
    normal = 0,
    /// Tail call — the target func_idx is stored separately.
    tail_call = 1,
};

// ── LEB128 helpers for bytecode ──────────────────────────────────────────

const Memarg = struct { mem_idx: u32, offset: u32 };

/// Read a memarg from bytecode: alignment (with multi-memory bit 6) + offset.
/// Returns the memory index and offset.
fn readMemarg(code: []const u8, ip: *usize) Memarg {
    const align_flags = readU32(code, ip);
    const mem_idx: u32 = if (align_flags & 0x40 != 0) readU32(code, ip) else 0;
    const offset = readU32(code, ip);
    return .{ .mem_idx = mem_idx, .offset = offset };
}

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

/// Prepare a tail call: move new args to the current frame's stack base
/// and pop the frame, leaving the stack ready for the next function.
/// Returns the target func_idx for the caller to loop on.
fn prepareTailCall(env: *ExecEnv, func_idx: u32) TrapError!void {
    const module = env.module_inst.module;
    const func_type = module.getFuncType(func_idx) orelse return error.UnknownFunction;
    const new_param_count: u32 = @intCast(func_type.params.len);

    const old_frame = env.currentFrame() orelse return error.CallStackUnderflow;
    const old_stack_base = old_frame.stack_base;

    // Bounds check before stack manipulation
    if (env.sp < new_param_count) return error.StackUnderflow;
    if (old_stack_base + new_param_count > env.operand_stack.len) return error.StackOverflow;

    // Move the new function's params from the top of the stack down
    // to the current frame's stack_base, then reset sp.
    const src_start = env.sp - new_param_count;
    for (0..new_param_count) |i| {
        env.operand_stack[old_stack_base + @as(u32, @intCast(i))] =
            env.operand_stack[src_start + @as(u32, @intCast(i))];
    }
    env.sp = old_stack_base + new_param_count;

    _ = try env.popFrame();
}

/// Execute a function by index within the module instance.
pub fn executeFunction(env: *ExecEnv, func_idx: u32) TrapError!void {
    var current_func_idx = func_idx;

    while (true) {
        const module = env.module_inst.module;
        if (current_func_idx < module.import_function_count) {
            // Dispatch to the imported module's actual function
            if (current_func_idx < env.module_inst.import_functions.len) {
                const imported = env.module_inst.import_functions[current_func_idx];
                const saved_module_inst = env.module_inst;
                env.module_inst = imported.module_inst;
                defer env.module_inst = saved_module_inst;
                try executeFunction(env, imported.func_idx);
                return;
            }
            // Fallback: no-op stub for unresolved imports (pop args, push zero results)
            const func_type = module.getFuncType(current_func_idx) orelse return error.UnknownFunction;
            var i: usize = func_type.params.len;
            while (i > 0) : (i -= 1) _ = try env.pop();
            for (func_type.results) |result_type| {
                try env.push(switch (result_type) {
                    .i32 => .{ .i32 = 0 },
                    .i64 => .{ .i64 = 0 },
                    .f32 => .{ .f32 = 0.0 },
                    .f64 => .{ .f64 = 0.0 },
                    .funcref => .{ .funcref = null },
                    .externref => .{ .externref = null },
                    .nonfuncref => .{ .nonfuncref = null },
                    .nonexternref => .{ .nonexternref = null },
                    .v128 => .{ .v128 = 0 },
                });
            }
            return;
        }
        const local_idx = std.math.sub(u32, current_func_idx, module.import_function_count) catch return error.UnknownFunction;

        if (local_idx >= module.functions.len) return error.UnknownFunction;
        const func = &module.functions[local_idx];
        if (func.type_idx >= module.types.len) return error.UnknownFunction;
        const func_type = module.types[func.type_idx];
        const param_count: u32 = @intCast(func_type.params.len);

        var total_locals: u32 = param_count;
        for (func.locals) |local| total_locals += local.count;

        const stack_base = env.sp - param_count;
        try env.pushFrame(.{
            .func_idx = current_func_idx,
            .ip = 0,
            .stack_base = stack_base,
            .local_count = total_locals,
            .return_arity = @intCast(func_type.results.len),
            .prev_sp = env.sp,
        });

        // Initialize non-param locals to zero with correct types.
        for (func.locals) |local| {
            var j: u32 = 0;
            while (j < local.count) : (j += 1) {
                try env.push(switch (local.val_type) {
                    .i32 => .{ .i32 = 0 },
                    .i64 => .{ .i64 = 0 },
                    .f32 => .{ .f32 = 0.0 },
                    .f64 => .{ .f64 = 0.0 },
                    .funcref => .{ .funcref = null },
                    .externref => .{ .externref = null },
                    .nonfuncref => .{ .nonfuncref = null },
                    .nonexternref => .{ .nonexternref = null },
                    .v128 => .{ .v128 = 0 },
                });
            }
        }

        var tail_call_target: u32 = 0;
        const result = try dispatchLoop(env, func.code, &tail_call_target);

        switch (result) {
            .normal => {
                // Normal return: save results, pop frame, restore sp, push results.
                const return_count: u32 = @intCast(func_type.results.len);
                var return_vals: [16]types.Value = undefined;
                {
                    var ri: u32 = return_count;
                    while (ri > 0) {
                        ri -= 1;
                        return_vals[ri] = try env.pop();
                    }
                }
                const frame = try env.popFrame();
                env.sp = frame.stack_base;
                {
                    var ri: u32 = 0;
                    while (ri < return_count) : (ri += 1) {
                        try env.push(return_vals[ri]);
                    }
                }
                return;
            },
            .tail_call => {
                // prepareTailCall already moved args and popped the frame.
                current_func_idx = tail_call_target;
                continue;
            },
        }
    }
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

/// Skip a block type immediate in bytecode: handles 0x40 (void), single-byte val types,
/// typed ref prefixes (0x63/0x64 + heaptype), and type index (LEB128).
fn skipBlockTypeBytes(code: []const u8, pos: usize) usize {
    if (pos >= code.len) return pos;
    const bt = code[pos];
    if (bt == 0x63 or bt == 0x64) {
        // Typed ref: prefix + heap type
        var p = pos + 1;
        if (p >= code.len) return p;
        const ht = code[p];
        if (ht == 0x70 or ht == 0x6F or ht == 0x73 or ht == 0x72) return p + 1;
        // Concrete type index LEB128
        while (p < code.len) {
            const b = code[p];
            p += 1;
            if (b & 0x80 == 0) break;
        }
        return p;
    }
    return skipLeb128(code, pos);
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
            .block, .loop, .@"if" => {
                depth += 1;
                pos = skipBlockTypeBytes(code, pos);
            },
            .end => {
                depth -= 1;
                if (depth == 0) return pos;
            },
            // Skip LEB128 immediates
            .br, .br_if, .local_get, .local_set, .local_tee,
            .global_get, .global_set, .i32_const, .call, .return_call,
            .ref_null, .ref_func,
            .table_get, .table_set,
            .call_ref, .return_call_ref, .br_on_null, .br_on_non_null,
            => {
                pos = skipLeb128(code, pos);
            },
            .i64_const => {
                pos = skipLeb128(code, pos);
            },
            .call_indirect, .return_call_indirect => {
                pos = skipLeb128(code, pos); // type_idx
                pos = skipLeb128(code, pos); // table_idx
            },
            .select_t => {
                const cnt = readU32Static(code, &pos);
                var j: u32 = 0;
                while (j < cnt) : (j += 1) pos = skipLeb128(code, pos);
            },
            .i32_load, .i64_load, .f32_load, .f64_load,
            .i32_load8_s, .i32_load8_u, .i32_load16_s, .i32_load16_u,
            .i64_load8_s, .i64_load8_u, .i64_load16_s, .i64_load16_u,
            .i64_load32_s, .i64_load32_u,
            .i32_store, .i64_store, .f32_store, .f64_store,
            .i32_store8, .i32_store16,
            .i64_store8, .i64_store16, .i64_store32,
            => {
                pos = skipLeb128(code, pos);
                pos = skipLeb128(code, pos);
            },
            .memory_size, .memory_grow => {
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
            .misc_prefix => {
                const sub_op = readU32Static(code, &pos);
                switch (sub_op) {
                    0...7 => {},
                    9, 11, 13, 15, 16, 17 => pos = skipLeb128(code, pos),
                    8, 10, 12, 14 => {
                        pos = skipLeb128(code, pos);
                        pos = skipLeb128(code, pos);
                    },
                    else => {},
                }
            },
            .simd_prefix => {
                const sub_op = readU32Static(code, &pos);
                switch (sub_op) {
                    0x00...0x0B => { // v128.load/store: memarg
                        pos = skipLeb128(code, pos); // align
                        pos = skipLeb128(code, pos); // offset
                    },
                    0x0C => pos += 16, // v128.const: 16 bytes
                    0x0D => pos += 16, // i8x16.shuffle: 16 lane bytes
                    0x15...0x22 => pos += 1, // lane extract/replace: 1 byte
                    0x54...0x5B => { // load/store lane: memarg + lane byte
                        pos = skipLeb128(code, pos);
                        pos = skipLeb128(code, pos);
                        pos += 1;
                    },
                    0x5C, 0x5D => { // load_zero: memarg
                        pos = skipLeb128(code, pos);
                        pos = skipLeb128(code, pos);
                    },
                    else => {}, // no immediates
                }
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
            .block, .loop, .@"if" => {
                depth += 1;
                pos = skipBlockTypeBytes(code, pos);
            },
            .end => {
                depth -= 1;
                if (depth == 0) return null; // hit end without else
            },
            .@"else" => {
                if (depth == 1) return pos;
            },
            .br, .br_if, .local_get, .local_set, .local_tee,
            .global_get, .global_set, .i32_const, .call, .return_call,
            .ref_null, .ref_func,
            .table_get, .table_set,
            .call_ref, .return_call_ref, .br_on_null, .br_on_non_null,
            => {
                pos = skipLeb128(code, pos);
            },
            .i64_const => {
                pos = skipLeb128(code, pos);
            },
            .call_indirect, .return_call_indirect => {
                pos = skipLeb128(code, pos);
                pos = skipLeb128(code, pos);
            },
            .select_t => {
                const cnt = readU32Static(code, &pos);
                var j: u32 = 0;
                while (j < cnt) : (j += 1) pos = skipLeb128(code, pos);
            },
            .i32_load, .i64_load, .f32_load, .f64_load,
            .i32_load8_s, .i32_load8_u, .i32_load16_s, .i32_load16_u,
            .i64_load8_s, .i64_load8_u, .i64_load16_s, .i64_load16_u,
            .i64_load32_s, .i64_load32_u,
            .i32_store, .i64_store, .f32_store, .f64_store,
            .i32_store8, .i32_store16,
            .i64_store8, .i64_store16, .i64_store32,
            => {
                pos = skipLeb128(code, pos);
                pos = skipLeb128(code, pos);
            },
            .memory_size, .memory_grow => {
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
            .misc_prefix => {
                const sub_op = readU32Static(code, &pos);
                switch (sub_op) {
                    0...7 => {},
                    9, 11, 13, 15, 16, 17 => pos = skipLeb128(code, pos),
                    8, 10, 12, 14 => {
                        pos = skipLeb128(code, pos);
                        pos = skipLeb128(code, pos);
                    },
                    else => {},
                }
            },
            .simd_prefix => {
                const sub_op = readU32Static(code, &pos);
                switch (sub_op) {
                    0x00...0x0B => {
                        pos = skipLeb128(code, pos);
                        pos = skipLeb128(code, pos);
                    },
                    0x0C => pos += 16,
                    0x0D => pos += 16,
                    0x15...0x22 => pos += 1,
                    0x54...0x5B => {
                        pos = skipLeb128(code, pos);
                        pos = skipLeb128(code, pos);
                        pos += 1;
                    },
                    0x5C, 0x5D => {
                        pos = skipLeb128(code, pos);
                        pos = skipLeb128(code, pos);
                    },
                    else => {},
                }
            },
            else => {},
        }
    }
    return null;
}

/// Parse a block type from bytecode, advancing ip past it.
/// Returns result arity and param arity (for multi-value type indices).
const BlockTypeInfo = struct { result_arity: u32, param_arity: u32 };

fn readBlockTypeInfo(code: []const u8, ip: *usize, module_types: []const types.FuncType) BlockTypeInfo {
    const first = code[ip.*];
    // Typed reference block types: 0x63 (ref null) / 0x64 (ref) + heap type
    if (first == 0x63 or first == 0x64) {
        ip.* += 1; // consume ref prefix
        if (ip.* < code.len) {
            const ht = code[ip.*];
            if (ht == 0x70 or ht == 0x6F or ht == 0x73 or ht == 0x72) {
                ip.* += 1; // abstract heap type (1 byte)
            } else {
                // Concrete type index (LEB128)
                while (ip.* < code.len) {
                    const b = code[ip.*];
                    ip.* += 1;
                    if (b & 0x80 == 0) break;
                }
            }
        }
        return .{ .result_arity = 1, .param_arity = 0 };
    }
    // Inline types: bytes 0x40-0x7F are single-byte negative s33 values.
    if (first >= 0x40 and first & 0x80 == 0) {
        ip.* += 1;
        return .{
            .result_arity = if (first == 0x40) @as(u32, 0) else @as(u32, 1),
            .param_arity = 0,
        };
    }
    // Type index (unsigned LEB128)
    const type_idx = readU32(code, ip);
    if (type_idx < module_types.len) {
        const ft = module_types[type_idx];
        return .{
            .result_arity = @intCast(ft.results.len),
            .param_arity = @intCast(ft.params.len),
        };
    }
    return .{ .result_arity = 0, .param_arity = 0 };
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

// ── Helpers ──────────────────────────────────────────────────────────────

fn funcTypesEqual(a: types.FuncType, b: types.FuncType) bool {
    if (a.params.len != b.params.len or a.results.len != b.results.len) return false;
    for (a.params, b.params) |ap, bp| if (ap.toNullable() != bp.toNullable()) return false;
    for (a.results, b.results) |ar, br| if (ar.toNullable() != br.toNullable()) return false;
    return true;
}

// ── Dispatch loop ────────────────────────────────────────────────────────

fn dispatchLoop(env: *ExecEnv, code: []const u8, tail_call_target: *u32) TrapError!DispatchResult {
    var ip: usize = 0;
    var labels: [MAX_LABELS]Label = undefined;
    var label_sp: u32 = 0;
    var fuel: u32 = 100_000_000;

    // Push implicit function-body label so br/br_if/br_table can target the function.
    const return_arity: u32 = if (env.currentFrame()) |frame| frame.return_arity else 0;
    labels[0] = .{
        .kind = .block,
        .target_ip = code.len,
        .stack_height = env.sp,
        .arity = return_arity,
    };
    label_sp = 1;

    while (ip < code.len) {
        fuel -|= 1;
        if (fuel == 0) return error.Unreachable;
        // Check for cross-thread trap (every 4096 iterations to minimize overhead)
        if (fuel % 4096 == 0) {
            if (env.thread_manager) |tm| {
                if (tm.hasTrap()) return error.Unreachable;
            }
        }
        const byte = code[ip];
        ip += 1;
        const op: Opcode = @enumFromInt(byte);

        switch (op) {
            // ── Control ──
            .@"unreachable" => return error.Unreachable,
            .nop => {},

            .block => {
                const module_types = env.module_inst.module.types;
                const bt_info = readBlockTypeInfo(code, &ip, module_types);
                const end_ip = findBlockEnd(code, ip);
                if (label_sp >= MAX_LABELS) return error.StackOverflow;
                labels[label_sp] = .{
                    .kind = .block,
                    .target_ip = end_ip,
                    .stack_height = env.sp - bt_info.param_arity,
                    .arity = bt_info.result_arity,
                };
                label_sp += 1;
            },

            .loop => {
                const module_types = env.module_inst.module.types;
                const bt_info = readBlockTypeInfo(code, &ip, module_types);
                if (label_sp >= MAX_LABELS) return error.StackOverflow;
                labels[label_sp] = .{
                    .kind = .loop,
                    .target_ip = ip, // right after the block-type byte(s)
                    .stack_height = env.sp - bt_info.param_arity,
                    .arity = bt_info.param_arity, // br to loop carries params
                };
                label_sp += 1;
            },

            .@"if" => {
                const module_types = env.module_inst.module.types;
                const bt_info = readBlockTypeInfo(code, &ip, module_types);
                const cond = try env.popI32();
                const end_ip = findBlockEnd(code, ip);
                if (cond != 0) {
                    // true branch: push label and execute body
                    if (label_sp >= MAX_LABELS) return error.StackOverflow;
                    labels[label_sp] = .{
                        .kind = .@"if",
                        .target_ip = end_ip,
                        .stack_height = env.sp - bt_info.param_arity,
                        .arity = bt_info.result_arity,
                    };
                    label_sp += 1;
                } else {
                    // false branch: skip to else or end
                    if (findElse(code, ip)) |else_ip| {
                        if (label_sp >= MAX_LABELS) return error.StackOverflow;
                        labels[label_sp] = .{
                            .kind = .@"if",
                            .target_ip = end_ip,
                            .stack_height = env.sp - bt_info.param_arity,
                            .arity = bt_info.result_arity,
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
                // Reached else from the true branch — skip to after end
                if (label_sp <= 1) return error.StackUnderflow;
                const label = labels[label_sp - 1];
                // Pop the label (since we jump past the end opcode, it won't pop it)
                label_sp -= 1;
                ip = label.target_ip;
            },

            .end => {
                if (label_sp <= 1) return .normal; // function-level end
                label_sp -= 1;
            },

            .@"return" => return .normal,

            .br => {
                const depth = readU32(code, &ip);
                if (depth >= label_sp) return error.StackUnderflow;
                const label = labels[label_sp - 1 - depth];
                // Preserve arity values on top of stack, unwind
                {
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
                    {
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
                {
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

            .call_indirect => {
                const type_idx = readU32(code, &ip);
                const table_idx = readU32(code, &ip);
                const elem_idx: u32 = @bitCast(try env.popI32());

                const table = if (table_idx < env.module_inst.tables.len) env.module_inst.tables[table_idx] else return error.OutOfBoundsTableAccess;
                if (elem_idx >= table.elements.len) return error.OutOfBoundsTableAccess;
                const funcref = table.elements[elem_idx] orelse return error.UninitializedElement;

                const module = env.module_inst.module;
                if (type_idx >= module.types.len) return error.IndirectCallTypeMismatch;
                const expected_type = module.types[type_idx];

                // Resolve function type from the funcref's source module
                const actual_type = funcref.module_inst.module.getFuncType(funcref.func_idx) orelse return error.UnknownFunction;
                if (!funcTypesEqual(expected_type, actual_type)) return error.IndirectCallTypeMismatch;

                const frame = env.currentFrameMut() orelse return error.CallStackUnderflow;
                frame.ip = @intCast(ip);

                // Dispatch to the funcref's source module
                if (funcref.module_inst != env.module_inst) {
                    const saved = env.module_inst;
                    env.module_inst = funcref.module_inst;
                    defer env.module_inst = saved;
                    try executeFunction(env, funcref.func_idx);
                } else {
                    try executeFunction(env, funcref.func_idx);
                }
            },

            .return_call => {
                const func_idx = readU32(code, &ip);
                try prepareTailCall(env, func_idx);
                tail_call_target.* = func_idx;
                return .tail_call;
            },

            .return_call_indirect => {
                const type_idx = readU32(code, &ip);
                const table_idx = readU32(code, &ip);
                const elem_idx: u32 = @bitCast(try env.popI32());

                const table = if (table_idx < env.module_inst.tables.len) env.module_inst.tables[table_idx] else return error.OutOfBoundsTableAccess;
                if (elem_idx >= table.elements.len) return error.OutOfBoundsTableAccess;
                const funcref = table.elements[elem_idx] orelse return error.UninitializedElement;

                const module = env.module_inst.module;
                if (type_idx >= module.types.len) return error.IndirectCallTypeMismatch;
                const expected_type = module.types[type_idx];
                const actual_type = funcref.module_inst.module.getFuncType(funcref.func_idx) orelse return error.UnknownFunction;
                if (!funcTypesEqual(expected_type, actual_type)) return error.IndirectCallTypeMismatch;

                if (funcref.module_inst != env.module_inst) {
                    env.module_inst = funcref.module_inst;
                }
                try prepareTailCall(env, funcref.func_idx);
                tail_call_target.* = funcref.func_idx;
                return .tail_call;
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
                try env.push(try env.getLocal(frame, idx));
            },
            .local_set => {
                const idx = readU32(code, &ip);
                const frame = env.currentFrame() orelse return error.CallStackUnderflow;
                try env.setLocal(frame, idx, try env.pop());
            },
            .local_tee => {
                const idx = readU32(code, &ip);
                const frame = env.currentFrame() orelse return error.CallStackUnderflow;
                try env.setLocal(frame, idx, try env.peek());
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
            .select_t => {
                const num_types = readU32(code, &ip);
                var j: u32 = 0;
                while (j < num_types) : (j += 1) _ = readU32(code, &ip);
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

            // ── Memory ──

            // i32 loads
            .i32_load => {
                const ma = readMemarg(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 4 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                try env.pushI32(std.mem.readInt(i32, mem.data[a..][0..4], .little));
            },
            .i32_load8_s => {
                const ma = readMemarg(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 1 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const signed_byte: i8 = @bitCast(mem.data[@intCast(addr)]);
                try env.pushI32(@as(i32, signed_byte));
            },
            .i32_load8_u => {
                const ma = readMemarg(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 1 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                try env.pushI32(@as(i32, @intCast(mem.data[@intCast(addr)])));
            },
            .i32_load16_s => {
                const ma = readMemarg(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 2 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                const val: i16 = std.mem.readInt(i16, mem.data[a..][0..2], .little);
                try env.pushI32(@as(i32, val));
            },
            .i32_load16_u => {
                const ma = readMemarg(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 2 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                const val: u16 = std.mem.readInt(u16, mem.data[a..][0..2], .little);
                try env.pushI32(@as(i32, @intCast(val)));
            },

            // i32 stores
            .i32_store => {
                const ma = readMemarg(code, &ip);
                const val = try env.popI32();
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 4 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                std.mem.writeInt(i32, mem.data[a..][0..4], val, .little);
            },
            .i32_store8 => {
                const ma = readMemarg(code, &ip);
                const val = try env.popI32();
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 1 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                mem.data[@intCast(addr)] = @truncate(@as(u32, @bitCast(val)));
            },
            .i32_store16 => {
                const ma = readMemarg(code, &ip);
                const val = try env.popI32();
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 2 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                std.mem.writeInt(u16, mem.data[a..][0..2], @truncate(@as(u32, @bitCast(val))), .little);
            },

            // i64 loads
            .i64_load => {
                const ma = readMemarg(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 8 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                try env.pushI64(std.mem.readInt(i64, mem.data[a..][0..8], .little));
            },
            .i64_load8_s => {
                const ma = readMemarg(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 1 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const signed_byte: i8 = @bitCast(mem.data[@intCast(addr)]);
                try env.pushI64(@as(i64, signed_byte));
            },
            .i64_load8_u => {
                const ma = readMemarg(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 1 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                try env.pushI64(@as(i64, @intCast(mem.data[@intCast(addr)])));
            },
            .i64_load16_s => {
                const ma = readMemarg(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 2 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                const val: i16 = std.mem.readInt(i16, mem.data[a..][0..2], .little);
                try env.pushI64(@as(i64, val));
            },
            .i64_load16_u => {
                const ma = readMemarg(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 2 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                const val: u16 = std.mem.readInt(u16, mem.data[a..][0..2], .little);
                try env.pushI64(@as(i64, @intCast(val)));
            },
            .i64_load32_s => {
                const ma = readMemarg(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 4 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                const val: i32 = std.mem.readInt(i32, mem.data[a..][0..4], .little);
                try env.pushI64(@as(i64, val));
            },
            .i64_load32_u => {
                const ma = readMemarg(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 4 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                const val: u32 = std.mem.readInt(u32, mem.data[a..][0..4], .little);
                try env.pushI64(@as(i64, @intCast(val)));
            },

            // i64 stores
            .i64_store => {
                const ma = readMemarg(code, &ip);
                const val = try env.popI64();
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 8 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                std.mem.writeInt(i64, mem.data[a..][0..8], val, .little);
            },
            .i64_store8 => {
                const ma = readMemarg(code, &ip);
                const val = try env.popI64();
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 1 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                mem.data[@intCast(addr)] = @truncate(@as(u64, @bitCast(val)));
            },
            .i64_store16 => {
                const ma = readMemarg(code, &ip);
                const val = try env.popI64();
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 2 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                std.mem.writeInt(u16, mem.data[a..][0..2], @truncate(@as(u64, @bitCast(val))), .little);
            },
            .i64_store32 => {
                const ma = readMemarg(code, &ip);
                const val = try env.popI64();
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 4 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                std.mem.writeInt(u32, mem.data[a..][0..4], @truncate(@as(u64, @bitCast(val))), .little);
            },

            // f32 load/store
            .f32_load => {
                const ma = readMemarg(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 4 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                try env.pushF32(@bitCast(std.mem.readInt(u32, mem.data[a..][0..4], .little)));
            },
            .f32_store => {
                const ma = readMemarg(code, &ip);
                const val = try env.popF32();
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 4 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                std.mem.writeInt(u32, mem.data[a..][0..4], @bitCast(val), .little);
            },

            // f64 load/store
            .f64_load => {
                const ma = readMemarg(code, &ip);
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 8 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                try env.pushF64(@bitCast(std.mem.readInt(u64, mem.data[a..][0..8], .little)));
            },
            .f64_store => {
                const ma = readMemarg(code, &ip);
                const val = try env.popF64();
                const base: u32 = @bitCast(try env.popI32());
                const addr = @as(u64, base) + ma.offset;
                const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                if (addr + 8 > mem.data.len) return error.OutOfBoundsMemoryAccess;
                const a: usize = @intCast(addr);
                std.mem.writeInt(u64, mem.data[a..][0..8], @bitCast(val), .little);
            },

            // memory.size / memory.grow
            .memory_size => {
                const mem_idx = readU32(code, &ip);
                const mem = env.module_inst.getMemory(mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                try env.pushI32(@intCast(mem.current_pages));
            },
            .memory_grow => {
                const mem_idx = readU32(code, &ip);
                const delta: u32 = @bitCast(try env.popI32());
                const mem = env.module_inst.getMemory(mem_idx) orelse {
                    try env.pushI32(-1);
                    continue;
                };
                const old_pages = mem.grow(delta, env.module_inst.allocator) catch {
                    try env.pushI32(-1);
                    continue;
                };
                try env.pushI32(@bitCast(old_pages));
            },

            // ── i64 comparison ──
            .i64_eqz => try env.pushI32(@intFromBool(try env.popI64() == 0)),
            .i64_eq => {
                const b = try env.popI64();
                try env.pushI32(@intFromBool(try env.popI64() == b));
            },
            .i64_ne => {
                const b = try env.popI64();
                try env.pushI32(@intFromBool(try env.popI64() != b));
            },
            .i64_lt_s => {
                const b = try env.popI64();
                try env.pushI32(@intFromBool(try env.popI64() < b));
            },
            .i64_lt_u => {
                const b: u64 = @bitCast(try env.popI64());
                const a: u64 = @bitCast(try env.popI64());
                try env.pushI32(@intFromBool(a < b));
            },
            .i64_gt_s => {
                const b = try env.popI64();
                try env.pushI32(@intFromBool(try env.popI64() > b));
            },
            .i64_gt_u => {
                const b: u64 = @bitCast(try env.popI64());
                const a: u64 = @bitCast(try env.popI64());
                try env.pushI32(@intFromBool(a > b));
            },
            .i64_le_s => {
                const b = try env.popI64();
                try env.pushI32(@intFromBool(try env.popI64() <= b));
            },
            .i64_le_u => {
                const b: u64 = @bitCast(try env.popI64());
                const a: u64 = @bitCast(try env.popI64());
                try env.pushI32(@intFromBool(a <= b));
            },
            .i64_ge_s => {
                const b = try env.popI64();
                try env.pushI32(@intFromBool(try env.popI64() >= b));
            },
            .i64_ge_u => {
                const b: u64 = @bitCast(try env.popI64());
                const a: u64 = @bitCast(try env.popI64());
                try env.pushI32(@intFromBool(a >= b));
            },

            // ── i64 unary ──
            .i64_clz => {
                const a: u64 = @bitCast(try env.popI64());
                try env.pushI64(@bitCast(@as(u64, @clz(a))));
            },
            .i64_ctz => {
                const a: u64 = @bitCast(try env.popI64());
                try env.pushI64(@bitCast(@as(u64, @ctz(a))));
            },
            .i64_popcnt => {
                const a: u64 = @bitCast(try env.popI64());
                try env.pushI64(@bitCast(@as(u64, @popCount(a))));
            },

            // ── i64 arithmetic ──
            .i64_add => { const b = try env.popI64(); try env.pushI64((try env.popI64()) +% b); },
            .i64_sub => { const b = try env.popI64(); try env.pushI64((try env.popI64()) -% b); },
            .i64_mul => { const b = try env.popI64(); try env.pushI64((try env.popI64()) *% b); },
            .i64_div_s => {
                const b = try env.popI64();
                const a = try env.popI64();
                if (b == 0) return error.IntegerDivisionByZero;
                if (a == std.math.minInt(i64) and b == -1) return error.IntegerOverflow;
                try env.pushI64(@divTrunc(a, b));
            },
            .i64_div_u => {
                const b: u64 = @bitCast(try env.popI64());
                const a: u64 = @bitCast(try env.popI64());
                if (b == 0) return error.IntegerDivisionByZero;
                try env.pushI64(@bitCast(a / b));
            },
            .i64_rem_s => {
                const b = try env.popI64();
                const a = try env.popI64();
                if (b == 0) return error.IntegerDivisionByZero;
                try env.pushI64(if (a == std.math.minInt(i64) and b == -1) 0 else @rem(a, b));
            },
            .i64_rem_u => {
                const b: u64 = @bitCast(try env.popI64());
                const a: u64 = @bitCast(try env.popI64());
                if (b == 0) return error.IntegerDivisionByZero;
                try env.pushI64(@bitCast(a % b));
            },
            .i64_and => { const b = try env.popI64(); try env.pushI64((try env.popI64()) & b); },
            .i64_or => { const b = try env.popI64(); try env.pushI64((try env.popI64()) | b); },
            .i64_xor => { const b = try env.popI64(); try env.pushI64((try env.popI64()) ^ b); },
            .i64_shl => {
                const b: u64 = @bitCast(try env.popI64());
                const a: u64 = @bitCast(try env.popI64());
                try env.pushI64(@bitCast(a << @intCast(b % 64)));
            },
            .i64_shr_s => {
                const b: u64 = @bitCast(try env.popI64());
                const a = try env.popI64();
                try env.pushI64(a >> @intCast(b % 64));
            },
            .i64_shr_u => {
                const b: u64 = @bitCast(try env.popI64());
                const a: u64 = @bitCast(try env.popI64());
                try env.pushI64(@bitCast(a >> @intCast(b % 64)));
            },
            .i64_rotl => {
                const b: u64 = @bitCast(try env.popI64());
                const a: u64 = @bitCast(try env.popI64());
                try env.pushI64(@bitCast(std.math.rotl(u64, a, b % 64)));
            },
            .i64_rotr => {
                const b: u64 = @bitCast(try env.popI64());
                const a: u64 = @bitCast(try env.popI64());
                try env.pushI64(@bitCast(std.math.rotr(u64, a, b % 64)));
            },

            // ── f32 comparison ──
            .f32_eq => {
                const b = try env.popF32();
                const a = try env.popF32();
                try env.pushI32(@intFromBool(a == b));
            },
            .f32_ne => {
                const b = try env.popF32();
                const a = try env.popF32();
                try env.pushI32(@intFromBool(a != b));
            },
            .f32_lt => {
                const b = try env.popF32();
                const a = try env.popF32();
                try env.pushI32(@intFromBool(a < b));
            },
            .f32_gt => {
                const b = try env.popF32();
                const a = try env.popF32();
                try env.pushI32(@intFromBool(a > b));
            },
            .f32_le => {
                const b = try env.popF32();
                const a = try env.popF32();
                try env.pushI32(@intFromBool(a <= b));
            },
            .f32_ge => {
                const b = try env.popF32();
                const a = try env.popF32();
                try env.pushI32(@intFromBool(a >= b));
            },

            // ── f32 unary ──
            .f32_abs => try env.pushF32(@abs(try env.popF32())),
            .f32_neg => try env.pushF32(-(try env.popF32())),
            .f32_ceil => try env.pushF32(canonF32(@ceil(try env.popF32()))),
            .f32_floor => try env.pushF32(canonF32(@floor(try env.popF32()))),
            .f32_trunc => try env.pushF32(canonF32(@trunc(try env.popF32()))),
            .f32_nearest => try env.pushF32(wasmNearestF32(try env.popF32())),
            .f32_sqrt => try env.pushF32(canonF32(@sqrt(try env.popF32()))),

            // ── f32 arithmetic ──
            .f32_add => { const b = try env.popF32(); try env.pushF32(canonF32(try env.popF32() + b)); },
            .f32_sub => { const b = try env.popF32(); try env.pushF32(canonF32(try env.popF32() - b)); },
            .f32_mul => { const b = try env.popF32(); try env.pushF32(canonF32(try env.popF32() * b)); },
            .f32_div => { const b = try env.popF32(); try env.pushF32(canonF32(try env.popF32() / b)); },
            .f32_min => {
                const b = try env.popF32();
                const a = try env.popF32();
                try env.pushF32(wasmMinF32(a, b));
            },
            .f32_max => {
                const b = try env.popF32();
                const a = try env.popF32();
                try env.pushF32(wasmMaxF32(a, b));
            },
            .f32_copysign => {
                const b = try env.popF32();
                const a = try env.popF32();
                try env.pushF32(std.math.copysign(a, b));
            },

            // ── f64 comparison ──
            .f64_eq => {
                const b = try env.popF64();
                const a = try env.popF64();
                try env.pushI32(@intFromBool(a == b));
            },
            .f64_ne => {
                const b = try env.popF64();
                const a = try env.popF64();
                try env.pushI32(@intFromBool(a != b));
            },
            .f64_lt => {
                const b = try env.popF64();
                const a = try env.popF64();
                try env.pushI32(@intFromBool(a < b));
            },
            .f64_gt => {
                const b = try env.popF64();
                const a = try env.popF64();
                try env.pushI32(@intFromBool(a > b));
            },
            .f64_le => {
                const b = try env.popF64();
                const a = try env.popF64();
                try env.pushI32(@intFromBool(a <= b));
            },
            .f64_ge => {
                const b = try env.popF64();
                const a = try env.popF64();
                try env.pushI32(@intFromBool(a >= b));
            },

            // ── f64 unary ──
            .f64_abs => try env.pushF64(@abs(try env.popF64())),
            .f64_neg => try env.pushF64(-(try env.popF64())),
            .f64_ceil => try env.pushF64(canonF64(@ceil(try env.popF64()))),
            .f64_floor => try env.pushF64(canonF64(@floor(try env.popF64()))),
            .f64_trunc => try env.pushF64(canonF64(@trunc(try env.popF64()))),
            .f64_nearest => try env.pushF64(wasmNearestF64(try env.popF64())),
            .f64_sqrt => try env.pushF64(canonF64(@sqrt(try env.popF64()))),

            // ── f64 arithmetic ──
            .f64_add => { const b = try env.popF64(); try env.pushF64(canonF64(try env.popF64() + b)); },
            .f64_sub => { const b = try env.popF64(); try env.pushF64(canonF64(try env.popF64() - b)); },
            .f64_mul => { const b = try env.popF64(); try env.pushF64(canonF64(try env.popF64() * b)); },
            .f64_div => { const b = try env.popF64(); try env.pushF64(canonF64(try env.popF64() / b)); },
            .f64_min => {
                const b = try env.popF64();
                const a = try env.popF64();
                try env.pushF64(wasmMinF64(a, b));
            },
            .f64_max => {
                const b = try env.popF64();
                const a = try env.popF64();
                try env.pushF64(wasmMaxF64(a, b));
            },
            .f64_copysign => {
                const b = try env.popF64();
                const a = try env.popF64();
                try env.pushF64(std.math.copysign(a, b));
            },

            // ── Conversions ──
            .i32_wrap_i64 => try env.pushI32(@truncate(try env.popI64())),
            .i64_extend_i32_s => try env.pushI64(@as(i64, try env.popI32())),
            .i64_extend_i32_u => try env.pushI64(@as(i64, @as(u32, @bitCast(try env.popI32())))),
            .f32_convert_i32_s => try env.pushF32(@floatFromInt(try env.popI32())),
            .f32_convert_i32_u => try env.pushF32(@floatFromInt(@as(u32, @bitCast(try env.popI32())))),
            .f64_convert_i32_s => try env.pushF64(@floatFromInt(try env.popI32())),
            .f64_convert_i32_u => try env.pushF64(@floatFromInt(@as(u32, @bitCast(try env.popI32())))),
            .f32_convert_i64_s => try env.pushF32(@floatFromInt(try env.popI64())),
            .f32_convert_i64_u => try env.pushF32(@floatFromInt(@as(u64, @bitCast(try env.popI64())))),
            .f64_convert_i64_s => try env.pushF64(@floatFromInt(try env.popI64())),
            .f64_convert_i64_u => try env.pushF64(@floatFromInt(@as(u64, @bitCast(try env.popI64())))),
            .f32_demote_f64 => try env.pushF32(@floatCast(try env.popF64())),
            .f64_promote_f32 => try env.pushF64(@as(f64, try env.popF32())),
            .i32_trunc_f32_s => {
                const v = try env.popF32();
                if (std.math.isNan(v) or std.math.isInf(v)) return error.InvalidConversionToInteger;
                if (v >= 2147483648.0 or v < -2147483648.0) return error.IntegerOverflow;
                try env.pushI32(@intFromFloat(v));
            },
            .i32_trunc_f32_u => {
                const v = try env.popF32();
                if (std.math.isNan(v) or std.math.isInf(v)) return error.InvalidConversionToInteger;
                if (v >= 4294967296.0 or v <= -1.0) return error.IntegerOverflow;
                try env.pushI32(@bitCast(@as(u32, @intFromFloat(v))));
            },
            .i32_trunc_f64_s => {
                const v = try env.popF64();
                if (std.math.isNan(v) or std.math.isInf(v)) return error.InvalidConversionToInteger;
                if (v >= 2147483648.0 or v <= -2147483649.0) return error.IntegerOverflow;
                try env.pushI32(@intFromFloat(v));
            },
            .i32_trunc_f64_u => {
                const v = try env.popF64();
                if (std.math.isNan(v) or std.math.isInf(v)) return error.InvalidConversionToInteger;
                if (v >= 4294967296.0 or v <= -1.0) return error.IntegerOverflow;
                try env.pushI32(@bitCast(@as(u32, @intFromFloat(v))));
            },
            .i64_trunc_f32_s => {
                const v = try env.popF32();
                if (std.math.isNan(v) or std.math.isInf(v)) return error.InvalidConversionToInteger;
                if (v >= 9223372036854775808.0 or v < -9223372036854775808.0) return error.IntegerOverflow;
                try env.pushI64(@intFromFloat(v));
            },
            .i64_trunc_f32_u => {
                const v = try env.popF32();
                if (std.math.isNan(v) or std.math.isInf(v)) return error.InvalidConversionToInteger;
                if (v >= 18446744073709551616.0 or v <= -1.0) return error.IntegerOverflow;
                try env.pushI64(@bitCast(@as(u64, @intFromFloat(v))));
            },
            .i64_trunc_f64_s => {
                const v = try env.popF64();
                if (std.math.isNan(v) or std.math.isInf(v)) return error.InvalidConversionToInteger;
                if (v >= 9223372036854775808.0 or v < -9223372036854775808.0) return error.IntegerOverflow;
                try env.pushI64(@intFromFloat(v));
            },
            .i64_trunc_f64_u => {
                const v = try env.popF64();
                if (std.math.isNan(v) or std.math.isInf(v)) return error.InvalidConversionToInteger;
                if (v >= 18446744073709551616.0 or v <= -1.0) return error.IntegerOverflow;
                try env.pushI64(@bitCast(@as(u64, @intFromFloat(v))));
            },

            // ── Reinterpretations ──
            .i32_reinterpret_f32 => {
                const v = try env.popF32();
                try env.pushI32(@bitCast(v));
            },
            .i64_reinterpret_f64 => {
                const v = try env.popF64();
                try env.pushI64(@bitCast(v));
            },
            .f32_reinterpret_i32 => {
                const v = try env.popI32();
                try env.pushF32(@bitCast(v));
            },
            .f64_reinterpret_i64 => {
                const v = try env.popI64();
                try env.pushF64(@bitCast(v));
            },

            // ── Sign-extension operators ──
            .i32_extend8_s => {
                const val = try env.popI32();
                try env.pushI32(@as(i32, @as(i8, @truncate(val))));
            },
            .i32_extend16_s => {
                const val = try env.popI32();
                try env.pushI32(@as(i32, @as(i16, @truncate(val))));
            },
            .i64_extend8_s => {
                const val = try env.popI64();
                try env.pushI64(@as(i64, @as(i8, @truncate(val))));
            },
            .i64_extend16_s => {
                const val = try env.popI64();
                try env.pushI64(@as(i64, @as(i16, @truncate(val))));
            },
            .i64_extend32_s => {
                const val = try env.popI64();
                try env.pushI64(@as(i64, @as(i32, @truncate(val))));
            },

            // ── Reference types ──
            .ref_null => {
                const ref_type = readU32(code, &ip);
                if (ref_type == @intFromEnum(types.ValType.externref) or ref_type == 0x72) {
                    try env.push(.{ .externref = null });
                } else {
                    try env.push(.{ .funcref = null });
                }
            },
            .ref_is_null => {
                const val = try env.pop();
                const is_null: bool = switch (val) {
                    .funcref, .nonfuncref => |r| r == null,
                    .externref, .nonexternref => |r| r == null,
                    else => false,
                };
                try env.pushI32(@intFromBool(is_null));
            },
            .ref_func => {
                const func_idx = readU32(code, &ip);
                try env.push(.{ .nonfuncref = func_idx });
            },

            // ── Typed function reference ops ──
            .ref_as_non_null => {
                const val = try env.peek();
                switch (val) {
                    .funcref, .nonfuncref => |r| if (r == null) return error.Unreachable,
                    .externref, .nonexternref => |r| if (r == null) return error.Unreachable,
                    else => return error.Unreachable,
                }
            },
            .br_on_null => {
                const depth = readU32(code, &ip);
                const val = try env.peek();
                const is_null = switch (val) {
                    .funcref, .nonfuncref => |r| r == null,
                    .externref, .nonexternref => |r| r == null,
                    else => false,
                };
                if (is_null) {
                    _ = try env.pop(); // consume the null ref
                    // Branch (same as br)
                    if (depth >= label_sp) return error.StackUnderflow;
                    const label = labels[label_sp - 1 - depth];
                    {
                        var results: [16]types.Value = undefined;
                        var ri: u32 = 0;
                        while (ri < label.arity) : (ri += 1) {
                            results[label.arity - 1 - ri] = try env.pop();
                        }
                        env.sp = label.stack_height;
                        ri = 0;
                        while (ri < label.arity) : (ri += 1) {
                            try env.push(results[ri]);
                        }
                    }
                    label_sp -= depth + 1;
                    if (label.kind == .loop) {
                        labels[label_sp] = label;
                        label_sp += 1;
                    }
                    ip = label.target_ip;
                }
                // If not null, continue with the non-null ref on the stack
            },
            .ref_eq => {
                const b = try env.pop();
                const a = try env.pop();
                const eq: bool = switch (a) {
                    .funcref, .nonfuncref => |ra| switch (b) {
                        .funcref, .nonfuncref => |rb| ra == rb,
                        else => false,
                    },
                    .externref, .nonexternref => |ra| switch (b) {
                        .externref, .nonexternref => |rb| ra == rb,
                        else => false,
                    },
                    else => false,
                };
                try env.pushI32(@intFromBool(eq));
            },
            .br_on_non_null => {
                const depth = readU32(code, &ip);
                const val = try env.peek();
                const is_null = switch (val) {
                    .funcref, .nonfuncref => |r| r == null,
                    .externref, .nonexternref => |r| r == null,
                    else => true,
                };
                if (!is_null) {
                    // Branch with the non-null ref on top
                    if (depth >= label_sp) return error.StackUnderflow;
                    const label = labels[label_sp - 1 - depth];
                    {
                        var results: [16]types.Value = undefined;
                        var ri: u32 = 0;
                        while (ri < label.arity) : (ri += 1) {
                            results[label.arity - 1 - ri] = try env.pop();
                        }
                        env.sp = label.stack_height;
                        ri = 0;
                        while (ri < label.arity) : (ri += 1) {
                            try env.push(results[ri]);
                        }
                    }
                    label_sp -= depth + 1;
                    if (label.kind == .loop) {
                        labels[label_sp] = label;
                        label_sp += 1;
                    }
                    ip = label.target_ip;
                } else {
                    _ = try env.pop(); // consume the null ref, don't branch
                }
            },
            .call_ref => {
                const type_idx = readU32(code, &ip);
                const ref_val = try env.pop();
                const func_idx = switch (ref_val) {
                    .funcref, .nonfuncref => |r| r orelse return error.Unreachable,
                    else => return error.Unreachable,
                };
                _ = type_idx;
                // Call the function by index (same as call)
                try executeFunction(env, func_idx);
            },
            .return_call_ref => {
                const type_idx = readU32(code, &ip);
                const ref_val = try env.pop();
                const func_idx = switch (ref_val) {
                    .funcref, .nonfuncref => |r| r orelse return error.Unreachable,
                    else => return error.Unreachable,
                };
                _ = type_idx;
                try prepareTailCall(env, func_idx);
                tail_call_target.* = func_idx;
                return .tail_call;
            },

            // ── Table ops ──
            .table_get => {
                const table_idx = readU32(code, &ip);
                const elem_idx: u32 = @bitCast(try env.popI32());
                const table = if (table_idx < env.module_inst.tables.len) env.module_inst.tables[table_idx] else return error.OutOfBoundsTableAccess;
                if (elem_idx >= table.elements.len) return error.OutOfBoundsTableAccess;
                const ref = table.elements[elem_idx];
                if (table.table_type.elem_type.isExternRef()) {
                    try env.push(.{ .externref = if (ref) |r| r.func_idx else null });
                } else {
                    try env.push(.{ .funcref = if (ref) |r| r.func_idx else null });
                }
            },
            .table_set => {
                const table_idx = readU32(code, &ip);
                const ref = try env.pop();
                const elem_idx: u32 = @bitCast(try env.popI32());
                const table = if (table_idx < env.module_inst.tables.len) env.module_inst.tables[table_idx] else return error.OutOfBoundsTableAccess;
                if (elem_idx >= table.elements.len) return error.OutOfBoundsTableAccess;
                const raw_ref: ?u32 = switch (ref) {
                    .funcref, .nonfuncref => |r| r,
                    .externref, .nonexternref => |r| r,
                    else => null,
                };
                table.elements[elem_idx] = if (raw_ref) |r| .{ .func_idx = r, .module_inst = env.module_inst } else null;
            },

            // ── Misc prefix (0xFC) ──
            .misc_prefix => {
                const sub_op = readU32(code, &ip);
                switch (sub_op) {
                    0 => { // i32.trunc_sat_f32_s
                        const v = try env.popF32();
                        if (std.math.isNan(v)) {
                            try env.pushI32(0);
                        } else if (v >= 2147483648.0) {
                            try env.pushI32(std.math.maxInt(i32));
                        } else if (v < -2147483648.0) {
                            try env.pushI32(std.math.minInt(i32));
                        } else {
                            try env.pushI32(@intFromFloat(v));
                        }
                    },
                    1 => { // i32.trunc_sat_f32_u
                        const v = try env.popF32();
                        if (std.math.isNan(v) or v < 0.0) {
                            try env.pushI32(0);
                        } else if (v >= 4294967296.0) {
                            try env.pushI32(@as(i32, @bitCast(@as(u32, std.math.maxInt(u32)))));
                        } else {
                            try env.pushI32(@bitCast(@as(u32, @intFromFloat(v))));
                        }
                    },
                    2 => { // i32.trunc_sat_f64_s
                        const v = try env.popF64();
                        if (std.math.isNan(v)) {
                            try env.pushI32(0);
                        } else if (v >= 2147483648.0) {
                            try env.pushI32(std.math.maxInt(i32));
                        } else if (v < -2147483648.0) {
                            try env.pushI32(std.math.minInt(i32));
                        } else {
                            try env.pushI32(@intFromFloat(v));
                        }
                    },
                    3 => { // i32.trunc_sat_f64_u
                        const v = try env.popF64();
                        if (std.math.isNan(v) or v < 0.0) {
                            try env.pushI32(0);
                        } else if (v >= 4294967296.0) {
                            try env.pushI32(@as(i32, @bitCast(@as(u32, std.math.maxInt(u32)))));
                        } else {
                            try env.pushI32(@bitCast(@as(u32, @intFromFloat(v))));
                        }
                    },
                    4 => { // i64.trunc_sat_f32_s
                        const v = try env.popF32();
                        if (std.math.isNan(v)) {
                            try env.pushI64(0);
                        } else if (v >= 9223372036854775808.0) {
                            try env.pushI64(std.math.maxInt(i64));
                        } else if (v < -9223372036854775808.0) {
                            try env.pushI64(std.math.minInt(i64));
                        } else {
                            try env.pushI64(@intFromFloat(v));
                        }
                    },
                    5 => { // i64.trunc_sat_f32_u
                        const v = try env.popF32();
                        if (std.math.isNan(v) or v < 0.0) {
                            try env.pushI64(0);
                        } else if (v >= 18446744073709551616.0) {
                            try env.pushI64(@as(i64, @bitCast(@as(u64, std.math.maxInt(u64)))));
                        } else {
                            try env.pushI64(@bitCast(@as(u64, @intFromFloat(v))));
                        }
                    },
                    6 => { // i64.trunc_sat_f64_s
                        const v = try env.popF64();
                        if (std.math.isNan(v)) {
                            try env.pushI64(0);
                        } else if (v >= 9223372036854775808.0) {
                            try env.pushI64(std.math.maxInt(i64));
                        } else if (v < -9223372036854775808.0) {
                            try env.pushI64(std.math.minInt(i64));
                        } else {
                            try env.pushI64(@intFromFloat(v));
                        }
                    },
                    7 => { // i64.trunc_sat_f64_u
                        const v = try env.popF64();
                        if (std.math.isNan(v) or v < 0.0) {
                            try env.pushI64(0);
                        } else if (v >= 18446744073709551616.0) {
                            try env.pushI64(@as(i64, @bitCast(@as(u64, std.math.maxInt(u64)))));
                        } else {
                            try env.pushI64(@bitCast(@as(u64, @intFromFloat(v))));
                        }
                    },
                    8 => { // memory.init
                        _ = readU32(code, &ip); // data_idx
                        _ = readU32(code, &ip); // memory index
                        return error.UnknownOpcode;
                    },
                    9 => { // data.drop
                        _ = readU32(code, &ip);
                    },
                    10 => { // memory.copy
                        const dst_mem_idx = readU32(code, &ip);
                        const src_mem_idx = readU32(code, &ip);
                        const n: u32 = @bitCast(try env.popI32());
                        const src: u32 = @bitCast(try env.popI32());
                        const dst: u32 = @bitCast(try env.popI32());
                        const dst_mem = env.module_inst.getMemory(dst_mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                        const src_mem = env.module_inst.getMemory(src_mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                        if (@as(u64, dst) + n > dst_mem.data.len or @as(u64, src) + n > src_mem.data.len) return error.OutOfBoundsMemoryAccess;
                        const d: usize = dst;
                        const s: usize = src;
                        const len: usize = n;
                        if (dst_mem == src_mem) {
                            if (d <= s) {
                                std.mem.copyForwards(u8, dst_mem.data[d .. d + len], src_mem.data[s .. s + len]);
                            } else {
                                std.mem.copyBackwards(u8, dst_mem.data[d .. d + len], src_mem.data[s .. s + len]);
                            }
                        } else {
                            @memcpy(dst_mem.data[d .. d + len], src_mem.data[s .. s + len]);
                        }
                    },
                    11 => { // memory.fill
                        const fill_mem_idx = readU32(code, &ip);
                        const n: u32 = @bitCast(try env.popI32());
                        const val: u8 = @truncate(@as(u32, @bitCast(try env.popI32())));
                        const dst: u32 = @bitCast(try env.popI32());
                        const mem = env.module_inst.getMemory(fill_mem_idx) orelse return error.OutOfBoundsMemoryAccess;
                        if (@as(u64, dst) + n > mem.data.len) return error.OutOfBoundsMemoryAccess;
                        const d: usize = dst;
                        const len: usize = n;
                        @memset(mem.data[d .. d + len], val);
                    },
                    12 => { // table.init
                        const elem_idx = readU32(code, &ip);
                        const table_idx = readU32(code, &ip);
                        const n: u32 = @bitCast(try env.popI32());
                        const s: u32 = @bitCast(try env.popI32());
                        const d: u32 = @bitCast(try env.popI32());
                        const module = env.module_inst.module;
                        if (elem_idx >= module.elements.len) return error.OutOfBoundsTableAccess;
                        // Check if segment has been dropped
                        if (elem_idx < env.module_inst.dropped_elems.len and env.module_inst.dropped_elems[elem_idx]) {
                            if (n > 0) return error.OutOfBoundsTableAccess;
                            continue;
                        }
                        const elem = &module.elements[elem_idx];
                        const table = if (table_idx < env.module_inst.tables.len) env.module_inst.tables[table_idx] else return error.OutOfBoundsTableAccess;
                        if (@as(u64, s) + n > elem.func_indices.len or @as(u64, d) + n > table.elements.len) return error.OutOfBoundsTableAccess;
                        for (0..n) |i| {
                            const mfi = elem.func_indices[s + @as(u32, @intCast(i))];
                            table.elements[d + @as(u32, @intCast(i))] = if (mfi) |fi|
                                .{ .func_idx = fi, .module_inst = env.module_inst }
                            else
                                null;
                        }
                    },
                    13 => { // elem.drop
                        const idx = readU32(code, &ip);
                        if (idx < env.module_inst.dropped_elems.len) {
                            env.module_inst.dropped_elems[idx] = true;
                        }
                    },
                    14 => { // table.copy
                        const dst_table_idx = readU32(code, &ip);
                        const src_table_idx = readU32(code, &ip);
                        const n: u32 = @bitCast(try env.popI32());
                        const s: u32 = @bitCast(try env.popI32());
                        const d: u32 = @bitCast(try env.popI32());
                        const dst_table = if (dst_table_idx < env.module_inst.tables.len) env.module_inst.tables[dst_table_idx] else return error.OutOfBoundsTableAccess;
                        const src_table = if (src_table_idx < env.module_inst.tables.len) env.module_inst.tables[src_table_idx] else return error.OutOfBoundsTableAccess;
                        if (@as(u64, s) + n > src_table.elements.len or @as(u64, d) + n > dst_table.elements.len) return error.OutOfBoundsTableAccess;
                        if (d <= s) {
                            for (0..n) |i| dst_table.elements[d + @as(u32, @intCast(i))] = src_table.elements[s + @as(u32, @intCast(i))];
                        } else {
                            var i: u32 = n;
                            while (i > 0) {
                                i -= 1;
                                dst_table.elements[d + i] = src_table.elements[s + i];
                            }
                        }
                    },
                    15 => { // table.grow
                        const table_idx = readU32(code, &ip);
                        const delta: u32 = @bitCast(try env.popI32());
                        const init_ref = try env.pop();
                        const table = if (table_idx < env.module_inst.tables.len) env.module_inst.tables[table_idx] else {
                            try env.pushI32(-1);
                            continue;
                        };
                        const old_size: u32 = @intCast(table.elements.len);
                        const new_size = @as(u64, old_size) + delta;
                        // Table size cannot exceed u32 range
                        if (new_size > std.math.maxInt(u32)) {
                            try env.pushI32(-1);
                            continue;
                        }
                        if (table.table_type.limits.max) |max| {
                            if (new_size > max) {
                                try env.pushI32(-1);
                                continue;
                            }
                        }
                        const new_elems = env.module_inst.allocator.realloc(table.elements, @intCast(new_size)) catch {
                            try env.pushI32(-1);
                            continue;
                        };
                        const init_val: ?types.FuncRef = switch (init_ref) {
                            .funcref, .nonfuncref => |r| if (r) |idx| .{ .func_idx = idx, .module_inst = env.module_inst } else null,
                            .externref, .nonexternref => |r| if (r) |idx| .{ .func_idx = idx, .module_inst = env.module_inst } else null,
                            else => null,
                        };
                        for (new_elems[old_size..]) |*e| e.* = init_val;
                        table.elements = new_elems;
                        try env.pushI32(@bitCast(old_size));
                    },
                    16 => { // table.size
                        const table_idx = readU32(code, &ip);
                        const table = if (table_idx < env.module_inst.tables.len) env.module_inst.tables[table_idx] else return error.OutOfBoundsTableAccess;
                        try env.pushI32(@intCast(table.elements.len));
                    },
                    17 => { // table.fill
                        const table_idx = readU32(code, &ip);
                        const n: u32 = @bitCast(try env.popI32());
                        const val = try env.pop();
                        const offset: u32 = @bitCast(try env.popI32());
                        const table = if (table_idx < env.module_inst.tables.len) env.module_inst.tables[table_idx] else return error.OutOfBoundsTableAccess;
                        if (@as(u64, offset) + n > table.elements.len) return error.OutOfBoundsTableAccess;
                        const ref_val: ?types.FuncRef = switch (val) {
                            .funcref, .nonfuncref => |r| if (r) |idx| .{ .func_idx = idx, .module_inst = env.module_inst } else null,
                            .externref, .nonexternref => |r| if (r) |idx| .{ .func_idx = idx, .module_inst = env.module_inst } else null,
                            else => null,
                        };
                        for (table.elements[offset..][0..n]) |*e| e.* = ref_val;
                    },
                    else => return error.UnknownOpcode,
                }
            },

            .simd_prefix => {
                const save_ip = ip;
                simd.executeSIMD(env, code, &ip) catch |err| switch (err) {
                    error.UnknownOpcode => {
                        ip = save_ip;
                        return error.UnknownOpcode;
                    },
                    error.OutOfBoundsMemoryAccess => return error.OutOfBoundsMemoryAccess,
                    error.Unreachable => return error.Unreachable,
                    error.StackOverflow => return error.StackOverflow,
                    error.StackUnderflow => return error.StackUnderflow,
                };
            },

            .atomic_prefix => {
                const sub_op = readU32(code, &ip);
                switch (sub_op) {
                    0x03 => { // atomic.fence
                        _ = readU32(code, &ip); // reserved byte (must be 0)
                        // Fence is a no-op for single-threaded execution.
                        // When multi-threading is implemented, use a proper barrier.
                    },
                    // atomic loads
                    0x10 => try atomicLoad(env, code, &ip, u32, u32, 4),   // i32.atomic.load
                    0x11 => try atomicLoad64(env, code, &ip, u64, 8),      // i64.atomic.load
                    0x12 => try atomicLoad(env, code, &ip, u8, u32, 1),    // i32.atomic.load8_u
                    0x13 => try atomicLoad(env, code, &ip, u16, u32, 2),   // i32.atomic.load16_u
                    0x14 => try atomicLoad64(env, code, &ip, u8, 1),       // i64.atomic.load8_u
                    0x15 => try atomicLoad64(env, code, &ip, u16, 2),      // i64.atomic.load16_u
                    0x16 => try atomicLoad64(env, code, &ip, u32, 4),      // i64.atomic.load32_u
                    // atomic stores
                    0x17 => try atomicStore(env, code, &ip, u32, 4),       // i32.atomic.store
                    0x18 => try atomicStore64(env, code, &ip, u64, 8),     // i64.atomic.store
                    0x19 => try atomicStore(env, code, &ip, u8, 1),        // i32.atomic.store8
                    0x1A => try atomicStore(env, code, &ip, u16, 2),       // i32.atomic.store16
                    0x1B => try atomicStore64(env, code, &ip, u8, 1),      // i64.atomic.store8
                    0x1C => try atomicStore64(env, code, &ip, u16, 2),     // i64.atomic.store16
                    0x1D => try atomicStore64(env, code, &ip, u32, 4),     // i64.atomic.store32
                    // RMW add
                    0x1E => try atomicRmw(env, code, &ip, u32, 4, .Add),
                    0x1F => try atomicRmw64(env, code, &ip, u64, 8, .Add),
                    0x20 => try atomicRmw(env, code, &ip, u8, 1, .Add),
                    0x21 => try atomicRmw(env, code, &ip, u16, 2, .Add),
                    0x22 => try atomicRmw64(env, code, &ip, u8, 1, .Add),
                    0x23 => try atomicRmw64(env, code, &ip, u16, 2, .Add),
                    0x24 => try atomicRmw64(env, code, &ip, u32, 4, .Add),
                    // RMW sub
                    0x25 => try atomicRmw(env, code, &ip, u32, 4, .Sub),
                    0x26 => try atomicRmw64(env, code, &ip, u64, 8, .Sub),
                    0x27 => try atomicRmw(env, code, &ip, u8, 1, .Sub),
                    0x28 => try atomicRmw(env, code, &ip, u16, 2, .Sub),
                    0x29 => try atomicRmw64(env, code, &ip, u8, 1, .Sub),
                    0x2A => try atomicRmw64(env, code, &ip, u16, 2, .Sub),
                    0x2B => try atomicRmw64(env, code, &ip, u32, 4, .Sub),
                    // RMW and
                    0x2C => try atomicRmw(env, code, &ip, u32, 4, .And),
                    0x2D => try atomicRmw64(env, code, &ip, u64, 8, .And),
                    0x2E => try atomicRmw(env, code, &ip, u8, 1, .And),
                    0x2F => try atomicRmw(env, code, &ip, u16, 2, .And),
                    0x30 => try atomicRmw64(env, code, &ip, u8, 1, .And),
                    0x31 => try atomicRmw64(env, code, &ip, u16, 2, .And),
                    0x32 => try atomicRmw64(env, code, &ip, u32, 4, .And),
                    // RMW or
                    0x33 => try atomicRmw(env, code, &ip, u32, 4, .Or),
                    0x34 => try atomicRmw64(env, code, &ip, u64, 8, .Or),
                    0x35 => try atomicRmw(env, code, &ip, u8, 1, .Or),
                    0x36 => try atomicRmw(env, code, &ip, u16, 2, .Or),
                    0x37 => try atomicRmw64(env, code, &ip, u8, 1, .Or),
                    0x38 => try atomicRmw64(env, code, &ip, u16, 2, .Or),
                    0x39 => try atomicRmw64(env, code, &ip, u32, 4, .Or),
                    // RMW xor
                    0x3A => try atomicRmw(env, code, &ip, u32, 4, .Xor),
                    0x3B => try atomicRmw64(env, code, &ip, u64, 8, .Xor),
                    0x3C => try atomicRmw(env, code, &ip, u8, 1, .Xor),
                    0x3D => try atomicRmw(env, code, &ip, u16, 2, .Xor),
                    0x3E => try atomicRmw64(env, code, &ip, u8, 1, .Xor),
                    0x3F => try atomicRmw64(env, code, &ip, u16, 2, .Xor),
                    0x40 => try atomicRmw64(env, code, &ip, u32, 4, .Xor),
                    // RMW xchg
                    0x41 => try atomicRmw(env, code, &ip, u32, 4, .Xchg),
                    0x42 => try atomicRmw64(env, code, &ip, u64, 8, .Xchg),
                    0x43 => try atomicRmw(env, code, &ip, u8, 1, .Xchg),
                    0x44 => try atomicRmw(env, code, &ip, u16, 2, .Xchg),
                    0x45 => try atomicRmw64(env, code, &ip, u8, 1, .Xchg),
                    0x46 => try atomicRmw64(env, code, &ip, u16, 2, .Xchg),
                    0x47 => try atomicRmw64(env, code, &ip, u32, 4, .Xchg),
                    // RMW cmpxchg
                    0x48 => try atomicCmpxchg(env, code, &ip, u32, 4),
                    0x49 => try atomicCmpxchg64(env, code, &ip, u64, 8),
                    0x4A => try atomicCmpxchg(env, code, &ip, u8, 1),
                    0x4B => try atomicCmpxchg(env, code, &ip, u16, 2),
                    0x4C => try atomicCmpxchg64(env, code, &ip, u8, 1),
                    0x4D => try atomicCmpxchg64(env, code, &ip, u16, 2),
                    0x4E => try atomicCmpxchg64(env, code, &ip, u32, 4),
                    // wait/notify (stub: single-threaded for now)
                    0x00 => { // memory.atomic.notify
                        _ = readU32(code, &ip); // align
                        _ = readU32(code, &ip); // offset
                        _ = try env.popI32(); // count
                        const _addr = try env.popI32(); // addr
                        _ = _addr;
                        try env.pushI32(0); // 0 waiters woken (single-threaded)
                    },
                    0x01 => { // memory.atomic.wait32
                        _ = readU32(code, &ip); // align
                        _ = readU32(code, &ip); // offset
                        _ = try env.popI64(); // timeout
                        _ = try env.popI32(); // expected
                        _ = try env.popI32(); // addr
                        try env.pushI32(1); // 1 = "not equal" (value already changed)
                    },
                    0x02 => { // memory.atomic.wait64
                        _ = readU32(code, &ip); // align
                        _ = readU32(code, &ip); // offset
                        _ = try env.popI64(); // timeout
                        _ = try env.popI64(); // expected
                        _ = try env.popI32(); // addr
                        try env.pushI32(1); // 1 = "not equal"
                    },
                    else => return error.UnknownOpcode,
                }
            },

            else => return error.UnknownOpcode,
        }
    }
    return .normal;
}

// ── Atomic operation helpers ─────────────────────────────────────────────

fn getAtomicAddr(env: *ExecEnv, code: []const u8, ip: *usize, comptime size: u32) !struct { ptr: [*]u8, addr: usize } {
    const ma = readMemarg(code, ip);
    const base: u32 = @bitCast(try env.popI32());
    const addr = @as(u64, base) + ma.offset;
    const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
    if (addr + size > mem.data.len) return error.OutOfBoundsMemoryAccess;
    // Alignment check (atomics require natural alignment)
    if (addr % size != 0) return error.UnalignedAtomicAccess;
    return .{ .ptr = mem.data.ptr + @as(usize, @intCast(addr)), .addr = @intCast(addr) };
}

fn atomicLoad(env: *ExecEnv, code: []const u8, ip: *usize, comptime T: type, comptime R: type, comptime size: u32) !void {
    const a = try getAtomicAddr(env, code, ip, size);
    const ptr: *const T = @ptrCast(@alignCast(a.ptr));
    const val = @atomicLoad(T, ptr, .seq_cst);
    try env.pushI32(@bitCast(@as(R, val)));
}

fn atomicLoad64(env: *ExecEnv, code: []const u8, ip: *usize, comptime T: type, comptime size: u32) !void {
    const a = try getAtomicAddr(env, code, ip, size);
    const ptr: *const T = @ptrCast(@alignCast(a.ptr));
    const val = @atomicLoad(T, ptr, .seq_cst);
    try env.pushI64(@bitCast(@as(u64, val)));
}

fn atomicStore(env: *ExecEnv, code: []const u8, ip: *usize, comptime T: type, comptime size: u32) !void {
    const ma = readMemarg(code, ip);
    const val: T = @truncate(@as(u32, @bitCast(try env.popI32())));
    const base: u32 = @bitCast(try env.popI32());
    const addr = @as(u64, base) + ma.offset;
    const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
    if (addr + size > mem.data.len) return error.OutOfBoundsMemoryAccess;
    if (addr % size != 0) return error.UnalignedAtomicAccess;
    const ptr: *T = @ptrCast(@alignCast(mem.data.ptr + @as(usize, @intCast(addr))));
    @atomicStore(T, ptr, val, .seq_cst);
}

fn atomicStore64(env: *ExecEnv, code: []const u8, ip: *usize, comptime T: type, comptime size: u32) !void {
    const ma = readMemarg(code, ip);
    const val: T = @truncate(@as(u64, @bitCast(try env.popI64())));
    const base: u32 = @bitCast(try env.popI32());
    const addr = @as(u64, base) + ma.offset;
    const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
    if (addr + size > mem.data.len) return error.OutOfBoundsMemoryAccess;
    if (addr % size != 0) return error.UnalignedAtomicAccess;
    const ptr: *T = @ptrCast(@alignCast(mem.data.ptr + @as(usize, @intCast(addr))));
    @atomicStore(T, ptr, val, .seq_cst);
}

fn atomicRmw(env: *ExecEnv, code: []const u8, ip: *usize, comptime T: type, comptime size: u32, comptime op: std.builtin.AtomicRmwOp) !void {
    const ma = readMemarg(code, ip);
    const val: T = @truncate(@as(u32, @bitCast(try env.popI32())));
    const base: u32 = @bitCast(try env.popI32());
    const addr = @as(u64, base) + ma.offset;
    const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
    if (addr + size > mem.data.len) return error.OutOfBoundsMemoryAccess;
    if (addr % size != 0) return error.UnalignedAtomicAccess;
    const ptr: *T = @ptrCast(@alignCast(mem.data.ptr + @as(usize, @intCast(addr))));
    const old = @atomicRmw(T, ptr, op, val, .seq_cst);
    try env.pushI32(@bitCast(@as(u32, old)));
}

fn atomicRmw64(env: *ExecEnv, code: []const u8, ip: *usize, comptime T: type, comptime size: u32, comptime op: std.builtin.AtomicRmwOp) !void {
    const ma = readMemarg(code, ip);
    const val: T = @truncate(@as(u64, @bitCast(try env.popI64())));
    const base: u32 = @bitCast(try env.popI32());
    const addr = @as(u64, base) + ma.offset;
    const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
    if (addr + size > mem.data.len) return error.OutOfBoundsMemoryAccess;
    if (addr % size != 0) return error.UnalignedAtomicAccess;
    const ptr: *T = @ptrCast(@alignCast(mem.data.ptr + @as(usize, @intCast(addr))));
    const old = @atomicRmw(T, ptr, op, val, .seq_cst);
    try env.pushI64(@bitCast(@as(u64, old)));
}

fn atomicCmpxchg(env: *ExecEnv, code: []const u8, ip: *usize, comptime T: type, comptime size: u32) !void {
    const ma = readMemarg(code, ip);
    const replacement: T = @truncate(@as(u32, @bitCast(try env.popI32())));
    const expected: T = @truncate(@as(u32, @bitCast(try env.popI32())));
    const base: u32 = @bitCast(try env.popI32());
    const addr = @as(u64, base) + ma.offset;
    const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
    if (addr + size > mem.data.len) return error.OutOfBoundsMemoryAccess;
    if (addr % size != 0) return error.UnalignedAtomicAccess;
    const ptr: *T = @ptrCast(@alignCast(mem.data.ptr + @as(usize, @intCast(addr))));
    const result = @cmpxchgStrong(T, ptr, expected, replacement, .seq_cst, .seq_cst);
    const old: u32 = if (result) |v| v else expected;
    try env.pushI32(@bitCast(old));
}

fn atomicCmpxchg64(env: *ExecEnv, code: []const u8, ip: *usize, comptime T: type, comptime size: u32) !void {
    const ma = readMemarg(code, ip);
    const replacement: T = @truncate(@as(u64, @bitCast(try env.popI64())));
    const expected: T = @truncate(@as(u64, @bitCast(try env.popI64())));
    const base: u32 = @bitCast(try env.popI32());
    const addr = @as(u64, base) + ma.offset;
    const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
    if (addr + size > mem.data.len) return error.OutOfBoundsMemoryAccess;
    if (addr % size != 0) return error.UnalignedAtomicAccess;
    const ptr: *T = @ptrCast(@alignCast(mem.data.ptr + @as(usize, @intCast(addr))));
    const result = @cmpxchgStrong(T, ptr, expected, replacement, .seq_cst, .seq_cst);
    const old: u64 = if (result) |v| v else expected;
    try env.pushI64(@bitCast(old));
}

// ── Tests ────────────────────────────────────────────────────────────────

const testing = std.testing;

/// Helper: run bytecode directly via dispatchLoop, returning stack top.
fn runCode(code: []const u8) !i32 {
    const alloc = testing.allocator;
    // Minimal module/instance for testing
    var dummy_module = types.WasmModule{};
    var memories = [_]*types.MemoryInstance{};
    var globals = [_]*types.GlobalInstance{};
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

    var dummy_target: u32 = 0;
    _ = try dispatchLoop(&env, code, &dummy_target);
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
    var dummy_target: u32 = 0;
    _ = try dispatchLoop(&env, &.{
        0x41, 42, // i32.const 42
        0x21, 1, // local.set 1
        0x20, 1, // local.get 1
        0x0B, // end
    }, &dummy_target);

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

    { var dummy_target: u32 = 0; _ = try dispatchLoop(&env, &code, &dummy_target); }
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
    { var dummy_target: u32 = 0; _ = try dispatchLoop(&env, &caller_code, &dummy_target); }
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

            { var dummy_target: u32 = 0; _ = try dispatchLoop(&env, &code, &dummy_target); }
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

// ── Helpers for i64/f32/f64 tests ────────────────────────────────────────

fn runCodeI64(code: []const u8) !i64 {
    const alloc = testing.allocator;
    var dummy_module = types.WasmModule{};
    var memories = [_]*types.MemoryInstance{};
    var globals = [_]*types.GlobalInstance{};
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
    try env.pushFrame(.{ .func_idx = 0, .ip = 0, .stack_base = 0, .local_count = 0, .return_arity = 1, .prev_sp = 0 });
    { var dummy_target: u32 = 0; _ = try dispatchLoop(&env, code, &dummy_target); }
    return env.popI64();
}

fn runCodeF32(code: []const u8) !f32 {
    const alloc = testing.allocator;
    var dummy_module = types.WasmModule{};
    var memories = [_]*types.MemoryInstance{};
    var globals = [_]*types.GlobalInstance{};
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
    try env.pushFrame(.{ .func_idx = 0, .ip = 0, .stack_base = 0, .local_count = 0, .return_arity = 1, .prev_sp = 0 });
    { var dummy_target: u32 = 0; _ = try dispatchLoop(&env, code, &dummy_target); }
    return env.popF32();
}

fn runCodeF64(code: []const u8) !f64 {
    const alloc = testing.allocator;
    var dummy_module = types.WasmModule{};
    var memories = [_]*types.MemoryInstance{};
    var globals = [_]*types.GlobalInstance{};
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
    try env.pushFrame(.{ .func_idx = 0, .ip = 0, .stack_base = 0, .local_count = 0, .return_arity = 1, .prev_sp = 0 });
    { var dummy_target: u32 = 0; _ = try dispatchLoop(&env, code, &dummy_target); }
    return env.popF64();
}

fn runCodeExpectTrapAny(code: []const u8, expected: TrapError) !void {
    const alloc = testing.allocator;
    var dummy_module = types.WasmModule{};
    var memories = [_]*types.MemoryInstance{};
    var globals = [_]*types.GlobalInstance{};
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
    try env.pushFrame(.{ .func_idx = 0, .ip = 0, .stack_base = 0, .local_count = 0, .return_arity = 1, .prev_sp = 0 });
    var dummy_target: u32 = 0;
    const result = dispatchLoop(&env, code, &dummy_target);
    try testing.expectError(expected, result);
}

// ── i64 tests ──

test "interp: i64.add" {
    // i64.const 3; i64.const 4; i64.add; end
    const result = try runCodeI64(&.{ 0x42, 3, 0x42, 4, 0x7C, 0x0B });
    try testing.expectEqual(@as(i64, 7), result);
}

test "interp: i64.sub" {
    const result = try runCodeI64(&.{ 0x42, 10, 0x42, 3, 0x7D, 0x0B });
    try testing.expectEqual(@as(i64, 7), result);
}

test "interp: i64.mul" {
    const result = try runCodeI64(&.{ 0x42, 6, 0x42, 7, 0x7E, 0x0B });
    try testing.expectEqual(@as(i64, 42), result);
}

test "interp: i64.div_s by zero traps" {
    try runCodeExpectTrapAny(&.{ 0x42, 1, 0x42, 0, 0x7F, 0x0B }, error.IntegerDivisionByZero);
}

test "interp: i64.div_u by zero traps" {
    try runCodeExpectTrapAny(&.{ 0x42, 1, 0x42, 0, 0x80, 0x0B }, error.IntegerDivisionByZero);
}

test "interp: i64.eqz true" {
    // i64.const 0; i64.eqz; end → 1 (as i32)
    const result = try runCode(&.{ 0x42, 0, 0x50, 0x0B });
    try testing.expectEqual(@as(i32, 1), result);
}

test "interp: i64.eqz false" {
    const result = try runCode(&.{ 0x42, 5, 0x50, 0x0B });
    try testing.expectEqual(@as(i32, 0), result);
}

test "interp: i64.eq" {
    // i64.const 42; i64.const 42; i64.eq; end → 1
    const result = try runCode(&.{ 0x42, 42, 0x42, 42, 0x51, 0x0B });
    try testing.expectEqual(@as(i32, 1), result);
}

test "interp: i64.lt_s" {
    // i64.const -1 (0x7F); i64.const 1; i64.lt_s; end → 1
    const result = try runCode(&.{ 0x42, 0x7F, 0x42, 1, 0x53, 0x0B });
    try testing.expectEqual(@as(i32, 1), result);
}

test "interp: i64.gt_u" {
    // i64.const -1 (0x7F as LEB128 sign-extends to max u64); i64.const 1; i64.gt_u → 1
    const result = try runCode(&.{ 0x42, 0x7F, 0x42, 1, 0x56, 0x0B });
    try testing.expectEqual(@as(i32, 1), result);
}

test "interp: i64.clz" {
    // i64.const 1; i64.clz; end → 63
    const result = try runCodeI64(&.{ 0x42, 1, 0x79, 0x0B });
    try testing.expectEqual(@as(i64, 63), result);
}

// ── f32 tests ──

test "interp: f32.add" {
    // f32.const 1.0; f32.const 2.0; f32.add; end
    // 1.0f = 0x3F800000 LE: 00 00 80 3F
    // 2.0f = 0x40000000 LE: 00 00 00 40
    const result = try runCodeF32(&.{
        0x43, 0x00, 0x00, 0x80, 0x3F, // f32.const 1.0
        0x43, 0x00, 0x00, 0x00, 0x40, // f32.const 2.0
        0x92, // f32.add
        0x0B, // end
    });
    try testing.expectApproxEqAbs(@as(f32, 3.0), result, 0.001);
}

test "interp: f32.sub" {
    const result = try runCodeF32(&.{
        0x43, 0x00, 0x00, 0x20, 0x41, // f32.const 10.0
        0x43, 0x00, 0x00, 0x80, 0x40, // f32.const 4.0
        0x93, // f32.sub
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f32, 6.0), result, 0.001);
}

test "interp: f32.mul" {
    const result = try runCodeF32(&.{
        0x43, 0x00, 0x00, 0x40, 0x40, // f32.const 3.0
        0x43, 0x00, 0x00, 0x80, 0x40, // f32.const 4.0
        0x94, // f32.mul
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f32, 12.0), result, 0.001);
}

test "interp: f32.div" {
    const result = try runCodeF32(&.{
        0x43, 0x00, 0x00, 0x20, 0x41, // f32.const 10.0
        0x43, 0x00, 0x00, 0x00, 0x40, // f32.const 2.0
        0x95, // f32.div
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f32, 5.0), result, 0.001);
}

test "interp: f32.eq" {
    const result = try runCode(&.{
        0x43, 0x00, 0x00, 0x80, 0x3F, // f32.const 1.0
        0x43, 0x00, 0x00, 0x80, 0x3F, // f32.const 1.0
        0x5B, // f32.eq
        0x0B,
    });
    try testing.expectEqual(@as(i32, 1), result);
}

test "interp: f32.ne" {
    const result = try runCode(&.{
        0x43, 0x00, 0x00, 0x80, 0x3F, // f32.const 1.0
        0x43, 0x00, 0x00, 0x00, 0x40, // f32.const 2.0
        0x5C, // f32.ne
        0x0B,
    });
    try testing.expectEqual(@as(i32, 1), result);
}

test "interp: f32.lt" {
    const result = try runCode(&.{
        0x43, 0x00, 0x00, 0x80, 0x3F, // f32.const 1.0
        0x43, 0x00, 0x00, 0x00, 0x40, // f32.const 2.0
        0x5D, // f32.lt
        0x0B,
    });
    try testing.expectEqual(@as(i32, 1), result);
}

// ── f64 tests ──

test "interp: f64.add" {
    // 1.0 as f64 LE = 0x3FF0000000000000 → 00 00 00 00 00 00 F0 3F
    // 2.0 as f64 LE = 0x4000000000000000 → 00 00 00 00 00 00 00 40
    const result = try runCodeF64(&.{
        0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // f64.const 1.0
        0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // f64.const 2.0
        0xA0, // f64.add
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f64, 3.0), result, 0.001);
}

test "interp: f64.sub" {
    // 10.0 f64 LE: 00 00 00 00 00 00 24 40
    // 4.0 f64 LE: 00 00 00 00 00 00 10 40
    const result = try runCodeF64(&.{
        0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, // f64.const 10.0
        0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x40, // f64.const 4.0
        0xA1, // f64.sub
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f64, 6.0), result, 0.001);
}

test "interp: f64.mul" {
    // 3.0 f64 LE: 00 00 00 00 00 00 08 40
    // 4.0 f64 LE: 00 00 00 00 00 00 10 40
    const result = try runCodeF64(&.{
        0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // f64.const 3.0
        0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x40, // f64.const 4.0
        0xA2, // f64.mul
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f64, 12.0), result, 0.001);
}

test "interp: f64.div" {
    const result = try runCodeF64(&.{
        0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, // f64.const 10.0
        0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // f64.const 2.0
        0xA3, // f64.div
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f64, 5.0), result, 0.001);
}

test "interp: f64.sqrt" {
    // 9.0 f64 LE: 00 00 00 00 00 00 22 40
    const result = try runCodeF64(&.{
        0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x22, 0x40, // f64.const 9.0
        0x9F, // f64.sqrt
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f64, 3.0), result, 0.001);
}

test "interp: f64.abs" {
    // -5.0 f64 LE: 00 00 00 00 00 00 14 C0
    const result = try runCodeF64(&.{
        0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14, 0xC0, // f64.const -5.0
        0x99, // f64.abs
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f64, 5.0), result, 0.001);
}

test "interp: f64.neg" {
    // 5.0 f64 LE: 00 00 00 00 00 00 14 40
    const result = try runCodeF64(&.{
        0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14, 0x40, // f64.const 5.0
        0x9A, // f64.neg
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f64, -5.0), result, 0.001);
}

// ── Conversion tests ──

test "interp: i64.extend_i32_s" {
    // i32.const -1 (0x7F); i64.extend_i32_s; end → -1 as i64
    const result = try runCodeI64(&.{ 0x41, 0x7F, 0xAC, 0x0B });
    try testing.expectEqual(@as(i64, -1), result);
}

test "interp: i64.extend_i32_u" {
    // i32.const -1 (0x7F); i64.extend_i32_u; end → 0xFFFFFFFF as i64
    const result = try runCodeI64(&.{ 0x41, 0x7F, 0xAD, 0x0B });
    try testing.expectEqual(@as(i64, 0xFFFFFFFF), result);
}

test "interp: i32.wrap_i64" {
    // i64.const 0x1_0000_0001 (LEB128); i32.wrap_i64; end → 1
    // 0x1_0000_0001 = 4294967297
    // LEB128: 0x81 0x80 0x80 0x80 0x10
    const result = try runCode(&.{
        0x42, 0x81, 0x80, 0x80, 0x80, 0x10, // i64.const 4294967297
        0xA7, // i32.wrap_i64
        0x0B,
    });
    try testing.expectEqual(@as(i32, 1), result);
}

test "interp: f32_reinterpret_i32 and i32_reinterpret_f32 roundtrip" {
    // i32.const 42; f32.reinterpret_i32; i32.reinterpret_f32; end → 42
    const result = try runCode(&.{
        0x41, 42, // i32.const 42
        0xBE, // f32.reinterpret_i32
        0xBC, // i32.reinterpret_f32
        0x0B,
    });
    try testing.expectEqual(@as(i32, 42), result);
}

test "interp: f64_reinterpret_i64 and i64_reinterpret_f64 roundtrip" {
    // i64.const 42; f64.reinterpret_i64; i64.reinterpret_f64; end → 42
    const result = try runCodeI64(&.{
        0x42, 42, // i64.const 42
        0xBF, // f64.reinterpret_i64
        0xBD, // i64.reinterpret_f64
        0x0B,
    });
    try testing.expectEqual(@as(i64, 42), result);
}

test "interp: i32_trunc_f32_s NaN traps" {
    // f32.const NaN; i32.trunc_f32_s → InvalidConversionToInteger
    // NaN f32 LE: 00 00 C0 7F
    try runCodeExpectTrapAny(&.{
        0x43, 0x00, 0x00, 0xC0, 0x7F, // f32.const NaN
        0xA8, // i32.trunc_f32_s
        0x0B,
    }, error.InvalidConversionToInteger);
}

test "interp: f32.convert_i32_s" {
    // i32.const -3; f32.convert_i32_s → -3.0
    const result = try runCodeF32(&.{
        0x41, 0x7D, // i32.const -3 (LEB128)
        0xB2, // f32.convert_i32_s
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f32, -3.0), result, 0.001);
}

test "interp: f64.convert_i32_u" {
    // i32.const 100 (LEB128: 0xE4, 0x00); f64.convert_i32_u → 100.0
    const result = try runCodeF64(&.{
        0x41, 0xE4, 0x00, // i32.const 100
        0xB8, // f64.convert_i32_u
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f64, 100.0), result, 0.001);
}

test "interp: f64.promote_f32" {
    // f32.const 1.5; f64.promote_f32 → 1.5
    // 1.5f = 0x3FC00000 LE: 00 00 C0 3F
    const result = try runCodeF64(&.{
        0x43, 0x00, 0x00, 0xC0, 0x3F, // f32.const 1.5
        0xBB, // f64.promote_f32
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f64, 1.5), result, 0.001);
}

test "interp: f32.demote_f64" {
    // f64.const 1.5; f32.demote_f64 → 1.5
    // 1.5 f64 LE: 00 00 00 00 00 00 F8 3F
    const result = try runCodeF32(&.{
        0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0x3F, // f64.const 1.5
        0xB6, // f32.demote_f64
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f32, 1.5), result, 0.001);
}

// ── Memory-aware test helpers ──

fn runCodeWithMem(code: []const u8) !i32 {
    const alloc = testing.allocator;
    var dummy_module = types.WasmModule{};
    const mem_data = try alloc.alloc(u8, types.MemoryInstance.page_size);
    @memset(mem_data, 0);
    var mem_inst = types.MemoryInstance{
        .memory_type = .{ .limits = .{ .min = 1, .max = 4 } },
        .data = mem_data,
        .current_pages = 1,
        .max_pages = 4,
    };
    defer alloc.free(mem_inst.data);
    var mem_ptrs = [_]*types.MemoryInstance{&mem_inst};
    var globals = [_]*types.GlobalInstance{};
    var dummy_inst = types.ModuleInstance{
        .module = &dummy_module,
        .memories = &mem_ptrs,
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
    try env.pushFrame(.{ .func_idx = 0, .ip = 0, .stack_base = 0, .local_count = 0, .return_arity = 1, .prev_sp = 0 });
    { var dummy_target: u32 = 0; _ = try dispatchLoop(&env, code, &dummy_target); }
    return env.popI32();
}

fn runCodeWithMemI64(code: []const u8) !i64 {
    const alloc = testing.allocator;
    var dummy_module = types.WasmModule{};
    const mem_data = try alloc.alloc(u8, types.MemoryInstance.page_size);
    @memset(mem_data, 0);
    var mem_inst = types.MemoryInstance{
        .memory_type = .{ .limits = .{ .min = 1, .max = 4 } },
        .data = mem_data,
        .current_pages = 1,
        .max_pages = 4,
    };
    defer alloc.free(mem_inst.data);
    var mem_ptrs = [_]*types.MemoryInstance{&mem_inst};
    var globals = [_]*types.GlobalInstance{};
    var dummy_inst = types.ModuleInstance{
        .module = &dummy_module,
        .memories = &mem_ptrs,
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
    try env.pushFrame(.{ .func_idx = 0, .ip = 0, .stack_base = 0, .local_count = 0, .return_arity = 1, .prev_sp = 0 });
    { var dummy_target: u32 = 0; _ = try dispatchLoop(&env, code, &dummy_target); }
    return env.popI64();
}

fn runCodeWithMemF32(code: []const u8) !f32 {
    const alloc = testing.allocator;
    var dummy_module = types.WasmModule{};
    const mem_data = try alloc.alloc(u8, types.MemoryInstance.page_size);
    @memset(mem_data, 0);
    var mem_inst = types.MemoryInstance{
        .memory_type = .{ .limits = .{ .min = 1, .max = 4 } },
        .data = mem_data,
        .current_pages = 1,
        .max_pages = 4,
    };
    defer alloc.free(mem_inst.data);
    var mem_ptrs = [_]*types.MemoryInstance{&mem_inst};
    var globals = [_]*types.GlobalInstance{};
    var dummy_inst = types.ModuleInstance{
        .module = &dummy_module,
        .memories = &mem_ptrs,
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
    try env.pushFrame(.{ .func_idx = 0, .ip = 0, .stack_base = 0, .local_count = 0, .return_arity = 1, .prev_sp = 0 });
    { var dummy_target: u32 = 0; _ = try dispatchLoop(&env, code, &dummy_target); }
    return env.popF32();
}

fn runCodeWithMemF64(code: []const u8) !f64 {
    const alloc = testing.allocator;
    var dummy_module = types.WasmModule{};
    const mem_data = try alloc.alloc(u8, types.MemoryInstance.page_size);
    @memset(mem_data, 0);
    var mem_inst = types.MemoryInstance{
        .memory_type = .{ .limits = .{ .min = 1, .max = 4 } },
        .data = mem_data,
        .current_pages = 1,
        .max_pages = 4,
    };
    defer alloc.free(mem_inst.data);
    var mem_ptrs = [_]*types.MemoryInstance{&mem_inst};
    var globals = [_]*types.GlobalInstance{};
    var dummy_inst = types.ModuleInstance{
        .module = &dummy_module,
        .memories = &mem_ptrs,
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
    try env.pushFrame(.{ .func_idx = 0, .ip = 0, .stack_base = 0, .local_count = 0, .return_arity = 1, .prev_sp = 0 });
    { var dummy_target: u32 = 0; _ = try dispatchLoop(&env, code, &dummy_target); }
    return env.popF64();
}

// ── Memory opcode tests ──

test "interp: memory.size on 1-page memory" {
    // memory.size (0x3F) with mem index 0x00; end → 1
    const result = try runCodeWithMem(&.{ 0x3F, 0x00, 0x0B });
    try testing.expectEqual(@as(i32, 1), result);
}

test "interp: memory.grow returns old page count" {
    // i32.const 1; memory.grow 0x00; end → 1 (old pages)
    // Then memory.size → 2
    // We test grow returns old=1 first:
    const result = try runCodeWithMem(&.{
        0x41, 1, // i32.const 1
        0x40, 0x00, // memory.grow
        0x0B,
    });
    try testing.expectEqual(@as(i32, 1), result);
}

test "interp: memory.grow then memory.size" {
    // i32.const 1; memory.grow; drop; memory.size; end → 2
    const result = try runCodeWithMem(&.{
        0x41, 1, // i32.const 1
        0x40, 0x00, // memory.grow
        0x1A, // drop
        0x3F, 0x00, // memory.size
        0x0B,
    });
    try testing.expectEqual(@as(i32, 2), result);
}

test "interp: memory.grow beyond max returns -1" {
    // max_pages=4, current=1, grow by 10 → -1
    const result = try runCodeWithMem(&.{
        0x41, 10, // i32.const 10
        0x40, 0x00, // memory.grow
        0x0B,
    });
    try testing.expectEqual(@as(i32, -1), result);
}

// ── Atomic opcode tests ──

test "interp: i32.atomic.store + i32.atomic.load" {
    // Store 42 at address 0, then load it back
    const result = try runCodeWithMem(&.{
        0x41, 0x00, // i32.const 0 (addr)
        0x41, 42, // i32.const 42 (value)
        0xFE, 0x17, 0x02, 0x00, // i32.atomic.store align=2 offset=0
        0x41, 0x00, // i32.const 0 (addr)
        0xFE, 0x10, 0x02, 0x00, // i32.atomic.load align=2 offset=0
        0x0B,
    });
    try testing.expectEqual(@as(i32, 42), result);
}

test "interp: i32.atomic.rmw.add" {
    // Store 10 at addr 0, then add 5, returns old value (10)
    const result = try runCodeWithMem(&.{
        0x41, 0x00, // i32.const 0 (addr)
        0x41, 10, // i32.const 10
        0xFE, 0x17, 0x02, 0x00, // i32.atomic.store align=2 offset=0
        0x41, 0x00, // i32.const 0 (addr)
        0x41, 5, // i32.const 5
        0xFE, 0x1E, 0x02, 0x00, // i32.atomic.rmw.add align=2 offset=0
        0x0B,
    });
    try testing.expectEqual(@as(i32, 10), result); // returns OLD value
}

test "interp: i32.atomic.rmw.cmpxchg success" {
    // Store 10 at addr 0, cmpxchg(expected=10, replacement=20) → returns old (10)
    const result = try runCodeWithMem(&.{
        0x41, 0x00, // i32.const 0 (addr)
        0x41, 10, // i32.const 10
        0xFE, 0x17, 0x02, 0x00, // i32.atomic.store align=2 offset=0
        0x41, 0x00, // i32.const 0 (addr)
        0x41, 10, // i32.const 10 (expected)
        0x41, 20, // i32.const 20 (replacement)
        0xFE, 0x48, 0x02, 0x00, // i32.atomic.rmw.cmpxchg align=2 offset=0
        0x0B,
    });
    try testing.expectEqual(@as(i32, 10), result);
}

test "interp: atomic.fence is a no-op" {
    // fence then return constant — no memory needed
    // 77 in signed LEB128 = 0xCD 0x00 (two bytes, since 77 > 63)
    const result = try runCode(&.{
        0xFE, 0x03, 0x00, // atomic.fence reserved=0
        0x41, 0xCD, 0x00, // i32.const 77
        0x0B,
    });
    try testing.expectEqual(@as(i32, 77), result);
}

test "interp: i32.load8_s with 0xFF" {
    // store 0xFF at addr 0, load8_s → -1
    const result = try runCodeWithMem(&.{
        0x41, 0, // i32.const 0 (addr)
        0x41, 0xFF, 0x01, // i32.const 255
        0x3A, 0x00, 0x00, // i32.store8 align=0 offset=0
        0x41, 0, // i32.const 0 (addr)
        0x2C, 0x00, 0x00, // i32.load8_s align=0 offset=0
        0x0B,
    });
    try testing.expectEqual(@as(i32, -1), result);
}

test "interp: i32.load8_u with 0xFF" {
    // store 0xFF at addr 0, load8_u → 255
    const result = try runCodeWithMem(&.{
        0x41, 0, // i32.const 0 (addr)
        0x41, 0xFF, 0x01, // i32.const 255
        0x3A, 0x00, 0x00, // i32.store8 align=0 offset=0
        0x41, 0, // i32.const 0 (addr)
        0x2D, 0x00, 0x00, // i32.load8_u align=0 offset=0
        0x0B,
    });
    try testing.expectEqual(@as(i32, 255), result);
}

test "interp: i32.store16/load16_s/load16_u roundtrip" {
    // store 0xFFFE (-2 as i16) at addr 0 via store16, then load16_s → -2, load16_u → 65534
    // Test load16_s:
    const result_s = try runCodeWithMem(&.{
        0x41, 0, // i32.const 0 (addr)
        0x41, 0xFE, 0xFF, 0x03, // i32.const 0xFFFE (65534)
        0x3B, 0x01, 0x00, // i32.store16 align=1 offset=0
        0x41, 0, // i32.const 0
        0x2E, 0x01, 0x00, // i32.load16_s align=1 offset=0
        0x0B,
    });
    try testing.expectEqual(@as(i32, -2), result_s);

    // Test load16_u:
    const result_u = try runCodeWithMem(&.{
        0x41, 0, // i32.const 0 (addr)
        0x41, 0xFE, 0xFF, 0x03, // i32.const 0xFFFE (65534)
        0x3B, 0x01, 0x00, // i32.store16 align=1 offset=0
        0x41, 0, // i32.const 0
        0x2F, 0x01, 0x00, // i32.load16_u align=1 offset=0
        0x0B,
    });
    try testing.expectEqual(@as(i32, 65534), result_u);
}

test "interp: i64.store/load roundtrip" {
    // i64.const -1 (0x7F LEB128); i64.store at addr 0; i64.load → -1
    const result = try runCodeWithMemI64(&.{
        0x41, 0, // i32.const 0 (addr)
        0x42, 0x7F, // i64.const -1
        0x37, 0x03, 0x00, // i64.store align=3 offset=0
        0x41, 0, // i32.const 0 (addr)
        0x29, 0x03, 0x00, // i64.load align=3 offset=0
        0x0B,
    });
    try testing.expectEqual(@as(i64, -1), result);
}

test "interp: f32.store/load roundtrip" {
    // f32.const 3.14; f32.store at addr 0; f32.load from addr 0
    // 3.14f ≈ 0x4048F5C3 LE: C3 F5 48 40
    const result = try runCodeWithMemF32(&.{
        0x41, 0, // i32.const 0 (addr)
        0x43, 0xC3, 0xF5, 0x48, 0x40, // f32.const 3.14
        0x38, 0x02, 0x00, // f32.store align=2 offset=0
        0x41, 0, // i32.const 0 (addr)
        0x2A, 0x02, 0x00, // f32.load align=2 offset=0
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f32, 3.14), result, 0.001);
}

test "interp: f64.store/load roundtrip" {
    // f64.const 2.718281828; f64.store; f64.load
    // 2.718281828 f64 LE: 0x4005BF0A8B125769 → 69 57 12 8B 0A BF 05 40
    const result = try runCodeWithMemF64(&.{
        0x41, 0, // i32.const 0 (addr)
        0x44, 0x69, 0x57, 0x12, 0x8B, 0x0A, 0xBF, 0x05, 0x40, // f64.const 2.718281828
        0x39, 0x03, 0x00, // f64.store align=3 offset=0
        0x41, 0, // i32.const 0 (addr)
        0x2B, 0x03, 0x00, // f64.load align=3 offset=0
        0x0B,
    });
    try testing.expectApproxEqAbs(@as(f64, 2.718281828), result, 0.000001);
}

test "interp: i64.store8/load8_u roundtrip" {
    // i64.const 0x1FF (511); i64.store8 at addr 0 → stores 0xFF; i64.load8_u → 255
    const result = try runCodeWithMemI64(&.{
        0x41, 0, // i32.const 0 (addr)
        0x42, 0xFF, 0x03, // i64.const 511
        0x3C, 0x00, 0x00, // i64.store8 align=0 offset=0
        0x41, 0, // i32.const 0 (addr)
        0x31, 0x00, 0x00, // i64.load8_u align=0 offset=0
        0x0B,
    });
    try testing.expectEqual(@as(i64, 255), result);
}
