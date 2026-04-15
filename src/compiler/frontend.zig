//! Wasm → IR Frontend
//!
//! Lowers WebAssembly bytecode functions into the compiler's SSA-form IR.

const std = @import("std");
const types = @import("../runtime/common/types.zig");
const ir = @import("ir/ir.zig");
const Opcode = @import("../runtime/interpreter/opcode.zig").Opcode;

pub const LowerError = error{
    OutOfMemory,
    InvalidBytecode,
    UnsupportedOpcode,
};

/// Lower an entire Wasm module into IR.
pub fn lowerModule(wasm_module: *const types.WasmModule, allocator: std.mem.Allocator) LowerError!ir.IrModule {
    var ir_module = ir.IrModule.init(allocator);
    errdefer ir_module.deinit();

    for (wasm_module.functions) |func| {
        const func_type = wasm_module.types[func.type_idx];
        const ir_func = try lowerFunction(&func, &func_type, allocator);
        _ = try ir_module.addFunction(ir_func);
    }

    return ir_module;
}

/// Lower a single Wasm function into IR.
fn lowerFunction(func: *const types.WasmFunction, func_type: *const types.FuncType, allocator: std.mem.Allocator) LowerError!ir.IrFunction {
    const param_count: u32 = @intCast(func_type.params.len);
    const result_count: u32 = @intCast(func_type.results.len);

    var total_locals: u32 = param_count;
    for (func.locals) |local| total_locals += local.count;

    var ir_func = ir.IrFunction.init(allocator, param_count, result_count, total_locals);
    errdefer ir_func.deinit();

    // Create entry block
    const entry_id = try ir_func.newBlock();

    // Simple value stack tracking (maps stack positions to vregs)
    var vreg_stack: std.ArrayList(ir.VReg) = .empty;
    defer vreg_stack.deinit(allocator);

    // Assign vregs for parameters (they're the first locals)
    var param_idx: u32 = 0;
    while (param_idx < param_count) : (param_idx += 1) {
        _ = ir_func.newVReg();
    }

    // ── Control flow block stack ──────────────────────────────────────
    const BlockFrame = struct {
        kind: enum { block, loop, @"if" },
        /// Block to jump to on `br` (end_block for block/if, header for loop).
        target_block: ir.BlockId,
        /// Continuation block after this construct ends.
        end_block: ir.BlockId,
        /// For if: the else block (if present).
        else_block: ?ir.BlockId,
        /// Value stack depth on entry (for multi-value cleanup).
        stack_depth: usize,
        /// Result arity of this block.
        result_arity: u32,
    };
    var block_stack: std.ArrayList(BlockFrame) = .empty;
    defer block_stack.deinit(allocator);

    // Track current block
    var current_block: ir.BlockId = entry_id;

    // Helper: read block type (-0x40 = void, else valtype)
    const readBlockType = struct {
        fn call(code: []const u8, ip_ptr: *usize) u32 {
            const byte = code[ip_ptr.*];
            ip_ptr.* += 1;
            if (byte == 0x40) return 0; // void
            return 1; // any valtype = 1 result
        }
    }.call;

    // Walk bytecode and emit IR instructions
    var ip: usize = 0;
    const code = func.code;

    while (ip < code.len) {
        const byte = code[ip];
        ip += 1;
        const op: Opcode = @enumFromInt(byte);

        switch (op) {
            .end => {
                if (block_stack.items.len == 0) {
                    // Function-level end: emit ret
                    const ret_val: ?ir.VReg = if (vreg_stack.items.len > 0) vreg_stack.pop().? else null;
                    try ir_func.getBlock(current_block).append(.{ .op = .{ .ret = ret_val } });
                    break;
                }
                // End of a block/loop/if: branch to continuation
                const frame = block_stack.pop().?;
                if (frame.result_arity > 0 and vreg_stack.items.len > frame.stack_depth) {
                    // Pass result value through
                }
                try ir_func.getBlock(current_block).append(.{ .op = .{ .br = frame.end_block } });
                current_block = frame.end_block;
            },
            .@"return" => {
                const ret_val: ?ir.VReg = if (vreg_stack.items.len > 0) vreg_stack.pop().? else null;
                try ir_func.getBlock(current_block).append(.{ .op = .{ .ret = ret_val } });
            },
            .@"unreachable" => {
                try ir_func.getBlock(current_block).append(.{ .op = .{ .@"unreachable" = {} } });
            },
            .nop => {},
            .block => {
                const arity = readBlockType(code, &ip);
                const end_block = try ir_func.newBlock();
                try block_stack.append(allocator, .{
                    .kind = .block,
                    .target_block = end_block, // br targets end
                    .end_block = end_block,
                    .else_block = null,
                    .stack_depth = vreg_stack.items.len,
                    .result_arity = arity,
                });
            },
            .loop => {
                const arity = readBlockType(code, &ip);
                const loop_header = try ir_func.newBlock();
                const end_block = try ir_func.newBlock();
                // Branch from current to loop header
                try ir_func.getBlock(current_block).append(.{ .op = .{ .br = loop_header } });
                current_block = loop_header;
                try block_stack.append(allocator, .{
                    .kind = .loop,
                    .target_block = loop_header, // br targets header (loop back)
                    .end_block = end_block,
                    .else_block = null,
                    .stack_depth = vreg_stack.items.len,
                    .result_arity = arity,
                });
            },
            .@"if" => {
                const arity = readBlockType(code, &ip);
                const cond = vreg_stack.pop().?;
                const then_block = try ir_func.newBlock();
                const else_block = try ir_func.newBlock();
                const end_block = try ir_func.newBlock();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .br_if = .{
                    .cond = cond,
                    .then_block = then_block,
                    .else_block = else_block,
                } } });
                current_block = then_block;
                try block_stack.append(allocator, .{
                    .kind = .@"if",
                    .target_block = end_block, // br targets end
                    .end_block = end_block,
                    .else_block = else_block,
                    .stack_depth = vreg_stack.items.len,
                    .result_arity = arity,
                });
            },
            .@"else" => {
                const frame = &block_stack.items[block_stack.items.len - 1];
                // End of then branch: jump to end
                try ir_func.getBlock(current_block).append(.{ .op = .{ .br = frame.end_block } });
                current_block = frame.else_block orelse return error.InvalidBytecode;
                frame.else_block = null; // consumed
            },
            .br => {
                const depth = readU32(code, &ip);
                if (depth < block_stack.items.len) {
                    const target_frame = block_stack.items[block_stack.items.len - 1 - depth];
                    try ir_func.getBlock(current_block).append(.{ .op = .{ .br = target_frame.target_block } });
                }
            },
            .br_if => {
                const depth = readU32(code, &ip);
                const cond = vreg_stack.pop().?;
                if (depth < block_stack.items.len) {
                    const target_frame = block_stack.items[block_stack.items.len - 1 - depth];
                    const fallthrough = try ir_func.newBlock();
                    try ir_func.getBlock(current_block).append(.{ .op = .{ .br_if = .{
                        .cond = cond,
                        .then_block = target_frame.target_block,
                        .else_block = fallthrough,
                    } } });
                    current_block = fallthrough;
                }
            },
            .call => {
                const func_idx = readU32(code, &ip);
                // For now, emit a call with current stack args.
                // Full ABI arg passing will be added later.
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .call = .{
                    .func_idx = func_idx,
                    .args = &.{},
                } }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .i32_const => {
                const val = readI32(code, &ip);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .iconst_32 = val }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .i64_const => {
                const val = readI64(code, &ip);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .iconst_64 = val }, .dest = dest, .type = .i64 });
                try vreg_stack.append(allocator, dest);
            },
            .local_get => {
                const idx = readU32(code, &ip);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .local_get = idx }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .local_set => {
                const idx = readU32(code, &ip);
                const val = vreg_stack.pop().?;
                try ir_func.getBlock(current_block).append(.{ .op = .{ .local_set = .{ .idx = idx, .val = val } } });
            },
            .local_tee => {
                const idx = readU32(code, &ip);
                const val = vreg_stack.items[vreg_stack.items.len - 1]; // peek, don't pop
                try ir_func.getBlock(current_block).append(.{ .op = .{ .local_set = .{ .idx = idx, .val = val } } });
            },
            .i32_add, .i32_sub, .i32_mul, .i32_and, .i32_or, .i32_xor => {
                const rhs = vreg_stack.pop().?;
                const lhs = vreg_stack.pop().?;
                const dest = ir_func.newVReg();
                const bin = ir.Inst.BinOp{ .lhs = lhs, .rhs = rhs };
                const ir_op: ir.Inst.Op = switch (op) {
                    .i32_add => .{ .add = bin },
                    .i32_sub => .{ .sub = bin },
                    .i32_mul => .{ .mul = bin },
                    .i32_and => .{ .@"and" = bin },
                    .i32_or => .{ .@"or" = bin },
                    .i32_xor => .{ .xor = bin },
                    else => unreachable,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .i32_shl, .i32_shr_s, .i32_shr_u, .i32_rotl, .i32_rotr => {
                const rhs = vreg_stack.pop().?;
                const lhs = vreg_stack.pop().?;
                const dest = ir_func.newVReg();
                const bin = ir.Inst.BinOp{ .lhs = lhs, .rhs = rhs };
                const ir_op: ir.Inst.Op = switch (op) {
                    .i32_shl => .{ .shl = bin },
                    .i32_shr_s => .{ .shr_s = bin },
                    .i32_shr_u => .{ .shr_u = bin },
                    .i32_rotl => .{ .rotl = bin },
                    .i32_rotr => .{ .rotr = bin },
                    else => unreachable,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .i32_div_s, .i32_div_u, .i32_rem_s, .i32_rem_u => {
                const rhs = vreg_stack.pop().?;
                const lhs = vreg_stack.pop().?;
                const dest = ir_func.newVReg();
                const bin = ir.Inst.BinOp{ .lhs = lhs, .rhs = rhs };
                const ir_op: ir.Inst.Op = switch (op) {
                    .i32_div_s => .{ .div_s = bin },
                    .i32_div_u => .{ .div_u = bin },
                    .i32_rem_s => .{ .rem_s = bin },
                    .i32_rem_u => .{ .rem_u = bin },
                    else => unreachable,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .i32_eq, .i32_ne, .i32_lt_s, .i32_lt_u, .i32_gt_s, .i32_gt_u, .i32_le_s, .i32_le_u, .i32_ge_s, .i32_ge_u => {
                const rhs = vreg_stack.pop().?;
                const lhs = vreg_stack.pop().?;
                const dest = ir_func.newVReg();
                const bin = ir.Inst.BinOp{ .lhs = lhs, .rhs = rhs };
                const ir_op: ir.Inst.Op = switch (op) {
                    .i32_eq => .{ .eq = bin },
                    .i32_ne => .{ .ne = bin },
                    .i32_lt_s => .{ .lt_s = bin },
                    .i32_lt_u => .{ .lt_u = bin },
                    .i32_gt_s => .{ .gt_s = bin },
                    .i32_gt_u => .{ .gt_u = bin },
                    .i32_le_s => .{ .le_s = bin },
                    .i32_le_u => .{ .le_u = bin },
                    .i32_ge_s => .{ .ge_s = bin },
                    .i32_ge_u => .{ .ge_u = bin },
                    else => unreachable,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .i32_eqz => {
                const val = vreg_stack.pop().?;
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .eqz = val }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .i32_clz => {
                const val = vreg_stack.pop().?;
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .clz = val }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .drop => _ = vreg_stack.pop(),
            .select => {
                const cond = vreg_stack.pop().?;
                const if_false = vreg_stack.pop().?;
                const if_true = vreg_stack.pop().?;
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .select = .{ .cond = cond, .if_true = if_true, .if_false = if_false } }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .global_get => {
                const idx = readU32(code, &ip);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .global_get = idx }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .global_set => {
                const idx = readU32(code, &ip);
                const val = vreg_stack.pop().?;
                try ir_func.getBlock(current_block).append(.{ .op = .{ .global_set = .{ .idx = idx, .val = val } } });
            },
            .i32_load => {
                _ = readU32(code, &ip); // alignment
                const offset = readU32(code, &ip);
                const base = vreg_stack.pop().?;
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .load = .{ .base = base, .offset = offset, .size = 4 } }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .i32_store => {
                _ = readU32(code, &ip); // alignment
                const offset = readU32(code, &ip);
                const val = vreg_stack.pop().?;
                const base = vreg_stack.pop().?;
                try ir_func.getBlock(current_block).append(.{ .op = .{ .store = .{ .base = base, .offset = offset, .size = 4, .val = val } } });
            },
            .br_table => {
                // Read count + targets, skip for now (emit unreachable)
                const count = readU32(code, &ip);
                var i: u32 = 0;
                while (i <= count) : (i += 1) _ = readU32(code, &ip);
                _ = vreg_stack.pop(); // condition
                try ir_func.getBlock(current_block).append(.{ .op = .{ .@"unreachable" = {} } });
            },
            .call_indirect => {
                _ = readU32(code, &ip); // type index
                _ = readU32(code, &ip); // table index
                _ = vreg_stack.pop(); // table element index
                try ir_func.getBlock(current_block).append(.{ .op = .{ .@"unreachable" = {} } });
            },
            .memory_size => {
                _ = readU32(code, &ip); // memory index (always 0)
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .iconst_32 = 0 }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .memory_grow => {
                _ = readU32(code, &ip); // memory index
                _ = vreg_stack.pop(); // pages
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .iconst_32 = -1 }, .dest = dest, .type = .i32 }); // fail
                try vreg_stack.append(allocator, dest);
            },
            else => return error.UnsupportedOpcode,
        }
    }

    return ir_func;
}

// ── LEB128 decoders ────────────────────────────────────────────────

fn readU32(code: []const u8, ip_ptr: *usize) u32 {
    var result: u32 = 0;
    var shift: u32 = 0;
    while (true) {
        const byte = code[ip_ptr.*];
        ip_ptr.* += 1;
        result |= @as(u32, byte & 0x7F) << @as(u5, @intCast(shift));
        if (byte & 0x80 == 0) break;
        shift += 7;
        if (shift >= 35) break;
    }
    return result;
}

fn readI32(code: []const u8, ip_ptr: *usize) i32 {
    var result: u32 = 0;
    var shift: u32 = 0;
    var byte: u8 = 0;
    while (true) {
        byte = code[ip_ptr.*];
        ip_ptr.* += 1;
        result |= @as(u32, byte & 0x7F) << @as(u5, @intCast(shift));
        if (byte & 0x80 == 0) break;
        shift += 7;
        if (shift >= 35) break;
    }
    // Sign-extend if the sign bit of the last byte is set
    if (shift < 32 and (byte & 0x40) != 0) {
        result |= @as(u32, 0xFFFFFFFF) << @as(u5, @intCast(shift));
    }
    return @bitCast(result);
}

fn readI64(code: []const u8, ip_ptr: *usize) i64 {
    var result: u64 = 0;
    var shift: u32 = 0;
    var byte: u8 = 0;
    while (true) {
        byte = code[ip_ptr.*];
        ip_ptr.* += 1;
        result |= @as(u64, byte & 0x7F) << @as(u6, @intCast(shift));
        if (byte & 0x80 == 0) break;
        shift += 7;
        if (shift >= 63) break;
    }
    // Sign-extend if the sign bit of the last byte is set
    if (shift < 64 and (byte & 0x40) != 0) {
        result |= @as(u64, 0xFFFFFFFFFFFFFFFF) << @as(u6, @intCast(shift));
    }
    return @bitCast(result);
}

// ── Tests ──────────────────────────────────────────────────────────

test "lower i32.const 42; end" {
    const allocator = std.testing.allocator;

    const func_type = types.FuncType{ .params = &.{}, .results = &.{.i32} };
    const func = types.WasmFunction{
        .type_idx = 0,
        .func_type = func_type,
        .local_count = 0,
        .locals = &.{},
        // i32.const 42, end
        .code = &[_]u8{ 0x41, 0x2A, 0x0B },
    };
    const wasm_module = types.WasmModule{
        .types = &[_]types.FuncType{func_type},
        .functions = &[_]types.WasmFunction{func},
    };

    var ir_module = try lowerModule(&wasm_module, allocator);
    defer ir_module.deinit();

    try std.testing.expectEqual(@as(usize, 1), ir_module.functions.items.len);

    const ir_func = &ir_module.functions.items[0];
    try std.testing.expectEqual(@as(usize, 1), ir_func.blocks.items.len);

    const insts = ir_func.blocks.items[0].instructions.items;
    try std.testing.expectEqual(@as(usize, 2), insts.len);

    // First instruction: iconst_32 = 42
    try std.testing.expectEqual(@as(i32, 42), insts[0].op.iconst_32);
    try std.testing.expect(insts[0].dest != null);

    // Second instruction: ret with value
    try std.testing.expect(insts[1].op.ret != null);
}

test "lower local.get 0; local.get 1; i32.add; end" {
    const allocator = std.testing.allocator;

    const func_type = types.FuncType{
        .params = &.{ .i32, .i32 },
        .results = &.{.i32},
    };
    const func = types.WasmFunction{
        .type_idx = 0,
        .func_type = func_type,
        .local_count = 2,
        .locals = &.{},
        // local.get 0, local.get 1, i32.add, end
        .code = &[_]u8{ 0x20, 0x00, 0x20, 0x01, 0x6A, 0x0B },
    };
    const wasm_module = types.WasmModule{
        .types = &[_]types.FuncType{func_type},
        .functions = &[_]types.WasmFunction{func},
    };

    var ir_module = try lowerModule(&wasm_module, allocator);
    defer ir_module.deinit();

    const ir_func = &ir_module.functions.items[0];
    const insts = ir_func.blocks.items[0].instructions.items;

    // local_get, local_get, add, ret = 4 instructions
    try std.testing.expectEqual(@as(usize, 4), insts.len);

    // Verify local_get instructions
    try std.testing.expectEqual(@as(u32, 0), insts[0].op.local_get);
    try std.testing.expectEqual(@as(u32, 1), insts[1].op.local_get);

    // Verify add
    const add_op = insts[2].op.add;
    try std.testing.expect(add_op.lhs != add_op.rhs);

    // Verify ret has a value
    try std.testing.expect(insts[3].op.ret != null);
}

test "lower unreachable" {
    const allocator = std.testing.allocator;

    const func_type = types.FuncType{ .params = &.{}, .results = &.{} };
    const func = types.WasmFunction{
        .type_idx = 0,
        .func_type = func_type,
        .local_count = 0,
        .locals = &.{},
        // unreachable
        .code = &[_]u8{0x00},
    };
    const wasm_module = types.WasmModule{
        .types = &[_]types.FuncType{func_type},
        .functions = &[_]types.WasmFunction{func},
    };

    var ir_module = try lowerModule(&wasm_module, allocator);
    defer ir_module.deinit();

    const ir_func = &ir_module.functions.items[0];
    const insts = ir_func.blocks.items[0].instructions.items;
    try std.testing.expectEqual(@as(usize, 1), insts.len);

    // Should be unreachable op
    try std.testing.expectEqual(ir.Inst.Op{ .@"unreachable" = {} }, insts[0].op);
}

test "lower empty module" {
    const allocator = std.testing.allocator;

    const wasm_module = types.WasmModule{};

    var ir_module = try lowerModule(&wasm_module, allocator);
    defer ir_module.deinit();

    try std.testing.expectEqual(@as(usize, 0), ir_module.functions.items.len);
}

test "readI32: positive value" {
    const code = [_]u8{ 0xE5, 0x8E, 0x26 }; // 624485
    var ip: usize = 0;
    const val = readI32(&code, &ip);
    try std.testing.expectEqual(@as(i32, 624485), val);
    try std.testing.expectEqual(@as(usize, 3), ip);
}

test "readI32: negative value" {
    const code = [_]u8{0x7F}; // -1
    var ip: usize = 0;
    const val = readI32(&code, &ip);
    try std.testing.expectEqual(@as(i32, -1), val);
}

test "readI32: single byte positive" {
    const code = [_]u8{0x2A}; // 42
    var ip: usize = 0;
    const val = readI32(&code, &ip);
    try std.testing.expectEqual(@as(i32, 42), val);
}

test "readU32: multi-byte" {
    const code = [_]u8{ 0x80, 0x01 }; // 128
    var ip: usize = 0;
    const val = readU32(&code, &ip);
    try std.testing.expectEqual(@as(u32, 128), val);
}

test "readI64: simple value" {
    const code = [_]u8{0x2A}; // 42
    var ip: usize = 0;
    const val = readI64(&code, &ip);
    try std.testing.expectEqual(@as(i64, 42), val);
}
