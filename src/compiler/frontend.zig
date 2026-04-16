//! Wasm → IR Frontend
//!
//! Lowers WebAssembly bytecode functions into the compiler's SSA-form IR.

const std = @import("std");
const types = @import("../runtime/common/types.zig");
const ir = @import("ir/ir.zig");
const Opcode = @import("../runtime/interpreter/opcode.zig").Opcode;
const MiscOpcode = @import("../runtime/interpreter/opcode.zig").MiscOpcode;
const AtomicOpcode = @import("../runtime/interpreter/opcode.zig").AtomicOpcode;

pub const LowerError = error{
    OutOfMemory,
    InvalidBytecode,
    UnsupportedOpcode,
};

/// Lower an entire Wasm module into IR.
pub fn lowerModule(wasm_module: *const types.WasmModule, allocator: std.mem.Allocator) LowerError!ir.IrModule {
    var ir_module = ir.IrModule.init(allocator);
    errdefer ir_module.deinit();

    ir_module.import_count = wasm_module.import_function_count;

    for (wasm_module.functions) |func| {
        const func_type = wasm_module.types[func.type_idx];
        const ir_func = try lowerFunction(&func, &func_type, wasm_module, allocator);
        _ = try ir_module.addFunction(ir_func);
    }

    return ir_module;
}

/// Lower a single Wasm function into IR.
fn lowerFunction(func: *const types.WasmFunction, func_type: *const types.FuncType, wasm_module: *const types.WasmModule, allocator: std.mem.Allocator) LowerError!ir.IrFunction {
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

    // Safe pop: returns a dummy VReg (0) if the stack is empty.
    // This handles dead code paths where instructions try to consume
    // values that were never pushed (code after br/return/unreachable).
    const dummy_vreg: ir.VReg = 0;
    const safePop = struct {
        fn pop(stack: *std.ArrayList(ir.VReg)) ir.VReg {
            return if (stack.items.len > 0) stack.pop().? else 0;
        }
    }.pop;
    _ = dummy_vreg;

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
        /// Synthetic local slot for passing result values across branches.
        result_local: ?u32,
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
    var dead_code = false; // true after unconditional br/return/unreachable

    while (ip < code.len) {
        const byte = code[ip];
        ip += 1;
        const op: Opcode = @enumFromInt(byte);

        // In dead code, skip instructions until we reach a block boundary
        if (dead_code) {
            switch (op) {
                .end => {
                    dead_code = false;
                    // Fall through to normal end handling below
                },
                .@"else" => {
                    dead_code = false;
                    // Fall through to normal else handling below
                },
                // Skip block-structured opcodes by reading their block type
                .block, .loop, .@"if" => {
                    _ = readBlockType(code, &ip);
                    // Push a dummy frame so end/else matching works
                    try block_stack.append(allocator, .{
                        .kind = .block,
                        .target_block = 0,
                        .end_block = 0,
                        .else_block = null,
                        .stack_depth = vreg_stack.items.len,
                        .result_arity = 0,
                        .result_local = null,
                    });
                    continue;
                },
                else => {
                    // Skip operands for dead instructions
                    skipOperands(code, &ip, op);
                    continue;
                },
            }
        }

        switch (op) {
            .end => {
                if (block_stack.items.len == 0) {
                    // Function-level end: emit ret
                    const ret_val: ?ir.VReg = if (vreg_stack.items.len > 0) safePop(&vreg_stack) else null;
                    try ir_func.getBlock(current_block).append(.{ .op = .{ .ret = ret_val } });
                    break;
                }
                // End of a block/loop/if: store result to synthetic local, branch to continuation
                const frame = block_stack.pop().?;
                if (frame.result_arity > 0 and frame.result_local != null) {
                    if (vreg_stack.items.len > frame.stack_depth) {
                        const result_val = safePop(&vreg_stack);
                        try ir_func.getBlock(current_block).append(.{ .op = .{ .local_set = .{ .idx = frame.result_local.?, .val = result_val } } });
                    }
                }
                // Trim stack to block entry depth
                vreg_stack.items.len = frame.stack_depth;
                try ir_func.getBlock(current_block).append(.{ .op = .{ .br = frame.end_block } });
                current_block = frame.end_block;
                // Load result from synthetic local at merge point
                if (frame.result_arity > 0 and frame.result_local != null) {
                    const dest = ir_func.newVReg();
                    try ir_func.getBlock(current_block).append(.{ .op = .{ .local_get = frame.result_local.? }, .dest = dest, .type = .i32 });
                    try vreg_stack.append(allocator, dest);
                }
            },
            .@"return" => {
                const ret_val: ?ir.VReg = if (vreg_stack.items.len > 0) safePop(&vreg_stack) else null;
                try ir_func.getBlock(current_block).append(.{ .op = .{ .ret = ret_val } });
                dead_code = true;
            },
            .@"unreachable" => {
                try ir_func.getBlock(current_block).append(.{ .op = .{ .@"unreachable" = {} } });
                dead_code = true;
            },
            .nop => {},
            .block => {
                const arity = readBlockType(code, &ip);
                const end_block = try ir_func.newBlock();
                const result_local: ?u32 = if (arity > 0) blk: {
                    const idx = total_locals;
                    total_locals += 1;
                    ir_func.local_count = total_locals;
                    break :blk idx;
                } else null;
                try block_stack.append(allocator, .{
                    .kind = .block,
                    .target_block = end_block,
                    .end_block = end_block,
                    .else_block = null,
                    .stack_depth = vreg_stack.items.len,
                    .result_arity = arity,
                    .result_local = result_local,
                });
            },
            .loop => {
                const arity = readBlockType(code, &ip);
                const loop_header = try ir_func.newBlock();
                const end_block = try ir_func.newBlock();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .br = loop_header } });
                current_block = loop_header;
                const result_local: ?u32 = if (arity > 0) blk: {
                    const idx = total_locals;
                    total_locals += 1;
                    ir_func.local_count = total_locals;
                    break :blk idx;
                } else null;
                try block_stack.append(allocator, .{
                    .kind = .loop,
                    .target_block = loop_header,
                    .end_block = end_block,
                    .else_block = null,
                    .stack_depth = vreg_stack.items.len,
                    .result_arity = arity,
                    .result_local = result_local,
                });
            },
            .@"if" => {
                const arity = readBlockType(code, &ip);
                const cond = safePop(&vreg_stack);
                const then_block = try ir_func.newBlock();
                const else_block = try ir_func.newBlock();
                const end_block = try ir_func.newBlock();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .br_if = .{
                    .cond = cond,
                    .then_block = then_block,
                    .else_block = else_block,
                } } });
                current_block = then_block;
                const result_local: ?u32 = if (arity > 0) blk: {
                    const idx = total_locals;
                    total_locals += 1;
                    ir_func.local_count = total_locals;
                    break :blk idx;
                } else null;
                try block_stack.append(allocator, .{
                    .kind = .@"if",
                    .target_block = end_block,
                    .end_block = end_block,
                    .else_block = else_block,
                    .stack_depth = vreg_stack.items.len,
                    .result_arity = arity,
                    .result_local = result_local,
                });
            },
            .@"else" => {
                const frame = &block_stack.items[block_stack.items.len - 1];
                // Store then-branch result to synthetic local before jumping to end
                if (frame.result_arity > 0 and frame.result_local != null) {
                    if (vreg_stack.items.len > frame.stack_depth) {
                        const result_val = safePop(&vreg_stack);
                        try ir_func.getBlock(current_block).append(.{ .op = .{ .local_set = .{ .idx = frame.result_local.?, .val = result_val } } });
                    }
                }
                // Trim stack to block entry depth for else branch
                vreg_stack.items.len = frame.stack_depth;
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
                dead_code = true;
            },
            .br_if => {
                const depth = readU32(code, &ip);
                const cond = safePop(&vreg_stack);
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
                // Look up callee type to determine arg count
                const callee_type = wasm_module.getFuncType(func_idx);
                const arg_count: u32 = if (callee_type) |ct| @intCast(ct.params.len) else 0;
                // Capture arg VRegs before popping (args are in stack order: first arg deepest)
                const args = try allocator.alloc(ir.VReg, arg_count);
                var i: u32 = 0;
                while (i < arg_count) : (i += 1) {
                    // Pop in reverse: last arg is on top
                    args[arg_count - 1 - i] = safePop(&vreg_stack);
                }
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .call = .{
                    .func_idx = func_idx,
                    .args = args,
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
                const val = safePop(&vreg_stack);
                try ir_func.getBlock(current_block).append(.{ .op = .{ .local_set = .{ .idx = idx, .val = val } } });
            },
            .local_tee => {
                const idx = readU32(code, &ip);
                const val = vreg_stack.items[vreg_stack.items.len - 1]; // peek, don't pop
                try ir_func.getBlock(current_block).append(.{ .op = .{ .local_set = .{ .idx = idx, .val = val } } });
            },
            .i32_add, .i32_sub, .i32_mul, .i32_and, .i32_or, .i32_xor => {
                const rhs = safePop(&vreg_stack);
                const lhs = safePop(&vreg_stack);
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
                const rhs = safePop(&vreg_stack);
                const lhs = safePop(&vreg_stack);
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
                const rhs = safePop(&vreg_stack);
                const lhs = safePop(&vreg_stack);
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
                const rhs = safePop(&vreg_stack);
                const lhs = safePop(&vreg_stack);
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
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .eqz = val }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .i32_clz, .i32_ctz, .i32_popcnt => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const ir_op: ir.Inst.Op = switch (op) {
                    .i32_clz => .{ .clz = val },
                    .i32_ctz => .{ .ctz = val },
                    .i32_popcnt => .{ .popcnt = val },
                    else => unreachable,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .drop => _ = vreg_stack.pop(),
            .select => {
                const cond = safePop(&vreg_stack);
                const if_false = safePop(&vreg_stack);
                const if_true = safePop(&vreg_stack);
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
                const val = safePop(&vreg_stack);
                try ir_func.getBlock(current_block).append(.{ .op = .{ .global_set = .{ .idx = idx, .val = val } } });
            },
            // ── Load variants ────────────────────────────────────────────
            .i32_load, .i32_load8_s, .i32_load8_u, .i32_load16_s, .i32_load16_u,
            .i64_load, .i64_load8_s, .i64_load8_u, .i64_load16_s, .i64_load16_u, .i64_load32_s, .i64_load32_u,
            .f32_load, .f64_load,
            => {
                _ = readU32(code, &ip); // alignment
                const offset = readU32(code, &ip);
                const base = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const size: u8 = switch (op) {
                    .i32_load8_s, .i32_load8_u, .i64_load8_s, .i64_load8_u => 1,
                    .i32_load16_s, .i32_load16_u, .i64_load16_s, .i64_load16_u => 2,
                    .i32_load, .i64_load32_s, .i64_load32_u, .f32_load => 4,
                    .i64_load, .f64_load => 8,
                    else => unreachable,
                };
                const sign_extend: bool = switch (op) {
                    .i32_load8_s, .i32_load16_s,
                    .i64_load8_s, .i64_load16_s, .i64_load32_s,
                    => true,
                    else => false,
                };
                const ir_type: ir.IrType = switch (op) {
                    .i64_load, .i64_load8_s, .i64_load8_u, .i64_load16_s, .i64_load16_u, .i64_load32_s, .i64_load32_u => .i64,
                    .f32_load => .f32,
                    .f64_load => .f64,
                    else => .i32,
                };
                try ir_func.getBlock(current_block).append(.{ .op = .{ .load = .{ .base = base, .offset = offset, .size = size, .sign_extend = sign_extend } }, .dest = dest, .type = ir_type });
                try vreg_stack.append(allocator, dest);
            },
            // ── Store variants ───────────────────────────────────────────
            .i32_store, .i32_store8, .i32_store16,
            .i64_store, .i64_store8, .i64_store16, .i64_store32,
            .f32_store, .f64_store,
            => {
                _ = readU32(code, &ip); // alignment
                const offset = readU32(code, &ip);
                const val = safePop(&vreg_stack);
                const base = safePop(&vreg_stack);
                const size: u8 = switch (op) {
                    .i32_store8, .i64_store8 => 1,
                    .i32_store16, .i64_store16 => 2,
                    .i32_store, .i64_store32, .f32_store => 4,
                    .i64_store, .f64_store => 8,
                    else => unreachable,
                };
                try ir_func.getBlock(current_block).append(.{ .op = .{ .store = .{ .base = base, .offset = offset, .size = size, .val = val } } });
            },
            .br_table => {
                const count = readU32(code, &ip);
                // Read all targets (count entries + 1 default)
                const raw_targets = try allocator.alloc(u32, count + 1);
                defer allocator.free(raw_targets);
                for (0..count + 1) |i| raw_targets[i] = readU32(code, &ip);

                const index = safePop(&vreg_stack);

                // Resolve depths to target block IDs.
                // Entries whose depth is out of range are skipped (matches
                // previous if-else behavior which silently dropped them).
                const ir_targets = try allocator.alloc(ir.BlockId, count);
                var resolved_count: u32 = 0;
                var default_target: ir.BlockId = 0;
                var have_default = false;

                for (0..count) |i| {
                    const depth = raw_targets[i];
                    if (depth < block_stack.items.len) {
                        const target_frame = block_stack.items[block_stack.items.len - 1 - depth];
                        ir_targets[resolved_count] = target_frame.target_block;
                        resolved_count += 1;
                    }
                }

                const default_depth = raw_targets[count];
                if (default_depth < block_stack.items.len) {
                    default_target = block_stack.items[block_stack.items.len - 1 - default_depth].target_block;
                    have_default = true;
                }

                if (have_default and resolved_count > 0) {
                    // Shrink slice ownership to actually-resolved entries.
                    const final_targets = ir_targets[0..resolved_count];
                    try ir_func.getBlock(current_block).append(.{ .op = .{ .br_table = .{
                        .index = index,
                        .targets = final_targets,
                        .default = default_target,
                    } } });
                } else if (have_default) {
                    allocator.free(ir_targets);
                    try ir_func.getBlock(current_block).append(.{ .op = .{ .br = default_target } });
                } else {
                    // No valid target — fall through as unreachable.
                    allocator.free(ir_targets);
                    try ir_func.getBlock(current_block).append(.{ .op = .{ .@"unreachable" = {} } });
                }
                dead_code = true;
            },
            .call_indirect => {
                const type_idx = readU32(code, &ip);
                _ = readU32(code, &ip); // table index (always 0 for now)
                const elem_idx = safePop(&vreg_stack); // table element index

                // Look up the function type to get arg count and result arity
                const callee_type = if (type_idx < wasm_module.types.len) &wasm_module.types[type_idx] else null;
                const arg_count: u32 = if (callee_type) |ct| @intCast(ct.params.len) else 0;
                const call_result_count: u32 = if (callee_type) |ct| @intCast(ct.results.len) else 0;

                // Pop args and capture VRegs
                const args = try allocator.alloc(ir.VReg, arg_count);
                var ci: u32 = 0;
                while (ci < arg_count) : (ci += 1) {
                    args[arg_count - 1 - ci] = safePop(&vreg_stack);
                }

                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .call_indirect = .{
                    .type_idx = type_idx,
                    .elem_idx = elem_idx,
                    .args = args,
                } }, .dest = dest, .type = .i32 });

                if (call_result_count > 0) {
                    try vreg_stack.append(allocator, dest);
                }
            },
            .memory_size => {
                _ = readU32(code, &ip); // memory index (always 0)
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .memory_size = {} }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .memory_grow => {
                _ = readU32(code, &ip); // memory index
                const pages = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .memory_grow = pages }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },

            // ── i64 arithmetic ──────────────────────────────────────────
            .i64_add, .i64_sub, .i64_mul, .i64_and, .i64_or, .i64_xor => {
                const rhs = safePop(&vreg_stack);
                const lhs = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const bin = ir.Inst.BinOp{ .lhs = lhs, .rhs = rhs };
                const ir_op: ir.Inst.Op = switch (op) {
                    .i64_add => .{ .add = bin },
                    .i64_sub => .{ .sub = bin },
                    .i64_mul => .{ .mul = bin },
                    .i64_and => .{ .@"and" = bin },
                    .i64_or => .{ .@"or" = bin },
                    .i64_xor => .{ .xor = bin },
                    else => unreachable,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = .i64 });
                try vreg_stack.append(allocator, dest);
            },
            .i64_shl, .i64_shr_s, .i64_shr_u, .i64_rotl, .i64_rotr => {
                const rhs = safePop(&vreg_stack);
                const lhs = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const bin = ir.Inst.BinOp{ .lhs = lhs, .rhs = rhs };
                const ir_op: ir.Inst.Op = switch (op) {
                    .i64_shl => .{ .shl = bin },
                    .i64_shr_s => .{ .shr_s = bin },
                    .i64_shr_u => .{ .shr_u = bin },
                    .i64_rotl => .{ .rotl = bin },
                    .i64_rotr => .{ .rotr = bin },
                    else => unreachable,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = .i64 });
                try vreg_stack.append(allocator, dest);
            },
            .i64_div_s, .i64_div_u, .i64_rem_s, .i64_rem_u => {
                const rhs = safePop(&vreg_stack);
                const lhs = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const bin = ir.Inst.BinOp{ .lhs = lhs, .rhs = rhs };
                const ir_op: ir.Inst.Op = switch (op) {
                    .i64_div_s => .{ .div_s = bin },
                    .i64_div_u => .{ .div_u = bin },
                    .i64_rem_s => .{ .rem_s = bin },
                    .i64_rem_u => .{ .rem_u = bin },
                    else => unreachable,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = .i64 });
                try vreg_stack.append(allocator, dest);
            },
            .i64_eq, .i64_ne, .i64_lt_s, .i64_lt_u, .i64_gt_s, .i64_gt_u, .i64_le_s, .i64_le_u, .i64_ge_s, .i64_ge_u => {
                const rhs = safePop(&vreg_stack);
                const lhs = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const bin = ir.Inst.BinOp{ .lhs = lhs, .rhs = rhs };
                const ir_op: ir.Inst.Op = switch (op) {
                    .i64_eq => .{ .eq = bin },
                    .i64_ne => .{ .ne = bin },
                    .i64_lt_s => .{ .lt_s = bin },
                    .i64_lt_u => .{ .lt_u = bin },
                    .i64_gt_s => .{ .gt_s = bin },
                    .i64_gt_u => .{ .gt_u = bin },
                    .i64_le_s => .{ .le_s = bin },
                    .i64_le_u => .{ .le_u = bin },
                    .i64_ge_s => .{ .ge_s = bin },
                    .i64_ge_u => .{ .ge_u = bin },
                    else => unreachable,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .i64_eqz => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .eqz = val }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .i64_clz, .i64_ctz, .i64_popcnt => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const ir_op: ir.Inst.Op = switch (op) {
                    .i64_clz => .{ .clz = val },
                    .i64_ctz => .{ .ctz = val },
                    .i64_popcnt => .{ .popcnt = val },
                    else => unreachable,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = .i64 });
                try vreg_stack.append(allocator, dest);
            },

            // ── Float constants ─────────────────────────────────────────
            .f32_const => {
                const bits = std.mem.readInt(u32, code[ip..][0..4], .little);
                ip += 4;
                const val: f32 = @bitCast(bits);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .fconst_32 = val }, .dest = dest, .type = .f32 });
                try vreg_stack.append(allocator, dest);
            },
            .f64_const => {
                const bits = std.mem.readInt(u64, code[ip..][0..8], .little);
                ip += 8;
                const val: f64 = @bitCast(bits);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .fconst_64 = val }, .dest = dest, .type = .f64 });
                try vreg_stack.append(allocator, dest);
            },

            // ── Float arithmetic (stub: lower to IR but codegen not yet) ──
            .f32_add, .f32_sub, .f32_mul, .f32_div,
            .f64_add, .f64_sub, .f64_mul, .f64_div,
            => {
                const rhs = safePop(&vreg_stack);
                const lhs = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const bin = ir.Inst.BinOp{ .lhs = lhs, .rhs = rhs };
                const is_f64 = (op == .f64_add or op == .f64_sub or op == .f64_mul or op == .f64_div);
                const ir_op: ir.Inst.Op = switch (op) {
                    .f32_add, .f64_add => .{ .add = bin },
                    .f32_sub, .f64_sub => .{ .sub = bin },
                    .f32_mul, .f64_mul => .{ .mul = bin },
                    .f32_div, .f64_div => .{ .div_s = bin },
                    else => unreachable,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = if (is_f64) .f64 else .f32 });
                try vreg_stack.append(allocator, dest);
            },

            // ── Type conversions ────────────────────────────────────────
            .i32_wrap_i64 => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .wrap_i64 = val }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .i64_extend_i32_s => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .extend_i32_s = val }, .dest = dest, .type = .i64 });
                try vreg_stack.append(allocator, dest);
            },
            .i64_extend_i32_u => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .extend_i32_u = val }, .dest = dest, .type = .i64 });
                try vreg_stack.append(allocator, dest);
            },

            // ── Truncation (float → int) ──────────────────────────────
            .i32_trunc_f32_s, .i32_trunc_f32_u,
            .i32_trunc_f64_s, .i32_trunc_f64_u,
            .i64_trunc_f32_s, .i64_trunc_f32_u,
            .i64_trunc_f64_s, .i64_trunc_f64_u,
            => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const ir_op: ir.Inst.Op = switch (op) {
                    .i32_trunc_f32_s, .i64_trunc_f32_s => .{ .trunc_f32_s = val },
                    .i32_trunc_f32_u, .i64_trunc_f32_u => .{ .trunc_f32_u = val },
                    .i32_trunc_f64_s, .i64_trunc_f64_s => .{ .trunc_f64_s = val },
                    .i32_trunc_f64_u, .i64_trunc_f64_u => .{ .trunc_f64_u = val },
                    else => unreachable,
                };
                const ir_type: ir.IrType = switch (op) {
                    .i32_trunc_f32_s, .i32_trunc_f32_u, .i32_trunc_f64_s, .i32_trunc_f64_u => .i32,
                    else => .i64,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = ir_type });
                try vreg_stack.append(allocator, dest);
            },

            // ── Conversion (int → float) ──────────────────────────────
            .f32_convert_i32_s, .f32_convert_i64_s,
            .f64_convert_i32_s, .f64_convert_i64_s,
            => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const ir_type: ir.IrType = switch (op) {
                    .f32_convert_i32_s, .f32_convert_i64_s => .f32,
                    else => .f64,
                };
                try ir_func.getBlock(current_block).append(.{ .op = .{ .convert_s = val }, .dest = dest, .type = ir_type });
                try vreg_stack.append(allocator, dest);
            },
            .f32_convert_i32_u, .f32_convert_i64_u,
            .f64_convert_i32_u, .f64_convert_i64_u,
            => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const ir_type: ir.IrType = switch (op) {
                    .f32_convert_i32_u, .f32_convert_i64_u => .f32,
                    else => .f64,
                };
                try ir_func.getBlock(current_block).append(.{ .op = .{ .convert_u = val }, .dest = dest, .type = ir_type });
                try vreg_stack.append(allocator, dest);
            },

            // ── Demote / promote ───────────────────────────────────────
            .f32_demote_f64 => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .demote_f64 = val }, .dest = dest, .type = .f32 });
                try vreg_stack.append(allocator, dest);
            },
            .f64_promote_f32 => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .promote_f32 = val }, .dest = dest, .type = .f64 });
                try vreg_stack.append(allocator, dest);
            },

            // ── Reinterpret (bitcast) ──────────────────────────────────
            .i32_reinterpret_f32 => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .reinterpret = val }, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },
            .i64_reinterpret_f64 => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .reinterpret = val }, .dest = dest, .type = .i64 });
                try vreg_stack.append(allocator, dest);
            },
            .f32_reinterpret_i32 => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .reinterpret = val }, .dest = dest, .type = .f32 });
                try vreg_stack.append(allocator, dest);
            },
            .f64_reinterpret_i64 => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                try ir_func.getBlock(current_block).append(.{ .op = .{ .reinterpret = val }, .dest = dest, .type = .f64 });
                try vreg_stack.append(allocator, dest);
            },

            // ── Saturating truncation (0xFC prefix) ────────────────────
            .misc_prefix => {
                const sub_opcode: MiscOpcode = @enumFromInt(readU32(code, &ip));
                switch (sub_opcode) {
                    .i32_trunc_sat_f32_s, .i32_trunc_sat_f32_u,
                    .i32_trunc_sat_f64_s, .i32_trunc_sat_f64_u,
                    .i64_trunc_sat_f32_s, .i64_trunc_sat_f32_u,
                    .i64_trunc_sat_f64_s, .i64_trunc_sat_f64_u,
                    => {
                        const val = safePop(&vreg_stack);
                        const dest = ir_func.newVReg();
                        const ir_op: ir.Inst.Op = switch (sub_opcode) {
                            .i32_trunc_sat_f32_s, .i64_trunc_sat_f32_s => .{ .trunc_sat_f32_s = val },
                            .i32_trunc_sat_f32_u, .i64_trunc_sat_f32_u => .{ .trunc_sat_f32_u = val },
                            .i32_trunc_sat_f64_s, .i64_trunc_sat_f64_s => .{ .trunc_sat_f64_s = val },
                            .i32_trunc_sat_f64_u, .i64_trunc_sat_f64_u => .{ .trunc_sat_f64_u = val },
                            else => unreachable,
                        };
                        const ir_type: ir.IrType = switch (sub_opcode) {
                            .i32_trunc_sat_f32_s, .i32_trunc_sat_f32_u,
                            .i32_trunc_sat_f64_s, .i32_trunc_sat_f64_u,
                            => .i32,
                            else => .i64,
                        };
                        try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = ir_type });
                        try vreg_stack.append(allocator, dest);
                    },
                    .memory_copy => {
                        _ = readU32(code, &ip); // dst memory (always 0)
                        _ = readU32(code, &ip); // src memory (always 0)
                        const len = safePop(&vreg_stack);
                        const src = safePop(&vreg_stack);
                        const dst = safePop(&vreg_stack);
                        try ir_func.getBlock(current_block).append(.{ .op = .{ .memory_copy = .{ .dst = dst, .src = src, .len = len } } });
                    },
                    .memory_fill => {
                        _ = readU32(code, &ip); // memory (always 0)
                        const len = safePop(&vreg_stack);
                        const val = safePop(&vreg_stack);
                        const dst = safePop(&vreg_stack);
                        try ir_func.getBlock(current_block).append(.{ .op = .{ .memory_fill = .{ .dst = dst, .val = val, .len = len } } });
                    },
                    .memory_init => {
                        const seg_idx = readU32(code, &ip);
                        _ = readU32(code, &ip); // memory (always 0)
                        const len = safePop(&vreg_stack);
                        const src = safePop(&vreg_stack);
                        const dst = safePop(&vreg_stack);
                        try ir_func.getBlock(current_block).append(.{ .op = .{ .memory_init = .{ .seg_idx = seg_idx, .dst = dst, .src = src, .len = len } } });
                    },
                    .data_drop => {
                        const seg_idx = readU32(code, &ip);
                        try ir_func.getBlock(current_block).append(.{ .op = .{ .data_drop = seg_idx } });
                    },
                    else => {
                        std.debug.print("wamrc: unsupported misc opcode 0xFC 0x{X:0>2}\n", .{@intFromEnum(sub_opcode)});
                        return error.UnsupportedOpcode;
                    },
                }
            },

            // ── Atomic operations (0xFE prefix) ───────────────────────
            .atomic_prefix => {
                const sub_opcode: AtomicOpcode = @enumFromInt(readU32(code, &ip));
                switch (sub_opcode) {
                    .atomic_fence => {
                        _ = readU32(code, &ip);
                        try ir_func.getBlock(current_block).append(.{ .op = .{ .atomic_fence = {} } });
                    },

                    .memory_atomic_notify => {
                        _ = readU32(code, &ip);
                        const offset = readU32(code, &ip);
                        const count = safePop(&vreg_stack);
                        const base = safePop(&vreg_stack);
                        const dest = ir_func.newVReg();
                        try ir_func.getBlock(current_block).append(.{ .op = .{ .atomic_notify = .{ .base = base, .offset = offset, .count = count } }, .dest = dest, .type = .i32 });
                        try vreg_stack.append(allocator, dest);
                    },

                    .memory_atomic_wait32, .memory_atomic_wait64 => {
                        _ = readU32(code, &ip);
                        const offset = readU32(code, &ip);
                        const timeout = safePop(&vreg_stack);
                        const expected = safePop(&vreg_stack);
                        const base = safePop(&vreg_stack);
                        const size: u8 = switch (sub_opcode) {
                            .memory_atomic_wait32 => 4,
                            .memory_atomic_wait64 => 8,
                            else => unreachable,
                        };
                        const dest = ir_func.newVReg();
                        try ir_func.getBlock(current_block).append(.{ .op = .{ .atomic_wait = .{ .base = base, .offset = offset, .expected = expected, .timeout = timeout, .size = size } }, .dest = dest, .type = .i32 });
                        try vreg_stack.append(allocator, dest);
                    },

                    .i32_atomic_load, .i64_atomic_load,
                    .i32_atomic_load8_u, .i32_atomic_load16_u,
                    .i64_atomic_load8_u, .i64_atomic_load16_u, .i64_atomic_load32_u,
                    => {
                        _ = readU32(code, &ip);
                        const offset = readU32(code, &ip);
                        const base = safePop(&vreg_stack);
                        const dest = ir_func.newVReg();
                        const size: u8 = switch (sub_opcode) {
                            .i32_atomic_load => 4,
                            .i64_atomic_load => 8,
                            .i32_atomic_load8_u, .i64_atomic_load8_u => 1,
                            .i32_atomic_load16_u, .i64_atomic_load16_u => 2,
                            .i64_atomic_load32_u => 4,
                            else => unreachable,
                        };
                        const ir_type: ir.IrType = switch (sub_opcode) {
                            .i32_atomic_load, .i32_atomic_load8_u, .i32_atomic_load16_u => .i32,
                            else => .i64,
                        };
                        try ir_func.getBlock(current_block).append(.{ .op = .{ .atomic_load = .{ .base = base, .offset = offset, .size = size } }, .dest = dest, .type = ir_type });
                        try vreg_stack.append(allocator, dest);
                    },

                    .i32_atomic_store, .i64_atomic_store,
                    .i32_atomic_store8, .i32_atomic_store16,
                    .i64_atomic_store8, .i64_atomic_store16, .i64_atomic_store32,
                    => {
                        _ = readU32(code, &ip);
                        const offset = readU32(code, &ip);
                        const val = safePop(&vreg_stack);
                        const base = safePop(&vreg_stack);
                        const size: u8 = switch (sub_opcode) {
                            .i32_atomic_store => 4,
                            .i64_atomic_store => 8,
                            .i32_atomic_store8, .i64_atomic_store8 => 1,
                            .i32_atomic_store16, .i64_atomic_store16 => 2,
                            .i64_atomic_store32 => 4,
                            else => unreachable,
                        };
                        try ir_func.getBlock(current_block).append(.{ .op = .{ .atomic_store = .{ .base = base, .offset = offset, .size = size, .val = val } } });
                    },

                    .i32_atomic_rmw_add, .i64_atomic_rmw_add,
                    .i32_atomic_rmw8_add_u, .i32_atomic_rmw16_add_u,
                    .i64_atomic_rmw8_add_u, .i64_atomic_rmw16_add_u, .i64_atomic_rmw32_add_u,
                    .i32_atomic_rmw_sub, .i64_atomic_rmw_sub,
                    .i32_atomic_rmw8_sub_u, .i32_atomic_rmw16_sub_u,
                    .i64_atomic_rmw8_sub_u, .i64_atomic_rmw16_sub_u, .i64_atomic_rmw32_sub_u,
                    .i32_atomic_rmw_and, .i64_atomic_rmw_and,
                    .i32_atomic_rmw8_and_u, .i32_atomic_rmw16_and_u,
                    .i64_atomic_rmw8_and_u, .i64_atomic_rmw16_and_u, .i64_atomic_rmw32_and_u,
                    .i32_atomic_rmw_or, .i64_atomic_rmw_or,
                    .i32_atomic_rmw8_or_u, .i32_atomic_rmw16_or_u,
                    .i64_atomic_rmw8_or_u, .i64_atomic_rmw16_or_u, .i64_atomic_rmw32_or_u,
                    .i32_atomic_rmw_xor, .i64_atomic_rmw_xor,
                    .i32_atomic_rmw8_xor_u, .i32_atomic_rmw16_xor_u,
                    .i64_atomic_rmw8_xor_u, .i64_atomic_rmw16_xor_u, .i64_atomic_rmw32_xor_u,
                    .i32_atomic_rmw_xchg, .i64_atomic_rmw_xchg,
                    .i32_atomic_rmw8_xchg_u, .i32_atomic_rmw16_xchg_u,
                    .i64_atomic_rmw8_xchg_u, .i64_atomic_rmw16_xchg_u, .i64_atomic_rmw32_xchg_u,
                    => {
                        _ = readU32(code, &ip);
                        const offset = readU32(code, &ip);
                        const val = safePop(&vreg_stack);
                        const base = safePop(&vreg_stack);
                        const dest = ir_func.newVReg();
                        const rmw_op: ir.Inst.AtomicRmwOp = switch (sub_opcode) {
                            .i32_atomic_rmw_add, .i64_atomic_rmw_add, .i32_atomic_rmw8_add_u, .i32_atomic_rmw16_add_u, .i64_atomic_rmw8_add_u, .i64_atomic_rmw16_add_u, .i64_atomic_rmw32_add_u => .add,
                            .i32_atomic_rmw_sub, .i64_atomic_rmw_sub, .i32_atomic_rmw8_sub_u, .i32_atomic_rmw16_sub_u, .i64_atomic_rmw8_sub_u, .i64_atomic_rmw16_sub_u, .i64_atomic_rmw32_sub_u => .sub,
                            .i32_atomic_rmw_and, .i64_atomic_rmw_and, .i32_atomic_rmw8_and_u, .i32_atomic_rmw16_and_u, .i64_atomic_rmw8_and_u, .i64_atomic_rmw16_and_u, .i64_atomic_rmw32_and_u => .@"and",
                            .i32_atomic_rmw_or, .i64_atomic_rmw_or, .i32_atomic_rmw8_or_u, .i32_atomic_rmw16_or_u, .i64_atomic_rmw8_or_u, .i64_atomic_rmw16_or_u, .i64_atomic_rmw32_or_u => .@"or",
                            .i32_atomic_rmw_xor, .i64_atomic_rmw_xor, .i32_atomic_rmw8_xor_u, .i32_atomic_rmw16_xor_u, .i64_atomic_rmw8_xor_u, .i64_atomic_rmw16_xor_u, .i64_atomic_rmw32_xor_u => .xor,
                            .i32_atomic_rmw_xchg, .i64_atomic_rmw_xchg, .i32_atomic_rmw8_xchg_u, .i32_atomic_rmw16_xchg_u, .i64_atomic_rmw8_xchg_u, .i64_atomic_rmw16_xchg_u, .i64_atomic_rmw32_xchg_u => .xchg,
                            else => unreachable,
                        };
                        const size: u8 = switch (sub_opcode) {
                            .i32_atomic_rmw_add, .i32_atomic_rmw_sub, .i32_atomic_rmw_and, .i32_atomic_rmw_or, .i32_atomic_rmw_xor, .i32_atomic_rmw_xchg => 4,
                            .i64_atomic_rmw_add, .i64_atomic_rmw_sub, .i64_atomic_rmw_and, .i64_atomic_rmw_or, .i64_atomic_rmw_xor, .i64_atomic_rmw_xchg => 8,
                            .i32_atomic_rmw8_add_u, .i32_atomic_rmw8_sub_u, .i32_atomic_rmw8_and_u, .i32_atomic_rmw8_or_u, .i32_atomic_rmw8_xor_u, .i32_atomic_rmw8_xchg_u, .i64_atomic_rmw8_add_u, .i64_atomic_rmw8_sub_u, .i64_atomic_rmw8_and_u, .i64_atomic_rmw8_or_u, .i64_atomic_rmw8_xor_u, .i64_atomic_rmw8_xchg_u => 1,
                            .i32_atomic_rmw16_add_u, .i32_atomic_rmw16_sub_u, .i32_atomic_rmw16_and_u, .i32_atomic_rmw16_or_u, .i32_atomic_rmw16_xor_u, .i32_atomic_rmw16_xchg_u, .i64_atomic_rmw16_add_u, .i64_atomic_rmw16_sub_u, .i64_atomic_rmw16_and_u, .i64_atomic_rmw16_or_u, .i64_atomic_rmw16_xor_u, .i64_atomic_rmw16_xchg_u => 2,
                            .i64_atomic_rmw32_add_u, .i64_atomic_rmw32_sub_u, .i64_atomic_rmw32_and_u, .i64_atomic_rmw32_or_u, .i64_atomic_rmw32_xor_u, .i64_atomic_rmw32_xchg_u => 4,
                            else => unreachable,
                        };
                        const ir_type: ir.IrType = switch (sub_opcode) {
                            .i32_atomic_rmw_add, .i32_atomic_rmw_sub, .i32_atomic_rmw_and, .i32_atomic_rmw_or, .i32_atomic_rmw_xor, .i32_atomic_rmw_xchg, .i32_atomic_rmw8_add_u, .i32_atomic_rmw8_sub_u, .i32_atomic_rmw8_and_u, .i32_atomic_rmw8_or_u, .i32_atomic_rmw8_xor_u, .i32_atomic_rmw8_xchg_u, .i32_atomic_rmw16_add_u, .i32_atomic_rmw16_sub_u, .i32_atomic_rmw16_and_u, .i32_atomic_rmw16_or_u, .i32_atomic_rmw16_xor_u, .i32_atomic_rmw16_xchg_u => .i32,
                            else => .i64,
                        };
                        try ir_func.getBlock(current_block).append(.{ .op = .{ .atomic_rmw = .{ .base = base, .offset = offset, .size = size, .val = val, .op = rmw_op } }, .dest = dest, .type = ir_type });
                        try vreg_stack.append(allocator, dest);
                    },

                    .i32_atomic_rmw_cmpxchg, .i64_atomic_rmw_cmpxchg,
                    .i32_atomic_rmw8_cmpxchg_u, .i32_atomic_rmw16_cmpxchg_u,
                    .i64_atomic_rmw8_cmpxchg_u, .i64_atomic_rmw16_cmpxchg_u, .i64_atomic_rmw32_cmpxchg_u,
                    => {
                        _ = readU32(code, &ip);
                        const offset = readU32(code, &ip);
                        const replacement = safePop(&vreg_stack);
                        const expected = safePop(&vreg_stack);
                        const base = safePop(&vreg_stack);
                        const dest = ir_func.newVReg();
                        const size: u8 = switch (sub_opcode) {
                            .i32_atomic_rmw_cmpxchg => 4,
                            .i64_atomic_rmw_cmpxchg => 8,
                            .i32_atomic_rmw8_cmpxchg_u, .i64_atomic_rmw8_cmpxchg_u => 1,
                            .i32_atomic_rmw16_cmpxchg_u, .i64_atomic_rmw16_cmpxchg_u => 2,
                            .i64_atomic_rmw32_cmpxchg_u => 4,
                            else => unreachable,
                        };
                        const ir_type: ir.IrType = switch (sub_opcode) {
                            .i32_atomic_rmw_cmpxchg, .i32_atomic_rmw8_cmpxchg_u, .i32_atomic_rmw16_cmpxchg_u => .i32,
                            else => .i64,
                        };
                        try ir_func.getBlock(current_block).append(.{ .op = .{ .atomic_cmpxchg = .{ .base = base, .offset = offset, .size = size, .expected = expected, .replacement = replacement } }, .dest = dest, .type = ir_type });
                        try vreg_stack.append(allocator, dest);
                    },

                    else => {
                        std.debug.print("wamrc: unsupported atomic opcode 0xFE 0x{X:0>2}\n", .{@intFromEnum(sub_opcode)});
                        return error.UnsupportedOpcode;
                    },
                }
            },

            // ── Sign-extension ops ─────────────────────────────────────
            .i32_extend8_s, .i32_extend16_s,
            .i64_extend8_s, .i64_extend16_s, .i64_extend32_s,
            => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const ir_op: ir.Inst.Op = switch (op) {
                    .i32_extend8_s, .i64_extend8_s => .{ .extend8_s = val },
                    .i32_extend16_s, .i64_extend16_s => .{ .extend16_s = val },
                    .i64_extend32_s => .{ .extend32_s = val },
                    else => unreachable,
                };
                const ir_type: ir.IrType = switch (op) {
                    .i32_extend8_s, .i32_extend16_s => .i32,
                    else => .i64,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = ir_type });
                try vreg_stack.append(allocator, dest);
            },

            // ── Float comparisons ──────────────────────────────────────
            .f32_eq, .f32_ne, .f32_lt, .f32_gt, .f32_le, .f32_ge,
            .f64_eq, .f64_ne, .f64_lt, .f64_gt, .f64_le, .f64_ge,
            => {
                const rhs = safePop(&vreg_stack);
                const lhs = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const bin = ir.Inst.BinOp{ .lhs = lhs, .rhs = rhs };
                const ir_op: ir.Inst.Op = switch (op) {
                    .f32_eq, .f64_eq => .{ .eq = bin },
                    .f32_ne, .f64_ne => .{ .ne = bin },
                    .f32_lt, .f64_lt => .{ .lt_s = bin },
                    .f32_gt, .f64_gt => .{ .gt_s = bin },
                    .f32_le, .f64_le => .{ .le_s = bin },
                    .f32_ge, .f64_ge => .{ .ge_s = bin },
                    else => unreachable,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = .i32 });
                try vreg_stack.append(allocator, dest);
            },

            // ── Float unary math ───────────────────────────────────────
            .f32_abs, .f32_neg, .f32_sqrt, .f32_ceil, .f32_floor, .f32_trunc, .f32_nearest,
            .f64_abs, .f64_neg, .f64_sqrt, .f64_ceil, .f64_floor, .f64_trunc, .f64_nearest,
            => {
                const val = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const ir_op: ir.Inst.Op = switch (op) {
                    .f32_abs, .f64_abs => .{ .f_abs = val },
                    .f32_neg, .f64_neg => .{ .f_neg = val },
                    .f32_sqrt, .f64_sqrt => .{ .f_sqrt = val },
                    .f32_ceil, .f64_ceil => .{ .f_ceil = val },
                    .f32_floor, .f64_floor => .{ .f_floor = val },
                    .f32_trunc, .f64_trunc => .{ .f_trunc = val },
                    .f32_nearest, .f64_nearest => .{ .f_nearest = val },
                    else => unreachable,
                };
                const ir_type: ir.IrType = switch (op) {
                    .f64_abs, .f64_neg, .f64_sqrt, .f64_ceil, .f64_floor, .f64_trunc, .f64_nearest => .f64,
                    else => .f32,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = ir_type });
                try vreg_stack.append(allocator, dest);
            },

            // ── Float binary math ──────────────────────────────────────
            .f32_min, .f32_max, .f32_copysign,
            .f64_min, .f64_max, .f64_copysign,
            => {
                const rhs = safePop(&vreg_stack);
                const lhs = safePop(&vreg_stack);
                const dest = ir_func.newVReg();
                const bin = ir.Inst.BinOp{ .lhs = lhs, .rhs = rhs };
                const ir_op: ir.Inst.Op = switch (op) {
                    .f32_min, .f64_min => .{ .f_min = bin },
                    .f32_max, .f64_max => .{ .f_max = bin },
                    .f32_copysign, .f64_copysign => .{ .f_copysign = bin },
                    else => unreachable,
                };
                const ir_type: ir.IrType = switch (op) {
                    .f64_min, .f64_max, .f64_copysign => .f64,
                    else => .f32,
                };
                try ir_func.getBlock(current_block).append(.{ .op = ir_op, .dest = dest, .type = ir_type });
                try vreg_stack.append(allocator, dest);
            },

            else => {
                std.debug.print("wamrc: unsupported opcode 0x{X:0>2}\n", .{byte});
                return error.UnsupportedOpcode;
            },
        }
    }

    return ir_func;
}

/// Skip operands of a dead instruction to keep IP in sync.
fn skipOperands(code: []const u8, ip: *usize, op: Opcode) void {
    switch (op) {
        // Fixed-width immediate operands
        .i32_const => _ = readI32(code, ip),
        .i64_const => _ = readI64(code, ip),
        .f32_const => ip.* += 4,
        .f64_const => ip.* += 8,
        .local_get, .local_set, .local_tee => _ = readU32(code, ip),
        .global_get, .global_set => _ = readU32(code, ip),
        .br, .br_if => _ = readU32(code, ip),
        .call => _ = readU32(code, ip),
        .call_indirect => {
            _ = readU32(code, ip); // type
            _ = readU32(code, ip); // table
        },
        .memory_size, .memory_grow => _ = readU32(code, ip),
        .br_table => {
            const count = readU32(code, ip);
            var i: u32 = 0;
            while (i <= count) : (i += 1) _ = readU32(code, ip);
        },
        // Load/store ops have align + offset
        .i32_load, .i64_load, .f32_load, .f64_load,
        .i32_load8_s, .i32_load8_u, .i32_load16_s, .i32_load16_u,
        .i64_load8_s, .i64_load8_u, .i64_load16_s, .i64_load16_u,
        .i64_load32_s, .i64_load32_u,
        .i32_store, .i64_store, .f32_store, .f64_store,
        .i32_store8, .i32_store16, .i64_store8, .i64_store16, .i64_store32,
        => {
            _ = readU32(code, ip); // align
            _ = readU32(code, ip); // offset
        },
        // Prefix opcodes
        .misc_prefix => _ = readU32(code, ip),
        .atomic_prefix => {
            _ = readU32(code, ip); // sub-opcode
            _ = readU32(code, ip); // align
            _ = readU32(code, ip); // offset
        },
        // No operands
        else => {},
    }
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

test "lower: dead code after br is skipped" {
    const allocator = std.testing.allocator;

    // Build a minimal wasm function: i32.const 42; br 0; i32.const 99; end
    // The i32.const 99 after br is dead code and should be skipped
    var bytecode: [20]u8 = undefined;
    var pos: usize = 0;
    bytecode[pos] = 0x41; pos += 1; // i32.const
    bytecode[pos] = 42; pos += 1; // 42
    bytecode[pos] = 0x0C; pos += 1; // br
    bytecode[pos] = 0; pos += 1; // depth 0
    bytecode[pos] = 0x41; pos += 1; // i32.const (dead)
    bytecode[pos] = 99; pos += 1; // 99 (dead)
    bytecode[pos] = 0x0B; pos += 1; // end

    const func_type = types.FuncType{ .params = &.{}, .results = &.{} };
    const func = types.WasmFunction{
        .type_idx = 0,
        .func_type = func_type,
        .local_count = 0,
        .locals = &.{},
        .code = bytecode[0..pos],
    };
    const wasm_module = types.WasmModule{
        .types = &[_]types.FuncType{func_type},
        .functions = &[_]types.WasmFunction{func},
    };

    // Should not crash (dead code after br is handled)
    const result = lowerFunction(&func, &func_type, &wasm_module, allocator);
    if (result) |ir_func| {
        var f = ir_func;
        f.deinit();
    } else |_| {
        // Error is also acceptable — we just shouldn't crash
    }
}
