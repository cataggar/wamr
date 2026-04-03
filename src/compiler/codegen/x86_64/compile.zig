//! x86-64 IR Compiler
//!
//! Walks IR functions and emits x86-64 machine code via CodeBuffer.
//! Uses a simple register allocator that maps VRegs to physical registers.

const std = @import("std");
const ir = @import("../../ir/ir.zig");
const emit = @import("emit.zig");

/// Simple VReg → physical register mapping.
/// Uses a linear scan over a fixed set of scratch registers.
/// Spills to stack when registers are exhausted.
const RegMap = struct {
    entries: std.AutoHashMap(ir.VReg, Location),
    reg_used: [scratch_regs.len]bool = [_]bool{false} ** scratch_regs.len,
    next_stack_offset: u32 = 0,

    const Location = union(enum) {
        reg: emit.Reg,
        stack: u32,
    };

    // Caller-saved scratch registers (excludes rsp/rbp)
    const scratch_regs = [_]emit.Reg{ .rax, .rcx, .rdx, .rsi, .rdi, .r8, .r9, .r10, .r11 };

    fn init(allocator: std.mem.Allocator) RegMap {
        return .{
            .entries = std.AutoHashMap(ir.VReg, Location).init(allocator),
        };
    }

    fn deinit(self: *RegMap) void {
        self.entries.deinit();
    }

    fn assign(self: *RegMap, vreg: ir.VReg) !Location {
        for (scratch_regs, 0..) |r, i| {
            if (!self.reg_used[i]) {
                self.reg_used[i] = true;
                const loc = Location{ .reg = r };
                try self.entries.put(vreg, loc);
                return loc;
            }
        }
        // Spill to stack
        const offset = self.next_stack_offset;
        self.next_stack_offset += 8;
        const loc = Location{ .stack = offset };
        try self.entries.put(vreg, loc);
        return loc;
    }

    fn get(self: *const RegMap, vreg: ir.VReg) ?Location {
        return self.entries.get(vreg);
    }
};

/// Compile an IR function to x86-64 machine code.
pub fn compileFunction(func: *const ir.IrFunction, allocator: std.mem.Allocator) ![]u8 {
    var code = emit.CodeBuffer.init(allocator);
    errdefer code.deinit();

    var reg_map = RegMap.init(allocator);
    defer reg_map.deinit();

    const frame_size: u32 = func.local_count * 8 + 256;
    try code.emitPrologue(frame_size);

    var last_was_ret = false;
    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            last_was_ret = isRet(inst.op);
            try compileInst(&code, inst, &reg_map);
        }
    }

    // Add epilogue if the function didn't end with a ret instruction
    if (!last_was_ret) {
        try code.emitEpilogue();
    }

    return code.bytes.toOwnedSlice(allocator);
}

fn isRet(op: ir.Inst.Op) bool {
    return switch (op) {
        .ret => true,
        else => false,
    };
}

fn compileInst(code: *emit.CodeBuffer, inst: ir.Inst, reg_map: *RegMap) !void {
    switch (inst.op) {
        .iconst_32 => |val| {
            const dest = inst.dest orelse return;
            const loc = try reg_map.assign(dest);
            switch (loc) {
                .reg => |r| try code.movRegImm32(r, val),
                .stack => {},
            }
        },
        .add => |bin| try emitBinOp(code, inst, bin, reg_map, .add),
        .sub => |bin| try emitBinOp(code, inst, bin, reg_map, .sub),
        .mul => |bin| try emitBinOp(code, inst, bin, reg_map, .mul),
        .@"and" => |bin| try emitBinOp(code, inst, bin, reg_map, .@"and"),
        .@"or" => |bin| try emitBinOp(code, inst, bin, reg_map, .@"or"),
        .xor => |bin| try emitBinOp(code, inst, bin, reg_map, .xor),
        .ret => |maybe_val| {
            if (maybe_val) |val| {
                if (reg_map.get(val)) |loc| {
                    switch (loc) {
                        .reg => |r| {
                            if (r != .rax) try code.movRegReg(.rax, r);
                        },
                        .stack => {},
                    }
                }
            }
            try code.emitEpilogue();
        },
        .@"unreachable" => try code.int3(),
        else => {},
    }
}

const BinOpKind = enum { add, sub, mul, @"and", @"or", xor };

fn emitBinOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    kind: BinOpKind,
) !void {
    const dest = inst.dest orelse return;
    const lhs_loc = reg_map.get(bin.lhs) orelse return;
    const rhs_loc = reg_map.get(bin.rhs) orelse return;
    const dest_loc = try reg_map.assign(dest);

    const dest_reg = switch (dest_loc) {
        .reg => |r| r,
        .stack => return,
    };
    const lhs_reg = switch (lhs_loc) {
        .reg => |r| r,
        .stack => return,
    };
    const rhs_reg = switch (rhs_loc) {
        .reg => |r| r,
        .stack => return,
    };

    try code.movRegReg(dest_reg, lhs_reg);
    switch (kind) {
        .add => try code.addRegReg(dest_reg, rhs_reg),
        .sub => try code.subRegReg(dest_reg, rhs_reg),
        .mul => try code.imulRegReg(dest_reg, rhs_reg),
        .@"and" => try code.andRegReg(dest_reg, rhs_reg),
        .@"or" => try code.orRegReg(dest_reg, rhs_reg),
        .xor => try code.xorRegReg(dest_reg, rhs_reg),
    }
}

/// Result of compiling an IR module.
pub const CompileResult = struct {
    code: []u8,
    offsets: []u32,
};

/// Compile all functions in an IR module to x86-64 machine code.
pub fn compileModule(ir_module: *const ir.IrModule, allocator: std.mem.Allocator) !CompileResult {
    var all_code: std.ArrayList(u8) = .empty;
    errdefer all_code.deinit(allocator);
    var offsets: std.ArrayList(u32) = .empty;
    errdefer offsets.deinit(allocator);

    for (ir_module.functions.items) |func| {
        try offsets.append(allocator, @intCast(all_code.items.len));
        const func_code = try compileFunction(&func, allocator);
        defer allocator.free(func_code);
        try all_code.appendSlice(allocator, func_code);
    }

    return .{
        .code = try all_code.toOwnedSlice(allocator),
        .offsets = try offsets.toOwnedSlice(allocator),
    };
}

// ── Tests ──────────────────────────────────────────────────────────

test "compileFunction: iconst_32 + ret produces mov and epilogue" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Should have prologue + mov + epilogue
    try std.testing.expect(code.len > 10);
    // Verify the immediate 42 (0x2A) appears in the code
    var found_42 = false;
    for (code) |b| {
        if (b == 0x2A) {
            found_42 = true;
            break;
        }
    }
    try std.testing.expect(found_42);
    // Last byte should be ret (0xC3)
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunction: two iconsts + add + ret produces reasonable code" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Prologue + 2 mov-imm + mov+add + mov-to-rax + epilogue
    try std.testing.expect(code.len > 15);
    try std.testing.expect(code.len < 200);
    // Last byte should be ret
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileModule: two functions have correct offsets" {
    const allocator = std.testing.allocator;
    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    _ = try f1.newBlock();
    const v0 = f1.newVReg();
    try f1.getBlock(0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try f1.getBlock(0).append(.{ .op = .{ .ret = v0 } });
    _ = try ir_module.addFunction(f1);

    var f2 = ir.IrFunction.init(allocator, 0, 1, 0);
    _ = try f2.newBlock();
    const v1 = f2.newVReg();
    try f2.getBlock(0).append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1 });
    try f2.getBlock(0).append(.{ .op = .{ .ret = v1 } });
    _ = try ir_module.addFunction(f2);

    const result = try compileModule(&ir_module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    try std.testing.expectEqual(@as(usize, 2), result.offsets.len);
    try std.testing.expectEqual(@as(u32, 0), result.offsets[0]);
    try std.testing.expect(result.offsets[1] > 0);
}

test "compileFunction: empty function produces prologue and epilogue" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    _ = try func.newBlock();

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Should have prologue + epilogue
    try std.testing.expect(code.len > 0);
    // Last byte should be ret (0xC3)
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}
