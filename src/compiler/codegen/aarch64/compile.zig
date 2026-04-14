//! AArch64 IR Compiler
//!
//! Walks IR functions and emits AArch64 machine code via CodeBuffer.
//! Uses the same pattern as the x86-64 backend with a linear-scan
//! register allocator mapping VRegs to physical registers.

const std = @import("std");
const ir = @import("../../ir/ir.zig");
const emit = @import("emit.zig");

/// Simple VReg → physical register mapping.
const RegMap = struct {
    entries: std.AutoHashMap(ir.VReg, Location),
    reg_used: [scratch_regs.len]bool = [_]bool{false} ** scratch_regs.len,
    next_stack_offset: u32 = 0,

    const Location = union(enum) {
        reg: emit.Reg,
        stack: u32, // offset from FP
    };

    // Caller-saved scratch registers (AAPCS64: X0–X15 are caller-saved)
    // Exclude X16/X17 (IP0/IP1) and X18 (platform register)
    const scratch_regs = [_]emit.Reg{
        .x0, .x1, .x2, .x3, .x4, .x5, .x6, .x7,
        .x8, .x9, .x10, .x11, .x12, .x13, .x14, .x15,
    };

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

/// Compile an IR function to AArch64 machine code.
pub fn compileFunction(func: *const ir.IrFunction, allocator: std.mem.Allocator) ![]u8 {
    var code = emit.CodeBuffer.init(allocator);
    errdefer code.deinit();

    var reg_map = RegMap.init(allocator);
    defer reg_map.deinit();

    // Frame size: aligned to 16 bytes (AArch64 SP alignment requirement)
    const raw_frame = func.local_count * 8 + 256;
    const frame_size: u32 = (raw_frame + 15) & ~@as(u32, 15);
    try code.emitPrologue(frame_size);

    var last_was_ret = false;
    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            last_was_ret = isRet(inst.op);
            try compileInst(&code, inst, &reg_map, frame_size);
        }
    }

    if (!last_was_ret) {
        try code.emitEpilogue(frame_size);
    }

    return code.bytes.toOwnedSlice(allocator);
}

fn isRet(op: ir.Inst.Op) bool {
    return switch (op) {
        .ret => true,
        else => false,
    };
}

fn compileInst(code: *emit.CodeBuffer, inst: ir.Inst, reg_map: *RegMap, frame_size: u32) !void {
    switch (inst.op) {
        .iconst_32 => |val| {
            const dest = inst.dest orelse return;
            const loc = try reg_map.assign(dest);
            switch (loc) {
                .reg => |r| try code.movImm32(r, val),
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
                            if (r != .x0) try code.movRegReg(.x0, r);
                        },
                        .stack => {},
                    }
                }
            }
            try code.emitEpilogue(frame_size);
        },
        .@"unreachable" => try code.brk(0),
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

    const dest_reg = switch (dest_loc) { .reg => |r| r, .stack => return };
    const lhs_reg = switch (lhs_loc) { .reg => |r| r, .stack => return };
    const rhs_reg = switch (rhs_loc) { .reg => |r| r, .stack => return };

    switch (kind) {
        .add => try code.addRegReg(dest_reg, lhs_reg, rhs_reg),
        .sub => try code.subRegReg(dest_reg, lhs_reg, rhs_reg),
        .mul => try code.mulRegReg(dest_reg, lhs_reg, rhs_reg),
        .@"and" => try code.andRegReg(dest_reg, lhs_reg, rhs_reg),
        .@"or" => try code.orrRegReg(dest_reg, lhs_reg, rhs_reg),
        .xor => try code.eorRegReg(dest_reg, lhs_reg, rhs_reg),
    }
}

/// Result of compiling an IR module.
pub const CompileResult = struct {
    code: []u8,
    offsets: []u32,
};

/// Compile all functions in an IR module to AArch64 machine code.
pub fn compileModule(ir_module: *const ir.IrModule, allocator: std.mem.Allocator) !CompileResult {
    var all_code: std.ArrayListUnmanaged(u8) = .{};
    errdefer all_code.deinit(allocator);
    var offsets: std.ArrayListUnmanaged(u32) = .{};
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

// ── Tests ───────────────────────────────────────────────────────────────────

test "compileFunction: iconst_32 + ret" {
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

    // Should produce: prologue + MOVZ + epilogue
    // All instructions are 4 bytes each
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0); // all 4-byte aligned
}

test "compileFunction: add two constants" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compileModule: records offsets" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    // Add two functions
    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    const b1 = try f1.newBlock();
    const v0 = f1.newVReg();
    try f1.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try f1.getBlock(b1).append(.{ .op = .{ .ret = v0 } });
    _ = try module.addFunction(f1);

    var f2 = ir.IrFunction.init(allocator, 0, 1, 0);
    const b2 = try f2.newBlock();
    const v1 = f2.newVReg();
    try f2.getBlock(b2).append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1 });
    try f2.getBlock(b2).append(.{ .op = .{ .ret = v1 } });
    _ = try module.addFunction(f2);

    const result = try compileModule(&module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    try std.testing.expectEqual(@as(usize, 2), result.offsets.len);
    try std.testing.expectEqual(@as(u32, 0), result.offsets[0]);
    try std.testing.expect(result.offsets[1] > 0); // second function starts after first
}
