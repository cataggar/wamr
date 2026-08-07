//! Small in-process interpreter for compiler IR property tests (#736).
//!
//! This is an optimizer oracle, not a production runtime. It executes a
//! deliberately bounded scalar subset with explicit outcomes for traps,
//! unsupported operations, and fuel exhaustion so property tests can compare
//! behavior without panicking or hanging.

const std = @import("std");
const ir = @import("ir.zig");

pub const Value = struct {
    ty: ir.IrType = .i32,
    bits: u64 = 0,

    pub fn i32v(x: i32) Value {
        return .{ .ty = .i32, .bits = @as(u32, @bitCast(x)) };
    }

    pub fn u32v(x: u32) Value {
        return .{ .ty = .i32, .bits = x };
    }

    pub fn i64v(x: i64) Value {
        return .{ .ty = .i64, .bits = @bitCast(x) };
    }

    pub fn u64v(x: u64) Value {
        return .{ .ty = .i64, .bits = x };
    }

    fn asI64(self: Value) i64 {
        return @bitCast(self.bits);
    }

    fn normalized(self: Value) Value {
        return switch (self.ty) {
            .i32 => .{ .ty = .i32, .bits = @as(u32, @truncate(self.bits)) },
            .i64 => .{ .ty = .i64, .bits = self.bits },
            else => self,
        };
    }
};

pub const Trap = enum {
    unreachable_reached,
    integer_divide_by_zero,
    integer_overflow,
    out_of_bounds_memory,
    uninitialized_vreg,
    invalid_local,
    invalid_global,
    invalid_phi,
    invalid_branch,
};

pub const Returned = struct {
    results: []Value,
    memory: []u8,
};

pub const Inconclusive = enum {
    fuel_exhausted,
};

pub const Outcome = union(enum) {
    returned: Returned,
    trapped: Trap,
    unsupported: []const u8,
    inconclusive: Inconclusive,

    pub fn deinit(self: *Outcome, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .returned => |r| {
                allocator.free(r.results);
                allocator.free(r.memory);
            },
            else => {},
        }
        self.* = .{ .unsupported = "deinitialized" };
    }
};

pub const RunOptions = struct {
    params: []const Value = &.{},
    globals: []const Value = &.{},
    memory: []const u8 = &.{},
    fuel: u32 = 10_000,
};

const Machine = struct {
    allocator: std.mem.Allocator,
    func: *const ir.IrFunction,
    vregs: []Value,
    vreg_init: []bool,
    locals: []Value,
    globals: []Value,
    memory: []u8,
    memory_transferred: bool = false,
    fuel: u32,
    block: ir.BlockId = 0,
    prev_block: ?ir.BlockId = null,
    inst_index: usize = 0,

    fn deinitScratch(self: *Machine) void {
        self.allocator.free(self.vregs);
        self.allocator.free(self.vreg_init);
        self.allocator.free(self.locals);
        self.allocator.free(self.globals);
        if (!self.memory_transferred) self.allocator.free(self.memory);
    }

    fn stepFuel(self: *Machine) ?Outcome {
        if (self.fuel == 0) return .{ .inconclusive = .fuel_exhausted };
        self.fuel -= 1;
        return null;
    }

    fn getVReg(self: *Machine, v: ir.VReg) !Value {
        if (v >= self.vregs.len or !self.vreg_init[v]) return error.UninitializedVReg;
        return self.vregs[v].normalized();
    }

    fn setVReg(self: *Machine, v: ir.VReg, value: Value) !void {
        if (v >= self.vregs.len) return error.UninitializedVReg;
        self.vregs[v] = value.normalized();
        self.vreg_init[v] = true;
    }

    fn getLocal(self: *Machine, idx: u32) !Value {
        if (idx >= self.locals.len) return error.InvalidLocal;
        return self.locals[idx].normalized();
    }

    fn setLocal(self: *Machine, idx: u32, value: Value) !void {
        if (idx >= self.locals.len) return error.InvalidLocal;
        self.locals[idx] = value.normalized();
    }

    fn getGlobal(self: *Machine, idx: u32) !Value {
        if (idx >= self.globals.len) return error.InvalidGlobal;
        return self.globals[idx].normalized();
    }

    fn setGlobal(self: *Machine, idx: u32, value: Value) !void {
        if (idx >= self.globals.len) return error.InvalidGlobal;
        self.globals[idx] = value.normalized();
    }

    fn branchTo(self: *Machine, target: ir.BlockId) !void {
        if (target >= self.func.blocks.items.len) return error.InvalidBranch;
        self.prev_block = self.block;
        self.block = target;
        self.inst_index = 0;
    }

    fn readMem(self: *Machine, ptr: u64, size: u8, sign_extend: bool) !Value {
        const start: usize = std.math.cast(usize, ptr) orelse return error.OutOfBoundsMemory;
        const end = start + @as(usize, size);
        if (size == 0 or size > 8 or end > self.memory.len) return error.OutOfBoundsMemory;

        var bits: u64 = 0;
        for (self.memory[start..end], 0..) |b, i| {
            bits |= @as(u64, b) << @intCast(i * 8);
        }
        if (sign_extend and size < 8) {
            const shift: u6 = @intCast((8 - size) * 8);
            bits = @bitCast((@as(i64, @bitCast(bits << shift))) >> shift);
        }
        return .{ .ty = if (size <= 4) .i32 else .i64, .bits = bits };
    }

    fn writeMem(self: *Machine, ptr: u64, size: u8, value: Value) !void {
        const start: usize = std.math.cast(usize, ptr) orelse return error.OutOfBoundsMemory;
        const end = start + @as(usize, size);
        if (size == 0 or size > 8 or end > self.memory.len) return error.OutOfBoundsMemory;

        var bits = value.bits;
        for (self.memory[start..end]) |*b| {
            b.* = @truncate(bits);
            bits >>= 8;
        }
    }

    fn mockCall(self: *Machine, cl: anytype, dest: ?ir.VReg, ty: ir.IrType) !void {
        var acc: u64 = @as(u64, cl.func_idx) *% 0x9e37_79b9;
        for (cl.args) |arg| {
            acc = (acc *% 0x1000_0000_01b3) ^ (try self.getVReg(arg)).bits;
        }
        if (self.memory.len > 0) {
            self.memory[0] +%= @truncate(acc);
        }
        if (dest) |d| try self.setVReg(d, .{ .ty = ty, .bits = acc });
    }

    fn processPhiGroup(self: *Machine) !void {
        const block = &self.func.blocks.items[self.block];
        if (self.inst_index >= block.instructions.items.len) return;
        if (block.instructions.items[self.inst_index].op != .phi) return;
        const pred = self.prev_block orelse return error.InvalidPhi;

        const start = self.inst_index;
        var end = start;
        while (end < block.instructions.items.len and block.instructions.items[end].op == .phi) : (end += 1) {}
        const count = end - start;

        const values = try self.allocator.alloc(Value, count);
        defer self.allocator.free(values);
        const dests = try self.allocator.alloc(ir.VReg, count);
        defer self.allocator.free(dests);

        for (block.instructions.items[start..end], 0..) |inst, i| {
            dests[i] = inst.dest orelse return error.InvalidPhi;
            const edges = inst.op.phi;
            for (edges) |edge| {
                if (edge.block == pred) {
                    values[i] = .{ .ty = inst.type, .bits = (try self.getVReg(edge.val)).bits };
                    break;
                }
            } else {
                return error.InvalidPhi;
            }
        }

        for (dests, values) |dest, value| try self.setVReg(dest, value);
        self.inst_index = end;
    }
};

pub fn run(allocator: std.mem.Allocator, func: *const ir.IrFunction, options: RunOptions) !Outcome {
    if (options.params.len != func.param_count) return error.InvalidArgs;
    if (func.blocks.items.len == 0) {
        return .{ .returned = .{
            .results = &.{},
            .memory = try allocator.dupe(u8, options.memory),
        } };
    }

    var machine = Machine{
        .allocator = allocator,
        .func = func,
        .vregs = try allocator.alloc(Value, func.next_vreg),
        .vreg_init = try allocator.alloc(bool, func.next_vreg),
        .locals = try allocator.alloc(Value, @max(func.local_count, func.param_count)),
        .globals = try allocator.dupe(Value, options.globals),
        .memory = try allocator.dupe(u8, options.memory),
        .fuel = options.fuel,
    };
    defer machine.deinitScratch();
    @memset(machine.vregs, .{});
    @memset(machine.vreg_init, false);
    @memset(machine.locals, .{});

    for (options.params, 0..) |param, i| {
        const normalized = param.normalized();
        machine.locals[i] = normalized;
        try machine.setVReg(@intCast(i), normalized);
    }

    while (true) {
        if (machine.stepFuel()) |outcome| return outcome;
        const block = &func.blocks.items[machine.block];
        machine.processPhiGroup() catch |err| switch (err) {
            error.UninitializedVReg => return .{ .trapped = .uninitialized_vreg },
            error.InvalidPhi => return .{ .trapped = .invalid_phi },
            else => return err,
        };
        if (machine.inst_index >= block.instructions.items.len) {
            return .{ .trapped = .invalid_branch };
        }
        const inst = block.instructions.items[machine.inst_index];
        machine.inst_index += 1;

        const maybe_outcome = execInst(&machine, inst) catch |err| switch (err) {
            error.UninitializedVReg => return .{ .trapped = .uninitialized_vreg },
            error.InvalidLocal => return .{ .trapped = .invalid_local },
            error.InvalidGlobal => return .{ .trapped = .invalid_global },
            error.InvalidPhi => return .{ .trapped = .invalid_phi },
            error.InvalidBranch => return .{ .trapped = .invalid_branch },
            error.OutOfBoundsMemory => return .{ .trapped = .out_of_bounds_memory },
            error.DivideByZero => return .{ .trapped = .integer_divide_by_zero },
            error.IntegerOverflow => return .{ .trapped = .integer_overflow },
            else => return err,
        };
        if (maybe_outcome) |outcome| return outcome;
    }
}

fn execInst(machine: *Machine, inst: ir.Inst) !?Outcome {
    switch (inst.op) {
        .iconst_32 => |x| if (inst.dest) |d| try machine.setVReg(d, Value.i32v(x)),
        .iconst_64 => |x| if (inst.dest) |d| try machine.setVReg(d, Value.i64v(x)),
        .add, .sub, .mul, .div_s, .div_u, .rem_s, .rem_u, .@"and", .@"or", .xor, .shl, .shr_s, .shr_u, .rotl, .rotr, .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u => |bin| {
            const lhs = try machine.getVReg(bin.lhs);
            const rhs = try machine.getVReg(bin.rhs);
            const bits = try evalBinOp(inst.op, lhs.asI64(), rhs.asI64(), inst.type);
            if (inst.dest) |d| try machine.setVReg(d, .{ .ty = resultTypeForBinOp(inst.op, inst.type), .bits = @bitCast(bits) });
        },
        .clz, .ctz, .popcnt, .eqz => |v| {
            const x = try machine.getVReg(v);
            const bits: u64 = switch (inst.op) {
                .clz => if (inst.type == .i64) @clz(x.bits) else @clz(@as(u32, @truncate(x.bits))),
                .ctz => if (inst.type == .i64) @ctz(x.bits) else @ctz(@as(u32, @truncate(x.bits))),
                .popcnt => if (inst.type == .i64) @popCount(x.bits) else @popCount(@as(u32, @truncate(x.bits))),
                .eqz => @intFromBool(x.bits == 0),
                else => unreachable,
            };
            if (inst.dest) |d| try machine.setVReg(d, .{ .ty = if (inst.op == .eqz) .i32 else inst.type, .bits = bits });
        },
        .lea => |l| {
            const base = try machine.getVReg(l.base);
            const index = try machine.getVReg(l.index);
            // base + index*scale + disp, computed in 64 bits and wrapped to
            // the result width by `setVReg`/`normalized`.
            const addr = base.asI64() +% index.asI64() *% @as(i64, l.scale) +% @as(i64, l.disp);
            if (inst.dest) |d| try machine.setVReg(d, .{ .ty = inst.type, .bits = @bitCast(addr) });
        },
        .select => |sel| {
            const cond = try machine.getVReg(sel.cond);
            const chosen = if (cond.bits != 0) try machine.getVReg(sel.if_true) else try machine.getVReg(sel.if_false);
            if (inst.dest) |d| try machine.setVReg(d, .{ .ty = inst.type, .bits = chosen.bits });
        },
        .local_get => |idx| {
            if (inst.dest) |d| try machine.setVReg(d, .{ .ty = inst.type, .bits = (try machine.getLocal(idx)).bits });
        },
        .local_set => |ls| try machine.setLocal(ls.idx, try machine.getVReg(ls.val)),
        .global_get => |idx| {
            if (inst.dest) |d| try machine.setVReg(d, .{ .ty = inst.type, .bits = (try machine.getGlobal(idx)).bits });
        },
        .global_set => |gs| try machine.setGlobal(gs.idx, try machine.getVReg(gs.val)),
        .load => |ld| {
            const base = try machine.getVReg(ld.base);
            const ptr = base.bits +% ld.offset;
            if (inst.dest) |d| try machine.setVReg(d, try machine.readMem(ptr, ld.size, ld.sign_extend));
        },
        .store => |st| {
            const base = try machine.getVReg(st.base);
            const ptr = base.bits +% st.offset;
            try machine.writeMem(ptr, st.size, try machine.getVReg(st.val));
        },
        .br => |target| try machine.branchTo(target),
        .br_if => |bi| {
            const cond = try machine.getVReg(bi.cond);
            try machine.branchTo(if (cond.bits != 0) bi.then_block else bi.else_block);
        },
        .br_table => |bt| {
            const idx: usize = @truncate((try machine.getVReg(bt.index)).bits);
            try machine.branchTo(if (idx < bt.targets.len) bt.targets[idx] else bt.default);
        },
        .ret => |maybe_v| {
            const results = if (maybe_v) |v| blk: {
                const value = (try machine.getVReg(v)).normalized();
                const out = try machine.allocator.alloc(Value, 1);
                out[0] = value;
                break :blk out;
            } else try machine.allocator.alloc(Value, 0);
            machine.memory_transferred = true;
            return .{ .returned = .{ .results = results, .memory = machine.memory } };
        },
        .ret_multi => |vregs| {
            const results = try machine.allocator.alloc(Value, vregs.len);
            errdefer machine.allocator.free(results);
            for (vregs, 0..) |v, i| results[i] = (try machine.getVReg(v)).normalized();
            machine.memory_transferred = true;
            return .{ .returned = .{ .results = results, .memory = machine.memory } };
        },
        .@"unreachable" => return .{ .trapped = .unreachable_reached },
        .phi => return error.InvalidPhi,
        .parallel_copy => |pairs| {
            var scratch = try machine.allocator.alloc(Value, pairs.len);
            defer machine.allocator.free(scratch);
            for (pairs, 0..) |p, i| scratch[i] = try machine.getVReg(p.src);
            for (pairs, 0..) |p, i| try machine.setVReg(p.dst, .{ .ty = p.ty, .bits = scratch[i].bits });
        },
        .call => |cl| try machine.mockCall(cl, inst.dest, inst.type),
        else => return .{ .unsupported = @tagName(inst.op) },
    }
    return null;
}

fn resultTypeForBinOp(op: ir.Inst.Op, operand_ty: ir.IrType) ir.IrType {
    return switch (op) {
        .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u => .i32,
        else => operand_ty,
    };
}

fn evalBinOp(op: ir.Inst.Op, lhs: i64, rhs: i64, ty: ir.IrType) !i64 {
    const mask: u64 = if (ty == .i64) 63 else 31;
    return switch (op) {
        .add => lhs +% rhs,
        .sub => lhs -% rhs,
        .mul => lhs *% rhs,
        .@"and" => lhs & rhs,
        .@"or" => lhs | rhs,
        .xor => lhs ^ rhs,
        .shl => blk: {
            const n: u6 = @intCast(@as(u64, @bitCast(rhs)) & mask);
            break :blk @bitCast(@as(u64, @bitCast(lhs)) << n);
        },
        .shr_s => blk: {
            const n: u6 = @intCast(@as(u64, @bitCast(rhs)) & mask);
            if (ty == .i64) break :blk lhs >> n;
            const l32: i32 = @truncate(lhs);
            break :blk @as(i64, l32 >> @as(u5, @intCast(n)));
        },
        .shr_u => blk: {
            const n: u6 = @intCast(@as(u64, @bitCast(rhs)) & mask);
            if (ty == .i64) break :blk @bitCast(@as(u64, @bitCast(lhs)) >> n);
            const l32: u32 = @truncate(@as(u64, @bitCast(lhs)));
            break :blk @as(i64, l32 >> @as(u5, @intCast(n)));
        },
        .rotl => blk: {
            const n: u6 = @intCast(@as(u64, @bitCast(rhs)) & mask);
            if (ty == .i64) break :blk @bitCast(std.math.rotl(u64, @bitCast(lhs), n));
            const l32: u32 = @truncate(@as(u64, @bitCast(lhs)));
            break :blk @as(i64, std.math.rotl(u32, l32, n));
        },
        .rotr => blk: {
            const n: u6 = @intCast(@as(u64, @bitCast(rhs)) & mask);
            if (ty == .i64) break :blk @bitCast(std.math.rotr(u64, @bitCast(lhs), n));
            const l32: u32 = @truncate(@as(u64, @bitCast(lhs)));
            break :blk @as(i64, std.math.rotr(u32, l32, n));
        },
        .eq => @intFromBool(lhs == rhs),
        .ne => @intFromBool(lhs != rhs),
        .lt_s => if (ty == .i64) @intFromBool(lhs < rhs) else @intFromBool(@as(i32, @truncate(lhs)) < @as(i32, @truncate(rhs))),
        .gt_s => if (ty == .i64) @intFromBool(lhs > rhs) else @intFromBool(@as(i32, @truncate(lhs)) > @as(i32, @truncate(rhs))),
        .le_s => if (ty == .i64) @intFromBool(lhs <= rhs) else @intFromBool(@as(i32, @truncate(lhs)) <= @as(i32, @truncate(rhs))),
        .ge_s => if (ty == .i64) @intFromBool(lhs >= rhs) else @intFromBool(@as(i32, @truncate(lhs)) >= @as(i32, @truncate(rhs))),
        .lt_u => if (ty == .i64) @intFromBool(@as(u64, @bitCast(lhs)) < @as(u64, @bitCast(rhs))) else @intFromBool(@as(u32, @truncate(@as(u64, @bitCast(lhs)))) < @as(u32, @truncate(@as(u64, @bitCast(rhs))))),
        .gt_u => if (ty == .i64) @intFromBool(@as(u64, @bitCast(lhs)) > @as(u64, @bitCast(rhs))) else @intFromBool(@as(u32, @truncate(@as(u64, @bitCast(lhs)))) > @as(u32, @truncate(@as(u64, @bitCast(rhs))))),
        .le_u => if (ty == .i64) @intFromBool(@as(u64, @bitCast(lhs)) <= @as(u64, @bitCast(rhs))) else @intFromBool(@as(u32, @truncate(@as(u64, @bitCast(lhs)))) <= @as(u32, @truncate(@as(u64, @bitCast(rhs))))),
        .ge_u => if (ty == .i64) @intFromBool(@as(u64, @bitCast(lhs)) >= @as(u64, @bitCast(rhs))) else @intFromBool(@as(u32, @truncate(@as(u64, @bitCast(lhs)))) >= @as(u32, @truncate(@as(u64, @bitCast(rhs))))),
        .div_s => blk: {
            if (rhs == 0) return error.DivideByZero;
            if (ty == .i64) {
                if (lhs == std.math.minInt(i64) and rhs == -1) return error.IntegerOverflow;
                break :blk @divTrunc(lhs, rhs);
            }
            const l32: i32 = @truncate(lhs);
            const r32: i32 = @truncate(rhs);
            if (l32 == std.math.minInt(i32) and r32 == -1) return error.IntegerOverflow;
            break :blk @as(i64, @divTrunc(l32, r32));
        },
        .div_u => blk: {
            if (rhs == 0) return error.DivideByZero;
            if (ty == .i64) break :blk @bitCast(@as(u64, @bitCast(lhs)) / @as(u64, @bitCast(rhs)));
            const l32: u32 = @truncate(@as(u64, @bitCast(lhs)));
            const r32: u32 = @truncate(@as(u64, @bitCast(rhs)));
            break :blk @as(i64, l32 / r32);
        },
        .rem_s => blk: {
            if (rhs == 0) return error.DivideByZero;
            if (ty == .i64) {
                if (lhs == std.math.minInt(i64) and rhs == -1) break :blk 0;
                break :blk @rem(lhs, rhs);
            }
            const l32: i32 = @truncate(lhs);
            const r32: i32 = @truncate(rhs);
            if (l32 == std.math.minInt(i32) and r32 == -1) break :blk 0;
            break :blk @as(i64, @rem(l32, r32));
        },
        .rem_u => blk: {
            if (rhs == 0) return error.DivideByZero;
            if (ty == .i64) break :blk @bitCast(@as(u64, @bitCast(lhs)) % @as(u64, @bitCast(rhs)));
            const l32: u32 = @truncate(@as(u64, @bitCast(lhs)));
            const r32: u32 = @truncate(@as(u64, @bitCast(rhs)));
            break :blk @as(i64, l32 % r32);
        },
        else => unreachable,
    };
}

test "interp: straight-line arithmetic returns i32" {
    const a = std.testing.allocator;
    var func = ir.IrFunction.init(a, 0, 1, 0);
    defer func.deinit();

    const b = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try func.getBlock(b).append(.{ .dest = v0, .type = .i32, .op = .{ .iconst_32 = 40 } });
    try func.getBlock(b).append(.{ .dest = v1, .type = .i32, .op = .{ .iconst_32 = 2 } });
    try func.getBlock(b).append(.{ .dest = v2, .type = .i32, .op = .{ .add = .{ .lhs = v0, .rhs = v1 } } });
    try func.getBlock(b).append(.{ .op = .{ .ret = v2 } });

    var outcome = try run(a, &func, .{});
    defer outcome.deinit(a);
    try std.testing.expect(outcome == .returned);
    try std.testing.expectEqual(@as(usize, 1), outcome.returned.results.len);
    try std.testing.expectEqual(@as(u64, 42), outcome.returned.results[0].bits);
}

test "interp: diamond phi selects incoming predecessor" {
    const a = std.testing.allocator;
    var func = ir.IrFunction.init(a, 1, 1, 1);
    defer func.deinit();
    _ = func.newVReg(); // param v0 / local 0

    const entry = try func.newBlock();
    const left = try func.newBlock();
    const right = try func.newBlock();
    const merge = try func.newBlock();
    try func.getBlock(left).addPredecessor(entry);
    try func.getBlock(right).addPredecessor(entry);
    try func.getBlock(merge).addPredecessor(left);
    try func.getBlock(merge).addPredecessor(right);

    const cond = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = cond, .type = .i32, .op = .{ .local_get = 0 } });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = left, .else_block = right } } });

    const v_left = func.newVReg();
    try func.getBlock(left).append(.{ .dest = v_left, .type = .i32, .op = .{ .iconst_32 = 11 } });
    try func.getBlock(left).append(.{ .op = .{ .br = merge } });

    const v_right = func.newVReg();
    try func.getBlock(right).append(.{ .dest = v_right, .type = .i32, .op = .{ .iconst_32 = 22 } });
    try func.getBlock(right).append(.{ .op = .{ .br = merge } });

    const edges = try a.dupe(ir.Inst.PhiEdge, &.{ .{ .block = left, .val = v_left }, .{ .block = right, .val = v_right } });
    const v_phi = func.newVReg();
    try func.getBlock(merge).append(.{ .dest = v_phi, .type = .i32, .op = .{ .phi = edges } });
    try func.getBlock(merge).append(.{ .op = .{ .ret = v_phi } });

    var left_out = try run(a, &func, .{ .params = &.{Value.i32v(1)} });
    defer left_out.deinit(a);
    try std.testing.expectEqual(@as(u64, 11), left_out.returned.results[0].bits);

    var right_out = try run(a, &func, .{ .params = &.{Value.i32v(0)} });
    defer right_out.deinit(a);
    try std.testing.expectEqual(@as(u64, 22), right_out.returned.results[0].bits);
}

test "interp: phi groups read incoming values in parallel" {
    const a = std.testing.allocator;
    var func = ir.IrFunction.init(a, 0, 1, 0);
    defer func.deinit();

    const entry = try func.newBlock();
    const header = try func.newBlock();
    const body = try func.newBlock();
    const exit = try func.newBlock();
    try func.getBlock(header).addPredecessor(entry);
    try func.getBlock(header).addPredecessor(body);
    try func.getBlock(body).addPredecessor(header);
    try func.getBlock(exit).addPredecessor(header);

    const zero = func.newVReg();
    const bound = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = zero, .type = .i32, .op = .{ .iconst_32 = 0 } });
    try func.getBlock(entry).append(.{ .dest = bound, .type = .i32, .op = .{ .iconst_32 = 3 } });
    try func.getBlock(entry).append(.{ .op = .{ .br = header } });

    const phi_i = func.newVReg();
    const phi_prev_i = func.newVReg();
    const inc = func.newVReg();
    const i_edges = try a.dupe(ir.Inst.PhiEdge, &.{ .{ .block = entry, .val = zero }, .{ .block = body, .val = inc } });
    const prev_edges = try a.dupe(ir.Inst.PhiEdge, &.{ .{ .block = entry, .val = zero }, .{ .block = body, .val = phi_i } });
    const cond = func.newVReg();
    try func.getBlock(header).append(.{ .dest = phi_i, .type = .i32, .op = .{ .phi = i_edges } });
    try func.getBlock(header).append(.{ .dest = phi_prev_i, .type = .i32, .op = .{ .phi = prev_edges } });
    try func.getBlock(header).append(.{ .dest = cond, .type = .i32, .op = .{ .lt_u = .{ .lhs = phi_i, .rhs = bound } } });
    try func.getBlock(header).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = body, .else_block = exit } } });

    const one = func.newVReg();
    try func.getBlock(body).append(.{ .dest = one, .type = .i32, .op = .{ .iconst_32 = 1 } });
    try func.getBlock(body).append(.{ .dest = inc, .type = .i32, .op = .{ .add = .{ .lhs = phi_i, .rhs = one } } });
    try func.getBlock(body).append(.{ .op = .{ .br = header } });
    try func.getBlock(exit).append(.{ .op = .{ .ret = phi_prev_i } });

    var outcome = try run(a, &func, .{ .fuel = 100 });
    defer outcome.deinit(a);
    try std.testing.expectEqual(@as(u64, 2), outcome.returned.results[0].bits);
}

test "interp: load and store round-trip memory" {
    const a = std.testing.allocator;
    var func = ir.IrFunction.init(a, 0, 1, 0);
    defer func.deinit();

    const b = try func.newBlock();
    const base = func.newVReg();
    const val = func.newVReg();
    const loaded = func.newVReg();
    try func.getBlock(b).append(.{ .dest = base, .type = .i32, .op = .{ .iconst_32 = 4 } });
    try func.getBlock(b).append(.{ .dest = val, .type = .i32, .op = .{ .iconst_32 = 0x11223344 } });
    try func.getBlock(b).append(.{ .op = .{ .store = .{ .base = base, .offset = 0, .size = 4, .val = val } } });
    try func.getBlock(b).append(.{ .dest = loaded, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(b).append(.{ .op = .{ .ret = loaded } });

    var outcome = try run(a, &func, .{ .memory = &([_]u8{0} ** 16) });
    defer outcome.deinit(a);
    try std.testing.expectEqual(@as(u64, 0x11223344), outcome.returned.results[0].bits);
    try std.testing.expectEqualSlices(u8, &.{ 0x44, 0x33, 0x22, 0x11 }, outcome.returned.memory[4..8]);
}

test "interp: traps and fuel are explicit outcomes" {
    const a = std.testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    try func.getBlock(b).append(.{ .op = .{ .br = b } });

    var outcome = try run(a, &func, .{ .fuel = 3 });
    defer outcome.deinit(a);
    try std.testing.expectEqual(Inconclusive.fuel_exhausted, outcome.inconclusive);

    var trap_func = ir.IrFunction.init(a, 0, 0, 0);
    defer trap_func.deinit();
    const tb = try trap_func.newBlock();
    try trap_func.getBlock(tb).append(.{ .op = .{ .@"unreachable" = {} } });
    var trap = try run(a, &trap_func, .{});
    defer trap.deinit(a);
    try std.testing.expectEqual(Trap.unreachable_reached, trap.trapped);
}
