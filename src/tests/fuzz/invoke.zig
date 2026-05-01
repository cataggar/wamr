//! Shared safe invocation policy for fuzz-interp and fuzz-diff.

const std = @import("std");
const wamr = @import("wamr");
const types = wamr.types;
const ExecEnv = wamr.exec_env.ExecEnv;
const Opcode = wamr.opcode.Opcode;

pub const default_fuel_per_export: u32 = 100_000;
pub const exec_stack_size: u32 = 4096;
pub const max_exports_per_input: usize = 8;
pub const max_results: usize = 1;

pub const ExportTarget = struct {
    name: []const u8,
    func_idx: u32,
    func_type: types.FuncType,
};

pub const ScalarValue = union(enum) {
    i32: i32,
    i64: i64,
    f32_bits: u32,
    f64_bits: u64,
};

pub fn runInterpModule(allocator: std.mem.Allocator, bytes: []const u8, fuel_per_export: u32) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const module = wamr.loader.load(bytes, a) catch return;
    if (!modulePolicyAllows(&module)) return;

    const inst = wamr.instance.instantiate(&module, allocator) catch return;
    defer wamr.instance.destroy(inst);

    var invoked: usize = 0;
    for (module.exports) |exp| {
        if (invoked >= max_exports_per_input) break;
        const target = selectExport(&module, exp) orelse continue;

        var result_buf: [max_results]ScalarValue = undefined;
        _ = invokeInterpExport(allocator, inst, target.func_idx, target.func_type, fuel_per_export, &result_buf) catch |err| switch (err) {
            error.ExpectedTrap, error.OutOfFuel => {},
            else => return err,
        };
        invoked += 1;
    }
}

pub fn modulePolicyAllows(module: *const types.WasmModule) bool {
    return module.imports.len == 0 and module.start_function == null;
}

pub fn selectExport(module: *const types.WasmModule, exp: types.ExportDesc) ?ExportTarget {
    if (exp.kind != .function) return null;
    if (exp.index < module.import_function_count) return null;
    const func_type = module.getFuncType(exp.index) orelse return null;
    if (!supportedFuncType(func_type)) return null;
    return .{ .name = exp.name, .func_idx = exp.index, .func_type = func_type };
}

pub fn supportedFuncType(func_type: types.FuncType) bool {
    if (func_type.params.len != 0) return false;
    if (func_type.results.len > max_results) return false;
    for (func_type.results) |result| {
        switch (result) {
            .i32, .i64, .f32, .f64 => {},
            else => return false,
        }
    }
    return true;
}

pub fn invokeInterpExport(
    allocator: std.mem.Allocator,
    inst: *types.ModuleInstance,
    func_idx: u32,
    func_type: types.FuncType,
    fuel_per_export: u32,
    results_out: []ScalarValue,
) ![]const ScalarValue {
    if (!supportedFuncType(func_type) or results_out.len < func_type.results.len)
        return error.UnsupportedSignature;

    var env = try ExecEnv.create(inst, exec_stack_size, allocator);
    defer env.destroy();

    wamr.interp.executeFunctionWithOptions(env, func_idx, .{ .fuel = fuel_per_export }) catch |err| switch (err) {
        error.OutOfFuel => return error.OutOfFuel,
        else => return error.ExpectedTrap,
    };

    return captureResults(env, func_type.results, results_out);
}

pub fn captureResults(env: *ExecEnv, result_types: []const types.ValType, results_out: []ScalarValue) ![]const ScalarValue {
    if (result_types.len > results_out.len) return error.UnsupportedSignature;
    if (@as(usize, @intCast(env.sp)) != result_types.len) return error.InvalidResultStack;

    var i = result_types.len;
    while (i > 0) {
        i -= 1;
        const value = try env.pop();
        results_out[i] = scalarFromValue(result_types[i], value) orelse return error.InvalidResultStack;
    }

    if (env.sp != 0) return error.InvalidResultStack;
    return results_out[0..result_types.len];
}

fn scalarFromValue(result_type: types.ValType, value: types.Value) ?ScalarValue {
    return switch (result_type) {
        .i32 => switch (value) {
            .i32 => |v| .{ .i32 = v },
            else => null,
        },
        .i64 => switch (value) {
            .i64 => |v| .{ .i64 = v },
            else => null,
        },
        .f32 => switch (value) {
            .f32 => |v| .{ .f32_bits = @as(u32, @bitCast(v)) },
            else => null,
        },
        .f64 => switch (value) {
            .f64 => |v| .{ .f64_bits = @as(u64, @bitCast(v)) },
            else => null,
        },
        else => null,
    };
}

pub fn scalarFromAot(result: wamr.aot_runtime.ScalarResult) ?ScalarValue {
    return switch (result) {
        .i32 => |v| .{ .i32 = v },
        .i64 => |v| .{ .i64 = v },
        .f32 => |v| .{ .f32_bits = @as(u32, @bitCast(v)) },
        .f64 => |v| .{ .f64_bits = @as(u64, @bitCast(v)) },
        else => null,
    };
}

pub fn scalarSlicesEqual(a: []const ScalarValue, b: []const ScalarValue) bool {
    if (a.len != b.len) return false;
    for (a, b) |av, bv| {
        if (!scalarEqual(av, bv)) return false;
    }
    return true;
}

fn scalarEqual(a: ScalarValue, b: ScalarValue) bool {
    return switch (a) {
        .i32 => |av| switch (b) {
            .i32 => |bv| av == bv,
            else => false,
        },
        .i64 => |av| switch (b) {
            .i64 => |bv| av == bv,
            else => false,
        },
        .f32_bits => |av| switch (b) {
            .f32_bits => |bv| av == bv or (isNanF32(av) and isNanF32(bv)),
            else => false,
        },
        .f64_bits => |av| switch (b) {
            .f64_bits => |bv| av == bv or (isNanF64(av) and isNanF64(bv)),
            else => false,
        },
    };
}

fn isNanF32(bits: u32) bool {
    const value: f32 = @bitCast(bits);
    return std.math.isNan(value);
}

fn isNanF64(bits: u64) bool {
    const value: f64 = @bitCast(bits);
    return std.math.isNan(value);
}

pub fn isAotSafeExport(module: *const types.WasmModule, func_idx: u32) bool {
    const code = functionCode(module, func_idx) orelse return false;
    return isAotSafeStraightLineCode(code);
}

pub fn functionCode(module: *const types.WasmModule, func_idx: u32) ?[]const u8 {
    if (func_idx < module.import_function_count) return null;
    const local_idx: usize = @intCast(func_idx - module.import_function_count);
    if (local_idx >= module.functions.len) return null;
    return module.functions[local_idx].code;
}

fn isAotSafeStraightLineCode(code: []const u8) bool {
    var ip: usize = 0;
    while (ip < code.len) {
        const op: Opcode = @enumFromInt(code[ip]);
        ip += 1;
        switch (op) {
            .end => return ip == code.len,
            .nop,
            .drop,
            .select,
            .i32_eqz,
            .i32_eq,
            .i32_ne,
            .i32_lt_s,
            .i32_lt_u,
            .i32_gt_s,
            .i32_gt_u,
            .i32_le_s,
            .i32_le_u,
            .i32_ge_s,
            .i32_ge_u,
            .i64_eqz,
            .i64_eq,
            .i64_ne,
            .i64_lt_s,
            .i64_lt_u,
            .i64_gt_s,
            .i64_gt_u,
            .i64_le_s,
            .i64_le_u,
            .i64_ge_s,
            .i64_ge_u,
            .f32_eq,
            .f32_ne,
            .f32_lt,
            .f32_gt,
            .f32_le,
            .f32_ge,
            .f64_eq,
            .f64_ne,
            .f64_lt,
            .f64_gt,
            .f64_le,
            .f64_ge,
            .i32_clz,
            .i32_ctz,
            .i32_popcnt,
            .i32_add,
            .i32_sub,
            .i32_mul,
            .i32_and,
            .i32_or,
            .i32_xor,
            .i32_shl,
            .i32_shr_s,
            .i32_shr_u,
            .i32_rotl,
            .i32_rotr,
            .i64_clz,
            .i64_ctz,
            .i64_popcnt,
            .i64_add,
            .i64_sub,
            .i64_mul,
            .i64_and,
            .i64_or,
            .i64_xor,
            .i64_shl,
            .i64_shr_s,
            .i64_shr_u,
            .i64_rotl,
            .i64_rotr,
            .f32_abs,
            .f32_neg,
            .f32_add,
            .f32_sub,
            .f32_mul,
            .f32_min,
            .f32_max,
            .f32_copysign,
            .f64_abs,
            .f64_neg,
            .f64_add,
            .f64_sub,
            .f64_mul,
            .f64_min,
            .f64_max,
            .f64_copysign,
            .i32_wrap_i64,
            .i64_extend_i32_s,
            .i64_extend_i32_u,
            .f32_convert_i32_s,
            .f32_convert_i32_u,
            .f32_convert_i64_s,
            .f32_convert_i64_u,
            .f32_demote_f64,
            .f64_convert_i32_s,
            .f64_convert_i32_u,
            .f64_convert_i64_s,
            .f64_convert_i64_u,
            .f64_promote_f32,
            .i32_reinterpret_f32,
            .i64_reinterpret_f64,
            .f32_reinterpret_i32,
            .f64_reinterpret_i64,
            .i32_extend8_s,
            .i32_extend16_s,
            .i64_extend8_s,
            .i64_extend16_s,
            .i64_extend32_s,
            => {},

            .local_get,
            .local_set,
            .local_tee,
            .i32_const,
            .i64_const,
            => if (!skipLeb(code, &ip)) return false,

            .f32_const => if (!skipFixed(code, &ip, 4)) return false,
            .f64_const => if (!skipFixed(code, &ip, 8)) return false,
            else => return false,
        }
    }
    return false;
}

fn skipLeb(code: []const u8, ip: *usize) bool {
    var n: usize = 0;
    while (n < 10 and ip.* < code.len) : (n += 1) {
        const b = code[ip.*];
        ip.* += 1;
        if ((b & 0x80) == 0) return true;
    }
    return false;
}

fn skipFixed(code: []const u8, ip: *usize, len: usize) bool {
    if (ip.* + len > code.len) return false;
    ip.* += len;
    return true;
}
