//! Interpreter-vs-AOT differential harness.
//!
//! For each embedded wasm module, this runs the exported `() -> i32`
//! function through both the bytecode interpreter and the AOT pipeline
//! (frontend → passes → codegen → emit_aot → aot_loader → aot_runtime),
//! and asserts the two results match.
//!
//! This is the minimum test that would have caught the `readI32`/`readI64`
//! LEB128 sign-extension bug fixed in commit 709ad073: a wasm module that
//! returns a 1-byte negative `i32.const` (e.g. `-4`) produces -1 in AOT
//! and -4 in the interpreter — a direct divergence.
//!
//! Add new test cases as new AOT codegen regressions are discovered.

const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

const loader_mod = @import("../runtime/interpreter/loader.zig");
const instance_mod = @import("../runtime/interpreter/instance.zig");
const interp = @import("../runtime/interpreter/interp.zig");
const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;

const frontend = @import("../compiler/frontend.zig");
const passes = @import("../compiler/ir/passes.zig");
const x86_64_compile = @import("../compiler/codegen/x86_64/compile.zig");
const aarch64_compile = @import("../compiler/codegen/aarch64/compile.zig");
const emit_aot = @import("../compiler/emit_aot.zig");
const aot_loader = @import("../runtime/aot/loader.zig");
const aot_runtime = @import("../runtime/aot/runtime.zig");

/// True on targets where the AOT runtime can execute generated code.
const can_exec_aot = switch (builtin.cpu.arch) {
    .x86_64, .aarch64 => true,
    else => false,
};

/// Run `name` (a `() -> i32` export) through the interpreter.
fn runInterpI32(allocator: std.mem.Allocator, wasm: []const u8, name: []const u8) !i32 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const module = try loader_mod.load(wasm, arena.allocator());

    const inst = try instance_mod.instantiate(&module, allocator);
    defer instance_mod.destroy(inst);

    const exp = inst.module.findExport(name, .function) orelse return error.FunctionNotFound;

    var env = try ExecEnv.create(inst, 4096, allocator);
    defer env.destroy();
    try interp.executeFunction(env, exp.index);
    return env.popI32();
}

/// Compile `wasm` through the full AOT pipeline, returning the AOT binary.
/// Caller owns the returned slice.
fn compileToAot(allocator: std.mem.Allocator, wasm: []const u8) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const module = try loader_mod.load(wasm, a);

    var ir_module = try frontend.lowerModule(&module, a);
    defer ir_module.deinit();

    _ = try passes.runPasses(&ir_module, passes.default_passes, a);

    const code: []const u8, const offsets: []const u32 = switch (builtin.cpu.arch) {
        .aarch64 => blk: {
            const r = try aarch64_compile.compileModule(&ir_module, a);
            break :blk .{ r.code, r.offsets };
        },
        else => blk: {
            const r = try x86_64_compile.compileModule(&ir_module, a);
            break :blk .{ r.code, r.offsets };
        },
    };

    var exports: std.ArrayList(emit_aot.ExportEntry) = .empty;
    for (module.exports) |exp| {
        try exports.append(a, .{
            .name = exp.name,
            .kind = @enumFromInt(@intFromEnum(exp.kind)),
            .index = exp.index,
        });
    }

    var arch_name = std.mem.zeroes([16]u8);
    switch (builtin.cpu.arch) {
        .aarch64 => @memcpy(arch_name[0..7], "aarch64"),
        else => @memcpy(arch_name[0..6], "x86-64"),
    }

    // emit_aot copies `code` / names into its own buffer, so we can return
    // it even though the arena is torn down on scope exit.
    return try emit_aot.emit(
        allocator,
        code,
        offsets,
        exports.items,
        .{ .arch = arch_name },
        null,
        null,
        null,
        null,
        null,
    );
}

/// Run `name` (a `() -> i32` export) through the AOT pipeline.
fn runAotI32(allocator: std.mem.Allocator, wasm: []const u8, name: []const u8) !i32 {
    const aot_bin = try compileToAot(allocator, wasm);
    defer allocator.free(aot_bin);

    const module = try aot_loader.load(aot_bin, allocator);
    defer aot_loader.unload(&module, allocator);

    const inst = try aot_runtime.instantiate(&module, allocator);
    defer aot_runtime.destroy(inst);

    try aot_runtime.mapCodeExecutable(inst);

    const func_idx = aot_runtime.findExportFunc(inst, name) orelse return error.FunctionNotFound;
    return aot_runtime.callFunc(inst, func_idx, i32);
}

/// Run `wasm` through both pipelines and assert they agree, and match `expected`.
fn expectDiffI32(wasm: []const u8, name: []const u8, expected: i32) !void {
    const interp_result = try runInterpI32(testing.allocator, wasm, name);
    try testing.expectEqual(expected, interp_result);

    if (comptime !can_exec_aot) return;
    const aot_result = try runAotI32(testing.allocator, wasm, name);
    try testing.expectEqual(expected, aot_result);
    try testing.expectEqual(interp_result, aot_result);
}

// ─── Wasm module builder ────────────────────────────────────────────────────

/// Encode an unsigned LEB128 value (u32) into `buf`.
fn encodeULEB128(buf: *std.ArrayList(u8), a: std.mem.Allocator, value: u32) !void {
    var v = value;
    while (true) {
        var byte: u8 = @intCast(v & 0x7F);
        v >>= 7;
        if (v != 0) byte |= 0x80;
        try buf.append(a, byte);
        if (v == 0) break;
    }
}

/// Encode a signed LEB128 value (i64) into `buf`.
fn encodeSLEB128(buf: *std.ArrayList(u8), a: std.mem.Allocator, value: i64) !void {
    var v = value;
    var more = true;
    while (more) {
        const byte: u8 = @as(u8, @truncate(@as(u64, @bitCast(v)))) & 0x7F;
        v >>= 7;
        const sign_bit = byte & 0x40;
        if ((v == 0 and sign_bit == 0) or (v == -1 and sign_bit != 0)) {
            more = false;
            try buf.append(a, byte);
        } else {
            try buf.append(a, byte | 0x80);
        }
    }
}

/// Build a wasm module exporting a single `() -> i32` function whose body is
/// `i32.const <value>; end`.
fn buildConstI32Module(allocator: std.mem.Allocator, value: i32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    // Magic + version
    try out.appendSlice(allocator, &[_]u8{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 });

    // Type section: 1 type — () -> i32
    try out.appendSlice(allocator, &[_]u8{
        0x01, // section id
        0x05, // section size
        0x01, // type count
        0x60, // func
        0x00, // param count
        0x01, // result count
        0x7F, // i32
    });

    // Function section: 1 function, type index 0
    try out.appendSlice(allocator, &[_]u8{ 0x03, 0x02, 0x01, 0x00 });

    // Export section: "f" -> func 0
    try out.appendSlice(allocator, &[_]u8{
        0x07, 0x05, 0x01, 0x01, 'f', 0x00, 0x00,
    });

    // Code section
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    try body.append(allocator, 0x00); // local decl count
    try body.append(allocator, 0x41); // i32.const
    try encodeSLEB128(&body, allocator, value);
    try body.append(allocator, 0x0B); // end

    var code: std.ArrayList(u8) = .empty;
    defer code.deinit(allocator);
    try code.append(allocator, 0x01); // function count
    try encodeULEB128(&code, allocator, @intCast(body.items.len));
    try code.appendSlice(allocator, body.items);

    try out.append(allocator, 0x0A); // code section id
    try encodeULEB128(&out, allocator, @intCast(code.items.len));
    try out.appendSlice(allocator, code.items);

    return out.toOwnedSlice(allocator);
}

/// Build a wasm module exporting `() -> i32` whose body performs
/// `(i32.const a) (i32.const b) <op>; end`.
fn buildBinI32Module(
    allocator: std.mem.Allocator,
    a_val: i32,
    b_val: i32,
    op: u8,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, &[_]u8{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 });
    try out.appendSlice(allocator, &[_]u8{
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7F,
    });
    try out.appendSlice(allocator, &[_]u8{ 0x03, 0x02, 0x01, 0x00 });
    try out.appendSlice(allocator, &[_]u8{
        0x07, 0x05, 0x01, 0x01, 'f', 0x00, 0x00,
    });

    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    try body.append(allocator, 0x00);
    try body.append(allocator, 0x41);
    try encodeSLEB128(&body, allocator, a_val);
    try body.append(allocator, 0x41);
    try encodeSLEB128(&body, allocator, b_val);
    try body.append(allocator, op);
    try body.append(allocator, 0x0B);

    var code: std.ArrayList(u8) = .empty;
    defer code.deinit(allocator);
    try code.append(allocator, 0x01);
    try encodeULEB128(&code, allocator, @intCast(body.items.len));
    try code.appendSlice(allocator, body.items);

    try out.append(allocator, 0x0A);
    try encodeULEB128(&out, allocator, @intCast(code.items.len));
    try out.appendSlice(allocator, code.items);

    return out.toOwnedSlice(allocator);
}

// ─── Tests ──────────────────────────────────────────────────────────────────

test "differential: i32.const 0" {
    const wasm = try buildConstI32Module(testing.allocator, 0);
    defer testing.allocator.free(wasm);
    try expectDiffI32(wasm, "f", 0);
}

test "differential: i32.const 1" {
    const wasm = try buildConstI32Module(testing.allocator, 1);
    defer testing.allocator.free(wasm);
    try expectDiffI32(wasm, "f", 1);
}

test "differential: i32.const -1 (1-byte signed LEB, the regression case)" {
    const wasm = try buildConstI32Module(testing.allocator, -1);
    defer testing.allocator.free(wasm);
    try expectDiffI32(wasm, "f", -1);
}

test "differential: i32.const -4 (coremark-style 1-byte negative)" {
    const wasm = try buildConstI32Module(testing.allocator, -4);
    defer testing.allocator.free(wasm);
    try expectDiffI32(wasm, "f", -4);
}

test "differential: i32.const 63 (1-byte positive boundary)" {
    const wasm = try buildConstI32Module(testing.allocator, 63);
    defer testing.allocator.free(wasm);
    try expectDiffI32(wasm, "f", 63);
}

test "differential: i32.const -64 (1-byte negative boundary)" {
    const wasm = try buildConstI32Module(testing.allocator, -64);
    defer testing.allocator.free(wasm);
    try expectDiffI32(wasm, "f", -64);
}

test "differential: i32.const 64 (first 2-byte positive)" {
    const wasm = try buildConstI32Module(testing.allocator, 64);
    defer testing.allocator.free(wasm);
    try expectDiffI32(wasm, "f", 64);
}

test "differential: i32.const -65 (first 2-byte negative)" {
    const wasm = try buildConstI32Module(testing.allocator, -65);
    defer testing.allocator.free(wasm);
    try expectDiffI32(wasm, "f", -65);
}

test "differential: i32.const INT32_MIN" {
    const wasm = try buildConstI32Module(testing.allocator, std.math.minInt(i32));
    defer testing.allocator.free(wasm);
    try expectDiffI32(wasm, "f", std.math.minInt(i32));
}

test "differential: i32.const INT32_MAX" {
    const wasm = try buildConstI32Module(testing.allocator, std.math.maxInt(i32));
    defer testing.allocator.free(wasm);
    try expectDiffI32(wasm, "f", std.math.maxInt(i32));
}

test "differential: (-4) & 1 == 0 (i32.and over negative const)" {
    // 0x6F = i32.and
    const wasm = try buildBinI32Module(testing.allocator, -4, 1, 0x71);
    defer testing.allocator.free(wasm);
    // -4 & 1 == 0
    try expectDiffI32(wasm, "f", 0);
}

test "differential: (-4) + 5 == 1 (i32.add over negative const)" {
    // 0x6A = i32.add
    const wasm = try buildBinI32Module(testing.allocator, -4, 5, 0x6A);
    defer testing.allocator.free(wasm);
    try expectDiffI32(wasm, "f", 1);
}
