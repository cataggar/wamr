//! #694 — AOT call_indirect to a function whose funcidx exceeds the
//! historical 256-entry `func_addrs` cap in `mapCodeExecutable`.
//!
//! Before the fix, `mapCodeExecutable` used a stack-allocated fixed
//! `[256]usize func_addrs` buffer. Active element segments referencing
//! a funcidx ≥ 256 silently dropped both the native pointer AND the
//! sig_id update on `TableInstance.type_backing`. The next
//! `call_indirect` against that slot saw `type_backing[slot] == 0`
//! while `sig_table[type_idx] != 0`, failed the sig-id type check, and
//! trapped via `trap_unreachable_fn`. This bit `codegen-cli` because
//! its `heap.ArenaAllocator.alloc` body dispatches through the Zig
//! stdlib `Allocator.VTable.alloc` funcref — type 2 — pointing at a
//! local funcidx well past 256 in the 2080-function module 0.
//!
//! This test programmatically builds a small module with 259 local
//! functions and verifies that a `call_indirect` through a funcidx
//! ≥ 256 dispatched via an active elem segment returns the expected
//! value. Pre-fix this test trapped on `unreachable`; post-fix it
//! returns 42.

const std = @import("std");
const wamr = @import("wamr");
const aot_harness = @import("aot_harness.zig");

const core_types = wamr.types;
const aot_loader_mod = wamr.aot_loader;
const aot_runtime_mod = wamr.aot_runtime;

const FILLER_COUNT: u32 = 256;
const TARGET_LOCAL_IDX: u32 = FILLER_COUNT; // local funcidx 256 — first one past the old 256-entry cap
const ENTRY_LOCAL_IDX: u32 = FILLER_COUNT + 1; // local funcidx 257 — entry point doing `call_indirect`

fn writeULEB128(buf: *std.ArrayList(u8), gpa: std.mem.Allocator, value: u64) !void {
    var v = value;
    while (true) {
        var byte: u8 = @intCast(v & 0x7f);
        v >>= 7;
        if (v != 0) byte |= 0x80;
        try buf.append(gpa, byte);
        if (v == 0) break;
    }
}

fn writeSLEB128(buf: *std.ArrayList(u8), gpa: std.mem.Allocator, value: i64) !void {
    var v = value;
    while (true) {
        const byte_unsigned: u8 = @as(u8, @intCast(@as(u64, @bitCast(v)) & 0x7f));
        const sign_bit_set = (byte_unsigned & 0x40) != 0;
        v >>= 7;
        const done = (v == 0 and !sign_bit_set) or (v == -1 and sign_bit_set);
        try buf.append(gpa, if (done) byte_unsigned else byte_unsigned | 0x80);
        if (done) break;
    }
}

fn writeSection(buf: *std.ArrayList(u8), gpa: std.mem.Allocator, id: u8, payload: []const u8) !void {
    try buf.append(gpa, id);
    try writeULEB128(buf, gpa, payload.len);
    try buf.appendSlice(gpa, payload);
}

/// Build a wasm module:
///   * 1 type:  type 0 = () -> i32
///   * 259 local functions, all of type 0
///       0..255  filler bodies (`i32.const 0`)
///       256     target body   (`i32.const 42`)  — funcidx ≥ 256
///       257     entry body    (`i32.const 0 ; call_indirect 0 (type 0)`)
///   * 1 funcref table of size 1
///   * 1 active elem segment placing func 256 at table[0]
///   * export the entry as "run"
fn buildWasm(gpa: std.mem.Allocator) ![]u8 {
    var bytes: std.ArrayList(u8) = .empty;
    try bytes.appendSlice(gpa, &.{ 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00 });

    // Type section: 1 type, () -> i32
    var sec: std.ArrayList(u8) = .empty;
    defer sec.deinit(gpa);
    try writeULEB128(&sec, gpa, 1);
    try sec.appendSlice(gpa, &.{ 0x60, 0x00, 0x01, 0x7f });
    try writeSection(&bytes, gpa, 0x01, sec.items);
    sec.clearRetainingCapacity();

    // Function section: 258 funcs, all type 0
    const n_funcs: u32 = ENTRY_LOCAL_IDX + 1;
    try writeULEB128(&sec, gpa, n_funcs);
    var i: u32 = 0;
    while (i < n_funcs) : (i += 1) try writeULEB128(&sec, gpa, 0);
    try writeSection(&bytes, gpa, 0x03, sec.items);
    sec.clearRetainingCapacity();

    // Table section: 1 table, funcref, min=1, max=1
    try writeULEB128(&sec, gpa, 1);
    try sec.appendSlice(gpa, &.{ 0x70, 0x01, 0x01, 0x01 });
    try writeSection(&bytes, gpa, 0x04, sec.items);
    sec.clearRetainingCapacity();

    // Export section: 1 export, "run" → entry func
    try writeULEB128(&sec, gpa, 1);
    try writeULEB128(&sec, gpa, 3);
    try sec.appendSlice(gpa, "run");
    try sec.append(gpa, 0x00);
    try writeULEB128(&sec, gpa, ENTRY_LOCAL_IDX);
    try writeSection(&bytes, gpa, 0x07, sec.items);
    sec.clearRetainingCapacity();

    // Element section: 1 active segment, table 0, offset i32.const 0,
    // funcref elemkind, 1 entry = target funcidx.
    try writeULEB128(&sec, gpa, 1);
    try sec.append(gpa, 0x00);
    try sec.append(gpa, 0x41);
    try writeSLEB128(&sec, gpa, 0);
    try sec.append(gpa, 0x0b);
    try writeULEB128(&sec, gpa, 1);
    try writeULEB128(&sec, gpa, TARGET_LOCAL_IDX);
    try writeSection(&bytes, gpa, 0x09, sec.items);
    sec.clearRetainingCapacity();

    // Code section: filler[0..256] + target + entry.
    try writeULEB128(&sec, gpa, n_funcs);
    var idx: u32 = 0;
    while (idx < FILLER_COUNT) : (idx += 1) {
        // body: 0 locals; i32.const 0; end. Body bytes = 0x00 0x41 0x00 0x0b = 4 bytes.
        try writeULEB128(&sec, gpa, 4);
        try sec.appendSlice(gpa, &.{ 0x00, 0x41, 0x00, 0x0b });
    }
    // Target: i32.const 42
    try writeULEB128(&sec, gpa, 4);
    try sec.appendSlice(gpa, &.{ 0x00, 0x41, 42, 0x0b });
    // Entry: i32.const 0; call_indirect 0 (type 0); end
    // body bytes: 0x00 (locals), 0x41 0x00 (i32.const 0), 0x11 0x00 0x00 (call_indirect type=0 table=0), 0x0b
    try writeULEB128(&sec, gpa, 7);
    try sec.appendSlice(gpa, &.{ 0x00, 0x41, 0x00, 0x11, 0x00, 0x00, 0x0b });
    try writeSection(&bytes, gpa, 0x0a, sec.items);
    sec.clearRetainingCapacity();

    return bytes.toOwnedSlice(gpa);
}

test "#694 call_indirect to funcidx beyond historical 256-entry cap" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const wasm_bytes = try buildWasm(allocator);
    defer allocator.free(wasm_bytes);

    const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, wasm_bytes);
    defer allocator.free(cwasm_bytes);

    var module = try aot_loader_mod.load(cwasm_bytes, allocator);
    defer aot_loader_mod.unload(&module, allocator);

    const inst = try aot_runtime_mod.instantiate(&module, allocator);
    defer aot_runtime_mod.destroy(inst);
    try aot_runtime_mod.mapCodeExecutable(inst);

    const fn_idx = aot_runtime_mod.findExportFunc(inst, "run") orelse return error.TestFailed;
    const no_params = [_]core_types.ValType{};
    const result_types = [_]core_types.ValType{.i32};
    const no_args = [_]core_types.Value{};
    var results_buf: [1]aot_runtime_mod.ScalarResult = .{.{ .i32 = 0 }};
    const results = try aot_runtime_mod.callFuncScalar(
        inst,
        fn_idx,
        &no_params,
        &result_types,
        &no_args,
        &results_buf,
    );
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqual(@as(i32, 42), results[0].i32);
}
