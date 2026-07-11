//! #676 — precompile + loadManifest + instantiate round-trip for a
//! composed component whose core module lives inside a nested
//! sub-component (the dominant shape of `wabt component compose -d`
//! / `wasm-tools compose` output).
//!
//! Builds a 2-level component in-process: the top-level component
//! has zero of its own cores and contains a single sub-component
//! (component section, id 4) which contains the `add42` core module
//! plus a core-instance section instantiating it. The top-level
//! instance section instantiates the sub-component.
//!
//! Validates that `precompileComponent` recurses into the
//! sub-component (writing one `<stem>.<N>.cwasm` per leaf core) and
//! that `loadManifest` resolves each entry to the live
//! `core_modules[i].data` slice by `core_sha256`, stamping
//! `core_wasm` so `Options.findPrecompiled` matches by slice
//! identity through the parent-options-propagation path at
//! `instance.zig:2292`.

const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");
const aot_harness = @import("aot_harness.zig");

const config = wamr.config;
const instance = wamr.component_instance;
const core_types = wamr.types;
const aot_runtime_mod = wamr.aot_runtime;
const component_aot = wamr.component_aot;
const component_aot_compile = wamr.component_aot_compile;

/// Same i32->i32 +42 module used by the phase-1 smoke test.
const core_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00, 0x05, 0x03, 0x01, 0x00,
    0x01, 0x07, 0x12, 0x02, 0x06, 'm',  'e',  'm',
    'o',  'r',  'y',  0x02, 0x00, 0x05, 'a',  'd',
    'd',  '4',  '2',  0x00, 0x00, 0x0a, 0x09, 0x01,
    0x07, 0x00, 0x20, 0x00, 0x41, 0x2a, 0x6a, 0x0b,
};

fn writeLeb(out: *std.ArrayList(u8), allocator: std.mem.Allocator, v: u32) !void {
    var x = v;
    while (true) {
        var b: u8 = @intCast(x & 0x7f);
        x >>= 7;
        if (x != 0) b |= 0x80;
        try out.append(allocator, b);
        if (x == 0) break;
    }
}

/// Build a minimal sub-component containing one core module + one
/// core-instance that instantiates it. Returned bytes are owned by
/// the caller.
fn buildSubComponent(allocator: std.mem.Allocator) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, &.{ 0x00, 0x61, 0x73, 0x6d, 0x0d, 0x00, 0x01, 0x00 });

    try out.append(allocator, 1);
    try writeLeb(&out, allocator, @intCast(core_wasm.len));
    try out.appendSlice(allocator, &core_wasm);

    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    try writeLeb(&body, allocator, 1);
    try body.append(allocator, 0x00);
    try writeLeb(&body, allocator, 0);
    try writeLeb(&body, allocator, 0);

    try out.append(allocator, 2);
    try writeLeb(&out, allocator, @intCast(body.items.len));
    try out.appendSlice(allocator, body.items);

    return out.toOwnedSlice(allocator);
}

/// Build a top-level component that embeds the sub-component as
/// section 4 (component) and instantiates it via section 5
/// (instance, tag 0x00).
fn buildNestedComponent(allocator: std.mem.Allocator) ![]u8 {
    const sub_bytes = try buildSubComponent(allocator);
    defer allocator.free(sub_bytes);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, &.{ 0x00, 0x61, 0x73, 0x6d, 0x0d, 0x00, 0x01, 0x00 });

    try out.append(allocator, 4);
    try writeLeb(&out, allocator, @intCast(sub_bytes.len));
    try out.appendSlice(allocator, sub_bytes);

    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    try writeLeb(&body, allocator, 1);
    try body.append(allocator, 0x00);
    try writeLeb(&body, allocator, 0);
    try writeLeb(&body, allocator, 0);

    try out.append(allocator, 5);
    try writeLeb(&out, allocator, @intCast(body.items.len));
    try out.appendSlice(allocator, body.items);

    return out.toOwnedSlice(allocator);
}

test "#676: precompile recurses into nested sub-components" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const component_bytes = try buildNestedComponent(allocator);
    defer allocator.free(component_bytes);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const manifest_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/nested.cwasm.json", .{tmp.sub_path});
    defer allocator.free(manifest_path);

    var precomp = try component_aot_compile.precompileComponent(allocator, component_bytes, manifest_path, .{});
    defer precomp.deinit();
    try std.testing.expectEqual(@as(usize, 1), precomp.manifest.modules.len);
    try std.testing.expect(precomp.manifest.modules[0].core_sha256 != null);
    try std.testing.expectEqual(@as(u32, 0), precomp.manifest.modules[0].idx);

    var loaded = try component_aot.loadManifest(allocator, manifest_path, component_bytes);
    defer loaded.deinit();

    const pcs = loaded.precompiledCores();
    try std.testing.expectEqual(@as(usize, 1), pcs.len);
    try std.testing.expect(pcs[0].core_wasm != null);
    try std.testing.expectEqual(@as(u32, 0), pcs[0].module_idx);

    var parse_arena = std.heap.ArenaAllocator.init(allocator);
    defer parse_arena.deinit();
    const component = try wamr.component_loader.load(component_bytes, parse_arena.allocator());

    try std.testing.expectEqual(@as(usize, 1), component.components.len);
    try std.testing.expectEqual(@as(usize, 0), component.core_modules.len);
    try std.testing.expectEqual(@as(usize, 1), component.components[0].core_modules.len);

    const inst = try instance.instantiateWithOptions(&component, allocator, .{
        .precompiled_cores = pcs,
    });
    defer inst.deinit();

    try std.testing.expectEqual(@as(usize, 1), inst.sub_instances.len);
    const sub = inst.sub_instances[0] orelse return error.TestFailed;
    try std.testing.expectEqual(@as(usize, 1), sub.core_instances.len);
    try std.testing.expect(sub.core_instances[0].module_inst == null);
    const ai = sub.core_instances[0].aot_inst orelse return error.TestFailed;

    const be = sub.core_instances[0].backend() orelse return error.TestFailed;
    const fn_idx = be.findExportFunc("add42") orelse return error.TestFailed;

    const param_types = [_]core_types.ValType{.i32};
    const result_types = [_]core_types.ValType{.i32};
    const args = [_]core_types.Value{.{ .i32 = 100 }};
    var results_buf: [1]aot_runtime_mod.ScalarResult = .{.{ .i32 = 0 }};
    const results = try aot_runtime_mod.callFuncScalar(
        ai,
        fn_idx,
        &param_types,
        &result_types,
        &args,
        &results_buf,
    );
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqual(@as(i32, 142), results[0].i32);
}

test "#889: nested sub-components attach lazy-JIT sidecars by core_wasm identity" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const component_bytes = try buildNestedComponent(allocator);
    defer allocator.free(component_bytes);

    var in_mem = try component_aot_compile.precompileComponentInMemory(allocator, component_bytes, .{
        .lazy_jit = true,
    });
    defer in_mem.deinit();

    var parse_arena = std.heap.ArenaAllocator.init(allocator);
    defer parse_arena.deinit();
    const component = try wamr.component_loader.load(component_bytes, parse_arena.allocator());

    const inst = try instance.instantiateWithOptions(&component, allocator, in_mem.instantiationOptions());
    defer inst.deinit();

    const sub = inst.sub_instances[0] orelse return error.TestFailed;
    const ai = sub.core_instances[0].aot_inst orelse return error.TestFailed;
    try std.testing.expectEqual(@as(usize, 1), ai.lazy_jit.pending.len);
    try std.testing.expect(ai.lazy_jit.pending[0]);
    try std.testing.expect(ai.lazy_jit.compiled[0] == null);

    const fn_idx = aot_runtime_mod.findExportFunc(ai, "add42") orelse return error.TestFailed;
    var results_buf: [1]aot_runtime_mod.ScalarResult = .{.{ .i32 = 0 }};

    const first = try aot_runtime_mod.callFuncScalar(
        ai,
        fn_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 100 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 142), first[0].i32);
    try std.testing.expect(!ai.lazy_jit.pending[0]);
    const compiled_addr = ai.lazy_jit.compiled[0].?.addr;

    const second = try aot_runtime_mod.callFuncScalar(
        ai,
        fn_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = 0 }},
        &results_buf,
    );
    try std.testing.expectEqual(@as(i32, 42), second[0].i32);
    try std.testing.expectEqual(@intFromPtr(compiled_addr), @intFromPtr(ai.lazy_jit.compiled[0].?.addr));
}
