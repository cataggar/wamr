//! #625 phase 2 — precompile → manifest → loadManifest → instantiate
//! round-trip smoke test.
//!
//! Builds a minimal component binary (preamble + one core module +
//! one core instance) in-process, runs `precompileComponent` against
//! a tmp directory, reads it back with `loadManifest`, and hands the
//! resulting `PrecompiledCore` slice to `instantiateWithOptions`.
//! Confirms the AOT artifact loaded from disk drives the same code
//! path the phase 1 smoke test exercises with in-memory bytes.

const std = @import("std");
const wamr = @import("wamr");
const aot_harness = @import("aot_harness.zig");

const instance = wamr.component_instance;
const core_types = wamr.types;
const aot_runtime_mod = wamr.aot_runtime;
const component_aot = wamr.component_aot;
const component_aot_compile = wamr.component_aot_compile;

/// Same i32->i32 +42 module used by the phase-1 smoke test.
const core_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x05, 0x03, 0x01, 0x00, 0x01,
    0x07, 0x12, 0x02,
    0x06, 'm', 'e', 'm', 'o', 'r', 'y', 0x02, 0x00,
    0x05, 'a', 'd', 'd', '4', '2', 0x00, 0x00,
    0x0a, 0x09, 0x01, 0x07, 0x00,
    0x20, 0x00,
    0x41, 0x2a,
    0x6a,
    0x0b,
};

/// LEB128-encode a u32 into `out` (assumed large enough; 5 bytes max).
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

fn buildMinimalComponent(allocator: std.mem.Allocator) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    // Preamble: magic + component version (0x0001_000d little-endian).
    try out.appendSlice(allocator, &.{ 0x00, 0x61, 0x73, 0x6d, 0x0d, 0x00, 0x01, 0x00 });

    // Section 1 (core_module): raw wasm bytes, prefixed by LEB size.
    try out.append(allocator, 1);
    try writeLeb(&out, allocator, @intCast(core_wasm.len));
    try out.appendSlice(allocator, &core_wasm);

    // Section 2 (core_instance): count=1, tag=0x00 (instantiate),
    // module_idx=0, arg_count=0.
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    try writeLeb(&body, allocator, 1); // count
    try body.append(allocator, 0x00); // tag
    try writeLeb(&body, allocator, 0); // module_idx
    try writeLeb(&body, allocator, 0); // arg_count

    try out.append(allocator, 2);
    try writeLeb(&out, allocator, @intCast(body.items.len));
    try out.appendSlice(allocator, body.items);

    return out.toOwnedSlice(allocator);
}

test "#625 phase 2: precompile + loadManifest + instantiate round-trip" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const component_bytes = try buildMinimalComponent(allocator);
    defer allocator.free(component_bytes);

    // tmp dir that we own + clean up. Path is `.zig-cache/tmp/<sub_path>`
    // per `std.testing.tmpDir`'s implementation. The manifest sidecar
    // lives at `<tmp>/component.cwasm.json` with cores at
    // `<tmp>/component.<N>.cwasm`.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const manifest_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/component.cwasm.json", .{tmp.sub_path});
    defer allocator.free(manifest_path);

    // Precompile + write manifest.
    var precomp = try component_aot_compile.precompileComponent(allocator, component_bytes, manifest_path, .{});
    defer precomp.deinit();
    try std.testing.expectEqual(@as(usize, 1), precomp.manifest.modules.len);

    // Load manifest from disk, verifying hashes + build id.
    var loaded = try component_aot.loadManifest(allocator, manifest_path, component_bytes);
    defer loaded.deinit();

    const pcs = loaded.precompiledCores();
    try std.testing.expectEqual(@as(usize, 1), pcs.len);
    try std.testing.expectEqual(@as(u32, 0), pcs[0].module_idx);
    // v2 manifests stamp `core_wasm` for slice-identity matching
    // across the runtime's separate parse of `component_bytes`.
    try std.testing.expect(pcs[0].core_wasm != null);

    // Parse `component_bytes` (rather than building a `Component`
    // literal here) so the `core_modules[i].data` slices the
    // runtime sees come from the same backing buffer the loader's
    // re-parse saw — matching slice identities are what
    // `findPrecompiled` uses to resolve a `PrecompiledCore`
    // produced from disk. (#676)
    var parse_arena = std.heap.ArenaAllocator.init(allocator);
    defer parse_arena.deinit();
    const component = try wamr.component_loader.load(component_bytes, parse_arena.allocator());

    const inst = try instance.instantiateWithOptions(&component, allocator, .{
        .precompiled_cores = pcs,
    });
    defer inst.deinit();

    try std.testing.expectEqual(@as(usize, 1), inst.core_instances.len);
    try std.testing.expect(inst.core_instances[0].module_inst == null);
    const ai = inst.core_instances[0].aot_inst orelse return error.TestFailed;

    const be = inst.core_instances[0].backend() orelse return error.TestFailed;
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

test "#625 phase 2: loadManifest rejects mismatched component bytes" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const component_bytes = try buildMinimalComponent(allocator);
    defer allocator.free(component_bytes);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const manifest_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/component.cwasm.json", .{tmp.sub_path});
    defer allocator.free(manifest_path);

    var precomp = try component_aot_compile.precompileComponent(allocator, component_bytes, manifest_path, .{});
    defer precomp.deinit();

    // Hash should reject any other bytes.
    const other_bytes = [_]u8{ 0xde, 0xad, 0xbe, 0xef };
    try std.testing.expectError(error.ManifestComponentMismatch, component_aot.loadManifest(allocator, manifest_path, &other_bytes));
}
