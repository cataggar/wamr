//! End-to-end regression for issue #402: a host function that returns a
//! `list<u8>` as a `.list_val` (the eagerly-lifted host shape) must
//! survive the canon-lower trampoline path and round-trip through
//! `canonical_abi.storeVal` without panicking on the union-tag mismatch
//! at canonical_abi.zig:782 (`val.list.ptr` access while `.list_val` is
//! the active arm).
//!
//! Scope: in-process — drives the same two-step sequence that
//! `executor.componentTrampoline` performs on every host result:
//!
//!   1. `ComponentInstance.lowerListVals(val, t, reg, allocator)` —
//!      materializes any `.list_val` subtrees into the canonical
//!      `.list = PtrLen` form via `cabi_realloc` (or, here, the
//!      `enableTestMem` shim).
//!   2. `canonical_abi.storeVal(memory, ptr, t, lowered)` — the
//!      previously-panicking lower step. Asserts the bytes land at the
//!      pointer recorded in the lowered list and round-trip back out of
//!      the test-memory buffer.
//!
//! The closest e2e fixture would be a real `wasm32-wasip2` component
//! whose host function returns a `list<u8>` (the bug originally
//! manifested in `wasi:http/proxy`); we cannot build one in this
//! checkout because `wasm-tools` is not available locally and the task
//! constraints forbid touching `s{instance,executor,
//! canonical_abi,wasi_cli_adapter}.zig` (parallel agent surfaces). This
//! test is the step-8 fallback called out in the task plan: lower
//! fidelity than a full guest-driven round-trip, but it locks in the
//! exact lower path that previously crashed and is wired into the
//! default `zig build test` graph.

const std = @import("std");

// Import the component-model surfaces directly (rather than via
// `src/root.zig`) so this regression test does not pull
// `wasi_cli_adapter.zig` into its compile graph — that file is owned
// by a parallel migration and may temporarily fail to compile.
const ctypes = @import("component/types.zig");
const abi = @import("component/canonical_abi.zig");
const instance_mod = @import("component/instance.zig");
const ComponentInstance = instance_mod.ComponentInstance;
const InterfaceValue = abi.InterfaceValue;

test "issue #402: list_val<u8> survives lowerListVals + storeVal without panicking" {
    const allocator = std.testing.allocator;

    // One-typedef component: a single `list<u8>` at type-idx 0. Mirrors
    // the smallest shape an exported host function could produce.
    const comp_types = [_]ctypes.TypeDef{
        .{ .list = .{ .element = .u8 } },
    };
    const comp_idxspace = [_]?u32{0};
    var component = std.mem.zeroes(ctypes.Component);
    component.types = &comp_types;
    component.type_indexspace = &comp_idxspace;

    var ci: ComponentInstance = undefined;
    ci.component = &component;
    ci.test_mem = null;
    ci.allocator = allocator;
    try ci.enableTestMem(allocator, 4096);
    defer ci.disableTestMem();

    // Build a host-style `.list_val` of 8 bytes — the exact shape that
    // wasi:http body adapters and other host functions produced before
    // the fix. Arms here are the eager `.u8` form so `lowerListVals`
    // has to walk the slice and copy each element into guest memory.
    const payload = "402-fix\n";
    const elems = try allocator.alloc(InterfaceValue, payload.len);
    for (payload, 0..) |b, i| elems[i] = .{ .u8 = b };
    const host_val: InterfaceValue = .{ .list_val = elems };

    // Step 1: trampoline normalization. Identical call shape to
    // executor.zig:1168 (the canon-lower trampoline).
    const reg = abi.TypeRegistry.init(&component);
    const lowered = try ci.lowerListVals(host_val, .{ .list = 0 }, reg, allocator);
    defer lowered.deinit(allocator);

    try std.testing.expect(lowered == .list);
    try std.testing.expectEqual(@as(u32, payload.len), lowered.list.len);

    // Step 2: canon-lower into a fresh "linear memory" buffer. This is
    // the call that previously panicked on `val.list.ptr` access when
    // the active arm was `.list_val`. We exercise both `storeVal`
    // (single value) and the registry-aware wrapper to mirror the two
    // store entry points wired up in executor.storeInterfaceValue.
    var linear_mem = [_]u8{0} ** 32;
    try abi.storeVal(&linear_mem, 0, .{ .list = 0 }, lowered);
    try abi.storeValReg(&linear_mem, 8, .{ .list = 0 }, lowered, reg);

    // Both stores must record the same (ptr, len) pair into linear
    // memory — i.e. the lowered `.list = PtrLen` was read correctly,
    // not the host `.list_val` slice.
    const ptr_a = std.mem.readInt(u32, linear_mem[0..4], .little);
    const len_a = std.mem.readInt(u32, linear_mem[4..8], .little);
    const ptr_b = std.mem.readInt(u32, linear_mem[8..12], .little);
    const len_b = std.mem.readInt(u32, linear_mem[12..16], .little);
    try std.testing.expectEqual(lowered.list.ptr, ptr_a);
    try std.testing.expectEqual(lowered.list.len, len_a);
    try std.testing.expectEqual(lowered.list.ptr, ptr_b);
    try std.testing.expectEqual(lowered.list.len, len_b);

    // Bytes pointed to by the lowered list must round-trip through the
    // test-memory buffer — i.e. each `.u8 = b` element was actually
    // written into guest memory rather than dropped.
    const buf = ci.test_mem.?.buffer;
    try std.testing.expectEqualSlices(
        u8,
        payload,
        buf[lowered.list.ptr .. lowered.list.ptr + lowered.list.len],
    );
}
