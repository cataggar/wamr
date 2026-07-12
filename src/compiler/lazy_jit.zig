//! #887 — lazy-JIT eligibility analysis for x86_64 direct-call graphs.
//!
//! Given a parsed core wasm module (for table / element-segment info) and its
//! lowered IR (for call / funcref usage), decide which LOCAL function indices
//! are safe to defer under the current lazy-JIT implementation:
//!
//!   - direct local-call graphs ARE allowed (non-leaf functions may be lazy);
//!   - `call_indirect` / `call_ref` remain eager-only;
//!   - functions that produce raw funcrefs via `ref.func` remain eager-only;
//!   - functions referenced by element segments remain eager-only for now;
//!   - functions reachable from `ref.func` remain eager-only.
//!
//! The returned slice is indexed by LOCAL function index (0-based, excluding
//! imports), matching `ir_module.functions.items`.
const std = @import("std");
const ir = @import("ir/ir.zig");
const core_types = @import("../runtime/common/types.zig");

fn markLocalIfPresent(
    eligible: []bool,
    import_count: u32,
    func_idx: u32,
) void {
    if (func_idx < import_count) return;
    const local_idx = func_idx - import_count;
    if (local_idx < eligible.len) eligible[local_idx] = false;
}

/// Returns a caller-owned `[]bool` where `true` marks a lazy-eligible local
/// function. Free with `allocator.free`.
pub fn findLazyEligibleFunctions(
    module: *const core_types.WasmModule,
    ir_module: *const ir.IrModule,
    allocator: std.mem.Allocator,
) ![]bool {
    const n = ir_module.functions.items.len;
    const eligible = try allocator.alloc(bool, n);
    @memset(eligible, true);

    // Table-reachable functions stay eager so #879's funcref/table follow-up
    // stays isolated, but unrelated locals can still be deferred.
    for (module.elements) |elem| {
        for (elem.func_indices) |maybe_func_idx| {
            if (maybe_func_idx) |func_idx| {
                markLocalIfPresent(eligible, ir_module.import_count, func_idx);
            }
        }
    }

    // Functions that themselves use indirect/funcref dispatch stay eager.
    // `ref.func` also makes the referenced function eager because its raw
    // function pointer can escape into tables or other funcref consumers.
    for (ir_module.functions.items, 0..) |func, local_idx| {
        outer: for (func.blocks.items) |block| {
            for (block.instructions.items) |inst| {
                switch (inst.op) {
                    .call_indirect, .call_ref => {
                        eligible[local_idx] = false;
                        break :outer;
                    },
                    .ref_func => |func_idx| {
                        eligible[local_idx] = false;
                        markLocalIfPresent(eligible, ir_module.import_count, func_idx);
                        break :outer;
                    },
                    else => {},
                }
            }
        }
    }

    return eligible;
}

test "findLazyEligibleFunctions: direct caller and callee can both be lazy" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    var callee = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try callee.newBlock();
    try ir_module.functions.append(allocator, callee);

    var caller = ir.IrFunction.init(allocator, 0, 0, 0);
    const caller_block = try caller.newBlock();
    try caller.getBlock(caller_block).append(.{ .op = .{ .call = .{ .func_idx = 0 } } });
    try ir_module.functions.append(allocator, caller);

    var module = core_types.WasmModule{};
    module.elements = &.{};

    const eligible = try findLazyEligibleFunctions(&module, &ir_module, allocator);
    defer allocator.free(eligible);

    try std.testing.expectEqual(@as(usize, 2), eligible.len);
    try std.testing.expect(eligible[0]);
    try std.testing.expect(eligible[1]);
}

test "findLazyEligibleFunctions: ref.func makes both user and target eager" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    var target = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try target.newBlock();
    try ir_module.functions.append(allocator, target);

    var ref_user = ir.IrFunction.init(allocator, 0, 0, 0);
    const ref_block = try ref_user.newBlock();
    const ref_val = ref_user.newVReg();
    try ref_user.getBlock(ref_block).append(.{
        .op = .{ .ref_func = 0 },
        .dest = ref_val,
        .type = .i64,
    });
    try ir_module.functions.append(allocator, ref_user);

    var unrelated = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try unrelated.newBlock();
    try ir_module.functions.append(allocator, unrelated);

    var module = core_types.WasmModule{};
    module.elements = &.{};

    const eligible = try findLazyEligibleFunctions(&module, &ir_module, allocator);
    defer allocator.free(eligible);

    try std.testing.expect(!eligible[0]);
    try std.testing.expect(!eligible[1]);
    try std.testing.expect(eligible[2]);
}

test "findLazyEligibleFunctions: call_indirect caller and elem targets stay eager" {
    const allocator = std.testing.allocator;

    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    var elem_target = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try elem_target.newBlock();
    try ir_module.functions.append(allocator, elem_target);

    var indirect_user = ir.IrFunction.init(allocator, 0, 0, 0);
    const block = try indirect_user.newBlock();
    const elem_idx = indirect_user.newVReg();
    try indirect_user.getBlock(block).append(.{
        .op = .{ .iconst_32 = 0 },
        .dest = elem_idx,
        .type = .i32,
    });
    try indirect_user.getBlock(block).append(.{
        .op = .{ .call_indirect = .{ .type_idx = 0, .elem_idx = elem_idx } },
    });
    try ir_module.functions.append(allocator, indirect_user);

    var unrelated = ir.IrFunction.init(allocator, 0, 0, 0);
    _ = try unrelated.newBlock();
    try ir_module.functions.append(allocator, unrelated);

    var module = core_types.WasmModule{};
    const elem = [_]core_types.ElemSegment{.{
        .table_idx = 0,
        .offset = null,
        .kind = .func_ref,
        .func_indices = &.{0},
    }};
    module.elements = &elem;

    const eligible = try findLazyEligibleFunctions(&module, &ir_module, allocator);
    defer allocator.free(eligible);

    try std.testing.expect(!eligible[0]);
    try std.testing.expect(!eligible[1]);
    try std.testing.expect(eligible[2]);
}
