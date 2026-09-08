const std = @import("std");

pub const schema_name = "wamr-aot-frame-attribution";
pub const legacy_x86_format_version: u32 = 1;
pub const format_version: u32 = 2;

pub const AccessKind = enum {
    load,
    store,
};

pub const AccessOrigin = enum {
    allocator_spill,
    wasm_local_or_phi,
    explicit_frame_storage,
    fixed_runtime_frame_state,
    unknown,
};

pub const RelocationKind = enum {
    x86_64_call_rel32,
    aarch64_call_imm26,
    aarch64_tail_call_imm26,
};

/// One architecture-specific direct-call relocation normalized for the
/// per-function native-code hash. The range is function-relative and
/// half-open. Consumers validate the instruction encoding before masking the
/// relocation field.
pub const Relocation = struct {
    native_start: u32,
    native_end: u32,
    kind: RelocationKind,
};

pub const FrameLayout = struct {
    frame_pointer: []const u8 = "rbp",
    frame_size: u32,
    local_count: u32,
    param_count: u32,
    reserved_vmctx_offset: i32,
    locals_first_offset: i32,
    explicit_storage_first_offset: i32,
    explicit_storage_slots: u32,
    spill_base: i32,
    spill_stride: i32,
    spill_slots: u32,
};

pub const SpillMetric = struct {
    slots: u32,
    spilled_vregs: u32,
    scalar: u32,
    v128: u32,
    slots_scalar: u32,
    slots_v128: u32,
    spill_ld: u32,
    spill_st: u32,
    remat: u32,
    callee_saved: u32,
};

/// One allocator value assigned to one or more consecutive 8-byte spill
/// slots. More than one value may name overlapping slots after slot reuse;
/// `reused` records that fact and per-access `vreg` is then present only when
/// the emitted IR range proves which value the machine instruction accesses.
pub const SpillValue = struct {
    vreg: u32,
    frame_offset: i32,
    slot: u32,
    slot_count: u8,
    value_type: ?[]const u8,
    live_start: ?u32,
    live_end: ?u32,
    defining_opcode: ?[]const u8,
    source_class: ?[]const u8,
    /// Pre-emission IR references/defs (useful for explaining folds).
    ir_use_count: u32,
    ir_def_count: u32,
    /// Actual emitted allocator loads/stores resolved to this vreg.
    reload_count: u32,
    store_count: u32,
    rematerialization_eligible: bool,
    reused: bool,
};

/// One memory element of a compiler-emitted frame instruction. AArch64 pair
/// instructions use two components under one native range so sampled
/// instructions are counted once while both accessed values remain explicit.
pub const AccessComponent = struct {
    /// Effective address after resolving compiler-materialized frame
    /// addresses. Usually x29/rbp plus `frame_offset`; transient outgoing ABI
    /// storage may remain sp/rsp relative.
    base: []const u8,
    frame_offset: i32,
    width: u8,
    /// Encoded architectural data register. AArch64 schema-v2 sidecars use
    /// this together with `data_register_class` to preserve pair identity.
    data_register: ?u8 = null,
    data_register_class: ?[]const u8 = null,
    origin: AccessOrigin,
    detail: []const u8,
    slot: ?u32 = null,
    local_index: ?u32 = null,
    explicit_slot: ?u32 = null,
    vreg: ?u32 = null,
    vreg_ambiguous: bool = false,
    defining_opcode: ?[]const u8 = null,
    source_class: ?[]const u8 = null,
    rematerialization_eligible: ?bool = null,
    ir_position: ?u32 = null,
    ir_opcode: ?[]const u8 = null,
};

/// A single compiler-emitted frame load/store instruction. Native ranges are
/// function-relative and half-open. Schema v1 x86 sidecars use the legacy
/// top-level semantic fields. Schema v2 additionally records the encoded
/// address and one or two effective-address components.
pub const Access = struct {
    native_start: u32,
    native_end: u32,
    kind: AccessKind,
    base: []const u8,
    frame_offset: i32,
    width: u8,
    origin: AccessOrigin,
    detail: []const u8,
    slot: ?u32 = null,
    local_index: ?u32 = null,
    explicit_slot: ?u32 = null,
    vreg: ?u32 = null,
    vreg_ambiguous: bool = false,
    defining_opcode: ?[]const u8 = null,
    source_class: ?[]const u8 = null,
    rematerialization_eligible: ?bool = null,
    ir_position: ?u32 = null,
    ir_opcode: ?[]const u8 = null,
    encoded_base: ?[]const u8 = null,
    encoded_offset: ?i32 = null,
    addressing_mode: ?[]const u8 = null,
    components: []const AccessComponent = &.{},
};

pub const FrameRegion = struct {
    start: i32,
    end: i32,
    origin: AccessOrigin,
    detail: []const u8,
};

pub const InlineDataRange = struct {
    native_start: u32,
    native_end: u32,
    kind: []const u8 = "br_table",
};

pub const Report = struct {
    schema: []const u8 = schema_name,
    schema_version: u32 = legacy_x86_format_version,
    cwasm_aot_version: u32,
    compiler_build_id: []const u8,
    architecture: []const u8 = "x86_64",
    abi: []const u8,
    module: u32,
    local_func: u32,
    function_name: []const u8,
    module_text_size: u32,
    module_text_sha256: []const u8,
    function_offset: u32,
    code_size: u32,
    normalized_code_sha256: []const u8,
    /// Four bytes at each listed function-relative offset are a direct-call
    /// rel32 relocation. The normalized hash replaces those bytes with zero
    /// both before and after module-level call patching.
    direct_call_rel32_offsets: []const u32,
    normalized_relocations: []const Relocation = &.{},
    inline_data_ranges: []const InlineDataRange,
    frame_layout: FrameLayout,
    frame_regions: []const FrameRegion = &.{},
    spill_metric: SpillMetric,
    emitted_allocator_loads: u32,
    emitted_allocator_stores: u32,
    allocator_values: []const SpillValue,
    accesses: []const Access,
};

pub fn serializeAlloc(allocator: std.mem.Allocator, report: Report) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var sw: std.json.Stringify = .{
        .writer = &aw.writer,
        .options = .{ .whitespace = .indent_2 },
    };
    try sw.write(report);
    return allocator.dupe(u8, aw.written());
}

/// Write one uniquely named function sidecar. The caller supplies an output
/// prefix rather than a directory so component and core-module compilers can
/// share the same codegen API without knowing final `.cwasm` paths.
pub fn writeReport(
    allocator: std.mem.Allocator,
    output_prefix: []const u8,
    module_idx: u32,
    func_idx: u32,
    report: Report,
) !void {
    const path = try std.fmt.allocPrint(
        allocator,
        "{s}.mod{d}.func{d}.json",
        .{ output_prefix, module_idx, func_idx },
    );
    defer allocator.free(path);

    const bytes = try serializeAlloc(allocator, report);
    defer allocator.free(bytes);

    const tmp_path = try std.mem.concat(allocator, u8, &.{ path, ".tmp" });
    defer allocator.free(tmp_path);

    const io = std.Io.Threaded.global_single_threaded.io();
    const cwd = std.Io.Dir.cwd();
    try cwd.writeFile(io, .{ .sub_path = tmp_path, .data = bytes });
    cwd.rename(tmp_path, cwd, path, io) catch {
        try cwd.writeFile(io, .{ .sub_path = path, .data = bytes });
        cwd.deleteFile(io, tmp_path) catch {};
    };
}

test "frame attribution JSON keeps signed offsets and null identities" {
    const allocator = std.testing.allocator;
    const values = [_]SpillValue{.{
        .vreg = 17,
        .frame_offset = -560,
        .slot = 0,
        .slot_count = 1,
        .value_type = "i64",
        .live_start = 2,
        .live_end = 9,
        .defining_opcode = "local_get",
        .source_class = "wasm_local_or_phi",
        .ir_use_count = 3,
        .ir_def_count = 1,
        .reload_count = 3,
        .store_count = 1,
        .rematerialization_eligible = false,
        .reused = false,
    }};
    const accesses = [_]Access{
        .{
            .native_start = 12,
            .native_end = 19,
            .kind = .load,
            .base = "rbp",
            .frame_offset = -560,
            .width = 8,
            .origin = .allocator_spill,
            .detail = "allocator_slot",
            .slot = 0,
            .vreg = 17,
        },
        .{
            .native_start = 20,
            .native_end = 27,
            .kind = .store,
            .base = "rbp",
            .frame_offset = 48,
            .width = 8,
            .origin = .fixed_runtime_frame_state,
            .detail = "incoming_abi_argument",
        },
    };
    const report: Report = .{
        .cwasm_aot_version = 8,
        .compiler_build_id = "test",
        .abi = "sysv",
        .module = 4,
        .local_func = 6145,
        .function_name = "<anon>",
        .module_text_size = 32,
        .module_text_sha256 = "1111111111111111111111111111111111111111111111111111111111111111",
        .function_offset = 0,
        .code_size = 32,
        .normalized_code_sha256 = "0000000000000000000000000000000000000000000000000000000000000000",
        .direct_call_rel32_offsets = &.{},
        .inline_data_ranges = &.{},
        .frame_layout = .{
            .frame_size = 576,
            .local_count = 4,
            .param_count = 1,
            .reserved_vmctx_offset = -8,
            .locals_first_offset = -16,
            .explicit_storage_first_offset = -48,
            .explicit_storage_slots = 64,
            .spill_base = -560,
            .spill_stride = -8,
            .spill_slots = 2,
        },
        .spill_metric = .{
            .slots = 2,
            .spilled_vregs = 1,
            .scalar = 1,
            .v128 = 0,
            .slots_scalar = 1,
            .slots_v128 = 0,
            .spill_ld = 3,
            .spill_st = 1,
            .remat = 0,
            .callee_saved = 2,
        },
        .emitted_allocator_loads = 3,
        .emitted_allocator_stores = 1,
        .allocator_values = &values,
        .accesses = &accesses,
    };

    const json = try serializeAlloc(allocator, report);
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"schema\": \"wamr-aot-frame-attribution\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"frame_offset\": -560") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"vreg\": null") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"vreg_ambiguous\": false") != null);
}
