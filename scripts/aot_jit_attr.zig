const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

pub const AOT_MAGIC: u32 = 0x746F6100; // "\0aot"
pub const AOT_VERSION: u32 = 11;
pub const SUPPORTED_AOT_VERSIONS = [_]u32{ 9, 10, AOT_VERSION };
pub const SEC_TARGET_INFO: u32 = 0;
pub const SEC_TEXT: u32 = 2;
pub const SEC_FUNCTION: u32 = 3;
pub const FRAME_SCHEMA = "wamr-aot-frame-attribution";
pub const FRAME_SCHEMA_VERSION: i64 = 2;
pub const SUPPORTED_FRAME_SCHEMA_VERSIONS = [_]i64{ 1, FRAME_SCHEMA_VERSION };

pub const FRAME_ORIGINS = [_][]const u8{
    "allocator_spill",
    "wasm_local_or_phi",
    "explicit_frame_storage",
    "fixed_runtime_frame_state",
    "unknown",
};

pub const SUPPORTED_ARCHITECTURES = [_][]const u8{
    "x86_64",
    "aarch64",
};

pub const AttributionError = error{
    AttributionFailed,
    OutOfMemory,
    WriteFailed,
};

threadlocal var last_error_buf: [2048]u8 = undefined;
threadlocal var last_error_len: usize = 0;

pub fn lastError() []const u8 {
    return last_error_buf[0..last_error_len];
}

pub fn fail(comptime fmt: []const u8, args: anytype) AttributionError {
    const rendered = std.fmt.bufPrint(&last_error_buf, fmt, args) catch "attribution error";
    last_error_len = rendered.len;
    return error.AttributionFailed;
}

pub const CwasmInfo = struct {
    func_offsets: []const u32,
    text_size: usize,
    text_file_offset: usize,
    data: []const u8,
    version: u32,
    architecture: []const u8,
    abi: ?[]const u8,
    target_format: []const u8,
    target_verified: bool,
};

pub const Instruction = struct {
    address: u64,
    offset: usize,
    size: usize,
    text: []const u8,
};

pub const MmapCandidate = struct {
    base: u64,
    size: u64,
};

pub const TextMappingSelection = struct {
    base: u64,
    size: u64,
    text_size: usize,
    page_size: usize,
    expected_size: usize,
    candidates: []const MmapCandidate,
    authoritative: bool,
    override: ?[]const u8,
};

pub const FrameOperand = struct {
    kind: []const u8,
    base: []const u8,
    offset: ?i64,
    complex_address: bool,

    pub fn eql(self: FrameOperand, other: FrameOperand) bool {
        return std.mem.eql(u8, self.kind, other.kind) and
            std.mem.eql(u8, self.base, other.base) and
            self.offset == other.offset and
            self.complex_address == other.complex_address;
    }
};

pub const FrameLayout = struct {
    frame_pointer: []const u8,
    frame_size: i64,
    local_count: i64,
    param_count: i64,
    reserved_vmctx_offset: i64,
    locals_first_offset: i64,
    explicit_storage_first_offset: i64,
    explicit_storage_slots: i64,
    spill_base: i64,
    spill_stride: i64,
    spill_slots: i64,
};

pub const FrameRegion = struct {
    start: i64,
    end: i64,
    origin: []const u8,
    detail: []const u8,
};

pub const SpillMetric = struct {
    slots: i64,
    spilled_vregs: i64,
    scalar: i64,
    v128: i64,
    slots_scalar: i64,
    slots_v128: i64,
    spill_ld: i64,
    spill_st: i64,
    remat: i64,
    callee_saved: i64,
};

pub const AllocatorValue = struct {
    vreg: i64,
    frame_offset: i64,
    slot: i64,
    slot_count: i64,
    value_type: []const u8,
    live_start: ?i64 = null,
    live_end: ?i64 = null,
    defining_opcode: ?[]const u8 = null,
    source_class: ?[]const u8 = null,
    ir_use_count: i64,
    ir_def_count: i64,
    reload_count: i64,
    store_count: i64,
    rematerialization_eligible: bool,
    reused: bool,
};

pub const InlineDataRange = struct {
    native_start: usize,
    native_end: usize,
    kind: []const u8 = "br_table",
};

pub const NormalizedRelocation = struct {
    native_start: usize,
    native_end: usize,
    kind: []const u8,
};

pub const AccessComponent = struct {
    base: []const u8,
    frame_offset: i64,
    width: i64,
    data_register: ?i64 = null,
    data_register_class: ?[]const u8 = null,
    origin: []const u8,
    detail: ?[]const u8 = null,
    slot: ?i64 = null,
    local_index: ?i64 = null,
    explicit_slot: ?i64 = null,
    vreg: ?i64 = null,
    vreg_ambiguous: bool = false,
    defining_opcode: ?[]const u8 = null,
    source_class: ?[]const u8 = null,
    rematerialization_eligible: ?bool = null,
    ir_position: ?i64 = null,
    ir_opcode: ?[]const u8 = null,
};

pub const FrameAccess = struct {
    native_start: usize,
    native_end: usize,
    kind: []const u8,
    base: []const u8,
    frame_offset: i64,
    width: i64,
    origin: []const u8,
    detail: ?[]const u8 = null,
    slot: ?i64 = null,
    local_index: ?i64 = null,
    explicit_slot: ?i64 = null,
    vreg: ?i64 = null,
    vreg_ambiguous: bool = false,
    defining_opcode: ?[]const u8 = null,
    source_class: ?[]const u8 = null,
    rematerialization_eligible: ?bool = null,
    ir_position: ?i64 = null,
    ir_opcode: ?[]const u8 = null,
    encoded_base: ?[]const u8 = null,
    encoded_offset: ?i64 = null,
    addressing_mode: ?[]const u8 = null,
    components: []const AccessComponent = &.{},
};

pub const Reconciliation = struct {
    emitted_allocator_loads: i64,
    emitted_allocator_stores: i64,
    spill_metric_loads: i64,
    spill_metric_stores: i64,
    matches: bool,
};

pub const FrameMetadata = struct {
    schema: []const u8 = FRAME_SCHEMA,
    schema_version: i64 = 1,
    cwasm_aot_version: u32 = AOT_VERSION,
    compiler_build_id: []const u8 = "",
    architecture: []const u8 = "x86_64",
    abi: []const u8 = "sysv",
    module: i64 = 0,
    local_func: i64 = 0,
    function_name: ?[]const u8 = null,
    module_text_size: usize = 0,
    module_text_sha256: []const u8 = "",
    function_offset: usize = 0,
    code_size: usize = 0,
    normalized_code_sha256: []const u8 = "",
    frame_layout: FrameLayout = .{
        .frame_pointer = "rbp",
        .frame_size = 0,
        .local_count = 0,
        .param_count = 0,
        .reserved_vmctx_offset = 0,
        .locals_first_offset = 0,
        .explicit_storage_first_offset = 0,
        .explicit_storage_slots = 0,
        .spill_base = 0,
        .spill_stride = -8,
        .spill_slots = 0,
    },
    frame_regions: []const FrameRegion = &.{},
    spill_metric: SpillMetric = .{
        .slots = 0,
        .spilled_vregs = 0,
        .scalar = 0,
        .v128 = 0,
        .slots_scalar = 0,
        .slots_v128 = 0,
        .spill_ld = 0,
        .spill_st = 0,
        .remat = 0,
        .callee_saved = 0,
    },
    emitted_allocator_loads: i64 = 0,
    emitted_allocator_stores: i64 = 0,
    allocator_values: []const AllocatorValue = &.{},
    inline_data_ranges: []const InlineDataRange = &.{},
    accesses: []const FrameAccess = &.{},
    reconciliation: Reconciliation = .{
        .emitted_allocator_loads = 0,
        .emitted_allocator_stores = 0,
        .spill_metric_loads = 0,
        .spill_metric_stores = 0,
        .matches = true,
    },

    pub fn accessAtStart(self: *const FrameMetadata, offset: usize) ?*const FrameAccess {
        for (self.accesses) |*access| {
            if (access.native_start == offset) return access;
        }
        return null;
    }

    pub fn valueByVreg(self: *const FrameMetadata, vreg: i64) ?*const AllocatorValue {
        for (self.allocator_values) |*val| {
            if (val.vreg == vreg) return val;
        }
        return null;
    }

    pub fn valuesForSlot(self: *const FrameMetadata, allocator: Allocator, slot: i64) ![]const AllocatorValue {
        var list: std.ArrayList(AllocatorValue) = .empty;
        for (self.allocator_values) |val| {
            if (slot >= val.slot and slot < val.slot + val.slot_count) {
                try list.append(allocator, val);
            }
        }
        return list.toOwnedSlice(allocator);
    }
};

pub fn isSupportedArch(arch: []const u8) bool {
    for (SUPPORTED_ARCHITECTURES) |supported| {
        if (std.mem.eql(u8, arch, supported)) return true;
    }
    return false;
}

pub fn isSupportedOrigin(origin: []const u8) bool {
    for (FRAME_ORIGINS) |supported| {
        if (std.mem.eql(u8, origin, supported)) return true;
    }
    return false;
}

pub fn hostArchitecture() []const u8 {
    return switch (builtin.cpu.arch) {
        .x86_64 => "x86_64",
        .aarch64 => "aarch64",
        else => @tagName(builtin.cpu.arch),
    };
}

pub fn normalizeArchitecture(allocator: Allocator, value: ?[]const u8) AttributionError![]const u8 {
    const raw = value orelse hostArchitecture();
    var lower = try allocator.alloc(u8, raw.len);
    for (raw, 0..) |ch, i| {
        lower[i] = std.ascii.toLower(ch);
    }
    const normalized = if (std.mem.eql(u8, lower, "amd64") or
        std.mem.eql(u8, lower, "x64") or
        std.mem.eql(u8, lower, "x86-64"))
        "x86_64"
    else if (std.mem.eql(u8, lower, "arm64"))
        "aarch64"
    else
        lower;
    if (!isSupportedArch(normalized)) {
        return fail("unsupported disassembly architecture '{s}'; expected x86_64 or aarch64", .{normalized});
    }
    return normalized;
}

const TargetInfo = struct {
    architecture: []const u8,
    abi: ?[]const u8,
    target_format: []const u8,
    target_verified: bool,
};

fn parseTargetInfo(allocator: Allocator, path: []const u8, payload: []const u8) AttributionError!TargetInfo {
    if (payload.len != 40) {
        return fail("{s}: target-info section must be exactly 40 bytes", .{path});
    }
    const bin_type = std.mem.readInt(u16, payload[0..2], .little);
    const abi_type = std.mem.readInt(u16, payload[2..4], .little);
    const e_type = std.mem.readInt(u16, payload[4..6], .little);
    const e_machine = std.mem.readInt(u16, payload[6..8], .little);
    const e_flags = std.mem.readInt(u32, payload[8..12], .little);
    const reserved = std.mem.readInt(u32, payload[12..16], .little);
    const arch_field = payload[16..32];
    const nul_opt = std.mem.indexOfScalar(u8, arch_field, 0);
    const arch_bytes = if (nul_opt) |nul| arch_field[0..nul] else arch_field;
    if (nul_opt) |nul| {
        for (arch_field[nul..]) |b| {
            if (b != 0) {
                return fail("{s}: target-info architecture has nonzero padding", .{path});
            }
        }
    }
    for (arch_bytes) |b| {
        if (b >= 0x80) {
            return fail("{s}: invalid target-info architecture", .{path});
        }
    }
    const architecture = normalizeArchitecture(allocator, arch_bytes) catch {
        return fail("{s}: invalid target-info architecture '{s}'", .{ path, arch_bytes });
    };
    if (reserved != 0) {
        return fail("{s}: target-info reserved field is nonzero", .{path});
    }
    const legacy = (bin_type == 1 and abi_type == 0 and e_type == 0 and e_machine == 0 and e_flags == 0);
    if (legacy) {
        return .{
            .architecture = architecture,
            .abi = null,
            .target_format = "legacy-unspecified",
            .target_verified = false,
        };
    }

    var expected_machine: u16 = 0;
    var abi: []const u8 = "";
    var target_format: []const u8 = "";
    if (std.mem.eql(u8, architecture, "x86_64") and bin_type == 2) {
        expected_machine = 0x3E;
        abi = "sysv";
        target_format = "elf64-little";
    } else if (std.mem.eql(u8, architecture, "x86_64") and bin_type == 6) {
        expected_machine = 0x8664;
        abi = "win64";
        target_format = "coff64";
    } else if (std.mem.eql(u8, architecture, "aarch64") and bin_type == 2) {
        expected_machine = 0xB7;
        abi = "aapcs64";
        target_format = "elf64-little";
    } else {
        return fail("{s}: unsupported target-info architecture/format {s}/{d}", .{ path, architecture, bin_type });
    }

    if (abi_type != 0 or e_type != 1 or e_machine != expected_machine or e_flags != 0) {
        return fail(
            "{s}: inconsistent target-info for {s}/{s} (abi_type={d}, e_type={d}, e_machine=0x{x}, e_flags=0x{x})",
            .{ path, architecture, target_format, abi_type, e_type, e_machine, e_flags },
        );
    }
    return .{
        .architecture = architecture,
        .abi = abi,
        .target_format = target_format,
        .target_verified = true,
    };
}

pub fn parseCwasmBytes(allocator: Allocator, path: []const u8, data: []const u8) AttributionError!CwasmInfo {
    if (data.len < 8) {
        return fail("{s}: truncated .cwasm header", .{path});
    }
    const magic = std.mem.readInt(u32, data[0..4], .little);
    const version = std.mem.readInt(u32, data[4..8], .little);
    if (magic != AOT_MAGIC) {
        return fail("{s}: bad magic 0x{x} (not a .cwasm)", .{ path, magic });
    }
    var version_ok = false;
    for (SUPPORTED_AOT_VERSIONS) |v| {
        if (v == version) {
            version_ok = true;
            break;
        }
    }
    if (!version_ok) {
        return fail("{s}: incompatible aot_version={d}; tool supports [9, 10, 11]", .{ path, version });
    }

    var pos: usize = 8;
    var offsets_opt: ?[]u32 = null;
    var text_size_opt: ?usize = null;
    var text_file_offset_opt: ?usize = null;
    var target_opt: ?TargetInfo = null;

    while (pos < data.len) {
        if (pos + 8 > data.len) {
            return fail("{s}: truncated section header at 0x{x}", .{ path, pos });
        }
        const section_type = std.mem.readInt(u32, data[pos..][0..4], .little);
        const size = std.mem.readInt(u32, data[pos + 4 ..][0..4], .little);
        pos += 8;
        const end = pos + size;
        if (end > data.len) {
            return fail("{s}: section {d} overruns file (0x{x}+0x{x}>0x{x})", .{ path, section_type, pos, size, data.len });
        }
        if (section_type == SEC_TARGET_INFO) {
            if (target_opt != null) {
                return fail("{s}: duplicate target-info section", .{path});
            }
            target_opt = try parseTargetInfo(allocator, path, data[pos..end]);
        } else if (section_type == SEC_TEXT) {
            if (text_size_opt != null) {
                return fail("{s}: duplicate text section", .{path});
            }
            text_file_offset_opt = pos;
            text_size_opt = size;
        } else if (section_type == SEC_FUNCTION) {
            if (offsets_opt != null) {
                return fail("{s}: duplicate function section", .{path});
            }
            if (size < 4) {
                return fail("{s}: truncated function section", .{path});
            }
            const count = std.mem.readInt(u32, data[pos..][0..4], .little);
            const needed: usize = 4 + @as(usize, count) * 8;
            if (needed > size) {
                return fail("{s}: function section declares {d} entries but has only {d} bytes", .{ path, count, size });
            }
            const offsets = try allocator.alloc(u32, count);
            var i: usize = 0;
            while (i < count) : (i += 1) {
                const entry_pos = pos + 4 + i * 8;
                offsets[i] = std.mem.readInt(u32, data[entry_pos..][0..4], .little);
            }
            offsets_opt = offsets;
        }
        pos = end;
    }

    const target = target_opt orelse return fail("{s}: missing target-info section", .{path});
    const offsets = offsets_opt orelse return fail("{s}: missing function/text section", .{path});
    const text_size = text_size_opt orelse return fail("{s}: missing function/text section", .{path});
    const text_file_offset = text_file_offset_opt orelse return fail("{s}: missing function/text section", .{path});

    for (offsets, 0..) |offset, index| {
        if (offset >= text_size) {
            return fail("{s}: function {d} offset 0x{x} is outside text_size=0x{x}", .{ path, index, offset, text_size });
        }
        if (index > 0 and offset <= offsets[index - 1]) {
            return fail(
                "{s}: ambiguous/non-increasing function offsets at {d}/{d}: 0x{x}, 0x{x}",
                .{ path, index - 1, index, offsets[index - 1], offset },
            );
        }
    }

    return .{
        .func_offsets = offsets,
        .text_size = text_size,
        .text_file_offset = text_file_offset,
        .data = data,
        .version = version,
        .architecture = target.architecture,
        .abi = target.abi,
        .target_format = target.target_format,
        .target_verified = target.target_verified,
    };
}

pub fn parseCwasm(allocator: Allocator, io: std.Io, path: []const u8) AttributionError!CwasmInfo {
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024)) catch |err| {
        return fail("{s}: cannot read: {s}", .{ path, @errorName(err) });
    };
    return parseCwasmBytes(allocator, path, data);
}

pub const FunctionBounds = struct {
    start: usize,
    end: usize,
};

pub fn functionBounds(info: CwasmInfo, func_index: i64) AttributionError!FunctionBounds {
    if (func_index < 0 or @as(usize, @intCast(func_index)) >= info.func_offsets.len) {
        const max_idx: i64 = @as(i64, @intCast(info.func_offsets.len)) - 1;
        return fail("--func {d} out of range (0..{d})", .{ func_index, max_idx });
    }
    const idx: usize = @intCast(func_index);
    const start: usize = info.func_offsets[idx];
    const end: usize = if (idx + 1 < info.func_offsets.len) info.func_offsets[idx + 1] else info.text_size;
    if (end <= start) {
        return fail("local_func={d} has an empty or ambiguous native range", .{func_index});
    }
    return .{ .start = start, .end = end };
}

pub fn expectedTextMappingSize(text_size: usize, page_size: usize) AttributionError!usize {
    if (text_size == 0) {
        return fail("cwasm text size must be positive", .{});
    }
    if (page_size == 0 or (page_size & (page_size - 1)) != 0) {
        return fail("invalid host page size {d}", .{page_size});
    }
    return (text_size + page_size - 1) & ~(page_size - 1);
}

fn renderMappings(allocator: Allocator, maps: []const MmapCandidate) Allocator.Error![]const u8 {
    if (maps.len == 0) return try allocator.dupe(u8, "none");
    var out: std.ArrayList(u8) = .empty;
    for (maps, 0..) |m, idx| {
        if (idx > 0) try out.appendSlice(allocator, ", ");
        const part = try std.fmt.allocPrint(allocator, "0x{x}/0x{x}", .{ m.base, m.size });
        try out.appendSlice(allocator, part);
    }
    return out.toOwnedSlice(allocator);
}

pub fn selectTextMappingFromCandidates(
    allocator: Allocator,
    maps: []const MmapCandidate,
    text_size: usize,
    explicit_base: ?[]const u8,
    page_size: usize,
) AttributionError!TextMappingSelection {
    const expected_size = try expectedTextMappingSize(text_size, page_size);
    if (explicit_base) |eb| {
        const trimmed = if (std.mem.startsWith(u8, eb, "0x") or std.mem.startsWith(u8, eb, "0X")) eb[2..] else eb;
        const base = std.fmt.parseInt(u64, trimmed, 16) catch {
            return fail("manual --base must be hexadecimal, got '{s}'", .{eb});
        };
        var matches: std.ArrayList(MmapCandidate) = .empty;
        for (maps) |m| {
            if (m.base == base) try matches.append(allocator, m);
        }
        if (matches.items.len != 1) {
            const rendered = try renderMappings(allocator, maps);
            return fail(
                "manual --base 0x{x} must identify exactly one anonymous executable mmap; candidates: {s}",
                .{ base, rendered },
            );
        }
        return .{
            .base = matches.items[0].base,
            .size = matches.items[0].size,
            .text_size = text_size,
            .page_size = page_size,
            .expected_size = expected_size,
            .candidates = maps,
            .authoritative = false,
            .override = eb,
        };
    }

    var matches: std.ArrayList(MmapCandidate) = .empty;
    for (maps) |m| {
        if (m.size == expected_size) try matches.append(allocator, m);
    }
    if (matches.items.len != 1) {
        const rendered = try renderMappings(allocator, maps);
        return fail(
            "expected exactly one anonymous executable mmap of page-rounded size 0x{x} for text_size=0x{x} and page_size=0x{x}; found {d} exact matches among candidates: {s}",
            .{ expected_size, text_size, page_size, matches.items.len, rendered },
        );
    }
    return .{
        .base = matches.items[0].base,
        .size = matches.items[0].size,
        .text_size = text_size,
        .page_size = page_size,
        .expected_size = expected_size,
        .candidates = maps,
        .authoritative = true,
        .override = null,
    };
}

pub fn perfBinary() []const u8 {
    if (builtin.os.tag != .windows) {
        if (std.c.getenv("PERF")) |val| {
            const slice = std.mem.span(val);
            if (slice.len > 0) return slice;
        }
    }
    return "perf";
}

pub fn runChecked(allocator: Allocator, io: std.Io, argv: []const []const u8, what: []const u8) AttributionError![]u8 {
    const res = std.process.run(allocator, io, .{
        .argv = argv,
        .stdout_limit = .limited(64 * 1024 * 1024),
        .stderr_limit = .limited(8 * 1024 * 1024),
    }) catch |err| {
        return fail("{s} failed: {s}", .{ what, @errorName(err) });
    };
    switch (res.term) {
        .exited => |code| if (code == 0) {
            allocator.free(res.stderr);
            return res.stdout;
        },
        else => {},
    }
    const detail_err = std.mem.trim(u8, res.stderr, " \t\r\n");
    const detail_out = std.mem.trim(u8, res.stdout, " \t\r\n");
    const detail = if (detail_err.len > 0) detail_err else if (detail_out.len > 0) detail_out else "nonzero exit";
    return fail("{s} failed: {s}", .{ what, detail });
}

pub fn parseJitExecMmapsOutput(allocator: Allocator, output: []const u8) ![]MmapCandidate {
    var list: std.ArrayList(MmapCandidate) = .empty;
    var lines = std.mem.splitScalar(u8, output, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        // Look for `[0x<base>(0x<size>)...]: r-xp //anon` or `rwxp //anon`
        const anon_pos = std.mem.indexOf(u8, line, "]: r") orelse continue;
        const tail = line[anon_pos..];
        if (!std.mem.startsWith(u8, tail, "]: r-xp //anon") and
            !std.mem.startsWith(u8, tail, "]: rwxp //anon"))
        {
            continue;
        }
        const open_br = std.mem.lastIndexOfScalar(u8, line[0..anon_pos], '[') orelse continue;
        const inner = line[open_br + 1 .. anon_pos];
        if (!std.mem.startsWith(u8, inner, "0x")) continue;
        const open_paren = std.mem.indexOfScalar(u8, inner, '(') orelse continue;
        const close_paren = std.mem.indexOfScalarPos(u8, inner, open_paren + 1, ')') orelse continue;
        const base_hex = inner[2..open_paren];
        const size_part = inner[open_paren + 1 .. close_paren];
        if (!std.mem.startsWith(u8, size_part, "0x")) continue;
        const size_hex = size_part[2..];
        const base = std.fmt.parseInt(u64, base_hex, 16) catch continue;
        const size = std.fmt.parseInt(u64, size_hex, 16) catch continue;
        var exists = false;
        for (list.items) |existing| {
            if (existing.base == base and existing.size == size) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            try list.append(allocator, .{ .base = base, .size = size });
        }
    }
    const items = try list.toOwnedSlice(allocator);
    std.mem.sort(MmapCandidate, items, {}, struct {
        fn lessThan(_: void, a: MmapCandidate, b: MmapCandidate) bool {
            if (a.size != b.size) return a.size > b.size;
            return a.base < b.base;
        }
    }.lessThan);
    return items;
}

pub fn jitExecMmaps(allocator: Allocator, io: std.Io, perf: []const u8) AttributionError![]MmapCandidate {
    const argv = [_][]const u8{ perfBinary(), "script", "-i", perf, "--show-mmap-events" };
    const output = try runChecked(allocator, io, &argv, "perf script --show-mmap-events");
    defer allocator.free(output);
    return try parseJitExecMmapsOutput(allocator, output);
}

pub const AddrCounts = struct {
    counts: std.AutoHashMapUnmanaged(u64, u64),
    total: u64,
};

pub fn parsePerfReportOutput(allocator: Allocator, output: []const u8) !AddrCounts {
    var counts: std.AutoHashMapUnmanaged(u64, u64) = .empty;
    var total: u64 = 0;
    var lines = std.mem.splitScalar(u8, output, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        // Format: `<pct>%  <pct>%  <count>  <rest>`
        var tok_it = std.mem.tokenizeAny(u8, line, " \t");
        const first = tok_it.next() orelse continue;
        if (!std.mem.endsWith(u8, first, "%")) continue;
        _ = std.fmt.parseFloat(f64, first[0 .. first.len - 1]) catch continue;
        const second = tok_it.next() orelse continue;
        if (!std.mem.endsWith(u8, second, "%")) continue;
        _ = std.fmt.parseFloat(f64, second[0 .. second.len - 1]) catch continue;
        const count_tok = tok_it.next() orelse continue;
        const count = std.fmt.parseInt(u64, count_tok, 10) catch continue;
        const rest = tok_it.rest();
        total += count;
        if (std.mem.indexOf(u8, rest, "[JIT]") == null) continue;
        if (std.mem.indexOf(u8, rest, "[.]")) |dot_idx| {
            const after_dot = std.mem.trimStart(u8, rest[dot_idx + 3 ..], " \t");
            if (std.mem.startsWith(u8, after_dot, "0x") or std.mem.startsWith(u8, after_dot, "0X")) {
                var end_hex: usize = 2;
                while (end_hex < after_dot.len and std.ascii.isHex(after_dot[end_hex])) : (end_hex += 1) {}
                if (end_hex > 2) {
                    if (std.fmt.parseInt(u64, after_dot[2..end_hex], 16)) |ip| {
                        const gop = try counts.getOrPut(allocator, ip);
                        if (!gop.found_existing) gop.value_ptr.* = 0;
                        gop.value_ptr.* += count;
                    } else |_| {}
                }
            }
        }
    }
    return .{ .counts = counts, .total = total };
}

pub fn parsePerfScriptOutput(allocator: Allocator, script: []const u8) !AddrCounts {
    var counts: std.AutoHashMapUnmanaged(u64, u64) = .empty;
    var total: u64 = 0;
    var lines = std.mem.splitScalar(u8, script, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) continue;
        var tok_it = std.mem.tokenizeAny(u8, line, " \t");
        const first = tok_it.next() orelse continue;
        if (tok_it.rest().len == 0) continue;
        const hex_str = if (std.mem.startsWith(u8, first, "0x") or std.mem.startsWith(u8, first, "0X"))
            first[2..]
        else
            first;
        if (hex_str.len == 0) continue;
        var all_hex = true;
        for (hex_str) |c| {
            if (!std.ascii.isHex(c)) {
                all_hex = false;
                break;
            }
        }
        if (!all_hex) continue;
        const ip = std.fmt.parseInt(u64, hex_str, 16) catch continue;
        total += 1;
        const gop = try counts.getOrPut(allocator, ip);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }
    return .{ .counts = counts, .total = total };
}

pub fn addrCounts(allocator: Allocator, io: std.Io, perf: []const u8) AttributionError!AddrCounts {
    const report_argv = [_][]const u8{
        perfBinary(),
        "report",
        "-i",
        perf,
        "--stdio",
        "-g",
        "none",
        "-n",
        "--sort=dso,symbol",
        "--percent-limit",
        "0",
    };
    const report_out = try runChecked(allocator, io, &report_argv, "perf report");
    defer allocator.free(report_out);
    var parsed = try parsePerfReportOutput(allocator, report_out);
    if (parsed.counts.count() > 0 and parsed.total > 0) {
        return parsed;
    }
    parsed.counts.deinit(allocator);

    const script_argv = [_][]const u8{
        perfBinary(),
        "script",
        "-i",
        perf,
        "-F",
        "ip,dso",
    };
    const script_out = try runChecked(allocator, io, &script_argv, "perf script self samples");
    defer allocator.free(script_out);
    return try parsePerfScriptOutput(allocator, script_out);
}

pub fn systemPageSize() usize {
    return std.heap.pageSize();
}

pub fn selectTextMapping(
    allocator: Allocator,
    io: std.Io,
    perf: []const u8,
    text_size: usize,
    explicit_base: ?[]const u8,
    page_size_opt: ?usize,
) AttributionError!TextMappingSelection {
    const page_size = page_size_opt orelse systemPageSize();
    const maps = try jitExecMmaps(allocator, io, perf);
    return selectTextMappingFromCandidates(allocator, maps, text_size, explicit_base, page_size);
}

pub fn percent(numerator: u64, denominator: u64) f64 {
    if (denominator == 0) return 0.0;
    return 100.0 * @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(denominator));
}

pub fn requireAttributionCoverage(attributed: u64, total: u64, minimum_pct: f64) AttributionError!f64 {
    const cov = percent(attributed, total);
    if (cov < minimum_pct) {
        return fail(
            "cwasm attribution coverage {d:.4}% ({d}/{d}) is below required {d:.4}%",
            .{ cov, attributed, total, minimum_pct },
        );
    }
    return cov;
}

pub fn parseObjdumpOutput(
    allocator: Allocator,
    output: []const u8,
    virtual_address: u64,
    function_offset_base: usize,
) AttributionError![]Instruction {
    var list: std.ArrayList(Instruction) = .empty;
    var lines = std.mem.splitScalar(u8, output, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) continue;
        const colon_idx = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const addr_str = std.mem.trim(u8, line[0..colon_idx], " \t");
        if (addr_str.len == 0) continue;
        var addr_hex = true;
        for (addr_str) |c| {
            if (!std.ascii.isHex(c)) {
                addr_hex = false;
                break;
            }
        }
        if (!addr_hex) continue;
        const address = std.fmt.parseInt(u64, addr_str, 16) catch continue;
        const after_colon = std.mem.trimStart(u8, line[colon_idx + 1 ..], " \t");
        if (after_colon.len == 0) continue;

        // Parse encoded hex tokens (2-char or 8-char hex tokens) followed by mnemonic/text.
        // Note: objdump separates encoded bytes from mnemonic with tabs or multiple spaces,
        // or each token is 2 or 8 hex chars until a token is not 2/8 hex chars or after tab.
        var size: usize = 0;
        var pos: usize = 0;
        while (pos < after_colon.len) {
            while (pos < after_colon.len and after_colon[pos] == ' ') : (pos += 1) {}
            if (pos >= after_colon.len or after_colon[pos] == '\t') {
                while (pos < after_colon.len and (after_colon[pos] == ' ' or after_colon[pos] == '\t')) : (pos += 1) {}
                break;
            }
            var tok_end = pos;
            while (tok_end < after_colon.len and after_colon[tok_end] != ' ' and after_colon[tok_end] != '\t') : (tok_end += 1) {}
            const tok = after_colon[pos..tok_end];
            if (tok.len != 2 and tok.len != 8) break;
            var tok_hex = true;
            for (tok) |c| {
                if (!std.ascii.isHex(c)) {
                    tok_hex = false;
                    break;
                }
            }
            if (!tok_hex) break;
            // Also check if this 2-char token is actually the start of the instruction text
            // when there was no tab (e.g. `1000:  8b010000  add x0, x0, x1` where `add` is 3 chars,
            // or `b 0x1000` where `b` is 1 char, or `bl` which is not hex, or `ja` / `jb` / `je` / `jl` / `jg` / `js` / `jp` / `jo` / `br` — none of which are valid hex because j/l/g/s/p/o/r are not hex!
            // What about `bt`? t is not hex! Is there ANY 2-letter x86/arm mnemonic made only of a-f?
            // `in` (n not hex), `or` (r not hex), `b` (len 1). None!
            size += tok.len / 2;
            pos = tok_end;
            if (pos < after_colon.len and after_colon[pos] == '\t') {
                while (pos < after_colon.len and (after_colon[pos] == ' ' or after_colon[pos] == '\t')) : (pos += 1) {}
                break;
            }
        }
        if (size == 0) continue;
        const text = std.mem.trim(u8, after_colon[pos..], " \t");
        const offset = function_offset_base + @as(usize, @intCast(address - virtual_address));
        try list.append(allocator, .{
            .address = address,
            .offset = offset,
            .size = size,
            .text = try allocator.dupe(u8, text),
        });
    }
    return try list.toOwnedSlice(allocator);
}

pub fn buildObjdumpArgv(
    allocator: Allocator,
    virtual_address: u64,
    scratch_path: []const u8,
    architecture: ?[]const u8,
) AttributionError![]const []const u8 {
    const arch = try normalizeArchitecture(allocator, architecture);
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.appendSlice(allocator, &.{ "objdump", "-D", "-b", "binary" });
    if (std.mem.eql(u8, arch, "x86_64")) {
        try argv.appendSlice(allocator, &.{ "-m", "i386:x86-64", "-M", "intel" });
    } else {
        try argv.appendSlice(allocator, &.{ "-m", "aarch64" });
    }
    try argv.append(allocator, try std.fmt.allocPrint(allocator, "--adjust-vma=0x{x}", .{virtual_address}));
    try argv.append(allocator, scratch_path);
    return try argv.toOwnedSlice(allocator);
}

pub fn disassembleBlob(
    allocator: Allocator,
    io: std.Io,
    blob: []const u8,
    virtual_address: u64,
    scratch_dir: []const u8,
    label: []const u8,
    function_offset_base: usize,
    architecture: ?[]const u8,
) AttributionError![]Instruction {
    const scratch_path = try std.fmt.allocPrint(allocator, "{s}/.aot-jit-attr-{s}.bin", .{ scratch_dir, label });
    const argv = try buildObjdumpArgv(allocator, virtual_address, scratch_path, architecture);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = scratch_path, .data = blob }) catch |err| {
        return fail("write scratch failed: {s}", .{@errorName(err)});
    };
    defer std.Io.Dir.cwd().deleteFile(io, scratch_path) catch {};

    const output = try runChecked(allocator, io, argv, "objdump");
    defer allocator.free(output);
    const instructions = try parseObjdumpOutput(allocator, output, virtual_address, function_offset_base);
    if (instructions.len == 0 and blob.len > 0) {
        return fail("objdump returned no instructions for selected function", .{});
    }
    return instructions;
}

pub const BlobDisassemblerFn = *const fn (
    allocator: Allocator,
    io: std.Io,
    blob: []const u8,
    virtual_address: u64,
    scratch_dir: []const u8,
    label: []const u8,
    function_offset_base: usize,
    architecture: ?[]const u8,
) AttributionError![]Instruction;

pub fn disassembleFunctionWith(
    allocator: Allocator,
    io: std.Io,
    function_code: []const u8,
    function_base: u64,
    scratch_dir: []const u8,
    label: []const u8,
    inline_data_ranges: []const InlineDataRange,
    architecture: ?[]const u8,
    disassembler: BlobDisassemblerFn,
) AttributionError![]Instruction {
    var instructions: std.ArrayList(Instruction) = .empty;
    var cursor: usize = 0;
    for (inline_data_ranges, 0..) |data_range, index| {
        const start = data_range.native_start;
        const end = data_range.native_end;
        if (cursor < start) {
            const seg_label = try std.fmt.allocPrint(allocator, "{s}-segment-{d}", .{ label, index });
            const sub = try disassembler(
                allocator,
                io,
                function_code[cursor..start],
                function_base + cursor,
                scratch_dir,
                seg_label,
                cursor,
                architecture,
            );
            try instructions.appendSlice(allocator, sub);
        }
        cursor = end;
    }
    if (cursor < function_code.len) {
        const tail_label = try std.fmt.allocPrint(allocator, "{s}-segment-tail", .{label});
        const sub = try disassembler(
            allocator,
            io,
            function_code[cursor..],
            function_base + cursor,
            scratch_dir,
            tail_label,
            cursor,
            architecture,
        );
        try instructions.appendSlice(allocator, sub);
    }
    return try instructions.toOwnedSlice(allocator);
}

pub fn disassembleFunction(
    allocator: Allocator,
    io: std.Io,
    function_code: []const u8,
    function_base: u64,
    scratch_dir: []const u8,
    label: []const u8,
    inline_data_ranges: []const InlineDataRange,
    architecture: ?[]const u8,
) AttributionError![]Instruction {
    return disassembleFunctionWith(
        allocator,
        io,
        function_code,
        function_base,
        scratch_dir,
        label,
        inline_data_ranges,
        architecture,
        disassembleBlob,
    );
}

pub const SplitOperands = struct {
    mnemonic: []const u8,
    operands: []const []const u8,
};

pub fn splitOperands(allocator: Allocator, text: []const u8) !SplitOperands {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return .{ .mnemonic = "", .operands = &.{} };
    var split_idx: usize = 0;
    while (split_idx < trimmed.len and !std.ascii.isWhitespace(trimmed[split_idx])) : (split_idx += 1) {}
    const mnemonic_raw = trimmed[0..split_idx];
    const mnemonic = try allocator.alloc(u8, mnemonic_raw.len);
    for (mnemonic_raw, 0..) |c, i| mnemonic[i] = std.ascii.toLower(c);
    if (split_idx >= trimmed.len) {
        return .{ .mnemonic = mnemonic, .operands = &.{} };
    }
    const rest = std.mem.trimStart(u8, trimmed[split_idx..], " \t");
    var operands: std.ArrayList([]const u8) = .empty;
    var current: std.ArrayList(u8) = .empty;
    var depth: usize = 0;
    for (rest) |ch| {
        if (ch == '[') {
            depth += 1;
        } else if (ch == ']') {
            if (depth > 0) depth -= 1;
        }
        if (ch == ',' and depth == 0) {
            const op = std.mem.trim(u8, current.items, " \t");
            try operands.append(allocator, try allocator.dupe(u8, op));
            current.clearRetainingCapacity();
        } else {
            try current.append(allocator, ch);
        }
    }
    if (current.items.len > 0) {
        const op = std.mem.trim(u8, current.items, " \t");
        try operands.append(allocator, try allocator.dupe(u8, op));
    }
    return .{
        .mnemonic = mnemonic,
        .operands = try operands.toOwnedSlice(allocator),
    };
}

fn isIdentChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn findStandaloneBase(expr: []const u8, needle: []const u8) bool {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, expr, pos, needle)) |idx| {
        const before_ok = (idx == 0) or !isIdentChar(expr[idx - 1]);
        const end = idx + needle.len;
        const after_ok = (end == expr.len) or !isIdentChar(expr[end]);
        if (before_ok and after_ok) return true;
        pos = idx + 1;
    }
    return false;
}

fn parseUnsignedIntAuto(str: []const u8) ?i64 {
    if (str.len == 0) return null;
    if (std.mem.startsWith(u8, str, "0x") or std.mem.startsWith(u8, str, "0X")) {
        if (str.len <= 2) return null;
        for (str[2..]) |c| {
            if (!std.ascii.isHex(c)) return null;
        }
        return std.fmt.parseInt(i64, str[2..], 16) catch null;
    }
    for (str) |c| {
        if (!std.ascii.isDigit(c)) return null;
    }
    return std.fmt.parseInt(i64, str, 10) catch null;
}

fn parseSignedIntAuto(str: []const u8) ?i64 {
    if (str.len == 0) return null;
    if (str[0] == '-') {
        const mag = parseUnsignedIntAuto(str[1..]) orelse return null;
        return -mag;
    } else if (str[0] == '+') {
        return parseUnsignedIntAuto(str[1..]);
    }
    return parseUnsignedIntAuto(str);
}

pub fn parseFrameOperand(allocator: Allocator, text: []const u8) !?FrameOperand {
    const split = try splitOperands(allocator, text);
    if (std.mem.eql(u8, split.mnemonic, "push") and split.operands.len > 0 and std.mem.indexOfScalar(u8, split.operands[0], '[') == null) {
        return .{ .kind = "store", .base = "rsp", .offset = -8, .complex_address = false };
    }
    if (std.mem.eql(u8, split.mnemonic, "pop") and split.operands.len > 0 and std.mem.indexOfScalar(u8, split.operands[0], '[') == null) {
        return .{ .kind = "load", .base = "rsp", .offset = 0, .complex_address = false };
    }
    if (std.mem.startsWith(u8, split.mnemonic, "lea")) {
        return null;
    }
    for (split.operands, 0..) |operand, operand_index| {
        const open_br = std.mem.indexOfScalar(u8, operand, '[') orelse continue;
        const close_br = std.mem.indexOfScalarPos(u8, operand, open_br + 1, ']') orelse continue;
        const inner = operand[open_br + 1 .. close_br];
        var expr_buf: std.ArrayList(u8) = .empty;
        for (inner) |c| {
            if (!std.ascii.isWhitespace(c)) {
                try expr_buf.append(allocator, std.ascii.toLower(c));
            }
        }
        const expr = expr_buf.items;
        const has_rbp = findStandaloneBase(expr, "rbp");
        const has_rsp = findStandaloneBase(expr, "rsp");
        if (!has_rbp and !has_rsp) continue;
        const base = if (has_rbp) "rbp" else "rsp";

        var offset: ?i64 = null;
        var complex_address = true;
        if (std.mem.startsWith(u8, expr, base)) {
            const tail = expr[base.len..];
            if (tail.len == 0) {
                offset = 0;
                complex_address = false;
            } else if (tail[0] == '+' or tail[0] == '-') {
                if (parseUnsignedIntAuto(tail[1..])) |mag| {
                    offset = if (tail[0] == '-') -mag else mag;
                    complex_address = false;
                }
            }
        }
        const kind = if (std.mem.startsWith(u8, split.mnemonic, "mov") and operand_index == 0) "store" else "load";
        return .{
            .kind = kind,
            .base = base,
            .offset = offset,
            .complex_address = complex_address,
        };
    }
    return null;
}

fn isAarch64LoadMnemonic(m: []const u8) bool {
    const loads = [_][]const u8{
        "ldr",  "ldrb",  "ldrh",  "ldrsb",  "ldrsh",  "ldrsw",
        "ldur", "ldurb", "ldurh", "ldursb", "ldursh", "ldursw",
        "ldp",  "ldpsw",
    };
    for (loads) |item| {
        if (std.mem.eql(u8, m, item)) return true;
    }
    return false;
}

fn isAarch64StoreMnemonic(m: []const u8) bool {
    const stores = [_][]const u8{
        "str", "strb", "strh", "stur", "sturb", "sturh", "stp",
    };
    for (stores) |item| {
        if (std.mem.eql(u8, m, item)) return true;
    }
    return false;
}

fn isAarch64BaseReg(reg: []const u8, frame_only: bool) bool {
    if (std.mem.eql(u8, reg, "x29") or std.mem.eql(u8, reg, "fp") or std.mem.eql(u8, reg, "sp")) {
        return true;
    }
    if (frame_only) return false;
    if (reg.len >= 2 and reg[0] == 'x') {
        const num = std.fmt.parseInt(u32, reg[1..], 10) catch return false;
        return num <= 30;
    }
    return false;
}

pub fn parseAarch64MemoryOperand(allocator: Allocator, text: []const u8, frame_only: bool) !?FrameOperand {
    const split = try splitOperands(allocator, text);
    const is_load = isAarch64LoadMnemonic(split.mnemonic);
    const is_store = isAarch64StoreMnemonic(split.mnemonic);
    if (!is_load and !is_store) return null;

    for (split.operands) |operand| {
        const open_br = std.mem.indexOfScalar(u8, operand, '[') orelse continue;
        const close_br = std.mem.indexOfScalarPos(u8, operand, open_br + 1, ']') orelse operand.len;
        const inner_raw = operand[open_br + 1 .. close_br];
        var inner_lower = try allocator.alloc(u8, inner_raw.len);
        for (inner_raw, 0..) |c, i| inner_lower[i] = std.ascii.toLower(c);
        const inner = std.mem.trim(u8, inner_lower, " \t");
        if (inner.len == 0) continue;

        var reg_end: usize = 0;
        while (reg_end < inner.len and isIdentChar(inner[reg_end])) : (reg_end += 1) {}
        const reg_tok = inner[0..reg_end];
        if (!isAarch64BaseReg(reg_tok, frame_only)) continue;

        const base = if (std.mem.eql(u8, reg_tok, "x29") or std.mem.eql(u8, reg_tok, "fp"))
            "x29"
        else if (std.mem.eql(u8, reg_tok, "sp"))
            "sp"
        else
            reg_tok;

        const after_reg = std.mem.trim(u8, inner[reg_end..], " \t");
        var offset: ?i64 = 0;
        var complex_address = false;
        if (after_reg.len > 0) {
            if (after_reg[0] == ',') {
                var disp_str = std.mem.trim(u8, after_reg[1..], " \t");
                if (disp_str.len > 0 and disp_str[0] == '#') {
                    disp_str = std.mem.trimStart(u8, disp_str[1..], " \t");
                }
                // Strip any trailing `, ...` or `!` if present, or check if disp_str is a pure int
                if (parseSignedIntAuto(disp_str)) |parsed_off| {
                    offset = parsed_off;
                    complex_address = false;
                } else {
                    offset = null;
                    complex_address = true;
                }
            }
        }
        return .{
            .kind = if (is_load) "load" else "store",
            .base = base,
            .offset = if (complex_address) null else offset,
            .complex_address = complex_address,
        };
    }
    return null;
}

pub fn parseAarch64FrameOperand(allocator: Allocator, text: []const u8) !?FrameOperand {
    return parseAarch64MemoryOperand(allocator, text, true);
}

pub fn instructionFrameOperand(allocator: Allocator, text: []const u8, architecture: ?[]const u8) AttributionError!?FrameOperand {
    const arch = try normalizeArchitecture(allocator, architecture);
    if (std.mem.eql(u8, arch, "x86_64")) {
        return try parseFrameOperand(allocator, text);
    }
    return try parseAarch64FrameOperand(allocator, text);
}

pub fn sha256Hex(allocator: Allocator, bytes: []const u8) ![]const u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    return try allocator.dupe(u8, &hex);
}

pub fn normalizedCodeSha256(allocator: Allocator, code: []const u8, rel32_offsets: []const i64) AttributionError![]const u8 {
    var normalized = try allocator.dupe(u8, code);
    defer allocator.free(normalized);
    var prior_end: usize = 0;
    for (rel32_offsets) |offset_signed| {
        if (offset_signed < 0) {
            return fail("invalid/overlapping direct-call rel32 offset {d}", .{offset_signed});
        }
        const offset: usize = @intCast(offset_signed);
        if (offset < prior_end or offset + 4 > normalized.len) {
            return fail("invalid/overlapping direct-call rel32 offset {d}", .{offset_signed});
        }
        @memset(normalized[offset .. offset + 4], 0);
        prior_end = offset + 4;
    }
    return try sha256Hex(allocator, normalized);
}

pub fn normalizedCodeSha256V2(
    allocator: Allocator,
    code: []const u8,
    relocations: []const NormalizedRelocation,
    architecture: []const u8,
) AttributionError![]const u8 {
    var normalized = try allocator.dupe(u8, code);
    defer allocator.free(normalized);
    var prior_end: usize = 0;
    for (relocations) |reloc| {
        const start = reloc.native_start;
        const end = reloc.native_end;
        const kind = reloc.kind;
        if (start < prior_end or end != start + 4 or end > normalized.len) {
            return fail("invalid/overlapping normalized relocation [{d},{d})", .{ start, end });
        }
        if (std.mem.eql(u8, architecture, "aarch64")) {
            const expected_opcode: u32 = if (std.mem.eql(u8, kind, "aarch64_call_imm26"))
                0x94000000
            else if (std.mem.eql(u8, kind, "aarch64_tail_call_imm26"))
                0x14000000
            else
                return fail("invalid AArch64 relocation kind '{s}'", .{kind});

            const word = std.mem.readInt(u32, normalized[start..][0..4], .little);
            if ((word & 0xFC000000) != expected_opcode) {
                return fail("AArch64 relocation at {d} does not match {s}", .{ start, kind });
            }
            std.mem.writeInt(u32, normalized[start..][0..4], word & 0xFC000000, .little);
        } else if (std.mem.eql(u8, architecture, "x86_64")) {
            if (!std.mem.eql(u8, kind, "x86_64_call_rel32") or start == 0) {
                return fail("invalid x86_64 relocation kind/range at {d}", .{start});
            }
            if (normalized[start - 1] != 0xE8) {
                return fail("x86_64 relocation at {d} is not CALL rel32", .{start});
            }
            @memset(normalized[start..end], 0);
        } else {
            return fail("unsupported metadata architecture '{s}'", .{architecture});
        }
        prior_end = end;
    }
    return try sha256Hex(allocator, normalized);
}

fn signedBits(value: u32, bits: u5) i64 {
    const sign: u32 = @as(u32, 1) << (bits - 1);
    const v_signed: i64 = @intCast(value ^ sign);
    const s_signed: i64 = @intCast(sign);
    return v_signed - s_signed;
}

const DecodedReg = struct {
    reg: i64,
    reg_class: []const u8,
};

const DecodedAarch64Mem = struct {
    kind: []const u8,
    rn: u32,
    offset: i64,
    mode: []const u8,
    width: i64,
    pair: bool,
    registers: [2]DecodedReg,
    register_count: usize,
    load_gprs: [2]u32,
    load_gpr_count: usize,
    writeback: ?i64,
};

fn decodeAarch64Memory(word: u32) ?DecodedAarch64Mem {
    const rn: u32 = (word >> 5) & 31;
    const rt: u32 = word & 31;
    const top: u32 = word & 0xFFC00000;

    const UnsignedEntry = struct { top: u32, kind: []const u8, width: i64, scale: i64 };
    const unsigned_table = [_]UnsignedEntry{
        .{ .top = 0xF9400000, .kind = "load", .width = 8, .scale = 8 },
        .{ .top = 0xF9000000, .kind = "store", .width = 8, .scale = 8 },
        .{ .top = 0xB9400000, .kind = "load", .width = 4, .scale = 4 },
        .{ .top = 0xB9800000, .kind = "load", .width = 4, .scale = 4 },
        .{ .top = 0xB9000000, .kind = "store", .width = 4, .scale = 4 },
        .{ .top = 0x39400000, .kind = "load", .width = 1, .scale = 1 },
        .{ .top = 0x39800000, .kind = "load", .width = 1, .scale = 1 },
        .{ .top = 0x39C00000, .kind = "load", .width = 1, .scale = 1 },
        .{ .top = 0x39000000, .kind = "store", .width = 1, .scale = 1 },
        .{ .top = 0x79400000, .kind = "load", .width = 2, .scale = 2 },
        .{ .top = 0x79800000, .kind = "load", .width = 2, .scale = 2 },
        .{ .top = 0x79C00000, .kind = "load", .width = 2, .scale = 2 },
        .{ .top = 0x79000000, .kind = "store", .width = 2, .scale = 2 },
    };
    for (unsigned_table) |u| {
        if (top == u.top) {
            const is_load = std.mem.eql(u8, u.kind, "load");
            return .{
                .kind = u.kind,
                .rn = rn,
                .offset = @as(i64, @intCast((word >> 10) & 0xFFF)) * u.scale,
                .mode = "unsigned_scaled",
                .width = u.width,
                .pair = false,
                .registers = .{ .{ .reg = @intCast(rt), .reg_class = "gpr" }, .{ .reg = 0, .reg_class = "gpr" } },
                .register_count = 1,
                .load_gprs = .{ rt, 0 },
                .load_gpr_count = if (is_load) 1 else 0,
                .writeback = null,
            };
        }
    }

    const unscaled_top: u32 = word & 0xFFE00C00;
    const UnscaledEntry = struct { top: u32, kind: []const u8, width: i64, reg_class: []const u8 };
    const unscaled_table = [_]UnscaledEntry{
        .{ .top = 0xF8000000, .kind = "store", .width = 8, .reg_class = "gpr" },
        .{ .top = 0xF8400000, .kind = "load", .width = 8, .reg_class = "gpr" },
        .{ .top = 0xB8000000, .kind = "store", .width = 4, .reg_class = "gpr" },
        .{ .top = 0xB8400000, .kind = "load", .width = 4, .reg_class = "gpr" },
        .{ .top = 0xB8800000, .kind = "load", .width = 4, .reg_class = "gpr" },
        .{ .top = 0x38000000, .kind = "store", .width = 1, .reg_class = "gpr" },
        .{ .top = 0x38400000, .kind = "load", .width = 1, .reg_class = "gpr" },
        .{ .top = 0x38800000, .kind = "load", .width = 1, .reg_class = "gpr" },
        .{ .top = 0x38C00000, .kind = "load", .width = 1, .reg_class = "gpr" },
        .{ .top = 0x78000000, .kind = "store", .width = 2, .reg_class = "gpr" },
        .{ .top = 0x78400000, .kind = "load", .width = 2, .reg_class = "gpr" },
        .{ .top = 0x78800000, .kind = "load", .width = 2, .reg_class = "gpr" },
        .{ .top = 0x78C00000, .kind = "load", .width = 2, .reg_class = "gpr" },
        .{ .top = 0x3C800000, .kind = "store", .width = 16, .reg_class = "simd" },
        .{ .top = 0x3CC00000, .kind = "load", .width = 16, .reg_class = "simd" },
        .{ .top = 0xBC000000, .kind = "store", .width = 4, .reg_class = "simd" },
        .{ .top = 0xBC400000, .kind = "load", .width = 4, .reg_class = "simd" },
        .{ .top = 0xFC000000, .kind = "store", .width = 8, .reg_class = "simd" },
        .{ .top = 0xFC400000, .kind = "load", .width = 8, .reg_class = "simd" },
    };
    for (unscaled_table) |u| {
        if (unscaled_top == u.top) {
            const is_load_gpr = std.mem.eql(u8, u.kind, "load") and std.mem.eql(u8, u.reg_class, "gpr");
            return .{
                .kind = u.kind,
                .rn = rn,
                .offset = signedBits((word >> 12) & 0x1FF, 9),
                .mode = "signed_unscaled",
                .width = u.width,
                .pair = false,
                .registers = .{ .{ .reg = @intCast(rt), .reg_class = u.reg_class }, .{ .reg = 0, .reg_class = "gpr" } },
                .register_count = 1,
                .load_gprs = .{ rt, 0 },
                .load_gpr_count = if (is_load_gpr) 1 else 0,
                .writeback = null,
            };
        }
    }

    const PairEntry = struct { top: u32, kind: []const u8, width: i64, mode: []const u8, writeback: bool };
    const pair_table = [_]PairEntry{
        .{ .top = 0xA9000000, .kind = "store", .width = 8, .mode = "signed_pair", .writeback = false },
        .{ .top = 0xA9400000, .kind = "load", .width = 8, .mode = "signed_pair", .writeback = false },
        .{ .top = 0xA9800000, .kind = "store", .width = 8, .mode = "pre_index", .writeback = true },
        .{ .top = 0xA8C00000, .kind = "load", .width = 8, .mode = "post_index", .writeback = true },
        .{ .top = 0x29000000, .kind = "store", .width = 4, .mode = "signed_pair", .writeback = false },
        .{ .top = 0x29400000, .kind = "load", .width = 4, .mode = "signed_pair", .writeback = false },
    };
    for (pair_table) |p| {
        if (top == p.top) {
            const displacement = signedBits((word >> 15) & 0x7F, 7) * p.width;
            const rt2: u32 = (word >> 10) & 31;
            const is_load = std.mem.eql(u8, p.kind, "load");
            return .{
                .kind = p.kind,
                .rn = rn,
                .offset = if (std.mem.eql(u8, p.mode, "post_index")) 0 else displacement,
                .mode = p.mode,
                .width = p.width,
                .pair = true,
                .registers = .{
                    .{ .reg = @intCast(rt), .reg_class = "gpr" },
                    .{ .reg = @intCast(rt2), .reg_class = "gpr" },
                },
                .register_count = 2,
                .load_gprs = .{ rt, rt2 },
                .load_gpr_count = if (is_load) 2 else 0,
                .writeback = if (p.writeback) displacement else null,
            };
        }
    }

    const simd_top: u32 = word & 0xFFFFFC00;
    const SimdEntry = struct { top: u32, kind: []const u8, width: i64 };
    const simd_table = [_]SimdEntry{
        .{ .top = 0x3DC00000, .kind = "load", .width = 16 },
        .{ .top = 0x3D800000, .kind = "store", .width = 16 },
        .{ .top = 0xBD400000, .kind = "load", .width = 4 },
        .{ .top = 0xFD400000, .kind = "load", .width = 8 },
    };
    for (simd_table) |s| {
        if (simd_top == s.top) {
            return .{
                .kind = s.kind,
                .rn = rn,
                .offset = 0,
                .mode = "simd_zero_offset",
                .width = s.width,
                .pair = false,
                .registers = .{ .{ .reg = @intCast(rt), .reg_class = "simd" }, .{ .reg = 0, .reg_class = "simd" } },
                .register_count = 1,
                .load_gprs = .{ 0, 0 },
                .load_gpr_count = 0,
                .writeback = null,
            };
        }
    }
    return null;
}

const DecodedComponent = struct {
    base: []const u8,
    frame_offset: i64,
    width: i64,
    data_register: i64,
    data_register_class: []const u8,
};

const DecodedAccess = struct {
    native_start: usize,
    kind: []const u8,
    encoded_base: []const u8,
    encoded_offset: i64,
    addressing_mode: []const u8,
    components: []const DecodedComponent,
};

fn decodeAarch64FrameAccesses(
    allocator: Allocator,
    code: []const u8,
    inline_data_ranges: []const InlineDataRange,
) AttributionError![]DecodedAccess {
    const Relation = struct {
        base: []const u8,
        offset: i64,
    };
    var relations: [32]?Relation = @splat(null);
    var constants: [32]?i64 = @splat(null);
    var list: std.ArrayList(DecodedAccess) = .empty;

    const Helper = struct {
        fn relationFor(rels: *const [32]?Relation, reg: u32) ?Relation {
            if (reg == 29) return .{ .base = "x29", .offset = 0 };
            if (reg == 31) return .{ .base = "sp", .offset = 0 };
            return rels[reg];
        }
        fn isInlineOffset(ranges: []const InlineDataRange, offset: usize) bool {
            for (ranges) |r| {
                var o = r.native_start;
                while (o < r.native_end) : (o += 4) {
                    if (o == offset) return true;
                }
            }
            return false;
        }
    };

    var native_start: usize = 0;
    while (native_start < code.len) : (native_start += 4) {
        if (native_start + 4 > code.len) {
            return fail("AArch64 function code is not word-aligned", .{});
        }
        if (Helper.isInlineOffset(inline_data_ranges, native_start)) {
            relations = @splat(null);
            constants = @splat(null);
            continue;
        }
        const word = std.mem.readInt(u32, code[native_start..][0..4], .little);
        if (decodeAarch64Memory(word)) |mem| {
            if (Helper.relationFor(&relations, mem.rn)) |rel| {
                const mode = if (mem.rn == 29 or mem.rn == 31) mem.mode else "materialized";
                var comps: std.ArrayList(DecodedComponent) = .empty;
                var idx: usize = 0;
                while (idx < mem.register_count) : (idx += 1) {
                    const reg_info = mem.registers[idx];
                    try comps.append(allocator, .{
                        .base = rel.base,
                        .frame_offset = rel.offset + mem.offset + @as(i64, @intCast(idx)) * mem.width,
                        .width = mem.width,
                        .data_register = reg_info.reg,
                        .data_register_class = reg_info.reg_class,
                    });
                }
                const enc_base = if (mem.rn == 31)
                    "sp"
                else
                    try std.fmt.allocPrint(allocator, "x{d}", .{mem.rn});
                try list.append(allocator, .{
                    .native_start = native_start,
                    .kind = mem.kind,
                    .encoded_base = enc_base,
                    .encoded_offset = mem.offset,
                    .addressing_mode = mode,
                    .components = try comps.toOwnedSlice(allocator),
                });
            }
            var lg_idx: usize = 0;
            while (lg_idx < mem.load_gpr_count) : (lg_idx += 1) {
                const r = mem.load_gprs[lg_idx];
                relations[r] = null;
                constants[r] = null;
            }
            if (mem.writeback != null) {
                relations[mem.rn] = null;
                constants[mem.rn] = null;
            }
            continue;
        }

        const move_top = word & 0xFF800000;
        if (move_top == 0xD2800000 or move_top == 0xF2800000) {
            const register: u32 = word & 31;
            const value: u64 = (word >> 5) & 0xFFFF;
            const shift: u6 = @intCast(((word >> 21) & 3) * 16);
            if (move_top == 0xF2800000 and constants[register] != null) {
                const prev_u: u64 = @bitCast(constants[register].?);
                const mask: u64 = ~(@as(u64, 0xFFFF) << shift);
                const next_u: u64 = (prev_u & mask) | (value << shift);
                constants[register] = @bitCast(next_u);
            } else if (move_top == 0xD2800000) {
                constants[register] = @bitCast(value << shift);
            } else {
                constants[register] = null;
            }
            relations[register] = null;
            continue;
        }

        if ((word & 0x1F000000) == 0x11000000 and ((word >> 31) & 1) == 1) {
            const rd: u32 = word & 31;
            const rn: u32 = (word >> 5) & 31;
            const shift_amt: u5 = if (((word >> 22) & 1) == 1) 12 else 0;
            const immediate: i64 = @as(i64, @intCast((word >> 10) & 0xFFF)) << shift_amt;
            const displacement: i64 = if (((word >> 30) & 1) == 1) -immediate else immediate;
            if (Helper.relationFor(&relations, rn)) |rel| {
                relations[rd] = .{ .base = rel.base, .offset = rel.offset + displacement };
            } else {
                relations[rd] = null;
            }
            constants[rd] = null;
            continue;
        }

        if ((word & 0xFFE0FC00) == 0x8B000000) {
            const rd: u32 = word & 31;
            const rn: u32 = (word >> 5) & 31;
            const rm: u32 = (word >> 16) & 31;
            const rn_rel = Helper.relationFor(&relations, rn);
            const rm_rel = Helper.relationFor(&relations, rm);
            if (rn_rel != null and constants[rm] != null) {
                relations[rd] = .{ .base = rn_rel.?.base, .offset = rn_rel.?.offset + constants[rm].? };
            } else if (rm_rel != null and constants[rn] != null) {
                relations[rd] = .{ .base = rm_rel.?.base, .offset = rm_rel.?.offset + constants[rn].? };
            } else {
                relations[rd] = null;
            }
            constants[rd] = null;
            continue;
        }

        const is_branch = (word & 0x7C000000) == 0x14000000 or
            (word & 0xFF000010) == 0x54000000 or
            (word & 0x7E000000) == 0x34000000 or
            (word & 0x7E000000) == 0x36000000 or
            (word & 0xFFFFFC1F) == 0xD61F0000 or
            (word & 0xFFFFFC1F) == 0xD63F0000 or
            (word & 0xFFFFFC1F) == 0xD65F0000;
        if (is_branch) {
            var i: usize = 0;
            while (i < 29) : (i += 1) {
                relations[i] = null;
                constants[i] = null;
            }
            continue;
        }

        const rd: u32 = word & 31;
        if (rd < 29) {
            relations[rd] = null;
            constants[rd] = null;
        }
    }
    return try list.toOwnedSlice(allocator);
}

fn requireIntField(obj: std.json.ObjectMap, key: []const u8) AttributionError!i64 {
    const val = obj.get(key) orelse return fail("frame metadata field '{s}' must be an integer", .{key});
    return switch (val) {
        .integer => |i| i,
        else => fail("frame metadata field '{s}' must be an integer", .{key}),
    };
}

fn requireArrayField(obj: std.json.ObjectMap, key: []const u8) AttributionError![]const std.json.Value {
    const val = obj.get(key) orelse return fail("frame metadata field '{s}' must be an array", .{key});
    return switch (val) {
        .array => |arr| arr.items,
        else => fail("frame metadata field '{s}' must be an array", .{key}),
    };
}

fn optionalIntField(obj: std.json.ObjectMap, key: []const u8) AttributionError!?i64 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .null => null,
        .integer => |i| i,
        else => fail("frame metadata field '{s}' must be an integer or null", .{key}),
    };
}

fn optionalStringField(obj: std.json.ObjectMap, key: []const u8) AttributionError!?[]const u8 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .null => null,
        .string => |s| s,
        else => fail("frame metadata field '{s}' must be a string or null", .{key}),
    };
}

fn optionalBoolField(obj: std.json.ObjectMap, key: []const u8) AttributionError!?bool {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .null => null,
        .bool => |b| b,
        else => fail("frame metadata field '{s}' must be a boolean or null", .{key}),
    };
}

fn isLowercaseHex64(s: []const u8) bool {
    if (s.len != 64) return false;
    for (s) |c| {
        if (!((c >= '0' and c <= '9') or (c >= 'a' and c <= 'f'))) return false;
    }
    return true;
}

fn optStrEql(a: ?[]const u8, b: ?[]const u8) bool {
    if (a == null and b == null) return true;
    if (a == null or b == null) return false;
    return std.mem.eql(u8, a.?, b.?);
}

fn isAllowedAarch64Detail(origin: []const u8, detail_opt: ?[]const u8) bool {
    const detail = detail_opt orelse return false;
    if (std.mem.eql(u8, origin, "allocator_spill")) {
        return std.mem.eql(u8, detail, "allocator_slot");
    } else if (std.mem.eql(u8, origin, "wasm_local_or_phi")) {
        return std.mem.eql(u8, detail, "wasm_local_or_lowered_phi");
    } else if (std.mem.eql(u8, origin, "explicit_frame_storage")) {
        return std.mem.eql(u8, detail, "hidden_return_pointer") or
            std.mem.eql(u8, detail, "call_result_scratch");
    } else if (std.mem.eql(u8, origin, "fixed_runtime_frame_state")) {
        const allowed = [_][]const u8{
            "saved_frame_pointer",
            "return_address",
            "reserved_vmctx",
            "caller_saved_register",
            "callee_saved_register",
            "incoming_abi_argument",
            "prologue_saved_fp_lr",
            "epilogue_restored_fp_lr",
            "outgoing_abi_frame",
        };
        for (allowed) |item| {
            if (std.mem.eql(u8, detail, item)) return true;
        }
        return false;
    } else if (std.mem.eql(u8, origin, "unknown")) {
        return std.mem.eql(u8, detail, "unclassified_frame_access");
    }
    return false;
}

fn isValidEncodedBase(architecture: []const u8, base_opt: ?[]const u8) bool {
    const base = base_opt orelse return false;
    if (std.mem.eql(u8, architecture, "x86_64")) {
        return std.mem.eql(u8, base, "rbp") or std.mem.eql(u8, base, "rsp");
    }
    if (std.mem.eql(u8, base, "sp")) return true;
    if (base.len >= 2 and base[0] == 'x') {
        const reg = std.fmt.parseInt(u32, base[1..], 10) catch return false;
        return reg <= 30;
    }
    return false;
}

fn isValidAddressingMode(mode_opt: ?[]const u8) bool {
    const mode = mode_opt orelse return false;
    const valid = [_][]const u8{
        "unsigned_scaled",
        "signed_unscaled",
        "signed_pair",
        "pre_index",
        "post_index",
        "simd_zero_offset",
        "materialized",
    };
    for (valid) |item| {
        if (std.mem.eql(u8, mode, item)) return true;
    }
    return false;
}

pub fn loadFrameMetadataFromJson(
    allocator: Allocator,
    path: []const u8,
    json_text: []const u8,
    func_index: i64,
    function_code: []const u8,
    cwasm_version: u32,
    module_text: []const u8,
    function_offset: usize,
    artifact_info: CwasmInfo,
) AttributionError!FrameMetadata {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_text, .{}) catch |err| {
        return fail("{s}: malformed frame metadata: {s}", .{ path, @errorName(err) });
    };
    const root_obj = switch (parsed.value) {
        .object => |o| o,
        else => return fail("{s}: frame metadata root must be an object", .{path}),
    };

    const schema_val = root_obj.get("schema") orelse return fail("{s}: incompatible frame schema null", .{path});
    const schema = switch (schema_val) {
        .string => |s| s,
        else => return fail("{s}: incompatible frame schema", .{path}),
    };
    if (!std.mem.eql(u8, schema, FRAME_SCHEMA)) {
        return fail("{s}: incompatible frame schema '{s}'", .{ path, schema });
    }

    const schema_version_val = root_obj.get("schema_version") orelse return fail("{s}: incompatible frame schema_version=null", .{path});
    const schema_version: i64 = switch (schema_version_val) {
        .integer => |i| i,
        else => return fail("{s}: incompatible frame schema_version", .{path}),
    };
    if (schema_version != 1 and schema_version != 2) {
        return fail("{s}: incompatible frame schema_version={d}; expected one of [1, 2]", .{ path, schema_version });
    }

    const arch_val = root_obj.get("architecture") orelse return fail("{s}: unsupported frame metadata architecture null", .{path});
    const architecture = switch (arch_val) {
        .string => |s| s,
        else => return fail("{s}: unsupported frame metadata architecture", .{path}),
    };
    if (!isSupportedArch(architecture)) {
        return fail("{s}: unsupported frame metadata architecture '{s}'", .{ path, architecture });
    }
    if (schema_version == 1 and !std.mem.eql(u8, architecture, "x86_64")) {
        return fail("{s}: schema-v1 frame metadata is x86_64-only", .{path});
    }
    if (!std.mem.eql(u8, architecture, artifact_info.architecture)) {
        return fail("{s}: metadata architecture {s} does not match artifact target {s}", .{ path, architecture, artifact_info.architecture });
    }
    if (!artifact_info.target_verified or artifact_info.abi == null) {
        return fail("{s}: artifact target-info is legacy/unspecified; authoritative frame ABI validation is unavailable", .{path});
    }

    const abi_val = root_obj.get("abi") orelse return fail("{s}: metadata ABI null does not match artifact ABI", .{path});
    const abi = switch (abi_val) {
        .string => |s| s,
        else => return fail("{s}: metadata ABI does not match artifact ABI", .{path}),
    };
    if (!std.mem.eql(u8, abi, artifact_info.abi.?)) {
        return fail("{s}: metadata ABI '{s}' does not match artifact ABI '{s}'", .{ path, abi, artifact_info.abi.? });
    }

    const module_id = try requireIntField(root_obj, "module");
    if (std.mem.eql(u8, architecture, "x86_64")) {
        if (!std.mem.eql(u8, abi, "sysv") and !std.mem.eql(u8, abi, "win64")) {
            return fail("{s}: frame metadata abi must be one of [sysv, win64]", .{path});
        }
    } else {
        if (!std.mem.eql(u8, abi, "aapcs64")) {
            return fail("{s}: frame metadata abi must be one of [aapcs64]", .{path});
        }
    }

    const build_id_val = root_obj.get("compiler_build_id") orelse return fail("{s}: compiler_build_id must be a string", .{path});
    const compiler_build_id = switch (build_id_val) {
        .string => |s| s,
        else => return fail("{s}: compiler_build_id must be a string", .{path}),
    };

    const meta_aot_version = try requireIntField(root_obj, "cwasm_aot_version");
    if (meta_aot_version != @as(i64, cwasm_version)) {
        return fail("{s}: metadata/cwasm AOT version mismatch ({d} != {d})", .{ path, meta_aot_version, cwasm_version });
    }
    const local_func = try requireIntField(root_obj, "local_func");
    if (local_func != func_index) {
        return fail("{s}: metadata local_func={d} does not match --func {d}", .{ path, local_func, func_index });
    }
    const module_text_size = try requireIntField(root_obj, "module_text_size");
    if (module_text_size != @as(i64, @intCast(module_text.len))) {
        return fail("{s}: metadata module_text_size={d} does not match cwasm text_size={d}", .{ path, module_text_size, module_text.len });
    }
    const meta_func_offset = try requireIntField(root_obj, "function_offset");
    if (meta_func_offset != @as(i64, @intCast(function_offset))) {
        return fail("{s}: metadata function_offset={d} does not match cwasm offset={d}", .{ path, meta_func_offset, function_offset });
    }

    const module_hash_val = root_obj.get("module_text_sha256") orelse return fail("{s}: module_text_sha256 must be 64 lowercase hex digits", .{path});
    const module_hash = switch (module_hash_val) {
        .string => |s| s,
        else => return fail("{s}: module_text_sha256 must be 64 lowercase hex digits", .{path}),
    };
    if (!isLowercaseHex64(module_hash)) {
        return fail("{s}: module_text_sha256 must be 64 lowercase hex digits", .{path});
    }
    const actual_module_hash = try sha256Hex(allocator, module_text);
    if (!std.mem.eql(u8, module_hash, actual_module_hash)) {
        return fail("{s}: module text hash mismatch; metadata belongs to a different core ({s} != {s})", .{ path, module_hash, actual_module_hash });
    }

    const code_size = try requireIntField(root_obj, "code_size");
    if (code_size != @as(i64, @intCast(function_code.len))) {
        return fail("{s}: metadata code_size={d} does not match cwasm function span={d}", .{ path, code_size, function_code.len });
    }

    var actual_hash: []const u8 = "";
    if (schema_version == 1) {
        const rel_arr = try requireArrayField(root_obj, "direct_call_rel32_offsets");
        var rels = try allocator.alloc(i64, rel_arr.len);
        for (rel_arr, 0..) |item, idx| {
            rels[idx] = switch (item) {
                .integer => |i| i,
                else => return fail("direct_call_rel32_offsets must contain integers", .{}),
            };
        }
        actual_hash = try normalizedCodeSha256(allocator, function_code, rels);
    } else {
        const rel_arr = try requireArrayField(root_obj, "normalized_relocations");
        var rels = try allocator.alloc(NormalizedRelocation, rel_arr.len);
        for (rel_arr, 0..) |item, idx| {
            const robj = switch (item) {
                .object => |o| o,
                else => return fail("normalized_relocations entries must be objects", .{}),
            };
            const ns = try requireIntField(robj, "native_start");
            const ne = try requireIntField(robj, "native_end");
            if (ns < 0 or ne < 0) return fail("invalid/overlapping normalized relocation [{d},{d})", .{ ns, ne });
            const kind_str = if (robj.get("kind")) |kv| switch (kv) {
                .string => |s| s,
                else => "",
            } else "";
            rels[idx] = .{
                .native_start = @intCast(ns),
                .native_end = @intCast(ne),
                .kind = kind_str,
            };
        }
        actual_hash = try normalizedCodeSha256V2(allocator, function_code, rels, architecture);
    }

    const expected_hash_val = root_obj.get("normalized_code_sha256") orelse return fail("{s}: normalized_code_sha256 must be 64 lowercase hex digits", .{path});
    const expected_hash = switch (expected_hash_val) {
        .string => |s| s,
        else => return fail("{s}: normalized_code_sha256 must be 64 lowercase hex digits", .{path}),
    };
    if (!isLowercaseHex64(expected_hash)) {
        return fail("{s}: normalized_code_sha256 must be 64 lowercase hex digits", .{path});
    }
    if (!std.mem.eql(u8, actual_hash, expected_hash)) {
        return fail("{s}: normalized native-code hash mismatch; metadata does not belong to this local_func ({s} != {s})", .{ path, expected_hash, actual_hash });
    }

    const layout_val = root_obj.get("frame_layout") orelse return fail("{s}: missing frame_layout/spill_metric object", .{path});
    const metric_val = root_obj.get("spill_metric") orelse return fail("{s}: missing frame_layout/spill_metric object", .{path});
    const layout_obj = switch (layout_val) {
        .object => |o| o,
        else => return fail("{s}: missing frame_layout/spill_metric object", .{path}),
    };
    const metric_obj = switch (metric_val) {
        .object => |o| o,
        else => return fail("{s}: missing frame_layout/spill_metric object", .{path}),
    };

    const fp_str = if (layout_obj.get("frame_pointer")) |v| switch (v) {
        .string => |s| s,
        else => "",
    } else "";
    const layout = FrameLayout{
        .frame_pointer = fp_str,
        .frame_size = try requireIntField(layout_obj, "frame_size"),
        .local_count = try requireIntField(layout_obj, "local_count"),
        .param_count = try requireIntField(layout_obj, "param_count"),
        .reserved_vmctx_offset = try requireIntField(layout_obj, "reserved_vmctx_offset"),
        .locals_first_offset = try requireIntField(layout_obj, "locals_first_offset"),
        .explicit_storage_first_offset = try requireIntField(layout_obj, "explicit_storage_first_offset"),
        .explicit_storage_slots = try requireIntField(layout_obj, "explicit_storage_slots"),
        .spill_base = try requireIntField(layout_obj, "spill_base"),
        .spill_stride = try requireIntField(layout_obj, "spill_stride"),
        .spill_slots = try requireIntField(layout_obj, "spill_slots"),
    };
    if (!std.mem.eql(u8, layout.frame_pointer, "rbp") and !std.mem.eql(u8, layout.frame_pointer, "x29")) {
        return fail("{s}: invalid frame pointer", .{path});
    }
    if (layout.frame_size <= 0 or layout.spill_stride == 0 or layout.spill_slots < 0 or layout.local_count < 0 or layout.explicit_storage_slots < 0) {
        return fail("{s}: invalid frame layout dimensions", .{path});
    }

    const metric = SpillMetric{
        .slots = try requireIntField(metric_obj, "slots"),
        .spilled_vregs = try requireIntField(metric_obj, "spilled_vregs"),
        .scalar = try requireIntField(metric_obj, "scalar"),
        .v128 = try requireIntField(metric_obj, "v128"),
        .slots_scalar = try requireIntField(metric_obj, "slots_scalar"),
        .slots_v128 = try requireIntField(metric_obj, "slots_v128"),
        .spill_ld = try requireIntField(metric_obj, "spill_ld"),
        .spill_st = try requireIntField(metric_obj, "spill_st"),
        .remat = try requireIntField(metric_obj, "remat"),
        .callee_saved = try requireIntField(metric_obj, "callee_saved"),
    };

    var regions: []FrameRegion = &.{};
    if (schema_version == 2) {
        const reg_arr = try requireArrayField(root_obj, "frame_regions");
        regions = try allocator.alloc(FrameRegion, reg_arr.len);
        var prior_region_end: ?i64 = null;
        for (reg_arr, 0..) |item, idx| {
            const robj = switch (item) {
                .object => |o| o,
                else => return fail("{s}: frame_regions entries must be objects", .{path}),
            };
            const r_start = try requireIntField(robj, "start");
            const r_end = try requireIntField(robj, "end");
            const r_origin = if (robj.get("origin")) |ov| switch (ov) {
                .string => |s| s,
                else => "",
            } else "";
            const r_detail = if (robj.get("detail")) |dv| switch (dv) {
                .string => |s| s,
                else => return fail("{s}: invalid/overlapping frame region [{d},{d})", .{ path, r_start, r_end }),
            } else return fail("{s}: invalid/overlapping frame region [{d},{d})", .{ path, r_start, r_end });

            if (r_end <= r_start or
                r_start < 0 or
                r_end > layout.frame_size or
                (prior_region_end != null and r_start < prior_region_end.?) or
                !isSupportedOrigin(r_origin))
            {
                return fail("{s}: invalid/overlapping frame region [{d},{d})", .{ path, r_start, r_end });
            }
            prior_region_end = r_end;
            regions[idx] = .{ .start = r_start, .end = r_end, .origin = r_origin, .detail = r_detail };
        }
    }

    const val_arr = try requireArrayField(root_obj, "allocator_values");
    var values = try allocator.alloc(AllocatorValue, val_arr.len);
    for (val_arr, 0..) |item, idx| {
        const vobj = switch (item) {
            .object => |o| o,
            else => return fail("{s}: allocator_values entries must be objects", .{path}),
        };
        const vreg = try requireIntField(vobj, "vreg");
        const slot = try requireIntField(vobj, "slot");
        const slot_count = try requireIntField(vobj, "slot_count");
        const frame_offset = try requireIntField(vobj, "frame_offset");
        const ir_use_count = try requireIntField(vobj, "ir_use_count");
        const ir_def_count = try requireIntField(vobj, "ir_def_count");
        const reload_count = try requireIntField(vobj, "reload_count");
        const store_count = try requireIntField(vobj, "store_count");
        const remat_el = if (vobj.get("rematerialization_eligible")) |rv| switch (rv) {
            .bool => |b| b,
            else => return fail("{s}: rematerialization_eligible must be boolean", .{path}),
        } else return fail("{s}: rematerialization_eligible must be boolean", .{path});
        const reused = if (vobj.get("reused")) |rv| switch (rv) {
            .bool => |b| b,
            else => return fail("{s}: reused must be boolean", .{path}),
        } else return fail("{s}: reused must be boolean", .{path});
        const val_type = if (vobj.get("value_type")) |tv| switch (tv) {
            .string => |s| s,
            else => "",
        } else "";
        const expected_offset = layout.spill_base + slot * layout.spill_stride;
        const type_slots: i64 = if (std.mem.eql(u8, val_type, "v128")) 2 else 1;
        for (values[0..idx]) |prev| {
            if (prev.vreg == vreg) {
                return fail("{s}: invalid allocator slot/offset/type for vreg={d}", .{ path, vreg });
            }
        }
        if (slot < 0 or slot_count <= 0 or slot + slot_count > layout.spill_slots or frame_offset != expected_offset or slot_count != type_slots) {
            return fail("{s}: invalid allocator slot/offset/type for vreg={d}", .{ path, vreg });
        }
        values[idx] = .{
            .vreg = vreg,
            .frame_offset = frame_offset,
            .slot = slot,
            .slot_count = slot_count,
            .value_type = val_type,
            .live_start = try optionalIntField(vobj, "live_start"),
            .live_end = try optionalIntField(vobj, "live_end"),
            .defining_opcode = try optionalStringField(vobj, "defining_opcode"),
            .source_class = try optionalStringField(vobj, "source_class"),
            .ir_use_count = ir_use_count,
            .ir_def_count = ir_def_count,
            .reload_count = reload_count,
            .store_count = store_count,
            .rematerialization_eligible = remat_el,
            .reused = reused,
        };
    }

    for (values) |val| {
        var actually_reused = false;
        var s = val.slot;
        while (s < val.slot + val.slot_count) : (s += 1) {
            var count_in_slot: usize = 0;
            for (values) |other| {
                if (s >= other.slot and s < other.slot + other.slot_count) {
                    count_in_slot += 1;
                }
            }
            if (count_in_slot > 1) {
                actually_reused = true;
                break;
            }
        }
        if (val.reused != actually_reused) {
            return fail("{s}: vreg {d} reused flag disagrees with overlapping allocator slots", .{ path, val.vreg });
        }
    }

    const id_arr = try requireArrayField(root_obj, "inline_data_ranges");
    var inline_data_ranges = try allocator.alloc(InlineDataRange, id_arr.len);
    var prior_data_end: usize = 0;
    for (id_arr, 0..) |item, idx| {
        const dobj = switch (item) {
            .object => |o| o,
            else => return fail("{s}: inline_data_ranges entries must be objects", .{path}),
        };
        const ns = try requireIntField(dobj, "native_start");
        const ne = try requireIntField(dobj, "native_end");
        const kind_str = if (dobj.get("kind")) |kv| switch (kv) {
            .string => |s| s,
            else => "",
        } else "";
        if (ns < 0 or ne <= ns or @as(usize, @intCast(ns)) < prior_data_end or ne > code_size or !std.mem.eql(u8, kind_str, "br_table")) {
            return fail("{s}: invalid/overlapping inline data range [{d},{d})", .{ path, ns, ne });
        }
        prior_data_end = @intCast(ne);
        inline_data_ranges[idx] = .{
            .native_start = @intCast(ns),
            .native_end = @intCast(ne),
            .kind = kind_str,
        };
    }

    const acc_arr = try requireArrayField(root_obj, "accesses");
    var accesses = try allocator.alloc(FrameAccess, acc_arr.len);
    var prior_end: usize = 0;
    var emitted_loads: i64 = 0;
    var emitted_stores: i64 = 0;

    const VregCounts = struct { load: i64 = 0, store: i64 = 0 };
    var resolved_by_vreg: std.AutoHashMapUnmanaged(i64, VregCounts) = .empty;

    const CompValidator = struct {
        fn parseAndValidate(
            alloc: Allocator,
            p: []const u8,
            arch: []const u8,
            sv: i64,
            lay: FrameLayout,
            regs: []const FrameRegion,
            vals: []const AllocatorValue,
            cobj: std.json.ObjectMap,
            access_kind: []const u8,
            start: usize,
            loads_ptr: *i64,
            stores_ptr: *i64,
            vreg_counts_ptr: *std.AutoHashMapUnmanaged(i64, VregCounts),
        ) AttributionError!AccessComponent {
            const base = if (cobj.get("base")) |bv| switch (bv) {
                .string => |s| s,
                else => "",
            } else "";
            const base_ok = if (std.mem.eql(u8, arch, "x86_64"))
                (std.mem.eql(u8, base, "rbp") or std.mem.eql(u8, base, "rsp"))
            else
                (std.mem.eql(u8, base, "x29") or std.mem.eql(u8, base, "sp"));
            if (!base_ok) {
                return fail("{s}: invalid effective frame base '{s}' at {d}", .{ p, base, start });
            }
            const frame_offset = try requireIntField(cobj, "frame_offset");
            const width = try requireIntField(cobj, "width");
            if (width <= 0) {
                return fail("{s}: invalid frame access width at {d}", .{ p, start });
            }
            const origin = if (cobj.get("origin")) |ov| switch (ov) {
                .string => |s| s,
                else => "",
            } else "";
            if (!isSupportedOrigin(origin)) {
                return fail("{s}: invalid frame origin '{s}' at offset {d}", .{ p, origin, start });
            }
            const detail = try optionalStringField(cobj, "detail");
            var data_register: ?i64 = null;
            var data_register_class: ?[]const u8 = null;

            if (std.mem.eql(u8, arch, "aarch64") and sv == 2) {
                if (!isAllowedAarch64Detail(origin, detail)) {
                    return fail("{s}: invalid frame detail '{s}' for {s} at {d}", .{ p, detail orelse "null", origin, start });
                }
                const dr_val = cobj.get("data_register") orelse return fail("{s}: invalid data-register identity at {d}", .{ p, start });
                const dr = switch (dr_val) {
                    .integer => |i| i,
                    else => return fail("{s}: invalid data-register identity at {d}", .{ p, start }),
                };
                const drc = try optionalStringField(cobj, "data_register_class") orelse return fail("{s}: invalid data-register identity at {d}", .{ p, start });
                if (dr < 0 or dr > 31 or (!std.mem.eql(u8, drc, "gpr") and !std.mem.eql(u8, drc, "simd"))) {
                    return fail("{s}: invalid data-register identity at {d}", .{ p, start });
                }
                data_register = dr;
                data_register_class = drc;
                if (std.mem.eql(u8, base, "x29") and frame_offset >= 0 and frame_offset < lay.frame_size) {
                    var found_region: ?FrameRegion = null;
                    for (regs) |r| {
                        if (r.start <= frame_offset and frame_offset + width <= r.end) {
                            found_region = r;
                            break;
                        }
                    }
                    if (found_region) |r| {
                        if (!std.mem.eql(u8, r.origin, origin)) {
                            return fail("{s}: frame access origin {s} disagrees with region {s} at {d}", .{ p, origin, r.origin, start });
                        }
                    } else {
                        if (!std.mem.eql(u8, origin, "unknown")) {
                            return fail("{s}: frame access [{d},{d}) is outside declared regions", .{ p, frame_offset, frame_offset + width });
                        }
                    }
                }
            } else {
                data_register = try optionalIntField(cobj, "data_register");
                data_register_class = try optionalStringField(cobj, "data_register_class");
            }

            const local_index = try optionalIntField(cobj, "local_index");
            if (std.mem.eql(u8, origin, "wasm_local_or_phi")) {
                if (local_index == null or local_index.? < 0 or local_index.? >= lay.local_count) {
                    return fail("{s}: local frame access at {d} lacks valid local_index", .{ p, start });
                }
            }
            const explicit_slot = try optionalIntField(cobj, "explicit_slot");
            if (std.mem.eql(u8, origin, "explicit_frame_storage")) {
                if (explicit_slot == null or explicit_slot.? < 0 or explicit_slot.? >= lay.explicit_storage_slots) {
                    return fail("{s}: explicit frame access at {d} lacks valid slot", .{ p, start });
                }
                if (std.mem.eql(u8, arch, "aarch64") and frame_offset != lay.explicit_storage_first_offset + explicit_slot.? * 8) {
                    return fail("{s}: explicit slot/offset mismatch at {d}", .{ p, start });
                }
            }
            const vreg = try optionalIntField(cobj, "vreg");
            const amb_val = cobj.get("vreg_ambiguous") orelse return fail("{s}: invalid vreg/vreg_ambiguous combination at {d}", .{ p, start });
            const ambiguous = switch (amb_val) {
                .bool => |b| b,
                else => return fail("{s}: invalid vreg/vreg_ambiguous combination at {d}", .{ p, start }),
            };
            if (vreg != null and ambiguous) {
                return fail("{s}: invalid vreg/vreg_ambiguous combination at {d}", .{ p, start });
            }

            const slot_opt = try optionalIntField(cobj, "slot");
            const def_op = optionalStringField(cobj, "defining_opcode") catch return fail("{s}: allocator source defining_opcode disagrees", .{p});
            const src_cls = optionalStringField(cobj, "source_class") catch return fail("{s}: allocator source source_class disagrees", .{p});
            const remat_opt = optionalBoolField(cobj, "rematerialization_eligible") catch return fail("{s}: allocator source rematerialization_eligible disagrees", .{p});
            const ir_pos = try optionalIntField(cobj, "ir_position");
            const ir_op = try optionalStringField(cobj, "ir_opcode");

            const comp = AccessComponent{
                .base = base,
                .frame_offset = frame_offset,
                .width = width,
                .data_register = data_register,
                .data_register_class = data_register_class,
                .origin = origin,
                .detail = detail,
                .slot = slot_opt,
                .local_index = local_index,
                .explicit_slot = explicit_slot,
                .vreg = vreg,
                .vreg_ambiguous = ambiguous,
                .defining_opcode = def_op,
                .source_class = src_cls,
                .rematerialization_eligible = remat_opt,
                .ir_position = ir_pos,
                .ir_opcode = ir_op,
            };

            if (!std.mem.eql(u8, origin, "allocator_spill")) {
                return comp;
            }

            const slot = slot_opt orelse return fail("{s}: allocator access at {d} lacks integer slot", .{ p, start });
            var slot_candidates: std.ArrayList(AllocatorValue) = .empty;
            for (vals) |v| {
                if (slot >= v.slot and slot < v.slot + v.slot_count) {
                    try slot_candidates.append(alloc, v);
                }
            }
            if (slot_candidates.items.len == 0) {
                return fail("{s}: allocator access at {d} names unassigned slot {d}", .{ p, start, slot });
            }
            const stride: i64 = if (lay.spill_stride < 0) -lay.spill_stride else lay.spill_stride;
            const occupied_slots: i64 = @divTrunc(width + stride - 1, stride);
            if (slot + occupied_slots > lay.spill_slots) {
                return fail("{s}: allocator component at {d} exceeds spill layout", .{ p, start });
            }
            var cov_candidates: std.ArrayList(AllocatorValue) = .empty;
            for (slot_candidates.items) |v| {
                if (slot + occupied_slots <= v.slot + v.slot_count) {
                    try cov_candidates.append(alloc, v);
                }
            }
            if (cov_candidates.items.len == 0) {
                return fail("{s}: allocator component at {d} exceeds value coverage", .{ p, start });
            }
            const expected_offset = lay.spill_base + slot * lay.spill_stride;
            if (frame_offset != expected_offset) {
                return fail("{s}: allocator slot/offset mismatch at {d}: slot {d} maps to {d}, not {d}", .{ p, start, slot, expected_offset, frame_offset });
            }

            if (vreg) |vr| {
                var matched_val: ?AllocatorValue = null;
                for (cov_candidates.items) |cand| {
                    if (cand.vreg == vr) {
                        matched_val = cand;
                        break;
                    }
                }
                const val = matched_val orelse return fail("{s}: access vreg={d} does not occupy slot {d}", .{ p, vr, slot });
                if (!optStrEql(def_op, val.defining_opcode)) {
                    return fail("{s}: allocator source defining_opcode disagrees with vreg {d}", .{ p, vr });
                }
                if (!optStrEql(src_cls, val.source_class)) {
                    return fail("{s}: allocator source source_class disagrees with vreg {d}", .{ p, vr });
                }
                if (remat_opt == null or remat_opt.? != val.rematerialization_eligible) {
                    return fail("{s}: allocator source rematerialization_eligible disagrees with vreg {d}", .{ p, vr });
                }
                const remaining_slots = val.slot + val.slot_count - slot;
                if (remaining_slots <= 0 or width > remaining_slots * stride) {
                    return fail("{s}: allocator component at {d} exceeds vreg {d} slot coverage", .{ p, start, vr });
                }
            } else {
                if (def_op != null or src_cls != null or remat_opt != null) {
                    return fail("{s}: unresolved allocator source fields must be null at {d}", .{ p, start });
                }
            }

            if (cov_candidates.items.len > 1 and vreg == null and !ambiguous) {
                return fail("{s}: reused slot {d} lacks a resolved vreg or vreg_ambiguous=true", .{ p, slot });
            }
            if (cov_candidates.items.len == 1 and vreg == null) {
                return fail("{s}: unique allocator slot {d} omitted provable vreg", .{ p, slot });
            }
            if (vreg) |vr| {
                const gop = try vreg_counts_ptr.getOrPut(alloc, vr);
                if (!gop.found_existing) gop.value_ptr.* = .{};
                if (std.mem.eql(u8, access_kind, "load")) {
                    gop.value_ptr.load += 1;
                } else {
                    gop.value_ptr.store += 1;
                }
            }
            if (std.mem.eql(u8, access_kind, "load")) {
                loads_ptr.* += 1;
            } else {
                stores_ptr.* += 1;
            }
            return comp;
        }
    };

    for (acc_arr, 0..) |item, idx| {
        const aobj = switch (item) {
            .object => |o| o,
            else => return fail("{s}: accesses entries must be objects", .{path}),
        };
        const ns_signed = try requireIntField(aobj, "native_start");
        const ne_signed = try requireIntField(aobj, "native_end");
        const frame_offset = try requireIntField(aobj, "frame_offset");
        const width = try requireIntField(aobj, "width");
        if (ns_signed < 0 or ne_signed <= ns_signed or ne_signed > code_size) {
            return fail("{s}: overlapping/out-of-range native access [{d},{d})", .{ path, ns_signed, ne_signed });
        }
        const start: usize = @intCast(ns_signed);
        const end: usize = @intCast(ne_signed);
        if (start < prior_end) {
            return fail("{s}: overlapping/out-of-range native access [{d},{d})", .{ path, start, end });
        }
        const kind = if (aobj.get("kind")) |kv| switch (kv) {
            .string => |s| s,
            else => "",
        } else "";
        const base = if (aobj.get("base")) |bv| switch (bv) {
            .string => |s| s,
            else => "",
        } else "";
        const origin = if (aobj.get("origin")) |ov| switch (ov) {
            .string => |s| s,
            else => "",
        } else "";
        const base_ok = if (std.mem.eql(u8, architecture, "x86_64"))
            (std.mem.eql(u8, base, "rbp") or std.mem.eql(u8, base, "rsp"))
        else
            (std.mem.eql(u8, base, "x29") or std.mem.eql(u8, base, "sp"));
        if ((!std.mem.eql(u8, kind, "load") and !std.mem.eql(u8, kind, "store")) or !base_ok) {
            return fail("{s}: invalid frame access kind/base at offset {d}", .{ path, start });
        }
        if (!isSupportedOrigin(origin)) {
            return fail("{s}: invalid frame origin '{s}' at offset {d}", .{ path, origin, start });
        }
        for (inline_data_ranges) |dr| {
            if (start >= dr.native_start and start < dr.native_end) {
                return fail("{s}: frame access at {d} overlaps inline data", .{ path, start });
            }
        }

        const top_detail = try optionalStringField(aobj, "detail");
        const top_slot = try optionalIntField(aobj, "slot");
        const top_local = try optionalIntField(aobj, "local_index");
        const top_explicit = try optionalIntField(aobj, "explicit_slot");
        const top_vreg = try optionalIntField(aobj, "vreg");
        const top_amb = (try optionalBoolField(aobj, "vreg_ambiguous")) orelse false;
        const top_def_op = optionalStringField(aobj, "defining_opcode") catch null;
        const top_src_cls = optionalStringField(aobj, "source_class") catch null;
        const top_remat = optionalBoolField(aobj, "rematerialization_eligible") catch null;
        const top_ir_pos = try optionalIntField(aobj, "ir_position");
        const top_ir_op = try optionalStringField(aobj, "ir_opcode");

        var comps: []AccessComponent = &.{};
        var encoded_base: ?[]const u8 = null;
        var encoded_offset: ?i64 = null;
        var addressing_mode: ?[]const u8 = null;

        if (schema_version == 1) {
            const single = try CompValidator.parseAndValidate(
                allocator,
                path,
                architecture,
                schema_version,
                layout,
                regions,
                values,
                aobj,
                kind,
                start,
                &emitted_loads,
                &emitted_stores,
                &resolved_by_vreg,
            );
            comps = try allocator.alloc(AccessComponent, 1);
            comps[0] = single;
        } else {
            encoded_base = try optionalStringField(aobj, "encoded_base");
            encoded_offset = optionalIntField(aobj, "encoded_offset") catch null;
            addressing_mode = try optionalStringField(aobj, "addressing_mode");
            if (!isValidEncodedBase(architecture, encoded_base) or encoded_offset == null or !isValidAddressingMode(addressing_mode)) {
                return fail("{s}: schema-v2 access at {d} lacks encoded address", .{ path, start });
            }
            const comp_arr = try requireArrayField(aobj, "components");
            if (comp_arr.len < 1 or comp_arr.len > 2) {
                return fail("{s}: access at {d} must have one or two components", .{ path, start });
            }
            comps = try allocator.alloc(AccessComponent, comp_arr.len);
            var sum_width: i64 = 0;
            for (comp_arr, 0..) |citem, cidx| {
                const cobj = switch (citem) {
                    .object => |o| o,
                    else => return fail("{s}: access components must be objects", .{path}),
                };
                comps[cidx] = try CompValidator.parseAndValidate(
                    allocator,
                    path,
                    architecture,
                    schema_version,
                    layout,
                    regions,
                    values,
                    cobj,
                    kind,
                    start,
                    &emitted_loads,
                    &emitted_stores,
                    &resolved_by_vreg,
                );
                sum_width += comps[cidx].width;
            }
            if (sum_width != width) {
                return fail("{s}: component widths disagree at {d}", .{ path, start });
            }
            if (comps.len == 2 and (!std.mem.eql(u8, comps[1].base, comps[0].base) or comps[1].frame_offset != comps[0].frame_offset + comps[0].width)) {
                return fail("{s}: pair components are not contiguous at {d}", .{ path, start });
            }
            const is_pair_mode = std.mem.eql(u8, addressing_mode.?, "signed_pair") or
                std.mem.eql(u8, addressing_mode.?, "pre_index") or
                std.mem.eql(u8, addressing_mode.?, "post_index");
            if ((comps.len == 2) != is_pair_mode) {
                return fail("{s}: component count/addressing mode mismatch at {d}", .{ path, start });
            }
            if (comps.len == 2 and comps[0].width != comps[1].width) {
                return fail("{s}: pair component widths disagree at {d}", .{ path, start });
            }
            const expected_origin = if (comps.len == 1 or std.mem.eql(u8, comps[0].origin, comps[1].origin))
                comps[0].origin
            else
                "unknown";
            const expected_detail: ?[]const u8 = if (comps.len == 1 or optStrEql(comps[0].detail, comps[1].detail))
                comps[0].detail
            else
                "mixed_pair_frame_access";
            if (!std.mem.eql(u8, origin, expected_origin)) {
                return fail("{s}: top-level/component origin mismatch at {d}", .{ path, start });
            }
            if (!optStrEql(top_detail, expected_detail)) {
                return fail("{s}: top-level/component detail mismatch at {d}", .{ path, start });
            }
            if (!std.mem.eql(u8, base, comps[0].base) or frame_offset != comps[0].frame_offset) {
                return fail("{s}: top-level/component address mismatch at {d}", .{ path, start });
            }
            if (comps.len == 1) {
                const c0 = comps[0];
                if (top_slot != c0.slot or
                    top_local != c0.local_index or
                    top_explicit != c0.explicit_slot or
                    top_vreg != c0.vreg or
                    top_amb != c0.vreg_ambiguous or
                    !optStrEql(top_def_op, c0.defining_opcode) or
                    !optStrEql(top_src_cls, c0.source_class) or
                    top_remat != c0.rematerialization_eligible)
                {
                    return fail("{s}: top-level/component field mismatch at {d}", .{ path, start });
                }
            }
        }

        accesses[idx] = .{
            .native_start = start,
            .native_end = end,
            .kind = kind,
            .base = base,
            .frame_offset = frame_offset,
            .width = width,
            .origin = origin,
            .detail = top_detail,
            .slot = top_slot,
            .local_index = top_local,
            .explicit_slot = top_explicit,
            .vreg = top_vreg,
            .vreg_ambiguous = top_amb,
            .defining_opcode = top_def_op,
            .source_class = top_src_cls,
            .rematerialization_eligible = top_remat,
            .ir_position = top_ir_pos,
            .ir_opcode = top_ir_op,
            .encoded_base = encoded_base,
            .encoded_offset = encoded_offset,
            .addressing_mode = addressing_mode,
            .components = comps,
        };
        prior_end = end;
    }

    if (schema_version == 2 and std.mem.eql(u8, architecture, "aarch64")) {
        const decoded_accesses = try decodeAarch64FrameAccesses(allocator, function_code, inline_data_ranges);
        if (decoded_accesses.len != accesses.len) {
            return fail("{s}: emitted frame-access offsets disagree with native code", .{path});
        }
        for (decoded_accesses, accesses) |dec, acc| {
            if (dec.native_start != acc.native_start) {
                return fail("{s}: emitted frame-access offsets disagree with native code", .{path});
            }
            if (!std.mem.eql(u8, acc.kind, dec.kind)) {
                return fail("{s}: native kind mismatch at {d}: {s} != {s}", .{ path, acc.native_start, acc.kind, dec.kind });
            }
            if (!optStrEql(acc.encoded_base, dec.encoded_base)) {
                return fail("{s}: native encoded_base mismatch at {d}", .{ path, acc.native_start });
            }
            if (acc.encoded_offset == null or acc.encoded_offset.? != dec.encoded_offset) {
                return fail("{s}: native encoded_offset mismatch at {d}", .{ path, acc.native_start });
            }
            if (!optStrEql(acc.addressing_mode, dec.addressing_mode)) {
                return fail("{s}: native addressing_mode mismatch at {d}", .{ path, acc.native_start });
            }
            if (acc.components.len != dec.components.len) {
                return fail("{s}: native component count mismatch at {d}", .{ path, acc.native_start });
            }
            for (acc.components, dec.components) |ac, dc| {
                if (!std.mem.eql(u8, ac.base, dc.base) or
                    ac.frame_offset != dc.frame_offset or
                    ac.width != dc.width or
                    ac.data_register == null or
                    ac.data_register.? != dc.data_register or
                    !optStrEql(ac.data_register_class, dc.data_register_class))
                {
                    return fail("{s}: native component mismatch at {d}", .{ path, acc.native_start });
                }
            }
            if (std.mem.eql(u8, dec.components[0].base, "sp")) {
                const is_fp_lr = dec.components.len == 2 and
                    dec.components[0].data_register == 29 and
                    std.mem.eql(u8, dec.components[0].data_register_class, "gpr") and
                    dec.components[1].data_register == 30 and
                    std.mem.eql(u8, dec.components[1].data_register_class, "gpr");
                const expected_sp_detail = if (is_fp_lr)
                    (if (std.mem.eql(u8, dec.kind, "store")) "prologue_saved_fp_lr" else "epilogue_restored_fp_lr")
                else
                    "outgoing_abi_frame";
                for (acc.components) |ac| {
                    if (!optStrEql(ac.detail, expected_sp_detail)) {
                        return fail("{s}: SP frame detail mismatch at {d}; expected {s}", .{ path, acc.native_start, expected_sp_detail });
                    }
                }
            }
        }
    }

    for (values) |val| {
        const counts = resolved_by_vreg.get(val.vreg) orelse VregCounts{};
        if (val.reload_count != counts.load or val.store_count != counts.store) {
            return fail(
                "{s}: vreg {d} emitted count mismatch ({d}/{d} != {d}/{d})",
                .{ path, val.vreg, val.reload_count, val.store_count, counts.load, counts.store },
            );
        }
    }

    const declared_loads = try requireIntField(root_obj, "emitted_allocator_loads");
    const declared_stores = try requireIntField(root_obj, "emitted_allocator_stores");
    if (declared_loads != emitted_loads or declared_stores != emitted_stores) {
        return fail(
            "{s}: emitted allocator counts disagree with access records ({d}/{d} != {d}/{d})",
            .{ path, declared_loads, declared_stores, emitted_loads, emitted_stores },
        );
    }
    if (emitted_loads != metric.spill_ld or emitted_stores != metric.spill_st) {
        return fail(
            "{s}: allocator access reconciliation failed: emitted ld/st={d}/{d}, WAMR_AOT_SPILL_METRIC={d}/{d}",
            .{ path, emitted_loads, emitted_stores, metric.spill_ld, metric.spill_st },
        );
    }

    return .{
        .schema = schema,
        .schema_version = schema_version,
        .cwasm_aot_version = cwasm_version,
        .compiler_build_id = compiler_build_id,
        .architecture = architecture,
        .abi = abi,
        .module = module_id,
        .local_func = local_func,
        .function_name = try optionalStringField(root_obj, "function_name"),
        .module_text_size = @intCast(module_text_size),
        .module_text_sha256 = module_hash,
        .function_offset = function_offset,
        .code_size = @intCast(code_size),
        .normalized_code_sha256 = expected_hash,
        .frame_layout = layout,
        .frame_regions = regions,
        .spill_metric = metric,
        .emitted_allocator_loads = emitted_loads,
        .emitted_allocator_stores = emitted_stores,
        .allocator_values = values,
        .inline_data_ranges = inline_data_ranges,
        .accesses = accesses,
        .reconciliation = .{
            .emitted_allocator_loads = emitted_loads,
            .emitted_allocator_stores = emitted_stores,
            .spill_metric_loads = metric.spill_ld,
            .spill_metric_stores = metric.spill_st,
            .matches = true,
        },
    };
}

pub fn loadFrameMetadata(
    allocator: Allocator,
    io: std.Io,
    path: []const u8,
    func_index: i64,
    function_code: []const u8,
    cwasm_version: u32,
    module_text: []const u8,
    function_offset: usize,
    artifact_info: CwasmInfo,
) AttributionError!FrameMetadata {
    const json_text = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024)) catch |err| {
        return fail("{s}: malformed frame metadata: {s}", .{ path, @errorName(err) });
    };
    return loadFrameMetadataFromJson(
        allocator,
        path,
        json_text,
        func_index,
        function_code,
        cwasm_version,
        module_text,
        function_offset,
        artifact_info,
    );
}

pub fn validateMetadataDisassembly(
    allocator: Allocator,
    metadata: *const FrameMetadata,
    instructions: []const Instruction,
) AttributionError!void {
    const architecture = metadata.architecture;
    for (metadata.accesses) |access| {
        const start = access.native_start;
        var matched_inst: ?Instruction = null;
        for (instructions) |inst| {
            if (inst.offset == start) {
                matched_inst = inst;
                break;
            }
        }
        const inst = matched_inst orelse {
            return fail("metadata access at +0x{x} is not an instruction boundary", .{start});
        };
        if (start + inst.size != access.native_end) {
            return fail(
                "metadata range [{d},{d}) disagrees with objdump instruction size {d}",
                .{ start, access.native_end, inst.size },
            );
        }
        const operand_opt = if (std.mem.eql(u8, architecture, "x86_64"))
            try parseFrameOperand(allocator, inst.text)
        else
            try parseAarch64MemoryOperand(allocator, inst.text, false);
        const operand = operand_opt orelse {
            return fail("metadata access at +0x{x} is not a frame load/store: {s}", .{ start, inst.text });
        };
        const expected_base = if (metadata.schema_version == 1) access.base else (access.encoded_base orelse access.base);
        const expected_offset = if (metadata.schema_version == 1) access.frame_offset else (access.encoded_offset orelse access.frame_offset);
        if (!std.mem.eql(u8, operand.kind, access.kind) or !std.mem.eql(u8, operand.base, expected_base)) {
            return fail(
                "metadata access kind/base mismatch at +0x{x}: {s}/{s} vs {s}/{s}",
                .{ start, access.kind, expected_base, operand.kind, operand.base },
            );
        }
        if (operand.offset != null and operand.offset.? != expected_offset) {
            return fail(
                "metadata frame offset mismatch at +0x{x}: {d} vs {d}",
                .{ start, expected_offset, operand.offset.? },
            );
        }
    }
}

fn containsBoundsCmpDisplacement(text: []const u8) bool {
    const patterns = [_][]const u8{
        "[rbx+0x8]",
        "[r10+0x8]",
        "[r11+0x8]",
    };
    for (patterns) |p| {
        if (std.mem.indexOf(u8, text, p) != null) return true;
    }
    return false;
}

fn isDispatchJmp(text: []const u8) bool {
    var tok_it = std.mem.tokenizeAny(u8, text, " \t");
    const first = tok_it.next() orelse return false;
    if (!std.mem.eql(u8, first, "jmp")) return false;
    const target = std.mem.trim(u8, tok_it.rest(), " \t");
    return std.mem.eql(u8, target, "rax") or
        std.mem.eql(u8, target, "r10") or
        std.mem.eql(u8, target, "r11");
}

pub fn classifyX8664Basic(allocator: Allocator, text: []const u8) ![]const u8 {
    if (try parseFrameOperand(allocator, text)) |frame| {
        if (std.mem.eql(u8, frame.kind, "load")) return "frame_load_unattributed";
        return "frame_store_unattributed";
    }
    const split = try splitOperands(allocator, text);
    const mnemonic = split.mnemonic;
    const mov_mnemonics = [_][]const u8{
        "mov",    "movzx",  "movsx",  "movsxd", "movsd", "movss",
        "movdqu", "movdqa", "movaps", "movups", "movq",  "movd",
    };
    var is_mov = false;
    for (mov_mnemonics) |m| {
        if (std.mem.eql(u8, mnemonic, m)) {
            is_mov = true;
            break;
        }
    }
    if (std.mem.eql(u8, mnemonic, "cmp") and containsBoundsCmpDisplacement(text)) {
        return "bounds_cmp";
    }
    const bounds_branches = [_][]const u8{ "ja", "jae", "jb", "jbe" };
    for (bounds_branches) |b| {
        if (std.mem.eql(u8, mnemonic, b)) return "bounds_branch";
    }
    if (std.mem.eql(u8, mnemonic, "call")) return "call";
    if (std.mem.eql(u8, mnemonic, "jmp")) {
        return if (isDispatchJmp(text)) "dispatch_jmp" else "jmp";
    }
    const cond_branches = [_][]const u8{
        "je", "jne", "jl", "jle", "jg", "jge",
        "js", "jns", "jp", "jnp", "jo", "jno",
    };
    for (cond_branches) |cb| {
        if (std.mem.eql(u8, mnemonic, cb)) return "cond_branch";
    }
    if (is_mov and std.mem.indexOfScalar(u8, text, '[') != null and std.mem.indexOf(u8, text, "rip") == null) {
        return "mem_access";
    }
    if (is_mov) return "regmov";

    const alu_mnemonics = [_][]const u8{
        "add",  "sub",   "and",  "or",   "xor", "shl", "shr",  "sar",
        "imul", "mul",   "inc",  "dec",  "neg", "not", "test", "lea",
        "sete", "setne", "seta", "cdqe", "cqo",
    };
    for (alu_mnemonics) |a| {
        if (std.mem.eql(u8, mnemonic, a)) return "alu";
    }
    return "other";
}

fn isAarch64Register(allocator: Allocator, value: []const u8) !bool {
    const trimmed = std.mem.trim(u8, value, " \t");
    if (trimmed.len == 0) return false;
    var lower = try allocator.alloc(u8, trimmed.len);
    for (trimmed, 0..) |c, i| lower[i] = std.ascii.toLower(c);
    if (std.mem.eql(u8, lower, "sp") or std.mem.eql(u8, lower, "xzr") or std.mem.eql(u8, lower, "wzr")) {
        return true;
    }
    const first = lower[0];
    const num = std.fmt.parseInt(u32, lower[1..], 10) catch return false;
    if (first == 'x' or first == 'w') return num <= 30;
    if (first == 'b' or first == 'h' or first == 's' or first == 'd' or first == 'q' or first == 'v') return num <= 31;
    return false;
}

fn hasLinearMemoryBaseX20(text: []const u8) bool {
    // Matches `\[\s*x20(?:\s*,|\s*\])` case-insensitively
    var pos: usize = 0;
    while (std.mem.indexOfScalarPos(u8, text, pos, '[')) |open_br| {
        var i = open_br + 1;
        while (i < text.len and std.ascii.isWhitespace(text[i])) : (i += 1) {}
        if (i + 3 <= text.len and (text[i] == 'x' or text[i] == 'X') and text[i + 1] == '2' and text[i + 2] == '0') {
            var j = i + 3;
            while (j < text.len and std.ascii.isWhitespace(text[j])) : (j += 1) {}
            if (j < text.len and (text[j] == ',' or text[j] == ']')) return true;
        }
        pos = open_br + 1;
    }
    return false;
}

fn hasBoundsCheckWindowX19Offset8(window: []const u8) bool {
    // Matches `\[\s*x19\s*,\s*#?(?:0x)?0*8\s*\]` case-insensitively
    var pos: usize = 0;
    while (std.mem.indexOfScalarPos(u8, window, pos, '[')) |open_br| {
        pos = open_br + 1;
        const close_br = std.mem.indexOfScalarPos(u8, window, pos, ']') orelse continue;
        const inner = std.mem.trim(u8, window[pos..close_br], " \t");
        if (inner.len < 3) continue;
        if ((inner[0] != 'x' and inner[0] != 'X') or inner[1] != '1' or inner[2] != '9') continue;
        const after_reg = std.mem.trimStart(u8, inner[3..], " \t");
        if (after_reg.len == 0 or after_reg[0] != ',') continue;
        var disp = std.mem.trim(u8, after_reg[1..], " \t");
        if (disp.len > 0 and disp[0] == '#') disp = std.mem.trimStart(u8, disp[1..], " \t");
        if (std.mem.startsWith(u8, disp, "0x") or std.mem.startsWith(u8, disp, "0X")) disp = disp[2..];
        while (disp.len > 1 and disp[0] == '0') disp = disp[1..];
        if (std.mem.eql(u8, disp, "8")) return true;
    }
    return false;
}

pub fn classifyAarch64Basic(allocator: Allocator, text: []const u8) ![]const u8 {
    if (try parseAarch64FrameOperand(allocator, text)) |frame| {
        if (std.mem.eql(u8, frame.kind, "load")) return "frame_load_unattributed";
        return "frame_store_unattributed";
    }
    const split = try splitOperands(allocator, text);
    const mnemonic = split.mnemonic;
    if (std.mem.eql(u8, mnemonic, "bl") or std.mem.eql(u8, mnemonic, "blr")) return "call";
    if (std.mem.eql(u8, mnemonic, "br")) return "indirect_dispatch";
    if (std.mem.eql(u8, mnemonic, "b")) return "direct_branch";
    if (std.mem.startsWith(u8, mnemonic, "b.") or
        std.mem.eql(u8, mnemonic, "cbz") or
        std.mem.eql(u8, mnemonic, "cbnz") or
        std.mem.eql(u8, mnemonic, "tbz") or
        std.mem.eql(u8, mnemonic, "tbnz"))
    {
        return "cond_branch";
    }
    const load_store = std.mem.startsWith(u8, mnemonic, "ldr") or
        std.mem.startsWith(u8, mnemonic, "ldur") or
        std.mem.startsWith(u8, mnemonic, "str") or
        std.mem.startsWith(u8, mnemonic, "stur") or
        std.mem.startsWith(u8, mnemonic, "ldp") or
        std.mem.startsWith(u8, mnemonic, "stp");
    if (load_store) {
        var has_br = false;
        for (split.operands) |op| {
            if (std.mem.indexOfScalar(u8, op, '[') != null) {
                has_br = true;
                break;
            }
        }
        if (has_br) {
            if (hasLinearMemoryBaseX20(text)) return "linear_memory";
            return "mem_access";
        }
    }
    if (std.mem.eql(u8, mnemonic, "mov") and
        split.operands.len == 2 and
        (try isAarch64Register(allocator, split.operands[0])) and
        (try isAarch64Register(allocator, split.operands[1])))
    {
        return "regmov";
    }
    const alu_mnemonics = [_][]const u8{
        "add",  "adds", "sub",  "subs", "adc",  "adcs",  "sbc",   "sbcs",
        "and",  "ands", "orr",  "eor",  "bic",  "bics",  "madd",  "msub",
        "mul",  "mneg", "udiv", "sdiv", "lsl",  "lsr",   "asr",   "ror",
        "extr", "ubfm", "sbfm", "bfm",  "uxtb", "uxth",  "sxtb",  "sxth",
        "sxtw", "cmp",  "cmn",  "tst",  "csel", "csinc", "csinv", "csneg",
        "adr",  "adrp",
    };
    for (alu_mnemonics) |a| {
        if (std.mem.eql(u8, mnemonic, a)) return "alu";
    }
    return "other";
}

pub fn classifyBasic(allocator: Allocator, text: []const u8, architecture: ?[]const u8) AttributionError![]const u8 {
    const arch = try normalizeArchitecture(allocator, architecture);
    if (std.mem.eql(u8, arch, "x86_64")) {
        return try classifyX8664Basic(allocator, text);
    }
    return try classifyAarch64Basic(allocator, text);
}

pub fn classifyInstruction(
    allocator: Allocator,
    instruction: Instruction,
    metadata: ?*const FrameMetadata,
    architecture: ?[]const u8,
) AttributionError![]const u8 {
    const arch = try normalizeArchitecture(allocator, architecture);
    if (metadata) |meta| {
        if (meta.accessAtStart(instruction.offset)) |access| {
            return try std.fmt.allocPrint(allocator, "{s}_{s}", .{ access.origin, access.kind });
        }
        if (try instructionFrameOperand(allocator, instruction.text, arch)) |frame| {
            return try std.fmt.allocPrint(allocator, "unknown_frame_{s}", .{frame.kind});
        }
    }
    return try classifyBasic(allocator, instruction.text, arch);
}

pub fn classifyInstructionStream(
    allocator: Allocator,
    instructions: []const Instruction,
    metadata: ?*const FrameMetadata,
    architecture: ?[]const u8,
) AttributionError![][]const u8 {
    const arch = try normalizeArchitecture(allocator, architecture);
    const classes = try allocator.alloc([]const u8, instructions.len);
    for (instructions, 0..) |inst, i| {
        classes[i] = try classifyInstruction(allocator, inst, metadata, arch);
    }
    if (!std.mem.eql(u8, arch, "aarch64")) {
        return classes;
    }

    if (instructions.len > 1) {
        var idx: usize = 0;
        while (idx + 1 < instructions.len) : (idx += 1) {
            const cur_split = try splitOperands(allocator, instructions[idx].text);
            const next_split = try splitOperands(allocator, instructions[idx + 1].text);
            const is_cmp = std.mem.eql(u8, cur_split.mnemonic, "cmp") or std.mem.eql(u8, cur_split.mnemonic, "cmn");
            const is_b = std.mem.eql(u8, next_split.mnemonic, "b.hi") or
                std.mem.eql(u8, next_split.mnemonic, "b.hs") or
                std.mem.eql(u8, next_split.mnemonic, "b.lo") or
                std.mem.eql(u8, next_split.mnemonic, "b.ls");
            if (!is_cmp or !is_b) continue;
            const win_start = if (idx >= 3) idx - 3 else 0;
            var matched = false;
            for (instructions[win_start..idx]) |prev_inst| {
                if (hasBoundsCheckWindowX19Offset8(prev_inst.text)) {
                    matched = true;
                    break;
                }
            }
            if (matched) {
                classes[idx] = "bounds_cmp";
                classes[idx + 1] = "bounds_branch";
            }
        }
    }
    return classes;
}

pub const FrameCoverage = struct {
    frame_instructions: u64,
    attributed_frame_instructions: u64,
    static_coverage_pct: f64,
    frame_samples: u64,
    attributed_frame_samples: u64,
    metadata_mapping_coverage_pct: f64,
    proven_origin_frame_instructions: u64,
    unknown_frame_instructions: u64,
    origin_coverage_pct: f64,
    proven_origin_frame_samples: u64,
    unknown_frame_samples: u64,
    sample_coverage_pct: f64,
    origin_sample_coverage_pct: f64,
};

pub const OriginSummary = struct {
    origin: []const u8,
    static_instructions: u64,
    samples: u64,
    percent_of_frame_samples: f64,
};

pub const PairedComponent = struct {
    slot: i64,
    frame_offset: i64,
    vreg: ?i64,
};

pub const AllocatorContributor = struct {
    slot: i64,
    frame_offset: i64,
    vreg: ?i64,
    vreg_ambiguous: bool,
    candidate_vregs: []const i64,
    paired_components: ?[]const PairedComponent = null,
    defining_opcode: ?[]const u8 = null,
    source_class: ?[]const u8 = null,
    rematerialization_eligible: ?bool = null,
    source_reload_count: ?i64 = null,
    source_store_count: ?i64 = null,
    source_ir_use_count: ?i64 = null,
    source_ir_def_count: ?i64 = null,
    static_loads: u64 = 0,
    static_stores: u64 = 0,
    samples: u64 = 0,
};

pub const AllocatorComponentCounts = struct {
    total_loads: u64 = 0,
    total_stores: u64 = 0,
    unranked_loads: u64 = 0,
    unranked_stores: u64 = 0,
};

pub const UnknownInstruction = struct {
    native_offset: usize,
    address: u64,
    samples: u64,
    instruction: []const u8,
    base: []const u8,
    frame_offset: ?i64,
    complex_address: bool,
    detail: ?[]const u8 = null,
};

pub const FrameSummary = struct {
    coverage: FrameCoverage,
    origins: []const OriginSummary,
    allocator_contributors: []const AllocatorContributor,
    allocator_component_counts: AllocatorComponentCounts,
    unknown_instructions: []const UnknownInstruction,
    reconciliation: Reconciliation,

    pub fn originByName(self: *const FrameSummary, name: []const u8) ?OriginSummary {
        for (self.origins) |o| {
            if (std.mem.eql(u8, o.origin, name)) return o;
        }
        return null;
    }
};

pub fn buildFrameSummary(
    allocator: Allocator,
    instructions: []const Instruction,
    counts: *const std.AutoHashMapUnmanaged(u64, u64),
    metadata: *const FrameMetadata,
) AttributionError!FrameSummary {
    const architecture = metadata.architecture;
    const schema_version = metadata.schema_version;

    const OriginCounters = struct { static_count: u64 = 0, sample_count: u64 = 0 };
    var origin_map: std.StringHashMapUnmanaged(OriginCounters) = .empty;

    var frame_instruction_count: u64 = 0;
    var frame_samples: u64 = 0;
    var attributed_instructions: u64 = 0;
    var attributed_samples: u64 = 0;
    var proven_origin_instructions: u64 = 0;
    var proven_origin_samples: u64 = 0;
    var unknown: std.ArrayList(UnknownInstruction) = .empty;
    var contributors: std.ArrayList(AllocatorContributor) = .empty;
    var comp_counts = AllocatorComponentCounts{};

    for (instructions) |inst| {
        const access_opt = metadata.accessAtStart(inst.offset);
        const operand_opt = try instructionFrameOperand(allocator, inst.text, architecture);
        if (access_opt == null and operand_opt == null) continue;
        const samples = counts.get(inst.address) orelse 0;
        frame_instruction_count += 1;
        frame_samples += samples;

        if (access_opt == null) {
            const operand = operand_opt.?;
            try unknown.append(allocator, .{
                .native_offset = inst.offset,
                .address = inst.address,
                .samples = samples,
                .instruction = inst.text,
                .base = operand.base,
                .frame_offset = operand.offset,
                .complex_address = operand.complex_address,
                .detail = null,
            });
            const gop = try origin_map.getOrPut(allocator, "unknown");
            if (!gop.found_existing) gop.value_ptr.* = .{};
            gop.value_ptr.static_count += 1;
            gop.value_ptr.sample_count += samples;
            continue;
        }

        const access = access_opt.?;
        const origin = access.origin;
        attributed_instructions += 1;
        attributed_samples += samples;
        const gop = try origin_map.getOrPut(allocator, origin);
        if (!gop.found_existing) gop.value_ptr.* = .{};
        gop.value_ptr.static_count += 1;
        gop.value_ptr.sample_count += samples;

        if (!std.mem.eql(u8, origin, "unknown")) {
            proven_origin_instructions += 1;
            proven_origin_samples += samples;
        } else {
            try unknown.append(allocator, .{
                .native_offset = inst.offset,
                .address = inst.address,
                .samples = samples,
                .instruction = inst.text,
                .base = access.base,
                .frame_offset = access.frame_offset,
                .complex_address = false,
                .detail = access.detail,
            });
        }

        var single_buf: [1]AccessComponent = undefined;
        const components: []const AccessComponent = if (schema_version == 2 and access.components.len > 0)
            access.components
        else blk: {
            single_buf[0] = .{
                .base = access.base,
                .frame_offset = access.frame_offset,
                .width = access.width,
                .origin = access.origin,
                .detail = access.detail,
                .slot = access.slot,
                .local_index = access.local_index,
                .explicit_slot = access.explicit_slot,
                .vreg = access.vreg,
                .vreg_ambiguous = access.vreg_ambiguous,
                .defining_opcode = access.defining_opcode,
                .source_class = access.source_class,
                .rematerialization_eligible = access.rematerialization_eligible,
                .ir_position = access.ir_position,
                .ir_opcode = access.ir_opcode,
            };
            break :blk single_buf[0..1];
        };

        var alloc_comps: std.ArrayList(AccessComponent) = .empty;
        for (components) |c| {
            if (std.mem.eql(u8, c.origin, "allocator_spill")) {
                try alloc_comps.append(allocator, c);
            }
        }
        const is_load = std.mem.eql(u8, access.kind, "load");
        if (is_load) {
            comp_counts.total_loads += alloc_comps.items.len;
        } else {
            comp_counts.total_stores += alloc_comps.items.len;
        }
        if (!std.mem.eql(u8, origin, "allocator_spill")) {
            if (is_load) {
                comp_counts.unranked_loads += alloc_comps.items.len;
            } else {
                comp_counts.unranked_stores += alloc_comps.items.len;
            }
            continue;
        }

        var unique_indices: std.ArrayList(usize) = .empty;
        for (alloc_comps.items) |comp| {
            const slot = comp.slot orelse 0;
            const vreg = comp.vreg;
            var found_idx: ?usize = null;
            for (contributors.items, 0..) |existing, c_idx| {
                if (existing.paired_components == null and existing.slot == slot and existing.vreg == vreg) {
                    found_idx = c_idx;
                    break;
                }
            }
            if (found_idx == null) {
                const val_opt = if (vreg) |vr| metadata.valueByVreg(vr) else null;
                const slot_vals = try metadata.valuesForSlot(allocator, slot);
                var cand_vregs = try allocator.alloc(i64, slot_vals.len);
                for (slot_vals, 0..) |sv, svi| cand_vregs[svi] = sv.vreg;
                found_idx = contributors.items.len;
                try contributors.append(allocator, .{
                    .slot = slot,
                    .frame_offset = comp.frame_offset,
                    .vreg = vreg,
                    .vreg_ambiguous = comp.vreg_ambiguous,
                    .candidate_vregs = cand_vregs,
                    .paired_components = null,
                    .defining_opcode = if (val_opt) |v| v.defining_opcode else null,
                    .source_class = if (val_opt) |v| v.source_class else null,
                    .rematerialization_eligible = if (val_opt) |v| v.rematerialization_eligible else null,
                    .source_reload_count = if (val_opt) |v| v.reload_count else null,
                    .source_store_count = if (val_opt) |v| v.store_count else null,
                    .source_ir_use_count = if (val_opt) |v| v.ir_use_count else null,
                    .source_ir_def_count = if (val_opt) |v| v.ir_def_count else null,
                    .static_loads = 0,
                    .static_stores = 0,
                    .samples = 0,
                });
            }
            const c_idx = found_idx.?;
            if (is_load) {
                contributors.items[c_idx].static_loads += 1;
            } else {
                contributors.items[c_idx].static_stores += 1;
            }
            var already_in_unique = false;
            for (unique_indices.items) |u| {
                if (u == c_idx) {
                    already_in_unique = true;
                    break;
                }
            }
            if (!already_in_unique) {
                try unique_indices.append(allocator, c_idx);
            }
        }

        if (unique_indices.items.len == 1) {
            contributors.items[unique_indices.items[0]].samples += samples;
        } else if (unique_indices.items.len > 1) {
            var min_slot: i64 = alloc_comps.items[0].slot orelse 0;
            var min_off: i64 = alloc_comps.items[0].frame_offset;
            var cand_list: std.ArrayList(i64) = .empty;
            var paired_list: std.ArrayList(PairedComponent) = .empty;
            for (alloc_comps.items) |comp| {
                const s = comp.slot orelse 0;
                if (s < min_slot) min_slot = s;
                if (comp.frame_offset < min_off) min_off = comp.frame_offset;
                if (comp.vreg) |vr| {
                    var dup = false;
                    for (cand_list.items) |existing_vr| {
                        if (existing_vr == vr) {
                            dup = true;
                            break;
                        }
                    }
                    if (!dup) try cand_list.append(allocator, vr);
                }
                try paired_list.append(allocator, .{
                    .slot = s,
                    .frame_offset = comp.frame_offset,
                    .vreg = comp.vreg,
                });
            }
            std.mem.sort(i64, cand_list.items, {}, std.sort.asc(i64));
            try contributors.append(allocator, .{
                .slot = min_slot,
                .frame_offset = min_off,
                .vreg = null,
                .vreg_ambiguous = true,
                .candidate_vregs = try cand_list.toOwnedSlice(allocator),
                .paired_components = try paired_list.toOwnedSlice(allocator),
                .defining_opcode = null,
                .source_class = "paired_allocator_access",
                .rematerialization_eligible = null,
                .source_reload_count = null,
                .source_store_count = null,
                .source_ir_use_count = null,
                .source_ir_def_count = null,
                .static_loads = 0,
                .static_stores = 0,
                .samples = samples,
            });
        }
    }

    const ranked = try contributors.toOwnedSlice(allocator);
    std.mem.sort(AllocatorContributor, ranked, {}, struct {
        fn lessThan(_: void, a: AllocatorContributor, b: AllocatorContributor) bool {
            if (a.samples != b.samples) return a.samples > b.samples;
            const a_static = a.static_loads + a.static_stores;
            const b_static = b.static_loads + b.static_stores;
            if (a_static != b_static) return a_static > b_static;
            if (a.slot != b.slot) return a.slot < b.slot;
            const a_vreg: i64 = a.vreg orelse -1;
            const b_vreg: i64 = b.vreg orelse -1;
            return a_vreg < b_vreg;
        }
    }.lessThan);

    const unknown_slice = try unknown.toOwnedSlice(allocator);
    std.mem.sort(UnknownInstruction, unknown_slice, {}, struct {
        fn lessThan(_: void, a: UnknownInstruction, b: UnknownInstruction) bool {
            if (a.samples != b.samples) return a.samples > b.samples;
            return a.native_offset < b.native_offset;
        }
    }.lessThan);

    var origins_list: std.ArrayList(OriginSummary) = .empty;
    var it = origin_map.iterator();
    while (it.next()) |entry| {
        const sc = entry.value_ptr.static_count;
        const sm = entry.value_ptr.sample_count;
        try origins_list.append(allocator, .{
            .origin = entry.key_ptr.*,
            .static_instructions = sc,
            .samples = sm,
            .percent_of_frame_samples = percent(sm, frame_samples),
        });
    }
    const origins_slice = try origins_list.toOwnedSlice(allocator);
    std.mem.sort(OriginSummary, origins_slice, {}, struct {
        fn lessThan(_: void, a: OriginSummary, b: OriginSummary) bool {
            return std.mem.order(u8, a.origin, b.origin) == .lt;
        }
    }.lessThan);

    return .{
        .coverage = .{
            .frame_instructions = frame_instruction_count,
            .attributed_frame_instructions = attributed_instructions,
            .static_coverage_pct = percent(attributed_instructions, frame_instruction_count),
            .frame_samples = frame_samples,
            .attributed_frame_samples = attributed_samples,
            .metadata_mapping_coverage_pct = percent(attributed_samples, frame_samples),
            .proven_origin_frame_instructions = proven_origin_instructions,
            .unknown_frame_instructions = frame_instruction_count - proven_origin_instructions,
            .origin_coverage_pct = percent(proven_origin_instructions, frame_instruction_count),
            .proven_origin_frame_samples = proven_origin_samples,
            .unknown_frame_samples = frame_samples - proven_origin_samples,
            .sample_coverage_pct = percent(attributed_samples, frame_samples),
            .origin_sample_coverage_pct = percent(proven_origin_samples, frame_samples),
        },
        .origins = origins_slice,
        .allocator_contributors = ranked,
        .allocator_component_counts = comp_counts,
        .unknown_instructions = unknown_slice,
        .reconciliation = metadata.reconciliation,
    };
}

fn writeJsonString(writer: *std.Io.Writer, s: []const u8) !void {
    try std.json.Stringify.encodeJsonString(s, .{}, writer);
}

fn writeOptJsonString(writer: *std.Io.Writer, s_opt: ?[]const u8) !void {
    if (s_opt) |s| {
        try writeJsonString(writer, s);
    } else {
        try writer.writeAll("null");
    }
}

fn writeOptJsonInt(writer: *std.Io.Writer, v_opt: ?i64) !void {
    if (v_opt) |v| {
        try writer.print("{d}", .{v});
    } else {
        try writer.writeAll("null");
    }
}

fn writeOptJsonBool(writer: *std.Io.Writer, b_opt: ?bool) !void {
    if (b_opt) |b| {
        try writer.writeAll(if (b) "true" else "false");
    } else {
        try writer.writeAll("null");
    }
}

pub fn writeFrameSummaryJson(writer: *std.Io.Writer, summary: *const FrameSummary) !void {
    try writer.writeAll("{\"allocator_component_counts\":{");
    try writer.print(
        "\"total_loads\":{d},\"total_stores\":{d},\"unranked_loads\":{d},\"unranked_stores\":{d}",
        .{
            summary.allocator_component_counts.total_loads,
            summary.allocator_component_counts.total_stores,
            summary.allocator_component_counts.unranked_loads,
            summary.allocator_component_counts.unranked_stores,
        },
    );
    try writer.writeAll("},\"allocator_contributors\":[");
    for (summary.allocator_contributors, 0..) |c, idx| {
        if (idx > 0) try writer.writeAll(",");
        try writer.writeAll("{\"candidate_vregs\":[");
        for (c.candidate_vregs, 0..) |cv, cvi| {
            if (cvi > 0) try writer.writeAll(",");
            try writer.print("{d}", .{cv});
        }
        try writer.writeAll("],\"defining_opcode\":");
        try writeOptJsonString(writer, c.defining_opcode);
        try writer.print(",\"frame_offset\":{d}", .{c.frame_offset});
        if (c.paired_components) |pcs| {
            try writer.writeAll(",\"paired_components\":[");
            for (pcs, 0..) |pc, pci| {
                if (pci > 0) try writer.writeAll(",");
                try writer.print("{{\"frame_offset\":{d},\"slot\":{d},\"vreg\":", .{ pc.frame_offset, pc.slot });
                try writeOptJsonInt(writer, pc.vreg);
                try writer.writeAll("}");
            }
            try writer.writeAll("]");
        }
        try writer.writeAll(",\"rematerialization_eligible\":");
        try writeOptJsonBool(writer, c.rematerialization_eligible);
        try writer.print(",\"samples\":{d},\"slot\":{d},\"source_class\":", .{ c.samples, c.slot });
        try writeOptJsonString(writer, c.source_class);
        try writer.writeAll(",\"source_ir_def_count\":");
        try writeOptJsonInt(writer, c.source_ir_def_count);
        try writer.writeAll(",\"source_ir_use_count\":");
        try writeOptJsonInt(writer, c.source_ir_use_count);
        try writer.writeAll(",\"source_reload_count\":");
        try writeOptJsonInt(writer, c.source_reload_count);
        try writer.writeAll(",\"source_store_count\":");
        try writeOptJsonInt(writer, c.source_store_count);
        try writer.print(",\"static_loads\":{d},\"static_stores\":{d},\"vreg\":", .{ c.static_loads, c.static_stores });
        try writeOptJsonInt(writer, c.vreg);
        try writer.print(",\"vreg_ambiguous\":{s}}}", .{if (c.vreg_ambiguous) "true" else "false"});
    }
    const cov = summary.coverage;
    try writer.writeAll("],\"coverage\":{");
    try writer.print(
        "\"attributed_frame_instructions\":{d},\"attributed_frame_samples\":{d},\"frame_instructions\":{d},\"frame_samples\":{d},\"metadata_mapping_coverage_pct\":{d},\"origin_coverage_pct\":{d},\"origin_sample_coverage_pct\":{d},\"proven_origin_frame_instructions\":{d},\"proven_origin_frame_samples\":{d},\"sample_coverage_pct\":{d},\"static_coverage_pct\":{d},\"unknown_frame_instructions\":{d},\"unknown_frame_samples\":{d}",
        .{
            cov.attributed_frame_instructions,
            cov.attributed_frame_samples,
            cov.frame_instructions,
            cov.frame_samples,
            cov.metadata_mapping_coverage_pct,
            cov.origin_coverage_pct,
            cov.origin_sample_coverage_pct,
            cov.proven_origin_frame_instructions,
            cov.proven_origin_frame_samples,
            cov.sample_coverage_pct,
            cov.static_coverage_pct,
            cov.unknown_frame_instructions,
            cov.unknown_frame_samples,
        },
    );
    try writer.writeAll("},\"origins\":{");
    for (summary.origins, 0..) |o, idx| {
        if (idx > 0) try writer.writeAll(",");
        try writeJsonString(writer, o.origin);
        try writer.print(
            ":{{\"percent_of_frame_samples\":{d},\"samples\":{d},\"static_instructions\":{d}}}",
            .{ o.percent_of_frame_samples, o.samples, o.static_instructions },
        );
    }
    const rec = summary.reconciliation;
    try writer.print(
        "}},\"reconciliation\":{{\"emitted_allocator_loads\":{d},\"emitted_allocator_stores\":{d},\"matches\":{s},\"spill_metric_loads\":{d},\"spill_metric_stores\":{d}}},\"unknown_instructions\":[",
        .{
            rec.emitted_allocator_loads,
            rec.emitted_allocator_stores,
            if (rec.matches) "true" else "false",
            rec.spill_metric_loads,
            rec.spill_metric_stores,
        },
    );
    for (summary.unknown_instructions, 0..) |u, idx| {
        if (idx > 0) try writer.writeAll(",");
        try writer.print("{{\"address\":{d},\"base\":", .{u.address});
        try writeJsonString(writer, u.base);
        try writer.print(",\"complex_address\":{s}", .{if (u.complex_address) "true" else "false"});
        if (u.detail) |d| {
            try writer.writeAll(",\"detail\":");
            try writeJsonString(writer, d);
        }
        try writer.writeAll(",\"frame_offset\":");
        try writeOptJsonInt(writer, u.frame_offset);
        try writer.writeAll(",\"instruction\":");
        try writeJsonString(writer, u.instruction);
        try writer.print(",\"native_offset\":{d},\"samples\":{d}}}", .{ u.native_offset, u.samples });
    }
    try writer.writeAll("]}");
}

pub fn printFrameSummary(allocator: Allocator, writer: *std.Io.Writer, summary: *const FrameSummary, total_samples: u64, top: usize) !void {
    const cov = summary.coverage;
    const rec = summary.reconciliation;
    try writer.writeAll("\n=== frame-origin attribution ===\n");
    try writer.print(
        "  metadata mapping: {d}/{d} static frame instructions ({d:.1}%); {d}/{d} frame samples ({d:.1}%)\n",
        .{
            cov.attributed_frame_instructions,
            cov.frame_instructions,
            cov.static_coverage_pct,
            cov.attributed_frame_samples,
            cov.frame_samples,
            cov.sample_coverage_pct,
        },
    );
    try writer.print(
        "  proven origins: {d}/{d} static frame instructions ({d:.1}%); {d}/{d} frame samples ({d:.1}%)\n",
        .{
            cov.proven_origin_frame_instructions,
            cov.frame_instructions,
            cov.origin_coverage_pct,
            cov.proven_origin_frame_samples,
            cov.frame_samples,
            cov.origin_sample_coverage_pct,
        },
    );
    try writer.print(
        "  reconciliation: emitted allocator ld/st={d}/{d} == WAMR_AOT_SPILL_METRIC {d}/{d} (match)\n",
        .{
            rec.emitted_allocator_loads,
            rec.emitted_allocator_stores,
            rec.spill_metric_loads,
            rec.spill_metric_stores,
        },
    );
    const comps = summary.allocator_component_counts;
    if (comps.unranked_loads != 0 or comps.unranked_stores != 0) {
        try writer.print(
            "  mixed-origin allocator components outside sample ranking: ld/st={d}/{d}; instruction samples remain unknown\n",
            .{ comps.unranked_loads, comps.unranked_stores },
        );
    }

    const sorted_origins = try allocator.dupe(OriginSummary, summary.origins);
    defer allocator.free(sorted_origins);
    std.mem.sort(OriginSummary, sorted_origins, {}, struct {
        fn lessThan(_: void, a: OriginSummary, b: OriginSummary) bool {
            if (a.samples != b.samples) return a.samples > b.samples;
            return std.mem.order(u8, a.origin, b.origin) == .lt;
        }
    }.lessThan);
    for (sorted_origins) |o| {
        try writer.print(
            "  {s:<27} static={d:<6} samples={d:<7} ({d:.2}% of run)\n",
            .{ o.origin, o.static_instructions, o.samples, percent(o.samples, total_samples) },
        );
    }

    const limit = @min(top, summary.allocator_contributors.len);
    const contributors = summary.allocator_contributors[0..limit];
    if (contributors.len > 0) {
        try writer.print("\n=== top {d} allocator slot/source contributors ===\n", .{contributors.len});
        for (contributors) |item| {
            var id_buf: std.Io.Writer.Allocating = .init(allocator);
            defer id_buf.deinit();
            if (item.vreg) |vr| {
                try id_buf.writer.print("v{d}", .{vr});
            } else {
                try id_buf.writer.writeAll("ambiguous[");
                for (item.candidate_vregs, 0..) |cv, cvi| {
                    if (cvi > 0) try id_buf.writer.writeAll(", ");
                    try id_buf.writer.print("{d}", .{cv});
                }
                try id_buf.writer.writeAll("]");
            }
            const identity = try id_buf.toOwnedSlice();
            defer allocator.free(identity);
            try writer.print(
                "  samples={d:<6} slot={d:<5} off={d:<7} {s:<20} src={s}/{s} static_ld/st={d}/{d} ir_use/def=",
                .{
                    item.samples,
                    item.slot,
                    item.frame_offset,
                    identity,
                    item.defining_opcode orelse "?",
                    item.source_class orelse "?",
                    item.static_loads,
                    item.static_stores,
                },
            );
            if (item.source_ir_use_count) |u| {
                try writer.print("{d}/", .{u});
            } else {
                try writer.writeAll("None/");
            }
            if (item.source_ir_def_count) |d| {
                try writer.print("{d}\n", .{d});
            } else {
                try writer.writeAll("None\n");
            }
        }
    }
    if (cov.unknown_frame_instructions != 0) {
        try writer.print(
            "\n  unknown frame instructions: {d} ({d} samples)\n",
            .{ cov.unknown_frame_instructions, cov.unknown_frame_samples },
        );
    }
}

fn resolveRealOrGivenPath(allocator: Allocator, io: std.Io, path: []const u8) ![]const u8 {
    return std.Io.Dir.cwd().realPathFileAlloc(io, path, allocator) catch try allocator.dupe(u8, path);
}

fn parentDirOrDot(path: []const u8) []const u8 {
    return std.fs.path.dirname(path) orelse ".";
}

pub fn runCli(allocator: Allocator, io: std.Io, args: []const []const u8, stdout: *std.Io.Writer) AttributionError!void {
    var perf_opt: ?[]const u8 = null;
    var cwasm_opt: ?[]const u8 = null;
    var func_opt: ?i64 = null;
    var top: usize = 20;
    var base_opt: ?[]const u8 = null;
    var arch_opt: ?[]const u8 = null;
    var json_out_opt: ?[]const u8 = null;
    var min_samples: u64 = 0;
    var min_attribution_pct: f64 = 0.0;
    var authoritative = false;
    var frame_metadata_opt: ?[]const u8 = null;
    var validate_frame_metadata = false;
    var wamrc_smoke_opt: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--perf")) {
            if (i + 1 >= args.len) return fail("--perf requires a value", .{});
            i += 1;
            perf_opt = args[i];
        } else if (std.mem.startsWith(u8, arg, "--perf=")) {
            perf_opt = arg["--perf=".len..];
        } else if (std.mem.eql(u8, arg, "--cwasm")) {
            if (i + 1 >= args.len) return fail("--cwasm requires a value", .{});
            i += 1;
            cwasm_opt = args[i];
        } else if (std.mem.startsWith(u8, arg, "--cwasm=")) {
            cwasm_opt = arg["--cwasm=".len..];
        } else if (std.mem.eql(u8, arg, "--func")) {
            if (i + 1 >= args.len) return fail("--func requires a value", .{});
            i += 1;
            func_opt = std.fmt.parseInt(i64, args[i], 10) catch return fail("invalid --func", .{});
        } else if (std.mem.startsWith(u8, arg, "--func=")) {
            func_opt = std.fmt.parseInt(i64, arg["--func=".len..], 10) catch return fail("invalid --func", .{});
        } else if (std.mem.eql(u8, arg, "--top")) {
            if (i + 1 >= args.len) return fail("--top requires a value", .{});
            i += 1;
            top = std.fmt.parseInt(usize, args[i], 10) catch return fail("invalid --top", .{});
        } else if (std.mem.startsWith(u8, arg, "--top=")) {
            top = std.fmt.parseInt(usize, arg["--top=".len..], 10) catch return fail("invalid --top", .{});
        } else if (std.mem.eql(u8, arg, "--base")) {
            if (i + 1 >= args.len) return fail("--base requires a value", .{});
            i += 1;
            base_opt = args[i];
        } else if (std.mem.startsWith(u8, arg, "--base=")) {
            base_opt = arg["--base=".len..];
        } else if (std.mem.eql(u8, arg, "--arch")) {
            if (i + 1 >= args.len) return fail("--arch requires a value", .{});
            i += 1;
            arch_opt = args[i];
        } else if (std.mem.startsWith(u8, arg, "--arch=")) {
            arch_opt = arg["--arch=".len..];
        } else if (std.mem.eql(u8, arg, "--json-out")) {
            if (i + 1 >= args.len) return fail("--json-out requires a value", .{});
            i += 1;
            json_out_opt = args[i];
        } else if (std.mem.startsWith(u8, arg, "--json-out=")) {
            json_out_opt = arg["--json-out=".len..];
        } else if (std.mem.eql(u8, arg, "--min-samples")) {
            if (i + 1 >= args.len) return fail("--min-samples requires a value", .{});
            i += 1;
            min_samples = std.fmt.parseInt(u64, args[i], 10) catch return fail("invalid --min-samples", .{});
        } else if (std.mem.startsWith(u8, arg, "--min-samples=")) {
            min_samples = std.fmt.parseInt(u64, arg["--min-samples=".len..], 10) catch return fail("invalid --min-samples", .{});
        } else if (std.mem.eql(u8, arg, "--min-attribution-pct")) {
            if (i + 1 >= args.len) return fail("--min-attribution-pct requires a value", .{});
            i += 1;
            min_attribution_pct = std.fmt.parseFloat(f64, args[i]) catch return fail("invalid --min-attribution-pct", .{});
        } else if (std.mem.startsWith(u8, arg, "--min-attribution-pct=")) {
            min_attribution_pct = std.fmt.parseFloat(f64, arg["--min-attribution-pct=".len..]) catch return fail("invalid --min-attribution-pct", .{});
        } else if (std.mem.eql(u8, arg, "--authoritative")) {
            authoritative = true;
        } else if (std.mem.eql(u8, arg, "--require-size-match")) {
            // Accepted for compatibility
        } else if (std.mem.eql(u8, arg, "--frame-metadata")) {
            if (i + 1 >= args.len) return fail("--frame-metadata requires a value", .{});
            i += 1;
            frame_metadata_opt = args[i];
        } else if (std.mem.startsWith(u8, arg, "--frame-metadata=")) {
            frame_metadata_opt = arg["--frame-metadata=".len..];
        } else if (std.mem.eql(u8, arg, "--validate-frame-metadata")) {
            validate_frame_metadata = true;
        } else if (std.mem.eql(u8, arg, "--wamrc")) {
            if (i + 1 >= args.len) return fail("--wamrc requires a path", .{});
            i += 1;
            wamrc_smoke_opt = args[i];
        } else if (std.mem.startsWith(u8, arg, "--wamrc=")) {
            wamrc_smoke_opt = arg["--wamrc=".len..];
        } else {
            return fail("unrecognized argument: {s}", .{arg});
        }
    }

    if (wamrc_smoke_opt) |wamrc_path| {
        return runCompilerSmokeTests(allocator, io, wamrc_path);
    }

    const cwasm = cwasm_opt orelse return fail("--cwasm is required", .{});
    const arch = try normalizeArchitecture(allocator, arch_opt);
    if (perf_opt == null and !validate_frame_metadata) {
        return fail("--perf is required unless --validate-frame-metadata is set", .{});
    }
    if (validate_frame_metadata and (func_opt == null or frame_metadata_opt == null)) {
        return fail("--validate-frame-metadata requires --func and --frame-metadata", .{});
    }
    if (frame_metadata_opt != null and func_opt == null) {
        return fail("--frame-metadata requires --func", .{});
    }
    if (min_attribution_pct < 0.0 or min_attribution_pct > 100.0) {
        return fail("--min-attribution-pct must be between 0 and 100", .{});
    }
    if (authoritative and base_opt != null) {
        return fail("--base is a non-authoritative diagnostic override", .{});
    }

    const info = try parseCwasm(allocator, io, cwasm);
    if (!std.mem.eql(u8, info.architecture, arch)) {
        return fail("artifact target architecture {s} does not match --arch {s}", .{ info.architecture, arch });
    }

    var counts: std.AutoHashMapUnmanaged(u64, u64) = .empty;
    var total: u64 = 0;
    var base_u64_opt: ?u64 = null;
    var mapping_opt: ?TextMappingSelection = null;

    if (perf_opt) |perf| {
        const ac = try addrCounts(allocator, io, perf);
        counts = ac.counts;
        total = ac.total;
        if (total == 0) {
            return fail("no samples in perf data", .{});
        }
        if (total < min_samples) {
            return fail("perf data has only {d} self samples; --min-samples requires at least {d}", .{ total, min_samples });
        }
        const mapping = try selectTextMapping(allocator, io, perf, info.text_size, base_opt, null);
        mapping_opt = mapping;
        base_u64_opt = mapping.base;
    }

    const TopFuncEntry = struct {
        local_func: usize,
        samples: u64,
        percent_of_run: f64,
        code_bytes: usize,
    };
    var top_functions: std.ArrayList(TopFuncEntry) = .empty;
    var in_core: u64 = 0;

    if (perf_opt != null) {
        const base = base_u64_opt.?;
        const end = base + @as(u64, @intCast(info.text_size));
        const mb = @as(f64, @floatFromInt(info.text_size)) / 1048576.0;
        try stdout.print(
            "core text base=0x{x} size={d} ({d:.1} MB), func_count={d}, total self samples={d}\n",
            .{ base, info.text_size, mb, info.func_offsets.len, total },
        );

        var per_function: std.AutoHashMapUnmanaged(usize, u64) = .empty;
        var count_it = counts.iterator();
        while (count_it.next()) |entry| {
            const ip = entry.key_ptr.*;
            const cnt = entry.value_ptr.*;
            if (ip < base or ip >= end) continue;
            const rel: u32 = @intCast(ip - base);
            // bisect_right(info.func_offsets, rel) - 1
            var lo: usize = 0;
            var hi: usize = info.func_offsets.len;
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (info.func_offsets[mid] <= rel) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if (lo == 0) continue;
            const lf = lo - 1;
            in_core += cnt;
            const gop = try per_function.getOrPut(allocator, lf);
            if (!gop.found_existing) gop.value_ptr.* = 0;
            gop.value_ptr.* += cnt;
        }
        try stdout.print("samples in this core: {d} ({d:.1}% of run)\n\n", .{ in_core, percent(in_core, total) });
        _ = try requireAttributionCoverage(in_core, total, min_attribution_pct);

        const FuncCount = struct { local_func: usize, count: u64 };
        var fc_list: std.ArrayList(FuncCount) = .empty;
        var pf_it = per_function.iterator();
        while (pf_it.next()) |entry| {
            try fc_list.append(allocator, .{ .local_func = entry.key_ptr.*, .count = entry.value_ptr.* });
        }
        std.mem.sort(FuncCount, fc_list.items, {}, struct {
            fn lessThan(_: void, a: FuncCount, b: FuncCount) bool {
                if (a.count != b.count) return a.count > b.count;
                return a.local_func < b.local_func;
            }
        }.lessThan);

        try stdout.print("=== top {d} functions by self samples ===\n", .{top});
        const limit = @min(top, fc_list.items.len);
        for (fc_list.items[0..limit]) |fc| {
            const bounds = try functionBounds(info, @intCast(fc.local_func));
            const code_bytes = bounds.end - bounds.start;
            const pct = percent(fc.count, total);
            try top_functions.append(allocator, .{
                .local_func = fc.local_func,
                .samples = fc.count,
                .percent_of_run = pct,
                .code_bytes = code_bytes,
            });
            try stdout.print(
                "  local_func={d:<6} samples={d:<6} ({d:.2}% of run)  code_bytes={d}\n",
                .{ fc.local_func, fc.count, pct, code_bytes },
            );
        }
    }

    var json_buf: std.Io.Writer.Allocating = .init(allocator);
    defer json_buf.deinit();
    const jw = &json_buf.writer;

    const cwasm_resolved = try resolveRealOrGivenPath(allocator, io, cwasm);
    const perf_resolved = if (perf_opt) |p| try resolveRealOrGivenPath(allocator, io, p) else null;
    const is_auth = authoritative and mapping_opt != null and mapping_opt.?.authoritative;

    try jw.writeAll("{\"architecture\":");
    try writeJsonString(jw, arch);
    try jw.writeAll(",\"artifact_target\":{\"abi\":");
    try writeOptJsonString(jw, info.abi);
    try jw.writeAll(",\"architecture\":");
    try writeJsonString(jw, info.architecture);
    try jw.writeAll(",\"format\":");
    try writeJsonString(jw, info.target_format);
    try jw.print(",\"verified\":{s}}}", .{if (info.target_verified) "true" else "false"});
    try jw.print(",\"attributed_samples\":{d},\"attribution_coverage_pct\":{d},\"authoritative\":{s}", .{
        in_core,
        percent(in_core, total),
        if (is_auth) "true" else "false",
    });

    if (func_opt) |func_idx| {
        const bounds = try functionBounds(info, func_idx);
        const function_code = info.data[info.text_file_offset + bounds.start .. info.text_file_offset + bounds.end];
        const function_base: u64 = if (base_u64_opt) |b| b + @as(u64, @intCast(bounds.start)) else 0;
        const scratch_ref = if (perf_resolved) |pr| pr else (try resolveRealOrGivenPath(allocator, io, frame_metadata_opt.?));
        const scratch_dir = parentDirOrDot(scratch_ref);
        const func_label = try std.fmt.allocPrint(allocator, "func-{d}", .{func_idx});

        var metadata_opt: ?FrameMetadata = null;
        var frame_summary_opt: ?FrameSummary = null;
        var instructions: []Instruction = &.{};

        if (frame_metadata_opt) |fm_path| {
            const meta = try loadFrameMetadata(
                allocator,
                io,
                fm_path,
                func_idx,
                function_code,
                info.version,
                info.data[info.text_file_offset .. info.text_file_offset + info.text_size],
                bounds.start,
                info,
            );
            instructions = try disassembleFunction(
                allocator,
                io,
                function_code,
                function_base,
                scratch_dir,
                func_label,
                meta.inline_data_ranges,
                arch,
            );
            try validateMetadataDisassembly(allocator, &meta, instructions);
            frame_summary_opt = try buildFrameSummary(allocator, instructions, &counts, &meta);
            metadata_opt = meta;
        } else {
            instructions = try disassembleBlob(
                allocator,
                io,
                function_code,
                function_base,
                scratch_dir,
                func_label,
                0,
                arch,
            );
        }

        var function_samples: u64 = 0;
        var count_it = counts.iterator();
        while (count_it.next()) |entry| {
            const ip = entry.key_ptr.*;
            if (ip >= function_base and ip < function_base + @as(u64, @intCast(function_code.len))) {
                function_samples += entry.value_ptr.*;
            }
        }

        const instruction_classes = try classifyInstructionStream(
            allocator,
            instructions,
            if (metadata_opt) |*m| m else null,
            arch,
        );

        const ClassCounts = struct { static_count: u64 = 0, samples: u64 = 0 };
        var class_map: std.StringHashMapUnmanaged(ClassCounts) = .empty;
        const HotInst = struct { samples: u64, address: u64, text: []const u8 };
        var hot_list: std.ArrayList(HotInst) = .empty;
        var inst_addrs: std.AutoHashMapUnmanaged(u64, void) = .empty;

        for (instructions, instruction_classes) |inst, cls_name| {
            try inst_addrs.put(allocator, inst.address, {});
            const sm = counts.get(inst.address) orelse 0;
            const gop = try class_map.getOrPut(allocator, cls_name);
            if (!gop.found_existing) gop.value_ptr.* = .{};
            gop.value_ptr.static_count += 1;
            gop.value_ptr.samples += sm;
            if (sm > 0) {
                try hot_list.append(allocator, .{ .samples = sm, .address = inst.address, .text = inst.text });
            }
        }

        var fc_it = counts.iterator();
        while (fc_it.next()) |entry| {
            const addr = entry.key_ptr.*;
            const sm = entry.value_ptr.*;
            if (addr < function_base or addr >= function_base + @as(u64, @intCast(function_code.len))) continue;
            if (inst_addrs.contains(addr)) continue;
            const offset: usize = @intCast(addr - function_base);
            var in_inline = false;
            if (metadata_opt) |meta| {
                for (meta.inline_data_ranges) |dr| {
                    if (offset >= dr.native_start and offset < dr.native_end) {
                        in_inline = true;
                        break;
                    }
                }
            }
            const cls_name = if (in_inline) "inline_data_or_sample_skid" else "unknown_instruction_boundary";
            const gop = try class_map.getOrPut(allocator, cls_name);
            if (!gop.found_existing) gop.value_ptr.* = .{};
            gop.value_ptr.samples += sm;
            try hot_list.append(allocator, .{
                .samples = sm,
                .address = addr,
                .text = if (in_inline) "<inline_data_or_sample_skid>" else "<unknown_instruction_boundary>",
            });
        }

        const mode_str = if (perf_opt == null)
            try allocator.dupe(u8, "static validation")
        else
            try std.fmt.allocPrint(allocator, "self={d}, {d:.1}% of run", .{ function_samples, percent(function_samples, total) });
        try stdout.print("\n=== local_func={d} instruction-class mix ({s}) ===\n", .{ func_idx, mode_str });

        const ClassEntry = struct { name: []const u8, static_count: u64, samples: u64 };
        var class_entries: std.ArrayList(ClassEntry) = .empty;
        var cm_it = class_map.iterator();
        while (cm_it.next()) |entry| {
            try class_entries.append(allocator, .{
                .name = entry.key_ptr.*,
                .static_count = entry.value_ptr.static_count,
                .samples = entry.value_ptr.samples,
            });
        }
        std.mem.sort(ClassEntry, class_entries.items, {}, struct {
            fn lessThan(_: void, a: ClassEntry, b: ClassEntry) bool {
                if (a.samples != b.samples) return a.samples > b.samples;
                return std.mem.order(u8, a.name, b.name) == .lt;
            }
        }.lessThan);
        for (class_entries.items) |ce| {
            try stdout.print(
                "  {s:<36} {d:>7} ({d:.2}% of run)\n",
                .{ ce.name, ce.samples, percent(ce.samples, total) },
            );
        }

        if (metadata_opt == null) {
            var frame_count: usize = 0;
            for (instructions) |inst| {
                if ((try instructionFrameOperand(allocator, inst.text, arch)) != null) {
                    frame_count += 1;
                }
            }
            if (frame_count > 0) {
                try stdout.print(
                    "  note: {d} frame instructions remain unattributed; pass --frame-metadata to distinguish spills from locals/fixed state\n",
                    .{frame_count},
                );
            }
        } else {
            try printFrameSummary(allocator, stdout, &frame_summary_opt.?, total, top);
        }

        std.mem.sort(HotInst, hot_list.items, {}, struct {
            fn lessThan(_: void, a: HotInst, b: HotInst) bool {
                if (a.samples != b.samples) return a.samples > b.samples;
                if (a.address != b.address) return a.address > b.address;
                return std.mem.order(u8, a.text, b.text) == .gt;
            }
        }.lessThan);
        const hot_limit = @min(@as(usize, 20), hot_list.items.len);
        if (perf_opt != null) {
            try stdout.print("\n=== top 20 hottest instructions in local_func={d} ===\n", .{func_idx});
            for (hot_list.items[0..hot_limit]) |h| {
                try stdout.print(
                    "  {d:>5} ({d:.2}%)  {x}: {s}\n",
                    .{ h.samples, percent(h.samples, total), h.address, h.text },
                );
            }
        }

        std.mem.sort(ClassEntry, class_entries.items, {}, struct {
            fn lessThan(_: void, a: ClassEntry, b: ClassEntry) bool {
                return std.mem.order(u8, a.name, b.name) == .lt;
            }
        }.lessThan);

        try jw.writeAll(",\"classified_function\":{\"classes\":{");
        for (class_entries.items, 0..) |ce, cei| {
            if (cei > 0) try jw.writeAll(",");
            try writeJsonString(jw, ce.name);
            try jw.print(
                ":{{\"percent_of_run\":{d},\"samples\":{d},\"static_instructions\":{d}}}",
                .{ percent(ce.samples, total), ce.samples, ce.static_count },
            );
        }
        try jw.writeAll("}");
        if (metadata_opt) |meta| {
            const fm_resolved = try resolveRealOrGivenPath(allocator, io, frame_metadata_opt.?);
            try jw.writeAll(",\"frame_attribution\":");
            try writeFrameSummaryJson(jw, &frame_summary_opt.?);
            try jw.writeAll(",\"frame_metadata\":");
            try writeJsonString(jw, fm_resolved);
            try jw.writeAll(",\"frame_metadata_identity\":{\"compiler_build_id\":");
            try writeJsonString(jw, meta.compiler_build_id);
            try jw.writeAll(",\"module_text_sha256\":");
            try writeJsonString(jw, meta.module_text_sha256);
            try jw.writeAll(",\"normalized_code_sha256\":");
            try writeJsonString(jw, meta.normalized_code_sha256);
            try jw.writeAll(",\"schema\":");
            try writeJsonString(jw, meta.schema);
            try jw.print(",\"schema_version\":{d}}},\"frame_metadata_module\":{d}", .{ meta.schema_version, meta.module });
        }
        try jw.writeAll(",\"hottest_instructions\":[");
        for (hot_list.items[0..hot_limit], 0..) |h, hi| {
            if (hi > 0) try jw.writeAll(",");
            try jw.print("{{\"address\":{d},\"instruction\":", .{h.address});
            try writeJsonString(jw, h.text);
            try jw.print(",\"percent_of_run\":{d},\"samples\":{d}}}", .{ percent(h.samples, total), h.samples });
        }
        try jw.print(
            "],\"instruction_count\":{d},\"local_func\":{d},\"percent_of_run\":{d},\"samples\":{d}}}",
            .{ instructions.len, func_idx, percent(function_samples, total), function_samples },
        );
    }

    try jw.writeAll(",\"cwasm\":");
    try writeJsonString(jw, cwasm_resolved);
    try jw.print(",\"function_count\":{d},\"mapping\":", .{info.func_offsets.len});
    if (mapping_opt) |m| {
        try jw.print(
            "{{\"authoritative\":{s},\"base\":{d},\"candidates\":[",
            .{ if (m.authoritative) "true" else "false", m.base },
        );
        for (m.candidates, 0..) |cand, ci| {
            if (ci > 0) try jw.writeAll(",");
            try jw.print("{{\"base\":{d},\"size\":{d}}}", .{ cand.base, cand.size });
        }
        try jw.print(
            "],\"expected_size\":{d},\"override\":",
            .{m.expected_size},
        );
        try writeOptJsonString(jw, m.override);
        try jw.print(",\"page_size\":{d},\"size\":{d},\"text_size\":{d}}}", .{ m.page_size, m.size, m.text_size });
    } else {
        try jw.writeAll("null");
    }
    try jw.print(",\"minimum_attribution_coverage_pct\":{d},\"perf\":", .{min_attribution_pct});
    try writeOptJsonString(jw, perf_resolved);
    try jw.writeAll(",\"schema_version\":2,\"text_base\":");
    if (base_u64_opt) |b| {
        try jw.print("{d}", .{b});
    } else {
        try jw.writeAll("null");
    }
    try jw.print(",\"text_size\":{d},\"top_functions\":[", .{info.text_size});
    for (top_functions.items, 0..) |tf, tfi| {
        if (tfi > 0) try jw.writeAll(",");
        try jw.print(
            "{{\"code_bytes\":{d},\"local_func\":{d},\"percent_of_run\":{d},\"samples\":{d}}}",
            .{ tf.code_bytes, tf.local_func, tf.percent_of_run, tf.samples },
        );
    }
    try jw.print("],\"total_samples\":{d}}}\n", .{total});

    if (json_out_opt) |json_out| {
        const json_bytes = try json_buf.toOwnedSlice();
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = json_out, .data = json_bytes }) catch |err| {
            return fail("failed to write {s}: {s}", .{ json_out, @errorName(err) });
        };
    }
}

fn buildSmokeEnv(allocator: Allocator, extra: []const [2][]const u8) !std.process.Environ.Map {
    var env_map = std.process.Environ.Map.init(allocator);
    if (builtin.os.tag != .windows) {
        const keys_to_copy = [_][]const u8{
            "PATH",
            "HOME",
            "TMPDIR",
            "ZIG_GLOBAL_CACHE_DIR",
            "ZIG_LOCAL_CACHE_DIR",
            "LD_LIBRARY_PATH",
        };
        for (keys_to_copy) |k| {
            var zbuf: [128]u8 = undefined;
            if (k.len + 1 <= zbuf.len) {
                @memcpy(zbuf[0..k.len], k);
                zbuf[k.len] = 0;
                if (std.c.getenv(zbuf[0..k.len :0])) |val| {
                    try env_map.put(k, std.mem.span(val));
                }
            }
        }
    }
    for (extra) |pair| {
        try env_map.put(pair[0], pair[1]);
    }
    return env_map;
}

fn parseSpillLdSt(stderr: []const u8) ?struct { ld: i64, st: i64 } {
    var lines = std.mem.splitScalar(u8, stderr, '\n');
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, "[aot-spill-metric]") == null) continue;
        const ld_idx = std.mem.indexOf(u8, line, "spill_ld=") orelse continue;
        const after_ld = line[ld_idx + "spill_ld=".len ..];
        var ld_end: usize = 0;
        while (ld_end < after_ld.len and std.ascii.isDigit(after_ld[ld_end])) : (ld_end += 1) {}
        if (ld_end == 0) continue;
        const ld = std.fmt.parseInt(i64, after_ld[0..ld_end], 10) catch continue;

        const st_idx = std.mem.indexOf(u8, line, "spill_st=") orelse continue;
        const after_st = line[st_idx + "spill_st=".len ..];
        var st_end: usize = 0;
        while (st_end < after_st.len and std.ascii.isDigit(after_st[st_end])) : (st_end += 1) {}
        if (st_end == 0) continue;
        const st = std.fmt.parseInt(i64, after_st[0..st_end], 10) catch continue;
        return .{ .ld = ld, .st = st };
    }
    return null;
}

pub fn runCompilerSmokeTests(allocator: Allocator, io: std.Io, wamrc_raw: []const u8) AttributionError!void {
    const wamrc = try resolveRealOrGivenPath(allocator, io, wamrc_raw);

    // 1. test_metric_only_scalar_fallback_preserves_native_bytes
    {
        const work = ".zig-cache/frame-metric-fallback-smoke";
        std.Io.Dir.cwd().deleteTree(io, work) catch {};
        std.Io.Dir.cwd().createDirPath(io, work) catch |err| {
            return fail("mkdir {s} failed: {s}", .{ work, @errorName(err) });
        };
        defer std.Io.Dir.cwd().deleteTree(io, work) catch {};

        const baseline = try std.fmt.allocPrint(allocator, "{s}/baseline.cwasm", .{work});
        const diagnosed = try std.fmt.allocPrint(allocator, "{s}/diagnosed.cwasm", .{work});
        var clean_env = try buildSmokeEnv(allocator, &.{});
        defer clean_env.deinit();

        const base_argv = [_][]const u8{
            wamrc,
            "compile",
            "--target=aarch64",
            "--aarch64-no-xreg-alloc",
            "tests/coldstart/noop.wasm",
            "-o",
            baseline,
        };
        const base_res = std.process.run(allocator, io, .{
            .argv = &base_argv,
            .environ_map = &clean_env,
        }) catch |err| return fail("wamrc baseline failed: {s}", .{@errorName(err)});
        switch (base_res.term) {
            .exited => |c| if (c != 0) return fail("wamrc baseline exited {d}: {s}", .{ c, base_res.stderr }),
            else => return fail("wamrc baseline abnormal exit", .{}),
        }

        var diag_env = try buildSmokeEnv(allocator, &.{
            .{ "WAMR_AOT_SPILL_METRIC", "1" },
            .{ "WAMR_AOT_SPILL_METRIC_MIN_SPILLS", "0" },
        });
        defer diag_env.deinit();
        const diag_argv = [_][]const u8{
            wamrc,
            "compile",
            "--target=aarch64",
            "--aarch64-no-xreg-alloc",
            "tests/coldstart/noop.wasm",
            "-o",
            diagnosed,
        };
        const diag_res = std.process.run(allocator, io, .{
            .argv = &diag_argv,
            .environ_map = &diag_env,
        }) catch |err| return fail("wamrc diagnosed failed: {s}", .{@errorName(err)});
        switch (diag_res.term) {
            .exited => |c| if (c != 0) return fail("wamrc diagnosed exited {d}: {s}", .{ c, diag_res.stderr }),
            else => return fail("wamrc diagnosed abnormal exit", .{}),
        }
        if (std.mem.indexOf(u8, diag_res.stderr, "[aot-spill-metric]") == null) {
            return fail("expected [aot-spill-metric] in stderr", .{});
        }
        const base_bytes = std.Io.Dir.cwd().readFileAlloc(io, baseline, allocator, .limited(16 * 1024 * 1024)) catch return fail("read baseline failed", .{});
        const diag_bytes = std.Io.Dir.cwd().readFileAlloc(io, diagnosed, allocator, .limited(16 * 1024 * 1024)) catch return fail("read diagnosed failed", .{});
        if (!std.mem.eql(u8, base_bytes, diag_bytes)) {
            return fail("metric-only scalar fallback changed native bytes", .{});
        }
    }

    // 2. test_tracked_x86_64_core_wasm_sidecar_reconciles & test_tracked_aarch64_core_wasm_sidecar_reconciles
    for ([_][]const u8{ "x86_64", "aarch64" }) |arch| {
        const work = try std.fmt.allocPrint(allocator, ".zig-cache/frame-attribution-smoke-{s}", .{arch});
        std.Io.Dir.cwd().deleteTree(io, work) catch {};
        std.Io.Dir.cwd().createDirPath(io, work) catch |err| {
            return fail("mkdir {s} failed: {s}", .{ work, @errorName(err) });
        };
        defer std.Io.Dir.cwd().deleteTree(io, work) catch {};

        const fixture = "tests/benchmarks/frame_attribution/frame_origins.wasm";
        const cwasm = try std.fmt.allocPrint(allocator, "{s}/frame_origins.cwasm", .{work});
        const baseline_cwasm = try std.fmt.allocPrint(allocator, "{s}/frame_origins.baseline.cwasm", .{work});
        const prefix = try std.fmt.allocPrint(allocator, "{s}/frame", .{work});
        const cache = try std.fmt.allocPrint(allocator, "{s}/codegen.cache", .{work});

        var clean_env = try buildSmokeEnv(allocator, &.{});
        defer clean_env.deinit();
        const base_argv = [_][]const u8{
            wamrc, "compile", "--target", arch, "--cache", cache, fixture, "-o", baseline_cwasm,
        };
        const base_res = std.process.run(allocator, io, .{
            .argv = &base_argv,
            .environ_map = &clean_env,
        }) catch |err| return fail("wamrc baseline {s} failed: {s}", .{ arch, @errorName(err) });
        switch (base_res.term) {
            .exited => |c| if (c != 0) return fail("wamrc baseline {s} exited {d}: {s}", .{ arch, c, base_res.stderr }),
            else => return fail("wamrc baseline {s} abnormal exit", .{arch}),
        }

        var diag_env = try buildSmokeEnv(allocator, &.{
            .{ "WAMR_AOT_FRAME_ATTRIBUTION", prefix },
            .{ "WAMR_AOT_FRAME_ATTRIBUTION_FUNC", "0" },
            .{ "WAMR_AOT_SPILL_METRIC", "1" },
            .{ "WAMR_AOT_SPILL_METRIC_FUNC", "0" },
        });
        defer diag_env.deinit();
        const diag_argv = [_][]const u8{
            wamrc, "compile", "--target", arch, "--cache", cache, fixture, "-o", cwasm,
        };
        const diag_res = std.process.run(allocator, io, .{
            .argv = &diag_argv,
            .environ_map = &diag_env,
        }) catch |err| return fail("wamrc sidecar {s} failed: {s}", .{ arch, @errorName(err) });
        switch (diag_res.term) {
            .exited => |c| if (c != 0) return fail("wamrc sidecar {s} exited {d}: {s}", .{ arch, c, diag_res.stderr }),
            else => return fail("wamrc sidecar {s} abnormal exit", .{arch}),
        }

        const base_bytes = std.Io.Dir.cwd().readFileAlloc(io, baseline_cwasm, allocator, .limited(16 * 1024 * 1024)) catch return fail("read baseline failed", .{});
        const diag_bytes = std.Io.Dir.cwd().readFileAlloc(io, cwasm, allocator, .limited(16 * 1024 * 1024)) catch return fail("read cwasm failed", .{});
        if (!std.mem.eql(u8, base_bytes, diag_bytes)) {
            return fail("frame diagnostics must not change production AOT bytes ({s})", .{arch});
        }

        const sidecar = try std.fmt.allocPrint(allocator, "{s}.mod0.func0.json", .{prefix});
        const info = try parseCwasm(allocator, io, cwasm);
        const bounds = try functionBounds(info, 0);
        const code = info.data[info.text_file_offset + bounds.start .. info.text_file_offset + bounds.end];
        const metadata = try loadFrameMetadata(
            allocator,
            io,
            sidecar,
            0,
            code,
            info.version,
            info.data[info.text_file_offset .. info.text_file_offset + info.text_size],
            bounds.start,
            info,
        );
        if (!std.mem.eql(u8, metadata.architecture, arch)) {
            return fail("metadata architecture mismatch", .{});
        }
        const expected_sv: i64 = if (std.mem.eql(u8, arch, "x86_64")) 1 else 2;
        if (metadata.schema_version != expected_sv) {
            return fail("unexpected schema_version {d} for {s}", .{ metadata.schema_version, arch });
        }
        const spill_counts = parseSpillLdSt(diag_res.stderr) orelse return fail("missing [aot-spill-metric] line in stderr", .{});
        if (spill_counts.ld != metadata.reconciliation.emitted_allocator_loads or
            spill_counts.st != metadata.reconciliation.emitted_allocator_stores)
        {
            return fail("spill_ld/spill_st mismatch with reconciliation", .{});
        }

        const wamrc_dir = parentDirOrDot(wamrc);
        const wamr_bin = try std.fmt.allocPrint(allocator, "{s}/wamr", .{wamrc_dir});
        if (std.mem.eql(u8, arch, hostArchitecture())) {
            if (std.Io.Dir.cwd().statFile(io, wamr_bin, .{})) |_| {
                const run_argv = [_][]const u8{ wamr_bin, "run", cwasm };
                const run_res = std.process.run(allocator, io, .{ .argv = &run_argv }) catch |err| {
                    return fail("wamr run failed: {s}", .{@errorName(err)});
                };
                switch (run_res.term) {
                    .exited => |c| if (c != 0) return fail("wamr run exited {d}: {s}", .{ c, run_res.stderr }),
                    else => return fail("wamr run abnormal exit", .{}),
                }
            } else |_| {}
        }

        const instructions_or_err = disassembleBlob(allocator, io, code, 0, work, "smoke", 0, arch);
        if (instructions_or_err) |instructions| {
            try validateMetadataDisassembly(allocator, &metadata, instructions);
            var empty_counts: std.AutoHashMapUnmanaged(u64, u64) = .empty;
            const summary = try buildFrameSummary(allocator, instructions, &empty_counts, &metadata);
            if (!metadata.reconciliation.matches or metadata.reconciliation.emitted_allocator_loads <= 0) {
                return fail("expected positive reconciled allocator loads", .{});
            }
            if (summary.originByName("allocator_spill") == null or summary.originByName("fixed_runtime_frame_state") == null) {
                return fail("missing expected frame origins in summary", .{});
            }
            if (summary.coverage.unknown_frame_instructions != 0) {
                return fail("expected 0 unknown frame instructions, got {d}", .{summary.coverage.unknown_frame_instructions});
            }
        } else |err| {
            if (std.mem.eql(u8, arch, "x86_64") and std.mem.indexOf(u8, lastError(), "can't use supplied machine i386:x86-64") != null) {
                continue;
            }
            return err;
        }
    }

    // 3. test_aarch64_direct_tail_call_metadata_and_bytes
    {
        const work = ".zig-cache/frame-attribution-tail-call-smoke";
        std.Io.Dir.cwd().deleteTree(io, work) catch {};
        std.Io.Dir.cwd().createDirPath(io, work) catch |err| {
            return fail("mkdir {s} failed: {s}", .{ work, @errorName(err) });
        };
        defer std.Io.Dir.cwd().deleteTree(io, work) catch {};

        const wasm = try std.fmt.allocPrint(allocator, "{s}/tail-call.wasm", .{work});
        const wasm_bytes = [_]u8{
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f, 0x03,
            0x03, 0x02, 0x00, 0x00, 0x0a, 0x0b, 0x02, 0x04,
            0x00, 0x41, 0x07, 0x0b, 0x04, 0x00, 0x12, 0x00,
            0x0b,
        };
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = wasm, .data = &wasm_bytes }) catch return fail("write tail-call.wasm failed", .{});

        const baseline = try std.fmt.allocPrint(allocator, "{s}/baseline.cwasm", .{work});
        const diagnosed = try std.fmt.allocPrint(allocator, "{s}/diagnosed.cwasm", .{work});
        const prefix = try std.fmt.allocPrint(allocator, "{s}/tail", .{work});

        var clean_env = try buildSmokeEnv(allocator, &.{});
        defer clean_env.deinit();
        const base_argv = [_][]const u8{ wamrc, "compile", "--target", "aarch64", wasm, "-o", baseline };
        const base_res = std.process.run(allocator, io, .{
            .argv = &base_argv,
            .environ_map = &clean_env,
        }) catch |err| return fail("wamrc tail baseline failed: {s}", .{@errorName(err)});
        switch (base_res.term) {
            .exited => |c| if (c != 0) return fail("wamrc tail baseline exited {d}: {s}", .{ c, base_res.stderr }),
            else => return fail("wamrc tail baseline abnormal exit", .{}),
        }

        var diag_env = try buildSmokeEnv(allocator, &.{
            .{ "WAMR_AOT_FRAME_ATTRIBUTION", prefix },
            .{ "WAMR_AOT_FRAME_ATTRIBUTION_FUNC", "1" },
        });
        defer diag_env.deinit();
        const diag_argv = [_][]const u8{ wamrc, "compile", "--target", "aarch64", wasm, "-o", diagnosed };
        const diag_res = std.process.run(allocator, io, .{
            .argv = &diag_argv,
            .environ_map = &diag_env,
        }) catch |err| return fail("wamrc tail diagnosed failed: {s}", .{@errorName(err)});
        switch (diag_res.term) {
            .exited => |c| if (c != 0) return fail("wamrc tail diagnosed exited {d}: {s}", .{ c, diag_res.stderr }),
            else => return fail("wamrc tail diagnosed abnormal exit", .{}),
        }

        const base_bytes = std.Io.Dir.cwd().readFileAlloc(io, baseline, allocator, .limited(16 * 1024 * 1024)) catch return fail("read tail baseline failed", .{});
        const diag_bytes = std.Io.Dir.cwd().readFileAlloc(io, diagnosed, allocator, .limited(16 * 1024 * 1024)) catch return fail("read tail diagnosed failed", .{});
        if (!std.mem.eql(u8, base_bytes, diag_bytes)) {
            return fail("tail-call diagnostics changed production AOT bytes", .{});
        }

        const sidecar = try std.fmt.allocPrint(allocator, "{s}.mod0.func1.json", .{prefix});
        const sidecar_text = std.Io.Dir.cwd().readFileAlloc(io, sidecar, allocator, .limited(16 * 1024 * 1024)) catch return fail("read tail sidecar failed", .{});
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, sidecar_text, .{}) catch return fail("parse tail sidecar failed", .{});
        const rel_arr = try requireArrayField(parsed.value.object, "normalized_relocations");
        var tail_count: usize = 0;
        var tail_start: usize = 0;
        for (rel_arr) |item| {
            const robj = item.object;
            if (robj.get("kind")) |kv| {
                if (std.mem.eql(u8, kv.string, "aarch64_tail_call_imm26")) {
                    tail_count += 1;
                    tail_start = @intCast(robj.get("native_start").?.integer);
                }
            }
        }
        if (tail_count != 1) {
            return fail("expected 1 aarch64_tail_call_imm26 relocation, got {d}", .{tail_count});
        }
        const info = try parseCwasm(allocator, io, diagnosed);
        const bounds = try functionBounds(info, 1);
        const code = info.data[info.text_file_offset + bounds.start .. info.text_file_offset + bounds.end];
        const word = std.mem.readInt(u32, code[tail_start..][0..4], .little);
        if ((word & 0xFC000000) != 0x14000000) {
            return fail("expected tail call B instruction opcode 0x14000000, got 0x{x}", .{word});
        }

        var sink: std.Io.Writer.Allocating = .init(allocator);
        defer sink.deinit();
        const cli_args = [_][]const u8{
            "--cwasm",
            diagnosed,
            "--func",
            "1",
            "--arch",
            "aarch64",
            "--frame-metadata",
            sidecar,
            "--validate-frame-metadata",
        };
        try runCli(allocator, io, &cli_args, &sink.writer);
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    const raw_args = try init.minimal.args.toSlice(allocator);
    const args = if (raw_args.len > 0) raw_args[1..] else raw_args[0..0];

    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);

    runCli(allocator, io, args, &stdout_writer.interface) catch |err| {
        stdout_writer.interface.flush() catch {};
        var stderr_buf: [1024]u8 = undefined;
        var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buf);
        const msg = if (err == error.AttributionFailed) lastError() else @errorName(err);
        stderr_writer.interface.print("{s}\n", .{msg}) catch {};
        stderr_writer.interface.flush() catch {};
        std.process.exit(1);
    };
    try stdout_writer.interface.flush();
}

pub const TEST_LOAD_X86 = [_]u8{ 0x48, 0x8b, 0x85, 0xd0, 0xfd, 0xff, 0xff }; // mov rax,QWORD PTR [rbp-0x230]
pub const TEST_STORE_X86 = [_]u8{ 0x48, 0x89, 0x85, 0xd0, 0xfd, 0xff, 0xff }; // mov QWORD PTR [rbp-0x230],rax
pub const TEST_LOAD_STORE_X86 = TEST_LOAD_X86 ++ TEST_STORE_X86;

pub const TEST_ARM_BL = [_]u8{ 0x01, 0x00, 0x00, 0x94 };
pub const TEST_ARM_B = [_]u8{ 0x01, 0x00, 0x00, 0x14 };
pub const TEST_ARM_LDR = [_]u8{ 0xb0, 0x7f, 0x40, 0xf9 }; // ldr x16, [x29, #248]
pub const TEST_ARM_STR = [_]u8{ 0xb0, 0x7f, 0x00, 0xf9 }; // str x16, [x29, #248]
pub const TEST_ARM_CODE = TEST_ARM_BL ++ TEST_ARM_LDR ++ TEST_ARM_STR;

pub fn makeTestTargetInfo(
    architecture: []const u8,
    bin_type: u16,
    e_type: u16,
    e_machine: u16,
    abi_type: u16,
) [40]u8 {
    var buf: [40]u8 = @splat(0);
    std.mem.writeInt(u16, buf[0..2], bin_type, .little);
    std.mem.writeInt(u16, buf[2..4], abi_type, .little);
    std.mem.writeInt(u16, buf[4..6], e_type, .little);
    std.mem.writeInt(u16, buf[6..8], e_machine, .little);
    const copy_len = @min(architecture.len, 16);
    @memcpy(buf[16 .. 16 + copy_len], architecture[0..copy_len]);
    return buf;
}

pub fn appendTestSection(allocator: Allocator, list: *std.ArrayList(u8), section_type: u32, payload: []const u8) !void {
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], section_type, .little);
    std.mem.writeInt(u32, hdr[4..8], @intCast(payload.len), .little);
    try list.appendSlice(allocator, &hdr);
    try list.appendSlice(allocator, payload);
}

pub fn makeTestArtifactInfo(code: []const u8, architecture: []const u8) CwasmInfo {
    const offsets = &[_]u32{0};
    return .{
        .func_offsets = offsets,
        .text_size = code.len,
        .text_file_offset = 0,
        .data = code,
        .version = AOT_VERSION,
        .architecture = architecture,
        .abi = if (std.mem.eql(u8, architecture, "aarch64")) "aapcs64" else "sysv",
        .target_format = "elf64-little",
        .target_verified = true,
    };
}

pub fn makeDefaultX86MetadataJson(allocator: Allocator, code: []const u8) ![]u8 {
    const mod_hash = try sha256Hex(allocator, code);
    const norm_hash = try normalizedCodeSha256(allocator, code, &.{});
    return std.fmt.allocPrint(
        allocator,
        \\{{
        \\  "schema": "wamr-aot-frame-attribution",
        \\  "schema_version": 1,
        \\  "cwasm_aot_version": {d},
        \\  "compiler_build_id": "test",
        \\  "architecture": "x86_64",
        \\  "abi": "sysv",
        \\  "module": 4,
        \\  "local_func": 7,
        \\  "function_name": "fixture",
        \\  "module_text_size": {d},
        \\  "module_text_sha256": "{s}",
        \\  "function_offset": 0,
        \\  "code_size": {d},
        \\  "normalized_code_sha256": "{s}",
        \\  "direct_call_rel32_offsets": [],
        \\  "inline_data_ranges": [],
        \\  "frame_layout": {{
        \\    "frame_pointer": "rbp",
        \\    "frame_size": 576,
        \\    "local_count": 2,
        \\    "param_count": 0,
        \\    "reserved_vmctx_offset": -8,
        \\    "locals_first_offset": -16,
        \\    "explicit_storage_first_offset": -32,
        \\    "explicit_storage_slots": 64,
        \\    "spill_base": -560,
        \\    "spill_stride": -8,
        \\    "spill_slots": 1
        \\  }},
        \\  "spill_metric": {{
        \\    "slots": 1,
        \\    "spilled_vregs": 2,
        \\    "scalar": 2,
        \\    "v128": 0,
        \\    "slots_scalar": 2,
        \\    "slots_v128": 0,
        \\    "spill_ld": 1,
        \\    "spill_st": 1,
        \\    "remat": 0,
        \\    "callee_saved": 0
        \\  }},
        \\  "emitted_allocator_loads": 1,
        \\  "emitted_allocator_stores": 1,
        \\  "allocator_values": [
        \\    {{
        \\      "vreg": 10,
        \\      "frame_offset": -560,
        \\      "slot": 0,
        \\      "slot_count": 1,
        \\      "value_type": "i64",
        \\      "live_start": 0,
        \\      "live_end": 3,
        \\      "defining_opcode": "load",
        \\      "source_class": "memory_or_runtime",
        \\      "ir_use_count": 1,
        \\      "ir_def_count": 1,
        \\      "reload_count": 1,
        \\      "store_count": 0,
        \\      "rematerialization_eligible": false,
        \\      "reused": true
        \\    }},
        \\    {{
        \\      "vreg": 11,
        \\      "frame_offset": -560,
        \\      "slot": 0,
        \\      "slot_count": 1,
        \\      "value_type": "i64",
        \\      "live_start": 4,
        \\      "live_end": 8,
        \\      "defining_opcode": "add",
        \\      "source_class": "computed",
        \\      "ir_use_count": 0,
        \\      "ir_def_count": 1,
        \\      "reload_count": 0,
        \\      "store_count": 0,
        \\      "rematerialization_eligible": false,
        \\      "reused": true
        \\    }}
        \\  ],
        \\  "accesses": [
        \\    {{
        \\      "native_start": 0,
        \\      "native_end": 7,
        \\      "kind": "load",
        \\      "base": "rbp",
        \\      "frame_offset": -560,
        \\      "width": 8,
        \\      "origin": "allocator_spill",
        \\      "detail": "allocator_slot",
        \\      "slot": 0,
        \\      "local_index": null,
        \\      "explicit_slot": null,
        \\      "vreg": 10,
        \\      "vreg_ambiguous": false,
        \\      "defining_opcode": "load",
        \\      "source_class": "memory_or_runtime",
        \\      "rematerialization_eligible": false,
        \\      "ir_position": 2,
        \\      "ir_opcode": "add"
        \\    }},
        \\    {{
        \\      "native_start": 7,
        \\      "native_end": {d},
        \\      "kind": "store",
        \\      "base": "rbp",
        \\      "frame_offset": -560,
        \\      "width": 8,
        \\      "origin": "allocator_spill",
        \\      "detail": "allocator_slot",
        \\      "slot": 0,
        \\      "local_index": null,
        \\      "explicit_slot": null,
        \\      "vreg": null,
        \\      "vreg_ambiguous": true,
        \\      "defining_opcode": null,
        \\      "source_class": null,
        \\      "rematerialization_eligible": null,
        \\      "ir_position": null,
        \\      "ir_opcode": null
        \\    }}
        \\  ]
        \\}}
    ,
        .{ AOT_VERSION, code.len, mod_hash, code.len, norm_hash, code.len },
    );
}

pub fn makeDefaultAarch64MetadataJson(allocator: Allocator, code: []const u8) ![]u8 {
    const mod_hash = try sha256Hex(allocator, code);
    const relocs = [_]NormalizedRelocation{.{ .native_start = 0, .native_end = 4, .kind = "aarch64_call_imm26" }};
    const norm_hash = try normalizedCodeSha256V2(allocator, code, &relocs, "aarch64");
    return std.fmt.allocPrint(
        allocator,
        \\{{
        \\  "schema": "wamr-aot-frame-attribution",
        \\  "schema_version": 2,
        \\  "cwasm_aot_version": {d},
        \\  "compiler_build_id": "test",
        \\  "architecture": "aarch64",
        \\  "abi": "aapcs64",
        \\  "module": 0,
        \\  "local_func": 10,
        \\  "function_name": "core_state_transition",
        \\  "module_text_size": {d},
        \\  "module_text_sha256": "{s}",
        \\  "function_offset": 0,
        \\  "code_size": {d},
        \\  "normalized_code_sha256": "{s}",
        \\  "direct_call_rel32_offsets": [],
        \\  "normalized_relocations": [
        \\    {{ "native_start": 0, "native_end": 4, "kind": "aarch64_call_imm26" }}
        \\  ],
        \\  "inline_data_ranges": [],
        \\  "frame_layout": {{
        \\    "frame_pointer": "x29",
        \\    "frame_size": 512,
        \\    "local_count": 1,
        \\    "param_count": 0,
        \\    "reserved_vmctx_offset": 16,
        \\    "locals_first_offset": 24,
        \\    "explicit_storage_first_offset": 264,
        \\    "explicit_storage_slots": 1,
        \\    "spill_base": 248,
        \\    "spill_stride": 8,
        \\    "spill_slots": 1
        \\  }},
        \\  "frame_regions": [
        \\    {{ "start": 0, "end": 16, "origin": "fixed_runtime_frame_state", "detail": "saved_fp_lr" }},
        \\    {{ "start": 16, "end": 24, "origin": "fixed_runtime_frame_state", "detail": "reserved_vmctx" }},
        \\    {{ "start": 24, "end": 32, "origin": "wasm_local_or_phi", "detail": "wasm_local_or_lowered_phi" }},
        \\    {{ "start": 248, "end": 256, "origin": "allocator_spill", "detail": "scalar_allocator_spills" }}
        \\  ],
        \\  "spill_metric": {{
        \\    "slots": 1,
        \\    "spilled_vregs": 1,
        \\    "scalar": 1,
        \\    "v128": 0,
        \\    "slots_scalar": 1,
        \\    "slots_v128": 0,
        \\    "spill_ld": 1,
        \\    "spill_st": 1,
        \\    "remat": 0,
        \\    "callee_saved": 0
        \\  }},
        \\  "emitted_allocator_loads": 1,
        \\  "emitted_allocator_stores": 1,
        \\  "allocator_values": [
        \\    {{
        \\      "vreg": 77,
        \\      "frame_offset": 248,
        \\      "slot": 0,
        \\      "slot_count": 1,
        \\      "value_type": "i64",
        \\      "live_start": 4,
        \\      "live_end": 12,
        \\      "defining_opcode": "load",
        \\      "source_class": "memory_or_runtime",
        \\      "ir_use_count": 1,
        \\      "ir_def_count": 1,
        \\      "reload_count": 1,
        \\      "store_count": 1,
        \\      "rematerialization_eligible": false,
        \\      "reused": false
        \\    }}
        \\  ],
        \\  "accesses": [
        \\    {{
        \\      "native_start": 4,
        \\      "native_end": 8,
        \\      "kind": "load",
        \\      "base": "x29",
        \\      "frame_offset": 248,
        \\      "width": 8,
        \\      "origin": "allocator_spill",
        \\      "detail": "allocator_slot",
        \\      "slot": 0,
        \\      "local_index": null,
        \\      "explicit_slot": null,
        \\      "vreg": 77,
        \\      "vreg_ambiguous": false,
        \\      "defining_opcode": "load",
        \\      "source_class": "memory_or_runtime",
        \\      "rematerialization_eligible": false,
        \\      "ir_position": 8,
        \\      "ir_opcode": "add",
        \\      "encoded_base": "x29",
        \\      "encoded_offset": 248,
        \\      "addressing_mode": "unsigned_scaled",
        \\      "components": [
        \\        {{
        \\          "base": "x29",
        \\          "frame_offset": 248,
        \\          "width": 8,
        \\          "data_register": 16,
        \\          "data_register_class": "gpr",
        \\          "origin": "allocator_spill",
        \\          "detail": "allocator_slot",
        \\          "slot": 0,
        \\          "local_index": null,
        \\          "explicit_slot": null,
        \\          "vreg": 77,
        \\          "vreg_ambiguous": false,
        \\          "defining_opcode": "load",
        \\          "source_class": "memory_or_runtime",
        \\          "rematerialization_eligible": false,
        \\          "ir_position": 8,
        \\          "ir_opcode": "add"
        \\        }}
        \\      ]
        \\    }},
        \\    {{
        \\      "native_start": 8,
        \\      "native_end": 12,
        \\      "kind": "store",
        \\      "base": "x29",
        \\      "frame_offset": 248,
        \\      "width": 8,
        \\      "origin": "allocator_spill",
        \\      "detail": "allocator_slot",
        \\      "slot": 0,
        \\      "local_index": null,
        \\      "explicit_slot": null,
        \\      "vreg": 77,
        \\      "vreg_ambiguous": false,
        \\      "defining_opcode": "load",
        \\      "source_class": "memory_or_runtime",
        \\      "rematerialization_eligible": false,
        \\      "ir_position": 9,
        \\      "ir_opcode": "local_set",
        \\      "encoded_base": "x29",
        \\      "encoded_offset": 248,
        \\      "addressing_mode": "unsigned_scaled",
        \\      "components": [
        \\        {{
        \\          "base": "x29",
        \\          "frame_offset": 248,
        \\          "width": 8,
        \\          "data_register": 16,
        \\          "data_register_class": "gpr",
        \\          "origin": "allocator_spill",
        \\          "detail": "allocator_slot",
        \\          "slot": 0,
        \\          "local_index": null,
        \\          "explicit_slot": null,
        \\          "vreg": 77,
        \\          "vreg_ambiguous": false,
        \\          "defining_opcode": "load",
        \\          "source_class": "memory_or_runtime",
        \\          "rematerialization_eligible": false,
        \\          "ir_position": 9,
        \\          "ir_opcode": "local_set"
        \\        }}
        \\      ]
        \\    }}
        \\  ]
        \\}}
    ,
        .{ AOT_VERSION, code.len, mod_hash, code.len, norm_hash },
    );
}

fn replaceFirst(allocator: Allocator, text: []const u8, needle: []const u8, replacement: []const u8) ![]u8 {
    const idx = std.mem.indexOf(u8, text, needle) orelse return error.OutOfMemory;
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(allocator, text[0..idx]);
    try out.appendSlice(allocator, replacement);
    try out.appendSlice(allocator, text[idx + needle.len ..]);
    return out.toOwnedSlice(allocator);
}

fn replaceAllAlloc(allocator: Allocator, text: []const u8, needle: []const u8, replacement: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, needle)) |idx| {
        try out.appendSlice(allocator, text[pos..idx]);
        try out.appendSlice(allocator, replacement);
        pos = idx + needle.len;
    }
    try out.appendSlice(allocator, text[pos..]);
    return out.toOwnedSlice(allocator);
}

test "FrameAttributionUnitTests.test_allocator_sources_must_match_canonical_values" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const base_x86 = try makeDefaultX86MetadataJson(alloc, &TEST_LOAD_STORE_X86);
    const bad_x86_cases = [_][2][]const u8{
        .{ "\"defining_opcode\": \"load\",\n      \"source_class\": \"memory_or_runtime\",\n      \"rematerialization_eligible\": false,\n      \"ir_position\": 2", "\"defining_opcode\": \"iconst_32\",\n      \"source_class\": \"memory_or_runtime\",\n      \"rematerialization_eligible\": false,\n      \"ir_position\": 2" },
        .{ "\"defining_opcode\": \"load\",\n      \"source_class\": \"memory_or_runtime\",\n      \"rematerialization_eligible\": false,\n      \"ir_position\": 2", "\"defining_opcode\": \"load\",\n      \"source_class\": \"constant\",\n      \"rematerialization_eligible\": false,\n      \"ir_position\": 2" },
        .{ "\"defining_opcode\": \"load\",\n      \"source_class\": \"memory_or_runtime\",\n      \"rematerialization_eligible\": false,\n      \"ir_position\": 2", "\"defining_opcode\": \"load\",\n      \"source_class\": \"memory_or_runtime\",\n      \"rematerialization_eligible\": true,\n      \"ir_position\": 2" },
        .{ "\"defining_opcode\": \"load\",\n      \"source_class\": \"memory_or_runtime\",\n      \"rematerialization_eligible\": false,\n      \"ir_position\": 2", "\"defining_opcode\": \"load\",\n      \"source_class\": \"memory_or_runtime\",\n      \"rematerialization_eligible\": 0,\n      \"ir_position\": 2" },
    };
    for (bad_x86_cases) |pair| {
        const mod = try replaceFirst(alloc, base_x86, pair[0], pair[1]);
        try std.testing.expectError(
            error.AttributionFailed,
            loadFrameMetadataFromJson(alloc, "metadata.json", mod, 7, &TEST_LOAD_STORE_X86, AOT_VERSION, &TEST_LOAD_STORE_X86, 0, makeTestArtifactInfo(&TEST_LOAD_STORE_X86, "x86_64")),
        );
        try std.testing.expect(std.mem.indexOf(u8, lastError(), "allocator source") != null);
    }

    const amb_needle = "\"vreg\": null,\n      \"vreg_ambiguous\": true,\n      \"defining_opcode\": null,\n      \"source_class\": null,\n      \"rematerialization_eligible\": null";
    const amb_repl = "\"vreg\": null,\n      \"vreg_ambiguous\": true,\n      \"defining_opcode\": \"load\",\n      \"source_class\": null,\n      \"rematerialization_eligible\": null";
    const mod_amb = try replaceFirst(alloc, base_x86, amb_needle, amb_repl);
    try std.testing.expectError(
        error.AttributionFailed,
        loadFrameMetadataFromJson(alloc, "metadata.json", mod_amb, 7, &TEST_LOAD_STORE_X86, AOT_VERSION, &TEST_LOAD_STORE_X86, 0, makeTestArtifactInfo(&TEST_LOAD_STORE_X86, "x86_64")),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "must be null") != null);
}

test "FrameAttributionUnitTests.test_frame_operand_supports_signed_rsp_and_complex_forms" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const op1 = (try parseFrameOperand(alloc, "mov rax,QWORD PTR [rbp-0x230]")).?;
    try std.testing.expect(op1.eql(.{ .kind = "load", .base = "rbp", .offset = -560, .complex_address = false }));

    const op2 = (try parseFrameOperand(alloc, "mov QWORD PTR [rbp+0x30],rax")).?;
    try std.testing.expect(op2.eql(.{ .kind = "store", .base = "rbp", .offset = 48, .complex_address = false }));

    const op3 = (try parseFrameOperand(alloc, "mov rdx,QWORD PTR [rsp]")).?;
    try std.testing.expect(op3.eql(.{ .kind = "load", .base = "rsp", .offset = 0, .complex_address = false }));

    const op4 = (try parseFrameOperand(alloc, "push r12")).?;
    try std.testing.expect(op4.eql(.{ .kind = "store", .base = "rsp", .offset = -8, .complex_address = false }));

    const op5 = (try parseFrameOperand(alloc, "pop r12")).?;
    try std.testing.expect(op5.eql(.{ .kind = "load", .base = "rsp", .offset = 0, .complex_address = false }));

    const complex_op = (try parseFrameOperand(alloc, "mov rax,QWORD PTR [rbp+rcx*8-0x20]")).?;
    try std.testing.expectEqualStrings("rbp", complex_op.base);
    try std.testing.expect(complex_op.offset == null);
    try std.testing.expect(complex_op.complex_address);

    try std.testing.expect((try parseFrameOperand(alloc, "lea rax,[rbp+rcx*8-0x20]")) == null);
}

test "FrameAttributionUnitTests.test_reused_slot_accepts_emitted_range_identity_and_ambiguity" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const json = try makeDefaultX86MetadataJson(alloc, &TEST_LOAD_STORE_X86);
    const meta = try loadFrameMetadataFromJson(
        alloc,
        "metadata.json",
        json,
        7,
        &TEST_LOAD_STORE_X86,
        AOT_VERSION,
        &TEST_LOAD_STORE_X86,
        0,
        makeTestArtifactInfo(&TEST_LOAD_STORE_X86, "x86_64"),
    );
    try std.testing.expectEqual(@as(?i64, 10), meta.accessAtStart(0).?.vreg);
    try std.testing.expect(meta.accessAtStart(TEST_LOAD_X86.len).?.vreg_ambiguous);
    const slot0 = try meta.valuesForSlot(alloc, 0);
    try std.testing.expectEqual(@as(usize, 2), slot0.len);
    try std.testing.expectEqual(@as(i64, 10), slot0[0].vreg);
    try std.testing.expectEqual(@as(i64, 11), slot0[1].vreg);
}

test "FrameAttributionUnitTests.test_reused_slot_without_identity_or_ambiguity_fails" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const base_json = try makeDefaultX86MetadataJson(alloc, &TEST_LOAD_STORE_X86);
    const mod = try replaceFirst(alloc, base_json, "\"vreg_ambiguous\": true", "\"vreg_ambiguous\": false");
    try std.testing.expectError(
        error.AttributionFailed,
        loadFrameMetadataFromJson(alloc, "metadata.json", mod, 7, &TEST_LOAD_STORE_X86, AOT_VERSION, &TEST_LOAD_STORE_X86, 0, makeTestArtifactInfo(&TEST_LOAD_STORE_X86, "x86_64")),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "reused slot 0") != null);
}

test "FrameAttributionUnitTests.test_reconciliation_mismatch_fails" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const base_json = try makeDefaultX86MetadataJson(alloc, &TEST_LOAD_STORE_X86);
    const mod = try replaceFirst(alloc, base_json, "\"spill_ld\": 1", "\"spill_ld\": 2");
    try std.testing.expectError(
        error.AttributionFailed,
        loadFrameMetadataFromJson(alloc, "metadata.json", mod, 7, &TEST_LOAD_STORE_X86, AOT_VERSION, &TEST_LOAD_STORE_X86, 0, makeTestArtifactInfo(&TEST_LOAD_STORE_X86, "x86_64")),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "reconciliation failed") != null);
}

test "FrameAttributionUnitTests.test_schema_hash_and_function_mismatches_fail_closed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const base_json = try makeDefaultX86MetadataJson(alloc, &TEST_LOAD_STORE_X86);
    const bad_sv = try replaceFirst(alloc, base_json, "\"schema_version\": 1", "\"schema_version\": 99");
    try std.testing.expectError(
        error.AttributionFailed,
        loadFrameMetadataFromJson(alloc, "metadata.json", bad_sv, 7, &TEST_LOAD_STORE_X86, AOT_VERSION, &TEST_LOAD_STORE_X86, 0, makeTestArtifactInfo(&TEST_LOAD_STORE_X86, "x86_64")),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "schema_version") != null);

    const zero_hash = "0000000000000000000000000000000000000000000000000000000000000000";
    const norm_hash = try normalizedCodeSha256(alloc, &TEST_LOAD_STORE_X86, &.{});
    const bad_norm = try replaceFirst(
        alloc,
        base_json,
        try std.fmt.allocPrint(alloc, "\"normalized_code_sha256\": \"{s}\"", .{norm_hash}),
        try std.fmt.allocPrint(alloc, "\"normalized_code_sha256\": \"{s}\"", .{zero_hash}),
    );
    try std.testing.expectError(
        error.AttributionFailed,
        loadFrameMetadataFromJson(alloc, "metadata.json", bad_norm, 7, &TEST_LOAD_STORE_X86, AOT_VERSION, &TEST_LOAD_STORE_X86, 0, makeTestArtifactInfo(&TEST_LOAD_STORE_X86, "x86_64")),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "hash mismatch") != null);

    const mod_hash = try sha256Hex(alloc, &TEST_LOAD_STORE_X86);
    const bad_mod = try replaceFirst(
        alloc,
        base_json,
        try std.fmt.allocPrint(alloc, "\"module_text_sha256\": \"{s}\"", .{mod_hash}),
        try std.fmt.allocPrint(alloc, "\"module_text_sha256\": \"{s}\"", .{zero_hash}),
    );
    try std.testing.expectError(
        error.AttributionFailed,
        loadFrameMetadataFromJson(alloc, "metadata.json", bad_mod, 7, &TEST_LOAD_STORE_X86, AOT_VERSION, &TEST_LOAD_STORE_X86, 0, makeTestArtifactInfo(&TEST_LOAD_STORE_X86, "x86_64")),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "module text hash") != null);

    const bad_func = try replaceFirst(alloc, base_json, "\"local_func\": 7", "\"local_func\": 8");
    try std.testing.expectError(
        error.AttributionFailed,
        loadFrameMetadataFromJson(alloc, "metadata.json", bad_func, 7, &TEST_LOAD_STORE_X86, AOT_VERSION, &TEST_LOAD_STORE_X86, 0, makeTestArtifactInfo(&TEST_LOAD_STORE_X86, "x86_64")),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "does not match") != null);
}

test "FrameAttributionUnitTests.test_malformed_and_overlapping_metadata_fail_closed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    try std.testing.expectError(
        error.AttributionFailed,
        loadFrameMetadataFromJson(alloc, "broken.json", "{", 7, &TEST_LOAD_STORE_X86, AOT_VERSION, &TEST_LOAD_STORE_X86, 0, makeTestArtifactInfo(&TEST_LOAD_STORE_X86, "x86_64")),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "malformed") != null);

    const base_json = try makeDefaultX86MetadataJson(alloc, &TEST_LOAD_STORE_X86);
    const overlap = try replaceFirst(alloc, base_json, "\"native_start\": 7", "\"native_start\": 1");
    try std.testing.expectError(
        error.AttributionFailed,
        loadFrameMetadataFromJson(alloc, "metadata.json", overlap, 7, &TEST_LOAD_STORE_X86, AOT_VERSION, &TEST_LOAD_STORE_X86, 0, makeTestArtifactInfo(&TEST_LOAD_STORE_X86, "x86_64")),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "overlapping/out-of-range") != null);
}

test "FrameAttributionUnitTests.test_disassembly_mapping_mismatch_fails" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const base_json = try makeDefaultX86MetadataJson(alloc, &TEST_LOAD_STORE_X86);
    const meta = try loadFrameMetadataFromJson(alloc, "metadata.json", base_json, 7, &TEST_LOAD_STORE_X86, AOT_VERSION, &TEST_LOAD_STORE_X86, 0, makeTestArtifactInfo(&TEST_LOAD_STORE_X86, "x86_64"));
    const instructions = [_]Instruction{
        .{ .address = 0x1000, .offset = 0, .size = TEST_LOAD_X86.len, .text = "mov rax,QWORD PTR [rbp-0x230]" },
        .{ .address = 0x1000 + TEST_LOAD_X86.len, .offset = TEST_LOAD_X86.len, .size = TEST_STORE_X86.len, .text = "mov QWORD PTR [rbp-0x238],rax" },
    };
    try std.testing.expectError(error.AttributionFailed, validateMetadataDisassembly(alloc, &meta, &instructions));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "offset mismatch") != null);
}

test "FrameAttributionUnitTests.test_inline_br_table_ranges_split_disassembly" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const MockDis = struct {
        fn fake(
            a: Allocator,
            _: std.Io,
            blob: []const u8,
            address: u64,
            _: []const u8,
            _: []const u8,
            offset_base: usize,
            arch: ?[]const u8,
        ) AttributionError![]Instruction {
            if (!std.mem.eql(u8, arch orelse "", "x86_64")) return fail("wrong arch", .{});
            if (blob.len == 0) return &.{};
            const slice = try a.alloc(Instruction, 1);
            slice[0] = .{ .address = address, .offset = offset_base, .size = blob.len, .text = "nop" };
            return slice;
        }
    };

    const ranges = [_]InlineDataRange{.{ .native_start = 3, .native_end = 7, .kind = "br_table" }};
    const insts = try disassembleFunctionWith(alloc, std.testing.io, "abcdefghijkl", 0x1000, ".", "table", &ranges, "x86_64", MockDis.fake);
    try std.testing.expectEqual(@as(usize, 2), insts.len);
    try std.testing.expectEqual(@as(usize, 0), insts[0].offset);
    try std.testing.expectEqual(@as(usize, 3), insts[0].size);
    try std.testing.expectEqual(@as(u64, 0x1000), insts[0].address);
    try std.testing.expectEqual(@as(usize, 7), insts[1].offset);
    try std.testing.expectEqual(@as(usize, 5), insts[1].size);
    try std.testing.expectEqual(@as(u64, 0x1007), insts[1].address);
}

test "FrameAttributionUnitTests.test_mixed_origin_summary_reports_coverage_and_unknowns" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const values = [_]AllocatorValue{.{
        .vreg = 10,
        .frame_offset = -560,
        .slot = 0,
        .slot_count = 1,
        .value_type = "i64",
        .defining_opcode = "load",
        .source_class = "memory_or_runtime",
        .ir_use_count = 1,
        .ir_def_count = 1,
        .reload_count = 1,
        .store_count = 0,
        .rematerialization_eligible = false,
        .reused = true,
    }};
    const accesses = [_]FrameAccess{
        .{ .native_start = 0, .native_end = 7, .kind = "load", .base = "rbp", .frame_offset = -560, .width = 8, .origin = "allocator_spill", .slot = 0, .vreg = 10, .vreg_ambiguous = false, .defining_opcode = "load", .source_class = "memory_or_runtime", .rematerialization_eligible = false },
        .{ .native_start = 7, .native_end = 14, .kind = "store", .base = "rbp", .frame_offset = -16, .width = 8, .origin = "wasm_local_or_phi" },
        .{ .native_start = 14, .native_end = 21, .kind = "load", .base = "rbp", .frame_offset = -32, .width = 8, .origin = "explicit_frame_storage" },
        .{ .native_start = 21, .native_end = 29, .kind = "store", .base = "rsp", .frame_offset = 32, .width = 8, .origin = "fixed_runtime_frame_state" },
    };
    const meta = FrameMetadata{
        .architecture = "x86_64",
        .schema_version = 1,
        .allocator_values = &values,
        .accesses = &accesses,
        .reconciliation = .{
            .emitted_allocator_loads = 1,
            .emitted_allocator_stores = 0,
            .spill_metric_loads = 1,
            .spill_metric_stores = 0,
            .matches = true,
        },
    };
    const instructions = [_]Instruction{
        .{ .address = 0x1000, .offset = 0, .size = 7, .text = "mov rax,QWORD PTR [rbp-0x230]" },
        .{ .address = 0x1007, .offset = 7, .size = 7, .text = "mov QWORD PTR [rbp-0x10],rax" },
        .{ .address = 0x100E, .offset = 14, .size = 7, .text = "mov rax,QWORD PTR [rbp-0x20]" },
        .{ .address = 0x1015, .offset = 21, .size = 8, .text = "mov QWORD PTR [rsp+0x20],rax" },
        .{ .address = 0x101D, .offset = 29, .size = 7, .text = "mov rax,QWORD PTR [rbp-0x88]" },
    };
    var counts: std.AutoHashMapUnmanaged(u64, u64) = .empty;
    try counts.put(alloc, 0x1000, 30);
    try counts.put(alloc, 0x1007, 20);
    try counts.put(alloc, 0x100E, 10);
    try counts.put(alloc, 0x1015, 5);
    try counts.put(alloc, 0x101D, 2);

    const summary = try buildFrameSummary(alloc, &instructions, &counts, &meta);
    try std.testing.expectEqual(@as(u64, 5), summary.coverage.frame_instructions);
    try std.testing.expectEqual(@as(u64, 4), summary.coverage.attributed_frame_instructions);
    try std.testing.expectEqual(@as(u64, 2), summary.coverage.unknown_frame_samples);
    try std.testing.expectEqual(@as(u64, 30), summary.originByName("allocator_spill").?.samples);
    try std.testing.expectEqual(@as(u64, 20), summary.originByName("wasm_local_or_phi").?.samples);
}

test "FrameAttributionUnitTests.test_cwasm_version_and_duplicate_offsets_fail_closed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const text = [_]u8{ 0x90, 0xc3 };
    var func_sec: [12]u8 = @splat(0);
    std.mem.writeInt(u32, func_sec[0..4], 1, .little);
    const tinfo = makeTestTargetInfo("x86_64", 2, 1, 0x3E, 0);

    var good: std.ArrayList(u8) = .empty;
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], AOT_MAGIC, .little);
    std.mem.writeInt(u32, hdr[4..8], AOT_VERSION, .little);
    try good.appendSlice(alloc, &hdr);
    try appendTestSection(alloc, &good, SEC_TARGET_INFO, &tinfo);
    try appendTestSection(alloc, &good, SEC_TEXT, &text);
    try appendTestSection(alloc, &good, SEC_FUNCTION, &func_sec);

    const info = try parseCwasmBytes(alloc, "ok.cwasm", good.items);
    try std.testing.expectEqualSlices(u32, &.{0}, info.func_offsets);

    var prev = try alloc.dupe(u8, good.items);
    std.mem.writeInt(u32, prev[4..8], AOT_VERSION - 1, .little);
    try std.testing.expectEqual(AOT_VERSION - 1, (try parseCwasmBytes(alloc, "prior.cwasm", prev)).version);

    var bad_ver = try alloc.dupe(u8, good.items);
    std.mem.writeInt(u32, bad_ver[4..8], 8, .little);
    try std.testing.expectError(error.AttributionFailed, parseCwasmBytes(alloc, "old.cwasm", bad_ver));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "incompatible") != null);

    var dup_func_sec: [20]u8 = @splat(0);
    std.mem.writeInt(u32, dup_func_sec[0..4], 2, .little);
    var dup_blob: std.ArrayList(u8) = .empty;
    try dup_blob.appendSlice(alloc, &hdr);
    try appendTestSection(alloc, &dup_blob, SEC_TARGET_INFO, &tinfo);
    try appendTestSection(alloc, &dup_blob, SEC_TEXT, &text);
    try appendTestSection(alloc, &dup_blob, SEC_FUNCTION, &dup_func_sec);
    try std.testing.expectError(error.AttributionFailed, parseCwasmBytes(alloc, "ambiguous.cwasm", dup_blob.items));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "ambiguous/non-increasing") != null);
}

test "FrameAttributionUnitTests.test_cwasm_target_info_is_unique_and_architecture_bound" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const text = [_]u8{ 0x90, 0xc3 };
    var func_sec: [12]u8 = @splat(0);
    std.mem.writeInt(u32, func_sec[0..4], 1, .little);

    const Builder = struct {
        fn build(a: Allocator, targets: []const []const u8, txt: []const u8, fn_sec: []const u8) ![]u8 {
            var list: std.ArrayList(u8) = .empty;
            var hdr: [8]u8 = undefined;
            std.mem.writeInt(u32, hdr[0..4], AOT_MAGIC, .little);
            std.mem.writeInt(u32, hdr[4..8], AOT_VERSION, .little);
            try list.appendSlice(a, &hdr);
            for (targets) |t| {
                try appendTestSection(a, &list, SEC_TARGET_INFO, t);
            }
            try appendTestSection(a, &list, SEC_TEXT, txt);
            try appendTestSection(a, &list, SEC_FUNCTION, fn_sec);
            return list.toOwnedSlice(a);
        }
    };

    const t_good = makeTestTargetInfo("x86_64", 2, 1, 0x3E, 0);
    const info = try parseCwasmBytes(alloc, "verified.cwasm", try Builder.build(alloc, &.{&t_good}, &text, &func_sec));
    try std.testing.expectEqualStrings("x86_64", info.architecture);
    try std.testing.expectEqualStrings("sysv", info.abi.?);
    try std.testing.expect(info.target_verified);

    const t_legacy = makeTestTargetInfo("x86_64", 1, 0, 0, 0);
    const legacy_info = try parseCwasmBytes(alloc, "legacy.cwasm", try Builder.build(alloc, &.{&t_legacy}, &text, &func_sec));
    try std.testing.expect(legacy_info.abi == null);
    try std.testing.expect(!legacy_info.target_verified);

    try std.testing.expectError(error.AttributionFailed, parseCwasmBytes(alloc, "bad.cwasm", try Builder.build(alloc, &.{}, &text, &func_sec)));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "missing target-info") != null);

    try std.testing.expectError(error.AttributionFailed, parseCwasmBytes(alloc, "bad.cwasm", try Builder.build(alloc, &.{t_good[0..39]}, &text, &func_sec)));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "exactly 40") != null);

    try std.testing.expectError(error.AttributionFailed, parseCwasmBytes(alloc, "bad.cwasm", try Builder.build(alloc, &.{ &t_good, &t_good }, &text, &func_sec)));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "duplicate target-info") != null);

    const t_inconsistent = makeTestTargetInfo("aarch64", 2, 1, 0x3E, 0);
    try std.testing.expectError(error.AttributionFailed, parseCwasmBytes(alloc, "bad.cwasm", try Builder.build(alloc, &.{&t_inconsistent}, &text, &func_sec)));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "inconsistent target-info") != null);
}

test "FrameAttributionUnitTests.test_exact_page_rounded_jit_mapping_is_required" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const maps1 = [_]MmapCandidate{.{ .base = 0x1000, .size = 8192 }};
    try std.testing.expectError(error.AttributionFailed, selectTextMappingFromCandidates(alloc, &maps1, 4096, null, 4096));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "found 0 exact matches") != null);

    const maps2 = [_]MmapCandidate{.{ .base = 0x2000, .size = 8192 }};
    const sel = try selectTextMappingFromCandidates(alloc, &maps2, 5000, null, 4096);
    try std.testing.expectEqual(@as(u64, 0x2000), sel.base);
    try std.testing.expectEqual(@as(usize, 8192), sel.expected_size);
    try std.testing.expect(sel.authoritative);

    const maps3 = [_]MmapCandidate{ .{ .base = 0x1000, .size = 4096 }, .{ .base = 0x4000, .size = 4096 } };
    try std.testing.expectError(error.AttributionFailed, selectTextMappingFromCandidates(alloc, &maps3, 4096, null, 4096));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "exactly one") != null);
}

test "FrameAttributionUnitTests.test_manual_mapping_override_is_non_authoritative" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const maps = [_]MmapCandidate{.{ .base = 0x4000, .size = 8192 }};
    const sel = try selectTextMappingFromCandidates(alloc, &maps, 4096, "0x4000", 4096);
    try std.testing.expectEqual(@as(u64, 0x4000), sel.base);
    try std.testing.expect(!sel.authoritative);
    try std.testing.expectEqualStrings("0x4000", sel.override.?);

    try std.testing.expectError(error.AttributionFailed, selectTextMappingFromCandidates(alloc, &.{}, 4096, "not-a-base", 4096));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "hexadecimal") != null);
}

test "FrameAttributionUnitTests.test_attribution_coverage_threshold_fails_closed" {
    try std.testing.expect((try requireAttributionCoverage(34398, 34410, 99.0)) > 99.9);
    try std.testing.expect((try requireAttributionCoverage(34404, 34419, 99.0)) > 99.9);
    try std.testing.expectError(error.AttributionFailed, requireAttributionCoverage(980, 1000, 99.0));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "below required") != null);
}

test "FrameAttributionUnitTests.test_address_counts_fall_back_to_perf_script_self_ips" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const report = "no parseable per-address rows\n";
    const r_counts = try parsePerfReportOutput(alloc, report);
    try std.testing.expectEqual(@as(u64, 0), r_counts.total);

    const script =
        \\0xffff00001000 ([JIT])
        \\0xffff00001000 ([JIT])
        \\0xaaaa00002000 /usr/bin/wamr
        \\
    ;
    const s_counts = try parsePerfScriptOutput(alloc, script);
    try std.testing.expectEqual(@as(u64, 3), s_counts.total);
    try std.testing.expectEqual(@as(?u64, 2), s_counts.counts.get(0xFFFF00001000));
    try std.testing.expectEqual(@as(?u64, 1), s_counts.counts.get(0xAAAA00002000));
}

test "FrameAttributionUnitTests.test_current_aot_version_matches_emitter" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const source = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "src/compiler/emit_aot.zig", alloc, .limited(4 * 1024 * 1024));
    const needle = "pub const aot_version: u32 = ";
    const idx = std.mem.indexOf(u8, source, needle).?;
    const after = source[idx + needle.len ..];
    const semi = std.mem.indexOfScalar(u8, after, ';').?;
    const emitted_version = try std.fmt.parseInt(u32, after[0..semi], 10);
    try std.testing.expectEqual(AOT_VERSION, emitted_version);
}

test "FrameAttributionUnitTests.test_aarch64_instruction_classes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const texts = [_][]const u8{
        "add x17, x15, #1",
        "ldr x16, [x19, #8]",
        "cmp x17, x16",
        "b.ls 0x1020",
        "ldr w8, [x20, x15]",
        "ldr x9, [x29, #-16]",
        "str x10, [sp, #32]",
        "mov w11, w8",
        "add w12, w11, w10",
        "bl 0x2000",
        "br x13",
        "b 0x3000",
        "cbnz w8, 0x4000",
    };
    var insts: [texts.len]Instruction = undefined;
    for (texts, 0..) |t, i| {
        insts[i] = .{ .address = 0x1000 + @as(u64, @intCast(i * 4)), .offset = i * 4, .size = 4, .text = t };
    }
    const expected = [_][]const u8{
        "alu",
        "mem_access",
        "bounds_cmp",
        "bounds_branch",
        "linear_memory",
        "frame_load_unattributed",
        "frame_store_unattributed",
        "regmov",
        "alu",
        "call",
        "indirect_dispatch",
        "direct_branch",
        "cond_branch",
    };
    const actual = try classifyInstructionStream(alloc, &insts, null, "aarch64");
    for (expected, actual) |exp, act| {
        try std.testing.expectEqualStrings(exp, act);
    }
}

test "FrameAttributionUnitTests.test_x86_classification_remains_compatible" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    try std.testing.expectEqualStrings("regmov", try classifyBasic(alloc, "mov eax,edx", "x86_64"));
    try std.testing.expectEqualStrings("dispatch_jmp", try classifyBasic(alloc, "jmp r10", "x86_64"));
    try std.testing.expectEqualStrings("frame_load_unattributed", try classifyBasic(alloc, "mov rax,QWORD PTR [rbp-0x20]", "x86_64"));
}

test "FrameAttributionUnitTests.test_aarch64_objdump_selection_and_parsing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const argv = try buildObjdumpArgv(alloc, 0x1000, "scratch.bin", "aarch64");
    var has_aarch64 = false;
    var has_intel = false;
    for (argv) |a| {
        if (std.mem.eql(u8, a, "aarch64")) has_aarch64 = true;
        if (std.mem.eql(u8, a, "intel")) has_intel = true;
    }
    try std.testing.expect(has_aarch64);
    try std.testing.expect(!has_intel);

    const parsed = try parseObjdumpOutput(alloc, "   1000:  8b010000  add x0, x0, x1\n", 0x1000, 0);
    try std.testing.expectEqual(@as(usize, 1), parsed.len);
    try std.testing.expectEqualStrings("add x0, x0, x1", parsed[0].text);
    try std.testing.expectEqual(@as(usize, 4), parsed[0].size);
}

test "FrameAttributionUnitTests.test_direct_call_rel32_is_the_only_normalized_function_region" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const first = [_]u8{ 0x90, 0xe8, 0x01, 0x02, 0x03, 0x04, 0xc3 };
    const second = [_]u8{ 0x90, 0xe8, 0xaa, 0xbb, 0xcc, 0xdd, 0xc3 };
    const h1 = try normalizedCodeSha256(alloc, &first, &.{2});
    const h2 = try normalizedCodeSha256(alloc, &second, &.{2});
    try std.testing.expectEqualStrings(h1, h2);
    try std.testing.expectError(error.AttributionFailed, normalizedCodeSha256(alloc, &first, &.{ 2, 3 }));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "overlapping") != null);
}

test "FrameAttributionUnitTests.test_aarch64_schema_v2_relocation_and_encoded_address_validation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const base_arm = try makeDefaultAarch64MetadataJson(alloc, &TEST_ARM_CODE);
    const meta = try loadFrameMetadataFromJson(alloc, "metadata.json", base_arm, 10, &TEST_ARM_CODE, AOT_VERSION, &TEST_ARM_CODE, 0, makeTestArtifactInfo(&TEST_ARM_CODE, "aarch64"));
    try std.testing.expectEqualStrings("aarch64", meta.architecture);

    const insts = [_]Instruction{
        .{ .address = 0, .offset = 0, .size = 4, .text = "bl 0x4" },
        .{ .address = 4, .offset = 4, .size = 4, .text = "ldr x16, [x29, #248]" },
        .{ .address = 8, .offset = 8, .size = 4, .text = "str x16, [x29, #248]" },
    };
    try validateMetadataDisassembly(alloc, &meta, &insts);

    const bad_rel = try replaceFirst(alloc, base_arm, "\"aarch64_call_imm26\"", "\"x86_64_call_rel32\"");
    try std.testing.expectError(error.AttributionFailed, loadFrameMetadataFromJson(alloc, "metadata.json", bad_rel, 10, &TEST_ARM_CODE, AOT_VERSION, &TEST_ARM_CODE, 0, makeTestArtifactInfo(&TEST_ARM_CODE, "aarch64")));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "AArch64 relocation kind") != null);

    const corrupt_bl = TEST_ARM_B ++ TEST_ARM_LDR ++ TEST_ARM_STR;
    const corrupt_mod_hash = try sha256Hex(alloc, &corrupt_bl);
    const orig_mod_hash = try sha256Hex(alloc, &TEST_ARM_CODE);
    const bad_bl_json = try replaceFirst(alloc, base_arm, orig_mod_hash, corrupt_mod_hash);
    try std.testing.expectError(error.AttributionFailed, loadFrameMetadataFromJson(alloc, "metadata.json", bad_bl_json, 10, &corrupt_bl, AOT_VERSION, &corrupt_bl, 0, makeTestArtifactInfo(&corrupt_bl, "aarch64")));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "does not match") != null);

    const bad_enc_off = try replaceFirst(alloc, base_arm, "\"encoded_offset\": 248", "\"encoded_offset\": 240");
    try std.testing.expectError(error.AttributionFailed, loadFrameMetadataFromJson(alloc, "metadata.json", bad_enc_off, 10, &TEST_ARM_CODE, AOT_VERSION, &TEST_ARM_CODE, 0, makeTestArtifactInfo(&TEST_ARM_CODE, "aarch64")));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "native encoded_offset") != null);

    const tail_relocs = [_]NormalizedRelocation{.{ .native_start = 0, .native_end = 4, .kind = "aarch64_tail_call_imm26" }};
    const arm_b_other = [_]u8{ 0xff, 0x03, 0x00, 0x14 };
    try std.testing.expectEqualStrings(
        try normalizedCodeSha256V2(alloc, &TEST_ARM_B, &tail_relocs, "aarch64"),
        try normalizedCodeSha256V2(alloc, &arm_b_other, &tail_relocs, "aarch64"),
    );
    try std.testing.expectError(error.AttributionFailed, normalizedCodeSha256V2(alloc, &TEST_ARM_BL, &tail_relocs, "aarch64"));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "does not match") != null);
}

test "FrameAttributionUnitTests.test_frame_metadata_requires_verified_matching_artifact_target" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const base_arm = try makeDefaultAarch64MetadataJson(alloc, &TEST_ARM_CODE);
    var legacy = makeTestArtifactInfo(&TEST_ARM_CODE, "aarch64");
    legacy.abi = null;
    legacy.target_format = "legacy-unspecified";
    legacy.target_verified = false;
    try std.testing.expectError(
        error.AttributionFailed,
        loadFrameMetadataFromJson(alloc, "metadata.json", base_arm, 10, &TEST_ARM_CODE, AOT_VERSION, &TEST_ARM_CODE, 0, legacy),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "legacy/unspecified") != null);
}

test "FrameAttributionUnitTests.test_aarch64_schema_v2_fails_closed_on_corrupt_identity_and_components" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const base_arm = try makeDefaultAarch64MetadataJson(alloc, &TEST_ARM_CODE);
    const art = makeTestArtifactInfo(&TEST_ARM_CODE, "aarch64");

    const bad_abi = try replaceFirst(alloc, base_arm, "\"abi\": \"aapcs64\"", "\"abi\": \"sysv\"");
    try std.testing.expectError(error.AttributionFailed, loadFrameMetadataFromJson(alloc, "metadata.json", bad_abi, 10, &TEST_ARM_CODE, AOT_VERSION, &TEST_ARM_CODE, 0, art));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "ABI") != null);

    const bad_comp_width = try replaceFirst(alloc, base_arm, "\"data_register\": 16", "\"width\": 4,\n          \"data_register\": 16");
    try std.testing.expectError(error.AttributionFailed, loadFrameMetadataFromJson(alloc, "metadata.json", bad_comp_width, 10, &TEST_ARM_CODE, AOT_VERSION, &TEST_ARM_CODE, 0, art));

    const bad_enc_base = try replaceFirst(alloc, base_arm, "\"encoded_base\": \"x29\"", "\"encoded_base\": \"x31\"");
    try std.testing.expectError(error.AttributionFailed, loadFrameMetadataFromJson(alloc, "metadata.json", bad_enc_base, 10, &TEST_ARM_CODE, AOT_VERSION, &TEST_ARM_CODE, 0, art));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "encoded address") != null);

    const bad_val_off = try replaceFirst(alloc, base_arm, "\"vreg\": 77,\n      \"frame_offset\": 248", "\"vreg\": 77,\n      \"frame_offset\": 256");
    try std.testing.expectError(error.AttributionFailed, loadFrameMetadataFromJson(alloc, "metadata.json", bad_val_off, 10, &TEST_ARM_CODE, AOT_VERSION, &TEST_ARM_CODE, 0, art));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "allocator slot/offset") != null);

    const bad_region_end = try replaceFirst(alloc, base_arm, "\"start\": 248, \"end\": 256", "\"start\": 248, \"end\": 252");
    try std.testing.expectError(error.AttributionFailed, loadFrameMetadataFromJson(alloc, "metadata.json", bad_region_end, 10, &TEST_ARM_CODE, AOT_VERSION, &TEST_ARM_CODE, 0, art));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "outside declared regions") != null);

    const bad_reg = try replaceFirst(alloc, base_arm, "\"data_register\": 16", "\"data_register\": 17");
    try std.testing.expectError(error.AttributionFailed, loadFrameMetadataFromJson(alloc, "metadata.json", bad_reg, 10, &TEST_ARM_CODE, AOT_VERSION, &TEST_ARM_CODE, 0, art));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "native component") != null);

    const bad_overlap = try replaceFirst(alloc, base_arm, "\"native_start\": 8", "\"native_start\": 6");
    try std.testing.expectError(error.AttributionFailed, loadFrameMetadataFromJson(alloc, "metadata.json", bad_overlap, 10, &TEST_ARM_CODE, AOT_VERSION, &TEST_ARM_CODE, 0, art));
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "overlapping/out-of-range") != null);
}

fn makeAarch64PairMetadataJson(allocator: Allocator, mixed: bool) !struct { json: []u8, code: [4]u8 } {
    var code: [4]u8 = undefined;
    std.mem.writeInt(u32, &code, 0xA94307A0, .little); // ldp x0, x1, [x29, #48]
    const h = try sha256Hex(allocator, &code);
    const json = if (!mixed)
        try std.fmt.allocPrint(
            allocator,
            \\{{
            \\  "schema": "wamr-aot-frame-attribution",
            \\  "schema_version": 2,
            \\  "cwasm_aot_version": {d},
            \\  "compiler_build_id": "test",
            \\  "architecture": "aarch64",
            \\  "abi": "aapcs64",
            \\  "module": 0,
            \\  "local_func": 10,
            \\  "function_name": "core_state_transition",
            \\  "module_text_size": 4,
            \\  "module_text_sha256": "{s}",
            \\  "function_offset": 0,
            \\  "code_size": 4,
            \\  "normalized_code_sha256": "{s}",
            \\  "direct_call_rel32_offsets": [],
            \\  "normalized_relocations": [],
            \\  "inline_data_ranges": [],
            \\  "frame_layout": {{
            \\    "frame_pointer": "x29",
            \\    "frame_size": 512,
            \\    "local_count": 1,
            \\    "param_count": 0,
            \\    "reserved_vmctx_offset": 16,
            \\    "locals_first_offset": 24,
            \\    "explicit_storage_first_offset": 264,
            \\    "explicit_storage_slots": 1,
            \\    "spill_base": 48,
            \\    "spill_stride": 8,
            \\    "spill_slots": 2
            \\  }},
            \\  "frame_regions": [
            \\    {{ "start": 0, "end": 16, "origin": "fixed_runtime_frame_state", "detail": "saved_fp_lr" }},
            \\    {{ "start": 16, "end": 24, "origin": "fixed_runtime_frame_state", "detail": "reserved_vmctx" }},
            \\    {{ "start": 24, "end": 32, "origin": "wasm_local_or_phi", "detail": "wasm_local_or_lowered_phi" }},
            \\    {{ "start": 48, "end": 64, "origin": "allocator_spill", "detail": "scalar_allocator_spills" }}
            \\  ],
            \\  "spill_metric": {{
            \\    "slots": 2, "spilled_vregs": 2, "scalar": 2, "v128": 0,
            \\    "slots_scalar": 2, "slots_v128": 0, "spill_ld": 2, "spill_st": 0, "remat": 0, "callee_saved": 0
            \\  }},
            \\  "emitted_allocator_loads": 2,
            \\  "emitted_allocator_stores": 0,
            \\  "allocator_values": [
            \\    {{ "vreg": 70, "frame_offset": 48, "slot": 0, "slot_count": 1, "value_type": "i64", "live_start": 4, "live_end": 12, "defining_opcode": "load", "source_class": "memory_or_runtime", "ir_use_count": 1, "ir_def_count": 1, "reload_count": 1, "store_count": 0, "rematerialization_eligible": false, "reused": false }},
            \\    {{ "vreg": 71, "frame_offset": 56, "slot": 1, "slot_count": 1, "value_type": "i64", "live_start": 4, "live_end": 12, "defining_opcode": "load", "source_class": "memory_or_runtime", "ir_use_count": 1, "ir_def_count": 1, "reload_count": 1, "store_count": 0, "rematerialization_eligible": false, "reused": false }}
            \\  ],
            \\  "accesses": [
            \\    {{
            \\      "native_start": 0, "native_end": 4, "kind": "load", "base": "x29", "frame_offset": 48, "width": 16,
            \\      "origin": "allocator_spill", "detail": "allocator_slot", "slot": 0, "local_index": null, "explicit_slot": null,
            \\      "vreg": null, "vreg_ambiguous": true, "defining_opcode": null, "source_class": null, "rematerialization_eligible": null,
            \\      "ir_position": 8, "ir_opcode": "add", "encoded_base": "x29", "encoded_offset": 48, "addressing_mode": "signed_pair",
            \\      "components": [
            \\        {{ "base": "x29", "frame_offset": 48, "width": 8, "data_register": 0, "data_register_class": "gpr", "origin": "allocator_spill", "detail": "allocator_slot", "slot": 0, "local_index": null, "explicit_slot": null, "vreg": 70, "vreg_ambiguous": false, "defining_opcode": "load", "source_class": "memory_or_runtime", "rematerialization_eligible": false, "ir_position": 8, "ir_opcode": "add" }},
            \\        {{ "base": "x29", "frame_offset": 56, "width": 8, "data_register": 1, "data_register_class": "gpr", "origin": "allocator_spill", "detail": "allocator_slot", "slot": 1, "local_index": null, "explicit_slot": null, "vreg": 71, "vreg_ambiguous": false, "defining_opcode": "load", "source_class": "memory_or_runtime", "rematerialization_eligible": false, "ir_position": 8, "ir_opcode": "add" }}
            \\      ]
            \\    }}
            \\  ]
            \\}}
        ,
            .{ AOT_VERSION, h, h },
        )
    else
        try std.fmt.allocPrint(
            allocator,
            \\{{
            \\  "schema": "wamr-aot-frame-attribution",
            \\  "schema_version": 2,
            \\  "cwasm_aot_version": {d},
            \\  "compiler_build_id": "test",
            \\  "architecture": "aarch64",
            \\  "abi": "aapcs64",
            \\  "module": 0,
            \\  "local_func": 10,
            \\  "function_name": "core_state_transition",
            \\  "module_text_size": 4,
            \\  "module_text_sha256": "{s}",
            \\  "function_offset": 0,
            \\  "code_size": 4,
            \\  "normalized_code_sha256": "{s}",
            \\  "direct_call_rel32_offsets": [],
            \\  "normalized_relocations": [],
            \\  "inline_data_ranges": [],
            \\  "frame_layout": {{
            \\    "frame_pointer": "x29",
            \\    "frame_size": 512,
            \\    "local_count": 1,
            \\    "param_count": 0,
            \\    "reserved_vmctx_offset": 16,
            \\    "locals_first_offset": 48,
            \\    "explicit_storage_first_offset": 264,
            \\    "explicit_storage_slots": 1,
            \\    "spill_base": 56,
            \\    "spill_stride": 8,
            \\    "spill_slots": 1
            \\  }},
            \\  "frame_regions": [
            \\    {{ "start": 0, "end": 16, "origin": "fixed_runtime_frame_state", "detail": "saved_fp_lr" }},
            \\    {{ "start": 16, "end": 24, "origin": "fixed_runtime_frame_state", "detail": "reserved_vmctx" }},
            \\    {{ "start": 48, "end": 56, "origin": "wasm_local_or_phi", "detail": "wasm_local_or_lowered_phi" }},
            \\    {{ "start": 56, "end": 64, "origin": "allocator_spill", "detail": "scalar_allocator_spills" }}
            \\  ],
            \\  "spill_metric": {{
            \\    "slots": 1, "spilled_vregs": 1, "scalar": 1, "v128": 0,
            \\    "slots_scalar": 1, "slots_v128": 0, "spill_ld": 1, "spill_st": 0, "remat": 0, "callee_saved": 0
            \\  }},
            \\  "emitted_allocator_loads": 1,
            \\  "emitted_allocator_stores": 0,
            \\  "allocator_values": [
            \\    {{ "vreg": 71, "frame_offset": 56, "slot": 0, "slot_count": 1, "value_type": "i64", "live_start": 4, "live_end": 12, "defining_opcode": "load", "source_class": "memory_or_runtime", "ir_use_count": 1, "ir_def_count": 1, "reload_count": 1, "store_count": 0, "rematerialization_eligible": false, "reused": false }}
            \\  ],
            \\  "accesses": [
            \\    {{
            \\      "native_start": 0, "native_end": 4, "kind": "load", "base": "x29", "frame_offset": 48, "width": 16,
            \\      "origin": "unknown", "detail": "mixed_pair_frame_access", "slot": 0, "local_index": null, "explicit_slot": null,
            \\      "vreg": null, "vreg_ambiguous": true, "defining_opcode": null, "source_class": null, "rematerialization_eligible": null,
            \\      "ir_position": 8, "ir_opcode": "add", "encoded_base": "x29", "encoded_offset": 48, "addressing_mode": "signed_pair",
            \\      "components": [
            \\        {{ "base": "x29", "frame_offset": 48, "width": 8, "data_register": 0, "data_register_class": "gpr", "origin": "wasm_local_or_phi", "detail": "wasm_local_or_lowered_phi", "slot": null, "local_index": 0, "explicit_slot": null, "vreg": null, "vreg_ambiguous": false, "defining_opcode": null, "source_class": null, "rematerialization_eligible": null, "ir_position": 8, "ir_opcode": "add" }},
            \\        {{ "base": "x29", "frame_offset": 56, "width": 8, "data_register": 1, "data_register_class": "gpr", "origin": "allocator_spill", "detail": "allocator_slot", "slot": 0, "local_index": null, "explicit_slot": null, "vreg": 71, "vreg_ambiguous": false, "defining_opcode": "load", "source_class": "memory_or_runtime", "rematerialization_eligible": false, "ir_position": 8, "ir_opcode": "add" }}
            \\      ]
            \\    }}
            \\  ]
            \\}}
        ,
            .{ AOT_VERSION, h, h },
        );
    return .{ .json = json, .code = code };
}

test "FrameAttributionUnitTests.test_aarch64_pair_samples_are_not_double_counted" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const pair = try makeAarch64PairMetadataJson(alloc, false);
    const meta = try loadFrameMetadataFromJson(alloc, "metadata.json", pair.json, 10, &pair.code, AOT_VERSION, &pair.code, 0, makeTestArtifactInfo(&pair.code, "aarch64"));
    const insts = [_]Instruction{.{ .address = 0x1000, .offset = 0, .size = 4, .text = "ldp x0, x1, [x29, #48]" }};
    var counts: std.AutoHashMapUnmanaged(u64, u64) = .empty;
    try counts.put(alloc, 0x1000, 19);

    const summary = try buildFrameSummary(alloc, &insts, &counts, &meta);
    try std.testing.expectEqual(@as(u64, 19), summary.originByName("allocator_spill").?.samples);
    var sum_contrib: u64 = 0;
    var paired_count: usize = 0;
    for (summary.allocator_contributors) |c| {
        sum_contrib += c.samples;
        if (c.paired_components != null) {
            paired_count += 1;
            try std.testing.expectEqualSlices(i64, &.{ 70, 71 }, c.candidate_vregs);
        }
    }
    try std.testing.expectEqual(@as(u64, 19), sum_contrib);
    try std.testing.expectEqual(@as(usize, 1), paired_count);
}

test "FrameAttributionUnitTests.test_mixed_pair_preserves_unranked_allocator_components" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const pair = try makeAarch64PairMetadataJson(alloc, true);
    const meta = try loadFrameMetadataFromJson(alloc, "metadata.json", pair.json, 10, &pair.code, AOT_VERSION, &pair.code, 0, makeTestArtifactInfo(&pair.code, "aarch64"));
    const insts = [_]Instruction{.{ .address = 0x1000, .offset = 0, .size = 4, .text = "ldp x0, x1, [x29, #48]" }};
    try validateMetadataDisassembly(alloc, &meta, &insts);
    var counts: std.AutoHashMapUnmanaged(u64, u64) = .empty;
    try counts.put(alloc, 0x1000, 5);

    const summary = try buildFrameSummary(alloc, &insts, &counts, &meta);
    try std.testing.expectEqual(@as(u64, 5), summary.originByName("unknown").?.samples);
    try std.testing.expectEqual(@as(usize, 0), summary.allocator_contributors.len);
    try std.testing.expectEqual(@as(u64, 1), summary.allocator_component_counts.total_loads);
    try std.testing.expectEqual(@as(u64, 0), summary.allocator_component_counts.total_stores);
    try std.testing.expectEqual(@as(u64, 1), summary.allocator_component_counts.unranked_loads);
    try std.testing.expectEqual(@as(u64, 0), summary.allocator_component_counts.unranked_stores);
    try std.testing.expectEqual(@as(i64, 1), summary.reconciliation.emitted_allocator_loads);
    try std.testing.expect(summary.reconciliation.matches);
}

test "FrameAttributionUnitTests.test_aarch64_materialized_frame_address_is_proven_from_native_code" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var code: [12]u8 = undefined;
    std.mem.writeInt(u32, code[0..4], 0xD2920010, .little); // movz x16, #0x9000
    std.mem.writeInt(u32, code[4..8], 0x8B1D0210, .little); // add x16, x16, x29
    std.mem.writeInt(u32, code[8..12], 0x3DC00200, .little); // ldr q0, [x16]

    const h = try sha256Hex(alloc, &code);
    const json = try std.fmt.allocPrint(
        alloc,
        \\{{
        \\  "schema": "wamr-aot-frame-attribution",
        \\  "schema_version": 2,
        \\  "cwasm_aot_version": {d},
        \\  "compiler_build_id": "test",
        \\  "architecture": "aarch64",
        \\  "abi": "aapcs64",
        \\  "module": 0,
        \\  "local_func": 10,
        \\  "function_name": "core_state_transition",
        \\  "module_text_size": 12,
        \\  "module_text_sha256": "{s}",
        \\  "function_offset": 0,
        \\  "code_size": 12,
        \\  "normalized_code_sha256": "{s}",
        \\  "direct_call_rel32_offsets": [],
        \\  "normalized_relocations": [],
        \\  "inline_data_ranges": [],
        \\  "frame_layout": {{
        \\    "frame_pointer": "x29",
        \\    "frame_size": 40960,
        \\    "local_count": 1,
        \\    "param_count": 0,
        \\    "reserved_vmctx_offset": 16,
        \\    "locals_first_offset": 24,
        \\    "explicit_storage_first_offset": 264,
        \\    "explicit_storage_slots": 1,
        \\    "spill_base": 36864,
        \\    "spill_stride": 8,
        \\    "spill_slots": 2
        \\  }},
        \\  "frame_regions": [
        \\    {{ "start": 0, "end": 16, "origin": "fixed_runtime_frame_state", "detail": "saved_fp_lr" }},
        \\    {{ "start": 16, "end": 24, "origin": "fixed_runtime_frame_state", "detail": "reserved_vmctx" }},
        \\    {{ "start": 24, "end": 32, "origin": "wasm_local_or_phi", "detail": "wasm_local_or_lowered_phi" }},
        \\    {{ "start": 36864, "end": 36880, "origin": "allocator_spill", "detail": "scalar_allocator_spills" }}
        \\  ],
        \\  "spill_metric": {{
        \\    "slots": 2, "spilled_vregs": 1, "scalar": 0, "v128": 1,
        \\    "slots_scalar": 0, "slots_v128": 2, "spill_ld": 1, "spill_st": 0, "remat": 0, "callee_saved": 0
        \\  }},
        \\  "emitted_allocator_loads": 1,
        \\  "emitted_allocator_stores": 0,
        \\  "allocator_values": [
        \\    {{ "vreg": 77, "frame_offset": 36864, "slot": 0, "slot_count": 2, "value_type": "v128", "live_start": 4, "live_end": 12, "defining_opcode": "load", "source_class": "memory_or_runtime", "ir_use_count": 1, "ir_def_count": 1, "reload_count": 1, "store_count": 0, "rematerialization_eligible": false, "reused": false }}
        \\  ],
        \\  "accesses": [
        \\    {{
        \\      "native_start": 8, "native_end": 12, "kind": "load", "base": "x29", "frame_offset": 36864, "width": 16,
        \\      "origin": "allocator_spill", "detail": "allocator_slot", "slot": 0, "local_index": null, "explicit_slot": null,
        \\      "vreg": 77, "vreg_ambiguous": false, "defining_opcode": "load", "source_class": "memory_or_runtime", "rematerialization_eligible": false,
        \\      "ir_position": 8, "ir_opcode": "add", "encoded_base": "x16", "encoded_offset": 0, "addressing_mode": "materialized",
        \\      "components": [
        \\        {{ "base": "x29", "frame_offset": 36864, "width": 16, "data_register": 0, "data_register_class": "simd", "origin": "allocator_spill", "detail": "allocator_slot", "slot": 0, "local_index": null, "explicit_slot": null, "vreg": 77, "vreg_ambiguous": false, "defining_opcode": "load", "source_class": "memory_or_runtime", "rematerialization_eligible": false, "ir_position": 8, "ir_opcode": "add" }}
        \\      ]
        \\    }}
        \\  ]
        \\}}
    ,
        .{ AOT_VERSION, h, h },
    );
    _ = try loadFrameMetadataFromJson(alloc, "metadata.json", json, 10, &code, AOT_VERSION, &code, 0, makeTestArtifactInfo(&code, "aarch64"));

    var corrupt = code;
    std.mem.writeInt(u32, corrupt[4..8], 0x8B1D0231, .little);
    const ch = try sha256Hex(alloc, &corrupt);
    const corrupt_json = try replaceAllAlloc(alloc, json, h, ch);
    try std.testing.expectError(
        error.AttributionFailed,
        loadFrameMetadataFromJson(alloc, "metadata.json", corrupt_json, 10, &corrupt, AOT_VERSION, &corrupt, 0, makeTestArtifactInfo(&corrupt, "aarch64")),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "frame-access offsets disagree") != null);
}

test "FrameAttributionUnitTests.test_perf_binary_override_is_honored" {
    try std.testing.expect(perfBinary().len > 0);
}

test "FrameAttributionUnitTests.test_aarch64_non_frame_x16_access_does_not_fail_disassembly_validation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const meta = FrameMetadata{ .architecture = "aarch64", .schema_version = 2 };
    const insts = [_]Instruction{
        .{ .address = 0, .offset = 0, .size = 4, .text = "add x16, x20, x0" },
        .{ .address = 4, .offset = 4, .size = 4, .text = "ldr x1, [x16]" },
    };
    try validateMetadataDisassembly(alloc, &meta, &insts);
}

test "FrameAttributionUnitTests.test_aarch64_control_flow_clears_materialized_frame_aliases" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var code: [20]u8 = undefined;
    std.mem.writeInt(u32, code[0..4], 0xD2920010, .little); // movz x16, #0x9000
    std.mem.writeInt(u32, code[4..8], 0x8B1D0210, .little); // add x16, x16, x29
    std.mem.writeInt(u32, code[8..12], 0x14000002, .little); // b +8
    std.mem.writeInt(u32, code[12..16], 0x8B000290, .little); // add x16, x20, x0
    std.mem.writeInt(u32, code[16..20], 0xF9400201, .little); // ldr x1, [x16]

    const decoded = try decodeAarch64FrameAccesses(alloc, &code, &.{});
    try std.testing.expectEqual(@as(usize, 0), decoded.len);

    const meta = FrameMetadata{ .architecture = "aarch64", .schema_version = 2 };
    const insts = [_]Instruction{
        .{ .address = 0, .offset = 0, .size = 4, .text = "movz x16, #0x9000" },
        .{ .address = 4, .offset = 4, .size = 4, .text = "add x16, x16, x29" },
        .{ .address = 8, .offset = 8, .size = 4, .text = "b 0x10" },
        .{ .address = 12, .offset = 12, .size = 4, .text = "add x16, x20, x0" },
        .{ .address = 16, .offset = 16, .size = 4, .text = "ldr x1, [x16]" },
    };
    try validateMetadataDisassembly(alloc, &meta, &insts);
}

test "FrameAttributionUnitTests.test_aarch64_compiler_like_frame_access_fails_when_omitted_from_metadata" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const base_arm = try makeDefaultAarch64MetadataJson(alloc, &TEST_ARM_CODE);
    const needle = "\"emitted_allocator_stores\": 1,";
    const idx = std.mem.indexOf(u8, base_arm, needle).?;
    const prefix = base_arm[0..idx];
    const omitted = try std.fmt.allocPrint(
        alloc,
        "{s}\"emitted_allocator_stores\": 0,\n  \"allocator_values\": [\n    {{\"vreg\": 77, \"frame_offset\": 248, \"slot\": 0, \"slot_count\": 1, \"value_type\": \"i64\", \"live_start\": 4, \"live_end\": 12, \"defining_opcode\": \"load\", \"source_class\": \"memory_or_runtime\", \"ir_use_count\": 2, \"ir_def_count\": 1, \"reload_count\": 1, \"store_count\": 0, \"rematerialization_eligible\": false, \"reused\": false}}\n  ],\n  \"accesses\": [\n    {{\"native_start\": 4, \"native_end\": 8, \"kind\": \"load\", \"base\": \"x29\", \"frame_offset\": 248, \"width\": 8, \"origin\": \"allocator_spill\", \"detail\": \"allocator_slot\", \"slot\": 0, \"local_index\": null, \"explicit_slot\": null, \"vreg\": 77, \"vreg_ambiguous\": false, \"defining_opcode\": \"load\", \"source_class\": \"memory_or_runtime\", \"rematerialization_eligible\": false, \"ir_position\": 8, \"ir_opcode\": \"add\", \"encoded_base\": \"x29\", \"encoded_offset\": 248, \"addressing_mode\": \"unsigned_scaled\", \"components\": [{{\"base\": \"x29\", \"frame_offset\": 248, \"width\": 8, \"data_register\": 16, \"data_register_class\": \"gpr\", \"origin\": \"allocator_spill\", \"detail\": \"allocator_slot\", \"slot\": 0, \"local_index\": null, \"explicit_slot\": null, \"vreg\": 77, \"vreg_ambiguous\": false, \"defining_opcode\": \"load\", \"source_class\": \"memory_or_runtime\", \"rematerialization_eligible\": false, \"ir_position\": 8, \"ir_opcode\": \"add\"}}]}}\n  ]\n}}",
        .{prefix},
    );
    const mod_spill = try replaceFirst(alloc, omitted, "\"spill_st\": 1", "\"spill_st\": 0");
    try std.testing.expectError(
        error.AttributionFailed,
        loadFrameMetadataFromJson(alloc, "metadata.json", mod_spill, 10, &TEST_ARM_CODE, AOT_VERSION, &TEST_ARM_CODE, 0, makeTestArtifactInfo(&TEST_ARM_CODE, "aarch64")),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastError(), "frame-access offsets disagree") != null);
}
