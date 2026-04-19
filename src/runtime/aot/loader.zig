//! AOT binary format loader.
//!
//! Parses an AOT (Ahead-Of-Time) compiled binary into an AotModule structure.
//! The AOT binary format is WAMR-specific and contains pre-compiled native
//! code along with metadata sections for linking and instantiation.
//!
//! Binary layout:
//!   [4 bytes] magic (0x746f6100 = "\0aot" little-endian)
//!   [4 bytes] version (6)
//!   [sections...] each: [4 bytes type] [4 bytes size] [payload]

const std = @import("std");
const types = @import("../common/types.zig");

// ─── AOT section types ──────────────────────────────────────────────────────

pub const AotSectionType = enum(u32) {
    target_info = 0,
    init_data = 1,
    text = 2,
    function = 3,
    @"export" = 4,
    data = 5,
    relocation = 6,
    name = 7,
    @"import" = 8,
    memory = 9,
    global = 10,
    elem = 11,
    start = 12,
    _,
};

// ─── Target info ────────────────────────────────────────────────────────────

pub const TargetInfo = struct {
    bin_type: u16, // 0=ELF, 1=AOT, 2=PE
    abi_type: u16,
    e_type: u16,
    e_machine: u16,
    e_flags: u32,
    reserved: u32,
    arch: [16]u8,
    features: u64,
};

// ─── AOT module ─────────────────────────────────────────────────────────────

/// A data segment parsed from the AOT data section.
pub const AotDataSegment = struct {
    memory_idx: u32,
    offset: u32,
    data: []const u8,
};

/// An import descriptor parsed from the AOT import section.
pub const AotImportDesc = struct {
    module_name: []const u8,
    field_name: []const u8,
    kind: types.ExternalKind,
    func_type_idx: u32,
};

/// A global initial value parsed from the AOT globals section.
pub const AotGlobalInit = struct {
    val_type: u8,
    mutability: u8,
    init_i64: i64,
};

/// An element segment parsed from the AOT element section.
pub const AotElemSegment = struct {
    table_idx: u32,
    offset: u32,
    func_indices: []const u32,
    /// If true, segment is only referenced by `table.init` / `elem.drop`
    /// (no active initializer at instantiation). For active segments,
    /// `offset` is the static table offset; for passive segments
    /// `offset` is 0 and unused.
    is_passive: bool = false,
};

pub const AotModule = struct {
    target_info: ?TargetInfo = null,
    text_section: ?[]const u8 = null,
    func_offsets: []const u32 = &.{},
    func_count: u32 = 0,
    exports: []const types.ExportDesc = &.{},
    import_function_count: u32 = 0,
    imports: []const AotImportDesc = &.{},
    memories: []const types.MemoryType = &.{},
    tables: []const types.TableType = &.{},
    data_segments: []const AotDataSegment = &.{},
    global_inits: []const AotGlobalInit = &.{},
    elem_segments: []const AotElemSegment = &.{},
    start_function: ?u32 = null,

    /// Find an export by name and kind.
    pub fn findExport(self: *const AotModule, name: []const u8, kind: types.ExternalKind) ?types.ExportDesc {
        for (self.exports) |exp| {
            if (exp.kind == kind and std.mem.eql(u8, exp.name, name)) return exp;
        }
        return null;
    }
};

// ─── Errors ─────────────────────────────────────────────────────────────────

pub const LoadError = error{
    InvalidMagic,
    InvalidVersion,
    InvalidSection,
    UnexpectedEnd,
    UnsupportedTarget,
    OutOfMemory,
};

// ─── Binary reader ──────────────────────────────────────────────────────────

const BinaryReader = struct {
    data: []const u8,
    pos: usize = 0,

    fn remaining(self: *const BinaryReader) usize {
        return self.data.len - self.pos;
    }

    fn readByte(self: *BinaryReader) LoadError!u8 {
        if (self.pos >= self.data.len) return error.UnexpectedEnd;
        const b = self.data[self.pos];
        self.pos += 1;
        return b;
    }

    fn readBytes(self: *BinaryReader, n: usize) LoadError![]const u8 {
        if (self.pos + n > self.data.len) return error.UnexpectedEnd;
        const slice = self.data[self.pos .. self.pos + n];
        self.pos += n;
        return slice;
    }

    fn readU32Le(self: *BinaryReader) LoadError!u32 {
        const bytes = try self.readBytes(4);
        return std.mem.readInt(u32, bytes[0..4], .little);
    }

    fn readU16Le(self: *BinaryReader) LoadError!u16 {
        const bytes = try self.readBytes(2);
        return std.mem.readInt(u16, bytes[0..2], .little);
    }

    fn readU64Le(self: *BinaryReader) LoadError!u64 {
        const bytes = try self.readBytes(8);
        return std.mem.readInt(u64, bytes[0..8], .little);
    }
};

// ─── Public API ─────────────────────────────────────────────────────────────

/// Load an AOT binary from raw bytes into an AotModule.
pub fn load(data: []const u8, allocator: std.mem.Allocator) LoadError!AotModule {
    var reader = BinaryReader{ .data = data };

    // Validate magic number
    const magic = try reader.readU32Le();
    if (magic != types.aot_magic) return error.InvalidMagic;

    // Validate version
    const version = try reader.readU32Le();
    if (version != types.aot_version) return error.InvalidVersion;

    var module = AotModule{};

    // Parse sections
    while (reader.remaining() >= 8) {
        const section_type_raw = try reader.readU32Le();
        const section_size = try reader.readU32Le();

        if (reader.remaining() < section_size) return error.UnexpectedEnd;

        const section_start = reader.pos;
        const section_type: AotSectionType = @enumFromInt(section_type_raw);

        switch (section_type) {
            .target_info => {
                module.target_info = try parseTargetInfo(&reader);
            },
            .text => {
                module.text_section = try reader.readBytes(section_size);
            },
            .function => {
                try parseFunctionSection(&reader, section_size, &module, allocator);
            },
            .@"export" => {
                try parseExportSection(&reader, section_size, &module, allocator);
            },
            .data => {
                try parseDataSection(&reader, section_size, &module, allocator);
            },
            .@"import" => {
                try parseImportSection(&reader, section_size, &module, allocator);
            },
            .memory => {
                try parseMemorySection(&reader, section_size, &module, allocator);
            },
            .global => {
                try parseGlobalSection(&reader, section_size, &module, allocator);
            },
            .elem => {
                try parseElemSection(&reader, section_size, &module, allocator);
            },
            .start => {
                if (section_size >= 4) {
                    module.start_function = try reader.readU32Le();
                }
            },
            else => {
                // Skip unknown/unhandled sections
                reader.pos += section_size;
            },
        }

        // Ensure we consumed exactly section_size bytes
        reader.pos = section_start + section_size;
    }

    return module;
}

/// Free all allocations owned by an AotModule.
pub fn unload(module: *const AotModule, allocator: std.mem.Allocator) void {
    if (module.func_offsets.len > 0) {
        allocator.free(module.func_offsets);
    }
    for (module.exports) |exp| {
        if (exp.name.len > 0) {
            allocator.free(exp.name);
        }
    }
    if (module.exports.len > 0) {
        allocator.free(module.exports);
    }
    if (module.memories.len > 0) {
        allocator.free(module.memories);
    }
    for (module.data_segments) |seg| {
        if (seg.data.len > 0) {
            allocator.free(seg.data);
        }
    }
    if (module.data_segments.len > 0) {
        allocator.free(module.data_segments);
    }
    if (module.tables.len > 0) {
        allocator.free(module.tables);
    }
    for (module.imports) |imp| {
        if (imp.module_name.len > 0) allocator.free(imp.module_name);
        if (imp.field_name.len > 0) allocator.free(imp.field_name);
    }
    if (module.imports.len > 0) {
        allocator.free(module.imports);
    }
    if (module.global_inits.len > 0) {
        allocator.free(module.global_inits);
    }
    for (module.elem_segments) |seg| {
        if (seg.func_indices.len > 0) allocator.free(seg.func_indices);
    }
    if (module.elem_segments.len > 0) {
        allocator.free(module.elem_segments);
    }
}

// ─── Section parsers ────────────────────────────────────────────────────────

fn parseTargetInfo(reader: *BinaryReader) LoadError!TargetInfo {
    var info: TargetInfo = undefined;
    info.bin_type = try reader.readU16Le();
    info.abi_type = try reader.readU16Le();
    info.e_type = try reader.readU16Le();
    info.e_machine = try reader.readU16Le();
    info.e_flags = try reader.readU32Le();
    info.reserved = try reader.readU32Le();
    const arch_bytes = try reader.readBytes(16);
    @memcpy(&info.arch, arch_bytes);
    info.features = try reader.readU64Le();
    return info;
}

fn parseFunctionSection(reader: *BinaryReader, section_size: u32, module: *AotModule, allocator: std.mem.Allocator) LoadError!void {
    if (section_size < 4) return error.InvalidSection;
    const count = try reader.readU32Le();
    if (count == 0) {
        module.func_count = 0;
        return;
    }
    const needed = @as(usize, count) * 4;
    if (reader.remaining() < needed) return error.UnexpectedEnd;

    const offsets = allocator.alloc(u32, count) catch return error.OutOfMemory;
    for (0..count) |i| {
        offsets[i] = try reader.readU32Le();
    }
    module.func_offsets = offsets;
    module.func_count = count;
}

fn parseExportSection(reader: *BinaryReader, section_size: u32, module: *AotModule, allocator: std.mem.Allocator) LoadError!void {
    if (section_size < 4) return error.InvalidSection;
    const count = try reader.readU32Le();
    if (count == 0) return;

    const exports = allocator.alloc(types.ExportDesc, count) catch return error.OutOfMemory;
    errdefer allocator.free(exports);

    for (0..count) |i| {
        const name_len = try reader.readU32Le();
        const name_bytes = try reader.readBytes(name_len);
        const name_copy = allocator.alloc(u8, name_len) catch return error.OutOfMemory;
        @memcpy(name_copy, name_bytes);

        const kind_raw = try reader.readByte();
        const index = try reader.readU32Le();

        exports[i] = .{
            .name = name_copy,
            .kind = @enumFromInt(kind_raw),
            .index = index,
        };
    }
    module.exports = exports;
}

fn parseDataSection(reader: *BinaryReader, section_size: u32, module: *AotModule, allocator: std.mem.Allocator) LoadError!void {
    if (section_size < 4) return error.InvalidSection;
    const count = try reader.readU32Le();
    if (count == 0) return;

    const segments = allocator.alloc(AotDataSegment, count) catch return error.OutOfMemory;
    var initialized: usize = 0;
    errdefer {
        for (0..initialized) |i| {
            if (segments[i].data.len > 0) allocator.free(segments[i].data);
        }
        allocator.free(segments);
    }

    for (0..count) |i| {
        const memory_idx = try reader.readU32Le();
        const offset = try reader.readU32Le();
        const data_len = try reader.readU32Le();
        const data_bytes = try reader.readBytes(data_len);
        const data_copy = allocator.alloc(u8, data_len) catch return error.OutOfMemory;
        @memcpy(data_copy, data_bytes);

        segments[i] = .{
            .memory_idx = memory_idx,
            .offset = offset,
            .data = data_copy,
        };
        initialized += 1;
    }
    module.data_segments = segments;
}

fn parseImportSection(reader: *BinaryReader, section_size: u32, module: *AotModule, allocator: std.mem.Allocator) LoadError!void {
    if (section_size < 4) return error.InvalidSection;
    const count = try reader.readU32Le();
    if (count == 0) return;

    const import_descs = allocator.alloc(AotImportDesc, count) catch return error.OutOfMemory;
    var initialized: usize = 0;
    errdefer {
        for (0..initialized) |i| {
            if (import_descs[i].module_name.len > 0) allocator.free(import_descs[i].module_name);
            if (import_descs[i].field_name.len > 0) allocator.free(import_descs[i].field_name);
        }
        allocator.free(import_descs);
    }

    var func_count: u32 = 0;
    for (0..count) |i| {
        const mod_name_len = try reader.readU32Le();
        const mod_name_bytes = try reader.readBytes(mod_name_len);
        const mod_name = allocator.alloc(u8, mod_name_len) catch return error.OutOfMemory;
        @memcpy(mod_name, mod_name_bytes);

        const field_name_len = try reader.readU32Le();
        const field_name_bytes = try reader.readBytes(field_name_len);
        const field_name = allocator.alloc(u8, field_name_len) catch return error.OutOfMemory;
        @memcpy(field_name, field_name_bytes);

        const kind_raw = try reader.readByte();
        const func_type_idx = try reader.readU32Le();
        const kind: types.ExternalKind = @enumFromInt(kind_raw);

        if (kind == .function) func_count += 1;

        import_descs[i] = .{
            .module_name = mod_name,
            .field_name = field_name,
            .kind = kind,
            .func_type_idx = func_type_idx,
        };
        initialized += 1;
    }

    module.imports = import_descs;
    module.import_function_count = func_count;
}

fn parseMemorySection(reader: *BinaryReader, section_size: u32, module: *AotModule, allocator: std.mem.Allocator) LoadError!void {
    if (section_size < 4) return error.InvalidSection;
    const count = try reader.readU32Le();
    if (count == 0) return;

    const mem_types = allocator.alloc(types.MemoryType, count) catch return error.OutOfMemory;
    errdefer allocator.free(mem_types);

    for (0..count) |i| {
        const min_pages = try reader.readU32Le();
        const has_max = try reader.readByte();
        const max_pages: ?u64 = if (has_max != 0) @as(u64, try reader.readU32Le()) else null;
        mem_types[i] = .{
            .limits = .{ .min = min_pages, .max = max_pages },
        };
    }

    module.memories = mem_types;
}

fn parseGlobalSection(reader: *BinaryReader, section_size: u32, module: *AotModule, allocator: std.mem.Allocator) LoadError!void {
    if (section_size < 4) return error.InvalidSection;
    const count = try reader.readU32Le();
    if (count == 0) return;

    const inits = allocator.alloc(AotGlobalInit, count) catch return error.OutOfMemory;
    errdefer allocator.free(inits);

    for (0..count) |i| {
        const val_type = try reader.readByte();
        const mutability = try reader.readByte();
        const init_bytes = try reader.readBytes(8);
        const init_i64 = std.mem.readInt(i64, init_bytes[0..8], .little);
        inits[i] = .{
            .val_type = val_type,
            .mutability = mutability,
            .init_i64 = init_i64,
        };
    }

    module.global_inits = inits;
}

fn parseElemSection(reader: *BinaryReader, section_size: u32, module: *AotModule, allocator: std.mem.Allocator) LoadError!void {
    if (section_size < 4) return error.InvalidSection;
    const count = try reader.readU32Le();
    if (count == 0) return;

    const segs = allocator.alloc(AotElemSegment, count) catch return error.OutOfMemory;
    var initialized: usize = 0;
    errdefer {
        for (0..initialized) |i| {
            if (segs[i].func_indices.len > 0) allocator.free(segs[i].func_indices);
        }
        allocator.free(segs);
    }

    for (0..count) |i| {
        const flag = try reader.readByte();
        const table_idx = try reader.readU32Le();
        const offset = try reader.readU32Le();
        const n_funcs = try reader.readU32Le();
        const indices = allocator.alloc(u32, n_funcs) catch return error.OutOfMemory;
        for (0..n_funcs) |j| {
            indices[j] = try reader.readU32Le();
        }
        segs[i] = .{
            .table_idx = table_idx,
            .offset = offset,
            .func_indices = indices,
            .is_passive = flag != 0,
        };
        initialized += 1;
    }

    module.elem_segments = segs;
}

// ─── Tests ──────────────────────────────────────────────────────────────────

fn writeU32Le(buf: []u8, offset: usize, val: u32) void {
    @memcpy(buf[offset..][0..4], &std.mem.toBytes(std.mem.nativeToLittle(u32, val)));
}

fn writeU16Le(buf: []u8, offset: usize, val: u16) void {
    @memcpy(buf[offset..][0..2], &std.mem.toBytes(std.mem.nativeToLittle(u16, val)));
}

fn writeU64Le(buf: []u8, offset: usize, val: u64) void {
    @memcpy(buf[offset..][0..8], &std.mem.toBytes(std.mem.nativeToLittle(u64, val)));
}

/// Build a minimal AOT binary: header + target_info section (56 bytes total).
fn buildMinimalAot() [56]u8 {
    var buf = [_]u8{0} ** 56;
    // Header: magic(4) + version(4) = 8 bytes
    writeU32Le(&buf, 0, types.aot_magic);
    writeU32Le(&buf, 4, types.aot_version);
    // Target info section header: type(4) + size(4) = 8 bytes
    writeU32Le(&buf, 8, 0); // section type = target_info
    writeU32Le(&buf, 12, 40); // section payload size
    // TargetInfo: bin_type(2) + abi_type(2) + e_type(2) + e_machine(2) +
    //   e_flags(4) + reserved(4) + arch(16) + features(8) = 40 bytes
    writeU16Le(&buf, 16, 1); // bin_type=AOT
    writeU16Le(&buf, 18, 0); // abi_type
    writeU16Le(&buf, 20, 2); // e_type
    writeU16Le(&buf, 22, 0x3E); // e_machine (x86_64)
    writeU32Le(&buf, 24, 0); // e_flags
    writeU32Le(&buf, 28, 0); // reserved
    @memcpy(buf[32..48], "x86_64\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
    writeU64Le(&buf, 48, 0); // features
    return buf;
}

test "load: invalid magic returns error" {
    var bad_magic = [_]u8{ 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00 };
    const result = load(&bad_magic, std.testing.allocator);
    try std.testing.expectError(error.InvalidMagic, result);
}

test "load: invalid version returns error" {
    var bad_version: [8]u8 = undefined;
    std.mem.writeInt(u32, bad_version[0..4], types.aot_magic, .little);
    std.mem.writeInt(u32, bad_version[4..8], 99, .little); // wrong version
    const result = load(&bad_version, std.testing.allocator);
    try std.testing.expectError(error.InvalidVersion, result);
}

test "load: valid minimal AOT binary parses target info" {
    const data = buildMinimalAot();

    const module = try load(&data, std.testing.allocator);
    defer unload(&module, std.testing.allocator);

    try std.testing.expect(module.target_info != null);
    const ti = module.target_info.?;
    try std.testing.expectEqual(@as(u16, 1), ti.bin_type);
    try std.testing.expectEqual(@as(u16, 0x3E), ti.e_machine);
    try std.testing.expect(std.mem.startsWith(u8, &ti.arch, "x86_64"));
}

test "load: truncated input returns UnexpectedEnd" {
    // Only magic, no version
    var short: [4]u8 = undefined;
    std.mem.writeInt(u32, short[0..4], types.aot_magic, .little);
    const result = load(&short, std.testing.allocator);
    try std.testing.expectError(error.UnexpectedEnd, result);
}

test "unload: frees without leaks on empty module" {
    const module = AotModule{};
    unload(&module, std.testing.allocator);
}

test "load: function section with offsets" {
    // Header(8) + section header(8) + count(4) + 3 offsets(12) = 32
    var data = [_]u8{0} ** 32;
    writeU32Le(&data, 0, types.aot_magic);
    writeU32Le(&data, 4, types.aot_version);
    writeU32Le(&data, 8, 3); // section type = function
    writeU32Le(&data, 12, 16); // section size: count(4) + 3*offset(12) = 16
    writeU32Le(&data, 16, 3); // count
    writeU32Le(&data, 20, 0); // offset 0
    writeU32Le(&data, 24, 64); // offset 1
    writeU32Le(&data, 28, 128); // offset 2

    const allocator = std.testing.allocator;
    const module = try load(&data, allocator);
    defer unload(&module, allocator);

    try std.testing.expectEqual(@as(u32, 3), module.func_count);
    try std.testing.expectEqual(@as(u32, 0), module.func_offsets[0]);
    try std.testing.expectEqual(@as(u32, 64), module.func_offsets[1]);
    try std.testing.expectEqual(@as(u32, 128), module.func_offsets[2]);
}

test "load: export section round-trip" {
    // Header(8) + section header(8) + count(4) + name_len(4) + "main"(4) + kind(1) + index(4) = 33
    var data = [_]u8{0} ** 33;
    writeU32Le(&data, 0, types.aot_magic);
    writeU32Le(&data, 4, types.aot_version);
    writeU32Le(&data, 8, 4); // section type = export
    writeU32Le(&data, 12, 17); // section size: count(4) + name_len(4) + "main"(4) + kind(1) + index(4) = 17
    writeU32Le(&data, 16, 1); // count
    writeU32Le(&data, 20, 4); // name_len
    @memcpy(data[24..28], "main");
    data[28] = 0x00; // kind = function
    writeU32Le(&data, 29, 42); // index

    const allocator = std.testing.allocator;
    const module = try load(&data, allocator);
    defer unload(&module, allocator);

    try std.testing.expectEqual(@as(usize, 1), module.exports.len);
    try std.testing.expect(std.mem.eql(u8, module.exports[0].name, "main"));
    try std.testing.expectEqual(types.ExternalKind.function, module.exports[0].kind);
    try std.testing.expectEqual(@as(u32, 42), module.exports[0].index);
}

test "AotModule: findExport" {
    const exports = [_]types.ExportDesc{
        .{ .name = "memory", .kind = .memory, .index = 0 },
        .{ .name = "run", .kind = .function, .index = 5 },
    };
    const module = AotModule{ .exports = &exports };

    const found = module.findExport("run", .function);
    try std.testing.expect(found != null);
    try std.testing.expectEqual(@as(u32, 5), found.?.index);

    // Wrong kind
    try std.testing.expectEqual(@as(?types.ExportDesc, null), module.findExport("run", .table));
    // Missing name
    try std.testing.expectEqual(@as(?types.ExportDesc, null), module.findExport("missing", .function));
}
