//! Component Model binary format loader.
//!
//! Parses a WebAssembly Component binary into the in-memory AST defined
//! in `types.zig`. Components use the same magic bytes as core modules
//! but a different layer field (0x01) and version (0x0d).
//!
//! Component sections can be interleaved (unlike core module sections which
//! have a strict ordering). Each section's definitions are appended to the
//! appropriate index space.

const std = @import("std");
const ctypes = @import("types.zig");
const core_types = @import("../runtime/common/types.zig");
const leb128_mod = @import("../shared/utils/leb128.zig");

pub const LoadError = error{
    InvalidMagic,
    InvalidVersion,
    UnexpectedEnd,
    InvalidSectionId,
    InvalidSectionSize,
    InvalidEncoding,
    UnsupportedFeature,
    OutOfMemory,
    Overflow,
    InvalidUtf8,
};

/// A streaming reader over the component binary.
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

    fn readU32(self: *BinaryReader) LoadError!u32 {
        const slice = self.data[self.pos..];
        const result = leb128_mod.readUnsigned(u32, slice) catch return error.UnexpectedEnd;
        self.pos += result.bytes_read;
        return result.value;
    }

    fn readFixedU32(self: *BinaryReader) LoadError!u32 {
        if (self.pos + 4 > self.data.len) return error.UnexpectedEnd;
        const val = std.mem.readInt(u32, self.data[self.pos..][0..4], .little);
        self.pos += 4;
        return val;
    }

    fn readName(self: *BinaryReader) LoadError![]const u8 {
        const len = try self.readU32();
        if (self.pos + len > self.data.len) return error.UnexpectedEnd;
        const name = self.data[self.pos .. self.pos + len];
        self.pos += len;
        // Validate UTF-8
        if (!std.unicode.utf8ValidateSlice(name)) return error.InvalidUtf8;
        return name;
    }
};

/// Component section IDs per the binary format spec.
const SectionId = enum(u8) {
    custom = 0,
    core_module = 1,
    core_instance = 2,
    core_type = 3,
    component = 4,
    instance = 5,
    alias = 6,
    type = 7,
    canon = 8,
    start = 9,
    @"import" = 10,
    @"export" = 11,
    value = 12,
};

/// Load a WebAssembly Component from binary data.
pub fn load(data: []const u8, allocator: std.mem.Allocator) LoadError!ctypes.Component {
    var reader = BinaryReader{ .data = data };

    // Validate preamble
    const magic = try reader.readFixedU32();
    if (magic != core_types.wasm_magic) return error.InvalidMagic;
    const version = try reader.readFixedU32();
    if (version != core_types.component_version) return error.InvalidVersion;

    // Collect sections into dynamic arrays
    var core_modules: std.ArrayListUnmanaged(ctypes.CoreModule) = .{};
    var core_instances: std.ArrayListUnmanaged(ctypes.CoreInstanceExpr) = .{};
    var core_type_defs: std.ArrayListUnmanaged(ctypes.CoreTypeDef) = .{};
    var components: std.ArrayListUnmanaged(*ctypes.Component) = .{};
    var instances: std.ArrayListUnmanaged(ctypes.InstanceExpr) = .{};
    var aliases: std.ArrayListUnmanaged(ctypes.Alias) = .{};
    var type_defs: std.ArrayListUnmanaged(ctypes.TypeDef) = .{};
    var canons: std.ArrayListUnmanaged(ctypes.Canon) = .{};
    var imports: std.ArrayListUnmanaged(ctypes.ImportDecl) = .{};
    var exports: std.ArrayListUnmanaged(ctypes.ExportDecl) = .{};
    var start: ?ctypes.Start = null;

    while (reader.remaining() > 0) {
        const section_id_byte = try reader.readByte();
        const section_size = try reader.readU32();

        const section_start = reader.pos;
        if (section_start + section_size > reader.data.len) return error.InvalidSectionSize;

        const section_id = std.meta.intToEnum(SectionId, section_id_byte) catch
            return error.InvalidSectionId;

        switch (section_id) {
            .custom => {
                // Skip custom sections
                reader.pos = section_start + section_size;
            },
            .core_module => {
                // The core module is stored as raw bytes (nested module binary)
                const module_data = reader.data[section_start .. section_start + section_size];
                try core_modules.append(allocator, .{ .data = module_data });
                reader.pos = section_start + section_size;
            },
            .core_instance => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    try core_instances.append(allocator, try parseCoreInstance(&reader, allocator));
                }
            },
            .core_type => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    try core_type_defs.append(allocator, try parseCoreType(&reader, allocator));
                }
            },
            .component => {
                // Nested component — recursively parse
                const comp_data = reader.data[section_start .. section_start + section_size];
                const child = try allocator.create(ctypes.Component);
                child.* = try load(comp_data, allocator);
                try components.append(allocator, child);
                reader.pos = section_start + section_size;
            },
            .instance => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    try instances.append(allocator, try parseInstance(&reader, allocator));
                }
            },
            .alias => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    try aliases.append(allocator, try parseAlias(&reader));
                }
            },
            .type => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    try type_defs.append(allocator, try parseTypeDef(&reader, allocator));
                }
            },
            .canon => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    try canons.append(allocator, try parseCanon(&reader, allocator));
                }
            },
            .start => {
                start = try parseStart(&reader, allocator);
            },
            .@"import" => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    try imports.append(allocator, try parseImport(&reader));
                }
            },
            .@"export" => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    try exports.append(allocator, try parseExport(&reader));
                }
            },
            .value => {
                // Value definitions — skip for now (gated feature)
                reader.pos = section_start + section_size;
            },
        }
    }

    return .{
        .core_modules = try core_modules.toOwnedSlice(allocator),
        .core_instances = try core_instances.toOwnedSlice(allocator),
        .core_types = try core_type_defs.toOwnedSlice(allocator),
        .components = try components.toOwnedSlice(allocator),
        .instances = try instances.toOwnedSlice(allocator),
        .aliases = try aliases.toOwnedSlice(allocator),
        .types = try type_defs.toOwnedSlice(allocator),
        .canons = try canons.toOwnedSlice(allocator),
        .start = start,
        .imports = try imports.toOwnedSlice(allocator),
        .exports = try exports.toOwnedSlice(allocator),
    };
}

// ── Section parsers ─────────────────────────────────────────────────────────

fn parseCoreInstance(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.CoreInstanceExpr {
    const tag = try reader.readByte();
    switch (tag) {
        0x00 => {
            const module_idx = try reader.readU32();
            const arg_count = try reader.readU32();
            const args = try allocator.alloc(ctypes.CoreInstantiateArg, arg_count);
            for (args) |*arg| {
                arg.name = try reader.readName();
                const sort_byte = try reader.readByte();
                _ = sort_byte; // Must be 0x12 (instance sort)
                arg.instance_idx = try reader.readU32();
            }
            return .{ .instantiate = .{ .module_idx = module_idx, .args = args } };
        },
        0x01 => {
            const count = try reader.readU32();
            const exps = try allocator.alloc(ctypes.CoreInlineExport, count);
            for (exps) |*e| {
                e.name = try reader.readName();
                const sort = try reader.readByte();
                const idx = try reader.readU32();
                e.sort_idx = .{
                    .sort = std.meta.intToEnum(ctypes.CoreSort, sort) catch return error.InvalidEncoding,
                    .idx = idx,
                };
            }
            return .{ .exports = exps };
        },
        else => return error.InvalidEncoding,
    }
}

fn parseCoreType(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.CoreTypeDef {
    const tag = try reader.readByte();
    switch (tag) {
        0x60 => {
            // Core function type
            const param_count = try reader.readU32();
            const params = try allocator.alloc(ctypes.CoreValType, param_count);
            for (params) |*p| p.* = try readCoreValType(reader);
            const result_count = try reader.readU32();
            const results = try allocator.alloc(ctypes.CoreValType, result_count);
            for (results) |*r| r.* = try readCoreValType(reader);
            return .{ .func = .{ .params = params, .results = results } };
        },
        0x50 => {
            // Core module type
            const decl_count = try reader.readU32();
            var imp_list: std.ArrayListUnmanaged(ctypes.CoreImportDecl) = .{};
            var exp_list: std.ArrayListUnmanaged(ctypes.CoreExportDecl) = .{};
            var i: u32 = 0;
            while (i < decl_count) : (i += 1) {
                const decl_tag = try reader.readByte();
                switch (decl_tag) {
                    0x00 => {
                        // import
                        const mod = try reader.readName();
                        const name = try reader.readName();
                        const type_idx = try reader.readU32();
                        try imp_list.append(allocator, .{ .module = mod, .name = name, .type_idx = type_idx });
                    },
                    0x01 => {
                        // export
                        const name = try reader.readName();
                        const type_idx = try reader.readU32();
                        try exp_list.append(allocator, .{ .name = name, .type_idx = type_idx });
                    },
                    else => return error.InvalidEncoding,
                }
            }
            return .{ .module = .{
                .imports = try imp_list.toOwnedSlice(allocator),
                .exports = try exp_list.toOwnedSlice(allocator),
            } };
        },
        else => return error.InvalidEncoding,
    }
}

fn readCoreValType(reader: *BinaryReader) LoadError!ctypes.CoreValType {
    const b = try reader.readByte();
    return std.meta.intToEnum(ctypes.CoreValType, b) catch return error.InvalidEncoding;
}

fn parseInstance(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.InstanceExpr {
    const tag = try reader.readByte();
    switch (tag) {
        0x00 => {
            const comp_idx = try reader.readU32();
            const arg_count = try reader.readU32();
            const args = try allocator.alloc(ctypes.InstantiateArg, arg_count);
            for (args) |*arg| {
                arg.name = try reader.readName();
                arg.sort_idx = try readSortIdx(reader);
            }
            return .{ .instantiate = .{ .component_idx = comp_idx, .args = args } };
        },
        0x01 => {
            const count = try reader.readU32();
            const exps = try allocator.alloc(ctypes.InlineExport, count);
            for (exps) |*e| {
                e.name = try reader.readName();
                e.sort_idx = try readSortIdx(reader);
            }
            return .{ .exports = exps };
        },
        else => return error.InvalidEncoding,
    }
}

fn readSortIdx(reader: *BinaryReader) LoadError!ctypes.SortIdx {
    const sort = try readSort(reader);
    const idx = try reader.readU32();
    return .{ .sort = sort, .idx = idx };
}

fn readSort(reader: *BinaryReader) LoadError!ctypes.Sort {
    const b = try reader.readByte();
    return switch (b) {
        0x00 => blk: {
            const cs = try reader.readByte();
            break :blk .{ .core = std.meta.intToEnum(ctypes.CoreSort, cs) catch return error.InvalidEncoding };
        },
        0x01 => .func,
        0x02 => .value,
        0x03 => .type,
        0x04 => .component,
        0x05 => .instance,
        else => error.InvalidEncoding,
    };
}

fn parseAlias(reader: *BinaryReader) LoadError!ctypes.Alias {
    const sort = try readSort(reader);
    const target_tag = try reader.readByte();
    switch (target_tag) {
        0x00 => {
            // export alias
            const instance_idx = try reader.readU32();
            const name = try reader.readName();
            return .{ .instance_export = .{ .sort = sort, .instance_idx = instance_idx, .name = name } };
        },
        0x01 => {
            // outer alias
            const outer_count = try reader.readU32();
            const idx = try reader.readU32();
            return .{ .outer = .{ .sort = sort, .outer_count = outer_count, .idx = idx } };
        },
        else => return error.InvalidEncoding,
    }
}

fn parseTypeDef(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.TypeDef {
    const tag = try reader.readByte();
    return switch (tag) {
        // Defined types
        0x72 => blk: {
            // record
            const count = try reader.readU32();
            const fields = try allocator.alloc(ctypes.Field, count);
            for (fields) |*f| {
                f.name = try reader.readName();
                f.type = try readValType(reader);
            }
            break :blk .{ .record = .{ .fields = fields } };
        },
        0x71 => blk: {
            // variant
            const count = try reader.readU32();
            const cases = try allocator.alloc(ctypes.Case, count);
            for (cases) |*c| {
                c.name = try reader.readName();
                const has_type = try reader.readByte();
                c.type = if (has_type != 0) try readValType(reader) else null;
                const has_refines = try reader.readByte();
                c.refines = if (has_refines != 0) try reader.readU32() else null;
            }
            break :blk .{ .variant = .{ .cases = cases } };
        },
        0x70 => blk: {
            // list
            const elem = try readValType(reader);
            break :blk .{ .list = .{ .element = elem } };
        },
        0x6F => blk: {
            // tuple
            const count = try reader.readU32();
            const fields = try allocator.alloc(ctypes.ValType, count);
            for (fields) |*f| f.* = try readValType(reader);
            break :blk .{ .tuple = .{ .fields = fields } };
        },
        0x6E => blk: {
            // flags
            const count = try reader.readU32();
            const names = try allocator.alloc([]const u8, count);
            for (names) |*n| n.* = try reader.readName();
            break :blk .{ .flags = .{ .names = names } };
        },
        0x6D => blk: {
            // enum
            const count = try reader.readU32();
            const names = try allocator.alloc([]const u8, count);
            for (names) |*n| n.* = try reader.readName();
            break :blk .{ .enum_ = .{ .names = names } };
        },
        0x6B => blk: {
            // option
            const inner = try readValType(reader);
            break :blk .{ .option = .{ .inner = inner } };
        },
        0x6A => blk: {
            // result
            const has_ok = try reader.readByte();
            const ok = if (has_ok != 0) try readValType(reader) else null;
            const has_err = try reader.readByte();
            const err = if (has_err != 0) try readValType(reader) else null;
            break :blk .{ .result = .{ .ok = ok, .err = err } };
        },
        0x3F => blk: {
            // resource
            const has_dtor = try reader.readByte();
            const dtor = if (has_dtor != 0) try reader.readU32() else null;
            break :blk .{ .resource = .{ .destructor = dtor } };
        },
        0x40 => blk: {
            // func type
            const param_count = try reader.readU32();
            const params = try allocator.alloc(ctypes.NamedValType, param_count);
            for (params) |*p| {
                p.name = try reader.readName();
                p.type = try readValType(reader);
            }
            const result_tag = try reader.readByte();
            const results: ctypes.FuncType.ResultList = switch (result_tag) {
                0x00 => .{ .named = blk2: {
                    const rcount = try reader.readU32();
                    const r = try allocator.alloc(ctypes.NamedValType, rcount);
                    for (r) |*rv| {
                        rv.name = try reader.readName();
                        rv.type = try readValType(reader);
                    }
                    break :blk2 r;
                } },
                0x01 => .{ .unnamed = try readValType(reader) },
                else => return error.InvalidEncoding,
            };
            break :blk .{ .func = .{ .params = params, .results = results } };
        },
        0x41 => blk: {
            // component type
            const count = try reader.readU32();
            var imp_list: std.ArrayListUnmanaged(ctypes.ImportDecl) = .{};
            var exp_list: std.ArrayListUnmanaged(ctypes.ExportDecl) = .{};
            var i: u32 = 0;
            while (i < count) : (i += 1) {
                const decl_tag = try reader.readByte();
                switch (decl_tag) {
                    0x03 => try imp_list.append(allocator, try parseImport(reader)),
                    0x04 => try exp_list.append(allocator, try parseExport(reader)),
                    else => return error.InvalidEncoding,
                }
            }
            break :blk .{ .component = .{
                .imports = try imp_list.toOwnedSlice(allocator),
                .exports = try exp_list.toOwnedSlice(allocator),
            } };
        },
        0x42 => blk: {
            // instance type
            const count = try reader.readU32();
            var exp_list: std.ArrayListUnmanaged(ctypes.ExportDecl) = .{};
            var i: u32 = 0;
            while (i < count) : (i += 1) {
                const decl_tag = try reader.readByte();
                if (decl_tag != 0x04) return error.InvalidEncoding;
                try exp_list.append(allocator, try parseExport(reader));
            }
            break :blk .{ .instance = .{
                .exports = try exp_list.toOwnedSlice(allocator),
            } };
        },
        else => error.InvalidEncoding,
    };
}

fn readValType(reader: *BinaryReader) LoadError!ctypes.ValType {
    const b = try reader.readByte();
    return switch (b) {
        0x7F => .bool,
        0x7E => .s8,
        0x7D => .u8,
        0x7C => .s16,
        0x7B => .u16,
        0x7A => .s32,
        0x79 => .u32,
        0x78 => .s64,
        0x77 => .u64,
        0x76 => .f32,
        0x75 => .f64,
        0x74 => .char,
        0x73 => .string,
        0x69 => .{ .own = try reader.readU32() },
        0x68 => .{ .borrow = try reader.readU32() },
        else => .{ .type_idx = @as(u32, b) },
    };
}

fn parseCanon(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.Canon {
    const tag = try reader.readByte();
    return switch (tag) {
        0x00 => blk: {
            // canon lift
            const sub = try reader.readByte();
            if (sub != 0x00) return error.InvalidEncoding;
            const core_func_idx = try reader.readU32();
            const opts = try readCanonOpts(reader, allocator);
            const type_idx = try reader.readU32();
            break :blk .{ .lift = .{
                .core_func_idx = core_func_idx,
                .type_idx = type_idx,
                .opts = opts,
            } };
        },
        0x01 => blk: {
            // canon lower
            const sub = try reader.readByte();
            if (sub != 0x00) return error.InvalidEncoding;
            const func_idx = try reader.readU32();
            const opts = try readCanonOpts(reader, allocator);
            break :blk .{ .lower = .{ .func_idx = func_idx, .opts = opts } };
        },
        0x02 => .{ .resource_new = try reader.readU32() },
        0x03 => .{ .resource_drop = try reader.readU32() },
        0x04 => .{ .resource_rep = try reader.readU32() },
        else => error.InvalidEncoding,
    };
}

fn readCanonOpts(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError![]const ctypes.CanonOpt {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    const opts = try allocator.alloc(ctypes.CanonOpt, count);
    for (opts) |*o| {
        const tag = try reader.readByte();
        o.* = switch (tag) {
            0x00 => .{ .string_encoding = .utf8 },
            0x01 => .{ .string_encoding = .utf16 },
            0x02 => .{ .string_encoding = .latin1_utf16 },
            0x03 => .{ .memory = try reader.readU32() },
            0x04 => .{ .realloc = try reader.readU32() },
            0x05 => .{ .post_return = try reader.readU32() },
            else => return error.InvalidEncoding,
        };
    }
    return opts;
}

fn parseStart(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.Start {
    const func_idx = try reader.readU32();
    const arg_count = try reader.readU32();
    const args = try allocator.alloc(u32, arg_count);
    for (args) |*a| a.* = try reader.readU32();
    const results = try reader.readU32();
    return .{ .func_idx = func_idx, .args = args, .results = results };
}

fn readExternDesc(reader: *BinaryReader) LoadError!ctypes.ExternDesc {
    const tag = try reader.readByte();
    return switch (tag) {
        0x00 => blk: {
            const sub = try reader.readByte();
            if (sub != 0x11) return error.InvalidEncoding; // module sort
            break :blk .{ .module = try reader.readU32() };
        },
        0x01 => .{ .func = try reader.readU32() },
        0x02 => .{ .value = try readValType(reader) },
        0x03 => blk: {
            const bound_tag = try reader.readByte();
            break :blk .{ .type = switch (bound_tag) {
                0x00 => .{ .eq = try reader.readU32() },
                0x01 => .sub_resource,
                else => return error.InvalidEncoding,
            } };
        },
        0x04 => .{ .component = try reader.readU32() },
        0x05 => .{ .instance = try reader.readU32() },
        else => error.InvalidEncoding,
    };
}

fn parseImport(reader: *BinaryReader) LoadError!ctypes.ImportDecl {
    const name = try reader.readName();
    const desc = try readExternDesc(reader);
    return .{ .name = name, .desc = desc };
}

fn parseExport(reader: *BinaryReader) LoadError!ctypes.ExportDecl {
    const name = try reader.readName();
    const desc = try readExternDesc(reader);
    return .{ .name = name, .desc = desc };
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "load: minimal empty component" {
    const data = [_]u8{
        // magic
        0x00, 0x61, 0x73, 0x6D,
        // version=0x0d, layer=0x01
        0x0d, 0x00, 0x01, 0x00,
    };
    const comp = try load(&data, std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), comp.core_modules.len);
    try std.testing.expectEqual(@as(usize, 0), comp.imports.len);
    try std.testing.expectEqual(@as(usize, 0), comp.exports.len);
    try std.testing.expectEqual(@as(usize, 0), comp.types.len);
    try std.testing.expect(comp.start == null);
}

test "load: invalid magic returns error" {
    const data = [_]u8{ 0x00, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x01, 0x00 };
    try std.testing.expectError(error.InvalidMagic, load(&data, std.testing.allocator));
}

test "load: core module version returns error" {
    const data = [_]u8{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 };
    try std.testing.expectError(error.InvalidVersion, load(&data, std.testing.allocator));
}

test "readValType: primitive types" {
    var reader = BinaryReader{ .data = &[_]u8{ 0x7F, 0x7A, 0x73 } };
    const v1 = try readValType(&reader);
    try std.testing.expect(v1 == .bool);
    const v2 = try readValType(&reader);
    try std.testing.expect(v2 == .s32);
    const v3 = try readValType(&reader);
    try std.testing.expect(v3 == .string);
}
