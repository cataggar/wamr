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

    fn peekByte(self: *BinaryReader) LoadError!u8 {
        if (self.pos >= self.data.len) return error.UnexpectedEnd;
        return self.data[self.pos];
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

    /// Read a signed LEB128 in `s33` form (component valtype discriminator).
    /// Non-negative values are type indices; negative values encode primitive
    /// valtypes and handle forms.
    fn readS33(self: *BinaryReader) LoadError!i64 {
        const slice = self.data[self.pos..];
        const result = leb128_mod.readSigned(i64, slice) catch |err| switch (err) {
            error.Overflow => return error.InvalidEncoding,
            error.UnexpectedEnd => return error.UnexpectedEnd,
        };
        // s33: value must fit in 33 signed bits.
        if (result.value < -(@as(i64, 1) << 32) or result.value >= (@as(i64, 1) << 32))
            return error.InvalidEncoding;
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
    var core_modules: std.ArrayListUnmanaged(ctypes.CoreModule) = .empty;
    var core_instances: std.ArrayListUnmanaged(ctypes.CoreInstanceExpr) = .empty;
    var core_type_defs: std.ArrayListUnmanaged(ctypes.CoreTypeDef) = .empty;
    var components: std.ArrayListUnmanaged(*ctypes.Component) = .empty;
    var instances: std.ArrayListUnmanaged(ctypes.InstanceExpr) = .empty;
    var aliases: std.ArrayListUnmanaged(ctypes.Alias) = .empty;
    var type_defs: std.ArrayListUnmanaged(ctypes.TypeDef) = .empty;
    var canons: std.ArrayListUnmanaged(ctypes.Canon) = .empty;
    var imports: std.ArrayListUnmanaged(ctypes.ImportDecl) = .empty;
    var exports: std.ArrayListUnmanaged(ctypes.ExportDecl) = .empty;
    var start: ?ctypes.Start = null;
    // Tracks the type index space in section-encounter order. Each entry
    // is the local idx into `type_defs` for that slot, or null when
    // the slot is consumed by an import (`.type`-bound) or alias whose
    // target type def we don't materialize. Required to resolve real
    // wasm32-wasip2 components where types and aliases interleave.
    var type_indexspace: std.ArrayListUnmanaged(?u32) = .empty;
    // Core-func index space contributors in binary declaration order.
    // Each canon that produces a core func and each `core(.func)` alias
    // appends one entry as it is parsed.
    var core_func_indexspace: std.ArrayListUnmanaged(ctypes.CoreFuncContributor) = .empty;

    while (reader.remaining() > 0) {
        const section_id_byte = try reader.readByte();
        const section_size = try reader.readU32();

        const section_start = reader.pos;
        if (section_start + section_size > reader.data.len) return error.InvalidSectionSize;

        const section_id = std.enums.fromInt(SectionId, section_id_byte) orelse
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
                    const a = try parseAlias(&reader);
                    const local_idx: u32 = @intCast(aliases.items.len);
                    try aliases.append(allocator, a);
                    // Aliases of sort .type contribute a slot to the
                    // type indexspace (target unresolved here → null).
                    const sort: ctypes.Sort = switch (a) {
                        .instance_export => |ie| ie.sort,
                        .outer => |o| o.sort,
                    };
                    if (sort == .type) try type_indexspace.append(allocator, null);
                    // Aliases of sort .core(.func) contribute to the
                    // core-func indexspace.
                    const is_core_func = switch (sort) {
                        .core => |cs| cs == .func,
                        else => false,
                    };
                    if (is_core_func) try core_func_indexspace.append(allocator, .{ .alias = local_idx });
                }
            },
            .type => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    const local_idx: u32 = @intCast(type_defs.items.len);
                    try type_defs.append(allocator, try parseTypeDef(&reader, allocator));
                    try type_indexspace.append(allocator, local_idx);
                }
            },
            .canon => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    const local_idx: u32 = @intCast(canons.items.len);
                    const c = try parseCanon(&reader, allocator);
                    try canons.append(allocator, c);
                    // Every canon kind except `.lift` contributes a slot
                    // to the core-func indexspace.
                    const contributes = switch (c) {
                        .lower, .resource_drop, .resource_new, .resource_rep => true,
                        .lift => false,
                    };
                    if (contributes) try core_func_indexspace.append(allocator, .{ .canon = local_idx });
                }
            },
            .start => {
                start = try parseStart(&reader, allocator);
            },
            .@"import" => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    const imp = try parseImport(&reader);
                    try imports.append(allocator, imp);
                    if (imp.desc == .type) try type_indexspace.append(allocator, null);
                }
            },
            .@"export" => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    try exports.append(allocator, try parseTopLevelExport(&reader));
                }
            },
            .value => {
                // Value definitions — skip for now (gated feature)
                reader.pos = section_start + section_size;
            },
        }
        // Defensive: every typed-section parser above should have consumed
        // exactly `section_size` bytes. If a bug causes under- or over-read
        // we'd otherwise misalign the next section header.
        if (reader.pos != section_start + section_size) return error.InvalidSectionSize;
    }

    return .{
        .core_modules = try core_modules.toOwnedSlice(allocator),
        .core_instances = try core_instances.toOwnedSlice(allocator),
        .core_types = try core_type_defs.toOwnedSlice(allocator),
        .components = try components.toOwnedSlice(allocator),
        .instances = try instances.toOwnedSlice(allocator),
        .aliases = try aliases.toOwnedSlice(allocator),
        .types = try type_defs.toOwnedSlice(allocator),
        .type_indexspace = try type_indexspace.toOwnedSlice(allocator),
        .canons = try canons.toOwnedSlice(allocator),
        .start = start,
        .imports = try imports.toOwnedSlice(allocator),
        .exports = try exports.toOwnedSlice(allocator),
        .core_func_indexspace = try core_func_indexspace.toOwnedSlice(allocator),
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
                    .sort = std.enums.fromInt(ctypes.CoreSort, sort) orelse return error.InvalidEncoding,
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
            var imp_list: std.ArrayListUnmanaged(ctypes.CoreImportDecl) = .empty;
            var exp_list: std.ArrayListUnmanaged(ctypes.CoreExportDecl) = .empty;
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
    return std.enums.fromInt(ctypes.CoreValType, b) orelse return error.InvalidEncoding;
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
            break :blk .{ .core = std.enums.fromInt(ctypes.CoreSort, cs) orelse return error.InvalidEncoding };
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
            // alias export: instance export
            const instance_idx = try reader.readU32();
            const name = try reader.readName();
            return .{ .instance_export = .{ .sort = sort, .instance_idx = instance_idx, .name = name } };
        },
        0x01 => {
            // alias core export: core instance export
            const instance_idx = try reader.readU32();
            const name = try reader.readName();
            return .{ .instance_export = .{ .sort = sort, .instance_idx = instance_idx, .name = name } };
        },
        0x02 => {
            // outer alias
            const outer_count = try reader.readU32();
            const idx = try reader.readU32();
            return .{ .outer = .{ .sort = sort, .outer_count = outer_count, .idx = idx } };
        },
        else => return error.InvalidEncoding,
    }
}

fn parseTypeDef(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.TypeDef {
    // `deftype` starts with a tag byte that selects a compound type;
    // the remaining space (primvaltypes 0x64..0x7F, own 0x69, borrow 0x68,
    // and any non-negative typeidx encoded as signed-LEB) is a bare valtype.
    // Peek the first byte and dispatch without consuming if not recognized.
    const tag = try reader.peekByte();
    return switch (tag) {
        0x72, 0x71, 0x70, 0x6F, 0x6E, 0x6D, 0x6B, 0x6A, 0x3F, 0x40, 0x41, 0x42 => parseCompoundTypeDef(reader, allocator),
        else => .{ .val = try readValType(reader) },
    };
}

fn parseCompoundTypeDef(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.TypeDef {
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
                // Current spec: case ends with a trailing 0x00 byte. (Older
                // drafts used a `refines` u32; no longer emitted.)
                const trailer = try reader.readByte();
                if (trailer != 0x00) return error.InvalidEncoding;
                c.refines = null;
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
            // Current spec (2024): paramlist is a bare vec<labelvaltype>,
            // resultlist is `0x00 valtype` (one result) | `0x01 0x00` (none).
            // See: <https://github.com/WebAssembly/component-model/blob/main/design/mvp/Binary.md#type-definitions>
            const param_count = try reader.readU32();
            const params = try allocator.alloc(ctypes.NamedValType, param_count);
            for (params) |*p| {
                p.name = try reader.readName();
                p.type = try readValType(reader);
            }
            const result_tag = try reader.readByte();
            const results: ctypes.FuncType.ResultList = switch (result_tag) {
                0x00 => .{ .unnamed = try readValType(reader) },
                0x01 => blk2: {
                    const zero = try reader.readByte();
                    if (zero != 0x00) return error.InvalidEncoding;
                    break :blk2 .none;
                },
                else => return error.InvalidEncoding,
            };
            break :blk .{ .func = .{ .params = params, .results = results } };
        },
        0x41 => blk: {
            // component type
            const count = try reader.readU32();
            const decls = try allocator.alloc(ctypes.Decl, count);
            errdefer allocator.free(decls);
            for (decls) |*d| d.* = try parseDecl(reader, allocator, .component_type);
            break :blk .{ .component = .{ .decls = decls } };
        },
        0x42 => blk: {
            // instance type
            const count = try reader.readU32();
            const decls = try allocator.alloc(ctypes.Decl, count);
            errdefer allocator.free(decls);
            for (decls) |*d| d.* = try parseDecl(reader, allocator, .instance_type);
            break :blk .{ .instance = .{ .decls = decls } };
        },
        else => error.InvalidEncoding,
    };
}

/// Scope of a declarator list: which decl tags are legal.
const DeclScope = enum { component_type, instance_type };

/// Parse a single declarator inside a component-type or instance-type body.
///
/// See: <https://github.com/WebAssembly/component-model/blob/main/design/mvp/Binary.md#type-definitions>
fn parseDecl(
    reader: *BinaryReader,
    allocator: std.mem.Allocator,
    scope: DeclScope,
) LoadError!ctypes.Decl {
    const tag = try reader.readByte();
    return switch (tag) {
        0x00 => .{ .core_type = try parseCoreType(reader, allocator) },
        0x01 => .{ .type = try parseTypeDef(reader, allocator) },
        0x02 => .{ .alias = try parseAlias(reader) },
        0x03 => blk: {
            if (scope != .component_type) return error.InvalidEncoding;
            break :blk .{ .import = try parseImport(reader) };
        },
        0x04 => .{ .@"export" = try parseExport(reader) },
        else => error.InvalidEncoding,
    };
}

/// Decode a component-model `valtype`.
///
/// Encoded as a signed LEB128 in `s33` form. Non-negative values are type
/// indices into the component type-index space; negative values are
/// primitive valtypes or handle forms. The `own` / `borrow` variants are
/// followed by an unsigned LEB128 `typeidx`.
///
/// See: <https://github.com/WebAssembly/component-model/blob/main/design/mvp/Binary.md#type-definitions>
fn readValType(reader: *BinaryReader) LoadError!ctypes.ValType {
    const raw = try reader.readS33();
    if (raw >= 0) {
        return .{ .type_idx = @intCast(raw) };
    }
    // Negative: primitive / handle form. Single-byte signed-LEB negatives
    // reach us as values -1..-64; the spec assigns each to a byte-tag
    // which equals `0x80 + raw` (so -1 → 0x7F, -24 → 0x68). Larger negative
    // values can't possibly be a primitive code.
    if (raw < -64) return error.InvalidEncoding;
    const tag: u8 = @intCast(raw + 0x80);
    return switch (tag) {
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
        else => error.InvalidEncoding,
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

/// Read an `importname'` / `exportname'` (identical grammar per spec):
///
///   importname' ::= 0x00 len:<u32> in:<importname>
///                 | 0x01 len:<u32> in:<importname>
///                 | 0x02 len:<u32> in:<importname> vs:<versionsuffix>
///
/// The prefix tag distinguishes plain names from annotated/versioned names.
/// For now we return the raw `in` bytes and swallow any `versionsuffix` —
/// the runtime only needs the interface name to match against host bindings.
fn readExternName(reader: *BinaryReader) LoadError![]const u8 {
    const prefix = try reader.readByte();
    if (prefix > 0x02) return error.InvalidEncoding;
    const name = try reader.readName();
    if (prefix == 0x02) {
        // versionsuffix ::= len:<u32> vs:<semversuffix> — skip over it.
        _ = try reader.readName();
    }
    return name;
}

fn parseImport(reader: *BinaryReader) LoadError!ctypes.ImportDecl {
    const name = try readExternName(reader);
    const desc = try readExternDesc(reader);
    return .{ .name = name, .desc = desc };
}

fn parseExport(reader: *BinaryReader) LoadError!ctypes.ExportDecl {
    // exportdecl (used inside component/instance type bodies):
    //   en:<exportname'> ed:<externdesc>
    const name = try readExternName(reader);
    const desc = try readExternDesc(reader);
    return .{ .name = name, .desc = desc };
}

/// Parse a top-level `export` entry from the export section:
///   export ::= en:<exportname'> si:<sortidx> ed?:<externdesc>?
///
/// Distinct from `parseExport` (the declarator form used inside
/// component/instance types), which has no sortidx and a mandatory descriptor.
fn parseTopLevelExport(reader: *BinaryReader) LoadError!ctypes.ExportDecl {
    const name = try readExternName(reader);
    const sort_idx = try readSortIdx(reader);
    const has_desc = try reader.readByte();
    const desc: ctypes.ExternDesc = switch (has_desc) {
        0x00 => inferExternDescFromSort(sort_idx),
        0x01 => try readExternDesc(reader),
        else => return error.InvalidEncoding,
    };
    return .{ .name = name, .desc = desc, .sort_idx = sort_idx };
}

/// When a top-level export omits its externdesc, the sortidx itself describes
/// the kind. For sorts that carry a type idx (func/component/instance) we
/// have no explicit type; fall back to a best-effort placeholder. The runtime
/// treats these as opaque until Phase 1B index-space resolution fills them in.
fn inferExternDescFromSort(si: ctypes.SortIdx) ctypes.ExternDesc {
    return switch (si.sort) {
        .func => .{ .func = 0 },
        .value => .{ .value = .{ .type_idx = 0 } },
        .type => .{ .type = .{ .eq = si.idx } },
        .component => .{ .component = 0 },
        .instance => .{ .instance = 0 },
        .core => .{ .module = 0 },
    };
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

test "readValType: own/borrow with typeidx" {
    // own 3, borrow 5
    var reader = BinaryReader{ .data = &[_]u8{ 0x69, 0x03, 0x68, 0x05 } };
    const v1 = try readValType(&reader);
    try std.testing.expectEqual(@as(u32, 3), v1.own);
    const v2 = try readValType(&reader);
    try std.testing.expectEqual(@as(u32, 5), v2.borrow);
}

test "readValType: typeidx non-negative, single byte" {
    // 0 and 63 encode as single bytes 0x00 and 0x3F in signed-LEB.
    var reader = BinaryReader{ .data = &[_]u8{ 0x00, 0x3F } };
    const v1 = try readValType(&reader);
    try std.testing.expectEqual(@as(u32, 0), v1.type_idx);
    const v2 = try readValType(&reader);
    try std.testing.expectEqual(@as(u32, 63), v2.type_idx);
}

test "readValType: typeidx >= 64 requires multi-byte signed LEB" {
    // typeidx 64: signed-LEB `0xC0 0x00` (cont + value 64, trailing 0).
    // typeidx 128: signed-LEB `0x80 0x01`.
    // typeidx 8192: signed-LEB `0x80 0xC0 0x00` (trailing 0 to keep sign positive).
    var reader = BinaryReader{ .data = &[_]u8{ 0xC0, 0x00, 0x80, 0x01, 0x80, 0xC0, 0x00 } };
    const v64 = try readValType(&reader);
    try std.testing.expectEqual(@as(u32, 64), v64.type_idx);
    const v128 = try readValType(&reader);
    try std.testing.expectEqual(@as(u32, 128), v128.type_idx);
    const v8192 = try readValType(&reader);
    try std.testing.expectEqual(@as(u32, 8192), v8192.type_idx);
}

test "readValType: rejects unknown negative code" {
    // 0x67 decodes as signed LEB -25, which has no primitive mapping.
    var reader = BinaryReader{ .data = &[_]u8{0x67} };
    try std.testing.expectError(error.InvalidEncoding, readValType(&reader));
}

test "parseTypeDef: instance type with `sub resource` type decl" {
    // Mirrors the first type definition in every Rust wasm32-wasip2 component:
    //   (type (instance
    //     (type $p (sub resource))            ; decl 0x01, type, resource with no dtor
    //     (export "pollable" (type (eq $p)))  ; decl 0x04, export, type bound eq 0
    //   ))
    // Binary form:
    //   0x42              ; instance-type tag
    //   0x02              ; 2 decls
    //   0x01              ; decl 1: type
    //     0x3F 0x00       ; resource with no destructor
    //   0x04              ; decl 2: export
    //     0x00            ; exportname' prefix
    //     0x08 "pollable" ; name (len=8)
    //     0x03            ; externdesc: type
    //     0x00 0x00       ; bound: eq, typeidx 0
    const data = [_]u8{
        0x42, 0x02,
        0x01, 0x3F, 0x00,
        0x04, 0x00, 0x08, 'p', 'o', 'l', 'l', 'a', 'b', 'l', 'e',
        0x03, 0x00, 0x00,
    };
    var reader = BinaryReader{ .data = &data };
    const td = try parseTypeDef(&reader, std.testing.allocator);
    defer {
        // Free the decls slice (individual decls don't own heap beyond exports' names which are slices into `data`).
        std.testing.allocator.free(td.instance.decls);
    }
    try std.testing.expect(td == .instance);
    try std.testing.expectEqual(@as(usize, 2), td.instance.decls.len);
    try std.testing.expect(td.instance.decls[0] == .type);
    try std.testing.expect(td.instance.decls[0].type == .resource);
    try std.testing.expect(td.instance.decls[1] == .@"export");
    try std.testing.expectEqualStrings("pollable", td.instance.decls[1].@"export".name);
    try std.testing.expect(td.instance.decls[1].@"export".desc == .type);
    try std.testing.expectEqual(@as(u32, 0), td.instance.decls[1].@"export".desc.type.eq);
}

test "parseTypeDef: component type with import and alias decls" {
    // (type (component
    //   (import "x" (instance (type 0)))   ; decl 0x03, import
    //   (alias outer 0 0 (type))           ; decl 0x02, alias outer
    // ))
    const data = [_]u8{
        0x41, 0x02,
        0x03, 0x00, 0x01, 'x', 0x05, 0x00, // import name'=(0x00 "x") externdesc=instance type 0
        0x02, 0x03, 0x02, 0x00, 0x00, // alias: sort=type(0x03), outer(0x02), count=0, idx=0
    };
    var reader = BinaryReader{ .data = &data };
    const td = try parseTypeDef(&reader, std.testing.allocator);
    defer std.testing.allocator.free(td.component.decls);
    try std.testing.expect(td == .component);
    try std.testing.expectEqual(@as(usize, 2), td.component.decls.len);
    try std.testing.expect(td.component.decls[0] == .import);
    try std.testing.expectEqualStrings("x", td.component.decls[0].import.name);
    try std.testing.expect(td.component.decls[1] == .alias);
}

test "parseTypeDef: instance type rejects import decl" {
    // Instance types cannot contain import decls (0x03).
    const data = [_]u8{ 0x42, 0x01, 0x03 };
    var reader = BinaryReader{ .data = &data };
    try std.testing.expectError(error.InvalidEncoding, parseTypeDef(&reader, std.testing.allocator));
}

test "load: real wasm32-wasip2 Rust component (stdio-echo)" {
    // Prebuilt binary of tests/component/src/stdio-echo/ — a minimal
    // Rust `fn main { println!("echo: ..."); }` compiled with
    // `cargo build --release --target wasm32-wasip2`. This is the canonical
    // Phase 1A regression fixture for #142: before the loader rework every
    // wasm32-wasip2 component failed at the first `type` section.
    const data = @embedFile("fixtures/stdio-echo.wasm");

    // The loader allocates many small slices but has no Component.deinit yet
    // (see #142 Phase 1B). Use an arena so the test doesn't leak.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const comp = try load(data, arena.allocator());

    // Verified via `wasm-tools component wit`: world `root` imports 13
    // wasi interfaces (io/poll, io/error, io/streams, cli/environment,
    // cli/exit, cli/stdin, cli/stdout, cli/stderr, and 5 cli/terminal-*),
    // and exports one: wasi:cli/run@0.2.0.
    try std.testing.expectEqual(@as(usize, 13), comp.imports.len);
    try std.testing.expectEqual(@as(usize, 1), comp.exports.len);

    // Every import is an instance import (WASI p2 pattern — never flat funcs).
    for (comp.imports) |imp| {
        try std.testing.expect(imp.desc == .instance);
    }
    // The sole export is the wasi:cli/run instance.
    try std.testing.expect(comp.exports[0].desc == .instance);
    try std.testing.expect(std.mem.startsWith(u8, comp.exports[0].name, "wasi:cli/run@"));

    // Spot-check a handful of imports against the golden list from wasm-tools.
    const expected = [_][]const u8{
        "wasi:io/poll@0.2.6",
        "wasi:io/streams@0.2.6",
        "wasi:cli/stdin@0.2.6",
        "wasi:cli/stdout@0.2.6",
        "wasi:cli/exit@0.2.6",
    };
    for (expected) |name| {
        var found = false;
        for (comp.imports) |imp| {
            if (std.mem.eql(u8, imp.name, name)) {
                found = true;
                break;
            }
        }
        std.testing.expect(found) catch |err| {
            std.debug.print("missing expected import: {s}\n", .{name});
            return err;
        };
    }
}
