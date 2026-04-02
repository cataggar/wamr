//! WebAssembly binary format loader.
//!
//! Parses a .wasm binary into a WasmModule structure according to the
//! WebAssembly specification §5.5 (Binary Format).

const std = @import("std");
const types = @import("../common/types.zig");
const leb128_mod = @import("../../shared/utils/leb128.zig");

pub const LoadError = error{
    InvalidMagic,
    InvalidVersion,
    InvalidSectionOrder,
    InvalidSectionSize,
    UnexpectedEnd,
    InvalidTypeIndex,
    InvalidFuncType,
    InvalidValType,
    InvalidImportKind,
    InvalidExportKind,
    InvalidLimits,
    InvalidInitExpr,
    InvalidDataSegment,
    InvalidElemSegment,
    TooManyLocals,
    Overflow,
    OutOfMemory,
};

/// A streaming reader over the Wasm binary.
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
        const result = leb128_mod.readUnsigned(u32, slice) catch |err| switch (err) {
            error.Overflow => return error.Overflow,
            error.UnexpectedEnd => return error.UnexpectedEnd,
        };
        self.pos += result.bytes_read;
        return result.value;
    }

    fn readI32(self: *BinaryReader) LoadError!i32 {
        const slice = self.data[self.pos..];
        const result = leb128_mod.readSigned(i32, slice) catch |err| switch (err) {
            error.Overflow => return error.Overflow,
            error.UnexpectedEnd => return error.UnexpectedEnd,
        };
        self.pos += result.bytes_read;
        return result.value;
    }

    fn readI64(self: *BinaryReader) LoadError!i64 {
        const slice = self.data[self.pos..];
        const result = leb128_mod.readSigned(i64, slice) catch |err| switch (err) {
            error.Overflow => return error.Overflow,
            error.UnexpectedEnd => return error.UnexpectedEnd,
        };
        self.pos += result.bytes_read;
        return result.value;
    }

    fn readU64(self: *BinaryReader) LoadError!u64 {
        const slice = self.data[self.pos..];
        const result = leb128_mod.readUnsigned(u64, slice) catch |err| switch (err) {
            error.Overflow => return error.Overflow,
            error.UnexpectedEnd => return error.UnexpectedEnd,
        };
        self.pos += result.bytes_read;
        return result.value;
    }

    fn readF32(self: *BinaryReader) LoadError!f32 {
        const bytes = try self.readBytes(4);
        return @bitCast(std.mem.readInt(u32, bytes[0..4], .little));
    }

    fn readF64(self: *BinaryReader) LoadError!f64 {
        const bytes = try self.readBytes(8);
        return @bitCast(std.mem.readInt(u64, bytes[0..8], .little));
    }

    fn readFixedU32(self: *BinaryReader) LoadError!u32 {
        const bytes = try self.readBytes(4);
        return std.mem.readInt(u32, bytes[0..4], .little);
    }

    fn readName(self: *BinaryReader) LoadError![]const u8 {
        const len = try self.readU32();
        return try self.readBytes(len);
    }

    fn skip(self: *BinaryReader, n: usize) LoadError!void {
        if (self.pos + n > self.data.len) return error.UnexpectedEnd;
        self.pos += n;
    }
};

// ─── Helpers ────────────────────────────────────────────────────────────────

fn readValType(reader: *BinaryReader) LoadError!types.ValType {
    const byte = try reader.readByte();
    return switch (byte) {
        0x7F => .i32,
        0x7E => .i64,
        0x7D => .f32,
        0x7C => .f64,
        0x7B => .v128,
        0x70 => .funcref,
        0x6F => .externref,
        else => error.InvalidValType,
    };
}

fn readLimits(reader: *BinaryReader) LoadError!types.Limits {
    const flags = try reader.readByte();
    const min = try reader.readU32();
    return switch (flags) {
        0x00 => .{ .min = min },
        0x01 => .{ .min = min, .max = try reader.readU32() },
        else => error.InvalidLimits,
    };
}

fn readTableType(reader: *BinaryReader) LoadError!types.TableType {
    const elem_type = try readValType(reader);
    const limits = try readLimits(reader);
    return .{ .elem_type = elem_type, .limits = limits };
}

fn readMemoryType(reader: *BinaryReader) LoadError!types.MemoryType {
    const limits = try readLimits(reader);
    return .{ .limits = limits };
}

fn readGlobalType(reader: *BinaryReader) LoadError!types.GlobalType {
    const val_type = try readValType(reader);
    const mut_byte = try reader.readByte();
    const mutability: types.GlobalType.Mutability = switch (mut_byte) {
        0 => .immutable,
        1 => .mutable,
        else => return error.InvalidValType,
    };
    return .{ .val_type = val_type, .mutability = mutability };
}

fn parseInitExpr(reader: *BinaryReader) LoadError!types.InitExpr {
    const opcode = try reader.readByte();
    const expr: types.InitExpr = switch (opcode) {
        0x41 => .{ .i32_const = try reader.readI32() },
        0x42 => .{ .i64_const = try reader.readI64() },
        0x43 => .{ .f32_const = try reader.readF32() },
        0x44 => .{ .f64_const = try reader.readF64() },
        0x23 => .{ .global_get = try reader.readU32() },
        0xD0 => .{ .ref_null = try readValType(reader) },
        0xD2 => .{ .ref_func = try reader.readU32() },
        else => return error.InvalidInitExpr,
    };
    const end = try reader.readByte();
    if (end != 0x0B) return error.InvalidInitExpr;
    return expr;
}

// ─── Section parsers ────────────────────────────────────────────────────────

fn parseTypeSection(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError![]const types.FuncType {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    const func_types = try allocator.alloc(types.FuncType, count);
    for (func_types) |*ft| {
        const tag = try reader.readByte();
        if (tag != 0x60) return error.InvalidFuncType;

        const param_count = try reader.readU32();
        const params: []const types.ValType = if (param_count > 0) blk: {
            const p = try allocator.alloc(types.ValType, param_count);
            for (p) |*v| v.* = try readValType(reader);
            break :blk p;
        } else &[_]types.ValType{};

        const result_count = try reader.readU32();
        const results: []const types.ValType = if (result_count > 0) blk: {
            const r = try allocator.alloc(types.ValType, result_count);
            for (r) |*v| v.* = try readValType(reader);
            break :blk r;
        } else &[_]types.ValType{};

        ft.* = .{ .params = params, .results = results };
    }
    return func_types;
}

fn parseImportSection(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError![]const types.ImportDesc {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    const imports = try allocator.alloc(types.ImportDesc, count);
    for (imports) |*imp| {
        const module_name = try reader.readName();
        const field_name = try reader.readName();
        const kind_byte = try reader.readByte();
        const kind: types.ExternalKind = switch (kind_byte) {
            0x00 => .function,
            0x01 => .table,
            0x02 => .memory,
            0x03 => .global,
            else => return error.InvalidImportKind,
        };

        imp.* = .{
            .module_name = module_name,
            .field_name = field_name,
            .kind = kind,
        };

        switch (kind) {
            .function => imp.func_type_idx = try reader.readU32(),
            .table => imp.table_type = try readTableType(reader),
            .memory => imp.memory_type = try readMemoryType(reader),
            .global => imp.global_type = try readGlobalType(reader),
        }
    }
    return imports;
}

fn parseFunctionSection(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError![]u32 {
    const count = try reader.readU32();
    if (count == 0) return &[_]u32{};
    const indices = try allocator.alloc(u32, count);
    for (indices) |*idx| {
        idx.* = try reader.readU32();
    }
    return indices;
}

fn parseTableSection(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError![]const types.TableType {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    const tables = try allocator.alloc(types.TableType, count);
    for (tables) |*t| {
        t.* = try readTableType(reader);
    }
    return tables;
}

fn parseMemorySection(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError![]const types.MemoryType {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    const memories = try allocator.alloc(types.MemoryType, count);
    for (memories) |*m| {
        m.* = try readMemoryType(reader);
    }
    return memories;
}

fn parseGlobalSection(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError![]const types.WasmGlobal {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    const globals = try allocator.alloc(types.WasmGlobal, count);
    for (globals) |*g| {
        const global_type = try readGlobalType(reader);
        const init_expr = try parseInitExpr(reader);
        g.* = .{ .global_type = global_type, .init_expr = init_expr };
    }
    return globals;
}

fn parseExportSection(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError![]const types.ExportDesc {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    const exports = try allocator.alloc(types.ExportDesc, count);
    for (exports) |*e| {
        const name = try reader.readName();
        const kind_byte = try reader.readByte();
        const kind: types.ExternalKind = switch (kind_byte) {
            0x00 => .function,
            0x01 => .table,
            0x02 => .memory,
            0x03 => .global,
            else => return error.InvalidExportKind,
        };
        const index = try reader.readU32();
        e.* = .{ .name = name, .kind = kind, .index = index };
    }
    return exports;
}

fn parseElementSection(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError![]const types.ElemSegment {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    const elements = try allocator.alloc(types.ElemSegment, count);
    for (elements) |*elem| {
        const flags = try reader.readU32();
        switch (flags) {
            0 => {
                // Active, table 0, offset expr, vec(funcidx)
                const offset = try parseInitExpr(reader);
                const num_elems = try reader.readU32();
                const func_indices: []const u32 = if (num_elems > 0) blk: {
                    const fi = try allocator.alloc(u32, num_elems);
                    for (fi) |*f| f.* = try reader.readU32();
                    break :blk fi;
                } else &[_]u32{};
                elem.* = .{
                    .table_idx = 0,
                    .offset = offset,
                    .kind = .func_ref,
                    .func_indices = func_indices,
                };
            },
            1 => {
                // Passive, elemkind, vec(funcidx)
                _ = try reader.readByte(); // elemkind (0x00 = funcref)
                const num_elems = try reader.readU32();
                const func_indices: []const u32 = if (num_elems > 0) blk: {
                    const fi = try allocator.alloc(u32, num_elems);
                    for (fi) |*f| f.* = try reader.readU32();
                    break :blk fi;
                } else &[_]u32{};
                elem.* = .{
                    .table_idx = 0,
                    .offset = null,
                    .kind = .func_ref,
                    .func_indices = func_indices,
                    .is_passive = true,
                };
            },
            2 => {
                // Active, tableidx, offset expr, elemkind, vec(funcidx)
                const table_idx = try reader.readU32();
                const offset = try parseInitExpr(reader);
                _ = try reader.readByte(); // elemkind
                const num_elems = try reader.readU32();
                const func_indices: []const u32 = if (num_elems > 0) blk: {
                    const fi = try allocator.alloc(u32, num_elems);
                    for (fi) |*f| f.* = try reader.readU32();
                    break :blk fi;
                } else &[_]u32{};
                elem.* = .{
                    .table_idx = table_idx,
                    .offset = offset,
                    .kind = .func_ref,
                    .func_indices = func_indices,
                };
            },
            else => return error.InvalidElemSegment,
        }
    }
    return elements;
}

fn parseCodeSection(
    reader: *BinaryReader,
    func_type_indices: []const u32,
    module_types: []const types.FuncType,
    allocator: std.mem.Allocator,
) LoadError![]const types.WasmFunction {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    if (count != func_type_indices.len) return error.InvalidSectionSize;

    const functions = try allocator.alloc(types.WasmFunction, count);
    for (functions, 0..) |*func, i| {
        const body_size = try reader.readU32();
        const body_start = reader.pos;

        const local_decl_count = try reader.readU32();
        var total_locals: u64 = 0;
        const locals: []const types.LocalDecl = if (local_decl_count > 0) blk: {
            const l = try allocator.alloc(types.LocalDecl, local_decl_count);
            for (l) |*ld| {
                const lcount = try reader.readU32();
                const val_type = try readValType(reader);
                ld.* = .{ .count = lcount, .val_type = val_type };
                total_locals += lcount;
                if (total_locals > std.math.maxInt(u32)) return error.TooManyLocals;
            }
            break :blk l;
        } else &[_]types.LocalDecl{};

        const code_end = body_start + body_size;
        if (code_end > reader.data.len) return error.UnexpectedEnd;
        const code = reader.data[reader.pos..code_end];
        reader.pos = code_end;

        const type_idx = func_type_indices[i];
        if (type_idx >= module_types.len) return error.InvalidTypeIndex;
        const func_type = module_types[type_idx];

        func.* = .{
            .type_idx = type_idx,
            .func_type = func_type,
            .local_count = @intCast(total_locals),
            .locals = locals,
            .code = code,
        };
    }
    return functions;
}

fn parseDataSection(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError![]const types.DataSegment {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    const segments = try allocator.alloc(types.DataSegment, count);
    for (segments) |*seg| {
        const flags = try reader.readU32();
        switch (flags) {
            0 => {
                // Active, memory 0, offset expr, data bytes
                const offset = try parseInitExpr(reader);
                const data_len = try reader.readU32();
                const data = try reader.readBytes(data_len);
                seg.* = .{
                    .memory_idx = 0,
                    .offset = offset,
                    .data = data,
                };
            },
            1 => {
                // Passive segment
                const data_len = try reader.readU32();
                const data = try reader.readBytes(data_len);
                seg.* = .{
                    .memory_idx = 0,
                    .offset = .{ .i32_const = 0 },
                    .data = data,
                    .is_passive = true,
                };
            },
            2 => {
                // Active, explicit memory index
                const memory_idx = try reader.readU32();
                const offset = try parseInitExpr(reader);
                const data_len = try reader.readU32();
                const data = try reader.readBytes(data_len);
                seg.* = .{
                    .memory_idx = memory_idx,
                    .offset = offset,
                    .data = data,
                };
            },
            else => return error.InvalidDataSegment,
        }
    }
    return segments;
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Load a Wasm module from binary data.
pub fn load(data: []const u8, allocator: std.mem.Allocator) LoadError!types.WasmModule {
    var reader = BinaryReader{ .data = data };

    const magic = try reader.readFixedU32();
    if (magic != types.wasm_magic) return error.InvalidMagic;
    const version = try reader.readFixedU32();
    if (version != types.wasm_version) return error.InvalidVersion;

    var module = types.WasmModule{};
    var last_section_id: ?u8 = null;
    var func_type_indices: []const u32 = &[_]u32{};

    while (reader.remaining() > 0) {
        const section_id = try reader.readByte();
        const section_size = try reader.readU32();

        // Enforce section ordering (custom sections can appear anywhere)
        if (section_id != 0) {
            if (last_section_id) |last| {
                if (section_id <= last) return error.InvalidSectionOrder;
            }
            last_section_id = section_id;
        }

        const section_start = reader.pos;
        if (section_start + section_size > reader.data.len) return error.InvalidSectionSize;

        if (section_id > @intFromEnum(types.SectionId.data_count)) {
            // Unknown section — skip it
            try reader.skip(section_size);
        } else {
            switch (@as(types.SectionId, @enumFromInt(section_id))) {
                .custom => try reader.skip(section_size),
                .type => module.types = try parseTypeSection(&reader, allocator),
                .import => {
                    module.imports = try parseImportSection(&reader, allocator);
                    for (module.imports) |imp| {
                        switch (imp.kind) {
                            .function => module.import_function_count += 1,
                            .table => module.import_table_count += 1,
                            .memory => module.import_memory_count += 1,
                            .global => module.import_global_count += 1,
                        }
                    }
                },
                .function => func_type_indices = try parseFunctionSection(&reader, allocator),
                .table => module.tables = try parseTableSection(&reader, allocator),
                .memory => module.memories = try parseMemorySection(&reader, allocator),
                .global => module.globals = try parseGlobalSection(&reader, allocator),
                .@"export" => module.exports = try parseExportSection(&reader, allocator),
                .start => module.start_function = try reader.readU32(),
                .element => module.elements = try parseElementSection(&reader, allocator),
                .code => module.functions = try parseCodeSection(&reader, func_type_indices, module.types, allocator),
                .data => module.data_segments = try parseDataSection(&reader, allocator),
                .data_count => module.data_count = try reader.readU32(),
            }
        }

        // Verify we consumed exactly section_size bytes
        if (reader.pos != section_start + section_size) return error.InvalidSectionSize;
    }

    return module;
}

// ─── Tests ──────────────────────────────────────────────────────────────────

const testing = std.testing;

/// Wasm header: \0asm followed by version 1.
const wasm_header = [_]u8{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 };

test "load: minimal valid module" {
    const module = try load(&wasm_header, testing.allocator);
    try testing.expectEqual(@as(?u32, null), module.start_function);
    try testing.expectEqual(@as(usize, 0), module.types.len);
    try testing.expectEqual(@as(usize, 0), module.imports.len);
    try testing.expectEqual(@as(usize, 0), module.functions.len);
    try testing.expectEqual(@as(usize, 0), module.memories.len);
    try testing.expectEqual(@as(usize, 0), module.exports.len);
}

test "load: module with one type (func() -> ())" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // type section: id=1, size=4, count=1, 0x60, params=0, results=0
    const data = wasm_header ++ [_]u8{ 0x01, 0x04, 0x01, 0x60, 0x00, 0x00 };
    const module = try load(&data, allocator);
    try testing.expectEqual(@as(usize, 1), module.types.len);
    try testing.expectEqual(@as(usize, 0), module.types[0].params.len);
    try testing.expectEqual(@as(usize, 0), module.types[0].results.len);
}

test "load: type with params and results" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // func(i32, i32) -> i32
    const data = wasm_header ++ [_]u8{
        0x01, 0x07, // type section, size=7
        0x01, // count=1
        0x60, // func tag
        0x02, 0x7F, 0x7F, // 2 params: i32, i32
        0x01, 0x7F, // 1 result: i32
    };
    const module = try load(&data, allocator);
    try testing.expectEqual(@as(usize, 1), module.types.len);
    try testing.expectEqual(@as(usize, 2), module.types[0].params.len);
    try testing.expectEqual(types.ValType.i32, module.types[0].params[0]);
    try testing.expectEqual(types.ValType.i32, module.types[0].params[1]);
    try testing.expectEqual(@as(usize, 1), module.types[0].results.len);
    try testing.expectEqual(types.ValType.i32, module.types[0].results[0]);
}

test "load: module with one function" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const data = wasm_header ++ [_]u8{
        // type section: 1 type, func() -> ()
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
        // function section: 1 func, type idx 0
        0x03, 0x02, 0x01, 0x00,
        // code section: 1 body, body_size=2, 0 locals, end
        0x0A, 0x04, 0x01, 0x02, 0x00, 0x0B,
    };
    const module = try load(&data, allocator);
    try testing.expectEqual(@as(usize, 1), module.functions.len);
    try testing.expectEqual(@as(u32, 0), module.functions[0].type_idx);
    try testing.expectEqual(@as(u32, 0), module.functions[0].local_count);
    try testing.expectEqual(@as(usize, 0), module.functions[0].locals.len);
    // Code should be just the end opcode
    try testing.expectEqual(@as(usize, 1), module.functions[0].code.len);
    try testing.expectEqual(@as(u8, 0x0B), module.functions[0].code[0]);
}

test "load: module with memory" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const data = wasm_header ++ [_]u8{
        // memory section: 1 memory, limits flag=0 (no max), min=1
        0x05, 0x03, 0x01, 0x00, 0x01,
    };
    const module = try load(&data, arena.allocator());
    try testing.expectEqual(@as(usize, 1), module.memories.len);
    try testing.expectEqual(@as(u32, 1), module.memories[0].limits.min);
    try testing.expectEqual(@as(?u32, null), module.memories[0].limits.max);
}

test "load: memory with max" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const data = wasm_header ++ [_]u8{
        // memory section: 1 memory, limits flag=1 (has max), min=1, max=4
        0x05, 0x04, 0x01, 0x01, 0x01, 0x04,
    };
    const module = try load(&data, arena.allocator());
    try testing.expectEqual(@as(usize, 1), module.memories.len);
    try testing.expectEqual(@as(u32, 1), module.memories[0].limits.min);
    try testing.expectEqual(@as(?u32, 4), module.memories[0].limits.max);
}

test "load: module with export" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const data = wasm_header ++ [_]u8{
        // type section
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
        // function section
        0x03, 0x02, 0x01, 0x00,
        // export section: 1 export, name="main", kind=function, idx=0
        0x07, 0x08, 0x01, 0x04, 'm', 'a', 'i', 'n', 0x00, 0x00,
        // code section
        0x0A, 0x04, 0x01, 0x02, 0x00, 0x0B,
    };
    const module = try load(&data, allocator);
    try testing.expectEqual(@as(usize, 1), module.exports.len);
    try testing.expect(std.mem.eql(u8, "main", module.exports[0].name));
    try testing.expectEqual(types.ExternalKind.function, module.exports[0].kind);
    try testing.expectEqual(@as(u32, 0), module.exports[0].index);
}

test "load: module with import" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const data = wasm_header ++ [_]u8{
        // type section: func() -> ()
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
        // import section: 1 import, module="env", field="f", kind=function, type=0
        0x02, 0x09, 0x01, 0x03, 'e', 'n', 'v', 0x01, 'f', 0x00, 0x00,
    };
    const module = try load(&data, allocator);
    try testing.expectEqual(@as(usize, 1), module.imports.len);
    try testing.expect(std.mem.eql(u8, "env", module.imports[0].module_name));
    try testing.expect(std.mem.eql(u8, "f", module.imports[0].field_name));
    try testing.expectEqual(types.ExternalKind.function, module.imports[0].kind);
    try testing.expectEqual(@as(?u32, 0), module.imports[0].func_type_idx);
    try testing.expectEqual(@as(u32, 1), module.import_function_count);
}

test "load: module with data segment" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const data = wasm_header ++ [_]u8{
        // memory section: 1 memory, min=1
        0x05, 0x03, 0x01, 0x00, 0x01,
        // data section: 1 segment, flags=0, i32.const(0), end, 2 bytes "hi"
        0x0B, 0x08, 0x01, 0x00, 0x41, 0x00, 0x0B, 0x02, 'h', 'i',
    };
    const module = try load(&data, allocator);
    try testing.expectEqual(@as(usize, 1), module.data_segments.len);
    try testing.expectEqual(@as(u32, 0), module.data_segments[0].memory_idx);
    try testing.expect(std.mem.eql(u8, "hi", module.data_segments[0].data));
    try testing.expectEqual(@as(i32, 0), module.data_segments[0].offset.i32_const);
}

test "load: module with global" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const data = wasm_header ++ [_]u8{
        // global section: 1 global, i32 mutable, init=i32.const(42), end
        0x06, 0x06, 0x01, 0x7F, 0x01, 0x41, 0x2A, 0x0B,
    };
    const module = try load(&data, allocator);
    try testing.expectEqual(@as(usize, 1), module.globals.len);
    try testing.expectEqual(types.ValType.i32, module.globals[0].global_type.val_type);
    try testing.expectEqual(types.GlobalType.Mutability.mutable, module.globals[0].global_type.mutability);
    try testing.expectEqual(@as(i32, 42), module.globals[0].init_expr.i32_const);
}

test "load: module with start function" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const data = wasm_header ++ [_]u8{
        // type section
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
        // function section
        0x03, 0x02, 0x01, 0x00,
        // start section: function index 0
        0x08, 0x01, 0x00,
        // code section
        0x0A, 0x04, 0x01, 0x02, 0x00, 0x0B,
    };
    const module = try load(&data, allocator);
    try testing.expectEqual(@as(?u32, 0), module.start_function);
}

test "load: module with table" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const data = wasm_header ++ [_]u8{
        // table section: 1 table, funcref (0x70), limits: no max, min=1
        0x04, 0x04, 0x01, 0x70, 0x00, 0x01,
    };
    const module = try load(&data, arena.allocator());
    try testing.expectEqual(@as(usize, 1), module.tables.len);
    try testing.expectEqual(types.ValType.funcref, module.tables[0].elem_type);
    try testing.expectEqual(@as(u32, 1), module.tables[0].limits.min);
}

test "load: function with locals" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const data = wasm_header ++ [_]u8{
        // type section: func() -> ()
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
        // function section: 1 func, type idx 0
        0x03, 0x02, 0x01, 0x00,
        // code section: 1 body, body_size=4, 1 local decl (2 x i32), end
        0x0A, 0x06, 0x01, 0x04, 0x01, 0x02, 0x7F, 0x0B,
    };
    const module = try load(&data, allocator);
    try testing.expectEqual(@as(usize, 1), module.functions.len);
    try testing.expectEqual(@as(u32, 2), module.functions[0].local_count);
    try testing.expectEqual(@as(usize, 1), module.functions[0].locals.len);
    try testing.expectEqual(@as(u32, 2), module.functions[0].locals[0].count);
    try testing.expectEqual(types.ValType.i32, module.functions[0].locals[0].val_type);
}

test "load: invalid magic" {
    const data = [_]u8{ 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 };
    try testing.expectError(error.InvalidMagic, load(&data, testing.allocator));
}

test "load: invalid version" {
    const data = [_]u8{ 0x00, 0x61, 0x73, 0x6D, 0x02, 0x00, 0x00, 0x00 };
    try testing.expectError(error.InvalidVersion, load(&data, testing.allocator));
}

test "load: truncated input (only magic)" {
    const data = [_]u8{ 0x00, 0x61, 0x73, 0x6D };
    try testing.expectError(error.UnexpectedEnd, load(&data, testing.allocator));
}

test "load: truncated input (empty)" {
    const data = [_]u8{};
    try testing.expectError(error.UnexpectedEnd, load(&data, testing.allocator));
}

test "load: duplicate section is rejected" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const data = wasm_header ++ [_]u8{
        // memory section (id=5)
        0x05, 0x03, 0x01, 0x00, 0x01,
        // duplicate memory section (id=5)
        0x05, 0x03, 0x01, 0x00, 0x02,
    };
    try testing.expectError(error.InvalidSectionOrder, load(&data, arena.allocator()));
}

test "load: out-of-order sections rejected" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const data = wasm_header ++ [_]u8{
        // function section (id=3) before type section (id=1) is out of order
        0x03, 0x02, 0x01, 0x00,
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
    };
    try testing.expectError(error.InvalidSectionOrder, load(&data, arena.allocator()));
}

test "load: custom section can appear anywhere" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const data = wasm_header ++ [_]u8{
        // custom section (id=0): size=5, name="test" (4 bytes)
        0x00, 0x05, 0x04, 't', 'e', 's', 't',
        // memory section
        0x05, 0x03, 0x01, 0x00, 0x01,
        // another custom section
        0x00, 0x05, 0x04, 't', 'e', 's', 't',
    };
    const module = try load(&data, arena.allocator());
    try testing.expectEqual(@as(usize, 1), module.memories.len);
}

test "load: data_count section" {
    const data = wasm_header ++ [_]u8{
        // data count section (id=12): size=1, count=0
        0x0C, 0x01, 0x00,
    };
    const module = try load(&data, testing.allocator);
    try testing.expectEqual(@as(?u32, 0), module.data_count);
}

test "load: section size mismatch rejected" {
    const data = wasm_header ++ [_]u8{
        // type section claims size=5 but only has 4 bytes of content
        0x01, 0x05, 0x01, 0x60, 0x00, 0x00,
    };
    try testing.expectError(error.InvalidSectionSize, load(&data, testing.allocator));
}
