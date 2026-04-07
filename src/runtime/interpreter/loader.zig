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
    InvalidUtf8,
    DuplicateExportName,
    UnknownFunction,
    UnknownTable,
    UnknownMemory,
    UnknownGlobal,
    TooManyMemories,
    TooManyTables,
    InvalidStartFunction,
    InvalidAlignment,
    DataCountMismatch,
    TypeMismatch,
    MalformedSectionId,
    DataCountRequired,
    IllegalOpcode,
    UndeclaredFuncRef,
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
        const bytes = try self.readBytes(len);
        if (!std.unicode.utf8ValidateSlice(bytes)) return error.InvalidUtf8;
        return bytes;
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
        // Typed reference types: ref null <heaptype> or ref <heaptype>
        // Accept abstract heap types, reject concrete type indices
        0x63, 0x64 => {
            const heap_byte = try reader.readByte();
            return switch (heap_byte) {
                0x70, 0x73 => .funcref, // func, nofunc → funcref
                0x6F, 0x72 => .externref, // extern, noextern → externref
                else => error.InvalidValType, // concrete type indices not yet supported
            };
        },
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

fn readTableType(reader: *BinaryReader, type_count: u32) LoadError!types.TableType {
    // Table element types support concrete typed refs (ref null $type)
    const first_byte = try reader.readByte();
    const elem_type: types.ValType = switch (first_byte) {
        0x70 => .funcref,
        0x6F => .externref,
        0x63, 0x64 => blk: {
            const heap_byte = try reader.readByte();
            break :blk switch (heap_byte) {
                0x70, 0x73 => .funcref,
                0x6F, 0x72 => .externref,
                else => {
                    // Concrete type index LEB128 — validate against type count
                    var type_idx: u32 = heap_byte & 0x7F;
                    if (heap_byte & 0x80 != 0) {
                        var shift: u5 = 7;
                        while (true) {
                            const b = try reader.readByte();
                            type_idx |= @as(u32, b & 0x7F) << shift;
                            if (b & 0x80 == 0) break;
                            shift +|= 7;
                        }
                    }
                    if (type_idx >= type_count) return error.InvalidValType;
                    break :blk .funcref; // concrete typed func ref → funcref
                },
            };
        },
        else => return error.InvalidValType,
    };
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
    const start_pos = reader.pos;
    const opcode = try reader.readByte();
    // Try to parse as a single-instruction init expression first
    const simple: ?types.InitExpr = switch (opcode) {
        0x41 => .{ .i32_const = try reader.readI32() },
        0x42 => .{ .i64_const = try reader.readI64() },
        0x43 => .{ .f32_const = try reader.readF32() },
        0x44 => .{ .f64_const = try reader.readF64() },
        0x23 => .{ .global_get = try reader.readU32() },
        0xD0 => .{ .ref_null = try readValType(reader) },
        0xD2 => .{ .ref_func = try reader.readU32() },
        else => null,
    };
    const end = try reader.readByte();
    if (end == 0x0B) {
        // Simple single-instruction expression
        if (simple) |s| return s;
        // Unknown single opcode is invalid
        return error.InvalidInitExpr;
    }
    // Compound expression: validate and scan forward to end
    // Only opcodes valid in constant expressions are accepted (spec §3.3.10)
    if (simple == null) return error.InvalidInitExpr; // first opcode must be valid
    reader.pos = start_pos;
    var stack_depth: i32 = 0;
    while (reader.pos < reader.data.len) {
        const b = try reader.readByte();
        switch (b) {
            0x0B => {
                if (stack_depth != 1) return error.InvalidInitExpr;
                return .{ .bytecode = reader.data[start_pos .. reader.pos - 1] };
            },
            // Valid const expr opcodes that push 1 value
            0x41 => { _ = try reader.readI32(); stack_depth += 1; },
            0x42 => { _ = try reader.readI64(); stack_depth += 1; },
            0x43 => { _ = try reader.readBytes(4); stack_depth += 1; },
            0x44 => { _ = try reader.readBytes(8); stack_depth += 1; },
            0x23 => { _ = try reader.readU32(); stack_depth += 1; },
            0xD0 => { _ = try readValType(reader); stack_depth += 1; },
            0xD2 => { _ = try reader.readU32(); stack_depth += 1; },
            // Valid const expr binary ops: pop 2, push 1
            0x6A, 0x6B, 0x6C, // i32.add, i32.sub, i32.mul
            0x7C, 0x7D, 0x7E, // i64.add, i64.sub, i64.mul
            => { stack_depth -= 1; },
            // Any other opcode is invalid in a constant expression
            else => return error.InvalidInitExpr,
        }
    }
    return error.InvalidInitExpr;
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

fn parseImportSection(reader: *BinaryReader, allocator: std.mem.Allocator, type_count: u32) LoadError![]const types.ImportDesc {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    var imports_list: std.ArrayList(types.ImportDesc) = .empty;
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const module_name = try reader.readName();
        const field_name = try reader.readName();
        const kind_byte = try reader.readByte();

        // Tag imports (0x04) from exception handling — skip
        if (kind_byte == 0x04) {
            _ = try reader.readByte(); // tag attribute
            _ = try reader.readU32(); // type index
            continue;
        }

        const kind: types.ExternalKind = switch (kind_byte) {
            0x00 => .function,
            0x01 => .table,
            0x02 => .memory,
            0x03 => .global,
            else => return error.InvalidImportKind,
        };

        var imp = types.ImportDesc{
            .module_name = module_name,
            .field_name = field_name,
            .kind = kind,
        };

        switch (kind) {
            .function => imp.func_type_idx = try reader.readU32(),
            .table => imp.table_type = try readTableType(reader, type_count),
            .memory => imp.memory_type = try readMemoryType(reader),
            .global => imp.global_type = try readGlobalType(reader),
        }
        imports_list.append(allocator, imp) catch return error.InvalidImportKind;
    }
    return imports_list.toOwnedSlice(allocator) catch return error.InvalidImportKind;
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

fn parseTableSection(reader: *BinaryReader, allocator: std.mem.Allocator, type_count: u32) LoadError![]const types.TableType {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    const tables = try allocator.alloc(types.TableType, count);
    for (tables) |*t| {
        t.* = try readTableType(reader, type_count);
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
    // Allocate max size, then shrink if we skip tag exports
    var exports_list: std.ArrayList(types.ExportDesc) = .empty;
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const name = try reader.readName();
        const kind_byte = try reader.readByte();
        const index = try reader.readU32();
        const kind: ?types.ExternalKind = switch (kind_byte) {
            0x00 => .function,
            0x01 => .table,
            0x02 => .memory,
            0x03 => .global,
            0x04 => null, // tag export (exception handling) — skip
            else => return error.InvalidExportKind,
        };
        if (kind) |k| {
            exports_list.append(allocator, .{ .name = name, .kind = k, .index = index }) catch
                return error.InvalidExportKind;
        }
    }
    return exports_list.toOwnedSlice(allocator) catch return error.InvalidExportKind;
}

fn parseElementSection(reader: *BinaryReader, allocator: std.mem.Allocator, type_count: u32) LoadError![]const types.ElemSegment {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    const elements = try allocator.alloc(types.ElemSegment, count);
    for (elements) |*elem| {
        const flags = try reader.readU32();
        if (flags > 7) return error.InvalidElemSegment;
        const is_passive = (flags & 1) != 0 and (flags & 2) == 0;
        const is_declarative = (flags & 1) != 0 and (flags & 2) != 0;
        const has_table_idx = (flags & 2) != 0 and (flags & 1) == 0; // active with explicit table
        const has_exprs = (flags & 4) != 0;

        // Table index (only for active segments with flag bit 1 clear, bit 2 set = flags 2 or 6)
        var table_idx: u32 = 0;
        if (has_table_idx) {
            table_idx = try reader.readU32();
        }

        // Offset expression (only for active segments: flags 0,2,4,6)
        var offset: ?types.InitExpr = null;
        if (!is_passive and !is_declarative) {
            offset = try parseInitExpr(reader);
        }

        if (has_exprs) {
            // flags 4,5,6,7: vec(expr) — reftype byte only for flags 5,6,7
            var kind: types.ElemSegment.ElemKind = .func_ref;
            if (flags != 4) {
                const ref_byte = try reader.readByte();
                kind = switch (ref_byte) {
                    0x70 => .func_ref,
                    0x6F => .extern_ref,
                    // Typed reference: ref null/ref + heaptype
                    0x63, 0x64 => blk: {
                        const ht = try reader.readByte();
                        if (ht == 0x6F or ht == 0x72) break :blk .extern_ref;
                        // func, nofunc → funcref
                        if (ht == 0x70 or ht == 0x73) break :blk .func_ref;
                        // Concrete type index — validate and consume LEB128
                        var type_idx: u32 = ht & 0x7F;
                        if (ht & 0x80 != 0) {
                            var shift: u5 = 7;
                            while (true) {
                                const b = try reader.readByte();
                                type_idx |= @as(u32, b & 0x7F) << shift;
                                if (b & 0x80 == 0) break;
                                shift +|= 7;
                            }
                        }
                        if (type_idx >= type_count) return error.InvalidValType;
                        break :blk .func_ref;
                    },
                    else => return error.InvalidElemSegment,
                };
            }
            const num_elems = try reader.readU32();
            var func_indices_list: std.ArrayList(?u32) = .empty;
            var j: u32 = 0;
            while (j < num_elems) : (j += 1) {
                const expr = try parseInitExpr(reader);
                switch (expr) {
                    .ref_func => |fidx| {
                        try func_indices_list.append(allocator, fidx);
                    },
                    .ref_null => |vt| {
                        if (kind == .func_ref and vt != .funcref) return error.TypeMismatch;
                        if (kind == .extern_ref and vt != .externref) return error.TypeMismatch;
                        try func_indices_list.append(allocator, null);
                    },
                    else => return error.TypeMismatch,
                }
            }
            elem.* = .{
                .table_idx = table_idx,
                .offset = offset,
                .kind = kind,
                .func_indices = try func_indices_list.toOwnedSlice(allocator),
                .is_passive = is_passive,
                .is_declarative = is_declarative,
            };
        } else {
            // flags 0,1,2,3: elemkind + vec(funcidx)
            if (flags & 3 != 0) {
                // Flags 1,2,3 have an explicit elemkind byte
                _ = try reader.readByte(); // elemkind (0x00 = funcref)
            }
            const num_elems = try reader.readU32();
            const func_indices: []const ?u32 = if (num_elems > 0) blk: {
                const fi = try allocator.alloc(?u32, num_elems);
                for (fi) |*f| f.* = try reader.readU32();
                break :blk fi;
            } else &[_]?u32{};
            elem.* = .{
                .table_idx = table_idx,
                .offset = offset,
                .kind = .func_ref,
                .func_indices = func_indices,
                .is_passive = is_passive,
                .is_declarative = is_declarative,
            };
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

        // Function body must end with END (0x0B) opcode
        if (code.len == 0 or code[code.len - 1] != 0x0B) return error.InvalidSectionSize;

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

        // Enforce section ordering (custom sections and proposal sections can appear anywhere)
        if (section_id != 0) {
            // Skip known proposal sections (tag section = 13) without enforcing order
            if (section_id > @intFromEnum(types.SectionId.data_count)) {
                const section_start = reader.pos;
                if (section_start + section_size > reader.data.len) return error.InvalidSectionSize;
                if (section_id == 13) {
                    reader.pos = section_start + section_size;
                    continue;
                }
                return error.MalformedSectionId;
            }
            if (last_section_id) |last| {
                if (section_id <= last) return error.InvalidSectionOrder;
            }
            last_section_id = section_id;
        }

        const section_start = reader.pos;
        if (section_start + section_size > reader.data.len) return error.InvalidSectionSize;

        switch (@as(types.SectionId, @enumFromInt(section_id))) {
            .custom => {
                // Parse the custom section name to validate UTF-8
                _ = try reader.readName();
                if (reader.pos > section_start + section_size) return error.InvalidSectionSize;
                reader.pos = section_start + section_size;
            },
                .type => module.types = try parseTypeSection(&reader, allocator),
                .import => {
                    const tc: u32 = @intCast(module.types.len);
                    module.imports = try parseImportSection(&reader, allocator, tc);
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
                .table => module.tables = try parseTableSection(&reader, allocator, @intCast(module.types.len)),
                .memory => module.memories = try parseMemorySection(&reader, allocator),
                .global => module.globals = try parseGlobalSection(&reader, allocator),
                .@"export" => module.exports = try parseExportSection(&reader, allocator),
                .start => module.start_function = try reader.readU32(),
                .element => module.elements = try parseElementSection(&reader, allocator, @intCast(module.types.len)),
                .code => module.functions = try parseCodeSection(&reader, func_type_indices, module.types, allocator),
                .data => module.data_segments = try parseDataSection(&reader, allocator),
                .data_count => module.data_count = try reader.readU32(),
            }

            // Verify we consumed exactly section_size bytes
            if (reader.pos != section_start + section_size) return error.InvalidSectionSize;
    }

    // Function section count must match code section count
    if (func_type_indices.len != module.functions.len) return error.InvalidSectionSize;

    // Validate function type indices from function section
    for (func_type_indices) |type_idx| {
        if (type_idx >= module.types.len) return error.InvalidTypeIndex;
    }

    try validateModule(&module);

    return module;
}

/// Post-parse module validation per WebAssembly MVP spec.
fn validateModule(module: *const types.WasmModule) LoadError!void {
    const total_funcs: u32 = module.import_function_count + @as(u32, @intCast(module.functions.len));
    const total_tables: u32 = module.import_table_count + @as(u32, @intCast(module.tables.len));
    const total_memories: u32 = module.import_memory_count + @as(u32, @intCast(module.memories.len));
    const total_globals: u32 = module.import_global_count + @as(u32, @intCast(module.globals.len));

    // Validate memory limits
    for (module.memories) |mem| {
        try validateMemoryLimits(mem.limits);
    }

    // Validate table limits
    for (module.tables) |table| {
        if (table.limits.max) |max| {
            if (table.limits.min > max) return error.InvalidLimits;
        }
    }

    // Validate import types and limits
    for (module.imports) |imp| {
        switch (imp.kind) {
            .function => {
                if (imp.func_type_idx) |idx| {
                    if (idx >= module.types.len) return error.InvalidTypeIndex;
                }
            },
            .memory => {
                if (imp.memory_type) |mem| {
                    try validateMemoryLimits(mem.limits);
                }
            },
            .table => {
                if (imp.table_type) |table| {
                    if (table.limits.max) |max| {
                        if (table.limits.min > max) return error.InvalidLimits;
                    }
                }
            },
            .global => {},
        }
    }

    // Validate export index bounds
    for (module.exports) |exp| {
        switch (exp.kind) {
            .function => {
                if (exp.index >= total_funcs) return error.UnknownFunction;
            },
            .table => {
                if (exp.index >= total_tables) return error.UnknownTable;
            },
            .memory => {
                if (exp.index >= total_memories) return error.UnknownMemory;
            },
            .global => {
                if (exp.index >= total_globals) return error.UnknownGlobal;
            },
        }
    }

    // Duplicate export names
    for (module.exports, 0..) |a, i| {
        for (module.exports[i + 1 ..]) |b| {
            if (std.mem.eql(u8, a.name, b.name)) return error.DuplicateExportName;
        }
    }

    // Start function validation
    if (module.start_function) |start_idx| {
        if (start_idx >= total_funcs) return error.InvalidStartFunction;
        const func_type = getFuncType(module, start_idx) orelse return error.InvalidStartFunction;
        if (func_type.params.len != 0 or func_type.results.len != 0) return error.InvalidStartFunction;
    }

    // Data count section must match data segment count
    if (module.data_count) |dc| {
        if (dc != @as(u32, @intCast(module.data_segments.len))) return error.DataCountMismatch;
    }

    // Validate data segment memory indices
    for (module.data_segments) |seg| {
        if (!seg.is_passive and seg.memory_idx >= total_memories) return error.UnknownMemory;
    }

    // Validate global init expressions - global_get must reference imported globals
    // Also validate init expression type matches global type
    // Validate global init expressions
    for (module.globals, 0..) |g, gi| {
        switch (g.init_expr) {
            .global_get => |idx| {
                // Extended-const: global.get can reference any preceding immutable global
                const max_idx = module.import_global_count + @as(u32, @intCast(gi));
                if (idx >= max_idx) return error.UnknownGlobal;
                if (idx < module.import_global_count) {
                    if (getImportGlobalType(module, idx)) |gt| {
                        if (gt.mutability == .mutable) return error.TypeMismatch;
                        if (gt.val_type != g.global_type.val_type) return error.TypeMismatch;
                    }
                } else {
                    const local_idx = idx - module.import_global_count;
                    if (local_idx < module.globals.len) {
                        if (module.globals[local_idx].global_type.mutability == .mutable) return error.TypeMismatch;
                    }
                }
            },
            .i32_const => { if (g.global_type.val_type != .i32) return error.TypeMismatch; },
            .i64_const => { if (g.global_type.val_type != .i64) return error.TypeMismatch; },
            .f32_const => { if (g.global_type.val_type != .f32) return error.TypeMismatch; },
            .f64_const => { if (g.global_type.val_type != .f64) return error.TypeMismatch; },
            .ref_null => |rt| {
                if (g.global_type.val_type != rt) return error.TypeMismatch;
            },
            .ref_func => |fidx| {
                if (g.global_type.val_type != .funcref) return error.TypeMismatch;
                if (fidx >= total_funcs) return error.UnknownFunction;
            },
            .bytecode => {}, // compound expressions validated at evaluation time
        }
    }

    // Validate data segment offset expressions must evaluate to i32
    for (module.data_segments) |seg| {
        if (!seg.is_passive) {
            switch (seg.offset) {
                .i32_const => {},
                .global_get => |idx| {
                    // Extended-const: allow any immutable global
                    if (idx >= total_globals) return error.UnknownGlobal;
                    if (idx < module.import_global_count) {
                        if (getImportGlobalType(module, idx)) |gt| {
                            if (gt.mutability == .mutable) return error.TypeMismatch;
                            if (gt.val_type != .i32) return error.TypeMismatch;
                        }
                    } else {
                        const local_idx = idx - module.import_global_count;
                        if (local_idx < module.globals.len) {
                            if (module.globals[local_idx].global_type.mutability == .mutable) return error.TypeMismatch;
                        }
                    }
                },
                .bytecode => {}, // compound offset expression validated at evaluation
                else => return error.TypeMismatch,
            }
        }
    }

    // Validate element segment table and function indices
    for (module.elements) |elem| {
        if (!elem.is_passive and !elem.is_declarative) {
            if (elem.table_idx >= total_tables) return error.UnknownTable;
            // Validate elem kind matches table elem type
            const table_elem_type = if (elem.table_idx < module.import_table_count)
                getImportTableElemType(module, elem.table_idx)
            else blk: {
                const local_idx = elem.table_idx - module.import_table_count;
                break :blk if (local_idx < module.tables.len) module.tables[local_idx].elem_type else null;
            };
            if (table_elem_type) |tet| {
                if (elem.kind == .func_ref and tet != .funcref) return error.TypeMismatch;
                if (elem.kind == .extern_ref and tet != .externref) return error.TypeMismatch;
            }
            // Validate offset expression type (must be i32)
            if (elem.offset) |offset| {
                switch (offset) {
                    .i32_const => {},
                    .global_get => |idx| {
                        // Extended-const: allow any immutable global
                        if (idx >= total_globals) return error.UnknownGlobal;
                        if (idx < module.import_global_count) {
                            if (getImportGlobalType(module, idx)) |gt| {
                                if (gt.mutability == .mutable) return error.TypeMismatch;
                                if (gt.val_type != .i32) return error.TypeMismatch;
                            }
                        } else {
                            const local_idx = idx - module.import_global_count;
                            if (local_idx < module.globals.len) {
                                if (module.globals[local_idx].global_type.mutability == .mutable) return error.TypeMismatch;
                            }
                        }
                    },
                    else => return error.TypeMismatch,
                }
            }
        }
        for (elem.func_indices) |mfidx| {
            if (mfidx) |fidx| {
                if (fidx >= total_funcs) return error.UnknownFunction;
            }
        }
    }

    // Validate function bodies (alignment, index bounds)
    for (module.functions) |func| {
        const total_locals = @as(u32, @intCast(func.func_type.params.len)) + func.local_count;
        try validateFunctionBody(func.code, module.types.len, total_funcs, total_tables, total_globals, total_locals, module.data_count != null);
    }

    // Type-stack validation for each function body (skip for imports w/ 0 local funcs)
    if (module.functions.len > 0) {
        for (module.functions) |func| {
            try validateFunctionTypes(module, &func);
        }
    }

    // Validate ref.func references are "declared" (in element segments, exports, or globals)
    try validateDeclaredFuncRefs(module, total_funcs);
}

fn validateMemoryLimits(limits: types.Limits) LoadError!void {
    if (limits.min > 65536) return error.InvalidLimits;
    if (limits.max) |max| {
        if (max > 65536) return error.InvalidLimits;
        if (limits.min > max) return error.InvalidLimits;
    }
}

/// Check that every ref.func index in function bodies references a "declared" function.
/// A function is declared if it appears in an element segment, an export, or the start function.
fn validateDeclaredFuncRefs(module: *const types.WasmModule, total_funcs: u32) LoadError!void {
    // Build declared set as a bit set (max 64K functions for stack-based approach)
    const max_track: u32 = if (total_funcs <= 8192) total_funcs else 8192;
    var declared_buf: [8192]bool = undefined;
    const declared = declared_buf[0..max_track];
    @memset(declared, false);

    // Functions in element segments are declared
    for (module.elements) |elem| {
        for (elem.func_indices) |mfidx| {
            if (mfidx) |fidx| {
                if (fidx < max_track) declared[fidx] = true;
            }
        }
    }
    // Exported functions are declared
    for (module.exports) |exp| {
        if (exp.kind == .function and exp.index < max_track) declared[exp.index] = true;
    }
    // Functions referenced by ref.func in global init expressions are declared
    for (module.globals) |g| {
        switch (g.init_expr) {
            .ref_func => |fidx| {
                if (fidx < max_track) declared[fidx] = true;
            },
            else => {},
        }
    }
    // Functions referenced in element segment init expressions are declared
    for (module.elements) |elem| {
        if (elem.offset) |offset| {
            switch (offset) {
                .ref_func => |fidx| {
                    if (fidx < max_track) declared[fidx] = true;
                },
                else => {},
            }
        }
    }

    // Scan function bodies for ref.func instructions
    for (module.functions) |func| {
        try checkRefFuncDeclared(func.code, declared);
    }
}

/// Skip a block type immediate: 0x40 (void), single-byte valtype, ref type (0x63/0x64 + heaptype), or type index (LEB128).
fn skipBlockTypeImm(code: []const u8, i: *usize) void {
    if (i.* >= code.len) return;
    const bt = code[i.*];
    if (bt == 0x40 or bt == 0x7F or bt == 0x7E or bt == 0x7D or bt == 0x7C or bt == 0x70 or bt == 0x6F) {
        i.* += 1;
    } else if (bt == 0x63 or bt == 0x64) {
        i.* += 1; // skip ref prefix
        if (i.* >= code.len) return;
        const ht = code[i.*];
        if (ht == 0x70 or ht == 0x6F or ht == 0x73 or ht == 0x72) {
            i.* += 1; // known heap type
        } else {
            // Type index LEB128
            const r = leb128_mod.readUnsigned(u32, code[i.*..]) catch return;
            i.* += r.bytes_read;
        }
    } else {
        const r = leb128_mod.readSigned(i64, code[i.*..]) catch return;
        i.* += r.bytes_read;
    }
}

fn checkRefFuncDeclared(code: []const u8, declared: []const bool) LoadError!void {
    var i: usize = 0;
    while (i < code.len) {
        const op = code[i];
        i += 1;
        switch (op) {
            // Skip opcodes with known immediate sizes
            0x02, 0x03, 0x04 => { // block/loop/if: blocktype
                skipBlockTypeImm(code, &i);
            },
            0x0C, 0x0D, 0xD4, 0xD6 => { const r = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r.bytes_read; },
            0x0E => { // br_table
                const cr = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += cr.bytes_read;
                var j: u32 = 0;
                while (j <= cr.value) : (j += 1) { const lr = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += lr.bytes_read; }
            },
            0x10, 0x12, 0x14, 0x15 => { const r = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r.bytes_read; },
            0x11, 0x13 => {
                const r1 = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r1.bytes_read;
                const r2 = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r2.bytes_read;
            },
            0x1C => { const cr = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += cr.bytes_read; i += cr.value; },
            0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26 => { const r = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r.bytes_read; },
            0x28...0x3E => { // memory load/store
                const r1 = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r1.bytes_read;
                const r2 = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r2.bytes_read;
            },
            0x3F, 0x40 => { const r = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r.bytes_read; },
            0x41 => { const r = leb128_mod.readSigned(i32, code[i..]) catch return; i += r.bytes_read; },
            0x42 => { const r = leb128_mod.readSigned(i64, code[i..]) catch return; i += r.bytes_read; },
            0x43 => i += 4,
            0x44 => i += 8,
            0xD0 => { // ref.null: skip heap type
                if (i < code.len) {
                    const ht = code[i];
                    i += 1;
                    // If heap type is a type index (not a known abstract type), consume LEB128
                    if (ht != 0x70 and ht != 0x6F and ht != 0x73 and ht != 0x72 and ht & 0x80 != 0) {
                        while (i < code.len) {
                            const b = code[i];
                            i += 1;
                            if (b & 0x80 == 0) break;
                        }
                    }
                }
            },
            0xD2 => { // ref.func - check declared
                const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += r.bytes_read;
                if (r.value < declared.len) {
                    if (!declared[r.value]) return error.UndeclaredFuncRef;
                }
            },
            0xFC => {
                const sr = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += sr.bytes_read;
                switch (sr.value) {
                    0...7 => {},
                    8, 10, 12, 14 => { const a = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += a.bytes_read; const b = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += b.bytes_read; },
                    9, 11, 13, 15, 16, 17 => { const a = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += a.bytes_read; },
                    else => {},
                }
            },
            else => {},
        }
    }
}

fn getFuncType(module: *const types.WasmModule, func_idx: u32) ?types.FuncType {
    if (func_idx < module.import_function_count) {
        var import_func_i: u32 = 0;
        for (module.imports) |imp| {
            if (imp.kind == .function) {
                if (import_func_i == func_idx) {
                    const type_idx = imp.func_type_idx orelse return null;
                    if (type_idx >= module.types.len) return null;
                    return module.types[type_idx];
                }
                import_func_i += 1;
            }
        }
        return null;
    }
    const local_idx = func_idx - module.import_function_count;
    if (local_idx < module.functions.len) {
        return module.functions[local_idx].func_type;
    }
    return null;
}

/// Validate a function body: alignment, index bounds for locals/globals/funcs/types.
fn validateFunctionBody(
    code: []const u8,
    num_types: usize,
    total_funcs: u32,
    total_tables: u32,
    total_globals: u32,
    total_locals: u32,
    has_data_count: bool,
) LoadError!void {
    var i: usize = 0;
    while (i < code.len) {
        const op = code[i];
        i += 1;

        const max_align: ?u8 = switch (op) {
            0x28, 0x2A, 0x36, 0x38 => 2, // i32.load, f32.load, i32.store, f32.store
            0x29, 0x2B, 0x37, 0x39 => 3, // i64.load, f64.load, i64.store, f64.store
            0x2C, 0x2D, 0x30, 0x31, 0x3A, 0x3C => 0, // *load8*, *store8
            0x2E, 0x2F, 0x32, 0x33, 0x3B, 0x3D => 1, // *load16*, *store16
            0x34, 0x35, 0x3E => 2, // i64.load32*, i64.store32
            else => null,
        };

        if (max_align) |ma| {
            const align_result = leb128_mod.readUnsigned(u32, code[i..]) catch return error.InvalidAlignment;
            i += align_result.bytes_read;
            if (align_result.value > ma) return error.InvalidAlignment;
            const offset_result = leb128_mod.readUnsigned(u32, code[i..]) catch return error.InvalidSectionSize;
            i += offset_result.bytes_read;
            continue;
        }

        switch (op) {
            // Block/loop/if: blocktype
            0x02, 0x03, 0x04 => {
                skipBlockTypeImm(code, &i);
            },

            // br, br_if: labelidx
            0x0C, 0x0D => {
                const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += r.bytes_read;
            },

            // br_table
            0x0E => {
                const count_r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += count_r.bytes_read;
                var j: u32 = 0;
                while (j <= count_r.value) : (j += 1) {
                    const lr = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                    i += lr.bytes_read;
                }
            },

            // call, return_call: funcidx
            0x10, 0x12 => {
                const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += r.bytes_read;
                if (r.value >= total_funcs) return error.UnknownFunction;
            },

            // call_indirect, return_call_indirect: typeidx + tableidx
            0x11, 0x13 => {
                const r1 = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += r1.bytes_read;
                if (r1.value >= num_types) return error.InvalidTypeIndex;
                const r2 = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += r2.bytes_read;
                if (r2.value >= total_tables) return error.UnknownTable;
            },

            // select_t: vec(valtype)
            0x1C => {
                const count_r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += count_r.bytes_read;
                i += count_r.value; // each valtype is one byte
            },

            // local.get, local.set, local.tee
            0x20, 0x21, 0x22 => {
                const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += r.bytes_read;
                if (r.value >= total_locals) return error.UnknownFunction; // reuse error for unknown local
            },

            // global.get, global.set
            0x23, 0x24 => {
                const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += r.bytes_read;
                if (r.value >= total_globals) return error.UnknownGlobal;
            },

            // table.get, table.set: tableidx
            0x25, 0x26 => {
                const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += r.bytes_read;
                if (r.value >= total_tables) return error.UnknownTable;
            },

            // memory.size, memory.grow: memidx (u32 LEB)
            0x3F, 0x40 => {
                const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += r.bytes_read;
            },

            // i32.const
            0x41 => {
                const r = leb128_mod.readSigned(i32, code[i..]) catch return;
                i += r.bytes_read;
            },

            // i64.const
            0x42 => {
                const r = leb128_mod.readSigned(i64, code[i..]) catch return;
                i += r.bytes_read;
            },

            // f32.const
            0x43 => i += 4,

            // f64.const
            0x44 => i += 8,

            // 0xFC prefix
            0xFC => {
                const sub_r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += sub_r.bytes_read;
                switch (sub_r.value) {
                    0...7 => {},
                    8 => { // memory.init: dataidx + memidx
                        const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                        i += r.bytes_read;
                        const m = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                        i += m.bytes_read;
                        if (!has_data_count) return error.DataCountRequired;
                    },
                    9 => { // data.drop
                        const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                        i += r.bytes_read;
                        if (!has_data_count) return error.DataCountRequired;
                    },
                    10 => { // memory.copy: memidx + memidx
                        const m1 = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                        i += m1.bytes_read;
                        const m2 = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                        i += m2.bytes_read;
                    },
                    11 => { // memory.fill: memidx
                        const m = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                        i += m.bytes_read;
                    },
                    12 => { // table.init: elemidx + tableidx
                        const r1 = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                        i += r1.bytes_read;
                        const r2 = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                        i += r2.bytes_read;
                        if (r2.value >= total_tables) return error.UnknownTable;
                    },
                    13 => { // elem.drop
                        const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                        i += r.bytes_read;
                    },
                    14 => { // table.copy: tableidx + tableidx
                        const r1 = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                        i += r1.bytes_read;
                        if (r1.value >= total_tables) return error.UnknownTable;
                        const r2 = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                        i += r2.bytes_read;
                        if (r2.value >= total_tables) return error.UnknownTable;
                    },
                    15, 16, 17 => { // table.grow, table.size, table.fill
                        const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                        i += r.bytes_read;
                        if (r.value >= total_tables) return error.UnknownTable;
                    },
                    else => {},
                }
            },

            // ref.null: heaptype (may be multi-byte for typed refs)
            0xD0 => {
                if (i < code.len) {
                    const ht = code[i];
                    i += 1;
                    if (ht != 0x70 and ht != 0x6F and ht != 0x73 and ht != 0x72 and ht & 0x80 != 0) {
                        while (i < code.len) {
                            if (code[i] & 0x80 == 0) { i += 1; break; }
                            i += 1;
                        }
                    }
                }
            },

            // ref.func: funcidx (u32 LEB)
            0xD2 => {
                const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += r.bytes_read;
                if (r.value >= total_funcs) return error.UnknownFunction;
            },

            // ref.is_null, ref.as_non_null, ref.eq have no immediate
            0xD1, 0xD3, 0xD5 => {},

            // FD prefix (SIMD) — skip sub-opcode and potential immediates
            0xFD => {
                const sr = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += sr.bytes_read;
                // SIMD ops may have additional immediates (e.g., lane indices);
                // We skip validation of SIMD details here.
            },
            // FE prefix (threads/atomics) — skip sub-opcode + memarg
            0xFE => {
                const sr = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += sr.bytes_read;
            },

            // All valid opcodes with no immediates (numerics, control, etc.)
            // 0x00=unreachable, 0x01=nop, 0x05=else, 0x0B=end, 0x0F=return
            // 0x1A=drop, 0x1B=select
            // 0x45..0xC4=numeric/comparison/conversion/sign-ext ops
            0x00, 0x01, 0x05, 0x0B, 0x0F, 0x1A, 0x1B => {},
            0x45...0xC4 => {},

            // call_ref, return_call_ref: typeidx (not yet fully validated)
            0x14, 0x15 => {
                const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += r.bytes_read;
            },

            // br_on_null, br_on_non_null: labelidx
            0xD4, 0xD6 => {
                const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += r.bytes_read;
            },

            // Opcodes in ranges 0x06-0x0A, 0x14-0x19, 0x1D-0x1F, 0x27,
            // 0xC5-0xCF, 0xD3-0xFB, 0xFF are reserved/illegal
            else => return error.IllegalOpcode,
        }
    }
}

/// Get the type of an imported global by its index among imported globals.
fn getImportGlobalType(module: *const types.WasmModule, idx: u32) ?types.GlobalType {
    if (idx >= module.import_global_count) return null;
    var gi: u32 = 0;
    for (module.imports) |imp| {
        if (imp.kind == .global) {
            if (gi == idx) return imp.global_type;
            gi += 1;
        }
    }
    return null;
}

fn getImportTableElemType(module: *const types.WasmModule, idx: u32) ?types.ValType {
    if (idx >= module.import_table_count) return null;
    var ti: u32 = 0;
    for (module.imports) |imp| {
        if (imp.kind == .table) {
            if (ti == idx) {
                if (imp.table_type) |tt| return tt.elem_type;
                return null;
            }
            ti += 1;
        }
    }
    return null;
}

/// Get the full GlobalType (including mutability) for any global index.
fn getFullGlobalType(module: *const types.WasmModule, idx: u32) ?types.GlobalType {
    if (idx < module.import_global_count) {
        return getImportGlobalType(module, idx);
    }
    const local_idx = idx - module.import_global_count;
    if (local_idx < module.globals.len) return module.globals[local_idx].global_type;
    return null;
}

// ─── Type-stack validation (WebAssembly spec §3.3) ─────────────────────────

const VT = types.ValType;

/// Control-frame entry on the control stack.
const CtrlFrame = struct {
    kind: enum { block, loop, @"if", function },
    start_height: u32,
    start_types: []const VT, // input types (used as label types for loop)
    end_types: []const VT, // result types (used as label types for block/if/function)
    has_else: bool = false,
    unreachable_flag: bool = false,
};

/// Block type: both params and results.
const BlockType = struct {
    params: []const VT = &.{},
    results: []const VT,
};

fn readBlockType(code: []const u8, pos: *usize, module_types: []const types.FuncType) LoadError!BlockType {
    if (pos.* >= code.len) return .{ .results = &.{} };
    const bt = code[pos.*];
    if (bt == 0x40) { pos.* += 1; return .{ .results = &.{} }; }
    if (bt == 0x7F or bt == 0x7E or bt == 0x7D or bt == 0x7C or bt == 0x70 or bt == 0x6F) {
        pos.* += 1;
        return switch (bt) {
            0x7F => .{ .results = &[_]VT{.i32} },
            0x7E => .{ .results = &[_]VT{.i64} },
            0x7D => .{ .results = &[_]VT{.f32} },
            0x7C => .{ .results = &[_]VT{.f64} },
            0x70 => .{ .results = &[_]VT{.funcref} },
            0x6F => .{ .results = &[_]VT{.externref} },
            else => .{ .results = &.{} },
        };
    }
    // Typed reference block types: ref null/ref + heaptype
    if (bt == 0x63 or bt == 0x64) {
        pos.* += 1; // consume 0x63/0x64
        if (pos.* >= code.len) return error.TypeMismatch;
        const ht = code[pos.*];
        if (ht == 0x70 or ht == 0x73) { pos.* += 1; return .{ .results = &[_]VT{.funcref} }; }
        if (ht == 0x6F or ht == 0x72) { pos.* += 1; return .{ .results = &[_]VT{.externref} }; }
        // Concrete type index (LEB128) — validate and treat as funcref result
        const tir = leb128_mod.readUnsigned(u32, code[pos.*..]) catch return error.TypeMismatch;
        pos.* += tir.bytes_read;
        if (tir.value >= module_types.len) return error.InvalidValType;
        return .{ .results = &[_]VT{.funcref} };
    }
    const r = leb128_mod.readSigned(i64, code[pos.*..]) catch return error.TypeMismatch;
    pos.* += r.bytes_read;
    if (r.value < 0) return error.TypeMismatch;
    const idx: usize = @intCast(r.value);
    if (idx < module_types.len) return .{ .params = module_types[idx].params, .results = module_types[idx].results };
    return error.InvalidTypeIndex;
}

fn readU32Leb(code: []const u8, pos: *usize) u32 {
    const r = leb128_mod.readUnsigned(u32, code[pos.*..]) catch return 0;
    pos.* += r.bytes_read;
    return r.value;
}

fn readI32Leb(code: []const u8, pos: *usize) i32 {
    const r = leb128_mod.readSigned(i32, code[pos.*..]) catch return 0;
    pos.* += r.bytes_read;
    return r.value;
}

fn readI64Leb(code: []const u8, pos: *usize) i64 {
    const r = leb128_mod.readSigned(i64, code[pos.*..]) catch return 0;
    pos.* += r.bytes_read;
    return r.value;
}

fn skipMemImm(code: []const u8, pos: *usize) void {
    _ = readU32Leb(code, pos);
    _ = readU32Leb(code, pos);
}

fn pushType(stack: []VT, sp: *u32, t: VT) void {
    if (sp.* < stack.len) { stack[sp.*] = t; sp.* += 1; }
}

fn popExpect(stack: []VT, sp: *u32, expected: VT, cf: ?*CtrlFrame) bool {
    if (sp.* == 0 or (cf != null and sp.* <= cf.?.start_height)) {
        return cf != null and cf.?.unreachable_flag;
    }
    sp.* -= 1;
    return stack[sp.*] == expected;
}

fn popAny(stack: []VT, sp: *u32, cf: ?*CtrlFrame) ?VT {
    if (sp.* == 0 or (cf != null and sp.* <= cf.?.start_height)) return null;
    sp.* -= 1;
    return stack[sp.*];
}

fn checkStackEnd(cf: *const CtrlFrame, stack: []const VT, sp: u32) bool {
    if (cf.unreachable_flag) {
        const expected = cf.end_types;
        // In unreachable mode: values above start_height are concrete and must match.
        // Extra values above start_height + expected.len are invalid.
        if (sp > cf.start_height + expected.len) return false;
        // Check concrete values on stack (above start_height) against expected types.
        // Values below start_height are polymorphic (match anything).
        for (expected, 0..) |t, j| {
            const stack_idx = cf.start_height + @as(u32, @intCast(j));
            if (stack_idx >= sp) continue; // below concrete stack → polymorphic, OK
            if (stack[stack_idx] != t) return false;
        }
        return true;
    }
    const expected = cf.end_types;
    if (sp != cf.start_height + expected.len) return false;
    for (expected, 0..) |t, j| {
        if (stack[cf.start_height + @as(u32, @intCast(j))] != t) return false;
    }
    return true;
}

fn doLoad(stack: []VT, sp: *u32, result: VT, cf: ?*CtrlFrame) LoadError!void {
    if (!popExpect(stack, sp, .i32, cf)) return error.TypeMismatch;
    pushType(stack, sp, result);
}

fn doStore(stack: []VT, sp: *u32, val_type: VT, cf: ?*CtrlFrame) LoadError!void {
    if (!popExpect(stack, sp, val_type, cf)) return error.TypeMismatch;
    if (!popExpect(stack, sp, .i32, cf)) return error.TypeMismatch;
}

fn doUnop(stack: []VT, sp: *u32, input: VT, output: VT, cf: ?*CtrlFrame) LoadError!void {
    if (!popExpect(stack, sp, input, cf)) return error.TypeMismatch;
    pushType(stack, sp, output);
}

fn doBinop(stack: []VT, sp: *u32, operand: VT, result: VT, cf: ?*CtrlFrame) LoadError!void {
    if (!popExpect(stack, sp, operand, cf)) return error.TypeMismatch;
    if (!popExpect(stack, sp, operand, cf)) return error.TypeMismatch;
    pushType(stack, sp, result);
}

fn getGlobalType(module: *const types.WasmModule, idx: u32) ?VT {
    if (idx < module.import_global_count) {
        var gi: u32 = 0;
        for (module.imports) |imp| {
            if (imp.kind == .global) {
                if (gi == idx) return if (imp.global_type) |gt| gt.val_type else null;
                gi += 1;
            }
        }
        return null;
    }
    const local_idx = idx - module.import_global_count;
    if (local_idx < module.globals.len) return module.globals[local_idx].global_type.val_type;
    return null;
}

fn getTableElemType(module: *const types.WasmModule, idx: u32) ?VT {
    if (idx < module.import_table_count) {
        var ti: u32 = 0;
        for (module.imports) |imp| {
            if (imp.kind == .table) {
                if (ti == idx) return if (imp.table_type) |tt| tt.elem_type else null;
                ti += 1;
            }
        }
        return null;
    }
    const local_idx = idx - module.import_table_count;
    if (local_idx < module.tables.len) return module.tables[local_idx].elem_type;
    return null;
}

/// Get label types for a branch target. For loop, label types are input types;
/// for block/if/function, label types are output types.
fn getLabelTypes(cf: *const CtrlFrame) []const VT {
    return if (cf.kind == .loop) cf.start_types else cf.end_types;
}

/// Pop the expected label types from the stack (validates a branch).
fn popLabelTypes(stack: []VT, sp: *u32, label_types: []const VT, cur_frame: ?*CtrlFrame) LoadError!void {
    var ri = label_types.len;
    while (ri > 0) {
        ri -= 1;
        if (!popExpect(stack, sp, label_types[ri], cur_frame))
            return error.TypeMismatch;
    }
}

/// Check (peek) that label types are on stack without consuming them.
fn peekLabelTypes(stack: []VT, sp: *u32, label_types: []const VT, cur_frame: ?*CtrlFrame) LoadError!void {
    const save_sp = sp.*;
    try popLabelTypes(stack, sp, label_types, cur_frame);
    sp.* = save_sp;
}

/// Validate the operand type stack of a function body (WebAssembly spec §3.3).
fn validateFunctionTypes(module: *const types.WasmModule, func: *const types.WasmFunction) LoadError!void {
    const code = func.code;
    const func_type = func.func_type;

    var local_types_buf: [1024]VT = undefined;
    var total_locals: u32 = @intCast(func_type.params.len);
    for (func_type.params, 0..) |p, li| {
        if (li >= local_types_buf.len) return;
        local_types_buf[li] = p;
    }
    for (func.locals) |ld| {
        var j: u32 = 0;
        while (j < ld.count) : (j += 1) {
            if (total_locals >= local_types_buf.len) return;
            local_types_buf[total_locals] = ld.val_type;
            total_locals += 1;
        }
    }
    const local_types = local_types_buf[0..total_locals];

    var stack_buf: [4096]VT = undefined;
    var sp: u32 = 0;

    var ctrl_buf: [256]CtrlFrame = undefined;
    var ctrl_sp: u32 = 0;

    // Push the function frame
    ctrl_buf[0] = .{
        .kind = .function,
        .start_height = 0,
        .start_types = &.{},
        .end_types = func_type.results,
    };
    ctrl_sp = 1;

    const ctrl_top = struct {
        fn get(ctrl: []CtrlFrame, csp: u32) ?*CtrlFrame {
            if (csp == 0) return null;
            return &ctrl[csp - 1];
        }
    };

    var i: usize = 0;
    while (i < code.len) {
        const op = code[i];
        i += 1;

        switch (op) {
            0x00 => { // unreachable
                if (ctrl_top.get(&ctrl_buf, ctrl_sp)) |cf| {
                    cf.unreachable_flag = true;
                    sp = cf.start_height;
                }
            },
            0x01 => {}, // nop

            // block, loop, if
            0x02, 0x03, 0x04 => {
                const bt = try readBlockType(code, &i, module.types);
                if (op == 0x04) {
                    if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                        return error.TypeMismatch;
                }
                // Pop block input types (for multi-value blocks)
                if (bt.params.len > 0) {
                    var pi = bt.params.len;
                    while (pi > 0) {
                        pi -= 1;
                        if (!popExpect(&stack_buf, &sp, bt.params[pi], ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                    }
                }
                if (ctrl_sp >= ctrl_buf.len) return;
                const start = sp;
                // Push input types as initial block stack
                for (bt.params) |p| pushType(&stack_buf, &sp, p);
                ctrl_buf[ctrl_sp] = .{
                    .kind = switch (op) {
                        0x02 => .block,
                        0x03 => .loop,
                        0x04 => .@"if",
                        else => unreachable,
                    },
                    .start_height = start,
                    .start_types = bt.params,
                    .end_types = bt.results,
                };
                ctrl_sp += 1;
            },

            0x05 => { // else
                const cf = ctrl_top.get(&ctrl_buf, ctrl_sp) orelse return error.TypeMismatch;
                if (cf.kind != .@"if") return error.TypeMismatch;
                if (!checkStackEnd(cf, &stack_buf, sp))
                    return error.TypeMismatch;
                cf.has_else = true;
                cf.unreachable_flag = false;
                sp = cf.start_height;
                // Push block input types for the else branch
                for (cf.start_types) |p| pushType(&stack_buf, &sp, p);
            },

            0x0B => { // end
                if (ctrl_sp == 0) return error.TypeMismatch;
                const cf = &ctrl_buf[ctrl_sp - 1];
                // An if-without-else is only valid when start_types == end_types
                if (cf.kind == .@"if" and !cf.has_else) {
                    if (cf.start_types.len != cf.end_types.len) return error.TypeMismatch;
                    for (cf.start_types, cf.end_types) |s, e| {
                        if (s != e) return error.TypeMismatch;
                    }
                }
                if (!checkStackEnd(cf, &stack_buf, sp))
                    return error.TypeMismatch;
                sp = cf.start_height;
                for (cf.end_types) |t| {
                    if (sp >= stack_buf.len) return;
                    stack_buf[sp] = t;
                    sp += 1;
                }
                ctrl_sp -= 1;
                if (ctrl_sp == 0) return;
            },

            0x0C => { // br
                const label = readU32Leb(code, &i);
                if (ctrl_sp <= label) return error.TypeMismatch;
                const target = &ctrl_buf[ctrl_sp - 1 - label];
                try popLabelTypes(&stack_buf, &sp, getLabelTypes(target), ctrl_top.get(&ctrl_buf, ctrl_sp));
                if (ctrl_top.get(&ctrl_buf, ctrl_sp)) |cf| {
                    cf.unreachable_flag = true;
                    sp = cf.start_height;
                }
            },
            0x0D => { // br_if
                const label = readU32Leb(code, &i);
                const cf = ctrl_top.get(&ctrl_buf, ctrl_sp);
                if (!popExpect(&stack_buf, &sp, .i32, cf))
                    return error.TypeMismatch;
                if (ctrl_sp <= label) return error.TypeMismatch;
                const target = &ctrl_buf[ctrl_sp - 1 - label];
                const label_types = getLabelTypes(target);
                // Per spec: br_if pops label types and re-pushes them.
                // In unreachable code, this materializes concrete types on the stack.
                try popLabelTypes(&stack_buf, &sp, label_types, cf);
                for (label_types) |t| pushType(&stack_buf, &sp, t);
            },
            0x0E => { // br_table
                const count = readU32Leb(code, &i);
                // Read all label indices
                var label_buf: [256]u32 = undefined;
                var lcount: u32 = 0;
                var j: u32 = 0;
                while (j <= count) : (j += 1) {
                    const l = readU32Leb(code, &i);
                    if (lcount < label_buf.len) {
                        label_buf[lcount] = l;
                        lcount += 1;
                    }
                }
                if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
                // Validate all labels are in range
                var li: u32 = 0;
                while (li < lcount) : (li += 1) {
                    if (ctrl_sp <= label_buf[li]) return error.TypeMismatch;
                }
                // Validate all labels have consistent types with the default
                if (lcount > 0) {
                    const default_label = label_buf[lcount - 1];
                    const default_target = &ctrl_buf[ctrl_sp - 1 - default_label];
                    const default_types = getLabelTypes(default_target);
                    const is_unreachable = if (ctrl_top.get(&ctrl_buf, ctrl_sp)) |cf| cf.unreachable_flag else false;
                    // Check each non-default label has the same arity (and types if reachable)
                    li = 0;
                    while (li < lcount - 1) : (li += 1) {
                        const target = &ctrl_buf[ctrl_sp - 1 - label_buf[li]];
                        const target_types = getLabelTypes(target);
                        if (target_types.len != default_types.len) return error.TypeMismatch;
                        if (!is_unreachable) {
                            for (target_types, default_types) |tt, dt| {
                                if (tt != dt) return error.TypeMismatch;
                            }
                        }
                    }
                    try popLabelTypes(&stack_buf, &sp, default_types, ctrl_top.get(&ctrl_buf, ctrl_sp));
                }
                if (ctrl_top.get(&ctrl_buf, ctrl_sp)) |cf| {
                    cf.unreachable_flag = true;
                    sp = cf.start_height;
                }
            },
            0x0F => { // return
                // Must have function result types on stack
                const func_frame = &ctrl_buf[0];
                try popLabelTypes(&stack_buf, &sp, func_frame.end_types, ctrl_top.get(&ctrl_buf, ctrl_sp));
                if (ctrl_top.get(&ctrl_buf, ctrl_sp)) |cf| {
                    cf.unreachable_flag = true;
                    sp = cf.start_height;
                }
            },

            // call, return_call
            0x10, 0x12 => {
                const fidx = readU32Leb(code, &i);
                if (module.getFuncType(fidx)) |ft| {
                    var pi = ft.params.len;
                    while (pi > 0) {
                        pi -= 1;
                        if (!popExpect(&stack_buf, &sp, ft.params[pi], ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                    }
                    if (op == 0x12) {
                        // return_call: callee results must match caller's return type
                        const func_results = ctrl_buf[0].end_types;
                        if (ft.results.len != func_results.len) return error.TypeMismatch;
                        for (ft.results, func_results) |cr, fr| {
                            if (cr != fr) return error.TypeMismatch;
                        }
                    } else {
                        for (ft.results) |rt| pushType(&stack_buf, &sp, rt);
                    }
                }
                if (op == 0x12) {
                    if (ctrl_top.get(&ctrl_buf, ctrl_sp)) |cf| {
                        cf.unreachable_flag = true;
                        sp = cf.start_height;
                    }
                }
            },
            // call_indirect, return_call_indirect
            0x11, 0x13 => {
                const tidx = readU32Leb(code, &i);
                const table_idx = readU32Leb(code, &i);
                // call_indirect requires a funcref table (spec §3.3.8.8)
                if (getTableElemType(module, table_idx)) |et| {
                    if (et != .funcref) return error.TypeMismatch;
                }
                if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
                if (tidx < module.types.len) {
                    const ft = module.types[tidx];
                    var pi = ft.params.len;
                    while (pi > 0) {
                        pi -= 1;
                        if (!popExpect(&stack_buf, &sp, ft.params[pi], ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                    }
                    if (op == 0x13) {
                        // return_call_indirect: callee results must match caller's return type
                        const func_results = ctrl_buf[0].end_types;
                        if (ft.results.len != func_results.len) return error.TypeMismatch;
                        for (ft.results, func_results) |cr, fr| {
                            if (cr != fr) return error.TypeMismatch;
                        }
                    } else {
                        for (ft.results) |rt| pushType(&stack_buf, &sp, rt);
                    }
                }
                if (op == 0x13) {
                    if (ctrl_top.get(&ctrl_buf, ctrl_sp)) |cf| {
                        cf.unreachable_flag = true;
                        sp = cf.start_height;
                    }
                }
            },

            // call_ref, return_call_ref: typeidx
            // Pop funcref (the ref to the function), pop params, push results
            0x14, 0x15 => {
                const tidx = readU32Leb(code, &i);
                // Pop the function reference
                _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                if (tidx < module.types.len) {
                    const ft = module.types[tidx];
                    var pi = ft.params.len;
                    while (pi > 0) {
                        pi -= 1;
                        if (!popExpect(&stack_buf, &sp, ft.params[pi], ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                    }
                    if (op == 0x15) {
                        // return_call_ref: callee results must match caller's return type
                        const func_results = ctrl_buf[0].end_types;
                        if (ft.results.len != func_results.len) return error.TypeMismatch;
                        for (ft.results, func_results) |cr, fr| {
                            if (cr != fr) return error.TypeMismatch;
                        }
                    } else {
                        for (ft.results) |rt| pushType(&stack_buf, &sp, rt);
                    }
                }
                if (op == 0x15) {
                    if (ctrl_top.get(&ctrl_buf, ctrl_sp)) |cf| {
                        cf.unreachable_flag = true;
                        sp = cf.start_height;
                    }
                }
            },

            0x1A => { _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp)); }, // drop
            0x1B => { // select (numeric types only; use select_t for ref types)
                const cf = ctrl_top.get(&ctrl_buf, ctrl_sp);
                if (!popExpect(&stack_buf, &sp, .i32, cf))
                    return error.TypeMismatch;
                const t2 = popAny(&stack_buf, &sp, cf);
                const t1 = popAny(&stack_buf, &sp, cf);
                const is_unreachable = cf != null and cf.?.unreachable_flag;
                // In non-unreachable code, both operands must be present
                if (!is_unreachable and (t1 == null or t2 == null)) return error.TypeMismatch;
                // select (untyped) requires numeric types — ref types need select_t
                if (t1) |v| { if (v == .funcref or v == .externref) return error.TypeMismatch; }
                if (t2) |v| { if (v == .funcref or v == .externref) return error.TypeMismatch; }
                if (t1 != null and t2 != null and t1.? != t2.?) return error.TypeMismatch;
                pushType(&stack_buf, &sp, t1 orelse t2 orelse .i32);
            },
            0x1C => { // select_t
                const count = readU32Leb(code, &i);
                if (count == 0 or i >= code.len) return error.TypeMismatch;
                const type_byte = code[i];
                // Validate the type is a known valtype
                const sel_type: VT = switch (type_byte) {
                    0x7F, 0x7E, 0x7D, 0x7C, 0x70, 0x6F => @enumFromInt(type_byte),
                    else => return error.TypeMismatch,
                };
                i += count;
                if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
                if (!popExpect(&stack_buf, &sp, sel_type, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
                if (!popExpect(&stack_buf, &sp, sel_type, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
                pushType(&stack_buf, &sp, sel_type);
            },

            // local.get
            0x20 => { const idx = readU32Leb(code, &i); if (idx < total_locals) pushType(&stack_buf, &sp, local_types[idx]); },
            // local.set
            0x21 => {
                const idx = readU32Leb(code, &i);
                if (idx < total_locals) {
                    if (!popExpect(&stack_buf, &sp, local_types[idx], ctrl_top.get(&ctrl_buf, ctrl_sp)))
                        return error.TypeMismatch;
                }
            },
            // local.tee
            0x22 => {
                const idx = readU32Leb(code, &i);
                if (idx < total_locals) {
                    if (!popExpect(&stack_buf, &sp, local_types[idx], ctrl_top.get(&ctrl_buf, ctrl_sp)))
                        return error.TypeMismatch;
                    pushType(&stack_buf, &sp, local_types[idx]);
                }
            },

            // global.get
            0x23 => { const gidx = readU32Leb(code, &i); if (getGlobalType(module, gidx)) |gt| pushType(&stack_buf, &sp, gt); },
            // global.set
            0x24 => {
                const gidx = readU32Leb(code, &i);
                // Check mutability
                if (getFullGlobalType(module, gidx)) |gt| {
                    if (gt.mutability != .mutable) return error.TypeMismatch;
                }
                if (getGlobalType(module, gidx)) |gt| {
                    if (!popExpect(&stack_buf, &sp, gt, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                        return error.TypeMismatch;
                }
            },

            // table.get: [i32] -> [t]
            0x25 => {
                const tidx = readU32Leb(code, &i);
                if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
                if (getTableElemType(module, tidx)) |et|
                    pushType(&stack_buf, &sp, et)
                else
                    pushType(&stack_buf, &sp, .funcref);
            },
            // table.set: [i32 t] -> []
            0x26 => {
                const tidx = readU32Leb(code, &i);
                const et = getTableElemType(module, tidx) orelse VT.funcref;
                if (!popExpect(&stack_buf, &sp, et, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
                if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
            },

            // Memory loads
            0x28 => { skipMemImm(code, &i); doLoad(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x29 => { skipMemImm(code, &i); doLoad(&stack_buf, &sp, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x2A => { skipMemImm(code, &i); doLoad(&stack_buf, &sp, .f32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x2B => { skipMemImm(code, &i); doLoad(&stack_buf, &sp, .f64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x2C, 0x2D, 0x2E, 0x2F => { skipMemImm(code, &i); doLoad(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x30, 0x31, 0x32, 0x33, 0x34, 0x35 => { skipMemImm(code, &i); doLoad(&stack_buf, &sp, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },

            // Memory stores
            0x36 => { skipMemImm(code, &i); doStore(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x37 => { skipMemImm(code, &i); doStore(&stack_buf, &sp, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x38 => { skipMemImm(code, &i); doStore(&stack_buf, &sp, .f32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x39 => { skipMemImm(code, &i); doStore(&stack_buf, &sp, .f64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x3A, 0x3B => { skipMemImm(code, &i); doStore(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x3C, 0x3D, 0x3E => { skipMemImm(code, &i); doStore(&stack_buf, &sp, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },

            // memory.size
            0x3F => { _ = readU32Leb(code, &i); pushType(&stack_buf, &sp, .i32); },
            // memory.grow
            0x40 => { _ = readU32Leb(code, &i);
                if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                pushType(&stack_buf, &sp, .i32);
            },

            // Constants
            0x41 => { _ = readI32Leb(code, &i); pushType(&stack_buf, &sp, .i32); },
            0x42 => { _ = readI64Leb(code, &i); pushType(&stack_buf, &sp, .i64); },
            0x43 => { i += 4; pushType(&stack_buf, &sp, .f32); },
            0x44 => { i += 8; pushType(&stack_buf, &sp, .f64); },

            // Comparison ops (i32)
            0x45 => doUnop(&stack_buf, &sp, .i32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0x46...0x4F => doBinop(&stack_buf, &sp, .i32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0x50 => doUnop(&stack_buf, &sp, .i64, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0x51...0x5A => doBinop(&stack_buf, &sp, .i64, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0x5B...0x60 => doBinop(&stack_buf, &sp, .f32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0x61...0x66 => doBinop(&stack_buf, &sp, .f64, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,

            // Arithmetic ops (i32)
            0x67, 0x68, 0x69 => doUnop(&stack_buf, &sp, .i32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0x6A...0x78 => doBinop(&stack_buf, &sp, .i32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            // Arithmetic ops (i64)
            0x79, 0x7A, 0x7B => doUnop(&stack_buf, &sp, .i64, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0x7C...0x8A => doBinop(&stack_buf, &sp, .i64, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            // Arithmetic ops (f32)
            0x8B...0x91 => doUnop(&stack_buf, &sp, .f32, .f32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0x92...0x98 => doBinop(&stack_buf, &sp, .f32, .f32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            // Arithmetic ops (f64)
            0x99...0x9F => doUnop(&stack_buf, &sp, .f64, .f64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xA0...0xA6 => doBinop(&stack_buf, &sp, .f64, .f64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,

            // Conversion ops
            0xA7 => doUnop(&stack_buf, &sp, .i64, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xA8, 0xA9 => doUnop(&stack_buf, &sp, .f32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xAA, 0xAB => doUnop(&stack_buf, &sp, .f64, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xAC, 0xAD => doUnop(&stack_buf, &sp, .i32, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xAE, 0xAF => doUnop(&stack_buf, &sp, .f32, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xB0, 0xB1 => doUnop(&stack_buf, &sp, .f64, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xB2, 0xB3 => doUnop(&stack_buf, &sp, .i32, .f32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xB4, 0xB5 => doUnop(&stack_buf, &sp, .i64, .f32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xB6 => doUnop(&stack_buf, &sp, .f64, .f32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xB7, 0xB8 => doUnop(&stack_buf, &sp, .i32, .f64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xB9, 0xBA => doUnop(&stack_buf, &sp, .i64, .f64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xBB => doUnop(&stack_buf, &sp, .f32, .f64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,

            // Reinterpret ops
            0xBC => doUnop(&stack_buf, &sp, .f32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xBD => doUnop(&stack_buf, &sp, .f64, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xBE => doUnop(&stack_buf, &sp, .i32, .f32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xBF => doUnop(&stack_buf, &sp, .i64, .f64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,

            // Sign extension
            0xC0, 0xC1 => doUnop(&stack_buf, &sp, .i32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
            0xC2, 0xC3, 0xC4 => doUnop(&stack_buf, &sp, .i64, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,

            // Reference ops
            0xD0 => { // ref.null: skip heap type, push ref type
                if (i < code.len) {
                    const ht = code[i];
                    i += 1;
                    if (ht == 0x6F or ht == 0x72) {
                        pushType(&stack_buf, &sp, .externref);
                    } else if (ht == 0x70 or ht == 0x73) {
                        pushType(&stack_buf, &sp, .funcref);
                    } else {
                        // Type index: consume remaining LEB128 bytes
                        if (ht & 0x80 != 0) {
                            while (i < code.len) {
                                if (code[i] & 0x80 == 0) { i += 1; break; }
                                i += 1;
                            }
                        }
                        pushType(&stack_buf, &sp, .funcref);
                    }
                }
            },
            0xD1 => { _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp)); pushType(&stack_buf, &sp, .i32); },
            0xD2 => { _ = readU32Leb(code, &i); pushType(&stack_buf, &sp, .funcref); },

            // br_on_null: pop ref, branch if null
            0xD4 => {
                _ = readU32Leb(code, &i); // label index
                _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
            },
            // ref.as_non_null: pop nullable ref, push non-nullable ref (or trap)
            0xD3 => {
                _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                pushType(&stack_buf, &sp, .funcref);
            },
            // br_on_non_null: pop ref, branch if non-null
            0xD6 => {
                _ = readU32Leb(code, &i); // label index
                _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
            },

            // 0xFC prefix
            0xFC => {
                const sub = readU32Leb(code, &i);
                switch (sub) {
                    0, 1 => doUnop(&stack_buf, &sp, .f32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
                    2, 3 => doUnop(&stack_buf, &sp, .f64, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
                    4, 5 => doUnop(&stack_buf, &sp, .f32, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
                    6, 7 => doUnop(&stack_buf, &sp, .f64, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch,
                    8 => { _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); }, // memory.init
                    9 => { _ = readU32Leb(code, &i); }, // data.drop
                    10 => { _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); }, // memory.copy
                    11 => { _ = readU32Leb(code, &i); }, // memory.fill
                    12 => { _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); }, // table.init
                    13 => { _ = readU32Leb(code, &i); }, // elem.drop
                    14 => { _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); }, // table.copy
                    15 => { // table.grow: [t i32] -> [i32]
                        const tidx = readU32Leb(code, &i);
                        const et = getTableElemType(module, tidx) orelse VT.funcref;
                        if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, et, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                        pushType(&stack_buf, &sp, .i32);
                    },
                    16 => { // table.size: [] -> [i32]
                        _ = readU32Leb(code, &i);
                        pushType(&stack_buf, &sp, .i32);
                    },
                    17 => { // table.fill: [i32 t i32] -> []
                        const tidx = readU32Leb(code, &i);
                        const et = getTableElemType(module, tidx) orelse VT.funcref;
                        if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, et, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                    },
                    else => {},
                }
            },

            else => {},
        }
    }

    // If we exit the loop without closing all blocks, the function body is truncated
    if (ctrl_sp != 0) return error.UnexpectedEnd;
}

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

test "load: data segment with no memory rejected" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    // data.45.wasm: data section with active segment for memory 0, but no memory
    const data = wasm_header ++ [_]u8{ 0x0B, 0x06, 0x01, 0x00, 0x41, 0x00, 0x0B, 0x00 };
    try testing.expectError(error.UnknownMemory, load(&data, arena.allocator()));
}

test "load: global.set on immutable global rejected" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    // global.1.wasm: immutable f32 global with a function that does global.set
    const data = wasm_header ++ [_]u8{
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00, // type: func()->()
        0x03, 0x02, 0x01, 0x00, // func section: 1 func, type 0
        0x06, 0x09, 0x01, 0x7D, 0x00, 0x43, 0x00, 0x00, 0x00, 0x00, 0x0B, // global: f32, immut, f32.const(0)
        0x0A, 0x0B, 0x01, 0x09, 0x00, 0x43, 0x00, 0x00, 0x80, 0x3F, 0x24, 0x00, 0x0B, // code: f32.const(1.0), global.set(0), end
    };
    try testing.expectError(error.TypeMismatch, load(&data, arena.allocator()));
}

test "load: illegal init expr in global rejected" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    // global.5.wasm: init expr with f32.const followed by f32.neg (not valid constant expr)
    const data = wasm_header ++ [_]u8{ 0x06, 0x0A, 0x01, 0x7D, 0x00, 0x43, 0x00, 0x00, 0x00, 0x00, 0x8C, 0x0B };
    try testing.expectError(error.InvalidInitExpr, load(&data, arena.allocator()));
}

test "load: illegal opcode 0xFF rejected" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    // binary.126.wasm: function body with 0xFF opcode
    const data = wasm_header ++ [_]u8{
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00, // type section
        0x03, 0x02, 0x01, 0x00, // func section
        0x0A, 0x08, 0x01, 0x06, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x0B, // code with 0xFF
    };
    try testing.expectError(error.IllegalOpcode, load(&data, arena.allocator()));
}
