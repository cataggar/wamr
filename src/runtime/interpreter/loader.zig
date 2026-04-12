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

/// Sentinel for "no concrete type index" (abstract type).
const NO_TIDX: u32 = 0xFFFF_FFFF;

/// Sentinel tidx for bottom of func ref hierarchy (nullfuncref / nullref).
/// A value with this tidx is a subtype of any funcref regardless of concrete tidx.
const BOTTOM_FUNC_TIDX: u32 = 0xFFFF_FFFE;

/// Sentinel tidx for bottom of extern ref hierarchy (nullexternref / nullexnref).
/// A value with this tidx is a subtype of any externref regardless of concrete tidx.
const BOTTOM_EXTERN_TIDX: u32 = 0xFFFF_FFFD;

fn isBottomTidx(tidx: u32) bool {
    return tidx == BOTTOM_FUNC_TIDX or tidx == BOTTOM_EXTERN_TIDX;
}

/// Check if two type indices are iso-recursively equivalent.
/// Two types are equivalent if they occupy the same position within rec groups
/// that have identical structure.
fn typeIdxEquivalent(module_types: []const types.FuncType, rec_groups: []const types.RecGroupInfo, a: u32, b: u32) bool {
    if (a == b) return true;
    if (a >= module_types.len or b >= module_types.len) return false;
    if (a >= rec_groups.len or b >= rec_groups.len) return false;
    const ga = rec_groups[a];
    const gb = rec_groups[b];
    // Must be at the same position within their rec groups and groups must be the same size
    if (ga.group_size != gb.group_size) return false;
    if (a - ga.group_start != b - gb.group_start) return false;
    // Compare all types in both rec groups pairwise
    var i: u32 = 0;
    while (i < ga.group_size) : (i += 1) {
        const ai = ga.group_start + i;
        const bi = gb.group_start + i;
        if (ai >= module_types.len or bi >= module_types.len) return false;
        if (!funcTypesStructurallyMatch(module_types, rec_groups, module_types[ai], module_types[bi], ga.group_start, gb.group_start)) return false;
    }
    return true;
}

/// Check if type `sub_idx` is a subtype of `super_idx` within the same module.
/// A type is a subtype of another if it declares that other as its supertype
/// (directly or transitively), and the types are iso-recursively compatible.
pub fn typeIdxIsSubtype(module_types: []const types.FuncType, rec_groups: []const types.RecGroupInfo, sub_idx: u32, super_idx: u32) bool {
    // Exact equivalence counts as subtype
    if (typeIdxEquivalent(module_types, rec_groups, sub_idx, super_idx)) return true;
    // Walk the supertype chain
    if (sub_idx >= module_types.len) return false;
    const sub_type = module_types[sub_idx];
    if (sub_type.supertype_idx != NO_TIDX) {
        return typeIdxIsSubtype(module_types, rec_groups, sub_type.supertype_idx, super_idx);
    }
    return false;
}

fn funcTypesStructurallyMatch(module_types: []const types.FuncType, rec_groups: []const types.RecGroupInfo, a: types.FuncType, b: types.FuncType, ga_start: u32, gb_start: u32) bool {
    if (a.kind != b.kind) return false;
    // Supertype and finality must match for iso-recursive equivalence
    if (a.is_final != b.is_final) return false;
    if (!tidxEquivalent(module_types, rec_groups, a.supertype_idx, b.supertype_idx, ga_start, gb_start)) return false;
    if (a.params.len != b.params.len or a.results.len != b.results.len) return false;
    for (a.params, b.params) |pa, pb| if (pa != pb) return false;
    for (a.results, b.results) |ra, rb| if (ra != rb) return false;
    // Compare type index references in params/results
    for (0..a.params.len) |pi| {
        const ta = if (pi < a.param_tidxs.len) a.param_tidxs[pi] else NO_TIDX;
        const tb = if (pi < b.param_tidxs.len) b.param_tidxs[pi] else NO_TIDX;
        if (!tidxEquivalent(module_types, rec_groups, ta, tb, ga_start, gb_start)) return false;
    }
    for (0..a.results.len) |ri| {
        const ta = if (ri < a.result_tidxs.len) a.result_tidxs[ri] else NO_TIDX;
        const tb = if (ri < b.result_tidxs.len) b.result_tidxs[ri] else NO_TIDX;
        if (!tidxEquivalent(module_types, rec_groups, ta, tb, ga_start, gb_start)) return false;
    }
    // Compare struct/array field type indices
    if (a.field_tidxs.len != b.field_tidxs.len) return false;
    for (a.field_tidxs, b.field_tidxs) |fa, fb| {
        if (!tidxEquivalent(module_types, rec_groups, fa, fb, ga_start, gb_start)) return false;
    }
    return true;
}

fn tidxEquivalent(_: []const types.FuncType, rec_groups: []const types.RecGroupInfo, ta: u32, tb: u32, ga_start: u32, gb_start: u32) bool {
    if (ta == tb) {
        // Same absolute index — but must check both are internal or both external
        const ga_size = if (ga_start < rec_groups.len) rec_groups[ga_start].group_size else 1;
        const gb_size = if (gb_start < rec_groups.len) rec_groups[gb_start].group_size else 1;
        const a_internal = ta >= ga_start and ta < ga_start + ga_size;
        const b_internal = tb >= gb_start and tb < gb_start + gb_size;
        if (a_internal != b_internal) return false;
        return true;
    }
    if (ta == NO_TIDX or tb == NO_TIDX) return ta == tb;
    // Check if both reference within their respective rec groups (self-references)
    // and at the same relative position
    const ga_size = if (ga_start < rec_groups.len) rec_groups[ga_start].group_size else 1;
    const gb_size = if (gb_start < rec_groups.len) rec_groups[gb_start].group_size else 1;
    const a_internal = ta >= ga_start and ta < ga_start + ga_size;
    const b_internal = tb >= gb_start and tb < gb_start + gb_size;
    if (a_internal and b_internal) {
        return (ta - ga_start) == (tb - gb_start);
    }
    // Both external: must be the same index
    if (!a_internal and !b_internal) return ta == tb;
    // One internal, one external: not equivalent
    return false;
}

/// Build a canonical type index mapping so structurally equivalent types share
/// the same canonical index. This allows exact-match comparison in validators.
fn canonicalizeTypeIndices(module: *types.WasmModule, allocator: std.mem.Allocator) void {
    const n: u32 = @intCast(module.types.len);
    if (n == 0) return;
    // canonical[i] = the lowest type index equivalent to i
    const canonical = allocator.alloc(u32, n) catch return;
    for (canonical, 0..) |*c, i| c.* = @intCast(i);

    // Iterate until convergence: rewrite tidxs using current canonical map,
    // then find new equivalences. Repeat until no new equivalences found.
    var changed = true;
    while (changed) {
        changed = false;
        // Rewrite type refs using current canonical mapping
        for (module.types) |ft| {
            for (@constCast(ft.param_tidxs)) |*t| {
                if (t.* != NO_TIDX and !isBottomTidx(t.*) and t.* < n and canonical[t.*] != t.*) t.* = canonical[t.*];
            }
            for (@constCast(ft.result_tidxs)) |*t| {
                if (t.* != NO_TIDX and !isBottomTidx(t.*) and t.* < n and canonical[t.*] != t.*) t.* = canonical[t.*];
            }
            for (@constCast(ft.field_tidxs)) |*t| {
                if (t.* != NO_TIDX and !isBottomTidx(t.*) and t.* < n and canonical[t.*] != t.*) t.* = canonical[t.*];
            }
            // Also canonicalize supertype index
            const st = @constCast(&ft.supertype_idx);
            if (st.* != NO_TIDX and st.* < n and canonical[st.*] != st.*) st.* = canonical[st.*];
        }
        // Find new equivalences
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            if (canonical[i] != i) continue;
            var j: u32 = i + 1;
            while (j < n) : (j += 1) {
                if (canonical[j] != j) continue;
                if (typeIdxEquivalent(module.types, module.rec_groups, i, j)) {
                    canonical[j] = i;
                    changed = true;
                }
            }
        }
    }
    // Final rewrite pass
    for (module.types) |ft| {
        for (@constCast(ft.param_tidxs)) |*t| {
            if (t.* != NO_TIDX and !isBottomTidx(t.*) and t.* < n) t.* = canonical[t.*];
        }
        for (@constCast(ft.result_tidxs)) |*t| {
            if (t.* != NO_TIDX and !isBottomTidx(t.*) and t.* < n) t.* = canonical[t.*];
        }
        for (@constCast(ft.field_tidxs)) |*t| {
            if (t.* != NO_TIDX and !isBottomTidx(t.*) and t.* < n) t.* = canonical[t.*];
        }
        const st = @constCast(&ft.supertype_idx);
        if (st.* != NO_TIDX and st.* < n) st.* = canonical[st.*];
    }
    module.canonical_type_map = canonical;
}

/// A value type paired with its concrete type index.
const ValTypeTidx = struct { vt: types.ValType, tidx: u32 };

fn readValType(reader: *BinaryReader) LoadError!types.ValType {
    return (try readValTypeWithTidx(reader, null)).vt;
}

/// Read a value type, validating concrete type indices against max_types if provided.
fn readValTypeChecked(reader: *BinaryReader, max_types: ?u32) LoadError!types.ValType {
    return (try readValTypeWithTidx(reader, max_types)).vt;
}

/// Read a value type together with its concrete type index.
fn readValTypeWithTidx(reader: *BinaryReader, max_types: ?u32) LoadError!ValTypeTidx {
    const byte = try reader.readByte();
    return switch (byte) {
        0x7F => .{ .vt = .i32, .tidx = NO_TIDX },
        0x7E => .{ .vt = .i64, .tidx = NO_TIDX },
        0x7D => .{ .vt = .f32, .tidx = NO_TIDX },
        0x7C => .{ .vt = .f64, .tidx = NO_TIDX },
        0x7B => .{ .vt = .v128, .tidx = NO_TIDX },
        0x70 => .{ .vt = .funcref, .tidx = NO_TIDX },
        0x6F => .{ .vt = .externref, .tidx = NO_TIDX },
        // GC proposal shorthand types (single-byte nullable ref types)
        0x6E => .{ .vt = .anyref, .tidx = NO_TIDX },
        0x6D => .{ .vt = .eqref, .tidx = NO_TIDX },
        0x6C => .{ .vt = .i31ref, .tidx = NO_TIDX },
        0x6B => .{ .vt = .structref, .tidx = NO_TIDX },
        0x6A => .{ .vt = .arrayref, .tidx = NO_TIDX },
        0x69 => .{ .vt = .exnref, .tidx = NO_TIDX },
        // Bottom types
        0x68 => .{ .vt = .externref, .tidx = BOTTOM_EXTERN_TIDX }, // nullexnref
        0x65 => .{ .vt = .nullref, .tidx = BOTTOM_FUNC_TIDX }, // nullref (ref null none)
        0x71 => .{ .vt = .funcref, .tidx = BOTTOM_FUNC_TIDX }, // nullfuncref (ref null nofunc)
        0x74 => .{ .vt = .externref, .tidx = BOTTOM_EXTERN_TIDX },
        0x73 => .{ .vt = .funcref, .tidx = BOTTOM_FUNC_TIDX },
        0x72 => .{ .vt = .externref, .tidx = BOTTOM_EXTERN_TIDX },
        // Typed reference types: ref null <heaptype> or ref <heaptype>
        0x63, 0x64 => {
            const is_nullable = (byte == 0x63);
            const heap_byte = try reader.readByte();
            return switch (heap_byte) {
                0x70 => .{ .vt = if (is_nullable) .funcref else .nonfuncref, .tidx = NO_TIDX },
                0x6F => .{ .vt = if (is_nullable) .externref else .nonexternref, .tidx = NO_TIDX },
                // GC abstract heap types
                0x6E => .{ .vt = if (is_nullable) .anyref else .anyref, .tidx = NO_TIDX },
                0x6D => .{ .vt = if (is_nullable) .eqref else .eqref, .tidx = NO_TIDX },
                0x6C => .{ .vt = if (is_nullable) .i31ref else .i31ref, .tidx = NO_TIDX },
                0x6B => .{ .vt = if (is_nullable) .structref else .structref, .tidx = NO_TIDX },
                0x6A => .{ .vt = if (is_nullable) .arrayref else .arrayref, .tidx = NO_TIDX },
                0x69 => .{ .vt = if (is_nullable) .exnref else .exnref, .tidx = NO_TIDX },
                // Bottom heap types — use sentinel tidx
                0x65, 0x71, 0x73 => .{ .vt = if (is_nullable) .funcref else .nonfuncref, .tidx = BOTTOM_FUNC_TIDX }, // none, nofunc
                0x68, 0x72, 0x74 => .{ .vt = if (is_nullable) .externref else .nonexternref, .tidx = BOTTOM_EXTERN_TIDX }, // noexn, noextern
                else => {
                    // Concrete type index (LEB128)
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
                    if (max_types) |mt| {
                        if (type_idx >= mt) {
                            std.debug.print("InvalidValType: byte=0x{x:0>2} heap=0x{x:0>2} tidx={d} max={d}\n", .{ byte, heap_byte, type_idx, mt });
                            return error.InvalidValType;
                        }
                    }
                    return .{ .vt = if (is_nullable) .funcref else .nonfuncref, .tidx = type_idx };
                },
            };
        },
        else => {
            std.debug.print("InvalidValType(readValType-catch-all): byte=0x{x:0>2}\n", .{byte});
            return error.InvalidValType;
        },
    };
}

/// Read a heap type and map to ValType. Heap types are: 0x70 (func), 0x6F (extern),
/// 0x73 (nofunc), 0x72 (noextern), or a concrete type index (unsigned LEB128).
fn readHeapTypeAsValType(reader: *BinaryReader) LoadError!types.ValType {
    const byte = try reader.readByte();
    return switch (byte) {
        0x70, 0x73 => .funcref,
        0x6F, 0x72 => .externref,
        0x6E => .anyref,
        0x6D => .eqref,
        0x6C => .i31ref,
        0x6B => .structref,
        0x6A => .arrayref,
        0x65 => .nullref,
        0x71 => .funcref, // nofunc
        0x69, 0x68, 0x74 => .externref, // exn, noexn, noextern
        else => {
            // Concrete type index: consume remaining LEB128 bytes
            if (byte & 0x80 != 0) {
                while (true) {
                    const b = try reader.readByte();
                    if (b & 0x80 == 0) break;
                }
            }
            return .nonfuncref;
        },
    };
}

const LimitsResult = struct {
    limits: types.Limits,
    is_64: bool,
};

fn readLimitsEx(reader: *BinaryReader) LoadError!LimitsResult {
    const flags = try reader.readByte();
    const has_max = (flags & 0x01) != 0;
    // bit 1 = shared (0x02) — accepted but not stored in Limits
    const is_64 = (flags & 0x04) != 0;
    // Reject unknown flag bits
    if (flags & ~@as(u8, 0x07) != 0) return error.InvalidLimits;

    if (is_64) {
        const min64 = try reader.readU64();
        if (min64 > std.math.maxInt(u32)) return error.InvalidLimits;
        const min: u32 = @intCast(min64);
        if (has_max) {
            const max64 = try reader.readU64();
            if (max64 > std.math.maxInt(u32)) return error.InvalidLimits;
            return .{ .limits = .{ .min = min, .max = @intCast(max64) }, .is_64 = true };
        }
        return .{ .limits = .{ .min = min }, .is_64 = true };
    } else {
        const min = try reader.readU32();
        if (has_max) {
            return .{ .limits = .{ .min = min, .max = try reader.readU32() }, .is_64 = false };
        }
        return .{ .limits = .{ .min = min }, .is_64 = false };
    }
}

fn readLimits(reader: *BinaryReader) LoadError!types.Limits {
    const result = try readLimitsEx(reader);
    return result.limits;
}

fn readTableType(reader: *BinaryReader, type_count: u32, _: u32) LoadError!types.TableType {
    const first_byte = try reader.readByte();

    // Table with init expression: 0x40 0x00 reftype limits expr
    if (first_byte == 0x40) {
        _ = try reader.readByte(); // reserved byte (must be 0)
        const info = try readValTypeWithTidx(reader, type_count);
        const lr = try readLimitsEx(reader);
        const init_expr = try parseInitExprChecked(reader, type_count);
        return .{ .elem_type = info.vt, .limits = lr.limits, .elem_tidx = info.tidx, .init_expr = init_expr, .is_table64 = lr.is_64 };
    }

    // Standard table: reftype limits
    var elem_tidx: u32 = NO_TIDX;
    const elem_type: types.ValType = switch (first_byte) {
        0x70 => .funcref,
        0x6F => .externref,
        0x63, 0x64 => blk: {
            const is_nullable = (first_byte == 0x63);
            const heap_byte = try reader.readByte();
            break :blk switch (heap_byte) {
                0x70, 0x73 => if (is_nullable) types.ValType.funcref else types.ValType.nonfuncref,
                0x6F, 0x72 => if (is_nullable) types.ValType.externref else types.ValType.nonexternref,
                else => {
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
                    if (type_idx >= type_count) {
                        std.debug.print("InvalidValType(readTableType-concrete): tidx={d} tc={d}\n", .{ type_idx, type_count });
                        return error.InvalidValType;
                    }
                    elem_tidx = type_idx;
                    break :blk if (is_nullable) types.ValType.funcref else types.ValType.nonfuncref;
                },
            };
        },
        else => {
            std.debug.print("InvalidValType(readTableType): byte=0x{x:0>2}\n", .{first_byte});
            return error.InvalidValType;
        },
    };
    const lr = try readLimitsEx(reader);
    return .{ .elem_type = elem_type, .limits = lr.limits, .elem_tidx = elem_tidx, .is_table64 = lr.is_64 };
}

/// Skip an init expression (scan to end opcode 0x0B).
fn skipInitExpr(reader: *BinaryReader) LoadError!void {
    while (reader.remaining() > 0) {
        const b = try reader.readByte();
        switch (b) {
            0x0B => return, // end
            0x41 => { _ = try reader.readI32(); },
            0x42 => { _ = try reader.readI64(); },
            0x43 => { _ = try reader.readBytes(4); },
            0x44 => { _ = try reader.readBytes(8); },
            0x23 => { _ = try reader.readU32(); },
            0xD0 => { _ = try readValTypeChecked(reader, null); },
            0xD2 => { _ = try reader.readU32(); },
            else => {},
        }
    }
    return error.UnexpectedEnd;
}

/// Validate and skip a table init expression. global.get can only reference imported globals.
fn validateAndSkipInitExpr(reader: *BinaryReader, max_global_idx: u32) LoadError!void {
    while (reader.remaining() > 0) {
        const b = try reader.readByte();
        switch (b) {
            0x0B => return,
            0x41 => { _ = try reader.readI32(); },
            0x42 => { _ = try reader.readI64(); },
            0x43 => { _ = try reader.readBytes(4); },
            0x44 => { _ = try reader.readBytes(8); },
            0x23 => {
                const idx = try reader.readU32();
                if (idx >= max_global_idx) return error.UnknownGlobal;
            },
            0xD0 => { _ = try readValTypeChecked(reader, null); },
            0xD2 => { _ = try reader.readU32(); },
            else => {},
        }
    }
    return error.UnexpectedEnd;
}

fn readMemoryType(reader: *BinaryReader) LoadError!types.MemoryType {
    const result = try readLimitsEx(reader);
    return .{ .limits = result.limits, .is_memory64 = result.is_64 };
}

fn readGlobalType(reader: *BinaryReader, type_count: ?u32) LoadError!types.GlobalType {
    const info = try readValTypeWithTidx(reader, type_count);
    const mut_byte = try reader.readByte();
    const mutability: types.GlobalType.Mutability = switch (mut_byte) {
        0 => .immutable,
        1 => .mutable,
        else => {
            std.debug.print("InvalidValType(readGlobalType): byte=0x{x:0>2}\n", .{mut_byte});
            return error.InvalidValType;
        },
    };
    return .{ .val_type = info.vt, .mutability = mutability, .type_idx = info.tidx };
}

fn parseInitExpr(reader: *BinaryReader) LoadError!types.InitExpr {
    return parseInitExprChecked(reader, null);
}

fn parseInitExprChecked(reader: *BinaryReader, type_count: ?u32) LoadError!types.InitExpr {
    const start_pos = reader.pos;
    const opcode = try reader.readByte();
    // Empty init expression (just 0x0B end) is invalid
    if (opcode == 0x0B) return error.TypeMismatch;
    // Try to parse as a single-instruction init expression first
    const simple: ?types.InitExpr = switch (opcode) {
        0x41 => .{ .i32_const = try reader.readI32() },
        0x42 => .{ .i64_const = try reader.readI64() },
        0x43 => .{ .f32_const = try reader.readF32() },
        0x44 => .{ .f64_const = try reader.readF64() },
        0x23 => .{ .global_get = try reader.readU32() },
        0xD0 => .{ .ref_null = try readHeapTypeAsValType(reader) },
        0xD2 => .{ .ref_func = try reader.readU32() },
        0xFB => blk: {
            // GC prefix: read sub-opcode
            const sub = try reader.readU32();
            switch (sub) {
                0x1C => break :blk null, // ref.i31 — compound (needs i32 operand)
                0x1A, 0x1B => break :blk null, // any.convert_extern, extern.convert_any — compound
                else => break :blk null,
            }
        },
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
            0xD0 => { _ = try readValTypeChecked(reader, type_count); stack_depth += 1; },
            0xD2 => { _ = try reader.readU32(); stack_depth += 1; },
            // Valid const expr binary ops: pop 2, push 1
            0x6A, 0x6B, 0x6C, // i32.add, i32.sub, i32.mul
            0x7C, 0x7D, 0x7E, // i64.add, i64.sub, i64.mul
            => { stack_depth -= 1; },
            // GC prefix opcodes valid in constant expressions
            0xFB => {
                const sub = try reader.readU32();
                switch (sub) {
                    0x1C => {}, // ref.i31: pop i32, push i31ref (net 0)
                    0x1A => {}, // any.convert_extern: pop externref, push anyref (net 0)
                    0x1B => {}, // extern.convert_any: pop anyref, push externref (net 0)
                    else => return error.InvalidInitExpr,
                }
            },
            // Any other opcode is invalid in a constant expression
            else => return error.InvalidInitExpr,
        }
    }
    return error.InvalidInitExpr;
}

// ─── Section parsers ────────────────────────────────────────────────────────

const TypeSectionResult = struct {
    types: []const types.FuncType,
    rec_groups: []const types.RecGroupInfo,
};

fn parseTypeSection(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!TypeSectionResult {
    const count = try reader.readU32();
    if (count == 0) return .{ .types = &.{}, .rec_groups = &.{} };

    // GC proposal: type entries may be rec groups (0x4E) containing sub types (0x50/0x4F)
    // We flatten them all into a single FuncType array.
    var func_types_list: std.ArrayList(types.FuncType) = .empty;
    var rec_groups_list: std.ArrayList(types.RecGroupInfo) = .empty;
    var entries_parsed: u32 = 0;
    while (entries_parsed < count) : (entries_parsed += 1) {
        const tag = try reader.readByte();
        if (tag == 0x4E) {
            // rec group: count of sub-entries, then sub-entries
            const rec_count = try reader.readU32();
            const group_start: u32 = @intCast(func_types_list.items.len);
            const max_type_in_group: u32 = @intCast(func_types_list.items.len + rec_count);
            var ri: u32 = 0;
            while (ri < rec_count) : (ri += 1) {
                const ft = try parseOneType(reader, allocator, @max(max_type_in_group, @as(u32, @intCast(func_types_list.items.len)) + count));
                func_types_list.append(allocator, ft) catch return error.OutOfMemory;
                rec_groups_list.append(allocator, .{ .group_start = group_start, .group_size = rec_count }) catch return error.OutOfMemory;
            }
        } else {
            // Single type entry (0x60 func, 0x50 sub, 0x4F sub final)
            reader.pos -= 1; // unread the tag
            const group_start: u32 = @intCast(func_types_list.items.len);
            const ft = try parseOneType(reader, allocator, @intCast(func_types_list.items.len + count));
            func_types_list.append(allocator, ft) catch return error.OutOfMemory;
            rec_groups_list.append(allocator, .{ .group_start = group_start, .group_size = 1 }) catch return error.OutOfMemory;
        }
    }
    return .{
        .types = func_types_list.toOwnedSlice(allocator) catch return error.OutOfMemory,
        .rec_groups = rec_groups_list.toOwnedSlice(allocator) catch return error.OutOfMemory,
    };
}

fn parseOneType(reader: *BinaryReader, allocator: std.mem.Allocator, max_types: u32) LoadError!types.FuncType {
    const tag = try reader.readByte();
    if (tag == 0x50 or tag == 0x4F) {
        // sub type: 0x50 <num_supers> <super_idx*> <comptype>
        // sub final type: 0x4F <num_supers> <super_idx*> <comptype>
        const is_final = (tag == 0x4F);
        const num_supers = try reader.readU32();
        var supertype_idx: u32 = NO_TIDX;
        var si: u32 = 0;
        while (si < num_supers) : (si += 1) {
            supertype_idx = try reader.readU32();
        }
        const comp_tag = try reader.readByte();
        var ft: types.FuncType = undefined;
        if (comp_tag == 0x60) {
            ft = try parseFuncType(reader, allocator, max_types);
        } else if (comp_tag == 0x5F) {
            ft = try parseStructType(reader, allocator, max_types);
        } else if (comp_tag == 0x5E) {
            ft = try parseArrayType(reader, allocator, max_types);
        } else {
            ft = .{ .params = &.{}, .results = &.{} };
        }
        ft.supertype_idx = supertype_idx;
        ft.is_final = is_final;
        return ft;
    }
    if (tag != 0x60) {
        // struct (0x5F) or array (0x5E) without sub wrapper
        if (tag == 0x5F) {
            return parseStructType(reader, allocator, max_types);
        } else if (tag == 0x5E) {
            return parseArrayType(reader, allocator, max_types);
        }
        return error.InvalidFuncType;
    }
    return parseFuncType(reader, allocator, max_types);
}

fn parseStructType(reader: *BinaryReader, allocator: std.mem.Allocator, max_types: u32) LoadError!types.FuncType {
    const field_count = try reader.readU32();
    var ftidxs = if (field_count > 0) try allocator.alloc(u32, field_count) else @as([]u32, &.{});
    var fi: u32 = 0;
    while (fi < field_count) : (fi += 1) {
        const info = try readValTypeWithTidx(reader, max_types);
        ftidxs[fi] = info.tidx;
        _ = try reader.readByte(); // mutability
    }
    return .{ .params = &.{}, .results = &.{}, .kind = .struct_, .field_tidxs = ftidxs };
}

fn parseArrayType(reader: *BinaryReader, allocator: std.mem.Allocator, max_types: u32) LoadError!types.FuncType {
    const info = try readValTypeWithTidx(reader, max_types);
    _ = try reader.readByte(); // mutability
    var ftidxs = try allocator.alloc(u32, 1);
    ftidxs[0] = info.tidx;
    return .{ .params = &.{}, .results = &.{}, .kind = .array, .field_tidxs = ftidxs };
}

fn parseFuncType(reader: *BinaryReader, allocator: std.mem.Allocator, max_types: u32) LoadError!types.FuncType {
    const param_count = try reader.readU32();
    var params: []types.ValType = &.{};
    var param_tidxs: []u32 = &.{};
    if (param_count > 0) {
        params = try allocator.alloc(types.ValType, param_count);
        param_tidxs = try allocator.alloc(u32, param_count);
        for (params, param_tidxs) |*v, *t| {
            const info = try readValTypeWithTidx(reader, max_types);
            v.* = info.vt;
            t.* = info.tidx;
        }
    }

    const result_count = try reader.readU32();
    var results: []types.ValType = &.{};
    var result_tidxs: []u32 = &.{};
    if (result_count > 0) {
        results = try allocator.alloc(types.ValType, result_count);
        result_tidxs = try allocator.alloc(u32, result_count);
        for (results, result_tidxs) |*v, *t| {
            const info = try readValTypeWithTidx(reader, max_types);
            v.* = info.vt;
            t.* = info.tidx;
        }
    }

    return .{ .params = params, .results = results, .param_tidxs = param_tidxs, .result_tidxs = result_tidxs };
}

fn parseImportSection(reader: *BinaryReader, allocator: std.mem.Allocator, type_count: u32, tag_count: *u32) LoadError![]const types.ImportDesc {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    var imports_list: std.ArrayList(types.ImportDesc) = .empty;
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const module_name = try reader.readName();
        const field_name = try reader.readName();
        const kind_byte = try reader.readByte();

        // Tag imports (0x04) from exception handling — store with kind = .tag
        if (kind_byte == 0x04) {
            _ = try reader.readByte(); // tag attribute
            const tag_tidx = try reader.readU32(); // type index
            tag_count.* += 1;
            imports_list.append(allocator, .{
                .module_name = module_name,
                .field_name = field_name,
                .kind = .tag,
                .tag_type_idx = tag_tidx,
            }) catch return error.OutOfMemory;
            continue;
        }

        const kind: types.ExternalKind = switch (kind_byte) {
            0x00 => .function,
            0x01 => .table,
            0x02 => .memory,
            0x03 => .global,
            0x04 => {
                // Tag import (exception handling) — store
                _ = try reader.readByte(); // attribute
                const tag_tidx = try reader.readU32(); // type index
                tag_count.* += 1;
                imports_list.append(allocator, .{
                    .module_name = module_name,
                    .field_name = field_name,
                    .kind = .tag,
                    .tag_type_idx = tag_tidx,
                }) catch return error.OutOfMemory;
                continue;
            },
            else => return error.InvalidImportKind,
        };

        var imp = types.ImportDesc{
            .module_name = module_name,
            .field_name = field_name,
            .kind = kind,
        };

        switch (kind) {
            .function => imp.func_type_idx = try reader.readU32(),
            .table => imp.table_type = try readTableType(reader, type_count, 0),
            .memory => imp.memory_type = try readMemoryType(reader),
            .global => imp.global_type = try readGlobalType(reader, type_count),
            .tag => {}, // tag imports handled above
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

fn parseTableSection(reader: *BinaryReader, allocator: std.mem.Allocator, type_count: u32, import_global_count: u32) LoadError![]const types.TableType {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    const tables = try allocator.alloc(types.TableType, count);
    for (tables) |*t| {
        t.* = try readTableType(reader, type_count, import_global_count);
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

fn parseGlobalSection(reader: *BinaryReader, allocator: std.mem.Allocator, type_count: u32) LoadError![]const types.WasmGlobal {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    const globals = try allocator.alloc(types.WasmGlobal, count);
    for (globals) |*g| {
        const global_type = try readGlobalType(reader, type_count);
        const init_expr = try parseInitExprChecked(reader, type_count);
        g.* = .{ .global_type = global_type, .init_expr = init_expr };
    }
    return globals;
}

fn parseExportSection(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError![]const types.ExportDesc {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    var exports_list: std.ArrayList(types.ExportDesc) = .empty;
    // Track ALL export names (including tags) for duplicate checking
    var seen_names: std.ArrayList([]const u8) = .empty;
    defer seen_names.deinit(allocator);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const name = try reader.readName();
        const kind_byte = try reader.readByte();
        const index = try reader.readU32();
        // Check for duplicate export names before filtering
        for (seen_names.items) |prev| {
            if (std.mem.eql(u8, prev, name)) return error.DuplicateExportName;
        }
        seen_names.append(allocator, name) catch return error.InvalidExportKind;
        const kind: ?types.ExternalKind = switch (kind_byte) {
            0x00 => .function,
            0x01 => .table,
            0x02 => .memory,
            0x03 => .global,
            0x04 => .tag,
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
            var seg_tidx: u32 = NO_TIDX;
            var is_nullable_elem = true; // default for flags=4 (no explicit reftype)
            if (flags != 4) {
                const ref_byte = try reader.readByte();
                // 0x70 (funcref) and 0x6F (externref) are nullable
                // 0x63 (ref null) is nullable, 0x64 (ref) is non-nullable
                is_nullable_elem = (ref_byte != 0x64);
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
                        var ht_idx: u32 = ht & 0x7F;
                        if (ht & 0x80 != 0) {
                            var shift: u5 = 7;
                            while (true) {
                                const b = try reader.readByte();
                                ht_idx |= @as(u32, b & 0x7F) << shift;
                                if (b & 0x80 == 0) break;
                                shift +|= 7;
                            }
                        }
                        if (ht_idx >= type_count) {
                            std.debug.print("InvalidValType(elemSection): ht={d} tc={d}\n", .{ ht_idx, type_count });
                            return error.InvalidValType;
                        }
                        seg_tidx = ht_idx;
                        break :blk .func_ref;
                    },
                    else => return error.InvalidElemSegment,
                };
            }
            const num_elems = try reader.readU32();
            var func_indices_list: std.ArrayList(?u32) = .empty;
            var elem_exprs_list: std.ArrayList(?types.InitExpr) = .empty;
            var has_runtime_exprs = false;
            var j: u32 = 0;
            while (j < num_elems) : (j += 1) {
                const expr = try parseInitExpr(reader);
                switch (expr) {
                    .ref_func => |fidx| {
                        try func_indices_list.append(allocator, fidx);
                        try elem_exprs_list.append(allocator, expr);
                    },
                    .ref_null => |vt| {
                        // For flags=4, kind defaults to func_ref but element may be externref
                        if (flags == 4 and vt.isExternRef()) kind = .extern_ref;
                        if (kind == .func_ref and !vt.isFuncRef()) return error.TypeMismatch;
                        if (kind == .extern_ref and !vt.isExternRef()) return error.TypeMismatch;
                        try func_indices_list.append(allocator, null);
                        try elem_exprs_list.append(allocator, null);
                    },
                    // global.get and compound expressions can produce funcref/externref
                    .global_get, .bytecode => {
                        try func_indices_list.append(allocator, null);
                        try elem_exprs_list.append(allocator, expr);
                        has_runtime_exprs = true;
                    },
                    else => return error.TypeMismatch,
                }
            }
            elem.* = .{
                .table_idx = table_idx,
                .offset = offset,
                .kind = kind,
                .func_indices = try func_indices_list.toOwnedSlice(allocator),
                .elem_exprs = if (has_runtime_exprs) try elem_exprs_list.toOwnedSlice(allocator) else &.{},
                .is_passive = is_passive,
                .is_declarative = is_declarative,
                .type_idx = seg_tidx,
                .nullable_elements = is_nullable_elem,
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
                .nullable_elements = false, // funcidx vectors are always non-null
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
                const info = try readValTypeWithTidx(reader, @intCast(module_types.len));
                ld.* = .{ .count = lcount, .val_type = info.vt, .type_idx = info.tidx };
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

        // Enforce section ordering (custom sections can appear anywhere)
        if (section_id != 0) {
            if (section_id > @intFromEnum(types.SectionId.tag)) {
                return error.MalformedSectionId;
            }
            // Tag section (13) and data count section (12) don't participate in strict ordering
            if (section_id != @intFromEnum(types.SectionId.tag) and
                section_id != @intFromEnum(types.SectionId.data_count))
            {
                if (last_section_id) |last| {
                    if (section_id <= last) return error.InvalidSectionOrder;
                }
                last_section_id = section_id;
            }
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
                .type => {
                    const type_result = try parseTypeSection(&reader, allocator);
                    module.types = type_result.types;
                    module.rec_groups = type_result.rec_groups;
                    canonicalizeTypeIndices(&module, allocator);
                },
                .import => {
                    const tc: u32 = @intCast(module.types.len);
                    module.imports = try parseImportSection(&reader, allocator, tc, &module.import_tag_count);
                    for (module.imports) |imp| {
                        switch (imp.kind) {
                            .function => module.import_function_count += 1,
                            .table => module.import_table_count += 1,
                            .memory => module.import_memory_count += 1,
                            .global => module.import_global_count += 1,
                            .tag => {}, // tag count already tracked by parseImportSection
                        }
                    }
                },
                .function => func_type_indices = try parseFunctionSection(&reader, allocator),
                .table => module.tables = try parseTableSection(&reader, allocator, @intCast(module.types.len), module.import_global_count),
                .memory => module.memories = try parseMemorySection(&reader, allocator),
                .global => module.globals = try parseGlobalSection(&reader, allocator, @intCast(module.types.len)),
                .@"export" => module.exports = try parseExportSection(&reader, allocator),
                .start => module.start_function = try reader.readU32(),
                .element => module.elements = try parseElementSection(&reader, allocator, @intCast(module.types.len)),
                .code => module.functions = try parseCodeSection(&reader, func_type_indices, module.types, allocator),
                .data => module.data_segments = try parseDataSection(&reader, allocator),
                .data_count => module.data_count = try reader.readU32(),
                .tag => {
                    // Tag section (exception handling): count + (attribute, type_idx)*
                    const tag_count = try reader.readU32();
                    if (tag_count > 0) {
                        const tag_types = allocator.alloc(u32, tag_count) catch return error.OutOfMemory;
                        for (tag_types) |*tt| {
                            _ = try reader.readByte(); // attribute (0 = exception)
                            tt.* = try reader.readU32(); // type index
                        }
                        module.tag_types = tag_types;
                    }
                },
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

    // Infer local tag count from tag exports when tag section is missing.
    // Some binary writers (e.g., wabt) emit tag exports but no tag section.
    if (module.tag_types.len == 0) {
        var max_tag_idx: ?u32 = null;
        for (module.exports) |exp| {
            if (exp.kind == .tag) {
                if (max_tag_idx == null or exp.index > max_tag_idx.?)
                    max_tag_idx = exp.index;
            }
        }
        if (max_tag_idx) |max_idx| {
            const local_count = if (max_idx >= module.import_tag_count) max_idx - module.import_tag_count + 1 else 0;
            if (local_count > 0) {
                if (allocator.alloc(u32, local_count)) |tag_types| {
                    for (tag_types) |*tt| tt.* = 0xFFFFFFFF;
                    module.tag_types = tag_types;
                } else |_| {}
            }
        }
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
        // Non-nullable element types require a table initializer
        if ((table.elem_type == .nonfuncref or table.elem_type == .nonexternref) and table.init_expr == null) {
            return error.TypeMismatch;
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
            .tag => {},
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
            .tag => {},
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
                        if (gt.val_type != g.global_type.val_type and
                            !gt.val_type.isSubtypeOf(g.global_type.val_type))
                            return error.TypeMismatch;
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
                if (g.global_type.val_type != rt and
                    !rt.isSubtypeOf(g.global_type.val_type))
                    return error.TypeMismatch;
            },
            .ref_func => |fidx| {
                if (!g.global_type.val_type.isFuncRef()) return error.TypeMismatch;
                if (fidx >= total_funcs) return error.UnknownFunction;
                // Check concrete type index compatibility (subtyping allowed)
                if (g.global_type.type_idx != NO_TIDX) {
                    const func_tidx = module.getFuncTypeIdx(fidx) orelse NO_TIDX;
                    if (func_tidx != NO_TIDX and func_tidx != g.global_type.type_idx) {
                        if (!typeIdxIsSubtype(module.types, module.rec_groups, func_tidx, g.global_type.type_idx))
                            return error.TypeMismatch;
                    }
                }
            },
            .bytecode => {}, // compound expressions validated at evaluation time
        }
    }

    // Validate data segment offset expressions
    for (module.data_segments) |seg| {
        if (!seg.is_passive) {
            // memory64 segments use i64 offsets, regular memory uses i32
            const is_mem64 = if (seg.memory_idx < module.import_memory_count) blk: {
                var mi: u32 = 0;
                for (module.imports) |imp| {
                    if (imp.kind == .memory) {
                        if (mi == seg.memory_idx) break :blk if (imp.memory_type) |mt| mt.is_memory64 else false;
                        mi += 1;
                    }
                }
                break :blk false;
            } else blk: {
                const li = seg.memory_idx - module.import_memory_count;
                break :blk if (li < module.memories.len) module.memories[li].is_memory64 else false;
            };
            switch (seg.offset) {
                .i32_const => { if (is_mem64) return error.TypeMismatch; },
                .i64_const => { if (!is_mem64) return error.TypeMismatch; },
                .global_get => |idx| {
                    if (idx >= total_globals) return error.UnknownGlobal;
                    if (idx < module.import_global_count) {
                        if (getImportGlobalType(module, idx)) |gt| {
                            if (gt.mutability == .mutable) return error.TypeMismatch;
                            const expected_type: VT = if (is_mem64) .i64 else .i32;
                            if (gt.val_type != expected_type) return error.TypeMismatch;
                        }
                    } else {
                        const local_idx = idx - module.import_global_count;
                        if (local_idx < module.globals.len) {
                            if (module.globals[local_idx].global_type.mutability == .mutable) return error.TypeMismatch;
                        }
                    }
                },
                .bytecode => {},
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
                if (elem.kind == .func_ref and !tet.isFuncRef()) return error.TypeMismatch;
                if (elem.kind == .extern_ref and !tet.isExternRef()) return error.TypeMismatch;
                // Non-nullable table requires non-nullable elements
                if ((tet == .nonfuncref or tet == .nonexternref) and elem.nullable_elements)
                    return error.TypeMismatch;
                // Check concrete type index compatibility
                const table_tidx = if (elem.table_idx < module.import_table_count)
                    getImportTableTidx(module, elem.table_idx)
                else blk: {
                    const local_idx = elem.table_idx - module.import_table_count;
                    break :blk if (local_idx < module.tables.len) module.tables[local_idx].elem_tidx else NO_TIDX;
                };
                // Both concrete: must match
                if (table_tidx != NO_TIDX and elem.type_idx != NO_TIDX and elem.type_idx != table_tidx) return error.TypeMismatch;
            }
            // Validate offset expression type
            if (elem.offset) |offset| {
                // table64 uses i64 offsets
                const is_tbl64 = if (elem.table_idx < module.import_table_count) blk: {
                    var ti: u32 = 0;
                    for (module.imports) |imp| {
                        if (imp.kind == .table) {
                            if (ti == elem.table_idx) break :blk if (imp.table_type) |tt| tt.is_table64 else false;
                            ti += 1;
                        }
                    }
                    break :blk false;
                } else blk: {
                    const li = elem.table_idx - module.import_table_count;
                    break :blk if (li < module.tables.len) module.tables[li].is_table64 else false;
                };
                switch (offset) {
                    .i32_const => { if (is_tbl64) return error.TypeMismatch; },
                    .i64_const => { if (!is_tbl64) return error.TypeMismatch; },
                    .global_get => |idx| {
                        if (idx >= total_globals) return error.UnknownGlobal;
                        if (idx < module.import_global_count) {
                            if (getImportGlobalType(module, idx)) |gt| {
                                if (gt.mutability == .mutable) return error.TypeMismatch;
                                const expected_type: VT = if (is_tbl64) .i64 else .i32;
                                if (gt.val_type != expected_type) return error.TypeMismatch;
                            }
                        } else {
                            const local_idx = idx - module.import_global_count;
                            if (local_idx < module.globals.len) {
                                if (module.globals[local_idx].global_type.mutability == .mutable) return error.TypeMismatch;
                            }
                        }
                    },
                    .bytecode => {},
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
        try validateFunctionBody(func.code, module.types.len, total_funcs, total_tables, total_memories, total_globals, total_locals, module.data_count != null);
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
            0x0C, 0x0D, 0xD5, 0xD6 => { const r = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r.bytes_read; },
            // Exception handling opcodes
            0x06 => { skipBlockTypeImm(code, &i); }, // try
            0x07, 0x08, 0x09, 0x19 => { const r = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r.bytes_read; },
            0x0A => {}, // throw_ref
            0x1F => { // try_table
                skipBlockTypeImm(code, &i);
                const clause_count_r = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += clause_count_r.bytes_read;
                var ci: u32 = 0;
                while (ci < clause_count_r.value) : (ci += 1) {
                    const ck = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += ck.bytes_read;
                    if (ck.value == 0 or ck.value == 1) { const tr = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += tr.bytes_read; }
                    const lr = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += lr.bytes_read;
                }
            },
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
                // Multi-memory: bit 6 of alignment signals a memory index follows
                if (r1.value & 0x40 != 0) {
                    const r_mi = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r_mi.bytes_read;
                }
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
    total_memories: u32,
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
            // Memory operations require at least one memory
            if (total_memories == 0) return error.UnknownMemory;
            const align_result = leb128_mod.readUnsigned(u32, code[i..]) catch return error.InvalidAlignment;
            i += align_result.bytes_read;
            // Multi-memory: bit 6 signals a memory index follows
            const has_mem_idx = align_result.value & 0x40 != 0;
            if (has_mem_idx) {
                const mem_result = leb128_mod.readUnsigned(u32, code[i..]) catch return error.InvalidSectionSize;
                i += mem_result.bytes_read;
            }
            // Alignment is in bits 0-5 (mask off multi-memory bit 6)
            const align_value = align_result.value & 0x3F;
            // But reject if any bits above bit 6 are set (invalid)
            if (align_result.value & ~@as(u32, 0x7F) != 0) return error.InvalidAlignment;
            if (align_value > ma) return error.InvalidAlignment;
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

            // memory.size, memory.grow: memory index (LEB128)
            0x3F, 0x40 => {
                if (total_memories == 0) return error.UnknownMemory;
                const r = leb128_mod.readUnsigned(u32, code[i..]) catch return error.InvalidAlignment;
                i += r.bytes_read;
                if (r.value >= total_memories) return error.UnknownMemory;
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

            // ref.is_null, ref.eq, ref.as_non_null have no immediate
            0xD1, 0xD3, 0xD4 => {},

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
            0xD5, 0xD6 => {
                const r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += r.bytes_read;
            },

            // Exception handling opcodes
            0x06 => { skipBlockTypeImm(code, &i); }, // try (legacy): blocktype
            0x07 => { const r = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r.bytes_read; }, // catch: tagidx
            0x08 => { const r = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r.bytes_read; }, // throw: tagidx
            0x09 => { const r = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r.bytes_read; }, // rethrow: labelidx
            0x0A => {}, // throw_ref: no immediates
            0x19 => { const r = leb128_mod.readUnsigned(u32, code[i..]) catch return; i += r.bytes_read; }, // delegate: labelidx
            0x1F => { // try_table: blocktype + catch clause list
                skipBlockTypeImm(code, &i);
                const clause_count = readU32Leb(code, &i);
                var ci: u32 = 0;
                while (ci < clause_count) : (ci += 1) {
                    const clause_kind = readU32Leb(code, &i); // 0=catch, 1=catch_ref, 2=catch_all, 3=catch_all_ref
                    if (clause_kind == 0 or clause_kind == 1) {
                        _ = readU32Leb(code, &i); // tag index
                    }
                    _ = readU32Leb(code, &i); // label index
                }
            },

            // GC prefix opcodes (0xFB)
            0xFB => {
                const sub_r = leb128_mod.readUnsigned(u32, code[i..]) catch return;
                i += sub_r.bytes_read;
                switch (sub_r.value) {
                    // struct ops
                    0x00, 0x01 => { _ = readU32Leb(code, &i); }, // struct.new, struct.new_default: typeidx
                    0x02, 0x03, 0x04 => { _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); }, // struct.get/get_s/get_u: typeidx + fieldidx
                    0x05 => { _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); }, // struct.set: typeidx + fieldidx
                    // array ops
                    0x06, 0x07 => { _ = readU32Leb(code, &i); }, // array.new, array.new_default: typeidx
                    0x08 => { _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); }, // array.new_fixed: typeidx + count
                    0x09, 0x0A => { _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); }, // array.new_data/elem: typeidx + idx
                    0x0B, 0x0C, 0x0D => { _ = readU32Leb(code, &i); }, // array.get/get_s/get_u: typeidx
                    0x0E => { _ = readU32Leb(code, &i); }, // array.set: typeidx
                    0x0F => {}, // array.len: no immediates
                    0x10 => { _ = readU32Leb(code, &i); }, // array.fill: typeidx
                    0x11 => { _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); }, // array.copy: typeidx + typeidx
                    0x12, 0x13 => { _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); }, // array.init_data/elem: typeidx + idx
                    // ref.test/ref.cast: heaptype
                    0x14, 0x15, 0x16, 0x17 => { _ = readU32Leb(code, &i); },
                    // br_on_cast: flags + label + 2 heaptypes
                    0x18, 0x19 => { _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); },
                    // No immediates
                    0x1A, 0x1B, 0x1C, 0x1D, 0x1E => {},
                    else => {},
                }
            },

            // Opcodes in reserved ranges
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

fn getImportTableTidx(module: *const types.WasmModule, idx: u32) u32 {
    if (idx >= module.import_table_count) return NO_TIDX;
    var ti: u32 = 0;
    for (module.imports) |imp| {
        if (imp.kind == .table) {
            if (ti == idx) return if (imp.table_type) |tt| tt.elem_tidx else NO_TIDX;
            ti += 1;
        }
    }
    return NO_TIDX;
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
    /// Type index of block type (NO_TIDX = simple/void block type).
    /// When set, start/end tidxs come from module.types[block_type_idx].
    block_type_idx: u32 = NO_TIDX,
    /// For single-result block types with a concrete ref type, stores the tidx directly.
    single_result_tidx: u32 = NO_TIDX,
    /// Index into init_save_stack where this frame's saved init state starts
    init_save_idx: u32 = 0,
};

/// Block type: both params and results.
const BlockType = struct {
    params: []const VT = &.{},
    results: []const VT,
    type_idx: u32 = NO_TIDX, // type index for multi-value block types
    single_result_tidx: u32 = NO_TIDX, // for simple block types with concrete ref result
};

fn readBlockType(code: []const u8, pos: *usize, module_types: []const types.FuncType) LoadError!BlockType {
    if (pos.* >= code.len) return .{ .results = &.{} };
    const bt = code[pos.*];
    if (bt == 0x40) { pos.* += 1; return .{ .results = &.{} }; }
    if (bt == 0x7F or bt == 0x7E or bt == 0x7D or bt == 0x7C or bt == 0x70 or bt == 0x6F or
        bt == 0x6E or bt == 0x6D or bt == 0x6C or bt == 0x6B or bt == 0x6A or bt == 0x65 or bt == 0x71 or
        bt == 0x69 or bt == 0x68 or bt == 0x74) {
        pos.* += 1;
        return switch (bt) {
            0x7F => .{ .results = &[_]VT{.i32} },
            0x7E => .{ .results = &[_]VT{.i64} },
            0x7D => .{ .results = &[_]VT{.f32} },
            0x7C => .{ .results = &[_]VT{.f64} },
            0x70 => .{ .results = &[_]VT{.funcref} },
            0x6F => .{ .results = &[_]VT{.externref} },
            // GC abstract ref types
            0x6E, 0x6D, 0x6C, 0x6B, 0x6A, 0x65, 0x71 => .{ .results = &[_]VT{.funcref} },
            0x69, 0x68, 0x74 => .{ .results = &[_]VT{.externref} },
            else => .{ .results = &.{} },
        };
    }
    // Typed reference block types: ref null/ref + heaptype
    if (bt == 0x63 or bt == 0x64) {
        const is_nullable = (bt == 0x63);
        pos.* += 1; // consume 0x63/0x64
        if (pos.* >= code.len) return error.TypeMismatch;
        const ht = code[pos.*];
        if (ht == 0x70 or ht == 0x73) { pos.* += 1; return .{ .results = if (is_nullable) &[_]VT{.funcref} else &[_]VT{.nonfuncref} }; }
        if (ht == 0x6F or ht == 0x72) { pos.* += 1; return .{ .results = if (is_nullable) &[_]VT{.externref} else &[_]VT{.nonexternref} }; }
        // GC abstract heap types
        if (ht == 0x6E or ht == 0x6D or ht == 0x6C or ht == 0x6B or ht == 0x6A or ht == 0x65 or ht == 0x71) {
            pos.* += 1; return .{ .results = if (is_nullable) &[_]VT{.funcref} else &[_]VT{.nonfuncref} };
        }
        if (ht == 0x69 or ht == 0x68 or ht == 0x74) {
            pos.* += 1; return .{ .results = if (is_nullable) &[_]VT{.externref} else &[_]VT{.nonexternref} };
        }
        // Concrete type index (LEB128) — validate and treat as funcref/nonfuncref result
        const tir = leb128_mod.readUnsigned(u32, code[pos.*..]) catch return error.TypeMismatch;
        pos.* += tir.bytes_read;
        if (tir.value >= module_types.len) {
            std.debug.print("InvalidValType(readBlockType): tidx={d} types={d}\n", .{ tir.value, module_types.len });
            return error.InvalidValType;
        }
        return .{ .results = if (is_nullable) &[_]VT{.funcref} else &[_]VT{.nonfuncref}, .single_result_tidx = tir.value };
    }
    const r = leb128_mod.readSigned(i64, code[pos.*..]) catch return error.TypeMismatch;
    pos.* += r.bytes_read;
    if (r.value < 0) return error.TypeMismatch;
    const idx: usize = @intCast(r.value);
    if (idx < module_types.len) return .{ .params = module_types[idx].params, .results = module_types[idx].results, .type_idx = @intCast(idx) };
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

fn skipMemImm(code: []const u8, pos: *usize) u32 {
    const align_flags = readU32Leb(code, pos);
    const mem_idx: u32 = if (align_flags & 0x40 != 0) readU32Leb(code, pos) else 0;
    _ = readU32Leb(code, pos); // offset
    return mem_idx;
}

fn pushType(stack: []VT, sp: *u32, t: VT, tidx: []u32) void {
    if (sp.* < stack.len) { stack[sp.*] = t; tidx[sp.*] = NO_TIDX; sp.* += 1; }
}

/// Push a ref type with a concrete type index.
fn pushV(stack: []VT, sp: *u32, t: VT, tidx: []u32, concrete: u32) void {
    if (sp.* < stack.len) { stack[sp.*] = t; tidx[sp.*] = concrete; sp.* += 1; }
}

fn popExpect(stack: []VT, sp: *u32, expected: VT, cf: ?*CtrlFrame) bool {
    if (sp.* == 0 or (cf != null and sp.* <= cf.?.start_height)) {
        return cf != null and cf.?.unreachable_flag;
    }
    sp.* -= 1;
    const actual = stack[sp.*];
    if (actual == expected or actual.isSubtypeOf(expected)) return true;
    // In unreachable code, ref types are polymorphic
    if (cf != null and cf.?.unreachable_flag and actual.isRef() and expected.isRef()) return true;
    return false;
}

/// Pop and also check concrete type index compatibility.
fn popExpectTidx(stack: []VT, sp: *u32, expected: VT, expected_tidx: u32, cf: ?*CtrlFrame, tidx: []const u32) bool {
    if (sp.* == 0 or (cf != null and sp.* <= cf.?.start_height)) {
        return cf != null and cf.?.unreachable_flag;
    }
    sp.* -= 1;
    const actual = stack[sp.*];
    if (actual != expected and !actual.isSubtypeOf(expected)) {
        if (!(cf != null and cf.?.unreachable_flag and actual.isRef() and expected.isRef())) return false;
    }
    const actual_tidx = if (sp.* < tidx.len) tidx[sp.*] else NO_TIDX;
    if (expected_tidx != NO_TIDX) {
        if (actual_tidx != expected_tidx) {
            if (actual_tidx == NO_TIDX and cf != null and cf.?.unreachable_flag) return true;
            // Bottom types are subtypes of any concrete type in their family
            if (isBottomTidx(actual_tidx)) return true;
            return false;
        }
    }
    return true;
}

/// Strict version for branch label matching: (ref null $t) != funcref.
fn popExpectTidxStrict(stack: []VT, sp: *u32, expected: VT, expected_tidx: u32, cf: ?*CtrlFrame, tidx: []const u32) bool {
    if (sp.* == 0 or (cf != null and sp.* <= cf.?.start_height)) {
        return cf != null and cf.?.unreachable_flag;
    }
    sp.* -= 1;
    const actual = stack[sp.*];
    if (actual != expected and !actual.isSubtypeOf(expected)) {
        if (!(cf != null and cf.?.unreachable_flag and actual.isRef() and expected.isRef())) return false;
    }
    const actual_tidx = if (sp.* < tidx.len) tidx[sp.*] else NO_TIDX;
    if (expected_tidx != NO_TIDX) {
        if (actual_tidx != expected_tidx) {
            if (actual_tidx == NO_TIDX and cf != null and cf.?.unreachable_flag) return true;
            // Bottom types are subtypes of any concrete type in their family
            if (isBottomTidx(actual_tidx)) return true;
            return false;
        }
    } else if (actual_tidx != NO_TIDX and actual.isRef()) {
        // Bottom types are also subtypes of abstract ref types
        if (isBottomTidx(actual_tidx)) return true;
        if (!(cf != null and cf.?.unreachable_flag)) return false;
    }
    return true;
}

fn popAny(stack: []VT, sp: *u32, cf: ?*CtrlFrame) ?VT {
    if (sp.* == 0 or (cf != null and sp.* <= cf.?.start_height)) return null;
    sp.* -= 1;
    return stack[sp.*];
}

fn checkStackEnd(cf: *const CtrlFrame, stack: []const VT, sp: u32) bool {
    if (cf.unreachable_flag) {
        const expected = cf.end_types;
        if (sp > cf.start_height + expected.len) return false;
        for (expected, 0..) |t, j| {
            const stack_idx = cf.start_height + @as(u32, @intCast(j));
            if (stack_idx >= sp) continue;
            if (stack[stack_idx] != t and !stack[stack_idx].isSubtypeOf(t)) return false;
        }
        return true;
    }
    const expected = cf.end_types;
    if (sp != cf.start_height + expected.len) return false;
    for (expected, 0..) |t, j| {
        const idx = cf.start_height + @as(u32, @intCast(j));
        if (stack[idx] != t and !stack[idx].isSubtypeOf(t)) return false;
    }
    return true;
}

/// Also validate type indices in checkStackEnd.
fn checkStackEndTidx(cf: *const CtrlFrame, stack: []const VT, sp: u32, tidx: []const u32, module: *const types.WasmModule) bool {
    const expected = cf.end_types;
    const expected_tidxs = getEndTidxs(cf, module);

    if (cf.unreachable_flag) {
        if (sp > cf.start_height + expected.len) return false;
        for (expected, 0..) |t, j| {
            const stack_idx = cf.start_height + @as(u32, @intCast(j));
            if (stack_idx >= sp) continue;
            const actual = stack[stack_idx];
            // In unreachable code, ref types are polymorphic (any ref matches any ref)
            if (actual != t and !actual.isSubtypeOf(t) and !(actual.isRef() and t.isRef())) return false;
            const et = if (j < expected_tidxs.len) expected_tidxs[j] else NO_TIDX;
            if (et != NO_TIDX and stack_idx < tidx.len) {
                if (tidx[stack_idx] != et and tidx[stack_idx] != NO_TIDX and !isBottomTidx(tidx[stack_idx])) return false;
            }
        }
        return true;
    }
    if (sp != cf.start_height + expected.len) return false;
    for (expected, 0..) |t, j| {
        const idx = cf.start_height + @as(u32, @intCast(j));
        if (stack[idx] != t and !stack[idx].isSubtypeOf(t)) return false;
        const et = if (j < expected_tidxs.len) expected_tidxs[j] else NO_TIDX;
        if (et != NO_TIDX and idx < tidx.len) {
            if (tidx[idx] != et and !isBottomTidx(tidx[idx])) return false;
        }
    }
    return true;
}

/// Get the end type indices for a control frame.
fn getEndTidxs(cf: *const CtrlFrame, module: *const types.WasmModule) []const u32 {
    if (cf.block_type_idx != NO_TIDX and cf.block_type_idx < module.types.len) {
        return module.types[cf.block_type_idx].result_tidxs;
    }
    if (cf.single_result_tidx != NO_TIDX and cf.end_types.len == 1) {
        return @as(*const [1]u32, &cf.single_result_tidx);
    }
    return &.{};
}

/// Get the start (param) type indices for a control frame.
fn getStartTidxs(cf: *const CtrlFrame, module: *const types.WasmModule) []const u32 {
    if (cf.block_type_idx != NO_TIDX and cf.block_type_idx < module.types.len) {
        return module.types[cf.block_type_idx].param_tidxs;
    }
    return &.{};
}

/// Get label type indices for a branch target.
fn getLabelTidxs(cf: *const CtrlFrame, module: *const types.WasmModule) []const u32 {
    return if (cf.kind == .loop) getStartTidxs(cf, module) else getEndTidxs(cf, module);
}

fn doLoad(stack: []VT, sp: *u32, result: VT, cf: ?*CtrlFrame, tidx: []u32) LoadError!void {
    if (!popExpect(stack, sp, .i32, cf)) return error.TypeMismatch;
    pushType(stack, sp, result, tidx);
}

fn doLoad64(stack: []VT, sp: *u32, result: VT, addr_type: VT, cf: ?*CtrlFrame, tidx: []u32) LoadError!void {
    if (!popExpect(stack, sp, addr_type, cf)) return error.TypeMismatch;
    pushType(stack, sp, result, tidx);
}

fn doStore(stack: []VT, sp: *u32, val_type: VT, cf: ?*CtrlFrame) LoadError!void {
    if (!popExpect(stack, sp, val_type, cf)) return error.TypeMismatch;
    if (!popExpect(stack, sp, .i32, cf)) return error.TypeMismatch;
}

fn doStore64(stack: []VT, sp: *u32, val_type: VT, addr_type: VT, cf: ?*CtrlFrame) LoadError!void {
    if (!popExpect(stack, sp, val_type, cf)) return error.TypeMismatch;
    if (!popExpect(stack, sp, addr_type, cf)) return error.TypeMismatch;
}

fn doUnop(stack: []VT, sp: *u32, input: VT, output: VT, cf: ?*CtrlFrame, tidx: []u32) LoadError!void {
    if (!popExpect(stack, sp, input, cf)) return error.TypeMismatch;
    pushType(stack, sp, output, tidx);
}

fn doBinop(stack: []VT, sp: *u32, operand: VT, result: VT, cf: ?*CtrlFrame, tidx: []u32) LoadError!void {
    if (!popExpect(stack, sp, operand, cf)) return error.TypeMismatch;
    if (!popExpect(stack, sp, operand, cf)) return error.TypeMismatch;
    pushType(stack, sp, result, tidx);
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

/// Get the concrete type index for a global's value type.
fn getGlobalTidx(module: *const types.WasmModule, idx: u32) u32 {
    if (idx < module.import_global_count) {
        var gi: u32 = 0;
        for (module.imports) |imp| {
            if (imp.kind == .global) {
                if (gi == idx) return if (imp.global_type) |gt| gt.type_idx else NO_TIDX;
                gi += 1;
            }
        }
        return NO_TIDX;
    }
    const local_idx = idx - module.import_global_count;
    if (local_idx < module.globals.len) return module.globals[local_idx].global_type.type_idx;
    return NO_TIDX;
}

/// Get the address type for a table (i64 for table64, i32 otherwise).
fn getTableAddrType(module: *const types.WasmModule, idx: u32) VT {
    if (idx < module.import_table_count) {
        var ti: u32 = 0;
        for (module.imports) |imp| {
            if (imp.kind == .table) {
                if (ti == idx) return if (imp.table_type) |tt| (if (tt.is_table64) VT.i64 else VT.i32) else VT.i32;
                ti += 1;
            }
        }
        return .i32;
    }
    const local_idx = idx - module.import_table_count;
    if (local_idx < module.tables.len) return if (module.tables[local_idx].is_table64) VT.i64 else VT.i32;
    return .i32;
}

/// Get the address type for a memory (i64 for memory64, i32 otherwise).
fn getMemAddrType(module: *const types.WasmModule, idx: u32) VT {
    if (idx < module.import_memory_count) {
        var mi: u32 = 0;
        for (module.imports) |imp| {
            if (imp.kind == .memory) {
                if (mi == idx) return if (imp.memory_type) |mt| (if (mt.is_memory64) VT.i64 else VT.i32) else VT.i32;
                mi += 1;
            }
        }
        return .i32;
    }
    const local_idx = idx - module.import_memory_count;
    if (local_idx < module.memories.len) return if (module.memories[local_idx].is_memory64) VT.i64 else VT.i32;
    return .i32;
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

/// Get table element type index.
fn getTableElemTidx(module: *const types.WasmModule, idx: u32) u32 {
    if (idx < module.import_table_count) {
        var ti: u32 = 0;
        for (module.imports) |imp| {
            if (imp.kind == .table) {
                if (ti == idx) return if (imp.table_type) |tt| tt.elem_tidx else NO_TIDX;
                ti += 1;
            }
        }
        return NO_TIDX;
    }
    const local_idx = idx - module.import_table_count;
    if (local_idx < module.tables.len) return module.tables[local_idx].elem_tidx;
    return NO_TIDX;
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

/// Pop label types with strict type index checking for branch targets.
/// Unlike popExpectTidx used for calls (which allows subtyping),
/// branch target matching requires type indices to match when present.
fn popLabelTypesTidx(stack: []VT, sp: *u32, label_types: []const VT, label_tidxs: []const u32, cur_frame: ?*CtrlFrame, tidx: []const u32) LoadError!void {
    var ri = label_types.len;
    while (ri > 0) {
        ri -= 1;
        const et = if (ri < label_tidxs.len) label_tidxs[ri] else NO_TIDX;
        if (!popExpectTidxStrict(stack, sp, label_types[ri], et, cur_frame, tidx))
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
    var local_tidx_buf: [1024]u32 = .{NO_TIDX} ** 1024;
    var total_locals: u32 = @intCast(func_type.params.len);
    for (func_type.params, 0..) |p, li| {
        if (li >= local_types_buf.len) return;
        local_types_buf[li] = p;
        if (li < func_type.param_tidxs.len) local_tidx_buf[li] = func_type.param_tidxs[li];
    }
    for (func.locals) |ld| {
        var j: u32 = 0;
        while (j < ld.count) : (j += 1) {
            if (total_locals >= local_types_buf.len) return;
            local_types_buf[total_locals] = ld.val_type;
            local_tidx_buf[total_locals] = ld.type_idx;
            total_locals += 1;
        }
    }
    const local_types = local_types_buf[0..total_locals];
    const local_tidxs = local_tidx_buf[0..total_locals];

    var stack_buf: [4096]VT = undefined;
    var stack_tidx: [4096]u32 = .{NO_TIDX} ** 4096;
    var sp: u32 = 0;

    var ctrl_buf: [256]CtrlFrame = undefined;
    var ctrl_sp: u32 = 0;

    // Track initialization of non-nullable ref locals
    // Parameters are always initialized; only local declarations may be uninitialized
    var local_init_buf: [1024]bool = undefined;
    const param_count = @as(u32, @intCast(func_type.params.len));
    {
        var li: u32 = 0;
        while (li < total_locals and li < local_init_buf.len) : (li += 1) {
            local_init_buf[li] = li < param_count; // params are initialized
        }
    }
    // Save stack for local init state at block boundaries
    // Each block entry saves total_locals bools; max nesting = 256
    var init_save_stack: [256 * 64]bool = undefined;
    var init_save_sp: u32 = 0;
    const save_count = @min(total_locals, 1024);

    // Push the function frame
    ctrl_buf[0] = .{
        .kind = .function,
        .start_height = 0,
        .start_types = &.{},
        .end_types = func_type.results,
        .block_type_idx = func.type_idx,
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

            // Exception handling: throw/throw_ref — mark unreachable
            0x08 => { // throw: tagidx
                _ = readU32Leb(code, &i);
                if (ctrl_top.get(&ctrl_buf, ctrl_sp)) |cf| {
                    cf.unreachable_flag = true;
                    sp = cf.start_height;
                }
            },
            0x0A => { // throw_ref — mark unreachable
                if (ctrl_top.get(&ctrl_buf, ctrl_sp)) |cf| {
                    cf.unreachable_flag = true;
                    sp = cf.start_height;
                }
            },
            // Exception handling: try_table — like block
            0x06, 0x1F => {
                const bt = try readBlockType(code, &i, module.types);
                if (op == 0x1F) {
                    // Skip catch clause list
                    const clause_count = readU32Leb(code, &i);
                    var ci: u32 = 0;
                    while (ci < clause_count) : (ci += 1) {
                        const ck = readU32Leb(code, &i);
                        if (ck == 0 or ck == 1) _ = readU32Leb(code, &i); // tag index
                        _ = readU32Leb(code, &i); // label index
                    }
                }
                const start: u32 = sp;
                if (bt.params.len > 0) {
                    var pi = bt.params.len;
                    while (pi > 0) {
                        pi -= 1;
                        if (!popExpect(&stack_buf, &sp, bt.params[pi], ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                    }
                }
                if (ctrl_sp >= ctrl_buf.len) return error.TypeMismatch;
                ctrl_buf[ctrl_sp] = .{
                    .kind = .block,
                    .start_height = start,
                    .start_types = bt.params,
                    .end_types = bt.results,
                    .block_type_idx = bt.type_idx,
                    .single_result_tidx = bt.single_result_tidx,
                };
                for (bt.params) |p| pushType(&stack_buf, &sp, p, &stack_tidx);
                ctrl_sp += 1;
            },
            // Exception handling: catch/catch_all/rethrow/delegate — skip immediates
            0x07 => { _ = readU32Leb(code, &i); }, // catch: tagidx
            0x09, 0x19 => { _ = readU32Leb(code, &i); }, // rethrow/delegate: labelidx

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
                if (bt.type_idx != NO_TIDX and bt.type_idx < module.types.len) {
                    const ft_tidxs = module.types[bt.type_idx].param_tidxs;
                    for (bt.params, 0..) |p, pi| {
                        const pt = if (pi < ft_tidxs.len) ft_tidxs[pi] else NO_TIDX;
                        pushV(&stack_buf, &sp, p, &stack_tidx, pt);
                    }
                } else {
                    for (bt.params) |p| pushType(&stack_buf, &sp, p, &stack_tidx);
                }
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
                    .block_type_idx = bt.type_idx,
                    .single_result_tidx = bt.single_result_tidx,
                    .init_save_idx = init_save_sp,
                };
                // Save local init state before entering block
                if (save_count > 0 and init_save_sp + save_count <= init_save_stack.len) {
                    @memcpy(init_save_stack[init_save_sp..][0..save_count], local_init_buf[0..save_count]);
                    init_save_sp += save_count;
                }
                ctrl_sp += 1;
            },

            0x05 => { // else
                const cf = ctrl_top.get(&ctrl_buf, ctrl_sp) orelse return error.TypeMismatch;
                if (cf.kind != .@"if") return error.TypeMismatch;
                if (!checkStackEndTidx(cf, &stack_buf, sp, &stack_tidx, module))
                    return error.TypeMismatch;
                cf.has_else = true;
                cf.unreachable_flag = false;
                sp = cf.start_height;
                // Push block input types for the else branch
                const start_tidxs = getStartTidxs(cf, module);
                for (cf.start_types, 0..) |p, pi| {
                    const pt = if (pi < start_tidxs.len) start_tidxs[pi] else NO_TIDX;
                    pushV(&stack_buf, &sp, p, &stack_tidx, pt);
                }
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
                if (!checkStackEndTidx(cf, &stack_buf, sp, &stack_tidx, module))
                    return error.TypeMismatch;
                sp = cf.start_height;
                const end_tidxs = getEndTidxs(cf, module);
                for (cf.end_types, 0..) |t, j| {
                    if (sp >= stack_buf.len) return;
                    stack_buf[sp] = t;
                    stack_tidx[sp] = if (j < end_tidxs.len) end_tidxs[j] else NO_TIDX;
                    sp += 1;
                }
                // Restore local init state from before this block (not function frame)
                if (cf.kind != .function and save_count > 0 and cf.init_save_idx + save_count <= init_save_stack.len) {
                    @memcpy(local_init_buf[0..save_count], init_save_stack[cf.init_save_idx..][0..save_count]);
                    init_save_sp = cf.init_save_idx;
                }
                ctrl_sp -= 1;
                if (ctrl_sp == 0) return;
            },

            0x0C => { // br
                const label = readU32Leb(code, &i);
                if (ctrl_sp <= label) return error.TypeMismatch;
                const target = &ctrl_buf[ctrl_sp - 1 - label];
                try popLabelTypesTidx(&stack_buf, &sp, getLabelTypes(target), getLabelTidxs(target, module), ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx);
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
                const label_tidxs_slice = getLabelTidxs(target, module);
                // Per spec: br_if pops label types and re-pushes them.
                // In unreachable code, this materializes concrete types on the stack.
                try popLabelTypesTidx(&stack_buf, &sp, label_types, label_tidxs_slice, cf, &stack_tidx);
                for (label_types, 0..) |t, li| {
                    const lt = if (li < label_tidxs_slice.len) label_tidxs_slice[li] else NO_TIDX;
                    pushV(&stack_buf, &sp, t, &stack_tidx, lt);
                }
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
                                if (tt != dt and !tt.isSubtypeOf(dt) and !dt.isSubtypeOf(tt)) return error.TypeMismatch;
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
                try popLabelTypesTidx(&stack_buf, &sp, func_frame.end_types, getEndTidxs(func_frame, module), ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx);
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
                        const pt = if (pi < ft.param_tidxs.len) ft.param_tidxs[pi] else NO_TIDX;
                        if (!popExpectTidx(&stack_buf, &sp, ft.params[pi], pt, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx))
                            return error.TypeMismatch;
                    }
                    if (op == 0x12) {
                        // return_call: callee results must match caller's return type
                        const caller_results = ctrl_buf[0].end_types;
                        const caller_rtidxs = getEndTidxs(&ctrl_buf[0], module);
                        if (ft.results.len != caller_results.len) return error.TypeMismatch;
                        for (ft.results, caller_results, 0..) |cr, fr, ri| {
                            if (cr != fr and !cr.isSubtypeOf(fr)) return error.TypeMismatch;
                            const callee_rt = if (ri < ft.result_tidxs.len) ft.result_tidxs[ri] else NO_TIDX;
                            const caller_rt = if (ri < caller_rtidxs.len) caller_rtidxs[ri] else NO_TIDX;
                            if (caller_rt != NO_TIDX and callee_rt != caller_rt) return error.TypeMismatch;
                        }
                    } else {
                        for (ft.results, 0..) |rt, ri| {
                            const rt_tidx = if (ri < ft.result_tidxs.len) ft.result_tidxs[ri] else NO_TIDX;
                            pushV(&stack_buf, &sp, rt, &stack_tidx, rt_tidx);
                        }
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
                const tidx_ci = readU32Leb(code, &i);
                const table_idx = readU32Leb(code, &i);
                // call_indirect requires a funcref table (spec §3.3.8.8)
                if (getTableElemType(module, table_idx)) |et| {
                    if (!et.isFuncRef()) return error.TypeMismatch;
                }
                const tat_ci = getTableAddrType(module, table_idx);
                if (!popExpect(&stack_buf, &sp, tat_ci, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
                if (tidx_ci < module.types.len) {
                    const ft = module.types[tidx_ci];
                    var pi = ft.params.len;
                    while (pi > 0) {
                        pi -= 1;
                        const pt = if (pi < ft.param_tidxs.len) ft.param_tidxs[pi] else NO_TIDX;
                        if (!popExpectTidx(&stack_buf, &sp, ft.params[pi], pt, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx))
                            return error.TypeMismatch;
                    }
                    if (op == 0x13) {
                        // return_call_indirect: callee results must match caller's return type
                        const caller_results = ctrl_buf[0].end_types;
                        const caller_rtidxs = getEndTidxs(&ctrl_buf[0], module);
                        if (ft.results.len != caller_results.len) return error.TypeMismatch;
                        for (ft.results, caller_results, 0..) |cr, fr, ri| {
                            if (cr != fr and !cr.isSubtypeOf(fr)) return error.TypeMismatch;
                            const callee_rt = if (ri < ft.result_tidxs.len) ft.result_tidxs[ri] else NO_TIDX;
                            const caller_rt = if (ri < caller_rtidxs.len) caller_rtidxs[ri] else NO_TIDX;
                            if (caller_rt != NO_TIDX and callee_rt != caller_rt) return error.TypeMismatch;
                        }
                    } else {
                        for (ft.results, 0..) |rt, ri| {
                            const rt_tidx = if (ri < ft.result_tidxs.len) ft.result_tidxs[ri] else NO_TIDX;
                            pushV(&stack_buf, &sp, rt, &stack_tidx, rt_tidx);
                        }
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
            // Pop funcref (nullable or non-nullable), pop params, push results
            0x14, 0x15 => {
                const tidx_cr = readU32Leb(code, &i);
                // Pop the function reference — must be a concrete funcref matching the type index
                const ref_sp = sp;
                const ref_type = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                if (ref_type) |rt| {
                    if (!rt.isFuncRef()) return error.TypeMismatch;
                    // The ref must have concrete type index matching the call_ref type
                    const ref_tidx_val = if (ref_sp > 0 and ref_sp - 1 < stack_tidx.len) stack_tidx[ref_sp - 1] else NO_TIDX;
                    if (ref_tidx_val != tidx_cr) {
                        // In unreachable code, accept abstract (NO_TIDX) as polymorphic
                        const is_unreachable = if (ctrl_top.get(&ctrl_buf, ctrl_sp)) |cf| cf.unreachable_flag else false;
                        if (!(ref_tidx_val == NO_TIDX and is_unreachable))
                            return error.TypeMismatch;
                    }
                }
                if (tidx_cr < module.types.len) {
                    const ft = module.types[tidx_cr];
                    var pi = ft.params.len;
                    while (pi > 0) {
                        pi -= 1;
                        const pt = if (pi < ft.param_tidxs.len) ft.param_tidxs[pi] else NO_TIDX;
                        if (!popExpectTidx(&stack_buf, &sp, ft.params[pi], pt, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx))
                            return error.TypeMismatch;
                    }
                    if (op == 0x15) {
                        // return_call_ref: callee results must match caller's return type
                        const caller_results = ctrl_buf[0].end_types;
                        const caller_rtidxs = getEndTidxs(&ctrl_buf[0], module);
                        if (ft.results.len != caller_results.len) return error.TypeMismatch;
                        for (ft.results, caller_results, 0..) |cr, fr, ri| {
                            if (cr != fr and !cr.isSubtypeOf(fr)) return error.TypeMismatch;
                            const callee_rt = if (ri < ft.result_tidxs.len) ft.result_tidxs[ri] else NO_TIDX;
                            const caller_rt = if (ri < caller_rtidxs.len) caller_rtidxs[ri] else NO_TIDX;
                            if (caller_rt != NO_TIDX and callee_rt != caller_rt) return error.TypeMismatch;
                        }
                    } else {
                        for (ft.results, 0..) |rt, ri| {
                            const rt_tidx = if (ri < ft.result_tidxs.len) ft.result_tidxs[ri] else NO_TIDX;
                            pushV(&stack_buf, &sp, rt, &stack_tidx, rt_tidx);
                        }
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
                if (t1) |v| { if (v.isRef()) return error.TypeMismatch; }
                if (t2) |v| { if (v.isRef()) return error.TypeMismatch; }
                if (t1 != null and t2 != null and t1.? != t2.?) return error.TypeMismatch;
                pushType(&stack_buf, &sp, t1 orelse t2 orelse .i32, &stack_tidx);
            },
            0x1C => { // select_t
                const count_sel = readU32Leb(code, &i);
                if (count_sel == 0 or i >= code.len) return error.TypeMismatch;
                const type_byte = code[i];
                // Validate the type is a known valtype
                const sel_type: VT = switch (type_byte) {
                    0x7F, 0x7E, 0x7D, 0x7C, 0x70, 0x6F => @enumFromInt(type_byte),
                    else => return error.TypeMismatch,
                };
                i += count_sel;
                if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
                if (!popExpect(&stack_buf, &sp, sel_type, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
                if (!popExpect(&stack_buf, &sp, sel_type, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
                pushType(&stack_buf, &sp, sel_type, &stack_tidx);
            },

            // local.get
            0x20 => {
                const idx = readU32Leb(code, &i);
                if (idx < total_locals) {
                    // Non-nullable ref locals must be initialized before use
                    if ((local_types[idx] == .nonfuncref or local_types[idx] == .nonexternref) and
                        idx < local_init_buf.len and !local_init_buf[idx])
                        return error.TypeMismatch;
                    pushV(&stack_buf, &sp, local_types[idx], &stack_tidx, local_tidxs[idx]);
                }
            },
            // local.set
            0x21 => {
                const idx = readU32Leb(code, &i);
                if (idx < total_locals) {
                    if (!popExpectTidx(&stack_buf, &sp, local_types[idx], local_tidxs[idx], ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx))
                        return error.TypeMismatch;
                    if (idx < local_init_buf.len) local_init_buf[idx] = true;
                }
            },
            // local.tee
            0x22 => {
                const idx = readU32Leb(code, &i);
                if (idx < total_locals) {
                    if (idx < local_init_buf.len) local_init_buf[idx] = true;
                    if (!popExpectTidx(&stack_buf, &sp, local_types[idx], local_tidxs[idx], ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx))
                        return error.TypeMismatch;
                    pushV(&stack_buf, &sp, local_types[idx], &stack_tidx, local_tidxs[idx]);
                }
            },

            // global.get
            0x23 => {
                const gidx = readU32Leb(code, &i);
                if (getGlobalType(module, gidx)) |gt| pushV(&stack_buf, &sp, gt, &stack_tidx, getGlobalTidx(module, gidx));
            },
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

            // table.get: [addr] -> [t]
            0x25 => {
                const tidx_tg = readU32Leb(code, &i);
                const tat = getTableAddrType(module, tidx_tg);
                if (!popExpect(&stack_buf, &sp, tat, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
                if (getTableElemType(module, tidx_tg)) |et|
                    pushV(&stack_buf, &sp, et, &stack_tidx, getTableElemTidx(module, tidx_tg))
                else
                    pushType(&stack_buf, &sp, .funcref, &stack_tidx);
            },
            // table.set: [addr t] -> []
            0x26 => {
                const tidx_ts = readU32Leb(code, &i);
                const tat = getTableAddrType(module, tidx_ts);
                const et = getTableElemType(module, tidx_ts) orelse VT.funcref;
                if (!popExpect(&stack_buf, &sp, et, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
                if (!popExpect(&stack_buf, &sp, tat, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                    return error.TypeMismatch;
            },

            // Memory loads
            0x28 => { const mi = skipMemImm(code, &i); const at = getMemAddrType(module, mi); doLoad64(&stack_buf, &sp, .i32, at, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch; },
            0x29 => { const mi = skipMemImm(code, &i); const at = getMemAddrType(module, mi); doLoad64(&stack_buf, &sp, .i64, at, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch; },
            0x2A => { const mi = skipMemImm(code, &i); const at = getMemAddrType(module, mi); doLoad64(&stack_buf, &sp, .f32, at, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch; },
            0x2B => { const mi = skipMemImm(code, &i); const at = getMemAddrType(module, mi); doLoad64(&stack_buf, &sp, .f64, at, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch; },
            0x2C, 0x2D, 0x2E, 0x2F => { const mi = skipMemImm(code, &i); const at = getMemAddrType(module, mi); doLoad64(&stack_buf, &sp, .i32, at, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch; },
            0x30, 0x31, 0x32, 0x33, 0x34, 0x35 => { const mi = skipMemImm(code, &i); const at = getMemAddrType(module, mi); doLoad64(&stack_buf, &sp, .i64, at, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch; },

            // Memory stores
            0x36 => { const mi = skipMemImm(code, &i); const at = getMemAddrType(module, mi); doStore64(&stack_buf, &sp, .i32, at, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x37 => { const mi = skipMemImm(code, &i); const at = getMemAddrType(module, mi); doStore64(&stack_buf, &sp, .i64, at, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x38 => { const mi = skipMemImm(code, &i); const at = getMemAddrType(module, mi); doStore64(&stack_buf, &sp, .f32, at, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x39 => { const mi = skipMemImm(code, &i); const at = getMemAddrType(module, mi); doStore64(&stack_buf, &sp, .f64, at, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x3A, 0x3B => { const mi = skipMemImm(code, &i); const at = getMemAddrType(module, mi); doStore64(&stack_buf, &sp, .i32, at, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },
            0x3C, 0x3D, 0x3E => { const mi = skipMemImm(code, &i); const at = getMemAddrType(module, mi); doStore64(&stack_buf, &sp, .i64, at, ctrl_top.get(&ctrl_buf, ctrl_sp)) catch return error.TypeMismatch; },

            // memory.size
            0x3F => { const mi = readU32Leb(code, &i); const at = getMemAddrType(module, mi); pushType(&stack_buf, &sp, at, &stack_tidx); },
            // memory.grow
            0x40 => { const mi = readU32Leb(code, &i); const at = getMemAddrType(module, mi);
                if (!popExpect(&stack_buf, &sp, at, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                pushType(&stack_buf, &sp, at, &stack_tidx);
            },

            // Constants
            0x41 => { _ = readI32Leb(code, &i); pushType(&stack_buf, &sp, .i32, &stack_tidx); },
            0x42 => { _ = readI64Leb(code, &i); pushType(&stack_buf, &sp, .i64, &stack_tidx); },
            0x43 => { i += 4; pushType(&stack_buf, &sp, .f32, &stack_tidx); },
            0x44 => { i += 8; pushType(&stack_buf, &sp, .f64, &stack_tidx); },

            // Comparison ops (i32)
            0x45 => doUnop(&stack_buf, &sp, .i32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0x46...0x4F => doBinop(&stack_buf, &sp, .i32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0x50 => doUnop(&stack_buf, &sp, .i64, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0x51...0x5A => doBinop(&stack_buf, &sp, .i64, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0x5B...0x60 => doBinop(&stack_buf, &sp, .f32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0x61...0x66 => doBinop(&stack_buf, &sp, .f64, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,

            // Arithmetic ops (i32)
            0x67, 0x68, 0x69 => doUnop(&stack_buf, &sp, .i32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0x6A...0x78 => doBinop(&stack_buf, &sp, .i32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            // Arithmetic ops (i64)
            0x79, 0x7A, 0x7B => doUnop(&stack_buf, &sp, .i64, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0x7C...0x8A => doBinop(&stack_buf, &sp, .i64, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            // Arithmetic ops (f32)
            0x8B...0x91 => doUnop(&stack_buf, &sp, .f32, .f32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0x92...0x98 => doBinop(&stack_buf, &sp, .f32, .f32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            // Arithmetic ops (f64)
            0x99...0x9F => doUnop(&stack_buf, &sp, .f64, .f64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xA0...0xA6 => doBinop(&stack_buf, &sp, .f64, .f64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,

            // Conversion ops
            0xA7 => doUnop(&stack_buf, &sp, .i64, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xA8, 0xA9 => doUnop(&stack_buf, &sp, .f32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xAA, 0xAB => doUnop(&stack_buf, &sp, .f64, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xAC, 0xAD => doUnop(&stack_buf, &sp, .i32, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xAE, 0xAF => doUnop(&stack_buf, &sp, .f32, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xB0, 0xB1 => doUnop(&stack_buf, &sp, .f64, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xB2, 0xB3 => doUnop(&stack_buf, &sp, .i32, .f32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xB4, 0xB5 => doUnop(&stack_buf, &sp, .i64, .f32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xB6 => doUnop(&stack_buf, &sp, .f64, .f32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xB7, 0xB8 => doUnop(&stack_buf, &sp, .i32, .f64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xB9, 0xBA => doUnop(&stack_buf, &sp, .i64, .f64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xBB => doUnop(&stack_buf, &sp, .f32, .f64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,

            // Reinterpret ops
            0xBC => doUnop(&stack_buf, &sp, .f32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xBD => doUnop(&stack_buf, &sp, .f64, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xBE => doUnop(&stack_buf, &sp, .i32, .f32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xBF => doUnop(&stack_buf, &sp, .i64, .f64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,

            // Sign extension
            0xC0, 0xC1 => doUnop(&stack_buf, &sp, .i32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
            0xC2, 0xC3, 0xC4 => doUnop(&stack_buf, &sp, .i64, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,

            // Reference ops
            0xD0 => { // ref.null: skip heap type, push ref type with concrete tidx
                if (i < code.len) {
                    const ht = code[i];
                    i += 1;
                    if (ht == 0x6F) {
                        // extern → abstract externref
                        pushType(&stack_buf, &sp, .externref, &stack_tidx);
                    } else if (ht == 0x72 or ht == 0x74) {
                        // noextern, noexn(alt) → bottom of extern hierarchy
                        pushV(&stack_buf, &sp, .externref, &stack_tidx, BOTTOM_EXTERN_TIDX);
                    } else if (ht == 0x69) {
                        // exn → abstract externref
                        pushType(&stack_buf, &sp, .externref, &stack_tidx);
                    } else if (ht == 0x68) {
                        // noexn → bottom of extern hierarchy
                        pushV(&stack_buf, &sp, .externref, &stack_tidx, BOTTOM_EXTERN_TIDX);
                    } else if (ht == 0x70) {
                        // func → abstract funcref
                        pushType(&stack_buf, &sp, .funcref, &stack_tidx);
                    } else if (ht == 0x73 or ht == 0x71) {
                        // nofunc → bottom of func hierarchy
                        pushV(&stack_buf, &sp, .funcref, &stack_tidx, BOTTOM_FUNC_TIDX);
                    } else if (ht == 0x6E or ht == 0x6D or ht == 0x6C or ht == 0x6B or ht == 0x6A) {
                        // any, eq, i31, struct, array → abstract funcref
                        pushType(&stack_buf, &sp, .funcref, &stack_tidx);
                    } else if (ht == 0x65) {
                        // none → bottom of func hierarchy (internal bottom)
                        pushV(&stack_buf, &sp, .funcref, &stack_tidx, BOTTOM_FUNC_TIDX);
                    } else {
                        // Concrete type index: parse LEB128 and track it
                        var concrete_tidx: u32 = ht & 0x7F;
                        if (ht & 0x80 != 0) {
                            var shift: u5 = 7;
                            while (i < code.len) {
                                const cb = code[i];
                                concrete_tidx |= @as(u32, cb & 0x7F) << shift;
                                i += 1;
                                if (cb & 0x80 == 0) break;
                                shift +|= 7;
                            }
                        }
                        // Canonicalize the type index for iso-recursive equivalence
                        const canon_tidx = if (concrete_tidx < module.canonical_type_map.len)
                            module.canonical_type_map[concrete_tidx]
                        else
                            concrete_tidx;
                        pushV(&stack_buf, &sp, .funcref, &stack_tidx, canon_tidx);
                    }
                }
            },
            0xD1 => { _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp)); pushType(&stack_buf, &sp, .i32, &stack_tidx); },
            0xD2 => { // ref.func: push nonfuncref with function's type index
                const func_idx = readU32Leb(code, &i);
                const func_tidx = if (module.getFuncTypeIdx(func_idx)) |ti| ti else NO_TIDX;
                pushV(&stack_buf, &sp, .nonfuncref, &stack_tidx, func_tidx);
            },

            // ref.as_non_null (0xD4): pop nullable ref, push non-nullable ref (or trap)
            0xD4 => {
                const popped_tidx = if (sp > 0 and sp - 1 < stack_tidx.len) stack_tidx[sp - 1] else NO_TIDX;
                const popped = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                if (popped) |pt| {
                    if (pt == .externref or pt == .nonexternref)
                        pushType(&stack_buf, &sp, .nonexternref, &stack_tidx)
                    else
                        pushV(&stack_buf, &sp, .nonfuncref, &stack_tidx, popped_tidx);
                } else {
                    pushType(&stack_buf, &sp, .nonfuncref, &stack_tidx);
                }
            },
            // br_on_null (0xD5): pop ref, branch if null; push non-nullable if not branching
            // Spec: br_on_null $l : [t* (ref null ht)] -> [t* (ref ht)]
            // Must validate that t* matches label $l's types
            0xD5 => {
                const label = readU32Leb(code, &i);
                const popped_tidx_br = if (sp > 0 and sp - 1 < stack_tidx.len) stack_tidx[sp - 1] else NO_TIDX;
                const popped = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                // Must be a reference type
                if (popped) |pt| {
                    if (!pt.isRef()) return error.TypeMismatch;
                }
                // Check branch target types (values under the ref must match label types)
                if (ctrl_sp > label) {
                    const target = &ctrl_buf[ctrl_sp - 1 - label];
                    const label_types = getLabelTypes(target);
                    const label_tidxs_slice = getLabelTidxs(target, module);
                    // Peek: check and pop label types, then re-push them
                    try popLabelTypesTidx(&stack_buf, &sp, label_types, label_tidxs_slice, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx);
                    for (label_types, 0..) |t, li| {
                        const lt = if (li < label_tidxs_slice.len) label_tidxs_slice[li] else NO_TIDX;
                        pushV(&stack_buf, &sp, t, &stack_tidx, lt);
                    }
                }
                // Push non-nullable version for the non-branch path
                if (popped) |pt| {
                    if (pt.isExternRef())
                        pushType(&stack_buf, &sp, .nonexternref, &stack_tidx)
                    else
                        pushV(&stack_buf, &sp, .nonfuncref, &stack_tidx, popped_tidx_br);
                } else {
                    pushType(&stack_buf, &sp, .nonfuncref, &stack_tidx);
                }
            },
            // ref.eq (0xD3): pop 2 refs, push i32
            0xD3 => {
                _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                pushType(&stack_buf, &sp, .i32, &stack_tidx);
            },
            // br_on_non_null (0xD6): pop ref, branch if non-null
            // Spec: br_on_non_null $l : [t* (ref null ht)] -> [t*]
            // On branch: delivers [t* (ref ht)] to label $l
            0xD6 => {
                const label = readU32Leb(code, &i);
                const popped_tidx_bnn = if (sp > 0 and sp - 1 < stack_tidx.len) stack_tidx[sp - 1] else NO_TIDX;
                const popped_bnn = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                if (popped_bnn) |pt| {
                    if (!pt.isRef()) return error.TypeMismatch;
                }
                // Check branch target: on branch, stack has [t* (ref ht)]
                // Push non-nullable ref temporarily, peek-check label types, then remove it
                if (ctrl_sp > label) {
                    const save_sp = sp;
                    if (popped_bnn) |pt| {
                        if (pt.isExternRef())
                            pushType(&stack_buf, &sp, .nonexternref, &stack_tidx)
                        else
                            pushV(&stack_buf, &sp, .nonfuncref, &stack_tidx, popped_tidx_bnn);
                    } else {
                        pushType(&stack_buf, &sp, .nonfuncref, &stack_tidx);
                    }
                    const target = &ctrl_buf[ctrl_sp - 1 - label];
                    const label_types = getLabelTypes(target);
                    const label_tidxs_slice = getLabelTidxs(target, module);
                    try popLabelTypesTidx(&stack_buf, &sp, label_types, label_tidxs_slice, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx);
                    // Restore stack to non-branch state [t*] (without the ref)
                    sp = save_sp;
                }
                // Non-branch path: ref was null, stack has [t*] without the ref
            },

            // 0xFC prefix
            0xFC => {
                const sub = readU32Leb(code, &i);
                switch (sub) {
                    0, 1 => doUnop(&stack_buf, &sp, .f32, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
                    2, 3 => doUnop(&stack_buf, &sp, .f64, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
                    4, 5 => doUnop(&stack_buf, &sp, .f32, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
                    6, 7 => doUnop(&stack_buf, &sp, .f64, .i64, ctrl_top.get(&ctrl_buf, ctrl_sp), &stack_tidx) catch return error.TypeMismatch,
                    8 => { // memory.init: [at i32 at] -> []
                        _ = readU32Leb(code, &i);
                        const memidx = readU32Leb(code, &i);
                        const mat = getMemAddrType(module, memidx);
                        if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, mat, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                    },
                    9 => { _ = readU32Leb(code, &i); }, // data.drop: [] -> []
                    10 => { // memory.copy: [at_d at_s n] -> []
                        const dst_mi = readU32Leb(code, &i);
                        const src_mi = readU32Leb(code, &i);
                        const dat = getMemAddrType(module, dst_mi);
                        const sat = getMemAddrType(module, src_mi);
                        const nat: VT = if (dat == .i64 or sat == .i64) .i64 else .i32;
                        if (!popExpect(&stack_buf, &sp, nat, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, sat, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, dat, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                    },
                    11 => { // memory.fill: [at i32 at] -> []
                        const memidx = readU32Leb(code, &i);
                        const mat = getMemAddrType(module, memidx);
                        if (!popExpect(&stack_buf, &sp, mat, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, mat, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                    },
                    12 => { // table.init: [at i32 at] -> []
                        const elemidx = readU32Leb(code, &i);
                        const tableidx = readU32Leb(code, &i);
                        const tat = getTableAddrType(module, tableidx);
                        const ntables = module.import_table_count + @as(u32, @intCast(module.tables.len));
                        if (elemidx < module.elements.len and tableidx < ntables) {
                            const elem_kind = module.elements[elemidx].kind;
                            const tet = getTableElemType(module, tableidx) orelse VT.funcref;
                            if (elem_kind == .func_ref and tet.isExternRef()) return error.TypeMismatch;
                            if (elem_kind == .extern_ref and tet.isFuncRef()) return error.TypeMismatch;
                        }
                        if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, tat, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                    },
                    13 => { _ = readU32Leb(code, &i); }, // elem.drop: [] -> []
                    14 => { // table.copy: [at_d at_s at] -> []
                        const dst_tidx = readU32Leb(code, &i);
                        const src_tidx = readU32Leb(code, &i);
                        const dat = getTableAddrType(module, dst_tidx);
                        const sat = getTableAddrType(module, src_tidx);
                        // n is max of both address types
                        const nat: VT = if (dat == .i64 or sat == .i64) .i64 else .i32;
                        if (!popExpect(&stack_buf, &sp, nat, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, sat, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, dat, ctrl_top.get(&ctrl_buf, ctrl_sp))) return error.TypeMismatch;
                    },
                    15 => { // table.grow: [t at] -> [at]
                        const tidx = readU32Leb(code, &i);
                        const tat = getTableAddrType(module, tidx);
                        const et = getTableElemType(module, tidx) orelse VT.funcref;
                        if (!popExpect(&stack_buf, &sp, tat, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, et, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                        pushType(&stack_buf, &sp, tat, &stack_tidx);
                    },
                    16 => { // table.size: [] -> [at]
                        const tidx = readU32Leb(code, &i);
                        const tat = getTableAddrType(module, tidx);
                        pushType(&stack_buf, &sp, tat, &stack_tidx);
                    },
                    17 => { // table.fill: [at t at] -> []
                        const tidx = readU32Leb(code, &i);
                        const tat = getTableAddrType(module, tidx);
                        const et = getTableElemType(module, tidx) orelse VT.funcref;
                        if (!popExpect(&stack_buf, &sp, tat, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, et, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                        if (!popExpect(&stack_buf, &sp, tat, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                    },
                    else => {},
                }
            },

            // 0xFB prefix (GC opcodes)
            0xFB => {
                const sub = readU32Leb(code, &i);
                switch (sub) {
                    0x1C => { // ref.i31: [i32] -> [i31ref]
                        if (!popExpect(&stack_buf, &sp, .i32, ctrl_top.get(&ctrl_buf, ctrl_sp)))
                            return error.TypeMismatch;
                        pushType(&stack_buf, &sp, .i31ref, &stack_tidx);
                    },
                    0x1D => { // i31.get_s: [i31ref] -> [i32]
                        // Accept any ref type that could be i31ref
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .i32, &stack_tidx);
                    },
                    0x1E => { // i31.get_u: [i31ref] -> [i32]
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .i32, &stack_tidx);
                    },
                    0x1A => { // any.convert_extern: [externref] -> [anyref]
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .anyref, &stack_tidx);
                    },
                    0x1B => { // extern.convert_any: [anyref] -> [externref]
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .externref, &stack_tidx);
                    },
                    // struct/array/ref.test/ref.cast/br_on_cast: skip immediates, approximate types
                    0x00 => { // struct.new: [fields...] -> [structref]
                        _ = readU32Leb(code, &i);
                        pushType(&stack_buf, &sp, .structref, &stack_tidx);
                    },
                    0x01 => { // struct.new_default
                        _ = readU32Leb(code, &i);
                        pushType(&stack_buf, &sp, .structref, &stack_tidx);
                    },
                    0x02, 0x03, 0x04 => { // struct.get/get_s/get_u
                        _ = readU32Leb(code, &i); _ = readU32Leb(code, &i);
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .i32, &stack_tidx); // approximate
                    },
                    0x05 => { // struct.set
                        _ = readU32Leb(code, &i); _ = readU32Leb(code, &i);
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                    },
                    0x06, 0x07 => { // array.new, array.new_default
                        _ = readU32Leb(code, &i);
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .arrayref, &stack_tidx);
                    },
                    0x08 => { // array.new_fixed
                        _ = readU32Leb(code, &i); _ = readU32Leb(code, &i);
                        pushType(&stack_buf, &sp, .arrayref, &stack_tidx);
                    },
                    0x09, 0x0A => { // array.new_data, array.new_elem
                        _ = readU32Leb(code, &i); _ = readU32Leb(code, &i);
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .arrayref, &stack_tidx);
                    },
                    0x0B, 0x0C, 0x0D => { // array.get/get_s/get_u
                        _ = readU32Leb(code, &i);
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .i32, &stack_tidx); // approximate
                    },
                    0x0E => { // array.set
                        _ = readU32Leb(code, &i);
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                    },
                    0x0F => { // array.len: [arrayref] -> [i32]
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .i32, &stack_tidx);
                    },
                    0x10 => { // array.fill
                        _ = readU32Leb(code, &i);
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                    },
                    0x11 => { // array.copy
                        _ = readU32Leb(code, &i); _ = readU32Leb(code, &i);
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                    },
                    0x12, 0x13 => { // array.init_data, array.init_elem
                        _ = readU32Leb(code, &i); _ = readU32Leb(code, &i);
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                    },
                    0x14, 0x15 => { // ref.test, ref.test_nullable
                        _ = readU32Leb(code, &i);
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .i32, &stack_tidx);
                    },
                    0x16, 0x17 => { // ref.cast, ref.cast_nullable
                        _ = readU32Leb(code, &i);
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .anyref, &stack_tidx); // approximate
                    },
                    0x18, 0x19 => { // br_on_cast, br_on_cast_fail
                        _ = readU32Leb(code, &i); _ = readU32Leb(code, &i);
                        _ = readU32Leb(code, &i); _ = readU32Leb(code, &i);
                    },
                    else => {},
                }
            },

            // 0xFD prefix (SIMD opcodes)
            0xFD => {
                const sub = readU32Leb(code, &i);
                switch (sub) {
                    // v128.load variants: [addr] -> [v128]
                    0x00...0x0A => {
                        _ = readU32Leb(code, &i); _ = readU32Leb(code, &i);
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .v128, &stack_tidx);
                    },
                    0x0B => { // v128.store: [addr v128] -> []
                        _ = readU32Leb(code, &i); _ = readU32Leb(code, &i);
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                    },
                    0x0C => { i += 16; pushType(&stack_buf, &sp, .v128, &stack_tidx); }, // v128.const
                    0x0D => { // i8x16.shuffle
                        i += 16;
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .v128, &stack_tidx);
                    },
                    // splat ops: [scalar] -> [v128]
                    0x0F, 0x10, 0x11, 0x12, 0x13 => {
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .v128, &stack_tidx);
                    },
                    // extract_lane: [v128] -> [scalar]
                    0x15, 0x16, 0x17, 0x18, 0x19, 0x1B, 0x1D => {
                        i += 1;
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .i32, &stack_tidx);
                    },
                    0x1A => { i += 1; _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp)); pushType(&stack_buf, &sp, .i64, &stack_tidx); }, // i64x2.extract_lane
                    0x1C => { i += 1; _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp)); pushType(&stack_buf, &sp, .f32, &stack_tidx); }, // f32x4.extract_lane
                    0x1E => { i += 1; _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp)); pushType(&stack_buf, &sp, .f64, &stack_tidx); }, // f64x2.extract_lane
                    // replace_lane: [v128 scalar] -> [v128]
                    0x1F, 0x20, 0x21, 0x22 => {
                        i += 1;
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .v128, &stack_tidx);
                    },
                    // v128.load lane: [addr v128] -> [v128]
                    0x54...0x57 => {
                        _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); i += 1;
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .v128, &stack_tidx);
                    },
                    // v128.store lane: [addr v128] -> []
                    0x58...0x5B => {
                        _ = readU32Leb(code, &i); _ = readU32Leb(code, &i); i += 1;
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                    },
                    // v128.load zero: [addr] -> [v128]
                    0x5C, 0x5D => {
                        _ = readU32Leb(code, &i); _ = readU32Leb(code, &i);
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .v128, &stack_tidx);
                    },
                    // All other SIMD: treat as v128 unary (conservative but allows validation)
                    else => {
                        _ = popAny(&stack_buf, &sp, ctrl_top.get(&ctrl_buf, ctrl_sp));
                        pushType(&stack_buf, &sp, .v128, &stack_tidx);
                    },
                }
            },

            else => {},
        }
    }
    // Exception: if ctrl_sp == 1 (only function frame) and we consumed all bytes,
    // the trailing 0x0B end opcode may have been consumed as an instruction immediate
    // (e.g., br_on_null's label index in unreachable code). This is valid per spec.
    if (ctrl_sp > 1) return error.UnexpectedEnd;
    if (ctrl_sp == 1 and i < code.len) return error.UnexpectedEnd;
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
