//! Canonical ABI — lifting and lowering between component and core values.
//!
//! Implements the Canonical ABI specification for converting between
//! component-level interface types and core WebAssembly values.
//! See: https://github.com/WebAssembly/component-model/blob/main/design/mvp/CanonicalABI.md

const std = @import("std");
const ctypes = @import("types.zig");
const Allocator = std.mem.Allocator;

// ── Type registry ───────────────────────────────────────────────────────────

/// Resolves type indices to their definitions from a parsed Component.
pub const TypeRegistry = struct {
    types: []const ctypes.TypeDef,
    /// When non-empty, indexed by type-indexspace position to map to a
    /// local `types[]` index (or null for non-materialized slots).
    /// When empty, `get(idx)` falls back to direct indexing of `types`.
    indexspace: []const ?u32 = &.{},

    pub fn init(component: *const ctypes.Component) TypeRegistry {
        return .{ .types = component.types, .indexspace = component.type_indexspace };
    }

    pub fn fromTypes(types: []const ctypes.TypeDef) TypeRegistry {
        return .{ .types = types };
    }

    /// Resolve a type indexspace position to its TypeDef.
    pub fn get(self: TypeRegistry, idx: u32) ?ctypes.TypeDef {
        if (self.indexspace.len > 0) {
            if (idx >= self.indexspace.len) return null;
            const local = self.indexspace[idx] orelse return null;
            if (local >= self.types.len) return null;
            return self.types[local];
        }
        if (idx >= self.types.len) return null;
        return self.types[idx];
    }

    /// Resolve a ValType that carries a type index to its TypeDef.
    /// Returns null for primitives and unknown indices.
    pub fn resolve(self: TypeRegistry, t: ctypes.ValType) ?ctypes.TypeDef {
        return switch (t) {
            .record => |idx| self.get(idx),
            .variant => |idx| self.get(idx),
            .list => |idx| self.get(idx),
            .tuple => |idx| self.get(idx),
            .flags => |idx| self.get(idx),
            .enum_ => |idx| self.get(idx),
            .option => |idx| self.get(idx),
            .result => |idx| self.get(idx),
            .type_idx => |idx| self.get(idx),
            else => null,
        };
    }
};

// ── Interface value representation ──────────────────────────────────────────

/// Runtime representation of a component interface value.
/// Primitives are stored inline; compound types use allocator-owned slices.
pub const InterfaceValue = union(enum) {
    // Primitives
    bool: bool,
    s8: i8,
    u8: u8,
    s16: i16,
    u16: u16,
    s32: i32,
    u32: u32,
    s64: i64,
    u64: u64,
    f32: u32, // bit pattern
    f64: u64, // bit pattern
    char: u32, // Unicode code point
    handle: u32, // resource handle

    // Memory references (ptr+len into guest linear memory)
    string: PtrLen,
    list: PtrLen,

    // Compound values (fully lifted, allocator-owned)
    record_val: []const InterfaceValue,
    variant_val: VariantVal,
    list_val: []const InterfaceValue,
    tuple_val: []const InterfaceValue,
    flags_val: []const u32, // packed bitfields, ceil(n/32) words
    enum_val: u32, // discriminant index
    option_val: OptionVal,
    result_val: ResultVal,

    pub const PtrLen = struct {
        ptr: u32,
        len: u32,
    };

    pub const VariantVal = struct {
        discriminant: u32,
        payload: ?*const InterfaceValue,
    };

    pub const OptionVal = struct {
        is_some: bool,
        payload: ?*const InterfaceValue,
    };

    pub const ResultVal = struct {
        is_ok: bool,
        payload: ?*const InterfaceValue,
    };

    /// Recursively free all allocator-owned memory in this value.
    pub fn deinit(self: InterfaceValue, allocator: Allocator) void {
        switch (self) {
            .record_val => |fields| {
                for (fields) |f| f.deinit(allocator);
                allocator.free(fields);
            },
            .variant_val => |v| {
                if (v.payload) |p| {
                    p.*.deinit(allocator);
                    allocator.destroy(p);
                }
            },
            .list_val => |elems| {
                for (elems) |e| e.deinit(allocator);
                allocator.free(elems);
            },
            .tuple_val => |fields| {
                for (fields) |f| f.deinit(allocator);
                allocator.free(fields);
            },
            .flags_val => |words| allocator.free(words),
            .option_val => |o| {
                if (o.payload) |p| {
                    p.*.deinit(allocator);
                    allocator.destroy(p);
                }
            },
            .result_val => |r| {
                if (r.payload) |p| {
                    p.*.deinit(allocator);
                    allocator.destroy(p);
                }
            },
            // Primitives and PtrLen refs own no heap memory.
            else => {},
        }
    }
};

/// Typed flat core value for lift/lower (not just u32 — handles i64/f32/f64).
pub const CoreVal = union(enum) {
    i32: u32,
    i64: u64,
    f32: u32, // bit pattern
    f64: u64, // bit pattern

    /// Get as u32, truncating i64 or reinterpreting floats.
    pub fn asU32(self: CoreVal) u32 {
        return switch (self) {
            .i32 => |v| v,
            .i64 => |v| @truncate(v),
            .f32 => |v| v,
            .f64 => |v| @truncate(v),
        };
    }
};

// ── Alignment and size (primitive-only, no registry) ────────────────────────

/// Byte alignment for primitive and reference types. Returns 0 for compounds
/// (use `alignOfType` with a TypeRegistry for compound types).
pub fn alignment(t: ctypes.ValType) u32 {
    return switch (t) {
        .bool, .s8, .u8 => 1,
        .s16, .u16 => 2,
        .s32, .u32, .f32, .char => 4,
        .s64, .u64, .f64 => 8,
        .string => 4,
        .own, .borrow => 4,
        .list => 4,
        .record, .variant, .tuple, .flags, .enum_, .option, .result, .type_idx => 0,
    };
}

/// Byte size for primitive and reference types. Returns 0 for compounds
/// (use `sizeOfType` with a TypeRegistry for compound types).
pub fn elemSize(t: ctypes.ValType) u32 {
    return switch (t) {
        .bool, .s8, .u8 => 1,
        .s16, .u16 => 2,
        .s32, .u32, .f32, .char => 4,
        .s64, .u64, .f64 => 8,
        .string => 8,
        .own, .borrow => 4,
        .list => 8,
        .record, .variant, .tuple, .flags, .enum_, .option, .result, .type_idx => 0,
    };
}

// ── Compound layout (registry-aware) ────────────────────────────────────────

pub fn alignUp(offset: u32, al: u32) u32 {
    if (al == 0) return offset;
    return (offset + al - 1) & ~(al - 1);
}

/// Byte size of a discriminant for n cases.
pub fn discriminantSize(n_cases: usize) u32 {
    if (n_cases <= 0xFF) return 1;
    if (n_cases <= 0xFFFF) return 2;
    return 4;
}

/// Byte alignment for any ValType, resolving compounds via the registry.
pub fn alignOfType(reg: TypeRegistry, t: ctypes.ValType) u32 {
    // Primitives
    const prim = alignment(t);
    if (prim != 0) return prim;

    const td = reg.resolve(t) orelse return 4;
    return alignOfTypeDef(reg, td);
}

fn alignOfTypeDef(reg: TypeRegistry, td: ctypes.TypeDef) u32 {
    return switch (td) {
        .record => |r| blk: {
            var max_align: u32 = 1;
            for (r.fields) |f| {
                max_align = @max(max_align, alignOfType(reg, f.type));
            }
            break :blk max_align;
        },
        .variant => |v| blk: {
            const disc_align = discriminantSize(v.cases.len);
            var payload_align: u32 = 1;
            for (v.cases) |c| {
                if (c.type) |ct| {
                    payload_align = @max(payload_align, alignOfType(reg, ct));
                }
            }
            break :blk @max(disc_align, payload_align);
        },
        .list => 4, // ptr + len
        .tuple => |tup| blk: {
            var max_align: u32 = 1;
            for (tup.fields) |f| {
                max_align = @max(max_align, alignOfType(reg, f));
            }
            break :blk max_align;
        },
        .flags => |fl| if (fl.names.len <= 8) @as(u32, 1) else if (fl.names.len <= 16) @as(u32, 2) else @as(u32, 4),
        .enum_ => |en| discriminantSize(en.names.len),
        .option => |opt| @max(1, alignOfType(reg, opt.inner)),
        .result => |res| blk: {
            var payload_align: u32 = 1;
            if (res.ok) |ok| payload_align = @max(payload_align, alignOfType(reg, ok));
            if (res.err) |er| payload_align = @max(payload_align, alignOfType(reg, er));
            break :blk @max(1, payload_align);
        },
        .resource => 4,
        else => 4,
    };
}

/// Byte size of any ValType, resolving compounds via the registry.
pub fn sizeOfType(reg: TypeRegistry, t: ctypes.ValType) u32 {
    // Primitives
    const prim = elemSize(t);
    if (prim != 0) return prim;

    const td = reg.resolve(t) orelse return 4;
    return sizeOfTypeDef(reg, td);
}

fn sizeOfTypeDef(reg: TypeRegistry, td: ctypes.TypeDef) u32 {
    return switch (td) {
        .record => |r| blk: {
            var offset: u32 = 0;
            for (r.fields) |f| {
                const fa = alignOfType(reg, f.type);
                offset = alignUp(offset, fa);
                offset += sizeOfType(reg, f.type);
            }
            break :blk alignUp(offset, alignOfTypeDef(reg, td));
        },
        .variant => |v| blk: {
            const disc_sz = discriminantSize(v.cases.len);
            var payload_size: u32 = 0;
            var payload_align: u32 = 1;
            for (v.cases) |c| {
                if (c.type) |ct| {
                    payload_size = @max(payload_size, sizeOfType(reg, ct));
                    payload_align = @max(payload_align, alignOfType(reg, ct));
                }
            }
            const payload_offset = alignUp(disc_sz, payload_align);
            const total_align = @max(disc_sz, payload_align);
            break :blk alignUp(payload_offset + payload_size, total_align);
        },
        .list => 8, // ptr + len
        .tuple => |tup| blk: {
            var offset: u32 = 0;
            for (tup.fields) |f| {
                const fa = alignOfType(reg, f);
                offset = alignUp(offset, fa);
                offset += sizeOfType(reg, f);
            }
            break :blk alignUp(offset, alignOfTypeDef(reg, td));
        },
        .flags => |fl| blk: {
            const n = fl.names.len;
            if (n == 0) break :blk 0;
            if (n <= 8) break :blk 1;
            if (n <= 16) break :blk 2;
            break :blk @as(u32, @intCast(((n + 31) / 32) * 4));
        },
        .enum_ => |en| discriminantSize(en.names.len),
        .option => |opt| blk: {
            // Despecialize: variant { none, some(inner) }
            const inner_size = sizeOfType(reg, opt.inner);
            const inner_align = alignOfType(reg, opt.inner);
            const payload_offset = alignUp(1, inner_align);
            const total_align = @max(@as(u32, 1), inner_align);
            break :blk alignUp(payload_offset + inner_size, total_align);
        },
        .result => |res| blk: {
            var payload_size: u32 = 0;
            var payload_align: u32 = 1;
            if (res.ok) |ok| {
                payload_size = @max(payload_size, sizeOfType(reg, ok));
                payload_align = @max(payload_align, alignOfType(reg, ok));
            }
            if (res.err) |er| {
                payload_size = @max(payload_size, sizeOfType(reg, er));
                payload_align = @max(payload_align, alignOfType(reg, er));
            }
            const payload_offset = alignUp(1, payload_align);
            const total_align = @max(@as(u32, 1), payload_align);
            break :blk alignUp(payload_offset + payload_size, total_align);
        },
        .resource => 4,
        else => 4,
    };
}

// ── Flattening ──────────────────────────────────────────────────────────────

/// Maximum number of flattened parameter core values before spilling to memory.
pub const MAX_FLAT_PARAMS: u32 = 16;
/// Maximum number of flattened result core values before spilling to memory.
pub const MAX_FLAT_RESULTS: u32 = 1;

/// Flatten a primitive interface type to a sequence of core value types.
/// For compound types, use `flattenType` with a TypeRegistry.
pub fn flatten(t: ctypes.ValType) []const ctypes.CoreValType {
    return switch (t) {
        .bool, .s8, .u8, .s16, .u16, .s32, .u32, .char, .own, .borrow => &.{.i32},
        .s64, .u64 => &.{.i64},
        .f32 => &.{.f32},
        .f64 => &.{.f64},
        .string => &.{ .i32, .i32 },
        .list => &.{ .i32, .i32 },
        // Compound types: caller must use flattenType() with registry
        .enum_ => &.{.i32},
        .flags, .record, .variant, .tuple, .option, .result, .type_idx => &.{.i32},
    };
}

/// Count the number of flat core values for a type (registry-aware).
/// Used to decide flat vs spill strategy.
pub fn flattenCount(reg: TypeRegistry, t: ctypes.ValType) u32 {
    return switch (t) {
        .bool, .s8, .u8, .s16, .u16, .s32, .u32, .char, .own, .borrow => 1,
        .s64, .u64 => 1,
        .f32, .f64 => 1,
        .string => 2,
        .list => 2,
        .enum_ => 1,
        .flags => |idx| blk: {
            const td = reg.get(idx) orelse break :blk 1;
            const fl = td.flags;
            break :blk @as(u32, @intCast((fl.names.len + 31) / 32));
        },
        .record => |idx| blk: {
            const td = reg.get(idx) orelse break :blk 1;
            var count: u32 = 0;
            for (td.record.fields) |f| count += flattenCount(reg, f.type);
            break :blk if (count == 0) 1 else count;
        },
        .tuple => |idx| blk: {
            const td = reg.get(idx) orelse break :blk 1;
            var count: u32 = 0;
            for (td.tuple.fields) |f| count += flattenCount(reg, f);
            break :blk if (count == 0) 1 else count;
        },
        .variant => |idx| blk: {
            const td = reg.get(idx) orelse break :blk 1;
            var max_payload: u32 = 0;
            for (td.variant.cases) |c| {
                if (c.type) |ct| max_payload = @max(max_payload, flattenCount(reg, ct));
            }
            break :blk 1 + max_payload; // discriminant + joined payload
        },
        .option => |idx| blk: {
            const td = reg.get(idx) orelse break :blk 2;
            break :blk 1 + flattenCount(reg, td.option.inner); // disc + payload
        },
        .result => |idx| blk: {
            const td = reg.get(idx) orelse break :blk 2;
            var max_payload: u32 = 0;
            if (td.result.ok) |ok| max_payload = @max(max_payload, flattenCount(reg, ok));
            if (td.result.err) |er| max_payload = @max(max_payload, flattenCount(reg, er));
            break :blk 1 + max_payload; // disc + max payload
        },
        .type_idx => |idx| blk: {
            const td = reg.get(idx) orelse break :blk 1;
            break :blk flattenCountDef(reg, td);
        },
    };
}

pub fn flattenCountDef(reg: TypeRegistry, td: ctypes.TypeDef) u32 {
    return switch (td) {
        .record => |r| blk: {
            var count: u32 = 0;
            for (r.fields) |f| count += flattenCount(reg, f.type);
            break :blk if (count == 0) 1 else count;
        },
        .tuple => |tup| blk: {
            var count: u32 = 0;
            for (tup.fields) |f| count += flattenCount(reg, f);
            break :blk if (count == 0) 1 else count;
        },
        .variant => |v| blk: {
            var max_payload: u32 = 0;
            for (v.cases) |c| {
                if (c.type) |ct| max_payload = @max(max_payload, flattenCount(reg, ct));
            }
            break :blk 1 + max_payload;
        },
        .flags => |fl| @as(u32, @intCast(@max(@as(usize, 1), (fl.names.len + 31) / 32))),
        .enum_ => 1,
        .option => |opt| 1 + flattenCount(reg, opt.inner),
        .result => |res| blk: {
            var max_payload: u32 = 0;
            if (res.ok) |ok| max_payload = @max(max_payload, flattenCount(reg, ok));
            if (res.err) |er| max_payload = @max(max_payload, flattenCount(reg, er));
            break :blk 1 + max_payload;
        },
        .resource => 1,
        else => 1,
    };
}

// ── Loading from linear memory ──────────────────────────────────────────────

/// Load a primitive interface value from linear memory.
/// Compound types return error.CompoundNeedsRegistry.
pub fn loadVal(memory: []const u8, ptr: u32, t: ctypes.ValType) !InterfaceValue {
    return switch (t) {
        .bool => .{ .bool = loadU8(memory, ptr) != 0 },
        .s8 => .{ .s8 = @bitCast(loadU8(memory, ptr)) },
        .u8 => .{ .u8 = loadU8(memory, ptr) },
        .s16 => .{ .s16 = @bitCast(loadU16(memory, ptr)) },
        .u16 => .{ .u16 = loadU16(memory, ptr) },
        .s32 => .{ .s32 = @bitCast(loadU32(memory, ptr)) },
        .u32 => .{ .u32 = loadU32(memory, ptr) },
        .s64 => .{ .s64 = @bitCast(loadU64(memory, ptr)) },
        .u64 => .{ .u64 = loadU64(memory, ptr) },
        .f32 => .{ .f32 = @bitCast(loadU32(memory, ptr)) },
        .f64 => .{ .f64 = @bitCast(loadU64(memory, ptr)) },
        .char => .{ .char = loadU32(memory, ptr) },
        .own, .borrow => .{ .handle = loadU32(memory, ptr) },
        .string => .{ .string = .{
            .ptr = loadU32(memory, ptr),
            .len = loadU32(memory, ptr + 4),
        } },
        .list => .{ .list = .{
            .ptr = loadU32(memory, ptr),
            .len = loadU32(memory, ptr + 4),
        } },
        .record, .variant, .tuple, .flags, .enum_, .option, .result, .type_idx => error.CompoundNeedsRegistry,
    };
}

/// Errors that can occur during compound value loading/storing.
pub const LoadError = error{
    CompoundNeedsRegistry,
    InvalidTypeIndex,
    InvalidDiscriminant,
    OutOfMemory,
};

/// Load any value from linear memory, resolving compound types via registry.
pub fn loadValReg(memory: []const u8, ptr: u32, t: ctypes.ValType, reg: TypeRegistry, alloc: Allocator) LoadError!InterfaceValue {
    return switch (t) {
        // Primitives delegate to existing path
        .bool, .s8, .u8, .s16, .u16, .s32, .u32, .s64, .u64, .f32, .f64, .char, .own, .borrow, .string, .list => loadVal(memory, ptr, t),
        .enum_ => |idx| blk: {
            const td = reg.get(idx) orelse break :blk error.InvalidTypeIndex;
            const disc_sz = discriminantSize(td.enum_.names.len);
            const disc: u32 = switch (disc_sz) {
                1 => loadU8(memory, ptr),
                2 => loadU16(memory, ptr),
                else => loadU32(memory, ptr),
            };
            if (disc >= td.enum_.names.len) break :blk error.InvalidDiscriminant;
            break :blk .{ .enum_val = disc };
        },
        .flags => |idx| blk: {
            const td = reg.get(idx) orelse break :blk error.InvalidTypeIndex;
            const n_flags = td.flags.names.len;
            const n_words: u32 = @intCast(@max(@as(usize, 1), (n_flags + 31) / 32));
            const words = try alloc.alloc(u32, n_words);
            if (n_flags <= 8) {
                words[0] = loadU8(memory, ptr);
            } else if (n_flags <= 16) {
                words[0] = loadU16(memory, ptr);
            } else {
                for (0..n_words) |i| {
                    words[i] = loadU32(memory, ptr + @as(u32, @intCast(i)) * 4);
                }
            }
            break :blk .{ .flags_val = words };
        },
        .record => |idx| blk: {
            const td = reg.get(idx) orelse break :blk error.InvalidTypeIndex;
            const fields = td.record.fields;
            const vals = try alloc.alloc(InterfaceValue, fields.len);
            var offset: u32 = ptr;
            for (fields, 0..) |f, i| {
                const fa = alignOfType(reg, f.type);
                offset = alignUp(offset, fa);
                vals[i] = try loadValReg(memory, offset, f.type, reg, alloc);
                offset += sizeOfType(reg, f.type);
            }
            break :blk .{ .record_val = vals };
        },
        .tuple => |idx| blk: {
            const td = reg.get(idx) orelse break :blk error.InvalidTypeIndex;
            const fields = td.tuple.fields;
            const vals = try alloc.alloc(InterfaceValue, fields.len);
            var offset: u32 = ptr;
            for (fields, 0..) |f, i| {
                const fa = alignOfType(reg, f);
                offset = alignUp(offset, fa);
                vals[i] = try loadValReg(memory, offset, f, reg, alloc);
                offset += sizeOfType(reg, f);
            }
            break :blk .{ .tuple_val = vals };
        },
        .variant => |idx| blk: {
            const td = reg.get(idx) orelse break :blk error.InvalidTypeIndex;
            const cases = td.variant.cases;
            const disc_sz = discriminantSize(cases.len);
            const disc: u32 = switch (disc_sz) {
                1 => loadU8(memory, ptr),
                2 => loadU16(memory, ptr),
                else => loadU32(memory, ptr),
            };
            if (disc >= cases.len) break :blk error.InvalidDiscriminant;
            const payload_ptr = if (cases[disc].type) |ct| p: {
                const pa = alignOfType(reg, ct);
                break :p try allocPayload(alloc, try loadValReg(memory, alignUp(ptr + disc_sz, pa), ct, reg, alloc));
            } else null;
            break :blk .{ .variant_val = .{ .discriminant = disc, .payload = payload_ptr } };
        },
        .option => |idx| blk: {
            const td = reg.get(idx) orelse break :blk error.InvalidTypeIndex;
            const disc = loadU8(memory, ptr);
            if (disc > 1) break :blk error.InvalidDiscriminant;
            if (disc == 0) {
                break :blk .{ .option_val = .{ .is_some = false, .payload = null } };
            }
            const inner_align = alignOfType(reg, td.option.inner);
            const payload_offset = alignUp(ptr + 1, inner_align);
            const payload = try allocPayload(alloc, try loadValReg(memory, payload_offset, td.option.inner, reg, alloc));
            break :blk .{ .option_val = .{ .is_some = true, .payload = payload } };
        },
        .result => |idx| blk: {
            const td = reg.get(idx) orelse break :blk error.InvalidTypeIndex;
            const disc = loadU8(memory, ptr);
            if (disc > 1) break :blk error.InvalidDiscriminant;
            const is_ok = disc == 0;
            const payload_type = if (is_ok) td.result.ok else td.result.err;
            const payload_ptr = if (payload_type) |pt| p: {
                const pa = alignOfType(reg, pt);
                break :p try allocPayload(alloc, try loadValReg(memory, alignUp(ptr + 1, pa), pt, reg, alloc));
            } else null;
            break :blk .{ .result_val = .{ .is_ok = is_ok, .payload = payload_ptr } };
        },
        .type_idx => |idx| blk: {
            // Resolve the indirection and re-dispatch
            const td = reg.get(idx) orelse break :blk error.InvalidTypeIndex;
            break :blk loadValFromDef(memory, ptr, td, reg, alloc);
        },
    };
}

fn loadValFromDef(memory: []const u8, ptr: u32, td: ctypes.TypeDef, reg: TypeRegistry, alloc: Allocator) LoadError!InterfaceValue {
    return switch (td) {
        .record => |r| blk: {
            const vals = try alloc.alloc(InterfaceValue, r.fields.len);
            var offset: u32 = ptr;
            for (r.fields, 0..) |f, i| {
                const fa = alignOfType(reg, f.type);
                offset = alignUp(offset, fa);
                vals[i] = try loadValReg(memory, offset, f.type, reg, alloc);
                offset += sizeOfType(reg, f.type);
            }
            break :blk .{ .record_val = vals };
        },
        .tuple => |tup| blk: {
            const vals = try alloc.alloc(InterfaceValue, tup.fields.len);
            var offset: u32 = ptr;
            for (tup.fields, 0..) |f, i| {
                const fa = alignOfType(reg, f);
                offset = alignUp(offset, fa);
                vals[i] = try loadValReg(memory, offset, f, reg, alloc);
                offset += sizeOfType(reg, f);
            }
            break :blk .{ .tuple_val = vals };
        },
        .variant => |v| blk: {
            const disc_sz = discriminantSize(v.cases.len);
            const disc: u32 = switch (disc_sz) {
                1 => loadU8(memory, ptr),
                2 => loadU16(memory, ptr),
                else => loadU32(memory, ptr),
            };
            if (disc >= v.cases.len) break :blk error.InvalidDiscriminant;
            const payload_ptr = if (v.cases[disc].type) |ct| p: {
                const pa = alignOfType(reg, ct);
                break :p try allocPayload(alloc, try loadValReg(memory, alignUp(ptr + disc_sz, pa), ct, reg, alloc));
            } else null;
            break :blk .{ .variant_val = .{ .discriminant = disc, .payload = payload_ptr } };
        },
        .flags => |fl| blk: {
            const n_words: u32 = @intCast(@max(@as(usize, 1), (fl.names.len + 31) / 32));
            const words = try alloc.alloc(u32, n_words);
            if (fl.names.len <= 8) {
                words[0] = loadU8(memory, ptr);
            } else if (fl.names.len <= 16) {
                words[0] = loadU16(memory, ptr);
            } else {
                for (0..n_words) |i| {
                    words[i] = loadU32(memory, ptr + @as(u32, @intCast(i)) * 4);
                }
            }
            break :blk .{ .flags_val = words };
        },
        .enum_ => |en| blk: {
            const disc_sz = discriminantSize(en.names.len);
            const disc: u32 = switch (disc_sz) {
                1 => loadU8(memory, ptr),
                2 => loadU16(memory, ptr),
                else => loadU32(memory, ptr),
            };
            if (disc >= en.names.len) break :blk error.InvalidDiscriminant;
            break :blk .{ .enum_val = disc };
        },
        .option => |opt| blk: {
            const disc = loadU8(memory, ptr);
            if (disc > 1) break :blk error.InvalidDiscriminant;
            if (disc == 0) break :blk .{ .option_val = .{ .is_some = false, .payload = null } };
            const inner_align = alignOfType(reg, opt.inner);
            const payload = try allocPayload(alloc, try loadValReg(memory, alignUp(ptr + 1, inner_align), opt.inner, reg, alloc));
            break :blk .{ .option_val = .{ .is_some = true, .payload = payload } };
        },
        .result => |res| blk: {
            const disc = loadU8(memory, ptr);
            if (disc > 1) break :blk error.InvalidDiscriminant;
            const is_ok = disc == 0;
            const pt = if (is_ok) res.ok else res.err;
            const payload_ptr = if (pt) |payload_type| p: {
                const pa = alignOfType(reg, payload_type);
                break :p try allocPayload(alloc, try loadValReg(memory, alignUp(ptr + 1, pa), payload_type, reg, alloc));
            } else null;
            break :blk .{ .result_val = .{ .is_ok = is_ok, .payload = payload_ptr } };
        },
        .resource => .{ .handle = loadU32(memory, ptr) },
        else => .{ .handle = 0 },
    };
}

fn allocPayload(alloc: Allocator, val: InterfaceValue) !*const InterfaceValue {
    const p = try alloc.create(InterfaceValue);
    p.* = val;
    return p;
}

/// Store a primitive interface value to linear memory.
/// Compound types return error.CompoundNeedsRegistry.
pub fn storeVal(memory: []u8, ptr: u32, t: ctypes.ValType, val: InterfaceValue) !void {
    switch (t) {
        .bool => storeU8(memory, ptr, if (val.bool) 1 else 0),
        .s8 => storeU8(memory, ptr, @bitCast(val.s8)),
        .u8 => storeU8(memory, ptr, val.u8),
        .s16 => storeU16(memory, ptr, @bitCast(val.s16)),
        .u16 => storeU16(memory, ptr, val.u16),
        .s32 => storeU32(memory, ptr, @bitCast(val.s32)),
        .u32 => storeU32(memory, ptr, val.u32),
        .s64 => storeU64(memory, ptr, @bitCast(val.s64)),
        .u64 => storeU64(memory, ptr, val.u64),
        .f32 => storeU32(memory, ptr, @bitCast(val.f32)),
        .f64 => storeU64(memory, ptr, @bitCast(val.f64)),
        .char => storeU32(memory, ptr, val.char),
        .own, .borrow => storeU32(memory, ptr, val.handle),
        .string => {
            storeU32(memory, ptr, val.string.ptr);
            storeU32(memory, ptr + 4, val.string.len);
        },
        .list => {
            storeU32(memory, ptr, val.list.ptr);
            storeU32(memory, ptr + 4, val.list.len);
        },
        .record, .variant, .tuple, .flags, .enum_, .option, .result, .type_idx => return error.CompoundNeedsRegistry,
    }
}

/// Errors that can occur during compound value storing.
pub const StoreError = error{
    CompoundNeedsRegistry,
    InvalidTypeIndex,
};

/// Store any value to linear memory, resolving compound types via registry.
pub fn storeValReg(memory: []u8, ptr: u32, t: ctypes.ValType, val: InterfaceValue, reg: TypeRegistry) StoreError!void {
    switch (t) {
        .bool, .s8, .u8, .s16, .u16, .s32, .u32, .s64, .u64, .f32, .f64, .char, .own, .borrow, .string, .list => try storeVal(memory, ptr, t, val),
        .enum_ => {
            const idx = t.enum_;
            const td = reg.get(idx) orelse return error.InvalidTypeIndex;
            const disc_sz = discriminantSize(td.enum_.names.len);
            switch (disc_sz) {
                1 => storeU8(memory, ptr, @intCast(val.enum_val)),
                2 => storeU16(memory, ptr, @intCast(val.enum_val)),
                else => storeU32(memory, ptr, val.enum_val),
            }
        },
        .flags => {
            const idx = t.flags;
            const td = reg.get(idx) orelse return error.InvalidTypeIndex;
            const n_flags = td.flags.names.len;
            if (n_flags <= 8) {
                storeU8(memory, ptr, @intCast(val.flags_val[0]));
            } else if (n_flags <= 16) {
                storeU16(memory, ptr, @intCast(val.flags_val[0]));
            } else {
                const n_words = (n_flags + 31) / 32;
                for (0..n_words) |i| {
                    storeU32(memory, ptr + @as(u32, @intCast(i)) * 4, val.flags_val[i]);
                }
            }
        },
        .record => {
            const idx = t.record;
            const td = reg.get(idx) orelse return error.InvalidTypeIndex;
            const fields = td.record.fields;
            var offset: u32 = ptr;
            for (fields, 0..) |f, i| {
                const fa = alignOfType(reg, f.type);
                offset = alignUp(offset, fa);
                try storeValReg(memory, offset, f.type, val.record_val[i], reg);
                offset += sizeOfType(reg, f.type);
            }
        },
        .tuple => {
            const idx = t.tuple;
            const td = reg.get(idx) orelse return error.InvalidTypeIndex;
            const fields = td.tuple.fields;
            var offset: u32 = ptr;
            for (fields, 0..) |f, i| {
                const fa = alignOfType(reg, f);
                offset = alignUp(offset, fa);
                try storeValReg(memory, offset, f, val.tuple_val[i], reg);
                offset += sizeOfType(reg, f);
            }
        },
        .variant => {
            const idx = t.variant;
            const td = reg.get(idx) orelse return error.InvalidTypeIndex;
            const cases = td.variant.cases;
            const disc_sz = discriminantSize(cases.len);
            switch (disc_sz) {
                1 => storeU8(memory, ptr, @intCast(val.variant_val.discriminant)),
                2 => storeU16(memory, ptr, @intCast(val.variant_val.discriminant)),
                else => storeU32(memory, ptr, val.variant_val.discriminant),
            }
            if (val.variant_val.payload) |payload| {
                const ct = cases[val.variant_val.discriminant].type.?;
                const pa = alignOfType(reg, ct);
                try storeValReg(memory, alignUp(ptr + disc_sz, pa), ct, payload.*, reg);
            }
        },
        .option => {
            const idx = t.option;
            const td = reg.get(idx) orelse return error.InvalidTypeIndex;
            if (val.option_val.is_some) {
                storeU8(memory, ptr, 1);
                const inner_align = alignOfType(reg, td.option.inner);
                try storeValReg(memory, alignUp(ptr + 1, inner_align), td.option.inner, val.option_val.payload.?.*, reg);
            } else {
                storeU8(memory, ptr, 0);
            }
        },
        .result => {
            const idx = t.result;
            const td = reg.get(idx) orelse return error.InvalidTypeIndex;
            if (val.result_val.is_ok) {
                storeU8(memory, ptr, 0);
                if (val.result_val.payload) |payload| {
                    const ok_type = td.result.ok.?;
                    const pa = alignOfType(reg, ok_type);
                    try storeValReg(memory, alignUp(ptr + 1, pa), ok_type, payload.*, reg);
                }
            } else {
                storeU8(memory, ptr, 1);
                if (val.result_val.payload) |payload| {
                    const err_type = td.result.err.?;
                    const pa = alignOfType(reg, err_type);
                    try storeValReg(memory, alignUp(ptr + 1, pa), err_type, payload.*, reg);
                }
            }
        },
        .type_idx => {
            const idx = t.type_idx;
            const td = reg.get(idx) orelse return error.InvalidTypeIndex;
            try storeValFromDef(memory, ptr, td, val, reg);
        },
    }
}

fn storeValFromDef(memory: []u8, ptr: u32, td: ctypes.TypeDef, val: InterfaceValue, reg: TypeRegistry) StoreError!void {
    switch (td) {
        .record => |r| {
            var offset: u32 = ptr;
            for (r.fields, 0..) |f, i| {
                const fa = alignOfType(reg, f.type);
                offset = alignUp(offset, fa);
                try storeValReg(memory, offset, f.type, val.record_val[i], reg);
                offset += sizeOfType(reg, f.type);
            }
        },
        .tuple => |tup| {
            var offset: u32 = ptr;
            for (tup.fields, 0..) |f, i| {
                const fa = alignOfType(reg, f);
                offset = alignUp(offset, fa);
                try storeValReg(memory, offset, f, val.tuple_val[i], reg);
                offset += sizeOfType(reg, f);
            }
        },
        .variant => |v| {
            const disc_sz = discriminantSize(v.cases.len);
            switch (disc_sz) {
                1 => storeU8(memory, ptr, @intCast(val.variant_val.discriminant)),
                2 => storeU16(memory, ptr, @intCast(val.variant_val.discriminant)),
                else => storeU32(memory, ptr, val.variant_val.discriminant),
            }
            if (val.variant_val.payload) |payload| {
                const ct = v.cases[val.variant_val.discriminant].type.?;
                const pa = alignOfType(reg, ct);
                try storeValReg(memory, alignUp(ptr + disc_sz, pa), ct, payload.*, reg);
            }
        },
        .flags => |fl| {
            if (fl.names.len <= 8) {
                storeU8(memory, ptr, @intCast(val.flags_val[0]));
            } else if (fl.names.len <= 16) {
                storeU16(memory, ptr, @intCast(val.flags_val[0]));
            } else {
                const n_words = (fl.names.len + 31) / 32;
                for (0..n_words) |i| {
                    storeU32(memory, ptr + @as(u32, @intCast(i)) * 4, val.flags_val[i]);
                }
            }
        },
        .enum_ => |en| {
            const disc_sz = discriminantSize(en.names.len);
            switch (disc_sz) {
                1 => storeU8(memory, ptr, @intCast(val.enum_val)),
                2 => storeU16(memory, ptr, @intCast(val.enum_val)),
                else => storeU32(memory, ptr, val.enum_val),
            }
        },
        .option => |opt| {
            if (val.option_val.is_some) {
                storeU8(memory, ptr, 1);
                const inner_align = alignOfType(reg, opt.inner);
                try storeValReg(memory, alignUp(ptr + 1, inner_align), opt.inner, val.option_val.payload.?.*, reg);
            } else {
                storeU8(memory, ptr, 0);
            }
        },
        .result => |res| {
            if (val.result_val.is_ok) {
                storeU8(memory, ptr, 0);
                if (val.result_val.payload) |payload| {
                    const ok_type = res.ok.?;
                    const pa = alignOfType(reg, ok_type);
                    try storeValReg(memory, alignUp(ptr + 1, pa), ok_type, payload.*, reg);
                }
            } else {
                storeU8(memory, ptr, 1);
                if (val.result_val.payload) |payload| {
                    const err_type = res.err.?;
                    const pa = alignOfType(reg, err_type);
                    try storeValReg(memory, alignUp(ptr + 1, pa), err_type, payload.*, reg);
                }
            }
        },
        .resource => storeU32(memory, ptr, val.handle),
        else => {},
    }
}

// ── Flat lifting (primitive) ────────────────────────────────────────────────

/// Lift a flat sequence of core values to a primitive interface value.
/// Compound types return error.CompoundNeedsRegistry.
pub fn liftFlat(core_vals: []const u32, t: ctypes.ValType) !InterfaceValue {
    if (core_vals.len == 0) return error.EmptyCoreVals;
    return switch (t) {
        .bool => .{ .bool = core_vals[0] != 0 },
        .s8 => .{ .s8 = @bitCast(@as(u8, @truncate(core_vals[0]))) },
        .u8 => .{ .u8 = @truncate(core_vals[0]) },
        .s16 => .{ .s16 = @bitCast(@as(u16, @truncate(core_vals[0]))) },
        .u16 => .{ .u16 = @truncate(core_vals[0]) },
        .s32 => .{ .s32 = @bitCast(core_vals[0]) },
        .u32, .char => .{ .u32 = core_vals[0] },
        .s64 => .{ .s64 = @bitCast(if (core_vals.len > 1) @as(u64, core_vals[1]) << 32 | core_vals[0] else @as(u64, core_vals[0])) },
        .u64 => .{ .u64 = if (core_vals.len > 1) @as(u64, core_vals[1]) << 32 | core_vals[0] else core_vals[0] },
        .f32 => .{ .f32 = core_vals[0] },
        .f64 => .{ .f64 = if (core_vals.len > 1) @as(u64, core_vals[1]) << 32 | core_vals[0] else core_vals[0] },
        .own, .borrow => .{ .handle = core_vals[0] },
        .string => .{ .string = .{
            .ptr = core_vals[0],
            .len = if (core_vals.len > 1) core_vals[1] else 0,
        } },
        .list => .{ .list = .{
            .ptr = core_vals[0],
            .len = if (core_vals.len > 1) core_vals[1] else 0,
        } },
        .record, .variant, .tuple, .flags, .enum_, .option, .result, .type_idx => error.CompoundNeedsRegistry,
    };
}

/// Lower a primitive interface value to flat core values. Writes into `out`.
/// Returns the number of core values written, or error for compound types.
pub fn lowerFlat(val: InterfaceValue, t: ctypes.ValType, out: []u32) !u32 {
    switch (t) {
        .bool => {
            out[0] = if (val.bool) 1 else 0;
            return 1;
        },
        .s8 => {
            out[0] = @as(u32, @intCast(@as(u8, @bitCast(val.s8))));
            return 1;
        },
        .u8 => {
            out[0] = val.u8;
            return 1;
        },
        .s16 => {
            out[0] = @as(u32, @intCast(@as(u16, @bitCast(val.s16))));
            return 1;
        },
        .u16 => {
            out[0] = val.u16;
            return 1;
        },
        .s32 => {
            out[0] = @bitCast(val.s32);
            return 1;
        },
        .u32, .char => {
            out[0] = val.u32;
            return 1;
        },
        .s64 => {
            const bits: u64 = @bitCast(val.s64);
            out[0] = @truncate(bits);
            if (out.len > 1) out[1] = @truncate(bits >> 32);
            return 2;
        },
        .u64 => {
            out[0] = @truncate(val.u64);
            if (out.len > 1) out[1] = @truncate(val.u64 >> 32);
            return 2;
        },
        .f32 => {
            out[0] = val.f32;
            return 1;
        },
        .f64 => {
            out[0] = @truncate(val.f64);
            if (out.len > 1) out[1] = @truncate(val.f64 >> 32);
            return 2;
        },
        .own, .borrow => {
            out[0] = val.handle;
            return 1;
        },
        .string => {
            out[0] = val.string.ptr;
            if (out.len > 1) out[1] = val.string.len;
            return 2;
        },
        .list => {
            out[0] = val.list.ptr;
            if (out.len > 1) out[1] = val.list.len;
            return 2;
        },
        .record, .variant, .tuple, .flags, .enum_, .option, .result, .type_idx => return error.CompoundNeedsRegistry,
    }
}

// ── String encoding ─────────────────────────────────────────────────────────

/// Validate and measure a UTF-8 encoded string in linear memory.
pub fn validateUtf8(memory: []const u8, ptr: u32, len: u32) bool {
    if (ptr + len > memory.len) return false;
    return std.unicode.utf8ValidateSlice(memory[ptr .. ptr + len]);
}

/// Transcode UTF-8 bytes to UTF-16LE, writing into `out`.
/// Returns the number of u16 code units written.
pub fn utf8ToUtf16(src: []const u8, out: []u16) !u32 {
    var written: u32 = 0;
    var view = std.unicode.Utf8View.initUnchecked(src);
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| {
        if (written >= out.len) return error.BufferTooSmall;
        if (cp <= 0xFFFF) {
            out[written] = @intCast(cp);
            written += 1;
        } else {
            // Surrogate pair
            if (written + 1 >= out.len) return error.BufferTooSmall;
            const adj = cp - 0x10000;
            out[written] = @intCast(0xD800 + (adj >> 10));
            out[written + 1] = @intCast(0xDC00 + (adj & 0x3FF));
            written += 2;
        }
    }
    return written;
}

/// Decode UTF-16LE bytes from linear memory into a UTF-8 string.
/// Returns a newly-allocated UTF-8 byte slice.
pub fn utf16ToUtf8(mem: []const u8, ptr: u32, code_units: u32, allocator: Allocator) ![]u8 {
    const byte_len = @as(u32, code_units) * 2;
    if (ptr + byte_len > mem.len) return error.BufferTooSmall;
    const slice = mem[ptr .. ptr + byte_len];

    // First pass: measure UTF-8 length
    var utf8_len: usize = 0;
    var i: usize = 0;
    while (i < slice.len) : (i += 2) {
        const cu = std.mem.readInt(u16, slice[i..][0..2], .little);
        const cp = blk: {
            if (cu >= 0xD800 and cu <= 0xDBFF) {
                // High surrogate — read low surrogate
                if (i + 2 >= slice.len) return error.BufferTooSmall;
                i += 2;
                const lo = std.mem.readInt(u16, slice[i..][0..2], .little);
                if (lo < 0xDC00 or lo > 0xDFFF) return error.BufferTooSmall;
                break :blk @as(u21, @intCast((@as(u32, cu - 0xD800) << 10) + (lo - 0xDC00) + 0x10000));
            }
            break :blk @as(u21, @intCast(cu));
        };
        utf8_len += std.unicode.utf8CodepointSequenceLength(cp) catch return error.BufferTooSmall;
    }

    // Second pass: encode
    const result = try allocator.alloc(u8, utf8_len);
    errdefer allocator.free(result);
    var out_i: usize = 0;
    i = 0;
    while (i < slice.len) : (i += 2) {
        const cu = std.mem.readInt(u16, slice[i..][0..2], .little);
        const cp = blk: {
            if (cu >= 0xD800 and cu <= 0xDBFF) {
                i += 2;
                const lo = std.mem.readInt(u16, slice[i..][0..2], .little);
                break :blk @as(u21, @intCast((@as(u32, cu - 0xD800) << 10) + (lo - 0xDC00) + 0x10000));
            }
            break :blk @as(u21, @intCast(cu));
        };
        const n = std.unicode.utf8Encode(cp, result[out_i..]) catch return error.BufferTooSmall;
        out_i += n;
    }
    return result;
}

/// latin1+utf16 tagged encoding: bit 31 of length indicates encoding.
/// If bit 31 is clear: Latin-1 (each byte = one code point, all ≤ 0xFF).
/// If bit 31 is set: UTF-16LE, lower 31 bits = code unit count.
pub const LATIN1_UTF16_TAG: u32 = 0x8000_0000;

/// Decode a latin1+utf16 tagged string from linear memory into UTF-8.
pub fn decodeLatin1Utf16(mem: []const u8, ptr: u32, tagged_len: u32, allocator: Allocator) ![]u8 {
    if (tagged_len & LATIN1_UTF16_TAG != 0) {
        // UTF-16 encoded
        const code_units = tagged_len & ~LATIN1_UTF16_TAG;
        return utf16ToUtf8(mem, ptr, code_units, allocator);
    } else {
        // Latin-1 encoded: each byte is a code point ≤ 0xFF
        if (ptr + tagged_len > mem.len) return error.BufferTooSmall;
        const latin1 = mem[ptr .. ptr + tagged_len];
        // Latin-1 → UTF-8: code points 0x80-0xFF become 2-byte sequences
        var utf8_len: usize = 0;
        for (latin1) |b| {
            utf8_len += if (b < 0x80) @as(usize, 1) else @as(usize, 2);
        }
        const result = try allocator.alloc(u8, utf8_len);
        errdefer allocator.free(result);
        var out_i: usize = 0;
        for (latin1) |b| {
            if (b < 0x80) {
                result[out_i] = b;
                out_i += 1;
            } else {
                result[out_i] = 0xC0 | (b >> 6);
                result[out_i + 1] = 0x80 | (b & 0x3F);
                out_i += 2;
            }
        }
        return result;
    }
}

/// Encode a UTF-8 string into UTF-16LE, writing to linear memory at `ptr`.
/// Returns number of UTF-16 code units written.
pub fn encodeUtf16ToMem(mem: []u8, ptr: u32, src: []const u8) !u32 {
    var written: u32 = 0;
    var view = std.unicode.Utf8View.initUnchecked(src);
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| {
        if (cp <= 0xFFFF) {
            if (ptr + written * 2 + 2 > mem.len) return error.BufferTooSmall;
            storeU16(mem, ptr + written * 2, @intCast(cp));
            written += 1;
        } else {
            if (ptr + written * 2 + 4 > mem.len) return error.BufferTooSmall;
            const adj = cp - 0x10000;
            storeU16(mem, ptr + written * 2, @intCast(0xD800 + (adj >> 10)));
            storeU16(mem, ptr + written * 2 + 2, @intCast(0xDC00 + (adj & 0x3FF)));
            written += 2;
        }
    }
    return written;
}

// ── Error set ───────────────────────────────────────────────────────────────

pub const AbiError = error{
    CompoundNeedsRegistry,
    InvalidTypeIndex,
    InvalidDiscriminant,
    EmptyCoreVals,
    BufferTooSmall,
    OutOfMemory,
};

// ── Memory helpers ──────────────────────────────────────────────────────────

fn loadU8(mem: []const u8, ptr: u32) u8 {
    if (ptr >= mem.len) return 0;
    return mem[ptr];
}

fn loadU16(mem: []const u8, ptr: u32) u16 {
    if (ptr + 2 > mem.len) return 0;
    return std.mem.readInt(u16, mem[ptr..][0..2], .little);
}

fn loadU32(mem: []const u8, ptr: u32) u32 {
    if (ptr + 4 > mem.len) return 0;
    return std.mem.readInt(u32, mem[ptr..][0..4], .little);
}

fn loadU64(mem: []const u8, ptr: u32) u64 {
    if (ptr + 8 > mem.len) return 0;
    return std.mem.readInt(u64, mem[ptr..][0..8], .little);
}

fn storeU8(mem: []u8, ptr: u32, val: u8) void {
    if (ptr >= mem.len) return;
    mem[ptr] = val;
}

fn storeU16(mem: []u8, ptr: u32, val: u16) void {
    if (ptr + 2 > mem.len) return;
    std.mem.writeInt(u16, mem[ptr..][0..2], val, .little);
}

fn storeU32(mem: []u8, ptr: u32, val: u32) void {
    if (ptr + 4 > mem.len) return;
    std.mem.writeInt(u32, mem[ptr..][0..4], val, .little);
}

fn storeU64(mem: []u8, ptr: u32, val: u64) void {
    if (ptr + 8 > mem.len) return;
    std.mem.writeInt(u64, mem[ptr..][0..8], val, .little);
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "alignment: primitive types" {
    try std.testing.expectEqual(@as(u32, 1), alignment(.bool));
    try std.testing.expectEqual(@as(u32, 4), alignment(.s32));
    try std.testing.expectEqual(@as(u32, 8), alignment(.f64));
    try std.testing.expectEqual(@as(u32, 4), alignment(.string));
    try std.testing.expectEqual(@as(u32, 4), alignment(.{ .own = 0 }));
}

test "alignment: compounds return 0 without registry" {
    try std.testing.expectEqual(@as(u32, 0), alignment(.{ .record = 0 }));
    try std.testing.expectEqual(@as(u32, 0), alignment(.{ .variant = 0 }));
}

test "elemSize: primitive types" {
    try std.testing.expectEqual(@as(u32, 1), elemSize(.bool));
    try std.testing.expectEqual(@as(u32, 4), elemSize(.u32));
    try std.testing.expectEqual(@as(u32, 8), elemSize(.string));
    try std.testing.expectEqual(@as(u32, 8), elemSize(.f64));
}

test "elemSize: compounds return 0 without registry" {
    try std.testing.expectEqual(@as(u32, 0), elemSize(.{ .record = 0 }));
}

test "flatten: basic types" {
    const flat_i32 = flatten(.s32);
    try std.testing.expectEqual(@as(usize, 1), flat_i32.len);
    try std.testing.expectEqual(ctypes.CoreValType.i32, flat_i32[0]);

    const flat_str = flatten(.string);
    try std.testing.expectEqual(@as(usize, 2), flat_str.len);
}

test "loadVal/storeVal: i32 roundtrip" {
    var mem = [_]u8{0} ** 16;
    try storeVal(&mem, 0, .s32, .{ .s32 = -42 });
    const val = try loadVal(&mem, 0, .s32);
    try std.testing.expectEqual(@as(i32, -42), val.s32);
}

test "loadVal/storeVal: string roundtrip" {
    var mem = [_]u8{0} ** 16;
    try storeVal(&mem, 0, .string, .{ .string = .{ .ptr = 100, .len = 5 } });
    const val = try loadVal(&mem, 0, .string);
    try std.testing.expectEqual(@as(u32, 100), val.string.ptr);
    try std.testing.expectEqual(@as(u32, 5), val.string.len);
}

test "loadVal: compound types return error" {
    var mem = [_]u8{0} ** 16;
    try std.testing.expectError(error.CompoundNeedsRegistry, loadVal(&mem, 0, .{ .record = 0 }));
    try std.testing.expectError(error.CompoundNeedsRegistry, loadVal(&mem, 0, .{ .variant = 0 }));
}

test "liftFlat/lowerFlat: bool roundtrip" {
    const vals = [_]u32{1};
    const lifted = try liftFlat(&vals, .bool);
    try std.testing.expect(lifted.bool);

    var out: [2]u32 = undefined;
    const count = try lowerFlat(lifted, .bool, &out);
    try std.testing.expectEqual(@as(u32, 1), count);
    try std.testing.expectEqual(@as(u32, 1), out[0]);
}

test "liftFlat/lowerFlat: string roundtrip" {
    const vals = [_]u32{ 200, 10 };
    const lifted = try liftFlat(&vals, .string);
    try std.testing.expectEqual(@as(u32, 200), lifted.string.ptr);
    try std.testing.expectEqual(@as(u32, 10), lifted.string.len);

    var out: [2]u32 = undefined;
    const count = try lowerFlat(lifted, .string, &out);
    try std.testing.expectEqual(@as(u32, 2), count);
    try std.testing.expectEqual(@as(u32, 200), out[0]);
    try std.testing.expectEqual(@as(u32, 10), out[1]);
}

test "liftFlat: compound types return error" {
    const vals = [_]u32{0};
    try std.testing.expectError(error.CompoundNeedsRegistry, liftFlat(&vals, .{ .record = 0 }));
}

test "validateUtf8: valid and invalid" {
    const mem = "Hello, world!";
    try std.testing.expect(validateUtf8(mem, 0, 13));
    try std.testing.expect(!validateUtf8(mem, 0, 100)); // out of bounds
}

test "utf8ToUtf16: basic ASCII" {
    const src = "Hi";
    var out: [4]u16 = undefined;
    const written = try utf8ToUtf16(src, &out);
    try std.testing.expectEqual(@as(u32, 2), written);
    try std.testing.expectEqual(@as(u16, 'H'), out[0]);
    try std.testing.expectEqual(@as(u16, 'i'), out[1]);
}

// ── Compound type layout tests ──────────────────────────────────────────────

test "TypeRegistry: resolve compound types" {
    const types = [_]ctypes.TypeDef{
        .{ .record = .{ .fields = &.{
            .{ .name = "x", .type = .s32 },
            .{ .name = "y", .type = .s32 },
        } } },
        .{ .enum_ = .{ .names = &.{ "a", "b", "c" } } },
    };
    const reg = TypeRegistry.fromTypes(&types);

    try std.testing.expect(reg.get(0) != null);
    try std.testing.expect(reg.get(1) != null);
    try std.testing.expect(reg.get(99) == null);
    try std.testing.expect(reg.resolve(.{ .record = 0 }) != null);
    try std.testing.expect(reg.resolve(.s32) == null);
}

test "discriminantSize" {
    try std.testing.expectEqual(@as(u32, 1), discriminantSize(2));
    try std.testing.expectEqual(@as(u32, 1), discriminantSize(255));
    try std.testing.expectEqual(@as(u32, 2), discriminantSize(256));
    try std.testing.expectEqual(@as(u32, 2), discriminantSize(65535));
    try std.testing.expectEqual(@as(u32, 4), discriminantSize(65536));
}

test "sizeOfType/alignOfType: record {s32, s32}" {
    const types = [_]ctypes.TypeDef{
        .{ .record = .{ .fields = &.{
            .{ .name = "x", .type = .s32 },
            .{ .name = "y", .type = .s32 },
        } } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    try std.testing.expectEqual(@as(u32, 8), sizeOfType(reg, .{ .record = 0 }));
    try std.testing.expectEqual(@as(u32, 4), alignOfType(reg, .{ .record = 0 }));
}

test "sizeOfType/alignOfType: record {u8, u32, u8}" {
    const types = [_]ctypes.TypeDef{
        .{ .record = .{ .fields = &.{
            .{ .name = "a", .type = .u8 },
            .{ .name = "b", .type = .u32 },
            .{ .name = "c", .type = .u8 },
        } } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    // u8 at 0, pad to 4, u32 at 4, u8 at 8 → size 9 → aligned to 4 → 12
    try std.testing.expectEqual(@as(u32, 12), sizeOfType(reg, .{ .record = 0 }));
    try std.testing.expectEqual(@as(u32, 4), alignOfType(reg, .{ .record = 0 }));
}

test "sizeOfType: enum with 3 cases" {
    const types = [_]ctypes.TypeDef{
        .{ .enum_ = .{ .names = &.{ "a", "b", "c" } } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    try std.testing.expectEqual(@as(u32, 1), sizeOfType(reg, .{ .enum_ = 0 }));
}

test "sizeOfType: flags with 33 names" {
    const names = [_][]const u8{"f"} ** 33;
    const types = [_]ctypes.TypeDef{
        .{ .flags = .{ .names = &names } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    // 33 flags → ceil(33/32) = 2 words → 8 bytes
    try std.testing.expectEqual(@as(u32, 8), sizeOfType(reg, .{ .flags = 0 }));
}

test "sizeOfType: option<u32>" {
    const types = [_]ctypes.TypeDef{
        .{ .option = .{ .inner = .u32 } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    // disc(1) + pad to align 4 → offset 4, payload 4 → 8
    try std.testing.expectEqual(@as(u32, 8), sizeOfType(reg, .{ .option = 0 }));
    try std.testing.expectEqual(@as(u32, 4), alignOfType(reg, .{ .option = 0 }));
}

test "sizeOfType: result<u32, u8>" {
    const types = [_]ctypes.TypeDef{
        .{ .result = .{ .ok = .u32, .err = .u8 } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    // disc(1) + pad to 4 → offset 4, max payload = 4 → 8
    try std.testing.expectEqual(@as(u32, 8), sizeOfType(reg, .{ .result = 0 }));
}

test "sizeOfType: variant with mixed payloads" {
    const types = [_]ctypes.TypeDef{
        .{ .variant = .{ .cases = &.{
            .{ .name = "a", .type = .u8 },
            .{ .name = "b", .type = .f64 },
            .{ .name = "c", .type = null },
        } } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    // disc 1 byte, max payload align=8 (f64), payload_offset=8, max payload size=8
    // total = alignUp(8+8, 8) = 16
    try std.testing.expectEqual(@as(u32, 16), sizeOfType(reg, .{ .variant = 0 }));
    try std.testing.expectEqual(@as(u32, 8), alignOfType(reg, .{ .variant = 0 }));
}

test "flattenCount: record {s32, f64}" {
    const types = [_]ctypes.TypeDef{
        .{ .record = .{ .fields = &.{
            .{ .name = "x", .type = .s32 },
            .{ .name = "y", .type = .f64 },
        } } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    try std.testing.expectEqual(@as(u32, 2), flattenCount(reg, .{ .record = 0 }));
}

test "flattenCount: option<u64>" {
    const types = [_]ctypes.TypeDef{
        .{ .option = .{ .inner = .u64 } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    // disc(1) + payload(1) = 2
    try std.testing.expectEqual(@as(u32, 2), flattenCount(reg, .{ .option = 0 }));
}

// ── Compound load/store roundtrip tests ─────────────────────────────────────

test "loadValReg/storeValReg: record {u8, u32} roundtrip" {
    const types = [_]ctypes.TypeDef{
        .{ .record = .{ .fields = &.{
            .{ .name = "a", .type = .u8 },
            .{ .name = "b", .type = .u32 },
        } } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    var mem = [_]u8{0} ** 32;

    const val = InterfaceValue{ .record_val = &.{
        .{ .u8 = 42 },
        .{ .u32 = 0xDEADBEEF },
    } };
    try storeValReg(&mem, 0, .{ .record = 0 }, val, reg);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const loaded = try loadValReg(&mem, 0, .{ .record = 0 }, reg, arena.allocator());

    try std.testing.expectEqual(@as(u8, 42), loaded.record_val[0].u8);
    try std.testing.expectEqual(@as(u32, 0xDEADBEEF), loaded.record_val[1].u32);
}

test "loadValReg/storeValReg: enum roundtrip" {
    const types = [_]ctypes.TypeDef{
        .{ .enum_ = .{ .names = &.{ "red", "green", "blue" } } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    var mem = [_]u8{0} ** 4;

    try storeValReg(&mem, 0, .{ .enum_ = 0 }, .{ .enum_val = 2 }, reg);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const loaded = try loadValReg(&mem, 0, .{ .enum_ = 0 }, reg, arena.allocator());
    try std.testing.expectEqual(@as(u32, 2), loaded.enum_val);
}

test "loadValReg/storeValReg: option<u32> some roundtrip" {
    const types = [_]ctypes.TypeDef{
        .{ .option = .{ .inner = .u32 } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    var mem = [_]u8{0} ** 16;

    var payload_val = InterfaceValue{ .u32 = 999 };
    const val = InterfaceValue{ .option_val = .{ .is_some = true, .payload = &payload_val } };
    try storeValReg(&mem, 0, .{ .option = 0 }, val, reg);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const loaded = try loadValReg(&mem, 0, .{ .option = 0 }, reg, arena.allocator());
    try std.testing.expect(loaded.option_val.is_some);
    try std.testing.expectEqual(@as(u32, 999), loaded.option_val.payload.?.u32);
}

test "loadValReg/storeValReg: option<u32> none roundtrip" {
    const types = [_]ctypes.TypeDef{
        .{ .option = .{ .inner = .u32 } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    var mem = [_]u8{0} ** 16;

    const val = InterfaceValue{ .option_val = .{ .is_some = false, .payload = null } };
    try storeValReg(&mem, 0, .{ .option = 0 }, val, reg);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const loaded = try loadValReg(&mem, 0, .{ .option = 0 }, reg, arena.allocator());
    try std.testing.expect(!loaded.option_val.is_some);
    try std.testing.expect(loaded.option_val.payload == null);
}

test "loadValReg/storeValReg: result<u32, u8> ok roundtrip" {
    const types = [_]ctypes.TypeDef{
        .{ .result = .{ .ok = .u32, .err = .u8 } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    var mem = [_]u8{0} ** 16;

    var payload_val = InterfaceValue{ .u32 = 42 };
    const val = InterfaceValue{ .result_val = .{ .is_ok = true, .payload = &payload_val } };
    try storeValReg(&mem, 0, .{ .result = 0 }, val, reg);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const loaded = try loadValReg(&mem, 0, .{ .result = 0 }, reg, arena.allocator());
    try std.testing.expect(loaded.result_val.is_ok);
    try std.testing.expectEqual(@as(u32, 42), loaded.result_val.payload.?.u32);
}

test "loadValReg/storeValReg: flags roundtrip" {
    const types = [_]ctypes.TypeDef{
        .{ .flags = .{ .names = &.{ "read", "write", "exec" } } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    var mem = [_]u8{0} ** 4;

    const words: []const u32 = &.{0b101}; // read + exec
    try storeValReg(&mem, 0, .{ .flags = 0 }, .{ .flags_val = words }, reg);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const loaded = try loadValReg(&mem, 0, .{ .flags = 0 }, reg, arena.allocator());
    try std.testing.expectEqual(@as(u32, 0b101), loaded.flags_val[0]);
}

test "loadValReg/storeValReg: variant roundtrip" {
    const types = [_]ctypes.TypeDef{
        .{ .variant = .{ .cases = &.{
            .{ .name = "none", .type = null },
            .{ .name = "some_u32", .type = .u32 },
        } } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    var mem = [_]u8{0} ** 16;

    // Store case 1 (some_u32) with payload 0x1234
    var payload_val = InterfaceValue{ .u32 = 0x1234 };
    const val = InterfaceValue{ .variant_val = .{ .discriminant = 1, .payload = &payload_val } };
    try storeValReg(&mem, 0, .{ .variant = 0 }, val, reg);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const loaded = try loadValReg(&mem, 0, .{ .variant = 0 }, reg, arena.allocator());
    try std.testing.expectEqual(@as(u32, 1), loaded.variant_val.discriminant);
    try std.testing.expectEqual(@as(u32, 0x1234), loaded.variant_val.payload.?.u32);
}

test "loadValReg: invalid discriminant returns error" {
    const types = [_]ctypes.TypeDef{
        .{ .enum_ = .{ .names = &.{ "a", "b" } } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    // Write discriminant 5 (out of range for 2 cases)
    var mem = [_]u8{0} ** 4;
    mem[0] = 5;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.InvalidDiscriminant, loadValReg(&mem, 0, .{ .enum_ = 0 }, reg, arena.allocator()));
}

test "loadValReg: nested record roundtrip" {
    // type 0: record {x: u8, y: u8}
    // type 1: record {inner: type_idx(0), z: u32}
    const types = [_]ctypes.TypeDef{
        .{ .record = .{ .fields = &.{
            .{ .name = "x", .type = .u8 },
            .{ .name = "y", .type = .u8 },
        } } },
        .{ .record = .{ .fields = &.{
            .{ .name = "inner", .type = .{ .record = 0 } },
            .{ .name = "z", .type = .u32 },
        } } },
    };
    const reg = TypeRegistry.fromTypes(&types);

    var mem = [_]u8{0} ** 32;
    // Store nested: inner = {x:10, y:20}, z = 0xABCD
    const inner_val = InterfaceValue{ .record_val = &.{
        .{ .u8 = 10 },
        .{ .u8 = 20 },
    } };
    const outer_val = InterfaceValue{ .record_val = &.{
        inner_val,
        .{ .u32 = 0xABCD },
    } };
    try storeValReg(&mem, 0, .{ .record = 1 }, outer_val, reg);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const loaded = try loadValReg(&mem, 0, .{ .record = 1 }, reg, arena.allocator());

    try std.testing.expectEqual(@as(u8, 10), loaded.record_val[0].record_val[0].u8);
    try std.testing.expectEqual(@as(u8, 20), loaded.record_val[0].record_val[1].u8);
    try std.testing.expectEqual(@as(u32, 0xABCD), loaded.record_val[1].u32);
}

test "loadValReg/storeValReg: tuple roundtrip" {
    const types = [_]ctypes.TypeDef{
        .{ .tuple = .{ .fields = &.{ .u8, .u32, .u8 } } },
    };
    const reg = TypeRegistry.fromTypes(&types);
    var mem = [_]u8{0} ** 16;

    const val = InterfaceValue{ .tuple_val = &.{
        .{ .u8 = 1 },
        .{ .u32 = 0xFF00 },
        .{ .u8 = 2 },
    } };
    try storeValReg(&mem, 0, .{ .tuple = 0 }, val, reg);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const loaded = try loadValReg(&mem, 0, .{ .tuple = 0 }, reg, arena.allocator());
    try std.testing.expectEqual(@as(u8, 1), loaded.tuple_val[0].u8);
    try std.testing.expectEqual(@as(u32, 0xFF00), loaded.tuple_val[1].u32);
    try std.testing.expectEqual(@as(u8, 2), loaded.tuple_val[2].u8);
}

// ── String encoding tests ──────────────────────────────────────────────────

test "utf16ToUtf8: ASCII" {
    const allocator = std.testing.allocator;
    // "Hi" in UTF-16LE: 0x48 0x00 0x69 0x00
    var mem = [_]u8{ 0x48, 0x00, 0x69, 0x00 };
    const result = try utf16ToUtf8(&mem, 0, 2, allocator);
    defer allocator.free(result);
    try std.testing.expectEqualSlices(u8, "Hi", result);
}

test "utf16ToUtf8: non-ASCII BMP" {
    const allocator = std.testing.allocator;
    // "é" = U+00E9, UTF-16LE: 0xE9 0x00
    var mem = [_]u8{ 0xE9, 0x00 };
    const result = try utf16ToUtf8(&mem, 0, 1, allocator);
    defer allocator.free(result);
    try std.testing.expectEqualSlices(u8, "\xC3\xA9", result);
}

test "utf16ToUtf8: surrogate pair" {
    const allocator = std.testing.allocator;
    // U+1F600 (😀) = D83D DE00 in UTF-16
    var mem = [_]u8{ 0x3D, 0xD8, 0x00, 0xDE };
    const result = try utf16ToUtf8(&mem, 0, 2, allocator);
    defer allocator.free(result);
    try std.testing.expectEqualSlices(u8, "\xF0\x9F\x98\x80", result);
}

test "decodeLatin1Utf16: latin1 mode" {
    const allocator = std.testing.allocator;
    // Latin-1: bytes 0x48 0xE9 → "Hé"
    var mem = [_]u8{ 0x48, 0xE9 };
    const result = try decodeLatin1Utf16(&mem, 0, 2, allocator);
    defer allocator.free(result);
    try std.testing.expectEqualSlices(u8, "H\xC3\xA9", result);
}

test "decodeLatin1Utf16: utf16 mode (tagged)" {
    const allocator = std.testing.allocator;
    // UTF-16 mode: tag bit set, 2 code units for "Hi"
    var mem = [_]u8{ 0x48, 0x00, 0x69, 0x00 };
    const tagged_len = LATIN1_UTF16_TAG | 2;
    const result = try decodeLatin1Utf16(&mem, 0, tagged_len, allocator);
    defer allocator.free(result);
    try std.testing.expectEqualSlices(u8, "Hi", result);
}

test "encodeUtf16ToMem: ASCII" {
    var mem = [_]u8{0} ** 16;
    const written = try encodeUtf16ToMem(&mem, 0, "Hi");
    try std.testing.expectEqual(@as(u32, 2), written);
    try std.testing.expectEqual(@as(u16, 0x48), std.mem.readInt(u16, mem[0..2], .little));
    try std.testing.expectEqual(@as(u16, 0x69), std.mem.readInt(u16, mem[2..4], .little));
}

test "encodeUtf16ToMem: surrogate pair" {
    var mem = [_]u8{0} ** 16;
    const written = try encodeUtf16ToMem(&mem, 0, "\xF0\x9F\x98\x80");
    try std.testing.expectEqual(@as(u32, 2), written); // surrogate pair = 2 code units
    try std.testing.expectEqual(@as(u16, 0xD83D), std.mem.readInt(u16, mem[0..2], .little));
    try std.testing.expectEqual(@as(u16, 0xDE00), std.mem.readInt(u16, mem[2..4], .little));
}
