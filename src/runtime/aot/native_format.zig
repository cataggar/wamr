//! Strict reader for the deliberately small native embedding contract.
//! All allocations belong to the caller's instance arena; input bytes are
//! copied by the API before parsing. Native text is trusted executable code,
//! not a sandbox boundary for malicious .cwasm producers.
const std = @import("std");
pub const abi = @import("native_abi.zig");
pub const null_function_index = std.math.maxInt(u32);
pub const ValType = enum(u8) { i32 = 0x7f, i64 = 0x7e, f32 = 0x7d, f64 = 0x7c };
pub const Value = union(ValType) {
    i32: i32,
    i64: i64,
    f32: f32,
    f64: f64,

    pub fn raw(self: Value) u64 {
        return switch (self) {
            .i32 => |v| @as(u32, @bitCast(v)),
            .i64 => |v| @bitCast(v),
            .f32 => |v| @as(u32, @bitCast(v)),
            .f64 => |v| @bitCast(v),
        };
    }

    pub fn fromRaw(t: ValType, bits: u64) Value {
        return switch (t) {
            .i32 => .{ .i32 = @bitCast(@as(u32, @truncate(bits))) },
            .i64 => .{ .i64 = @bitCast(bits) },
            .f32 => .{ .f32 = @bitCast(@as(u32, @truncate(bits))) },
            .f64 => .{ .f64 = @bitCast(bits) },
        };
    }
};
pub const Error = error{ InvalidMagic, InvalidVersion, InvalidSection, UnexpectedEnd, UnsupportedTarget, UnsupportedFeature, InvalidIndex, InvalidLimits, OutOfMemory };
pub const Signature = struct { params: []const ValType, results: []const ValType };
pub const Function = struct { offset: u32, type_index: u32 };
pub const Import = struct { module: []const u8, name: []const u8, type_index: u32 };
pub const Export = struct { name: []const u8, kind: u8, index: u32 };
pub const Limits = struct { min: u32, max: ?u32 };
pub const Data = struct { memory: u32, kind: u8, offset: u32, bytes: []const u8 };
pub const Element = struct { table: u32, passive: bool, offset: u32, indices: []const u32 };
pub const Global = struct { value_type: ValType, mutable: bool, bits: u64 };

pub const Module = struct {
    fuel_metered: bool = false,
    text: []const u8 = &.{},
    functions: []const Function = &.{},
    signatures: []const Signature = &.{},
    imports: []const Import = &.{},
    exports: []const Export = &.{},
    memory: ?Limits = null,
    tables: []const Limits = &.{},
    globals: []const Global = &.{},
    data: []const Data = &.{},
    elements: []const Element = &.{},
    start: ?u32 = null,

    pub fn signature(self: Module, index: u32) Error!Signature {
        const ti = if (index < self.imports.len) self.imports[index].type_index else blk: {
            const local = index - self.imports.len;
            if (local >= self.functions.len) return error.InvalidIndex;
            break :blk self.functions[local].type_index;
        };
        if (ti >= self.signatures.len) return error.InvalidIndex;
        return self.signatures[ti];
    }

    pub fn validate(self: Module) Error!void {
        for (self.functions) |f| {
            if (f.offset >= self.text.len or f.type_index >= self.signatures.len) return error.InvalidIndex;
        }
        for (self.imports) |f| {
            if (f.type_index >= self.signatures.len) return error.InvalidIndex;
        }
        for (self.exports, 0..) |e, i| {
            for (self.exports[0..i]) |prev| {
                if (std.mem.eql(u8, e.name, prev.name)) return error.InvalidSection;
            }
            const count = switch (e.kind) {
                0 => self.imports.len + self.functions.len,
                1 => self.tables.len,
                2 => @as(usize, if (self.memory != null) 1 else 0),
                3 => self.globals.len,
                else => return error.UnsupportedFeature,
            };
            if (e.index >= count) return error.InvalidIndex;
        }
        for (self.data) |d| {
            if (d.memory != 0 or self.memory == null) return error.InvalidIndex;
            if (d.kind == 1) return error.UnsupportedFeature; // imported-global offsets
        }
        for (self.elements) |e| {
            if (e.table >= self.tables.len) return error.InvalidIndex;
            for (e.indices) |index| {
                if (index != null_function_index and index >= self.imports.len + self.functions.len) return error.InvalidIndex;
            }
        }
        if (self.start) |index| {
            const sig = try self.signature(index);
            if (sig.params.len != 0 or sig.results.len != 0) return error.InvalidSection;
        }
    }
};

fn readTarget(section: *Reader, cpu_features: u64) Error!bool {
    if (section.bytes.len != 40) return error.InvalidSection;
    if (try section.int(u16) != 2 or try section.int(u16) != 0 or
        try section.int(u16) != 1 or try section.int(u16) != 0x3e)
        return error.UnsupportedTarget;
    const profile = try section.int(u32);
    if ((profile != abi.profile_flag and profile != abi.fuel_profile_flag) or
        try section.int(u32) != abi.contract_version) return error.UnsupportedTarget;
    if (!std.mem.eql(u8, try section.take(16), "x86_64" ++ "\x00" ** 10)) return error.UnsupportedTarget;
    const features = try section.int(u64);
    if (features != abi.cpu_features or features & ~cpu_features != 0) return error.UnsupportedTarget;
    return profile == abi.fuel_profile_flag;
}

/// Allocation-free target admission so missing metered budgets cannot allow an
/// unbounded input copy or metadata parse. Full load still validates every section.
pub fn fuelMetered(bytes: []const u8, cpu_features: u64) Error!bool {
    var reader: Reader = .{ .bytes = bytes };
    if (try reader.int(u32) != abi.magic) return error.InvalidMagic;
    if (try reader.int(u32) != abi.format_version) return error.InvalidVersion;
    var seen: u32 = 0;
    var metered: ?bool = null;
    while (reader.bytes.len != 0) {
        const id = try reader.int(u32);
        const length = try reader.int(u32);
        if (id > 15 or seen & (@as(u32, 1) << @intCast(id)) != 0) return error.InvalidSection;
        seen |= @as(u32, 1) << @intCast(id);
        var section: Reader = .{ .bytes = try reader.take(length) };
        if (id == 0) metered = try readTarget(&section, cpu_features);
    }
    return metered orelse error.InvalidSection;
}

pub fn load(bytes: []const u8, allocator: std.mem.Allocator, cpu_features: u64) Error!Module {
    var r = Reader{ .bytes = bytes };
    if (try r.int(u32) != abi.magic) return error.InvalidMagic;
    if (try r.int(u32) != abi.format_version) return error.InvalidVersion;
    var module: Module = .{};
    var seen: u32 = 0;
    while (r.bytes.len != 0) {
        const id = try r.int(u32);
        const length = try r.int(u32);
        if (id > 15 or seen & (@as(u32, 1) << @intCast(id)) != 0) return error.InvalidSection;
        seen |= @as(u32, 1) << @intCast(id);
        var section = Reader{ .bytes = try r.take(length) };
        switch (id) {
            0 => module.fuel_metered = try readTarget(&section, cpu_features),
            2 => module.text = try section.take(length),
            3 => {
                const entries = try section.allocate(Function, allocator, 8);
                for (entries) |*entry| entry.* = .{ .offset = try section.int(u32), .type_index = try section.int(u32) };
                module.functions = entries;
            },
            4 => {
                const entries = try section.allocate(Export, allocator, 9);
                for (entries) |*entry| entry.* = .{ .name = try section.string(), .kind = try section.int(u8), .index = try section.int(u32) };
                module.exports = entries;
            },
            5 => {
                const entries = try section.allocate(Data, allocator, 13);
                for (entries) |*entry| {
                    const kind = try section.int(u8);
                    const memory = try section.int(u32);
                    const offset = try section.int(u32);
                    const data = try section.string();
                    if (kind > 2) return error.InvalidSection;
                    entry.* = .{ .memory = memory, .kind = kind, .offset = offset, .bytes = data };
                }
                module.data = entries;
            },
            7 => {
                const count = try section.int(u32);
                for (0..count) |_| {
                    _ = try section.int(u32);
                    _ = try section.string();
                }
            },
            8 => {
                const entries = try section.allocate(Import, allocator, 13);
                for (entries) |*entry| {
                    const module_name = try section.string();
                    const name = try section.string();
                    if (try section.int(u8) != 0) return error.UnsupportedFeature;
                    entry.* = .{ .module = module_name, .name = name, .type_index = try section.int(u32) };
                }
                module.imports = entries;
            },
            9 => {
                const count = try section.int(u32);
                if (count > 1) return error.UnsupportedFeature;
                if (count == 1) {
                    module.memory = try section.limits(65536);
                    if (try section.int(u8) != 0) return error.UnsupportedFeature;
                }
            },
            10 => {
                const entries = try section.allocate(Global, allocator, 10);
                for (entries) |*entry| {
                    const t = try section.valType();
                    const mutable = try section.boolean();
                    entry.* = .{ .value_type = t, .mutable = mutable, .bits = try section.int(u64) };
                }
                module.globals = entries;
            },
            11 => {
                const entries = try section.allocate(Element, allocator, 13);
                for (entries) |*entry| {
                    const passive = try section.boolean();
                    const table = try section.int(u32);
                    const offset = try section.int(u32);
                    const indices = try section.allocate(u32, allocator, 4);
                    for (indices) |*index| index.* = try section.int(u32);
                    entry.* = .{ .table = table, .passive = passive, .offset = offset, .indices = indices };
                }
                module.elements = entries;
            },
            12 => module.start = try section.int(u32),
            13 => {
                const entries = try section.allocate(Signature, allocator, 8);
                for (entries) |*entry| {
                    const params = try section.allocate(ValType, allocator, 1);
                    for (params) |*t| t.* = try section.valType();
                    const results = try section.allocate(ValType, allocator, 1);
                    for (results) |*t| t.* = try section.valType();
                    if (params.len > 16 or results.len > 1) return error.UnsupportedFeature;
                    entry.* = .{ .params = params, .results = results };
                }
                module.signatures = entries;
            },
            15 => {
                const entries = try section.allocate(Limits, allocator, 6);
                for (entries) |*entry| {
                    if (try section.int(u8) != 0x70) return error.UnsupportedFeature;
                    entry.* = try section.limits(std.math.maxInt(u32));
                }
                module.tables = entries;
            },
            else => return error.UnsupportedFeature,
        }
        if (section.bytes.len != 0) return error.InvalidSection;
    }
    const required = (1 << 0) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 13);
    if (seen & required != required or module.text.len == 0) return error.InvalidSection;
    try module.validate();
    return module;
}

const Reader = struct {
    bytes: []const u8,
    fn take(self: *Reader, n: usize) Error![]const u8 {
        if (n > self.bytes.len) return error.UnexpectedEnd;
        const result = self.bytes[0..n];
        self.bytes = self.bytes[n..];
        return result;
    }
    fn int(self: *Reader, comptime T: type) Error!T {
        return std.mem.readInt(T, (try self.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }
    fn string(self: *Reader) Error![]const u8 {
        return self.take(try self.int(u32));
    }
    fn boolean(self: *Reader) Error!bool {
        return switch (try self.int(u8)) {
            0 => false,
            1 => true,
            else => error.InvalidSection,
        };
    }
    fn valType(self: *Reader) Error!ValType {
        return std.enums.fromInt(ValType, try self.int(u8)) orelse error.UnsupportedFeature;
    }
    fn limits(self: *Reader, ceiling: u32) Error!Limits {
        const min = try self.int(u32);
        const max = if (try self.boolean()) try self.int(u32) else null;
        if (min > ceiling or (max != null and (max.? < min or max.? > ceiling))) return error.InvalidLimits;
        return .{ .min = min, .max = max };
    }
    fn allocate(self: *Reader, comptime T: type, allocator: std.mem.Allocator, min_size: usize) Error![]T {
        const count = try self.int(u32);
        if (count > self.bytes.len / min_size) return error.UnexpectedEnd;
        return allocator.alloc(T, count);
    }
};
