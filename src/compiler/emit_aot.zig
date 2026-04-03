//! AOT Binary File Emitter
//!
//! Produces WAMR AOT binary files from compiled native code and metadata.
//! The output format matches what `runtime/aot/loader.zig` expects:
//!
//!   [4 bytes] magic (0x746f6100)
//!   [4 bytes] version (6)
//!   [sections...] each: [4 bytes type] [4 bytes size] [payload]

const std = @import("std");

/// AOT binary magic number ("\0aot" little-endian).
pub const aot_magic: u32 = 0x746f6100;

/// AOT format version.
pub const aot_version: u32 = 6;

/// Export kinds (matches WebAssembly spec §2.5 and runtime ExternalKind).
pub const ExternalKind = enum(u8) {
    function = 0x00,
    table = 0x01,
    memory = 0x02,
    global = 0x03,
};

pub const AotEmitOptions = struct {
    arch: [16]u8 = std.mem.zeroes([16]u8),
    abi_type: u16 = 0,
    e_type: u16 = 0,
    e_machine: u16 = 0,
    e_flags: u32 = 0,
    target_features: u64 = 0,
};

pub const ExportEntry = struct {
    name: []const u8,
    kind: ExternalKind,
    index: u32,
};

/// Emit an AOT binary to an owned byte buffer.
pub fn emit(
    allocator: std.mem.Allocator,
    native_code: []const u8,
    func_offsets: []const u32,
    exports: []const ExportEntry,
    options: AotEmitOptions,
) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    // Header: magic + version
    try buf.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u32, aot_magic)));
    try buf.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u32, aot_version)));

    // Section 0: target_info (40 bytes)
    try emitSection(allocator, &buf, 0, &buildTargetInfo(options));

    // Section 2: text (native code)
    try emitSection(allocator, &buf, 2, native_code);

    // Section 3: function offsets
    {
        var tmp: std.ArrayList(u8) = .empty;
        defer tmp.deinit(allocator);
        try appendU32Le(&tmp, allocator, @intCast(func_offsets.len));
        for (func_offsets) |offset| {
            try appendU32Le(&tmp, allocator, offset);
        }
        try emitSection(allocator, &buf, 3, tmp.items);
    }

    // Section 4: exports  (must match loader: name_len, name, kind, index)
    {
        var tmp: std.ArrayList(u8) = .empty;
        defer tmp.deinit(allocator);
        try appendU32Le(&tmp, allocator, @intCast(exports.len));
        for (exports) |exp| {
            try appendU32Le(&tmp, allocator, @intCast(exp.name.len));
            try tmp.appendSlice(allocator, exp.name);
            try tmp.append(allocator, @intFromEnum(exp.kind));
            try appendU32Le(&tmp, allocator, exp.index);
        }
        try emitSection(allocator, &buf, 4, tmp.items);
    }

    return buf.toOwnedSlice(allocator);
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn emitSection(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), section_type: u32, payload: []const u8) !void {
    try appendU32Le(buf, allocator, section_type);
    try appendU32Le(buf, allocator, @intCast(payload.len));
    try buf.appendSlice(allocator, payload);
}

fn appendU32Le(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, val: u32) !void {
    try buf.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u32, val)));
}

/// Build the 40-byte TargetInfo payload matching the loader struct layout:
///   bin_type(u16) abi_type(u16) e_type(u16) e_machine(u16)
///   e_flags(u32) reserved(u32) arch([16]u8) features(u64)
fn buildTargetInfo(options: AotEmitOptions) [40]u8 {
    var info = std.mem.zeroes([40]u8);
    std.mem.writeInt(u16, info[0..2], 1, .little); // bin_type = AOT
    std.mem.writeInt(u16, info[2..4], options.abi_type, .little);
    std.mem.writeInt(u16, info[4..6], options.e_type, .little);
    std.mem.writeInt(u16, info[6..8], options.e_machine, .little);
    std.mem.writeInt(u32, info[8..12], options.e_flags, .little);
    std.mem.writeInt(u32, info[12..16], 0, .little); // reserved
    @memcpy(info[16..32], &options.arch);
    std.mem.writeInt(u64, info[32..40], options.target_features, .little);
    return info;
}

// ─── Tests ──────────────────────────────────────────────────────────────────

test "emit: minimal (no functions, no exports) has correct magic and version" {
    const allocator = std.testing.allocator;
    const data = try emit(allocator, &.{}, &.{}, &.{}, .{});
    defer allocator.free(data);

    // At least header (8) + target_info section (8+40) + text section (8+0) + func section (8+4) + export section (8+4)
    try std.testing.expect(data.len >= 8);
    const magic = std.mem.readInt(u32, data[0..4], .little);
    const version = std.mem.readInt(u32, data[4..8], .little);
    try std.testing.expectEqual(aot_magic, magic);
    try std.testing.expectEqual(aot_version, version);
}

test "emit: one function offset produces valid function section" {
    const allocator = std.testing.allocator;
    const code = [_]u8{ 0xCC, 0xC3 }; // int3; ret
    const offsets = [_]u32{0};
    const data = try emit(allocator, &code, &offsets, &.{}, .{});
    defer allocator.free(data);

    // Walk sections to find section type 3 (function)
    var pos: usize = 8; // skip header
    var found_func_section = false;
    while (pos + 8 <= data.len) {
        const sec_type = std.mem.readInt(u32, data[pos..][0..4], .little);
        const sec_size = std.mem.readInt(u32, data[pos + 4 ..][0..4], .little);
        pos += 8;
        if (sec_type == 3) {
            // function section: count(4) + offsets
            try std.testing.expect(sec_size >= 4);
            const count = std.mem.readInt(u32, data[pos..][0..4], .little);
            try std.testing.expectEqual(@as(u32, 1), count);
            const off = std.mem.readInt(u32, data[pos + 4 ..][0..4], .little);
            try std.testing.expectEqual(@as(u32, 0), off);
            found_func_section = true;
        }
        pos += sec_size;
    }
    try std.testing.expect(found_func_section);
}

test "emit: export section encodes name, kind, and index" {
    const allocator = std.testing.allocator;
    const exports = [_]ExportEntry{.{
        .name = "main",
        .kind = .function,
        .index = 7,
    }};
    const data = try emit(allocator, &.{}, &.{}, &exports, .{});
    defer allocator.free(data);

    // Walk sections to find section type 4 (export)
    var pos: usize = 8;
    var found_export_section = false;
    while (pos + 8 <= data.len) {
        const sec_type = std.mem.readInt(u32, data[pos..][0..4], .little);
        const sec_size = std.mem.readInt(u32, data[pos + 4 ..][0..4], .little);
        pos += 8;
        if (sec_type == 4) {
            var p = pos;
            const count = std.mem.readInt(u32, data[p..][0..4], .little);
            p += 4;
            try std.testing.expectEqual(@as(u32, 1), count);
            const name_len = std.mem.readInt(u32, data[p..][0..4], .little);
            p += 4;
            try std.testing.expectEqual(@as(u32, 4), name_len);
            try std.testing.expect(std.mem.eql(u8, data[p..][0..4], "main"));
            p += 4;
            try std.testing.expectEqual(@as(u8, 0x00), data[p]); // function
            p += 1;
            const index = std.mem.readInt(u32, data[p..][0..4], .little);
            try std.testing.expectEqual(@as(u32, 7), index);
            found_export_section = true;
        }
        pos += sec_size;
    }
    try std.testing.expect(found_export_section);
}

test "roundtrip: emit then load with AOT loader" {
    const allocator = std.testing.allocator;
    const aot_loader = @import("../runtime/aot/loader.zig");

    const code = [_]u8{ 0xCC, 0x90, 0xC3 };
    const offsets = [_]u32{ 0, 2 };
    const exports = [_]ExportEntry{
        .{ .name = "add", .kind = .function, .index = 0 },
        .{ .name = "mem", .kind = .memory, .index = 0 },
    };
    var arch_name = std.mem.zeroes([16]u8);
    @memcpy(arch_name[0..6], "x86_64");

    const data = try emit(allocator, &code, &offsets, &exports, .{
        .arch = arch_name,
        .e_machine = 0x3E,
    });
    defer allocator.free(data);

    // Parse back with the AOT loader
    const module = try aot_loader.load(data, allocator);
    defer aot_loader.unload(&module, allocator);

    // Verify target info
    try std.testing.expect(module.target_info != null);
    const ti = module.target_info.?;
    try std.testing.expectEqual(@as(u16, 1), ti.bin_type);
    try std.testing.expectEqual(@as(u16, 0x3E), ti.e_machine);
    try std.testing.expect(std.mem.startsWith(u8, &ti.arch, "x86_64"));

    // Verify text section
    try std.testing.expect(module.text_section != null);
    try std.testing.expectEqual(@as(usize, 3), module.text_section.?.len);
    try std.testing.expectEqual(@as(u8, 0xCC), module.text_section.?[0]);

    // Verify function offsets
    try std.testing.expectEqual(@as(u32, 2), module.func_count);
    try std.testing.expectEqual(@as(u32, 0), module.func_offsets[0]);
    try std.testing.expectEqual(@as(u32, 2), module.func_offsets[1]);

    // Verify exports
    try std.testing.expectEqual(@as(usize, 2), module.exports.len);
    try std.testing.expect(std.mem.eql(u8, module.exports[0].name, "add"));
    try std.testing.expectEqual(@as(u8, 0x00), @intFromEnum(module.exports[0].kind)); // function
    try std.testing.expectEqual(@as(u32, 0), module.exports[0].index);
    try std.testing.expect(std.mem.eql(u8, module.exports[1].name, "mem"));
    try std.testing.expectEqual(@as(u8, 0x02), @intFromEnum(module.exports[1].kind)); // memory
}
