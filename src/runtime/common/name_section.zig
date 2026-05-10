//! Wasm `name` custom-section parser (function-names subsection only).
//!
//! Spec: https://webassembly.github.io/spec/core/appendix/custom.html#binary-namesection
//!
//! Used by the CoreMark profiling runner (`zig build coremark-profile`) to
//! turn raw wasm function indices into symbolic names. Intentionally
//! decoupled from the interpreter's `loader.zig`: the default load /
//! instantiate path skips custom sections (per §5.5.1) and we don't want to
//! pay for name parsing on the hot path. This file is opt-in.
//!
//! Only subsection id `1` (function names) is decoded. Other subsections
//! (module name, locals, labels, types, tables, memories, globals, …) are
//! skipped — the profiler doesn't need them.

const std = @import("std");
const types = @import("types.zig");
const leb128_mod = @import("../../shared/utils/leb128.zig");

pub const FunctionName = types.NameSection.FunctionName;

pub const Error = error{
    InvalidMagic,
    InvalidVersion,
    UnexpectedEnd,
    Overflow,
    InvalidUtf8,
    OutOfMemory,
};

const Reader = struct {
    data: []const u8,
    pos: usize = 0,

    fn remaining(self: *const Reader) usize {
        return self.data.len - self.pos;
    }

    fn readByte(self: *Reader) Error!u8 {
        if (self.pos >= self.data.len) return error.UnexpectedEnd;
        const b = self.data[self.pos];
        self.pos += 1;
        return b;
    }

    fn readBytes(self: *Reader, n: usize) Error![]const u8 {
        if (self.pos + n > self.data.len) return error.UnexpectedEnd;
        const slice = self.data[self.pos .. self.pos + n];
        self.pos += n;
        return slice;
    }

    fn readU32(self: *Reader) Error!u32 {
        const slice = self.data[self.pos..];
        const r = leb128_mod.readUnsigned(u32, slice) catch |err| switch (err) {
            error.Overflow => return error.Overflow,
            error.UnexpectedEnd => return error.UnexpectedEnd,
        };
        self.pos += r.bytes_read;
        return r.value;
    }

    fn readU32FixedLe(self: *Reader) Error!u32 {
        const bytes = try self.readBytes(4);
        return std.mem.readInt(u32, bytes[0..4], .little);
    }

    fn readName(self: *Reader) Error![]const u8 {
        const len = try self.readU32();
        const bytes = try self.readBytes(len);
        if (!std.unicode.utf8ValidateSlice(bytes)) return error.InvalidUtf8;
        return bytes;
    }
};

/// Parse the `name` custom section's function-name subsection out of a
/// complete `.wasm` binary.
///
/// Returns an owned slice of `FunctionName` entries, sorted by ascending
/// `index` (per the spec — entries within the namemap are required to be
/// sorted by index without duplicates; we don't validate strictly but we
/// preserve the file ordering).
///
/// If the wasm has no `name` custom section, or has one without subsection
/// id `1`, returns an empty slice (caller still owns it).
///
/// On any structural error (bad magic, truncated section, invalid UTF-8 in a
/// name, etc.) the function returns an error rather than partial data — the
/// caller (a diagnostic harness) should fall back to numeric names.
pub fn parseFunctionNames(
    wasm_bytes: []const u8,
    allocator: std.mem.Allocator,
) Error![]FunctionName {
    var r = Reader{ .data = wasm_bytes };

    // Magic + version (\0asm + 0x01 0x00 0x00 0x00)
    const magic = try r.readU32FixedLe();
    if (magic != 0x6d736100) return error.InvalidMagic;
    const version = try r.readU32FixedLe();
    if (version != 1) return error.InvalidVersion;

    while (r.remaining() > 0) {
        const section_id = try r.readByte();
        const section_size = try r.readU32();
        const section_start = r.pos;
        if (section_size > r.remaining()) return error.UnexpectedEnd;

        if (section_id != 0) {
            // Not a custom section — skip.
            r.pos = section_start + section_size;
            continue;
        }

        const name = try r.readName();
        if (!std.mem.eql(u8, name, "name")) {
            r.pos = section_start + section_size;
            continue;
        }

        // Inside the `name` custom section: a sequence of subsections.
        const section_end = section_start + section_size;
        while (r.pos < section_end) {
            const sub_id = try r.readByte();
            const sub_size = try r.readU32();
            const sub_end = r.pos + sub_size;
            if (sub_end > section_end) return error.UnexpectedEnd;

            if (sub_id == 1) {
                return parseFunctionNameMap(&r, sub_end, allocator);
            }
            r.pos = sub_end;
        }

        // `name` section had no function-names subsection.
        return allocator.alloc(FunctionName, 0);
    }

    // No `name` custom section at all.
    return allocator.alloc(FunctionName, 0);
}

fn parseFunctionNameMap(
    r: *Reader,
    sub_end: usize,
    allocator: std.mem.Allocator,
) Error![]FunctionName {
    const count = try r.readU32();
    if (count == 0) {
        if (r.pos != sub_end) return error.UnexpectedEnd;
        return allocator.alloc(FunctionName, 0);
    }

    const entries = allocator.alloc(FunctionName, count) catch
        return error.OutOfMemory;
    errdefer allocator.free(entries);

    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const idx = try r.readU32();
        const nm = try r.readName();
        entries[i] = .{ .index = idx, .name = nm };
    }
    if (r.pos != sub_end) return error.UnexpectedEnd;
    return entries;
}

/// Linear scan over a previously-parsed slice. Profiler tables are O(top-N)
/// so a binary search would be overkill.
pub fn lookup(entries: []const FunctionName, func_idx: u32) ?[]const u8 {
    for (entries) |e| {
        if (e.index == func_idx) return e.name;
    }
    return null;
}

// ─── Tests ──────────────────────────────────────────────────────────────────

const testing = std.testing;

fn appendLeb(buf: *std.ArrayList(u8), v: u32, gpa: std.mem.Allocator) !void {
    var x = v;
    while (true) {
        var b: u8 = @intCast(x & 0x7f);
        x >>= 7;
        if (x != 0) b |= 0x80;
        try buf.append(gpa, b);
        if (x == 0) break;
    }
}

fn appendName(buf: *std.ArrayList(u8), s: []const u8, gpa: std.mem.Allocator) !void {
    try appendLeb(buf, @intCast(s.len), gpa);
    try buf.appendSlice(gpa, s);
}

fn buildMinimalWasmWithNameSection(
    gpa: std.mem.Allocator,
    entries: []const FunctionName,
) ![]u8 {
    var bin: std.ArrayList(u8) = .empty;
    defer bin.deinit(gpa);

    // Magic + version
    try bin.appendSlice(gpa, &[_]u8{ 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00 });

    // Build the function-names subsection payload.
    var fn_sub: std.ArrayList(u8) = .empty;
    defer fn_sub.deinit(gpa);
    try appendLeb(&fn_sub, @intCast(entries.len), gpa);
    for (entries) |e| {
        try appendLeb(&fn_sub, e.index, gpa);
        try appendName(&fn_sub, e.name, gpa);
    }

    // Build the `name` custom section body: name + subsection (id 1).
    var name_body: std.ArrayList(u8) = .empty;
    defer name_body.deinit(gpa);
    try appendName(&name_body, "name", gpa);
    try name_body.append(gpa, 0x01); // subsection id 1
    try appendLeb(&name_body, @intCast(fn_sub.items.len), gpa);
    try name_body.appendSlice(gpa, fn_sub.items);

    // Custom section header: id=0, size, body.
    try bin.append(gpa, 0x00);
    try appendLeb(&bin, @intCast(name_body.items.len), gpa);
    try bin.appendSlice(gpa, name_body.items);

    return try gpa.dupe(u8, bin.items);
}

test "parseFunctionNames: empty when section absent" {
    const gpa = testing.allocator;
    const wasm = [_]u8{ 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00 };
    const names = try parseFunctionNames(&wasm, gpa);
    defer gpa.free(names);
    try testing.expectEqual(@as(usize, 0), names.len);
}

test "parseFunctionNames: round-trip" {
    const gpa = testing.allocator;
    const want = [_]FunctionName{
        .{ .index = 0, .name = "core_main" },
        .{ .index = 7, .name = "matrix_mul_matrix" },
        .{ .index = 23, .name = "crcu16" },
    };
    const wasm = try buildMinimalWasmWithNameSection(gpa, &want);
    defer gpa.free(wasm);

    const got = try parseFunctionNames(wasm, gpa);
    defer gpa.free(got);

    try testing.expectEqual(@as(usize, 3), got.len);
    for (want, got) |w, g| {
        try testing.expectEqual(w.index, g.index);
        try testing.expectEqualStrings(w.name, g.name);
    }

    try testing.expectEqualStrings("matrix_mul_matrix", lookup(got, 7).?);
    try testing.expectEqual(@as(?[]const u8, null), lookup(got, 99));
}

test "parseFunctionNames: skips other custom sections before name" {
    const gpa = testing.allocator;
    var bin: std.ArrayList(u8) = .empty;
    defer bin.deinit(gpa);

    try bin.appendSlice(gpa, &[_]u8{ 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00 });

    // First custom section: name "producers", body "abc".
    var other: std.ArrayList(u8) = .empty;
    defer other.deinit(gpa);
    try appendName(&other, "producers", gpa);
    try other.appendSlice(gpa, "abc");
    try bin.append(gpa, 0x00);
    try appendLeb(&bin, @intCast(other.items.len), gpa);
    try bin.appendSlice(gpa, other.items);

    // Then the real name section.
    const want = [_]FunctionName{.{ .index = 5, .name = "hot" }};
    const wasm_named = try buildMinimalWasmWithNameSection(gpa, &want);
    defer gpa.free(wasm_named);
    // Drop the magic/version prefix from the second blob and append.
    try bin.appendSlice(gpa, wasm_named[8..]);

    const got = try parseFunctionNames(bin.items, gpa);
    defer gpa.free(got);
    try testing.expectEqual(@as(usize, 1), got.len);
    try testing.expectEqualStrings("hot", got[0].name);
}

test "parseFunctionNames: rejects bad magic" {
    const gpa = testing.allocator;
    const wasm = [_]u8{ 0xde, 0xad, 0xbe, 0xef, 0x01, 0x00, 0x00, 0x00 };
    try testing.expectError(error.InvalidMagic, parseFunctionNames(&wasm, gpa));
}
