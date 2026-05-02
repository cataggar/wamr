//! fuzz-canon — Canonical ABI lift/lower and string codecs over
//! attacker-controlled guest memory.
//!
//! The host must tolerate any (ptr, len, type) triple from a malicious
//! guest. This harness drives `loadVal`, the UTF-8 validator, and the
//! UTF-8/UTF-16/Latin-1 codecs against arbitrary memory bytes and
//! arbitrary ptr/len values. Typed errors and silent zero-fill on
//! out-of-bounds reads are expected; panics, safety-checked UB,
//! over-allocation crashes, or process aborts are bugs.

const std = @import("std");
const wamr = @import("wamr");
const ctypes = wamr.component_types;
const canon = wamr.canonical_abi;
const common = @import("common.zig");

/// Hard cap on memory size after we mirror the input bytes. Keeps each
/// iteration bounded and avoids allocator pressure on tiny seeds.
const max_memory_bytes: usize = 64 * 1024;
/// Hard cap on the destination buffer for write-back codecs.
const max_out_bytes: usize = 16 * 1024;

const PrimitiveKind = enum(u8) {
    bool_,
    s8,
    u8_,
    s16,
    u16_,
    s32,
    u32_,
    s64,
    u64_,
    f32_,
    f64_,
    char,
    own_,
    borrow_,
    string,
    list,
};

fn primitiveValType(kind: PrimitiveKind) ctypes.ValType {
    return switch (kind) {
        .bool_ => .bool,
        .s8 => .s8,
        .u8_ => .u8,
        .s16 => .s16,
        .u16_ => .u16,
        .s32 => .s32,
        .u32_ => .u32,
        .s64 => .s64,
        .u64_ => .u64,
        .f32_ => .f32,
        .f64_ => .f64,
        .char => .char,
        .own_ => .{ .own = 0 },
        .borrow_ => .{ .borrow = 0 },
        .string => .string,
        .list => .{ .list = 0 },
    };
}

const Mode = enum(u8) {
    load_primitive,
    validate_utf8,
    utf8_to_utf16,
    utf16_to_utf8,
    decode_latin1_utf16,
    encode_utf16_to_mem,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());

    const args = try common.Args.parse(argv);
    var corpus = try common.Corpus.load(allocator, io, args.corpus_dir);
    defer corpus.deinit();

    if (corpus.count() == 0) {
        std.log.err("empty corpus at {s}", .{args.corpus_dir});
        return error.EmptyCorpus;
    }

    const deadline = common.Deadline.init(io, args.duration_ms);
    var iter: u64 = 0;
    var idx: usize = 0;
    while (!deadline.expired(io)) : (iter += 1) {
        const input = corpus.get(idx);
        idx +%= 1;

        try common.markInFlight(io, args.crashes_dir, input);
        try runOnce(allocator, input);
        common.clearInFlight(io, args.crashes_dir);
    }

    std.log.info("fuzz-canon: {d} iterations over {d} inputs", .{ iter, corpus.count() });
}

const Header = struct {
    mode: Mode,
    ptr: u32,
    len: u32,
    primitive: PrimitiveKind,
    body: []const u8,
};

fn parseHeader(input: []const u8) ?Header {
    if (input.len < 11) return null;
    const mode_byte = input[0];
    const prim_byte = input[1];
    const ptr = std.mem.readInt(u32, input[2..6], .little);
    const len = std.mem.readInt(u32, input[6..10], .little);
    const mode = std.enums.fromInt(Mode, mode_byte) orelse return null;
    const prim = std.enums.fromInt(PrimitiveKind, prim_byte % @as(u8, @intCast(@typeInfo(PrimitiveKind).@"enum".fields.len))) orelse return null;
    return .{ .mode = mode, .ptr = ptr, .len = len, .primitive = prim, .body = input[10..] };
}

fn runOnce(allocator: std.mem.Allocator, input: []const u8) !void {
    const header = parseHeader(input) orelse return;

    const mem_len = @min(header.body.len, max_memory_bytes);
    const memory = try allocator.alloc(u8, mem_len);
    defer allocator.free(memory);
    @memcpy(memory, header.body[0..mem_len]);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    switch (header.mode) {
        .load_primitive => {
            // `loadVal` accepts only primitive ValTypes (no registry).
            const t = primitiveValType(header.primitive);
            _ = canon.loadVal(memory, header.ptr, t) catch {};
        },
        .validate_utf8 => {
            _ = canon.validateUtf8(memory, header.ptr, header.len);
        },
        .utf8_to_utf16 => {
            const src_len = @min(header.len, @as(u32, @intCast(memory.len)));
            const src_start = @min(header.ptr, @as(u32, @intCast(memory.len)));
            const src = memory[src_start .. src_start + @min(src_len, @as(u32, @intCast(memory.len - src_start)))];
            // Allow worst-case 1:1 expansion plus surrogate pairs.
            const out_units: usize = @min(src.len * 2 + 1, max_out_bytes / 2);
            const out = try arena.allocator().alloc(u16, out_units);
            _ = canon.utf8ToUtf16(src, out) catch {};
        },
        .utf16_to_utf8 => {
            // utf16ToUtf8 reads 2*code_units bytes from memory[ptr..]. Cap.
            const code_units = @min(header.len, @as(u32, @intCast(max_out_bytes / 2)));
            const out = canon.utf16ToUtf8(memory, header.ptr, code_units, arena.allocator()) catch return;
            _ = out;
        },
        .decode_latin1_utf16 => {
            const tagged_len = @min(header.len, @as(u32, @intCast(max_out_bytes / 2))) | (header.len & canon.LATIN1_UTF16_TAG);
            const out = canon.decodeLatin1Utf16(memory, header.ptr, tagged_len, arena.allocator()) catch return;
            _ = out;
        },
        .encode_utf16_to_mem => {
            const dst_len = @min(memory.len, max_out_bytes);
            const dst = try arena.allocator().alloc(u8, dst_len);
            const src_start = @min(header.ptr, @as(u32, @intCast(header.body.len)));
            const src_end = @min(header.body.len, src_start + header.len);
            const src = header.body[src_start..src_end];
            _ = canon.encodeUtf16ToMem(dst, 0, src) catch {};
        },
    }
}
