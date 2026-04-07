//! WebAssembly opcode definitions.
//!
//! Complete set of opcodes from the Wasm MVP spec plus all ratified proposals:
//! SIMD, atomics, reference types, bulk memory, tail call, exception handling.

const std = @import("std");
const testing = std.testing;

/// Single-byte Wasm opcodes.
pub const Opcode = enum(u8) {
    // ──── Control flow ────
    @"unreachable" = 0x00,
    nop = 0x01,
    block = 0x02,
    loop = 0x03,
    @"if" = 0x04,
    @"else" = 0x05,
    @"try" = 0x06,
    @"catch" = 0x07,
    throw = 0x08,
    rethrow = 0x09,
    // 0x0A unused
    end = 0x0B,
    br = 0x0C,
    br_if = 0x0D,
    br_table = 0x0E,
    @"return" = 0x0F,
    call = 0x10,
    call_indirect = 0x11,
    return_call = 0x12,
    return_call_indirect = 0x13,
    call_ref = 0x14,
    return_call_ref = 0x15,
    // 0x16–0x17 unused
    delegate = 0x18,
    catch_all = 0x19,

    // ──── Parametric ────
    drop = 0x1A,
    select = 0x1B,
    select_t = 0x1C,

    // ──── Variable (local / global) ────
    local_get = 0x20,
    local_set = 0x21,
    local_tee = 0x22,
    global_get = 0x23,
    global_set = 0x24,

    // ──── Table ────
    table_get = 0x25,
    table_set = 0x26,

    // ──── Memory load ────
    i32_load = 0x28,
    i64_load = 0x29,
    f32_load = 0x2A,
    f64_load = 0x2B,
    i32_load8_s = 0x2C,
    i32_load8_u = 0x2D,
    i32_load16_s = 0x2E,
    i32_load16_u = 0x2F,
    i64_load8_s = 0x30,
    i64_load8_u = 0x31,
    i64_load16_s = 0x32,
    i64_load16_u = 0x33,
    i64_load32_s = 0x34,
    i64_load32_u = 0x35,

    // ──── Memory store ────
    i32_store = 0x36,
    i64_store = 0x37,
    f32_store = 0x38,
    f64_store = 0x39,
    i32_store8 = 0x3A,
    i32_store16 = 0x3B,
    i64_store8 = 0x3C,
    i64_store16 = 0x3D,
    i64_store32 = 0x3E,

    // ──── Memory ────
    memory_size = 0x3F,
    memory_grow = 0x40,

    // ──── Constants ────
    i32_const = 0x41,
    i64_const = 0x42,
    f32_const = 0x43,
    f64_const = 0x44,

    // ──── i32 comparison ────
    i32_eqz = 0x45,
    i32_eq = 0x46,
    i32_ne = 0x47,
    i32_lt_s = 0x48,
    i32_lt_u = 0x49,
    i32_gt_s = 0x4A,
    i32_gt_u = 0x4B,
    i32_le_s = 0x4C,
    i32_le_u = 0x4D,
    i32_ge_s = 0x4E,
    i32_ge_u = 0x4F,

    // ──── i64 comparison ────
    i64_eqz = 0x50,
    i64_eq = 0x51,
    i64_ne = 0x52,
    i64_lt_s = 0x53,
    i64_lt_u = 0x54,
    i64_gt_s = 0x55,
    i64_gt_u = 0x56,
    i64_le_s = 0x57,
    i64_le_u = 0x58,
    i64_ge_s = 0x59,
    i64_ge_u = 0x5A,

    // ──── f32 comparison ────
    f32_eq = 0x5B,
    f32_ne = 0x5C,
    f32_lt = 0x5D,
    f32_gt = 0x5E,
    f32_le = 0x5F,
    f32_ge = 0x60,

    // ──── f64 comparison ────
    f64_eq = 0x61,
    f64_ne = 0x62,
    f64_lt = 0x63,
    f64_gt = 0x64,
    f64_le = 0x65,
    f64_ge = 0x66,

    // ──── i32 arithmetic ────
    i32_clz = 0x67,
    i32_ctz = 0x68,
    i32_popcnt = 0x69,
    i32_add = 0x6A,
    i32_sub = 0x6B,
    i32_mul = 0x6C,
    i32_div_s = 0x6D,
    i32_div_u = 0x6E,
    i32_rem_s = 0x6F,
    i32_rem_u = 0x70,
    i32_and = 0x71,
    i32_or = 0x72,
    i32_xor = 0x73,
    i32_shl = 0x74,
    i32_shr_s = 0x75,
    i32_shr_u = 0x76,
    i32_rotl = 0x77,
    i32_rotr = 0x78,

    // ──── i64 arithmetic ────
    i64_clz = 0x79,
    i64_ctz = 0x7A,
    i64_popcnt = 0x7B,
    i64_add = 0x7C,
    i64_sub = 0x7D,
    i64_mul = 0x7E,
    i64_div_s = 0x7F,
    i64_div_u = 0x80,
    i64_rem_s = 0x81,
    i64_rem_u = 0x82,
    i64_and = 0x83,
    i64_or = 0x84,
    i64_xor = 0x85,
    i64_shl = 0x86,
    i64_shr_s = 0x87,
    i64_shr_u = 0x88,
    i64_rotl = 0x89,
    i64_rotr = 0x8A,

    // ──── f32 arithmetic ────
    f32_abs = 0x8B,
    f32_neg = 0x8C,
    f32_ceil = 0x8D,
    f32_floor = 0x8E,
    f32_trunc = 0x8F,
    f32_nearest = 0x90,
    f32_sqrt = 0x91,
    f32_add = 0x92,
    f32_sub = 0x93,
    f32_mul = 0x94,
    f32_div = 0x95,
    f32_min = 0x96,
    f32_max = 0x97,
    f32_copysign = 0x98,

    // ──── f64 arithmetic ────
    f64_abs = 0x99,
    f64_neg = 0x9A,
    f64_ceil = 0x9B,
    f64_floor = 0x9C,
    f64_trunc = 0x9D,
    f64_nearest = 0x9E,
    f64_sqrt = 0x9F,
    f64_add = 0xA0,
    f64_sub = 0xA1,
    f64_mul = 0xA2,
    f64_div = 0xA3,
    f64_min = 0xA4,
    f64_max = 0xA5,
    f64_copysign = 0xA6,

    // ──── Conversions ────
    i32_wrap_i64 = 0xA7,
    i32_trunc_f32_s = 0xA8,
    i32_trunc_f32_u = 0xA9,
    i32_trunc_f64_s = 0xAA,
    i32_trunc_f64_u = 0xAB,
    i64_extend_i32_s = 0xAC,
    i64_extend_i32_u = 0xAD,
    i64_trunc_f32_s = 0xAE,
    i64_trunc_f32_u = 0xAF,
    i64_trunc_f64_s = 0xB0,
    i64_trunc_f64_u = 0xB1,
    f32_convert_i32_s = 0xB2,
    f32_convert_i32_u = 0xB3,
    f32_convert_i64_s = 0xB4,
    f32_convert_i64_u = 0xB5,
    f32_demote_f64 = 0xB6,
    f64_convert_i32_s = 0xB7,
    f64_convert_i32_u = 0xB8,
    f64_convert_i64_s = 0xB9,
    f64_convert_i64_u = 0xBA,
    f64_promote_f32 = 0xBB,

    // ──── Reinterpretations ────
    i32_reinterpret_f32 = 0xBC,
    i64_reinterpret_f64 = 0xBD,
    f32_reinterpret_i32 = 0xBE,
    f64_reinterpret_i64 = 0xBF,

    // ──── Sign-extension (post-MVP) ────
    i32_extend8_s = 0xC0,
    i32_extend16_s = 0xC1,
    i64_extend8_s = 0xC2,
    i64_extend16_s = 0xC3,
    i64_extend32_s = 0xC4,

    // ──── Reference types (post-MVP) ────
    ref_null = 0xD0,
    ref_is_null = 0xD1,
    ref_func = 0xD2,
    ref_eq = 0xD3,
    ref_as_non_null = 0xD4,
    br_on_null = 0xD5,
    br_on_non_null = 0xD6,

    // ──── Prefix bytes for extended opcode spaces ────
    gc_prefix = 0xFB,
    misc_prefix = 0xFC,
    simd_prefix = 0xFD,
    atomic_prefix = 0xFE,

    _, // Allow unknown / internal opcodes

    /// Decode an opcode from a raw byte.
    pub fn fromByte(byte: u8) Opcode {
        return @enumFromInt(byte);
    }

    /// Encode this opcode back to its byte representation.
    pub fn toByte(self: Opcode) u8 {
        return @intFromEnum(self);
    }

    /// Returns true if the opcode is a control-flow instruction.
    pub fn isControl(self: Opcode) bool {
        return switch (self) {
            .@"unreachable",
            .block,
            .loop,
            .@"if",
            .@"else",
            .@"try",
            .@"catch",
            .throw,
            .rethrow,
            .end,
            .br,
            .br_if,
            .br_table,
            .@"return",
            .delegate,
            .catch_all,
            => true,
            else => false,
        };
    }

    /// Returns true if the opcode is a call instruction.
    pub fn isCall(self: Opcode) bool {
        return switch (self) {
            .call,
            .call_indirect,
            .return_call,
            .return_call_indirect,
            .call_ref,
            .return_call_ref,
            => true,
            else => false,
        };
    }

    /// Returns the number of fixed-width immediate bytes for common opcodes.
    /// Returns `null` for variable-length immediates (LEB128-encoded values,
    /// `br_table`, `call_indirect`, etc.).
    pub fn immediateSize(self: Opcode) ?u8 {
        return switch (self) {
            .@"unreachable", .nop, .@"else", .end, .@"return", .drop, .select => 0,

            // block-type byte
            .block, .loop, .@"if" => null,

            // LEB128 index / depth
            .br, .br_if, .call, .local_get, .local_set, .local_tee, .global_get, .global_set => null,

            // variable-length
            .br_table, .call_indirect, .select_t => null,

            // memory load/store: alignment + offset (both LEB128)
            .i32_load,
            .i64_load,
            .f32_load,
            .f64_load,
            .i32_load8_s,
            .i32_load8_u,
            .i32_load16_s,
            .i32_load16_u,
            .i64_load8_s,
            .i64_load8_u,
            .i64_load16_s,
            .i64_load16_u,
            .i64_load32_s,
            .i64_load32_u,
            .i32_store,
            .i64_store,
            .f32_store,
            .f64_store,
            .i32_store8,
            .i32_store16,
            .i64_store8,
            .i64_store16,
            .i64_store32,
            => null,

            // memory.size / memory.grow: reserved byte (LEB128 0x00)
            .memory_size, .memory_grow => null,

            // constants (LEB128)
            .i32_const, .i64_const => null,
            .f32_const => 4,
            .f64_const => 8,

            // zero-immediate arithmetic / comparison / conversion opcodes
            .i32_eqz,
            .i32_eq,
            .i32_ne,
            .i32_lt_s,
            .i32_lt_u,
            .i32_gt_s,
            .i32_gt_u,
            .i32_le_s,
            .i32_le_u,
            .i32_ge_s,
            .i32_ge_u,
            .i64_eqz,
            .i64_eq,
            .i64_ne,
            .i64_lt_s,
            .i64_lt_u,
            .i64_gt_s,
            .i64_gt_u,
            .i64_le_s,
            .i64_le_u,
            .i64_ge_s,
            .i64_ge_u,
            .f32_eq,
            .f32_ne,
            .f32_lt,
            .f32_gt,
            .f32_le,
            .f32_ge,
            .f64_eq,
            .f64_ne,
            .f64_lt,
            .f64_gt,
            .f64_le,
            .f64_ge,
            .i32_clz,
            .i32_ctz,
            .i32_popcnt,
            .i32_add,
            .i32_sub,
            .i32_mul,
            .i32_div_s,
            .i32_div_u,
            .i32_rem_s,
            .i32_rem_u,
            .i32_and,
            .i32_or,
            .i32_xor,
            .i32_shl,
            .i32_shr_s,
            .i32_shr_u,
            .i32_rotl,
            .i32_rotr,
            .i64_clz,
            .i64_ctz,
            .i64_popcnt,
            .i64_add,
            .i64_sub,
            .i64_mul,
            .i64_div_s,
            .i64_div_u,
            .i64_rem_s,
            .i64_rem_u,
            .i64_and,
            .i64_or,
            .i64_xor,
            .i64_shl,
            .i64_shr_s,
            .i64_shr_u,
            .i64_rotl,
            .i64_rotr,
            .f32_abs,
            .f32_neg,
            .f32_ceil,
            .f32_floor,
            .f32_trunc,
            .f32_nearest,
            .f32_sqrt,
            .f32_add,
            .f32_sub,
            .f32_mul,
            .f32_div,
            .f32_min,
            .f32_max,
            .f32_copysign,
            .f64_abs,
            .f64_neg,
            .f64_ceil,
            .f64_floor,
            .f64_trunc,
            .f64_nearest,
            .f64_sqrt,
            .f64_add,
            .f64_sub,
            .f64_mul,
            .f64_div,
            .f64_min,
            .f64_max,
            .f64_copysign,
            .i32_wrap_i64,
            .i32_trunc_f32_s,
            .i32_trunc_f32_u,
            .i32_trunc_f64_s,
            .i32_trunc_f64_u,
            .i64_extend_i32_s,
            .i64_extend_i32_u,
            .i64_trunc_f32_s,
            .i64_trunc_f32_u,
            .i64_trunc_f64_s,
            .i64_trunc_f64_u,
            .f32_convert_i32_s,
            .f32_convert_i32_u,
            .f32_convert_i64_s,
            .f32_convert_i64_u,
            .f32_demote_f64,
            .f64_convert_i32_s,
            .f64_convert_i32_u,
            .f64_convert_i64_s,
            .f64_convert_i64_u,
            .f64_promote_f32,
            .i32_reinterpret_f32,
            .i64_reinterpret_f64,
            .f32_reinterpret_i32,
            .f64_reinterpret_i64,
            .i32_extend8_s,
            .i32_extend16_s,
            .i64_extend8_s,
            .i64_extend16_s,
            .i64_extend32_s,
            => 0,

            // reference types (LEB128 immediates)
            .ref_null, .ref_func => null,
            .ref_is_null, .ref_eq, .ref_as_non_null => 0,
            .br_on_null, .br_on_non_null => null,

            // prefix bytes – the "immediate" is a secondary opcode (LEB128 u32)
            .gc_prefix, .misc_prefix, .simd_prefix, .atomic_prefix => null,

            // exception handling
            .@"try" => null,
            .@"catch", .throw, .rethrow, .delegate, .catch_all => null,

            // tail calls
            .return_call, .return_call_indirect, .call_ref, .return_call_ref => null,

            // table ops
            .table_get, .table_set => null,

            _ => null,
        };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Extended opcode enums (secondary opcode after a prefix byte)
// ─────────────────────────────────────────────────────────────────────────────

/// Extended opcodes after the 0xFB prefix byte (GC proposal).
pub const GcOpcode = enum(u32) {
    struct_new = 0x00,
    struct_new_default = 0x01,
    struct_get = 0x02,
    struct_get_s = 0x03,
    struct_get_u = 0x04,
    struct_set = 0x05,

    array_new = 0x06,
    array_new_default = 0x07,
    array_new_fixed = 0x08,
    array_new_data = 0x09,
    array_new_elem = 0x0A,
    array_get = 0x0B,
    array_get_s = 0x0C,
    array_get_u = 0x0D,
    array_set = 0x0E,
    array_len = 0x0F,
    array_fill = 0x10,
    array_copy = 0x11,
    array_init_data = 0x12,
    array_init_elem = 0x13,

    ref_test = 0x14,
    ref_test_nullable = 0x15,
    ref_cast = 0x16,
    ref_cast_nullable = 0x17,

    br_on_cast = 0x18,
    br_on_cast_fail = 0x19,

    any_convert_extern = 0x1A,
    extern_convert_any = 0x1B,

    ref_i31 = 0x1C,
    i31_get_s = 0x1D,
    i31_get_u = 0x1E,

    // ──── Stringref ────
    string_new_utf8 = 0x80,
    string_new_wtf16 = 0x81,
    string_const = 0x82,
    string_measure_utf8 = 0x83,
    string_measure_wtf8 = 0x84,
    string_measure_wtf16 = 0x85,
    string_encode_utf8 = 0x86,
    string_encode_wtf16 = 0x87,
    string_concat = 0x88,
    string_eq = 0x89,
    string_is_usv_sequence = 0x8A,
    string_new_lossy_utf8 = 0x8B,
    string_new_wtf8 = 0x8C,
    string_encode_lossy_utf8 = 0x8D,
    string_encode_wtf8 = 0x8E,

    string_as_wtf8 = 0x90,
    stringview_wtf8_advance = 0x91,
    stringview_wtf8_encode_utf8 = 0x92,
    stringview_wtf8_slice = 0x93,
    stringview_wtf8_encode_lossy_utf8 = 0x94,
    stringview_wtf8_encode_wtf8 = 0x95,

    string_as_wtf16 = 0x98,
    stringview_wtf16_length = 0x99,
    stringview_wtf16_get_codeunit = 0x9A,
    stringview_wtf16_encode = 0x9B,
    stringview_wtf16_slice = 0x9C,

    string_as_iter = 0xA0,
    stringview_iter_next = 0xA1,
    stringview_iter_advance = 0xA2,
    stringview_iter_rewind = 0xA3,
    stringview_iter_slice = 0xA4,

    string_new_utf8_array = 0xB0,
    string_new_wtf16_array = 0xB1,
    string_encode_utf8_array = 0xB2,
    string_encode_wtf16_array = 0xB3,
    string_new_lossy_utf8_array = 0xB4,
    string_new_wtf8_array = 0xB5,
    string_encode_lossy_utf8_array = 0xB6,
    string_encode_wtf8_array = 0xB7,

    _,
};

/// Extended opcodes after the 0xFC prefix byte (saturating truncation, bulk memory, table ops).
pub const MiscOpcode = enum(u32) {
    i32_trunc_sat_f32_s = 0x00,
    i32_trunc_sat_f32_u = 0x01,
    i32_trunc_sat_f64_s = 0x02,
    i32_trunc_sat_f64_u = 0x03,
    i64_trunc_sat_f32_s = 0x04,
    i64_trunc_sat_f32_u = 0x05,
    i64_trunc_sat_f64_s = 0x06,
    i64_trunc_sat_f64_u = 0x07,
    memory_init = 0x08,
    data_drop = 0x09,
    memory_copy = 0x0A,
    memory_fill = 0x0B,
    table_init = 0x0C,
    elem_drop = 0x0D,
    table_copy = 0x0E,
    table_grow = 0x0F,
    table_size = 0x10,
    table_fill = 0x11,

    _,
};

/// Extended opcodes after the 0xFD prefix byte (SIMD).
pub const SimdOpcode = enum(u32) {
    // ──── Memory ────
    v128_load = 0x00,
    v128_load8x8_s = 0x01,
    v128_load8x8_u = 0x02,
    v128_load16x4_s = 0x03,
    v128_load16x4_u = 0x04,
    v128_load32x2_s = 0x05,
    v128_load32x2_u = 0x06,
    v128_load8_splat = 0x07,
    v128_load16_splat = 0x08,
    v128_load32_splat = 0x09,
    v128_load64_splat = 0x0A,
    v128_store = 0x0B,

    // ──── Basic ────
    v128_const = 0x0C,
    i8x16_shuffle = 0x0D,
    i8x16_swizzle = 0x0E,

    // ──── Splat ────
    i8x16_splat = 0x0F,
    i16x8_splat = 0x10,
    i32x4_splat = 0x11,
    i64x2_splat = 0x12,
    f32x4_splat = 0x13,
    f64x2_splat = 0x14,

    // ──── Lane ────
    i8x16_extract_lane_s = 0x15,
    i8x16_extract_lane_u = 0x16,
    i8x16_replace_lane = 0x17,
    i16x8_extract_lane_s = 0x18,
    i16x8_extract_lane_u = 0x19,
    i16x8_replace_lane = 0x1A,
    i32x4_extract_lane = 0x1B,
    i32x4_replace_lane = 0x1C,
    i64x2_extract_lane = 0x1D,
    i64x2_replace_lane = 0x1E,
    f32x4_extract_lane = 0x1F,
    f32x4_replace_lane = 0x20,
    f64x2_extract_lane = 0x21,
    f64x2_replace_lane = 0x22,

    // ──── i8x16 comparison ────
    i8x16_eq = 0x23,
    i8x16_ne = 0x24,
    i8x16_lt_s = 0x25,
    i8x16_lt_u = 0x26,
    i8x16_gt_s = 0x27,
    i8x16_gt_u = 0x28,
    i8x16_le_s = 0x29,
    i8x16_le_u = 0x2A,
    i8x16_ge_s = 0x2B,
    i8x16_ge_u = 0x2C,

    // ──── i16x8 comparison ────
    i16x8_eq = 0x2D,
    i16x8_ne = 0x2E,
    i16x8_lt_s = 0x2F,
    i16x8_lt_u = 0x30,
    i16x8_gt_s = 0x31,
    i16x8_gt_u = 0x32,
    i16x8_le_s = 0x33,
    i16x8_le_u = 0x34,
    i16x8_ge_s = 0x35,
    i16x8_ge_u = 0x36,

    // ──── i32x4 comparison ────
    i32x4_eq = 0x37,
    i32x4_ne = 0x38,
    i32x4_lt_s = 0x39,
    i32x4_lt_u = 0x3A,
    i32x4_gt_s = 0x3B,
    i32x4_gt_u = 0x3C,
    i32x4_le_s = 0x3D,
    i32x4_le_u = 0x3E,
    i32x4_ge_s = 0x3F,
    i32x4_ge_u = 0x40,

    // ──── f32x4 comparison ────
    f32x4_eq = 0x41,
    f32x4_ne = 0x42,
    f32x4_lt = 0x43,
    f32x4_gt = 0x44,
    f32x4_le = 0x45,
    f32x4_ge = 0x46,

    // ──── f64x2 comparison ────
    f64x2_eq = 0x47,
    f64x2_ne = 0x48,
    f64x2_lt = 0x49,
    f64x2_gt = 0x4A,
    f64x2_le = 0x4B,
    f64x2_ge = 0x4C,

    // ──── v128 bitwise ────
    v128_not = 0x4D,
    v128_and = 0x4E,
    v128_andnot = 0x4F,
    v128_or = 0x50,
    v128_xor = 0x51,
    v128_bitselect = 0x52,
    v128_any_true = 0x53,

    // ──── Load/store lane ────
    v128_load8_lane = 0x54,
    v128_load16_lane = 0x55,
    v128_load32_lane = 0x56,
    v128_load64_lane = 0x57,
    v128_store8_lane = 0x58,
    v128_store16_lane = 0x59,
    v128_store32_lane = 0x5A,
    v128_store64_lane = 0x5B,
    v128_load32_zero = 0x5C,
    v128_load64_zero = 0x5D,

    // ──── Float conversion ────
    f32x4_demote_f64x2_zero = 0x5E,
    f64x2_promote_low_f32x4_zero = 0x5F,

    // ──── i8x16 operations ────
    i8x16_abs = 0x60,
    i8x16_neg = 0x61,
    i8x16_popcnt = 0x62,
    i8x16_all_true = 0x63,
    i8x16_bitmask = 0x64,
    i8x16_narrow_i16x8_s = 0x65,
    i8x16_narrow_i16x8_u = 0x66,
    f32x4_ceil = 0x67,
    f32x4_floor = 0x68,
    f32x4_trunc = 0x69,
    f32x4_nearest = 0x6A,
    i8x16_shl = 0x6B,
    i8x16_shr_s = 0x6C,
    i8x16_shr_u = 0x6D,
    i8x16_add = 0x6E,
    i8x16_add_sat_s = 0x6F,
    i8x16_add_sat_u = 0x70,
    i8x16_sub = 0x71,
    i8x16_sub_sat_s = 0x72,
    i8x16_sub_sat_u = 0x73,
    f64x2_ceil = 0x74,
    f64x2_floor = 0x75,
    i8x16_min_s = 0x76,
    i8x16_min_u = 0x77,
    i8x16_max_s = 0x78,
    i8x16_max_u = 0x79,
    f64x2_trunc = 0x7A,
    i8x16_avgr_u = 0x7B,
    i16x8_extadd_pairwise_i8x16_s = 0x7C,
    i16x8_extadd_pairwise_i8x16_u = 0x7D,
    i32x4_extadd_pairwise_i16x8_s = 0x7E,
    i32x4_extadd_pairwise_i16x8_u = 0x7F,

    // ──── i16x8 operations ────
    i16x8_abs = 0x80,
    i16x8_neg = 0x81,
    i16x8_q15mulr_sat_s = 0x82,
    i16x8_all_true = 0x83,
    i16x8_bitmask = 0x84,
    i16x8_narrow_i32x4_s = 0x85,
    i16x8_narrow_i32x4_u = 0x86,
    i16x8_extend_low_i8x16_s = 0x87,
    i16x8_extend_high_i8x16_s = 0x88,
    i16x8_extend_low_i8x16_u = 0x89,
    i16x8_extend_high_i8x16_u = 0x8A,
    i16x8_shl = 0x8B,
    i16x8_shr_s = 0x8C,
    i16x8_shr_u = 0x8D,
    i16x8_add = 0x8E,
    i16x8_add_sat_s = 0x8F,
    i16x8_add_sat_u = 0x90,
    i16x8_sub = 0x91,
    i16x8_sub_sat_s = 0x92,
    i16x8_sub_sat_u = 0x93,
    f64x2_nearest = 0x94,
    i16x8_mul = 0x95,
    i16x8_min_s = 0x96,
    i16x8_min_u = 0x97,
    i16x8_max_s = 0x98,
    i16x8_max_u = 0x99,
    // 0x9A placeholder
    i16x8_avgr_u = 0x9B,
    i16x8_extmul_low_i8x16_s = 0x9C,
    i16x8_extmul_high_i8x16_s = 0x9D,
    i16x8_extmul_low_i8x16_u = 0x9E,
    i16x8_extmul_high_i8x16_u = 0x9F,

    // ──── i32x4 operations ────
    i32x4_abs = 0xA0,
    i32x4_neg = 0xA1,
    // 0xA2 placeholder
    i32x4_all_true = 0xA3,
    i32x4_bitmask = 0xA4,
    // 0xA5–0xA6 placeholder
    i32x4_extend_low_i16x8_s = 0xA7,
    i32x4_extend_high_i16x8_s = 0xA8,
    i32x4_extend_low_i16x8_u = 0xA9,
    i32x4_extend_high_i16x8_u = 0xAA,
    i32x4_shl = 0xAB,
    i32x4_shr_s = 0xAC,
    i32x4_shr_u = 0xAD,
    i32x4_add = 0xAE,
    // 0xAF–0xB0 placeholder
    i32x4_sub = 0xB1,
    // 0xB2–0xB4 placeholder
    i32x4_mul = 0xB5,
    i32x4_min_s = 0xB6,
    i32x4_min_u = 0xB7,
    i32x4_max_s = 0xB8,
    i32x4_max_u = 0xB9,
    i32x4_dot_i16x8_s = 0xBA,
    // 0xBB placeholder
    i32x4_extmul_low_i16x8_s = 0xBC,
    i32x4_extmul_high_i16x8_s = 0xBD,
    i32x4_extmul_low_i16x8_u = 0xBE,
    i32x4_extmul_high_i16x8_u = 0xBF,

    // ──── i64x2 operations ────
    i64x2_abs = 0xC0,
    i64x2_neg = 0xC1,
    // 0xC2 placeholder
    i64x2_all_true = 0xC3,
    i64x2_bitmask = 0xC4,
    // 0xC5–0xC6 placeholder
    i64x2_extend_low_i32x4_s = 0xC7,
    i64x2_extend_high_i32x4_s = 0xC8,
    i64x2_extend_low_i32x4_u = 0xC9,
    i64x2_extend_high_i32x4_u = 0xCA,
    i64x2_shl = 0xCB,
    i64x2_shr_s = 0xCC,
    i64x2_shr_u = 0xCD,
    i64x2_add = 0xCE,
    // 0xCF–0xD0 placeholder
    i64x2_sub = 0xD1,
    // 0xD2–0xD4 placeholder
    i64x2_mul = 0xD5,
    i64x2_eq = 0xD6,
    i64x2_ne = 0xD7,
    i64x2_lt_s = 0xD8,
    i64x2_gt_s = 0xD9,
    i64x2_le_s = 0xDA,
    i64x2_ge_s = 0xDB,
    i64x2_extmul_low_i32x4_s = 0xDC,
    i64x2_extmul_high_i32x4_s = 0xDD,
    i64x2_extmul_low_i32x4_u = 0xDE,
    i64x2_extmul_high_i32x4_u = 0xDF,

    // ──── f32x4 operations ────
    f32x4_abs = 0xE0,
    f32x4_neg = 0xE1,
    // 0xE2 placeholder
    f32x4_sqrt = 0xE3,
    f32x4_add = 0xE4,
    f32x4_sub = 0xE5,
    f32x4_mul = 0xE6,
    f32x4_div = 0xE7,
    f32x4_min = 0xE8,
    f32x4_max = 0xE9,
    f32x4_pmin = 0xEA,
    f32x4_pmax = 0xEB,

    // ──── f64x2 operations ────
    f64x2_abs = 0xEC,
    f64x2_neg = 0xED,
    // 0xEE placeholder
    f64x2_sqrt = 0xEF,
    f64x2_add = 0xF0,
    f64x2_sub = 0xF1,
    f64x2_mul = 0xF2,
    f64x2_div = 0xF3,
    f64x2_min = 0xF4,
    f64x2_max = 0xF5,
    f64x2_pmin = 0xF6,
    f64x2_pmax = 0xF7,

    // ──── Conversions ────
    i32x4_trunc_sat_f32x4_s = 0xF8,
    i32x4_trunc_sat_f32x4_u = 0xF9,
    f32x4_convert_i32x4_s = 0xFA,
    f32x4_convert_i32x4_u = 0xFB,
    i32x4_trunc_sat_f64x2_s_zero = 0xFC,
    i32x4_trunc_sat_f64x2_u_zero = 0xFD,
    f64x2_convert_low_i32x4_s = 0xFE,
    f64x2_convert_low_i32x4_u = 0xFF,

    _,
};

/// Extended opcodes after the 0xFE prefix byte (threads / atomics).
pub const AtomicOpcode = enum(u32) {
    // ──── Wait / notify ────
    memory_atomic_notify = 0x00,
    memory_atomic_wait32 = 0x01,
    memory_atomic_wait64 = 0x02,
    atomic_fence = 0x03,

    // ──── Load ────
    i32_atomic_load = 0x10,
    i64_atomic_load = 0x11,
    i32_atomic_load8_u = 0x12,
    i32_atomic_load16_u = 0x13,
    i64_atomic_load8_u = 0x14,
    i64_atomic_load16_u = 0x15,
    i64_atomic_load32_u = 0x16,

    // ──── Store ────
    i32_atomic_store = 0x17,
    i64_atomic_store = 0x18,
    i32_atomic_store8 = 0x19,
    i32_atomic_store16 = 0x1A,
    i64_atomic_store8 = 0x1B,
    i64_atomic_store16 = 0x1C,
    i64_atomic_store32 = 0x1D,

    // ──── RMW add ────
    i32_atomic_rmw_add = 0x1E,
    i64_atomic_rmw_add = 0x1F,
    i32_atomic_rmw8_add_u = 0x20,
    i32_atomic_rmw16_add_u = 0x21,
    i64_atomic_rmw8_add_u = 0x22,
    i64_atomic_rmw16_add_u = 0x23,
    i64_atomic_rmw32_add_u = 0x24,

    // ──── RMW sub ────
    i32_atomic_rmw_sub = 0x25,
    i64_atomic_rmw_sub = 0x26,
    i32_atomic_rmw8_sub_u = 0x27,
    i32_atomic_rmw16_sub_u = 0x28,
    i64_atomic_rmw8_sub_u = 0x29,
    i64_atomic_rmw16_sub_u = 0x2A,
    i64_atomic_rmw32_sub_u = 0x2B,

    // ──── RMW and ────
    i32_atomic_rmw_and = 0x2C,
    i64_atomic_rmw_and = 0x2D,
    i32_atomic_rmw8_and_u = 0x2E,
    i32_atomic_rmw16_and_u = 0x2F,
    i64_atomic_rmw8_and_u = 0x30,
    i64_atomic_rmw16_and_u = 0x31,
    i64_atomic_rmw32_and_u = 0x32,

    // ──── RMW or ────
    i32_atomic_rmw_or = 0x33,
    i64_atomic_rmw_or = 0x34,
    i32_atomic_rmw8_or_u = 0x35,
    i32_atomic_rmw16_or_u = 0x36,
    i64_atomic_rmw8_or_u = 0x37,
    i64_atomic_rmw16_or_u = 0x38,
    i64_atomic_rmw32_or_u = 0x39,

    // ──── RMW xor ────
    i32_atomic_rmw_xor = 0x3A,
    i64_atomic_rmw_xor = 0x3B,
    i32_atomic_rmw8_xor_u = 0x3C,
    i32_atomic_rmw16_xor_u = 0x3D,
    i64_atomic_rmw8_xor_u = 0x3E,
    i64_atomic_rmw16_xor_u = 0x3F,
    i64_atomic_rmw32_xor_u = 0x40,

    // ──── RMW xchg ────
    i32_atomic_rmw_xchg = 0x41,
    i64_atomic_rmw_xchg = 0x42,
    i32_atomic_rmw8_xchg_u = 0x43,
    i32_atomic_rmw16_xchg_u = 0x44,
    i64_atomic_rmw8_xchg_u = 0x45,
    i64_atomic_rmw16_xchg_u = 0x46,
    i64_atomic_rmw32_xchg_u = 0x47,

    // ──── RMW cmpxchg ────
    i32_atomic_rmw_cmpxchg = 0x48,
    i64_atomic_rmw_cmpxchg = 0x49,
    i32_atomic_rmw8_cmpxchg_u = 0x4A,
    i32_atomic_rmw16_cmpxchg_u = 0x4B,
    i64_atomic_rmw8_cmpxchg_u = 0x4C,
    i64_atomic_rmw16_cmpxchg_u = 0x4D,
    i64_atomic_rmw32_cmpxchg_u = 0x4E,

    _,
};

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test "opcode values match Wasm spec" {
    // MVP control flow
    try testing.expectEqual(@as(u8, 0x00), @intFromEnum(Opcode.@"unreachable"));
    try testing.expectEqual(@as(u8, 0x01), @intFromEnum(Opcode.nop));
    try testing.expectEqual(@as(u8, 0x0B), @intFromEnum(Opcode.end));
    try testing.expectEqual(@as(u8, 0x0F), @intFromEnum(Opcode.@"return"));
    try testing.expectEqual(@as(u8, 0x10), @intFromEnum(Opcode.call));
    try testing.expectEqual(@as(u8, 0x11), @intFromEnum(Opcode.call_indirect));

    // Variables
    try testing.expectEqual(@as(u8, 0x20), @intFromEnum(Opcode.local_get));
    try testing.expectEqual(@as(u8, 0x24), @intFromEnum(Opcode.global_set));

    // Memory
    try testing.expectEqual(@as(u8, 0x28), @intFromEnum(Opcode.i32_load));
    try testing.expectEqual(@as(u8, 0x36), @intFromEnum(Opcode.i32_store));
    try testing.expectEqual(@as(u8, 0x3F), @intFromEnum(Opcode.memory_size));
    try testing.expectEqual(@as(u8, 0x40), @intFromEnum(Opcode.memory_grow));

    // Constants
    try testing.expectEqual(@as(u8, 0x41), @intFromEnum(Opcode.i32_const));
    try testing.expectEqual(@as(u8, 0x44), @intFromEnum(Opcode.f64_const));

    // Arithmetic boundary checks
    try testing.expectEqual(@as(u8, 0x6A), @intFromEnum(Opcode.i32_add));
    try testing.expectEqual(@as(u8, 0x7C), @intFromEnum(Opcode.i64_add));
    try testing.expectEqual(@as(u8, 0xA0), @intFromEnum(Opcode.f64_add));

    // Conversions
    try testing.expectEqual(@as(u8, 0xA7), @intFromEnum(Opcode.i32_wrap_i64));
    try testing.expectEqual(@as(u8, 0xBB), @intFromEnum(Opcode.f64_promote_f32));

    // Reinterpret
    try testing.expectEqual(@as(u8, 0xBC), @intFromEnum(Opcode.i32_reinterpret_f32));
    try testing.expectEqual(@as(u8, 0xBF), @intFromEnum(Opcode.f64_reinterpret_i64));

    // Sign extension
    try testing.expectEqual(@as(u8, 0xC0), @intFromEnum(Opcode.i32_extend8_s));
    try testing.expectEqual(@as(u8, 0xC4), @intFromEnum(Opcode.i64_extend32_s));

    // Reference types
    try testing.expectEqual(@as(u8, 0xD0), @intFromEnum(Opcode.ref_null));
    try testing.expectEqual(@as(u8, 0xD2), @intFromEnum(Opcode.ref_func));
}

test "prefix opcodes are correct" {
    try testing.expectEqual(@as(u8, 0xFB), @intFromEnum(Opcode.gc_prefix));
    try testing.expectEqual(@as(u8, 0xFC), @intFromEnum(Opcode.misc_prefix));
    try testing.expectEqual(@as(u8, 0xFD), @intFromEnum(Opcode.simd_prefix));
    try testing.expectEqual(@as(u8, 0xFE), @intFromEnum(Opcode.atomic_prefix));
}

test "fromByte / toByte roundtrip" {
    const op = Opcode.fromByte(0x10);
    try testing.expectEqual(Opcode.call, op);
    try testing.expectEqual(@as(u8, 0x10), op.toByte());

    // Unknown byte survives the roundtrip.
    const unknown = Opcode.fromByte(0xE7);
    try testing.expectEqual(@as(u8, 0xE7), unknown.toByte());
}

test "isControl identifies control-flow opcodes" {
    try testing.expect(Opcode.block.isControl());
    try testing.expect(Opcode.loop.isControl());
    try testing.expect(Opcode.@"if".isControl());
    try testing.expect(Opcode.br.isControl());
    try testing.expect(Opcode.br_table.isControl());
    try testing.expect(Opcode.@"return".isControl());
    try testing.expect(Opcode.end.isControl());
    try testing.expect(Opcode.@"unreachable".isControl());

    try testing.expect(!Opcode.call.isControl());
    try testing.expect(!Opcode.i32_add.isControl());
    try testing.expect(!Opcode.nop.isControl());
}

test "isCall identifies call opcodes" {
    try testing.expect(Opcode.call.isCall());
    try testing.expect(Opcode.call_indirect.isCall());
    try testing.expect(Opcode.return_call.isCall());
    try testing.expect(Opcode.return_call_indirect.isCall());
    try testing.expect(Opcode.call_ref.isCall());
    try testing.expect(Opcode.return_call_ref.isCall());

    try testing.expect(!Opcode.br.isCall());
    try testing.expect(!Opcode.i32_add.isCall());
}

test "immediateSize for known opcodes" {
    try testing.expectEqual(@as(?u8, 0), Opcode.nop.immediateSize());
    try testing.expectEqual(@as(?u8, 0), Opcode.drop.immediateSize());
    try testing.expectEqual(@as(?u8, 0), Opcode.i32_add.immediateSize());
    try testing.expectEqual(@as(?u8, 4), Opcode.f32_const.immediateSize());
    try testing.expectEqual(@as(?u8, 8), Opcode.f64_const.immediateSize());
    try testing.expectEqual(@as(?u8, null), Opcode.br_table.immediateSize());
    try testing.expectEqual(@as(?u8, null), Opcode.call.immediateSize());
    try testing.expectEqual(@as(?u8, null), Opcode.i32_const.immediateSize());
}

test "MiscOpcode values" {
    try testing.expectEqual(@as(u32, 0), @intFromEnum(MiscOpcode.i32_trunc_sat_f32_s));
    try testing.expectEqual(@as(u32, 8), @intFromEnum(MiscOpcode.memory_init));
    try testing.expectEqual(@as(u32, 0x0A), @intFromEnum(MiscOpcode.memory_copy));
    try testing.expectEqual(@as(u32, 0x0B), @intFromEnum(MiscOpcode.memory_fill));
    try testing.expectEqual(@as(u32, 0x11), @intFromEnum(MiscOpcode.table_fill));
}

test "SimdOpcode values" {
    try testing.expectEqual(@as(u32, 0x00), @intFromEnum(SimdOpcode.v128_load));
    try testing.expectEqual(@as(u32, 0x0B), @intFromEnum(SimdOpcode.v128_store));
    try testing.expectEqual(@as(u32, 0x0C), @intFromEnum(SimdOpcode.v128_const));
    try testing.expectEqual(@as(u32, 0x0D), @intFromEnum(SimdOpcode.i8x16_shuffle));
    try testing.expectEqual(@as(u32, 0xFF), @intFromEnum(SimdOpcode.f64x2_convert_low_i32x4_u));
}

test "AtomicOpcode values" {
    try testing.expectEqual(@as(u32, 0x00), @intFromEnum(AtomicOpcode.memory_atomic_notify));
    try testing.expectEqual(@as(u32, 0x01), @intFromEnum(AtomicOpcode.memory_atomic_wait32));
    try testing.expectEqual(@as(u32, 0x03), @intFromEnum(AtomicOpcode.atomic_fence));
    try testing.expectEqual(@as(u32, 0x10), @intFromEnum(AtomicOpcode.i32_atomic_load));
    try testing.expectEqual(@as(u32, 0x48), @intFromEnum(AtomicOpcode.i32_atomic_rmw_cmpxchg));
    try testing.expectEqual(@as(u32, 0x4E), @intFromEnum(AtomicOpcode.i64_atomic_rmw32_cmpxchg_u));
}

test "GcOpcode values" {
    try testing.expectEqual(@as(u32, 0x00), @intFromEnum(GcOpcode.struct_new));
    try testing.expectEqual(@as(u32, 0x06), @intFromEnum(GcOpcode.array_new));
    try testing.expectEqual(@as(u32, 0x1C), @intFromEnum(GcOpcode.ref_i31));
    try testing.expectEqual(@as(u32, 0x80), @intFromEnum(GcOpcode.string_new_utf8));
}

test {
    _ = @import("loader.zig");
}
