//! Comptime Canonical ABI Reflection, Lowering, Lifting & WAMR Binding Engine
//!
//! Ported from `wit-zigar`'s `src/canon.zig` and integrated with WAMR's
//! `ComponentInstance` and `CallFrame` (`InterpFrame` / `AotFrame`).
//!
//! Supports both `wasm32` (`u32` addresses/lengths) and `wasm64` (`memory64`,
//! `u64` addresses/lengths), comptime type layout & flat-register ABI
//! derivation (`CanonLift` / `CanonLower`), zero-heap-`InterfaceValue`
//! host import adapters (`makeHostFunc` / `makeHostFuncWithArch`), and
//! comptime-typed guest export calls (`callTypedExport`).

const std = @import("std");
const core_types = @import("../runtime/common/types.zig");
const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
const aot_runtime = @import("../runtime/aot/runtime.zig");
const abi = @import("canonical_abi.zig");
const ctypes = @import("types.zig");
const comp_instance = @import("instance.zig");
const call_frame = @import("call_frame.zig");
const config = @import("../config.zig");

const CallFrame = call_frame.CallFrame;
const InterpFrame = call_frame.InterpFrame;
const AotFrame = call_frame.AotFrame;
const CoreFuncIdxLocal = call_frame.CoreFuncIdxLocal;

// ============================================================================
// Section 1: Target Architecture & Core Flat Value Types
// ============================================================================

/// Target linear-memory address width for Canonical ABI operations.
/// - `.wasm32`: standard Component Model ABI (4-byte `i32`/`u32` pointers & lengths)
/// - `.wasm64`: memory64 extension (8-byte `i64`/`u64` pointers & lengths)
pub const Arch = enum {
    wasm32,
    wasm64,

    pub fn AddrInt(comptime self: Arch) type {
        return switch (self) {
            .wasm32 => u32,
            .wasm64 => u64,
        };
    }

    pub fn FlatAddrType(comptime self: Arch) type {
        return switch (self) {
            .wasm32 => i32,
            .wasm64 => i64,
        };
    }

    pub fn flatAddrValType(comptime self: Arch) FlatValType {
        return switch (self) {
            .wasm32 => .i32,
            .wasm64 => .i64,
        };
    }
};

pub fn AddrInt(comptime arch: Arch) type {
    return arch.AddrInt();
}

pub fn FlatAddrType(comptime arch: Arch) type {
    return arch.FlatAddrType();
}

pub fn PtrInt(comptime is_memory64: bool) type {
    return if (is_memory64) u64 else u32;
}

/// Core WebAssembly value types used in the flattened register representation
/// of the Canonical ABI.
pub const FlatValType = enum {
    i32,
    i64,
    f32,
    f64,

    /// Returns the corresponding Zig scalar type for a flat core Wasm value type.
    pub fn toZigType(comptime self: FlatValType) type {
        return switch (self) {
            .i32 => i32,
            .i64 => i64,
            .f32 => f32,
            .f64 => f64,
        };
    }

    pub const ToZigType = toZigType;

    pub fn toCoreValType(self: FlatValType) core_types.ValType {
        return switch (self) {
            .i32 => .i32,
            .i64 => .i64,
            .f32 => .f32,
            .f64 => .f64,
        };
    }
};

/// Canonical ABI `join` operator for unifying flat core types across variant arms.
/// Per the Component Model spec:
/// - `join(t, t) = t`
/// - `join(i32, f32) = join(f32, i32) = i32`
/// - all other pairs `join(a, b) = i64`
pub fn joinFlatType(a: FlatValType, b: FlatValType) FlatValType {
    if (a == b) return a;
    if ((a == .i32 and b == .f32) or (a == .f32 and b == .i32)) return .i32;
    return .i64;
}

// ============================================================================
// Section 2: WIT Companion Wrapper Types & Error Set
// ============================================================================

/// Distinct type for WIT `char` (Unicode scalar value: `0..=0x10FFFF` excluding
/// surrogate range `0xD800..=0xDFFF`).
pub const Char = enum(u32) {
    _,

    pub fn fromU32(raw: u32) Error!Char {
        return validateChar(raw);
    }

    pub fn toU32(self: Char) u32 {
        return @backingInt(self);
    }
};

pub fn validateChar(raw: u32) Error!Char {
    if (raw > 0x10FFFF or (raw >= 0xD800 and raw <= 0xDFFF)) {
        return error.InvalidChar;
    }
    return @fromBackingInt(@intCast(raw));
}

/// Companion wrapper for WIT `result<Ok, Err>` when represented as a value
/// (supports arbitrary non-error-set `Err` payloads, including strings/records,
/// as well as `void` for either `Ok` or `Err`).
pub fn Result(comptime Ok: type, comptime Err: type) type {
    return struct {
        pub const is_wit_result = true;
        pub const OkType = Ok;
        pub const ErrType = Err;
        pub const Payload = union(enum) {
            ok: Ok,
            err: Err,
        };

        payload: Payload,

        pub fn ok(val: Ok) @This() {
            return .{ .payload = .{ .ok = val } };
        }

        pub fn err(val: Err) @This() {
            return .{ .payload = .{ .err = val } };
        }

        pub fn isOk(self: @This()) bool {
            return self.payload == .ok;
        }

        pub fn isErr(self: @This()) bool {
            return self.payload == .err;
        }
    };
}

/// Kind of WIT handle (`own<R>`, `borrow<R>`, `stream<T>`, `future<T>`, `error-context`).
pub const HandleKind = enum {
    own,
    borrow,
    stream,
    future,
    error_context,
};

fn resourceTypeIdxFor(comptime Resource: type) u32 {
    if (@typeInfo(Resource) == .@"struct" and @hasDecl(Resource, "resource_type_idx")) {
        return Resource.resource_type_idx;
    }
    return 0;
}

fn resourceUsesAutoRep(comptime Resource: type) bool {
    if (@typeInfo(Resource) == .@"struct" and @hasDecl(Resource, "auto_rep")) {
        return Resource.auto_rep;
    }
    return false;
}

/// Companion wrapper for WIT `own<Resource>` (32-bit handle index in the
/// component instance's resource table).
pub fn Own(comptime Resource: type) type {
    return struct {
        idx: u32,
        pub const is_wit_handle = true;
        pub const handle_kind = HandleKind.own;
        pub const ResourceType = Resource;

        pub fn rep(self: @This(), ci: *comp_instance.ComponentInstance) ?u32 {
            const rt = ci.getOrCreateResourceTable(resourceTypeIdxFor(Resource)) catch return null;
            return rt.rep(self.idx);
        }

        pub fn drop(self: @This(), ci: *comp_instance.ComponentInstance, allocator: std.mem.Allocator) ?u32 {
            const rt = ci.getOrCreateResourceTable(resourceTypeIdxFor(Resource)) catch return null;
            return rt.drop(self.idx, allocator);
        }

        pub fn newHandle(ci: *comp_instance.ComponentInstance, representation: u32, allocator: std.mem.Allocator) Error!@This() {
            const rt = ci.getOrCreateResourceTable(resourceTypeIdxFor(Resource)) catch return error.OutOfMemory;
            const h = rt.new(representation, true, allocator) catch return error.OutOfMemory;
            return .{ .idx = h };
        }
    };
}

/// Companion wrapper for WIT `borrow<Resource>` (32-bit handle index or
/// direct representation in the component instance's resource table).
pub fn Borrow(comptime Resource: type) type {
    return struct {
        idx: u32,
        pub const is_wit_handle = true;
        pub const handle_kind = HandleKind.borrow;
        pub const ResourceType = Resource;

        pub fn rep(self: @This(), ci: *comp_instance.ComponentInstance) ?u32 {
            const rt = ci.getOrCreateResourceTable(resourceTypeIdxFor(Resource)) catch return null;
            return rt.rep(self.idx);
        }
    };
}

/// Companion wrapper for WIT `stream<T>` (WASI 0.3 async stream handle).
/// Carries a 32-bit waitable stream descriptor index.
pub fn Stream(comptime T: type) type {
    return struct {
        idx: u32,
        pub const is_wit_handle = true;
        pub const handle_kind = HandleKind.stream;
        pub const PayloadType = T;
        pub const ElemType = T;
    };
}

/// Companion wrapper for WIT `future<T>` (WASI 0.3 async future handle).
/// Carries a 32-bit waitable future descriptor index.
pub fn Future(comptime T: type) type {
    return struct {
        idx: u32,
        pub const is_wit_handle = true;
        pub const handle_kind = HandleKind.future;
        pub const PayloadType = T;
        pub const ElemType = T;
    };
}

/// Companion wrapper for WIT `error-context` (WASI 0.3 rich error context handle).
pub const ErrorContext = struct {
    idx: u32,
    pub const is_wit_handle = true;
    pub const handle_kind = HandleKind.error_context;
};

/// Distinct wrapper for WIT `string` so `string` (UTF-8 validated) can be
/// distinguished at comptime from `list<u8>` (`[]const u8`).
pub const WitString = struct {
    bytes: []const u8,
    pub const is_wit_string = true;

    pub fn init(s: []const u8) WitString {
        return .{ .bytes = s };
    }

    pub fn slice(self: WitString) []const u8 {
        return self.bytes;
    }
};

/// Generic string wrapper around a slice type `T` (e.g. `String([]const u8)`).
pub fn String(comptime T: type) type {
    return struct {
        slice: T,
        pub const is_wit_string = true;
    };
}

/// Errors produced by Canonical ABI memory validation, UTF-8 checking,
/// discriminant decoding, resource table operations, and core execution.
pub const Error = error{
    OutOfBounds,
    InvalidUtf8,
    InvalidDiscriminant,
    InvalidChar,
    InvalidBool,
    ReallocFailed,
    UnalignedPointer,
    NullRealloc,
    Trap,
    AsyncPending,
    StackOverflow,
    StackUnderflow,
    MemoryNotAvailable,
    TrapInCoreFunction,
    FunctionNotFound,
    OutOfMemory,
    // Compatibility aliases with wit-zigar canon.zig
    MissingRealloc,
    MissingAllocator,
    AlignmentError,
    AsyncCallPending,
    AsyncTaskDidNotReturn,
};

pub const CanonError = Error;

pub fn CanonResult(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .error_union => |eu| (Error || eu.error_set)!eu.payload,
        else => Error!T,
    };
}

fn isWitResult(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "is_wit_result") and T.is_wit_result;
}

fn isWitHandle(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "is_wit_handle") and T.is_wit_handle;
}

fn isResourceHandle(comptime T: type) bool {
    if (!isWitHandle(T)) return false;
    return T.handle_kind == .own or T.handle_kind == .borrow;
}

fn isWitString(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "is_wit_string") and T.is_wit_string;
}

fn isGenericWitString(comptime T: type) bool {
    return isWitString(T) and @hasField(T, "slice") and !@hasField(T, "bytes");
}

// ============================================================================
// Section 3: Comptime Type Layout & Flat Representation Engine
// ============================================================================

/// Comptime-computed Canonical ABI layout and flat register representation for
/// a Zig type.
pub const TypeLayout = struct {
    /// Byte size in linear memory (`size(t)`).
    size: u64,
    /// Byte alignment in linear memory (`align(t)`), always a power of 2.
    align_: u32,
    /// Flattened core Wasm value types (`flatten_type(t)`).
    flat_types: []const FlatValType,
    /// True if lowering this type requires `cabi_realloc` (strings, slices,
    /// or compound types containing them).
    needs_realloc: bool,

    pub fn flatCount(comptime self: TypeLayout) usize {
        return self.flat_types.len;
    }
};

/// Rounds `offset` up to the next multiple of `alignment` (`alignment` must be power of 2).
pub fn alignForward(offset: u64, alignment: u32) u64 {
    if (alignment <= 1) return offset;
    const mask: u64 = @as(u64, alignment) - 1;
    return (offset + mask) & ~mask;
}

/// Discriminant byte size for a variant/enum with `num_cases` cases:
/// `1` for `<= 256` cases, `2` for `<= 65536`, `4` otherwise.
pub fn variantDiscSize(comptime num_cases: usize) u32 {
    if (num_cases == 0) @compileError("WIT variant/enum must have at least 1 case");
    if (num_cases <= 0xFF) return 1;
    if (num_cases <= 0xFFFF) return 2;
    return 4;
}

/// Returns true if `T` can be transferred between Zig and Wasm linear memory
/// via a direct byte copy on little-endian hosts without pointer fixups or
/// per-element validation.
pub fn isBlittable(comptime T: type, comptime arch: Arch) bool {
    return comptime blk: {
        _ = arch;
        if (T == Char or isWitResult(T) or isWitHandle(T) or isWitString(T)) break :blk false;
        break :blk switch (@typeInfo(T)) {
            .int => |info| switch (info.bits) {
                8, 16, 32, 64 => true,
                else => false,
            },
            .float => |info| switch (info.bits) {
                32, 64 => true,
                else => false,
            },
            else => false,
        };
    };
}

/// Computes the Canonical ABI layout (`size`, `align_`, `flat_types`, `needs_realloc`)
/// of any supported Zig type `T` at comptime.
pub fn layoutOf(comptime T: type, comptime arch: Arch) TypeLayout {
    return comptime outer: {
        @setEvalBranchQuota(100_000);
        const ptr_size: u64 = if (arch == .wasm64) 8 else 4;
        const ptr_align: u32 = if (arch == .wasm64) 8 else 4;
        const ptr_flat = arch.flatAddrValType();

        if (T == void) {
            break :outer .{
                .size = 0,
                .align_ = 1,
                .flat_types = &.{},
                .needs_realloc = false,
            };
        }

        if (T == Char) {
            break :outer .{
                .size = 4,
                .align_ = 4,
                .flat_types = &.{.i32},
                .needs_realloc = false,
            };
        }

        if (isWitHandle(T)) {
            break :outer .{
                .size = 4,
                .align_ = 4,
                .flat_types = &.{.i32},
                .needs_realloc = false,
            };
        }

        if (isWitString(T)) {
            break :outer .{
                .size = ptr_size * 2,
                .align_ = ptr_align,
                .flat_types = &.{ ptr_flat, ptr_flat },
                .needs_realloc = true,
            };
        }

        if (isWitResult(T)) {
            break :outer resultLayout(T.OkType, T.ErrType, arch);
        }

        break :outer switch (@typeInfo(T)) {
            .bool => .{
                .size = 1,
                .align_ = 1,
                .flat_types = &.{.i32},
                .needs_realloc = false,
            },
            .int => |info| switch (info.bits) {
                8 => .{ .size = 1, .align_ = 1, .flat_types = &.{.i32}, .needs_realloc = false },
                16 => .{ .size = 2, .align_ = 2, .flat_types = &.{.i32}, .needs_realloc = false },
                32 => .{ .size = 4, .align_ = 4, .flat_types = &.{.i32}, .needs_realloc = false },
                64 => .{ .size = 8, .align_ = 8, .flat_types = &.{.i64}, .needs_realloc = false },
                else => @compileError("Unsupported integer bit width for Canonical ABI: " ++ @typeName(T)),
            },
            .float => |info| switch (info.bits) {
                32 => .{ .size = 4, .align_ = 4, .flat_types = &.{.f32}, .needs_realloc = false },
                64 => .{ .size = 8, .align_ = 8, .flat_types = &.{.f64}, .needs_realloc = false },
                else => @compileError("Unsupported float bit width for Canonical ABI: " ++ @typeName(T)),
            },
            .pointer => |ptr| blk: {
                if (ptr.size != .slice) {
                    @compileError("Only slices ([]const T or []T) are supported in Canonical ABI, got: " ++ @typeName(T));
                }
                _ = layoutOf(ptr.child, arch);
                break :blk .{
                    .size = ptr_size * 2,
                    .align_ = ptr_align,
                    .flat_types = &.{ ptr_flat, ptr_flat },
                    .needs_realloc = true,
                };
            },
            .array => |arr| blk: {
                const elem_layout = layoutOf(arr.child, arch);
                var flat: []const FlatValType = &.{};
                for (0..arr.len) |_| {
                    flat = flat ++ elem_layout.flat_types;
                }
                break :blk .{
                    .size = elem_layout.size * arr.len,
                    .align_ = elem_layout.align_,
                    .flat_types = flat,
                    .needs_realloc = elem_layout.needs_realloc,
                };
            },
            .optional => |opt| blk: {
                const payload = layoutOf(opt.child, arch);
                const max_align = @max(@as(u32, 1), payload.align_);
                const payload_offset = alignForward(1, payload.align_);
                const total_size = alignForward(payload_offset + payload.size, max_align);
                const flat = &[_]FlatValType{.i32} ++ payload.flat_types;
                break :blk .{
                    .size = total_size,
                    .align_ = max_align,
                    .flat_types = flat,
                    .needs_realloc = payload.needs_realloc,
                };
            },
            .error_union => |eu| resultLayout(eu.payload, eu.error_set, arch),
            .error_set => .{
                .size = 4,
                .align_ = 4,
                .flat_types = &.{.i32},
                .needs_realloc = false,
            },
            .@"enum" => |en| blk: {
                const disc_size = variantDiscSize(en.field_names.len);
                break :blk .{
                    .size = disc_size,
                    .align_ = disc_size,
                    .flat_types = &.{.i32},
                    .needs_realloc = false,
                };
            },
            .@"struct" => |st| blk: {
                if (st.layout == .@"packed") {
                    var num_flags: usize = 0;
                    for (st.field_types) |FType| {
                        if (FType == bool) {
                            num_flags += 1;
                        } else if (@typeInfo(FType) == .int) {
                            // Padding field
                        } else {
                            @compileError("Packed struct (WIT flags) fields must be bool or padding int, got: " ++ @typeName(FType));
                        }
                    }
                    if (num_flags == 0) {
                        break :blk .{ .size = 0, .align_ = 1, .flat_types = &.{}, .needs_realloc = false };
                    } else if (num_flags <= 8) {
                        break :blk .{ .size = 1, .align_ = 1, .flat_types = &.{.i32}, .needs_realloc = false };
                    } else if (num_flags <= 16) {
                        break :blk .{ .size = 2, .align_ = 2, .flat_types = &.{.i32}, .needs_realloc = false };
                    } else if (num_flags <= 32) {
                        break :blk .{ .size = 4, .align_ = 4, .flat_types = &.{.i32}, .needs_realloc = false };
                    } else {
                        const num_words = (num_flags + 31) / 32;
                        var flat: []const FlatValType = &.{};
                        for (0..num_words) |_| {
                            flat = flat ++ &[_]FlatValType{.i32};
                        }
                        break :blk .{
                            .size = @as(u64, num_words) * 4,
                            .align_ = 4,
                            .flat_types = flat,
                            .needs_realloc = false,
                        };
                    }
                }

                var offset: u64 = 0;
                var max_align: u32 = 1;
                var flat: []const FlatValType = &.{};
                var needs_realloc = false;

                for (st.field_types) |FType| {
                    const fl = layoutOf(FType, arch);
                    offset = alignForward(offset, fl.align_);
                    offset += fl.size;
                    if (fl.align_ > max_align) max_align = fl.align_;
                    flat = flat ++ fl.flat_types;
                    if (fl.needs_realloc) needs_realloc = true;
                }
                offset = alignForward(offset, max_align);
                break :blk .{
                    .size = offset,
                    .align_ = max_align,
                    .needs_realloc = needs_realloc,
                    .flat_types = flat,
                };
            },
            .@"union" => |un| blk: {
                if (un.tag_type == null) {
                    @compileError("WIT variant unions must be tagged union(enum): " ++ @typeName(T));
                }
                const disc_size = variantDiscSize(un.field_types.len);
                var max_payload_size: u64 = 0;
                var max_payload_align: u32 = 1;
                var max_flat_len: usize = 0;
                var needs_realloc = false;

                var case_layouts: [un.field_types.len]TypeLayout = undefined;
                for (un.field_types, 0..) |FType, i| {
                    const cl = layoutOf(FType, arch);
                    case_layouts[i] = cl;
                    if (cl.size > max_payload_size) max_payload_size = cl.size;
                    if (cl.align_ > max_payload_align) max_payload_align = cl.align_;
                    if (cl.flat_types.len > max_flat_len) max_flat_len = cl.flat_types.len;
                    if (cl.needs_realloc) needs_realloc = true;
                }

                var joined_payload: [max_flat_len]FlatValType = undefined;
                for (0..max_flat_len) |slot| {
                    var have_slot = false;
                    var cur: FlatValType = .i32;
                    for (case_layouts) |cl| {
                        if (slot < cl.flat_types.len) {
                            if (!have_slot) {
                                cur = cl.flat_types[slot];
                                have_slot = true;
                            } else {
                                cur = joinFlatType(cur, cl.flat_types[slot]);
                            }
                        }
                    }
                    joined_payload[slot] = cur;
                }

                const max_align = @max(disc_size, max_payload_align);
                const payload_offset = alignForward(disc_size, max_payload_align);
                const total_size = alignForward(payload_offset + max_payload_size, max_align);
                const joined_const = joined_payload;
                const flat = &[_]FlatValType{.i32} ++ joined_const[0..max_flat_len];
                break :blk .{
                    .size = total_size,
                    .align_ = max_align,
                    .flat_types = flat,
                    .needs_realloc = needs_realloc,
                };
            },
            else => @compileError("Unsupported type for Canonical ABI: " ++ @typeName(T)),
        };
    };
}

fn resultLayout(comptime Ok: type, comptime Err: type, comptime arch: Arch) TypeLayout {
    return comptime blk: {
        const ok_l = layoutOf(Ok, arch);
        const err_l = layoutOf(Err, arch);
        const max_payload_size = @max(ok_l.size, err_l.size);
        const max_payload_align = @max(ok_l.align_, err_l.align_);
        const max_align = @max(@as(u32, 1), max_payload_align);
        const payload_offset = alignForward(1, max_payload_align);
        const total_size = alignForward(payload_offset + max_payload_size, max_align);

        const max_flat_len = @max(ok_l.flat_types.len, err_l.flat_types.len);
        var joined_payload: [max_flat_len]FlatValType = undefined;
        for (0..max_flat_len) |slot| {
            const has_ok = slot < ok_l.flat_types.len;
            const has_err = slot < err_l.flat_types.len;
            if (has_ok and has_err) {
                joined_payload[slot] = joinFlatType(ok_l.flat_types[slot], err_l.flat_types[slot]);
            } else if (has_ok) {
                joined_payload[slot] = ok_l.flat_types[slot];
            } else {
                joined_payload[slot] = err_l.flat_types[slot];
            }
        }
        const joined_const = joined_payload;
        const flat = &[_]FlatValType{.i32} ++ joined_const[0..max_flat_len];
        break :blk .{
            .size = total_size,
            .align_ = max_align,
            .flat_types = flat,
            .needs_realloc = ok_l.needs_realloc or err_l.needs_realloc,
        };
    };
}

/// Comptime helper returning the byte offset of field `field_idx` inside a
/// non-packed struct `T` under the Canonical ABI.
pub fn fieldOffset(comptime T: type, comptime field_idx: usize, comptime arch: Arch) usize {
    return comptime blk: {
        const field_types = @typeInfo(T).@"struct".field_types;
        var offset: u64 = 0;
        for (0..field_idx) |i| {
            const fl = layoutOf(field_types[i], arch);
            offset = alignForward(offset, fl.align_);
            offset += fl.size;
        }
        const target_fl = layoutOf(field_types[field_idx], arch);
        offset = alignForward(offset, target_fl.align_);
        break :blk @intCast(offset);
    };
}

/// Comptime helper returning the payload byte offset for an optional `?Payload`.
pub fn optionPayloadOffset(comptime Payload: type, comptime arch: Arch) usize {
    return comptime blk: {
        const pl = layoutOf(Payload, arch);
        break :blk @intCast(alignForward(1, pl.align_));
    };
}

/// Comptime helper returning the payload byte offset for a variant or result.
pub fn variantPayloadOffset(comptime T: type, comptime arch: Arch) usize {
    return comptime blk: {
        if (isWitResult(T)) {
            const ok_l = layoutOf(T.OkType, arch);
            const err_l = layoutOf(T.ErrType, arch);
            const max_align = @max(ok_l.align_, err_l.align_);
            break :blk @intCast(alignForward(1, max_align));
        }
        switch (@typeInfo(T)) {
            .error_union => |eu| {
                const ok_l = layoutOf(eu.payload, arch);
                const err_l = layoutOf(eu.error_set, arch);
                const max_align = @max(ok_l.align_, err_l.align_);
                break :blk @intCast(alignForward(1, max_align));
            },
            .@"union" => |un| {
                const disc_size = variantDiscSize(un.field_types.len);
                var max_align: u32 = 1;
                for (un.field_types) |FType| {
                    const cl = layoutOf(FType, arch);
                    if (cl.align_ > max_align) max_align = cl.align_;
                }
                break :blk @intCast(alignForward(disc_size, max_align));
            },
            else => unreachable,
        }
    };
}

// ============================================================================
// Section 4: MemoryContext (Bounds-Checked Linear Memory + cabi_realloc)
// ============================================================================

/// Callback signature for `cabi_realloc(ctx, old_ptr, old_size, align, new_size) -> new_ptr`.
pub const ReallocFn = *const fn (
    ctx: ?*anyopaque,
    old_ptr: u64,
    old_size: u64,
    alignment: u32,
    new_size: u64,
) Error!u64;

/// Callback signature for `cabi_post_<func>(ctx, retptr_or_flat)`.
pub const PostReturnFn = *const fn (ctx: ?*anyopaque, arg: u64) void;

/// Callback signature for driving an async subtask to completion.
pub const AwaitSubtaskFn = *const fn (ctx: ?*anyopaque, subtask_handle: u32) Error!void;

/// Callback signature for polling or driving an async lifted task that returned
/// `BLOCKED` (`status != 2`) before calling `task.return`.
pub const PollTaskFn = *const fn (ctx: ?*anyopaque, callback_code: u32) Error!u32;

/// Runtime context for Canonical ABI linear-memory operations.
pub const MemoryContext = struct {
    /// Linear memory slice (may be backed by a Wasm instance's memory or a test buffer).
    memory: []u8,
    /// Optional base address offset if `memory` is a window starting at `base_addr`.
    base_addr: usize = 0,
    /// Opaque user context passed to `realloc_fn`.
    realloc_ctx: ?*anyopaque = null,
    /// Guest `cabi_realloc` callback used when lowering variable-length data into Wasm memory.
    realloc_fn: ?ReallocFn = null,
    /// Opaque user context passed to `post_return_fn`.
    post_return_ctx: ?*anyopaque = null,
    /// Optional `cabi_post_*` cleanup callback invoked after lifting results.
    post_return_fn: ?PostReturnFn = null,
    /// Opaque user context passed to `task_return_fn` (WASI 0.3 `task.return` intrinsic).
    task_return_ctx: ?*anyopaque = null,
    /// Optional WASI 0.3 `task.return` callback invoked by async lifted exports.
    task_return_fn: ?PostReturnFn = null,
    /// Opaque user context passed to `await_subtask_fn`.
    await_subtask_ctx: ?*anyopaque = null,
    /// Optional WASI 0.3 async subtask driver callback.
    await_subtask_fn: ?AwaitSubtaskFn = null,
    /// Opaque user context passed to `poll_task_fn`.
    poll_task_ctx: ?*anyopaque = null,
    /// Optional WASI 0.3 async lifted task poll/callback driver.
    poll_task_fn: ?PollTaskFn = null,
    /// Host allocator used when lifting non-blittable slices or allocating copies.
    allocator: ?std.mem.Allocator = null,
    /// When true, lifting `[]const u8` or blittable `[]const T` returns a zero-copy slice
    /// directly into `memory` instead of copying into `allocator`.
    zero_copy_slices: bool = true,
    /// When true, validate UTF-8 on `WitString` (and `[]const u8` when `strict_utf8_on_u8_slices` is set).
    validate_utf8: bool = true,
    /// When true, also enforce UTF-8 validation on bare `[]const u8` slices.
    strict_utf8_on_u8_slices: bool = false,
    /// Optional pointer to the owning WAMR `ComponentInstance`.
    instance: ?*comp_instance.ComponentInstance = null,
    /// Optional counter incremented on every `realloc` invocation.
    realloc_counter: ?*usize = null,

    /// Validates that `[addr, addr + byte_len)` lies within `self.memory`,
    /// safely rejecting both 32-bit and 64-bit arithmetic overflow.
    /// Returns the validated byte offset into `self.memory`.
    pub fn validateRange(self: MemoryContext, addr: u64, byte_len: u64) Error!usize {
        const base_u64: u64 = @intCast(self.base_addr);
        if (addr < base_u64) return error.OutOfBounds;
        const off_u64 = addr - base_u64;
        const end_u64 = std.math.add(u64, off_u64, byte_len) catch return error.OutOfBounds;
        if (end_u64 > @as(u64, @intCast(self.memory.len))) return error.OutOfBounds;
        const off_usize = std.math.cast(usize, off_u64) orelse return error.OutOfBounds;
        _ = std.math.cast(usize, end_u64) orelse return error.OutOfBounds;
        return off_usize;
    }

    /// Translates a linear-memory address to an offset into `self.memory` and
    /// validates that `[addr, addr + byte_len)` is in bounds.
    pub fn addrToOffset(self: MemoryContext, addr: usize, byte_len: usize) Error!usize {
        return self.validateRange(@intCast(addr), @intCast(byte_len));
    }

    /// Allocates `new_size` bytes with `alignment` in linear memory via `realloc_fn`,
    /// validating that the returned pointer is aligned and in bounds.
    pub fn realloc(self: *MemoryContext, old_ptr: u64, old_size: u64, alignment: u32, new_size: u64) Error!usize {
        const realloc_fn = self.realloc_fn orelse return error.NullRealloc;
        if (self.realloc_counter) |cnt| {
            _ = @atomicRmw(usize, cnt, .Add, 1, .monotonic);
        }
        const raw_ptr = try realloc_fn(self.realloc_ctx, old_ptr, old_size, alignment, new_size);
        if (self.realloc_fn == &instanceReallocCallback) {
            if (self.instance) |ci| {
                const refreshed = instanceMemorySlice(ci);
                if (refreshed.len > 0) self.memory = refreshed;
            }
        }
        if (alignment > 1 and (raw_ptr % alignment) != 0) return error.UnalignedPointer;
        const allow_zero_ptr = self.instance != null and self.instance.?.test_mem != null;
        if (new_size > 0 and raw_ptr == 0 and !allow_zero_ptr) return error.ReallocFailed;
        _ = try self.validateRange(raw_ptr, new_size);
        return std.math.cast(usize, raw_ptr) orelse return error.OutOfBounds;
    }

    pub fn readBytes(self: MemoryContext, addr: u64, byte_len: u64, alignment: u32) Error![]const u8 {
        if (alignment > 1 and (addr % alignment) != 0) return error.UnalignedPointer;
        const off = try self.validateRange(addr, byte_len);
        const len_usize = std.math.cast(usize, byte_len) orelse return error.OutOfBounds;
        return self.memory[off .. off + len_usize];
    }

    pub fn writeBytes(self: MemoryContext, addr: u64, data: []const u8, alignment: u32) Error!void {
        if (alignment > 1 and (addr % alignment) != 0) return error.UnalignedPointer;
        const off = try self.validateRange(addr, @intCast(data.len));
        @memcpy(self.memory[off .. off + data.len], data);
    }

    pub fn readInt(self: MemoryContext, comptime IntT: type, addr: u64, alignment: u32) Error!IntT {
        const size = @sizeOf(IntT);
        const bytes = try self.readBytes(addr, size, alignment);
        return std.mem.readInt(IntT, bytes[0..size], .little);
    }

    pub fn writeInt(self: MemoryContext, comptime IntT: type, addr: u64, val: IntT, alignment: u32) Error!void {
        const size = @sizeOf(IntT);
        if (alignment > 1 and (addr % alignment) != 0) return error.UnalignedPointer;
        const off = try self.validateRange(addr, size);
        std.mem.writeInt(IntT, self.memory[off..][0..size], val, .little);
    }
};

/// Simple bump-allocator helper for tests and scratch linear memory regions.
pub const BumpAllocator = struct {
    next_offset: usize = 16,
    limit: usize,
    call_count: usize = 0,

    pub fn init(start_offset: usize, limit: usize) BumpAllocator {
        return .{ .next_offset = @max(start_offset, 8), .limit = limit };
    }

    pub fn reallocCallback(
        ctx_Raw: ?*anyopaque,
        old_ptr: u64,
        old_size: u64,
        alignment: u32,
        new_size: u64,
    ) Error!u64 {
        _ = old_ptr;
        _ = old_size;
        const self: *BumpAllocator = @ptrCast(@alignCast(ctx_Raw.?));
        self.call_count += 1;
        const aligned = alignForward(@intCast(self.next_offset), alignment);
        const end = std.math.add(u64, aligned, new_size) catch return error.ReallocFailed;
        if (end > self.limit) return error.ReallocFailed;
        self.next_offset = @intCast(end);
        return aligned;
    }

    pub fn memoryContext(self: *BumpAllocator, memory: []u8, allocator: ?std.mem.Allocator) MemoryContext {
        return .{
            .memory = memory,
            .realloc_ctx = self,
            .realloc_fn = &BumpAllocator.reallocCallback,
            .allocator = allocator,
        };
    }
};

// ============================================================================
// Section 5: Linear Memory Load / Store (`storeToMemory` & `loadFromMemory`)
// ============================================================================

fn validateStringBytes(ctx: *const MemoryContext, bytes: []const u8, comptime is_explicit_string: bool) Error!void {
    if (!ctx.validate_utf8) return;
    if (is_explicit_string or ctx.strict_utf8_on_u8_slices) {
        if (!std.unicode.utf8ValidateSlice(bytes)) {
            return error.InvalidUtf8;
        }
    }
}

fn storeSliceData(
    comptime Child: type,
    comptime arch: Arch,
    ctx: *MemoryContext,
    slice: []const Child,
) Error!struct { ptr: u64, len: u64 } {
    const elem_layout = comptime layoutOf(Child, arch);
    const len: u64 = @intCast(slice.len);
    if (len == 0) {
        return .{ .ptr = elem_layout.align_, .len = 0 };
    }
    const total_bytes = std.math.mul(u64, len, elem_layout.size) catch return error.OutOfBounds;
    const raw_ptr = try ctx.realloc(0, 0, elem_layout.align_, total_bytes);

    if (comptime Child == u8) {
        try ctx.writeBytes(@intCast(raw_ptr), slice, 1);
    } else if (comptime isBlittable(Child, arch) and @sizeOf(Child) == elem_layout.size) {
        const byte_slice = std.mem.sliceAsBytes(slice);
        try ctx.writeBytes(@intCast(raw_ptr), byte_slice, elem_layout.align_);
    } else {
        for (slice, 0..) |item, i| {
            const item_off = raw_ptr + i * @as(usize, @intCast(elem_layout.size));
            try storeToMemory(Child, arch, ctx, item_off, item);
        }
    }
    return .{ .ptr = @intCast(raw_ptr), .len = len };
}

fn loadSliceFromPtrLen(
    comptime Child: type,
    comptime arch: Arch,
    ctx: *const MemoryContext,
    raw_ptr: u64,
    raw_len: u64,
    comptime is_explicit_string: bool,
) Error![]const Child {
    const elem_layout = comptime layoutOf(Child, arch);
    if (elem_layout.align_ > 1 and (raw_ptr % elem_layout.align_) != 0) {
        return error.UnalignedPointer;
    }
    if (raw_len == 0) {
        return &[_]Child{};
    }

    const total_bytes = std.math.mul(u64, raw_len, elem_layout.size) catch return error.OutOfBounds;
    const elem_off = try ctx.validateRange(raw_ptr, total_bytes);
    const len = std.math.cast(usize, raw_len) orelse return error.OutOfBounds;
    const total_bytes_usize = std.math.cast(usize, total_bytes) orelse return error.OutOfBounds;

    if (comptime Child == u8) {
        const bytes = try ctx.readBytes(raw_ptr, total_bytes, 1);
        try validateStringBytes(ctx, bytes, is_explicit_string);
        if (ctx.zero_copy_slices) {
            return bytes;
        }
        const alloc = ctx.allocator orelse return error.OutOfMemory;
        return alloc.dupe(u8, bytes) catch return error.OutOfMemory;
    }

    if (comptime isBlittable(Child, arch) and @sizeOf(Child) == elem_layout.size) {
        const bytes = try ctx.readBytes(raw_ptr, total_bytes, elem_layout.align_);
        if (ctx.zero_copy_slices) {
            const aligned_ptr: [*]const Child = @ptrCast(@alignCast(bytes.ptr));
            return aligned_ptr[0..len];
        }
        const alloc = ctx.allocator orelse return error.OutOfMemory;
        const out = alloc.alloc(Child, len) catch return error.OutOfMemory;
        @memcpy(std.mem.sliceAsBytes(out), ctx.memory[elem_off .. elem_off + total_bytes_usize]);
        return out;
    }

    const alloc = ctx.allocator orelse return error.OutOfMemory;
    const out = alloc.alloc(Child, len) catch return error.OutOfMemory;
    errdefer alloc.free(out);
    for (0..len) |i| {
        const item_addr = raw_ptr + @as(u64, @intCast(i)) * elem_layout.size;
        const item_off = std.math.cast(usize, item_addr) orelse return error.OutOfBounds;
        out[i] = try loadFromMemory(Child, arch, ctx, item_off);
    }
    return out;
}

fn loadSliceData(
    comptime Child: type,
    comptime arch: Arch,
    ctx: *const MemoryContext,
    base_offset: usize,
    comptime is_explicit_string: bool,
) Error![]const Child {
    const AddrT = arch.AddrInt();
    const ptr_align: u32 = if (arch == .wasm64) 8 else 4;
    const ptr_size: u64 = if (arch == .wasm64) 8 else 4;

    const base_u64: u64 = @intCast(base_offset);
    const len_addr = std.math.add(u64, base_u64, ptr_size) catch return error.OutOfBounds;
    const raw_ptr: u64 = @intCast(try ctx.readInt(AddrT, base_u64, ptr_align));
    const raw_len: u64 = @intCast(try ctx.readInt(AddrT, len_addr, ptr_align));

    return loadSliceFromPtrLen(Child, arch, ctx, raw_ptr, raw_len, is_explicit_string);
}

fn lowerResourceHandle(comptime T: type, ctx: *const MemoryContext, val: T) Error!u32 {
    if (comptime isResourceHandle(T) and resourceUsesAutoRep(T.ResourceType)) {
        if (ctx.instance) |ci| {
            if (T.handle_kind == .own) {
                const alloc = ctx.allocator orelse ci.allocator;
                const h = try T.newHandle(ci, val.idx, alloc);
                return h.idx;
            }
        }
    }
    return val.idx;
}

fn liftResourceHandle(comptime T: type, ctx: *const MemoryContext, raw_handle: u32) Error!T {
    if (comptime isResourceHandle(T) and resourceUsesAutoRep(T.ResourceType)) {
        if (ctx.instance) |ci| {
            const rt = ci.getOrCreateResourceTable(resourceTypeIdxFor(T.ResourceType)) catch return error.OutOfMemory;
            if (T.handle_kind == .own) {
                const alloc = ctx.allocator orelse ci.allocator;
                const rep = rt.drop(raw_handle, alloc) orelse return error.InvalidDiscriminant;
                return .{ .idx = rep };
            } else {
                const rep = rt.rep(raw_handle) orelse return error.InvalidDiscriminant;
                return .{ .idx = rep };
            }
        }
    }
    return .{ .idx = raw_handle };
}

/// Stores a Zig value `val` of type `T` at `base_offset` in `ctx.memory`
/// according to the Canonical ABI memory layout rules.
pub fn storeToMemory(
    comptime T: type,
    comptime arch: Arch,
    ctx: *MemoryContext,
    base_offset: usize,
    val: T,
) Error!void {
    const top_layout = comptime layoutOf(T, arch);
    const base_u64: u64 = @intCast(base_offset);
    if (top_layout.align_ > 1 and (base_u64 % top_layout.align_) != 0) {
        return error.UnalignedPointer;
    }
    _ = try ctx.validateRange(base_u64, top_layout.size);

    if (comptime T == void) return;

    if (comptime T == Char) {
        const valid = try validateChar(@backingInt(val));
        try ctx.writeInt(u32, base_u64, @backingInt(valid), 4);
        return;
    }

    if (comptime isWitHandle(T)) {
        const h = try lowerResourceHandle(T, ctx, val);
        try ctx.writeInt(u32, base_u64, h, 4);
        return;
    }

    if (comptime T == WitString) {
        try validateStringBytes(ctx, val.bytes, true);
        const AddrT = arch.AddrInt();
        const ptr_align: u32 = if (arch == .wasm64) 8 else 4;
        const ptr_size: u64 = if (arch == .wasm64) 8 else 4;
        const stored = try storeSliceData(u8, arch, ctx, val.bytes);
        try ctx.writeInt(AddrT, base_u64, @intCast(stored.ptr), ptr_align);
        try ctx.writeInt(AddrT, base_u64 + ptr_size, @intCast(stored.len), ptr_align);
        return;
    }

    if (comptime isGenericWitString(T)) {
        try validateStringBytes(ctx, val.slice, true);
        const AddrT = arch.AddrInt();
        const ptr_align: u32 = if (arch == .wasm64) 8 else 4;
        const ptr_size: u64 = if (arch == .wasm64) 8 else 4;
        const stored = try storeSliceData(u8, arch, ctx, val.slice);
        try ctx.writeInt(AddrT, base_u64, @intCast(stored.ptr), ptr_align);
        try ctx.writeInt(AddrT, base_u64 + ptr_size, @intCast(stored.len), ptr_align);
        return;
    }

    if (comptime isWitResult(T)) {
        const pay_off = base_offset + comptime variantPayloadOffset(T, arch);
        switch (val.payload) {
            .ok => |ok_val| {
                try ctx.writeInt(u8, base_u64, 0, 1);
                try storeToMemory(T.OkType, arch, ctx, pay_off, ok_val);
            },
            .err => |err_val| {
                try ctx.writeInt(u8, base_u64, 1, 1);
                try storeToMemory(T.ErrType, arch, ctx, pay_off, err_val);
            },
        }
        return;
    }

    switch (@typeInfo(T)) {
        .bool => try ctx.writeInt(u8, base_u64, @intFromBool(val), 1),
        .int => |info| switch (info.bits) {
            8, 16, 32, 64 => try ctx.writeInt(T, base_u64, val, @sizeOf(T)),
            else => unreachable,
        },
        .float => |info| switch (info.bits) {
            32 => try ctx.writeInt(u32, base_u64, @bitCast(val), 4),
            64 => try ctx.writeInt(u64, base_u64, @bitCast(val), 8),
            else => unreachable,
        },
        .pointer => |ptr| {
            if (comptime ptr.child == u8) {
                try validateStringBytes(ctx, val, false);
            }
            const AddrT = arch.AddrInt();
            const ptr_align: u32 = if (arch == .wasm64) 8 else 4;
            const ptr_size: u64 = if (arch == .wasm64) 8 else 4;
            const stored = try storeSliceData(ptr.child, arch, ctx, val);
            try ctx.writeInt(AddrT, base_u64, @intCast(stored.ptr), ptr_align);
            try ctx.writeInt(AddrT, base_u64 + ptr_size, @intCast(stored.len), ptr_align);
        },
        .array => |arr| {
            const elem_layout = comptime layoutOf(arr.child, arch);
            for (val, 0..) |item, i| {
                try storeToMemory(arr.child, arch, ctx, base_offset + i * @as(usize, @intCast(elem_layout.size)), item);
            }
        },
        .optional => |opt| {
            const pay_off = base_offset + comptime optionPayloadOffset(opt.child, arch);
            if (val) |some_val| {
                try ctx.writeInt(u8, base_u64, 1, 1);
                try storeToMemory(opt.child, arch, ctx, pay_off, some_val);
            } else {
                try ctx.writeInt(u8, base_u64, 0, 1);
            }
        },
        .error_union => |eu| {
            const pay_off = base_offset + comptime variantPayloadOffset(T, arch);
            if (val) |ok_val| {
                try ctx.writeInt(u8, base_u64, 0, 1);
                try storeToMemory(eu.payload, arch, ctx, pay_off, ok_val);
            } else |err_val| {
                try ctx.writeInt(u8, base_u64, 1, 1);
                try storeToMemory(eu.error_set, arch, ctx, pay_off, err_val);
            }
        },
        .error_set => |es| {
            const names = comptime errorSetNames(T, es);
            var code: u32 = 0;
            inline for (names, 0..) |name, idx| {
                if (val == @field(T, name)) {
                    code = @intCast(idx);
                }
            }
            try ctx.writeInt(u32, base_u64, code, 4);
        },
        .@"enum" => |en| {
            const disc_size = comptime variantDiscSize(en.field_names.len);
            const disc: u32 = @backingInt(val);
            if (disc >= en.field_names.len) return error.InvalidDiscriminant;
            switch (disc_size) {
                1 => try ctx.writeInt(u8, base_u64, @intCast(disc), 1),
                2 => try ctx.writeInt(u16, base_u64, @intCast(disc), 2),
                4 => try ctx.writeInt(u32, base_u64, disc, 4),
                else => unreachable,
            }
        },
        .@"struct" => |st| {
            if (st.layout == .@"packed") {
                const BackingInt = st.backing_integer.?;
                const bits: BackingInt = @bitCast(val);
                const layout = comptime layoutOf(T, arch);
                if (layout.size == 0) return;
                if (layout.size == 1) {
                    try ctx.writeInt(u8, base_u64, @intCast(bits), 1);
                } else if (layout.size == 2) {
                    try ctx.writeInt(u16, base_u64, @intCast(bits), 2);
                } else if (layout.size == 4) {
                    try ctx.writeInt(u32, base_u64, @intCast(bits), 4);
                } else {
                    const num_words = @as(usize, @intCast(layout.size / 4));
                    inline for (0..num_words) |w| {
                        const shift = @as(std.math.Log2Int(BackingInt), @intCast(w * 32));
                        const word: u32 = @truncate(bits >> shift);
                        try ctx.writeInt(u32, base_u64 + w * 4, word, 4);
                    }
                }
                return;
            }
            inline for (st.field_names, 0..) |fname, i| {
                const FType = st.field_types[i];
                const f_off = base_offset + comptime fieldOffset(T, i, arch);
                try storeToMemory(FType, arch, ctx, f_off, @field(val, fname));
            }
        },
        .@"union" => |un| {
            const disc_size = comptime variantDiscSize(un.field_types.len);
            const active_tag = std.meta.activeTag(val);
            const pay_off = base_offset + comptime variantPayloadOffset(T, arch);

            inline for (un.field_names, 0..) |fname, case_idx| {
                const FType = un.field_types[case_idx];
                if (active_tag == @field(un.tag_type.?, fname)) {
                    switch (disc_size) {
                        1 => try ctx.writeInt(u8, base_u64, @intCast(case_idx), 1),
                        2 => try ctx.writeInt(u16, base_u64, @intCast(case_idx), 2),
                        4 => try ctx.writeInt(u32, base_u64, @intCast(case_idx), 4),
                        else => unreachable,
                    }
                    try storeToMemory(FType, arch, ctx, pay_off, @field(val, fname));
                    return;
                }
            }
            return error.InvalidDiscriminant;
        },
        else => unreachable,
    }
}

fn ChildIsU8(comptime C: type) bool {
    return C == u8;
}

/// Loads a Zig value of type `T` from `base_offset` in `ctx.memory`
/// according to the Canonical ABI memory layout rules.
pub fn loadFromMemory(
    comptime T: type,
    comptime arch: Arch,
    ctx: *const MemoryContext,
    base_offset: usize,
) CanonResult(T) {
    const top_layout = comptime layoutOf(T, arch);
    const base_u64: u64 = @intCast(base_offset);
    if (top_layout.align_ > 1 and (base_u64 % top_layout.align_) != 0) {
        return error.UnalignedPointer;
    }
    _ = try ctx.validateRange(base_u64, top_layout.size);

    if (comptime T == void) return {};

    if (comptime T == Char) {
        const raw = try ctx.readInt(u32, base_u64, 4);
        return validateChar(raw);
    }

    if (comptime isWitHandle(T)) {
        const raw = try ctx.readInt(u32, base_u64, 4);
        return liftResourceHandle(T, ctx, raw);
    }

    if (comptime T == WitString) {
        const bytes = try loadSliceData(u8, arch, ctx, base_offset, true);
        return WitString{ .bytes = bytes };
    }

    if (comptime isGenericWitString(T)) {
        const bytes = try loadSliceData(u8, arch, ctx, base_offset, true);
        return T{ .slice = bytes };
    }

    if (comptime isWitResult(T)) {
        const disc = try ctx.readInt(u8, base_u64, 1);
        const pay_off = base_offset + comptime variantPayloadOffset(T, arch);
        return switch (disc) {
            0 => T{ .payload = .{ .ok = try loadFromMemory(T.OkType, arch, ctx, pay_off) } },
            1 => T{ .payload = .{ .err = try loadFromMemory(T.ErrType, arch, ctx, pay_off) } },
            else => error.InvalidDiscriminant,
        };
    }

    return switch (@typeInfo(T)) {
        .bool => blk: {
            const raw = try ctx.readInt(u8, base_u64, 1);
            break :blk raw != 0;
        },
        .int => |info| switch (info.bits) {
            8, 16, 32, 64 => try ctx.readInt(T, base_u64, @sizeOf(T)),
            else => unreachable,
        },
        .float => |info| switch (info.bits) {
            32 => @as(f32, @bitCast(try ctx.readInt(u32, base_u64, 4))),
            64 => @as(f64, @bitCast(try ctx.readInt(u64, base_u64, 8))),
            else => unreachable,
        },
        .pointer => |ptr| try loadSliceData(ptr.child, arch, ctx, base_offset, false),
        .array => |arr| blk: {
            var result: T = undefined;
            const elem_layout = comptime layoutOf(arr.child, arch);
            for (0..arr.len) |i| {
                result[i] = try loadFromMemory(arr.child, arch, ctx, base_offset + i * @as(usize, @intCast(elem_layout.size)));
            }
            break :blk result;
        },
        .optional => |opt| blk: {
            const disc = try ctx.readInt(u8, base_u64, 1);
            const pay_off = base_offset + comptime optionPayloadOffset(opt.child, arch);
            break :blk switch (disc) {
                0 => null,
                1 => try loadFromMemory(opt.child, arch, ctx, pay_off),
                else => error.InvalidDiscriminant,
            };
        },
        .error_union => |eu| blk: {
            const disc = try ctx.readInt(u8, base_u64, 1);
            const pay_off = base_offset + comptime variantPayloadOffset(T, arch);
            if (disc == 0) {
                break :blk try loadFromMemory(eu.payload, arch, ctx, pay_off);
            } else if (disc == 1) {
                const err_val = try loadFromMemory(eu.error_set, arch, ctx, pay_off);
                return err_val;
            } else {
                return error.InvalidDiscriminant;
            }
        },
        .error_set => |es| blk: {
            const disc = try ctx.readInt(u32, base_u64, 4);
            const names = comptime errorSetNames(T, es);
            inline for (names, 0..) |name, idx| {
                if (disc == idx) {
                    break :blk @field(T, name);
                }
            }
            return error.InvalidDiscriminant;
        },
        .@"enum" => |en| blk: {
            const disc_size = comptime variantDiscSize(en.field_names.len);
            const disc: u32 = switch (disc_size) {
                1 => try ctx.readInt(u8, base_u64, 1),
                2 => try ctx.readInt(u16, base_u64, 2),
                4 => try ctx.readInt(u32, base_u64, 4),
                else => unreachable,
            };
            if (disc >= en.field_names.len) return error.InvalidDiscriminant;
            break :blk @fromBackingInt(@intCast(disc));
        },
        .@"struct" => |st| blk: {
            if (st.layout == .@"packed") {
                const BackingInt = st.backing_integer.?;
                const layout = comptime layoutOf(T, arch);
                if (layout.size == 0) break :blk @as(T, @bitCast(@as(BackingInt, 0)));
                var bits: BackingInt = 0;
                if (layout.size == 1) {
                    bits = @truncate(try ctx.readInt(u8, base_u64, 1));
                } else if (layout.size == 2) {
                    bits = @truncate(try ctx.readInt(u16, base_u64, 2));
                } else if (layout.size == 4) {
                    bits = @truncate(try ctx.readInt(u32, base_u64, 4));
                } else {
                    const num_words = @as(usize, @intCast(layout.size / 4));
                    inline for (0..num_words) |w| {
                        const word = try ctx.readInt(u32, base_u64 + w * 4, 4);
                        const shift = @as(std.math.Log2Int(BackingInt), @intCast(w * 32));
                        bits |= @as(BackingInt, @intCast(word)) << shift;
                    }
                }
                break :blk @bitCast(bits);
            }
            var result: T = undefined;
            inline for (st.field_names, 0..) |fname, i| {
                const FType = st.field_types[i];
                const f_off = base_offset + comptime fieldOffset(T, i, arch);
                @field(result, fname) = try loadFromMemory(FType, arch, ctx, f_off);
            }
            break :blk result;
        },
        .@"union" => |un| blk: {
            const disc_size = comptime variantDiscSize(un.field_types.len);
            const disc: u32 = switch (disc_size) {
                1 => try ctx.readInt(u8, base_u64, 1),
                2 => try ctx.readInt(u16, base_u64, 2),
                4 => try ctx.readInt(u32, base_u64, 4),
                else => unreachable,
            };
            const pay_off = base_offset + comptime variantPayloadOffset(T, arch);
            inline for (un.field_names, 0..) |fname, case_idx| {
                const FType = un.field_types[case_idx];
                if (disc == case_idx) {
                    const payload = try loadFromMemory(FType, arch, ctx, pay_off);
                    break :blk @unionInit(T, fname, payload);
                }
            }
            return error.InvalidDiscriminant;
        },
        else => unreachable,
    };
}

/// Loads all fields of a struct/tuple `T` from `base_offset` in linear memory,
/// even when individual fields are error unions (`!Payload`).
pub fn loadStructFieldsFromMemory(
    comptime T: type,
    comptime arch: Arch,
    ctx: *const MemoryContext,
    base_offset: usize,
) Error!T {
    const top_layout = comptime layoutOf(T, arch);
    const base_u64: u64 = @intCast(base_offset);
    if (top_layout.align_ > 1 and (base_u64 % top_layout.align_) != 0) {
        return error.UnalignedPointer;
    }
    _ = try ctx.validateRange(base_u64, top_layout.size);

    const st = @typeInfo(T).@"struct";
    var result: T = undefined;
    inline for (st.field_names, 0..) |fname, i| {
        const FType = st.field_types[i];
        const f_off = base_offset + comptime fieldOffset(T, i, arch);
        if (@typeInfo(FType) == .error_union) {
            const eu = @typeInfo(FType).error_union;
            const disc = try ctx.readInt(u8, @intCast(f_off), 1);
            const pay_off = f_off + comptime variantPayloadOffset(FType, arch);
            if (disc == 0) {
                @field(result, fname) = try loadFromMemory(eu.payload, arch, ctx, pay_off);
            } else if (disc == 1) {
                const err_val = try loadFromMemory(eu.error_set, arch, ctx, pay_off);
                @field(result, fname) = err_val;
            } else {
                return error.InvalidDiscriminant;
            }
        } else {
            @field(result, fname) = try loadFromMemory(FType, arch, ctx, f_off);
        }
    }
    return result;
}

fn errorSetNames(comptime T: type, comptime es: std.builtin.Type.ErrorSet) []const [:0]const u8 {
    return es.error_names orelse @compileError("Inferred/anyerror sets cannot be lowered in Canonical ABI; use an explicit error set: " ++ @typeName(T));
}

// ============================================================================
// Section 6: Flat Register Lowering & Lifting (`lowerToFlat` & `liftFromFlat`)
// ============================================================================

/// Returns a Zig tuple type whose fields correspond to `flat_types`.
pub fn FlatTuple(comptime flat_types: []const FlatValType) type {
    var types: [flat_types.len]type = undefined;
    for (flat_types, 0..) |ft, i| {
        types[i] = ft.toZigType();
    }
    return @Tuple(&types);
}

/// Zero-initializes a flat slot of type `ft`.
pub fn zeroFlatVal(comptime ft: FlatValType) ft.toZigType() {
    return switch (ft) {
        .i32 => @as(i32, 0),
        .i64 => @as(i64, 0),
        .f32 => @as(f32, 0.0),
        .f64 => @as(f64, 0.0),
    };
}

/// Coerces a flat value of type `from` into a joined flat slot of type `into`
/// per the Canonical ABI `coerce` rules (`f32 -> i32`, `i32 -> i64`, `f32 -> i64`, `f64 -> i64`).
pub fn coerceToJoinedSlot(comptime from: FlatValType, comptime into: FlatValType, val: from.toZigType()) into.toZigType() {
    if (from == into) return val;
    if (from == .f32 and into == .i32) {
        return @bitCast(@as(u32, @bitCast(val)));
    }
    if (into == .i64) {
        return switch (from) {
            .i32 => @bitCast(@as(u64, @as(u32, @bitCast(val)))),
            .f32 => @bitCast(@as(u64, @as(u32, @bitCast(val)))),
            .f64 => @bitCast(@as(u64, @bitCast(val))),
            .i64 => val,
        };
    }
    unreachable;
}

/// Uncoerces a value from a joined flat slot of type `from_joined` back to `target`.
pub fn uncoerceFromJoinedSlot(comptime from_joined: FlatValType, comptime target: FlatValType, val: from_joined.toZigType()) target.toZigType() {
    if (from_joined == target) return val;
    if (from_joined == .i32 and target == .f32) {
        return @bitCast(@as(u32, @bitCast(val)));
    }
    if (from_joined == .i64) {
        const raw: u64 = @bitCast(val);
        return switch (target) {
            .i32 => @bitCast(@as(u32, @truncate(raw))),
            .f32 => @bitCast(@as(u32, @truncate(raw))),
            .f64 => @bitCast(raw),
            .i64 => val,
        };
    }
    unreachable;
}

/// Lowers a Zig value `val` of type `T` into its flat register tuple representation
/// (`FlatTuple(layoutOf(T, arch).flat_types)`).
pub fn lowerToFlat(
    comptime T: type,
    comptime arch: Arch,
    ctx: *MemoryContext,
    val: T,
) Error!FlatTuple(layoutOf(T, arch).flat_types) {
    const layout = comptime layoutOf(T, arch);
    const TupleT = FlatTuple(layout.flat_types);

    if (comptime T == void) return TupleT{};

    if (comptime T == Char) {
        const valid = try validateChar(@backingInt(val));
        return .{@as(i32, @bitCast(@backingInt(valid)))};
    }

    if (comptime isWitHandle(T)) {
        const h = try lowerResourceHandle(T, ctx, val);
        return .{@as(i32, @bitCast(h))};
    }

    if (comptime T == WitString) {
        try validateStringBytes(ctx, val.bytes, true);
        const FlatAddr = arch.FlatAddrType();
        const AddrU = arch.AddrInt();
        const stored = try storeSliceData(u8, arch, ctx, val.bytes);
        return .{
            @as(FlatAddr, @bitCast(@as(AddrU, @intCast(stored.ptr)))),
            @as(FlatAddr, @bitCast(@as(AddrU, @intCast(stored.len)))),
        };
    }

    if (comptime isGenericWitString(T)) {
        try validateStringBytes(ctx, val.slice, true);
        const FlatAddr = arch.FlatAddrType();
        const AddrU = arch.AddrInt();
        const stored = try storeSliceData(u8, arch, ctx, val.slice);
        return .{
            @as(FlatAddr, @bitCast(@as(AddrU, @intCast(stored.ptr)))),
            @as(FlatAddr, @bitCast(@as(AddrU, @intCast(stored.len)))),
        };
    }

    if (comptime isWitResult(T)) {
        return lowerVariantLikeResult(T.OkType, T.ErrType, arch, ctx, val.payload);
    }

    return switch (@typeInfo(T)) {
        .bool => .{@as(i32, @intFromBool(val))},
        .int => |info| switch (info.bits) {
            8 => if (info.signedness == .signed)
                .{@as(i32, val)}
            else
                .{@as(i32, @intCast(val))},
            16 => if (info.signedness == .signed)
                .{@as(i32, val)}
            else
                .{@as(i32, @intCast(val))},
            32 => .{@as(i32, @bitCast(val))},
            64 => .{@as(i64, @bitCast(val))},
            else => unreachable,
        },
        .float => |info| switch (info.bits) {
            32 => .{val},
            64 => .{val},
            else => unreachable,
        },
        .pointer => |ptr| blk: {
            if (comptime ptr.child == u8) {
                try validateStringBytes(ctx, val, false);
            }
            const FlatAddr = arch.FlatAddrType();
            const AddrU = arch.AddrInt();
            const stored = try storeSliceData(ptr.child, arch, ctx, val);
            break :blk .{
                @as(FlatAddr, @bitCast(@as(AddrU, @intCast(stored.ptr)))),
                @as(FlatAddr, @bitCast(@as(AddrU, @intCast(stored.len)))),
            };
        },
        .array => |arr| blk: {
            var out: TupleT = undefined;
            const elem_flat_len = comptime layoutOf(arr.child, arch).flat_types.len;
            inline for (0..arr.len) |i| {
                const sub = try lowerToFlat(arr.child, arch, ctx, val[i]);
                inline for (0..elem_flat_len) |j| {
                    out[i * elem_flat_len + j] = sub[j];
                }
            }
            break :blk out;
        },
        .optional => |opt| blk: {
            var out: TupleT = undefined;
            if (val) |some_val| {
                out[0] = @as(i32, 1);
                const sub = try lowerToFlat(opt.child, arch, ctx, some_val);
                inline for (0..sub.len) |j| {
                    out[1 + j] = sub[j];
                }
            } else {
                out[0] = @as(i32, 0);
                inline for (1..layout.flat_types.len) |j| {
                    out[j] = zeroFlatVal(layout.flat_types[j]);
                }
            }
            break :blk out;
        },
        .error_union => |eu| blk: {
            const PayloadUnion = Result(eu.payload, eu.error_set).Payload;
            const pu: PayloadUnion = if (val) |ok_v| .{ .ok = ok_v } else |err_v| .{ .err = err_v };
            break :blk try lowerVariantLikeResult(eu.payload, eu.error_set, arch, ctx, pu);
        },
        .error_set => |es| blk: {
            const names = comptime errorSetNames(T, es);
            var code: u32 = 0;
            inline for (names, 0..) |name, idx| {
                if (val == @field(T, name)) code = @intCast(idx);
            }
            break :blk .{@as(i32, @bitCast(code))};
        },
        .@"enum" => |en| blk: {
            const disc: u32 = @backingInt(val);
            if (disc >= en.field_names.len) return error.InvalidDiscriminant;
            break :blk .{@as(i32, @bitCast(disc))};
        },
        .@"struct" => |st| blk: {
            if (st.layout == .@"packed") {
                const BackingInt = st.backing_integer.?;
                const bits: BackingInt = @bitCast(val);
                var out: TupleT = undefined;
                if (layout.flat_types.len == 0) break :blk out;
                if (layout.flat_types.len == 1) {
                    out[0] = @as(i32, @bitCast(@as(u32, @intCast(bits))));
                } else {
                    inline for (0..layout.flat_types.len) |w| {
                        const shift = @as(std.math.Log2Int(BackingInt), @intCast(w * 32));
                        const word: u32 = @truncate(bits >> shift);
                        out[w] = @as(i32, @bitCast(word));
                    }
                }
                break :blk out;
            }
            var out: TupleT = undefined;
            comptime var slot_idx: usize = 0;
            inline for (st.field_names, 0..) |fname, i| {
                const FType = st.field_types[i];
                const fl = comptime layoutOf(FType, arch);
                const sub = try lowerToFlat(FType, arch, ctx, @field(val, fname));
                inline for (0..fl.flat_types.len) |j| {
                    out[slot_idx + j] = sub[j];
                }
                slot_idx += fl.flat_types.len;
            }
            break :blk out;
        },
        .@"union" => |un| blk: {
            var out: TupleT = undefined;
            const active_tag = std.meta.activeTag(val);
            inline for (un.field_names, 0..) |fname, case_idx| {
                const FType = un.field_types[case_idx];
                if (active_tag == @field(un.tag_type.?, fname)) {
                    out[0] = @as(i32, @intCast(case_idx));
                    const case_l = comptime layoutOf(FType, arch);
                    const sub = try lowerToFlat(FType, arch, ctx, @field(val, fname));
                    inline for (0..case_l.flat_types.len) |j| {
                        out[1 + j] = coerceToJoinedSlot(case_l.flat_types[j], layout.flat_types[1 + j], sub[j]);
                    }
                    inline for (1 + case_l.flat_types.len..layout.flat_types.len) |j| {
                        out[j] = zeroFlatVal(layout.flat_types[j]);
                    }
                    break :blk out;
                }
            }
            return error.InvalidDiscriminant;
        },
        else => unreachable,
    };
}

fn lowerVariantLikeResult(
    comptime Ok: type,
    comptime Err: type,
    comptime arch: Arch,
    ctx: *MemoryContext,
    payload: Result(Ok, Err).Payload,
) Error!FlatTuple(resultLayout(Ok, Err, arch).flat_types) {
    const layout = comptime resultLayout(Ok, Err, arch);
    const ok_l = comptime layoutOf(Ok, arch);
    const err_l = comptime layoutOf(Err, arch);
    var out: FlatTuple(layout.flat_types) = undefined;

    switch (payload) {
        .ok => |ok_val| {
            out[0] = @as(i32, 0);
            const sub = try lowerToFlat(Ok, arch, ctx, ok_val);
            inline for (0..ok_l.flat_types.len) |j| {
                out[1 + j] = coerceToJoinedSlot(ok_l.flat_types[j], layout.flat_types[1 + j], sub[j]);
            }
            inline for (1 + ok_l.flat_types.len..layout.flat_types.len) |j| {
                out[j] = zeroFlatVal(layout.flat_types[j]);
            }
        },
        .err => |err_val| {
            out[0] = @as(i32, 1);
            const sub = try lowerToFlat(Err, arch, ctx, err_val);
            inline for (0..err_l.flat_types.len) |j| {
                out[1 + j] = coerceToJoinedSlot(err_l.flat_types[j], layout.flat_types[1 + j], sub[j]);
            }
            inline for (1 + err_l.flat_types.len..layout.flat_types.len) |j| {
                out[j] = zeroFlatVal(layout.flat_types[j]);
            }
        },
    }
    return out;
}

/// Lifts a Zig value of type `T` from its flat register tuple representation
/// (`FlatTuple(layoutOf(T, arch).flat_types)`).
pub fn liftFromFlat(
    comptime T: type,
    comptime arch: Arch,
    ctx: *const MemoryContext,
    flat: FlatTuple(layoutOf(T, arch).flat_types),
) CanonResult(T) {
    const layout = comptime layoutOf(T, arch);

    if (comptime T == void) return {};

    if (comptime T == Char) {
        const raw: u32 = @bitCast(flat[0]);
        return validateChar(raw);
    }

    if (comptime isWitHandle(T)) {
        const raw: u32 = @bitCast(flat[0]);
        return liftResourceHandle(T, ctx, raw);
    }

    if (comptime T == WitString) {
        const AddrU = arch.AddrInt();
        const raw_ptr: u64 = @intCast(@as(AddrU, @bitCast(flat[0])));
        const raw_len: u64 = @intCast(@as(AddrU, @bitCast(flat[1])));
        const bytes = try loadSliceFromPtrLen(u8, arch, ctx, raw_ptr, raw_len, true);
        return WitString{ .bytes = bytes };
    }

    if (comptime isGenericWitString(T)) {
        const AddrU = arch.AddrInt();
        const raw_ptr: u64 = @intCast(@as(AddrU, @bitCast(flat[0])));
        const raw_len: u64 = @intCast(@as(AddrU, @bitCast(flat[1])));
        const bytes = try loadSliceFromPtrLen(u8, arch, ctx, raw_ptr, raw_len, true);
        return T{ .slice = bytes };
    }

    if (comptime isWitResult(T)) {
        const disc: u32 = @bitCast(flat[0]);
        const ok_l = comptime layoutOf(T.OkType, arch);
        const err_l = comptime layoutOf(T.ErrType, arch);
        if (disc == 0) {
            var ok_flat: FlatTuple(ok_l.flat_types) = undefined;
            inline for (0..ok_l.flat_types.len) |j| {
                ok_flat[j] = uncoerceFromJoinedSlot(layout.flat_types[1 + j], ok_l.flat_types[j], flat[1 + j]);
            }
            return T{ .payload = .{ .ok = try liftFromFlat(T.OkType, arch, ctx, ok_flat) } };
        } else if (disc == 1) {
            var err_flat: FlatTuple(err_l.flat_types) = undefined;
            inline for (0..err_l.flat_types.len) |j| {
                err_flat[j] = uncoerceFromJoinedSlot(layout.flat_types[1 + j], err_l.flat_types[j], flat[1 + j]);
            }
            return T{ .payload = .{ .err = try liftFromFlat(T.ErrType, arch, ctx, err_flat) } };
        } else {
            return error.InvalidDiscriminant;
        }
    }

    return switch (@typeInfo(T)) {
        .bool => flat[0] != 0,
        .int => |info| switch (info.bits) {
            8 => @bitCast(@as(u8, @truncate(@as(u32, @bitCast(flat[0]))))),
            16 => @bitCast(@as(u16, @truncate(@as(u32, @bitCast(flat[0]))))),
            32 => @bitCast(flat[0]),
            64 => @bitCast(flat[0]),
            else => unreachable,
        },
        .float => flat[0],
        .pointer => |ptr| blk: {
            const AddrU = arch.AddrInt();
            const raw_ptr: u64 = @intCast(@as(AddrU, @bitCast(flat[0])));
            const raw_len: u64 = @intCast(@as(AddrU, @bitCast(flat[1])));
            break :blk try loadSliceFromPtrLen(ptr.child, arch, ctx, raw_ptr, raw_len, false);
        },
        .array => |arr| blk: {
            var result: T = undefined;
            const elem_l = comptime layoutOf(arr.child, arch);
            const elem_flat_len = elem_l.flat_types.len;
            inline for (0..arr.len) |i| {
                var sub_flat: FlatTuple(elem_l.flat_types) = undefined;
                inline for (0..elem_flat_len) |j| {
                    sub_flat[j] = flat[i * elem_flat_len + j];
                }
                result[i] = try liftFromFlat(arr.child, arch, ctx, sub_flat);
            }
            break :blk result;
        },
        .optional => |opt| blk: {
            const disc: u32 = @bitCast(flat[0]);
            if (disc == 0) break :blk null;
            if (disc != 1) return error.InvalidDiscriminant;
            const pay_l = comptime layoutOf(opt.child, arch);
            var sub_flat: FlatTuple(pay_l.flat_types) = undefined;
            inline for (0..pay_l.flat_types.len) |j| {
                sub_flat[j] = flat[1 + j];
            }
            break :blk try liftFromFlat(opt.child, arch, ctx, sub_flat);
        },
        .error_union => |eu| blk: {
            const disc: u32 = @bitCast(flat[0]);
            const ok_l = comptime layoutOf(eu.payload, arch);
            const err_l = comptime layoutOf(eu.error_set, arch);
            if (disc == 0) {
                var ok_flat: FlatTuple(ok_l.flat_types) = undefined;
                inline for (0..ok_l.flat_types.len) |j| {
                    ok_flat[j] = uncoerceFromJoinedSlot(layout.flat_types[1 + j], ok_l.flat_types[j], flat[1 + j]);
                }
                break :blk try liftFromFlat(eu.payload, arch, ctx, ok_flat);
            } else if (disc == 1) {
                var err_flat: FlatTuple(err_l.flat_types) = undefined;
                inline for (0..err_l.flat_types.len) |j| {
                    err_flat[j] = uncoerceFromJoinedSlot(layout.flat_types[1 + j], err_l.flat_types[j], flat[1 + j]);
                }
                const err_val = try liftFromFlat(eu.error_set, arch, ctx, err_flat);
                return err_val;
            } else {
                return error.InvalidDiscriminant;
            }
        },
        .error_set => |es| blk: {
            const disc: u32 = @bitCast(flat[0]);
            const names = comptime errorSetNames(T, es);
            inline for (names, 0..) |name, idx| {
                if (disc == idx) break :blk @field(T, name);
            }
            return error.InvalidDiscriminant;
        },
        .@"enum" => |en| blk: {
            const disc: u32 = @bitCast(flat[0]);
            if (disc >= en.field_names.len) return error.InvalidDiscriminant;
            break :blk @fromBackingInt(@intCast(disc));
        },
        .@"struct" => |st| blk: {
            if (st.layout == .@"packed") {
                const BackingInt = st.backing_integer.?;
                if (layout.flat_types.len == 0) break :blk @as(T, @bitCast(@as(BackingInt, 0)));
                var bits: BackingInt = 0;
                if (layout.flat_types.len == 1) {
                    const word: u32 = @bitCast(flat[0]);
                    bits = @truncate(word);
                } else {
                    inline for (0..layout.flat_types.len) |w| {
                        const word: u32 = @bitCast(flat[w]);
                        const shift = @as(std.math.Log2Int(BackingInt), @intCast(w * 32));
                        bits |= @as(BackingInt, @intCast(word)) << shift;
                    }
                }
                break :blk @bitCast(bits);
            }
            var result: T = undefined;
            comptime var slot_idx: usize = 0;
            inline for (st.field_names, 0..) |fname, i| {
                const FType = st.field_types[i];
                const fl = comptime layoutOf(FType, arch);
                var sub_flat: FlatTuple(fl.flat_types) = undefined;
                inline for (0..fl.flat_types.len) |j| {
                    sub_flat[j] = flat[slot_idx + j];
                }
                @field(result, fname) = try liftFromFlat(FType, arch, ctx, sub_flat);
                slot_idx += fl.flat_types.len;
            }
            break :blk result;
        },
        .@"union" => |un| blk: {
            const disc: u32 = @bitCast(flat[0]);
            inline for (un.field_names, 0..) |fname, case_idx| {
                const FType = un.field_types[case_idx];
                if (disc == case_idx) {
                    const case_l = comptime layoutOf(FType, arch);
                    var sub_flat: FlatTuple(case_l.flat_types) = undefined;
                    inline for (0..case_l.flat_types.len) |j| {
                        sub_flat[j] = uncoerceFromJoinedSlot(layout.flat_types[1 + j], case_l.flat_types[j], flat[1 + j]);
                    }
                    const payload = try liftFromFlat(FType, arch, ctx, sub_flat);
                    break :blk @unionInit(T, fname, payload);
                }
            }
            return error.InvalidDiscriminant;
        },
        else => unreachable,
    };
}

// ============================================================================
// Section 7: Function Signature Classification (`CanonLift` & `CanonLower`)
// ============================================================================

/// Canonical ABI parameter/result register thresholds.
pub const MAX_FLAT_PARAMS: usize = 16;
pub const MAX_FLAT_RESULTS: usize = 1;
pub const MAX_FLAT_ASYNC_PARAMS: usize = 4;

pub const Direction = enum {
    /// Host calls a Wasm guest export (or Wasm calls a host import):
    /// When `results_flat > MAX_FLAT_RESULTS` and `direction == .lower`,
    /// a `retptr` parameter is appended to the core parameter list and core return is `void`.
    lower,
    /// Wasm guest export is lifted (or Wasm exports a function):
    /// When `results_flat > MAX_FLAT_RESULTS` and `direction == .lift`,
    /// the core function returns a single pointer (`i32`/`i64`) to its return area.
    lift,
};

pub const CallOptions = struct {
    arch: Arch = .wasm32,
    direction: Direction = .lower,
    is_async: bool = false,
};

fn normalizeCallOptions(comptime opts_or_mem64: anytype) CallOptions {
    const T = @TypeOf(opts_or_mem64);
    if (T == bool) {
        return .{ .arch = if (opts_or_mem64) .wasm64 else .wasm32 };
    }
    if (T == Arch) {
        return .{ .arch = opts_or_mem64 };
    }
    if (T == CallOptions) {
        return opts_or_mem64;
    }
    if (@typeInfo(T) == .@"struct") {
        var res = CallOptions{};
        if (@hasField(T, "arch")) res.arch = opts_or_mem64.arch;
        if (@hasField(T, "is_memory64")) res.arch = if (opts_or_mem64.is_memory64) .wasm64 else .wasm32;
        if (@hasField(T, "direction")) res.direction = opts_or_mem64.direction;
        if (@hasField(T, "is_async")) res.is_async = opts_or_mem64.is_async;
        return res;
    }
    @compileError("Unsupported options type for CanonLower/CanonLift: " ++ @typeName(T));
}

/// Comptime description of how a Zig function signature maps to a core Wasm
/// function signature under the Canonical ABI.
pub const FnClassification = struct {
    arch: Arch,
    direction: Direction,
    is_async: bool,
    /// Total flattened core values across all parameters before spilling.
    params_flat: usize,
    /// True when `params_flat > max_flat_params` (all parameters are spilled to a
    /// single indirect pointer parameter in linear memory).
    params_spill: bool,
    /// Total flattened core values for the return type before spilling.
    results_flat: usize,
    /// True when `results_flat > max_flat_results`.
    results_spill: bool,
    /// True if any parameter or spilled parameter block requires `cabi_realloc`.
    params_need_realloc: bool,
    /// Flattened core parameter types of the resulting core Wasm function
    /// (including trailing `retptr` if `results_spill and direction == .lower`).
    core_param_types: []const FlatValType,
    /// Flattened core result types of the resulting core Wasm function.
    core_result_types: []const FlatValType,
    /// Synthetic tuple struct representing all parameters laid out sequentially
    /// in linear memory (used when `params_spill == true`).
    ArgsStruct: type,
};

/// Inspects a Zig function type `FnType` at comptime and computes its Canonical
/// ABI core Wasm signature, spill behavior, and memory layout.
pub fn classifyZigFn(comptime FnType: type, comptime opts: CallOptions) FnClassification {
    return comptime blk: {
        @setEvalBranchQuota(100_000);
        const fn_info = @typeInfo(FnType).@"fn";
        const RawRet = fn_info.return_type orelse void;
        const RetType = switch (@typeInfo(RawRet)) {
            .error_union => |eu| eu.payload,
            else => RawRet,
        };
        const ptr_flat = opts.arch.flatAddrValType();

        var param_flat: []const FlatValType = &.{};
        var params_need_realloc = false;

        for (fn_info.param_types) |maybe_p| {
            const PType = maybe_p orelse @compileError("Generic/anytype parameters are not supported in Canonical ABI");
            const pl = layoutOf(PType, opts.arch);
            param_flat = param_flat ++ pl.flat_types;
            if (pl.needs_realloc) params_need_realloc = true;
        }

        const ArgsStruct = std.meta.ArgsTuple(FnType);

        const ret_layout = layoutOf(RetType, opts.arch);

        const max_params = if (opts.is_async) MAX_FLAT_ASYNC_PARAMS else MAX_FLAT_PARAMS;
        const params_spill = param_flat.len > max_params;
        const results_spill = if (opts.is_async)
            (opts.direction == .lower and ret_layout.flat_types.len > 0)
        else
            (ret_layout.flat_types.len > MAX_FLAT_RESULTS);

        var core_params: []const FlatValType = if (params_spill)
            &[_]FlatValType{ptr_flat}
        else
            param_flat;

        var core_results: []const FlatValType = &.{};
        if (opts.is_async) {
            if (opts.direction == .lower) {
                if (ret_layout.flat_types.len > 0) {
                    core_params = core_params ++ &[_]FlatValType{ptr_flat};
                }
                core_results = &[_]FlatValType{.i32};
            } else {
                core_results = &[_]FlatValType{.i32};
            }
        } else if (results_spill) {
            if (opts.direction == .lower) {
                core_params = core_params ++ &[_]FlatValType{ptr_flat};
                core_results = &.{};
            } else {
                core_results = &[_]FlatValType{ptr_flat};
            }
        } else {
            core_results = ret_layout.flat_types;
        }

        break :blk .{
            .arch = opts.arch,
            .direction = opts.direction,
            .is_async = opts.is_async,
            .params_flat = param_flat.len,
            .params_spill = params_spill,
            .results_flat = ret_layout.flat_types.len,
            .results_spill = results_spill,
            .params_need_realloc = params_need_realloc or params_spill,
            .core_param_types = core_params,
            .core_result_types = core_results,
            .ArgsStruct = ArgsStruct,
        };
    };
}

/// Returns the tuple of core Wasm parameter types for a classified Zig function.
pub fn CoreParamTuple(comptime FnType: type, comptime opts: CallOptions) type {
    const cls = classifyZigFn(FnType, opts);
    return FlatTuple(cls.core_param_types);
}

/// Returns the core Wasm return type (`void`, `i32`, `i64`, `f32`, or `f64`)
/// for a classified Zig function.
pub fn CoreReturnType(comptime FnType: type, comptime opts: CallOptions) type {
    const cls = classifyZigFn(FnType, opts);
    if (cls.core_result_types.len == 0) return void;
    return cls.core_result_types[0].toZigType();
}

fn makeFnTypeFromTypes(comptime param_types: []const type, comptime RetType: type) type {
    const P = param_types;
    return switch (P.len) {
        0 => fn () RetType,
        1 => fn (P[0]) RetType,
        2 => fn (P[0], P[1]) RetType,
        3 => fn (P[0], P[1], P[2]) RetType,
        4 => fn (P[0], P[1], P[2], P[3]) RetType,
        5 => fn (P[0], P[1], P[2], P[3], P[4]) RetType,
        6 => fn (P[0], P[1], P[2], P[3], P[4], P[5]) RetType,
        7 => fn (P[0], P[1], P[2], P[3], P[4], P[5], P[6]) RetType,
        8 => fn (P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7]) RetType,
        9 => fn (P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8]) RetType,
        10 => fn (P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9]) RetType,
        11 => fn (P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10]) RetType,
        12 => fn (P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11]) RetType,
        13 => fn (P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12]) RetType,
        14 => fn (P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12], P[13]) RetType,
        15 => fn (P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12], P[13], P[14]) RetType,
        16 => fn (P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12], P[13], P[14], P[15]) RetType,
        17 => fn (P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12], P[13], P[14], P[15], P[16]) RetType,
        18 => fn (P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12], P[13], P[14], P[15], P[16], P[17]) RetType,
        19 => fn (P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12], P[13], P[14], P[15], P[16], P[17], P[18]) RetType,
        20 => fn (P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12], P[13], P[14], P[15], P[16], P[17], P[18], P[19]) RetType,
        else => @compileError("Too many function parameters for makeFnTypeFromTypes"),
    };
}

/// Returns the Zig function pointer type for the underlying core Wasm function.
pub fn CoreFnType(comptime FnType: type, comptime opts: CallOptions) type {
    const cls = classifyZigFn(FnType, opts);
    var param_types: [cls.core_param_types.len]type = undefined;
    for (cls.core_param_types, 0..) |pt, i| {
        param_types[i] = pt.toZigType();
    }
    return makeFnTypeFromTypes(&param_types, CoreReturnType(FnType, opts));
}

/// Unwraps the return type of `FnType` (preserving payload if `FnType` returns an error union).
pub fn ReturnType(comptime FnType: type) type {
    const Ret = @typeInfo(FnType).@"fn".return_type orelse void;
    return switch (@typeInfo(Ret)) {
        .error_union => |eu| eu.payload,
        else => Ret,
    };
}

// ============================================================================
// Section 8: Comptime Call Adapters (`lowerCall`, `callLifted`, `liftCall`, `CanonLower`, `CanonLift`)
// ============================================================================

/// Lowers high-level Zig `args` according to `FnType`'s Canonical ABI signature,
/// invokes `core_fn` with the lowered core arguments, and lifts the result back
/// into a Zig value (calling `ctx.post_return_fn` if present).
pub fn lowerCall(
    comptime FnType: type,
    comptime opts: CallOptions,
    ctx: *MemoryContext,
    core_fn: *const CoreFnType(FnType, opts),
    args: std.meta.ArgsTuple(FnType),
) CanonResult(@typeInfo(FnType).@"fn".return_type orelse void) {
    const cls = comptime classifyZigFn(FnType, opts);
    const RetType = @typeInfo(FnType).@"fn".return_type orelse void;
    const FlatAddr = opts.arch.FlatAddrType();
    const AddrU = opts.arch.AddrInt();

    var core_args: CoreParamTuple(FnType, opts) = undefined;

    if (comptime cls.params_spill) {
        const args_layout = comptime layoutOf(cls.ArgsStruct, opts.arch);
        const raw_ptr = try ctx.realloc(0, 0, args_layout.align_, args_layout.size);
        try storeToMemory(cls.ArgsStruct, opts.arch, ctx, raw_ptr, args);
        core_args[0] = @as(FlatAddr, @bitCast(@as(AddrU, @intCast(raw_ptr))));
    } else {
        const fn_params = @typeInfo(FnType).@"fn".param_types;
        comptime var slot_idx: usize = 0;
        inline for (fn_params, 0..) |maybe_p, i| {
            const PType = maybe_p.?;
            const pl = comptime layoutOf(PType, opts.arch);
            const sub = try lowerToFlat(PType, opts.arch, ctx, args[i]);
            inline for (0..pl.flat_types.len) |j| {
                core_args[slot_idx + j] = sub[j];
            }
            slot_idx += pl.flat_types.len;
        }
    }

    if (comptime opts.is_async and opts.direction == .lower) {
        const ret_layout = comptime layoutOf(RetType, opts.arch);
        var ret_offset: usize = 0;
        if (comptime ret_layout.flat_types.len > 0) {
            ret_offset = if (ret_layout.size > 0)
                try ctx.realloc(0, 0, ret_layout.align_, ret_layout.size)
            else
                ret_layout.align_;
            core_args[cls.core_param_types.len - 1] = @as(FlatAddr, @bitCast(@as(AddrU, @intCast(ret_offset))));
        }

        const packed_status: u32 = @bitCast(@call(.auto, core_fn, core_args));
        const status_code: u4 = @truncate(packed_status);
        const subtask_handle: u32 = packed_status >> 4;
        if (status_code != 2) {
            if (ctx.await_subtask_fn) |await_fn| {
                try await_fn(ctx.await_subtask_ctx, subtask_handle);
            } else {
                return error.AsyncPending;
            }
        }
        if (comptime ret_layout.flat_types.len == 0) {
            return {};
        }
        return loadFromMemory(RetType, opts.arch, ctx, ret_offset);
    }

    if (comptime cls.results_spill) {
        const ret_layout = comptime layoutOf(RetType, opts.arch);
        if (comptime opts.direction == .lower) {
            const ret_offset = try ctx.realloc(0, 0, ret_layout.align_, ret_layout.size);
            core_args[cls.core_param_types.len - 1] = @as(FlatAddr, @bitCast(@as(AddrU, @intCast(ret_offset))));
            @call(.auto, core_fn, core_args);
            const lifted = loadFromMemory(RetType, opts.arch, ctx, ret_offset);
            if (ctx.post_return_fn) |post_fn| {
                post_fn(ctx.post_return_ctx, @intCast(ret_offset));
            }
            return lifted;
        } else {
            const ret_addr = @call(.auto, core_fn, core_args);
            const raw_addr: u64 = @intCast(@as(AddrU, @bitCast(ret_addr)));
            const ret_offset = try ctx.validateRange(raw_addr, ret_layout.size);
            const lifted = loadFromMemory(RetType, opts.arch, ctx, ret_offset);
            if (ctx.post_return_fn) |post_fn| {
                post_fn(ctx.post_return_ctx, raw_addr);
            }
            return lifted;
        }
    } else if (comptime cls.results_flat == 1) {
        const raw_ret = @call(.auto, core_fn, core_args);
        const lifted = liftFromFlat(RetType, opts.arch, ctx, .{raw_ret});
        if (ctx.post_return_fn) |post_fn| {
            const arg_u64: u64 = switch (cls.core_result_types[0]) {
                .i32 => @as(u32, @bitCast(raw_ret)),
                .i64 => @as(u64, @bitCast(raw_ret)),
                .f32 => @as(u32, @bitCast(raw_ret)),
                .f64 => @as(u64, @bitCast(raw_ret)),
            };
            post_fn(ctx.post_return_ctx, arg_u64);
        }
        return lifted;
    } else {
        @call(.auto, core_fn, core_args);
        if (ctx.post_return_fn) |post_fn| {
            post_fn(ctx.post_return_ctx, 0);
        }
        return {};
    }
}

/// Calls a core function exported via `canon lift` (where `results_flat > 1`
/// returns a single pointer from the core function rather than taking a caller
/// `retptr` parameter).
pub fn callLifted(
    comptime FnType: type,
    comptime opts_in: CallOptions,
    ctx: *MemoryContext,
    core_fn: *const CoreFnType(FnType, .{ .arch = opts_in.arch, .direction = .lift, .is_async = opts_in.is_async }),
    args: std.meta.ArgsTuple(FnType),
) CanonResult(@typeInfo(FnType).@"fn".return_type orelse void) {
    const lift_opts = CallOptions{ .arch = opts_in.arch, .direction = .lift, .is_async = opts_in.is_async };
    if (comptime !opts_in.is_async) {
        return lowerCall(FnType, lift_opts, ctx, core_fn, args);
    }

    const cls = comptime classifyZigFn(FnType, lift_opts);
    const RetType = @typeInfo(FnType).@"fn".return_type orelse void;
    const FlatAddr = opts_in.arch.FlatAddrType();
    const AddrU = opts_in.arch.AddrInt();

    var core_args: CoreParamTuple(FnType, lift_opts) = undefined;
    if (comptime cls.params_spill) {
        const args_layout = comptime layoutOf(cls.ArgsStruct, opts_in.arch);
        const raw_ptr = try ctx.realloc(0, 0, args_layout.align_, args_layout.size);
        try storeToMemory(cls.ArgsStruct, opts_in.arch, ctx, raw_ptr, args);
        core_args[0] = @as(FlatAddr, @bitCast(@as(AddrU, @intCast(raw_ptr))));
    } else {
        const fn_params = @typeInfo(FnType).@"fn".param_types;
        comptime var slot_idx: usize = 0;
        inline for (fn_params, 0..) |maybe_p, i| {
            const PType = maybe_p.?;
            const pl = comptime layoutOf(PType, opts_in.arch);
            const sub = try lowerToFlat(PType, opts_in.arch, ctx, args[i]);
            inline for (0..pl.flat_types.len) |j| {
                core_args[slot_idx + j] = sub[j];
            }
            slot_idx += pl.flat_types.len;
        }
    }

    const ReturnState = struct {
        called: bool = false,
        arg: u64 = 0,
        fn onTaskReturn(cb_ctx: ?*anyopaque, val: u64) void {
            const self: *@This() = @ptrCast(@alignCast(cb_ctx.?));
            self.called = true;
            self.arg = val;
        }
    };
    var ret_state = ReturnState{};
    const prev_tr_ctx = ctx.task_return_ctx;
    const prev_tr_fn = ctx.task_return_fn;
    ctx.task_return_ctx = &ret_state;
    ctx.task_return_fn = &ReturnState.onTaskReturn;
    defer {
        ctx.task_return_ctx = prev_tr_ctx;
        ctx.task_return_fn = prev_tr_fn;
    }

    var status: u32 = @bitCast(@call(.auto, core_fn, core_args));
    while (!ret_state.called) {
        if (ctx.poll_task_fn) |poll_fn| {
            status = try poll_fn(ctx.poll_task_ctx, status);
            if (status == 0 and !ret_state.called) return error.AsyncTaskDidNotReturn;
        } else {
            return error.AsyncPending;
        }
    }

    if (comptime RetType == void) return {};
    const ret_layout = comptime layoutOf(RetType, opts_in.arch);
    if (comptime ret_layout.flat_types.len == 1) {
        const ft = ret_layout.flat_types[0];
        const flat_val: ft.toZigType() = switch (ft) {
            .i32 => @bitCast(@as(u32, @truncate(ret_state.arg))),
            .i64 => @bitCast(ret_state.arg),
            .f32 => @bitCast(@as(u32, @truncate(ret_state.arg))),
            .f64 => @bitCast(ret_state.arg),
        };
        return liftFromFlat(RetType, opts_in.arch, ctx, .{flat_val});
    }
    const ret_offset = try ctx.validateRange(ret_state.arg, ret_layout.size);
    return loadFromMemory(RetType, opts_in.arch, ctx, ret_offset);
}

/// Lifts core Wasm arguments `core_args` into Zig arguments for `FnType`,
/// invokes `zig_fn`, and lowers the Zig return value back into core registers
/// or `retptr` memory.
pub fn liftCall(
    comptime FnType: type,
    comptime opts: CallOptions,
    ctx: *MemoryContext,
    zig_fn: *const FnType,
    core_args: CoreParamTuple(FnType, opts),
) Error!CoreReturnType(FnType, opts) {
    const cls = comptime classifyZigFn(FnType, opts);
    const RetType = @typeInfo(FnType).@"fn".return_type orelse void;
    const AddrU = opts.arch.AddrInt();
    const FlatAddr = opts.arch.FlatAddrType();

    var zig_args: std.meta.ArgsTuple(FnType) = undefined;

    if (comptime cls.params_spill) {
        const raw_ptr: usize = @intCast(@as(AddrU, @bitCast(core_args[0])));
        const args_layout = comptime layoutOf(cls.ArgsStruct, opts.arch);
        const offset = try ctx.validateRange(@intCast(raw_ptr), args_layout.size);
        zig_args = try loadStructFieldsFromMemory(cls.ArgsStruct, opts.arch, ctx, offset);
    } else {
        const fn_params = @typeInfo(FnType).@"fn".param_types;
        comptime var slot_idx: usize = 0;
        inline for (fn_params, 0..) |maybe_p, i| {
            const PType = maybe_p.?;
            const pl = comptime layoutOf(PType, opts.arch);
            var sub_flat: FlatTuple(pl.flat_types) = undefined;
            inline for (0..pl.flat_types.len) |j| {
                sub_flat[j] = core_args[slot_idx + j];
            }
            if (@typeInfo(PType) == .error_union) {
                if (liftFromFlat(PType, opts.arch, ctx, sub_flat)) |ok_v| {
                    zig_args[i] = ok_v;
                } else |err_v| {
                    const es = @typeInfo(PType).error_union.error_set;
                    const names = comptime errorSetNames(es, @typeInfo(es).error_set);
                    var matched = false;
                    inline for (names) |name| {
                        if (err_v == @field(es, name)) {
                            zig_args[i] = @field(es, name);
                            matched = true;
                        }
                    }
                    if (!matched) return @errorCast(err_v);
                }
            } else {
                zig_args[i] = try liftFromFlat(PType, opts.arch, ctx, sub_flat);
            }
            slot_idx += pl.flat_types.len;
        }
    }

    const zig_ret: RetType = @call(.auto, zig_fn, zig_args);

    if (comptime opts.is_async) {
        const ret_layout = comptime layoutOf(RetType, opts.arch);
        if (comptime opts.direction == .lower) {
            if (comptime ret_layout.flat_types.len > 0) {
                const ret_raw = core_args[cls.core_param_types.len - 1];
                const ret_addr: usize = @intCast(@as(AddrU, @bitCast(ret_raw)));
                const ret_offset = try ctx.validateRange(@intCast(ret_addr), ret_layout.size);
                try storeToMemory(RetType, opts.arch, ctx, ret_offset, zig_ret);
            }
            return @as(i32, 2); // STATUS_RETURNED = 2
        } else {
            var task_ret_arg: u64 = 0;
            if (comptime ret_layout.flat_types.len == 1) {
                const flat_res = try lowerToFlat(RetType, opts.arch, ctx, zig_ret);
                task_ret_arg = switch (ret_layout.flat_types[0]) {
                    .i32 => @as(u32, @bitCast(flat_res[0])),
                    .i64 => @as(u64, @bitCast(flat_res[0])),
                    .f32 => @as(u32, @bitCast(flat_res[0])),
                    .f64 => @as(u64, @bitCast(flat_res[0])),
                };
            } else if (comptime ret_layout.flat_types.len > 1) {
                const ret_offset = try ctx.realloc(0, 0, ret_layout.align_, ret_layout.size);
                try storeToMemory(RetType, opts.arch, ctx, ret_offset, zig_ret);
                task_ret_arg = @intCast(ret_offset);
            }
            if (ctx.task_return_fn) |tr_fn| {
                tr_fn(ctx.task_return_ctx, task_ret_arg);
            }
            return @as(i32, 0); // CALLBACK_EXIT = 0
        }
    }

    if (comptime cls.results_spill) {
        const ret_layout = comptime layoutOf(RetType, opts.arch);
        if (comptime opts.direction == .lower) {
            const ret_raw = core_args[cls.core_param_types.len - 1];
            const ret_addr: usize = @intCast(@as(AddrU, @bitCast(ret_raw)));
            const ret_offset = try ctx.validateRange(@intCast(ret_addr), ret_layout.size);
            try storeToMemory(RetType, opts.arch, ctx, ret_offset, zig_ret);
            return;
        } else {
            const ret_offset = try ctx.realloc(0, 0, ret_layout.align_, ret_layout.size);
            try storeToMemory(RetType, opts.arch, ctx, ret_offset, zig_ret);
            return @as(FlatAddr, @bitCast(@as(AddrU, @intCast(ret_offset))));
        }
    } else if (comptime cls.results_flat == 1) {
        const flat_res = try lowerToFlat(RetType, opts.arch, ctx, zig_ret);
        return flat_res[0];
    } else {
        return;
    }
}

/// Generates a comptime wrapper struct exposing `call` for lifting a Zig
/// handler function from core Wasm parameters.
pub fn createLiftThunk(
    comptime handler: anytype,
    comptime opts: CallOptions,
) type {
    const HandlerType = @TypeOf(handler);
    const FnType = switch (@typeInfo(HandlerType)) {
        .@"fn" => HandlerType,
        .pointer => |p| p.child,
        else => @compileError("createLiftThunk expects a function or function pointer"),
    };
    const fn_ptr: *const FnType = switch (@typeInfo(HandlerType)) {
        .@"fn" => &handler,
        .pointer => handler,
        else => unreachable,
    };

    return struct {
        pub const CoreParams = CoreParamTuple(FnType, opts);
        pub const CoreReturn = CoreReturnType(FnType, opts);

        pub fn call(ctx: *MemoryContext, core_args: CoreParams) Error!CoreReturn {
            return liftCall(FnType, opts, ctx, fn_ptr, core_args);
        }
    };
}

/// Comptime-computed signature metadata for `CanonLower` / `CanonLift`.
pub const CanonSig = struct {
    param_types: []const FlatValType,
    result_types: []const FlatValType,
    params_indirect: bool,
    results_indirect: bool,
    needs_realloc: bool,
};

/// High-level comptime adapter for `canon lower` (calling a core Wasm function
/// from Zig using high-level Zig types).
pub fn CanonLower(comptime FnType: type, comptime opts_or_mem64: anytype) type {
    const base_opts = normalizeCallOptions(opts_or_mem64);
    const opts = CallOptions{ .arch = base_opts.arch, .direction = .lower, .is_async = base_opts.is_async };
    const cls = classifyZigFn(FnType, opts);
    return struct {
        pub const classification = cls;
        pub const CoreParams = CoreParamTuple(FnType, opts);
        pub const CoreReturn = CoreReturnType(FnType, opts);
        pub const CoreFn = CoreFnType(FnType, opts);
        pub const ArgsTuple = std.meta.ArgsTuple(FnType);
        pub const ZigReturn = @typeInfo(FnType).@"fn".return_type orelse void;

        pub const sig = CanonSig{
            .param_types = cls.core_param_types,
            .result_types = cls.core_result_types,
            .params_indirect = cls.params_spill,
            .results_indirect = cls.results_spill,
            .needs_realloc = cls.params_need_realloc or cls.results_spill,
        };

        pub fn lowerArgs(ctx: *MemoryContext, args: ArgsTuple) Error!CoreParams {
            const FlatAddr = opts.arch.FlatAddrType();
            const AddrU = opts.arch.AddrInt();
            var core_args: CoreParams = undefined;

            if (comptime cls.params_spill) {
                const args_layout = comptime layoutOf(cls.ArgsStruct, opts.arch);
                const raw_ptr = try ctx.realloc(0, 0, args_layout.align_, args_layout.size);
                try storeToMemory(cls.ArgsStruct, opts.arch, ctx, raw_ptr, args);
                core_args[0] = @as(FlatAddr, @bitCast(@as(AddrU, @intCast(raw_ptr))));
            } else {
                const fn_params = @typeInfo(FnType).@"fn".param_types;
                comptime var slot_idx: usize = 0;
                inline for (fn_params, 0..) |maybe_p, i| {
                    const PType = maybe_p.?;
                    const pl = comptime layoutOf(PType, opts.arch);
                    const sub = try lowerToFlat(PType, opts.arch, ctx, args[i]);
                    inline for (0..pl.flat_types.len) |j| {
                        core_args[slot_idx + j] = sub[j];
                    }
                    slot_idx += pl.flat_types.len;
                }
            }

            if (comptime cls.results_spill) {
                const ret_layout = comptime layoutOf(ZigReturn, opts.arch);
                const ret_offset = try ctx.realloc(0, 0, ret_layout.align_, ret_layout.size);
                core_args[cls.core_param_types.len - 1] = @as(FlatAddr, @bitCast(@as(AddrU, @intCast(ret_offset))));
            }

            return core_args;
        }

        pub fn call(
            ctx: *MemoryContext,
            core_fn: *const CoreFn,
            args: ArgsTuple,
        ) CanonResult(ZigReturn) {
            return lowerCall(FnType, opts, ctx, core_fn, args);
        }
    };
}

/// High-level comptime adapter for `canon lift` (exposing a Zig function to be
/// called with core Wasm flat arguments).
pub fn CanonLift(comptime FnType: type, comptime opts_or_mem64: anytype) type {
    const base_opts = normalizeCallOptions(opts_or_mem64);
    const opts = CallOptions{ .arch = base_opts.arch, .direction = .lift, .is_async = base_opts.is_async };
    const lower_opts = CallOptions{ .arch = base_opts.arch, .direction = .lower, .is_async = base_opts.is_async };
    const cls = classifyZigFn(FnType, opts);
    return struct {
        pub const classification = cls;
        pub const CoreParams = CoreParamTuple(FnType, opts);
        pub const CoreReturn = CoreReturnType(FnType, opts);
        pub const CoreFn = CoreFnType(FnType, opts);
        pub const ArgsTuple = std.meta.ArgsTuple(FnType);
        pub const ZigReturn = @typeInfo(FnType).@"fn".return_type orelse void;

        pub const sig = CanonSig{
            .param_types = cls.core_param_types,
            .result_types = cls.core_result_types,
            .params_indirect = cls.params_spill,
            .results_indirect = cls.results_spill,
            .needs_realloc = cls.params_need_realloc or cls.results_spill,
        };

        pub fn liftArgs(ctx: *const MemoryContext, core_args: anytype) Error!ArgsTuple {
            const AddrU = opts.arch.AddrInt();
            var zig_args: ArgsTuple = undefined;

            if (comptime cls.params_spill) {
                const raw_ptr: usize = @intCast(@as(AddrU, @bitCast(core_args[0])));
                const args_layout = comptime layoutOf(cls.ArgsStruct, opts.arch);
                const offset = try ctx.validateRange(@intCast(raw_ptr), args_layout.size);
                zig_args = try loadStructFieldsFromMemory(cls.ArgsStruct, opts.arch, ctx, offset);
            } else {
                const fn_params = @typeInfo(FnType).@"fn".param_types;
                comptime var slot_idx: usize = 0;
                inline for (fn_params, 0..) |maybe_p, i| {
                    const PType = maybe_p.?;
                    const pl = comptime layoutOf(PType, opts.arch);
                    var sub_flat: FlatTuple(pl.flat_types) = undefined;
                    inline for (0..pl.flat_types.len) |j| {
                        sub_flat[j] = core_args[slot_idx + j];
                    }
                    if (@typeInfo(PType) == .error_union) {
                        if (liftFromFlat(PType, opts.arch, ctx, sub_flat)) |ok_v| {
                            zig_args[i] = ok_v;
                        } else |err_v| {
                            const es = @typeInfo(PType).error_union.error_set;
                            const names = comptime errorSetNames(es, @typeInfo(es).error_set);
                            var matched = false;
                            inline for (names) |name| {
                                if (err_v == @field(es, name)) {
                                    zig_args[i] = @field(es, name);
                                    matched = true;
                                }
                            }
                            if (!matched) return @errorCast(err_v);
                        }
                    } else {
                        zig_args[i] = try liftFromFlat(PType, opts.arch, ctx, sub_flat);
                    }
                    slot_idx += pl.flat_types.len;
                }
            }
            return zig_args;
        }

        pub fn call(
            ctx: *MemoryContext,
            zig_fn: *const FnType,
            core_args: CoreParams,
        ) Error!CoreReturn {
            return liftCall(FnType, opts, ctx, zig_fn, core_args);
        }

        pub fn callLowered(
            ctx: *MemoryContext,
            zig_fn: *const FnType,
            core_args: CoreParamTuple(FnType, lower_opts),
        ) Error!CoreReturnType(FnType, lower_opts) {
            return liftCall(FnType, lower_opts, ctx, zig_fn, core_args);
        }
    };
}

// ============================================================================
// Section 9: WAMR ComponentInstance & CallFrame Integration
// ============================================================================

/// Returns the active linear memory slice for `ci`, checking:
/// 1. `ci.test_mem` (if enabled for unit tests)
/// 2. `ci.activeCanonLowerCallCtx()` (pinned memory for an in-flight `canon.lower` call)
/// 3. `ci.canonicalMemory()` (first interpreter core memory)
/// 4. `ci.firstBackendMemory()` (first interpreter or AOT core memory)
pub fn instanceMemorySlice(ci: *comp_instance.ComponentInstance) []u8 {
    if (ci.test_mem) |tm| return tm.buffer;
    if (ci.activeCanonLowerCallCtx()) |cctx| {
        if (cctx.memory) |mem| return mem.data;
    }
    if (ci.canonicalMemory()) |mem| return mem.data;
    if (ci.firstBackendMemory()) |mem| return mem.data;
    return &.{};
}

fn instanceReallocCallback(
    ctx_opaque: ?*anyopaque,
    old_ptr: u64,
    old_size: u64,
    alignment: u32,
    new_size: u64,
) Error!u64 {
    _ = old_ptr;
    _ = old_size;
    const ci: *comp_instance.ComponentInstance = @ptrCast(@alignCast(ctx_opaque orelse return error.NullRealloc));
    const size_u32 = std.math.cast(u32, new_size) orelse return error.ReallocFailed;
    const ptr = ci.hostAllocGuest(size_u32, alignment) orelse return error.ReallocFailed;
    return @as(u64, ptr);
}

/// Builds a `MemoryContext` backed by `ci`'s active `CanonLowerCallCtx` memory,
/// `ci.canonicalMemory()`, `ci.firstBackendMemory()`, or `ci.test_mem`, with a
/// `realloc_fn` callback that invokes `ci.hostAllocGuest(@intCast(new_size), @intCast(alignment))`.
pub fn memoryContextForInstance(
    ci: *comp_instance.ComponentInstance,
    allocator: std.mem.Allocator,
) MemoryContext {
    return .{
        .memory = instanceMemorySlice(ci),
        .base_addr = 0,
        .realloc_ctx = ci,
        .realloc_fn = &instanceReallocCallback,
        .allocator = allocator,
        .zero_copy_slices = true,
        .validate_utf8 = true,
        .instance = ci,
    };
}

// ── Flat u64 slot helpers ────────────────────────────────────────────────────

pub fn rawSlotToFlatVal(comptime ft: FlatValType, raw: u64) ft.toZigType() {
    return switch (ft) {
        .i32 => @bitCast(@as(u32, @truncate(raw))),
        .i64 => @bitCast(raw),
        .f32 => @bitCast(@as(u32, @truncate(raw))),
        .f64 => @bitCast(raw),
    };
}

pub fn flatValToRawSlot(comptime ft: FlatValType, val: ft.toZigType()) u64 {
    return switch (ft) {
        .i32 => @as(u64, @as(u32, @bitCast(val))),
        .i64 => @bitCast(val),
        .f32 => @as(u64, @as(u32, @bitCast(val))),
        .f64 => @bitCast(val),
    };
}

pub fn coreValueToRawSlot(comptime ft: FlatValType, val: core_types.Value) u64 {
    return switch (ft) {
        .i32 => @as(u64, @as(u32, @bitCast(val.i32))),
        .i64 => @bitCast(val.i64),
        .f32 => @as(u64, @as(u32, @bitCast(val.f32))),
        .f64 => @bitCast(val.f64),
    };
}

pub fn flatValToCoreValue(comptime ft: FlatValType, val: ft.toZigType()) core_types.Value {
    return switch (ft) {
        .i32 => .{ .i32 = val },
        .i64 => .{ .i64 = val },
        .f32 => .{ .f32 = val },
        .f64 => .{ .f64 = val },
    };
}

// ── InterfaceValue <-> Zig conversion helpers for HostFunc.call ──────────────

pub fn interfaceValueToZig(
    comptime T: type,
    comptime arch: Arch,
    ctx: *const MemoryContext,
    iv: abi.InterfaceValue,
) Error!T {
    if (comptime T == void) return {};

    if (comptime T == Char) {
        const raw: u32 = switch (iv) {
            .char => |v| v,
            .u32 => |v| v,
            .s32 => |v| @bitCast(v),
            else => return error.InvalidChar,
        };
        return validateChar(raw);
    }

    if (comptime isWitHandle(T)) {
        const raw: u32 = switch (iv) {
            .handle => |v| v,
            .u32 => |v| v,
            .s32 => |v| @bitCast(v),
            else => return error.InvalidDiscriminant,
        };
        return liftResourceHandle(T, ctx, raw);
    }

    if (comptime T == WitString) {
        const pl: abi.InterfaceValue.PtrLen = switch (iv) {
            .string => |s| s,
            .list => |l| l,
            else => return error.InvalidDiscriminant,
        };
        const bytes = try loadSliceFromPtrLen(u8, arch, ctx, pl.ptr, pl.len, true);
        return WitString{ .bytes = bytes };
    }

    if (comptime isGenericWitString(T)) {
        const pl: abi.InterfaceValue.PtrLen = switch (iv) {
            .string => |s| s,
            .list => |l| l,
            else => return error.InvalidDiscriminant,
        };
        const bytes = try loadSliceFromPtrLen(u8, arch, ctx, pl.ptr, pl.len, true);
        return T{ .slice = bytes };
    }

    if (comptime isWitResult(T)) {
        switch (iv) {
            .result_val => |rv| {
                if (rv.is_ok) {
                    if (comptime T.OkType == void) return T{ .payload = .{ .ok = {} } };
                    const p = rv.payload orelse return error.InvalidDiscriminant;
                    return T{ .payload = .{ .ok = try interfaceValueToZig(T.OkType, arch, ctx, p.*) } };
                } else {
                    if (comptime T.ErrType == void) return T{ .payload = .{ .err = {} } };
                    const p = rv.payload orelse return error.InvalidDiscriminant;
                    return T{ .payload = .{ .err = try interfaceValueToZig(T.ErrType, arch, ctx, p.*) } };
                }
            },
            else => return error.InvalidDiscriminant,
        }
    }

    return switch (@typeInfo(T)) {
        .bool => switch (iv) {
            .bool => |b| b,
            .u32 => |u| u != 0,
            .s32 => |s| s != 0,
            else => error.InvalidBool,
        },
        .int => |info| switch (info.bits) {
            8 => if (info.signedness == .signed) switch (iv) {
                .s8 => |v| v,
                .u8 => |v| @bitCast(v),
                .s32 => |v| @truncate(v),
                .u32 => |v| @bitCast(@as(u8, @truncate(v))),
                else => error.InvalidDiscriminant,
            } else switch (iv) {
                .u8 => |v| v,
                .s8 => |v| @bitCast(v),
                .u32 => |v| @truncate(v),
                .s32 => |v| @truncate(@as(u32, @bitCast(v))),
                else => error.InvalidDiscriminant,
            },
            16 => if (info.signedness == .signed) switch (iv) {
                .s16 => |v| v,
                .u16 => |v| @bitCast(v),
                .s32 => |v| @truncate(v),
                .u32 => |v| @bitCast(@as(u16, @truncate(v))),
                else => error.InvalidDiscriminant,
            } else switch (iv) {
                .u16 => |v| v,
                .s16 => |v| @bitCast(v),
                .u32 => |v| @truncate(v),
                .s32 => |v| @truncate(@as(u32, @bitCast(v))),
                else => error.InvalidDiscriminant,
            },
            32 => if (info.signedness == .signed) switch (iv) {
                .s32 => |v| v,
                .u32 => |v| @bitCast(v),
                else => error.InvalidDiscriminant,
            } else switch (iv) {
                .u32 => |v| v,
                .s32 => |v| @bitCast(v),
                .char => |v| v,
                .handle => |v| v,
                else => error.InvalidDiscriminant,
            },
            64 => if (info.signedness == .signed) switch (iv) {
                .s64 => |v| v,
                .u64 => |v| @bitCast(v),
                else => error.InvalidDiscriminant,
            } else switch (iv) {
                .u64 => |v| v,
                .s64 => |v| @bitCast(v),
                else => error.InvalidDiscriminant,
            },
            else => unreachable,
        },
        .float => |info| switch (info.bits) {
            32 => switch (iv) {
                .f32 => |v| @bitCast(v),
                .u32 => |v| @bitCast(v),
                else => error.InvalidDiscriminant,
            },
            64 => switch (iv) {
                .f64 => |v| @bitCast(v),
                .u64 => |v| @bitCast(v),
                else => error.InvalidDiscriminant,
            },
            else => unreachable,
        },
        .pointer => |ptr| blk: {
            const pl: abi.InterfaceValue.PtrLen = switch (iv) {
                .string => |s| s,
                .list => |l| l,
                else => return error.InvalidDiscriminant,
            };
            const is_str = (comptime ptr.child == u8) and (iv == .string);
            break :blk try loadSliceFromPtrLen(ptr.child, arch, ctx, pl.ptr, pl.len, is_str);
        },
        .array => |arr| blk: {
            const items = switch (iv) {
                .tuple_val => |t| t,
                .record_val => |r| r,
                else => return error.InvalidDiscriminant,
            };
            if (items.len < arr.len) return error.InvalidDiscriminant;
            var out: T = undefined;
            for (0..arr.len) |i| {
                out[i] = try interfaceValueToZig(arr.child, arch, ctx, items[i]);
            }
            break :blk out;
        },
        .optional => |opt| switch (iv) {
            .option_val => |ov| blk: {
                if (!ov.is_some) break :blk null;
                if (comptime opt.child == void) break :blk {};
                const p = ov.payload orelse return error.InvalidDiscriminant;
                break :blk try interfaceValueToZig(opt.child, arch, ctx, p.*);
            },
            else => error.InvalidDiscriminant,
        },
        .@"enum" => |en| blk: {
            const disc: u32 = switch (iv) {
                .enum_val => |v| v,
                .u32 => |v| v,
                .s32 => |v| @bitCast(v),
                else => return error.InvalidDiscriminant,
            };
            if (disc >= en.field_names.len) return error.InvalidDiscriminant;
            break :blk @fromBackingInt(@intCast(disc));
        },
        .@"struct" => |st| blk: {
            if (st.layout == .@"packed") {
                const BackingInt = st.backing_integer.?;
                var bits: BackingInt = 0;
                switch (iv) {
                    .flags_val => |words| {
                        for (words, 0..) |w, i| {
                            if (i * 32 < @bitSizeOf(BackingInt)) {
                                const shift = @as(std.math.Log2Int(BackingInt), @intCast(i * 32));
                                bits |= @as(BackingInt, @intCast(w)) << shift;
                            }
                        }
                    },
                    .u32 => |w| bits = @truncate(w),
                    else => return error.InvalidDiscriminant,
                }
                break :blk @bitCast(bits);
            }
            const vals = switch (iv) {
                .record_val => |r| r,
                .tuple_val => |t| t,
                else => return error.InvalidDiscriminant,
            };
            if (vals.len < st.field_names.len) return error.InvalidDiscriminant;
            var out: T = undefined;
            inline for (st.field_names, 0..) |fname, i| {
                const FType = st.field_types[i];
                @field(out, fname) = try interfaceValueToZig(FType, arch, ctx, vals[i]);
            }
            break :blk out;
        },
        .@"union" => |un| switch (iv) {
            .variant_val => |vv| blk: {
                inline for (un.field_names, 0..) |fname, case_idx| {
                    const FType = un.field_types[case_idx];
                    if (vv.discriminant == case_idx) {
                        if (comptime FType == void) {
                            break :blk @unionInit(T, fname, {});
                        }
                        const p = vv.payload orelse return error.InvalidDiscriminant;
                        const pay = try interfaceValueToZig(FType, arch, ctx, p.*);
                        break :blk @unionInit(T, fname, pay);
                    }
                }
                return error.InvalidDiscriminant;
            },
            else => error.InvalidDiscriminant,
        },
        else => unreachable,
    };
}

pub fn zigToInterfaceValue(
    comptime T: type,
    comptime arch: Arch,
    ctx: *MemoryContext,
    val: T,
    allocator: std.mem.Allocator,
) Error!abi.InterfaceValue {
    if (comptime T == void) return .{ .u32 = 0 };

    if (comptime T == Char) {
        const valid = try validateChar(@backingInt(val));
        return .{ .char = @backingInt(valid) };
    }

    if (comptime isWitHandle(T)) {
        const h = try lowerResourceHandle(T, ctx, val);
        return .{ .handle = h };
    }

    if (comptime T == WitString) {
        try validateStringBytes(ctx, val.bytes, true);
        const stored = try storeSliceData(u8, arch, ctx, val.bytes);
        return .{ .string = .{
            .ptr = std.math.cast(u32, stored.ptr) orelse return error.OutOfBounds,
            .len = std.math.cast(u32, stored.len) orelse return error.OutOfBounds,
        } };
    }

    if (comptime isGenericWitString(T)) {
        try validateStringBytes(ctx, val.slice, true);
        const stored = try storeSliceData(u8, arch, ctx, val.slice);
        return .{ .string = .{
            .ptr = std.math.cast(u32, stored.ptr) orelse return error.OutOfBounds,
            .len = std.math.cast(u32, stored.len) orelse return error.OutOfBounds,
        } };
    }

    if (comptime isWitResult(T)) {
        switch (val.payload) {
            .ok => |ok_v| {
                var pay_ptr: ?*abi.InterfaceValue = null;
                if (comptime T.OkType != void) {
                    const p = allocator.create(abi.InterfaceValue) catch return error.OutOfMemory;
                    errdefer allocator.destroy(p);
                    p.* = try zigToInterfaceValue(T.OkType, arch, ctx, ok_v, allocator);
                    pay_ptr = p;
                }
                return .{ .result_val = .{ .is_ok = true, .payload = pay_ptr } };
            },
            .err => |err_v| {
                var pay_ptr: ?*abi.InterfaceValue = null;
                if (comptime T.ErrType != void) {
                    const p = allocator.create(abi.InterfaceValue) catch return error.OutOfMemory;
                    errdefer allocator.destroy(p);
                    p.* = try zigToInterfaceValue(T.ErrType, arch, ctx, err_v, allocator);
                    pay_ptr = p;
                }
                return .{ .result_val = .{ .is_ok = false, .payload = pay_ptr } };
            },
        }
    }

    return switch (@typeInfo(T)) {
        .bool => .{ .bool = val },
        .int => |info| switch (info.bits) {
            8 => if (info.signedness == .signed) .{ .s8 = val } else .{ .u8 = val },
            16 => if (info.signedness == .signed) .{ .s16 = val } else .{ .u16 = val },
            32 => if (info.signedness == .signed) .{ .s32 = val } else .{ .u32 = val },
            64 => if (info.signedness == .signed) .{ .s64 = val } else .{ .u64 = val },
            else => unreachable,
        },
        .float => |info| switch (info.bits) {
            32 => .{ .f32 = @bitCast(val) },
            64 => .{ .f64 = @bitCast(val) },
            else => unreachable,
        },
        .pointer => |ptr| blk: {
            if (comptime ptr.child == u8) {
                try validateStringBytes(ctx, val, false);
                const stored = try storeSliceData(u8, arch, ctx, val);
                break :blk .{ .string = .{
                    .ptr = std.math.cast(u32, stored.ptr) orelse return error.OutOfBounds,
                    .len = std.math.cast(u32, stored.len) orelse return error.OutOfBounds,
                } };
            }
            const stored = try storeSliceData(ptr.child, arch, ctx, val);
            break :blk .{ .list = .{
                .ptr = std.math.cast(u32, stored.ptr) orelse return error.OutOfBounds,
                .len = std.math.cast(u32, stored.len) orelse return error.OutOfBounds,
            } };
        },
        .array => |arr| blk: {
            const vals = allocator.alloc(abi.InterfaceValue, arr.len) catch return error.OutOfMemory;
            errdefer allocator.free(vals);
            for (val, 0..) |item, i| {
                vals[i] = try zigToInterfaceValue(arr.child, arch, ctx, item, allocator);
            }
            break :blk .{ .tuple_val = vals };
        },
        .optional => |opt| blk: {
            if (val) |some_v| {
                var pay_ptr: ?*abi.InterfaceValue = null;
                if (comptime opt.child != void) {
                    const p = allocator.create(abi.InterfaceValue) catch return error.OutOfMemory;
                    errdefer allocator.destroy(p);
                    p.* = try zigToInterfaceValue(opt.child, arch, ctx, some_v, allocator);
                    pay_ptr = p;
                }
                break :blk .{ .option_val = .{ .is_some = true, .payload = pay_ptr } };
            } else {
                break :blk .{ .option_val = .{ .is_some = false, .payload = null } };
            }
        },
        .@"enum" => .{ .enum_val = @backingInt(val) },
        .@"struct" => |st| blk: {
            if (st.layout == .@"packed") {
                const BackingInt = st.backing_integer.?;
                const bits: BackingInt = @bitCast(val);
                const layout = comptime layoutOf(T, arch);
                const num_words = @max(@as(usize, 1), layout.flat_types.len);
                const words = allocator.alloc(u32, num_words) catch return error.OutOfMemory;
                for (0..num_words) |w| {
                    if (w * 32 < @bitSizeOf(BackingInt)) {
                        const shift = @as(std.math.Log2Int(BackingInt), @intCast(w * 32));
                        words[w] = @truncate(bits >> shift);
                    } else {
                        words[w] = 0;
                    }
                }
                break :blk .{ .flags_val = words };
            }
            const vals = allocator.alloc(abi.InterfaceValue, st.field_names.len) catch return error.OutOfMemory;
            errdefer allocator.free(vals);
            inline for (st.field_names, 0..) |fname, i| {
                const FType = st.field_types[i];
                vals[i] = try zigToInterfaceValue(FType, arch, ctx, @field(val, fname), allocator);
            }
            if (st.is_tuple) {
                break :blk .{ .tuple_val = vals };
            }
            break :blk .{ .record_val = vals };
        },
        .@"union" => |un| blk: {
            const active_tag = std.meta.activeTag(val);
            inline for (un.field_names, 0..) |fname, case_idx| {
                const FType = un.field_types[case_idx];
                if (active_tag == @field(un.tag_type.?, fname)) {
                    var pay_ptr: ?*abi.InterfaceValue = null;
                    if (comptime FType != void) {
                        const p = allocator.create(abi.InterfaceValue) catch return error.OutOfMemory;
                        errdefer allocator.destroy(p);
                        p.* = try zigToInterfaceValue(FType, arch, ctx, @field(val, fname), allocator);
                        pay_ptr = p;
                    }
                    break :blk .{ .variant_val = .{ .discriminant = @intCast(case_idx), .payload = pay_ptr } };
                }
            }
            return error.InvalidDiscriminant;
        },
        else => unreachable,
    };
}

// ── Comptime Host Import Adapter (`makeHostFunc` / `makeHostFuncWithArch`) ───

const HandlerPrefixKind = enum {
    none,
    ci_only,
    ctx_only,
    ctx_and_ci,
};

fn isComponentInstancePtr(comptime T: type) bool {
    return T == *comp_instance.ComponentInstance or T == *const comp_instance.ComponentInstance;
}

fn isOpaqueCtxPtr(comptime T: type) bool {
    return T == ?*anyopaque or T == *anyopaque or T == ?*const anyopaque or T == *const anyopaque;
}

fn detectHandlerPrefix(comptime FnType: type) struct {
    prefix: HandlerPrefixKind,
    wit_start: usize,
    has_trailing_allocator: bool,
    wit_end: usize,
    SyntheticWitFn: type,
} {
    const fn_info = @typeInfo(FnType).@"fn";
    const params = fn_info.param_types;

    var prefix: HandlerPrefixKind = .none;
    var wit_start: usize = 0;
    if (params.len >= 2 and isOpaqueCtxPtr(params[0].?) and isComponentInstancePtr(params[1].?)) {
        prefix = .ctx_and_ci;
        wit_start = 2;
    } else if (params.len >= 1 and isComponentInstancePtr(params[0].?)) {
        prefix = .ci_only;
        wit_start = 1;
    } else if (params.len >= 1 and isOpaqueCtxPtr(params[0].?)) {
        prefix = .ctx_only;
        wit_start = 1;
    }

    const has_trailing_allocator = (params.len > wit_start and params[params.len - 1].? == std.mem.Allocator);
    const wit_end: usize = if (has_trailing_allocator) params.len - 1 else params.len;
    const wit_count = wit_end - wit_start;

    var wit_param_types: [wit_count]type = undefined;
    for (0..wit_count) |i| {
        wit_param_types[i] = params[wit_start + i].?;
    }

    const RawRet = fn_info.return_type orelse void;
    const WitRet = switch (@typeInfo(RawRet)) {
        .error_union => |eu| eu.payload,
        else => RawRet,
    };

    const SyntheticWitFn = makeFnTypeFromTypes(&wit_param_types, WitRet);

    return .{
        .prefix = prefix,
        .wit_start = wit_start,
        .has_trailing_allocator = has_trailing_allocator,
        .wit_end = wit_end,
        .SyntheticWitFn = SyntheticWitFn,
    };
}

/// Generates a comptime host function adapter struct for `handler` and `arch`.
/// Exposes:
/// - `call`: `HostFunc.call` implementation bridging `[]const abi.InterfaceValue`
/// - `callFlatSlots`: `HostFunc.call_flat` implementation operating directly on
///   flat `u64` core slots and linear memory without heap `InterfaceValue`s
/// - `hostFunc(ctx)`: constructs a `comp_instance.HostFunc`
pub fn HostFuncAdapter(comptime handler: anytype, comptime arch: Arch) type {
    const HandlerType = @TypeOf(handler);
    const HandlerFn = switch (@typeInfo(HandlerType)) {
        .@"fn" => HandlerType,
        .pointer => |p| p.child,
        else => @compileError("makeHostFunc requires a function or function pointer, got: " ++ @typeName(HandlerType)),
    };
    const fn_ptr: *const HandlerFn = switch (@typeInfo(HandlerType)) {
        .@"fn" => &handler,
        .pointer => handler,
        else => unreachable,
    };

    const meta = comptime detectHandlerPrefix(HandlerFn);
    const SyntheticWitFn = meta.SyntheticWitFn;
    const WitArgsTuple = std.meta.ArgsTuple(SyntheticWitFn);
    const RawRet = @typeInfo(HandlerFn).@"fn".return_type orelse void;
    const WitRet = ReturnType(HandlerFn);
    const lower_opts = CallOptions{ .arch = arch, .direction = .lower };
    const cls = comptime classifyZigFn(SyntheticWitFn, lower_opts);

    return struct {
        pub const classification = cls;

        fn invokeHandler(
            ctx: ?*anyopaque,
            ci: *comp_instance.ComponentInstance,
            wit_args: WitArgsTuple,
            allocator: std.mem.Allocator,
        ) anyerror!WitRet {
            var full_args: std.meta.ArgsTuple(HandlerFn) = undefined;
            switch (comptime meta.prefix) {
                .none => {},
                .ci_only => {
                    full_args[0] = ci;
                },
                .ctx_only => {
                    const P0 = @typeInfo(HandlerFn).@"fn".param_types[0].?;
                    if (@typeInfo(P0) == .optional) {
                        full_args[0] = ctx;
                    } else {
                        full_args[0] = ctx orelse return error.Trap;
                    }
                },
                .ctx_and_ci => {
                    const P0 = @typeInfo(HandlerFn).@"fn".param_types[0].?;
                    if (@typeInfo(P0) == .optional) {
                        full_args[0] = ctx;
                    } else {
                        full_args[0] = ctx orelse return error.Trap;
                    }
                    full_args[1] = ci;
                },
            }

            const wit_count = comptime (meta.wit_end - meta.wit_start);
            inline for (0..wit_count) |i| {
                full_args[meta.wit_start + i] = wit_args[i];
            }
            if (comptime meta.has_trailing_allocator) {
                full_args[meta.wit_end] = allocator;
            }

            if (comptime @typeInfo(RawRet) == .error_union) {
                return try @call(.auto, fn_ptr, full_args);
            } else {
                return @call(.auto, fn_ptr, full_args);
            }
        }

        /// Direct flat-slot comptime thunk (`HostFunc.call_flat`).
        /// Lifts arguments directly from `flat_args` (or spilled memory at `flat_args[0]`)
        /// and lowers results directly into `flat_results` or `retptr` in linear memory
        /// without allocating `InterfaceValue`s.
        pub fn callFlatSlots(
            ctx: ?*anyopaque,
            ci: *comp_instance.ComponentInstance,
            flat_args: []const u64,
            flat_results: []u64,
            retptr: u64,
            allocator: std.mem.Allocator,
        ) anyerror!void {
            var mem_ctx = memoryContextForInstance(ci, allocator);
            var wit_args: WitArgsTuple = undefined;

            const param_slot_count: usize = if (comptime cls.params_spill) 1 else cls.params_flat;
            if (flat_args.len < param_slot_count) return error.StackUnderflow;

            if (comptime cls.params_spill) {
                const AddrU = arch.AddrInt();
                const raw_ptr: u64 = @intCast(@as(AddrU, @truncate(flat_args[0])));
                const args_layout = comptime layoutOf(cls.ArgsStruct, arch);
                const offset = try mem_ctx.validateRange(raw_ptr, args_layout.size);
                wit_args = try loadStructFieldsFromMemory(cls.ArgsStruct, arch, &mem_ctx, offset);
            } else {
                const wit_params = @typeInfo(SyntheticWitFn).@"fn".param_types;
                comptime var slot_idx: usize = 0;
                inline for (wit_params, 0..) |maybe_p, i| {
                    const PType = maybe_p.?;
                    const pl = comptime layoutOf(PType, arch);
                    var sub_flat: FlatTuple(pl.flat_types) = undefined;
                    inline for (0..pl.flat_types.len) |j| {
                        sub_flat[j] = rawSlotToFlatVal(pl.flat_types[j], flat_args[slot_idx + j]);
                    }
                    wit_args[i] = try liftFromFlat(PType, arch, &mem_ctx, sub_flat);
                    slot_idx += pl.flat_types.len;
                }
            }

            const ret_val = try invokeHandler(ctx, ci, wit_args, allocator);
            mem_ctx.memory = instanceMemorySlice(ci);

            if (comptime WitRet == void) {
                return;
            } else if (comptime !cls.results_spill) {
                if (flat_results.len < 1) return error.StackOverflow;
                const flat_res = try lowerToFlat(WitRet, arch, &mem_ctx, ret_val);
                const ret_ft = comptime layoutOf(WitRet, arch).flat_types[0];
                flat_results[0] = flatValToRawSlot(ret_ft, flat_res[0]);
            } else {
                const has_retptr_arg = flat_args.len > param_slot_count;
                const effective_retptr: u64 = if (retptr != 0)
                    retptr
                else if (has_retptr_arg)
                    flat_args[param_slot_count]
                else
                    0;

                if (effective_retptr == 0 and !has_retptr_arg and flat_results.len >= cls.results_flat) {
                    const flat_res = try lowerToFlat(WitRet, arch, &mem_ctx, ret_val);
                    const ret_layout = comptime layoutOf(WitRet, arch);
                    inline for (0..ret_layout.flat_types.len) |j| {
                        flat_results[j] = flatValToRawSlot(ret_layout.flat_types[j], flat_res[j]);
                    }
                    return;
                }

                const ret_layout = comptime layoutOf(WitRet, arch);
                const AddrU = arch.AddrInt();
                const bounded_retptr: u64 = @intCast(@as(AddrU, @truncate(effective_retptr)));
                const ret_offset = try mem_ctx.validateRange(bounded_retptr, ret_layout.size);
                try storeToMemory(WitRet, arch, &mem_ctx, ret_offset, ret_val);
            }
        }

        /// `HostFunc.call` bridge converting `[]const abi.InterfaceValue` <-> typed Zig args/results.
        pub fn call(
            ctx: ?*anyopaque,
            ci: *comp_instance.ComponentInstance,
            args: []const abi.InterfaceValue,
            results: []abi.InterfaceValue,
            allocator: std.mem.Allocator,
        ) anyerror!void {
            var mem_ctx = memoryContextForInstance(ci, allocator);
            var wit_args: WitArgsTuple = undefined;

            const wit_params = @typeInfo(SyntheticWitFn).@"fn".param_types;
            if (args.len < wit_params.len) return error.StackUnderflow;

            inline for (wit_params, 0..) |maybe_p, i| {
                const PType = maybe_p.?;
                wit_args[i] = try interfaceValueToZig(PType, arch, &mem_ctx, args[i]);
            }

            const ret_val = try invokeHandler(ctx, ci, wit_args, allocator);
            mem_ctx.memory = instanceMemorySlice(ci);
            if (comptime WitRet != void) {
                if (results.len > 0) {
                    results[0] = try zigToInterfaceValue(WitRet, arch, &mem_ctx, ret_val, allocator);
                }
            }
        }

        pub fn hostFunc(context: ?*anyopaque) comp_instance.HostFunc {
            return .{
                .context = context,
                .call = &call,
                .call_flat = &callFlatSlots,
            };
        }
    };
}

/// Generates a `comp_instance.HostFunc` (with `.call` and `.call_flat` populated)
/// for a typed Zig host function `handler` under `wasm32`.
pub fn makeHostFunc(comptime handler: anytype) comp_instance.HostFunc {
    return HostFuncAdapter(handler, .wasm32).hostFunc(null);
}

/// Generates a `comp_instance.HostFunc` (with `.call`, `.call_flat`, and `.context`
/// populated) for a typed Zig host function `handler` under `wasm32`.
pub fn makeHostFuncWithCtx(comptime handler: anytype, context: ?*anyopaque) comp_instance.HostFunc {
    return HostFuncAdapter(handler, .wasm32).hostFunc(context);
}

/// Generates a `comp_instance.HostFunc` (with `.call` and `.call_flat` populated)
/// for a typed Zig host function `handler` under `arch` (`.wasm32` or `.wasm64`).
pub fn makeHostFuncWithArch(comptime handler: anytype, comptime arch: Arch) comp_instance.HostFunc {
    return HostFuncAdapter(handler, arch).hostFunc(null);
}

/// Generates a `comp_instance.HostFunc` (with `.call`, `.call_flat`, and `.context`
/// populated) for a typed Zig host function `handler` under `arch`.
pub fn makeHostFuncWithContext(
    comptime handler: anytype,
    comptime arch: Arch,
    context: ?*anyopaque,
) comp_instance.HostFunc {
    return HostFuncAdapter(handler, arch).hostFunc(context);
}

// ============================================================================
// Section 10: Comptime Guest Export Caller (`callTypedExport` / `callTypedExportByLocal`)
// ============================================================================

const FrameReallocCtx = struct {
    frame: *CallFrame,
    owner_inst: *comp_instance.ComponentInstance,
    realloc_idx: CoreFuncIdxLocal,
    mem_idx: u32,
    mem_ctx: *MemoryContext,
};

fn frameReallocCallback(
    ctx_opaque: ?*anyopaque,
    old_ptr: u64,
    old_size: u64,
    alignment: u32,
    new_size: u64,
) Error!u64 {
    const rctx: *FrameReallocCtx = @ptrCast(@alignCast(ctx_opaque orelse return error.NullRealloc));
    const old_ptr_u32 = std.math.cast(u32, old_ptr) orelse return error.ReallocFailed;
    const old_size_u32 = std.math.cast(u32, old_size) orelse return error.ReallocFailed;
    const new_size_u32 = std.math.cast(u32, new_size) orelse return error.ReallocFailed;
    const saved_may_leave = @atomicLoad(bool, &rctx.owner_inst.may_leave, .acquire);
    @atomicStore(bool, &rctx.owner_inst.may_leave, false, .release);
    defer @atomicStore(bool, &rctx.owner_inst.may_leave, saved_may_leave, .release);
    const ptr = rctx.frame.realloc(
        rctx.realloc_idx,
        old_ptr_u32,
        old_size_u32,
        alignment,
        new_size_u32,
    ) catch return error.ReallocFailed;
    if (rctx.frame.memory(rctx.mem_idx)) |mem| {
        rctx.mem_ctx.memory = mem;
    }
    return @as(u64, ptr);
}

fn coreFuncReturnsResultPtr(
    entry: comp_instance.ComponentInstance.CoreInstanceEntry,
    core_func_idx: u32,
) bool {
    if (entry.module_inst) |mi| {
        if (mi.module.getFuncType(core_func_idx)) |ft| {
            return ft.results.len == 1;
        }
    }
    if (entry.aot_inst) |ai| {
        if (core_func_idx < ai.module.func_type_indices.len) {
            const type_idx = ai.module.func_type_indices[core_func_idx];
            if (type_idx < ai.module.func_types.len) {
                return ai.module.func_types[type_idx].results.len == 1;
            }
        }
    }
    return true;
}

/// Calls a lifted guest export identified by `local` on `owner_inst` using
/// comptime-derived Canonical ABI lowering and lifting (`CanonLower`).
/// Works across both interpreter (`module_inst`) and AOT (`aot_inst`) core
/// instances via `CallFrame`.
pub fn callTypedExportByLocal(
    comptime FnType: type,
    comptime opts: anytype,
    owner_inst: *comp_instance.ComponentInstance,
    local: comp_instance.ComponentInstance.ExportedFunc.Local,
    args: CanonLower(FnType, opts).ArgsTuple,
    allocator: std.mem.Allocator,
) !CanonLower(FnType, opts).ZigReturn {
    const norm_opts = comptime normalizeCallOptions(opts);
    const arch = norm_opts.arch;
    const lower_opts = CallOptions{ .arch = arch, .direction = .lower, .is_async = norm_opts.is_async };
    const cls = comptime classifyZigFn(FnType, lower_opts);
    const ZigReturn = CanonLower(FnType, opts).ZigReturn;

    if (local.core_instance_idx >= owner_inst.core_instances.len) {
        return error.FunctionNotFound;
    }
    const core_entry = owner_inst.core_instances[local.core_instance_idx];

    var default_mem_idx: u32 = 0;
    var opt_realloc_idx: ?u32 = if (@hasField(@TypeOf(local), "realloc_core_func_idx"))
        local.realloc_core_func_idx
    else
        null;
    var opt_post_return_idx: ?u32 = if (@hasField(@TypeOf(local), "post_return_core_func_idx"))
        local.post_return_core_func_idx
    else
        null;
    for (local.opts) |opt| {
        switch (opt) {
            .memory => |m| default_mem_idx = m,
            .realloc => |r| if (opt_realloc_idx == null) {
                opt_realloc_idx = r;
            },
            .post_return => |p| if (opt_post_return_idx == null) {
                opt_post_return_idx = p;
            },
            .async_lift => if (!norm_opts.is_async) return error.AsyncLiftUnsupported,
            else => {},
        }
    }

    const resolved_realloc_idx: ?CoreFuncIdxLocal = if (opt_realloc_idx) |comp_idx| blk: {
        if (owner_inst.resolveTopLevelCoreFuncAny(comp_idx)) |r| {
            break :blk switch (r) {
                .interp => |ip| {
                    if (core_entry.module_inst != ip.mi) return error.ReallocNotAvailable;
                    break :blk ip.local_idx;
                },
                .aot => |ap| {
                    if (core_entry.aot_inst != ap.ai) return error.ReallocNotAvailable;
                    break :blk ap.local_idx;
                },
            };
        }
        break :blk CoreFuncIdxLocal.from(comp_idx);
    } else null;

    const resolved_post_return_idx: ?CoreFuncIdxLocal = if (opt_post_return_idx) |comp_idx| blk: {
        if (owner_inst.resolveTopLevelCoreFuncAny(comp_idx)) |r| {
            break :blk switch (r) {
                .interp => |ip| {
                    if (core_entry.module_inst != ip.mi) return error.ReallocNotAvailable;
                    break :blk ip.local_idx;
                },
                .aot => |ap| {
                    if (core_entry.aot_inst != ap.ai) return error.ReallocNotAvailable;
                    break :blk ap.local_idx;
                },
            };
        }
        break :blk CoreFuncIdxLocal.from(comp_idx);
    } else null;

    // Construct a CallFrame backed by either interpreter ExecEnv or AotInstance.
    var interp_env: ?*ExecEnv = null;
    defer if (interp_env) |env| env.destroy();

    var frame: CallFrame = blk: {
        if (core_entry.module_inst) |mi| {
            const env = ExecEnv.create(mi, 4096, allocator) catch return error.OutOfMemory;
            if (comptime config.lib_wasi_threads) {
                env.setThreadManager(&owner_inst.thread_manager);
            }
            interp_env = env;
            break :blk .{ .interp = InterpFrame.init(env) };
        }
        if (core_entry.aot_inst) |ai| {
            break :blk .{ .aot = AotFrame.init(ai, allocator) };
        }
        return error.FunctionNotFound;
    };
    defer frame.deinit();

    var active_ctx_scope = if (interp_env) |env|
        env.thread_context.enter()
    else if (core_entry.aot_inst) |ai|
        ai.thread_context.enter()
    else
        null;
    defer if (active_ctx_scope) |*scope| scope.deinit();

    // Build MemoryContext backed by the core frame's linear memory (or owner_inst fallback).
    const initial_mem_slice: []u8 = if (frame.memory(default_mem_idx)) |mem|
        mem
    else
        instanceMemorySlice(owner_inst);

    var mem_ctx = MemoryContext{
        .memory = initial_mem_slice,
        .base_addr = 0,
        .allocator = allocator,
        // When post_return will run and free guest allocations, copy lifted
        // slices into `allocator` before invoking post_return.
        .zero_copy_slices = (resolved_post_return_idx == null),
        .validate_utf8 = true,
        .instance = owner_inst,
    };

    var frame_realloc_ctx: FrameReallocCtx = undefined;
    if (resolved_realloc_idx) |r_idx| {
        frame_realloc_ctx = .{
            .frame = &frame,
            .owner_inst = owner_inst,
            .realloc_idx = r_idx,
            .mem_idx = default_mem_idx,
            .mem_ctx = &mem_ctx,
        };
        mem_ctx.realloc_ctx = &frame_realloc_ctx;
        mem_ctx.realloc_fn = &frameReallocCallback;
        mem_ctx.realloc_counter = &owner_inst.realloc_call_count;
    } else {
        mem_ctx.realloc_ctx = owner_inst;
        mem_ctx.realloc_fn = &instanceReallocCallback;
    }

    // Lower arguments at comptime and push onto `frame`.
    const addr_core_type: core_types.ValType = if (arch == .wasm64) .i64 else .i32;

    if (comptime cls.params_spill) {
        if (mem_ctx.memory.len == 0) return error.MemoryNotAvailable;
        const args_layout = comptime layoutOf(cls.ArgsStruct, arch);
        const raw_ptr = try mem_ctx.realloc(0, 0, args_layout.align_, args_layout.size);
        try storeToMemory(cls.ArgsStruct, arch, &mem_ctx, raw_ptr, args);
        if (arch == .wasm64) {
            try frame.pushSlot(.{ .i64 = @bitCast(@as(u64, @intCast(raw_ptr))) });
        } else {
            try frame.pushSlot(.{ .i32 = @bitCast(@as(u32, @intCast(raw_ptr))) });
        }
    } else {
        const fn_params = @typeInfo(FnType).@"fn".param_types;
        inline for (fn_params, 0..) |maybe_p, i| {
            const PType = maybe_p.?;
            const pl = comptime layoutOf(PType, arch);
            if (comptime pl.needs_realloc) {
                if (mem_ctx.memory.len == 0) return error.MemoryNotAvailable;
            }
            const sub = try lowerToFlat(PType, arch, &mem_ctx, args[i]);
            inline for (0..pl.flat_types.len) |j| {
                try frame.pushSlot(flatValToCoreValue(pl.flat_types[j], sub[j]));
            }
        }
    }

    const core_func_local = CoreFuncIdxLocal.from(local.core_func_idx);
    const param_count_no_retptr = comptime (if (cls.params_spill) 1 else cls.params_flat);
    var core_param_types_buf: [param_count_no_retptr + 1]core_types.ValType = undefined;
    inline for (0..param_count_no_retptr) |i| {
        core_param_types_buf[i] = cls.core_param_types[i].toCoreValType();
    }

    // Execute core function and lift results.
    if (comptime cls.results_spill) {
        if (mem_ctx.memory.len == 0) return error.MemoryNotAvailable;
        const ret_layout = comptime layoutOf(ZigReturn, arch);
        const callee_returns_ptr = coreFuncReturnsResultPtr(core_entry, local.core_func_idx);

        var ret_ptr_u64: u64 = 0;
        if (callee_returns_ptr) {
            const result_types = [_]core_types.ValType{addr_core_type};
            try frame.executeCore(core_func_local, core_param_types_buf[0..param_count_no_retptr], &result_types);
            const ptr_val = try frame.popSlot(addr_core_type);
            ret_ptr_u64 = if (arch == .wasm64)
                @bitCast(ptr_val.i64)
            else
                @as(u64, @as(u32, @bitCast(ptr_val.i32)));
        } else {
            const alloc_offset = try mem_ctx.realloc(0, 0, ret_layout.align_, ret_layout.size);
            ret_ptr_u64 = @intCast(alloc_offset);
            if (arch == .wasm64) {
                try frame.pushSlot(.{ .i64 = @bitCast(ret_ptr_u64) });
            } else {
                try frame.pushSlot(.{ .i32 = @bitCast(@as(u32, @intCast(ret_ptr_u64))) });
            }
            core_param_types_buf[param_count_no_retptr] = addr_core_type;
            try frame.executeCore(core_func_local, core_param_types_buf[0 .. param_count_no_retptr + 1], &.{});
        }

        // Refresh memory slice in case callee grew linear memory during execution.
        if (frame.memory(default_mem_idx)) |mem| {
            mem_ctx.memory = mem;
        } else {
            const refreshed = instanceMemorySlice(owner_inst);
            if (refreshed.len > 0) mem_ctx.memory = refreshed;
        }

        const ret_offset = try mem_ctx.validateRange(ret_ptr_u64, ret_layout.size);
        const lifted = try loadFromMemory(ZigReturn, arch, &mem_ctx, ret_offset);

        if (resolved_post_return_idx) |pr_idx| {
            if (arch == .wasm64) {
                try frame.pushSlot(.{ .i64 = @bitCast(ret_ptr_u64) });
            } else {
                try frame.pushSlot(.{ .i32 = @bitCast(@as(u32, @truncate(ret_ptr_u64))) });
            }
            const pr_params = [_]core_types.ValType{addr_core_type};
            try frame.executeCore(pr_idx, &pr_params, &.{});
        }

        return lifted;
    } else if (comptime cls.results_flat == 1) {
        const ret_ft = comptime cls.core_result_types[0];
        const ret_core_vt = comptime ret_ft.toCoreValType();
        const result_types = [_]core_types.ValType{ret_core_vt};

        try frame.executeCore(core_func_local, core_param_types_buf[0..param_count_no_retptr], &result_types);
        const raw_val = try frame.popSlot(ret_core_vt);

        if (frame.memory(default_mem_idx)) |mem| {
            mem_ctx.memory = mem;
        } else {
            const refreshed = instanceMemorySlice(owner_inst);
            if (refreshed.len > 0) mem_ctx.memory = refreshed;
        }

        const raw_u64 = coreValueToRawSlot(ret_ft, raw_val);
        const flat_scalar = rawSlotToFlatVal(ret_ft, raw_u64);
        const lifted = try liftFromFlat(ZigReturn, arch, &mem_ctx, .{flat_scalar});

        if (resolved_post_return_idx) |pr_idx| {
            try frame.pushSlot(raw_val);
            const pr_params = [_]core_types.ValType{ret_core_vt};
            try frame.executeCore(pr_idx, &pr_params, &.{});
        }

        return lifted;
    } else {
        try frame.executeCore(core_func_local, core_param_types_buf[0..param_count_no_retptr], &.{});
        if (resolved_post_return_idx) |pr_idx| {
            try frame.executeCore(pr_idx, &.{}, &.{});
        }
        return {};
    }
}

/// Looks up `export_name` on `ci` (following any `.forwarded` sub-instance export
/// chain to its `.local` export) and invokes it using comptime-derived Canonical
/// ABI lowering and lifting (`CanonLower(FnType, opts)`).
pub fn callTypedExport(
    comptime FnType: type,
    comptime opts: anytype,
    ci: *comp_instance.ComponentInstance,
    export_name: []const u8,
    args: CanonLower(FnType, opts).ArgsTuple,
    allocator: std.mem.Allocator,
) !CanonLower(FnType, opts).ZigReturn {
    var cur_owner: *comp_instance.ComponentInstance = ci;
    var cur_name: []const u8 = export_name;
    var hops: usize = 0;
    while (hops < 32) : (hops += 1) {
        const exp = cur_owner.getExport(cur_name) orelse return error.FunctionNotFound;
        switch (exp) {
            .local => |local| return callTypedExportByLocal(FnType, opts, cur_owner, local, args, allocator),
            .forwarded => |fwd| {
                cur_owner = fwd.owner;
                cur_name = fwd.owner_export_name;
            },
        }
    }
    return error.FunctionNotFound;
}

// ============================================================================
// Section 11: Unit Tests
// ============================================================================

const testing = std.testing;

test "layoutOf: primitive scalars and handles (wasm32 & wasm64)" {
    try testing.expectEqual(@as(u64, 1), layoutOf(bool, .wasm32).size);
    try testing.expectEqual(@as(u32, 1), layoutOf(bool, .wasm32).align_);
    try testing.expectEqualSlices(FlatValType, &.{.i32}, layoutOf(bool, .wasm32).flat_types);

    try testing.expectEqual(@as(u64, 2), layoutOf(u16, .wasm32).size);
    try testing.expectEqual(@as(u32, 2), layoutOf(i16, .wasm32).align_);

    try testing.expectEqual(@as(u64, 4), layoutOf(u32, .wasm32).size);
    try testing.expectEqual(@as(u64, 8), layoutOf(i64, .wasm32).size);
    try testing.expectEqualSlices(FlatValType, &.{.i64}, layoutOf(i64, .wasm32).flat_types);

    try testing.expectEqual(@as(u64, 4), layoutOf(f32, .wasm32).size);
    try testing.expectEqualSlices(FlatValType, &.{.f32}, layoutOf(f32, .wasm32).flat_types);
    try testing.expectEqual(@as(u64, 8), layoutOf(f64, .wasm32).size);
    try testing.expectEqualSlices(FlatValType, &.{.f64}, layoutOf(f64, .wasm32).flat_types);

    try testing.expectEqual(@as(u64, 4), layoutOf(Char, .wasm32).size);
    try testing.expectEqualSlices(FlatValType, &.{.i32}, layoutOf(Char, .wasm32).flat_types);

    const DummyRes = struct {};
    try testing.expectEqual(@as(u64, 4), layoutOf(Own(DummyRes), .wasm64).size);
    try testing.expectEqualSlices(FlatValType, &.{.i32}, layoutOf(Borrow(DummyRes), .wasm64).flat_types);
    try testing.expectEqual(@as(u64, 4), layoutOf(Stream(u8), .wasm32).size);
    try testing.expectEqual(@as(u64, 4), layoutOf(Future(u32), .wasm32).size);
    try testing.expectEqual(@as(u64, 4), layoutOf(ErrorContext, .wasm32).size);
}

test "layoutOf: strings and slices in wasm32 vs wasm64" {
    const s32 = layoutOf([]const u8, .wasm32);
    try testing.expectEqual(@as(u64, 8), s32.size);
    try testing.expectEqual(@as(u32, 4), s32.align_);
    try testing.expectEqualSlices(FlatValType, &.{ .i32, .i32 }, s32.flat_types);
    try testing.expect(s32.needs_realloc);

    const s64 = layoutOf([]const u8, .wasm64);
    try testing.expectEqual(@as(u64, 16), s64.size);
    try testing.expectEqual(@as(u32, 8), s64.align_);
    try testing.expectEqualSlices(FlatValType, &.{ .i64, .i64 }, s64.flat_types);
    try testing.expect(s64.needs_realloc);

    const ws64 = layoutOf(WitString, .wasm64);
    try testing.expectEqual(@as(u64, 16), ws64.size);
    try testing.expectEqual(@as(u32, 8), ws64.align_);
    try testing.expectEqualSlices(FlatValType, &.{ .i64, .i64 }, ws64.flat_types);
}

test "layoutOf: records, variants, flags, options, results, and joinFlatType" {
    const Point = struct {
        tag: u8,
        x: f64,
        y: u16,
    };
    const pl = layoutOf(Point, .wasm32);
    try testing.expectEqual(@as(u64, 24), pl.size);
    try testing.expectEqual(@as(u32, 8), pl.align_);
    try testing.expectEqual(@as(usize, 0), fieldOffset(Point, 0, .wasm32));
    try testing.expectEqual(@as(usize, 8), fieldOffset(Point, 1, .wasm32));
    try testing.expectEqual(@as(usize, 16), fieldOffset(Point, 2, .wasm32));
    try testing.expectEqualSlices(FlatValType, &.{ .i32, .f64, .i32 }, pl.flat_types);

    const MixedVariant = union(enum) {
        int_arm: i32,
        float_arm: f32,
        wide_arm: f64,
        empty_arm: void,
    };
    const vl = layoutOf(MixedVariant, .wasm32);
    try testing.expectEqual(@as(u64, 16), vl.size);
    try testing.expectEqual(@as(u32, 8), vl.align_);
    try testing.expectEqualSlices(FlatValType, &.{ .i32, .i64 }, vl.flat_types);

    const SmallResult = Result(f32, i32);
    const rl = layoutOf(SmallResult, .wasm32);
    try testing.expectEqual(@as(u64, 8), rl.size);
    try testing.expectEqual(@as(u32, 4), rl.align_);
    try testing.expectEqualSlices(FlatValType, &.{ .i32, .i32 }, rl.flat_types);

    const Perms = packed struct(u8) {
        read: bool = false,
        write: bool = false,
        exec: bool = false,
        _pad: u5 = 0,
    };
    const fl = layoutOf(Perms, .wasm32);
    try testing.expectEqual(@as(u64, 1), fl.size);
    try testing.expectEqual(@as(u32, 1), fl.align_);
    try testing.expectEqualSlices(FlatValType, &.{.i32}, fl.flat_types);
}

test "MemoryContext.validateRange: rejects 32-bit and 64-bit overflow and out-of-bounds" {
    var mem: [256]u8 = @splat(0);
    const ctx = MemoryContext{ .memory = &mem };

    try testing.expectEqual(@as(usize, 0), try ctx.validateRange(0, 256));
    try testing.expectEqual(@as(usize, 128), try ctx.validateRange(128, 128));
    try testing.expectError(error.OutOfBounds, ctx.validateRange(128, 129));
    try testing.expectError(error.OutOfBounds, ctx.validateRange(257, 0));

    // 32-bit wraparound attempt: 0xFFFF_FFF0 + 0x20 wraps in u32 to 0x10
    try testing.expectError(error.OutOfBounds, ctx.validateRange(0xFFFF_FFF0, 0x20));

    // 64-bit wraparound attempt: maxInt(u64) - 15 + 32 wraps in u64
    try testing.expectError(error.OutOfBounds, ctx.validateRange(std.math.maxInt(u64) - 15, 32));
    try testing.expectError(error.OutOfBounds, ctx.validateRange(100, std.math.maxInt(u64)));
}

test "storeToMemory and loadFromMemory: round-trip complex nested types (wasm32 & wasm64)" {
    const Status = enum(u8) { ok, busy, err };
    const Flags = packed struct(u16) {
        a: bool = false,
        b: bool = false,
        c: bool = false,
        _pad: u13 = 0,
    };
    const Event = union(enum) {
        none: void,
        code: i32,
        message: WitString,
    };
    const Record = struct {
        id: u64,
        status: Status,
        flags: Flags,
        ch: Char,
        opt: ?u32,
        res: Result(WitString, u32),
        event: Event,
        numbers: []const i32,
    };

    const sample_nums = [_]i32{ -10, 0, 42, 9999 };
    const original = Record{
        .id = 0x1234_5678_9ABC_DEF0,
        .status = .busy,
        .flags = .{ .a = true, .c = true },
        .ch = @fromBackingInt(@intCast(0x1F680)),
        .opt = 777,
        .res = Result(WitString, u32).ok(WitString.init("hello wit-zigar")),
        .event = .{ .message = WitString.init("nested variant string") },
        .numbers = &sample_nums,
    };

    // Test wasm32
    {
        var mem: [4096]u8 = @splat(0);
        var bump = BumpAllocator.init(256, mem.len);
        var ctx = bump.memoryContext(&mem, testing.allocator);

        try storeToMemory(Record, .wasm32, &ctx, 0, original);
        const loaded = try loadFromMemory(Record, .wasm32, &ctx, 0);

        try testing.expectEqual(original.id, loaded.id);
        try testing.expectEqual(original.status, loaded.status);
        try testing.expectEqual(original.flags.a, loaded.flags.a);
        try testing.expectEqual(original.flags.b, loaded.flags.b);
        try testing.expectEqual(original.flags.c, loaded.flags.c);
        try testing.expectEqual(original.ch, loaded.ch);
        try testing.expectEqual(original.opt, loaded.opt);
        try testing.expectEqualStrings("hello wit-zigar", loaded.res.payload.ok.slice());
        try testing.expectEqualStrings("nested variant string", loaded.event.message.slice());
        try testing.expectEqualSlices(i32, &sample_nums, loaded.numbers);
    }

    // Test wasm64 (memory64)
    {
        var mem: [4096]u8 = @splat(0);
        var bump = BumpAllocator.init(512, mem.len);
        var ctx = bump.memoryContext(&mem, testing.allocator);

        try storeToMemory(Record, .wasm64, &ctx, 0, original);
        const loaded = try loadFromMemory(Record, .wasm64, &ctx, 0);

        try testing.expectEqual(original.id, loaded.id);
        try testing.expectEqualStrings("hello wit-zigar", loaded.res.payload.ok.slice());
        try testing.expectEqualStrings("nested variant string", loaded.event.message.slice());
        try testing.expectEqualSlices(i32, &sample_nums, loaded.numbers);
    }
}

test "CanonLower and CanonLift: direct register call and indirect spill (>16 params, >1 result)" {
    // 1. Direct register call
    const DirectFn = fn (a: i32, b: f64, label: WitString) u64;
    const Impl1 = struct {
        fn handle(a: i32, b: f64, label: WitString) u64 {
            return @as(u64, @intCast(a)) + @as(u64, @intFromFloat(b)) + label.slice().len;
        }
    };

    var mem: [4096]u8 = @splat(0);
    var bump = BumpAllocator.init(64, mem.len);
    var ctx = bump.memoryContext(&mem, testing.allocator);

    const Lower1 = CanonLower(DirectFn, .{ .arch = .wasm32 });
    const Lift1 = CanonLift(DirectFn, .{ .arch = .wasm32 });
    try testing.expect(!Lower1.sig.params_indirect);
    try testing.expect(!Lower1.sig.results_indirect);

    const core_args1 = try Lower1.lowerArgs(&ctx, .{ 10, 20.0, WitString.init("abcde") });
    const core_ret1 = try Lift1.call(&ctx, &Impl1.handle, core_args1);
    try testing.expectEqual(@as(i64, 35), core_ret1);

    // 2. Spilled params (>16 flat slots) and spilled result (>1 flat slot)
    const BigInput = struct {
        v: [17]u32,
    };
    const BigOutput = struct {
        sum: u64,
        first: u32,
        last: u32,
    };
    const SpillFn = fn (input: BigInput) BigOutput;
    const Impl2 = struct {
        fn compute(input: BigInput) BigOutput {
            var s: u64 = 0;
            for (input.v) |x| s += x;
            return .{ .sum = s, .first = input.v[0], .last = input.v[16] };
        }
    };

    const Lower2 = CanonLower(SpillFn, .{ .arch = .wasm32 });
    const Lift2 = CanonLift(SpillFn, .{ .arch = .wasm32 });
    try testing.expect(Lower2.sig.params_indirect);
    try testing.expect(Lower2.sig.results_indirect);
    try testing.expectEqualSlices(FlatValType, &.{ .i32, .i32 }, Lower2.sig.param_types);
    try testing.expectEqualSlices(FlatValType, &.{}, Lower2.sig.result_types);

    var arr: [17]u32 = undefined;
    for (0..17) |i| arr[i] = @intCast(i + 1);
    const core_args2 = try Lower2.lowerArgs(&ctx, .{BigInput{ .v = arr }});
    try Lift2.callLowered(&ctx, &Impl2.compute, core_args2);

    const retptr: usize = @intCast(@as(u32, @bitCast(core_args2[1])));
    const out = try loadFromMemory(BigOutput, .wasm32, &ctx, retptr);
    try testing.expectEqual(@as(u64, 153), out.sum);
    try testing.expectEqual(@as(u32, 1), out.first);
    try testing.expectEqual(@as(u32, 17), out.last);
}

test "makeHostFunc: bridges both flat core slots (call_flat) and InterfaceValue (call)" {
    var comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    const ci = try comp_instance.instantiate(&comp, testing.allocator);
    defer ci.deinit();
    try ci.enableTestMem(testing.allocator, 4096);
    defer ci.disableTestMem();

    const Host = struct {
        fn addAndLen(inst: *comp_instance.ComponentInstance, a: u32, b: u32, msg: WitString) u32 {
            _ = inst;
            return a + b + @as(u32, @intCast(msg.slice().len));
        }
    };

    const hf = makeHostFunc(Host.addAndLen);
    try testing.expect(hf.call != null);
    try testing.expect(hf.call_flat != null);

    // Write "hello!" into guest test memory at offset 64.
    const msg = "hello!";
    @memcpy(ci.writableGuestBytes(64, @intCast(msg.len)).?, msg);

    // 1. Invoke via flat-slot fast path (`call_flat`) — zero heap allocations, zero cabi_realloc calls!
    const realloc_before = ci.realloc_call_count;
    const flat_args = [_]u64{ 10, 20, 64, msg.len };
    var flat_results = [_]u64{0};
    try hf.call_flat.?(hf.context, ci, &flat_args, &flat_results, 0, testing.allocator);
    try testing.expectEqual(@as(u64, 36), flat_results[0]);
    try testing.expectEqual(realloc_before, ci.realloc_call_count);

    // 2. Invoke via InterfaceValue bridge (`call`)
    const iv_args = [_]abi.InterfaceValue{
        .{ .u32 = 100 },
        .{ .u32 = 200 },
        .{ .string = .{ .ptr = 64, .len = @intCast(msg.len) } },
    };
    var iv_results: [1]abi.InterfaceValue = undefined;
    try hf.call.?(hf.context, ci, &iv_args, &iv_results, testing.allocator);
    try testing.expectEqual(@as(u32, 306), iv_results[0].u32);
}

test "makeHostFunc: Own(R) and Borrow(R) resource table round-trip and spilled string result" {
    var comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    const ci = try comp_instance.instantiate(&comp, testing.allocator);
    defer ci.deinit();
    try ci.enableTestMem(testing.allocator, 4096);
    defer ci.disableTestMem();

    const CounterRes = struct {
        pub const resource_type_idx: u32 = 0;
        pub const auto_rep: bool = true;
    };

    const Host = struct {
        fn createCounter(base: u32) Own(CounterRes) {
            return .{ .idx = base + 100 };
        }
        fn inspectCounter(b: Borrow(CounterRes)) u32 {
            return b.idx * 2;
        }
        fn greet(ci_ptr: *comp_instance.ComponentInstance, name: WitString) WitString {
            _ = ci_ptr;
            if (std.mem.eql(u8, name.slice(), "world")) {
                return WitString.init("hello, world!");
            }
            return WitString.init("hello!");
        }
    };

    const create_hf = makeHostFunc(Host.createCounter);
    const inspect_hf = makeHostFunc(Host.inspectCounter);
    const greet_hf = makeHostFunc(Host.greet);

    // Call createCounter(23) -> allocates resource handle in ci's ResourceTable for rep=123
    var res_slot = [_]u64{0};
    try create_hf.call_flat.?(null, ci, &.{23}, &res_slot, 0, testing.allocator);
    const handle: u32 = @truncate(res_slot[0]);

    // Call inspectCounter(borrow_handle) -> looks up rep=123 in ResourceTable, returns 246
    try inspect_hf.call_flat.?(null, ci, &.{handle}, &res_slot, 0, testing.allocator);
    try testing.expectEqual(@as(u64, 246), res_slot[0]);

    // Call greet("world") with retptr = 256
    @memcpy(ci.writableGuestBytes(128, 5).?, "world");
    try greet_hf.call_flat.?(null, ci, &.{ 128, 5 }, &.{}, 256, testing.allocator);
    const mem_ctx = memoryContextForInstance(ci, testing.allocator);
    const greeted = try loadFromMemory(WitString, .wasm32, &mem_ctx, 256);
    try testing.expectEqualStrings("hello, world!", greeted.slice());
}

test "callTypedExport: calls lifted guest export on ComponentInstance via CallFrame" {
    // Minimal component with 1 core module exporting "add": (param i32 i32) (result i32)
    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: (i32, i32) -> i32
        0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01,
        0x7f,
        // function section: 1 fn of type 0
        0x03, 0x02, 0x01, 0x00,
        // export section: "add" -> func 0
        0x07, 0x07, 0x01,
        0x03, 'a',  'd',  'd',  0x00, 0x00,
        // code section: local.get 0, local.get 1, i32.add
        0x0a, 0x09,
        0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a,
        0x0b,
    };

    const core_modules = [_]ctypes.CoreModule{.{ .data = &core_wasm }};
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .instantiate = .{ .module_idx = 0, .args = &.{} } },
    };
    const aliases_decl = [_]ctypes.Alias{
        .{ .instance_export = .{
            .sort = .{ .core = .func },
            .instance_idx = 0,
            .name = "add",
        } },
    };
    const params = [_]ctypes.NamedValType{
        .{ .name = "a", .type = .s32 },
        .{ .name = "b", .type = .s32 },
    };
    const type_defs = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &params, .results = .{ .unnamed = .s32 } } },
    };
    const canons = [_]ctypes.Canon{
        .{ .lift = .{ .core_func_idx = 0, .type_idx = 0, .opts = &.{} } },
    };
    const exports_decl = [_]ctypes.ExportDecl{
        .{
            .name = "add",
            .desc = .{ .func = 0 },
            .sort_idx = .{ .sort = .func, .idx = 0 },
        },
    };
    var component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &aliases_decl,
        .types = &type_defs,
        .canons = &canons,
        .imports = &.{},
        .exports = &exports_decl,
    };

    const ci = try comp_instance.instantiate(&component, testing.allocator);
    defer ci.deinit();

    const sum = try callTypedExport(fn (i32, i32) i32, .{}, ci, "add", .{ 19, 23 }, testing.allocator);
    try testing.expectEqual(@as(i32, 42), sum);
}
