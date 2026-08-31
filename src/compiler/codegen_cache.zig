//! AOT codegen cache (#761 Phase 2a — infrastructure).
//!
//! Per-function compiled code is fully position-independent modulo a
//! list of inter-function PC-relative call patches. That lets us cache
//! `(ir_sha256, code_bytes, call_patches)` per function in a sidecar
//! file and, on a recompile, memcpy cached code into the new module
//! layout while re-resolving the call patches against fresh offsets.
//!
//! This file owns the **infrastructure** for that scheme: the on-disk
//! cache format (magic, version, header, per-function entries), the
//! canonical IR hasher (`hashFunction`), the module-level epoch hasher
//! (`hashModuleEpoch`), and the binary serialiser / deserialiser. It
//! does **not** integrate with the codegen pipeline — that wires up in
//! Phase 2b via `compileModuleCached` in each backend.
//!
//! ## Cache integrity model
//!
//! A cache is rejected wholesale (header mismatch) on any of:
//!   - magic byte sequence mismatch
//!   - format version mismatch
//!   - wamr build id mismatch (different compiler binary)
//!   - target arch + ABI mismatch (different machine code shape)
//!   - module epoch mismatch (IrModule-level invariants codegen reads:
//!     import_count, memory mode, global_types, global_offsets,
//!     global_storage_size, func_types, func_type_indices)
//!   - func_count mismatch (functions added/removed)
//!
//! Within a valid cache, each per-function entry is reused iff its
//! stored `ir_sha256` matches a freshly-computed hash of the IR
//! function the caller hands in. Mismatches fall back to a normal
//! codegen call for that function.
//!
//! ## Per-function `ir_sha256`
//!
//! We hash the IR **structurally** (not via the diagnostic printer in
//! `print.zig`, which intentionally elides codegen-observable fields
//! like SIMD lane indices and memory offsets to stay readable). The
//! walker (`HashWriter.writeAny`) reflects over every field of every
//! Inst.Op union variant via comptime, so adding a new op variant or
//! payload field is automatically covered. Two explicit skip lists
//! handle non-codegen-observable metadata:
//!
//!   - `IrFunction`: skip `name` (debug-only), `allocator`,
//!     `next_vreg` (high-water mark recomputable from any block's
//!     instructions), `owned_br_table_targets` (memory-pinning helper;
//!     content lives inline on `Inst.Op.br_table.targets` and is
//!     hashed there).
//!   - `BasicBlock`: skip `allocator`, `predecessors` (recomputable
//!     from any block's terminator).

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("ir/ir.zig");
const passes = @import("ir/passes.zig");

const Sha256 = std.crypto.hash.sha2.Sha256;

pub const cache_magic = "WAMRCAC\x00".*;
pub const cache_format_version: u32 = 1;

/// Sentinel cap on cache-file size we'll accept on load. 256 MiB is
/// more than 10× the largest cwasm we've seen (#743 keyvault ≈ 37 MB
/// + ~equal codegen output) and small enough to refuse pathological
/// inputs without an OOM probe.
pub const max_cache_file_bytes: usize = 256 * 1024 * 1024;

/// Per-function entry's hard cap on emitted code length and patch
/// count. Defensive deserialiser bound; real functions are kilobytes.
pub const max_func_code_bytes: u32 = 16 * 1024 * 1024;
pub const max_call_patches_per_func: u32 = 1 << 20;

/// Per-compile-call statistics for diagnosis. Populated by
/// `compileModuleCached` and surfaced by the CLI / tests.
pub const CacheStats = struct {
    reused: u32 = 0,
    recompiled: u32 = 0,
};

/// Output of `<arch>.compileModuleCached`. Shared between backends so
/// the CLI can dispatch without a per-arch wrapper type.
pub const CompileResultCached = struct {
    /// Concatenated native code for every function in the module,
    /// inter-function PC-relative calls already resolved.
    code: []u8,
    /// Per-local-function byte offset into `code`.
    offsets: []u32,
    /// Per-function cache entries. Caller owns and must free each
    /// entry's `code` + `call_patches` (or use the entries to build a
    /// `Cache` and call `Cache.deinit`).
    cache_functions: []CachedFunction,
    stats: CacheStats,
};

/// Inter-function PC-relative call patch carried in a cached function.
/// `patch_offset` is relative to the start of this function's code
/// (not the module's global code blob); the reuse path re-bases by
/// adding `func_start_in_new_module`. `target_func_idx` is the
/// **local** function index (post-import-offset) of the call target.
pub const FuncCallPatch = extern struct {
    patch_offset: u32,
    target_func_idx: u32,
};

/// Identifies the platform ABI variant that affects codegen.
/// `target_arch` alone is too weak — x86_64 SysV vs Win64 emit
/// substantially different prologues / arg-reg sequences (see
/// `x86_64_caller_saved_indices` in `compile.zig`).
pub const TargetAbi = enum(u8) {
    x86_64_sysv = 0,
    x86_64_win64 = 1,
    aarch64_aapcs = 2,

    pub fn forHost(arch: passes.TargetArch) TargetAbi {
        return switch (arch) {
            .x86_64 => switch (builtin.os.tag) {
                .windows => .x86_64_win64,
                else => .x86_64_sysv,
            },
            .aarch64 => .aarch64_aapcs,
        };
    }
};

/// Inputs to `hashModuleEpoch`. All fields the codegen back-ends read
/// from `IrModule` (directly or via `compileModuleWithOptions`) belong
/// here — otherwise a cache could be reused against an IrModule that
/// produces structurally-different code despite identical per-function
/// IR.
pub const ModuleEpochInputs = struct {
    wamr_build_id: []const u8,
    target_arch: passes.TargetArch,
    target_abi: TargetAbi,
    import_count: u32,
    has_memory64: bool = false,
    has_shared_memory: bool = false,
    /// Optional; treated as empty slice when null.
    global_types: ?[]const ir.IrType = null,
    /// Optional; treated as empty slice when null.
    global_offsets: ?[]const u32 = null,
    global_storage_size: u32 = 0,
    /// `IrFuncType` carries `params: []const IrType, results: []const IrType`.
    func_types: []const ir.IrFuncType = &.{},
    func_type_indices: []const u32 = &.{},
};

/// One cached function's codegen output.
pub const CachedFunction = struct {
    ir_sha256: [32]u8,
    code: []u8,
    call_patches: []FuncCallPatch,
};

/// In-memory representation of a cache file. All slices are owned by
/// the allocator passed to `deserialize` (or the builder that
/// constructed the cache); call `deinit` to free them.
pub const Cache = struct {
    wamr_build_id: []const u8,
    target_arch: passes.TargetArch,
    target_abi: TargetAbi,
    module_epoch: [32]u8,
    functions: []CachedFunction,

    pub fn deinit(self: *Cache, allocator: std.mem.Allocator) void {
        allocator.free(self.wamr_build_id);
        for (self.functions) |*f| {
            allocator.free(f.code);
            allocator.free(f.call_patches);
        }
        allocator.free(self.functions);
        self.* = undefined;
    }
};

// ── Hashers ────────────────────────────────────────────────────────────────

/// Compute the canonical SHA-256 of a function's codegen-observable
/// IR. Deterministic across runs and platforms.
pub fn hashFunction(func: *const ir.IrFunction) [32]u8 {
    var sh = Sha256.init(.{});
    var w: HashWriter = .{ .h = &sh };
    w.writeIrFunction(func);
    var out: [32]u8 = undefined;
    sh.final(&out);
    return out;
}

/// Compute the canonical SHA-256 of module-level codegen invariants.
/// Mismatch invalidates the entire cache (per-function reuse becomes
/// unsafe regardless of per-function IR equality).
pub fn hashModuleEpoch(inputs: ModuleEpochInputs) [32]u8 {
    var sh = Sha256.init(.{});
    var w: HashWriter = .{ .h = &sh };
    // Domain separator so this hash can never collide with hashFunction.
    w.writeBytes("epoch\x00");
    w.writeBytes(inputs.wamr_build_id);
    w.writeInt(u8, 0); // separator after variable-length build id
    w.writeInt(u32, @intFromEnum(inputs.target_arch));
    w.writeInt(u8, @intFromEnum(inputs.target_abi));
    w.writeInt(u32, inputs.import_count);
    w.writeInt(u8, @intFromBool(inputs.has_memory64));
    w.writeInt(u8, @intFromBool(inputs.has_shared_memory));
    w.writeOptSliceEnum(ir.IrType, inputs.global_types);
    w.writeOptSliceInt(u32, inputs.global_offsets);
    w.writeInt(u32, inputs.global_storage_size);
    w.writeInt(u32, @intCast(inputs.func_types.len));
    for (inputs.func_types) |ft| {
        w.writeInt(u32, @intCast(ft.params.len));
        for (ft.params) |p| w.writeInt(u32, @intFromEnum(p));
        w.writeInt(u32, @intCast(ft.results.len));
        for (ft.results) |r| w.writeInt(u32, @intFromEnum(r));
    }
    w.writeInt(u32, @intCast(inputs.func_type_indices.len));
    for (inputs.func_type_indices) |i| w.writeInt(u32, i);
    var out: [32]u8 = undefined;
    sh.final(&out);
    return out;
}

/// Comptime-reflective canonical hasher. Writes a deterministic byte
/// stream into the wrapped Sha256 by walking every field of every
/// reachable struct / union / slice / enum / primitive — so adding a
/// new `Inst.Op` variant or payload field is hashed automatically with
/// no maintenance.
const HashWriter = struct {
    h: *Sha256,

    fn writeBytes(self: HashWriter, b: []const u8) void {
        self.h.update(b);
    }
    fn writeInt(self: HashWriter, comptime T: type, v: T) void {
        var buf: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &buf, v, .little);
        self.h.update(&buf);
    }
    fn writeBool(self: HashWriter, v: bool) void {
        self.h.update(&[_]u8{if (v) 1 else 0});
    }
    fn writeOptSliceEnum(self: HashWriter, comptime E: type, opt: ?[]const E) void {
        if (opt) |s| {
            self.writeInt(u8, 1);
            self.writeInt(u32, @intCast(s.len));
            for (s) |e| self.writeInt(u32, @intFromEnum(e));
        } else {
            self.writeInt(u8, 0);
        }
    }
    fn writeOptSliceInt(self: HashWriter, comptime T: type, opt: ?[]const T) void {
        if (opt) |s| {
            self.writeInt(u8, 1);
            self.writeInt(u32, @intCast(s.len));
            for (s) |v| self.writeInt(T, v);
        } else {
            self.writeInt(u8, 0);
        }
    }

    /// Manual walker for `IrFunction`: skips `name` (debug-only),
    /// `allocator`, `next_vreg` (recomputable), and
    /// `owned_br_table_targets` (memory-pin helper; content lives
    /// inline on `Inst.Op.br_table.targets` and is hashed there).
    fn writeIrFunction(self: HashWriter, func: *const ir.IrFunction) void {
        self.writeBytes("func\x00");
        self.writeInt(u32, func.param_count);
        self.writeInt(u32, func.result_count);
        self.writeInt(u32, func.local_count);
        self.writeOptInt(u32, func.phi_synth_local_start);
        self.writeOptInt(u32, func.phi_synth_local_end);
        self.writeOptSliceEnum(ir.IrType, func.local_types);
        self.writeInt(u32, @intCast(func.blocks.items.len));
        for (func.blocks.items) |*blk| self.writeBasicBlock(blk);
    }

    fn writeOptInt(self: HashWriter, comptime T: type, opt: ?T) void {
        if (opt) |v| {
            self.writeInt(u8, 1);
            self.writeInt(T, v);
        } else {
            self.writeInt(u8, 0);
        }
    }

    /// Manual walker for `BasicBlock`: skips `allocator` and
    /// `predecessors` (recomputable from any block's terminator —
    /// passes may leave stale predecessors that codegen rebuilds).
    fn writeBasicBlock(self: HashWriter, blk: *const ir.BasicBlock) void {
        self.writeBytes("blk\x00");
        self.writeInt(u32, blk.id);
        self.writeInt(u32, @intCast(blk.instructions.items.len));
        for (blk.instructions.items) |inst| self.writeAny(inst);
    }

    /// Reflective walker for any other type. Handles primitives,
    /// enums, optionals, arrays, slices, structs, and **tagged**
    /// unions. Refuses non-data types (pointers other than slices,
    /// raw allocators, etc.) at comptime.
    fn writeAny(self: HashWriter, value: anytype) void {
        const T = @TypeOf(value);
        const info = @typeInfo(T);
        switch (info) {
            .int => |i| {
                // Promote arbitrary-width integers to u64 / i64 (sign
                // preserving) so payload fields like `lane: u4`,
                // `align: u8`, `width: u3` all hash deterministically
                // without per-width specialisation. The wider field
                // doesn't lose info because zero-/sign-extension is
                // unique.
                if (i.signedness == .signed) {
                    const w: i64 = @intCast(value);
                    self.writeInt(i64, w);
                } else {
                    if (i.bits <= 64) {
                        const w: u64 = @intCast(value);
                        self.writeInt(u64, w);
                    } else if (i.bits == 128) {
                        self.writeInt(u128, value);
                    } else {
                        @compileError("unsupported int bits: " ++ @typeName(T));
                    }
                }
            },
            .float => |f| switch (f.bits) {
                32 => {
                    const bits: u32 = @bitCast(value);
                    self.writeInt(u32, bits);
                },
                64 => {
                    const bits: u64 = @bitCast(value);
                    self.writeInt(u64, bits);
                },
                else => @compileError("unsupported float bits: " ++ @typeName(T)),
            },
            .bool => self.writeBool(value),
            .@"enum" => self.writeInt(u64, @intFromEnum(value)),
            .void => {},
            .optional => {
                if (value) |v| {
                    self.writeInt(u8, 1);
                    self.writeAny(v);
                } else {
                    self.writeInt(u8, 0);
                }
            },
            .array => for (value) |elem| self.writeAny(elem),
            .pointer => |p| switch (p.size) {
                .slice => {
                    self.writeInt(u32, @intCast(value.len));
                    if (p.child == u8) {
                        self.writeBytes(value);
                    } else {
                        for (value) |elem| self.writeAny(elem);
                    }
                },
                .one => self.writeAny(value.*),
                else => @compileError("unsupported pointer size " ++ @tagName(p.size) ++ " for " ++ @typeName(T)),
            },
            .@"struct" => |s| {
                inline for (s.fields) |f| {
                    self.writeAny(@field(value, f.name));
                }
            },
            .@"union" => |u| {
                if (u.tag_type == null) @compileError("non-tagged unions not supported: " ++ @typeName(T));
                self.writeInt(u32, @intFromEnum(std.meta.activeTag(value)));
                switch (value) {
                    inline else => |payload| self.writeAny(payload),
                }
            },
            else => @compileError("unsupported type in canonical hash: " ++ @typeName(T)),
        }
    }
};

// ── Serialiser ─────────────────────────────────────────────────────────────
//
// On-disk format (little-endian throughout):
//   magic              [8]u8 = "WAMRCAC\0"
//   version            u32   (= cache_format_version)
//   wamr_build_id_len  u32
//   wamr_build_id      [N]u8
//   target_arch        u8
//   target_abi         u8
//   module_epoch       [32]u8
//   func_count         u32
//   for each function:
//     ir_sha256        [32]u8
//     code_len         u32
//     code             [L]u8
//     call_patches_cnt u32
//     call_patches     [P * 8]u8   (u32 patch_offset, u32 target_func_idx)

pub const SerializeError = error{OutOfMemory};

pub const DeserializeError = error{
    Truncated,
    BadMagic,
    UnsupportedVersion,
    OversizeField,
    InvalidPatchOffset,
    InvalidTargetFuncIdx,
    InvalidTargetArch,
    InvalidTargetAbi,
    OutOfMemory,
};

pub fn serialize(cache: *const Cache, allocator: std.mem.Allocator) SerializeError![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    try buf.appendSlice(allocator, &cache_magic);
    try appendU32(&buf, allocator, cache_format_version);
    try appendU32(&buf, allocator, @intCast(cache.wamr_build_id.len));
    try buf.appendSlice(allocator, cache.wamr_build_id);
    try buf.append(allocator, @intFromEnum(cache.target_arch));
    try buf.append(allocator, @intFromEnum(cache.target_abi));
    try buf.appendSlice(allocator, &cache.module_epoch);
    try appendU32(&buf, allocator, @intCast(cache.functions.len));
    for (cache.functions) |f| {
        try buf.appendSlice(allocator, &f.ir_sha256);
        try appendU32(&buf, allocator, @intCast(f.code.len));
        try buf.appendSlice(allocator, f.code);
        try appendU32(&buf, allocator, @intCast(f.call_patches.len));
        for (f.call_patches) |p| {
            try appendU32(&buf, allocator, p.patch_offset);
            try appendU32(&buf, allocator, p.target_func_idx);
        }
    }
    return buf.toOwnedSlice(allocator);
}

pub fn deserialize(bytes: []const u8, allocator: std.mem.Allocator) DeserializeError!Cache {
    if (bytes.len > max_cache_file_bytes) return error.OversizeField;
    var r: Reader = .{ .buf = bytes, .pos = 0 };
    const magic = try r.takeBytes(cache_magic.len);
    if (!std.mem.eql(u8, magic, &cache_magic)) return error.BadMagic;
    const version = try r.readU32();
    if (version != cache_format_version) return error.UnsupportedVersion;

    const build_id_len = try r.readU32();
    if (build_id_len > 1024) return error.OversizeField;
    const build_id_src = try r.takeBytes(build_id_len);
    const build_id = try allocator.dupe(u8, build_id_src);
    errdefer allocator.free(build_id);

    const arch_byte = try r.readU8();
    const arch: passes.TargetArch = switch (arch_byte) {
        0 => .x86_64,
        1 => .aarch64,
        else => return error.InvalidTargetArch,
    };
    const abi_byte = try r.readU8();
    const abi: TargetAbi = switch (abi_byte) {
        0 => .x86_64_sysv,
        1 => .x86_64_win64,
        2 => .aarch64_aapcs,
        else => return error.InvalidTargetAbi,
    };
    var module_epoch: [32]u8 = undefined;
    @memcpy(&module_epoch, try r.takeBytes(32));

    const func_count = try r.readU32();
    if (func_count > (1 << 24)) return error.OversizeField;

    var funcs = try allocator.alloc(CachedFunction, func_count);
    // Track how many entries are fully initialised so the errdefer cleans
    // up only those — partially-allocated tail entries must not be freed.
    var initialised: usize = 0;
    errdefer {
        for (funcs[0..initialised]) |*f| {
            allocator.free(f.code);
            allocator.free(f.call_patches);
        }
        allocator.free(funcs);
    }

    for (funcs) |*f| {
        var ir_sha256: [32]u8 = undefined;
        @memcpy(&ir_sha256, try r.takeBytes(32));
        const code_len = try r.readU32();
        if (code_len > max_func_code_bytes) return error.OversizeField;
        const code_src = try r.takeBytes(code_len);
        const code = try allocator.dupe(u8, code_src);
        errdefer allocator.free(code);

        const patches_cnt = try r.readU32();
        if (patches_cnt > max_call_patches_per_func) return error.OversizeField;
        const patches = try allocator.alloc(FuncCallPatch, patches_cnt);
        errdefer allocator.free(patches);
        for (patches) |*p| {
            p.patch_offset = try r.readU32();
            p.target_func_idx = try r.readU32();
            // Reject patches that would write outside this function's
            // own code blob — the post-load assembler would otherwise
            // corrupt a neighbour function's bytes during reuse.
            if (@as(u64, p.patch_offset) + 4 > code_len) return error.InvalidPatchOffset;
            if (p.target_func_idx >= func_count) return error.InvalidTargetFuncIdx;
        }
        f.* = .{ .ir_sha256 = ir_sha256, .code = code, .call_patches = patches };
        initialised += 1;
    }

    if (r.pos != bytes.len) return error.Truncated;
    return .{
        .wamr_build_id = build_id,
        .target_arch = arch,
        .target_abi = abi,
        .module_epoch = module_epoch,
        .functions = funcs,
    };
}

fn appendU32(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, v: u32) !void {
    var tmp: [4]u8 = undefined;
    std.mem.writeInt(u32, &tmp, v, .little);
    try buf.appendSlice(allocator, &tmp);
}

const Reader = struct {
    buf: []const u8,
    pos: usize,

    fn takeBytes(self: *Reader, n: usize) DeserializeError![]const u8 {
        if (n > self.buf.len - self.pos) return error.Truncated;
        const s = self.buf[self.pos .. self.pos + n];
        self.pos += n;
        return s;
    }
    fn readU8(self: *Reader) DeserializeError!u8 {
        const s = try self.takeBytes(1);
        return s[0];
    }
    fn readU32(self: *Reader) DeserializeError!u32 {
        const s = try self.takeBytes(4);
        return std.mem.readInt(u32, s[0..4], .little);
    }
};

// ── Tests ──────────────────────────────────────────────────────────────────

const testing = std.testing;

fn buildAddFunc(allocator: std.mem.Allocator) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    errdefer func.deinit();
    const b = try func.newBlock();
    var blk = &func.blocks.items[b];
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try blk.append(.{ .op = .{ .iconst_32 = 3 }, .dest = v0 });
    try blk.append(.{ .op = .{ .iconst_32 = 4 }, .dest = v1 });
    try blk.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try blk.append(.{ .op = .{ .ret = v2 } });
    return func;
}

test "hashFunction: deterministic across two builds of equivalent IR" {
    var a = try buildAddFunc(testing.allocator);
    defer a.deinit();
    var b = try buildAddFunc(testing.allocator);
    defer b.deinit();
    try testing.expectEqual(hashFunction(&a), hashFunction(&b));
}

test "hashFunction: insensitive to debug-only fields (name)" {
    var a = try buildAddFunc(testing.allocator);
    defer a.deinit();
    var b = try buildAddFunc(testing.allocator);
    defer b.deinit();
    a.name = "alpha";
    b.name = "beta";
    try testing.expectEqual(hashFunction(&a), hashFunction(&b));
}

test "hashFunction: insensitive to next_vreg high-water mark" {
    var a = try buildAddFunc(testing.allocator);
    defer a.deinit();
    var b = try buildAddFunc(testing.allocator);
    defer b.deinit();
    // Allocating extra vregs without using them should not affect codegen.
    _ = b.newVReg();
    _ = b.newVReg();
    try testing.expectEqual(hashFunction(&a), hashFunction(&b));
}

test "hashFunction: insensitive to stale block predecessors" {
    var a = try buildAddFunc(testing.allocator);
    defer a.deinit();
    var b = try buildAddFunc(testing.allocator);
    defer b.deinit();
    try b.blocks.items[0].addPredecessor(999);
    try testing.expectEqual(hashFunction(&a), hashFunction(&b));
}

test "hashFunction: sensitive to a constant value change" {
    var a = try buildAddFunc(testing.allocator);
    defer a.deinit();
    var b = try buildAddFunc(testing.allocator);
    defer b.deinit();
    b.blocks.items[0].instructions.items[0].op = .{ .iconst_32 = 5 };
    try testing.expect(!std.mem.eql(u8, &hashFunction(&a), &hashFunction(&b)));
}

test "hashFunction: sensitive to local_count change" {
    var a = try buildAddFunc(testing.allocator);
    defer a.deinit();
    var b = try buildAddFunc(testing.allocator);
    defer b.deinit();
    b.local_count += 1;
    try testing.expect(!std.mem.eql(u8, &hashFunction(&a), &hashFunction(&b)));
}

test "hashFunction: sensitive to op-tag change (add → sub)" {
    var a = try buildAddFunc(testing.allocator);
    defer a.deinit();
    var b = try buildAddFunc(testing.allocator);
    defer b.deinit();
    const add_inst = b.blocks.items[0].instructions.items[2];
    const dest = add_inst.dest;
    const Bin = @TypeOf(add_inst.op.add);
    const lhs_rhs: Bin = add_inst.op.add;
    b.blocks.items[0].instructions.items[2] = .{ .op = .{ .sub = lhs_rhs }, .dest = dest };
    try testing.expect(!std.mem.eql(u8, &hashFunction(&a), &hashFunction(&b)));
}

test "hashFunction: SIMD lane-index sensitivity (i32x4_extract_lane)" {
    // Regression for the rubber-duck finding that `print.formatFunc`
    // would have collided on this case because it emits only the
    // operand vregs for extract-lane ops. The canonical hasher walks
    // the payload struct fields directly so distinct lane indices
    // produce distinct hashes.
    var a = ir.IrFunction.init(testing.allocator, 1, 1, 0);
    defer a.deinit();
    const b0 = try a.newBlock();
    const v_in = a.newVReg();
    const v_out = a.newVReg();
    try a.blocks.items[b0].append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = v_in, .lane = 0 } },
        .dest = v_out,
    });
    try a.blocks.items[b0].append(.{ .op = .{ .ret = v_out } });

    var b = ir.IrFunction.init(testing.allocator, 1, 1, 0);
    defer b.deinit();
    const c0 = try b.newBlock();
    const w_in = b.newVReg();
    const w_out = b.newVReg();
    try b.blocks.items[c0].append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = w_in, .lane = 2 } },
        .dest = w_out,
    });
    try b.blocks.items[c0].append(.{ .op = .{ .ret = w_out } });

    try testing.expect(!std.mem.eql(u8, &hashFunction(&a), &hashFunction(&b)));
}

test "hashFunction: SIMD memory-offset sensitivity (v128_load)" {
    var a = ir.IrFunction.init(testing.allocator, 1, 1, 0);
    defer a.deinit();
    const b0 = try a.newBlock();
    const vbase_a = a.newVReg();
    const vdest_a = a.newVReg();
    try a.blocks.items[b0].append(.{
        .op = .{ .v128_load = .{ .base = vbase_a, .offset = 0, .alignment = 16 } },
        .dest = vdest_a,
        .type = .v128,
    });
    try a.blocks.items[b0].append(.{ .op = .{ .ret = vdest_a } });

    var b = ir.IrFunction.init(testing.allocator, 1, 1, 0);
    defer b.deinit();
    const c0 = try b.newBlock();
    const vbase_b = b.newVReg();
    const vdest_b = b.newVReg();
    try b.blocks.items[c0].append(.{
        .op = .{ .v128_load = .{ .base = vbase_b, .offset = 16, .alignment = 16 } },
        .dest = vdest_b,
        .type = .v128,
    });
    try b.blocks.items[c0].append(.{ .op = .{ .ret = vdest_b } });

    try testing.expect(!std.mem.eql(u8, &hashFunction(&a), &hashFunction(&b)));
}

test "hashModuleEpoch: deterministic for identical inputs" {
    const ft = [_]ir.IrFuncType{.{ .params = &.{ .i32, .i32 }, .results = &.{.i32} }};
    const tidxs = [_]u32{0};
    const gtypes = [_]ir.IrType{ .i32, .i64 };
    const goffs = [_]u32{ 0, 8 };
    const e = ModuleEpochInputs{
        .wamr_build_id = "test-build-1",
        .target_arch = .x86_64,
        .target_abi = .x86_64_sysv,
        .import_count = 2,
        .global_types = &gtypes,
        .global_offsets = &goffs,
        .global_storage_size = 16,
        .func_types = &ft,
        .func_type_indices = &tidxs,
    };
    try testing.expectEqual(hashModuleEpoch(e), hashModuleEpoch(e));
}

test "hashModuleEpoch: sensitive to import_count" {
    const e1 = ModuleEpochInputs{
        .wamr_build_id = "test-build-1",
        .target_arch = .x86_64,
        .target_abi = .x86_64_sysv,
        .import_count = 2,
    };
    var e2 = e1;
    e2.import_count = 3;
    try testing.expect(!std.mem.eql(u8, &hashModuleEpoch(e1), &hashModuleEpoch(e2)));
}

test "hashModuleEpoch: sensitive to memory mode" {
    const e1 = ModuleEpochInputs{
        .wamr_build_id = "test-build-1",
        .target_arch = .aarch64,
        .target_abi = .aarch64_aapcs,
        .import_count = 0,
    };
    var memory64 = e1;
    memory64.has_memory64 = true;
    var shared = e1;
    shared.has_shared_memory = true;
    try testing.expect(!std.mem.eql(u8, &hashModuleEpoch(e1), &hashModuleEpoch(memory64)));
    try testing.expect(!std.mem.eql(u8, &hashModuleEpoch(e1), &hashModuleEpoch(shared)));
}

test "hashModuleEpoch: sensitive to target_abi (sysv vs win64)" {
    const e1 = ModuleEpochInputs{
        .wamr_build_id = "test-build-1",
        .target_arch = .x86_64,
        .target_abi = .x86_64_sysv,
        .import_count = 0,
    };
    var e2 = e1;
    e2.target_abi = .x86_64_win64;
    try testing.expect(!std.mem.eql(u8, &hashModuleEpoch(e1), &hashModuleEpoch(e2)));
}

test "hashModuleEpoch: sensitive to global_offsets contents" {
    const g1 = [_]u32{ 0, 8 };
    const g2 = [_]u32{ 0, 16 };
    const e1 = ModuleEpochInputs{
        .wamr_build_id = "test-build-1",
        .target_arch = .x86_64,
        .target_abi = .x86_64_sysv,
        .import_count = 0,
        .global_offsets = &g1,
    };
    var e2 = e1;
    e2.global_offsets = &g2;
    try testing.expect(!std.mem.eql(u8, &hashModuleEpoch(e1), &hashModuleEpoch(e2)));
}

test "hashModuleEpoch: epoch != hashFunction domain (separator works)" {
    // Cheap sanity check: an empty module epoch and an empty
    // (zero-block, zero-local) function should not produce the same
    // hash because the two hashers use distinct domain separators.
    var f = ir.IrFunction.init(testing.allocator, 0, 0, 0);
    defer f.deinit();
    const e = ModuleEpochInputs{
        .wamr_build_id = "",
        .target_arch = .x86_64,
        .target_abi = .x86_64_sysv,
        .import_count = 0,
    };
    try testing.expect(!std.mem.eql(u8, &hashFunction(&f), &hashModuleEpoch(e)));
}

// Serialiser round-trip and validation tests.

fn buildSampleCache(allocator: std.mem.Allocator) !Cache {
    const build_id = try allocator.dupe(u8, "wamr-test-0.1");
    errdefer allocator.free(build_id);
    var funcs = try allocator.alloc(CachedFunction, 2);
    errdefer allocator.free(funcs);
    funcs[0] = .{
        .ir_sha256 = .{1} ** 32,
        // 8 bytes of placeholder code so a patch at offset 0 has
        // 4 bytes of room (0+4 <= 8) per the deserialiser's bounds
        // check.
        .code = try allocator.dupe(u8, &[_]u8{ 0xCC, 0xC3, 0x00, 0x00, 0x90, 0x90, 0x90, 0x90 }),
        .call_patches = try allocator.dupe(FuncCallPatch, &.{
            .{ .patch_offset = 0, .target_func_idx = 1 },
        }),
    };
    errdefer {
        allocator.free(funcs[0].code);
        allocator.free(funcs[0].call_patches);
    }
    funcs[1] = .{
        .ir_sha256 = .{2} ** 32,
        .code = try allocator.dupe(u8, &[_]u8{ 0x90, 0x90, 0x90, 0x90, 0xC3 }),
        .call_patches = try allocator.dupe(FuncCallPatch, &.{}),
    };
    return .{
        .wamr_build_id = build_id,
        .target_arch = .x86_64,
        .target_abi = .x86_64_sysv,
        .module_epoch = .{7} ** 32,
        .functions = funcs,
    };
}

test "serialize/deserialize: round-trip preserves all fields" {
    const allocator = testing.allocator;
    var cache = try buildSampleCache(allocator);
    defer cache.deinit(allocator);
    const bytes = try serialize(&cache, allocator);
    defer allocator.free(bytes);
    var got = try deserialize(bytes, allocator);
    defer got.deinit(allocator);
    try testing.expectEqualSlices(u8, cache.wamr_build_id, got.wamr_build_id);
    try testing.expectEqual(cache.target_arch, got.target_arch);
    try testing.expectEqual(cache.target_abi, got.target_abi);
    try testing.expectEqualSlices(u8, &cache.module_epoch, &got.module_epoch);
    try testing.expectEqual(@as(usize, 2), got.functions.len);
    try testing.expectEqualSlices(u8, &cache.functions[0].ir_sha256, &got.functions[0].ir_sha256);
    try testing.expectEqualSlices(u8, cache.functions[0].code, got.functions[0].code);
    try testing.expectEqual(@as(usize, 1), got.functions[0].call_patches.len);
    try testing.expectEqual(cache.functions[0].call_patches[0].patch_offset, got.functions[0].call_patches[0].patch_offset);
    try testing.expectEqual(cache.functions[0].call_patches[0].target_func_idx, got.functions[0].call_patches[0].target_func_idx);
    try testing.expectEqualSlices(u8, cache.functions[1].code, got.functions[1].code);
    try testing.expectEqual(@as(usize, 0), got.functions[1].call_patches.len);
}

test "deserialize: rejects bad magic" {
    const allocator = testing.allocator;
    var cache = try buildSampleCache(allocator);
    defer cache.deinit(allocator);
    const bytes = try serialize(&cache, allocator);
    defer allocator.free(bytes);
    var mut = try allocator.dupe(u8, bytes);
    defer allocator.free(mut);
    mut[0] = 0xFF;
    try testing.expectError(error.BadMagic, deserialize(mut, allocator));
}

test "deserialize: rejects wrong version" {
    const allocator = testing.allocator;
    var cache = try buildSampleCache(allocator);
    defer cache.deinit(allocator);
    const bytes = try serialize(&cache, allocator);
    defer allocator.free(bytes);
    var mut = try allocator.dupe(u8, bytes);
    defer allocator.free(mut);
    // version is at offset 8 (after 8-byte magic).
    std.mem.writeInt(u32, mut[8..12], 999, .little);
    try testing.expectError(error.UnsupportedVersion, deserialize(mut, allocator));
}

test "deserialize: rejects truncated input" {
    const allocator = testing.allocator;
    var cache = try buildSampleCache(allocator);
    defer cache.deinit(allocator);
    const bytes = try serialize(&cache, allocator);
    defer allocator.free(bytes);
    try testing.expectError(error.Truncated, deserialize(bytes[0 .. bytes.len - 5], allocator));
}

test "deserialize: rejects out-of-range patch_offset" {
    const allocator = testing.allocator;
    // Build a tiny cache by hand whose first function has a patch
    // that points past the end of its code blob.
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    try buf.appendSlice(allocator, &cache_magic);
    try appendU32(&buf, allocator, cache_format_version);
    try appendU32(&buf, allocator, 0); // empty build id
    try buf.append(allocator, @intFromEnum(passes.TargetArch.x86_64));
    try buf.append(allocator, @intFromEnum(TargetAbi.x86_64_sysv));
    try buf.appendSlice(allocator, &[_]u8{0} ** 32); // module_epoch
    try appendU32(&buf, allocator, 1); // func_count
    try buf.appendSlice(allocator, &[_]u8{0} ** 32); // ir_sha256
    try appendU32(&buf, allocator, 4); // code_len = 4
    try buf.appendSlice(allocator, &[_]u8{ 0, 0, 0, 0 });
    try appendU32(&buf, allocator, 1); // patches_cnt
    try appendU32(&buf, allocator, 5); // patch_offset = 5 (> 4-4 = 0; 5+4=9 > 4)
    try appendU32(&buf, allocator, 0); // target_func_idx
    try testing.expectError(error.InvalidPatchOffset, deserialize(buf.items, allocator));
}

test "deserialize: rejects target_func_idx >= func_count" {
    const allocator = testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    try buf.appendSlice(allocator, &cache_magic);
    try appendU32(&buf, allocator, cache_format_version);
    try appendU32(&buf, allocator, 0);
    try buf.append(allocator, @intFromEnum(passes.TargetArch.x86_64));
    try buf.append(allocator, @intFromEnum(TargetAbi.x86_64_sysv));
    try buf.appendSlice(allocator, &[_]u8{0} ** 32);
    try appendU32(&buf, allocator, 1); // func_count = 1
    try buf.appendSlice(allocator, &[_]u8{0} ** 32);
    try appendU32(&buf, allocator, 8);
    try buf.appendSlice(allocator, &[_]u8{0} ** 8);
    try appendU32(&buf, allocator, 1);
    try appendU32(&buf, allocator, 0);
    try appendU32(&buf, allocator, 5); // target_func_idx = 5 >= func_count
    try testing.expectError(error.InvalidTargetFuncIdx, deserialize(buf.items, allocator));
}

test "deserialize: rejects invalid arch / abi bytes" {
    const allocator = testing.allocator;
    var cache = try buildSampleCache(allocator);
    defer cache.deinit(allocator);
    const bytes = try serialize(&cache, allocator);
    defer allocator.free(bytes);
    {
        var mut = try allocator.dupe(u8, bytes);
        defer allocator.free(mut);
        // Header layout:
        //   magic [8] + version [4] + build_id_len [4] + build_id [N]
        //   + arch [1] + abi [1] + module_epoch [32] + func_count [4] ...
        const arch_off = 8 + 4 + 4 + cache.wamr_build_id.len;
        mut[arch_off] = 99;
        try testing.expectError(error.InvalidTargetArch, deserialize(mut, allocator));
    }
    {
        var mut = try allocator.dupe(u8, bytes);
        defer allocator.free(mut);
        const abi_off = 8 + 4 + 4 + cache.wamr_build_id.len + 1;
        mut[abi_off] = 99;
        try testing.expectError(error.InvalidTargetAbi, deserialize(mut, allocator));
    }
}
