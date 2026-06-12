//! Component Model binary format loader.
//!
//! Parses a WebAssembly Component binary into the in-memory AST defined
//! in `types.zig`. Components use the same magic bytes as core modules
//! but a different layer field (0x01) and version (0x0d).
//!
//! Component sections can be interleaved (unlike core module sections which
//! have a strict ordering). Each section's definitions are appended to the
//! appropriate index space.

const std = @import("std");
const ctypes = @import("types.zig");
const core_types = @import("../runtime/common/types.zig");
const leb128_mod = @import("../shared/utils/leb128.zig");

pub const LoadError = error{
    InvalidMagic,
    InvalidVersion,
    UnexpectedEnd,
    InvalidSectionId,
    InvalidSectionSize,
    InvalidEncoding,
    UnsupportedFeature,
    OutOfMemory,
    Overflow,
    InvalidUtf8,
};

/// A streaming reader over the component binary.
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

    fn peekByte(self: *BinaryReader) LoadError!u8 {
        if (self.pos >= self.data.len) return error.UnexpectedEnd;
        return self.data[self.pos];
    }

    fn readBytes(self: *BinaryReader, n: usize) LoadError![]const u8 {
        if (self.pos + n > self.data.len) return error.UnexpectedEnd;
        const slice = self.data[self.pos .. self.pos + n];
        self.pos += n;
        return slice;
    }

    fn readU32(self: *BinaryReader) LoadError!u32 {
        const slice = self.data[self.pos..];
        const result = leb128_mod.readUnsigned(u32, slice) catch return error.UnexpectedEnd;
        self.pos += result.bytes_read;
        return result.value;
    }

    /// Read a signed LEB128 in `s33` form (component valtype discriminator).
    /// Non-negative values are type indices; negative values encode primitive
    /// valtypes and handle forms.
    fn readS33(self: *BinaryReader) LoadError!i64 {
        const slice = self.data[self.pos..];
        const result = leb128_mod.readSigned(i64, slice) catch |err| switch (err) {
            error.Overflow => return error.InvalidEncoding,
            error.UnexpectedEnd => return error.UnexpectedEnd,
        };
        // s33: value must fit in 33 signed bits.
        if (result.value < -(@as(i64, 1) << 32) or result.value >= (@as(i64, 1) << 32))
            return error.InvalidEncoding;
        self.pos += result.bytes_read;
        return result.value;
    }

    fn readFixedU32(self: *BinaryReader) LoadError!u32 {
        if (self.pos + 4 > self.data.len) return error.UnexpectedEnd;
        const val = std.mem.readInt(u32, self.data[self.pos..][0..4], .little);
        self.pos += 4;
        return val;
    }

    fn readName(self: *BinaryReader) LoadError![]const u8 {
        const len = try self.readU32();
        if (self.pos + len > self.data.len) return error.UnexpectedEnd;
        const name = self.data[self.pos .. self.pos + len];
        self.pos += len;
        // Validate UTF-8
        if (!std.unicode.utf8ValidateSlice(name)) return error.InvalidUtf8;
        return name;
    }
};

/// Component section IDs per the binary format spec.
const SectionId = enum(u8) {
    custom = 0,
    core_module = 1,
    core_instance = 2,
    core_type = 3,
    component = 4,
    instance = 5,
    alias = 6,
    type = 7,
    canon = 8,
    start = 9,
    @"import" = 10,
    @"export" = 11,
    value = 12,
};

/// Load a WebAssembly Component from binary data.
pub fn load(data: []const u8, allocator: std.mem.Allocator) LoadError!ctypes.Component {
    var reader = BinaryReader{ .data = data };

    // Validate preamble
    const magic = try reader.readFixedU32();
    if (magic != core_types.wasm_magic) return error.InvalidMagic;
    const version = try reader.readFixedU32();
    if (version != core_types.component_version) return error.InvalidVersion;

    // Collect sections into dynamic arrays
    var core_modules: std.ArrayListUnmanaged(ctypes.CoreModule) = .empty;
    var core_instances: std.ArrayListUnmanaged(ctypes.CoreInstanceExpr) = .empty;
    var core_type_defs: std.ArrayListUnmanaged(ctypes.CoreTypeDef) = .empty;
    var components: std.ArrayListUnmanaged(*ctypes.Component) = .empty;
    var instances: std.ArrayListUnmanaged(ctypes.InstanceExpr) = .empty;
    var aliases: std.ArrayListUnmanaged(ctypes.Alias) = .empty;
    var type_defs: std.ArrayListUnmanaged(ctypes.TypeDef) = .empty;
    var canons: std.ArrayListUnmanaged(ctypes.Canon) = .empty;
    var imports: std.ArrayListUnmanaged(ctypes.ImportDecl) = .empty;
    var exports: std.ArrayListUnmanaged(ctypes.ExportDecl) = .empty;
    var start: ?ctypes.Start = null;
    // Tracks the type index space in section-encounter order. Each entry
    // is the local idx into `type_defs` for that slot, or null when
    // the slot is consumed by an import (`.type`-bound) or alias whose
    // target type def we don't materialize. Required to resolve real
    // wasm32-wasip2 components where types and aliases interleave.
    var type_indexspace: std.ArrayListUnmanaged(?u32) = .empty;
    // Core-func index space contributors in binary declaration order.
    // Each canon that produces a core func and each `core(.func)` alias
    // appends one entry as it is parsed.
    var core_func_indexspace: std.ArrayListUnmanaged(ctypes.CoreFuncContributor) = .empty;
    // Component-instance index space contributors in binary
    // declaration order. Required for `wasm-tools compose` output
    // where instance and alias-of-instance sections interleave —
    // the legacy "imports then instances then aliases" walk in
    // `indexspace.resolveCompInstance` only handles non-interleaved
    // layouts (issue #355).
    var comp_instance_indexspace: std.ArrayListUnmanaged(ctypes.CompInstanceContributor) = .empty;
    // Parent type_indexspace slot contributed by each alias (or null
    // for aliases that don't add a type slot). Indexed by alias
    // position. Used by `resolveTopLevelTypeAliases` to back-fill
    // null slots produced by `(alias <inst> "<name>" (type …))` (#534).
    var alias_type_slot: std.ArrayListUnmanaged(?u32) = .empty;

    while (reader.remaining() > 0) {
        const section_id_byte = try reader.readByte();
        const section_size = try reader.readU32();

        const section_start = reader.pos;
        if (section_start + section_size > reader.data.len) return error.InvalidSectionSize;

        const section_id = std.enums.fromInt(SectionId, section_id_byte) orelse
            return error.InvalidSectionId;

        switch (section_id) {
            .custom => {
                // Skip custom sections
                reader.pos = section_start + section_size;
            },
            .core_module => {
                // The core module is stored as raw bytes (nested module binary)
                const module_data = reader.data[section_start .. section_start + section_size];
                try core_modules.append(allocator, .{ .data = module_data });
                reader.pos = section_start + section_size;
            },
            .core_instance => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    try core_instances.append(allocator, try parseCoreInstance(&reader, allocator));
                }
            },
            .core_type => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    try core_type_defs.append(allocator, try parseCoreType(&reader, allocator));
                }
            },
            .component => {
                // Nested component — recursively parse
                const comp_data = reader.data[section_start .. section_start + section_size];
                const child = try allocator.create(ctypes.Component);
                child.* = try load(comp_data, allocator);
                try components.append(allocator, child);
                reader.pos = section_start + section_size;
            },
            .instance => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    const local_idx: u32 = @intCast(instances.items.len);
                    try instances.append(allocator, try parseInstance(&reader, allocator));
                    try comp_instance_indexspace.append(allocator, .{ .instance = local_idx });
                }
            },
            .alias => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    const a = try parseAlias(&reader);
                    const local_idx: u32 = @intCast(aliases.items.len);
                    try aliases.append(allocator, a);
                    // Aliases of sort .type contribute a slot to the
                    // type indexspace (target unresolved here → null).
                    const sort: ctypes.Sort = switch (a) {
                        .instance_export => |ie| ie.sort,
                        .outer => |o| o.sort,
                    };
                    if (sort == .type) {
                        try alias_type_slot.append(allocator, @intCast(type_indexspace.items.len));
                        try type_indexspace.append(allocator, null);
                    } else {
                        try alias_type_slot.append(allocator, null);
                    }
                    // Aliases of sort .core(.func) contribute to the
                    // core-func indexspace.
                    const is_core_func = switch (sort) {
                        .core => |cs| cs == .func,
                        else => false,
                    };
                    if (is_core_func) try core_func_indexspace.append(allocator, .{ .alias = local_idx });
                    // Aliases of sort .instance contribute to the
                    // component-instance index space.
                    if (sort == .instance) {
                        try comp_instance_indexspace.append(allocator, .{ .alias = local_idx });
                    }
                }
            },
            .type => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    const local_idx: u32 = @intCast(type_defs.items.len);
                    try type_defs.append(allocator, try parseTypeDef(&reader, allocator));
                    try type_indexspace.append(allocator, local_idx);
                }
            },
            .canon => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    const local_idx: u32 = @intCast(canons.items.len);
                    const c = try parseCanon(&reader, allocator);
                    try canons.append(allocator, c);
                    // Every canon kind except `.lift` contributes a slot
                    // to the core-func indexspace.
                    const contributes = switch (c) {
                        .lower,
                        .resource_drop,
                        .resource_new,
                        .resource_rep,
                        .task_yield,
                        .context_get,
                        .context_set,
                        .task_return,
                        .async_canon,
                        => true,
                        .lift => false,
                    };
                    if (contributes) try core_func_indexspace.append(allocator, .{ .canon = local_idx });
                }
            },
            .start => {
                start = try parseStart(&reader, allocator);
            },
            .@"import" => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    const local_idx: u32 = @intCast(imports.items.len);
                    const imp = try parseImport(&reader);
                    try imports.append(allocator, imp);
                    if (imp.desc == .type) try type_indexspace.append(allocator, null);
                    // Instance-typed imports contribute to the
                    // component-instance index space (issue #355).
                    if (imp.desc == .instance) {
                        try comp_instance_indexspace.append(allocator, .{ .import = local_idx });
                    }
                }
            },
            .@"export" => {
                const count = try reader.readU32();
                var i: u32 = 0;
                while (i < count) : (i += 1) {
                    const local_idx: u32 = @intCast(exports.items.len);
                    const exp = try parseTopLevelExport(&reader);
                    try exports.append(allocator, exp);
                    // An `(export <name> (instance N))` decl contributes a
                    // new compinstance-indexspace slot that aliases the
                    // referenced instance (per Binary.md: every top-level
                    // export adds a fresh slot in its sort's indexspace).
                    // Required for binaries like wit-component 0.245's
                    // P3 components that intersperse two `wasi:cli/run`
                    // instance exports with an inner instance section
                    // between them — without this, downstream index
                    // lookups for the second export miss the slot
                    // contributed by the first export and either return
                    // null or land on the wrong instance. (#520)
                    if (exp.sort_idx) |si| {
                        if (si.sort == .instance) {
                            try comp_instance_indexspace.append(allocator, .{ .exported_alias = local_idx });
                        }
                    }
                }
            },
            .value => {
                // Value definitions — skip for now (gated feature)
                reader.pos = section_start + section_size;
            },
        }
        // Defensive: every typed-section parser above should have consumed
        // exactly `section_size` bytes. If a bug causes under- or over-read
        // we'd otherwise misalign the next section header.
        if (reader.pos != section_start + section_size) return error.InvalidSectionSize;
    }

    // Post-parse: resolve top-level type-aliases against imported instance
    // type bodies. Required for components that import a separate
    // `wasi:clocks/types@0.3.x` instance to expose `duration` (and
    // similar) shared types — without this, `canon.lower` lift/lower
    // of values whose declared type goes through such an alias trips
    // `CompoundNeedsRegistry`. See `resolveTopLevelTypeAliases` for the
    // full algorithm and limitations. (#534)
    try resolveTopLevelTypeAliases(
        allocator,
        aliases.items,
        alias_type_slot.items,
        imports.items,
        comp_instance_indexspace.items,
        &type_defs,
        &type_indexspace,
    );

    return .{
        .core_modules = try core_modules.toOwnedSlice(allocator),
        .core_instances = try core_instances.toOwnedSlice(allocator),
        .core_types = try core_type_defs.toOwnedSlice(allocator),
        .components = try components.toOwnedSlice(allocator),
        .instances = try instances.toOwnedSlice(allocator),
        .aliases = try aliases.toOwnedSlice(allocator),
        .types = try type_defs.toOwnedSlice(allocator),
        .type_indexspace = try type_indexspace.toOwnedSlice(allocator),
        .canons = try canons.toOwnedSlice(allocator),
        .start = start,
        .imports = try imports.toOwnedSlice(allocator),
        .exports = try exports.toOwnedSlice(allocator),
        .core_func_indexspace = try core_func_indexspace.toOwnedSlice(allocator),
        .comp_instance_indexspace = try comp_instance_indexspace.toOwnedSlice(allocator),
        .alias_type_slot = try alias_type_slot.toOwnedSlice(allocator),
    };
}

// ── Top-level type-alias resolution (#534) ─────────────────────────────────
//
// `(alias instance_export <inst> "<name>" (type …))` produces a parent
// `type_indexspace` slot whose target is the type exported by the
// imported instance under `<name>`. The on-wire encoding doesn't include
// the resolved type — we have to walk the imported instance's type body
// at load time to recover it.
//
// Scope: handles the case where `<inst>` resolves to a *direct* instance
// import whose type body's exported declarator has bound `eq <inner_idx>`,
// and `<inner_idx>` points at a `TypeDef.val` (primitive aliases like
// `type duration = u64`). Compound exported types — records, variants,
// resources — fall through unresolved (slot stays null); the affected
// `canon.lower` trampolines will continue to surface `CompoundNeedsRegistry`
// until a future slice extends this resolver to deep-copy compound
// payloads through the parent type space. The clock fixtures targeted
// by #534 only expose `type duration = u64` at the top level, so the
// primitive case is sufficient.
//
// Multi-hop alias chains (`alias-of-alias-of-import-export`) are
// supported via the fixed-point loop. The pass bails after a defensive
// bound on the number of iterations.
fn resolveTopLevelTypeAliases(
    allocator: std.mem.Allocator,
    aliases: []const ctypes.Alias,
    alias_type_slot: []const ?u32,
    imports: []const ctypes.ImportDecl,
    comp_instance_indexspace: []const ctypes.CompInstanceContributor,
    type_defs: *std.ArrayListUnmanaged(ctypes.TypeDef),
    type_indexspace: *std.ArrayListUnmanaged(?u32),
) LoadError!void {
    if (aliases.len == 0) return;
    var iteration: u32 = 0;
    while (iteration < 8) : (iteration += 1) {
        var any_change = false;
        for (aliases, 0..) |a, ai| {
            if (a != .instance_export) continue;
            const ie = a.instance_export;
            if (ie.sort != .type) continue;
            if (ai >= alias_type_slot.len) continue;
            const slot = alias_type_slot[ai] orelse continue;
            if (slot >= type_indexspace.items.len) continue;
            if (type_indexspace.items[slot] != null) continue;

            const resolved = resolveAliasInstanceExportType(
                ie.instance_idx,
                ie.name,
                imports,
                comp_instance_indexspace,
                type_defs.items,
                type_indexspace.items,
            ) orelse continue;

            // Materialize the resolved TypeDef in the parent's type
            // pool. Primitive `TypeDef.val` can be copied directly.
            //
            // For structural compounds (`.record` / `.variant` / etc.)
            // a primitive-only payload (no nested `.type_idx` / `.record`
            // / `.variant` / `.list` / ... refs) can also be copied
            // directly — its referenced ValTypes carry no source-body-
            // local indices. This covers `wasi:clocks/wall-clock`
            // `datetime = record { seconds: u64, nanoseconds: u32 }`
            // (the 0.2 shape) and `wasi:clocks/system-clock`
            // `record instant { seconds: s64, nanoseconds: u32 }` (the
            // 0.3 shape), which `wasi:filesystem` aliases via
            // `(alias outer 1 K (type))` for `new-timestamp`'s
            // `timestamp(datetime)` case payload. (`#571`.)
            //
            // Compound types whose nested references point at other
            // types in the source instance-type body (e.g.
            // `wasi:http/types` `error-code`, a variant whose
            // `HTTP-request-body-size(option<u64>)` cases reference a
            // separate `option<u64>` type) are deep-materialized: the
            // referenced types are copied into the parent pool with
            // fresh type-indexspace slots and the nested ValType indices
            // are remapped to those slots. Without this, `error-code`'s
            // align-8 `option<u64>` case is invisible across the
            // instance boundary and the canonical-ABI layout collapses
            // to align-4 (issue #814).
            switch (resolved) {
                .val => {
                    const new_idx: u32 = @intCast(type_defs.items.len);
                    try type_defs.append(allocator, resolved);
                    type_indexspace.items[slot] = new_idx;
                    any_change = true;
                },
                .record, .variant, .tuple, .flags, .enum_, .option, .result, .list => {
                    if (typeDefHasOnlyPrimitiveRefs(resolved)) {
                        const new_idx: u32 = @intCast(type_defs.items.len);
                        try type_defs.append(allocator, resolved);
                        type_indexspace.items[slot] = new_idx;
                        any_change = true;
                        continue;
                    }
                    // Deep path: materialize the export and its nested
                    // referenced types, remapping source-body-local
                    // ValType indices into fresh parent slots (#814).
                    const pos = (try materializeAliasInstanceExportTypeDeep(
                        allocator,
                        ie.instance_idx,
                        ie.name,
                        imports,
                        comp_instance_indexspace,
                        type_defs,
                        type_indexspace,
                    )) orelse continue;
                    type_indexspace.items[slot] = type_indexspace.items[pos];
                    any_change = true;
                },
                else => {},
            }
        }
        if (!any_change) return;
    }
}

/// Deep-materialize the type exported as `name` by the imported instance
/// referenced through `inst_ci_idx`, copying it and every type it
/// transitively references from the source instance-type body into the
/// parent `type_defs` pool. Each materialized type gets a fresh
/// `type_indexspace` slot so the canonical-ABI registry can resolve it,
/// and nested ValType indices (which are local to the source body) are
/// remapped onto those parent slots. Returns the parent type-indexspace
/// position of the exported type, or `null` when the chain can't be
/// followed (e.g. a nested ref goes through an unresolved alias) — in
/// which case the caller leaves the alias slot unresolved, exactly as
/// before. (#814.)
fn materializeAliasInstanceExportTypeDeep(
    allocator: std.mem.Allocator,
    inst_ci_idx: u32,
    name: []const u8,
    imports: []const ctypes.ImportDecl,
    comp_instance_indexspace: []const ctypes.CompInstanceContributor,
    type_defs: *std.ArrayListUnmanaged(ctypes.TypeDef),
    type_indexspace: *std.ArrayListUnmanaged(?u32),
) LoadError!?u32 {
    const decls = resolveAliasInstanceExportBody(
        inst_ci_idx,
        imports,
        comp_instance_indexspace,
        type_defs.items,
        type_indexspace.items,
    ) orelse return null;

    var slots = try buildInstanceBodyInnerSlots(allocator, decls);
    defer slots.deinit(allocator);

    const export_inner = slots.exportInnerIdx(name) orelse return null;

    var memo: std.AutoHashMapUnmanaged(u32, u32) = .empty;
    defer memo.deinit(allocator);

    return try materializeInnerType(
        allocator,
        slots,
        export_inner,
        type_defs,
        type_indexspace,
        &memo,
        0,
    );
}

/// Inner type-indexspace table for one instance-type body. `concrete`
/// holds the embedded `TypeDef` for `.type` slots (null for `.alias`
/// slots, which this resolver does not follow), `eq` holds the back-ref
/// target for `(export … (type (eq N)))` slots, and `export_names` maps
/// each type export's name to its slot. Slot numbering mirrors
/// `resolveInstanceTypeExportByName`: `.type`, `.alias`, and
/// `.export`-with-type decls each consume one slot.
const InstanceBodyInnerSlots = struct {
    concrete: []?ctypes.TypeDef,
    eq: []?u32,
    export_slot_of: []?u32,
    export_names: [][]const u8,

    fn deinit(self: *InstanceBodyInnerSlots, allocator: std.mem.Allocator) void {
        allocator.free(self.concrete);
        allocator.free(self.eq);
        allocator.free(self.export_slot_of);
        allocator.free(self.export_names);
    }

    /// Follow `eq` back-refs from `idx` to the slot that holds a
    /// concrete `TypeDef`. Returns null for unresolved (`.alias`) slots,
    /// out-of-range indices, or `eq` cycles.
    fn concreteIdx(self: InstanceBodyInnerSlots, idx: u32) ?u32 {
        var cur = idx;
        var hops: u32 = 0;
        while (hops <= self.concrete.len) : (hops += 1) {
            if (cur >= self.concrete.len) return null;
            if (self.eq[cur]) |t| {
                cur = t;
                continue;
            }
            if (self.concrete[cur] != null) return cur;
            return null;
        }
        return null;
    }

    /// Resolve the type export named `name` to the slot holding its
    /// concrete `TypeDef` (following the `eq` chain).
    fn exportInnerIdx(self: InstanceBodyInnerSlots, name: []const u8) ?u32 {
        for (self.export_names, 0..) |n, i| {
            if (std.mem.eql(u8, n, name)) {
                return self.concreteIdx(self.export_slot_of[i].?);
            }
        }
        return null;
    }
};

fn buildInstanceBodyInnerSlots(
    allocator: std.mem.Allocator,
    decls: []const ctypes.Decl,
) LoadError!InstanceBodyInnerSlots {
    var slot_count: u32 = 0;
    var export_count: u32 = 0;
    for (decls) |d| switch (d) {
        .type => slot_count += 1,
        .alias => slot_count += 1,
        .@"export" => |e| if (e.desc == .type) {
            slot_count += 1;
            export_count += 1;
        },
        else => {},
    };

    const concrete = try allocator.alloc(?ctypes.TypeDef, slot_count);
    errdefer allocator.free(concrete);
    const eq = try allocator.alloc(?u32, slot_count);
    errdefer allocator.free(eq);
    const export_slot_of = try allocator.alloc(?u32, export_count);
    errdefer allocator.free(export_slot_of);
    const export_names = try allocator.alloc([]const u8, export_count);
    errdefer allocator.free(export_names);
    @memset(concrete, null);
    @memset(eq, null);
    @memset(export_slot_of, null);

    var slot_i: u32 = 0;
    var export_i: u32 = 0;
    for (decls) |d| switch (d) {
        .type => |td| {
            concrete[slot_i] = td;
            slot_i += 1;
        },
        .alias => slot_i += 1,
        .@"export" => |e| if (e.desc == .type) {
            switch (e.desc.type) {
                .eq => |target| if (target < slot_i) {
                    eq[slot_i] = target;
                },
                .sub_resource => {},
            }
            export_names[export_i] = e.name;
            export_slot_of[export_i] = slot_i;
            export_i += 1;
            slot_i += 1;
        },
        else => {},
    };

    return .{
        .concrete = concrete,
        .eq = eq,
        .export_slot_of = export_slot_of,
        .export_names = export_names,
    };
}

/// Recursively copy the source-body type at inner slot `inner_idx` into
/// the parent pool, returning its fresh parent type-indexspace position.
/// `memo` maps an already-materialized concrete inner slot to its parent
/// position, breaking shared/recursive references. Returns null when a
/// nested reference can't be resolved. (#814.)
fn materializeInnerType(
    allocator: std.mem.Allocator,
    slots: InstanceBodyInnerSlots,
    inner_idx: u32,
    type_defs: *std.ArrayListUnmanaged(ctypes.TypeDef),
    type_indexspace: *std.ArrayListUnmanaged(?u32),
    memo: *std.AutoHashMapUnmanaged(u32, u32),
    depth: u32,
) LoadError!?u32 {
    if (depth > 64) return null;
    const cidx = slots.concreteIdx(inner_idx) orelse return null;
    if (memo.get(cidx)) |pos| return pos;

    const td = slots.concrete[cidx].?;

    // Reserve a parent slot before recursing so self/mutually-recursive
    // references resolve to this same position.
    const local: u32 = @intCast(type_defs.items.len);
    try type_defs.append(allocator, td);
    const pos: u32 = @intCast(type_indexspace.items.len);
    try type_indexspace.append(allocator, local);
    try memo.put(allocator, cidx, pos);

    const remapped = (try remapInstanceBodyTypeDef(
        allocator,
        slots,
        td,
        type_defs,
        type_indexspace,
        memo,
        depth,
    )) orelse return null;
    type_defs.items[local] = remapped;
    return pos;
}

fn remapInstanceBodyTypeDef(
    allocator: std.mem.Allocator,
    slots: InstanceBodyInnerSlots,
    td: ctypes.TypeDef,
    type_defs: *std.ArrayListUnmanaged(ctypes.TypeDef),
    type_indexspace: *std.ArrayListUnmanaged(?u32),
    memo: *std.AutoHashMapUnmanaged(u32, u32),
    depth: u32,
) LoadError!?ctypes.TypeDef {
    switch (td) {
        .val => |v| return .{ .val = (try remapInstanceBodyValType(allocator, slots, v, type_defs, type_indexspace, memo, depth)) orelse return null },
        .option => |o| return .{ .option = .{ .inner = (try remapInstanceBodyValType(allocator, slots, o.inner, type_defs, type_indexspace, memo, depth)) orelse return null } },
        .list => |l| return .{ .list = .{ .element = (try remapInstanceBodyValType(allocator, slots, l.element, type_defs, type_indexspace, memo, depth)) orelse return null } },
        .result => |r| {
            const ok: ?ctypes.ValType = if (r.ok) |t|
                ((try remapInstanceBodyValType(allocator, slots, t, type_defs, type_indexspace, memo, depth)) orelse return null)
            else
                null;
            const err: ?ctypes.ValType = if (r.err) |t|
                ((try remapInstanceBodyValType(allocator, slots, t, type_defs, type_indexspace, memo, depth)) orelse return null)
            else
                null;
            return .{ .result = .{ .ok = ok, .err = err } };
        },
        .record => |rec| {
            const new_fields = try allocator.alloc(ctypes.Field, rec.fields.len);
            for (rec.fields, 0..) |f, i| {
                new_fields[i] = .{
                    .name = f.name,
                    .type = (try remapInstanceBodyValType(allocator, slots, f.type, type_defs, type_indexspace, memo, depth)) orelse return null,
                };
            }
            return .{ .record = .{ .fields = new_fields } };
        },
        .tuple => |tup| {
            const new_fields = try allocator.alloc(ctypes.ValType, tup.fields.len);
            for (tup.fields, 0..) |f, i| {
                new_fields[i] = (try remapInstanceBodyValType(allocator, slots, f, type_defs, type_indexspace, memo, depth)) orelse return null;
            }
            return .{ .tuple = .{ .fields = new_fields } };
        },
        .variant => |v| {
            const new_cases = try allocator.alloc(ctypes.Case, v.cases.len);
            for (v.cases, 0..) |c, i| {
                new_cases[i] = .{
                    .name = c.name,
                    .type = if (c.type) |ct|
                        ((try remapInstanceBodyValType(allocator, slots, ct, type_defs, type_indexspace, memo, depth)) orelse return null)
                    else
                        null,
                    .refines = c.refines,
                };
            }
            return .{ .variant = .{ .cases = new_cases } };
        },
        // `.flags` / `.enum_` carry only name lists; `.resource` /
        // `.func` / `.component` / `.instance` carry no value-type refs
        // that participate in canonical-ABI layout here.
        else => return td,
    }
}

fn remapInstanceBodyValType(
    allocator: std.mem.Allocator,
    slots: InstanceBodyInnerSlots,
    vt: ctypes.ValType,
    type_defs: *std.ArrayListUnmanaged(ctypes.TypeDef),
    type_indexspace: *std.ArrayListUnmanaged(?u32),
    memo: *std.AutoHashMapUnmanaged(u32, u32),
    depth: u32,
) LoadError!?ctypes.ValType {
    const mapIdx = struct {
        fn call(
            a: std.mem.Allocator,
            s: InstanceBodyInnerSlots,
            inner: u32,
            tds: *std.ArrayListUnmanaged(ctypes.TypeDef),
            tis: *std.ArrayListUnmanaged(?u32),
            m: *std.AutoHashMapUnmanaged(u32, u32),
            d: u32,
        ) LoadError!?u32 {
            return materializeInnerType(a, s, inner, tds, tis, m, d + 1);
        }
    }.call;

    return switch (vt) {
        .record => |i| .{ .record = (try mapIdx(allocator, slots, i, type_defs, type_indexspace, memo, depth)) orelse return null },
        .variant => |i| .{ .variant = (try mapIdx(allocator, slots, i, type_defs, type_indexspace, memo, depth)) orelse return null },
        .list => |i| .{ .list = (try mapIdx(allocator, slots, i, type_defs, type_indexspace, memo, depth)) orelse return null },
        .tuple => |i| .{ .tuple = (try mapIdx(allocator, slots, i, type_defs, type_indexspace, memo, depth)) orelse return null },
        .flags => |i| .{ .flags = (try mapIdx(allocator, slots, i, type_defs, type_indexspace, memo, depth)) orelse return null },
        .enum_ => |i| .{ .enum_ = (try mapIdx(allocator, slots, i, type_defs, type_indexspace, memo, depth)) orelse return null },
        .option => |i| .{ .option = (try mapIdx(allocator, slots, i, type_defs, type_indexspace, memo, depth)) orelse return null },
        .result => |i| .{ .result = (try mapIdx(allocator, slots, i, type_defs, type_indexspace, memo, depth)) orelse return null },
        .type_idx => |i| .{ .type_idx = (try mapIdx(allocator, slots, i, type_defs, type_indexspace, memo, depth)) orelse return null },
        // `.own` / `.borrow` reference resource types (i32 handles whose
        // layout the registry never consults), and `.future` / `.stream`
        // / `.error_context` are likewise i32-shaped. Primitives carry no
        // index. None need remapping for canonical-ABI layout.
        else => vt,
    };
}

/// True iff `td` references only primitive ValTypes (no `.type_idx`
/// indirection into another indexspace, no nested `.record` / `.variant`
/// / `.list` / etc. structural refs). Used by `resolveTopLevelTypeAliases`
/// (#571) to gate which structural typedefs are safe to materialize
/// across instance-type-body boundaries without a remap pass — typically
/// `wasi:clocks/wall-clock` `datetime = record { seconds: u64,
/// nanoseconds: u32 }` and similar primitive-only records.
fn typeDefHasOnlyPrimitiveRefs(td: ctypes.TypeDef) bool {
    return switch (td) {
        .val => |v| isPrimitiveValType(v),
        .record => |r| blk: {
            for (r.fields) |f| if (!isPrimitiveValType(f.type)) break :blk false;
            break :blk true;
        },
        .tuple => |t| blk: {
            for (t.fields) |f| if (!isPrimitiveValType(f)) break :blk false;
            break :blk true;
        },
        .variant => |v| blk: {
            for (v.cases) |c| if (c.type) |ct| {
                if (!isPrimitiveValType(ct)) break :blk false;
            };
            break :blk true;
        },
        .option => |o| isPrimitiveValType(o.inner),
        .result => |r| blk: {
            if (r.ok) |t| if (!isPrimitiveValType(t)) break :blk false;
            if (r.err) |t| if (!isPrimitiveValType(t)) break :blk false;
            break :blk true;
        },
        .list => |l| isPrimitiveValType(l.element),
        // `.flags` / `.enum_` carry only name lists — always safe.
        .flags, .enum_ => true,
        else => false,
    };
}

fn isPrimitiveValType(vt: ctypes.ValType) bool {
    return switch (vt) {
        .bool,
        .s8,
        .u8,
        .s16,
        .u16,
        .s32,
        .u32,
        .s64,
        .u64,
        .f32,
        .f64,
        .char,
        .string,
        => true,
        else => false,
    };
}

/// Resolve `(alias <inst_ci_idx> "<name>" (type …))` to the exported
/// `TypeDef` of the imported instance, or `null` if the chain can't be
/// followed yet. See `resolveTopLevelTypeAliases` for scope.
fn resolveAliasInstanceExportType(
    inst_ci_idx: u32,
    name: []const u8,
    imports: []const ctypes.ImportDecl,
    comp_instance_indexspace: []const ctypes.CompInstanceContributor,
    type_defs: []const ctypes.TypeDef,
    type_indexspace: []const ?u32,
) ?ctypes.TypeDef {
    const inst_decls = resolveAliasInstanceExportBody(
        inst_ci_idx,
        imports,
        comp_instance_indexspace,
        type_defs,
        type_indexspace,
    ) orelse return null;

    // Walk the instance type body, building the inner type-indexspace
    // map and locating the named export.
    return resolveInstanceTypeExportByName(inst_decls, name);
}

/// Resolve the `(alias <inst_ci_idx> …)` instance reference to the
/// declarator list of the imported instance's type body, or `null` when
/// the chain can't be followed. Shared by `resolveAliasInstanceExportType`
/// (which then finds a named export) and the deep materializer (#814).
fn resolveAliasInstanceExportBody(
    inst_ci_idx: u32,
    imports: []const ctypes.ImportDecl,
    comp_instance_indexspace: []const ctypes.CompInstanceContributor,
    type_defs: []const ctypes.TypeDef,
    type_indexspace: []const ?u32,
) ?[]const ctypes.Decl {
    // 1. Resolve the instance reference to a direct import declarator.
    if (inst_ci_idx >= comp_instance_indexspace.len) return null;
    const contributor = comp_instance_indexspace[inst_ci_idx];
    const imp_idx: u32 = switch (contributor) {
        .import => |i| i,
        // Local instances / alias-of-instance / exported_alias chains
        // are unsupported here; the type bodies they reference are
        // resolved elsewhere (or not at all). Skip.
        else => return null,
    };
    if (imp_idx >= imports.len) return null;
    const imp = imports[imp_idx];
    const imp_type_idx: u32 = switch (imp.desc) {
        .instance => |ti| ti,
        else => return null,
    };

    // 2. Look up the import's instance-type body in `type_defs`,
    // honouring `type_indexspace` indirection when present.
    const local_idx: u32 = blk: {
        if (type_indexspace.len > 0) {
            if (imp_type_idx >= type_indexspace.len) return null;
            break :blk type_indexspace[imp_type_idx] orelse return null;
        }
        break :blk imp_type_idx;
    };
    if (local_idx >= type_defs.len) return null;
    const inst_td = type_defs[local_idx];
    return switch (inst_td) {
        .instance => |inst| inst.decls,
        else => null,
    };
}

/// Walk an instance-type body's declarator list, build an inner
/// type-indexspace ↔ inner `TypeDef` table, find the export named
/// `name`, resolve its bound, and return the corresponding `TypeDef`.
fn resolveInstanceTypeExportByName(
    decls: []const ctypes.Decl,
    name: []const u8,
) ?ctypes.TypeDef {
    // Inner type-indexspace, populated in declaration order. Each
    // entry is either an embedded TypeDef (`.type`), or a `.eq <slot>`
    // back-reference (introduced by an export-of-type declarator).
    // The actual storage uses a small fixed-size buffer to avoid an
    // allocation; instance type bodies in real fixtures rarely exceed
    // a few dozen entries.
    const max_inner_slots = 256;
    var inner_slots: [max_inner_slots]?ctypes.TypeDef = [_]?ctypes.TypeDef{null} ** max_inner_slots;
    var inner_eq: [max_inner_slots]?u32 = [_]?u32{null} ** max_inner_slots;
    var slot_count: u32 = 0;
    for (decls) |d| switch (d) {
        .type => |td| {
            if (slot_count >= max_inner_slots) return null;
            inner_slots[slot_count] = td;
            slot_count += 1;
        },
        .alias => {
            // Outer / instance-export inner aliases — not resolved
            // here. The clock fixtures the #534 pass targets don't
            // use these for top-level `type` exports.
            if (slot_count >= max_inner_slots) return null;
            slot_count += 1;
        },
        .@"export" => |e| {
            if (e.desc == .type) {
                if (slot_count >= max_inner_slots) return null;
                switch (e.desc.type) {
                    .eq => |target| {
                        if (target < slot_count) inner_eq[slot_count] = target;
                    },
                    .sub_resource => {},
                }
                if (std.mem.eql(u8, e.name, name)) {
                    // Found the requested export. Follow `eq` chain.
                    var cur: u32 = slot_count;
                    var hops: u32 = 0;
                    while (hops < max_inner_slots) : (hops += 1) {
                        if (inner_eq[cur]) |t| {
                            cur = t;
                            continue;
                        }
                        if (inner_slots[cur]) |td| return td;
                        return null;
                    }
                    return null;
                }
                slot_count += 1;
            }
        },
        else => {},
    };
    return null;
}

// ── Section parsers ─────────────────────────────────────────────────────────

fn parseCoreInstance(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.CoreInstanceExpr {
    const tag = try reader.readByte();
    switch (tag) {
        0x00 => {
            const module_idx = try reader.readU32();
            const arg_count = try reader.readU32();
            const args = try allocator.alloc(ctypes.CoreInstantiateArg, arg_count);
            for (args) |*arg| {
                arg.name = try reader.readName();
                const sort_byte = try reader.readByte();
                _ = sort_byte; // Must be 0x12 (instance sort)
                arg.instance_idx = try reader.readU32();
            }
            return .{ .instantiate = .{ .module_idx = module_idx, .args = args } };
        },
        0x01 => {
            const count = try reader.readU32();
            const exps = try allocator.alloc(ctypes.CoreInlineExport, count);
            for (exps) |*e| {
                e.name = try reader.readName();
                const sort = try reader.readByte();
                const idx = try reader.readU32();
                e.sort_idx = .{
                    .sort = std.enums.fromInt(ctypes.CoreSort, sort) orelse return error.InvalidEncoding,
                    .idx = idx,
                };
            }
            return .{ .exports = exps };
        },
        else => return error.InvalidEncoding,
    }
}

fn parseCoreType(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.CoreTypeDef {
    const tag = try reader.readByte();
    switch (tag) {
        0x60 => {
            // Core function type
            const param_count = try reader.readU32();
            const params = try allocator.alloc(ctypes.CoreValType, param_count);
            for (params) |*p| p.* = try readCoreValType(reader);
            const result_count = try reader.readU32();
            const results = try allocator.alloc(ctypes.CoreValType, result_count);
            for (results) |*r| r.* = try readCoreValType(reader);
            return .{ .func = .{ .params = params, .results = results } };
        },
        0x50 => {
            // Core module type
            const decl_count = try reader.readU32();
            var imp_list: std.ArrayListUnmanaged(ctypes.CoreImportDecl) = .empty;
            var exp_list: std.ArrayListUnmanaged(ctypes.CoreExportDecl) = .empty;
            var i: u32 = 0;
            while (i < decl_count) : (i += 1) {
                const decl_tag = try reader.readByte();
                switch (decl_tag) {
                    0x00 => {
                        // import
                        const mod = try reader.readName();
                        const name = try reader.readName();
                        const type_idx = try reader.readU32();
                        try imp_list.append(allocator, .{ .module = mod, .name = name, .type_idx = type_idx });
                    },
                    0x01 => {
                        // export
                        const name = try reader.readName();
                        const type_idx = try reader.readU32();
                        try exp_list.append(allocator, .{ .name = name, .type_idx = type_idx });
                    },
                    else => return error.InvalidEncoding,
                }
            }
            return .{ .module = .{
                .imports = try imp_list.toOwnedSlice(allocator),
                .exports = try exp_list.toOwnedSlice(allocator),
            } };
        },
        else => return error.InvalidEncoding,
    }
}

fn readCoreValType(reader: *BinaryReader) LoadError!ctypes.CoreValType {
    const b = try reader.readByte();
    return std.enums.fromInt(ctypes.CoreValType, b) orelse return error.InvalidEncoding;
}

fn parseInstance(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.InstanceExpr {
    const tag = try reader.readByte();
    switch (tag) {
        0x00 => {
            const comp_idx = try reader.readU32();
            const arg_count = try reader.readU32();
            const args = try allocator.alloc(ctypes.InstantiateArg, arg_count);
            for (args) |*arg| {
                arg.name = try reader.readName();
                arg.sort_idx = try readSortIdx(reader);
            }
            return .{ .instantiate = .{ .component_idx = comp_idx, .args = args } };
        },
        0x01 => {
            const count = try reader.readU32();
            const exps = try allocator.alloc(ctypes.InlineExport, count);
            for (exps) |*e| {
                // inlineexport ::= n:<exportname'> si:<sortidx>
                // exportname' carries a 0x00/0x01/0x02 prefix tag — must use readExternName.
                e.name = try readExternName(reader);
                e.sort_idx = try readSortIdx(reader);
            }
            return .{ .exports = exps };
        },
        else => return error.InvalidEncoding,
    }
}

fn readSortIdx(reader: *BinaryReader) LoadError!ctypes.SortIdx {
    const sort = try readSort(reader);
    const idx = try reader.readU32();
    return .{ .sort = sort, .idx = idx };
}

fn readSort(reader: *BinaryReader) LoadError!ctypes.Sort {
    const b = try reader.readByte();
    return switch (b) {
        0x00 => blk: {
            const cs = try reader.readByte();
            break :blk .{ .core = std.enums.fromInt(ctypes.CoreSort, cs) orelse return error.InvalidEncoding };
        },
        0x01 => .func,
        0x02 => .value,
        0x03 => .type,
        0x04 => .component,
        0x05 => .instance,
        else => error.InvalidEncoding,
    };
}

fn parseAlias(reader: *BinaryReader) LoadError!ctypes.Alias {
    const sort = try readSort(reader);
    const target_tag = try reader.readByte();
    switch (target_tag) {
        0x00 => {
            // alias export: instance export
            const instance_idx = try reader.readU32();
            const name = try reader.readName();
            return .{ .instance_export = .{ .sort = sort, .instance_idx = instance_idx, .name = name } };
        },
        0x01 => {
            // alias core export: core instance export
            const instance_idx = try reader.readU32();
            const name = try reader.readName();
            return .{ .instance_export = .{ .sort = sort, .instance_idx = instance_idx, .name = name } };
        },
        0x02 => {
            // outer alias
            const outer_count = try reader.readU32();
            const idx = try reader.readU32();
            return .{ .outer = .{ .sort = sort, .outer_count = outer_count, .idx = idx } };
        },
        else => return error.InvalidEncoding,
    }
}

fn parseTypeDef(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.TypeDef {
    // `deftype` starts with a tag byte that selects a compound type;
    // the remaining space (primvaltypes 0x64..0x7F, own 0x69, borrow 0x68,
    // and any non-negative typeidx encoded as signed-LEB) is a bare valtype.
    // Peek the first byte and dispatch without consuming if not recognized.
    //
    // The component-model spec gained the `0x43` async-functype tag
    // alongside the existing synchronous `0x40` functype encoding;
    // both share the same body grammar (paramlist + resultlist) and
    // differ only in whether the lifted function is invoked through
    // the async-lifted dispatch path. We dispatch both into the same
    // body parser and record the async-ness on the returned `FuncType`.
    // (#520 — required to load wasm32-wasip3 fixtures emitted by
    // wit-bindgen 0.45 / wit-component 0.245.)
    const tag = try reader.peekByte();
    return switch (tag) {
        0x72, 0x71, 0x70, 0x6F, 0x6E, 0x6D, 0x6B, 0x6A, 0x3F, 0x40, 0x41, 0x42, 0x43 => parseCompoundTypeDef(reader, allocator),
        else => .{ .val = try readValType(reader) },
    };
}

fn parseCompoundTypeDef(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.TypeDef {
    const tag = try reader.readByte();
    return switch (tag) {
        // Defined types
        0x72 => blk: {
            // record
            const count = try reader.readU32();
            const fields = try allocator.alloc(ctypes.Field, count);
            for (fields) |*f| {
                f.name = try reader.readName();
                f.type = try readValType(reader);
            }
            break :blk .{ .record = .{ .fields = fields } };
        },
        0x71 => blk: {
            // variant
            const count = try reader.readU32();
            const cases = try allocator.alloc(ctypes.Case, count);
            for (cases) |*c| {
                c.name = try reader.readName();
                const has_type = try reader.readByte();
                c.type = if (has_type != 0) try readValType(reader) else null;
                // Current spec: case ends with a trailing 0x00 byte. (Older
                // drafts used a `refines` u32; no longer emitted.)
                const trailer = try reader.readByte();
                if (trailer != 0x00) return error.InvalidEncoding;
                c.refines = null;
            }
            break :blk .{ .variant = .{ .cases = cases } };
        },
        0x70 => blk: {
            // list
            const elem = try readValType(reader);
            break :blk .{ .list = .{ .element = elem } };
        },
        0x6F => blk: {
            // tuple
            const count = try reader.readU32();
            const fields = try allocator.alloc(ctypes.ValType, count);
            for (fields) |*f| f.* = try readValType(reader);
            break :blk .{ .tuple = .{ .fields = fields } };
        },
        0x6E => blk: {
            // flags
            const count = try reader.readU32();
            const names = try allocator.alloc([]const u8, count);
            for (names) |*n| n.* = try reader.readName();
            break :blk .{ .flags = .{ .names = names } };
        },
        0x6D => blk: {
            // enum
            const count = try reader.readU32();
            const names = try allocator.alloc([]const u8, count);
            for (names) |*n| n.* = try reader.readName();
            break :blk .{ .enum_ = .{ .names = names } };
        },
        0x6B => blk: {
            // option
            const inner = try readValType(reader);
            break :blk .{ .option = .{ .inner = inner } };
        },
        0x6A => blk: {
            // result
            const has_ok = try reader.readByte();
            const ok = if (has_ok != 0) try readValType(reader) else null;
            const has_err = try reader.readByte();
            const err = if (has_err != 0) try readValType(reader) else null;
            break :blk .{ .result = .{ .ok = ok, .err = err } };
        },
        0x3F => blk: {
            // resource
            const has_dtor = try reader.readByte();
            const dtor = if (has_dtor != 0) try reader.readU32() else null;
            break :blk .{ .resource = .{ .destructor = dtor } };
        },
        0x40, 0x43 => blk: {
            // func type. Synchronous (`0x40`) and asynchronous (`0x43`)
            // share the body grammar; only the leading tag byte differs.
            // Current spec (2024-2025): paramlist is a bare vec<labelvaltype>,
            // resultlist is `0x00 valtype` (one result) | `0x01 0x00` (none).
            // See: <https://github.com/WebAssembly/component-model/blob/main/design/mvp/Binary.md#type-definitions>
            const is_async = (tag == 0x43);
            const param_count = try reader.readU32();
            const params = try allocator.alloc(ctypes.NamedValType, param_count);
            for (params) |*p| {
                p.name = try reader.readName();
                p.type = try readValType(reader);
            }
            const result_tag = try reader.readByte();
            const results: ctypes.FuncType.ResultList = switch (result_tag) {
                0x00 => .{ .unnamed = try readValType(reader) },
                0x01 => blk2: {
                    const zero = try reader.readByte();
                    if (zero != 0x00) return error.InvalidEncoding;
                    break :blk2 .none;
                },
                else => return error.InvalidEncoding,
            };
            break :blk .{ .func = .{ .params = params, .results = results, .is_async = is_async } };
        },
        0x41 => blk: {
            // component type
            const count = try reader.readU32();
            const decls = try allocator.alloc(ctypes.Decl, count);
            errdefer allocator.free(decls);
            for (decls) |*d| d.* = try parseDecl(reader, allocator, .component_type);
            break :blk .{ .component = .{ .decls = decls } };
        },
        0x42 => blk: {
            // instance type
            const count = try reader.readU32();
            const decls = try allocator.alloc(ctypes.Decl, count);
            errdefer allocator.free(decls);
            for (decls) |*d| d.* = try parseDecl(reader, allocator, .instance_type);
            break :blk .{ .instance = .{ .decls = decls } };
        },
        else => error.InvalidEncoding,
    };
}

/// Scope of a declarator list: which decl tags are legal.
const DeclScope = enum { component_type, instance_type };

/// Parse a single declarator inside a component-type or instance-type body.
///
/// See: <https://github.com/WebAssembly/component-model/blob/main/design/mvp/Binary.md#type-definitions>
fn parseDecl(
    reader: *BinaryReader,
    allocator: std.mem.Allocator,
    scope: DeclScope,
) LoadError!ctypes.Decl {
    const tag = try reader.readByte();
    return switch (tag) {
        0x00 => .{ .core_type = try parseCoreType(reader, allocator) },
        0x01 => .{ .type = try parseTypeDef(reader, allocator) },
        0x02 => .{ .alias = try parseAlias(reader) },
        0x03 => blk: {
            if (scope != .component_type) return error.InvalidEncoding;
            break :blk .{ .import = try parseImport(reader) };
        },
        0x04 => .{ .@"export" = try parseExport(reader) },
        else => error.InvalidEncoding,
    };
}

/// Decode a component-model `valtype`.
///
/// Encoded as a signed LEB128 in `s33` form. Non-negative values are type
/// indices into the component type-index space; negative values are
/// primitive valtypes or handle forms. The `own` / `borrow` variants are
/// followed by an unsigned LEB128 `typeidx`.
///
/// See: <https://github.com/WebAssembly/component-model/blob/main/design/mvp/Binary.md#type-definitions>
fn readValType(reader: *BinaryReader) LoadError!ctypes.ValType {
    const raw = try reader.readS33();
    if (raw >= 0) {
        return .{ .type_idx = @intCast(raw) };
    }
    // Negative: primitive / handle form. Single-byte signed-LEB negatives
    // reach us as values -1..-64; the spec assigns each to a byte-tag
    // which equals `0x80 + raw` (so -1 → 0x7F, -24 → 0x68). Larger negative
    // values can't possibly be a primitive code.
    if (raw < -64) return error.InvalidEncoding;
    const tag: u8 = @intCast(raw + 0x80);
    return switch (tag) {
        0x7F => .bool,
        0x7E => .s8,
        0x7D => .u8,
        0x7C => .s16,
        0x7B => .u16,
        0x7A => .s32,
        0x79 => .u32,
        0x78 => .s64,
        0x77 => .u64,
        0x76 => .f32,
        0x75 => .f64,
        0x74 => .char,
        0x73 => .string,
        0x69 => .{ .own = try reader.readU32() },
        0x68 => .{ .borrow = try reader.readU32() },
        0x64 => .error_context,
        0x66 => try readPayloadedValType(reader, .stream),
        0x65 => try readPayloadedValType(reader, .future),
        else => error.InvalidEncoding,
    };
}

/// Read `(stream|future) t?` per defvaltype 0x66/0x65.
///
/// Three legal encodings:
///   - `0x00`             → empty form: `(stream)` / `(future)`. Stored
///     with payload `STREAM_FUTURE_EMPTY`.
///   - `0x01 <typeidx>`   → `(stream T)` / `(future T)` where T is a
///     component type-indexspace entry. Stored as the raw typeidx.
///   - `0x01 <primvaltype>` → `(stream p)` / `(future p)` where p is a
///     primitive value type (e.g. `u8`). Stored with payload
///     `encodeStreamFuturePrimitiveByte(...)`. Required to load real
///     wasm32-wasip3 components emitted by wit-bindgen — `wasi:cli` 0.3
///     declares `stream<u8>` directly in the type section.
fn readPayloadedValType(reader: *BinaryReader, comptime kind: enum { future, stream }) LoadError!ctypes.ValType {
    const present = try reader.readByte();
    if (present == 0x00) {
        return switch (kind) {
            .future => .{ .future = ctypes.STREAM_FUTURE_EMPTY },
            .stream => .{ .stream = ctypes.STREAM_FUTURE_EMPTY },
        };
    }
    if (present != 0x01) return error.InvalidEncoding;
    const inner = try readValType(reader);
    const idx: u32 = switch (inner) {
        .type_idx => |i| i,
        // Primitive payloads — encode as a sentinel so downstream code
        // can recover the original primitive via `decodeStreamFutureInner`.
        .bool => ctypes.encodeStreamFuturePrimitiveByte(0x7F),
        .s8 => ctypes.encodeStreamFuturePrimitiveByte(0x7E),
        .u8 => ctypes.encodeStreamFuturePrimitiveByte(0x7D),
        .s16 => ctypes.encodeStreamFuturePrimitiveByte(0x7C),
        .u16 => ctypes.encodeStreamFuturePrimitiveByte(0x7B),
        .s32 => ctypes.encodeStreamFuturePrimitiveByte(0x7A),
        .u32 => ctypes.encodeStreamFuturePrimitiveByte(0x79),
        .s64 => ctypes.encodeStreamFuturePrimitiveByte(0x78),
        .u64 => ctypes.encodeStreamFuturePrimitiveByte(0x77),
        .f32 => ctypes.encodeStreamFuturePrimitiveByte(0x76),
        .f64 => ctypes.encodeStreamFuturePrimitiveByte(0x75),
        .char => ctypes.encodeStreamFuturePrimitiveByte(0x74),
        .string => ctypes.encodeStreamFuturePrimitiveByte(0x73),
        else => return error.InvalidEncoding,
    };
    return switch (kind) {
        .future => .{ .future = idx },
        .stream => .{ .stream = idx },
    };
}

fn parseCanon(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.Canon {
    const tag = try reader.readByte();
    return switch (tag) {
        0x00 => blk: {
            // canon lift
            const sub = try reader.readByte();
            if (sub != 0x00) return error.InvalidEncoding;
            const core_func_idx = try reader.readU32();
            const opts = try readCanonOpts(reader, allocator);
            const type_idx = try reader.readU32();
            break :blk .{ .lift = .{
                .core_func_idx = core_func_idx,
                .type_idx = type_idx,
                .opts = opts,
            } };
        },
        0x01 => blk: {
            // canon lower
            const sub = try reader.readByte();
            if (sub != 0x00) return error.InvalidEncoding;
            const func_idx = try reader.readU32();
            const opts = try readCanonOpts(reader, allocator);
            break :blk .{ .lower = .{ .func_idx = func_idx, .opts = opts } };
        },
        0x02 => .{ .resource_new = try reader.readU32() },
        0x03 => .{ .resource_drop = try reader.readU32() },
        0x04 => .{ .resource_rep = try reader.readU32() },
        0x05 => .{ .async_canon = .task_cancel },
        0x0a => try parseContextCanon(reader, .get),
        0x0b => try parseContextCanon(reader, .set),
        0x0c => blk: {
            // canon thread.yield (formerly task.yield) — Binary.md tag 0x0c
            // immediate is a single `cancel?` byte: 0x00 = plain, 0x01 = cancellable.
            const cancel = try reader.readByte();
            const cancellable = switch (cancel) {
                0x00 => false,
                0x01 => true,
                else => return error.InvalidEncoding,
            };
            break :blk .{ .task_yield = .{ .cancellable = cancellable } };
        },
        0x09 => blk: {
            // canon task.return rs:<resultlist> opts:<opts>. The resultlist
            // shares its encoding with `FuncType` results (see parseTypeDef
            // 0x40 above).
            const result_tag = try reader.readByte();
            const results: ctypes.FuncType.ResultList = switch (result_tag) {
                0x00 => .{ .unnamed = try readValType(reader) },
                0x01 => blk2: {
                    const zero = try reader.readByte();
                    if (zero != 0x00) return error.InvalidEncoding;
                    break :blk2 .none;
                },
                else => return error.InvalidEncoding,
            };
            const opts = try readCanonOpts(reader, allocator);
            break :blk .{ .task_return = .{ .results = results, .opts = opts } };
        },
        // ── Async ABI canon tags (#478 sub-PR 3) ────────────────────────────
        // All of these route through the unified `Canon.async_canon` union.
        0x06 => blk: {
            const async_byte = try parseAsyncByte(reader);
            break :blk .{ .async_canon = .{ .subtask_cancel = .{ .is_async = async_byte } } };
        },
        0x0d => .{ .async_canon = .subtask_drop },
        0x0e => .{ .async_canon = .{ .stream_new = .{ .type_idx = try reader.readU32() } } },
        0x0f => blk: {
            const ti = try reader.readU32();
            const opts = try readCanonOpts(reader, allocator);
            break :blk .{ .async_canon = .{ .stream_read = .{ .type_idx = ti, .opts = opts } } };
        },
        0x10 => blk: {
            const ti = try reader.readU32();
            const opts = try readCanonOpts(reader, allocator);
            break :blk .{ .async_canon = .{ .stream_write = .{ .type_idx = ti, .opts = opts } } };
        },
        0x11 => blk: {
            const ti = try reader.readU32();
            const is_async = try parseAsyncByte(reader);
            break :blk .{ .async_canon = .{ .stream_cancel_read = .{ .type_idx = ti, .is_async = is_async } } };
        },
        0x12 => blk: {
            const ti = try reader.readU32();
            const is_async = try parseAsyncByte(reader);
            break :blk .{ .async_canon = .{ .stream_cancel_write = .{ .type_idx = ti, .is_async = is_async } } };
        },
        0x13 => .{ .async_canon = .{ .stream_drop_readable = .{ .type_idx = try reader.readU32() } } },
        0x14 => .{ .async_canon = .{ .stream_drop_writable = .{ .type_idx = try reader.readU32() } } },
        0x15 => .{ .async_canon = .{ .future_new = .{ .type_idx = try reader.readU32() } } },
        0x16 => blk: {
            const ti = try reader.readU32();
            const opts = try readCanonOpts(reader, allocator);
            break :blk .{ .async_canon = .{ .future_read = .{ .type_idx = ti, .opts = opts } } };
        },
        0x17 => blk: {
            const ti = try reader.readU32();
            const opts = try readCanonOpts(reader, allocator);
            break :blk .{ .async_canon = .{ .future_write = .{ .type_idx = ti, .opts = opts } } };
        },
        0x18 => blk: {
            const ti = try reader.readU32();
            const is_async = try parseAsyncByte(reader);
            break :blk .{ .async_canon = .{ .future_cancel_read = .{ .type_idx = ti, .is_async = is_async } } };
        },
        0x19 => blk: {
            const ti = try reader.readU32();
            const is_async = try parseAsyncByte(reader);
            break :blk .{ .async_canon = .{ .future_cancel_write = .{ .type_idx = ti, .is_async = is_async } } };
        },
        0x1a => .{ .async_canon = .{ .future_drop_readable = .{ .type_idx = try reader.readU32() } } },
        0x1b => .{ .async_canon = .{ .future_drop_writable = .{ .type_idx = try reader.readU32() } } },
        0x1c => .{ .async_canon = .{ .error_context_new = .{ .opts = try readCanonOpts(reader, allocator) } } },
        0x1d => .{ .async_canon = .{ .error_context_debug_message = .{ .opts = try readCanonOpts(reader, allocator) } } },
        0x1e => .{ .async_canon = .error_context_drop },
        0x1f => .{ .async_canon = .waitable_set_new },
        0x20 => blk: {
            const cancellable = try parseCancelByte(reader);
            const mem = try reader.readU32();
            break :blk .{ .async_canon = .{ .waitable_set_wait = .{ .cancellable = cancellable, .memory = mem } } };
        },
        0x21 => blk: {
            const cancellable = try parseCancelByte(reader);
            const mem = try reader.readU32();
            break :blk .{ .async_canon = .{ .waitable_set_poll = .{ .cancellable = cancellable, .memory = mem } } };
        },
        0x22 => .{ .async_canon = .waitable_set_drop },
        0x23 => .{ .async_canon = .waitable_join },
        else => error.InvalidEncoding,
    };
}

/// Parse the immediate of `canon context.get v i` / `canon context.set v i`
/// (Binary.md tags 0x0a / 0x0b). The valtype byte is restricted to `i32`
/// in sub-PR 1; widen later as conformance grows.
fn parseContextCanon(reader: *BinaryReader, comptime kind: enum { get, set }) LoadError!ctypes.Canon {
    const val_byte = try reader.readByte();
    if (val_byte != @intFromEnum(ctypes.CoreValType.i32)) return error.InvalidEncoding;
    const slot = try reader.readU32();
    return switch (kind) {
        .get => .{ .context_get = .{ .val_type = ctypes.CoreValType.i32, .slot = slot } },
        .set => .{ .context_set = .{ .val_type = ctypes.CoreValType.i32, .slot = slot } },
    };
}

/// Parse the `async?` immediate byte: 0x00 → false, 0x01 → true, else
/// InvalidEncoding. Used by `subtask.cancel` and the stream/future
/// `cancel-*` tags. (#478 sub-PR 3.)
fn parseAsyncByte(reader: *BinaryReader) LoadError!bool {
    const b = try reader.readByte();
    return switch (b) {
        0x00 => false,
        0x01 => true,
        else => error.InvalidEncoding,
    };
}

/// Parse the `cancel?` immediate byte. Same encoding as `async?` but
/// different semantic name — kept distinct so the IR matches Binary.md
/// vocabulary at the leaf. (#478 sub-PR 3.)
fn parseCancelByte(reader: *BinaryReader) LoadError!bool {
    return parseAsyncByte(reader);
}

fn readCanonOpts(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError![]const ctypes.CanonOpt {
    const count = try reader.readU32();
    if (count == 0) return &.{};
    const opts = try allocator.alloc(ctypes.CanonOpt, count);
    for (opts) |*o| {
        const tag = try reader.readByte();
        o.* = switch (tag) {
            0x00 => .{ .string_encoding = .utf8 },
            0x01 => .{ .string_encoding = .utf16 },
            0x02 => .{ .string_encoding = .latin1_utf16 },
            0x03 => .{ .memory = try reader.readU32() },
            0x04 => .{ .realloc = try reader.readU32() },
            0x05 => .{ .post_return = try reader.readU32() },
            0x06 => .async_lift,
            0x07 => .{ .callback = try reader.readU32() },
            else => return error.InvalidEncoding,
        };
    }
    return opts;
}

fn parseStart(reader: *BinaryReader, allocator: std.mem.Allocator) LoadError!ctypes.Start {
    const func_idx = try reader.readU32();
    const arg_count = try reader.readU32();
    const args = try allocator.alloc(u32, arg_count);
    for (args) |*a| a.* = try reader.readU32();
    const results = try reader.readU32();
    return .{ .func_idx = func_idx, .args = args, .results = results };
}

fn readExternDesc(reader: *BinaryReader) LoadError!ctypes.ExternDesc {
    const tag = try reader.readByte();
    return switch (tag) {
        0x00 => blk: {
            const sub = try reader.readByte();
            if (sub != 0x11) return error.InvalidEncoding; // module sort
            break :blk .{ .module = try reader.readU32() };
        },
        0x01 => .{ .func = try reader.readU32() },
        0x02 => .{ .value = try readValType(reader) },
        0x03 => blk: {
            const bound_tag = try reader.readByte();
            break :blk .{ .type = switch (bound_tag) {
                0x00 => .{ .eq = try reader.readU32() },
                0x01 => .sub_resource,
                else => return error.InvalidEncoding,
            } };
        },
        0x04 => .{ .component = try reader.readU32() },
        0x05 => .{ .instance = try reader.readU32() },
        else => error.InvalidEncoding,
    };
}

/// Read an `importname'` / `exportname'` (identical grammar per spec):
///
///   importname' ::= 0x00 len:<u32> in:<importname>
///                 | 0x01 len:<u32> in:<importname>
///                 | 0x02 len:<u32> in:<importname> vs:<versionsuffix>
///
/// The prefix tag distinguishes plain names from annotated/versioned names.
/// For now we return the raw `in` bytes and swallow any `versionsuffix` —
/// the runtime only needs the interface name to match against host bindings.
fn readExternName(reader: *BinaryReader) LoadError![]const u8 {
    const prefix = try reader.readByte();
    if (prefix > 0x02) return error.InvalidEncoding;
    const name = try reader.readName();
    if (prefix == 0x02) {
        // versionsuffix ::= len:<u32> vs:<semversuffix> — skip over it.
        _ = try reader.readName();
    }
    return name;
}

fn parseImport(reader: *BinaryReader) LoadError!ctypes.ImportDecl {
    const name = try readExternName(reader);
    const desc = try readExternDesc(reader);
    return .{ .name = name, .desc = desc };
}

fn parseExport(reader: *BinaryReader) LoadError!ctypes.ExportDecl {
    // exportdecl (used inside component/instance type bodies):
    //   en:<exportname'> ed:<externdesc>
    const name = try readExternName(reader);
    const desc = try readExternDesc(reader);
    return .{ .name = name, .desc = desc };
}

/// Parse a top-level `export` entry from the export section:
///   export ::= en:<exportname'> si:<sortidx> ed?:<externdesc>?
///
/// Distinct from `parseExport` (the declarator form used inside
/// component/instance types), which has no sortidx and a mandatory descriptor.
fn parseTopLevelExport(reader: *BinaryReader) LoadError!ctypes.ExportDecl {
    const name = try readExternName(reader);
    const sort_idx = try readSortIdx(reader);
    const has_desc = try reader.readByte();
    const desc: ctypes.ExternDesc = switch (has_desc) {
        0x00 => inferExternDescFromSort(sort_idx),
        0x01 => try readExternDesc(reader),
        else => return error.InvalidEncoding,
    };
    return .{ .name = name, .desc = desc, .sort_idx = sort_idx };
}

/// When a top-level export omits its externdesc, the sortidx itself describes
/// the kind. For sorts that carry a type idx (func/component/instance) we
/// have no explicit type; fall back to a best-effort placeholder. The runtime
/// treats these as opaque until Phase 1B index-space resolution fills them in.
fn inferExternDescFromSort(si: ctypes.SortIdx) ctypes.ExternDesc {
    return switch (si.sort) {
        .func => .{ .func = 0 },
        .value => .{ .value = .{ .type_idx = 0 } },
        .type => .{ .type = .{ .eq = si.idx } },
        .component => .{ .component = 0 },
        .instance => .{ .instance = 0 },
        .core => .{ .module = 0 },
    };
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "load: minimal empty component" {
    const data = [_]u8{
        // magic
        0x00, 0x61, 0x73, 0x6D,
        // version=0x0d, layer=0x01
        0x0d, 0x00, 0x01, 0x00,
    };
    const comp = try load(&data, std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), comp.core_modules.len);
    try std.testing.expectEqual(@as(usize, 0), comp.imports.len);
    try std.testing.expectEqual(@as(usize, 0), comp.exports.len);
    try std.testing.expectEqual(@as(usize, 0), comp.types.len);
    try std.testing.expect(comp.start == null);
}

test "load: invalid magic returns error" {
    const data = [_]u8{ 0x00, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x01, 0x00 };
    try std.testing.expectError(error.InvalidMagic, load(&data, std.testing.allocator));
}

test "load: core module version returns error" {
    const data = [_]u8{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 };
    try std.testing.expectError(error.InvalidVersion, load(&data, std.testing.allocator));
}

test "readValType: primitive types" {
    var reader = BinaryReader{ .data = &[_]u8{ 0x7F, 0x7A, 0x73 } };
    const v1 = try readValType(&reader);
    try std.testing.expect(v1 == .bool);
    const v2 = try readValType(&reader);
    try std.testing.expect(v2 == .s32);
    const v3 = try readValType(&reader);
    try std.testing.expect(v3 == .string);
}

test "readValType: own/borrow with typeidx" {
    // own 3, borrow 5
    var reader = BinaryReader{ .data = &[_]u8{ 0x69, 0x03, 0x68, 0x05 } };
    const v1 = try readValType(&reader);
    try std.testing.expectEqual(@as(u32, 3), v1.own);
    const v2 = try readValType(&reader);
    try std.testing.expectEqual(@as(u32, 5), v2.borrow);
}

test "readValType: typeidx non-negative, single byte" {
    // 0 and 63 encode as single bytes 0x00 and 0x3F in signed-LEB.
    var reader = BinaryReader{ .data = &[_]u8{ 0x00, 0x3F } };
    const v1 = try readValType(&reader);
    try std.testing.expectEqual(@as(u32, 0), v1.type_idx);
    const v2 = try readValType(&reader);
    try std.testing.expectEqual(@as(u32, 63), v2.type_idx);
}

test "readValType: typeidx >= 64 requires multi-byte signed LEB" {
    // typeidx 64: signed-LEB `0xC0 0x00` (cont + value 64, trailing 0).
    // typeidx 128: signed-LEB `0x80 0x01`.
    // typeidx 8192: signed-LEB `0x80 0xC0 0x00` (trailing 0 to keep sign positive).
    var reader = BinaryReader{ .data = &[_]u8{ 0xC0, 0x00, 0x80, 0x01, 0x80, 0xC0, 0x00 } };
    const v64 = try readValType(&reader);
    try std.testing.expectEqual(@as(u32, 64), v64.type_idx);
    const v128 = try readValType(&reader);
    try std.testing.expectEqual(@as(u32, 128), v128.type_idx);
    const v8192 = try readValType(&reader);
    try std.testing.expectEqual(@as(u32, 8192), v8192.type_idx);
}

test "readValType: rejects unknown negative code" {
    // 0x67 decodes as signed LEB -25, which has no primitive mapping.
    var reader = BinaryReader{ .data = &[_]u8{0x67} };
    try std.testing.expectError(error.InvalidEncoding, readValType(&reader));
}

test "parseCanon: task.yield with cancel? = 0x00" {
    // tag 0x0c, cancel? 0x00 → task_yield with cancellable=false
    var reader = BinaryReader{ .data = &[_]u8{ 0x0c, 0x00 } };
    const c = try parseCanon(&reader, std.testing.allocator);
    try std.testing.expect(c == .task_yield);
    try std.testing.expectEqual(false, c.task_yield.cancellable);
}

test "parseCanon: task.yield with cancel? = 0x01 (cancellable)" {
    var reader = BinaryReader{ .data = &[_]u8{ 0x0c, 0x01 } };
    const c = try parseCanon(&reader, std.testing.allocator);
    try std.testing.expect(c == .task_yield);
    try std.testing.expectEqual(true, c.task_yield.cancellable);
}

test "parseCanon: task.yield rejects invalid cancel byte" {
    var reader = BinaryReader{ .data = &[_]u8{ 0x0c, 0x02 } };
    try std.testing.expectError(error.InvalidEncoding, parseCanon(&reader, std.testing.allocator));
}

test "parseCanon: context.get i32 slot=0" {
    // tag 0x0a, valtype byte for i32 = 0x7F, slot LEB = 0x00
    var reader = BinaryReader{ .data = &[_]u8{ 0x0a, 0x7F, 0x00 } };
    const c = try parseCanon(&reader, std.testing.allocator);
    try std.testing.expect(c == .context_get);
    try std.testing.expectEqual(ctypes.CoreValType.i32, c.context_get.val_type);
    try std.testing.expectEqual(@as(u32, 0), c.context_get.slot);
}

test "parseCanon: context.set i32 slot=1" {
    var reader = BinaryReader{ .data = &[_]u8{ 0x0b, 0x7F, 0x01 } };
    const c = try parseCanon(&reader, std.testing.allocator);
    try std.testing.expect(c == .context_set);
    try std.testing.expectEqual(@as(u32, 1), c.context_set.slot);
}

test "parseCanon: context.get rejects non-i32 valtype" {
    // tag 0x0a, valtype byte for i64 = 0x7E → rejected in sub-PR 1
    var reader = BinaryReader{ .data = &[_]u8{ 0x0a, 0x7E, 0x00 } };
    try std.testing.expectError(error.InvalidEncoding, parseCanon(&reader, std.testing.allocator));
}

test "parseCanon: context.set rejects non-i32 valtype" {
    var reader = BinaryReader{ .data = &[_]u8{ 0x0b, 0x7D, 0x00 } };
    try std.testing.expectError(error.InvalidEncoding, parseCanon(&reader, std.testing.allocator));
}

test "readCanonOpts: async_lift + callback opts (#478 sub-PR 2)" {
    // count=2, [0x06], [0x07 0x09]
    var reader = BinaryReader{ .data = &[_]u8{ 0x02, 0x06, 0x07, 0x09 } };
    const opts = try readCanonOpts(&reader, std.testing.allocator);
    defer std.testing.allocator.free(opts);
    try std.testing.expectEqual(@as(usize, 2), opts.len);
    try std.testing.expect(opts[0] == .async_lift);
    try std.testing.expect(opts[1] == .callback);
    try std.testing.expectEqual(@as(u32, 9), opts[1].callback);
}

test "parseCanon: task.return with one unnamed i32 result" {
    // tag 0x09, resultlist `0x00 0x7F` (unnamed i32 valtype), no opts (count=0)
    var reader = BinaryReader{ .data = &[_]u8{ 0x09, 0x00, 0x7F, 0x00 } };
    const c = try parseCanon(&reader, std.testing.allocator);
    try std.testing.expect(c == .task_return);
    try std.testing.expect(c.task_return.results == .unnamed);
    try std.testing.expectEqual(@as(usize, 0), c.task_return.opts.len);
}

test "parseCanon: task.return with no results" {
    // tag 0x09, resultlist `0x01 0x00`, no opts.
    var reader = BinaryReader{ .data = &[_]u8{ 0x09, 0x01, 0x00, 0x00 } };
    const c = try parseCanon(&reader, std.testing.allocator);
    try std.testing.expect(c == .task_return);
    try std.testing.expect(c.task_return.results == .none);
}

test "parseCanon: task.return rejects malformed resultlist" {
    var reader = BinaryReader{ .data = &[_]u8{ 0x09, 0x02, 0x7F, 0x00 } };
    try std.testing.expectError(error.InvalidEncoding, parseCanon(&reader, std.testing.allocator));
}

// ── Sub-PR 3: future / stream / error-context / waitable-set canon tags ──

test "readValType: error-context (0x64)" {
    var reader = BinaryReader{ .data = &[_]u8{0x64} };
    const v = try readValType(&reader);
    try std.testing.expect(v == .error_context);
}

test "readValType: future<u32> primitive payload" {
    // 0x65 0x01 0x79 (future + present + u32 primvaltype). Inner u32 is
    // a primvaltype; we encode it via the sentinel scheme so downstream
    // code can recover the primitive ValType. See loader.zig
    // `readPayloadedValType` for the encoding.
    var reader = BinaryReader{ .data = &[_]u8{ 0x65, 0x01, 0x79 } };
    const v = try readValType(&reader);
    try std.testing.expect(v == .future);
    const decoded = ctypes.decodeStreamFutureInner(v.future);
    try std.testing.expect(decoded == .primitive);
    try std.testing.expect(decoded.primitive == .u32);
}

test "readValType: stream with typeidx payload" {
    // typeidx 3 = 0x03 (signed LEB single byte).
    var reader = BinaryReader{ .data = &[_]u8{ 0x66, 0x01, 0x03 } };
    const v = try readValType(&reader);
    try std.testing.expect(v == .stream);
    try std.testing.expectEqual(@as(u32, 3), v.stream);
    const decoded = ctypes.decodeStreamFutureInner(v.stream);
    try std.testing.expect(decoded == .typeidx);
    try std.testing.expectEqual(@as(u32, 3), decoded.typeidx);
}

test "readValType: future with typeidx payload" {
    var reader = BinaryReader{ .data = &[_]u8{ 0x65, 0x01, 0x05 } };
    const v = try readValType(&reader);
    try std.testing.expect(v == .future);
    try std.testing.expectEqual(@as(u32, 5), v.future);
}

test "readValType: stream<u8> primitive payload (#537)" {
    // wit-bindgen emits `stream<u8>` directly in the type section for
    // wasi:cli@0.3 — see cli-stdio.wasm fixture. The primvaltype byte
    // tag for u8 is 0x7D; we encode via the sentinel scheme.
    var reader = BinaryReader{ .data = &[_]u8{ 0x66, 0x01, 0x7D } };
    const v = try readValType(&reader);
    try std.testing.expect(v == .stream);
    const decoded = ctypes.decodeStreamFutureInner(v.stream);
    try std.testing.expect(decoded == .primitive);
    try std.testing.expect(decoded.primitive == .u8);
}

test "readValType: stream empty form (#537)" {
    // `(stream)` with no element — encoded as `0x66 0x00`. Required by
    // some pure-signaling stream uses.
    var reader = BinaryReader{ .data = &[_]u8{ 0x66, 0x00 } };
    const v = try readValType(&reader);
    try std.testing.expect(v == .stream);
    const decoded = ctypes.decodeStreamFutureInner(v.stream);
    try std.testing.expect(decoded == .empty);
}

test "readValType: future empty form (#537)" {
    var reader = BinaryReader{ .data = &[_]u8{ 0x65, 0x00 } };
    const v = try readValType(&reader);
    try std.testing.expect(v == .future);
    const decoded = ctypes.decodeStreamFutureInner(v.future);
    try std.testing.expect(decoded == .empty);
}

test "parseCanon: subtask.cancel async?=0x01" {
    var reader = BinaryReader{ .data = &[_]u8{ 0x06, 0x01 } };
    const c = try parseCanon(&reader, std.testing.allocator);
    try std.testing.expect(c == .async_canon);
    try std.testing.expect(c.async_canon == .subtask_cancel);
    try std.testing.expectEqual(true, c.async_canon.subtask_cancel.is_async);
}

test "parseCanon: task.cancel (tag 0x05, no immediates)" {
    // Binary.md `canon task.cancel` is a single tag byte 0x05 with no
    // payload. Distinct from `subtask.cancel` at tag 0x06 which has
    // a one-byte `async?` immediate.
    var reader = BinaryReader{ .data = &[_]u8{0x05} };
    const c = try parseCanon(&reader, std.testing.allocator);
    try std.testing.expect(c == .async_canon);
    try std.testing.expect(c.async_canon == .task_cancel);
}

test "parseCanon: subtask.drop" {
    var reader = BinaryReader{ .data = &[_]u8{0x0d} };
    const c = try parseCanon(&reader, std.testing.allocator);
    try std.testing.expect(c == .async_canon);
    try std.testing.expect(c.async_canon == .subtask_drop);
}

test "parseCanon: stream.new with typeidx" {
    // tag 0x0e, typeidx u32 = 7
    var reader = BinaryReader{ .data = &[_]u8{ 0x0e, 0x07 } };
    const c = try parseCanon(&reader, std.testing.allocator);
    try std.testing.expect(c == .async_canon);
    try std.testing.expect(c.async_canon == .stream_new);
    try std.testing.expectEqual(@as(u32, 7), c.async_canon.stream_new.type_idx);
}

test "parseCanon: future.new with typeidx" {
    var reader = BinaryReader{ .data = &[_]u8{ 0x15, 0x02 } };
    const c = try parseCanon(&reader, std.testing.allocator);
    try std.testing.expect(c == .async_canon);
    try std.testing.expect(c.async_canon == .future_new);
}

test "parseCanon: stream.read with opts" {
    // tag 0x0f, typeidx=1, opts count=1, memory=0
    var reader = BinaryReader{ .data = &[_]u8{ 0x0f, 0x01, 0x01, 0x03, 0x00 } };
    const c = try parseCanon(&reader, std.testing.allocator);
    defer std.testing.allocator.free(c.async_canon.stream_read.opts);
    try std.testing.expect(c == .async_canon);
    try std.testing.expect(c.async_canon == .stream_read);
    try std.testing.expectEqual(@as(u32, 1), c.async_canon.stream_read.type_idx);
    try std.testing.expectEqual(@as(usize, 1), c.async_canon.stream_read.opts.len);
}

test "parseCanon: error-context.{new,drop,debug-message}" {
    // new with no opts
    var r1 = BinaryReader{ .data = &[_]u8{ 0x1c, 0x00 } };
    const c1 = try parseCanon(&r1, std.testing.allocator);
    try std.testing.expect(c1.async_canon == .error_context_new);
    // debug-message with no opts
    var r2 = BinaryReader{ .data = &[_]u8{ 0x1d, 0x00 } };
    const c2 = try parseCanon(&r2, std.testing.allocator);
    try std.testing.expect(c2.async_canon == .error_context_debug_message);
    // drop
    var r3 = BinaryReader{ .data = &[_]u8{0x1e} };
    const c3 = try parseCanon(&r3, std.testing.allocator);
    try std.testing.expect(c3.async_canon == .error_context_drop);
}

test "parseCanon: waitable-set.{new,wait,poll,drop} + waitable.join" {
    var r1 = BinaryReader{ .data = &[_]u8{0x1f} };
    try std.testing.expect((try parseCanon(&r1, std.testing.allocator)).async_canon == .waitable_set_new);
    // wait cancel?=0, memory=0
    var r2 = BinaryReader{ .data = &[_]u8{ 0x20, 0x00, 0x00 } };
    try std.testing.expect((try parseCanon(&r2, std.testing.allocator)).async_canon == .waitable_set_wait);
    // poll
    var r3 = BinaryReader{ .data = &[_]u8{ 0x21, 0x01, 0x00 } };
    const c3 = try parseCanon(&r3, std.testing.allocator);
    try std.testing.expect(c3.async_canon == .waitable_set_poll);
    try std.testing.expectEqual(true, c3.async_canon.waitable_set_poll.cancellable);
    // drop
    var r4 = BinaryReader{ .data = &[_]u8{0x22} };
    try std.testing.expect((try parseCanon(&r4, std.testing.allocator)).async_canon == .waitable_set_drop);
    // join
    var r5 = BinaryReader{ .data = &[_]u8{0x23} };
    try std.testing.expect((try parseCanon(&r5, std.testing.allocator)).async_canon == .waitable_join);
}

test "parseTypeDef: instance type with `sub resource` type decl" {
    // Mirrors the first type definition in every Rust wasm32-wasip2 component:
    //   (type (instance
    //     (type $p (sub resource))            ; decl 0x01, type, resource with no dtor
    //     (export "pollable" (type (eq $p)))  ; decl 0x04, export, type bound eq 0
    //   ))
    // Binary form:
    //   0x42              ; instance-type tag
    //   0x02              ; 2 decls
    //   0x01              ; decl 1: type
    //     0x3F 0x00       ; resource with no destructor
    //   0x04              ; decl 2: export
    //     0x00            ; exportname' prefix
    //     0x08 "pollable" ; name (len=8)
    //     0x03            ; externdesc: type
    //     0x00 0x00       ; bound: eq, typeidx 0
    const data = [_]u8{
        0x42, 0x02,
        0x01, 0x3F, 0x00,
        0x04, 0x00, 0x08, 'p', 'o', 'l', 'l', 'a', 'b', 'l', 'e',
        0x03, 0x00, 0x00,
    };
    var reader = BinaryReader{ .data = &data };
    const td = try parseTypeDef(&reader, std.testing.allocator);
    defer {
        // Free the decls slice (individual decls don't own heap beyond exports' names which are slices into `data`).
        std.testing.allocator.free(td.instance.decls);
    }
    try std.testing.expect(td == .instance);
    try std.testing.expectEqual(@as(usize, 2), td.instance.decls.len);
    try std.testing.expect(td.instance.decls[0] == .type);
    try std.testing.expect(td.instance.decls[0].type == .resource);
    try std.testing.expect(td.instance.decls[1] == .@"export");
    try std.testing.expectEqualStrings("pollable", td.instance.decls[1].@"export".name);
    try std.testing.expect(td.instance.decls[1].@"export".desc == .type);
    try std.testing.expectEqual(@as(u32, 0), td.instance.decls[1].@"export".desc.type.eq);
}

test "parseTypeDef: component type with import and alias decls" {
    // (type (component
    //   (import "x" (instance (type 0)))   ; decl 0x03, import
    //   (alias outer 0 0 (type))           ; decl 0x02, alias outer
    // ))
    const data = [_]u8{
        0x41, 0x02,
        0x03, 0x00, 0x01, 'x', 0x05, 0x00, // import name'=(0x00 "x") externdesc=instance type 0
        0x02, 0x03, 0x02, 0x00, 0x00, // alias: sort=type(0x03), outer(0x02), count=0, idx=0
    };
    var reader = BinaryReader{ .data = &data };
    const td = try parseTypeDef(&reader, std.testing.allocator);
    defer std.testing.allocator.free(td.component.decls);
    try std.testing.expect(td == .component);
    try std.testing.expectEqual(@as(usize, 2), td.component.decls.len);
    try std.testing.expect(td.component.decls[0] == .import);
    try std.testing.expectEqualStrings("x", td.component.decls[0].import.name);
    try std.testing.expect(td.component.decls[1] == .alias);
}

test "parseTypeDef: instance type rejects import decl" {
    // Instance types cannot contain import decls (0x03).
    const data = [_]u8{ 0x42, 0x01, 0x03 };
    var reader = BinaryReader{ .data = &data };
    try std.testing.expectError(error.InvalidEncoding, parseTypeDef(&reader, std.testing.allocator));
}

test "parseTypeDef: async functype 0x43 (#520)" {
    // The component-model spec gained an `0x43` deftype tag for
    // asynchronous functypes alongside the existing synchronous `0x40`.
    // Body shape (paramlist + resultlist) matches `0x40`; only the
    // `FuncType.is_async` flag distinguishes them. Required to load
    // wasm32-wasip3 components emitted by recent wit-bindgen versions.
    //
    // Encoding: `0x43` tag, paramcount=0, resultlist `0x00 valtype`
    // (one unnamed result), valtype = primitive `u32` (signed-LEB 0x79).
    const data = [_]u8{ 0x43, 0x00, 0x00, 0x79 };
    var reader = BinaryReader{ .data = &data };
    const td = try parseTypeDef(&reader, std.testing.allocator);
    defer switch (td) {
        .func => |ft| std.testing.allocator.free(ft.params),
        else => {},
    };
    try std.testing.expect(td == .func);
    try std.testing.expectEqual(true, td.func.is_async);
    try std.testing.expectEqual(@as(usize, 0), td.func.params.len);
    try std.testing.expect(td.func.results == .unnamed);
    try std.testing.expect(td.func.results.unnamed == .u32);
    try std.testing.expectEqual(data.len, reader.pos);
}

test "parseTypeDef: sync functype 0x40 has is_async=false (#520)" {
    // Same body as the 0x43 test above but with the synchronous tag;
    // confirms the new is_async flag defaults to false for legacy
    // encodings.
    const data = [_]u8{ 0x40, 0x00, 0x00, 0x79 };
    var reader = BinaryReader{ .data = &data };
    const td = try parseTypeDef(&reader, std.testing.allocator);
    defer switch (td) {
        .func => |ft| std.testing.allocator.free(ft.params),
        else => {},
    };
    try std.testing.expect(td == .func);
    try std.testing.expectEqual(false, td.func.is_async);
}

test "parseInstance: inline-export form expects exportname' (0x00 prefix) on each name" {
    // Regression test for `wabt component compose` interop:
    //
    // Component-model spec:
    //   instance       ::= 0x00 c arg* | 0x01 ie*
    //   inlineexport   ::= n:<exportname'> si:<sortidx>
    //   exportname'    ::= 0x00 en:<exportname>          ; tagged form
    //
    // wabt's `component compose` emits the inline-export form (tag 0x01)
    // for sub-component instantiation, while wasm-tools always uses the
    // `instantiate` form (tag 0x00). The bug fixed alongside this test
    // had `parseInstance` reading the inline-export name with bare
    // `readName` (vec(byte)), which mis-aligned into the sortidx and
    // surfaced as `error.InvalidSectionSize` from the section-end check.
    const data = [_]u8{
        // tag = 0x01 (inline-export form)
        0x01,
        // count of inline-exports = 1
        0x01,
        // exportname' = 0x00 prefix, len=3, "add"
        0x00, 0x03, 'a', 'd', 'd',
        // sortidx = sort=0x01 (func), idx=0
        0x01, 0x00,
    };
    var reader = BinaryReader{ .data = &data };
    const inst = try parseInstance(&reader, std.testing.allocator);
    defer std.testing.allocator.free(inst.exports);
    try std.testing.expect(inst == .exports);
    try std.testing.expectEqual(@as(usize, 1), inst.exports.len);
    try std.testing.expectEqualStrings("add", inst.exports[0].name);
    try std.testing.expect(inst.exports[0].sort_idx.sort == .func);
    try std.testing.expectEqual(@as(u32, 0), inst.exports[0].sort_idx.idx);
    // Reader must have consumed every byte — the section-end check in
    // `load` enforces this for the real call site.
    try std.testing.expectEqual(data.len, reader.pos);
}

test "load: real wasm32-wasip2 Rust component (stdio-echo)" {
    // Prebuilt binary of examples/stdio-echo/ — a minimal
    // Rust `fn main { println!("echo: ..."); }` compiled with
    // `cargo build --release --target wasm32-wasip2`. This is the canonical
    // Phase 1A regression fixture for #142: before the loader rework every
    // wasm32-wasip2 component failed at the first `type` section.
    const data = @embedFile("fixtures/stdio-echo.wasm");

    // The loader allocates many small slices but has no Component.deinit yet
    // (see #142 Phase 1B). Use an arena so the test doesn't leak.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const comp = try load(data, arena.allocator());

    // Verified via `wasm-tools component wit`: world `root` imports 13
    // wasi interfaces (io/poll, io/error, io/streams, cli/environment,
    // cli/exit, cli/stdin, cli/stdout, cli/stderr, and 5 cli/terminal-*),
    // and exports one: wasi:cli/run@0.2.0.
    try std.testing.expectEqual(@as(usize, 13), comp.imports.len);
    try std.testing.expectEqual(@as(usize, 1), comp.exports.len);

    // Every import is an instance import (WASI p2 pattern — never flat funcs).
    for (comp.imports) |imp| {
        try std.testing.expect(imp.desc == .instance);
    }
    // The sole export is the wasi:cli/run instance.
    try std.testing.expect(comp.exports[0].desc == .instance);
    try std.testing.expect(std.mem.startsWith(u8, comp.exports[0].name, "wasi:cli/run@"));

    // Spot-check a handful of imports against the golden list from wasm-tools.
    const expected = [_][]const u8{
        "wasi:io/poll@0.2.6",
        "wasi:io/streams@0.2.6",
        "wasi:cli/stdin@0.2.6",
        "wasi:cli/stdout@0.2.6",
        "wasi:cli/exit@0.2.6",
    };
    for (expected) |name| {
        var found = false;
        for (comp.imports) |imp| {
            if (std.mem.eql(u8, imp.name, name)) {
                found = true;
                break;
            }
        }
        std.testing.expect(found) catch |err| {
            return err;
        };
    }
}

test "loader #814: cross-instance use'd error-code resolves with align-8 layout" {
    // Regression for #814. The fixture mirrors `wasi:http`: a `types`
    // instance defines `error-code` (a variant whose `body-size`
    // case carries `option<u64>` → alignment 8), and an
    // `outgoing-handler` instance `use`s it via a top-level
    // `(alias export $types "error-code")` so its `handle` result is
    // `result<own<future-incoming-response>, error-code>`.
    //
    // Before the fix, `resolveTopLevelTypeAliases` refused to
    // materialize the variant across the instance boundary (its
    // `option<u64>` case is not primitive-only), leaving the alias
    // slot unresolved. `alignOfType` then fell back to 4, so the
    // `handle` ok payload landed at byte 4 instead of byte 8. The
    // deep-materialization path copies `error-code` *and* its nested
    // `option<u64>` into the parent type pool so the layout is align-8.
    const canon_abi = @import("canonical_abi.zig");
    const data = @embedFile("fixtures/http-error-code-align8.wasm");

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const comp = try load(data, arena.allocator());

    // Locate the component-level `(alias export <types> "error-code")`
    // type slot — the cross-instance `use` site.
    var ec_slot: ?u32 = null;
    for (comp.aliases, 0..) |a, ai| {
        if (a != .instance_export) continue;
        const ie = a.instance_export;
        if (ie.sort != .type) continue;
        if (!std.mem.eql(u8, ie.name, "error-code")) continue;
        if (ai < comp.alias_type_slot.len) ec_slot = comp.alias_type_slot[ai];
    }
    try std.testing.expect(ec_slot != null);

    const reg = canon_abi.TypeRegistry.init(&comp);

    // The alias slot must now resolve to the concrete variant.
    try std.testing.expect(reg.get(ec_slot.?) != null);

    // Canonical-ABI layout of `error-code`: alignment 8 (from its
    // `option<u64>` case), size 24 (disc 1 + pad 7 + option<u64> 16).
    const ec_vt = ctypes.ValType{ .variant = ec_slot.? };
    try std.testing.expectEqual(@as(u32, 8), canon_abi.alignOfType(reg, ec_vt));
    try std.testing.expectEqual(@as(u32, 24), canon_abi.sizeOfType(reg, ec_vt));

    // And therefore `result<own<…>, error-code>` places its ok payload
    // at byte 8: payload offset = alignUp(disc 1, max(align own=4,
    // align error-code=8)) = 8, matching wasmtime (#814). Before the
    // fix this collapsed to alignUp(1, max(4, 4)) = 4.
    const payload_align = @max(canon_abi.alignment(.{ .own = 0 }), canon_abi.alignOfType(reg, ec_vt));
    try std.testing.expectEqual(@as(u32, 8), canon_abi.alignUp(1, payload_align));
}

test "loader #571: typeDefHasOnlyPrimitiveRefs accepts primitive-only compounds" {
    const testing = std.testing;

    // `datetime` from `wasi:clocks/wall-clock` / `system-clock.instant`:
    // record { seconds: s64, nanoseconds: u32 } — primitives only.
    const datetime_fields = [_]ctypes.Field{
        .{ .name = "seconds", .type = .s64 },
        .{ .name = "nanoseconds", .type = .u32 },
    };
    const datetime_td: ctypes.TypeDef = .{ .record = .{ .fields = &datetime_fields } };
    try testing.expect(typeDefHasOnlyPrimitiveRefs(datetime_td));

    // Flags / enum — always safe (name lists, no nested ValType refs).
    const flags_names = [_][]const u8{ "a", "b" };
    const flags_td: ctypes.TypeDef = .{ .flags = .{ .names = &flags_names } };
    try testing.expect(typeDefHasOnlyPrimitiveRefs(flags_td));

    const enum_names = [_][]const u8{ "x", "y" };
    const enum_td: ctypes.TypeDef = .{ .enum_ = .{ .names = &enum_names } };
    try testing.expect(typeDefHasOnlyPrimitiveRefs(enum_td));

    // option<u64> and result<u32, string> are primitive-only.
    const opt_td: ctypes.TypeDef = .{ .option = .{ .inner = .u64 } };
    try testing.expect(typeDefHasOnlyPrimitiveRefs(opt_td));

    const res_td: ctypes.TypeDef = .{ .result = .{ .ok = .u32, .err = .string } };
    try testing.expect(typeDefHasOnlyPrimitiveRefs(res_td));
}

test "loader #571: typeDefHasOnlyPrimitiveRefs rejects records with structural ref fields" {
    const testing = std.testing;

    // Record whose field is `.type_idx = 7` — a parent indexspace
    // reference. `resolveTopLevelTypeAliases` must NOT auto-materialize
    // these because the index is local to the source instance-type body
    // and would mean nothing in the parent's pool.
    const fields = [_]ctypes.Field{
        .{ .name = "inner", .type = .{ .type_idx = 7 } },
    };
    const td: ctypes.TypeDef = .{ .record = .{ .fields = &fields } };
    try testing.expect(!typeDefHasOnlyPrimitiveRefs(td));

    // Record whose field is a structural `.record` ref by index — also
    // unsafe to direct-materialize without remapping.
    const fields2 = [_]ctypes.Field{
        .{ .name = "nested", .type = .{ .record = 3 } },
    };
    const td2: ctypes.TypeDef = .{ .record = .{ .fields = &fields2 } };
    try testing.expect(!typeDefHasOnlyPrimitiveRefs(td2));
}

test "loader #571: isPrimitiveValType discriminates primitives vs compound refs" {
    const testing = std.testing;

    try testing.expect(isPrimitiveValType(.bool));
    try testing.expect(isPrimitiveValType(.u8));
    try testing.expect(isPrimitiveValType(.s8));
    try testing.expect(isPrimitiveValType(.u32));
    try testing.expect(isPrimitiveValType(.s64));
    try testing.expect(isPrimitiveValType(.u64));
    try testing.expect(isPrimitiveValType(.f32));
    try testing.expect(isPrimitiveValType(.f64));
    try testing.expect(isPrimitiveValType(.char));
    try testing.expect(isPrimitiveValType(.string));

    try testing.expect(!isPrimitiveValType(.{ .record = 0 }));
    try testing.expect(!isPrimitiveValType(.{ .variant = 0 }));
    try testing.expect(!isPrimitiveValType(.{ .type_idx = 0 }));
    try testing.expect(!isPrimitiveValType(.{ .list = 0 }));
    try testing.expect(!isPrimitiveValType(.{ .own = 0 }));
}
