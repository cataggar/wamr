//! Core WebAssembly types used throughout the runtime.

const std = @import("std");
const config = @import("config");
const platform = @import("../../platform/platform.zig");
const shared_memory = @import("shared_memory.zig");
const parking_lot = @import("../../platform/parking_lot.zig");
const stable_resource = @import("../../shared/stable_resource.zig");
const execution_context = @import("execution_context.zig");

/// WebAssembly value types (§2.3.1)
pub const ValType = enum(u8) {
    i32 = 0x7F,
    i64 = 0x7E,
    f32 = 0x7D,
    f64 = 0x7C,
    v128 = 0x7B,
    funcref = 0x70,
    externref = 0x6F,
    anyref = 0x6E,
    eqref = 0x6D,
    i31ref = 0x6C,
    structref = 0x6B,
    arrayref = 0x6A,
    exnref = 0x69,
    nullref = 0x65,
    nonfuncref = 0x14,
    nonexternref = 0x15,

    pub fn isNumeric(self: ValType) bool {
        return switch (self) {
            .i32, .i64, .f32, .f64 => true,
            else => false,
        };
    }

    pub fn isVector(self: ValType) bool {
        return self == .v128;
    }

    pub fn isRef(self: ValType) bool {
        return switch (self) {
            .funcref, .externref, .anyref, .eqref, .i31ref, .structref, .arrayref, .exnref, .nullref, .nonfuncref, .nonexternref => true,
            else => false,
        };
    }

    pub fn isFuncRef(self: ValType) bool {
        return self == .funcref or self == .nonfuncref;
    }

    pub fn isExternRef(self: ValType) bool {
        return self == .externref or self == .nonexternref;
    }

    /// Non-nullable types are subtypes of their nullable counterparts.
    /// Also implements GC type hierarchy subtyping.
    pub fn isSubtypeOf(self: ValType, other: ValType) bool {
        if (self == other) return true;
        // non-nullable → nullable
        if (self == .nonfuncref and other == .funcref) return true;
        if (self == .nonexternref and other == .externref) return true;
        // nullref (none) is the bottom type — subtype of all internal ref types
        if (self == .nullref) return other == .anyref or other == .eqref or other == .i31ref or
            other == .structref or other == .arrayref or other == .funcref or other == .nonfuncref;
        // GC type hierarchy: i31ref/structref/arrayref <: eqref <: anyref
        if (other == .anyref) return self == .eqref or self == .i31ref or self == .structref or self == .arrayref;
        if (other == .eqref) return self == .i31ref or self == .structref or self == .arrayref;
        return false;
    }

    /// Map non-nullable ref types to their nullable equivalents.
    pub fn toNullable(self: ValType) ValType {
        return switch (self) {
            .nonfuncref => .funcref,
            .nonexternref => .externref,
            else => self,
        };
    }

    pub fn byteSize(self: ValType) usize {
        return switch (self) {
            .i32, .f32 => 4,
            .i64, .f64 => 8,
            .v128 => 16,
            .funcref, .externref, .anyref, .eqref, .i31ref, .structref, .arrayref, .exnref, .nullref, .nonfuncref, .nonexternref => @sizeOf(usize),
        };
    }
};

/// WebAssembly runtime value
pub const Value = union(ValType) {
    i32: i32,
    i64: i64,
    f32: f32,
    f64: f64,
    v128: u128,
    funcref: ?u32,
    externref: ?u32,
    anyref: ?u32,
    eqref: ?u32,
    i31ref: ?u32,
    structref: ?u32,
    arrayref: ?u32,
    /// Exception reference — index into ExecEnv.exception_refs pool.
    exnref: ?u32,
    nullref: ?u32,
    nonfuncref: ?u32,
    nonexternref: ?u32,
};

/// Function type (§2.3.5), also used as placeholder for struct/array types.
pub const FuncType = struct {
    params: []const ValType,
    results: []const ValType,
    /// Concrete type indices parallel to params (0xFFFFFFFF = abstract).
    param_tidxs: []const u32 = &.{},
    /// Concrete type indices parallel to results (0xFFFFFFFF = abstract).
    result_tidxs: []const u32 = &.{},
    /// Type kind (func/struct/array) for iso-recursive equivalence.
    kind: Kind = .func,
    /// For struct types: field type indices (for equivalence comparison).
    /// For array types: single element type index (len=1).
    field_tidxs: []const u32 = &.{},
    /// For struct/array: field value types (parallel to field_tidxs).
    field_types: []const ValType = &.{},
    /// For struct/array: field mutability (0=immutable, 1=mutable).
    field_muts: []const u8 = &.{},
    /// Declared supertype index (0xFFFFFFFF = none).
    supertype_idx: u32 = 0xFFFFFFFF,
    /// Whether this type is declared `final` (cannot be subtyped).
    is_final: bool = false,
    /// Recursive group size (1 = singleton/implicit group).
    rec_group_size: u16 = 1,
    /// Position of this type within its recursive group (0-based).
    rec_group_position: u16 = 0,

    pub const Kind = enum(u2) { func, struct_, array };
};

/// Limits (§2.3.4) — uses u64 to support memory64 proposal.
pub const Limits = struct {
    min: u64,
    max: ?u64 = null,
};

/// Table type (§2.3.6)
pub const TableType = struct {
    elem_type: ValType,
    limits: Limits,
    /// Concrete type index for elem_type (0xFFFFFFFF = abstract).
    elem_tidx: u32 = 0xFFFFFFFF,
    /// Init expression for table elements (from 0x40 prefix encoding).
    init_expr: ?InitExpr = null,
    /// Whether this table uses 64-bit addressing (table64 proposal).
    is_table64: bool = false,
};

/// Memory type (§2.3.7)
pub const MemoryType = struct {
    limits: Limits,
    is_shared: bool = false,
    is_memory64: bool = false,
};

/// Global type (§2.3.8)
pub const GlobalType = struct {
    val_type: ValType,
    mutability: Mutability,
    /// Concrete type index for val_type (0xFFFFFFFF = abstract).
    type_idx: u32 = 0xFFFFFFFF,

    pub const Mutability = enum(u1) {
        immutable = 0,
        mutable = 1,
    };

    /// Check if an exported global type is compatible with an imported global type.
    /// For mutable globals: types must match exactly (invariant).
    /// For immutable globals: export type must be a subtype of import type (covariant).
    pub fn importMatches(exp: GlobalType, imp: GlobalType) bool {
        if (exp.mutability != imp.mutability) return false;
        if (imp.mutability == .mutable) {
            return exp.val_type == imp.val_type and exp.type_idx == imp.type_idx;
        }
        // Covariant: export subtype of import
        if (exp.val_type == imp.val_type) {
            if (imp.type_idx == 0xFFFFFFFF) return true;
            return exp.type_idx == imp.type_idx;
        }
        if (exp.val_type == .nonfuncref and imp.val_type == .funcref) {
            if (imp.type_idx == 0xFFFFFFFF) return true;
            return exp.type_idx == imp.type_idx;
        }
        if (exp.val_type == .nonexternref and imp.val_type == .externref) {
            if (imp.type_idx == 0xFFFFFFFF) return true;
            return exp.type_idx == imp.type_idx;
        }
        return false;
    }
};

/// Import/Export kinds (§2.5)
pub const ExternalKind = enum(u8) {
    function = 0x00,
    table = 0x01,
    memory = 0x02,
    global = 0x03,
    tag = 0x04,
};

/// Wasm section IDs
pub const SectionId = enum(u8) {
    custom = 0,
    type = 1,
    import = 2,
    function = 3,
    table = 4,
    memory = 5,
    global = 6,
    @"export" = 7,
    start = 8,
    element = 9,
    code = 10,
    data = 11,
    data_count = 12,
    tag = 13,
};

/// Wasm binary magic number
pub const wasm_magic: u32 = 0x6d736100; // "\0asm"

/// Core module binary version (version=1, layer=0)
pub const wasm_version: u32 = 0x01;

/// Component binary version and layer (version=0x0d, layer=0x01).
/// Encoded as: version(u16 LE) ++ layer(u16 LE) = 0x0d 0x00 0x01 0x00.
pub const component_version: u32 = 0x0001_000d;

/// AOT binary magic number
pub const aot_magic: u32 = 0x746f6100; // "\0aot"

/// AOT format version
pub const aot_version: u32 = 9;

test "ValType: numeric classification" {
    try std.testing.expect(ValType.i32.isNumeric());
    try std.testing.expect(ValType.f64.isNumeric());
    try std.testing.expect(!ValType.funcref.isNumeric());
    try std.testing.expect(!ValType.v128.isNumeric());
}

test "ValType: byte sizes" {
    try std.testing.expectEqual(@as(usize, 4), ValType.i32.byteSize());
    try std.testing.expectEqual(@as(usize, 8), ValType.i64.byteSize());
    try std.testing.expectEqual(@as(usize, 16), ValType.v128.byteSize());
}

test "SectionId: ordering" {
    try std.testing.expect(@intFromEnum(SectionId.type) < @intFromEnum(SectionId.import));
    try std.testing.expect(@intFromEnum(SectionId.import) < @intFromEnum(SectionId.function));
    try std.testing.expect(@intFromEnum(SectionId.code) < @intFromEnum(SectionId.data));
}

// ─── Module-level structures ────────────────────────────────────────────────

/// Import descriptor (§2.5.1)
pub const ImportDesc = struct {
    module_name: []const u8,
    field_name: []const u8,
    kind: ExternalKind,
    // kind-specific payload:
    func_type_idx: ?u32 = null,
    table_type: ?TableType = null,
    memory_type: ?MemoryType = null,
    global_type: ?GlobalType = null,
    tag_type_idx: ?u32 = null,
};

/// Export descriptor (§2.5.2)
pub const ExportDesc = struct {
    name: []const u8,
    kind: ExternalKind,
    index: u32,
};

/// Function representation
pub const WasmFunction = struct {
    type_idx: u32,
    func_type: FuncType,
    local_count: u32,
    locals: []const LocalDecl,
    code: []const u8,
    max_stack_size: u32 = 0,
    max_block_depth: u32 = 0,
};

/// Local variable declaration inside a function body
pub const LocalDecl = struct {
    count: u32,
    val_type: ValType,
    /// Concrete type index (0xFFFFFFFF = abstract).
    type_idx: u32 = 0xFFFFFFFF,
};

/// Global variable definition
pub const WasmGlobal = struct {
    global_type: GlobalType,
    init_expr: InitExpr,
};

/// Constant expression used for global/data/element initializers.
/// Stores raw bytecode for lazy evaluation, supporting compound expressions.
pub const InitExpr = union(enum) {
    i32_const: i32,
    i64_const: i64,
    f32_const: f32,
    f64_const: f64,
    global_get: u32,
    ref_null: ValType,
    ref_func: u32,
    /// Raw bytecode for compound constant expressions (without trailing 0x0B)
    bytecode: []const u8,
};

/// Data segment (§2.5.7)
pub const DataSegment = struct {
    memory_idx: u32,
    offset: InitExpr,
    data: []const u8,
    is_passive: bool = false,
};

/// Element segment (§2.5.6)
pub const ElemSegment = struct {
    table_idx: u32,
    offset: ?InitExpr, // null for passive/declarative
    kind: ElemKind,
    func_indices: []const ?u32,
    /// Init expressions for elements that need runtime evaluation (global.get, bytecode)
    elem_exprs: []const ?InitExpr = &.{},
    is_passive: bool = false,
    is_declarative: bool = false,
    /// Concrete type index for element type (0xFFFFFFFF = abstract).
    type_idx: u32 = 0xFFFFFFFF,
    /// Whether element values can be null (expression vectors can, funcidx vectors can't)
    nullable_elements: bool = true,

    pub const ElemKind = enum { func_ref, extern_ref, gc_ref };
};

/// Rec group boundary for a type (used for iso-recursive equivalence).
pub const RecGroupInfo = struct {
    group_start: u32,
    group_size: u32,
};

/// Full parsed WebAssembly module
pub const WasmModule = struct {
    // Sections
    types: []const FuncType = &.{},
    /// Rec group info parallel to types (same length).
    rec_groups: []const RecGroupInfo = &.{},
    /// Canonical type index mapping (same length as types).
    /// canonical_type_map[i] = the canonical index for type i.
    canonical_type_map: []const u32 = &.{},
    imports: []const ImportDesc = &.{},
    functions: []const WasmFunction = &.{},
    tables: []const TableType = &.{},
    memories: []const MemoryType = &.{},
    globals: []const WasmGlobal = &.{},
    exports: []const ExportDesc = &.{},
    start_function: ?u32 = null,
    elements: []const ElemSegment = &.{},
    data_segments: []const DataSegment = &.{},
    data_count: ?u32 = null,
    /// Tag type indices (local tags, excluding imports).
    tag_types: []const u32 = &.{},

    // Derived counts (imports + local definitions)
    import_function_count: u32 = 0,
    import_table_count: u32 = 0,
    import_memory_count: u32 = 0,
    import_global_count: u32 = 0,
    import_tag_count: u32 = 0,

    // Custom sections
    name_section: ?NameSection = null,

    /// Find an export by name and kind.
    pub fn findExport(self: *const WasmModule, name: []const u8, kind: ExternalKind) ?ExportDesc {
        for (self.exports) |exp| {
            if (exp.kind == kind and std.mem.eql(u8, exp.name, name)) return exp;
        }
        return null;
    }

    /// Get the function type for a given function index (import or local).
    pub fn getFuncType(self: *const WasmModule, func_idx: u32) ?FuncType {
        if (func_idx < self.import_function_count) {
            var import_func_idx: u32 = 0;
            for (self.imports) |imp| {
                if (imp.kind == .function) {
                    if (import_func_idx == func_idx) {
                        const tidx = imp.func_type_idx orelse return null;
                        return if (tidx < self.types.len) self.types[tidx] else null;
                    }
                    import_func_idx += 1;
                }
            }
            return null;
        }
        const local_idx = std.math.sub(u32, func_idx, self.import_function_count) catch return null;
        if (local_idx < self.functions.len) {
            const tidx = self.functions[local_idx].type_idx;
            return if (tidx < self.types.len) self.types[tidx] else null;
        }
        return null;
    }

    /// Get the type index for a given function index (import or local).
    /// Returns the canonical type index if a canonical mapping exists.
    pub fn getFuncTypeIdx(self: *const WasmModule, func_idx: u32) ?u32 {
        const raw_tidx = self.getRawFuncTypeIdx(func_idx) orelse return null;
        if (raw_tidx < self.canonical_type_map.len) return self.canonical_type_map[raw_tidx];
        return raw_tidx;
    }

    pub fn getRawFuncTypeIdx(self: *const WasmModule, func_idx: u32) ?u32 {
        if (func_idx < self.import_function_count) {
            var import_func_idx: u32 = 0;
            for (self.imports) |imp| {
                if (imp.kind == .function) {
                    if (import_func_idx == func_idx) return imp.func_type_idx;
                    import_func_idx += 1;
                }
            }
            return null;
        }
        const local_idx = std.math.sub(u32, func_idx, self.import_function_count) catch return null;
        if (local_idx < self.functions.len) return self.functions[local_idx].type_idx;
        return null;
    }
};

/// Name custom section (§Appendix: Custom sections)
pub const NameSection = struct {
    module_name: ?[]const u8 = null,
    function_names: []const FunctionName = &.{},

    pub const FunctionName = struct {
        index: u32,
        name: []const u8,
    };
};

// ─── Instance-level structures ──────────────────────────────────────────────

/// Runtime memory instance
/// Runtime memory instance (refcounted for cross-module sharing)
pub const MemoryInstance = struct {
    memory_type: MemoryType,
    /// For non-shared memory this is the allocated/visible buffer. For
    /// shared memory it is the immutable full reservation; callers must use
    /// `byteLen`/`bytes` for the acquire-published visible extent.
    data: []u8,
    /// Legacy non-shared page count. Shared readers use `pageCount`.
    current_pages: u32,
    max_pages: u32,
    /// Non-shared ownership count. It compiles out when WASI threads are
    /// disabled; shared memory keeps using its dedicated control block.
    ref_count: stable_resource.ConditionalLifetimeRefCount =
        stable_resource.ConditionalLifetimeRefCount.init(1),
    /// Present exactly for shared memories. Owns the immutable reservation,
    /// atomic size publication, serialized grow, and keyed parking lot.
    shared_control: ?*shared_memory.Control = null,
    /// AOT vmctx mirrors that currently reference this memory. Stored as
    /// opaque pointers to avoid a common/types.zig -> aot/runtime.zig cycle.
    vmctx_subscribers: std.ArrayListUnmanaged(*anyopaque) = .empty,
    subscriber_mutex: platform.Mutex = .init,
    /// When non-null, `data.ptr == reserved_base` and `[reserved_base,
    /// reserved_base+reserved_size)` is a stable virtual address
    /// reservation (POSIX mmap, Windows VirtualAlloc). Non-shared `grow`
    /// extends `data.len`; shared `data.len` is the constant reservation
    /// capacity and `byteLen` publishes the committed extent. The pointer
    /// never moves, so external aliases into this memory (e.g.
    /// SpiderMonkey/StarlingMonkey external strings, host-held slices
    /// taken before a `memory.grow`) remain valid. Issue #752: TCGC's
    /// componentize-js engine reads a 1.3 MiB string as an external
    /// string; a subsequent `Map.set` triggers `memory.grow`; under
    /// the legacy `allocator.realloc` path that grow relocated the
    /// buffer and silently corrupted the external string, surfacing
    /// as TCGC's "Resolved to / which is outside package file://"
    /// module-resolution failure.
    reserved_base: ?[*]align(page_size_min) u8 = null,
    /// Size of the reservation when `reserved_base != null`. Constant
    /// for the lifetime of the memory.
    reserved_size: usize = 0,

    pub const page_size: u32 = 65536;
    pub const max_addressable_pages: u32 = @intCast(@min(
        @as(u64, 65536),
        @as(u64, std.math.maxInt(usize) / @as(usize, page_size)),
    ));

    /// Page size of the host platform — used for mmap/mprotect alignment.
    /// Distinct from `page_size` above (the wasm linear-memory page = 64 KiB)
    /// because Linux page-grained mprotect needs 4 KiB on x86_64 / 16 KiB on
    /// aarch64. wasm pages are always a multiple of any reasonable host page,
    /// so the OS calls never see sub-page slack.
    pub const page_size_min: u29 = std.heap.page_size_min;

    pub fn byteSizeForPages(pages: u32) ?usize {
        if (pages > max_addressable_pages) return null;
        return std.math.mul(usize, @intCast(pages), page_size) catch null;
    }

    /// Allocate a `MemoryInstance` with **stable-address backing** when the
    /// host supports it. The memory's `data.ptr` is pinned to a virtual
    /// address reservation of `cap_pages * 64 KiB`, with `initial_pages`
    /// committed up front. `grow` extends the committed window via
    /// `mprotect`; the pointer never moves, so external aliases into the
    /// memory survive any `memory.grow` calls. (Issue #752.)
    ///
    /// `cap_pages` should be `min(mem_type.limits.max orelse 65536, 65536)`.
    /// When the OS reservation fails, this function returns null and the
    /// non-shared caller may fall back to the legacy
    /// `allocator.realloc`-backed path (which may relocate `data.ptr`
    /// on grow). The caller owns release via `MemoryInstance.release`.
    pub fn createReserved(
        mem_type: MemoryType,
        initial_pages: u32,
        cap_pages: u32,
        allocator: std.mem.Allocator,
    ) ?*MemoryInstance {
        if (mem_type.is_shared) return null;
        if (!platform.supports_reserved_memory) return null;
        if (cap_pages == 0) return null;
        const reserved_size = byteSizeForPages(cap_pages) orelse return null;
        const base = platform.reserveAddressSpace(reserved_size) orelse return null;
        errdefer platform.releaseAddressSpace(base, reserved_size);

        const initial_size = byteSizeForPages(initial_pages) orelse return null;
        if (initial_size > 0) {
            platform.commitPages(base, initial_size) catch {
                platform.releaseAddressSpace(base, reserved_size);
                return null;
            };
            // `mmap` with `MAP_ANONYMOUS` zeroes new pages; mprotect on a
            // freshly reserved region preserves that. No memset needed.
        }

        const mem = allocator.create(MemoryInstance) catch {
            platform.releaseAddressSpace(base, reserved_size);
            return null;
        };
        mem.* = .{
            .memory_type = mem_type,
            .data = base[0..initial_size],
            .current_pages = initial_pages,
            .max_pages = cap_pages,
            .reserved_base = base,
            .reserved_size = reserved_size,
        };
        return mem;
    }

    /// Create shared memory. The declared maximum is mandatory and its
    /// entire virtual address range is reserved before this function
    /// succeeds. There is deliberately no relocating allocator fallback.
    pub fn createShared(
        mem_type: MemoryType,
        allocator: std.mem.Allocator,
    ) shared_memory.CreateError!*MemoryInstance {
        if (!mem_type.is_shared) return error.InvalidLimits;
        const max_u64 = mem_type.limits.max orelse return error.InvalidLimits;
        if (mem_type.limits.min > max_u64 or max_u64 > max_addressable_pages)
            return error.InvalidLimits;
        const initial_pages: u32 = @intCast(mem_type.limits.min);
        const max_pages: u32 = @intCast(max_u64);
        const control = try shared_memory.Control.create(initial_pages, max_pages, allocator);
        errdefer {
            std.debug.assert(control.release());
            control.destroy(allocator);
        }

        const mem = allocator.create(MemoryInstance) catch return error.OutOfMemory;
        mem.* = .{
            .memory_type = mem_type,
            .data = control.capacity(),
            .current_pages = initial_pages,
            .max_pages = max_pages,
            .shared_control = control,
            .reserved_base = control.base,
            .reserved_size = control.reserved_bytes,
        };
        return mem;
    }

    /// Acquire-published current page count.
    pub fn pageCount(self: *const MemoryInstance) u32 {
        if (self.shared_control) |control| return control.pageCount();
        return self.current_pages;
    }

    /// Acquire-published current byte length.
    pub fn byteLen(self: *const MemoryInstance) usize {
        if (self.shared_control) |control| return control.byteLen();
        return self.data.len;
    }

    /// Visible memory slice. For shared memory the slice length is formed
    /// only after the acquire load in `byteLen`.
    pub fn bytes(self: *MemoryInstance) []u8 {
        return self.data[0..self.byteLen()];
    }

    pub const SharedWaitError = shared_memory.WaitError || error{NotShared};

    pub fn grow(self: *MemoryInstance, delta: u32, allocator: std.mem.Allocator) !u32 {
        if (self.shared_control) |control| {
            return control.grow(delta) catch |err| switch (err) {
                error.MemoryGrowFailed => error.MemoryGrowFailed,
            };
        }
        const old_pages = self.current_pages;
        const new_pages = std.math.add(u32, old_pages, delta) catch return error.MemoryGrowFailed;
        if (self.memory_type.limits.max) |max| {
            if (new_pages > max) return error.MemoryGrowFailed;
        }
        const new_size = byteSizeForPages(new_pages) orelse return error.MemoryGrowFailed;
        const old_size = self.data.len;
        if (old_size < new_size) {
            if (self.reserved_base) |base| {
                // Stable-address path: commit additional pages within
                // the reservation. `data.ptr` is unchanged; we only
                // extend `data.len`. mmap PROT_NONE pages were
                // anonymous so they are already zero; mprotect to
                // READ|WRITE makes them touchable. (#752)
                if (new_size > self.reserved_size) return error.MemoryGrowFailed;
                platform.commitPages(@alignCast(base + old_size), new_size - old_size) catch
                    return error.MemoryGrowFailed;
                self.data = base[0..new_size];
            } else {
                self.data = try allocator.realloc(self.data, new_size);
                // Zero-initialize the new pages (wasm spec requirement)
                @memset(self.data[old_size..new_size], 0);
            }
        }
        self.current_pages = new_pages;
        return old_pages;
    }

    pub fn retain(self: *MemoryInstance) void {
        if (self.shared_control) |control| {
            std.debug.assert(control.retain());
            return;
        }
        self.ref_count.retain();
    }

    pub fn wait32(self: *MemoryInstance, offset: usize, expected: u32, timeout_ns: i64) SharedWaitError!parking_lot.WaitResult {
        const control = self.shared_control orelse return error.NotShared;
        return control.wait32(offset, expected, timeout_ns);
    }

    pub fn wait32Cancellable(
        self: *MemoryInstance,
        offset: usize,
        expected: u32,
        timeout_ns: i64,
        cancellation: ?parking_lot.CancellationEpoch.Ticket,
    ) SharedWaitError!parking_lot.WaitResult {
        const control = self.shared_control orelse return error.NotShared;
        return control.wait32Cancellable(offset, expected, timeout_ns, cancellation);
    }

    pub fn wait64(self: *MemoryInstance, offset: usize, expected: u64, timeout_ns: i64) SharedWaitError!parking_lot.WaitResult {
        const control = self.shared_control orelse return error.NotShared;
        return control.wait64(offset, expected, timeout_ns);
    }

    pub fn wait64Cancellable(
        self: *MemoryInstance,
        offset: usize,
        expected: u64,
        timeout_ns: i64,
        cancellation: ?parking_lot.CancellationEpoch.Ticket,
    ) SharedWaitError!parking_lot.WaitResult {
        const control = self.shared_control orelse return error.NotShared;
        return control.wait64Cancellable(offset, expected, timeout_ns, cancellation);
    }

    pub fn notify(self: *MemoryInstance, offset: usize, count: u32) SharedWaitError!u32 {
        const control = self.shared_control orelse return error.NotShared;
        return control.notify(offset, count);
    }

    pub fn cancelWaiters(self: *MemoryInstance) parking_lot.BackendError!u32 {
        const control = self.shared_control orelse return 0;
        return control.cancelAll();
    }

    pub fn cancelWaitersForEpoch(
        self: *MemoryInstance,
        ticket: parking_lot.CancellationEpoch.Ticket,
    ) parking_lot.BackendError!u32 {
        const control = self.shared_control orelse return 0;
        return control.cancelEpoch(ticket);
    }

    pub fn subscribeVmCtx(self: *MemoryInstance, vmctx: *anyopaque, allocator: std.mem.Allocator) !void {
        self.subscriber_mutex.lock();
        defer self.subscriber_mutex.unlock();
        for (self.vmctx_subscribers.items) |subscriber| {
            if (subscriber == vmctx) return;
        }
        try self.vmctx_subscribers.append(allocator, vmctx);
    }

    pub fn unsubscribeVmCtx(self: *MemoryInstance, vmctx: *anyopaque) void {
        self.subscriber_mutex.lock();
        defer self.subscriber_mutex.unlock();
        for (self.vmctx_subscribers.items, 0..) |subscriber, i| {
            if (subscriber == vmctx) {
                _ = self.vmctx_subscribers.swapRemove(i);
                return;
            }
        }
    }

    pub fn release(self: *MemoryInstance, allocator: std.mem.Allocator) void {
        if (self.shared_control) |control| {
            if (!control.release()) return;
            self.vmctx_subscribers.deinit(allocator);
            control.destroy(allocator);
            allocator.destroy(self);
            return;
        }
        if (!self.ref_count.release()) return;
        self.vmctx_subscribers.deinit(allocator);
        if (self.reserved_base) |base| {
            platform.releaseAddressSpace(base, self.reserved_size);
        } else if (self.data.len > 0) {
            allocator.free(self.data);
        }
        allocator.destroy(self);
    }

    pub fn referenceCount(self: *const MemoryInstance) usize {
        if (self.shared_control) |control| return @intCast(control.referenceCount());
        return self.ref_count.count();
    }
};

/// A function reference stored in a table element.
/// Tracks the source module for cross-module call_indirect dispatch.
pub const FuncRef = struct {
    func_idx: u32,
    module_inst: *ModuleInstance,
};

/// A table element that preserves the exact Value type (i31ref, structref, etc.)
/// plus module provenance for function references.
pub const TableElement = struct {
    value: Value,
    /// For funcref/nonfuncref: the originating module instance (for cross-module dispatch).
    module_inst: ?*ModuleInstance = null,

    /// Create a null element for a given ref type.
    pub fn nullForType(vt: ValType) TableElement {
        return .{ .value = switch (vt) {
            .funcref => .{ .funcref = null },
            .externref => .{ .externref = null },
            .anyref => .{ .anyref = null },
            .eqref => .{ .eqref = null },
            .i31ref => .{ .i31ref = null },
            .structref => .{ .structref = null },
            .arrayref => .{ .arrayref = null },
            .nullref => .{ .nullref = null },
            .exnref => .{ .exnref = null },
            .nonfuncref => .{ .nonfuncref = null },
            .nonexternref => .{ .nonexternref = null },
            else => .{ .funcref = null },
        } };
    }

    /// Create a table element from a Value, preserving module provenance for funcrefs.
    pub fn fromValue(val: Value, source_module: ?*ModuleInstance) TableElement {
        return .{
            .value = val,
            .module_inst = switch (val) {
                .funcref, .nonfuncref => source_module,
                else => null,
            },
        };
    }

    /// Extract as FuncRef for call_indirect. Returns null if not a callable funcref.
    pub fn asFuncRef(self: TableElement) ?FuncRef {
        const func_idx: ?u32 = switch (self.value) {
            .funcref, .nonfuncref => |r| r,
            else => null,
        };
        const idx = func_idx orelse return null;
        const mi = self.module_inst orelse return null;
        return .{ .func_idx = idx, .module_inst = mi };
    }

    /// Check if this element is a null reference.
    pub fn isNull(self: TableElement) bool {
        return switch (self.value) {
            .funcref, .nonfuncref => |r| r == null,
            .externref, .nonexternref => |r| r == null,
            .anyref, .eqref, .i31ref, .structref, .arrayref, .nullref, .exnref => |r| r == null,
            else => true,
        };
    }
};

const StableAotTableBacking = if (config.lib_wasi_threads) struct {
    const pointer_alignment = std.heap.page_size_min;
    // 64 KiB is a multiple of supported host page sizes (including 64 KiB
    // AArch64 Linux kernels), so every incremental mprotect/commit range is
    // valid without querying a process-global runtime page size.
    const commit_granularity: usize = 64 * 1024;

    native_base: ?[*]align(pointer_alignment) u8 = null,
    native_reserved_bytes: usize = 0,
    native_committed_bytes: usize = 0,
    type_base: ?[*]align(pointer_alignment) u8 = null,
    type_reserved_bytes: usize = 0,
    type_committed_bytes: usize = 0,
    capacity: usize = 0,
    retired_native_backing: []usize = &.{},
    retired_type_backing: []u32 = &.{},
    thread_stable: bool = false,

    fn roundedBytes(element_count: usize, element_size: usize) !usize {
        const bytes = std.math.mul(usize, element_count, element_size) catch
            return error.OutOfMemory;
        if (bytes == 0) return 0;
        const with_slack = std.math.add(
            usize,
            bytes,
            commit_granularity - 1,
        ) catch
            return error.OutOfMemory;
        return with_slack & ~(commit_granularity - 1);
    }

    fn reserve(self: *@This(), capacity: usize) !void {
        std.debug.assert(self.native_base == null);
        std.debug.assert(self.type_base == null);
        if (!platform.supports_reserved_memory) return error.OutOfMemory;

        const native_bytes = try roundedBytes(capacity, @sizeOf(usize));
        const type_bytes = try roundedBytes(capacity, @sizeOf(u32));
        if (native_bytes == 0 or type_bytes == 0) {
            self.capacity = capacity;
            return;
        }

        const native_base = platform.reserveAddressSpace(native_bytes) orelse
            return error.OutOfMemory;
        errdefer platform.releaseAddressSpace(native_base, native_bytes);
        const type_base = platform.reserveAddressSpace(type_bytes) orelse
            return error.OutOfMemory;

        self.native_base = native_base;
        self.native_reserved_bytes = native_bytes;
        self.type_base = type_base;
        self.type_reserved_bytes = type_bytes;
        self.capacity = capacity;
    }

    fn commitThrough(self: *@This(), element_count: usize) !void {
        if (element_count > self.capacity) return error.OutOfMemory;
        const native_target = try roundedBytes(element_count, @sizeOf(usize));
        const type_target = try roundedBytes(element_count, @sizeOf(u32));

        if (native_target > self.native_committed_bytes) {
            const base = self.native_base orelse return error.OutOfMemory;
            const start: [*]align(pointer_alignment) u8 =
                @alignCast(base + self.native_committed_bytes);
            platform.commitPages(
                start,
                native_target - self.native_committed_bytes,
            ) catch return error.OutOfMemory;
            self.native_committed_bytes = native_target;
        }
        if (type_target > self.type_committed_bytes) {
            const base = self.type_base orelse return error.OutOfMemory;
            const start: [*]align(pointer_alignment) u8 =
                @alignCast(base + self.type_committed_bytes);
            platform.commitPages(
                start,
                type_target - self.type_committed_bytes,
            ) catch return error.OutOfMemory;
            self.type_committed_bytes = type_target;
        }
    }

    fn nativeSlice(self: *const @This(), len: usize) []usize {
        if (len == 0) return &.{};
        const base = self.native_base orelse unreachable;
        const ptr: [*]usize = @ptrCast(base);
        return ptr[0..len];
    }

    fn typeSlice(self: *const @This(), len: usize) []u32 {
        if (len == 0) return &.{};
        const base = self.type_base orelse unreachable;
        const ptr: [*]u32 = @ptrCast(base);
        return ptr[0..len];
    }

    fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        if (self.native_base) |base|
            platform.releaseAddressSpace(base, self.native_reserved_bytes);
        if (self.type_base) |base|
            platform.releaseAddressSpace(base, self.type_reserved_bytes);
        if (self.retired_native_backing.len > 0)
            allocator.free(self.retired_native_backing);
        if (self.retired_type_backing.len > 0)
            allocator.free(self.retired_type_backing);
        self.* = .{};
    }
} else void;

/// Runtime table instance (refcounted for cross-module sharing)
pub const TableInstance = struct {
    table_type: TableType,
    elements: []TableElement,
    ref_count: stable_resource.ConditionalLifetimeRefCount =
        stable_resource.ConditionalLifetimeRefCount.init(1),
    mutex: stable_resource.ConditionalMutex(stable_resource.LockRank.core_table) = .init,
    /// AOT-only: native code-pointer backing used by call_indirect / call_ref
    /// / table.init / table.set. Owned by this TableInstance and freed on
    /// final release. When a table is imported by another module the
    /// importer aliases this slice so that cross-module mutations (e.g.
    /// active elem segments or table.init in a start function) are visible
    /// to the exporter's compiled code.
    native_backing: []usize = &.{},
    /// AOT-only: parallel to `native_backing`, one canonical sig_id per
    /// slot. 0 = null/uninitialized. Written in lockstep with
    /// `native_backing` by all table-mutation paths (elem copy, table.set,
    /// table.init/copy/fill/grow) so that call_indirect can do a single
    /// 4-byte equality check against the caller's expected sig_id.
    type_backing: []u32 = &.{},
    /// Once the first AOT thread clone is requested, the native pointer and
    /// signature arrays move into bounded reserved address ranges. Ordinary
    /// AOT modules stay heap-backed, so maxless tables do not reserve the
    /// theoretical 2^32-element address range merely by instantiating.
    stable_aot_backing: StableAotTableBacking =
        if (config.lib_wasi_threads) .{} else {},
    /// AOT VmCtx mirrors that cache this table's native backing pointer and
    /// length. Growth refreshes every subscriber after initializing the new
    /// heap buffer or address-stable backing tail.
    vmctx_subscribers: if (config.lib_wasi_threads)
        std.ArrayListUnmanaged(*anyopaque)
    else
        void = if (config.lib_wasi_threads) .empty else {},
    subscriber_mutex: if (config.lib_wasi_threads)
        platform.Mutex
    else
        void = if (config.lib_wasi_threads) .init else {},

    pub fn retain(self: *TableInstance) void {
        self.ref_count.retain();
    }

    pub fn release(self: *TableInstance, allocator: std.mem.Allocator) void {
        if (!self.ref_count.release()) return;
        if (comptime config.lib_wasi_threads)
            self.vmctx_subscribers.deinit(allocator);
        if (self.elements.len > 0) allocator.free(self.elements);
        if (comptime config.lib_wasi_threads) {
            if (self.stable_aot_backing.thread_stable) {
                self.stable_aot_backing.deinit(allocator);
            } else {
                if (self.native_backing.len > 0) allocator.free(self.native_backing);
                if (self.type_backing.len > 0) allocator.free(self.type_backing);
            }
        } else {
            if (self.native_backing.len > 0) allocator.free(self.native_backing);
            if (self.type_backing.len > 0) allocator.free(self.type_backing);
        }
        allocator.destroy(self);
    }

    /// Maximum number of entries reserved when a maxless or very-large table
    /// first becomes reachable from a native thread clone. `table.grow` may
    /// fail beyond this host resource limit, as permitted by WebAssembly.
    pub const max_thread_stable_elements: usize = 1 << 20;

    /// Move the current heap caches into a bounded, address-stable
    /// reservation before publishing them to a child AOT thread. Failure
    /// leaves the heap-backed table intact so module instantiation remains
    /// usable; the attempted thread spawn fails instead.
    pub fn ensureThreadStableAotBacking(
        self: *TableInstance,
        allocator: std.mem.Allocator,
    ) !void {
        if (comptime !config.lib_wasi_threads) return;
        if (self.stable_aot_backing.thread_stable) return;

        const declared_max = self.table_type.limits.max orelse
            @as(u64, max_thread_stable_elements);
        const capacity_u64 = @min(
            declared_max,
            @as(u64, max_thread_stable_elements),
        );
        const capacity = std.math.cast(usize, capacity_u64) orelse
            return error.OutOfMemory;
        if (self.native_backing.len > capacity or
            self.type_backing.len != self.native_backing.len)
            return error.OutOfMemory;

        const old_native = self.native_backing;
        const old_types = self.type_backing;
        try self.stable_aot_backing.reserve(capacity);
        errdefer self.stable_aot_backing.deinit(allocator);
        try self.stable_aot_backing.commitThrough(old_native.len);

        const native = self.stable_aot_backing.nativeSlice(old_native.len);
        const sig_ids = self.stable_aot_backing.typeSlice(old_types.len);
        @memset(native, 0);
        @memset(sig_ids, 0);
        @memcpy(native, old_native);
        @memcpy(sig_ids, old_types);

        self.native_backing = native;
        self.type_backing = sig_ids;
        self.stable_aot_backing.retired_native_backing = old_native;
        self.stable_aot_backing.retired_type_backing = old_types;
        self.stable_aot_backing.thread_stable = true;
    }

    pub inline fn hasThreadStableAotBacking(self: *const TableInstance) bool {
        if (comptime !config.lib_wasi_threads) return false;
        return self.stable_aot_backing.thread_stable;
    }

    pub fn aotBackingReservedBytes(self: *const TableInstance) usize {
        if (comptime !config.lib_wasi_threads) return 0;
        return self.stable_aot_backing.native_reserved_bytes +
            self.stable_aot_backing.type_reserved_bytes;
    }

    /// Resize the AOT-native table caches. Thread-shared tables commit in
    /// place; ordinary tables retain the historical heap copy/reallocation
    /// path and therefore never reserve large virtual ranges at instantiation.
    pub fn resizeAotBacking(
        self: *TableInstance,
        allocator: std.mem.Allocator,
        new_len: usize,
    ) !void {
        if (comptime config.lib_wasi_threads) {
            if (self.stable_aot_backing.thread_stable) {
                const old_len = self.native_backing.len;
                try self.stable_aot_backing.commitThrough(new_len);
                self.native_backing = self.stable_aot_backing.nativeSlice(new_len);
                self.type_backing = self.stable_aot_backing.typeSlice(new_len);
                if (new_len > old_len) {
                    @memset(self.native_backing[old_len..], 0);
                    @memset(self.type_backing[old_len..], 0);
                }
                return;
            }
        }

        if (self.native_backing.len == new_len and
            self.type_backing.len == new_len) return;
        const native = try allocator.alloc(usize, new_len);
        errdefer if (native.len > 0) allocator.free(native);
        const sig_ids = try allocator.alloc(u32, new_len);
        @memset(native, 0);
        @memset(sig_ids, 0);
        @memcpy(
            native[0..@min(native.len, self.native_backing.len)],
            self.native_backing[0..@min(native.len, self.native_backing.len)],
        );
        @memcpy(
            sig_ids[0..@min(sig_ids.len, self.type_backing.len)],
            self.type_backing[0..@min(sig_ids.len, self.type_backing.len)],
        );
        if (self.native_backing.len > 0) allocator.free(self.native_backing);
        if (self.type_backing.len > 0) allocator.free(self.type_backing);
        self.native_backing = native;
        self.type_backing = sig_ids;
    }

    pub fn referenceCount(self: *const TableInstance) usize {
        return self.ref_count.count();
    }

    pub inline fn lock(self: *TableInstance) void {
        self.mutex.lock();
    }

    pub inline fn unlock(self: *TableInstance) void {
        self.mutex.unlock();
    }

    pub fn subscribeVmCtx(
        self: *TableInstance,
        vmctx: *anyopaque,
        allocator: std.mem.Allocator,
    ) !void {
        if (comptime !config.lib_wasi_threads) return;
        self.subscriber_mutex.lock();
        defer self.subscriber_mutex.unlock();
        for (self.vmctx_subscribers.items) |subscriber| {
            if (subscriber == vmctx) return;
        }
        try self.vmctx_subscribers.append(allocator, vmctx);
    }

    pub fn unsubscribeVmCtx(self: *TableInstance, vmctx: *anyopaque) void {
        if (comptime !config.lib_wasi_threads) return;
        self.subscriber_mutex.lock();
        defer self.subscriber_mutex.unlock();
        for (self.vmctx_subscribers.items, 0..) |subscriber, i| {
            if (subscriber == vmctx) {
                _ = self.vmctx_subscribers.swapRemove(i);
                return;
            }
        }
    }

    pub inline fn elementCount(self: *TableInstance) usize {
        self.lock();
        defer self.unlock();
        return self.elements.len;
    }

    pub inline fn getElement(self: *TableInstance, index: usize) ?TableElement {
        self.lock();
        defer self.unlock();
        if (index >= self.elements.len) return null;
        return self.elements[index];
    }

    pub inline fn setElement(self: *TableInstance, index: usize, value: TableElement) bool {
        self.lock();
        defer self.unlock();
        if (index >= self.elements.len) return false;
        self.elements[index] = value;
        return true;
    }
};

/// Runtime global instance
pub const GlobalInstance = struct {
    global_type: GlobalType,
    value: Value,
    owned: bool = true,
    ref_count: stable_resource.ConditionalLifetimeRefCount =
        stable_resource.ConditionalLifetimeRefCount.init(1),
    /// For funcref globals: the module instance that owns the referenced function
    source_module: ?*ModuleInstance = null,

    pub fn retain(self: *GlobalInstance) void {
        self.ref_count.retain();
    }

    pub fn release(self: *GlobalInstance, allocator: std.mem.Allocator) void {
        if (self.ref_count.release()) allocator.destroy(self);
    }

    pub fn referenceCount(self: *const GlobalInstance) usize {
        return self.ref_count.count();
    }
};

/// Tag instance (identity via pointer equality).
pub const TagInstance = struct {
    /// Number of parameters this tag carries.
    param_arity: u32,
};

/// A resolved imported function target
pub const ImportedFunction = struct {
    module_inst: *ModuleInstance,
    func_idx: u32,
};

/// Host function callable from Wasm via imports.
/// The callback receives an opaque pointer to an ExecEnv (to avoid circular
/// type dependencies between types.zig and exec_env.zig). Implementations
/// must cast: `const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));`
pub const HostFn = *const fn (env_opaque: *anyopaque) HostFnError!void;

/// Host function variant that carries a per-slot context pointer. Used by
/// higher layers (e.g. the component-model canon-lower trampoline) that need
/// to associate state with a specific import slot. The legacy `HostFn`
/// dispatch path is preserved; see `ModuleInstance.host_func_entries` for the
/// context-carrying path.
pub const HostFnWithCtx = *const fn (env_opaque: *anyopaque, ctx: ?*anyopaque) HostFnError!void;

pub const HostFnEntry = struct {
    func: HostFnWithCtx,
    ctx: ?*anyopaque = null,
};

pub const HostFnError = error{
    Trap,
    ThreadCancelled,
    StackOverflow,
    StackUnderflow,
    OutOfBoundsMemoryAccess,
};

/// Instantiated module
/// GC heap object (struct or array instance).
pub const GcObject = struct {
    type_idx: u32,
    fields: []Value,
};

pub const ModuleInstance = struct {
    module: *const WasmModule,
    memories: []*MemoryInstance,
    tables: []*TableInstance,
    globals: []*GlobalInstance,
    import_functions: []const ImportedFunction = &.{},
    /// Host (native) functions indexed by import function index.
    /// `host_functions[i]` is the native callback for import function i, or null.
    host_functions: []const ?HostFn = &.{},
    /// Whether host_functions was allocated by this instance (vs shared from parent).
    owns_host_functions: bool = false,
    /// Context-carrying host function entries, parallel to `host_functions`.
    /// When both slots are populated, `host_func_entries[i]` takes priority
    /// and receives the per-slot `ctx` pointer. Used by the component-model
    /// canon-lower trampoline and by any caller that needs per-import state.
    host_func_entries: []const ?HostFnEntry = &.{},
    /// Whether host_func_entries was allocated by this instance.
    owns_host_func_entries: bool = false,
    tags: []*TagInstance = &.{},
    allocator: std.mem.Allocator,
    /// Process-scoped host state retained by this instance. For WASI this is
    /// a `WasiProcessState`; component and non-WASI instances leave it null.
    process_state: ?execution_context.ProcessStateRef = null,
    state_mutex: stable_resource.ConditionalMutex(stable_resource.LockRank.core_instance) = .init,
    /// Thread manager (shared across all instances in a thread group).
    thread_manager: ?*@import("../../wasi/thread_manager.zig").ThreadManager = null,
    /// Track dropped elem segments (active segments dropped after instantiation)
    dropped_elems: []bool = &.{},
    /// Track dropped data segments (for data.drop instruction)
    dropped_data: []bool = &.{},
    /// GC heap for struct/array objects
    gc_objects: std.ArrayListUnmanaged(GcObject) = .empty,
    /// Cached evaluated element segment values (spec requires one-time evaluation)
    cached_elem_values: []?[]Value = &.{},

    pub fn getExportFunc(self: *const ModuleInstance, name: []const u8) ?u32 {
        const exp = self.module.findExport(name, .function) orelse return null;
        return exp.index;
    }

    pub fn getMemory(self: *const ModuleInstance, idx: u32) ?*MemoryInstance {
        if (idx < self.memories.len) return self.memories[idx];
        return null;
    }

    /// Retain and attach process-scoped host state. Acquiring before
    /// releasing the old value makes replacing a state with itself safe.
    pub fn attachProcessState(
        self: *ModuleInstance,
        process_state: execution_context.ProcessStateRef,
    ) void {
        const retained = process_state.acquire();
        if (self.process_state) |old| old.release();
        self.process_state = retained;
    }

    pub fn detachProcessState(self: *ModuleInstance) void {
        if (self.process_state) |state| state.release();
        self.process_state = null;
    }

    pub inline fn lockState(self: *ModuleInstance) void {
        self.state_mutex.lock();
    }

    pub inline fn unlockState(self: *ModuleInstance) void {
        self.state_mutex.unlock();
    }

    pub fn isDataSegmentDropped(self: *ModuleInstance, idx: u32) bool {
        self.lockState();
        defer self.unlockState();
        return idx < self.dropped_data.len and self.dropped_data[idx];
    }

    pub fn dropDataSegment(self: *ModuleInstance, idx: u32) void {
        self.lockState();
        defer self.unlockState();
        if (idx < self.dropped_data.len) self.dropped_data[idx] = true;
    }

    pub fn dropElementSegment(self: *ModuleInstance, idx: u32) void {
        var retired_values: ?[]Value = null;
        self.lockState();
        if (idx < self.dropped_elems.len) self.dropped_elems[idx] = true;
        if (idx < self.cached_elem_values.len) {
            retired_values = self.cached_elem_values[idx];
            self.cached_elem_values[idx] = null;
        }
        self.unlockState();
        if (retired_values) |values| self.allocator.free(values);
    }

    /// Clone this instance for a new thread (WASI-threads instance-per-thread model).
    /// Shared: memories, tables, immutable import bindings, thread manager,
    /// and retained process state.
    /// Cloned: globals (mutable globals are thread-local).
    pub fn cloneForThread(self: *const ModuleInstance, allocator: std.mem.Allocator) !*ModuleInstance {
        const inst = try allocator.create(ModuleInstance);
        var globals_initialized: usize = 0;
        var cached_values_initialized: usize = 0;

        inst.* = .{
            .module = self.module,
            .memories = &.{},
            .tables = &.{},
            .globals = &.{},
            .import_functions = self.import_functions,
            .host_functions = self.host_functions, // shared, not owned
            .owns_host_functions = false,
            .host_func_entries = self.host_func_entries, // shared, not owned
            .owns_host_func_entries = false,
            .tags = self.tags, // immutable identity, parent lifetime owns storage
            .thread_manager = self.thread_manager,
            .allocator = allocator,
        };
        if (self.process_state) |state| {
            inst.process_state = state.acquire();
        }
        errdefer {
            for (inst.memories) |memory| memory.release(allocator);
            if (inst.memories.len > 0) allocator.free(inst.memories);
            for (inst.tables) |table| table.release(allocator);
            if (inst.tables.len > 0) allocator.free(inst.tables);
            for (inst.globals[0..globals_initialized]) |global| allocator.destroy(global);
            if (inst.globals.len > 0) allocator.free(inst.globals);
            if (inst.dropped_elems.len > 0) allocator.free(inst.dropped_elems);
            if (inst.dropped_data.len > 0) allocator.free(inst.dropped_data);
            for (inst.cached_elem_values[0..cached_values_initialized]) |maybe_values| {
                if (maybe_values) |values| allocator.free(values);
            }
            if (inst.cached_elem_values.len > 0) allocator.free(inst.cached_elem_values);
            if (inst.process_state) |state| state.release();
            allocator.destroy(inst);
        }

        // Share memories (retain ref counts)
        if (self.memories.len > 0) {
            inst.memories = try allocator.alloc(*MemoryInstance, self.memories.len);
            for (self.memories, 0..) |m, i| {
                m.retain();
                inst.memories[i] = m;
            }
        }

        // Share tables (retain ref counts)
        if (self.tables.len > 0) {
            inst.tables = try allocator.alloc(*TableInstance, self.tables.len);
            for (self.tables, 0..) |t, i| {
                t.retain();
                inst.tables[i] = t;
            }
        }

        // Clone globals (each thread gets its own mutable global state)
        if (self.globals.len > 0) {
            inst.globals = try allocator.alloc(*GlobalInstance, self.globals.len);
            for (self.globals, 0..) |g, i| {
                const clone = try allocator.create(GlobalInstance);
                clone.* = .{
                    .global_type = g.global_type,
                    .value = g.value,
                    .source_module = g.source_module,
                };
                inst.globals[i] = clone;
                globals_initialized += 1;
            }
        }

        // Segment drop state is execution-local in the Preview-1
        // instance-per-thread model.
        if (self.dropped_elems.len > 0 or
            self.dropped_data.len > 0 or
            self.cached_elem_values.len > 0)
        {
            const mutable_self = @constCast(self);
            mutable_self.lockState();
            defer mutable_self.unlockState();

            if (self.dropped_elems.len > 0) {
                inst.dropped_elems = try allocator.dupe(bool, self.dropped_elems);
            }
            if (self.dropped_data.len > 0) {
                inst.dropped_data = try allocator.dupe(bool, self.dropped_data);
            }

            // Cached element expressions are evaluated once at instantiation,
            // then cloned so elem.drop in one thread cannot invalidate another.
            if (self.cached_elem_values.len > 0) {
                inst.cached_elem_values = try allocator.alloc(?[]Value, self.cached_elem_values.len);
                @memset(inst.cached_elem_values, null);
                for (self.cached_elem_values, 0..) |maybe_values, i| {
                    if (maybe_values) |values| {
                        inst.cached_elem_values[i] = try allocator.dupe(Value, values);
                    }
                    cached_values_initialized += 1;
                }
            }
        }

        return inst;
    }

    pub fn destroyThreadClone(self: *ModuleInstance) void {
        const allocator = self.allocator;
        for (self.memories) |memory| memory.release(allocator);
        if (self.memories.len > 0) allocator.free(self.memories);
        for (self.tables) |table| table.release(allocator);
        if (self.tables.len > 0) allocator.free(self.tables);
        for (self.globals) |global| allocator.destroy(global);
        if (self.globals.len > 0) allocator.free(self.globals);
        if (self.dropped_elems.len > 0) allocator.free(self.dropped_elems);
        if (self.dropped_data.len > 0) allocator.free(self.dropped_data);
        for (self.cached_elem_values) |maybe_values| {
            if (maybe_values) |values| allocator.free(values);
        }
        if (self.cached_elem_values.len > 0) allocator.free(self.cached_elem_values);
        for (self.gc_objects.items) |object| allocator.free(object.fields);
        self.gc_objects.deinit(allocator);
        if (self.process_state) |state| state.release();
        allocator.destroy(self);
    }
};

// ─── Tests for module-level structures ──────────────────────────────────────

test "core resource lifetime references release exactly once concurrently" {
    if (!config.lib_wasi_threads) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const memory = try allocator.create(MemoryInstance);
    memory.* = .{
        .memory_type = .{ .limits = .{ .min = 0, .max = 0 } },
        .data = &.{},
        .current_pages = 0,
        .max_pages = 0,
    };
    memory.retain();

    const elements = try allocator.alloc(TableElement, 1);
    elements[0] = TableElement.nullForType(.funcref);
    const table = try allocator.create(TableInstance);
    table.* = .{
        .table_type = .{ .elem_type = .funcref, .limits = .{ .min = 1, .max = 1 } },
        .elements = elements,
    };
    table.retain();

    const global = try allocator.create(GlobalInstance);
    global.* = .{
        .global_type = .{ .val_type = .i32, .mutability = .mutable },
        .value = .{ .i32 = 0 },
    };
    global.retain();

    const Releaser = struct {
        fn releaseMemory(target: *MemoryInstance, alloc: std.mem.Allocator) void {
            target.release(alloc);
        }
        fn releaseTable(target: *TableInstance, alloc: std.mem.Allocator) void {
            target.release(alloc);
        }
        fn releaseGlobal(target: *GlobalInstance, alloc: std.mem.Allocator) void {
            target.release(alloc);
        }
    };

    const memory_first = try std.Thread.spawn(.{}, Releaser.releaseMemory, .{ memory, allocator });
    const memory_second = try std.Thread.spawn(.{}, Releaser.releaseMemory, .{ memory, allocator });
    const table_first = try std.Thread.spawn(.{}, Releaser.releaseTable, .{ table, allocator });
    const table_second = try std.Thread.spawn(.{}, Releaser.releaseTable, .{ table, allocator });
    const global_first = try std.Thread.spawn(.{}, Releaser.releaseGlobal, .{ global, allocator });
    const global_second = try std.Thread.spawn(.{}, Releaser.releaseGlobal, .{ global, allocator });
    memory_first.join();
    memory_second.join();
    table_first.join();
    table_second.join();
    global_first.join();
    global_second.join();
}

test "shared table accessors serialize concurrent mutation" {
    if (!config.lib_wasi_threads) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const elements = try allocator.alloc(TableElement, 4);
    for (elements) |*element| element.* = TableElement.nullForType(.funcref);
    const table = try allocator.create(TableInstance);
    table.* = .{
        .table_type = .{ .elem_type = .funcref, .limits = .{ .min = 4, .max = 4 } },
        .elements = elements,
    };
    defer table.release(allocator);

    const Writer = struct {
        fn run(target: *TableInstance, index: usize, value: u32) void {
            var i: usize = 0;
            while (i < 10_000) : (i += 1) {
                std.debug.assert(target.setElement(index, .{
                    .value = .{ .funcref = value },
                }));
                const observed = target.getElement(index).?;
                std.debug.assert(observed.value.funcref != null);
            }
        }
    };

    var threads: [4]std.Thread = undefined;
    for (&threads, 0..) |*thread, i| {
        thread.* = try std.Thread.spawn(
            .{},
            Writer.run,
            .{ table, i, @as(u32, @intCast(i + 10)) },
        );
    }
    for (threads) |thread| thread.join();
    for (0..4) |i| {
        try std.testing.expectEqual(
            @as(?u32, @intCast(i + 10)),
            table.getElement(i).?.value.funcref,
        );
    }
}

test "core resource concurrent elem.drop frees cached values exactly once" {
    if (!config.lib_wasi_threads) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var module = WasmModule{};
    var dropped = [_]bool{false};
    const values = try allocator.alloc(Value, 1);
    values[0] = .{ .i32 = 1 };
    var cached = [_]?[]Value{values};
    var instance = ModuleInstance{
        .module = &module,
        .memories = &.{},
        .tables = &.{},
        .globals = &.{},
        .allocator = allocator,
        .dropped_elems = &dropped,
        .cached_elem_values = &cached,
    };

    const Dropper = struct {
        fn run(target: *ModuleInstance) void {
            target.dropElementSegment(0);
        }
    };
    const first = try std.Thread.spawn(.{}, Dropper.run, .{&instance});
    const second = try std.Thread.spawn(.{}, Dropper.run, .{&instance});
    first.join();
    second.join();

    try std.testing.expect(instance.dropped_elems[0]);
    try std.testing.expect(instance.cached_elem_values[0] == null);
}

test "cloneForThread rolls back shared resource retains on allocation failure" {
    const allocator = std.testing.allocator;
    var module = WasmModule{};
    const ProcessTracker = struct {
        refs: usize = 1,

        fn retain(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.refs += 1;
        }

        fn release(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.refs -= 1;
        }
    };
    const process_ops = execution_context.ProcessStateOps{
        .retain = ProcessTracker.retain,
        .release = ProcessTracker.release,
    };
    var process_tracker = ProcessTracker{};
    const process_ref = execution_context.ProcessStateRef.init(
        @ptrCast(&process_tracker),
        &process_ops,
    );

    const memory = try allocator.create(MemoryInstance);
    memory.* = .{
        .memory_type = .{ .limits = .{ .min = 0, .max = 0 } },
        .data = &.{},
        .current_pages = 0,
        .max_pages = 0,
    };
    defer memory.release(allocator);

    const elements = try allocator.alloc(TableElement, 1);
    elements[0] = TableElement.nullForType(.funcref);
    const table = try allocator.create(TableInstance);
    table.* = .{
        .table_type = .{ .elem_type = .funcref, .limits = .{ .min = 1, .max = 1 } },
        .elements = elements,
    };
    defer table.release(allocator);

    const global = try allocator.create(GlobalInstance);
    global.* = .{
        .global_type = .{ .val_type = .i32, .mutability = .mutable },
        .value = .{ .i32 = 7 },
    };
    defer global.release(allocator);

    var memories = [_]*MemoryInstance{memory};
    var tables = [_]*TableInstance{table};
    var globals = [_]*GlobalInstance{global};
    var parent = ModuleInstance{
        .module = &module,
        .memories = &memories,
        .tables = &tables,
        .globals = &globals,
        .allocator = allocator,
    };
    parent.attachProcessState(process_ref);
    defer {
        parent.detachProcessState();
        process_ref.release();
    }

    var fail_tables = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 2 });
    try std.testing.expectError(
        error.OutOfMemory,
        parent.cloneForThread(fail_tables.allocator()),
    );
    try std.testing.expectEqual(@as(usize, 1), memory.referenceCount());
    try std.testing.expectEqual(@as(usize, 1), table.referenceCount());
    try std.testing.expectEqual(@as(usize, 2), process_tracker.refs);

    var fail_global = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 4 });
    try std.testing.expectError(
        error.OutOfMemory,
        parent.cloneForThread(fail_global.allocator()),
    );
    try std.testing.expectEqual(@as(usize, 1), memory.referenceCount());
    try std.testing.expectEqual(@as(usize, 1), table.referenceCount());
    try std.testing.expectEqual(@as(usize, 2), process_tracker.refs);
}

test "cloneForThread copies mutable segment state" {
    const allocator = std.testing.allocator;
    var module = WasmModule{};
    var dropped_elems = [_]bool{ false, true };
    var dropped_data = [_]bool{ true, false };
    var cached_values = [_]Value{.{ .i32 = 7 }};
    var cached = [_]?[]Value{cached_values[0..]};
    var parent = ModuleInstance{
        .module = &module,
        .memories = &.{},
        .tables = &.{},
        .globals = &.{},
        .allocator = allocator,
        .dropped_elems = &dropped_elems,
        .dropped_data = &dropped_data,
        .cached_elem_values = &cached,
    };

    const clone = try parent.cloneForThread(allocator);
    defer clone.destroyThreadClone();
    clone.dropped_elems[0] = true;
    clone.dropped_data[0] = false;
    clone.cached_elem_values[0].?[0] = .{ .i32 = 99 };

    try std.testing.expect(!parent.dropped_elems[0]);
    try std.testing.expect(parent.dropped_data[0]);
    try std.testing.expectEqual(@as(i32, 7), parent.cached_elem_values[0].?[0].i32);
    clone.dropElementSegment(0);
    try std.testing.expect(clone.dropped_elems[0]);
    try std.testing.expect(clone.cached_elem_values[0] == null);
}

test "WasmModule: findExport returns null on empty module" {
    const module = WasmModule{};
    try std.testing.expectEqual(null, module.findExport("main", .function));
}

test "WasmModule: findExport finds matching export" {
    const exports = [_]ExportDesc{
        .{ .name = "memory", .kind = .memory, .index = 0 },
        .{ .name = "main", .kind = .function, .index = 1 },
    };
    const module = WasmModule{ .exports = &exports };
    const result = module.findExport("main", .function);
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(u32, 1), result.?.index);
    // Wrong kind should not match
    try std.testing.expectEqual(null, module.findExport("main", .table));
}

test "MemoryInstance: grow returns old page count" {
    const allocator = std.testing.allocator;
    const data = try allocator.alloc(u8, MemoryInstance.page_size);
    defer allocator.free(data);

    var mem = MemoryInstance{
        .memory_type = .{ .limits = .{ .min = 1, .max = 4 } },
        .data = data,
        .current_pages = 1,
        .max_pages = 4,
    };
    // Test with a zero-delta grow which is always valid and avoids
    // reallocation (so the deferred free remains correct).
    const old = try mem.grow(0, allocator);
    try std.testing.expectEqual(@as(u32, 1), old);
    try std.testing.expectEqual(@as(u32, 1), mem.current_pages);
}

test "MemoryInstance: grow fails when exceeding max" {
    const allocator = std.testing.allocator;
    const data = try allocator.alloc(u8, MemoryInstance.page_size);
    defer allocator.free(data);

    var mem = MemoryInstance{
        .memory_type = .{ .limits = .{ .min = 1, .max = 2 } },
        .data = data,
        .current_pages = 1,
        .max_pages = 2,
    };
    const result = mem.grow(3, allocator);
    try std.testing.expectError(error.MemoryGrowFailed, result);
    // Page count should be unchanged
    try std.testing.expectEqual(@as(u32, 1), mem.current_pages);
}

// #752: stable-address backing for linear memory. A reserved-mem
// `MemoryInstance` MUST keep `data.ptr` pinned across any number of
// `grow` calls so external aliases (SpiderMonkey/StarlingMonkey
// "external strings" pointing at canon-lifted bytes, host-side
// slices captured across cross-instance marshal) remain valid.
test "MemoryInstance: createReserved keeps data.ptr stable across grow" {
    if (!platform.supports_reserved_memory) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const mem_type: MemoryType = .{ .limits = .{ .min = 1, .max = 8 } };
    const mem = MemoryInstance.createReserved(mem_type, 1, 8, allocator) orelse
        return error.SkipZigTest;
    defer mem.release(allocator);

    try std.testing.expect(mem.reserved_base != null);
    try std.testing.expectEqual(@as(usize, 8) * MemoryInstance.page_size, mem.reserved_size);
    try std.testing.expectEqual(@as(u32, 1), mem.current_pages);
    try std.testing.expectEqual(@as(usize, MemoryInstance.page_size), mem.data.len);

    // Capture a host-side slice and a few sentinel bytes before grow.
    const pinned_ptr = mem.data.ptr;
    mem.data[0] = 0xAB;
    mem.data[MemoryInstance.page_size - 1] = 0xCD;
    const pinned_slice = mem.data[0..16];
    pinned_slice[5] = 0x5A;

    const old_pages = try mem.grow(3, allocator);
    try std.testing.expectEqual(@as(u32, 1), old_pages);
    try std.testing.expectEqual(@as(u32, 4), mem.current_pages);
    try std.testing.expectEqual(@as(usize, 4) * MemoryInstance.page_size, mem.data.len);

    // Pointer stability: must be the same as before grow.
    try std.testing.expectEqual(pinned_ptr, mem.data.ptr);

    // Pre-grow bytes must still be readable.
    try std.testing.expectEqual(@as(u8, 0xAB), mem.data[0]);
    try std.testing.expectEqual(@as(u8, 0xCD), mem.data[MemoryInstance.page_size - 1]);
    try std.testing.expectEqual(@as(u8, 0x5A), pinned_slice[5]);

    // Newly committed pages must be zero (wasm spec).
    try std.testing.expectEqual(@as(u8, 0), mem.data[MemoryInstance.page_size]);
    try std.testing.expectEqual(@as(u8, 0), mem.data[4 * MemoryInstance.page_size - 1]);

    // Another grow → still stable.
    _ = try mem.grow(4, allocator);
    try std.testing.expectEqual(pinned_ptr, mem.data.ptr);
    try std.testing.expectEqual(@as(u32, 8), mem.current_pages);

    // Growing past the cap fails.
    try std.testing.expectError(error.MemoryGrowFailed, mem.grow(1, allocator));
    try std.testing.expectEqual(@as(u32, 8), mem.current_pages);
}

test "MemoryInstance: page sizing respects host pointer width" {
    const expected_max: u32 = if (@bitSizeOf(usize) == 32) 65535 else 65536;
    try std.testing.expectEqual(expected_max, MemoryInstance.max_addressable_pages);
    try std.testing.expect(MemoryInstance.byteSizeForPages(expected_max) != null);
    if (expected_max < 65536) {
        try std.testing.expect(MemoryInstance.byteSizeForPages(expected_max + 1) == null);
    }
}

test "MemoryInstance: shared control keeps base stable and lifetime refcounted" {
    if (!platform.supports_reserved_memory) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const mem_type: MemoryType = .{
        .limits = .{ .min = 1, .max = 3 },
        .is_shared = true,
    };
    const mem = try MemoryInstance.createShared(mem_type, allocator);

    const base = mem.data.ptr;
    try std.testing.expectEqual(@as(usize, 3 * MemoryInstance.page_size), mem.data.len);
    try std.testing.expectEqual(@as(usize, MemoryInstance.page_size), mem.byteLen());
    try std.testing.expectEqual(@as(u32, 1), mem.pageCount());
    try std.testing.expectEqual(@as(u32, 1), mem.shared_control.?.referenceCount());

    mem.retain();
    try std.testing.expectEqual(@as(u32, 2), mem.shared_control.?.referenceCount());
    mem.release(allocator);
    try std.testing.expectEqual(@as(u32, 1), mem.shared_control.?.referenceCount());

    try std.testing.expectEqual(@as(u32, 1), try mem.grow(1, allocator));
    try std.testing.expectEqual(base, mem.data.ptr);
    try std.testing.expectEqual(@as(usize, 2 * MemoryInstance.page_size), mem.byteLen());
    try std.testing.expectEqual(@as(u8, 0), mem.data[2 * MemoryInstance.page_size - 1]);
    mem.release(allocator);
}

test "MemoryInstance: shared creation requires declared maximum" {
    const mem_type: MemoryType = .{
        .limits = .{ .min = 1 },
        .is_shared = true,
    };
    try std.testing.expectError(
        error.InvalidLimits,
        MemoryInstance.createShared(mem_type, std.testing.allocator),
    );
}

test "WasmModule: getFuncType for import and local functions" {
    const i32_type = ValType.i32;
    const func_types = [_]FuncType{
        .{ .params = &.{i32_type}, .results = &.{i32_type} }, // type 0
        .{ .params = &.{}, .results = &.{} }, // type 1
    };
    const imports = [_]ImportDesc{
        .{
            .module_name = "env",
            .field_name = "imported_fn",
            .kind = .function,
            .func_type_idx = 0,
        },
    };
    const locals = [_]WasmFunction{
        .{
            .type_idx = 1,
            .func_type = func_types[1],
            .local_count = 0,
            .locals = &.{},
            .code = &.{},
        },
    };
    const module = WasmModule{
        .types = &func_types,
        .imports = &imports,
        .functions = &locals,
        .import_function_count = 1,
    };

    // func index 0 -> imported, should resolve to type 0
    const import_ft = module.getFuncType(0);
    try std.testing.expect(import_ft != null);
    try std.testing.expectEqual(@as(usize, 1), import_ft.?.params.len);

    // func index 1 -> local function, should resolve to type 1
    const local_ft = module.getFuncType(1);
    try std.testing.expect(local_ft != null);
    try std.testing.expectEqual(@as(usize, 0), local_ft.?.params.len);

    // out of bounds index should return null
    try std.testing.expectEqual(null, module.getFuncType(99));
}
