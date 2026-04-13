//! Core WebAssembly types used throughout the runtime.

const std = @import("std");

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
        // GC type hierarchy: nullref <: i31ref/structref/arrayref <: eqref <: anyref
        if (other == .anyref) return self == .eqref or self == .i31ref or self == .structref or self == .arrayref or self == .nullref;
        if (other == .eqref) return self == .i31ref or self == .structref or self == .arrayref or self == .nullref;
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
    /// Declared supertype index (0xFFFFFFFF = none).
    supertype_idx: u32 = 0xFFFFFFFF,
    /// Whether this type is declared `final` (cannot be subtyped).
    is_final: bool = false,

    pub const Kind = enum(u2) { func, struct_, array };
};

/// Limits (§2.3.4)
pub const Limits = struct {
    min: u32,
    max: ?u32 = null,
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

/// Wasm binary version
pub const wasm_version: u32 = 0x01;

/// AOT binary magic number
pub const aot_magic: u32 = 0x746f6100; // "\0aot"

/// AOT format version
pub const aot_version: u32 = 6;

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
    data: []u8,
    current_pages: u32,
    max_pages: u32,
    ref_count: u32 = 1,

    pub const page_size: u32 = 65536;

    pub fn grow(self: *MemoryInstance, delta: u32, allocator: std.mem.Allocator) !u32 {
        const old_pages = self.current_pages;
        const new_pages = std.math.add(u32, old_pages, delta) catch return error.MemoryGrowFailed;
        if (self.memory_type.limits.max) |max| {
            if (new_pages > max) return error.MemoryGrowFailed;
        }
        if (new_pages > 65536) return error.MemoryGrowFailed;
        const new_size = @as(usize, new_pages) * page_size;
        const old_size = self.data.len;
        if (old_size < new_size) {
            self.data = try allocator.realloc(self.data, new_size);
            // Zero-initialize the new pages (wasm spec requirement)
            @memset(self.data[old_size..new_size], 0);
        }
        self.current_pages = new_pages;
        return old_pages;
    }

    pub fn retain(self: *MemoryInstance) void {
        self.ref_count += 1;
    }

    pub fn release(self: *MemoryInstance, allocator: std.mem.Allocator) void {
        self.ref_count -= 1;
        if (self.ref_count == 0) {
            if (self.data.len > 0) allocator.free(self.data);
            allocator.destroy(self);
        }
    }
};

/// A function reference stored in a table element.
/// Tracks the source module for cross-module call_indirect dispatch.
pub const FuncRef = struct {
    func_idx: u32,
    module_inst: *ModuleInstance,
};

/// Runtime table instance (refcounted for cross-module sharing)
pub const TableInstance = struct {
    table_type: TableType,
    elements: []?FuncRef,
    ref_count: u32 = 1,

    pub fn retain(self: *TableInstance) void {
        self.ref_count += 1;
    }

    pub fn release(self: *TableInstance, allocator: std.mem.Allocator) void {
        self.ref_count -= 1;
        if (self.ref_count == 0) {
            if (self.elements.len > 0) allocator.free(self.elements);
            allocator.destroy(self);
        }
    }
};

/// Runtime global instance
pub const GlobalInstance = struct {
    global_type: GlobalType,
    value: Value,
    owned: bool = true,
    ref_count: u32 = 1,
    /// For funcref globals: the module instance that owns the referenced function
    source_module: ?*ModuleInstance = null,

    pub fn retain(self: *GlobalInstance) void {
        self.ref_count += 1;
    }

    pub fn release(self: *GlobalInstance, allocator: std.mem.Allocator) void {
        self.ref_count -= 1;
        if (self.ref_count == 0) {
            allocator.destroy(self);
        }
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

/// Instantiated module
pub const ModuleInstance = struct {
    module: *const WasmModule,
    memories: []*MemoryInstance,
    tables: []*TableInstance,
    globals: []*GlobalInstance,
    import_functions: []const ImportedFunction = &.{},
    tags: []*TagInstance = &.{},
    allocator: std.mem.Allocator,
    /// Track dropped elem segments (active segments dropped after instantiation)
    dropped_elems: []bool = &.{},
    /// Track dropped data segments (for data.drop instruction)
    dropped_data: []bool = &.{},

    pub fn getExportFunc(self: *const ModuleInstance, name: []const u8) ?u32 {
        const exp = self.module.findExport(name, .function) orelse return null;
        return exp.index;
    }

    pub fn getMemory(self: *const ModuleInstance, idx: u32) ?*MemoryInstance {
        if (idx < self.memories.len) return self.memories[idx];
        return null;
    }

    /// Clone this instance for a new thread (WASI-threads instance-per-thread model).
    /// Shared: memories, tables, import_functions (retained via ref_count).
    /// Cloned: globals (mutable globals are thread-local).
    pub fn cloneForThread(self: *const ModuleInstance, allocator: std.mem.Allocator) !*ModuleInstance {
        const inst = try allocator.create(ModuleInstance);
        errdefer allocator.destroy(inst);

        inst.* = .{
            .module = self.module,
            .memories = &.{},
            .tables = &.{},
            .globals = &.{},
            .import_functions = self.import_functions,
            .allocator = allocator,
        };

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
            }
        }

        // Clone dropped_elems
        if (self.dropped_elems.len > 0) {
            inst.dropped_elems = try allocator.alloc(bool, self.dropped_elems.len);
            @memcpy(inst.dropped_elems, self.dropped_elems);
        }

        return inst;
    }
};

// ─── Tests for module-level structures ──────────────────────────────────────

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
