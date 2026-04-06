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
        return self == .funcref or self == .externref;
    }

    pub fn byteSize(self: ValType) usize {
        return switch (self) {
            .i32, .f32 => 4,
            .i64, .f64 => 8,
            .v128 => 16,
            .funcref, .externref => @sizeOf(usize),
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
};

/// Function type (§2.3.5)
pub const FuncType = struct {
    params: []const ValType,
    results: []const ValType,
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

    pub const Mutability = enum(u1) {
        immutable = 0,
        mutable = 1,
    };
};

/// Import/Export kinds (§2.5)
pub const ExternalKind = enum(u8) {
    function = 0x00,
    table = 0x01,
    memory = 0x02,
    global = 0x03,
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
};

/// Global variable definition
pub const WasmGlobal = struct {
    global_type: GlobalType,
    init_expr: InitExpr,
};

/// Constant expression used for global/data/element initializers
pub const InitExpr = union(enum) {
    i32_const: i32,
    i64_const: i64,
    f32_const: f32,
    f64_const: f64,
    global_get: u32,
    ref_null: ValType,
    ref_func: u32,
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
    func_indices: []const u32,
    is_passive: bool = false,
    is_declarative: bool = false,

    pub const ElemKind = enum { func_ref, extern_ref };
};

/// Full parsed WebAssembly module
pub const WasmModule = struct {
    // Sections
    types: []const FuncType = &.{},
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

    // Derived counts (imports + local definitions)
    import_function_count: u32 = 0,
    import_table_count: u32 = 0,
    import_memory_count: u32 = 0,
    import_global_count: u32 = 0,

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
        const local_idx = func_idx - self.import_function_count;
        if (local_idx < self.functions.len) {
            const tidx = self.functions[local_idx].type_idx;
            return if (tidx < self.types.len) self.types[tidx] else null;
        }
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
        const new_pages = old_pages + delta;
        if (self.memory_type.limits.max) |max| {
            if (new_pages > max) return error.MemoryGrowFailed;
        }
        if (new_pages > 65536) return error.MemoryGrowFailed;
        const new_size = @as(usize, new_pages) * page_size;
        if (self.data.len < new_size) {
            self.data = try allocator.realloc(self.data, new_size);
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

/// Runtime table instance (refcounted for cross-module sharing)
pub const TableInstance = struct {
    table_type: TableType,
    elements: []?u32,
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
    allocator: std.mem.Allocator,

    pub fn getExportFunc(self: *const ModuleInstance, name: []const u8) ?u32 {
        const exp = self.module.findExport(name, .function) orelse return null;
        return exp.index;
    }

    pub fn getMemory(self: *const ModuleInstance, idx: u32) ?*MemoryInstance {
        if (idx < self.memories.len) return self.memories[idx];
        return null;
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
