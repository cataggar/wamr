//! Component Model types — in-memory AST for parsed components.
//!
//! Defines the data structures that represent a parsed WebAssembly Component
//! per the Component Model binary format specification. A Component is a
//! higher-level container that can embed core modules, other components, and
//! defines interface types, canonical functions, and resource lifecycles.

const std = @import("std");

// ── Primitive interface value types ─────────────────────────────────────────

/// Interface value types used in the Component Model type system.
/// These extend core Wasm value types with higher-level constructs.
pub const ValType = union(enum) {
    // Primitives
    bool,
    s8,
    u8,
    s16,
    u16,
    s32,
    u32,
    s64,
    u64,
    f32,
    f64,
    char,
    string,

    // Compound (index into component type index space)
    record: u32,
    variant: u32,
    list: u32,
    tuple: u32,
    flags: u32,
    enum_: u32,
    option: u32,
    result: u32,

    // Resource handles
    own: u32, // resource type index
    borrow: u32, // resource type index

    /// Type index reference (for recursive/named types).
    type_idx: u32,
};

// ── Compound type definitions ───────────────────────────────────────────────

pub const Field = struct {
    name: []const u8,
    type: ValType,
};

pub const Case = struct {
    name: []const u8,
    type: ?ValType, // null for cases with no payload
    /// Refined index for cases that refine other cases.
    refines: ?u32 = null,
};

pub const RecordType = struct {
    fields: []const Field,
};

pub const VariantType = struct {
    cases: []const Case,
};

pub const ListType = struct {
    element: ValType,
};

pub const TupleType = struct {
    fields: []const ValType,
};

pub const FlagsType = struct {
    names: []const []const u8,
};

pub const EnumType = struct {
    names: []const []const u8,
};

pub const OptionType = struct {
    inner: ValType,
};

pub const ResultType = struct {
    ok: ?ValType,
    err: ?ValType,
};

pub const ResourceType = struct {
    /// Destructor function index (in the canon function index space), or null.
    destructor: ?u32 = null,
    /// Representation type (always i32 per spec).
    rep: CoreValType = .i32,
};

// ── Function types ──────────────────────────────────────────────────────────

pub const NamedValType = struct {
    name: []const u8,
    type: ValType,
};

/// Component-level function type. Unlike core Wasm functions which use
/// value type stacks, component functions have named parameters and
/// can return either a single unnamed type or named results.
pub const FuncType = struct {
    params: []const NamedValType,
    results: ResultList,

    pub const ResultList = union(enum) {
        /// Single unnamed result type.
        unnamed: ValType,
        /// Named result types (like a record).
        named: []const NamedValType,
    };
};

// ── Core types within component scope ───────────────────────────────────────

/// Core Wasm value types (subset used in component core type definitions).
pub const CoreValType = enum(u8) {
    i32 = 0x7F,
    i64 = 0x7E,
    f32 = 0x7D,
    f64 = 0x7C,
};

pub const CoreFuncType = struct {
    params: []const CoreValType,
    results: []const CoreValType,
};

pub const CoreModuleType = struct {
    imports: []const CoreImportDecl,
    exports: []const CoreExportDecl,
};

pub const CoreImportDecl = struct {
    module: []const u8,
    name: []const u8,
    /// Type reference (function type index in core type space).
    type_idx: u32,
};

pub const CoreExportDecl = struct {
    name: []const u8,
    type_idx: u32,
};

// ── Component type definitions (section 7) ──────────────────────────────────

/// A type definition in the component type index space.
pub const TypeDef = union(enum) {
    // Compound types
    record: RecordType,
    variant: VariantType,
    list: ListType,
    tuple: TupleType,
    flags: FlagsType,
    enum_: EnumType,
    option: OptionType,
    result: ResultType,
    resource: ResourceType,

    // Function and component/instance types
    func: FuncType,
    component: ComponentTypeDecl,
    instance: InstanceTypeDecl,
};

/// Core type definition in the core type index space within a component.
pub const CoreTypeDef = union(enum) {
    func: CoreFuncType,
    module: CoreModuleType,
};

/// Declares the shape of a component type (its imports and exports).
pub const ComponentTypeDecl = struct {
    imports: []const ImportDecl,
    exports: []const ExportDecl,
};

/// Declares the shape of an instance type (its exports only).
pub const InstanceTypeDecl = struct {
    exports: []const ExportDecl,
};

// ── Sorts ───────────────────────────────────────────────────────────────────

/// Core-level sort discriminator (for core index spaces).
pub const CoreSort = enum(u8) {
    func = 0x00,
    table = 0x01,
    memory = 0x02,
    global = 0x03,
    tag = 0x04,
    type = 0x10,
    module = 0x11,
    instance = 0x12,
};

/// Component-level sort discriminator.
pub const Sort = union(enum) {
    core: CoreSort,
    func,
    value,
    type,
    component,
    instance,
};

/// A typed index: sort + index into that sort's index space.
pub const SortIdx = struct {
    sort: Sort,
    idx: u32,
};

// ── Aliases ─────────────────────────────────────────────────────────────────

pub const Alias = union(enum) {
    /// Alias an export of an instance: (alias export <instance> <name>)
    instance_export: struct {
        sort: Sort,
        instance_idx: u32,
        name: []const u8,
    },
    /// Alias from an outer component scope: (alias outer <count> <idx>)
    outer: struct {
        sort: Sort,
        outer_count: u32,
        idx: u32,
    },
};

// ── Canonical functions ─────────────────────────────────────────────────────

pub const StringEncoding = enum(u8) {
    utf8 = 0x00,
    utf16 = 0x01,
    latin1_utf16 = 0x02,
};

pub const CanonOpt = union(enum) {
    /// Linear memory to use for indirect loads/stores.
    memory: u32, // core memory index
    /// Realloc function for allocating memory in the callee.
    realloc: u32, // core func index
    /// Post-return cleanup function.
    post_return: u32, // core func index
    /// String encoding to use.
    string_encoding: StringEncoding,
};

/// Canonical function definitions.
pub const Canon = union(enum) {
    /// Lift a core function to a component function.
    lift: struct {
        core_func_idx: u32,
        type_idx: u32, // component func type
        opts: []const CanonOpt,
    },
    /// Lower a component function to a core function.
    lower: struct {
        func_idx: u32, // component func index
        opts: []const CanonOpt,
    },
    /// Create a new resource handle from a representation.
    resource_new: u32, // resource type index
    /// Drop a resource handle, calling its destructor.
    resource_drop: u32, // resource type index
    /// Get the representation of a resource handle.
    resource_rep: u32, // resource type index
};

// ── Imports and exports ─────────────────────────────────────────────────────

/// Extern descriptor for component-level imports/exports.
pub const ExternDesc = union(enum) {
    /// Module type (for core module imports).
    module: u32, // core type index
    /// Function type.
    func: u32, // component func type index
    /// Value type.
    value: ValType,
    /// Named type.
    type: TypeBound,
    /// Component type.
    component: u32, // type index
    /// Instance type.
    instance: u32, // type index
};

pub const TypeBound = union(enum) {
    /// Type must be equal to the given type index.
    eq: u32,
    /// Type must be a subtype of the given type index.
    sub_resource,
};

pub const ImportDecl = struct {
    name: []const u8,
    desc: ExternDesc,
};

pub const ExportDecl = struct {
    name: []const u8,
    desc: ExternDesc,
};

// ── Instance expressions ────────────────────────────────────────────────────

pub const CoreInstanceExpr = union(enum) {
    /// Instantiate a core module with arguments.
    instantiate: struct {
        module_idx: u32,
        args: []const CoreInstantiateArg,
    },
    /// Inline exports (bundle of named core items).
    exports: []const CoreInlineExport,
};

pub const CoreInstantiateArg = struct {
    name: []const u8,
    instance_idx: u32,
};

pub const CoreInlineExport = struct {
    name: []const u8,
    sort_idx: struct { sort: CoreSort, idx: u32 },
};

pub const InstanceExpr = union(enum) {
    /// Instantiate a component with arguments.
    instantiate: struct {
        component_idx: u32,
        args: []const InstantiateArg,
    },
    /// Inline exports.
    exports: []const InlineExport,
};

pub const InstantiateArg = struct {
    name: []const u8,
    sort_idx: SortIdx,
};

pub const InlineExport = struct {
    name: []const u8,
    sort_idx: SortIdx,
};

// ── Start function ──────────────────────────────────────────────────────────

pub const Start = struct {
    func_idx: u32,
    args: []const u32, // value indices
    results: u32, // number of result values
};

// ── Top-level Component ─────────────────────────────────────────────────────

/// A parsed WebAssembly Component.
///
/// Unlike core modules which have a fixed section order, components can
/// interleave sections. The `sections` list preserves the original order,
/// and each section's definitions are added to the appropriate index space.
pub const Component = struct {
    /// Core modules embedded in this component.
    core_modules: []const CoreModule,
    /// Core instances.
    core_instances: []const CoreInstanceExpr,
    /// Core type definitions.
    core_types: []const CoreTypeDef,
    /// Nested sub-components.
    components: []const *Component,
    /// Component-level instances.
    instances: []const InstanceExpr,
    /// Aliases.
    aliases: []const Alias,
    /// Component-level type definitions.
    types: []const TypeDef,
    /// Canonical function definitions.
    canons: []const Canon,
    /// Start function.
    start: ?Start = null,
    /// Component imports.
    imports: []const ImportDecl,
    /// Component exports.
    exports: []const ExportDecl,
};

/// A core module embedded within a component (stored as raw bytes
/// to be loaded on demand via the existing core loader).
pub const CoreModule = struct {
    /// Raw binary of the core module (including preamble).
    data: []const u8,
};

// ── Tests ───────────────────────────────────────────────────────────────────

test "ValType: primitive sizes" {
    const v1: ValType = .bool;
    const v2: ValType = .{ .record = 5 };
    const v3: ValType = .{ .own = 3 };
    try std.testing.expect(v1 == .bool);
    try std.testing.expect(v2 == .record);
    try std.testing.expect(v3 == .own);
}

test "TypeDef: record construction" {
    const fields = [_]Field{
        .{ .name = "x", .type = .s32 },
        .{ .name = "y", .type = .s32 },
    };
    const td = TypeDef{ .record = .{ .fields = &fields } };
    try std.testing.expect(td == .record);
    try std.testing.expectEqual(@as(usize, 2), td.record.fields.len);
}

test "Canon: lift construction" {
    const opts = [_]CanonOpt{
        .{ .memory = 0 },
        .{ .string_encoding = .utf8 },
    };
    const canon = Canon{ .lift = .{
        .core_func_idx = 0,
        .type_idx = 1,
        .opts = &opts,
    } };
    try std.testing.expect(canon == .lift);
    try std.testing.expectEqual(@as(usize, 2), canon.lift.opts.len);
}

test "Sort: core and component sorts" {
    const s1 = Sort{ .core = .func };
    const s2: Sort = .func;
    try std.testing.expect(s1 == .core);
    try std.testing.expect(s2 == .func);
}

test "Alias: instance export" {
    const a = Alias{ .instance_export = .{
        .sort = .func,
        .instance_idx = 0,
        .name = "my-func",
    } };
    try std.testing.expect(a == .instance_export);
    try std.testing.expectEqualStrings("my-func", a.instance_export.name);
}
