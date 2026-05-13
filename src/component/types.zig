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

    // Async ABI value types (Component Model 🔀 / 📝). Per Binary.md:
    //   0x64                       => error-context
    //   0x65 t?:<valtype>?         => (future t?)
    //   0x66 t?:<valtype>?         => (stream t?)
    // Sub-PR 3 of #478 only accepts the typeidx-carrying form; the
    // payload-less spelling (`0x65 0x00` / `0x66 0x00`) is rejected at
    // load with `error.InvalidEncoding` — Wasmtime's tests don't emit it
    // either, and supporting it cleanly requires a payload variant.
    error_context,
    future: u32, // typeidx of payload element
    stream: u32, // typeidx of payload element

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
        /// No result (spec: `0x01 0x00`).
        none,
        /// Single unnamed result type (spec: `0x00 <valtype>`).
        unnamed: ValType,
        /// Named result types. Retained for backward compatibility with older
        /// component-model encodings; no longer produced by the current spec.
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
    // Primitive / handle / indexed value type used directly as a type def
    // (per the `defvaltype` grammar, a type def can be a single valtype byte
    // e.g. `(type (borrow $r))` or `(type u32)` inside an instance-type body).
    val: ValType,

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

/// Declarator inside a component-type or instance-type declaration.
///
/// Per the component-model binary format, the declarator list mixes
/// type definitions, aliases, and import/export descriptors; their
/// relative order establishes the nested type-index space used by
/// subsequent declarators in the same list.
///
/// See: <https://github.com/WebAssembly/component-model/blob/main/design/mvp/Binary.md#type-definitions>
pub const Decl = union(enum) {
    core_type: CoreTypeDef,
    type: TypeDef,
    alias: Alias,
    import: ImportDecl,
    @"export": ExportDecl,
};

/// Declares the shape of a component type.
///
/// `decls` preserves the on-wire order of nested core-type, type, alias,
/// import, and export declarations. The `imports` and `exports` helpers
/// provide filtered views for callers that only care about externally
/// visible declarators.
pub const ComponentTypeDecl = struct {
    decls: []const Decl,

    pub fn importCount(self: ComponentTypeDecl) usize {
        var n: usize = 0;
        for (self.decls) |d| if (d == .import) {
            n += 1;
        };
        return n;
    }

    pub fn exportCount(self: ComponentTypeDecl) usize {
        var n: usize = 0;
        for (self.decls) |d| if (d == .@"export") {
            n += 1;
        };
        return n;
    }
};

/// Declares the shape of an instance type.
///
/// Like `ComponentTypeDecl` but without import decls.
pub const InstanceTypeDecl = struct {
    decls: []const Decl,

    pub fn exportCount(self: InstanceTypeDecl) usize {
        var n: usize = 0;
        for (self.decls) |d| if (d == .@"export") {
            n += 1;
        };
        return n;
    }
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
    /// Lift this function asynchronously (Binary.md canonopt tag `0x06`).
    /// Forces the canonical-ABI to return a packed status immediately and
    /// expect `task.return` to deliver the real results. (#478 sub-PR 2.)
    async_lift,
    /// Callback core funcidx for resumption after an async yield
    /// (Binary.md canonopt tag `0x07`). Single-threaded poll-cycle stub
    /// in sub-PR 2; sub-PR 3 wires it into the real `future`/`stream`
    /// polling story. (#478 sub-PR 2.)
    callback: u32,
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

    // ── Async ABI canonical built-ins (WASIp3 / Component Model 🔀) ──────────
    //
    // Binary tags per design/mvp/Binary.md "Canonical Definitions". The current
    // upstream spec renames `task.yield` → `thread.yield` but the surface lives
    // on the per-task state machine; #478 still calls it `task_yield` and
    // sub-PR 1 keeps that name. Tags here are sub-PR-1 scope; sub-PR 2/3 will
    // grow the union with subtask/future/stream variants.

    /// Cooperatively suspend the currently executing task. Tag `0x0c`.
    /// `cancellable == true` corresponds to `cancel? = 0x01` in the binary
    /// format; when set, the returned discriminant indicates whether the
    /// suspending task was cancelled while parked.
    task_yield: struct { cancellable: bool },

    /// Read a per-task context slot. Tag `0x0a`. The spec allows arbitrary
    /// `valtype`, but sub-PR 1 only admits `i32` (Wasmtime's current limit);
    /// other types are rejected at load with `error.InvalidEncoding`.
    context_get: struct { val_type: CoreValType, slot: u32 },

    /// Write a per-task context slot. Tag `0x0b`. Same `i32`-only restriction
    /// as `context_get`.
    context_set: struct { val_type: CoreValType, slot: u32 },

    /// Deliver the results of an async-lifted callee. Tag `0x09`. The
    /// `results` shape matches the lifted-callee's result types (encoded
    /// identically to `FuncType.ResultList`: `0x00 valtype` for one
    /// unnamed result, `0x01 0x00` for none). `opts` carries memory /
    /// realloc / string-encoding needed to lower compound results back
    /// into the caller's memory. The callee invokes this from inside the
    /// core-wasm body to transition its task from `.started` to
    /// `.returned`. (#478 sub-PR 2.)
    task_return: struct { results: FuncType.ResultList, opts: []const CanonOpt },

    /// Catch-all for the broader WASIp3 async-canon surface: subtask /
    /// future / stream / error-context / waitable-set / waitable.join.
    /// Each entry retains its binary-tag-specific payload via
    /// `AsyncCanonOp`. Sub-PR 3 of #478 lands the IR + loader + a
    /// minimum dispatch wiring so the conformance suite stops bailing
    /// on `error.InvalidEncoding`; semantics for some operations remain
    /// placeholders (documented per-arm in `dispatchCanonBuiltin`).
    async_canon: AsyncCanonOp,
};

/// Per-tag payload for `Canon.async_canon`. Tag bytes come straight from
/// Binary.md "Canonical Definitions" (the 🔀 / 📝-annotated entries that
/// aren't handled by their own dedicated `Canon` variants). Keeping a
/// single union variant on `Canon` (instead of ~17 separate ones) avoids
/// rippling exhaustive switches across every consumer.
pub const AsyncCanonOp = union(enum) {
    /// `canon subtask.cancel async?` — Binary.md tag `0x06`.
    subtask_cancel: struct { is_async: bool },
    /// `canon subtask.drop` — tag `0x0d`.
    subtask_drop,
    /// `canon stream.new t` — tag `0x0e`.
    stream_new: struct { type_idx: u32 },
    /// `canon stream.read t opts` — tag `0x0f`.
    stream_read: struct { type_idx: u32, opts: []const CanonOpt },
    /// `canon stream.write t opts` — tag `0x10`.
    stream_write: struct { type_idx: u32, opts: []const CanonOpt },
    /// `canon stream.cancel-read t async?` — tag `0x11`.
    stream_cancel_read: struct { type_idx: u32, is_async: bool },
    /// `canon stream.cancel-write t async?` — tag `0x12`.
    stream_cancel_write: struct { type_idx: u32, is_async: bool },
    /// `canon stream.drop-readable t` — tag `0x13`.
    stream_drop_readable: struct { type_idx: u32 },
    /// `canon stream.drop-writable t` — tag `0x14`.
    stream_drop_writable: struct { type_idx: u32 },
    /// `canon future.new t` — tag `0x15`.
    future_new: struct { type_idx: u32 },
    /// `canon future.read t opts` — tag `0x16`.
    future_read: struct { type_idx: u32, opts: []const CanonOpt },
    /// `canon future.write t opts` — tag `0x17`.
    future_write: struct { type_idx: u32, opts: []const CanonOpt },
    /// `canon future.cancel-read t async?` — tag `0x18`.
    future_cancel_read: struct { type_idx: u32, is_async: bool },
    /// `canon future.cancel-write t async?` — tag `0x19`.
    future_cancel_write: struct { type_idx: u32, is_async: bool },
    /// `canon future.drop-readable t` — tag `0x1a`.
    future_drop_readable: struct { type_idx: u32 },
    /// `canon future.drop-writable t` — tag `0x1b`.
    future_drop_writable: struct { type_idx: u32 },
    /// `canon error-context.new opts` — tag `0x1c`.
    error_context_new: struct { opts: []const CanonOpt },
    /// `canon error-context.debug-message opts` — tag `0x1d`.
    error_context_debug_message: struct { opts: []const CanonOpt },
    /// `canon error-context.drop` — tag `0x1e`.
    error_context_drop,
    /// `canon waitable-set.new` — tag `0x1f`.
    waitable_set_new,
    /// `canon waitable-set.wait cancel? (memory m)` — tag `0x20`.
    waitable_set_wait: struct { cancellable: bool, memory: u32 },
    /// `canon waitable-set.poll cancel? (memory m)` — tag `0x21`.
    waitable_set_poll: struct { cancellable: bool, memory: u32 },
    /// `canon waitable-set.drop` — tag `0x22`.
    waitable_set_drop,
    /// `canon waitable.join` — tag `0x23`.
    waitable_join,
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
    /// Set for top-level component exports (see spec `export` rule):
    ///   export ::= en:<exportname'> si:<sortidx> ed?:<externdesc>?
    /// Null for export declarators inside a component-type / instance-type
    /// body, which carry only a name and a descriptor.
    sort_idx: ?SortIdx = null,
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
    /// Type index-space → local `types[]` mapping. Each entry is the
    /// local idx into `types` for slots produced by a `(type ...)` def,
    /// or null for slots produced by a `(import ... (type ...))` /
    /// `(alias ... (type ...))` whose target def isn't materialized
    /// in `types`. When empty (hand-authored fixtures), callers fall
    /// back to direct indexing of `types`.
    type_indexspace: []const ?u32 = &.{},
    /// Canonical function definitions.
    canons: []const Canon,
    /// Start function.
    start: ?Start = null,
    /// Component imports.
    imports: []const ImportDecl,
    /// Component exports.
    exports: []const ExportDecl,
    /// Core-func index-space contributors in binary declaration order.
    /// Each entry records whether the slot was contributed by a canon
    /// or by an `Alias.instance_export` with `sort = .core(.func)`,
    /// along with the index into the corresponding per-section array.
    /// Empty when the component was constructed without a loader (e.g.
    /// hand-authored test fixtures); callers in that case fall back to
    /// the section-order heuristic in `indexspace.resolveCoreFunc`.
    core_func_indexspace: []const CoreFuncContributor = &.{},
    /// Component-instance index-space contributors in binary
    /// declaration order. Each entry records whether the slot was
    /// contributed by an `.instance`-typed `import`, an `instance`
    /// section entry, or an `alias` section entry of sort
    /// `.instance`. Empty when the component was constructed without
    /// a loader (e.g. hand-authored test fixtures); callers in that
    /// case fall back to the legacy "imports, then instances, then
    /// aliases" walk in `indexspace.resolveCompInstance`. The legacy
    /// walk is correct for non-composed components and Phase 2A
    /// fixtures; the section-ordered slice is required for
    /// `wasm-tools compose` output where instance and alias sections
    /// interleave (issue #355).
    comp_instance_indexspace: []const CompInstanceContributor = &.{},
};

/// A single contributor to the core-func index space.
pub const CoreFuncContributor = union(enum) {
    /// Index into `component.canons`. Only canon kinds that contribute
    /// to the core-func indexspace appear here — every kind except
    /// `.lift` (which produces a *component* func, not a core func).
    canon: u32,
    /// Index into `component.aliases`.
    alias: u32,
};

/// A single contributor to the component-instance index space, in the
/// order it appeared in the binary. Composed components (output by
/// `wasm-tools compose`) interleave `.instance` and `.alias` (sort
/// `.instance`) section entries — so the index space cannot be derived
/// by walking imports, then instances, then aliases as separate phases.
/// (Issue #355.)
pub const CompInstanceContributor = union(enum) {
    /// Index into `component.imports` (must be `.instance` desc).
    import: u32,
    /// Index into `component.instances`.
    instance: u32,
    /// Index into `component.aliases` (must be `instance_export` of
    /// sort `.instance`).
    alias: u32,
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
