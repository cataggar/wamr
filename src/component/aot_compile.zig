//! Component / core-module AOT precompilation (compile-only half).
//!
//! Split out of `src/component/aot.zig` so the runtime CLI (`wamr`)
//! can link against the manifest loader without dragging in the
//! compiler crate. The runtime-only declarations
//! (`LoadedManifest`, `loadManifest`, `defaultManifestPathFor`,
//! manifest JSON schema, `isComponent`) live in `aot.zig`; this file
//! owns `compileCoreWasm`, `precompileComponent`, and the
//! `emit_aot` field-table builders that only the compile side uses.
//!
//! `Manifest` / `ManifestModuleEntry` / `manifest_format_version` are
//! re-exported from `aot.zig` here so the on-disk JSON shape stays
//! single-source-of-truth in `aot.zig`.

const std = @import("std");
const builtin = @import("builtin");
const ctypes = @import("types.zig");
const component_loader = @import("loader.zig");
const aot = @import("aot.zig");
const core_backend = @import("core_backend.zig");
const aot_runtime = @import("../runtime/aot/runtime.zig");
const host_trampolines = @import("../runtime/aot/host_trampolines.zig");
const platform = @import("../platform/platform.zig");
const core_loader = @import("../runtime/interpreter/loader.zig");
const frontend = @import("../compiler/frontend.zig");
const passes = @import("../compiler/ir/passes.zig");
const verifier = @import("../compiler/ir/verifier.zig");
const ir = @import("../compiler/ir/ir.zig");
const lazy_jit = @import("../compiler/lazy_jit.zig");
const x86_64_compile = @import("../compiler/codegen/x86_64/compile.zig");
const aarch64_compile = @import("../compiler/codegen/aarch64/compile.zig");
const emit_aot = @import("../compiler/emit_aot.zig");
const name_section_mod = @import("../runtime/common/name_section.zig");
const core_types = @import("../runtime/common/types.zig");
const interp_instance = @import("../runtime/interpreter/instance.zig");
const config = @import("../config.zig");
const aot_bisect = @import("../compiler/aot_bisect.zig");
const codegen_cache = @import("../compiler/codegen_cache.zig");

// Re-export the on-disk JSON schema from `aot.zig` so `precompileComponent`
// and `loadManifest` agree on layout without duplicating the schema.
pub const Manifest = aot.Manifest;
pub const ManifestModuleEntry = aot.ManifestModuleEntry;
pub const manifest_format_version = aot.manifest_format_version;

pub const PrecompileError = error{
    InvalidComponent,
    CoreCompileFailed,
    WriteFailed,
    OutOfMemory,
    OpenDirFailed,
    JsonSerializationFailed,
};

/// Surface the IR verifier's diagnostic detail to stderr when a
/// per-pass verify check trips inside `precompileComponent`. The
/// verifier's `last_failure` thread-local already carries the
/// pass name, function/block/inst indices, vreg and a short
/// human-readable detail string; without printing it the only
/// thing the user sees is `error: CoreCompileFailed`, which
/// makes catching e.g. a #754-class operand-type mismatch
/// almost as painful as the original silent miscompile.
fn logVerifierFailure(err: anyerror) void {
    const f = verifier.last_failure;
    if (f.func_index == null and f.detail.len == 0) {
        std.log.err("aot-compile failed: {s}", .{@errorName(err)});
        return;
    }
    var buf: [512]u8 = undefined;
    var fbs: std.Io.Writer = .fixed(&buf);
    f.format(&fbs) catch {};
    const written = fbs.buffered();
    std.log.err("aot-compile failed: {s}", .{written});
}

/// Options controlling `precompileComponent` and per-core compilation.
pub const PrecompileOptions = struct {
    target_arch: passes.TargetArch = switch (builtin.cpu.arch) {
        .aarch64 => .aarch64,
        else => .x86_64,
    },
    /// When false, skip IR optimization passes (mirrors `wamrc compile -O0`).
    /// Useful for #743-style AOT-codegen bisection: if a bug disappears at
    /// `-O0` we know it's in the optimization pipeline; if it persists the
    /// bug is in the frontend / SSA / codegen.
    optimize: bool = true,
    /// #860: which pass pipeline to run when `optimize` is true. Defaults
    /// to `.full` (today's `defaultPassesForTarget` pipeline, unchanged)
    /// so `wamrc compile`'s output is untouched by this option's
    /// existence; the in-process JIT call sites in `src/main.zig`
    /// explicitly pass `.fast` as their default (overridable via
    /// `WAMR_JIT_FULL_OPT`) since compile latency there is part of the
    /// user-visible cold start rather than a one-time cost amortized
    /// over many runs of a cached `.cwasm`.
    pass_preset: passes.PassPreset = .full,
    /// #761 Phase 2: per-core codegen cache directory. When non-null,
    /// `precompileComponent` reads `<dir>/core<N>.cache` for each core
    /// (if present, header-compatible) to reuse cached per-function
    /// native code, and writes the updated cache back. Has no effect
    /// on `compileCoreWasm` itself — use `compileCoreWasmCached`
    /// directly for in-memory cache reuse.
    cache_dir: ?[]const u8 = null,
    /// #743 / #761: per-core module index used by `:mod=N` bisect
    /// filters. The component driver sets this to the per-core index
    /// before each `compileCoreWasm` call; the single-module
    /// `wamrc compile` path leaves it at the default 0 (which still
    /// matches `:mod=0` filters, as intended).
    module_idx: u32 = 0,
    /// Optional pass-loop timing/progress diagnostics, normally parsed by
    /// `wamrc` from `WAMR_AOT_PASS_TIMING*`.
    pass_timing: passes.PassTimingOptions = .{},
    /// Optional analysis recomputation diagnostics, normally parsed by
    /// `wamrc` from `WAMR_AOT_ANALYSIS_TIMING*`.
    analysis_timing: passes.AnalysisTimingOptions = .{},
    /// Optional native-codegen timing diagnostics (#778), normally parsed
    /// by `wamrc` from `WAMR_AOT_CODEGEN_TIMING*`.
    codegen_timing: passes.CodegenTimingOptions = .{},
    /// Optional per-function spill-cost diagnostics (#808 Lever 1),
    /// normally parsed by `wamrc` from `WAMR_AOT_SPILL_METRIC*`.
    spill_metric: passes.SpillMetricOptions = .{},
    /// Tail-duplication compile-time guard/cap options.
    tail_duplication: passes.TailDuplicationOptions = .{},
    /// IR verifier mode for optimized builds. Defaults match the historical
    /// component path: safe builds verify after each pass; release builds do
    /// not verify unless explicitly requested.
    verify_mode: verifier.VerifyMode = if (std.debug.runtime_safety) .after_each_pass else .off,
    /// Lazy-JIT opt-in. When true (and only ever meaningful under
    /// `config.lazy_jit`), `compileCoreWasmCached` computes the current
    /// x86_64 eligibility set (`lazy_jit.findLazyEligibleFunctions`) and
    /// emits stable entry stubs for deferred functions, deferring their
    /// real body compilation until first entry. Requires
    /// `cache_ctx.lazy_jit_out` to be set so the retained IR survives
    /// past this call — see `compileCoreWasmCached`'s doc comment.
    /// x86_64 only; `target_arch == .aarch64` with `lazy_jit = true`
    /// returns `error.CoreCompileFailed`.
    lazy_jit: bool = false,
};

/// #862 lazy-JIT design-spike: retained state a caller needs to compile
/// deferred (lazy-eligible) functions on demand after `compileCoreWasmCached`
/// returns. Populated only when `opts.lazy_jit` is true and
/// `CompileCacheCtx.lazy_jit_out` is non-null.
pub const LazyJitOut = struct {
    /// The module's lowered IR, kept alive (ownership moved here) so
    /// `ir_module.functions.items[i]` for `i in lazy_local_indices` can
    /// be compiled later via the same per-backend `compileFunction`
    /// entry point used eagerly. The *whole* module is retained for
    /// simplicity, not just the deferred functions — a known
    /// memory-inefficiency documented as follow-up work in
    /// docs/design/lazy-jit-spike.md. Call `.deinit()` when done (after
    /// destroying the corresponding `AotInstance`).
    ir_module: ir.IrModule = undefined,
    /// LOCAL function indices (matching `ir_module.functions.items`)
    /// that were skipped during this compile. Caller-owned; free with
    /// the allocator passed to `compileCoreWasmCached`.
    lazy_local_indices: []const u32 = &.{},
    /// Parallel to `lazy_local_indices`: `true` at position `i` means
    /// `lazy_local_indices[i]` needs the native trampoline mechanism
    /// (#888 — table/`ref.func`/`call_indirect`-reachable leaf
    /// function) rather than the lighter-weight text-section entry
    /// stub (#887 — direct-call-graph function). Caller-owned; free
    /// with the allocator passed to `compileCoreWasmCached`.
    needs_trampoline: []const bool = &.{},
};

/// Optional cache I/O for `compileCoreWasm` (#761 Phase 2).
pub const CompileCacheCtx = struct {
    /// Optional prior cache to consult for per-function reuse.
    /// Borrowed; not freed by the callee.
    reuse: ?*const codegen_cache.Cache = null,
    /// When non-null, the freshly-built cache (along with build-id /
    /// arch / abi / module_epoch suitable for the on-disk format) is
    /// moved here. Caller owns and must call `Cache.deinit(allocator)`.
    /// When null, the new cache is built and discarded internally
    /// (zero observable change from old behaviour).
    produced: ?*?codegen_cache.Cache = null,
    /// #862 lazy-JIT spike: when `opts.lazy_jit` is true, the retained
    /// IR + deferred-function index list are written here so the
    /// caller can compile them on demand later. Must be non-null
    /// whenever `opts.lazy_jit` is true (`compileCoreWasmCached`
    /// returns `error.CoreCompileFailed` otherwise, since silently
    /// dropping the retained IR would leak the skipped functions'
    /// compiled-ness forever).
    lazy_jit_out: ?*LazyJitOut = null,
};

/// AOT-compile a single core wasm module. Returns freshly-allocated
/// `.cwasm` bytes owned by the caller (free with `allocator.free`).
///
/// Pure data in / data out — no filesystem I/O — so the same helper
/// services both `precompileComponent` and any future programmatic
/// caller that wants per-core artifacts in memory.
pub fn compileCoreWasm(
    allocator: std.mem.Allocator,
    wasm_bytes: []const u8,
    opts: PrecompileOptions,
) PrecompileError![]u8 {
    return compileCoreWasmCached(allocator, wasm_bytes, opts, .{});
}

/// Cache-aware variant of `compileCoreWasm`. Pass a `cache_ctx` to
/// reuse cached per-function codegen from a prior compile and/or to
/// capture the freshly-built cache for persisting next to the cwasm.
pub fn compileCoreWasmCached(
    allocator: std.mem.Allocator,
    wasm_bytes: []const u8,
    opts: PrecompileOptions,
    cache_ctx: CompileCacheCtx,
) PrecompileError![]u8 {
    // Lifetime split mirrors `src/compiler/main.zig` (`wamrc compile`)
    // to bound peak memory on large modules (issue #640). The parsed
    // module lives in a transient `module_arena`; IR + passes +
    // codegen go on the outer GPA so per-function intermediates are
    // freed back as each pass completes; emit-side field tables live
    // in a short-lived `emit_arena`. Lumping everything on one arena
    // (the previous implementation) retained tens of GB on a
    // 12 610-function core.
    var module_arena = std.heap.ArenaAllocator.init(allocator);
    defer module_arena.deinit();
    const ma = module_arena.allocator();

    // Don't dupe `wasm_bytes`: `core_loader.load` only borrows from
    // the slice (returned `module` fields are slices into it), and
    // the caller's bytes outlive this call.
    const module = core_loader.load(wasm_bytes, ma) catch return error.CoreCompileFailed;

    var ir_module = frontend.lowerModule(&module, allocator) catch return error.CoreCompileFailed;
    var keep_ir_module_for_lazy_jit = false;
    defer if (!keep_ir_module_for_lazy_jit) ir_module.deinit();

    // Compute the lazy-eligibility set before codegen so it can be threaded
    // into the x86_64 backend's `lazy_skip`. Requires `cache_ctx.lazy_jit_out`
    // so the retained IR survives for on-demand body compilation later.
    var lazy_skip: []bool = &.{};
    defer if (lazy_skip.len > 0) allocator.free(lazy_skip);
    var lazy_needs_trampoline: []bool = &.{};
    defer if (lazy_needs_trampoline.len > 0) allocator.free(lazy_needs_trampoline);
    if (opts.lazy_jit) {
        if (cache_ctx.lazy_jit_out == null) return error.CoreCompileFailed;
        if (opts.target_arch != .x86_64) return error.CoreCompileFailed;
        const eligibility = lazy_jit.findLazyEligibleFunctions(
            &module,
            &ir_module,
            switch (opts.target_arch) {
                .x86_64 => .x86_64,
                .aarch64 => .aarch64,
            },
            allocator,
        ) catch return error.CoreCompileFailed;
        lazy_skip = eligibility.eligible;
        lazy_needs_trampoline = eligibility.needs_trampoline;
    }

    if (opts.optimize) {
        _ = passes.runPassesWithOptions(
            &ir_module,
            passes.passesForPreset(opts.target_arch, opts.pass_preset),
            allocator,
            .{
                // The default `opts.verify_mode` matches single-module
                // `wamrc compile`: run the IR verifier after every pass in
                // safe builds, and keep it off in release unless requested.
                // Components are heavy
                // (the StarlingMonkey-bundled wasms used by jco have
                // ~12 k functions), but verifier-caught bugs like #754
                // — operand-type-mismatch on `select` propagating
                // through chained selects — are silent miscompiles
                // that cost far more to debug at runtime than to catch
                // here, but `wamrc compile-component --verify-ir=off` can
                // force it off for compile-time triage.
                //
                // Note: `-O0` (this branch's else arm) intentionally
                // skips the verifier because the frontend emits
                // dead instructions after `unreachable` / unconditional
                // `br` that `scrubUnreachableBlocks` (a runPasses-
                // internal cleanup, not a public pass) trims before
                // verify — without that cleanup the structural
                // checks trip benign `MultipleTerminators` errors.
                // Users who want belt-and-suspenders verification on
                // an `-O0` build should drop `-O0` instead.
                .verify_mode = opts.verify_mode,
                // #761 / #743: thread the global bisect spec through
                // so `WAMR_AOT_SKIP_PASS=...:fn=<idx>` narrows the
                // partial pipeline to one suspect function. The pass
                // loop forces verify off per-function for any function
                // affected by the spec to keep partial-pipeline IR
                // states from tripping benign structural checks.
                .bisect = aot_bisect.global,
                // Per-core index honoured by `:mod=N` bisect filters.
                .module_idx = opts.module_idx,
                .pass_timing = opts.pass_timing,
                .analysis_timing = opts.analysis_timing,
                .tail_duplication = opts.tail_duplication,
            },
        ) catch |err| {
            logVerifierFailure(err);
            return error.CoreCompileFailed;
        };
    }

    // #761 Phase 2: drive codegen through the cache-aware path so a
    // caller-provided `reuse` short-circuits per-function compile for
    // matching IR hashes, and the freshly-built cache can be moved
    // back to the caller (typically the `wamrc compile-component
    // --cache-dir <dir>` driver, which then persists one file per
    // core). When `cache_ctx.reuse == null` and `cache_ctx.produced
    // == null` the behaviour is identical to the historical
    // non-cached path — we build the cache in memory and immediately
    // free it on return.
    const target_abi = codegen_cache.TargetAbi.forHost(opts.target_arch);
    const epoch_inputs: codegen_cache.ModuleEpochInputs = .{
        .wamr_build_id = config.version,
        .target_arch = opts.target_arch,
        .target_abi = target_abi,
        .import_count = ir_module.import_count,
        .global_types = ir_module.global_types,
        .global_offsets = ir_module.global_offsets,
        .global_storage_size = ir_module.global_storage_size,
        .func_types = ir_module.func_types.items,
        .func_type_indices = ir_module.func_type_indices.items,
    };
    const module_epoch = codegen_cache.hashModuleEpoch(epoch_inputs);

    const compiled: codegen_cache.CompileResultCached = switch (opts.target_arch) {
        .aarch64 => aarch64_compile.compileModuleCachedWithOptions(&ir_module, cache_ctx.reuse, allocator, .{
            .codegen_timing = opts.codegen_timing,
            .spill_metric = opts.spill_metric,
            .module_idx = opts.module_idx,
        }) catch
            return error.CoreCompileFailed,
        .x86_64 => x86_64_compile.compileModuleCachedWithOptions(&ir_module, cache_ctx.reuse, allocator, .{
            .codegen_timing = opts.codegen_timing,
            .spill_metric = opts.spill_metric,
            .module_idx = opts.module_idx,
            .lazy_skip = lazy_skip,
        }) catch
            return error.CoreCompileFailed,
    };
    const code = compiled.code;
    const offsets = compiled.offsets;
    defer allocator.free(code);
    defer allocator.free(offsets);

    // Decide ownership of the freshly-built per-function cache entries.
    // If the caller wants the cache, we move it into `*produced`;
    // otherwise we own and free it on return.
    var release_cache_funcs = true;
    defer if (release_cache_funcs) {
        for (compiled.cache_functions) |*f| {
            allocator.free(f.code);
            allocator.free(f.call_patches);
        }
        allocator.free(compiled.cache_functions);
    };
    if (cache_ctx.produced) |dst| {
        const build_id_dup = allocator.dupe(u8, config.version) catch return error.OutOfMemory;
        errdefer allocator.free(build_id_dup);
        dst.* = codegen_cache.Cache{
            .wamr_build_id = build_id_dup,
            .target_arch = opts.target_arch,
            .target_abi = target_abi,
            .module_epoch = module_epoch,
            .functions = compiled.cache_functions,
        };
        release_cache_funcs = false;
    }

    // Short-lived arena for the emit-side field tables. `emit_aot.emit`
    // copies what it needs into its caller-owned output buffer, so
    // everything in here (and in `module_arena`) is safe to drop on
    // return.
    var emit_arena = std.heap.ArenaAllocator.init(allocator);
    defer emit_arena.deinit();
    const ea = emit_arena.allocator();

    var exports: std.ArrayList(emit_aot.ExportEntry) = .empty;
    for (module.exports) |exp| {
        if (exp.kind == .tag) continue;
        exports.append(ea, .{
            .name = exp.name,
            .kind = @enumFromInt(@intFromEnum(exp.kind)),
            .index = exp.index,
        }) catch return error.OutOfMemory;
    }

    var imports: std.ArrayList(emit_aot.ImportEntry) = .empty;
    for (module.imports) |imp| {
        switch (imp.kind) {
            .function => imports.append(ea, .{
                .module_name = imp.module_name,
                .field_name = imp.field_name,
                .kind = .function,
                .func_type_idx = imp.func_type_idx orelse 0,
            }) catch return error.OutOfMemory,
            .table => {
                const table_type = imp.table_type orelse continue;
                imports.append(ea, .{
                    .module_name = imp.module_name,
                    .field_name = imp.field_name,
                    .kind = .table,
                    .table_elem_type = table_type.elem_type,
                    .table_min = @intCast(table_type.limits.min),
                    .table_max = if (table_type.limits.max) |m| @as(?u32, @intCast(m)) else null,
                }) catch return error.OutOfMemory;
            },
            .memory => {
                const memory_type = imp.memory_type orelse continue;
                imports.append(ea, .{
                    .module_name = imp.module_name,
                    .field_name = imp.field_name,
                    .kind = .memory,
                    .memory_min = @intCast(memory_type.limits.min),
                    .memory_max = if (memory_type.limits.max) |m| @as(?u32, @intCast(m)) else null,
                    .memory_is64 = memory_type.is_memory64,
                }) catch return error.OutOfMemory;
            },
            .global => {
                const global_type = imp.global_type orelse continue;
                imports.append(ea, .{
                    .module_name = imp.module_name,
                    .field_name = imp.field_name,
                    .kind = .global,
                    .global_val_type = global_type.val_type,
                    .global_mutable = global_type.mutability == .mutable,
                }) catch return error.OutOfMemory;
            },
            .tag => {
                const type_idx = imp.tag_type_idx orelse continue;
                imports.append(ea, .{
                    .module_name = imp.module_name,
                    .field_name = imp.field_name,
                    .kind = .tag,
                    .tag_type_idx = type_idx,
                }) catch return error.OutOfMemory;
            },
        }
    }

    var mem_entries: std.ArrayList(emit_aot.MemoryEntry) = .empty;
    for (module.memories) |mem| {
        mem_entries.append(ea, .{
            .min_pages = @intCast(mem.limits.min),
            .max_pages = if (mem.limits.max) |m| @as(?u32, @intCast(m)) else null,
        }) catch return error.OutOfMemory;
    }

    // Locally-defined tables (#681). Imported tables already round-trip
    // via `import_entries`; local tables need section 15 so the loader
    // populates `module.tables` and `allocateTables` creates the matching
    // `TableInstance`s. wit-component cores rely on this for
    // `(table $imports)`.
    var table_entries: std.ArrayList(emit_aot.TableEntry) = .empty;
    for (module.tables) |t| {
        table_entries.append(ea, .{
            .elem_type = t.elem_type,
            .min = @intCast(t.limits.min),
            .max = if (t.limits.max) |m| @as(?u32, @intCast(m)) else null,
        }) catch return error.OutOfMemory;
    }

    var data_segs: std.ArrayList(emit_aot.DataSegmentEntry) = .empty;
    for (module.data_segments) |seg| {
        if (seg.is_passive) continue;
        const offset: u32 = switch (seg.offset) {
            .i32_const => |v| @bitCast(v),
            else => continue,
        };
        data_segs.append(ea, .{
            .memory_idx = seg.memory_idx,
            .offset = offset,
            .data = seg.data,
        }) catch return error.OutOfMemory;
    }

    var func_type_entries: std.ArrayList(emit_aot.FuncTypeEntry) = .empty;
    for (module.types) |ft| {
        if (ft.kind != .func) {
            func_type_entries.append(ea, .{ .params = &.{}, .results = &.{} }) catch return error.OutOfMemory;
            continue;
        }
        const params_bytes = ea.alloc(u8, ft.params.len) catch return error.OutOfMemory;
        for (ft.params, 0..) |p, j| params_bytes[j] = @intFromEnum(p);
        const results_bytes = ea.alloc(u8, ft.results.len) catch return error.OutOfMemory;
        for (ft.results, 0..) |r, j| results_bytes[j] = @intFromEnum(r);
        func_type_entries.append(ea, .{ .params = params_bytes, .results = results_bytes }) catch return error.OutOfMemory;
    }

    var local_func_tidx_list: std.ArrayList(u32) = .empty;
    for (module.functions) |f| {
        local_func_tidx_list.append(ea, f.type_idx) catch return error.OutOfMemory;
    }

    // Locally-declared tags (#672). `module.tag_types` is parallel to the
    // wasm tag section: each entry is a function-type index describing the
    // exception parameters.
    var tag_entries: std.ArrayList(emit_aot.TagEntry) = .empty;
    for (module.tag_types) |type_idx| {
        tag_entries.append(ea, .{ .type_idx = type_idx }) catch return error.OutOfMemory;
    }

    // Build global entries in wasm-flat order (imported globals first,
    // then local globals). `evalInitExpr` is shared with the on-disk
    // wamrc path so the resulting `.cwasm` is byte-identical for this
    // section. `tmp_globals` is the running prefix `evalInitExpr` uses
    // to resolve `global.get` in a `global` init expression.
    var global_entries: std.ArrayList(emit_aot.GlobalEntry) = .empty;
    var tmp_globals: std.ArrayList(*core_types.GlobalInstance) = .empty;
    defer {
        for (tmp_globals.items) |g| allocator.destroy(g);
        tmp_globals.deinit(allocator);
    }
    for (module.imports) |imp| {
        if (imp.kind != .global) continue;
        const gt = imp.global_type orelse continue;
        const val = defaultZeroValue(gt.val_type);
        const gi = allocator.create(core_types.GlobalInstance) catch return error.OutOfMemory;
        gi.* = .{ .global_type = gt, .value = val };
        tmp_globals.append(allocator, gi) catch return error.OutOfMemory;
        global_entries.append(ea, .{
            .val_type = @intFromEnum(gt.val_type),
            .mutability = if (gt.mutability == .mutable) @as(u8, 1) else @as(u8, 0),
            .init_i64 = valueToI64(val),
            .init_v128 = valueToV128(val),
        }) catch return error.OutOfMemory;
    }
    for (module.globals) |g| {
        const val = interp_instance.evalInitExpr(g.init_expr, tmp_globals.items, null) catch defaultZeroValue(g.global_type.val_type);
        const gi = allocator.create(core_types.GlobalInstance) catch return error.OutOfMemory;
        gi.* = .{ .global_type = g.global_type, .value = val };
        tmp_globals.append(allocator, gi) catch return error.OutOfMemory;
        global_entries.append(ea, .{
            .val_type = @intFromEnum(g.global_type.val_type),
            .mutability = if (g.global_type.mutability == .mutable) @as(u8, 1) else @as(u8, 0),
            .init_i64 = valueToI64(val),
            .init_v128 = valueToV128(val),
        }) catch return error.OutOfMemory;
    }

    // Build element segment entries. Declarative segments contribute
    // nothing at runtime; passive segments emit with offset = 0 and
    // `is_passive = true` (only usable via `table.init`). Funcidx
    // null sentinels are encoded as `0xFFFFFFFF`, matching the loader.
    var elem_entries: std.ArrayList(emit_aot.ElemEntry) = .empty;
    for (module.elements) |seg| {
        if (seg.is_declarative) continue;
        const offset: u32 = if (seg.is_passive) 0 else blk: {
            const off = seg.offset orelse continue;
            break :blk switch (off) {
                .i32_const => |v| @as(u32, @bitCast(v)),
                else => continue,
            };
        };
        const indices = ea.alloc(u32, seg.func_indices.len) catch return error.OutOfMemory;
        for (seg.func_indices, 0..) |fi, j| {
            indices[j] = fi orelse 0xFFFFFFFF;
        }
        elem_entries.append(ea, .{
            .table_idx = seg.table_idx,
            .offset = offset,
            .func_indices = indices,
            .is_passive = seg.is_passive,
        }) catch return error.OutOfMemory;
    }

    var arch_name = std.mem.zeroes([16]u8);
    switch (opts.target_arch) {
        .x86_64 => @memcpy(arch_name[0..6], "x86-64"),
        .aarch64 => @memcpy(arch_name[0..7], "aarch64"),
    }

    // Parse the wasm `name` custom section directly from the source
    // bytes for trap-decode diagnostics (#694). The interpreter loader
    // skips custom sections, so this is a separate pass over the same
    // buffer. A malformed or absent name section is non-fatal:
    // fall back to no names and let the trap helpers print
    // `local_func[N]` without a symbol.
    const fn_name_entries: ?[]emit_aot.FunctionNameEntry = blk: {
        const parsed = name_section_mod.parseFunctionNames(wasm_bytes, ea) catch break :blk null;
        if (parsed.len == 0) break :blk null;
        const entries = ea.alloc(emit_aot.FunctionNameEntry, parsed.len) catch break :blk null;
        for (parsed, 0..) |p, idx| {
            entries[idx] = .{ .index = p.index, .name = p.name };
        }
        break :blk entries;
    };

    const cwasm = emit_aot.emit(
        allocator,
        code,
        offsets,
        exports.items,
        .{ .arch = arch_name },
        if (data_segs.items.len > 0) data_segs.items else null,
        if (imports.items.len > 0) imports.items else null,
        if (mem_entries.items.len > 0) mem_entries.items else null,
        // Plain wasm32-wasip1 modules (AssemblyScript / Rust / clang
        // wasi-libc) carry a mutable `i32` shadow stack pointer global
        // that every function prologue reads/writes; failing to emit it
        // leaves SP=0 at startup and AS aborts inside its runtime with
        // `abort:  in (1:1)`. Componentize-js cores tolerate empty
        // globals/elems because canon-lift wraps them; standalone cores
        // do not. See `phase-b-diagnosis.md` (#662 Phase B).
        if (global_entries.items.len > 0) global_entries.items else null,
        if (elem_entries.items.len > 0) elem_entries.items else null,
        module.start_function,
        if (func_type_entries.items.len > 0) func_type_entries.items else null,
        if (local_func_tidx_list.items.len > 0) local_func_tidx_list.items else null,
        if (tag_entries.items.len > 0) tag_entries.items else null,
        if (table_entries.items.len > 0) table_entries.items else null,
        fn_name_entries,
    ) catch return error.CoreCompileFailed;

    // Hand the retained IR + deferred-function index list back to the caller
    // instead of freeing it, so the functions in `lazy_skip` can be compiled
    // on demand later. Convert
    // the dense `lazy_skip: []bool` into a compact index list here so
    // the caller doesn't need to re-derive it.
    if (opts.lazy_jit) {
        var indices: std.ArrayList(u32) = .empty;
        errdefer indices.deinit(allocator);
        var needs_trampoline: std.ArrayList(bool) = .empty;
        errdefer needs_trampoline.deinit(allocator);
        for (lazy_skip, 0..) |skip, idx| {
            if (skip) {
                indices.append(allocator, @intCast(idx)) catch return error.CoreCompileFailed;
                const needs_tramp = idx < lazy_needs_trampoline.len and lazy_needs_trampoline[idx];
                needs_trampoline.append(allocator, needs_tramp) catch return error.CoreCompileFailed;
            }
        }
        const out = cache_ctx.lazy_jit_out orelse unreachable; // checked above
        out.* = .{
            .ir_module = ir_module,
            .lazy_local_indices = indices.toOwnedSlice(allocator) catch return error.CoreCompileFailed,
            .needs_trampoline = needs_trampoline.toOwnedSlice(allocator) catch return error.CoreCompileFailed,
        };
        keep_ir_module_for_lazy_jit = true;
    }

    return cwasm;
}

/// #862 lazy-JIT design-spike: heap-owned driver bridging a `LazyJitOut`
/// (retained IR + deferred indices) to `AotInstance`'s type-erased
/// `LazyJitState.compile_fn` hook. x86_64 only, matching
/// `PrecompileOptions.lazy_jit`'s scope.
///
/// Lifetime: caller creates one via `setupLazyJit` right after
/// instantiating the `AotInstance` these functions belong to, and must
/// call `deinit` only AFTER destroying that `AotInstance` (the instance
/// may call back into this driver — via `inst.lazy_jit.compile_fn` —
/// any time up until `destroy()`).
pub const LazyCompileDriver = struct {
    lazy_out: LazyJitOut,
    allocator: std.mem.Allocator,

    fn compileFn(ctx_opaque: *anyopaque, local_idx: u32) aot_runtime.RuntimeError!aot_runtime.LazyCompiledFunc {
        const self: *LazyCompileDriver = @ptrCast(@alignCast(ctx_opaque));
        if (local_idx >= self.lazy_out.ir_module.functions.items.len) return error.CodeMappingFailed;
        const func = &self.lazy_out.ir_module.functions.items[local_idx];
        // Real regalloc-based per-function codegen — the SAME entry point
        // `compileModuleCachedWithOptions`'s per-function loop uses, not the
        // naive standalone `compileFunction`. Lazy bodies lower local direct
        // calls indirectly through vmctx.funcptrs_ptr so separately mapped
        // code never needs inter-function rel32 patching.
        const result = x86_64_compile.compileFunctionRAWithGlobalOffsetsPublic(
            func,
            self.lazy_out.ir_module.import_count,
            self.lazy_out.ir_module.global_offsets orelse &.{},
            self.allocator,
            .{ .local_call_lowering = .via_funcptrs },
        ) catch |err| {
            std.log.err("lazy-JIT spike: compiling deferred function {d} failed: {s}", .{ local_idx, @errorName(err) });
            return error.CodeMappingFailed;
        };
        if (result.call_patches.len > 0) {
            std.log.err("lazy-JIT: deferred function {d} unexpectedly emitted {d} call patches", .{ local_idx, result.call_patches.len });
            self.allocator.free(result.call_patches);
            self.allocator.free(result.code);
            return error.CodeMappingFailed;
        }
        defer self.allocator.free(result.call_patches);
        defer self.allocator.free(result.code);
        return aot_runtime.mapTrackedExecutableCode(result.code) catch |err| {
            if (err != error.CodeBudgetExceeded) {
                std.log.err("lazy-JIT spike: tracked executable mapping failed for deferred function {d}: {s}", .{ local_idx, @errorName(err) });
            }
            return err;
        };
    }

    pub fn deinit(self: *LazyCompileDriver) void {
        var out = self.lazy_out;
        out.ir_module.deinit();
        self.allocator.free(out.lazy_local_indices);
        if (out.needs_trampoline.len > 0) self.allocator.free(out.needs_trampoline);
        self.allocator.destroy(self);
    }
};

/// Wire `lazy_out` (produced by a `compileCoreWasmCached` call with
/// `opts.lazy_jit = true`) into `inst`, marking every deferred local's
/// slot state as `pending` and preallocating one stable trampoline
/// stub for each deferred local that needs the trampoline mechanism
/// (`lazy_out.needs_trampoline`; #888). Call this BEFORE
/// `mapCodeExecutable()` so the published `funcptrs` / table state can
/// reuse those stub addresses. Returns the heap-owned driver — see
/// `LazyCompileDriver`'s doc comment for its lifetime contract. `inst`
/// must have been produced from the SAME compile (its `.cwasm` bytes
/// came from the same `compileCoreWasmCached` call whose
/// `lazy_jit_out` is `lazy_out`) — indices are meaningless otherwise.
pub fn setupLazyJit(
    inst: *aot_runtime.AotInstance,
    lazy_out: LazyJitOut,
    allocator: std.mem.Allocator,
) !*LazyCompileDriver {
    const driver = try allocator.create(LazyCompileDriver);
    errdefer allocator.destroy(driver);
    driver.* = .{ .lazy_out = lazy_out, .allocator = allocator };

    const func_count = lazy_out.ir_module.functions.items.len;
    const slot_states = try allocator.alloc(std.atomic.Value(u8), func_count);
    errdefer allocator.free(slot_states);
    for (slot_states) |*slot| {
        slot.* = std.atomic.Value(u8).init(@intFromEnum(aot_runtime.LazyJitState.SlotState.inactive));
    }
    const compiled = try allocator.alloc(?aot_runtime.LazyCompiledFunc, func_count);
    errdefer allocator.free(compiled);
    @memset(compiled, null);
    const trampolines = try allocator.alloc(usize, func_count);
    errdefer allocator.free(trampolines);
    @memset(trampolines, 0);

    var trampoline_count: u32 = 0;
    for (lazy_out.needs_trampoline) |nt| {
        if (nt) trampoline_count += 1;
    }

    var pool_ptr: ?*host_trampolines.TrampolinePool = null;
    errdefer if (pool_ptr) |pool| {
        pool.deinit(allocator);
        allocator.destroy(pool);
    };
    if (trampoline_count > 0) {
        const pool = try allocator.create(host_trampolines.TrampolinePool);
        errdefer allocator.destroy(pool);
        pool.* = try host_trampolines.TrampolinePool.initWithCap(
            allocator,
            trampoline_count,
        );
        pool_ptr = pool;
    }

    for (lazy_out.lazy_local_indices, 0..) |idx, pos| {
        if (idx < slot_states.len) {
            slot_states[idx].store(@intFromEnum(aot_runtime.LazyJitState.SlotState.pending), .monotonic);
        }

        const needs_tramp = pos < lazy_out.needs_trampoline.len and lazy_out.needs_trampoline[pos];
        if (!needs_tramp) continue;
        const pool = pool_ptr orelse continue;
        if (idx >= inst.module.local_func_type_indices.len) return error.InvalidFuncType;
        const type_idx = inst.module.local_func_type_indices[idx];
        if (type_idx >= inst.module.func_types.len) return error.InvalidFuncType;
        const ft = inst.module.func_types[type_idx];
        const stub = try pool.allocLazyAotSlot(@ptrCast(inst), idx, .{
            .param_types = ft.params,
            .result_types = ft.results,
            .has_retptr = ft.results.len > 1,
        });
        trampolines[idx] = @intFromPtr(stub);
    }

    inst.lazy_jit = .{
        .slot_states = slot_states,
        .compiled = compiled,
        .trampoline_pool = pool_ptr,
        .trampolines = trampolines,
        .compile_ctx = driver,
        .compile_fn = &LazyCompileDriver.compileFn,
    };
    return driver;
}

// ── Global / element entry builders ─────────────────────────────────────
//
// Mirrors `src/compiler/main.zig`'s `runCompile` so the in-process AOT
// compile produces the same `.cwasm` layout the on-disk path does. The
// helpers here are deliberately self-contained (no cross-imports of
// file-private functions in `compiler/main.zig`) to keep this fix
// surgical; the duplication can be deduplicated in a follow-up by
// hoisting them into `emit_aot.zig` or a new shared file.

fn defaultZeroValue(vt: core_types.ValType) core_types.Value {
    return switch (vt) {
        .i32 => .{ .i32 = 0 },
        .i64 => .{ .i64 = 0 },
        .f32 => .{ .f32 = 0 },
        .f64 => .{ .f64 = 0 },
        .v128 => .{ .v128 = 0 },
        .funcref => .{ .funcref = null },
        .externref => .{ .externref = null },
        .nonfuncref => .{ .nonfuncref = null },
        .nonexternref => .{ .nonexternref = null },
        else => .{ .i64 = 0 },
    };
}

fn valueToI64(v: core_types.Value) i64 {
    return switch (v) {
        .i32 => |x| @as(i64, @as(u32, @bitCast(x))),
        .i64 => |x| x,
        .f32 => |x| @as(i64, @as(u32, @bitCast(x))),
        .f64 => |x| @as(i64, @bitCast(x)),
        .funcref, .nonfuncref => |maybe| if (maybe) |x| @as(i64, @as(u32, x)) + 1 else 0,
        .externref, .nonexternref => |maybe| if (maybe) |x| @as(i64, @as(u32, x)) + 1 else 0,
        else => 0,
    };
}

fn valueToV128(v: core_types.Value) u128 {
    return switch (v) {
        .v128 => |x| x,
        else => 0,
    };
}

/// Hex-encode a sha256 digest into a fresh allocator-owned string.
fn hexSha256(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = try allocator.alloc(u8, digest.len * 2);
    const hex_chars = "0123456789abcdef";
    for (digest, 0..) |b, i| {
        hex[i * 2] = hex_chars[b >> 4];
        hex[i * 2 + 1] = hex_chars[b & 0x0f];
    }
    return hex;
}

/// Result of `precompileComponent`. The returned `Manifest`'s strings
/// are arena-allocated and freed by `deinit` together with the arena.
pub const PrecompileResult = struct {
    manifest: Manifest,
    arena: *std.heap.ArenaAllocator,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PrecompileResult) void {
        self.arena.deinit();
        self.allocator.destroy(self.arena);
    }
};

/// Precompile every embedded core module of `component_bytes` and
/// write them as `<stem>.<idx>.cwasm` next to `manifest_path`, with
/// the manifest JSON itself at `manifest_path`. The "stem" is the
/// basename of `manifest_path` with the `.cwasm.json` suffix stripped
/// (or the whole basename if the suffix is absent); cores share that
/// stem so multiple components can coexist in the same directory.
///
/// The parent directory of `manifest_path` is created if missing.
/// Existing files at the target paths are overwritten — callers
/// responsible for stale-cache management beyond the manifest's
/// build-id check.
pub fn precompileComponent(
    allocator: std.mem.Allocator,
    component_bytes: []const u8,
    manifest_path: []const u8,
    opts: PrecompileOptions,
) PrecompileError!PrecompileResult {
    // Snapshot per-core byte slices off the parsed component, then
    // drop `load_arena` before the compile loop so the parsed
    // component (large for componentize-js workloads) isn't live in
    // memory while each core is being AOT-compiled. The `core_mod.data`
    // slices borrow from `component_bytes` (owned by the caller), so
    // they remain valid after `load_arena.deinit()`.
    //
    // Recurses into `component.components` so cores inside nested
    // sub-components (the dominant shape of `wabt component compose -d`
    // / `wasm-tools compose` output) are precompiled too — without
    // this the manifest is always empty for composed components and
    // `--precompiled-manifest` falls back to in-memory AOT on every
    // cold start. (#676)
    const CoreRef = struct {
        data: []const u8,
        /// Position within the `core_modules[]` of the (sub-)component
        /// that directly contains this core. Persisted as
        /// `ManifestModuleEntry.idx` for human debugging and for the
        /// v1-fallback loader path.
        local_idx: u32,
    };
    var core_data_list: std.ArrayList(CoreRef) = .empty;
    defer core_data_list.deinit(allocator);
    {
        var load_arena = std.heap.ArenaAllocator.init(allocator);
        defer load_arena.deinit();
        const component = component_loader.load(component_bytes, load_arena.allocator()) catch
            return error.InvalidComponent;
        const Walker = struct {
            fn walk(
                comp: *const ctypes.Component,
                list: *std.ArrayList(CoreRef),
                alloc: std.mem.Allocator,
            ) PrecompileError!void {
                for (comp.core_modules, 0..) |cm, mi| {
                    list.append(alloc, .{ .data = cm.data, .local_idx = @intCast(mi) }) catch
                        return error.OutOfMemory;
                }
                for (comp.components) |child| {
                    try walk(child, list, alloc);
                }
            }
        };
        try Walker.walk(&component, &core_data_list, allocator);
    }

    // Result-owned arena holds all strings the caller might read off
    // the returned Manifest (paths, hex hashes, build id).
    const result_arena = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(result_arena);
    result_arena.* = std.heap.ArenaAllocator.init(allocator);
    errdefer result_arena.deinit();
    const ra = result_arena.allocator();

    const cwd = std.Io.Dir.cwd();
    const io = std.Io.Threaded.global_single_threaded.io();
    const parent_dir_path = std.fs.path.dirname(manifest_path) orelse ".";
    const manifest_filename = std.fs.path.basename(manifest_path);
    const core_stem = if (std.mem.endsWith(u8, manifest_filename, ".cwasm.json"))
        manifest_filename[0 .. manifest_filename.len - ".cwasm.json".len]
    else
        manifest_filename;
    cwd.createDirPath(io, parent_dir_path) catch return error.OpenDirFailed;
    var dir = cwd.openDir(io, parent_dir_path, .{}) catch return error.OpenDirFailed;
    defer dir.close(io);

    var entries: std.ArrayList(ManifestModuleEntry) = .empty;
    entries.ensureTotalCapacity(ra, core_data_list.items.len) catch return error.OutOfMemory;

    for (core_data_list.items, 0..) |core_ref, idx| {
        // #761 Phase 2: per-core codegen cache. Build the per-core
        // cache path once; load+validate existing cache for reuse, and
        // capture the freshly-built cache to persist back. Header
        // mismatches are NOT errors — they just yield a warning + a
        // full recompile for that core, matching the single-module
        // `wamrc compile --cache` behaviour.
        var per_core_cache_path: ?[]u8 = null;
        defer if (per_core_cache_path) |p| allocator.free(p);
        var loaded_cache: ?codegen_cache.Cache = null;
        defer if (loaded_cache) |*c| c.deinit(allocator);
        var produced_cache: ?codegen_cache.Cache = null;
        defer if (produced_cache) |*c| c.deinit(allocator);

        const cache_ctx: CompileCacheCtx = blk: {
            const cd = opts.cache_dir orelse break :blk .{};
            const p = std.fmt.allocPrint(allocator, "{s}/core{d}.cache", .{ cd, idx }) catch
                return error.OutOfMemory;
            per_core_cache_path = p;
            cwd.createDirPath(io, cd) catch {};
            // Best-effort load — any error degrades to full recompile
            // for this core, with the new cache written below.
            const bytes_or = cwd.readFileAlloc(io, p, allocator, @enumFromInt(codegen_cache.max_cache_file_bytes));
            if (bytes_or) |bytes| {
                defer allocator.free(bytes);
                if (codegen_cache.deserialize(bytes, allocator)) |c| {
                    loaded_cache = c;
                } else |_| {}
            } else |_| {}
            const reuse_ptr: ?*const codegen_cache.Cache = if (loaded_cache) |*c| c else null;
            break :blk .{ .reuse = reuse_ptr, .produced = &produced_cache };
        };

        // Override opts.module_idx per core so `:mod=N` bisect filters resolve correctly.
        var per_core_opts = opts;
        per_core_opts.module_idx = @intCast(idx);

        const cwasm = compileCoreWasmCached(allocator, core_ref.data, per_core_opts, cache_ctx) catch |err| {
            std.log.err("precompileComponent: core {d} compile failed: {s}", .{ idx, @errorName(err) });
            return err;
        };
        defer allocator.free(cwasm);

        const rel_path = std.fmt.allocPrint(ra, "{s}.{d}.cwasm", .{ core_stem, idx }) catch return error.OutOfMemory;

        dir.writeFile(io, .{ .sub_path = rel_path, .data = cwasm }) catch |err| {
            std.log.err("precompileComponent: write {s}/{s} failed: {s}", .{ parent_dir_path, rel_path, @errorName(err) });
            return error.WriteFailed;
        };

        // Persist the freshly-built cache (if any) AFTER the cwasm is
        // on disk so a partial failure can't leave a cache that
        // outlives its artifact.
        if (per_core_cache_path) |p| if (produced_cache) |*pc| {
            const bytes = codegen_cache.serialize(pc, allocator) catch |err| blk: {
                std.log.warn("precompileComponent: core {d} cache serialise failed: {s}", .{ idx, @errorName(err) });
                break :blk null;
            };
            if (bytes) |b| {
                defer allocator.free(b);
                cwd.writeFile(io, .{ .sub_path = p, .data = b }) catch |err| {
                    std.log.warn("precompileComponent: core {d} cache write to {s} failed: {s}", .{ idx, p, @errorName(err) });
                };
            }
        };

        const hex = hexSha256(ra, cwasm) catch return error.OutOfMemory;
        const core_hex = hexSha256(ra, core_ref.data) catch return error.OutOfMemory;
        entries.append(ra, .{
            .idx = core_ref.local_idx,
            .path = rel_path,
            .sha256 = hex,
            .core_sha256 = core_hex,
        }) catch return error.OutOfMemory;
    }

    const component_hex = hexSha256(ra, component_bytes) catch return error.OutOfMemory;
    const build_id = ra.dupe(u8, config.version) catch return error.OutOfMemory;

    const manifest = Manifest{
        .version = manifest_format_version,
        .wamr_build_id = build_id,
        .component_sha256 = component_hex,
        .modules = entries.items,
    };

    // Serialize the manifest sidecar.
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var stringify: std.json.Stringify = .{ .writer = &aw.writer, .options = .{ .whitespace = .indent_2 } };
    stringify.write(manifest) catch return error.JsonSerializationFailed;
    dir.writeFile(io, .{ .sub_path = manifest_filename, .data = aw.written() }) catch return error.WriteFailed;

    return .{
        .manifest = manifest,
        .arena = result_arena,
        .allocator = allocator,
    };
}

/// In-memory counterpart to `precompileComponent` (#854): the same
/// core-module walk and per-core `compileCoreWasm` calls, but with no
/// filesystem I/O at all — no manifest JSON, no per-core `.cwasm`
/// files, no per-core codegen cache directory. Used by the in-process
/// JIT path so `wamr run some_component.wasm` (no sibling
/// `.cwasm.json`) can compile and instantiate in one process.
///
/// Each returned `PrecompiledCore.core_wasm` borrows directly from
/// `component_bytes` (this function's own zero-copy component parse
/// uses a throwaway scratch arena for bookkeeping only — the byte
/// slices themselves alias the caller's buffer and outlive it). A
/// caller that later re-parses the *same* `component_bytes` buffer
/// (e.g. `runComponent`'s own `component_loader.load` call during
/// instantiation) gets byte-identical (ptr+len) slices for each core
/// — the same zero-copy-parse invariant `loadManifest`'s doc comment
/// already relies on for its `core_wasm` slice-identity matching in
/// `findPrecompiled` — so this in-memory path is a drop-in swap for
/// the on-disk manifest at the `PrecompiledCore` level.
pub const InMemoryPrecompiled = struct {
    /// One owned `.cwasm` buffer per compiled core, in the same order
    /// as `pcs`. Freed by `deinit`.
    cwasm_buffers: []const []u8,
    /// Ready to pass as `Options.precompiled_cores`. Borrows from
    /// `cwasm_buffers` (bytes) and the caller's `component_bytes`
    /// (`core_wasm`); valid for the lifetime of this struct.
    pcs: []const core_backend.PrecompiledCore,
    allocator: std.mem.Allocator,

    pub fn precompiledCores(self: *const InMemoryPrecompiled) []const core_backend.PrecompiledCore {
        return self.pcs;
    }

    pub fn deinit(self: *InMemoryPrecompiled) void {
        for (self.cwasm_buffers) |buf| self.allocator.free(buf);
        self.allocator.free(self.cwasm_buffers);
        self.allocator.free(self.pcs);
    }
};

pub fn precompileComponentInMemory(
    allocator: std.mem.Allocator,
    component_bytes: []const u8,
    opts: PrecompileOptions,
) PrecompileError!InMemoryPrecompiled {
    // Same recursive walk as `precompileComponent`'s `Walker` — visits
    // every leaf core module anywhere in the (possibly composed,
    // #676) component tree.
    const CoreRef = struct {
        data: []const u8,
        local_idx: u32,
    };
    var core_data_list: std.ArrayList(CoreRef) = .empty;
    defer core_data_list.deinit(allocator);
    {
        var load_arena = std.heap.ArenaAllocator.init(allocator);
        defer load_arena.deinit();
        const component = component_loader.load(component_bytes, load_arena.allocator()) catch
            return error.InvalidComponent;
        const Walker = struct {
            fn walk(
                comp: *const ctypes.Component,
                list: *std.ArrayList(CoreRef),
                alloc: std.mem.Allocator,
            ) PrecompileError!void {
                for (comp.core_modules, 0..) |cm, mi| {
                    list.append(alloc, .{ .data = cm.data, .local_idx = @intCast(mi) }) catch
                        return error.OutOfMemory;
                }
                for (comp.components) |child| {
                    try walk(child, list, alloc);
                }
            }
        };
        try Walker.walk(&component, &core_data_list, allocator);
    }

    const n = core_data_list.items.len;
    const cwasm_buffers = allocator.alloc([]u8, n) catch return error.OutOfMemory;
    var built: usize = 0;
    errdefer {
        for (cwasm_buffers[0..built]) |b| allocator.free(b);
        allocator.free(cwasm_buffers);
    }
    const pcs = allocator.alloc(core_backend.PrecompiledCore, n) catch return error.OutOfMemory;
    errdefer allocator.free(pcs);

    for (core_data_list.items, 0..) |core_ref, idx| {
        // Override opts.module_idx per core so `:mod=N` bisect
        // filters resolve correctly, matching `precompileComponent`.
        var per_core_opts = opts;
        per_core_opts.module_idx = @intCast(idx);

        const cwasm = compileCoreWasm(allocator, core_ref.data, per_core_opts) catch |err| {
            std.log.err("precompileComponentInMemory: core {d} compile failed: {s}", .{ idx, @errorName(err) });
            return err;
        };
        cwasm_buffers[idx] = cwasm;
        built += 1;
        pcs[idx] = .{
            .module_idx = core_ref.local_idx,
            .cwasm_bytes = cwasm,
            .core_wasm = core_ref.data,
        };
    }

    return .{ .cwasm_buffers = cwasm_buffers, .pcs = pcs, .allocator = allocator };
}
