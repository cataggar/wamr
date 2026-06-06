//! wamrc — WebAssembly AOT Compiler (Zig implementation)
//!
//! Compiles .wasm files to a `.cwasm` AOT-compiled binary using the
//! Zig-native compiler backend.

const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");
const emit_aot = wamr.emit_aot;
const x86_64_compile = wamr.x86_64_compile;
const aarch64_compile = wamr.aarch64_compile;
const passes = wamr.passes;
const ir_print = wamr.ir_print;
const ir = wamr.ir;
const regalloc = wamr.regalloc;
const codegen_cache = wamr.codegen_cache;

const TargetArch = passes.TargetArch;

const verify_mod = @import("verify.zig");
// Bring the unit tests in `verify.zig` and `verify_args.zig` into the
// `wamrc_unit_tests` test runner's reachable set. Without this they
// would only execute if some non-test code path imported them — which
// is true for `verify_mod` itself, but `verify_args` is reached only
// transitively through it, so the test discovery sweep needs an
// explicit anchor.
comptime {
    _ = @import("verify_args.zig");
}

const Subcommand = enum { compile, compile_component, run, verify, version, help };

/// #392 step 3b-i diagnostic: log SSA-aware vs legacy (naive in-block phi)
/// allocator spill counts for one phi-form function. Wired into the pass
/// pipeline only when `WAMR_SSA_REGALLOC_MEASURE` is set. Errors and
/// phi-free functions are silently skipped.
fn ssaSpillMeasure(func: *const ir.IrFunction, allocator: std.mem.Allocator) void {
    const delta = (regalloc.measurePhiSpillDelta(func, allocator) catch return) orelse return;
    std.debug.print(
        "[ssa-spill-measure] fn={s} phis={d} ssa_spills={d} naive_spills={d}\n",
        .{ func.name orelse "<anon>", delta.phis, delta.ssa_spills, delta.naive_spills },
    );
}

fn parseSubcommand(s: []const u8) ?Subcommand {
    if (std.mem.eql(u8, s, "compile")) return .compile;
    if (std.mem.eql(u8, s, "compile-component")) return .compile_component;
    if (std.mem.eql(u8, s, "run")) return .run;
    if (std.mem.eql(u8, s, "verify")) return .verify;
    if (std.mem.eql(u8, s, "version")) return .version;
    if (std.mem.eql(u8, s, "help")) return .help;
    return null;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    // #761 / #743: stamp the process-global PassBisectSpec from
    // WAMR_AOT_SKIP_PASS{,ES} / WAMR_AOT_PASSES_LIMIT. Both
    // compile-paths (`runCompile`, `compileCoreWasm`) thread
    // `wamr.aot_bisect.global` into `passes.RunOptions.bisect`.
    // Use the process arena so the spec storage (tens of bytes) is
    // reclaimed cleanly at exit without tripping the GPA leak
    // detector — the spec is borrowed by every `runPassesWithOptions`
    // call for the lifetime of the wamrc invocation.
    wamr.aot_bisect.parseFromEnv(init.environ_map, init.arena.allocator());

    if (args.len < 2) {
        std.debug.print("error: missing subcommand — try `wamrc help`\n", .{});
        std.process.exit(1);
    }

    const subcmd = parseSubcommand(args[1]) orelse {
        std.debug.print("error: unknown subcommand '{s}' — try `wamrc help`\n", .{args[1]});
        std.process.exit(1);
    };

    switch (subcmd) {
        .version => try runVersion(init.io, args[2..]),
        .help => runHelp(init.io, args[2..]),
        .compile => try runCompile(init, allocator, args[2..]),
        .compile_component => try runCompileComponent(init, allocator, args[2..]),
        .run => try runRun(init, allocator, args[2..]),
        .verify => try runVerify(init, allocator, args[2..]),
    }
}

fn runVersion(io: std.Io, args: []const []const u8) !void {
    if (args.len == 1 and std.mem.eql(u8, args[0], "help")) {
        writeStdout(io, version_usage);
        return;
    }
    writeStdout(io, "wamrc " ++ wamr.version.string ++ "\n");
    writeStdout(io, "optimize " ++ @tagName(wamr.version.mode) ++ "\n");
}

fn runCompile(init: std.process.Init, allocator: std.mem.Allocator, sub_args: []const []const u8) !void {
    if (sub_args.len == 1 and std.mem.eql(u8, sub_args[0], "help")) {
        writeStdout(init.io, compile_usage);
        return;
    }
    var input_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var optimize = true;
    var enable_aarch64_scheduler = true;
    var enable_aarch64_xreg_alloc = true;
    var target_arch: TargetArch = switch (builtin.cpu.arch) {
        .aarch64 => .aarch64,
        else => .x86_64,
    };
    // #761 Phase 2: optional per-function codegen cache (sidecar file).
    // When set, wamrc reads <cache_path> at start (if it exists) and
    // reuses cached per-function code for IR hashes that still match;
    // writes the (possibly-updated) cache back to <cache_path> at end.
    var cache_path: ?[]const u8 = null;

    // --dump-ir-after / --dump-ir-functions / --dump-ir-out collection.
    // `dump_pass_names` and `dump_func_globs` are owned by the arena
    // below; their entries are borrowed slices of the input argv (and
    // thus live for the whole subcommand). When no globs are provided
    // we default to `*` (match every function).
    var dump_arena = std.heap.ArenaAllocator.init(allocator);
    defer dump_arena.deinit();
    const dump_alloc = dump_arena.allocator();
    var dump_pass_names: std.ArrayList([]const u8) = .empty;
    var dump_func_globs: std.ArrayList([]const u8) = .empty;
    var dump_out_dir: ?[]const u8 = null;

    // Default to `.after_each_pass` in safe builds (Debug / ReleaseSafe)
    // and `.off` in release builds, matching the cost-vs-diagnostic
    // tradeoff documented in #624. The user can override with
    // `WAMR_AOT_VERIFY_IR`, `--verify-ir[=…]`, or `--no-verify-ir`.
    var verify_mode = verifyModeFromEnvOrDefault(init.environ_map);

    var i: usize = 0;
    while (i < sub_args.len) : (i += 1) {
        const a = sub_args[i];
        if (std.mem.eql(u8, a, "-o") and i + 1 < sub_args.len) {
            i += 1;
            output_path = sub_args[i];
        } else if (std.mem.eql(u8, a, "-O0")) {
            optimize = false;
        } else if (std.mem.eql(u8, a, "--target") and i + 1 < sub_args.len) {
            i += 1;
            if (std.mem.eql(u8, sub_args[i], "aarch64")) {
                target_arch = .aarch64;
            } else if (std.mem.eql(u8, sub_args[i], "x86_64") or std.mem.eql(u8, sub_args[i], "x86-64")) {
                target_arch = .x86_64;
            } else {
                std.debug.print("error: unknown target '{s}' (supported: x86_64, aarch64)\n", .{sub_args[i]});
                std.process.exit(1);
            }
        } else if (std.mem.startsWith(u8, a, "--target=")) {
            const t = a["--target=".len..];
            if (std.mem.eql(u8, t, "aarch64")) {
                target_arch = .aarch64;
            } else if (std.mem.eql(u8, t, "x86_64") or std.mem.eql(u8, t, "x86-64")) {
                target_arch = .x86_64;
            } else {
                std.debug.print("error: unknown target '{s}' (supported: x86_64, aarch64)\n", .{t});
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, a, "--aarch64-no-scheduler")) {
            enable_aarch64_scheduler = false;
        } else if (std.mem.eql(u8, a, "--aarch64-no-xreg-alloc")) {
            enable_aarch64_xreg_alloc = false;
        } else if (std.mem.startsWith(u8, a, "--dump-ir-after=")) {
            try dump_pass_names.append(dump_alloc, a["--dump-ir-after=".len..]);
        } else if (std.mem.startsWith(u8, a, "--dump-ir-functions=")) {
            try dump_func_globs.append(dump_alloc, a["--dump-ir-functions=".len..]);
        } else if (std.mem.startsWith(u8, a, "--dump-ir-out=")) {
            dump_out_dir = a["--dump-ir-out=".len..];
        } else if (std.mem.eql(u8, a, "--verify-ir")) {
            verify_mode = .after_each_pass;
        } else if (std.mem.startsWith(u8, a, "--verify-ir=")) {
            verify_mode = parseVerifyModeOrDie("--verify-ir", a["--verify-ir=".len..]);
        } else if (std.mem.eql(u8, a, "--no-verify-ir")) {
            verify_mode = .off;
        } else if (std.mem.eql(u8, a, "--cache") and i + 1 < sub_args.len) {
            i += 1;
            cache_path = sub_args[i];
        } else if (std.mem.startsWith(u8, a, "--cache=")) {
            cache_path = a["--cache=".len..];
        } else if (a.len > 0 and a[0] == '-') {
            std.debug.print("error: unknown option '{s}' — try `wamrc compile help`\n", .{a});
            std.process.exit(1);
        } else if (input_path == null) {
            input_path = a;
        } else {
            std.debug.print("error: unexpected positional argument '{s}'\n", .{a});
            std.process.exit(1);
        }
    }

    if (dump_func_globs.items.len == 0) try dump_func_globs.append(dump_alloc, "*");

    const in_path = input_path orelse {
        std.debug.print("error: missing input wasm file — usage: wamrc compile <input.wasm> [-o <output.cwasm>]\n", .{});
        std.process.exit(1);
    };
    var derived_out_path: ?[]u8 = null;
    defer if (derived_out_path) |p| allocator.free(p);
    const out_path: []const u8 = if (output_path) |p| p else blk: {
        const d = try deriveOutputPath(allocator, in_path);
        derived_out_path = d;
        break :blk d;
    };

    // 1. Read input wasm
    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const wasm_data = cwd.readFileAlloc(io, in_path, allocator, @enumFromInt(64 * 1024 * 1024)) catch |err| {
        wamr.utils.read_file.dieReadFileError(in_path, err);
    };
    defer allocator.free(wasm_data);

    std.debug.print("Loaded {s} ({d} bytes)\n", .{ in_path, wasm_data.len });

    // 2. Parse wasm module.
    //
    // The loader allocates many slices on `module` (types, rec_groups,
    // canonical_type_map, imports, functions, exports, data_segments,
    // …) but `WasmModule` has no deinit and `runCompile` doesn't
    // otherwise tear it down before returning to `main`. Scope all
    // loader allocations to an arena so they're freed in one shot when
    // this function returns, mirroring every other call site of
    // `loader.load` in the repo (api/c_api.zig, api/wamr.zig,
    // wast_runner, fuzz harnesses, runtime/interpreter/instance.zig,
    // component/instance.zig). Eliminates the DebugAllocator leak
    // reports that `wamrc` emits at exit when invoked from build steps.
    var module_arena = std.heap.ArenaAllocator.init(allocator);
    defer module_arena.deinit();
    const module = wamr.loader.load(wasm_data, module_arena.allocator()) catch |err| {
        std.debug.print("Error parsing wasm: {}\n", .{err});
        std.process.exit(1);
    };

    std.debug.print("Parsed: {d} types, {d} functions, {d} exports\n", .{
        module.types.len, module.functions.len, module.exports.len,
    });

    // 3. Lower to IR
    var ir_module = wamr.frontend.lowerModule(&module, allocator) catch |err| {
        std.debug.print("Error lowering to IR: {}\n", .{err});
        std.process.exit(1);
    };
    defer ir_module.deinit();

    std.debug.print("Lowered {d} functions to IR\n", .{ir_module.functions.items.len});

    // Build a func-index → exported-name lookup. Wasm name custom
    // sections aren't currently parsed by the loader, so fall back to
    // the first export pointing at each local function. Without this,
    // `--dump-ir-functions=core_*` couldn't match anything for binaries
    // built without a names section (CoreMark, hand-written .wat).
    const ir_func_count = ir_module.functions.items.len;
    const export_names = try allocator.alloc(?[]const u8, ir_func_count);
    defer allocator.free(export_names);
    for (export_names) |*slot| slot.* = null;
    for (module.exports) |exp| {
        if (exp.kind != .function) continue;
        // Wasm function indices are imports-first, then locals.
        if (exp.index < module.imports.len) continue;
        const local_idx = exp.index - ir_module.import_count;
        if (local_idx >= ir_func_count) continue;
        if (export_names[local_idx] == null) export_names[local_idx] = exp.name;
    }

    // Set up the IR-dump hook (no-op when no `--dump-ir-after` flags
    // were passed). The Dumper borrows the parsed `dump_pass_names` /
    // `dump_func_globs` / `dump_out_dir` slices for the lifetime of
    // `runCompile`. Synthetic pass names `"initial"` (post-frontend,
    // pre-opt) and `"final"` (post-opt) are emitted directly here; all
    // other names are matched against pass invocations inside
    // `runPassesWithOptions`.
    var dumper = Dumper{
        .allocator = allocator,
        .io = io,
        .pass_names = dump_pass_names.items,
        .func_globs = dump_func_globs.items,
        .out_dir = dump_out_dir,
        .export_names = export_names,
    };
    if (dumper.pass_names.len > 0 and dumper.out_dir != null) {
        std.Io.Dir.cwd().createDirPath(io, dumper.out_dir.?) catch |err| {
            std.debug.print("error: failed to create --dump-ir-out dir '{s}': {}\n", .{ dumper.out_dir.?, err });
            std.process.exit(1);
        };
    }
    if (dumper.matchesPass("initial")) {
        for (ir_module.functions.items, 0..) |*f, fi| {
            if (!dumper.matchesFunc(f, @intCast(fi))) continue;
            try dumper.write(.{
                .pass_name = "initial",
                .func = f,
                .func_index = @intCast(fi),
                .changed = true,
                .iter = 0,
                .outer_iter = 0,
            });
        }
    }

    // 4. Optimize IR (unless -O0)
    if (optimize) {
        // #392 step 3b-i: when WAMR_SSA_REGALLOC_MEASURE is set, log, per phi
        // function, the spill count of SSA-aware allocation vs the legacy
        // (naive in-block phi) interval model. Diagnostic only — no codegen
        // change.
        const ssa_spill_measure: ?*const fn (*const ir.IrFunction, std.mem.Allocator) void =
            if (init.environ_map.get("WAMR_SSA_REGALLOC_MEASURE") != null) &ssaSpillMeasure else null;
        const run_opts: passes.RunOptions = .{
            .dump_hook = if (dumper.pass_names.len == 0) null else .{
                .ctx = @ptrCast(&dumper),
                .callback = Dumper.callback,
            },
            .verify_mode = verify_mode,
            .bisect = wamr.aot_bisect.global,
            .pass_timing = passes.passTimingOptionsFromEnv(init.environ_map),
            .analysis_timing = passes.analysisTimingOptionsFromEnv(init.environ_map),
            .tail_duplication = passes.tailDuplicationOptionsFromEnv(init.environ_map),
            .phi_spill_measure = ssa_spill_measure,
        };
        const opt_changes = passes.runPassesWithOptions(&ir_module, passes.defaultPassesForTarget(target_arch), allocator, run_opts) catch |err| {
            // If the IR verifier tripped, surface its diagnostic before
            // the generic "Error optimizing IR" line so the user sees the
            // pass name + block/inst/vreg coordinates.
            switch (err) {
                error.UnboundVRegUse,
                error.VRegDefinedTwice,
                error.MissingTerminator,
                error.MultipleTerminators,
                error.DanglingBlockRef,
                error.StalePredecessor,
                error.MissingPredecessor,
                => {
                    const f = wamr.ir_verifier.last_failure;
                    var buf: [256]u8 = undefined;
                    var w = std.Io.Writer.fixed(&buf);
                    f.format(&w) catch {};
                    std.debug.print("{s}\n", .{w.buffered()});
                },
                else => {},
            }
            std.debug.print("Error optimizing IR: {}\n", .{err});
            std.process.exit(1);
        };
        std.debug.print("Optimization: {d} passes made changes\n", .{opt_changes});
        if (verify_mode != .off) {
            std.debug.print("IR verifier: enabled ({s})\n", .{@tagName(verify_mode)});
        }
    }

    // #540: Route phi-resolution through register MOV instead of frame
    // round-trip on aarch64. The IR op `parallel_copy`, codegen
    // emitter (4-phase resolver), and lowering pass
    // `coalescePhiLocalsToParallelCopy` are all implemented and unit-
    // tested. The wiring is intentionally disabled here pending
    // investigation of a CoreMark hang (the converted IR appears to
    // confuse downstream regalloc on certain self-loop / multi-edge
    // patterns; reduced repros pass cleanly). x86_64 is out of scope
    // and keeps the existing frameStore/frameLoad lowering anyway.
    if (false and target_arch == .aarch64) {
        for (ir_module.functions.items) |*f| {
            _ = passes.coalescePhiLocalsToParallelCopy(f, allocator) catch |err| {
                std.debug.print("Error lowering phi parallel-copies: {}\n", .{err});
                std.process.exit(1);
            };
        }
    }

    if (dumper.matchesPass("final")) {
        for (ir_module.functions.items, 0..) |*f, fi| {
            if (!dumper.matchesFunc(f, @intCast(fi))) continue;
            try dumper.write(.{
                .pass_name = "final",
                .func = f,
                .func_index = @intCast(fi),
                .changed = true,
                .iter = 0,
                .outer_iter = 0,
            });
        }
    }

    // 5. Compile IR to native code (target-dependent).
    //
    // #761 Phase 2: when `--cache <path>` was supplied, try to load
    // the prior cache, validate every header field (magic / version /
    // build id / arch / abi / module epoch / func count), and pass it
    // to the cached codegen path. Any per-function ir_sha256 still
    // matching the rebuilt IR reuses the cached native bytes; the
    // rest fall back to full codegen. The new cache (always produced
    // by compileModuleCached) is serialised + atomically written to
    // <path> after a successful emit so the next bisect cycle can
    // reuse it.
    const target_abi = codegen_cache.TargetAbi.forHost(target_arch);
    const epoch_inputs: codegen_cache.ModuleEpochInputs = .{
        .wamr_build_id = wamr.version.string,
        .target_arch = target_arch,
        .target_abi = target_abi,
        .import_count = ir_module.import_count,
        .global_types = ir_module.global_types,
        .global_offsets = ir_module.global_offsets,
        .global_storage_size = ir_module.global_storage_size,
        .func_types = ir_module.func_types.items,
        .func_type_indices = ir_module.func_type_indices.items,
    };
    const module_epoch = codegen_cache.hashModuleEpoch(epoch_inputs);

    var loaded_cache: ?codegen_cache.Cache = null;
    defer if (loaded_cache) |*c| c.deinit(allocator);
    if (cache_path) |cp| {
        loaded_cache = loadCacheCompat(io, allocator, cp, .{
            .wamr_build_id = wamr.version.string,
            .target_arch = target_arch,
            .target_abi = target_abi,
            .module_epoch = module_epoch,
            .func_count = @intCast(ir_module.functions.items.len),
        });
    }
    const reuse_ptr: ?*const codegen_cache.Cache = if (loaded_cache) |*c| c else null;

    const codegen_timing = passes.codegenTimingOptionsFromEnv(init.environ_map);
    const compiled: codegen_cache.CompileResultCached = switch (target_arch) {
        .x86_64 => x86_64_compile.compileModuleCachedWithOptions(&ir_module, reuse_ptr, allocator, .{
            .codegen_timing = codegen_timing,
        }) catch |err| {
            std.debug.print("Error compiling to x86-64: {}\n", .{err});
            std.process.exit(1);
        },
        .aarch64 => aarch64_compile.compileModuleCachedWithOptions(&ir_module, reuse_ptr, allocator, .{
            .enable_scheduler = enable_aarch64_scheduler,
            .enable_xreg_alloc = enable_aarch64_xreg_alloc,
            .codegen_timing = codegen_timing,
        }) catch |err| {
            std.debug.print("Error compiling to AArch64: {}\n", .{err});
            std.process.exit(1);
        },
    };
    defer allocator.free(compiled.code);
    defer allocator.free(compiled.offsets);
    defer {
        for (compiled.cache_functions) |*f| {
            allocator.free(f.code);
            allocator.free(f.call_patches);
        }
        allocator.free(compiled.cache_functions);
    }

    if (cache_path != null) {
        std.debug.print(
            "Codegen cache: {d} reused, {d} recompiled (of {d} functions)\n",
            .{ compiled.stats.reused, compiled.stats.recompiled, ir_module.functions.items.len },
        );
    }

    std.debug.print("Generated {d} bytes of native code\n", .{compiled.code.len});

    // Build export entries
    var exports: std.ArrayList(emit_aot.ExportEntry) = .empty;
    defer exports.deinit(allocator);
    for (module.exports) |exp| {
        try exports.append(allocator, .{
            .name = exp.name,
            .kind = @enumFromInt(@intFromEnum(exp.kind)),
            .index = exp.index,
        });
    }

    // 6. Emit AOT binary
    var arch_name = std.mem.zeroes([16]u8);
    switch (target_arch) {
        .x86_64 => @memcpy(arch_name[0..6], "x86-64"),
        .aarch64 => @memcpy(arch_name[0..7], "aarch64"),
    }

    // Build data segment entries from the parsed wasm module
    var data_segs: std.ArrayList(emit_aot.DataSegmentEntry) = .empty;
    defer data_segs.deinit(allocator);
    for (module.data_segments) |seg| {
        if (seg.is_passive) continue;
        const offset: u32 = switch (seg.offset) {
            .i32_const => |v| @bitCast(v),
            else => continue,
        };
        try data_segs.append(allocator, .{
            .memory_idx = seg.memory_idx,
            .offset = offset,
            .data = seg.data,
        });
    }

    // Build import entries from the parsed wasm module
    var import_entries: std.ArrayList(emit_aot.ImportEntry) = .empty;
    defer import_entries.deinit(allocator);
    for (module.imports) |imp| {
        switch (imp.kind) {
            .function => try import_entries.append(allocator, .{
                .module_name = imp.module_name,
                .field_name = imp.field_name,
                .kind = .function,
                .func_type_idx = imp.func_type_idx orelse 0,
            }),
            .table => {
                const table_type = imp.table_type orelse continue;
                try import_entries.append(allocator, .{
                    .module_name = imp.module_name,
                    .field_name = imp.field_name,
                    .kind = .table,
                    .table_elem_type = table_type.elem_type,
                    .table_min = @intCast(table_type.limits.min),
                    .table_max = if (table_type.limits.max) |m| @as(?u32, @intCast(m)) else null,
                });
            },
            .memory, .global, .tag => {},
        }
    }

    // Locally-declared tags (#672). Mirrors `module.tag_types` 1:1.
    var tag_entries: std.ArrayList(emit_aot.TagEntry) = .empty;
    defer tag_entries.deinit(allocator);
    for (module.tag_types) |type_idx| {
        try tag_entries.append(allocator, .{ .type_idx = type_idx });
    }

    // Surface tag imports too. The outer `for (module.imports)` loop above
    // intentionally still drops `.memory` / `.global` because the standalone
    // `wamrc` path doesn't carry imported memory/global descriptors; tags
    // are different because the component-mode path (`src/component/aot.zig`)
    // and this path both need the loader to reconstruct `imported_tags` for
    // cross-instance wiring.
    for (module.imports) |imp| {
        if (imp.kind != .tag) continue;
        const type_idx = imp.tag_type_idx orelse continue;
        try import_entries.append(allocator, .{
            .module_name = imp.module_name,
            .field_name = imp.field_name,
            .kind = .tag,
            .tag_type_idx = type_idx,
        });
    }

    // Build memory entries from the parsed wasm module
    var mem_entries: std.ArrayList(emit_aot.MemoryEntry) = .empty;
    defer mem_entries.deinit(allocator);
    for (module.memories) |mem| {
        try mem_entries.append(allocator, .{
            .min_pages = @intCast(mem.limits.min),
            .max_pages = if (mem.limits.max) |m| @as(?u32, @intCast(m)) else null,
        });
    }

    // Locally-defined tables (#681). Imported tables already round-trip
    // through `import_entries`; without this section the loader can't
    // recover `module.tables`, leaving exported local tables
    // (e.g. wit-component's `(table $imports)`) unallocated at runtime.
    var table_entries: std.ArrayList(emit_aot.TableEntry) = .empty;
    defer table_entries.deinit(allocator);
    for (module.tables) |t| {
        try table_entries.append(allocator, .{
            .elem_type = t.elem_type,
            .min = @intCast(t.limits.min),
            .max = if (t.limits.max) |m| @as(?u32, @intCast(m)) else null,
        });
    }

    // Build global entries in wasm-flat order (imported globals first, then
    // local globals) so codegen offsets match runtime storage.
    var global_entries: std.ArrayList(emit_aot.GlobalEntry) = .empty;
    defer global_entries.deinit(allocator);
    var tmp_globals: std.ArrayList(*wamr.types.GlobalInstance) = .empty;
    defer {
        for (tmp_globals.items) |g| allocator.destroy(g);
        tmp_globals.deinit(allocator);
    }
    for (module.imports) |imp| {
        if (imp.kind != .global) continue;
        const gt = imp.global_type orelse continue;
        const val = defaultZeroValue(gt.val_type);
        const gi = try allocator.create(wamr.types.GlobalInstance);
        gi.* = .{ .global_type = gt, .value = val };
        try tmp_globals.append(allocator, gi);
        try global_entries.append(allocator, .{
            .val_type = @intFromEnum(gt.val_type),
            .mutability = if (gt.mutability == .mutable) @as(u8, 1) else @as(u8, 0),
            .init_i64 = valueToI64(val),
            .init_v128 = valueToV128(val),
        });
    }
    for (module.globals) |g| {
        const val = wamr.instance.evalInitExpr(g.init_expr, tmp_globals.items, null) catch defaultZeroValue(g.global_type.val_type);
        const gi = try allocator.create(wamr.types.GlobalInstance);
        gi.* = .{ .global_type = g.global_type, .value = val };
        try tmp_globals.append(allocator, gi);
        try global_entries.append(allocator, .{
            .val_type = @intFromEnum(g.global_type.val_type),
            .mutability = if (g.global_type.mutability == .mutable) @as(u8, 1) else @as(u8, 0),
            .init_i64 = valueToI64(val),
            .init_v128 = valueToV128(val),
        });
    }

    // Build element segment entries. Each entry owns a freshly-allocated
    // `func_indices` slice (borrowed by `emit_aot.emit` for the duration of
    // this function), so the defer must free those per-entry slices too —
    // mirroring `func_type_entries` below. Freeing only the list backing
    // leaked one block per active element segment (#789).
    var elem_entries: std.ArrayList(emit_aot.ElemEntry) = .empty;
    defer {
        for (elem_entries.items) |ee| {
            if (ee.func_indices.len > 0) allocator.free(ee.func_indices);
        }
        elem_entries.deinit(allocator);
    }
    for (module.elements) |seg| {
        if (seg.is_declarative) continue;
        const offset: u32 = if (seg.is_passive) 0 else blk: {
            const off = seg.offset orelse continue;
            break :blk switch (off) {
                .i32_const => |v| @as(u32, @bitCast(v)),
                else => continue,
            };
        };
        // Extract function indices from the segment. Use 0xFFFFFFFF as a
        // null sentinel (wasm funcidx 0 is a valid function). The runtime
        // writes 0 into the native backing for null entries.
        const indices = try allocator.alloc(u32, seg.func_indices.len);
        for (seg.func_indices, 0..) |fi, j| {
            indices[j] = fi orelse 0xFFFFFFFF;
        }
        try elem_entries.append(allocator, .{
            .table_idx = seg.table_idx,
            .offset = offset,
            .func_indices = indices,
            .is_passive = seg.is_passive,
        });
    }

    // Build func-type entries from the parsed wasm module. One entry per
    // module.types index; non-func (struct/array) kinds serialize as empty
    // params/results placeholders so that all `type_idx` references remain
    // valid. Slices are allocator-owned for the lifetime of this function.
    var func_type_entries: std.ArrayList(emit_aot.FuncTypeEntry) = .empty;
    defer {
        for (func_type_entries.items) |fte| {
            if (fte.params.len > 0) allocator.free(fte.params);
            if (fte.results.len > 0) allocator.free(fte.results);
        }
        func_type_entries.deinit(allocator);
    }
    for (module.types) |ft| {
        if (ft.kind != .func) {
            try func_type_entries.append(allocator, .{ .params = &.{}, .results = &.{} });
            continue;
        }
        const params_bytes = try allocator.alloc(u8, ft.params.len);
        for (ft.params, 0..) |p, j| params_bytes[j] = @intFromEnum(p);
        const results_bytes = try allocator.alloc(u8, ft.results.len);
        for (ft.results, 0..) |r, j| results_bytes[j] = @intFromEnum(r);
        try func_type_entries.append(allocator, .{ .params = params_bytes, .results = results_bytes });
    }

    // Build local function → type_idx map (one entry per local function,
    // in the order they were compiled).
    var local_func_tidx_list: std.ArrayList(u32) = .empty;
    defer local_func_tidx_list.deinit(allocator);
    for (module.functions) |f| {
        try local_func_tidx_list.append(allocator, f.type_idx);
    }

    // Parse the wasm `name` custom section directly from the source
    // bytes. The interpreter loader skips custom sections, so this is
    // a separate pass over the same buffer. A malformed name section is
    // non-fatal (diagnostic-only): fall back to no names and let the
    // trap helpers print `local_func[N]` without a symbol. Allocates
    // into the same arena as the other emit-side scratch buffers.
    const fn_name_entries: ?[]emit_aot.FunctionNameEntry = blk: {
        const parsed = wamr.name_section.parseFunctionNames(wasm_data, allocator) catch break :blk null;
        defer allocator.free(parsed);
        if (parsed.len == 0) break :blk null;
        const entries = allocator.alloc(emit_aot.FunctionNameEntry, parsed.len) catch break :blk null;
        for (parsed, 0..) |p, idx| {
            entries[idx] = .{ .index = p.index, .name = p.name };
        }
        break :blk entries;
    };
    defer if (fn_name_entries) |e| allocator.free(e);

    const aot_binary = try emit_aot.emit(
        allocator,
        compiled.code,
        compiled.offsets,
        exports.items,
        .{ .arch = arch_name },
        if (data_segs.items.len > 0) data_segs.items else null,
        if (import_entries.items.len > 0) import_entries.items else null,
        if (mem_entries.items.len > 0) mem_entries.items else null,
        if (global_entries.items.len > 0) global_entries.items else null,
        if (elem_entries.items.len > 0) elem_entries.items else null,
        module.start_function,
        if (func_type_entries.items.len > 0) func_type_entries.items else null,
        if (local_func_tidx_list.items.len > 0) local_func_tidx_list.items else null,
        if (tag_entries.items.len > 0) tag_entries.items else null,
        if (table_entries.items.len > 0) table_entries.items else null,
        fn_name_entries,
    );
    defer allocator.free(aot_binary);

    // 7. Write output
    const out_file = cwd.createFile(io, out_path, .{}) catch |err| {
        std.debug.print("Error creating {s}: {}\n", .{ out_path, err });
        std.process.exit(1);
    };
    defer out_file.close(io);
    out_file.writeStreamingAll(io, aot_binary) catch |err| {
        std.debug.print("Error writing {s}: {}\n", .{ out_path, err });
        std.process.exit(1);
    };

    std.debug.print("Written {s} ({d} bytes)\n", .{ out_path, aot_binary.len });

    // 8. Persist the codegen cache (Phase 2) if --cache was passed.
    //    Write only after the cwasm itself is on disk so a partial
    //    failure can't leave a cache pointing at an absent / older
    //    artifact.
    if (cache_path) |cp| {
        const cache_to_write: codegen_cache.Cache = .{
            .wamr_build_id = wamr.version.string,
            .target_arch = target_arch,
            .target_abi = target_abi,
            .module_epoch = module_epoch,
            .functions = compiled.cache_functions,
        };
        const cache_bytes = codegen_cache.serialize(&cache_to_write, allocator) catch |err| {
            std.debug.print("warning: codegen cache: serialise failed: {} — cache not updated\n", .{err});
            return;
        };
        defer allocator.free(cache_bytes);
        writeFileAtomic(io, cp, cache_bytes) catch |err| {
            std.debug.print("warning: codegen cache: write to {s} failed: {} — cache not updated\n", .{ cp, err });
            return;
        };
        std.debug.print("Cache written {s} ({d} bytes)\n", .{ cp, cache_bytes.len });
    }
}

/// Inputs for `loadCacheCompat`'s header-compatibility check.
const CacheLoadCheck = struct {
    wamr_build_id: []const u8,
    target_arch: TargetArch,
    target_abi: codegen_cache.TargetAbi,
    module_epoch: [32]u8,
    func_count: u32,
};

/// Try to read + deserialise + header-validate a codegen-cache sidecar.
/// Returns null on any mismatch (with a one-line warning explaining
/// which header field differed) so a fresh-but-incompatible cache
/// degrades cleanly to "full recompile" instead of failing the build.
fn loadCacheCompat(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    expect: CacheLoadCheck,
) ?codegen_cache.Cache {
    const cwd = std.Io.Dir.cwd();
    const bytes = cwd.readFileAlloc(io, path, allocator, @enumFromInt(codegen_cache.max_cache_file_bytes)) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("Codegen cache: no existing cache at {s} — full recompile\n", .{path});
            return null;
        },
        else => {
            std.debug.print("warning: codegen cache: read {s} failed: {} — full recompile\n", .{ path, err });
            return null;
        },
    };
    defer allocator.free(bytes);

    var cache = codegen_cache.deserialize(bytes, allocator) catch |err| {
        std.debug.print("warning: codegen cache: {s} unreadable ({}) — full recompile\n", .{ path, err });
        return null;
    };
    errdefer cache.deinit(allocator);

    if (!std.mem.eql(u8, cache.wamr_build_id, expect.wamr_build_id)) {
        std.debug.print("Codegen cache: build-id mismatch (cached={s}, current={s}) — full recompile\n", .{ cache.wamr_build_id, expect.wamr_build_id });
        cache.deinit(allocator);
        return null;
    }
    if (cache.target_arch != expect.target_arch) {
        std.debug.print("Codegen cache: target-arch mismatch — full recompile\n", .{});
        cache.deinit(allocator);
        return null;
    }
    if (cache.target_abi != expect.target_abi) {
        std.debug.print("Codegen cache: target-abi mismatch — full recompile\n", .{});
        cache.deinit(allocator);
        return null;
    }
    if (!std.mem.eql(u8, &cache.module_epoch, &expect.module_epoch)) {
        std.debug.print("Codegen cache: module-epoch mismatch (imports/globals/types changed) — full recompile\n", .{});
        cache.deinit(allocator);
        return null;
    }
    if (cache.functions.len != expect.func_count) {
        std.debug.print("Codegen cache: func-count mismatch (cached={d}, current={d}) — full recompile\n", .{ cache.functions.len, expect.func_count });
        cache.deinit(allocator);
        return null;
    }
    return cache;
}

fn defaultZeroValue(vt: wamr.types.ValType) wamr.types.Value {
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

fn valueToI64(v: wamr.types.Value) i64 {
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

fn valueToV128(v: wamr.types.Value) u128 {
    return switch (v) {
        .v128 => |x| x,
        else => 0,
    };
}

fn writeStdout(io: std.Io, text: []const u8) void {
    var stdout_file = std.Io.File.stdout();
    stdout_file.writeStreamingAll(io, text) catch {};
}

fn runCompileComponent(init: std.process.Init, allocator: std.mem.Allocator, sub_args: []const []const u8) !void {
    if (sub_args.len == 1 and std.mem.eql(u8, sub_args[0], "help")) {
        writeStdout(init.io, compile_component_usage);
        return;
    }
    var input_path: ?[]const u8 = null;
    var output_manifest: ?[]const u8 = null;
    var optimize: bool = true;
    var target_arch: TargetArch = switch (builtin.cpu.arch) {
        .aarch64 => .aarch64,
        else => .x86_64,
    };
    // #761 Phase 2: per-core codegen cache directory.
    var cache_dir: ?[]const u8 = null;
    var verify_mode = verifyModeFromEnvOrDefault(init.environ_map);

    var i: usize = 0;
    while (i < sub_args.len) : (i += 1) {
        const a = sub_args[i];
        if (std.mem.eql(u8, a, "-o")) {
            i += 1;
            if (i >= sub_args.len) {
                std.debug.print("error: -o requires an argument\n", .{});
                std.process.exit(1);
            }
            output_manifest = sub_args[i];
        } else if (std.mem.eql(u8, a, "-O0")) {
            optimize = false;
        } else if (std.mem.eql(u8, a, "--verify-ir")) {
            verify_mode = .after_each_pass;
        } else if (std.mem.startsWith(u8, a, "--verify-ir=")) {
            verify_mode = parseVerifyModeOrDie("--verify-ir", a["--verify-ir=".len..]);
        } else if (std.mem.eql(u8, a, "--no-verify-ir")) {
            verify_mode = .off;
        } else if (std.mem.startsWith(u8, a, "--target=")) {
            const v = a["--target=".len..];
            if (std.mem.eql(u8, v, "x86_64") or std.mem.eql(u8, v, "x86-64")) {
                target_arch = .x86_64;
            } else if (std.mem.eql(u8, v, "aarch64")) {
                target_arch = .aarch64;
            } else {
                std.debug.print("error: unknown target '{s}'\n", .{v});
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, a, "--cache-dir") and i + 1 < sub_args.len) {
            i += 1;
            cache_dir = sub_args[i];
        } else if (std.mem.startsWith(u8, a, "--cache-dir=")) {
            cache_dir = a["--cache-dir=".len..];
        } else if (std.mem.startsWith(u8, a, "-")) {
            std.debug.print("error: unknown option '{s}'\n", .{a});
            std.process.exit(1);
        } else if (input_path == null) {
            input_path = a;
        } else {
            std.debug.print("error: unexpected argument '{s}'\n", .{a});
            std.process.exit(1);
        }
    }

    const in_path = input_path orelse {
        std.debug.print("error: missing input component path\n", .{});
        std.process.exit(1);
    };
    const manifest_path: []const u8 = if (output_manifest) |o| o else blk: {
        break :blk try wamr.component_aot.defaultManifestPathFor(allocator, in_path);
    };

    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const component_data = cwd.readFileAlloc(io, in_path, allocator, @enumFromInt(256 * 1024 * 1024)) catch |err| {
        wamr.utils.read_file.dieReadFileError(in_path, err);
    };
    defer allocator.free(component_data);

    std.debug.print("Loaded component {s} ({d} bytes)\n", .{ in_path, component_data.len });

    const is_comp = wamr.component_aot.isComponent(component_data) catch {
        std.debug.print("error: input is not a valid wasm module or component\n", .{});
        std.process.exit(1);
    };
    if (!is_comp) {
        std.debug.print("error: input is a core wasm module, not a component — use `wamrc compile`\n", .{});
        std.process.exit(1);
    }

    var result = wamr.component_aot_compile.precompileComponent(allocator, component_data, manifest_path, .{
        .target_arch = target_arch,
        .optimize = optimize,
        .cache_dir = cache_dir,
        .pass_timing = passes.passTimingOptionsFromEnv(init.environ_map),
        .analysis_timing = passes.analysisTimingOptionsFromEnv(init.environ_map),
        .codegen_timing = passes.codegenTimingOptionsFromEnv(init.environ_map),
        .tail_duplication = passes.tailDuplicationOptionsFromEnv(init.environ_map),
        .verify_mode = verify_mode,
    }) catch |err| {
        std.debug.print("error: precompile failed: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
    defer result.deinit();

    std.debug.print("Wrote {d} core module(s) + {s}\n", .{
        result.manifest.modules.len, manifest_path,
    });
}

/// `wamrc run` — compile (if needed) and then execute via the `wamr`
/// runtime as a subprocess. Default output lives next to the input:
/// `foo.wasm` → `foo.cwasm` (core) or `foo.cwasm.json` + `foo.<N>.cwasm`
/// (component). The existing artifact is reused when a "target format"
/// check passes (see `coreArtifactFresh` / component manifest
/// verification); otherwise it is recompiled in place.
fn runRun(init: std.process.Init, allocator: std.mem.Allocator, sub_args: []const []const u8) !void {
    if (sub_args.len == 1 and std.mem.eql(u8, sub_args[0], "help")) {
        writeStdout(init.io, run_usage);
        return;
    }

    var input_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var force = false;
    var target_arch: TargetArch = switch (builtin.cpu.arch) {
        .aarch64 => .aarch64,
        else => .x86_64,
    };

    // After we encounter `--` or the first non-option positional, every
    // remaining token is forwarded verbatim to `wamr run`. We split the
    // input on `--` so users can disambiguate `wamrc run -O0 foo.wasm`
    // (still our flag) from `wamrc run foo.wasm -- --listen ...`
    // (`--listen` belongs to `wamr`).
    var forward_args: std.ArrayList([]const u8) = .empty;
    defer forward_args.deinit(allocator);

    var i: usize = 0;
    var saw_dashdash = false;
    while (i < sub_args.len) : (i += 1) {
        const a = sub_args[i];
        if (saw_dashdash) {
            try forward_args.append(allocator, a);
            continue;
        }
        if (std.mem.eql(u8, a, "--")) {
            saw_dashdash = true;
            continue;
        }
        if (std.mem.eql(u8, a, "-o") and i + 1 < sub_args.len) {
            i += 1;
            output_path = sub_args[i];
        } else if (std.mem.eql(u8, a, "--force")) {
            force = true;
        } else if (std.mem.eql(u8, a, "--target") and i + 1 < sub_args.len) {
            i += 1;
            target_arch = parseTargetArchOrDie(sub_args[i]);
        } else if (std.mem.startsWith(u8, a, "--target=")) {
            target_arch = parseTargetArchOrDie(a["--target=".len..]);
        } else if (a.len > 0 and a[0] == '-' and input_path == null) {
            std.debug.print("error: unknown option '{s}' — try `wamrc run help`\n", .{a});
            std.process.exit(1);
        } else if (input_path == null) {
            input_path = a;
        } else {
            // Implicit forward: positionals after the input wasm go to
            // `wamr run` without requiring `--`.
            try forward_args.append(allocator, a);
        }
    }

    const in_path = input_path orelse {
        std.debug.print("error: missing input wasm file — usage: wamrc run <input.wasm> [-- args...]\n", .{});
        std.process.exit(1);
    };

    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const wasm_data = cwd.readFileAlloc(io, in_path, allocator, @enumFromInt(256 * 1024 * 1024)) catch |err| {
        wamr.utils.read_file.dieReadFileError(in_path, err);
    };
    defer allocator.free(wasm_data);

    const is_comp = wamr.component_aot.isComponent(wasm_data) catch {
        std.debug.print("error: '{s}' is not a valid wasm module or component\n", .{in_path});
        std.process.exit(1);
    };

    // Resolve the artifact path. For components this is a manifest
    // sidecar (`<stem>.cwasm.json`); for core wasm it's the `.cwasm`
    // itself. Default lives next to the input.
    var derived_output: ?[]u8 = null;
    defer if (derived_output) |d| allocator.free(d);

    const artifact_path: []const u8 = if (is_comp) blk: {
        if (output_path) |p| break :blk p;
        const d = try wamr.component_aot.defaultManifestPathFor(allocator, in_path);
        derived_output = d;
        break :blk d;
    } else blk: {
        if (output_path) |p| break :blk p;
        const d = try deriveOutputPath(allocator, in_path);
        derived_output = d;
        break :blk d;
    };

    // Freshness check. On a hit we skip compilation entirely; on a miss
    // (or `--force`) we (re)compile in place.
    var did_compile = false;
    if (is_comp) {
        const fresh = !force and componentArtifactFresh(allocator, artifact_path, wasm_data);
        if (!fresh) {
            std.debug.print("wamrc: compiling {s} → {s}\n", .{ in_path, artifact_path });
            var result = wamr.component_aot_compile.precompileComponent(allocator, wasm_data, artifact_path, .{
                .target_arch = target_arch,
                .pass_timing = passes.passTimingOptionsFromEnv(init.environ_map),
                .analysis_timing = passes.analysisTimingOptionsFromEnv(init.environ_map),
                .codegen_timing = passes.codegenTimingOptionsFromEnv(init.environ_map),
                .tail_duplication = passes.tailDuplicationOptionsFromEnv(init.environ_map),
                .verify_mode = verifyModeFromEnvOrDefault(init.environ_map),
            }) catch |err| {
                std.debug.print("error: precompile failed: {s}\n", .{@errorName(err)});
                std.process.exit(1);
            };
            result.deinit();
            did_compile = true;
        }
    } else {
        const fresh = !force and coreArtifactFresh(allocator, artifact_path, wasm_data, target_arch, in_path);
        if (!fresh) {
            std.debug.print("wamrc: compiling {s} → {s}\n", .{ in_path, artifact_path });
            const cwasm = wamr.component_aot_compile.compileCoreWasm(allocator, wasm_data, .{
                .target_arch = target_arch,
                .pass_timing = passes.passTimingOptionsFromEnv(init.environ_map),
                .analysis_timing = passes.analysisTimingOptionsFromEnv(init.environ_map),
                .codegen_timing = passes.codegenTimingOptionsFromEnv(init.environ_map),
                .tail_duplication = passes.tailDuplicationOptionsFromEnv(init.environ_map),
                .verify_mode = verifyModeFromEnvOrDefault(init.environ_map),
            }) catch |err| {
                std.debug.print("error: compile failed: {s}\n", .{@errorName(err)});
                std.process.exit(1);
            };
            defer allocator.free(cwasm);
            writeFileAtomic(io, artifact_path, cwasm) catch |err| {
                std.debug.print("error: failed to write '{s}': {s}\n", .{ artifact_path, @errorName(err) });
                std.process.exit(1);
            };
            writeCoreSidecar(allocator, io, artifact_path, wasm_data, target_arch) catch |err| {
                std.debug.print("warning: failed to write fingerprint sidecar for '{s}': {s}\n", .{ artifact_path, @errorName(err) });
            };
            did_compile = true;
        }
    }
    if (!did_compile) {
        std.debug.print("wamrc: reusing {s} (up to date)\n", .{artifact_path});
    }

    // Build the argv for `wamr run`. Layout:
    //   wamr run <artifact> [forwarded args...]
    const wamr_bin = findWamrBinary(allocator, io, init.environ_map) catch |err| {
        std.debug.print("error: could not locate `wamr` binary: {s}\n" ++
            "  Set WAMR_BIN, install `wamr` on PATH, or place it next to wamrc.\n", .{@errorName(err)});
        std.process.exit(1);
    };
    defer allocator.free(wamr_bin);

    var child_argv: std.ArrayList([]const u8) = .empty;
    defer child_argv.deinit(allocator);
    try child_argv.append(allocator, wamr_bin);
    try child_argv.append(allocator, "run");
    // For components, `wamr run` takes the source `.wasm` and
    // auto-discovers the sibling `<stem>.cwasm.json` (or honours
    // `--precompiled-manifest` if the user picked a non-default
    // location). For core wasm we hand it the freshly-written
    // `.cwasm` directly.
    if (is_comp) {
        if (output_path != null) {
            try child_argv.append(allocator, "--precompiled-manifest");
            try child_argv.append(allocator, artifact_path);
        }
        try child_argv.append(allocator, in_path);
    } else {
        try child_argv.append(allocator, artifact_path);
    }
    for (forward_args.items) |fa| try child_argv.append(allocator, fa);

    var child = std.process.spawn(io, .{
        .argv = child_argv.items,
        .stdin = .inherit,
        .stdout = .inherit,
        .stderr = .inherit,
    }) catch |err| {
        std.debug.print("error: failed to spawn '{s}': {s}\n", .{ wamr_bin, @errorName(err) });
        std.process.exit(1);
    };

    const term = child.wait(io) catch |err| {
        std.debug.print("error: failed waiting for `wamr` subprocess: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };

    switch (term) {
        .exited => |code| std.process.exit(code),
        .signal => |sig| {
            std.debug.print("error: `wamr` was killed by signal {d}\n", .{@intFromEnum(sig)});
            std.process.exit(1);
        },
        .stopped => |sig| {
            std.debug.print("error: `wamr` was stopped by signal {d}\n", .{@intFromEnum(sig)});
            std.process.exit(1);
        },
        .unknown => |code| {
            std.debug.print("error: `wamr` terminated with unknown status {d}\n", .{code});
            std.process.exit(1);
        },
    }
}

fn runVerify(init: std.process.Init, allocator: std.mem.Allocator, sub_args: []const []const u8) !void {
    if (sub_args.len == 1 and std.mem.eql(u8, sub_args[0], "help")) {
        writeStdout(init.io, verify_usage);
        return;
    }

    var diag: verify_mod.ParseDiagnostic = .{};
    var parsed = verify_mod.parse(allocator, sub_args, &diag) catch |err| {
        switch (err) {
            error.MissingInputPath => std.debug.print(
                "error: missing input wasm file — try `wamrc verify help`\n",
                .{},
            ),
            error.DuplicateInputPath => std.debug.print(
                "error: unexpected extra positional '{s}' (only one input wasm allowed) — try `wamrc verify help`\n",
                .{diag.value},
            ),
            error.UnknownOption => std.debug.print(
                "error: unknown option '{s}' — try `wamrc verify help`\n",
                .{diag.option},
            ),
            error.MissingValue => std.debug.print(
                "error: option '{s}' requires a value — try `wamrc verify help`\n",
                .{diag.option},
            ),
            error.InvalidIntegerArgument => std.debug.print(
                "error: option '{s}' expects an integer, got '{s}' — try `wamrc verify help`\n",
                .{ diag.option, diag.value },
            ),
            error.OutOfMemory => std.debug.print("error: out of memory parsing args\n", .{}),
        }
        std.process.exit(2);
    };
    defer parsed.deinit();

    const exit_code = verify_mod.run(init, allocator, parsed.options) catch |err| {
        std.debug.print("error: verify failed: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    std.process.exit(exit_code);
}

fn parseTargetArchOrDie(s: []const u8) TargetArch {
    if (std.mem.eql(u8, s, "aarch64")) return .aarch64;
    if (std.mem.eql(u8, s, "x86_64") or std.mem.eql(u8, s, "x86-64")) return .x86_64;
    std.debug.print("error: unknown target '{s}' (supported: x86_64, aarch64)\n", .{s});
    std.process.exit(1);
}

fn defaultVerifyMode() wamr.ir_verifier.VerifyMode {
    return if (std.debug.runtime_safety) .after_each_pass else .off;
}

fn parseVerifyMode(value: []const u8) ?wamr.ir_verifier.VerifyMode {
    if (std.ascii.eqlIgnoreCase(value, "default")) return defaultVerifyMode();
    if (std.ascii.eqlIgnoreCase(value, "off") or
        std.mem.eql(u8, value, "0") or
        std.ascii.eqlIgnoreCase(value, "false") or
        std.ascii.eqlIgnoreCase(value, "no"))
    {
        return .off;
    }
    if (std.ascii.eqlIgnoreCase(value, "after-each-pass") or
        std.ascii.eqlIgnoreCase(value, "after_each_pass") or
        std.ascii.eqlIgnoreCase(value, "on") or
        std.mem.eql(u8, value, "1") or
        std.ascii.eqlIgnoreCase(value, "true") or
        std.ascii.eqlIgnoreCase(value, "yes"))
    {
        return .after_each_pass;
    }
    if (std.ascii.eqlIgnoreCase(value, "load-forwarding") or
        std.ascii.eqlIgnoreCase(value, "load_forwarding"))
    {
        return .load_forwarding;
    }
    if (std.ascii.eqlIgnoreCase(value, "paranoid")) return .paranoid;
    return null;
}

fn parseVerifyModeOrDie(source: []const u8, value: []const u8) wamr.ir_verifier.VerifyMode {
    return parseVerifyMode(value) orelse {
        std.debug.print(
            "error: invalid {s} value '{s}' (expected: off, after-each-pass, load-forwarding, paranoid, default)\n",
            .{ source, value },
        );
        std.process.exit(1);
    };
}

fn verifyModeFromEnvOrDefault(env: *const std.process.Environ.Map) wamr.ir_verifier.VerifyMode {
    const value = env.get("WAMR_AOT_VERIFY_IR") orelse return defaultVerifyMode();
    return parseVerifyModeOrDie("WAMR_AOT_VERIFY_IR", value);
}

fn targetArchName(arch: TargetArch) []const u8 {
    return switch (arch) {
        .aarch64 => "aarch64",
        .x86_64 => "x86_64",
    };
}

/// Fingerprint persisted next to a `<input>.cwasm` so `wamrc run` can
/// decide whether to reuse the artifact. The cwasm header alone only
/// stores `aot_magic + aot_version`; the sidecar adds the bits we
/// otherwise can't recover (target arch, wamr build id, source hash).
const CoreSidecar = struct {
    aot_version: u32,
    wamr_build_id: []const u8,
    target_arch: []const u8,
    wasm_sha256: []const u8,
};

fn coreSidecarPathAlloc(allocator: std.mem.Allocator, artifact_path: []const u8) ![]u8 {
    return std.mem.concat(allocator, u8, &.{ artifact_path, ".id" });
}

fn writeCoreSidecar(
    allocator: std.mem.Allocator,
    io: std.Io,
    artifact_path: []const u8,
    wasm_data: []const u8,
    target_arch: TargetArch,
) !void {
    var hex_hash: [64]u8 = undefined;
    hexSha256Into(wasm_data, &hex_hash);

    const sidecar: CoreSidecar = .{
        .aot_version = emit_aot.aot_version,
        .wamr_build_id = wamr.version.string,
        .target_arch = targetArchName(target_arch),
        .wasm_sha256 = &hex_hash,
    };

    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var sw: std.json.Stringify = .{ .writer = &aw.writer };
    try sw.write(sidecar);

    const sidecar_path = try coreSidecarPathAlloc(allocator, artifact_path);
    defer allocator.free(sidecar_path);
    try writeFileAtomic(io, sidecar_path, aw.written());
}

fn hexSha256Into(data: []const u8, out: *[64]u8) void {
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    const hex = "0123456789abcdef";
    for (digest, 0..) |b, j| {
        out[j * 2] = hex[b >> 4];
        out[j * 2 + 1] = hex[b & 0x0f];
    }
}

fn writeFileAtomic(io: std.Io, path: []const u8, bytes: []const u8) !void {
    // Write to `<path>.tmp` then rename for crash safety. Falls back
    // to a plain overwrite if the rename fails (e.g. cross-device).
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd = std.Io.Dir.cwd();
    if (path.len + 4 > tmp_buf.len) {
        // Path too long for the buffer — just overwrite directly.
        try cwd.writeFile(io, .{ .sub_path = path, .data = bytes });
        return;
    }
    @memcpy(tmp_buf[0..path.len], path);
    @memcpy(tmp_buf[path.len..][0..4], ".tmp");
    const tmp_path = tmp_buf[0 .. path.len + 4];
    try cwd.writeFile(io, .{ .sub_path = tmp_path, .data = bytes });
    cwd.rename(tmp_path, cwd, path, io) catch {
        // Best-effort: overwrite directly, then clean up the tmp.
        try cwd.writeFile(io, .{ .sub_path = path, .data = bytes });
        cwd.deleteFile(io, tmp_path) catch {};
    };
}

/// True iff `artifact_path` is a `.cwasm` whose magic + version match
/// the current `emit_aot.aot_version` AND whose sidecar matches the
/// current wamr build id, target arch, and source wasm hash.
fn coreArtifactFresh(
    allocator: std.mem.Allocator,
    artifact_path: []const u8,
    wasm_data: []const u8,
    target_arch: TargetArch,
    in_path: []const u8,
) bool {
    _ = in_path;
    const io = std.Io.Threaded.global_single_threaded.io();
    const cwd = std.Io.Dir.cwd();

    // 1. Read enough of the cwasm to validate the header.
    var header: [8]u8 = undefined;
    const got = cwd.readFile(io, artifact_path, &header) catch return false;
    if (got.len < 8) return false;
    const magic = std.mem.readInt(u32, header[0..4], .little);
    const version = std.mem.readInt(u32, header[4..8], .little);
    if (magic != emit_aot.aot_magic) return false;
    if (version != emit_aot.aot_version) return false;

    // 2. Read + parse the sidecar.
    const sidecar_path = coreSidecarPathAlloc(allocator, artifact_path) catch return false;
    defer allocator.free(sidecar_path);
    const json_bytes = cwd.readFileAlloc(io, sidecar_path, allocator, @enumFromInt(64 * 1024)) catch return false;
    defer allocator.free(json_bytes);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const parsed = std.json.parseFromSliceLeaky(
        CoreSidecar,
        arena.allocator(),
        json_bytes,
        .{ .ignore_unknown_fields = true },
    ) catch return false;

    if (parsed.aot_version != emit_aot.aot_version) return false;
    if (!std.mem.eql(u8, parsed.wamr_build_id, wamr.version.string)) return false;
    if (!std.mem.eql(u8, parsed.target_arch, targetArchName(target_arch))) return false;

    var hex_hash: [64]u8 = undefined;
    hexSha256Into(wasm_data, &hex_hash);
    if (!std.mem.eql(u8, parsed.wasm_sha256, &hex_hash)) return false;

    return true;
}

/// True iff `manifest_path` holds a manifest that loads cleanly
/// against the supplied component bytes. `loadManifest` already
/// validates `wamr_build_id`, `component_sha256`, and every per-core
/// `sha256`, so a successful load *is* the freshness signal.
fn componentArtifactFresh(
    allocator: std.mem.Allocator,
    manifest_path: []const u8,
    wasm_data: []const u8,
) bool {
    var loaded = wamr.component_aot.loadManifest(allocator, manifest_path, wasm_data) catch return false;
    loaded.deinit();
    return true;
}

/// Resolve a path to a sibling `wamr` binary. Resolution order:
///   1. `WAMR_BIN` environment variable (caller's responsibility to
///      point at a real file).
///   2. `<dir-of-wamrc>/wamr` — covers `zig build` (both binaries land
///      in `zig-out/bin/`) and most install layouts.
///   3. PATH lookup of `wamr` (we let `process.spawn` do this implicitly
///      by returning the bare name `"wamr"`).
fn findWamrBinary(
    allocator: std.mem.Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
) ![]u8 {
    if (environ_map.get("WAMR_BIN")) |env_path| {
        if (env_path.len > 0) return allocator.dupe(u8, env_path);
    }

    // Locate sibling next to wamrc.
    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (std.process.executablePath(io, &exe_buf)) |n| {
        const exe_path = exe_buf[0..n];
        const dir = std.fs.path.dirname(exe_path) orelse "";
        if (dir.len > 0) {
            const candidate = try std.fs.path.join(allocator, &.{ dir, "wamr" });
            // Don't access() here — let spawn surface a clearer error
            // (e.g. "FileNotFound") with the candidate path baked in.
            return candidate;
        }
    } else |_| {}

    // Final fallback: let the OS look up "wamr" on PATH.
    return allocator.dupe(u8, "wamr");
}

const top_usage =
    \\wamrc - WebAssembly AOT Compiler
    \\
    \\Usage: wamrc <subcommand> [args...]
    \\
    \\Subcommands:
    \\  compile             Compile a .wasm module to a .cwasm AOT binary
    \\  compile-component   Precompile every embedded core of a component
    \\  run                 Compile (if needed) and execute via the wamr runtime
    \\  verify              Differential-test wamr-AOT vs wasmtime on a wasm
    \\  version             Print version and exit
    \\  help                Print this help
    \\
    \\Run `wamrc <subcommand> help` to show help for a specific subcommand.
    \\
;

const compile_usage =
    \\Usage: wamrc compile [options] <input.wasm> [-o <output.cwasm>]
    \\
    \\Compile a .wasm module to a .cwasm AOT binary. If `-o` is omitted,
    \\the output filename is derived by replacing the `.wasm` suffix on the
    \\input with `.cwasm` (or appending `.cwasm` if no `.wasm` suffix).
    \\
    \\Options:
    \\  -o <path>                     Output .cwasm path (default: <input>.cwasm)
    \\  --target=<x86_64|aarch64>     Target architecture (default: host)
    \\  -O0                           Disable IR optimizations
    \\  --aarch64-no-scheduler        Disable AArch64 instruction scheduler
    \\  --aarch64-no-xreg-alloc       Disable AArch64 X-register allocator
    \\
    \\IR diagnostics (post-frontend IR snapshots; one snapshot per
    \\matching function-pass pair, last-write-wins on repeated runs):
    \\  --dump-ir-after=<name>        Dump IR after the named pass. Use
    \\                                 `initial` for the post-frontend
    \\                                 pre-opt state and `final` for the
    \\                                 fully-optimised state. Pass names
    \\                                 are the Zig pipeline function
    \\                                 names (e.g. `forwardLocalGet`,
    \\                                 `forwardRedundantLoads`,
    \\                                 `lowerPhisToLocals`). Repeat to
    \\                                 dump after multiple passes.
    \\  --dump-ir-functions=<glob>    Restrict dumps to functions whose
    \\                                 wasm name matches the glob.
    \\                                 Supports `*` wildcards. Repeat to
    \\                                 union multiple globs. Default: `*`.
    \\                                 Each function is tried against the
    \\                                 glob under three names: its IR
    \\                                 name (currently only set via
    \\                                 frontend name-section parsing —
    \\                                 not yet implemented), the first
    \\                                 matching wasm export, and the
    \\                                 synthetic `func<N>` /
    \\                                 `func<0N>` (zero-padded) indices.
    \\  --dump-ir-out=<dir>           Directory to write dumps into
    \\                                 (created if missing). Default:
    \\                                 stdout, one snapshot per pass per
    \\                                 function preceded by a header
    \\                                 comment.
    \\  --verify-ir[=<mode>]          Run the IR invariant checker (#624)
    \\                                 after every pass that mutated the
    \\                                 function. Modes:
    \\                                   off
    \\                                   after-each-pass (default with --verify-ir)
    \\                                   load-forwarding (adds the #738/#794
    \\                                                    load-forwarding
    \\                                                    soundness check)
    \\                                   paranoid        (adds operand-width
    \\                                                    sanity check, #628)
    \\                                 Default: on for safety builds, off
    \\                                 for release builds; overridden by
    \\                                 WAMR_AOT_VERIFY_IR when set.
    \\  --no-verify-ir                Disable the IR verifier (overrides
    \\                                 the safety-build default).
    \\  --cache <path>                #761 Phase 2 codegen cache sidecar.
    \\                                 Read on entry (if exists) to reuse
    \\                                 cached native code per function
    \\                                 whose IR hash still matches; write
    \\                                 the (possibly-updated) cache back
    \\                                 to <path> after a successful emit.
    \\                                 Cache is invalidated wholesale on
    \\                                 any header mismatch (build id,
    \\                                 target arch+abi, module-level
    \\                                 codegen invariants, func count).
    \\
;

const compile_component_usage =
    \\Usage: wamrc compile-component [options] <component.wasm> [-o <manifest.json>]
    \\
    \\Precompile every embedded core module of a component, writing each
    \\as `<stem>.<N>.cwasm` next to the source `.wasm` and a versioned
    \\`<stem>.cwasm.json` manifest sidecar (with build-id and component
    \\sha256; rejected on mismatch). If `-o` is omitted, the manifest
    \\path is derived from the input by stripping `.wasm` and appending
    \\`.cwasm.json`; per-core artifacts land in the manifest's directory.
    \\
    \\Options:
    \\  -o <manifest.json>            Manifest path (default: <input>.cwasm.json)
    \\  --target=<x86_64|aarch64>     Target architecture (default: host)
    \\  -O0                           Disable IR optimizations (every core)
    \\  --verify-ir[=<mode>]          Run the IR verifier after optimized IR
    \\                                 passes. Modes: off, after-each-pass,
    \\                                 load-forwarding, paranoid, default.
    \\                                 Default: on for safety builds, off for
    \\                                 release builds; overridden by
    \\                                 WAMR_AOT_VERIFY_IR when set.
    \\  --no-verify-ir                Disable the IR verifier.
    \\  --cache-dir <dir>             #761 Phase 2 codegen cache root. Per
    \\                                 core: read/write `<dir>/core<N>.cache`
    \\                                 to reuse cached native code for any
    \\                                 function whose IR hash still matches.
    \\                                 Header-incompatible caches fall back
    \\                                 to full recompile of that core
    \\                                 (warn-only, never errors).
    \\
    \\Diagnostics env:
    \\  WAMR_AOT_PASS_TIMING=1        Log slow pass and verifier phases
    \\                                 (existing pass-timing diagnostics).
    \\  WAMR_AOT_PASS_TIMING_THRESHOLD_MS=<ms>
    \\                                 Pass/verifier threshold (default: 100).
    \\  WAMR_AOT_ANALYSIS_TIMING=1    Log slow buildSuccessors /
    \\                                 computeDominators calls.
    \\  WAMR_AOT_ANALYSIS_TIMING_THRESHOLD_MS=<ms>
    \\                                 Slow-call threshold (default: 100).
    \\  WAMR_AOT_ANALYSIS_TIMING_MODULE=<N>
    \\  WAMR_AOT_ANALYSIS_TIMING_FUNC=<N>
    \\                                 Optional module/function filters.
    \\  WAMR_AOT_CODEGEN_TIMING=1     Log native-codegen cost per function
    \\                                 (x86_64: setup/liveness/regalloc/emit
    \\                                 sub-phases + hash; module summary).
    \\  WAMR_AOT_CODEGEN_TIMING_THRESHOLD_MS=<ms>
    \\                                 Per-function threshold (default: 100).
    \\  WAMR_AOT_CODEGEN_TIMING_EVERY_N_FUNCS=<N>
    \\                                 Also log every Nth function.
    \\  WAMR_AOT_CODEGEN_TIMING_MODULE=<N>
    \\  WAMR_AOT_CODEGEN_TIMING_FUNC=<N>
    \\                                 Optional module/function filters.
    \\
;

const run_usage =
    \\Usage: wamrc run [options] <input.wasm> [-- <wamr args...>]
    \\
    \\Compile <input.wasm> if the existing artifact is missing or stale,
    \\then spawn `wamr run <artifact>` with stdin/stdout/stderr inherited
    \\and propagate its exit code. By default the artifact is written
    \\next to the source:
    \\
    \\  core wasm  : foo.wasm  ->  foo.cwasm          (+ foo.cwasm.id sidecar)
    \\  component  : foo.wasm  ->  foo.cwasm.json     (+ foo.<N>.cwasm per core)
    \\
    \\The artifact is reused only when the target-format check passes:
    \\
    \\  * core     : cwasm magic + aot_version match, plus the sidecar
    \\               records the current wamr build id, target arch, and
    \\               the source's sha256.
    \\  * component: `loadManifest` accepts the manifest sidecar (it
    \\               already checks wamr_build_id, the component sha256,
    \\               and every per-core sha256).
    \\
    \\Anything after `--` (or any positional after the input) is forwarded
    \\verbatim to `wamr run`. The `wamr` binary is located via
    \\$WAMR_BIN, then a sibling of wamrc, then PATH.
    \\
    \\Options:
    \\  -o <file>                     Override the artifact path (a `.cwasm`
    \\                                 for core wasm, a `.cwasm.json` manifest
    \\                                 for components)
    \\  --force                       Recompile even if the artifact is fresh
    \\  --target=<x86_64|aarch64>     Target architecture (default: host)
    \\
;

const verify_usage =
    \\Usage: wamrc verify [options] <input.wasm> [-- <guest args...>]
    \\
    \\Differential-test wamr-AOT against wasmtime (the reference oracle)
    \\on <input.wasm>. Compiles the wasm to AOT, runs it under both
    \\runtimes with the same flags + guest argv, diffs stdout (by
    \\default) / stderr / exit codes, and exits:
    \\
    \\  0  outputs match
    \\  1  divergence found (first-difference offset is printed with
    \\     hex+ASCII context on each side; see --hex-context)
    \\  2  setup error (missing wasmtime, AOT compile failure, …)
    \\
    \\Why wasmtime: it's the de-facto reference implementation, already
    \\installed on every wamr dev box, and a black-box oracle catches
    \\the bigger bug class — things both wamr engines share (canon-lift
    \\quirks, adapter bugs) that an interp-vs-AOT diff would miss.
    \\
    \\Motivating case: issue #754 took ~6 hours to bisect; the actual
    \\fix was 13 lines. A built-in differential tester would have
    \\shortened that to minutes. See also
    \\`.github/skills/aot-diff-debug/SKILL.md`.
    \\
    \\Binary resolution:
    \\  wasmtime: --wasmtime-bin → $WASMTIME_BIN → `wasmtime` on $PATH.
    \\  wamr:     --wamr-bin → $WAMR_BIN → sibling-of-wamrc → `wamr` on $PATH.
    \\
    \\Options:
    \\  --map-dir HOST::GUEST    Mount a host directory on both runs.
    \\                           Translates to `--dir` for wasmtime and
    \\                           `--map-dir` for wamr. Repeatable.
    \\  --env KEY=VALUE          Pass an env var to both runs. Repeatable.
    \\  --max-runtime <SEC>      Per-runtime watchdog (default 60).
    \\                           Killed and reported as divergence if
    \\                           exceeded. 0 disables the watchdog.
    \\  --hex-context <N>        Bytes of hex+ASCII context shown on each
    \\                           side of the first-diff offset (default 32).
    \\  --stdout-only            Diff only stdout (default).
    \\  --stderr-only            Diff only stderr.
    \\  --diff-everything        Diff stdout AND stderr.
    \\  --strict-exit            Require matching exit codes too. Off by
    \\                           default — wamr's "successful run then
    \\                           host SIGSEGV" failure class (#760) would
    \\                           otherwise drown out codegen regressions.
    \\  --keep-cwasm             Leave the precompiled artifacts in the
    \\                           staging dir for post-mortem inspection.
    \\                           Default: cleaned up on exit.
    \\  --json                   Emit a single-line JSON report instead
    \\                           of human-readable text. Exit-code
    \\                           semantics are unchanged.
    \\  --wasmtime-bin <path>    Override wasmtime resolution.
    \\  --wamr-bin <path>        Override wamr resolution.
    \\
;

const version_usage =
    \\Usage: wamrc version
    \\
    \\Print the wamrc version and exit.
    \\
;

const help_usage =
    \\Usage: wamrc help
    \\
    \\Print top-level help and exit.
    \\
;

fn runHelp(io: std.Io, args: []const []const u8) void {
    if (args.len == 1 and std.mem.eql(u8, args[0], "help")) {
        writeStdout(io, help_usage);
        return;
    }
    writeStdout(io, top_usage);
}

/// Derive an output `.cwasm` path from the input wasm path. Strips a
/// trailing `.wasm` suffix and appends `.cwasm`; if there is no
/// `.wasm` suffix, just appends `.cwasm`. Caller owns the returned slice.
fn deriveOutputPath(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const stem = if (std.mem.endsWith(u8, input, ".wasm"))
        input[0 .. input.len - ".wasm".len]
    else
        input;
    return std.mem.concat(allocator, u8, &.{ stem, ".cwasm" });
}

/// IR dump hook target for `wamrc --dump-ir-after=…`. Borrows its
/// `pass_names` / `func_globs` / `out_dir` / `export_names` slices from
/// the parent `runCompile` invocation; `allocator` is used only for
/// transient per-callback rendering buffers. `export_names[i]` carries
/// the first matching export name for the i-th local function (or null
/// if no export points at it) and is used as a display-friendly fallback
/// when `IrFunction.name` isn't set.
const Dumper = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    pass_names: []const []const u8,
    func_globs: []const []const u8,
    out_dir: ?[]const u8,
    export_names: []const ?[]const u8,

    /// Entry point for `passes.DumpHook.callback`. Filters by pass name
    /// and function-glob before rendering.
    fn callback(ctx: *anyopaque, info: passes.DumpInfo) anyerror!void {
        const self: *Dumper = @ptrCast(@alignCast(ctx));
        if (!self.matchesPass(info.pass_name)) return;
        if (!self.matchesFunc(info.func, info.func_index)) return;
        try self.write(info);
    }

    fn matchesPass(self: *const Dumper, name: []const u8) bool {
        for (self.pass_names) |p| {
            if (std.mem.eql(u8, p, name)) return true;
        }
        return false;
    }

    fn matchesFunc(self: *const Dumper, func: *const ir.IrFunction, func_index: u32) bool {
        const ir_name = func.name orelse "";
        const exp_name: []const u8 = if (func_index < self.export_names.len) (self.export_names[func_index] orelse "") else "";
        var idx_buf: [24]u8 = undefined;
        const idx_name = std.fmt.bufPrint(&idx_buf, "func{d}", .{func_index}) catch "func";
        var padded_buf: [24]u8 = undefined;
        const padded_name = std.fmt.bufPrint(&padded_buf, "func{d:0>4}", .{func_index}) catch "func";
        for (self.func_globs) |g| {
            if (matchGlob(g, ir_name)) return true;
            if (exp_name.len > 0 and matchGlob(g, exp_name)) return true;
            if (matchGlob(g, idx_name)) return true;
            if (matchGlob(g, padded_name)) return true;
        }
        return false;
    }

    fn displayName(self: *const Dumper, func: *const ir.IrFunction, func_index: u32) []const u8 {
        if (func.name) |n| return n;
        if (func_index < self.export_names.len) {
            if (self.export_names[func_index]) |n| return n;
        }
        return "<anon>";
    }

    fn write(self: *Dumper, info: passes.DumpInfo) !void {
        var aw: std.Io.Writer.Allocating = .init(self.allocator);
        defer aw.deinit();
        try ir_print.formatFunc(info.func, info.func_index, &aw.writer);

        const disp = self.displayName(info.func, info.func_index);

        if (self.out_dir) |dir| {
            const sanitized = try sanitizeFilename(self.allocator, disp);
            defer self.allocator.free(sanitized);
            const path = try std.fmt.allocPrint(self.allocator, "{s}/func{d:0>4}_{s}.{s}.ir", .{
                dir,
                info.func_index,
                sanitized,
                info.pass_name,
            });
            defer self.allocator.free(path);
            try std.Io.Dir.cwd().writeFile(self.io, .{
                .sub_path = path,
                .data = aw.written(),
            });
        } else {
            const header = try std.fmt.allocPrint(self.allocator, "; === pass={s} func=#{d} {s} iter={d} outer={d} changed={any} ===\n", .{
                info.pass_name,
                info.func_index,
                disp,
                info.iter,
                info.outer_iter,
                info.changed,
            });
            defer self.allocator.free(header);
            var stdout_file = std.Io.File.stdout();
            stdout_file.writeStreamingAll(self.io, header) catch {};
            stdout_file.writeStreamingAll(self.io, aw.written()) catch {};
        }
    }
};

/// Sanitize a wasm function name into a filesystem-safe basename:
/// replaces every byte that isn't `[A-Za-z0-9_.\-+]` with `_`. Result
/// is heap-allocated and owned by the caller.
fn sanitizeFilename(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    const max_len = 96;
    const src_len = @min(name.len, max_len);
    if (src_len == 0) return allocator.dupe(u8, "_");
    const out = try allocator.alloc(u8, src_len);
    for (name[0..src_len], 0..) |c, i| {
        out[i] = if (isFsSafeByte(c)) c else '_';
    }
    return out;
}

fn isFsSafeByte(c: u8) bool {
    return (c >= 'A' and c <= 'Z') or
        (c >= 'a' and c <= 'z') or
        (c >= '0' and c <= '9') or
        c == '_' or c == '.' or c == '-' or c == '+';
}

/// Match a `name` against a shell-like glob with `*` wildcards. `*`
/// matches zero or more bytes; every other byte is literal. Recursive
/// backtracking — fine for the short patterns the CLI accepts.
pub fn matchGlob(pattern: []const u8, name: []const u8) bool {
    if (pattern.len == 0) return name.len == 0;
    if (pattern[0] == '*') {
        if (matchGlob(pattern[1..], name)) return true;
        if (name.len == 0) return false;
        return matchGlob(pattern, name[1..]);
    }
    if (name.len == 0) return false;
    if (pattern[0] != name[0]) return false;
    return matchGlob(pattern[1..], name[1..]);
}

test "subcommand parsing" {
    try std.testing.expectEqual(@as(?Subcommand, .compile), parseSubcommand("compile"));
    try std.testing.expectEqual(@as(?Subcommand, .version), parseSubcommand("version"));
    try std.testing.expectEqual(@as(?Subcommand, .help), parseSubcommand("help"));
    try std.testing.expectEqual(@as(?Subcommand, null), parseSubcommand("--help"));
    try std.testing.expectEqual(@as(?Subcommand, null), parseSubcommand("foo.wasm"));
    try std.testing.expectEqual(@as(?Subcommand, null), parseSubcommand(""));
}

test "deriveOutputPath strips .wasm and appends .cwasm" {
    const a = std.testing.allocator;

    const r1 = try deriveOutputPath(a, "foo.wasm");
    defer a.free(r1);
    try std.testing.expectEqualStrings("foo.cwasm", r1);

    const r2 = try deriveOutputPath(a, "/tmp/path/to/bar.wasm");
    defer a.free(r2);
    try std.testing.expectEqualStrings("/tmp/path/to/bar.cwasm", r2);

    const r3 = try deriveOutputPath(a, "noext");
    defer a.free(r3);
    try std.testing.expectEqualStrings("noext.cwasm", r3);

    const r4 = try deriveOutputPath(a, "weird.wasm.bin");
    defer a.free(r4);
    try std.testing.expectEqualStrings("weird.wasm.bin.cwasm", r4);
}

test "matchGlob: exact match" {
    try std.testing.expect(matchGlob("foo", "foo"));
    try std.testing.expect(!matchGlob("foo", "bar"));
    try std.testing.expect(!matchGlob("foo", "foobar"));
    try std.testing.expect(!matchGlob("foo", "fo"));
}

test "matchGlob: leading/trailing wildcards" {
    try std.testing.expect(matchGlob("*", ""));
    try std.testing.expect(matchGlob("*", "anything_goes"));
    try std.testing.expect(matchGlob("core_*", "core_bench_list"));
    try std.testing.expect(matchGlob("core_*", "core_"));
    try std.testing.expect(!matchGlob("core_*", "no_match"));
    try std.testing.expect(matchGlob("*_test", "foo_test"));
    try std.testing.expect(!matchGlob("*_test", "foo_other"));
}

test "matchGlob: interior wildcards" {
    try std.testing.expect(matchGlob("a*b*c", "abc"));
    try std.testing.expect(matchGlob("a*b*c", "aXbYc"));
    try std.testing.expect(!matchGlob("a*b*c", "axbxd"));
}

test "sanitizeFilename: keeps safe bytes, replaces unsafe with _" {
    const a = std.testing.allocator;
    const s = try sanitizeFilename(a, "core/bench/list:func()");
    defer a.free(s);
    try std.testing.expectEqualStrings("core_bench_list_func__", s);
}

test "sanitizeFilename: empty name yields placeholder" {
    const a = std.testing.allocator;
    const s = try sanitizeFilename(a, "");
    defer a.free(s);
    try std.testing.expectEqualStrings("_", s);
}
