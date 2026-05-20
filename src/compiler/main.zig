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

const TargetArch = passes.TargetArch;

const Subcommand = enum { compile, version, help };

fn parseSubcommand(s: []const u8) ?Subcommand {
    if (std.mem.eql(u8, s, "compile")) return .compile;
    if (std.mem.eql(u8, s, "version")) return .version;
    if (std.mem.eql(u8, s, "help")) return .help;
    return null;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

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
    }
}

fn runVersion(io: std.Io, args: []const []const u8) !void {
    if (args.len == 1 and std.mem.eql(u8, args[0], "help")) {
        writeStdout(io, version_usage);
        return;
    }
    writeStdout(io, "wamrc " ++ wamr.version.string ++ "\n");
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
    // `--verify-ir[=…]` or `--no-verify-ir`.
    var verify_mode: wamr.ir_verifier.VerifyMode =
        if (std.debug.runtime_safety) .after_each_pass else .off;

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
        } else if (std.mem.eql(u8, a, "--verify-ir=after-each-pass")) {
            verify_mode = .after_each_pass;
        } else if (std.mem.eql(u8, a, "--verify-ir=paranoid")) {
            verify_mode = .paranoid;
        } else if (std.mem.eql(u8, a, "--no-verify-ir")) {
            verify_mode = .off;
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
        const run_opts: passes.RunOptions = .{
            .dump_hook = if (dumper.pass_names.len == 0) null else .{
                .ctx = @ptrCast(&dumper),
                .callback = Dumper.callback,
            },
            .verify_mode = verify_mode,
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

    // 5. Compile IR to native code (target-dependent)
    const CompileResult = x86_64_compile.CompileResult;
    const compiled: CompileResult = switch (target_arch) {
        .x86_64 => x86_64_compile.compileModule(&ir_module, allocator) catch |err| {
            std.debug.print("Error compiling to x86-64: {}\n", .{err});
            std.process.exit(1);
        },
        .aarch64 => blk: {
            const r = aarch64_compile.compileModuleWithOptions(&ir_module, allocator, .{
                .enable_scheduler = enable_aarch64_scheduler,
                .enable_xreg_alloc = enable_aarch64_xreg_alloc,
            }) catch |err| {
                std.debug.print("Error compiling to AArch64: {}\n", .{err});
                std.process.exit(1);
            };
            break :blk .{ .code = r.code, .offsets = r.offsets };
        },
    };
    defer allocator.free(compiled.code);
    defer allocator.free(compiled.offsets);

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
        if (imp.kind == .function) {
            try import_entries.append(allocator, .{
                .module_name = imp.module_name,
                .field_name = imp.field_name,
                .kind = .function,
                .func_type_idx = imp.func_type_idx orelse 0,
            });
        }
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

    // Build element segment entries
    var elem_entries: std.ArrayList(emit_aot.ElemEntry) = .empty;
    defer elem_entries.deinit(allocator);
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

const top_usage =
    \\wamrc - WebAssembly AOT Compiler
    \\
    \\Usage: wamrc <subcommand> [args...]
    \\
    \\Subcommands:
    \\  compile   Compile a .wasm module to a .cwasm AOT binary
    \\  version   Print version and exit
    \\  help      Print this help
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
    \\                                   after-each-pass (default with --verify-ir)
    \\                                   paranoid        (reserved; same as
    \\                                                    after-each-pass today)
    \\                                 Default: on for safety builds, off
    \\                                 for release builds.
    \\  --no-verify-ir                Disable the IR verifier (overrides
    \\                                 the safety-build default).
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
