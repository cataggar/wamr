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
        .version => {
            writeStdout(init.io, "wamrc " ++ wamr.version.string ++ "\n");
            return;
        },
        .help => {
            runHelp(init.io, args[2..]);
            return;
        },
        .compile => try runCompile(init, allocator, args[2..]),
    }
}

fn runCompile(init: std.process.Init, allocator: std.mem.Allocator, sub_args: []const []const u8) !void {
    var input_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var optimize = true;
    var enable_aarch64_scheduler = true;
    var enable_aarch64_xreg_alloc = true;
    var target_arch: TargetArch = switch (builtin.cpu.arch) {
        .aarch64 => .aarch64,
        else => .x86_64,
    };

    var i: usize = 0;
    while (i < sub_args.len) : (i += 1) {
        const a = sub_args[i];
        if (std.mem.eql(u8, a, "-h") or std.mem.eql(u8, a, "--help")) {
            runHelp(init.io, &.{"compile"});
            return;
        } else if (std.mem.eql(u8, a, "-o") and i + 1 < sub_args.len) {
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
        } else if (a.len > 0 and a[0] == '-') {
            std.debug.print("error: unknown option '{s}' — try `wamrc help compile`\n", .{a});
            std.process.exit(1);
        } else if (input_path == null) {
            input_path = a;
        } else {
            std.debug.print("error: unexpected positional argument '{s}'\n", .{a});
            std.process.exit(1);
        }
    }

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

    // 2. Parse wasm module
    const module = wamr.loader.load(wasm_data, allocator) catch |err| {
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

    // 4. Optimize IR (unless -O0)
    if (optimize) {
        const opt_changes = passes.runPasses(&ir_module, passes.defaultPassesForTarget(target_arch), allocator) catch |err| {
            std.debug.print("Error optimizing IR: {}\n", .{err});
            std.process.exit(1);
        };
        std.debug.print("Optimization: {d} passes made changes\n", .{opt_changes});
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
    \\  help      Print this help; `wamrc help <subcommand>` for details
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
    \\  -h, --help                    Show this help
    \\
;

const version_usage =
    \\Usage: wamrc version
    \\
    \\Print the wamrc version and exit.
    \\
;

const help_usage =
    \\Usage: wamrc help [subcommand]
    \\
    \\Print top-level help, or help for a specific subcommand.
    \\
;

fn runHelp(io: std.Io, args: []const []const u8) void {
    if (args.len == 0) {
        writeStdout(io, top_usage);
        return;
    }
    const sub = parseSubcommand(args[0]) orelse {
        std.debug.print("error: unknown subcommand '{s}' — try `wamrc help`\n", .{args[0]});
        std.process.exit(1);
    };
    writeStdout(io, switch (sub) {
        .compile => compile_usage,
        .version => version_usage,
        .help => help_usage,
    });
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
