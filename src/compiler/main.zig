//! wamrc — WebAssembly AOT Compiler (Zig implementation)
//!
//! Compiles .wasm files to WAMR AOT format using the Zig-native compiler backend.

const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");
const emit_aot = wamr.emit_aot;
const x86_64_compile = wamr.x86_64_compile;
const aarch64_compile = wamr.aarch64_compile;
const passes = wamr.passes;

const TargetArch = enum { x86_64, aarch64 };

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    const args = try init.minimal.args.toSlice(init.arena.allocator());

    if (args.len < 3) {
        std.debug.print("Usage: wamrc <input.wasm> -o <output.aot>\n", .{});
        std.debug.print("\nWebAssembly AOT Compiler (Zig)\n", .{});
        std.process.exit(1);
    }

    var input_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var optimize = true;
    var target_arch: TargetArch = switch (builtin.cpu.arch) {
        .aarch64 => .aarch64,
        else => .x86_64,
    };

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "-o") and i + 1 < args.len) {
            i += 1;
            output_path = args[i];
        } else if (std.mem.eql(u8, args[i], "-O0")) {
            optimize = false;
        } else if (std.mem.eql(u8, args[i], "--target") and i + 1 < args.len) {
            i += 1;
            if (std.mem.eql(u8, args[i], "aarch64")) {
                target_arch = .aarch64;
            } else if (std.mem.eql(u8, args[i], "x86_64") or std.mem.eql(u8, args[i], "x86-64")) {
                target_arch = .x86_64;
            } else {
                std.debug.print("Error: unknown target '{s}' (supported: x86_64, aarch64)\n", .{args[i]});
                std.process.exit(1);
            }
        } else {
            input_path = args[i];
        }
    }

    const in_path = input_path orelse {
        std.debug.print("Error: no input file specified\n", .{});
        std.process.exit(1);
    };
    const out_path = output_path orelse {
        std.debug.print("Error: no output file specified (use -o)\n", .{});
        std.process.exit(1);
    };

    // 1. Read input wasm
    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const wasm_data = cwd.readFileAlloc(io, in_path, allocator, @enumFromInt(64 * 1024 * 1024)) catch |err| {
        std.debug.print("Error reading {s}: {}\n", .{ in_path, err });
        std.process.exit(1);
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
        const opt_changes = passes.runPasses(&ir_module, passes.default_passes, allocator) catch |err| {
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
            const r = aarch64_compile.compileModule(&ir_module, allocator) catch |err| {
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

    const aot_binary = try emit_aot.emit(
        allocator,
        compiled.code,
        compiled.offsets,
        exports.items,
        .{ .arch = arch_name },
        if (data_segs.items.len > 0) data_segs.items else null,
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
