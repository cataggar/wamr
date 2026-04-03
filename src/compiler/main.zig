//! wamrc — WebAssembly AOT Compiler (Zig implementation)
//!
//! Compiles .wasm files to WAMR AOT format using the Zig-native compiler backend.

const std = @import("std");
const wamr = @import("wamr");
const emit_aot = wamr.emit_aot;
const compile = wamr.x86_64_compile;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // Collect command-line arguments
    var args_list: std.ArrayList([]const u8) = .empty;
    defer args_list.deinit(allocator);
    {
        var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
        defer it.deinit();
        while (it.next()) |arg| {
            try args_list.append(allocator, arg);
        }
    }
    const args = args_list.items;

    if (args.len < 3) {
        std.debug.print("Usage: wamrc <input.wasm> -o <output.aot>\n", .{});
        std.debug.print("\nWebAssembly AOT Compiler (Zig)\n", .{});
        std.process.exit(1);
    }

    var input_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "-o") and i + 1 < args.len) {
            i += 1;
            output_path = args[i];
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
    const cwd = std.Io.Dir.cwd();
    const io = init.io;
    const wasm_data = cwd.readFileAlloc(io, in_path, allocator, std.Io.Limit.limited(64 * 1024 * 1024)) catch |err| {
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

    // 4–5. Compile IR to native x86-64 code
    const compiled = compile.compileModule(&ir_module, allocator) catch |err| {
        std.debug.print("Error compiling to native code: {}\n", .{err});
        std.process.exit(1);
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
    @memcpy(arch_name[0..6], "x86-64");

    const aot_binary = try emit_aot.emit(
        allocator,
        compiled.code,
        compiled.offsets,
        exports.items,
        .{ .arch = arch_name },
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
