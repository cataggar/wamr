const std = @import("std");
const wamr = @import("wamr");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Parse arguments
    var wasm_path: ?[]const u8 = null;
    var wasm_args: std.ArrayListUnmanaged([]const u8) = .{};
    defer wasm_args.deinit(allocator);
    var show_version = false;
    var stack_size: u32 = 64 * 1024;
    var past_options = false;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (!past_options and arg.len > 0 and arg[0] == '-') {
            if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--version")) {
                show_version = true;
            } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
                printUsage();
                return;
            } else if (std.mem.startsWith(u8, arg, "--stack-size=")) {
                stack_size = std.fmt.parseInt(u32, arg["--stack-size=".len..], 10) catch {
                    std.debug.print("Error: invalid --stack-size value\n", .{});
                    std.process.exit(1);
                };
            } else if (std.mem.startsWith(u8, arg, "--heap-size=")) {
                // Reserved for future WASI heap allocation
            } else if (std.mem.eql(u8, arg, "--")) {
                past_options = true;
            } else {
                std.debug.print("Error: unknown option '{s}'\n", .{arg});
                printUsage();
                std.process.exit(1);
            }
        } else if (wasm_path == null) {
            wasm_path = arg;
            past_options = true;
        } else {
            try wasm_args.append(allocator, arg);
        }
    }

    if (show_version) {
        std.debug.print("iwasm (Zig) {s}\n", .{wamr.version.string});
        if (wasm_path == null) return;
    }

    const path = wasm_path orelse {
        printUsage();
        return;
    };

    // Load wasm binary
    const cwd = std.fs.cwd();
    const wasm_data = cwd.readFileAlloc(allocator, path, 256 * 1024 * 1024) catch |err| {
        std.debug.print("Error: cannot read '{s}': {}\n", .{ path, err });
        std.process.exit(1);
    };
    defer allocator.free(wasm_data);

    // Detect file type by magic bytes: AOT (\0aot) vs Wasm (\0asm)
    if (wasm_data.len >= 4 and std.mem.readInt(u32, wasm_data[0..4], .little) == wamr.types.aot_magic) {
        runAot(wasm_data, allocator);
        return;
    }

    // Wasm module (core or component)
    runWasm(wasm_data, stack_size, &wasm_args, allocator);
}

fn runAot(data: []const u8, allocator: std.mem.Allocator) void {
    const aot_loader = wamr.aot_loader;
    const aot_runtime = wamr.aot_runtime;

    const aot_module = aot_loader.load(data, allocator) catch |err| {
        std.debug.print("Error: failed to load AOT module: {}\n", .{err});
        std.process.exit(1);
    };

    const aot_inst = aot_runtime.instantiate(&aot_module, allocator) catch |err| {
        std.debug.print("Error: failed to instantiate AOT module: {}\n", .{err});
        std.process.exit(1);
    };
    _ = aot_inst;

    // AOT execution: find _start export and call native code
    const start_exp = aot_module.findExport("_start", .function) orelse
        aot_module.findExport("main", .function) orelse {
        std.debug.print("Error: no _start or main function exported in AOT module\n", .{});
        std.process.exit(1);
    };
    _ = start_exp;

    // TODO: map native code as executable and call via function pointer
    std.debug.print("AOT module loaded ({} functions, {} exports). Native execution not yet implemented.\n", .{
        aot_module.func_count, aot_module.exports.len,
    });
}

fn runWasm(
    wasm_data: []const u8,
    stack_size: u32,
    wasm_args: *std.ArrayListUnmanaged([]const u8),
    allocator: std.mem.Allocator,
) void {
    var runtime = wamr.wamr.Runtime.init(allocator);
    defer runtime.deinit();

    var module = runtime.loadModule(wasm_data) catch |err| {
        std.debug.print("Error: failed to load module: {}\n", .{err});
        std.process.exit(1);
    };
    defer module.deinit();

    var instance = module.instantiate() catch |err| {
        std.debug.print("Error: failed to instantiate: {}\n", .{err});
        std.process.exit(1);
    };
    defer instance.deinit();

    const start_func = module.findExport("_start", .function) orelse
        module.findExport("main", .function) orelse {
        std.debug.print("Error: no _start or main function exported\n", .{});
        std.process.exit(1);
    };

    const func_type = module.inner.getFuncType(start_func.index);
    const param_count = if (func_type) |ft| ft.params.len else 0;

    var env = wamr.exec_env.ExecEnv.create(instance.inner, stack_size, allocator) catch |err| {
        std.debug.print("Error: failed to create execution environment: {}\n", .{err});
        std.process.exit(1);
    };
    defer env.destroy();

    if (param_count >= 2) {
        env.pushI32(@intCast(wasm_args.items.len + 1)) catch {};
        env.pushI32(0) catch {};
    }

    wamr.interp.executeFunction(env, start_func.index) catch |err| {
        std.debug.print("Error: execution trapped: {}\n", .{err});
        std.process.exit(1);
    };
}

fn printUsage() void {
    std.debug.print(
        \\iwasm (Zig) - WebAssembly Micro Runtime
        \\
        \\Usage: wamr [options] <file.wasm> [args...]
        \\
        \\Options:
        \\  -v, --version           Show version
        \\  -h, --help              Show this help
        \\  --stack-size=<bytes>     Set stack size (default: 65536)
        \\  --heap-size=<bytes>     Set heap size (default: 262144)
        \\
    , .{});
}
