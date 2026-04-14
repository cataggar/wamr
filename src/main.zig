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

    // Load module
    var runtime = wamr.wamr.Runtime.init(allocator);
    defer runtime.deinit();

    var module = runtime.loadModule(wasm_data) catch |err| {
        std.debug.print("Error: failed to load module: {}\n", .{err});
        std.process.exit(1);
    };
    defer module.deinit();

    // Instantiate
    var instance = module.instantiate() catch |err| {
        std.debug.print("Error: failed to instantiate: {}\n", .{err});
        std.process.exit(1);
    };
    defer instance.deinit();

    // Look for _start or main
    const start_func = module.findExport("_start", .function) orelse
        module.findExport("main", .function) orelse {
        std.debug.print("Error: no _start or main function exported\n", .{});
        std.process.exit(1);
    };

    // Execute
    const func_type = module.inner.getFuncType(start_func.index);
    const param_count = if (func_type) |ft| ft.params.len else 0;

    var env = wamr.exec_env.ExecEnv.create(instance.inner, stack_size, allocator) catch |err| {
        std.debug.print("Error: failed to create execution environment: {}\n", .{err});
        std.process.exit(1);
    };
    defer env.destroy();

    // Push dummy args if _start expects parameters (argc, argv for WASI)
    if (param_count >= 2) {
        try env.pushI32(@intCast(wasm_args.items.len + 1)); // argc
        try env.pushI32(0); // argv (null — no linear memory argv support yet)
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
