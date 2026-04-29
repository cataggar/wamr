const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");

const aot_supported = switch (builtin.cpu.arch) {
    .x86_64, .aarch64 => true,
    else => false,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    const args = try init.minimal.args.toSlice(init.arena.allocator());

    // Parse arguments
    var wasm_path: ?[]const u8 = null;
    var wasm_args: std.ArrayListUnmanaged([]const u8) = .empty;
    defer wasm_args.deinit(allocator);
    var show_version = false;
    var listen_address: ?std.Io.net.IpAddress = null;
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
            } else if (std.mem.startsWith(u8, arg, "--listen=")) {
                const spec = arg["--listen=".len..];
                listen_address = parseListenAddress(spec) catch {
                    std.debug.print("Error: invalid --listen address '{s}'\n", .{spec});
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
        printVersion();
        if (wasm_path == null) return;
    }

    const path = wasm_path orelse {
        printUsage();
        return;
    };

    // Load wasm binary
    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const wasm_data = cwd.readFileAlloc(io, path, allocator, @enumFromInt(256 * 1024 * 1024)) catch |err| {
        std.debug.print("Error: cannot read '{s}': {}\n", .{ path, err });
        std.process.exit(1);
    };
    defer allocator.free(wasm_data);

    // Detect file type by magic bytes: AOT (\0aot) vs Wasm (\0asm)
    if (wasm_data.len >= 4 and std.mem.readInt(u32, wasm_data[0..4], .little) == wamr.types.aot_magic) {
        if (listen_address != null) {
            std.debug.print("Error: --listen is only supported for WASI HTTP components\n", .{});
            std.process.exit(1);
        }
        runAot(wasm_data, allocator);
        return;
    }

    // Distinguish core wasm from a component by the version word
    // (both share `\0asm`). Core wasm = 0x0000_0001, component = 0x0001_000d.
    if (wasm_data.len >= 8 and std.mem.readInt(u32, wasm_data[0..4], .little) == wamr.types.wasm_magic) {
        const version = std.mem.readInt(u32, wasm_data[4..8], .little);
        if (version == wamr.types.component_version) {
            // Inherit host env. EnvVar slices borrow from the existing
            // environ_map (lifetime: the entire process).
            var env_list: std.ArrayListUnmanaged(wamr.wasi_cli_adapter.EnvVar) = .empty;
            defer env_list.deinit(allocator);
            var it = init.environ_map.array_hash_map.iterator();
            while (it.next()) |kv| {
                env_list.append(allocator, .{ .name = kv.key_ptr.*, .value = kv.value_ptr.* }) catch {};
            }
            if (listen_address) |addr| {
                runHttpComponent(wasm_data, allocator, path, wasm_args.items, env_list.items, addr);
                return;
            }
            runComponent(wasm_data, allocator, io, path, wasm_args.items, env_list.items);
            return;
        }
    }

    if (listen_address != null) {
        std.debug.print("Error: --listen is only supported for WASI HTTP components\n", .{});
        std.process.exit(1);
    }

    // Wasm module (core)
    runWasm(wasm_data, stack_size, &wasm_args, allocator);
}

fn parseListenAddress(spec: []const u8) !std.Io.net.IpAddress {
    if (spec.len == 0) return error.InvalidAddress;

    var host: []const u8 = undefined;
    var port_text: []const u8 = undefined;
    if (spec[0] == '[') {
        const close = std.mem.indexOfScalar(u8, spec, ']') orelse return error.InvalidAddress;
        if (close + 1 >= spec.len or spec[close + 1] != ':') return error.InvalidAddress;
        host = spec[1..close];
        port_text = spec[close + 2 ..];
    } else {
        var colon: ?usize = null;
        var i = spec.len;
        while (i > 0) {
            i -= 1;
            if (spec[i] == ':') {
                colon = i;
                break;
            }
        }
        const c = colon orelse return error.InvalidAddress;
        host = spec[0..c];
        port_text = spec[c + 1 ..];
    }

    if (host.len == 0 or port_text.len == 0) return error.InvalidAddress;
    const port = try std.fmt.parseInt(u16, port_text, 10);
    return std.Io.net.IpAddress.parse(host, port);
}

fn runComponent(
    data: []const u8,
    allocator: std.mem.Allocator,
    io: std.Io,
    wasm_path: []const u8,
    wasm_args: []const []const u8,
    env_vars: []const wamr.wasi_cli_adapter.EnvVar,
) void {
    const adapter_mod = wamr.wasi_cli_adapter;
    var adapter = adapter_mod.WasiCliAdapter.init(allocator);
    defer adapter.deinit();

    // argv[0] = wasm path, rest = user args (matches wasmtime convention).
    var argv_buf = allocator.alloc([]const u8, 1 + wasm_args.len) catch
        std.process.exit(1);
    defer allocator.free(argv_buf);
    argv_buf[0] = wasm_path;
    for (wasm_args, 0..) |a, i| argv_buf[i + 1] = a;
    adapter.setArguments(argv_buf);
    adapter.setEnvironment(env_vars);

    const outcome = adapter_mod.runComponentBytes(data, allocator, &adapter) catch |err| {
        switch (err) {
            error.NoRunExport => std.debug.print(
                "Error: component does not expose a top-level `run` export. " ++
                    "Real wasi:cli/run instance exports are not yet wired (issue #142).\n",
                .{},
            ),
            error.LinkFailed => std.debug.print(
                "Error: component imports something other than wasi:cli/stdout + wasi:io/streams " ++
                    "(only those are wired in this build).\n",
                .{},
            ),
            error.LoadFailed => std.debug.print("Error: failed to load component\n", .{}),
            error.InstantiateFailed => std.debug.print("Error: failed to instantiate component\n", .{}),
            error.Trap => std.debug.print("Error: component trapped during run\n", .{}),
            else => std.debug.print("Error: component run failed: {}\n", .{err}),
        }
        std.process.exit(1);
    };

    // Flush captured stdout to the host. Buffered + flush at end is the
    // simplest cross-platform path; streaming output is deferred until
    // the io/poll-aware adapter lands (#154).
    const captured = adapter.getStdoutBytes();
    if (captured.len > 0) {
        var stdout_file = std.Io.File.stdout();
        stdout_file.writeStreamingAll(io, captured) catch {};
    }

    std.process.exit(if (outcome.is_ok) 0 else 1);
}

fn runHttpComponent(
    data: []const u8,
    allocator: std.mem.Allocator,
    wasm_path: []const u8,
    wasm_args: []const []const u8,
    env_vars: []const wamr.wasi_cli_adapter.EnvVar,
    listen_address: std.Io.net.IpAddress,
) void {
    const adapter_mod = wamr.wasi_cli_adapter;
    var adapter = adapter_mod.WasiCliAdapter.init(allocator);
    defer adapter.deinit();

    var argv_buf = allocator.alloc([]const u8, 1 + wasm_args.len) catch
        std.process.exit(1);
    defer allocator.free(argv_buf);
    argv_buf[0] = wasm_path;
    for (wasm_args, 0..) |a, i| argv_buf[i + 1] = a;
    adapter.setArguments(argv_buf);
    adapter.setEnvironment(env_vars);

    adapter_mod.serveHttpComponentBytes(data, allocator, &adapter, .{
        .listen_address = listen_address,
    }) catch |err| {
        switch (err) {
            error.NoIncomingHandlerExport => std.debug.print(
                "Error: component does not export `wasi:http/incoming-handler.handle`.\n",
                .{},
            ),
            error.LinkFailed => std.debug.print(
                "Error: component imports an unsupported WASI interface for HTTP server mode.\n",
                .{},
            ),
            error.ListenFailed => std.debug.print("Error: failed to bind --listen address\n", .{}),
            error.AcceptFailed => std.debug.print("Error: failed to accept HTTP connection\n", .{}),
            error.LoadFailed => std.debug.print("Error: failed to load component\n", .{}),
            error.InstantiateFailed => std.debug.print("Error: failed to instantiate component\n", .{}),
            else => std.debug.print("Error: HTTP server failed: {}\n", .{err}),
        }
        std.process.exit(1);
    };
}

fn runAot(data: []const u8, allocator: std.mem.Allocator) void {
    if (comptime aot_supported) {
        runAotReal(data, allocator);
    } else {
        std.debug.print("Error: AOT execution not supported on this architecture\n", .{});
        std.process.exit(1);
    }
}

fn runAotReal(data: []const u8, allocator: std.mem.Allocator) void {
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
    defer aot_runtime.destroy(aot_inst);

    // Map native code as executable
    aot_runtime.mapCodeExecutable(aot_inst) catch |err| {
        std.debug.print("Error: failed to map code as executable: {}\n", .{err});
        std.process.exit(1);
    };

    // Find _start or main export
    const func_idx = aot_runtime.findExportFunc(aot_inst, "_start") orelse
        aot_runtime.findExportFunc(aot_inst, "main") orelse {
        std.debug.print("Error: no _start or main function exported in AOT module\n", .{});
        std.process.exit(1);
    };

    // Execute
    const result = aot_runtime.callFunc(aot_inst, func_idx, i32) catch |err| {
        std.debug.print("Error: AOT execution failed: {}\n", .{err});
        std.process.exit(1);
    };
    std.debug.print("{d}\n", .{result});
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

fn versionLine() []const u8 {
    return "wamr " ++ wamr.version.string ++ "\n";
}

fn printVersion() void {
    std.debug.print("{s}", .{versionLine()});
}

fn printUsage() void {
    std.debug.print(
        \\wamr - WebAssembly Micro Runtime
        \\
        \\Usage: wamr [options] <file.wasm> [args...]
        \\
        \\Options:
        \\  -v, --version           Show version
        \\  -h, --help              Show this help
        \\  --stack-size=<bytes>     Set stack size (default: 65536)
        \\  --heap-size=<bytes>     Set heap size (default: 262144)
        \\  --listen=<ip:port>       Serve a WASI HTTP component on a TCP address
        \\
    , .{});
}

test "version line uses wamr name and build version" {
    try std.testing.expectEqualStrings("wamr " ++ wamr.version.string ++ "\n", versionLine());
}
