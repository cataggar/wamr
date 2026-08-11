const std = @import("std");
const build_config = @import("config");
const types = @import("runtime/common/types.zig");
const loader = @import("runtime/interpreter/loader.zig");
const instance = @import("runtime/interpreter/instance.zig");
const interp = @import("runtime/interpreter/interp.zig");
const ExecEnv = @import("runtime/common/exec_env.zig").ExecEnv;
const WasiCtx = @import("wasi/wasi.zig").WasiCtx;

const default_stack_size: u32 = 64 * 1024;

const MapDir = struct {
    host_path: []const u8,
    guest_name: []const u8,
};

pub fn main(init: std.process.Init) !u8 {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const cli_args = if (args.len > 1 and std.mem.eql(u8, args[1], "--"))
        args[2..]
    else
        args[1..];

    if (cli_args.len == 0) {
        std.debug.print("error: missing subcommand - try `wamr help`\n", .{});
        return 2;
    }
    if (std.mem.eql(u8, cli_args[0], "--version") or std.mem.eql(u8, cli_args[0], "version")) {
        if (cli_args.len != 1) {
            std.debug.print("error: version takes no arguments\n", .{});
            return 2;
        }
        writeStdout(init.io, "wamr " ++ build_config.version ++ "\n");
        return 0;
    }
    if (std.mem.eql(u8, cli_args[0], "help")) {
        writeStdout(init.io, top_usage);
        return 0;
    }
    if (std.mem.eql(u8, cli_args[0], "run")) {
        return runCore(init, allocator, cli_args[1..]);
    }

    std.debug.print(
        "error: unsupported subcommand '{s}' in the wasm32-wasi interpreter build\n",
        .{cli_args[0]},
    );
    return 2;
}

fn runCore(
    init: std.process.Init,
    allocator: std.mem.Allocator,
    run_args: []const []const u8,
) !u8 {
    if (run_args.len == 1 and std.mem.eql(u8, run_args[0], "help")) {
        writeStdout(init.io, run_usage);
        return 0;
    }

    var stack_size = default_stack_size;
    var wasm_path: ?[]const u8 = null;
    var wasm_args: std.ArrayListUnmanaged([]const u8) = .empty;
    defer wasm_args.deinit(allocator);
    var env_flags: std.ArrayListUnmanaged([]const u8) = .empty;
    defer env_flags.deinit(allocator);
    var map_dirs: std.ArrayListUnmanaged(MapDir) = .empty;
    defer map_dirs.deinit(allocator);
    var past_options = false;

    var i: usize = 0;
    while (i < run_args.len) : (i += 1) {
        const arg = run_args[i];
        if (!past_options and arg.len > 0 and arg[0] == '-') {
            if (std.mem.eql(u8, arg, "--")) {
                past_options = true;
            } else if (std.mem.startsWith(u8, arg, "--stack-size=")) {
                const value = arg["--stack-size=".len..];
                stack_size = std.fmt.parseInt(u32, value, 10) catch {
                    std.debug.print("error: invalid --stack-size value '{s}'\n", .{value});
                    return 2;
                };
                if (stack_size == 0) {
                    std.debug.print("error: --stack-size must be greater than zero\n", .{});
                    return 2;
                }
            } else if (std.mem.eql(u8, arg, "--env") or std.mem.startsWith(u8, arg, "--env=")) {
                const spec = if (std.mem.eql(u8, arg, "--env")) blk: {
                    i += 1;
                    if (i >= run_args.len) {
                        std.debug.print("error: --env requires KEY=VALUE\n", .{});
                        return 2;
                    }
                    break :blk run_args[i];
                } else arg["--env=".len..];
                if (std.mem.indexOfScalar(u8, spec, '=') == null) {
                    std.debug.print("error: --env value '{s}' is missing '='\n", .{spec});
                    return 2;
                }
                try env_flags.append(allocator, spec);
            } else if (std.mem.eql(u8, arg, "--map-dir") or std.mem.startsWith(u8, arg, "--map-dir=")) {
                const spec = if (std.mem.eql(u8, arg, "--map-dir")) blk: {
                    i += 1;
                    if (i >= run_args.len) {
                        std.debug.print("error: --map-dir requires HOST::GUEST\n", .{});
                        return 2;
                    }
                    break :blk run_args[i];
                } else arg["--map-dir=".len..];
                const mapping = parseMapDir(spec) catch {
                    std.debug.print(
                        "error: --map-dir value '{s}' must be HOST::GUEST\n",
                        .{spec},
                    );
                    return 2;
                };
                try map_dirs.append(allocator, mapping);
            } else {
                std.debug.print("error: unknown option '{s}' - try `wamr run help`\n", .{arg});
                return 2;
            }
        } else if (wasm_path == null) {
            wasm_path = arg;
            past_options = true;
        } else {
            try wasm_args.append(allocator, arg);
        }
    }

    const path = wasm_path orelse {
        std.debug.print("error: missing core wasm file\n", .{});
        return 2;
    };

    const cwd = std.Io.Dir.cwd();
    const wasm_data = cwd.readFileAlloc(
        init.io,
        path,
        allocator,
        @enumFromInt(256 * 1024 * 1024),
    ) catch |err| {
        std.debug.print("error: cannot read '{s}': {s}\n", .{ path, @errorName(err) });
        return 1;
    };
    defer allocator.free(wasm_data);

    if (wasm_data.len >= 4 and
        std.mem.readInt(u32, wasm_data[0..4], .little) == types.aot_magic)
    {
        std.debug.print("error: AOT modules cannot execute inside a wasm32-wasi host\n", .{});
        return 2;
    }
    if (wasm_data.len >= 8 and
        std.mem.readInt(u32, wasm_data[0..4], .little) == types.wasm_magic and
        std.mem.readInt(u32, wasm_data[4..8], .little) == types.component_version)
    {
        std.debug.print("error: Component Model execution is not supported by this wasm32-wasi build\n", .{});
        return 2;
    }

    return executeCore(
        init.io,
        allocator,
        wasm_data,
        path,
        wasm_args.items,
        env_flags.items,
        map_dirs.items,
        stack_size,
    );
}

fn executeCore(
    io: std.Io,
    allocator: std.mem.Allocator,
    wasm_data: []const u8,
    wasm_path: []const u8,
    wasm_args: []const []const u8,
    env_flags: []const []const u8,
    map_dirs: []const MapDir,
    stack_size: u32,
) u8 {
    var module_arena = std.heap.ArenaAllocator.init(allocator);
    defer module_arena.deinit();

    const module = loader.load(wasm_data, module_arena.allocator()) catch |err| {
        std.debug.print("error: failed to load core module: {s}\n", .{@errorName(err)});
        return 1;
    };
    const module_inst = instance.instantiate(&module, allocator) catch |err| {
        std.debug.print("error: failed to instantiate core module: {s}\n", .{@errorName(err)});
        return 1;
    };
    defer instance.destroy(module_inst);

    const wasi_start = module.findExport("_start", .function);
    const start_func = wasi_start orelse module.findExport("main", .function) orelse {
        std.debug.print("error: no _start or main function exported\n", .{});
        return 1;
    };
    const is_main = wasi_start == null;
    const func_type = module.getFuncType(start_func.index) orelse {
        std.debug.print("error: entry point has no function type\n", .{});
        return 1;
    };
    if (func_type.params.len != 0) {
        std.debug.print("error: entry point parameters are not supported; use a WASI _start export\n", .{});
        return 1;
    }
    if ((!is_main and func_type.results.len != 0) or
        (is_main and (func_type.results.len > 1 or
            (func_type.results.len == 1 and func_type.results[0] != .i32))))
    {
        std.debug.print("error: unsupported entry point result signature\n", .{});
        return 1;
    }

    var env = ExecEnv.create(module_inst, stack_size, allocator) catch |err| {
        std.debug.print("error: failed to create execution environment: {s}\n", .{@errorName(err)});
        return 1;
    };
    defer env.destroy();

    var argv: std.ArrayListUnmanaged([]const u8) = .empty;
    defer argv.deinit(allocator);
    argv.append(allocator, wasm_path) catch return 1;
    for (wasm_args) |arg| argv.append(allocator, arg) catch return 1;

    const wasi_ctx = WasiCtx.init(allocator, io) catch |err| {
        std.debug.print("error: failed to create WASI context: {s}\n", .{@errorName(err)});
        return 1;
    };
    defer wasi_ctx.deinit();
    wasi_ctx.setArgs(argv.items);
    wasi_ctx.setEnv(env_flags);

    for (map_dirs) |mapping| {
        _ = wasi_ctx.openMappedDir(mapping.host_path, mapping.guest_name) catch |err| {
            std.debug.print(
                "error: cannot pre-open '{s}' as '{s}': {s}\n",
                .{ mapping.host_path, mapping.guest_name, @errorName(err) },
            );
            return 1;
        };
    }
    env.wasi_ctx = @ptrCast(wasi_ctx);

    interp.executeFunction(env, start_func.index) catch |err| {
        if (wasi_ctx.exit_code) |code| return @truncate(code);
        std.debug.print("error: execution trapped: {s}\n", .{@errorName(err)});
        return 1;
    };
    if (wasi_ctx.exit_code) |code| return @truncate(code);
    if (is_main and func_type.results.len == 1) {
        const code = env.popI32() catch return 1;
        return @truncate(@as(u32, @bitCast(code)));
    }
    return 0;
}

fn parseMapDir(spec: []const u8) !MapDir {
    const separator = std.mem.indexOf(u8, spec, "::") orelse
        return error.MissingSeparator;
    if (separator == 0 or separator + 2 == spec.len)
        return error.MissingSeparator;
    return .{
        .host_path = spec[0..separator],
        .guest_name = spec[separator + 2 ..],
    };
}

fn writeStdout(io: std.Io, text: []const u8) void {
    var stdout_file = std.Io.File.stdout();
    stdout_file.writeStreamingAll(io, text) catch {};
}

const top_usage =
    \\wamr - WebAssembly Micro Runtime (wasm32-wasi interpreter build)
    \\
    \\Usage: wamr <subcommand> [args...]
    \\
    \\Subcommands:
    \\  run       Interpret a core .wasm module
    \\  version   Print version and exit
    \\  help      Print this help
    \\
    \\AOT, JIT, Component Model, serve, and host threads are unavailable.
    \\
;

const run_usage =
    \\Usage: wamr run [options] <file.wasm> [args...]
    \\
    \\Options:
    \\  --stack-size=<bytes>     Interpreter stack size (default: 65536)
    \\  --env KEY=VALUE          Set a WASI environment variable (repeatable)
    \\  --map-dir HOST::GUEST    Pre-open HOST as GUEST (repeatable)
    \\
;
