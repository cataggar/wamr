const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");

const aot_supported = switch (builtin.cpu.arch) {
    .x86_64, .aarch64 => true,
    else => false,
};

/// `--map-dir HOST::GUEST` flag value: pre-open `host_path` on the host and
/// expose it to the guest under `guest_name`. Slices borrow from `args`,
/// which lives for the entire process.
pub const MapDir = struct {
    host_path: []const u8,
    guest_name: []const u8,
};

const Subcommand = enum { run, version, help };

fn parseSubcommand(s: []const u8) ?Subcommand {
    if (std.mem.eql(u8, s, "run")) return .run;
    if (std.mem.eql(u8, s, "version")) return .version;
    if (std.mem.eql(u8, s, "help")) return .help;
    return null;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    if (args.len < 2) {
        std.debug.print("error: missing subcommand — try `wamr help`\n", .{});
        std.process.exit(1);
    }

    const subcmd = parseSubcommand(args[1]) orelse {
        std.debug.print("error: unknown subcommand '{s}' — try `wamr help`\n", .{args[1]});
        std.process.exit(1);
    };

    switch (subcmd) {
        .version => {
            writeStdout(init.io, "wamr " ++ wamr.version.string ++ "\n");
            return;
        },
        .help => {
            runHelp(init.io, args[2..]);
            return;
        },
        .run => try runRun(init, allocator, args[2..]),
    }
}

fn runRun(init: std.process.Init, allocator: std.mem.Allocator, run_args: []const []const u8) !void {
    var wasm_path: ?[]const u8 = null;
    var wasm_args: std.ArrayListUnmanaged([]const u8) = .empty;
    defer wasm_args.deinit(allocator);
    var env_flags: std.ArrayListUnmanaged([]const u8) = .empty;
    defer env_flags.deinit(allocator);
    var map_dirs: std.ArrayListUnmanaged(MapDir) = .empty;
    defer map_dirs.deinit(allocator);
    var listen_address: ?std.Io.net.IpAddress = null;
    var stack_size: u32 = 64 * 1024;
    var past_options = false;

    var i: usize = 0;
    while (i < run_args.len) : (i += 1) {
        const arg = run_args[i];
        if (!past_options and arg.len > 0 and arg[0] == '-') {
            if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
                runHelp(init.io, &.{"run"});
                return;
            } else if (std.mem.startsWith(u8, arg, "--stack-size=")) {
                stack_size = std.fmt.parseInt(u32, arg["--stack-size=".len..], 10) catch {
                    std.debug.print("error: invalid --stack-size value\n", .{});
                    std.process.exit(1);
                };
            } else if (std.mem.startsWith(u8, arg, "--listen=")) {
                if (listen_address != null) {
                    std.debug.print("error: --listen specified more than once; only one listening socket preopen is supported\n", .{});
                    std.process.exit(1);
                }
                const spec = arg["--listen=".len..];
                listen_address = parseListenAddress(spec) catch {
                    std.debug.print("error: invalid --listen address '{s}'\n", .{spec});
                    std.process.exit(1);
                };
            } else if (std.mem.startsWith(u8, arg, "--heap-size=")) {
                // Reserved for future WASI heap allocation
            } else if (std.mem.eql(u8, arg, "--env") or std.mem.startsWith(u8, arg, "--env=")) {
                const spec = if (std.mem.eql(u8, arg, "--env")) blk: {
                    i += 1;
                    if (i >= run_args.len) {
                        std.debug.print("error: --env requires KEY=VALUE\n", .{});
                        std.process.exit(1);
                    }
                    break :blk run_args[i];
                } else arg["--env=".len..];
                if (std.mem.indexOfScalar(u8, spec, '=') == null) {
                    std.debug.print("error: --env value '{s}' is missing '='\n", .{spec});
                    std.process.exit(1);
                }
                env_flags.append(allocator, spec) catch std.process.exit(1);
            } else if (std.mem.eql(u8, arg, "--map-dir") or std.mem.startsWith(u8, arg, "--map-dir=")) {
                const spec = if (std.mem.eql(u8, arg, "--map-dir")) blk: {
                    i += 1;
                    if (i >= run_args.len) {
                        std.debug.print("error: --map-dir requires HOST::GUEST\n", .{});
                        std.process.exit(1);
                    }
                    break :blk run_args[i];
                } else arg["--map-dir=".len..];
                const md = parseMapDir(spec) catch {
                    std.debug.print("error: --map-dir value '{s}' must be 'HOST::GUEST'\n", .{spec});
                    std.process.exit(1);
                };
                map_dirs.append(allocator, md) catch std.process.exit(1);
            } else if (std.mem.eql(u8, arg, "--")) {
                past_options = true;
            } else {
                std.debug.print("error: unknown option '{s}' — try `wamr help run`\n", .{arg});
                std.process.exit(1);
            }
        } else if (wasm_path == null) {
            wasm_path = arg;
            past_options = true;
        } else {
            try wasm_args.append(allocator, arg);
        }
    }

    const path = wasm_path orelse {
        std.debug.print("error: missing wasm/cwasm file — usage: wamr run <file> [args...]\n", .{});
        std.process.exit(1);
    };

    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const wasm_data = cwd.readFileAlloc(io, path, allocator, @enumFromInt(256 * 1024 * 1024)) catch |err| {
        wamr.utils.read_file.dieReadFileError(path, err);
    };
    defer allocator.free(wasm_data);

    // Detect file type by magic bytes: AOT (\0aot) vs Wasm (\0asm)
    if (wasm_data.len >= 4 and std.mem.readInt(u32, wasm_data[0..4], .little) == wamr.types.aot_magic) {
        if (listen_address != null) {
            std.debug.print("Error: --listen is not supported for AOT modules; rerun without --aot\n", .{});
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
            // For components: prefer explicit --env entries; otherwise inherit
            // the host environment. EnvVar slices borrow from the existing
            // environ_map (lifetime: the entire process).
            var env_list: std.ArrayListUnmanaged(wamr.wasi_cli_adapter.EnvVar) = .empty;
            defer env_list.deinit(allocator);
            if (env_flags.items.len > 0) {
                for (env_flags.items) |kv| {
                    const eq = std.mem.indexOfScalar(u8, kv, '=').?;
                    env_list.append(allocator, .{ .name = kv[0..eq], .value = kv[eq + 1 ..] }) catch {};
                }
            } else {
                var it = init.environ_map.array_hash_map.iterator();
                while (it.next()) |kv| {
                    env_list.append(allocator, .{ .name = kv.key_ptr.*, .value = kv.value_ptr.* }) catch {};
                }
            }
            if (listen_address) |addr| {
                runHttpComponent(wasm_data, allocator, path, wasm_args.items, env_list.items, addr);
                return;
            }
            runComponent(wasm_data, allocator, io, path, wasm_args.items, env_list.items);
            return;
        }
    }

    if (listen_address) |addr| {
        // Core wasm path: --listen registers a TCP listening socket as a
        // socket preopen (kind = .socket, fd ≥ 3) for the guest's WASI
        // preview1 sock_accept. The component-model path above already
        // handled --listen for HTTP servers.
        runWasm(wasm_data, stack_size, path, &wasm_args, env_flags.items, map_dirs.items, addr, allocator, io);
        return;
    }

    // Wasm module (core)
    runWasm(wasm_data, stack_size, path, &wasm_args, env_flags.items, map_dirs.items, null, allocator, io);
}

fn parseMapDir(spec: []const u8) !MapDir {
    const sep = std.mem.indexOf(u8, spec, "::") orelse return error.MissingSeparator;
    const host = spec[0..sep];
    const guest = spec[sep + 2 ..];
    if (host.len == 0 or guest.len == 0) return error.MissingSeparator;
    return .{ .host_path = host, .guest_name = guest };
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
            error.StartTrapped => std.debug.print(
                "Error: component trapped during initialization (a core (start ...) directive — typically `_initialize` — failed; see [component init trap] line above for details)\n",
                .{},
            ),
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
            error.StartTrapped => std.debug.print(
                "Error: component trapped during initialization (see [component init trap] line above)\n",
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
    wasm_path: []const u8,
    wasm_args: *std.ArrayListUnmanaged([]const u8),
    env_flags: []const []const u8,
    map_dirs: []const MapDir,
    listen_address: ?std.Io.net.IpAddress,
    allocator: std.mem.Allocator,
    io: std.Io,
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

    // Build argv for WASI: [wasm_path, wasm_args...]
    var argv_list: std.ArrayListUnmanaged([]const u8) = .empty;
    defer argv_list.deinit(allocator);
    argv_list.append(allocator, wasm_path) catch std.process.exit(1);
    for (wasm_args.items) |a| argv_list.append(allocator, a) catch std.process.exit(1);

    var ctx = wamr.WasiCtx.init(allocator, io) catch |err| {
        std.debug.print("Error: failed to create WASI context: {}\n", .{err});
        std.process.exit(1);
    };
    defer ctx.deinit();

    ctx.setArgs(argv_list.items);
    ctx.setEnv(env_flags);

    for (map_dirs) |md| {
        _ = ctx.openMappedDir(md.host_path, md.guest_name) catch |err| {
            std.debug.print("Error: cannot pre-open '{s}' as '{s}': {}\n", .{ md.host_path, md.guest_name, err });
            std.process.exit(1);
        };
    }

    if (listen_address) |addr| {
        const listen_fd = openListenSocket(addr) catch |err| {
            std.debug.print("Error: --listen bind failed: {}\n", .{err});
            std.process.exit(1);
        };
        _ = ctx.addPreopenSocket(listen_fd) catch |err| {
            std.debug.print("Error: cannot register --listen socket preopen: {}\n", .{err});
            std.process.exit(1);
        };
    }

    env.wasi_ctx = @ptrCast(ctx);

    if (param_count >= 2) {
        env.pushI32(@intCast(wasm_args.items.len + 1)) catch {};
        env.pushI32(0) catch {};
    }

    wamr.interp.executeFunction(env, start_func.index) catch |err| {
        if (ctx.exit_code) |code| std.process.exit(@intCast(code));
        std.debug.print("Error: execution trapped: {}\n", .{err});
        std.process.exit(1);
    };

    if (ctx.exit_code) |code| std.process.exit(@intCast(code));
}

/// Bind a TCP socket to `addr` and put it in the listen state. Used by the
/// core wasm `--listen=` flow to register the listener as a socket
/// preopen (fd ≥ 3) for the guest's WASI preview1 sock_accept. Backlog
/// matches `SOMAXCONN` on modern Linux (128 is the historical floor).
///
/// Implemented with raw `std.os.linux` syscalls because the cross-platform
/// `std.posix.{socket,bind,listen,close}` wrappers were removed in Zig 0.16.0.
/// `--listen` is therefore Linux-only on the core-wasm path; the CLI
/// front-end rejects it elsewhere.
fn openListenSocket(addr: std.Io.net.IpAddress) !std.posix.fd_t {
    if (comptime builtin.os.tag != .linux) return error.UnsupportedPlatform;
    const linux = std.os.linux;

    const family: u32 = switch (addr) {
        .ip4 => linux.AF.INET,
        .ip6 => linux.AF.INET6,
    };
    const sock_type: u32 = linux.SOCK.STREAM | linux.SOCK.CLOEXEC;
    const sock_rc = linux.socket(family, sock_type, linux.IPPROTO.TCP);
    if (linux.errno(sock_rc) != .SUCCESS) return error.SocketFailed;
    const fd: std.posix.fd_t = @intCast(@as(isize, @bitCast(sock_rc)));
    errdefer _ = linux.close(fd);

    // SO_REUSEADDR: lets the host re-bind quickly after a restart while
    // a previous accepted client lingers in TIME_WAIT.
    const one: c_int = 1;
    const so_rc = linux.setsockopt(
        fd,
        linux.SOL.SOCKET,
        linux.SO.REUSEADDR,
        @ptrCast(&one),
        @sizeOf(c_int),
    );
    if (linux.errno(so_rc) != .SUCCESS) return error.SetSockOptFailed;

    switch (addr) {
        .ip4 => |v4| {
            const sa = linux.sockaddr.in{
                .port = std.mem.nativeToBig(u16, v4.port),
                .addr = @bitCast(v4.bytes),
            };
            const b = linux.bind(fd, @ptrCast(&sa), @sizeOf(@TypeOf(sa)));
            if (linux.errno(b) != .SUCCESS) return error.BindFailed;
        },
        .ip6 => |v6| {
            const sa = linux.sockaddr.in6{
                .port = std.mem.nativeToBig(u16, v6.port),
                .flowinfo = v6.flow,
                .addr = v6.bytes,
                .scope_id = v6.interface.index,
            };
            const b = linux.bind(fd, @ptrCast(&sa), @sizeOf(@TypeOf(sa)));
            if (linux.errno(b) != .SUCCESS) return error.BindFailed;
        },
    }

    const lr = linux.listen(fd, 128);
    if (linux.errno(lr) != .SUCCESS) return error.ListenFailed;
    return fd;
}

fn writeStdout(io: std.Io, text: []const u8) void {
    var stdout_file = std.Io.File.stdout();
    stdout_file.writeStreamingAll(io, text) catch {};
}

const top_usage =
    \\wamr - WebAssembly Micro Runtime
    \\
    \\Usage: wamr <subcommand> [args...]
    \\
    \\Subcommands:
    \\  run       Run a .wasm or .cwasm file
    \\  version   Print version and exit
    \\  help      Print this help; `wamr help <subcommand>` for details
    \\
;

const run_usage =
    \\Usage: wamr run [options] <file.wasm|file.cwasm> [args...]
    \\
    \\Options:
    \\  --stack-size=<bytes>     Stack size for the interpreter (default: 65536)
    \\  --heap-size=<bytes>      Reserved (currently ignored)
    \\  --listen=<ip:port>       For components: serve WASI HTTP on the address.
    \\                           For core wasm: bind a TCP listening socket and
    \\                           expose it to the guest as the next preopen fd
    \\                           (≥ 3) for `sock_accept` (single use only).
    \\  --env KEY=VALUE          Set a WASI environment variable (repeatable)
    \\  --map-dir HOST::GUEST    Pre-open `HOST` host directory as `GUEST`
    \\                           inside the guest WASI sandbox (repeatable)
    \\  -h, --help               Show this help
    \\
;

const version_usage =
    \\Usage: wamr version
    \\
    \\Print the wamr version and exit.
    \\
;

const help_usage =
    \\Usage: wamr help [subcommand]
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
        std.debug.print("error: unknown subcommand '{s}' — try `wamr help`\n", .{args[0]});
        std.process.exit(1);
    };
    writeStdout(io, switch (sub) {
        .run => run_usage,
        .version => version_usage,
        .help => help_usage,
    });
}

test "subcommand parsing" {
    try std.testing.expectEqual(@as(?Subcommand, .run), parseSubcommand("run"));
    try std.testing.expectEqual(@as(?Subcommand, .version), parseSubcommand("version"));
    try std.testing.expectEqual(@as(?Subcommand, .help), parseSubcommand("help"));
    try std.testing.expectEqual(@as(?Subcommand, null), parseSubcommand("--version"));
    try std.testing.expectEqual(@as(?Subcommand, null), parseSubcommand("foo.wasm"));
    try std.testing.expectEqual(@as(?Subcommand, null), parseSubcommand(""));
}

test "parseMapDir splits HOST::GUEST" {
    const md = try parseMapDir("/tmp/host::/sandbox");
    try std.testing.expectEqualStrings("/tmp/host", md.host_path);
    try std.testing.expectEqualStrings("/sandbox", md.guest_name);
}

test "parseMapDir rejects missing separator" {
    try std.testing.expectError(error.MissingSeparator, parseMapDir("/tmp/host"));
    try std.testing.expectError(error.MissingSeparator, parseMapDir("/tmp/host:/sandbox"));
    try std.testing.expectError(error.MissingSeparator, parseMapDir("::guest"));
    try std.testing.expectError(error.MissingSeparator, parseMapDir("host::"));
}

test "version line second whitespace token is the version (parsable by wasi-testsuite adapter)" {
    // The upstream `wasi-testsuite` Python adapter calls `wamr version` and
    // parses with `result.stdout.splitlines()[0].split(" ")[1]`. Mirror that
    // logic exactly here so any change to the version output that breaks
    // adapter parsing fails this test.
    const line = "wamr " ++ wamr.version.string ++ "\n";
    const newline = std.mem.indexOfScalar(u8, line, '\n').?;
    const first_line = line[0..newline];
    var it = std.mem.splitScalar(u8, first_line, ' ');
    _ = it.next().?; // "wamr"
    const version_token = it.next().?;
    try std.testing.expectEqualStrings(wamr.version.string, version_token);
}
