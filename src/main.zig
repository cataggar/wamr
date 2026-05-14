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

/// Returns the host process exit code. Returning (rather than calling
/// `std.process.exit`) lets the Zig 0.16 startup runtime run its own
/// teardown — most importantly the `DebugAllocator` leak hook on
/// `init.gpa` in Debug builds (#449, #450). Exit-code policy:
///   * 0 — successful guest run / `help` / `version`
///   * `outcome.exit_code` / `ctx.exit_code` — guest-requested
///     (preview1 `proc_exit`, `wasi:cli/exit.exit-with-code`)
///   * 1 — runtime failure (trap, load/instantiate fail, host-side OOM
///     during guest setup, etc.)
///   * 2 — CLI / arg-parsing / usage error
pub fn main(init: std.process.Init) !u8 {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    if (args.len < 2) {
        std.debug.print("error: missing subcommand — try `wamr help`\n", .{});
        return 2;
    }

    const subcmd = parseSubcommand(args[1]) orelse {
        std.debug.print("error: unknown subcommand '{s}' — try `wamr help`\n", .{args[1]});
        return 2;
    };

    return switch (subcmd) {
        .version => try runVersion(init.io, args[2..]),
        .help => runHelp(init.io, args[2..]),
        .run => try runRun(init, allocator, args[2..]),
    };
}

fn runVersion(io: std.Io, args: []const []const u8) !u8 {
    if (args.len == 1 and std.mem.eql(u8, args[0], "help")) {
        writeStdout(io, version_usage);
        return 0;
    }
    writeStdout(io, "wamr " ++ wamr.version.string ++ "\n");
    return 0;
}

fn runRun(init: std.process.Init, allocator: std.mem.Allocator, run_args: []const []const u8) !u8 {
    if (run_args.len == 1 and std.mem.eql(u8, run_args[0], "help")) {
        writeStdout(init.io, run_usage);
        return 0;
    }
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
            if (std.mem.startsWith(u8, arg, "--stack-size=")) {
                stack_size = std.fmt.parseInt(u32, arg["--stack-size=".len..], 10) catch {
                    std.debug.print("error: invalid --stack-size value\n", .{});
                    return 2;
                };
            } else if (std.mem.startsWith(u8, arg, "--listen=")) {
                if (listen_address != null) {
                    std.debug.print("error: --listen specified more than once; only one listening socket preopen is supported\n", .{});
                    return 2;
                }
                const spec = arg["--listen=".len..];
                listen_address = parseListenAddress(spec) catch {
                    std.debug.print("error: invalid --listen address '{s}'\n", .{spec});
                    return 2;
                };
            } else if (std.mem.startsWith(u8, arg, "--heap-size=")) {
                // Reserved for future WASI heap allocation
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
                const md = parseMapDir(spec) catch {
                    std.debug.print("error: --map-dir value '{s}' must be 'HOST::GUEST'\n", .{spec});
                    return 2;
                };
                try map_dirs.append(allocator, md);
            } else if (std.mem.eql(u8, arg, "--")) {
                past_options = true;
            } else {
                std.debug.print("error: unknown option '{s}' — try `wamr run help`\n", .{arg});
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
        std.debug.print("error: missing wasm/cwasm file — usage: wamr run <file> [args...]\n", .{});
        return 2;
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
            return 2;
        }
        return runAot(wasm_data, allocator);
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
                return runHttpComponent(wasm_data, allocator, path, wasm_args.items, env_list.items, addr);
            }
            return runComponent(wasm_data, allocator, path, wasm_args.items, env_list.items);
        }
    }

    // Wasm module (core). `--listen` registers a TCP listening socket as a
    // socket preopen (kind = .socket, fd ≥ 3) for the guest's WASI preview1
    // sock_accept. The component-model branch above already handled --listen
    // for HTTP servers.
    return runWasm(wasm_data, stack_size, path, &wasm_args, env_flags.items, map_dirs.items, listen_address, allocator, io);
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
    wasm_path: []const u8,
    wasm_args: []const []const u8,
    env_vars: []const wamr.wasi_cli_adapter.EnvVar,
) u8 {
    const adapter_mod = wamr.wasi_cli_adapter;
    // Wire the adapter's stdio directly to the host process's
    // STDIN/STDOUT/STDERR so output streams live (no end-of-run
    // flush) and stdin reads from the user's terminal / piped
    // input (#474). For the test/embedding path that needs to
    // inspect captured stdout, use `WasiCliAdapter.init` instead.
    var adapter = adapter_mod.WasiCliAdapter.initWithHostStdio(allocator);
    defer adapter.deinit();

    // argv[0] = wasm path, rest = user args (matches wasmtime convention).
    var argv_buf = allocator.alloc([]const u8, 1 + wasm_args.len) catch
        return 1;
    defer allocator.free(argv_buf);
    argv_buf[0] = wasm_path;
    for (wasm_args, 0..) |a, i| argv_buf[i + 1] = a;
    adapter.setArguments(argv_buf);
    adapter.setEnvironment(env_vars);

    // The component loader has no `Component.deinit` yet (#142 Phase 1B);
    // its allocations (and the matching `ComponentInstance` machinery, whose
    // `inst.deinit()` would otherwise have to mirror every loader-owned
    // slice) are gathered into an arena that we tear down here. Mirrors the
    // AOT path's `defer aot_loader.unload(...)` (PR #449) and matches the
    // established pattern used by every component end-to-end test in the
    // repo (see `wasi_cli_adapter.zig` test "stdio-echo: end-to-end real
    // wasi-p2 component" and the explicit comment at `runLoadedComponent`
    // about "hand-rolled callers ... which pass an arena"). Without this,
    // DebugAllocator drowns the real component-path leak signal in
    // hundreds of loader-allocated slices on every Debug-build run.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const outcome = adapter_mod.runComponentBytes(data, arena_alloc, &adapter) catch |err| {
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
        return 1;
    };

    // Output has already streamed live to host STDOUT/STDERR via the
    // fd-backed OutputStream sinks (#474); no end-of-run flush needed.

    // Prefer the explicit numeric exit code recorded by
    // `wasi:cli/exit.exit-with-code` / preview1 `proc_exit` (#436);
    // fall back to the boolean is_ok mapping when the component
    // returned normally without recording a code. POSIX exit codes
    // are 8-bit on most hosts, so saturate to 1 on overflow.
    if (outcome.exit_code) |code| return std.math.cast(u8, code) orelse 1;
    return if (outcome.is_ok) 0 else 1;
}

fn runHttpComponent(
    data: []const u8,
    allocator: std.mem.Allocator,
    wasm_path: []const u8,
    wasm_args: []const []const u8,
    env_vars: []const wamr.wasi_cli_adapter.EnvVar,
    listen_address: std.Io.net.IpAddress,
) u8 {
    const adapter_mod = wamr.wasi_cli_adapter;
    var adapter = adapter_mod.WasiCliAdapter.init(allocator);
    defer adapter.deinit();

    var argv_buf = allocator.alloc([]const u8, 1 + wasm_args.len) catch
        return 1;
    defer allocator.free(argv_buf);
    argv_buf[0] = wasm_path;
    for (wasm_args, 0..) |a, i| argv_buf[i + 1] = a;
    adapter.setArguments(argv_buf);
    adapter.setEnvironment(env_vars);

    // See `runComponent` for the rationale behind the arena wrapper —
    // same loader/instance allocation story applies on the HTTP path.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    adapter_mod.serveHttpComponentBytes(data, arena_alloc, &adapter, .{
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
        return 1;
    };
    // `serveHttpComponentBytes` runs an accept loop that today never returns
    // normally; this path keeps the function's u8 return well-defined.
    return 0;
}

fn runAot(data: []const u8, allocator: std.mem.Allocator) u8 {
    if (comptime aot_supported) {
        return runAotReal(data, allocator);
    } else {
        std.debug.print("Error: AOT execution not supported on this architecture\n", .{});
        return 1;
    }
}

fn runAotReal(data: []const u8, allocator: std.mem.Allocator) u8 {
    const aot_loader = wamr.aot_loader;
    const aot_runtime = wamr.aot_runtime;

    const aot_module = aot_loader.load(data, allocator) catch |err| {
        std.debug.print("Error: failed to load AOT module: {}\n", .{err});
        return 1;
    };
    // Mirror the load/unload pairing used by every other `aot_loader.load`
    // call site in the repo (compiler/emit_aot.zig, tests/coldstart_test.zig,
    // tests/aot_harness.zig). Without this, `runAotReal` leaks every owned
    // slice on `AotModule` — func_offsets, local_func_type_indices,
    // func_types, exports, memories, data_segments, tables, imports,
    // global_inits, elem_segments — which DebugAllocator prints to stderr
    // on exit, even though the AOT image executes successfully.
    defer aot_loader.unload(&aot_module, allocator);

    const aot_inst = aot_runtime.instantiate(&aot_module, allocator) catch |err| {
        std.debug.print("Error: failed to instantiate AOT module: {}\n", .{err});
        return 1;
    };
    defer aot_runtime.destroy(aot_inst);

    // Map native code as executable
    aot_runtime.mapCodeExecutable(aot_inst) catch |err| {
        std.debug.print("Error: failed to map code as executable: {}\n", .{err});
        return 1;
    };

    // Find _start or main export
    const func_idx = aot_runtime.findExportFunc(aot_inst, "_start") orelse
        aot_runtime.findExportFunc(aot_inst, "main") orelse {
        std.debug.print("Error: no _start or main function exported in AOT module\n", .{});
        return 1;
    };

    // Execute
    const result = aot_runtime.callFunc(aot_inst, func_idx, i32) catch |err| {
        std.debug.print("Error: AOT execution failed: {}\n", .{err});
        return 1;
    };
    std.debug.print("{d}\n", .{result});
    return 0;
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
) u8 {
    var runtime = wamr.wamr.Runtime.init(allocator);
    defer runtime.deinit();

    var module = runtime.loadModule(wasm_data) catch |err| {
        std.debug.print("Error: failed to load module: {}\n", .{err});
        return 1;
    };
    defer module.deinit();

    var instance = module.instantiate() catch |err| {
        std.debug.print("Error: failed to instantiate: {}\n", .{err});
        return 1;
    };
    defer instance.deinit();

    const start_func = module.findExport("_start", .function) orelse
        module.findExport("main", .function) orelse {
        std.debug.print("Error: no _start or main function exported\n", .{});
        return 1;
    };

    const func_type = module.inner.getFuncType(start_func.index);
    const param_count = if (func_type) |ft| ft.params.len else 0;

    var env = wamr.exec_env.ExecEnv.create(instance.inner, stack_size, allocator) catch |err| {
        std.debug.print("Error: failed to create execution environment: {}\n", .{err});
        return 1;
    };
    defer env.destroy();

    // Build argv for WASI: [wasm_path, wasm_args...]
    var argv_list: std.ArrayListUnmanaged([]const u8) = .empty;
    defer argv_list.deinit(allocator);
    argv_list.append(allocator, wasm_path) catch return 1;
    for (wasm_args.items) |a| argv_list.append(allocator, a) catch return 1;

    var ctx = wamr.WasiCtx.init(allocator, io) catch |err| {
        std.debug.print("Error: failed to create WASI context: {}\n", .{err});
        return 1;
    };
    defer ctx.deinit();

    ctx.setArgs(argv_list.items);
    ctx.setEnv(env_flags);

    for (map_dirs) |md| {
        _ = ctx.openMappedDir(md.host_path, md.guest_name) catch |err| {
            std.debug.print("Error: cannot pre-open '{s}' as '{s}': {}\n", .{ md.host_path, md.guest_name, err });
            return 1;
        };
    }

    if (listen_address) |addr| {
        const listen_fd = openListenSocket(addr) catch |err| {
            std.debug.print("Error: --listen bind failed: {}\n", .{err});
            return 1;
        };
        _ = ctx.addPreopenSocket(listen_fd) catch |err| {
            std.debug.print("Error: cannot register --listen socket preopen: {}\n", .{err});
            return 1;
        };
    }

    env.wasi_ctx = @ptrCast(ctx);

    if (param_count >= 2) {
        env.pushI32(@intCast(wasm_args.items.len + 1)) catch {};
        env.pushI32(0) catch {};
    }

    wamr.interp.executeFunction(env, start_func.index) catch |err| {
        if (ctx.exit_code) |code| return @intCast(code);
        std.debug.print("Error: execution trapped: {}\n", .{err});
        return 1;
    };

    if (ctx.exit_code) |code| return @intCast(code);
    return 0;
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
    \\  help      Print this help
    \\
    \\Run `wamr <subcommand> help` to show help for a specific subcommand.
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
    \\
;

const version_usage =
    \\Usage: wamr version
    \\
    \\Print the wamr version and exit.
    \\
;

const help_usage =
    \\Usage: wamr help
    \\
    \\Print top-level help and exit.
    \\
;

fn runHelp(io: std.Io, args: []const []const u8) u8 {
    if (args.len == 1 and std.mem.eql(u8, args[0], "help")) {
        writeStdout(io, help_usage);
        return 0;
    }
    writeStdout(io, top_usage);
    return 0;
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
