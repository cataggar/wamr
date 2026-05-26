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

    // Process-global AOT debug toggle. See `core_backend.setDebugAotEnabled`
    // for what it controls and #644 for context.
    if (init.environ_map.get("WAMR_AOT_DEBUG")) |v| {
        const on = !(v.len == 0 or std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "false"));
        wamr.component_core_backend.setDebugAotEnabled(on);
    }

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
    // `--allow-net=CIDR` (repeatable): seed the component-side WASI
    // sockets allow-list with the given CIDR blocks. Without at least
    // one entry the default-deny policy rejects every bind/connect
    // with `error-code::access-denied` — the wasi-testsuite sockets
    // fixtures need 127.0.0.0/8 (and ::1/128) to exercise the
    // loopback-bind paths. (#520 wave 2)
    var allow_net: std.ArrayListUnmanaged([]const u8) = .empty;
    defer allow_net.deinit(allocator);
    // `--config-store=<path>` (#583 B6): path to a JSON file whose flat
    // `{ "key": "value", ... }` object becomes the `wasi:config/store`
    // layer-1 backing. Combined with the `WAMR_CONFIG_*` env vars (layer
    // 2) in `runComponent`. Null when not provided.
    var config_store_path: ?[]const u8 = null;
    // `--keyvalue-store=<path>` (#583 B4 follow-up): path to a JSON
    // file whose `{"bucket-name": {"key": "<base64-value>", ...}, ...}`
    // object backs the `wasi:keyvalue` adapter. Reads on startup,
    // rewrites synchronously after every mutation. Null ⇒ in-memory
    // only (default, behaviour-compatible with PR #601).
    var keyvalue_store_path: ?[]const u8 = null;
    // `--precompiled-dir=<path>` (#642): explicit path to a
    // `wamrc compile-component` output bundle (manifest.json + per-core
    // .cwasm artifacts). When unset, `runRun` probes for a sibling
    // `<input>.cwasm.d/manifest.json` next to the component file and
    // uses it on a best-effort basis (mismatch → warning + interpreter
    // fallback). When set, the bundle is mandatory: any error is fatal.
    var precompiled_dir: ?[]const u8 = null;
    var listen_address: ?std.Io.net.IpAddress = null;
    // `--tls-cert=<path>` + `--tls-key=<path>` (or the combined
    // `--tls-pem=<path>`) (#583 follow-up to #595): when both cert
    // and key (or just the combined PEM) are provided alongside
    // `--listen`, the HTTP service is *prepared* to terminate TLS
    // on each accepted connection. Today the actual handshake call
    // is upstream-blocked on Zig 0.16 std not shipping a server-side
    // TLS API (only `std.crypto.tls.Client`). The cert + key are
    // still loaded + validated at startup so the CLI surface is
    // stable; once upstream lands `std.crypto.tls.Server` (tracked
    // by cataggar/wamr#609) the handshake goes live without
    // breaking the existing flag shape. Null when not provided.
    var tls_cert_path: ?[]const u8 = null;
    var tls_key_path: ?[]const u8 = null;
    var tls_pem_path: ?[]const u8 = null;
    // `--log-level=<name>` (#583 B5). Sets the host-side
    // `wasi:logging/logging.log` severity filter. The CLI flag wins
    // over the `WAMR_LOG_LEVEL` env var, which is consulted at the
    // run-component step when no CLI flag is set. Default `null` →
    // no override (adapter keeps its built-in default of `.trace`,
    // which admits every level).
    var log_level: ?wamr.wasi_cli_adapter.WasiLogLevel = null;
    var past_options = false;

    var i: usize = 0;
    while (i < run_args.len) : (i += 1) {
        const arg = run_args[i];
        if (!past_options and arg.len > 0 and arg[0] == '-') {
            if (std.mem.startsWith(u8, arg, "--stack-size=")) {
                // #644: the interpreter stack-size knob is now a no-op
                // since the CLI is AOT-only. Accept the flag silently
                // so existing invocations keep working; warn once on
                // stderr so out-of-tree callers can adapt.
                std.debug.print("warning: --stack-size is ignored under the AOT-only CLI (issue #644)\n", .{});
            } else if (std.mem.eql(u8, arg, "--listen") or std.mem.startsWith(u8, arg, "--listen=")) {
                if (listen_address != null) {
                    std.debug.print("error: --listen specified more than once; only one listening socket preopen is supported\n", .{});
                    return 2;
                }
                if (std.mem.eql(u8, arg, "--listen")) {
                    // Bare `--listen` -> 127.0.0.1:0 (kernel-assigned ephemeral
                    // port). The actual bound address is printed to stdout
                    // after bind so test drivers can scrape it.
                    listen_address = std.Io.net.IpAddress.parse("127.0.0.1", 0) catch unreachable;
                } else {
                    const spec = arg["--listen=".len..];
                    listen_address = parseListenAddress(spec) catch {
                        std.debug.print("error: invalid --listen address '{s}'\n", .{spec});
                        return 2;
                    };
                }
            } else if (std.mem.startsWith(u8, arg, "--heap-size=")) {
                // Reserved for future WASI heap allocation
            } else if (std.mem.eql(u8, arg, "--tls-cert") or std.mem.startsWith(u8, arg, "--tls-cert=")) {
                if (tls_cert_path != null) {
                    std.debug.print("error: --tls-cert specified more than once\n", .{});
                    return 2;
                }
                const spec = if (std.mem.eql(u8, arg, "--tls-cert")) blk: {
                    i += 1;
                    if (i >= run_args.len) {
                        std.debug.print("error: --tls-cert requires a path to a PEM file\n", .{});
                        return 2;
                    }
                    break :blk run_args[i];
                } else arg["--tls-cert=".len..];
                if (spec.len == 0) {
                    std.debug.print("error: --tls-cert path is empty\n", .{});
                    return 2;
                }
                tls_cert_path = spec;
            } else if (std.mem.eql(u8, arg, "--tls-key") or std.mem.startsWith(u8, arg, "--tls-key=")) {
                if (tls_key_path != null) {
                    std.debug.print("error: --tls-key specified more than once\n", .{});
                    return 2;
                }
                const spec = if (std.mem.eql(u8, arg, "--tls-key")) blk: {
                    i += 1;
                    if (i >= run_args.len) {
                        std.debug.print("error: --tls-key requires a path to a PEM file\n", .{});
                        return 2;
                    }
                    break :blk run_args[i];
                } else arg["--tls-key=".len..];
                if (spec.len == 0) {
                    std.debug.print("error: --tls-key path is empty\n", .{});
                    return 2;
                }
                tls_key_path = spec;
            } else if (std.mem.eql(u8, arg, "--tls-pem") or std.mem.startsWith(u8, arg, "--tls-pem=")) {
                if (tls_pem_path != null) {
                    std.debug.print("error: --tls-pem specified more than once\n", .{});
                    return 2;
                }
                const spec = if (std.mem.eql(u8, arg, "--tls-pem")) blk: {
                    i += 1;
                    if (i >= run_args.len) {
                        std.debug.print("error: --tls-pem requires a path to a combined PEM file\n", .{});
                        return 2;
                    }
                    break :blk run_args[i];
                } else arg["--tls-pem=".len..];
                if (spec.len == 0) {
                    std.debug.print("error: --tls-pem path is empty\n", .{});
                    return 2;
                }
                tls_pem_path = spec;
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
            } else if (std.mem.eql(u8, arg, "--allow-net") or std.mem.startsWith(u8, arg, "--allow-net=")) {
                const spec = if (std.mem.eql(u8, arg, "--allow-net")) blk: {
                    i += 1;
                    if (i >= run_args.len) {
                        std.debug.print("error: --allow-net requires CIDR (e.g. 127.0.0.0/8)\n", .{});
                        return 2;
                    }
                    break :blk run_args[i];
                } else arg["--allow-net=".len..];
                try allow_net.append(allocator, spec);
            } else if (std.mem.eql(u8, arg, "--log-level") or std.mem.startsWith(u8, arg, "--log-level=")) {
                const spec = if (std.mem.eql(u8, arg, "--log-level")) blk: {
                    i += 1;
                    if (i >= run_args.len) {
                        std.debug.print("error: --log-level requires <trace|debug|info|warn|error|critical>\n", .{});
                        return 2;
                    }
                    break :blk run_args[i];
                } else arg["--log-level=".len..];
                log_level = wamr.wasi_cli_adapter.WasiLogLevel.fromString(spec) orelse {
                    std.debug.print(
                        "error: --log-level value '{s}' is not one of trace|debug|info|warn|error|critical\n",
                        .{spec},
                    );
                    return 2;
                };
            } else if (std.mem.eql(u8, arg, "--config-store") or std.mem.startsWith(u8, arg, "--config-store=")) {
                if (config_store_path != null) {
                    std.debug.print("error: --config-store specified more than once\n", .{});
                    return 2;
                }
                const spec = if (std.mem.eql(u8, arg, "--config-store")) blk: {
                    i += 1;
                    if (i >= run_args.len) {
                        std.debug.print("error: --config-store requires a path to a JSON file\n", .{});
                        return 2;
                    }
                    break :blk run_args[i];
                } else arg["--config-store=".len..];
                if (spec.len == 0) {
                    std.debug.print("error: --config-store path is empty\n", .{});
                    return 2;
                }
                config_store_path = spec;
            } else if (std.mem.eql(u8, arg, "--keyvalue-store") or std.mem.startsWith(u8, arg, "--keyvalue-store=")) {
                if (keyvalue_store_path != null) {
                    std.debug.print("error: --keyvalue-store specified more than once\n", .{});
                    return 2;
                }
                const spec = if (std.mem.eql(u8, arg, "--keyvalue-store")) blk: {
                    i += 1;
                    if (i >= run_args.len) {
                        std.debug.print("error: --keyvalue-store requires a path to a JSON file\n", .{});
                        return 2;
                    }
                    break :blk run_args[i];
                } else arg["--keyvalue-store=".len..];
                if (spec.len == 0) {
                    std.debug.print("error: --keyvalue-store path is empty\n", .{});
                    return 2;
                }
                keyvalue_store_path = spec;
            } else if (std.mem.eql(u8, arg, "--precompiled-dir") or std.mem.startsWith(u8, arg, "--precompiled-dir=")) {
                // `--precompiled-dir <path>` (#642): explicit path to a
                // `wamrc compile-component` output directory containing
                // `manifest.json` + `module<N>.cwasm`. Cores in the bundle
                // are loaded as AOT; cores missing from the manifest fall
                // back to the interpreter. A mismatched / stale manifest
                // is a hard error (vs. the silent fallback used by the
                // sibling auto-detect path below).
                if (precompiled_dir != null) {
                    std.debug.print("error: --precompiled-dir specified more than once\n", .{});
                    return 2;
                }
                const spec = if (std.mem.eql(u8, arg, "--precompiled-dir")) blk: {
                    i += 1;
                    if (i >= run_args.len) {
                        std.debug.print("error: --precompiled-dir requires a path to a wamrc compile-component output dir\n", .{});
                        return 2;
                    }
                    break :blk run_args[i];
                } else arg["--precompiled-dir=".len..];
                if (spec.len == 0) {
                    std.debug.print("error: --precompiled-dir path is empty\n", .{});
                    return 2;
                }
                precompiled_dir = spec;
            } else if (std.mem.eql(u8, arg, "--trace-aot-wasi")) {
                wamr.aot_host_bridge.trace_enabled = true;
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

    // ── TLS flag validation (#583 follow-up / #609) ──────────────────────
    // Disallowed combinations:
    //   * `--tls-cert` without `--tls-key` (or vice versa) — the pair is
    //     load-bearing together.
    //   * `--tls-pem` combined with either `--tls-cert` or `--tls-key` —
    //     pick one of the two input shapes.
    //   * Any of the TLS flags without `--listen` — TLS termination only
    //     makes sense on the HTTP service path.
    if (tls_pem_path != null and (tls_cert_path != null or tls_key_path != null)) {
        std.debug.print("error: --tls-pem is mutually exclusive with --tls-cert / --tls-key\n", .{});
        return 2;
    }
    if ((tls_cert_path == null) != (tls_key_path == null)) {
        std.debug.print("error: --tls-cert and --tls-key must be specified together\n", .{});
        return 2;
    }
    const tls_requested = tls_cert_path != null or tls_pem_path != null;
    if (tls_requested and listen_address == null) {
        std.debug.print("error: --tls-cert / --tls-key / --tls-pem require --listen\n", .{});
        return 2;
    }

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
            // `WAMR_LOG_LEVEL` is consulted only when `--log-level` was
            // not passed on the command line; the CLI flag wins. Mirrors
            // RUST_LOG / similar conventions. (#583 B5)
            const effective_log_level: ?wamr.wasi_cli_adapter.WasiLogLevel = log_level orelse blk: {
                const raw = init.environ_map.get("WAMR_LOG_LEVEL") orelse break :blk null;
                if (raw.len == 0) break :blk null;
                const parsed = wamr.wasi_cli_adapter.WasiLogLevel.fromString(raw) orelse {
                    std.debug.print(
                        "warning: WAMR_LOG_LEVEL='{s}' is not one of trace|debug|info|warn|error|critical; ignoring\n",
                        .{raw},
                    );
                    break :blk null;
                };
                break :blk parsed;
            };
            // #583 B6: assemble `wasi:config/store` from layered sources.
            // The arena owns the JSON-parsed strings and lower-cased env
            // keys; the resulting slice's lifetime equals the arena's
            // (we hand it to runComponent and tear down on return).
            var cfg_arena = std.heap.ArenaAllocator.init(allocator);
            defer cfg_arena.deinit();
            const cfg_entries = loadComponentConfigStore(
                cfg_arena.allocator(),
                init.environ_map,
                config_store_path,
            ) catch |err| {
                std.debug.print("Error: --config-store '{s}': {}\n", .{ config_store_path orelse "", err });
                return 2;
            };
            // #642 / #644: resolve the AOT precompiled-cores bundle. The
            // `wamr` CLI is AOT-only — every component core must be
            // available as a precompiled `.cwasm`, or instantiation
            // fails hard with `error.AotImportUnresolvable` (the
            // interp fallback is disabled at the CLI surface; library
            // callers still get the legacy silent demotion by leaving
            // `Options.aot_only` unset).
            //
            //  * `--precompiled-dir <path>` (explicit): any error
            //    opening / validating the manifest is fatal so the user
            //    knows the AOT path isn't being taken.
            //  * sibling `<input>.cwasm.d/manifest.json` (auto-detect):
            //    used when present and valid; mismatch / stale → one
            //    `warning: ...` line on stderr and we fall through to
            //    the in-memory auto-precompile path below.
            //  * neither found → AOT-compile each core to memory on
            //    the fly. Single-run cost; users who care about
            //    cold-start should `wamrc compile-component` once and
            //    pass `--precompiled-dir` or use a sibling
            //    `<input>.cwasm.d`.
            //
            // The `LoadedManifest`'s lifetime must outlive
            // `runComponent` / `runHttpComponent` because the mmapped
            // `.cwasm` buffers it owns are borrowed by the
            // `PrecompiledCore` slice handed to the instance loader.
            // The in-memory auto-precompile path owns its `.cwasm`
            // slices on `allocator`; the defer below frees them.
            var loaded_manifest: ?wamr.component_aot.LoadedManifest = null;
            defer if (loaded_manifest) |*lm| lm.deinit();

            var auto_cores: std.ArrayListUnmanaged(wamr.component_core_backend.PrecompiledCore) = .empty;
            defer {
                for (auto_cores.items) |pc| allocator.free(pc.cwasm_bytes);
                auto_cores.deinit(allocator);
            }

            if (precompiled_dir) |dir| {
                loaded_manifest = wamr.component_aot.loadManifest(allocator, dir, wasm_data) catch |err| {
                    std.debug.print("Error: --precompiled-dir '{s}': {s}\n", .{ dir, loadManifestErrorMessage(err) });
                    return 2;
                };
                const n = loaded_manifest.?.precompiledCores().len;
                std.debug.print("wamr: loaded AOT bundle from {s} ({d} core{s} precompiled)\n", .{ dir, n, if (n == 1) @as([]const u8, "") else "s" });
            } else {
                // Auto-probe `<input>.cwasm.d/manifest.json`. Don't
                // fail the run if the bundle is absent / stale — fall
                // through to the in-memory auto-precompile path
                // below.
                const sibling = wamr.component_aot.defaultPrecompiledDirFor(allocator, path) catch return 1;
                defer allocator.free(sibling);
                const probe = std.fs.path.join(allocator, &.{ sibling, "manifest.json" }) catch return 1;
                defer allocator.free(probe);
                const io_probe = init.io;
                const cwd_probe = std.Io.Dir.cwd();
                const exists = blk: {
                    var f = cwd_probe.openFile(io_probe, probe, .{}) catch break :blk false;
                    f.close(io_probe);
                    break :blk true;
                };
                if (exists) {
                    if (wamr.component_aot.loadManifest(allocator, sibling, wasm_data)) |lm| {
                        loaded_manifest = lm;
                        const n = lm.precompiledCores().len;
                        std.debug.print("wamr: loaded AOT bundle from {s} ({d} core{s} precompiled)\n", .{ sibling, n, if (n == 1) @as([]const u8, "") else "s" });
                    } else |err| {
                        std.debug.print("warning: ignoring AOT bundle at {s}: {s}\n", .{ sibling, loadManifestErrorMessage(err) });
                    }
                }

                // #644: no on-disk bundle → AOT-compile every core to
                // memory now. The cwasm slices live on `allocator`
                // (freed by the outer defer) so they outlive both the
                // parser arena and the `runComponent` call.
                if (loaded_manifest == null) {
                    var load_arena = std.heap.ArenaAllocator.init(allocator);
                    defer load_arena.deinit();
                    const parsed = wamr.component_loader.load(wasm_data, load_arena.allocator()) catch |err| {
                        std.debug.print("Error: failed to parse component for auto-precompile: {s}\n", .{@errorName(err)});
                        return 1;
                    };
                    // Recursively collect every core module from the
                    // top level *and* every nested sub-component. The
                    // top level of a `wabt component compose -d`
                    // composed component has zero cores of its own —
                    // the actual command + adder cores live inside
                    // its two nested sub-components (#662 phase D).
                    // Each entry pairs the core's raw module bytes
                    // (a stable cross-parse identity) with its
                    // *local* `module_idx` within the component that
                    // contains it; `Options.findPrecompiled` matches
                    // on the bytes-slice so sub-component
                    // instantiations don't collide on shared local
                    // indices.
                    const CoreRef = struct {
                        data: []const u8,
                        local_idx: u32,
                    };
                    var all_cores: std.ArrayListUnmanaged(CoreRef) = .empty;
                    defer all_cores.deinit(load_arena.allocator());
                    const Walker = struct {
                        fn walk(
                            comp: *const wamr.component_types.Component,
                            list: *std.ArrayListUnmanaged(CoreRef),
                            arena: std.mem.Allocator,
                        ) !void {
                            for (comp.core_modules, 0..) |cm, mi| {
                                try list.append(arena, .{ .data = cm.data, .local_idx = @intCast(mi) });
                            }
                            for (comp.components) |child| {
                                try walk(child, list, arena);
                            }
                        }
                    };
                    Walker.walk(&parsed, &all_cores, load_arena.allocator()) catch return 1;
                    const ncore = all_cores.items.len;
                    if (ncore == 0) {
                        std.debug.print("Error: component contains no core modules\n", .{});
                        return 1;
                    }
                    std.debug.print(
                        "wamr: auto-precompiling {d} core{s} in memory (no on-disk bundle; pass --precompiled-dir or run `wamrc compile-component` to persist)\n",
                        .{ ncore, if (ncore == 1) @as([]const u8, "") else "s" },
                    );
                    auto_cores.ensureTotalCapacity(allocator, ncore) catch return 1;
                    for (all_cores.items, 0..) |ref, idx| {
                        const cwasm = wamr.component_aot.compileCoreWasm(allocator, ref.data, .{}) catch |err| {
                            std.debug.print(
                                "Error: auto-precompile failed for core module {d}: {s} (issue #644)\n",
                                .{ idx, @errorName(err) },
                            );
                            return 1;
                        };
                        auto_cores.appendAssumeCapacity(.{
                            .module_idx = ref.local_idx,
                            .cwasm_bytes = cwasm,
                            .core_wasm = ref.data,
                        });
                    }
                }
            }
            const precompiled_cores: []const wamr.component_core_backend.PrecompiledCore =
                if (loaded_manifest) |lm| lm.precompiledCores() else auto_cores.items;

            // #644: AOT-only policy. `wamr run` requires a precompiled
            // bundle for every component — there is no interp fallback
            // at the CLI surface. With auto-precompile above this
            // should only fire when the component had no core modules
            // (an empty / malformed input that survived parsing).
            if (precompiled_cores.len == 0) {
                std.debug.print(
                    "Error: component has no AOT-compiled cores and `wamr run` is AOT-only.\n" ++
                        "  See issue #644.\n",
                    .{},
                );
                return 2;
            }

            if (listen_address) |addr| {
                // Load + parse the TLS cert + key at startup, so a
                // missing file / malformed PEM surfaces before `bind`
                // (matches the documented startup-validation rule for
                // every other host-config flag).
                var tls_config: ?wamr.wasi_cli_adapter.HttpsTlsConfig = null;
                if (tls_pem_path) |p| {
                    tls_config = wamr.wasi_cli_adapter.HttpsTlsConfig.loadFromCombinedPath(allocator, p) catch |err| {
                        std.debug.print("Error: --tls-pem '{s}': {s}\n", .{ p, tlsLoadErrorMessage(err) });
                        return 2;
                    };
                } else if (tls_cert_path) |cp| {
                    const kp = tls_key_path.?;
                    tls_config = wamr.wasi_cli_adapter.HttpsTlsConfig.loadFromPaths(allocator, cp, kp) catch |err| {
                        std.debug.print("Error: --tls-cert '{s}' / --tls-key '{s}': {s}\n", .{ cp, kp, tlsLoadErrorMessage(err) });
                        return 2;
                    };
                }
                defer if (tls_config) |*c| c.deinit();
                return runHttpComponent(wasm_data, allocator, path, wasm_args.items, env_list.items, addr, effective_log_level, if (tls_config) |*c| c else null, precompiled_cores);
            }
            return runComponent(wasm_data, allocator, path, wasm_args.items, env_list.items, map_dirs.items, allow_net.items, effective_log_level, cfg_entries, keyvalue_store_path, precompiled_cores);
        }
    }

    // Plain core wasm. The `wamr` CLI is AOT-only as of #644 — there
    // is no interp fallback at the CLI surface. Auto-compile the
    // module in-process to AOT and run it through the AOT runtime
    // with WASI preview1 wired up via `src/runtime/aot/host_bridge.zig`.
    if (comptime !aot_supported) {
        std.debug.print(
            "Error: AOT execution is not supported on this architecture, and `wamr run` is AOT-only.\n" ++
                "  See issue #644.\n",
            .{},
        );
        return 1;
    }
    if (listen_address != null) {
        std.debug.print("Error: --listen is not supported for plain core wasm modules (use a component)\n", .{});
        return 2;
    }
    return runCoreAot(
        init.io,
        allocator,
        wasm_data,
        wasm_args.items,
        env_flags.items,
        init.environ_map,
        map_dirs.items,
    );
}

/// Compile a plain core-wasm module to AOT in-process and execute it
/// with WASI preview1 host functions resolved via the AOT host
/// bridge (`src/runtime/aot/host_bridge.zig`). This is the only
/// supported path for `\0asm` (core) inputs after #644 — the legacy
/// interpreter `runWasm` was removed. The interpreter remains
/// available as a library for tests / embedders.
fn runCoreAot(
    io: std.Io,
    allocator: std.mem.Allocator,
    wasm_data: []const u8,
    wasm_args: []const []const u8,
    env_flags: []const []const u8,
    environ_map: *const std.process.Environ.Map,
    map_dirs: []const MapDir,
) u8 {
    const aot_loader = wamr.aot_loader;
    const aot_runtime = wamr.aot_runtime;

    // 1. Build the WasiCtx (args, env, preopens). Same lifetime model
    // as runComponent: borrows strings from caller-owned slices.
    var ctx = wamr.WasiCtx.init(allocator, io) catch |err| {
        std.debug.print("Error: failed to init WASI ctx: {s}\n", .{@errorName(err)});
        return 1;
    };
    defer ctx.deinit();

    // argv[0] is the wasm filename (matches the interpreter precedent).
    // We can't pass `path` here because the caller-side variable lives
    // higher up; pick "wasm" as a stable placeholder (wasi-libc only
    // uses argv[0] for `program_invocation_name`, not for routing).
    var argv_buf = allocator.alloc([]const u8, 1 + wasm_args.len) catch return 1;
    defer allocator.free(argv_buf);
    argv_buf[0] = "wasm";
    for (wasm_args, 0..) |a, i| argv_buf[1 + i] = a;
    ctx.setArgs(argv_buf);

    var env_buf: std.ArrayListUnmanaged([]const u8) = .empty;
    defer env_buf.deinit(allocator);
    if (env_flags.len > 0) {
        env_buf.ensureTotalCapacity(allocator, env_flags.len) catch return 1;
        for (env_flags) |kv| env_buf.appendAssumeCapacity(kv);
    } else {
        var it = environ_map.array_hash_map.iterator();
        while (it.next()) |kv| {
            // wasi-libc expects "NAME=value" strings.
            const joined = std.fmt.allocPrint(allocator, "{s}={s}", .{ kv.key_ptr.*, kv.value_ptr.* }) catch return 1;
            env_buf.append(allocator, joined) catch {
                allocator.free(joined);
                return 1;
            };
        }
    }
    defer if (env_flags.len == 0) {
        // Only free strings we allocated ourselves above.
        for (env_buf.items) |s| allocator.free(s);
    };
    ctx.setEnv(env_buf.items);

    for (map_dirs) |md| {
        _ = ctx.openMappedDir(md.host_path, md.guest_name) catch |err| {
            std.debug.print("Error: --map-dir '{s}::{s}': {s}\n", .{ md.host_path, md.guest_name, @errorName(err) });
            return 1;
        };
    }

    // 2. AOT-compile the core wasm in-process. `compileCoreWasm` mirrors
    // `wamrc compile` exactly so the same artifact would be produced.
    const cwasm = wamr.component_aot.compileCoreWasm(allocator, wasm_data, .{}) catch |err| {
        std.debug.print("Error: failed to AOT-compile core wasm: {s}\n", .{@errorName(err)});
        return 1;
    };
    defer allocator.free(cwasm);

    const aot_module = aot_loader.load(cwasm, allocator) catch |err| {
        std.debug.print("Error: failed to load freshly compiled AOT module: {s}\n", .{@errorName(err)});
        return 1;
    };
    defer aot_loader.unload(&aot_module, allocator);

    const inst = aot_runtime.instantiate(&aot_module, allocator) catch |err| {
        std.debug.print("Error: failed to instantiate AOT module: {s}\n", .{@errorName(err)});
        return 1;
    };
    defer aot_runtime.destroy(inst);

    aot_runtime.mapCodeExecutable(inst) catch |err| {
        std.debug.print("Error: failed to map AOT code as executable: {s}\n", .{@errorName(err)});
        return 1;
    };

    // 3. Wire the WasiCtx so the AOT WASI adapters in
    // `runtime/aot/host_bridge.zig` can resolve it via `vmctx.wasi_ctx`.
    inst.wasi_ctx = @intFromPtr(ctx);

    const func_idx = aot_runtime.findExportFunc(inst, "_start") orelse
        aot_runtime.findExportFunc(inst, "main") orelse {
        std.debug.print("Error: no _start or main export\n", .{});
        return 1;
    };

    // `_start` / `main` take no params and return no values for the
    // wasi-libc CRT shape; route through `callFuncScalar` with empty
    // slices. Any guest call to `proc_exit` short-circuits via
    // `std.process.exit` inside the host bridge.
    var results_buf: [0]wamr.aot_runtime.ScalarResult = .{};
    _ = aot_runtime.callFuncScalar(inst, func_idx, &.{}, &.{}, &.{}, &results_buf) catch |err| {
        std.debug.print("Error: AOT execution failed: {s}\n", .{@errorName(err)});
        return 1;
    };

    // If the guest never called proc_exit, exit_code is null → success.
    if (ctx.exit_code) |code| return @intCast(code & 0xFF);
    return 0;
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

/// Assemble the merged `wasi:config/store@0.2.0-rc.1` backing slice
/// from two layered sources (#583 B6):
///
///   1. **Env-var layer** — every host process env var whose name
///      begins with `WAMR_CONFIG_`. The prefix is stripped and the
///      remainder is **lower-cased ASCII-only** (so
///      `WAMR_CONFIG_API_KEY=secret` becomes
///      `ConfigEntry{ name = "api_key", value = "secret" }`). The
///      value is taken verbatim. Empty keys (`WAMR_CONFIG_=`) are
///      skipped.
///   2. **File layer** — if `config_path` is non-null, the JSON file
///      at that path is parsed as a flat `{ "key": "value", ... }`
///      object. Non-string values are rejected with
///      `error.ConfigStoreInvalidValueType`. Nested objects / arrays
///      are also rejected — the wasi:config WIT only models flat
///      `string → string` pairs.
///
/// Precedence: **file overrides env**. Env entries whose lower-cased
/// key matches a file-layer key are dropped before concatenation, so
/// `configStoreGet`'s first-match walk in the adapter returns the
/// file value. The file entries are listed first in the returned
/// slice and the surviving env entries are appended afterwards;
/// `lookupConfig`'s walk semantics preserve the precedence regardless.
///
/// Lifetimes: all string storage is allocated through `arena`. Caller
/// owns the arena and is responsible for tearing it down after the
/// adapter has finished its run.
fn loadComponentConfigStore(
    arena: std.mem.Allocator,
    environ_map: *const std.process.Environ.Map,
    config_path: ?[]const u8,
) ![]wamr.wasi_cli_adapter.ConfigEntry {
    const prefix = "WAMR_CONFIG_";

    // ── Layer 2 (file) — parsed first so we know which env entries
    //    to drop. Build a small key-set keyed by file entries.
    var file_entries: std.ArrayListUnmanaged(wamr.wasi_cli_adapter.ConfigEntry) = .empty;
    var file_keys: std.StringHashMapUnmanaged(void) = .empty;
    if (config_path) |p| {
        const cwd = std.Io.Dir.cwd();
        const io = std.Io.Threaded.global_single_threaded.io();
        const bytes = cwd.readFileAlloc(io, p, arena, @enumFromInt(8 * 1024 * 1024)) catch |err| switch (err) {
            error.FileNotFound => return error.ConfigStoreNotFound,
            else => return err,
        };
        var parsed = std.json.parseFromSlice(std.json.Value, arena, bytes, .{}) catch {
            return error.ConfigStoreInvalidJson;
        };
        defer parsed.deinit();
        switch (parsed.value) {
            .object => |obj| {
                var it = obj.iterator();
                while (it.next()) |kv| {
                    const value_str = switch (kv.value_ptr.*) {
                        .string => |s| s,
                        else => return error.ConfigStoreInvalidValueType,
                    };
                    const name_dup = try arena.dupe(u8, kv.key_ptr.*);
                    const value_dup = try arena.dupe(u8, value_str);
                    try file_entries.append(arena, .{ .name = name_dup, .value = value_dup });
                    try file_keys.put(arena, name_dup, {});
                }
            },
            else => return error.ConfigStoreInvalidRoot,
        }
    }

    // ── Layer 1 (env) — scan host env, lowercase keys, drop entries
    //    whose key collides with the file layer.
    var env_entries: std.ArrayListUnmanaged(wamr.wasi_cli_adapter.ConfigEntry) = .empty;
    var it = environ_map.array_hash_map.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        const value = kv.value_ptr.*;
        if (!std.mem.startsWith(u8, name, prefix)) continue;
        const tail = name[prefix.len..];
        if (tail.len == 0) continue;
        const lowered = try std.ascii.allocLowerString(arena, tail);
        if (file_keys.contains(lowered)) continue;
        const value_dup = try arena.dupe(u8, value);
        try env_entries.append(arena, .{ .name = lowered, .value = value_dup });
    }

    // ── Merge: file entries first (so first-match wins matches the
    //    documented precedence), env-only leftovers second.
    var merged: std.ArrayListUnmanaged(wamr.wasi_cli_adapter.ConfigEntry) = .empty;
    try merged.ensureTotalCapacity(arena, file_entries.items.len + env_entries.items.len);
    try merged.appendSlice(arena, file_entries.items);
    try merged.appendSlice(arena, env_entries.items);
    return merged.items;
}

fn runComponent(
    data: []const u8,
    allocator: std.mem.Allocator,
    wasm_path: []const u8,
    wasm_args: []const []const u8,
    env_vars: []const wamr.wasi_cli_adapter.EnvVar,
    map_dirs: []const MapDir,
    allow_net_cidrs: []const []const u8,
    log_level: ?wamr.wasi_cli_adapter.WasiLogLevel,
    config_store: []const wamr.wasi_cli_adapter.ConfigEntry,
    keyvalue_store_path: ?[]const u8,
    precompiled_cores: []const wamr.component_core_backend.PrecompiledCore,
) u8 {
    const adapter_mod = wamr.wasi_cli_adapter;
    // Wire the adapter's stdio directly to the host process's
    // STDIN/STDOUT/STDERR so output streams live (no end-of-run
    // flush) and stdin reads from the user's terminal / piped
    // input (#474). For the test/embedding path that needs to
    // inspect captured stdout, use `WasiCliAdapter.init` instead.
    var adapter = adapter_mod.WasiCliAdapter.initWithHostStdio(allocator);
    defer adapter.deinit();

    // argv[0] = basename of the wasm path, rest = user args. Matches
    // the wasmtime convention (and wasi-testsuite fixtures' assumption,
    // see wasm32-wasip3 `cli-env.rs` asserting on `"cli-env.wasm"`).
    var argv_buf = allocator.alloc([]const u8, 1 + wasm_args.len) catch
        return 1;
    defer allocator.free(argv_buf);
    argv_buf[0] = std.fs.path.basename(wasm_path);
    for (wasm_args, 0..) |a, i| argv_buf[i + 1] = a;
    adapter.setArguments(argv_buf);
    adapter.setEnvironment(env_vars);
    // Seed `wasi:config/store@0.2.0-rc.1` (#583 B6). Caller assembles
    // the merged env-then-file layered slice in `runRun`; an empty
    // slice means no config source was provided and the guest sees
    // an empty store.
    adapter.setConfigStore(config_store);

    // `--keyvalue-store=<path>` (#583 B4 follow-up): activate the
    // file-backed persistence layer for `wasi:keyvalue`. The flag
    // is null by default — guest sees the in-memory store from PR
    // #601, byte-for-byte unchanged.
    if (keyvalue_store_path) |p| {
        adapter.setKeyvalueStorePath(p) catch |err| {
            std.debug.print("Error: --keyvalue-store '{s}': {}\n", .{ p, err });
            return 1;
        };
    }

    // `--log-level=<name>` / `WAMR_LOG_LEVEL=<name>` host-side
    // `wasi:logging/logging.log` severity filter (#583 B5). Default
    // (`null` here) leaves the adapter at `.trace`, which admits
    // every level — matches the README "verbose by default" stance.
    if (log_level) |lvl| adapter.setLogLevel(lvl);

    // Seed the WASI sockets allow-list from `--allow-net=CIDR` flags
    // (default deny-all). (#520 wave 2)
    if (allow_net_cidrs.len > 0) {
        adapter.setSocketsAllowList(allow_net_cidrs) catch |err| {
            std.debug.print("Error: invalid --allow-net CIDR: {}\n", .{err});
            return 1;
        };
    }

    // Register each `--map-dir HOST::GUEST` flag as a filesystem
    // preopen. wasi-testsuite fixtures (e.g. wasm32-wasip3
    // `filesystem-stat`) discover their fs-test directory via
    // `wasi:filesystem/preopens.get-directories`; without these
    // registrations the test asserts no preopens and exits early
    // with a usage message. (#520 wave 2)
    if (map_dirs.len > 0) {
        const io = std.Io.Threaded.global_single_threaded.io();
        const cwd_dir = std.Io.Dir.cwd();
        for (map_dirs) |md| {
            // Open with `.iterate = true` so the guest can enumerate
            // the preopen (and any sub-dir opened via `open-at`) via
            // `descriptor.read-directory`. Without it, `getdents64`
            // returns BADF on Linux. (#571 — fixes filesystem-read-directory.)
            const opened = cwd_dir.openDir(io, md.host_path, .{ .iterate = true }) catch |err| {
                std.debug.print(
                    "Error: cannot pre-open '{s}' as '{s}': {}\n",
                    .{ md.host_path, md.guest_name, err },
                );
                return 1;
            };
            _ = adapter.addPreopen(md.guest_name, opened) catch |err| {
                std.debug.print(
                    "Error: cannot register preopen '{s}': {}\n",
                    .{ md.guest_name, err },
                );
                opened.close(io);
                return 1;
            };
        }
    }

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

    const outcome = adapter_mod.runComponentBytes(data, arena_alloc, &adapter, .{
        .precompiled_cores = precompiled_cores,
        .aot_only = true,
    }) catch |err| {
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
            error.AotImportUnresolvable => std.debug.print(
                "Error: AOT cannot run this component yet. The `wamr` CLI is AOT-only (see [aot reject] log line(s) above for the failing import). " ++
                    "Use the library API directly to run on the interpreter (issue #644).\n",
                .{},
            ),
            error.InstantiateFailed => std.debug.print(
                "Error: failed to instantiate component (the `wamr` CLI runs all components through the AOT runtime; see issue #644)\n",
                .{},
            ),
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
    log_level: ?wamr.wasi_cli_adapter.WasiLogLevel,
    tls_config: ?*const wamr.wasi_cli_adapter.HttpsTlsConfig,
    precompiled_cores: []const wamr.component_core_backend.PrecompiledCore,
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
    if (log_level) |lvl| adapter.setLogLevel(lvl);

    // See `runComponent` for the rationale behind the arena wrapper —
    // same loader/instance allocation story applies on the HTTP path.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    adapter_mod.serveHttpComponentBytes(data, arena_alloc, &adapter, .{
        .listen_address = listen_address,
        .announce_listening = listen_address.getPort() == 0,
        .tls_config = tls_config,
    }, .{
        .precompiled_cores = precompiled_cores,
        .aot_only = true,
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
            error.AddressInUse => std.debug.print(
                "Error: --listen address already in use (another process is bound to this port)\n",
                .{},
            ),
            error.AcceptFailed => std.debug.print("Error: failed to accept HTTP connection\n", .{}),
            error.LoadFailed => std.debug.print("Error: failed to load component\n", .{}),
            error.AotImportUnresolvable => std.debug.print(
                "Error: AOT cannot run this component yet. The `wamr` CLI is AOT-only (see [aot reject] log line(s) above for the failing import). " ++
                    "Use the library API directly to run on the interpreter (issue #644).\n",
                .{},
            ),
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


fn writeStdout(io: std.Io, text: []const u8) void {
    var stdout_file = std.Io.File.stdout();
    stdout_file.writeStreamingAll(io, text) catch {};
}

/// Map an `HttpsTlsConfig.LoadError` to a short, user-facing
/// diagnostic string. Used by `runRun` to surface cert / key load
/// failures before bind in a uniform format.
fn loadManifestErrorMessage(err: wamr.component_aot.LoadError) []const u8 {
    return switch (err) {
        error.ManifestNotFound => "manifest.json not found in the directory",
        error.ManifestParseFailed => "manifest.json could not be parsed",
        error.ManifestVersionMismatch => "manifest format version not understood by this build",
        error.ManifestBuildIdMismatch => "manifest was produced by a different wamr build (recompile with `wamrc compile-component`)",
        error.ManifestComponentMismatch => "manifest's component hash does not match this component (stale bundle — recompile with `wamrc compile-component`)",
        error.CwasmReadFailed => "could not read a .cwasm artifact referenced from the manifest",
        error.CwasmHashMismatch => "a .cwasm artifact's contents differ from the manifest's recorded hash (tampered or partial write)",
        error.OutOfMemory => "out of memory while loading the manifest",
    };
}

fn tlsLoadErrorMessage(err: wamr.wasi_cli_adapter.HttpsTlsConfig.LoadError) []const u8 {
    return switch (err) {
        error.TlsCertFileNotFound => "certificate file not found",
        error.TlsKeyFileNotFound => "private-key file not found",
        error.TlsCertReadFailed => "failed to read certificate file",
        error.TlsKeyReadFailed => "failed to read private-key file",
        error.TlsCertParseFailed => "certificate PEM did not parse (missing END marker or malformed base64)",
        error.TlsKeyParseFailed => "private-key PEM did not parse (unrecognised BEGIN marker — expected PRIVATE KEY, RSA PRIVATE KEY, or EC PRIVATE KEY)",
        error.TlsCertEmpty => "certificate file did not contain any PEM blocks",
        error.TlsKeyEmpty => "private-key block missing from combined PEM file",
        error.OutOfMemory => "out of memory while loading TLS material",
    };
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
    \\The `wamr` CLI is AOT-only (issue #644):
    \\  * `.cwasm`/`.aot` core modules (magic `\0aot`) run via the AOT
    \\    runtime directly.
    \\  * Component-model `.wasm` files run via AOT cores loaded from a
    \\    `wamrc compile-component` bundle (see --precompiled-dir below).
    \\    Components without a bundle fail with a clear error.
    \\  * Plain core wasm modules are not supported — pre-compile to
    \\    `.cwasm`/`.aot` first.
    \\
    \\Options:
    \\  --stack-size=<bytes>     (ignored; kept for backward compat)
    \\  --heap-size=<bytes>      Reserved (currently ignored)
    \\  --listen[=<ip:port>]     For components: serve WASI HTTP on the address.
    \\                           Port 0 (and bare `--listen`, which means
    \\                           127.0.0.1:0) requests a kernel-assigned
    \\                           ephemeral port; the resolved address is
    \\                           printed to stdout after bind.
    \\  --env KEY=VALUE          Set a WASI environment variable (repeatable)
    \\  --map-dir HOST::GUEST    Pre-open `HOST` host directory as `GUEST`
    \\                           inside the guest WASI sandbox (repeatable)
    \\  --allow-net CIDR         For components: allow wasi:sockets bind /
    \\                           connect / DNS to addresses inside CIDR
    \\                           (e.g. `127.0.0.0/8`). Default deny-all.
    \\                           Repeatable.
    \\  --log-level NAME         For components: filter `wasi:logging` calls
    \\                           below NAME. One of trace|debug|info|warn|
    \\                           error|critical (default: trace = admit all).
    \\                           Falls back to the WAMR_LOG_LEVEL env var
    \\                           when this flag is absent.
    \\  --config-store PATH      For components: load layered `wasi:config/store`
    \\                           values from a JSON file with a flat
    \\                           {"key":"value",...} object. Combined with
    \\                           env vars matching `WAMR_CONFIG_<KEY>=<value>`
    \\                           (key lower-cased, prefix stripped). File
    \\                           overrides env when a key is set by both.
    \\  --keyvalue-store PATH    For components: back the `wasi:keyvalue`
    \\                           adapter with a JSON file. Format:
    \\                           {"bucket":{"key":"<base64-value>",...},...}.
    \\                           Loads on startup (missing file is OK),
    \\                           rewrites synchronously on every mutation.
    \\                           Omit to keep the default in-memory store.
    \\  --tls-cert PATH          For components serving HTTP: PEM-encoded
    \\                           certificate chain (leaf first). Requires
    \\                           --listen and a matching --tls-key. Today
    \\                           the cert + key are parsed + validated at
    \\                           startup but the handshake is upstream-
    \\                           blocked on Zig std (see cataggar/wamr#609);
    \\                           the listener serves plaintext with a
    \\                           single stderr warning.
    \\  --tls-key PATH           For components serving HTTP: PEM-encoded
    \\                           private key (PKCS#8, RSA, or EC). Pairs
    \\                           with --tls-cert.
    \\  --tls-pem PATH           For components serving HTTP: combined PEM
    \\                           file containing both certificate chain
    \\                           and private key. Mutually exclusive with
    \\                           --tls-cert / --tls-key.
    \\  --precompiled-dir PATH   For components: load AOT-compiled cores from
    \\                           a `wamrc compile-component` output directory
    \\                           (containing manifest.json + module<N>.cwasm).
    \\                           When omitted, `wamr run` auto-detects a
    \\                           sibling `<input>.cwasm.d/manifest.json`. The
    \\                           bundle is mandatory: components without one
    \\                           fail with a clear error (issue #644).
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

test "loadComponentConfigStore: env-only picks up WAMR_CONFIG_* with lowercase keys (#583 B6)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var env_map = std.process.Environ.Map.init(arena.allocator());
    try env_map.put("WAMR_CONFIG_FOO", "bar");
    try env_map.put("WAMR_CONFIG_API_KEY", "s3cr3t");
    try env_map.put("PATH", "/usr/bin"); // not prefixed → ignored
    try env_map.put("WAMR_CONFIG_", "empty-key"); // empty tail → skipped

    const entries = try loadComponentConfigStore(arena.allocator(), &env_map, null);
    try std.testing.expectEqual(@as(usize, 2), entries.len);

    var saw_foo = false;
    var saw_api = false;
    for (entries) |e| {
        if (std.mem.eql(u8, e.name, "foo")) {
            try std.testing.expectEqualStrings("bar", e.value);
            saw_foo = true;
        } else if (std.mem.eql(u8, e.name, "api_key")) {
            try std.testing.expectEqualStrings("s3cr3t", e.value);
            saw_api = true;
        } else {
            return error.UnexpectedKey;
        }
    }
    try std.testing.expect(saw_foo);
    try std.testing.expect(saw_api);
}

test "loadComponentConfigStore: file-only loads flat JSON object (#583 B6)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var env_map = std.process.Environ.Map.init(arena.allocator());

    // Write a JSON file under .zig-cache to avoid polluting the
    // repository working tree. The path is relative to the cwd the
    // `zig build test` step runs in — typically the repo root.
    const cwd = std.Io.Dir.cwd();
    const io = std.Io.Threaded.global_single_threaded.io();
    const tmp_dir = cwd.openDir(io, ".zig-cache", .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    const file_name = "wasi-config-test-store.json";
    var file = try tmp_dir.createFile(io, file_name, .{ .truncate = true });
    defer {
        file.close(io);
        tmp_dir.deleteFile(io, file_name) catch {};
    }
    try file.writeStreamingAll(io, "{\"foo\":\"bar\",\"answer\":\"42\"}");

    const path = ".zig-cache/" ++ file_name;
    const entries = try loadComponentConfigStore(arena.allocator(), &env_map, path);
    try std.testing.expectEqual(@as(usize, 2), entries.len);
    // File entries are emitted in JSON-source order.
    try std.testing.expectEqualStrings("foo", entries[0].name);
    try std.testing.expectEqualStrings("bar", entries[0].value);
    try std.testing.expectEqualStrings("answer", entries[1].name);
    try std.testing.expectEqualStrings("42", entries[1].value);
}

test "loadComponentConfigStore: file overrides env on duplicate key (#583 B6)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var env_map = std.process.Environ.Map.init(arena.allocator());
    try env_map.put("WAMR_CONFIG_PORT", "9999"); // env layer
    try env_map.put("WAMR_CONFIG_HOST", "localhost"); // env-only, survives

    const cwd = std.Io.Dir.cwd();
    const io = std.Io.Threaded.global_single_threaded.io();
    const tmp_dir = cwd.openDir(io, ".zig-cache", .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    const file_name = "wasi-config-test-override.json";
    var file = try tmp_dir.createFile(io, file_name, .{ .truncate = true });
    defer {
        file.close(io);
        tmp_dir.deleteFile(io, file_name) catch {};
    }
    try file.writeStreamingAll(io, "{\"port\":\"8080\"}");

    const path = ".zig-cache/" ++ file_name;
    const entries = try loadComponentConfigStore(arena.allocator(), &env_map, path);
    try std.testing.expectEqual(@as(usize, 2), entries.len);

    // Layout: file entries first (so first-match walk in
    // configStoreGet returns the file value), env-only leftovers
    // appended afterwards.
    try std.testing.expectEqualStrings("port", entries[0].name);
    try std.testing.expectEqualStrings("8080", entries[0].value);
    try std.testing.expectEqualStrings("host", entries[1].name);
    try std.testing.expectEqualStrings("localhost", entries[1].value);
}

test "loadComponentConfigStore: rejects non-string values and non-object roots (#583 B6)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var env_map = std.process.Environ.Map.init(arena.allocator());

    const cwd = std.Io.Dir.cwd();
    const io = std.Io.Threaded.global_single_threaded.io();
    const tmp_dir = cwd.openDir(io, ".zig-cache", .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    // Non-string value (int).
    {
        const fname = "wasi-config-test-bad-value.json";
        var file = try tmp_dir.createFile(io, fname, .{ .truncate = true });
        defer {
            file.close(io);
            tmp_dir.deleteFile(io, fname) catch {};
        }
        try file.writeStreamingAll(io, "{\"port\":8080}");
        try std.testing.expectError(
            error.ConfigStoreInvalidValueType,
            loadComponentConfigStore(arena.allocator(), &env_map, ".zig-cache/" ++ fname),
        );
    }

    // Non-object root.
    {
        const fname = "wasi-config-test-bad-root.json";
        var file = try tmp_dir.createFile(io, fname, .{ .truncate = true });
        defer {
            file.close(io);
            tmp_dir.deleteFile(io, fname) catch {};
        }
        try file.writeStreamingAll(io, "[\"not\", \"an\", \"object\"]");
        try std.testing.expectError(
            error.ConfigStoreInvalidRoot,
            loadComponentConfigStore(arena.allocator(), &env_map, ".zig-cache/" ++ fname),
        );
    }

    // Missing file.
    try std.testing.expectError(
        error.ConfigStoreNotFound,
        loadComponentConfigStore(arena.allocator(), &env_map, ".zig-cache/wasi-config-test-does-not-exist.json"),
    );
}

test "tlsLoadErrorMessage covers every load-error arm (#583 follow-up / #609)" {
    // Each arm of `HttpsTlsConfig.LoadError` must have a unique
    // user-facing diagnostic string. A new arm added upstream
    // without a matching switch arm here causes a Zig compile
    // error (`error.X is not handled`), which is the safety net.
    const testing = std.testing;
    try testing.expect(tlsLoadErrorMessage(error.TlsCertFileNotFound).len > 0);
    try testing.expect(tlsLoadErrorMessage(error.TlsKeyFileNotFound).len > 0);
    try testing.expect(tlsLoadErrorMessage(error.TlsCertReadFailed).len > 0);
    try testing.expect(tlsLoadErrorMessage(error.TlsKeyReadFailed).len > 0);
    try testing.expect(tlsLoadErrorMessage(error.TlsCertParseFailed).len > 0);
    try testing.expect(tlsLoadErrorMessage(error.TlsKeyParseFailed).len > 0);
    try testing.expect(tlsLoadErrorMessage(error.TlsCertEmpty).len > 0);
    try testing.expect(tlsLoadErrorMessage(error.TlsKeyEmpty).len > 0);
    try testing.expect(tlsLoadErrorMessage(error.OutOfMemory).len > 0);
    // Make sure two distinct arms produce distinct messages —
    // catches accidental copy-paste collapses in the switch.
    try testing.expect(!std.mem.eql(
        u8,
        tlsLoadErrorMessage(error.TlsCertFileNotFound),
        tlsLoadErrorMessage(error.TlsKeyFileNotFound),
    ));
}
