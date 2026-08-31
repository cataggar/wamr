const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");

// Compiler debug logs are runtime diagnostics, not guest stderr. Keeping the
// CLI threshold at info prevents in-process JIT optimization messages from
// contaminating conformance fixtures that assert byte-exact stderr.
pub const std_options: std.Options = .{ .log_level = .info };

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

const Subcommand = enum { run, serve, version, help };

/// Set by `WAMR_TRACE_CLI_ADAPTER=1|verbose` at startup (#715). Each
/// `WasiCliAdapter` constructed during `runRun` / `runComponent`
/// receives the on/off flag via `setTraceCalls` and the verbose flag
/// via `setTraceVerbose`. Process-global mirrors the `WAMR_AOT_DEBUG`
/// toggle next door.
var wasi_cli_adapter_trace_enabled: bool = false;
var wasi_cli_adapter_trace_verbose: bool = false;

fn parseSubcommand(s: []const u8) ?Subcommand {
    if (std.mem.eql(u8, s, "run")) return .run;
    if (std.mem.eql(u8, s, "serve")) return .serve;
    if (std.mem.eql(u8, s, "version")) return .version;
    if (std.mem.eql(u8, s, "help")) return .help;
    return null;
}

/// Returns the host process exit code. Returning (rather than calling
/// `std.process.exit`) lets the Zig 0.16 startup runtime run its own
/// teardown — most importantly the `DebugAllocator` leak hook on
/// `init.gpa` in Debug builds (#449, #450). Exit-code policy:
///   * 0 — successful guest run / `help` / `version`
///   * `outcome.exit_code` / `ctx.getExitCode()` — guest-requested
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
    // Process-global toggle for the AOT cross-instance fast-thunk
    // cross-memory guard (#719 Bug B). The thunk forwards raw core args
    // between sibling AOT instances with no canon.lower / canon.lift
    // marshaling; when source and target memories differ, any i32 the
    // caller intended as a pointer silently dereferences whatever bytes
    // live at the same numeric offset in the target's memory. By default
    // we only `std.log.warn` (once per import) on first detection so
    // working configurations aren't broken; when this env var is set we
    // also trap, which converts silent corruption into a clean error
    // with the import label.
    if (init.environ_map.get("WAMR_TRAP_CROSS_MEMORY_THUNK")) |v| {
        const on = !(v.len == 0 or std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "false"));
        wamr.component_core_backend.setTrapCrossMemoryEnabled(on);
    }
    // Per-member adapter-call trace (#715). When `WAMR_TRACE_CLI_ADAPTER`
    // is set to a non-empty / non-`0` / non-`false` value, every call
    // into a `WasiCliAdapter` filesystem host import emits an
    // `[adapter→]`/`[adapter←]` warn line. The adapter constructed in
    // `runRun` (or `runComponent`) reads this flag through `setTraceCalls`.
    if (init.environ_map.get("WAMR_TRACE_CLI_ADAPTER")) |v| {
        const is_verbose = std.mem.eql(u8, v, "verbose");
        const on = is_verbose or !(v.len == 0 or std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "false"));
        wasi_cli_adapter_trace_enabled = on;
        wasi_cli_adapter_trace_verbose = is_verbose;
    }
    if (init.environ_map.get("WAMR_TRAP_OOB_DUMP")) |v| {
        if (v.len > 0) {
            wamr.aot_runtime.g_trap_oob_dump_env = v;
        }
    }
    // Tune the per-`TrampolinePool` slot cap (#756). Componentize-js
    // wasms that import lots of unsatisfied `wasi:*` interfaces can
    // chew through hundreds of slots on instantiation; the default
    // (`host_trampolines.DEFAULT_MAX_SLOTS`) is sized to comfortably
    // cover the workloads we know about, but the env var lets users
    // size the pool exactly without rebuilding. Clamped at
    // `MAX_SLOTS_HARD` inside `TrampolinePool.init`.
    if (init.environ_map.get("WAMR_MAX_TRAMPOLINE_SLOTS")) |v| {
        if (v.len > 0) {
            if (std.fmt.parseInt(u32, v, 10)) |n| {
                wamr.host_trampolines.current_max_slots = n;
            } else |err| {
                std.debug.print(
                    "warning: WAMR_MAX_TRAMPOLINE_SLOTS={s}: {s} — using default {d}\n",
                    .{ v, @errorName(err), wamr.host_trampolines.current_max_slots },
                );
            }
        }
    }
    // #857: opt-in cap on total resident JIT/AOT executable code
    // across every live `AotInstance` in this process (bytes). Default
    // 0 = unlimited (existing behavior unchanged) — set this to make a
    // long-lived embedding host/dev-server fail fast with a clear
    // `error.CodeBudgetExceeded` instead of growing unbounded.
    if (init.environ_map.get("WAMR_JIT_CODE_BUDGET_BYTES")) |v| {
        if (v.len > 0) {
            if (std.fmt.parseInt(usize, v, 10)) |n| {
                wamr.aot_runtime.JitCodeCache.budget_bytes = n;
            } else |err| {
                std.debug.print(
                    "warning: WAMR_JIT_CODE_BUDGET_BYTES={s}: {s} — using default (unlimited)\n",
                    .{ v, @errorName(err) },
                );
            }
        }
    }
    if (init.environ_map.get("WAMR_WATCH_ADDR")) |v| {
        if (v.len > 0) {
            wamr.aot_runtime.initWatchAddrFromEnv(v) catch |err| {
                std.debug.print("[watch-addr] init failed: {s}\n", .{@errorName(err)});
            };
        }
    }
    if (args.len < 2) {
        std.debug.print("error: missing subcommand — try `wamr help`\n", .{});
        return 2;
    }

    if (std.mem.eql(u8, args[1], "--version")) {
        return try runVersion(init.io, args[2..]);
    }

    const subcmd = parseSubcommand(args[1]) orelse {
        std.debug.print("error: unknown subcommand '{s}' — try `wamr help`\n", .{args[1]});
        return 2;
    };

    return switch (subcmd) {
        .version => try runVersion(init.io, args[2..]),
        .help => runHelp(init.io, args[2..]),
        .run => try runRun(init, allocator, args[2..]),
        .serve => try runServe(init, allocator, args[2..]),
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

/// #860: default optimization-pass preset for the in-process JIT compile
/// path (`wamr run`/`wamr serve` under `-Djit=true`). The JIT compiles
/// synchronously on every invocation — compile latency is part of the
/// user-visible cold start — unlike `wamrc compile`, whose `.cwasm` is
/// written once and reused across many runs. Default to `.fast` (skips
/// the passes that mainly pay off on long-running steady-state code:
/// global value numbering, loop-invariant hoisting, dominator-based
/// redundant-load forwarding, tail duplication, loop-bounds-check
/// hoisting/elision — see `passes.jit_fast_passes` for the full
/// rationale). Set `WAMR_JIT_FULL_OPT` to a non-empty / non-`0` /
/// non-`false` value to opt back into the full pipeline (better
/// steady-state throughput, higher compile latency) for both call
/// sites below.
fn jitPassPresetFromEnv(env: *const std.process.Environ.Map) wamr.passes.PassPreset {
    const v = env.get("WAMR_JIT_FULL_OPT") orelse return .fast;
    const on = !(v.len == 0 or std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "false"));
    return if (on) .full else .fast;
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
    // `--precompiled-manifest=<path>` (#642, #680): explicit path to a
    // `wamrc compile-component` manifest sidecar (`<stem>.cwasm.json`).
    // When unset, `runRun` probes for a sibling `<input>.cwasm.json`
    // next to the component file. Missing/stale bundle is fatal — the
    // `wamr` CLI no longer embeds the AOT compiler.
    var precompiled_manifest: ?[]const u8 = null;
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
            } else if (std.mem.eql(u8, arg, "--precompiled-manifest") or std.mem.startsWith(u8, arg, "--precompiled-manifest=")) {
                // `--precompiled-manifest <path>` (#642, #680): explicit
                // path to a `wamrc compile-component` manifest sidecar
                // (`<stem>.cwasm.json`). Per-core `.cwasm` files
                // resolve relative to the manifest's directory. A
                // mismatched / stale manifest is a hard error.
                if (precompiled_manifest != null) {
                    std.debug.print("error: --precompiled-manifest specified more than once\n", .{});
                    return 2;
                }
                const spec = if (std.mem.eql(u8, arg, "--precompiled-manifest")) blk: {
                    i += 1;
                    if (i >= run_args.len) {
                        std.debug.print("error: --precompiled-manifest requires a path to a wamrc compile-component manifest\n", .{});
                        return 2;
                    }
                    break :blk run_args[i];
                } else arg["--precompiled-manifest=".len..];
                if (spec.len == 0) {
                    std.debug.print("error: --precompiled-manifest path is empty\n", .{});
                    return 2;
                }
                precompiled_manifest = spec;
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

    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const wasm_data = cwd.readFileAlloc(io, path, allocator, @enumFromInt(256 * 1024 * 1024)) catch |err| {
        wamr.utils.read_file.dieReadFileError(path, err);
    };
    defer allocator.free(wasm_data);

    // Detect file type by magic bytes: AOT (\0aot) vs Wasm (\0asm).
    // For an AOT image we still need a WASI host context (env, argv,
    // preopens) — the same wiring `runCoreAot` used pre-#680 before the
    // embedded compiler was removed. The `.cwasm` path now flows
    // through this code via `wamrc compile` (or `wamrc run`).
    if (wasm_data.len >= 4 and std.mem.readInt(u32, wasm_data[0..4], .little) == wamr.types.aot_magic) {
        return runAot(
            init.io,
            allocator,
            wasm_data,
            wasm_args.items,
            env_flags.items,
            init.environ_map,
            map_dirs.items,
        );
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
            // #644 / #680 / #854: resolve the AOT precompiled-cores
            // manifest (shared with `wamr serve`). The `ComponentCoresSource`'s
            // lifetime must outlive `runComponent` because the mmapped
            // (or, on a `-Djit=true` in-process JIT compile, heap-owned)
            // `.cwasm` buffers it owns are borrowed by the `PrecompiledCore`
            // slice handed to the instance loader.
            var loaded_manifest: ?ComponentCoresSource = null;
            defer if (loaded_manifest) |*lm| lm.deinit();
            const manifest_rc = loadComponentManifestOrPrint(init, allocator, "run", path, wasm_data, precompiled_manifest, &loaded_manifest);
            if (manifest_rc != 0) return manifest_rc;
            const component_core_opts = loaded_manifest.?.instantiationOptions();

            return runComponent(wasm_data, allocator, path, wasm_args.items, env_list.items, map_dirs.items, allow_net.items, effective_log_level, cfg_entries, keyvalue_store_path, component_core_opts);
        }
    }

    // Plain core wasm.
    if (comptime wamr.config.jit) {
        // #853: in-process JIT. Compile in memory and execute in this
        // same process — no `.cwasm` written to disk, no `wamr`/`wamrc`
        // subprocess spawn (contrast with `wamrc run`, which does both
        // of those). `runAot` doesn't care whether `data` came from a
        // file or was just produced by the compiler above; it's the
        // same load/instantiate/execute path a precompiled `.cwasm`
        // takes today.
        // #860: default to the fast/baseline compile preset (see
        // `jitPassPresetFromEnv`) rather than `wamrc compile`'s
        // steady-state-optimized default.
        const cwasm = wamr.component_aot_compile.compileCoreWasm(allocator, wasm_data, .{
            .pass_preset = jitPassPresetFromEnv(init.environ_map),
        }) catch |err| {
            std.debug.print("Error: JIT compile of '{s}' failed: {s}\n", .{ path, @errorName(err) });
            return 1;
        };
        defer allocator.free(cwasm);
        return runAot(
            init.io,
            allocator,
            cwasm,
            wasm_args.items,
            env_flags.items,
            init.environ_map,
            map_dirs.items,
        );
    }

    // The `wamr` runtime doesn't embed the AOT compiler by default
    // (#680) — users must precompile with `wamrc compile` (or use
    // `wamrc run` for one-shot test execution), unless this binary was
    // built with `-Djit=true` for in-process JIT support (#852/#853).
    _ = aot_supported;
    std.debug.print(
        "Error: '{s}' is a plain core wasm module and the `wamr` runtime is AOT-only (#644)\n" ++
            "       without an embedded compiler (#680).\n" ++
            "  Run `wamrc compile {s}` to produce a `.cwasm`, then `wamr run <output>.cwasm`, or\n" ++
            "  run `wamrc run {s} [-- args...]` to compile and execute in one step, or\n" ++
            "  rebuild `wamr` with `-Djit=true` for in-process compile+run (#852).\n",
        .{ path, path, path },
    );
    return 2;
}

/// `wamr serve <component.wasm> [options]` (#845): serve a
/// `wasi:http/incoming-handler` component as a long-lived HTTP server.
/// Mirrors `wasmtime serve`:
///   * `--addr <ip:port>` bind address (default `127.0.0.1:8080`).
///   * `--addr 127.0.0.1:0` binds a kernel-assigned ephemeral port and
///     prints the resolved address to stdout (for test drivers).
///   * `--tls-cert` / `--tls-key` / `--tls-pem` terminate HTTPS.
/// AOT-only like `run`: the component needs a `wamrc compile-component`
/// manifest (explicit `--precompiled-manifest` or sibling
/// `<input>.cwasm.json`).
fn runServe(init: std.process.Init, allocator: std.mem.Allocator, serve_args: []const []const u8) !u8 {
    if (serve_args.len == 1 and std.mem.eql(u8, serve_args[0], "help")) {
        writeStdout(init.io, serve_usage);
        return 0;
    }
    // #918: block SIGINT/SIGTERM before any of the (potentially
    // multi-second) startup work below — component read, manifest /
    // AOT-or-JIT precompile, and instantiation — so a signal that races
    // startup is held pending on this (single) thread rather than lost to
    // the inherited `SIG_IGN` disposition. The real async-signal-safe
    // handler is installed and the signals unblocked later, right before
    // the accept loop (`serveLoadedHttpComponent` → `armHttpShutdown`), at
    // which point the pending signal is delivered and shuts the server
    // down cleanly. No-op on Windows.
    wamr.wasi_cli_adapter.blockHttpShutdownSignals();
    var wasm_path: ?[]const u8 = null;
    var wasm_args: std.ArrayListUnmanaged([]const u8) = .empty;
    defer wasm_args.deinit(allocator);
    var env_flags: std.ArrayListUnmanaged([]const u8) = .empty;
    defer env_flags.deinit(allocator);
    var addr: ?std.Io.net.IpAddress = null;
    var tls_cert_path: ?[]const u8 = null;
    var tls_key_path: ?[]const u8 = null;
    var tls_pem_path: ?[]const u8 = null;
    var log_level: ?wamr.wasi_cli_adapter.WasiLogLevel = null;
    var precompiled_manifest: ?[]const u8 = null;
    var past_options = false;

    var i: usize = 0;
    while (i < serve_args.len) : (i += 1) {
        const arg = serve_args[i];
        if (!past_options and arg.len > 0 and arg[0] == '-') {
            if (std.mem.eql(u8, arg, "--addr") or std.mem.startsWith(u8, arg, "--addr=")) {
                if (addr != null) {
                    std.debug.print("error: --addr specified more than once\n", .{});
                    return 2;
                }
                const spec = if (std.mem.eql(u8, arg, "--addr")) blk: {
                    i += 1;
                    if (i >= serve_args.len) {
                        std.debug.print("error: --addr requires an <ip:port> value\n", .{});
                        return 2;
                    }
                    break :blk serve_args[i];
                } else arg["--addr=".len..];
                addr = parseAddr(spec) catch {
                    std.debug.print("error: invalid --addr address '{s}'\n", .{spec});
                    return 2;
                };
            } else if (std.mem.eql(u8, arg, "--tls-cert") or std.mem.startsWith(u8, arg, "--tls-cert=")) {
                if (tls_cert_path != null) {
                    std.debug.print("error: --tls-cert specified more than once\n", .{});
                    return 2;
                }
                const spec = if (std.mem.eql(u8, arg, "--tls-cert")) blk: {
                    i += 1;
                    if (i >= serve_args.len) {
                        std.debug.print("error: --tls-cert requires a path to a PEM file\n", .{});
                        return 2;
                    }
                    break :blk serve_args[i];
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
                    if (i >= serve_args.len) {
                        std.debug.print("error: --tls-key requires a path to a PEM file\n", .{});
                        return 2;
                    }
                    break :blk serve_args[i];
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
                    if (i >= serve_args.len) {
                        std.debug.print("error: --tls-pem requires a path to a combined PEM file\n", .{});
                        return 2;
                    }
                    break :blk serve_args[i];
                } else arg["--tls-pem=".len..];
                if (spec.len == 0) {
                    std.debug.print("error: --tls-pem path is empty\n", .{});
                    return 2;
                }
                tls_pem_path = spec;
            } else if (std.mem.eql(u8, arg, "--env") or std.mem.startsWith(u8, arg, "--env=")) {
                const spec = if (std.mem.eql(u8, arg, "--env")) blk: {
                    i += 1;
                    if (i >= serve_args.len) {
                        std.debug.print("error: --env requires KEY=VALUE\n", .{});
                        return 2;
                    }
                    break :blk serve_args[i];
                } else arg["--env=".len..];
                if (std.mem.indexOfScalar(u8, spec, '=') == null) {
                    std.debug.print("error: --env value '{s}' is missing '='\n", .{spec});
                    return 2;
                }
                try env_flags.append(allocator, spec);
            } else if (std.mem.eql(u8, arg, "--log-level") or std.mem.startsWith(u8, arg, "--log-level=")) {
                const spec = if (std.mem.eql(u8, arg, "--log-level")) blk: {
                    i += 1;
                    if (i >= serve_args.len) {
                        std.debug.print("error: --log-level requires <trace|debug|info|warn|error|critical>\n", .{});
                        return 2;
                    }
                    break :blk serve_args[i];
                } else arg["--log-level=".len..];
                log_level = wamr.wasi_cli_adapter.WasiLogLevel.fromString(spec) orelse {
                    std.debug.print(
                        "error: --log-level value '{s}' is not one of trace|debug|info|warn|error|critical\n",
                        .{spec},
                    );
                    return 2;
                };
            } else if (std.mem.eql(u8, arg, "--precompiled-manifest") or std.mem.startsWith(u8, arg, "--precompiled-manifest=")) {
                if (precompiled_manifest != null) {
                    std.debug.print("error: --precompiled-manifest specified more than once\n", .{});
                    return 2;
                }
                const spec = if (std.mem.eql(u8, arg, "--precompiled-manifest")) blk: {
                    i += 1;
                    if (i >= serve_args.len) {
                        std.debug.print("error: --precompiled-manifest requires a path to a wamrc compile-component manifest\n", .{});
                        return 2;
                    }
                    break :blk serve_args[i];
                } else arg["--precompiled-manifest=".len..];
                if (spec.len == 0) {
                    std.debug.print("error: --precompiled-manifest path is empty\n", .{});
                    return 2;
                }
                precompiled_manifest = spec;
            } else if (std.mem.eql(u8, arg, "--trace-aot-wasi")) {
                wamr.aot_host_bridge.trace_enabled = true;
            } else if (std.mem.eql(u8, arg, "--")) {
                past_options = true;
            } else {
                std.debug.print("error: unknown option '{s}' — try `wamr serve help`\n", .{arg});
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
        std.debug.print("error: missing component file — usage: wamr serve [options] <component.wasm>\n", .{});
        return 2;
    };

    // TLS flag validation (#609): cert/key are load-bearing together;
    // the combined PEM is mutually exclusive with the split pair.
    if (tls_pem_path != null and (tls_cert_path != null or tls_key_path != null)) {
        std.debug.print("error: --tls-pem is mutually exclusive with --tls-cert / --tls-key\n", .{});
        return 2;
    }
    if ((tls_cert_path == null) != (tls_key_path == null)) {
        std.debug.print("error: --tls-cert and --tls-key must be specified together\n", .{});
        return 2;
    }

    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const wasm_data = cwd.readFileAlloc(io, path, allocator, @enumFromInt(256 * 1024 * 1024)) catch |err| {
        wamr.utils.read_file.dieReadFileError(path, err);
    };
    defer allocator.free(wasm_data);

    // `serve` is component-only. Reject AOT core images and plain core
    // wasm with a pointer at `run` (matches the proxy-vs-command split).
    if (wasm_data.len >= 4 and std.mem.readInt(u32, wasm_data[0..4], .little) == wamr.types.aot_magic) {
        std.debug.print("Error: `wamr serve` requires a wasi:http/incoming-handler component, not an AOT core module (use `wamr run`).\n", .{});
        return 2;
    }
    const is_component = wasm_data.len >= 8 and
        std.mem.readInt(u32, wasm_data[0..4], .little) == wamr.types.wasm_magic and
        std.mem.readInt(u32, wasm_data[4..8], .little) == wamr.types.component_version;
    if (!is_component) {
        std.debug.print("Error: `wamr serve` requires a wasi:http/incoming-handler component (got a non-component wasm). See `wamr serve help`.\n", .{});
        return 2;
    }

    // For components: prefer explicit --env entries; otherwise inherit
    // the host environment. EnvVar slices borrow from environ_map (which
    // lives for the entire process).
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

    // `--log-level` wins over `WAMR_LOG_LEVEL` (consulted only when the
    // flag is absent). (#583 B5)
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

    var loaded_manifest: ?ComponentCoresSource = null;
    defer if (loaded_manifest) |*lm| lm.deinit();
    const manifest_rc = loadComponentManifestOrPrint(init, allocator, "serve", path, wasm_data, precompiled_manifest, &loaded_manifest);
    if (manifest_rc != 0) return manifest_rc;
    const component_core_opts = loaded_manifest.?.instantiationOptions();

    // Load + parse the TLS cert + key at startup, so a missing file /
    // malformed PEM surfaces before `bind`.
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

    // Default to wasmtime's `127.0.0.1:8080` when `--addr` is omitted.
    const bind_addr = addr orelse (std.Io.net.IpAddress.parse("127.0.0.1", 8080) catch unreachable);
    return runHttpComponent(wasm_data, allocator, path, wasm_args.items, env_list.items, bind_addr, effective_log_level, if (tls_config) |*c| c else null, component_core_opts);
}

fn parseMapDir(spec: []const u8) !MapDir {
    const sep = std.mem.indexOf(u8, spec, "::") orelse return error.MissingSeparator;
    const host = spec[0..sep];
    const guest = spec[sep + 2 ..];
    if (host.len == 0 or guest.len == 0) return error.MissingSeparator;
    return .{ .host_path = host, .guest_name = guest };
}

/// Parse a `--addr <ip:port>` bind spec into an `IpAddress`. Matches
/// `wasmtime serve --addr`: an IP literal + port (no hostname
/// resolution). IPv6 is bracketed, e.g. `[::1]:8080`.
fn parseAddr(spec: []const u8) !std.Io.net.IpAddress {
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

/// Result of `loadComponentManifestOrPrint`: either an on-disk manifest
/// (explicit `--precompiled-manifest` or an auto-detected sibling
/// `<stem>.cwasm.json`) or, on a `-Djit=true` build with no manifest
/// found, an in-process JIT compile of the component (#854). Both
/// variants expose precompiled cores; the in-memory path may also carry
/// an internal lazy-JIT attach hook for the instantiated AOT cores
/// (#889), so callers downstream of `loadComponentManifestOrPrint`
/// consume backend options rather than raw slices alone.
const ComponentCoresSource = union(enum) {
    manifest: wamr.component_aot.LoadedManifest,
    in_memory: wamr.component_aot_compile.InMemoryPrecompiled,

    fn precompiledCores(self: *const ComponentCoresSource) []const wamr.component_core_backend.PrecompiledCore {
        return switch (self.*) {
            .manifest => |*m| m.precompiledCores(),
            .in_memory => |*im| im.precompiledCores(),
        };
    }

    fn instantiationOptions(self: *ComponentCoresSource) wamr.component_core_backend.Options {
        return switch (self.*) {
            .manifest => |*m| .{ .precompiled_cores = m.precompiledCores() },
            .in_memory => |*im| im.instantiationOptions(),
        };
    }

    fn deinit(self: *ComponentCoresSource) void {
        switch (self.*) {
            .manifest => |*m| m.deinit(),
            .in_memory => |*im| im.deinit(),
        }
    }
};

/// Resolve the AOT precompiled-cores manifest for a component, shared by
/// `wamr run` and `wamr serve` (#644 / #680 / #845).
///
/// The `wamr` CLI is AOT-only and the runtime no longer embeds the
/// compiler, so every component core must be available as a precompiled
/// `.cwasm` or instantiation fails hard — unless this binary was built
/// with `-Djit=true` (#852), in which case the last resolution step
/// compiles in memory instead of erroring. Resolution order:
///
///   * `--precompiled-manifest <path>` (explicit): any error opening /
///     validating the manifest is fatal so the user knows the AOT path
///     isn't being taken — this always reads from disk, even on a
///     `-Djit=true` build, since the user explicitly asked for that
///     artifact.
///   * sibling `<input>.cwasm.json` (auto-detect): used when present and
///     valid.
///   * neither found, `-Djit=true`: in-process JIT compile (#854) — no
///     manifest, no `.cwasm` files, zero filesystem I/O.
///   * neither found, default build: hard error directing the user at
///     `wamrc`.
///
/// On success returns 0 and sets `out_source`. On failure prints a
/// diagnostic and returns a non-zero exit code; `out_source` may still
/// be set (the caller's `defer …deinit()` handles cleanup either way).
/// `verb` is the invoking subcommand ("run" / "serve") and only flavours
/// the error text.
fn loadComponentManifestOrPrint(
    init: std.process.Init,
    allocator: std.mem.Allocator,
    verb: []const u8,
    path: []const u8,
    wasm_data: []const u8,
    precompiled_manifest: ?[]const u8,
    out_source: *?ComponentCoresSource,
) u8 {
    if (precompiled_manifest) |mp| {
        const loaded = wamr.component_aot.loadManifest(allocator, mp, wasm_data) catch |err| {
            std.debug.print("Error: --precompiled-manifest '{s}': {s}\n", .{ mp, loadManifestErrorMessage(err) });
            return 2;
        };
        out_source.* = .{ .manifest = loaded };
        const n = out_source.*.?.precompiledCores().len;
        if (wamr.component_core_backend.debugAotEnabled())
            std.debug.print("wamr: loaded AOT manifest from {s} ({d} core{s} precompiled)\n", .{ mp, n, if (n == 1) @as([]const u8, "") else "s" });
    } else {
        // Auto-probe `<input>.cwasm.json`. Without an embedded compiler
        // the absence of a sibling manifest is fatal — direct the user
        // at `wamrc`.
        const sibling = wamr.component_aot.defaultManifestPathFor(allocator, path) catch return 1;
        defer allocator.free(sibling);
        const io_probe = init.io;
        const cwd_probe = std.Io.Dir.cwd();
        const exists = blk: {
            var f = cwd_probe.openFile(io_probe, sibling, .{}) catch break :blk false;
            f.close(io_probe);
            break :blk true;
        };
        if (!exists) {
            if (comptime wamr.config.jit) {
                // #854: in-process JIT. Compile every core module of
                // the component in memory and instantiate directly —
                // no manifest, no `.cwasm` files written to disk.
                // #860: default to the fast/baseline compile preset
                // (see `jitPassPresetFromEnv`) rather than `wamrc
                // compile`'s steady-state-optimized default.
                const in_mem = wamr.component_aot_compile.precompileComponentInMemory(allocator, wasm_data, .{
                    .pass_preset = jitPassPresetFromEnv(init.environ_map),
                    .lazy_jit = comptime wamr.config.lazy_jit and builtin.cpu.arch == .x86_64,
                }) catch |err| {
                    std.debug.print("Error: JIT compile of component '{s}' failed: {s}\n", .{ path, @errorName(err) });
                    return 1;
                };
                out_source.* = .{ .in_memory = in_mem };
                const n = out_source.*.?.precompiledCores().len;
                if (wamr.component_core_backend.debugAotEnabled())
                    std.debug.print("wamr: JIT-compiled {d} core{s} for {s} (in-process, no disk artifact)\n", .{ n, if (n == 1) @as([]const u8, "") else "s", path });
            } else {
                std.debug.print(
                    "Error: no AOT manifest found for '{s}'. The `wamr` runtime no longer embeds the compiler (#680).\n" ++
                        "  Run `wamrc compile-component {s}` to produce '{s}', or\n" ++
                        "  run `wamrc {s} {s}` to compile and serve/execute in one step, or\n" ++
                        "  rebuild `wamr` with `-Djit=true` for in-process compile+run (#852/#854).\n",
                    .{ path, path, sibling, verb, path },
                );
                return 2;
            }
        } else {
            const loaded = wamr.component_aot.loadManifest(allocator, sibling, wasm_data) catch |err| {
                std.debug.print(
                    "Error: AOT manifest at '{s}' is stale or invalid: {s}.\n" ++
                        "  Rebuild it with `wamrc compile-component {s}` or run `wamrc {s} {s}`.\n",
                    .{ sibling, loadManifestErrorMessage(err), path, verb, path },
                );
                return 2;
            };
            out_source.* = .{ .manifest = loaded };
            const n = out_source.*.?.precompiledCores().len;
            if (wamr.component_core_backend.debugAotEnabled())
                std.debug.print("wamr: loaded AOT manifest from {s} ({d} core{s} precompiled)\n", .{ sibling, n, if (n == 1) @as([]const u8, "") else "s" });
        }
    }

    // #644: AOT-only policy. `wamr run`/`serve` require a precompiled
    // bundle for every component — there is no interp fallback at the CLI
    // surface, and no in-process compiler unless `-Djit=true`. An empty
    // bundle means the component had no core modules (an empty /
    // malformed input that survived parsing).
    if (out_source.*.?.precompiledCores().len == 0) {
        std.debug.print(
            "Error: component has no AOT-compiled cores and `wamr {s}` is AOT-only.\n" ++
                "  See issue #644.\n",
            .{verb},
        );
        return 2;
    }
    return 0;
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
    component_core_opts: wamr.component_core_backend.Options,
) u8 {
    const adapter_mod = wamr.wasi_cli_adapter;
    // Wire the adapter's stdio directly to the host process's
    // STDIN/STDOUT/STDERR so output streams live (no end-of-run
    // flush) and stdin reads from the user's terminal / piped
    // input (#474). For the test/embedding path that needs to
    // inspect captured stdout, use `WasiCliAdapter.init` instead.
    var adapter = adapter_mod.WasiCliAdapter.initWithHostStdio(allocator);
    defer adapter.deinit();
    adapter.setTraceCalls(wasi_cli_adapter_trace_enabled);
    adapter.setTraceVerbose(wasi_cli_adapter_trace_verbose);

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

    var instantiation_opts = component_core_opts;
    instantiation_opts.aot_only = true;
    const outcome = adapter_mod.runComponentBytes(data, arena_alloc, &adapter, instantiation_opts) catch |err| {
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
    tls_config: ?*wamr.wasi_cli_adapter.HttpsTlsConfig,
    component_core_opts: wamr.component_core_backend.Options,
) u8 {
    const adapter_mod = wamr.wasi_cli_adapter;
    var adapter = adapter_mod.WasiCliAdapter.init(allocator);
    defer adapter.deinit();
    adapter.setTraceCalls(wasi_cli_adapter_trace_enabled);
    adapter.setTraceVerbose(wasi_cli_adapter_trace_verbose);

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

    var instantiation_opts = component_core_opts;
    instantiation_opts.aot_only = true;
    adapter_mod.serveHttpComponentBytes(data, arena_alloc, &adapter, .{
        .listen_address = listen_address,
        .announce_listening = listen_address.getPort() == 0,
        .tls_config = tls_config,
    }, instantiation_opts) catch |err| {
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
            error.ListenFailed => std.debug.print("Error: failed to bind --addr address\n", .{}),
            error.AddressInUse => std.debug.print(
                "Error: --addr address already in use (another process is bound to this port)\n",
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

fn runAot(
    io: std.Io,
    allocator: std.mem.Allocator,
    data: []const u8,
    wasm_args: []const []const u8,
    env_flags: []const []const u8,
    environ_map: *const std.process.Environ.Map,
    map_dirs: []const MapDir,
) u8 {
    if (comptime aot_supported) {
        return runAotReal(io, allocator, data, wasm_args, env_flags, environ_map, map_dirs);
    } else {
        std.debug.print("Error: AOT execution not supported on this architecture\n", .{});
        return 1;
    }
}

fn runAotReal(
    io: std.Io,
    allocator: std.mem.Allocator,
    data: []const u8,
    wasm_args: []const []const u8,
    env_flags: []const []const u8,
    environ_map: *const std.process.Environ.Map,
    map_dirs: []const MapDir,
) u8 {
    const aot_loader = wamr.aot_loader;
    const aot_runtime = wamr.aot_runtime;

    // 1. Build the WasiCtx (args, env, preopens). Same lifetime model
    // as runComponent: borrows strings from caller-owned slices. This
    // mirrors the pre-#680 in-process `runCoreAot` setup so the AOT
    // host-bridge WASI adapters can resolve `vmctx.wasi_ctx`.
    var ctx = wamr.WasiCtx.init(allocator, io) catch |err| {
        std.debug.print("Error: failed to init WASI ctx: {s}\n", .{@errorName(err)});
        return 1;
    };
    defer ctx.deinit();

    // argv[0] is a stable placeholder ("wasm"); wasi-libc only uses
    // it for `program_invocation_name`, not for routing.
    var argv_buf = allocator.alloc([]const u8, 1 + wasm_args.len) catch return 1;
    defer allocator.free(argv_buf);
    argv_buf[0] = "wasm";
    for (wasm_args, 0..) |a, i| argv_buf[1 + i] = a;
    ctx.setArgs(argv_buf) catch return 1;

    var env_buf: std.ArrayListUnmanaged([]const u8) = .empty;
    defer env_buf.deinit(allocator);
    if (env_flags.len > 0) {
        env_buf.ensureTotalCapacity(allocator, env_flags.len) catch return 1;
        for (env_flags) |kv| env_buf.appendAssumeCapacity(kv);
    } else {
        var it = environ_map.array_hash_map.iterator();
        while (it.next()) |kv| {
            const joined = std.fmt.allocPrint(allocator, "{s}={s}", .{ kv.key_ptr.*, kv.value_ptr.* }) catch return 1;
            env_buf.append(allocator, joined) catch {
                allocator.free(joined);
                return 1;
            };
        }
    }
    defer if (env_flags.len == 0) {
        for (env_buf.items) |s| allocator.free(s);
    };
    ctx.setEnv(env_buf.items) catch return 1;

    for (map_dirs) |md| {
        _ = ctx.openMappedDir(md.host_path, md.guest_name) catch |err| {
            std.debug.print("Error: --map-dir '{s}::{s}': {s}\n", .{ md.host_path, md.guest_name, @errorName(err) });
            return 1;
        };
    }

    // 2. Load + instantiate the AOT image.
    const aot_module = aot_loader.load(data, allocator) catch |err| {
        std.debug.print("Error: failed to load AOT module: {}\n", .{err});
        return 1;
    };
    // Mirror the load/unload pairing used by every other `aot_loader.load`
    // call site in the repo (compiler/emit_aot.zig, tests/coldstart_test.zig,
    // tests/aot_harness.zig). Without this, owned slices on `AotModule`
    // leak (DebugAllocator prints to stderr on exit).
    defer aot_loader.unload(&aot_module, allocator);

    const aot_inst = aot_runtime.instantiate(&aot_module, allocator) catch |err| {
        std.debug.print("Error: failed to instantiate AOT module: {}\n", .{err});
        return 1;
    };
    defer aot_runtime.destroy(aot_inst);

    aot_runtime.mapCodeExecutable(aot_inst) catch |err| {
        std.debug.print("Error: failed to map code as executable: {}\n", .{err});
        return 1;
    };

    // 3. Retain the process-scoped WASI state on the AOT execution context.
    aot_inst.attachProcessState(ctx.processStateRef());

    const func_idx = aot_runtime.findExportFunc(aot_inst, "_start") orelse
        aot_runtime.findExportFunc(aot_inst, "main") orelse {
        std.debug.print("Error: no _start or main function exported in AOT module\n", .{});
        return 1;
    };

    // `_start` / `main` take no params and return no values for the
    // wasi-libc CRT shape; route through `callFuncScalar` with empty
    // slices. Any guest call to `proc_exit` short-circuits via
    // `std.process.exit` inside the host bridge.
    var results_buf: [0]wamr.aot_runtime.ScalarResult = .{};
    _ = aot_runtime.callFuncScalar(aot_inst, func_idx, &.{}, &.{}, &.{}, &results_buf) catch |err| {
        std.debug.print("Error: AOT execution failed: {}\n", .{err});
        return 1;
    };

    if (ctx.getExitCode()) |code| return @intCast(code & 0xFF);
    return 0;
}

fn writeStdout(io: std.Io, text: []const u8) void {
    var stdout_file = std.Io.File.stdout();
    stdout_file.writeStreamingAll(io, text) catch {};
}

/// Map a `component_aot.LoadError` to a short, user-facing
/// diagnostic string. Used by `runRun` to surface manifest load
/// failures in a uniform format.
fn loadManifestErrorMessage(err: wamr.component_aot.LoadError) []const u8 {
    return switch (err) {
        error.ManifestNotFound => "manifest sidecar not found at the given path",
        error.ManifestParseFailed => "manifest sidecar could not be parsed",
        error.ManifestVersionMismatch => "manifest format version not understood by this build",
        error.ManifestBuildIdMismatch => "manifest was produced by a different wamr build (recompile with `wamrc compile-component`)",
        error.ManifestComponentMismatch => "manifest's component hash does not match this component (stale — recompile with `wamrc compile-component`)",
        error.ManifestCoreMismatch => "manifest references a core module that does not appear in this component (stale — recompile with `wamrc compile-component`)",
        error.ComponentParseFailed => "manifest is valid but the component bytes failed to re-parse for nested-core resolution",
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
    \\  serve     Serve a wasi:http/incoming-handler component over HTTP
    \\  version   Print version and exit
    \\  help      Print this help
    \\
    \\Run `wamr <subcommand> help` to show help for a specific subcommand.
    \\
;

const run_usage_options =
    \\Options:
    \\  --stack-size=<bytes>     (ignored; kept for backward compat)
    \\  --heap-size=<bytes>      Reserved (currently ignored)
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
    \\  --precompiled-manifest PATH
    \\                           For components: load AOT-compiled cores by
    \\                           reading a `wamrc compile-component` manifest
    \\                           sidecar (`<stem>.cwasm.json`). Per-core
    \\                           `.cwasm` files resolve relative to the
    \\                           manifest's directory. When omitted, `wamr
    \\                           run` auto-detects a sibling
    \\                           `<input>.cwasm.json`.
    \\
    \\To serve a wasi:http/incoming-handler component over HTTP, use
    \\`wamr serve` instead (see `wamr serve help`).
    \\
;

// #855: `wamr run help` describes the behavior of *this specific binary*,
// which is a comptime-known build (`wamr.config.jit`) — so rather than
// one generic paragraph hedging over both configurations, `run_usage` is
// comptime-selected between two accurate intros. See README.md's "JIT
// mode" section for the side-by-side comparison of both build flavors.
const run_usage_intro_aot_only =
    \\Usage: wamr run [options] <file.wasm|file.cwasm> [args...]
    \\
    \\This `wamr` binary is AOT-only (no `-Djit` build support compiled
    \\in; issue #644):
    \\  * `.cwasm`/`.aot` core modules (magic `\0aot`) run via the AOT
    \\    runtime directly.
    \\  * Component-model `.wasm` files run via AOT cores loaded from a
    \\    `wamrc compile-component` manifest (see --precompiled-manifest
    \\    below). Components without a manifest fail with a clear error.
    \\  * Plain core wasm modules are not supported — pre-compile to
    \\    `.cwasm`/`.aot` first with `wamrc compile`, or use `wamrc run`
    \\    to compile and execute in one step.
    \\
    \\For a `wasmtime run`-style one-shot experience (`wamr run foo.wasm`
    \\compiles and executes in one process, no `wamrc` step, no `.cwasm`
    \\artifact), rebuild `wamr` with `-Djit=true` (issue #852).
    \\
    \\
;

const run_usage_intro_jit =
    \\Usage: wamr run [options] <file.wasm|file.cwasm> [args...]
    \\
    \\This `wamr` binary was built with in-process JIT support
    \\(`-Djit=true`; issues #852-#854):
    \\  * Plain core `.wasm` modules are compiled in memory and executed
    \\    in this same process — no `.cwasm` file is written, no `wamrc`
    \\    subprocess is spawned.
    \\  * Component-model `.wasm` files: a sibling `wamrc compile-component`
    \\    manifest (or an explicit --precompiled-manifest below) is used
    \\    when present; otherwise every core module is JIT-compiled in
    \\    memory and instantiated directly — same zero-disk-artifact
    \\    behavior as the core-wasm case.
    \\  * `.cwasm`/`.aot` core modules (magic `\0aot`) still run via the
    \\    AOT runtime directly, same as an AOT-only build.
    \\
    \\
;

const run_usage = if (wamr.config.jit) run_usage_intro_jit ++ run_usage_options else run_usage_intro_aot_only ++ run_usage_options;

const serve_usage_options =
    \\Options:
    \\  --addr <ip:port>         Bind address (an IP literal + port, no
    \\                           hostname resolution). Default 127.0.0.1:8080.
    \\                           Use `0.0.0.0:<port>` / `[::1]:<port>` for
    \\                           broader binding. `--addr 127.0.0.1:0`
    \\                           requests a kernel-assigned ephemeral port
    \\                           and prints the resolved address to stdout
    \\                           (handy for test drivers).
    \\  --env KEY=VALUE          Set a WASI environment variable (repeatable)
    \\  --log-level NAME         Filter `wasi:logging` calls below NAME. One
    \\                           of trace|debug|info|warn|error|critical
    \\                           (default: trace = admit all). Falls back to
    \\                           the WAMR_LOG_LEVEL env var when absent.
    \\  --tls-cert PATH          PEM-encoded certificate chain (leaf first).
    \\                           Terminates HTTPS on each connection. Requires
    \\                           a matching --tls-key.
    \\  --tls-key PATH           PEM-encoded private key (PKCS#8, RSA, or EC).
    \\                           Pairs with --tls-cert.
    \\  --tls-pem PATH           Combined PEM file containing both certificate
    \\                           chain and private key. Mutually exclusive
    \\                           with --tls-cert / --tls-key.
    \\  --precompiled-manifest PATH
    \\                           Load AOT-compiled cores from a `wamrc
    \\                           compile-component` manifest sidecar
    \\                           (`<stem>.cwasm.json`). When omitted, a
    \\                           sibling `<input>.cwasm.json` is auto-detected.
    \\
;

const serve_usage_intro_aot_only =
    \\Usage: wamr serve [options] <component.wasm>
    \\
    \\Serve a `wasi:http/incoming-handler` component as a long-lived HTTP
    \\server (the proxy world), aligned with `wasmtime serve`. This
    \\`wamr` binary is AOT-only (no `-Djit` build support compiled in):
    \\the component needs a `wamrc compile-component` manifest (an
    \\explicit --precompiled-manifest or a sibling `<input>.cwasm.json`).
    \\Use `wamrc serve <component.wasm>` to precompile-if-stale and serve
    \\in one step, or rebuild `wamr` with `-Djit=true` (issue #852) to
    \\serve directly without a manifest.
    \\
    \\
;

const serve_usage_intro_jit =
    \\Usage: wamr serve [options] <component.wasm>
    \\
    \\Serve a `wasi:http/incoming-handler` component as a long-lived HTTP
    \\server (the proxy world), aligned with `wasmtime serve`. This
    \\`wamr` binary was built with in-process JIT support (`-Djit=true`;
    \\issue #854): a sibling `wamrc compile-component` manifest (or an
    \\explicit --precompiled-manifest below) is used when present;
    \\otherwise every core module is JIT-compiled in memory and served
    \\directly — no manifest, no `.cwasm` files written to disk.
    \\
    \\
;

const serve_usage = if (wamr.config.jit) serve_usage_intro_jit ++ serve_usage_options else serve_usage_intro_aot_only ++ serve_usage_options;

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

// #852: proves `-Djit=true` actually threads through to a real, reachable
// reference to the in-process compiler from `main.zig`'s module graph —
// not just a config plumbing exercise. `wamr.config.jit` is a comptime-known
// build option, so under the default `-Djit=false` build Zig's comptime
// branch elimination drops everything below the early return, and this
// test (like the rest of `zig build test`'s output) never links
// `component_aot_compile` into the *production* `wamr`/`wamrc` executables
// either way — those are built from a separate, test-free module (see
// `exe_module` vs `exe_test_module` in build.zig). Full end-to-end
// `wamr run foo.wasm` in-process compile+execute CLI wiring is #853/#854;
// this only proves the flag is live.
test "config.jit gates in-process compiler reachability from main.zig (#852)" {
    if (!wamr.config.jit) return error.SkipZigTest;

    // Minimal valid core wasm module: magic + version, no sections.
    const empty_core_wasm = [_]u8{ 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00 };
    const cwasm = try wamr.component_aot_compile.compileCoreWasm(std.testing.allocator, &empty_core_wasm, .{});
    defer std.testing.allocator.free(cwasm);
    try std.testing.expect(cwasm.len >= 8);
    try std.testing.expectEqual(@as(u32, wamr.emit_aot.aot_magic), std.mem.readInt(u32, cwasm[0..4], .little));
}

test "subcommand parsing" {
    try std.testing.expectEqual(@as(?Subcommand, .run), parseSubcommand("run"));
    try std.testing.expectEqual(@as(?Subcommand, .serve), parseSubcommand("serve"));
    try std.testing.expectEqual(@as(?Subcommand, .version), parseSubcommand("version"));
    try std.testing.expectEqual(@as(?Subcommand, .help), parseSubcommand("help"));
    try std.testing.expectEqual(@as(?Subcommand, null), parseSubcommand("--version"));
    try std.testing.expectEqual(@as(?Subcommand, null), parseSubcommand("foo.wasm"));
    try std.testing.expectEqual(@as(?Subcommand, null), parseSubcommand(""));
}

test "parseAddr accepts ipv4/ipv6 literals and rejects junk" {
    const a = try parseAddr("127.0.0.1:8080");
    try std.testing.expectEqual(@as(u16, 8080), a.getPort());
    const eph = try parseAddr("127.0.0.1:0");
    try std.testing.expectEqual(@as(u16, 0), eph.getPort());
    const v6 = try parseAddr("[::1]:8080");
    try std.testing.expectEqual(@as(u16, 8080), v6.getPort());
    try std.testing.expectError(error.InvalidAddress, parseAddr(""));
    try std.testing.expectError(error.InvalidAddress, parseAddr("127.0.0.1"));
    try std.testing.expectError(error.InvalidAddress, parseAddr(":8080"));
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
