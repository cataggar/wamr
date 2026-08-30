const std = @import("std");
const threads_feature = @import("src/threads_feature.zig");

const fuzz_seed_wasms = [_][]const u8{
    "diamond_call_indirect_then_load.wasm",
    "loop_backedge_barrier.wasm",
    "loop_body_call.wasm",
    "triangle_store.wasm",
    "nested_ifs_mixed.wasm",
    "indirect_vs_direct_call.wasm",
    "atomic_barrier.wasm",
    "bulk_memory_barrier.wasm",
    "tail_call_barrier.wasm",
    "multi_memory_forwarding.wasm",
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Whether the selected target CPU can execute AOT code natively.
    // Test/bench binaries tied to native AOT support are only installed on
    // these arches so cross-compiled release builds for e.g. riscv64 don't
    // try to compile or run the AOT execution path.
    const target_arch = target.result.cpu.arch;
    const is_wasm_wasi_host = target.result.os.tag == .wasi and
        (target_arch == .wasm32 or target_arch == .wasm64);
    const aot_executable_target = switch (target_arch) {
        .x86_64, .aarch64 => true,
        else => false,
    };
    // Subset of `aot_executable_target` where the host-import trampoline
    // pool is supported. Mirrors `supports_pool` in
    // `src/runtime/aot/host_trampolines.zig:341`: Windows can't mmap RWX
    // pages the same way, and macOS aarch64 needs MAP_JIT +
    // `pthread_jit_write_protect_np` plumbing that isn't wired. The
    // canon-lower (aot) dispatcher path — and therefore any
    // component-importing-host-fn end-to-end test — needs this.
    const aot_trampoline_pool_target = aot_executable_target and
        target.result.os.tag != .windows and
        !(target.result.os.tag == .macos and target_arch == .aarch64);
    // codegen-bench emits x86-64 machine code but does not execute it. It can
    // run on the native AOT arches with a portable timer fallback.
    const bench_target = aot_executable_target;

    // ── Build flags ────────────────────────────────────────────────────
    const strip = b.option(bool, "strip", "Strip debug info from binaries") orelse false;
    const stack_protector = b.option(bool, "stack-protector", "Enable stack protector (requires libc)") orelse false;
    const link_libc = b.option(bool, "link-libc", "Link libc") orelse
        (stack_protector or target.result.os.tag == .wasi);
    const version_string = b.option([]const u8, "version", "Version string") orelse "dev";
    const ir_property_iterations = b.option(u32, "ir-property-iters", "IR optimizer property-test iterations per shape/pass") orelse 8;

    // ── Feature flags ──────────────────────────────────────────────────
    const options = b.addOptions();
    options.addOption([]const u8, "version", version_string);

    const interp = b.option(bool, "interp", "Enable interpreter") orelse true;
    options.addOption(bool, "interp", interp);

    const aot = b.option(bool, "aot", "Enable AOT support") orelse !is_wasm_wasi_host;
    options.addOption(bool, "aot", aot);

    const fast_interp = b.option(bool, "fast_interp", "Enable fast interpreter") orelse true;
    options.addOption(bool, "fast_interp", fast_interp);

    const jit = b.option(bool, "jit", "Enable the in-process JIT (compile+run a .wasm in one step, opt-in — see #852)") orelse false;
    options.addOption(bool, "jit", jit);

    // #862: lazy per-function on-demand compilation spike, layered on
    // top of -Djit=true. Off by default even when -Djit=true is set —
    // this is a narrow, leaf-functions-only prototype, not a general
    // feature (see docs/design/lazy-jit-spike.md).
    const lazy_jit = b.option(bool, "lazy_jit", "Enable the lazy-JIT leaf-function spike on top of -Djit=true (#862)") orelse false;
    options.addOption(bool, "lazy_jit", lazy_jit);

    const fast_jit = b.option(bool, "fast_jit", "Enable fast JIT") orelse false;
    options.addOption(bool, "fast_jit", fast_jit);

    const wamr_compiler = b.option(bool, "wamr_compiler", "Enable AOT compiler") orelse false;
    options.addOption(bool, "wamr_compiler", wamr_compiler);

    const libc_builtin = b.option(bool, "libc_builtin", "Enable built-in libc") orelse true;
    options.addOption(bool, "libc_builtin", libc_builtin);

    const libc_wasi = b.option(bool, "libc_wasi", "Enable WASI libc") orelse true;
    options.addOption(bool, "libc_wasi", libc_wasi);

    const simd = b.option(bool, "simd", "Enable SIMD support") orelse true;
    options.addOption(bool, "simd", simd);

    const ref_types = b.option(bool, "ref_types", "Enable reference types") orelse true;
    options.addOption(bool, "ref_types", ref_types);

    const multi_module = b.option(bool, "multi_module", "Enable multi-module") orelse false;
    options.addOption(bool, "multi_module", multi_module);

    const lib_pthread = b.option(bool, "lib_pthread", "Enable pthread library") orelse false;
    options.addOption(bool, "lib_pthread", lib_pthread);

    const lib_wasi_threads = b.option(bool, "lib_wasi_threads", "Enable the WASI threads configuration contract (production spawning is not implemented)") orelse false;
    options.addOption(bool, "lib_wasi_threads", lib_wasi_threads);

    const thread_mgr = (b.option(bool, "thread_mgr", "Enable thread manager") orelse false) or lib_wasi_threads;
    options.addOption(bool, "thread_mgr", thread_mgr);

    const debug_interp = b.option(bool, "debug_interp", "Enable interpreter debugging") orelse false;
    options.addOption(bool, "debug_interp", debug_interp);

    const bulk_memory = b.option(bool, "bulk_memory", "Enable bulk memory ops") orelse false;
    options.addOption(bool, "bulk_memory", bulk_memory);

    const shared_memory = (b.option(bool, "shared_memory", "Enable shared memory") orelse false) or lib_wasi_threads;
    options.addOption(bool, "shared_memory", shared_memory);

    const wasm_atomics = shared_memory or lib_wasi_threads;
    options.addOption(bool, "wasm_atomics", wasm_atomics);

    const heap_aux_stack_allocation = (b.option(bool, "heap_aux_stack_allocation", "Allocate auxiliary WASM stacks on the heap") orelse false) or lib_wasi_threads;
    options.addOption(bool, "heap_aux_stack_allocation", heap_aux_stack_allocation);

    const tail_call = b.option(bool, "tail_call", "Enable tail call") orelse false;
    options.addOption(bool, "tail_call", tail_call);

    const gc = b.option(bool, "gc", "Enable garbage collection") orelse false;
    options.addOption(bool, "gc", gc);

    const memory64 = b.option(bool, "memory64", "Enable memory64") orelse false;
    options.addOption(bool, "memory64", memory64);

    const multi_memory = b.option(bool, "multi_memory", "Enable multi-memory") orelse false;
    options.addOption(bool, "multi_memory", multi_memory);

    const exce_handling = b.option(bool, "exce_handling", "Enable exception handling") orelse false;
    options.addOption(bool, "exce_handling", exce_handling);

    const shared_heap = b.option(bool, "shared_heap", "Enable shared heap") orelse false;
    options.addOption(bool, "shared_heap", shared_heap);

    const wasi_nn = b.option(bool, "wasi_nn", "Enable WASI neural network") orelse false;
    options.addOption(bool, "wasi_nn", wasi_nn);

    const component_model = b.option(bool, "component_model", "Enable Component Model") orelse false;
    options.addOption(bool, "component_model", component_model);

    if (is_wasm_wasi_host) {
        if (!interp)
            std.debug.panic("wasm32-wasi host builds require the interpreter", .{});
        if (aot)
            std.debug.panic("wasm32-wasi host builds do not support the native AOT runtime", .{});
        if (jit or fast_jit or wamr_compiler)
            std.debug.panic("wasm32-wasi host builds do not support native code generation", .{});
        if (component_model)
            std.debug.panic("wasm32-wasi host builds do not support Component Model execution", .{});
        if (lib_pthread or lib_wasi_threads or thread_mgr or shared_memory)
            std.debug.panic("wasm32-wasi host builds do not support host threads or shared memory", .{});
    }

    const threads_inputs = threads_feature.Inputs{
        .enabled = lib_wasi_threads,
        .pointer_bits = target.result.ptrBitWidth(),
        .wasm_host = switch (target_arch) {
            .wasm32, .wasm64 => true,
            else => false,
        },
        .single_threaded = target.result.os.tag == .freestanding,
        .interp = interp,
        .aot = aot,
        .jit = jit,
        .fast_jit = fast_jit,
        .libc_wasi = libc_wasi,
        .heap_aux_stack_allocation = heap_aux_stack_allocation,
        .shared_memory = shared_memory,
        .thread_manager = thread_mgr,
        .wasm_atomics = wasm_atomics,
    };
    if (threads_feature.validationError(threads_inputs)) |err| {
        std.debug.panic("invalid WASI threads configuration: {s}", .{threads_feature.validationMessage(err)});
    }

    const network_tests = b.option(
        bool,
        "network_tests",
        "Enable opt-in unit tests that perform real outbound HTTPS requests (#521)",
    ) orelse false;
    options.addOption(bool, "network_tests", network_tests);

    const skip_coldstart = b.option(bool, "skip-coldstart", "Skip cold-start budget tests (issue #395)") orelse false;
    const verify_ir_triage = b.option(bool, "verify-ir-triage", "Run differential tests with the IR verifier enabled and print per-test verifier failures (issue #627)") orelse false;
    const wamr_strict_canon = b.option(bool, "wamr-strict-canon", "Enable strict canonical ABI ptr/len diagnostics") orelse switch (optimize) {
        .ReleaseFast => false,
        else => true,
    };
    options.addOption(bool, "wamr_strict_canon", wamr_strict_canon);
    const aot_broken_components = b.option(
        bool,
        "aot-broken-components",
        "Run `examples-run` / `wasi-p2-testsuite` fixtures that currently fail under AOT-only `wamr run` (#662). Defaults true so local `zig build examples-run` still exercises them; CI sets to false to keep the gate green.",
    ) orelse true;

    const config_module = options.createModule();

    // A WebAssembly host cannot execute the native code produced by WAMR's
    // AOT backends. Build a deliberately reduced, interpreter-only CLI and
    // keep native-only dependencies out of the module graph entirely.
    if (is_wasm_wasi_host) {
        const wasm_host_module = b.createModule(.{
            .root_source_file = b.path("src/main_wasm_host.zig"),
            .target = target,
            .optimize = optimize,
            .strip = if (strip) true else null,
            .stack_protector = if (stack_protector) true else null,
            .link_libc = if (link_libc) true else null,
        });
        wasm_host_module.addImport("config", config_module);

        const wasm_host_exe = b.addExecutable(.{
            .name = "wamr-exe",
            .root_module = wasm_host_module,
        });
        const install_wasm_host = b.addInstallArtifact(wasm_host_exe, .{
            .dest_sub_path = b.fmt("wamr{s}", .{target.result.exeFileExt()}),
        });
        b.getInstallStep().dependOn(&install_wasm_host.step);
        return;
    }

    const stable_resources_test_step = b.step(
        "test-stable-resources",
        "Run conditional synchronization and stable resource tests",
    );
    inline for (.{ false, true }) |threads_enabled| {
        const stable_options = b.addOptions();
        stable_options.addOption(bool, "lib_wasi_threads", threads_enabled);
        const stable_test_module = b.createModule(.{
            .root_source_file = b.path("src/shared/stable_resource.zig"),
            .target = target,
            .optimize = optimize,
        });
        stable_test_module.addImport("config", stable_options.createModule());
        const stable_tests = b.addTest(.{ .root_module = stable_test_module });
        const run_stable_tests = b.addRunArtifact(stable_tests);
        stable_resources_test_step.dependOn(&run_stable_tests.step);
    }

    const threads_contract_test_module = b.createModule(.{
        .root_source_file = b.path("src/config.zig"),
        .target = target,
        .optimize = optimize,
    });
    threads_contract_test_module.addImport("config", config_module);
    const threads_contract_tests = b.addTest(.{
        .root_module = threads_contract_test_module,
    });
    const run_threads_contract_tests = b.addRunArtifact(threads_contract_tests);
    const threads_contract_test_step = b.step(
        "test-threads-contract",
        "Run WASI threads feature-contract tests",
    );
    threads_contract_test_step.dependOn(&run_threads_contract_tests.step);

    // ── TLS library (cataggar/tls.zig, zig16 branch) ──────────────────
    // Pure-Zig TLS 1.2/1.3 client + server. Used for `wasi:http@0.3`
    // incoming-handler HTTPS termination (#609): upstream Zig std ships
    // only `std.crypto.tls.Client`, so the server-side handshake comes
    // from this dependency. Exposes the `tls` module via `addModule`.
    const tls_dep = b.dependency("tls", .{
        .target = target,
        .optimize = optimize,
    });
    const tls_module = tls_dep.module("tls");

    const threads_runtime_test_module = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    threads_runtime_test_module.addImport("config", config_module);
    threads_runtime_test_module.addImport("tls", tls_module);
    const threads_runtime_tests = b.addTest(.{
        .root_module = threads_runtime_test_module,
        .filters = &.{
            "instantiate rejects configured WASI threads",
            "host functions resolved for wasi thread-spawn import",
        },
    });
    const run_threads_runtime_tests = b.addRunArtifact(threads_runtime_tests);
    threads_contract_test_step.dependOn(&run_threads_runtime_tests.step);

    // ── Root module for the library ────────────────────────────────────
    const lib_module = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_module.addImport("config", config_module);
    lib_module.addImport("tls", tls_module);

    // ── Static library ─────────────────────────────────────────────────
    const lib = b.addLibrary(.{
        .name = "wamr",
        .root_module = lib_module,
    });
    b.installArtifact(lib);

    // ── wamr executable ────────────────────────────────────────────────
    const exe_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .strip = if (strip) true else null,
        .stack_protector = if (stack_protector) true else null,
        .link_libc = if (link_libc) true else null,
    });
    exe_module.addImport("config", config_module);
    exe_module.addImport("wamr", lib_module);

    const exe = b.addExecutable(.{
        .name = "wamr-exe",
        .root_module = exe_module,
    });
    // Keep dependency artifact names unique without changing the installed CLI name.
    const install_exe = b.addInstallArtifact(exe, .{
        .dest_sub_path = b.fmt("wamr{s}", .{target.result.exeFileExt()}),
    });
    b.getInstallStep().dependOn(&install_exe.step);

    // ── wamrc AOT compiler ────────────────────────────────────────────
    const wamrc_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/main.zig"),
        .target = target,
        .optimize = optimize,
        .strip = if (strip) true else null,
        .stack_protector = if (stack_protector) true else null,
        .link_libc = if (link_libc) true else null,
    });
    wamrc_module.addImport("config", config_module);
    wamrc_module.addImport("wamr", lib_module);

    const wamrc = b.addExecutable(.{
        .name = "wamrc",
        .root_module = wamrc_module,
    });
    b.installArtifact(wamrc);

    // ── Spec test runner ─────────────────────────────────────────────
    // wabt v3.0.0-dev.4 ships both a library and a CLI exe both named "wabt",
    // so `wabt_dep.artifact("wabt")` panics with "artifact name 'wabt' is
    // ambiguous", and the package no longer calls `b.addModule(...)` either.
    // Build the wabt library module ourselves from the dep's `src/root.zig`,
    // wired with a synthetic `build_options.version` (the only build option
    // wabt's root.zig consumes).
    const wabt_dep = b.dependency("wabt", .{
        .target = target,
        .optimize = .ReleaseSafe,
    });

    const wabt_build_options = b.addOptions();
    wabt_build_options.addOption([]const u8, "version", "v3.0.0-dev.4");

    const wabt_module = b.createModule(.{
        .root_source_file = wabt_dep.path("src/root.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    wabt_module.addImport("build_options", wabt_build_options.createModule());

    const spec_runner_module = b.createModule(.{
        .root_source_file = b.path("src/tests/run_spec_tests.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    spec_runner_module.addImport("config", config_module);
    spec_runner_module.addImport("wamr", lib_module);
    spec_runner_module.addImport("wabt", wabt_module);

    const spec_runner_exe = b.addExecutable(.{
        .name = "spec-test-runner",
        .root_module = spec_runner_module,
    });
    if (aot_executable_target) b.installArtifact(spec_runner_exe);

    // Run the spec suite through the AOT pipeline. Non-blocking convenience
    // step; not wired into the default `test` aggregate while codegen gaps
    // and the skiplist stabilize (see src/tests/aot_skiplist.zig).
    const run_spec_aot = b.addRunArtifact(spec_runner_exe);
    run_spec_aot.addArg("--mode=aot");
    run_spec_aot.addArg("tests/spec-json");
    const spec_aot_step = b.step("spec-tests-aot", "Run the spec-json suite through the AOT pipeline");
    spec_aot_step.dependOn(&run_spec_aot.step);

    // Full WebAssembly/spec conformance suite (257 .wast files, ~65k
    // assertions) from the vendored `third_party/testsuite` submodule,
    // gated against `tests/spec-baseline.tsv` so any regression fails CI.
    // Requires: git submodule update --init third_party/testsuite
    const run_spec_baseline = b.addSystemCommand(&.{
        "python3",
        "scripts/check_spec_baseline.py",
    });
    run_spec_baseline.step.dependOn(b.getInstallStep());
    const spec_baseline_step = b.step(
        "spec-testsuite",
        "Run the full WebAssembly/spec suite and check it against tests/spec-baseline.tsv",
    );
    spec_baseline_step.dependOn(&run_spec_baseline.step);

    // ── WASI conformance suite ────────────────────────────────────────
    // Drives the vendored `WebAssembly/wasi-testsuite` against the just-built
    // `wamr` CLI through the in-tree adapter. Skiplist entries must each
    // carry a rationale + follow-up issue number — see
    // `tests/wasi-testsuite-expectations.toml`. Not wired into the default `test`
    // aggregate (it requires Python 3 + the runner's deps), but the CI job
    // gates regressions on every PR. Run locally with `zig build wasi-testsuite`.
    // The runner entry point is launched through
    // `tests/wasi-testsuite-runner-patch/wasi_test_runner.py`, which
    // imports the upstream package without requiring installation.
    // The wamr adapter's `get_timeout_seconds` hook honours the
    // `WAMR_TESTSUITE_TIMEOUT` env var (seconds; defaults to 5s when
    // unset). Without that override, slow developer VMs flake on
    // `http-fields` and other fixtures whose runtime drifts past 5s.
    // See #583 A7 + the wrapper's module docstring for rationale.
    const wasi_runner = b.addSystemCommand(&.{
        "python3",
        "tests/wasi-testsuite-runner-patch/wasi_test_runner.py",
        "--test-suite",
        "tests/wasi-testsuite/tests/c/testsuite/wasm32-wasip1",
        "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip1",
        "tests/wasi-testsuite/tests/assemblyscript/testsuite/wasm32-wasip1",
        "--runtime-adapter",
        "tests/wasi-testsuite-adapter/wamr-zig.py",
        "--expectations",
        "tests/wasi-testsuite-expectations.toml",
    });
    // Point the adapter at the freshly-installed wamr binary so we don't pick
    // up a stale system iwasm.
    wasi_runner.setEnvironmentVariable("WAMR", b.getInstallPath(.bin, "wamr"));
    wasi_runner.setEnvironmentVariable("WAMRC", b.getInstallPath(.bin, "wamrc"));
    wasi_runner.step.dependOn(b.getInstallStep());
    const wasi_testsuite_step = b.step(
        "wasi-testsuite",
        "Run the WebAssembly/wasi-testsuite conformance suite",
    );
    wasi_testsuite_step.dependOn(&wasi_runner.step);

    // ── WASI Preview 3 conformance gate (#489) ────────────────────────
    // Drives the vendored `wasm32-wasip3` fixtures at
    // `tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip3/` through
    // the just-built `wamr` CLI via the same in-tree adapter as the
    // Preview 1 gate. Acts as a CI gate against regressions in the WASI
    // Preview 3 adapter surface (`src/component/wasi_cli_adapter.zig`,
    // P3 wave A–C: #481–#487). The expectations file is currently empty;
    // every one of the 41 fixtures is required to pass.
    // Not wired into the default `test` aggregate (it requires Python 3
    // + the runner's deps); CI gates regressions on every PR. Run
    // locally with `zig build wasi-p3-testsuite`.
    const wasi_p3_runner = b.addSystemCommand(&.{
        "python3",
        "tests/wasi-testsuite-runner-patch/wasi_test_runner.py",
        "--test-suite",
        "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip3",
        "--runtime-adapter",
        "tests/wasi-testsuite-adapter/wamr-zig.py",
        "--expectations",
        "tests/wasi-p3-testsuite-expectations.toml",
    });
    wasi_p3_runner.setEnvironmentVariable("WAMR", b.getInstallPath(.bin, "wamr"));
    wasi_p3_runner.setEnvironmentVariable("WAMRC", b.getInstallPath(.bin, "wamrc"));
    wasi_p3_runner.step.dependOn(b.getInstallStep());
    const wasi_p3_testsuite_step = b.step(
        "wasi-p3-testsuite",
        "Run the WASI Preview 3 conformance gate (wasm32-wasip3 fixtures)",
    );
    wasi_p3_testsuite_step.dependOn(&wasi_p3_runner.step);

    // This is intentionally a separate no-filter path. Its Python harness
    // asserts every fixture executed and passed, so a future all-skipped
    // filter cannot make the P3 gate look green.
    const wasi_p3_unfiltered_runner = b.addSystemCommand(&.{
        "python3",
        "tests/wasi-p3-unfiltered.py",
    });
    wasi_p3_unfiltered_runner.setEnvironmentVariable("WAMR", b.getInstallPath(.bin, "wamr"));
    wasi_p3_unfiltered_runner.setEnvironmentVariable("WAMRC", b.getInstallPath(.bin, "wamrc"));
    wasi_p3_unfiltered_runner.setEnvironmentVariable("WAMR_TESTSUITE_TIMEOUT", "30");
    wasi_p3_unfiltered_runner.step.dependOn(b.getInstallStep());
    const wasi_p3_unfiltered_step = b.step(
        "wasi-p3-testsuite-unfiltered",
        "Run all WASI Preview 3 fixtures and assert the unfiltered execution contract",
    );
    wasi_p3_unfiltered_step.dependOn(&wasi_p3_unfiltered_runner.step);

    // ── In-process JIT parity gates (#856) ────────────────────────────
    // Only meaningful on a `-Djit=true` build (see #852): reruns the
    // exact same P1 / P3 conformance corpora and skip-lists as
    // `wasi-testsuite` / `wasi-p3-testsuite` above, but with
    // `WAMR_JIT_TESTSUITE=1` telling the adapter's `_precompile` to
    // skip `wamrc` entirely and hand the raw `.wasm` straight to
    // `wamr run` / `wamr serve`, which JIT-compiles it in memory. Both
    // gates should report the *exact same* pass/skip counts as their
    // AOT-precompiled siblings — same compiler, same AOT loader/runtime,
    // only the "compile ahead of time to disk" vs "compile in memory on
    // first use" timing differs. A divergence here means the JIT path
    // has a real behavioral bug, not a pre-existing AOT compiler gap
    // (those are already accounted for by the shared skip-lists).
    //
    // Gated on `if (jit)` so these steps don't even exist on a default
    // build — running them there would just hard-error on every
    // fixture (issue #644/#680's AOT-only policy has no JIT fallback).
    if (jit) {
        const wasi_runner_jit = b.addSystemCommand(&.{
            "python3",
            "tests/wasi-testsuite-runner-patch/wasi_test_runner.py",
            "--test-suite",
            "tests/wasi-testsuite/tests/c/testsuite/wasm32-wasip1",
            "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip1",
            "tests/wasi-testsuite/tests/assemblyscript/testsuite/wasm32-wasip1",
            "--runtime-adapter",
            "tests/wasi-testsuite-adapter/wamr-zig.py",
            "--expectations",
            "tests/wasi-testsuite-expectations.toml",
        });
        wasi_runner_jit.setEnvironmentVariable("WAMR", b.getInstallPath(.bin, "wamr"));
        wasi_runner_jit.setEnvironmentVariable("WAMRC", b.getInstallPath(.bin, "wamrc"));
        wasi_runner_jit.setEnvironmentVariable("WAMR_JIT_TESTSUITE", "1");
        wasi_runner_jit.step.dependOn(b.getInstallStep());
        const wasi_testsuite_jit_step = b.step(
            "wasi-testsuite-jit",
            "Run the WASI Preview 1 conformance suite through the in-process JIT path (-Djit=true; #856)",
        );
        wasi_testsuite_jit_step.dependOn(&wasi_runner_jit.step);

        const wasi_p3_runner_jit = b.addSystemCommand(&.{
            "python3",
            "tests/wasi-testsuite-runner-patch/wasi_test_runner.py",
            "--test-suite",
            "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip3",
            "--runtime-adapter",
            "tests/wasi-testsuite-adapter/wamr-zig.py",
            "--expectations",
            "tests/wasi-p3-testsuite-expectations.toml",
        });
        wasi_p3_runner_jit.setEnvironmentVariable("WAMR", b.getInstallPath(.bin, "wamr"));
        wasi_p3_runner_jit.setEnvironmentVariable("WAMRC", b.getInstallPath(.bin, "wamrc"));
        wasi_p3_runner_jit.setEnvironmentVariable("WAMR_JIT_TESTSUITE", "1");
        // JIT compiles every fixture from scratch on every invocation
        // (no cross-process cache like the `.cwasm` mtime check the
        // AOT path gets), so per-test wall time is higher than the
        // AOT-precompiled gate above. Bump the runner's per-`wait`
        // timeout accordingly (see #583 A7 / README's
        // `WAMR_TESTSUITE_TIMEOUT` docs) rather than risk flaking on a
        // loaded CI runner.
        wasi_p3_runner_jit.setEnvironmentVariable("WAMR_TESTSUITE_TIMEOUT", "120");
        wasi_p3_runner_jit.step.dependOn(b.getInstallStep());
        const wasi_p3_testsuite_jit_step = b.step(
            "wasi-p3-testsuite-jit",
            "Run the WASI Preview 3 conformance gate through the in-process JIT path (-Djit=true; #856)",
        );
        wasi_p3_testsuite_jit_step.dependOn(&wasi_p3_runner_jit.step);

        const wasi_p3_unfiltered_runner_jit = b.addSystemCommand(&.{
            "python3",
            "tests/wasi-p3-unfiltered.py",
        });
        wasi_p3_unfiltered_runner_jit.setEnvironmentVariable("WAMR", b.getInstallPath(.bin, "wamr"));
        wasi_p3_unfiltered_runner_jit.setEnvironmentVariable("WAMRC", b.getInstallPath(.bin, "wamrc"));
        wasi_p3_unfiltered_runner_jit.setEnvironmentVariable("WAMR_JIT_TESTSUITE", "1");
        wasi_p3_unfiltered_runner_jit.setEnvironmentVariable("WAMR_TESTSUITE_TIMEOUT", "120");
        wasi_p3_unfiltered_runner_jit.step.dependOn(b.getInstallStep());
        const wasi_p3_unfiltered_jit_step = b.step(
            "wasi-p3-testsuite-jit-unfiltered",
            "Run all WASI Preview 3 fixtures through JIT and assert no-sidecar execution",
        );
        wasi_p3_unfiltered_jit_step.dependOn(&wasi_p3_unfiltered_runner_jit.step);
    }

    // ── Wasmtime parity gate (#583 C1, original #489 proposal) ────────
    // Runs the *same* `wasm32-wasip3` fixtures through upstream Wasmtime
    // (CI pin: v46.0.1, matching the updated upstream suite — see the
    // `examples-wasmtime` job in `.github/workflows/ci.yml`).
    // The wasm test corpus is identical, so a wamr regression that
    // Wasmtime *also* exhibits flags as a fixture bug rather than a
    // wamr bug (see `scripts/diff-testsuite-reports.py`).
    //
    // Resolves the Wasmtime binary from the `WASMTIME` env var (set by
    // CI; defaults to `wasmtime` on `PATH`). The `WASMTIME` env var is
    // intentionally not pinned in-tree so a developer can point the
    // step at a locally-built wasmtime for triage. The CI workflow at
    // `.github/workflows/wasi-p3-parity.yml` is the source of truth
    // for the pinned version.
    const wasi_p3_runner_wasmtime = b.addSystemCommand(&.{
        "python3",
        "tests/wasi-testsuite-runner-patch/wasi_test_runner.py",
        "--test-suite",
        "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip3",
        "--runtime-adapter",
        "tests/wasi-testsuite-adapter/wasmtime.py",
        "--expectations",
        "tests/wasi-p3-testsuite-expectations.toml",
    });
    // No `WAMR` env var needed here; the wasmtime adapter reads
    // `WASMTIME` instead (the test binaries themselves are the same
    // `*.wasm` files that the wamr-side gate exercises).
    const wasi_p3_testsuite_wasmtime_step = b.step(
        "wasi-p3-testsuite-wasmtime",
        "Run the WASI Preview 3 fixtures through Wasmtime (#583 C1 parity gate)",
    );
    wasi_p3_testsuite_wasmtime_step.dependOn(&wasi_p3_runner_wasmtime.step);

    // ── Wasmtime parity diff ──────────────────────────────────────────
    // Convenience step that runs both runtimes through
    // `scripts/wasi-p3-parity.py` (which writes JSON reports + a
    // classifier summary, then forwards the diff exit code). The
    // orchestrator is a Python script so both runtime reports are
    // produced even if either runner exits non-zero; the diff step
    // then classifies deltas as regressions vs fixture/runtime bugs.
    // Output JSONs live under `zig-out/test-reports/` so CI can upload
    // them as artifacts on failure.
    const reports_dir = b.pathJoin(&.{ b.install_path, "test-reports" });
    const parity_orchestrator = b.addSystemCommand(&.{
        "python3",
        "scripts/wasi-p3-parity.py",
        "--output-dir",
        reports_dir,
    });
    parity_orchestrator.setEnvironmentVariable("WAMR", b.getInstallPath(.bin, "wamr"));
    parity_orchestrator.step.dependOn(b.getInstallStep());
    const wasi_p3_parity_step = b.step(
        "wasi-p3-parity",
        "Run wamr + Wasmtime against wasm32-wasip3 fixtures and diff (#583 C1)",
    );
    wasi_p3_parity_step.dependOn(&parity_orchestrator.step);

    // ── Tests ──────────────────────────────────────────────────────────
    const test_module = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("config", config_module);
    test_module.addImport("tls", tls_module);

    const lib_unit_tests = b.addTest(.{
        .root_module = test_module,
    });
    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const shared_memory_unit_tests = b.addTest(.{
        .root_module = test_module,
        .filters = &.{
            "SharedMemory:",
            "ParkingLot:",
            "MemoryInstance: shared",
            "emit: memory section round-trip",
            "emit: import section round-trip",
            "reserved address space",
        },
    });
    const run_shared_memory_unit_tests = b.addRunArtifact(shared_memory_unit_tests);
    const shared_memory_test_step = b.step(
        "test-shared-memory",
        "Run stable shared-memory and parking-lot tests",
    );
    shared_memory_test_step.dependOn(&run_shared_memory_unit_tests.step);

    const exe_test_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe_test_module.addImport("config", config_module);
    exe_test_module.addImport("wamr", lib_module);

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_test_module,
    });
    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const keyvault_harness_tests = b.addSystemCommand(&.{
        "python3",
        "tests/test_bench_keyvault.py",
    });
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
    test_step.dependOn(stable_resources_test_step);
    test_step.dependOn(&keyvault_harness_tests.step);

    const artifact_consumer_test = b.addSystemCommand(&.{ b.graph.zig_exe, "build" });
    artifact_consumer_test.setCwd(b.path("tests/artifact-consumer"));
    test_step.dependOn(&artifact_consumer_test.step);

    // wamrc unit tests (subcommand parsing, deriveOutputPath).
    const wamrc_test_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    wamrc_test_module.addImport("config", config_module);
    wamrc_test_module.addImport("wamr", lib_module);
    const wamrc_unit_tests = b.addTest(.{
        .root_module = wamrc_test_module,
    });
    const run_wamrc_unit_tests = b.addRunArtifact(wamrc_unit_tests);
    test_step.dependOn(&run_wamrc_unit_tests.step);

    // CLI smoke assertions: subcommand layout, exit codes, version stdout.
    {
        const wamr_version_line = b.fmt("wamr {s}\n", .{version_string});
        const wamrc_version_line = b.fmt("wamrc {s}\n", .{version_string});

        const wamr_version_run = b.addRunArtifact(exe);
        wamr_version_run.addArg("version");
        wamr_version_run.expectExitCode(0);
        wamr_version_run.expectStdOutEqual(wamr_version_line);
        test_step.dependOn(&wamr_version_run.step);

        const wamr_long_version_run = b.addRunArtifact(exe);
        wamr_long_version_run.addArg("--version");
        wamr_long_version_run.expectExitCode(0);
        wamr_long_version_run.expectStdOutEqual(wamr_version_line);
        test_step.dependOn(&wamr_long_version_run.step);

        const wamrc_version_run = b.addRunArtifact(wamrc);
        wamrc_version_run.addArg("version");
        wamrc_version_run.expectExitCode(0);
        wamrc_version_run.expectStdOutEqual(wamrc_version_line);
        test_step.dependOn(&wamrc_version_run.step);

        const wamr_help_run = b.addRunArtifact(exe);
        wamr_help_run.addArgs(&.{ "help", "run" });
        wamr_help_run.expectExitCode(0);
        test_step.dependOn(&wamr_help_run.step);

        const wamrc_help_compile = b.addRunArtifact(wamrc);
        wamrc_help_compile.addArgs(&.{ "help", "compile" });
        wamrc_help_compile.expectExitCode(0);
        test_step.dependOn(&wamrc_help_compile.step);

        const wamrc_help_run = b.addRunArtifact(wamrc);
        wamrc_help_run.addArgs(&.{ "help", "run" });
        wamrc_help_run.expectExitCode(0);
        test_step.dependOn(&wamrc_help_run.step);
    }

    // `wamrc run` smoke (#665): compile + spawn the just-built `wamr`
    // against the 36-byte noop core wasm. Exercises the full subprocess
    // pipeline (sibling-output → freshness sidecar → wamr discovery via
    // `WAMR_BIN` → exit-code propagation). Output goes to a build-cache
    // file via `-o` so the source tree stays clean.
    if (aot_executable_target) {
        const wamrc_run_smoke = b.addRunArtifact(wamrc);
        wamrc_run_smoke.addArg("run");
        wamrc_run_smoke.addArg("-o");
        const smoke_out = wamrc_run_smoke.addOutputFileArg("noop.cwasm");
        _ = smoke_out;
        wamrc_run_smoke.addFileArg(b.path("tests/coldstart/noop.wasm"));
        wamrc_run_smoke.setEnvironmentVariable("WAMR_BIN", b.getInstallPath(.bin, "wamr"));
        wamrc_run_smoke.step.dependOn(b.getInstallStep());
        wamrc_run_smoke.expectExitCode(0);
        test_step.dependOn(&wamrc_run_smoke.step);
    }

    // #853: in-process JIT smoke test. Only meaningful for a `-Djit=true`
    // build (see #852) — spawns the just-built `wamr` directly against a
    // plain core wasm module with no `.cwasm` in sight, proving the
    // compile-then-execute happens in one process/one command, matching
    // the `wasmtime run foo.wasm` UX. `tests/coldstart/hello_stdout.wasm`
    // is a tiny hand-authored WASI module (`fd_write("hello from jit\n")`)
    // regenerable via:
    //   wat2wasm hello_stdout.wat -o tests/coldstart/hello_stdout.wasm
    // (source not checked in, mirrors `tests/coldstart/noop.wasm`).
    if (jit and aot_executable_target) {
        const jit_hello_smoke = b.addRunArtifact(exe);
        jit_hello_smoke.addArg("run");
        jit_hello_smoke.addFileArg(b.path("tests/coldstart/hello_stdout.wasm"));
        jit_hello_smoke.expectExitCode(0);
        jit_hello_smoke.expectStdOutEqual("hello from jit\n");
        test_step.dependOn(&jit_hello_smoke.step);
    }

    // #854: in-process JIT smoke test for components. Only meaningful
    // for a `-Djit=true` build — spawns the just-built `wamr` directly
    // against `stdio-echo.wasm` (the canonical end-to-end WASI-P2
    // fixture, #156) with no sibling `.cwasm.json` manifest in sight,
    // proving the component in-process JIT path compiles every core
    // module and instantiates in one process/one command. Gated on
    // `aot_trampoline_pool_target` (not just `aot_executable_target`)
    // because `stdio-echo.wasm`'s WASI preview1 imports need the
    // host-import trampoline pool, unsupported on Windows / macOS-aarch64
    // (see the `760-aot-cli-exit` regression gate above for the same
    // rationale).
    if (jit and aot_trampoline_pool_target) {
        const jit_component_smoke = b.addRunArtifact(exe);
        jit_component_smoke.addArg("run");
        jit_component_smoke.addFileArg(b.path("src/component/fixtures/stdio-echo.wasm"));
        jit_component_smoke.setStdIn(.{ .bytes = "hello\n" });
        jit_component_smoke.expectExitCode(0);
        jit_component_smoke.expectStdOutEqual("echo: hello\n");
        test_step.dependOn(&jit_component_smoke.step);

        // #889 follow-up: `expectExitCode`/`expectStdOutEqual` above
        // can't see this — Zig's `DebugAllocator` leak diagnostics print
        // to stderr but deliberately "do not affect return code" (see
        // lib/std/start.zig's `callMain`), so a `compileCoreWasmCached`
        // in-memory lazy-JIT leak (like the `lazy_local_indices` /
        // `needs_trampoline` ownership bug this regression-tests) would
        // otherwise pass the checks above silently. `Step.Run`'s check
        // list only supports match/exact assertions, not "must not
        // contain", so capture stderr from a second identical run and
        // fail the build via `grep` if it contains a leak diagnostic —
        // deliberately not asserting exact stderr content, since
        // unrelated compiler debug-log lines also land there and vary
        // with optimizer behavior / log level.
        if (lazy_jit) {
            const jit_component_smoke_leak_check = b.addRunArtifact(exe);
            jit_component_smoke_leak_check.addArg("run");
            jit_component_smoke_leak_check.addFileArg(b.path("src/component/fixtures/stdio-echo.wasm"));
            jit_component_smoke_leak_check.setStdIn(.{ .bytes = "hello\n" });
            jit_component_smoke_leak_check.expectExitCode(0);
            const captured_stderr = jit_component_smoke_leak_check.captureStdErr(.{});
            const check_no_leak = b.addSystemCommand(&.{
                "sh", "-c", "! grep -qiE 'DebugAllocator|leaked' \"$0\"",
            });
            check_no_leak.addFileArg(captured_stderr);
            check_no_leak.setName("check jit_component_smoke has no DebugAllocator leaks");
            test_step.dependOn(&check_no_leak.step);
        }
    }

    // #918 regression: `wamr serve` must shut down cleanly (exit 0) on
    // SIGINT/SIGTERM rather than dying from the signal (macOS
    // `Popen(-2)`) or wedging in `accept` until force-killed (Linux). A
    // small driver (tests/component-http-shutdown/driver.zig) spawns
    // `wamr serve --addr=127.0.0.1:0 http-service.wasm`, scrapes the
    // kernel-assigned port from stdout, confirms one `GET / -> 200`
    // request, then sends SIGINT and requires a *bounded, clean* exit 0.
    //
    // Gated on `jit` (the `-Djit=true` CI jobs build the in-process JIT
    // needed to compile the ~3 MB P3 component in one process) and
    // `aot_trampoline_pool_target` (its WASI imports need the host-import
    // trampoline pool, unsupported on Windows / macOS-aarch64, same as
    // `jit_component_smoke`). Also Linux-only because the driver drives
    // the child with raw Linux `kill`/socket syscalls.
    if (jit and aot_trampoline_pool_target and target.result.os.tag == .linux) {
        const http_shutdown_driver = b.addExecutable(.{
            .name = "component-http-shutdown-driver",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tests/component-http-shutdown/driver.zig"),
                .target = b.graph.host,
                .optimize = .Debug,
            }),
        });
        const run_http_shutdown = b.addRunArtifact(http_shutdown_driver);
        run_http_shutdown.addFileArg(exe.getEmittedBin());
        run_http_shutdown.addFileArg(b.path("tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip3/http-service.wasm"));
        run_http_shutdown.expectExitCode(0);
        test_step.dependOn(&run_http_shutdown.step);
    }

    // #760 regression: AOT `wasi:cli/exit.exit` must terminate the host
    // process with the requested discriminant rather than returning the
    // post-#714 sentinel through the canon-lower(aot) trampoline. Two
    // hand-rolled WAT fixtures (`tests/regressions/760-aot-cli-exit/`)
    // import `wasi:cli/exit@0.2.0` directly (no preview1 → preview2
    // adapter, so unrelated to the cross-instance #662 blocker that
    // gates the adapter-based command examples) and exit with the ok / err
    // discriminants. Before the fix both crashed the host with SIGSEGV
    // (exit 139) after the wit-bindgen adapter's "host exit
    // implementation didn't exit!" assertion fired on the sentinel
    // return; after the fix the host exits 0 / 1 respectively.
    //
    // Gated on `aot_trampoline_pool_target` (not just
    // `aot_executable_target`) because canon-lower for a host import
    // requires the host-import trampoline pool, which isn't supported
    // on Windows / macOS-aarch64 — running there fails with
    // `[aot reject] ... UnsupportedPlatform` regardless of #760.
    if (aot_trampoline_pool_target) {
        const fixtures = [_]struct { name: []const u8, expected_exit: u8 }{
            .{ .name = "exit-ok.wasm", .expected_exit = 0 },
            .{ .name = "exit-with-code-7.wasm", .expected_exit = 1 },
        };
        for (fixtures) |f| {
            const run = b.addRunArtifact(wamrc);
            run.addArg("run");
            run.addArg("-o");
            _ = run.addOutputFileArg(b.fmt("760-{s}.cwasm", .{f.name}));
            run.addFileArg(b.path(b.fmt("tests/regressions/760-aot-cli-exit/{s}", .{f.name})));
            run.setEnvironmentVariable("WAMR_BIN", b.getInstallPath(.bin, "wamr"));
            run.step.dependOn(b.getInstallStep());
            run.expectExitCode(f.expected_exit);
            test_step.dependOn(&run.step);
        }
    }

    // #794: load-forwarding soundness gate. Compiles the shipped hand-written
    // wasm corpus with the dedicated Check-11 verify mode
    // (`--verify-ir=load-forwarding`) and requires a clean exit. Check 11 is a
    // SOUND OVER-APPROXIMATION: it never misses an unsound forward but can
    // false-positive on legitimate "snapshot" loads (a load value reused across
    // an aliasing store), which are pervasive in LLVM-optimised wasm. The gated
    // corpus is therefore deliberately limited to hand-written fixtures. The
    // strong, false-positive-free regression net for the #743 / #793 bug class
    // is the differential property fuzzer
    // (`src/compiler/ir/property_test.zig`, `loop_forwarded_load` shape), which
    // executes original vs optimised IR and compares observables.
    const verify_ir_soundness_step = b.step(
        "verify-ir-soundness",
        "Compile the hand-written wasm corpus with the load-forwarding soundness check (#794)",
    );
    if (aot_executable_target) {
        const soundness_fixtures = [_][]const u8{
            "tests/spec-json/linking.trap401.wasm",
            "tests/spec-json/linking.trap413.wasm",
            "tests/spec-json/linking.trap554.wasm",
            "tests/spec-json/linking.trap566.wasm",
            "tests/spec-json/linking.trap592.wasm",
            "tests/coldstart/noop.wasm",
        };
        for (soundness_fixtures) |fixture| {
            const run = b.addRunArtifact(wamrc);
            run.addArgs(&.{ "compile", "--verify-ir=load-forwarding", "-o" });
            _ = run.addOutputFileArg("verified.cwasm");
            run.addFileArg(b.path(fixture));
            run.expectExitCode(0);
            verify_ir_soundness_step.dependOn(&run.step);
            test_step.dependOn(&run.step);
        }
    }

    // #757 `wamrc verify` smoke (no wasmtime required):
    //   * `verify help` → exit 0 + non-empty stdout.
    //   * `verify --wasmtime-bin=/dev/null/nope <wasm>` → spawn fails →
    //     exit 2 (setup error). Exercises the binary-resolution +
    //     "spawn failed = setup error" wiring without needing
    //     wasmtime on the test runner's PATH.
    // Reuses the 339-byte `tests/regressions/760-aot-cli-exit/exit-ok.wasm`
    // (a valid component the AOT precompile accepts) so the smoke
    // gets past the precompile step before the wasmtime spawn fails.
    if (aot_trampoline_pool_target) {
        const verify_help = b.addRunArtifact(wamrc);
        verify_help.addArgs(&.{ "verify", "help" });
        verify_help.expectExitCode(0);
        test_step.dependOn(&verify_help.step);

        const verify_no_wasmtime = b.addRunArtifact(wamrc);
        verify_no_wasmtime.addArgs(&.{ "verify", "--wasmtime-bin=/dev/null/nope-wamrc-757" });
        verify_no_wasmtime.addFileArg(b.path("tests/regressions/760-aot-cli-exit/exit-ok.wasm"));
        verify_no_wasmtime.setEnvironmentVariable("WAMR_BIN", b.getInstallPath(.bin, "wamr"));
        verify_no_wasmtime.step.dependOn(b.getInstallStep());
        verify_no_wasmtime.expectExitCode(2);
        test_step.dependOn(&verify_no_wasmtime.step);
    }

    // Compiler IR passes tests (separate module to avoid root/wamr conflict)
    const passes_test_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/ir/passes.zig"),
        .target = target,
        .optimize = optimize,
    });
    const passes_tests = b.addTest(.{
        .root_module = passes_test_module,
    });
    const run_passes_tests = b.addRunArtifact(passes_tests);
    test_step.dependOn(&run_passes_tests.step);

    // Compiler IR analysis tests
    const analysis_test_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/ir/analysis.zig"),
        .target = target,
        .optimize = optimize,
    });
    const analysis_tests = b.addTest(.{
        .root_module = analysis_test_module,
    });
    const run_analysis_tests = b.addRunArtifact(analysis_tests);
    test_step.dependOn(&run_analysis_tests.step);

    // Compiler local-init analysis tests
    const local_init_test_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/ir/local_init.zig"),
        .target = target,
        .optimize = optimize,
    });
    const local_init_tests = b.addTest(.{
        .root_module = local_init_test_module,
    });
    const run_local_init_tests = b.addRunArtifact(local_init_tests);
    test_step.dependOn(&run_local_init_tests.step);

    // Compiler register allocator tests
    const regalloc_test_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/ir/regalloc.zig"),
        .target = target,
        .optimize = optimize,
    });
    const regalloc_tests = b.addTest(.{
        .root_module = regalloc_test_module,
    });
    const run_regalloc_tests = b.addRunArtifact(regalloc_tests);
    test_step.dependOn(&run_regalloc_tests.step);

    // Compiler loop-aware live-range splitting tests (#383 / #524). The
    // file's tests were previously not wired into any module and so never
    // ran; this module makes `zig build test` cover them.
    const range_split_test_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/ir/range_split.zig"),
        .target = target,
        .optimize = optimize,
    });
    const range_split_tests = b.addTest(.{
        .root_module = range_split_test_module,
    });
    const run_range_split_tests = b.addRunArtifact(range_split_tests);
    test_step.dependOn(&run_range_split_tests.step);

    // Compiler IR printer tests
    const ir_print_test_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/ir/print_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    const ir_print_tests = b.addTest(.{
        .root_module = ir_print_test_module,
    });
    const run_ir_print_tests = b.addRunArtifact(ir_print_tests);
    test_step.dependOn(&run_ir_print_tests.step);

    // IR verifier tests (#624).
    const verifier_test_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/ir/verifier.zig"),
        .target = target,
        .optimize = optimize,
    });
    const verifier_tests = b.addTest(.{
        .root_module = verifier_test_module,
    });
    const run_verifier_tests = b.addRunArtifact(verifier_tests);
    test_step.dependOn(&run_verifier_tests.step);

    // IR interpreter tests (#736).
    const ir_interp_test_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/ir/interp.zig"),
        .target = target,
        .optimize = optimize,
    });
    const ir_interp_tests = b.addTest(.{
        .root_module = ir_interp_test_module,
    });
    const run_ir_interp_tests = b.addRunArtifact(ir_interp_tests);
    test_step.dependOn(&run_ir_interp_tests.step);

    // IR deterministic generator tests (#736).
    const ir_fuzz_test_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/ir/fuzz.zig"),
        .target = target,
        .optimize = optimize,
    });
    const ir_fuzz_tests = b.addTest(.{
        .root_module = ir_fuzz_test_module,
    });
    const run_ir_fuzz_tests = b.addRunArtifact(ir_fuzz_tests);
    test_step.dependOn(&run_ir_fuzz_tests.step);

    // IR optimizer property tests (#736).
    const ir_property_test_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/ir/property_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    const ir_property_options = b.addOptions();
    ir_property_options.addOption(u32, "iterations", ir_property_iterations);
    ir_property_test_module.addImport("ir_property_options", ir_property_options.createModule());
    const ir_property_tests = b.addTest(.{
        .root_module = ir_property_test_module,
    });
    const run_ir_property_tests = b.addRunArtifact(ir_property_tests);
    test_step.dependOn(&run_ir_property_tests.step);

    // Dominator-aware redundant-load forwarder tests (#391).
    const dom_frl_test_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/ir/forward_redundant_loads_dominator.zig"),
        .target = target,
        .optimize = optimize,
    });
    const dom_frl_tests = b.addTest(.{
        .root_module = dom_frl_test_module,
    });
    const run_dom_frl_tests = b.addRunArtifact(dom_frl_tests);
    test_step.dependOn(&run_dom_frl_tests.step);

    // Interp-vs-AOT differential tests. Own module (with its own `wamr`
    // alias) so `aot_harness.zig` — which `differential.zig` imports — is
    // reached through the `wamr` module and not duplicated into it. The
    // standalone `aot_harness` module below (used by fuzz targets) must
    // own the file exclusively; pulling it in via the main `wamr` lib
    // module would trigger Zig's "file exists in modules X and Y" error.
    const differential_test_module = b.createModule(.{
        .root_source_file = b.path("src/tests/differential.zig"),
        .target = target,
        .optimize = optimize,
    });
    differential_test_module.addImport("wamr", lib_module);
    const differential_options = b.addOptions();
    differential_options.addOption(bool, "verify_ir_triage", verify_ir_triage);
    differential_test_module.addImport("differential_options", differential_options.createModule());
    const differential_tests = b.addTest(.{
        .root_module = differential_test_module,
    });
    const run_differential_tests = b.addRunArtifact(differential_tests);
    test_step.dependOn(&run_differential_tests.step);

    // #694: regression test for active elem segments referencing
    // funcidx ≥ 256 (was capped by a fixed 256-entry buffer in
    // `mapCodeExecutable`, silently dropping both the native pointer
    // and the sig_id update for any high-funcidx slot).
    const aot_high_funcidx_module = b.createModule(.{
        .root_source_file = b.path("src/tests/aot_high_funcidx_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    aot_high_funcidx_module.addImport("wamr", lib_module);
    const aot_high_funcidx_tests = b.addTest(.{
        .root_module = aot_high_funcidx_module,
    });
    const run_aot_high_funcidx_tests = b.addRunArtifact(aot_high_funcidx_tests);
    test_step.dependOn(&run_aot_high_funcidx_tests.step);

    // #859: thread-safety stress test for the in-process JIT compile
    // entry points — N threads independently compile+load+instantiate+
    // execute+destroy a small AOT module concurrently and each must
    // observe the correct, uncorrupted result for its own distinct input.
    const jit_thread_safety_module = b.createModule(.{
        .root_source_file = b.path("src/tests/jit_thread_safety_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    jit_thread_safety_module.addImport("wamr", lib_module);
    const jit_thread_safety_tests = b.addTest(.{
        .root_module = jit_thread_safety_module,
    });
    const run_jit_thread_safety_tests = b.addRunArtifact(jit_thread_safety_tests);
    test_step.dependOn(&run_jit_thread_safety_tests.step);

    // #860: fast/baseline compile preset correctness + compile-time
    // comparison against the full pipeline (CoreMark fixture).
    const jit_fast_preset_module = b.createModule(.{
        .root_source_file = b.path("src/tests/jit_fast_preset_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    jit_fast_preset_module.addImport("wamr", lib_module);
    jit_fast_preset_module.addAnonymousImport("coremark_wasm", .{
        .root_source_file = b.path("tests/benchmarks/coremark/coremark_wasi.wasm"),
    });
    const jit_fast_preset_tests = b.addTest(.{
        .root_module = jit_fast_preset_module,
    });
    const run_jit_fast_preset_tests = b.addRunArtifact(jit_fast_preset_tests);
    test_step.dependOn(&run_jit_fast_preset_tests.step);

    // #862/#890: lazy-JIT design-spike prototype (leaf functions only,
    // x86_64/aarch64). Self-skips via `config.lazy_jit`/arch checks when
    // not built with `-Djit=true -Dlazy_jit=true`.
    const lazy_jit_spike_module = b.createModule(.{
        .root_source_file = b.path("src/tests/lazy_jit_spike_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    lazy_jit_spike_module.addImport("wamr", lib_module);
    lazy_jit_spike_module.addAnonymousImport("lazy_bench_fixture_wasm", .{
        .root_source_file = b.path("src/tests/lazy_bench_fixture.wasm"),
    });
    const lazy_jit_spike_tests = b.addTest(.{
        .root_module = lazy_jit_spike_module,
    });
    const run_lazy_jit_spike_tests = b.addRunArtifact(lazy_jit_spike_tests);
    test_step.dependOn(&run_lazy_jit_spike_tests.step);

    // #625 phase 1: AOT-backed component-core smoke test. Lives in its
    // own test step for the same reason `differential.zig` does:
    // `aot_harness.zig` cannot be pulled into the `wamr` lib module
    // (it's already owned by the test runners that route through it).
    const component_aot_smoke_module = b.createModule(.{
        .root_source_file = b.path("src/tests/component_aot_smoke_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    component_aot_smoke_module.addImport("wamr", lib_module);
    const component_aot_smoke_tests = b.addTest(.{
        .root_module = component_aot_smoke_module,
    });
    const run_component_aot_smoke_tests = b.addRunArtifact(component_aot_smoke_tests);
    test_step.dependOn(&run_component_aot_smoke_tests.step);

    // Phase 2: precompile → manifest → loadManifest → instantiate
    // round-trip test (#625). Uses the same separate-module pattern
    // as the phase 1 smoke test above.
    const component_precompile_module = b.createModule(.{
        .root_source_file = b.path("src/tests/component_precompile_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    component_precompile_module.addImport("wamr", lib_module);
    const component_precompile_tests = b.addTest(.{
        .root_module = component_precompile_module,
    });
    const run_component_precompile_tests = b.addRunArtifact(component_precompile_tests);
    test_step.dependOn(&run_component_precompile_tests.step);

    // #676: wamrc compile-component must recurse into nested
    // sub-components (the dominant shape of `wabt component
    // compose -d` / `wasm-tools compose` output). Companion to the
    // single-core round-trip above.
    const component_precompile_nested_module = b.createModule(.{
        .root_source_file = b.path("src/tests/component_precompile_nested_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    component_precompile_nested_module.addImport("wamr", lib_module);
    const component_precompile_nested_tests = b.addTest(.{
        .root_module = component_precompile_nested_module,
    });
    const run_component_precompile_nested_tests = b.addRunArtifact(component_precompile_nested_tests);
    test_step.dependOn(&run_component_precompile_nested_tests.step);

    // Phase 3: canon.lift dispatches onto AOT cores (#625).
    const component_aot_canonlift_module = b.createModule(.{
        .root_source_file = b.path("src/tests/component_aot_canonlift_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    component_aot_canonlift_module.addImport("wamr", lib_module);
    const component_aot_canonlift_tests = b.addTest(.{
        .root_module = component_aot_canonlift_module,
    });
    const run_component_aot_canonlift_tests = b.addRunArtifact(component_aot_canonlift_tests);
    test_step.dependOn(&run_component_aot_canonlift_tests.step);

    // Cold-start budget tests (issue #395). In-process timing companion
    // to the subprocess harness in #394. Compile a 36-byte noop wasm
    // through the just-built `wamrc` to produce a `.cwasm` fixture, then
    // run two timing tests asserting WAMR-internal load+invoke stays
    // under fixed budgets. Disable with `-Dskip-coldstart=true`.
    if (aot_executable_target) {
        const wamrc_compile_noop = b.addRunArtifact(wamrc);
        wamrc_compile_noop.addArg("compile");
        wamrc_compile_noop.addFileArg(b.path("tests/coldstart/noop.wasm"));
        wamrc_compile_noop.addArg("-o");
        const noop_cwasm = wamrc_compile_noop.addOutputFileArg("noop.cwasm");

        const coldstart_options = b.addOptions();
        coldstart_options.addOption(bool, "skip", skip_coldstart);

        const coldstart_test_module = b.createModule(.{
            .root_source_file = b.path("src/tests/coldstart_test.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        });
        coldstart_test_module.addImport("wamr", lib_module);
        coldstart_test_module.addImport("coldstart_options", coldstart_options.createModule());
        coldstart_test_module.addAnonymousImport("noop_wasm", .{
            .root_source_file = b.path("tests/coldstart/noop.wasm"),
        });
        coldstart_test_module.addAnonymousImport("noop_cwasm", .{
            .root_source_file = noop_cwasm,
        });

        const coldstart_tests = b.addTest(.{
            .root_module = coldstart_test_module,
        });
        const run_coldstart_tests = b.addRunArtifact(coldstart_tests);
        test_step.dependOn(&run_coldstart_tests.step);
    }

    // ── WASI sockets (#437) end-to-end ────────────────────────────────
    // Previously compiled a tiny `wasm32-wasi` echo server and drove
    // `wamr run --listen=…` against it. Removed in #644 alongside the
    // AOT-only CLI policy: plain core wasm is no longer accepted by
    // `wamr run`, and `--listen` on core wasm depended on the
    // interpreter `runWasm` path which no longer exists. The
    // socket-preopen plumbing it exercised lives in the library
    // (`WasiCtx.addPreopenSocket`) and is covered by lower-level
    // tests in `src/wasi/`.

    // ── Benchmark ─────────────────────────────────────────────────────
    const bench_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/bench_codegen.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        // Darwin's clock_gettime lives in libSystem; the timer needs libc
        // linked. Linux uses a raw syscall and Windows uses ntdll, so libc
        // is only required here.
        .link_libc = if (target.result.os.tag.isDarwin()) true else null,
    });

    const bench_exe = b.addExecutable(.{
        .name = "codegen-bench",
        .root_module = bench_module,
    });
    if (bench_target) b.installArtifact(bench_exe);

    const run_bench = b.addRunArtifact(bench_exe);
    const bench_step = b.step("bench", "Run codegen benchmarks");
    bench_step.dependOn(&run_bench.step);

    // ── Fuzz harnesses ────────────────────────────────────────────────
    // CLI binaries that replay corpus inputs through a specific
    // pipeline (core loader / component loader / interp / aot /
    // interp-vs-aot diff) and leave
    // a reproducer at <crashes>/in-flight.wasm if the process aborts.
    // See src/tests/fuzz/common.zig, tests/fuzz/README.md, and
    // .github/workflows/fuzz.yml.
    const aot_harness_module = b.createModule(.{
        .root_source_file = b.path("src/tests/aot_harness.zig"),
        .target = target,
        .optimize = optimize,
    });
    aot_harness_module.addImport("wamr", lib_module);

    // ── CoreMark AOT runner ────────────────────────────────────────────
    // Loads a CoreMark wasi `.wasm` and executes it through the Zig AOT
    // backend (same pipeline as differential tests). Replaces the old
    // C-based standalone coremark runner for gating the Zig backend on
    // real CoreMark workloads.
    const coremark_module = b.createModule(.{
        .root_source_file = b.path("src/tests/coremark_aot_runner.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    coremark_module.addImport("wamr", lib_module);

    const coremark_exe = b.addExecutable(.{
        .name = "coremark-aot-runner",
        .root_module = coremark_module,
    });
    if (aot_executable_target) b.installArtifact(coremark_exe);

    const run_coremark_nofp = b.addRunArtifact(coremark_exe);
    run_coremark_nofp.addArg("tests/benchmarks/coremark/coremark_wasi_nofp.wasm");
    const run_coremark_fp = b.addRunArtifact(coremark_exe);
    run_coremark_fp.addArg("tests/benchmarks/coremark/coremark_wasi.wasm");
    const coremark_step = b.step(
        "coremark-aot",
        "Run the CoreMark wasi benchmarks through the Zig AOT backend",
    );
    coremark_step.dependOn(&run_coremark_nofp.step);
    coremark_step.dependOn(&run_coremark_fp.step);

    // ── CoreMark profile runner ────────────────────────────────────────
    // SIGPROF-based sampling profiler for the CoreMark AOT, plus
    // disassembly of the top-3 hot functions. Linux + aarch64/x86_64
    // only — see src/tests/profile/sigprof.zig. See
    // tests/benchmarks/coremark/README.md "Profiling" section.
    const coremark_profile_module = b.createModule(.{
        .root_source_file = b.path("src/tests/coremark_profile_runner.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    coremark_profile_module.addImport("wamr", lib_module);

    const coremark_profile_exe = b.addExecutable(.{
        .name = "coremark-profile-runner",
        .root_module = coremark_profile_module,
    });
    if (aot_executable_target) b.installArtifact(coremark_profile_exe);

    const run_coremark_profile = b.addRunArtifact(coremark_profile_exe);
    run_coremark_profile.addArg("tests/benchmarks/coremark/coremark_wasi_nofp.wasm");
    const coremark_profile_step = b.step(
        "coremark-profile",
        "Sampling-profile the CoreMark AOT and dump top-3 hot-function disassembly",
    );
    coremark_profile_step.dependOn(&run_coremark_profile.step);

    // ── SIMD benchmark runner ───────────────────────────────────────────
    // Builds small in-memory SIMD modules and reports interpreter vs AOT
    // status/timing. Optional runner args after `--` can enable external
    // baselines such as Wasmtime.
    const simd_bench_module = b.createModule(.{
        .root_source_file = b.path("src/tests/simd_bench_runner.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    simd_bench_module.addImport("wamr", lib_module);

    const simd_bench_exe = b.addExecutable(.{
        .name = "simd-bench-runner",
        .root_module = simd_bench_module,
    });
    if (aot_executable_target) b.installArtifact(simd_bench_exe);

    const simd_bench_step = b.step(
        "simd-bench",
        "Run SIMD interpreter/AOT benchmark status probes",
    );
    if (aot_executable_target) {
        const run_simd_bench = b.addRunArtifact(simd_bench_exe);
        run_simd_bench.addArg("--iterations");
        run_simd_bench.addArg("10000");
        if (b.args) |args| run_simd_bench.addArgs(args);
        simd_bench_step.dependOn(&run_simd_bench.step);
    }

    // ── WASI stream zero-copy microbench (#583 B2) ──────────────────────
    // Drives the executor's `stream.read` rendezvous against a synthetic
    // host_driver to compare scratch-buffer vs zero-copy specialisation
    // wall-clock + allocation cost. See
    // `tests/benchmarks/wasi-streams/microbench.zig`.
    const wasi_streams_bench_module = b.createModule(.{
        .root_source_file = b.path("tests/benchmarks/wasi-streams/microbench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    wasi_streams_bench_module.addImport("wamr", lib_module);
    const wasi_streams_bench_exe = b.addExecutable(.{
        .name = "wasi-streams-bench",
        .root_module = wasi_streams_bench_module,
    });
    b.installArtifact(wasi_streams_bench_exe);
    const wasi_streams_bench_step = b.step(
        "wasi-streams-bench",
        "Run the wasi:streams zero-copy microbench (#583 B2)",
    );
    const run_wasi_streams_bench = b.addRunArtifact(wasi_streams_bench_exe);
    if (b.args) |args| run_wasi_streams_bench.addArgs(args);
    wasi_streams_bench_step.dependOn(&run_wasi_streams_bench.step);

    // ── WASI host-path micro-bench / regression detector (#583 W11-6) ─
    // Drives `executor.dispatchCanonBuiltin` → `AsyncStream` →
    // host_driver across the four hot Preview-3 host paths
    // (http-service keep-alive RTs, UDP receive, fs read/write-via-
    // stream) with synthetic drivers shaped like the production
    // `wasi_cli_adapter` callbacks. Runs `--samples` rounds, reports
    // median + p95 + RSS-peak, and compares medians against
    // `tests/benchmarks/wasi-microbench/budget.json` (default
    // regression threshold +10 %). See
    // `tests/benchmarks/wasi-microbench/microbench.zig`.
    const wasi_microbench_module = b.createModule(.{
        .root_source_file = b.path("tests/benchmarks/wasi-microbench/microbench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    wasi_microbench_module.addImport("wamr", lib_module);
    const wasi_microbench_exe = b.addExecutable(.{
        .name = "wasi-microbench",
        .root_module = wasi_microbench_module,
    });
    b.installArtifact(wasi_microbench_exe);
    const wasi_microbench_step = b.step(
        "wasi-microbench",
        "Run the WASI host-path microbench + regression check (#583 W11-6)",
    );
    const run_wasi_microbench = b.addRunArtifact(wasi_microbench_exe);
    if (b.args) |args| run_wasi_microbench.addArgs(args);
    wasi_microbench_step.dependOn(&run_wasi_microbench.step);

    const fuzz_step = b.step("fuzz", "Build fuzz harnesses (loader, component-loader, interp, aot, diff, canon, wasi)");
    inline for (.{
        .{ .name = "loader", .file = "loader.zig", .needs_aot = false },
        .{ .name = "component-loader", .file = "component_loader.zig", .needs_aot = false },
        .{ .name = "interp", .file = "interp.zig", .needs_aot = false },
        .{ .name = "aot", .file = "aot.zig", .needs_aot = true },
        .{ .name = "diff", .file = "diff.zig", .needs_aot = true },
        .{ .name = "canon", .file = "canon.zig", .needs_aot = false },
        .{ .name = "wasi", .file = "wasi.zig", .needs_aot = false },
    }) |tgt| {
        const fuzz_mod = b.createModule(.{
            .root_source_file = b.path("src/tests/fuzz/" ++ tgt.file),
            .target = target,
            .optimize = optimize,
        });
        fuzz_mod.addImport("config", config_module);
        fuzz_mod.addImport("wamr", lib_module);
        if (tgt.needs_aot) {
            fuzz_mod.addImport("aot_harness", aot_harness_module);
        }

        const fuzz_exe = b.addExecutable(.{
            .name = "fuzz-" ++ tgt.name,
            .root_module = fuzz_mod,
        });
        const install_fuzz = b.addInstallArtifact(fuzz_exe, .{});
        fuzz_step.dependOn(&install_fuzz.step);
    }

    inline for (fuzz_seed_wasms) |seed_wasm| {
        const install_seed = b.addInstallFileWithDir(
            b.path("src/tests/fuzz/seeds/" ++ seed_wasm),
            .{ .custom = "fuzz-seeds" },
            seed_wasm,
        );
        fuzz_step.dependOn(&install_seed.step);
    }

    // ── OSS-Fuzz shim libraries ──────────────────────────────────────
    // Static archives that export `LLVMFuzzerTestOneInput` for the
    // high-priority deterministic harnesses (loader, component-loader,
    // canon). `oss-fuzz/build.sh` links each archive with
    // `$LIB_FUZZING_ENGINE` (clang `-fsanitize=fuzzer`) to produce the
    // libFuzzer binary. Repository-local only — no upstream OSS-Fuzz
    // submission is implied. See tests/fuzz/OSS_FUZZ.md.
    const fuzz_oss_step = b.step("fuzz-oss", "Build OSS-Fuzz shim static libraries (loader, component-loader, canon)");
    inline for (.{
        .{ .name = "loader", .file = "oss_loader.zig" },
        .{ .name = "component-loader", .file = "oss_component_loader.zig" },
        .{ .name = "canon", .file = "oss_canon.zig" },
    }) |tgt| {
        const oss_mod = b.createModule(.{
            .root_source_file = b.path("src/tests/fuzz/" ++ tgt.file),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .pic = true,
        });
        oss_mod.addImport("config", config_module);
        oss_mod.addImport("wamr", lib_module);

        const oss_lib = b.addLibrary(.{
            .name = "fuzz-oss-" ++ tgt.name,
            .root_module = oss_mod,
            .linkage = .static,
        });
        // Bundle libc references symbols so static-link with libFuzzer driver works.
        const install_oss = b.addInstallArtifact(oss_lib, .{});
        fuzz_oss_step.dependOn(&install_oss.step);
    }

    // ── Component-model examples ─────────────────────────────────────
    // Reproducible Zig (and one mixed Zig+Rust) WebAssembly component
    // examples under `examples/`. Opt-in: not reachable from the default
    // `zig build` or `zig build test` graphs. See `examples/README.md`
    // for prereqs and runtime status.
    //
    // The mixed Zig+Rust example needs the `wasm32-wasip1` rustup target.
    // Probe for it at configure time and skip that one example when it's
    // absent (or cargo/rustup isn't installed) so a fresh checkout still
    // builds the all-Zig examples. Override the auto-detection with
    // `-Drust-examples=true|false`.
    const rust_examples = b.option(
        bool,
        "rust-examples",
        "Build the mixed Zig+Rust component example (needs cargo + the wasm32-wasip1 rustup target). Defaults to auto-detecting the target.",
    ) orelse rustWasip1TargetAvailable(b);
    const component_runs = addComponentExamples(b, exe, aot_broken_components, rust_examples);

    // ── WASI Preview 2 conformance gate (#479) ───────────────────────
    // Curated set of Preview-2 component fixtures (sources under
    // `examples/`) run through `./zig-out/bin/wamr` with byte-exact
    // stdout + exit-code assertions. Strict superset of `examples-run`:
    // every component the latter runs is also a member of this gate. The
    // named step gives CI a discoverable entry point, mirroring
    // `wasi-testsuite` for Preview 1.
    //
    // Curation rationale + future-skips: `tests/wasi-p2-testsuite-skip.json`
    // (header `_comment` documents the format). Currently empty — every
    // wired component is expected to pass on every platform we ship.
    //
    // Opt-in: not in `zig build` / `zig build test`. Reach via
    // `zig build wasi-p2-testsuite`.
    const wasi_p2_step = b.step(
        "wasi-p2-testsuite",
        "Run the WASI Preview 2 conformance gate (curated component examples)",
    );
    wasi_p2_step.dependOn(component_runs.wamr);
}

/// Wires up the Component-Model example pipeline (sources under
/// `examples/`). Three opt-in steps are exposed:
///   * `zig build examples`               — build + validate all four
///   * `zig build examples-run`           — run them through `./zig-out/bin/wamr`
///   * `zig build examples-run-wasmtime`  — run them through
///                                          `wasmtime run -S cli-exit-with-code`
///                                          (cross-runtime parity gate; wasmtime v44+).
/// None are reachable from `zig build` or `zig build test`.
///
/// Returns the two run-step parents so the caller can layer additional
/// named entry points on top (e.g. `wasi-p2-testsuite` in #479, which
/// is a strict superset of `examples-run`).
///
/// Pinned versions:
///   * `cataggar/wabt` ≥ v3.0.0-dev.13 on PATH. `component new` embeds the
///     WIT on the fly (`--wit <dir> --world <name>`) and wraps + validates
///     in one call, so no separate `component embed` step is needed.
///     Also provides `component compose`. The wasi-preview1 → component
///     adapter is embedded in `wabt` and auto-attached by
///     `wabt component new`; no external adapter fetch.
///   * `cargo` with `wasm32-wasip1` target for the mixed example
///
/// Configure-time probe for the `wasm32-wasip1` rustup target. Runs
/// `rustup target list --installed` and checks for the target in the
/// output. Returns `false` (skip the Rust example) when rustup isn't on
/// PATH, the command fails, or the target isn't listed — so a checkout
/// without the Rust toolchain still builds the all-Zig examples.
/// Developers with a non-rustup Rust install that has the target can
/// force it on with `-Drust-examples=true`.
fn rustWasip1TargetAvailable(b: *std.Build) bool {
    var code: u8 = undefined;
    const stdout = b.runAllowFail(
        &.{ "rustup", "target", "list", "--installed" },
        &code,
        .ignore,
    ) catch return false;
    defer b.allocator.free(stdout);
    return std.mem.indexOf(u8, stdout, "wasm32-wasip1") != null;
}

fn addComponentExamples(b: *std.Build, wamr_exe: *std.Build.Step.Compile, aot_broken_components: bool, rust_examples: bool) ComponentRunSteps {
    const wasip2 = b.dependency("wasip2", .{});
    const examples_step = b.step(
        "examples",
        "Build the WebAssembly Component examples in examples/",
    );
    const run_step = b.step(
        "examples-run",
        "Run the runnable component examples through ./zig-out/bin/wamr",
    );
    // Cross-runtime parity step (cataggar/wamr#457): the same four
    // fixtures + assertions, but run under `wasmtime run -S
    // cli-exit-with-code` (v44+ required for the `@unstable`
    // `wasi:cli/exit.exit-with-code` feature gate the wabt-bundled
    // adapter lowers `proc_exit` through). Opt-in — failure mode
    // when `wasmtime` is missing from PATH is a clear systemcommand
    // error.
    const run_step_wasmtime = b.step(
        "examples-run-wasmtime",
        "Run the runnable component examples through `wasmtime run -S cli-exit-with-code` for cross-runtime parity validation",
    );
    const runs: ComponentRunSteps = .{ .wamr = run_step, .wasmtime = run_step_wasmtime };

    // ── zig-hello ──────────────────────────────────────────────────
    // Native-`wasi:cli` command: exports `wasi:cli/run@0.2.6#run` and
    // writes a greeting through `wasi:cli/stdout` + `wasi:io/streams`
    // (no preview1 `_start` / adapter). Built `wasm32-freestanding`; the
    // canonical-ABI plumbing lives in the shared `wasi_cli` guest helper.
    const hello_core = compileZigWasm(b, .{
        .source = "examples/zig-hello/src/main.zig",
        .target_triple = "wasm32-freestanding",
        .exports = &.{ "wasi:cli/run@0.2.6#run", "cabi_realloc" },
        .output = "zig-hello.core.wasm",
        .imports = &.{
            .{ .name = "wasi_cli", .path = wasip2.path("src/wasi_cli.zig"), .deps = &.{"wasi_io"} },
            .{ .name = "wasi_io", .path = wasip2.path("src/wasi_io.zig"), .deps = &.{"abi"}, .root_dep = false },
            .{ .name = "abi", .path = wasip2.path("src/abi.zig"), .root_dep = false },
        },
        // `no_lld = true` traps at runtime: Zig 0.16's self-hosted wasm
        // linker mis-sets `__stack_pointer` to -1048576. Keep on LLD.
        // https://github.com/cataggar/wamr/issues/843
        .no_llvm = false,
        .no_lld = false,
    });
    const hello = makeComponent(b, .{
        .core = hello_core,
        .wit_dir = "examples/zig-hello/wit",
        .world = "hello",
        .output = "zig-hello.wasm",
    });
    installAndValidate(b, examples_step, hello, "zig-hello.wasm");

    // wamr's `populateWasiCliRun` binds `get-stdout` + the output-stream
    // write; both wamr and wasmtime flush to the host's actual stdout.
    wireComponentRun(b, runs, wamr_exe, hello, "hello from zig component\n", 0, .{ .skip_wamr = !aot_broken_components });

    // ── zig-adder ──────────────────────────────────────────────────
    // Library component (no `run`): exports `docs:adder/add@0.1.0`.
    const adder_core = compileZigWasm(b, .{
        .source = "examples/zig-adder/src/main.zig",
        .target_triple = "wasm32-freestanding",
        .exports = &.{"docs:adder/add@0.1.0#add"},
        .output = "zig-adder.core.wasm",
    });
    // `wabt component compose` (like `wasm-tools compose`) requires
    // kebab-case file basenames (no dots before the .wasm extension).
    const adder = makeComponent(b, .{
        .core = adder_core,
        .wit_dir = "examples/zig-adder/wit",
        .world = "adder",
        .output = "zig-adder.wasm",
    });
    installAndValidate(b, examples_step, adder, "zig-adder.wasm");

    // ── zig-calculator-cmd (Zig command importing zig-adder) ───────
    const calc_core = compileZigWasm(b, .{
        .source = "examples/zig-calculator-cmd/src/main.zig",
        .target_triple = "wasm32-wasi",
        .exports = &.{"_start"},
        .output = "zig-calculator-cmd.core.wasm",
    });
    const calc_cmd = makeComponent(b, .{
        .core = calc_core,
        .wit_dir = "examples/zig-calculator-cmd/wit",
        .world = "app",
        .output = "zig-calculator-cmd.wasm",
    });

    // Compose: link `docs:adder/add@0.1.0` import against the Zig adder.
    const calc_compose = b.addSystemCommand(&.{ "wabt", "component", "compose", "-d" });
    calc_compose.addFileArg(adder);
    calc_compose.addFileArg(calc_cmd);
    calc_compose.addArg("-o");
    const calc_final = calc_compose.addOutputFileArg("zig-calculator-cmd.composed.wasm");
    installAndValidate(b, examples_step, calc_final, "zig-calculator-cmd.wasm");

    // Run the composed Zig calculator command. The wabt-bundled
    // wasi-preview1 adapter lowers `fd_write(1, …)` through
    // `wasi:io/streams.blocking-write-and-flush`, which both runtimes
    // flush to the host's actual stdout. Issue #355 wired the alias
    // chain + sub-component instantiation that this composed command
    // needs to reach its `wasi:cli/run` export.
    wireComponentRun(b, runs, wamr_exe, calc_final, "40 + 2 = 42\n100 + 200 = 300\n", 0, .{ .skip_wamr = !aot_broken_components });

    // ── mixed-zig-rust-calc (Zig adder + Rust command, composed) ───
    // Rust command builds via cargo on `wasm32-wasip1`; we then wrap it
    // into a component and compose against the Zig adder. Skipped when the
    // `wasm32-wasip1` rustup target is unavailable (see the `rust-examples`
    // option / probe in `build`).
    if (rust_examples) {
        const cargo = b.addSystemCommand(&.{
            "cargo",                                           "build",
            "--release",                                       "--target",
            "wasm32-wasip1",                                   "--manifest-path",
            "examples/mixed-zig-rust-calc/command/Cargo.toml",
        });
        cargo.setName("cargo build (mixed-zig-rust-calc command)");
        // Cargo writes its outputs to a deterministic path; we surface the
        // wasm via a follow-up `cp` so downstream addFileArg gets a
        // build-graph-tracked LazyPath.
        const cargo_pickup = b.addSystemCommand(&.{
            "cp",
            "examples/mixed-zig-rust-calc/command/target/wasm32-wasip1/release/mixed_zig_rust_command.wasm",
        });
        cargo_pickup.step.dependOn(&cargo.step);
        const rust_core = cargo_pickup.addOutputFileArg("mixed_zig_rust_command.core.wasm");

        // Kebab-case basename for `wabt component compose`.
        const rust_cmd = makeComponent(b, .{
            .core = rust_core,
            .wit_dir = "examples/mixed-zig-rust-calc/command/wit",
            .world = "app",
            .output = "mixed-rust-command.wasm",
        });

        const mixed_compose = b.addSystemCommand(&.{ "wabt", "component", "compose", "-d" });
        mixed_compose.addFileArg(adder);
        mixed_compose.addFileArg(rust_cmd);
        mixed_compose.addArg("-o");
        const mixed_final = mixed_compose.addOutputFileArg("mixed-zig-rust-calc.composed.wasm");
        installAndValidate(b, examples_step, mixed_final, "mixed-zig-rust-calc.wasm");

        // Run the composed Rust-command + Zig-adder. Same alias-walking
        // path as `zig-calculator-cmd` (issue #355); produces the same
        // two-line output. With the wabt-bundled adapter (#453) wamr +
        // wasmtime both run the composed component end-to-end.
        wireComponentRun(b, runs, wamr_exe, mixed_final, "40 + 2 = 42\n100 + 200 = 300\n", 0, .{ .skip_wamr = !aot_broken_components });
    }

    // ── zig-http (Zig wasi:http/incoming-handler component) ────────
    // Mirrors the bytecodealliance Rust HTTP-in-components tutorial:
    // `GET /` → `200 "Hello, world!\n"`, anything else → `404`.
    // Built `wasm32-freestanding` so no preview1 imports leak into
    // the core wasm — wabt's plain (non-adapter) component_new
    // wraps it directly. `cabi_realloc` is exported alongside
    // `wasi:http/incoming-handler@0.2.6#handle` for the canonical
    // ABI lifts of `option<string>` / `list<u8>` payloads the host
    // materializes into guest memory.
    const http_core = compileZigWasm(b, .{
        .source = "examples/zig-http/src/main.zig",
        .target_triple = "wasm32-freestanding",
        .exports = &.{ "wasi:http/incoming-handler@0.2.6#handle", "cabi_realloc" },
        .output = "zig-http.core.wasm",
        .imports = &.{
            .{ .name = "wasi_http", .path = wasip2.path("src/wasi_http.zig"), .deps = &.{"abi"} },
            .{ .name = "abi", .path = wasip2.path("src/abi.zig"), .root_dep = false },
        },
    });
    const http_component = makeComponent(b, .{
        .core = http_core,
        .wit_dir = "examples/zig-http/wit",
        .world = "http-hello",
        .output = "zig-http.wasm",
    });
    installAndValidate(b, examples_step, http_component, "zig-http.wasm");

    // End-to-end serve smoke: a small driver (tests/component-http-smoke/
    // driver.zig) spawns `wamr serve --addr=127.0.0.1:<port>` against the
    // built component, then hits `/` and `/missing` over TCP and asserts
    // the expected `200 "Hello, world!\n"` / `404` shapes. Mirrors the
    // wasi-sock driver pattern (#437).
    //
    // Wamr-only for now — `wireComponentRun` registers both arms on a
    // single stdout/exit-code assertion, which doesn't fit a serve loop.
    // Cross-runtime parity through `wasmtime serve -Scli -Shttp` is a
    // future follow-up.
    const http_smoke_driver = b.addExecutable(.{
        .name = "component-http-smoke-driver",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/component-http-smoke/driver.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });
    const run_http_smoke = b.addRunArtifact(http_smoke_driver);
    run_http_smoke.addFileArg(wamr_exe.getEmittedBin());
    run_http_smoke.addFileArg(http_component);
    // Fixed port; CI is single-tenant so collisions are unlikely. Override
    // by editing this literal if it ever conflicts.
    run_http_smoke.addArg("18080");
    run_http_smoke.expectExitCode(0);
    if (aot_broken_components) {
        // Gated off CI under `-Daot-broken-components=false` (#662). The
        // wasi:http/incoming-handler core imports cross-instance functions
        // the AOT host bridge does not yet wire.
        runs.wamr.dependOn(&run_http_smoke.step);
    }

    // ── zig-http-petstore (Zig wasi:http handler, TypeSpec petstore) ──
    // Implements the Microsoft TypeSpec petstore sample API
    // (packages/samples/specs/petstore/petstore.tsp) over
    // `wasi:http/incoming-handler@0.2.6`. A strict superset of zig-http:
    // it also reads the request method + body (`POST /pets`) and sets a
    // `content-type: application/json` response header. Same
    // `wasm32-freestanding` + `cabi_realloc` build shape as zig-http.
    const petstore_core = compileZigWasm(b, .{
        .source = "examples/zig-http-petstore/src/main.zig",
        .target_triple = "wasm32-freestanding",
        .exports = &.{ "wasi:http/incoming-handler@0.2.6#handle", "cabi_realloc" },
        .output = "zig-http-petstore.core.wasm",
        .imports = &.{
            .{ .name = "wasi_http", .path = wasip2.path("src/wasi_http.zig"), .deps = &.{"abi"} },
            .{ .name = "wasi_keyvalue", .path = wasip2.path("src/wasi_keyvalue.zig"), .deps = &.{"abi"} },
            .{ .name = "abi", .path = wasip2.path("src/abi.zig"), .root_dep = false },
        },
    });
    const petstore_component = makeComponent(b, .{
        .core = petstore_core,
        .wit_dir = "examples/zig-http-petstore/wit",
        .world = "petstore",
        .output = "zig-http-petstore.wasm",
    });
    installAndValidate(b, examples_step, petstore_component, "zig-http-petstore.wasm");

    // End-to-end serve smoke: the driver spawns `wamr serve --addr=…`
    // against the built component, then exercises the petstore routes
    // (GET/POST/DELETE on /pets, /pets/{id}, /pets/{id}/toys) over TCP.
    // Wamr-only, same rationale as the zig-http smoke above.
    const petstore_smoke_driver = b.addExecutable(.{
        .name = "component-http-petstore-smoke-driver",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/component-http-petstore-smoke/driver.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });
    const run_petstore_smoke = b.addRunArtifact(petstore_smoke_driver);
    run_petstore_smoke.addFileArg(wamr_exe.getEmittedBin());
    run_petstore_smoke.addFileArg(petstore_component);
    // Fixed port; distinct from the zig-http smoke's 18080.
    run_petstore_smoke.addArg("18081");
    run_petstore_smoke.expectExitCode(0);
    if (aot_broken_components) {
        // Gated off CI under `-Daot-broken-components=false` (#662),
        // same as the zig-http smoke.
        runs.wamr.dependOn(&run_petstore_smoke.step);
    }

    return runs;
}

const ComponentRunSteps = struct {
    /// Parent build-step for the wamr in-tree run.
    wamr: *std.Build.Step,
    /// Parent build-step for the cross-runtime parity run under
    /// `wasmtime run -S cli-exit-with-code` (assumed on `PATH`;
    /// wasmtime v44+ required for the `cli-exit-with-code`
    /// `@unstable` feature gate).
    wasmtime: *std.Build.Step,
};

/// Register one component fixture against both run parents. The wamr
/// arm spawns the freshly-built `wamrc run` (which compiles the
/// component into sibling `<stem>.cwasm.json` + `<stem>.<N>.cwasm`
/// files if missing or stale and then spawns `wamr run` itself with
/// stdio inherited). The wasmtime arm uses `wasmtime run -S
/// cli-exit-with-code`. Both arms assert the same expected stdout
/// + exit code, so byte-equivalent output across runtimes is the
/// parity invariant the CI cross-validation job enforces
/// (cataggar/wamr#457).
fn wireComponentRun(
    b: *std.Build,
    runs: ComponentRunSteps,
    wamr_exe: *std.Build.Step.Compile,
    component: std.Build.LazyPath,
    expected_stdout: []const u8,
    expected_exit: u8,
    opts: WireRunOptions,
) void {
    _ = wamr_exe;
    if (!opts.skip_wamr) {
        const wamrc_path = b.getInstallPath(.bin, "wamrc");
        const wamr_path = b.getInstallPath(.bin, "wamr");
        const run = b.addSystemCommand(&.{ wamrc_path, "run" });
        run.addFileArg(component);
        run.setEnvironmentVariable("WAMR_BIN", wamr_path);
        run.step.dependOn(b.getInstallStep());
        run.expectExitCode(expected_exit);
        run.expectStdOutEqual(expected_stdout);
        runs.wamr.dependOn(&run.step);
    }
    {
        const run = b.addSystemCommand(&.{ "wasmtime", "run", "-S", "cli-exit-with-code" });
        run.addFileArg(component);
        run.expectExitCode(expected_exit);
        run.expectStdOutEqual(expected_stdout);
        runs.wasmtime.dependOn(&run.step);
    }
}

/// Options for `wireComponentRun`. `skip_wamr` is set when the fixture is
/// known to fail under the AOT-only `wamr run` policy introduced in #644 /
/// #661 and the build is being run with `-Daot-broken-components=false`
/// (the default in CI). Tracked in #662; the wasmtime arm always runs so
/// the host-side fixture itself is still exercised.
const WireRunOptions = struct {
    skip_wamr: bool = false,
};

const ZigWasmCompile = struct {
    source: []const u8,
    /// `wasm32-wasi` for command components, `wasm32-freestanding` for
    /// library components without WASI imports.
    target_triple: []const u8,
    /// Names of symbols passed via `--export=<name>`. The first element
    /// also names the entrypoint when `_start` is the only export.
    exports: []const []const u8,
    output: []const u8,
    /// Extra Zig modules made importable from the root source via
    /// `@import("<name>")`. The wasi:cli / wasi:http / wasi:keyvalue
    /// examples use this to pull in the guest-side helper modules from
    /// the `cataggar/wabt` `wasip2` dependency (sourced via
    /// `wasip2.path("src/<module>.zig")`).
    /// Modules may declare their own `deps` (e.g. each `wasi_*` helper
    /// depends on the shared `abi` module), and the dependency graph is
    /// wired via `--dep` / `-M` flags. Every name referenced as a `dep`
    /// must also appear as a module in this list (the shared `abi` module
    /// is listed once and referenced by several helpers, so it resolves
    /// to a single instance — important because it owns the sole
    /// `cabi_realloc` export).
    imports: []const ZigWasmImport = &.{},
    /// Pass `-fno-llvm` (self-hosted wasm codegen instead of LLVM).
    no_llvm: bool = false,
    /// Pass `-fno-lld` (self-hosted wasm linker instead of LLD).
    no_lld: bool = false,
};

const ZigWasmImport = struct {
    /// Import name, e.g. `wasi_http` for `@import("wasi_http")`.
    name: []const u8,
    /// Build-graph path to the module's root source file (e.g.
    /// `wasip2.path("src/wasi_http.zig")` from the `wasip2` dependency).
    path: std.Build.LazyPath,
    /// Names of other modules in the same `imports` list this module
    /// `@import`s (e.g. `&.{"abi"}`). Wired as `--dep` flags preceding
    /// this module's `-M` entry.
    deps: []const []const u8 = &.{},
    /// When true, the root source `@import`s this module directly (so it
    /// gets a `--dep` on the root). Transitive-only modules (like `abi`,
    /// reached via the `wasi_*` helpers) set this false so they are wired
    /// as modules but not injected into the root's import namespace.
    root_dep: bool = true,
};

/// Invokes `zig build-exe -target <…> -O ReleaseSmall -fno-entry --export=<…>`
/// via `b.graph.zig_exe`, capturing the output as a build-graph LazyPath.
/// When `opts.imports` is non-empty the source is passed via `-Mroot=`
/// and each module as `--dep …  -M<name>=<path>` so the import graph
/// (`root` → `wasi_*` → `abi`) is reconstructed for the standalone
/// `build-exe` invocation.
fn compileZigWasm(b: *std.Build, opts: ZigWasmCompile) std.Build.LazyPath {
    const cmd = b.addSystemCommand(&.{
        b.graph.zig_exe, "build-exe",
        "-target",       opts.target_triple,
        "-O",            "ReleaseSmall",
        "-fno-entry",
    });
    if (opts.no_llvm) {
        cmd.addArg("-fno-llvm");
    }
    if (opts.no_lld) {
        cmd.addArg("-fno-lld");
    }
    for (opts.exports) |sym| {
        cmd.addArg(b.fmt("--export={s}", .{sym}));
    }
    if (opts.imports.len == 0) {
        cmd.addFileArg(b.path(opts.source));
    } else {
        // `--dep` flags attach to the next `-M` module. The root module
        // (`-Mroot=`) gets a `--dep` for every module it imports directly;
        // each helper module then gets `--dep` flags for its own deps
        // before its `-M` entry. A `dep` name resolves to the single
        // matching `-M<name>=` module, so a shared module (`abi`) is one
        // instance across all importers.
        for (opts.imports) |imp| {
            if (!imp.root_dep) continue;
            cmd.addArg("--dep");
            cmd.addArg(imp.name);
        }
        cmd.addPrefixedFileArg("-Mroot=", b.path(opts.source));
        for (opts.imports) |imp| {
            for (imp.deps) |dep| {
                cmd.addArg("--dep");
                cmd.addArg(dep);
            }
            cmd.addPrefixedFileArg(b.fmt("-M{s}=", .{imp.name}), imp.path);
        }
    }
    const out = cmd.addPrefixedOutputFileArg("-femit-bin=", opts.output);
    cmd.setName(b.fmt("zig build-exe {s}", .{opts.output}));
    return out;
}

const ReactorComponent = struct {
    core: std.Build.LazyPath,
    /// WIT package directory to embed (`--wit`).
    wit_dir: []const u8,
    /// World to embed (`--world`).
    world: []const u8,
    /// Output basename for the produced component LazyPath.
    output: []const u8,
};

/// One-step `wabt component new --world <world> --wit <dir>` (wabt
/// ≥ v3.0.0-dev.13): embeds the `component-type:<world>` section from the
/// WIT directory, wraps the core into a component, and validates — in a
/// single call, collapsing the former `component embed` + `component new`
/// two-step. Cores with `wasi_snapshot_preview1.*` imports still get the
/// bundled adapter auto-attached.
fn makeComponent(b: *std.Build, opts: ReactorComponent) std.Build.LazyPath {
    const cmd = b.addSystemCommand(&.{ "wabt", "component", "new", "--world", opts.world, "--wit" });
    cmd.addDirectoryArg(b.path(opts.wit_dir));
    cmd.addFileArg(opts.core);
    cmd.addArg("-o");
    return cmd.addOutputFileArg(opts.output);
}

/// Validates the component and installs it under
/// `zig-out/examples/<basename>`.
fn installAndValidate(
    b: *std.Build,
    parent: *std.Build.Step,
    component: std.Build.LazyPath,
    install_basename: []const u8,
) void {
    const validate = b.addSystemCommand(&.{ "wabt", "module", "validate" });
    validate.addFileArg(component);
    validate.setName(b.fmt("wabt module validate {s}", .{install_basename}));

    const install = b.addInstallFileWithDir(
        component,
        .{ .custom = "examples" },
        install_basename,
    );
    install.step.dependOn(&validate.step);
    parent.dependOn(&install.step);
}
