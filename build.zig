const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Whether the selected target CPU can execute AOT code natively.
    // Test/bench binaries tied to native AOT support are only installed on
    // these arches so cross-compiled release builds for e.g. riscv64 don't
    // try to compile or run the AOT execution path.
    const target_arch = target.result.cpu.arch;
    const aot_executable_target = switch (target_arch) {
        .x86_64, .aarch64 => true,
        else => false,
    };
    // codegen-bench emits x86-64 machine code but does not execute it. It can
    // run on the native AOT arches with a portable timer fallback.
    const bench_target = aot_executable_target;

    // ── Build flags ────────────────────────────────────────────────────
    const strip = b.option(bool, "strip", "Strip debug info from binaries") orelse false;
    const stack_protector = b.option(bool, "stack-protector", "Enable stack protector (requires libc)") orelse false;
    const link_libc = b.option(bool, "link-libc", "Link libc") orelse
        (stack_protector or target.result.os.tag == .wasi);
    const version_string = b.option([]const u8, "version", "Version string") orelse "dev";

    // ── Feature flags ──────────────────────────────────────────────────
    const options = b.addOptions();
    options.addOption([]const u8, "version", version_string);

    const interp = b.option(bool, "interp", "Enable interpreter") orelse true;
    options.addOption(bool, "interp", interp);

    const aot = b.option(bool, "aot", "Enable AOT support") orelse true;
    options.addOption(bool, "aot", aot);

    const fast_interp = b.option(bool, "fast_interp", "Enable fast interpreter") orelse true;
    options.addOption(bool, "fast_interp", fast_interp);

    const jit = b.option(bool, "jit", "Enable LLVM JIT") orelse false;
    options.addOption(bool, "jit", jit);

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

    const lib_wasi_threads = b.option(bool, "lib_wasi_threads", "Enable WASI threads") orelse false;
    options.addOption(bool, "lib_wasi_threads", lib_wasi_threads);

    const thread_mgr = b.option(bool, "thread_mgr", "Enable thread manager") orelse false;
    options.addOption(bool, "thread_mgr", thread_mgr);

    const debug_interp = b.option(bool, "debug_interp", "Enable interpreter debugging") orelse false;
    options.addOption(bool, "debug_interp", debug_interp);

    const bulk_memory = b.option(bool, "bulk_memory", "Enable bulk memory ops") orelse false;
    options.addOption(bool, "bulk_memory", bulk_memory);

    const shared_memory = b.option(bool, "shared_memory", "Enable shared memory") orelse false;
    options.addOption(bool, "shared_memory", shared_memory);

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

    const network_tests = b.option(
        bool,
        "network_tests",
        "Enable opt-in unit tests that perform real outbound HTTPS requests (#521)",
    ) orelse false;
    options.addOption(bool, "network_tests", network_tests);

    const skip_coldstart = b.option(bool, "skip-coldstart", "Skip cold-start budget tests (issue #395)") orelse false;

    const config_module = options.createModule();

    // ── Root module for the library ────────────────────────────────────
    const lib_module = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_module.addImport("config", config_module);

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
        .name = "wamr",
        .root_module = exe_module,
    });
    b.installArtifact(exe);

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

    // ── WASI conformance suite ────────────────────────────────────────
    // Drives the vendored `WebAssembly/wasi-testsuite` against the just-built
    // `wamr` CLI through the in-tree adapter. Skiplist entries must each
    // carry a rationale + follow-up issue number — see
    // `tests/wasi-testsuite-skip.json`. Not wired into the default `test`
    // aggregate (it requires Python 3 + the runner's deps), but the CI job
    // gates regressions on every PR. Run locally with `zig build wasi-testsuite`.
    const wasi_runner = b.addSystemCommand(&.{
        "python3",
        "tests/wasi-testsuite/test-runner/wasi_test_runner.py",
        "--test-suite",
        "tests/wasi-testsuite/tests/c/testsuite/wasm32-wasip1",
        "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip1",
        "tests/wasi-testsuite/tests/assemblyscript/testsuite/wasm32-wasip1",
        "--runtime-adapter",
        "tests/wasi-testsuite-adapter/wamr-zig.py",
        "--exclude-filter",
        "tests/wasi-testsuite-skip.json",
    });
    // Point the adapter at the freshly-installed wamr binary so we don't pick
    // up a stale system iwasm.
    wasi_runner.setEnvironmentVariable("WAMR", b.getInstallPath(.bin, "wamr"));
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
    // P3 wave A–C: #481–#487). Skip-list entries must each carry a
    // rationale + tracking issue — see `tests/wasi-p3-testsuite-skip.json`.
    // Not wired into the default `test` aggregate (it requires Python 3
    // + the runner's deps); CI gates regressions on every PR. Run
    // locally with `zig build wasi-p3-testsuite`.
    const wasi_p3_runner = b.addSystemCommand(&.{
        "python3",
        "tests/wasi-testsuite/test-runner/wasi_test_runner.py",
        "--test-suite",
        "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip3",
        "--runtime-adapter",
        "tests/wasi-testsuite-adapter/wamr-zig.py",
        "--exclude-filter",
        "tests/wasi-p3-testsuite-skip.json",
    });
    wasi_p3_runner.setEnvironmentVariable("WAMR", b.getInstallPath(.bin, "wamr"));
    wasi_p3_runner.step.dependOn(b.getInstallStep());
    const wasi_p3_testsuite_step = b.step(
        "wasi-p3-testsuite",
        "Run the WASI Preview 3 conformance gate (wasm32-wasip3 fixtures)",
    );
    wasi_p3_testsuite_step.dependOn(&wasi_p3_runner.step);

    // ── Tests ──────────────────────────────────────────────────────────
    const test_module = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("config", config_module);

    const lib_unit_tests = b.addTest(.{
        .root_module = test_module,
    });
    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

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

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);

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
    const differential_tests = b.addTest(.{
        .root_module = differential_test_module,
    });
    const run_differential_tests = b.addRunArtifact(differential_tests);
    test_step.dependOn(&run_differential_tests.step);

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
    // Compiles a tiny `wasm32-wasi` echo server that calls preview1
    // `sock_accept`/`sock_recv`/`sock_send` against an embedder-provided
    // socket preopen at fd 3, then spawns `wamr run --listen=127.0.0.1:<port>`
    // and round-trips a single payload from a host client.
    //
    // Linux-only — host sockets and the `--listen` CLI plumbing return
    // ENOSYS / NotSupported on other targets. Uses a fixed high port to
    // avoid teaching the CLI an out-of-band port-discovery channel.
    if (target.result.os.tag == .linux) {
        const echo_wasm = compileZigWasm(b, .{
            .source = "tests/wasi-sock/echo_server.zig",
            .target_triple = "wasm32-wasi",
            .exports = &.{"_start"},
            .output = "echo_server.wasm",
        });

        const driver_module = b.createModule(.{
            .root_source_file = b.path("tests/wasi-sock/driver.zig"),
            .target = target,
            .optimize = optimize,
        });
        const driver_exe = b.addExecutable(.{
            .name = "wasi-sock-driver",
            .root_module = driver_module,
        });

        const run_sock = b.addRunArtifact(driver_exe);
        run_sock.addFileArg(exe.getEmittedBin());
        run_sock.addFileArg(echo_wasm);
        run_sock.addArg("43657");
        run_sock.expectExitCode(0);

        const sock_step = b.step(
            "test-wasi-sock",
            "Run the WASI sockets end-to-end echo test (#437; Linux only)",
        );
        sock_step.dependOn(&run_sock.step);
        test_step.dependOn(&run_sock.step);
    }

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
    // examples under `examples/components/`. Opt-in: not reachable from
    // the default `zig build` or `zig build test` graphs. See
    // `examples/components/README.md` for prereqs and runtime status.
    const component_runs = addComponentExamples(b, exe);

    // ── WASI Preview 2 conformance gate (#479) ───────────────────────
    // Curated set of Preview-2 component fixtures (sources under
    // `examples/components/`) run through `./zig-out/bin/wamr` with
    // byte-exact stdout + exit-code assertions. Strict superset of
    // `component-examples-run`: every component the latter runs is
    // also a member of this gate. The named step gives CI a
    // discoverable entry point, mirroring `wasi-testsuite` for
    // Preview 1.
    //
    // Curation rationale + future-skips: `tests/wasi-p2-testsuite-skip.json`
    // (header `_comment` documents the format; mirrors the
    // wasi-testsuite-skip.json shape). Currently empty — every
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
/// `examples/components/`). Three opt-in steps are exposed:
///   * `zig build component-examples`               — build + validate all four
///   * `zig build component-examples-run`           — run them through `./zig-out/bin/wamr`
///   * `zig build component-examples-run-wasmtime`  — run them through
///                                                    `wasmtime run -S cli-exit-with-code`
///                                                    (cross-runtime parity gate; wasmtime v44+).
/// None are reachable from `zig build` or `zig build test`.
///
/// Returns the two run-step parents so the caller can layer additional
/// named entry points on top (e.g. `wasi-p2-testsuite` in #479, which
/// is a strict superset of `component-examples-run`).
///
/// Pinned versions:
///   * `cataggar/wabt` ≥ v3.0.0-dev.6 on PATH (provides `component embed`,
///     `component new`, `component compose`, `module validate`).
///     The wasi-preview1 → component adapter is embedded in `wabt` and
///     auto-attached by `wabt component new`; no external adapter fetch.
///   * `cargo` with `wasm32-wasip1` target for the mixed example
fn addComponentExamples(b: *std.Build, wamr_exe: *std.Build.Step.Compile) ComponentRunSteps {
    const examples_step = b.step(
        "component-examples",
        "Build the WebAssembly Component examples in examples/components/",
    );
    const run_step = b.step(
        "component-examples-run",
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
        "component-examples-run-wasmtime",
        "Run the runnable component examples through `wasmtime run -S cli-exit-with-code` for cross-runtime parity validation",
    );
    const runs: ComponentRunSteps = .{ .wamr = run_step, .wasmtime = run_step_wasmtime };

    // ── zig-hello ──────────────────────────────────────────────────
    // Pure-Zig WASI command: `_start` writes a greeting via fd_write.
    const hello_core = compileZigWasm(b, .{
        .source = "examples/components/zig-hello/src/main.zig",
        .target_triple = "wasm32-wasi",
        .exports = &.{"_start"},
        .output = "zig-hello.core.wasm",
    });
    const hello = makeCommandComponent(b, .{
        .name = "zig-hello",
        .core = hello_core,
    });
    installAndValidate(b, examples_step, hello, "zig-hello.component.wasm");

    // The wabt-bundled adapter lowers `fd_write(1, …)` through
    // `wasi:io/streams.blocking-write-and-flush` against
    // `wasi:cli/stdout.get-stdout`, which both runtimes flush to
    // the host's actual stdout.
    wireComponentRun(b, runs, wamr_exe, hello, "hello from zig component\n", 0);

    // ── zig-exit ───────────────────────────────────────────────────
    // Exercises the component exit-code path (issue #436): `_start`
    // writes a marker line then calls `proc_exit(7)`; through wabt's
    // bundled wasi-preview1 → preview2 adapter that becomes
    // `wasi:cli/exit.exit-with-code(7)`, which `runLoadedComponent`
    // propagates as `RunOutcome.exit_code` and `main.zig:runComponent`
    // maps to host exit code 7.
    const exit_core = compileZigWasm(b, .{
        .source = "examples/components/zig-exit/src/main.zig",
        .target_triple = "wasm32-wasi",
        .exports = &.{"_start"},
        .output = "zig-exit.core.wasm",
    });
    const exit_component = makeCommandComponent(b, .{
        .name = "zig-exit",
        .core = exit_core,
    });
    installAndValidate(b, examples_step, exit_component, "zig-exit.component.wasm");

    wireComponentRun(b, runs, wamr_exe, exit_component, "exiting with code 7\n", 7);

    // ── zig-adder ──────────────────────────────────────────────────
    // Library component (no `run`): exports `docs:adder/add@0.1.0`.
    const adder_core = compileZigWasm(b, .{
        .source = "examples/components/zig-adder/src/main.zig",
        .target_triple = "wasm32-freestanding",
        .exports = &.{"docs:adder/add@0.1.0#add"},
        .output = "zig-adder.core.wasm",
    });
    const adder_embed = b.addSystemCommand(&.{ "wabt", "component", "embed", "--world", "adder" });
    adder_embed.addDirectoryArg(b.path("examples/components/zig-adder/wit"));
    adder_embed.addFileArg(adder_core);
    adder_embed.addArg("-o");
    const adder_embedded = adder_embed.addOutputFileArg("zig-adder.embed.wasm");

    const adder_new = b.addSystemCommand(&.{ "wabt", "component", "new" });
    adder_new.addFileArg(adder_embedded);
    adder_new.addArg("-o");
    // `wabt component compose` (like `wasm-tools compose`) requires
    // kebab-case file basenames (no dots before the .wasm extension).
    // Emit `zig-adder.wasm` for use as a compose dependency below; the
    // install copy uses the more descriptive `.component.wasm` suffix
    // for end-user discoverability.
    const adder = adder_new.addOutputFileArg("zig-adder.wasm");
    installAndValidate(b, examples_step, adder, "zig-adder.component.wasm");

    // ── zig-calculator-cmd (Zig command importing zig-adder) ───────
    const calc_core = compileZigWasm(b, .{
        .source = "examples/components/zig-calculator-cmd/src/main.zig",
        .target_triple = "wasm32-wasi",
        .exports = &.{"_start"},
        .output = "zig-calculator-cmd.core.wasm",
    });
    const calc_embed = b.addSystemCommand(&.{ "wabt", "component", "embed", "--world", "app" });
    calc_embed.addDirectoryArg(b.path("examples/components/zig-calculator-cmd/wit"));
    calc_embed.addFileArg(calc_core);
    calc_embed.addArg("-o");
    const calc_embedded = calc_embed.addOutputFileArg("zig-calculator-cmd.embed.wasm");

    const calc_new = b.addSystemCommand(&.{ "wabt", "component", "new" });
    calc_new.addFileArg(calc_embedded);
    calc_new.addArg("-o");
    // Kebab-case basename for `wabt component compose` consumption.
    const calc_cmd = calc_new.addOutputFileArg("zig-calculator-cmd.wasm");

    // Compose: link `docs:adder/add@0.1.0` import against the Zig adder.
    const calc_compose = b.addSystemCommand(&.{ "wabt", "component", "compose", "-d" });
    calc_compose.addFileArg(adder);
    calc_compose.addFileArg(calc_cmd);
    calc_compose.addArg("-o");
    const calc_final = calc_compose.addOutputFileArg("zig-calculator-cmd.composed.wasm");
    installAndValidate(b, examples_step, calc_final, "zig-calculator-cmd.composed.wasm");

    // Run the composed Zig calculator command. The wabt-bundled
    // wasi-preview1 adapter lowers `fd_write(1, …)` through
    // `wasi:io/streams.blocking-write-and-flush`, which both runtimes
    // flush to the host's actual stdout. Issue #355 wired the alias
    // chain + sub-component instantiation that this composed command
    // needs to reach its `wasi:cli/run` export.
    wireComponentRun(b, runs, wamr_exe, calc_final, "40 + 2 = 42\n100 + 200 = 300\n", 0);

    // ── mixed-zig-rust-calc (Zig adder + Rust command, composed) ───
    // Rust command builds via cargo on `wasm32-wasip1`; we then run the
    // standard wabt component embed/new pipeline and compose against
    // the Zig adder. Build is opt-in — failure mode if cargo / target
    // not present is a clear cargo error.
    const cargo = b.addSystemCommand(&.{
        "cargo",                                                      "build",
        "--release",                                                  "--target",
        "wasm32-wasip1",                                              "--manifest-path",
        "examples/components/mixed-zig-rust-calc/command/Cargo.toml",
    });
    cargo.setName("cargo build (mixed-zig-rust-calc command)");
    // Cargo writes its outputs to a deterministic path; we surface the
    // wasm via a follow-up `cp` so downstream addFileArg gets a
    // build-graph-tracked LazyPath.
    const cargo_pickup = b.addSystemCommand(&.{
        "cp",
        "examples/components/mixed-zig-rust-calc/command/target/wasm32-wasip1/release/mixed_zig_rust_command.wasm",
    });
    cargo_pickup.step.dependOn(&cargo.step);
    const rust_core = cargo_pickup.addOutputFileArg("mixed_zig_rust_command.core.wasm");

    const rust_embed = b.addSystemCommand(&.{ "wabt", "component", "embed", "--world", "app" });
    rust_embed.addDirectoryArg(b.path("examples/components/mixed-zig-rust-calc/command/wit"));
    rust_embed.addFileArg(rust_core);
    rust_embed.addArg("-o");
    const rust_embedded = rust_embed.addOutputFileArg("mixed-rust-command.embed.wasm");

    const rust_new = b.addSystemCommand(&.{ "wabt", "component", "new" });
    rust_new.addFileArg(rust_embedded);
    rust_new.addArg("-o");
    // Kebab-case basename for `wabt component compose`.
    const rust_cmd = rust_new.addOutputFileArg("mixed-rust-command.wasm");

    const mixed_compose = b.addSystemCommand(&.{ "wabt", "component", "compose", "-d" });
    mixed_compose.addFileArg(adder);
    mixed_compose.addFileArg(rust_cmd);
    mixed_compose.addArg("-o");
    const mixed_final = mixed_compose.addOutputFileArg("mixed-zig-rust-calc.composed.wasm");
    installAndValidate(b, examples_step, mixed_final, "mixed-zig-rust-calc.composed.wasm");

    // Run the composed Rust-command + Zig-adder. Same alias-walking
    // path as `zig-calculator-cmd` (issue #355); produces the same
    // two-line output. With the wabt-bundled adapter (#453) wamr +
    // wasmtime both run the composed component end-to-end.
    wireComponentRun(b, runs, wamr_exe, mixed_final, "40 + 2 = 42\n100 + 200 = 300\n", 0);

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
        .source = "examples/components/zig-http/src/main.zig",
        .target_triple = "wasm32-freestanding",
        .exports = &.{ "wasi:http/incoming-handler@0.2.6#handle", "cabi_realloc" },
        .output = "zig-http.core.wasm",
    });
    const http_embed = b.addSystemCommand(&.{ "wabt", "component", "embed", "--world", "http-hello" });
    http_embed.addDirectoryArg(b.path("examples/components/zig-http/wit"));
    http_embed.addFileArg(http_core);
    http_embed.addArg("-o");
    const http_embedded = http_embed.addOutputFileArg("zig-http.embed.wasm");

    const http_new = b.addSystemCommand(&.{ "wabt", "component", "new" });
    http_new.addFileArg(http_embedded);
    http_new.addArg("-o");
    const http_component = http_new.addOutputFileArg("zig-http.component.wasm");
    installAndValidate(b, examples_step, http_component, "zig-http.component.wasm");

    // End-to-end serve smoke: a small driver (tests/component-http-smoke/
    // driver.zig) spawns `wamr run --listen=127.0.0.1:<port>` against the
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
    runs.wamr.dependOn(&run_http_smoke.step);

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

/// Register one component fixture against both run parents. Wires
/// up the wamr arm via `b.addRunArtifact(wamr_exe)` and the wasmtime
/// arm via `b.addSystemCommand({"wasmtime", "run", "-S",
/// "cli-exit-with-code"})`. Both arms assert the same expected
/// stdout + exit code, so byte-equivalent output across runtimes is
/// the parity invariant the CI cross-validation job enforces
/// (cataggar/wamr#457).
fn wireComponentRun(
    b: *std.Build,
    runs: ComponentRunSteps,
    wamr_exe: *std.Build.Step.Compile,
    component: std.Build.LazyPath,
    expected_stdout: []const u8,
    expected_exit: u8,
) void {
    {
        const run = b.addRunArtifact(wamr_exe);
        run.addArg("run");
        run.addFileArg(component);
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

const ZigWasmCompile = struct {
    source: []const u8,
    /// `wasm32-wasi` for command components, `wasm32-freestanding` for
    /// library components without WASI imports.
    target_triple: []const u8,
    /// Names of symbols passed via `--export=<name>`. The first element
    /// also names the entrypoint when `_start` is the only export.
    exports: []const []const u8,
    output: []const u8,
};

/// Invokes `zig build-exe -target <…> -O ReleaseSmall -fno-entry --export=<…>`
/// via `b.graph.zig_exe`, capturing the output as a build-graph LazyPath.
fn compileZigWasm(b: *std.Build, opts: ZigWasmCompile) std.Build.LazyPath {
    const cmd = b.addSystemCommand(&.{
        b.graph.zig_exe, "build-exe",
        "-target",       opts.target_triple,
        "-O",            "ReleaseSmall",
        "-fno-entry",
    });
    for (opts.exports) |sym| {
        cmd.addArg(b.fmt("--export={s}", .{sym}));
    }
    cmd.addFileArg(b.path(opts.source));
    const out = cmd.addPrefixedOutputFileArg("-femit-bin=", opts.output);
    cmd.setName(b.fmt("zig build-exe {s}", .{opts.output}));
    return out;
}

const CommandComponent = struct {
    name: []const u8,
    core: std.Build.LazyPath,
};

/// Wraps `wabt component new`. The wasi-preview1 → component
/// adapter is bundled inside wabt and auto-attached when the
/// embed has unresolved `wasi_snapshot_preview1.*` imports
/// (see `cataggar/wabt#156`); no `--adapt` plumbing required.
fn makeCommandComponent(b: *std.Build, opts: CommandComponent) std.Build.LazyPath {
    const cmd = b.addSystemCommand(&.{ "wabt", "component", "new" });
    cmd.addFileArg(opts.core);
    cmd.addArg("-o");
    return cmd.addOutputFileArg(b.fmt("{s}.component.wasm", .{opts.name}));
}

/// Validates the component and installs it under
/// `zig-out/component-examples/<basename>`.
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
        .{ .custom = "component-examples" },
        install_basename,
    );
    install.step.dependOn(&validate.step);
    parent.dependOn(&install.step);
}
