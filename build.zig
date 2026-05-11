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
    addComponentExamples(b, exe);
}

/// Wires up the Component-Model example pipeline (sources under
/// `examples/components/`). Two opt-in steps are exposed:
///   * `zig build component-examples`     — build + validate all four
///   * `zig build component-examples-run` — runs `zig-hello` through `wamr`
/// Neither is reachable from `zig build` or `zig build test`.
///
/// Pinned versions:
///   * Wasmtime preview1 → component adapter v36.0.9 (sha256 verified)
///   * `cataggar/wabt` ≥ v3.0.0-dev.4 on PATH (provides `component embed`,
///     `component new`, `component compose`, `validate`)
///   * `cargo` with `wasm32-wasip1` target for the mixed example
fn addComponentExamples(b: *std.Build, wamr_exe: *std.Build.Step.Compile) void {
    const adapter_url =
        "https://github.com/bytecodealliance/wasmtime/releases/download/v36.0.9/wasi_snapshot_preview1.command.wasm";
    const adapter_sha256 =
        "2b0afc5edd1301716580c2df9d14b350529d770b54804def60c69807ed7600e0";

    // Fetch + sha256-verify the wasi-preview1 → component adapter.
    // sh -c "<script>" -- $1=output-path
    const fetch_script = b.fmt(
        "set -eu\n" ++
            "curl --fail --silent --show-error --location --output \"$1\" \"{s}\"\n" ++
            "echo \"{s}  $1\" | sha256sum -c -\n",
        .{ adapter_url, adapter_sha256 },
    );
    const fetch_adapter = b.addSystemCommand(&.{ "sh", "-c", fetch_script, "--" });
    fetch_adapter.setName("fetch wasi-preview1 adapter (v36.0.9)");
    const adapter = fetch_adapter.addOutputFileArg("wasi_snapshot_preview1.command.wasm");

    const examples_step = b.step(
        "component-examples",
        "Build the WebAssembly Component examples in examples/components/",
    );
    const run_step = b.step(
        "component-examples-run",
        "Run the runnable component examples (zig-hello) through ./zig-out/bin/wamr",
    );

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
        .adapter = adapter,
    });
    installAndValidate(b, examples_step, hello, "zig-hello.component.wasm");

    // The hello component is the only example that runs end-to-end on
    // wamr today; wire it into `component-examples-run`. wamr's
    // component-model CLI captures stdout via the wasi-cli adapter and
    // currently flushes it to the host's *stderr* fd (the captured
    // bytes go through `init.io`'s File-write dispatch which lands on
    // fd 2 today; tracking issue separate from this PR).
    //
    // The `expectStdErrEqual("hello from zig component\n")` assertion
    // is currently **disabled**: with #448's gate active, the bytes
    // route through the wasmtime preview1 adapter, but our `wabt
    // component {embed,new}` build pipeline emits an adapter-composed
    // component whose stdout path silently drops the bytes (verified:
    // even `wasmtime run` against the same component yields empty
    // output). Restore the full assertion once #453 ships a
    // wamr-native preview1 adapter built in-tree.
    const run_hello = b.addRunArtifact(wamr_exe);
    run_hello.addArg("run");
    run_hello.addFileArg(hello);
    run_hello.expectExitCode(0);
    // run_hello.expectStdErrEqual("hello from zig component\n");  // gated on #453
    run_step.dependOn(&run_hello.step);

    // ── zig-exit ───────────────────────────────────────────────────
    // Exercises the component exit-code path (issue #436): `_start`
    // writes a marker line then calls `proc_exit(7)`; through the
    // wasi-preview1 adapter that becomes `wasi:cli/exit.exit-with-code(7)`,
    // which `runLoadedComponent` propagates as `RunOutcome.exit_code`
    // and `main.zig:runComponent` maps to host exit code 7.
    const exit_core = compileZigWasm(b, .{
        .source = "examples/components/zig-exit/src/main.zig",
        .target_triple = "wasm32-wasi",
        .exports = &.{"_start"},
        .output = "zig-exit.core.wasm",
    });
    const exit_component = makeCommandComponent(b, .{
        .name = "zig-exit",
        .core = exit_core,
        .adapter = adapter,
    });
    installAndValidate(b, examples_step, exit_component, "zig-exit.component.wasm");

    const run_exit = b.addRunArtifact(wamr_exe);
    run_exit.addArg("run");
    run_exit.addFileArg(exit_component);
    run_exit.expectExitCode(7);
    // The `expectStdErrEqual("exiting with code 7\n")` assertion is
    // currently **disabled**: with #448's gate active, `fd_write`
    // routes through the wasmtime preview1 adapter rather than wamr's
    // `wasiFdWrite` host fn, and our `wabt component {embed,new}`
    // pipeline emits an adapter-composed component whose stdout path
    // drops the bytes (verified: even `wasmtime run` against the
    // produced component yields empty output). The exit-code check is
    // preserved by the `proc_exit` cross-dispatch interception added
    // alongside #448 in `src/runtime/interpreter/interp.zig`. Restore
    // the full assertion once #453 ships a wamr-native preview1
    // adapter built in-tree.
    // run_exit.expectStdErrEqual("exiting with code 7\n");  // gated on #453
    run_step.dependOn(&run_exit.step);

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
    calc_new.addArg("--adapt");
    calc_new.addPrefixedFileArg("wasi_snapshot_preview1=", adapter);
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

    // Run the composed Zig calculator command. The wasi-cli adapter
    // captures stdout into the adapter buffer and flushes via
    // `std.Io.File.stdout().writeStreamingAll`; empirically this
    // lands on host stderr (fd 2) — same as `zig-hello` above.
    // Issue #355 wired the alias chain + sub-component instantiation
    // that this composed command needs to reach its `wasi:cli/run`
    // export.
    const run_calc = b.addRunArtifact(wamr_exe);
    run_calc.addArg("run");
    run_calc.addFileArg(calc_final);
    run_calc.expectExitCode(0);
    // `expectStdErrEqual("40 + 2 = 42\n100 + 200 = 300\n")` is
    // currently **disabled** for the same reason as `zig-hello` and
    // `zig-exit` above: the `wabt component {embed,new,compose}`
    // pipeline emits adapter-composed components whose stdout path
    // drops bytes. Restore once #453 lands.
    // run_calc.expectStdErrEqual("40 + 2 = 42\n100 + 200 = 300\n");  // gated on #453
    run_step.dependOn(&run_calc.step);

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
    rust_new.addArg("--adapt");
    rust_new.addPrefixedFileArg("wasi_snapshot_preview1=", adapter);
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
    // two-line output, again on host stderr.
    //
    // **Currently disabled** in `component-examples-run`: the
    // `wabt component compose` output traps even under `wasmtime
    // run` (verified 2026-05-11), so wamr can't run it end-to-end
    // either. The example still builds + validates + installs via
    // `examples_step` (above). Re-enable in `run_step` once #453
    // ships a wamr-native preview1 adapter and the compose step is
    // moved off `cataggar/wabt`'s `component` subcommands.
    //
    // const run_mixed = b.addRunArtifact(wamr_exe);
    // run_mixed.addArg("run");
    // run_mixed.addFileArg(mixed_final);
    // run_mixed.expectExitCode(0);
    // run_mixed.expectStdErrEqual("40 + 2 = 42\n100 + 200 = 300\n");
    // run_step.dependOn(&run_mixed.step);
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
    adapter: std.Build.LazyPath,
};

/// Wraps `wabt component new --adapt wasi_snapshot_preview1=<adapter>`.
fn makeCommandComponent(b: *std.Build, opts: CommandComponent) std.Build.LazyPath {
    const cmd = b.addSystemCommand(&.{ "wabt", "component", "new" });
    cmd.addFileArg(opts.core);
    cmd.addArg("--adapt");
    cmd.addPrefixedFileArg("wasi_snapshot_preview1=", opts.adapter);
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
