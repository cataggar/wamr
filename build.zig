const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Whether the selected target CPU can execute AOT code natively.
    // Test/bench binaries that exercise AOT execution (codegen-bench,
    // spec-test-runner, coremark-aot-runner) are only installed on these
    // arches so cross-compiled release builds for e.g. riscv64 don't try
    // to compile x86-only inline asm or the AOT execution path.
    const target_arch = target.result.cpu.arch;
    const aot_executable_target = switch (target_arch) {
        .x86_64, .aarch64 => true,
        else => false,
    };
    // codegen-bench uses x86-specific `rdtsc` inline asm and only
    // exercises the x86-64 codegen path.
    const bench_target = target_arch == .x86_64;

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
    const wabt_dep = b.dependency("wabt", .{
        .target = target,
        .optimize = .ReleaseSafe,
    });

    const spec_runner_module = b.createModule(.{
        .root_source_file = b.path("src/tests/run_spec_tests.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    spec_runner_module.addImport("config", config_module);
    spec_runner_module.addImport("wamr", lib_module);
    spec_runner_module.addImport("wabt", wabt_dep.artifact("wabt").root_module);

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

    // ── Benchmark ─────────────────────────────────────────────────────
    const bench_module = b.createModule(.{
        .root_source_file = b.path("src/compiler/bench_codegen.zig"),
        .target = target,
        .optimize = .ReleaseFast,
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
    // pipeline (loader / interp / aot / interp-vs-aot diff) and leave
    // a reproducer at <crashes>/in-flight.wasm if the process aborts.
    // See src/tests/fuzz/common.zig and .github/workflows/fuzz.yml.
    const aot_harness_module = b.createModule(.{
        .root_source_file = b.path("src/tests/aot_harness.zig"),
        .target = target,
        .optimize = optimize,
    });
    aot_harness_module.addImport("wamr", lib_module);

    // ── CoreMark AOT runner ────────────────────────────────────────────
    // Loads a CoreMark wasi `.wasm` and executes it through the Zig AOT
    // backend (same pipeline as differential tests). Replaces the old
    // C-based `tests/standalone/coremark/run.sh` for gating the Zig
    // backend on real CoreMark workloads.
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
    run_coremark_nofp.addArg("tests/standalone/coremark/coremark_wasi_nofp.wasm");
    const run_coremark_fp = b.addRunArtifact(coremark_exe);
    run_coremark_fp.addArg("tests/standalone/coremark/coremark_wasi.wasm");
    const coremark_step = b.step(
        "coremark-aot",
        "Run the CoreMark wasi benchmarks through the Zig AOT backend",
    );
    coremark_step.dependOn(&run_coremark_nofp.step);
    coremark_step.dependOn(&run_coremark_fp.step);

    // ── SIMD benchmark runner ───────────────────────────────────────────
    // Builds small in-memory SIMD modules and reports interpreter vs AOT
    // status/timing.  SIMD AOT is expected to report "unsupported" until the
    // first native v128 lowering slice lands.
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
        simd_bench_step.dependOn(&run_simd_bench.step);
    }

    const fuzz_step = b.step("fuzz", "Build fuzz harnesses (loader, interp, aot, diff)");
    inline for (.{ "loader", "interp", "aot", "diff" }) |tgt| {
        const fuzz_mod = b.createModule(.{
            .root_source_file = b.path("src/tests/fuzz/" ++ tgt ++ ".zig"),
            .target = target,
            .optimize = optimize,
        });
        fuzz_mod.addImport("config", config_module);
        fuzz_mod.addImport("wamr", lib_module);
        if (std.mem.eql(u8, tgt, "aot") or std.mem.eql(u8, tgt, "diff")) {
            fuzz_mod.addImport("aot_harness", aot_harness_module);
        }

        const fuzz_exe = b.addExecutable(.{
            .name = "fuzz-" ++ tgt,
            .root_module = fuzz_mod,
        });
        const install_fuzz = b.addInstallArtifact(fuzz_exe, .{});
        fuzz_step.dependOn(&install_fuzz.step);
    }
}
