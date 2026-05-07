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
    // examples under `tests/component/src/`. Opt-in: not reachable from
    // the default `zig build` or `zig build test` graphs. See
    // `tests/component/README.md` for prereqs and runtime status.
    addComponentExamples(b, exe);
}

/// Wires up the Component-Model example pipeline (sources under
/// `tests/component/src/`). Two opt-in steps are exposed:
///   * `zig build component-examples`     — build + validate all four
///   * `zig build component-examples-run` — runs `zig-hello` through `wamr`
/// Neither is reachable from `zig build` or `zig build test`.
///
/// Pinned versions:
///   * Wasmtime preview1 → component adapter v36.0.9 (sha256 verified)
///   * `wasm-tools` ≥ 1.220 expected on PATH (also provides `validate`/`compose`)
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
        "Build the WebAssembly Component examples in tests/component/src/",
    );
    const run_step = b.step(
        "component-examples-run",
        "Run the runnable component examples (zig-hello) through ./zig-out/bin/wamr",
    );

    // ── zig-hello ──────────────────────────────────────────────────
    // Pure-Zig WASI command: `_start` writes a greeting via fd_write.
    const hello_core = compileZigWasm(b, .{
        .source = "tests/component/src/zig-hello/src/main.zig",
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
    // fd 2 today; tracking issue separate from this PR). Pin the
    // assertion to the observed behaviour so the run step is stable.
    const run_hello = b.addRunArtifact(wamr_exe);
    run_hello.addFileArg(hello);
    run_hello.expectExitCode(0);
    run_hello.expectStdErrEqual("hello from zig component\n");
    run_step.dependOn(&run_hello.step);

    // ── zig-adder ──────────────────────────────────────────────────
    // Library component (no `run`): exports `docs:adder/add@0.1.0`.
    const adder_core = compileZigWasm(b, .{
        .source = "tests/component/src/zig-adder/src/main.zig",
        .target_triple = "wasm32-freestanding",
        .exports = &.{"docs:adder/add@0.1.0#add"},
        .output = "zig-adder.core.wasm",
    });
    const adder_embed = b.addSystemCommand(&.{ "wasm-tools", "component", "embed", "--world", "adder" });
    adder_embed.addDirectoryArg(b.path("tests/component/src/zig-adder/wit"));
    adder_embed.addFileArg(adder_core);
    adder_embed.addArg("-o");
    const adder_embedded = adder_embed.addOutputFileArg("zig-adder.embed.wasm");

    const adder_new = b.addSystemCommand(&.{ "wasm-tools", "component", "new" });
    adder_new.addFileArg(adder_embedded);
    adder_new.addArg("-o");
    // `wasm-tools compose` requires kebab-case file basenames (no dots
    // before the .wasm extension). Emit `zig-adder.wasm` for use as a
    // compose dependency below; the install copy uses the more
    // descriptive `.component.wasm` suffix for end-user discoverability.
    const adder = adder_new.addOutputFileArg("zig-adder.wasm");
    installAndValidate(b, examples_step, adder, "zig-adder.component.wasm");

    // ── zig-calculator-cmd (Zig command importing zig-adder) ───────
    const calc_core = compileZigWasm(b, .{
        .source = "tests/component/src/zig-calculator-cmd/src/main.zig",
        .target_triple = "wasm32-wasi",
        .exports = &.{"_start"},
        .output = "zig-calculator-cmd.core.wasm",
    });
    const calc_embed = b.addSystemCommand(&.{ "wasm-tools", "component", "embed", "--world", "app" });
    calc_embed.addDirectoryArg(b.path("tests/component/src/zig-calculator-cmd/wit"));
    calc_embed.addFileArg(calc_core);
    calc_embed.addArg("-o");
    const calc_embedded = calc_embed.addOutputFileArg("zig-calculator-cmd.embed.wasm");

    const calc_new = b.addSystemCommand(&.{ "wasm-tools", "component", "new" });
    calc_new.addFileArg(calc_embedded);
    calc_new.addArg("--adapt");
    calc_new.addPrefixedFileArg("wasi_snapshot_preview1=", adapter);
    calc_new.addArg("-o");
    // Kebab-case basename for `wasm-tools compose` consumption.
    const calc_cmd = calc_new.addOutputFileArg("zig-calculator-cmd.wasm");

    // Compose: link `docs:adder/add@0.1.0` import against the Zig adder.
    // wasm-tools compose is deprecated upstream in favour of `wac`; we use
    // it because it ships with wasm-tools 1.220.
    const calc_compose = b.addSystemCommand(&.{ "wasm-tools", "compose", "-d" });
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
    run_calc.addFileArg(calc_final);
    run_calc.expectExitCode(0);
    run_calc.expectStdErrEqual("40 + 2 = 42\n100 + 200 = 300\n");
    run_step.dependOn(&run_calc.step);

    // ── mixed-zig-rust-calc (Zig adder + Rust command, composed) ───
    // Rust command builds via cargo on `wasm32-wasip1`; we then run the
    // standard wasm-tools embed/new pipeline and compose against the
    // Zig adder. Build is opt-in — failure mode if cargo / target not
    // present is a clear cargo error.
    const cargo = b.addSystemCommand(&.{
        "cargo",                                                      "build",
        "--release",                                                  "--target",
        "wasm32-wasip1",                                              "--manifest-path",
        "tests/component/src/mixed-zig-rust-calc/command/Cargo.toml",
    });
    cargo.setName("cargo build (mixed-zig-rust-calc command)");
    // Cargo writes its outputs to a deterministic path; we surface the
    // wasm via a follow-up `cp` so downstream addFileArg gets a
    // build-graph-tracked LazyPath.
    const cargo_pickup = b.addSystemCommand(&.{
        "cp",
        "tests/component/src/mixed-zig-rust-calc/command/target/wasm32-wasip1/release/mixed_zig_rust_command.wasm",
    });
    cargo_pickup.step.dependOn(&cargo.step);
    const rust_core = cargo_pickup.addOutputFileArg("mixed_zig_rust_command.core.wasm");

    const rust_embed = b.addSystemCommand(&.{ "wasm-tools", "component", "embed", "--world", "app" });
    rust_embed.addDirectoryArg(b.path("tests/component/src/mixed-zig-rust-calc/command/wit"));
    rust_embed.addFileArg(rust_core);
    rust_embed.addArg("-o");
    const rust_embedded = rust_embed.addOutputFileArg("mixed-rust-command.embed.wasm");

    const rust_new = b.addSystemCommand(&.{ "wasm-tools", "component", "new" });
    rust_new.addFileArg(rust_embedded);
    rust_new.addArg("--adapt");
    rust_new.addPrefixedFileArg("wasi_snapshot_preview1=", adapter);
    rust_new.addArg("-o");
    // Kebab-case basename for `wasm-tools compose`.
    const rust_cmd = rust_new.addOutputFileArg("mixed-rust-command.wasm");

    const mixed_compose = b.addSystemCommand(&.{ "wasm-tools", "compose", "-d" });
    mixed_compose.addFileArg(adder);
    mixed_compose.addFileArg(rust_cmd);
    mixed_compose.addArg("-o");
    const mixed_final = mixed_compose.addOutputFileArg("mixed-zig-rust-calc.composed.wasm");
    installAndValidate(b, examples_step, mixed_final, "mixed-zig-rust-calc.composed.wasm");

    // Run the composed Rust-command + Zig-adder. Same alias-walking
    // path as `zig-calculator-cmd` (issue #355); produces the same
    // two-line output, again on host stderr.
    const run_mixed = b.addRunArtifact(wamr_exe);
    run_mixed.addFileArg(mixed_final);
    run_mixed.expectExitCode(0);
    run_mixed.expectStdErrEqual("40 + 2 = 42\n100 + 200 = 300\n");
    run_step.dependOn(&run_mixed.step);
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

/// Wraps `wasm-tools component new --adapt wasi_snapshot_preview1=<adapter>`.
fn makeCommandComponent(b: *std.Build, opts: CommandComponent) std.Build.LazyPath {
    const cmd = b.addSystemCommand(&.{ "wasm-tools", "component", "new" });
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
    const validate = b.addSystemCommand(&.{ "wasm-tools", "validate" });
    validate.addFileArg(component);
    validate.setName(b.fmt("wasm-tools validate {s}", .{install_basename}));

    const install = b.addInstallFileWithDir(
        component,
        .{ .custom = "component-examples" },
        install_basename,
    );
    install.step.dependOn(&validate.step);
    parent.dependOn(&install.step);
}
