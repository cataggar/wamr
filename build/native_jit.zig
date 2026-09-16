const std = @import("std");

pub fn addTests(b: *std.Build, wamrc: *std.Build.Step.Compile, optimize: std.builtin.OptimizeMode) void {
    const wasm = workload(b);
    const fixture_step = b.step("native-jit-fixture", "Install the deterministic no-import JIT workload");
    fixture_step.dependOn(&b.addInstallFile(wasm, "fixtures/native-jit.wasm").step);
    const files = b.addWriteFiles();
    _ = files.addCopyFile(wasm, "fixture.wasm");
    const fixture = files.add("fixture.zig", "pub const bytes = @embedFile(\"fixture.wasm\");\n");
    addBenchmark(b, wamrc, wasm, optimize);
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/jit_native_tests.zig"),
            .target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .musl }),
            .optimize = .ReleaseSafe,
            .single_threaded = true,
        }),
        .filters = &.{"native JIT"},
    });
    configure(b, tests.root_module);
    tests.root_module.addAnonymousImport("jit_fixture", .{ .root_source_file = fixture });
    const step = b.step("test-native-jit", "Run embedded wasm through opt-in native fast/full JIT");
    if (b.graph.host.result.cpu.arch == .x86_64 and b.graph.host.result.os.tag == .linux) {
        step.dependOn(&b.addRunArtifact(tests).step);
    } else {
        const run = b.addSystemCommand(&.{ "qemu-x86_64", "-cpu", "max" });
        run.addArtifactArg(tests);
        step.dependOn(&run.step);
    }
}

fn workload(b: *std.Build) std.Build.LazyPath {
    const wasm = b.addExecutable(.{
        .name = "native-jit-fixture",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/unikraft-jit/fixture.zig"),
            .target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding }),
            .optimize = .ReleaseSmall,
        }),
    });
    wasm.entry = .disabled;
    wasm.rdynamic = true;
    wasm.stack_size = 16384;
    wasm.initial_memory = 131072;
    wasm.max_memory = 524288;
    return wasm.getEmittedBin();
}

fn addBenchmark(b: *std.Build, wamrc: *std.Build.Step.Compile, wasm: std.Build.LazyPath, optimize: std.builtin.OptimizeMode) void {
    const compile = b.addRunArtifact(wamrc);
    compile.addArgs(&.{ "compile", "--target=x86_64", "--profile=unikraft-x86_64" });
    compile.addFileArg(wasm);
    compile.addArg("-o");
    const aot = compile.addOutputFileArg("matched.cwasm");
    const files = b.addWriteFiles();
    _ = files.addCopyFile(wasm, "matched.wasm");
    _ = files.addCopyFile(aot, "matched.cwasm");
    const fixture = files.add("matched.zig",
        \\const std = @import("std");
        \\pub const wasm = @embedFile("matched.wasm");
        \\pub const aot = @embedFile("matched.cwasm");
        \\pub const wasm_size = wasm.len;
        \\pub const wasm_sha256 = blk: {
        \\    @setEvalBranchQuota(100000);
        \\    var digest: [32]u8 = undefined;
        \\    std.crypto.hash.sha2.Sha256.hash(wasm, &digest, .{});
        \\    break :blk std.fmt.bytesToHex(digest, .lower);
        \\};
        \\
    );
    const target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .musl });
    const minimal = b.createModule(.{
        .root_source_file = b.path("src/wasi/minimal.zig"),
        .target = target,
        .optimize = optimize,
    });
    const step = b.step("native-jit-bench", "Build explicit fast/full sampler and separately compiler-free AOT comparator");
    inline for (.{ true, false }) |with_compiler| {
        const name = if (with_compiler) "wamr-native-jit-bench" else "wamr-native-jit-aot-compare";
        var options = @import("native_aot.zig").moduleOptions(target, optimize);
        options.root_source_file = b.path(if (with_compiler) "src/native_jit_bench.zig" else "src/native_jit_aot_compare.zig");
        const module = b.createModule(options);
        module.addImport("minimal-wasi", minimal);
        module.addAnonymousImport("jit_bench_fixture", .{ .root_source_file = fixture });
        if (with_compiler) configure(b, module);
        const executable = b.addExecutable(.{ .name = name, .root_module = module });
        step.dependOn(&b.addInstallArtifact(executable, .{}).step);
    }
    step.dependOn(&b.addInstallFile(wasm, "native-jit-bench/matched.wasm").step);
    step.dependOn(&b.addInstallFile(aot, "native-jit-bench/matched.cwasm").step);
    const tests = b.addSystemCommand(&.{ "python3", "-m", "unittest", "scripts.test_native_jit_benchmark" });
    tests.setEnvironmentVariable("WAMR_JIT_BENCH_AOT", b.getInstallPath(.bin, "wamr-native-jit-aot-compare"));
    tests.setEnvironmentVariable("WAMR_JIT_BENCH_JIT", b.getInstallPath(.bin, "wamr-native-jit-bench"));
    if (b.graph.host.result.cpu.arch != .x86_64)
        tests.setEnvironmentVariable("WAMR_JIT_BENCH_RUNNER", "qemu-x86_64 -cpu max");
    tests.step.dependOn(step);
    const test_step = b.step("test-native-jit-bench", "Validate real matched fast/full/AOT sampling and opt-in qualification gates");
    test_step.dependOn(&tests.step);
}

pub fn configure(b: *std.Build, module: *std.Build.Module) void {
    const options = b.addOptions();
    options.addOption(bool, "unikraft_jit", true);
    options.addOption(bool, "aot", true);
    options.addOption(bool, "jit", true);
    options.addOption(bool, "lib_wasi_threads", false);
    module.addOptions("config", options);
}

pub fn build(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    if (target.result.cpu.arch != .x86_64 or target.result.os.tag != .freestanding or target.result.abi != .none)
        std.debug.panic("unikraft-jit requires x86_64-freestanding-none", .{});
    inline for (.{ "interp", "fast_interp", "lazy_jit", "fast_jit", "component_model", "lib_pthread", "lib_wasi_threads", "thread_mgr", "shared_memory", "link-libc", "libc_builtin", "libc_wasi", "simd", "memory64", "multi_memory", "gc", "exce_handling" }) |name| {
        if (b.option(bool, name, "Unsupported in the native JIT profile") orelse false)
            std.debug.panic("unikraft-jit excludes -D{s}=true", .{name});
    }
    var module_options = @import("native_aot.zig").moduleOptions(target, optimize);
    module_options.root_source_file = b.path("src/jit_native.zig");
    const module = b.addModule("wamr-jit", module_options);
    configure(b, module);
    const library = b.addLibrary(.{ .name = "wamr-jit", .linkage = .static, .root_module = module });
    library.bundle_compiler_rt = true;
    b.installArtifact(library);
    const check = b.addExecutable(.{ .name = "wamr-jit-link-check", .root_module = module });
    check.entry = .{ .symbol_name = "wamr_jit_link_check" };
    check.rdynamic = true;
    _ = check.getEmittedBin();
    const step = b.step("native-jit-check", "Link opt-in compiler with no hosted dependencies");
    step.dependOn(&check.step);
    b.getInstallStep().dependOn(step);

    var compare_options = @import("native_aot.zig").moduleOptions(target, optimize);
    compare_options.root_source_file = b.path("src/jit_compare_native.zig");
    const compare = b.addModule("wamr-jit-aot-sample", compare_options);
    const compare_library = b.addLibrary(.{ .name = "wamr-jit-aot-sample", .linkage = .static, .root_module = compare });
    compare_library.bundle_compiler_rt = true;
    b.installArtifact(compare_library);
    const compare_check = b.addExecutable(.{ .name = "wamr-jit-aot-sample-link-check", .root_module = compare });
    compare_check.entry = .{ .symbol_name = "wamr_jit_aot_sample_link_check" };
    compare_check.rdynamic = true;
    _ = compare_check.getEmittedBin();
    step.dependOn(&compare_check.step);

    const wasm = workload(b);
    const files = b.addWriteFiles();
    _ = files.addCopyFile(wasm, "matched.wasm");
    const embedded = files.add("workload.zig", "pub const wasm = @embedFile(\"matched.wasm\");\n");
    const embedded_module = b.addModule("wamr-jit-workload", .{ .root_source_file = embedded, .target = target, .optimize = optimize });
    b.getInstallStep().dependOn(&b.addInstallFile(wasm, "native-jit-bench/matched.wasm").step);
    inline for (.{ true, false }) |with_compiler| {
        var guest_options = @import("native_aot.zig").moduleOptions(target, optimize);
        guest_options.root_source_file = b.path("src/jit_guest_link_check.zig");
        const guest = b.createModule(guest_options);
        guest.addImport("guest", if (with_compiler) module else compare);
        guest.addImport("wamr-jit-workload", embedded_module);
        const audit_options = b.addOptions();
        audit_options.addOption(bool, "with_compiler", with_compiler);
        guest.addOptions("guest_audit_options", audit_options);
        const audit = b.addExecutable(.{
            .name = if (with_compiler) "wamr-jit-guest-link-check" else "wamr-jit-aot-guest-link-check",
            .root_module = guest,
        });
        audit.entry = .{ .symbol_name = "wamr_jit_guest_sample_link_check" };
        audit.rdynamic = true;
        _ = audit.getEmittedBin();
        step.dependOn(&audit.step);
    }
}
