const std = @import("std");

pub fn addTests(b: *std.Build, wamrc: *std.Build.Step.Compile, hosted_module: *std.Build.Module, wabt: *std.Build.Module) void {
    const abi_tests = b.addTest(.{
        .root_module = hosted_module,
        .filters = &.{
            "native embedding ABI",
            "trap_jmp:",
            "compileFunctionRA: division uses rax/rdx",
            "compileFunctionRA: div does not emit dead r11 save",
        },
    });
    const abi_step = b.step("test-native-aot-abi", "Verify hosted/native VmCtx layout and explicit trap unwind");
    abi_step.dependOn(&b.addRunArtifact(abi_tests).step);
    const policy_tests = b.addTest(.{
        // Policy tests belong to the library module, not the CLI's imports.
        .root_module = hosted_module,
        .filters = &.{"native profile:"},
    });
    const policy_step = b.step("test-native-aot-policy", "Verify compiler native-profile admission policy");
    policy_step.dependOn(&b.addRunArtifact(policy_tests).step);
    const wasm = b.addExecutable(.{
        .name = "native-aot-fixture",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/unikraft-aot/fixture.zig"),
            .target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding }),
            .optimize = .ReleaseSmall,
        }),
    });
    wasm.entry = .disabled;
    wasm.rdynamic = true;
    wasm.stack_size = 16384;
    wasm.initial_memory = 131072;
    wasm.max_memory = 524288;
    const compile = b.addRunArtifact(wamrc);
    compile.addArgs(&.{ "compile", "--target=x86_64", "--profile=unikraft-x86_64" });
    compile.addFileArg(wasm.getEmittedBin());
    compile.addArg("-o");
    const fixture = compile.addOutputFileArg("native-fixture.cwasm");
    const compile_noop = b.addRunArtifact(wamrc);
    compile_noop.addArgs(&.{ "compile", "--target=x86_64", "--profile=unikraft-x86_64" });
    compile_noop.addFileArg(b.path("tests/coldstart/noop.wasm"));
    compile_noop.addArg("-o");
    const no_imports = compile_noop.addOutputFileArg("native-noop.cwasm");
    const generator_module = b.createModule(.{
        .root_source_file = b.path("tests/unikraft-aot/generate.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
    });
    generator_module.addImport("wabt", wabt);
    const generator = b.addExecutable(.{ .name = "generate-native-aot-fixture", .root_module = generator_module });
    const generate = b.addRunArtifact(generator);
    generate.addFileArg(b.path("tests/unikraft-aot/tables.wat"));
    const tables_wasm = generate.addOutputFileArg("native-tables.wasm");
    const compile_tables = b.addRunArtifact(wamrc);
    compile_tables.addArgs(&.{ "compile", "--target=x86_64", "--profile=unikraft-x86_64" });
    compile_tables.addFileArg(tables_wasm);
    compile_tables.addArg("-o");
    const tables = compile_tables.addOutputFileArg("native-tables.cwasm");
    const files = b.addWriteFiles();
    _ = files.addCopyFile(fixture, "native-fixture.cwasm");
    _ = files.addCopyFile(no_imports, "native-noop.cwasm");
    _ = files.addCopyFile(tables, "native-tables.cwasm");
    const fixture_module = files.add("fixture.zig", "pub const bytes = @embedFile(\"native-fixture.cwasm\");\npub const no_imports = @embedFile(\"native-noop.cwasm\");\npub const tables = @embedFile(\"native-tables.cwasm\");\n");
    const target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .musl });
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/aot_native_tests.zig"),
            .target = target,
            .optimize = .ReleaseSafe,
        }),
    });
    tests.root_module.addAnonymousImport("native_fixture", .{ .root_source_file = fixture_module });
    const test_step = b.step("test-native-aot", "Run real precompiled native embedding API regressions");
    test_step.dependOn(policy_step);
    const format_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/aot_native_format_tests.zig"),
            .target = b.graph.host,
            .optimize = .ReleaseSafe,
        }),
    });
    format_tests.root_module.addAnonymousImport("native_fixture", .{ .root_source_file = fixture_module });
    const format_step = b.step("test-native-aot-format", "Run native artifact parser regressions on the build host");
    format_step.dependOn(&b.addRunArtifact(format_tests).step);
    test_step.dependOn(format_step);
    const compile_step = b.step("test-native-aot-build", "Build native AOT regressions without executing x86 code");
    _ = tests.getEmittedBin();
    compile_step.dependOn(&tests.step);
    if (b.graph.host.result.cpu.arch == .x86_64 and b.graph.host.result.os.tag == .linux) {
        test_step.dependOn(&b.addRunArtifact(tests).step);
    } else {
        const run = b.addSystemCommand(&.{ "qemu-x86_64", "-cpu", "max" });
        run.addArtifactArg(tests);
        test_step.dependOn(&run.step);
    }
    const fixture_step = b.step("native-aot-fixture", "Build the matching host-wamrc native fixture");
    fixture_step.dependOn(&b.addInstallArtifact(wamrc, .{}).step);
    const install_fixture = b.addInstallFile(fixture, "fixtures/native-fixture.cwasm");
    const install_wasm = b.addInstallFile(wasm.getEmittedBin(), "fixtures/native-fixture.wasm");
    fixture_step.dependOn(&install_fixture.step);
    fixture_step.dependOn(&install_wasm.step);
    fixture_step.dependOn(&b.addInstallFile(no_imports, "fixtures/native-noop.cwasm").step);
    fixture_step.dependOn(&b.addInstallFile(tables_wasm, "fixtures/native-tables.wasm").step);
    fixture_step.dependOn(&b.addInstallFile(tables, "fixtures/native-tables.cwasm").step);
}

pub fn moduleOptions(target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) std.Build.Module.CreateOptions {
    return .{
        .target = target,
        .optimize = optimize,
        .single_threaded = true,
        .red_zone = false,
        .stack_check = false,
        .stack_protector = false,
        .unwind_tables = .none,
        .error_tracing = false,
        .link_libc = false,
    };
}

pub fn build(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    if (target.result.cpu.arch != .x86_64 or target.result.os.tag != .freestanding or target.result.abi != .none)
        std.debug.panic("unikraft-aot requires x86_64-freestanding-none", .{});
    inline for (.{ "interp", "fast_interp", "jit", "lazy_jit", "fast_jit", "wamr_compiler", "component_model", "lib_pthread", "lib_wasi_threads", "thread_mgr", "shared_memory", "link-libc" }) |name| {
        if (b.option(bool, name, "Unsupported in the native AOT profile") orelse false)
            std.debug.panic("unikraft-aot excludes -D{s}=true", .{name});
    }
    var options = moduleOptions(target, optimize);
    options.root_source_file = b.path("src/aot_native.zig");
    const module = b.addModule("wamr-aot", options);
    const library = b.addLibrary(.{ .name = "wamr-aot", .linkage = .static, .root_module = module });
    library.bundle_compiler_rt = true;
    b.installArtifact(library);
    b.installFile("include/wamr_aot.h", "include/wamr_aot.h");
    // Export every real embedding entry point in a freestanding ELF. Unlike a
    // library-only compile this catches unresolved hosted/runtime dependencies.
    const link_check = b.addExecutable(.{ .name = "wamr-aot-link-check", .root_module = module });
    link_check.entry = .{ .symbol_name = "wamr_aot_contract_version" };
    link_check.rdynamic = true;
    _ = link_check.getEmittedBin();
    const check_step = b.step("native-aot-check", "Link the complete compiler-free freestanding API");
    check_step.dependOn(&link_check.step);
    b.getInstallStep().dependOn(check_step);
}
