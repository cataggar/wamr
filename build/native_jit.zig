const std = @import("std");

pub fn addTests(b: *std.Build) void {
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
    const fixture_step = b.step("native-jit-fixture", "Install the deterministic no-import JIT workload");
    fixture_step.dependOn(&b.addInstallFile(wasm.getEmittedBin(), "fixtures/native-jit.wasm").step);
    const files = b.addWriteFiles();
    _ = files.addCopyFile(wasm.getEmittedBin(), "fixture.wasm");
    const fixture = files.add("fixture.zig", "pub const bytes = @embedFile(\"fixture.wasm\");\n");
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
    const module = b.addModule("wamr-jit", .{
        .root_source_file = b.path("src/jit_native.zig"),
        .target = target,
        .optimize = optimize,
        .single_threaded = true,
        .red_zone = false,
        .stack_check = false,
        .stack_protector = false,
        .unwind_tables = .none,
        .error_tracing = false,
        .link_libc = false,
    });
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
}
