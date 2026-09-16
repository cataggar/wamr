const std = @import("std");

pub fn add(b: *std.Build, wamrc: *std.Build.Step.Compile, optimize: std.builtin.OptimizeMode) void {
    const target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .musl });
    const wasi = b.createModule(.{
        .root_source_file = b.path("src/wasi/minimal.zig"),
        .target = target,
        .optimize = optimize,
    });
    const adapter = b.createModule(.{
        .root_source_file = b.path("src/wasi/native_aot.zig"),
        .target = target,
        .optimize = optimize,
    });
    adapter.addImport("minimal-wasi", wasi);
    const root = b.createModule(.{
        .root_source_file = b.path("src/native_bench_linux.zig"),
        .target = target,
        .optimize = optimize,
        .single_threaded = true,
    });
    root.addImport("minimal-wasi", wasi);
    root.addImport("native-wasi", adapter);
    const executable = b.addExecutable(.{ .name = "wamr-native-bench", .root_module = root });
    const install = b.addInstallArtifact(executable, .{});
    const producer = b.step("native-bench-linux", "Build the compiler-free Linux embedding benchmark producer");
    producer.dependOn(&install.step);

    const fixtures = b.step("native-bench-fixtures", "Ahead-of-time compile matching native benchmark artifacts");
    const wasm = b.addExecutable(.{
        .name = "native-benchmark-fixture",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/unikraft-aot/benchmark_fixture.zig"),
            .target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding }),
            .optimize = .ReleaseSmall,
        }),
    });
    wasm.entry = .disabled;
    wasm.rdynamic = true;
    wasm.stack_size = 16384;
    wasm.initial_memory = 131072;
    wasm.max_memory = 524288;
    const wasi_wasm = b.addExecutable(.{
        .name = "native-benchmark-wasi-fixture",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/unikraft-aot/benchmark_wasi_fixture.zig"),
            .target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding }),
            .optimize = .ReleaseSmall,
        }),
    });
    wasi_wasm.entry = .disabled;
    wasi_wasm.rdynamic = true;
    wasi_wasm.stack_size = 16384;
    wasi_wasm.initial_memory = 131072;
    wasi_wasm.max_memory = 524288;
    const names = [_][]const u8{ "deterministic", "coremark", "coremark-nofp", "wasi-callbacks", "compute", "memory" };
    const sources = [_]std.Build.LazyPath{
        wasm.getEmittedBin(),
        b.path("tests/benchmarks/coremark/coremark_wasi.wasm"),
        b.path("tests/benchmarks/coremark/coremark_wasi_nofp.wasm"),
        wasi_wasm.getEmittedBin(),
        b.path("tests/benchmarks/loop-passes/unroll4.wasm"),
        b.path("tests/benchmarks/loop-passes/iv_store.wasm"),
    };
    var artifacts: [names.len]std.Build.LazyPath = undefined;
    for (names, sources, &artifacts) |name, source, *artifact| {
        const compile = b.addRunArtifact(wamrc);
        compile.addArgs(&.{ "compile", "--target=x86_64", "--profile=unikraft-x86_64" });
        compile.addFileArg(source);
        compile.addArg("-o");
        artifact.* = compile.addOutputFileArg(b.fmt("{s}.cwasm", .{name}));
        fixtures.dependOn(&b.addInstallFile(artifact.*, b.fmt("native-bench/{s}.cwasm", .{name})).step);
        fixtures.dependOn(&b.addInstallFile(source, b.fmt("native-bench/{s}.wasm", .{name})).step);
    }
    const files = b.addWriteFiles();
    for (names[0..4], artifacts[0..4]) |name, artifact|
        _ = files.addCopyFile(artifact, b.fmt("{s}.cwasm", .{name}));
    const embedded = files.add("fixtures.zig",
        \\pub const deterministic = @embedFile("deterministic.cwasm");
        \\pub const coremark = @embedFile("coremark.cwasm");
        \\pub const nofp = @embedFile("coremark-nofp.cwasm");
        \\pub const wasi = @embedFile("wasi-callbacks.cwasm");
        \\
    );
    const test_root = b.createModule(.{
        .root_source_file = b.path("src/native_bench_tests.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .single_threaded = true,
    });
    test_root.addImport("minimal-wasi", wasi);
    test_root.addImport("native-wasi", adapter);
    test_root.addAnonymousImport("fixtures", .{ .root_source_file = embedded });
    const filters = b.option([]const []const u8, "native-bench-filter", "Filter native embedding producer regressions (repeatable)") orelse &.{};
    const tests = b.addTest(.{ .root_module = test_root, .filters = filters });
    const check = b.step("test-native-bench", "Exercise real Linux native AOT/WASI callbacks and repeated CoreMark CRC");
    const unit = b.step("test-native-bench-unit", "Run native embedding producer API/callback regressions");
    check.dependOn(unit);
    const build_only = b.step("test-native-bench-build", "Build Linux embedding tests without native execution");
    _ = tests.getEmittedBin();
    build_only.dependOn(&tests.step);
    if (b.graph.host.result.os.tag == .linux and b.graph.host.result.cpu.arch == .x86_64) {
        unit.dependOn(&b.addRunArtifact(tests).step);
    } else if (b.graph.host.result.os.tag == .linux) {
        const run = b.addSystemCommand(&.{ "qemu-x86_64", "-cpu", "max" });
        run.addArtifactArg(tests);
        unit.dependOn(&run.step);
    } else {
        unit.dependOn(build_only);
    }
    const python = b.addSystemCommand(&.{ "python3", "-m", "unittest", "scripts.test_bench_coremark.NativeLinuxProducerTests" });
    python.setEnvironmentVariable("WAMR_NATIVE_PRODUCER", b.getInstallPath(.bin, "wamr-native-bench"));
    python.setEnvironmentVariable("WAMR_NATIVE_FIXTURE_DIR", b.getInstallPath(.prefix, "native-bench"));
    if (b.graph.host.result.cpu.arch != .x86_64)
        python.setEnvironmentVariable("WAMR_NATIVE_PRODUCER_RUNNER", "qemu-x86_64 -cpu max");
    python.step.dependOn(producer);
    python.step.dependOn(fixtures);
    const protocol = b.step("test-native-bench-protocol", "Exercise real Linux producer through existing native protocol tests");
    if (b.graph.host.result.os.tag == .linux) {
        protocol.dependOn(&python.step);
        check.dependOn(protocol);
    }
}
