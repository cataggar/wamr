const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const optimize: std.builtin.OptimizeMode = .ReleaseFast;

    const exe_ext = if (builtin.os.tag == .windows) ".exe" else "";
    const wamrc_default = "../../../zig-out/bin/wamrc" ++ exe_ext;
    const wamr_default = "../../../zig-out/bin/wamr" ++ exe_ext;

    const wamrc_path = b.option([]const u8, "wamrc", "Path to wamrc AOT compiler") orelse wamrc_default;
    const wamr_path = b.option([]const u8, "wamr", "Path to wamr runtime") orelse wamr_default;

    const coremark_sources: []const []const u8 = &.{
        "core_list_join.c",
        "core_main.c",
        "core_matrix.c",
        "core_state.c",
        "core_util.c",
        "posix/core_portme.c",
    };

    const iterations = b.option([]const u8, "iterations", "CoreMark iterations (default: 400000)") orelse "400000";

    const coremark_flags: []const []const u8 = &.{
        b.fmt("-DITERATIONS={s}", .{iterations}),
        "-DSEED_METHOD=SEED_VOLATILE",
        "-DPERFORMANCE_RUN=1",
        "-DHAS_PRINTF=1",
        "-DFLAGS_STR=\"-O3\"",
    };

    // ── Native executable ─────────────────────────────────────────────
    const native_module = b.createModule(.{
        .target = b.graph.host,
        .optimize = optimize,
        .link_libc = true,
    });
    native_module.addCSourceFiles(.{
        .root = b.path("coremark"),
        .files = coremark_sources,
        .flags = coremark_flags,
    });
    native_module.addIncludePath(b.path("coremark"));
    native_module.addIncludePath(b.path("coremark/posix"));

    const native_exe = b.addExecutable(.{
        .name = "coremark-native",
        .root_module = native_module,
    });
    b.installArtifact(native_exe);

    // ── WASM executable (wasm32-wasi) ─────────────────────────────────
    const wasm_module = b.createModule(.{
        .target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .wasi }),
        .optimize = optimize,
        .link_libc = true,
    });
    wasm_module.addCSourceFiles(.{
        .root = b.path("coremark"),
        .files = coremark_sources,
        .flags = coremark_flags,
    });
    wasm_module.addIncludePath(b.path("coremark"));
    wasm_module.addIncludePath(b.path("coremark/posix"));

    const wasm_exe = b.addExecutable(.{
        .name = "coremark",
        .root_module = wasm_module,
    });
    b.installArtifact(wasm_exe);

    // ── AOT compilation (wamrc: .wasm → .aot) ────────────────────────
    const aot_cmd = b.addSystemCommand(&.{wamrc_path});
    aot_cmd.addArg("-o");
    const aot_output = aot_cmd.addOutputFileArg("coremark.aot");
    aot_cmd.addFileArg(wasm_exe.getEmittedBin());

    const install_aot = b.addInstallFile(aot_output, "bin/coremark.aot");
    const aot_step = b.step("aot", "Compile CoreMark .wasm to .aot via wamrc");
    aot_step.dependOn(&install_aot.step);

    // ── Run steps ─────────────────────────────────────────────────────
    const run_native = b.addRunArtifact(native_exe);
    const run_native_step = b.step("run-native", "Run CoreMark native");
    run_native_step.dependOn(&run_native.step);

    const run_aot = b.addSystemCommand(&.{wamr_path});
    run_aot.addFileArg(aot_output);
    const run_aot_step = b.step("run-aot", "Run CoreMark via WAMR AOT");
    run_aot_step.dependOn(&run_aot.step);

    const run_interp = b.addSystemCommand(&.{wamr_path});
    run_interp.addFileArg(wasm_exe.getEmittedBin());
    const run_interp_step = b.step("run-interp", "Run CoreMark via WAMR interpreter");
    run_interp_step.dependOn(&run_interp.step);

    // ── Bench step (all three) ────────────────────────────────────────
    const bench_step = b.step("bench", "Run all CoreMark benchmarks (native, AOT, interpreter)");
    bench_step.dependOn(&run_native.step);
    bench_step.dependOn(&run_aot.step);
    bench_step.dependOn(&run_interp.step);
}
