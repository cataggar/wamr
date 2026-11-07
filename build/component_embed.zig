//! Build-system helper for embedding a `.wasm` Component (and optional
//! precompiled `.cwasm` cores + `.cwasm.json` manifest) into a Zig host
//! application via `@embedFile`.
//!
//! Modeled after `build/native_aot.zig`. Host applications can either:
//!   1. Call `createEmbeddedComponentModule` to copy a `.wasm` component
//!      into a `WriteFile` step (and optionally invoke `wamrc compile-component`
//!      at build time to emit `<name>.cwasm.json` alongside it), or
//!   2. Call `createEmbeddedComponentModuleFromPrecompiled` with explicit
//!      `.cwasm` `LazyPath` outputs to embed the component, optional
//!      manifest sidecar, and every precompiled core into a single
//!      importable Zig module for zero-filesystem runtime instantiation
//!      via `wamr.component_aot.resolveEmbeddedCores` /
//!      `wamr.component_aot.loadEmbeddedManifest`.

const std = @import("std");

pub const EmbedComponentOptions = struct {
    name: []const u8,
    component_file: std.Build.LazyPath,
    aot: bool = true,
    target_arch: ?[]const u8 = null,
    opt_level: ?[]const u8 = null,
    expected_core_count: usize = 0,
    cwasm_files: []const std.Build.LazyPath = &.{},
    manifest_file: ?std.Build.LazyPath = null,
};

pub const EmbeddedComponentModule = struct {
    module: *std.Build.Module,
    write_files: *std.Build.Step.WriteFile,
    manifest_file: ?std.Build.LazyPath = null,
};

fn appendEscapedZigString(
    out: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    s: []const u8,
) !void {
    const hex_chars = "0123456789abcdef";
    for (s) |c| {
        switch (c) {
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '"' => try out.appendSlice(allocator, "\\\""),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            0x20...0x21, 0x23...0x5b, 0x5d...0x7e => try out.append(allocator, c),
            else => {
                try out.appendSlice(allocator, "\\x");
                try out.append(allocator, hex_chars[c >> 4]);
                try out.append(allocator, hex_chars[c & 0x0f]);
            },
        }
    }
}

/// Generate the Zig source file for the embedded component module.
pub fn generateEmbeddedComponentSource(
    allocator: std.mem.Allocator,
    name: []const u8,
    aot_enabled: bool,
    has_manifest: bool,
    core_count: usize,
) ![]u8 {
    var src: std.ArrayList(u8) = .empty;
    errdefer src.deinit(allocator);

    try src.appendSlice(allocator, "pub const name: []const u8 = \"");
    try appendEscapedZigString(&src, allocator, name);
    try src.appendSlice(allocator, "\";\n");

    try src.appendSlice(allocator, if (aot_enabled)
        "pub const aot_enabled: bool = true;\n"
    else
        "pub const aot_enabled: bool = false;\n");

    try src.appendSlice(allocator, "pub const component_bytes: []const u8 = @embedFile(\"component.wasm\");\n");

    if (has_manifest) {
        try src.appendSlice(allocator, "pub const manifest_json: ?[]const u8 = @embedFile(\"component.cwasm.json\");\n");
    } else {
        try src.appendSlice(allocator, "pub const manifest_json: ?[]const u8 = null;\n");
    }

    try src.appendSlice(allocator,
        \\pub const CoreArtifact = struct {
        \\    idx: u32 = 0,
        \\    cwasm_bytes: []const u8,
        \\    sha256: ?[]const u8 = null,
        \\    core_sha256: ?[]const u8 = null,
        \\};
        \\
    );

    if (core_count == 0) {
        try src.appendSlice(allocator, "pub const cwasm_blobs: []const []const u8 = &.{};\n");
        try src.appendSlice(allocator, "pub const precompiled_cores: []const CoreArtifact = &.{};\n");
    } else {
        try src.appendSlice(allocator, "pub const cwasm_blobs: []const []const u8 = &.{\n");
        for (0..core_count) |i| {
            const line = try std.fmt.allocPrint(allocator, "    @embedFile(\"core_{d}.cwasm\"),\n", .{i});
            defer allocator.free(line);
            try src.appendSlice(allocator, line);
        }
        try src.appendSlice(allocator, "};\n");

        try src.appendSlice(allocator, "pub const precompiled_cores: []const CoreArtifact = &.{\n");
        for (0..core_count) |i| {
            const line = try std.fmt.allocPrint(
                allocator,
                "    .{{ .idx = {d}, .cwasm_bytes = @embedFile(\"core_{d}.cwasm\") }},\n",
                .{ i, i },
            );
            defer allocator.free(line);
            try src.appendSlice(allocator, line);
        }
        try src.appendSlice(allocator, "};\n");
    }

    return src.toOwnedSlice(allocator);
}

/// Create an anonymous build module that embeds `opts.component_file` as
/// `"component.wasm"` and (when `opts.aot` is enabled) either runs
/// `wamrc compile-component` or embeds the supplied precompiled artifacts.
pub fn createEmbeddedComponentModule(
    b: *std.Build,
    wamrc: ?*std.Build.Step.Compile,
    opts: EmbedComponentOptions,
) EmbeddedComponentModule {
    const write_files = b.addWriteFiles();
    _ = write_files.addCopyFile(opts.component_file, "component.wasm");

    var resolved_manifest: ?std.Build.LazyPath = opts.manifest_file;
    var effective_core_count: usize = opts.cwasm_files.len;
    if (opts.aot and wamrc != null and resolved_manifest == null) {
        const compile = b.addRunArtifact(wamrc.?);
        compile.addArg("compile-component");
        if (opts.target_arch) |arch| {
            compile.addArg(b.fmt("--target={s}", .{arch}));
        }
        if (opts.opt_level) |opt| {
            if (std.mem.eql(u8, opt, "0") or std.mem.eql(u8, opt, "-O0") or std.mem.eql(u8, opt, "O0")) {
                compile.addArg("-O0");
            } else if (std.mem.eql(u8, opt, "1") or std.mem.eql(u8, opt, "-O1") or std.mem.eql(u8, opt, "O1")) {
                compile.addArg("-O1");
            } else if (std.mem.eql(u8, opt, "2") or std.mem.eql(u8, opt, "-O2") or std.mem.eql(u8, opt, "O2")) {
                compile.addArg("-O2");
            } else if (std.mem.eql(u8, opt, "3") or std.mem.eql(u8, opt, "-O3") or std.mem.eql(u8, opt, "O3")) {
                compile.addArg("-O3");
            }
        }
        compile.addFileArg(opts.component_file);
        compile.addArg("-o");
        const manifest_lp = compile.addOutputFileArg("component.cwasm.json");
        resolved_manifest = manifest_lp;
        if (opts.cwasm_files.len == 0 and opts.expected_core_count > 0) {
            const manifest_dir = manifest_lp.dirname();
            for (0..opts.expected_core_count) |i| {
                const cwasm_lp = manifest_dir.path(b, b.fmt("component.{d}.cwasm", .{i}));
                _ = write_files.addCopyFile(cwasm_lp, b.fmt("core_{d}.cwasm", .{i}));
            }
            effective_core_count = opts.expected_core_count;
        }
    }

    if (resolved_manifest) |mf| {
        _ = write_files.addCopyFile(mf, "component.cwasm.json");
    }

    for (opts.cwasm_files, 0..) |cwasm_file, i| {
        _ = write_files.addCopyFile(cwasm_file, b.fmt("core_{d}.cwasm", .{i}));
    }

    const aot_enabled = opts.aot and (wamrc != null or resolved_manifest != null or effective_core_count > 0);
    const generated_src = generateEmbeddedComponentSource(
        b.allocator,
        opts.name,
        aot_enabled,
        resolved_manifest != null,
        effective_core_count,
    ) catch @panic("OOM generating embedded_component.zig");

    const root_source = write_files.add("embedded_component.zig", generated_src);
    const module = b.createModule(.{
        .root_source_file = root_source,
    });

    return .{
        .module = module,
        .write_files = write_files,
        .manifest_file = resolved_manifest,
    };
}

/// Create an anonymous build module from a `.wasm` component, optional
/// `<stem>.cwasm.json` manifest sidecar, and an explicit list of precompiled
/// `.cwasm` core artifacts (`core_0.cwasm`, `core_1.cwasm`, ...).
pub fn createEmbeddedComponentModuleFromPrecompiled(
    b: *std.Build,
    name: []const u8,
    component_file: std.Build.LazyPath,
    manifest_file: ?std.Build.LazyPath,
    cwasm_files: []const std.Build.LazyPath,
) *std.Build.Module {
    const res = createEmbeddedComponentModule(b, null, .{
        .name = name,
        .component_file = component_file,
        .aot = cwasm_files.len > 0 or manifest_file != null,
        .cwasm_files = cwasm_files,
        .manifest_file = manifest_file,
    });
    return res.module;
}

test "generateEmbeddedComponentSource: emits expected @embedFile bindings" {
    const alloc = std.testing.allocator;

    const interp_src = try generateEmbeddedComponentSource(alloc, "echo-comp", false, false, 0);
    defer alloc.free(interp_src);
    try std.testing.expect(std.mem.indexOf(u8, interp_src, "pub const name: []const u8 = \"echo-comp\";") != null);
    try std.testing.expect(std.mem.indexOf(u8, interp_src, "pub const aot_enabled: bool = false;") != null);
    try std.testing.expect(std.mem.indexOf(u8, interp_src, "@embedFile(\"component.wasm\")") != null);
    try std.testing.expect(std.mem.indexOf(u8, interp_src, "pub const precompiled_cores: []const CoreArtifact = &.{};") != null);

    const aot_src = try generateEmbeddedComponentSource(alloc, "aot-comp", true, true, 2);
    defer alloc.free(aot_src);
    try std.testing.expect(std.mem.indexOf(u8, aot_src, "pub const aot_enabled: bool = true;") != null);
    try std.testing.expect(std.mem.indexOf(u8, aot_src, "@embedFile(\"component.cwasm.json\")") != null);
    try std.testing.expect(std.mem.indexOf(u8, aot_src, "@embedFile(\"core_0.cwasm\")") != null);
    try std.testing.expect(std.mem.indexOf(u8, aot_src, "@embedFile(\"core_1.cwasm\")") != null);
}
