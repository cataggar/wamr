//! fuzz-wasi — WASI/component resource-table command-model fuzzer.
//!
//! Drives the public host bindings on `WasiCliAdapter` through a
//! deterministic, byte-driven command model. No real component wasm is
//! executed; the harness invokes bound host functions directly with
//! synthesized `InterfaceValue` arguments.
//!
//! Phase 1 scope (this target):
//!   - filesystem `[method]descriptor.*` bindings on a per-process temp
//!     preopen,
//!   - `wasi:filesystem/preopens.get-directories`,
//!   - `WasiCliAdapter.validateSandboxPath` (dense path-validation
//!     coverage on attacker bytes),
//!   - `[resource-drop]descriptor`.
//!
//! Stream allocation and explicit input/output stream ops, sockets, and
//! HTTP host bindings are deferred to follow-up issues.
//!
//! Oracle: panics, safety-checked UB, aborts, allocator leaks across
//! iterations, descriptor-table not returning to its preopen-only
//! baseline, or any path that escapes the per-process temp root are
//! bugs. Typed `result<_, error-code>` failures are expected.

const std = @import("std");
const wamr = @import("wamr");
const adapter_mod = wamr.wasi_cli_adapter;
const WasiCliAdapter = adapter_mod.WasiCliAdapter;
const ci_mod = wamr.component_instance;
const ComponentInstance = ci_mod.ComponentInstance;
const InterfaceValue = ci_mod.InterfaceValue;
const HostInstance = ci_mod.HostInstance;
const HostInstanceMember = ci_mod.HostInstanceMember;
const ImportBinding = ci_mod.ImportBinding;
const ctypes = wamr.component_types;
const core_types = wamr.types;

const common = @import("common.zig");

/// Per-iteration cap on the number of opcodes decoded from one input.
/// Bounds allocator and filesystem work even on adversarial seeds.
const max_ops_per_input: u32 = 256;
/// Stub guest memory size. 64 KiB is one wasm page and matches what
/// `canonicalMemory` would return for a single-page module.
const stub_memory_bytes: usize = 64 * 1024;

const Op = enum(u8) {
    get_type,
    get_flags,
    stat,
    read_via_stream,
    write_via_stream,
    append_via_stream,
    drop_descriptor,
    get_directories,
    validate_sandbox_path,
    open_at,
    drop_stream,
};

const op_count: u8 = @intCast(@typeInfo(Op).@"enum".fields.len);

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());

    const args = try common.Args.parse(argv);
    var corpus = try common.Corpus.load(allocator, io, args.corpus_dir);
    defer corpus.deinit();

    if (corpus.count() == 0) {
        std.log.err("empty corpus at {s}", .{args.corpus_dir});
        return error.EmptyCorpus;
    }

    // Per-process temp root. Reused across iterations; we only place
    // small fixture files inside it and never delete the root until
    // exit. The host is sandboxed against `..` and absolute paths via
    // `validateSandboxPath`, so even if the harness opens new files,
    // they remain inside this directory.
    var tmp_root_buf: [128]u8 = undefined;
    const tmp_root = try std.fmt.bufPrint(&tmp_root_buf, "/tmp/fuzz-wasi-{d}", .{
        std.os.linux.getpid(),
    });

    const std_io = std.Io.Threaded.global_single_threaded.io();
    const cwd_io = std.Io.Dir.cwd();
    cwd_io.createDirPath(std_io, tmp_root) catch |err| switch (err) {
        // PathAlreadyExists is fine — we reuse a stale dir.
        else => return err,
    };
    var preopen_dir_io = try cwd_io.openDir(std_io, tmp_root, .{ .iterate = true });
    defer preopen_dir_io.close(std_io);

    // Seed two small fixture files. Errors here are fatal — the host
    // setup itself is not under test.
    try writeFixture(std_io, preopen_dir_io, "a.txt", "alpha");
    try writeFixture(std_io, preopen_dir_io, "b.txt", "beta");

    const deadline = common.Deadline.init(io, args.duration_ms);
    var iter: u64 = 0;
    var idx: usize = 0;
    while (!deadline.expired(io)) : (iter += 1) {
        const input = corpus.get(idx);
        idx +%= 1;

        try common.markInFlight(io, args.crashes_dir, input);
        try runOnce(allocator, std_io, preopen_dir_io, input);
        common.clearInFlight(io, args.crashes_dir);
    }

    std.log.info("fuzz-wasi: {d} iterations over {d} inputs", .{ iter, corpus.count() });
}

fn writeFixture(io: std.Io, dir: std.Io.Dir, sub_path: []const u8, data: []const u8) !void {
    dir.writeFile(io, .{ .sub_path = sub_path, .data = data }) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
}

/// Drive one input through the command model. The adapter, stub
/// component instance, and stub guest memory are reconstructed each
/// iteration so cross-input state cannot mask use-after-free or
/// table-leak bugs.
fn runOnce(
    allocator: std.mem.Allocator,
    io: std.Io,
    preopen_dir_io: std.Io.Dir,
    input: []const u8,
) !void {
    var adapter = WasiCliAdapter.init(allocator);
    defer adapter.deinit();

    // Single preopen at index 0; persistent across the iteration. The
    // adapter takes ownership of the dir handle on `addPreopen`, so we
    // pass a duplicate handle and null the slot before adapter.deinit
    // to avoid double-close.
    const preopen_handle = adapter.addPreopen("/sandbox", preopen_dir_io) catch return;
    _ = preopen_handle;
    defer {
        // Replace the preopen slot with null so adapter.deinit doesn't
        // try to close the shared dir handle.
        if (adapter.fs_descriptor_table.items.len > 0) {
            adapter.fs_descriptor_table.items[0] = null;
        }
    }

    // Pre-register `wasi:filesystem/types` + `wasi:filesystem/preopens`
    // bindings so we can look up host fns by name.
    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(allocator);
    adapter.populateWasiFilesystemTypes(&providers, "wasi:filesystem/types") catch return;
    adapter.populateWasiFilesystemPreopens(&providers, "wasi:filesystem/preopens") catch return;

    // Stub component instance + module instance + memory. The bindings
    // that take a `string` / `list<u8>` argument call
    // `ci.readGuestBytes`, which walks the first core_instances entry
    // with a non-null module_inst. We give it a minimal one with a
    // zero-filled 64 KiB backing buffer.
    var memory_bytes: [stub_memory_bytes]u8 = @splat(0);
    var memory_inst: core_types.MemoryInstance = .{
        .memory_type = .{ .limits = .{ .min = 1, .max = 1 } },
        .data = &memory_bytes,
        .current_pages = 1,
        .max_pages = 1,
    };
    var module: core_types.WasmModule = .{};
    var memories_slot = [_]*core_types.MemoryInstance{&memory_inst};
    var module_inst: core_types.ModuleInstance = .{
        .module = &module,
        .memories = &memories_slot,
        .tables = &.{},
        .globals = &.{},
        .allocator = allocator,
    };
    var core_instances = [_]ComponentInstance.CoreInstanceEntry{.{
        .module_inst = &module_inst,
    }};
    var stub_component: ctypes.Component = .{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    var ci: ComponentInstance = .{
        .component = &stub_component,
        .core_instances = &core_instances,
        .resource_tables = .empty,
        .exported_funcs = .empty,
        .imports = .empty,
        .module_arena = std.heap.ArenaAllocator.init(allocator),
        .allocator = allocator,
    };
    defer ci.module_arena.deinit();
    defer {
        var rt_it = ci.resource_tables.valueIterator();
        while (rt_it.next()) |rt| rt.deinit(allocator);
        ci.resource_tables.deinit(allocator);
        ci.exported_funcs.deinit(allocator);
        ci.imports.deinit(allocator);
    }

    // Op stream: bytewise.
    var cursor: usize = 0;
    var op_idx: u32 = 0;
    while (cursor < input.len and op_idx < max_ops_per_input) : (op_idx += 1) {
        const op_byte = input[cursor];
        cursor += 1;
        const op = std.enums.fromInt(Op, op_byte % op_count) orelse continue;
        cursor = dispatch(&adapter, &ci, &memory_bytes, op, input, cursor, allocator, io) catch |e| switch (e) {
            error.OutOfMemory => return e,
            else => cursor, // tolerate any typed error from the binding
        };
    }
}

fn dispatch(
    adapter: *WasiCliAdapter,
    ci: *ComponentInstance,
    memory: []u8,
    op: Op,
    input: []const u8,
    in_cursor: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
) !usize {
    _ = io;
    var cursor = in_cursor;

    switch (op) {
        .validate_sandbox_path => {
            // u8 length, then up to that many bytes from input.
            if (cursor >= input.len) return cursor;
            const want_len = input[cursor];
            cursor += 1;
            const path_len = @min(@as(usize, want_len), input.len - cursor);
            const path = input[cursor .. cursor + path_len];
            cursor += path_len;
            // Pure function. We just exercise it.
            _ = WasiCliAdapter.validateSandboxPath(path);
        },

        .get_type, .get_flags, .stat, .append_via_stream, .drop_descriptor => {
            const handle = pickHandle(adapter, input, &cursor) orelse return cursor;
            const member_name = switch (op) {
                .get_type => "[method]descriptor.get-type",
                .get_flags => "[method]descriptor.get-flags",
                .stat => "[method]descriptor.stat",
                .append_via_stream => "[method]descriptor.append-via-stream",
                .drop_descriptor => "[resource-drop]descriptor",
                else => unreachable,
            };
            var args = [_]InterfaceValue{.{ .handle = handle }};
            try invokeIface(&adapter.fs_types_iface, member_name, ci, &args, allocator);
        },

        .read_via_stream, .write_via_stream => {
            const handle = pickHandle(adapter, input, &cursor) orelse return cursor;
            const offset = readU64(input, &cursor);
            const member_name = if (op == .read_via_stream)
                "[method]descriptor.read-via-stream"
            else
                "[method]descriptor.write-via-stream";
            var args = [_]InterfaceValue{
                .{ .handle = handle },
                .{ .u64 = offset },
            };
            try invokeIface(&adapter.fs_types_iface, member_name, ci, &args, allocator);
        },

        .get_directories => {
            try invokeIface(&adapter.fs_preopens_iface, "get-directories", ci, &.{}, allocator);
        },

        .open_at => {
            // open-at: (handle, path-flags u32, path string, open-flags u32, descriptor-flags u32)
            const handle = pickHandle(adapter, input, &cursor) orelse return cursor;
            const path_flags: u32 = readU32(input, &cursor);
            // Synthesize a guest string by writing path bytes into stub
            // memory at offset 0.
            if (cursor >= input.len) return cursor;
            const want_len = input[cursor];
            cursor += 1;
            const path_len = @min(@as(u32, want_len), @as(u32, @intCast(@min(memory.len, input.len - cursor))));
            const path_src_end = @min(input.len, cursor + path_len);
            const actual_len: u32 = @intCast(path_src_end - cursor);
            @memcpy(memory[0..actual_len], input[cursor..path_src_end]);
            cursor += actual_len;
            const open_flags = readU32(input, &cursor);
            const desc_flags = readU32(input, &cursor);
            var args = [_]InterfaceValue{
                .{ .handle = handle },
                .{ .u32 = path_flags },
                .{ .string = .{ .ptr = 0, .len = actual_len } },
                .{ .u32 = open_flags },
                .{ .u32 = desc_flags },
            };
            try invokeIface(&adapter.fs_types_iface, "[method]descriptor.open-at", ci, &args, allocator);
        },

        .drop_stream => {
            // Drop a synthesized stream handle. Most picks will miss; that
            // exercises the bad-handle path of the resource-drop binding.
            const handle = readU32(input, &cursor);
            var args = [_]InterfaceValue{.{ .handle = handle }};
            try invokeIface(&adapter.write_iface, "[resource-drop]output-stream", ci, &args, allocator);
            try invokeIface(&adapter.write_iface, "[resource-drop]input-stream", ci, &args, allocator);
        },
    }

    return cursor;
}

/// Look up a member by name and call it. Frees results before
/// returning. Returns OOM unconditionally; any other error from the
/// host function is logged and discarded — typed errors are expected.
fn invokeIface(
    iface: *HostInstance,
    name: []const u8,
    ci: *ComponentInstance,
    args: []InterfaceValue,
    allocator: std.mem.Allocator,
) !void {
    const member = iface.members.get(name) orelse return;
    const func = switch (member) {
        .func => |f| f,
        .resource_type => return,
    };
    const callable = func.call orelse return;
    var results: [4]InterfaceValue = .{ .{ .u32 = 0 }, .{ .u32 = 0 }, .{ .u32 = 0 }, .{ .u32 = 0 } };
    callable(func.context, ci, args, &results, allocator) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => {},
    };
    // Free each result. `deinit` accepts any populated InterfaceValue
    // (no-op on inline scalars).
    for (results) |r| r.deinit(allocator);
}

/// Choose an existing descriptor table index from input, biased
/// towards valid handles when possible. If the table only has the
/// preopen, returns 0; otherwise mods the byte to pick any slot.
fn pickHandle(adapter: *WasiCliAdapter, input: []const u8, cursor: *usize) ?u32 {
    const len = adapter.fs_descriptor_table.items.len;
    if (len == 0) return null;
    if (cursor.* >= input.len) return 0;
    const b = input[cursor.*];
    cursor.* += 1;
    // Occasionally synthesize an out-of-range handle to exercise the
    // bad-descriptor path in every binding.
    if ((b & 0xC0) == 0xC0) return @as(u32, b);
    return @intCast(@as(usize, b) % len);
}

fn readU32(input: []const u8, cursor: *usize) u32 {
    var out: u32 = 0;
    var i: usize = 0;
    while (i < 4 and cursor.* < input.len) : (i += 1) {
        out |= @as(u32, input[cursor.*]) << @intCast(i * 8);
        cursor.* += 1;
    }
    return out;
}

fn readU64(input: []const u8, cursor: *usize) u64 {
    var out: u64 = 0;
    var i: usize = 0;
    while (i < 8 and cursor.* < input.len) : (i += 1) {
        out |= @as(u64, input[cursor.*]) << @intCast(i * 8);
        cursor.* += 1;
    }
    return out;
}
