//! Reusable in-memory AOT compile+instantiate harness.
//!
//! Wraps the end-to-end AOT pipeline (interpreter loader → IR frontend →
//! IR passes → x86_64/aarch64 codegen → emit_aot → aot_loader →
//! aot_runtime.instantiate → mapCodeExecutable) behind a small typed API
//! usable by test runners.
//!
//! Factored out of `src/tests/differential.zig` so the spec-test runner
//! can drive the same pipeline (see issue #102).

const std = @import("std");
const builtin = @import("builtin");

const root = @import("wamr");
const types = root.types;
const loader_mod = root.loader;
const frontend = root.frontend;
const passes = root.passes;
const x86_64_compile = root.x86_64_compile;
const aarch64_compile = root.aarch64_compile;
const emit_aot = root.emit_aot;
const aot_loader = root.aot_loader;
const aot_runtime = root.aot_runtime;

/// True on targets where the AOT runtime can execute generated code.
pub const can_exec_aot = switch (builtin.cpu.arch) {
    .x86_64, .aarch64 => true,
    else => false,
};

pub const Error = error{
    AotUnsupportedArch,
    CompileFailed,
    LoadFailed,
    InstantiateFailed,
    MapExecutableFailed,
} || aot_runtime.ScalarCallError;

/// A fully instantiated AOT module ready to invoke.
///
/// Owns the interpreter-loaded `WasmModule` (for function-type lookup),
/// the emitted AOT binary, the loaded `AotModule`, and the `AotInstance`.
pub const Harness = struct {
    allocator: std.mem.Allocator,
    arena: *std.heap.ArenaAllocator,
    wasm_module: types.WasmModule,
    aot_bin: []u8,
    aot_module: aot_loader.AotModule,
    inst: *aot_runtime.AotInstance,

    /// Compile `wasm_bytes` through the full AOT pipeline and instantiate it.
    /// On success the caller owns the harness and must call `deinit` then
    /// free the pointer with the same allocator.
    pub fn init(allocator: std.mem.Allocator, wasm_bytes: []const u8) Error!*Harness {
        if (comptime !can_exec_aot) return error.AotUnsupportedArch;

        const h = allocator.create(Harness) catch return error.OutOfMemory;
        errdefer allocator.destroy(h);

        const arena = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
        errdefer allocator.destroy(arena);
        arena.* = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const a = arena.allocator();

        const wasm_module = loader_mod.load(wasm_bytes, a) catch return error.CompileFailed;

        const aot_bin = compileToAot(allocator, arena, &wasm_module) catch return error.CompileFailed;
        errdefer allocator.free(aot_bin);

        const aot_module = aot_loader.load(aot_bin, allocator) catch return error.LoadFailed;

        h.* = .{
            .allocator = allocator,
            .arena = arena,
            .wasm_module = wasm_module,
            .aot_bin = aot_bin,
            .aot_module = aot_module,
            .inst = undefined,
        };
        errdefer aot_loader.unload(&h.aot_module, allocator);

        // `&h.aot_module` is stable for the lifetime of the heap-allocated
        // Harness, so `inst.module` will remain valid until `deinit`.
        h.inst = aot_runtime.instantiate(&h.aot_module, allocator) catch
            return error.InstantiateFailed;
        errdefer aot_runtime.destroy(h.inst);

        aot_runtime.mapCodeExecutable(h.inst) catch return error.MapExecutableFailed;

        return h;
    }

    pub fn deinit(self: *Harness) void {
        const allocator = self.allocator;
        aot_runtime.destroy(self.inst);
        aot_loader.unload(&self.aot_module, allocator);
        allocator.free(self.aot_bin);
        self.arena.deinit();
        allocator.destroy(self.arena);
        allocator.destroy(self);
    }

    /// Look up an exported function by name, returning the module-level
    /// function index (same index space as `aot_runtime.findExportFunc`).
    pub fn findFuncExport(self: *const Harness, name: []const u8) ?u32 {
        return aot_runtime.findExportFunc(self.inst, name);
    }

    /// Look up the function type of a module function by index, using the
    /// interpreter-loaded `WasmModule` (the AOT module drops type info).
    pub fn getFuncType(self: *const Harness, func_idx: u32) ?types.FuncType {
        return self.wasm_module.getFuncType(func_idx);
    }

    /// Find an exported global by name and return its current value.
    pub fn findGlobalExport(self: *const Harness, name: []const u8) ?types.Value {
        for (self.wasm_module.exports) |exp| {
            if (exp.kind == .global and std.mem.eql(u8, exp.name, name)) {
                if (exp.index >= self.inst.globals.len) return null;
                return self.inst.globals[exp.index].value;
            }
        }
        return null;
    }

    /// Invoke an AOT function by index with runtime-typed scalar args.
    /// See `aot_runtime.callFuncScalar` for the supported signature envelope.
    pub fn callScalar(
        self: *Harness,
        func_idx: u32,
        args: []const types.Value,
    ) aot_runtime.ScalarCallError!aot_runtime.ScalarResult {
        const ft = self.getFuncType(func_idx) orelse return error.FunctionNotFound;
        const result_type: ?types.ValType = if (ft.results.len == 0)
            null
        else if (ft.results.len == 1)
            ft.results[0]
        else
            return error.UnsupportedSignature;
        return aot_runtime.callFuncScalar(self.inst, func_idx, ft.params, result_type, args);
    }
};

fn compileToAot(
    allocator: std.mem.Allocator,
    arena: *std.heap.ArenaAllocator,
    module: *const types.WasmModule,
) ![]u8 {
    const a = arena.allocator();

    var ir_module = try frontend.lowerModule(module, a);
    defer ir_module.deinit();

    _ = try passes.runPasses(&ir_module, passes.default_passes, a);

    const code: []const u8, const offsets: []const u32 = switch (builtin.cpu.arch) {
        .aarch64 => blk: {
            const r = try aarch64_compile.compileModule(&ir_module, a);
            break :blk .{ r.code, r.offsets };
        },
        else => blk: {
            const r = try x86_64_compile.compileModule(&ir_module, a);
            break :blk .{ r.code, r.offsets };
        },
    };

    var exports: std.ArrayList(emit_aot.ExportEntry) = .empty;
    for (module.exports) |exp| {
        try exports.append(a, .{
            .name = exp.name,
            .kind = @enumFromInt(@intFromEnum(exp.kind)),
            .index = exp.index,
        });
    }

    var mem_entries: std.ArrayList(emit_aot.MemoryEntry) = .empty;
    for (module.memories) |mem| {
        try mem_entries.append(a, .{
            .min_pages = @intCast(mem.limits.min),
            .max_pages = if (mem.limits.max) |m| @as(?u32, @intCast(m)) else null,
        });
    }

    var arch_name = std.mem.zeroes([16]u8);
    switch (builtin.cpu.arch) {
        .aarch64 => @memcpy(arch_name[0..7], "aarch64"),
        else => @memcpy(arch_name[0..6], "x86-64"),
    }

    return try emit_aot.emit(
        allocator,
        code,
        offsets,
        exports.items,
        .{ .arch = arch_name },
        null,
        null,
        if (mem_entries.items.len > 0) mem_entries.items else null,
        null,
        null,
    );
}
