//! WAMR — Idiomatic Zig embedding API.
//!
//! Usage:
//!   const wamr = @import("wamr");
//!   var runtime = wamr.Runtime.init(allocator);
//!   defer runtime.deinit();
//!   var module = try runtime.loadModule(wasm_bytes);
//!   defer module.deinit();
//!   var instance = try module.instantiate();
//!   defer instance.deinit();
//!   const result = try instance.callI32("add", &.{ 3, 4 });

const std = @import("std");
const types = @import("../runtime/common/types.zig");
const loader_mod = @import("../runtime/interpreter/loader.zig");
const instance_mod = @import("../runtime/interpreter/instance.zig");
const interp = @import("../runtime/interpreter/interp.zig");
const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
const comp_types = @import("../component/types.zig");
const comp_loader = @import("../component/loader.zig");
const comp_instance = @import("../component/instance.zig");

/// Host function registration types.
pub const host = @import("host.zig");
pub const HostContext = host.HostContext;
pub const HostImports = host.HostImports;

/// The WAMR runtime — manages the lifecycle of modules and instances.
pub const Runtime = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Runtime {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Runtime) void {
        _ = self;
    }

    /// Load a WebAssembly module from binary data.
    /// Returns error.IsComponent if the binary is a component (use loadComponent instead).
    pub fn loadModule(self: *Runtime, wasm_bytes: []const u8) !Module {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        errdefer arena.deinit();
        const mod = try loader_mod.load(wasm_bytes, arena.allocator());
        return .{ .inner = mod, .arena = arena, .allocator = self.allocator };
    }

    /// Load a WebAssembly Component from binary data.
    pub fn loadComponent(self: *Runtime, wasm_bytes: []const u8) !Component {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        errdefer arena.deinit();
        const comp = try comp_loader.load(wasm_bytes, arena.allocator());
        return .{ .inner = comp, .arena = arena, .allocator = self.allocator };
    }
};

/// A loaded WebAssembly module (not yet instantiated).
pub const Module = struct {
    inner: types.WasmModule,
    arena: std.heap.ArenaAllocator,
    allocator: std.mem.Allocator,

    /// Free all memory allocated during module loading.
    pub fn deinit(self: *Module) void {
        self.arena.deinit();
    }

    /// Instantiate this module, creating runnable memory/tables/globals.
    pub fn instantiate(self: *Module) !Instance {
        const inst = try instance_mod.instantiate(&self.inner, self.allocator);
        return .{ .inner = inst, .allocator = self.allocator };
    }

    /// Instantiate with pre-resolved imports.
    pub fn instantiateWithImports(self: *Module, import_ctx: instance_mod.ImportContext) !Instance {
        const inst = try instance_mod.instantiateWithImports(&self.inner, self.allocator, import_ctx);
        return .{ .inner = inst, .allocator = self.allocator };
    }

    /// Instantiate with comptime-typed host functions.
    /// Custom host functions take priority; unmatched imports fall back to WASI.
    pub fn instantiateWithHosts(self: *Module, comptime HostImportsT: type) !Instance {
        const inst = try instance_mod.instantiateWithHosts(&self.inner, self.allocator, HostImportsT);
        return .{ .inner = inst, .allocator = self.allocator };
    }

    /// Find an exported function by name.
    pub fn findExport(self: *const Module, name: []const u8, kind: types.ExternalKind) ?types.ExportDesc {
        return self.inner.findExport(name, kind);
    }
};

/// A loaded WebAssembly Component (not yet instantiated).
pub const Component = struct {
    inner: comp_types.Component,
    arena: std.heap.ArenaAllocator,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Component) void {
        self.arena.deinit();
    }

    /// Instantiate this component.
    pub fn instantiate(self: *Component) !ComponentInstance {
        const inst = try comp_instance.instantiate(&self.inner, self.allocator);
        return .{ .inner = inst, .allocator = self.allocator };
    }
};

/// A running Component instance.
pub const ComponentInstance = struct {
    inner: *comp_instance.ComponentInstance,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ComponentInstance) void {
        self.inner.deinit();
    }

    /// Look up an exported function by name.
    pub fn getExport(self: *const ComponentInstance, name: []const u8) ?comp_instance.ComponentInstance.ExportedFunc {
        return self.inner.getExport(name);
    }
};

/// A running WebAssembly module instance.
pub const Instance = struct {
    inner: *types.ModuleInstance,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Instance) void {
        instance_mod.destroy(self.inner);
    }

    /// Call an exported function with typed arguments, returning typed results.
    /// Caller owns the returned slice and must free it with `self.allocator`.
    pub fn call(self: *Instance, name: []const u8, args: []const types.Value) ![]types.Value {
        const exp = self.inner.module.findExport(name, .function) orelse return error.FunctionNotFound;
        const func_idx = exp.index;

        var env = try ExecEnv.create(self.inner, 4096, self.allocator);
        defer env.destroy();

        for (args) |arg| {
            try env.push(arg);
        }

        try interp.executeFunction(env, func_idx);

        const module = self.inner.module;
        const func_type = module.getFuncType(func_idx) orelse return error.FunctionNotFound;
        const result_count = func_type.results.len;

        var results = try self.allocator.alloc(types.Value, result_count);
        errdefer self.allocator.free(results);

        var i: usize = result_count;
        while (i > 0) {
            i -= 1;
            results[i] = try env.pop();
        }

        return results;
    }

    /// Call a void-returning exported function with typed arguments.
    pub fn callVoid(self: *Instance, name: []const u8, args: []const types.Value) !void {
        const results = try self.call(name, args);
        self.allocator.free(results);
    }

    /// Call an exported function by name with i32 arguments, returning i32.
    pub fn callI32(self: *Instance, name: []const u8, args: []const i32) !i32 {
        const exp = self.inner.module.findExport(name, .function) orelse return error.FunctionNotFound;
        const func_idx = exp.index;

        var env = try ExecEnv.create(self.inner, 4096, self.allocator);
        defer env.destroy();

        for (args) |arg| {
            try env.pushI32(arg);
        }

        try interp.executeFunction(env, func_idx);

        return env.popI32();
    }

    /// Get a memory instance by index.
    pub fn getMemory(self: *Instance, idx: u32) ?*types.MemoryInstance {
        return self.inner.getMemory(idx);
    }
};

// ── Tests ──────────────────────────────────────────────────────────────────

const testing = std.testing;

test "wamr: load and instantiate empty module" {
    var runtime = Runtime.init(testing.allocator);
    defer runtime.deinit();
    const data = [_]u8{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 };
    var module = try runtime.loadModule(&data);
    defer module.deinit();
    var instance = try module.instantiate();
    defer instance.deinit();
}

test "wamr: call exported i32.add function" {
    // (func (export "add") (param i32 i32) (result i32) local.get 0 local.get 1 i32.add)
    const wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
        // type section: 1 type — (param i32 i32) (result i32)
        0x01, 0x07, 0x01, 0x60, 0x02, 0x7F, 0x7F, 0x01, 0x7F,
        // function section: 1 function, type index 0
        0x03, 0x02, 0x01, 0x00,
        // export section: "add" -> func 0
        0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
        // code section: 1 body
        0x0A, 0x09, 0x01, // section id, size, count
        0x07, 0x00, // body size, local decl count
        0x20, 0x00, // local.get 0
        0x20, 0x01, // local.get 1
        0x6A, // i32.add
        0x0B, // end
    };
    var runtime = Runtime.init(testing.allocator);
    defer runtime.deinit();
    var module = try runtime.loadModule(&wasm);
    defer module.deinit();
    var instance = try module.instantiate();
    defer instance.deinit();
    const result = try instance.callI32("add", &.{ 3, 4 });
    try testing.expectEqual(@as(i32, 7), result);
}

test "wamr: callI32 returns FunctionNotFound for missing export" {
    var runtime = Runtime.init(testing.allocator);
    defer runtime.deinit();
    const data = [_]u8{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 };
    var module = try runtime.loadModule(&data);
    defer module.deinit();
    var instance = try module.instantiate();
    defer instance.deinit();
    try testing.expectError(error.FunctionNotFound, instance.callI32("nope", &.{}));
}

test "wamr: findExport on loaded module" {
    const wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
        // type section
        0x01, 0x07, 0x01, 0x60, 0x02, 0x7F, 0x7F, 0x01, 0x7F,
        // function section
        0x03, 0x02, 0x01, 0x00,
        // export section: "add" -> func 0
        0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
        // code section
        0x0A, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6A, 0x0B,
    };
    var runtime = Runtime.init(testing.allocator);
    defer runtime.deinit();
    var module = try runtime.loadModule(&wasm);
    defer module.deinit();

    const exp = module.findExport("add", .function);
    try testing.expect(exp != null);
    try testing.expectEqualStrings("add", exp.?.name);
    try testing.expectEqual(types.ExternalKind.function, exp.?.kind);

    try testing.expect(module.findExport("missing", .function) == null);
}

test "wamr: load empty component" {
    var runtime = Runtime.init(testing.allocator);
    defer runtime.deinit();
    // Component preamble: magic + version=0x0d + layer=0x01
    const data = [_]u8{ 0x00, 0x61, 0x73, 0x6D, 0x0d, 0x00, 0x01, 0x00 };
    var component = try runtime.loadComponent(&data);
    defer component.deinit();

    var instance = try component.instantiate();
    defer instance.deinit();

    try testing.expect(instance.getExport("missing") == null);
}

test "wamr: loadModule rejects component binary" {
    var runtime = Runtime.init(testing.allocator);
    defer runtime.deinit();
    const data = [_]u8{ 0x00, 0x61, 0x73, 0x6D, 0x0d, 0x00, 0x01, 0x00 };
    try testing.expectError(error.IsComponent, runtime.loadModule(&data));
}

test "wamr: host function add via HostImports" {
    // Wasm module that imports (env, add) and exports "call_add":
    //   (import "env" "add" (func $add (param i32 i32) (result i32)))
    //   (func (export "call_add") (result i32)
    //     i32.const 3  i32.const 4  call $add)
    const wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: 2 types
        0x01, 0x0b, 0x02,
        0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f, // type 0: (i32,i32)->i32
        0x60, 0x00, 0x01, 0x7f, // type 1: ()->i32
        // import section: 1 import
        0x02, 0x0b, 0x01,
        0x03, 'e', 'n', 'v', // module "env"
        0x03, 'a', 'd', 'd', // field "add"
        0x00, 0x00, // func, type 0
        // function section: 1 local function, type 1
        0x03, 0x02, 0x01, 0x01,
        // export section: "call_add" -> func 1
        0x07, 0x0c, 0x01,
        0x08, 'c', 'a', 'l', 'l', '_', 'a', 'd', 'd',
        0x00, 0x01,
        // code section
        0x0a, 0x0a, 0x01,
        0x08, 0x00, // body size, 0 locals
        0x41, 0x03, // i32.const 3
        0x41, 0x04, // i32.const 4
        0x10, 0x00, // call func 0 (imported add)
        0x0b, // end
    };

    const MyHosts = host.HostImports(.{
        .{ "env", "add", struct {
            fn f(_: host.HostContext, a: i32, b: i32) i32 {
                return a + b;
            }
        }.f },
    });

    var runtime = Runtime.init(testing.allocator);
    defer runtime.deinit();
    var module = try runtime.loadModule(&wasm);
    defer module.deinit();
    var instance = try module.instantiateWithHosts(MyHosts);
    defer instance.deinit();

    const result = try instance.callI32("call_add", &.{});
    try testing.expectEqual(@as(i32, 7), result);
}

test "wamr: host function reads memory" {
    // Wasm module with memory that imports (env, sum_bytes):
    //   (memory 1)
    //   (import "env" "sum_bytes" (func $sum (param i32 i32) (result i32)))
    //   (data (i32.const 0) "\x0a\x14\x1e")  ;; bytes 10, 20, 30 at offset 0
    //   (func (export "test") (result i32)
    //     i32.const 0  i32.const 3  call $sum)
    const wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: 2 types
        0x01, 0x0b, 0x02,
        0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f, // (i32,i32)->i32
        0x60, 0x00, 0x01, 0x7f, // ()->i32
        // import section
        0x02, 0x11, 0x01,
        0x03, 'e', 'n', 'v',
        0x09, 's', 'u', 'm', '_', 'b', 'y', 't', 'e', 's',
        0x00, 0x00,
        // function section
        0x03, 0x02, 0x01, 0x01,
        // memory section
        0x05, 0x03, 0x01, 0x00, 0x01,
        // export section: "test" -> func 1
        0x07, 0x08, 0x01,
        0x04, 't', 'e', 's', 't',
        0x00, 0x01,
        // code section
        0x0a, 0x0a, 0x01,
        0x08, 0x00,
        0x41, 0x00, // i32.const 0
        0x41, 0x03, // i32.const 3
        0x10, 0x00, // call $sum
        0x0b,
        // data section: 3 bytes at offset 0
        0x0b, 0x09, 0x01,
        0x00, 0x41, 0x00, 0x0b, // active, offset = i32.const 0
        0x03, 0x0a, 0x14, 0x1e, // 3 bytes: 10, 20, 30
    };

    const MyHosts = host.HostImports(.{
        .{ "env", "sum_bytes", struct {
            fn f(ctx: host.HostContext, ptr: i32, len: i32) i32 {
                const mem = ctx.memory() orelse return -1;
                const start: u32 = @bitCast(ptr);
                const end: u32 = start + @as(u32, @bitCast(len));
                if (end > mem.len) return -1;
                var sum: i32 = 0;
                for (mem[start..end]) |b| sum += b;
                return sum;
            }
        }.f },
    });

    var runtime = Runtime.init(testing.allocator);
    defer runtime.deinit();
    var module = try runtime.loadModule(&wasm);
    defer module.deinit();
    var instance = try module.instantiateWithHosts(MyHosts);
    defer instance.deinit();

    const result = try instance.callI32("test", &.{});
    try testing.expectEqual(@as(i32, 60), result); // 10 + 20 + 30
}
