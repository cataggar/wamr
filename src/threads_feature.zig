//! Compile-time contract for the legacy Preview 1 `wasi.thread-spawn`
//! feature. Interpreter support is production-ready when the validated
//! capability set is enabled; AOT support remains explicitly unavailable.

const std = @import("std");

pub const TargetSupport = enum {
    supported,
    wasm_host,
    unsupported_pointer_width,
    single_threaded_host,
};

pub const BackendSupport = enum {
    disabled,
    target_unsupported,
    production,
    architecture_abi_not_implemented,
};

pub const Capabilities = struct {
    shared_memory: bool,
    thread_manager: bool,
    wasm_atomics: bool,
};

pub const ImplementationStatus = struct {
    resource_locking: bool = false,
    interpreter_thread_spawning: bool = false,
    wasm_atomics: bool = false,
    aot_thread_spawning: bool = false,
};

pub const Inputs = struct {
    enabled: bool,
    pointer_bits: u16,
    wasm_host: bool,
    single_threaded: bool,
    interp: bool,
    aot: bool,
    jit: bool,
    fast_jit: bool,
    libc_wasi: bool,
    heap_aux_stack_allocation: bool,
    shared_memory: bool,
    thread_manager: bool,
    wasm_atomics: bool,
};

pub const ValidationError = enum {
    wasm_host,
    unsupported_pointer_width,
    single_threaded_host,
    wasi_required,
    interpreter_required,
    aot_architecture_abi_not_implemented,
    jit_backend_not_implemented,
    fast_jit_backend_not_implemented,
    heap_aux_stack_required,
    shared_memory_required,
    thread_manager_required,
    wasm_atomics_required,
};

pub const Report = struct {
    enabled: bool,
    target: TargetSupport,
    interpreter_backend: BackendSupport,
    aot_backend: BackendSupport,
    required: Capabilities,
    configured: Capabilities,
    implementation: ImplementationStatus = .{},
};

pub fn targetSupport(inputs: Inputs) TargetSupport {
    if (inputs.wasm_host) return .wasm_host;
    if (inputs.pointer_bits < 64) return .unsupported_pointer_width;
    if (inputs.single_threaded) return .single_threaded_host;
    return .supported;
}

pub fn validationError(inputs: Inputs) ?ValidationError {
    if (!inputs.enabled) return null;

    switch (targetSupport(inputs)) {
        .supported => {},
        .wasm_host => return .wasm_host,
        .unsupported_pointer_width => return .unsupported_pointer_width,
        .single_threaded_host => return .single_threaded_host,
    }
    if (!inputs.libc_wasi) return .wasi_required;
    if (!inputs.interp) return .interpreter_required;
    if (inputs.aot) return .aot_architecture_abi_not_implemented;
    if (inputs.jit) return .jit_backend_not_implemented;
    if (inputs.fast_jit) return .fast_jit_backend_not_implemented;
    if (!inputs.heap_aux_stack_allocation) return .heap_aux_stack_required;
    if (!inputs.shared_memory) return .shared_memory_required;
    if (!inputs.thread_manager) return .thread_manager_required;
    if (!inputs.wasm_atomics) return .wasm_atomics_required;
    return null;
}

pub fn report(inputs: Inputs) Report {
    const target = targetSupport(inputs);
    const required = Capabilities{
        .shared_memory = inputs.enabled,
        .thread_manager = inputs.enabled,
        .wasm_atomics = inputs.enabled,
    };
    const configured = Capabilities{
        .shared_memory = inputs.shared_memory,
        .thread_manager = inputs.thread_manager,
        .wasm_atomics = inputs.wasm_atomics,
    };
    if (!inputs.enabled) {
        return .{
            .enabled = false,
            .target = target,
            .interpreter_backend = .disabled,
            .aot_backend = .disabled,
            .required = required,
            .configured = configured,
        };
    }
    if (target != .supported) {
        return .{
            .enabled = true,
            .target = target,
            .interpreter_backend = .target_unsupported,
            .aot_backend = .target_unsupported,
            .required = required,
            .configured = configured,
        };
    }
    return .{
        .enabled = true,
        .target = .supported,
        .interpreter_backend = .production,
        .aot_backend = .architecture_abi_not_implemented,
        .required = required,
        .configured = configured,
        .implementation = .{
            .resource_locking = true,
            .interpreter_thread_spawning = true,
            .wasm_atomics = true,
            .aot_thread_spawning = false,
        },
    };
}

pub fn validationMessage(err: ValidationError) []const u8 {
    return switch (err) {
        .wasm_host => "lib_wasi_threads does not support wasm host execution",
        .unsupported_pointer_width => "lib_wasi_threads requires a 64-bit host target",
        .single_threaded_host => "lib_wasi_threads requires a multithreaded host target",
        .wasi_required => "lib_wasi_threads requires libc_wasi",
        .interpreter_required => "lib_wasi_threads requires the interpreter backend",
        .aot_architecture_abi_not_implemented => "lib_wasi_threads AOT architecture/ABI support is not implemented; disable aot",
        .jit_backend_not_implemented => "lib_wasi_threads is incompatible with the AOT-based JIT backend",
        .fast_jit_backend_not_implemented => "lib_wasi_threads is incompatible with the fast JIT backend",
        .heap_aux_stack_required => "lib_wasi_threads requires heap auxiliary stack allocation",
        .shared_memory_required => "lib_wasi_threads requires shared memory",
        .thread_manager_required => "lib_wasi_threads requires the thread manager",
        .wasm_atomics_required => "lib_wasi_threads requires WebAssembly atomics",
    };
}

pub fn isThreadSpawnImport(module_name: []const u8, field_name: []const u8) bool {
    if (!std.mem.eql(u8, field_name, "thread-spawn")) return false;
    return std.mem.eql(u8, module_name, "wasi") or
        std.mem.eql(u8, module_name, "wasi_snapshot_preview1") or
        std.mem.eql(u8, module_name, "wasi_unstable");
}

fn validEnabledInputs() Inputs {
    return .{
        .enabled = true,
        .pointer_bits = 64,
        .wasm_host = false,
        .single_threaded = false,
        .interp = true,
        .aot = false,
        .jit = false,
        .fast_jit = false,
        .libc_wasi = true,
        .heap_aux_stack_allocation = true,
        .shared_memory = true,
        .thread_manager = true,
        .wasm_atomics = true,
    };
}

test "disabled contract preserves defaults without requiring thread capabilities" {
    var inputs = validEnabledInputs();
    inputs.enabled = false;
    inputs.pointer_bits = 32;
    inputs.wasm_host = true;
    inputs.interp = false;
    inputs.aot = true;
    inputs.libc_wasi = false;
    inputs.shared_memory = false;
    inputs.thread_manager = false;
    inputs.wasm_atomics = false;

    try std.testing.expectEqual(@as(?ValidationError, null), validationError(inputs));
    const actual = report(inputs);
    try std.testing.expectEqual(TargetSupport.wasm_host, actual.target);
    try std.testing.expectEqual(BackendSupport.disabled, actual.interpreter_backend);
    try std.testing.expectEqual(BackendSupport.disabled, actual.aot_backend);
    try std.testing.expect(!actual.required.shared_memory);
}

test "enabled configuration reports production interpreter support" {
    const actual = report(validEnabledInputs());
    try std.testing.expectEqual(@as(?ValidationError, null), validationError(validEnabledInputs()));
    try std.testing.expectEqual(TargetSupport.supported, actual.target);
    try std.testing.expectEqual(BackendSupport.production, actual.interpreter_backend);
    try std.testing.expectEqual(BackendSupport.architecture_abi_not_implemented, actual.aot_backend);
    try std.testing.expect(actual.required.shared_memory);
    try std.testing.expect(actual.configured.shared_memory);
    try std.testing.expect(actual.implementation.resource_locking);
    try std.testing.expect(actual.implementation.interpreter_thread_spawning);
    try std.testing.expect(actual.implementation.wasm_atomics);
    try std.testing.expect(!actual.implementation.aot_thread_spawning);
}

test "enabled contract rejects unsupported targets, backends, and missing capabilities" {
    const Case = struct {
        mutate: enum {
            wasm_host,
            pointer_width,
            single_threaded,
            wasi,
            interpreter,
            aot,
            jit,
            fast_jit,
            aux_stack,
            shared_memory,
            thread_manager,
            atomics,
        },
        expected: ValidationError,
    };
    const cases = [_]Case{
        .{ .mutate = .wasm_host, .expected = .wasm_host },
        .{ .mutate = .pointer_width, .expected = .unsupported_pointer_width },
        .{ .mutate = .single_threaded, .expected = .single_threaded_host },
        .{ .mutate = .wasi, .expected = .wasi_required },
        .{ .mutate = .interpreter, .expected = .interpreter_required },
        .{ .mutate = .aot, .expected = .aot_architecture_abi_not_implemented },
        .{ .mutate = .jit, .expected = .jit_backend_not_implemented },
        .{ .mutate = .fast_jit, .expected = .fast_jit_backend_not_implemented },
        .{ .mutate = .aux_stack, .expected = .heap_aux_stack_required },
        .{ .mutate = .shared_memory, .expected = .shared_memory_required },
        .{ .mutate = .thread_manager, .expected = .thread_manager_required },
        .{ .mutate = .atomics, .expected = .wasm_atomics_required },
    };

    for (cases) |case| {
        var inputs = validEnabledInputs();
        switch (case.mutate) {
            .wasm_host => inputs.wasm_host = true,
            .pointer_width => inputs.pointer_bits = 32,
            .single_threaded => inputs.single_threaded = true,
            .wasi => inputs.libc_wasi = false,
            .interpreter => inputs.interp = false,
            .aot => inputs.aot = true,
            .jit => inputs.jit = true,
            .fast_jit => inputs.fast_jit = true,
            .aux_stack => inputs.heap_aux_stack_allocation = false,
            .shared_memory => inputs.shared_memory = false,
            .thread_manager => inputs.thread_manager = false,
            .atomics => inputs.wasm_atomics = false,
        }
        try std.testing.expectEqual(case.expected, validationError(inputs).?);
    }
}

test "thread-spawn import recognition is limited to Preview 1 WASI modules" {
    try std.testing.expect(isThreadSpawnImport("wasi", "thread-spawn"));
    try std.testing.expect(isThreadSpawnImport("wasi_snapshot_preview1", "thread-spawn"));
    try std.testing.expect(!isThreadSpawnImport("env", "thread-spawn"));
    try std.testing.expect(!isThreadSpawnImport("wasi", "proc_exit"));
}
