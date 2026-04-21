//! WebAssembly Micro Runtime (WAMR) - Zig Implementation
//!
//! A lightweight standalone WebAssembly runtime with support for
//! interpreter, AOT, and JIT execution modes.

const std = @import("std");

/// Compile-time configuration (feature flags and constants).
pub const config = @import("config.zig");

/// C-compatible API for embedding (matches wasm_export.h).
pub const c_api = @import("api/c_api.zig");

/// Idiomatic Zig embedding API.
pub const wamr = @import("api/wamr.zig");

/// Core WebAssembly types.
pub const types = @import("runtime/common/types.zig");

/// Execution environment (operand stack, call frames).
pub const exec_env = @import("runtime/common/exec_env.zig");

/// Process-global canonical FuncType → u32 sig_id registry (for AOT call_indirect).
pub const sig_registry = @import("runtime/common/sig_registry.zig");

/// Wasm opcode definitions.
pub const opcode = @import("runtime/interpreter/opcode.zig");

/// Wasm binary loader.
pub const loader = @import("runtime/interpreter/loader.zig");

/// Module instantiation.
pub const instance = @import("runtime/interpreter/instance.zig");

/// Bytecode interpreter.
pub const interp = @import("runtime/interpreter/interp.zig");

/// AOT binary loader.
pub const aot_loader = @import("runtime/aot/loader.zig");

/// AOT runtime.
pub const aot_runtime = @import("runtime/aot/runtime.zig");

/// AOT ↔ WASI host function bridge.
pub const aot_host_bridge = @import("runtime/aot/host_bridge.zig");

/// WASI core logic (pure functions, shared by interpreter and AOT).
pub const wasi_core = @import("wasi/wasi_core.zig");

// Compiler
/// Compiler IR (SSA-form intermediate representation).
pub const ir = @import("compiler/ir/ir.zig");

/// Wasm → IR frontend (lowering).
pub const frontend = @import("compiler/frontend.zig");

/// AOT binary file emitter.
pub const emit_aot = @import("compiler/emit_aot.zig");

/// x86-64 machine code emitter.
pub const x86_64_emit = @import("compiler/codegen/x86_64/emit.zig");

/// x86-64 IR-to-native compiler.
pub const x86_64_compile = @import("compiler/codegen/x86_64/compile.zig");

/// AArch64 machine code emitter.
pub const aarch64_emit = @import("compiler/codegen/aarch64/emit.zig");

/// AArch64 IR-to-native compiler.
pub const aarch64_compile = @import("compiler/codegen/aarch64/compile.zig");

/// AOT compiler optimization passes.
pub const passes = @import("compiler/ir/passes.zig");

// Testing
/// Spec test runner infrastructure.
pub const spec_runner = @import("tests/spec_runner.zig");

// `differential.zig` is deliberately NOT exported here: it belongs to its
// own test module in build.zig (which also brings `aot_harness.zig` with
// it). Re-exporting would duplicate `aot_harness.zig` into both the `wamr`
// module AND the standalone `aot_harness` module used by the fuzz targets,
// which Zig rejects.

/// WASI preview1 implementation.
/// Note: Uses std.fs.File; tests require IO-aware runner.
/// Excluded from refAllDecls to avoid test runner hang.
const _wasi = @import("wasi/wasi.zig");

/// Thread manager for WASI-threads.
pub const thread_manager = @import("wasi/thread_manager.zig");

/// WASI host function implementations (thread-spawn, etc.).
pub const wasi_host = @import("wasi/host_functions.zig");

// Component Model
/// Component Model types (AST).
pub const component_types = @import("component/types.zig");

/// Component Model binary format loader.
pub const component_loader = @import("component/loader.zig");

/// Component Model canonical ABI (lifting/lowering).
pub const canonical_abi = @import("component/canonical_abi.zig");

/// Component Model instance and resource store.
pub const component_instance = @import("component/instance.zig");

/// Component Model async ABI (tasks, futures, streams).
pub const component_async = @import("component/async.zig");

/// Component composition and linking.
pub const component_compose = @import("component/compose.zig");

/// WASI Preview 2 core interfaces (clocks, random, CLI, filesystem).
pub const wasi_p2_core = @import("wasi/preview2/core.zig");

/// WASI Preview 2 I/O streams and poll.
pub const wasi_p2_streams = @import("wasi/preview2/streams.zig");

/// WASI Preview 2 sockets (TCP, UDP, name lookup).
pub const wasi_p2_sockets = @import("wasi/preview2/sockets.zig");

/// WASI Preview 2 HTTP types and handler.
pub const wasi_p2_http = @import("wasi/preview2/http.zig");

/// WASIp1 polyfill layer (maps p1 calls to p2 interfaces).
pub const wasi_p1_polyfill = @import("wasi/preview2/polyfill.zig");

/// Component Model async canonical ABI extensions.
pub const component_async_canon = @import("component/async_canon.zig");

// Phase 1: Foundation layer
/// Platform abstraction (mmap, threads, time, cache flush).
pub const platform = @import("platform/platform.zig");

/// Memory allocators (EMS pool allocator, default allocator).
pub const mem_alloc = @import("shared/mem_alloc/allocator.zig");

/// Shared utilities (logging, LEB128, hashmap, file I/O, crypto).
pub const utils = @import("shared/utils/utils.zig");

/// Cryptographic hashing (SHA-256, replaces BoringSSL).
pub const crypto = @import("shared/utils/crypto.zig");

/// WAMR version information.
pub const version = .{
    .major = 0,
    .minor = 1,
    .patch = 0,
    .string = "0.1.0-zig",
};

test {
    std.testing.refAllDecls(@This());
}

