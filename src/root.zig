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

// Testing
/// Spec test runner infrastructure.
pub const spec_runner = @import("tests/spec_runner.zig");

/// WASI preview1 implementation.
/// Note: Uses std.fs.File; tests require IO-aware runner.
/// Excluded from refAllDecls to avoid test runner hang.
const _wasi = @import("wasi/wasi.zig");

/// Thread manager for WASI-threads.
pub const thread_manager = @import("wasi/thread_manager.zig");

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

