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

/// Host function registration (comptime-typed native function imports).
pub const host = @import("api/host.zig");

/// Core WebAssembly types.
pub const types = @import("runtime/common/types.zig");

/// Execution environment (operand stack, call frames).
pub const exec_env = @import("runtime/common/exec_env.zig");

/// Process-global canonical FuncType → u32 sig_id registry (for AOT call_indirect).
pub const sig_registry = @import("runtime/common/sig_registry.zig");

/// Wasm `name` custom section parser (function names; for diagnostics).
pub const name_section = @import("runtime/common/name_section.zig");

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

/// AOT host-import trampoline pool (#662, #756). Exposed so `wamr
/// run` can tune the per-pool slot cap via the
/// `WAMR_MAX_TRAMPOLINE_SLOTS` env var.
pub const host_trampolines = @import("runtime/aot/host_trampolines.zig");

/// WASI core logic (pure functions, shared by interpreter and AOT).
pub const wasi_core = @import("wasi/wasi_core.zig");

// Compiler
/// Compiler IR (SSA-form intermediate representation).
pub const ir = @import("compiler/ir/ir.zig");

/// IR pretty-printer (used by `wamrc --dump-ir-after=…`).
pub const ir_print = @import("compiler/ir/print.zig");

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

/// AArch64 peephole optimizer.
pub const aarch64_peephole = @import("compiler/codegen/aarch64/peephole.zig");

/// AOT compiler optimization passes.
pub const passes = @import("compiler/ir/passes.zig");

/// AOT codegen bisection knobs — env-var parsing + process-global
/// `PassBisectSpec` consumed by `runPassesWithOptions` (#761 / #743).
pub const aot_bisect = @import("compiler/aot_bisect.zig");

/// AOT codegen cache (#761 Phase 2) — sidecar file format + canonical
/// IR hasher + module-epoch hasher. Lets a recompile reuse cached
/// native code for any function whose IR is unchanged from the
/// previous build.
pub const codegen_cache = @import("compiler/codegen_cache.zig");

/// IR invariant checker run between passes (#624).
pub const ir_verifier = @import("compiler/ir/verifier.zig");

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
pub const WasiCtx = _wasi.WasiCtx;

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

/// Component Model function executor.
pub const component_executor = @import("component/executor.zig");

/// Component Model index-space resolvers.
pub const component_indexspace = @import("component/indexspace.zig");

/// Component Model instance and resource store.
pub const component_instance = @import("component/instance.zig");
pub const component_core_backend = @import("component/core_backend.zig");
pub const component_aot = @import("component/aot.zig");
pub const component_aot_compile = @import("component/aot_compile.zig");

/// Component Model async ABI (tasks, futures, streams).
pub const component_async = @import("component/async.zig");

/// Component composition and linking.
pub const component_compose = @import("component/compose.zig");

/// Minimal WASI cli/run-style host adapter for component instances.
pub const wasi_cli_adapter = @import("component/wasi_cli_adapter.zig");

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

/// WASIp2 → WASIp3 I/O polyfill (#481): virtualize 0.2 `input-stream` /
/// `output-stream` resources over the 0.3 `stream<u8>` canonical type.
pub const wasi_p2_to_p3_io_polyfill = @import("wasi/preview3/p2_to_p3_io_polyfill.zig");

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
    .string = config.version,
    .mode = @import("builtin").mode,
};

test {
    std.testing.refAllDecls(@This());
}

test "version string comes from build config" {
    try std.testing.expectEqualStrings(config.version, version.string);
}
