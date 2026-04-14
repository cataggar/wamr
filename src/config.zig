//! Compile-time configuration for WAMR.
//!
//! Zig port of `core/config.h`. Every C `#define WASM_ENABLE_*` flag becomes a
//! `pub const` bool here, and every numeric constant becomes a typed `pub const`.
//!
//! Build-time overrides come from the `config` module (populated by build.zig).
//! Flags not yet exposed by build.zig fall back to their C defaults via
//! `@hasDecl` checks, so this file is resilient to a partially-wired build.

const builtin = @import("builtin");
const build_options = @import("config");

// ---------------------------------------------------------------------------
// Build mode
// ---------------------------------------------------------------------------

/// True when building in Debug mode (mirrors C `BH_DEBUG`).
pub const bh_debug = builtin.mode == .Debug;

// ---------------------------------------------------------------------------
// Architecture detection  (replaces BUILD_TARGET_* macros)
// ---------------------------------------------------------------------------

pub const Arch = enum {
    x86_64,
    x86,
    aarch64,
    arm,
    thumb,
    mips,
    xtensa,
    riscv64,
    riscv32,
    arc,
    unknown,
};

/// Target architecture derived from `builtin.cpu.arch`.
pub const arch: Arch = switch (builtin.cpu.arch) {
    .x86_64 => .x86_64,
    .x86 => .x86,
    .aarch64 => .aarch64,
    .arm => .arm,
    .thumb => .thumb,
    .mips, .mipsel, .mips64, .mips64el => .mips,
    .xtensa => .xtensa,
    .riscv64 => .riscv64,
    .riscv32 => .riscv32,
    .arc => .arc,
    else => .unknown,
};

/// Target OS tag from builtin.
pub const os = builtin.os.tag;

/// True when the target CPU allows unaligned memory accesses.
pub const cpu_supports_unaligned_addr_access: bool = switch (arch) {
    .x86_64, .x86, .aarch64 => true,
    else => false,
};

// ---------------------------------------------------------------------------
// Memory allocator selection
// ---------------------------------------------------------------------------

pub const MemAllocator = enum { ems, tlsf };
pub const default_mem_allocator: MemAllocator = .ems;

// ---------------------------------------------------------------------------
// Helper: read a bool build option with a comptime default.
// ---------------------------------------------------------------------------

fn opt(comptime name: []const u8, comptime default: bool) bool {
    return if (@hasDecl(build_options, name)) @field(build_options, name) else default;
}

fn optInt(comptime T: type, comptime name: []const u8, comptime default: T) T {
    return if (@hasDecl(build_options, name)) @field(build_options, name) else default;
}

// ---------------------------------------------------------------------------
// Feature flags  (bool, read from build options with fallback defaults)
// ---------------------------------------------------------------------------

/// Enable the bytecode interpreter.
pub const interp = opt("interp", false);

/// Enable Ahead-Of-Time compilation support.
pub const aot = opt("aot", false);

/// Enable dynamic AOT debugging.
pub const dynamic_aot_debug = opt("dynamic_aot_debug", false);

/// Force word-aligned reads (needed on some MCUs).
pub const word_align_read = opt("word_align_read", false);

/// Enable LLVM-based JIT (requires AOT).
pub const jit = if (aot) opt("jit", false) else false;

/// Enable lazy JIT compilation (requires JIT).
pub const lazy_jit = if (jit) opt("lazy_jit", false) else false;

/// Enable the lightweight fast JIT backend.
pub const fast_jit = opt("fast_jit", false);

/// Dump fast-JIT generated code for debugging.
pub const fast_jit_dump = opt("fast_jit_dump", false);

/// Build the AOT compiler (wamrc) component.
pub const wamr_compiler = opt("wamr_compiler", false);

/// Enable libc builtin APIs for WASM apps.
pub const libc_builtin = opt("libc_builtin", false);

/// Enable WASI (WebAssembly System Interface) support.
pub const libc_wasi = opt("libc_wasi", false);

/// Enable WASI-NN machine-learning extension.
pub const wasi_nn = opt("wasi_nn", false);

/// Enable GPU backend for WASI-NN.
pub const wasi_nn_gpu = opt("wasi_nn_gpu", false);

/// Enable external delegate for WASI-NN.
pub const wasi_nn_external_delegate = opt("wasi_nn_external_delegate", false);

/// Enable ephemeral WASI-NN API surface.
pub const wasi_ephemeral_nn = opt("wasi_ephemeral_nn", false);

/// Enable Emscripten libc compatibility.
pub const libc_emcc = opt("libc_emcc", false);

/// Enable pthreads library for WASM apps.
pub const lib_pthread = opt("lib_pthread", false);

/// Enable pthread semaphore support.
pub const lib_pthread_semaphore = opt("lib_pthread_semaphore", false);

/// Enable WASI threads proposal.
pub const lib_wasi_threads = opt("lib_wasi_threads", false);

/// Allocate auxiliary stacks on the heap (follows lib_wasi_threads).
pub const heap_aux_stack_allocation = opt("heap_aux_stack_allocation", lib_wasi_threads);

/// Enable copying the call stack for diagnostics.
pub const copy_call_stack = opt("copy_call_stack", false);

/// Enable the base library.
pub const base_lib = opt("base_lib", false);

/// Enable the application framework.
pub const app_framework = opt("app_framework", false);

/// Platform supports mremap(2).
pub const have_mremap = opt("have_mremap", false);

/// Enable WASM bulk-memory operations proposal.
pub const bulk_memory = opt("bulk_memory", false);

/// Optimized bulk-memory operations.
pub const bulk_memory_opt = opt("bulk_memory_opt", false);

/// Enable WASM shared memory proposal.
pub const shared_memory = opt("shared_memory", false);

/// Enable the thread manager component.
pub const thread_mgr = opt("thread_mgr", false);

/// Enable source-level interpreter debugging.
pub const debug_interp = opt("debug_interp", false);

/// Enable AOT debug information.
pub const debug_aot = opt("debug_aot", false);

/// Allow loading custom WASM sections.
pub const load_custom_section = opt("load_custom_section", false);

/// Enable the WAMR log system.
pub const log = opt("log", true);

/// Delegate socket initialisation to the host (Windows WSAStartup).
pub const host_socket_init = opt("host_socket_init", false);

/// Use GCC computed-goto / labels-as-values in the interpreter.
/// In Zig this is always available via comptime dispatch.
pub const labels_as_values = opt("labels_as_values", true);

/// Enable the fast interpreter.
pub const fast_interp = opt("fast_interp", false);

/// Enable opcode execution counters.
pub const opcode_counter = opt("opcode_counter", false);

/// Support loading multiple inter-dependent modules.
pub const multi_module = opt("multi_module", false);

/// Use the minimal WASM loader (smaller code size, fewer checks).
pub const mini_loader = opt("mini_loader", false);

/// Disable hardware-trap–based bounds checking.
pub const disable_hw_bound_check = opt("disable_hw_bound_check", false);

/// Disable hardware-trap–based stack overflow checking.
pub const disable_stack_hw_bound_check = opt("disable_stack_hw_bound_check", false);

/// Enable the WASM SIMD proposal.
pub const simd = opt("simd", false);

/// GC performance profiling counters.
pub const gc_perf_profiling = opt("gc_perf_profiling", false);

/// Runtime memory-usage profiling.
pub const memory_profiling = opt("memory_profiling", false);

/// Runtime memory allocation/free tracing.
pub const memory_tracing = opt("memory_tracing", false);

/// Execution performance profiling.
pub const perf_profiling = opt("perf_profiling", false);

/// Dump the call stack on traps/exceptions.
pub const dump_call_stack = opt("dump_call_stack", false);

/// Maintain AOT stack frames for diagnostics.
pub const aot_stack_frame = opt("aot_stack_frame", false);

/// Enable GC heap verification (debug aid).
pub const gc_verify = opt("gc_verify", false);

/// Enable GC heap corruption checks (default on).
pub const gc_corruption_check = opt("gc_corruption_check", true);

/// Use a single global heap pool. Forced on when gc_verify is set.
pub const global_heap_pool = gc_verify or opt("global_heap_pool", false);

/// Build for the WASM spec-test suite.
pub const spec_test = opt("spec_test", false);

/// Build for the WASI test suite.
pub const wasi_test = opt("wasi_test", false);

/// Enable the tail-call proposal.
pub const tail_call = opt("tail_call", false);

/// Load the "name" custom section for symbolic debugging.
pub const custom_name_section = opt("custom_name_section", false);

/// Enable the reference-types proposal.
pub const ref_types = opt("ref_types", false);

/// Allow overlong encodings in call_indirect.
pub const call_indirect_overlong = opt("call_indirect_overlong", false);

/// Enable the branch-hinting proposal.
pub const branch_hints = opt("branch_hints", false);

/// Enable the GC (garbage collection) proposal.
pub const gc = opt("gc", false);

/// Enable the stringref proposal (requires GC).
pub const stringref = opt("stringref", false);

/// Enable the exception-handling proposal.
pub const exce_handling = opt("exce_handling", false);

/// Enable the tags (exception tags) proposal.
pub const tags = opt("tags", false);

/// Enable the Component Model (WASIp3).
pub const component_model = opt("component_model", false);

/// Pass user data pointer to the memory allocator.
pub const mem_alloc_with_user_data = opt("mem_alloc_with_user_data", false);

/// Enable WASM module caching.
pub const wasm_cache = opt("wasm_cache", false);

/// Enable static Profile-Guided Optimisation.
pub const static_pgo = opt("static_pgo", false);

/// Disable writing the linear-memory base to the GS segment register.
pub const disable_write_gs_base = opt("disable_write_gs_base", false);

/// Allow bounds-check mode to be set at runtime.
pub const configurable_bounds_checks = opt("configurable_bounds_checks", false);

/// Support dual-bus memory mirroring (some MCU targets).
pub const mem_dual_bus_mirror = opt("mem_dual_bus_mirror", false);

/// Linux perf integration for JIT code.
pub const linux_perf = opt("linux_perf", false);

/// Register quick AOT/JIT entry points for common signatures.
pub const quick_aot_entry = opt("quick_aot_entry", true);

/// Built-in AOT intrinsic function support.
pub const aot_intrinsics = opt("aot_intrinsics", true);

/// Enable the memory64 proposal (64-bit linear memory).
pub const memory64 = opt("memory64", false);

/// Enable the multi-memory proposal.
pub const multi_memory = opt("multi_memory", false);

/// Tag allocations with a usage category.
pub const mem_alloc_with_usage = opt("mem_alloc_with_usage", false);

/// Build for fuzz-testing (enables allocation caps).
pub const fuzz_test = opt("fuzz_test", false);

/// Enable the shared-heap extension.
pub const shared_heap = opt("shared_heap", false);

/// Shrink linear memory when possible (default on).
pub const shrunk_memory = opt("shrunk_memory", true);

/// Enable AOT module validation.
pub const aot_validator = opt("aot_validator", false);

/// Enable instruction-level metering / gas counting.
pub const instruction_metering = opt("instruction_metering", false);

/// Enable extended constant expressions proposal.
pub const extended_const_expr = opt("extended_const_expr", false);

// ---------------------------------------------------------------------------
// Numeric constants
// ---------------------------------------------------------------------------

/// AOT binary magic number (little-endian "\0aot").
pub const aot_magic_number: u32 = 0x746f6100;

/// Current AOT binary format version.
pub const aot_current_version: u32 = 6;

/// Number of LLVM ORC JIT backend threads.
pub const orc_jit_backend_thread_num: u32 = optInt(u32, "orc_jit_backend_thread_num", 4);

/// Number of LLVM ORC JIT compilation threads.
pub const orc_jit_compile_thread_num: u32 = optInt(u32, "orc_jit_compile_thread_num", 4);

/// Default code cache size for the fast JIT (bytes).
pub const fast_jit_default_code_cache_size: usize = optInt(usize, "fast_jit_default_code_cache_size", 10 * 1024 * 1024);

/// Memory reserved for the debug-interp execution engine.
pub const debug_execution_memory_size: usize = optInt(usize, "debug_execution_memory_size", 0x85000);

/// Size of the global heap pool (bytes).
pub const global_heap_size: usize = optInt(usize, "global_heap_size", 10 * 1024 * 1024);

/// Default message-queue length.
pub const default_queue_length: u32 = 50;

/// Max fraction of the global heap an app may use (integer divisor: 1/3).
pub const app_memory_max_global_heap_divisor: u32 = 3;

/// Default per-app heap size (bytes).
pub const app_heap_size_default: usize = 8 * 1024;
/// Minimum per-app heap size (bytes).
pub const app_heap_size_min: usize = 256;
/// Maximum per-app heap size (bytes, limited by EMS allocator to 1 GB).
pub const app_heap_size_max: usize = 1024 * 1024 * 1024;

/// Default per-app GC heap size (bytes).
pub const gc_heap_size_default: usize = 128 * 1024;
/// Minimum GC heap size (bytes).
pub const gc_heap_size_min: usize = 4 * 1024;
/// Maximum GC heap size (bytes).
pub const gc_heap_size_max: usize = 1024 * 1024 * 1024;

/// Default WASM operand stack size (bytes, larger on x86_64).
pub const default_wasm_stack_size: usize = switch (arch) {
    .x86_64 => 16 * 1024,
    else => 12 * 1024,
};

/// Minimum auxiliary stack size per WASM thread (bytes).
pub const wasm_thread_aux_stack_size_min: usize = 256;

/// Default native thread stack size (bytes).
pub const app_thread_stack_size_default: usize = optInt(usize, "app_thread_stack_size_default", 128 * 1024);

/// Minimum native thread stack size (bytes).
pub const app_thread_stack_size_min: usize = optInt(usize, "app_thread_stack_size_min", 24 * 1024);

/// Maximum native thread stack size (bytes).
pub const app_thread_stack_size_max: usize = 8 * 1024 * 1024;

/// Guard bytes between the WASM stack and native stack boundary.
pub const wasm_stack_guard_size: usize = optInt(usize, "wasm_stack_guard_size", blk: {
    const is_desktop = switch (os) {
        .macos, .linux, .freebsd, .netbsd, .openbsd => true,
        else => false,
    };
    break :blk if (is_desktop)
        (if (bh_debug) 5 * 1024 else 3 * 1024)
    else
        1024;
});

/// Number of guard pages for HW stack-overflow detection.
pub const stack_overflow_check_guard_page_count: u32 = switch (os) {
    .macos, .ios => switch (arch) {
        .aarch64 => 1,
        else => 3,
    },
    else => 3,
};

/// Entries in the block-address cache.
pub const block_addr_cache_size: u32 = 64;
/// Conflict-chain length in the block-address cache.
pub const block_addr_conflict_size: u32 = 2;

/// Default maximum threads per cluster.
pub const cluster_max_thread_num: u32 = 4;

/// Constant-expression evaluation stack depth.
pub const wasm_const_expr_stack_size: u32 = if (gc) 8 else 4;

/// Default GC ref-type map capacity.
pub const gc_reftype_map_size_default: u32 = 64;
/// Default GC RTT-object map capacity.
pub const gc_rttobj_map_size_default: u32 = 64;

/// Maximum number of per-instance context slots.
pub const wasm_max_instance_contexts: u32 = 8;

/// Maximum table size (elements).
pub const wasm_table_max_size: u32 = optInt(u32, "wasm_table_max_size", 1024);

/// Maximum single allocation size under fuzz-testing (~2 GB).
pub const wasm_mem_alloc_max_size: usize = if (fuzz_test) 2 * 1024 * 1024 * 1024 else 0;

// ---------------------------------------------------------------------------
// Compile-time validation (mirrors the C #error checks)
// ---------------------------------------------------------------------------

comptime {
    if (orc_jit_backend_thread_num < 1)
        @compileError("orc_jit_backend_thread_num must be >= 1");
    if (orc_jit_compile_thread_num < 1)
        @compileError("orc_jit_compile_thread_num must be >= 1");
    if (heap_aux_stack_allocation == false and lib_wasi_threads == true)
        @compileError("heap_aux_stack_allocation must be enabled for WASI threads");
}
