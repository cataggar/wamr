# Security Audit Checklist

This checklist supports proactive reviews of sandbox-critical WAMR paths. It is
not the vulnerability reporting policy; see [SECURITY.md](SECURITY.md) for how
to report security issues.

## Threat model

Assume attackers can provide arbitrary core WebAssembly modules, component-model
binaries, exported function arguments, and guest linear-memory contents. Host
applications decide which imports, WASI capabilities, preopens, sockets, HTTP
clients, and resource handles are exposed; runtime code must preserve those
capability boundaries.

Sandbox-critical failures include host memory access from generated native code,
incorrect interpreter/AOT behavior that skips WebAssembly traps, integer
overflow in guest-memory range checks, stale or confused resource handles, and
incorrect canonical ABI lifting/lowering across guest memory.

Out of scope for this checklist: performance-only work, the public security
advisory process, and broad fuzz-campaign operations. Fuzzing infrastructure is
tracked in [tests/fuzz/README.md](tests/fuzz/README.md).

## Review checklist

Use this list when reviewing a subsystem. Mark each item as reviewed in the
audit notes and record exact files/functions examined.

### Loader and instantiation

- Validate type, function, memory, table, global, element, data, and start
  section indices before runtime use.
- Check active data and element segment offsets with overflow-safe arithmetic.
- Reject negative or oversized init-expression offsets before copying into
  guest memory or tables.
- Preserve import type compatibility, including mutable-global invariance,
  function signatures, memory/table limits, and reference types.
- Ensure memory growth and shared-memory initialization keep size, capacity,
  waiter queues, and refcounts consistent.

### Interpreter runtime boundary

- Check every memory access as `addr + offset + width` before slicing host
  memory.
- Check `memory.copy`, `memory.fill`, atomics, SIMD lane loads/stores, and bulk
  memory operations with overlap and overflow cases.
- Check table indices, null references, dropped element/data segments, and
  `call_indirect` signature mismatches before dispatch.
- Verify trap paths return errors or traps consistently and do not continue with
  partially updated state.

### AOT compiler and native runtime

- Confirm frontend lowering preserves every WebAssembly trap condition for
  loads/stores, atomics, memory growth/copy/fill, numeric traps, table ops,
  `br_table`, `call_indirect`, and reference calls.
- Review IR passes for rewrites that could move, drop, or duplicate checks on
  only some control-flow paths.
- Review register allocation and spilling under high register pressure,
  helper-call clobbers, callee-saved preservation, stack-frame layout, and
  multi-value returns.
- Check generated native code performs bounds checks before native memory
  access and uses overflow-safe effective-address formation.
- Check `VmCtx` field layout, helper pointers, table/function-pointer dispatch,
  global synchronization, and platform-specific trap recovery.
- On platforms where in-process AOT trap recovery is unsupported, validate trap
  reproducers in a subprocess or sentinel harness instead of killing the test
  runner.

### WASI and component host adapters

- Treat every guest pointer/length pair as untrusted and check it before
  reading or writing guest memory.
- Preserve preopen and path sandbox rules; reject absolute paths, `..`
  components, Windows drive prefixes, and alternate separators where relevant.
- Keep socket, HTTP, stream, descriptor, and resource tables capability-scoped;
  stale handles must not resolve to newly authorized resources accidentally.
- Record ownership for every resource type. Resource-drop paths must either
  close owned resources exactly once or intentionally no-op for borrowed
  adapter-owned resources.
- Default-deny outbound network/HTTP behavior unless a capability allow-list is
  explicitly implemented and tested.

### Component canonical ABI

- Check `ptr + len`, `ptr + offset`, and `ptr + element_size * count` with
  overflow-safe host-sized arithmetic before slicing linear memory.
- Validate UTF-8, UTF-16, latin1+utf16, list, string, record, tuple, variant,
  flags, option, result, and resource layouts before lifting/lowering values.
- Ensure allocator-owned `InterfaceValue` payloads are released exactly once on
  success and error paths.
- Ensure canonical lower trampolines resolve the intended memory and
  `cabi_realloc`, including alias-exported memory edge cases.

## Findings workflow

Confirmed findings should be filed as separate GitHub issues. Each issue should
include:

- Affected files/functions.
- Security or correctness impact.
- Reproducer shape or regression test, when possible.
- Suggested fix direction.
- Links to any follow-up PRs or audit records.

Do not file speculative findings without code evidence or a reproducible path.
Document uncertain risks as follow-up audit notes instead.

## Issue #223 audit record

### Completed targeted pass: AArch64 AOT backend and AOT runtime boundary

Reviewed files:

- `src/compiler/frontend.zig`
- `src/compiler/ir/analysis.zig`
- `src/compiler/ir/passes.zig`
- `src/compiler/ir/regalloc.zig`
- `src/compiler/codegen/aarch64/compile.zig`
- `src/compiler/codegen/aarch64/emit.zig`
- `src/runtime/aot/runtime.zig`

Reviewed areas:

- Memory load/store and atomic lowering.
- `memory.grow`, `memory.copy`, `memory.fill`, `br_table`,
  `call_indirect`, return-call forms, table/reference ops, and numeric traps.
- CFG/liveness, SSA/mem2reg, phi placement/lowering, register allocation,
  clobber handling, and spill offsets.
- AArch64 memory-address formation, bounds-check branches, helper-call ABI,
  callee-saved/caller-saved handling, stack frame layout, table/function calls,
  and multi-value returns.
- AOT `VmCtx` setup, memory/table/helper pointers, trap helpers,
  `callFuncScalar` argument/result marshalling, and global synchronization.

Result: no confirmed AArch64 AOT sandbox finding in this pass.

Residual risks to test in follow-up work:

- AArch64 `call_indirect` and `call_ref` tail-call paths with multi-result
  signatures and fixed `VmCtx` offsets.
- High register-pressure code that combines helper calls, spills, phis, and
  memory/table checks.
- Windows-specific AOT trap-as-error recovery behavior.

### Follow-on triage

Reviewed files:

- `src/runtime/interpreter/instance.zig`
- `src/runtime/interpreter/interp.zig`
- `src/runtime/interpreter/loader.zig`
- `src/runtime/common/types.zig`
- `src/component/instance.zig`
- `src/component/canonical_abi.zig`
- `src/component/wasi_cli_adapter.zig`
- `src/wasi/preview2/streams.zig`
- `src/wasi/preview2/sockets.zig`
- `src/wasi/preview2/http.zig`

Confirmed finding filed separately:

- [#240](https://github.com/cataggar/wamr/issues/240): canonical ABI
  guest-memory range checks could overflow on wrapping `u32` offsets.

Potential follow-up audit areas:

- Component memory and `cabi_realloc` resolution with alias-exported memories.
- WASI adapter resource-drop behavior for borrowed/preopen handles.
- Interpreter table grow/fill/copy behavior with table64 and cross-module
  funcref type checks.

## Validation entry points

- `zig build test`
- `zig build spec-tests-aot`
- `zig build fuzz` (see [tests/fuzz/README.md](tests/fuzz/README.md))
