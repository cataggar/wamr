# Native x86_64 AOT embedding

This is a **library-only, compiler-free embedding boundary**, not an Unikraft
image/application. Native boot and exact-image qualification remain external in
[cataggar/unikraft#156](https://github.com/cataggar/unikraft/issues/156).
Do not treat a Linux test run or a successful freestanding link as native boot
acceptance. No Azure resources are created by this build.

## Build boundaries

```sh
zig build -Dprofile=unikraft-aot -Doptimize=ReleaseSafe -j2
```

The default target for this profile is `x86_64-freestanding-none`. Other targets,
libc, interpreter/JIT/compiler/component/thread feature flags are rejected.
The early build-graph return precedes hosted dependencies and configuration.
The installed artifacts are `lib/libwamr-aot.a` and `include/wamr_aot.h`.
The exported Zig module is `wamr-aot`; its `aot` namespace provides the Zig API.
Both APIs reach the same implementation.
The same module now optionally exposes `.benchmark` and `.runner` for the
[bounded freestanding v2 guest producer](bench/native-guest-producer.md). Its
Platform/Session/WASI types are shared, not copies imported through another root.
The optional `minimal-wasi` context and `native-wasi` import adapter are also
exported with the same freestanding module settings. They are not linked into
`wamr-aot` unless the embedding application imports them; their hosted test
artifacts are not constructed in this profile.

The core runtime graph comprises `aot_native.zig`, `api/aot.zig`, `native_format.zig`,
`native_abi.zig`, `trap_jmp.zig`, `shared/allocation_limit.zig`,
`platform/unikraft.zig`, and Zig standard-library
allocation/intrinsic support. It does **not** import the hosted `runtime.zig`,
`common/types.zig`, host bridge, WASI, compiler, interpreter, components or thread
manager. The optional benchmark adds only the explicit minimal-WASI adapter,
Session, bounded JSON/hash/report helpers and requested-allocation counter.
The embedding archive does **not** bundle Zig `compiler_rt`; the native final
link owns compiler intrinsics and strong memory helpers. The Unikraft EFI final
link uses `zig cc -rtlib=compiler-rt`. Bundling weak/hidden Zig `memcpy`, `memset`
or `memmove` into this archive can make Unikraft's strong symbols hidden/local
and violate its existing IRQ/scheduler binding checks. Do not weaken those
checks. Standalone audit executables resolve their own final-link intrinsics;
neither compiler runtime intrinsics nor those memory helpers are a wasm compiler.

`native-aot-check` links an ELF with **all public C entry points retained**
(`rdynamic`), so lazy unused imports or linker garbage collection cannot hide
unresolved dependencies. This ELF is a link audit, not a bootable image. Its
entry symbol is deliberately the contract-version query. Do not execute it.
The library uses PIC for the native EFI application's PIE final link,
the x86_64 SysV ABI, no red zone, no libc stack protector, and
single-threaded Zig support, with stack checking, unwind tables and Zig error
tracing disabled. These match ordinary native application objects in
`support/build/native-target-object.zig` at Unikraft commit
`40d8fc7096bd10fa94bb1491993724eeecf2f180`. They are **not ISR settings**: do not
compile this runtime as an interrupt handler or disable the SSE facilities its
application ABI and generated scalar floating-point code require. The native
application must preserve Unikraft's IRQ return and FPU ownership contract.
Hosted defaults remain unchanged.

The default install also runs `native-aot-guest-check`, a separate freestanding
all-entry-path link audit of the optional request/result producer. It retains
real `benchmark.run` and Session invocation/reset/evidence calls, including error
paths. Both API/producer audits are PIE links. A third audit links the actual
`libwamr-aot.a` into a freestanding PIE, so executable-side implicit PIC cannot
hide non-PIC archive objects. None is bootable or introduces a compiler.

The separately selected `-Dprofile=unikraft-jit` is documented in
[unikraft-jit.md](unikraft-jit.md). It does not add a compiler to this profile.

## Matching host compiler and artifact identity

```sh
# Run on a SysV host (Linux/macOS), not a Windows-built compiler.
zig build native-aot-fixture -j2
# For a caller's core wasm:
zig-out/bin/wamrc compile --target=x86_64 \
  --profile=unikraft-x86_64 input.wasm -o workload.cwasm
```

`native-aot-fixture` builds `tests/unikraft-aot/fixture.zig` with the installed
Zig, then invokes **this checkout's** host-side wamrc. It installs that compiler
in `zig-out/bin/` and the original wasm and cwasm in `zig-out/fixtures/`, so their
identities can be retained together. It does not copy a Linux artifact and
change its target label. The explicit compiler profile checks the module and
pre-optimization IR against its supported feature policy before emission.
Unsupported atomics/threads, SIMD, EH, memory64, multiple memories, non-function
imports, reference-valued signatures, or unsupported IR operations fail closed.
The policy also rejects source `ref.as_non_null` (including instructions erased
by lowering), declarative element segments, runtime element expressions and
nonconstant active-element offsets. These must not silently lose traps or
change the element index space during emission. Active and passive funcref
segments support both function indices and null entries.
The same target also compiles the existing 36-byte `tests/coldstart/noop.wasm`
as `native-noop.cwasm`, exercising the production C API without any imports.
`tables.wat` uses the existing pinned WABT assembler and matching wamrc to
produce `native-tables.wasm`/`.cwasm` for reference-table regressions.

Retain the source revision, `zig version`, host compiler hash/build options,
fixture-source hash, wasm hash, cwasm hash, native library hash and image hash
with downstream evidence, for example:

```sh
git rev-parse HEAD
zig version
sha256sum tests/unikraft-aot/fixture.zig \
  zig-out/bin/wamrc \
  zig-out/fixtures/native-fixture.wasm zig-out/fixtures/native-fixture.cwasm \
  tests/coldstart/noop.wasm zig-out/fixtures/native-noop.cwasm \
  tests/unikraft-aot/tables.wat zig-out/fixtures/native-tables.wasm \
  zig-out/fixtures/native-tables.cwasm \
  zig-out/lib/libwamr-aot.a
```

The native loader requires format version 11, the ELF64LE/x86_64/SysV target
tuple, profile flag `0x554b0001`, runtime contract 1, and the exact known feature
mask. Ordinary hosted containers have no native contract and are rejected.
CPUID checks SSE2, SSE4.1, POPCNT, BMI1 and LZCNT before executable mapping.
The embedder must enable/preserve SSE state. Native text remains **trusted
executable code**: metadata validation does not make arbitrary `.cwasm` safe.
Use only source-pinned artifacts from the matching compiler.

## Caller-owned platform

`Platform`/`wamr_aot_config` are explicit capability contracts, not successful
stubs for unsupported system calls:

* Allocation is entirely caller-owned. No page allocator, libc allocator or
  global fallback exists. Alignment is part of the C allocator contract.
* Reserve returns an inaccessible 4096-aligned, stable virtual range. Commit
  atomically makes an uncommitted subrange RW with zero-filled pages. Failure
  must leave earlier committed bytes and the reservation valid.
* Protection is exactly NONE, RW or RX; there is no RWX mode. Text is copied
  while RW, then transitioned to RX before function pointers become callable.
  A failed transition destroys the instance instead of executing writable code.
* Unmap releases the **whole original reservation**, including uncommitted
  pages. It is an infallible ownership-release callback; adapters must provide
  that guarantee, not silently ignore an underlying teardown error.
* Monotonic time comes exclusively from the caller. Clock failure is explicit.

The corresponding native Unikraft interfaces are `uk_posix_memalign`/`uk_free`
in `lib/ukalloc/include/uk/alloc.h`, `uk_vas_get_active`, reservation/mapping/
attribute APIs in `lib/ukvmem/include/uk/vmem.h`, low-level
`uk_paging_page_map`/`uk_paging_page_set_attr`/`uk_paging_page_unmap` in
`lib/ukpaging/include/uk/paging.h`, and `ukplat_monotonic_clock` in
`include/uk/plat/time.h` on the `zig16` branch, pinned for this integration to
`40d8fc7096bd10fa94bb1491993724eeecf2f180`. These are **integration references,
not implemented shims**. In particular, simply forwarding commit to a lazy
anonymous map is insufficient for deterministic allocation failure, and
`uk_vma_set_attr` can enter `UK_CRASH` on a low-level protection failure.
The concrete adapter mapping is:

| Capability | Pinned native API | Required behavior |
| --- | --- | --- |
| allocation/free | `uk_posix_memalign` / `uk_free` | Use the explicitly selected `uk_alloc`, preserving requested alignment. |
| reservation | `uk_vma_reserve(vas, &vaddr, len)` | Reserve virtual space without physical pages in the selected `uk_vas`. |
| commit | `uk_vma_map_anon` at the exact reserved address, `UK_VMA_MAP_REPLACE \| UK_VMA_MAP_POPULATE` | Eagerly allocate only the requested prefix extension; restore the original reservation on failure if the underlying operation replaced it. |
| protection | `uk_vma_set_attr` / appropriate checked lower-level paging operation | Apply RW→RX without an RWX interval; qualify low-level failure handling rather than treating a crash path as a returned error. |
| release | `uk_vma_unmap` plus owned mapping bookkeeping | Release the complete original reservation and physical pages; establish reliable teardown despite VMA splitting/merging. |
| monotonic clock | `ukplat_monotonic_clock()` | C ABI `u64` nanoseconds; never relabel as realtime or CPU time. |

At the pinned revision, `plat/hyperv/time.c:892` implements monotonic time as
`hyperv_reference_delta_ns(hyperv_reference_time(), hyperv_boot_ref)`. The
separate wall-clock implementation at line 898 uses
`hyperv_wall_time_ns(hyperv_epoch_ns, hyperv_efi_ref, hyperv_reference_time())`.
Both helpers saturate at `UINT64_MAX` on conversion/addition overflow; the
embedding clock boundary rejects this sentinel as `ClockFailed`.
`plat/hyperv/include/hyperv/clock.h` defines `HYPERV_REFERENCE_TICK_NS = 100`:
reference ticks are multiplied by 100 to produce nanoseconds. **Units are not
resolution**; a native WASI provider must report the established 100 ns
resolution, not 1 ns merely because the timestamp's unit is nanoseconds.
This library exposes only monotonic time and makes no resolution claim.
An optional WASI provider must verify
the presence of a valid EFI epoch before exposing realtime, and must return an
unsupported result for unavailable process/thread CPU clocks. Neither boot
time nor monotonic time is a substitute for those clocks.

The downstream adapter must establish transactional eager commit, safe VMA/page
ownership and reliable teardown for its configured native page tables before
claiming this platform contract. Its application profile must explicitly enable
and initialize `LIBUKVMEM` (default **off**) and the monotonic clock;
`LIBUKVMEM` selects ukpaging, ukalloc, ukdebug, ukisrlib and uklcpu. Their
availability must not be inferred from an existing networking image. No POSIX `mmap`,
`mprotect`, signals or
`process.exit` occurs in the native library.

### Checked lower-level paging alternative

Native error-returning page hooks **are available**. At the pinned revision,
[`uk/paging.h`](https://github.com/cataggar/unikraft/blob/40d8fc7096bd10fa94bb1491993724eeecf2f180/lib/ukpaging/include/uk/paging.h)
exposes `uk_paging_pt_get_active()`, `uk_paging_page_mapx()` (and its
`uk_paging_page_map` convenience macro), `uk_paging_page_set_attr()` and
`uk_paging_page_unmap()`. The mapping/attribute/unmap functions return errors
directly, unlike the fatal-error paths in the ukvmem wrappers. Mapping with
`UK_PAL_PADDR_INV` allocates physical frames; requesting 4 KiB pages with
`UK_PAGING_PAGE_FLAG_FORCE_SIZE` keeps new mappings at that size, avoiding
later large-page splits within an exclusively owned range. Unmap skips absent
mappings and normally releases frames. `KEEP_FRAMES` and `KEEP_PTES` explicitly
retain frames or the page-table hierarchy; they are ownership choices, not
rollback guarantees.

A candidate adapter could use ukvmem only to reserve an inaccessible virtual
range, then exclusively populate/protect/release its page mappings through
ukpaging. This is **not a qualified implementation**. Before adopting it,
downstream integration must inspect the complete pinned implementations and
establish:

* Exclusive VMA, frame and page-table ownership, with no overlapping mappings
  or competing ukvmem operations; reservation release must not double-free
  pages already released through ukpaging. The globally selected page table
  must also be active on the executing CPU, with correct synchronization and
  TLB handling.
* Zero-filled eager commit and rollback after partial mapping, frame-allocation,
  page-table-allocation or protection failures. Returned errors do not imply
  atomicity; rollback must preserve previous committed bytes and reservation
  ownership, including when cleanup itself fails.
* Separate reserved-VA, committed-frame and page-table/allocator metadata
  accounting, and reliable release of the whole reservation without leaks.
  A fallible lower-level unmap does not by itself satisfy this SDK's infallible
  release callback.
* Real native failure-injection and image evidence for these guarantees,
  including W^X transitions, stable growth and teardown under allocation
  pressure. Fixed-size pages alone do not prove that failures are impossible.

This lower-level route remains an explicit alternative for cataggar/unikraft#156,
not a successful stub or a reason to remove Wasm bounds checks, import validation,
W^X or trap isolation. The missing acceptance is transactional ownership and
native execution evidence, not the existence of error-returning paging APIs.

## Calls, memory and terminal results

The C header documents layout, lifetimes, callbacks and terminal tags. Zig:

```zig
const instance = try aot.Instance.load(allocator, platform, caller_bytes,
    imports, .{ .max_memory_pages = 256, .max_table_elements = 65536 });
defer instance.deinit();
const startup = try instance.start();
// Inspect startup: returned, trap, exit, or host_error.
var results: [1]aot.Value = undefined;
const outcome = try instance.call("add",
    &.{ .{ .i32 = 20 }, .{ .i32 = 22 } }, &results);
// Only read results[0] if outcome == .returned and outcome.returned == 1.
```

C callers may pass NULL array/string pointers when the corresponding length is
zero, including no-import modules and zero-argument/zero-result exports. NULL
with a nonzero length returns `InvalidArgument`; mandatory platform/configuration,
handle and output objects must still be valid.

Load copies caller bytes and resolves every required import against its full
module/name/parameter/result signature before mapping executable code. Export
calls check argument count/type and result capacity. Calls support up to 16
scalar parameters and one scalar result; imports support up to five scalar
parameters and one result, with 64 statically compiled import entry points.
There is no executable host-trampoline generator.

Start is explicit and may run once. A module with a start function cannot be
called before start. A terminal start outcome prevents subsequent calls.
Ordinary exported-function traps do not prevent subsequent checked calls.

Memory32 reserves its maximum up front, bounded by caller limits, and commits
only the initial/grown prefix. Growth never relocates the base and never exposes
failed commits. Existing bytes survive; newly visible bytes are zero.
`memory.grow` returns `-1` for limit/allocation failure. Globals, funcref tables
and active/passive initialization support normal core scalar workloads.

Generated bounds/arithmetic/unreachable helpers unwind through a per-instance
assembly continuation, not signals or process exit. A host callback requesting
`HostContext.terminate(code)` / `wamr_aot_host_exit` must then **return normally**:
host cleanup runs first and the dispatcher unwinds guest frames afterwards.
Host errors use the same terminal boundary. No guest instruction following a
terminal import executes. Instances must be serialized; same-instance reentry
returns `Busy`. Caller contexts must outlive the instance.

This is not a general native fault handler: arbitrary machine faults and
exhaustion of the caller's native stack are not recovered. Supply an adequate
native stack and a bounded trusted workload; native recursion/stack budgeting is
not yet qualified. Compiler reuse, executable allocation and typed dispatch are
separated so a later opt-in compiler/JIT layer can reuse this boundary without
adding compilation to the default guest graph.

### Honest phase instrumentation

`Options.timings = &LoadTimings{}` enables three reads of the supplied monotonic
clock at real internal boundaries; `wamr_aot_load_timed` exposes the same
implementation through C. Untimed calls make no clock reads:

1. Load begins immediately before allocating/copying the owned input.
2. Instantiation begins after format/ABI/CPU/index validation. Required-import
   binding/signature validation belongs to instantiation.
3. Instantiation ends after globals, memories, tables, initialization, executable
   mapping/protection and vmctx helper wiring are complete.

`load_ns` and `instantiate_ns` are valid only when their respective `completed`
bits (1 and 2) are set. Failures preserve only fully measured phases, never a
fabricated zero. Failed/saturated/backwards clock readings return `ClockFailed`;
even a final timing failure rolls back the newly created instance. The timestamp
units remain ns and do not assert a clock resolution.

The same boundary is explicitly available as `Instance.loadModule` followed by
`instance.instantiate`. These staged methods do not read clocks themselves;
the Linux producer times its WASI setup together with the real instantiate call.
`Instance.load` and the C timed-load entry point remain convenience wrappers over
these stages. Uninstantiated owners cannot start/call/grow, and an instantiation
attempt cannot be retried after failure.
All non-timing options are frozen at `loadModule`; changing any at instantiation
returns `OptionsMismatch` before mapping. This also applies to additive budget
options when integrating the opt-in JIT layer; a second stage cannot weaken
already-admitted limits.

These cover the common loader/instance implementation, not filesystem reads,
CLI/subprocess setup, C argument adaptation or caller-owned WASI context setup.
A benchmark producer must measure any additional required setup separately and
report its coverage honestly. First/repeated invocation timing remains the
producer's responsibility. In particular, a fresh instance per sample is not
same-instance steady state, and clearing a sticky WASI exit alone does not prove
that repeatedly invoking `_start` is valid for a workload's libc/descriptors/
globals. This API does not claim that unqualified repeatability.

`Instance.restoreMemory` supports an inactive embedder's memory snapshot reset,
including protection revocation and logical-bound restoration after growth.
It does not reset globals/tables/passive segments or host state by itself.
If protection revocation fails, the instance becomes non-callable and must be
deinitialized: a failed provider may already have changed part of the mapping.
The [Linux producer](bench/native-linux-producer.md) implements and tests that
complete same-instance snapshot policy for the pinned workloads.

### Performance choices follow the boundary

The official [Unikraft architecture guidance](https://unikraft.org/docs/internals/architecture)
emphasizes narrow interchangeable micro-libraries and static linking instead of
bypassing APIs for speed. This implementation applies that guidance concretely:

* The default guest artifact is one statically linked AOT library. WASI and
  platform adapters remain separate narrow modules; POSIX, a filesystem and a
  scheduler are not compatibility prerequisites for the runtime.
* The caller's metadata allocator is independent of the reserve/commit/
  protection policy for linear memory and code. An embedding application can
  select appropriate `ukalloc` implementations for boot/application lifetimes
  without changing the runtime or pretending address reservation is committed
  RAM. Allocator ownership must remain valid through corresponding frees.
* Load/instantiation allocate metadata and table capacity once. Invocation,
  scalar dispatch, host thunks and explicit trap handling use bounded stack
  scratch rather than per-call heap allocation. A regression freezes the
  caller allocator after instantiation and exercises repeated exact-result
  calls, host calls, noop, traps and memory growth. Growth still invokes the
  separate page-commit capability: it is not claimed to avoid physical allocation.
* A suitable bounded application may use run-to-completion without adding a
  runtime scheduler. Whether the actual pinned image can omit a scheduler is a
  native integration decision, not established by `single_threaded=true`.
* Instance reuse belongs outside a measured steady region only after workload
  restart semantics are qualified. Fresh-instance samples must be reported as
  such, rather than mislabeled as same-instance steady state.

Assess these choices with real load/instantiate/invocation phases and explicit
image/reserved/committed-memory evidence, including measurement coverage. This
does not predict any CoreMark OS speedup. Unikraft components share a protection
domain; that is **not** a reason to remove wasm bounds checks, W^X, import/ABI
validation or per-instance trap isolation. The pinned fork's native Zig build
and image configuration remain authoritative over generic Make/Kconfig examples.
No network or storage scope is added here.

## Regression evidence and remaining acceptance

```sh
zig build test-native-aot-format test-native-aot-abi -j2
zig build test-native-aot -j2
zig build test-native-aot-build -j2    # compile-only, not execution acceptance
```

The real API fixture covers exact i32/i64/f64 results, noop, host imports and
indirect calls; memory load/store/growth; OOB, unreachable, divide/overflow
traps; checked API errors; terminal exit/host error with cleanup; C ABI byte
ownership; every allocator failure and reserve/commit/protect rollback.
Table tests execute in-range and out-of-range `table.get`, active/passive null
initialization, indirect calls, `table.init` and dropped-segment traps, and
check that null entries clear both pointer and signature storage. Existing
backend table-read/indirect traps report `unreachable_instruction`; runtime
table-mutation bounds checks report `out_of_bounds_table`.
Format tests reject incompatible metadata, malformed section bounds, duplicate
sections and invalid indices, including all interior truncations. ABI tests
compare **every VmCtx field offset** with the hosted runtime.
Compiler tests lower real source opcodes before checking admission, including
erased reference instructions, opcode-valued immediates and element index
preservation. These policy tests also run with ordinary compiler unit tests.

Linux x86_64 runs directly. A non-x86 build host needs an already-provisioned
`qemu-x86_64 -cpu max` to execute these Linux test binaries; the test target does
not install it. The existing x86_64 CI job runs both real API tests and the
freestanding compile/link boundary. Linux/QEMU tests are not Unikraft boots.

Remaining external acceptance: implement and compile the concrete native
Unikraft allocator/page/clock adapter, integrate the library and embedded pinned
fixture into the separate native app/image, verify real W^X/PTEs and stable
growth under native allocation pressure, boot the exact image and retain its
typed terminal evidence. Real image boot and cloud execution are **pending**;
this work references rather than closes #1047.
