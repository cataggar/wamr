# Frame-attribution smoke fixture

`frame_origins.wasm` is the tracked binary form of `frame_origins.wat`.
It keeps a lowered loop local in the frame and holds twenty loaded values
across a WASI call, forcing deterministic allocator spill stores/reloads.

`tests/test_aot_jit_attr.py --wamrc <path>` compiles it with
`WAMR_AOT_FRAME_ATTRIBUTION*` for x86_64 and AArch64, verifies diagnostic-on
and diagnostic-off `.cwasm` byte identity, validates each sidecar against the
final emitted instructions, and reconciles allocator accesses with
`WAMR_AOT_SPILL_METRIC`. It executes the host-architecture module when the
sibling `wamr` binary is available; the other architecture is cross-compiled
and statically validated when a matching `objdump` is installed.

The same test builds a real AArch64 `return_call` module and verifies that its
direct tail-call `b` relocation is distinct from `bl`, diagnostic-on/off bytes
are identical, and the standalone `aot_jit_attr.py
--validate-frame-metadata` CLI accepts the exact retained `.cwasm`/sidecar
pair. Consumer fixtures also cover target-info architecture/ABI binding,
materialized large offsets, pair register/width identity, and corrupt
offset/slot/region metadata.
