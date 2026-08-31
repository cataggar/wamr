# Frame-attribution smoke fixture

`frame_origins.wasm` is the tracked binary form of `frame_origins.wat`.
It keeps a lowered loop local in the x86 frame and holds twenty loaded values
across a WASI call, forcing deterministic allocator spill stores/reloads.

`tests/test_aot_jit_attr.py --wamrc <path>` compiles it with
`WAMR_AOT_FRAME_ATTRIBUTION*`, verifies the sidecar against the emitted
`.cwasm` and `objdump`, reconciles allocator accesses with
`WAMR_AOT_SPILL_METRIC`, and executes the resulting module when the sibling
`wamr` binary is available.
