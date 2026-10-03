# CoreMark scalar allocator diagnostic (#985)

Compile the checked-in CoreMark fixture locally without running it:

```sh
WAMR_AOT_ALLOC_TRACE_FUNC10=1 zig-out/bin/wamrc compile --target=aarch64 \
  tests/benchmarks/coremark/coremark_wasi.wasm -o coremark-diagnostic.cwasm
```

The opt-in flag prints read-only linear-scan snapshots to compiler stderr for
**module 0, local function 10**, scalar vregs **200 and 190**, including an
incoming interval's lifetime and maximum loop depth, hint index, intervening
clobber count/mask, free and clobber-safe register masks, chosen register or
spill, active competitors in allocator order with eviction eligibility, and
final FP offsets/slot indices. Register masks use indices in the AArch64
`alloc_regs` array; `reg` fields are physical register numbers. `safe_idx`
includes busy registers; `free_idx` does not imply clobber safety.
`clobbers` includes all modeled clobber points, not solely function calls.
The trace describes the **pre-coalescing** scalar allocator decision. The
`final` line reports the allocation's stack slot before optional coalescing
(which only retargets physical register homes).

With the checked-in fixture at main `49403b8e`, both intervals start at
instruction 33, end at 152, and have depth 1. At their admission no register
is free; all 23 are clobber-safe and there are no intervening clobbers. Every
occupied register ends no later than the incoming range, so the allocator's
strict `active.end > incoming.end` eviction rule rules out eviction. They
receive scalar spill slots 1 and 2 (FP+248 and FP+256), respectively.
These are **linear-scan interval summaries**, not dynamic use counts, proven
IR def semantics, or performance predictions. The diagnostic makes no
counterfactual codegen change. Repeat runs have identical trace text; with
or without the flag the AArch64 artifact is byte-identical, and the x86_64
codegen does not consult this AArch64-only flag. Without AArch64 scalar
allocation enabled, this trace has no output. If a codegen cache is used, the
selected function is recompiled instead of reused so the trace is available.
