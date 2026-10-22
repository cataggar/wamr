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
`final` line reports the allocation's stack slot or physical register before
optional coalescing (which only retargets physical register homes).

Before #985 constant-pressure relief (`e96573c5`), both intervals started at
instruction 33, ended at 152, and had depth 1. At their admission no register
was free because 16 hoisted loop-preheader `iconst_32` intervals occupied
physical registers across `[8..152]`, and the allocator's strict
`active.end > incoming.end` rule prevented evicting equal-end constants,
spilling `v200` and `v190` to scalar spill slots 1 and 2 (FP+248 and FP+256).

After excluding suppressed immediate-folded `iconst_*` definitions from scalar
live ranges, folding 12-bit comparison immediates into `CMP`/`SUBS`, and wiring
rematerialisation-aware eviction on AArch64, 6 registers remain free at
admission (`free_idx=0x7e0000` / `0x7c0000`) and `v200` and `v190` receive
physical registers (`x23` and `x24`) with zero spills in `local_func=10`.
These are **linear-scan interval summaries**, not dynamic use counts, proven
IR def semantics, or performance predictions. The diagnostic makes no
counterfactual codegen change. Repeat runs have identical trace text; with
or without the flag the AArch64 artifact is byte-identical, and the x86_64
codegen does not consult this AArch64-only flag. Without AArch64 scalar
allocation enabled, this trace has no output. If a codegen cache is used, the
selected function is recompiled instead of reused so the trace is available.
