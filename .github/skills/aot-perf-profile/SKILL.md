---
name: aot-perf-profile
description: Profile a precompiled WAMR AOT run (a component like keyvault/#798, or a core module like CoreMark/#393) on an Azure Linux x86_64 VM and attribute cycles per wasm function and per instruction class. Installs `perf` (kernel-tools), records the precompiled load+execute run, then de-anonymizes the `[JIT]` mapping that perf cannot symbolize by mapping sample IPs to `local_func` indices via the `.cwasm` function section. With the opt-in compiler sidecar it distinguishes allocator spills from wasm local/phi traffic, explicit frame scratch, fixed runtime/ABI state, and unknown frame accesses. Use for runtime-gap investigations to decide whether allocator spills, local/frame traffic, zero-extension/move overhead, bounds-checks, or dispatch dominate before committing to codegen levers.
---

# AOT perf profiling + JIT de-anonymization

Attribute where a **precompiled** WAMR AOT run spends its cycles, broken
down per wasm `local_func` and per instruction class. Built for the #798
"close the WAMR↔Wasmtime AOT runtime gap" work (the keyvault/TCGC /
SpiderMonkey workload), but applies to any precompiled component run.

The hard part this skill solves: WAMR maps generated AOT code as an
**anonymous RX mmap**, so `perf` lumps ~98% of the run into one
unsymbolized `[JIT]` bucket. This skill recovers per-function and
per-instruction attribution from the `.cwasm` layout.

## When to use

- You have a precompiled WAMR run that is slower than a reference
  (wasmtime/native) and need to know **what** is slow — spills vs.
  bounds-checks vs. dispatch vs. memcpy — before picking a codegen lever
  (#798 Tier 0; gates #808).
- You want a per-function runtime ranking to confirm which function is
  actually hot (it is often **not** the biggest compile-time function —
  see the worked example below).
- You're validating that a codegen/regalloc change moved the hot path.

Do **not** use this skill for:

- AOT *correctness* bugs (wrong output / traps) — use `aot-diff-debug`
  then `aot-pass-bisect`.
- Compile-time cost — that's `WAMR_AOT_CODEGEN_TIMING` / pass-timing, not
  a runtime `perf record`.

## Pinned keyvault harness (preferred)

For #798 comparisons, use `scripts/bench_keyvault.py` and a completed
`tests/benchmarks/keyvault/manifest.example.json` rather than assembling
commands ad hoc. The harness validates full source revisions and SHA-256 values
for every binary/input/mounted tree, precompiles both runtimes outside measured
execution, requires at least five measured runs, and rejects response-byte or
generated-file drift. It also records host/tool/input identity and the
Wasmtime-time/WAMR-time ratio:

```sh
python3 scripts/bench_keyvault.py \
  --manifest /path/to/keyvault.lock.json \
  --work-dir "$PWD/zig-out/keyvault-798" \
  --warmups 2 --runs 7
```

Add `--profile` to capture and attribute a WAMR perf run. In that mode missing
perf/objdump, inadequate permissions or sample count, and insufficient
attribution coverage are hard failures; the harness never silently falls back
to an unprofiled success. See `tests/benchmarks/keyvault/README.md` for the
lockfile and equivalence contract.

## Environment

This VM is **Microsoft Azure Linux 3 (`azurelinux`, `azl3`)**, x86_64.
Package manager is `tdnf` (and `dnf`). `perf` is **not** preinstalled and
ships in the **`kernel-tools`** package, which must match the running
kernel.

## Step 1 — install `perf` (Azure Linux)

```sh
uname -m          # expect x86_64
uname -r          # e.g. 6.6.139.1-1.azl3 — note this exact version
sudo tdnf install -y kernel-tools     # installs the matching perf
perf --version    # should print perf version == uname -r
```

`tdnf` resolves `kernel-tools` to the version matching the running
kernel; if it picks an older build, the mismatch usually still records
fine, but prefer the exact match. Confirm sampling is allowed:

```sh
cat /proc/sys/kernel/perf_event_paranoid   # 0 (or -1) = full user+kernel
cat /proc/sys/kernel/kptr_restrict         # 0 = kernel symbols visible
# If paranoid > 1:  sudo sysctl kernel.perf_event_paranoid=0
```

## Step 2 — build wamr/wamrc (ReleaseFast, isolated cache)

The bench/perf convention is **ReleaseFast** (a Debug host inflates
host-side symbols; the generated AOT code is identical either way). Per
#374, isolate the Zig cache per worktree.

```sh
cd <wamr>
unset ZIG_LOCAL_CACHE_DIR
export ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache"
zig build -Doptimize=ReleaseFast
./zig-out/bin/wamr version    # expect: optimize ReleaseFast
```

## Step 3 — precompile with the spill metric (compile-time ranking)

Precompile the component once with the #809/#810 spill metric on. This
ranks functions by **emitted reloads** (`spill_ld` = real traffic) and is
the compile-time half of the picture.

```sh
WAMR_AOT_SPILL_METRIC=1 WAMR_AOT_SPILL_METRIC_MIN_SPILLS=1 \
WAMR_AOT_CODEGEN_TIMING=1 WAMR_AOT_CODEGEN_TIMING_THRESHOLD_MS=0 \
  ./zig-out/bin/wamrc compile-component \
  -o /work/perf/comp.cwasm.json <component>.wasm 2> spill.log
```

Rank spillers (the `spill_ld` field is `=`-delimited token 12, but it is
followed by more text, so sort with awk, not `sort -t= -k12`):

```sh
awk '/\[aot-spill-metric\]/{for(i=1;i<=NF;i++){n=index($i,"=");
  if(n>0)f[substr($i,1,n-1)]=substr($i,n+1)}
  printf "%d\tmod=%s lfunc=%s insts=%s spilled_vregs=%s spill_st=%s\n",
  f["spill_ld"],f["mod"],f["local_func"],f["insts"],f["spilled_vregs"],f["spill_st"]}' \
  spill.log | sort -k1,1 -nr | head -15
```

Note the per-core `<stem>.<N>.cwasm` files written next to the manifest —
the biggest one is the hot core's text and you need it in Step 5.

### Emit sound frame-origin metadata for a selected x86 function

After the first profile identifies the runtime-hot core/function, recompile
that selection with the versioned sidecar enabled (or enable it before a
fresh perf capture). The diagnostic does not change native code or default
logging. Its parent directory must already exist:

```sh
mkdir -p /work/perf/frame
WAMR_AOT_FRAME_ATTRIBUTION=/work/perf/frame/keyvault \
WAMR_AOT_FRAME_ATTRIBUTION_MODULE=4 \
WAMR_AOT_FRAME_ATTRIBUTION_FUNC=6145 \
WAMR_AOT_SPILL_METRIC=1 \
WAMR_AOT_SPILL_METRIC_MODULE=4 \
WAMR_AOT_SPILL_METRIC_FUNC=6145 \
  ./zig-out/bin/wamrc compile-component \
  -o /work/perf/comp.cwasm.json <component>.wasm
```

This writes
`/work/perf/frame/keyvault.mod4.func6145.json`. Each compiler-emitted
`rbp`/`rsp` frame access (including fixed prologue/epilogue pushes and pops)
has a half-open function-relative native byte range, signed displacement,
width, load/store direction, and one of:

- `allocator_spill`
- `wasm_local_or_phi` (the lowered IR cannot soundly distinguish the two)
- `explicit_frame_storage`
- `fixed_runtime_frame_state`
- `unknown`

Allocator values additionally carry physical slot/frame offset, vreg,
defining IR opcode + stable source class, live range, pre-emission IR use/def
counts, exact emitted reload/store counts, and current rematerialization
eligibility. If future allocation reuses a slot, an access names a vreg only
when its emitted native range and owning IR instruction prove the identity;
otherwise it is explicitly ambiguous. Inline `br_table` data ranges are also
listed so objdump is restarted after data instead of silently decoding jump
table bytes as instructions. A selected function bypasses codegen-cache reuse
for that compile so metadata is regenerated from the exact current IR/emission;
unselected functions may still reuse their cache entries.

### Core wasm modules (e.g. CoreMark, #393)

For a **single core module** (not a component) — CoreMark, a microbench,
any `.wasm` core — use `wamrc compile` instead of `compile-component`,
and run it with `wamr run <m>.cwasm` directly (the CLI runs `.cwasm`
core modules). Everything else (Steps 4–6) is identical; pass that one
`.cwasm` to `aot_jit_attr.py` — its function indices are module-local.

```sh
./zig-out/bin/wamrc compile tests/benchmarks/coremark/coremark_wasi.wasm \
  -o /work/cm/coremark.cwasm
perf record -g --call-graph=dwarf -F 999 -o /work/cm/cm.perf -- \
  ./zig-out/bin/wamr run /work/cm/coremark.cwasm
python3 .github/skills/aot-perf-profile/aot_jit_attr.py \
  --perf /work/cm/cm.perf --cwasm /work/cm/coremark.cwasm --func 10
```

(Spill ranking from Step 3 still works on a core module: `wamrc compile`
honours `WAMR_AOT_SPILL_METRIC=1`; there's just one module, `mod=0`.)

## Step 4 — `perf record` the precompiled run (runtime, the gate)

Profile **load+execute only** (precompiled — no compile in the loop). The
`wamr` CLI uses `--map-dir HOST::GUEST` preopens and auto-discovers a
sibling `<input>.cwasm.json`, or takes `--precompiled-manifest <path>`.
For the keyvault repro, the azure-sdk-for-zig `run.sh` is wasmtime-only;
this skill ships `run_wamr_keyvault.sh` that mirrors its WASI preopens
against the `wamr` binary (paths overridable via env).

```sh
SKILL=.github/skills/aot-perf-profile
perf record -g --call-graph=dwarf -F 999 -o wamr.perf -- \
  env WAMR_BIN=$PWD/zig-out/bin/wamr MANIFEST=/work/perf/comp.cwasm.json \
      OUT=/work/perf/out $SKILL/run_wamr_keyvault.sh
# ...or for any component:
perf record -g --call-graph=dwarf -F 999 -o wamr.perf -- <wamr run cmd>
perf report -i wamr.perf --stdio --sort=dso | head   # sanity: ~98% [JIT]
```

Expect ~98% in `[JIT]` (generated code) and <2% in the `wamr`
host/libc/kernel. That alone tells you **host helpers (memcpy / trap_oob
/ dispatch trampolines) are immaterial** — the cost is in generated code.

## Step 5 — de-anonymize `[JIT]` (per-function + instruction mix)

`perf` cannot symbolize the anonymous JIT region, and **DWARF cannot
unwind it** (the generated code has no CFI), so `perf script` callchains
are mostly empty — do **not** rely on them. Use the per-address **self**
sample counts from `perf report` instead. The helper script does all of
this: it parses `func_offsets[]` from the `.cwasm` function section,
finds the text mmap base in the perf data (by matching the core's text
size), buckets IPs per `local_func`, and classifies one function's
instruction mix.

```sh
SKILL=.github/skills/aot-perf-profile
# Per-function runtime ranking (which function is actually hot):
python3 $SKILL/aot_jit_attr.py --perf wamr.perf --cwasm comp.4.cwasm

# Add --func <N> to classify the hot function's instruction mix:
python3 $SKILL/aot_jit_attr.py --perf wamr.perf --cwasm comp.4.cwasm --func 6145

# Machine-readable output + a minimum useful sample gate:
python3 $SKILL/aot_jit_attr.py --perf wamr.perf --cwasm comp.4.cwasm \
  --func 6145 \
  --frame-metadata /work/perf/frame/keyvault.mod4.func6145.json \
  --min-samples 5000 --json-out attribution.json
```

It prints: total samples, % of run in this core, top functions by
self-samples, and for `--func` the class breakdown plus the 20 hottest
instructions. With `--frame-metadata`, it also ranks allocator contributors by
slot/vreg/source, reports static and sampled frame-attribution coverage and
unknowns, and requires exact reconciliation between emitted allocator
load/store records and the sidecar's `WAMR_AOT_SPILL_METRIC` totals.
On x86, `spill_ld`/`spill_st` are now emitter-traced totals (so folded
constant operands and fully suppressed constant defs no longer inflate the
metric); the allocator-value records retain their IR use/def counts to make
that difference inspectable. AArch64's standalone spill metric remains the
pre-emission IR estimate.

The tool fails closed on an incompatible AOT/metadata schema, stale full-core
text hash or function native-code hash, malformed or overlapping native
ranges, ambiguous mmap selection, or reconciliation mismatch. The full text
hash binds identical-looking functions to the correct component core.
Direct-call rel32 bytes are the only normalized relocations in the function
hash and are listed explicitly in the sidecar.
Without metadata, frame moves are reported as **unattributed frame traffic**,
not guessed to be spills.

Size matching must identify exactly one mapping. For multiple equal-size
cores, pass `--base 0x...` explicitly; get candidate bases from
`perf script -i wamr.perf --show-mmap-events | grep '//anon' | grep -E 'r[w-]xp'`.

Static sidecar/cwasm validation (no perf needed) is useful before a long run:

```sh
python3 $SKILL/aot_jit_attr.py \
  --cwasm /work/perf/comp.4.cwasm --func 6145 \
  --frame-metadata /work/perf/frame/keyvault.mod4.func6145.json \
  --validate-frame-metadata
```

The tracked `tests/benchmarks/frame_attribution/frame_origins.wasm` fixture is
the small end-to-end smoke module used by `tests/test_aot_jit_attr.py`; it
contains both lowered-local frame traffic and allocator spills across a call.

## Step 6 (optional) — register-supply sensitivity sweep

To decide whether the hot function is **register-supply-bound** vs.
**spill-quality-bound** without writing real codegen: temporarily add
GPRs to `x86_64_alloc_regs` / `x86_64_callee_saved_indices` in
`src/compiler/codegen/x86_64/compile.zig`, rebuild wamrc, re-run Step 3,
and diff the hot fn's `spill_ld`. **The metric is computed from the
allocation, so it is valid even though the emitted code is wrong** — just
do not *run* that build. (Worked example: 8→13 GPRs cut the keyvault hot
fn's `spill_ld` only 1.8% → supply is not the constraint.) Revert after.

## Worked example (#798 keyvault, captured 2026-06)

| metric | value |
|---|---|
| run | 21.4 s, `[JIT]` = 98.2% of cycles |
| hot fn | `mod4 local_func=6145` = **75%** of run (SpiderMonkey interp loop) |
| historical hot-fn heuristic | rbp loads 34.6%, rbp stores 17.4% → **frame traffic ~52%** |
| bounds-check | 2.4% · dispatch ~0.3% · linear-mem 1.5% |
| compile-time #1 spiller | `local_func=11396` (`spill_ld`=514,965) — **runtime-COLD (0 samples)** |

These 2026-06 frame percentages predate the origin sidecar and must not be
read as proof that every frame move was an allocator spill. Re-profile with
`--frame-metadata` for durable origin attribution. The biggest *compile-time*
spiller was never executed; the hot
function was the **#3** spiller. Always confirm the runtime-hot function
with Step 5 before optimizing the compile-time "monster".

## Worked example (#393 CoreMark, captured 2026-06)

Same tooling on the core module `coremark_wasi.wasm` (`wamr run`, 16.5 s):

| metric | value |
|---|---|
| `[JIT]` | 99.9% of cycles |
| hot fns | `local_func` 10 / 3 / 7 ≈ **30% / 28% / 28%** (list / matrix / state) |
| historical hot-fn heuristic | **reg-reg mov 15–18%** each, ALU 5–7%, rbp traffic 3–4% |
| dominant instrs | wasm i32 zero-extends (`mov esi,esi`, `mov edx,edx`, …) + a `br_table` (`add r10,r11; jmp r10`) |

Lesson: CoreMark is **not** spill-bound (confirms #393/#524 — its hot
loops aren't register-pressure-bound); the cost is **redundant i32
zero-extension `mov eXX,eXX`** + branch/dispatch — a different lever
(zero-ext elimination / coalescing) than keyvault's spills. The skill
distinguishes the two workload classes directly.

## Anti-patterns

| Anti-pattern | Why it misleads | Do instead |
|---|---|---|
| Reading `perf report` symbols and stopping at `98% [JIT]` | It is one anonymous bucket; tells you nothing per-function | Run Step 5 to map IPs → `local_func` |
| Using `perf script -F ip` callchains | DWARF can't unwind the no-CFI JIT → ~empty stacks | Use per-address **self** counts (`perf report -g none -n`) |
| Optimizing the largest compile-time function | It may be runtime-cold | Rank by **runtime** self-samples (Step 5) first |
| `sort -t= -k12` on the spill log | `spill_ld`'s field has trailing text → lexical sort | Parse with awk (Step 3) |
| Debug build for the runtime profile | Inflates host symbols, not representative | Build ReleaseFast (Step 2) |
| Installing a random `perf` | Azure Linux ships it in `kernel-tools` keyed to the kernel | `sudo tdnf install kernel-tools` (Step 1) |

## References

- Issue #798 — the runtime-gap tracker + lever list this profiles for.
- Issue #808 / PRs #809, #810 — the `WAMR_AOT_SPILL_METRIC*` diagnostic
  (`src/compiler/codegen/timing.zig:printSpill`,
  `src/compiler/ir/passes.zig:spillMetricOptionsFromEnv`).
- `src/compiler/codegen/frame_attribution.zig` and x86
  `compile.zig`/`emit.zig` — schema, slot/source metadata, and exact emitted
  native ranges behind `WAMR_AOT_FRAME_ATTRIBUTION*`.
- `src/runtime/aot/loader.zig:parseFunctionSection` — the `.cwasm`
  function section (`type=3`: count then `(offset:u32, type_idx:u32)`)
  that `aot_jit_attr.py` parses for `func_offsets[]`. Text section is
  `type=2`; magic `\0aot` (`0x746f6100`), version 9.
- `src/runtime/aot/runtime.zig` — the `local_func[N]+0x..` trap
  symbolizer that uses the same `func_offsets`; r15 mmap'd RX at
  `runtime.zig` `mprotect`.
- `src/compiler/codegen/x86_64/compile.zig` — `x86_64_alloc_regs`
  (8 allocatable GPRs; `rbx`=vmctx #465, `r15`=memory_base #466) for the
  Step 6 sweep.

## Companion skills

- `aot-diff-debug` / `aot-pass-bisect` — for AOT *correctness* bugs.
- `nvme-worktree` — isolate the build on `/work` NVMe for Step 2.
