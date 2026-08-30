# Keyvault SpiderMonkey/TCGC AOT harness

`scripts/bench_keyvault.py` compares precompiled WAMR and Wasmtime execution
of the keyvault codegen workload from #798. It is deliberately an orchestrator,
not an installer: it never clones repositories, downloads runtimes, installs
`perf`, builds the component, or guesses a floating version.

## Reproduction contract

Create a local manifest from `manifest.example.json`. Every placeholder must be
replaced with:

- full 40-character revisions for the WAMR, azure-sdk-for-zig, and
  azure-rest-api-specs checkouts used to produce the workload;
- SHA-256 values for `wamr`, `wamrc`, Wasmtime, the composed component, and the
  generated `stdlib-preopens.txt`;
- the harness tree SHA-256 for `/spec` and every stdlib preopen;
- the complete guest environment (an empty object means no workload-specific
  variables);
- the exact expected `got N bytes back from tcgc.compile` byte count.

The example records the known #743 component identity
`fe08b02f...d9da8c92` and SDK revision
`19e06fc47073bcb9a6c461e3a0bf5298dc49cc60`. It is not a runnable lockfile:
machine-specific paths, locally built tool hashes, stdlib tree hashes, and the
exact spec cohort still need to be supplied. Do not substitute current branch
tips and call the result the #798 baseline.

Compute hashes with the harness's canonical algorithm:

```bash
python3 scripts/bench_keyvault.py --hash /path/to/file
python3 scripts/bench_keyvault.py --hash /path/to/tree
```

Directory hashes cover the sorted relative path, exact size, and SHA-256 of
every regular file. Symlinks are rejected. The `preopens_file` mappings must
exactly match all configured mounts except `/spec`; the output mount `/out` is
created separately for every run.

Validate prerequisites without compiling or executing:

```bash
python3 scripts/bench_keyvault.py \
  --manifest /path/to/keyvault.lock.json \
  --validate-only
```

Validation is intentionally strict. Missing binaries/checkouts, shortened or
wrong revisions, tracked source modifications, placeholder hashes, changed
inputs, malformed preopens, or a tool whose version output does not contain the
pinned version are hard errors with the expected and actual identities.

## Authoritative timing

The harness currently targets the Linux x86_64 #798 cohort. Build WAMR
`ReleaseFast`, obtain a pinned Wasmtime binary (the repository CI currently
uses `v46.0.1`), build the component and TCGC stdlib, and prepare the pinned
spec checkout before invoking it:

```bash
python3 scripts/bench_keyvault.py \
  --manifest /path/to/keyvault.lock.json \
  --work-dir "$PWD/zig-out/keyvault-798" \
  --warmups 2 \
  --runs 7 \
  --report-json "$PWD/zig-out/keyvault-798/report.json" \
  --report-markdown "$PWD/zig-out/keyvault-798/report.md"
```

`--runs` cannot be less than five. WAMR `compile-component` and Wasmtime
`compile` happen once, before warmups, and their durations are reported but
excluded from measured execution. Measured runs alternate WAMR-first and
Wasmtime-first ordering. The host subprocess environment is reduced to
`PATH`, `HOME`, `LANG=C`, `LC_ALL=C`, and `TZ=UTC`, preventing ambient WAMR or
Wasmtime tuning variables from changing the cohort. Both guests receive only
the manifest environment plus a fixed harness marker, which also prevents the
WAMR component CLI from inheriting the caller's environment. The report
includes:

- host/kernel/CPU details;
- exact tool versions, binary/input hashes, and source revisions;
- every timing sample plus mean, median, minimum, maximum, and range;
- the requested **Wasmtime-time/WAMR-time** ratio;
- every generated path, size, and SHA-256 plus a canonical tree hash.

Every warmup and measured run must emit exactly one TCGC byte marker, equal the
manifest's pinned value, and generate an identical output tree. Consequently,
the historical 58187-vs-58188 response-size drift is a failure even if the
generated files happen to match.

No report is produced if validation, execution, or equivalence fails. In
particular, absent external workload assets cannot produce a success-shaped or
fabricated baseline.

## Perf capture and attribution

Perf is opt-in, never an automatic fallback:

```bash
python3 scripts/bench_keyvault.py \
  --manifest /path/to/keyvault.lock.json \
  --work-dir "$PWD/zig-out/keyvault-798" \
  --warmups 2 --runs 7 --profile
```

`--profile` requires Linux x86_64, `perf`, `objdump`, sampling permissions, the
configured WAMR core index, at least `perf.min_samples`, and at least
`perf.min_attribution_coverage_pct`. Any unmet requirement fails the command.
The profiled run is checked against the same response byte count and generated
output tree before `aot_jit_attr.py` maps self samples to `local_func` indices.
The JSON/Markdown reports record total samples, attributed samples, coverage,
top functions, and (when `hot_func` is configured) instruction-class data.

For Azure Linux setup and interpretation details, see
`.github/skills/aot-perf-profile/SKILL.md`.

## Compare the same hot wasm function in WAMR and Wasmtime

`scripts/compare_hot_function.py` turns the WAMR profile into a strict,
archiveable cross-engine code comparison. It does **not** equate WAMR's
`local_func=N` with Wasmtime's engine-local module/function labels.

First run the benchmark with `--profile`. Then extract the raw core modules
from the exact pinned component without re-encoding them. For example,
`wasm-tools component unbundle` can extract embedded modules; composed
components may need recursive unbundling. Hash candidates with
`bench_keyvault.py --hash` and select the unique file whose SHA-256 equals the
chosen WAMR manifest entry's `core_sha256`. Ambiguous or absent matches are a
hard error, not a reason to use the core's ordinal.

Copy `hot-comparison.example.json`, pin its report and core hashes, and set:

- `wamr.cwasm` to the exact artifact named by the version-2 WAMR manifest;
- `wamr.local_func` to the function classified in `perf.attribution`;
- `function.wasm_index` to the full wasm index (imports first), or set it to
  `null` and provide a unique exact name-section value.

The capture validates that
`wasm_index == imported_function_count + WAMR local_func`, validates any
name-section value, and rejects duplicate-name selection. It also validates
the component, raw-core, WAMR cwasm, attribution, benchmark report, and
Wasmtime binary hashes.

```bash
python3 scripts/compare_hot_function.py capture \
  --config tests/benchmarks/keyvault/hot-comparison.local.json \
  --work-dir "$PWD/zig-out/keyvault-798/hot-comparison" \
  --output "$PWD/zig-out/keyvault-798/hot-capture.json"

python3 scripts/compare_hot_function.py report \
  --capture "$PWD/zig-out/keyvault-798/hot-capture.json" \
  --report-json "$PWD/zig-out/keyvault-798/hot-comparison.json" \
  --report-markdown "$PWD/zig-out/keyvault-798/hot-comparison.md"
```

For static Wasmtime code, the capture command takes the exact hash-matched raw
core and reruns `wasmtime compile` with the options recorded in the benchmark
report. It cross-checks the precompiled artifact's
`wasmtime objdump --addresses --bytes` symbol prefix against the structured
`window.ASM` payload emitted by `wasmtime explore` (Wasmtime may place cold
blocks beyond the ELF symbol's displayed extent), then uses wasm address
mappings to exclude trailing inline data from instruction counts. The full
explorer function extent remains the reported code size. This is deliberate:

- `wasmtime explore` recompiles the input and emits human-oriented HTML rather
  than a standalone JSON file, so the pinned payload parser fails closed if
  its structured `window.ASM` data is absent or malformed;
- `wasmtime objdump` identifies a component function as
  `wasm[N]::function[M]`, where `M` is the full wasm function index, but `N`
  is a Wasmtime compilation-local module ordinal and the output does not carry
  the raw core SHA-256;
- compiling the exact extracted core as standalone `wasm[0]` therefore gives a
  sound module-hash + wasm-index comparison without guessing that ordinal.

`bench_keyvault.py` currently collects WAMR dynamic samples only. Wasmtime
dynamic columns remain explicitly unavailable unless `wasmtime.profile`
points to a pinned JSON document with this shape:

```json
{
  "schema_version": 1,
  "kind": "wasm-hot-function-profile",
  "engine": "wasmtime",
  "mapping_verified": true,
  "mapping_evidence": "how the component-local symbol was tied to this raw core",
  "module_sha256": "<same exact raw core SHA-256>",
  "wasm_function_index": 0,
  "function_name": null,
  "total_samples": 10000,
  "function_samples": 7000,
  "metrics": {
    "frame_loads": { "status": "measured", "samples": 2000 },
    "frame_stores": { "status": "unavailable", "reason": "not classified" }
  }
}
```

The report separates static instruction counts from perf self samples and
reports native instruction count, code bytes, recoverable fixed frame bytes,
move-form frame loads/stores, register moves, `lea` address generation, branches,
indirect jumps/calls, and calls for both engines. Fixed frame size is marked
unavailable when dynamic alignment or a non-constant prologue adjustment makes
recovery unsound. Later temporary outgoing-call stack changes are not included
in the fixed frame. WAMR's emitted x86_64 `br_table` sequences are validated
against their exact dispatch pattern and target count; their inline 32-bit jump
table data is masked before disassembly and excluded from instruction counts.
Static categories can overlap.

For each instruction class, conservative theoretical run-wide headroom is:

```
WAMR hot-function measured share
× min(WAMR dynamic sample share, WAMR static instruction share)
× static Wasmtime reduction fraction
```

The values are not additive. An optimization is recommended only when at
least two exact dynamic+static differences each reach 5% run-wide headroom;
otherwise the report says that no lever clears the gate. Missing metrics and
static-only upper bounds never clear it.

The repository does not include the external keyvault component/spec/stdlib
cohort. Controlled fixtures test the full capture/report path, but they are
not real workload findings and must not be presented as such.
