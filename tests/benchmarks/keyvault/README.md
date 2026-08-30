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
