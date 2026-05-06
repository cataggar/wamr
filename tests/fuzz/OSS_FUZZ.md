# OSS-Fuzz feasibility and local packaging

This document is the deliverable for #249 and the readiness gate for
#262. It evaluates whether to integrate this repository with
[OSS-Fuzz][oss-fuzz] and records the decision so future contributors
can revisit it without redoing the analysis.

[oss-fuzz]: https://github.com/google/oss-fuzz

## Summary

**Decision: ship local-only OSS-Fuzz packaging; do not submit upstream.**

Issue #262 listed four prerequisites for re-opening the OSS-Fuzz
question. The repository now matches the technical prerequisites:

1. #248 (corpus minimization and reducer workflow) is closed; reduced
   inputs land under `tests/fuzz/regression/<target>/` per
   `tests/fuzz/regression/README.md` and `scripts/fuzz_reduce.py`.
2. We accept the cost of maintaining a hand-rolled
   `LLVMFuzzerTestOneInput` shim and a pinned Zig toolchain.
   Zig's `std.testing.fuzz` is still moving and OSS-Fuzz's base image
   does not ship Zig as a first-class language; the shim path is the
   stable choice.
3. The high-priority deterministic harnesses (`fuzz-loader`,
   `fuzz-component-loader`, `fuzz-canon`) have been running on the
   `.github/workflows/fuzz.yml` daily schedule without crashers.
4. The repository's `SECURITY.md` and `SECURITY_PROCESS.md`
   intentionally do not promise a formal SLA, embargo, or CVE pipeline.
   Submitting this project upstream to OSS-Fuzz would import
   ClusterFuzz's default 90-day disclosure window and create downstream
   expectations the maintainer has not agreed to. **For that reason
   this work is local-only and the upstream submission step is
   explicitly not in scope.**

Local-only deliverables in this repository:

- `src/tests/fuzz/oss_loader.zig`, `oss_component_loader.zig`,
  `oss_canon.zig` — Zig shim sources that export
  `LLVMFuzzerTestOneInput` and call the same `runOnce` functions the
  CLI corpus-replay harnesses use.
- `build.zig` `fuzz-oss` step — emits
  `zig-out/lib/libfuzz-oss-{loader,component-loader,canon}.a` static
  archives.
- `oss-fuzz/Dockerfile`, `oss-fuzz/build.sh`, `oss-fuzz/README.md` —
  package the static archives into libFuzzer binaries by linking
  against `$LIB_FUZZING_ENGINE` in an OSS-Fuzz-style local build.

OSS-Fuzz integration as a continuous service is **not** wired in.
There is no `projects/wamr/` PR upstream and the `oss-fuzz/` directory
must not be pushed to `google/oss-fuzz`. Re-evaluate that step only if
a maintainer agrees to track ClusterFuzz disclosures and accept the
upstream operational expectations.

## Per-harness assessment

| Harness                 | Entry point                              | Determinism | Resource bound                  | Coverage-fuzz priority | OSS-Fuzz shim |
| ----------------------- | ---------------------------------------- | ----------- | ------------------------------- | ---------------------- | ------------- |
| `fuzz-loader`           | `src/tests/fuzz/loader.zig`              | Yes         | 16MB input cap, no I/O          | High — small inputs, big surface, no I/O | Yes (`oss_loader.zig`) |
| `fuzz-component-loader` | `src/tests/fuzz/component_loader.zig`    | Yes         | 16MB input cap, no I/O          | High — same as core loader | Yes (`oss_component_loader.zig`) |
| `fuzz-canon`            | `src/tests/fuzz/canon.zig`               | Yes         | 64KB memory cap, arena per iter | High — pointer/length surface, no I/O | Yes (`oss_canon.zig`) |
| `fuzz-interp`           | `src/tests/fuzz/interp.zig` + `invoke.zig` | Yes (with fuel) | Per-export fuel cap (default 100k) | Medium — needs fuel piped from libFuzzer entry | No |
| `fuzz-aot`              | `src/tests/fuzz/aot.zig`                 | Yes (no execute) | `invoke_start = false`         | Medium — heavier per-iteration AOT compile | No |
| `fuzz-diff`             | `src/tests/fuzz/diff.zig` + `invoke.zig` | Yes (with fuel + safe filter) | Static AOT-safe straight-line subset only | Low — limited by Linux/AArch64 native trap host-termination; safe subset is small | No |
| `fuzz-wasi`             | `src/tests/fuzz/wasi.zig`                | Per-iteration adapter rebuild | Per-process temp preopen | Medium — WASI sandbox surface | No |

The "high" priority harnesses are good candidates for coverage-guided
mutation. They have small, fast iterations; no host file-system or
network effects; and a deterministic oracle (typed errors OK,
panic/UB/abort is a bug). The medium and low priority harnesses would
require additional hardening before they pay back the OSS-Fuzz
integration cost; their shims are intentionally not built.

## Zig + libFuzzer interop

OSS-Fuzz expects a fuzz target compiled as a binary that links the
libFuzzer runtime and exports the standard entry point:

```c
extern int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);
```

Two viable paths existed:

1. **Hand-rolled C-ABI shim (chosen).** `src/tests/fuzz/oss_*.zig`
   exposes:

   ```zig
   export fn LLVMFuzzerTestOneInput(data: [*]const u8, size: usize) c_int {
       // reset shared arena, call harness.runOnce(arena, data[0..size])
       return 0;
   }
   ```

   The Zig shims are compiled into static archives by `zig build
   fuzz-oss`. `oss-fuzz/build.sh` links each archive with
   `$LIB_FUZZING_ENGINE` (clang `-fsanitize=fuzzer`) using
   `-Wl,--whole-archive` so the linker keeps the exported symbol.
   This approach does not depend on Zig's coverage instrumentation;
   libFuzzer drives the shim as a black-box per-input function. See
   "Limitations" below.

2. **Use Zig's `std.testing.fuzz`** — Zig 0.16 has experimental
   fuzzing support but the API is still moving and OSS-Fuzz does not
   ship Zig in the base image. Tracking this would lock the build to
   a specific Zig release in OSS-Fuzz's base image. Deferred.

### Sanitizer notes

- Zig 0.16 ships its own runtime safety checks; combining with
  AddressSanitizer/MSAN/UBSAN under libFuzzer requires verifying that
  Zig's safety panic handler does not interfere with libFuzzer's crash
  detection. The default (ReleaseSafe) keeps Zig safety checks
  visible.
- Stack overflows inside Zig's interpreter (especially for hostile
  control flow) need a configured stack guard size; OSS-Fuzz already
  runs targets under ASan with a guard region.

## Build environment

`oss-fuzz/Dockerfile` mirrors `gcr.io/oss-fuzz-base/base-builder`,
installs a pinned Zig 0.16.0 with a SHA-256 verified tarball, and
clones this repository. `oss-fuzz/build.sh` runs:

- `zig build fuzz-oss -Doptimize=ReleaseSafe` to emit the static
  archives;
- `clang++ -Wl,--whole-archive lib*.a -Wl,--no-whole-archive
  $LIB_FUZZING_ENGINE -o $OUT/fuzz-oss-<name>` for each high-priority
  target;
- per-target `*_seed_corpus.zip` archives sourced from
  `tests/malformed/fuzz`, `tests/spec-json` (loader), generated
  minimal seeds (component-loader, canon), and any committed
  regression seeds in `tests/fuzz/regression/<target>/`.

Local builds without Docker are documented in `oss-fuzz/README.md`.

## Triage flow

If we ever proceed with continuous OSS-Fuzz coverage, the local
packaging is meant to plug in directly:

1. OSS-Fuzz reports a crash with a reduced reproducer.
2. The maintainer downloads the reproducer, runs the matching local
   CLI harness via `--corpus <dir>` to confirm.
3. If reproduction succeeds, follow the disclosure flow in
   `SECURITY.md` and `SECURITY_PROCESS.md`. ClusterFuzz's 90-day
   disclosure window would need to be reconciled with the
   repository's best-effort process before any upstream submission.
4. After fix, the minimized reproducer becomes a regression seed
   under `tests/fuzz/regression/<target>/` (per #248), so the in-repo
   harnesses keep covering the original input.

## Limitations

- **No Zig-side coverage instrumentation.** libFuzzer feedback is
  limited to whatever the C/C++ runtime sees plus crash detection;
  end-to-end smoke runs of the shims report
  `WARNING: no interesting inputs were found so far. Is the code
  instrumented for coverage?` — expected for this approach.
  Coverage-guided mutation will still help but not as effectively as
  a fully instrumented build.
- **Local Docker only.** The image is reproducible locally via the
  upstream OSS-Fuzz `helper.py` flow, but is not connected to
  ClusterFuzz.
- **Three high-priority targets only.** `fuzz-interp`, `fuzz-aot`,
  `fuzz-diff`, and `fuzz-wasi` keep their CLI-only form until each
  has a libFuzzer-friendly resource policy (fuel, subprocess
  isolation, or sandbox redo per-iteration).

## Re-evaluating upstream submission

Open a follow-up issue, not another PR, if any of the following
change:

- a maintainer agrees to track ClusterFuzz advisories and accept the
  90-day disclosure default;
- Zig becomes a first-class OSS-Fuzz language with stable `fuzz`
  instrumentation, removing the hand-rolled shim cost;
- coverage data from local libFuzzer runs shows the in-repo CLI
  workflow is corpus-bound and would benefit from continuous
  coverage-guided mutation.

Until then, the local packaging is the project's complete OSS-Fuzz
posture.
