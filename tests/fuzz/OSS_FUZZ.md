# OSS-Fuzz feasibility evaluation

This document is the deliverable for #249. It evaluates whether to integrate
this repository with [OSS-Fuzz][oss-fuzz] now, after the local fuzz harnesses
introduced in #222, #245, #246, and #247 stabilised, and records the
decision so future contributors can revisit it without redoing the analysis.

[oss-fuzz]: https://github.com/google/oss-fuzz

## Summary

**Decision: defer.**

The current corpus-replay harnesses already satisfy our near-term goal of
catching panics, safety-checked UB, and process aborts on attacker-controlled
binaries via scheduled CI runs. OSS-Fuzz adds genuine value (continuous
coverage-guided mutation, ClusterFuzz triage, public regressions corpus) but
the integration cost is non-trivial because:

- This repo is written in Zig 0.16. OSS-Fuzz first-class language support
  is C, C++, Rust, Go, Python, Java/JVM, JavaScript, and Swift. Zig is not
  on that list, so we must either ship a C-ABI shim or rely on Zig's own
  `std.testing.fuzz`, which is still in flux upstream.
- Each existing harness is a CLI corpus replay (`pub fn main`), not a
  libFuzzer-shaped `LLVMFuzzerTestOneInput` function. Adapting them is
  mostly mechanical but is real work.
- Several harnesses (`fuzz-aot`, `fuzz-diff`) have host-process trap
  limitations on Linux/AArch64 that already constrain in-repo coverage;
  those constraints carry over to OSS-Fuzz.

We will revisit this decision after:

1. #248 lands a documented minimization workflow so triage is repeatable;
2. Zig's `std.testing.fuzz` stabilises in a 0.x release we plan to track,
   or we accept the cost of writing a hand-rolled C-ABI shim;
3. We have at least one harness that has run for ≥1 month in CI without
   producing crashers, indicating the trivial bug surface is exhausted and
   coverage-guided mutation would actually find new bugs.

A follow-up issue captures the prerequisite list above; see the bottom of
this document.

## Per-harness assessment

| Harness                 | Entry point                              | Determinism | Resource bound                  | Coverage-fuzz priority |
| ----------------------- | ---------------------------------------- | ----------- | ------------------------------- | ---------------------- |
| `fuzz-loader`           | `src/tests/fuzz/loader.zig`              | Yes         | 16MB input cap, no I/O          | High — small inputs, big surface, no I/O |
| `fuzz-component-loader` | `src/tests/fuzz/component_loader.zig`    | Yes         | 16MB input cap, no I/O          | High — same as core loader |
| `fuzz-canon`            | `src/tests/fuzz/canon.zig`               | Yes         | 64KB memory cap, arena per iter | High — pointer/length surface, no I/O |
| `fuzz-interp`           | `src/tests/fuzz/interp.zig` + `invoke.zig` | Yes (with fuel) | Per-export fuel cap (default 100k) | Medium — needs fuel piped from libFuzzer entry |
| `fuzz-aot`              | `src/tests/fuzz/aot.zig`                 | Yes (no execute) | `invoke_start = false`         | Medium — heavier per-iteration AOT compile |
| `fuzz-diff`             | `src/tests/fuzz/diff.zig` + `invoke.zig` | Yes (with fuel + safe filter) | Static AOT-safe straight-line subset only | Low — limited by Linux/AArch64 native trap host-termination; safe subset is small |

The "high" priority harnesses are good candidates for a future
coverage-guided integration. They have small, fast iterations; no host
file-system or network effects; and a deterministic oracle (typed errors
OK, panic/UB/abort is a bug). The medium and low priority harnesses would
require additional hardening before they pay back the OSS-Fuzz integration
cost.

## Zig + libFuzzer interop

OSS-Fuzz expects a fuzz target compiled as a binary that links the
libFuzzer runtime and exports the standard entry point:

```c
extern int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);
```

Two viable paths exist:

1. **Hand-rolled C-ABI shim** — add a `fuzz_target_<name>.zig` that exposes:

   ```zig
   export fn LLVMFuzzerTestOneInput(data: [*]const u8, size: usize) c_int {
       const bytes = data[0..size];
       runOnce(global_allocator, bytes) catch {};
       return 0;
   }
   ```

   and compile it with `-fsanitize=fuzzer-no-link` plus the system
   libFuzzer runtime. This is conceptually simple but requires us to
   manage a global allocator (libFuzzer reuses the process per iteration)
   and to ensure no harness state leaks between calls.

2. **Use Zig's `std.testing.fuzz`** — Zig 0.16 has experimental fuzzing
   support but the API is still moving. OSS-Fuzz's Zig support, if any,
   would have to follow upstream Zig. Tracking this would lock our build
   to a specific Zig release in OSS-Fuzz's base image.

Either path adds a parallel fuzz build on top of the existing
`zig build fuzz` step. It does not replace the existing CLI harnesses.

### Sanitizer notes

- Zig 0.16 ships its own runtime safety checks; combining with
  AddressSanitizer/MSAN/UBSAN under libFuzzer requires verifying that
  Zig's safety panic handler does not interfere with libFuzzer's crash
  detection.
- Stack overflows inside Zig's interpreter (especially for hostile control
  flow) need a configured stack guard size; OSS-Fuzz already runs targets
  under ASan with a guard region.

## Build environment

OSS-Fuzz integration requires submitting two files to the OSS-Fuzz repo:

- `projects/wamr/Dockerfile` — base on `gcr.io/oss-fuzz-base/base-builder`,
  install Zig 0.16 (download tarball, verify checksum), install
  `wasm-tools` for seed generation, and `git clone` this repo.
- `projects/wamr/build.sh` — run `zig build fuzz -Doptimize=ReleaseSafe`
  and the libFuzzer-shim build for the high-priority harnesses, then copy
  binaries to `$OUT/`.

We also need an `oss-fuzz`-friendly seed corpus. The existing
`tests/malformed/fuzz/*.wasm` and the planned regression seeds from #248
can be packaged as the initial OSS-Fuzz seed corpus.

## Triage flow

If we proceed:

1. OSS-Fuzz reports a crash with a reduced reproducer.
2. The maintainer downloads the reproducer, runs the matching local
   harness via `--corpus <dir>` to confirm.
3. If reproduction succeeds, follow the disclosure flow in `SECURITY.md`
   and `SECURITY_PROCESS.md`. ClusterFuzz's 90-day disclosure window
   should be aligned with our SECURITY policy when filing the OSS-Fuzz
   project.
4. After fix, the minimized reproducer becomes a regression seed under
   `tests/fuzz/regression/<target>/` (per #248), so the in-repo harnesses
   keep covering the original input.

## Cost vs benefit

| Item                                               | Cost                          | Benefit                          |
| -------------------------------------------------- | ----------------------------- | -------------------------------- |
| C-ABI shim per high-priority harness               | ~100 lines + per-target build | Coverage-guided mutation         |
| OSS-Fuzz `Dockerfile` + `build.sh`                 | One-time, modest              | ClusterFuzz infra and triage     |
| Pin Zig version in OSS-Fuzz base                   | Ongoing maintenance           | Reproducible builds              |
| Address Zig + sanitizer interactions               | Real, possibly fragile        | Memory-safety bug detection     |
| Disclosure pipeline alignment with `SECURITY.md`   | Light                         | Coordinated public/private flow  |
| Long-running coverage corpus distribution          | Free (ClusterFuzz hosts it)   | Stable seeds beyond what fits in repo |

Today the in-repo harnesses already catch the fast wins. We have no
evidence yet that we are corpus-bound (CI fuzzing is finishing without
crashers on stable code paths). Until coverage-guided mutation would
plausibly find bugs the corpus replay misses, the maintenance cost is
greater than the benefit.

## Decision

Defer OSS-Fuzz integration. Re-evaluate after the prerequisites listed in
the Summary land. Tracked in the follow-up issue linked from this
document.

## Prerequisite list (for the future re-evaluation)

1. #248 corpus minimization and reducer workflow merged; reduced inputs
   landing as regression seeds.
2. Either Zig 0.16+ `std.testing.fuzz` stable enough to depend on in
   OSS-Fuzz's base image, **or** a written agreement that we maintain a
   hand-rolled `LLVMFuzzerTestOneInput` shim and a pinned Zig toolchain.
3. At least one high-priority harness has run on the daily CI schedule
   for ≥30 days with no crashers found, suggesting coverage-guided
   mutation would extend our reach.
4. Disclosure-window policy (90-day default in ClusterFuzz) reconciled
   with `SECURITY.md`/`SECURITY_PROCESS.md`.

When all four conditions are met, open a focused PR that:

- adds `fuzz_oss_<target>.zig` shim files for the high-priority harnesses;
- adds `oss-fuzz/Dockerfile` and `oss-fuzz/build.sh` mirroring the
  upstream OSS-Fuzz layout;
- submits the OSS-Fuzz `projects/wamr/` PR upstream.
