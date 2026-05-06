# Local OSS-Fuzz packaging

This directory packages the high-priority Zig fuzz harnesses
(`fuzz-loader`, `fuzz-component-loader`, `fuzz-canon`) into the
[OSS-Fuzz `projects/<name>/`][oss-fuzz-projects] layout so a maintainer
can run libFuzzer-driven, coverage-guided fuzzing locally.

[oss-fuzz-projects]: https://github.com/google/oss-fuzz/tree/master/projects

## Scope

This packaging is **local-only**. There is no upstream OSS-Fuzz
`projects/wamr/` submission and none is planned in this PR. The intent
is to keep the per-input shim and Docker layout in-repository so:

- maintainers can reproduce the libFuzzer binaries locally;
- the fuzz harness one-input contract stays exercised under
  coverage-guided mutation;
- if upstream submission is reconsidered later, the wiring already
  exists.

See `tests/fuzz/OSS_FUZZ.md` for the readiness analysis.

## Layout

| File | Role |
| --- | --- |
| `Dockerfile` | Mirrors `gcr.io/oss-fuzz-base/base-builder` and pins Zig 0.16.0 with a checksum. |
| `build.sh` | Runs `zig build fuzz-oss` and links the resulting `libfuzz-oss-*.a` archives against `$LIB_FUZZING_ENGINE` to produce libFuzzer binaries. Packages seed corpora as `*_seed_corpus.zip`. |

The shim sources live next to the CLI harnesses:

```
src/tests/fuzz/
    loader.zig                # CLI corpus replay + pub fn runOnce
    oss_loader.zig            # exports LLVMFuzzerTestOneInput
    component_loader.zig
    oss_component_loader.zig
    canon.zig
    oss_canon.zig
```

## Local build (no Docker)

The Zig side of the integration (the static archives) is reproducible
without Docker:

```sh
zig build fuzz-oss -Doptimize=ReleaseSafe
ls zig-out/lib/libfuzz-oss-*.a
nm --defined-only zig-out/lib/libfuzz-oss-loader.a | grep LLVMFuzzer
```

Each archive defines `LLVMFuzzerTestOneInput`. To produce a runnable
libFuzzer binary, link against a libFuzzer driver, for example via
clang:

```sh
clang++ \
    -Wl,--whole-archive zig-out/lib/libfuzz-oss-loader.a -Wl,--no-whole-archive \
    -fsanitize=fuzzer \
    -o /tmp/fuzz-oss-loader
/tmp/fuzz-oss-loader -runs=1000 tests/malformed/fuzz
```

This was verified on aarch64-linux with the upstream
`LLVM-22.1.5-Linux-ARM64` toolchain (provides
`libclang_rt.fuzzer-aarch64.a`); 2000-run smoke loops on all three
shims completed cleanly. libFuzzer prints
`WARNING: no interesting inputs were found so far. Is the code
instrumented for coverage?` because the Zig static archive does not
ship SanitizerCoverage instrumentation — see the Limitations section
in `tests/fuzz/OSS_FUZZ.md`.

## Local Docker build (OSS-Fuzz helper)

To reproduce the full OSS-Fuzz packaging locally, follow the upstream
[helper documentation][oss-fuzz-helper]:

```sh
git clone https://github.com/google/oss-fuzz.git /tmp/oss-fuzz
mkdir -p /tmp/oss-fuzz/projects/wamr
cp oss-fuzz/Dockerfile /tmp/oss-fuzz/projects/wamr/
cp oss-fuzz/build.sh /tmp/oss-fuzz/projects/wamr/
echo 'language: c++' > /tmp/oss-fuzz/projects/wamr/project.yaml

cd /tmp/oss-fuzz
python infra/helper.py build_image wamr
python infra/helper.py build_fuzzers --sanitizer address wamr
python infra/helper.py run_fuzzer wamr fuzz-oss-loader -- -runs=10000
```

Anything written under `/tmp/oss-fuzz/build/out/wamr/` is the local
build artifact. **Do not** push these files to the upstream OSS-Fuzz
repository as part of this work.

[oss-fuzz-helper]: https://google.github.io/oss-fuzz/getting-started/new-project-guide/

## Triage

A libFuzzer crash drops a reproducer next to the binary:

```
fuzz-oss-loader -runs=10000 corpus/
# crash-<sha1> written on first crash
```

Reproduce against the in-repo CLI harness for a stable trace:

```sh
mkdir -p /tmp/wamr-repro
cp crash-<sha1> /tmp/wamr-repro/
./zig-out/bin/fuzz-loader \
    --corpus /tmp/wamr-repro \
    --crashes /tmp/wamr-fuzz-crashes \
    --duration 5
```

Follow the public/private split documented in `tests/fuzz/README.md`
and `tests/fuzz/regression/README.md`. Sensitive reproducers go
through the private vulnerability flow described in `SECURITY.md`.
