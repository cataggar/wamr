# Fuzzing security-sensitive boundaries

This directory documents the Zig fuzz harnesses in `src/tests/fuzz`. The
harnesses are bounded corpus-replay CLIs intended for CI smoke runs, scheduled
security fuzzing, and local crash reproduction. They complement the sandbox
review checklist in `SECURITY_AUDIT.md`; they are not a replacement for targeted
unit tests or private vulnerability triage.

## Goals

Treat all Wasm, component-model binaries, exported arguments, and guest
linear-memory contents as attacker-controlled. The fuzz harnesses should expose:

- loader panics, safety-checked integer overflows, and malformed-binary crashes;
- compiler or codegen panics on hostile but bounded inputs;
- runtime trap handling paths that terminate the host process unexpectedly;
- component and WASI boundary bugs that corrupt resource or memory isolation.

Typed loader, validation, compile, or runtime errors are expected outcomes for
malformed inputs. A panic, segfault, safety-check failure, unexplained process
abort, or small-input OOM is a bug.

## Targets

| Binary | Boundary | Oracle |
| --- | --- | --- |
| `fuzz-loader` | Core Wasm loader and validation | A valid module or typed loader error is OK. A panic, abort, or safety-check failure is a bug. |
| `fuzz-component-loader` | Component-model binary loader | A valid component or typed component loader error is OK. A panic, abort, or safety-check failure is a bug. |
| `fuzz-interp` | Interpreter loader and validation scaffold | Loader/validation errors are OK. Full bounded export invocation is tracked as follow-up work. |
| `fuzz-aot` | AOT compile, AOT loader, and instantiate path | Compile/instantiate errors are OK. The harness deliberately sets `invoke_start = false` and does not execute attacker-supplied start functions. |
| `fuzz-diff` | Interpreter load plus AOT compile/instantiate scaffold | Pipeline errors are OK. Result comparison for supported exports is tracked as follow-up work. |

## Local use

Build all harnesses:

```sh
zig build fuzz -Doptimize=ReleaseSafe
```

Run a short smoke over the malformed corpus:

```sh
rm -rf /tmp/wamr-fuzz-crashes
mkdir -p /tmp/wamr-fuzz-crashes
./zig-out/bin/fuzz-loader --corpus tests/malformed/fuzz --crashes /tmp/wamr-fuzz-crashes --duration 5
./zig-out/bin/fuzz-component-loader --corpus tests/malformed/fuzz --crashes /tmp/wamr-fuzz-crashes --duration 5
```

For component-loader smoke with at least one valid component seed:

```sh
rm -rf /tmp/wamr-component-corpus /tmp/wamr-fuzz-crashes
mkdir -p /tmp/wamr-component-corpus /tmp/wamr-fuzz-crashes
printf '\000asm\015\000\001\000' > /tmp/wamr-component-corpus/minimal-component.wasm
./zig-out/bin/fuzz-component-loader --corpus /tmp/wamr-component-corpus --crashes /tmp/wamr-fuzz-crashes --duration 5
```

The harnesses replay every `.wasm` file under `--corpus` until the duration
expires. They are not coverage-guided mutators by themselves; use them as stable
entry points for CI and future mutation/generation infrastructure.

## Crash artifacts and reproduction

Every harness writes the current input to `<crashes>/in-flight.wasm` before
calling the target and deletes it after a clean return. If the process aborts,
the file remains as a reproducer. Harnesses may also write named crashers such
as `<tag>-<hash>.wasm` for explicit oracle failures.

To reproduce a crash:

1. Download the `fuzz-crashes-<target>` artifact from the failed workflow.
2. Put `in-flight.wasm` or the named crasher in a clean corpus directory.
3. Re-run the same target locally with a short duration and `-Doptimize=ReleaseSafe`.
4. Minimize or redact the reproducer before making it public if it may disclose
   an unfixed security issue. Use the private reporting flow in `SECURITY.md`
   and `SECURITY_PROCESS.md` for sensitive payloads.

## CI workflow

`.github/workflows/fuzz.yml` runs on a daily schedule, on demand, and on PRs that
touch runtime/compiler/component/fuzz code. The workflow:

- builds all harnesses with `zig build fuzz -Doptimize=ReleaseSafe`;
- seeds per-target corpora from `tests/malformed/fuzz` and `tests/spec-json`;
- adds a generated minimal component seed for `fuzz-component-loader`;
- uploads `.fuzz-crashes` artifacts with 30-day retention;
- uploads per-target corpus artifacts with 7-day retention;
- fails the job when any crash artifact remains.

Manual runs can choose `all`, `loader`, `component-loader`, `interp`, `aot`, or
`diff` and can set a per-target duration.

## Current limitations and follow-ups

The current targets are intentionally conservative. Do not remove these limits
without adding bounded execution, subprocess isolation, or another equivalent
host-protection mechanism.

- `fuzz-interp` does not yet invoke exports (#245). A future target should instantiate
  safe modules, provide deterministic host imports, and use fuel/timeouts or
  subprocesses so valid infinite loops cannot hang CI.
- `fuzz-diff` does not yet compare typed interpreter and AOT results (#246). A future
  oracle should select supported nullary or bounded-argument scalar exports and
  handle platform-specific AOT traps outside the test runner.
- Direct WASI and component-adapter fuzzing needs a deterministic byte-command
  model for resource tables, paths, descriptors, sockets, HTTP streams, and
  guest-memory pointer/length pairs (#247).
- Corpus minimization should preserve a small checked-in seed set while storing
  larger evolving corpora as workflow artifacts (#248).
- OSS-Fuzz integration should be evaluated after the local targets are stable,
  resource usage is bounded, and generated corpora are useful (#249).
