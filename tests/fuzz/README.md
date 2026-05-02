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
| `fuzz-interp` | Interpreter load, instantiate, and bounded nullary export invocation | Loader/validation/instantiation errors, guest traps, fuel exhaustion, and unsupported signatures are OK. Panics, aborts, or result-stack shape mismatches are bugs. |
| `fuzz-aot` | AOT compile, AOT loader, and instantiate path | Compile/instantiate errors are OK. The harness deliberately sets `invoke_start = false` and does not execute attacker-supplied start functions. |
| `fuzz-diff` | Interpreter vs AOT result oracle for a statically safe straight-line scalar subset | Loader/instantiate errors, interpreter traps, fuel exhaustion, unsupported signatures, and exports whose bytecode is outside the safe subset are OK. A result-shape, tag, or value mismatch is saved as `diff-mismatch-<hash>.wasm` and is a bug. |
| `fuzz-canon` | Canonical ABI lift and string codecs over attacker-controlled guest memory | Typed errors and silent zero-fill on out-of-bounds reads are OK. Panics, safety-checked UB, oversize allocations, or process aborts are bugs. |

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

For interpreter invocation smoke with at least one valid nullary export:

```sh
rm -rf /tmp/wamr-interp-corpus /tmp/wamr-fuzz-crashes
mkdir -p /tmp/wamr-interp-corpus /tmp/wamr-fuzz-crashes
printf '\000asm\001\000\000\000\001\005\001\140\000\001\177\003\002\001\000\007\007\001\003run\000\000\012\006\001\004\000\101\052\013' > /tmp/wamr-interp-corpus/minimal-interp-run.wasm
./zig-out/bin/fuzz-interp --corpus /tmp/wamr-interp-corpus --crashes /tmp/wamr-fuzz-crashes --duration 5 --fuel 100000
```

For `fuzz-canon` smoke (pointer/length and string-codec exercise):

```sh
rm -rf /tmp/wamr-canon-corpus /tmp/wamr-fuzz-crashes
mkdir -p /tmp/wamr-canon-corpus /tmp/wamr-fuzz-crashes
# mode=load_primitive, prim=string, ptr=0, len=8, body 16B.
printf '\x00\x0e\x00\x00\x00\x00\x08\x00\x00\x00Hello, world!\x00\x00\x00' > /tmp/wamr-canon-corpus/seed-load-string.wasm
# mode=validate_utf8, ptr=0, len=11, body "hello world".
printf '\x01\x00\x00\x00\x00\x00\x0b\x00\x00\x00hello world' > /tmp/wamr-canon-corpus/seed-validate-utf8.wasm
./zig-out/bin/fuzz-canon --corpus /tmp/wamr-canon-corpus --crashes /tmp/wamr-fuzz-crashes --duration 5
```

`fuzz-canon` reads a small fixed header from each input (`mode`, primitive
selector, `ptr`, `len`) and treats the rest as guest memory. It exercises
`canonical_abi.loadVal` for primitive types and the UTF-8/UTF-16/Latin-1
string codecs against attacker-controlled bytes.

The harnesses replay every `.wasm` file under `--corpus` until the duration
expires. They are not coverage-guided mutators by themselves; use them as stable
entry points for CI and future mutation/generation infrastructure.

`fuzz-interp` and `fuzz-diff` accept `--fuel <instructions>` as a per-export
interpreter instruction budget. Fuel exhaustion is an expected outcome and is
reported separately from a guest `unreachable` trap.

For `fuzz-diff` smoke with the same minimal seed:

```sh
rm -rf /tmp/wamr-diff-corpus /tmp/wamr-fuzz-crashes
mkdir -p /tmp/wamr-diff-corpus /tmp/wamr-fuzz-crashes
printf '\000asm\001\000\000\000\001\005\001\140\000\001\177\003\002\001\000\007\007\001\003run\000\000\012\006\001\004\000\101\052\013' > /tmp/wamr-diff-corpus/minimal-interp-run.wasm
./zig-out/bin/fuzz-diff --corpus /tmp/wamr-diff-corpus --crashes /tmp/wamr-fuzz-crashes --duration 5 --fuel 100000
```

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
- adds a generated nullary scalar export seed for `fuzz-interp` and `fuzz-diff`;
- uploads `.fuzz-crashes` artifacts with 30-day retention;
- uploads per-target corpus artifacts with 7-day retention;
- fails the job when any crash artifact remains.

Manual runs can choose `all`, `loader`, `component-loader`, `interp`, `aot`,
`diff`, or `canon`, set a per-target duration, and set the per-export
interpreter fuel budget shared by `fuzz-interp` and `fuzz-diff`.

## Current limitations and follow-ups

The current targets are intentionally conservative. Do not remove these limits
without adding bounded execution, subprocess isolation, or another equivalent
host-protection mechanism.

- `fuzz-interp` skips modules with imports or start functions, then invokes at
  most eight exported local functions whose signatures are `() -> ()`,
  `() -> i32`, `() -> i64`, `() -> f32`, or `() -> f64`. Parameterized,
  reference-typed, `v128`, imported-function, and multi-result exports remain
  out of scope until a deterministic input and host-import policy is designed.
- `fuzz-diff` only compares results for nullary scalar exports whose bytecode
  passes a conservative straight-line filter that rejects calls, branches,
  loops, memory/table/atomic/exception/SIMD/ref ops, integer div/rem,
  truncation conversions, and `unreachable`. Broader AOT execution (arbitrary
  exports, parameterized exports, native trap differential) needs subprocess
  isolation because the AOT runtime only converts traps to typed errors on
  Windows x86_64; on Linux/AArch64 a native AOT trap can terminate the host.
- `fuzz-canon` exercises only primitive `loadVal` and the string codecs
  (`validateUtf8`, `utf8ToUtf16`, `utf16ToUtf8`, `decodeLatin1Utf16`,
  `encodeUtf16ToMem`). Compound types (record, variant, list, tuple, flags,
  enum, option, result) and the storeVal round-trip are intentionally out of
  scope here; covering them needs a controlled `TypeRegistry` builder driven
  from the same input bytes.
- WASI/component resource-table command-model fuzzing — descriptor lifecycle,
  preopen escape checks, stream drops, socket option get/set on disconnected
  sockets, and HTTP stream lifetimes — is a follow-up that needs a thin
  test wrapper around `WasiCliAdapter` so the fuzzer can drive the host
  surface without a guest module.
- Direct WASI and component-adapter fuzzing needs a deterministic byte-command
  model for resource tables, paths, descriptors, sockets, HTTP streams, and
  guest-memory pointer/length pairs (#247).
- Corpus minimization should preserve a small checked-in seed set while storing
  larger evolving corpora as workflow artifacts (#248).
- OSS-Fuzz integration should be evaluated after the local targets are stable,
  resource usage is bounded, and generated corpora are useful (#249).
