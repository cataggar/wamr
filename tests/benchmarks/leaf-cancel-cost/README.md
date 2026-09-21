# Leaf-call cancellation-poll cost (#963)

This is an isolated evidence harness for the function-entry cancellation polls
added by #1010. It is deliberately separate from the #966 WASI-thread
benchmark: it has a distinct fixture, report kind/schema, sizing rule, workflow,
and output directory. It does not change #966's plan, fixture identity,
calibration, budgets, or smoke evidence.

## Workload and comparison

The checked-in `leaf_calls.wasm` creates one wasi-libc pthread. Inside that
worker it times a noinline leaf function called 64 times per outer-loop batch.
The leaf performs one wrapping integer addition, making the timed path
call/return-heavy rather than loop-body-heavy. Every measured sample executes
the same number of leaf calls and validates the closed-form checksum:

`seed + leaf_calls * step (mod 2^64)`.

The harness builds one threads-enabled AOT runtime and compiles the exact wasm
fixture twice with the same `wamrc`:

- `cancel-points-on`: normal threaded AOT compilation;
- `cancel-points-off`: adds the existing benchmark-only
  `--benchmark-disable-cancel-points` flag.

The report retains both exact compile commands and verifies their normalized
forms are identical apart from the condition-bound output path and
`--benchmark-disable-cancel-points`. It parses every cwasm section, function
offset/type entry, import, and export. All non-code sections must be
byte-identical. For every function, the enabled artifact is normalized by
removing only the complete architecture-specific poll sequence and adjusting
only control-flow displacements, function offsets, and inline jump-table
targets that are directly changed by those insertions; the resulting function
must exactly match the disabled artifact. Any unrelated instruction, data, or
metadata difference fails closed.

The AOT export/import mapping must resolve `leaf_step` to one local function.
That exact function must contain one complete enabled poll sequence, none in
the disabled artifact, immediately follow the architecture's compiler ABI
entry-prefix tail, and have an otherwise identical normalized body. Only
after that proof does the harness report one leaf-entry opportunity per call.
A synthetic signature elsewhere in the artifact is insufficient. Sample order
alternates off/on then on/off. Guest process-CPU time excludes process startup,
module loading, pthread creation/join, and JSON output. Both report formats
retain commands, normalization proof, source/tool/fixture/AOT hashes, host
identity, raw records, and paired cost per leaf call.

The timed outer loop contributes at most one loop-header poll opportunity per
64 leaf calls, plus one entry poll for the noinline driver. The leaf-entry
share therefore approaches 98.46% for sized runs. The report computes and
retains the exact per-run lower bound and labels only the exact leaf-entry
opportunities (`leaf_calls`); it does not present the amortized delta as an
uncontaminated single-poll latency or mislabel every static libc poll site as
dynamically executed.

## Rebuild the fixture

Use pinned wasi-sdk 25.0 (clang 19.1.5):

```sh
mkdir -p zig-out/leaf-cancel-sdk/{tmp,download}
export TMPDIR="$PWD/zig-out/leaf-cancel-sdk/tmp"
curl --fail --location \
  https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-25/wasi-sdk-25.0-x86_64-linux.tar.gz \
  -o zig-out/leaf-cancel-sdk/download/wasi-sdk.tar.gz
echo '52640dde13599bf127a95499e61d6d640256119456d1af8897ab6725bcf3d89c  zig-out/leaf-cancel-sdk/download/wasi-sdk.tar.gz' |
  sha256sum -c -
mkdir -p zig-out/leaf-cancel-sdk/wasi-sdk
tar -xzf zig-out/leaf-cancel-sdk/download/wasi-sdk.tar.gz \
  -C zig-out/leaf-cancel-sdk/wasi-sdk --strip-components=1
WASI_SDK_PATH="$PWD/zig-out/leaf-cancel-sdk/wasi-sdk" \
  tests/benchmarks/leaf-cancel-cost/build-fixture.sh
git diff --exit-code -- tests/benchmarks/leaf-cancel-cost
```

## Native evidence

From the repository root on x86_64 or AArch64 Linux:

```sh
export TMPDIR="$PWD/zig-out/leaf-cancel-cost/tmp"
export XDG_CACHE_HOME="$PWD/zig-out/leaf-cancel-cost/xdg-cache"
python3 scripts/bench_leaf_cancel_cost.py \
  --platform-id local-linux-$(uname -m) \
  --runner-environment local \
  --output-dir "$PWD/zig-out/leaf-cancel-cost"
```

Defaults retain one sizing pilot, two discarded balanced warmup pairs, twelve
balanced measured pairs, a 3-second target, and a 1.25-second quality floor.
Use `--calls` to freeze an already reviewed call count and skip the pilot.

## AArch64 routes

The separate manual workflow `.github/workflows/leaf-cancel-cost.yml` runs the
same harness natively on GitHub's x86_64 and AArch64 Linux runners and uploads
both report formats. It accepts only an immutable commit SHA, bounds fixed work
and pair counts, uses pinned actions and run-local caches, and is dispatch-only
so preparation changes do not start remote measurement automatically.

For the repository's qemu route on an x86_64 host with `qemu-aarch64`:

```sh
python3 scripts/bench_leaf_cancel_cost.py \
  --target aarch64-linux-musl \
  --aot-target aarch64 \
  --runner qemu-aarch64 \
  --platform-id qemu-aarch64 \
  --runner-environment local-qemu \
  --output-dir "$PWD/zig-out/leaf-cancel-cost-aarch64"
```

Qemu is appropriate for portability and correctness checks. Closing-cost
numbers should use the native AArch64 workflow route.

## CoreMark unchanged-path follow-up

CoreMark is non-threaded and the compiler toggle intentionally rejects its
use there, so mixing CoreMark into this report would weaken its single-variable
identity. Collect the existing non-threaded evidence independently:

```sh
python3 scripts/bench_coremark.py \
  --baseline <issue-963-parent-sha> \
  --target <issue-963-candidate-sha> \
  --profile authoritative \
  --wasmtime-baseline auto
```

Run that command once on x86_64 and through
`.github/workflows/coremark-aarch64.yml` on AArch64. Keep those existing
CoreMark reports alongside, rather than inside, the `leaf-cancel-cost-v1`
evidence.
