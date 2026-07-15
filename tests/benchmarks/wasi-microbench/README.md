# `wasi-microbench` — host-path regression detector (#583, W11-6)

CoreMark-aware micro-bench parallel to the existing `coremark` +
`coldstart` benches. Drives `executor.dispatchCanonBuiltin` →
`AsyncStream` → host_driver across four hot Preview-3 host paths so
WAMR-side performance regressions surface in CI before they land in a
release.

## Scenarios

| name | shape | what it pins |
|------|-------|--------------|
| `http-service-keepalive-100rt` | 100 × (`stream.write` + `stream.read` of 256 B) | request/response keep-alive dispatch cost (mirrors `wasi:http` over `wasi:sockets@0.3.x` keep-alive) |
| `udp-receive-1mb` | 128 × `stream.read` of 8 KiB | UDP datagram receive zero-copy lowering (mirrors `wasi:sockets@0.3.x.udp-socket.receive`) |
| `fs-write-via-stream-1mb` | 16 × `stream.write` of 64 KiB | `wasi:filesystem.write-via-stream` lowering |
| `fs-read-via-stream-1mb` | 16 × `stream.read` of 64 KiB | `wasi:filesystem.read-via-stream` lowering |

The drivers are synthetic — no real sockets, no `pwrite(2)`. The cost
shape mirrors `wasi_cli_adapter`'s `on_read_into` / `on_write_from`
callbacks (one `@memcpy` of `bytes_per_iter`, no syscall), so the
bench isolates the canonical-ABI lowering from kernel jitter. Real-
network coverage stays under the conformance gates.

## Run

```sh
# Default: 4 scenarios × 10 samples + 2 warmup, regression threshold 10 %.
zig build wasi-microbench

# More samples for tighter medians.
zig build wasi-microbench -- --samples 20 --warmup 5

# Skip the regression check (e.g. when calibrating a new budget).
zig build wasi-microbench -- --no-budget

# Emit machine-readable summary.
zig build wasi-microbench -- --json wasi-microbench.json
```

## Budget

[`budget.json`](budget.json) holds per-scenario `median_ns_budget`
+ `samples`. A scenario whose median wall-clock exceeds
`median_ns_budget × (1 + threshold/100)` fails the run. Default
threshold is +10 %.

The current budgets were calibrated on 2026-07-15 from 98 retained
`ubuntu-22.04` x86_64 workflow artifact reports spanning 2026-06-06
through 2026-07-14. Each run's scenario median is one observation. The
effective failure gates target approximately the mean of those medians
plus two sample standard deviations, with an observed whole-run pass
rate of 95/98 (96.9 %). Because the harness applies the +10 % threshold,
`budget.json` stores each effective gate divided by 1.10; the threshold
is the noise margin rather than a second margin.

When an intentional perf change lands, follow the recipe under
[`docs/wasi.md` § "Updating the wasi-microbench budget"](../../../docs/wasi.md).

## CI

[`.github/workflows/wasi-microbench.yml`](../../../.github/workflows/wasi-microbench.yml)
runs on pushes to main and PRs that touch `src/{component,runtime,wasi}/`,
`build.zig`, `build.zig.zon`, this directory, or the workflow itself;
it can also be dispatched manually.
The benchmark no longer uses `continue-on-error`: a regression fails
the workflow job whenever this path-filtered workflow runs, while the
report artifact is still uploaded. Branch protection does not currently
require this workflow context, so its failure does not by itself block
every PR from merging.
