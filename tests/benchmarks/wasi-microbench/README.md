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
threshold is +10 %; budgets are calibrated 1.5× the median observed on
the project AArch64 dev VM so noise doesn't trip CI.

When an intentional perf change lands, follow the recipe under
[`docs/wasi.md` § "Updating the wasi-microbench budget"](../../../docs/wasi.md).

## CI

[`.github/workflows/wasi-microbench.yml`](../../../.github/workflows/wasi-microbench.yml)
runs the bench on push-to-main + PRs that touch
`src/{component,runtime,wasi}/`, `build.zig`, or this directory.
`continue-on-error: true` while the hosted-x86_64 baseline stabilises.
