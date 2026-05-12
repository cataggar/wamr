# WebAssembly Component examples

Source-only Zig (and one mixed Zig+Rust) examples that exercise the
[bytecodealliance Component-Model][bca] story end-to-end through
[`cataggar/wabt`][cw] and wamr. The examples mirror the structure of the
[component-docs][cd] adder/calculator/command tutorial — Zig is the
language column the upstream docs do not yet cover.

[bca]: https://component-model.bytecodealliance.org/
[cd]: https://github.com/bytecodealliance/component-docs/tree/main/component-model/examples/tutorial
[cw]: https://github.com/cataggar/wabt

No `.wasm` artifacts are checked in; everything is built reproducibly
from the sources in this directory via the repo's root `build.zig`.

## Layout

```
examples/components/
├── README.md                       (this file)
├── stdio-echo/                     (existing Rust example, see #156)
├── zig-hello/                      smallest end-to-end Zig command
├── zig-adder/                      library exporting docs:adder/add@0.1.0
├── zig-calculator-cmd/             Zig command importing docs:adder
└── mixed-zig-rust-calc/            Zig adder + Rust command, composed
    └── command/                    cargo + wit-bindgen, wasm32-wasip1
```

Each example directory has its own `README.md` with example-specific
build details, source-walkthrough notes, and runtime status.

## Build steps

Two opt-in `build.zig` steps drive the pipeline. Neither runs as part
of the default `zig build` or `zig build test` graph — they are skipped
on machines that do not have the external toolchain installed.

| Step                            | What it does                                                     |
|---------------------------------|------------------------------------------------------------------|
| `zig build component-examples`     | Build, encode, and validate all four components.                 |
| `zig build component-examples-run` | Run `zig-hello` through `./zig-out/bin/wamr` (smoke test).       |

Outputs land under `zig-out/component-examples/`:

```
zig-out/component-examples/
├── zig-hello.component.wasm
├── zig-adder.component.wasm
├── zig-calculator-cmd.composed.wasm
└── mixed-zig-rust-calc.composed.wasm
```

## Pinned tool versions

| Tool                                  | Version    | Notes                                                    |
|---------------------------------------|-----------:|----------------------------------------------------------|
| Zig                                   | 0.16.x     | Toolchain already required by the rest of the repo.       |
| `wabt` (cataggar/wabt)                | v3.0.0-dev.5 | Provides `component embed/new`, `module validate`, `component compose`. The wasi-preview1 → component adapter is bundled inside `wabt` and auto-attached by `wabt component new`. |
| Rust toolchain (`cargo`, `rustup`)    | recent stable | Mixed example only.                                  |
| `wasm32-wasip1` Rust target           | —          | `rustup target add wasm32-wasip1`. Mixed example only.    |

## What runs in wamr today

| Example                   | Builds | Validates | Runs in wamr today | Notes                                                      |
|---------------------------|:------:|:---------:|:------------------:|------------------------------------------------------------|
| `zig-hello`               |   ✓    |     ✓     |         ✓          | End-to-end (greeting written via the captured-stdout flush). |
| `zig-adder`               |   ✓    |     ✓     |        n/a         | Library component — no `wasi:cli/run`.                     |
| `zig-calculator-cmd` (composed) | ✓ |     ✓     |         ✓          | Composed end-to-end through `wabt component compose`.       |
| `mixed-zig-rust-calc`     |   ✓    |     ✓     |         ✓          | Composed end-to-end through `wabt component compose`.       |

All composed examples produce valid components that run in
[Wasmtime][wasmtime] and on wamr today. `wabt component compose` emits
the cross-component plumbing in the inline-export form (rather than the
aliased-instance form `wasm-tools compose` historically used); both
shapes are spec-valid, and wamr's component loader handles both.

The `zig-hello` runtime test asserts that wamr exits 0 and emits the
greeting on host stdout (fd 1).

[wasmtime]: https://wasmtime.dev/

## Caveats

- Component-Model support in wamr is still in **preview**; behaviour
  may shift between releases. Each example's README pins its
  expectations to the current state of the runtime.
- The wasi-preview1 → Component adapter is bundled inside the
  `cataggar/wabt` CLI (see `cataggar/wabt#145`/#156) and is
  auto-attached by `wabt component new` for command-style embeds
  whose `wasi_snapshot_preview1.*` imports are otherwise unresolved.
  Languages (Zig, Rust, …) whose toolchains target `wasm32-wasip1`
  natively but do not yet emit Component-Model artifacts directly
  reach the preview2 surface through this adapter.
- The adapter lowers preview1 `proc_exit(code)` through
  `wasi:cli/exit.exit-with-code(u8)` (with a defensive fallback to
  the stable `exit(result<_, _>)`). `exit-with-code` is the
  `@unstable(feature = cli-exit-with-code)` extension of the
  wasi-cli@0.2.6 interface — wamr supports it unconditionally,
  while Wasmtime v44 gates the linker binding behind
  `-S cli-exit-with-code`. To run `zig-exit` / `mixed-zig-rust-calc`
  on wasmtime: `wasmtime run -S cli-exit-with-code <component>.wasm`.
