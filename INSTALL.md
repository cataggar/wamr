# Installing WAMR

Pre-built binaries are published to [GitHub Releases](https://github.com/cataggar/wamr/releases). PyPI packages are no longer maintained.

## ghr (recommended)

[ghr](https://github.com/cataggar/ghr) is an installer for GitHub releases. It downloads the right binary for your platform, places it on `PATH`, and can upgrade it later.

```sh
ghr install cataggar/wamr
```

To install a specific version:

```sh
ghr install cataggar/wamr@0.1.0
```

Upgrade to the latest release:

```sh
ghr upgrade wamr
```

## dist

[dist](https://github.com/ekristen/distillery) is a GitHub release installer.

```sh
dist install cataggar/wamr
```

## From source

Requires [Zig](https://ziglang.org/) 0.16.x. No other dependencies.

```sh
git clone https://github.com/cataggar/wamr
cd wamr
zig build -Doptimize=safe
```

Binaries are written to `zig-out/bin/`.

### JIT mode

The pre-built binaries above (GitHub Releases, `ghr`, `dist`) are all
**AOT-only** — the release workflow does not pass
`-Djit`. To get a `wamr` that compiles and runs a plain `.wasm` module
directly in one step (no separate `wamrc` precompile), build from
source with the flag added:

```sh
zig build -Doptimize=safe -Djit=true
```

See the [JIT mode section of the README](README.md#jit-mode) for what
this trades off (bigger binary, per-invocation compile latency) and
how the two flows compare.
