# Installing WAMR

Pre-built binaries are published to [GitHub Releases](https://github.com/cataggar/wamr/releases) and [PyPI](https://pypi.org/project/wamr-bin/).

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

## uv

```sh
uv tool install wamr-bin
```

The AOT compiler is published separately:

```sh
uv tool install wamrc-bin
```

## pip

```sh
python3 -m pip install wamr-bin
```

```sh
python3 -m pip install wamrc-bin
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
zig build -Doptimize=ReleaseSafe
```

Binaries are written to `zig-out/bin/`.
