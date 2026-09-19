#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
DIR="$ROOT/tests/benchmarks/leaf-cancel-cost"
SDK=${WASI_SDK_PATH:-}
EXPECTED_VERSION='clang version 19.1.5-wasi-sdk'

if [[ -z "$SDK" || ! -x "$SDK/bin/clang" ]]; then
  echo "WASI_SDK_PATH must name an extracted wasi-sdk-25.0 installation" >&2
  exit 2
fi
CLANG_VERSION=$("$SDK/bin/clang" --version | sed -n '1p')
if [[ "$CLANG_VERSION" != "$EXPECTED_VERSION"* ]]; then
  echo "WASI_SDK_PATH is not the pinned wasi-sdk-25.0 toolchain" >&2
  exit 2
fi

TMP_ROOT=${TMPDIR:-"$ROOT/zig-out/leaf-cancel-fixture-tmp"}
mkdir -p "$TMP_ROOT"
"$SDK/bin/clang" \
  --target=wasm32-wasi-threads \
  "--sysroot=$SDK/share/wasi-sysroot" \
  -O3 \
  -std=c11 \
  -Wall \
  -Wextra \
  -Werror \
  "-ffile-prefix-map=$ROOT=." \
  "-fdebug-prefix-map=$ROOT=." \
  -pthread \
  -matomics \
  -mbulk-memory \
  -mmutable-globals \
  -Wl,--max-memory=67108864 \
  -Wl,--strip-all \
  "$DIR/leaf_calls.c" \
  -o "$DIR/leaf_calls.wasm"

chmod 0644 "$DIR/leaf_calls.wasm"
(
  cd "$DIR"
  sha256sum -c fixtures.sha256
)
