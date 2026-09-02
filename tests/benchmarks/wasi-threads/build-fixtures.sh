#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
DIR="$ROOT/tests/benchmarks/wasi-threads"
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

TMP_ROOT=${TMPDIR:-"$ROOT/zig-out/wasi-thread-fixture-tmp"}
mkdir -p "$TMP_ROOT"
COMMON=(
  -O3
  -std=c11
  -Wall
  -Wextra
  -Werror
  "-ffile-prefix-map=$ROOT=."
  "-fdebug-prefix-map=$ROOT=."
  -Wl,--strip-all
)

"$SDK/bin/clang" \
  --target=wasm32-wasi \
  "--sysroot=$SDK/share/wasi-sysroot" \
  "${COMMON[@]}" \
  "$DIR/single.c" \
  -o "$DIR/single.wasm"

"$SDK/bin/clang" \
  --target=wasm32-wasi-threads \
  "--sysroot=$SDK/share/wasi-sysroot" \
  "${COMMON[@]}" \
  -pthread \
  -matomics \
  -mbulk-memory \
  -mmutable-globals \
  -Wl,--max-memory=536870912 \
  "$DIR/threaded.c" \
  -o "$DIR/threaded.wasm"

chmod 0644 "$DIR/single.wasm" "$DIR/threaded.wasm"

(
  cd "$DIR"
  sha256sum -c fixtures.sha256
)
