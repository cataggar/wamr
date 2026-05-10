#!/usr/bin/env bash
# Profile the CoreMark wasm under Wasmtime, mirroring `zig build coremark-profile`
# on the wamr side. Produces a top-N hot-function report from `perf record` and
# the disassembly of the top function via `llvm-objdump` on the precompiled
# `.cwasm` ELF.
#
# Usage:
#   scripts/profile_wasmtime_coremark.sh [path/to/coremark.wasm]
#
# Defaults to tests/benchmarks/coremark/coremark_wasi_nofp.wasm.
#
# Prerequisites (not auto-installed):
#   - wasmtime           (https://wasmtime.dev)
#   - perf               (linux-tools-$(uname -r))
#   - llvm-objdump       (LLVM 17+)
#
# `perf record -e cycles:u` requires `kernel.perf_event_paranoid <= 2`.
# Bump it for the session with:
#   sudo sysctl -w kernel.perf_event_paranoid=1

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
wasm_path="${1:-${repo_root}/tests/benchmarks/coremark/coremark_wasi_nofp.wasm}"

if [[ ! -f "$wasm_path" ]]; then
  echo "wasm not found: $wasm_path" >&2
  exit 1
fi

for tool in wasmtime perf llvm-objdump; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "required tool '$tool' not in PATH" >&2
    exit 1
  fi
done

work_dir="$(mktemp -d -t wasmtime-coremark-XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

cwasm="${work_dir}/coremark.cwasm"
perf_data="${work_dir}/perf.data"

echo "[wasmtime-profile] precompiling $(basename "$wasm_path") -> $(basename "$cwasm")"
wasmtime compile -O opt-level=2 "$wasm_path" -o "$cwasm"

echo "[wasmtime-profile] perf record (cycles:u, freq=1000)"
perf record -F 1000 -e cycles:u -o "$perf_data" -- \
  wasmtime run --allow-precompiled "$cwasm"

echo
echo "[wasmtime-profile] perf report (top 20, --no-children)"
perf report -i "$perf_data" --stdio --no-children -n 2>/dev/null | \
  grep -E '^\s*[0-9]' | head -20 || true

echo
echo "[wasmtime-profile] cwasm ELF disassembly summary"
llvm-objdump -d "$cwasm" 2>/dev/null | head -80 || true

echo
echo "[wasmtime-profile] perf.data preserved at: $perf_data"
echo "    rerun manually:  perf report -i $perf_data --stdio"
trap - EXIT
