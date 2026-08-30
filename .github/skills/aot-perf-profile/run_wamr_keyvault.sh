#!/usr/bin/env bash
# Example WAMR runner for the #798 keyvault/Secrets TCGC codegen workload.
#
# The azure-sdk-for-zig `codegen/cli/scripts/run.sh` is wasmtime-only; this
# mirrors its WASI preopens but drives the wamr AOT binary, so the run is
# load+execute only (precompiled). Paths are overridable via env vars so the
# script works across checkouts. Use under `perf record` (Step 4 of SKILL.md).
#
#   WAMR_BIN=<wamr>  MANIFEST=<comp.cwasm.json>  OUT=<dir>  ./run_wamr_keyvault.sh
set -euo pipefail

WAMR_BIN="${WAMR_BIN:?set WAMR_BIN to the ReleaseFast wamr binary}"
ASZ="${ASZ:-$HOME/azure-sdk-for-zig}"
WASM="${WASM:-$ASZ/codegen/cli/zig-out/bin/codegen-cli.composed.wasm}"
PREOPENS="${PREOPENS:-$ASZ/codegen/tcgc-component/dist/stdlib-preopens.txt}"
SPEC="${SPEC:-$HOME/azure-rest-api-specs/specification/keyvault/data-plane/Secrets}"
OUT="${OUT:?set OUT to a dedicated output directory}"
MANIFEST="${MANIFEST:-}"   # optional --precompiled-manifest (else sibling auto-probe)

mkdir -p "$OUT"
SPEC="$(cd "$SPEC" && pwd)"; OUT="$(cd "$OUT" && pwd)"

args=( run )
[ -n "$MANIFEST" ] && args+=( --precompiled-manifest "$MANIFEST" )
args+=( --map-dir "$SPEC::/spec" --map-dir "$OUT::/out" )
while IFS= read -r line; do
    [ -z "$line" ] && continue
    host="${line%%=*}"; virt="${line#*=}"
    args+=( --map-dir "$host::$virt" )
done < "$PREOPENS"
args+=( "$WASM" /spec /out --package-name keyvault-secrets )

exec "$WAMR_BIN" "${args[@]}"
