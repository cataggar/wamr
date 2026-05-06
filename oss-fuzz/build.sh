#!/usr/bin/env bash
# OSS-Fuzz build script for the high-priority Zig fuzz harnesses.
#
# Repository-local: produces libFuzzer binaries for fuzz-loader,
# fuzz-component-loader, and fuzz-canon by linking the static-archive
# shims emitted by `zig build fuzz-oss` against $LIB_FUZZING_ENGINE.
#
# Expected environment (set by the OSS-Fuzz base-builder image):
#   $SRC, $OUT, $WORK, $CC, $CXX, $CFLAGS, $CXXFLAGS, $LIB_FUZZING_ENGINE
#
# This script is **local-only**. There is no upstream OSS-Fuzz
# `projects/wamr/` submission. See tests/fuzz/OSS_FUZZ.md.

set -euo pipefail

cd "$SRC/wamr"

# Build the static-archive shims. ReleaseSafe keeps Zig safety checks
# enabled so panics surface as libFuzzer crashes.
zig build fuzz-oss -Doptimize=ReleaseSafe

mkdir -p "$OUT"

declare -a TARGETS=(loader component-loader canon)

for target in "${TARGETS[@]}"; do
    archive="zig-out/lib/libfuzz-oss-${target}.a"
    if [ ! -f "$archive" ]; then
        echo "::error::missing $archive after zig build fuzz-oss" >&2
        exit 1
    fi

    # Link the libFuzzer driver (provides main + coverage feedback)
    # against the Zig static archive. --whole-archive ensures
    # LLVMFuzzerTestOneInput is not stripped before the driver finds it.
    "$CXX" $CXXFLAGS \
        -Wl,--whole-archive "$archive" -Wl,--no-whole-archive \
        $LIB_FUZZING_ENGINE \
        -o "$OUT/fuzz-oss-${target}"
done

# Seed corpora. Reuse the existing in-repo corpora and any committed
# regression seeds so libFuzzer starts from a known-good distribution.
for target in "${TARGETS[@]}"; do
    seed_zip="$OUT/fuzz-oss-${target}_seed_corpus.zip"
    seed_dir="$WORK/seeds/${target}"
    rm -rf "$seed_dir"
    mkdir -p "$seed_dir"

    if [ -d tests/malformed/fuzz ]; then
        find tests/malformed/fuzz -maxdepth 1 -name '*.wasm' \
            -exec cp -n {} "$seed_dir/" \;
    fi

    case "$target" in
        loader)
            if [ -d tests/spec-json ]; then
                find tests/spec-json -maxdepth 1 -name '*.wasm' \
                    -exec cp -n {} "$seed_dir/" \;
            fi
            ;;
        component-loader)
            # Minimal valid component header so libFuzzer has at least
            # one well-formed seed beyond the malformed corpus.
            printf '\000asm\015\000\001\000' \
                > "$seed_dir/minimal-component.wasm"
            ;;
        canon)
            # mode=load_primitive, prim=string, ptr=0, len=8.
            printf '\x00\x0e\x00\x00\x00\x00\x08\x00\x00\x00Hello, world!\x00\x00\x00' \
                > "$seed_dir/seed-load-string.wasm"
            # mode=validate_utf8, ptr=0, len=11, body "hello world".
            printf '\x01\x00\x00\x00\x00\x00\x0b\x00\x00\x00hello world' \
                > "$seed_dir/seed-validate-utf8.wasm"
            ;;
    esac

    # Pull in committed regression seeds for this target if present.
    if [ -d "tests/fuzz/regression/${target}" ]; then
        find "tests/fuzz/regression/${target}" -maxdepth 1 -name '*.wasm' \
            -exec cp -n {} "$seed_dir/" \;
    fi

    (cd "$seed_dir" && zip -q -r "$seed_zip" .)
done
