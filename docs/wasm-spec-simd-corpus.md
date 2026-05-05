# WAMR: Spec SIMD corpus integration

## Purpose
Integrate the wasm-spec SIMD test suite (WebAssembly/simd) as a differential corpus under tests/wasm-spec-simd, linked to #307.

## Usage
Run all SIMD .wast files with the harness:

    python3 tests/wamr-test-suites/spec-test-script/runtest.py --dir tests/wasm-spec-simd/test/core/simd

Or process individual files (see tests/wasm-spec-simd-test-files.txt).

