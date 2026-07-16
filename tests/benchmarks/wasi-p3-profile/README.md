# WASI P3 macOS ARM64 profiling

The manual `wasi-p3-macos-profile.yml` workflow compares the same revision on
native macOS and Linux ARM64. It runs ReleaseSafe AOT and supported in-process
JIT modes, with 5 cold and 10 warm unfiltered samples per mode/platform.
Every retained sample must pass 41/41.

`WAMR_PROFILE_TIMINGS` opts into JSONL phase events. The adapter records
component/core precompile time and cache state; the in-tree runner launcher
records guest-process execution without wrapping or changing guest streams.
Normal conformance runs are unchanged when the variable is absent.

The workflow also runs `wasi-microbench --no-budget`; Linux x86_64 budgets are
deliberately not reused for macOS. The aggregate job validates both JSON
schemas and selects macOS sampling profiles for `http-fields`, phases taking
at least 20% of suite wall time, and phases consistently at least 2x Linux.
