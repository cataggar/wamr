#ifndef WAMR_WASI_THREAD_BENCH_TIMING_H
#define WAMR_WASI_THREAD_BENCH_TIMING_H

#include <stdint.h>
#include <time.h>

struct bench_timing {
    uint64_t raw_elapsed_ns;
    uint64_t timing_overhead_ns;
    uint64_t elapsed_ns;
};

static inline int bench_now_ns(uint64_t *result) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0 ||
        value.tv_sec < 0 || value.tv_nsec < 0 ||
        value.tv_nsec >= 1000000000L) {
        return -1;
    }
    uint64_t seconds = (uint64_t)value.tv_sec;
    if (seconds > (UINT64_MAX - (uint64_t)value.tv_nsec) / UINT64_C(1000000000))
        return -1;
    *result = seconds * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
    return 0;
}

static inline int bench_clock_overhead(uint64_t *result) {
    uint64_t minimum = UINT64_MAX;
    for (uint32_t i = 0; i < 9; ++i) {
        uint64_t start = 0;
        uint64_t end = 0;
        if (bench_now_ns(&start) != 0 || bench_now_ns(&end) != 0 ||
            end < start) {
            return -1;
        }
        uint64_t elapsed = end - start;
        if (elapsed < minimum) minimum = elapsed;
    }
    *result = minimum;
    return 0;
}

static inline int bench_finish_timing(
    uint64_t start,
    uint64_t end,
    uint64_t overhead,
    struct bench_timing *result) {
    if (end <= start || end - start <= overhead) return -1;
    result->raw_elapsed_ns = end - start;
    result->timing_overhead_ns = overhead;
    result->elapsed_ns = result->raw_elapsed_ns - overhead;
    return 0;
}

#endif
