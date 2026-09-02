#ifndef WAMR_WASI_THREAD_BENCH_KERNEL_H
#define WAMR_WASI_THREAD_BENCH_KERNEL_H

#include <stdint.h>

static inline uint64_t bench_seed(uint32_t worker_index) {
    return UINT64_C(0x243f6a8885a308d3) ^
           (UINT64_C(0x9e3779b97f4a7c15) * (worker_index + 1));
}

static inline uint64_t bench_hot_kernel(uint64_t seed, uint64_t iterations) {
    volatile uint64_t value = seed;
    for (uint64_t i = 0; i < iterations; ++i) {
        uint64_t current = value;
        current = (current << 7) | (current >> 57);
        value = current ^ (i + UINT64_C(0xd1b54a32d192ed03));
    }
    return value;
}

#endif
