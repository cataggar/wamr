#include "kernel.h"
#include "output.h"

#include <errno.h>
#include <stdlib.h>

static int parse_u64(const char *text, uint64_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0) return -1;
    *out = (uint64_t)value;
    return 0;
}

int main(int argc, char **argv) {
    uint64_t iterations = 0;
    if (argc != 2 || parse_u64(argv[1], &iterations) != 0) {
        return 2;
    }

    uint64_t warmup_iterations = iterations / 64;
    if (warmup_iterations < 1024) warmup_iterations = 1024;
    (void)bench_hot_kernel(bench_seed(0), warmup_iterations);

    uint64_t overhead = 0;
    uint64_t start = 0;
    uint64_t end = 0;
    struct bench_timing timing;
    if (bench_clock_overhead(&overhead) != 0 ||
        bench_now_ns(&start) != 0) {
        return 1;
    }
    uint64_t checksum = bench_hot_kernel(bench_seed(0), iterations);
    if (bench_now_ns(&end) != 0 ||
        bench_finish_timing(start, end, overhead, &timing) != 0) {
        return 1;
    }
    return bench_write_result(
        "single-hot",
        1,
        iterations,
        iterations,
        checksum,
        "steady-state-kernel",
        iterations,
        &timing);
}
