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

    uint64_t checksum = bench_hot_kernel(bench_seed(0), iterations);
    return bench_write_result(
        "single-hot", 1, iterations, iterations, checksum);
}
