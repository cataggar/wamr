#ifndef WAMR_WASI_THREAD_BENCH_OUTPUT_H
#define WAMR_WASI_THREAD_BENCH_OUTPUT_H

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "timing.h"

static inline int bench_write_result(
    const char *workload,
    uint32_t threads,
    uint64_t iterations,
    uint64_t operations,
    uint64_t checksum,
    const char *metric_kind,
    uint64_t timed_loop_backedges,
    const struct bench_timing *timing) {
    char buffer[512];
    if (timing->raw_elapsed_ns == 0) return 1;
    uint64_t timing_overhead_ppm =
        (uint64_t)(((unsigned __int128)timing->timing_overhead_ns *
                       UINT64_C(1000000)) /
                   timing->raw_elapsed_ns);
    int written = snprintf(
        buffer,
        sizeof(buffer),
        "{\"kind\":\"wasi-thread-benchmark-result\",\"workload\":\"%s\","
        "\"threads\":%" PRIu32 ",\"iterations\":%" PRIu64
        ",\"operations\":%" PRIu64 ",\"checksum\":%" PRIu64
        ",\"clock_id\":\"wasi-monotonic\",\"metric_kind\":\"%s\","
        "\"raw_elapsed_ns\":%" PRIu64 ",\"timing_overhead_ns\":%" PRIu64
        ",\"elapsed_ns\":%" PRIu64 ",\"timing_overhead_ppm\":%" PRIu64
        ",\"timed_loop_backedges\":%" PRIu64
        ",\"clock_calls_in_timed_loop\":0}\n",
        workload,
        threads,
        iterations,
        operations,
        checksum,
        metric_kind,
        timing->raw_elapsed_ns,
        timing->timing_overhead_ns,
        timing->elapsed_ns,
        timing_overhead_ppm,
        timed_loop_backedges);
    if (written < 0 || (size_t)written >= sizeof(buffer)) return 1;
    size_t length = (size_t)written;
    return write(STDOUT_FILENO, buffer, length) == (ssize_t)length ? 0 : 1;
}

#endif
