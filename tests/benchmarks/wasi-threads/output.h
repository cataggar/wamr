#ifndef WAMR_WASI_THREAD_BENCH_OUTPUT_H
#define WAMR_WASI_THREAD_BENCH_OUTPUT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

static inline void bench_append_text(
    char *buffer, size_t *length, const char *text) {
    size_t text_length = strlen(text);
    memcpy(buffer + *length, text, text_length);
    *length += text_length;
}

static inline void bench_append_u64(
    char *buffer, size_t *length, uint64_t value) {
    char reversed[20];
    size_t digits = 0;
    do {
        reversed[digits++] = (char)('0' + value % 10);
        value /= 10;
    } while (value != 0);
    while (digits > 0) buffer[(*length)++] = reversed[--digits];
}

static inline int bench_write_result(
    const char *workload,
    uint32_t threads,
    uint64_t iterations,
    uint64_t operations,
    uint64_t checksum) {
    char buffer[256];
    size_t length = 0;
    bench_append_text(
        buffer,
        &length,
        "{\"kind\":\"wasi-thread-benchmark-result\",\"workload\":\"");
    bench_append_text(buffer, &length, workload);
    bench_append_text(buffer, &length, "\",\"threads\":");
    bench_append_u64(buffer, &length, threads);
    bench_append_text(buffer, &length, ",\"iterations\":");
    bench_append_u64(buffer, &length, iterations);
    bench_append_text(buffer, &length, ",\"operations\":");
    bench_append_u64(buffer, &length, operations);
    bench_append_text(buffer, &length, ",\"checksum\":");
    bench_append_u64(buffer, &length, checksum);
    bench_append_text(buffer, &length, "}\n");
    return write(STDOUT_FILENO, buffer, length) == (ssize_t)length ? 0 : 1;
}

#endif
