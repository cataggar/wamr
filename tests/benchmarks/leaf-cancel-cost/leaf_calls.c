#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <wasi/api.h>

#define LEAF_UNROLL UINT64_C(64)
#define LEAF_SEED UINT64_C(0x243f6a8885a308d3)
#define LEAF_STEP UINT64_C(0x9e3779b97f4a7c15)

struct timing {
    uint64_t raw_elapsed_ns;
    uint64_t overhead_ns;
    uint64_t elapsed_ns;
};

struct worker_state {
    uint64_t calls;
    uint64_t checksum;
    struct timing timing;
    int status;
};

__attribute__((noinline, export_name("leaf_step")))
uint64_t leaf_step(uint64_t value) {
    return value + LEAF_STEP;
}

#define CALL_LEAF() value = leaf_step(value)
#define CALL_LEAF_2() CALL_LEAF(); CALL_LEAF()
#define CALL_LEAF_4() CALL_LEAF_2(); CALL_LEAF_2()
#define CALL_LEAF_8() CALL_LEAF_4(); CALL_LEAF_4()
#define CALL_LEAF_16() CALL_LEAF_8(); CALL_LEAF_8()
#define CALL_LEAF_32() CALL_LEAF_16(); CALL_LEAF_16()
#define CALL_LEAF_64() CALL_LEAF_32(); CALL_LEAF_32()

__attribute__((noinline))
static uint64_t run_leaf_calls(uint64_t calls) {
    uint64_t value = LEAF_SEED;
    for (uint64_t batch = 0; batch < calls / LEAF_UNROLL; ++batch) {
        CALL_LEAF_64();
    }
    return value;
}

static int process_cpu_now(uint64_t *result) {
    __wasi_timestamp_t value = 0;
    if (__wasi_clock_time_get(
            __WASI_CLOCKID_PROCESS_CPUTIME_ID, 0, &value) != 0) {
        return -1;
    }
    *result = value;
    return 0;
}

static int clock_overhead(uint64_t *result) {
    uint64_t minimum = UINT64_MAX;
    for (uint32_t i = 0; i < 9; ++i) {
        uint64_t start = 0;
        uint64_t end = 0;
        if (process_cpu_now(&start) != 0 || process_cpu_now(&end) != 0 ||
            end < start) {
            return -1;
        }
        if (end - start < minimum) minimum = end - start;
    }
    *result = minimum;
    return 0;
}

static void *worker(void *opaque) {
    struct worker_state *state = opaque;
    uint64_t warmup = state->calls / 128;
    if (warmup < LEAF_UNROLL) warmup = LEAF_UNROLL;
    warmup -= warmup % LEAF_UNROLL;
    (void)run_leaf_calls(warmup);

    uint64_t overhead = 0;
    uint64_t start = 0;
    uint64_t end = 0;
    if (clock_overhead(&overhead) != 0 || process_cpu_now(&start) != 0) {
        state->status = 1;
        return NULL;
    }
    state->checksum = run_leaf_calls(state->calls);
    if (process_cpu_now(&end) != 0 || end <= start ||
        end - start <= overhead) {
        state->status = 1;
        return NULL;
    }
    state->timing.raw_elapsed_ns = end - start;
    state->timing.overhead_ns = overhead;
    state->timing.elapsed_ns = end - start - overhead;
    return NULL;
}

static int parse_calls(const char *text, uint64_t *result) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 ||
        value % LEAF_UNROLL != 0) {
        return -1;
    }
    *result = (uint64_t)value;
    return 0;
}

int main(int argc, char **argv) {
    struct worker_state state = {0};
    if (argc != 2 || parse_calls(argv[1], &state.calls) != 0) return 2;

    pthread_t thread;
    if (pthread_create(&thread, NULL, worker, &state) != 0) return 3;
    if (pthread_join(thread, NULL) != 0 || state.status != 0) return 4;

    uint64_t overhead_ppm =
        (uint64_t)(((unsigned __int128)state.timing.overhead_ns *
                    UINT64_C(1000000)) /
                   state.timing.raw_elapsed_ns);
    char output[640];
    int length = snprintf(
        output,
        sizeof(output),
        "{\"kind\":\"leaf-cancel-cost-result\",\"leaf_calls\":%" PRIu64
        ",\"leaf_unroll\":%" PRIu64 ",\"batches\":%" PRIu64
        ",\"checksum\":%" PRIu64 ",\"clock_id\":\"wasi-process-cputime\""
        ",\"raw_elapsed_ns\":%" PRIu64 ",\"timing_overhead_ns\":%" PRIu64
        ",\"elapsed_ns\":%" PRIu64 ",\"timing_overhead_ppm\":%" PRIu64
        "}\n",
        state.calls,
        LEAF_UNROLL,
        state.calls / LEAF_UNROLL,
        state.checksum,
        state.timing.raw_elapsed_ns,
        state.timing.overhead_ns,
        state.timing.elapsed_ns,
        overhead_ppm);
    if (length < 0 || (size_t)length >= sizeof(output)) return 5;
    return write(STDOUT_FILENO, output, (size_t)length) == length ? 0 : 5;
}
