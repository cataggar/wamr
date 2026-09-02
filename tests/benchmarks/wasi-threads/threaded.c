#include "kernel.h"
#include "output.h"

#include <errno.h>
#include <pthread.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_THREADS 8
#define THREAD_STACK_BYTES (16 * 1024)

enum workload {
    WORKLOAD_HOT,
    WORKLOAD_ATOMIC,
    WORKLOAD_WAIT_NOTIFY,
    WORKLOAD_SPAWN_JOIN,
};

struct worker_arg {
    uint32_t index;
    uint64_t iterations;
    uint64_t result;
    int error;
};

static _Atomic uint64_t shared_counter;
static _Atomic int32_t gates[MAX_THREADS];
static _Atomic int32_t acknowledgements[MAX_THREADS];
static alignas(16) unsigned char
    worker_stacks[MAX_THREADS][THREAD_STACK_BYTES];

static int parse_u32(const char *text, uint32_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 ||
        value > UINT32_MAX) {
        return -1;
    }
    *out = (uint32_t)value;
    return 0;
}

static int parse_u64(const char *text, uint64_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0) return -1;
    *out = (uint64_t)value;
    return 0;
}

static void *hot_worker(void *opaque) {
    struct worker_arg *arg = opaque;
    arg->result = bench_hot_kernel(bench_seed(arg->index), arg->iterations);
    return NULL;
}

static void *atomic_worker(void *opaque) {
    struct worker_arg *arg = opaque;
    for (uint64_t i = 0; i < arg->iterations; ++i) {
        atomic_fetch_add_explicit(&shared_counter, 1, memory_order_relaxed);
    }
    arg->result = arg->iterations;
    return NULL;
}

static int wait_while_equal(_Atomic int32_t *address, int32_t expected) {
    while (atomic_load_explicit(address, memory_order_seq_cst) == expected) {
        int result = __builtin_wasm_memory_atomic_wait32(
            (int32_t *)address, expected, INT64_C(-1));
        if (result != 0 && result != 1) return -1;
    }
    return 0;
}

static void notify_one(_Atomic int32_t *address) {
    (void)__builtin_wasm_memory_atomic_notify((int32_t *)address, 1);
}

static void *wait_notify_worker(void *opaque) {
    struct worker_arg *arg = opaque;
    _Atomic int32_t *gate = &gates[arg->index];
    _Atomic int32_t *ack = &acknowledgements[arg->index];
    for (int32_t epoch = 1; epoch <= (int32_t)arg->iterations; ++epoch) {
        if (wait_while_equal(gate, epoch - 1) != 0 ||
            atomic_load_explicit(gate, memory_order_seq_cst) != epoch) {
            arg->error = 1;
            return NULL;
        }
        atomic_store_explicit(ack, epoch, memory_order_seq_cst);
        notify_one(ack);
    }
    arg->result = arg->iterations;
    return NULL;
}

static void *spawn_join_worker(void *opaque) {
    struct worker_arg *arg = opaque;
    arg->result = arg->index + 1;
    return NULL;
}

static int spawn_and_join(
    uint32_t threads,
    uint64_t iterations,
    void *(*worker)(void *),
    struct worker_arg args[MAX_THREADS]) {
    pthread_t tids[MAX_THREADS];
    for (uint32_t i = 0; i < threads; ++i) {
        pthread_attr_t attr;
        if (pthread_attr_init(&attr) != 0 ||
            pthread_attr_setstack(
                &attr, worker_stacks[i], sizeof worker_stacks[i]) != 0) {
            fprintf(stderr, "pthread_attr setup[%u] failed\n", i);
            return -1;
        }
        args[i] = (struct worker_arg){
            .index = i,
            .iterations = iterations,
            .result = 0,
            .error = 0,
        };
        int rc = pthread_create(&tids[i], &attr, worker, &args[i]);
        (void)pthread_attr_destroy(&attr);
        if (rc != 0) {
            fprintf(stderr, "pthread_create[%u] failed: %d\n", i, rc);
            return -1;
        }
    }
    for (uint32_t i = 0; i < threads; ++i) {
        int rc = pthread_join(tids[i], NULL);
        if (rc != 0) {
            fprintf(stderr, "pthread_join[%u] failed: %d\n", i, rc);
            return -1;
        }
        if (args[i].error != 0) {
            fprintf(stderr, "worker[%u] reported a correctness error\n", i);
            return -1;
        }
    }
    return 0;
}

static uint64_t sum_results(
    uint32_t threads, const struct worker_arg args[MAX_THREADS]) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < threads; ++i) sum += args[i].result;
    return sum;
}

static int run_wait_notify(
    uint32_t threads,
    uint64_t iterations,
    struct worker_arg args[MAX_THREADS],
    uint64_t *checksum) {
    pthread_t tids[MAX_THREADS];
    for (uint32_t i = 0; i < threads; ++i) {
        pthread_attr_t attr;
        if (pthread_attr_init(&attr) != 0 ||
            pthread_attr_setstack(
                &attr, worker_stacks[i], sizeof worker_stacks[i]) != 0) {
            fprintf(stderr, "pthread_attr setup[%u] failed\n", i);
            return -1;
        }
        atomic_store_explicit(&gates[i], 0, memory_order_seq_cst);
        atomic_store_explicit(&acknowledgements[i], 0, memory_order_seq_cst);
        args[i] = (struct worker_arg){
            .index = i,
            .iterations = iterations,
            .result = 0,
            .error = 0,
        };
        int rc = pthread_create(
            &tids[i], &attr, wait_notify_worker, &args[i]);
        (void)pthread_attr_destroy(&attr);
        if (rc != 0) {
            fprintf(stderr, "pthread_create[%u] failed: %d\n", i, rc);
            return -1;
        }
    }

    for (int32_t epoch = 1; epoch <= (int32_t)iterations; ++epoch) {
        for (uint32_t i = 0; i < threads; ++i) {
            atomic_store_explicit(&gates[i], epoch, memory_order_seq_cst);
            notify_one(&gates[i]);
            if (wait_while_equal(&acknowledgements[i], epoch - 1) != 0 ||
                atomic_load_explicit(
                    &acknowledgements[i], memory_order_seq_cst) != epoch) {
                fputs("controller observed an invalid acknowledgement\n", stderr);
                return -1;
            }
        }
    }

    for (uint32_t i = 0; i < threads; ++i) {
        int rc = pthread_join(tids[i], NULL);
        if (rc != 0 || args[i].error != 0) {
            fprintf(stderr, "wait/notify worker[%u] failed: %d\n", i, rc);
            return -1;
        }
    }
    *checksum = sum_results(threads, args);
    return 0;
}

static int parse_workload(const char *name, enum workload *workload) {
    if (strcmp(name, "hot") == 0) {
        *workload = WORKLOAD_HOT;
    } else if (strcmp(name, "atomic") == 0) {
        *workload = WORKLOAD_ATOMIC;
    } else if (strcmp(name, "wait-notify") == 0) {
        *workload = WORKLOAD_WAIT_NOTIFY;
    } else if (strcmp(name, "spawn-join") == 0) {
        *workload = WORKLOAD_SPAWN_JOIN;
    } else {
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    enum workload workload;
    uint32_t threads = 0;
    uint64_t iterations = 0;
    if (argc != 4 || parse_workload(argv[1], &workload) != 0 ||
        parse_u32(argv[2], &threads) != 0 || threads > MAX_THREADS ||
        parse_u64(argv[3], &iterations) != 0 ||
        (workload == WORKLOAD_WAIT_NOTIFY && iterations > INT32_MAX)) {
        fputs(
            "usage: threaded.wasm "
            "hot|atomic|wait-notify|spawn-join THREADS ITERATIONS\n",
            stderr);
        return 2;
    }

    struct worker_arg args[MAX_THREADS];
    uint64_t checksum = 0;
    const char *workload_name = argv[1];
    switch (workload) {
        case WORKLOAD_HOT:
            if (spawn_and_join(threads, iterations, hot_worker, args) != 0)
                return 1;
            checksum = sum_results(threads, args);
            break;
        case WORKLOAD_ATOMIC:
            atomic_store_explicit(&shared_counter, 0, memory_order_seq_cst);
            if (spawn_and_join(threads, iterations, atomic_worker, args) != 0)
                return 1;
            checksum = atomic_load_explicit(
                &shared_counter, memory_order_seq_cst);
            if (checksum != iterations * threads) return 1;
            break;
        case WORKLOAD_WAIT_NOTIFY:
            if (run_wait_notify(
                    threads, iterations, args, &checksum) != 0) {
                return 1;
            }
            break;
        case WORKLOAD_SPAWN_JOIN:
            for (uint64_t round = 0; round < iterations; ++round) {
                if (spawn_and_join(
                        threads, 1, spawn_join_worker, args) != 0) {
                    return 1;
                }
                checksum += sum_results(threads, args);
            }
            break;
    }

    uint64_t operations = iterations * threads;
    return bench_write_result(
        workload_name, threads, iterations, operations, checksum);
}
