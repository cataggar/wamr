#include "kernel.h"
#include "output.h"

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_THREADS 8
#define THREAD_STACK_BYTES (16 * 1024)
#define CALIBRATION_EPOCHS 5
#define WORK_EPOCH (CALIBRATION_EPOCHS + 1)

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
static _Atomic uint32_t ready_count;
static _Atomic uint32_t completed_count;
static _Atomic int32_t phase;
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

static int wait_while_equal(_Atomic int32_t *address, int32_t expected) {
    while (atomic_load_explicit(address, memory_order_seq_cst) == expected) {
        int result = __builtin_wasm_memory_atomic_wait32(
            (int32_t *)address, expected, INT64_C(-1));
        if (result != 0 && result != 1) return -1;
    }
    return 0;
}

static void notify_all(_Atomic int32_t *address) {
    (void)__builtin_wasm_memory_atomic_notify(
        (int32_t *)address, INT32_MAX);
}

static void notify_count(_Atomic uint32_t *address) {
    (void)__builtin_wasm_memory_atomic_notify(
        (int32_t *)address, INT32_MAX);
}

static int wait_for_count(_Atomic uint32_t *address, uint32_t expected) {
    while (1) {
        uint32_t current =
            atomic_load_explicit(address, memory_order_seq_cst);
        if (current >= expected) return 0;
        int result = __builtin_wasm_memory_atomic_wait32(
            (int32_t *)address, (int32_t)current, INT64_C(-1));
        if (result != 0 && result != 1) return -1;
    }
}

static void signal_count(_Atomic uint32_t *address) {
    atomic_fetch_add_explicit(address, 1, memory_order_seq_cst);
    notify_count(address);
}

static int worker_barrier_prelude(void) {
    signal_count(&ready_count);
    for (int32_t epoch = 1; epoch <= CALIBRATION_EPOCHS; ++epoch) {
        if (wait_while_equal(&phase, epoch - 1) != 0) return -1;
        signal_count(&completed_count);
    }
    return wait_while_equal(&phase, CALIBRATION_EPOCHS);
}

static void worker_complete(struct worker_arg *arg) {
    signal_count(&completed_count);
    if (arg->error != 0) notify_count(&completed_count);
}

static void *hot_worker(void *opaque) {
    struct worker_arg *arg = opaque;
    uint64_t warmup_iterations = arg->iterations / 64;
    if (warmup_iterations < 1024) warmup_iterations = 1024;
    (void)bench_hot_kernel(bench_seed(arg->index), warmup_iterations);
    if (worker_barrier_prelude() != 0) {
        arg->error = 1;
        worker_complete(arg);
        return NULL;
    }
    arg->result =
        bench_hot_kernel(bench_seed(arg->index), arg->iterations);
    worker_complete(arg);
    return NULL;
}

static void *atomic_worker(void *opaque) {
    struct worker_arg *arg = opaque;
    _Atomic uint64_t warmup = 0;
    uint64_t warmup_iterations = arg->iterations / 64;
    if (warmup_iterations < 1024) warmup_iterations = 1024;
    for (uint64_t i = 0; i < warmup_iterations; ++i)
        atomic_fetch_add_explicit(&warmup, 1, memory_order_relaxed);
    if (worker_barrier_prelude() != 0) {
        arg->error = 1;
        worker_complete(arg);
        return NULL;
    }
    for (uint64_t i = 0; i < arg->iterations; ++i)
        atomic_fetch_add_explicit(
            &shared_counter, 1, memory_order_relaxed);
    arg->result = arg->iterations;
    worker_complete(arg);
    return NULL;
}

static void *wait_notify_worker(void *opaque) {
    struct worker_arg *arg = opaque;
    if (worker_barrier_prelude() != 0) {
        arg->error = 1;
        worker_complete(arg);
        return NULL;
    }
    _Atomic int32_t *gate = &gates[arg->index];
    _Atomic int32_t *ack = &acknowledgements[arg->index];
    for (int32_t epoch = 1; epoch <= (int32_t)arg->iterations; ++epoch) {
        if (wait_while_equal(gate, epoch - 1) != 0 ||
            atomic_load_explicit(gate, memory_order_seq_cst) != epoch) {
            arg->error = 1;
            worker_complete(arg);
            return NULL;
        }
        atomic_store_explicit(ack, epoch, memory_order_seq_cst);
        notify_all(ack);
    }
    arg->result = arg->iterations;
    worker_complete(arg);
    return NULL;
}

static void *spawn_join_worker(void *opaque) {
    struct worker_arg *arg = opaque;
    arg->result = arg->index + 1;
    return NULL;
}

static int start_workers(
    uint32_t threads,
    uint64_t iterations,
    void *(*worker)(void *),
    struct worker_arg args[MAX_THREADS],
    pthread_t tids[MAX_THREADS]) {
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
    return 0;
}

static int join_workers(
    uint32_t threads,
    pthread_t tids[MAX_THREADS],
    struct worker_arg args[MAX_THREADS]) {
    for (uint32_t i = 0; i < threads; ++i) {
        int rc = pthread_join(tids[i], NULL);
        if (rc != 0) {
            fprintf(stderr, "pthread_join[%u] failed: %d\n", i, rc);
            return -1;
        }
        if (args[i].error != 0) {
            fprintf(stderr, "worker[%u] failed: %d\n", i, args[i].error);
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

static void reset_timed_state(void) {
    atomic_store_explicit(&ready_count, 0, memory_order_seq_cst);
    atomic_store_explicit(&completed_count, 0, memory_order_seq_cst);
    atomic_store_explicit(&phase, 0, memory_order_seq_cst);
}

static int measure_barrier_overhead(
    uint32_t threads, uint64_t *minimum) {
    if (wait_for_count(&ready_count, threads) != 0) return -1;
    *minimum = UINT64_MAX;
    for (int32_t epoch = 1; epoch <= CALIBRATION_EPOCHS; ++epoch) {
        atomic_store_explicit(
            &completed_count, 0, memory_order_seq_cst);
        uint64_t start = 0;
        uint64_t end = 0;
        if (bench_now_ns(&start) != 0) return -1;
        atomic_store_explicit(&phase, epoch, memory_order_seq_cst);
        notify_all(&phase);
        if (wait_for_count(&completed_count, threads) != 0 ||
            bench_now_ns(&end) != 0 || end <= start) {
            return -1;
        }
        uint64_t elapsed = end - start;
        if (elapsed < *minimum) *minimum = elapsed;
    }
    return 0;
}

static int begin_work(uint64_t *start) {
    atomic_store_explicit(&completed_count, 0, memory_order_seq_cst);
    if (bench_now_ns(start) != 0) return -1;
    atomic_store_explicit(&phase, WORK_EPOCH, memory_order_seq_cst);
    notify_all(&phase);
    return 0;
}

static int finish_work(
    uint32_t threads,
    uint64_t start,
    uint64_t overhead,
    struct bench_timing *timing) {
    uint64_t end = 0;
    if (wait_for_count(&completed_count, threads) != 0 ||
        bench_now_ns(&end) != 0) {
        return -1;
    }
    return bench_finish_timing(start, end, overhead, timing);
}

static int run_timed_workers(
    uint32_t threads,
    uint64_t iterations,
    void *(*worker)(void *),
    struct worker_arg args[MAX_THREADS],
    struct bench_timing *timing) {
    pthread_t tids[MAX_THREADS];
    reset_timed_state();
    if (start_workers(threads, iterations, worker, args, tids) != 0)
        return -1;
    uint64_t overhead = 0;
    uint64_t start = 0;
    if (measure_barrier_overhead(threads, &overhead) != 0 ||
        begin_work(&start) != 0 ||
        finish_work(threads, start, overhead, timing) != 0) {
        return -1;
    }
    return join_workers(threads, tids, args);
}

static int run_wait_notify(
    uint32_t threads,
    uint64_t iterations,
    struct worker_arg args[MAX_THREADS],
    uint64_t *checksum,
    struct bench_timing *timing) {
    pthread_t tids[MAX_THREADS];
    reset_timed_state();
    for (uint32_t i = 0; i < threads; ++i) {
        atomic_store_explicit(&gates[i], 0, memory_order_seq_cst);
        atomic_store_explicit(
            &acknowledgements[i], 0, memory_order_seq_cst);
    }
    if (start_workers(
            threads, iterations, wait_notify_worker, args, tids) != 0) {
        return -1;
    }
    uint64_t overhead = 0;
    uint64_t start = 0;
    if (measure_barrier_overhead(threads, &overhead) != 0 ||
        begin_work(&start) != 0) {
        return -1;
    }
    for (int32_t epoch = 1; epoch <= (int32_t)iterations; ++epoch) {
        for (uint32_t i = 0; i < threads; ++i) {
            atomic_store_explicit(
                &gates[i], epoch, memory_order_seq_cst);
            notify_all(&gates[i]);
            if (wait_while_equal(
                    &acknowledgements[i], epoch - 1) != 0 ||
                atomic_load_explicit(
                    &acknowledgements[i],
                    memory_order_seq_cst) != epoch) {
                return -1;
            }
        }
    }
    if (finish_work(threads, start, overhead, timing) != 0 ||
        join_workers(threads, tids, args) != 0) {
        return -1;
    }
    *checksum = sum_results(threads, args);
    return 0;
}

static int run_spawn_join(
    uint32_t threads,
    uint64_t iterations,
    struct worker_arg args[MAX_THREADS],
    uint64_t *checksum,
    struct bench_timing *timing) {
    pthread_t tids[MAX_THREADS];
    uint64_t overhead = 0;
    uint64_t start = 0;
    uint64_t end = 0;
    if (bench_clock_overhead(&overhead) != 0 ||
        bench_now_ns(&start) != 0) {
        return -1;
    }
    for (uint64_t round = 0; round < iterations; ++round) {
        if (start_workers(
                threads, 1, spawn_join_worker, args, tids) != 0 ||
            join_workers(threads, tids, args) != 0) {
            return -1;
        }
        *checksum += sum_results(threads, args);
    }
    if (bench_now_ns(&end) != 0)
        return -1;
    return bench_finish_timing(start, end, overhead, timing);
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
    struct bench_timing timing;
    uint64_t checksum = 0;
    uint64_t operations = iterations * threads;
    const char *metric_kind = "steady-state-kernel";
    uint64_t timed_loop_backedges = 0;
    switch (workload) {
        case WORKLOAD_HOT:
            if (run_timed_workers(
                    threads, iterations, hot_worker, args, &timing) != 0) {
                return 1;
            }
            checksum = sum_results(threads, args);
            timed_loop_backedges = operations;
            break;
        case WORKLOAD_ATOMIC:
            atomic_store_explicit(
                &shared_counter, 0, memory_order_seq_cst);
            if (run_timed_workers(
                    threads, iterations, atomic_worker, args, &timing) != 0) {
                return 1;
            }
            checksum = atomic_load_explicit(
                &shared_counter, memory_order_seq_cst);
            if (checksum != operations) return 1;
            break;
        case WORKLOAD_WAIT_NOTIFY:
            if (run_wait_notify(
                    threads,
                    iterations,
                    args,
                    &checksum,
                    &timing) != 0) {
                return 1;
            }
            break;
        case WORKLOAD_SPAWN_JOIN:
            metric_kind = "spawn-join-lifecycle";
            if (run_spawn_join(
                    threads,
                    iterations,
                    args,
                    &checksum,
                    &timing) != 0) {
                return 1;
            }
            break;
    }

    return bench_write_result(
        argv[1],
        threads,
        iterations,
        operations,
        checksum,
        metric_kind,
        timed_loop_backedges,
        &timing);
}
