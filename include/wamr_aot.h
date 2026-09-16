#ifndef WAMR_AOT_H
#define WAMR_AOT_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wamr_aot_handle wamr_aot_handle;
typedef struct wamr_aot_host_context wamr_aot_host_context;
enum wamr_aot_protection {
    WAMR_AOT_NONE = 0, WAMR_AOT_RW = 1, WAMR_AOT_RX = 2
};
enum wamr_aot_value_kind {
    WAMR_AOT_I32 = 0x7f, WAMR_AOT_I64 = 0x7e,
    WAMR_AOT_F32 = 0x7d, WAMR_AOT_F64 = 0x7c
};
enum wamr_aot_outcome {
    WAMR_AOT_RETURNED = 0, WAMR_AOT_TRAP = 1, WAMR_AOT_EXIT = 2,
    WAMR_AOT_HOST_ERROR = 3, WAMR_AOT_ERROR = 4
};
enum wamr_aot_trap {
    WAMR_AOT_OOB_MEMORY = 0, WAMR_AOT_OOB_TABLE = 1,
    WAMR_AOT_UNREACHABLE = 2, WAMR_AOT_DIV_ZERO = 3,
    WAMR_AOT_INT_OVERFLOW = 4, WAMR_AOT_INVALID_CONVERSION = 5,
    WAMR_AOT_UNSUPPORTED_OPERATION = 6, WAMR_AOT_BAD_HOST_RESULT = 7
};
enum wamr_aot_host_error {
    WAMR_AOT_HOST_OK = 0, WAMR_AOT_HOST_UNSUPPORTED = 1,
    WAMR_AOT_HOST_INVALID_ARGUMENT = 2, WAMR_AOT_HOST_IO = 3,
    WAMR_AOT_HOST_OOM = 4
};
typedef struct {
    uint32_t kind, reserved;
    uint64_t bits;
} wamr_aot_value;
typedef struct {
    uint32_t kind, detail;
    size_t count;
    const char *error_name;
} wamr_aot_result;
typedef struct {
    uint64_t load_ns, instantiate_ns;
    /* Bit0: load_ns measured; bit1: instantiate_ns measured. Other bits zero.
     * Zero duration is meaningful only when its completed bit is set. */
    uint32_t completed, reserved;
} wamr_aot_load_timings;

/* Config and import descriptors must outlive the instance. All operations on
 * one instance must be serialized. Never destroy an active instance.
 * The application owns page-table management. Reserve returns inaccessible
 * 4096-aligned virtual space; commit atomically enables RW zero-filled pages;
 * protect applies exactly NONE/RW/RX; unmap infallibly releases the complete
 * reservation. Mapping callbacks return 0 on success, nonzero on failure.
 * There is no permission downgrade fallback, RWX mapping, or allocator fallback.
 * The application must preserve x86 SSE state and enable SSE before executing.
 */
typedef struct {
    void *context;
    void *(*alloc)(void *, size_t size, size_t alignment);
    void (*free)(void *, void *, size_t size, size_t alignment);
    void *(*reserve)(void *, size_t);
    int (*commit)(void *, void *, size_t);
    int (*protect)(void *, void *, size_t, uint32_t protection);
    void (*unmap)(void *, void *, size_t);
    int (*monotonic_ns)(void *, uint64_t *);
    uint32_t max_memory_pages, max_table_elements;
} wamr_aot_config;
typedef struct {
    const uint8_t *module;
    size_t module_len;
    const uint8_t *name;
    size_t name_len;
    const uint8_t *params;
    size_t param_count;
    const uint8_t *results;
    size_t result_count;
    void *context;
    uint32_t (*callback)(void *, wamr_aot_host_context *,
                        const wamr_aot_value *, size_t, wamr_aot_value *, size_t);
} wamr_aot_import;

uint32_t wamr_aot_contract_version(void);
/* Trusted matching-wamrc artifacts only, not arbitrary native code. Input bytes
 * are copied; out is null on failure. Start is explicit and runs at most once.
 * Limits are mandatory: zero permits zero pages/elements, not "unlimited". */
wamr_aot_result wamr_aot_load(const wamr_aot_config *, const uint8_t *, size_t,
                            const wamr_aot_import *, size_t, wamr_aot_handle **out);
/* Uses the configured monotonic clock at real internal phase boundaries.
 * Clock failure (including backwards time/saturation) fails load and cleans up.
 * Results cover the shared loader/instance implementation, not C argument
 * adaptation or caller workload-specific WASI-context construction. */
wamr_aot_result wamr_aot_load_timed(const wamr_aot_config *, const uint8_t *, size_t,
                                  const wamr_aot_import *, size_t,
                                  wamr_aot_handle **out, wamr_aot_load_timings *);
void wamr_aot_destroy(wamr_aot_handle *);
wamr_aot_result wamr_aot_start(wamr_aot_handle *);
wamr_aot_result wamr_aot_call(wamr_aot_handle *, const uint8_t *name, size_t,
                            const wamr_aot_value *, size_t, wamr_aot_value *, size_t);
uint8_t *wamr_aot_memory(wamr_aot_handle *, size_t *length);
uint8_t *wamr_aot_host_memory(wamr_aot_host_context *, size_t *length);
int wamr_aot_host_clock(wamr_aot_host_context *, uint64_t *nanoseconds);
/* Records a terminal outcome. Return normally from the host callback to allow
 * its cleanup to run; the dispatcher then unwinds guest frames without signals
 * or process exit. It never invokes the host callback again in that call. */
void wamr_aot_host_exit(wamr_aot_host_context *, uint32_t);

#ifdef __cplusplus
}
#endif
#endif
