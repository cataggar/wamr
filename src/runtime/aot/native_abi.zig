//! Compiler/runtime contract for the compiler-free native embedding profile.
//! Keep this leaf independent of both compiler and hosted runtime modules.
pub const magic: u32 = 0x746f6100;
pub const format_version: u32 = 11;
pub const contract_version: u32 = 1;
pub const profile_flag: u32 = 0x554b0001;
/// Same ABI, with mandatory single-threaded fuel at entry/back-edge targets.
pub const fuel_profile_flag: u32 = 0x554b0002;
pub const max_fuel_call_depth: usize = 32;
pub const max_fuel_frame_bytes: usize = 8192;
// Conservative requirements of the x86 backend, including scalar bit counts
// and float rounding. SIMD wasm instructions are not part of this profile.
pub const cpu_features: u64 = 0x1f; // SSE2, SSE4.1, POPCNT, BMI1, LZCNT

pub const VmCtx = extern struct {
    memory_base: usize = 0,
    memory_size: usize = 0,
    globals_ptr: usize = 0,
    host_functions_ptr: usize = 0,
    memory_max_size: usize = 0,
    func_table_ptr: usize = 0,
    globals_count: u32 = 0,
    host_functions_count: u32 = 0,
    memory_pages: u32 = 0,
    func_table_len: u32 = 0,
    mem_grow_fn: usize = 0,
    instance_ptr: usize = 0,
    trap_oob_fn: usize = 0,
    trap_unreachable_fn: usize = 0,
    trap_idivz_fn: usize = 0,
    trap_iovf_fn: usize = 0,
    trap_ivc_fn: usize = 0,
    funcptrs_ptr: usize = 0,
    table_grow_fn: usize = 0,
    tables_info_ptr: usize = 0,
    table_init_fn: usize = 0,
    elem_drop_fn: usize = 0,
    sig_table_ptr: usize = 0,
    func_sig_ids_ptr: usize = 0,
    ptr_to_sig_ptr: usize = 0,
    ptr_to_sig_len: u32 = 0,
    _pad_pts: u32 = 0,
    table_set_fn: usize = 0,
    futex_wait32_fn: usize = 0,
    futex_wait64_fn: usize = 0,
    futex_notify_fn: usize = 0,
    mem_fill_fn: usize = 0,
    mem_copy_fn: usize = 0,
    tags_ptr: usize = 0,
    tags_count: u32 = 0,
    _pad_tags: u32 = 0,
    aot_throw_uncaught_fn: usize = 0,
    exception_params: [16]u64 = @splat(0),
    exception_param_count: u32 = 0,
    _pad_exc: u32 = 0,
    wasi_ctx: usize = 0,
    lazy_compile_fn: usize = 0,
    thread_context: usize = 0,
    trap_unaligned_fn: usize = 0,
    cancel_flag: u32 = 0,
    _pad_cancel: u32 = 0,
    cancel_point_fn: usize = 0,
    memory_init_fn: usize = 0,
    data_drop_fn: usize = 0,
    cancel_group_token: u32 = 0,
    _pad_cancel_group: u32 = 0,
};

pub const Table = extern struct {
    ptr: usize = 0,
    len: u32 = 0,
    padding: u32 = 0,
    type_backing_ptr: usize = 0,
};
