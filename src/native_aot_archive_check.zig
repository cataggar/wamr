//! Link-only consumer of the real static archive; not an application entry.
extern fn wamr_aot_contract_version() u32;

export fn wamr_native_archive_pie_entry() u32 {
    return wamr_aot_contract_version();
}
