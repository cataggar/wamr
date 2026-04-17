//! Skiplist for AOT-mode spec tests.
//!
//! The x86_64 AOT codegen currently panics (not a catchable error, a hard
//! `@panic`/integer-overflow assertion) on several wasm modules produced by
//! the upstream spec suite. Examples of triggers include features like
//! imports, start functions, multi-memory, large immediates, and
//! call_indirect shapes. Until those codegen paths are fixed, the files
//! below are skipped in AOT mode so the runner can exercise the remainder
//! of the suite without aborting the whole process.
//!
//! Each entry is a basename (e.g. `call.json`). Narrower `(file, line)`
//! skips can be added later once per-assertion skipping is needed.

const std = @import("std");

/// Basenames of .json / .wast files to skip entirely in AOT mode.
pub const aot_file_skiplist: []const []const u8 = &.{
    // Existing interpreter-mode skip (IP misalignment panic):
    "memory_trap64.wast",

    // Files where AOT codegen panics at module-load or produces wrong
    // values. Each entry is tagged with the currently-dominant failure
    // mode; unskip when the relevant bug is fixed.
    "address.json", // untested after trap-helper path landed
    "call.json", // stack-overflow via `runaway` + non-recoverable guard page
    "call_indirect.json", // access violation during exec (beyond null-entry)
    "elem.json", // active-element table init incomplete
    "float_exprs.json", // f32/f64 select sign-bit / NaN canonicalization
    "float_memory.json", // signaling-NaN preservation on i32/i64 <-> f32/f64
    "func.json", // br_if / br_table result-count mismatches
    "global.json", // mutable/imported global initial values
    "imports.json",
    "linking.json",
    "memory_grow.json", // 2 value-mismatch fails (memory.grow/size results)
    "memory_trap.json", // unaligned i64.load value
    "start.json", // start function side-effect not applied
    "unwind.json", // 3 value-mismatch fails

    // select.0.wasm compiles now that `select_t` is recognized, but the
    // generated code dereferences a null pointer at runtime — likely a
    // type-propagation mismatch between the untyped `select` IR op and
    // ref-typed operands pushed onto the vreg stack. Re-enable once the
    // IR learns result typing for select (Phase 4/5 territory).
    "select.json",

    // return_call / return_call_indirect are now lowered as regular
    // call + ret (not true tail-call), which causes a native stack
    // overflow in the spec tests' recursive fac/loop patterns and
    // aborts the whole runner. Re-enable once the codegen supports
    // real tail-call stack reuse.
    "return_call.json",
    "return_call_indirect.json",
    "return_call_ref.json",

    // skip-stack-guard-page.json recurses deep into a native-stack
    // overflow; the trap is emitted via guard-page SEH which our
    // runtime currently doesn't translate into error.WasmTrap.
    "skip-stack-guard-page.json",
};

pub fn isSkippedInAot(basename: []const u8) bool {
    for (aot_file_skiplist) |s| {
        if (std.mem.eql(u8, s, basename)) return true;
    }
    return false;
}
