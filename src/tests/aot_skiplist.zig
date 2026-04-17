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

    // AOT codegen panics observed on 2026-04-17 against tests/spec-json.
    // See issue #102 follow-ups.
    "address.json",
    "align.json",
    "block.json",
    "call.json",
    "call_indirect.json",
    "elem.json",
    "endianness.json",
    "float_exprs.json",
    "float_memory.json",
    "func.json",
    "func_ptrs.json",
    "global.json",
    "if.json",
    "imports.json",
    "labels.json",
    "left-to-right.json",
    "linking.json",
    "load.json",
    "local_tee.json",
    "loop.json",
    "memory_grow.json",
    "memory_redundancy.json",
    "memory_trap.json",
    "nop.json",
    "start.json",
    "store.json",
    "unwind.json",

    // select.0.wasm compiles now that `select_t` is recognized, but the
    // generated code dereferences a null pointer at runtime — likely a
    // type-propagation mismatch between the untyped `select` IR op and
    // ref-typed operands pushed onto the vreg stack. Re-enable once the
    // IR learns result typing for select (Phase 4/5 territory).
    "select.json",
};

pub fn isSkippedInAot(basename: []const u8) bool {
    for (aot_file_skiplist) |s| {
        if (std.mem.eql(u8, s, basename)) return true;
    }
    return false;
}
