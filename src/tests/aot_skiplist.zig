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
    // "address.json", // FIXED round-6: large memory offset no longer panics
    "call.json", // stack-overflow via runaway (guard page non-recoverable)
    "call_indirect.json", // hangs; signature-mismatch traps not emitted
    // "elem.json", // 7 call-in-table WasmTrap (passive elem + imports)
    // "float_exprs.json", // FIXED round-7: f_le/f_lt/f_eq/f_ne setcc r11 (was rdx)
    // "func.json", // FIXED: function-level br_if/br_table now emit ret
    // "global.json", // FIXED round-6: imported globals + global_get init expr
    "imports.json", // AV during exec (host-import call path)
    "linking.json", // 38 cross-module register/resolution fails
    // "memory_grow.json", // 2 fails: cross-module imported memory (linking)
    // "start.json", // FIXED: emit start section + invoke after instantiate
    // "unwind.json", // FIXED: function-level br_if/br_table now emit ret

    // select.0.wasm compiles now that `select_t` is recognized, but the
    // generated code dereferences a null pointer at runtime — likely a
    // type-propagation mismatch between the untyped `select` IR op and
    // ref-typed operands pushed onto the vreg stack. Re-enable once the
    // IR learns result typing for select (Phase 4/5 territory).
    // "select.json", // 147 pass, 4 call_indirect arg-routing fails (pre-existing)

    // return_call / return_call_indirect / return_call_ref now use true
    // tail-call codegen (JMP into callee with the current frame torn down)
    // for the common case of ≤ register-param-count args + no HRP on stack.
    // Deep recursion in the spec tests no longer overflows the native stack.
    // "return_call.json",
    "return_call_indirect.json", // 75/79 pass; 4 "indirect call type mismatch" asserts (pre-existing, same as call_indirect.json — no dynamic sig check on indirect calls)
    // "return_call_ref.json",

    // skip-stack-guard-page.json recurses deep into a native-stack
    // overflow; the trap is emitted via guard-page SEH which our
    // runtime currently doesn't translate into error.WasmTrap.
    // Infrastructure for recovery (SetThreadStackGuarantee +
    // resetStackGuardPage) exists in runtime.zig but VEH dispatch on
    // an overflowed stack still hangs — needs deeper investigation,
    // likely an alternate stack / exception-dispatch rework.
    "skip-stack-guard-page.json",
};

pub fn isSkippedInAot(basename: []const u8) bool {
    for (aot_file_skiplist) |s| {
        if (std.mem.eql(u8, s, basename)) return true;
    }
    return false;
}

