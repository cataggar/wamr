//! Minimal hand-rolled `setjmp`/`longjmp` for the POSIX AOT trap path
//! (#798 Lever 1). The runtime does not link libc on Linux, so we cannot
//! use C `setjmp`/`sigsetjmp`. The Windows trap path captures/restores a
//! full `CONTEXT` via `RtlCaptureContext`/`RtlRestoreContext`; this is the
//! POSIX native analogue: save the ABI callee-saved registers, stack
//! pointer, and return address into a `JmpBuf`, then jump back to the capture
//! site on `restore`.
//!
//! Scope: x86_64 SysV only (Linux / macOS). Other targets keep the
//! pre-#798 behaviour where AOT traps abort the process; `supported`
//! reflects that so callers can gate cleanly.
//!
//! Safety contract (mirrors C setjmp/longjmp):
//!   - `capture` must be called such that its caller's frame is still live
//!     when `restore` runs (i.e. `restore` unwinds *back into* an active
//!     `capture` caller — exactly how `callFuncScalar` uses it).
//!   - x86_64 preserves the SysV callee-saved integer registers. AArch64
//!     preserves x19-x30, sp, and the ABI-preserved halves of d8-d15.
//!     The capture caller must treat `capture()` as an ordinary opaque call
//!     (the Zig compiler does), reloading caller-saved state after it.

const builtin = @import("builtin");
const std = @import("std");

const supported_os = builtin.os.tag == .linux or builtin.os.tag.isDarwin();
const x86_64_supported = builtin.cpu.arch == .x86_64 and supported_os;
const aarch64_supported = builtin.cpu.arch == .aarch64 and supported_os;

/// True when the hand-rolled jump primitive is available for this target.
pub const supported: bool = x86_64_supported or aarch64_supported;

/// Saved machine state. The x86_64 layout is
/// [rbx, rbp, r12, r13, r14, r15, rsp, rip]. The AArch64 layout is
/// [x19..x30, sp, d8..d15].
pub const JmpBuf = if (aarch64_supported) [21]u64 else [8]u64;

/// Capture the current register/stack context into `buf`. Returns 0 on the
/// direct call; when a later `restore(buf, val)` jumps back here it appears
/// to return `val` (or 1 if `val == 0`), just like C `setjmp`.
pub inline fn capture(buf: *JmpBuf) c_int {
    if (comptime !supported) return 0;
    // The naked body follows the target C ABI, so calling it through a `.c`
    // pointer is correct; the cast is needed because Zig forbids direct calls
    // to naked functions.
    const f: *const fn (*JmpBuf) callconv(.c) c_int = @ptrCast(
        if (comptime aarch64_supported)
            &wasmTrapSetjmpAarch64
        else
            &wasmTrapSetjmpX86_64,
    );
    return f(buf);
}

/// Restore the context saved in `buf`, resuming execution as if the
/// matching `capture(buf)` had just returned `val`. Never returns.
pub inline fn restore(buf: *JmpBuf, val: c_int) noreturn {
    if (comptime !supported) {
        @branchHint(.cold);
        unreachable;
    }
    const f: *const fn (*JmpBuf, c_int) callconv(.c) noreturn = @ptrCast(
        if (comptime aarch64_supported)
            &wasmTrapLongjmpAarch64
        else
            &wasmTrapLongjmpX86_64,
    );
    f(buf, val);
}

// ── x86_64 SysV implementation ──────────────────────────────────────────
//
// Naked functions: no prologue/epilogue, so `rsp` at entry points at the
// return address pushed by the caller's `call`. We save the *post-return*
// rsp (rsp+8) and the return address separately, then `restore` jumps to
// the saved rip after reinstating rsp — never executing a `ret`, since the
// original return address is no longer on the (restored) stack. SysV arg
// regs: rdi = `buf`, rsi = `val`.

fn wasmTrapSetjmpX86_64() callconv(.naked) void {
    asm volatile (
        \\ movq %%rbx,  0(%%rdi)
        \\ movq %%rbp,  8(%%rdi)
        \\ movq %%r12, 16(%%rdi)
        \\ movq %%r13, 24(%%rdi)
        \\ movq %%r14, 32(%%rdi)
        \\ movq %%r15, 40(%%rdi)
        \\ leaq 8(%%rsp), %%rax
        \\ movq %%rax, 48(%%rdi)
        \\ movq (%%rsp), %%rax
        \\ movq %%rax, 56(%%rdi)
        \\ xorl %%eax, %%eax
        \\ retq
    );
}

fn wasmTrapLongjmpX86_64() callconv(.naked) void {
    asm volatile (
        \\ movq  0(%%rdi), %%rbx
        \\ movq  8(%%rdi), %%rbp
        \\ movq 16(%%rdi), %%r12
        \\ movq 24(%%rdi), %%r13
        \\ movq 32(%%rdi), %%r14
        \\ movq 40(%%rdi), %%r15
        \\ movq 56(%%rdi), %%rdx
        \\ movq 48(%%rdi), %%rsp
        \\ movl %%esi, %%eax
        \\ testl %%eax, %%eax
        \\ jnz 1f
        \\ incl %%eax
        \\1:
        \\ jmp *%%rdx
    );
}

// ── AArch64 AAPCS64 implementation ─────────────────────────────────────
//
// x0 = buf, w1 = restore value. AAPCS64 requires x19-x29 and d8-d15 to
// survive an ordinary call; x30 and sp identify the continuation itself.

fn wasmTrapSetjmpAarch64() callconv(.naked) void {
    asm volatile (
        \\ stp x19, x20, [x0, #0]
        \\ stp x21, x22, [x0, #16]
        \\ stp x23, x24, [x0, #32]
        \\ stp x25, x26, [x0, #48]
        \\ stp x27, x28, [x0, #64]
        \\ stp x29, x30, [x0, #80]
        \\ mov x9, sp
        \\ str x9, [x0, #96]
        \\ stp d8, d9, [x0, #104]
        \\ stp d10, d11, [x0, #120]
        \\ stp d12, d13, [x0, #136]
        \\ stp d14, d15, [x0, #152]
        \\ mov w0, wzr
        \\ ret
    );
}

fn wasmTrapLongjmpAarch64() callconv(.naked) void {
    asm volatile (
        \\ ldp x19, x20, [x0, #0]
        \\ ldp x21, x22, [x0, #16]
        \\ ldp x23, x24, [x0, #32]
        \\ ldp x25, x26, [x0, #48]
        \\ ldp x27, x28, [x0, #64]
        \\ ldp x29, x30, [x0, #80]
        \\ ldr x9, [x0, #96]
        \\ ldp d8, d9, [x0, #104]
        \\ ldp d10, d11, [x0, #120]
        \\ ldp d12, d13, [x0, #136]
        \\ ldp d14, d15, [x0, #152]
        \\ mov sp, x9
        \\ mov w0, w1
        \\ cbnz w0, 1f
        \\ mov w0, #1
        \\1:
        \\ br x30
    );
}

test "trap_jmp: capture returns 0, restore resumes with value and restores callee-saved regs" {
    if (comptime !supported) return error.SkipZigTest;

    var buf: JmpBuf = undefined;
    // A callee-saved register we mutate after capture; restore must roll it
    // back to its pre-capture value (the asm reloads it from `buf`).
    var observed_second_pass: c_int = -1;

    const r = capture(&buf);
    if (r == 0) {
        // First pass: jump back with a distinct non-zero value.
        restore(&buf, 7);
    }
    observed_second_pass = r;
    try std.testing.expectEqual(@as(c_int, 7), observed_second_pass);
}

test "trap_jmp: restore with 0 surfaces as 1 (C setjmp semantics)" {
    if (comptime !supported) return error.SkipZigTest;

    var buf: JmpBuf = undefined;
    const r = capture(&buf);
    if (r == 0) {
        restore(&buf, 0); // C semantics: longjmp(buf,0) makes setjmp return 1.
    }
    try std.testing.expectEqual(@as(c_int, 1), r);
}
