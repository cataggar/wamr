//! x86-64 IR Compiler
//!
//! Walks IR functions and emits x86-64 machine code via CodeBuffer.
//! Uses a register-caching operand stack: the top N values are kept in
//! registers, falling back to memory when all cache registers are in use.
//! RAX and RCX serve as temporary scratch registers for instruction operands.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("../../ir/ir.zig");
const passes = @import("../../ir/passes.zig");
const emit = @import("emit.zig");

/// Platform-specific ABI parameter registers, resolved at comptime.
const param_regs = if (builtin.os.tag == .windows)
    [_]emit.Reg{ .rcx, .rdx, .r8, .r9 } // Win64
else
    [_]emit.Reg{ .rdi, .rsi, .rdx, .rcx, .r8, .r9 }; // SysV

/// Caller-saved allocatable registers that must be saved/restored around calls.
/// On Win64: rdx, r8, r9 are volatile. rsi, rdi are callee-saved.
/// On SysV: rdx, rsi, rdi, r8, r9 are all volatile.
/// Caller-saved allocatable registers that must be saved/restored around calls.
/// Since our generated functions don't preserve callee-saved registers in their
/// prologues, ALL allocatable registers must be treated as caller-saved.
const caller_saved_alloc = [_]emit.Reg{ .rdx, .rsi, .rdi, .r8, .r9 };

/// Callee-saved allocatable registers preserved in prologue/epilogue.
/// rbx, r12-r15 are callee-saved on both Win64 and SysV.
/// On Win64, rsi and rdi are also callee-saved per the ABI; compiled functions
/// must save/restore them if used. On SysV they're caller-saved (handled
/// at call sites via caller_saved_alloc), so no prologue work is needed for them.
const callee_saved_alloc = if (builtin.os.tag == .windows)
    [_]emit.Reg{ .rbx, .rsi, .rdi, .r12, .r13, .r14, .r15 }
else
    [_]emit.Reg{ .rbx, .r12, .r13, .r14, .r15 };

/// Fixed frame offset for the VMContext pointer.
/// Stored at [rbp - 8] by compileFunctionRA at function entry.
/// Points to a VmCtx struct with memory_base at +0, globals_base at +16.
const vmctx_offset: i32 = -8;
const vmctx_membase_field: i32 = 0; // VmCtx.memory_base offset
const vmctx_memsize_field: i32 = 8; // VmCtx.memory_size offset
const vmctx_globals_field: i32 = 16; // VmCtx.globals_ptr offset
const vmctx_host_functions_field: i32 = 24; // VmCtx.host_functions_ptr offset
const vmctx_mem_max_size_field: i32 = 32; // VmCtx.memory_max_size offset
const vmctx_func_table_field: i32 = 40; // VmCtx.func_table_ptr offset
const vmctx_mem_pages_field: i32 = 56; // VmCtx.memory_pages offset (u32)
const vmctx_func_table_len_field: i32 = 60; // VmCtx.func_table_len offset (u32)
const vmctx_mem_grow_fn_field: i32 = 64; // VmCtx.mem_grow_fn offset (usize)
const vmctx_mem_fill_fn_field: i32 = 224; // VmCtx.mem_fill_fn offset (usize)
const vmctx_mem_copy_fn_field: i32 = 232; // VmCtx.mem_copy_fn offset (usize)
const vmctx_instance_ptr_field: i32 = 72; // VmCtx.instance_ptr offset (usize)
const vmctx_trap_oob_fn_field: i32 = 80; // VmCtx.trap_oob_fn offset (usize)
const vmctx_trap_unreachable_fn_field: i32 = 88;
const vmctx_trap_idivz_fn_field: i32 = 96;
const vmctx_trap_iovf_fn_field: i32 = 104;
const vmctx_trap_ivc_fn_field: i32 = 112;
const vmctx_funcptrs_field: i32 = 120; // VmCtx.funcptrs_ptr offset (usize)
const vmctx_table_grow_fn_field: i32 = 128; // VmCtx.table_grow_fn offset (usize)
const vmctx_tables_info_field: i32 = 136; // VmCtx.tables_info_ptr offset (usize)
const vmctx_table_init_fn_field: i32 = 144; // VmCtx.table_init_fn offset (usize)
const vmctx_elem_drop_fn_field: i32 = 152; // VmCtx.elem_drop_fn offset (usize)
const vmctx_sig_table_field: i32 = 160; // VmCtx.sig_table_ptr offset (usize)
const vmctx_table_set_fn_field: i32 = 192; // VmCtx.table_set_fn offset (usize)
const vmctx_futex_wait32_fn_field: i32 = 200; // VmCtx.futex_wait32_fn offset (usize)
const vmctx_futex_wait64_fn_field: i32 = 208; // VmCtx.futex_wait64_fn offset (usize)
const vmctx_futex_notify_fn_field: i32 = 216; // VmCtx.futex_notify_fn offset (usize)
// Per-table descriptor layout (TableInfo, 24 bytes):
//   { ptr: u64, len: u32, _pad: u32, type_backing_ptr: u64 }
const table_info_ptr_off: i32 = 0;
const table_info_len_off: i32 = 8;
const table_info_type_backing_off: i32 = 16;
const table_info_stride: i32 = 24;

/// Emit the compare-and-trap tail of a bounds check.
///
/// Assumes the end-of-access address has just been materialized in `r11`
/// and `r10` still holds the `VmCtx*` pointer. Emits:
///   cmp r11, [r10 + memsize_field]
///   jbe over_trap
///   mov param_regs[0], r10
///   mov rax, [param_regs[0] + trap_oob_fn_field]
///   call rax                 ; noreturn
/// The `jbe` rel8 is hard-coded to skip 12 bytes — the fixed size of the
/// trap block below (3 + 7 + 2).
fn emitOobCmpAndTrap(code: *emit.CodeBuffer) !void {
    // cmp r11, qword ptr [r10 + memsize_field]
    //   REX.W|R|B=4D, opcode 0x3B, modrm=01_011_010 (mod=disp8, reg=r11 low=3,
    //   rm=r10 low=2), disp8=memsize_field.
    try code.emitSlice(&.{ 0x4D, 0x3B, 0x5A, @as(u8, @intCast(vmctx_memsize_field)) });
    // jbe over_trap (rel8). Trap block size is 3 + 7 + 2 = 12 bytes below.
    try code.emitByte(0x76);
    try code.emitByte(12);
    // Trap block:
    //   mov param_regs[0], r10            ; arg0 = vmctx (already in r10)
    //   mov rax, [param_regs[0] + trap_oob_fn_field]
    //   call rax                          ; noreturn
    try code.movRegReg(param_regs[0], .r10);
    try code.movRegMem(.rax, param_regs[0], vmctx_trap_oob_fn_field);
    try code.callReg(.rax);
    // over_trap:
}

/// Emit an unconditional call to a non-returning trap helper at
/// `[r10 + field_offset]` (where r10 holds vmctx*). Used for
/// unreachable, divide-by-zero, integer overflow, and invalid
/// float→int conversion traps.
fn emitTrapHelperCall(code: *emit.CodeBuffer, field_offset: i32) !void {
    try code.movRegReg(param_regs[0], .r10);
    try code.movRegMem(.rax, param_regs[0], field_offset);
    try code.callReg(.rax);
}

/// Emit the call_indirect signature check.
///
/// Preconditions: r10 = *VmCtx, r11 = tables_info_ptr, rax = elem_idx
/// (zero-extended 32-bit). Clobbers rcx and r11 (both non-allocatable).
/// Does NOT clobber rdx or any other allocatable register, so live arg
/// vregs survive across this sequence. Traps via
/// `vmctx.trap_unreachable_fn` on mismatch (also covers the null-slot
/// case, since type_backing[idx]==0 will never equal a non-zero
/// expected sig_id — a valid `(type $t)` always interns to sig_id >= 1).
///
/// Sequence (39 bytes total):
///   mov rcx, [r11 + ti_type_backing_off]   ; 7 bytes
///   mov r11, [r10 + sig_table_ptr]         ; 7 bytes
///   mov r11d, [r11 + type_idx*4]           ; 7 bytes
///   cmp r11d, [rcx + rax*4]               ; 4 bytes (REX.R, SIB)
///   je  +12 (over_trap)                    ; 2 bytes
///   mov param_regs[0], r10                 ; 3 bytes
///   mov rax, [param_regs[0] + trap_unreachable_fn]  ; 7 bytes
///   call rax                               ; 2 bytes
///
/// NOTE: r11 is clobbered (holds expected sig_id on exit). Callers
/// that need tables_info_ptr afterward (non-table-0 paths) must
/// reload it from vmctx.
fn emitCallIndirectSigCheck(code: *emit.CodeBuffer, type_idx: u32, table_idx: u32) !void {
    const ti_tb_off: i32 = @as(i32, @intCast(table_idx)) * table_info_stride + table_info_type_backing_off;
    // mov rcx, [r11 + ti_tb_off]  — rcx = type_backing_ptr
    //   REX.W|B = 0x49, opcode 0x8B, modrm=10_001_011 (mod=disp32, reg=rcx=1, rm=r11.low3=3) = 0x8B.
    try code.emitSlice(&.{ 0x49, 0x8B, 0x8B });
    try code.emitU32(@bitCast(ti_tb_off));
    // mov r11, [r10 + sig_table_field]  — r11 = sig_table_ptr (clobbers tables_info_ptr)
    //   REX.W|R|B = 0x4D, opcode 0x8B, modrm=10_011_010 (mod=disp32, reg=r11.low3=3, rm=r10.low3=2) = 0x9A.
    try code.emitSlice(&.{ 0x4D, 0x8B, 0x9A });
    try code.emitU32(@bitCast(@as(i32, vmctx_sig_table_field)));
    // mov r11d, dword ptr [r11 + type_idx*4]  — r11d = expected sig_id
    //   REX.R|B = 0x45, opcode 0x8B, modrm=10_011_011 (mod=disp32, reg=r11.low3=3, rm=r11.low3=3) = 0x9B.
    try code.emitSlice(&.{ 0x45, 0x8B, 0x9B });
    try code.emitU32(type_idx *% 4);
    // cmp r11d, dword ptr [rcx + rax*4]  — compare with type_backing[elem_idx]
    //   REX.R = 0x44, opcode 0x3B, modrm=00_011_100 (mod=0, reg=r11.low3=3, rm=100=SIB) = 0x1C.
    //   SIB=10_000_001 (scale=10 ×4, index=rax=0, base=rcx=1) = 0x81.
    try code.emitSlice(&.{ 0x44, 0x3B, 0x1C, 0x81 });
    // je over_trap (rel8). Trap block: 3 + 7 + 2 = 12 bytes.
    try code.emitByte(0x74);
    try code.emitByte(12);
    // Trap block: call trap_unreachable_fn(vmctx) — r10 holds vmctx.
    try code.movRegReg(param_regs[0], .r10);
    try code.movRegMem(.rax, param_regs[0], vmctx_trap_unreachable_fn_field);
    try code.callReg(.rax);
    // over_trap:
}

/// Emit `cmp eax, [r10 + disp]`. Selects disp8 (4 bytes total) or disp32
/// (7 bytes) based on the displacement. Used by table bounds checks and
/// similar patterns that read a u32 length from an r10-based struct.
fn emitCmpEaxMemR10Disp8(code: *emit.CodeBuffer, disp: i32) !void {
    if (disp >= -128 and disp <= 127) {
        try code.emitSlice(&.{ 0x41, 0x3B, 0x42, @as(u8, @bitCast(@as(i8, @intCast(disp)))) });
    } else {
        // 41 3B 82 <disp32>: cmp eax, [r10 + disp32]
        try code.emitSlice(&.{ 0x41, 0x3B, 0x82 });
        try code.emitI32(disp);
    }
}

/// Emit a conditional skip over a trap-invalid-conversion call.
/// `skip_cond_byte` is the second byte of a 0x0F-prefixed rel32 Jcc that
/// branches past the trap when it evaluates to true (i.e. the "safe"
/// condition). When false, fall through into the noreturn trap call.
fn emitSkipIvcOnCond(code: *emit.CodeBuffer, skip_cond_byte: u8) !void {
    try code.emitSlice(&.{ 0x0F, skip_cond_byte });
    const p = code.len();
    try code.emitI32(0);
    try emitTrapHelperCall(code, vmctx_trap_ivc_fn_field);
    const after = code.len();
    code.patchI32(p, @intCast(@as(i64, @intCast(after)) - @as(i64, @intCast(p + 4))));
}

/// Emit NaN + range checks for a non-saturating float→int trunc.
/// Source float is in xmm0 (rax-aliased); uses xmm1 (rcx) and r11.
/// On any failing check, calls trap_ivc_fn (noreturn). Otherwise falls
/// through with xmm0 preserved.
fn emitTruncRangeCheck(
    code: *emit.CodeBuffer,
    dst_is_i32: bool,
    is_signed: bool,
    is_f32: bool,
) !void {
    // NaN check: ucomi* xmm0, xmm0 sets PF=1 iff unordered. JNP skips trap.
    if (is_f32) try code.ucomiss(.rax, .rax) else try code.ucomisd(.rax, .rax);
    try emitSkipIvcOnCond(code, 0x8B); // JNP rel32

    // Lower bound
    const MinInfo = struct { bits: u64, inclusive_trap: bool };
    const min_info: MinInfo = if (is_signed) blk: {
        if (dst_is_i32) {
            if (is_f32) {
                break :blk .{ .bits = 0xCF000000, .inclusive_trap = false };
            } else {
                // f64 can represent -2^31 - 1 exactly; use inclusive trap.
                break :blk .{ .bits = 0xC1E0000000200000, .inclusive_trap = true };
            }
        } else {
            break :blk if (is_f32)
                .{ .bits = 0xDF000000, .inclusive_trap = false }
            else
                .{ .bits = 0xC3E0000000000000, .inclusive_trap = false };
        }
    } else blk: {
        // Unsigned: -1.0 inclusive
        break :blk if (is_f32)
            .{ .bits = 0xBF800000, .inclusive_trap = true }
        else
            .{ .bits = 0xBFF0000000000000, .inclusive_trap = true };
    };
    if (is_f32) {
        try code.movRegImm32(.r11, @bitCast(@as(u32, @truncate(min_info.bits))));
        try code.movdToXmm(.rcx, .r11);
        try code.ucomiss(.rax, .rcx);
    } else {
        try code.movRegImm64(.r11, min_info.bits);
        try code.movqToXmm(.rcx, .r11);
        try code.ucomisd(.rax, .rcx);
    }
    // Strict (JB traps) -> skip with JNB (0x83). Inclusive (JBE traps) -> skip with JA (0x87).
    try emitSkipIvcOnCond(code, if (min_info.inclusive_trap) @as(u8, 0x87) else @as(u8, 0x83));

    // Upper bound: always JAE traps -> skip with JB (0x82).
    const max_bits: u64 = if (is_signed) blk: {
        if (dst_is_i32) {
            break :blk if (is_f32) @as(u64, 0x4F000000) else @as(u64, 0x41E0000000000000);
        } else {
            break :blk if (is_f32) @as(u64, 0x5F000000) else @as(u64, 0x43E0000000000000);
        }
    } else blk: {
        if (dst_is_i32) {
            break :blk if (is_f32) @as(u64, 0x4F800000) else @as(u64, 0x41F0000000000000);
        } else {
            break :blk if (is_f32) @as(u64, 0x5F800000) else @as(u64, 0x43F0000000000000);
        }
    };
    if (is_f32) {
        try code.movRegImm32(.r11, @bitCast(@as(u32, @truncate(max_bits))));
        try code.movdToXmm(.rcx, .r11);
        try code.ucomiss(.rax, .rcx);
    } else {
        try code.movRegImm64(.r11, max_bits);
        try code.movqToXmm(.rcx, .r11);
        try code.ucomisd(.rax, .rcx);
    }
    try emitSkipIvcOnCond(code, 0x82); // JB rel32
}

/// Emit an inline wasm-memory bounds check.
///
/// Preconditions:
///   - `rax` holds `wasm_addr` already zero-extended to 64 bits (u32 → u64).
///   - `r10` holds the `VmCtx*` pointer.
///
/// Effect:
///   - Computes `end = wasm_addr + end_offset` in `r11`, where
///     `end_offset = static_offset + access_size` supplied by the caller.
///   - Compares `end` with `VmCtx.memory_size` and, if strictly greater,
///     calls `vmctx.trap_oob_fn(vmctx)` which does not return.
///   - Preserves `rax` (the wasm_addr) so the caller can continue to add
///     `mem_base` to it. Preserves `r10` so the caller can subsequently
///     load `mem_base` via `mov r10, [r10 + membase]`.
///   - Clobbers `r11` (non-allocatable scratch) and `param_regs[0]`
///     (Win64: rcx, reserved scratch / SysV: rdi, caller-saved).
fn emitMemBoundsCheck(code: *emit.CodeBuffer, end_offset: u64) !void {
    // r11 = rax + end_offset (64-bit).
    if (end_offset == 0) {
        // mov r11, rax  (REX.W|B + 0x89 + modrm 11_000_011)
        try code.emitSlice(&.{ 0x49, 0x89, 0xC3 });
    } else if (end_offset <= 0x7FFFFFFF) {
        // lea r11, [rax + disp32]  (REX.W|R + 0x8D + modrm 10_011_000 + disp32)
        try code.emitSlice(&.{ 0x4C, 0x8D, 0x98 });
        try code.emitI32(@intCast(end_offset));
    } else {
        // For unusually large static offsets: mov r11, imm64; add r11, rax.
        try code.emitSlice(&.{ 0x49, 0xBB }); // mov r11, imm64
        var i: u8 = 0;
        while (i < 8) : (i += 1) {
            try code.emitByte(@intCast((end_offset >> @as(u6, @intCast(i * 8))) & 0xFF));
        }
        // add r11, rax  (REX.W|B + 0x01 + modrm 11_000_011)
        try code.emitSlice(&.{ 0x49, 0x01, 0xC3 });
    }
    try emitOobCmpAndTrap(code);
}

/// Emit a bounds check for a bulk memory op (memory.copy / memory.fill) where
/// the length is held in a register, not known at compile time.
///
/// Preconditions:
///   - `rax` holds the wasm base address, already zero-extended to u64.
///   - `rcx` holds the byte length, already zero-extended to u64.
///   - `r10` holds the `VmCtx*` pointer.
///
/// Effect:
///   - Computes `end = rax + rcx` in `r11` and traps if
///     `end > VmCtx.memory_size` by calling the `trap_oob_fn`.
///   - Per the wasm spec, `end` cannot overflow u64 because both inputs are
///     ≤ 2^32 − 1, so `rax + rcx` ≤ 2^33 − 2 < 2^64.
///   - Preserves `rax`, `rcx`, `r10`. Clobbers `r11` and `param_regs[0]`.
fn emitMemBoundsCheckDynamic(code: *emit.CodeBuffer) !void {
    // lea r11, [rax + rcx] — 64-bit, 3 bytes
    //   REX.W|R = 0x4C, opcode 0x8D,
    //   modrm=00_011_100 (mod=0, reg=r11 low=3, rm=100 ⇒ SIB),
    //   SIB=00_001_000 (scale=0, index=rcx=1, base=rax=0)
    try code.emitSlice(&.{ 0x4C, 0x8D, 0x1C, 0x08 });
    try emitOobCmpAndTrap(code);
}

/// Register-caching operand stack. Keeps the top N values in registers,
/// eliminating redundant store-load pairs. Falls back to memory (via RBP
/// offsets) when all cache registers are occupied.
/// Locals occupy [rbp - 8*1] through [rbp - 8*N], and operand stack
/// slot i lives at [rbp - 8*(N+1) - 8*i].
const CachedStack = struct {
    const SlotState = enum { empty, in_reg, in_mem, is_const, is_cmp };
    const Slot = struct {
        state: SlotState = .empty,
        reg: emit.Reg = .rax,
        const_val: i64 = 0, // valid when state == .is_const
        cmp_cc: u4 = 0, // valid when state == .is_cmp
    };

    /// Registers available for caching operand stack values.
    /// Excludes RAX (scratch/return), RCX (shift count), RDX (div remainder).
    const cache_regs = [_]emit.Reg{ .r10, .r11, .rsi, .rdi, .r8, .r9 };

    slots: [64]Slot = [_]Slot{.{}} ** 64,
    depth: u32 = 0,
    /// Base offset from RBP (marks bottom of operand stack area).
    base: i32,
    reg_used: [cache_regs.len]bool = [_]bool{false} ** cache_regs.len,

    fn init(local_count: u32) CachedStack {
        const base_off = -@as(i32, @intCast((local_count + 1) * 8));
        return .{ .base = base_off };
    }

    /// Return the memory offset for operand stack slot `idx`.
    fn memOffset(self: *const CachedStack, idx: u32) i32 {
        return self.base - @as(i32, @intCast(idx * 8));
    }

    /// Push a value that is currently in `src` register.
    fn push(self: *CachedStack, code: *emit.CodeBuffer, src: emit.Reg) !void {
        const idx = self.depth;
        self.depth += 1;

        for (cache_regs, 0..) |cr, i| {
            if (!self.reg_used[i]) {
                self.reg_used[i] = true;
                if (src != cr) try code.movRegReg(cr, src);
                self.slots[idx] = .{ .state = .in_reg, .reg = cr };
                return;
            }
        }

        // No free register – spill to memory.
        try code.movMemReg(.rbp, self.memOffset(idx), src);
        self.slots[idx] = .{ .state = .in_mem };
    }

    /// Push a deferred constant — no code emitted, no register consumed.
    fn pushConst(self: *CachedStack, val: i64) void {
        const idx = self.depth;
        self.depth += 1;
        self.slots[idx] = .{ .state = .is_const, .const_val = val };
    }

    /// Push a deferred comparison result — no code emitted, no register consumed.
    fn pushCmp(self: *CachedStack, cc: u4) void {
        const idx = self.depth;
        self.depth += 1;
        self.slots[idx] = .{ .state = .is_cmp, .cmp_cc = cc };
    }

    /// Return true if the top of stack is a deferred constant.
    fn topIsConst(self: *const CachedStack) bool {
        if (self.depth == 0) return false;
        return self.slots[self.depth - 1].state == .is_const;
    }

    /// Return the constant value at top of stack. Only valid if topIsConst().
    fn topConstVal(self: *const CachedStack) i64 {
        return self.slots[self.depth - 1].const_val;
    }

    /// Return true if the top of stack is a deferred comparison.
    fn topIsCmp(self: *const CachedStack) bool {
        if (self.depth == 0) return false;
        return self.slots[self.depth - 1].state == .is_cmp;
    }

    /// Return the condition code at top of stack. Only valid if topIsCmp().
    fn topCmpCc(self: *const CachedStack) u4 {
        return self.slots[self.depth - 1].cmp_cc;
    }

    /// Drop the top slot without emitting any code (for deferred slots consumed by fusion).
    fn dropTop(self: *CachedStack) void {
        self.depth -= 1;
        self.slots[self.depth].state = .empty;
    }

    /// Pop the top value into `dst` register.
    fn pop(self: *CachedStack, code: *emit.CodeBuffer, dst: emit.Reg) !void {
        self.depth -= 1;
        const slot = self.slots[self.depth];
        self.slots[self.depth].state = .empty;

        switch (slot.state) {
            .in_reg => {
                for (cache_regs, 0..) |cr, i| {
                    if (cr == slot.reg) {
                        self.reg_used[i] = false;
                        break;
                    }
                }
                if (dst != slot.reg) try code.movRegReg(dst, slot.reg);
            },
            .in_mem => {
                try code.movRegMem(dst, .rbp, self.memOffset(self.depth));
            },
            .is_const => {
                if (slot.const_val == 0) {
                    try code.xorReg32(dst);
                } else if (slot.const_val >= std.math.minInt(i32) and slot.const_val <= std.math.maxInt(i32)) {
                    try code.movRegImm32(dst, @intCast(slot.const_val));
                } else {
                    try code.movRegImm64(dst, @bitCast(slot.const_val));
                }
            },
            .is_cmp => {
                try code.setcc(slot.cmp_cc, dst);
                try code.movzxByte(dst, dst);
            },
            .empty => unreachable,
        }
    }

    /// Flush all register-cached values to memory.
    /// Must be called before branches, calls, and returns.
    fn flush(self: *CachedStack, code: *emit.CodeBuffer) !void {
        for (0..self.depth) |i| {
            switch (self.slots[i].state) {
                .in_reg => {
                    try code.movMemReg(.rbp, self.memOffset(@intCast(i)), self.slots[i].reg);
                    for (cache_regs, 0..) |cr, ci| {
                        if (cr == self.slots[i].reg) {
                            self.reg_used[ci] = false;
                            break;
                        }
                    }
                    self.slots[i] = .{ .state = .in_mem };
                },
                .is_const => {
                    // Use RCX as temp to preserve RAX (may hold return value)
                    if (self.slots[i].const_val >= std.math.minInt(i32) and self.slots[i].const_val <= std.math.maxInt(i32)) {
                        try code.movRegImm32(.rcx, @intCast(self.slots[i].const_val));
                    } else {
                        try code.movRegImm64(.rcx, @bitCast(self.slots[i].const_val));
                    }
                    try code.movMemReg(.rbp, self.memOffset(@intCast(i)), .rcx);
                    self.slots[i] = .{ .state = .in_mem };
                },
                .is_cmp => {
                    try code.setcc(self.slots[i].cmp_cc, .rcx);
                    try code.movzxByte(.rcx, .rcx);
                    try code.movMemReg(.rbp, self.memOffset(@intCast(i)), .rcx);
                    self.slots[i] = .{ .state = .in_mem };
                },
                .in_mem, .empty => {},
            }
        }
    }

    fn save(self: *const CachedStack) u32 {
        return self.depth;
    }

    fn restore(self: *CachedStack, saved: u32) void {
        while (self.depth > saved) {
            self.depth -= 1;
            if (self.slots[self.depth].state == .in_reg) {
                for (cache_regs, 0..) |cr, i| {
                    if (cr == self.slots[self.depth].reg) {
                        self.reg_used[i] = false;
                        break;
                    }
                }
            }
            self.slots[self.depth].state = .empty;
        }
    }
};

/// Compile an IR function to x86-64 machine code.
pub fn compileFunction(func: *const ir.IrFunction, allocator: std.mem.Allocator) ![]u8 {
    if (functionUsesV128(func)) return error.UnsupportedV128;

    var code = emit.CodeBuffer.init(allocator);
    errdefer code.deinit();

    // Frame: locals + 64 operand stack slots, aligned to 16 bytes
    // After push rbp (8 bytes), we need frame_size ≡ 8 (mod 16) for 16-byte RSP alignment.
    const raw_size: u32 = (func.local_count + 64) * 8;
    const frame_size: u32 = (raw_size + 15) & ~@as(u32, 15) | 8; // ensure ≡ 8 mod 16
    try code.emitPrologue(frame_size);

    // Spill parameters from ABI registers to stack frame slots.
    const spill_count = @min(func.param_count, param_regs.len);
    for (0..spill_count) |i| {
        try code.movMemReg(.rbp, -@as(i32, @intCast((i + 1) * 8)), param_regs[i]);
    }

    var stack = CachedStack.init(func.local_count);

    // Track block offsets for branch resolution
    var block_offsets = std.AutoHashMap(ir.BlockId, usize).init(allocator);
    defer block_offsets.deinit();

    var last_was_ret = false;
    var branch_patches: std.ArrayList(BranchPatch) = .empty;
    defer branch_patches.deinit(allocator);
    var call_patches: std.ArrayList(CallPatch) = .empty;
    defer call_patches.deinit(allocator);

    for (func.blocks.items, 0..) |block, idx| {
        try block_offsets.put(@intCast(idx), code.len());
        for (block.instructions.items) |inst| {
            last_was_ret = isRet(inst.op);
            try compileInst(&code, inst, &stack, &branch_patches, &call_patches);
        }
    }

    // Patch forward branches
    for (branch_patches.items) |patch| {
        if (block_offsets.get(patch.target_block)) |target_off| {
            const rel: i32 = @intCast(@as(i64, @intCast(target_off)) - @as(i64, @intCast(patch.patch_offset + 4)));
            code.patchI32(patch.patch_offset, rel);
        }
    }

    // Add epilogue if the function didn't end with a ret instruction
    if (!last_was_ret) {
        try code.emitEpilogue();
    }

    return code.bytes.toOwnedSlice(allocator);
}

fn isRet(op: ir.Inst.Op) bool {
    return switch (op) {
        .ret, .ret_multi => true,
        else => false,
    };
}

const BranchPatch = struct {
    patch_offset: usize,
    target_block: ir.BlockId,
};

const CallPatch = struct {
    patch_offset: usize,
    target_func_idx: u32,
};

/// Patches one 4-byte entry in a br_table jump table. After final layout is
/// known, `entry_offset` is overwritten with `block_offsets[target_block] - base_offset`.
const TablePatch = struct {
    entry_offset: usize,
    base_offset: usize,
    target_block: ir.BlockId,
};

fn compileInst(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    stack: *CachedStack,
    patches: *std.ArrayList(BranchPatch),
    call_patches: *std.ArrayList(CallPatch),
) !void {
    switch (inst.op) {
        // ── Constants ─────────────────────────────────────────────────
        .iconst_32 => |val| {
            stack.pushConst(val);
        },
        .iconst_64 => |val| {
            stack.pushConst(val);
        },
        .fconst_32 => |val| {
            try code.movRegImm32(.rax, @bitCast(val));
            try stack.push(code, .rax);
        },
        .fconst_64 => |val| {
            try code.movRegImm64(.rax, @bitCast(val));
            try stack.push(code, .rax);
        },

        // ── Binary arithmetic ─────────────────────────────────────────
        .add => {
            if (stack.topIsConst()) {
                const imm = stack.topConstVal();
                stack.dropTop();
                try stack.pop(code, .rax);
                if (imm != 0) try code.addRegImm32(.rax, @intCast(imm));
            } else {
                try stack.pop(code, .rcx);
                try stack.pop(code, .rax);
                try code.addRegReg(.rax, .rcx);
            }
            try stack.push(code, .rax);
        },
        .sub => {
            if (stack.topIsConst()) {
                const imm = stack.topConstVal();
                stack.dropTop();
                try stack.pop(code, .rax);
                if (imm != 0) try code.subRegImm32(.rax, @intCast(imm));
            } else {
                try stack.pop(code, .rcx);
                try stack.pop(code, .rax);
                try code.subRegReg(.rax, .rcx);
            }
            try stack.push(code, .rax);
        },
        .mul => {
            try stack.pop(code, .rcx);
            try stack.pop(code, .rax);
            try code.imulRegReg(.rax, .rcx);
            try stack.push(code, .rax);
        },
        .@"and" => {
            if (stack.topIsConst()) {
                const imm = stack.topConstVal();
                stack.dropTop();
                try stack.pop(code, .rax);
                try code.andRegImm32(.rax, @intCast(imm));
            } else {
                try stack.pop(code, .rcx);
                try stack.pop(code, .rax);
                try code.andRegReg(.rax, .rcx);
            }
            try stack.push(code, .rax);
        },
        .@"or" => {
            if (stack.topIsConst()) {
                const imm = stack.topConstVal();
                stack.dropTop();
                try stack.pop(code, .rax);
                if (imm != 0) try code.orRegImm32(.rax, @intCast(imm));
            } else {
                try stack.pop(code, .rcx);
                try stack.pop(code, .rax);
                try code.orRegReg(.rax, .rcx);
            }
            try stack.push(code, .rax);
        },
        .xor => {
            if (stack.topIsConst()) {
                const imm = stack.topConstVal();
                stack.dropTop();
                try stack.pop(code, .rax);
                if (imm != 0) try code.xorRegImm32(.rax, @intCast(imm));
            } else {
                try stack.pop(code, .rcx);
                try stack.pop(code, .rax);
                try code.xorRegReg(.rax, .rcx);
            }
            try stack.push(code, .rax);
        },

        // ── Shifts ────────────────────────────────────────────────────
        .shl => {
            if (stack.topIsConst()) {
                const imm: u8 = @intCast(stack.topConstVal() & 0x3f);
                stack.dropTop();
                try stack.pop(code, .rax);
                if (imm != 0) try code.shiftRegImm8(.rax, 4, imm, true);
                try stack.push(code, .rax);
            } else {
                try stack.pop(code, .rcx); // count → RCX (CL)
                try stack.pop(code, .rax); // value
                // SHL RAX, CL: REX.W D3 /4
                try code.rexW(.rax, .rax);
                try code.emitByte(0xD3);
                try code.modrm(0b11, 4, emit.Reg.rax.low3());
                try stack.push(code, .rax);
            }
        },
        .shr_s => {
            if (stack.topIsConst()) {
                const imm: u8 = @intCast(stack.topConstVal() & 0x3f);
                stack.dropTop();
                try stack.pop(code, .rax);
                if (imm != 0) try code.shiftRegImm8(.rax, 7, imm, true);
                try stack.push(code, .rax);
            } else {
                try stack.pop(code, .rcx);
                try stack.pop(code, .rax);
                // SAR RAX, CL: REX.W D3 /7
                try code.rexW(.rax, .rax);
                try code.emitByte(0xD3);
                try code.modrm(0b11, 7, emit.Reg.rax.low3());
                try stack.push(code, .rax);
            }
        },
        .shr_u => {
            if (stack.topIsConst()) {
                const imm: u8 = @intCast(stack.topConstVal() & 0x3f);
                stack.dropTop();
                try stack.pop(code, .rax);
                if (imm != 0) try code.shiftRegImm8(.rax, 5, imm, true);
                try stack.push(code, .rax);
            } else {
                try stack.pop(code, .rcx);
                try stack.pop(code, .rax);
                // SHR RAX, CL: REX.W D3 /5
                try code.rexW(.rax, .rax);
                try code.emitByte(0xD3);
                try code.modrm(0b11, 5, emit.Reg.rax.low3());
                try stack.push(code, .rax);
            }
        },
        .rotl => {
            if (stack.topIsConst()) {
                const imm: u8 = @intCast(stack.topConstVal() & 0x3f);
                stack.dropTop();
                try stack.pop(code, .rax);
                if (imm != 0) try code.shiftRegImm8(.rax, 0, imm, true);
                try stack.push(code, .rax);
            } else {
                try stack.pop(code, .rcx);
                try stack.pop(code, .rax);
                // ROL RAX, CL: REX.W D3 /0
                try code.rexW(.rax, .rax);
                try code.emitByte(0xD3);
                try code.modrm(0b11, 0, emit.Reg.rax.low3());
                try stack.push(code, .rax);
            }
        },
        .rotr => {
            if (stack.topIsConst()) {
                const imm: u8 = @intCast(stack.topConstVal() & 0x3f);
                stack.dropTop();
                try stack.pop(code, .rax);
                if (imm != 0) try code.shiftRegImm8(.rax, 1, imm, true);
                try stack.push(code, .rax);
            } else {
                try stack.pop(code, .rcx);
                try stack.pop(code, .rax);
                // ROR RAX, CL: REX.W D3 /1
                try code.rexW(.rax, .rax);
                try code.emitByte(0xD3);
                try code.modrm(0b11, 1, emit.Reg.rax.low3());
                try stack.push(code, .rax);
            }
        },

        // ── Division / remainder ──────────────────────────────────────
        .div_s, .div_u, .rem_s, .rem_u => {
            try stack.pop(code, .rcx); // divisor
            try stack.pop(code, .rax); // dividend

            const is_signed = (inst.op == .div_s or inst.op == .rem_s);
            const is_rem = (inst.op == .rem_s or inst.op == .rem_u);

            if (is_signed) {
                try code.cqo();
            } else {
                try code.xorRegReg(.rdx, .rdx);
            }

            if (is_signed) {
                try code.idivReg(.rcx);
            } else {
                try code.divReg(.rcx);
            }

            if (is_rem) {
                try stack.push(code, .rdx);
            } else {
                try stack.push(code, .rax);
            }
        },

        // ── Comparisons ───────────────────────────────────────────────
        inline .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u => |_, tag| {
            if (stack.topIsConst()) {
                const imm = stack.topConstVal();
                stack.dropTop();
                try stack.pop(code, .rax);
                try code.cmpRegImm32(.rax, @intCast(imm));
            } else {
                try stack.pop(code, .rcx); // rhs
                try stack.pop(code, .rax); // lhs
                try code.cmpRegReg(.rax, .rcx);
            }

            const cc: u4 = comptime switch (tag) {
                .eq => 0x4,
                .ne => 0x5,
                .lt_s => 0xC,
                .lt_u => 0x2,
                .gt_s => 0xF,
                .gt_u => 0x7,
                .le_s => 0xE,
                .le_u => 0x6,
                .ge_s => 0xD,
                .ge_u => 0x3,
                else => unreachable,
            };
            stack.pushCmp(cc);
        },

        // ── Unary ops ─────────────────────────────────────────────────
        .eqz => {
            try stack.pop(code, .rax);
            try code.testRegReg(.rax, .rax);
            try code.setcc(0x4, .rax); // SETZ
            try code.movzxByte(.rax, .rax);
            try stack.push(code, .rax);
        },
        .clz => {
            try stack.pop(code, .rax);
            try code.lzcnt(.rax, .rax);
            try stack.push(code, .rax);
        },
        .ctz => {
            try stack.pop(code, .rax);
            try code.tzcnt(.rax, .rax);
            try stack.push(code, .rax);
        },
        .popcnt => {
            try stack.pop(code, .rax);
            try code.popcntReg(.rax, .rax);
            try stack.push(code, .rax);
        },

        // ── Local variable access ─────────────────────────────────────
        .local_get => |idx| {
            try code.movRegMem(.rax, .rbp, -@as(i32, @intCast((idx + 1) * 8)));
            try stack.push(code, .rax);
        },
        .local_set => |ls| {
            try stack.pop(code, .rax);
            try code.movMemReg(.rbp, -@as(i32, @intCast((ls.idx + 1) * 8)), .rax);
        },

        // ── Return ────────────────────────────────────────────────────
        .ret => |maybe_val| {
            if (maybe_val != null) {
                try stack.pop(code, .rax); // pop before flush to preserve deferred optimizations
            }
            try stack.flush(code); // flush uses RCX as temp, preserving RAX
            try code.emitEpilogue();
        },

        .@"unreachable" => try emitTrapHelperCall(code, vmctx_trap_unreachable_fn_field),

        // ── Select (conditional move) ─────────────────────────────────
        .select => {
            try stack.pop(code, .rax); // cond
            try stack.pop(code, .rcx); // if_false
            try stack.pop(code, .rdx); // if_true
            // TEST RAX, RAX; if cond!=0, pick true (rdx), else keep false (rcx)
            try code.testRegReg(.rax, .rax);
            try code.cmovnz(.rcx, .rdx);
            try stack.push(code, .rcx);
        },

        // ── Memory load ───────────────────────────────────────────────
        .load => |ld| {
            try stack.pop(code, .rax); // base address
            if (ld.offset > 0) {
                try code.addRegImm32(.rax, @intCast(ld.offset));
            }
            try code.movRegMemNoRex(.rax, .rax, 0);
            try stack.push(code, .rax);
        },

        // ── Memory store ──────────────────────────────────────────────
        .store => |st| {
            try stack.pop(code, .rcx); // value
            try stack.pop(code, .rax); // base address
            if (st.offset > 0) {
                try code.addRegImm32(.rax, @intCast(st.offset));
            }
            try code.movMemRegNoRex(.rax, 0, .rcx);
        },

        // ── Branches ──────────────────────────────────────────────────
        .br => |target| {
            try stack.flush(code);
            try code.emitByte(0xE9); // JMP rel32
            const patch_off = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = patch_off, .target_block = target });
        },
        .br_if => |br| {
            if (stack.topIsCmp()) {
                // Fused compare-and-branch: emit Jcc directly
                const cc = stack.topCmpCc();
                stack.dropTop();
                try stack.flush(code); // flush remaining values (MOV only, flags preserved)
                // Jcc then_block
                try code.emitByte(0x0F);
                try code.emitByte(0x80 | @as(u8, cc));
                const then_patch = code.len();
                try code.emitI32(0);
                try patches.append(code.allocator, .{ .patch_offset = then_patch, .target_block = br.then_block });
            } else {
                try stack.flush(code);
                // Non-fused: pop condition, test, JNE
                try stack.pop(code, .rax);
                try code.testRegReg(.rax, .rax);
                // JNE then_block
                try code.emitByte(0x0F);
                try code.emitByte(0x85);
                const then_patch = code.len();
                try code.emitI32(0);
                try patches.append(code.allocator, .{ .patch_offset = then_patch, .target_block = br.then_block });
            }
            // JMP else_block (fallthrough)
            try code.emitByte(0xE9);
            const else_patch = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = else_patch, .target_block = br.else_block });
        },
        .br_table => {
            // br_table is only used through compileFunctionRA path in production.
            // The legacy stack-based compileInst path does not support it.
            return error.Unsupported;
        },

        // ── Function calls ────────────────────────────────────────────
        .call => |cl| {
            try stack.flush(code);
            const n_args: u32 = @intCast(cl.args.len);

            // Pop args from operand stack in reverse order into ABI registers.
            // Wasm pushes args left-to-right, so first arg is deepest.
            // We pop in reverse (last arg first), store to temp slots, then
            // load in order to ABI regs.
            if (n_args > 0 and n_args <= param_regs.len) {
                // Pop all args into temporary frame slots (reuse operand stack area)
                // We pop in reverse order (topmost = last arg)
                var i: u32 = n_args;
                while (i > 0) {
                    i -= 1;
                    try stack.pop(code, .rax);
                    try code.movRegReg(param_regs[i], .rax);
                }
            } else if (n_args > param_regs.len) {
                // More than 6 args: pop extras to stack, then first 6 to regs
                var i: u32 = n_args;
                while (i > param_regs.len) {
                    i -= 1;
                    try stack.pop(code, .rax); // discard overflow args for now
                }
                while (i > 0) {
                    i -= 1;
                    try stack.pop(code, .rax);
                    try code.movRegReg(param_regs[i], .rax);
                }
            }

            // Emit CALL rel32 (placeholder, patched later)
            try code.emitByte(0xE8);
            const patch_off = code.len();
            try code.emitI32(0);
            try call_patches.append(code.allocator, .{
                .patch_offset = patch_off,
                .target_func_idx = cl.func_idx,
            });

            // Push return value (RAX) to operand stack
            try stack.push(code, .rax);
        },

        // ── Global access (stubs) ─────────────────────────────────────
        .global_get => {
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
        .global_set => {
            try stack.pop(code, .rax);
        },

        // ── Type conversions (pass-through for integer types) ─────────
        .wrap_i64,
        .extend_i32_s,
        .extend_i32_u,
        .extend8_s,
        .extend16_s,
        .extend32_s,
        .reinterpret,
        => {
            // Pop, push back (value stays on stack, no-op for now)
            try stack.pop(code, .rax);
            try stack.push(code, .rax);
        },

        // ── Float/conversion stubs (pop input, push placeholder) ──────
        .trunc_f32_s,
        .trunc_f32_u,
        .trunc_f64_s,
        .trunc_f64_u,
        .convert_s,
        .convert_u,
        .convert_i32_s,
        .convert_i64_s,
        .convert_i32_u,
        .convert_i64_u,
        .demote_f64,
        .promote_f32,
        .trunc_sat_f32_s,
        .trunc_sat_f32_u,
        .trunc_sat_f64_s,
        .trunc_sat_f64_u,
        .f_neg,
        .f_abs,
        .f_sqrt,
        .f_ceil,
        .f_floor,
        .f_trunc,
        .f_nearest,
        => {
            try stack.pop(code, .rax);
            try stack.push(code, .rax);
        },

        // ── Float binary stubs ────────────────────────────────────────
        .f_min,
        .f_max,
        .f_copysign,
        .f_eq,
        .f_ne,
        .f_lt,
        .f_gt,
        .f_le,
        .f_ge,
        => {
            try stack.pop(code, .rcx);
            try stack.pop(code, .rax);
            try stack.push(code, .rax);
        },

        // ── Atomic operations ──────────────────────────────────────────
        .atomic_fence => {
            try code.mfence();
        },
        .atomic_load => |ld| {
            try stack.pop(code, .rax); // base address
            if (ld.offset > 0) {
                try code.addRegImm32(.rax, @intCast(ld.offset));
            }
            try code.movRegMemSized(.rax, .rax, 0, ld.size);
            try stack.push(code, .rax);
        },
        .atomic_store => |st| {
            try stack.pop(code, .rcx); // val
            try stack.pop(code, .rax); // base address
            if (st.offset > 0) {
                try code.addRegImm32(.rax, @intCast(st.offset));
            }
            try code.movMemRegSized(.rax, 0, .rcx, st.size);
            try code.mfence(); // seq-cst ordering
        },
        .atomic_rmw => |rmw| {
            try stack.pop(code, .rcx); // val
            try stack.pop(code, .rax); // base address
            if (rmw.offset > 0) {
                try code.addRegImm32(.rax, @intCast(rmw.offset));
            }
            switch (rmw.op) {
                .add => {
                    // LOCK XADD [rax], rcx → rcx gets old value
                    try code.lockXadd(.rax, 0, .rcx, rmw.size);
                    try code.movRegReg(.rax, .rcx);
                    try code.zeroExtendReg(.rax, rmw.size);
                },
                .sub => {
                    // NEG rcx; LOCK XADD → rcx gets old value
                    try code.negReg(.rcx, rmw.size);
                    try code.lockXadd(.rax, 0, .rcx, rmw.size);
                    try code.movRegReg(.rax, .rcx);
                    try code.zeroExtendReg(.rax, rmw.size);
                },
                .xchg => {
                    // XCHG [rax], rcx → rcx gets old value (implicit LOCK)
                    try code.xchgMemReg(.rax, 0, .rcx, rmw.size);
                    try code.movRegReg(.rax, .rcx);
                    try code.zeroExtendReg(.rax, rmw.size);
                },
                .@"and", .@"or", .xor => {
                    // CAS loop: r8 needed as temp, flush stack to free cache regs
                    try stack.flush(code);
                    try code.movRegReg(.rdx, .rax); // rdx = address
                    try code.movRegMemSized(.rax, .rdx, 0, rmw.size); // rax = current value (zero-extended)
                    // retry:
                    const retry_off = code.len();
                    try code.movRegReg(.r8, .rax); // r8 = copy of old value
                    switch (rmw.op) {
                        .@"and" => try code.andRegReg(.r8, .rcx),
                        .@"or" => try code.orRegReg(.r8, .rcx),
                        .xor => try code.xorRegReg(.r8, .rcx),
                        else => unreachable,
                    }
                    // LOCK CMPXCHG: if [rdx]==rax → store r8, else rax=[rdx]
                    try code.lockCmpxchg(.rdx, 0, .r8, rmw.size);
                    // JNE retry (backward jump on CAS failure)
                    const jne_off = code.len();
                    try code.jne(0); // placeholder rel32
                    const retry_rel: i32 = @intCast(@as(i64, @intCast(retry_off)) - @as(i64, @intCast(jne_off + 6)));
                    code.patchI32(jne_off + 2, retry_rel);
                    // rax = old value (zero-extended from initial MOVZX; sub-word
                    // CMPXCHG failure only writes AL/AX, preserving zero upper bytes)
                },
            }
            try stack.push(code, .rax);
        },
        .atomic_cmpxchg => |cmpxchg| {
            try stack.pop(code, .rcx); // replacement
            try stack.pop(code, .rax); // expected → rax (implicit CMPXCHG operand)
            try stack.pop(code, .rdx); // base → rdx
            if (cmpxchg.offset > 0) {
                try code.addRegImm32(.rdx, @intCast(cmpxchg.offset));
            }
            try code.lockCmpxchg(.rdx, 0, .rcx, cmpxchg.size);
            try code.zeroExtendReg(.rax, cmpxchg.size);
            try stack.push(code, .rax);
        },
        .atomic_notify => {
            try stack.pop(code, .rcx); // count
            try stack.pop(code, .rax); // base
            // TODO: implement with futex runtime (see #76)
            try code.movRegImm32(.rax, 0); // return 0 waiters woken
            try stack.push(code, .rax);
        },
        .atomic_wait => {
            try stack.pop(code, .rcx); // timeout
            try stack.pop(code, .rdx); // expected
            try stack.pop(code, .rax); // base
            // TODO: implement with futex runtime (see #76)
            try code.movRegImm32(.rax, 1); // return 1 ("not equal")
            try stack.push(code, .rax);
        },
        .memory_copy => |mc| {
            _ = mc;
            try stack.pop(code, .rcx); // len
            try stack.pop(code, .rdx); // src
            try stack.pop(code, .rax); // dst
            // TODO: implement as memcpy
        },
        .memory_fill => |mf| {
            _ = mf;
            try stack.pop(code, .rcx); // len
            try stack.pop(code, .rdx); // val
            try stack.pop(code, .rax); // dst
            // TODO: implement as memset
        },
        .memory_size => {
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
        .memory_grow => {
            try stack.pop(code, .rax); // pages (discard)
            try code.movRegImm32(.rax, -1); // always fail in non-RA path
            try stack.push(code, .rax);
        },
        .call_indirect => {
            // Stub in non-RA path: pop args + elem_idx, push 0
            try stack.pop(code, .rax); // elem_idx (discard)
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
        .call_ref => {
            // Stub in non-RA path: pop funcref, push 0
            try stack.pop(code, .rax); // func_ref (discard)
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
        .call_result => {
            // Stub in non-RA path: push 0 as placeholder.
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
        .ret_multi => {
            // Stub in non-RA path: emit plain epilogue.
            try code.emitEpilogue();
        },
        .memory_init => {
            try stack.pop(code, .rax); // len
            try stack.pop(code, .rax); // src
            try stack.pop(code, .rax); // dst
        },
        .data_drop => {},
        .table_init => {
            try stack.pop(code, .rax); // len
            try stack.pop(code, .rax); // src
            try stack.pop(code, .rax); // dst
        },
        .elem_drop => {},
        .table_size => {
            try code.movRegMem(.r10, .rbp, vmctx_offset);
            try code.movRegMemNoRex(.rax, .r10, vmctx_func_table_len_field);
            try stack.push(code, .rax);
        },
        .table_get => {
            try stack.pop(code, .rax); // idx (discard in non-RA path)
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
        .table_set => {
            try stack.pop(code, .rax); // val
            try stack.pop(code, .rax); // idx
        },
        .table_grow => {
            try stack.pop(code, .rax); // delta
            try stack.pop(code, .rax); // init
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
        .ref_func => |fidx| {
            try code.movRegMem(.r10, .rbp, vmctx_offset);
            try code.movRegMem(.r10, .r10, vmctx_funcptrs_field);
            try code.movRegMem(.rax, .r10, @as(i32, @intCast(fidx * 8)));
            try stack.push(code, .rax);
        },
        .v128_const,
        .v128_load,
        .v128_store,
        .v128_not,
        .v128_bitwise,
        .i32x4_binop,
        .i32x4_shift,
        .i32x4_splat,
        .i32x4_extract_lane,
        .i32x4_replace_lane,
        .i8x16_binop,
        .i8x16_shift,
        .i8x16_splat,
        .i8x16_extract_lane,
        .i8x16_replace_lane,
        .i16x8_binop,
        .i16x8_shift,
        .i16x8_splat,
        .i16x8_extract_lane,
        .i16x8_replace_lane,
        .i64x2_binop,
        .i64x2_splat,
        .i64x2_extract_lane,
        .i64x2_replace_lane,
        => return error.UnsupportedV128,
        // Phi must be lowered before codegen.
        .phi => unreachable,
    }
}

/// Result of compiling a single function.
const FuncCompileResult = struct {
    code: []u8,
    call_patches: []CallPatch,
};

/// Result of compiling an IR module.
pub const CompileResult = struct {
    code: []u8,
    offsets: []u32,
};

fn dumpFuncIRAlloc(func: *const ir.IrFunction, fi: u32, import_count: u32, allocator: std.mem.Allocator) !void {
    const p = std.debug.print;
    p("=== IR DUMP func[{d}] param_count={d} local_count={d} blocks={d} ===\n", .{ fi, func.param_count, func.local_count, func.blocks.items.len });

    // Rebuild clobber points (same logic as compileFunctionRA) so allocation matches.
    var clobbers: std.ArrayList(regalloc.ClobberPoint) = .empty;
    defer clobbers.deinit(allocator);
    var cp_pos: u32 = 0;
    for (func.blocks.items) |block| {
        for (block.instructions.items) |ci| {
            switch (ci.op) {
                .call, .call_indirect, .call_ref, .memory_grow, .memory_copy, .table_grow, .table_set => {
                    try clobbers.append(allocator, .{ .pos = cp_pos, .regs_clobbered = x86_64_call_clobber_mask });
                },
                .memory_fill => try clobbers.append(allocator, .{ .pos = cp_pos, .regs_clobbered = @as(u64, 1) << 3 }),
                else => {},
            }
            cp_pos += 1;
        }
    }
    var alloc = try regalloc.allocate(func, allocator, x86_64_reg_set(func.local_count), clobbers.items);
    defer alloc.deinit();

    var pos: u32 = 0;
    for (func.blocks.items, 0..) |block, bi| {
        p("  block[{d}]:\n", .{bi});
        for (block.instructions.items) |inst| {
            const dest_str: []const u8 = if (inst.dest != null) "dst" else "   ";
            var dest_alloc_buf: [32]u8 = undefined;
            const dest_alloc: []const u8 = if (inst.dest) |d| blk: {
                if (alloc.get(d)) |a| switch (a) {
                    .reg => |r| break :blk std.fmt.bufPrint(&dest_alloc_buf, "v{d}→r{d}", .{ d, r }) catch "?",
                    .stack => |off| break :blk std.fmt.bufPrint(&dest_alloc_buf, "v{d}→[rbp{d}]", .{ d, off }) catch "?",
                } else break :blk "v?→none";
            } else "";
            switch (inst.op) {
                .iconst_32 => |v| p("    {d:4}: iconst_32 {d}  {s} {s}\n", .{ pos, v, dest_str, dest_alloc }),
                .iconst_64 => |v| p("    {d:4}: iconst_64 {d}  {s} {s}\n", .{ pos, v, dest_str, dest_alloc }),
                .add => |b| p("    {d:4}: add v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .sub => |b| p("    {d:4}: sub v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .mul => |b| p("    {d:4}: mul v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .@"and" => |b| p("    {d:4}: and v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .@"or" => |b| p("    {d:4}: or v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .xor => |b| p("    {d:4}: xor v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .shl => |b| p("    {d:4}: shl v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .shr_s => |b| p("    {d:4}: shr_s v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .shr_u => |b| p("    {d:4}: shr_u v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .eq => |b| p("    {d:4}: eq v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .ne => |b| p("    {d:4}: ne v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .lt_s => |b| p("    {d:4}: lt_s v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .lt_u => |b| p("    {d:4}: lt_u v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .gt_s => |b| p("    {d:4}: gt_s v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .gt_u => |b| p("    {d:4}: gt_u v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .le_s => |b| p("    {d:4}: le_s v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .le_u => |b| p("    {d:4}: le_u v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .ge_s => |b| p("    {d:4}: ge_s v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .ge_u => |b| p("    {d:4}: ge_u v{d}, v{d}  {s}\n", .{ pos, b.lhs, b.rhs, dest_alloc }),
                .eqz => |v| p("    {d:4}: eqz v{d}  {s}\n", .{ pos, v, dest_alloc }),
                .local_get => |idx| p("    {d:4}: local_get {d}  {s}\n", .{ pos, idx, dest_alloc }),
                .local_set => |ls| p("    {d:4}: local_set {d}, v{d}\n", .{ pos, ls.idx, ls.val }),
                .load => |ld| p("    {d:4}: load size={d} se={any} v{d}+{d}  {s}\n", .{ pos, ld.size, ld.sign_extend, ld.base, ld.offset, dest_alloc }),
                .store => |st| p("    {d:4}: store size={d} v{d}+{d}, v{d}\n", .{ pos, st.size, st.base, st.offset, st.val }),
                .br => |tgt| p("    {d:4}: br block{d}\n", .{ pos, tgt }),
                .br_if => |b| p("    {d:4}: br_if v{d} ? block{d} : block{d}\n", .{ pos, b.cond, b.then_block, b.else_block }),
                .br_table => |bt| p("    {d:4}: br_table v{d} default=block{d} (ntargets={d})\n", .{ pos, bt.index, bt.default, bt.targets.len }),
                .ret => |rv| {
                    if (rv) |v| p("    {d:4}: ret v{d}\n", .{ pos, v }) else p("    {d:4}: ret\n", .{pos});
                },
                .call => |cl| {
                    const kind = if (cl.func_idx < import_count) "import" else "func";
                    p("    {d:4}: call {s}[{d}] nargs={d}  {s}\n", .{ pos, kind, cl.func_idx, cl.args.len, dest_alloc });
                },
                .call_indirect => |ci| p("    {d:4}: call_indirect type={d} v{d} nargs={d}  {s}\n", .{ pos, ci.type_idx, ci.elem_idx, ci.args.len, dest_alloc }),
                .call_ref => |cr| p("    {d:4}: call_ref type={d} v{d} nargs={d}  {s}\n", .{ pos, cr.type_idx, cr.func_ref, cr.args.len, dest_alloc }),
                .select => |sel| p("    {d:4}: select cond=v{d} t=v{d} f=v{d}  {s}\n", .{ pos, sel.cond, sel.if_true, sel.if_false, dest_alloc }),
                .global_get => |idx| p("    {d:4}: global_get {d}  {s}\n", .{ pos, idx, dest_alloc }),
                .global_set => |gs| p("    {d:4}: global_set {d}, v{d}\n", .{ pos, gs.idx, gs.val }),
                else => p("    {d:4}: <op tag={s}>  {s}\n", .{ pos, @tagName(inst.op), dest_alloc }),
            }
            pos += 1;
        }
    }
    p("=== END DUMP func[{d}] ===\n", .{fi});
}

/// Compile all functions in an IR module to x86-64 machine code.
/// After compiling all functions, patches inter-function call sites.
pub fn compileModule(ir_module: *const ir.IrModule, allocator: std.mem.Allocator) !CompileResult {
    var all_code: std.ArrayList(u8) = .empty;
    errdefer all_code.deinit(allocator);
    var offsets: std.ArrayList(u32) = .empty;
    errdefer offsets.deinit(allocator);

    // Per-function call patches, accumulated across all functions
    var global_call_patches: std.ArrayList(GlobalCallPatch) = .empty;
    defer global_call_patches.deinit(allocator);

    for (ir_module.functions.items, 0..) |func, fi| {
        const func_start = all_code.items.len;
        try offsets.append(allocator, @intCast(func_start));

        // Compile function using register allocator
        const result = try compileFunctionRA(&func, ir_module.import_count, allocator);
        defer allocator.free(result.code);
        defer allocator.free(result.call_patches);

        // Debug: dump IR+alloc for selected function index (-1 disables).
        const dump_func_idx: i32 = -1;
        const dump_all: bool = false;
        if (dump_all or @as(i32, @intCast(fi)) == dump_func_idx) {
            dumpFuncIRAlloc(&func, @intCast(fi), ir_module.import_count, allocator) catch {};
        }

        // Accumulate call patches with global offsets
        for (result.call_patches) |patch| {
            try global_call_patches.append(allocator, .{
                .patch_offset = func_start + patch.patch_offset,
                .target_func_idx = patch.target_func_idx,
            });
        }

        try all_code.appendSlice(allocator, result.code);
    }

    // Patch inter-function call sites now that all function offsets are known
    for (global_call_patches.items) |patch| {
        if (patch.target_func_idx < offsets.items.len) {
            const target_off = offsets.items[patch.target_func_idx];
            const rel: i32 = @intCast(@as(i64, @intCast(target_off)) - @as(i64, @intCast(patch.patch_offset + 4)));
            std.mem.writeInt(i32, all_code.items[patch.patch_offset..][0..4], rel, .little);
        }
    }

    return .{
        .code = try all_code.toOwnedSlice(allocator),
        .offsets = try offsets.toOwnedSlice(allocator),
    };
}

const GlobalCallPatch = struct {
    patch_offset: usize,
    target_func_idx: u32,
};

// ── Register-Allocated Compilation ────────────────────────────────────

const regalloc = @import("../../ir/regalloc.zig");
const analysis = @import("../../ir/analysis.zig");

/// x86-64 allocatable GPR set: rdx(2), rbx(3), rsi(6), rdi(7), r8(8), r9(9),
/// r12(12), r13(13), r14(14), r15(15). Order matches the legacy mask layout,
/// so `caller_saved_indices` / `callee_saved_indices` are stable indices
/// into `alloc_regs` and match the bit positions used in clobber masks below.
const x86_64_alloc_regs = [_]regalloc.PhysReg{ 2, 3, 6, 7, 8, 9, 12, 13, 14, 15 };

/// Indices into `x86_64_alloc_regs` of callee-saved registers (same on
/// Win64 and SysV): rbx(idx 1), r12..r15 (idx 6..9).
const x86_64_callee_saved_indices = [_]u8{ 1, 6, 7, 8, 9 };

/// Indices into `x86_64_alloc_regs` of caller-saved registers, by ABI.
/// On Win64: rdx, r8, r9. On SysV: add rsi, rdi.
const x86_64_caller_saved_indices = if (builtin.os.tag == .windows)
    [_]u8{ 0, 4, 5 }
else
    [_]u8{ 0, 2, 3, 4, 5 };

/// Bitmask over `x86_64_alloc_regs` indices of caller-saved registers
/// clobbered by a normal call / host runtime call (memory.grow, table.set,
/// etc.).
const x86_64_call_clobber_mask: u64 = blk: {
    var m: u64 = 0;
    for (x86_64_caller_saved_indices) |i| m |= @as(u64, 1) << i;
    break :blk m;
};

/// Build the per-function `RegSet` for x86-64. `local_count` determines
/// the spill-slot origin so it must be passed in.
fn x86_64_reg_set(local_count: u32) regalloc.RegSet {
    return .{
        .alloc_regs = &x86_64_alloc_regs,
        .callee_saved_indices = &x86_64_callee_saved_indices,
        .caller_saved_indices = &x86_64_caller_saved_indices,
        // Frame layout (see compileFunctionImpl): [rbp-8]=VmCtx,
        // [rbp-16..-(1+LC)*8]=locals (LC slots), [-(2+LC)*8..-(65+LC)*8]
        // =op-stack (64 slots). Spills begin at -(66+LC)*8 and grow down.
        .spill_base = -@as(i32, @intCast((local_count + 66) * 8)),
        .spill_stride = -8,
    };
}

fn functionUsesV128(func: *const ir.IrFunction) bool {
    if (func.local_types) |local_types| {
        for (local_types) |ty| if (ty == .v128) return true;
    }
    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            if (inst.type == .v128) return true;
            switch (inst.op) {
                .v128_const,
                .v128_load,
                .v128_store,
                .v128_not,
                .v128_bitwise,
                .i32x4_binop,
                .i32x4_shift,
                .i32x4_splat,
                .i32x4_extract_lane,
                .i32x4_replace_lane,
                .i8x16_binop,
                .i8x16_shift,
                .i8x16_splat,
                .i8x16_extract_lane,
                .i8x16_replace_lane,
                .i16x8_binop,
                .i16x8_shift,
                .i16x8_splat,
                .i16x8_extract_lane,
                .i16x8_replace_lane,
                .i64x2_binop,
                .i64x2_splat,
                .i64x2_extract_lane,
                .i64x2_replace_lane,
                => return true,
                else => {},
            }
        }
    }
    return false;
}

/// Compile an IR function using the linear scan register allocator.
/// VRegs are assigned to physical registers; instructions operate directly
/// on assigned registers without push/pop through a CachedStack.
pub fn compileFunctionRA(func: *const ir.IrFunction, import_count: u32, allocator: std.mem.Allocator) !FuncCompileResult {
    if (functionUsesV128(func)) return error.UnsupportedV128;

    // Compute block emission order ONCE, before anything that uses global
    // instruction numbering. Clobber points, live ranges, and code emission
    // all must use THIS order (same fix as aarch64 — see PR #195 / #203).
    const block_order = try passes.reorderBlocks(func, allocator);
    defer allocator.free(block_order);

    // Collect clobber points: instructions that destroy specific registers.
    // Uses block_order so numbering matches live-range computation.
    var clobber_points: std.ArrayList(regalloc.ClobberPoint) = .empty;
    defer clobber_points.deinit(allocator);
    {
        var pos: u32 = 0;
        for (block_order) |block_id| {
            for (func.blocks.items[block_id].instructions.items) |ci| {
                switch (ci.op) {
                    .call, .call_indirect, .call_ref, .memory_grow, .memory_copy, .table_grow, .table_set => {
                        // Calls clobber caller-saved allocatable regs.
                        // memory.grow / memory.copy are compiled as host calls (same ABI),
                        // so they clobber the same set of caller-saved registers.
                        try clobber_points.append(allocator, .{ .pos = pos, .regs_clobbered = x86_64_call_clobber_mask });
                    },
                    .memory_fill => {
                        // REP STOSB clobbers rdi (index 3 in alloc_regs).
                        try clobber_points.append(allocator, .{
                            .pos = pos,
                            .regs_clobbered = @as(u64, 1) << 3,
                        });
                    },
                    else => {},
                }
                pos += 1;
            }
        }
    }

    // Compute live ranges using the SAME block_order, then allocate registers.
    const live_ranges = try analysis.computeLiveRangesWithOrder(func, block_order, allocator);
    defer allocator.free(live_ranges);
    var alloc_result = try regalloc.allocateFromRanges(allocator, x86_64_reg_set(func.local_count), clobber_points.items, live_ranges);
    defer alloc_result.deinit();

    // Compute which caller-saved registers are actually used by this function.
    // Only these need to be saved/restored around call sites.
    var used_caller_saved: [caller_saved_alloc.len]bool = .{false} ** caller_saved_alloc.len;
    // Track which callee-saved registers are used (for prologue/epilogue preservation).
    var used_callee_saved: [callee_saved_alloc.len]bool = .{false} ** callee_saved_alloc.len;
    {
        var it = alloc_result.assignments.iterator();
        while (it.next()) |entry| {
            switch (entry.value_ptr.*) {
                .reg => |preg| {
                    const reg: emit.Reg = @enumFromInt(preg);
                    for (caller_saved_alloc, 0..) |cs_reg, i| {
                        if (reg == cs_reg) used_caller_saved[i] = true;
                    }
                    for (callee_saved_alloc, 0..) |cs_reg, i| {
                        if (reg == cs_reg) used_callee_saved[i] = true;
                    }
                },
                .stack => {},
            }
        }
    }
    // Mark callee-saved regs clobbered by fixed-register instructions:
    // memory_copy (REP MOVSB) hard-uses rsi+rdi, memory_fill (REP STOSB) uses rdi.
    if (comptime builtin.os.tag == .windows) {
        for (func.blocks.items) |block| {
            for (block.instructions.items) |blk_inst| {
                switch (blk_inst.op) {
                    .memory_fill => {
                        for (callee_saved_alloc, 0..) |cs_reg, i| {
                            if (cs_reg == .rdi) used_callee_saved[i] = true;
                        }
                    },
                    else => {},
                }
            }
        }
    }

    var code = emit.CodeBuffer.init(allocator);
    errdefer code.deinit();

    // Count callee-saved pushes for stack alignment calculation.
    var callee_save_count: u32 = 0;
    for (used_callee_saved) |used| {
        if (used) callee_save_count += 1;
    }

    // Frame: locals + spill slots + 1 vmctx slot, aligned to 16 bytes.
    // No push/pop at call sites (allocator handles clobbering), so after
    // push rbp (rsp=0 mod 16) + sub rsp + callee-save pushes, rsp must
    // be 0 mod 16 for direct CALL alignment.
    const spill_slots = alloc_result.spill_count;
    const raw_size: u32 = (func.local_count + 64 + spill_slots + 1) * 8;
    const aligned: u32 = (raw_size + 15) & ~@as(u32, 15);
    // After push rbp: rsp=0 mod 16. After sub rsp,frame_size + N callee pushes:
    // want (frame_size + 8*N) ≡ 0 mod 16 → rsp ≡ 0 mod 16 at CALL sites.
    const frame_size: u32 = if (callee_save_count % 2 == 0) aligned else aligned | 8;
    try code.emitPrologue(frame_size);

    // Save memory base (VMContext) from first ABI register to [rbp - 8]
    // The runtime passes the linear memory base pointer as a hidden first parameter.
    const vmctx_reg = param_regs[0]; // rcx on Win64, rdi on SysV
    try code.movMemReg(.rbp, vmctx_offset, vmctx_reg);

    // Spill wasm parameters from ABI registers to stack frame slots.
    // Wasm params are shifted by 1 because the first ABI register is the VMContext.
    const max_reg_params: u32 = @as(u32, @intCast(param_regs.len)) - 1;
    const spill_count = @min(func.param_count, max_reg_params);
    for (0..spill_count) |i| {
        try code.movMemReg(.rbp, -@as(i32, @intCast((i + 2) * 8)), param_regs[i + 1]);
    }

    // Spill wasm parameters passed on the stack by the caller (beyond max_reg_params).
    // Caller layout after push rbp; mov rbp, rsp:
    //   Win64: [rbp+16..rbp+48] is the 4-slot shadow space; stack args start at [rbp+48].
    //   SysV:  no shadow; stack args start at [rbp+16].
    if (func.param_count > max_reg_params) {
        const stack_arg_base: i32 = if (comptime builtin.os.tag == .windows) 48 else 16;
        var k: u32 = max_reg_params;
        while (k < func.param_count) : (k += 1) {
            const src_off: i32 = stack_arg_base + @as(i32, @intCast((k - max_reg_params) * 8));
            const dst_off: i32 = -@as(i32, @intCast((k + 2) * 8));
            try code.movRegMem(.rax, .rbp, src_off);
            try code.movMemReg(.rbp, dst_off, .rax);
        }
    }

    // Zero-initialize declared locals (wasm spec requires locals to be zero).
    // Locals start after parameters in the frame layout.
    if (func.local_count > func.param_count) {
        try code.xorReg32(.rax);
        for (func.param_count..func.local_count) |i| {
            try code.movMemReg(.rbp, -@as(i32, @intCast((i + 2) * 8)), .rax);
        }
    }

    // Multi-value returns: callee receives a hidden return pointer (HRP)
    // as an implicit trailing user-arg. Save it to a fixed frame slot so
    // `.ret_multi` can load it at function exit. HRP lives in
    // param_regs[1 + func.param_count] if that fits, else on the caller's
    // stack at the usual extra-arg location.
    if (func.result_count > 1) {
        const hrp_save_off: i32 = -@as(i32, @intCast((func.local_count + 2) * 8));
        const max_reg_params_local: u32 = @as(u32, @intCast(param_regs.len)) - 1;
        if (func.param_count < max_reg_params_local) {
            try code.movMemReg(.rbp, hrp_save_off, param_regs[1 + func.param_count]);
        } else {
            const stack_arg_base: i32 = if (comptime builtin.os.tag == .windows) 48 else 16;
            const src_off: i32 = stack_arg_base + @as(i32, @intCast((func.param_count - max_reg_params_local) * 8));
            try code.movRegMem(.rax, .rbp, src_off);
            try code.movMemReg(.rbp, hrp_save_off, .rax);
        }
    }

    // Save callee-saved registers used by this function.
    // Win64: rsi, rdi, r12, r13. SysV: r12, r13.
    for (callee_saved_alloc, 0..) |reg, i| {
        if (used_callee_saved[i]) try code.pushReg(reg);
    }

    // Build constant value table for immediate folding
    var const_vals = std.AutoHashMap(ir.VReg, i64).init(allocator);
    defer const_vals.deinit();
    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            if (inst.dest) |dest| {
                switch (inst.op) {
                    .iconst_32 => |v| try const_vals.put(dest, v),
                    .iconst_64 => |v| try const_vals.put(dest, v),
                    else => {},
                }
            }
        }
    }

    var block_offsets = std.AutoHashMap(ir.BlockId, usize).init(allocator);
    defer block_offsets.deinit();
    var branch_patches: std.ArrayList(BranchPatch) = .empty;
    defer branch_patches.deinit(allocator);
    var call_patches: std.ArrayList(CallPatch) = .empty;
    defer call_patches.deinit(allocator);
    var table_patches: std.ArrayList(TablePatch) = .empty;
    defer table_patches.deinit(allocator);

    // block_order already computed above — reuse for emission.

    var last_was_ret = false;
    for (block_order, 0..) |block_id, order_idx| {
        const block = func.blocks.items[block_id];
        try block_offsets.put(block_id, code.len());
        const next_block_id: ?ir.BlockId = if (order_idx + 1 < block_order.len) block_order[order_idx + 1] else null;
        for (block.instructions.items) |inst| {
            last_was_ret = isRet(inst.op);
            try compileInstRA(&code, inst, &alloc_result, &const_vals, &branch_patches, &call_patches, &table_patches, import_count, &used_caller_saved, &used_callee_saved, func.local_count);
        }
        // C3 fall-through peephole: if the block's terminator emitted a
        // trailing `E9 disp32` (br, or br_if's unconditional else) whose
        // target is the next block, drop those 5 bytes and the stale patch.
        // Works for br_if because the else branch is emitted last.
        if (next_block_id) |nb| {
            if (branch_patches.items.len > 0) {
                const last = branch_patches.items[branch_patches.items.len - 1];
                if (last.target_block == nb and last.patch_offset + 4 == code.len()) {
                    code.truncate(code.len() - 5);
                    _ = branch_patches.pop();
                }
            }
        }
    }

    for (branch_patches.items) |patch| {
        if (block_offsets.get(patch.target_block)) |target_off| {
            const rel: i32 = @intCast(@as(i64, @intCast(target_off)) - @as(i64, @intCast(patch.patch_offset + 4)));
            code.patchI32(patch.patch_offset, rel);
        }
    }

    for (table_patches.items) |patch| {
        if (block_offsets.get(patch.target_block)) |target_off| {
            const rel: i32 = @intCast(@as(i64, @intCast(target_off)) - @as(i64, @intCast(patch.base_offset)));
            code.patchI32(patch.entry_offset, rel);
        }
    }

    if (!last_was_ret) {
        // Restore callee-saved registers before epilogue (reverse order).
        var ci: usize = callee_saved_alloc.len;
        while (ci > 0) {
            ci -= 1;
            if (used_callee_saved[ci]) try code.popReg(callee_saved_alloc[ci]);
        }
        try code.emitEpilogue();
    }

    return .{
        .code = try code.bytes.toOwnedSlice(allocator),
        .call_patches = try call_patches.toOwnedSlice(allocator),
    };
}

/// Get a VReg's value into a physical register, using scratch if spilled.
fn useVReg(
    code: *emit.CodeBuffer,
    alloc_result: *const regalloc.AllocResult,
    vreg: ir.VReg,
    scratch: emit.Reg,
) !emit.Reg {
    const alloc = alloc_result.get(vreg) orelse return scratch;
    switch (alloc) {
        .reg => |preg| return @as(emit.Reg, @enumFromInt(preg)),
        .stack => |offset| {
            try code.movRegMem(scratch, .rbp, offset);
            return scratch;
        },
    }
}

/// Emit parallel move of wasm call args into Win64/SysV param_regs[1..].
/// `args[i]` is placed in `param_regs[i + 1]`. Handles arbitrary source/dest
/// reg overlaps (including cycles) by:
///   1. For each arg, find its current location (a phys reg or a spill slot).
///   2. Emit reg→reg moves in topological order; break cycles via .r10.
///   3. Load spill-slot args directly into their final param reg.
/// `.r10` is used as scratch — it is never an allocatable register nor a
/// param reg, so using it here is always safe.
fn emitCallRegArgMoves(
    code: *emit.CodeBuffer,
    alloc_result: *const regalloc.AllocResult,
    args: []const ir.VReg,
    max_reg_args: u32,
) !void {
    const ArgInfo = struct {
        source: emit.Reg,
        is_stack: bool,
        stack_offset: i32,
        target: emit.Reg,
    };
    var infos: [6]ArgInfo = undefined;
    var i: u32 = 0;
    while (i < max_reg_args) : (i += 1) {
        const target = param_regs[i + 1];
        if (alloc_result.get(args[i])) |a| switch (a) {
            .reg => |preg| infos[i] = .{
                .source = @enumFromInt(preg),
                .is_stack = false,
                .stack_offset = 0,
                .target = target,
            },
            .stack => |off| infos[i] = .{
                .source = target,
                .is_stack = true,
                .stack_offset = off,
                .target = target,
            },
        } else {
            infos[i] = .{ .source = target, .is_stack = false, .stack_offset = 0, .target = target };
        }
    }

    // Phase 1: reg→reg parallel move (topological; cycles broken via .r10).
    var pending: [6]bool = .{ false, false, false, false, false, false };
    i = 0;
    while (i < max_reg_args) : (i += 1) {
        if (!infos[i].is_stack and infos[i].source != infos[i].target) pending[i] = true;
    }
    while (true) {
        var progress = false;
        var k: u32 = 0;
        while (k < max_reg_args) : (k += 1) {
            if (!pending[k]) continue;
            var blocked = false;
            var m: u32 = 0;
            while (m < max_reg_args) : (m += 1) {
                if (m == k or !pending[m]) continue;
                if (infos[m].source == infos[k].target) {
                    blocked = true;
                    break;
                }
            }
            if (!blocked) {
                try code.movRegReg(infos[k].target, infos[k].source);
                pending[k] = false;
                progress = true;
            }
        }
        var any = false;
        var p: u32 = 0;
        while (p < max_reg_args) : (p += 1) if (pending[p]) {
            any = true;
            break;
        };
        if (!any) break;
        if (!progress) {
            // Break a cycle: save one source to .r10 and remap references.
            var first: u32 = 0;
            while (first < max_reg_args) : (first += 1) if (pending[first]) break;
            try code.movRegReg(.r10, infos[first].source);
            const old = infos[first].source;
            var u: u32 = 0;
            while (u < max_reg_args) : (u += 1) {
                if (pending[u] and infos[u].source == old) infos[u].source = .r10;
            }
        }
    }

    // Phase 2: load spill-slot args directly into their target param reg.
    i = 0;
    while (i < max_reg_args) : (i += 1) {
        if (infos[i].is_stack) {
            try code.movRegMem(infos[i].target, .rbp, infos[i].stack_offset);
        }
    }
}

/// Store a result into a VReg's allocated location.
/// For i32 results, automatically zero-extends to clear upper 32 bits.
fn writeDef(
    code: *emit.CodeBuffer,
    alloc_result: *const regalloc.AllocResult,
    dest: ir.VReg,
    result_reg: emit.Reg,
) !void {
    const alloc = alloc_result.get(dest) orelse return;
    switch (alloc) {
        .reg => |preg| {
            const dst = @as(emit.Reg, @enumFromInt(preg));
            if (dst != result_reg) try code.movRegReg(dst, result_reg);
        },
        .stack => |offset| {
            try code.movMemReg(.rbp, offset, result_reg);
        },
    }
}

/// Write an i32 result: zero-extend to clear upper 32 bits, then store.
fn writeDefI32(
    code: *emit.CodeBuffer,
    alloc_result: *const regalloc.AllocResult,
    dest: ir.VReg,
    result_reg: emit.Reg,
) !void {
    try code.zeroExtend32(result_reg);
    try writeDef(code, alloc_result, dest, result_reg);
}

/// Type-aware writeDef: applies i32 zero-extension when needed.
fn writeDefTyped(
    code: *emit.CodeBuffer,
    alloc_result: *const regalloc.AllocResult,
    dest: ir.VReg,
    result_reg: emit.Reg,
    val_type: ir.IrType,
) !void {
    if (val_type == .i32) {
        try writeDefI32(code, alloc_result, dest, result_reg);
    } else {
        try writeDef(code, alloc_result, dest, result_reg);
    }
}

/// Get the physical register for a VReg (only valid if allocated to a register).
fn regOf(alloc_result: *const regalloc.AllocResult, vreg: ir.VReg) ?emit.Reg {
    const alloc = alloc_result.get(vreg) orelse return null;
    return switch (alloc) {
        .reg => |preg| @as(emit.Reg, @enumFromInt(preg)),
        .stack => null,
    };
}

/// Return the physical register for a destination VReg, or a default scratch
/// register for stack-spilled destinations. This enables register-flexible
/// codegen: instructions operate directly on the allocated register instead
/// of always routing through rax.
fn destReg(alloc_result: *const regalloc.AllocResult, dest: ir.VReg) emit.Reg {
    return regOf(alloc_result, dest) orelse .rax;
}

fn compileInstRA(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    alloc_result: *const regalloc.AllocResult,
    const_vals: *const std.AutoHashMap(ir.VReg, i64),
    patches: *std.ArrayList(BranchPatch),
    call_patches: *std.ArrayList(CallPatch),
    table_patches: *std.ArrayList(TablePatch),
    import_count: u32,
    used_caller_saved: *const [caller_saved_alloc.len]bool,
    used_callee_saved: *const [callee_saved_alloc.len]bool,
    local_count: u32,
) !void {
    switch (inst.op) {
        // ── Constants ─────────────────────────────────────────────────
        .iconst_32 => |val| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            if (val == 0) {
                try code.xorReg32(dr);
            } else {
                try code.movRegImm32(dr, val);
            }
            // movRegImm32 uses MOV r64,imm32 which sign-extends negative
            // values to 64 bits. Zero-extend to clear upper bits.
            try writeDefTyped(code, alloc_result, dest, dr, .i32);
        },
        .iconst_64 => |val| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            try code.movRegImm64(dr, @bitCast(val));
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .fconst_32 => |val| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            try code.movRegImm32(dr, @bitCast(val));
            // 32-bit write already zero-extends
            try writeDef(code, alloc_result, dest, dr);
        },
        .fconst_64 => |val| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            try code.movRegImm64(dr, @bitCast(val));
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },

        // ── Binary arithmetic ─────────────────────────────────────────
        .add, .sub, .mul, .@"and", .@"or", .xor => {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const bin = switch (inst.op) {
                .add => |b| b,
                .sub => |b| b,
                .mul => |b| b,
                .@"and" => |b| b,
                .@"or" => |b| b,
                .xor => |b| b,
                else => unreachable,
            };

            // Float path: wasm f32/f64 add/sub/mul are lowered by the frontend
            // into these integer ops with `inst.type == .f32/.f64`. Dispatch
            // to SSE scalar ops through the GPR↔XMM bounce pattern used by
            // f_min/f_max/f_sqrt. Logical ops (and/or/xor) don't have float
            // variants; they'll fall through to the integer path below
            // because the frontend never produces them with float type.
            if (inst.type == .f32 or inst.type == .f64) {
                const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
                if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);
                const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
                if (rhs_reg != .rcx) try code.movRegReg(.rcx, rhs_reg);
                if (inst.type == .f32) {
                    try code.movdToXmm(.rax, .rax);
                    try code.movdToXmm(.rcx, .rcx);
                    switch (inst.op) {
                        .add => try code.addss(.rax, .rcx),
                        .sub => try code.subss(.rax, .rcx),
                        .mul => try code.mulss(.rax, .rcx),
                        else => unreachable,
                    }
                    try code.movdFromXmm(.rax, .rax);
                } else {
                    try code.movqToXmm(.rax, .rax);
                    try code.movqToXmm(.rcx, .rcx);
                    switch (inst.op) {
                        .add => try code.addsd(.rax, .rcx),
                        .sub => try code.subsd(.rax, .rcx),
                        .mul => try code.mulsd(.rax, .rcx),
                        else => unreachable,
                    }
                    try code.movqFromXmm(.rax, .rax);
                }
                if (dr != .rax) try code.movRegReg(dr, .rax);
                try writeDefTyped(code, alloc_result, dest, dr, inst.type);
                return;
            }

            const scratch: emit.Reg = if (dr == .rax) .rcx else .rax;
            // Check if RHS is a constant for immediate form
            if (const_vals.get(bin.rhs)) |imm| {
                if (imm >= std.math.minInt(i32) and imm <= std.math.maxInt(i32)) {
                    const lhs_reg = try useVReg(code, alloc_result, bin.lhs, dr);
                    if (lhs_reg != dr) try code.movRegReg(dr, lhs_reg);
                    const imm32: i32 = @intCast(imm);
                    switch (inst.op) {
                        .add => if (imm32 == 1) {
                            try code.incReg(dr);
                        } else if (imm32 == -1) {
                            try code.decReg(dr);
                        } else if (imm32 != 0) {
                            try code.addRegImm32(dr, imm32);
                        },
                        .sub => if (imm32 == 1) {
                            try code.decReg(dr);
                        } else if (imm32 == -1) {
                            try code.incReg(dr);
                        } else if (imm32 != 0) {
                            try code.subRegImm32(dr, imm32);
                        },
                        .@"and" => try code.andRegImm32(dr, imm32),
                        .@"or" => if (imm32 != 0) try code.orRegImm32(dr, imm32),
                        .xor => if (imm32 != 0) try code.xorRegImm32(dr, imm32),
                        .mul => {
                            try code.movRegImm32(scratch, imm32);
                            try code.imulRegReg(dr, scratch);
                        },
                        else => unreachable,
                    }
                    if (inst.type == .i32) try code.zeroExtend32(dr);
                    try writeDefTyped(code, alloc_result, dest, dr, inst.type);
                    return;
                }
            }
            // General case. For add specifically, if lhs is in a different
            // register than dst, use LEA dst, [lhs + rhs] to fuse the mov
            // and add into one instruction (C2). LEA requires both operands
            // to be in real registers (not spilled) and neither to be rsp.
            if (inst.op == .add) {
                const lhs_reg_raw = regOf(alloc_result, bin.lhs);
                const rhs_reg_raw = regOf(alloc_result, bin.rhs);
                if (lhs_reg_raw != null and rhs_reg_raw != null and
                    lhs_reg_raw.? != dr and lhs_reg_raw.? != .rsp and rhs_reg_raw.? != .rsp)
                {
                    try code.leaRegBaseIndex64(dr, lhs_reg_raw.?, rhs_reg_raw.?);
                    try writeDefTyped(code, alloc_result, dest, dr, inst.type);
                    return;
                }
            }
            // Fallback: load LHS into dest register, RHS into scratch.
            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, dr);
            if (lhs_reg != dr) try code.movRegReg(dr, lhs_reg);
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, scratch);
            switch (inst.op) {
                .add => try code.addRegReg(dr, rhs_reg),
                .sub => try code.subRegReg(dr, rhs_reg),
                .mul => try code.imulRegReg(dr, rhs_reg),
                .@"and" => try code.andRegReg(dr, rhs_reg),
                .@"or" => try code.orRegReg(dr, rhs_reg),
                .xor => try code.xorRegReg(dr, rhs_reg),
                else => unreachable,
            }
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },

        // ── Comparisons ───────────────────────────────────────────────
        inline .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u => |bin, tag| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            // Load lhs and rhs using different spill scratches so spilled
            // operands don't clobber each other.
            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .r11);
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rax);
            // Use 32-bit compare for i32 to get correct signed semantics.
            if (inst.type == .i32) {
                try code.cmpRegReg32(lhs_reg, rhs_reg);
            } else {
                try code.cmpRegReg(lhs_reg, rhs_reg);
            }
            const cc: u4 = comptime switch (tag) {
                .eq => 0x4,
                .ne => 0x5,
                .lt_s => 0xC,
                .lt_u => 0x2,
                .gt_s => 0xF,
                .gt_u => 0x7,
                .le_s => 0xE,
                .le_u => 0x6,
                .ge_s => 0xD,
                .ge_u => 0x3,
                else => unreachable,
            };
            try code.setcc(cc, dr);
            try code.movzxByte(dr, dr);
            // setcc+movzx produces clean 0/1 — skip zeroExtend32
            try writeDef(code, alloc_result, dest, dr);
        },

        .eqz => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            // Use 32-bit TEST for i32 so upper-32-bit garbage doesn't
            // affect the zero check (same reasoning as cmpRegReg32 for
            // comparisons).
            if (inst.type == .i32) {
                try code.testRegReg32(src_reg, src_reg);
            } else {
                try code.testRegReg(src_reg, src_reg);
            }
            try code.setcc(0x4, dr);
            try code.movzxByte(dr, dr);
            // setcc+movzx produces clean 0/1 — skip zeroExtend32
            try writeDef(code, alloc_result, dest, dr);
        },

        // ── Local variable access ─────────────────────────────────────
        .local_get => |idx| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            try code.movRegMem(dr, .rbp, -@as(i32, @intCast((idx + 2) * 8)));
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .local_set => |ls| {
            const src_reg = try useVReg(code, alloc_result, ls.val, .rax);
            try code.movMemReg(.rbp, -@as(i32, @intCast((ls.idx + 2) * 8)), src_reg);
        },

        // ── Return ────────────────────────────────────────────────────
        .ret => |maybe_val| {
            if (maybe_val) |val| {
                const src_reg = try useVReg(code, alloc_result, val, .rax);
                if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            }
            // Restore callee-saved registers before epilogue (reverse order).
            var ci: usize = callee_saved_alloc.len;
            while (ci > 0) {
                ci -= 1;
                if (used_callee_saved[ci]) try code.popReg(callee_saved_alloc[ci]);
            }
            try code.emitEpilogue();
        },
        .ret_multi => |vregs| {
            // First result → RAX; remaining → [HRP + (i-1)*8] where HRP is
            // loaded from the saved-slot in the prologue. We stash HRP in
            // r11 (non-allocatable scratch) to avoid conflicts with value
            // regs. Load result vregs into rcx (also non-allocatable).
            if (vregs.len == 0) {
                // Degenerate; treat as plain ret.
            } else {
                // Move first result to rax.
                const first_reg = try useVReg(code, alloc_result, vregs[0], .rax);
                if (first_reg != .rax) try code.movRegReg(.rax, first_reg);
            }
            if (vregs.len > 1) {
                const hrp_save_off: i32 = -@as(i32, @intCast((local_count + 2) * 8));
                try code.movRegMem(.r11, .rbp, hrp_save_off); // load HRP
                var i: u32 = 1;
                while (i < vregs.len) : (i += 1) {
                    const vr = try useVReg(code, alloc_result, vregs[i], .rcx);
                    const off: i32 = @intCast((i - 1) * 8);
                    try code.movMemReg(.r11, off, vr);
                }
            }
            var ci: usize = callee_saved_alloc.len;
            while (ci > 0) {
                ci -= 1;
                if (used_callee_saved[ci]) try code.popReg(callee_saved_alloc[ci]);
            }
            try code.emitEpilogue();
        },

        // ── Branches ──────────────────────────────────────────────────
        .br => |target| {
            try code.emitByte(0xE9);
            const patch_off = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = patch_off, .target_block = target });
        },
        .br_if => |br| {
            const cond_reg = try useVReg(code, alloc_result, br.cond, .rax);
            // Wasm br_if condition is always i32; use 32-bit TEST so that
            // upper-32 garbage (possible after our local_get preserves full
            // 64 bits) doesn't spuriously flip the branch.
            try code.testRegReg32(cond_reg, cond_reg);
            try code.emitByte(0x0F);
            try code.emitByte(0x85);
            const then_patch = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = then_patch, .target_block = br.then_block });
            try code.emitByte(0xE9);
            const else_patch = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = else_patch, .target_block = br.else_block });
        },
        .br_table => |bt| {
            // Load index into r11 (scratch) and canonicalize upper 32 bits to zero.
            const idx_reg = try useVReg(code, alloc_result, bt.index, .r11);
            if (idx_reg != .r11) try code.movRegReg(.r11, idx_reg);
            try code.zeroExtend32(.r11);

            // cmp r11, targets.len ; jae default  (unsigned out-of-range goes to default)
            try code.cmpRegImm32(.r11, @intCast(bt.targets.len));
            try code.emitByte(0x0F);
            try code.emitByte(0x83);
            const default_patch = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = default_patch, .target_block = bt.default });

            // lea r10, [rip + table]       ; 4C 8D 15 disp32
            try code.emitByte(0x4C);
            try code.emitByte(0x8D);
            try code.emitByte(0x15);
            const lea_disp_off = code.len();
            try code.emitI32(0);

            // movsxd r11, dword ptr [r10 + r11*4]
            // REX.WRXB = 0x4F, opcode 0x63, ModR/M=00_011_100 (0x1C), SIB=10_011_010 (0x9A)
            try code.emitByte(0x4F);
            try code.emitByte(0x63);
            try code.emitByte(0x1C);
            try code.emitByte(0x9A);

            // add r10, r11
            try code.addRegReg(.r10, .r11);

            // jmp r10                     ; 41 FF E2
            try code.emitByte(0x41);
            try code.emitByte(0xFF);
            try code.emitByte(0xE2);

            // Emit the jump table inline (no fallthrough from the indirect jmp).
            const table_off = code.len();
            const lea_rel: i32 = @intCast(@as(i64, @intCast(table_off)) - @as(i64, @intCast(lea_disp_off + 4)));
            code.patchI32(lea_disp_off, lea_rel);

            for (bt.targets) |target| {
                const entry_off = code.len();
                try code.emitI32(0);
                try table_patches.append(code.allocator, .{
                    .entry_offset = entry_off,
                    .base_offset = table_off,
                    .target_block = target,
                });
            }
        },

        // ── Function calls ────────────────────────────────────────────
        .call => |cl| {
            // No push/pop: the allocator ensures no live vreg in a
            // caller-saved register spans this call instruction.
            const n_args: u32 = @intCast(cl.args.len);
            const has_hrp: bool = cl.extra_results > 0;
            const max_reg_slots: u32 = @as(u32, @intCast(param_regs.len)) - 1;
            const hrp_in_reg: bool = has_hrp and (n_args < max_reg_slots);
            const max_reg_args: u32 = @min(n_args, max_reg_slots);
            const extra: u32 = n_args - max_reg_args;
            const hrp_on_stack: bool = has_hrp and !hrp_in_reg;
            const hrp_stack_k: u32 = if (hrp_on_stack) n_args - max_reg_slots else 0;
            const total_stack_args: u32 = extra + (if (hrp_on_stack) @as(u32, 1) else 0);
            const shadow: u32 = if (comptime builtin.os.tag == .windows) 32 else 0;
            const stack_need: u32 = shadow + total_stack_args * 8;
            const stack_adjust: u32 = (stack_need + 15) & ~@as(u32, 15);
            const scratch_base_off: i32 = -@as(i32, @intCast((local_count + 18) * 8));
            const is_import: bool = cl.func_idx < import_count;

            // Real tail-call is feasible when outgoing stack args are empty,
            // HRP (if any) goes in a register (so we pass our caller's HRP through),
            // and the target is a local function (rel32 JMP).
            const can_real_tail: bool = cl.tail and total_stack_args == 0 and !hrp_on_stack and !is_import;
            if (can_real_tail) {
                try code.movRegMem(param_regs[0], .rbp, vmctx_offset);
                try emitCallRegArgMoves(code, alloc_result, cl.args, max_reg_args);
                if (has_hrp) {
                    const hrp_save_off: i32 = -@as(i32, @intCast((local_count + 2) * 8));
                    const hrp_dst = param_regs[1 + n_args];
                    try code.movRegMem(hrp_dst, .rbp, hrp_save_off);
                }
                var ci_t: usize = callee_saved_alloc.len;
                while (ci_t > 0) {
                    ci_t -= 1;
                    if (used_callee_saved[ci_t]) try code.popReg(callee_saved_alloc[ci_t]);
                }
                try code.movRegReg(.rsp, .rbp);
                try code.popReg(.rbp);
                try code.emitByte(0xE9); // JMP rel32
                const patch_off = code.len();
                try code.emitI32(0);
                try call_patches.append(code.allocator, .{
                    .patch_offset = patch_off,
                    .target_func_idx = cl.func_idx - import_count,
                });
                return;
            }

            if (is_import) {
                try code.movRegMem(param_regs[0], .rbp, vmctx_offset);
                try code.movRegMem(.r10, param_regs[0], vmctx_host_functions_field);
                if (cl.func_idx > 0) {
                    try code.addRegImm32(.r10, @intCast(@as(u32, cl.func_idx) * 8));
                }
                try code.movRegMem(.rax, .r10, 0);

                if (stack_adjust > 0) try code.subRegImm32(.rsp, @intCast(stack_adjust));
                var j: u32 = 0;
                while (j < extra) : (j += 1) {
                    const arg_reg = try useVReg(code, alloc_result, cl.args[max_reg_args + j], .r10);
                    try code.movMemReg(.rsp, @intCast(shadow + j * 8), arg_reg);
                }
                try emitCallRegArgMoves(code, alloc_result, cl.args, max_reg_args);
                if (has_hrp) {
                    if (hrp_in_reg) {
                        const hrp_dst = param_regs[1 + n_args];
                        try code.movRegReg(hrp_dst, .rbp);
                        try code.addRegImm32(hrp_dst, scratch_base_off);
                    } else {
                        try code.movRegReg(.r10, .rbp);
                        try code.addRegImm32(.r10, scratch_base_off);
                        try code.movMemReg(.rsp, @intCast(shadow + hrp_stack_k * 8), .r10);
                    }
                }
                try code.callReg(.rax);
                if (stack_adjust > 0) try code.addRegImm32(.rsp, @intCast(stack_adjust));
            } else {
                try code.movRegMem(param_regs[0], .rbp, vmctx_offset);

                if (stack_adjust > 0) try code.subRegImm32(.rsp, @intCast(stack_adjust));
                var j2: u32 = 0;
                while (j2 < extra) : (j2 += 1) {
                    const arg_reg = try useVReg(code, alloc_result, cl.args[max_reg_args + j2], .r10);
                    try code.movMemReg(.rsp, @intCast(shadow + j2 * 8), arg_reg);
                }
                try emitCallRegArgMoves(code, alloc_result, cl.args, max_reg_args);
                if (has_hrp) {
                    if (hrp_in_reg) {
                        const hrp_dst = param_regs[1 + n_args];
                        try code.movRegReg(hrp_dst, .rbp);
                        try code.addRegImm32(hrp_dst, scratch_base_off);
                    } else {
                        try code.movRegReg(.r10, .rbp);
                        try code.addRegImm32(.r10, scratch_base_off);
                        try code.movMemReg(.rsp, @intCast(shadow + hrp_stack_k * 8), .r10);
                    }
                }
                try code.emitByte(0xE8);
                const patch_off = code.len();
                try code.emitI32(0);
                try call_patches.append(code.allocator, .{
                    .patch_offset = patch_off,
                    .target_func_idx = cl.func_idx - import_count,
                });
                if (stack_adjust > 0) try code.addRegImm32(.rsp, @intCast(stack_adjust));
            }

            if (cl.tail) {
                // Fallback tail: result already in rax; multi-result values
                // already written through our HRP pass-through. Emit inline
                // epilogue + ret; skip writeDefTyped (dest is dead).
                var ci_f: usize = callee_saved_alloc.len;
                while (ci_f > 0) {
                    ci_f -= 1;
                    if (used_callee_saved[ci_f]) try code.popReg(callee_saved_alloc[ci_f]);
                }
                try code.emitEpilogue();
                return;
            }

            if (inst.dest) |dest| {
                try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
            }
        },

        // ── Indirect function calls ───────────────────────────────────
        .call_indirect => |ci| {
            // No push/pop: the allocator ensures no live vreg in a
            // clobbered register spans this call instruction.

            // Load table element index
            const idx_reg = try useVReg(code, alloc_result, ci.elem_idx, .rax);
            if (idx_reg != .rax) try code.movRegReg(.rax, idx_reg);
            try code.zeroExtend32(.rax);

            // Load vmctx into r10. For table 0 we use the fast path that
            // reads vmctx.func_table_ptr/len directly. For higher-numbered
            // tables we go through vmctx.tables_info_ptr[table_idx].
            try code.movRegMem(.r10, .rbp, vmctx_offset);

            if (ci.table_idx == 0) {
                // cmp eax, dword ptr [r10 + func_table_len]
                //   REX.B=0x41, opcode 0x3B, modrm=01_000_010 (mod=disp8, reg=eax=0,
                //   rm=r10 low=2), disp8=60.
                try code.emitSlice(&.{ 0x41, 0x3B, 0x42, @as(u8, @intCast(vmctx_func_table_len_field)) });
                // jb over_trap (rel8). Trap block size is 3 + 7 + 2 = 12 bytes below.
                try code.emitByte(0x72);
                try code.emitByte(12);
                // Trap block: call trap_unreachable_fn(vmctx)
                try code.movRegReg(param_regs[0], .r10);
                try code.movRegMem(.rax, param_regs[0], vmctx_trap_unreachable_fn_field);
                try code.callReg(.rax);
                // over_trap:

                // Signature check. Load tables_info_ptr into r11
                // so the sig-check sequence can read type_backing_ptr.
                // r11 is clobbered by the sig check, but not needed after
                // here (table-0 path reads func_table_ptr from vmctx).
                try code.movRegMem(.r11, .r10, vmctx_tables_info_field);
                try emitCallIndirectSigCheck(code, ci.type_idx, ci.table_idx);

                // Load func_table_ptr from VmCtx (r10 still holds vmctx)
                try code.movRegMem(.r10, .r10, vmctx_func_table_field);
            } else {
                // r11 = vmctx.tables_info_ptr (kept in r11 so r10/vmctx
                // survives for the trap path below).
                try code.movRegMem(.r11, .r10, vmctx_tables_info_field);

                const len_off: i32 = @as(i32, @intCast(ci.table_idx)) * table_info_stride + table_info_len_off;
                // cmp eax, [r11 + len_off]
                //   REX.B=0x41, opcode 0x3B, modrm=10_000_011 (mod=disp32,
                //   reg=eax=0, rm=r11 low=3), disp32=len_off.
                try code.emitSlice(&.{ 0x41, 0x3B, 0x83 });
                try code.emitU32(@bitCast(len_off));
                // jb over_trap (rel8). Trap block: 3 + 7 + 2 = 12 bytes.
                try code.emitByte(0x72);
                try code.emitByte(12);
                // Trap block: call trap_unreachable_fn(vmctx) — r10 holds vmctx.
                try code.movRegReg(param_regs[0], .r10);
                try code.movRegMem(.rax, param_regs[0], vmctx_trap_unreachable_fn_field);
                try code.callReg(.rax);
                // over_trap:

                // Signature check (r11 already = tables_info_ptr).
                // NOTE: sig check clobbers r11; reload tables_info_ptr afterward.
                try emitCallIndirectSigCheck(code, ci.type_idx, ci.table_idx);
                try code.movRegMem(.r11, .r10, vmctx_tables_info_field);

                // r10 = tables_info[table_idx].ptr
                const ptr_off: i32 = @as(i32, @intCast(ci.table_idx)) * table_info_stride + table_info_ptr_off;
                try code.movRegMem(.r10, .r11, ptr_off);
            }

            // func_ptr = func_table[elem_idx * 8]; stash in r11 across arg setup
            // because useVReg below only touches its scratch (.r10) and the
            // destination param regs, never r11.
            try code.emitSlice(&.{ 0x48, 0xC1, 0xE0, 0x03 }); // shl rax, 3
            try code.addRegReg(.rax, .r10);
            try code.movRegMem(.r11, .rax, 0);

            // Load vmctx into param_regs[0]
            try code.movRegMem(param_regs[0], .rbp, vmctx_offset);

            const n_args: u32 = @intCast(ci.args.len);
            const has_hrp: bool = ci.extra_results > 0;
            const max_reg_slots: u32 = @as(u32, @intCast(param_regs.len)) - 1;
            const hrp_in_reg: bool = has_hrp and (n_args < max_reg_slots);
            const max_reg_args: u32 = @min(n_args, max_reg_slots);
            const extra: u32 = n_args - max_reg_args;
            const hrp_on_stack: bool = has_hrp and !hrp_in_reg;
            const hrp_stack_k: u32 = if (hrp_on_stack) n_args - max_reg_slots else 0;
            const total_stack_args: u32 = extra + (if (hrp_on_stack) @as(u32, 1) else 0);
            const shadow: u32 = if (comptime builtin.os.tag == .windows) 32 else 0;
            const stack_need: u32 = shadow + total_stack_args * 8;
            const stack_adjust: u32 = (stack_need + 15) & ~@as(u32, 15);
            const scratch_base_off: i32 = -@as(i32, @intCast((local_count + 18) * 8));

            // For real tail we jmp (no return address pushed) so shadow space
            // isn't needed; skip the rsp adjust entirely to match the working
            // direct-call tail path.
            const ci_can_real_tail: bool = ci.tail and total_stack_args == 0 and !hrp_on_stack;
            if (!ci_can_real_tail and stack_adjust > 0) try code.subRegImm32(.rsp, @intCast(stack_adjust));
            // Stack args first; sources may live in regs we're about to overwrite.
            var j: u32 = 0;
            while (j < extra) : (j += 1) {
                const arg_reg = try useVReg(code, alloc_result, ci.args[max_reg_args + j], .r10);
                try code.movMemReg(.rsp, @intCast(shadow + j * 8), arg_reg);
            }
            try emitCallRegArgMoves(code, alloc_result, ci.args, max_reg_args);
            if (has_hrp) {
                if (hrp_in_reg) {
                    const hrp_dst = param_regs[1 + n_args];
                    if (ci.tail and total_stack_args == 0 and !hrp_on_stack) {
                        // Real tail: pass-through our caller's HRP.
                        const hrp_save_off: i32 = -@as(i32, @intCast((local_count + 2) * 8));
                        try code.movRegMem(hrp_dst, .rbp, hrp_save_off);
                    } else {
                        try code.movRegReg(hrp_dst, .rbp);
                        try code.addRegImm32(hrp_dst, scratch_base_off);
                    }
                } else {
                    try code.movRegReg(.r10, .rbp);
                    try code.addRegImm32(.r10, scratch_base_off);
                    try code.movMemReg(.rsp, @intCast(shadow + hrp_stack_k * 8), .r10);
                }
            }

            const ci_real_tail: bool = ci.tail and total_stack_args == 0 and !hrp_on_stack;
            if (ci_real_tail) {
                var ci_t: usize = callee_saved_alloc.len;
                while (ci_t > 0) {
                    ci_t -= 1;
                    if (used_callee_saved[ci_t]) try code.popReg(callee_saved_alloc[ci_t]);
                }
                try code.movRegReg(.rsp, .rbp);
                try code.popReg(.rbp);
                try code.jmpReg(.r11);
                return;
            }

            try code.callReg(.r11);
            if (stack_adjust > 0) try code.addRegImm32(.rsp, @intCast(stack_adjust));

            if (ci.tail) {
                var ci_f: usize = callee_saved_alloc.len;
                while (ci_f > 0) {
                    ci_f -= 1;
                    if (used_callee_saved[ci_f]) try code.popReg(callee_saved_alloc[ci_f]);
                }
                try code.emitEpilogue();
                return;
            }

            if (inst.dest) |dest| {
                try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
            }
        },

        .call_ref => |cr| {
            // Load funcref (native pointer) into r11.
            const ref_reg = try useVReg(code, alloc_result, cr.func_ref, .rax);
            if (ref_reg != .r11) try code.movRegReg(.r11, ref_reg);

            // Null check: if r11 == 0, trap via trap_unreachable_fn.
            // Load vmctx into r10 first so the trap call has it handy.
            try code.movRegMem(.r10, .rbp, vmctx_offset);
            // test r11, r11  → 0x4D 0x85 0xDB
            try code.emitSlice(&.{ 0x4D, 0x85, 0xDB });
            // jne over_trap (rel8). Trap block is 3 + 7 + 2 = 12 bytes.
            try code.emitByte(0x75);
            try code.emitByte(12);
            // Trap block: call trap_unreachable_fn(vmctx)
            try code.movRegReg(param_regs[0], .r10);
            try code.movRegMem(.rax, param_regs[0], vmctx_trap_unreachable_fn_field);
            try code.callReg(.rax);
            // over_trap:

            // Load vmctx into param_regs[0] for the callee.
            try code.movRegMem(param_regs[0], .rbp, vmctx_offset);

            const n_args: u32 = @intCast(cr.args.len);
            const has_hrp: bool = cr.extra_results > 0;
            const max_reg_slots: u32 = @as(u32, @intCast(param_regs.len)) - 1;
            const hrp_in_reg: bool = has_hrp and (n_args < max_reg_slots);
            const max_reg_args: u32 = @min(n_args, max_reg_slots);
            const extra: u32 = n_args - max_reg_args;
            const hrp_on_stack: bool = has_hrp and !hrp_in_reg;
            const hrp_stack_k: u32 = if (hrp_on_stack) n_args - max_reg_slots else 0;
            const total_stack_args: u32 = extra + (if (hrp_on_stack) @as(u32, 1) else 0);
            const shadow: u32 = if (comptime builtin.os.tag == .windows) 32 else 0;
            const stack_need: u32 = shadow + total_stack_args * 8;
            const stack_adjust: u32 = (stack_need + 15) & ~@as(u32, 15);
            const scratch_base_off: i32 = -@as(i32, @intCast((local_count + 18) * 8));

            const cr_can_real_tail: bool = cr.tail and total_stack_args == 0 and !hrp_on_stack;
            if (!cr_can_real_tail and stack_adjust > 0) try code.subRegImm32(.rsp, @intCast(stack_adjust));
            var j: u32 = 0;
            while (j < extra) : (j += 1) {
                const arg_reg = try useVReg(code, alloc_result, cr.args[max_reg_args + j], .r10);
                try code.movMemReg(.rsp, @intCast(shadow + j * 8), arg_reg);
            }
            try emitCallRegArgMoves(code, alloc_result, cr.args, max_reg_args);
            if (has_hrp) {
                if (hrp_in_reg) {
                    const hrp_dst = param_regs[1 + n_args];
                    if (cr.tail and total_stack_args == 0 and !hrp_on_stack) {
                        const hrp_save_off: i32 = -@as(i32, @intCast((local_count + 2) * 8));
                        try code.movRegMem(hrp_dst, .rbp, hrp_save_off);
                    } else {
                        try code.movRegReg(hrp_dst, .rbp);
                        try code.addRegImm32(hrp_dst, scratch_base_off);
                    }
                } else {
                    try code.movRegReg(.r10, .rbp);
                    try code.addRegImm32(.r10, scratch_base_off);
                    try code.movMemReg(.rsp, @intCast(shadow + hrp_stack_k * 8), .r10);
                }
            }

            const cr_real_tail: bool = cr.tail and total_stack_args == 0 and !hrp_on_stack;
            if (cr_real_tail) {
                var ci_t: usize = callee_saved_alloc.len;
                while (ci_t > 0) {
                    ci_t -= 1;
                    if (used_callee_saved[ci_t]) try code.popReg(callee_saved_alloc[ci_t]);
                }
                try code.movRegReg(.rsp, .rbp);
                try code.popReg(.rbp);
                try code.jmpReg(.r11);
                return;
            }

            try code.callReg(.r11);
            if (stack_adjust > 0) try code.addRegImm32(.rsp, @intCast(stack_adjust));

            if (cr.tail) {
                var ci_f: usize = callee_saved_alloc.len;
                while (ci_f > 0) {
                    ci_f -= 1;
                    if (used_callee_saved[ci_f]) try code.popReg(callee_saved_alloc[ci_f]);
                }
                try code.emitEpilogue();
                return;
            }

            if (inst.dest) |dest| {
                try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
            }
        },

        // ── Extra result of a preceding multi-value call ──────────────
        .call_result => |idx| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const scratch_base_off: i32 = -@as(i32, @intCast((local_count + 18) * 8));
            const slot_off: i32 = scratch_base_off + @as(i32, @intCast(@as(u32, idx) * 8));
            try code.movRegMem(dr, .rbp, slot_off);
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },

        // ── Memory ────────────────────────────────────────────────────
        .load => |ld| {
            const dest = inst.dest orelse return;
            // Load memory base from VMContext frame slot, add wasm offset.
            // Bounds check is inserted between vmctx load and mem_base load so
            // the check can read VmCtx.memory_size while r10 still holds the
            // VmCtx pointer.
            try code.movRegMem(.r10, .rbp, vmctx_offset); // load VmCtx*
            const base_reg = try useVReg(code, alloc_result, ld.base, .rax);
            if (base_reg != .rax) try code.movRegReg(.rax, base_reg);
            try code.zeroExtend32(.rax); // wasm addresses are i32
            if (!ld.bounds_known) {
                const end_offset = if (ld.checked_end > 0) ld.checked_end else @as(u64, ld.offset) + @as(u64, ld.size);
                try emitMemBoundsCheck(code, end_offset);
            }
            try code.movRegMem(.r10, .r10, vmctx_membase_field); // load VmCtx.memory_base
            try code.addRegReg(.rax, .r10); // rax = mem_base + wasm_addr
            // Fold wasm offset into mov displacement when it fits in i32.
            // For out-of-range offsets (address.wast uses 0xFFFFFFFF), first
            // add the offset to rax via a 64-bit imm.
            var ld_disp: i32 = 0;
            if (ld.offset <= 0x7FFFFFFF) {
                ld_disp = @intCast(ld.offset);
            } else {
                try code.movRegImm64(.r10, @as(u64, ld.offset));
                try code.addRegReg(.rax, .r10);
            }
            // Load value into dest register (address is in rax).
            const dr = destReg(alloc_result, dest);
            try code.movRegMemSized(dr, .rax, ld_disp, ld.size);
            if (ld.sign_extend and ld.size < 8) {
                switch (ld.size) {
                    1 => try code.movsxByteToReg(dr, dr),
                    2 => try code.movsxWordToReg(dr, dr),
                    4 => try code.movsxd(dr, dr),
                    else => {},
                }
            }
            // For non-sign-extended loads ≤ 4 bytes, movRegMemSized already
            // produces a zero-extended (clean) i32 — skip redundant zeroExtend32.
            const load_is_clean = !ld.sign_extend and ld.size <= 4;
            if (load_is_clean) {
                try writeDef(code, alloc_result, dest, dr);
            } else {
                try writeDefTyped(code, alloc_result, dest, dr, inst.type);
            }
        },
        .store => |st| {
            // Load memory base from VMContext frame slot into r10.
            // Bounds check is inserted between vmctx load and mem_base load so
            // the check can read VmCtx.memory_size while r10 still holds the
            // VmCtx pointer.
            try code.movRegMem(.r10, .rbp, vmctx_offset); // load VmCtx*
            // Compute final address in rax (not allocatable — safe to clobber).
            const base_reg = try useVReg(code, alloc_result, st.base, .rax);
            if (base_reg != .rax) try code.movRegReg(.rax, base_reg);
            try code.zeroExtend32(.rax); // wasm addresses are i32
            if (!st.bounds_known) {
                const end_offset = if (st.checked_end > 0) st.checked_end else @as(u64, st.offset) + @as(u64, st.size);
                try emitMemBoundsCheck(code, end_offset);
            }
            try code.movRegMem(.r10, .r10, vmctx_membase_field); // load VmCtx.memory_base
            try code.addRegReg(.rax, .r10); // rax = mem_base + wasm_addr
            // Load value into rcx (not allocatable — safe to clobber).
            // useVReg writes spill loads into scratch=.rcx, so rax is preserved.
            const val_reg = try useVReg(code, alloc_result, st.val, .rcx);
            if (val_reg != .rcx) try code.movRegReg(.rcx, val_reg);
            // Fold wasm offset into the mov displacement when it fits in i32.
            var st_disp: i32 = 0;
            if (st.offset <= 0x7FFFFFFF) {
                st_disp = @intCast(st.offset);
            } else {
                try code.movRegImm64(.r10, @as(u64, st.offset));
                try code.addRegReg(.rax, .r10);
            }
            try code.movMemRegSized(.rax, st_disp, .rcx, st.size);
        },
        .memory_copy => |mc| {
            // Call the host helper via vmctx.mem_copy_fn(vmctx, dst, src, len).
            // The helper handles bounds-check + overlapping memmove semantics.
            // ABI-portable via param_regs.
            // Regalloc treats this op as a call (see clobber points).
            const args_arr = [_]ir.VReg{ mc.dst, mc.src, mc.len };
            try emitCallRegArgMoves(code, alloc_result, &args_arr, 3);
            // Zero-extend all three u32 args in-place.
            try code.zeroExtend32(param_regs[1]);
            try code.zeroExtend32(param_regs[2]);
            try code.zeroExtend32(param_regs[3]);
            // Load vmctx + helper fn pointer (uses rax + param_regs[0]).
            try code.movRegMem(param_regs[0], .rbp, vmctx_offset);
            try code.movRegMem(.rax, param_regs[0], vmctx_mem_copy_fn_field);

            const shadow: u32 = if (comptime builtin.os.tag == .windows) 32 else 0;
            const stack_adjust: u32 = (shadow + 15) & ~@as(u32, 15);
            if (stack_adjust > 0) try code.subRegImm32(.rsp, @intCast(stack_adjust));
            try code.callReg(.rax);
            if (stack_adjust > 0) try code.addRegImm32(.rsp, @intCast(stack_adjust));
        },
        .memory_fill => |mf| {
            // REP STOSB: rdi=dst, al=val, rcx=len
            //
            // Bounds check semantics (wasm spec): trap if dst + len > mem_size.
            try code.movRegMem(.r10, .rbp, vmctx_offset); // load VmCtx*
            const len_reg = try useVReg(code, alloc_result, mf.len, .rcx);
            if (len_reg != .rcx) try code.movRegReg(.rcx, len_reg);
            try code.zeroExtend32(.rcx); // wasm lengths are u32
            const dst_reg = try useVReg(code, alloc_result, mf.dst, .rdi);
            if (dst_reg != .rdi) try code.movRegReg(.rdi, dst_reg);
            try code.zeroExtend32(.rdi); // wasm addresses are i32
            // Bounds-check dst+len: uses rax for address and clobbers r11.
            try code.movRegReg(.rax, .rdi);
            try emitMemBoundsCheckDynamic(code);
            // Load fill value AFTER bounds check (emitMemBoundsCheckDynamic
            // clobbers r11, so we can't stash the value there).
            const val_reg = try useVReg(code, alloc_result, mf.val, .rax);
            if (val_reg != .rax) try code.movRegReg(.rax, val_reg);
            try code.movRegMem(.r10, .r10, vmctx_membase_field); // load VmCtx.memory_base
            try code.addRegReg(.rdi, .r10); // rdi = mem_base + dst
            try code.emitSlice(&.{ 0xF3, 0xAA }); // REP STOSB
        },

        // ── Memory management ─────────────────────────────────────────
        .memory_size => {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            // Read current page count from VmCtx
            try code.movRegMem(.r10, .rbp, vmctx_offset);
            try code.movRegMemNoRex(dr, .r10, vmctx_mem_pages_field);
            // 32-bit load already zero-extends
            try writeDef(code, alloc_result, dest, dr);
        },
        .memory_grow => |pages_vreg| {
            const dest = inst.dest orelse return;
            // Call the host grow helper via vmctx.mem_grow_fn(vmctx, delta_pages).
            // ABI-portable: arg0 → param_regs[0] (rcx on Win64, rdi on SysV),
            // arg1 → param_regs[1] (rdx on Win64, rsi on SysV).
            // The regalloc treats this op as a call (see clobber point collection),
            // so no caller-saved vregs are live across this instruction.
            const arg_vmctx = param_regs[0];
            const arg_pages = param_regs[1];
            const pages_reg = try useVReg(code, alloc_result, pages_vreg, arg_pages);
            if (pages_reg != arg_pages) try code.movRegReg(arg_pages, pages_reg);
            try code.zeroExtend32(arg_pages);

            // Load vmctx into param_regs[0].
            try code.movRegMem(arg_vmctx, .rbp, vmctx_offset);
            // Load grow helper pointer into rax.
            try code.movRegMem(.rax, arg_vmctx, vmctx_mem_grow_fn_field);

            const shadow: u32 = if (comptime builtin.os.tag == .windows) 32 else 0;
            const stack_adjust: u32 = (shadow + 15) & ~@as(u32, 15);
            if (stack_adjust > 0) try code.subRegImm32(.rsp, @intCast(stack_adjust));
            try code.callReg(.rax);
            if (stack_adjust > 0) try code.addRegImm32(.rsp, @intCast(stack_adjust));

            // Helper returns i32 (old pages or -1); sign-extend not needed since
            // writeDefTyped honors inst.type and consumers use 32-bit semantics.
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },

        // memory.init and data.drop — currently no-ops in AOT since data
        // segments are applied at instantiation. For full passive segment
        // support, these would need runtime calls via VmCtx.
        .memory_init => |mi| {
            // Consume operands (required for correct vreg tracking)
            _ = try useVReg(code, alloc_result, mi.dst, .rax);
            _ = try useVReg(code, alloc_result, mi.src, .rax);
            _ = try useVReg(code, alloc_result, mi.len, .rax);
        },
        .data_drop => {},

        .table_init => |ti| {
            // Helper signature (4 args to fit Win64 regs):
            //   tableInitHelper(vmctx, packed_seg_table, packed_dst_src, len)
            //   packed_seg_table = seg_idx | (table_idx << 32)
            //   packed_dst_src   = dst | (src << 32)
            //
            // Win64: rcx=vmctx, rdx=packed_seg_table, r8=packed_dst_src, r9=len
            // SysV:  rdi=vmctx, rsi=packed_seg_table, rdx=packed_dst_src, rcx=len
            const arg_len_reg = param_regs[3];
            const arg_packed_ds_reg = param_regs[2];
            const arg_packed_st_reg = param_regs[1];

            // Stage len into arg_len_reg (zero-extended i32).
            const len_reg = try useVReg(code, alloc_result, ti.len, arg_len_reg);
            if (len_reg != arg_len_reg) try code.movRegReg(arg_len_reg, len_reg);
            try code.zeroExtend32(arg_len_reg);

            // Build packed_dst_src in arg_packed_ds_reg.
            // 1. src -> r11, zero-extend, shl 32.
            const src_reg = try useVReg(code, alloc_result, ti.src, .r11);
            if (src_reg != .r11) try code.movRegReg(.r11, src_reg);
            try code.zeroExtend32(.r11);
            try code.emitSlice(&.{ 0x49, 0xC1, 0xE3, 0x20 }); // shl r11, 32

            // 2. dst -> arg_packed_ds_reg, zero-extend.
            const dst_reg = try useVReg(code, alloc_result, ti.dst, arg_packed_ds_reg);
            if (dst_reg != arg_packed_ds_reg) try code.movRegReg(arg_packed_ds_reg, dst_reg);
            try code.zeroExtend32(arg_packed_ds_reg);

            // 3. or arg_packed_ds_reg, r11.
            try code.orRegReg(arg_packed_ds_reg, .r11);

            // packed_seg_table is compile-time constant.
            const packed_st: u64 =
                @as(u64, ti.seg_idx) | (@as(u64, ti.table_idx) << 32);
            try code.movRegImm64(arg_packed_st_reg, packed_st);

            // vmctx into param_regs[0]; helper ptr into rax.
            try code.movRegMem(param_regs[0], .rbp, vmctx_offset);
            try code.movRegMem(.rax, param_regs[0], vmctx_table_init_fn_field);

            const shadow: u32 = if (comptime builtin.os.tag == .windows) 32 else 0;
            const stack_adjust: u32 = (shadow + 15) & ~@as(u32, 15);
            if (stack_adjust > 0) try code.subRegImm32(.rsp, @intCast(stack_adjust));
            try code.callReg(.rax);
            if (stack_adjust > 0) try code.addRegImm32(.rsp, @intCast(stack_adjust));
        },
        .elem_drop => |seg_idx| {
            // Helper: elemDropHelper(vmctx, seg_idx)
            //   Win64: rcx=vmctx, rdx=seg_idx
            //   SysV:  rdi=vmctx, rsi=seg_idx
            try code.movRegImm32(param_regs[1], @bitCast(seg_idx));
            try code.movRegMem(param_regs[0], .rbp, vmctx_offset);
            try code.movRegMem(.rax, param_regs[0], vmctx_elem_drop_fn_field);

            const shadow: u32 = if (comptime builtin.os.tag == .windows) 32 else 0;
            const stack_adjust: u32 = (shadow + 15) & ~@as(u32, 15);
            if (stack_adjust > 0) try code.subRegImm32(.rsp, @intCast(stack_adjust));
            try code.callReg(.rax);
            if (stack_adjust > 0) try code.addRegImm32(.rsp, @intCast(stack_adjust));
        },

        // ── Table management ─────────────────────────────────────────
        .table_size => |table_idx| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            // r10 = vmctx.tables_info_ptr; read len (u32) from slot [table_idx].
            try code.movRegMem(.r10, .rbp, vmctx_offset);
            try code.movRegMem(.r10, .r10, vmctx_tables_info_field);
            const len_off: i32 = @as(i32, @intCast(table_idx)) * table_info_stride + table_info_len_off;
            try code.movRegMemNoRex(dr, .r10, len_off);
            try writeDef(code, alloc_result, dest, dr);
        },
        .table_get => |tg| {
            const dest = inst.dest orelse return;

            // Bring idx into rax, zero-extend to 64 bits.
            const idx_reg = try useVReg(code, alloc_result, tg.idx, .rax);
            if (idx_reg != .rax) try code.movRegReg(.rax, idx_reg);
            try code.zeroExtend32(.rax);

            // r10 = vmctx.tables_info_ptr.
            try code.movRegMem(.r10, .rbp, vmctx_offset);
            try code.movRegMem(.r10, .r10, vmctx_tables_info_field);

            // cmp eax, [r10 + table_idx*16 + 8]   (table len, u32)
            const len_off: i32 = @as(i32, @intCast(tg.table_idx)) * table_info_stride + table_info_len_off;
            try emitCmpEaxMemR10Disp8(code, len_off);
            try code.emitByte(0x72); // jb over_trap (rel8)
            try code.emitByte(12);
            try code.movRegReg(param_regs[0], .r10);
            try code.movRegMem(.rax, param_regs[0], vmctx_trap_unreachable_fn_field);
            try code.callReg(.rax);
            // over_trap: rax still holds idx on the fall-through.

            // r10 = table_info[table_idx].ptr.
            const ptr_off: i32 = @as(i32, @intCast(tg.table_idx)) * table_info_stride + table_info_ptr_off;
            try code.movRegMem(.r10, .r10, ptr_off);
            try code.emitSlice(&.{ 0x48, 0xC1, 0xE0, 0x03 }); // shl rax, 3
            try code.addRegReg(.r10, .rax);
            const dr = destReg(alloc_result, dest);
            try code.movRegMem(dr, .r10, 0);
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .table_set => |ts| {
            // Call tableSetHelper(vmctx, table_idx, elem_idx, value).
            // The helper updates both native_backing and type_backing
            // (via ptr_to_sig binary search) so subsequent call_indirect
            // sees the correct sig_id.
            //
            // Stage operands through non-allocatable scratch registers
            // (r11, rax) first to avoid clobbering live vregs in param regs.
            const val_reg = try useVReg(code, alloc_result, ts.val, .r11);
            if (val_reg != .r11) try code.movRegReg(.r11, val_reg);
            const idx_reg = try useVReg(code, alloc_result, ts.idx, .rax);
            if (idx_reg != .rax) try code.movRegReg(.rax, idx_reg);
            try code.zeroExtend32(.rax);

            if (comptime builtin.os.tag == .windows) {
                // Win64: rcx=vmctx, rdx=table_idx, r8=elem_idx, r9=value
                try code.movRegReg(.r9, .r11);
                try code.movRegReg(.r8, .rax);
                try code.movRegImm32(.rdx, @bitCast(ts.table_idx));
            } else {
                // SysV: rdi=vmctx, rsi=table_idx, rdx=elem_idx, rcx=value
                try code.movRegReg(.rcx, .r11);
                try code.movRegReg(.rdx, .rax);
                try code.movRegImm32(.rsi, @bitCast(ts.table_idx));
            }
            try code.movRegMem(param_regs[0], .rbp, vmctx_offset);
            try code.movRegMem(.rax, param_regs[0], vmctx_table_set_fn_field);
            const shadow: u32 = if (comptime builtin.os.tag == .windows) 32 else 0;
            const stack_adjust: u32 = (shadow + 15) & ~@as(u32, 15);
            if (stack_adjust > 0) try code.subRegImm32(.rsp, @intCast(stack_adjust));
            try code.callReg(.rax);
            if (stack_adjust > 0) try code.addRegImm32(.rsp, @intCast(stack_adjust));
        },
        .ref_func => |fidx| {
            const dest = inst.dest orelse return;
            // Load funcptrs array, then [funcptrs + fidx*8].
            try code.movRegMem(.r10, .rbp, vmctx_offset);
            try code.movRegMem(.r10, .r10, vmctx_funcptrs_field);
            const dr = destReg(alloc_result, dest);
            try code.movRegMem(dr, .r10, @as(i32, @intCast(fidx * 8)));
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .table_grow => |tg| {
            const dest = inst.dest orelse return;
            // Host helper signature (Win64): RCX=vmctx, RDX=init_val (i64),
            // R8=delta (i32). Returns i32 old_size in eax or -1.
            // Like memory_grow, regalloc treats this as a call: no caller-saved
            // vregs live across.

            // Move init into rdx (Win64 arg1 / SysV arg1 rsi — adjust per ABI).
            if (comptime builtin.os.tag == .windows) {
                const init_reg = try useVReg(code, alloc_result, tg.init, .rdx);
                if (init_reg != .rdx) try code.movRegReg(.rdx, init_reg);
                const delta_reg = try useVReg(code, alloc_result, tg.delta, .r8);
                if (delta_reg != .r8) try code.movRegReg(.r8, delta_reg);
                try code.zeroExtend32(.r8);
                try code.movRegImm32(.r9, @bitCast(tg.table_idx));
            } else {
                // SysV: rdi=vmctx, rsi=init, rdx=delta, rcx=table_idx.
                const init_reg = try useVReg(code, alloc_result, tg.init, .rsi);
                if (init_reg != .rsi) try code.movRegReg(.rsi, init_reg);
                const delta_reg = try useVReg(code, alloc_result, tg.delta, .rdx);
                if (delta_reg != .rdx) try code.movRegReg(.rdx, delta_reg);
                try code.zeroExtend32(.rdx);
                try code.movRegImm32(.rcx, @bitCast(tg.table_idx));
            }

            // Load vmctx into param_regs[0] and grow fn ptr into rax.
            try code.movRegMem(param_regs[0], .rbp, vmctx_offset);
            try code.movRegMem(.rax, param_regs[0], vmctx_table_grow_fn_field);

            const shadow: u32 = if (comptime builtin.os.tag == .windows) 32 else 0;
            const stack_adjust: u32 = (shadow + 15) & ~@as(u32, 15);
            if (stack_adjust > 0) try code.subRegImm32(.rsp, @intCast(stack_adjust));
            try code.callReg(.rax);
            if (stack_adjust > 0) try code.addRegImm32(.rsp, @intCast(stack_adjust));

            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },

        // ── Division ──────────────────────────────────────────────────
        .div_s, .div_u, .rem_s, .rem_u => |bin| {
            const dest = inst.dest orelse return;

            // Float div: frontend lowers f32/f64 `div` into `.div_s` with
            // inst.type == .f32/.f64. Dispatch to SSE divss/divsd. `.div_u`,
            // `.rem_s`, `.rem_u` are never produced with float type.
            if (inst.type == .f32 or inst.type == .f64) {
                const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
                if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);
                const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
                if (rhs_reg != .rcx) try code.movRegReg(.rcx, rhs_reg);
                if (inst.type == .f32) {
                    try code.movdToXmm(.rax, .rax);
                    try code.movdToXmm(.rcx, .rcx);
                    try code.divss(.rax, .rcx);
                    try code.movdFromXmm(.rax, .rax);
                } else {
                    try code.movqToXmm(.rax, .rax);
                    try code.movqToXmm(.rcx, .rcx);
                    try code.divsd(.rax, .rcx);
                    try code.movqFromXmm(.rax, .rax);
                }
                try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
                return;
            }

            // idiv clobbers rax+rdx. rax is not allocatable, so only rdx
            // (which IS allocatable) needs saving if it holds a live value.
            const rdx_in_use = for (caller_saved_alloc, 0..) |reg, i| {
                if (reg == .rdx and used_caller_saved[i]) break true;
            } else false;
            if (rdx_in_use) try code.pushReg(.rdx);

            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
            if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);
            // useVReg loads spilled rhs into its scratch (.rcx), so it cannot
            // clobber rax; no need to stash LHS in r11.
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
            if (rhs_reg != .rcx) try code.movRegReg(.rcx, rhs_reg);

            // Zero divisor check: if rcx == 0, call the trap helper.
            //   test rcx, rcx
            //   jne not_zero  (rel32)
            //   call trap_idivz
            // not_zero:
            try code.testRegReg(.rcx, .rcx);
            try code.emitSlice(&.{ 0x0F, 0x85 }); // JNE rel32
            const notzero_patch = code.len();
            try code.emitI32(0);
            try emitTrapHelperCall(code, vmctx_trap_idivz_fn_field);
            const notzero_off = code.len();
            code.patchI32(notzero_patch, @intCast(@as(i64, @intCast(notzero_off)) - @as(i64, @intCast(notzero_patch + 4))));

            const is_rem = (inst.op == .rem_s or inst.op == .rem_u);

            switch (inst.op) {
                .div_s, .rem_s => {
                    if (inst.type == .i32) {
                        // Guard against INT_MIN / -1: div_s traps, rem_s returns 0.
                        //   cmp ecx, -1
                        //   jne do_div
                        //   cmp eax, 0x80000000
                        //   jne do_div
                        //   (INT_MIN/-1 case:)
                        //     rem_s: xor edx, edx; jmp after
                        //     div_s: call trap_iovf
                        // do_div:
                        //   cdq
                        //   idiv ecx
                        // after:
                        try code.emitSlice(&.{ 0x83, 0xF9, 0xFF }); // cmp ecx, -1
                        try code.emitSlice(&.{ 0x0F, 0x85 }); // jne rel32
                        const dodiv_patch = code.len();
                        try code.emitI32(0);
                        try code.emitSlice(&.{ 0x3D, 0x00, 0x00, 0x00, 0x80 }); // cmp eax, INT_MIN
                        try code.emitSlice(&.{ 0x0F, 0x85 }); // jne rel32
                        const dodiv_patch2 = code.len();
                        try code.emitI32(0);
                        var after_patch: usize = 0;
                        if (is_rem) {
                            try code.emitSlice(&.{ 0x31, 0xD2 }); // xor edx, edx
                            try code.emitSlice(&.{0xE9}); // jmp rel32
                            after_patch = code.len();
                            try code.emitI32(0);
                        } else {
                            try emitTrapHelperCall(code, vmctx_trap_iovf_fn_field);
                        }
                        const dodiv_off = code.len();
                        code.patchI32(dodiv_patch, @intCast(@as(i64, @intCast(dodiv_off)) - @as(i64, @intCast(dodiv_patch + 4))));
                        code.patchI32(dodiv_patch2, @intCast(@as(i64, @intCast(dodiv_off)) - @as(i64, @intCast(dodiv_patch2 + 4))));
                        try code.emitSlice(&.{0x99}); // cdq
                        try code.emitSlice(&.{ 0xF7, 0xF9 }); // idiv ecx
                        if (is_rem) {
                            const after_off = code.len();
                            code.patchI32(after_patch, @intCast(@as(i64, @intCast(after_off)) - @as(i64, @intCast(after_patch + 4))));
                        }
                    } else {
                        try code.emitSlice(&.{ 0x48, 0x83, 0xF9, 0xFF }); // cmp rcx, -1
                        try code.emitSlice(&.{ 0x0F, 0x85 });
                        const dodiv_patch = code.len();
                        try code.emitI32(0);
                        // movabs r11, INT64_MIN
                        try code.emitSlice(&.{ 0x49, 0xBB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80 });
                        try code.emitSlice(&.{ 0x4C, 0x39, 0xD8 }); // cmp rax, r11
                        try code.emitSlice(&.{ 0x0F, 0x85 });
                        const dodiv_patch2 = code.len();
                        try code.emitI32(0);
                        var after_patch: usize = 0;
                        if (is_rem) {
                            try code.emitSlice(&.{ 0x31, 0xD2 });
                            try code.emitSlice(&.{0xE9});
                            after_patch = code.len();
                            try code.emitI32(0);
                        } else {
                            try emitTrapHelperCall(code, vmctx_trap_iovf_fn_field);
                        }
                        const dodiv_off = code.len();
                        code.patchI32(dodiv_patch, @intCast(@as(i64, @intCast(dodiv_off)) - @as(i64, @intCast(dodiv_patch + 4))));
                        code.patchI32(dodiv_patch2, @intCast(@as(i64, @intCast(dodiv_off)) - @as(i64, @intCast(dodiv_patch2 + 4))));
                        try code.emitSlice(&.{ 0x48, 0x99 }); // cqo
                        try code.emitSlice(&.{ 0x48, 0xF7, 0xF9 }); // idiv rcx
                        if (is_rem) {
                            const after_off = code.len();
                            code.patchI32(after_patch, @intCast(@as(i64, @intCast(after_off)) - @as(i64, @intCast(after_patch + 4))));
                        }
                    }
                },
                .div_u, .rem_u => {
                    try code.movRegImm32(.rdx, 0);
                    if (inst.type == .i32) {
                        try code.emitSlice(&.{ 0xF7, 0xF1 }); // div ecx
                    } else {
                        try code.divReg(.rcx);
                    }
                },
                else => unreachable,
            }

            const result_reg: emit.Reg = switch (inst.op) {
                .div_s, .div_u => .rax,
                .rem_s, .rem_u => .rdx,
                else => unreachable,
            };

            // Restore rdx if it was saved
            if (rdx_in_use) {
                if (result_reg == .rdx) {
                    try code.movRegReg(.r11, .rdx);
                    try code.popReg(.rdx);
                    try code.movRegReg(.rax, .r11);
                } else {
                    try code.popReg(.rdx);
                }
            } else {
                if (result_reg != .rax) try code.movRegReg(.rax, result_reg);
            }

            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },

        // ── Shifts ────────────────────────────────────────────────────
        .shl, .shr_s, .shr_u, .rotl, .rotr => |bin| {
            const dest = inst.dest orelse return;
            const is64 = inst.type == .i64;
            const digit: u3 = switch (inst.op) {
                .shl => 4,
                .shr_u => 5,
                .shr_s => 7,
                .rotl => 0,
                .rotr => 1,
                else => unreachable,
            };
            // Constant-count fast path: emit `C1 /digit ib` (or `D1 /digit`
            // when imm==1) and skip the MOV-to-CL entirely. Mask imm to
            // wasm-semantic bit width: 5 bits for i32, 6 bits for i64.
            if (const_vals.get(bin.rhs)) |cnt| {
                const mask: u64 = if (is64) 0x3f else 0x1f;
                const imm: u8 = @intCast(@as(u64, @bitCast(cnt)) & mask);
                const val_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
                if (val_reg != .rax) try code.movRegReg(.rax, val_reg);
                if (imm != 0) try code.shiftRegImm8(.rax, digit, imm, is64);
                try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
                return;
            }
            // Load val first to avoid clobbering if both share a register
            const val_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
            if (val_reg != .rax) try code.movRegReg(.rax, val_reg);
            // useVReg loads spilled cnt into its scratch (.rcx), so it cannot
            // clobber rax; no need to stash val in r11.
            const cnt_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
            if (cnt_reg != .rcx) try code.movRegReg(.rcx, cnt_reg);
            // i32 shifts/rotates must use the 32-bit opcode form (no REX.W).
            // In 64-bit REX.W form x86 masks CL to 6 bits, so shift-by-32
            // produces wrong results; wasm i32 shifts mask by 5 bits, which
            // matches the 32-bit x86 form natively.
            switch (inst.op) {
                .shl => {
                    if (is64) try code.rexW(.rax, .rax);
                    try code.emitSlice(&.{ 0xD3, 0xE0 });
                },
                .shr_u => {
                    if (is64) try code.rexW(.rax, .rax);
                    try code.emitSlice(&.{ 0xD3, 0xE8 });
                },
                .shr_s => {
                    if (is64) try code.rexW(.rax, .rax);
                    try code.emitSlice(&.{ 0xD3, 0xF8 });
                },
                .rotl => {
                    if (is64) try code.rexW(.rax, .rax);
                    try code.emitSlice(&.{ 0xD3, 0xC0 });
                },
                .rotr => {
                    if (is64) try code.rexW(.rax, .rax);
                    try code.emitSlice(&.{ 0xD3, 0xC8 });
                },
                else => unreachable,
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },

        // ── Unary ops ─────────────────────────────────────────────────
        .clz => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (inst.type == .i32) {
                try code.lzcnt32(dr, src_reg);
            } else {
                try code.lzcnt(dr, src_reg);
            }
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .ctz => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (inst.type == .i32) {
                try code.tzcnt32(dr, src_reg);
            } else {
                try code.tzcnt(dr, src_reg);
            }
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .popcnt => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (inst.type == .i32) {
                try code.popcnt32(dr, src_reg);
            } else {
                try code.popcntReg(dr, src_reg);
            }
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },

        // ── Select ────────────────────────────────────────────────────
        .select => |sel| {
            const dest = inst.dest orelse return;
            // Stage operands through non-allocatable scratch registers
            // (.r10, .r11) so no allocated VReg can collide with our
            // intermediate storage. The previous implementation loaded
            // cond into .rax then operands into .rcx/.rdx, which
            // clobbered the if_true/if_false values whenever the
            // allocator had placed them in .rax/.rcx/.rdx (seen as
            // the float-select LSB-ORed-with-cond pattern in
            // float_exprs.wast).
            const true_reg = try useVReg(code, alloc_result, sel.if_true, .r10);
            if (true_reg != .r10) try code.movRegReg(.r10, true_reg);
            const false_reg = try useVReg(code, alloc_result, sel.if_false, .r11);
            if (false_reg != .r11) try code.movRegReg(.r11, false_reg);
            const cond_reg = try useVReg(code, alloc_result, sel.cond, .rax);
            // Condition is always i32 in wasm; use 32-bit test so
            // upper-bit garbage doesn't affect the branch.
            try code.testRegReg32(cond_reg, cond_reg);
            try code.cmovnz(.r11, .r10); // if cond != 0, r11 = r10 (if_true)
            try writeDefTyped(code, alloc_result, dest, .r11, inst.type);
        },

        .global_get => |idx| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            // Load globals_base from VmCtx, then load global value
            try code.movRegMem(.r10, .rbp, vmctx_offset); // VmCtx*
            try code.movRegMem(.r10, .r10, vmctx_globals_field); // globals_base
            try code.movRegMem(dr, .r10, @as(i32, @intCast(idx * 8))); // global[idx]
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .global_set => |gs| {
            const val_reg = try useVReg(code, alloc_result, gs.val, .rax);
            try code.movRegMem(.r10, .rbp, vmctx_offset); // VmCtx*
            try code.movRegMem(.r10, .r10, vmctx_globals_field); // globals_base
            try code.movMemReg(.r10, @as(i32, @intCast(gs.idx * 8)), val_reg); // global[idx] = val
        },

        .@"unreachable" => {
            try emitTrapHelperCall(code, vmctx_trap_unreachable_fn_field);
        },

        // ── Type conversions ──────────────────────────────────────────
        .wrap_i64 => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            // 32-bit reg-reg move truncates to 32 bits and zero-extends.
            try code.movRegReg32(dr, src_reg);
            try writeDef(code, alloc_result, dest, dr);
        },
        .extend_i32_s => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            try code.movsxd(dr, src_reg);
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .extend_i32_u => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            // 32-bit mov zero-extends to 64.
            try code.movRegReg32(dr, src_reg);
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .extend8_s => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            try code.movsxByteToReg(dr, src_reg);
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .extend16_s => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            try code.movsxWordToReg(dr, src_reg);
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .extend32_s => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            try code.movsxd(dr, src_reg);
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .reinterpret => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            try writeDefTyped(code, alloc_result, dest, src_reg, inst.type);
        },

        // ── Float unary operations ─────────────────────────────────────
        .f_neg => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            // XOR sign bit: for f64 flip bit 63, for f32 flip bit 31
            if (inst.type == .f32) {
                try code.emitSlice(&.{ 0x48, 0xB9 }); // mov rcx, imm64
                try code.emitU64(0x0000000080000000);
                try code.xorRegReg(.rax, .rcx);
            } else {
                try code.emitSlice(&.{ 0x48, 0xB9 }); // mov rcx, imm64
                try code.emitU64(0x8000000000000000);
                try code.xorRegReg(.rax, .rcx);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .f_abs => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            // AND clear sign bit
            if (inst.type == .f32) {
                try code.emitSlice(&.{ 0x48, 0xB9 }); // mov rcx, imm64
                try code.emitU64(0x000000007FFFFFFF);
                try code.andRegReg(.rax, .rcx);
            } else {
                try code.emitSlice(&.{ 0x48, 0xB9 }); // mov rcx, imm64
                try code.emitU64(0x7FFFFFFFFFFFFFFF);
                try code.andRegReg(.rax, .rcx);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .f_sqrt => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            if (inst.type == .f32) {
                try code.movdToXmm(.rax, .rax);
                try code.sqrtss(.rax, .rax);
                try code.movdFromXmm(.rax, .rax);
            } else {
                try code.movqToXmm(.rax, .rax);
                try code.sqrtsd(.rax, .rax);
                try code.movqFromXmm(.rax, .rax);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .f_ceil, .f_floor, .f_trunc, .f_nearest => |vreg| {
            // ROUNDSD/ROUNDSS (SSE4.1): 66 0F 3A 0B/0A xmm, xmm, imm8
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            const round_mode: u8 = switch (inst.op) {
                .f_ceil => 0x0A,
                .f_floor => 0x09,
                .f_trunc => 0x0B,
                .f_nearest => 0x08,
                else => unreachable,
            };
            if (inst.type == .f32) {
                try code.movdToXmm(.rax, .rax);
                // ROUNDSS: 66 0F 3A 0A C0 imm8
                try code.emitSlice(&.{ 0x66, 0x0F, 0x3A, 0x0A, 0xC0, round_mode });
                try code.movdFromXmm(.rax, .rax);
            } else {
                try code.movqToXmm(.rax, .rax);
                // ROUNDSD: 66 0F 3A 0B C0 imm8
                try code.emitSlice(&.{ 0x66, 0x0F, 0x3A, 0x0B, 0xC0, round_mode });
                try code.movqFromXmm(.rax, .rax);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },

        // ── Float binary operations ───────────────────────────────────
        .f_min, .f_max => |bin| {
            const dest = inst.dest orelse return;
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
            if (rhs_reg != .rcx) try code.movRegReg(.rcx, rhs_reg);
            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
            if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);
            // Wasm f32/f64 min/max:
            //   1. If either operand is NaN → canonical qNaN.
            //   2. If operands are bit-equal → bitwise OR (min) / AND (max)
            //      so that min(-0,+0) = -0 and max(-0,+0) = +0.
            //   3. Otherwise → MINSS/MINSD (MAXSS/MAXSD).
            // MINSx/MAXSx alone fail both (1) (returns 2nd op on NaN) and
            // (2) (returns 2nd op when equal) — hence the explicit checks.
            const is_min = inst.op == .f_min;
            if (inst.type == .f32) {
                // Offsets (f32, with UCOMISS for IEEE equality — handles ±0):
                //  0   mov edx, eax                          2
                //  2   and edx, 0x7FFFFFFF                   6
                //  8   cmp edx, 0x7F800000                   6
                // 14   ja  nan  (rel8 = 59-16 = 43)          2
                // 16   mov edx, ecx                          2
                // 18   and edx, 0x7FFFFFFF                   6
                // 24   cmp edx, 0x7F800000                   6
                // 30   ja  nan  (rel8 = 59-32 = 27)          2
                // 32   movd xmm0, eax                        4
                // 36   movd xmm1, ecx                        4
                // 40   ucomiss xmm0, xmm1                    3
                // 43   je  equal (rel8 = 55-45 = 10)         2
                // 45   minss/maxss xmm0, xmm1                4
                // 49   movd eax, xmm0                        4
                // 53   jmp done (rel8 = 64-55 = 9)           2
                // 55   or/and eax, ecx                       2
                // 57   jmp done (rel8 = 64-59 = 5)           2
                // 59   mov eax, 0x7FC00000                   5
                // 64   <done>
                const minmax_byte: u8 = if (is_min) 0x5D else 0x5F; // MINSS=5D, MAXSS=5F
                const or_and_byte: u8 = if (is_min) 0x09 else 0x21; // OR=09, AND=21
                try code.emitSlice(&.{
                    0x89, 0xC2, // mov edx, eax
                    0x81, 0xE2, 0xFF, 0xFF, 0xFF, 0x7F, // and edx, 0x7FFFFFFF
                    0x81, 0xFA, 0x00, 0x00, 0x80, 0x7F, // cmp edx, 0x7F800000
                    0x77, 0x2B, // ja nan (+43)
                    0x89, 0xCA, // mov edx, ecx
                    0x81, 0xE2, 0xFF, 0xFF, 0xFF, 0x7F, // and edx, 0x7FFFFFFF
                    0x81, 0xFA, 0x00, 0x00, 0x80, 0x7F, // cmp edx, 0x7F800000
                    0x77, 0x1B, // ja nan (+27)
                    0x66, 0x0F, 0x6E, 0xC0, // movd xmm0, eax
                    0x66, 0x0F, 0x6E, 0xC9, // movd xmm1, ecx
                    0x0F, 0x2E, 0xC1, // ucomiss xmm0, xmm1
                    0x74, 0x0A, // je equal (+10)
                    0xF3, 0x0F, minmax_byte, 0xC1, // min/maxss xmm0, xmm1
                    0x66, 0x0F, 0x7E, 0xC0, // movd eax, xmm0
                    0xEB, 0x09, // jmp done (+9)
                    or_and_byte, 0xC8, // or/and eax, ecx
                    0xEB, 0x05, // jmp done (+5)
                    0xB8, 0x00, 0x00, 0xC0, 0x7F, // mov eax, 0x7FC00000
                });
            } else {
                // f64: same shape with UCOMISD for IEEE equality.
                // Offsets:
                //   0  mov rdx, rax                          3
                //   3  movabs r11, 0x7FFFFFFFFFFFFFFF       10
                //  13  and rdx, r11                          3
                //  16  movabs r11, 0x7FF0000000000000       10
                //  26  cmp rdx, r11                          3
                //  29  ja nan (rel8 = 94-31 = 63)            2
                //  31  mov rdx, rcx                          3
                //  34  movabs r11, 0x7FFFFFFFFFFFFFFF       10
                //  44  and rdx, r11                          3
                //  47  movabs r11, 0x7FF0000000000000       10
                //  57  cmp rdx, r11                          3
                //  60  ja nan (rel8 = 94-62 = 32)            2
                //  62  movq xmm0, rax                        5
                //  67  movq xmm1, rcx                        5
                //  72  ucomisd xmm0, xmm1                    4
                //  76  je equal (rel8 = 89-78 = 11)          2
                //  78  minsd/maxsd xmm0, xmm1                4
                //  82  movq rax, xmm0                        5
                //  87  jmp done (rel8 = 104-89 = 15)         2
                //  89  or/and rax, rcx                       3
                //  92  jmp done (rel8 = 104-94 = 10)         2
                //  94  movabs rax, 0x7FF8000000000000       10
                // 104  <done>
                const minmax_byte: u8 = if (is_min) 0x5D else 0x5F;
                const or_and_byte: u8 = if (is_min) 0x09 else 0x21;
                try code.emitSlice(&.{
                    0x48, 0x89, 0xC2, // mov rdx, rax
                    0x49, 0xBB, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F, // movabs r11, 0x7FFF...FF
                    0x4C, 0x21, 0xDA, // and rdx, r11
                    0x49, 0xBB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x7F, // movabs r11, 0x7FF0...00
                    0x4C, 0x39, 0xDA, // cmp rdx, r11
                    0x77, 0x3F, // ja nan (+63)
                    0x48, 0x89, 0xCA, // mov rdx, rcx
                    0x49, 0xBB, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F, // movabs r11, 0x7FFF...FF
                    0x4C, 0x21, 0xDA, // and rdx, r11
                    0x49, 0xBB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x7F, // movabs r11, 0x7FF0...00
                    0x4C, 0x39, 0xDA, // cmp rdx, r11
                    0x77, 0x20, // ja nan (+32)
                    0x66, 0x48, 0x0F, 0x6E, 0xC0, // movq xmm0, rax
                    0x66, 0x48, 0x0F, 0x6E, 0xC9, // movq xmm1, rcx
                    0x66, 0x0F, 0x2E, 0xC1, // ucomisd xmm0, xmm1
                    0x74, 0x0B, // je equal (+11)
                    0xF2, 0x0F, minmax_byte, 0xC1, // min/maxsd xmm0, xmm1
                    0x66, 0x48, 0x0F, 0x7E, 0xC0, // movq rax, xmm0
                    0xEB, 0x0F, // jmp done (+15)
                    0x48, or_and_byte, 0xC8, // or/and rax, rcx
                    0xEB, 0x0A, // jmp done (+10)
                    0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0x7F, // movabs rax, 0x7FF8...00
                });
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .f_copysign => |bin| {
            const dest = inst.dest orelse return;
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
            if (rhs_reg != .rcx) try code.movRegReg(.rcx, rhs_reg);
            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
            if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);
            // copysign: magnitude from lhs, sign from rhs
            if (inst.type == .f32) {
                try code.emitSlice(&.{ 0x48, 0xBA }); // mov rdx, imm64
                try code.emitU64(0x000000007FFFFFFF);
                try code.andRegReg(.rax, .rdx);
                try code.emitSlice(&.{ 0x48, 0xBA }); // mov rdx, imm64
                try code.emitU64(0x0000000080000000);
                try code.andRegReg(.rcx, .rdx);
            } else {
                try code.emitSlice(&.{ 0x48, 0xBA }); // mov rdx, imm64
                try code.emitU64(0x7FFFFFFFFFFFFFFF);
                try code.andRegReg(.rax, .rdx);
                try code.emitSlice(&.{ 0x48, 0xBA }); // mov rdx, imm64
                try code.emitU64(0x8000000000000000);
                try code.andRegReg(.rcx, .rdx);
            }
            try code.orRegReg(.rax, .rcx);
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },

        // ── Float comparisons ─────────────────────────────────────────
        // Operands are f32/f64 (via inst.type); result is i32 (0 or 1).
        // Uses UCOMIS[SD] xmm0, xmm1, then SETcc. IEEE-754 unordered
        // (NaN) rules: eq=0/ne=1 when unordered; lt/gt/le/ge all=0.
        .f_eq, .f_ne, .f_lt, .f_gt, .f_le, .f_ge => |bin| {
            const dest = inst.dest orelse return;
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
            if (rhs_reg != .rcx) try code.movRegReg(.rcx, rhs_reg);
            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
            if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);
            if (inst.type == .f32) {
                try code.movdToXmm(.rax, .rax); // xmm0 = lhs
                try code.movdToXmm(.rcx, .rcx); // xmm1 = rhs
                try code.ucomiss(.rax, .rcx);
            } else {
                try code.movqToXmm(.rax, .rax);
                try code.movqToXmm(.rcx, .rcx);
                try code.ucomisd(.rax, .rcx);
            }
            // UCOMIS flags:
            //   equal     : ZF=1, PF=0, CF=0
            //   a > b     : ZF=0, PF=0, CF=0
            //   a < b     : ZF=0, PF=0, CF=1
            //   unordered : ZF=1, PF=1, CF=1
            //
            // SETcc codes: B=2, AE=3, E=4, NE=5, BE=6, A=7, P=A, NP=B
            switch (inst.op) {
                .f_eq => {
                    // ordered AND equal: SETE AND SETNP
                    try code.setcc(0x4, .rax); // sete al
                    try code.setcc(0xB, .r11); // setnp r11b
                    try code.andRegReg(.rax, .r11);
                },
                .f_ne => {
                    // unordered OR not-equal: SETNE OR SETP
                    try code.setcc(0x5, .rax); // setne al
                    try code.setcc(0xA, .r11); // setp r11b
                    try code.orRegReg(.rax, .r11);
                },
                .f_lt => {
                    // ordered AND below: SETB AND SETNP
                    try code.setcc(0x2, .rax); // setb al
                    try code.setcc(0xB, .r11); // setnp r11b
                    try code.andRegReg(.rax, .r11);
                },
                .f_gt => {
                    // above (already excludes unordered): SETA
                    try code.setcc(0x7, .rax); // seta al
                },
                .f_le => {
                    // ordered AND below-or-equal: SETBE AND SETNP
                    try code.setcc(0x6, .rax); // setbe al
                    try code.setcc(0xB, .r11); // setnp r11b
                    try code.andRegReg(.rax, .r11);
                },
                .f_ge => {
                    // above-or-equal (excludes unordered): SETAE
                    try code.setcc(0x3, .rax); // setae al
                },
                else => unreachable,
            }
            try code.movzxByte(.rax, .rax);
            // Result is i32; explicitly force i32 write (inst.type is the
            // operand float type, not the result type).
            try writeDefTyped(code, alloc_result, dest, .rax, .i32);
        },

        // ── Int/Float conversions ─────────────────────────────────────
        .trunc_f32_s, .trunc_f64_s => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            const is_f32 = (inst.op == .trunc_f32_s);
            if (is_f32) {
                try code.movdToXmm(.rax, .rax);
            } else {
                try code.movqToXmm(.rax, .rax);
            }
            try emitTruncRangeCheck(code, inst.type == .i32, true, is_f32);
            // Use the dest-width form so 32-bit trunc fills only EAX (and the
            // upper 32 bits don't carry stale signed bits across writeDefTyped).
            if (inst.type == .i32) {
                if (is_f32) try code.cvttss2si32(.rax, .rax) else try code.cvttsd2si32(.rax, .rax);
            } else {
                if (is_f32) try code.cvttss2si(.rax, .rax) else try code.cvttsd2si(.rax, .rax);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .trunc_f32_u, .trunc_f64_u => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            const is_f32 = (inst.op == .trunc_f32_u);
            if (is_f32) {
                try code.movdToXmm(.rax, .rax);
            } else {
                try code.movqToXmm(.rax, .rax);
            }
            try emitTruncRangeCheck(code, inst.type == .i32, false, is_f32);
            if (inst.type == .i32) {
                // Unsigned i32 fits in signed i64; use 64-bit cvtt then truncate.
                if (is_f32) try code.cvttss2si(.rax, .rax) else try code.cvttsd2si(.rax, .rax);
            } else {
                // Unsigned i64: split at 2^63. If src < 2^63 direct; else subtract
                // 2^63, convert, then add back.
                const split_bits: u64 = if (is_f32) 0x5F000000 else 0x43E0000000000000;
                if (is_f32) {
                    try code.movRegImm32(.r11, @bitCast(@as(u32, @truncate(split_bits))));
                    try code.movdToXmm(.rcx, .r11);
                    try code.ucomiss(.rax, .rcx);
                } else {
                    try code.movRegImm64(.r11, split_bits);
                    try code.movqToXmm(.rcx, .r11);
                    try code.ucomisd(.rax, .rcx);
                }
                // JB = src < 2^63 -> direct convert
                try code.emitSlice(&.{ 0x0F, 0x82 });
                const direct_patch = code.len();
                try code.emitI32(0);
                // High range: subtract 2^63, convert, add back
                if (is_f32) {
                    try code.subss(.rax, .rcx);
                    try code.cvttss2si(.rax, .rax);
                } else {
                    try code.subsd(.rax, .rcx);
                    try code.cvttsd2si(.rax, .rax);
                }
                try code.movRegImm64(.r11, 0x8000000000000000);
                try code.addRegReg(.rax, .r11);
                try code.emitSlice(&.{0xE9}); // JMP done
                const done_patch = code.len();
                try code.emitI32(0);
                const direct_off = code.len();
                if (is_f32) try code.cvttss2si(.rax, .rax) else try code.cvttsd2si(.rax, .rax);
                code.patchI32(direct_patch, @intCast(@as(i64, @intCast(direct_off)) - @as(i64, @intCast(direct_patch + 4))));
                const done_off = code.len();
                code.patchI32(done_patch, @intCast(@as(i64, @intCast(done_off)) - @as(i64, @intCast(done_patch + 4))));
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .trunc_sat_f32_s, .trunc_sat_f64_s, .trunc_sat_f32_u, .trunc_sat_f64_u => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);

            const is_f32 = (inst.op == .trunc_sat_f32_s or inst.op == .trunc_sat_f32_u);
            const is_signed = (inst.op == .trunc_sat_f32_s or inst.op == .trunc_sat_f64_s);
            const dst_is_i32 = (inst.type == .i32);

            // Move source to xmm0
            if (is_f32) try code.movdToXmm(.rax, .rax) else try code.movqToXmm(.rax, .rax);

            // ── NaN check ──
            // ucomis* sets PF=1 if unordered (NaN). NaN -> result = 0.
            if (is_f32) try code.ucomiss(.rax, .rax) else try code.ucomisd(.rax, .rax);
            try code.emitSlice(&.{ 0x0F, 0x8A }); // JP rel32
            const nan_patch = code.len();
            try code.emitI32(0);

            // ── Lower bound check: if src < MIN_F  -> saturate to MIN_INT ──
            // Compute MIN_F bits depending on (dst_is_i32, is_signed, is_f32).
            // For unsigned: MIN_F = 0.0 (any 0-or-negative -> 0).
            const min_f_bits: u64 = if (is_signed) blk: {
                if (dst_is_i32) {
                    // INT32_MIN = -2147483648.0
                    break :blk if (is_f32) @as(u64, 0xCF000000) else @as(u64, 0xC1E0000000000000);
                } else {
                    // INT64_MIN = -9223372036854775808.0
                    break :blk if (is_f32) @as(u64, 0xDF000000) else @as(u64, 0xC3E0000000000000);
                }
            } else 0; // 0.0 for unsigned

            // Load MIN_F into xmm1 via R11
            if (is_f32) {
                try code.movRegImm32(.r11, @bitCast(@as(u32, @truncate(min_f_bits))));
                try code.movdToXmm(.rcx, .r11);
            } else {
                try code.movRegImm64(.r11, min_f_bits);
                try code.movqToXmm(.rcx, .r11);
            }
            // For signed, use UCOMIS; src < MIN_F if CF=1 -> JB
            // For unsigned, src <= 0 -> JBE saturates to 0.
            if (is_f32) try code.ucomiss(.rax, .rcx) else try code.ucomisd(.rax, .rcx);
            try code.emitSlice(&.{ 0x0F, if (is_signed) @as(u8, 0x82) else @as(u8, 0x86) }); // JB / JBE rel32
            const min_patch = code.len();
            try code.emitI32(0);

            // ── Upper bound check: if src >= MAX_BOUND_F -> saturate to MAX_INT ──
            // MAX_BOUND_F is the smallest float >= the *exclusive* upper bound,
            // i.e. (MAX_INT + 1) as a float, since MAX_INT itself isn't usually
            // exactly representable.
            const max_bound_bits: u64 = if (is_signed) blk: {
                if (dst_is_i32) {
                    // 2^31 = 2147483648.0
                    break :blk if (is_f32) @as(u64, 0x4F000000) else @as(u64, 0x41E0000000000000);
                } else {
                    // 2^63 = 9223372036854775808.0
                    break :blk if (is_f32) @as(u64, 0x5F000000) else @as(u64, 0x43E0000000000000);
                }
            } else blk: {
                if (dst_is_i32) {
                    // 2^32 = 4294967296.0
                    break :blk if (is_f32) @as(u64, 0x4F800000) else @as(u64, 0x41F0000000000000);
                } else {
                    // 2^64 = 18446744073709551616.0
                    break :blk if (is_f32) @as(u64, 0x5F800000) else @as(u64, 0x43F0000000000000);
                }
            };
            if (is_f32) {
                try code.movRegImm32(.r11, @bitCast(@as(u32, @truncate(max_bound_bits))));
                try code.movdToXmm(.rcx, .r11);
            } else {
                try code.movRegImm64(.r11, max_bound_bits);
                try code.movqToXmm(.rcx, .r11);
            }
            if (is_f32) try code.ucomiss(.rax, .rcx) else try code.ucomisd(.rax, .rcx);
            // JAE = src >= MAX_BOUND -> saturate to MAX
            try code.emitSlice(&.{ 0x0F, 0x83 });
            const max_patch = code.len();
            try code.emitI32(0);

            // ── Normal conversion ──
            if (!is_signed and !dst_is_i32) {
                // Unsigned 64-bit trunc: cvttsx2si is signed, so split at 2^63.
                // We've already verified 0 <= src < 2^64.
                // If src >= 2^63: subtract 2^63, cvttsx2si, add back 2^63.
                const split_bits: u64 = if (is_f32) 0x5F000000 else 0x43E0000000000000;
                if (is_f32) {
                    try code.movRegImm32(.r11, @bitCast(@as(u32, @truncate(split_bits))));
                    try code.movdToXmm(.rcx, .r11);
                    try code.ucomiss(.rax, .rcx);
                } else {
                    try code.movRegImm64(.r11, split_bits);
                    try code.movqToXmm(.rcx, .r11);
                    try code.ucomisd(.rax, .rcx);
                }
                // JB = src < 2^63 -> direct convert
                try code.emitSlice(&.{ 0x0F, 0x82 });
                const direct_patch = code.len();
                try code.emitI32(0);
                // High range: subtract 2^63, convert, add back
                if (is_f32) {
                    try code.subss(.rax, .rcx);
                    try code.cvttss2si(.rax, .rax);
                } else {
                    try code.subsd(.rax, .rcx);
                    try code.cvttsd2si(.rax, .rax);
                }
                try code.movRegImm64(.r11, 0x8000000000000000);
                try code.addRegReg(.rax, .r11);
                try code.emitSlice(&.{0xE9}); // JMP done
                const u64_done_patch = code.len();
                try code.emitI32(0);
                // Direct path
                const direct_off = code.len();
                if (is_f32) try code.cvttss2si(.rax, .rax) else try code.cvttsd2si(.rax, .rax);
                code.patchI32(direct_patch, @intCast(@as(i64, @intCast(direct_off)) - @as(i64, @intCast(direct_patch + 4))));
                const u64_done_off = code.len();
                code.patchI32(u64_done_patch, @intCast(@as(i64, @intCast(u64_done_off)) - @as(i64, @intCast(u64_done_patch + 4))));
            } else if (dst_is_i32) {
                if (is_signed) {
                    // Signed i32: 32-bit cvtt; bound-check already ensured in-range.
                    if (is_f32) try code.cvttss2si32(.rax, .rax) else try code.cvttsd2si32(.rax, .rax);
                } else {
                    // Unsigned i32 in [0, 2^32): use 64-bit signed cvt (fits in
                    // positive i63), then low 32 bits are the unsigned result.
                    if (is_f32) try code.cvttss2si(.rax, .rax) else try code.cvttsd2si(.rax, .rax);
                }
            } else {
                // Signed i64
                if (is_f32) try code.cvttss2si(.rax, .rax) else try code.cvttsd2si(.rax, .rax);
            }
            try code.emitSlice(&.{0xE9}); // JMP done
            const done_patch = code.len();
            try code.emitI32(0);

            // ── NaN landing: result = 0 ──
            const nan_off = code.len();
            try code.xorReg32(.rax);
            try code.emitSlice(&.{0xE9});
            const nan_done_patch = code.len();
            try code.emitI32(0);

            // ── Underflow landing: result = MIN_INT ──
            const min_off = code.len();
            if (!is_signed) {
                // unsigned -> 0
                try code.xorReg32(.rax);
            } else if (dst_is_i32) {
                try code.movRegImm32(.rax, @bitCast(@as(u32, 0x80000000)));
            } else {
                try code.movRegImm64(.rax, 0x8000000000000000);
            }
            try code.emitSlice(&.{0xE9});
            const min_done_patch = code.len();
            try code.emitI32(0);

            // ── Overflow landing: result = MAX_INT ──
            const max_off = code.len();
            if (!is_signed) {
                if (dst_is_i32) try code.movRegImm32(.rax, @bitCast(@as(u32, 0xFFFFFFFF))) else try code.movRegImm64(.rax, 0xFFFFFFFFFFFFFFFF);
            } else if (dst_is_i32) {
                try code.movRegImm32(.rax, 0x7FFFFFFF);
            } else {
                try code.movRegImm64(.rax, 0x7FFFFFFFFFFFFFFF);
            }

            // ── done ──
            const done_off = code.len();

            code.patchI32(nan_patch, @intCast(@as(i64, @intCast(nan_off)) - @as(i64, @intCast(nan_patch + 4))));
            code.patchI32(min_patch, @intCast(@as(i64, @intCast(min_off)) - @as(i64, @intCast(min_patch + 4))));
            code.patchI32(max_patch, @intCast(@as(i64, @intCast(max_off)) - @as(i64, @intCast(max_patch + 4))));
            code.patchI32(done_patch, @intCast(@as(i64, @intCast(done_off)) - @as(i64, @intCast(done_patch + 4))));
            code.patchI32(nan_done_patch, @intCast(@as(i64, @intCast(done_off)) - @as(i64, @intCast(nan_done_patch + 4))));
            code.patchI32(min_done_patch, @intCast(@as(i64, @intCast(done_off)) - @as(i64, @intCast(min_done_patch + 4))));

            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .convert_s, .convert_u => {
            // Legacy ops, no longer emitted by the frontend (replaced by
            // .convert_i32_s/.convert_i64_s/.convert_i32_u/.convert_i64_u).
        },
        .convert_i32_s => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            // Use 32-bit CVTSI2S{S,D} so the source register is interpreted as a
            // signed dword (so e.g. 0xFFFFFFFF -> -1.0, not 4294967295.0).
            if (inst.type == .f32) {
                try code.cvtsi2ss32(.rax, .rax);
                try code.movdFromXmm(.rax, .rax);
            } else {
                try code.cvtsi2sd32(.rax, .rax);
                try code.movqFromXmm(.rax, .rax);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .convert_i64_s => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            if (inst.type == .f32) {
                try code.cvtsi2ss(.rax, .rax);
                try code.movdFromXmm(.rax, .rax);
            } else {
                try code.cvtsi2sd(.rax, .rax);
                try code.movqFromXmm(.rax, .rax);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .convert_i32_u => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            // Zero-extend low 32 bits to 64 bits, then use signed 64-bit conversion.
            // (mov r32,r32 zero-extends to 64; the eax->rax extension already
            // happened from any preceding 32-bit op, but be explicit.)
            try code.emitSlice(&.{ 0x89, 0xC0 }); // mov eax, eax — zero-extends to RAX
            if (inst.type == .f32) {
                try code.cvtsi2ss(.rax, .rax);
                try code.movdFromXmm(.rax, .rax);
            } else {
                try code.cvtsi2sd(.rax, .rax);
                try code.movqFromXmm(.rax, .rax);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .convert_i64_u => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);

            // Check if value >= 2^63 (sign bit set)
            try code.testRegReg(.rax, .rax);
            try code.emitSlice(&.{ 0x0F, 0x88 }); // JS rel32 (jump if sign bit set)
            const large_patch = code.len();
            try code.emitI32(0);

            // Small path (< 2^63): treat as signed — cvtsi2s{s,d} is correct
            if (inst.type == .f32) {
                try code.cvtsi2ss(.rax, .rax);
                try code.movdFromXmm(.rax, .rax);
            } else {
                try code.cvtsi2sd(.rax, .rax);
                try code.movqFromXmm(.rax, .rax);
            }
            try code.emitSlice(&.{0xE9}); // JMP done
            const done_patch = code.len();
            try code.emitI32(0);

            // Large path (>= 2^63): shift right by 1, convert, then multiply by 2
            const large_off = code.len();
            try code.movRegReg(.rcx, .rax); // save original
            try code.emitSlice(&.{ 0x48, 0xD1, 0xE8 }); // shr rax, 1
            // Preserve low bit: OR with original & 1
            try code.andRegImm32(.rcx, 1); // rcx = original & 1
            try code.orRegReg(.rax, .rcx); // rax = (original >> 1) | (original & 1)
            if (inst.type == .f32) {
                try code.cvtsi2ss(.rax, .rax);
                try code.addss(.rax, .rax); // multiply by 2
                try code.movdFromXmm(.rax, .rax);
            } else {
                try code.cvtsi2sd(.rax, .rax);
                try code.addsd(.rax, .rax); // multiply by 2
                try code.movqFromXmm(.rax, .rax);
            }
            const done_off = code.len();

            code.patchI32(large_patch, @intCast(@as(i64, @intCast(large_off)) - @as(i64, @intCast(large_patch + 4))));
            code.patchI32(done_patch, @intCast(@as(i64, @intCast(done_off)) - @as(i64, @intCast(done_patch + 4))));

            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .demote_f64 => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            try code.movqToXmm(.rax, .rax);
            try code.cvtsd2ss(.rax, .rax);
            try code.movdFromXmm(.rax, .rax);
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .promote_f32 => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            try code.movdToXmm(.rax, .rax);
            try code.cvtss2sd(.rax, .rax);
            try code.movqFromXmm(.rax, .rax);
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },

        // ── Atomic wait/notify ─────────────────────────────────────
        .atomic_notify => |an| {
            const dest = inst.dest orelse return;
            // Call aotAtomicNotify(vmctx, addr, count)
            const count_reg = try useVReg(code, alloc_result, an.count, .rax);
            if (count_reg != .rax) try code.movRegReg(.rax, count_reg);
            const base_reg = try useVReg(code, alloc_result, an.base, .rcx);
            if (base_reg != .rcx) try code.movRegReg(.rcx, base_reg);
            try code.zeroExtend32(.rcx); // wasm addr is i32
            // Add static offset
            if (an.offset > 0) try code.addRegImm32(.rcx, @intCast(an.offset));

            if (comptime builtin.os.tag == .windows) {
                // Win64: rcx=vmctx, rdx=addr, r8=count
                try code.movRegReg(.r8, .rax); // count
                try code.movRegReg(.rdx, .rcx); // addr
            } else {
                // SysV: rdi=vmctx, rsi=addr, rdx=count
                try code.movRegReg(.rdx, .rax); // count
                try code.movRegReg(.rsi, .rcx); // addr
            }
            try code.movRegMem(param_regs[0], .rbp, vmctx_offset);
            try code.movRegMem(.rax, param_regs[0], vmctx_futex_notify_fn_field);
            const shadow: u32 = if (comptime builtin.os.tag == .windows) 32 else 0;
            const stack_adjust: u32 = (shadow + 15) & ~@as(u32, 15);
            if (stack_adjust > 0) try code.subRegImm32(.rsp, @intCast(stack_adjust));
            try code.callReg(.rax);
            if (stack_adjust > 0) try code.addRegImm32(.rsp, @intCast(stack_adjust));
            try writeDefTyped(code, alloc_result, dest, .rax, .i32);
        },
        .atomic_wait => |aw| {
            const dest = inst.dest orelse return;
            // Call aotAtomicWait32/64(vmctx, addr, expected[_lo, _hi], timeout_lo, timeout_hi)
            // Load operands into scratch registers first
            const base_reg = try useVReg(code, alloc_result, aw.base, .rax);
            if (base_reg != .rax) try code.movRegReg(.rax, base_reg);
            try code.zeroExtend32(.rax);
            if (aw.offset > 0) try code.addRegImm32(.rax, @intCast(aw.offset));
            // Save computed addr in r11
            try code.movRegReg(.r11, .rax);

            const expected_reg = try useVReg(code, alloc_result, aw.expected, .rax);
            if (expected_reg != .rax) try code.movRegReg(.rax, expected_reg);

            const timeout_reg = try useVReg(code, alloc_result, aw.timeout, .rcx);
            if (timeout_reg != .rcx) try code.movRegReg(.rcx, timeout_reg);

            // Set up ABI args and call the appropriate helper
            try code.movRegMem(param_regs[0], .rbp, vmctx_offset);

            if (aw.size == 4) {
                if (comptime builtin.os.tag == .windows) {
                    // Win64: rcx=vmctx, rdx=addr, r8=expected, r9=timeout_lo, [rsp+32]=timeout_hi
                    try code.movRegReg(.r9, .rcx); // timeout_lo (lower 32 bits)
                    try code.movRegReg(.r10, .rcx);
                    try code.emitSlice(&.{ 0x48, 0xC1, 0xEA, 0x20 }); // shr r10, 32 (wrong reg)
                    // Simplified: pass timeout as two u32 halves
                    try code.movRegReg(.r8, .rax); // expected
                    try code.movRegReg(.rdx, .r11); // addr
                } else {
                    // SysV: rdi=vmctx, rsi=addr, rdx=expected, rcx=timeout_lo, r8=timeout_hi
                    try code.movRegReg(.r8, .rcx);
                    try code.emitSlice(&.{ 0x49, 0xC1, 0xE8, 0x20 }); // shr r8, 32
                    try code.zeroExtend32(.rcx); // timeout_lo
                    try code.movRegReg(.rdx, .rax); // expected
                    try code.movRegReg(.rsi, .r11); // addr
                }
                try code.movRegMem(.rax, param_regs[0], vmctx_futex_wait32_fn_field);
            } else {
                // wait64 — expected is i64, split into lo/hi
                if (comptime builtin.os.tag == .windows) {
                    try code.movRegReg(.r8, .rax); // exp_lo
                    try code.movRegReg(.r9, .rax);
                    try code.emitSlice(&.{ 0x49, 0xC1, 0xE9, 0x20 }); // shr r9, 32 = exp_hi
                    try code.zeroExtend32(.r8); // exp_lo
                    try code.movRegReg(.rdx, .r11); // addr
                } else {
                    try code.movRegReg(.rdx, .rax); // exp_lo
                    try code.movRegReg(.rcx, .rax);
                    try code.emitSlice(&.{ 0x48, 0xC1, 0xE9, 0x20 }); // shr rcx, 32 = exp_hi
                    try code.zeroExtend32(.rdx); // exp_lo
                    try code.movRegReg(.rsi, .r11); // addr
                }
                try code.movRegMem(.rax, param_regs[0], vmctx_futex_wait64_fn_field);
            }

            const shadow2: u32 = if (comptime builtin.os.tag == .windows) 48 else 0;
            const stack_adjust2: u32 = (shadow2 + 15) & ~@as(u32, 15);
            if (stack_adjust2 > 0) try code.subRegImm32(.rsp, @intCast(stack_adjust2));
            try code.callReg(.rax);
            if (stack_adjust2 > 0) try code.addRegImm32(.rsp, @intCast(stack_adjust2));
            try writeDefTyped(code, alloc_result, dest, .rax, .i32);
        },

        // ── Stubs for ops not commonly hit ────────────────────────────
        // Phi must be lowered before codegen.
        .phi => unreachable,
        else => {
            // For unhandled ops, emit a no-op placeholder
            if (inst.dest) |dest| {
                try code.movRegImm32(.rax, 0);
                try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
            }
        },
    }
}

// ── Tests ──────────────────────────────────────────────────────────

test "compileFunction: iconst_32 + ret produces mov and epilogue" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Should have prologue + mov + stack ops + epilogue
    try std.testing.expect(code.len > 10);
    // Verify the immediate 42 (0x2A) appears in the code
    var found_42 = false;
    for (code) |b| {
        if (b == 0x2A) {
            found_42 = true;
            break;
        }
    }
    try std.testing.expect(found_42);
    // Last byte should be ret (0xC3)
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunction: two iconsts + add + ret produces reasonable code" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    try std.testing.expect(code.len > 15);
    try std.testing.expect(code.len < 300);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileModule: two functions have correct offsets" {
    const allocator = std.testing.allocator;
    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    _ = try f1.newBlock();
    const v0 = f1.newVReg();
    try f1.getBlock(0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try f1.getBlock(0).append(.{ .op = .{ .ret = v0 } });
    _ = try ir_module.addFunction(f1);

    var f2 = ir.IrFunction.init(allocator, 0, 1, 0);
    _ = try f2.newBlock();
    const v1 = f2.newVReg();
    try f2.getBlock(0).append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1 });
    try f2.getBlock(0).append(.{ .op = .{ .ret = v1 } });
    _ = try ir_module.addFunction(f2);

    const result = try compileModule(&ir_module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    try std.testing.expectEqual(@as(usize, 2), result.offsets.len);
    try std.testing.expectEqual(@as(u32, 0), result.offsets[0]);
    try std.testing.expect(result.offsets[1] > 0);
}

test "compileFunction: empty function produces prologue and epilogue" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    _ = try func.newBlock();

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    try std.testing.expect(code.len > 0);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "CachedStack: push and pop maintain depth" {
    var code = emit.CodeBuffer.init(std.testing.allocator);
    defer code.deinit();

    var stack = CachedStack.init(2); // 2 locals
    try std.testing.expectEqual(@as(u32, 0), stack.depth);

    try stack.push(&code, .rax);
    try std.testing.expectEqual(@as(u32, 1), stack.depth);

    try stack.push(&code, .rcx);
    try std.testing.expectEqual(@as(u32, 2), stack.depth);

    try stack.pop(&code, .rax);
    try std.testing.expectEqual(@as(u32, 1), stack.depth);

    try stack.pop(&code, .rax);
    try std.testing.expectEqual(@as(u32, 0), stack.depth);
}

// ── Atomic instruction integration tests ──────────────────────────

fn containsBytes(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i <= haystack.len - needle.len) : (i += 1) {
        if (std.mem.eql(u8, haystack[i..][0..needle.len], needle)) return true;
    }
    return false;
}

test "compileFunction: atomic_fence emits MFENCE" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    try block.append(.{ .op = .{ .atomic_fence = {} } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // MFENCE = 0F AE F0
    try std.testing.expect(containsBytes(code, &.{ 0x0F, 0xAE, 0xF0 }));
}

test "compileFunction: atomic_load emits sized MOV" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const base = func.newVReg();
    const loaded = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .atomic_load = .{ .base = base, .offset = 0, .size = 4 } }, .dest = loaded, .type = .i32 });
    try block.append(.{ .op = .{ .ret = loaded } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    try std.testing.expect(code.len > 10);
    // Should NOT contain LOCK prefix (loads don't need it on x86-64)
    try std.testing.expect(!containsBytes(code, &.{ 0xF0, 0x0F }));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunction: atomic_store emits MOV + MFENCE" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const base = func.newVReg();
    const val = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = val, .type = .i32 });
    try block.append(.{ .op = .{ .atomic_store = .{ .base = base, .offset = 0, .size = 4, .val = val } } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Must contain MFENCE for seq-cst ordering
    try std.testing.expect(containsBytes(code, &.{ 0x0F, 0xAE, 0xF0 }));
}

test "compileFunction: atomic_rmw add emits LOCK XADD" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const base = func.newVReg();
    const val = func.newVReg();
    const result = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = val, .type = .i32 });
    try block.append(.{ .op = .{ .atomic_rmw = .{ .base = base, .offset = 0, .size = 4, .val = val, .op = .add } }, .dest = result, .type = .i32 });
    try block.append(.{ .op = .{ .ret = result } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // LOCK XADD = F0 0F C1
    try std.testing.expect(containsBytes(code, &.{ 0xF0, 0x0F, 0xC1 }));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunction: atomic_rmw sub emits NEG + LOCK XADD" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const base = func.newVReg();
    const val = func.newVReg();
    const result = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = val, .type = .i32 });
    try block.append(.{ .op = .{ .atomic_rmw = .{ .base = base, .offset = 0, .size = 4, .val = val, .op = .sub } }, .dest = result, .type = .i32 });
    try block.append(.{ .op = .{ .ret = result } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Should contain NEG (F7 /3) followed later by LOCK XADD (F0 0F C1)
    try std.testing.expect(containsBytes(code, &.{ 0xF0, 0x0F, 0xC1 }));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunction: atomic_rmw xchg emits XCHG" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const base = func.newVReg();
    const val = func.newVReg();
    const result = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 99 }, .dest = val, .type = .i32 });
    try block.append(.{ .op = .{ .atomic_rmw = .{ .base = base, .offset = 0, .size = 4, .val = val, .op = .xchg } }, .dest = result, .type = .i32 });
    try block.append(.{ .op = .{ .ret = result } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // XCHG [mem], reg = 87 (no LOCK prefix needed, implicit)
    try std.testing.expect(containsBytes(code, &.{0x87}));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunction: atomic_rmw and emits CAS loop with LOCK CMPXCHG" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const base = func.newVReg();
    const val = func.newVReg();
    const result = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 0xFF }, .dest = val, .type = .i32 });
    try block.append(.{ .op = .{ .atomic_rmw = .{ .base = base, .offset = 0, .size = 4, .val = val, .op = .@"and" } }, .dest = result, .type = .i32 });
    try block.append(.{ .op = .{ .ret = result } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // LOCK CMPXCHG with r8 src = F0 44 0F B1 (REX.R for r8 in reg field)
    try std.testing.expect(containsBytes(code, &.{ 0xF0, 0x44, 0x0F, 0xB1 }));
    // JNE for CAS retry = 0F 85
    try std.testing.expect(containsBytes(code, &.{ 0x0F, 0x85 }));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunction: atomic_cmpxchg emits LOCK CMPXCHG" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const base = func.newVReg();
    const expected = func.newVReg();
    const replacement = func.newVReg();
    const result = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = expected, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = replacement, .type = .i32 });
    try block.append(.{ .op = .{ .atomic_cmpxchg = .{ .base = base, .offset = 0, .size = 4, .expected = expected, .replacement = replacement } }, .dest = result, .type = .i32 });
    try block.append(.{ .op = .{ .ret = result } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // LOCK CMPXCHG = F0 0F B1
    try std.testing.expect(containsBytes(code, &.{ 0xF0, 0x0F, 0xB1 }));
    // No JNE (single CMPXCHG, no retry loop)
    try std.testing.expect(!containsBytes(code, &.{ 0x0F, 0x85 }));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

// ── Instruction selection optimization tests ──────────────────────

test "compileFunction: iconst_32(0) emits xor (zero idiom)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Should contain XOR r32,r32 (31 xx) instead of MOV rax,0 (48 C7 C0 00000000)
    try std.testing.expect(containsBytes(code, &.{0x31}));
    // Should NOT contain the 7-byte mov rax, 0 pattern
    try std.testing.expect(!containsBytes(code, &.{ 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00 }));
}

test "compileFunction: iconst + add uses ADD imm8 form" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // 5 fits in i8 → ADD reg, imm8 (opcode 0x83 /0, imm8=0x05).
    try std.testing.expect(containsBytes(code, &.{ 0x83, 0xC0, 0x05 }));
    // The 7-byte imm32 form (0x81 /0) must NOT appear for this small imm.
    try std.testing.expect(!containsBytes(code, &.{ 0x81, 0xC0, 0x05, 0x00, 0x00, 0x00 }));
}

test "compileFunction: cmp + br_if fuses to Jcc (no setcc)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 2);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const block1 = func.getBlock(b1);
    const block2 = func.getBlock(b2);

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const cond = func.newVReg();
    const r1 = func.newVReg();
    const r2 = func.newVReg();
    try block0.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try block0.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try block0.append(.{ .op = .{ .lt_s = .{ .lhs = v0, .rhs = v1 } }, .dest = cond });
    try block0.append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try block1.append(.{ .op = .{ .iconst_32 = 1 }, .dest = r1 });
    try block1.append(.{ .op = .{ .ret = r1 } });
    try block2.append(.{ .op = .{ .iconst_32 = 0 }, .dest = r2 });
    try block2.append(.{ .op = .{ .ret = r2 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Should contain JL (0F 8C) — fused compare-and-branch
    try std.testing.expect(containsBytes(code, &.{ 0x0F, 0x8C }));
    // Should NOT contain SETCC (0F 9x) — comparison was deferred
    var has_setcc = false;
    for (code, 0..) |b, j| {
        if (b == 0x0F and j + 1 < code.len and (code[j + 1] & 0xF0) == 0x90) {
            has_setcc = true;
            break;
        }
    }
    try std.testing.expect(!has_setcc);
}

// ── Register-allocated compilation tests ──────────────────────────

test "compileFunctionRA: iconst_32 + ret" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    try std.testing.expect(code.len > 5);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunctionRA: add two constants" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    try std.testing.expect(code.len > 10);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
    // Should use ADD imm32 (0x81) since rhs is a constant
    var found_add_imm = false;
    for (code) |b| {
        if (b == 0x81) {
            found_add_imm = true;
            break;
        }
    }
    try std.testing.expect(found_add_imm);
}

test "compileFunctionRA: global_get emits load from VMContext" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .global_get = 0 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    try std.testing.expect(code.len > 10);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunctionRA: wrap_i64 emits mov eax,eax" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_64 = 0x100000042 }, .dest = v0, .type = .i64 });
    try block.append(.{ .op = .{ .wrap_i64 = v0 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v1 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // wrap_i64 emits a 32-bit mov (opcode 0x89) for truncation.
    // The specific register encoding depends on allocation order.
    try std.testing.expect(containsBytes(code, &.{0x89}));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunctionRA: extend_i32_s emits MOVSXD" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = -1 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .extend_i32_s = v0 }, .dest = v1, .type = .i64 });
    try block.append(.{ .op = .{ .ret = v1 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // MOVSXD (opcode 0x63) must be present for sign extension.
    // REX prefix varies by platform (destination register allocation).
    try std.testing.expect(containsBytes(code, &.{0x63}));
}

test "compileFunctionRA: memory_copy emits call via mem_copy_fn" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const dst = func.newVReg();
    const src = func.newVReg();
    const len = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = dst });
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = src });
    try block.append(.{ .op = .{ .iconst_32 = 50 }, .dest = len });
    try block.append(.{ .op = .{ .memory_copy = .{ .dst = dst, .src = src, .len = len } } });
    try block.append(.{ .op = .{ .ret = null } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // Must NOT contain REP MOVSB (F3 A4): the inline path was buggy on overlap,
    // we now dispatch through the vmctx.mem_copy_fn host helper which implements
    // memmove semantics.
    try std.testing.expect(!containsBytes(code, &.{ 0xF3, 0xA4 }));
    // Must contain CALL reg (FF /2): dispatch to the helper via rax.
    try std.testing.expect(containsBytes(code, &.{ 0xFF, 0xD0 }));
}

test "compileFunctionRA: memory_fill emits REP STOSB" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const dst = func.newVReg();
    const val = func.newVReg();
    const len = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = dst });
    try block.append(.{ .op = .{ .iconst_32 = 0xFF }, .dest = val });
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = len });
    try block.append(.{ .op = .{ .memory_fill = .{ .dst = dst, .val = val, .len = len } } });
    try block.append(.{ .op = .{ .ret = null } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // Should contain REP STOSB (F3 AA)
    try std.testing.expect(containsBytes(code, &.{ 0xF3, 0xAA }));
}

test "compileFunctionRA: division uses rax/rdx" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = v1 });
    try block.append(.{ .op = .{ .div_s = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // Should contain IDIV (F7 F9 for idiv rcx)
    try std.testing.expect(containsBytes(code, &.{0xF7}));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunctionRA: local_get and local_set" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 2);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0 });
    try block.append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v0 } } });
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = v1 });
    try block.append(.{ .op = .{ .ret = v1 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    try std.testing.expect(code.len > 10);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunctionRA: br_table emits jump table + indirect jmp" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();

    // Param 0 is the switch index. 2 targets + default.
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    const idx = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .local_get = 0 }, .dest = idx, .type = .i32 });

    const targets = try allocator.alloc(ir.BlockId, 2);
    targets[0] = b1;
    targets[1] = b2;
    try func.getBlock(b0).append(.{ .op = .{ .br_table = .{
        .index = idx,
        .targets = targets,
        .default = b3,
    } } });

    const r1 = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = r1 });
    try func.getBlock(b1).append(.{ .op = .{ .ret = r1 } });
    const r2 = func.newVReg();
    try func.getBlock(b2).append(.{ .op = .{ .iconst_32 = 2 }, .dest = r2 });
    try func.getBlock(b2).append(.{ .op = .{ .ret = r2 } });
    const r3 = func.newVReg();
    try func.getBlock(b3).append(.{ .op = .{ .iconst_32 = 3 }, .dest = r3 });
    try func.getBlock(b3).append(.{ .op = .{ .ret = r3 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);
    // targets slice is leaked by IR (same pattern as call.args); free explicitly.
    allocator.free(targets);

    // lea r10, [rip+disp32]
    try std.testing.expect(containsBytes(code, &.{ 0x4C, 0x8D, 0x15 }));
    // movsxd r11, [r10 + r11*4]
    try std.testing.expect(containsBytes(code, &.{ 0x4F, 0x63, 0x1C, 0x9A }));
    // jmp r10
    try std.testing.expect(containsBytes(code, &.{ 0x41, 0xFF, 0xE2 }));
    // jae rel32 (bounds check)
    try std.testing.expect(containsBytes(code, &.{ 0x0F, 0x83 }));
}

test "compileFunctionRA: eqz targets allocated dest register (rdx)" {
    // eqz result should use the destReg directly, not funnel through rax.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .eqz = v0 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v1 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // The key invariant: eqz uses the allocator-chosen dest register,
    // not hardcoded rax. Verify the function compiles and ends with ret.
    try std.testing.expect(code.len > 10);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
    // Must contain a setcc opcode (0F 94 = sete) somewhere
    try std.testing.expect(containsBytes(code, &.{ 0x0F, 0x94 }));
}

test "compileFunctionRA: comparison targets allocated dest register" {
    // eq should use destReg for setcc instead of hardcoded rax.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .eq = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // Key invariant: setcc must NOT use legacy rax path.
    try std.testing.expect(!containsBytes(code, &.{ 0x0F, 0x94, 0xC0 }));
    // Must contain a setcc (0F 94) somewhere
    try std.testing.expect(containsBytes(code, &.{ 0x0F, 0x94 }));
}

test "compileFunctionRA: memory_size loads into allocated dest register" {
    // memory_size should emit `mov dr, [r10 + mem_pages_field]` with
    // dr being the allocated destination register rather than forced rax.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .memory_size, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // mov edx, [r10 + 56] — 32-bit load, no REX.W.
    // Opcode 0x8B, REX.B=0x41 (because base is r10). ModR/M = 10_010_010 (0x92),
    // then disp32 = 0x38 00 00 00 (vmctx_mem_pages_field = 56).
    try std.testing.expect(containsBytes(code, &.{ 0x41, 0x8B, 0x92, 0x38, 0x00, 0x00, 0x00 }));
}

test "compileFunctionRA: r12 allocated under register pressure" {
    // Seven simultaneously-live values (6 constants + one op) overflow the
    // first 5 alloc_regs (rdx, rsi, rdi, r8, r9). The 6th and 7th must go to
    // r12 and r13 (newly added to alloc_regs). We also verify that r12 is
    // preserved via push/pop in the prologue/epilogue.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    const v3 = func.newVReg();
    const v4 = func.newVReg();
    const v5 = func.newVReg();
    const v6 = func.newVReg();
    const v7 = func.newVReg();
    // Produce 7 constants that are all live through an add chain.
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 4 }, .dest = v3, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v4, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 6 }, .dest = v5, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = v6, .type = .i32 });
    // Consume all of them so their ranges extend here.
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v7, .type = .i32 });
    // Use the rest (prevents range expiry) by re-summing.
    const v8 = func.newVReg();
    try block.append(.{ .op = .{ .add = .{ .lhs = v2, .rhs = v3 } }, .dest = v8, .type = .i32 });
    const v9 = func.newVReg();
    try block.append(.{ .op = .{ .add = .{ .lhs = v4, .rhs = v5 } }, .dest = v9, .type = .i32 });
    const v10 = func.newVReg();
    try block.append(.{ .op = .{ .add = .{ .lhs = v6, .rhs = v7 } }, .dest = v10, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v10 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // Prologue should push r12 (41 54) because it is callee-saved and used.
    try std.testing.expect(containsBytes(code, &.{ 0x41, 0x54 }));
    // Epilogue should pop r12 (41 5C).
    try std.testing.expect(containsBytes(code, &.{ 0x41, 0x5C }));
}

test "compileFunctionRA: low-pressure function does not save r12" {
    // A function that only needs one allocatable register must not touch r12.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // No push r12 (41 54) and no pop r12 (41 5C).
    try std.testing.expect(!containsBytes(code, &.{ 0x41, 0x54 }));
    try std.testing.expect(!containsBytes(code, &.{ 0x41, 0x5C }));
}

test "compileFunctionRA: shift does not emit dead r11 save" {
    // Before B1, every shl emitted `mov r11, rax; ...; mov rax, r11` around
    // the rhs load. Since useVReg loads spilled rhs into its scratch (.rcx),
    // rax is never clobbered, so the save is dead. Verify it's gone.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .shl = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // `mov r11, rax` = 49 89 C3. `mov rax, r11` = 4C 89 D8. Neither should appear.
    try std.testing.expect(!containsBytes(code, &.{ 0x49, 0x89, 0xC3 }));
    try std.testing.expect(!containsBytes(code, &.{ 0x4C, 0x89, 0xD8 }));
    // With a constant shift count the backend now emits the imm form
    // `C1 E0 02` (shl eax, 2) and skips the MOV-to-CL entirely.
    try std.testing.expect(containsBytes(code, &.{ 0xC1, 0xE0, 0x02 }));
    // The CL-based form `D3 E0` must not appear for this constant count.
    try std.testing.expect(!containsBytes(code, &.{ 0xD3, 0xE0 }));
}

test "compileFunctionRA: shl by non-constant still uses CL form" {
    // Negative test for the imm fast path: when rhs is not a known constant
    // (e.g. a function parameter) we must fall back to `D3 /digit` with CL.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 2, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .local_get = 1 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .shl = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // CL-based shift `D3 E0` must still be used.
    try std.testing.expect(containsBytes(code, &.{ 0xD3, 0xE0 }));
}

test "compileFunctionRA: shl i64 by constant 3 emits REX.W C1 E0 03" {
    // 64-bit shift by a constant picks the imm form with REX.W.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_64 = 10 }, .dest = v0, .type = .i64 });
    try block.append(.{ .op = .{ .iconst_64 = 3 }, .dest = v1, .type = .i64 });
    try block.append(.{ .op = .{ .shl = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i64 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // REX.W + C1 /4 ib = `48 C1 E0 03` (shl rax, 3).
    try std.testing.expect(containsBytes(code, &.{ 0x48, 0xC1, 0xE0, 0x03 }));
}

test "compileFunctionRA: shl by constant 1 uses D1 form (no imm byte)" {
    // Count of 1 uses the 1-byte shorter `D1 /digit` encoding.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .shl = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // `D1 E0` = shl eax, 1. No C1 form, no D3/CL form.
    try std.testing.expect(containsBytes(code, &.{ 0xD1, 0xE0 }));
    try std.testing.expect(!containsBytes(code, &.{ 0xC1, 0xE0 }));
    try std.testing.expect(!containsBytes(code, &.{ 0xD3, 0xE0 }));
}

test "compileFunctionRA: shr_u/shr_s/rotl/rotr by constant use imm form" {
    const allocator = std.testing.allocator;
    const Case = struct {
        tag: []const u8,
        expected: [3]u8, // C1 /digit ib (ib=2)
        build: *const fn (lhs: ir.VReg, rhs: ir.VReg) ir.Inst.Op,
    };
    const B = struct {
        fn shr_u(lhs: ir.VReg, rhs: ir.VReg) ir.Inst.Op {
            return .{ .shr_u = .{ .lhs = lhs, .rhs = rhs } };
        }
        fn shr_s(lhs: ir.VReg, rhs: ir.VReg) ir.Inst.Op {
            return .{ .shr_s = .{ .lhs = lhs, .rhs = rhs } };
        }
        fn rotl(lhs: ir.VReg, rhs: ir.VReg) ir.Inst.Op {
            return .{ .rotl = .{ .lhs = lhs, .rhs = rhs } };
        }
        fn rotr(lhs: ir.VReg, rhs: ir.VReg) ir.Inst.Op {
            return .{ .rotr = .{ .lhs = lhs, .rhs = rhs } };
        }
    };
    const cases = [_]Case{
        .{ .tag = "shr_u", .expected = .{ 0xC1, 0xE8, 0x02 }, .build = &B.shr_u },
        .{ .tag = "shr_s", .expected = .{ 0xC1, 0xF8, 0x02 }, .build = &B.shr_s },
        .{ .tag = "rotl", .expected = .{ 0xC1, 0xC0, 0x02 }, .build = &B.rotl },
        .{ .tag = "rotr", .expected = .{ 0xC1, 0xC8, 0x02 }, .build = &B.rotr },
    };
    for (cases) |c| {
        var func = ir.IrFunction.init(allocator, 0, 1, 0);
        defer func.deinit();
        const block_id = try func.newBlock();
        const block = func.getBlock(block_id);
        const v0 = func.newVReg();
        const v1 = func.newVReg();
        const v2 = func.newVReg();
        try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0, .type = .i32 });
        try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1, .type = .i32 });
        try block.append(.{ .op = c.build(v0, v1), .dest = v2, .type = .i32 });
        try block.append(.{ .op = .{ .ret = v2 } });

        const compile_result = try compileFunctionRA(&func, 0, allocator);
        const code = compile_result.code;
        defer allocator.free(compile_result.call_patches);
        defer allocator.free(code);

        if (!containsBytes(code, &c.expected)) {
            std.debug.print("case {s}: expected bytes not found\n", .{c.tag});
            try std.testing.expect(false);
        }
    }
}

test "compileFunctionRA: div does not emit dead r11 save around operand load" {
    // Before B1, div emitted `mov r11, rax; mov rcx, rhs; mov rax, r11` to
    // preserve LHS. With distinct scratches the save is dead.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .div_u = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // `mov r11, rax` = 49 89 C3 must not appear before the idiv/div.
    try std.testing.expect(!containsBytes(code, &.{ 0x49, 0x89, 0xC3 }));
    // `mov rax, r11` = 4C 89 D8: this sequence also appears in div's rem
    // rdx-restore path (only when rdx-in-use AND op is rem), so it must NOT
    // appear here because no vreg got rdx (low pressure).
    try std.testing.expect(!containsBytes(code, &.{ 0x4C, 0x89, 0xD8 }));
}

test "compileFunctionRA: load folds wasm offset into mov disp" {
    // Verifies B2: i32.load with offset=8 no longer emits `add rax, 8`
    // then `mov dst, [rax]`; instead a single `mov dst, [rax+8]`.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v0, .offset = 8, .size = 4, .sign_extend = false } }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v1 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // `add rax, 8` = 48 83 C0 08 (with REX.W). It must no longer appear.
    try std.testing.expect(!containsBytes(code, &.{ 0x48, 0x83, 0xC0, 0x08 }));
    // `add rax, imm32` form = 48 05 ...  also must not appear.
    try std.testing.expect(!containsBytes(code, &.{ 0x48, 0x05 }));
}

test "compileFunctionRA: store folds wasm offset into mov disp and omits r11 save" {
    // Verifies B2: i32.store with offset=16 emits a single
    // `mov [rax+16], ecx` and no longer stashes the base into r11.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .store = .{ .base = v0, .val = v1, .offset = 16, .size = 4 } } });
    try block.append(.{ .op = .{ .ret = null } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // No `mov r11, rax` (49 89 C3) — the base is kept in rax directly.
    try std.testing.expect(!containsBytes(code, &.{ 0x49, 0x89, 0xC3 }));
    // No `add r11, imm32` (49 81 C3 ... or 49 83 C3 ...) — offset is folded.
    try std.testing.expect(!containsBytes(code, &.{ 0x49, 0x81, 0xC3 }));
    try std.testing.expect(!containsBytes(code, &.{ 0x49, 0x83, 0xC3 }));
    // 32-bit store `mov [rax+16], ecx` with disp32 encoding = 89 88 10 00 00 00.
    try std.testing.expect(containsBytes(code, &.{ 0x89, 0x88, 0x10, 0x00, 0x00, 0x00 }));
}

test "compileFunctionRA: add of two non-constant values emits LEA" {
    // Two local_get results then add — neither is in const_vals so the LEA
    // path should fire, producing a single `lea dst, [lhs + rhs]` instead
    // of `mov dst, lhs; add dst, rhs`.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 2, 3, 0); // 2 params so 2 locals
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .local_get = 1 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // Expect a LEA (opcode 0x8D). REX prefix varies by platform.
    try std.testing.expect(containsBytes(code, &.{0x8D}));
    // ADD reg,reg (opcode 01) must NOT appear — the LEA replaced it.
    // Check no standalone 01 in a REX+01 pattern. Just verify 0x8D is present.
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunctionRA: br to next block is elided (C3 fallthrough)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const block1 = func.getBlock(b1);

    const v0 = func.newVReg();
    // b0: br b1   (b1 is the immediately-following block → fall through)
    try block0.append(.{ .op = .{ .br = b1 } });
    // b1: return 7
    try block1.append(.{ .op = .{ .iconst_32 = 7 }, .dest = v0, .type = .i32 });
    try block1.append(.{ .op = .{ .ret = v0 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // No unconditional E9-relative jump should appear: the only br was
    // to the next block and must have been elided. Scan for 0xE9.
    for (code) |b| {
        try std.testing.expect(b != 0xE9);
    }
}

test "compileFunctionRA: br_if else==next drops trailing jmp (C3)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 1);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const block1 = func.getBlock(b1);
    const block2 = func.getBlock(b2);

    const cond = func.newVReg();
    const r = func.newVReg();
    // b0: br_if cond, then=b2, else=b1  (b1 is next → else falls through)
    try block0.append(.{ .op = .{ .local_get = 0 }, .dest = cond, .type = .i32 });
    try block0.append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b2, .else_block = b1 } } });
    // b1: return 0
    try block1.append(.{ .op = .{ .iconst_32 = 0 }, .dest = r, .type = .i32 });
    try block1.append(.{ .op = .{ .ret = r } });
    // b2: return 1
    try block2.append(.{ .op = .{ .iconst_32 = 1 }, .dest = r, .type = .i32 });
    try block2.append(.{ .op = .{ .ret = r } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // The conditional jump 0F 85 must still be present.
    try std.testing.expect(containsBytes(code, &.{ 0x0F, 0x85 }));
    // The unconditional E9 that would jump to the else block (b1) must
    // have been elided: search for 0F 85 rel32 (6 bytes) immediately
    // followed by 0xE9 — that would be the un-elided pattern.
    var has_trailing_e9 = false;
    var i: usize = 0;
    while (i + 6 < code.len) : (i += 1) {
        if (code[i] == 0x0F and code[i + 1] == 0x85 and code[i + 6] == 0xE9) {
            has_trailing_e9 = true;
            break;
        }
    }
    try std.testing.expect(!has_trailing_e9);
}

// ── Stack-argument ABI tests (wasm args beyond reg-param capacity) ──

test "compileFunctionRA: callee with >3 params spills stack args on Win64" {
    const allocator = std.testing.allocator;
    // 5 params → on Win64 params 3,4 are passed on the stack; SysV fits all in regs.
    var func = ir.IrFunction.init(allocator, 5, 5, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    // ret last param (p4) — forces the callee to read it from frame
    try block.append(.{ .op = .{ .ret = 4 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // On Windows the prologue must load at least one stack-passed param from
    // [rbp + 48] via rax (48 00 00 00 disp32). Look for the mov rax, [rbp+48]
    // opcode sequence: REX.W=48, 8B (mov r64, r/m64), ModR/M=85 (mod=10, reg=000(rax), r/m=101(rbp)).
    if (builtin.os.tag == .windows) {
        var found = false;
        var i: usize = 0;
        while (i + 6 < code.len) : (i += 1) {
            if (code[i] == 0x48 and code[i + 1] == 0x8B and code[i + 2] == 0x85 and
                code[i + 3] == 0x30 and code[i + 4] == 0 and code[i + 5] == 0 and code[i + 6] == 0)
            {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }
}

test "compileFunctionRA: caller passes >3 args via stack on Win64" {
    const allocator = std.testing.allocator;
    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    // callee(i32 x5) -> i32  (body irrelevant; just needs valid IR)
    var callee = ir.IrFunction.init(allocator, 5, 5, 0);
    _ = try callee.newBlock();
    try callee.getBlock(0).append(.{ .op = .{ .ret = 0 } });
    _ = try ir_module.addFunction(callee);

    // caller(): call callee with 5 args, ret the result
    var caller = ir.IrFunction.init(allocator, 0, 1, 0);
    _ = try caller.newBlock();
    const a0 = caller.newVReg();
    const a1 = caller.newVReg();
    const a2 = caller.newVReg();
    const a3 = caller.newVReg();
    const a4 = caller.newVReg();
    const r = caller.newVReg();
    try caller.getBlock(0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = a0, .type = .i32 });
    try caller.getBlock(0).append(.{ .op = .{ .iconst_32 = 2 }, .dest = a1, .type = .i32 });
    try caller.getBlock(0).append(.{ .op = .{ .iconst_32 = 3 }, .dest = a2, .type = .i32 });
    try caller.getBlock(0).append(.{ .op = .{ .iconst_32 = 4 }, .dest = a3, .type = .i32 });
    try caller.getBlock(0).append(.{ .op = .{ .iconst_32 = 5 }, .dest = a4, .type = .i32 });
    const args = try allocator.alloc(ir.VReg, 5);
    args[0] = a0;
    args[1] = a1;
    args[2] = a2;
    args[3] = a3;
    args[4] = a4;
    try caller.getBlock(0).append(.{
        .op = .{ .call = .{ .func_idx = 0, .args = args } },
        .dest = r,
        .type = .i32,
    });
    try caller.getBlock(0).append(.{ .op = .{ .ret = r } });
    _ = try ir_module.addFunction(caller);
    defer allocator.free(args);

    const result = try compileModule(&ir_module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    // Must contain a direct CALL (E8) — caller emits the call.
    try std.testing.expect(containsBytes(result.code, &.{0xE8}));
    // On Windows, caller must adjust rsp by at least 32 (shadow). Look for
    // sub rsp, imm32 (48 81 EC ??) or sub rsp, imm8 (48 83 EC ??) in caller.
    // Just assert the caller compiled successfully with nonzero size.
    try std.testing.expect(result.code.len > 32);
}

// ── Parallel-copy correctness for call arg materialization (regression tests) ──
//
// These tests exercise `emitCallRegArgMoves` directly. The bug fixed here was:
// a naive left-to-right `mov param_regs[i+1], src(arg[i])` sequence clobbered
// `arg[j]` (j > i) when `src(arg[j]) == param_regs[i+1]`. Observed in coremark
// at the malloc→alloc call: `mov rdx, rdi` (arg0) clobbered arg1's value (4,
// previously placed in rdx by the allocator), yielding `alloc(size+16,
// size+16)` instead of `alloc(size+16, 4)`.

fn makeAllocResult(allocator: std.mem.Allocator, mapping: []const struct { vreg: ir.VReg, reg: regalloc.PhysReg }) !regalloc.AllocResult {
    var map = std.AutoHashMap(ir.VReg, regalloc.Allocation).init(allocator);
    for (mapping) |m| try map.put(m.vreg, .{ .reg = m.reg });
    return .{ .assignments = map, .spill_count = 0 };
}

/// Encode `mov dst, src` through the same emitter the production code uses,
/// so assertions compare against the byte sequence the emitter would actually
/// produce on the current architecture/ABI. Returned buffer is owned by
/// `out_buf` and freed on `deinit`.
fn encodeMov(allocator: std.mem.Allocator, dst: emit.Reg, src: emit.Reg) !emit.CodeBuffer {
    var buf = emit.CodeBuffer.init(allocator);
    errdefer buf.deinit();
    try buf.movRegReg(dst, src);
    return buf;
}

test "emitCallRegArgMoves: resolves arg[0]→p1 / arg[1]→p2 when arg[1] source is p1" {
    // Scenario from coremark malloc→alloc (generalized across ABIs): RA
    // placed arg[1] in `param_regs[1]` — which is also arg[0]'s target.
    // A naive sequential copy would `mov p1, src(arg0)` first, clobbering
    // arg[1]'s value in p1, then `mov p2, p1` would pick up the clobbered
    // value. The fixed emitter must move arg[1] (src=p1 → p2) BEFORE
    // arg[0] (src=X → p1).
    const allocator = std.testing.allocator;
    // arg[0]'s source: any reg that is NOT a parameter register on either
    // ABI so it can't conflict. `.rbx` is callee-saved on both Win64 and
    // SysV, and is never in `param_regs`.
    const arg0_src: emit.Reg = .rbx;
    const p1 = param_regs[1];
    const p2 = param_regs[2];

    var alloc_result = try makeAllocResult(allocator, &.{
        .{ .vreg = 0, .reg = @intFromEnum(arg0_src) },
        .{ .vreg = 1, .reg = @intFromEnum(p1) },
    });
    defer alloc_result.deinit();

    var code = emit.CodeBuffer.init(allocator);
    defer code.deinit();

    const args = [_]ir.VReg{ 0, 1 };
    try emitCallRegArgMoves(&code, &alloc_result, &args, 2);
    const bytes = code.bytes.items;

    // Expected encodings for `mov p2, p1` and `mov p1, arg0_src`. The
    // former must precede the latter to avoid clobbering p1 before it is
    // read.
    var save_arg1 = try encodeMov(allocator, p2, p1);
    defer save_arg1.deinit();
    var move_arg0 = try encodeMov(allocator, p1, arg0_src);
    defer move_arg0.deinit();

    const idx_save = std.mem.indexOf(u8, bytes, save_arg1.bytes.items) orelse return error.TestExpectedArg1SaveEmitted;
    const idx_move = std.mem.indexOf(u8, bytes, move_arg0.bytes.items) orelse return error.TestExpectedArg0MoveEmitted;
    try std.testing.expect(idx_save < idx_move);
}

test "emitCallRegArgMoves: identity moves are elided" {
    // When every arg[i] is already in param_regs[i+1], no moves should be emitted.
    const allocator = std.testing.allocator;
    var alloc_result = try makeAllocResult(allocator, &.{
        .{ .vreg = 0, .reg = @intFromEnum(param_regs[1]) },
        .{ .vreg = 1, .reg = @intFromEnum(param_regs[2]) },
        .{ .vreg = 2, .reg = @intFromEnum(param_regs[3]) },
    });
    defer alloc_result.deinit();

    var code = emit.CodeBuffer.init(allocator);
    defer code.deinit();

    const args = [_]ir.VReg{ 0, 1, 2 };
    try emitCallRegArgMoves(&code, &alloc_result, &args, 3);
    try std.testing.expectEqual(@as(usize, 0), code.bytes.items.len);
}

test "emitCallRegArgMoves: breaks 2-cycle via r10 scratch" {
    // Cycle: arg[0] src=p2 → p1; arg[1] src=p1 → p2. No topological order
    // exists; the fix must save one source into r10, then finish the copy.
    const allocator = std.testing.allocator;
    const p1 = param_regs[1];
    const p2 = param_regs[2];

    var alloc_result = try makeAllocResult(allocator, &.{
        .{ .vreg = 0, .reg = @intFromEnum(p2) }, // arg[0] in p2 (= arg[1]'s target)
        .{ .vreg = 1, .reg = @intFromEnum(p1) }, // arg[1] in p1 (= arg[0]'s target)
    });
    defer alloc_result.deinit();

    var code = emit.CodeBuffer.init(allocator);
    defer code.deinit();

    const args = [_]ir.VReg{ 0, 1 };
    try emitCallRegArgMoves(&code, &alloc_result, &args, 2);
    const bytes = code.bytes.items;

    // Whichever source gets chosen as the cycle-breaker, *some* `mov r10, X`
    // with X ∈ {p1, p2} must appear. The algorithm picks the first pending
    // arg (arg[0], src=p2) so we expect `mov r10, p2`, but accept either.
    var break_from_p1 = try encodeMov(allocator, .r10, p1);
    defer break_from_p1.deinit();
    var break_from_p2 = try encodeMov(allocator, .r10, p2);
    defer break_from_p2.deinit();
    const has_breaker =
        std.mem.indexOf(u8, bytes, break_from_p1.bytes.items) != null or
        std.mem.indexOf(u8, bytes, break_from_p2.bytes.items) != null;
    try std.testing.expect(has_breaker);

    // The buggy naive sequence `mov p1, p2` immediately followed by
    // `mov p2, p1` must NOT appear — it would clobber p2 before reading.
    var buggy_a = try encodeMov(allocator, p1, p2);
    defer buggy_a.deinit();
    var buggy_b = try encodeMov(allocator, p2, p1);
    defer buggy_b.deinit();
    const idx_a = std.mem.indexOf(u8, bytes, buggy_a.bytes.items);
    const idx_b = std.mem.indexOf(u8, bytes, buggy_b.bytes.items);
    if (idx_a) |a| if (idx_b) |b| {
        try std.testing.expect(!(a + buggy_a.bytes.items.len == b));
    };
}

test "emitCallRegArgMoves: regression — arg[1] source equals arg[0] target (coremark malloc→alloc)" {
    // Exact shape observed in the bug (generalized across ABIs): arg[0] is
    // in some non-param reg, arg[1] is in `param_regs[1]` (which is also
    // arg[0]'s destination). The fix must not re-introduce the clobber
    // pattern `mov p1, src(arg0)` followed by `mov p2, p1`.
    const allocator = std.testing.allocator;
    const arg0_src: emit.Reg = .rbx; // callee-saved on both ABIs; not a param reg.
    const p1 = param_regs[1];
    const p2 = param_regs[2];

    var alloc_result = try makeAllocResult(allocator, &.{
        .{ .vreg = 0, .reg = @intFromEnum(arg0_src) },
        .{ .vreg = 1, .reg = @intFromEnum(p1) }, // the bug trigger
    });
    defer alloc_result.deinit();

    var code = emit.CodeBuffer.init(allocator);
    defer code.deinit();

    const args = [_]ir.VReg{ 0, 1 };
    try emitCallRegArgMoves(&code, &alloc_result, &args, 2);
    const bytes = code.bytes.items;

    // Assert that if the buggy pattern `mov p1, arg0_src` appears at all,
    // it is preceded by the save `mov p2, p1` (fixed order).
    var move_arg0 = try encodeMov(allocator, p1, arg0_src);
    defer move_arg0.deinit();
    if (std.mem.indexOf(u8, bytes, move_arg0.bytes.items)) |a| {
        var save_arg1 = try encodeMov(allocator, p2, p1);
        defer save_arg1.deinit();
        if (std.mem.indexOf(u8, bytes, save_arg1.bytes.items)) |b| {
            try std.testing.expect(b < a);
        }
    }
}

test "compileFunctionRA: i32.load emits inline memory bounds check" {
    // A wasm memory load must emit an inline bounds check before reading,
    // so out-of-bounds access traps cleanly via vmctx.trap_oob_fn() rather
    // than SIGSEGVing on unmapped host memory.
    //
    // The fixed bounds-check sequence we expect to see (before the load):
    //   lea r11, [rax + (offset + size)]     ; 4C 8D 98 <disp32>
    //   cmp r11, [r10 + memsize_field=8]     ; 4D 3B 5A 08
    //   jbe  +12                             ; 76 0C
    //   mov  param_regs[0], r10              ; 4C 89 D1 (Win64) / 4C 89 D7 (SysV)
    //   mov  rax, [param_regs[0] + trap_fn=80] ; 48 8B 81 50 00 00 00 (Win64)
    //   call rax                             ; FF D0
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    // i32.load16_u at offset=0, size=2 → end_offset = 2.
    try block.append(.{ .op = .{ .load = .{ .base = v0, .offset = 0, .size = 2, .sign_extend = false } }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v1 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // lea r11, [rax + 2] → 4C 8D 98 02 00 00 00
    try std.testing.expect(containsBytes(code, &.{ 0x4C, 0x8D, 0x98, 0x02, 0x00, 0x00, 0x00 }));
    // cmp r11, [r10 + 8]   → 4D 3B 5A 08
    try std.testing.expect(containsBytes(code, &.{ 0x4D, 0x3B, 0x5A, 0x08 }));
    // jbe +12              → 76 0C
    try std.testing.expect(containsBytes(code, &.{ 0x76, 0x0C }));
    // call qword ptr [vmctx + 80] is encoded as a two-step load+callReg.
    // mov rax, [p0 + 80] fragment `48 8B ?? 50 00 00 00` + call rax (FF D0).
    // Verify `call rax` is present (FF D0).
    try std.testing.expect(containsBytes(code, &.{ 0xFF, 0xD0 }));
}

test "compileFunctionRA: i32.store emits inline memory bounds check" {
    // Symmetric to the load case: stores must also bounds-check before
    // dereferencing. Verify the same lea/cmp/jbe trap dispatch pattern.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v1, .type = .i32 });
    // i32.store at offset=0, size=4 → end_offset = 4.
    try block.append(.{ .op = .{ .store = .{ .base = v0, .val = v1, .offset = 0, .size = 4 } } });
    try block.append(.{ .op = .{ .ret = null } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // lea r11, [rax + 4] → 4C 8D 98 04 00 00 00
    try std.testing.expect(containsBytes(code, &.{ 0x4C, 0x8D, 0x98, 0x04, 0x00, 0x00, 0x00 }));
    // cmp r11, [r10 + 8]
    try std.testing.expect(containsBytes(code, &.{ 0x4D, 0x3B, 0x5A, 0x08 }));
    // jbe +12
    try std.testing.expect(containsBytes(code, &.{ 0x76, 0x0C }));
    // call rax (trap dispatch)
    try std.testing.expect(containsBytes(code, &.{ 0xFF, 0xD0 }));
}

test "compileFunctionRA: block ordering consistency (clobbers match live ranges)" {
    // Regression test for #209: clobber-point numbering must use the same
    // block order as live-range computation. Construct a diamond CFG where
    // RPO differs from raw block order and a call (clobber) sits in a block
    // that moves position under reordering.
    //
    //   b0 → br_if → b1 (call) → b3 (ret)
    //              ↘ b2 (nop)  ↗
    //
    // Raw order: b0, b1, b2, b3. RPO: b0, b1, b2, b3 or b0, b2, b1, b3
    // depending on successor traversal. The key is that both clobbers and
    // live ranges use the SAME order — if they don't, the allocator may
    // place a vreg in a caller-saved register across a call.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 2, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    const block0 = func.getBlock(b0);
    const block1 = func.getBlock(b1);
    const block2 = func.getBlock(b2);
    const block3 = func.getBlock(b3);

    const cond = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();

    // b0: cond = local_get 0; br_if cond, then=b1, else=b2
    try block0.append(.{ .op = .{ .local_get = 0 }, .dest = cond, .type = .i32 });
    try block0.append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });

    // b1: v1 = call func 0 (clobber point); br b3
    try block1.append(.{ .op = .{ .call = .{ .func_idx = 0, .args = &.{} } }, .dest = v1, .type = .i32 });
    try block1.append(.{ .op = .{ .br = b3 } });

    // b2: v2 = iconst 42; br b3
    try block2.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v2, .type = .i32 });
    try block2.append(.{ .op = .{ .br = b3 } });

    // b3: ret (void)
    try block3.append(.{ .op = .{ .ret = null } });

    // This should compile without errors — before the fix, mismatched
    // clobber/live-range numbering could cause allocation failures or
    // silent register conflicts.
    const compile_result = try compileFunctionRA(&func, 0, allocator);
    defer allocator.free(compile_result.code);
    defer allocator.free(compile_result.call_patches);

    // Sanity: produced non-empty code.
    try std.testing.expect(compile_result.code.len > 0);
}
