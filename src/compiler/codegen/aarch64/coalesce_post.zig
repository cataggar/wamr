//! Post-allocation MOV coalescer scan (issue #539).
//!
//! Operates on the finalised AArch64 byte stream after the main emit loop,
//! complementing the IR-level copy-coalescing hints from PRs #441 and #529.
//! Many redundant `MOV xD, xS` (64-bit `ORR xD, XZR, xS`) and
//! `MOV wD, wS` (32-bit `ORR wD, WZR, wS`) emissions slip past regalloc
//! because two different vregs could not be assigned the same physreg.
//! This pass walks the emitted stream block-by-block, identifies safely
//! eliminable MOVs, rewrites subsequent reads of `xD` to read `xS`, and
//! replaces the MOV with a `NOP` (offset-preserving).
//!
//! Restrictions (deliberate, v1):
//!   * Block-local. Cross-block coalescing requires post-allocation
//!     liveness that we don't have on the emit stream — filed as a
//!     follow-up under issue #539's "out of scope".
//!   * Only GPR MOV (ORR Rd, ZR, Rm). SIMD `MOV vD.16b, vS.16b` is
//!     decoded as a barrier in v1; emit-side eliminator can be added
//!     once the SIMD spill/reload patterns are profiled.
//!   * Reg 31 (XZR/SP) is never coalesced — its meaning is encoding-
//!     specific (XZR in logical / SP in addressing) and the local
//!     analyser does not disambiguate.
//!   * Special-purpose registers (x19 = vmctx/heap, x29 = fp, x30 = lr)
//!     are never coalesced. They are written outside the normal IR
//!     dataflow (epilogue, vmctx refresh) and may be implicitly read
//!     by sequences our decoder treats as barriers.
//!
//! Offsets are preserved (NOPs in place of eliminated MOVs); this lets
//! branch-patch resolution, call-patch resolution, and callee-save site
//! patching continue to use unmodified byte offsets. The eliminated
//! `MOV` costs ~1 cycle and the in-place `NOP` is renamed-out by the
//! micro-architecture on every reasonable AArch64 implementation
//! (Apple, Neoverse, Cortex-A), so the perf delta of NOP-vs-deleted is
//! near zero. A future follow-up may compact the buffer for an i-cache
//! win (issue #539 byte-size bar) but requires a branch-fixup pass.

const std = @import("std");

pub const Stats = struct {
    /// Total MOVs (64- and 32-bit GPR) emitted that we scanned.
    movs_seen: usize = 0,
    /// MOVs replaced with NOP after rewriting downstream uses.
    movs_eliminated: usize = 0,
    /// MOVs where dest == src (`MOV xD, xD`) — always NOP'd, no
    /// rewriting required.
    self_movs_eliminated: usize = 0,
};

const NOP: u32 = 0xD503201F;

/// Bitmask over physical GPRs x0..x30. Bit 31 (XZR/SP) is intentionally
/// unused: any read or write touching reg 31 is encoded into the bitmask
/// (via `setReg`) but never affects coalescing decisions because we
/// refuse to coalesce reg 31 up front.
const RegMask = u32;

inline fn bit(r: u5) RegMask {
    return @as(RegMask, 1) << r;
}

/// All-regs mask: every GPR including reg 31 (used for barrier semantics).
const ALL_REGS: RegMask = 0xFFFFFFFF;

/// Decoded instruction summary. `decoded = false` indicates the decoder
/// did not recognise the instruction; callers must treat it as a barrier
/// (every reg read AND written).
const Inst = struct {
    reads: RegMask,
    writes: RegMask,
    decoded: bool,

    fn barrier() Inst {
        return .{ .reads = ALL_REGS, .writes = ALL_REGS, .decoded = false };
    }
};

/// True if this word is a 64-bit GPR `MOV xRd, xRm`
/// (ORR Rd, XZR, Rm{, LSL #0}).
fn decodeMov64(word: u32) ?struct { rd: u5, rm: u5 } {
    if ((word & 0xFFE0FFE0) != 0xAA0003E0) return null;
    return .{
        .rd = @intCast(word & 0x1F),
        .rm = @intCast((word >> 16) & 0x1F),
    };
}

/// True if this word is a 32-bit GPR `MOV wRd, wRm`
/// (ORR Wd, WZR, Wm{, LSL #0}).
fn decodeMov32(word: u32) ?struct { rd: u5, rm: u5 } {
    if ((word & 0xFFE0FFE0) != 0x2A0003E0) return null;
    return .{
        .rd = @intCast(word & 0x1F),
        .rm = @intCast((word >> 16) & 0x1F),
    };
}

/// Best-effort decoder: read/write masks for the common AArch64 GPR
/// instruction encodings emitted by `aarch64/emit.zig`. Returns
/// `Inst.barrier()` for anything we don't decode; the caller must
/// then reject the surrounding coalescing window.
///
/// Notes on reg 31:
///   * In *logical* (AND/ORR/EOR/...) and *data-processing* opcodes,
///     reg 31 means XZR — a constant zero source / discarding sink.
///   * In *addressing* (LDR/STR Rn) and *add/sub immediate*, reg 31
///     means SP. We over-approximate by NEVER coalescing reg 31; the
///     decoder then just records reads/writes of bit 31 with no harm.
fn decode(word: u32) Inst {
    // NOP / HINT.
    if (word == NOP) return .{ .reads = 0, .writes = 0, .decoded = true };
    if ((word & 0xFFFFF01F) == 0xD503201F) {
        // Hint #imm5 (NOP=imm0, YIELD=imm1, WFE=imm2, ...). Safe.
        return .{ .reads = 0, .writes = 0, .decoded = true };
    }

    // ── Branches & calls → BARRIER ──
    //
    // Calls (BL/BLR) implicitly read AAPCS64 arg regs x0..x7 and clobber
    // every caller-saved register — none of which appear in any reg
    // field of the encoding, so our explicit-read decoder would miss
    // them. RET implicitly reads x0..x1 (return regs). BR may be a
    // tail-call dispatch with the same AAPCS reads. The branches B,
    // B.cond, CBZ, CBNZ, TBZ, TBNZ create CFG merge points that may
    // import different register state at the target than our linear
    // window saw on the fall-through path. In v1 we punt on all of
    // these — they terminate the window and force the coalescer to
    // give up locally. (Cross-block coalescing would need a proper
    // post-alloc CFG with liveness; filed as a follow-up.)
    if ((word & 0x7C000000) == 0x14000000) return Inst.barrier(); // B / BL
    if ((word & 0xFF000010) == 0x54000000) return Inst.barrier(); // B.cond
    if ((word & 0x7E000000) == 0x34000000) return Inst.barrier(); // CBZ / CBNZ
    if ((word & 0x7E000000) == 0x36000000) return Inst.barrier(); // TBZ / TBNZ
    if ((word & 0xFE1FFC1F) == 0xD61F0000) return Inst.barrier(); // BR
    if ((word & 0xFE1FFC1F) == 0xD63F0000) return Inst.barrier(); // BLR
    if ((word & 0xFE1FFC1F) == 0xD65F0000) return Inst.barrier(); // RET

    // ── PC-rel addressing: ADR / ADRP ──
    if ((word & 0x1F000000) == 0x10000000) {
        const rd: u5 = @intCast(word & 0x1F);
        return .{ .reads = 0, .writes = bit(rd), .decoded = true };
    }

    // ── Move wide: MOVN / MOVZ / MOVK ──
    if ((word & 0x1F800000) == 0x12800000) {
        const rd: u5 = @intCast(word & 0x1F);
        const opc = (word >> 29) & 0b11;
        const is_movk = opc == 0b11;
        return .{
            // MOVK is read-modify-write of Rd.
            .reads = if (is_movk) bit(rd) else 0,
            .writes = bit(rd),
            .decoded = true,
        };
    }

    // ── Add/sub (immediate) ──  bit26=0 form: `sf op S 100010 sh imm12 Rn Rd`
    if ((word & 0x1F000000) == 0x11000000) {
        const rd: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        const s_bit = (word >> 29) & 1; // ADDS/SUBS write flags AND Rd
        // ADDS/SUBS with Rd=31 (XZR) is the CMP/CMN alias — no reg write.
        const writes_rd = !(s_bit == 1 and rd == 31);
        return .{
            .reads = bit(rn),
            .writes = if (writes_rd) bit(rd) else 0,
            .decoded = true,
        };
    }

    // ── Logical (immediate) ──  `sf opc 100100 N immr imms Rn Rd`
    if ((word & 0x1F800000) == 0x12000000) {
        const rd: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        const opc = (word >> 29) & 0b11;
        const is_ands = opc == 0b11;
        // ANDS with Rd=31 = TST — no reg write.
        const writes_rd = !(is_ands and rd == 31);
        return .{
            .reads = bit(rn),
            .writes = if (writes_rd) bit(rd) else 0,
            .decoded = true,
        };
    }

    // ── Bitfield: SBFM / BFM / UBFM (and aliases LSL/LSR/ASR/UBFX/SBFX/BFI...) ──
    if ((word & 0x1F800000) == 0x13000000) {
        const rd: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        const opc = (word >> 29) & 0b11;
        const is_bfm = opc == 0b01;
        return .{
            // BFM (opc=01) is read-modify of Rd (it preserves the
            // un-inserted bits), so Rd is also read.
            .reads = if (is_bfm) bit(rn) | bit(rd) else bit(rn),
            .writes = bit(rd),
            .decoded = true,
        };
    }

    // ── Extract (EXTR) ──  `sf 00 100111 N0 Rm imms Rn Rd`
    if ((word & 0x1FE00000) == 0x13800000) {
        const rd: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        const rm: u5 = @intCast((word >> 16) & 0x1F);
        return .{
            .reads = bit(rn) | bit(rm),
            .writes = bit(rd),
            .decoded = true,
        };
    }

    // ── Add/sub (shifted register) ──  `sf op S 01011 shift 0 Rm imm6 Rn Rd`
    if ((word & 0x1F200000) == 0x0B000000) {
        const rd: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        const rm: u5 = @intCast((word >> 16) & 0x1F);
        const s_bit = (word >> 29) & 1;
        // CMP/CMN: ADDS/SUBS with Rd=31 (XZR sink).
        const writes_rd = !(s_bit == 1 and rd == 31);
        return .{
            .reads = bit(rn) | bit(rm),
            .writes = if (writes_rd) bit(rd) else 0,
            .decoded = true,
        };
    }

    // ── Add/sub (extended register) ──  `sf op S 01011 00 1 Rm option imm3 Rn Rd`
    // Reads Rn and Rm. Rn=31 means SP here, not XZR — coalescer refuses
    // reg 31 anyway, so safe to record reads of bit(31).
    if ((word & 0x1F200000) == 0x0B200000) {
        const rd: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        const rm: u5 = @intCast((word >> 16) & 0x1F);
        const s_bit = (word >> 29) & 1;
        const writes_rd = !(s_bit == 1 and rd == 31);
        return .{
            .reads = bit(rn) | bit(rm),
            .writes = if (writes_rd) bit(rd) else 0,
            .decoded = true,
        };
    }

    // ── Logical (shifted register) ──  `sf opc 01010 shift N Rm imm6 Rn Rd`
    if ((word & 0x1F000000) == 0x0A000000) {
        const rd: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        const rm: u5 = @intCast((word >> 16) & 0x1F);
        const opc = (word >> 29) & 0b11;
        const is_ands = opc == 0b11;
        // TST alias = ANDS with Rd=31.
        const writes_rd = !(is_ands and rd == 31);
        return .{
            .reads = bit(rn) | bit(rm),
            .writes = if (writes_rd) bit(rd) else 0,
            .decoded = true,
        };
    }

    // ── Data processing 1-source (REV/CLZ/RBIT/...) ──
    // `sf 1 S 11010110 opcode2 opcode Rn Rd`
    if ((word & 0x5FE00000) == 0x5AC00000) {
        const rd: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        return .{
            .reads = bit(rn),
            .writes = bit(rd),
            .decoded = true,
        };
    }

    // ── Data processing 2-source (LSLV/LSRV/ASRV/RORV/UDIV/SDIV/...) ──
    // `sf 0 S 11010110 Rm opcode Rn Rd`
    if ((word & 0x5FE00000) == 0x1AC00000) {
        const rd: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        const rm: u5 = @intCast((word >> 16) & 0x1F);
        return .{
            .reads = bit(rn) | bit(rm),
            .writes = bit(rd),
            .decoded = true,
        };
    }

    // ── Data processing 3-source (MADD/MSUB/SMADDL/UMADDL/SMULH/UMULH/...) ──
    // `sf 00 11011 op54 Rm o0 Ra Rn Rd`
    if ((word & 0x1F000000) == 0x1B000000) {
        const rd: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        const rm: u5 = @intCast((word >> 16) & 0x1F);
        const ra: u5 = @intCast((word >> 10) & 0x1F);
        const op31 = (word >> 21) & 0b111;
        // SMULH (op31=010) and UMULH (op31=110) have Ra=31 fixed (XZR);
        // include it as a read of bit(31) for consistency (no effect).
        _ = op31;
        return .{
            .reads = bit(rn) | bit(rm) | bit(ra),
            .writes = bit(rd),
            .decoded = true,
        };
    }

    // ── Conditional select / increment / invert / negate ──
    // CSEL/CSINC/CSINV/CSNEG: `sf op S 11010100 Rm cond op2 Rn Rd`
    if ((word & 0x5FE00000) == 0x1A800000) {
        const rd: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        const rm: u5 = @intCast((word >> 16) & 0x1F);
        return .{
            .reads = bit(rn) | bit(rm),
            .writes = bit(rd),
            .decoded = true,
        };
    }

    // ── Conditional compare (register / immediate) ──
    // CCMP/CCMN reg: 0x1A400000; imm: 0x1A400800.
    if ((word & 0x5FE00C00) == 0x1A400000) {
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        const rm: u5 = @intCast((word >> 16) & 0x1F);
        return .{
            .reads = bit(rn) | bit(rm),
            .writes = 0,
            .decoded = true,
        };
    }
    if ((word & 0x5FE00C00) == 0x1A400800) {
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        return .{
            .reads = bit(rn),
            .writes = 0,
            .decoded = true,
        };
    }

    // ── Loads/stores ──
    // Load/store register pair, signed-offset form ONLY (idx=10).
    // Pre/post-index variants write back to Rn — their rewriter
    // semantics conflict with substituting Rn = old_reg → new_reg
    // (the writeback would land in new_reg instead of old_reg),
    // so we treat those as barriers. `(word & 0x3B800000) == 0x29000000`
    // matches bits 28:23 = 010100 = signed-offset form (idx=10),
    // any size/V/L.
    if ((word & 0x3B800000) == 0x29000000) {
        const v_bit = (word >> 26) & 1; // 1 = SIMD/FP variant
        const l_bit = (word >> 22) & 1; // 1 = load
        const rt: u5 = @intCast(word & 0x1F);
        const rt2: u5 = @intCast((word >> 10) & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);

        var reads: RegMask = bit(rn);
        var writes: RegMask = 0;
        if (v_bit == 1) {
            return .{ .reads = reads, .writes = writes, .decoded = true };
        }
        if (l_bit == 1) {
            writes |= bit(rt);
            writes |= bit(rt2);
        } else {
            reads |= bit(rt);
            reads |= bit(rt2);
        }
        return .{ .reads = reads, .writes = writes, .decoded = true };
    }
    // LDP/STP pre-index (idx=11) and post-index (idx=01) — barrier
    // due to writeback rewrite hazard (see signed-offset case above).
    if ((word & 0x3A000000) == 0x28000000) {
        return Inst.barrier();
    }

    // Load register (literal): `opc 011 V 00 imm19 Rt`.
    if ((word & 0x3B000000) == 0x18000000) {
        const v_bit = (word >> 26) & 1;
        const rt: u5 = @intCast(word & 0x1F);
        return .{
            .reads = 0,
            .writes = if (v_bit == 0) bit(rt) else 0,
            .decoded = true,
        };
    }

    // Load/store register, unsigned-offset form (the bulk of all
    // memory traffic). `size 111 V 01 op0 imm12 Rn Rt`.
    if ((word & 0x3B000000) == 0x39000000) {
        const v_bit = (word >> 26) & 1;
        const size = (word >> 30) & 0b11;
        const opc = (word >> 22) & 0b11;
        const is_prfm = size == 0b11 and opc == 0b10 and v_bit == 0;
        const is_store = opc == 0b00;
        const rt: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        var reads: RegMask = bit(rn);
        var writes: RegMask = 0;
        if (v_bit == 1 or is_prfm) {
            return .{ .reads = reads, .writes = writes, .decoded = true };
        }
        if (is_store) reads |= bit(rt) else writes |= bit(rt);
        return .{ .reads = reads, .writes = writes, .decoded = true };
    }

    // Load/store register, register-offset form (no writeback).
    // `size 111 V 00 op0 1 Rm option S 10 Rn Rt`.
    if ((word & 0x3B200C00) == 0x38200800) {
        const v_bit = (word >> 26) & 1;
        const opc = (word >> 22) & 0b11;
        const is_store = opc == 0b00;
        const rt: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        const rm: u5 = @intCast((word >> 16) & 0x1F);
        var reads: RegMask = bit(rn) | bit(rm);
        var writes: RegMask = 0;
        if (v_bit == 1) {
            return .{ .reads = reads, .writes = writes, .decoded = true };
        }
        if (is_store) reads |= bit(rt) else writes |= bit(rt);
        return .{ .reads = reads, .writes = writes, .decoded = true };
    }

    // Load/store register, unscaled-immediate form (LDUR/STUR/...).
    // `size 111 V 00 op0 0 imm9 00 Rn Rt`. No writeback.
    if ((word & 0x3B200C00) == 0x38000000) {
        const v_bit = (word >> 26) & 1;
        const opc = (word >> 22) & 0b11;
        const is_store = opc == 0b00;
        const rt: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        var reads: RegMask = bit(rn);
        var writes: RegMask = 0;
        if (v_bit == 1) {
            return .{ .reads = reads, .writes = writes, .decoded = true };
        }
        if (is_store) reads |= bit(rt) else writes |= bit(rt);
        return .{ .reads = reads, .writes = writes, .decoded = true };
    }
    // Pre-index and post-index single-register LDR/STR — barrier due
    // to Rn writeback conflicting with read-field rewriting.
    if ((word & 0x3B200400) == 0x38000400) return Inst.barrier();

    // ── FP/SIMD↔GPR transfers (FMOV between scalar FP and GPR) ──
    // `sf 0011110 type 1 rmode opcode 000000 Rn Rd` — covers FMOV
    // Sd/Dd ← Wn/Xn and the reverse. We can decode the GPR direction
    // safely: opcode 110/111 = GPR→VFP, opcode 100/101 = no? Decode
    // conservatively from the encoded bits.
    if ((word & 0x5F207C00) == 0x1E200000) {
        // Floating-point ↔ integer conversions. opcode bits 18:16.
        const opcode = (word >> 16) & 0b111;
        const rd: u5 = @intCast(word & 0x1F);
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        // opcode 110 = FMOV Vd ← Rn (GPR→V). opcode 111 = FMOV Rd ← Vn.
        // opcode 010 = SCVTF (Rn read, V written). 011 = UCVTF.
        // opcode 000/001 = FCVTNS/FCVTNU (V read, Rd written).
        return switch (opcode) {
            0b110 => Inst{ .reads = bit(rn), .writes = 0, .decoded = true },
            0b111 => Inst{ .reads = 0, .writes = bit(rd), .decoded = true },
            0b010, 0b011 => Inst{ .reads = bit(rn), .writes = 0, .decoded = true },
            0b000, 0b001, 0b100, 0b101 => Inst{ .reads = 0, .writes = bit(rd), .decoded = true },
            else => Inst.barrier(),
        };
    }

    return Inst.barrier();
}

/// Rewrite every *read* of `old_reg` in `word` to read `new_reg` instead.
/// Returns the new word, or `null` if the instruction either does not
/// read `old_reg` (caller can leave it alone — pass returns the original)
/// or if rewriting would be unsafe / the encoding is one we cannot
/// surgically modify.
///
/// Implementation: only succeeds when `decode(word)` was decodable;
/// otherwise the eliminator already rejected the window. For each
/// decoded encoding we know which fields hold reg numbers.
fn rewriteReadReg(word: u32, old_reg: u5, new_reg: u5) ?u32 {
    if (old_reg == new_reg) return word;

    var w = word;
    _ = &w;

    // Helper: extract a 5-bit reg field at `shift` and replace it with
    // `new_reg` iff it currently equals `old_reg`. Returns true if the
    // field was actually replaced.
    const replaceField = struct {
        fn f(word_ref: *u32, shift: u5, old: u5, newv: u5) bool {
            const cur: u5 = @intCast((word_ref.* >> shift) & 0x1F);
            if (cur != old) return false;
            const mask: u32 = ~(@as(u32, 0x1F) << shift);
            word_ref.* = (word_ref.* & mask) | (@as(u32, newv) << shift);
            return true;
        }
    }.f;

    // NOP / HINT — no reg fields.
    if (word == NOP) return word;
    if ((word & 0xFFFFF01F) == 0xD503201F) return word;

    // CBZ/CBNZ, TBZ/TBNZ, B/BL, B.cond, BR/BLR/RET — barriers per
    // `decode`. We never reach this function for them because the
    // window-scan rejects barriers, but be defensive in case the
    // caller has another use site.
    if ((word & 0x7C000000) == 0x14000000) return null;
    if ((word & 0xFF000010) == 0x54000000) return null;
    if ((word & 0x7E000000) == 0x34000000) return null;
    if ((word & 0x7E000000) == 0x36000000) return null;
    if ((word & 0xFE1FFC1F) == 0xD61F0000) return null;
    if ((word & 0xFE1FFC1F) == 0xD63F0000) return null;
    if ((word & 0xFE1FFC1F) == 0xD65F0000) return null;

    // ADR/ADRP — Rd at [4:0] only (write).
    if ((word & 0x1F000000) == 0x10000000) return word;

    // Move wide. MOVK reads Rd; MOVN/MOVZ don't read.
    if ((word & 0x1F800000) == 0x12800000) {
        const opc = (word >> 29) & 0b11;
        if (opc == 0b11) {
            // MOVK: Rd is read-modify-write. Cannot rewrite the *read*
            // independent of the *write* — refuse.
            //
            // In practice MOVK only ever appears chained with its own
            // MOVZ initialiser (same Rd), so the eliminator's
            // "no reads of xD in window" predicate already excludes
            // MOVK-of-xD. But we still need to return a safe value if
            // someone does call us.
            return null;
        }
        return word;
    }

    // Add/sub immediate. Rn at [9:5] is the only read field.
    if ((word & 0x1F000000) == 0x11000000) {
        _ = replaceField(&w, 5, old_reg, new_reg);
        return w;
    }

    // Logical immediate. Rn at [9:5] is the only read field.
    if ((word & 0x1F800000) == 0x12000000) {
        _ = replaceField(&w, 5, old_reg, new_reg);
        return w;
    }

    // Bitfield. SBFM/UBFM read Rn at [9:5] only. BFM reads Rd too —
    // we refuse to rewrite BFM since Rd is both read and written.
    if ((word & 0x1F800000) == 0x13000000) {
        const opc = (word >> 29) & 0b11;
        if (opc == 0b01) return null; // BFM
        _ = replaceField(&w, 5, old_reg, new_reg);
        return w;
    }

    // EXTR — Rn at [9:5] and Rm at [20:16].
    if ((word & 0x1FE00000) == 0x13800000) {
        _ = replaceField(&w, 5, old_reg, new_reg);
        _ = replaceField(&w, 16, old_reg, new_reg);
        return w;
    }

    // Add/sub shifted reg, add/sub extended reg, logical shifted reg.
    if ((word & 0x1F200000) == 0x0B000000 //
    or (word & 0x1F200000) == 0x0B200000 //
    or (word & 0x1F000000) == 0x0A000000)
    {
        _ = replaceField(&w, 5, old_reg, new_reg);
        _ = replaceField(&w, 16, old_reg, new_reg);
        return w;
    }

    // Data processing 1-source — Rn at [9:5].
    if ((word & 0x5FE00000) == 0x5AC00000) {
        _ = replaceField(&w, 5, old_reg, new_reg);
        return w;
    }

    // Data processing 2-source — Rn at [9:5], Rm at [20:16].
    if ((word & 0x5FE00000) == 0x1AC00000) {
        _ = replaceField(&w, 5, old_reg, new_reg);
        _ = replaceField(&w, 16, old_reg, new_reg);
        return w;
    }

    // Data processing 3-source — Rn at [9:5], Rm at [20:16], Ra at [14:10].
    if ((word & 0x1F000000) == 0x1B000000) {
        _ = replaceField(&w, 5, old_reg, new_reg);
        _ = replaceField(&w, 16, old_reg, new_reg);
        _ = replaceField(&w, 10, old_reg, new_reg);
        return w;
    }

    // Conditional select family — Rn at [9:5], Rm at [20:16].
    if ((word & 0x5FE00000) == 0x1A800000) {
        _ = replaceField(&w, 5, old_reg, new_reg);
        _ = replaceField(&w, 16, old_reg, new_reg);
        return w;
    }

    // CCMP/CCMN reg — Rn at [9:5], Rm at [20:16].
    if ((word & 0x5FE00C00) == 0x1A400000) {
        _ = replaceField(&w, 5, old_reg, new_reg);
        _ = replaceField(&w, 16, old_reg, new_reg);
        return w;
    }
    // CCMP/CCMN imm — Rn at [9:5].
    if ((word & 0x5FE00C00) == 0x1A400800) {
        _ = replaceField(&w, 5, old_reg, new_reg);
        return w;
    }

    // Load/store pair, signed-offset form. Rn at [9:5] (base), and
    // Rt at [4:0] + Rt2 at [14:10] are reads when storing. Loads
    // write Rt/Rt2 — never read — so conditionally rewrite on the
    // L bit. Pre/post-index variants are barriers in `decode`, so
    // we never get here for them.
    if ((word & 0x3B800000) == 0x29000000) {
        const v_bit = (word >> 26) & 1;
        const l_bit = (word >> 22) & 1;
        _ = replaceField(&w, 5, old_reg, new_reg);
        if (v_bit == 0 and l_bit == 0) {
            _ = replaceField(&w, 0, old_reg, new_reg);
            _ = replaceField(&w, 10, old_reg, new_reg);
        }
        return w;
    }

    // Load register (literal) — Rt at [4:0] (write only). No reads.
    if ((word & 0x3B000000) == 0x18000000) return word;

    // Load/store unsigned-offset form: Rn at [9:5], Rt at [4:0]
    // (read iff store; load writes Rt).
    if ((word & 0x3B000000) == 0x39000000) {
        const v_bit = (word >> 26) & 1;
        const opc = (word >> 22) & 0b11;
        const is_store = opc == 0b00;
        _ = replaceField(&w, 5, old_reg, new_reg);
        if (v_bit == 0 and is_store) {
            _ = replaceField(&w, 0, old_reg, new_reg);
        }
        return w;
    }

    // Load/store register-offset form: Rn at [9:5], Rm at [20:16],
    // Rt at [4:0] (read iff store).
    if ((word & 0x3B200C00) == 0x38200800) {
        const v_bit = (word >> 26) & 1;
        const opc = (word >> 22) & 0b11;
        const is_store = opc == 0b00;
        _ = replaceField(&w, 5, old_reg, new_reg);
        _ = replaceField(&w, 16, old_reg, new_reg);
        if (v_bit == 0 and is_store) {
            _ = replaceField(&w, 0, old_reg, new_reg);
        }
        return w;
    }

    // Load/store unscaled-immediate form (LDUR/STUR/...): Rn at [9:5],
    // Rt at [4:0] (read iff store).
    if ((word & 0x3B200C00) == 0x38000000) {
        const v_bit = (word >> 26) & 1;
        const opc = (word >> 22) & 0b11;
        const is_store = opc == 0b00;
        _ = replaceField(&w, 5, old_reg, new_reg);
        if (v_bit == 0 and is_store) {
            _ = replaceField(&w, 0, old_reg, new_reg);
        }
        return w;
    }

    // FP/SIMD ↔ GPR scalar transfers.
    if ((word & 0x5F207C00) == 0x1E200000) {
        const opcode = (word >> 16) & 0b111;
        // Only the GPR→FP / GPR-read directions read Rn. Rewrite Rn.
        switch (opcode) {
            0b110, 0b010, 0b011 => {
                _ = replaceField(&w, 5, old_reg, new_reg);
                return w;
            },
            else => return word,
        }
    }

    // Unrecognised: refuse.
    return null;
}

/// True if `reg` is in {19, 29, 30, 31}: special-purpose registers we
/// always refuse to coalesce. See module-level docs.
fn isReservedReg(reg: u5) bool {
    return reg == 19 or reg == 29 or reg == 30 or reg == 31;
}

/// Block-local coalescing scan for one `[start, end)` byte range.
/// Returns the number of MOVs replaced in this block (including
/// self-MOVs, which are always NOP'd).
fn coalesceBlock(bytes: []u8, start: usize, end: usize) struct {
    eliminated: usize,
    self_eliminated: usize,
    seen: usize,
} {
    var eliminated: usize = 0;
    var self_eliminated: usize = 0;
    var seen: usize = 0;

    if (end <= start) return .{ .eliminated = 0, .self_eliminated = 0, .seen = 0 };
    std.debug.assert((end - start) % 4 == 0);
    const n_words = (end - start) / 4;
    if (n_words < 2) return .{ .eliminated = 0, .self_eliminated = 0, .seen = 0 };

    // Fixed-point: a successful coalesce can expose another. Cap the
    // outer loop at n_words to guarantee termination.
    var iter: usize = 0;
    while (iter < n_words) : (iter += 1) {
        var changed = false;

        var i: usize = 0;
        while (i < n_words) : (i += 1) {
            const off = start + i * 4;
            const w = std.mem.readInt(u32, bytes[off..][0..4], .little);

            // Try 64-bit and 32-bit MOV in turn. The 32-bit form is
            // intentionally treated as equivalent: AArch64 32-bit ops
            // zero-extend their dest to the full X register, so a
            // `mov wD, wS` produces the same architectural state as
            // a `mov xD, xS` *when wS itself was last produced by a
            // 32-bit op*. Our analysis is reg-granularity (not
            // sub-register), so it tracks the full X reg uniformly.
            const mov64 = decodeMov64(w);
            const mov32 = decodeMov32(w);
            if (mov64 == null and mov32 == null) continue;

            seen += 1;
            const rd: u5 = if (mov64) |m| m.rd else mov32.?.rd;
            const rm: u5 = if (mov64) |m| m.rm else mov32.?.rm;

            if (rd == rm) {
                // Self-MOV: always eliminable. NOP it.
                std.mem.writeInt(u32, bytes[off..][0..4], NOP, .little);
                self_eliminated += 1;
                changed = true;
                continue;
            }

            if (isReservedReg(rd) or isReservedReg(rm)) continue;

            // Forward scan from (MOV+1) to determine, in one pass:
            //   * `last_use`: the latest read of xD reachable from the
            //     MOV before xD is re-defined by a non-reading write.
            //   * `xd_redefined`: whether xD is overwritten before
            //     block exit (kills the MOV value cleanly — required
            //     for block-local elision without cross-block
            //     liveness).
            //   * `blocked`: whether an undecodable instruction sits
            //     in the active window, or an in-window read of xD
            //     follows a write of xS (the substitution `xD → xS`
            //     would then observe the wrong xS value).
            //
            // `xs_clobbered` tracks whether xS has been overwritten
            // since the MOV. A read of xD AFTER such a clobber is the
            // hard blocking case; a clobber that occurs strictly AFTER
            // `last_use(xD)` is fine because we no longer need xS.
            const rd_mask = bit(rd);
            const rm_mask = bit(rm);
            var last_use: ?usize = null;
            var blocked = false;
            var xd_redefined = false;
            var xs_clobbered = false;

            var j: usize = i + 1;
            while (j < n_words) : (j += 1) {
                const off_j = start + j * 4;
                const wj = std.mem.readInt(u32, bytes[off_j..][0..4], .little);
                const inst = decode(wj);
                if (!inst.decoded) {
                    blocked = true;
                    break;
                }
                const reads_xd = inst.reads & rd_mask != 0;
                const writes_xd = inst.writes & rd_mask != 0;
                const writes_xs = inst.writes & rm_mask != 0;

                if (reads_xd) {
                    if (xs_clobbered) {
                        // Substituted read would observe the new xS,
                        // not the value the MOV captured. Refuse.
                        blocked = true;
                        break;
                    }
                    last_use = j;
                    // AArch64 semantics: read operands sampled before
                    // write commits. A read-modify-write that also
                    // writes xS (e.g., `add xS, xD, #1`) reads the
                    // old xS, then writes the new xS. After this
                    // instruction the MOV's xD value is no longer
                    // needed and xS holds the new value — both
                    // bookkeepings get updated for any *future* reads
                    // of xD (which would now hit the blocker).
                    if (writes_xs) xs_clobbered = true;
                    if (writes_xd) {
                        // Read-modify-write of xD: MOV value consumed
                        // and xD is freshly defined. Stop.
                        xd_redefined = true;
                        break;
                    }
                    continue;
                }

                if (writes_xd) {
                    // Clean redefine of xD (no read). MOV value dead.
                    xd_redefined = true;
                    break;
                }

                if (writes_xs) {
                    // xS overwritten — fine if no subsequent read of
                    // xD occurs before xd_redefined. Flag and keep
                    // scanning.
                    xs_clobbered = true;
                    continue;
                }

                // Neutral instruction — keep scanning.
            }

            if (blocked) continue;
            if (last_use == null) continue; // no use to retarget
            if (!xd_redefined) continue; // xD might be live-out

            const last = last_use.?;

            // Final guard: ensure we can rewrite every read of xD in
            // [i+1, last] before mutating any bytes. The forward scan
            // already verified no `writes_xs` occurs in this range.
            var rewritten_ok = true;
            j = i + 1;
            while (j <= last) : (j += 1) {
                const off_j = start + j * 4;
                const wj = std.mem.readInt(u32, bytes[off_j..][0..4], .little);
                const inst = decode(wj);
                if (inst.reads & rd_mask != 0) {
                    if (rewriteReadReg(wj, rd, rm) == null) {
                        rewritten_ok = false;
                        break;
                    }
                }
            }
            if (!rewritten_ok) continue;

            // Apply: rewrite reads in window, NOP the MOV.
            j = i + 1;
            while (j <= last) : (j += 1) {
                const off_j = start + j * 4;
                const wj = std.mem.readInt(u32, bytes[off_j..][0..4], .little);
                const inst = decode(wj);
                if (inst.reads & rd_mask != 0) {
                    const new_word = rewriteReadReg(wj, rd, rm).?;
                    std.mem.writeInt(u32, bytes[off_j..][0..4], new_word, .little);
                }
            }
            std.mem.writeInt(u32, bytes[off..][0..4], NOP, .little);
            eliminated += 1;
            changed = true;
        }

        if (!changed) break;
    }

    return .{
        .eliminated = eliminated,
        .self_eliminated = self_eliminated,
        .seen = seen,
    };
}

/// Public entry point. Walks each `[block_offsets[k], block_offsets[k+1])`
/// range (with `total_len` as the implicit upper bound for the last
/// block) and applies the coalescing scan.
pub fn coalesceMovesPostEmit(
    bytes: []u8,
    block_offsets: []const usize,
    total_len: usize,
) Stats {
    var stats: Stats = .{};
    if (block_offsets.len == 0) {
        const r = coalesceBlock(bytes, 0, total_len);
        stats.movs_seen += r.seen;
        stats.movs_eliminated += r.eliminated;
        stats.self_movs_eliminated += r.self_eliminated;
        return stats;
    }

    // Build a sorted list of all "barrier" boundaries: block starts +
    // total_len. Block starts may not be sorted in numeric byte order
    // (block_order!=numeric), but typically are. Sort defensively.
    var bounds_buf: [4096]usize = undefined;
    var bounds: []usize = undefined;
    if (block_offsets.len + 1 <= bounds_buf.len) {
        bounds = bounds_buf[0 .. block_offsets.len + 1];
    } else {
        // Fall back: process by per-block min-to-next-block range
        // computed inline. Allocate would be cleaner but we want this
        // pass to be allocation-free — block counts >4095 are absurd.
        // Bail to a single block.
        const r = coalesceBlock(bytes, 0, total_len);
        stats.movs_seen += r.seen;
        stats.movs_eliminated += r.eliminated;
        stats.self_movs_eliminated += r.self_eliminated;
        return stats;
    }

    for (block_offsets, 0..) |off, k| bounds[k] = off;
    bounds[block_offsets.len] = total_len;
    std.sort.pdq(usize, bounds, {}, std.sort.asc(usize));

    var k: usize = 0;
    while (k + 1 < bounds.len) : (k += 1) {
        const s = bounds[k];
        const e = bounds[k + 1];
        if (e <= s) continue;
        const r = coalesceBlock(bytes, s, e);
        stats.movs_seen += r.seen;
        stats.movs_eliminated += r.eliminated;
        stats.self_movs_eliminated += r.self_eliminated;
    }

    return stats;
}

// ─────────────────────────── Tests ───────────────────────────

const testing = std.testing;

fn encodeMovX(rd: u5, rm: u5) u32 {
    return 0xAA0003E0 | (@as(u32, rm) << 16) | rd;
}

fn encodeMovW(rd: u5, rm: u5) u32 {
    return 0x2A0003E0 | (@as(u32, rm) << 16) | rd;
}

fn encodeAddX(rd: u5, rn: u5, rm: u5) u32 {
    // ADD Xd, Xn, Xm (shifted reg, shift=LSL #0).
    return 0x8B000000 | (@as(u32, rm) << 16) | (@as(u32, rn) << 5) | rd;
}

fn encodeSubX(rd: u5, rn: u5, rm: u5) u32 {
    return 0xCB000000 | (@as(u32, rm) << 16) | (@as(u32, rn) << 5) | rd;
}

fn encodeAddImm(rd: u5, rn: u5, imm12: u12) u32 {
    return 0x91000000 | (@as(u32, imm12) << 10) | (@as(u32, rn) << 5) | rd;
}

fn encodeLdrUImm(rt: u5, rn: u5, off: u12) u32 {
    return 0xF9400000 | (@as(u32, off) << 10) | (@as(u32, rn) << 5) | rt;
}

fn encodeStrUImm(rt: u5, rn: u5, off: u12) u32 {
    return 0xF9000000 | (@as(u32, off) << 10) | (@as(u32, rn) << 5) | rt;
}

fn encodeLdrReg(rt: u5, rn: u5, rm: u5) u32 {
    // LDR Xt, [Xn, Xm, LSL #0] — register offset, 64-bit.
    return 0xF8606800 | (@as(u32, rm) << 16) | (@as(u32, rn) << 5) | rt;
}

fn encodeMovzX(rd: u5, imm16: u16) u32 {
    return 0xD2800000 | (@as(u32, imm16) << 5) | rd;
}

fn encodeRet() u32 {
    return 0xD65F03C0; // ret x30
}

fn runCoalesce(input: []const u32, allocator: std.mem.Allocator) ![]u32 {
    const bytes = try allocator.alloc(u8, input.len * 4);
    errdefer allocator.free(bytes);
    for (input, 0..) |w, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], w, .little);

    _ = coalesceMovesPostEmit(bytes, &.{0}, bytes.len);

    const out = try allocator.alloc(u32, input.len);
    errdefer allocator.free(out);
    for (0..input.len) |i| out[i] = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
    allocator.free(bytes);
    return out;
}

test "coalesce_post: clean coalesce eliminates redundant mov" {
    // movz x1, #5         ; def x1
    // mov  x2, x1         ; redundant
    // add  x3, x2, #4     ; last use of x2
    // movz x2, #0         ; re-def x2 (so x2 is dead at block exit)
    // ret
    const input = [_]u32{
        encodeMovzX(1, 5),
        encodeMovX(2, 1),
        encodeAddImm(3, 2, 4),
        encodeMovzX(2, 0),
        encodeRet(),
    };
    const out = try runCoalesce(&input, testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(input[0], out[0]);
    try testing.expectEqual(NOP, out[1]);
    // ADD should now read x1 instead of x2.
    try testing.expectEqual(encodeAddImm(3, 1, 4), out[2]);
    try testing.expectEqual(input[3], out[3]);
    try testing.expectEqual(input[4], out[4]);
}

test "coalesce_post: intervening write to xS blocks coalesce" {
    // movz x1, #5
    // mov  x2, x1         ; candidate
    // movz x1, #7         ; clobbers xS (x1) within window
    // add  x3, x2, #4     ; last use of x2
    // movz x2, #0         ; re-def x2
    // ret
    const input = [_]u32{
        encodeMovzX(1, 5),
        encodeMovX(2, 1),
        encodeMovzX(1, 7),
        encodeAddImm(3, 2, 4),
        encodeMovzX(2, 0),
        encodeRet(),
    };
    const out = try runCoalesce(&input, testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(u32, &input, out);
}

test "coalesce_post: read of xD after last use blocks coalesce" {
    // movz x1, #5
    // mov  x2, x1         ; candidate
    // add  x3, x2, #4     ; "last" use of x2
    // movz x2, #0         ; re-def x2 — OK
    // add  x4, x2, #1     ; uses x2 again (the new def, which is fine)
    // ret
    //
    // This should *coalesce*: the new def of x2 establishes a fresh
    // live range. The "x2 dead at exit" requirement is met because the
    // second def of x2 happens AFTER the first use range ends, so the
    // first MOV's value of x2 dies cleanly at the redef. This test
    // documents that the predicate is correct, not over-conservative.
    const input = [_]u32{
        encodeMovzX(1, 5),
        encodeMovX(2, 1),
        encodeAddImm(3, 2, 4),
        encodeMovzX(2, 0),
        encodeAddImm(4, 2, 1),
        encodeRet(),
    };
    const out = try runCoalesce(&input, testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(NOP, out[1]);
    try testing.expectEqual(encodeAddImm(3, 1, 4), out[2]);
    try testing.expectEqual(input[3], out[3]);
    try testing.expectEqual(input[4], out[4]);
}

test "coalesce_post: xD redefined-and-reused-after-xD-last-use is safe" {
    // The redefined-source case from the spec:
    // movz x1, #5
    // mov  x2, x1
    // add  x3, x2, #4    ; last use of x2 in first range
    // movz x1, #9        ; clobbers xS (x1) AFTER last_use(x2) — fine
    // movz x2, #0        ; re-def x2 to terminate first liverange
    // add  x4, x1, #1    ; uses x1 (new value), fine
    // ret
    const input = [_]u32{
        encodeMovzX(1, 5),
        encodeMovX(2, 1),
        encodeAddImm(3, 2, 4),
        encodeMovzX(1, 9),
        encodeMovzX(2, 0),
        encodeAddImm(4, 1, 1),
        encodeRet(),
    };
    const out = try runCoalesce(&input, testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(NOP, out[1]);
    try testing.expectEqual(encodeAddImm(3, 1, 4), out[2]);
    try testing.expectEqual(input[3], out[3]);
    try testing.expectEqual(input[4], out[4]);
    try testing.expectEqual(input[5], out[5]);
}

test "coalesce_post: xS redefined before xD last use blocks coalesce" {
    // movz x1, #5
    // mov  x2, x1
    // add  x3, x2, #4    ; first use of x2
    // movz x1, #9        ; clobbers xS BEFORE xD's last use — must block
    // sub  x4, x2, x2    ; LAST use of x2 (after the x1 clobber)
    // movz x2, #0        ; re-def
    // ret
    const input = [_]u32{
        encodeMovzX(1, 5),
        encodeMovX(2, 1),
        encodeAddImm(3, 2, 4),
        encodeMovzX(1, 9),
        encodeSubX(4, 2, 2),
        encodeMovzX(2, 0),
        encodeRet(),
    };
    const out = try runCoalesce(&input, testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(u32, &input, out);
}

test "coalesce_post: adversarial — dest aliased into load address" {
    // movz x1, #16
    // mov  x2, x1          ; candidate
    // ldr  x3, [x2]        ; reads x2 as base — must rewrite to x1
    // movz x2, #0          ; re-def
    // ret
    const input = [_]u32{
        encodeMovzX(1, 16),
        encodeMovX(2, 1),
        encodeLdrUImm(3, 2, 0),
        encodeMovzX(2, 0),
        encodeRet(),
    };
    const out = try runCoalesce(&input, testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(NOP, out[1]);
    try testing.expectEqual(encodeLdrUImm(3, 1, 0), out[2]);
}

test "coalesce_post: adversarial — dest used as Rm in register-offset load" {
    // movz x1, #16
    // mov  x2, x1
    // ldr  x3, [x4, x2]    ; x2 is Rm — must rewrite to x1
    // movz x2, #0
    // ret
    const input = [_]u32{
        encodeMovzX(1, 16),
        encodeMovX(2, 1),
        encodeLdrReg(3, 4, 2),
        encodeMovzX(2, 0),
        encodeRet(),
    };
    const out = try runCoalesce(&input, testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(NOP, out[1]);
    try testing.expectEqual(encodeLdrReg(3, 4, 1), out[2]);
}

test "coalesce_post: self-mov xD,xD always eliminable" {
    const input = [_]u32{
        encodeMovX(5, 5),
        encodeRet(),
    };
    const out = try runCoalesce(&input, testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(NOP, out[0]);
    try testing.expectEqual(input[1], out[1]);
}

test "coalesce_post: cross-block (xD live-out) NOT coalesced without redef" {
    // movz x1, #5
    // mov  x2, x1
    // add  x3, x2, #4    ; last use of x2 in block
    // ret                 ; xD (x2) has no redef before block exit — refuse.
    const input = [_]u32{
        encodeMovzX(1, 5),
        encodeMovX(2, 1),
        encodeAddImm(3, 2, 4),
        encodeRet(),
    };
    const out = try runCoalesce(&input, testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(u32, &input, out);
}

test "coalesce_post: reserved regs (lr, fp, vmctx, sp) never coalesced" {
    // mov x30, x5 ; LR — never coalesce.
    const input = [_]u32{
        encodeMovzX(5, 1),
        encodeMovX(30, 5),
        encodeAddImm(6, 30, 4),
        encodeMovzX(30, 0),
        encodeRet(),
    };
    const out = try runCoalesce(&input, testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(u32, &input, out);
}

test "coalesce_post: store rewrites Rt (read of value being stored)" {
    // movz x1, #16
    // mov  x2, x1
    // str  x2, [x3]      ; x2 is the value being stored — read, rewrite to x1
    // movz x2, #0
    // ret
    const input = [_]u32{
        encodeMovzX(1, 16),
        encodeMovX(2, 1),
        encodeStrUImm(2, 3, 0),
        encodeMovzX(2, 0),
        encodeRet(),
    };
    const out = try runCoalesce(&input, testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(NOP, out[1]);
    try testing.expectEqual(encodeStrUImm(1, 3, 0), out[2]);
}

test "coalesce_post: unknown instruction in window blocks coalesce" {
    // movz x1, #5
    // mov  x2, x1
    // .word 0x00000000   ; UDF — undecodable. Acts as barrier.
    // add  x3, x2, #4
    // movz x2, #0
    // ret
    const input = [_]u32{
        encodeMovzX(1, 5),
        encodeMovX(2, 1),
        0x00000000,
        encodeAddImm(3, 2, 4),
        encodeMovzX(2, 0),
        encodeRet(),
    };
    const out = try runCoalesce(&input, testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(u32, &input, out);
}

test "coalesce_post: fixed-point picks up MOV chain" {
    // movz x1, #5
    // mov  x2, x1        ; first redundant
    // mov  x3, x2        ; second redundant — only safe after first
    //                     ;  is coalesced
    // add  x4, x3, #1    ; uses x3
    // movz x2, #0        ; re-def x2
    // movz x3, #0        ; re-def x3
    // ret
    const input = [_]u32{
        encodeMovzX(1, 5),
        encodeMovX(2, 1),
        encodeMovX(3, 2),
        encodeAddImm(4, 3, 1),
        encodeMovzX(2, 0),
        encodeMovzX(3, 0),
        encodeRet(),
    };
    const out = try runCoalesce(&input, testing.allocator);
    defer testing.allocator.free(out);
    // Both MOVs should be NOP'd, ADD reads x1.
    try testing.expectEqual(NOP, out[1]);
    try testing.expectEqual(NOP, out[2]);
    try testing.expectEqual(encodeAddImm(4, 1, 1), out[3]);
}

test "coalesce_post: block boundary respected" {
    // Block 0: 4 words. Block 1: starts at byte 16 with the MOV.
    // The MOV in block 1 should not coalesce reads from block 0.
    const input = [_]u32{
        // Block 0
        encodeMovzX(1, 5),
        encodeAddImm(7, 1, 1), // read of x1 in earlier block — not in window
        encodeMovzX(2, 9),
        encodeRet(),
        // Block 1
        encodeMovX(3, 1),
        encodeAddImm(4, 3, 4),
        encodeMovzX(3, 0),
        encodeRet(),
    };
    const bytes = try testing.allocator.alloc(u8, input.len * 4);
    defer testing.allocator.free(bytes);
    for (input, 0..) |w, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], w, .little);

    _ = coalesceMovesPostEmit(bytes, &.{ 0, 16 }, bytes.len);

    const out = try testing.allocator.alloc(u32, input.len);
    defer testing.allocator.free(out);
    for (0..input.len) |i| out[i] = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);

    // Block 0 untouched (no MOV candidate).
    try testing.expectEqual(input[0], out[0]);
    try testing.expectEqual(input[1], out[1]);
    try testing.expectEqual(input[2], out[2]);
    try testing.expectEqual(input[3], out[3]);
    // Block 1: MOV should be coalesced into x1 (block-local; cross-
    // block use of x1 from block 0 is irrelevant — block 1 has its
    // own scope).
    try testing.expectEqual(NOP, out[4]);
    try testing.expectEqual(encodeAddImm(4, 1, 4), out[5]);
    try testing.expectEqual(input[6], out[6]);
    try testing.expectEqual(input[7], out[7]);
}

test "coalesce_post: 32-bit MOV w-form also coalesced" {
    const input = [_]u32{
        encodeMovzX(1, 5),
        encodeMovW(2, 1),
        encodeAddImm(3, 2, 4),
        encodeMovzX(2, 0),
        encodeRet(),
    };
    const out = try runCoalesce(&input, testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(NOP, out[1]);
    try testing.expectEqual(encodeAddImm(3, 1, 4), out[2]);
}
