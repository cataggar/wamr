//! AOT host-import trampoline pool.
//!
//! Each slot contains a tiny arch-specific shim that bakes in the slot index
//! and forwards the flat core ABI arguments to `genericDispatcher`.

const std = @import("std");
const builtin = @import("builtin");
const platform = @import("../../platform/platform.zig");
const core_types = @import("../common/types.zig");

const page_size = std.heap.page_size_min;
// Stubs forward up to 10 lowered C-ABI args (a0..a9) into `genericDispatcher`,
// so the dispatcher kinds can carry up to 9 wasm-level params (after dropping
// the importer's vmctx for the AOT-codegen variants). See #700 — widened from
// 9 (a0..a8) so `wasi_snapshot_preview1.path_open` (9 wasm params) stops
// trap-stubbing on the WASIp1→WASIp2 adapter path. Previously widened from
// 6 → 9 in #689 to lower WASIp2 filesystem methods like `link-at` (7 wasm
// params).
const x86_64_stub_bytes: usize = 63;
const aarch64_stub_bytes: usize = 84;

pub const STUB_BYTES: usize = switch (builtin.cpu.arch) {
    .x86_64 => x86_64_stub_bytes,
    .aarch64 => aarch64_stub_bytes,
    else => 1,
};
pub const MAX_SLOTS: u32 = 256;

const StubFn = *const fn () callconv(.c) void;

pub const LoweredSig = struct {
    param_types: []const core_types.ValType,
    result_types: []const core_types.ValType,
    has_retptr: bool = false,
};

pub const DispatchResult = extern struct {
    status: u32,
    value: u64,
};

extern fn wamrAotDispatchComponentTrampoline(
    ctx_opaque: *anyopaque,
    lowered_sig: *const LoweredSig,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    a7: u64,
    a8: u64,
    a9: u64,
) callconv(.c) DispatchResult;

/// AOT-codegen-flavoured wrapper around `wamrAotDispatchComponentTrampoline`
/// for canon.lower imports of an AOT core module. AOT-emitted host-import
/// call sites pass the importer's vmctx as the first arg, so `a0` here is
/// vmctx (which the host trampoline ignores) and `a1..a8` are the lowered
/// wasm args. (#687, widened in #689.)
extern fn wamrAotDispatchComponentTrampolineAot(
    ctx_opaque: *anyopaque,
    lowered_sig: *const LoweredSig,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    a7: u64,
    a8: u64,
    a9: u64,
) callconv(.c) DispatchResult;

/// Cross-instance core-to-core fn-import dispatcher (#662). Implemented in
/// `src/component/executor.zig`. The first slot (`a0`) is the importer's
/// vmctx, which the dispatcher ignores; `a1..a8` are the lowered wasm args
/// per the AOT codegen calling convention.
extern fn wamrAotDispatchCrossInstance(
    ctx_opaque: *anyopaque,
    lowered_sig: *const LoweredSig,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    a7: u64,
    a8: u64,
    a9: u64,
) callconv(.c) DispatchResult;

/// Canon-builtin dispatcher for AOT-compiled core modules whose imports
/// resolve through a sibling inline-export to a `canon.resource.{drop,new,rep}`
/// (or other canon-builtin) contributor. Same vmctx-as-first-arg convention
/// as the other `*Aot` dispatchers — `a0` is the importer's vmctx (ignored),
/// `a1` is the wasm-level handle / rep. Implemented in
/// `src/component/executor.zig`. (#701, follow-up to #687.)
extern fn wamrAotDispatchCanonBuiltin(
    ctx_opaque: *anyopaque,
    lowered_sig: *const LoweredSig,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    a7: u64,
    a8: u64,
    a9: u64,
) callconv(.c) DispatchResult;

/// Trap-on-call stub for non-WASI function imports that have no AOT-side
/// wiring yet (e.g. adapter-core canon.lower imports left over after #662
/// Phase C). Returns a failing `DispatchResult` so the caller traps with
/// a clean status rather than jumping through a null host slot.
extern fn wamrAotDispatchTrapStub(
    ctx_opaque: *anyopaque,
    lowered_sig: *const LoweredSig,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    a7: u64,
    a8: u64,
    a9: u64,
) callconv(.c) DispatchResult;

pub const DispatchKind = enum(u8) {
    canon_lower,
    canon_lower_aot,
    cross_instance,
    canon_builtin_aot,
    trap_stub,
};

pub const Slot = struct {
    component_inst: *anyopaque,
    canon_lower_idx: u32,
    ctx: ?*anyopaque = null,
    lowered_sig: LoweredSig = .{
        .param_types = &.{},
        .result_types = &.{},
        .has_retptr = false,
    },
    dispatch_kind: DispatchKind = .canon_lower,
};

var g_active_pool: ?*TrampolinePool = null;

pub fn setActivePool(pool: ?*TrampolinePool) void {
    g_active_pool = pool;
}

pub fn genericDispatcher(slot: u32, a0: u64, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64, a6: u64, a7: u64, a8: u64, a9: u64) callconv(.c) u64 {
    const pool = g_active_pool orelse return 0;
    if (slot >= pool.next_slot) return 0;

    const entry = pool.slots[slot];
    const ctx = entry.ctx orelse return 0;
    const dispatched = switch (entry.dispatch_kind) {
        .canon_lower => wamrAotDispatchComponentTrampoline(ctx, &entry.lowered_sig, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9),
        .canon_lower_aot => wamrAotDispatchComponentTrampolineAot(ctx, &entry.lowered_sig, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9),
        .cross_instance => wamrAotDispatchCrossInstance(ctx, &entry.lowered_sig, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9),
        .canon_builtin_aot => wamrAotDispatchCanonBuiltin(ctx, &entry.lowered_sig, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9),
        .trap_stub => wamrAotDispatchTrapStub(ctx, &entry.lowered_sig, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9),
    };
    if (dispatched.status != 0) return 0;
    return dispatched.value;
}

pub const TrampolinePool = struct {
    memory: []align(page_size) u8,
    slots: []Slot,
    next_slot: u32 = 0,

    // Whether RWX trampoline pages are supported on the build target.
    // Gated at comptime so the `std.posix.mmap` call below is not analysed
    // on Windows / unsupported arches — that previously forced an explicit
    // `linkLibC` on every binary reachable from `TrampolinePool.init`
    // (broke `wamr.exe` Windows build under #662 Phase C, where the lazy
    // pool ctor became reachable from the CLI).
    const supports_pool: bool = blk: {
        if (builtin.os.tag == .windows) break :blk false;
        if (builtin.cpu.arch != .x86_64 and builtin.cpu.arch != .aarch64) break :blk false;
        // macOS aarch64 forbids RWX mmap without MAP_JIT + pthread_jit_write_protect_np;
        // not worth wiring up for stdio-echo's needs. AOT layer falls back to interp.
        if (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64) break :blk false;
        break :blk true;
    };

    pub fn init(allocator: std.mem.Allocator) !TrampolinePool {
        if (comptime !supports_pool) return error.UnsupportedPlatform;

        const slots = try allocator.alloc(Slot, MAX_SLOTS);
        errdefer allocator.free(slots);

        const map_len = std.mem.alignForward(usize, @as(usize, MAX_SLOTS) * STUB_BYTES, page_size);
        const memory = try std.posix.mmap(
            null,
            map_len,
            .{ .READ = true, .WRITE = true, .EXEC = true },
            .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
            -1,
            0,
        );
        @memset(memory, 0);

        return .{
            .memory = memory,
            .slots = slots,
        };
    }

    pub fn allocSlotWithCtx(self: *TrampolinePool, ctx: *anyopaque, lowered_sig: LoweredSig) !StubFn {
        const slot = self.next_slot;
        if (slot >= MAX_SLOTS) return error.OutOfTrampolineSlots;

        self.slots[slot] = .{
            .component_inst = ctx,
            .canon_lower_idx = 0,
            .ctx = ctx,
            .lowered_sig = lowered_sig,
        };
        self.next_slot += 1;
        writeStub(self.stubBytes(slot), slot);
        platform.icacheFlush(self.stubPtr(slot), STUB_BYTES);
        g_active_pool = self;

        return @ptrFromInt(@intFromPtr(self.stubPtr(slot)));
    }

    pub fn allocSlot(self: *TrampolinePool, component_inst: *anyopaque, canon_lower_idx: u32) !StubFn {
        const slot = self.next_slot;
        if (slot >= MAX_SLOTS) return error.OutOfTrampolineSlots;

        self.slots[slot] = .{
            .component_inst = component_inst,
            .canon_lower_idx = canon_lower_idx,
        };
        self.next_slot += 1;
        writeStub(self.stubBytes(slot), slot);
        platform.icacheFlush(self.stubPtr(slot), STUB_BYTES);
        g_active_pool = self;

        return @ptrFromInt(@intFromPtr(self.stubPtr(slot)));
    }

    /// Allocate a slot for a canon-lower-backed cross-instance thunk where
    /// the *importer* is an AOT-compiled core module. Same shape as
    /// `allocSlotWithCtx` but routes through `wamrAotDispatchComponentTrampolineAot`
    /// so the dispatcher skips the AOT codegen's leading vmctx argument
    /// before treating the remaining registers as lowered wasm args. (#687.)
    pub fn allocCanonLowerAotSlot(self: *TrampolinePool, ctx: *anyopaque, lowered_sig: LoweredSig) !StubFn {
        const slot = self.next_slot;
        if (slot >= MAX_SLOTS) return error.OutOfTrampolineSlots;

        self.slots[slot] = .{
            .component_inst = ctx,
            .canon_lower_idx = 0,
            .ctx = ctx,
            .lowered_sig = lowered_sig,
            .dispatch_kind = .canon_lower_aot,
        };
        self.next_slot += 1;
        writeStub(self.stubBytes(slot), slot);
        platform.icacheFlush(self.stubPtr(slot), STUB_BYTES);
        g_active_pool = self;

        return @ptrFromInt(@intFromPtr(self.stubPtr(slot)));
    }

    /// Allocate a slot for a cross-instance core-to-core fn-import thunk
    /// (#662). The stub forwards the importer's vmctx + lowered wasm args
    /// to `wamrAotDispatchCrossInstance`, which re-issues the call into
    /// the sibling AotInstance via `aot_runtime.callFuncScalar`.
    pub fn allocCrossInstanceSlot(self: *TrampolinePool, ctx: *anyopaque, lowered_sig: LoweredSig) !StubFn {
        const slot = self.next_slot;
        if (slot >= MAX_SLOTS) return error.OutOfTrampolineSlots;

        self.slots[slot] = .{
            .component_inst = ctx,
            .canon_lower_idx = 0,
            .ctx = ctx,
            .lowered_sig = lowered_sig,
            .dispatch_kind = .cross_instance,
        };
        self.next_slot += 1;
        writeStub(self.stubBytes(slot), slot);
        platform.icacheFlush(self.stubPtr(slot), STUB_BYTES);
        g_active_pool = self;

        return @ptrFromInt(@intFromPtr(self.stubPtr(slot)));
    }

    /// Allocate a slot for a canon-builtin (`resource.{drop,new,rep}`, etc.)
    /// imported by an AOT-compiled core module. Routes through
    /// `wamrAotDispatchCanonBuiltin`, which switches on the `CanonBuiltinTrampolineCtx`
    /// to invoke the right pure helper against the component's resource
    /// table. (#701, follow-up to #687.)
    pub fn allocCanonBuiltinAotSlot(self: *TrampolinePool, ctx: *anyopaque, lowered_sig: LoweredSig) !StubFn {
        const slot = self.next_slot;
        if (slot >= MAX_SLOTS) return error.OutOfTrampolineSlots;

        self.slots[slot] = .{
            .component_inst = ctx,
            .canon_lower_idx = 0,
            .ctx = ctx,
            .lowered_sig = lowered_sig,
            .dispatch_kind = .canon_builtin_aot,
        };
        self.next_slot += 1;
        writeStub(self.stubBytes(slot), slot);
        platform.icacheFlush(self.stubPtr(slot), STUB_BYTES);
        g_active_pool = self;

        return @ptrFromInt(@intFromPtr(self.stubPtr(slot)));
    }

    /// Allocate a slot for a "trap on call" stub. The instantiation flow
    /// installs these for non-WASI fn imports that have no AOT-side wiring
    /// (canon.lower / canon-builtin) yet — instantiation succeeds, but any
    /// actual call traps with a clean status (#662 follow-up).
    pub fn allocTrapSlot(self: *TrampolinePool, ctx: *anyopaque) !StubFn {
        const slot = self.next_slot;
        if (slot >= MAX_SLOTS) return error.OutOfTrampolineSlots;

        self.slots[slot] = .{
            .component_inst = ctx,
            .canon_lower_idx = 0,
            .ctx = ctx,
            .dispatch_kind = .trap_stub,
        };
        self.next_slot += 1;
        writeStub(self.stubBytes(slot), slot);
        platform.icacheFlush(self.stubPtr(slot), STUB_BYTES);
        g_active_pool = self;

        return @ptrFromInt(@intFromPtr(self.stubPtr(slot)));
    }

    pub fn deinit(self: *TrampolinePool, allocator: std.mem.Allocator) void {
        if (comptime !supports_pool) {
            allocator.free(self.slots);
            self.* = undefined;
            return;
        }
        if (g_active_pool == self) g_active_pool = null;
        std.posix.munmap(self.memory);
        allocator.free(self.slots);
        self.* = undefined;
    }

    fn stubPtr(self: *TrampolinePool, slot: u32) [*]u8 {
        return self.memory.ptr + (@as(usize, slot) * STUB_BYTES);
    }

    fn stubBytes(self: *TrampolinePool, slot: u32) []u8 {
        const start = @as(usize, slot) * STUB_BYTES;
        return self.memory[start .. start + STUB_BYTES];
    }
};

fn writeStub(bytes: []u8, slot: u32) void {
    @memset(bytes, 0);
    switch (builtin.cpu.arch) {
        .x86_64 => encodeX8664Stub(bytes, slot, dispatcherAddr()),
        .aarch64 => encodeAarch64Stub(bytes, slot, dispatcherAddr()),
        else => unreachable,
    }
}

fn dispatcherAddr() usize {
    return @intFromPtr(&genericDispatcher);
}

fn encodeX8664Stub(bytes: []u8, slot: u32, dispatcher: usize) void {
    std.debug.assert(bytes.len >= STUB_BYTES);

    // Caller passes up to 10 args (a0..a9): a0..a5 in regs (rdi, rsi, rdx,
    // rcx, r8, r9), a6/a7/a8/a9 on stack at [rsp+8/+16/+24/+32]. We inject
    // `slot` as the new first arg and forward a0..a9, so the dispatcher
    // receives 11 args (slot + a0..a9): rdi=slot, rsi/rdx/rcx/r8/r9=a0..a4,
    // and a5..a9 on stack at [rsp+8..+40] after `call rax` pushes the return
    // address. We pre-push caller_a9/_a8/_a7/_a6 (read out of the caller's
    // outgoing stack-arg region) and r9 (caller_a5) to land them at the
    // right offsets.
    //
    // SysV alignment (#691, #700): at stub entry rsp%16==8 (caller's pre-call
    // rsp was 16-aligned, `call` pushed 8). The 5 pushes below shift rsp
    // by -40 → rsp%16==0 before `call rax`, which pushes another 8 → the
    // dispatcher enters with rsp%16==8, satisfying SysV's
    // `(rsp+8)%16==0` invariant. No extra align pad needed — the previous
    // 4-push (a6..a8) layout in #689 did need a `sub rsp, 8` pad; widening
    // to 5 pushes (a6..a9) in #700 naturally re-aligns.
    //
    // Prologue (26 bytes): read caller stack args into our frame, then
    // push r9. Reads always come from [rsp+0x20] because each push shifts
    // the remaining caller args up by 8 (caller_a9 was at [rsp+32]; after
    // pushing it, caller_a8 is at [rsp+32]; and so on).
    const prologue_stack = [_]u8{
        0x48, 0x8B, 0x44, 0x24, 0x20, // mov rax, [rsp+32]  (caller_a9)
        0x50, // push rax
        0x48, 0x8B, 0x44, 0x24, 0x20, // mov rax, [rsp+32]  (caller_a8, shifted)
        0x50, // push rax
        0x48, 0x8B, 0x44, 0x24, 0x20, // mov rax, [rsp+32]  (caller_a7, shifted)
        0x50, // push rax
        0x48, 0x8B, 0x44, 0x24, 0x20, // mov rax, [rsp+32]  (caller_a6, shifted)
        0x50, // push rax
        0x41, 0x51, // push r9             (caller_a5)
    };
    // Register-shift + slot-inject (20 bytes), unchanged from the 6-arg
    // version: r9..rsi each receive their lower neighbour, edi loads slot.
    const shift = [_]u8{
        0x4D, 0x89, 0xC1, // mov r9, r8
        0x49, 0x89, 0xC8, // mov r8, rcx
        0x48, 0x89, 0xD1, // mov rcx, rdx
        0x48, 0x89, 0xF2, // mov rdx, rsi
        0x48, 0x89, 0xFE, // mov rsi, rdi
        0xBF, // mov edi, imm32 (slot)
    };
    const movabs = [_]u8{ 0x48, 0xB8 };
    // Epilogue (7 bytes): call dispatcher, drop our 40-byte spill, return.
    const epilogue = [_]u8{
        0xFF, 0xD0, // call rax
        0x48, 0x83, 0xC4, 0x28, // add rsp, 40
        0xC3, // ret
    };

    var cursor: usize = 0;
    @memcpy(bytes[cursor .. cursor + prologue_stack.len], &prologue_stack);
    cursor += prologue_stack.len;
    @memcpy(bytes[cursor .. cursor + shift.len], &shift);
    cursor += shift.len;
    writeIntLittle(u32, bytes[cursor .. cursor + 4], slot);
    cursor += 4;
    @memcpy(bytes[cursor .. cursor + movabs.len], &movabs);
    cursor += movabs.len;
    writeIntLittle(u64, bytes[cursor .. cursor + 8], @intCast(dispatcher));
    cursor += 8;
    @memcpy(bytes[cursor .. cursor + epilogue.len], &epilogue);
    cursor += epilogue.len;

    std.debug.assert(cursor == x86_64_stub_bytes);
}

fn encodeAarch64Stub(bytes: []u8, slot: u32, dispatcher: usize) void {
    std.debug.assert(bytes.len >= STUB_BYTES);

    // AAPCS64: a0..a7 in x0..x7, a8 on stack at [sp+0], a9 at [sp+8]. We
    // inject `slot` as x0 and forward a0..a9 to the dispatcher, so
    // dispatcher sees (slot, a0..a9) = 11 args: x0=slot, x1..x7=a0..a6,
    // [sp+0]=a7, [sp+8]=a8, [sp+16]=a9.
    //
    // Unlike the 6-arg version we can no longer tail-call: we own a stack
    // frame for the dispatcher's stack args + a saved LR. So we use
    // `blr x16` and restore LR/SP on return.
    var cursor: usize = 0;

    // str x30, [sp, #-16]!  -- save LR, sp -= 16. Keeps sp 16-byte aligned.
    emitAarch64(bytes, &cursor, 0xF81F0FFE);
    // ldr x9,  [sp, #16]    -- x9  = caller_a8 (was at [sp+0] pre-push).
    emitAarch64(bytes, &cursor, 0xF9400BE9);
    // ldr x10, [sp, #24]    -- x10 = caller_a9 (was at [sp+8] pre-push).
    emitAarch64(bytes, &cursor, 0xF9400FEA);
    // str x10, [sp, #-16]!  -- push caller_a9; sp -= 16, [sp+0]=a9.
    // (The upper 8 bytes are padding; the dispatcher won't read them.)
    emitAarch64(bytes, &cursor, 0xF81F0FEA);
    // stp x7, x9, [sp, #-16]!  -- push caller_a7 + caller_a8; sp -= 16.
    // After this: [sp+0]=a7, [sp+8]=a8, [sp+16]=a9 (the dispatcher's stack
    // args).
    emitAarch64(bytes, &cursor, 0xA9BF27E7);

    // Shift x0..x7 right by one register (x7=x6, x6=x5, ..., x1=x0).
    inline for ([_]struct { dst: u5, src: u5 }{
        .{ .dst = 7, .src = 6 },
        .{ .dst = 6, .src = 5 },
        .{ .dst = 5, .src = 4 },
        .{ .dst = 4, .src = 3 },
        .{ .dst = 3, .src = 2 },
        .{ .dst = 2, .src = 1 },
        .{ .dst = 1, .src = 0 },
    }) |move| {
        emitAarch64(bytes, &cursor, 0xAA0003E0 | (@as(u32, move.src) << 16) | move.dst);
    }

    // movz x0, #slot_low (slot fits in a single u16 since MAX_SLOTS=256).
    emitAarch64(bytes, &cursor, movz32(0, @truncate(slot), 0));

    // movz/movk x16 = dispatcher address.
    const addr = @as(u64, @intCast(dispatcher));
    emitAarch64(bytes, &cursor, movz64(16, @truncate(addr), 0));
    emitAarch64(bytes, &cursor, movk64(16, @truncate(addr >> 16), 1));
    emitAarch64(bytes, &cursor, movk64(16, @truncate(addr >> 32), 2));
    emitAarch64(bytes, &cursor, movk64(16, @truncate(addr >> 48), 3));

    // blr x16                 -- call dispatcher.
    emitAarch64(bytes, &cursor, 0xD63F0200);
    // add sp, sp, #32         -- pop the two 16-byte pushes (a7+a8 + a9+pad).
    emitAarch64(bytes, &cursor, 0x910083FF);
    // ldr x30, [sp], #16      -- restore LR with post-increment.
    emitAarch64(bytes, &cursor, 0xF84107FE);
    // ret                     -- branch to LR.
    emitAarch64(bytes, &cursor, 0xD65F03C0);

    std.debug.assert(cursor == aarch64_stub_bytes);
}

fn emitAarch64(bytes: []u8, cursor: *usize, word: u32) void {
    writeIntLittle(u32, bytes[cursor.* .. cursor.* + 4], word);
    cursor.* += 4;
}

fn movz32(rd: u5, imm16: u16, shift: u2) u32 {
    return 0x52800000 | (@as(u32, shift) << 21) | (@as(u32, imm16) << 5) | rd;
}

fn movz64(rd: u5, imm16: u16, shift: u2) u32 {
    return 0xD2800000 | (@as(u32, shift) << 21) | (@as(u32, imm16) << 5) | rd;
}

fn movk64(rd: u5, imm16: u16, shift: u2) u32 {
    return 0xF2800000 | (@as(u32, shift) << 21) | (@as(u32, imm16) << 5) | rd;
}

fn writeIntLittle(comptime T: type, bytes: []u8, value: T) void {
    std.mem.writeInt(T, bytes[0..@sizeOf(T)], value, .little);
}

test "#648 phase 2: x86_64 trampoline encoder emits slot and dispatcher immediates" {
    var bytes: [aarch64_stub_bytes]u8 = undefined;
    @memset(&bytes, 0);

    encodeX8664Stub(&bytes, 0x11223344, 0x1122334455667788);

    // Stub layout (#700): pre-spill caller_a9/_a8/_a7/_a6 from [rsp+32],
    // then push r9 (caller_a5); shift rdi..r9; mov edi, slot;
    // movabs rax, dispatcher; call rax; add rsp, 40; ret. The 5 pushes
    // self-align SysV (8 - 40 ≡ 0 mod 16) so no `sub rsp, 8` pad is needed.
    const expected = [_]u8{
        0x48, 0x8B, 0x44, 0x24, 0x20, 0x50,
        0x48, 0x8B, 0x44, 0x24, 0x20, 0x50,
        0x48, 0x8B, 0x44, 0x24, 0x20, 0x50,
        0x48, 0x8B, 0x44, 0x24, 0x20, 0x50,
        0x41, 0x51,
        0x4D, 0x89, 0xC1,
        0x49, 0x89, 0xC8,
        0x48, 0x89, 0xD1,
        0x48, 0x89, 0xF2,
        0x48, 0x89, 0xFE,
        0xBF, 0x44, 0x33, 0x22, 0x11,
        0x48, 0xB8,
        0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
        0xFF, 0xD0,
        0x48, 0x83, 0xC4, 0x28,
        0xC3,
    };

    try std.testing.expectEqual(@as(usize, x86_64_stub_bytes), expected.len);
    try std.testing.expectEqualSlices(u8, &expected, bytes[0..expected.len]);
    for (bytes[expected.len..]) |byte| {
        try std.testing.expectEqual(@as(u8, 0), byte);
    }
}

test "#648 phase 2: aarch64 trampoline encoder emits slot and dispatcher immediates" {
    var bytes: [aarch64_stub_bytes]u8 = undefined;
    @memset(&bytes, 0);

    encodeAarch64Stub(&bytes, 0x1234, 0x1122334455667788);

    // Stub layout (#700): save LR, read caller_a8/a9 into x9/x10, push
    // caller_a9 (with pad), push x7+caller_a8; shift x0..x7;
    // movz x0,slot; movz/movk x16,dispatcher; blr x16; pop the two 16-byte
    // pushes (add sp, sp, #32); restore LR; ret.
    const expected = [_]u32{
        0xF81F0FFE, // str x30, [sp, #-16]!
        0xF9400BE9, // ldr x9,  [sp, #16]
        0xF9400FEA, // ldr x10, [sp, #24]
        0xF81F0FEA, // str x10, [sp, #-16]!
        0xA9BF27E7, // stp x7, x9, [sp, #-16]!
        0xAA0603E7, // mov x7, x6
        0xAA0503E6, // mov x6, x5
        0xAA0403E5, // mov x5, x4
        0xAA0303E4, // mov x4, x3
        0xAA0203E3, // mov x3, x2
        0xAA0103E2, // mov x2, x1
        0xAA0003E1, // mov x1, x0
        0x52824680, // movz w0, #0x1234
        0xD28EF110, // movz x16, #0x7788
        0xF2AAACD0, // movk x16, #0x5566, lsl 16
        0xF2C66890, // movk x16, #0x3344, lsl 32
        0xF2E22450, // movk x16, #0x1122, lsl 48
        0xD63F0200, // blr x16
        0x910083FF, // add sp, sp, #32
        0xF84107FE, // ldr x30, [sp], #16
        0xD65F03C0, // ret
    };

    try std.testing.expectEqual(@as(usize, aarch64_stub_bytes), expected.len * @sizeOf(u32));
    for (expected, 0..) |word, idx| {
        const start = idx * @sizeOf(u32);
        try std.testing.expectEqual(word, std.mem.readInt(u32, bytes[start..][0..4], .little));
    }
}

// Test scaffolding for the #691 SysV alignment regression test (below). A
// naked recorder lets us capture rsp at dispatcher entry with zero prologue,
// so the assertion checks the actual call-site alignment rather than
// whatever offset Zig's prologue might introduce.
var alignment_captured_rsp: usize = 0;

fn alignmentRecorder() callconv(.naked) void {
    asm volatile (
        \\ movq %%rsp, %[dst]
        \\ xorl %%eax, %%eax
        \\ retq
        : [dst] "=m" (alignment_captured_rsp),
    );
}

test "#691: x86_64 stub keeps SysV 16-byte stack alignment at dispatcher entry" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    if (builtin.os.tag == .windows or builtin.os.tag == .macos) return error.SkipZigTest;

    // Allocate one RWX page for a hand-written stub that targets
    // `alignmentRecorder` (instead of `genericDispatcher`). The recorder
    // captures rsp at entry and returns; we then assert SysV's
    // `(rsp+8) % 16 == 0` invariant — equivalent to `rsp % 16 == 8` —
    // holds at dispatcher entry. Before #691 the four-push prologue
    // introduced by #690 left rsp at 0 mod 16 at the dispatcher, which
    // faulted the first 16-aligned op inside `callFuncScalar` (e.g.
    // `SmpAllocator.alloc` via vtable) on cross-instance calls.
    const memory = try std.posix.mmap(
        null,
        page_size,
        .{ .READ = true, .WRITE = true, .EXEC = true },
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
        -1,
        0,
    );
    defer std.posix.munmap(memory);
    @memset(memory, 0);

    encodeX8664Stub(memory[0..STUB_BYTES], 0, @intFromPtr(&alignmentRecorder));

    alignment_captured_rsp = 0;
    const Stub = *const fn (u64, u64, u64, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
    const stub: Stub = @ptrCast(memory.ptr);
    _ = stub(1, 2, 3, 4, 5, 6, 7, 8, 9);

    try std.testing.expect(alignment_captured_rsp != 0);
    try std.testing.expectEqual(@as(usize, 8), alignment_captured_rsp & 0xF);
}
