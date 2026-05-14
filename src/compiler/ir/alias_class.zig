//! Alias-class keys for value-tracking IR passes.
//!
//! Centralised here so single-block (`forward_redundant_loads.zig`) and
//! dominator-aware (`forward_redundant_loads_dominator.zig`) load forwarders
//! share one definition. The union form leaves room for Agent B's
//! wasm-local-slot aliasing work (#467) — add a `local: u32` variant when
//! that lands. Today only `mem` is populated.

const ir = @import("ir.zig");

/// Identifies a linear-memory load location for redundant-load tracking.
/// Two loads with equal `MemKey`s and no intervening aliasing store, call,
/// or barrier yield the same value.
pub const MemKey = struct {
    base: ir.VReg,
    offset: u32,
    size: u8,
    sign_extend: bool,
};

/// Build a `MemKey` from an `Op.load` payload.
pub fn memKeyFromLoad(ld: anytype) MemKey {
    return .{
        .base = ld.base,
        .offset = ld.offset,
        .size = ld.size,
        .sign_extend = ld.sign_extend,
    };
}

/// True iff a store at `(st.base, st.offset, st.size)` may overlap a load
/// keyed at `key`. Conservative: same base register, byte ranges overlap.
pub fn storeAliases(key: MemKey, st: anytype) bool {
    if (key.base != st.base) return false;
    const key_end: u64 = @as(u64, key.offset) + @as(u64, key.size);
    const st_end: u64 = @as(u64, st.offset) + @as(u64, st.size);
    return !(key_end <= st.offset or st_end <= key.offset);
}
