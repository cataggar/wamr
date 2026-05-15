//! Alias-class keys for value-tracking IR passes.
//!
//! Centralised here so single-block (`forward_redundant_loads.zig`) and
//! dominator-aware (`forward_redundant_loads_dominator.zig`) load forwarders
//! share one definition. `LoadKey` is a tagged union covering both
//! wasm-memory loads (`.mem`) and wasm-local slots (`.local`, #467).

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

/// Tagged alias-class key for value-tracking passes that handle both
/// wasm linear-memory loads and wasm-local-slot reads in one table.
///
/// `mem` and `local` are disjoint alias classes:
///   * A linear-memory `store` invalidates only overlapping `.mem`
///     entries — never `.local`.
///   * A `local_set i, v` invalidates only `.local = i` — never `.mem`,
///     and never `.local = j` for `j != i` (wasm locals are not aliased
///     across distinct indices).
///   * Calls and other coarse barriers clear the entire table at the
///     pass level (conservative; wasm semantics technically allow
///     preserving `.local` across calls since callees cannot mutate
///     the caller's locals, but inter-procedural escape analysis
///     hasn't landed yet — see #467 PR for the trade-off).
pub const LoadKey = union(enum) {
    mem: MemKey,
    local: u32,
};

/// True iff a memory store may invalidate a value held under `key`.
/// `.local` entries are never invalidated by memory stores.
pub fn storeAliasesLoad(key: LoadKey, st: anytype) bool {
    return switch (key) {
        .mem => |m| storeAliases(m, st),
        .local => false,
    };
}
