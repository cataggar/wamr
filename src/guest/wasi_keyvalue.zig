//! `wasi_keyvalue` — guest-side helper for the `wasi:keyvalue/store`
//! interface (pinned to `@0.2.0-draft2`, the version wamr's host adapter
//! and `wasmtime serve -S keyvalue` both implement).
//!
//! Like the sibling `wasi_http`, this hand-writes the canonical-ABI
//! `extern` declarations for the host imports and exposes a small typed
//! Zig API. The shared `cabi_realloc` scratch arena + ret-area live in
//! the `abi` module, so a guest may combine this with `wasi_http` in one
//! component (e.g. an HTTP handler that persists state in a bucket)
//! without duplicate `cabi_realloc` exports.
//!
//! Why a host store instead of guest globals: a component's linear
//! memory does not necessarily persist across requests. wamr's serial
//! serve loop reuses one instance (so module-level globals survive), but
//! `wasmtime serve` instantiates a **fresh instance per request** — its
//! memory, and any guest globals, reset every time. State that must
//! survive across requests therefore has to live in the host, which is
//! exactly what `wasi:keyvalue` provides. The store is durable on wamr
//! via `wamr run --keyvalue-store=<path>` and on wasmtime via
//! `wasmtime serve -S keyvalue` (in-memory for the process lifetime).
//!
//! ## Usage
//!
//! ```zig
//! const kv = @import("wasi_keyvalue");
//!
//! const bucket = kv.open("") orelse return; // "" = default bucket
//! _ = bucket.set("pet:1", "{\"name\":\"Fluffy\"}");
//! if (bucket.get("pet:1")) |bytes| { ... }
//! ```
//!
//! Returned `get` slices borrow the scratch arena and stay valid until
//! the next `abi.resetScratch` (i.e. for the duration of the request).
//!
//! Bucket handles are not explicitly dropped: `[resource-drop]` is a
//! canonical built-in (not a host import) and runtimes differ on
//! exposing it, so — like the `wasi_http` resources — handles are left
//! for the runtime to reclaim at instance teardown. Open the bucket once
//! per instance (cache the handle) rather than per request to avoid
//! accumulating handles on runtimes that reuse one instance.

const abi = @import("abi");

const retPtr = abi.retPtr;
const retWords = abi.retWords;

// ── Host imports (canonical-ABI lowered signatures) ────────────────
//
// `wasi:keyvalue/store@0.2.0-draft2`. The `error` variant
// (`no-such-store | access-denied | other(string)`) flattens to 3 i32s,
// so every `result<…, error>` here is ≥ 4 flat words and spills through
// the trailing `retptr`. Grouped in a `raw` namespace so the import
// symbol `open` doesn't collide with the public `open` wrapper.

const raw = struct {
    /// `open: func(identifier: string) -> result<own<bucket>, error>`.
    /// retptr → [outer_disc, bucket_handle | err_disc, …].
    extern "wasi:keyvalue/store@0.2.0-draft2" fn open(ident_ptr: i32, ident_len: i32, retptr: i32) void;

    /// `[method]bucket.get: (borrow<bucket>, key: string)
    ///   -> result<option<list<u8>>, error>`.
    /// retptr → [outer_disc, opt_disc, ptr, len].
    extern "wasi:keyvalue/store@0.2.0-draft2" fn @"[method]bucket.get"(self: i32, key_ptr: i32, key_len: i32, retptr: i32) void;

    /// `[method]bucket.set: (borrow<bucket>, key: string, value: list<u8>)
    ///   -> result<_, error>`. retptr → [outer_disc, …err].
    extern "wasi:keyvalue/store@0.2.0-draft2" fn @"[method]bucket.set"(
        self: i32,
        key_ptr: i32,
        key_len: i32,
        value_ptr: i32,
        value_len: i32,
        retptr: i32,
    ) void;

    /// `[method]bucket.delete: (borrow<bucket>, key: string)
    ///   -> result<_, error>`. retptr → [outer_disc, …err].
    extern "wasi:keyvalue/store@0.2.0-draft2" fn @"[method]bucket.delete"(self: i32, key_ptr: i32, key_len: i32, retptr: i32) void;

    /// `[method]bucket.exists: (borrow<bucket>, key: string)
    ///   -> result<bool, error>`. retptr → [outer_disc, bool, …].
    extern "wasi:keyvalue/store@0.2.0-draft2" fn @"[method]bucket.exists"(self: i32, key_ptr: i32, key_len: i32, retptr: i32) void;
};

// ── Public API ──────────────────────────────────────────────────────

/// An open key-value bucket handle. Errors (`no-such-store`,
/// `access-denied`, `other`) are collapsed to `null` / `false` here —
/// the example-grade API trades the error detail for brevity; a
/// production wrapper could surface the `error` variant instead.
pub const Bucket = struct {
    handle: i32,

    /// `bucket.get`. Returns the stored bytes (borrowing the scratch
    /// arena, valid until the next `abi.resetScratch`), or null for a
    /// missing key or an error.
    pub fn get(self: Bucket, key: []const u8) ?[]const u8 {
        raw.@"[method]bucket.get"(self.handle, bytesPtr(key), @intCast(key.len), retPtr());
        const w = retWords();
        if (w[0] != 0) return null; // err arm
        if (w[1] != 1) return null; // none
        const p: [*]const u8 = @ptrFromInt(w[2]);
        return p[0..w[3]];
    }

    /// `bucket.set`. Returns true on the ok arm.
    pub fn set(self: Bucket, key: []const u8, value: []const u8) bool {
        raw.@"[method]bucket.set"(
            self.handle,
            bytesPtr(key),
            @intCast(key.len),
            bytesPtr(value),
            @intCast(value.len),
            retPtr(),
        );
        return retWords()[0] == 0;
    }

    /// `bucket.delete`. Returns true on the ok arm (deleting a missing
    /// key is also ok per the WIT contract).
    pub fn delete(self: Bucket, key: []const u8) bool {
        raw.@"[method]bucket.delete"(self.handle, bytesPtr(key), @intCast(key.len), retPtr());
        return retWords()[0] == 0;
    }

    /// `bucket.exists`. Returns true only when the key is present (an
    /// error arm reports false).
    pub fn exists(self: Bucket, key: []const u8) bool {
        raw.@"[method]bucket.exists"(self.handle, bytesPtr(key), @intCast(key.len), retPtr());
        const w = retWords();
        return w[0] == 0 and w[1] != 0;
    }
};

/// `store.open`. Opens (creating on first access) the bucket named
/// `identifier`; `""` selects the default bucket. Returns null on the
/// error arm.
pub fn open(identifier: []const u8) ?Bucket {
    raw.open(bytesPtr(identifier), @intCast(identifier.len), retPtr());
    const handle = abi.readResultHandle() orelse return null;
    return .{ .handle = handle };
}

inline fn bytesPtr(s: []const u8) i32 {
    return @intCast(@intFromPtr(s.ptr));
}
