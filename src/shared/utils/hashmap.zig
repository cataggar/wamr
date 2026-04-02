//! Hash map utilities for WAMR (replaces `bh_hashmap.h`).
//!
//! The C version implements a custom chained hash map with optional locking,
//! user-supplied hash/equal/destroy callbacks, and `void *` key/value pairs.
//! In Zig the standard library already provides high-quality hash maps, so
//! this module simply re-exports the relevant types and adds WAMR-specific
//! convenience aliases.

const std = @import("std");

/// A general-purpose hash map for the WAMR runtime.
/// Wraps `std.AutoHashMap` with the allocator pattern used throughout the
/// codebase.  Keys are hashed automatically via Zig's `autoHash`.
pub fn HashMap(comptime K: type, comptime V: type) type {
    return std.AutoHashMap(K, V);
}

/// String-keyed hash map — common in WAMR for symbol / export lookup.
/// Uses `[]const u8` keys with the standard string hash & eql functions.
pub fn StringHashMap(comptime V: type) type {
    return std.StringHashMap(V);
}

// ── Tests ───────────────────────────────────────────────────────────────

test "HashMap: basic insert and lookup" {
    var map = HashMap(u32, []const u8).init(std.testing.allocator);
    defer map.deinit();

    try map.put(1, "one");
    try map.put(2, "two");
    try map.put(3, "three");

    try std.testing.expectEqualStrings("one", map.get(1).?);
    try std.testing.expectEqualStrings("two", map.get(2).?);
    try std.testing.expectEqualStrings("three", map.get(3).?);
    try std.testing.expectEqual(@as(?[]const u8, null), map.get(99));
}

test "HashMap: remove" {
    var map = HashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    try map.put(10, 100);
    try std.testing.expect(map.remove(10));
    try std.testing.expectEqual(@as(?u32, null), map.get(10));
}

test "StringHashMap: insert and lookup" {
    var map = StringHashMap(i64).init(std.testing.allocator);
    defer map.deinit();

    try map.put("hello", 42);
    try map.put("world", -7);

    try std.testing.expectEqual(@as(i64, 42), map.get("hello").?);
    try std.testing.expectEqual(@as(i64, -7), map.get("world").?);
    try std.testing.expectEqual(@as(?i64, null), map.get("missing"));
}

test "StringHashMap: overwrite existing key" {
    var map = StringHashMap(u8).init(std.testing.allocator);
    defer map.deinit();

    try map.put("key", 1);
    try map.put("key", 2);
    try std.testing.expectEqual(@as(u8, 2), map.get("key").?);
}

test "HashMap: count" {
    var map = HashMap(u8, u8).init(std.testing.allocator);
    defer map.deinit();

    try map.put(1, 10);
    try map.put(2, 20);
    try std.testing.expectEqual(@as(u32, 2), map.count());
}
