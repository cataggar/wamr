//! Crash-safe, metadata-preserving persistence layer for `wasi:keyvalue` (#972 A6 & E4).
//!
//! Uses a write-ahead redo journal (`<path>.journal`) paired with an in-place
//! overwrite of `<path>` (`createFile` with `.truncate = false`,
//! `writePositionalAll` at offset 0, `setEndPos`, `sync`) so `<path>`'s inode,
//! permissions, ownership, ACLs, xattrs, and hardlinks are preserved across
//! flushes and crash recovery.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const JOURNAL_HEADER_MAGIC = "WAMRKVJ1";
pub const JOURNAL_COMMIT_MAGIC = "WAMRKVJC";
pub const JOURNAL_VERSION: u32 = 1;
pub const JOURNAL_HEADER_LEN: usize = 8 + 4 + 8 + Sha256.digest_length; // 52 bytes
pub const JOURNAL_TRAILER_LEN: usize = 8;
pub const JOURNAL_OVERHEAD_LEN: usize = JOURNAL_HEADER_LEN + JOURNAL_TRAILER_LEN; // 60 bytes
pub const MAX_SNAPSHOT_BYTES: usize = 8 * 1024 * 1024;

/// `wasi:keyvalue` persistence-layer bucket snapshot.
///
/// Mirrors one bucket in the JSON file at `WasiCliAdapter.keyvalue_store_path` —
/// each entry holds the bucket's key/value pairs as adapter-allocator-owned slices.
pub const PersistedBucket = struct {
    entries: std.StringHashMapUnmanaged([]const u8) = .empty,

    pub fn deinit(self: *PersistedBucket, allocator: Allocator) void {
        var it = self.entries.iterator();
        while (it.next()) |e| {
            allocator.free(e.key_ptr.*);
            allocator.free(e.value_ptr.*);
        }
        self.entries.deinit(allocator);
    }
};

pub const PersistedStoreMap = std.StringHashMapUnmanaged(*PersistedBucket);

pub fn deinitPersistedStoreMap(map: *PersistedStoreMap, allocator: Allocator) void {
    var it = map.iterator();
    while (it.next()) |kv| {
        allocator.free(kv.key_ptr.*);
        kv.value_ptr.*.deinit(allocator);
        allocator.destroy(kv.value_ptr.*);
    }
    map.deinit(allocator);
    map.* = .empty;
}

/// Lookup or lazily create a `PersistedBucket` for `identifier` in `map`.
pub fn getOrCreateBucket(
    map: *PersistedStoreMap,
    allocator: Allocator,
    identifier: []const u8,
) !*PersistedBucket {
    const gop = try map.getOrPut(allocator, identifier);
    if (!gop.found_existing) {
        const key_dup = allocator.dupe(u8, identifier) catch |err| {
            _ = map.remove(identifier);
            return err;
        };
        const bucket = allocator.create(PersistedBucket) catch |err| {
            allocator.free(key_dup);
            _ = map.remove(identifier);
            return err;
        };
        bucket.* = .{};
        gop.key_ptr.* = key_dup;
        gop.value_ptr.* = bucket;
    }
    return gop.value_ptr.*;
}

pub fn journalPathAlloc(allocator: Allocator, path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}.journal", .{path});
}

fn stringLessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

/// Serialize `persisted` deterministically: top-level JSON object with
/// lexicographically sorted bucket names, each mapping to an object with
/// lexicographically sorted entry keys and standard base64-encoded values.
pub fn serializeSnapshotAlloc(
    allocator: Allocator,
    persisted: *const PersistedStoreMap,
) ![]u8 {
    var bucket_names: std.ArrayListUnmanaged([]const u8) = .empty;
    defer bucket_names.deinit(allocator);
    try bucket_names.ensureTotalCapacityPrecise(allocator, persisted.count());

    var bit = persisted.keyIterator();
    while (bit.next()) |bname| {
        bucket_names.appendAssumeCapacity(bname.*);
    }
    std.mem.sort([]const u8, bucket_names.items, {}, stringLessThan);

    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var stringify: std.json.Stringify = .{ .writer = &aw.writer, .options = .{} };
    try stringify.beginObject();

    var entry_keys: std.ArrayListUnmanaged([]const u8) = .empty;
    defer entry_keys.deinit(allocator);

    for (bucket_names.items) |bname| {
        const bucket = persisted.get(bname).?;
        try stringify.objectField(bname);
        try stringify.beginObject();

        entry_keys.clearRetainingCapacity();
        try entry_keys.ensureTotalCapacityPrecise(allocator, bucket.entries.count());
        var kit = bucket.entries.keyIterator();
        while (kit.next()) |k| {
            entry_keys.appendAssumeCapacity(k.*);
        }
        std.mem.sort([]const u8, entry_keys.items, {}, stringLessThan);

        for (entry_keys.items) |k| {
            const value_bytes = bucket.entries.get(k).?;
            try stringify.objectField(k);
            const enc_len = std.base64.standard.Encoder.calcSize(value_bytes.len);
            const enc_buf = try allocator.alloc(u8, enc_len);
            defer allocator.free(enc_buf);
            const encoded = std.base64.standard.Encoder.encode(enc_buf, value_bytes);
            try stringify.write(encoded);
        }
        try stringify.endObject();
    }
    try stringify.endObject();

    return aw.toOwnedSlice();
}

/// Encode a redo journal frame:
///   - `"WAMRKVJ1"` (8 bytes)
///   - `version`: `u32` little-endian = 1 (4 bytes)
///   - `payload_len`: `u64` little-endian (8 bytes)
///   - `sha256(payload)`: `[32]u8` (32 bytes)
///   - `payload`: `payload_len` bytes
///   - `"WAMRKVJC"` (8 bytes)
pub fn encodeJournalFrame(allocator: Allocator, payload: []const u8) ![]u8 {
    const total_len = std.math.add(usize, JOURNAL_OVERHEAD_LEN, payload.len) catch
        return error.OutOfMemory;
    const buf = try allocator.alloc(u8, total_len);
    @memcpy(buf[0..8], JOURNAL_HEADER_MAGIC);
    std.mem.writeInt(u32, buf[8..12], JOURNAL_VERSION, .little);
    std.mem.writeInt(u64, buf[12..20], @intCast(payload.len), .little);
    var digest: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(payload, &digest, .{});
    @memcpy(buf[20..52], &digest);
    if (payload.len > 0) {
        @memcpy(buf[JOURNAL_HEADER_LEN .. JOURNAL_HEADER_LEN + payload.len], payload);
    }
    @memcpy(buf[JOURNAL_HEADER_LEN + payload.len ..][0..JOURNAL_TRAILER_LEN], JOURNAL_COMMIT_MAGIC);
    return buf;
}

/// Validate a redo journal frame and return the borrowed payload slice if valid.
/// Returns `null` if the header magic, version, exact frame length, commit
/// trailer magic, or SHA-256 payload digest is invalid.
pub fn decodeVerifiedJournalPayload(journal_bytes: []const u8) ?[]const u8 {
    if (journal_bytes.len < JOURNAL_OVERHEAD_LEN) return null;
    if (!std.mem.eql(u8, journal_bytes[0..8], JOURNAL_HEADER_MAGIC)) return null;
    const version = std.mem.readInt(u32, journal_bytes[8..12], .little);
    if (version != JOURNAL_VERSION) return null;

    const payload_len_u64 = std.mem.readInt(u64, journal_bytes[12..20], .little);
    const payload_len = std.math.cast(usize, payload_len_u64) orelse return null;
    if (journal_bytes.len - JOURNAL_OVERHEAD_LEN != payload_len) return null;

    const trailer_start = JOURNAL_HEADER_LEN + payload_len;
    const trailer = journal_bytes[trailer_start .. trailer_start + JOURNAL_TRAILER_LEN];
    if (!std.mem.eql(u8, trailer, JOURNAL_COMMIT_MAGIC)) return null;

    const payload = journal_bytes[JOURNAL_HEADER_LEN..trailer_start];
    var computed_digest: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(payload, &computed_digest, .{});
    if (!std.mem.eql(u8, &computed_digest, journal_bytes[20..52])) return null;

    return payload;
}

fn setEndPos(file: std.Io.File, io: std.Io, length: u64) !void {
    try file.setLength(io, length);
}

/// Overwrite `path` in place without replacing its inode or metadata:
/// `createFile` with `.truncate = false`, `writePositionalAll` at offset 0,
/// `setEndPos(io, payload.len)`, and `sync(io)`.
pub fn writeFileInPlace(io: std.Io, path: []const u8, payload: []const u8) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = false });
    defer file.close(io);
    try file.writePositionalAll(io, payload, 0);
    try setEndPos(file, io, @intCast(payload.len));
    try file.sync(io);
}

fn writeJournalFile(io: std.Io, journal_path: []const u8, frame: []const u8) !void {
    const file = try std.Io.Dir.cwd().createFile(io, journal_path, .{ .truncate = true });
    errdefer std.Io.Dir.cwd().deleteFile(io, journal_path) catch {};
    defer file.close(io);
    try file.writePositionalAll(io, frame, 0);
    try setEndPos(file, io, @intCast(frame.len));
    try file.sync(io);
}

/// Flush `persisted` to `path` using the write-ahead redo journal + in-place
/// target overwrite sequence:
///   1. Deterministically serialize `persisted` to canonical JSON.
///   2. Write and sync `<path>.journal`.
///   3. Overwrite `<path>` in place (`.truncate = false`, `writePositionalAll`,
///      `setEndPos`, `sync`).
///   4. Delete `<path>.journal`.
pub fn flushSnapshot(
    allocator: Allocator,
    path: []const u8,
    persisted: *const PersistedStoreMap,
) !void {
    const payload = try serializeSnapshotAlloc(allocator, persisted);
    defer allocator.free(payload);

    const frame = try encodeJournalFrame(allocator, payload);
    defer allocator.free(frame);

    const journal_path = try journalPathAlloc(allocator, path);
    defer allocator.free(journal_path);

    const io = std.Io.Threaded.global_single_threaded.io();
    try writeJournalFile(io, journal_path, frame);
    try writeFileInPlace(io, path, payload);
    std.Io.Dir.cwd().deleteFile(io, journal_path) catch {};
}

/// Parse a JSON snapshot buffer into `persisted`.
pub fn parseSnapshotBytes(
    allocator: Allocator,
    bytes: []const u8,
    persisted: *PersistedStoreMap,
) !void {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch {
        return error.KeyvalueStoreInvalidJson;
    };
    defer parsed.deinit();

    switch (parsed.value) {
        .object => |obj| {
            var bit = obj.iterator();
            while (bit.next()) |bkv| {
                const inner = switch (bkv.value_ptr.*) {
                    .object => |o| o,
                    else => return error.KeyvalueStoreInvalidBucket,
                };
                const bucket = try getOrCreateBucket(persisted, allocator, bkv.key_ptr.*);
                var eit = inner.iterator();
                while (eit.next()) |ekv| {
                    const value_str = switch (ekv.value_ptr.*) {
                        .string => |s| s,
                        else => return error.KeyvalueStoreInvalidValueType,
                    };
                    const dec_len = std.base64.standard.Decoder.calcSizeForSlice(value_str) catch {
                        return error.KeyvalueStoreInvalidBase64;
                    };
                    const value_dup = try allocator.alloc(u8, dec_len);
                    errdefer allocator.free(value_dup);
                    std.base64.standard.Decoder.decode(value_dup, value_str) catch {
                        return error.KeyvalueStoreInvalidBase64;
                    };
                    const key_dup = try allocator.dupe(u8, ekv.key_ptr.*);
                    errdefer allocator.free(key_dup);
                    const gop = try bucket.entries.getOrPut(allocator, key_dup);
                    if (gop.found_existing) {
                        allocator.free(key_dup);
                        allocator.free(gop.value_ptr.*);
                        gop.value_ptr.* = value_dup;
                    } else {
                        gop.key_ptr.* = key_dup;
                        gop.value_ptr.* = value_dup;
                    }
                }
            }
        },
        else => return error.KeyvalueStoreInvalidRoot,
    }
}

/// Recover from any pending `<path>.journal` and load `<path>` into `persisted`:
///   1. If `<path>.journal` exists and is a valid committed journal frame,
///      replay its payload onto `<path>` in place and remove `<path>.journal`.
///   2. If `<path>.journal` exists but is truncated/uncommitted/corrupted,
///      discard `<path>.journal` and leave `<path>` untouched.
///   3. Read `<path>` (treating `error.FileNotFound` as an empty store) and
///      hydrate `persisted`.
pub fn recoverAndLoadSnapshot(
    allocator: Allocator,
    path: []const u8,
    persisted: *PersistedStoreMap,
) !void {
    const cwd = std.Io.Dir.cwd();
    const io = std.Io.Threaded.global_single_threaded.io();

    const journal_path = try journalPathAlloc(allocator, path);
    defer allocator.free(journal_path);

    const max_journal_len = MAX_SNAPSHOT_BYTES + JOURNAL_OVERHEAD_LEN;
    if (cwd.readFileAlloc(
        io,
        journal_path,
        allocator,
        @fromBackingInt(@intCast(max_journal_len)),
    )) |journal_bytes| {
        defer allocator.free(journal_bytes);
        if (decodeVerifiedJournalPayload(journal_bytes)) |payload| {
            try writeFileInPlace(io, path, payload);
        }
        cwd.deleteFile(io, journal_path) catch {};
    } else |err| switch (err) {
        error.FileNotFound => {},
        error.StreamTooLong => {
            // Corrupt/oversized journal — discard and leave `<path>` untouched.
            cwd.deleteFile(io, journal_path) catch {};
        },
        else => return err,
    }

    const bytes = cwd.readFileAlloc(
        io,
        path,
        allocator,
        @fromBackingInt(@intCast(MAX_SNAPSHOT_BYTES)),
    ) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    defer allocator.free(bytes);

    try parseSnapshotBytes(allocator, bytes, persisted);
}

fn putTestEntry(
    allocator: Allocator,
    map: *PersistedStoreMap,
    bucket_name: []const u8,
    key: []const u8,
    value: []const u8,
) !void {
    const bucket = try getOrCreateBucket(map, allocator, bucket_name);
    const val_dup = try allocator.dupe(u8, value);
    errdefer allocator.free(val_dup);
    if (bucket.entries.getEntry(key)) |existing| {
        allocator.free(existing.value_ptr.*);
        existing.value_ptr.* = val_dup;
    } else {
        const key_dup = try allocator.dupe(u8, key);
        errdefer allocator.free(key_dup);
        try bucket.entries.putNoClobber(allocator, key_dup, val_dup);
    }
}

test "keyvalue_persistence #972: deterministic sorted-key JSON output" {
    const testing = std.testing;
    var map_a: PersistedStoreMap = .empty;
    defer deinitPersistedStoreMap(&map_a, testing.allocator);

    try putTestEntry(testing.allocator, &map_a, "zeta", "k3", "v3");
    try putTestEntry(testing.allocator, &map_a, "alpha", "z-key", "last");
    try putTestEntry(testing.allocator, &map_a, "alpha", "a-key", "first");
    try putTestEntry(testing.allocator, &map_a, "beta", "only", "mid");

    var map_b: PersistedStoreMap = .empty;
    defer deinitPersistedStoreMap(&map_b, testing.allocator);

    try putTestEntry(testing.allocator, &map_b, "beta", "only", "mid");
    try putTestEntry(testing.allocator, &map_b, "alpha", "a-key", "first");
    try putTestEntry(testing.allocator, &map_b, "zeta", "k3", "v3");
    try putTestEntry(testing.allocator, &map_b, "alpha", "z-key", "last");

    const json_a = try serializeSnapshotAlloc(testing.allocator, &map_a);
    defer testing.allocator.free(json_a);
    const json_b = try serializeSnapshotAlloc(testing.allocator, &map_b);
    defer testing.allocator.free(json_b);

    try testing.expectEqualStrings(
        "{\"alpha\":{\"a-key\":\"Zmlyc3Q=\",\"z-key\":\"bGFzdA==\"},\"beta\":{\"only\":\"bWlk\"},\"zeta\":{\"k3\":\"djM=\"}}",
        json_a,
    );
    try testing.expectEqualStrings(json_a, json_b);
}

test "keyvalue_persistence #972: flushSnapshot preserves target inode and cleans up journal" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();

    const rel_path = try std.fmt.allocPrint(
        testing.allocator,
        ".zig-cache/tmp/{s}/kv-store.json",
        .{tmp.sub_path[0..]},
    );
    defer testing.allocator.free(rel_path);

    var map: PersistedStoreMap = .empty;
    defer deinitPersistedStoreMap(&map, testing.allocator);

    // Flush 1: initial creation with a larger payload.
    try putTestEntry(testing.allocator, &map, "b1", "long-key", "long-initial-payload-0123456789");
    try flushSnapshot(testing.allocator, rel_path, &map);

    const st1 = try tmp.dir.statFile(io, "kv-store.json", .{});
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "kv-store.json.journal", .{}));

    // Flush 2: overwrite with a shorter payload to verify setEndPos truncation + inode preservation.
    deinitPersistedStoreMap(&map, testing.allocator);
    try putTestEntry(testing.allocator, &map, "b1", "k", "v");
    try flushSnapshot(testing.allocator, rel_path, &map);

    const st2 = try tmp.dir.statFile(io, "kv-store.json", .{});
    try testing.expectEqual(st1.inode, st2.inode);
    try testing.expect(st2.size < st1.size);
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "kv-store.json.journal", .{}));

    var loaded: PersistedStoreMap = .empty;
    defer deinitPersistedStoreMap(&loaded, testing.allocator);
    try recoverAndLoadSnapshot(testing.allocator, rel_path, &loaded);
    try testing.expectEqualSlices(u8, "v", loaded.get("b1").?.entries.get("k").?);
}

test "keyvalue_persistence #972: crash recovery replays valid committed journal in place" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();

    const rel_path = try std.fmt.allocPrint(
        testing.allocator,
        ".zig-cache/tmp/{s}/kv-store.json",
        .{tmp.sub_path[0..]},
    );
    defer testing.allocator.free(rel_path);

    // Write a corrupted/truncated target file first and record its inode.
    try tmp.dir.writeFile(io, .{ .sub_path = "kv-store.json", .data = "{\"corrupt\":true" });
    const st_before = try tmp.dir.statFile(io, "kv-store.json", .{});

    // Build a valid committed journal for the new snapshot.
    var map: PersistedStoreMap = .empty;
    defer deinitPersistedStoreMap(&map, testing.allocator);
    try putTestEntry(testing.allocator, &map, "recovered", "alpha", "committed-value");
    const payload = try serializeSnapshotAlloc(testing.allocator, &map);
    defer testing.allocator.free(payload);
    const frame = try encodeJournalFrame(testing.allocator, payload);
    defer testing.allocator.free(frame);
    try tmp.dir.writeFile(io, .{ .sub_path = "kv-store.json.journal", .data = frame });

    // Recover and load.
    var loaded: PersistedStoreMap = .empty;
    defer deinitPersistedStoreMap(&loaded, testing.allocator);
    try recoverAndLoadSnapshot(testing.allocator, rel_path, &loaded);

    // Inode of kv-store.json is preserved, journal is removed, and target has committed payload.
    const st_after = try tmp.dir.statFile(io, "kv-store.json", .{});
    try testing.expectEqual(st_before.inode, st_after.inode);
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "kv-store.json.journal", .{}));
    try testing.expectEqualSlices(
        u8,
        "committed-value",
        loaded.get("recovered").?.entries.get("alpha").?,
    );
    const disk_bytes = try tmp.dir.readFileAlloc(
        io,
        "kv-store.json",
        testing.allocator,
        @fromBackingInt(@intCast(MAX_SNAPSHOT_BYTES)),
    );
    defer testing.allocator.free(disk_bytes);
    try testing.expectEqualStrings(payload, disk_bytes);
}

test "keyvalue_persistence #972: crash recovery discards uncommitted or corrupt journal and keeps prior target" {
    const testing = std.testing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();

    const rel_path = try std.fmt.allocPrint(
        testing.allocator,
        ".zig-cache/tmp/{s}/kv-store.json",
        .{tmp.sub_path[0..]},
    );
    defer testing.allocator.free(rel_path);

    // Flush a valid initial snapshot to kv-store.json.
    var initial_map: PersistedStoreMap = .empty;
    defer deinitPersistedStoreMap(&initial_map, testing.allocator);
    try putTestEntry(testing.allocator, &initial_map, "stable", "k1", "prior-intact-value");
    try flushSnapshot(testing.allocator, rel_path, &initial_map);
    const st_before = try tmp.dir.statFile(io, "kv-store.json", .{});

    // Write a journal for a newer state, then truncate its commit trailer (simulating crash mid-journal-write).
    var newer_map: PersistedStoreMap = .empty;
    defer deinitPersistedStoreMap(&newer_map, testing.allocator);
    try putTestEntry(testing.allocator, &newer_map, "stable", "k1", "uncommitted-new-value");
    const newer_payload = try serializeSnapshotAlloc(testing.allocator, &newer_map);
    defer testing.allocator.free(newer_payload);
    const full_frame = try encodeJournalFrame(testing.allocator, newer_payload);
    defer testing.allocator.free(full_frame);

    try tmp.dir.writeFile(io, .{
        .sub_path = "kv-store.json.journal",
        .data = full_frame[0 .. full_frame.len - 3],
    });

    var loaded: PersistedStoreMap = .empty;
    defer deinitPersistedStoreMap(&loaded, testing.allocator);
    try recoverAndLoadSnapshot(testing.allocator, rel_path, &loaded);

    const st_after = try tmp.dir.statFile(io, "kv-store.json", .{});
    try testing.expectEqual(st_before.inode, st_after.inode);
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "kv-store.json.journal", .{}));
    try testing.expectEqualSlices(
        u8,
        "prior-intact-value",
        loaded.get("stable").?.entries.get("k1").?,
    );
}
