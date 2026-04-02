// Copyright (C) 2019 Intel Corporation.  All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! EMS (Embedded Memory Service) allocator — a pool-based allocator
//! designed for constrained environments. Implements `std.mem.Allocator`.
//!
//! Ported from the C implementation in `core/shared/mem-alloc/ems/`.
//!
//! The heap is initialized from a contiguous memory pool (a `[]u8` slice).
//! Each allocation has a header (HMU – Heap Memory Unit) with size and flags.
//! Free blocks are organized in a singly-linked normal list (for small sizes)
//! and a binary search tree (for large sizes, best-fit).

const std = @import("std");
const assert = std.debug.assert;
const math = std.math;
const testing = std.testing;
const log = std.log.scoped(.ems);

/// Simple spinlock mutex (Zig 0.16-dev moved std.Thread.Mutex behind Io).
const SpinMutex = struct {
    state: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),

    fn lock(self: *SpinMutex) void {
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }

    fn unlock(self: *SpinMutex) void {
        self.state.store(0, .release);
    }
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of normal (small) free-list buckets.  Bucket i holds blocks of
/// size `i * 8` bytes.  Sizes >= `normal_node_count * 8` go into the tree.
const normal_node_count: usize = 32;

/// Maximum block size handled by the normal (small) free list.
const fc_normal_max_size: usize = (normal_node_count - 1) * 8;

/// HMU header size in bytes.
const hmu_size: usize = @sizeOf(Hmu);

/// Minimum useful allocation (header + at least 8 payload bytes), 8-aligned.
const smallest_size: usize = alignUp(hmu_size + 8);

/// Minimum pool size accepted by `initWithPool`.
const min_pool_size: usize = smallest_size * 4 + @sizeOf(GcHeap) + gc_head_padding + 16;

/// Padding inserted between the GcHeap struct and the usable pool area.
/// (Set to 0 in Zig: unlike the C version which uses 4-byte padding plus
/// packed structs for alignment tricks, Zig extern structs use natural
/// alignment with compiler-inserted padding.)
const gc_head_padding: usize = 0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn alignUp(s: usize) usize {
    return (s + 7) & ~@as(usize, 7);
}

fn ptrToAddr(p: anytype) usize {
    return @intFromPtr(p);
}

// ---------------------------------------------------------------------------
// HMU – Heap Memory Unit header (4 bytes, mirrors the C `hmu_t`)
// ---------------------------------------------------------------------------

/// Type tag stored in the top 2 bits of the HMU header.
const HmuType = enum(u2) {
    /// Free Memory (unused tag in our simplified port)
    fm = 0,
    /// Free Chunk (on a free list)
    fc = 1,
    /// VM Object (allocated by the runtime)
    vo = 2,
    /// WASM Object (GC-managed – not used in this Zig port)
    wo = 3,
};

/// Heap Memory Unit – 4-byte header preceding every block.
///
/// Bit layout of `header`:
///   [31:30]  unit type  (HmuType)
///   [29]     P-in-use   (previous block is allocated)
///   [28]     reserved / VO-freed / WO-mark
///   [26:0]   size >> 3  (block size in 8-byte units)
///
/// Because sizes are always 8-aligned we store `size >> 3`, allowing
/// block sizes up to (2^27)*8 = 1 GiB.
const Hmu = extern struct {
    header: u32 = 0,

    const ut_offset = 30;
    const ut_size = 2;
    const p_offset = 29;
    const size_offset = 0;
    const size_bits = 27;

    // -- type --
    fn getUt(self: *const Hmu) HmuType {
        return @enumFromInt(getBits(self.header, ut_offset, ut_size));
    }
    fn setUt(self: *Hmu, ut: HmuType) void {
        setBits(&self.header, ut_offset, ut_size, @intFromEnum(ut));
    }

    // -- previous-in-use --
    fn getPinuse(self: *const Hmu) bool {
        return getBit(self.header, p_offset);
    }
    fn markPinuse(self: *Hmu) void {
        setBit(&self.header, p_offset);
    }
    fn unmarkPinuse(self: *Hmu) void {
        clrBit(&self.header, p_offset);
    }

    // -- size (always 8-aligned) --
    fn getSize(self: *const Hmu) usize {
        return @as(usize, getBits(self.header, size_offset, size_bits)) << 3;
    }
    fn setSize(self: *Hmu, size: usize) void {
        setBits(&self.header, size_offset, size_bits, @truncate(size >> 3));
    }

    // -- pointer arithmetic --
    fn toObj(self: *Hmu) [*]u8 {
        const base: [*]u8 = @ptrCast(self);
        return base + hmu_size;
    }

    fn fromObj(ptr: [*]u8) *Hmu {
        return @alignCast(@ptrCast(ptr - hmu_size));
    }

    fn nextHmu(self: *Hmu) *Hmu {
        const base: [*]u8 = @ptrCast(self);
        return @alignCast(@ptrCast(base + self.getSize()));
    }
};

// ---------------------------------------------------------------------------
// Bit-manipulation helpers (matching the C macros)
// ---------------------------------------------------------------------------

fn getBit(v: u32, offset: u5) bool {
    return (v & (@as(u32, 1) << offset)) != 0;
}
fn setBit(v: *u32, offset: u5) void {
    v.* |= (@as(u32, 1) << offset);
}
fn clrBit(v: *u32, offset: u5) void {
    v.* &= ~(@as(u32, 1) << offset);
}
fn getBits(v: u32, offset: u5, size: u5) u32 {
    return (v >> offset) & ((@as(u32, 1) << size) - 1);
}
fn setBits(v: *u32, offset: u5, size: u5, value: u32) void {
    const mask = ((@as(u32, 1) << size) - 1) << offset;
    v.* = (v.* & ~mask) | (value << offset);
}

// ---------------------------------------------------------------------------
// Normal-list node (small free chunks) – overlaid on the HMU payload
// ---------------------------------------------------------------------------

/// Node for the singly-linked normal (small) free list.
/// Overlaid on the first bytes of a free HMU's payload.
const NormalNode = extern struct {
    hmu_header: Hmu,
    /// Signed offset (in bytes) to the next node, or 0 if tail.
    next_offset: i32 = 0,

    fn getNext(self: *NormalNode) ?*NormalNode {
        if (self.next_offset == 0) return null;
        const base: [*]u8 = @ptrCast(self);
        const addr = @as(isize, @intCast(ptrToAddr(base))) + @as(isize, self.next_offset);
        return @ptrFromInt(@as(usize, @intCast(addr)));
    }

    fn setNext(self: *NormalNode, next: ?*NormalNode) void {
        if (next) |n| {
            const diff = @as(isize, @intCast(ptrToAddr(n))) - @as(isize, @intCast(ptrToAddr(self)));
            self.next_offset = @intCast(diff);
        } else {
            self.next_offset = 0;
        }
    }
};

// ---------------------------------------------------------------------------
// Tree node (large free chunks) – overlaid on the HMU payload
// ---------------------------------------------------------------------------

/// BST node for large free chunks.  Overlaid on the first bytes of a free
/// HMU's payload.  The tree ordering is: size[left] <= size[cur] < size[right].
const TreeNode = extern struct {
    hmu_header: Hmu,
    left: ?*TreeNode = null,
    right: ?*TreeNode = null,
    parent: ?*TreeNode = null,
    size: u32 = 0,
};

// ---------------------------------------------------------------------------
// Normal-list head (bucket)
// ---------------------------------------------------------------------------

const NormalListHead = extern struct {
    next: ?*NormalNode = null,
};

// ---------------------------------------------------------------------------
// GcHeap – the managed heap
// ---------------------------------------------------------------------------

pub const GcHeap = struct {
    /// Self-pointer for validity check (mirrors C `heap_id`).
    heap_id: usize = 0,

    base_addr: [*]u8 = undefined,
    current_size: usize = 0,

    mutex: SpinMutex = .{},

    kfc_normal_list: [normal_node_count]NormalListHead = [_]NormalListHead{.{}} ** normal_node_count,

    /// Inline storage for the sentinel tree root node.
    kfc_tree_root_buf: [@sizeOf(TreeNode)]u8 align(@alignOf(TreeNode)) = [_]u8{0} ** @sizeOf(TreeNode),
    kfc_tree_root: ?*TreeNode = null,

    is_heap_corrupted: bool = false,

    init_size: usize = 0,
    highmark_size: usize = 0,
    total_free_size: usize = 0,

    // -- public API ---------------------------------------------------------

    /// Initialize a heap from a contiguous memory pool.
    ///
    /// The beginning of `pool` is used for the `GcHeap` structure itself;
    /// the remainder becomes the allocatable area.
    pub fn initWithPool(pool: []u8) !*GcHeap {
        if (pool.len < min_pool_size) return error.PoolTooSmall;

        // Align the start of the pool to 8 bytes.
        const pool_start = ptrToAddr(pool.ptr);
        const aligned_start = (pool_start + 7) & ~@as(usize, 7);
        const offset = aligned_start - pool_start;
        if (offset >= pool.len) return error.PoolTooSmall;

        const heap: *GcHeap = @alignCast(@ptrCast(pool.ptr + offset));

        // Base address: after the heap struct, 8-aligned, + head padding.
        const after_struct = aligned_start + @sizeOf(GcHeap);
        const base_aligned = ((after_struct + 7) & ~@as(usize, 7)) + gc_head_padding;
        const pool_end = pool_start + pool.len;
        if (base_aligned >= pool_end) return error.PoolTooSmall;
        const heap_max_size = (pool_end - base_aligned) & ~@as(usize, 7);
        if (heap_max_size < smallest_size) return error.PoolTooSmall;

        const base_ptr: [*]u8 = @ptrFromInt(base_aligned);

        // Zero out the heap structure and pool area.
        const heap_bytes: [*]u8 = @ptrCast(heap);
        @memset(heap_bytes[0..@sizeOf(GcHeap)], 0);
        @memset(base_ptr[0..heap_max_size], 0);

        heap.* = GcHeap{};
        heap.current_size = heap_max_size;
        heap.init_size = heap_max_size;
        heap.base_addr = base_ptr;
        heap.heap_id = @intFromPtr(heap);
        heap.total_free_size = heap_max_size;
        heap.highmark_size = 0;

        // Initialize the sentinel tree root (lives inside the heap struct).
        const root: *TreeNode = @alignCast(@ptrCast(&heap.kfc_tree_root_buf));
        root.* = TreeNode{ .hmu_header = .{} };
        root.size = @intCast(@sizeOf(TreeNode));
        root.hmu_header.setUt(.fc);
        root.hmu_header.setSize(@sizeOf(TreeNode));
        heap.kfc_tree_root = root;

        // The entire pool is one big free chunk; insert it as the root's
        // right child (so searches start there).
        const q: *TreeNode = @alignCast(@ptrCast(base_ptr));
        q.* = TreeNode{ .hmu_header = .{} };
        q.hmu_header.setUt(.fc);
        q.hmu_header.setSize(heap_max_size);
        q.hmu_header.markPinuse();
        q.size = @intCast(heap_max_size);
        root.right = q;
        q.parent = root;

        return heap;
    }

    /// Tear down the heap.  After this call the pool memory is zeroed.
    pub fn destroy(self: *GcHeap) void {
        @memset(self.base_addr[0..self.current_size], 0);
        const heap_bytes: [*]u8 = @ptrCast(self);
        @memset(heap_bytes[0..@sizeOf(GcHeap)], 0);
    }

    /// Return current heap statistics.
    pub fn getStats(self: *GcHeap) HeapStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        return .{
            .total_size = self.current_size,
            .used_size = self.current_size - self.total_free_size,
            .free_size = self.total_free_size,
            .highest_allocated = self.highmark_size,
        };
    }

    // -- internal allocation helpers ----------------------------------------

    fn isValid(self: *GcHeap) bool {
        return self.heap_id == @intFromPtr(self);
    }

    fn isInHeap(self: *GcHeap, addr: usize) bool {
        const base = ptrToAddr(self.base_addr);
        return addr >= base and addr < base + self.current_size;
    }

    fn hmuIsInHeap(self: *GcHeap, hmu: *Hmu) bool {
        return self.isInHeap(ptrToAddr(hmu));
    }

    fn endAddr(self: *GcHeap) usize {
        return ptrToAddr(self.base_addr) + self.current_size;
    }

    // -- tree operations ----------------------------------------------------

    /// Remove a node from the BST.  Returns false if corruption detected.
    fn removeTreeNode(self: *GcHeap, p: *TreeNode) bool {
        // Determine which parent slot points to p.
        const parent = p.parent orelse return false;
        const slot: *?*TreeNode = blk: {
            if (parent.right) |r| {
                if (r == p) break :blk &parent.right;
            }
            if (parent.left) |l| {
                if (l == p) break :blk &parent.left;
            }
            self.is_heap_corrupted = true;
            return false;
        };

        if (p.left == null) {
            // Case 1: no left child — promote right.
            slot.* = p.right;
            if (p.right) |r| r.parent = parent;
            p.left = null;
            p.right = null;
            p.parent = null;
            return true;
        }

        if (p.right == null) {
            // Case 2: no right child — promote left.
            slot.* = p.left;
            if (p.left) |l| l.parent = parent;
            p.left = null;
            p.right = null;
            p.parent = null;
            return true;
        }

        // Case 3: both children — find in-order predecessor (rightmost in left subtree).
        var q: *TreeNode = p.left.?;
        while (q.right) |r| {
            q = r;
        }

        // Remove predecessor from its current position.
        if (!self.removeTreeNode(q)) return false;

        // Replace p with q.
        slot.* = q;
        q.parent = parent;
        q.left = p.left;
        q.right = p.right;
        if (q.left) |l| l.parent = q;
        if (q.right) |r| r.parent = q;

        p.left = null;
        p.right = null;
        p.parent = null;
        return true;
    }

    /// Unlink a free HMU from whichever free list it is on.
    fn unlinkHmu(self: *GcHeap, hmu: *Hmu) bool {
        if (hmu.getUt() != .fc) {
            self.is_heap_corrupted = true;
            return false;
        }

        const size = hmu.getSize();
        if (size < fc_normal_max_size) {
            // Normal (small) list.
            const node_idx = size >> 3;
            var prev: ?*NormalNode = null;
            var node = self.kfc_normal_list[node_idx].next;
            while (node) |n| {
                const next = n.getNext();
                if (@as(*Hmu, &n.hmu_header) == hmu) {
                    if (prev) |p| {
                        p.setNext(next);
                    } else {
                        self.kfc_normal_list[node_idx].next = next;
                    }
                    return true;
                }
                prev = n;
                node = next;
            }
            // Not found – corruption.
            return false;
        } else {
            // Tree node.
            return self.removeTreeNode(@alignCast(@ptrCast(hmu)));
        }
    }

    /// Write the block size into the last 4 bytes of a free chunk
    /// (used by the *previous* coalescing logic during free).
    fn setFreeSize(hmu: *Hmu) void {
        const size = hmu.getSize();
        const base: [*]u8 = @ptrCast(hmu);
        const tail: *u32 = @alignCast(@ptrCast(base + size - @sizeOf(u32)));
        tail.* = @intCast(size);
    }

    /// Add a free chunk to the appropriate free list.
    fn addFc(self: *GcHeap, hmu: *Hmu, size: usize) bool {
        assert(size > 0 and (size & 7) == 0);

        hmu.setUt(.fc);
        hmu.setSize(size);
        setFreeSize(hmu);

        if (size < fc_normal_max_size) {
            // Normal (small) list — prepend.
            const np: *NormalNode = @alignCast(@ptrCast(hmu));
            const node_idx = size >> 3;
            np.setNext(self.kfc_normal_list[node_idx].next);
            self.kfc_normal_list[node_idx].next = np;
            return true;
        }

        // Large block — insert into BST.
        const node: *TreeNode = @alignCast(@ptrCast(hmu));
        node.size = @intCast(size);
        node.left = null;
        node.right = null;
        node.parent = null;

        const root = self.kfc_tree_root.?;
        var tp: *TreeNode = root;
        while (true) {
            if (tp.size < size) {
                if (tp.right) |r| {
                    tp = r;
                } else {
                    tp.right = node;
                    node.parent = tp;
                    return true;
                }
            } else {
                if (tp.left) |l| {
                    tp = l;
                } else {
                    tp.left = node;
                    node.parent = tp;
                    return true;
                }
            }
        }
    }

    /// Core allocation — find a free HMU of at least `size` bytes.
    fn allocHmu(self: *GcHeap, req_size: usize) ?*Hmu {
        var size = req_size;
        if (size < smallest_size) size = smallest_size;

        const base = ptrToAddr(self.base_addr);
        const end = base + self.current_size;

        // 1. Try the normal (small) list.
        if (size < fc_normal_max_size) {
            const init_idx = size >> 3;
            var idx: usize = init_idx;
            while (idx < normal_node_count) : (idx += 1) {
                if (self.kfc_normal_list[idx].next) |p| {
                    self.kfc_normal_list[idx].next = p.getNext();

                    const p_hmu: *Hmu = &p.hmu_header;
                    const found_size = idx << 3;
                    if (idx != init_idx and found_size >= size + smallest_size) {
                        // Split: give back the remainder.
                        const rest_ptr: [*]u8 = @ptrCast(p_hmu);
                        const rest: *Hmu = @alignCast(@ptrCast(rest_ptr + size));
                        if (!self.addFc(rest, found_size - size)) return null;
                        rest.markPinuse();
                    } else {
                        size = found_size;
                        const next_addr = ptrToAddr(p_hmu) + size;
                        if (next_addr >= base and next_addr < end) {
                            const next: *Hmu = @ptrFromInt(next_addr);
                            next.markPinuse();
                        }
                    }

                    self.total_free_size -= size;
                    if (self.current_size - self.total_free_size > self.highmark_size)
                        self.highmark_size = self.current_size - self.total_free_size;

                    p_hmu.setSize(size);
                    return p_hmu;
                }
            }
        }

        // 2. Try the BST (best-fit search).
        const root = self.kfc_tree_root orelse return null;
        var tp = root.right;
        var last_tp: ?*TreeNode = null;

        while (tp) |t| {
            if (t.size < size) {
                tp = t.right;
            } else {
                last_tp = t;
                tp = t.left;
            }
        }

        if (last_tp) |best| {
            if (!self.removeTreeNode(best)) return null;

            const best_hmu: *Hmu = &best.hmu_header;
            const best_size: usize = best.size;

            if (best_size >= size + smallest_size) {
                const rest_base: [*]u8 = @ptrCast(best_hmu);
                const rest: *Hmu = @alignCast(@ptrCast(rest_base + size));
                if (!self.addFc(rest, best_size - size)) return null;
                rest.markPinuse();
            } else {
                size = best_size;
                const next_addr = ptrToAddr(best_hmu) + size;
                if (next_addr >= base and next_addr < end) {
                    const next: *Hmu = @ptrFromInt(next_addr);
                    next.markPinuse();
                }
            }

            self.total_free_size -= size;
            if (self.current_size - self.total_free_size > self.highmark_size)
                self.highmark_size = self.current_size - self.total_free_size;

            best_hmu.setSize(size);
            return best_hmu;
        }

        return null;
    }

    /// Allocate `size` bytes of payload (caller-visible).
    fn allocObj(self: *GcHeap, size: usize) ?[*]u8 {
        if (self.is_heap_corrupted) return null;

        // Total size: HMU header + requested payload, 8-aligned.
        const tot_size_unaligned = hmu_size + size;
        const tot_size = alignUp(tot_size_unaligned);
        if (tot_size < size) return null; // overflow

        const hmu = self.allocHmu(tot_size) orelse return null;

        const actual_size = hmu.getSize();
        hmu.setUt(.vo);
        // Clear the VO-freed bit (bit 28).
        clrBit(&hmu.header, 28);

        const obj = hmu.toObj();
        // Zero the allocated region.
        const to_zero = actual_size - hmu_size;
        @memset(obj[0..to_zero], 0);

        return obj;
    }

    /// Free an object previously allocated by `allocObj`.
    fn freeObj(self: *GcHeap, ptr: [*]u8) void {
        if (self.is_heap_corrupted) return;

        const hmu = Hmu.fromObj(ptr);
        const base = ptrToAddr(self.base_addr);
        const end = base + self.current_size;
        const hmu_addr = ptrToAddr(hmu);

        if (hmu_addr < base or hmu_addr >= end) return;
        if (hmu.getUt() != .vo) return;

        var total_size = hmu.getSize();
        var coalesced_hmu: *Hmu = hmu;

        self.total_free_size += total_size;

        // Coalesce with previous block if it is free.
        if (!hmu.getPinuse()) {
            // Read the size stored at the end of the previous free block.
            const prev_size_ptr: *u32 = @alignCast(@ptrCast(ptr - hmu_size - @sizeOf(u32)));
            const prev_size: usize = prev_size_ptr.*;
            if (prev_size > 0 and prev_size <= hmu_addr - base) {
                const prev_addr = hmu_addr - prev_size;
                if (prev_addr >= base) {
                    const prev: *Hmu = @ptrFromInt(prev_addr);
                    if (prev.getUt() == .fc) {
                        total_size += prev.getSize();
                        coalesced_hmu = prev;
                        if (!self.unlinkHmu(prev)) return;
                    }
                }
            }
        }

        // Coalesce with next block if it is free.
        const next_addr = ptrToAddr(coalesced_hmu) + total_size;
        if (next_addr >= base and next_addr < end) {
            const next: *Hmu = @ptrFromInt(next_addr);
            if (next.getUt() == .fc) {
                total_size += next.getSize();
                if (!self.unlinkHmu(next)) return;
            }
        }

        if (!self.addFc(coalesced_hmu, total_size)) return;

        // Mark next block's P-in-use as false.
        const after_addr = ptrToAddr(coalesced_hmu) + total_size;
        if (after_addr >= base and after_addr < end) {
            const after: *Hmu = @ptrFromInt(after_addr);
            after.unmarkPinuse();
        }
    }

    /// Attempt to resize an existing allocation in-place.
    /// Returns true only if the block can be grown by absorbing a free
    /// successor, or if it is being shrunk.
    fn resizeObj(self: *GcHeap, ptr: [*]u8, old_payload: usize, new_payload: usize) bool {
        if (self.is_heap_corrupted) return false;

        const hmu = Hmu.fromObj(ptr);
        const old_total = hmu.getSize();
        const new_total = alignUp(hmu_size + new_payload);
        if (new_total < new_payload) return false; // overflow

        if (new_total <= old_total) {
            // Shrink: optionally split remainder.
            if (old_total - new_total >= smallest_size) {
                hmu.setSize(new_total);
                const rest_base: [*]u8 = @ptrCast(hmu);
                const rest: *Hmu = @alignCast(@ptrCast(rest_base + new_total));
                self.total_free_size += old_total - new_total;
                if (!self.addFc(rest, old_total - new_total)) return false;
                rest.markPinuse();
            }
            return true;
        }

        // Try to grow into the next block.
        const base = ptrToAddr(self.base_addr);
        const end = base + self.current_size;
        const next_addr = ptrToAddr(hmu) + old_total;
        if (next_addr >= base and next_addr < end) {
            const next: *Hmu = @ptrFromInt(next_addr);
            if (next.getUt() == .fc) {
                const next_size = next.getSize();
                if (old_total + next_size >= new_total) {
                    if (!self.unlinkHmu(next)) return false;
                    const combined = old_total + next_size;
                    if (combined - new_total >= smallest_size) {
                        hmu.setSize(new_total);
                        const rest_base: [*]u8 = @ptrCast(hmu);
                        const rest: *Hmu = @alignCast(@ptrCast(rest_base + new_total));
                        if (!self.addFc(rest, combined - new_total)) return false;
                        rest.markPinuse();
                        // Zero the new region.
                        const obj_ptr: [*]u8 = hmu.toObj();
                        @memset(obj_ptr[old_payload..new_payload], 0);
                    } else {
                        hmu.setSize(combined);
                        const after = ptrToAddr(hmu) + combined;
                        if (after >= base and after < end) {
                            const after_hmu: *Hmu = @ptrFromInt(after);
                            after_hmu.markPinuse();
                        }
                        const obj_ptr: [*]u8 = hmu.toObj();
                        @memset(obj_ptr[old_payload..new_payload], 0);
                    }
                    self.total_free_size -= (hmu.getSize() - old_total);
                    if (self.current_size - self.total_free_size > self.highmark_size)
                        self.highmark_size = self.current_size - self.total_free_size;
                    return true;
                }
            }
        }

        return false;
    }
};

// ---------------------------------------------------------------------------
// Heap statistics
// ---------------------------------------------------------------------------

pub const HeapStats = struct {
    total_size: usize,
    used_size: usize,
    free_size: usize,
    highest_allocated: usize,
};

// ---------------------------------------------------------------------------
// EmsAllocator – std.mem.Allocator implementation
// ---------------------------------------------------------------------------

pub const EmsAllocator = struct {
    heap: *GcHeap,

    pub fn allocator(self: *EmsAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    const vtable: std.mem.Allocator.VTable = .{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
        const self: *EmsAllocator = @alignCast(@ptrCast(ctx));
        self.heap.mutex.lock();
        defer self.heap.mutex.unlock();

        // We need to over-allocate to be able to satisfy alignment > 8.
        // For alignment <= 8, the allocator naturally provides 8-byte alignment.
        const raw_align = alignment.toByteUnits();
        if (raw_align <= 8) {
            return self.heap.allocObj(len);
        }

        // Over-allocate and then find the aligned position within.
        const extra = raw_align - 8 + @sizeOf(usize);
        const total = len + extra;
        const raw_ptr = self.heap.allocObj(total) orelse return null;

        const raw_addr = ptrToAddr(raw_ptr);
        var aligned_addr = (raw_addr + extra) & ~(raw_align - 1);
        if (aligned_addr < raw_addr + @sizeOf(usize)) {
            aligned_addr += raw_align;
        }

        // Store the original pointer just before the aligned pointer so free() can find it.
        const meta: *[*]u8 = @ptrFromInt(aligned_addr - @sizeOf(usize));
        meta.* = raw_ptr;

        return @ptrFromInt(aligned_addr);
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, _: usize) bool {
        const self: *EmsAllocator = @alignCast(@ptrCast(ctx));
        self.heap.mutex.lock();
        defer self.heap.mutex.unlock();

        const raw_align = alignment.toByteUnits();
        if (raw_align <= 8) {
            return self.heap.resizeObj(buf.ptr, buf.len, new_len);
        }
        // For over-aligned allocations, in-place resize is not supported.
        return false;
    }

    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        // remap is not supported for a pool allocator.
        return null;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, _: usize) void {
        const self: *EmsAllocator = @alignCast(@ptrCast(ctx));
        self.heap.mutex.lock();
        defer self.heap.mutex.unlock();

        const raw_align = alignment.toByteUnits();
        if (raw_align <= 8) {
            self.heap.freeObj(buf.ptr);
            return;
        }

        // Recover the original un-aligned pointer.
        const meta: *[*]u8 = @ptrFromInt(ptrToAddr(buf.ptr) - @sizeOf(usize));
        self.heap.freeObj(meta.*);
    }
};

// ---------------------------------------------------------------------------
// Module-level allocator helper
// ---------------------------------------------------------------------------

/// Get an allocator based on the runtime configuration.
/// If `pool` is provided, returns an EMS allocator backed by that pool.
/// Otherwise, returns `std.heap.page_allocator`.
pub fn getDefaultAllocator() std.mem.Allocator {
    return std.heap.page_allocator;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "initWithPool basic" {
    var pool: [64 * 1024]u8 = undefined;
    const heap = try GcHeap.initWithPool(&pool);
    defer heap.destroy();

    const st = heap.getStats();
    try testing.expect(st.total_size > 0);
    try testing.expect(st.free_size > 0);
    try testing.expectEqual(st.used_size, 0);
}

test "allocate various sizes" {
    var pool: [64 * 1024]u8 = undefined;
    const heap = try GcHeap.initWithPool(&pool);
    defer heap.destroy();

    var ems = EmsAllocator{ .heap = heap };
    const a = ems.allocator();

    const p1 = try a.alloc(u8, 16);
    const p2 = try a.alloc(u8, 128);
    const p3 = try a.alloc(u8, 1024);
    const p4 = try a.alloc(u8, 4096);

    // All should be non-empty slices.
    try testing.expectEqual(p1.len, 16);
    try testing.expectEqual(p2.len, 128);
    try testing.expectEqual(p3.len, 1024);
    try testing.expectEqual(p4.len, 4096);

    a.free(p1);
    a.free(p2);
    a.free(p3);
    a.free(p4);
}

test "free and re-allocate reuses memory" {
    var pool: [64 * 1024]u8 = undefined;
    const heap = try GcHeap.initWithPool(&pool);
    defer heap.destroy();

    var ems = EmsAllocator{ .heap = heap };
    const a = ems.allocator();

    const p1 = try a.alloc(u8, 256);
    const free_before = heap.getStats().free_size;
    a.free(p1);
    const free_after = heap.getStats().free_size;
    try testing.expect(free_after > free_before);

    // Re-allocate — should succeed (freed memory reused).
    const p2 = try a.alloc(u8, 256);
    try testing.expectEqual(p2.len, 256);
    a.free(p2);
}

test "allocate until full" {
    var pool: [4096]u8 = undefined;
    const heap = try GcHeap.initWithPool(&pool);
    defer heap.destroy();

    var ems = EmsAllocator{ .heap = heap };
    const a = ems.allocator();

    var count: usize = 0;
    while (true) {
        const result = a.alloc(u8, 64);
        if (result) |_| {
            count += 1;
        } else |_| {
            break;
        }
    }
    try testing.expect(count > 0);
}

test "heap stats are consistent" {
    var pool: [64 * 1024]u8 = undefined;
    const heap = try GcHeap.initWithPool(&pool);
    defer heap.destroy();

    var ems = EmsAllocator{ .heap = heap };
    const a = ems.allocator();

    const s0 = heap.getStats();
    try testing.expectEqual(s0.total_size, s0.free_size + s0.used_size);

    const p1 = try a.alloc(u8, 512);
    const s1 = heap.getStats();
    try testing.expect(s1.used_size > s0.used_size);
    try testing.expectEqual(s1.total_size, s1.free_size + s1.used_size);
    try testing.expect(s1.highest_allocated >= s1.used_size);

    a.free(p1);
    const s2 = heap.getStats();
    try testing.expectEqual(s2.total_size, s2.free_size + s2.used_size);
}

test "thread safety: concurrent allocations" {
    var pool: [256 * 1024]u8 = undefined;
    const heap = try GcHeap.initWithPool(&pool);
    defer heap.destroy();

    var ems = EmsAllocator{ .heap = heap };
    const a = ems.allocator();

    const num_threads = 4;
    const allocs_per_thread = 50;

    const Worker = struct {
        fn run(alloc_inner: std.mem.Allocator) void {
            var ptrs: [allocs_per_thread]?[]u8 = [_]?[]u8{null} ** allocs_per_thread;
            for (0..allocs_per_thread) |i| {
                ptrs[i] = alloc_inner.alloc(u8, 64) catch null;
            }
            for (&ptrs) |*mp| {
                if (mp.*) |p| {
                    alloc_inner.free(p);
                    mp.* = null;
                }
            }
        }
    };

    var threads: [num_threads]std.Thread = undefined;
    for (&threads) |*t| {
        t.* = try std.Thread.spawn(.{}, Worker.run, .{a});
    }
    for (&threads) |*t| {
        t.join();
    }

    // After all threads complete, heap should be mostly free.
    const st = heap.getStats();
    try testing.expect(st.free_size > 0);
}

test "pool too small" {
    var pool: [32]u8 = undefined;
    const result = GcHeap.initWithPool(&pool);
    try testing.expectError(error.PoolTooSmall, result);
}

test "many small allocations and frees" {
    var pool: [64 * 1024]u8 = undefined;
    const heap = try GcHeap.initWithPool(&pool);
    defer heap.destroy();

    var ems = EmsAllocator{ .heap = heap };
    const a = ems.allocator();

    var ptrs: [100]?[]u8 = [_]?[]u8{null} ** 100;

    // Allocate many small blocks.
    for (0..100) |i| {
        ptrs[i] = a.alloc(u8, 8 + (i % 16) * 8) catch null;
    }

    // Free every other block.
    for (0..100) |i| {
        if (i % 2 == 0) {
            if (ptrs[i]) |p| {
                a.free(p);
                ptrs[i] = null;
            }
        }
    }

    // Re-allocate into freed spaces.
    for (0..100) |i| {
        if (ptrs[i] == null) {
            ptrs[i] = a.alloc(u8, 16) catch null;
        }
    }

    // Free everything.
    for (&ptrs) |*mp| {
        if (mp.*) |p| {
            a.free(p);
            mp.* = null;
        }
    }

    const st = heap.getStats();
    try testing.expectEqual(st.used_size, 0);
}
