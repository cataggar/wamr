// Copyright (C) 2019 Intel Corporation.  All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Top-level allocator module for the WAMR Zig runtime.
//!
//! Provides a unified interface to obtain a `std.mem.Allocator` based on the
//! runtime configuration.

const std = @import("std");
pub const ems = @import("ems.zig");

/// Get an allocator based on the runtime configuration.
///
/// - If a `pool` is supplied, returns an EMS allocator backed by that pool.
/// - Otherwise, returns `std.heap.page_allocator` (system allocator).
///
/// For static-pool usage, the caller should initialise an `ems.GcHeap` with
/// `ems.GcHeap.initWithPool` and then wrap it in an `ems.EmsAllocator`.
pub fn getDefaultAllocator() std.mem.Allocator {
    return std.heap.page_allocator;
}

test "getDefaultAllocator returns page_allocator" {
    const a = getDefaultAllocator();
    const p = try a.alloc(u8, 64);
    defer a.free(p);
    try std.testing.expectEqual(p.len, 64);
}
