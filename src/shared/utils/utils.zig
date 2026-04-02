//! Top-level re-export module for shared utilities.
//!
//! Aggregates every utility sub-module so that downstream code can simply
//! `@import("shared/utils/utils.zig")` and reach everything through a
//! single namespace.

pub const log = @import("log.zig");
pub const leb128 = @import("leb128.zig");
pub const HashMap = @import("hashmap.zig").HashMap;
pub const StringHashMap = @import("hashmap.zig").StringHashMap;
pub const read_file = @import("read_file.zig");

test {
    // Pull in all child tests so that `zig test utils.zig` runs them.
    @import("std").testing.refAllDecls(@This());
}
