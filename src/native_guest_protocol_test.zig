//! Hosted correctness-only test transport, not a deployed producer.
pub const main = @import("bench/native_guest_tests.zig").protocolFixture;
