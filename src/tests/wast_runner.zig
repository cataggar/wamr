//! WAMR-native .wast spec test runner.
//!
//! Parses .wast S-expressions using wabt's text parser, converts modules
//! to .wasm binary, then loads and executes with WAMR's interpreter.
//!
//! Currently delegates to wabt's wast_runner which uses wabt's own
//! interpreter. This validates conformance against the same 65K+ test
//! suite that wabt passes. A future version will replace wabt's
//! interpreter calls with WAMR's for full end-to-end WAMR validation.

const std = @import("std");
const wabt = @import("wabt");

pub const WastResult = struct {
    passed: u32 = 0,
    failed: u32 = 0,
    skipped: u32 = 0,

    pub fn total(self: WastResult) u32 {
        return self.passed + self.failed + self.skipped;
    }
};

/// Run a .wast file. Uses wabt's parser and interpreter for conformance.
pub fn run(allocator: std.mem.Allocator, source: []const u8) WastResult {
    const wabt_result = wabt.wast_runner.run(allocator, source);
    return .{
        .passed = wabt_result.passed,
        .failed = wabt_result.failed,
        .skipped = wabt_result.skipped,
    };
}
