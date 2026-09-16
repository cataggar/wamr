//! Keep the standalone benchmark's shared config and native ABI inside its module.
pub const main = @import("compiler/bench_codegen.zig").main;
