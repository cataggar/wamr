//! Own each standalone IR suite under src/, without importing the wamr module.
comptime {
    // Discovery must also run when --test-filter excludes any wrapper tests.
    _ = switch (@import("compiler_ir_suite").suite) {
        .passes => @import("compiler/ir/passes.zig"),
        .analysis => @import("compiler/ir/analysis.zig"),
        .local_init => @import("compiler/ir/local_init.zig"),
        .regalloc => @import("compiler/ir/regalloc.zig"),
        .range_split => @import("compiler/ir/range_split.zig"),
        .print_test => @import("compiler/ir/print_test.zig"),
        .verifier => @import("compiler/ir/verifier.zig"),
        .interp => @import("compiler/ir/interp.zig"),
        .fuzz => @import("compiler/ir/fuzz.zig"),
        .property_test => @import("compiler/ir/property_test.zig"),
        .forward_redundant_loads_dominator => @import("compiler/ir/forward_redundant_loads_dominator.zig"),
    };
}
