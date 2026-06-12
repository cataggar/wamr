const cli = @import("wasi_cli");

comptime {
    cli.exportRun(run);
}

fn run() u8 {
    cli.println("hello from zig component");
    return 0;
}
