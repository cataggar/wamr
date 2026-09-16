//! Link audit only; not a boot entry point and never executed.
const guest = @import("wamr-aot").benchmark;

export fn wamr_native_guest_run(options: *const guest.Options) u32 {
    guest.run(options.*) catch return 1;
    return 0;
}

export fn wamr_native_guest_session(session: *guest.runner.Session, writer: *guest.Writer) u32 {
    const reset = session.resetTimed();
    if (reset.outcome != .completed) return 1;
    const call = session.invoke("_start") catch return 2;
    guest.runner.writeInvocationEvidence(writer, "steady", 1, call) catch return 3;
    return 0;
}

export fn wamr_native_guest_identity(bytes: [*]const u8, length: usize, storage: *[64]u8) u64 {
    return guest.Identity.fromBytes(bytes[0..length], storage).bytes;
}
