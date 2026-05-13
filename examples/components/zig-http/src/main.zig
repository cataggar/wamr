//! Zig WASI HTTP component — placeholder.
//!
//! Full source authored under todo `write-zig-source`, which is
//! blocked on cataggar/wabt#191. Once that lands and the wamr wabt
//! pin bumps, this file will hold the canonical-ABI-mangled
//! `wasi:http/incoming-handler@0.2.6#handle` export plus the
//! `wasi:http/types@0.2.6` / `wasi:io/streams@0.2.6` imports the
//! `/` → 200 "Hello, world!\n" / else → 404 handler needs. See
//! `../../../README.md` for the example shape and the canonical-ABI
//! lowering notes.

export fn @"wasi:http/incoming-handler@0.2.6#handle"(_: i32, _: i32) void {
    // TODO(write-zig-source): build outgoing-response, write
    // "Hello, world!\n" via outgoing-body / output-stream, deliver
    // through response-outparam.set. Blocked on cataggar/wabt#191.
}
