//! Zig WASI HTTP component — mirrors the bytecodealliance
//! [Rust HTTP-in-components tutorial][rust-http]:
//!
//!   GET /       -> 200 "Hello, world!\n"
//!   anything    -> 404
//!
//! Runs end-to-end through wamr:
//!
//!   wamr run --listen=127.0.0.1:8080 zig-http.component.wasm
//!   curl -i http://127.0.0.1:8080/
//!
//! The canonical-ABI plumbing (host imports, ret-area decoding, the
//! `cabi_realloc` arena, and the `wasi:http/incoming-handler` export
//! wiring) lives in the shared `wit_http` helper module (imported as
//! `@import("wit_http")`; source at `src/guest/wit_http.zig`). This
//! example is just the routing logic; see that module's doc comment for
//! how the canonical ABI is bridged and why a guest still imports only
//! the host functions it actually calls.
//!
//! [rust-http]: https://component-model.bytecodealliance.org/language-support/using-http-in-components/rust.html
//!
//! See `../README.md` for the WIT layout and build pipeline.

const std = @import("std");
const wit = @import("wit_http");

comptime {
    wit.exportIncomingHandler(handle);
}

fn handle(req: wit.Request, res: *wit.Responder) void {
    const path = req.path() orelse "/";
    if (std.mem.eql(u8, path, "/")) {
        res.respond(200, "Hello, world!\n");
    } else {
        res.respond(404, "");
    }
}
