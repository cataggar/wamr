//! `wasi_cli` — guest-side helper for writing `wasi:cli/run@0.2.6`
//! command components in pure Zig.
//!
//! Like the sibling `wasi_http`, this hand-writes the canonical-ABI
//! `extern` declarations for the host imports and exposes a small typed
//! Zig API. The shared ret-area lives in the `abi` module, so a guest
//! may combine this with other `wasi_*` wrappers without duplicate
//! `cabi_realloc` exports.
//!
//! A command writes to stdout through `wasi:cli/stdout.get-stdout`
//! (which yields a `wasi:io/streams.output-stream`) and the stream's
//! `blocking-write-and-flush`. The host (wamr's `populateWasiCliRun`,
//! and `wasmtime run`) flushes that to the process's real stdout.
//!
//! ## Usage
//!
//! ```zig
//! const cli = @import("wasi_cli");
//!
//! comptime {
//!     cli.exportRun(run);
//! }
//!
//! fn run() u8 {
//!     cli.print("hello from zig component\n");
//!     return 0; // process exit code
//! }
//! ```
//!
//! `exportRun` takes your entry point as a `comptime` function value and
//! emits the canonical `wasi:cli/run@0.2.6#run` export, so the verbose
//! export name lives here instead of in every guest. The entry point
//! returns a `u8` exit code, which the wrapper reports to the host via
//! `wasi:cli/exit.exit-with-code` — so every run produces an explicit
//! return code (0 = success).

const abi = @import("abi");

const retPtr = abi.retPtr;

// ── Host imports (canonical-ABI lowered signatures) ────────────────

/// `wasi:cli/stdout.get-stdout() -> own<output-stream>`. The result is a
/// single core value (the stream handle), returned directly.
extern "wasi:cli/stdout@0.2.6" fn @"get-stdout"() i32;

/// `wasi:io/streams.[method]output-stream.blocking-write-and-flush(
///   borrow<output-stream>, list<u8>) -> result<_, stream-error>`.
/// `list<u8>` lowers to (ptr, len); the result is 2 flat words →
/// retptr [disc, stream_err_disc]. The helper ignores the result.
extern "wasi:io/streams@0.2.6" fn @"[method]output-stream.blocking-write-and-flush"(
    self: i32,
    contents_ptr: i32,
    contents_len: i32,
    retptr: i32,
) void;

/// `wasi:cli/exit.exit-with-code(status-code: u8)`. `u8` lowers to a
/// single i32 flat param; the function does not return (it terminates
/// the instance, analogous to a trap). `@unstable(feature =
/// cli-exit-with-code)`: wasmtime requires `-S cli-exit-with-code`,
/// while wamr registers it unconditionally.
extern "wasi:cli/exit@0.2.6" fn @"exit-with-code"(status_code: i32) void;

// ── Public API ──────────────────────────────────────────────────────

/// Write `bytes` to stdout (best-effort; the `result<_, stream-error>`
/// is ignored). Opens stdout lazily and caches the handle.
pub fn print(bytes: []const u8) void {
    if (bytes.len == 0) return;
    @"[method]output-stream.blocking-write-and-flush"(
        stdout(),
        @intCast(@intFromPtr(bytes.ptr)),
        @intCast(bytes.len),
        retPtr(),
    );
}

/// Write `bytes` followed by a newline.
pub fn println(bytes: []const u8) void {
    print(bytes);
    print("\n");
}

/// Terminate the instance with the given process exit code via
/// `wasi:cli/exit.exit-with-code`. Does not return.
pub fn exit(code: u8) noreturn {
    @"exit-with-code"(@intCast(code));
    unreachable;
}

// The stdout output-stream handle is opened once and cached. The handle
// is not explicitly dropped: `[resource-drop]` is a canonical built-in
// (not a portable host import), so it is left for the runtime to reclaim
// at instance teardown — mirroring the other `wasi_*` wrappers.
var cached_stdout: ?i32 = null;

fn stdout() i32 {
    if (cached_stdout) |h| return h;
    const h = @"get-stdout"();
    cached_stdout = h;
    return h;
}

// ── comptime export wiring ─────────────────────────────────────────

/// Emit the canonical `wasi:cli/run@0.2.6#run` export that dispatches to
/// `entry`. Call once at file scope:
///
/// ```zig
/// comptime { cli.exportRun(run); }
/// fn run() u8 { ...; return 0; }
/// ```
///
/// `entry` returns a `u8` process exit code. The wrapper reports it to
/// the host via `wasi:cli/exit.exit-with-code`, which terminates the
/// instance and never returns — so the canonical `run -> result` value
/// (`0` = ok) after it is unreachable but keeps the signature well-typed.
pub fn exportRun(comptime entry: fn () u8) void {
    const Wrapper = struct {
        fn run() callconv(.c) i32 {
            exit(entry());
            return 0; // result::ok (unreachable: exit-with-code terminates)
        }
    };
    @export(&Wrapper.run, .{ .name = "wasi:cli/run@0.2.6#run" });
}
