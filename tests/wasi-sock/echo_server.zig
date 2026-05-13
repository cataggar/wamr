//! WASI preview1 echo server fixture for the `--listen` integration test.
//!
//! Compiled to `wasm32-wasi`. Assumes the embedder has installed a TCP
//! listening socket as fd 3 (the only preopen in this layout). Accepts a
//! single connection, echoes a single recv buffer back to the peer, then
//! `proc_exit(0)`.
//!
//! Calls the preview1 host functions directly via `@extern`, bypassing
//! wasi-libc. This keeps the fixture small and lets it exercise the new
//! `sock_accept` / `sock_recv` / `sock_send` host functions without
//! pulling in wasi-libc's socket emulation.

const Iovec = extern struct { buf: [*]u8, len: u32 };
const ConstIovec = extern struct { buf: [*]const u8, len: u32 };

extern "wasi_snapshot_preview1" fn sock_accept(fd: i32, fdflags: i32, ro_fd: *i32) i32;
extern "wasi_snapshot_preview1" fn sock_recv(
    fd: i32,
    ri_data: *const Iovec,
    ri_data_len: u32,
    ri_flags: i32,
    ro_datalen: *u32,
    ro_flags: *u32,
) i32;
extern "wasi_snapshot_preview1" fn sock_send(
    fd: i32,
    si_data: *const ConstIovec,
    si_data_len: u32,
    si_flags: i32,
    so_datalen: *u32,
) i32;
extern "wasi_snapshot_preview1" fn fd_close(fd: i32) i32;
extern "wasi_snapshot_preview1" fn proc_exit(code: u32) noreturn;

const LISTEN_FD: i32 = 3;

export fn _start() void {
    var client_fd: i32 = 0;
    if (sock_accept(LISTEN_FD, 0, &client_fd) != 0) proc_exit(11);

    var buf: [256]u8 = undefined;
    const iov = Iovec{ .buf = &buf, .len = buf.len };
    var datalen: u32 = 0;
    var roflags: u32 = 0;
    if (sock_recv(client_fd, &iov, 1, 0, &datalen, &roflags) != 0) proc_exit(12);
    if (datalen == 0) proc_exit(13);

    const civ = ConstIovec{ .buf = &buf, .len = datalen };
    var sent: u32 = 0;
    if (sock_send(client_fd, &civ, 1, 0, &sent) != 0) proc_exit(14);
    if (sent != datalen) proc_exit(15);

    _ = fd_close(client_fd);
    proc_exit(0);
}
