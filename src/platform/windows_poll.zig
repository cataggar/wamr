//! Windows readiness support for WASI Preview-1 polling.
//!
//! Console handles are waitable, but anonymous/named pipe handles do not
//! report byte readiness through `WaitForMultipleObjects`. Pipes are therefore
//! probed with `PeekNamedPipe` between short waits on the thread group's
//! manual-reset cancellation event. Disk/character files are always readable
//! in the same sense as POSIX regular files.

const std = @import("std");
const builtin = @import("builtin");

const is_windows = builtin.os.tag == .windows;
const windows = std.os.windows;

pub const Handle = windows.HANDLE;

const api = if (is_windows) struct {
    extern "kernel32" fn CreateEventW(
        lpEventAttributes: ?*windows.SECURITY_ATTRIBUTES,
        bManualReset: windows.BOOL,
        bInitialState: windows.BOOL,
        lpName: ?windows.LPCWSTR,
    ) callconv(.winapi) ?Handle;

    extern "kernel32" fn SetEvent(hEvent: Handle) callconv(.winapi) windows.BOOL;

    extern "kernel32" fn WaitForMultipleObjects(
        nCount: windows.DWORD,
        lpHandles: [*]const Handle,
        bWaitAll: windows.BOOL,
        dwMilliseconds: windows.DWORD,
    ) callconv(.winapi) windows.DWORD;

    extern "kernel32" fn Sleep(dwMilliseconds: windows.DWORD) callconv(.winapi) void;

    extern "kernel32" fn GetTickCount64() callconv(.winapi) u64;

    extern "kernel32" fn GetConsoleMode(
        hConsoleHandle: Handle,
        lpMode: *windows.DWORD,
    ) callconv(.winapi) windows.BOOL;

    extern "kernel32" fn GetNumberOfConsoleInputEvents(
        hConsoleInput: Handle,
        lpcNumberOfEvents: *windows.DWORD,
    ) callconv(.winapi) windows.BOOL;

    extern "kernel32" fn PeekConsoleInputW(
        hConsoleInput: Handle,
        lpBuffer: [*]InputRecord,
        nLength: windows.DWORD,
        lpNumberOfEventsRead: *windows.DWORD,
    ) callconv(.winapi) windows.BOOL;

    extern "kernel32" fn PeekNamedPipe(
        hNamedPipe: Handle,
        lpBuffer: ?*anyopaque,
        nBufferSize: windows.DWORD,
        lpBytesRead: ?*windows.DWORD,
        lpTotalBytesAvail: ?*windows.DWORD,
        lpBytesLeftThisMessage: ?*windows.DWORD,
    ) callconv(.winapi) windows.BOOL;

    extern "kernel32" fn GetFileType(hFile: Handle) callconv(.winapi) windows.DWORD;
} else struct {};

const wait_object_0: windows.DWORD = 0;
const wait_timeout: windows.DWORD = 258;
const wait_failed: windows.DWORD = std.math.maxInt(windows.DWORD);
const infinite: windows.DWORD = std.math.maxInt(windows.DWORD);
const max_wait_handles = 64;
const pipe_probe_slice_ms: windows.DWORD = 10;

const file_type_unknown: windows.DWORD = 0;
const file_type_disk: windows.DWORD = 1;
const file_type_char: windows.DWORD = 2;
const file_type_pipe: windows.DWORD = 3;

const key_event: windows.WORD = 0x0001;
const enable_line_input: windows.DWORD = 0x0002;

const CharUnion = extern union {
    UnicodeChar: windows.WCHAR,
    AsciiChar: windows.CHAR,
};

const KeyEventRecord = extern struct {
    bKeyDown: windows.BOOL,
    wRepeatCount: windows.WORD,
    wVirtualKeyCode: windows.WORD,
    wVirtualScanCode: windows.WORD,
    uChar: CharUnion,
    dwControlKeyState: windows.DWORD,
};

const InputEvent = extern union {
    KeyEvent: KeyEventRecord,
    padding: [16]u8,
};

const InputRecord = extern struct {
    EventType: windows.WORD,
    padding: windows.WORD,
    Event: InputEvent,
};

comptime {
    std.debug.assert(@sizeOf(KeyEventRecord) == 16);
    std.debug.assert(@sizeOf(InputRecord) == 20);
}

/// Manual-reset event owned by a WASI thread manager. Once group termination
/// starts it remains signalled for the rest of that manager's lifetime.
pub const CancelEvent = if (is_windows) struct {
    handle: Handle,

    pub fn init() @This() {
        const handle = api.CreateEventW(null, .TRUE, .FALSE, null) orelse
            @panic("failed to create WASI thread cancellation event");
        return .{ .handle = handle };
    }

    pub fn deinit(self: *@This()) void {
        windows.CloseHandle(self.handle);
    }

    pub fn signal(self: *@This()) void {
        _ = api.SetEvent(self.handle);
    }

    pub fn opaqueHandle(self: *const @This()) ?*anyopaque {
        return self.handle;
    }
} else struct {
    pub fn init() @This() {
        return .{};
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }

    pub fn signal(self: *@This()) void {
        _ = self;
    }

    pub fn opaqueHandle(self: *const @This()) ?*anyopaque {
        _ = self;
        return null;
    }
};

pub const InputStatus = enum {
    pending,
    ready,
    hangup,
    bad_handle,
    io_error,
};

pub const ReadInput = struct {
    handle: Handle,
    status: InputStatus = .pending,
    nbytes: u64 = 0,
    console: bool = false,
};

pub const WaitResult = enum {
    ready,
    timed_out,
    cancelled,
    failed,
};

fn consoleReady(handle: Handle, mode: windows.DWORD) InputStatus {
    var event_count: windows.DWORD = 0;
    if (!api.GetNumberOfConsoleInputEvents(handle, &event_count).toBool())
        return .io_error;
    if (event_count == 0) return .pending;

    // Keep canonical console reads from being reported ready before Enter is
    // present. This covers long pasted lines without consuming console events.
    var records: [256]InputRecord = undefined;
    var records_read: windows.DWORD = 0;
    const count: windows.DWORD = @min(event_count, records.len);
    if (!api.PeekConsoleInputW(handle, &records, count, &records_read).toBool())
        return .io_error;

    const line_mode = (mode & enable_line_input) != 0;
    for (records[0..records_read]) |record| {
        if (record.EventType != key_event) continue;
        const key = record.Event.KeyEvent;
        if (!key.bKeyDown.toBool()) continue;
        const char = key.uChar.UnicodeChar;
        if (char == 0) continue;
        if (!line_mode or char == '\r' or char == '\n' or char == 0x1a)
            return .ready;
    }
    return .pending;
}

fn probeInput(input: *ReadInput) void {
    input.status = .pending;
    input.nbytes = 0;
    input.console = false;

    if (input.handle == windows.INVALID_HANDLE_VALUE) {
        input.status = .bad_handle;
        return;
    }

    var console_mode: windows.DWORD = 0;
    if (api.GetConsoleMode(input.handle, &console_mode).toBool()) {
        input.console = true;
        input.status = consoleReady(input.handle, console_mode);
        if (input.status == .ready) input.nbytes = 1;
        return;
    }

    var available: windows.DWORD = 0;
    if (api.PeekNamedPipe(input.handle, null, 0, null, &available, null).toBool()) {
        if (available != 0) {
            input.status = .ready;
            input.nbytes = available;
        }
        return;
    }

    switch (windows.GetLastError()) {
        .BROKEN_PIPE, .PIPE_NOT_CONNECTED, .NO_DATA => {
            input.status = .hangup;
            return;
        },
        .INVALID_HANDLE => {
            input.status = .bad_handle;
            return;
        },
        else => {},
    }

    switch (api.GetFileType(input.handle)) {
        file_type_disk, file_type_char => input.status = .ready,
        file_type_pipe => input.status = .io_error,
        file_type_unknown => input.status = .bad_handle,
        else => input.status = .ready,
    }
}

fn appendUniqueHandle(handles: *[max_wait_handles]Handle, count: *usize, handle: Handle) bool {
    for (handles[0..count.*]) |existing| {
        if (existing == handle) return true;
    }
    if (count.* == handles.len) return false;
    handles[count.*] = handle;
    count.* += 1;
    return true;
}

fn remainingTimeout(start_ms: u64, timeout_ms: i32) ?windows.DWORD {
    if (timeout_ms < 0) return infinite;
    const elapsed = api.GetTickCount64() -% start_ms;
    const total: u64 = @intCast(timeout_ms);
    if (elapsed >= total) return null;
    return @intCast(total - elapsed);
}

/// Wait until one input is readable/closed, the timeout expires, or the
/// manager cancellation event is signalled. The cancellation handle occupies
/// slot zero whenever present. Pipe readiness and wait sets larger than the
/// Win32 64-handle limit fall back to bounded cancellation-event waits.
pub fn waitForReadiness(
    cancel_handle_opaque: ?*anyopaque,
    inputs: []ReadInput,
    timeout_ms: i32,
) WaitResult {
    if (!is_windows) unreachable;

    const cancel_handle: ?Handle = if (cancel_handle_opaque) |handle|
        @ptrCast(handle)
    else
        null;
    const start_ms = api.GetTickCount64();

    while (true) {
        var handles: [max_wait_handles]Handle = undefined;
        var handle_count: usize = 0;
        var has_polled_input = false;
        var overflow = false;

        if (cancel_handle) |handle| {
            handles[0] = handle;
            handle_count = 1;
        }

        var any_ready = false;
        for (inputs) |*input| {
            probeInput(input);
            switch (input.status) {
                .ready, .hangup, .bad_handle, .io_error => any_ready = true,
                .pending => {
                    if (input.console) {
                        if (!appendUniqueHandle(&handles, &handle_count, input.handle))
                            overflow = true;
                    } else {
                        has_polled_input = true;
                    }
                },
            }
        }
        if (any_ready) return .ready;
        if (timeout_ms == 0) return .timed_out;

        var wait_ms = remainingTimeout(start_ms, timeout_ms) orelse
            return .timed_out;
        if (has_polled_input or overflow)
            wait_ms = @min(wait_ms, pipe_probe_slice_ms);

        if (handle_count == 0) {
            api.Sleep(wait_ms);
            if (wait_ms == infinite) return .failed;
            continue;
        }

        const result = api.WaitForMultipleObjects(
            @intCast(handle_count),
            &handles,
            .FALSE,
            wait_ms,
        );
        if (result == wait_failed) return .failed;
        if (result == wait_timeout) continue;
        if (result < wait_object_0 or result >= wait_object_0 + handle_count)
            return .failed;
        const index: usize = @intCast(result - wait_object_0);
        if (cancel_handle != null and index == 0) return .cancelled;
        // A console handle fired. Re-probe to filter non-key events and, in
        // canonical line mode, incomplete input that would still block ReadFile.
    }
}
