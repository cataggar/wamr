//! Windows readiness and cancellable stdin I/O for WASI Preview-1.

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

    extern "kernel32" fn ReadFile(
        hFile: Handle,
        lpBuffer: *anyopaque,
        nNumberOfBytesToRead: windows.DWORD,
        lpNumberOfBytesRead: ?*windows.DWORD,
        lpOverlapped: ?*anyopaque,
    ) callconv(.winapi) windows.BOOL;

    extern "kernel32" fn CreatePipe(
        hReadPipe: *Handle,
        hWritePipe: *Handle,
        lpPipeAttributes: ?*windows.SECURITY_ATTRIBUTES,
        nSize: windows.DWORD,
    ) callconv(.winapi) windows.BOOL;

    extern "kernel32" fn WriteFile(
        hFile: Handle,
        lpBuffer: *const anyopaque,
        nNumberOfBytesToWrite: windows.DWORD,
        lpNumberOfBytesWritten: ?*windows.DWORD,
        lpOverlapped: ?*anyopaque,
    ) callconv(.winapi) windows.BOOL;

    extern "kernel32" fn DuplicateHandle(
        hSourceProcessHandle: Handle,
        hSourceHandle: Handle,
        hTargetProcessHandle: Handle,
        lpTargetHandle: *Handle,
        dwDesiredAccess: windows.DWORD,
        bInheritHandle: windows.BOOL,
        dwOptions: windows.DWORD,
    ) callconv(.winapi) windows.BOOL;
} else struct {};

const wait_object_0: windows.DWORD = 0;
const wait_timeout: windows.DWORD = 258;
const wait_failed: windows.DWORD = std.math.maxInt(windows.DWORD);
const infinite: windows.DWORD = std.math.maxInt(windows.DWORD);
const max_wait_handles = 64;
const polling_slice_ms: windows.DWORD = 10;
const worker_stop_budget_ms: windows.DWORD = 100;
const duplicate_same_access: windows.DWORD = 0x0000_0002;
const max_pipe_workers = max_wait_handles - 1;
var pipe_worker_count = std.atomic.Value(usize).init(0);

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

pub const CancelEventError = error{SystemResources};

/// Lazily-created manual-reset event owned by a WASI thread manager.
pub const CancelEvent = if (is_windows) struct {
    handle: ?Handle = null,

    pub fn init() @This() {
        return .{};
    }

    pub fn ensureInitialized(self: *@This(), fail_for_test: bool) CancelEventError!void {
        if (self.handle != null) return;
        if (fail_for_test) return error.SystemResources;
        self.handle = api.CreateEventW(null, .TRUE, .FALSE, null) orelse
            return error.SystemResources;
    }

    pub fn deinit(self: *@This()) void {
        if (self.handle) |handle| windows.CloseHandle(handle);
        self.handle = null;
    }

    pub fn signal(self: *@This()) bool {
        const handle = self.handle orelse return true;
        return api.SetEvent(handle).toBool();
    }

    pub fn opaqueHandle(self: *const @This()) ?*anyopaque {
        return self.handle;
    }
} else struct {
    pub fn init() @This() {
        return .{};
    }

    pub fn ensureInitialized(self: *@This(), fail_for_test: bool) CancelEventError!void {
        _ = self;
        _ = fail_for_test;
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }

    pub fn signal(self: *@This()) bool {
        _ = self;
        return true;
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
    pipe: bool = false,
    /// The handle is currently signalled by low-level events, but a blocking
    /// cooked read would not complete. Poll it in bounded slices rather than
    /// waiting directly on the perpetually-signalled console handle.
    poll_only: bool = false,
};

pub const WaitResult = enum {
    ready,
    timed_out,
    cancelled,
    failed,
};

const ConsoleStatus = struct {
    status: InputStatus,
    poll_only: bool,
};

fn inspectConsoleRecords(mode: windows.DWORD, records: []const InputRecord) ConsoleStatus {
    const line_mode = (mode & enable_line_input) != 0;
    for (records) |record| {
        if (record.EventType != key_event) continue;
        const key = record.Event.KeyEvent;
        if (!key.bKeyDown.toBool()) continue;
        const char = key.uChar.UnicodeChar;
        if (char == 0) continue;
        if (!line_mode or char == '\r' or char == '\n')
            return .{ .status = .ready, .poll_only = false };
    }
    return .{
        .status = .pending,
        .poll_only = records.len != 0,
    };
}

fn consoleReady(
    allocator: std.mem.Allocator,
    handle: Handle,
    mode: windows.DWORD,
) ConsoleStatus {
    var event_count: windows.DWORD = 0;
    if (!api.GetNumberOfConsoleInputEvents(handle, &event_count).toBool())
        return .{ .status = .io_error, .poll_only = false };
    if (event_count == 0)
        return .{ .status = .pending, .poll_only = false };

    const records = allocator.alloc(InputRecord, event_count) catch
        return .{ .status = .io_error, .poll_only = false };
    defer allocator.free(records);

    var records_read: windows.DWORD = 0;
    if (!api.PeekConsoleInputW(handle, records.ptr, event_count, &records_read).toBool())
        return .{ .status = .io_error, .poll_only = false };
    return inspectConsoleRecords(mode, records[0..records_read]);
}

const PipeProbeResult = union(enum) {
    pending,
    ready: u64,
    hangup,
    bad_handle,
    io_error,
};

fn cancelSynchronousWorker(thread: std.Thread) void {
    const thread_handle: Handle = thread.getHandle();
    var wait_handles = [_]Handle{thread_handle};
    while (true) {
        var iosb: windows.IO_STATUS_BLOCK = undefined;
        _ = windows.ntdll.NtCancelSynchronousIoFile(thread_handle, null, &iosb);
        _ = windows.ntdll.NtAlertThread(thread_handle);
        const result = api.WaitForMultipleObjects(1, &wait_handles, .FALSE, polling_slice_ms);
        if (result == wait_object_0 or result == wait_failed) break;
    }
    thread.join();
}

fn probePipe(handle: Handle) PipeProbeResult {
    var available: windows.DWORD = 0;
    if (api.PeekNamedPipe(handle, null, 0, null, &available, null).toBool())
        return if (available == 0) .pending else .{ .ready = available };
    return switch (windows.GetLastError()) {
        .BROKEN_PIPE, .PIPE_NOT_CONNECTED, .HANDLE_EOF => .hangup,
        .NO_DATA => .pending,
        .INVALID_HANDLE => .bad_handle,
        .OPERATION_ABORTED => .pending,
        else => .io_error,
    };
}

fn probeInput(allocator: std.mem.Allocator, input: *ReadInput) void {
    input.status = .pending;
    input.nbytes = 0;
    input.console = false;
    input.poll_only = false;

    if (input.pipe) {
        input.poll_only = true;
        return;
    }

    if (input.handle == windows.INVALID_HANDLE_VALUE) {
        input.status = .bad_handle;
        return;
    }

    var console_mode: windows.DWORD = 0;
    if (api.GetConsoleMode(input.handle, &console_mode).toBool()) {
        input.console = true;
        const result = consoleReady(allocator, input.handle, console_mode);
        input.status = result.status;
        input.poll_only = result.poll_only;
        if (input.status == .ready) input.nbytes = 1;
        return;
    }

    switch (api.GetFileType(input.handle)) {
        file_type_disk, file_type_char => input.status = .ready,
        file_type_pipe => {
            input.pipe = true;
            input.poll_only = true;
        },
        file_type_unknown => input.status = .bad_handle,
        else => input.status = .ready,
    }
}

const PipeProbeWorker = struct {
    original_handle: Handle,
    owned_handle: Handle,
    status: InputStatus = .pending,
    nbytes: u64 = 0,
    stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    refs: std.atomic.Value(usize) = std.atomic.Value(usize).init(1),

    fn reserveSlot() bool {
        var current = pipe_worker_count.load(.monotonic);
        while (current < max_pipe_workers) {
            current = pipe_worker_count.cmpxchgWeak(
                current,
                current + 1,
                .acq_rel,
                .monotonic,
            ) orelse return true;
        }
        return false;
    }

    fn create(handle: Handle) !*PipeProbeWorker {
        if (!reserveSlot()) return error.SystemResources;
        errdefer _ = pipe_worker_count.fetchSub(1, .acq_rel);

        var owned_handle: Handle = undefined;
        const process = windows.GetCurrentProcess();
        if (!api.DuplicateHandle(
            process,
            handle,
            process,
            &owned_handle,
            0,
            .FALSE,
            duplicate_same_access,
        ).toBool()) return error.SystemResources;
        errdefer windows.CloseHandle(owned_handle);

        const self = try std.heap.page_allocator.create(PipeProbeWorker);
        self.* = .{
            .original_handle = handle,
            .owned_handle = owned_handle,
        };
        return self;
    }

    fn retain(self: *PipeProbeWorker) void {
        _ = self.refs.fetchAdd(1, .monotonic);
    }

    fn release(self: *PipeProbeWorker) void {
        if (self.refs.fetchSub(1, .acq_rel) != 1) return;
        windows.CloseHandle(self.owned_handle);
        _ = pipe_worker_count.fetchSub(1, .acq_rel);
        std.heap.page_allocator.destroy(self);
    }

    fn run(self: *PipeProbeWorker) void {
        defer self.release();
        while (!self.stop.load(.acquire)) {
            self.status = .pending;
            self.nbytes = 0;
            switch (probePipe(self.owned_handle)) {
                .pending => {},
                .ready => |nbytes| {
                    self.status = .ready;
                    self.nbytes = nbytes;
                    return;
                },
                .hangup => {
                    self.status = .hangup;
                    return;
                },
                .bad_handle => {
                    self.status = .bad_handle;
                    return;
                },
                .io_error => {
                    self.status = .io_error;
                    return;
                },
            }
            if (self.stop.load(.acquire)) return;
            api.Sleep(polling_slice_ms);
        }
    }

    fn copyResults(self: *const PipeProbeWorker, inputs: []ReadInput) void {
        for (inputs) |*input| {
            if (!input.pipe or input.handle != self.original_handle) continue;
            input.status = self.status;
            input.nbytes = self.nbytes;
            input.poll_only = self.status == .pending;
        }
    }
};

const PipeWorkerEntry = struct {
    worker: *PipeProbeWorker,
    thread: ?std.Thread,
};

fn classifyPipeInputs(inputs: []ReadInput) bool {
    var has_pipe = false;
    for (inputs) |*input| {
        input.pipe = false;
        if (input.handle == windows.INVALID_HANDLE_VALUE) continue;
        var console_mode: windows.DWORD = 0;
        if (api.GetConsoleMode(input.handle, &console_mode).toBool()) continue;
        if (api.GetFileType(input.handle) == file_type_pipe) {
            input.pipe = true;
            has_pipe = true;
        }
    }
    return has_pipe;
}

fn pipeWorkerFinished(thread: std.Thread) bool {
    var handles = [_]Handle{thread.getHandle()};
    return api.WaitForMultipleObjects(1, &handles, .FALSE, 0) == wait_object_0;
}

fn joinPipeWorker(entry: *PipeWorkerEntry, inputs: []ReadInput) void {
    const running = entry.thread orelse return;
    running.join();
    entry.thread = null;
    entry.worker.copyResults(inputs);
}

fn signalPipeWorkerStop(entry: *PipeWorkerEntry) void {
    const running = entry.thread orelse return;
    entry.worker.stop.store(true, .release);
    var iosb: windows.IO_STATUS_BLOCK = undefined;
    _ = windows.ntdll.NtCancelSynchronousIoFile(running.getHandle(), null, &iosb);
    _ = windows.ntdll.NtAlertThread(running.getHandle());
}

fn stopPipeWorkers(entries: []PipeWorkerEntry) void {
    for (entries) |*entry| signalPipeWorkerStop(entry);
    const started = api.GetTickCount64();
    while (true) {
        var active = false;
        for (entries) |*entry| {
            const running = entry.thread orelse continue;
            active = true;
            if (pipeWorkerFinished(running)) {
                running.join();
                entry.thread = null;
            } else {
                signalPipeWorkerStop(entry);
            }
        }
        if (!active) return;
        if (api.GetTickCount64() - started >= worker_stop_budget_ms) break;
        api.Sleep(polling_slice_ms);
    }
    for (entries) |*entry| {
        const running = entry.thread orelse continue;
        running.detach();
        entry.thread = null;
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

const WaitHooks = struct {
    ctx: *anyopaque,
    now_ms: *const fn (*anyopaque) u64,
    sleep: *const fn (*anyopaque, windows.DWORD) void,
    wait: *const fn (*anyopaque, []const Handle, windows.DWORD) windows.DWORD,
    probe: *const fn (*anyopaque, *ReadInput) void,
};

fn currentMs(hooks: ?WaitHooks) u64 {
    if (hooks) |installed| return installed.now_ms(installed.ctx);
    return api.GetTickCount64();
}

fn sleepMs(hooks: ?WaitHooks, timeout_ms: windows.DWORD) void {
    if (hooks) |installed| {
        installed.sleep(installed.ctx, timeout_ms);
        return;
    }
    api.Sleep(timeout_ms);
}

fn waitHandles(
    hooks: ?WaitHooks,
    handles: []const Handle,
    timeout_ms: windows.DWORD,
) windows.DWORD {
    if (hooks) |installed| return installed.wait(installed.ctx, handles, timeout_ms);
    return api.WaitForMultipleObjects(
        @intCast(handles.len),
        handles.ptr,
        .FALSE,
        timeout_ms,
    );
}

fn remainingTimeout(start_ms: u64, now_ms: u64, timeout_ms: i32) ?windows.DWORD {
    if (timeout_ms < 0) return infinite;
    const elapsed = now_ms -% start_ms;
    const total: u64 = @intCast(timeout_ms);
    if (elapsed >= total) return null;
    return @intCast(total - elapsed);
}

/// Wait until one input is readable/closed, the timeout expires, or the
/// manager cancellation event is signalled.
pub fn waitForReadiness(
    allocator: std.mem.Allocator,
    cancel_handle_opaque: ?*anyopaque,
    inputs: []ReadInput,
    timeout_ms: i32,
) WaitResult {
    return waitForReadinessWithHooks(
        allocator,
        cancel_handle_opaque,
        inputs,
        timeout_ms,
        null,
    );
}

fn waitForReadinessWithHooks(
    allocator: std.mem.Allocator,
    cancel_handle_opaque: ?*anyopaque,
    inputs: []ReadInput,
    timeout_ms: i32,
    hooks: ?WaitHooks,
) WaitResult {
    if (!is_windows) unreachable;

    const cancel_handle: ?Handle = if (cancel_handle_opaque) |handle|
        @ptrCast(handle)
    else
        null;
    const start_ms = currentMs(hooks);

    var pipe_workers: std.ArrayListUnmanaged(PipeWorkerEntry) = .empty;
    defer {
        stopPipeWorkers(pipe_workers.items);
        for (pipe_workers.items) |entry| entry.worker.release();
        pipe_workers.deinit(allocator);
    }
    if (hooks == null and classifyPipeInputs(inputs)) {
        var unique_count: usize = 0;
        for (inputs, 0..) |input, input_index| {
            if (!input.pipe) continue;
            var duplicate = false;
            for (inputs[0..input_index]) |previous| {
                if (previous.pipe and previous.handle == input.handle) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) unique_count += 1;
        }
        if (unique_count > max_pipe_workers) return .failed;
        pipe_workers.ensureTotalCapacity(allocator, unique_count) catch return .failed;

        for (inputs, 0..) |input, input_index| {
            if (!input.pipe) continue;
            var duplicate = false;
            for (inputs[0..input_index]) |previous| {
                if (previous.pipe and previous.handle == input.handle) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;

            const worker = PipeProbeWorker.create(input.handle) catch return .failed;
            worker.retain();
            const thread = std.Thread.spawn(.{}, PipeProbeWorker.run, .{worker}) catch {
                worker.release();
                worker.release();
                return .failed;
            };
            pipe_workers.appendAssumeCapacity(.{ .worker = worker, .thread = thread });
        }
    }

    while (true) {
        for (pipe_workers.items) |*entry| {
            if (entry.thread) |thread| {
                if (pipeWorkerFinished(thread)) joinPipeWorker(entry, inputs);
            }
        }

        if (cancel_handle) |cancel| {
            var cancel_only = [_]Handle{cancel};
            const cancel_result = waitHandles(hooks, &cancel_only, 0);
            if (cancel_result == wait_object_0) return .cancelled;
            if (cancel_result == wait_failed) return .failed;
        }

        var handles: [max_wait_handles]Handle = undefined;
        var handle_count: usize = 0;
        var needs_bounded_poll = false;
        var overflow = false;
        var worker_indices: [max_wait_handles]?usize = @splat(null);

        if (cancel_handle) |handle| {
            handles[0] = handle;
            handle_count = 1;
        }
        for (pipe_workers.items, 0..) |entry, entry_index| {
            const thread = entry.thread orelse continue;
            if (handle_count == handles.len) break;
            worker_indices[handle_count] = entry_index;
            handles[handle_count] = thread.getHandle();
            handle_count += 1;
        }

        var any_ready = false;
        for (inputs) |*input| {
            if (hooks) |installed| {
                installed.probe(installed.ctx, input);
            } else if (input.pipe) {
                var active = false;
                for (pipe_workers.items) |entry| {
                    if (entry.worker.original_handle == input.handle and
                        entry.thread != null)
                    {
                        active = true;
                        break;
                    }
                }
                if (active) continue;
            } else {
                probeInput(allocator, input);
            }
            switch (input.status) {
                .ready, .hangup, .bad_handle, .io_error => any_ready = true,
                .pending => {
                    if (input.console and !input.poll_only) {
                        if (!appendUniqueHandle(&handles, &handle_count, input.handle))
                            overflow = true;
                    } else {
                        needs_bounded_poll = true;
                    }
                },
            }
        }
        if (any_ready or timeout_ms == 0) {
            const grace_deadline = currentMs(hooks) + polling_slice_ms;
            while (true) {
                var grace_handles: [max_wait_handles]Handle = undefined;
                var grace_workers: [max_wait_handles]?usize = @splat(null);
                var grace_count: usize = 0;
                if (cancel_handle) |cancel| {
                    grace_handles[0] = cancel;
                    grace_count = 1;
                }
                for (pipe_workers.items, 0..) |entry, entry_index| {
                    const thread = entry.thread orelse continue;
                    grace_workers[grace_count] = entry_index;
                    grace_handles[grace_count] = thread.getHandle();
                    grace_count += 1;
                }
                const non_worker_handles: usize =
                    if (cancel_handle == null) 0 else 1;
                if (grace_count == non_worker_handles) break;
                const now = currentMs(hooks);
                if (now >= grace_deadline) break;
                const grace_result = waitHandles(
                    hooks,
                    grace_handles[0..grace_count],
                    @intCast(grace_deadline - now),
                );
                if (grace_result == wait_timeout) break;
                if (grace_result == wait_failed) return .failed;
                const signaled_index: usize = @intCast(grace_result - wait_object_0);
                if (cancel_handle != null and signaled_index == 0)
                    return .cancelled;
                const entry_index = grace_workers[signaled_index] orelse
                    return .failed;
                joinPipeWorker(&pipe_workers.items[entry_index], inputs);
            }
            for (inputs) |input| {
                if (!input.pipe) continue;
                switch (input.status) {
                    .ready, .hangup, .bad_handle, .io_error => any_ready = true,
                    .pending => {},
                }
            }
            return if (any_ready) .ready else .timed_out;
        }

        var wait_ms = remainingTimeout(start_ms, currentMs(hooks), timeout_ms) orelse
            return .timed_out;
        if (needs_bounded_poll or overflow)
            wait_ms = @min(wait_ms, polling_slice_ms);

        if (handle_count == 0) {
            if (wait_ms == infinite) return .failed;
            sleepMs(hooks, wait_ms);
            continue;
        }

        const result = waitHandles(hooks, handles[0..handle_count], wait_ms);
        if (result == wait_failed) return .failed;
        if (result == wait_timeout) continue;
        if (result >= wait_object_0 + handle_count) return .failed;
        const index: usize = @intCast(result - wait_object_0);
        if (cancel_handle != null and index == 0) return .cancelled;
        if (worker_indices[index]) |entry_index|
            joinPipeWorker(&pipe_workers.items[entry_index], inputs);
    }
}

pub const ReadError = error{
    Cancelled,
    BadHandle,
    IoError,
    ThreadSpawnFailed,
    WaitFailed,
    WouldBlock,
};

const ReadOutcome = enum {
    pending,
    success,
    eof,
    would_block,
    cancelled,
    bad_handle,
    io_error,
};

const ReadJob = struct {
    handle: Handle,
    buffer: []u8,
    outcome: ReadOutcome = .pending,
    nread: usize = 0,

    fn run(self: *ReadJob) void {
        var nread: windows.DWORD = 0;
        if (api.ReadFile(
            self.handle,
            self.buffer.ptr,
            @intCast(self.buffer.len),
            &nread,
            null,
        ).toBool()) {
            self.outcome = .success;
            self.nread = nread;
            return;
        }
        const err = windows.GetLastError();
        if (err == .MORE_DATA and nread != 0) {
            self.outcome = .success;
            self.nread = nread;
            return;
        }
        self.outcome = switch (err) {
            .BROKEN_PIPE, .PIPE_NOT_CONNECTED, .HANDLE_EOF => .eof,
            .NO_DATA => .would_block,
            .OPERATION_ABORTED => .cancelled,
            .INVALID_HANDLE => .bad_handle,
            else => .io_error,
        };
    }
};

fn finishRead(thread: std.Thread, job: *const ReadJob) ReadError!usize {
    thread.join();
    return switch (job.outcome) {
        .success => job.nread,
        .eof => 0,
        .would_block => error.WouldBlock,
        .cancelled => error.Cancelled,
        .bad_handle => error.BadHandle,
        .pending, .io_error => error.IoError,
    };
}

/// Perform the actual synchronous `ReadFile` on a helper thread, then wait on
/// both that thread and the manager cancellation event. On cancellation the
/// blocked syscall itself is stopped with `NtCancelSynchronousIoFile`.
pub fn readCancellable(
    allocator: std.mem.Allocator,
    cancel_handle_opaque: *anyopaque,
    handle: Handle,
    buffer: []u8,
) ReadError!usize {
    if (!is_windows) unreachable;
    if (buffer.len == 0) return 0;
    if (buffer.len > std.math.maxInt(windows.DWORD)) return error.IoError;

    const cancel_handle: Handle = @ptrCast(cancel_handle_opaque);
    const is_pipe = api.GetFileType(handle) == file_type_pipe;

    while (true) {
        if (is_pipe) {
            var inputs = [_]ReadInput{.{ .handle = handle }};
            switch (waitForReadiness(
                allocator,
                cancel_handle_opaque,
                &inputs,
                -1,
            )) {
                .cancelled => return error.Cancelled,
                .failed, .timed_out => return error.WaitFailed,
                .ready => switch (inputs[0].status) {
                    .ready => {},
                    .hangup => return 0,
                    .bad_handle => return error.BadHandle,
                    .io_error => return error.IoError,
                    .pending => continue,
                },
            }
        }

        var job = ReadJob{ .handle = handle, .buffer = buffer };
        var thread = std.Thread.spawn(.{}, ReadJob.run, .{&job}) catch
            return error.ThreadSpawnFailed;
        const thread_handle: Handle = thread.getHandle();
        var handles = [_]Handle{ cancel_handle, thread_handle };

        const result = api.WaitForMultipleObjects(handles.len, &handles, .FALSE, infinite);
        if (result == wait_object_0 + 1) {
            const nread = finishRead(thread, &job) catch |err| switch (err) {
                error.WouldBlock => continue,
                else => return err,
            };
            return nread;
        }
        if (result == wait_object_0) {
            cancelSynchronousWorker(thread);
            return error.Cancelled;
        }

        cancelSynchronousWorker(thread);
        return error.WaitFailed;
    }
}

fn testKeyRecord(char: windows.WCHAR) InputRecord {
    return .{
        .EventType = key_event,
        .padding = 0,
        .Event = .{ .KeyEvent = .{
            .bKeyDown = .TRUE,
            .wRepeatCount = 1,
            .wVirtualKeyCode = 0,
            .wVirtualScanCode = 0,
            .uChar = .{ .UnicodeChar = char },
            .dwControlKeyState = 0,
        } },
    };
}

test "Windows poll review: cooked console requires Enter and scans the complete queue" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;

    var ctrl_z = [_]InputRecord{testKeyRecord(0x1a)};
    const incomplete = inspectConsoleRecords(enable_line_input, &ctrl_z);
    try std.testing.expectEqual(InputStatus.pending, incomplete.status);
    try std.testing.expect(incomplete.poll_only);

    var records: [300]InputRecord = undefined;
    for (&records) |*record| record.* = testKeyRecord('x');
    records[299] = testKeyRecord('\r');
    const complete = inspectConsoleRecords(enable_line_input, &records);
    try std.testing.expectEqual(InputStatus.ready, complete.status);
    try std.testing.expect(!complete.poll_only);
}

test "Windows poll review: non-key records do not spin and raw Unicode is ready" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;

    var non_key = [_]InputRecord{.{
        .EventType = 0x0002,
        .padding = 0,
        .Event = .{ .padding = @splat(0) },
    }};
    const ignored = inspectConsoleRecords(enable_line_input, &non_key);
    try std.testing.expectEqual(InputStatus.pending, ignored.status);
    try std.testing.expect(ignored.poll_only);

    var unicode = [_]InputRecord{testKeyRecord(0x03bb)};
    const raw = inspectConsoleRecords(0, &unicode);
    try std.testing.expectEqual(InputStatus.ready, raw.status);
}

const FakeWait = struct {
    now_ms: u64 = 0,
    probe_count: usize = 0,
    sleep_count: usize = 0,
    wait_count: usize = 0,
    max_wait_handles: usize = 0,
    max_wait_ms: windows.DWORD = 0,
    poll_only: bool = true,
    ready_handle: ?Handle = null,
    ready_after_probe: usize = std.math.maxInt(usize),

    fn now(raw: *anyopaque) u64 {
        const self: *FakeWait = @ptrCast(@alignCast(raw));
        return self.now_ms;
    }

    fn sleep(raw: *anyopaque, timeout_ms: windows.DWORD) void {
        const self: *FakeWait = @ptrCast(@alignCast(raw));
        self.sleep_count += 1;
        self.max_wait_ms = @max(self.max_wait_ms, timeout_ms);
        self.now_ms += timeout_ms;
    }

    fn wait(
        raw: *anyopaque,
        handles: []const Handle,
        timeout_ms: windows.DWORD,
    ) windows.DWORD {
        const self: *FakeWait = @ptrCast(@alignCast(raw));
        self.wait_count += 1;
        self.max_wait_handles = @max(self.max_wait_handles, handles.len);
        self.max_wait_ms = @max(self.max_wait_ms, timeout_ms);
        self.now_ms += timeout_ms;
        return wait_timeout;
    }

    fn probe(raw: *anyopaque, input: *ReadInput) void {
        const self: *FakeWait = @ptrCast(@alignCast(raw));
        self.probe_count += 1;
        input.status = if (self.ready_handle == input.handle and
            self.probe_count >= self.ready_after_probe)
            .ready
        else
            .pending;
        input.console = true;
        input.poll_only = self.poll_only;
        input.nbytes = if (input.status == .ready) 1 else 0;
    }

    fn hooks(self: *FakeWait) WaitHooks {
        return .{
            .ctx = @ptrCast(self),
            .now_ms = now,
            .sleep = sleep,
            .wait = wait,
            .probe = probe,
        };
    }
};

test "Windows poll review: incomplete cooked input is probed at a bounded rate" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    var fake = FakeWait{};
    var inputs = [_]ReadInput{.{ .handle = @ptrFromInt(1) }};
    try std.testing.expectEqual(
        WaitResult.timed_out,
        waitForReadinessWithHooks(
            std.testing.allocator,
            null,
            &inputs,
            35,
            fake.hooks(),
        ),
    );
    try std.testing.expectEqual(@as(usize, 5), fake.probe_count);
    try std.testing.expectEqual(@as(usize, 4), fake.sleep_count);
    try std.testing.expectEqual(@as(windows.DWORD, polling_slice_ms), fake.max_wait_ms);
}

test "Windows poll review: more than 64 console handles stay bounded and complete" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    var inputs: [70]ReadInput = undefined;
    for (&inputs, 0..) |*input, index| {
        input.* = .{ .handle = @ptrFromInt(index + 1) };
    }
    var fake = FakeWait{
        .poll_only = false,
        .ready_handle = inputs[69].handle,
        .ready_after_probe = inputs.len + 1,
    };
    try std.testing.expectEqual(
        WaitResult.ready,
        waitForReadinessWithHooks(
            std.testing.allocator,
            null,
            &inputs,
            100,
            fake.hooks(),
        ),
    );
    try std.testing.expectEqual(@as(usize, max_wait_handles), fake.max_wait_handles);
    try std.testing.expectEqual(@as(windows.DWORD, polling_slice_ms), fake.max_wait_ms);
    try std.testing.expectEqual(InputStatus.ready, inputs[69].status);
}

test "Windows poll rereview: blocked pipe metadata worker respects the caller timeout" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;

    var read = windows.INVALID_HANDLE_VALUE;
    var write = windows.INVALID_HANDLE_VALUE;
    if (!api.CreatePipe(&read, &write, null, 0).toBool())
        return error.TestUnexpectedResult;
    defer if (read != windows.INVALID_HANDLE_VALUE) windows.CloseHandle(read);
    defer windows.CloseHandle(write);

    const Reader = struct {
        fn run(handle: Handle) void {
            var byte: [1]u8 = undefined;
            var nread: windows.DWORD = 0;
            _ = api.ReadFile(handle, &byte, 1, &nread, null);
        }
    };
    const FallbackWriter = struct {
        fn run(handle: Handle) void {
            api.Sleep(1000);
            var written: windows.DWORD = 0;
            const byte = "x";
            _ = api.WriteFile(handle, byte.ptr, 1, &written, null);
        }
    };
    const reader = try std.Thread.spawn(.{}, Reader.run, .{read});
    const fallback = try std.Thread.spawn(.{}, FallbackWriter.run, .{write});
    api.Sleep(20);

    const started = api.GetTickCount64();
    var inputs = [_]ReadInput{.{ .handle = read }};
    const result = waitForReadiness(
        std.heap.page_allocator,
        null,
        &inputs,
        50,
    );
    const elapsed = api.GetTickCount64() - started;
    fallback.join();
    reader.join();

    try std.testing.expectEqual(WaitResult.timed_out, result);
    try std.testing.expect(elapsed < 250);
}

test "Windows poll rereview: idle pipe wait uses one persistent worker" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;

    var read = windows.INVALID_HANDLE_VALUE;
    var write = windows.INVALID_HANDLE_VALUE;
    if (!api.CreatePipe(&read, &write, null, 0).toBool())
        return error.TestUnexpectedResult;
    defer windows.CloseHandle(read);
    defer windows.CloseHandle(write);

    var inputs = [_]ReadInput{.{ .handle = read }};
    const started = api.GetTickCount64();
    try std.testing.expectEqual(
        WaitResult.timed_out,
        waitForReadiness(std.testing.allocator, null, &inputs, 50),
    );
    const elapsed = api.GetTickCount64() - started;
    try std.testing.expect(elapsed < 250);
}

test "Windows poll review 983: blocked pipe cannot hide a ready peer" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;

    var blocked_read = windows.INVALID_HANDLE_VALUE;
    var blocked_write = windows.INVALID_HANDLE_VALUE;
    if (!api.CreatePipe(&blocked_read, &blocked_write, null, 0).toBool())
        return error.TestUnexpectedResult;
    defer windows.CloseHandle(blocked_read);
    defer windows.CloseHandle(blocked_write);

    var ready_read = windows.INVALID_HANDLE_VALUE;
    var ready_write = windows.INVALID_HANDLE_VALUE;
    if (!api.CreatePipe(&ready_read, &ready_write, null, 0).toBool())
        return error.TestUnexpectedResult;
    defer windows.CloseHandle(ready_read);
    defer windows.CloseHandle(ready_write);

    const Reader = struct {
        fn run(handle: Handle) void {
            var byte: [1]u8 = undefined;
            var nread: windows.DWORD = 0;
            _ = api.ReadFile(handle, &byte, 1, &nread, null);
        }
    };
    const FallbackWriter = struct {
        fn run(handle: Handle) void {
            api.Sleep(1000);
            var written: windows.DWORD = 0;
            const byte = "x";
            _ = api.WriteFile(handle, byte.ptr, 1, &written, null);
        }
    };
    const blocker = try std.Thread.spawn(.{}, Reader.run, .{blocked_read});
    const fallback = try std.Thread.spawn(.{}, FallbackWriter.run, .{blocked_write});
    var written: windows.DWORD = 0;
    const ready_byte = "y";
    if (!api.WriteFile(ready_write, ready_byte.ptr, 1, &written, null).toBool())
        return error.TestUnexpectedResult;
    api.Sleep(20);

    const baseline_workers = pipe_worker_count.load(.acquire);
    const started = api.GetTickCount64();
    var inputs = [_]ReadInput{
        .{ .handle = blocked_read },
        .{ .handle = ready_read },
    };
    const result = waitForReadiness(
        std.heap.page_allocator,
        null,
        &inputs,
        0,
    );
    const elapsed = api.GetTickCount64() - started;
    fallback.join();
    blocker.join();

    const cleanup_deadline = api.GetTickCount64() + 500;
    while (pipe_worker_count.load(.acquire) != baseline_workers and
        api.GetTickCount64() < cleanup_deadline)
    {
        api.Sleep(10);
    }

    try std.testing.expectEqual(WaitResult.ready, result);
    try std.testing.expectEqual(InputStatus.ready, inputs[1].status);
    try std.testing.expectEqual(@as(u64, 1), inputs[1].nbytes);
    try std.testing.expect(elapsed < 250);
    try std.testing.expectEqual(baseline_workers, pipe_worker_count.load(.acquire));
}

test "Windows poll review 983: worker quota is bounded and handles are owned" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    const cleanup_deadline = api.GetTickCount64() + 500;
    while (pipe_worker_count.load(.acquire) != 0 and
        api.GetTickCount64() < cleanup_deadline)
    {
        api.Sleep(10);
    }
    try std.testing.expectEqual(@as(usize, 0), pipe_worker_count.load(.acquire));

    var read = windows.INVALID_HANDLE_VALUE;
    var write = windows.INVALID_HANDLE_VALUE;
    if (!api.CreatePipe(&read, &write, null, 0).toBool())
        return error.TestUnexpectedResult;
    defer windows.CloseHandle(read);
    defer windows.CloseHandle(write);

    var workers: [max_pipe_workers]*PipeProbeWorker = undefined;
    var count: usize = 0;
    defer for (workers[0..count]) |worker| worker.release();
    while (count < workers.len) : (count += 1)
        workers[count] = try PipeProbeWorker.create(read);
    try std.testing.expectError(error.SystemResources, PipeProbeWorker.create(read));
    try std.testing.expectEqual(max_pipe_workers, pipe_worker_count.load(.acquire));

    windows.CloseHandle(read);
    read = windows.INVALID_HANDLE_VALUE;
    var written: windows.DWORD = 0;
    const byte = "z";
    try std.testing.expect(api.WriteFile(write, byte.ptr, 1, &written, null).toBool());
    switch (probePipe(workers[0].owned_handle)) {
        .ready => |nbytes| try std.testing.expectEqual(@as(u64, 1), nbytes),
        else => return error.TestUnexpectedResult,
    }
}
