const std = @import("std");
const http = @import("wasi_http");

comptime {
    http.exportIncomingHandler(handle);
}

fn handle(req: http.Request, res: *http.Responder) void {
    const path = req.path() orelse "/";
    if (std.mem.eql(u8, path, "/")) {
        res.respond(200, "Hello, world!\n");
    } else {
        res.respond(404, "");
    }
}
