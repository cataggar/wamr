//! Collect Authenticode signature and timestamp evidence on Windows.
const std = @import("std");
const builtin = @import("builtin");
const archive = @import("release_archive.zig");
const process_control = @import("process_control.zig");
const Allocator = std.mem.Allocator;

pub const Signature = struct {
    name: []const u8,
    sha256: []const u8,
    status: []const u8,
    signer: []const u8,
    timestamp_signer: []const u8,
};

const script =
    \\$ErrorActionPreference = 'Stop'
    \\$result = @()
    \\foreach ($tool in @('wamr.exe', 'wamrc.exe')) {
    \\  $path = Join-Path $env:RELEASE_SIGNATURE_BIN $tool
    \\  $sig = Get-AuthenticodeSignature -LiteralPath $path
    \\  if ($sig.Status -ne 'Valid' -or $null -eq $sig.TimeStamperCertificate) {
    \\    throw "Invalid or untimestamped signature: $tool ($($sig.Status))"
    \\  }
    \\  $result += @{
    \\    name = $tool
    \\    sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
    \\    status = $sig.Status.ToString()
    \\    signer = $sig.SignerCertificate.Thumbprint
    \\    timestamp_signer = $sig.TimeStamperCertificate.Thumbprint
    \\  }
    \\}
    \\ConvertTo-Json -InputObject $result -Depth 4 -Compress
;

fn parse(allocator: Allocator, json: []const u8, expected: [2][]const u8) !std.json.Parsed([]Signature) {
    var parsed = try std.json.parseFromSlice([]Signature, allocator, json, .{ .allocate = .alloc_always });
    errdefer parsed.deinit();
    if (parsed.value.len != 2) return error.InvalidWindowsSignatures;
    var seen: [2]bool = @splat(false);
    for (parsed.value) |signature| {
        const index: usize = if (std.mem.eql(u8, signature.name, "wamr.exe"))
            0
        else if (std.mem.eql(u8, signature.name, "wamrc.exe"))
            1
        else
            return error.InvalidWindowsSignatures;
        if (seen[index] or !std.mem.eql(u8, signature.status, "Valid") or
            !std.mem.eql(u8, signature.sha256, expected[index]) or
            signature.signer.len == 0 or signature.timestamp_signer.len == 0)
            return error.InvalidWindowsSignatures;
        seen[index] = true;
    }
    return parsed;
}

/// Inspector must contain the same validated binaries that were extracted.
pub fn collect(allocator: Allocator, io: std.Io, environment: *const std.process.Environ.Map, bin_dir: []const u8, inspector: *const archive.Inspector) !std.json.Parsed([]Signature) {
    if (builtin.os.tag != .windows or inspector.target.system != .windows)
        return error.WindowsSignaturesRequireWindows;
    try inspector.finish();
    var env = try environment.clone(allocator);
    defer env.deinit();
    try env.put("RELEASE_SIGNATURE_BIN", bin_dir);
    const result = try process_control.runTimed(allocator, io, .{
        .argv = &.{ "powershell", "-NoProfile", "-NonInteractive", "-Command", script },
        .environ_map = &env,
    }, 120);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("[release] signature verification failed ({f}): {s}\n", .{ result.term, result.stderr });
        return error.SignatureVerificationFailed;
    }
    var digests: [2][64]u8 = undefined;
    for (0..2) |index| {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(inspector.files[index + 2].?.data, &digest, .{});
        digests[index] = std.fmt.bytesToHex(digest, .lower);
    }
    return parse(allocator, result.stdout, .{ &digests[0], &digests[1] });
}

test "signature evidence binds names, hashes, and timestamp signers" {
    const allocator = std.testing.allocator;
    const good =
        \\[{"name":"wamr.exe","sha256":"one","status":"Valid","signer":"S","timestamp_signer":"T"},
        \\{"name":"wamrc.exe","sha256":"two","status":"Valid","signer":"S","timestamp_signer":"T"}]
    ;
    var parsed = try parse(allocator, good, .{ "one", "two" });
    defer parsed.deinit();
    try std.testing.expectEqualStrings("wamr.exe", parsed.value[0].name);
    try std.testing.expectError(error.InvalidWindowsSignatures, parse(allocator, good, .{ "wrong", "two" }));
    const duplicate =
        \\[{"name":"wamr.exe","sha256":"one","status":"Valid","signer":"S","timestamp_signer":"T"},
        \\{"name":"wamr.exe","sha256":"one","status":"Valid","signer":"S","timestamp_signer":"T"}]
    ;
    try std.testing.expectError(error.InvalidWindowsSignatures, parse(allocator, duplicate, .{ "one", "two" }));
    const unsigned =
        \\[{"name":"wamr.exe","sha256":"one","status":"Valid","signer":"S","timestamp_signer":""},
        \\{"name":"wamrc.exe","sha256":"two","status":"Valid","signer":"S","timestamp_signer":"T"}]
    ;
    try std.testing.expectError(error.InvalidWindowsSignatures, parse(allocator, unsigned, .{ "one", "two" }));
}
